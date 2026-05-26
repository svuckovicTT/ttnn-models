# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Standalone xla.py reproducing tt-xla benchmark test_llms.py::test_llama_3_1_8b_instruct_tp.
### Resolved model: meta-llama/Llama-3.1-8B-Instruct (HF) via AutoModelForCausalLM.
### Original benchmark: n300-llmbox (1x8 wormhole). xla.py retargeted to QB2 (1x4 blackhole).
### No imports from tt-xla/third_party/tt_forge_models or tt-xla/tests/benchmark/*.

import os
from pathlib import Path

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache
from tt_torch import codegen_py
from tt_torch.sharding import sharding_constraint_hook

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATA_FORMAT = torch.bfloat16     # test_llms.py: DEFAULT_DATA_FORMAT = "bfloat16"
BATCH_SIZE = 32                  # test_llms.py: DEFAULT_BATCH_SIZE = 32
INPUT_SEQUENCE_LENGTH = 128      # test_llms.py: DEFAULT_INPUT_SEQUENCE_LENGTH = 128
DEFAULT_INPUT_PROMPT = (
    "Here is an exaustive list of the best practices for writing clean code:"
)

OUTPUT_DIR = str(Path(__file__).resolve().parent / "model")

# Per-test overrides applied:
#   optimization_level=2  (DEFAULT_TP_OPTIMIZATION_LEVEL from test_llms.py, passed via test_llm_tp)
# Benchmark-file defaults applied (test_llms.py):
#   experimental_weight_dtype="bfp_bf8"  (DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE)
#   experimental_enable_permute_matmul_fusion=False  (DEFAULT_EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION)
COMPILE_OPTIONS = {
    "optimization_level": 2,
    # "enable_trace": True,  # benchmark used trace_enabled=True (DEFAULT_TRACE_ENABLED)
    "experimental_weight_dtype": "bfp_bf8",
    "experimental_enable_permute_matmul_fusion": False,
}


def load_pytorch_model():
    # Mirrors third_party/tt_forge_models/llama/causal_lm/pytorch/loader.py:load_model
    # (HUGGING_FACE branch): from_pretrained with torch_dtype passed at load time.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DATA_FORMAT)

    # setup_model_and_tokenizer patches from benchmarks/llm_benchmark.py:73-78.
    if hasattr(model.config, "layer_types"):
        model.config.layer_types = ["full_attention"] * len(model.config.layer_types)
    if hasattr(model.config, "_experts_implementation"):
        model.config._experts_implementation = "dense"

    model.eval()
    return model


def load_input():
    """Replicates construct_inputs() from benchmarks/llm_benchmark.py:85-169."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompts = [DEFAULT_INPUT_PROMPT] * BATCH_SIZE
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=INPUT_SEQUENCE_LENGTH,
        truncation=True,
    )
    input_ids = tokenized["input_ids"]

    # StaticCache lives on CPU and is moved to the device explicitly later — see
    # https://github.com/tenstorrent/tt-xla/issues/1645 for why we don't construct
    # it directly on the device.
    model = load_pytorch_model()
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=BATCH_SIZE,
        max_cache_len=INPUT_SEQUENCE_LENGTH,
        device="cpu",
        dtype=DATA_FORMAT,
    )

    cache_position = torch.arange(0, input_ids.shape[1])

    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "cache_position": cache_position,
        "use_cache": True,
    }


def _inputs_to_device(inputs, device):
    """Mirrors transfer_to_device() from benchmarks/llm_benchmark.py:179-205 for the
    StaticCache path (no MLA layers in Llama 3.1)."""
    out = dict(inputs)
    out["input_ids"] = out["input_ids"].to(device)
    out["cache_position"] = out["cache_position"].to(device)
    for layer in out["past_key_values"].layers:
        # StaticLayer uses lazy initialization — keys/values may be None until the
        # first forward call materializes them. Skip those layers; they'll be
        # initialized on-device on first use.
        if layer.keys is not None:
            layer.keys = layer.keys.to(device)
        if layer.values is not None:
            layer.values = layer.values.to(device)
    return out


def _mesh_config(num_devices):
    # Inlined from third_party/tt_forge_models/llama/causal_lm/pytorch/loader.py:get_mesh_config
    # for variant LLAMA_3_1_8B_INSTRUCT. The loader returns (1, num_devices) for all
    # non-70B/405B variants on the default "batch"/"model" mesh.
    # Original benchmark target: n300-llmbox (8 devices) -> (1, 8).
    # Retargeted to QB2 (4 devices) -> (1, 4).
    return (1, num_devices), ("batch", "model")


def _shard_spec(model):
    # Inlined from third_party/tt_forge_models/llama/causal_lm/pytorch/loader.py:load_shard_spec
    # for variant LLAMA_3_1_8B_INSTRUCT (NOT in the no-shard list, so the full FSDP-style
    # shard map applies).
    shard_specs = {}
    shard_specs[model.model.embed_tokens.weight] = (None, "batch")
    shard_specs[model.lm_head.weight] = ("model", "batch")
    shard_specs[model.model.norm.weight] = ("batch",)
    for layer in model.model.layers:
        shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
        shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
        shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

        shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[layer.input_layernorm.weight] = ("batch",)
        shard_specs[layer.post_attention_layernorm.weight] = ("batch",)

    return shard_specs


def _apply_tp_sharding(model):
    """Mirrors the multichip setup block in benchmark_llm_torch_xla
    (benchmarks/llm_benchmark.py:436-447)."""
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    mesh_shape, mesh_name = _mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, mesh_name)

    shard_specs = _shard_spec(model)
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    # Apply sharding constraint on lm_head output to all_gather logits.
    if hasattr(model, "lm_head") and model.lm_head is not None:
        hook = sharding_constraint_hook(model.lm_head, mesh, (None, None, None))
        model.lm_head.register_forward_hook(hook)

    return mesh


def run_pytorch_model():
    model = load_pytorch_model()
    inputs = load_input()

    with torch.no_grad():
        output = model(**inputs)

    return output.logits


def run_tt_model():
    xr.set_device_type("TT")
    device = torch_xla.device()

    torch_xla.set_custom_compile_options(COMPILE_OPTIONS)

    model = load_pytorch_model()
    model.compile(backend="tt", options={"tt_legacy_compile": True})
    model = model.to(device)

    _apply_tp_sharding(model)

    inputs = _inputs_to_device(load_input(), device)

    with torch.no_grad():
        output = model(**inputs)

    return output.logits.cpu()


def codegen_model():
    os.environ["XLA_HLO_DEBUG"] = "1"

    model = load_pytorch_model()
    inputs = load_input()

    codegen_py(
        model,
        inputs,
        export_path=OUTPUT_DIR,
        export_tensors=True,
        compiler_options=COMPILE_OPTIONS,
    )


def compare_pytorch_and_tt_runs():
    # Capture exact PCC from first --golden run and paste here.
    exact_pcc = None

    pt_output = run_pytorch_model()
    tt_output = run_tt_model()

    assert pt_output.shape == tt_output.shape, (
        f"shape mismatch: {pt_output.shape} vs {tt_output.shape}"
    )
    assert pt_output.dtype == tt_output.dtype, (
        f"dtype mismatch: {pt_output.dtype} vs {tt_output.dtype}"
    )
    x, y = pt_output.flatten(), tt_output.flatten()
    vx, vy = x - x.mean(), y - y.mean()
    pcc = ((vx @ vy) / (vx.norm() * vy.norm())).item()
    print(f"PCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="test_llama_3_1_8b_instruct_tp codegen pipeline")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-pt", action="store_true", help="Run PyTorch model on CPU")
    mode.add_argument("--run-tt", action="store_true", help="Run model on TT hardware")
    mode.add_argument("--codegen", action="store_true", help="Generate TTNN code")
    mode.add_argument(
        "--golden", action="store_true", help="Compare PyTorch and TTNN runs"
    )

    args = parser.parse_args()

    if args.run_pt:
        run_pytorch_model()
    if args.run_tt:
        run_tt_model()
    if args.codegen:
        codegen_model()
    if args.golden:
        compare_pytorch_and_tt_runs()
