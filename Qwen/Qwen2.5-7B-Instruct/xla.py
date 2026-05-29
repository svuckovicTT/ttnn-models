# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Standalone xla.py reproducing tt-xla benchmark test_llms.py::test_qwen_2_5_7b.
### Resolved model: Qwen/Qwen2.5-7B-Instruct (HF) via Qwen2ForCausalLM.
### No imports from tt-xla/third_party/tt_forge_models or tt-xla/tests/benchmark/*.

import os
from pathlib import Path

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import AutoTokenizer, Qwen2ForCausalLM
from transformers.cache_utils import StaticCache
from tt_torch import codegen_py

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_FORMAT = torch.bfloat16     # test_llms.py: DEFAULT_DATA_FORMAT = "bfloat16"
BATCH_SIZE = 32                  # test_llms.py: DEFAULT_BATCH_SIZE = 32
INPUT_SEQUENCE_LENGTH = 128      # test_llms.py: DEFAULT_INPUT_SEQUENCE_LENGTH = 128
DEFAULT_INPUT_PROMPT = (
    "Here is an exaustive list of the best practices for writing clean code:"
)

OUTPUT_DIR = str(Path(__file__).resolve().parent / "model")

# Per-test overrides applied:
#   optimization_level=1 (explicit override in test_qwen_2_5_7b; test_llms.DEFAULT_OPTIMIZATION_LEVEL is 2)
#   required_pcc=0.90 (validation knob, not a compile option)
# Benchmark-file overrides applied (test_llms.py module-level DEFAULT_*):
#   experimental_weight_dtype="bfp_bf8"  (DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE)
#   experimental_enable_permute_matmul_fusion=False  (DEFAULT_EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION)
COMPILE_OPTIONS = {
    "optimization_level": 1,
    # "enable_trace": True,  # benchmark used trace_enabled=True (DEFAULT_TRACE_ENABLED)
    "experimental_weight_dtype": "bfp_bf8",
    "experimental_enable_permute_matmul_fusion": False,
}


def load_pytorch_model():
    # Mirrors third_party/tt_forge_models/qwen_2_5/causal_lm/pytorch/loader.py:load_model
    # (HUGGING_FACE branch): Qwen2ForCausalLM.from_pretrained with torch_dtype passed at load time.
    model = Qwen2ForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DATA_FORMAT)

    # setup_model_and_tokenizer patches from benchmarks/llm_benchmark.py:73-79.
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

    # StaticCache lives on CPU and is moved to the device explicitly later - see
    # https://github.com/tenstorrent/tt-xla/issues/1645 for why we don't construct
    # it directly on the device. Mirrors init_static_cache in
    # tests/benchmark/llm_utils/decode_utils.py - need early_initialization so
    # layer key/value tensors exist before transfer_to_device.
    model = load_pytorch_model()
    config = model.config
    if hasattr(config, "head_dim") and getattr(config, "head_dim"):
        head_dim = config.head_dim
    else:
        head_dim = config.hidden_size // config.num_attention_heads
    num_key_value_heads = getattr(
        config, "num_key_value_heads", config.num_attention_heads
    )
    past_key_values = StaticCache(
        config=config,
        max_batch_size=BATCH_SIZE,
        max_cache_len=INPUT_SEQUENCE_LENGTH,
        device="cpu",
        dtype=DATA_FORMAT,
    )
    past_key_values.early_initialization(
        batch_size=BATCH_SIZE,
        num_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=DATA_FORMAT,
        device="cpu",
    )

    cache_position = torch.arange(0, input_ids.shape[1])

    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "cache_position": cache_position,
        "use_cache": True,
    }


def _inputs_to_device(inputs, device):
    """Mirrors transfer_to_device() from benchmarks/llm_benchmark.py for the
    StaticCache path (Qwen 2.5 uses standard attention, not MLA)."""
    out = dict(inputs)
    out["input_ids"] = out["input_ids"].to(device)
    out["cache_position"] = out["cache_position"].to(device)
    for layer in out["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
        # transformers >=5.x tracks per-layer cumulative_length and a layer.device
        # attribute - both must follow the keys/values to the device, else either
        # past_seen_tokens stays on cpu, or modeling_qwen2.forward emits
        # torch.arange(..., device=self.device) on cpu while inputs are on xla,
        # causing "two different devices" errors.
        if hasattr(layer, "cumulative_length") and torch.is_tensor(
            layer.cumulative_length
        ):
            layer.cumulative_length = layer.cumulative_length.to(device)
        if hasattr(layer, "device"):
            layer.device = device
    return out


def run_pytorch_model():
    model = load_pytorch_model()
    inputs = load_input()

    with torch.no_grad():
        output = model(**inputs)

    return output.logits


def run_tt_model():
    device = torch_xla.device()

    torch_xla.set_custom_compile_options(COMPILE_OPTIONS)

    model = load_pytorch_model()
    model.compile(backend="tt", options={"tt_legacy_compile": True})
    model = model.to(device)
    inputs = _inputs_to_device(load_input(), device)

    with torch.no_grad():
        output = model(**inputs)

    return output.logits.cpu()


def codegen_model():
    os.environ["XLA_HLO_DEBUG"] = "1"

    model = load_pytorch_model()
    inputs = load_input()

    # codegen_py only forwards tensor args/kwargs; past_key_values (StaticCache)
    # and use_cache (bool) would be dropped, leaving the model with no input_ids
    # context. Move the cache to the device manually and call the model directly,
    # mirroring what codegen_py does for tensors.
    real_compile_options = {
        **COMPILE_OPTIONS,
        "backend": "codegen_py",
        "export_path": OUTPUT_DIR,
        "export_tensors": True,
    }
    torch_xla.set_custom_compile_options(real_compile_options)

    device = torch_xla.device()
    model.compile(backend="tt", options={"tt_legacy_compile": True})
    model = model.to(device, dtype=DATA_FORMAT)
    inputs = _inputs_to_device(inputs, device)

    with torch.no_grad():
        model(**inputs)

    import torch_xla.core.xla_model as xm

    xm.wait_device_ops()


def compare_pytorch_and_tt_runs():
    # Capture exact PCC from first --golden run and paste here.
    exact_pcc = 0.996094

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

    parser = argparse.ArgumentParser(description="test_qwen_2_5_7b codegen pipeline")

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
