# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Standalone xla.py reproducing tt-xla benchmark test_llms.py::test_llama_3_1_8b_instruct_tp.
### Resolved model: meta-llama/Llama-3.1-8B-Instruct (HF) via AutoModelForCausalLM.
### Original benchmark: n300-llmbox (1x8 wormhole), tensor-parallel. Per user request this is
### retargeted to a SINGLE p150 (blackhole) chip with NO tensor parallelism. The benchmark only
### shards when num_devices > 1 (benchmarks/llm_benchmark.py:340-342), so on one chip it runs
### unsharded — all mesh / mark_sharding / KV-cache-sharding / lm_head-hook machinery is dropped.
### The weight-dtype overrides are kept: the benchmark applies them regardless of chip count.
### No imports from tt-xla/third_party/tt_forge_models or tt-xla/tests/benchmark/*.

import os
from pathlib import Path

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache
from tt_torch.weight_dtype import apply_weight_dtype_overrides

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATA_FORMAT = torch.bfloat16     # test_llms.py: DEFAULT_DATA_FORMAT = "bfloat16"
BATCH_SIZE = 32                  # test_llms.py: DEFAULT_BATCH_SIZE = 32
INPUT_SEQUENCE_LENGTH = 128      # test_llms.py: DEFAULT_INPUT_SEQUENCE_LENGTH = 128
DEFAULT_INPUT_PROMPT = (
    "Here is an exaustive list of the best practices for writing clean code:"
)

OUTPUT_DIR = str(Path(__file__).resolve().parent / "model")

# Per-test override applied (via test_llm_tp -> test_llm):
#   optimization_level=2  (DEFAULT_TP_OPTIMIZATION_LEVEL from test_llms.py)
# Benchmark-file defaults applied (test_llms.py DEFAULT_* constants forwarded to
# benchmark_llm_torch_xla, which builds the options dict at llm_benchmark.py:457-466):
#   experimental_weight_dtype="bfp_bf8"  (DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE)
#   experimental_enable_permute_matmul_fusion=False  (DEFAULT_EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION)
COMPILE_OPTIONS = {
    "optimization_level": 2,
    # "enable_trace": True,  # benchmark used trace_enabled=True (DEFAULT_TRACE_ENABLED)
    "experimental_weight_dtype": "bfp_bf8",
    "experimental_enable_permute_matmul_fusion": False,
}

# Auto-discovered by the benchmark via model_loader.get_weight_dtype_config_path()
# (llm_benchmark.py:478-485 fallback, since the test passes no explicit
# weight_dtype_overrides). Resolves to mixed_precision_configs/Llama-3.1-8B-Instruct.json
# next to the loader; replicated inline here.
WEIGHT_DTYPE_OVERRIDES = {
    "model.layers.*.mlp.gate_proj.weight": "bfp_bf4",
    "model.layers.*.mlp.up_proj.weight": "bfp_bf4",
    "default": "bfp_bf8",
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


class LastTokenLogitsWrapper(torch.nn.Module):
    """Wraps a CausalLM so the last-token logits slice is inside the compiled graph,
    avoiding materializing the full [batch, seq, vocab] tensor on device. Mirrors the
    slicing role of LLMSamplingWrapper in tests/benchmark/llm_utils/decode_utils.py."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        output = self.model(**kwargs)
        return output.logits[:, -1]


def load_input():
    """Replicates construct_inputs() (benchmarks/llm_benchmark.py:85-169) plus
    init_static_cache() (llm_utils/decode_utils.py:113-145). Every returned key and
    every initialization step must match the benchmark exactly."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token  # loader._load_tokenizer sets this
    prompts = [DEFAULT_INPUT_PROMPT] * BATCH_SIZE
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=INPUT_SEQUENCE_LENGTH,
        truncation=True,
    )
    input_ids = tokenized["input_ids"]

    # StaticCache lives on CPU and is moved to device explicitly later — see
    # https://github.com/tenstorrent/tt-xla/issues/1645 for why we don't construct
    # it directly on the device.
    model = load_pytorch_model()
    config = model.config
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
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
    # early_initialization pre-allocates cache tensors on CPU so they are not lazily
    # created inside the compiled graph (which the TTIR->TTNN pipeline can't legalize).
    past_key_values.early_initialization(
        batch_size=BATCH_SIZE,
        num_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=DATA_FORMAT,
        device="cpu",
    )

    cache_position = torch.arange(0, input_ids.shape[1])

    # position_ids is NOT part of construct_inputs() in the benchmark, which runs on the
    # pinned transformers==5.2.0. This venv has 5.9.0, whose LlamaModel.forward computes
    #   position_ids = torch.arange(seq_len, device=xla) + past_key_values.get_seq_length()
    # (modeling_llama.py:394-397) WITHOUT moving the cache's CPU seq-length tensor to the
    # device, which crashes FakeTensor device propagation during dynamo tracing on TT.
    # (The mask path handles this via .to(device); the position_ids path does not.) We pass
    # position_ids explicitly to skip that branch; for a fresh-cache prefill it equals what
    # the model would compute (arange(seq_len) + 0), so the CPU golden is unchanged.
    position_ids = cache_position.unsqueeze(0)

    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "cache_position": cache_position,
        "position_ids": position_ids,
        "use_cache": True,
    }


def _inputs_to_device(inputs, device):
    """Mirrors transfer_to_device() from benchmarks/llm_benchmark.py:179-204 for the
    StaticCache path (Llama 3.1 has no MLA layers). use_cache is a bool and stays put."""
    out = dict(inputs)
    out["input_ids"] = out["input_ids"].to(device)
    out["cache_position"] = out["cache_position"].to(device)
    out["position_ids"] = out["position_ids"].to(device)
    for layer in out["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
    return out


def run_pytorch_model():
    model = load_pytorch_model()
    inputs = load_input()

    with torch.no_grad():
        output = model(**inputs)

    return output.logits[:, -1]


def run_tt_model():
    xr.set_device_type("TT")
    device = torch_xla.device()

    model = load_pytorch_model()
    model = model.to(device, dtype=DATA_FORMAT)

    torch_xla.set_custom_compile_options(COMPILE_OPTIONS)
    # Weight dtype parametrization (llm_benchmark.py:474-485). Single-chip, so no
    # mark_sharding ordering constraint applies.
    apply_weight_dtype_overrides(model, WEIGHT_DTYPE_OVERRIDES)

    wrapper = LastTokenLogitsWrapper(model)
    compiled = torch.compile(wrapper, backend="tt")

    inputs = _inputs_to_device(load_input(), device)

    with torch.no_grad():
        output = compiled(**inputs)

    return output.cpu()


def codegen_model():
    os.environ["XLA_HLO_DEBUG"] = "1"

    model = load_pytorch_model()
    inputs = load_input()

    # codegen_py() (tt_torch/codegen.py) only forwards *args/**kwargs that are
    # torch.Tensor. Our inputs include a StaticCache (past_key_values) and a bool
    # (use_cache), which its filter would drop — leaving the model with no cache and
    # no input_ids context (raises "specify exactly one of input_ids or inputs_embeds").
    # So we inline codegen_py's body but call model(**inputs) directly with the full
    # dict, moving the cache to device ourselves (as run_tt_model does).
    real_compile_options = {
        **COMPILE_OPTIONS,
        "backend": "codegen_py",
        "export_path": OUTPUT_DIR,
        "export_tensors": True,
    }
    torch_xla.set_custom_compile_options(real_compile_options)

    device = torch_xla.device()
    model = model.to(device, dtype=DATA_FORMAT)
    # Same weight dtype parametrization as run_tt_model so codegen matches the device run.
    apply_weight_dtype_overrides(model, WEIGHT_DTYPE_OVERRIDES)
    model.compile(backend="tt", options={"tt_legacy_compile": True})
    inputs = _inputs_to_device(inputs, device)

    with torch.no_grad():
        model(**inputs)

    import torch_xla.core.xla_model as xm

    xm.wait_device_ops()


def compare_pytorch_and_tt_runs():
    # Capture exact PCC from first --golden run and paste here.
    exact_pcc = 0.9921875

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

    parser = argparse.ArgumentParser(
        description="test_llama_3_1_8b_instruct_tp codegen pipeline"
    )

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
