# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Standalone xla.py reproducing tt-xla benchmark test_llms.py::test_llama_3_1_8b_instruct_tp
### in DECODE-ONLY mode. Resolved model: meta-llama/Llama-3.1-8B-Instruct (HF) via AutoModelForCausalLM.
### Original benchmark: n300-llmbox (1x8 wormhole), tensor-parallel. Per user request this is
### retargeted to a SINGLE p150 (blackhole) chip with NO tensor parallelism AND runs decode-only:
### PREFILL is run on CPU to populate the KV cache, then ONLY the first decode step (single token vs
### the populated cache) runs on the TT device. This mirrors benchmark_llm_torch_xla's decode_only=True
### path (CPU prefill baseline; device runs only decode). The device therefore compiles a SINGLE graph
### (decode); --codegen emits just graph_0. The benchmark only shards when num_devices > 1
### (llm_benchmark.py:340-342), so on one chip it runs unsharded — all mesh/sharding machinery is dropped.
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


def _build_prefill_inputs(model):
    """Full-prompt prefill inputs against a fresh StaticCache, built from the model's
    (patched) config. Replicates construct_inputs() (benchmarks/llm_benchmark.py:85-169)
    plus init_static_cache() (llm_utils/decode_utils.py:113-145)."""
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
    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "cache_position": cache_position,
        "position_ids": cache_position.unsqueeze(0),
        "use_cache": True,
    }


def cpu_prefill_to_decode_inputs(model):
    """Run PREFILL on CPU to populate the KV cache (this is the ONLY prefill anywhere —
    it never runs on device), then return the first-decode inputs: the prefill's argmax
    token as the single input, cache_position/position_ids at the slot right after the
    prompt, and the now-populated CPU StaticCache. The device path moves this cache over
    and runs ONLY the decode step. Mirrors decode_only=True in benchmark_llm_torch_xla
    (llm_benchmark.py:399-430) and LLMSamplingWrapper's next_token = logits[:, -1].argmax
    and next_cache_position = cache_position[-1:] + 1 (llm_utils/decode_utils.py:67,80).

    `model` must be on CPU here so the cache is generated on CPU.
    """
    prefill_inputs = _build_prefill_inputs(model)
    with torch.no_grad():
        prefill_logits = model(**prefill_inputs).logits[:, -1]  # [batch, vocab]
    next_token = prefill_logits.argmax(dim=-1, keepdim=True)  # [batch, 1]
    cache_position = prefill_inputs["cache_position"][-1:] + 1  # [1], value == prompt_len

    # position_ids must be passed explicitly for the DEVICE decode: this venv is
    # transformers 5.9.0 (the benchmark pins 5.2.0), whose LlamaModel.forward computes
    #   position_ids = torch.arange(seq_len, device=xla) + past_key_values.get_seq_length()
    # (modeling_llama.py:394-397) WITHOUT moving the cache's CPU seq-length tensor to the
    # device, crashing FakeTensor device propagation during tracing on TT. (The mask path
    # handles it via .to(device); the position_ids path does not.) For a single decode
    # token at position prompt_len this equals cache_position, so it is exact.
    return {
        "input_ids": next_token,
        "past_key_values": prefill_inputs["past_key_values"],  # populated by the CPU prefill
        "cache_position": cache_position,
        "position_ids": cache_position.unsqueeze(0),
        "use_cache": True,
    }


def _inputs_to_device(inputs, device):
    """Mirrors transfer_to_device() from benchmarks/llm_benchmark.py:179-204 for the
    StaticCache path (Llama 3.1 has no MLA layers). use_cache is a bool and stays put;
    cumulative_length stays on CPU (the mask path moves it to device where needed)."""
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
    # Prefill on CPU populates the cache; then the CPU decode is the PCC reference.
    decode_inputs = cpu_prefill_to_decode_inputs(model)

    with torch.no_grad():
        decode_logits = model(**decode_inputs).logits[:, -1]

    print(f"[cpu] decode logits {tuple(decode_logits.shape)}")
    return decode_logits


def run_tt_model():
    xr.set_device_type("TT")
    device = torch_xla.device()

    model = load_pytorch_model()

    # 1) PREFILL ON CPU — generate the KV cache with the model still on CPU.
    decode_inputs = cpu_prefill_to_decode_inputs(model)

    # 2) Move the model to device and compile ONLY the decode step.
    model = model.to(device, dtype=DATA_FORMAT)
    torch_xla.set_custom_compile_options(COMPILE_OPTIONS)
    # Weight dtype parametrization (llm_benchmark.py:474-485). Single-chip, so no
    # mark_sharding ordering constraint applies.
    apply_weight_dtype_overrides(model, WEIGHT_DTYPE_OVERRIDES)
    wrapper = LastTokenLogitsWrapper(model)
    compiled = torch.compile(wrapper, backend="tt")

    # 3) Move the CPU-populated cache + decode inputs to device; run the single decode graph.
    decode_inputs = _inputs_to_device(decode_inputs, device)
    with torch.no_grad():
        decode_logits = compiled(**decode_inputs)

    return decode_logits.cpu()


def codegen_model():
    os.environ["XLA_HLO_DEBUG"] = "1"

    model = load_pytorch_model()

    # PREFILL ON CPU — generate the KV cache (model on CPU).
    decode_inputs = cpu_prefill_to_decode_inputs(model)

    # codegen_py() (tt_torch/codegen.py) only forwards *args/**kwargs that are
    # torch.Tensor. Our inputs include a StaticCache (past_key_values) and a bool
    # (use_cache), which its filter would drop. So we inline codegen_py's body but call
    # model(**decode_inputs) directly, moving the cache to device ourselves. A single
    # decode forward => a single graph (graph_0).
    real_compile_options = {
        **COMPILE_OPTIONS,
        "backend": "codegen_py",
        "export_path": OUTPUT_DIR,
        "export_tensors": True,
    }
    torch_xla.set_custom_compile_options(real_compile_options)

    device = torch_xla.device()
    model = model.to(device, dtype=DATA_FORMAT)
    apply_weight_dtype_overrides(model, WEIGHT_DTYPE_OVERRIDES)
    model.compile(backend="tt", options={"tt_legacy_compile": True})
    decode_inputs = _inputs_to_device(decode_inputs, device)

    with torch.no_grad():
        model(**decode_inputs)

    import torch_xla.core.xla_model as xm

    xm.wait_device_ops()


def compare_pytorch_and_tt_runs():
    # Decode PCC. Capture from the first --golden run and paste here.
    # (Not yet captured on device — device was busy when decode-only was added.)
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

    parser = argparse.ArgumentParser(
        description="test_llama_3_1_8b_instruct_tp decode-only codegen pipeline"
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
