# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Standalone xla.py reproducing tt-xla benchmark test_llms.py::test_gemma_1_1_2b.
### Resolved model: google/gemma-1.1-2b-it (HF) via AutoModelForCausalLM.
### No imports from tt-xla/third_party/tt_forge_models or tt-xla/tests/benchmark/*.

import os
from pathlib import Path

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache
from tt_torch import codegen_py
from tt_torch.weight_dtype import apply_weight_dtype_overrides

MODEL_ID = "google/gemma-1.1-2b-it"
DATA_FORMAT = torch.bfloat16     # test_llms.py: DEFAULT_DATA_FORMAT = "bfloat16"
BATCH_SIZE = 32                  # test_llms.py: DEFAULT_BATCH_SIZE = 32
INPUT_SEQUENCE_LENGTH = 128      # test_llms.py: DEFAULT_INPUT_SEQUENCE_LENGTH = 128
DEFAULT_INPUT_PROMPT = (
    "Here is an exaustive list of the best practices for writing clean code:"
)

OUTPUT_DIR = str(Path(__file__).resolve().parent / "model")

# Per-test overrides applied:
#   optimization_level=2 (DEFAULT_OPTIMIZATION_LEVEL from test_llms.py)
# Benchmark-file defaults applied:
#   experimental_weight_dtype="bfp_bf8"  (DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE in test_llms.py)
#   experimental_enable_permute_matmul_fusion=False  (DEFAULT_EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION)
COMPILE_OPTIONS = {
    "optimization_level": 2,
    # "enable_trace": True,  # benchmark used trace_enabled=True (DEFAULT_TRACE_ENABLED)
    "experimental_weight_dtype": "bfp_bf8",
    "experimental_enable_permute_matmul_fusion": False,
}


def load_pytorch_model():
    # Mirrors third_party/tt_forge_models/gemma/pytorch/loader.py:load_model
    # (HUGGING_FACE branch): AutoModelForCausalLM.from_pretrained with torch_dtype at load time.
    # The loader also sets config.use_cache = False before loading.
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DATA_FORMAT, config=config
    )

    # setup_model_and_tokenizer patches from benchmarks/llm_benchmark.py:73-78.
    if hasattr(model.config, "layer_types"):
        model.config.layer_types = ["full_attention"] * len(model.config.layer_types)
    if hasattr(model.config, "_experts_implementation"):
        model.config._experts_implementation = "dense"

    model.eval()
    return model


class LastTokenLogitsWrapper(torch.nn.Module):
    """Mirrors LLMSamplingWrapper from llm_utils/decode_utils.py."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, past_key_values, cache_position, use_cache=True):
        output = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
        )
        logits = output.logits
        next_token_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
        next_cache_position = cache_position[-1:] + 1
        return next_token_ids, next_token_ids, next_cache_position


def load_input():
    """Replicates construct_inputs() from benchmarks/llm_benchmark.py:85-169."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    # early_initialization is required to pre-allocate key/value tensors so that
    # _inputs_to_device can move them to the XLA device before the forward pass
    # (mirrors init_static_cache in benchmarks/llm_utils/decode_utils.py:113-145).
    model = load_pytorch_model()
    cfg = model.config
    head_dim = (
        cfg.head_dim
        if hasattr(cfg, "head_dim") and cfg.head_dim
        else cfg.hidden_size // cfg.num_attention_heads
    )
    num_key_value_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)

    past_key_values = StaticCache(
        config=cfg,
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
    """Mirrors transfer_to_device() from benchmarks/llm_benchmark.py:179-205 for the
    StaticCache path (no MLA layers in Gemma 1.1-2b-it).
    Also moves cumulative_length to the device so that get_seq_length() returns a tensor
    on the same device as inputs_embeds when the model computes position_ids."""
    out = dict(inputs)
    out["input_ids"] = out["input_ids"].to(device)
    out["cache_position"] = out["cache_position"].to(device)
    for layer in out["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
        if hasattr(layer, "cumulative_length") and isinstance(
            layer.cumulative_length, torch.Tensor
        ):
            layer.cumulative_length = layer.cumulative_length.to(device)
        # StaticLayer.device is used internally for torch.arange in update(); keep in sync.
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
    xr.set_device_type("TT")
    device = torch_xla.device()

    model = load_pytorch_model()
    model = model.to(device, dtype=DATA_FORMAT)

    torch_xla.set_custom_compile_options(COMPILE_OPTIONS)
    apply_weight_dtype_overrides(model, {"default": "bfp_bf8"})

    wrapper = LastTokenLogitsWrapper(model)
    compiled = torch.compile(wrapper, backend="tt")

    inputs = _inputs_to_device(load_input(), device)

    with torch.no_grad():
        next_token_ids, _, _ = compiled(**inputs)

    return next_token_ids.cpu()


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
    exact_pcc = 0.7578125

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

    parser = argparse.ArgumentParser(description="test_gemma_1_1_2b codegen pipeline")

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
