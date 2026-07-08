# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Standalone xla.py reproducing tt-xla benchmark
###   tests/benchmark/test_fibo_text_encoder.py::test_fibo_text_encoder
### Resolved model: HuggingFaceTB/SmolLM3-3B (HF) loaded as SmolLM3Model
###   — the FIBO (briaai/FIBO) text encoder is a base SmolLM3Model (the
###   SmolLM3-3B causal LM with the vocab head discarded); FIBO conditions its
###   DiT on this encoder's last_hidden_state.
###
### Tensor-parallel over a (1, 4) mesh (TP-4) on a 4-chip Blackhole device
### (qb2) — the model-bringup baseline the benchmark targets. The encoder OOMs
### on a single chip at long context, so TP is mandatory.
###
### No imports from tt-xla/third_party/tt_forge_models or tt-xla/tests/benchmark/*.
### The benchmark's text_encoder loader (fibo.text_encoder.pytorch.loader) is
### re-expressed here with stock transformers: the public SmolLM3-3B checkpoint,
### a Megatron-1D shard spec built inline, and the (1, 4) mesh from the loader's
### FIBO shard-spec conventions.

import os
from pathlib import Path

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoTokenizer, SmolLM3Model
from tt_torch import codegen_py

MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
DATA_FORMAT = torch.bfloat16  # test pins load_model(dtype_override=torch.bfloat16)
BATCH_SIZE = 1  # benchmark drives batch_size=1

# Benchmark pins the context length to the TP-4-validated maximum (24576;
# 32768 OOMs at runtime, 65536 needs more chips). The real test forces this via
# os.environ["FIBO_TE_CONTEXT_LENGTH"]. Kept identical here as the default, but
# honoring the env var lets you validate PCC locally at a small context (the
# CPU golden of a 3B model at 24576 is impractical on CPU).
CONTEXT_LENGTH = int(os.environ.get("FIBO_TE_CONTEXT_LENGTH", "24576"))

# Tensor-parallel mesh: pure Megatron-1D, "model" axis = TP degree 4.
# Mirrors the FIBO loader's get_mesh_config / shard-spec conventions
# (tt_forge_models/fibo/.../shard_specs.py): MESH_NAMES = (None, "model"),
# MESH_SHAPES[4] = (1, 4).
MESH_SHAPE = (1, 4)
MESH_NAMES = (None, "model")

OUTPUT_DIR = str(Path(__file__).resolve().parent / "model")

# From COMPILER_CONFIG = CompilerConfig(optimization_level=1, enable_trace=False)
# in test_fibo_text_encoder.py, converted via to_torch_compile_options(). Only
# optimization_level is emitted (enable_trace=False leaves the key unset;
# experimental_enable_permute_matmul_fusion defaults True → not emitted).
# NOTE: the harness runtime-instrumentation keys (export_path, export_model_name,
# ttnn_perf_metrics_*) are intentionally dropped — xla.py produces no perf JSON.
COMPILE_OPTIONS = {
    "optimization_level": "1",
    # "enable_trace": "true",  # benchmark used enable_trace=False
}

# Stub structured-JSON prompt lifted from the FIBO loader (model_utils.BRINGUP_PROMPT).
# FIBO is trained on structured captions; the encoder tokenizes them through
# SmolLM3 either way. Content is immaterial to PCC (golden and TT see the same
# input); it is padded to CONTEXT_LENGTH to match the benchmark's fixed-length
# text-encoder contract.
BRINGUP_PROMPT = (
    '{"subject":"a hyper-detailed, ultra-fluffy owl in moonlit trees",'
    '"style_medium":"photograph","camera":"85mm prime, shallow depth of field",'
    '"lighting":"cool moonlight with subtle silver highlights"}'
)


class FiboTextEncoderWrapper(torch.nn.Module):
    """Adapts SmolLM3Model to the ``wrapper(input_ids, attention_mask) -> tensor``
    harness contract, extracting ``last_hidden_state`` — the conditioning
    embedding FIBO's DiT consumes. Copied from test_fibo_text_encoder.py."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state


def load_pytorch_model():
    # Mirrors the text-encoder loader.load_model(dtype_override=torch.bfloat16):
    # the FIBO text encoder is a base SmolLM3Model (no LM head), loaded from the
    # public SmolLM3-3B checkpoint with the dtype applied at load time.
    model = SmolLM3Model.from_pretrained(MODEL_ID, torch_dtype=DATA_FORMAT)
    wrapper = FiboTextEncoderWrapper(model).eval()
    return wrapper


def load_input():
    # Mirrors the text-encoder loader.load_inputs(batch_size=1): tokenize the
    # prompt and pad to the fixed context length, returning input_ids +
    # attention_mask.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    sentences = [BRINGUP_PROMPT for _ in range(BATCH_SIZE)]
    tokenized = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=CONTEXT_LENGTH,
        return_tensors="pt",
    )
    return tokenized["input_ids"], tokenized["attention_mask"]


def _build_mesh():
    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, MESH_SHAPE, MESH_NAMES)
    print(f"Created device mesh: {MESH_SHAPE} with {num_devices} devices.")
    return mesh


def _shard_spec(model):
    """Megatron-1D column→row shard spec for the SmolLM3 decoder stack.

    Reproduces the text-encoder loader.load_shard_spec: attention/MLP are
    sharded column→row on the ``model`` axis (Q/K/V and gate/up column-parallel,
    O and down row-parallel → one all-reduce per pair). SmolLM3 has no linear
    biases (attention_bias=mlp_bias=False). Everything else (embeddings, norms,
    rotary) is replicated by omission.
    """
    specs = {}
    for layer in model.layers:
        attn = layer.self_attn
        specs[attn.q_proj.weight] = ("model", None)
        specs[attn.k_proj.weight] = ("model", None)
        specs[attn.v_proj.weight] = ("model", None)
        specs[attn.o_proj.weight] = (None, "model")
        mlp = layer.mlp
        specs[mlp.gate_proj.weight] = ("model", None)
        specs[mlp.up_proj.weight] = ("model", None)
        specs[mlp.down_proj.weight] = (None, "model")
    return specs


def _apply_tp_sharding(wrapper_on_device, mesh):
    # wrapper_on_device.model is the SmolLM3Model. Must be called AFTER the
    # wrapper is moved to the XLA device (mark_sharding needs XLA tensors).
    for tensor, spec in _shard_spec(wrapper_on_device.model).items():
        xs.mark_sharding(tensor, mesh, spec)


def run_pytorch_model():
    wrapper = load_pytorch_model()
    input_ids, attention_mask = load_input()

    with torch.no_grad():
        output = wrapper(input_ids, attention_mask)

    return output


def run_tt_model():
    # Order mirrors benchmark_video_gen_torch_xla: set device type, build mesh,
    # enable SPMD, set compile options, then move to device and shard.
    xr.set_device_type("TT")

    mesh = _build_mesh()

    # enable_spmd(): ConvertStableHloToSdy so tt-mlir sees shardy annotations.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    torch_xla.set_custom_compile_options(COMPILE_OPTIONS)

    device = xm.xla_device()

    wrapper = load_pytorch_model()
    wrapper_on_device = wrapper.to(device)  # model already bf16 from load

    input_ids, attention_mask = load_input()
    inputs_on_device = [input_ids.to(device), attention_mask.to(device)]

    # Sharding must see XLA tensors — apply after .to(device).
    _apply_tp_sharding(wrapper_on_device, mesh)

    compiled = torch.compile(wrapper_on_device, backend="tt")

    with torch.no_grad():
        output = compiled(*inputs_on_device)

    return output.cpu()


def codegen_model():
    os.environ["XLA_HLO_DEBUG"] = "1"
    xr.set_device_type("TT")

    mesh = _build_mesh()

    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    device = xm.xla_device()

    wrapper = load_pytorch_model()
    wrapper_on_device = wrapper.to(device)

    # CRITICAL: TP sharding must be applied here too (before codegen), or the
    # emitted graph is single-chip instead of sharded. codegen_py's internal
    # model.to(device) is a no-op on the already-on-device wrapper and preserves
    # the sharding annotations.
    _apply_tp_sharding(wrapper_on_device, mesh)

    input_ids, attention_mask = load_input()

    codegen_py(
        wrapper_on_device,
        input_ids,
        attention_mask,
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
    x, y = pt_output.flatten().float(), tt_output.flatten().float()
    vx, vy = x - x.mean(), y - y.mean()
    pcc = ((vx @ vy) / (vx.norm() * vy.norm())).item()
    print(f"PCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="test_fibo_text_encoder codegen pipeline"
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
