# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Standalone xla.py reproducing tt-xla benchmark tests/benchmark/test_fibo.py::test_fibo.
### Resolved model: briaai/FIBO (HF, gated) -> only the DiT sub-network
###                  (diffusers BriaFiboTransformer2DModel), TP-4 (Megatron-1D).
### No imports from tt-xla/third_party/tt_forge_models or tt-xla/tests/benchmark/*.
###
### FIBO is BRIA AI's 8B DiT flow-matching text-to-image model (SmolLM3-3B text
### encoder + Wan 2.2 VAE + DimFusion conditioning DiT). This script benchmarks
### ONLY the compute-dominant DiT forward, exactly like test_fibo.py, which drives
### it through benchmarks/video_gen_benchmark.py::benchmark_video_gen_torch_xla.
###
### Hardware target: QB2 (4x Blackhole). This is the benchmark's NATIVE target
### (perf-bench-matrix.json: runs-on = qb2-blackhole) -- NOT a retarget. The DiT
### is 8B and OOMs on a single chip, so it only runs sharded (TP-4).
###
### The gated briaai/FIBO repo requires accepting the bria-fibo license on HF and
### authenticating (HF_TOKEN) the first time; afterwards it loads from the HF cache.
### diffusers must expose BriaFiboPipeline (git-main, or a build that ships it).

import os
from pathlib import Path

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from tt_torch import codegen_py  # noqa: F401 -- kept for parity; see codegen_model note

MODEL_ID = "briaai/FIBO"
DATA_FORMAT = torch.bfloat16  # test_fibo.py: DTYPE = torch.bfloat16
BATCH_SIZE = 2  # CFG doubles the pipeline's batch_size=1 -> 2 effective (uncond + text)

# Informational only (asserted by the benchmark, not by --golden here).
# test_fibo.py: REQUIRED_PCC = 0.97
REQUIRED_PCC = 0.97

OUTPUT_DIR = str(Path(__file__).resolve().parent / "model")

# From test_fibo.py: CompilerConfig(optimization_level=1, enable_trace=False).
# CompilerConfig.to_torch_compile_options() stringifies optimization_level and
# omits every option left at its default (permute-matmul fusion, all-reduce
# workaround, weight dtype, trace) -- so the benchmark's effective dict is just
# {"optimization_level": "1"}. The benchmark also injects export_path /
# export_model_name / ttnn_perf_metrics_* (runtime instrumentation) -- dropped here.
COMPILE_OPTIONS = {
    "optimization_level": "1",
    # "enable_trace": "true",  # benchmark used enable_trace=False (trace off)
}

# ---------------------------------------------------------------------------
# Tensor-parallel mesh + shard spec (inlined from the FIBO loader's
# src/shard_specs.py). Pure Megatron-1D over a (None, "model") mesh.
# ---------------------------------------------------------------------------

MESH_NAMES = (None, "model")
MESH_SHAPES = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (1, 8)}


def _build_mesh():
    """Mirror loader.get_mesh_config -> get_mesh_shape + infra get_mesh.

    On qb2 (4 chips) this is a (1, 4) mesh over (None, "model") -- pure TP-4.
    """
    num_devices = xr.global_runtime_device_count()
    if num_devices not in MESH_SHAPES:
        raise ValueError(
            f"FIBO TP supports {sorted(MESH_SHAPES)} devices, got {num_devices}."
        )
    mesh_shape = MESH_SHAPES[num_devices]
    device_ids = np.array(range(num_devices))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices.")
    return Mesh(device_ids, mesh_shape, MESH_NAMES)


def _enable_spmd():
    """Mirror infra.utilities.torch_multichip_utils.enable_spmd."""
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def _add(specs, tensor, spec):
    if tensor is None:
        return
    if tensor.dim() != len(spec):
        raise ValueError(
            f"Shard spec {spec} rank {len(spec)} != tensor rank {tensor.dim()} "
            f"(shape {tuple(tensor.shape)})."
        )
    specs[tensor] = spec


def _shard_linear_column(specs, linear, has_bias=True):
    """Column-parallel: shard output dim of weight [out, in] on model."""
    _add(specs, linear.weight, ("model", None))
    if has_bias and getattr(linear, "bias", None) is not None:
        _add(specs, linear.bias, ("model",))


def _shard_linear_row(specs, linear):
    """Row-parallel: shard contracting dim; output all-reduced -> replicated bias."""
    _add(specs, linear.weight, (None, "model"))
    if getattr(linear, "bias", None) is not None:
        _add(specs, linear.bias, (None,))


def _shard_feed_forward(specs, ff):
    """diffusers FeedForward: net[0] is GELU(.proj) (column), net[2] Linear (row)."""
    _shard_linear_column(specs, ff.net[0].proj)
    _shard_linear_row(specs, ff.net[2])


def _shard_mmdit_block(specs, block):
    """Shard one BriaFiboTransformerBlock (joint image/text MMDiT block)."""
    _shard_linear_row(specs, block.norm1.linear)
    _shard_linear_row(specs, block.norm1_context.linear)

    attn = block.attn
    _shard_linear_column(specs, attn.to_q)
    _shard_linear_column(specs, attn.to_k)
    _shard_linear_column(specs, attn.to_v)
    _shard_linear_column(specs, attn.add_q_proj)
    _shard_linear_column(specs, attn.add_k_proj)
    _shard_linear_column(specs, attn.add_v_proj)
    _shard_linear_row(specs, attn.to_out[0])
    _shard_linear_row(specs, attn.to_add_out)

    _shard_feed_forward(specs, block.ff)
    _shard_feed_forward(specs, block.ff_context)


def _shard_single_block(specs, block):
    """Shard one BriaFiboSingleTransformerBlock (fused attn+MLP block)."""
    _shard_linear_row(specs, block.norm.linear)

    attn = block.attn
    _shard_linear_column(specs, attn.to_q)
    _shard_linear_column(specs, attn.to_k)
    _shard_linear_column(specs, attn.to_v)

    _shard_linear_column(specs, block.proj_mlp)
    _shard_linear_row(specs, block.proj_out)


def _build_shard_spec(model):
    """Build {parameter: partition_spec}; absent params are replicated."""
    transformer = getattr(model, "transformer", model)
    specs = {}
    for block in transformer.transformer_blocks:
        _shard_mmdit_block(specs, block)
    for block in transformer.single_transformer_blocks:
        _shard_single_block(specs, block)
    norm_out = getattr(transformer, "norm_out", None)
    if norm_out is not None and hasattr(norm_out, "linear"):
        _shard_linear_row(specs, norm_out.linear)
    return specs


def _apply_tp_sharding(model, mesh):
    """mark_sharding each (tensor, spec); parameters absent are replicated."""
    for tensor, spec in _build_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, spec)


# ---------------------------------------------------------------------------
# Model + input construction (inlined from the FIBO loader's src/model_utils.py).
# The loader loads the full pipeline, drives one short pipe(prompt=...) call with
# a monkey-patched transformer.forward to capture the exact positional tensors the
# DiT consumes, then replays them -- robust to diffusers schema drift.
# ---------------------------------------------------------------------------

# Stub structured-JSON prompt (FIBO is trained on structured captions; SmolLM3
# tokenizes it either way). From src/model_utils.py::BRINGUP_PROMPT.
BRINGUP_PROMPT = (
    '{"subject":"a hyper-detailed, ultra-fluffy owl in moonlit trees",'
    '"style_medium":"photograph","camera":"85mm prime, shallow depth of field",'
    '"lighting":"cool moonlight with subtle silver highlights"}'
)
GUIDANCE_SCALE = 5.0  # loader default (model card Generate example)
NUM_INFERENCE_STEPS = 50
SEED = 42


class _ShortCircuit(Exception):
    """Raised inside the patched transformer to abort the pipeline after step 0."""


class FiboTransformerWrapper(torch.nn.Module):
    """Replay the captured (args, kwargs) call shape against the FIBO DiT.

    The benchmark calls model(*inputs) positionally; this splits the flat
    positional tuple back into the transformer's args + kwargs.
    """

    def __init__(self, transformer, capture):
        super().__init__()
        self.transformer = transformer
        self._num_args = len(capture["args"])
        self._kwarg_keys = tuple(capture["kwargs"].keys())

    def forward(self, *inputs):
        expected = self._num_args + len(self._kwarg_keys)
        if len(inputs) != expected:
            raise ValueError(
                f"FiboTransformerWrapper expected {expected} positional inputs "
                f"(got {len(inputs)})."
            )
        args = inputs[: self._num_args]
        kwargs = dict(zip(self._kwarg_keys, inputs[self._num_args :]))
        out = self.transformer(*args, **kwargs)
        if isinstance(out, (list, tuple)):
            return out[0]
        if hasattr(out, "sample"):
            return out.sample
        return out


# Module-level cache: the pipeline + capture are expensive (24G on disk) and both
# load_pytorch_model() and load_input() need them, so build once (as the loader
# does: load_model triggers the capture, load_inputs replays it).
_PIPE = None
_CAPTURE = None


def _load_pipe():
    try:
        from diffusers import BriaFiboPipeline  # type: ignore
    except ImportError:
        from diffusers import DiffusionPipeline as BriaFiboPipeline  # type: ignore

    pipe = BriaFiboPipeline.from_pretrained(MODEL_ID, torch_dtype=DATA_FORMAT)
    pipe.to("cpu")
    for attr in ("text_encoder", "transformer", "vae"):
        module = getattr(pipe, attr, None)
        if module is None:
            continue
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False
    return pipe


def _capture_transformer_inputs(pipe):
    if not hasattr(pipe, "transformer"):
        raise RuntimeError("FIBO pipeline does not expose a .transformer attribute.")

    capture = {}
    original_forward = pipe.transformer.forward

    def patched_forward(*args, **kwargs):
        capture["args"] = args
        capture["kwargs"] = kwargs
        out = original_forward(*args, **kwargs)
        capture["output"] = out
        raise _ShortCircuit()

    pipe.transformer.forward = patched_forward
    try:
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        with torch.no_grad():
            try:
                pipe(
                    prompt=BRINGUP_PROMPT,
                    negative_prompt=None,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=generator,
                    output_type="latent",
                )
            except _ShortCircuit:
                pass
    finally:
        pipe.transformer.forward = original_forward

    if "args" not in capture:
        raise RuntimeError("FIBO pipeline never invoked the patched transformer.")
    return capture


def _ensure_pipe_and_capture():
    global _PIPE, _CAPTURE
    if _CAPTURE is not None:
        return _PIPE, _CAPTURE
    pipe = _load_pipe()
    capture = _capture_transformer_inputs(pipe)
    _PIPE, _CAPTURE = pipe, capture
    return _PIPE, _CAPTURE


def load_pytorch_model():
    pipe, capture = _ensure_pipe_and_capture()
    # Mirror loader.load_model: explicitly (re)cast the transformer to the dtype.
    pipe.transformer = pipe.transformer.to(DATA_FORMAT)
    return FiboTransformerWrapper(pipe.transformer, capture).eval()


def load_input():
    """Replay positional_inputs_from_capture + load_inputs' float-only cast.

    Returns a positional tuple:
      (hidden_states, timestep, encoder_hidden_states, text_encoder_layers,
       joint_attention_kwargs, return_dict, txt_ids, img_ids)
    Non-tensor entries (joint_attention_kwargs=None, return_dict=False) and the
    text_encoder_layers list are passed through unchanged (already bf16 from the
    bf16 pipeline); only floating-point tensors are cast to DATA_FORMAT.
    """
    _, capture = _ensure_pipe_and_capture()
    flat = list(capture["args"])
    for value in capture["kwargs"].values():
        flat.append(value)

    cast = []
    for value in flat:
        if torch.is_tensor(value) and value.is_floating_point():
            cast.append(value.to(DATA_FORMAT))
        else:
            cast.append(value)
    return tuple(cast)


def _to_device(x, device):
    """Recursively move tensors in nested containers to device (mirror harness)."""
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


def run_pytorch_model():
    model = load_pytorch_model()
    inputs = load_input()

    with torch.no_grad():
        output = model(*inputs)  # positional replay; wrapper returns the sample tensor

    return output


def run_tt_model():
    # Order mirrors benchmark_video_gen_torch_xla: build mesh -> enable SPMD ->
    # set options -> device -> model.to(device) -> mark_sharding -> compile.
    xr.set_device_type("TT")

    model = load_pytorch_model()  # already bf16 (CPU)

    mesh = _build_mesh()
    _enable_spmd()  # must precede xla_device() so tensors get presharded annotations

    torch_xla.set_custom_compile_options(COMPILE_OPTIONS)
    device = xm.xla_device()

    model = model.to(device)  # benchmark moves without dtype (weights already bf16)
    _apply_tp_sharding(model, mesh)  # mark_sharding on the on-device weights

    compiled = torch.compile(model, backend="tt")

    inputs = _to_device(load_input(), device)

    with torch.no_grad():
        output = compiled(*inputs)

    return output.cpu()


def codegen_model():
    # NOTE: tt_torch.codegen_py drops non-tensor positional args (and can't take a
    # list arg like text_encoder_layers), which would break the wrapper's replay.
    # We inline codegen_py's body but move inputs with the recursive _to_device and
    # apply TP sharding -- otherwise identical (backend=codegen_py, tt_legacy_compile).
    os.environ["XLA_HLO_DEBUG"] = "1"
    xr.set_device_type("TT")

    model = load_pytorch_model()

    mesh = _build_mesh()
    _enable_spmd()

    compile_opts = {
        **COMPILE_OPTIONS,
        "backend": "codegen_py",
        "export_path": OUTPUT_DIR,
        "export_tensors": True,
    }
    torch_xla.set_custom_compile_options(compile_opts)
    device = xm.xla_device()

    model = model.to(device)
    _apply_tp_sharding(model, mesh)  # CRITICAL: sharded codegen, not single-chip

    model.compile(backend="tt", options={"tt_legacy_compile": True})
    inputs = _to_device(load_input(), device)

    with torch.no_grad():
        model(*inputs)
    xm.wait_device_ops()


def compare_pytorch_and_tt_runs():
    # Capture exact PCC from first --golden run and paste here.
    exact_pcc = 0.999389

    pt_output = run_pytorch_model()
    tt_output = run_tt_model()

    assert pt_output.shape == tt_output.shape, (
        f"shape mismatch: {pt_output.shape} vs {tt_output.shape}"
    )
    assert pt_output.dtype == tt_output.dtype, (
        f"dtype mismatch: {pt_output.dtype} vs {tt_output.dtype}"
    )
    x, y = pt_output.flatten().to(torch.float32), tt_output.flatten().to(torch.float32)
    vx, vy = x - x.mean(), y - y.mean()
    pcc = ((vx @ vy) / (vx.norm() * vy.norm())).item()
    print(f"PCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="test_fibo codegen pipeline")

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
