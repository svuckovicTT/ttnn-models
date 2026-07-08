# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Standalone xla.py reproducing tt-xla benchmark tests/benchmark/test_fibo.py::test_fibo.
### Resolved model: briaai/FIBO (HF, GATED) -- only the DiT sub-network (pipe.transformer,
###   a diffusers BriaFiboTransformer2DModel), driven through the stock BriaFiboPipeline.
### No imports from tt-xla/third_party/tt_forge_models or tt-xla/tests/benchmark/*.
###
### Tensor-parallel: Megatron-1D TP-4 over a (None, "model") mesh on QB2 (native target,
###   no retargeting). Effective batch = 2 (classifier-free guidance, guidance_scale=5.0).
###
### NOTE: briaai/FIBO is a GATED Hugging Face repo. Accept the bria-fibo license on HF and
###   export HF_TOKEN before running. Loading drives one short pipe() call (SmolLM3-3B text
###   encoder on CPU) to capture the exact positional tensors the transformer consumes -- the
###   same input-capture the tt_forge_models FIBO loader performs.

import os
from pathlib import Path

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

MODEL_ID = "briaai/FIBO"
DATA_FORMAT = torch.bfloat16     # test_fibo.py: DTYPE = torch.bfloat16
BATCH_SIZE = 2                   # effective batch from CFG (guidance_scale=5.0 -> 2 latents)

# FIBO model-card Generate example defaults (mirrors src/model_utils.py).
BRINGUP_PROMPT = (
    '{"subject":"a hyper-detailed, ultra-fluffy owl in moonlit trees",'
    '"style_medium":"photograph","camera":"85mm prime, shallow depth of field",'
    '"lighting":"cool moonlight with subtle silver highlights"}'
)
DEFAULT_NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 5.0
CAPTURE_SEED = 42

# Tensor-parallel mesh (mirrors src/shard_specs.py). Pure Megatron-1D: only the
# "model" axis shards; leading axis unnamed (no data parallelism).
MESH_NAMES = (None, "model")
MESH_SHAPES = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (1, 8)}

OUTPUT_DIR = str(Path(__file__).resolve().parent / "model")

# Per-test override: CompilerConfig(optimization_level=1, enable_trace=False).
# opt=1 is the model-perf-tuning winner (~5.6% faster DiT forward than opt=0
# bringup default). Values are strings, matching CompilerConfig.to_torch_compile_options.
COMPILE_OPTIONS = {
    "optimization_level": "1",
    # "enable_trace": "true",  # benchmark used enable_trace=False (trace was inert here)
}


# --------------------------------------------------------------------------- #
# Inlined from tt_forge_models/fibo/pytorch/src/model_utils.py
# --------------------------------------------------------------------------- #
class _ShortCircuit(Exception):
    """Raised inside the patched transformer to abort the pipeline after step 0."""


def _load_pipe(pretrained_model_name, dtype_override):
    """Load the FIBO pipeline (stock diffusers), eval + requires_grad disabled."""
    try:
        from diffusers import BriaFiboPipeline  # type: ignore
    except ImportError:
        from diffusers import DiffusionPipeline as BriaFiboPipeline  # type: ignore

    pipe_kwargs = {}
    if dtype_override is not None:
        pipe_kwargs["torch_dtype"] = dtype_override
    pipe = BriaFiboPipeline.from_pretrained(pretrained_model_name, **pipe_kwargs)
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


def _capture_transformer_inputs(pipe, prompt, guidance_scale):
    """Drive pipe(prompt=...) for one transformer step and capture its inputs.

    Monkey-patches transformer.forward to record the exact positional args +
    kwargs, then short-circuits the denoising loop after the first forward.
    """
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
        generator = torch.Generator(device="cpu").manual_seed(CAPTURE_SEED)
        with torch.no_grad():
            try:
                pipe(
                    prompt=prompt,
                    negative_prompt=None,
                    num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
                    guidance_scale=guidance_scale,
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


def _positional_inputs_from_capture(capture):
    """Flatten captured (args, kwargs) into a positional tuple: args then kwargs.values()."""
    flat = list(capture["args"])
    for value in capture["kwargs"].values():
        flat.append(value)
    return tuple(flat)


class FiboTransformerWrapper(torch.nn.Module):
    """Wrap the FIBO transformer so it accepts the captured positional inputs.

    Splits the positional inputs back into the captured (args, kwargs) call shape,
    forwards to the underlying transformer, and returns a single output tensor.
    """

    def __init__(self, transformer, capture):
        super().__init__()
        self.transformer = transformer
        self._num_args = len(capture["args"])
        self._kwarg_keys = tuple(capture["kwargs"].keys())

    def forward(self, *inputs):
        if len(inputs) != self._num_args + len(self._kwarg_keys):
            raise ValueError(
                f"FiboTransformerWrapper expected "
                f"{self._num_args + len(self._kwarg_keys)} positional inputs "
                f"(got {len(inputs)})."
            )
        args = inputs[: self._num_args]
        kwargs = dict(zip(self._kwarg_keys, inputs[self._num_args:]))
        out = self.transformer(*args, **kwargs)
        if isinstance(out, (list, tuple)):
            return out[0]
        if hasattr(out, "sample"):
            return out.sample
        return out


# --------------------------------------------------------------------------- #
# Inlined shard spec from tt_forge_models/fibo/pytorch/src/shard_specs.py
# --------------------------------------------------------------------------- #
def _add(specs, tensor, spec):
    if tensor is None:
        return
    if tensor.dim() != len(spec):
        raise ValueError(
            f"Shard spec {spec} has rank {len(spec)} but tensor has rank "
            f"{tensor.dim()} (shape {tuple(tensor.shape)})."
        )
    specs[tensor] = spec


def _shard_linear_column(specs, linear, has_bias=True):
    _add(specs, linear.weight, ("model", None))
    if has_bias and getattr(linear, "bias", None) is not None:
        _add(specs, linear.bias, ("model",))


def _shard_linear_row(specs, linear):
    _add(specs, linear.weight, (None, "model"))
    if getattr(linear, "bias", None) is not None:
        _add(specs, linear.bias, (None,))


def _shard_feed_forward(specs, ff):
    _shard_linear_column(specs, ff.net[0].proj)
    _shard_linear_row(specs, ff.net[2])


def _shard_mmdit_block(specs, block):
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
    _shard_linear_row(specs, block.norm.linear)
    attn = block.attn
    _shard_linear_column(specs, attn.to_q)
    _shard_linear_column(specs, attn.to_k)
    _shard_linear_column(specs, attn.to_v)
    _shard_linear_column(specs, block.proj_mlp)
    _shard_linear_row(specs, block.proj_out)


def _build_shard_spec(wrapper):
    """Build {parameter: partition_spec} for the on-device FIBO transformer."""
    transformer = getattr(wrapper, "transformer", wrapper)
    specs = {}
    for block in transformer.transformer_blocks:
        _shard_mmdit_block(specs, block)
    for block in transformer.single_transformer_blocks:
        _shard_single_block(specs, block)
    norm_out = getattr(transformer, "norm_out", None)
    if norm_out is not None and hasattr(norm_out, "linear"):
        _shard_linear_row(specs, norm_out.linear)
    return specs


# --------------------------------------------------------------------------- #
# Pipeline + capture caching (capture is expensive: loads the full pipeline)
# --------------------------------------------------------------------------- #
_PIPE = None
_CAPTURE = None


def _ensure_pipe_and_capture():
    global _PIPE, _CAPTURE
    if _CAPTURE is not None:
        return _PIPE, _CAPTURE
    _PIPE = _load_pipe(MODEL_ID, DATA_FORMAT)
    _CAPTURE = _capture_transformer_inputs(
        _PIPE, prompt=BRINGUP_PROMPT, guidance_scale=GUIDANCE_SCALE
    )
    return _PIPE, _CAPTURE


def _to_device(value, device):
    """Recursively move tensors in nested containers to device; pass others through."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(v, device) for v in value)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------- #
# Entry-point building blocks
# --------------------------------------------------------------------------- #
def load_pytorch_model():
    pipe, capture = _ensure_pipe_and_capture()
    pipe.transformer = pipe.transformer.to(DATA_FORMAT)
    return FiboTransformerWrapper(pipe.transformer, capture).eval()


def load_input():
    """Replay the captured positional inputs, casting floating tensors to DATA_FORMAT."""
    _, capture = _ensure_pipe_and_capture()
    inputs = _positional_inputs_from_capture(capture)
    cast = []
    for value in inputs:
        if torch.is_tensor(value) and value.is_floating_point():
            cast.append(value.to(DATA_FORMAT))
        else:
            cast.append(value)
    return list(cast)


def _build_mesh():
    num_devices = xr.global_runtime_device_count()
    if num_devices not in MESH_SHAPES:
        raise ValueError(
            f"FIBO tensor-parallel supports device counts {sorted(MESH_SHAPES)}, "
            f"got {num_devices}."
        )
    device_ids = list(range(num_devices))
    return Mesh(device_ids, MESH_SHAPES[num_devices], MESH_NAMES)


def _apply_sharding(wrapper_on_device, mesh):
    """mark_sharding each mapped weight; unmapped params replicate across the mesh."""
    for tensor, spec in _build_shard_spec(wrapper_on_device).items():
        xs.mark_sharding(tensor, mesh, spec)


def _enable_spmd():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def run_pytorch_model():
    model = load_pytorch_model()
    inputs = load_input()

    with torch.no_grad():
        output = model(*inputs)

    return output


def run_tt_model():
    xr.set_device_type("TT")
    _enable_spmd()                       # SPMD BEFORE device() so tensors get presharded
    mesh = _build_mesh()
    torch_xla.set_custom_compile_options(COMPILE_OPTIONS)
    device = torch_xla.device()

    wrapper = load_pytorch_model()
    wrapper_on_device = wrapper.to(device)
    if hasattr(wrapper_on_device, "tie_weights"):
        wrapper_on_device.tie_weights()

    inputs = [_to_device(t, device) for t in load_input()]

    # mark_sharding on the on-device weights (matches benchmark ordering).
    _apply_sharding(wrapper_on_device, mesh)

    compiled = torch.compile(wrapper_on_device, backend="tt")

    with torch.no_grad():
        output = compiled(*inputs)

    return output.cpu()


def codegen_model():
    os.environ["XLA_HLO_DEBUG"] = "1"
    xr.set_device_type("TT")
    _enable_spmd()
    mesh = _build_mesh()

    # Mirror tt_torch.codegen.codegen_py, but with recursive input device-move and
    # SPMD sharding (codegen_py drops non-tensor / nested inputs, which FIBO has).
    torch_xla.set_custom_compile_options(
        {
            **COMPILE_OPTIONS,
            "backend": "codegen_py",
            "export_path": OUTPUT_DIR,
            "export_tensors": True,
        }
    )
    device = xm.xla_device()

    wrapper = load_pytorch_model()
    wrapper_on_device = wrapper.to(device)
    if hasattr(wrapper_on_device, "tie_weights"):
        wrapper_on_device.tie_weights()

    _apply_sharding(wrapper_on_device, mesh)

    wrapper_on_device.compile(backend="tt", options={"tt_legacy_compile": True})
    inputs = [_to_device(t, device) for t in load_input()]

    with torch.no_grad():
        wrapper_on_device(*inputs)
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
    x, y = pt_output.flatten().float(), tt_output.flatten().float()
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
