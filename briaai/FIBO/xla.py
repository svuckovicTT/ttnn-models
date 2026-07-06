# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Standalone xla.py reproducing tt-xla runner test
###   tests/runner/test_models.py::test_all_models_torch[fibo/pytorch-Base-tensor_parallel-inference]
### Resolved model: briaai/FIBO (BRIA AI 8B DiT text-to-image transformer)
### Target: qb2 (1x4 blackhole), Megatron-1D tensor parallelism over a (None, "model") mesh.
###
### DEVIATION FROM THE benchmark-to-xla SKILL CONTRACT:
###   This is NOT a tests/benchmark/ test — it is a tests/runner/ auto-runner model
###   test. More importantly, FIBO cannot be rebuilt with stock transformers/timm/
###   torchvision: it needs `BriaFiboPipeline` (diffusers git-main only), gated
###   `briaai/FIBO` weights (accept the bria-fibo license + set HF_TOKEN), and its
###   inputs are captured by driving one `pipe(prompt=...)` call with a monkey-patched
###   transformer.forward. There is no stock-API equivalent, so this script imports the
###   tt_forge_models FIBO loader directly and reuses its load_model / load_inputs /
###   get_mesh_config / load_shard_spec. Everything else (SPMD init, mark_sharding,
###   torch.compile(backend="tt"), codegen, PCC) mirrors the runner's Torch TP path.
###
### Prerequisites:
###   - `pip install "git+https://github.com/huggingface/diffusers"` (for BriaFiboPipeline)
###   - Accept the gated bria-fibo license on https://huggingface.co/briaai/FIBO
###   - export HF_TOKEN=<your token>
###   - tt-xla venv active, with third_party/tt_forge_models importable (see TT_XLA_ROOT below).

import os
import sys
from pathlib import Path

import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr

# --- Make the tt_forge_models FIBO loader importable -------------------------
# The loader lives under tt-xla/third_party. This script is lifted out of
# ttnn-models, so we locate tt-xla via the TT_XLA_ROOT env var, falling back to
# the known checkout path. We import the loader module by file path to avoid
# depending on the exact top-level package name on sys.path.
TT_XLA_ROOT = os.environ.get(
    "TT_XLA_ROOT", "/localdev/svuckovic/_workspace/repos/tt-xla"
)
if TT_XLA_ROOT not in sys.path:
    sys.path.insert(0, TT_XLA_ROOT)

from third_party.tt_forge_models.fibo.pytorch.loader import (  # noqa: E402
    ModelLoader,
    ModelVariant,
)

MODEL_ID = "briaai/FIBO"
DATA_FORMAT = torch.bfloat16  # runner loads FIBO with dtype_override=torch.bfloat16
MESH_SHAPE = (1, 4)  # qb2: Megatron-1D TP over 4 blackhole chips
MESH_NAMES = (None, "model")  # from fibo/pytorch/src/shard_specs.py (MESH_NAMES)

OUTPUT_DIR = str(Path(__file__).resolve().parent / "model")

# Runner default CompilerConfig for this test: optimization_level=0, enable_trace=False,
# no weight-dtype override (the YAML entry sets neither optimization_level nor
# enable_weight_bfp8_conversion). opt-level 0 emits no compiler key, so the dict is empty.
COMPILE_OPTIONS = {
    # "optimization_level": 0,   # runner default; opt 0 emits no option key
    # "enable_trace": False,     # runner default (trace disabled)
}


# A single loader instance is shared so the (expensive) pipeline load + input
# capture happens once per process and stays consistent between model and inputs.
_LOADER = ModelLoader(variant=ModelVariant.BASE)


def load_pytorch_model():
    """Return the FiboTransformerWrapper (DiT wrapped for positional inputs)."""
    model = _LOADER.load_model(dtype_override=DATA_FORMAT)
    model.eval()
    return model


def load_input():
    """Return the positional tensor tuple captured from the FIBO pipeline."""
    return _LOADER.load_inputs(dtype_override=DATA_FORMAT)


def _build_mesh():
    """Create the (1, 4) Megatron-1D mesh, matching the runner's get_mesh()."""
    import numpy as np
    from torch_xla.distributed.spmd import Mesh

    num_devices = xr.global_runtime_device_count()
    assert num_devices == MESH_SHAPE[1], (
        f"Expected {MESH_SHAPE[1]} devices for FIBO TP on qb2, got {num_devices}. "
        f"Run on a 4-chip box (or adjust MESH_SHAPE)."
    )
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, MESH_SHAPE, MESH_NAMES)
    xs.set_global_mesh(mesh)
    return mesh


def _apply_tp_sharding(model, mesh):
    """Mark FIBO weight shards on the mesh (loader's Megatron column->row spec).

    Mirrors the runner: shard_spec_fn(model) -> {param: partition_spec}, then
    xs.mark_sharding for each. Must run BEFORE the compiled forward executes.
    Params absent from the spec are replicated.
    """
    shard_specs = _LOADER.load_shard_spec(model)
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)


def _enable_spmd():
    """Enable torch_xla SPMD + Shardy lowering, matching infra.enable_spmd()."""
    # tt-mlir's stablehlo pipeline expects Shardy annotations from pytorch/xla.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def _inputs_to_device(inputs, device):
    """Recursively move every tensor leaf to ``device``, leaving non-tensors as-is.

    FIBO's captured inputs are a mix of tensors, lists of per-layer tensors
    (caption / text-encoder hidden states), and non-tensor values. The runner
    moves them with tree_map; a shallow top-level move leaves the inner tensors
    on CPU and crashes FakeTensor device propagation inside the traced graph.
    """
    from torch.utils._pytree import tree_map

    return tree_map(
        lambda x: x.to(device) if torch.is_tensor(x) else x,
        inputs,
    )


def run_pytorch_model():
    model = load_pytorch_model()
    inputs = load_input()

    with torch.no_grad():
        output = model(*inputs)

    return output


def run_tt_model():
    xr.set_device_type("TT")
    _enable_spmd()  # BEFORE torch_xla.device(): tensors get presharded annotations
    device = torch_xla.device()

    mesh = _build_mesh()

    model = load_pytorch_model()
    model = model.to(device, dtype=DATA_FORMAT)

    # Mark sharding on the real weight tensors (FIBO has no weight-dtype
    # parametrization, so ordering vs. overrides is moot — but keep sharding
    # before compile, matching the runner's put-on-device step).
    _apply_tp_sharding(model, mesh)

    torch_xla.set_custom_compile_options(COMPILE_OPTIONS)

    # torch.compile is lazy; actual trace/compile happens on first call, after
    # sharding is marked — same as the runner's compile_torch_workload_for_tt_device.
    compiled = torch.compile(model, backend="tt", options=COMPILE_OPTIONS)

    inputs = _inputs_to_device(load_input(), device)

    with torch.no_grad():
        output = compiled(*inputs)

    return output.cpu()


def codegen_model():
    os.environ["XLA_HLO_DEBUG"] = "1"
    xr.set_device_type("TT")
    _enable_spmd()  # same SPMD init as run_tt_model
    device = torch_xla.device()

    mesh = _build_mesh()

    model = load_pytorch_model()
    model = model.to(device, dtype=DATA_FORMAT)
    # CRITICAL: TP sharding must be applied here too, else codegen emits a
    # single-chip graph instead of the sharded one. mark_sharding requires the
    # weights to already be XLA tensors, so this must run after model.to(device).
    _apply_tp_sharding(model, mesh)

    # We can't call tt_torch.codegen_py directly: it filters args/kwargs down to
    # top-level torch.Tensors, which would silently drop FIBO's nested list-of-
    # tensor inputs (caption / text-encoder hidden states) and break the
    # wrapper's strict positional arg-count check. Inline codegen_py's body here
    # but move the model + every input to device ourselves (via tree_map).
    import torch_xla.core.xla_model as xm

    codegen_options = {
        **COMPILE_OPTIONS,
        "backend": "codegen_py",
        "export_path": OUTPUT_DIR,
        "export_tensors": True,
    }
    torch_xla.set_custom_compile_options(codegen_options)
    # tt_legacy_compile mirrors codegen_py: makes MetaDataProp work and avoids
    # codegenning graphs that never execute.
    model.compile(backend="tt", options={"tt_legacy_compile": True})

    inputs = _inputs_to_device(load_input(), device)
    with torch.no_grad():
        model(*inputs)
    xm.wait_device_ops()  # ensure codegen files are fully written


def compare_pytorch_and_tt_runs():
    # Capture exact PCC from first --golden run and paste here.
    exact_pcc = 0.999494

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
        description="FIBO (briaai/FIBO) TP=4 codegen pipeline"
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
