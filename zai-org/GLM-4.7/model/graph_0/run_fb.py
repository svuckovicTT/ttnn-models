"""Decisive split experiment for the codegen-vs-runtime PCC mismatch.

Runs the runtime flatbuffer (model.ttnn) through the TTNN MLIR runtime using the
SAME saved ./tensors/argN.tensorbin inputs that main.py (codegen) consumes, then
compares its final logits against the CPU PyTorch golden.

  * PCC ~0.85  -> flatbuffer + saved tensors are fine; the 0.013 is EmitPy-execution
                 specific (codegen Python diverges from the runtime). Next: op-by-op.
  * PCC ~0.013 -> the SAVED tensors (weights/activations) are the problem, shared by
                 both paths; not an EmitPy bug. Next: compare saved-vs-live tensors.
"""

import json
import math
import os

import torch
# Bind directly to the C-extension submodules: the ttrt.runtime / ttrt.binary
# __init__.py re-exports get bypassed by namespace-package resolution here.
import ttrt.runtime._ttmlir_runtime.runtime as rt
import ttrt.runtime._ttmlir_runtime.binary as tb
import ttnn

GRAPH_DIR = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(GRAPH_DIR, "model.ttnn")
TENSORS = os.path.join(GRAPH_DIR, "tensors")
PROG = 0
VOCAB = 151552  # last dim of the logits output

_T2D = {
    torch.float32: rt.DataType.Float32,
    torch.bfloat16: rt.DataType.BFloat16,
    torch.int32: rt.DataType.Int32,
    torch.uint32: rt.DataType.UInt32,
    torch.uint16: rt.DataType.UInt16,
    torch.uint8: rt.DataType.UInt8,
}
_D2T = {v: k for k, v in _T2D.items()}


def create_tensor(shards, mesh_shape):
    f = shards[0]
    if len(shards) > 1:
        return rt.create_multi_device_borrowed_host_tensor(
            [t.data_ptr() for t in shards],
            list(f.shape), list(f.stride()), f.element_size(),
            _T2D[f.dtype], {}, mesh_shape,
        )
    return rt.create_borrowed_host_tensor(
        f.data_ptr(), list(f.shape), list(f.stride()), f.element_size(), _T2D[f.dtype]
    )


def to_torch_rt(t):
    buf = bytearray(t.get_data_buffer())
    return torch.frombuffer(buf, dtype=_D2T[t.get_dtype()]).reshape(t.get_shape())


def pcc(x, y):
    x, y = x.flatten().float(), y.flatten().float()
    vx, vy = x - x.mean(), y - y.mean()
    denom = vx.norm() * vy.norm()
    return float("nan") if denom == 0 else float((vx @ vy) / denom)


# --- 1. Load flatbuffer, derive mesh shape + input count
fbb = tb.load_binary_from_path(BIN)
mesh_shape = list(fbb.get_program_mesh_shape(PROG))
num_inputs = len(json.loads(fbb.get_program_inputs_as_json(PROG)))
print(f"mesh_shape={mesh_shape} ({math.prod(mesh_shape)} dev) num_inputs={num_inputs}", flush=True)

# --- 2. Fabric + open mesh (matches codegen utils.py: FABRIC_1D_RING for >1 dev)
rt.set_fabric_config(rt.FabricConfig.FABRIC_1D_RING)
mo = rt.MeshDeviceOptions()
mo.mesh_shape = mesh_shape
device = rt.open_mesh_device(mo)

fb_logits = None
try:
    # --- 3. Build inputs from the saved tensors (same files main.py loads)
    keep = []  # keep torch refs alive: borrowed host tensors don't own their data
    inputs = []
    for i in range(num_inputs):
        t = ttnn.load_tensor(os.path.join(TENSORS, f"arg{i}.tensorbin"))
        if t.layout != ttnn.Layout.ROW_MAJOR:
            t = ttnn.to_layout(t, ttnn.Layout.ROW_MAJOR)
        shards = [ttnn.to_torch(s).contiguous() for s in ttnn.get_device_tensors(t)]
        keep.extend(shards)
        inputs.append(create_tensor(shards, mesh_shape))
        if i < 3 or i >= num_inputs - 3:
            print(f"  arg{i}: nshards={len(shards)} local={tuple(shards[0].shape)} "
                  f"dtype={shards[0].dtype}", flush=True)

    inputs = [rt.to_layout(inp, device, rt.get_layout(fbb, PROG, i), True)
              for i, inp in enumerate(inputs)]

    # --- 4. Run
    print("submitting...", flush=True)
    outs = rt.submit(device, fbb, PROG, inputs)
    rt.wait(outs)

    for i, o in enumerate(outs):
        host = rt.to_host(o, untilize=True)
        sh = [to_torch_rt(s) for s in host]
        shp = tuple(sh[0].shape)
        print(f"  out{i}: nshards={len(sh)} shard_shape={shp} dtype={sh[0].dtype}", flush=True)
        if shp and shp[-1] == VOCAB:  # logits output (replicated -> shard 0 is full)
            fb_logits = sh[0].float().clone()
finally:
    rt.close_mesh_device(device)
    rt.set_fabric_config(rt.FabricConfig.DISABLED)

assert fb_logits is not None, "no logits output (vocab dim) found among outputs"
torch.save(fb_logits, "/tmp/fb_logits.pt")
print(f"\nfb_logits: shape={tuple(fb_logits.shape)}  -> saved /tmp/fb_logits.pt", flush=True)

# --- 5. CPU golden (pure torch) + PCC
print("=== computing CPU golden (model_pt.run_pytorch_model) ===", flush=True)
import model_pt  # noqa: E402

golden = model_pt.run_pytorch_model().float().cpu()
torch.save(golden, "/tmp/golden_logits.pt")
print(f"golden: shape={tuple(golden.shape)}  -> saved /tmp/golden_logits.pt")

if fb_logits.shape == golden.shape:
    print(f"\n>>> PCC flatbuffer-vs-golden = {pcc(fb_logits, golden):.6f}")
    print(">>> (codegen-vs-golden = 0.013; runtime-vs-golden via XLA = 0.85)")
else:
    print(f"\n!!! shape mismatch fb={tuple(fb_logits.shape)} golden={tuple(golden.shape)}")
