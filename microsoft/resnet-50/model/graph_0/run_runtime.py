"""Run the flatbuffer with REAL inputs (loaded from ./tensors/argN.tensorbin)
via the tt-mlir runtime Python API.

This avoids ttrt run's default randn initialization, which makes the runtime
path comparable to the codegen path.

Saves output to /tmp/runtime_real_final.tensorbin and torch tensor to /tmp/runtime_real_final.pt.
"""
import os
import sys
import torch
import ttnn
import ttrt.binary
import ttrt.runtime
from ttrt.common.util import Binary, convert_runtime_to_torch_tensor

GRAPH_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_PATH = os.path.join(GRAPH_DIR, "model.ttnn")
TENSORS_DIR = os.path.join(GRAPH_DIR, "tensors")

print(f"Loading binary {BIN_PATH}")
fbb = ttrt.binary.load_binary_from_path(BIN_PATH)
program_idx = 0
program_info = ttrt.binary.program_inputs_as_dict(fbb, program_idx)
num_inputs = len(program_info)
print(f"Program {program_idx} has {num_inputs} inputs")

print("Opening mesh device")
opts = ttrt.runtime.MeshDeviceOptions()
device = ttrt.runtime.open_mesh_device(opts)

print("Loading input tensorbins and creating runtime tensors")
torch_inputs = []  # keep refs to avoid GC of borrowed memory
runtime_inputs = []
for i in range(num_inputs):
    path = os.path.join(TENSORS_DIR, f"arg{i}.tensorbin")
    t = ttnn.load_tensor(path)
    # Convert to row-major if needed and bring to host
    if t.layout != ttnn.Layout.ROW_MAJOR:
        t = ttnn.to_layout(t, ttnn.Layout.ROW_MAJOR)
    torch_t = ttnn.to_torch(t).contiguous()
    torch_inputs.append(torch_t)
    rt_t = ttrt.runtime.create_borrowed_host_tensor(
        torch_t.data_ptr(),
        list(torch_t.shape),
        list(torch_t.stride()),
        torch_t.element_size(),
        Binary.Program.to_data_type(torch_t.dtype),
    )
    runtime_inputs.append(rt_t)
    if i % 50 == 0:
        print(f"  loaded arg{i}: shape={tuple(torch_t.shape)} dtype={torch_t.dtype}")

print("Converting input layouts to flatbuffer-expected layouts")
converted = []
for i in range(num_inputs):
    layout = ttrt.runtime.get_layout(fbb, program_idx, i)
    converted.append(ttrt.runtime.to_layout(runtime_inputs[i], device, layout, True))
runtime_inputs = converted

print("Submitting...")
outputs = ttrt.runtime.submit(device, fbb, program_idx, runtime_inputs)
ttrt.runtime.wait(outputs)

print(f"Got {len(outputs)} outputs")
for i, rt_out in enumerate(outputs):
    out_host = ttrt.runtime.to_host(rt_out, untilize=True)
    print(f"  output {i}: num_shards={len(out_host)}")
    torch_out = convert_runtime_to_torch_tensor(out_host[0])
    print(f"    shape={tuple(torch_out.shape)} dtype={torch_out.dtype} "
          f"min={torch_out.float().min().item():.4g} max={torch_out.float().max().item():.4g}")
    # Save
    torch.save(torch_out, os.path.join(GRAPH_DIR, f"runtime_real_output_{i}.pt"))

print("Closing mesh device")
ttrt.runtime.close_mesh_device(device)
