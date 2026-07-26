# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn


def open_device():
    mesh_shape = (1, 4)
    fabric_config = ttnn.FabricConfig.FABRIC_1D_RING
    ttnn.set_fabric_config(fabric_config)
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(mesh_shape),
        l1_small_size=1 << 15,
    )
    print(f"Device: {device}")
    return device


def get_scalar_from_tensor(tensor: ttnn.Tensor) -> int:
    assert tensor.logical_volume() == 1, "expected scalar tensor"
    assert tensor.dtype == ttnn.DataType.UINT32, "expected uint32 tensor"

    host_tensor = ttnn.from_device(tensor)
    return host_tensor.item()


def load_tensor(file_path: str, layout, dtype, device, memory_config) -> ttnn.Tensor:
    loaded_tensor = ttnn.load_tensor(file_path)

    assert loaded_tensor.device() is None, "loaded tensor must be on host"

    if loaded_tensor.layout != layout:
        loaded_tensor = ttnn.to_layout(loaded_tensor, layout)
    if loaded_tensor.dtype != dtype:
        loaded_tensor = ttnn.to_dtype(loaded_tensor, dtype)
    if device is not None:
        loaded_tensor = ttnn.to_device(loaded_tensor, device, memory_config)

    return loaded_tensor


# Heavy-lifting helper for CPU-hoisted functions. Mirrors the runtime logic in
# runtime/lib/ttnn/operations/cpu/cpu.cpp (runSingleChip / runMultiChip):
# CPU-hoisted segments are barrier-free local compute, so each device's shard is
# computed independently on the host and the per-shard results are reassembled
# into a multi-device tensor.
def execute_cpu_hoisted_function(inputs, function, mesh_device):
    """Run a pure-torch CPU-hoisted body shard-by-shard over a mesh.

    inputs:      list of ttnn.Tensor operands (device-resident, possibly sharded).
    function:    pure-torch callable mapping torch tensors -> torch tensor(s).
    mesh_device: the mesh device handle (or None for host-only execution).
    Returns a single ttnn.Tensor, or a tuple of them for multi-output bodies.
    """

    def _wrap_outputs(result):
        return result if isinstance(result, (list, tuple)) else (result,)

    # No mesh context: run the body once on the host and return host tensor(s).
    if mesh_device is None:
        torch_inputs = [ttnn.to_torch(tensor) for tensor in inputs]
        outputs = _wrap_outputs(function(*torch_inputs))
        host_outputs = [ttnn.from_torch(out) for out in outputs]
        return host_outputs[0] if len(host_outputs) == 1 else tuple(host_outputs)

    mesh_shape = mesh_device.shape
    num_shards = mesh_device.get_num_devices()

    # Split each input into per-device torch shards. get_device_tensors returns
    # one shard per device for a sharded tensor, or a single shard for an
    # unsharded (replicated) tensor, which is then reused across devices.
    input_shards = []
    for tensor in inputs:
        if tensor.device() is not None:
            tensor = ttnn.from_device(tensor)
        shards = ttnn.get_device_tensors(tensor)
        input_shards.append([ttnn.to_torch(shard) for shard in shards])

    # Run the body once per device shard.
    output_shards = []
    for shard_idx in range(num_shards):
        args = [
            shards[shard_idx] if len(shards) > 1 else shards[0]
            for shards in input_shards
        ]
        output_shards.append(_wrap_outputs(function(*args)))

    # Reassemble each output across shards into a multi-device host tensor.
    num_outputs = len(output_shards[0])
    results = []
    for out_idx in range(num_outputs):
        torch_shards = [output_shards[s][out_idx] for s in range(num_shards)]
        ttnn_shards = [ttnn.from_torch(shard) for shard in torch_shards]
        results.append(ttnn.from_host_shards(ttnn_shards, mesh_shape))
    return results[0] if num_outputs == 1 else tuple(results)


# Helpers for distributed RMS norm EmitPy support.
# These mirror the runtime logic in
# runtime/lib/ttnn/operations/normalization/distributed_rms_norm.cpp
# TODO(jserbedzija): Remove this once the following issue if fixed in tt-metal: https://github.com/tenstorrent/tt-metal/issues/37746
def create_global_semaphore(input_tensor):
    """Create a global semaphore from the input tensor's device and shard grid."""
    mesh_device = input_tensor.device()
    shard_spec = input_tensor.memory_config().shard_spec
    return ttnn.create_global_semaphore(mesh_device, shard_spec.grid, 0)


def calculate_pcc(x, y):
    # This function calculates the PCC between two torch tensors

    # Assert both are torch tensors
    assert isinstance(x, torch.Tensor), "x must be a torch tensor"
    assert isinstance(y, torch.Tensor), "y must be a torch tensor"

    if x.shape != y.shape:
        raise ValueError(
            f"Shapes of x and y must be the same, but got {x.shape} and {y.shape}"
        )

    # Calculate PCC
    x_flat, y_flat = x.flatten(), y.flatten()
    vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()

    return float("nan") if denom == 0 else ((vx @ vy) / denom).item()
