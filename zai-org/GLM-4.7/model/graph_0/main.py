import math

import torch
import ttnn
import utils
from utils import calculate_pcc

from model_ttnn import ModelTTNN

MESH_SHAPE = (4, 8)
L1_SMALL_SIZE = 1 << 15

_decode_state = None


def _decode_inputs():
    """Load the PyTorch model and run a CPU prefill to reach the decode state
    (mirrors xla.py, which produced the serialized activations). Cached so the
    expensive prefill runs once even if activations are requested repeatedly."""
    global _decode_state
    if _decode_state is None:
        import model_pt

        model = model_pt.load_pytorch_model()
        _decode_state = model_pt._cpu_prefill_to_decode_state(model)
    return _decode_state


def open_device():
    if math.prod(MESH_SHAPE) >= 2:
        # ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(MESH_SHAPE),
        l1_small_size=L1_SMALL_SIZE,
    )
    return device


def close_device(device):
    ttnn.close_mesh_device(device)
    if math.prod(MESH_SHAPE) >= 2:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def load_activations_for__main(device):
    """Construct the decode-step inputs the way xla.py did when it produced the
    serialized activation tensorbins: load the PyTorch model, run a CPU prefill to
    populate the StaticCache, and snapshot the resulting decode state (next-token
    input_ids, advanced cache_position, populated per-layer KV caches), then
    distribute them across the mesh. Replaces loading the serialized tensors from
    disk."""
    decode_args = _decode_inputs()
    input_ids = decode_args["input_ids"]
    cache_position = decode_args["cache_position"]
    layers = decode_args["past_key_values"].layers
    # cumulative_length == number of prefilled tokens == the decode cache position.
    cumulative_length = int(cache_position.reshape(-1)[0].item())

    dram = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    )

    def to_device_tensor(torch_tensor, dtype, layout, mesh_mapper):
        ttnn_tensor = ttnn.from_torch(torch_tensor, mesh_mapper=mesh_mapper)
        ttnn_tensor = ttnn.to_layout(ttnn_tensor, layout)
        ttnn_tensor = ttnn.to_dtype(ttnn_tensor, dtype)
        return ttnn.to_device(ttnn_tensor, device, dram)

    # input_ids and the KV caches are sharded along the mesh "batch" axis (the 4
    # rows); cache_position and the per-layer cumulative_length are replicated.
    def batch_sharded():
        return ttnn.ShardTensor2dMesh(device, MESH_SHAPE, (0, None))

    def replicated():
        return ttnn.ReplicateTensorToMesh(device)

    activations = [
        to_device_tensor(
            input_ids.to(torch.int32),
            ttnn.DataType.INT32,
            ttnn.Layout.TILE,
            batch_sharded(),
        ),
        to_device_tensor(
            cache_position.to(torch.int32),
            ttnn.DataType.INT32,
            ttnn.Layout.ROW_MAJOR,
            replicated(),
        ),
    ]
    for layer in layers:
        activations.append(
            to_device_tensor(
                torch.tensor([cumulative_length], dtype=torch.int32),
                ttnn.DataType.INT32,
                ttnn.Layout.ROW_MAJOR,
                replicated(),
            )
        )
        activations.append(
            to_device_tensor(
                layer.keys.to(torch.bfloat16),
                ttnn.DataType.BFLOAT16,
                ttnn.Layout.TILE,
                batch_sharded(),
            )
        )
        activations.append(
            to_device_tensor(
                layer.values.to(torch.bfloat16),
                ttnn.DataType.BFLOAT16,
                ttnn.Layout.TILE,
                batch_sharded(),
            )
        )
    return activations


def main():
    device = open_device()
    load_activations_for__main_0 = load_activations_for__main(device)
    model = ModelTTNN(device)
    _main_0 = model(load_activations_for__main_0)
    close_device(device)
    return 0


def test_main():
    import time
    import model_pt

    exact_pcc = 0.85546875

    device = open_device()
    activations = load_activations_for__main(device)
    model = ModelTTNN(device)
    outputs = model(activations)

    ttnn_output = [ttnn.from_device(output) for output in outputs]
    golden_output = model_pt.run_pytorch_model()

    # outputs[-1] is the final logits, fully replicated across the 4x8 mesh by
    # the trailing all_gathers (dim 0 over cluster_axis 0, dim 2 over cluster_axis 1).
    # Every device holds an identical copy, so grab a single shard before converting.
    final_output = ttnn.get_device_tensors(ttnn_output[-1])[0]

    pcc = calculate_pcc(ttnn.to_torch(final_output), golden_output)
    print(f"\nPCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"

    for i in range(3):
        activations = load_activations_for__main(device)
        num_tokens = math.prod(activations[0].shape)
        start = time.perf_counter()
        outputs = model(activations)
        ttnn.synchronize_device(device)
        end = time.perf_counter()
        elapsed = end - start
        tps = num_tokens / elapsed
        print(f"Run {i}: {elapsed:.4f}s, TPS: {tps:.2f}")

    close_device(device)


if __name__ == "__main__":
    main()
