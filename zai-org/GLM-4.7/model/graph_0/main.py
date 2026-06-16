import math

import ttnn
import utils
from utils import calculate_pcc

from model_ttnn import ModelTTNN

MESH_SHAPE = (4, 8)
L1_SMALL_SIZE = 1 << 15


def open_device():
    if math.prod(MESH_SHAPE) >= 2:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
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
    utils_load_tensor_0 = utils.load_tensor(
        "./tensors/arg4.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_1 = utils.load_tensor(
        "./tensors/arg6.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_2 = utils.load_tensor(
        "./tensors/arg8.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_3 = utils.load_tensor(
        "./tensors/arg9.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_4 = utils.load_tensor(
        "./tensors/arg12.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_5 = utils.load_tensor(
        "./tensors/arg25.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_6 = utils.load_tensor(
        "./tensors/arg26.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_7 = utils.load_tensor(
        "./tensors/arg29.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_8 = utils.load_tensor(
        "./tensors/arg42.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_9 = utils.load_tensor(
        "./tensors/arg43.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_10 = utils.load_tensor(
        "./tensors/arg46.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_11 = utils.load_tensor(
        "./tensors/arg59.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_12 = utils.load_tensor(
        "./tensors/arg60.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_13 = utils.load_tensor(
        "./tensors/arg63.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [
        utils_load_tensor_0,
        utils_load_tensor_1,
        utils_load_tensor_2,
        utils_load_tensor_3,
        utils_load_tensor_4,
        utils_load_tensor_5,
        utils_load_tensor_6,
        utils_load_tensor_7,
        utils_load_tensor_8,
        utils_load_tensor_9,
        utils_load_tensor_10,
        utils_load_tensor_11,
        utils_load_tensor_12,
        utils_load_tensor_13,
    ]



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
    model = ModelTTNN(device)

    activations = load_activations_for__main(device)
    host_activations = [ttnn.from_device(a) for a in activations]

    input_dram_tensors = []
    for a in activations:
        input_dram_tensors.append(
            ttnn.allocate_tensor_on_device(
                a.shape, a.dtype, a.layout, device, a.memory_config()
            )
        )
    for a in activations:
        ttnn.deallocate(a, False)

    num_tokens = math.prod(host_activations[0].shape)

    for h, d in zip(host_activations, input_dram_tensors):
        ttnn.copy_host_to_device_tensor(h, d, cq_id=0)
    start = time.perf_counter()
    output_tensors = model(input_dram_tensors)
    ttnn.synchronize_device(device)
    end = time.perf_counter()
    elapsed = end - start
    tps = num_tokens / elapsed
    print(f"Run 0 (compile): {elapsed:.4f}s, TPS: {tps:.2f}")

    ttnn_output = [ttnn.from_device(output) for output in output_tensors]
    golden_output = model_pt.run_pytorch_model()

    # outputs[-1] is the final logits, fully replicated across the 4x8 mesh by
    # the trailing all_gathers (dim 0 over cluster_axis 0, dim 2 over cluster_axis 1).
    # Every device holds an identical copy, so grab a single shard before converting.
    final_output = ttnn.get_device_tensors(ttnn_output[-1])[0]

    pcc = calculate_pcc(ttnn.to_torch(final_output), golden_output)
    print(f"\nPCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"

    for h, d in zip(host_activations, input_dram_tensors):
        ttnn.copy_host_to_device_tensor(h, d, cq_id=0)
    start = time.perf_counter()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensors = model(input_dram_tensors)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    end = time.perf_counter()
    elapsed = end - start
    tps = num_tokens / elapsed
    print(f"Run 1 (trace capture): {elapsed:.4f}s, TPS: {tps:.2f}")

    for i in range(3):
        for h, d in zip(host_activations, input_dram_tensors):
            ttnn.copy_host_to_device_tensor(h, d, cq_id=0)
        start = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        host_output = output_tensors[-1].cpu(blocking=False)
        ttnn.synchronize_device(device)
        end = time.perf_counter()
        elapsed = end - start
        tps = num_tokens / elapsed
        print(f"Run {i + 2} (trace execute): {elapsed:.4f}s, TPS: {tps:.2f}")

    ttnn.release_trace(device, tid)
    close_device(device)


if __name__ == "__main__":
    main()
