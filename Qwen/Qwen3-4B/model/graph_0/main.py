import time

import ttnn
import model_pt
import utils
from model_ttnn import ModelTTNN
from utils import calculate_pcc


def load_activations_for__main(device):
    utils_load_tensor_0 = utils.load_tensor(
        "./tensors/arg2.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [utils_load_tensor_0]


def main():
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape((1, 1)),
        l1_small_size=1 << 15,
    )
    print(f"Device: {device}")
    try:
        load_activations_for__main_0 = load_activations_for__main(device)
        model = ModelTTNN(device)
        _main_0 = model(load_activations_for__main_0)
    finally:
        ttnn.close_mesh_device(device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return 0


def test_main():
    exact_pcc = 0.9765207171440125

    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape((1, 1)),
        l1_small_size=1 << 15,
    )
    print(f"Device: {device}")
    try:
        torch_input = model_pt.load_input()
        num_tokens = torch_input.numel()

        host_tensor = ttnn.from_torch(torch_input)
        host_tensor = ttnn.to_layout(host_tensor, ttnn.Layout.ROW_MAJOR)
        host_tensor = ttnn.to_dtype(host_tensor, ttnn.DataType.INT32)

        dram_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        )
        input_dram_tensor = ttnn.allocate_tensor_on_device(
            list(torch_input.shape),
            ttnn.DataType.INT32,
            ttnn.Layout.ROW_MAJOR,
            device,
            dram_memory_config,
        )

        model = ModelTTNN(device)

        # Run 1: compile model / fill program cache
        ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=0)
        start = time.perf_counter()
        outputs = model([input_dram_tensor])
        ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - start
        tps = num_tokens / elapsed
        print(f"Iteration 1: Time: {elapsed:.4f}s, TPS: {tps:.2f}")

        # Run 2: capture trace
        ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=0)
        start = time.perf_counter()
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        outputs = model([input_dram_tensor])
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - start
        tps = num_tokens / elapsed
        print(f"Iteration 2: Time: {elapsed:.4f}s, TPS: {tps:.2f}")

        # Runs 3-5: execute trace
        host_output_tensor = None
        for i in range(3):
            ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=0)
            start = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            host_output_tensor = outputs[-1].cpu(blocking=False)
            ttnn.synchronize_device(device)
            elapsed = time.perf_counter() - start
            tps = num_tokens / elapsed
            print(f"Iteration {i + 3}: Time: {elapsed:.4f}s, TPS: {tps:.2f}")

        ttnn_output = ttnn.to_torch(host_output_tensor)
        golden_output = model_pt.run_pytorch_model()

        pcc = calculate_pcc(ttnn_output, golden_output)
        print(f"\nPCC: {pcc:.6f}")
        assert pcc >= exact_pcc, f"PCC {pcc} is not greater than threshold {exact_pcc}"
    finally:
        ttnn.close_mesh_device(device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
