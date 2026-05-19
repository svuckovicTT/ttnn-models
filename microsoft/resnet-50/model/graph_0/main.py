import time

import ttnn
import torch
import utils
import model_pt
from model_ttnn import ModelTTNN
from utils import calculate_pcc


def load_activations_for__main(device):
    utils_load_tensor_0 = utils.load_tensor(
        "./tensors/arg27.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [utils_load_tensor_0]


def open_device():
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape((1, 1)),
        l1_small_size=1 << 15,
    )
    print(f"Device: {device}")
    return device


def main():
    device = open_device()
    try:
        load_activations_for__main_0 = load_activations_for__main(device)
        model = ModelTTNN(device)
        _main_0 = model(load_activations_for__main_0)
        return 0
    finally:
        ttnn.close_mesh_device(device)


def test_main():
    pcc_threshold = 0.98

    device = open_device()
    try:
        model = ModelTTNN(device)

        host_tensor = utils.load_tensor(
            "./tensors/arg27.tensorbin",
            ttnn.Layout.ROW_MAJOR,
            ttnn.DataType.BFLOAT16,
            None,
            None,
        )
        batch_size = host_tensor.shape[0]

        dram_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        )
        input_dram_tensor = ttnn.allocate_tensor_on_device(
            host_tensor.shape,
            ttnn.DataType.BFLOAT16,
            ttnn.Layout.ROW_MAJOR,
            device,
            dram_mem_config,
        )

        # Iteration 1: compile ops and fill program cache
        start_time = time.perf_counter()
        ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=0)
        output_tensor = model([input_dram_tensor])
        ttnn.synchronize_device(device)
        end_time = time.perf_counter()
        time_taken = end_time - start_time
        fps = batch_size / time_taken
        print(f"Iteration 1: Time taken: {time_taken:.4f}s, FPS: {fps:.2f}")

        # Iteration 2: capture trace
        start_time = time.perf_counter()
        ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=0)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        output_tensor = model([input_dram_tensor])
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        end_time = time.perf_counter()
        time_taken = end_time - start_time
        fps = batch_size / time_taken
        print(f"Iteration 2: Time taken: {time_taken:.4f}s, FPS: {fps:.2f}")

        # Iterations 3-5: execute trace
        for i in range(3):
            start_time = time.perf_counter()
            ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=0)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            host_output_tensor = output_tensor[0].cpu(blocking=False)
            ttnn.synchronize_device(device)
            end_time = time.perf_counter()
            time_taken = end_time - start_time
            fps = batch_size / time_taken
            print(f"Iteration {i + 3}: Time taken: {time_taken:.4f}s, FPS: {fps:.2f}")

        ttnn_output = ttnn.to_torch(host_output_tensor).reshape(1, 1000).to(torch.float32)
        golden_output = model_pt.main().logits.reshape(1, 1000).to(torch.float32)

        pcc = calculate_pcc(ttnn_output, golden_output)
        print(f"\nPCC: {pcc:.6f}")
        assert pcc >= pcc_threshold, f"PCC {pcc} is below threshold of {pcc_threshold}"
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
