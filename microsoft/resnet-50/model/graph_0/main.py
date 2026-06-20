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


def main():
    device = utils.open_device()
    model = ModelTTNN(device)
    load_activations_for__main_0 = load_activations_for__main(device)
    _main_0 = model(load_activations_for__main_0)
    utils.close_device(device)
    return 0


def test_main():
    exact_pcc = 0.97265625

    device = utils.open_device()
    model = ModelTTNN(device)
    input_tensor = model_pt.load_input()
    dram_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=device, memory_config=dram_mem_config)
    activations = [ttnn_input]

    outputs = model(activations)

    ttnn_output = ttnn.to_torch(ttnn.from_device(outputs[0]))
    golden_output = model_pt.run_pytorch_model()

    pcc = calculate_pcc(ttnn_output.to(golden_output.dtype), golden_output)
    print(f"\nPCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"

    batch_size = input_tensor.shape[0]
    host_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR)
    input_dram_tensor = ttnn.allocate_tensor_on_device(host_tensor.shape, host_tensor.dtype, host_tensor.layout, device, dram_mem_config)

    print(f"\nPerformance with metal trace (batch_size={batch_size}):")

    # Run 1: compile ops and fill program cache
    ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=0)
    start = time.perf_counter()
    output_tensor = model([input_dram_tensor])
    ttnn.synchronize_device(device)
    end = time.perf_counter()
    elapsed = end - start
    fps = batch_size / elapsed
    print(f"  Run 1 (compile): {elapsed:.4f}s, FPS: {fps:.2f}")

    # Run 2: capture trace
    ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=0)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    start = time.perf_counter()
    output_tensor = model([input_dram_tensor])
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    end = time.perf_counter()
    elapsed = end - start
    fps = batch_size / elapsed
    print(f"  Run 2 (capture): {elapsed:.4f}s, FPS: {fps:.2f}")

    # Runs 3-5: execute trace
    for i in range(3):
        ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=0)
        start = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        host_output_tensor = output_tensor[0].cpu(blocking=False)
        ttnn.synchronize_device(device)
        end = time.perf_counter()
        elapsed = end - start
        fps = batch_size / elapsed
        print(f"  Run {i + 3} (trace): {elapsed:.4f}s, FPS: {fps:.2f}")

    ttnn.release_trace(device, tid)


if __name__ == "__main__":
    main()
