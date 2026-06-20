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
    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=device, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
    activations = [ttnn_input]

    outputs = model(activations)

    ttnn_output = ttnn.to_torch(ttnn.from_device(outputs[0]))
    golden_output = model_pt.run_pytorch_model()

    pcc = calculate_pcc(ttnn_output.to(golden_output.dtype), golden_output)
    print(f"\nPCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"

    batch_size = input_tensor.shape[0]
    print(f"\nPerformance (batch_size={batch_size}):")
    for i in range(3):
        ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=device, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
        activations = [ttnn_input]
        start = time.perf_counter()
        outputs = model(activations)
        ttnn.synchronize_device(device)
        end = time.perf_counter()
        elapsed = end - start
        fps = batch_size / elapsed
        print(f"  Run {i + 1}: {elapsed:.4f}s, FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
