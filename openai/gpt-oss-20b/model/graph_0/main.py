import ttnn
import utils
from utils import calculate_pcc
from model_ttnn import ModelTTNN
import torch


def load_activations_for__main(device):
    utils_load_tensor_0 = utils.load_tensor(
        "./tensors/arg3.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
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
    return 0


def test_main():
    import model_pt
    import time

    exact_pcc = 0.98

    device = utils.open_device()
    model = ModelTTNN(device)

    def to_host_torch(tensor):
        tensor = ttnn.from_device(tensor)
        if device is not None:
            return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
        return ttnn.to_torch(tensor)

    pt_input = model_pt.load_input()
    input_ids_pt = pt_input["input_ids"]
    num_tokens = input_ids_pt.numel()

    def make_ttnn_input():
        t = ttnn.from_torch(
            input_ids_pt, dtype=ttnn.DataType.INT32, layout=ttnn.Layout.ROW_MAJOR
        )
        return ttnn.to_device(
            t,
            device,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

    outputs = model([make_ttnn_input()])

    ttnn_output = to_host_torch(outputs[4])[:, -1]
    golden_output = model_pt.run_pytorch_model()

    pcc = calculate_pcc(ttnn_output.to(torch.float32), golden_output.to(torch.float32))
    print(f"\nPCC: {pcc:.6f}")
    assert pcc > exact_pcc, f"PCC {pcc} is below expected {exact_pcc}"

    print("\n--- Performance ---")
    for i in range(3):
        ttnn_input = make_ttnn_input()
        start = time.perf_counter()
        outputs = model([ttnn_input])
        ttnn.synchronize_device(device)
        end = time.perf_counter()
        elapsed = end - start
        tps = num_tokens / elapsed
        print(f"Run {i}: {elapsed:.4f}s, TPS: {tps:.2f}")


if __name__ == "__main__":
    main()
