import ttnn
import utils
from utils import calculate_pcc
from params import load_weights_for__main
from model_ttnn import _main
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
    load_activations_for__main_0 = load_activations_for__main(device)
    load_weights_for__main_0 = load_weights_for__main(device)
    _main_0 = _main(load_activations_for__main_0, load_weights_for__main_0, device)
    return 0


def test_main():
    import model_pt

    exact_pcc = 0.98

    device = utils.open_device()

    def to_host_torch(tensor):
        tensor = ttnn.from_device(tensor)
        if device is not None:
            return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
        return ttnn.to_torch(tensor)

    pt_input = model_pt.load_input()
    ttnn_input = ttnn.from_torch(
        pt_input["input_ids"], dtype=ttnn.DataType.INT32, layout=ttnn.Layout.ROW_MAJOR
    )
    ttnn_input = ttnn.to_device(
        ttnn_input,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )

    weights = load_weights_for__main(device)
    outputs = _main([ttnn_input], weights, device)

    ttnn_output = to_host_torch(outputs[4])[:, -1]
    golden_output = model_pt.run_pytorch_model()

    pcc = calculate_pcc(ttnn_output.to(torch.float32), golden_output.to(torch.float32))
    print(f"\nPCC: {pcc:.6f}")
    assert pcc > exact_pcc, f"PCC {pcc} is below expected {exact_pcc}"


if __name__ == "__main__":
    main()
