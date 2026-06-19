import ttnn
import torch
import utils
import model_pt
from model_ttnn import _main
from utils import calculate_pcc
from params import load_weights_for__main_from_state_dict as load_weights_for__main


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
    load_activations_for__main_0 = load_activations_for__main(device)
    load_weights_for__main_0 = load_weights_for__main()
    _main_0 = _main(device, load_activations_for__main_0, load_weights_for__main_0)
    utils.close_device(device)
    return 0


def test_main():
    exact_pcc = 0.97265625

    device = utils.open_device()
    input_tensor = model_pt.load_input()
    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=device, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
    activations = [ttnn_input]

    weights = load_weights_for__main()
    outputs = _main(device, activations, weights)

    ttnn_output = ttnn.to_torch(ttnn.from_device(outputs[0]))
    golden_output = model_pt.run_pytorch_model()

    pcc = calculate_pcc(ttnn_output.to(golden_output.dtype), golden_output)
    print(f"\nPCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"


if __name__ == "__main__":
    main()
