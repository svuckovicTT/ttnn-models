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
        load_activations_for__main_0 = load_activations_for__main(device)
        model = ModelTTNN(device)
        _main_0 = model(load_activations_for__main_0)

        ttnn_output = ttnn.to_torch(_main_0[0]).reshape(1, 1000).to(torch.float32)
        golden_output = model_pt.main().logits.reshape(1, 1000).to(torch.float32)

        pcc = calculate_pcc(ttnn_output, golden_output)
        print(f"\nPCC: {pcc:.6f}")
        assert pcc >= pcc_threshold, f"PCC {pcc} is below threshold of {pcc_threshold}"
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
