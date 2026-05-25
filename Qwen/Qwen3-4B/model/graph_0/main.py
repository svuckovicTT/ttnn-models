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
        input = model_pt.load_input()
        input = ttnn.from_torch(input)
        input = ttnn.to_layout(input, ttnn.Layout.ROW_MAJOR)
        input = ttnn.to_dtype(input, ttnn.DataType.INT32)
        input = ttnn.to_device(
            input,
            device,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        model = ModelTTNN(device)
        outputs = model([input])

        ttnn_output = ttnn.to_torch(ttnn.from_device(outputs[-1]))
        golden_output = model_pt.run_pytorch_model()

        pcc = calculate_pcc(ttnn_output, golden_output)
        print(f"\nPCC: {pcc:.6f}")
        assert pcc >= exact_pcc, f"PCC {pcc} is not greater than threshold {exact_pcc}"
    finally:
        ttnn.close_mesh_device(device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
