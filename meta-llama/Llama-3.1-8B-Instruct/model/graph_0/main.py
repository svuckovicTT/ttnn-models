import ttnn
import utils
import torch
import model_pt
from utils import calculate_pcc
from model_ttnn import ModelTTNN


def test_main():
    exact_pcc = 0.9921875

    model = model_pt.load_pytorch_model()
    pytorch_input = model_pt.load_input(model)

    device = utils.open_device()
    dram_interleaved = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    )

    def to_int32(t):
        return ttnn.to_device(
            ttnn.from_torch(t, dtype=ttnn.DataType.INT32, layout=ttnn.Layout.ROW_MAJOR),
            device,
            dram_interleaved,
        )

    def to_bf16_tile(t):
        return ttnn.to_device(
            ttnn.from_torch(t, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE),
            device,
            dram_interleaved,
        )

    cache_position = pytorch_input["cache_position"]
    input_ids = pytorch_input["input_ids"]
    layers = pytorch_input["past_key_values"].layers

    # Build activations in the same order as load_activations_for__main():
    # [0] cache_position, [1] input_ids
    activations = [to_int32(cache_position), to_int32(input_ids)]
    # Layer 0: cp, keys, cp, values, cp (extra for attention mask)
    activations.extend([
        to_int32(cache_position),
        to_bf16_tile(layers[0].keys),
        to_int32(cache_position),
        to_bf16_tile(layers[0].values),
        to_int32(cache_position),
    ])
    # Layers 1-31: cp, keys, cp, values
    for layer in layers[1:]:
        activations.extend([
            to_int32(cache_position),
            to_bf16_tile(layer.keys),
            to_int32(cache_position),
            to_bf16_tile(layer.values),
        ])

    model = ModelTTNN(device)
    outputs = model(activations)

    ttnn_output = ttnn.to_torch(ttnn.from_device(outputs[-1]))
    ttnn_output = ttnn_output[:, -1, :]

    golden_output = model_pt.run_pytorch_model()

    pcc = calculate_pcc(ttnn_output, golden_output)
    print(f"\nPCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"


if __name__ == "__main__":
    main()
