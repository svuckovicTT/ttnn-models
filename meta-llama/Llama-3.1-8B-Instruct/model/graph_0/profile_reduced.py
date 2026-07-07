import time

try:
    from tracy import signpost
except ImportError:
    def signpost(_message):
        pass

import ttnn
import utils
import model_pt
from model_ttnn import ModelTTNN


BATCH_SIZE = model_pt.BATCH_SIZE
NUM_TOKENS_PER_SAMPLE = 1


def main():
    model = model_pt.load_pytorch_model()
    pytorch_input = model_pt.load_input(model)

    device = utils.open_device()
    dram_interleaved = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    )

    def to_int32_host(t):
        return ttnn.from_torch(t, dtype=ttnn.DataType.INT32, layout=ttnn.Layout.ROW_MAJOR)

    def to_bf16_tile_host(t):
        return ttnn.from_torch(t, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE)

    cache_position = pytorch_input["cache_position"]
    input_ids = pytorch_input["input_ids"]
    layers = pytorch_input["past_key_values"].layers

    host_activations = [to_int32_host(cache_position), to_int32_host(input_ids)]
    host_activations.extend(
        [
            to_int32_host(cache_position),
            to_bf16_tile_host(layers[0].keys),
            to_int32_host(cache_position),
            to_bf16_tile_host(layers[0].values),
            to_int32_host(cache_position),
        ]
    )
    for layer in layers[1:]:
        host_activations.extend(
            [
                to_int32_host(cache_position),
                to_bf16_tile_host(layer.keys),
                to_int32_host(cache_position),
                to_bf16_tile_host(layer.values),
            ]
        )

    device_activations = [
        ttnn.allocate_tensor_on_device(
            h.shape, h.dtype, h.layout, device, dram_interleaved
        )
        for h in host_activations
    ]

    def copy_inputs():
        for h, d in zip(host_activations, device_activations):
            ttnn.copy_host_to_device_tensor(h, d, cq_id=0)

    ttnn_model = ModelTTNN(device, num_layers=1)
    tokens_per_run = BATCH_SIZE * NUM_TOKENS_PER_SAMPLE

    copy_inputs()
    start = time.perf_counter()
    _outputs = ttnn_model(device_activations)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - start
    print(f"Reduced compile: {elapsed:.4f}s, {tokens_per_run / elapsed:.2f} TPS")

    copy_inputs()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    _trace_outputs = ttnn_model(device_activations)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)

    copy_inputs()
    signpost("PERF_TRACE_DECODE")
    start = time.perf_counter()
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - start
    signpost("PERF_TRACE_DECODE_END")
    print(f"Reduced trace execute: {elapsed:.4f}s, {tokens_per_run / elapsed:.2f} TPS")

    ttnn.release_trace(device, tid)


if __name__ == "__main__":
    main()
