import time

import ttnn
import utils
import torch
import model_pt
from utils import calculate_pcc
from model_ttnn import ModelTTNN

BATCH_SIZE = model_pt.BATCH_SIZE
NUM_TOKENS_PER_SAMPLE = 1
NUM_PERF_RUNS = 3


def test_main():
    exact_pcc = 0.9921875

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

    # Build host-side tensors (not on device) for copy_host_to_device_tensor.
    def build_host_activations():
        host = [to_int32_host(cache_position), to_int32_host(input_ids)]
        host.extend([
            to_int32_host(cache_position),
            to_bf16_tile_host(layers[0].keys),
            to_int32_host(cache_position),
            to_bf16_tile_host(layers[0].values),
            to_int32_host(cache_position),
        ])
        for layer in layers[1:]:
            host.extend([
                to_int32_host(cache_position),
                to_bf16_tile_host(layer.keys),
                to_int32_host(cache_position),
                to_bf16_tile_host(layer.values),
            ])
        return host

    host_activations = build_host_activations()

    # Allocate persistent DRAM tensors that survive across trace runs.
    device_activations = [
        ttnn.allocate_tensor_on_device(
            h.shape, h.dtype, h.layout, device, dram_interleaved
        )
        for h in host_activations
    ]

    def copy_inputs():
        for h, d in zip(host_activations, device_activations):
            ttnn.copy_host_to_device_tensor(h, d, cq_id=0)

    ttnn_model = ModelTTNN(device)

    tokens_per_run = BATCH_SIZE * NUM_TOKENS_PER_SAMPLE
    print(f"\nPerf measurement (5 runs: 1 compile + 1 trace capture + {NUM_PERF_RUNS} trace execute):")

    # Run 1: compile run — populates program cache.
    copy_inputs()
    start = time.perf_counter()
    outputs = ttnn_model(device_activations)
    ttnn.synchronize_device(device)
    end = time.perf_counter()
    elapsed = end - start
    tps = tokens_per_run / elapsed
    print(f"  Run 1 (compile): {elapsed:.4f}s, {tps:.2f} TPS")

    ttnn_output = ttnn.to_torch(ttnn.from_device(outputs[-1]))
    ttnn_output = ttnn_output[:, -1, :]

    golden_output = model_pt.run_pytorch_model()

    pcc = calculate_pcc(ttnn_output, golden_output)
    print(f"\nPCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"

    # Run 2: trace capture.
    copy_inputs()
    start = time.perf_counter()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    trace_outputs = ttnn_model(device_activations)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    end = time.perf_counter()
    elapsed = end - start
    tps = tokens_per_run / elapsed
    print(f"  Run 2 (trace capture): {elapsed:.4f}s, {tps:.2f} TPS")

    # Runs 3-5: trace execute.
    for i in range(NUM_PERF_RUNS):
        copy_inputs()
        start = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        end = time.perf_counter()
        elapsed = end - start
        tps = tokens_per_run / elapsed
        print(f"  Run {i + 3} (trace execute): {elapsed:.4f}s, {tps:.2f} TPS")

    ttnn.release_trace(device, tid)


if __name__ == "__main__":
    main()
