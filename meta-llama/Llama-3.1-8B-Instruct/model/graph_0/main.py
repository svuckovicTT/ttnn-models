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

    cache_position = pytorch_input["cache_position"]
    input_ids = pytorch_input["input_ids"]
    layers = pytorch_input["past_key_values"].layers

    # Build host-side TTNN tensors (not on device)
    def to_host_int32(t):
        return ttnn.from_torch(t, dtype=ttnn.DataType.INT32, layout=ttnn.Layout.ROW_MAJOR)

    def to_host_bf16_tile(t):
        return ttnn.from_torch(t, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE)

    # Same order as the original build_activations:
    # [0] cache_position, [1] input_ids
    host_tensors = [to_host_int32(cache_position), to_host_int32(input_ids)]
    # Layer 0: cp, keys, cp, values, cp (extra for attention mask)
    host_tensors.extend([
        to_host_int32(cache_position),
        to_host_bf16_tile(layers[0].keys),
        to_host_int32(cache_position),
        to_host_bf16_tile(layers[0].values),
        to_host_int32(cache_position),
    ])
    # Layers 1-31: cp, keys, cp, values
    for layer in layers[1:]:
        host_tensors.extend([
            to_host_int32(cache_position),
            to_host_bf16_tile(layer.keys),
            to_host_int32(cache_position),
            to_host_bf16_tile(layer.values),
        ])

    # Allocate persistent DRAM input buffers
    device_tensors = [
        ttnn.allocate_tensor_on_device(
            ht.shape, ht.dtype, ht.layout, device, dram_interleaved
        )
        for ht in host_tensors
    ]

    def copy_inputs():
        for ht, dt in zip(host_tensors, device_tensors):
            ttnn.copy_host_to_device_tensor(ht, dt, cq_id=0)

    ttnn_model = ModelTTNN(device)
    tokens_per_run = BATCH_SIZE * NUM_TOKENS_PER_SAMPLE

    # Run 1: Compile run (fills program cache)
    copy_inputs()
    start = time.perf_counter()
    outputs = ttnn_model(device_tensors)
    ttnn.synchronize_device(device)
    end = time.perf_counter()
    elapsed = end - start
    tps = tokens_per_run / elapsed
    print(f"  Compile: {elapsed:.4f}s, {tps:.2f} TPS")

    # PCC check on compile run output
    ttnn_output = ttnn.to_torch(ttnn.from_device(outputs[-1]))
    ttnn_output = ttnn_output[:, -1, :]

    golden_output = model_pt.run_pytorch_model()

    pcc = calculate_pcc(ttnn_output, golden_output)
    print(f"\nPCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"

    # Run 2: Trace capture
    copy_inputs()
    start = time.perf_counter()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    outputs = ttnn_model(device_tensors)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    end = time.perf_counter()
    elapsed = end - start
    tps = tokens_per_run / elapsed
    print(f"  Trace capture: {elapsed:.4f}s, {tps:.2f} TPS")

    # Runs 3-5: Trace execution
    print(f"\nPerf measurement ({NUM_PERF_RUNS} runs):")
    for i in range(NUM_PERF_RUNS):
        copy_inputs()
        start = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs[-1].cpu(blocking=False)
        ttnn.synchronize_device(device)
        end = time.perf_counter()
        elapsed = end - start
        tps = tokens_per_run / elapsed
        print(f"  Run {i + 1}: {elapsed:.4f}s, {tps:.2f} TPS")


if __name__ == "__main__":
    main()
