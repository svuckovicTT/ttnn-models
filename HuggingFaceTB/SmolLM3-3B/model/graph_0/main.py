import time

import ttnn
import utils
import model_pt
from utils import calculate_pcc
from model_ttnn import ModelTTNN

# ttnn registers a top-level module also named "activations", which shadows the
# sibling activations.py in this directory on sys.path. Load our file directly by
# path so the correct module is used without disturbing ttnn's.
import importlib.util as _importlib_util
from pathlib import Path as _Path

_activations_spec = _importlib_util.spec_from_file_location(
    "graph_activations", _Path(__file__).resolve().parent / "activations.py"
)
_activations = _importlib_util.module_from_spec(_activations_spec)
_activations_spec.loader.exec_module(_activations)
load_inputs = _activations.load_inputs


def main():
    device = utils.open_device()
    try:
        model = ModelTTNN(device)
        load_inputs_0 = load_inputs(device)
        _main_0 = model(load_inputs_0)
    finally:
        utils.close_device(device)
    return 0


def test_main():
    exact_pcc = 0.984375

    # This is an LLM, so report throughput in tokens/second (TPS).
    num_tokens = model_pt.BATCH_SIZE * model_pt.CONTEXT_LENGTH

    device = utils.open_device()
    try:
        model = ModelTTNN(device)

        # Single-CQ metal trace with persistent DRAM inputs. The graph inputs
        # produced by load_inputs are the persistent DRAM tensors the trace reads
        # from; each run refills them (outside the timed region) from a host copy
        # via copy_host_to_device_tensor, keeping their device addresses fixed.
        dram_inputs = load_inputs(device)
        host_inputs = [ttnn.from_device(dram_tensor) for dram_tensor in dram_inputs]

        def fill_inputs():
            for host_tensor, dram_tensor in zip(host_inputs, dram_inputs):
                ttnn.copy_host_to_device_tensor(host_tensor, dram_tensor, cq_id=0)

        def log_perf(run, elapsed):
            print(f"\nRun {run}: time {elapsed:.4f}s, TPS {num_tokens / elapsed:.2f}")

        # Run 1: run the model to compile the ops and fill the program cache.
        fill_inputs()
        start = time.perf_counter()
        outputs = model(dram_inputs)
        ttnn.synchronize_device(device)
        log_perf(1, time.perf_counter() - start)

        # Run 2: capture the trace. Keep the output tensor on device (returned
        # here) so we have its address to read from after executing the trace.
        fill_inputs()
        start = time.perf_counter()
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        outputs = model(dram_inputs)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        log_perf(2, time.perf_counter() - start)

        # Runs 3-5: execute the captured trace.
        host_output = None
        for run in range(3, 6):
            fill_inputs()
            start = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            host_output = outputs[0].cpu(blocking=False)
            ttnn.synchronize_device(device)
            log_perf(run, time.perf_counter() - start)

        ttnn.release_trace(device, tid)

        # The graph runs tensor-parallel on a (1, 4) mesh. Its single output (the
        # final hidden state) is replicated across the mesh -- the last collective
        # feeding it is an all_gather -- so bare ttnn.to_torch would abort without a
        # mesh composer. For a replicated tensor any one device shard is the full
        # tensor, so select a single replica to compose it back to torch.
        ttnn_output = ttnn.to_torch(ttnn.get_device_tensors(host_output)[0])
    finally:
        utils.close_device(device)

    golden_output = model_pt.run_pytorch_model()

    pcc = calculate_pcc(ttnn_output, golden_output)
    print(f"\nPCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"


if __name__ == "__main__":
    main()
