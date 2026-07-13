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

        # Run the same input through the model 3 times and log the perf. The
        # forward deallocates its input tensors, so rebuild the (identical) input
        # each iteration; the reload happens outside the timed region.
        for run in range(3):
            input = load_inputs(device)
            start = time.perf_counter()
            outputs = model(input)
            # Wait for the device to finish before reading the end time, so the
            # measurement covers the whole forward and not just op dispatch.
            ttnn.synchronize_device(device)
            elapsed = time.perf_counter() - start
            tps = num_tokens / elapsed
            print(f"\nRun {run + 1}: time {elapsed:.4f}s, TPS {tps:.2f}")

        # The graph runs tensor-parallel on a (1, 4) mesh. Its single output (the
        # final hidden state) is replicated across the mesh -- the last collective
        # feeding it is an all_gather -- so bare ttnn.to_torch would abort without a
        # mesh composer. For a replicated tensor any one device shard is the full
        # tensor, so select a single replica to compose it back to torch.
        ttnn_output = ttnn.to_torch(
            ttnn.get_device_tensors(ttnn.from_device(outputs[0]))[0]
        )
    finally:
        utils.close_device(device)

    golden_output = model_pt.run_pytorch_model()

    pcc = calculate_pcc(ttnn_output, golden_output)
    print(f"\nPCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"


if __name__ == "__main__":
    main()
