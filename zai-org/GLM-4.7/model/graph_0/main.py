import math

import torch
import ttnn
import utils
from utils import calculate_pcc

from model_ttnn import ModelTTNN

MESH_SHAPE = (4, 8)
L1_SMALL_SIZE = 1 << 15

_decode_state = None


def _decode_inputs():
    """Load the PyTorch model and run a CPU prefill to reach the decode state
    (mirrors xla.py, which produced the serialized activations). Cached so the
    expensive prefill runs once even if activations are requested repeatedly."""
    global _decode_state
    if _decode_state is None:
        import model_pt

        model = model_pt.load_pytorch_model()
        _decode_state = model_pt._cpu_prefill_to_decode_state(model)
    return _decode_state


def open_device():
    if math.prod(MESH_SHAPE) >= 2:
        # FABRIC_1D_RING is required by the fused moe_compute / dispatch_metadata
        # CCL ops (ring topology); also matches the original run-graph config.
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    # COL dispatch axis: the fused moe_compute op hardwires its tilize cores at
    # the (5-6, 8-9) grid corner; under the default ROW dispatch the DRAM
    # matmul-core assignment spans the whole 8x9 grid and overlaps them
    # ("tilize and matmul bounding boxes cannot overlap"). COL dispatch (matching
    # the deepseek TG reference) reshapes the usable grid so they don't collide.
    dispatch_core_config = ttnn.DispatchCoreConfig(
        ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.COL
    )
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(MESH_SHAPE),
        l1_small_size=L1_SMALL_SIZE,
        dispatch_core_config=dispatch_core_config,
    )
    return device


def close_device(device):
    ttnn.close_mesh_device(device)
    if math.prod(MESH_SHAPE) >= 2:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def load_activations_for__main(device):
    """Construct the decode-step inputs the way xla.py did when it produced the
    serialized activation tensorbins: load the PyTorch model, run a CPU prefill to
    populate the StaticCache, and snapshot the resulting decode state (next-token
    input_ids, advanced cache_position, populated per-layer KV caches), then
    distribute them across the mesh. Replaces loading the serialized tensors from
    disk."""
    decode_args = _decode_inputs()
    input_ids = decode_args["input_ids"]
    cache_position = decode_args["cache_position"]
    layers = decode_args["past_key_values"].layers
    # cumulative_length == number of prefilled tokens == the decode cache position.
    cumulative_length = int(cache_position.reshape(-1)[0].item())

    dram = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    )

    def to_device_tensor(torch_tensor, dtype, layout, mesh_mapper):
        ttnn_tensor = ttnn.from_torch(torch_tensor, mesh_mapper=mesh_mapper)
        ttnn_tensor = ttnn.to_layout(ttnn_tensor, layout)
        ttnn_tensor = ttnn.to_dtype(ttnn_tensor, dtype)
        return ttnn.to_device(ttnn_tensor, device, dram)

    # input_ids is sharded along the mesh "batch" axis (the 4 rows); the per-layer
    # cumulative_length and cache_position are replicated.
    def batch_sharded():
        return ttnn.ShardTensor2dMesh(device, MESH_SHAPE, (0, None))

    # KV caches use the ("batch", "model", None, None) layout: batch on the mesh
    # rows and KV heads on the mesh columns (model axis), so each device owns a
    # single KV head. paged_update_cache / SDPA then run on the local shard with
    # no cross-device point-to-point (see tt-xla issue 5096).
    def kv_head_sharded():
        return ttnn.ShardTensor2dMesh(device, MESH_SHAPE, (0, 1))

    def replicated():
        return ttnn.ReplicateTensorToMesh(device)

    activations = [
        to_device_tensor(
            input_ids.to(torch.int32),
            ttnn.DataType.INT32,
            ttnn.Layout.TILE,
            batch_sharded(),
        ),
        to_device_tensor(
            cache_position.to(torch.int32),
            ttnn.DataType.INT32,
            ttnn.Layout.ROW_MAJOR,
            replicated(),
        ),
    ]
    for layer in layers:
        activations.append(
            to_device_tensor(
                torch.tensor([cumulative_length], dtype=torch.int32),
                ttnn.DataType.INT32,
                ttnn.Layout.ROW_MAJOR,
                replicated(),
            )
        )
        activations.append(
            to_device_tensor(
                layer.keys.to(torch.bfloat16),
                ttnn.DataType.BFLOAT16,
                ttnn.Layout.TILE,
                kv_head_sharded(),
            )
        )
        activations.append(
            to_device_tensor(
                layer.values.to(torch.bfloat16),
                ttnn.DataType.BFLOAT16,
                ttnn.Layout.TILE,
                kv_head_sharded(),
            )
        )
    return activations


def main():
    """Single decode forward. This is the tracy profiling target: run with
    `./run -t` to capture device op perf, then scope tt-perf-report to the
    signposts. Device time (not wall-clock TPS) is the perf metric.

    Set GLM_CHECK_PCC=1 to additionally run the CPU golden and verify PCC; left
    off for tracy runs so profiling stays lean (no CPU prefill/decode golden)."""
    import os

    import sys

    device = open_device()
    activations = load_activations_for__main(device)
    model = ModelTTNN(device)
    print(">>> model built; FORWARD START", flush=True, file=sys.stderr)
    outputs = model(activations)
    print(">>> FORWARD DONE (pre-sync)", flush=True, file=sys.stderr)
    ttnn.synchronize_device(device)
    print(">>> SYNC DONE", flush=True, file=sys.stderr)

    if os.environ.get("GLM_CHECK_PCC") == "1":
        import model_pt

        # PCC floor for perf tuning: the routing-fixed fused-moe_compute model has
        # 0.992; hold >= 0.99 (the initial good version's quality) while optimizing.
        exact_pcc = 0.99

        ttnn_output = [ttnn.from_device(output) for output in outputs]

        # The CPU golden decode is deterministic (fixed model + prefill) and
        # independent of any device-side change, so compute it once and cache it
        # to disk; every subsequent PCC run reuses it instead of re-running the
        # expensive pytorch decode. Delete golden_output.pt if model_pt or the
        # decode inputs ever change.
        golden_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_output.pt")
        if os.path.exists(golden_path):
            golden_output = torch.load(golden_path)
            print(f"Loaded cached golden from {golden_path}")
        else:
            golden_output = model_pt.run_pytorch_model()
            torch.save(golden_output, golden_path)
            print(f"Computed and cached golden to {golden_path}")

        # outputs[-1] is the final logits, fully replicated across the 4x8 mesh
        # by the trailing all_gathers (dim 0 over cluster_axis 0, dim 2 over
        # cluster_axis 1). Every device holds an identical copy, so grab a single
        # shard before converting.
        final_output = ttnn.get_device_tensors(ttnn_output[-1])[0]

        pcc = calculate_pcc(ttnn.to_torch(final_output), golden_output)
        print(f"\nPCC: {pcc:.6f}")
        # This PCC is gauge-invariant w.r.t. the sharding/grid choices ported from
        # the tt-xla "good" benchmark: expert (batch,model) layout, head-sharded KV,
        # and the 8x9 MoE sparse_matmul grid all leave it bit-identical at 0.894531.
        # The remaining gap to the tt-xla SPMD benchmark's ~0.993 is precision, not
        # sharding -- the codegen path quantizes experts to bf8 and runs default
        # (LoFi) math fidelity on those matmuls; the SPMD runtime keeps higher
        # effective precision. Compare with tolerance instead of exact equality.
        assert pcc >= exact_pcc, f"PCC {pcc} below floor {exact_pcc}"
        print(f"PCC check passed (floor {exact_pcc})")

    close_device(device)
    return 0


if __name__ == "__main__":
    main()
