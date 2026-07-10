# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Verify the two graph-input loaders agree.

``load_activations_for__main`` loads the serialized inputs from disk;
``load_inputs`` rebuilds them from the golden PyTorch model's inputs. This test
asserts they produce the same inputs: same count, same order, same tensor
values, and the same per-input mesh distribution (replicated vs sharded, and the
shard dim) -- not just the values.
"""
import ttnn
import torch

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
load_activations_for__main = _activations.load_activations_for__main
load_inputs = _activations.load_inputs


def _reassemble(tensor):
    """Return ``(distribution, full_torch_tensor)`` for a (possibly mesh) tensor.

    ``distribution`` is ``"single"``, ``"replicated"``, or ``("sharded", dim)``.
    It is decided empirically (per mesh_tensors.md): a tensor whose per-device
    shards are all equal is replicated -- even when its metadata labels it a
    shard of a size-1 dim, as the disk-loaded activations here do -- and a
    genuinely sharded tensor is concatenated back along the dim whose per-shard
    sizes differ. Never call bare ``ttnn.to_torch`` on the multi-device tensor.
    """
    if tensor.device() is not None:
        tensor = ttnn.from_device(tensor)

    shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]

    if len(shards) == 1:
        return "single", shards[0]

    if all(torch.equal(shards[0], shard) for shard in shards):
        return "replicated", shards[0]

    ref = shards[0]
    shard_dim = next(
        (d for d in range(ref.dim()) if any(s.shape[d] != ref.shape[d] for s in shards)),
        0,
    )
    return ("sharded", shard_dim), torch.cat(shards, dim=shard_dim)


def test_verify_activations():
    disk_inputs = load_activations_for__main()
    rebuilt_inputs = load_inputs()

    assert len(disk_inputs) == len(rebuilt_inputs), (
        f"Count mismatch: disk={len(disk_inputs)}, load_inputs={len(rebuilt_inputs)}."
    )

    mismatches = []
    for i, (t1, t2) in enumerate(zip(disk_inputs, rebuilt_inputs)):
        dist1, pt1 = _reassemble(t1)
        dist2, pt2 = _reassemble(t2)

        # Distribution must match (replicated vs sharded, and the shard dim).
        assert dist1 == dist2, (
            f"arg{i}: distribution mismatch: disk={dist1}, load_inputs={dist2}."
        )

        # Values must match exactly.
        assert pt1.shape == pt2.shape, (
            f"arg{i}: shape mismatch: disk={tuple(pt1.shape)}, "
            f"load_inputs={tuple(pt2.shape)}."
        )
        if not torch.equal(pt1, pt2):
            max_diff = (pt1.float() - pt2.float()).abs().max().item()
            mismatches.append((i, max_diff))
            print(f"  MISMATCH: arg{i} (max diff: {max_diff})")
        else:
            print(f"  OK: arg{i} ({dist1})")

    assert not mismatches, (
        f"{len(mismatches)} / {len(disk_inputs)} inputs mismatch: "
        + ", ".join(f"arg{i}={d}" for i, d in mismatches)
    )
    print(f"All {len(disk_inputs)} inputs match!")
