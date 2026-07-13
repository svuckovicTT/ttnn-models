# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Verify the two weight loaders agree.

``load_weights_for__main`` loads the serialized weights from disk;
``load_weights_for__main_from_state_dict`` rebuilds them from the golden PyTorch
model's ``state_dict``. This test asserts they produce the same weights: same
count, same keys, same tensor values, and the same per-key mesh distribution
(replicated vs sharded, and the shard dim) -- not just the values.
"""
import ttnn
import torch

# Load params.py directly by path so the correct sibling module is used
# regardless of pytest's import mode / sys.path.
import importlib.util as _importlib_util
from pathlib import Path as _Path

_params_spec = _importlib_util.spec_from_file_location(
    "graph_params", _Path(__file__).resolve().parent / "params.py"
)
_params = _importlib_util.module_from_spec(_params_spec)
_params_spec.loader.exec_module(_params)
load_weights_for__main = _params.load_weights_for__main
load_weights_for__main_from_state_dict = _params.load_weights_for__main_from_state_dict


def _reassemble(tensor):
    """Return ``(distribution, full_torch_tensor)`` for a (possibly mesh) tensor.

    ``distribution`` is ``"single"``, ``"replicated"``, or ``("sharded", dim)``.
    It is decided empirically (per mesh_tensors.md): a tensor whose per-device
    shards are all equal is replicated, and a genuinely sharded tensor is
    concatenated back along the dim whose per-shard sizes differ. Never call bare
    ``ttnn.to_torch`` on the multi-device tensor.
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


def test_verify_weights():
    disk_weights = load_weights_for__main()
    sd_weights = load_weights_for__main_from_state_dict()

    assert len(disk_weights) == len(sd_weights), (
        f"Count mismatch: disk={len(disk_weights)}, "
        f"state_dict={len(sd_weights)}."
    )

    disk_keys = set(disk_weights.keys())
    sd_keys = set(sd_weights.keys())
    assert disk_keys == sd_keys, (
        f"Key mismatch. In disk but not state_dict: {disk_keys - sd_keys}. "
        f"In state_dict but not disk: {sd_keys - disk_keys}."
    )

    mismatches = []
    for key in sorted(disk_weights.keys()):
        dist1, pt1 = _reassemble(disk_weights[key])
        dist2, pt2 = _reassemble(sd_weights[key])

        # Distribution must match (replicated vs sharded, and the shard dim).
        assert dist1 == dist2, (
            f"{key}: distribution mismatch: disk={dist1}, state_dict={dist2}."
        )

        # Values must match exactly.
        assert pt1.shape == pt2.shape, (
            f"{key}: shape mismatch: disk={tuple(pt1.shape)}, "
            f"state_dict={tuple(pt2.shape)}."
        )
        if not torch.equal(pt1, pt2):
            max_diff = (pt1.float() - pt2.float()).abs().max().item()
            mismatches.append((key, max_diff))
            print(f"  MISMATCH: {key} (max diff: {max_diff})")
        else:
            print(f"  OK: {key} ({dist1})")

    assert not mismatches, (
        f"{len(mismatches)} / {len(disk_weights)} weights mismatch: "
        + ", ".join(f"{k}={d}" for k, d in mismatches)
    )
    print(f"All {len(disk_weights)} weights match!")
