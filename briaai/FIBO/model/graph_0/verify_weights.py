# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
from params import load_weights_for__main, load_weights_for__main_from_state_dict


def _to_logical(t, target_shape=None):
    # Reconstruct the full logical torch tensor from a ttnn tensor that may be
    # distributed across the (1, 4) mesh. ttnn.to_torch() rejects a multi-device
    # tensor without a mesh composer, so decompose the shards explicitly. Weights
    # come in three forms: single-device, replicated (all shards identical), or
    # sharded (each shard is a distinct slice, e.g. Megatron column-split linears).
    shards = [ttnn.to_torch(s) for s in ttnn.get_device_tensors(t)]
    if len(shards) == 1 or all(torch.equal(s, shards[0]) for s in shards):
        return shards[0]  # single-device or replicated -> a replica is the whole tensor
    # sharded: concatenate along the dim that reproduces the peer's logical shape
    if target_shape is not None:
        for d in range(shards[0].ndim):
            cat = torch.cat(shards, dim=d)
            if tuple(cat.shape) == tuple(target_shape):
                return cat
    return torch.cat(shards, dim=shards[0].ndim - 1)


def test_verify_weights():
    disk_weights = load_weights_for__main()
    sd_weights = load_weights_for__main_from_state_dict()

    disk_keys = set(disk_weights.keys())
    sd_keys = set(sd_weights.keys())
    assert disk_keys == sd_keys, (
        f"Key mismatch. In disk but not state_dict: {disk_keys - sd_keys}. "
        f"In state_dict but not disk: {sd_keys - disk_keys}."
    )

    mismatches = []
    for key in sorted(disk_weights.keys()):
        # sd weights are always single/replicated, so this yields the full logical
        # tensor; use its shape to reconstruct the (possibly sharded) disk weight.
        sd_logical = _to_logical(sd_weights[key])
        pt2 = sd_logical.flatten()
        pt1 = _to_logical(disk_weights[key], sd_logical.shape).flatten()

        if not torch.equal(pt1, pt2):
            max_diff = (pt1 - pt2).abs().max().item()
            mismatches.append((key, max_diff))
            print(f"  MISMATCH: {key} (max diff: {max_diff})")
        else:
            print(f"  OK: {key}")

    assert not mismatches, (
        f"{len(mismatches)} / {len(disk_weights)} weights mismatch: "
        + ", ".join(f"{k}={d}" for k, d in mismatches)
    )
    print(f"All {len(disk_weights)} weights match!")
