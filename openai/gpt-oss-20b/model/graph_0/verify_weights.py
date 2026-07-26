# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch
from params import load_weights_for__main, load_weights_for__main_from_state_dict


def test_verify_weights():
    disk_weights = load_weights_for__main()
    sd_weights = load_weights_for__main_from_state_dict()

    disk_keys = set(disk_weights.keys())
    sd_keys = set(sd_weights.keys())
    assert disk_keys == sd_keys, (
        f"Key mismatch. In disk but not state_dict: {disk_keys - sd_keys}. "
        f"In state_dict but not disk: {sd_keys - disk_keys}."
    )

    def _shards(t):
        if t.device() is not None:
            t = ttnn.from_device(t)
        return [ttnn.to_torch(s) for s in ttnn.get_device_tensors(t)]

    def _logical(shards, ref_shape=None):
        if len(shards) == 1:
            return shards[0]
        if all(torch.equal(shards[0], s) for s in shards[1:]):
            return shards[0]
        if ref_shape is not None:
            for d in range(shards[0].dim()):
                if shards[0].shape[d] * len(shards) == ref_shape[d]:
                    return torch.cat(shards, dim=d)
        return torch.cat(shards, dim=0)

    mismatches = []
    for key in sorted(disk_weights.keys()):
        sd_logical = _logical(_shards(sd_weights[key]))
        pt2 = sd_logical.flatten()
        pt1 = _logical(_shards(disk_weights[key]), sd_logical.shape).flatten()

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
