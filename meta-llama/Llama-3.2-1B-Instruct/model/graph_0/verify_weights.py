# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from params import (
    load_weights_for__main,
    load_weights_for__main_from_state_dict,
)


def test_verify_weights():
    disk_weights = load_weights_for__main()
    sd_weights = load_weights_for__main_from_state_dict()

    assert len(disk_weights) == len(sd_weights), (
        f"Element count mismatch: disk={len(disk_weights)}, state_dict={len(sd_weights)}."
    )

    disk_keys = set(disk_weights.keys())
    sd_keys = set(sd_weights.keys())
    assert disk_keys == sd_keys, (
        f"Key mismatch. In disk but not state_dict: {disk_keys - sd_keys}. "
        f"In state_dict but not disk: {sd_keys - disk_keys}."
    )

    mismatches = []
    for key in sorted(disk_weights.keys()):
        t1 = disk_weights[key]
        t2 = sd_weights[key]

        if t1.device() is not None:
            t1 = ttnn.from_device(t1)
        if t2.device() is not None:
            t2 = ttnn.from_device(t2)

        pt1 = ttnn.to_torch(t1).flatten()
        pt2 = ttnn.to_torch(t2).flatten()

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
