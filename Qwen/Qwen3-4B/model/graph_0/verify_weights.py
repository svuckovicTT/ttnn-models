# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys

import torch
import ttnn

import model_pt
from params import (
    load_weights_for__main,
    load_weights_for__main_from_state_dict,
)


def verify() -> bool:
    disk_weights = load_weights_for__main()
    pt_model = model_pt.load_pytorch_model()
    sd_weights = load_weights_for__main_from_state_dict(pt_model.state_dict())

    disk_keys = set(disk_weights.keys())
    sd_keys = set(sd_weights.keys())
    if disk_keys != sd_keys:
        missing = disk_keys - sd_keys
        extra = sd_keys - disk_keys
        if missing:
            print(f"Keys in disk but not state_dict: {missing}")
        if extra:
            print(f"Keys in state_dict but not disk: {extra}")
        return False

    mismatches = []
    for key in sorted(disk_weights.keys()):
        t1 = disk_weights[key]
        t2 = sd_weights[key]

        # Bring both to host as torch tensors for comparison
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

    print()
    if mismatches:
        print(f"{len(mismatches)} / {len(disk_weights)} weights have mismatches:")
        for key, diff in mismatches:
            print(f"  {key}: {diff}")
        return False
    else:
        print(f"All {len(disk_weights)} weights match!")
        return True


if __name__ == "__main__":
    sys.exit(0 if verify() else 1)
