"""Load both dumps and compare numerically.

If md5 differs but numerical content matches: difference is in metadata (layout/memcfg).
If numerical content also differs: real accuracy bug at that op.
"""

import os
import sys
import torch
import ttnn

GRAPH_DIR = os.path.dirname(os.path.abspath(__file__))
CODEGEN = os.path.join(GRAPH_DIR, "dumps", "codegen")
RUNTIME = os.path.join(GRAPH_DIR, "dumps", "runtime")

LABELS = [
    "conv2d_0",
    "max_pool2d_0",
    "add_0", "add_1", "add_2",
    "add_3", "add_4", "add_5", "add_6",
    "add_7", "add_8", "add_9", "add_10", "add_11", "add_12",
    "add_13", "add_14", "add_15",
    "linear_0",
    "final",
]


def load_to_torch(path):
    t = ttnn.load_tensor(path)
    # Bring to row-major + host before conversion
    if t.layout != ttnn.Layout.ROW_MAJOR:
        try:
            t = ttnn.to_layout(t, ttnn.Layout.ROW_MAJOR)
        except Exception:
            pass
    return ttnn.to_torch(t)


def summarize_diff(cg, rt):
    if cg.shape != rt.shape:
        return f"SHAPE MISMATCH cg={tuple(cg.shape)} rt={tuple(rt.shape)}"
    if cg.dtype != rt.dtype:
        return f"DTYPE MISMATCH cg={cg.dtype} rt={rt.dtype}"
    eq = torch.equal(cg, rt)
    if eq:
        return "EQUAL"
    cg_f = cg.float()
    rt_f = rt.float()
    diff = (cg_f - rt_f).abs()
    nz = (cg_f != rt_f).sum().item()
    total = cg_f.numel()
    return (f"DIFF nz={nz}/{total} ({100.0*nz/total:.2f}%) "
            f"max_abs={diff.max().item():.6g} mean_abs={diff.mean().item():.6g} "
            f"cg_range=[{cg_f.min().item():.4g},{cg_f.max().item():.4g}] "
            f"rt_range=[{rt_f.min().item():.4g},{rt_f.max().item():.4g}]")


def main():
    first_data_diff = None
    print(f"{'label':<14} {'shape':<26} status")
    print("-" * 100)
    for label in LABELS:
        cg_path = os.path.join(CODEGEN, f"{label}.tensorbin")
        rt_path = os.path.join(RUNTIME, f"{label}.tensorbin")
        try:
            cg = load_to_torch(cg_path)
            rt = load_to_torch(rt_path)
        except Exception as e:
            print(f"{label:<14} ERROR loading: {e}")
            continue
        shape = str(tuple(cg.shape))
        msg = summarize_diff(cg, rt)
        print(f"{label:<14} {shape:<26} {msg}")
        if msg != "EQUAL" and not msg.startswith("DIFF nz=0") and first_data_diff is None:
            first_data_diff = label

    print()
    print(f"First numerical divergence: {first_data_diff}")


if __name__ == "__main__":
    main()
