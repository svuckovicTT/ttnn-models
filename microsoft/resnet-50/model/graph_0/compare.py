"""Compare runtime vs codegen tensor dumps. Reports the first divergent op."""

import hashlib
import os

GRAPH_DIR = os.path.dirname(os.path.abspath(__file__))
CODEGEN = os.path.join(GRAPH_DIR, "dumps", "codegen")
RUNTIME = os.path.join(GRAPH_DIR, "dumps", "runtime")

LABELS = [
    "conv2d_0",
    "max_pool2d_0",
    "conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4",
    "add_0", "add_1", "add_2",
    "add_3", "add_4", "add_5", "add_6",
    "add_7", "add_8", "add_9", "add_10", "add_11", "add_12",
    "add_13", "add_14", "add_15",
    "linear_0",
    "final",
]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    print(f"{'label':<14} {'cg_size':>10} {'rt_size':>10}   match    cg_md5[:8] rt_md5[:8]")
    print("-" * 80)
    first_diff = None
    for label in LABELS:
        cg_path = os.path.join(CODEGEN, f"{label}.tensorbin")
        rt_path = os.path.join(RUNTIME, f"{label}.tensorbin")
        if not (os.path.exists(cg_path) and os.path.exists(rt_path)):
            print(f"{label:<14} MISSING")
            continue
        cg_size = os.path.getsize(cg_path)
        rt_size = os.path.getsize(rt_path)
        cg_md5 = md5(cg_path)
        rt_md5 = md5(rt_path)
        match = cg_md5 == rt_md5
        marker = "OK " if match else "DIFF"
        print(f"{label:<14} {cg_size:>10} {rt_size:>10}   {marker}    {cg_md5[:8]}    {rt_md5[:8]}")
        if not match and first_diff is None:
            first_diff = label

    print()
    if first_diff is None:
        print("All tensors match bitwise!")
    else:
        print(f"First divergence: {first_diff}")


if __name__ == "__main__":
    main()
