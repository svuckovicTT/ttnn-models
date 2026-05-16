"""
Instrument main.py and ttnn.mlir to dump tensors at corresponding points.

Dump targets:
- After ttnn_conv2d_0 (initial 7x7 conv)
- After ttnn_max_pool2d_0
- After each ttnn_add_N (N=0..15) - bottleneck ends
- After ttnn_linear_0
- After final return value
"""

import re
import os

GRAPH_DIR = os.path.dirname(os.path.abspath(__file__))
DUMP_BASE = os.path.join(GRAPH_DIR, "dumps")
CODEGEN_DUMP_DIR = os.path.join(DUMP_BASE, "codegen")
RUNTIME_DUMP_DIR = os.path.join(DUMP_BASE, "runtime")
os.makedirs(CODEGEN_DUMP_DIR, exist_ok=True)
os.makedirs(RUNTIME_DUMP_DIR, exist_ok=True)

# (label, python_var, mlir_ssa) - label is shared, used for filename.
DUMP_POINTS = [
    ("conv2d_0",     "ttnn_conv2d_0",     "%7"),
    ("max_pool2d_0", "ttnn_max_pool2d_0", "%8"),
    ("conv2d_1",     "ttnn_conv2d_1",     "%10"),
    ("conv2d_2",     "ttnn_conv2d_2",     "%11"),
    ("conv2d_3",     "ttnn_conv2d_3",     "%12"),
    ("conv2d_4",     "ttnn_conv2d_4",     "%13"),
    ("add_0",        "ttnn_add_0",        "%14"),
    ("add_1",        "ttnn_add_1",        "%19"),
    ("add_2",        "ttnn_add_2",        "%24"),
    ("add_3",        "ttnn_add_3",        "%32"),
    ("add_4",        "ttnn_add_4",        "%37"),
    ("add_5",        "ttnn_add_5",        "%42"),
    ("add_6",        "ttnn_add_6",        "%47"),
    ("add_7",        "ttnn_add_7",        "%53"),
    ("add_8",        "ttnn_add_8",        "%59"),
    ("add_9",        "ttnn_add_9",        "%65"),
    ("add_10",       "ttnn_add_10",       "%71"),
    ("add_11",       "ttnn_add_11",       "%77"),
    ("add_12",       "ttnn_add_12",       "%83"),
    ("add_13",       "ttnn_add_13",       "%90"),
    ("add_14",       "ttnn_add_14",       "%96"),
    ("add_15",       "ttnn_add_15",       "%101"),
    ("linear_0",     "ttnn_linear_0",     "%107"),
    # final return value - special handling below
]


def patch_main_py(in_path, out_path):
    with open(in_path) as f:
        src = f.read()
    lines = src.split("\n")

    # For each (label, var) pair, find the line: `    <var> = ttnn.X(`
    # then find the matching closing `    )` line (4-space indented closing paren),
    # and inject a `ttnn.dump_tensor(...)` call after it.
    insertions = {}  # line_idx -> [new_lines_to_insert_after]
    for label, var, _ in DUMP_POINTS:
        # Find assignment line. Pattern: `    <var> = ttnn.`
        assign_re = re.compile(r"^    " + re.escape(var) + r"\s*=\s*ttnn\.")
        idx = None
        for i, line in enumerate(lines):
            if assign_re.match(line):
                idx = i
                break
        if idx is None:
            raise RuntimeError(f"Did not find assignment for {var}")
        # Walk to find matching `    )` line (4-space indent, just close paren)
        close_idx = None
        for j in range(idx + 1, len(lines)):
            if lines[j] == "    )":
                close_idx = j
                break
        if close_idx is None:
            raise RuntimeError(f"Did not find closing ')' for {var}")
        dump_path = os.path.join(CODEGEN_DUMP_DIR, f"{label}.tensorbin")
        insertions.setdefault(close_idx, []).append(
            f'    ttnn.dump_tensor("{dump_path}", {var})'
        )

    # Final return value: `return [ttnn_to_memory_config_11]`
    # Inject dump_tensor BEFORE the return line in `_main`.
    return_re = re.compile(r"^    return \[(\w+)\]")
    final_var = None
    final_idx = None
    for i, line in enumerate(lines):
        m = return_re.match(line)
        if m:
            final_var = m.group(1)
            final_idx = i
            break
    assert final_var, "Did not find return"
    # Insert before the return line (i.e., as line at index final_idx - 1 ending position)
    dump_path = os.path.join(CODEGEN_DUMP_DIR, "final.tensorbin")
    # Insert AFTER the line preceding return (i.e., we want dump line just before return).
    # We add to insertions at final_idx-1.
    insertions.setdefault(final_idx - 1, []).append(
        f'    ttnn.dump_tensor("{dump_path}", {final_var})'
    )

    out_lines = []
    for i, line in enumerate(lines):
        out_lines.append(line)
        if i in insertions:
            out_lines.extend(insertions[i])
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
    print(f"Patched {in_path} -> {out_path} ({len(DUMP_POINTS) + 1} dumps)")


def patch_mlir(in_path, out_path):
    with open(in_path) as f:
        lines = f.readlines()

    # Find @main function bounds
    main_start = None
    main_end = None
    for i, line in enumerate(lines):
        if "func.func @main" in line and main_start is None:
            main_start = i
        elif main_start is not None and re.match(r"^\s+\}\s+loc\(", line) and main_end is None:
            main_end = i
            break
    assert main_start is not None and main_end is not None, "Did not find @main bounds"

    # Find the SSA value of the return: `return %N : ...`
    return_re = re.compile(r"^\s+return (%\d+)\s")
    final_ssa = None
    return_idx = None
    for i in range(main_start, main_end + 1):
        m = return_re.match(lines[i])
        if m:
            final_ssa = m.group(1)
            return_idx = i
            break
    assert final_ssa is not None, "Did not find return SSA"

    # For each (label, _, ssa), find line `        <ssa> = "ttnn.X"...` within @main,
    # extract its type signature -> tensor<...>, then inject ttnn.dump_tensor immediately after.
    targets = [(label, ssa) for (label, _, ssa) in DUMP_POINTS]
    # Append final
    targets.append(("final", final_ssa))

    # For each target ssa, find the def line in @main and capture its result type.
    # Pattern: `<spaces>%N = "ttnn.OP"(...) ... -> tensor<...> loc(...)
    # Capture the type after the last `->` up to ` loc(`.
    insertions = {}  # idx -> list of new lines
    for label, ssa in targets:
        def_re = re.compile(r"^(\s+)" + re.escape(ssa) + r" = \"ttnn\.")
        idx = None
        for i in range(main_start, main_end + 1):
            if def_re.match(lines[i]):
                idx = i
                break
        if idx is None:
            raise RuntimeError(f"Did not find def for {ssa}")
        defn = lines[idx]
        indent = re.match(r"^(\s+)", defn).group(1)
        # Extract result type. Walk last `-> ` then up to ` loc(`.
        # The line format is `... -> tensor<...> loc(#locN)` possibly with attributes.
        # Find rightmost ` -> ` and take from there to ` loc(`.
        arrow = defn.rfind(" -> ")
        loc_at = defn.find(" loc(", arrow if arrow >= 0 else 0)
        if arrow < 0:
            raise RuntimeError(f"No -> in def for {ssa}: {defn}")
        if loc_at < 0:
            loc_at = len(defn)
        result_type = defn[arrow + len(" -> "):loc_at].strip()
        dump_path = os.path.join(RUNTIME_DUMP_DIR, f"{label}.tensorbin")
        new_line = (
            f'{indent}"ttnn.dump_tensor"({ssa}) <{{file_path = "{dump_path}"}}> '
            f': ({result_type}) -> () loc(#loc)\n'
        )
        insertions.setdefault(idx, []).append(new_line)

    out_lines = []
    for i, line in enumerate(lines):
        out_lines.append(line)
        if i in insertions:
            out_lines.extend(insertions[i])
    with open(out_path, "w") as f:
        f.writelines(out_lines)
    print(f"Patched {in_path} -> {out_path} ({len(targets)} dumps)")


if __name__ == "__main__":
    patch_main_py(
        os.path.join(GRAPH_DIR, "main.py"),
        os.path.join(GRAPH_DIR, "main_instrumented.py"),
    )
    patch_mlir(
        os.path.join(GRAPH_DIR, "ttnn.mlir"),
        os.path.join(GRAPH_DIR, "ttnn_instrumented.mlir"),
    )
