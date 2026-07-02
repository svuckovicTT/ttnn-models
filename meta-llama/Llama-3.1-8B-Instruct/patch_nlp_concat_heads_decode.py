#!/usr/bin/env python3
"""Patch a codegen'd TTNN main.py so every ttnn.experimental.nlp_concat_heads_decode
call passes `sub_core_grids`.

Why: the decode-attention input is HEIGHT_SHARDED on a *sub-core* grid (its
CoreRangeSet has >1 range). tt-metal's invoke() then auto-sets on_subcoregrids=True
(nlp_concat_heads_decode_device_operation.cpp:133-140) but leaves sub_core_grids=nullopt
unless the caller passes it. compute_output_specs() (same file:98-99) then does
sub_core_grids.value() -> `bad optional access`. (The friendly TT_FATAL in validate() is
skipped because ttnn runs with enable_fast_runtime_mode=true.)

The tt-mlir *runtime* path sets it as `in.shard_spec().value().grid`
(runtime/lib/ttnn/operations/transformer/nlp_concat_heads_decode.cpp:19-30); EmitPy codegen
does not. This patch reproduces the runtime behavior by deriving the grid from the input
tensor at run time: sub_core_grids=<input>.memory_config().shard_spec.grid.
"""
import shutil
import sys

CALL = "ttnn.experimental.nlp_concat_heads_decode("


def patch(path):
    shutil.copyfile(path, path + ".orig")
    with open(path) as f:
        lines = f.readlines()

    out, i, patched = [], 0, 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        # A call-open line ends with "nlp_concat_heads_decode(" and the very next
        # line is the sole positional arg: the input tensor variable.
        if CALL in line and i + 1 < len(lines):
            inp_line = lines[i + 1]
            already = i + 2 < len(lines) and "sub_core_grids" in lines[i + 2]
            var = inp_line.strip().rstrip(",")
            indent = inp_line[: len(inp_line) - len(inp_line.lstrip())]
            out.append(inp_line)  # keep the positional input arg
            if not already:
                out.append(
                    f"{indent}sub_core_grids={var}.memory_config().shard_spec.grid,\n"
                )
                patched += 1
            i += 2
            continue
        i += 1

    with open(path, "w") as f:
        f.writelines(out)
    print(f"patched {patched} nlp_concat_heads_decode call(s); backup at {path}.orig")


if __name__ == "__main__":
    patch(sys.argv[1])
