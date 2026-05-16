"""Inject ttnn.dump_tensor ops in ttnn_instrumented.mlir for all 106 outputs
of @main_const_eval_0 (the ttcore.load_cached call), so the runtime path
materializes them to disk. Then we can substitute them into the codegen.
"""

import re
import os

GRAPH_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.join(GRAPH_DIR, "dumps", "runtime_consteval")
os.makedirs(RUNTIME_DIR, exist_ok=True)

with open(os.path.join(GRAPH_DIR, "ttnn_instrumented.mlir")) as f:
    src = f.read()

# Find the load_cached call for @main_const_eval_0. It's a single line.
# Pattern: %0:106 = ttcore.load_cached(@main_const_eval_0, [...]) : (...) -> (TYPES) loc(...)
m = re.search(r"(%0:106\s*=\s*ttcore\.load_cached\(@main_const_eval_0,.*?\))\s*:\s*\(.*?\)\s*->\s*\((.*?)\)\s*loc\(", src, re.DOTALL)
assert m, "load_cached for main_const_eval_0 not found"
result_types_str = m.group(2)

# Split result types - they're comma-separated tensor<...> entries.
# Be careful: nested angle brackets in tensor<...>. Track depth.
types = []
depth = 0
buf = ""
for ch in result_types_str:
    if ch == "<":
        depth += 1
        buf += ch
    elif ch == ">":
        depth -= 1
        buf += ch
    elif ch == "," and depth == 0:
        types.append(buf.strip())
        buf = ""
    else:
        buf += ch
if buf.strip():
    types.append(buf.strip())
assert len(types) == 106, f"Expected 106 result types, got {len(types)}"

# Build a block of dump_tensor ops to insert after the load_cached line.
dumps = []
for i, t in enumerate(types):
    path = os.path.join(RUNTIME_DIR, f"ve_{i}.tensorbin")
    dumps.append(f'        "ttnn.dump_tensor"(%0#{i}) <{{file_path = "{path}"}}> : ({t}) -> () loc(#loc)')

# Locate the line containing the load_cached call and insert dumps after.
lines = src.split("\n")
new_lines = []
inserted = False
for line in lines:
    new_lines.append(line)
    if "%0:106 = ttcore.load_cached(@main_const_eval_0" in line and not inserted:
        new_lines.extend(dumps)
        inserted = True
assert inserted

with open(os.path.join(GRAPH_DIR, "ttnn_ce_dumped.mlir"), "w") as f:
    f.write("\n".join(new_lines))
print(f"Wrote ttnn_ce_dumped.mlir with {len(dumps)} dump_tensor ops")
print(f"Dumps will go to {RUNTIME_DIR}")
