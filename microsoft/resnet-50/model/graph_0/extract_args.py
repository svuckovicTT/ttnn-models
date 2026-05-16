"""Extract @main argument metadata from ttnn.mlir."""
import re
import os
GRAPH_DIR = os.path.dirname(os.path.abspath(__file__))
mlir_path = os.path.join(GRAPH_DIR, "ttnn.mlir")
with open(mlir_path) as f:
    src = f.read()

# Find func.func @main(...)
m = re.search(r"func\.func @main\((.*?)\)\s*->", src, re.DOTALL)
assert m, "Could not find @main signature"
sig = m.group(1)
# Split args by `, %argN:` boundary - careful with nested types
# We'll match each %argN: ... up to next %argN+1: or end
arg_pattern = re.compile(
    r"%arg(\d+):\s*(.*?)\s*(?=,\s*%arg\d+:|$)",
    re.DOTALL,
)
arg_types_pattern = re.compile(r"^tensor<([^,]+),")
name_pattern = re.compile(r'ttir\.name = "([^"]+)"')
argtype_pattern = re.compile(r"ttcore\.argument_type<(\w+)>")

args = []
for am in arg_pattern.finditer(sig):
    idx = int(am.group(1))
    body = am.group(2)
    nm = name_pattern.search(body)
    at = argtype_pattern.search(body)
    tt = arg_types_pattern.search(body.strip())
    args.append({
        "index": idx,
        "name": nm.group(1) if nm else None,
        "type": at.group(1) if at else None,
        "shape": tt.group(1).strip() if tt else None,
    })

# Sanity print
print(f"Total args: {len(args)}")
argtypes_seen = {}
for a in args:
    argtypes_seen[a["type"]] = argtypes_seen.get(a["type"], 0) + 1
print("Arg type counts:", argtypes_seen)

# Find inputs (non-parameter, non-constant — likely "input")
inputs = [a for a in args if a["type"] not in ("parameter", "constant")]
print(f"\nNon-param/const args ({len(inputs)}):")
for a in inputs:
    print(f"  arg{a['index']:3} type={a['type']!r:15} name={a['name']!r:40} shape={a['shape']}")

# Show the first/last few args
print("\nFirst 5 args:")
for a in args[:5]:
    print(f"  arg{a['index']:3} type={a['type']!r:15} name={a['name']!r:40} shape={a['shape']}")
print("Last 5 args:")
for a in args[-5:]:
    print(f"  arg{a['index']:3} type={a['type']!r:15} name={a['name']!r:40} shape={a['shape']}")

# Now parse main.py to learn ttir.name -> argN.tensorbin
main_py = os.path.join(GRAPH_DIR, "main.py")
with open(main_py) as f:
    py = f.read()

# Pattern: utils_load_tensor_K = utils.load_tensor("./tensors/argM.tensorbin", ...);
# _main_weights["NAME"] = utils_load_tensor_K
load_re = re.compile(
    r'utils_load_tensor_(\d+)\s*=\s*utils\.load_tensor\(\s*"\./tensors/(arg\d+)\.tensorbin"',
)
assign_weight_re = re.compile(
    r'_main_weights\[\s*"([^"]+)"\s*\]\s*=\s*\(?\s*utils_load_tensor_(\d+)',
)
# Some lines wrap with parens or are multi-line; allow flexible match.
assign_weight_re2 = re.compile(
    r'_main_weights\[\s*\n?\s*"([^"]+)"\s*\n?\s*\]\s*=\s*\(?\s*utils_load_tensor_(\d+)',
    re.DOTALL,
)

# Map K -> argM
k_to_argbin = {int(k): argm for k, argm in load_re.findall(py)}
# Map K -> ttir name
k_to_name = {}
for nm, k in assign_weight_re2.findall(py):
    k_to_name[int(k)] = nm
# Build name -> argbin
name_to_argbin = {}
for k, argbin in k_to_argbin.items():
    if k in k_to_name:
        name_to_argbin[k_to_name[k]] = argbin

print(f"\nLoaded {len(k_to_argbin)} tensor loads, {len(k_to_name)} weight assignments")
print(f"Name-> tensorbin mappings: {len(name_to_argbin)}")
# Show a few examples
for name, argbin in list(name_to_argbin.items())[:5]:
    print(f"  {name!r} -> {argbin}.tensorbin")

# Find activation file (used in load_activations_for__main)
act_re = re.compile(r'load_activations.*?"\./tensors/(arg\d+)\.tensorbin"', re.DOTALL)
am = act_re.search(py)
print(f"\nActivation file: {am.group(1)+'.tensorbin' if am else 'NOT FOUND'}")

# Now map each MLIR @main %argI to a tensorbin file
unmapped = []
mapping = []
for a in args:
    name = a["name"]
    if name in name_to_argbin:
        mapping.append((a["index"], name_to_argbin[name], a["shape"], a["type"]))
    else:
        unmapped.append(a)
print(f"\nMapped @main args: {len(mapping)}/{len(args)}")
print(f"Unmapped: {len(unmapped)}")
for u in unmapped[:5]:
    print(f"  arg{u['index']:3} type={u['type']!r} name={u['name']!r}")

# Write mapping out
import json
with open(os.path.join(GRAPH_DIR, "arg_mapping.json"), "w") as f:
    json.dump({
        "mapping": [{"index": i, "tensorbin": tb, "shape": sh, "type": t} for (i, tb, sh, t) in mapping],
        "unmapped": [{"index": u["index"], "name": u["name"], "type": u["type"], "shape": u["shape"]} for u in unmapped],
        "activation": am.group(1) + ".tensorbin" if am else None,
    }, f, indent=2)
print("\nWrote arg_mapping.json")
