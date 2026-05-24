import re
from collections import defaultdict

path = "/localdev/svuckovic/_workspace/repos/project-alchemy/ttnn-models/Qwen/Qwen3-4B/model/graph_0/main.py"
with open(path) as f:
    lines = f.readlines()

start = None
end = None
for i, line in enumerate(lines):
    if line.startswith("def load_weights_for__main"):
        start = i
    elif line.startswith("def main("):
        end = i
        break

section = lines[start:end]
text = "".join(section)

pattern_start = re.compile(r'utils_load_tensor_(\d+)\s*=\s*utils\.load_tensor\(')

groups = defaultdict(list)
total = 0

N = len(text)
matches = list(pattern_start.finditer(text))

for idx, m in enumerate(matches):
    tensor_id = m.group(1)
    pos = m.end()
    depth = 1
    j = pos
    while j < N and depth > 0:
        c = text[j]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        j += 1
    call_content = text[pos:j-1]

    rest_end = matches[idx+1].start() if idx+1 < len(matches) else N
    rest = text[j:rest_end]

    key_match = re.search(r'_main_weights\["([^"]+)"\]\s*=', rest)
    if not key_match:
        print(f"NO KEY FOUND for tensor {tensor_id}")
        continue
    key = key_match.group(1)

    args = []
    depth = 0
    cur = []
    for c in call_content:
        if c == '(':
            depth += 1
            cur.append(c)
        elif c == ')':
            depth -= 1
            cur.append(c)
        elif c == ',' and depth == 0:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
    if "".join(cur).strip():
        args.append("".join(cur).strip())

    if len(args) != 5:
        print(f"Unexpected arg count {len(args)} for tensor {tensor_id} key {key}")
        print(repr(args))
        continue

    filename, layout, dtype, device, memcfg = args
    device_is_none = (device.strip() == "None")

    memcfg_norm = re.sub(r'\s+', ' ', memcfg).strip()
    layout_norm = layout.strip()
    dtype_norm = dtype.strip()

    group_key = (layout_norm, dtype_norm, device_is_none, memcfg_norm)
    groups[group_key].append(key)
    total += 1

out_lines = []
out_lines.append(f"TOTAL WEIGHTS PARSED: {total}")
out_lines.append(f"NUMBER OF GROUPS: {len(groups)}")
out_lines.append("")
for i, (gkey, keys) in enumerate(groups.items(), 1):
    layout, dtype, dev_none, memcfg = gkey
    out_lines.append(f"=== GROUP {i} ===")
    out_lines.append(f"layout={layout}")
    out_lines.append(f"dtype={dtype}")
    out_lines.append(f"device_is_none={dev_none}")
    out_lines.append(f"memory_config={memcfg}")
    out_lines.append(f"COUNT: {len(keys)}")
    out_lines.append(f"KEYS:")
    for k in keys:
        out_lines.append(f"  {k}")
    out_lines.append("")

out_path = "/localdev/svuckovic/_workspace/repos/project-alchemy/ttnn-models/Qwen/Qwen3-4B/parse_out.txt"
with open(out_path, "w") as f:
    f.write("\n".join(out_lines))

print(f"Wrote {len(out_lines)} lines to {out_path}")
print(f"TOTAL: {total}, GROUPS: {len(groups)}")
