"""Patch main_instrumented.py so that _main bypasses consteval__main and instead
loads the 106 prepared-weight tensors that the runtime path dumped via
dumps/runtime_consteval/ve_*.tensorbin. This isolates whether the entire
divergence is caused by the codegen's CPU-hoisted const_eval.
"""

import os, re

GRAPH_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_CE = os.path.join(GRAPH_DIR, "dumps", "runtime_consteval")

with open(os.path.join(GRAPH_DIR, "main_instrumented.py")) as f:
    src = f.read()

# Replace the body that builds var_0 from consteval__main with a load loop.
# Target snippet:
#     ce_cache__main = consteval__main(ce_cache__main, weights)
#     args_0 = activations[0]
#     var_0 = ce_cache__main["main_const_eval_0"]
#
# Replace consteval call + var_0 assignment with explicit loads.
replacement = (
    '    ce_cache__main = consteval__main(ce_cache__main, weights)\n'
    '    # === PATCHED: override main_const_eval_0 outputs with runtime dumps ===\n'
    '    _ce_device = utils.DeviceGetter.get_device((1, 1))\n'
    '    _ce_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)\n'
    '    ce_cache__main["main_const_eval_0"] = [\n'
    '        ttnn.to_device(\n'
    '            ttnn.load_tensor(f"' + RUNTIME_CE + '/ve_{i}.tensorbin"),\n'
    '            _ce_device, memory_config=_ce_mc,\n'
    '        )\n'
    '        for i in range(106)\n'
    '    ]\n'
    '    args_0 = activations[0]\n'
    '    var_0 = ce_cache__main["main_const_eval_0"]\n'
)
new_src, n = re.subn(
    r"    ce_cache__main = consteval__main\(ce_cache__main, weights\)\n"
    r"    args_0 = activations\[0\]\n"
    r"    var_0 = ce_cache__main\[\"main_const_eval_0\"\]\n",
    replacement,
    src,
    count=1,
)
assert n == 1, f"replacement count = {n}"

with open(os.path.join(GRAPH_DIR, "main_runtime_ce.py"), "w") as f:
    f.write(new_src)
print("Wrote main_runtime_ce.py")
