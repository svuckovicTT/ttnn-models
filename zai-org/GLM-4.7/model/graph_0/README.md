# GLM-4.7 decode graph_0 — raw tt-xla codegen ("099")

This branch holds the **raw `tt-xla` codegen** of the GLM-4.7 4-layer decode graph,
as emitted from the `test_glm_4_7_tp_galaxy_4_layers` benchmark (the test whose
`required_pcc` threshold is **0.99**). It is the literal compiled graph the benchmark
runs — *not* the hand-prettified rewrite on the `mvasiljevic/glm-4.7-wh-galaxy-good-pcc`
branch. Keep them on separate branches to compare emitted-vs-prettified.

## What's here

- `main.py` — emitted decode graph. `forward(inputs, device)` consumes the serialized
  input tensors and returns the decode outputs (logits last). Monolithic, machine-generated.
- `utils.py`, `ttir_cpu.py`, `__init__.py`, `run`, `ttnn.mlir`, `irs/` — supporting
  emitted artifacts and the MLIR dumps (shlo / ttir / ttnn) for reference.
- `pcc_harness.py` — loads `tensors/arg*.tensorbin`, runs `forward`, and compares the
  logits against `../golden_logits.pt`.
- `../golden_logits.pt` — CPU decode golden saved by the benchmark.

## MoE sharding

Emitted with the **good** expert sharding: experts dim sharded `{"_axis_0","_axis_1"}`
on mesh `["_axis_0"=4, "_axis_1"=8]`, i.e. natural placement (block b → device b),
matching the fixed `expert_mapping`. (The "bad" variant only differs in those three
expert weight tensors' axis order; the graph itself is byte-identical.)

## Running

The 80 input tensors (`tensors/arg0..arg79.tensorbin`, ~15 GB) are **gitignored** due to
size. Regenerate them with the tt-xla codegen export:

```
python -m pytest -s tests/benchmark/test_llms.py::test_glm_4_7_tp_galaxy_4_layers \
  --decode-only --codegen-py-export-path <this-dir>/..
```

then `python pcc_harness.py` from this directory.

Note: the emitted code needs the row-major workarounds for `mesh_partition` at
`optimization_level=1` (auto-inserted by the patched tt-mlir); KV-cache inputs must be
tilized before `forward` (see `pcc_harness.py`).
