# tt-metal issues to file (found during GLM-4.7 decode perf tuning)

These are tt-metal limitations that block in-graph perf wins (documented so they can be
filed as GitHub issues on tenstorrent/tt-metal). Each has an observed behavior, a repro
anchor, and a proposed change. Build: tt-metal under tt-xla third_party (moe_compute fix
#45764 present; checkout HEAD 13adda80c11).

---

## ISSUE 1 — `moe_compute` combine output is ROW_MAJOR regardless of the output tensor's declared TILE layout (forces a ~90us tilize per MoE layer)

**Severity:** perf (blocks ~15ms/token of format churn on GLM-4.7 decode, 4x8 galaxy).

**Op:** `ttnn.experimental.moe_compute(..., optional_output_tensor=T, optional_cross_device_semaphore=sem, cluster_axis=0, mux_core_range_set=...)`

**Observed:** The fused-combine output (`mc_outs[-1]`, shape [k=8, tokens=16, hidden=5120]
per device) is written in ROW_MAJOR element order. Downstream the epilogue must scale it by
the routing weights and sum over k, which needs TILE, so a `TilizeWithValPaddingDeviceOperation`
runs on the full [8,16,5120] (128 tile-rows) = ~90us/layer (measured, tt-perf-report).

Tried to avoid it by allocating the `optional_output_tensor` in **TILE** layout
(`ttnn.moreh_full(..., layout=ttnn.Layout.TILE)`) so moe_compute writes TILE directly. Result:
the op ACCEPTS the TILE tensor and runs without error, but writes ROW_MAJOR-ordered bytes into
the TILE buffer -> the data is mis-packed -> **model PCC drops 0.9922 -> 0.8008**. So the op
ignores the output tensor's layout and always emits ROW_MAJOR.

Also tried doing the epilogue scale(multiply)+reduce(sum) in ROW_MAJOR and tilizing only the
small [1,16,5120] result: ttnn `multiply`/`sum` tilize internally (compute needs TILE), so the
90us tilize just relocates into the multiply — net ~0. So the tilize is unavoidable in-graph.

**Repro:** `moe_compute_smoke.py` (this dir) sets up the full bf4 dispatch+moe_compute path with
random GLM-shaped weights. Add a variant that allocates `optional_output_tensor` with
`layout=ttnn.Layout.TILE` vs `ROW_MAJOR` and compares the combine output against a torch
reference — TILE gives ~0.80 correlation, ROW_MAJOR gives ~1.0.

**Proposed change (either):**
1. Make moe_compute honor the `optional_output_tensor`'s layout — pack the combine output to
   TILE when the provided tensor is TILE; OR
2. Add a `combine_output_layout=TILE` option that tilizes inside the op (cheaper than a separate
   downstream tilize, and lets the op fuse it with the combine writeback).

Either removes ~90us/MoE-layer (~8ms/token here) with zero PCC change.

---

## ISSUE 2 — `all_to_all_dispatch_metadata` hard-requires ROW_MAJOR input (forces a ~38us untilize + ~46us FillPad per MoE layer)

**Severity:** perf (part of the same ~15ms/token MoE format churn; pairs with Issue 1).

**Op:** `ttnn.experimental.all_to_all_dispatch_metadata(disp_x, disp_idx, disp_scores, expert_mapping, cluster_axis=0, ...)`

**Observed:** the dispatch device op asserts `input_tensor.layout() == ROW_MAJOR`:
```
TT_FATAL: Input tensor must be in row major layout
ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_dispatch_metadata/device/all_to_all_dispatch_metadata_device_operation.cpp:23
```
So the token activations (`disp_x`), which arrive from the norm in TILE, must be untilized to
ROW_MAJOR before dispatch: a `UntilizeWithUnpadding` (~38us) + a `FillPad` (~46us, 70 cores) per
MoE layer (measured). Combined with Issue 1's tilize, ~170us/MoE-layer (~15ms/token) is pure
ROW_MAJOR<->TILE format conversion around the dispatch/moe_compute pair.

**Repro:** in the full model set `disp_x` to `ttnn.Layout.TILE` before the dispatch -> the assert
above fires immediately. (moe_compute_smoke.py exercises the same dispatch call.)

**Proposed change:** allow `all_to_all_dispatch_metadata` to accept TILE input (the token scatter
is a row/gather over the token dim; a TILE-aware reader would avoid forcing callers to untilize).
Together with Issue 1 (TILE combine output), the whole MoE would stay TILE end-to-end and the
~170us/layer of Tilize/Untilize/FillPad churn disappears.

---

## moe_compute knobs tested (2026-07-08)
- `optional_output_tensor` layout=TILE -> ISSUE 1 (writes RM into it, PCC 0.80). Not usable.
- `num_links=2` on the fused combine -> **DEADLOCK** (forward wedges at "moe_compute enqueued",
  no completion in 12min; 241% host CPU spinning). The fused selective_reduce_combine is
  hang-sensitive to link count; leave num_links unset. (Logged in HANGS.md.)
- `num_shared_experts_per_device` (feature #46509 "proper TP sharding for shared expert in
  moe_compute" IS in the checkout) — the biggest remaining moe_compute lever: it would absorb
  the shared expert into the fused op, removing the 3 shared-expert matmuls (~74us/layer post
  iter11) + the merge-add (~7ms/token). NOT attempted: requires packing the shared-expert
  weights into moe_compute's bf4 expert format (params.py) + feeding them as extra experts, and
  the #46509 semantics/layout would need validation. Documented as the top follow-up (needs a
  weight-prep change, not a graph change). `mux_core_range_set` enlarge and `topology` are the
  same hang-risk class as num_links; not pursued.

## Summary
Both ISSUE 1 (moe_compute emits ROW_MAJOR, ignores TILE output tensor) and ISSUE 2
(all_to_all_dispatch requires ROW_MAJOR input) force the GLM-4.7 decode MoE to bounce
ROW_MAJOR<->TILE around the fused dispatch/compute pair: ~170us/MoE-layer, ~15ms/token
(89 MoE layers), ~7% of the 199ms/token model. Fixing both (TILE end-to-end through
dispatch+moe_compute) is the single largest remaining perf win and needs no model/graph change,
only the two tt-metal ops accepting/producing TILE.
