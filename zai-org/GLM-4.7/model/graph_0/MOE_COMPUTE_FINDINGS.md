# GLM-4.7 fused moe_compute integration — investigation findings (2026-06-18)

## Goal
Replace the hand-emitted MoE (all_to_all_dispatch + 3× sparse_matmul + all_to_all_combine)
in the GLM-4.7 4-layer WH-galaxy decode model with the fused
`ttnn.experimental.moe_compute` path, for a device-perf win (MoE MLP ≈ 88% of
~6.81 s/token in the baseline). Mesh (4,8), cluster_axis=0 (4-device dispatch ring,
8 replicated cols), 160 experts, experts_per_device=5, H=5120, N=1536, K=8, COL dispatch,
FABRIC_1D_RING.

## ROOT CAUSE (definitive)
**moe_compute's fused combine (`selective_reduce_combine` ring) deadlocks in the full
model.** Pinpointed via `synchronize_device` after each MoE sub-op:
`dispatch_metadata` syncs OK → `moe_compute` sync hangs. The deadlock is in the combine,
not dispatch, not the lm_head.

It is a **full-model-only interaction**: the standalone `moe_compute_smoke.py` (identical
dims/config, random weights) PASSES on every tt-metal version tried; the full model HANGS.

## What was RULED OUT (all still hang; smoke always passes)
- tt-metal version: base dev20260519 AND latest main (`6008ff55566`).
- PRs: `#45764` (moe_compute functional), `#46544` (dynamic mux buffers),
  `#46863` (dynamic core placement) — all cherry-picked/built/verified-in-`.so`, all still hang.
- cluster_axis 0 AND 1 (8-device ring reorientation; full row/col token reshard).
- topology Ring AND Linear; num_links 4 AND None.
- mux_core_range_set `((3,0),(4,7))` AND canonical `((1,1),(3,3))`.
- Per-forward fresh semaphores + dispatch prealloc (ruled out staleness).
- KV-cache L1 core overlap (SMOKE_KV_HOLD probe passed).
- dispatch_metadata: `GLM_SKIP_DISPATCH=1` (feed moe_compute the zero prealloc, no dispatch) → still hangs ⇒ dispatch exonerated.
- layer-3 attention: `GLM_SKIP_ATTN=1` → still hangs ⇒ attention exonerated.
- dense layers 0-2: `GLM_MOE_ONLY=1` → still hangs ⇒ dense layers exonerated.

⇒ The combine deadlock is triggered by the resident full-model device/weight state, and is
independent of every moe_compute version/config and of the surrounding model ops.

## THE BYPASS (works): moe_compute `compute_only=True`
The `#46863`-era moe_compute has `compute_only=True` (requires `cluster_axis=None`; no
combine, mux, topology, num_links, semaphore, output_tensor). It runs the **matmul only**
and returns matmul_output (slot 4). **It COMPLETES in the full model** — confirmed:
`SYNC after moe_compute(compute_only) OK`, matmul_output shape `(70, 2, 32, 5120)`. The
full forward (incl. the lm_head's axis-0 all_gather) then completes — confirming the
**fused combine was the sole deadlock**.

`compute_only` is designed/tested for single-card (1×1 mesh, no cross-device combine).
For our multi-device case the matmul output still needs a cross-device combine.

## OPEN GAP
Completing the MoE on multi-device needs a deadlock-free combine of compute_only's
matmul_output. Options, neither a clean drop-in:
- `selective_reduce_combine` (standalone) — same ring that deadlocks.
- hand-emitted `all_to_all_combine` — works on cluster_axis=0, but pairs with
  `all_to_all_dispatch` (different routing/format than `dispatch_metadata`/moe_compute).

## ESCALATION (for moe_compute owners: gajanan-choudhary / amorrison)
Precise repro: `selective_reduce_combine` ring deadlocks inside `moe_compute` (Full mode)
in the GLM-4.7 full model on a (4,8) WH galaxy, cluster_axis=0, while (a) the matmul
(`compute_only=True`) and (b) the isolated `moe_compute_smoke.py` both pass — version- and
config-independent (latest main + #46863). Branch + repro harness pushed.

## Diagnostic env gates (in model_ttnn.py)
- `GLM_MOE_PINPOINT=1` — synchronize_device + marker after each MoE sub-op.
- `GLM_MOE_ONLY=1` — run only the MoE layer (skip dense layers 0-2).
- `GLM_SKIP_ATTN=1` — skip layer-3 attention.
- `GLM_SKIP_DISPATCH=1` — feed moe_compute the zero prealloc (no dispatch_metadata).
- `GLM_MOE_COMPUTE_ONLY=1` — run moe_compute matmul-only (compute_only), placeholder combine
  (PERF/diagnostic only; values garbage).

## DEVICE PERF (tracy, compute_only matmul vs hand-emitted baseline)
Tracy profile of the compute_only path (latest-main+#46863 build), per MoE invocation:
- **MoEComputeDeviceOperation (fused matmul): ~50.6 µs**
- AllToAllDispatchMetadata: ~32.6 µs ; FastReduceNC: ~18.6 µs

Baseline hand-emitted MoE (baseline_profiles/L3_mlp.txt, L3 MoE segment):
- **SparseMatmul ×3 = 58,183 µs (86.4% of the MoE segment)** — flagged SLOW (LoFi, ~0.5% DRAM util)
- AllToAllDispatch 130 µs ; AllToAllCombine 683 µs ; MoeExpertTokenRemap 119 µs

⇒ moe_compute's fused bf4 matmul (~50 µs) replaces the hand-emitted 3× sparse_matmul
(~58 ms ≈ 86% of the MoE, and the MoE was ~88% of the ~6.81 s/token baseline) — a ~1000×
reduction on the dominant MoE-compute cost. The combine deadlock is the ONLY thing blocking
this win. Caveats: combine not included (it deadlocks); cross-build comparison (baseline from
the original session). For a rigorous same-build number, profile the hand-emitted good-pcc graph
on this build and compare.

## WORKING: moe_compute(compute_only) + Ring CCL combine (no deadlock) + FULL-MODEL PERF
The fused combine deadlocks even with all-Ring topology. The working path replaces it with
moe_compute(compute_only) matmul + a deadlock-free Ring `all_gather(cluster_axis=0)+sum` CCL
combine (GLM_MOE_COMPUTE_ONLY=1). Full model runs end-to-end (FORWARD DONE / SYNC DONE, no hang).
(Gotcha: to_memory_config(matmul_output, DRAM) first — its L1 shards clash with downstream CBs.)

Tracy full-model device perf (32 decode steps, moe_compute + CCL combine):
- **Total device time: 44.46 ms / decode step.** Dominant: AllGather 12.0 ms, Tilize 9.5 ms,
  Transpose 8.8 ms, Matmul 2.3 ms, LayerNorm 1.9 ms. moe_compute matmul ~50 us (negligible).
- Batch 64 tokens/step -> **~1,440 tokens/s device-bound**.
- vs hand-emitted (sparse_matmul ~58 ms/step) ~102 ms/step ~628 SPS -> **~2.3x full-model speedup**.

CAVEATS: (1) CCL combine is a PERF PROXY (reshape/slice of matmul_output, not the correct
per-expert-token combine) -> PCC NOT validated; a correct combine needs expert_token_counts +
the dispatch permutation (see validate_matmul in test_moe_compute_6U.py). (2) DEVICE time, un-traced
-> wall-clock is host-bound; needs metal trace to realize. (3) hand-emitted comparison is cross-build.

## Run env
docker `tt-xla-ird-mvasiljev`; `USE_TORCH_XLA=0 ACCELERATE_USE_XLA=false` required (else
transformers→accelerate→torch_xla→libTTMLIRRuntime.so ABI crash vs the rebuilt tt-metal).
Build-tree tt-metal at latest main + #46863 cherry-picked (detached HEAD).
