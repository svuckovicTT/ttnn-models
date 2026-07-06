# GLM-4.7 4-layer decode — MoE compute tuning log

Baseline (branch mvasiljevic/glm-4.7-wh-galaxy):
- WH-working (50ab52b): PCC 0.859375.
- expert_mapping fix (5b54f15): PCC 0.894531.

Goal: replace hand-emitted MoE (router+topk+all_to_all_dispatch+moe_expert_token_remap
+3x sparse_matmul+all_to_all_combine) with fused ttnn.experimental.moe_compute, get good
PCC, then tune. Perf metric = tracy device time (run -t + tt-perf-report), not TPS.

## Model structure (GLM-4.7, from HF config.json)

num_hidden_layers=92, first_k_dense_replace=3 => 3 dense layers (0-2) + 89 MoE layers
(3-91). Plus 1 embedding + 1 final-norm/lm_head. The 1 MTP layer (num_nextn_predict_layers)
is dropped by codegen. Dims: hidden 5120, dense interm 12288, moe interm 1536, 160 experts
/ 8 per tok / 1 shared expert, 96 q-heads / 8 kv-heads / head_dim 128, vocab 151552.

The 4-layer test model (layers 0-2 dense + layer 3 MoE) holds EXACTLY one of each segment
type, so full-model device time extrapolates cleanly:
  full_DT = preamble + lm_head + 92*attn + 3*dense_mlp + 89*moe_mlp

## Signposts (setup, commit)

tracy `signpost()` markers in model_ttnn.py: `preamble` (embedding+rotary+attn-mask, once),
per-layer `L{i}_attn` / `L{i}_mlp`, `lm_head` (final norm+lm_head+all_gathers, once). Scope
with `tt-perf-report --start-signpost X --end-signpost Y`. Per-segment summary txts saved to
baseline_profiles/. Use the FIRST total ("... signposts  N μs ...") = device FW duration;
percentages in the stacked report are relative to it. (The 2nd huge number is unreliable.)

## BASELINE device perf (2026-06-17, branch glm-4.7-perf-tuning, PCC 0.894531)

Single decode forward, tracy run, ops_perf_results_2026_06_17_12_41_28.csv:

| segment    | DT (us) | full count | full DT (us) | % full |
|------------|---------|------------|--------------|--------|
| preamble   |     126 | x1         |          126 |  0.00% |
| attention  |   8,547 | x92 (avg L0-3) |    786,324 | 11.55% |
| dense MLP  |     595 | x3  (avg L0-2) |      1,785 |  0.03% |
| MoE MLP    |  67,381 | x89        |    5,996,909 | 88.10% |
| lm_head    |  22,084 | x1         |       22,084 |  0.32% |
| **TOTAL**  |         |            | **6,807,228** | 100%  |

Per-layer measured: L0_attn 8508, L1_attn 8583, L2_attn 8581, L3_attn 8516 (us);
L0_mlp 626, L1_mlp 581, L2_mlp 579 (dense); L3_mlp 67381 (MoE).

KEY: MoE MLP = 88% of full-model device time => prime target (moe_compute fusion).
Attention = 11.6%, dominated by PointToPointOp (KV-cache distribute p2p, ~86% of each
attn block). dense MLP + lm_head + preamble together < 0.4%.

| # | patch | scope | tracy DT delta | PCC delta | decision | why |
|---|-------|-------|----------------|-----------|----------|-----|
| 0 | add tracy signposts (preamble/L*_attn/L*_mlp/lm_head) + baseline profile | setup | baseline = 6.807s full est | 0.894531 (hold) | keep | enables per-segment scoping + full-model extrapolation |
| 1 | FABRIC_1D -> FABRIC_1D_RING (prereq for moe_compute CCL) | full | tbd | hold @0.894? | tbd | ring topology required by moe_compute/dispatch_metadata |

## PCC investigation (before perf tuning) — 2026-06-16

Goal was good PCC. Findings (all on cached golden, mesh 4x8):
- expert_mapping column-major fix: 0.858 -> 0.894531 (committed 5b54f15). The
  gain is purely from making expert PLACEMENT (_arrange_experts column-major) and
  ROUTING (expert_mapping) consistent. Verified my mapping == reference
  get_linearized_mesh_coord(cluster_axis=0) exactly.
- experts bf8->bf16: 0.894531 (no change). all weights bf8->bf16: 0.894531 (no
  change). => precision is NOT the limiter.
- matmuls already run HiFi4 (29 compute_kernel_config sites). => fidelity maxed.
- row-major layout (identity _arrange_experts + row-major expert_mapping): also
  0.894531, bit-identical to column-major. => both self-consistent layouts are
  equivalent; routing is correct.
- MoE CCL op attributes are coherent (axis0=batch/expert dispatch+combine,
  axis1=tensor-parallel reduce_scatter/all_gather, shared by attn/dense TP).
  Swapping cluster_axis would break TP. Not an in-place bug.

Conclusion: expert routing is correct; 0.894531 is the robust accumulated-fidelity
ceiling of this 4-layer bf8 decode with correct MoE (matches codegen authors note
that ~0.86 is genuine bf8 fidelity). Issue-5096 0.99 is the upstream full-precision
benchmark, not this bf8 codegen. Reaching 0.99 in-place would need graph
regeneration from the corrected tt-xla sharding spec or higher precision throughout.
Next: moe_compute fused-op integration (perf goal; replaces dispatch+matmul+combine
with a structurally-correct fused op).

## 2026-06-29 — fused moe_compute + good-PCC routing: full-model perf estimate ~0.33 s/token

Branch `mvasiljevic/glm-moe-compute-l1-goodpcc` (fused moe_compute default, routing
fix = global iota + native gather, all_to_all_dispatch inputs in L1). PCC 0.992 (all
64 users), no deadlock (build carries combine fix #45764). tracy `./run -t`, signpost-
scoped, tt-perf-report device-merged (per-device == wall-clock under SPMD). Methodology
matches the pre-moe_compute baseline (preamble 126.6μs == baseline 126μs).

Per-segment device time (μs), now vs baseline (commit 35b98b0, hand-emitted MoE):
| segment            | now (fused) | baseline | note |
|--------------------|-------------|----------|------|
| preamble           |     126.6   |    126   | unchanged |
| attention / layer  |     812     |   8547   | 10.5x: good-PCC head-sharded KV dropped ~1792 axis-0 KV P2P ops |
| dense MLP / layer  |     641     |    595   | unchanged |
| MoE / layer (fused)|    2810     |  67381   | **24x**: ttnn.experimental.moe_compute replaces dispatch+token_remap+3 sparse_matmul+combine |
| lm_head (+tail)    |    3779     |  22084   | argmax-over-vocab dominated |

Full-model extrapolation (full = preamble + lm_head + 92*attn + 3*dense + 89*moe):
- **~330.7 ms = 0.331 s/token** (was ~6.81 s/token) → **~20x faster end-to-end**, at PCC 0.992.
- Composition: MoE **76%** (89 x 2.81ms = 250ms), attention **23%** (92 x 812μs = 75ms),
  lm_head ~1%, dense ~1%, preamble ~0%.

Next perf levers (now that MoE is no longer 88%): MoE is still the top at 76% — moe_compute
internals (num_links / mux / fused-combine vs compute_only) + prune the now-dead embedding
chain still left in the L3 router (superseded by the native gather; drops an all_gather_10).
Then attention (23%). lm_head argmax (~per-device) is the 3rd item. Note the MoE segment time
includes the full router (router gate + topk + group-mask + the dead embedding path) + dispatch
+ epilogue, not just the moe_compute op itself (~0.4ms) — most of the 2.81ms is glue/CCL.

## Tuning ledger (branch mvasiljevic/glm-moe-compute-l1-goodpcc-perf, baseline = fused moe_compute 0.300... see iter0)
| # | patch | scope | tracy DT (full est) | PCC | decision | why |
|---|-------|-------|---------------------|-----|----------|-----|
| 0 | baseline: fused moe_compute + routing fix + L1 dispatch | full | 0.331 s/tok (moe 2.81ms, attn 0.81ms, dense 0.64ms, lmhead 3.78ms) | 0.992188 | base | starting point |
| 1 | prune dead embedding chain in L3 router (drop all_gather_10 + token*160+expert index matmul + ttnn.embedding) | MoE | 0.300 s/tok (moe 2.46ms; -12% moe, -9% full) | 0.992188 | keep | dead code superseded by native ttnn.gather |
| 2 | remove vacuous group-mask branch (n_group=1): drop topk(2)+topk(1), concat_25 all_gather_9, scatter, mesh_partition_0, repeat_interleave(160), ne, where; top-8 directly on biased scores | MoE | 0.250 s/tok (moe 1.90ms; -23% moe, -17% full) | 0.992188 | keep | group-limited routing is a no-op for n_group=1 |
| 3 | rms_norm HiFi4->HiFi2 (all per-layer norms) | MoE+attn | 0.250 s/tok (no change; LayerNorm stayed 192us) | 0.992188 | REVERT | rms_norm is DRAM-bound not fidelity-bound; reverted. (PCC floor set to 0.99 per user, kept.) |
| 4 | num_links=3 on MoE epilogue reduce_scatter + all_gather (cluster_axis=1) | MoE | 0.247 s/tok (moe 1869us; -1.5% moe, -1% full) | 0.992188 | keep | parallelize cross-col TP reduce; small win (decode tensors are small) |
| 5 | num_links=3 on all other cluster_axis=1 CCLs (attention/dense/shared/lm_head) | full | 0.251 s/tok (WORSE +1.6%: lm_head vocab all_gather +713us, moe +37us variance) | 0.992188 | REVERT | num_links=3 hurts the large lm_head vocab all_gather; attn/dense gains within noise |

## Summary after iters 1-5 (branch mvasiljevic/glm-moe-compute-l1-goodpcc-perf)
Full-model device-time estimate: **0.331 -> 0.247 s/token (-25%)**, PCC held at 0.992188 (>= 0.99 floor).
Significant wins (both structural / dead-code, MoE segment 2.81 -> 1.87 ms):
- iter1: prune dead embedding chain (all_gather_10 + token*160+expert index matmul + ttnn.embedding) — superseded by the native ttnn.gather. -0.031 s/tok.
- iter2: remove the vacuous group-mask branch (n_group=1: topk(2)+topk(1), concat_25 all_gather_9, scatter, mesh_partition, repeat_interleave(160), ne, where). -0.050 s/tok.
- iter4: num_links=3 on the MoE epilogue cluster_axis=1 reduce_scatter+all_gather. -0.003 s/tok.
Washes / reverted: iter3 (rms_norm HiFi2 — DRAM-bound, no gain), iter5 (num_links=3 on remaining axis1 CCLs — regressed the large lm_head vocab all_gather).
Remaining levers (bigger, need restructuring; PCC margin is tight at 0.992 vs 0.99 floor so precision trades are mostly off the table):
- MoE TM churn (~400us: Reshape/FillPad/Tilize/Untilize around dispatch/moe_compute) — DRAM-bound, needs L1-sharded op chains.
- LayerNorm (MoE 192us + attn 236us, DRAM-bound) — needs L1-sharded input.
- MoEComputeDeviceOperation 421us — would need moe_compute internal knobs (num_links / mux) or the compute_only path (not exact PCC).
- Attention (30% of full): qkv/o_proj matmuls already DRAM-sharded; remaining is the 3 rms_norms (DRAM-bound) + CCL.
| 6 | fold MoE router TMs: remove identity reshape_73 [16,8]->[16,8]; single multiply_3->[16,1,1,8] reshape (was 2x) | MoE | 0.247 s/tok (moe 1866us; -2.7us, flat) | 0.992188 | keep | cleaner; confirms cheap reshapes are ~free |

## TM investigation (non-L1) conclusion
Scanned all MoE+attention TM ops for reorder/fuse/commute (no L1):
- FOLDABLE reshapes (identity / duplicate) are essentially FREE on device (iter6 moved 0). The ReshapeView cost is the few genuinely-shape-changing ones (e.g. matmul_output->[4480,5120]); those can't be folded.
- The EXPENSIVE MoE TMs are Tilize(88us)+FillPad(93us)+Untilize(41us)=~222us, ALL at the dispatch(ROW_MAJOR)<->compute(TILE) boundary: disp_x untilizes post_normed[16,5120]; moe_compute combine output re-tilizes; dispatch idx/scores untilize. Each is a ONE-WAY conversion feeding a specific op (verified: no cancellable TILE<->ROW_MAJOR inverse pair in the chain), so reorder/commute cannot remove them. They need either L1-sharded op chains (the skill's last item) OR a moe_compute/all_to_all_dispatch API that accepts TILE inputs (a tt-metal change -> TT_MLIR_RECOMMENDATIONS).
- Attention TMs (Slice/Concat from partial-RoPE) were found load-bearing in prior attention tuning (see top of this log) and don't fold.
So: non-L1 TM headroom here is ~0; the real TM win requires L1-sharding or a TILE-accepting dispatch.

## 2026-07-06 — fresh baseline (post-iter6) + resumed tuning (branch ...-perf, HEAD 361c79d)

Re-profiled HEAD on the current build. Per-segment device-FW (tt-perf-report, signpost-scoped, 32-device merged):
| segment | device time | note |
|---------|-------------|------|
| attention / layer | 799 us | LayerNorm 238us(3 norms, input_layernorm ~191us on 1 CORE) + Matmul 158us(qkv+o_proj) + RS 85 + AG 49 (CCL 134) + RoPE 63 + reshapes/slices |
| dense MLP / layer | 602 us | dominated by the single-core post_attn norm + gate/up/down + CCL |
| MoE / layer | 1901 us | MoECompute 415 + RS 220 + AG 124 (CCL 344) + Matmul 194(router+shared) + LayerNorm 193 + TM churn (Reshape150/FillPad94/Tilize89/Untilize41) + TopK 99 + dispatch 82 |
| lm_head (x1) | 16,195 us | argmax-over-vocab dominated |
Full-model extrapolation (preamble + 92*attn + 3*dense + 89*moe + lm_head) = ~260.7 ms. Composition: MoE 65%, attention 28%, lm_head 6%.
KEY finding: the hidden-state rms_norms run on ONE core (whole 160-tile-wide reduction serialized) -> ~191us each, the single fattest addressable op.

| # | patch | scope | tracy DT | PCC | decision | why |
|---|-------|-------|----------|-----|----------|-----|
| 7-abandon | all_reduce_async to fuse RS+AG TP all-reduce | (attempt) | n/a (API error) | n/a | ABANDON | pybind needs math_op + barrier/rs/ag global-semaphore sequences (async/persistent variant); no reference in sparse checkout for correct semaphore counts; too hang-risky. See METAL_BLOCKERS.md |
| 7 | width-sharded rms_norm (8-core 4x2 WIDTH_SHARDED LayerNormShardedMultiCore) at input_layernorm(x92)+post_attn(dense x3, MoE x89) | attn+dense+moe | attn 799->629 (-170), dense 602->429 (-173), moe 1901->1867 (-34); full ~260.7->~241.6 ms (-7.3%) | 0.992188 (== baseline, bit-identical) | keep | single-core 191us norm -> ~20us on 8 cores. Micro-validated (micro_norm.py PCC 0.999997). MoE gain small because its [16,1,5120] input forced reshape detours (fixed in iter8) |
