# GLM-4.7 attention-block perf tuning

Branch: `mvasiljevic/attention-perf` (forked from tt-xla codegen commit `df754b6`).
Scope: **attention block only** (`Glm4MoeAttention.forward`), the per-layer sub-block
wrapped by signposts `attn_L{idx}_start` / `attn_L{idx}_end`.

## Environment

- Container: `tt-xla-ird-mvasiljev`, user `4123:4123`.
- Env: `cd /home/mvasiljev/tt-xla && source venv/activate` (only venv with
  torch/transformers/huggingface_hub). Sets `TT_MLIR_HOME` to tt-xla's nested
  tt-mlir; runtime tt-metal = `.../tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal`
  (tracy `ENABLE_TRACY=ON`).
- Helpers (NFS-shared, host `/data/mvasiljev` == container `/home/mvasiljev`):
  - `glm_pcc.sh`   — `GLM_CHECK_PCC=1 ./run`  (PCC vs CPU golden)
  - `glm_tracy.sh` — `./run -t`  (tracy profile → `generated/profiler/reports/<ts>/ops_perf_results_*.csv`)
- perf report: `tt-perf-report` v1.2.2 on host. Scope a signpost window with
  `tt-perf-report <csv> --start-signpost attn_L0_start --end-signpost attn_L0_end --summary-file out.txt`.
- Build note (2026-06-19): runtime tt-metal `.so` was **stale** vs source; rebuilt
  with `ninja` (relinked `_ttnn.so`/`_ttnncpp.so`/`libtt_metal.so`) before baselining.

## Model shape (this slice)

4-layer decode slice: 3 dense + 1 MoE, batch 16, hidden 5120, 12 Q heads,
1 KV head, head_dim 128 (partial RoPE on first 64), on a 4x8 (32-chip) galaxy mesh.
KV cache head-sharded across mesh columns. PCC compared with tol 0.01 to golden.

## Metric

Per-iteration: sum of the 4 attention-window device times (tracy, scoped per
signpost) + full-slice device time. Full-model impact extrapolated linearly
(attention block is identical across all real-model layers).

## Summary of significant changes (kept)

| Change | Op effect | Attention/layer |
|--------|-----------|-----------------|
| #5 DRAM-sharded qkv matmul | qkv matmul 107 -> 47 μs (31% -> 67.6% DRAM BW) | -53 μs |
| #7 o_proj all_gather num_links=3 | all_gather 80 -> ~55 μs | -22 μs |
| #4 DRAM-sharded o_proj matmul | o_proj matmul 47 -> 40 μs (60.7% -> 67.7% BW) | -8 μs |
| **Total (kept)** | | **~620 -> ~538 μs/layer (-13%)** |

PCC 0.894531 -> **0.902344** (improved; the DRAM-sharded matmuls auto-select LoFi but
accumulate more accurately on this path). Stopped after 8 iterations: remaining device
time (reduce_scatter 78 μs transport-bound; rotary 66 μs / TM ~100 μs from partial-RoPE,
both load-bearing per METAL_BLOCKERS) is at HW limits or below the ~±20 μs/layer noise
floor. The generalizable win (DRAM-shard skinny decode matmuls) is in TT_MLIR_RECOMMENDATIONS.

Full-model extrapolation: the attention block is identical across all real-model layers,
so the -82 μs/layer scales linearly with layer count (e.g. ~-7.5 ms across 92 layers),
independent of the MoE layers that dominate this 4-layer slice's 156 ms device time.

## Candidate optimizations (pre-analysis of Glm4MoeAttention.forward)

1. **Redundant post-rotary re-slice** — after `rotary_embedding` on the 64-wide
   slice, the output is re-sliced `[0:64]` (`q_rotary_sliced`, `k_rotary_sliced`),
   which is a no-op slice (input already 64 wide). Candidate: drop both slices
   (2 ops/layer x 4 = 8 ops).
2. **reduce_scatter + all_gather (o_proj)** — dim3/cluster_axis1 reduce_scatter
   then dim1/cluster_axis1 all_gather = an all-reduce. Check for a fused
   `all_reduce` / `all_gather`-matmul variant that's cheaper on galaxy.
3. **Math fidelity / kernel config** — qkv `linear` and o_proj `matmul` pass
   `compute_kernel_config=None` (default LoFi). rms_norms use HiFi4. Tune per
   perf-report impact.
4. **TM minimization** — the slice/rotary/slice/concat partial-RoPE pattern and
   the several reshapes (qkv reshape, v reshape, k reshape, q-for-sdpa reshape).
   Look for fold/cancel opportunities and a partial-rotary fused op.
5. **Tile alignment / sharded variants** — SDPA-decode, paged_update_cache L1
   shard specs; check perf-report advice columns.

## Baseline profile (2026-06-19, fresh build)

Per-layer attention device time (tracy, scoped per signpost): **L0 632 / L1 621 / L2 619 / L3 623 μs**
(L0 inflated op-to-op = first-call warmup). Use **~621 μs/layer** as the clean baseline.
Whole 4-layer slice device time: **155,885 μs** (dominated by the single MoE layer; attention
is a small fraction of this slice but is identical across every real-model layer, so attention
optimizations scale linearly with layer count in the full model).

Per-op Device Time within one attention window (sums to 621 μs):
- ReduceScatter (o_proj CCL) 78 μs + AllGather (o_proj CCL) 80 μs = **158 μs (25%)** — rs+ag == all-reduce.
- qkv_proj Matmul 32x5120x1792 **107 μs (17%)** — flagged SLOW, DRAM-bound (90 GB/s, 31% BW, HiFi2).
- RotaryEmbedding Q 58 μs (12 heads) vs K 9 μs (1 head).
- o_proj Matmul 32x1536x5120 47 μs. NlpCreateHeads 39 μs. q_norm 36 / k_norm 9 μs. SdpaDecode 23 μs.
- TM: 3 Reshapes 62 μs + 6 Slices 41 μs + 2 Concats 20 μs = **123 μs (20%)**.

Top levers: CCL fusion (all_reduce / matmul_reduce_scatter), qkv matmul knobs, TM folding,
2 redundant post-rotary re-slices.

## Ledger

| # | patch | scope | tracy DT delta | PCC delta | decision | why |
|---|-------|-------|----------------|-----------|----------|-----|
| 0 | baseline (signposts only) | — | — | PCC 0.894531 | keep | establishes baseline; matches golden tol 0.01 |
| 1 | drop post-rotary re-slice (q/k) | attn | — | run FAILED | revert | hypothesis wrong: rotary_embedding output != [.,.,.,64], concat shapes_match TT_FATAL. Re-slice is load-bearing (strips rotary output seq-pad 1->32). Committed broken + reverted. |
| 2 | o_proj reduce_scatter+all_gather -> all_reduce | attn | +24 μs/layer (L1 621->645) WORSE | PCC 0.894531 (bit-identical) | revert | ttnn.all_reduce decomposes to reduce_scatter_minimal_async (102 μs) + all_gather (80 μs); the minimal_async RS is slower than the explicit reduce_scatter (78 μs). Op count unchanged (25). Explicit rs+ag is better here. |
| 3 | drop qkv reshape [16,1792]->[16,1,1792] before split_heads | attn | — | run FAILED | revert | hypothesis wrong: qkv linear outputs 2D [16,1792]; split_query_key_value_and_split_heads requires rank 3 (TT_FATAL input_shape.rank()==3). Reshape adds the seq dim (real re-tile) -> load-bearing. Reverted (uncommitted). |

| 4 | DRAM-sharded o_proj matmul (+L1-sharded in0, reshard out) | attn | -8 μs/layer (L1 621->613) | PCC 0.894531->0.902344 (better) | **keep** | o_proj matmul 47->40 μs, now DRAM-bound 67.7% BW (was 60.7%), auto-LoFi. +2 reshard ops (~4 μs). Net positive; PCC improved. Weight width-sharded across 12 DRAM banks in consteval. |

| 5 | DRAM-sharded qkv matmul (+L1-sharded in0, reshard out, separate bias add) | attn | -53 μs/layer (L1 613->560; -61 vs baseline) | PCC 0.902344 (better) | **keep** | qkv matmul **107->47 μs**, DRAM-bound 67.6% BW (was 31%), auto-LoFi. +bias add (BinaryNg 4 μs) + 2 reshards (~4 μs). Biggest single win. Both attn matmuls now DRAM-bound ~68%. |

| 6 | o_proj reduce_scatter + all_gather num_links=3 | attn | mixed (all_gather 80->~60, reduce_scatter flat/noisy) | PCC 0.902344, no hang | superseded by #7 | links help the pure-DM all_gather but not the HiFi4-reduction reduce_scatter. |
| 7 | keep all_gather num_links=3, reduce_scatter back to auto | attn | -22 μs/layer on clean layers (L2/L3 560->538) | PCC 0.902344 | **keep** | isolates the all_gather win (80->~55 μs, 15 cores); reduce_scatter back to ~78 μs (5 cores). L1 is a systematic ~35 μs noisy outlier across runs -> use L2/L3 as clean signal. |

| 8 | o_proj reduce_scatter compute HiFi4 -> LoFi | attn | none (reduce_scatter 78->78-100, within noise) | PCC 0.902344 (unchanged) | revert | confirms reduce_scatter is transport-bound, not compute-bound: lowering reduction fidelity does not speed it (and num_links didn't either, #6). It is at its floor on this HW. |

| 9 | fuse K+V paged_update_cache -> paged_fused_update_cache (V resharded to rows 2-3) | attn | -4 μs/layer (10->6 μs, within noise) | PCC 0.902344 | **keep** | fused op requires K/V on non-overlapping cores. Small win; validates the fused-op path for the head-prep restructure. |

| 10 | partial-RoPE on [1,batch,heads,head_dim] layout (+ cos/sin replicated to [1,1,32,64]) | attn | **-91 μs/layer** (L2/L3 538->447); Q rotary **58->9 μs** | PCC 0.902344 (correct + unchanged) | **keep** | rotary_embedding tile-pads dim2; with heads(12) in dim2 instead of seq(1) the 32x seq-pad waste is gone. Folded the q_for_sdpa / k_reshaped reshapes into the head-prep. Biggest win after qkv DRAM-shard. |

### Running total: attention ~620 -> ~447 μs/layer on clean layers (**-28%**), PCC 0.894531 -> 0.902344 (improved).

### Larger restructure (in progress): fused-op / efficient-layout head-prep

Head-prep (create_heads + q/k-norm + partial-RoPE slices/concats + reshapes) is ~50% of
the block (~270 μs). The rotary alone is 66 μs because GLM applies it to a
[batch=16, heads=12, seq=1, 64] tensor: `rotary_embedding` tile-pads the seq dim 1->32, so
it does 16*12*32 work. The gemma4 decode template (same HW, also partial-RoPE + qk-norm)
applies RoPE on the [1, batch, heads, head_dim] layout, where the padded dim is heads
(12->32) with batch in dim1 -> ~12x less rotary work. Plan: reorder the partial-RoPE onto
[1,batch,heads,head_dim] (iter10); then adopt nlp_create_qkv_heads_decode for a sharded
head layout end-to-end; then matmul_reduce_scatter for o_proj.

Kept changes: (#4) DRAM-sharded o_proj matmul, (#5) DRAM-sharded qkv matmul + bias add,
(#7) o_proj all_gather num_links=3. Dominant win is #5 (qkv 107->47 μs). Measurement noise
floor is ~±20 μs/layer (CCL run-to-run variance), so sub-20 μs tweaks below are not reliably
attributable; the reduce_scatter (78 μs), rotary (66 μs) and the partial-RoPE TM reshapes/
slices (~100 μs) are the remaining device time but sit near HW limits / the noise floor.

### Interim finding (after iters 1-3)

The attention block is already tightly generated: the three "obviously redundant" TM ops
(post-rotary re-slice, qkv-rank reshape) are all load-bearing, and the one CCL fusion
(all_reduce) is slower than the explicit reduce_scatter+all_gather it would replace.
Remaining device time is dominated by **DRAM-bandwidth-bound matmuls** (qkv 107 μs at a
90 GB/s / 31%-util wall; o_proj 47 μs) and **near-optimal CCL** (158 μs). The textbook fix
for the matmul wall is **DRAM-sharded weights + L1-width-sharded activation dataflow**
(cf. models/demos/llama3_70b_galaxy, same HW), which the skill sequences last because it is
a whole-block restructure (reshard in/out around each matmul, or keep the block sharded
end-to-end). That is the next lever.
