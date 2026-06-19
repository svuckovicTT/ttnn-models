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
