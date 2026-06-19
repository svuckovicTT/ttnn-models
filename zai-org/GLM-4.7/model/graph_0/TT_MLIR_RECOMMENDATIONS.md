# tt-mlir codegen recommendations (from GLM-4.7 attention hand-tuning)

Improvements worth generalizing in the tt-mlir EmitTTNN codegen, each with the
hand-applied commit(s) here and the per-op device-time gain measured on the 4x8
(32-chip) WH galaxy decode slice (batch 16, hidden 5120). Baseline attention
~620 μs/layer; tuned ~538 μs/layer (-13%), PCC 0.894531 -> 0.902344.

## 1. DRAM-shard skinny decode projection matmuls (HIGH impact)

**Commits:** `attn-perf #5` (qkv), `attn-perf #4` (o_proj) + `dram_matmul.py`.

For decode (M=16, padded to 1 tile) the projection matmuls are pure DRAM-bandwidth
bound on the weight read. The codegen currently emits a plain
`ttnn.matmul`/`ttnn.linear` with `program_config=None`, DRAM-interleaved weight:
- qkv (k=5120, n=1792): **107 μs**, 90 GB/s, **31% DRAM BW util**, HiFi2.
- o_proj (k=1536, n=5120): 47 μs, 175 GB/s, 60.7% BW util.

Width-sharding the weight across the 12 DRAM banks + an L1-width-sharded activation +
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` (config math copied verbatim
from `models/demos/llama3_70b_galaxy/tt/model_config.py`: `find_grid_k_n`,
`create_dram_sharded_mem_config`, `dram_matmul_config`) gave:
- qkv: **107 -> 47 μs** (67.6% BW util), the single biggest win (~-60 μs/layer).
- o_proj: 47 -> 40 μs (67.7% BW util).

Cost: an `interleaved->L1-width-sharded` reshard on the activation (~1-2 μs) and an
`L1-sharded->interleaved` reshard on the output (~2-3 μs). The weight resharding is a
const-eval (one-time, zero per-step cost).

**Recommendation:** for decode-shape (small-M) projection matmuls reading a
DRAM-interleaved weight, the codegen should (a) emit the weight in a DRAM-width-sharded
layout, and (b) emit the `DRAMShardedProgramConfig` matmul with the activation
width-sharded into L1. The grid/config math is a closed form of (k, n, num_cores).
Even better: keep the whole attention block in L1-sharded layouts end-to-end so the
reshards disappear (the llama3_70b_galaxy path keeps activations sharded between ops).

## 2. Bias on a DRAM-sharded matmul

**Commit:** `attn-perf #5`. The qkv linear has a bias; the DRAM-sharded path here runs
the matmul unbiased and adds the bias with a separate `ttnn.add` (BinaryNg, ~4 μs) after
resharding to interleaved. If the codegen fuses bias into the DRAM-sharded matmul (where
supported) it saves the extra eltwise op + keeps the output sharded.

## 3. CCL num_links for pure-data-movement collectives

**Commit:** `attn-perf #7`. The o_proj all-reduce is emitted as `reduce_scatter` +
`all_gather` with `num_links=None` (auto -> 1 link). Galaxy fabric supports up to 4
links. Bumping the **all_gather** to `num_links=3` cut it **80 -> ~55 μs** (-20 μs/layer).
The **reduce_scatter** did NOT benefit (it carries a HiFi4 reduction; stayed ~78 μs,
noisier). **Recommendation:** the codegen should set `num_links` > 1 on transport-bound
collectives (all_gather / all_broadcast), but leave reduction-carrying collectives
(reduce_scatter) at the lower link count — they are not link-bound here.

## 4. Do NOT fuse this reduce_scatter+all_gather into all_reduce

See METAL_BLOCKERS.md #1. `ttnn.all_reduce` decomposes into
`reduce_scatter_minimal_async` (102 μs) + `all_gather` (80 μs), which is *slower* than
the explicit `reduce_scatter` (78 μs) + `all_gather`. The codegen's explicit pair is the
better choice on this HW/shape; an all_reduce-fusion pass would regress here.
