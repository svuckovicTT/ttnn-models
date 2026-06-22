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

## 1b. Decode RoPE / per-head RMSNorm: use the [1, batch, heads, head_dim] layout (HIGH impact)

**Commits:** `attn-perf #10` (RoPE), `#11` (q_norm). The codegen emitted the decode per-head
ops on a `[batch, heads, seq=1, head_dim]` layout. `rotary_embedding` (and `rms_norm`)
tile-pad dim2; with seq=1 in dim2 it pads 1->32, so the op does `batch*heads*32` work --
a 32x waste for a single decode token. Reordering to `[1, batch, heads, head_dim]` (heads in
dim2, padded 12->32 once, batch in dim1) cut:
- **Q rotary 58 -> 9 μs** (#10), **q_norm 36 -> 9 μs** (#11),
- and as a side effect the partial-RoPE **slices 41 -> 11 μs** and **concats 20 -> 5 μs**.

Total ~-120 μs/layer (the single largest structural win, ahead of the matmul DRAM-sharding).
Caveats handled: (a) the op requires `cos_seq_len >= heads`, so the position-specific cos/sin
`[1,1,1,head_dim]` must be replicated to `[1,1,32,head_dim]` (all heads share the decode
position); (b) build Q/K on this layout *before* norm+RoPE so the `q_for_sdpa`/`k_cache`
reshapes fold in for free. **Recommendation:** the codegen should emit decode-time per-head
RoPE and norms on the `[1, batch, heads, head_dim]` layout, matching the production decode
demos (gemma4, llama3_70b_galaxy), rather than the `[batch, heads, 1, head_dim]` layout that
forces a 32x seq-pad on every per-head op.

## 1c. Use nlp_create_qkv_heads_decode for the decode head split (HIGH impact)

**Commit:** `attn-perf #12`. The codegen emitted `split_query_key_value_and_split_heads`
(NlpCreateHeadsDeviceOperation, **39 μs**) producing `[batch, heads, 1, head_dim]`, which then
needs the `[.,.,1,d]->[1,batch,heads,d]` repack reshape (34 μs) for the efficient decode
layout (see 1b). Replacing both with a single `ttnn.experimental.nlp_create_qkv_heads_decode`
(num_heads, num_kv_heads, `memory_config=L1_HEIGHT_SHARDED`) emits the `[1, batch, heads,
head_dim]` layout **directly in 6 μs** and also subsumes the V reshape -> **-68 μs/layer**.
Notes: the op wants a `[1, 1, B, fused]` **L1** input (DRAM input hits a WH reader-alignment
path -- move to L1 first); it assumes Q,K,V fused order; outputs are L1 height-sharded (here
resharded to DRAM for the still-interleaved norms/RoPE -- 4 reshards ~17 μs, which an
end-to-end L1-sharded head-prep would remove). **Recommendation:** emit the decode head split
as `nlp_create_qkv_heads_decode` rather than `split_query_key_value_and_split_heads` +
layout reshape.

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
