# tt-mlir / codegen recommendations from GLM-4.7 decode perf tuning

Improvements worth generalizing in the tt-xla EmitTTNN codegen (or tt-metal), each
found by hand-tuning this 4-layer GLM-4.7 decode graph. Perf is tracy device time,
per-segment (signpost-scoped), PCC held at 0.992188 (== the good-routing baseline).

## 1. Width-shard the decode hidden-state rms_norm (BIGGEST WIN)  — iters 7, 8, 10

The codegen emits `ttnn.rms_norm(x[16,5120], memory_config=DRAM, program_config=None)`
for every hidden-state layernorm (input_layernorm, post_attention_layernorm, final norm).
On a decode step the token dim is a single tile-row (16 -> 1 tile), so the default DRAM
program factory runs the entire 160-tile-wide reduction on **ONE core** (~191 us each).

Fix (hand-applied `sharded_rms_norm` in model_ttnn.py): reshard the [16,5120] input to an
8-core (4x2) WIDTH_SHARDED L1 tensor and run the sharded multicore layernorm
(`ttnn.LayerNormShardedMultiCoreProgramConfig(grid=(4,2), block_h=1, block_w=20,
subblock_w=1)` via `create_layernorm_program_config`). This parallelizes the reduction
~8-way: ~191 us -> ~20 us on device, **bit-identical** output (micro-bench PCC 0.999997;
full-model PCC unchanged at 0.992188). The gamma weight can stay DRAM-interleaved (no
reshard needed). A 4x2 block on cols 0-3 avoids the COL-dispatch-reserved grid column x=7.
Measured: attention block -170 us/layer, dense MLP -173 us/layer.

RECOMMENDATION: codegen should emit the sharded layernorm program config for hidden-size
rms_norm on small-token-count (decode) inputs, instead of the default single-core DRAM path.
Caveat: the sharded path needs a 2D physical [32,5120] layout, so a `[16,1,5120]` input
(which tile-pads to physical height 512) must be flattened to `[16,5120]` first — the
codegen already had a spurious `[16,1,5120]` reshape feeding the MoE norm that should just
be `[16,5120]` (iter8).

## 2. Merge sibling TP all-reduces before the reduce (linearity)  — iter 9

The MoE emits two independent cluster_axis=1 all-reduces (reduce_scatter+all_gather over
the 8 columns): one for the sparse-expert epilogue, one for the shared-expert output, then
adds the two replicated results. Because all-reduce is linear,
`allreduce(sparse) + allreduce(shared) == allreduce(sparse + shared)`: add the two
column-partial [1,1,16,5120] tensors first, then do ONE reduce_scatter+all_gather. Removes
a whole RS+AG pair per MoE layer. Exact (PCC unchanged).

RECOMMENDATION: a codegen pass that recognizes `add(all_reduce(a), all_reduce(b))` on the
same cluster_axis and rewrites to `all_reduce(add(a,b))` (generally: hoist elementwise ops
that are linear over the reduction out through sibling collectives). Applies anywhere two
TP-partial tensors are separately all-reduced and then combined.

## 4b. CONFIRMED: the MoE dispatch/combine TM churn is unavoidable in-graph (needs #4)

Empirically verified (perf iter14, reverted): moe_compute emits its combine output in
ROW_MAJOR, and all_to_all_dispatch consumes ROW_MAJOR, while the rest of the graph is TILE.
This forces, per MoE layer: an Untilize (~38us) + FillPad (~46us) to feed dispatch, and a
TilizeWithValPadding (~89us) to bring the [8,16,5120] combine back to TILE for the epilogue
scale+sum. Attempted to dodge the epilogue tilize by doing the scale (multiply) + k-reduction
(sum) in ROW_MAJOR and tilizing only the small [1,16,5120] result -- but ttnn's multiply/sum
tilize internally (compute needs TILE), so the 89us tilize just moves into the multiply; net
~0 (PCC stayed 0.992188). So ~170us/MoE-layer (~15ms full model) of pure format churn cannot
be removed in-graph. It needs EITHER a TILE-accepting all_to_all_dispatch + a TILE-output
moe_compute, OR a fused moe-combine op that takes the per-expert output + routing scores and
returns the TILE weighted-sum directly (deepseek_moe_fast_reduce_nc_fused is close but reduces
by expert index across devices, not a local scale-by-scores+sum-over-k). This is the single
largest remaining MoE cost after the fused op's own (irreducible bf4 expert) FLOPs.

## 3. all_reduce_async pybind doc vs reality (blocker, see METAL_BLOCKERS.md #iter7)

`ttnn.experimental.all_reduce_async`'s docstring advertises a simple
`(input, cluster_axis, mesh_device)` call, but every real pybind overload also requires
`math_op` (ReduceType) and three global-semaphore sequences (barrier/rs/ag). Either the
docstring should show the real required args, or a managed (semaphore-allocating) convenience
overload should be provided — the current gap makes the fused all-reduce impractical to adopt
from codegen without a reference for the semaphore counts.

## 4. TILE-accepting all_to_all_dispatch / moe_compute (open, see TUNING_LOG TM section)

The MoE dispatch/compute boundary forces ROW_MAJOR<->TILE conversions
(Tilize ~89us + FillPad ~94us + Untilize ~41us per MoE layer) because
`all_to_all_dispatch_metadata` / `moe_compute` consume ROW_MAJOR while the surrounding
graph is TILE. A TILE-accepting dispatch API (or an L1-sharded op chain across the boundary)
would remove ~220 us/MoE-layer of pure format churn. This is the largest remaining
non-moe_compute MoE cost and needs a tt-metal-side change.

## 4c. Attention RoPE: fused rotary_embedding_llama_fused_qk is convention-incompatible

The fused QK rotary (rotary_embedding_llama_fused_qk) would collapse the partial-RoPE churn
(RotaryEmbedding 63us + Slice 47us + Concat 20us = ~130us/attn-layer, ~12ms full) into one op.
BUT it uses Meta-style INTERLEAVED rotary ([r,i,r,i], trans_mat swaps adjacent pairs), while
GLM/HF uses rotate-half (the codegen emits slice[0:64]/slice[64:128]+concat = halves). Bridging
needs dim permutes (halves<->interleaved) that add TM back and carry high PCC risk. Not a clean
win unless the model is regenerated to emit interleaved-layout QKV weights (a tt-xla/codegen
choice). RECOMMENDATION: if codegen can emit weights in interleaved RoPE layout, the fused_qk
op becomes usable and removes the slice/concat churn.

## 4d. Attention o_proj all-reduce: fusable but needs sub-device + persistent-semaphore infra

o_proj's reduce_scatter+all_gather (cluster_axis=1 TP all-reduce, ~140us/attn-layer) could be
one ttnn.experimental.all_reduce_async. The usable (llama3_70b_galaxy/tt/llama_ccl.py) overload
needs a persistent output buffer + a cycled multi_device_global_semaphore + a worker
subdevice_id. Setting up a sub-device risks colliding with moe_compute's core usage (the
deadlock-prone op), for ~2% reward -- deferred. RECOMMENDATION: a managed ttnn.all_reduce that
allocates its own semaphores/buffer (like the sync reduce_scatter/all_gather do) would make this
adoptable from codegen without sub-device plumbing.

## 5. lm_head argmax gathers the full vocab (open)

The lm_head all-gathers the full 151552-wide logits to every device (AllGather ~11.3 ms,
70% of the lm_head segment) purely to run a global argmax. A local-argmax-per-vocab-shard +
tiny (value,index) all-gather + index-offset combine would avoid gathering the full vocab.
NOTE: this benchmark's PCC check compares the full replicated logits (outputs[-1]), so the
full vocab gather can't be dropped without also changing what the harness verifies; in a real
decode (only the argmax token is needed) it is pure waste.
