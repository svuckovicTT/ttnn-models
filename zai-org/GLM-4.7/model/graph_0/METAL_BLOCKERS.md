# tt-metal blockers / negative results (GLM-4.7 attention tuning)

Things that were blocked or measured worse, with the op/config and why. None of
these are hard crashes in tt-metal; they are cases where the "obvious" optimization
did not help on this HW (4x8 WH galaxy) / shape (decode, batch 16).

## 1. ttnn.all_reduce slower than explicit reduce_scatter + all_gather

- **Where:** o_proj output all-reduce over the model axis (cluster_axis=1). Tried
  replacing `reduce_scatter(dim3) + reshape + all_gather(dim1)` with a single
  `ttnn.all_reduce(cluster_axis=1, topology=Linear)`.
- **Result:** PCC bit-identical (0.894531) but **+24 μs/layer slower**. `ttnn.all_reduce`
  decomposes to `reduce_scatter_minimal_async` (**102 μs**) + `all_gather` (80 μs); the
  `minimal_async` reduce-scatter is slower than the explicit `reduce_scatter` (78 μs),
  and the op count did not drop (the intermediate reshapes I removed are re-introduced
  internally). Reverted (`attn-perf #2` + revert).
- **Blocker:** there is no fused all-reduce on this HW/shape that beats the explicit
  reduce_scatter+all_gather pair. A faster fused line-all-reduce (or a `reduce_scatter`
  that matches the non-async kernel's speed) would be needed.

## 2. num_links does not speed up the reduce_scatter

- **Where:** o_proj `reduce_scatter`, `num_links=None`(=1) -> `3`.
- **Result:** stayed ~78 μs (noisy 78-99). The all_gather on the same axis improved
  80 -> ~55 μs with the same bump. The reduce_scatter carries a HiFi4 reduction
  (`compute_kernel_config` with fp32 acc); it appears reduction-serialized, not
  link-bound, so extra ethernet links don't help. Left reduce_scatter at auto.

## 3. Partial-RoPE TM ops are not removable as written (not a blocker, a constraint)

- `ttnn.experimental.rotary_embedding` on a `[16,12,1,64]` slice returns a seq-padded
  tile shape (dim2 1 -> 32), so the following `slice [.,.,.,0:64]` is **load-bearing**
  (strips the pad), not redundant -- removing it fails concat `shapes_match` TT_FATAL.
- `split_query_key_value_and_split_heads` requires rank-3 input, so the
  `[16,1792] -> [16,1,1792]` reshape after the qkv matmul is **load-bearing**
  (rank()==3 TT_FATAL if removed).
- These TM ops (~100 μs/layer total: rotary 66, reshapes 60, slices 41, concats 20)
  are the largest remaining non-matmul/non-CCL device time but cannot be folded at the
  python level given the current op contracts. A fused partial-RoPE op (slice+rotary+
  concat in one kernel, decode-seq-aware to avoid the 1->32 seq padding) would remove
  most of it -- worth a tt-metal op request.
