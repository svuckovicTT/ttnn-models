# moe_compute cross-col CCL deadlock (smoke harness) — 2026-06-17

VALIDATED working via moe_compute_smoke.py (random GLM-shaped weights, no HF load):
- bf4 weight packing (prepare_w0_w1/prepare_w2, get_weight_core_shard_maps/get_weight_mem_configs), ShardTensorToMesh(dim0): packed w0w1 (384,1,5,2,5152,128), w2 (384,1,5,4,1568,128).
- all_to_all_dispatch_metadata: needs expert_mapping rank-2 [devices,experts] LINEARIZED (NOT one-hot [1,1,E,D] as docstring claims) and input batch-sharded along cluster_axis=0. Produces sparse (1,64,5120).
- moe_compute: needs dispatch_core_axis=COL at device open (else "tilize and matmul bounding boxes cannot overlap": tilize cores fixed at (5-6,8-9), DRAM matmul cores span whole 8x9 grid under ROW dispatch). Returns 6 tensors; combine_output (8,16,5120) per device. WORKS.

BLOCKER: ANY cluster_axis=1 (cross-col, 8-device) CCL immediately after moe_compute DEADLOCKS (hang, timeout) in the smoke harness:
- ttnn.reduce_scatter (Ring and Linear, rank-3 and rank-4, with/without HiFi4 compute_kernel_config): hang.
- ttnn.all_gather (Linear, dim=0): hang.
- ttnn.experimental.deepseek_moe_reduce_scatter (the ring-native op, valid L1 NdShard 5-core x128 config, 8 input tensors): hang.
Tried and did NOT help: sync after moe_compute, dealloc of moe_compute aux outputs (0,1,2,4), batch-sharded vs replicated dispatch inputs, fast_reduce_nc before the CCL.
The SAME reduce_scatter(cluster_axis=1,Linear) WORKS in the existing hand-emitted MoE (after all_to_all_combine), so it is specific to following moe_compute's fused combine. The deepseek TG reference (test_optimized_moe_decode_block_tg.py) runs moe_compute->reduce_scatter(cluster_axis=1) successfully on deepseek dims, so it is possible; the smoke harness has an unidentified setup difference (mux/semaphore/drain-core/mesh-orientation). Next: integrate into the full model and test the cross-col reduce in the real fabric context.

# moe_compute -> cluster_axis=0 CCL deadlock IN THE FULL MODEL — 2026-06-17 (CONFIRMED)

moe_compute is fully integrated (params.py bf4 weights + lin expert_mapping; model_ttnn.py
dispatch_metadata -> moe_compute -> epilogue; main.py COL dispatch). With per-op stderr
markers the forward reaches, in order: layer 0/1/2 done (dense), moe dispatch_metadata done,
moe_compute done, moe reduce_scatter done, **layer 3 done** (full MoE layer incl. shared
experts completes with correct data flow), lm_head matmul done -> then HANGS at the lm_head
`all_gather_16` (dim=0, cluster_axis=0).

ROOT CAUSE (robust, deterministic): moe_compute uses cluster_axis=0 for its fused
dispatch+combine. After it, ANY cluster_axis=0 CCL deadlocks (the combine leaves the axis-0
fabric routers active / a non-terminating combine program — a bare ttnn.synchronize_device
placed after the MoE layer ALSO hangs, proving pending non-draining device work). In contrast
ALL cluster_axis=1 CCLs after moe_compute work fine (MoE epilogue reduce_scatter+all_gather,
shared-expert reduce_scatter+all_gather, dense layers' CCLs) because the dense layers warm up
axis-1 and moe_compute never touches it.

The model's lm_head needs BOTH axes: all_gather_16 (dim0,axis0=gather batch across 4 rows),
all_gather_17 (dim2,axis1=gather vocab across 8 cols), mesh_partition_2 (dim0,axis0). So
whichever axis moe_compute uses, the lm_head's ops on that axis deadlock -> unavoidable with
this model structure.

Ruled OUT as fixes: lm_head all_gather topology Ring vs Linear; lm_head num_links 4 vs auto;
moe_compute topology=Ring + num_links=4; synchronize_device between MoE and lm_head (itself
hangs); USE_TORUS_MODE=1 env (demo-only, not read in tt-metal core — only selects
FABRIC_1D_RING/Ring-topology which were already set).

Why the deepseek TG reference doesn't hit this: it runs on TG8X4 (cluster_axis=0 = 8 dispatch
devices, vs our (4,8) = 4), and after moe_compute it does ONLY a cluster_axis=1 reduce then
RETURNS — it never issues a cluster_axis=0 CCL after moe_compute. So the post-moe-compute
axis-0 CCL path appears genuinely untested/unsupported in this tt-metal build.

Needs (tt-metal side or design): proper teardown/drain of moe_compute's axis-0 fabric before a
subsequent same-axis CCL (a fabric barrier/reset), OR running moe_compute's dispatch on an axis
the rest of the model doesn't collective on (not possible here — lm_head uses both axes), OR a
sub-device isolation for moe_compute's fabric. moe_compute NUMERICS are not yet PCC-validated
because the forward can't complete to produce logits.

# all_reduce_async not a drop-in for the RS+AG all-reduce pattern — 2026-07-06 (perf iter7)

Goal: collapse each cluster_axis=1 reduce_scatter+all_gather TP all-reduce pair (o_proj x92,
dense down, MoE epilogue x89, shared experts x89 — ~344 us/MoE-layer, ~134 us/attn-layer of CCL)
into one ttnn.experimental.all_reduce_async. The pybind __doc__ advertises a simple
(input_tensor, cluster_axis, mesh_device) form, but the ACTUAL overloads all require, in
addition: math_op (ttnn.reduction.ReduceType, no default) AND three global-semaphore sequences
(barrier_semaphores, rs_global_semaphores, ag_global_semaphores). i.e. it is the async/persistent
variant needing explicit semaphore lifecycle management (create-once, per-op, matching link
count), exactly the class of state that caused the moe_compute combine hang. The sparse tt-metal
checkout under tt-mlir has no tests/models to copy the correct semaphore counts from. Abandoned
as too hang-risky for the reward without a reference; the RS+AG pair is left intact.
Alternatives if revisited: llama_rs_matmul / matmul_reduce_scatter_async (fuse the matmul with
the RS) — same semaphore requirement; or a synchronous all_reduce if one exists in a later build.
