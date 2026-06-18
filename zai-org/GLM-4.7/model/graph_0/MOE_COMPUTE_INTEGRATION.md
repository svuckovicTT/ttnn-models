# moe_compute integration reference (confirmed against build tree)

Build tree (authoritative): `/data/mvasiljev/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal`
(host path; docker path `/home/mvasiljev/...`). Reference: `models/demos/deepseek_v3/tests/tg_moe_tests/test_optimized_moe_decode_block_tg.py`.

## GLM-4.7 layer-3 MoE params
hidden H=5120, intermediate N=1536, experts=160, experts_per_tok k=8, mesh (4,8),
cluster_axis=0, num_dispatch_devices=4, num_replicated_devices=8, experts_per_device=5,
experts_per_cluster=experts/num_replicated=160/8=20. tokens_per_device=16 (batch 64 sharded /4),
total_tokens = batch=64 (=16*4 dispatch devices). seq=1 (decode).

## moe_compute signature (build tree nanobind)
```
ttnn.experimental.moe_compute(
  tilize_input_tensor, tilize_expert_indices_tensor, tilize_expert_scores_tensor,
  tilize_expert_mapping_tensor, matmul_w0_w1_tensor, matmul_w2_tensor,   # 6 positional
  layer_id=, output_height_shard_dim=, intermediate_size=, cluster_axis=,  # required kw
  has_bias=False, topology=None, num_links=None, mux_core_range_set=None,
  output_memory_config=None, optional_output_tensor=None,
  optional_cross_device_semaphore=None, activation_type=None)             # optional kw
```
Returns 6 (python unpacks): (per_expert_total_tokens, expert_activation, e_t, _tile_out,
matmul_output, combine_output). USE combine_output (last). It is [k=8, tokens_per_device=16, H=5120]
per device, ROW_MAJOR. moe_compute FUSES the cluster_axis=0 combine; cross-col (axis1) reduce is the epilogue.

## all_to_all_dispatch_metadata signature (build tree)
```
ttnn.experimental.all_to_all_dispatch_metadata(
  input_tensor, expert_indices_tensor, expert_scores_tensor, expert_mapping_tensor,  # 4 pos
  cluster_axis=, num_links=4, drain_sync_tilizer_core=None, worker_mode=ttnn.WorkerMode.DIRECT,
  dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_MCAST_SHORTEST_PATH,
  output_tensors=<3-tuple prealloc>, cross_device_semaphore=<GlobalSemaphore>)
```
Returns (sparse_buffer, indices, scores). input [B,S,1,H] row-major per device (B=tokens_per_device=16,S=1).
indices/scores [B,S,1,K] row-major. **expert_mapping = ONE-HOT [1,1,E,D]** (== existing consteval.expert_mapping_u16 / var_2).

## TWO expert_mapping formats (CONFIRMED DIFFERENT)
- dispatch_metadata: one-hot [1,1,160,32] uint16 = existing `var_2`. REUSE.
- moe_compute: linearized-coord [num_devices=32, experts=160] uint16, replicated. NEW.
  Built via get_linearized_mesh_coord(num_replicated=8, cluster_axis=0, e, experts_per_cluster=20, experts_per_device=5)
  = device_id_within_cluster*8 + cluster_id where cluster_id=e//20, dev_in_cluster=(e%20)//5.
  VERIFIED == params.py column-major device_of_expert = row*8+col (block=e//5,row=block%4,col=block//4). Same placement.
  ttnn.from_torch([32,160] uint16, ROW_MAJOR, DRAM, ShardTensor2dMesh(dims=(None,None)) = replicated).

## Weights (bf4, packed). In params.py on torch tensors.
get raw GLM experts (HF stacked): gate_proj/up_proj [160,1536,5120]=(E,out=N,in=H); down_proj [160,5120,1536]=(E,out=H,in=N).
prepare_w0_w1 wants torch_w0 (L,E,K=H,N): per-expert gate transposed [H=5120,N=1536]. So w0 = gate.transpose(-1,-2).
prepare_w2 wants torch_w2 (L,E,N,K=H): per-expert down transposed [N=1536,H=5120]. So w2 = down.transpose(-1,-2).
Flow (from test L638-699):
```
w0w1_map, w2_map, dram_crs = get_weight_core_shard_maps(mesh_device, H=5120, N=1536)
w0w1_per_dev=[None]*32; w2_per_dev=[None]*32
for e in range(0,160,5):                         # group experts_per_device=5 consecutive
  w0 = cat([gate_T[e+j].view(1,1,H,N) for j in 5], dim=1)  # (L=1, E=5, H, N)
  w1 = cat([up_T[e+j].view(1,1,H,N) ...], dim=1)
  w2 = cat([down_T[e+j].view(1,1,N,H) ...], dim=1)         # (L=1, E=5, N, H)
  w0w1_r = prepare_w0_w1_tensor_for_moe_compute(w0,w1, L=1,E=5,K=5120,N=1536, w0w1_map)
  w2_r   = prepare_w2_tensor_for_moe_compute(w2, L=1,E=5,N=1536,K=5120, w2_map, w0w1_map)
  d = get_linearized_mesh_coord(8,0,e,20,5)               # == column-major dev
  w0w1_per_dev[d]=w0w1_r; w2_per_dev[d]=w2_r
torch_w0w1 = cat(w0w1_per_dev, dim=0); torch_w2 = cat(w2_per_dev, dim=0)
w0w1_mem, w2_mem, _, _ = get_weight_mem_configs(1, 5, 5120, 1536, w0w1_map, w2_map, dram_crs)
tt_w0_w1 = from_torch(torch_w0w1, device, TILE, bfloat4_b, w0w1_mem, ShardTensorToMesh(dim=0))
tt_w2    = from_torch(torch_w2,   device, TILE, bfloat4_b, w2_mem,   ShardTensorToMesh(dim=0))
```

## Setup objects (need device; create in model init / consteval)
- compute_grid = mesh_device.compute_with_storage_grid_size(); worker_cores = CoreRangeSet(full grid).
- dispatch_sem = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)
- combine_sem  = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)
- combine_mux_cores = ttnn.CoreRangeSet([ttnn.CoreRange(CoreCoord(3,0), CoreCoord(4,7))])  # ((3,0),(4,7))
- output_height_shard_dim = 4 (from test parametrize).
- dispatch prealloc (3-tuple), persistent, ShardTensor2dMesh(dims=shard_dims=(0,None) for cluster_axis=0):
  - sparse_buffer: zeros [num_dispatch_devices=4, total_tokens=64, H=5120] bf16 ROW_MAJOR DRAM.
  - indices: [4,64,8] uint16, L1 HEIGHT_SHARDED, shard spec on compute_tilize_drain_core=CoreCoord(6,9), shard [total_tokens=64, k=8].
  - scores: [4,64,8] bf16, same L1 HEIGHT_SHARDED shard spec.
- combine prealloc (per-forward, zeroed): ttnn.moreh_full([k=8, tokens_per_device=16, H=5120], 0, device, ROW_MAJOR, bf16, DRAM).

## Forward swap (A2aSparseMLPWithSharedExperts.forward), keep router+topk + shared experts:
Inputs available from router: expert indices [16,8] int32 (ttnn_typecast_40), topk scaled weights.
1. Build dispatch inputs (per device, B=16,S=1):
   - x: post_normed reshaped [16,1,1,5120] bf16 ROW_MAJOR (replicated on cols).
   - idx: [16,1,1,8] uint16 ROW_MAJOR.
   - scores: [16,1,1,8] bf16 ROW_MAJOR (the normalized*routed_scaling topk weights).
2. dispatch_metadata(x, idx, scores, var_2 one-hot, cluster_axis=0, num_links=4, output_tensors=prealloc, cross_device_semaphore=dispatch_sem) -> sparse,disp_idx,disp_scores.
3. moe_compute(sparse, disp_idx, disp_scores, expert_mapping_lin, tt_w0_w1, tt_w2, layer_id=0,
   output_height_shard_dim=4, intermediate_size=1536, has_bias=False, cluster_axis=0,
   mux_core_range_set=combine_mux_cores, optional_output_tensor=combine_prealloc,
   optional_cross_device_semaphore=combine_sem, activation_type=SiLU?) -> ...combine_output [8,16,5120].
   NOTE activation: GLM uses SiLU(gate)*up (SwiGLU). Check MoEActivationFunction enum (SILU vs SWIGLU). default None.
4. Epilogue (-> sparse_output [16,5120] full hidden to match shared-expert add):
   to_layout TILE; unsqueeze(dim=1) -> [8,1,16,5120]; scores_perm = topk weights as [8,1,16,1];
   mul -> [8,1,16,5120]; sum(dim=0) -> [1,16,5120] (or keepdim) ; this is partial over THIS device's experts.
   cross-col all-reduce (cluster_axis=1): reduce_scatter(dim=-1,cluster_axis=1,Ring) + all_gather(dim=-1,cluster_axis=1)
   -> full hidden 5120 replicated across cols; reshape [16,5120] = sparse_output.
   (Current code already does reduce_scatter+all_gather over cols; mirror it.)
5. shared experts: UNCHANGED. moe_output = sparse_output + shared_ag. return moe_output.

## Risks / watch
- activation_type: confirm SiLU/SwiGLU enum name in ttnn.experimental (MoEActivationFunction not on ttnn top-level).
- topk score ordering must match combine_output k-dim (dispatch reorders scores alongside; epilogue in test uses ORIGINAL pre-dispatch scores permuted -> so use original [16,8] topk weights, not dispatched).
- bf4 experts lower precision than current bf8 -> PCC may dip slightly; baseline ceiling 0.894, tol allows >=0.88ish. exact_pcc var in main.py = 0.855 with tol 0.01.
- the test computes fast_reduce_nc but RETURNS the sum+reduce_scatter path; fast_reduce is the optimized variant (moe.py forward_decode uses it for ring+tp8). Start with sum+reduce_scatter+all_gather for correctness.
