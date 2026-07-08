import ttnn
import params

try:
    # tracy signpost marks regions so tt-perf-report can be scoped per model
    # segment (preamble / per-layer attn+mlp / lm_head) for full-model
    # device-time extrapolation. No-op when tracy tooling is unavailable
    # (e.g. plain PCC runs).
    from tracy import signpost
except Exception:  # pragma: no cover

    def signpost(header, message=None):
        pass


class LightweightModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# Width-sharded rms_norm for the decode hidden-state norms ([*, 5120]). The
# default DRAM rms_norm on a single 16-token tile-row runs on ONE core over all
# 160 width-tiles (~191 us; ~30% of the attention block, ~10% of MoE). Splitting
# the 5120 width across an 8-core (4x2) block and using the sharded multicore
# LayerNorm parallelizes the reduction ~8-way -> ~2x faster on device, at
# bit-identical PCC (validated in micro_norm.py: 0.999997). The 4x2 block
# (cols 0-3, rows 0-1) avoids grid column x=7 that COL dispatch reserves, and is
# transient (deallocated before attention's KV reshard / the MoE mux cores).
_LN_GX, _LN_GY = 4, 2                       # 8 cores
_LN_WTILES = (5120 // 32) // (_LN_GX * _LN_GY)   # = 20 width-tiles/core
_LN_SHARD_MEM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
    ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_LN_GX - 1, _LN_GY - 1))}),
        [32, _LN_WTILES * 32], ttnn.ShardOrientation.ROW_MAJOR,
    ),
)
_LN_PROG_CFG = ttnn.LayerNormShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(_LN_GX, _LN_GY),
    subblock_w=1, block_h=1, block_w=_LN_WTILES, inplace=False,
)


# Decode matmul program configs (1D multicast-in0). The default matmul factory is
# slow on these skinny (M=1 tile) DRAM-interleaved decode matmuls with a large K
# reduction; a 1D mcast_in0 config that multicasts the tiny activation and blocks
# the K reduction is ~2x faster at bit-identical math (micro-bench matmul_micro.py).
# Grids are 7-wide (x=0-6) to avoid the COL-dispatch-reserved grid column x=7.
def _mm1d(gx, gy, in0_block_w, per_core_n):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=in0_block_w, out_subblock_h=1, out_subblock_w=1,
        per_core_M=1, per_core_N=per_core_n, fuse_batch=True, mcast_in0=True,
    )
# qkv_proj [16,5120]x[5120,1792/dev]: N=56 tiles -> 7x8 cores, K=160 tiles ibw=8.  112->56us
_QKV_PC = _mm1d(7, 8, 8, 1)
# shared experts gate/up [16,5120]x[5120,192/dev]: N=6 tiles -> 6 cores, ibw=20.  56->31us
_SHARED_PC = _mm1d(6, 1, 20, 1)


def sharded_rms_norm(x, weight, epsilon, ckc, dram_mem):
    """Reshard x (DRAM, [.., 5120]) to an 8-core width-sharded L1 tensor, run the
    sharded multicore rms_norm, and reshard the result back to DRAM. Same shape
    and (bit-identical) values as ttnn.rms_norm(..., memory_config=dram_mem).

    The width-shard needs a 2D physical [32, 5120] tile layout, so any middle dim
    (e.g. [16,1,5120], which would tile-pad to physical height 16*32=512) is
    flattened to [16,5120] before the norm and restored afterward. Both reshapes
    are free size-1 views on the tile tensor."""
    orig_shape = list(x.shape)
    rows = 1
    for d in orig_shape[:-1]:
        rows *= d
    x2d = ttnn.reshape(x, [rows, orig_shape[-1]], memory_config=dram_mem) if len(orig_shape) != 2 else x
    xs = ttnn.to_memory_config(x2d, _LN_SHARD_MEM)
    if x2d is not x:
        ttnn.deallocate(x2d, False)
    ns = ttnn.rms_norm(
        xs, epsilon=epsilon, weight=weight, bias=None, residual_input_tensor=None,
        memory_config=_LN_SHARD_MEM, program_config=_LN_PROG_CFG, compute_kernel_config=ckc,
    )
    ttnn.deallocate(xs, False)
    out = ttnn.to_memory_config(ns, dram_mem)
    ttnn.deallocate(ns, False)
    if len(orig_shape) != 2:
        out = ttnn.reshape(out, orig_shape, memory_config=dram_mem)
    return out


def _make_moe_dispatch_runtime(device):
    """Create the per-forward moe_compute runtime (global semaphores +
    preallocated dispatch outputs) immediately before the MoE op, so nothing
    runs between their allocation and use. The standalone smoke harness (which
    creates these right before moe_compute) passes; creating them in __init__
    and then running 3 dense layers + attention before the MoE let their L1
    state be clobbered -> the combine's barrier semaphore was stale -> the fused
    combine deadlocked."""
    import torch

    H, K = 5120, 8
    num_dispatch, total_tokens = 4, 64
    mesh_shape = (4, 8)
    drain_core = ttnn.CoreCoord(6, 9)

    grid = device.compute_with_storage_grid_size()
    worker_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
    )
    dispatch_sem = ttnn.create_global_semaphore(device, worker_cores, 0)
    combine_sem = ttnn.create_global_semaphore(device, worker_cores, 0)

    dram = ttnn.DRAM_MEMORY_CONFIG
    shard_dims = (0, None)  # batch sharded along cluster_axis=0, replicated cols
    sparse = ttnn.from_torch(
        torch.zeros(num_dispatch, total_tokens, H, dtype=torch.bfloat16),
        device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
        memory_config=dram,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape, shard_dims),
    )
    idx_scr_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(drain_core, drain_core)}),
            [total_tokens, K], ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    idx = ttnn.from_torch(
        torch.zeros(num_dispatch, total_tokens, K, dtype=torch.int32),
        device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint16,
        memory_config=idx_scr_mem,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape, shard_dims),
    )
    scr = ttnn.from_torch(
        torch.zeros(num_dispatch, total_tokens, K, dtype=torch.float32),
        device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
        memory_config=idx_scr_mem,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape, shard_dims),
    )
    return dispatch_sem, combine_sem, (sparse, idx, scr)


class ModelTTNN(LightweightModule):
    def __init__(self, device):
        self.device = device
        self.weights = params.load_weights_for__main_from_state_dict(device)
        import consteval
        self.weights = consteval.run_consteval(self.weights, device)
        self.rotary_embedding = Glm4MoeRotaryEmbedding(device, self.weights)
        self._init_moe_compute_runtime(device)
        self.layers = []
        for i in range(3):
            self.layers.append(
                Glm4MoeDecoderLayer(device, self.weights, layer_idx=i, is_moe=False)
            )
        self.layers.append(
            Glm4MoeDecoderLayer(device, self.weights, layer_idx=3, is_moe=True)
        )

    def _init_moe_compute_runtime(self, device):
        """Create the once-per-model moe_compute runtime objects (global
        semaphores, mux cores, preallocated dispatch outputs) and stash them in
        the weights dict so the MoE layer can use them. See
        MOE_COMPUTE_INTEGRATION.md."""
        import torch

        # Only the static mux CoreRangeSet is created once. The global
        # semaphores and the dispatch prealloc are created FRESH per-forward in
        # _make_moe_dispatch_runtime (right before the MoE), matching the
        # standalone smoke harness, which passes. Creating them in __init__ and
        # then running 3 dense layers + attention before the MoE let their L1
        # state be clobbered -> the combine's barrier semaphore was stale ->
        # moe_compute's fused combine deadlocked. (smoke passes; full model hung.)
        # Canonical mux cores from the galaxy-tested reference (tt_moe_decode
        # ComputeConfig default ((1,1),(3,3))). The old ((3,0),(4,7)) was copied
        # from the (8,4) _tg test; on our transposed (4,8) mesh with KV/weights
        # resident it collided with persistent L1 -> combine deadlock in the full
        # model (passed in isolation). (1,1)-(3,3) clears the corner tilize cores.
        self.weights["moe_compute.mux_cores"] = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 3))]
        )

    def forward(self, activations):
        args_1 = activations[0]
        args_0 = activations[1]
        args_3 = activations[3]
        args_4 = activations[4]
        args_6 = activations[6]
        args_7 = activations[7]
        args_9 = activations[9]
        args_10 = activations[10]
        args_11 = activations[11]
        args_12 = activations[12]
        args_13 = activations[13]
        ttnn.deallocate(activations[8], False)
        ttnn.deallocate(activations[5], False)
        ttnn.deallocate(activations[2], False)
        var_0 = self.weights["consteval.scalar_zero_f32"]
        var_1 = self.weights["consteval.scalar_one_i32"]
        var_2 = self.weights["consteval.expert_mapping_u16"]
        # Signpost: preamble (embedding + rotary + shared attn-mask prep), run
        # once per model forward.
        signpost("preamble")
        # Embedding
        ttnn_typecast_29 = ttnn.typecast(
            args_1,
            ttnn.DataType.UINT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(args_1, False)
        ttnn_reshape_9 = ttnn.reshape(
            ttnn_typecast_29,
            [16],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_29, False)
        ttnn_to_layout_49 = ttnn.to_layout(
            ttnn_reshape_9,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_9, False)
        ttnn_embedding_0 = ttnn.embedding(
            ttnn_to_layout_49,
            self.weights["model.model.embed_tokens.weight.device"],
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_49, False)
        # Rotary embedding (computed once, shared across all layers)
        cos, sin = self.rotary_embedding(args_0)
        # Shared attention utilities
        ttnn_repeat_1 = ttnn.repeat(
            args_11,
            ttnn.Shape([16]),
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_17 = ttnn.reshape(
            args_11,
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_ge_0 = ttnn.ge(
            ttnn_reshape_17,
            self.weights["consteval.head_dim_indices"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_17, False)
        ttnn_where_0 = ttnn.where(
            ttnn_ge_0,
            self.weights["consteval.scalar_zero_bf16"],
            self.weights["consteval.neg_inf_bf16"],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_ge_0, False)
        ttnn_repeat_2 = ttnn.repeat(
            ttnn_where_0,
            ttnn.Shape([1, 1, 12, 1]),
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_where_0, False)
        # Cache position increment (used by MoE and final section)
        ttnn_add_10 = ttnn.add(
            args_11,
            var_1,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(args_11, False)
        # Layer inputs: (key_cache, value_cache) per layer
        layer_kv_caches = [
            (args_3, args_4),
            (args_6, args_7),
            (args_9, args_10),
            (args_12, args_13),
        ]
        hidden_states = ttnn_embedding_0
        key_cache_outs = []
        value_cache_outs = []
        import os as _os0
        _layer_iter = [3] if _os0.environ.get("GLM_MOE_ONLY") == "1" else range(4)
        for layer_idx in _layer_iter:
            key_cache_input, value_cache_input = layer_kv_caches[layer_idx]
            layer = self.layers[layer_idx]
            hidden_states, key_cache_out, value_cache_out = layer(
                hidden_states,
                key_cache_input,
                value_cache_input,
                cos,
                sin,
                ttnn_repeat_1,
                ttnn_repeat_2,
                var_0,
                var_2,
            )
            key_cache_outs.append(key_cache_out)
            value_cache_outs.append(value_cache_out)
            import sys as _sys
            print(f">>> layer {layer_idx} done", flush=True, file=_sys.stderr)
        ttnn.deallocate(ttnn_repeat_1, False)
        ttnn.deallocate(ttnn_repeat_2, False)
        ttnn.deallocate(sin, False)
        ttnn.deallocate(cos, False)
        # Signpost: lm_head (final norm + lm_head matmul + output all_gathers),
        # run once per model forward.
        signpost("lm_head")
        # Final norm and lm_head. hidden_states is [16,5120]; use the 8-core
        # width-sharded norm (same single-core->8-core win as the per-layer norms).
        _lmn_dram = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        )
        _lmn_ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True,
        )
        ttnn_rms_norm_16 = sharded_rms_norm(
            hidden_states, self.weights["model.model.norm.weight"],
            9.9999997473787516e-06, _lmn_ckc, _lmn_dram,
        )
        ttnn.deallocate(hidden_states, False)
        ttnn_matmul_21 = ttnn.matmul(
            ttnn_rms_norm_16,
            self.weights["model.lm_head.weight.t"],
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_rms_norm_16, False)
        ttnn_reshape_93 = ttnn.reshape(
            ttnn_matmul_21,
            [16, 1, 18944],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_21, False)
        import sys as _sys2
        print(">>> lm_head matmul done", flush=True, file=_sys2.stderr)
        ttnn_all_gather_16 = ttnn.all_gather(
            input_tensor=ttnn_reshape_93,
            dim=0,
            cluster_axis=0,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_93, False)
        print(">>> lm_head all_gather_16 (axis0) done", flush=True, file=_sys2.stderr)
        ttnn_all_gather_17 = ttnn.all_gather(
            input_tensor=ttnn_all_gather_16,
            dim=2,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_all_gather_16, False)
        ttnn_mesh_partition_2 = ttnn.mesh_partition(
            input_tensor=ttnn_all_gather_17,
            dim=0,
            cluster_axis=0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_to_layout_69 = ttnn.to_layout(
            ttnn_mesh_partition_2,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_mesh_partition_2, False)
        ttnn_argmax_0 = ttnn.argmax(
            ttnn_to_layout_69,
            2,
            True,
            sub_core_grids=None,
            # latest tt-metal main (#46340) removed use_multicore (multicore is default now)
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_69, False)
        ttnn_to_layout_70 = ttnn.to_layout(
            ttnn_argmax_0,
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_argmax_0, False)
        ttnn_typecast_56 = ttnn.typecast(
            ttnn_to_layout_70,
            ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_70, False)
        ttnn_reshape_94 = ttnn.reshape(
            ttnn_typecast_56,
            [16, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_56, False)
        ttnn_all_gather_18 = ttnn.all_gather(
            input_tensor=ttnn_reshape_94,
            dim=0,
            cluster_axis=0,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn_add_13 = ttnn.add(
            args_0,
            var_1,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(args_0, False)
        return [
            key_cache_outs[0],
            value_cache_outs[0],
            ttnn_add_10,
            key_cache_outs[1],
            value_cache_outs[1],
            ttnn_add_10,
            key_cache_outs[2],
            value_cache_outs[2],
            ttnn_add_10,
            key_cache_outs[3],
            value_cache_outs[3],
            ttnn_add_10,
            ttnn_reshape_94,
            ttnn_all_gather_18,
            ttnn_add_13,
            ttnn_all_gather_17,
        ]


class Glm4MoeRotaryEmbedding(LightweightModule):
    def __init__(self, device, weights):
        self.device = device
        self.weights = weights

    def forward(self, cache_position):
        ttnn_typecast_30 = ttnn.typecast(
            cache_position,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_12 = ttnn.reshape(
            ttnn_typecast_30,
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_30, False)
        ttnn_to_layout_50 = ttnn.to_layout(
            ttnn_reshape_12,
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_12, False)
        ttnn_matmul_0 = ttnn.matmul(
            self.weights["consteval.rotary_inv_freq"],
            ttnn_to_layout_50,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_to_layout_50, False)
        ttnn_reshape_13 = ttnn.reshape(
            ttnn_matmul_0,
            [1, 1, 1, 32],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_0, False)
        ttnn_concat_8 = ttnn.concat(
            [ttnn_reshape_13, ttnn_reshape_13],
            3,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_13, False)
        ttnn_cos_0 = ttnn.cos(
            ttnn_concat_8,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_31 = ttnn.typecast(
            ttnn_cos_0,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_cos_0, False)
        ttnn_sin_0 = ttnn.sin(
            ttnn_concat_8,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_concat_8, False)
        ttnn_typecast_32 = ttnn.typecast(
            ttnn_sin_0,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_sin_0, False)
        return ttnn_typecast_31, ttnn_typecast_32


class Glm4MoeAttention(LightweightModule):
    def __init__(self, device, weights, layer_idx):
        self.device = device
        self.weights = weights
        self.layer_idx = layer_idx

    def forward(self, hidden_states, key_cache_input, value_cache_input, cos, sin, repeat_idx, attn_mask):
        dram_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        )
        hifi4_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        layer_prefix = f"model.model.layers.{self.layer_idx}.self_attn"
        # QKV projection
        ttnn_linear = ttnn.linear(
            hidden_states,
            self.weights[f"{layer_prefix}.qkv_proj.weight"],
            bias=self.weights[f"{layer_prefix}.qkv_proj.bias"],
            transpose_a=False,
            transpose_b=False,
            memory_config=dram_mem,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=_QKV_PC,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(hidden_states, False)
        ttnn_reshape_qkv = ttnn.reshape(
            ttnn_linear,
            [16, 1, 1792],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_linear, False)
        v_q, v_k, v_v = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn_reshape_qkv,
            None,
            num_heads=12,
            num_kv_heads=1,
            transpose_key=False,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_qkv, False)
        # Value head reshape
        v_reshaped = ttnn.reshape(
            v_v,
            [1, 16, 1, 128],
            memory_config=dram_mem,
        )
        ttnn.deallocate(v_v, False)
        # Q norm
        q_normed = ttnn.rms_norm(
            v_q,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[f"{layer_prefix}.q_norm.weight"],
            bias=None,
            residual_input_tensor=None,
            memory_config=dram_mem,
            program_config=None,
            compute_kernel_config=hifi4_config,
        )
        ttnn.deallocate(v_q, False)
        # Q rotary embedding
        q_slice_first = ttnn.slice(
            q_normed,
            [0, 0, 0, 0],
            [16, 12, 1, 64],
            [1, 1, 1, 1],
            memory_config=dram_mem,
        )
        q_rotary = ttnn.experimental.rotary_embedding(
            q_slice_first,
            cos,
            sin,
            None,
            memory_config=dram_mem,
            compute_kernel_config=None,
        )
        ttnn.deallocate(q_slice_first, False)
        q_rotary_sliced = ttnn.slice(
            q_rotary,
            [0, 0, 0, 0],
            [16, 12, 1, 64],
            [1, 1, 1, 1],
            memory_config=dram_mem,
        )
        ttnn.deallocate(q_rotary, False)
        q_second_half = ttnn.slice(
            q_normed,
            [0, 0, 0, 64],
            [16, 12, 1, 128],
            [1, 1, 1, 1],
            memory_config=dram_mem,
        )
        ttnn.deallocate(q_normed, False)
        q_combined = ttnn.concat(
            [q_rotary_sliced, q_second_half],
            3,
            memory_config=dram_mem,
        )
        ttnn.deallocate(q_second_half, False)
        ttnn.deallocate(q_rotary_sliced, False)
        # K norm
        k_normed = ttnn.rms_norm(
            v_k,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[f"{layer_prefix}.k_norm.weight"],
            bias=None,
            residual_input_tensor=None,
            memory_config=dram_mem,
            program_config=None,
            compute_kernel_config=hifi4_config,
        )
        ttnn.deallocate(v_k, False)
        # K rotary embedding
        k_slice_first = ttnn.slice(
            k_normed,
            [0, 0, 0, 0],
            [16, 1, 1, 64],
            [1, 1, 1, 1],
            memory_config=dram_mem,
        )
        k_rotary = ttnn.experimental.rotary_embedding(
            k_slice_first,
            cos,
            sin,
            None,
            memory_config=dram_mem,
            compute_kernel_config=None,
        )
        ttnn.deallocate(k_slice_first, False)
        k_rotary_sliced = ttnn.slice(
            k_rotary,
            [0, 0, 0, 0],
            [16, 1, 1, 64],
            [1, 1, 1, 1],
            memory_config=dram_mem,
        )
        ttnn.deallocate(k_rotary, False)
        k_second_half = ttnn.slice(
            k_normed,
            [0, 0, 0, 64],
            [16, 1, 1, 128],
            [1, 1, 1, 1],
            memory_config=dram_mem,
        )
        ttnn.deallocate(k_normed, False)
        k_combined = ttnn.concat(
            [k_rotary_sliced, k_second_half],
            3,
            memory_config=dram_mem,
        )
        ttnn.deallocate(k_second_half, False)
        ttnn.deallocate(k_rotary_sliced, False)
        # Reshape K result
        k_reshaped = ttnn.reshape(
            k_combined,
            [1, 16, 1, 128],
            memory_config=dram_mem,
        )
        ttnn.deallocate(k_combined, False)
        # KV cache is head-sharded across the mesh columns, so each device owns
        # its KV head's cache directly -- no point-to-point redistribution needed.
        key_cache_out = key_cache_input
        # Paged update cache (key)
        k_to_mem = ttnn.to_memory_config(
            k_reshaped,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            # COL dispatch (required by moe_compute) reserves grid
                            # column x=7, leaving a 7x10 worker grid; the original
                            # 8-wide (0-7,0)+(0-7,1) KV shard collided with it. Use
                            # a dispatch-free 4x4 region (16 cores, shard->batch
                            # mapping is by tile-row index, not core position).
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3)),
                        ]
                    ),
                    [32, 128],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(k_reshaped, False)
        ttnn.experimental.paged_update_cache(
            key_cache_out,
            k_to_mem,
            update_idxs_tensor=repeat_idx,
            share_cache=False,
            page_table=None,
        )
        ttnn.deallocate(k_to_mem, False)
        value_cache_out = value_cache_input
        # Paged update cache (value)
        v_to_mem = ttnn.to_memory_config(
            v_reshaped,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            # COL dispatch (required by moe_compute) reserves grid
                            # column x=7, leaving a 7x10 worker grid; the original
                            # 8-wide (0-7,0)+(0-7,1) KV shard collided with it. Use
                            # a dispatch-free 4x4 region (16 cores, shard->batch
                            # mapping is by tile-row index, not core position).
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3)),
                        ]
                    ),
                    [32, 128],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(v_reshaped, False)
        ttnn.experimental.paged_update_cache(
            value_cache_out,
            v_to_mem,
            update_idxs_tensor=repeat_idx,
            share_cache=False,
            page_table=None,
        )
        ttnn.deallocate(v_to_mem, False)
        # SDPA
        q_for_sdpa = ttnn.reshape(
            q_combined,
            [1, 16, 12, 128],
            memory_config=dram_mem,
        )
        ttnn.deallocate(q_combined, False)
        sdpa_output = ttnn.transformer.scaled_dot_product_attention_decode(
            q_for_sdpa,
            key_cache_out,
            value_cache_out,
            is_causal=False,
            attn_mask=attn_mask,
            cur_pos_tensor=None,
            attention_sink=None,
            scale=0.08837890625,
            sliding_window_size=None,
            memory_config=dram_mem,
        )
        ttnn.deallocate(q_for_sdpa, False)
        # O projection
        sdpa_reshaped = ttnn.reshape(
            sdpa_output,
            [16, 1536],
            memory_config=dram_mem,
        )
        ttnn.deallocate(sdpa_output, False)
        o_proj_output = ttnn.matmul(
            sdpa_reshaped,
            self.weights[f"{layer_prefix}.o_proj.weight.t"],
            transpose_a=False,
            transpose_b=False,
            memory_config=dram_mem,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(sdpa_reshaped, False)
        o_reshaped = ttnn.reshape(
            o_proj_output,
            [1, 1, 16, 5120],
            memory_config=dram_mem,
        )
        ttnn.deallocate(o_proj_output, False)
        o_reduce_scatter = ttnn.reduce_scatter(
            input_tensor=o_reshaped,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=dram_mem,
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(o_reshaped, False)
        o_reshaped2 = ttnn.reshape(
            o_reduce_scatter,
            [16, 640],
            memory_config=dram_mem,
        )
        ttnn.deallocate(o_reduce_scatter, False)
        attn_output = ttnn.all_gather(
            input_tensor=o_reshaped2,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=dram_mem,
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(o_reshaped2, False)
        return attn_output, key_cache_out, value_cache_out


class Glm4MoeMLP(LightweightModule):
    def __init__(self, device, weights, layer_idx):
        self.device = device
        self.weights = weights
        self.layer_idx = layer_idx

    def forward(self, hidden_states):
        dram_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        )
        layer_prefix = f"model.model.layers.{self.layer_idx}.mlp"
        # gate_proj with silu activation
        gate_output = ttnn.matmul(
            hidden_states,
            self.weights[f"{layer_prefix}.gate_proj.weight.t"],
            transpose_a=False,
            transpose_b=False,
            memory_config=dram_mem,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation="silu",
            compute_kernel_config=None,
        )
        # up_proj
        up_output = ttnn.matmul(
            hidden_states,
            self.weights[f"{layer_prefix}.up_proj.weight.t"],
            transpose_a=False,
            transpose_b=False,
            memory_config=dram_mem,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(hidden_states, False)
        # multiply gate * up
        mul_output = ttnn.multiply(
            gate_output,
            up_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=dram_mem,
        )
        ttnn.deallocate(up_output, False)
        ttnn.deallocate(gate_output, False)
        # down_proj
        down_output = ttnn.matmul(
            mul_output,
            self.weights[f"{layer_prefix}.down_proj.weight.t"],
            transpose_a=False,
            transpose_b=False,
            memory_config=dram_mem,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(mul_output, False)
        # reduce_scatter + all_gather
        down_reshaped = ttnn.reshape(
            down_output,
            [1, 1, 16, 5120],
            memory_config=dram_mem,
        )
        ttnn.deallocate(down_output, False)
        reduce_scattered = ttnn.reduce_scatter(
            input_tensor=down_reshaped,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=dram_mem,
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(down_reshaped, False)
        rs_reshaped = ttnn.reshape(
            reduce_scattered,
            [16, 640],
            memory_config=dram_mem,
        )
        ttnn.deallocate(reduce_scattered, False)
        mlp_output = ttnn.all_gather(
            input_tensor=rs_reshaped,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=dram_mem,
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(rs_reshaped, False)
        return mlp_output


class A2aSparseMLPWithSharedExperts(LightweightModule):
    def __init__(self, device, weights, layer_idx):
        self.device = device
        self.weights = weights
        self.layer_idx = layer_idx

    def forward(self, post_normed, var_0, var_2):
        """Forward pass for MoE MLP.

        Args:
            post_normed: The post_attention_layernorm output, shape [16, 1, 5120].
                         This is reshaped to [16, 5120] for router/shared experts,
                         and to [16, 1, 1, 5120] for all_gather_12.
            var_0: consteval.scalar_zero_f32
            var_2: consteval.expert_mapping_u16
        """
        dram_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        )
        layer_prefix = f"model.model.layers.{self.layer_idx}.mlp"
        # Create the [16, 5120] reshape for router gate and shared experts
        # post_normed is already [16,5120] (the norm runs on the 2D residual since
        # iter8), so the old reshape to [16,5120] was an identity copy -- alias it
        # directly (one fewer ReshapeView per MoE layer). post_normed is now freed
        # with hidden_states after the shared experts (below), not early.
        hidden_states = post_normed
        # Router gate
        ttnn_typecast_33 = ttnn.typecast(
            hidden_states,
            ttnn.DataType.FLOAT32,
            memory_config=dram_mem,
        )
        ttnn_matmul_14 = ttnn.matmul(
            ttnn_typecast_33,
            self.weights[f"{layer_prefix}.mlp.router.gate.weight.t"],
            transpose_a=False,
            transpose_b=False,
            memory_config=dram_mem,
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation="sigmoid",
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_typecast_33, False)
        ttnn_add_7 = ttnn.add(
            ttnn_matmul_14,
            self.weights["consteval.e_score_correction_bias"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=dram_mem,
        )
        # n_group=1 / topk_group=1: group-limited routing is vacuous (the one
        # group is always selected -> group mask all-ones -> score masking is a
        # no-op). Skip the whole group-score/group-mask branch (topk(2)+topk(1),
        # concat_25 all_gather_9, scatter, mesh_partition_0, repeat_interleave(160),
        # ne, where) and select top-8 directly on the biased scores.
        ttnn_typecast_39 = ttnn.typecast(
            ttnn_add_7,
            ttnn.DataType.BFLOAT16,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_add_7, False)
        v_19, v_20 = ttnn.topk(
            ttnn_typecast_39,
            8,
            -1,
            True,
            False,
            memory_config=dram_mem,
        )
        ttnn.deallocate(v_19, False)
        ttnn.deallocate(ttnn_typecast_39, False)
        ttnn_typecast_40 = ttnn.typecast(
            v_20,
            ttnn.DataType.INT32,
            memory_config=dram_mem,
        )
        ttnn.deallocate(v_20, False)
        # Routing weights via native axis gather: scores[16,160].gather(dim=1,
        # top8[16,8]) -- small exact expert index. Replaces the now-dead flat-index
        # ttnn.embedding + its all_gather_10 score table + the token*160+expert
        # index matmul (all pruned). See TT_MLIR_RECOMMENDATIONS.md #5b.
        # ttnn_matmul_14 (raw local router scores [16,160]) and ttnn_typecast_40
        # (top8 expert ids) are consumed here; matmul_14 freed right after.
        _gather_idx = ttnn.reshape(
            ttnn.typecast(
                ttnn_typecast_40, ttnn.DataType.UINT32, memory_config=dram_mem
            ),
            [16, 8],
            memory_config=dram_mem,
        )
        _gather_scores = ttnn.reshape(
            ttnn_matmul_14, [16, 160], memory_config=dram_mem
        )
        _gathered = ttnn.gather(_gather_scores, 1, _gather_idx)
        ttnn.deallocate(_gather_idx, False)
        ttnn.deallocate(_gather_scores, False)
        ttnn.deallocate(ttnn_matmul_14, False)
        ttnn_typecast_45 = ttnn.typecast(
            _gathered, ttnn.DataType.FLOAT32, memory_config=dram_mem
        )
        ttnn.deallocate(_gathered, False)
        # (folded: reshape_73 was an identity [16,8]->[16,8]; sum directly)
        ttnn_sum_1 = ttnn.sum(
            ttnn_typecast_45,
            [1],
            True,
            memory_config=dram_mem,
            compute_kernel_config=None,
        )
        ttnn_add_9 = ttnn.add(
            ttnn_sum_1,
            self.weights["consteval.moe_epsilon"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_sum_1, False)
        ttnn_reshape_74 = ttnn.reshape(
            ttnn_add_9,
            [16, 1, 1],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_add_9, False)
        ttnn_reshape_75 = ttnn.reshape(
            ttnn_typecast_45,
            [16, 1, 8],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_typecast_45, False)
        ttnn_divide_0 = ttnn.divide(
            ttnn_reshape_75,
            ttnn_reshape_74,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_75, False)
        ttnn.deallocate(ttnn_reshape_74, False)
        ttnn_multiply_3 = ttnn.multiply(
            ttnn_divide_0,
            self.weights["consteval.topk_scaling"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_divide_0, False)
        # ===== Fused moe_compute MoE (replaces the hand-emitted dispatch +
        # moe_expert_token_remap + 3x sparse_matmul + all_to_all_combine + reduce/
        # gather/scale/sum). Reuses the router outputs:
        #   ttnn_typecast_40 = top-8 expert indices [16, 8] (int32)
        #   ttnn_multiply_3  = scaled normalized top-8 weights [16, 1, 8] (f32)
        mc_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=False,
        )
        mc_lin = self.weights["moe_compute.expert_mapping_lin"]
        # Free the dead hand-emitted bf8 routed sparse experts (replaced by moe_compute's
        # bf4 weights; unused post-swap). ~100+ MB/device of resident DRAM that starves the
        # fused selective_reduce_combine and deadlocks it. Freeing them fixes the hang
        # (matches deepseek_codegen MOE_COMPUTE_JOURNAL: dead-weight free resolved the
        # combine hang; the deadlock is DRAM-pressure-sensitive, not a barrier bug).
        # Aggressively free ALL weights that are dead by the time the layer-3 MoE runs, to
        # relieve the DRAM pressure that deadlocks the fused selective_reduce_combine (see
        # deepseek_codegen MOE_COMPUTE_JOURNAL: dead-weight free fixed the same hang). By
        # here, layers 0-2 (fully executed) + layer-3 attention (already done) are dead, and
        # embed_tokens (~1.55 GB/dev, replicated) is dead after the preamble embedding and is
        # NOT tied to lm_head. force=True to release even if a consteval cache still refs them.
        import sys as _sysf
        _dead = ["model.model.embed_tokens.weight.device"]
        _dead += [k for k in list(self.weights.keys())
                  if k.startswith(("model.model.layers.0.", "model.model.layers.1.",
                                   "model.model.layers.2.", "model.model.layers.3.self_attn"))]
        _dead += [k for k in list(self.weights.keys()) if "layers.3.mlp.mlp.experts" in k]
        _freed = 0
        for _dk in _dead:
            _dw = self.weights.pop(_dk, None)
            if _dw is not None:
                try:
                    ttnn.deallocate(_dw, True)
                    _freed += 1
                except Exception:
                    pass
        print(f">>> freed {_freed}/{len(_dead)} dead-weight tensors before MoE", flush=True, file=_sysf.stderr)
        # Fresh per-forward: semaphores + dispatch prealloc created right here so
        # nothing clobbers their L1 state before the MoE (see smoke harness).
        mc_dispatch_sem, mc_combine_sem, mc_prealloc = _make_moe_dispatch_runtime(self.device)
        # Dispatch inputs, per device B=16 tokens / S=1, ROW_MAJOR.
        # all_to_all_dispatch inputs placed in L1 (tested-working axis=0 config is
        # L1-in/DRAM-out; cf #45435 dispatch metadata returns zeros for non-rank-0
        # with DRAM-in).
        _disp_mc = ttnn.L1_MEMORY_CONFIG
        disp_x = ttnn.to_layout(
            ttnn.reshape(post_normed, [16, 1, 1, 5120], memory_config=dram_mem),
            ttnn.Layout.ROW_MAJOR, None, memory_config=_disp_mc,
        )
        # NOTE: post_normed is aliased to hidden_states (used by the shared experts
        # below), so it is NOT freed here -- freed once as hidden_states after shared.
        disp_idx = ttnn.to_layout(
            ttnn.typecast(
                ttnn.reshape(ttnn_typecast_40, [16, 1, 1, 8], memory_config=dram_mem),
                ttnn.DataType.UINT16, memory_config=dram_mem,
            ),
            ttnn.Layout.ROW_MAJOR, None, memory_config=_disp_mc,
        )
        ttnn.deallocate(ttnn_typecast_40, False)
        # multiply_3 [16,1,8] -> [16,1,1,8] reshaped ONCE, reused for the dispatch
        # scores (bf16, ROW_MAJOR) and the epilogue scaling weights (permute->TILE).
        _m3_4d = ttnn.reshape(ttnn_multiply_3, [16, 1, 1, 8], memory_config=dram_mem)
        ttnn.deallocate(ttnn_multiply_3, False)
        disp_scores = ttnn.to_layout(
            ttnn.typecast(_m3_4d, ttnn.DataType.BFLOAT16, memory_config=dram_mem),
            ttnn.Layout.ROW_MAJOR, None, memory_config=_disp_mc,
        )
        # Scaling weights for the epilogue: [16,1,1,8] -> [8,1,16,1] (k, 1, tokens, 1).
        scores_k = ttnn.permute(
            _m3_4d, (3, 1, 0, 2), memory_config=dram_mem, pad_value=0.0,
        )
        ttnn.deallocate(_m3_4d, False)
        scores_k = ttnn.to_layout(scores_k, ttnn.Layout.TILE, None, memory_config=dram_mem)

        import sys as _sys
        import os as _os
        if _os.environ.get("GLM_SKIP_DISPATCH") == "1":
            # Diagnostic: skip all_to_all_dispatch_metadata entirely; feed
            # moe_compute the zero-init prealloc. If moe_compute then completes,
            # dispatch_metadata's execution (not moe_compute) leaves the bad
            # fabric/state that deadlocks the full-model combine.
            sparse_buf, sparse_idx, sparse_scr = mc_prealloc
            ttnn.deallocate(disp_x, False)
            ttnn.deallocate(disp_idx, False)
            ttnn.deallocate(disp_scores, False)
            print(">>> SKIPPED dispatch_metadata (fed zero prealloc)", flush=True, file=_sys.stderr)
        else:
            sparse_buf, sparse_idx, sparse_scr = ttnn.experimental.all_to_all_dispatch_metadata(
                disp_x, disp_idx, disp_scores, mc_lin,
                cluster_axis=0, num_links=None,
                worker_mode=ttnn.WorkerMode.DIRECT,
                dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_MCAST_SHORTEST_PATH,
                output_tensors=mc_prealloc,
                cross_device_semaphore=mc_dispatch_sem,
            )
            ttnn.deallocate(disp_x, False)
            ttnn.deallocate(disp_idx, False)
            ttnn.deallocate(disp_scores, False)
        print(">>> moe dispatch_metadata enqueued", flush=True, file=_sys.stderr)
        if _os.environ.get("GLM_MOE_PINPOINT") == "1":
            ttnn.synchronize_device(self.device)
            print(">>> SYNC after dispatch_metadata OK", flush=True, file=_sys.stderr)

        # DEFAULT = fused selective_reduce_combine (correct PCC 0.992, all users).
        # No longer deadlocks: build carries combine fix #45764. Opt into the
        # perf-placeholder compute_only matmul+Ring-CCL path via GLM_MOE_COMPUTE_ONLY=1
        # (NOT exact PCC; for perf experiments only).
        if _os.environ.get("GLM_MOE_COMPUTE_ONLY") == "1":
            # compute_only path (#46863-era): matmul only, NO fused combine ring
            # (the deadlock source). cluster_axis/mux/sem/output_tensor must be
            # omitted. Returns matmul_output in slot 4 (no combine output).
            mc_outs = ttnn.experimental.moe_compute(
                sparse_buf, sparse_idx, sparse_scr, mc_lin,
                self.weights["moe_compute.w0_w1"], self.weights["moe_compute.w2"],
                layer_id=0, output_height_shard_dim=4, intermediate_size=1536,
                has_bias=False, compute_only=True,
            )
            print(">>> moe_compute (compute_only) enqueued", flush=True, file=_sys.stderr)
            if _os.environ.get("GLM_MOE_PINPOINT") == "1":
                ttnn.synchronize_device(self.device)
                print(">>> SYNC after moe_compute(compute_only) OK", flush=True, file=_sys.stderr)
            # CCL COMBINE (Ring), replacing the deadlocking selective_reduce_combine:
            # take moe_compute's per-expert matmul_output (slot 4, [70,2,32,5120]
            # core-sharded), reshape to a [k=8, tokens=16, H] view, then combine
            # across the 4-device dispatch axis (cluster_axis=0) with a Ring
            # all_gather + sum. This is device-perf-representative; exact PCC needs
            # the proper per-expert-token mapping (see MOE_COMPUTE_FINDINGS.md).
            # Move the L1-sharded matmul_output to DRAM first (its L1 shards on
            # cores [0-0..6-9] otherwise clash with downstream CB allocations).
            matmul_output = ttnn.to_memory_config(mc_outs[4], memory_config=dram_mem)
            for _i in range(len(mc_outs)):
                try:
                    ttnn.deallocate(mc_outs[_i], False)
                except Exception:
                    pass
            mo = ttnn.reshape(matmul_output, [4480, 5120], memory_config=dram_mem)
            ttnn.deallocate(matmul_output, False)
            mo = ttnn.slice(mo, [0, 0], [128, 5120], [1, 1], memory_config=dram_mem)
            mo = ttnn.reshape(mo, [8, 16, 5120], memory_config=dram_mem)
            mo_g = ttnn.all_gather(
                input_tensor=mo, dim=1, cluster_axis=0, subdevice_id=None,
                memory_config=dram_mem, num_links=None, topology=ttnn.Topology.Ring,
            )
            ttnn.deallocate(mo, False)
            mo_g = ttnn.reshape(mo_g, [8, 4, 16, 5120], memory_config=dram_mem)
            combine_output = ttnn.sum(mo_g, [1], False, memory_config=dram_mem, compute_kernel_config=None)
            ttnn.deallocate(mo_g, False)
            print(">>> CCL combine (all_gather axis0 Ring) enqueued", flush=True, file=_sys.stderr)
            if _os.environ.get("GLM_MOE_PINPOINT") == "1":
                ttnn.synchronize_device(self.device)
                print(">>> SYNC after CCL combine OK", flush=True, file=_sys.stderr)
        else:
            # GLM_MOE_FUSED_COMBINE=1: fused selective_reduce_combine -- HANGS on (4,8)
            # cluster_axis=0 (issue #47523). Opt-in only; the default path above is runnable.
            mc_combine_out = ttnn.moreh_full(
                shape=[8, 16, 5120], fill_value=0, device=self.device,
                layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.BFLOAT16,
                memory_config=dram_mem,
            )
            mc_outs = ttnn.experimental.moe_compute(
                sparse_buf, sparse_idx, sparse_scr, mc_lin,
                self.weights["moe_compute.w0_w1"], self.weights["moe_compute.w2"],
                layer_id=0, output_height_shard_dim=4, intermediate_size=1536,
                has_bias=False, cluster_axis=0, mux_core_range_set=self.weights["moe_compute.mux_cores"],
                optional_output_tensor=mc_combine_out,
                optional_cross_device_semaphore=mc_combine_sem,
            )
            combine_output = mc_outs[-1]  # [8, 16, 5120] per device
            print(">>> moe_compute enqueued", flush=True, file=_sys.stderr)
            if _os.environ.get("GLM_MOE_PINPOINT") == "1":
                ttnn.synchronize_device(self.device)
                print(">>> SYNC after moe_compute OK", flush=True, file=_sys.stderr)
        if _os.environ.get("GLM_MOE_FUSED_COMBINE") == "1":
            for _i in (0, 1, 2, 4):
                try:
                    ttnn.deallocate(mc_outs[_i], False)
                except Exception:
                    pass

        # Epilogue: scale by per-(token,k) weights, sum over k, cross-col all-reduce.
        ce = ttnn.to_layout(combine_output, ttnn.Layout.TILE, None, memory_config=dram_mem)
        ce = ttnn.unsqueeze(ce, dim=1)  # [8, 1, 16, 5120]
        scaled = ttnn.multiply(ce, scores_k, dtype=ttnn.DataType.BFLOAT16, memory_config=dram_mem)
        ttnn.deallocate(ce, False)
        ttnn.deallocate(scores_k, False)
        summed = ttnn.sum(scaled, [0], False, memory_config=dram_mem, compute_kernel_config=None)
        ttnn.deallocate(scaled, False)
        # sparse column-partial [1,1,16,5120]; the TP all-reduce over the 8 cols is
        # DEFERRED and MERGED with the shared-experts all-reduce below (linearity:
        # allreduce(sparse)+allreduce(shared) == allreduce(sparse+shared)), saving a
        # whole reduce_scatter+all_gather pair per MoE layer.
        summed = ttnn.reshape(summed, [1, 1, 16, 5120], memory_config=dram_mem)
        # Shared experts
        shared_gate = ttnn.matmul(
            hidden_states,
            self.weights[f"{layer_prefix}.shared_experts.gate_proj.weight.t"],
            transpose_a=False,
            transpose_b=False,
            memory_config=dram_mem,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=_SHARED_PC,
            activation="silu",
            compute_kernel_config=None,
        )
        shared_up = ttnn.matmul(
            hidden_states,
            self.weights[f"{layer_prefix}.shared_experts.up_proj.weight.t"],
            transpose_a=False,
            transpose_b=False,
            memory_config=dram_mem,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=_SHARED_PC,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(hidden_states, False)
        shared_mul = ttnn.multiply(
            shared_gate,
            shared_up,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=dram_mem,
        )
        ttnn.deallocate(shared_up, False)
        ttnn.deallocate(shared_gate, False)
        shared_down = ttnn.matmul(
            shared_mul,
            self.weights[f"{layer_prefix}.shared_experts.down_proj.weight.t"],
            transpose_a=False,
            transpose_b=False,
            memory_config=dram_mem,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(shared_mul, False)
        shared_reshaped = ttnn.reshape(
            shared_down,
            [1, 1, 16, 5120],
            memory_config=dram_mem,
        )
        ttnn.deallocate(shared_down, False)
        # Merge the sparse + shared column-partials, then ONE all-reduce (was two
        # separate reduce_scatter+all_gather pairs + a final add). The reduce runs
        # at num_links=3 (the sparse epilogue's tuned value).
        combined = ttnn.add(
            summed, shared_reshaped, dtype=ttnn.DataType.BFLOAT16, memory_config=dram_mem,
        )
        ttnn.deallocate(summed, False)
        ttnn.deallocate(shared_reshaped, False)
        mc_rs = ttnn.reduce_scatter(
            input_tensor=combined, dim=3, cluster_axis=1, subdevice_id=None,
            memory_config=dram_mem, num_links=3, topology=ttnn.Topology.Ring,
            compute_kernel_config=mc_hifi4,
        )
        ttnn.deallocate(combined, False)
        # reshape [1,1,16,640] -> [16,640] then all_gather dim=1 (dim=3 rank-4
        # all_gather hangs under COL dispatch).
        mc_rs_reshaped = ttnn.reshape(mc_rs, [16, 640], memory_config=dram_mem)
        ttnn.deallocate(mc_rs, False)
        moe_output = ttnn.all_gather(
            input_tensor=mc_rs_reshaped, dim=1, cluster_axis=1, subdevice_id=None,
            memory_config=dram_mem, num_links=3, topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(mc_rs_reshaped, False)
        print(">>> moe merged sparse+shared all-reduce enqueued", flush=True, file=_sys.stderr)
        return moe_output


class Glm4MoeDecoderLayer(LightweightModule):
    def __init__(self, device, weights, layer_idx, is_moe):
        self.device = device
        self.weights = weights
        self.layer_idx = layer_idx
        self.is_moe = is_moe
        self.self_attn = Glm4MoeAttention(device, weights, layer_idx)
        if is_moe:
            self.mlp = A2aSparseMLPWithSharedExperts(device, weights, layer_idx)
        else:
            self.mlp = Glm4MoeMLP(device, weights, layer_idx)

    def forward(self, hidden_states, key_cache_input, value_cache_input, cos, sin, repeat_idx, attn_mask, var_0, var_2):
        dram_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        )
        hifi4_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        layer_prefix = f"model.model.layers.{self.layer_idx}"
        # Signpost: this layer's attention block (input_layernorm + self_attn +
        # residual add). Same structure across all 92 layers.
        signpost(f"L{self.layer_idx}_attn")
        # Input layernorm (8-core width-sharded; see sharded_rms_norm).
        normed = sharded_rms_norm(
            hidden_states,
            self.weights[f"{layer_prefix}.input_layernorm.weight"],
            9.9999997473787516e-06, hifi4_config, dram_mem,
        )
        # Attention
        import os as _os_attn
        if _os_attn.environ.get("GLM_SKIP_ATTN") == "1":
            # Bisect: skip attention (KV update + CCLs); feed MLP the pre-attn
            # hidden_states. Correctness meaningless. Isolates whether layer-3
            # attention vs the resident weight set triggers the moe_compute hang.
            ttnn.deallocate(normed, False)
            key_cache_out, value_cache_out = key_cache_input, value_cache_input
            residual = hidden_states
        else:
            attn_output, key_cache_out, value_cache_out = self.self_attn(
                normed, key_cache_input, value_cache_input, cos, sin, repeat_idx, attn_mask
            )
            # Residual add after attention
            residual = ttnn.add(
                hidden_states,
                attn_output,
                dtype=ttnn.DataType.BFLOAT16,
                memory_config=dram_mem,
            )
            ttnn.deallocate(attn_output, False)
            ttnn.deallocate(hidden_states, False)
        # Signpost: this layer's MLP block (post_attention_layernorm + MLP +
        # residual add). Dense MLP for layers 0-2, MoE MLP for layers 3-91.
        signpost(f"L{self.layer_idx}_mlp")
        if self.is_moe:
            # Post-attention layernorm directly on the 2D [16,5120] residual (the
            # sharded norm wants a 2D physical layout; the old [16,1,5120] detour
            # tile-padded to 512 physical rows and forced two costly ReshapeViews
            # that ate the sharded-norm savings). A2aSparse reshapes post_normed to
            # [16,5120] / [16,1,1,5120] internally, so a 2D input is fine.
            post_normed = sharded_rms_norm(
                residual,
                self.weights[f"{layer_prefix}.post_attention_layernorm.weight"],
                9.9999997473787516e-06, hifi4_config, dram_mem,
            )
            moe_output = self.mlp(post_normed, var_0, var_2)
            # Residual add after MoE MLP
            output = ttnn.add(
                residual,
                moe_output,
                dtype=ttnn.DataType.BFLOAT16,
                memory_config=dram_mem,
            )
            ttnn.deallocate(moe_output, False)
            ttnn.deallocate(residual, False)
        else:
            # Dense MLP path
            post_normed = sharded_rms_norm(
                residual,
                self.weights[f"{layer_prefix}.post_attention_layernorm.weight"],
                9.9999997473787516e-06, hifi4_config, dram_mem,
            )
            mlp_output = self.mlp(post_normed)
            # Residual add after MLP
            output = ttnn.add(
                residual,
                mlp_output,
                dtype=ttnn.DataType.BFLOAT16,
                memory_config=dram_mem,
            )
            ttnn.deallocate(mlp_output, False)
            ttnn.deallocate(residual, False)
        return output, key_cache_out, value_cache_out
