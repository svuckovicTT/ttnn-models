import ttnn
import params


class LightweightModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ModelTTNN(LightweightModule):
    def __init__(self, device):
        self.device = device
        self.weights = params.load_weights_for__main_from_state_dict(device)
        import consteval
        self.weights = consteval.run_consteval(self.weights, device)
        self.rotary_embedding = Glm4MoeRotaryEmbedding(device, self.weights)
        self.layers = []
        for i in range(3):
            self.layers.append(
                Glm4MoeDecoderLayer(device, self.weights, layer_idx=i, is_moe=False)
            )
        self.layers.append(
            Glm4MoeDecoderLayer(device, self.weights, layer_idx=3, is_moe=True)
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
        for layer_idx in range(4):
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
        ttnn.deallocate(ttnn_repeat_1, False)
        ttnn.deallocate(ttnn_repeat_2, False)
        ttnn.deallocate(sin, False)
        ttnn.deallocate(cos, False)
        # Final norm and lm_head
        ttnn_rms_norm_16 = ttnn.rms_norm(
            hidden_states,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["model.model.norm.weight"],
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
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
        ttnn_all_gather_16 = ttnn.all_gather(
            input_tensor=ttnn_reshape_93,
            dim=0,
            cluster_axis=0,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Linear,
        )
        ttnn.deallocate(ttnn_reshape_93, False)
        ttnn_all_gather_17 = ttnn.all_gather(
            input_tensor=ttnn_all_gather_16,
            dim=2,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Linear,
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
            use_multicore=True,
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
            topology=ttnn.Topology.Linear,
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
            program_config=None,
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
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(7, 1)),
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
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(7, 1)),
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
        # DRAM-sharded o_proj matmul: width-shard the [16,1536] activation into L1
        # across the compute grid, matmul against the DRAM-width-sharded weight, then
        # bring the [16,5120] output back to DRAM-interleaved for the CCL path below.
        import dram_matmul
        o_rows, o_cols, o_cores = dram_matmul.compute_grid(1536, 5120)
        sdpa_sharded = ttnn.to_memory_config(
            sdpa_reshaped,
            dram_matmul.in0_l1_width_sharded_config(16, 1536, o_rows, o_cols),
        )
        ttnn.deallocate(sdpa_reshaped, False)
        o_proj_sharded = ttnn.matmul(
            sdpa_sharded,
            self.weights[f"{layer_prefix}.o_proj.weight.t"],
            transpose_a=False,
            transpose_b=False,
            memory_config=dram_matmul.out_l1_width_sharded_config(16, 5120, o_rows, o_cols),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=dram_matmul.program_config(16, 1536, 5120, o_cores),
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(sdpa_sharded, False)
        o_proj_output = ttnn.to_memory_config(o_proj_sharded, dram_mem)
        ttnn.deallocate(o_proj_sharded, False)
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
            topology=ttnn.Topology.Linear,
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
            topology=ttnn.Topology.Linear,
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
            topology=ttnn.Topology.Linear,
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
            topology=ttnn.Topology.Linear,
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
        hidden_states = ttnn.reshape(
            post_normed,
            [16, 5120],
            memory_config=dram_mem,
        )
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
        ttnn_reshape_65 = ttnn.reshape(
            ttnn_add_7,
            [16, 1, 160],
            memory_config=dram_mem,
        )
        ttnn_typecast_34 = ttnn.typecast(
            ttnn_reshape_65,
            ttnn.DataType.BFLOAT16,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_65, False)
        v_15, v_16 = ttnn.topk(
            ttnn_typecast_34,
            2,
            -1,
            True,
            True,
            memory_config=dram_mem,
        )
        ttnn.deallocate(v_16, False)
        ttnn.deallocate(ttnn_typecast_34, False)
        ttnn_typecast_35 = ttnn.typecast(
            v_15,
            ttnn.DataType.FLOAT32,
            memory_config=dram_mem,
        )
        ttnn.deallocate(v_15, False)
        ttnn_sum_0 = ttnn.sum(
            ttnn_typecast_35,
            [2],
            False,
            memory_config=dram_mem,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_typecast_35, False)
        ttnn_typecast_36 = ttnn.typecast(
            ttnn_sum_0,
            ttnn.DataType.BFLOAT16,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_sum_0, False)
        v_17, v_18 = ttnn.topk(
            ttnn_typecast_36,
            1,
            -1,
            True,
            False,
            memory_config=dram_mem,
        )
        ttnn.deallocate(v_17, False)
        ttnn.deallocate(ttnn_typecast_36, False)
        ttnn_typecast_37 = ttnn.typecast(
            v_18,
            ttnn.DataType.INT32,
            memory_config=dram_mem,
        )
        ttnn.deallocate(v_18, False)
        ttnn_reshape_66 = ttnn.reshape(
            ttnn_typecast_37,
            [16, 1, 1],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_typecast_37, False)
        ttnn_concat_25 = ttnn.concat(
            [self.weights["consteval.batch_indices_i32"], ttnn_reshape_66],
            2,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_66, False)
        ttnn_all_gather_9 = ttnn.all_gather(
            input_tensor=ttnn_concat_25,
            dim=0,
            cluster_axis=0,
            subdevice_id=None,
            memory_config=dram_mem,
            num_links=None,
            topology=ttnn.Topology.Linear,
        )
        ttnn.deallocate(ttnn_concat_25, False)
        ttnn_reshape_67 = ttnn.reshape(
            ttnn_all_gather_9,
            [64, 2],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_all_gather_9, False)
        ttnn_slice_96 = ttnn.slice(
            ttnn_reshape_67,
            [0, 0],
            [64, 1],
            [1, 1],
            memory_config=dram_mem,
        )
        ttnn_slice_97 = ttnn.slice(
            ttnn_reshape_67,
            [0, 1],
            [64, 2],
            [1, 1],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_67, False)
        ttnn_add_8 = ttnn.add(
            ttnn_slice_96,
            ttnn_slice_97,
            dtype=ttnn.DataType.INT32,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_slice_97, False)
        ttnn.deallocate(ttnn_slice_96, False)
        ttnn_reshape_68 = ttnn.reshape(
            ttnn_add_8,
            [64],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_add_8, False)
        ttnn_to_layout_51 = ttnn.to_layout(
            ttnn_reshape_68,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_68, False)
        ttnn_scatter_0 = ttnn.scatter(
            input=self.weights["consteval.mesh_zeros"],
            dim=0,
            index=ttnn_to_layout_51,
            src=self.weights["consteval.mesh_ones"],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_to_layout_51, False)
        ttnn_to_layout_52 = ttnn.to_layout(
            ttnn_scatter_0,
            ttnn.Layout.TILE,
            None,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_scatter_0, False)
        ttnn_reshape_69 = ttnn.reshape(
            ttnn_to_layout_52,
            [64, 1],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_to_layout_52, False)
        ttnn_to_layout_53 = ttnn.to_layout(
            ttnn_reshape_69,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_69, False)
        ttnn_mesh_partition_0 = ttnn.mesh_partition(
            input_tensor=ttnn_to_layout_53,
            dim=0,
            cluster_axis=0,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_to_layout_53, False)
        ttnn_to_layout_54 = ttnn.to_layout(
            ttnn_mesh_partition_0,
            ttnn.Layout.TILE,
            None,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_mesh_partition_0, False)
        ttnn_repeat_interleave_0 = ttnn.repeat_interleave(
            ttnn_to_layout_54,
            160,
            1,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_to_layout_54, False)
        ttnn_ne_0 = ttnn.ne(
            ttnn_repeat_interleave_0,
            var_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_repeat_interleave_0, False)
        ttnn_typecast_38 = ttnn.typecast(
            ttnn_ne_0,
            ttnn.DataType.FLOAT32,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_ne_0, False)
        ttnn_where_1 = ttnn.where(
            ttnn_typecast_38,
            ttnn_add_7,
            var_0,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_typecast_38, False)
        ttnn.deallocate(ttnn_add_7, False)
        ttnn_typecast_39 = ttnn.typecast(
            ttnn_where_1,
            ttnn.DataType.BFLOAT16,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_where_1, False)
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
        ttnn_typecast_41 = ttnn.typecast(
            ttnn_typecast_40,
            ttnn.DataType.UINT32,
            memory_config=dram_mem,
        )
        ttnn_reshape_70 = ttnn.reshape(
            ttnn_typecast_41,
            [16, 8, 1],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_typecast_41, False)
        ttnn_concat_26 = ttnn.concat(
            [self.weights["consteval.moe_batch_indices"], ttnn_reshape_70],
            2,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_70, False)
        ttnn_all_gather_10 = ttnn.all_gather(
            input_tensor=ttnn_matmul_14,
            dim=0,
            cluster_axis=0,
            subdevice_id=None,
            memory_config=dram_mem,
            num_links=None,
            topology=ttnn.Topology.Linear,
        )
        ttnn.deallocate(ttnn_matmul_14, False)
        ttnn_reshape_71 = ttnn.reshape(
            ttnn_all_gather_10,
            [10240, 1],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_all_gather_10, False)
        ttnn_typecast_42 = ttnn.typecast(
            ttnn_concat_26,
            ttnn.DataType.FLOAT32,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_concat_26, False)
        ttnn_matmul_15 = ttnn.matmul(
            ttnn_typecast_42,
            self.weights["consteval.moe_constants"],
            transpose_a=False,
            transpose_b=False,
            memory_config=dram_mem,
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_typecast_42, False)
        ttnn_reshape_72 = ttnn.reshape(
            ttnn_matmul_15,
            [128],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_matmul_15, False)
        ttnn_typecast_43 = ttnn.typecast(
            ttnn_reshape_72,
            ttnn.DataType.UINT32,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_72, False)
        ttnn_to_layout_55 = ttnn.to_layout(
            ttnn_typecast_43,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_typecast_43, False)
        ttnn_typecast_44 = ttnn.typecast(
            ttnn_reshape_71,
            ttnn.DataType.BFLOAT16,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_71, False)
        ttnn_to_layout_56 = ttnn.to_layout(
            ttnn_typecast_44,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_typecast_44, False)
        ttnn_embedding_1 = ttnn.embedding(
            ttnn_to_layout_55,
            ttnn_to_layout_56,
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_to_layout_56, False)
        ttnn.deallocate(ttnn_to_layout_55, False)
        ttnn_typecast_45 = ttnn.typecast(
            ttnn_embedding_1,
            ttnn.DataType.FLOAT32,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_embedding_1, False)
        ttnn_reshape_73 = ttnn.reshape(
            ttnn_typecast_45,
            [16, 8],
            memory_config=dram_mem,
        )
        ttnn_sum_1 = ttnn.sum(
            ttnn_reshape_73,
            [1],
            True,
            memory_config=dram_mem,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_reshape_73, False)
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
        ttnn_reshape_76 = ttnn.reshape(
            ttnn_typecast_40,
            [16, 8, 1],
            memory_config=dram_mem,
        )
        ttnn_eq_0 = ttnn.eq(
            ttnn_reshape_76,
            self.weights["consteval.expert_indices"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_76, False)
        ttnn_typecast_46 = ttnn.typecast(
            ttnn_eq_0,
            ttnn.DataType.FLOAT32,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_eq_0, False)
        ttnn_matmul_16 = ttnn.matmul(
            ttnn_multiply_3,
            ttnn_typecast_46,
            transpose_a=False,
            transpose_b=False,
            memory_config=dram_mem,
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_3, False)
        ttnn_reshape_77 = ttnn.reshape(
            ttnn_matmul_16,
            [1, 16, 160],
            memory_config=dram_mem,
        )
        ttnn_concat_27 = ttnn.concat(
            [ttnn_reshape_77, ttnn_reshape_77, ttnn_reshape_77, ttnn_reshape_77],
            1,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_77, False)
        ttnn_all_gather_11 = ttnn.all_gather(
            input_tensor=ttnn_concat_27,
            dim=1,
            cluster_axis=0,
            subdevice_id=None,
            memory_config=dram_mem,
            num_links=None,
            topology=ttnn.Topology.Linear,
        )
        ttnn.deallocate(ttnn_concat_27, False)
        ttnn_reshape_78 = ttnn.reshape(
            ttnn_all_gather_11,
            [1, 1, 256, 160],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_all_gather_11, False)
        ttnn_reshape_79 = ttnn.reshape(
            post_normed,
            [16, 1, 1, 5120],
            memory_config=dram_mem,
        )
        ttnn.deallocate(post_normed, False)
        ttnn_reshape_80 = ttnn.reshape(
            ttnn_typecast_40,
            [16, 1, 1, 8],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_typecast_40, False)
        ttnn_all_gather_12 = ttnn.all_gather(
            input_tensor=ttnn_reshape_79,
            dim=0,
            cluster_axis=0,
            subdevice_id=None,
            memory_config=dram_mem,
            num_links=None,
            topology=ttnn.Topology.Linear,
        )
        ttnn.deallocate(ttnn_reshape_79, False)
        ttnn_all_gather_13 = ttnn.all_gather(
            input_tensor=ttnn_reshape_80,
            dim=0,
            cluster_axis=0,
            subdevice_id=None,
            memory_config=dram_mem,
            num_links=None,
            topology=ttnn.Topology.Linear,
        )
        ttnn.deallocate(ttnn_reshape_80, False)
        ttnn_to_layout_57 = ttnn.to_layout(
            ttnn_all_gather_12,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_all_gather_12, False)
        ttnn_typecast_47 = ttnn.typecast(
            ttnn_all_gather_13,
            ttnn.DataType.UINT16,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_all_gather_13, False)
        ttnn_from_device_25 = ttnn.from_device(ttnn_typecast_47)
        ttnn.deallocate(ttnn_typecast_47, False)
        ttnn_to_layout_58 = ttnn.to_layout(
            ttnn_from_device_25, ttnn.Layout.ROW_MAJOR, None, memory_config=None
        )
        ttnn.deallocate(ttnn_from_device_25, False)
        ttnn_to_device_74 = ttnn.to_device(
            ttnn_to_layout_58,
            device=self.device,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_to_layout_58, False)
        v_21, v_22 = ttnn.all_to_all_dispatch(
            input_tensor=ttnn_to_layout_57,
            expert_indices_tensor=ttnn_to_device_74,
            expert_mapping_tensor=var_2,
            cluster_axis=0,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_to_device_74, False)
        ttnn.deallocate(ttnn_to_layout_57, False)
        ttnn_to_layout_59 = ttnn.to_layout(
            v_22,
            ttnn.Layout.TILE,
            None,
            memory_config=dram_mem,
        )
        ttnn.deallocate(v_22, False)
        ttnn_typecast_48 = ttnn.typecast(
            ttnn_to_layout_59,
            ttnn.DataType.INT32,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_to_layout_59, False)
        ttnn_to_layout_60 = ttnn.to_layout(
            v_21,
            ttnn.Layout.TILE,
            None,
            memory_config=dram_mem,
        )
        ttnn.deallocate(v_21, False)
        ttnn_reshape_81 = ttnn.reshape(
            ttnn_typecast_48,
            [1, 1, 256, 8],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_typecast_48, False)
        ttnn_typecast_49 = ttnn.typecast(
            ttnn_reshape_78,
            ttnn.DataType.BFLOAT16,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_78, False)
        ttnn_to_layout_61 = ttnn.to_layout(
            ttnn_typecast_49,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_typecast_49, False)
        ttnn_typecast_50 = ttnn.typecast(
            ttnn_reshape_81,
            ttnn.DataType.UINT16,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_81, False)
        ttnn_from_device_26 = ttnn.from_device(ttnn_typecast_50)
        ttnn.deallocate(ttnn_typecast_50, False)
        ttnn_to_layout_62 = ttnn.to_layout(
            ttnn_from_device_26, ttnn.Layout.ROW_MAJOR, None, memory_config=None
        )
        ttnn.deallocate(ttnn_from_device_26, False)
        ttnn_to_device_75 = ttnn.to_device(
            ttnn_to_layout_62,
            device=self.device,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_to_layout_62, False)
        v_23, v_24 = ttnn.moe_expert_token_remap(
            topk_tensor=ttnn_to_layout_61,
            expert_mapping_tensor=var_2,
            expert_metadata_tensor=ttnn_to_device_75,
            reduction_size=32,
            memory_config=dram_mem,
        )
        ttnn.deallocate(v_23, False)
        ttnn.deallocate(ttnn_to_layout_61, False)
        ttnn_to_layout_63 = ttnn.to_layout(
            v_24,
            ttnn.Layout.TILE,
            None,
            memory_config=dram_mem,
        )
        ttnn_typecast_51 = ttnn.typecast(
            ttnn_to_layout_63,
            ttnn.DataType.FLOAT32,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_to_layout_63, False)
        ttnn_reshape_82 = ttnn.reshape(
            ttnn_to_layout_60,
            [8, 1, 32, 5120],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_to_layout_60, False)
        ttnn_reshape_83 = ttnn.reshape(
            ttnn_typecast_51,
            [8, 1, 1, 5],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_typecast_51, False)
        ttnn_typecast_52 = ttnn.typecast(
            ttnn_reshape_83,
            ttnn.DataType.BFLOAT16,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_83, False)
        ttnn_to_layout_64 = ttnn.to_layout(
            ttnn_typecast_52,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_typecast_52, False)
        sparse_matmul_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 9),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=6,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
            gather_in0=False,
            hop_cores=ttnn.CoreRangeSet([]),
            num_global_cb_receivers=0,
            untilize_out=False,
        )
        ttnn_sparse_matmul_0 = ttnn.sparse_matmul(
            input_tensor_a=ttnn_reshape_82,
            input_tensor_b=self.weights[f"{layer_prefix}.mlp.experts.gate_proj.reshaped"],
            sparsity=ttnn_to_layout_64,
            program_config=sparse_matmul_config,
            nnz=None,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=dram_mem,
            dtype=None,
        )
        ttnn_reshape_84 = ttnn.reshape(
            ttnn_sparse_matmul_0,
            [8, 5, 32, 1536],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_sparse_matmul_0, False)
        ttnn_silu_0 = ttnn.silu(
            ttnn_reshape_84,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_84, False)
        ttnn_sparse_matmul_1 = ttnn.sparse_matmul(
            input_tensor_a=ttnn_reshape_82,
            input_tensor_b=self.weights[f"{layer_prefix}.mlp.experts.up_proj.reshaped"],
            sparsity=ttnn_to_layout_64,
            program_config=sparse_matmul_config,
            nnz=None,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=dram_mem,
            dtype=None,
        )
        ttnn.deallocate(ttnn_to_layout_64, False)
        ttnn.deallocate(ttnn_reshape_82, False)
        ttnn_reshape_85 = ttnn.reshape(
            ttnn_sparse_matmul_1,
            [8, 5, 32, 1536],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_sparse_matmul_1, False)
        ttnn_multiply_4 = ttnn.multiply(
            ttnn_silu_0,
            ttnn_reshape_85,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_85, False)
        ttnn.deallocate(ttnn_silu_0, False)
        ttnn_from_device_27 = ttnn.from_device(v_24)
        ttnn.deallocate(v_24, False)
        ttnn_typecast_53 = ttnn.typecast(
            ttnn_from_device_27, ttnn.DataType.BFLOAT16, memory_config=None
        )
        ttnn.deallocate(ttnn_from_device_27, False)
        ttnn_to_device_76 = ttnn.to_device(
            ttnn_typecast_53,
            device=self.device,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_typecast_53, False)
        ttnn_sparse_matmul_2 = ttnn.sparse_matmul(
            input_tensor_a=ttnn_multiply_4,
            input_tensor_b=self.weights[f"{layer_prefix}.mlp.experts.down_proj.reshaped"],
            sparsity=ttnn_to_device_76,
            program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 9),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=1,
                per_core_M=1,
                per_core_N=20,
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
                gather_in0=False,
                hop_cores=ttnn.CoreRangeSet([]),
                num_global_cb_receivers=0,
                untilize_out=False,
            ),
            nnz=None,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            memory_config=dram_mem,
            dtype=None,
        )
        ttnn.deallocate(ttnn_to_device_76, False)
        ttnn.deallocate(ttnn_multiply_4, False)
        ttnn_permute_30 = ttnn.permute(
            ttnn_sparse_matmul_2,
            [1, 0, 2, 3],
            memory_config=dram_mem,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_sparse_matmul_2, False)
        ttnn_reshape_86 = ttnn.reshape(
            ttnn_permute_30,
            [5, 1, 256, 5120],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_permute_30, False)
        ttnn_to_layout_65 = ttnn.to_layout(
            ttnn_reshape_86,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_86, False)
        ttnn_all_to_all_combine_0 = ttnn.all_to_all_combine(
            input_tensor=ttnn_to_layout_65,
            expert_metadata_tensor=ttnn_to_device_75,
            expert_mapping_tensor=var_2,
            cluster_axis=0,
            output_shard_dim=2,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_to_layout_65, False)
        ttnn.deallocate(ttnn_to_device_75, False)
        ttnn_to_layout_66 = ttnn.to_layout(
            ttnn_all_to_all_combine_0,
            ttnn.Layout.TILE,
            None,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_all_to_all_combine_0, False)
        ttnn_reduce_scatter_7 = ttnn.reduce_scatter(
            input_tensor=ttnn_to_layout_66,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=dram_mem,
            num_links=None,
            topology=ttnn.Topology.Linear,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_to_layout_66, False)
        ttnn_all_gather_14 = ttnn.all_gather(
            input_tensor=ttnn_reduce_scatter_7,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=dram_mem,
            num_links=None,
            topology=ttnn.Topology.Linear,
        )
        ttnn.deallocate(ttnn_reduce_scatter_7, False)
        ttnn_to_layout_67 = ttnn.to_layout(
            ttnn_all_gather_14,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_all_gather_14, False)
        ttnn_mesh_partition_1 = ttnn.mesh_partition(
            input_tensor=ttnn_to_layout_67,
            dim=2,
            cluster_axis=0,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_to_layout_67, False)
        ttnn_to_layout_68 = ttnn.to_layout(
            ttnn_mesh_partition_1,
            ttnn.Layout.TILE,
            None,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_mesh_partition_1, False)
        ttnn_typecast_54 = ttnn.typecast(
            ttnn_to_layout_68,
            ttnn.DataType.FLOAT32,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_to_layout_68, False)
        ttnn_reshape_87 = ttnn.reshape(
            ttnn_matmul_16,
            [16, 160, 1],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_matmul_16, False)
        ttnn_matmul_17 = ttnn.matmul(
            ttnn_typecast_46,
            ttnn_reshape_87,
            transpose_a=False,
            transpose_b=False,
            memory_config=dram_mem,
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_reshape_87, False)
        ttnn.deallocate(ttnn_typecast_46, False)
        ttnn_reshape_88 = ttnn.reshape(
            ttnn_matmul_17,
            [16, 8],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_matmul_17, False)
        ttnn_permute_31 = ttnn.permute(
            ttnn_reshape_88,
            [1, 0],
            memory_config=dram_mem,
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_88, False)
        ttnn_reshape_89 = ttnn.reshape(
            ttnn_permute_31,
            [8, 1, 16, 1],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_permute_31, False)
        ttnn_multiply_5 = ttnn.multiply(
            ttnn_typecast_54,
            ttnn_reshape_89,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_reshape_89, False)
        ttnn.deallocate(ttnn_typecast_54, False)
        ttnn_sum_2 = ttnn.sum(
            ttnn_multiply_5,
            [0],
            False,
            memory_config=dram_mem,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_5, False)
        ttnn_typecast_55 = ttnn.typecast(
            ttnn_sum_2,
            ttnn.DataType.BFLOAT16,
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_sum_2, False)
        sparse_output = ttnn.reshape(
            ttnn_typecast_55,
            [16, 5120],
            memory_config=dram_mem,
        )
        ttnn.deallocate(ttnn_typecast_55, False)
        # Shared experts
        shared_gate = ttnn.matmul(
            hidden_states,
            self.weights[f"{layer_prefix}.shared_experts.gate_proj.weight.t"],
            transpose_a=False,
            transpose_b=False,
            memory_config=dram_mem,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
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
            program_config=None,
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
        shared_rs = ttnn.reduce_scatter(
            input_tensor=shared_reshaped,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=dram_mem,
            num_links=None,
            topology=ttnn.Topology.Linear,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(shared_reshaped, False)
        shared_rs_reshaped = ttnn.reshape(
            shared_rs,
            [16, 640],
            memory_config=dram_mem,
        )
        ttnn.deallocate(shared_rs, False)
        shared_ag = ttnn.all_gather(
            input_tensor=shared_rs_reshaped,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=dram_mem,
            num_links=None,
            topology=ttnn.Topology.Linear,
        )
        ttnn.deallocate(shared_rs_reshaped, False)
        # Combine sparse + shared
        moe_output = ttnn.add(
            sparse_output,
            shared_ag,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=dram_mem,
        )
        ttnn.deallocate(shared_ag, False)
        ttnn.deallocate(sparse_output, False)
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
        # Input layernorm
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[f"{layer_prefix}.input_layernorm.weight"],
            bias=None,
            residual_input_tensor=None,
            memory_config=dram_mem,
            program_config=None,
            compute_kernel_config=hifi4_config,
        )
        # Attention
        ttnn.tracy_message(f"`TT_SIGNPOST: attn_L{self.layer_idx}_start`")
        attn_output, key_cache_out, value_cache_out = self.self_attn(
            normed, key_cache_input, value_cache_input, cos, sin, repeat_idx, attn_mask
        )
        ttnn.tracy_message(f"`TT_SIGNPOST: attn_L{self.layer_idx}_end`")
        # Residual add after attention
        residual = ttnn.add(
            hidden_states,
            attn_output,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=dram_mem,
        )
        ttnn.deallocate(attn_output, False)
        ttnn.deallocate(hidden_states, False)
        if self.is_moe:
            # MoE path: reshape before post_attention_layernorm
            reshaped_for_norm = ttnn.reshape(
                residual,
                [16, 1, 5120],
                memory_config=dram_mem,
            )
            post_normed = ttnn.rms_norm(
                reshaped_for_norm,
                epsilon=9.9999997473787516e-06,
                weight=self.weights[f"{layer_prefix}.post_attention_layernorm.weight"],
                bias=None,
                residual_input_tensor=None,
                memory_config=dram_mem,
                program_config=None,
                compute_kernel_config=hifi4_config,
            )
            ttnn.deallocate(reshaped_for_norm, False)
            # Pass post_normed [16, 1, 5120] to MoE - it creates both
            # [16, 5120] for router/shared experts and [16, 1, 1, 5120] for all_gather
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
            post_normed = ttnn.rms_norm(
                residual,
                epsilon=9.9999997473787516e-06,
                weight=self.weights[f"{layer_prefix}.post_attention_layernorm.weight"],
                bias=None,
                residual_input_tensor=None,
                memory_config=dram_mem,
                program_config=None,
                compute_kernel_config=hifi4_config,
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
