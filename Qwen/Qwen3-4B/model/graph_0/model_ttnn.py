import ttnn
import params
import consteval


class LightweightModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ModelTTNN(LightweightModule):
    def __init__(self, device):
        self.device = device
        self.weights = params.load_weights_for__main_from_state_dict(device)
        self.weights = consteval.run_consteval(self.weights, device)
        self.layers = [Qwen3DecoderLayer(self.weights, i) for i in range(36)]

    def forward(self, activations):
        args_0 = activations[0]
        cos = self.weights["L__self___model_rotary_emb_cos"]
        sin = self.weights["L__self___model_rotary_emb_sin"]
        ttnn_typecast_0 = ttnn.typecast(
            args_0,
            ttnn.DataType.UINT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(args_0, False)
        ttnn_reshape_0 = ttnn.reshape(
            ttnn_typecast_0,
            [16],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_0, False)
        ttnn_embedding_0 = ttnn.embedding(
            ttnn_reshape_0,
            self.weights["L__self___model_embed_tokens.weight"],
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_0, False)
        embed_residual = ttnn.reshape(
            ttnn_embedding_0,
            [1, 16, 2560],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        embed_norm_input = ttnn.to_memory_config(
            ttnn_embedding_0,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 0))]
                    ),
                    [32, 256],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_embedding_0, False)

        hidden_states, v_cache, k_cache = self.layers[0](
            embed_norm_input, embed_residual, cos, sin
        )
        kv_caches = [v_cache, k_cache]

        for i in range(1, 36):
            bridge_inter = ttnn.to_memory_config(
                hidden_states,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            bridge_reshape = ttnn.reshape(
                bridge_inter,
                [16, 2560],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(bridge_inter, False)
            bridge_block = ttnn.to_memory_config(
                bridge_reshape,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 0))]
                        ),
                        [32, 256],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(bridge_reshape, False)
            hidden_states, v_cache, k_cache = self.layers[i](
                bridge_block, hidden_states, cos, sin
            )
            kv_caches.extend([v_cache, k_cache])

        final_inter = ttnn.to_memory_config(
            hidden_states,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(hidden_states, False)
        final_reshape = ttnn.reshape(
            final_inter,
            [16, 2560],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(final_inter, False)
        final_block = ttnn.to_memory_config(
            final_reshape,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 0))]
                    ),
                    [32, 256],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(final_reshape, False)
        final_norm = ttnn.rms_norm(
            final_block,
            epsilon=9.9999999747524271e-07,
            weight=self.weights["L__self___model_norm_weight"],
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 0))]
                    ),
                    [32, 256],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(final_block, False)
        lm_head_input = ttnn.to_memory_config(
            final_norm,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 2)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 3), ttnn.CoreCoord(6, 3)),
                        ]
                    ),
                    [32, 64],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(final_norm, False)
        logits_matmul = ttnn.matmul(
            lm_head_input,
            self.weights["L__self___lm_head.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(8, 9)),
                        ]
                    ),
                    [32, 1408],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=4,
                out_block_h=1,
                out_block_w=44,
                per_core_M=1,
                per_core_N=44,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
                gather_in0=False,
                hop_cores=ttnn.CoreRangeSet([]),
                num_global_cb_receivers=0,
                untilize_out=False,
            ),
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(lm_head_input, False)
        logits_inter = ttnn.to_memory_config(
            logits_matmul,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(logits_matmul, False)
        logits_reshape = ttnn.reshape(
            logits_inter,
            [1, 16, 151936],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(logits_inter, False)
        logits = ttnn.to_memory_config(
            logits_reshape,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(logits_reshape, False)
        return kv_caches + [logits]


class Qwen3DecoderLayer(LightweightModule):
    def __init__(self, weights, layer_idx):
        self.weights = weights
        self.layer_idx = layer_idx
        self.self_attn = Qwen3Attention(weights, layer_idx)
        self.mlp = Qwen3MLP(weights, layer_idx)

    def forward(self, hidden_states, residual, cos, sin):
        input_layernorm_out = ttnn.rms_norm(
            hidden_states,
            epsilon=9.9999999747524271e-07,
            weight=self.weights[
                f"L__self___model_layers_{self.layer_idx}_input_layernorm_weight"
            ],
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 0))]
                    ),
                    [32, 256],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
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

        attn_out, v_cache, k_cache = self.self_attn(input_layernorm_out, cos, sin)

        attn_residual = ttnn.add(
            attn_out,
            residual,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(2, 7)),
                        ]
                    ),
                    [32, 32],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(attn_out, False)
        ttnn.deallocate(residual, False)

        post_norm_inter = ttnn.to_memory_config(
            attn_residual,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        post_norm_reshape = ttnn.reshape(
            post_norm_inter,
            [16, 2560],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(post_norm_inter, False)
        post_norm_block = ttnn.to_memory_config(
            post_norm_reshape,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 0))]
                    ),
                    [32, 256],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(post_norm_reshape, False)
        post_attention_layernorm_out = ttnn.rms_norm(
            post_norm_block,
            epsilon=9.9999999747524271e-07,
            weight=self.weights[
                f"L__self___model_layers_{self.layer_idx}_post_attention_layernorm_weight"
            ],
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 0))]
                    ),
                    [32, 256],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(post_norm_block, False)

        mlp_out = self.mlp(post_attention_layernorm_out)

        mlp_residual = ttnn.add(
            mlp_out,
            attn_residual,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(2, 7)),
                        ]
                    ),
                    [32, 32],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(mlp_out, False)
        ttnn.deallocate(attn_residual, False)

        return mlp_residual, v_cache, k_cache


class Qwen3Attention(LightweightModule):
    def __init__(self, weights, layer_idx):
        self.weights = weights
        self.layer_idx = layer_idx

    def forward(self, hidden_states, cos, sin):
        ttnn_to_memory_config_1 = ttnn.to_memory_config(
            hidden_states,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 2)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 3), ttnn.CoreCoord(6, 3)),
                        ]
                    ),
                    [32, 64],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(hidden_states, False)
        ttnn_matmul_0 = ttnn.matmul(
            ttnn_to_memory_config_1,
            self.weights[
                f"L__self___model_layers_{self.layer_idx}_self_attn_qkv_proj.weight"
            ],
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(7, 8)),
                        ]
                    ),
                    [32, 64],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=2,
                out_block_h=1,
                out_block_w=2,
                per_core_M=1,
                per_core_N=2,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
                gather_in0=False,
                hop_cores=ttnn.CoreRangeSet([]),
                num_global_cb_receivers=0,
                untilize_out=False,
            ),
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_to_memory_config_1, False)
        ttnn_to_memory_config_2 = ttnn.to_memory_config(
            ttnn_matmul_0,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn_slice_0 = ttnn.slice(
            ttnn_to_memory_config_2,
            [0, 0],
            [16, 1024],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_to_memory_config_2, False)
        ttnn_to_memory_config_3 = ttnn.to_memory_config(
            ttnn_matmul_0,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn_slice_1 = ttnn.slice(
            ttnn_to_memory_config_3,
            [0, 1024],
            [16, 5120],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_to_memory_config_3, False)
        ttnn_to_memory_config_4 = ttnn.to_memory_config(
            ttnn_matmul_0,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_0, False)
        ttnn_slice_2 = ttnn.slice(
            ttnn_to_memory_config_4,
            [0, 5120],
            [16, 6144],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_to_memory_config_4, False)
        ttnn_reshape_2 = ttnn.reshape(
            ttnn_slice_0,
            [1, 16, 8, 128],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_slice_0, False)
        ttnn_permute_0 = ttnn.permute(
            ttnn_reshape_2,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_2, False)
        v_cache = ttnn.to_memory_config(
            ttnn_permute_0,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_3 = ttnn.reshape(
            ttnn_slice_2,
            [1, 16, 8, 128],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_slice_2, False)
        ttnn_rms_norm_1 = ttnn.rms_norm(
            ttnn_reshape_3,
            epsilon=9.9999999747524271e-07,
            weight=self.weights[
                f"L__self___model_layers_{self.layer_idx}_self_attn_k_norm_weight"
            ],
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))]
                    ),
                    [64, 32],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_reshape_3, False)
        ttnn_permute_1 = ttnn.permute(
            ttnn_rms_norm_1,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_rms_norm_1, False)
        ttnn_experimental_rotary_embedding_0 = ttnn.experimental.rotary_embedding(
            ttnn_permute_1,
            cos,
            sin,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_permute_1, False)
        ttnn_slice_3 = ttnn.slice(
            ttnn_experimental_rotary_embedding_0,
            [0, 0, 0, 0],
            [1, 8, 16, 128],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_experimental_rotary_embedding_0, False)
        k_cache = ttnn.to_memory_config(
            ttnn_slice_3,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_4 = ttnn.reshape(
            ttnn_slice_1,
            [1, 16, 32, 128],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_slice_1, False)
        ttnn_to_memory_config_7 = ttnn.to_memory_config(
            ttnn_reshape_4,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))]
                    ),
                    [64, 32],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_reshape_4, False)
        ttnn_rms_norm_2 = ttnn.rms_norm(
            ttnn_to_memory_config_7,
            epsilon=9.9999999747524271e-07,
            weight=self.weights[
                f"L__self___model_layers_{self.layer_idx}_self_attn_q_norm_weight"
            ],
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))]
                    ),
                    [64, 32],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_to_memory_config_7, False)
        ttnn_permute_2 = ttnn.permute(
            ttnn_rms_norm_2,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_rms_norm_2, False)
        ttnn_experimental_rotary_embedding_1 = ttnn.experimental.rotary_embedding(
            ttnn_permute_2,
            cos,
            sin,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_permute_2, False)
        ttnn_slice_4 = ttnn.slice(
            ttnn_experimental_rotary_embedding_1,
            [0, 0, 0, 0],
            [1, 32, 16, 128],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_experimental_rotary_embedding_1, False)
        ttnn_transformer_scaled_dot_product_attention_0 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_slice_4,
                ttnn_slice_3,
                ttnn_permute_0,
                attn_mask=self.weights[
                    f"L__self___model_layers_{self.layer_idx}_causal_mask"
                ],
                is_causal=False,
                scale=0.088388338685035706,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn.deallocate(ttnn_slice_4, False)
        ttnn.deallocate(ttnn_slice_3, False)
        ttnn.deallocate(ttnn_permute_0, False)
        ttnn_transformer_concatenate_heads_0 = ttnn.transformer.concatenate_heads(
            ttnn_transformer_scaled_dot_product_attention_0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_0, False)
        ttnn_reshape_5 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_0,
            [16, 4096],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_transformer_concatenate_heads_0, False)
        attn_out = ttnn.matmul(
            ttnn_reshape_5,
            self.weights[
                f"L__self___model_layers_{self.layer_idx}_self_attn_o_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(2, 7)),
                        ]
                    ),
                    [32, 32],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 8),
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=1,
                per_core_M=1,
                per_core_N=1,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
                gather_in0=False,
                hop_cores=ttnn.CoreRangeSet([]),
                num_global_cb_receivers=0,
                untilize_out=False,
            ),
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_reshape_5, False)

        return attn_out, v_cache, k_cache


class Qwen3MLP(LightweightModule):
    def __init__(self, weights, layer_idx):
        self.weights = weights
        self.layer_idx = layer_idx

    def forward(self, hidden_states):
        ttnn_to_memory_config_0 = ttnn.to_memory_config(
            hidden_states,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 2)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 3), ttnn.CoreCoord(6, 3)),
                        ]
                    ),
                    [32, 64],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn_matmul_0 = ttnn.matmul(
            ttnn_to_memory_config_0,
            self.weights[
                f"L__self___model_layers_{self.layer_idx}_mlp_gate_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(2, 9)),
                        ]
                    ),
                    [32, 96],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=3,
                out_block_h=1,
                out_block_w=3,
                per_core_M=1,
                per_core_N=3,
                fuse_batch=True,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
                mcast_in0=True,
                gather_in0=False,
                hop_cores=ttnn.CoreRangeSet([]),
                num_global_cb_receivers=0,
                untilize_out=False,
            ),
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_to_memory_config_0, False)
        ttnn_to_memory_config_1 = ttnn.to_memory_config(
            hidden_states,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 2)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 3), ttnn.CoreCoord(6, 3)),
                        ]
                    ),
                    [32, 64],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(hidden_states, False)
        ttnn_matmul_1 = ttnn.matmul(
            ttnn_to_memory_config_1,
            self.weights[
                f"L__self___model_layers_{self.layer_idx}_mlp_up_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(2, 9)),
                        ]
                    ),
                    [32, 96],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=3,
                out_block_h=1,
                out_block_w=3,
                per_core_M=1,
                per_core_N=3,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
                gather_in0=False,
                hop_cores=ttnn.CoreRangeSet([]),
                num_global_cb_receivers=0,
                untilize_out=False,
            ),
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_to_memory_config_1, False)
        ttnn_multiply_0 = ttnn.multiply(
            ttnn_matmul_0,
            ttnn_matmul_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(2, 9)),
                        ]
                    ),
                    [32, 96],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_matmul_1, False)
        ttnn.deallocate(ttnn_matmul_0, False)
        ttnn_to_memory_config_2 = ttnn.to_memory_config(
            ttnn_multiply_0,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_0, False)
        mlp_out = ttnn.matmul(
            ttnn_to_memory_config_2,
            self.weights[
                f"L__self___model_layers_{self.layer_idx}_mlp_down_proj.weight"
            ],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(2, 7)),
                        ]
                    ),
                    [32, 32],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 8),
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=1,
                per_core_M=1,
                per_core_N=1,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
                gather_in0=False,
                hop_cores=ttnn.CoreRangeSet([]),
                num_global_cb_receivers=0,
                untilize_out=False,
            ),
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_to_memory_config_2, False)

        return mlp_out
