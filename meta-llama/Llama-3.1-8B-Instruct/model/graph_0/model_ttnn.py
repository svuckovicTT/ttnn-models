# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import consteval
import params


class LightweightModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ModelTTNN(LightweightModule):
    def __init__(self, device):
        self.device = device
        self.weights = params.load_weights_for__main_from_state_dict(device)
        self.weights = consteval.run_consteval(self.weights, device)
        self.layers = [LlamaDecoderLayer(i) for i in range(32)]

    def forward(self, activations):
        device = self.device
        weights = self.weights
        cache_position = activations[0]
        input_ids = activations[1]
        attn_mask_scalar = activations[6]

        # Per-layer KV cache and cache-position indices. The cache tensors are
        # updated in place by each decoder layer and returned as the new cache
        # state. Layer 0 occupies activations[2:6]; every subsequent layer
        # follows a regular 4-slot stride (key_idx, key, value_idx, value)
        # starting at index 7.
        key_cache_idx = [None] * 32
        key_cache = [None] * 32
        value_cache_idx = [None] * 32
        value_cache = [None] * 32

        key_cache_idx[0] = activations[2]
        key_cache[0] = activations[3]
        value_cache_idx[0] = activations[4]
        value_cache[0] = activations[5]
        for i in range(1, 32):
            base = 4 * i + 3
            key_cache_idx[i] = activations[base]
            key_cache[i] = activations[base + 1]
            value_cache_idx[i] = activations[base + 2]
            value_cache[i] = activations[base + 3]

        ttnn_typecast_548 = ttnn.typecast(
            input_ids,
            ttnn.DataType.UINT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_0 = ttnn.reshape(
            ttnn_typecast_548,
            [32],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_548, False)
        ttnn_to_memory_config_0 = ttnn.to_memory_config(
            ttnn_reshape_0,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_0, False)
        ttnn_embedding_0 = ttnn.embedding(
            ttnn_to_memory_config_0,
            weights["model.embed_tokens.parametrizations.weight.original"],
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_to_memory_config_0, False)
        ttnn_to_memory_config_1 = ttnn.to_memory_config(
            ttnn_embedding_0,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 0))]
                    ),
                    [32, 384],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )

        # Run decoder layers
        hidden, mlp_result, cos, sin, attn_mask = self.layers[0](
            ttnn_to_memory_config_1, ttnn_embedding_0,
            weights, device, None, None, None,
            key_cache_idx[0], key_cache[0], value_cache_idx[0], value_cache[0],
            cache_position=cache_position, attn_mask_scalar=attn_mask_scalar,
        )

        for i in range(1, 32):
            hidden, mlp_result, cos, sin, attn_mask = self.layers[i](
                hidden, mlp_result,
                weights, device, cos, sin, attn_mask,
                key_cache_idx[i], key_cache[i], value_cache_idx[i], value_cache[i],
            )

        # Final norm + lm_head
        ttnn_rms_norm_64 = ttnn.rms_norm(
            hidden,
            epsilon=9.9999997473787516e-06,
            weight=weights["model.norm.parametrizations.weight.original"],
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1))]
                    ),
                    [32, 192],
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
        ttnn.deallocate(hidden, False)
        ttnn_matmul_161 = ttnn.matmul(
            ttnn_rms_norm_64,
            weights["lm_head.parametrizations.weight.original"],
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(9, 9)),
                        ]
                    ),
                    [32, 1184],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=37,
                per_core_M=1,
                per_core_N=37,
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
        ttnn.deallocate(ttnn_rms_norm_64, False)
        ttnn_to_memory_config_456 = ttnn.to_memory_config(
            ttnn_matmul_161,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_to_memory_config_457 = ttnn.to_memory_config(
            ttnn_matmul_161,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_161, False)
        ttnn_reshape_164 = ttnn.reshape(
            ttnn_to_memory_config_457,
            [32, 1, 128256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_memory_config_457, False)

        # Updated KV cache (key, value per layer), then the two model outputs.
        kv_cache_out = []
        for i in range(32):
            kv_cache_out.append(key_cache[i])
            kv_cache_out.append(value_cache[i])

        return kv_cache_out + [ttnn_to_memory_config_456, ttnn_reshape_164]


class LlamaDecoderLayer(LightweightModule):
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        self.attention = LlamaAttention(layer_idx)
        self.mlp = LlamaMLP(layer_idx)

    def forward(
        self,
        hidden_states,
        residual,
        weights,
        device,
        cos,
        sin,
        attn_mask,
        key_cache_idx,
        key_cache,
        value_cache_idx,
        value_cache,
        cache_position=None,
        attn_mask_scalar=None,
    ):
        # Attention
        attn_result, cos, sin, attn_mask = self.attention(
            hidden_states, residual, weights, device, cos, sin, attn_mask,
            key_cache_idx, key_cache, value_cache_idx, value_cache,
            cache_position=cache_position, attn_mask_scalar=attn_mask_scalar,
        )

        # Attention-to-MLP shim (re-layout for MLP norm)
        mlp_hidden = ttnn.to_memory_config(
            attn_result,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1))]
                    ),
                    [32, 192],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )

        # MLP
        mlp_result = self.mlp(mlp_hidden, attn_result, weights, device)

        # MLP-to-next-layer shim (re-layout for next layer's norm)
        next_hidden = ttnn.to_memory_config(
            mlp_result,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1))]
                    ),
                    [32, 192],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        if self.layer_idx == 31:
            # Last layer: mlp_result is not consumed downstream.
            ttnn.deallocate(mlp_result, False)

        return next_hidden, mlp_result, cos, sin, attn_mask


class LlamaAttention(LightweightModule):
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states,
        residual,
        weights,
        device,
        cos,
        sin,
        attn_mask,
        key_cache_idx,
        key_cache,
        value_cache_idx,
        value_cache,
        cache_position=None,
        attn_mask_scalar=None,
    ):
        if self.layer_idx == 0:
            # Map parameters to original variable names
            ttnn_to_memory_config_1 = hidden_states
            ttnn_embedding_0 = residual
            args_1 = cache_position
            activation_2 = attn_mask_scalar
            activation_0 = key_cache_idx
            args_2 = key_cache
            activation_1 = value_cache_idx
            args_3 = value_cache

            ttnn_rms_norm_0 = ttnn.rms_norm(
                ttnn_to_memory_config_1,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.0.input_layernorm.parametrizations.weight.original"
                ],
                bias=None,
                residual_input_tensor=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 0))]
                        ),
                        [32, 384],
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
            ttnn.deallocate(ttnn_to_memory_config_1, False)
            ttnn_to_memory_config_2 = ttnn.to_memory_config(
                ttnn_rms_norm_0,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [
                                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 4)),
                                ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(8, 5)),
                            ]
                        ),
                        [32, 64],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_rms_norm_0, False)
            ttnn_matmul_0 = ttnn.matmul(
                ttnn_to_memory_config_2,
                weights["model.layers.0.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_2, False)
            ttnn_to_memory_config_3 = ttnn.to_memory_config(
                ttnn_matmul_0,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_0, False)
            ttnn_reshape_1 = ttnn.reshape(
                ttnn_to_memory_config_3,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_3, False)
            v_0, v_1, v_2 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_1,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_1, False)
            ttnn_typecast_549 = ttnn.typecast(
                args_1,
                ttnn.DataType.FLOAT32,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_reshape_2 = ttnn.reshape(
                ttnn_typecast_549,
                [1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_typecast_549, False)
            ttnn_to_layout_162 = ttnn.to_layout(
                ttnn_reshape_2,
                ttnn.Layout.TILE,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_2, False)
            ttnn_matmul_1 = ttnn.matmul(
                weights["model.rotary_emb.inv_freq"],
                ttnn_to_layout_162,
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
            ttnn.deallocate(ttnn_to_layout_162, False)
            ttnn_reshape_3 = ttnn.reshape(
                ttnn_matmul_1,
                [1, 1, 1, 64],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_1, False)
            ttnn_concat_1 = ttnn.concat(
                [ttnn_reshape_3, ttnn_reshape_3],
                3,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_3, False)
            ttnn_to_memory_config_4 = ttnn.to_memory_config(
                ttnn_concat_1,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))]
                        ),
                        [32, 32],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn_cos_0 = ttnn.cos(
                ttnn_to_memory_config_4,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))]
                        ),
                        [32, 32],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_4, False)
            ttnn_typecast_550 = ttnn.typecast(
                ttnn_cos_0,
                ttnn.DataType.BFLOAT16,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))]
                        ),
                        [32, 32],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_cos_0, False)
            ttnn_to_memory_config_5 = ttnn.to_memory_config(
                ttnn_concat_1,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))]
                        ),
                        [32, 32],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_concat_1, False)
            ttnn_sin_0 = ttnn.sin(
                ttnn_to_memory_config_5,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))]
                        ),
                        [32, 32],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_5, False)
            ttnn_typecast_551 = ttnn.typecast(
                ttnn_sin_0,
                ttnn.DataType.BFLOAT16,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))]
                        ),
                        [32, 32],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_sin_0, False)
            ttnn_to_memory_config_6 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_7 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_0 = ttnn.experimental.rotary_embedding(
                v_2,
                ttnn_to_memory_config_7,
                ttnn_to_memory_config_6,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_7, False)
            ttnn.deallocate(ttnn_to_memory_config_6, False)
            ttnn.deallocate(v_2, False)
            ttnn_slice_3 = ttnn.slice(
                ttnn_experimental_rotary_embedding_0,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_0, False)
            ttnn_reshape_4 = ttnn.reshape(
                ttnn_slice_3,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_3, False)
            ttnn_repeat_0 = ttnn.repeat(
                activation_0,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_to_memory_config_8 = ttnn.to_memory_config(
                ttnn_reshape_4,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [
                                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1)),
                                ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(9, 2)),
                            ]
                        ),
                        [32, 128],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_reshape_4, False)
            ttnn.experimental.paged_update_cache(
                args_2,
                ttnn_to_memory_config_8,
                update_idxs_tensor=ttnn_repeat_0,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_8, False)
            ttnn.deallocate(ttnn_repeat_0, False)
            ttnn_reshape_5 = ttnn.reshape(
                v_1,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_1, False)
            ttnn_repeat_1 = ttnn.repeat(
                activation_1,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_to_memory_config_9 = ttnn.to_memory_config(
                ttnn_reshape_5,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [
                                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1)),
                                ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(9, 2)),
                            ]
                        ),
                        [32, 128],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_reshape_5, False)
            ttnn.experimental.paged_update_cache(
                args_3,
                ttnn_to_memory_config_9,
                update_idxs_tensor=ttnn_repeat_1,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_9, False)
            ttnn.deallocate(ttnn_repeat_1, False)
            ttnn_to_memory_config_10 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_11 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_1 = ttnn.experimental.rotary_embedding(
                v_0,
                ttnn_to_memory_config_11,
                ttnn_to_memory_config_10,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_11, False)
            ttnn.deallocate(ttnn_to_memory_config_10, False)
            ttnn.deallocate(v_0, False)
            ttnn_slice_4 = ttnn.slice(
                ttnn_experimental_rotary_embedding_1,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_1, False)
            ttnn_reshape_6 = ttnn.reshape(
                activation_2,
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_to_memory_config_12 = ttnn.to_memory_config(
                weights["arange_128"],
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))]
                        ),
                        [32, 32],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn_ge_0 = ttnn.ge(
                ttnn_reshape_6,
                ttnn_to_memory_config_12,
                dtype=ttnn.DataType.BFLOAT16,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))]
                        ),
                        [32, 32],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_12, False)
            ttnn.deallocate(ttnn_reshape_6, False)
            ttnn_to_memory_config_13 = ttnn.to_memory_config(
                weights["attn_mask_neg_inf"],
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                        ),
                        [32, 32],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn_to_memory_config_14 = ttnn.to_memory_config(
                weights["attn_mask_zero"],
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                        ),
                        [32, 32],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn_where_0 = ttnn.where(
                ttnn_ge_0,
                ttnn_to_memory_config_14,
                ttnn_to_memory_config_13,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))]
                        ),
                        [32, 32],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_14, False)
            ttnn.deallocate(ttnn_to_memory_config_13, False)
            ttnn.deallocate(ttnn_ge_0, False)
            ttnn_reshape_7 = ttnn.reshape(
                ttnn_slice_4,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_4, False)
            ttnn_repeat_2 = ttnn.repeat(
                ttnn_where_0,
                ttnn.Shape([1, 1, 32, 1]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))]
                        ),
                        [32, 32],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_where_0, False)
            ttnn_to_memory_config_15 = ttnn.to_memory_config(
                ttnn_reshape_7,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_7, False)
            ttnn_to_memory_config_16 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_0 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_15,
                    args_2,
                    args_3,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_16,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_16, False)
            ttnn.deallocate(ttnn_to_memory_config_15, False)
            ttnn_to_memory_config_17 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_0,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [
                                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1)),
                                ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(9, 2)),
                            ]
                        ),
                        [32, 128],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_0, False)
            ttnn_experimental_nlp_concat_heads_decode_0 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_17,
                    sub_core_grids=ttnn_to_memory_config_17.memory_config().shard_spec.grid,
                    num_heads=32,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        ttnn.BufferType.L1,
                        ttnn.ShardSpec(
                            ttnn.CoreRangeSet(
                                [
                                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1)),
                                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(9, 2)),
                                ]
                            ),
                            [32, 128],
                            ttnn.ShardOrientation.ROW_MAJOR,
                        ),
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_17, False)
            ttnn_to_memory_config_18 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_0,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_0, False)
            ttnn_reshape_8 = ttnn.reshape(
                ttnn_to_memory_config_18,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_18, False)
            ttnn_matmul_2 = ttnn.matmul(
                ttnn_reshape_8,
                weights["model.layers.0.self_attn.o_proj.parametrizations.weight.original"],
                transpose_a=False,
                transpose_b=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [
                                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 4)),
                                ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(8, 5)),
                            ]
                        ),
                        [32, 64],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
                dtype=ttnn.DataType.BFLOAT16,
                program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(11, 6),
                    in0_block_w=8,
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
            ttnn.deallocate(ttnn_reshape_8, False)
            ttnn_add_0 = ttnn.add(
                ttnn_matmul_2,
                ttnn_embedding_0,
                dtype=ttnn.DataType.BFLOAT16,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [
                                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 4)),
                                ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(8, 5)),
                            ]
                        ),
                        [32, 64],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_matmul_2, False)
            ttnn.deallocate(ttnn_embedding_0, False)

            return ttnn_add_0, ttnn_typecast_550, ttnn_typecast_551, ttnn_repeat_2
        elif self.layer_idx == 31:
            # Map parameters to original variable names
            ttnn_to_memory_config_441 = hidden_states
            ttnn_add_61 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_63 = key_cache_idx
            args_64 = key_cache
            activation_64 = value_cache_idx
            args_65 = value_cache

            ttnn_rms_norm_62 = ttnn.rms_norm(
                ttnn_to_memory_config_441,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.31.input_layernorm.parametrizations.weight.original"
                ],
                bias=None,
                residual_input_tensor=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1))]
                        ),
                        [32, 192],
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
            ttnn.deallocate(ttnn_to_memory_config_441, False)
            ttnn_matmul_156 = ttnn.matmul(
                ttnn_rms_norm_62,
                weights["model.layers.31.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_62, False)
            ttnn_to_memory_config_442 = ttnn.to_memory_config(
                ttnn_matmul_156,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_156, False)
            ttnn_reshape_159 = ttnn.reshape(
                ttnn_to_memory_config_442,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_442, False)
            v_93, v_94, v_95 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_159,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_159, False)
            ttnn_to_memory_config_443 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_444 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_62 = ttnn.experimental.rotary_embedding(
                v_94,
                ttnn_to_memory_config_444,
                ttnn_to_memory_config_443,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_444, False)
            ttnn.deallocate(ttnn_to_memory_config_443, False)
            ttnn.deallocate(v_94, False)
            ttnn_slice_65 = ttnn.slice(
                ttnn_experimental_rotary_embedding_62,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_62, False)
            ttnn_reshape_160 = ttnn.reshape(
                ttnn_slice_65,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_65, False)
            ttnn_repeat_63 = ttnn.repeat(
                activation_63,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_to_memory_config_445 = ttnn.to_memory_config(
                ttnn_reshape_160,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [
                                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1)),
                                ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(9, 2)),
                            ]
                        ),
                        [32, 128],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_reshape_160, False)
            ttnn.experimental.paged_update_cache(
                args_64,
                ttnn_to_memory_config_445,
                update_idxs_tensor=ttnn_repeat_63,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_445, False)
            ttnn.deallocate(ttnn_repeat_63, False)
            ttnn_reshape_161 = ttnn.reshape(
                v_95,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_95, False)
            ttnn_repeat_64 = ttnn.repeat(
                activation_64,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_to_memory_config_446 = ttnn.to_memory_config(
                ttnn_reshape_161,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [
                                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1)),
                                ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(9, 2)),
                            ]
                        ),
                        [32, 128],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_reshape_161, False)
            ttnn.experimental.paged_update_cache(
                args_65,
                ttnn_to_memory_config_446,
                update_idxs_tensor=ttnn_repeat_64,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_446, False)
            ttnn.deallocate(ttnn_repeat_64, False)
            ttnn_to_memory_config_447 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_typecast_551, False)
            ttnn_to_memory_config_448 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_typecast_550, False)
            ttnn_experimental_rotary_embedding_63 = ttnn.experimental.rotary_embedding(
                v_93,
                ttnn_to_memory_config_448,
                ttnn_to_memory_config_447,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_448, False)
            ttnn.deallocate(ttnn_to_memory_config_447, False)
            ttnn.deallocate(v_93, False)
            ttnn_slice_66 = ttnn.slice(
                ttnn_experimental_rotary_embedding_63,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_63, False)
            ttnn_reshape_162 = ttnn.reshape(
                ttnn_slice_66,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_66, False)
            ttnn_to_memory_config_449 = ttnn.to_memory_config(
                ttnn_reshape_162,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_162, False)
            ttnn_to_memory_config_450 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_repeat_2, False)
            ttnn_transformer_scaled_dot_product_attention_decode_31 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_449,
                    args_64,
                    args_65,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_450,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_450, False)
            ttnn.deallocate(ttnn_to_memory_config_449, False)
            ttnn_to_memory_config_451 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_31,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [
                                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1)),
                                ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(9, 2)),
                            ]
                        ),
                        [32, 128],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_31, False)
            ttnn_experimental_nlp_concat_heads_decode_31 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_451,
                    sub_core_grids=ttnn_to_memory_config_451.memory_config().shard_spec.grid,
                    num_heads=32,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        ttnn.BufferType.L1,
                        ttnn.ShardSpec(
                            ttnn.CoreRangeSet(
                                [
                                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1)),
                                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(9, 2)),
                                ]
                            ),
                            [32, 128],
                            ttnn.ShardOrientation.ROW_MAJOR,
                        ),
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_451, False)
            ttnn_to_memory_config_452 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_31,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_31, False)
            ttnn_reshape_163 = ttnn.reshape(
                ttnn_to_memory_config_452,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_452, False)
            ttnn_matmul_157 = ttnn.matmul(
                ttnn_reshape_163,
                weights["model.layers.31.self_attn.o_proj.parametrizations.weight.original"],
                transpose_a=False,
                transpose_b=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [
                                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 4)),
                                ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(8, 5)),
                            ]
                        ),
                        [32, 64],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
                dtype=ttnn.DataType.BFLOAT16,
                program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(11, 6),
                    in0_block_w=8,
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
            ttnn.deallocate(ttnn_reshape_163, False)
            ttnn_add_62 = ttnn.add(
                ttnn_matmul_157,
                ttnn_add_61,
                dtype=ttnn.DataType.BFLOAT16,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [
                                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 4)),
                                ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(8, 5)),
                            ]
                        ),
                        [32, 64],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_matmul_157, False)
            ttnn.deallocate(ttnn_add_61, False)

            return ttnn_add_62, cos, sin, None
        else:
            # Map parameters to original variable names
            ttnn_to_memory_config_21 = hidden_states
            ttnn_add_1 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_3 = key_cache_idx
            args_4 = key_cache
            activation_4 = value_cache_idx
            args_5 = value_cache

            ttnn_rms_norm_2 = ttnn.rms_norm(
                ttnn_to_memory_config_21,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    f"model.layers.{self.layer_idx}.input_layernorm.parametrizations.weight.original"
                ],
                bias=None,
                residual_input_tensor=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1))]
                        ),
                        [32, 192],
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
            ttnn.deallocate(ttnn_to_memory_config_21, False)
            ttnn_matmul_6 = ttnn.matmul(
                ttnn_rms_norm_2,
                weights[f"model.layers.{self.layer_idx}.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_2, False)
            ttnn_to_memory_config_22 = ttnn.to_memory_config(
                ttnn_matmul_6,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_6, False)
            ttnn_reshape_9 = ttnn.reshape(
                ttnn_to_memory_config_22,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_22, False)
            v_3, v_4, v_5 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_9,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_9, False)
            ttnn_to_memory_config_23 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_24 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_2 = ttnn.experimental.rotary_embedding(
                v_5,
                ttnn_to_memory_config_24,
                ttnn_to_memory_config_23,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_24, False)
            ttnn.deallocate(ttnn_to_memory_config_23, False)
            ttnn.deallocate(v_5, False)
            ttnn_slice_5 = ttnn.slice(
                ttnn_experimental_rotary_embedding_2,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_2, False)
            ttnn_reshape_10 = ttnn.reshape(
                ttnn_slice_5,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_5, False)
            ttnn_repeat_3 = ttnn.repeat(
                activation_3,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_to_memory_config_25 = ttnn.to_memory_config(
                ttnn_reshape_10,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [
                                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1)),
                                ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(9, 2)),
                            ]
                        ),
                        [32, 128],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_reshape_10, False)
            ttnn.experimental.paged_update_cache(
                args_4,
                ttnn_to_memory_config_25,
                update_idxs_tensor=ttnn_repeat_3,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_25, False)
            ttnn.deallocate(ttnn_repeat_3, False)
            ttnn_reshape_11 = ttnn.reshape(
                v_4,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_4, False)
            ttnn_repeat_4 = ttnn.repeat(
                activation_4,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_to_memory_config_26 = ttnn.to_memory_config(
                ttnn_reshape_11,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [
                                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1)),
                                ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(9, 2)),
                            ]
                        ),
                        [32, 128],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_reshape_11, False)
            ttnn.experimental.paged_update_cache(
                args_5,
                ttnn_to_memory_config_26,
                update_idxs_tensor=ttnn_repeat_4,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_26, False)
            ttnn.deallocate(ttnn_repeat_4, False)
            ttnn_to_memory_config_27 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_28 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_3 = ttnn.experimental.rotary_embedding(
                v_3,
                ttnn_to_memory_config_28,
                ttnn_to_memory_config_27,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_28, False)
            ttnn.deallocate(ttnn_to_memory_config_27, False)
            ttnn.deallocate(v_3, False)
            ttnn_slice_6 = ttnn.slice(
                ttnn_experimental_rotary_embedding_3,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_3, False)
            ttnn_reshape_12 = ttnn.reshape(
                ttnn_slice_6,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_6, False)
            ttnn_to_memory_config_29 = ttnn.to_memory_config(
                ttnn_reshape_12,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_12, False)
            ttnn_to_memory_config_30 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_1 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_29,
                    args_4,
                    args_5,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_30,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_30, False)
            ttnn.deallocate(ttnn_to_memory_config_29, False)
            ttnn_to_memory_config_31 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_1,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [
                                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1)),
                                ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(9, 2)),
                            ]
                        ),
                        [32, 128],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_1, False)
            ttnn_experimental_nlp_concat_heads_decode_1 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_31,
                    sub_core_grids=ttnn_to_memory_config_31.memory_config().shard_spec.grid,
                    num_heads=32,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        ttnn.BufferType.L1,
                        ttnn.ShardSpec(
                            ttnn.CoreRangeSet(
                                [
                                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1)),
                                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(9, 2)),
                                ]
                            ),
                            [32, 128],
                            ttnn.ShardOrientation.ROW_MAJOR,
                        ),
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_31, False)
            ttnn_to_memory_config_32 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_1,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_1, False)
            ttnn_reshape_13 = ttnn.reshape(
                ttnn_to_memory_config_32,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_32, False)
            ttnn_matmul_7 = ttnn.matmul(
                ttnn_reshape_13,
                weights[f"model.layers.{self.layer_idx}.self_attn.o_proj.parametrizations.weight.original"],
                transpose_a=False,
                transpose_b=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [
                                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 4)),
                                ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(8, 5)),
                            ]
                        ),
                        [32, 64],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
                dtype=ttnn.DataType.BFLOAT16,
                program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(11, 6),
                    in0_block_w=8,
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
            ttnn.deallocate(ttnn_reshape_13, False)
            ttnn_add_2 = ttnn.add(
                ttnn_matmul_7,
                ttnn_add_1,
                dtype=ttnn.DataType.BFLOAT16,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet(
                            [
                                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 4)),
                                ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(8, 5)),
                            ]
                        ),
                        [32, 64],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            ttnn.deallocate(ttnn_matmul_7, False)
            ttnn.deallocate(ttnn_add_1, False)

            return ttnn_add_2, cos, sin, attn_mask


class LlamaMLP(LightweightModule):
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx

    def forward(self, hidden_states, residual, weights, device):
        # Map parameters to original variable names
        ttnn_to_memory_config_19 = hidden_states
        ttnn_add_0 = residual

        ttnn_rms_norm_1 = ttnn.rms_norm(
            ttnn_to_memory_config_19,
            epsilon=9.9999997473787516e-06,
            weight=weights[
                f"model.layers.{self.layer_idx}.post_attention_layernorm.parametrizations.weight.original"
            ],
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1))]
                    ),
                    [32, 192],
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
        ttnn.deallocate(ttnn_to_memory_config_19, False)
        ttnn_matmul_3 = ttnn.matmul(
            ttnn_rms_norm_1,
            weights[f"model.layers.{self.layer_idx}.mlp.gate_proj.parametrizations.weight.original"],
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(1, 8)),
                        ]
                    ),
                    [32, 160],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=5,
                out_block_h=1,
                out_block_w=5,
                per_core_M=1,
                per_core_N=5,
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
        ttnn_matmul_4 = ttnn.matmul(
            ttnn_rms_norm_1,
            weights[f"model.layers.{self.layer_idx}.mlp.up_proj.parametrizations.weight.original"],
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(1, 8)),
                        ]
                    ),
                    [32, 160],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=5,
                out_block_h=1,
                out_block_w=5,
                per_core_M=1,
                per_core_N=5,
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
        ttnn.deallocate(ttnn_rms_norm_1, False)
        ttnn_multiply_0 = ttnn.multiply(
            ttnn_matmul_3,
            ttnn_matmul_4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(1, 8)),
                        ]
                    ),
                    [32, 160],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_matmul_4, False)
        ttnn.deallocate(ttnn_matmul_3, False)
        ttnn_to_memory_config_20 = ttnn.to_memory_config(
            ttnn_multiply_0,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 4)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(0, 5)),
                        ]
                    ),
                    [32, 256],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_multiply_0, False)
        ttnn_matmul_5 = ttnn.matmul(
            ttnn_to_memory_config_20,
            weights[f"model.layers.{self.layer_idx}.mlp.down_proj.parametrizations.weight.original"],
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 4)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(8, 5)),
                        ]
                    ),
                    [32, 64],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 6),
                in0_block_w=8,
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
        ttnn.deallocate(ttnn_to_memory_config_20, False)
        ttnn_add_1 = ttnn.add(
            ttnn_matmul_5,
            ttnn_add_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 4)),
                            ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(8, 5)),
                        ]
                    ),
                    [32, 64],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_matmul_5, False)
        ttnn.deallocate(ttnn_add_0, False)

        return ttnn_add_1

