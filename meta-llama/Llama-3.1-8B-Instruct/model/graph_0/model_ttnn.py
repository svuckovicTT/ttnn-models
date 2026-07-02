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
        ttnn.deallocate(input_ids, False)
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
            ttnn.deallocate(args_1, False)
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
            ttnn.deallocate(activation_0, False)
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
            ttnn.deallocate(activation_1, False)
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
            ttnn.deallocate(activation_2, False)
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

        elif self.layer_idx == 1:
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
                    "model.layers.1.input_layernorm.parametrizations.weight.original"
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
                weights["model.layers.1.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(activation_3, False)
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
            ttnn.deallocate(activation_4, False)
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
                weights["model.layers.1.self_attn.o_proj.parametrizations.weight.original"],
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

        elif self.layer_idx == 2:
            # Map parameters to original variable names
            ttnn_to_memory_config_35 = hidden_states
            ttnn_add_3 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_5 = key_cache_idx
            args_6 = key_cache
            activation_6 = value_cache_idx
            args_7 = value_cache

            ttnn_rms_norm_4 = ttnn.rms_norm(
                ttnn_to_memory_config_35,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.2.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_35, False)
            ttnn_matmul_11 = ttnn.matmul(
                ttnn_rms_norm_4,
                weights["model.layers.2.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_4, False)
            ttnn_to_memory_config_36 = ttnn.to_memory_config(
                ttnn_matmul_11,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_11, False)
            ttnn_reshape_14 = ttnn.reshape(
                ttnn_to_memory_config_36,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_36, False)
            v_6, v_7, v_8 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_14,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_14, False)
            ttnn_to_memory_config_37 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_38 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_4 = ttnn.experimental.rotary_embedding(
                v_8,
                ttnn_to_memory_config_38,
                ttnn_to_memory_config_37,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_38, False)
            ttnn.deallocate(ttnn_to_memory_config_37, False)
            ttnn.deallocate(v_8, False)
            ttnn_slice_7 = ttnn.slice(
                ttnn_experimental_rotary_embedding_4,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_4, False)
            ttnn_reshape_15 = ttnn.reshape(
                ttnn_slice_7,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_7, False)
            ttnn_repeat_5 = ttnn.repeat(
                activation_5,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_5, False)
            ttnn_to_memory_config_39 = ttnn.to_memory_config(
                ttnn_reshape_15,
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
            ttnn.deallocate(ttnn_reshape_15, False)
            ttnn.experimental.paged_update_cache(
                args_6,
                ttnn_to_memory_config_39,
                update_idxs_tensor=ttnn_repeat_5,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_39, False)
            ttnn.deallocate(ttnn_repeat_5, False)
            ttnn_reshape_16 = ttnn.reshape(
                v_7,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_7, False)
            ttnn_repeat_6 = ttnn.repeat(
                activation_6,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_6, False)
            ttnn_to_memory_config_40 = ttnn.to_memory_config(
                ttnn_reshape_16,
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
            ttnn.deallocate(ttnn_reshape_16, False)
            ttnn.experimental.paged_update_cache(
                args_7,
                ttnn_to_memory_config_40,
                update_idxs_tensor=ttnn_repeat_6,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_40, False)
            ttnn.deallocate(ttnn_repeat_6, False)
            ttnn_to_memory_config_41 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_42 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_5 = ttnn.experimental.rotary_embedding(
                v_6,
                ttnn_to_memory_config_42,
                ttnn_to_memory_config_41,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_42, False)
            ttnn.deallocate(ttnn_to_memory_config_41, False)
            ttnn.deallocate(v_6, False)
            ttnn_slice_8 = ttnn.slice(
                ttnn_experimental_rotary_embedding_5,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_5, False)
            ttnn_reshape_17 = ttnn.reshape(
                ttnn_slice_8,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_8, False)
            ttnn_to_memory_config_43 = ttnn.to_memory_config(
                ttnn_reshape_17,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_17, False)
            ttnn_to_memory_config_44 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_2 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_43,
                    args_6,
                    args_7,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_44,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_44, False)
            ttnn.deallocate(ttnn_to_memory_config_43, False)
            ttnn_to_memory_config_45 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_2,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_2, False)
            ttnn_experimental_nlp_concat_heads_decode_2 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_45,
                    sub_core_grids=ttnn_to_memory_config_45.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_45, False)
            ttnn_to_memory_config_46 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_2, False)
            ttnn_reshape_18 = ttnn.reshape(
                ttnn_to_memory_config_46,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_46, False)
            ttnn_matmul_12 = ttnn.matmul(
                ttnn_reshape_18,
                weights["model.layers.2.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_18, False)
            ttnn_add_4 = ttnn.add(
                ttnn_matmul_12,
                ttnn_add_3,
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
            ttnn.deallocate(ttnn_matmul_12, False)
            ttnn.deallocate(ttnn_add_3, False)

            return ttnn_add_4, cos, sin, attn_mask

        elif self.layer_idx == 3:
            # Map parameters to original variable names
            ttnn_to_memory_config_49 = hidden_states
            ttnn_add_5 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_7 = key_cache_idx
            args_8 = key_cache
            activation_8 = value_cache_idx
            args_9 = value_cache

            ttnn_rms_norm_6 = ttnn.rms_norm(
                ttnn_to_memory_config_49,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.3.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_49, False)
            ttnn_matmul_16 = ttnn.matmul(
                ttnn_rms_norm_6,
                weights["model.layers.3.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_6, False)
            ttnn_to_memory_config_50 = ttnn.to_memory_config(
                ttnn_matmul_16,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_16, False)
            ttnn_reshape_19 = ttnn.reshape(
                ttnn_to_memory_config_50,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_50, False)
            v_9, v_10, v_11 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_19,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_19, False)
            ttnn_to_memory_config_51 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_52 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_6 = ttnn.experimental.rotary_embedding(
                v_11,
                ttnn_to_memory_config_52,
                ttnn_to_memory_config_51,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_52, False)
            ttnn.deallocate(ttnn_to_memory_config_51, False)
            ttnn.deallocate(v_11, False)
            ttnn_slice_9 = ttnn.slice(
                ttnn_experimental_rotary_embedding_6,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_6, False)
            ttnn_reshape_20 = ttnn.reshape(
                ttnn_slice_9,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_9, False)
            ttnn_repeat_7 = ttnn.repeat(
                activation_7,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_7, False)
            ttnn_to_memory_config_53 = ttnn.to_memory_config(
                ttnn_reshape_20,
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
            ttnn.deallocate(ttnn_reshape_20, False)
            ttnn.experimental.paged_update_cache(
                args_8,
                ttnn_to_memory_config_53,
                update_idxs_tensor=ttnn_repeat_7,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_53, False)
            ttnn.deallocate(ttnn_repeat_7, False)
            ttnn_reshape_21 = ttnn.reshape(
                v_10,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_10, False)
            ttnn_repeat_8 = ttnn.repeat(
                activation_8,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_8, False)
            ttnn_to_memory_config_54 = ttnn.to_memory_config(
                ttnn_reshape_21,
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
            ttnn.deallocate(ttnn_reshape_21, False)
            ttnn.experimental.paged_update_cache(
                args_9,
                ttnn_to_memory_config_54,
                update_idxs_tensor=ttnn_repeat_8,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_54, False)
            ttnn.deallocate(ttnn_repeat_8, False)
            ttnn_to_memory_config_55 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_56 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_7 = ttnn.experimental.rotary_embedding(
                v_9,
                ttnn_to_memory_config_56,
                ttnn_to_memory_config_55,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_56, False)
            ttnn.deallocate(ttnn_to_memory_config_55, False)
            ttnn.deallocate(v_9, False)
            ttnn_slice_10 = ttnn.slice(
                ttnn_experimental_rotary_embedding_7,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_7, False)
            ttnn_reshape_22 = ttnn.reshape(
                ttnn_slice_10,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_10, False)
            ttnn_to_memory_config_57 = ttnn.to_memory_config(
                ttnn_reshape_22,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_22, False)
            ttnn_to_memory_config_58 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_3 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_57,
                    args_8,
                    args_9,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_58,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_58, False)
            ttnn.deallocate(ttnn_to_memory_config_57, False)
            ttnn_to_memory_config_59 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_3,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_3, False)
            ttnn_experimental_nlp_concat_heads_decode_3 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_59,
                    sub_core_grids=ttnn_to_memory_config_59.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_59, False)
            ttnn_to_memory_config_60 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_3,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_3, False)
            ttnn_reshape_23 = ttnn.reshape(
                ttnn_to_memory_config_60,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_60, False)
            ttnn_matmul_17 = ttnn.matmul(
                ttnn_reshape_23,
                weights["model.layers.3.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_23, False)
            ttnn_add_6 = ttnn.add(
                ttnn_matmul_17,
                ttnn_add_5,
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
            ttnn.deallocate(ttnn_matmul_17, False)
            ttnn.deallocate(ttnn_add_5, False)

            return ttnn_add_6, cos, sin, attn_mask

        elif self.layer_idx == 4:
            # Map parameters to original variable names
            ttnn_to_memory_config_63 = hidden_states
            ttnn_add_7 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_9 = key_cache_idx
            args_10 = key_cache
            activation_10 = value_cache_idx
            args_11 = value_cache

            ttnn_rms_norm_8 = ttnn.rms_norm(
                ttnn_to_memory_config_63,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.4.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_63, False)
            ttnn_matmul_21 = ttnn.matmul(
                ttnn_rms_norm_8,
                weights["model.layers.4.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_8, False)
            ttnn_to_memory_config_64 = ttnn.to_memory_config(
                ttnn_matmul_21,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_21, False)
            ttnn_reshape_24 = ttnn.reshape(
                ttnn_to_memory_config_64,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_64, False)
            v_12, v_13, v_14 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_24,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_24, False)
            ttnn_to_memory_config_65 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_66 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_8 = ttnn.experimental.rotary_embedding(
                v_14,
                ttnn_to_memory_config_66,
                ttnn_to_memory_config_65,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_66, False)
            ttnn.deallocate(ttnn_to_memory_config_65, False)
            ttnn.deallocate(v_14, False)
            ttnn_slice_11 = ttnn.slice(
                ttnn_experimental_rotary_embedding_8,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_8, False)
            ttnn_reshape_25 = ttnn.reshape(
                ttnn_slice_11,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_11, False)
            ttnn_repeat_9 = ttnn.repeat(
                activation_9,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_9, False)
            ttnn_to_memory_config_67 = ttnn.to_memory_config(
                ttnn_reshape_25,
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
            ttnn.deallocate(ttnn_reshape_25, False)
            ttnn.experimental.paged_update_cache(
                args_10,
                ttnn_to_memory_config_67,
                update_idxs_tensor=ttnn_repeat_9,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_67, False)
            ttnn.deallocate(ttnn_repeat_9, False)
            ttnn_reshape_26 = ttnn.reshape(
                v_13,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_13, False)
            ttnn_repeat_10 = ttnn.repeat(
                activation_10,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_10, False)
            ttnn_to_memory_config_68 = ttnn.to_memory_config(
                ttnn_reshape_26,
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
            ttnn.deallocate(ttnn_reshape_26, False)
            ttnn.experimental.paged_update_cache(
                args_11,
                ttnn_to_memory_config_68,
                update_idxs_tensor=ttnn_repeat_10,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_68, False)
            ttnn.deallocate(ttnn_repeat_10, False)
            ttnn_to_memory_config_69 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_70 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_9 = ttnn.experimental.rotary_embedding(
                v_12,
                ttnn_to_memory_config_70,
                ttnn_to_memory_config_69,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_70, False)
            ttnn.deallocate(ttnn_to_memory_config_69, False)
            ttnn.deallocate(v_12, False)
            ttnn_slice_12 = ttnn.slice(
                ttnn_experimental_rotary_embedding_9,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_9, False)
            ttnn_reshape_27 = ttnn.reshape(
                ttnn_slice_12,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_12, False)
            ttnn_to_memory_config_71 = ttnn.to_memory_config(
                ttnn_reshape_27,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_27, False)
            ttnn_to_memory_config_72 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_4 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_71,
                    args_10,
                    args_11,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_72,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_72, False)
            ttnn.deallocate(ttnn_to_memory_config_71, False)
            ttnn_to_memory_config_73 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_4,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_4, False)
            ttnn_experimental_nlp_concat_heads_decode_4 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_73,
                    sub_core_grids=ttnn_to_memory_config_73.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_73, False)
            ttnn_to_memory_config_74 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_4,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_4, False)
            ttnn_reshape_28 = ttnn.reshape(
                ttnn_to_memory_config_74,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_74, False)
            ttnn_matmul_22 = ttnn.matmul(
                ttnn_reshape_28,
                weights["model.layers.4.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_28, False)
            ttnn_add_8 = ttnn.add(
                ttnn_matmul_22,
                ttnn_add_7,
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
            ttnn.deallocate(ttnn_matmul_22, False)
            ttnn.deallocate(ttnn_add_7, False)

            return ttnn_add_8, cos, sin, attn_mask

        elif self.layer_idx == 5:
            # Map parameters to original variable names
            ttnn_to_memory_config_77 = hidden_states
            ttnn_add_9 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_11 = key_cache_idx
            args_12 = key_cache
            activation_12 = value_cache_idx
            args_13 = value_cache

            ttnn_rms_norm_10 = ttnn.rms_norm(
                ttnn_to_memory_config_77,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.5.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_77, False)
            ttnn_matmul_26 = ttnn.matmul(
                ttnn_rms_norm_10,
                weights["model.layers.5.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_10, False)
            ttnn_to_memory_config_78 = ttnn.to_memory_config(
                ttnn_matmul_26,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_26, False)
            ttnn_reshape_29 = ttnn.reshape(
                ttnn_to_memory_config_78,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_78, False)
            v_15, v_16, v_17 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_29,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_29, False)
            ttnn_to_memory_config_79 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_80 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_10 = ttnn.experimental.rotary_embedding(
                v_17,
                ttnn_to_memory_config_80,
                ttnn_to_memory_config_79,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_80, False)
            ttnn.deallocate(ttnn_to_memory_config_79, False)
            ttnn.deallocate(v_17, False)
            ttnn_slice_13 = ttnn.slice(
                ttnn_experimental_rotary_embedding_10,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_10, False)
            ttnn_reshape_30 = ttnn.reshape(
                ttnn_slice_13,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_13, False)
            ttnn_repeat_11 = ttnn.repeat(
                activation_11,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_11, False)
            ttnn_to_memory_config_81 = ttnn.to_memory_config(
                ttnn_reshape_30,
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
            ttnn.deallocate(ttnn_reshape_30, False)
            ttnn.experimental.paged_update_cache(
                args_12,
                ttnn_to_memory_config_81,
                update_idxs_tensor=ttnn_repeat_11,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_81, False)
            ttnn.deallocate(ttnn_repeat_11, False)
            ttnn_reshape_31 = ttnn.reshape(
                v_16,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_16, False)
            ttnn_repeat_12 = ttnn.repeat(
                activation_12,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_12, False)
            ttnn_to_memory_config_82 = ttnn.to_memory_config(
                ttnn_reshape_31,
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
            ttnn.deallocate(ttnn_reshape_31, False)
            ttnn.experimental.paged_update_cache(
                args_13,
                ttnn_to_memory_config_82,
                update_idxs_tensor=ttnn_repeat_12,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_82, False)
            ttnn.deallocate(ttnn_repeat_12, False)
            ttnn_to_memory_config_83 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_84 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_11 = ttnn.experimental.rotary_embedding(
                v_15,
                ttnn_to_memory_config_84,
                ttnn_to_memory_config_83,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_84, False)
            ttnn.deallocate(ttnn_to_memory_config_83, False)
            ttnn.deallocate(v_15, False)
            ttnn_slice_14 = ttnn.slice(
                ttnn_experimental_rotary_embedding_11,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_11, False)
            ttnn_reshape_32 = ttnn.reshape(
                ttnn_slice_14,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_14, False)
            ttnn_to_memory_config_85 = ttnn.to_memory_config(
                ttnn_reshape_32,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_32, False)
            ttnn_to_memory_config_86 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_5 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_85,
                    args_12,
                    args_13,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_86,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_86, False)
            ttnn.deallocate(ttnn_to_memory_config_85, False)
            ttnn_to_memory_config_87 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_5,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_5, False)
            ttnn_experimental_nlp_concat_heads_decode_5 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_87,
                    sub_core_grids=ttnn_to_memory_config_87.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_87, False)
            ttnn_to_memory_config_88 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_5,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_5, False)
            ttnn_reshape_33 = ttnn.reshape(
                ttnn_to_memory_config_88,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_88, False)
            ttnn_matmul_27 = ttnn.matmul(
                ttnn_reshape_33,
                weights["model.layers.5.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_33, False)
            ttnn_add_10 = ttnn.add(
                ttnn_matmul_27,
                ttnn_add_9,
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
            ttnn.deallocate(ttnn_matmul_27, False)
            ttnn.deallocate(ttnn_add_9, False)

            return ttnn_add_10, cos, sin, attn_mask

        elif self.layer_idx == 6:
            # Map parameters to original variable names
            ttnn_to_memory_config_91 = hidden_states
            ttnn_add_11 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_13 = key_cache_idx
            args_14 = key_cache
            activation_14 = value_cache_idx
            args_15 = value_cache

            ttnn_rms_norm_12 = ttnn.rms_norm(
                ttnn_to_memory_config_91,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.6.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_91, False)
            ttnn_matmul_31 = ttnn.matmul(
                ttnn_rms_norm_12,
                weights["model.layers.6.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_12, False)
            ttnn_to_memory_config_92 = ttnn.to_memory_config(
                ttnn_matmul_31,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_31, False)
            ttnn_reshape_34 = ttnn.reshape(
                ttnn_to_memory_config_92,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_92, False)
            v_18, v_19, v_20 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_34,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_34, False)
            ttnn_to_memory_config_93 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_94 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_12 = ttnn.experimental.rotary_embedding(
                v_20,
                ttnn_to_memory_config_94,
                ttnn_to_memory_config_93,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_94, False)
            ttnn.deallocate(ttnn_to_memory_config_93, False)
            ttnn.deallocate(v_20, False)
            ttnn_slice_15 = ttnn.slice(
                ttnn_experimental_rotary_embedding_12,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_12, False)
            ttnn_reshape_35 = ttnn.reshape(
                ttnn_slice_15,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_15, False)
            ttnn_repeat_13 = ttnn.repeat(
                activation_13,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_13, False)
            ttnn_to_memory_config_95 = ttnn.to_memory_config(
                ttnn_reshape_35,
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
            ttnn.deallocate(ttnn_reshape_35, False)
            ttnn.experimental.paged_update_cache(
                args_14,
                ttnn_to_memory_config_95,
                update_idxs_tensor=ttnn_repeat_13,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_95, False)
            ttnn.deallocate(ttnn_repeat_13, False)
            ttnn_reshape_36 = ttnn.reshape(
                v_19,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_19, False)
            ttnn_repeat_14 = ttnn.repeat(
                activation_14,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_14, False)
            ttnn_to_memory_config_96 = ttnn.to_memory_config(
                ttnn_reshape_36,
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
            ttnn.deallocate(ttnn_reshape_36, False)
            ttnn.experimental.paged_update_cache(
                args_15,
                ttnn_to_memory_config_96,
                update_idxs_tensor=ttnn_repeat_14,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_96, False)
            ttnn.deallocate(ttnn_repeat_14, False)
            ttnn_to_memory_config_97 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_98 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_13 = ttnn.experimental.rotary_embedding(
                v_18,
                ttnn_to_memory_config_98,
                ttnn_to_memory_config_97,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_98, False)
            ttnn.deallocate(ttnn_to_memory_config_97, False)
            ttnn.deallocate(v_18, False)
            ttnn_slice_16 = ttnn.slice(
                ttnn_experimental_rotary_embedding_13,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_13, False)
            ttnn_reshape_37 = ttnn.reshape(
                ttnn_slice_16,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_16, False)
            ttnn_to_memory_config_99 = ttnn.to_memory_config(
                ttnn_reshape_37,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_37, False)
            ttnn_to_memory_config_100 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_6 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_99,
                    args_14,
                    args_15,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_100,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_100, False)
            ttnn.deallocate(ttnn_to_memory_config_99, False)
            ttnn_to_memory_config_101 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_6,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_6, False)
            ttnn_experimental_nlp_concat_heads_decode_6 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_101,
                    sub_core_grids=ttnn_to_memory_config_101.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_101, False)
            ttnn_to_memory_config_102 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_6,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_6, False)
            ttnn_reshape_38 = ttnn.reshape(
                ttnn_to_memory_config_102,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_102, False)
            ttnn_matmul_32 = ttnn.matmul(
                ttnn_reshape_38,
                weights["model.layers.6.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_38, False)
            ttnn_add_12 = ttnn.add(
                ttnn_matmul_32,
                ttnn_add_11,
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
            ttnn.deallocate(ttnn_matmul_32, False)
            ttnn.deallocate(ttnn_add_11, False)

            return ttnn_add_12, cos, sin, attn_mask

        elif self.layer_idx == 7:
            # Map parameters to original variable names
            ttnn_to_memory_config_105 = hidden_states
            ttnn_add_13 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_15 = key_cache_idx
            args_16 = key_cache
            activation_16 = value_cache_idx
            args_17 = value_cache

            ttnn_rms_norm_14 = ttnn.rms_norm(
                ttnn_to_memory_config_105,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.7.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_105, False)
            ttnn_matmul_36 = ttnn.matmul(
                ttnn_rms_norm_14,
                weights["model.layers.7.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_14, False)
            ttnn_to_memory_config_106 = ttnn.to_memory_config(
                ttnn_matmul_36,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_36, False)
            ttnn_reshape_39 = ttnn.reshape(
                ttnn_to_memory_config_106,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_106, False)
            v_21, v_22, v_23 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_39,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_39, False)
            ttnn_to_memory_config_107 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_108 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_14 = ttnn.experimental.rotary_embedding(
                v_23,
                ttnn_to_memory_config_108,
                ttnn_to_memory_config_107,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_108, False)
            ttnn.deallocate(ttnn_to_memory_config_107, False)
            ttnn.deallocate(v_23, False)
            ttnn_slice_17 = ttnn.slice(
                ttnn_experimental_rotary_embedding_14,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_14, False)
            ttnn_reshape_40 = ttnn.reshape(
                ttnn_slice_17,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_17, False)
            ttnn_repeat_15 = ttnn.repeat(
                activation_15,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_15, False)
            ttnn_to_memory_config_109 = ttnn.to_memory_config(
                ttnn_reshape_40,
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
            ttnn.deallocate(ttnn_reshape_40, False)
            ttnn.experimental.paged_update_cache(
                args_16,
                ttnn_to_memory_config_109,
                update_idxs_tensor=ttnn_repeat_15,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_109, False)
            ttnn.deallocate(ttnn_repeat_15, False)
            ttnn_reshape_41 = ttnn.reshape(
                v_22,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_22, False)
            ttnn_repeat_16 = ttnn.repeat(
                activation_16,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_16, False)
            ttnn_to_memory_config_110 = ttnn.to_memory_config(
                ttnn_reshape_41,
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
            ttnn.deallocate(ttnn_reshape_41, False)
            ttnn.experimental.paged_update_cache(
                args_17,
                ttnn_to_memory_config_110,
                update_idxs_tensor=ttnn_repeat_16,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_110, False)
            ttnn.deallocate(ttnn_repeat_16, False)
            ttnn_to_memory_config_111 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_112 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_15 = ttnn.experimental.rotary_embedding(
                v_21,
                ttnn_to_memory_config_112,
                ttnn_to_memory_config_111,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_112, False)
            ttnn.deallocate(ttnn_to_memory_config_111, False)
            ttnn.deallocate(v_21, False)
            ttnn_slice_18 = ttnn.slice(
                ttnn_experimental_rotary_embedding_15,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_15, False)
            ttnn_reshape_42 = ttnn.reshape(
                ttnn_slice_18,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_18, False)
            ttnn_to_memory_config_113 = ttnn.to_memory_config(
                ttnn_reshape_42,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_42, False)
            ttnn_to_memory_config_114 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_7 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_113,
                    args_16,
                    args_17,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_114,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_114, False)
            ttnn.deallocate(ttnn_to_memory_config_113, False)
            ttnn_to_memory_config_115 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_7,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_7, False)
            ttnn_experimental_nlp_concat_heads_decode_7 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_115,
                    sub_core_grids=ttnn_to_memory_config_115.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_115, False)
            ttnn_to_memory_config_116 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_7,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_7, False)
            ttnn_reshape_43 = ttnn.reshape(
                ttnn_to_memory_config_116,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_116, False)
            ttnn_matmul_37 = ttnn.matmul(
                ttnn_reshape_43,
                weights["model.layers.7.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_43, False)
            ttnn_add_14 = ttnn.add(
                ttnn_matmul_37,
                ttnn_add_13,
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
            ttnn.deallocate(ttnn_matmul_37, False)
            ttnn.deallocate(ttnn_add_13, False)

            return ttnn_add_14, cos, sin, attn_mask

        elif self.layer_idx == 8:
            # Map parameters to original variable names
            ttnn_to_memory_config_119 = hidden_states
            ttnn_add_15 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_17 = key_cache_idx
            args_18 = key_cache
            activation_18 = value_cache_idx
            args_19 = value_cache

            ttnn_rms_norm_16 = ttnn.rms_norm(
                ttnn_to_memory_config_119,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.8.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_119, False)
            ttnn_matmul_41 = ttnn.matmul(
                ttnn_rms_norm_16,
                weights["model.layers.8.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_16, False)
            ttnn_to_memory_config_120 = ttnn.to_memory_config(
                ttnn_matmul_41,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_41, False)
            ttnn_reshape_44 = ttnn.reshape(
                ttnn_to_memory_config_120,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_120, False)
            v_24, v_25, v_26 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_44,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_44, False)
            ttnn_to_memory_config_121 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_122 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_16 = ttnn.experimental.rotary_embedding(
                v_26,
                ttnn_to_memory_config_122,
                ttnn_to_memory_config_121,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_122, False)
            ttnn.deallocate(ttnn_to_memory_config_121, False)
            ttnn.deallocate(v_26, False)
            ttnn_slice_19 = ttnn.slice(
                ttnn_experimental_rotary_embedding_16,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_16, False)
            ttnn_reshape_45 = ttnn.reshape(
                ttnn_slice_19,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_19, False)
            ttnn_repeat_17 = ttnn.repeat(
                activation_17,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_17, False)
            ttnn_to_memory_config_123 = ttnn.to_memory_config(
                ttnn_reshape_45,
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
            ttnn.deallocate(ttnn_reshape_45, False)
            ttnn.experimental.paged_update_cache(
                args_18,
                ttnn_to_memory_config_123,
                update_idxs_tensor=ttnn_repeat_17,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_123, False)
            ttnn.deallocate(ttnn_repeat_17, False)
            ttnn_reshape_46 = ttnn.reshape(
                v_25,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_25, False)
            ttnn_repeat_18 = ttnn.repeat(
                activation_18,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_18, False)
            ttnn_to_memory_config_124 = ttnn.to_memory_config(
                ttnn_reshape_46,
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
            ttnn.deallocate(ttnn_reshape_46, False)
            ttnn.experimental.paged_update_cache(
                args_19,
                ttnn_to_memory_config_124,
                update_idxs_tensor=ttnn_repeat_18,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_124, False)
            ttnn.deallocate(ttnn_repeat_18, False)
            ttnn_to_memory_config_125 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_126 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_17 = ttnn.experimental.rotary_embedding(
                v_24,
                ttnn_to_memory_config_126,
                ttnn_to_memory_config_125,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_126, False)
            ttnn.deallocate(ttnn_to_memory_config_125, False)
            ttnn.deallocate(v_24, False)
            ttnn_slice_20 = ttnn.slice(
                ttnn_experimental_rotary_embedding_17,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_17, False)
            ttnn_reshape_47 = ttnn.reshape(
                ttnn_slice_20,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_20, False)
            ttnn_to_memory_config_127 = ttnn.to_memory_config(
                ttnn_reshape_47,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_47, False)
            ttnn_to_memory_config_128 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_8 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_127,
                    args_18,
                    args_19,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_128,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_128, False)
            ttnn.deallocate(ttnn_to_memory_config_127, False)
            ttnn_to_memory_config_129 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_8,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_8, False)
            ttnn_experimental_nlp_concat_heads_decode_8 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_129,
                    sub_core_grids=ttnn_to_memory_config_129.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_129, False)
            ttnn_to_memory_config_130 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_8,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_8, False)
            ttnn_reshape_48 = ttnn.reshape(
                ttnn_to_memory_config_130,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_130, False)
            ttnn_matmul_42 = ttnn.matmul(
                ttnn_reshape_48,
                weights["model.layers.8.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_48, False)
            ttnn_add_16 = ttnn.add(
                ttnn_matmul_42,
                ttnn_add_15,
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
            ttnn.deallocate(ttnn_matmul_42, False)
            ttnn.deallocate(ttnn_add_15, False)

            return ttnn_add_16, cos, sin, attn_mask

        elif self.layer_idx == 9:
            # Map parameters to original variable names
            ttnn_to_memory_config_133 = hidden_states
            ttnn_add_17 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_19 = key_cache_idx
            args_20 = key_cache
            activation_20 = value_cache_idx
            args_21 = value_cache

            ttnn_rms_norm_18 = ttnn.rms_norm(
                ttnn_to_memory_config_133,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.9.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_133, False)
            ttnn_matmul_46 = ttnn.matmul(
                ttnn_rms_norm_18,
                weights["model.layers.9.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_18, False)
            ttnn_to_memory_config_134 = ttnn.to_memory_config(
                ttnn_matmul_46,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_46, False)
            ttnn_reshape_49 = ttnn.reshape(
                ttnn_to_memory_config_134,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_134, False)
            v_27, v_28, v_29 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_49,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_49, False)
            ttnn_to_memory_config_135 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_136 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_18 = ttnn.experimental.rotary_embedding(
                v_29,
                ttnn_to_memory_config_136,
                ttnn_to_memory_config_135,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_136, False)
            ttnn.deallocate(ttnn_to_memory_config_135, False)
            ttnn.deallocate(v_29, False)
            ttnn_slice_21 = ttnn.slice(
                ttnn_experimental_rotary_embedding_18,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_18, False)
            ttnn_reshape_50 = ttnn.reshape(
                ttnn_slice_21,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_21, False)
            ttnn_repeat_19 = ttnn.repeat(
                activation_19,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_19, False)
            ttnn_to_memory_config_137 = ttnn.to_memory_config(
                ttnn_reshape_50,
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
            ttnn.deallocate(ttnn_reshape_50, False)
            ttnn.experimental.paged_update_cache(
                args_20,
                ttnn_to_memory_config_137,
                update_idxs_tensor=ttnn_repeat_19,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_137, False)
            ttnn.deallocate(ttnn_repeat_19, False)
            ttnn_reshape_51 = ttnn.reshape(
                v_28,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_28, False)
            ttnn_repeat_20 = ttnn.repeat(
                activation_20,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_20, False)
            ttnn_to_memory_config_138 = ttnn.to_memory_config(
                ttnn_reshape_51,
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
            ttnn.deallocate(ttnn_reshape_51, False)
            ttnn.experimental.paged_update_cache(
                args_21,
                ttnn_to_memory_config_138,
                update_idxs_tensor=ttnn_repeat_20,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_138, False)
            ttnn.deallocate(ttnn_repeat_20, False)
            ttnn_to_memory_config_139 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_140 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_19 = ttnn.experimental.rotary_embedding(
                v_27,
                ttnn_to_memory_config_140,
                ttnn_to_memory_config_139,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_140, False)
            ttnn.deallocate(ttnn_to_memory_config_139, False)
            ttnn.deallocate(v_27, False)
            ttnn_slice_22 = ttnn.slice(
                ttnn_experimental_rotary_embedding_19,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_19, False)
            ttnn_reshape_52 = ttnn.reshape(
                ttnn_slice_22,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_22, False)
            ttnn_to_memory_config_141 = ttnn.to_memory_config(
                ttnn_reshape_52,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_52, False)
            ttnn_to_memory_config_142 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_9 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_141,
                    args_20,
                    args_21,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_142,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_142, False)
            ttnn.deallocate(ttnn_to_memory_config_141, False)
            ttnn_to_memory_config_143 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_9,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_9, False)
            ttnn_experimental_nlp_concat_heads_decode_9 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_143,
                    sub_core_grids=ttnn_to_memory_config_143.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_143, False)
            ttnn_to_memory_config_144 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_9,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_9, False)
            ttnn_reshape_53 = ttnn.reshape(
                ttnn_to_memory_config_144,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_144, False)
            ttnn_matmul_47 = ttnn.matmul(
                ttnn_reshape_53,
                weights["model.layers.9.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_53, False)
            ttnn_add_18 = ttnn.add(
                ttnn_matmul_47,
                ttnn_add_17,
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
            ttnn.deallocate(ttnn_matmul_47, False)
            ttnn.deallocate(ttnn_add_17, False)

            return ttnn_add_18, cos, sin, attn_mask

        elif self.layer_idx == 10:
            # Map parameters to original variable names
            ttnn_to_memory_config_147 = hidden_states
            ttnn_add_19 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_21 = key_cache_idx
            args_22 = key_cache
            activation_22 = value_cache_idx
            args_23 = value_cache

            ttnn_rms_norm_20 = ttnn.rms_norm(
                ttnn_to_memory_config_147,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.10.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_147, False)
            ttnn_matmul_51 = ttnn.matmul(
                ttnn_rms_norm_20,
                weights["model.layers.10.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_20, False)
            ttnn_to_memory_config_148 = ttnn.to_memory_config(
                ttnn_matmul_51,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_51, False)
            ttnn_reshape_54 = ttnn.reshape(
                ttnn_to_memory_config_148,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_148, False)
            v_30, v_31, v_32 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_54,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_54, False)
            ttnn_to_memory_config_149 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_150 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_20 = ttnn.experimental.rotary_embedding(
                v_32,
                ttnn_to_memory_config_150,
                ttnn_to_memory_config_149,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_150, False)
            ttnn.deallocate(ttnn_to_memory_config_149, False)
            ttnn.deallocate(v_32, False)
            ttnn_slice_23 = ttnn.slice(
                ttnn_experimental_rotary_embedding_20,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_20, False)
            ttnn_reshape_55 = ttnn.reshape(
                ttnn_slice_23,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_23, False)
            ttnn_repeat_21 = ttnn.repeat(
                activation_21,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_21, False)
            ttnn_to_memory_config_151 = ttnn.to_memory_config(
                ttnn_reshape_55,
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
            ttnn.deallocate(ttnn_reshape_55, False)
            ttnn.experimental.paged_update_cache(
                args_22,
                ttnn_to_memory_config_151,
                update_idxs_tensor=ttnn_repeat_21,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_151, False)
            ttnn.deallocate(ttnn_repeat_21, False)
            ttnn_reshape_56 = ttnn.reshape(
                v_31,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_31, False)
            ttnn_repeat_22 = ttnn.repeat(
                activation_22,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_22, False)
            ttnn_to_memory_config_152 = ttnn.to_memory_config(
                ttnn_reshape_56,
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
            ttnn.deallocate(ttnn_reshape_56, False)
            ttnn.experimental.paged_update_cache(
                args_23,
                ttnn_to_memory_config_152,
                update_idxs_tensor=ttnn_repeat_22,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_152, False)
            ttnn.deallocate(ttnn_repeat_22, False)
            ttnn_to_memory_config_153 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_154 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_21 = ttnn.experimental.rotary_embedding(
                v_30,
                ttnn_to_memory_config_154,
                ttnn_to_memory_config_153,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_154, False)
            ttnn.deallocate(ttnn_to_memory_config_153, False)
            ttnn.deallocate(v_30, False)
            ttnn_slice_24 = ttnn.slice(
                ttnn_experimental_rotary_embedding_21,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_21, False)
            ttnn_reshape_57 = ttnn.reshape(
                ttnn_slice_24,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_24, False)
            ttnn_to_memory_config_155 = ttnn.to_memory_config(
                ttnn_reshape_57,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_57, False)
            ttnn_to_memory_config_156 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_10 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_155,
                    args_22,
                    args_23,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_156,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_156, False)
            ttnn.deallocate(ttnn_to_memory_config_155, False)
            ttnn_to_memory_config_157 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_10,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_10, False)
            ttnn_experimental_nlp_concat_heads_decode_10 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_157,
                    sub_core_grids=ttnn_to_memory_config_157.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_157, False)
            ttnn_to_memory_config_158 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_10,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_10, False)
            ttnn_reshape_58 = ttnn.reshape(
                ttnn_to_memory_config_158,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_158, False)
            ttnn_matmul_52 = ttnn.matmul(
                ttnn_reshape_58,
                weights["model.layers.10.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_58, False)
            ttnn_add_20 = ttnn.add(
                ttnn_matmul_52,
                ttnn_add_19,
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
            ttnn.deallocate(ttnn_matmul_52, False)
            ttnn.deallocate(ttnn_add_19, False)

            return ttnn_add_20, cos, sin, attn_mask

        elif self.layer_idx == 11:
            # Map parameters to original variable names
            ttnn_to_memory_config_161 = hidden_states
            ttnn_add_21 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_23 = key_cache_idx
            args_24 = key_cache
            activation_24 = value_cache_idx
            args_25 = value_cache

            ttnn_rms_norm_22 = ttnn.rms_norm(
                ttnn_to_memory_config_161,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.11.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_161, False)
            ttnn_matmul_56 = ttnn.matmul(
                ttnn_rms_norm_22,
                weights["model.layers.11.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_22, False)
            ttnn_to_memory_config_162 = ttnn.to_memory_config(
                ttnn_matmul_56,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_56, False)
            ttnn_reshape_59 = ttnn.reshape(
                ttnn_to_memory_config_162,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_162, False)
            v_33, v_34, v_35 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_59,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_59, False)
            ttnn_to_memory_config_163 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_164 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_22 = ttnn.experimental.rotary_embedding(
                v_35,
                ttnn_to_memory_config_164,
                ttnn_to_memory_config_163,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_164, False)
            ttnn.deallocate(ttnn_to_memory_config_163, False)
            ttnn.deallocate(v_35, False)
            ttnn_slice_25 = ttnn.slice(
                ttnn_experimental_rotary_embedding_22,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_22, False)
            ttnn_reshape_60 = ttnn.reshape(
                ttnn_slice_25,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_25, False)
            ttnn_repeat_23 = ttnn.repeat(
                activation_23,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_23, False)
            ttnn_to_memory_config_165 = ttnn.to_memory_config(
                ttnn_reshape_60,
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
            ttnn.deallocate(ttnn_reshape_60, False)
            ttnn.experimental.paged_update_cache(
                args_24,
                ttnn_to_memory_config_165,
                update_idxs_tensor=ttnn_repeat_23,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_165, False)
            ttnn.deallocate(ttnn_repeat_23, False)
            ttnn_reshape_61 = ttnn.reshape(
                v_34,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_34, False)
            ttnn_repeat_24 = ttnn.repeat(
                activation_24,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_24, False)
            ttnn_to_memory_config_166 = ttnn.to_memory_config(
                ttnn_reshape_61,
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
            ttnn.deallocate(ttnn_reshape_61, False)
            ttnn.experimental.paged_update_cache(
                args_25,
                ttnn_to_memory_config_166,
                update_idxs_tensor=ttnn_repeat_24,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_166, False)
            ttnn.deallocate(ttnn_repeat_24, False)
            ttnn_to_memory_config_167 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_168 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_23 = ttnn.experimental.rotary_embedding(
                v_33,
                ttnn_to_memory_config_168,
                ttnn_to_memory_config_167,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_168, False)
            ttnn.deallocate(ttnn_to_memory_config_167, False)
            ttnn.deallocate(v_33, False)
            ttnn_slice_26 = ttnn.slice(
                ttnn_experimental_rotary_embedding_23,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_23, False)
            ttnn_reshape_62 = ttnn.reshape(
                ttnn_slice_26,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_26, False)
            ttnn_to_memory_config_169 = ttnn.to_memory_config(
                ttnn_reshape_62,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_62, False)
            ttnn_to_memory_config_170 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_11 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_169,
                    args_24,
                    args_25,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_170,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_170, False)
            ttnn.deallocate(ttnn_to_memory_config_169, False)
            ttnn_to_memory_config_171 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_11,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_11, False)
            ttnn_experimental_nlp_concat_heads_decode_11 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_171,
                    sub_core_grids=ttnn_to_memory_config_171.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_171, False)
            ttnn_to_memory_config_172 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_11,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_11, False)
            ttnn_reshape_63 = ttnn.reshape(
                ttnn_to_memory_config_172,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_172, False)
            ttnn_matmul_57 = ttnn.matmul(
                ttnn_reshape_63,
                weights["model.layers.11.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_63, False)
            ttnn_add_22 = ttnn.add(
                ttnn_matmul_57,
                ttnn_add_21,
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
            ttnn.deallocate(ttnn_matmul_57, False)
            ttnn.deallocate(ttnn_add_21, False)

            return ttnn_add_22, cos, sin, attn_mask

        elif self.layer_idx == 12:
            # Map parameters to original variable names
            ttnn_to_memory_config_175 = hidden_states
            ttnn_add_23 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_25 = key_cache_idx
            args_26 = key_cache
            activation_26 = value_cache_idx
            args_27 = value_cache

            ttnn_rms_norm_24 = ttnn.rms_norm(
                ttnn_to_memory_config_175,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.12.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_175, False)
            ttnn_matmul_61 = ttnn.matmul(
                ttnn_rms_norm_24,
                weights["model.layers.12.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_24, False)
            ttnn_to_memory_config_176 = ttnn.to_memory_config(
                ttnn_matmul_61,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_61, False)
            ttnn_reshape_64 = ttnn.reshape(
                ttnn_to_memory_config_176,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_176, False)
            v_36, v_37, v_38 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_64,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_64, False)
            ttnn_to_memory_config_177 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_178 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_24 = ttnn.experimental.rotary_embedding(
                v_38,
                ttnn_to_memory_config_178,
                ttnn_to_memory_config_177,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_178, False)
            ttnn.deallocate(ttnn_to_memory_config_177, False)
            ttnn.deallocate(v_38, False)
            ttnn_slice_27 = ttnn.slice(
                ttnn_experimental_rotary_embedding_24,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_24, False)
            ttnn_reshape_65 = ttnn.reshape(
                ttnn_slice_27,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_27, False)
            ttnn_repeat_25 = ttnn.repeat(
                activation_25,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_25, False)
            ttnn_to_memory_config_179 = ttnn.to_memory_config(
                ttnn_reshape_65,
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
            ttnn.deallocate(ttnn_reshape_65, False)
            ttnn.experimental.paged_update_cache(
                args_26,
                ttnn_to_memory_config_179,
                update_idxs_tensor=ttnn_repeat_25,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_179, False)
            ttnn.deallocate(ttnn_repeat_25, False)
            ttnn_reshape_66 = ttnn.reshape(
                v_37,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_37, False)
            ttnn_repeat_26 = ttnn.repeat(
                activation_26,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_26, False)
            ttnn_to_memory_config_180 = ttnn.to_memory_config(
                ttnn_reshape_66,
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
            ttnn.deallocate(ttnn_reshape_66, False)
            ttnn.experimental.paged_update_cache(
                args_27,
                ttnn_to_memory_config_180,
                update_idxs_tensor=ttnn_repeat_26,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_180, False)
            ttnn.deallocate(ttnn_repeat_26, False)
            ttnn_to_memory_config_181 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_182 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_25 = ttnn.experimental.rotary_embedding(
                v_36,
                ttnn_to_memory_config_182,
                ttnn_to_memory_config_181,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_182, False)
            ttnn.deallocate(ttnn_to_memory_config_181, False)
            ttnn.deallocate(v_36, False)
            ttnn_slice_28 = ttnn.slice(
                ttnn_experimental_rotary_embedding_25,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_25, False)
            ttnn_reshape_67 = ttnn.reshape(
                ttnn_slice_28,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_28, False)
            ttnn_to_memory_config_183 = ttnn.to_memory_config(
                ttnn_reshape_67,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_67, False)
            ttnn_to_memory_config_184 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_12 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_183,
                    args_26,
                    args_27,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_184,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_184, False)
            ttnn.deallocate(ttnn_to_memory_config_183, False)
            ttnn_to_memory_config_185 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_12,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_12, False)
            ttnn_experimental_nlp_concat_heads_decode_12 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_185,
                    sub_core_grids=ttnn_to_memory_config_185.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_185, False)
            ttnn_to_memory_config_186 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_12,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_12, False)
            ttnn_reshape_68 = ttnn.reshape(
                ttnn_to_memory_config_186,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_186, False)
            ttnn_matmul_62 = ttnn.matmul(
                ttnn_reshape_68,
                weights["model.layers.12.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_68, False)
            ttnn_add_24 = ttnn.add(
                ttnn_matmul_62,
                ttnn_add_23,
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
            ttnn.deallocate(ttnn_matmul_62, False)
            ttnn.deallocate(ttnn_add_23, False)

            return ttnn_add_24, cos, sin, attn_mask

        elif self.layer_idx == 13:
            # Map parameters to original variable names
            ttnn_to_memory_config_189 = hidden_states
            ttnn_add_25 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_27 = key_cache_idx
            args_28 = key_cache
            activation_28 = value_cache_idx
            args_29 = value_cache

            ttnn_rms_norm_26 = ttnn.rms_norm(
                ttnn_to_memory_config_189,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.13.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_189, False)
            ttnn_matmul_66 = ttnn.matmul(
                ttnn_rms_norm_26,
                weights["model.layers.13.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_26, False)
            ttnn_to_memory_config_190 = ttnn.to_memory_config(
                ttnn_matmul_66,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_66, False)
            ttnn_reshape_69 = ttnn.reshape(
                ttnn_to_memory_config_190,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_190, False)
            v_39, v_40, v_41 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_69,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_69, False)
            ttnn_to_memory_config_191 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_192 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_26 = ttnn.experimental.rotary_embedding(
                v_41,
                ttnn_to_memory_config_192,
                ttnn_to_memory_config_191,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_192, False)
            ttnn.deallocate(ttnn_to_memory_config_191, False)
            ttnn.deallocate(v_41, False)
            ttnn_slice_29 = ttnn.slice(
                ttnn_experimental_rotary_embedding_26,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_26, False)
            ttnn_reshape_70 = ttnn.reshape(
                ttnn_slice_29,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_29, False)
            ttnn_repeat_27 = ttnn.repeat(
                activation_27,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_27, False)
            ttnn_to_memory_config_193 = ttnn.to_memory_config(
                ttnn_reshape_70,
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
            ttnn.deallocate(ttnn_reshape_70, False)
            ttnn.experimental.paged_update_cache(
                args_28,
                ttnn_to_memory_config_193,
                update_idxs_tensor=ttnn_repeat_27,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_193, False)
            ttnn.deallocate(ttnn_repeat_27, False)
            ttnn_reshape_71 = ttnn.reshape(
                v_40,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_40, False)
            ttnn_repeat_28 = ttnn.repeat(
                activation_28,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_28, False)
            ttnn_to_memory_config_194 = ttnn.to_memory_config(
                ttnn_reshape_71,
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
            ttnn.deallocate(ttnn_reshape_71, False)
            ttnn.experimental.paged_update_cache(
                args_29,
                ttnn_to_memory_config_194,
                update_idxs_tensor=ttnn_repeat_28,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_194, False)
            ttnn.deallocate(ttnn_repeat_28, False)
            ttnn_to_memory_config_195 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_196 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_27 = ttnn.experimental.rotary_embedding(
                v_39,
                ttnn_to_memory_config_196,
                ttnn_to_memory_config_195,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_196, False)
            ttnn.deallocate(ttnn_to_memory_config_195, False)
            ttnn.deallocate(v_39, False)
            ttnn_slice_30 = ttnn.slice(
                ttnn_experimental_rotary_embedding_27,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_27, False)
            ttnn_reshape_72 = ttnn.reshape(
                ttnn_slice_30,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_30, False)
            ttnn_to_memory_config_197 = ttnn.to_memory_config(
                ttnn_reshape_72,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_72, False)
            ttnn_to_memory_config_198 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_13 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_197,
                    args_28,
                    args_29,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_198,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_198, False)
            ttnn.deallocate(ttnn_to_memory_config_197, False)
            ttnn_to_memory_config_199 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_13,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_13, False)
            ttnn_experimental_nlp_concat_heads_decode_13 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_199,
                    sub_core_grids=ttnn_to_memory_config_199.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_199, False)
            ttnn_to_memory_config_200 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_13,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_13, False)
            ttnn_reshape_73 = ttnn.reshape(
                ttnn_to_memory_config_200,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_200, False)
            ttnn_matmul_67 = ttnn.matmul(
                ttnn_reshape_73,
                weights["model.layers.13.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_73, False)
            ttnn_add_26 = ttnn.add(
                ttnn_matmul_67,
                ttnn_add_25,
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
            ttnn.deallocate(ttnn_matmul_67, False)
            ttnn.deallocate(ttnn_add_25, False)

            return ttnn_add_26, cos, sin, attn_mask

        elif self.layer_idx == 14:
            # Map parameters to original variable names
            ttnn_to_memory_config_203 = hidden_states
            ttnn_add_27 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_29 = key_cache_idx
            args_30 = key_cache
            activation_30 = value_cache_idx
            args_31 = value_cache

            ttnn_rms_norm_28 = ttnn.rms_norm(
                ttnn_to_memory_config_203,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.14.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_203, False)
            ttnn_matmul_71 = ttnn.matmul(
                ttnn_rms_norm_28,
                weights["model.layers.14.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_28, False)
            ttnn_to_memory_config_204 = ttnn.to_memory_config(
                ttnn_matmul_71,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_71, False)
            ttnn_reshape_74 = ttnn.reshape(
                ttnn_to_memory_config_204,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_204, False)
            v_42, v_43, v_44 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_74,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_74, False)
            ttnn_to_memory_config_205 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_206 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_28 = ttnn.experimental.rotary_embedding(
                v_44,
                ttnn_to_memory_config_206,
                ttnn_to_memory_config_205,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_206, False)
            ttnn.deallocate(ttnn_to_memory_config_205, False)
            ttnn.deallocate(v_44, False)
            ttnn_slice_31 = ttnn.slice(
                ttnn_experimental_rotary_embedding_28,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_28, False)
            ttnn_reshape_75 = ttnn.reshape(
                ttnn_slice_31,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_31, False)
            ttnn_repeat_29 = ttnn.repeat(
                activation_29,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_29, False)
            ttnn_to_memory_config_207 = ttnn.to_memory_config(
                ttnn_reshape_75,
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
            ttnn.deallocate(ttnn_reshape_75, False)
            ttnn.experimental.paged_update_cache(
                args_30,
                ttnn_to_memory_config_207,
                update_idxs_tensor=ttnn_repeat_29,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_207, False)
            ttnn.deallocate(ttnn_repeat_29, False)
            ttnn_reshape_76 = ttnn.reshape(
                v_43,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_43, False)
            ttnn_repeat_30 = ttnn.repeat(
                activation_30,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_30, False)
            ttnn_to_memory_config_208 = ttnn.to_memory_config(
                ttnn_reshape_76,
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
            ttnn.deallocate(ttnn_reshape_76, False)
            ttnn.experimental.paged_update_cache(
                args_31,
                ttnn_to_memory_config_208,
                update_idxs_tensor=ttnn_repeat_30,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_208, False)
            ttnn.deallocate(ttnn_repeat_30, False)
            ttnn_to_memory_config_209 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_210 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_29 = ttnn.experimental.rotary_embedding(
                v_42,
                ttnn_to_memory_config_210,
                ttnn_to_memory_config_209,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_210, False)
            ttnn.deallocate(ttnn_to_memory_config_209, False)
            ttnn.deallocate(v_42, False)
            ttnn_slice_32 = ttnn.slice(
                ttnn_experimental_rotary_embedding_29,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_29, False)
            ttnn_reshape_77 = ttnn.reshape(
                ttnn_slice_32,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_32, False)
            ttnn_to_memory_config_211 = ttnn.to_memory_config(
                ttnn_reshape_77,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_77, False)
            ttnn_to_memory_config_212 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_14 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_211,
                    args_30,
                    args_31,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_212,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_212, False)
            ttnn.deallocate(ttnn_to_memory_config_211, False)
            ttnn_to_memory_config_213 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_14,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_14, False)
            ttnn_experimental_nlp_concat_heads_decode_14 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_213,
                    sub_core_grids=ttnn_to_memory_config_213.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_213, False)
            ttnn_to_memory_config_214 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_14,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_14, False)
            ttnn_reshape_78 = ttnn.reshape(
                ttnn_to_memory_config_214,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_214, False)
            ttnn_matmul_72 = ttnn.matmul(
                ttnn_reshape_78,
                weights["model.layers.14.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_78, False)
            ttnn_add_28 = ttnn.add(
                ttnn_matmul_72,
                ttnn_add_27,
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
            ttnn.deallocate(ttnn_matmul_72, False)
            ttnn.deallocate(ttnn_add_27, False)

            return ttnn_add_28, cos, sin, attn_mask

        elif self.layer_idx == 15:
            # Map parameters to original variable names
            ttnn_to_memory_config_217 = hidden_states
            ttnn_add_29 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_31 = key_cache_idx
            args_32 = key_cache
            activation_32 = value_cache_idx
            args_33 = value_cache

            ttnn_rms_norm_30 = ttnn.rms_norm(
                ttnn_to_memory_config_217,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.15.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_217, False)
            ttnn_matmul_76 = ttnn.matmul(
                ttnn_rms_norm_30,
                weights["model.layers.15.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_30, False)
            ttnn_to_memory_config_218 = ttnn.to_memory_config(
                ttnn_matmul_76,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_76, False)
            ttnn_reshape_79 = ttnn.reshape(
                ttnn_to_memory_config_218,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_218, False)
            v_45, v_46, v_47 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_79,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_79, False)
            ttnn_to_memory_config_219 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_220 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_30 = ttnn.experimental.rotary_embedding(
                v_47,
                ttnn_to_memory_config_220,
                ttnn_to_memory_config_219,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_220, False)
            ttnn.deallocate(ttnn_to_memory_config_219, False)
            ttnn.deallocate(v_47, False)
            ttnn_slice_33 = ttnn.slice(
                ttnn_experimental_rotary_embedding_30,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_30, False)
            ttnn_reshape_80 = ttnn.reshape(
                ttnn_slice_33,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_33, False)
            ttnn_repeat_31 = ttnn.repeat(
                activation_31,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_31, False)
            ttnn_to_memory_config_221 = ttnn.to_memory_config(
                ttnn_reshape_80,
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
            ttnn.deallocate(ttnn_reshape_80, False)
            ttnn.experimental.paged_update_cache(
                args_32,
                ttnn_to_memory_config_221,
                update_idxs_tensor=ttnn_repeat_31,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_221, False)
            ttnn.deallocate(ttnn_repeat_31, False)
            ttnn_reshape_81 = ttnn.reshape(
                v_46,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_46, False)
            ttnn_repeat_32 = ttnn.repeat(
                activation_32,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_32, False)
            ttnn_to_memory_config_222 = ttnn.to_memory_config(
                ttnn_reshape_81,
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
            ttnn.deallocate(ttnn_reshape_81, False)
            ttnn.experimental.paged_update_cache(
                args_33,
                ttnn_to_memory_config_222,
                update_idxs_tensor=ttnn_repeat_32,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_222, False)
            ttnn.deallocate(ttnn_repeat_32, False)
            ttnn_to_memory_config_223 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_224 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_31 = ttnn.experimental.rotary_embedding(
                v_45,
                ttnn_to_memory_config_224,
                ttnn_to_memory_config_223,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_224, False)
            ttnn.deallocate(ttnn_to_memory_config_223, False)
            ttnn.deallocate(v_45, False)
            ttnn_slice_34 = ttnn.slice(
                ttnn_experimental_rotary_embedding_31,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_31, False)
            ttnn_reshape_82 = ttnn.reshape(
                ttnn_slice_34,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_34, False)
            ttnn_to_memory_config_225 = ttnn.to_memory_config(
                ttnn_reshape_82,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_82, False)
            ttnn_to_memory_config_226 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_15 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_225,
                    args_32,
                    args_33,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_226,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_226, False)
            ttnn.deallocate(ttnn_to_memory_config_225, False)
            ttnn_to_memory_config_227 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_15,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_15, False)
            ttnn_experimental_nlp_concat_heads_decode_15 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_227,
                    sub_core_grids=ttnn_to_memory_config_227.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_227, False)
            ttnn_to_memory_config_228 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_15,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_15, False)
            ttnn_reshape_83 = ttnn.reshape(
                ttnn_to_memory_config_228,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_228, False)
            ttnn_matmul_77 = ttnn.matmul(
                ttnn_reshape_83,
                weights["model.layers.15.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_83, False)
            ttnn_add_30 = ttnn.add(
                ttnn_matmul_77,
                ttnn_add_29,
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
            ttnn.deallocate(ttnn_matmul_77, False)
            ttnn.deallocate(ttnn_add_29, False)

            return ttnn_add_30, cos, sin, attn_mask

        elif self.layer_idx == 16:
            # Map parameters to original variable names
            ttnn_to_memory_config_231 = hidden_states
            ttnn_add_31 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_33 = key_cache_idx
            args_34 = key_cache
            activation_34 = value_cache_idx
            args_35 = value_cache

            ttnn_rms_norm_32 = ttnn.rms_norm(
                ttnn_to_memory_config_231,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.16.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_231, False)
            ttnn_matmul_81 = ttnn.matmul(
                ttnn_rms_norm_32,
                weights["model.layers.16.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_32, False)
            ttnn_to_memory_config_232 = ttnn.to_memory_config(
                ttnn_matmul_81,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_81, False)
            ttnn_reshape_84 = ttnn.reshape(
                ttnn_to_memory_config_232,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_232, False)
            v_48, v_49, v_50 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_84,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_84, False)
            ttnn_to_memory_config_233 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_234 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_32 = ttnn.experimental.rotary_embedding(
                v_50,
                ttnn_to_memory_config_234,
                ttnn_to_memory_config_233,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_234, False)
            ttnn.deallocate(ttnn_to_memory_config_233, False)
            ttnn.deallocate(v_50, False)
            ttnn_slice_35 = ttnn.slice(
                ttnn_experimental_rotary_embedding_32,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_32, False)
            ttnn_reshape_85 = ttnn.reshape(
                ttnn_slice_35,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_35, False)
            ttnn_repeat_33 = ttnn.repeat(
                activation_33,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_33, False)
            ttnn_to_memory_config_235 = ttnn.to_memory_config(
                ttnn_reshape_85,
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
            ttnn.deallocate(ttnn_reshape_85, False)
            ttnn.experimental.paged_update_cache(
                args_34,
                ttnn_to_memory_config_235,
                update_idxs_tensor=ttnn_repeat_33,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_235, False)
            ttnn.deallocate(ttnn_repeat_33, False)
            ttnn_reshape_86 = ttnn.reshape(
                v_49,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_49, False)
            ttnn_repeat_34 = ttnn.repeat(
                activation_34,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_34, False)
            ttnn_to_memory_config_236 = ttnn.to_memory_config(
                ttnn_reshape_86,
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
            ttnn.deallocate(ttnn_reshape_86, False)
            ttnn.experimental.paged_update_cache(
                args_35,
                ttnn_to_memory_config_236,
                update_idxs_tensor=ttnn_repeat_34,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_236, False)
            ttnn.deallocate(ttnn_repeat_34, False)
            ttnn_to_memory_config_237 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_238 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_33 = ttnn.experimental.rotary_embedding(
                v_48,
                ttnn_to_memory_config_238,
                ttnn_to_memory_config_237,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_238, False)
            ttnn.deallocate(ttnn_to_memory_config_237, False)
            ttnn.deallocate(v_48, False)
            ttnn_slice_36 = ttnn.slice(
                ttnn_experimental_rotary_embedding_33,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_33, False)
            ttnn_reshape_87 = ttnn.reshape(
                ttnn_slice_36,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_36, False)
            ttnn_to_memory_config_239 = ttnn.to_memory_config(
                ttnn_reshape_87,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_87, False)
            ttnn_to_memory_config_240 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_16 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_239,
                    args_34,
                    args_35,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_240,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_240, False)
            ttnn.deallocate(ttnn_to_memory_config_239, False)
            ttnn_to_memory_config_241 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_16,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_16, False)
            ttnn_experimental_nlp_concat_heads_decode_16 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_241,
                    sub_core_grids=ttnn_to_memory_config_241.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_241, False)
            ttnn_to_memory_config_242 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_16,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_16, False)
            ttnn_reshape_88 = ttnn.reshape(
                ttnn_to_memory_config_242,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_242, False)
            ttnn_matmul_82 = ttnn.matmul(
                ttnn_reshape_88,
                weights["model.layers.16.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_88, False)
            ttnn_add_32 = ttnn.add(
                ttnn_matmul_82,
                ttnn_add_31,
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
            ttnn.deallocate(ttnn_matmul_82, False)
            ttnn.deallocate(ttnn_add_31, False)

            return ttnn_add_32, cos, sin, attn_mask

        elif self.layer_idx == 17:
            # Map parameters to original variable names
            ttnn_to_memory_config_245 = hidden_states
            ttnn_add_33 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_35 = key_cache_idx
            args_36 = key_cache
            activation_36 = value_cache_idx
            args_37 = value_cache

            ttnn_rms_norm_34 = ttnn.rms_norm(
                ttnn_to_memory_config_245,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.17.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_245, False)
            ttnn_matmul_86 = ttnn.matmul(
                ttnn_rms_norm_34,
                weights["model.layers.17.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_34, False)
            ttnn_to_memory_config_246 = ttnn.to_memory_config(
                ttnn_matmul_86,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_86, False)
            ttnn_reshape_89 = ttnn.reshape(
                ttnn_to_memory_config_246,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_246, False)
            v_51, v_52, v_53 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_89,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_89, False)
            ttnn_to_memory_config_247 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_248 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_34 = ttnn.experimental.rotary_embedding(
                v_53,
                ttnn_to_memory_config_248,
                ttnn_to_memory_config_247,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_248, False)
            ttnn.deallocate(ttnn_to_memory_config_247, False)
            ttnn.deallocate(v_53, False)
            ttnn_slice_37 = ttnn.slice(
                ttnn_experimental_rotary_embedding_34,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_34, False)
            ttnn_reshape_90 = ttnn.reshape(
                ttnn_slice_37,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_37, False)
            ttnn_repeat_35 = ttnn.repeat(
                activation_35,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_35, False)
            ttnn_to_memory_config_249 = ttnn.to_memory_config(
                ttnn_reshape_90,
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
            ttnn.deallocate(ttnn_reshape_90, False)
            ttnn.experimental.paged_update_cache(
                args_36,
                ttnn_to_memory_config_249,
                update_idxs_tensor=ttnn_repeat_35,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_249, False)
            ttnn.deallocate(ttnn_repeat_35, False)
            ttnn_reshape_91 = ttnn.reshape(
                v_52,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_52, False)
            ttnn_repeat_36 = ttnn.repeat(
                activation_36,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_36, False)
            ttnn_to_memory_config_250 = ttnn.to_memory_config(
                ttnn_reshape_91,
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
            ttnn.deallocate(ttnn_reshape_91, False)
            ttnn.experimental.paged_update_cache(
                args_37,
                ttnn_to_memory_config_250,
                update_idxs_tensor=ttnn_repeat_36,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_250, False)
            ttnn.deallocate(ttnn_repeat_36, False)
            ttnn_to_memory_config_251 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_252 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_35 = ttnn.experimental.rotary_embedding(
                v_51,
                ttnn_to_memory_config_252,
                ttnn_to_memory_config_251,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_252, False)
            ttnn.deallocate(ttnn_to_memory_config_251, False)
            ttnn.deallocate(v_51, False)
            ttnn_slice_38 = ttnn.slice(
                ttnn_experimental_rotary_embedding_35,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_35, False)
            ttnn_reshape_92 = ttnn.reshape(
                ttnn_slice_38,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_38, False)
            ttnn_to_memory_config_253 = ttnn.to_memory_config(
                ttnn_reshape_92,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_92, False)
            ttnn_to_memory_config_254 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_17 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_253,
                    args_36,
                    args_37,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_254,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_254, False)
            ttnn.deallocate(ttnn_to_memory_config_253, False)
            ttnn_to_memory_config_255 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_17,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_17, False)
            ttnn_experimental_nlp_concat_heads_decode_17 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_255,
                    sub_core_grids=ttnn_to_memory_config_255.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_255, False)
            ttnn_to_memory_config_256 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_17,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_17, False)
            ttnn_reshape_93 = ttnn.reshape(
                ttnn_to_memory_config_256,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_256, False)
            ttnn_matmul_87 = ttnn.matmul(
                ttnn_reshape_93,
                weights["model.layers.17.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_93, False)
            ttnn_add_34 = ttnn.add(
                ttnn_matmul_87,
                ttnn_add_33,
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
            ttnn.deallocate(ttnn_matmul_87, False)
            ttnn.deallocate(ttnn_add_33, False)

            return ttnn_add_34, cos, sin, attn_mask

        elif self.layer_idx == 18:
            # Map parameters to original variable names
            ttnn_to_memory_config_259 = hidden_states
            ttnn_add_35 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_37 = key_cache_idx
            args_38 = key_cache
            activation_38 = value_cache_idx
            args_39 = value_cache

            ttnn_rms_norm_36 = ttnn.rms_norm(
                ttnn_to_memory_config_259,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.18.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_259, False)
            ttnn_matmul_91 = ttnn.matmul(
                ttnn_rms_norm_36,
                weights["model.layers.18.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_36, False)
            ttnn_to_memory_config_260 = ttnn.to_memory_config(
                ttnn_matmul_91,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_91, False)
            ttnn_reshape_94 = ttnn.reshape(
                ttnn_to_memory_config_260,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_260, False)
            v_54, v_55, v_56 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_94,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_94, False)
            ttnn_to_memory_config_261 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_262 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_36 = ttnn.experimental.rotary_embedding(
                v_56,
                ttnn_to_memory_config_262,
                ttnn_to_memory_config_261,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_262, False)
            ttnn.deallocate(ttnn_to_memory_config_261, False)
            ttnn.deallocate(v_56, False)
            ttnn_slice_39 = ttnn.slice(
                ttnn_experimental_rotary_embedding_36,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_36, False)
            ttnn_reshape_95 = ttnn.reshape(
                ttnn_slice_39,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_39, False)
            ttnn_repeat_37 = ttnn.repeat(
                activation_37,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_37, False)
            ttnn_to_memory_config_263 = ttnn.to_memory_config(
                ttnn_reshape_95,
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
            ttnn.deallocate(ttnn_reshape_95, False)
            ttnn.experimental.paged_update_cache(
                args_38,
                ttnn_to_memory_config_263,
                update_idxs_tensor=ttnn_repeat_37,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_263, False)
            ttnn.deallocate(ttnn_repeat_37, False)
            ttnn_reshape_96 = ttnn.reshape(
                v_55,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_55, False)
            ttnn_repeat_38 = ttnn.repeat(
                activation_38,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_38, False)
            ttnn_to_memory_config_264 = ttnn.to_memory_config(
                ttnn_reshape_96,
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
            ttnn.deallocate(ttnn_reshape_96, False)
            ttnn.experimental.paged_update_cache(
                args_39,
                ttnn_to_memory_config_264,
                update_idxs_tensor=ttnn_repeat_38,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_264, False)
            ttnn.deallocate(ttnn_repeat_38, False)
            ttnn_to_memory_config_265 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_266 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_37 = ttnn.experimental.rotary_embedding(
                v_54,
                ttnn_to_memory_config_266,
                ttnn_to_memory_config_265,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_266, False)
            ttnn.deallocate(ttnn_to_memory_config_265, False)
            ttnn.deallocate(v_54, False)
            ttnn_slice_40 = ttnn.slice(
                ttnn_experimental_rotary_embedding_37,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_37, False)
            ttnn_reshape_97 = ttnn.reshape(
                ttnn_slice_40,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_40, False)
            ttnn_to_memory_config_267 = ttnn.to_memory_config(
                ttnn_reshape_97,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_97, False)
            ttnn_to_memory_config_268 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_18 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_267,
                    args_38,
                    args_39,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_268,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_268, False)
            ttnn.deallocate(ttnn_to_memory_config_267, False)
            ttnn_to_memory_config_269 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_18,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_18, False)
            ttnn_experimental_nlp_concat_heads_decode_18 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_269,
                    sub_core_grids=ttnn_to_memory_config_269.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_269, False)
            ttnn_to_memory_config_270 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_18,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_18, False)
            ttnn_reshape_98 = ttnn.reshape(
                ttnn_to_memory_config_270,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_270, False)
            ttnn_matmul_92 = ttnn.matmul(
                ttnn_reshape_98,
                weights["model.layers.18.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_98, False)
            ttnn_add_36 = ttnn.add(
                ttnn_matmul_92,
                ttnn_add_35,
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
            ttnn.deallocate(ttnn_matmul_92, False)
            ttnn.deallocate(ttnn_add_35, False)

            return ttnn_add_36, cos, sin, attn_mask

        elif self.layer_idx == 19:
            # Map parameters to original variable names
            ttnn_to_memory_config_273 = hidden_states
            ttnn_add_37 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_39 = key_cache_idx
            args_40 = key_cache
            activation_40 = value_cache_idx
            args_41 = value_cache

            ttnn_rms_norm_38 = ttnn.rms_norm(
                ttnn_to_memory_config_273,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.19.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_273, False)
            ttnn_matmul_96 = ttnn.matmul(
                ttnn_rms_norm_38,
                weights["model.layers.19.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_38, False)
            ttnn_to_memory_config_274 = ttnn.to_memory_config(
                ttnn_matmul_96,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_96, False)
            ttnn_reshape_99 = ttnn.reshape(
                ttnn_to_memory_config_274,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_274, False)
            v_57, v_58, v_59 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_99,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_99, False)
            ttnn_to_memory_config_275 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_276 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_38 = ttnn.experimental.rotary_embedding(
                v_59,
                ttnn_to_memory_config_276,
                ttnn_to_memory_config_275,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_276, False)
            ttnn.deallocate(ttnn_to_memory_config_275, False)
            ttnn.deallocate(v_59, False)
            ttnn_slice_41 = ttnn.slice(
                ttnn_experimental_rotary_embedding_38,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_38, False)
            ttnn_reshape_100 = ttnn.reshape(
                ttnn_slice_41,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_41, False)
            ttnn_repeat_39 = ttnn.repeat(
                activation_39,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_39, False)
            ttnn_to_memory_config_277 = ttnn.to_memory_config(
                ttnn_reshape_100,
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
            ttnn.deallocate(ttnn_reshape_100, False)
            ttnn.experimental.paged_update_cache(
                args_40,
                ttnn_to_memory_config_277,
                update_idxs_tensor=ttnn_repeat_39,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_277, False)
            ttnn.deallocate(ttnn_repeat_39, False)
            ttnn_reshape_101 = ttnn.reshape(
                v_58,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_58, False)
            ttnn_repeat_40 = ttnn.repeat(
                activation_40,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_40, False)
            ttnn_to_memory_config_278 = ttnn.to_memory_config(
                ttnn_reshape_101,
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
            ttnn.deallocate(ttnn_reshape_101, False)
            ttnn.experimental.paged_update_cache(
                args_41,
                ttnn_to_memory_config_278,
                update_idxs_tensor=ttnn_repeat_40,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_278, False)
            ttnn.deallocate(ttnn_repeat_40, False)
            ttnn_to_memory_config_279 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_280 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_39 = ttnn.experimental.rotary_embedding(
                v_57,
                ttnn_to_memory_config_280,
                ttnn_to_memory_config_279,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_280, False)
            ttnn.deallocate(ttnn_to_memory_config_279, False)
            ttnn.deallocate(v_57, False)
            ttnn_slice_42 = ttnn.slice(
                ttnn_experimental_rotary_embedding_39,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_39, False)
            ttnn_reshape_102 = ttnn.reshape(
                ttnn_slice_42,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_42, False)
            ttnn_to_memory_config_281 = ttnn.to_memory_config(
                ttnn_reshape_102,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_102, False)
            ttnn_to_memory_config_282 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_19 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_281,
                    args_40,
                    args_41,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_282,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_282, False)
            ttnn.deallocate(ttnn_to_memory_config_281, False)
            ttnn_to_memory_config_283 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_19,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_19, False)
            ttnn_experimental_nlp_concat_heads_decode_19 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_283,
                    sub_core_grids=ttnn_to_memory_config_283.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_283, False)
            ttnn_to_memory_config_284 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_19,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_19, False)
            ttnn_reshape_103 = ttnn.reshape(
                ttnn_to_memory_config_284,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_284, False)
            ttnn_matmul_97 = ttnn.matmul(
                ttnn_reshape_103,
                weights["model.layers.19.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_103, False)
            ttnn_add_38 = ttnn.add(
                ttnn_matmul_97,
                ttnn_add_37,
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
            ttnn.deallocate(ttnn_matmul_97, False)
            ttnn.deallocate(ttnn_add_37, False)

            return ttnn_add_38, cos, sin, attn_mask

        elif self.layer_idx == 20:
            # Map parameters to original variable names
            ttnn_to_memory_config_287 = hidden_states
            ttnn_add_39 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_41 = key_cache_idx
            args_42 = key_cache
            activation_42 = value_cache_idx
            args_43 = value_cache

            ttnn_rms_norm_40 = ttnn.rms_norm(
                ttnn_to_memory_config_287,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.20.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_287, False)
            ttnn_matmul_101 = ttnn.matmul(
                ttnn_rms_norm_40,
                weights["model.layers.20.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_40, False)
            ttnn_to_memory_config_288 = ttnn.to_memory_config(
                ttnn_matmul_101,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_101, False)
            ttnn_reshape_104 = ttnn.reshape(
                ttnn_to_memory_config_288,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_288, False)
            v_60, v_61, v_62 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_104,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_104, False)
            ttnn_to_memory_config_289 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_290 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_40 = ttnn.experimental.rotary_embedding(
                v_62,
                ttnn_to_memory_config_290,
                ttnn_to_memory_config_289,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_290, False)
            ttnn.deallocate(ttnn_to_memory_config_289, False)
            ttnn.deallocate(v_62, False)
            ttnn_slice_43 = ttnn.slice(
                ttnn_experimental_rotary_embedding_40,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_40, False)
            ttnn_reshape_105 = ttnn.reshape(
                ttnn_slice_43,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_43, False)
            ttnn_repeat_41 = ttnn.repeat(
                activation_41,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_41, False)
            ttnn_to_memory_config_291 = ttnn.to_memory_config(
                ttnn_reshape_105,
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
            ttnn.deallocate(ttnn_reshape_105, False)
            ttnn.experimental.paged_update_cache(
                args_42,
                ttnn_to_memory_config_291,
                update_idxs_tensor=ttnn_repeat_41,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_291, False)
            ttnn.deallocate(ttnn_repeat_41, False)
            ttnn_reshape_106 = ttnn.reshape(
                v_61,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_61, False)
            ttnn_repeat_42 = ttnn.repeat(
                activation_42,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_42, False)
            ttnn_to_memory_config_292 = ttnn.to_memory_config(
                ttnn_reshape_106,
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
            ttnn.deallocate(ttnn_reshape_106, False)
            ttnn.experimental.paged_update_cache(
                args_43,
                ttnn_to_memory_config_292,
                update_idxs_tensor=ttnn_repeat_42,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_292, False)
            ttnn.deallocate(ttnn_repeat_42, False)
            ttnn_to_memory_config_293 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_294 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_41 = ttnn.experimental.rotary_embedding(
                v_60,
                ttnn_to_memory_config_294,
                ttnn_to_memory_config_293,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_294, False)
            ttnn.deallocate(ttnn_to_memory_config_293, False)
            ttnn.deallocate(v_60, False)
            ttnn_slice_44 = ttnn.slice(
                ttnn_experimental_rotary_embedding_41,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_41, False)
            ttnn_reshape_107 = ttnn.reshape(
                ttnn_slice_44,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_44, False)
            ttnn_to_memory_config_295 = ttnn.to_memory_config(
                ttnn_reshape_107,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_107, False)
            ttnn_to_memory_config_296 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_20 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_295,
                    args_42,
                    args_43,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_296,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_296, False)
            ttnn.deallocate(ttnn_to_memory_config_295, False)
            ttnn_to_memory_config_297 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_20,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_20, False)
            ttnn_experimental_nlp_concat_heads_decode_20 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_297,
                    sub_core_grids=ttnn_to_memory_config_297.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_297, False)
            ttnn_to_memory_config_298 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_20,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_20, False)
            ttnn_reshape_108 = ttnn.reshape(
                ttnn_to_memory_config_298,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_298, False)
            ttnn_matmul_102 = ttnn.matmul(
                ttnn_reshape_108,
                weights["model.layers.20.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_108, False)
            ttnn_add_40 = ttnn.add(
                ttnn_matmul_102,
                ttnn_add_39,
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
            ttnn.deallocate(ttnn_matmul_102, False)
            ttnn.deallocate(ttnn_add_39, False)

            return ttnn_add_40, cos, sin, attn_mask

        elif self.layer_idx == 21:
            # Map parameters to original variable names
            ttnn_to_memory_config_301 = hidden_states
            ttnn_add_41 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_43 = key_cache_idx
            args_44 = key_cache
            activation_44 = value_cache_idx
            args_45 = value_cache

            ttnn_rms_norm_42 = ttnn.rms_norm(
                ttnn_to_memory_config_301,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.21.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_301, False)
            ttnn_matmul_106 = ttnn.matmul(
                ttnn_rms_norm_42,
                weights["model.layers.21.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_42, False)
            ttnn_to_memory_config_302 = ttnn.to_memory_config(
                ttnn_matmul_106,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_106, False)
            ttnn_reshape_109 = ttnn.reshape(
                ttnn_to_memory_config_302,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_302, False)
            v_63, v_64, v_65 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_109,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_109, False)
            ttnn_to_memory_config_303 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_304 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_42 = ttnn.experimental.rotary_embedding(
                v_65,
                ttnn_to_memory_config_304,
                ttnn_to_memory_config_303,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_304, False)
            ttnn.deallocate(ttnn_to_memory_config_303, False)
            ttnn.deallocate(v_65, False)
            ttnn_slice_45 = ttnn.slice(
                ttnn_experimental_rotary_embedding_42,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_42, False)
            ttnn_reshape_110 = ttnn.reshape(
                ttnn_slice_45,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_45, False)
            ttnn_repeat_43 = ttnn.repeat(
                activation_43,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_43, False)
            ttnn_to_memory_config_305 = ttnn.to_memory_config(
                ttnn_reshape_110,
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
            ttnn.deallocate(ttnn_reshape_110, False)
            ttnn.experimental.paged_update_cache(
                args_44,
                ttnn_to_memory_config_305,
                update_idxs_tensor=ttnn_repeat_43,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_305, False)
            ttnn.deallocate(ttnn_repeat_43, False)
            ttnn_reshape_111 = ttnn.reshape(
                v_64,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_64, False)
            ttnn_repeat_44 = ttnn.repeat(
                activation_44,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_44, False)
            ttnn_to_memory_config_306 = ttnn.to_memory_config(
                ttnn_reshape_111,
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
            ttnn.deallocate(ttnn_reshape_111, False)
            ttnn.experimental.paged_update_cache(
                args_45,
                ttnn_to_memory_config_306,
                update_idxs_tensor=ttnn_repeat_44,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_306, False)
            ttnn.deallocate(ttnn_repeat_44, False)
            ttnn_to_memory_config_307 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_308 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_43 = ttnn.experimental.rotary_embedding(
                v_63,
                ttnn_to_memory_config_308,
                ttnn_to_memory_config_307,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_308, False)
            ttnn.deallocate(ttnn_to_memory_config_307, False)
            ttnn.deallocate(v_63, False)
            ttnn_slice_46 = ttnn.slice(
                ttnn_experimental_rotary_embedding_43,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_43, False)
            ttnn_reshape_112 = ttnn.reshape(
                ttnn_slice_46,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_46, False)
            ttnn_to_memory_config_309 = ttnn.to_memory_config(
                ttnn_reshape_112,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_112, False)
            ttnn_to_memory_config_310 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_21 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_309,
                    args_44,
                    args_45,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_310,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_310, False)
            ttnn.deallocate(ttnn_to_memory_config_309, False)
            ttnn_to_memory_config_311 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_21,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_21, False)
            ttnn_experimental_nlp_concat_heads_decode_21 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_311,
                    sub_core_grids=ttnn_to_memory_config_311.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_311, False)
            ttnn_to_memory_config_312 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_21,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_21, False)
            ttnn_reshape_113 = ttnn.reshape(
                ttnn_to_memory_config_312,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_312, False)
            ttnn_matmul_107 = ttnn.matmul(
                ttnn_reshape_113,
                weights["model.layers.21.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_113, False)
            ttnn_add_42 = ttnn.add(
                ttnn_matmul_107,
                ttnn_add_41,
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
            ttnn.deallocate(ttnn_matmul_107, False)
            ttnn.deallocate(ttnn_add_41, False)

            return ttnn_add_42, cos, sin, attn_mask

        elif self.layer_idx == 22:
            # Map parameters to original variable names
            ttnn_to_memory_config_315 = hidden_states
            ttnn_add_43 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_45 = key_cache_idx
            args_46 = key_cache
            activation_46 = value_cache_idx
            args_47 = value_cache

            ttnn_rms_norm_44 = ttnn.rms_norm(
                ttnn_to_memory_config_315,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.22.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_315, False)
            ttnn_matmul_111 = ttnn.matmul(
                ttnn_rms_norm_44,
                weights["model.layers.22.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_44, False)
            ttnn_to_memory_config_316 = ttnn.to_memory_config(
                ttnn_matmul_111,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_111, False)
            ttnn_reshape_114 = ttnn.reshape(
                ttnn_to_memory_config_316,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_316, False)
            v_66, v_67, v_68 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_114,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_114, False)
            ttnn_to_memory_config_317 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_318 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_44 = ttnn.experimental.rotary_embedding(
                v_68,
                ttnn_to_memory_config_318,
                ttnn_to_memory_config_317,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_318, False)
            ttnn.deallocate(ttnn_to_memory_config_317, False)
            ttnn.deallocate(v_68, False)
            ttnn_slice_47 = ttnn.slice(
                ttnn_experimental_rotary_embedding_44,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_44, False)
            ttnn_reshape_115 = ttnn.reshape(
                ttnn_slice_47,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_47, False)
            ttnn_repeat_45 = ttnn.repeat(
                activation_45,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_45, False)
            ttnn_to_memory_config_319 = ttnn.to_memory_config(
                ttnn_reshape_115,
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
            ttnn.deallocate(ttnn_reshape_115, False)
            ttnn.experimental.paged_update_cache(
                args_46,
                ttnn_to_memory_config_319,
                update_idxs_tensor=ttnn_repeat_45,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_319, False)
            ttnn.deallocate(ttnn_repeat_45, False)
            ttnn_reshape_116 = ttnn.reshape(
                v_67,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_67, False)
            ttnn_repeat_46 = ttnn.repeat(
                activation_46,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_46, False)
            ttnn_to_memory_config_320 = ttnn.to_memory_config(
                ttnn_reshape_116,
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
            ttnn.deallocate(ttnn_reshape_116, False)
            ttnn.experimental.paged_update_cache(
                args_47,
                ttnn_to_memory_config_320,
                update_idxs_tensor=ttnn_repeat_46,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_320, False)
            ttnn.deallocate(ttnn_repeat_46, False)
            ttnn_to_memory_config_321 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_322 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_45 = ttnn.experimental.rotary_embedding(
                v_66,
                ttnn_to_memory_config_322,
                ttnn_to_memory_config_321,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_322, False)
            ttnn.deallocate(ttnn_to_memory_config_321, False)
            ttnn.deallocate(v_66, False)
            ttnn_slice_48 = ttnn.slice(
                ttnn_experimental_rotary_embedding_45,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_45, False)
            ttnn_reshape_117 = ttnn.reshape(
                ttnn_slice_48,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_48, False)
            ttnn_to_memory_config_323 = ttnn.to_memory_config(
                ttnn_reshape_117,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_117, False)
            ttnn_to_memory_config_324 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_22 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_323,
                    args_46,
                    args_47,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_324,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_324, False)
            ttnn.deallocate(ttnn_to_memory_config_323, False)
            ttnn_to_memory_config_325 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_22,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_22, False)
            ttnn_experimental_nlp_concat_heads_decode_22 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_325,
                    sub_core_grids=ttnn_to_memory_config_325.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_325, False)
            ttnn_to_memory_config_326 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_22,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_22, False)
            ttnn_reshape_118 = ttnn.reshape(
                ttnn_to_memory_config_326,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_326, False)
            ttnn_matmul_112 = ttnn.matmul(
                ttnn_reshape_118,
                weights["model.layers.22.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_118, False)
            ttnn_add_44 = ttnn.add(
                ttnn_matmul_112,
                ttnn_add_43,
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
            ttnn.deallocate(ttnn_matmul_112, False)
            ttnn.deallocate(ttnn_add_43, False)

            return ttnn_add_44, cos, sin, attn_mask

        elif self.layer_idx == 23:
            # Map parameters to original variable names
            ttnn_to_memory_config_329 = hidden_states
            ttnn_add_45 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_47 = key_cache_idx
            args_48 = key_cache
            activation_48 = value_cache_idx
            args_49 = value_cache

            ttnn_rms_norm_46 = ttnn.rms_norm(
                ttnn_to_memory_config_329,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.23.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_329, False)
            ttnn_matmul_116 = ttnn.matmul(
                ttnn_rms_norm_46,
                weights["model.layers.23.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_46, False)
            ttnn_to_memory_config_330 = ttnn.to_memory_config(
                ttnn_matmul_116,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_116, False)
            ttnn_reshape_119 = ttnn.reshape(
                ttnn_to_memory_config_330,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_330, False)
            v_69, v_70, v_71 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_119,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_119, False)
            ttnn_to_memory_config_331 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_332 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_46 = ttnn.experimental.rotary_embedding(
                v_71,
                ttnn_to_memory_config_332,
                ttnn_to_memory_config_331,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_332, False)
            ttnn.deallocate(ttnn_to_memory_config_331, False)
            ttnn.deallocate(v_71, False)
            ttnn_slice_49 = ttnn.slice(
                ttnn_experimental_rotary_embedding_46,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_46, False)
            ttnn_reshape_120 = ttnn.reshape(
                ttnn_slice_49,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_49, False)
            ttnn_repeat_47 = ttnn.repeat(
                activation_47,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_47, False)
            ttnn_to_memory_config_333 = ttnn.to_memory_config(
                ttnn_reshape_120,
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
            ttnn.deallocate(ttnn_reshape_120, False)
            ttnn.experimental.paged_update_cache(
                args_48,
                ttnn_to_memory_config_333,
                update_idxs_tensor=ttnn_repeat_47,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_333, False)
            ttnn.deallocate(ttnn_repeat_47, False)
            ttnn_reshape_121 = ttnn.reshape(
                v_70,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_70, False)
            ttnn_repeat_48 = ttnn.repeat(
                activation_48,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_48, False)
            ttnn_to_memory_config_334 = ttnn.to_memory_config(
                ttnn_reshape_121,
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
            ttnn.deallocate(ttnn_reshape_121, False)
            ttnn.experimental.paged_update_cache(
                args_49,
                ttnn_to_memory_config_334,
                update_idxs_tensor=ttnn_repeat_48,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_334, False)
            ttnn.deallocate(ttnn_repeat_48, False)
            ttnn_to_memory_config_335 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_336 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_47 = ttnn.experimental.rotary_embedding(
                v_69,
                ttnn_to_memory_config_336,
                ttnn_to_memory_config_335,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_336, False)
            ttnn.deallocate(ttnn_to_memory_config_335, False)
            ttnn.deallocate(v_69, False)
            ttnn_slice_50 = ttnn.slice(
                ttnn_experimental_rotary_embedding_47,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_47, False)
            ttnn_reshape_122 = ttnn.reshape(
                ttnn_slice_50,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_50, False)
            ttnn_to_memory_config_337 = ttnn.to_memory_config(
                ttnn_reshape_122,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_122, False)
            ttnn_to_memory_config_338 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_23 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_337,
                    args_48,
                    args_49,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_338,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_338, False)
            ttnn.deallocate(ttnn_to_memory_config_337, False)
            ttnn_to_memory_config_339 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_23,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_23, False)
            ttnn_experimental_nlp_concat_heads_decode_23 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_339,
                    sub_core_grids=ttnn_to_memory_config_339.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_339, False)
            ttnn_to_memory_config_340 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_23,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_23, False)
            ttnn_reshape_123 = ttnn.reshape(
                ttnn_to_memory_config_340,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_340, False)
            ttnn_matmul_117 = ttnn.matmul(
                ttnn_reshape_123,
                weights["model.layers.23.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_123, False)
            ttnn_add_46 = ttnn.add(
                ttnn_matmul_117,
                ttnn_add_45,
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
            ttnn.deallocate(ttnn_matmul_117, False)
            ttnn.deallocate(ttnn_add_45, False)

            return ttnn_add_46, cos, sin, attn_mask

        elif self.layer_idx == 24:
            # Map parameters to original variable names
            ttnn_to_memory_config_343 = hidden_states
            ttnn_add_47 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_49 = key_cache_idx
            args_50 = key_cache
            activation_50 = value_cache_idx
            args_51 = value_cache

            ttnn_rms_norm_48 = ttnn.rms_norm(
                ttnn_to_memory_config_343,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.24.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_343, False)
            ttnn_matmul_121 = ttnn.matmul(
                ttnn_rms_norm_48,
                weights["model.layers.24.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_48, False)
            ttnn_to_memory_config_344 = ttnn.to_memory_config(
                ttnn_matmul_121,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_121, False)
            ttnn_reshape_124 = ttnn.reshape(
                ttnn_to_memory_config_344,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_344, False)
            v_72, v_73, v_74 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_124,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_124, False)
            ttnn_to_memory_config_345 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_346 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_48 = ttnn.experimental.rotary_embedding(
                v_74,
                ttnn_to_memory_config_346,
                ttnn_to_memory_config_345,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_346, False)
            ttnn.deallocate(ttnn_to_memory_config_345, False)
            ttnn.deallocate(v_74, False)
            ttnn_slice_51 = ttnn.slice(
                ttnn_experimental_rotary_embedding_48,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_48, False)
            ttnn_reshape_125 = ttnn.reshape(
                ttnn_slice_51,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_51, False)
            ttnn_repeat_49 = ttnn.repeat(
                activation_49,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_49, False)
            ttnn_to_memory_config_347 = ttnn.to_memory_config(
                ttnn_reshape_125,
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
            ttnn.deallocate(ttnn_reshape_125, False)
            ttnn.experimental.paged_update_cache(
                args_50,
                ttnn_to_memory_config_347,
                update_idxs_tensor=ttnn_repeat_49,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_347, False)
            ttnn.deallocate(ttnn_repeat_49, False)
            ttnn_reshape_126 = ttnn.reshape(
                v_73,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_73, False)
            ttnn_repeat_50 = ttnn.repeat(
                activation_50,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_50, False)
            ttnn_to_memory_config_348 = ttnn.to_memory_config(
                ttnn_reshape_126,
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
            ttnn.deallocate(ttnn_reshape_126, False)
            ttnn.experimental.paged_update_cache(
                args_51,
                ttnn_to_memory_config_348,
                update_idxs_tensor=ttnn_repeat_50,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_348, False)
            ttnn.deallocate(ttnn_repeat_50, False)
            ttnn_to_memory_config_349 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_350 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_49 = ttnn.experimental.rotary_embedding(
                v_72,
                ttnn_to_memory_config_350,
                ttnn_to_memory_config_349,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_350, False)
            ttnn.deallocate(ttnn_to_memory_config_349, False)
            ttnn.deallocate(v_72, False)
            ttnn_slice_52 = ttnn.slice(
                ttnn_experimental_rotary_embedding_49,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_49, False)
            ttnn_reshape_127 = ttnn.reshape(
                ttnn_slice_52,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_52, False)
            ttnn_to_memory_config_351 = ttnn.to_memory_config(
                ttnn_reshape_127,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_127, False)
            ttnn_to_memory_config_352 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_24 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_351,
                    args_50,
                    args_51,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_352,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_352, False)
            ttnn.deallocate(ttnn_to_memory_config_351, False)
            ttnn_to_memory_config_353 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_24,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_24, False)
            ttnn_experimental_nlp_concat_heads_decode_24 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_353,
                    sub_core_grids=ttnn_to_memory_config_353.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_353, False)
            ttnn_to_memory_config_354 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_24,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_24, False)
            ttnn_reshape_128 = ttnn.reshape(
                ttnn_to_memory_config_354,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_354, False)
            ttnn_matmul_122 = ttnn.matmul(
                ttnn_reshape_128,
                weights["model.layers.24.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_128, False)
            ttnn_add_48 = ttnn.add(
                ttnn_matmul_122,
                ttnn_add_47,
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
            ttnn.deallocate(ttnn_matmul_122, False)
            ttnn.deallocate(ttnn_add_47, False)

            return ttnn_add_48, cos, sin, attn_mask

        elif self.layer_idx == 25:
            # Map parameters to original variable names
            ttnn_to_memory_config_357 = hidden_states
            ttnn_add_49 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_51 = key_cache_idx
            args_52 = key_cache
            activation_52 = value_cache_idx
            args_53 = value_cache

            ttnn_rms_norm_50 = ttnn.rms_norm(
                ttnn_to_memory_config_357,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.25.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_357, False)
            ttnn_matmul_126 = ttnn.matmul(
                ttnn_rms_norm_50,
                weights["model.layers.25.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_50, False)
            ttnn_to_memory_config_358 = ttnn.to_memory_config(
                ttnn_matmul_126,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_126, False)
            ttnn_reshape_129 = ttnn.reshape(
                ttnn_to_memory_config_358,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_358, False)
            v_75, v_76, v_77 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_129,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_129, False)
            ttnn_to_memory_config_359 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_360 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_50 = ttnn.experimental.rotary_embedding(
                v_77,
                ttnn_to_memory_config_360,
                ttnn_to_memory_config_359,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_360, False)
            ttnn.deallocate(ttnn_to_memory_config_359, False)
            ttnn.deallocate(v_77, False)
            ttnn_slice_53 = ttnn.slice(
                ttnn_experimental_rotary_embedding_50,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_50, False)
            ttnn_reshape_130 = ttnn.reshape(
                ttnn_slice_53,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_53, False)
            ttnn_repeat_51 = ttnn.repeat(
                activation_51,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_51, False)
            ttnn_to_memory_config_361 = ttnn.to_memory_config(
                ttnn_reshape_130,
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
            ttnn.deallocate(ttnn_reshape_130, False)
            ttnn.experimental.paged_update_cache(
                args_52,
                ttnn_to_memory_config_361,
                update_idxs_tensor=ttnn_repeat_51,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_361, False)
            ttnn.deallocate(ttnn_repeat_51, False)
            ttnn_reshape_131 = ttnn.reshape(
                v_76,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_76, False)
            ttnn_repeat_52 = ttnn.repeat(
                activation_52,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_52, False)
            ttnn_to_memory_config_362 = ttnn.to_memory_config(
                ttnn_reshape_131,
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
            ttnn.deallocate(ttnn_reshape_131, False)
            ttnn.experimental.paged_update_cache(
                args_53,
                ttnn_to_memory_config_362,
                update_idxs_tensor=ttnn_repeat_52,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_362, False)
            ttnn.deallocate(ttnn_repeat_52, False)
            ttnn_to_memory_config_363 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_364 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_51 = ttnn.experimental.rotary_embedding(
                v_75,
                ttnn_to_memory_config_364,
                ttnn_to_memory_config_363,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_364, False)
            ttnn.deallocate(ttnn_to_memory_config_363, False)
            ttnn.deallocate(v_75, False)
            ttnn_slice_54 = ttnn.slice(
                ttnn_experimental_rotary_embedding_51,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_51, False)
            ttnn_reshape_132 = ttnn.reshape(
                ttnn_slice_54,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_54, False)
            ttnn_to_memory_config_365 = ttnn.to_memory_config(
                ttnn_reshape_132,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_132, False)
            ttnn_to_memory_config_366 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_25 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_365,
                    args_52,
                    args_53,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_366,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_366, False)
            ttnn.deallocate(ttnn_to_memory_config_365, False)
            ttnn_to_memory_config_367 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_25,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_25, False)
            ttnn_experimental_nlp_concat_heads_decode_25 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_367,
                    sub_core_grids=ttnn_to_memory_config_367.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_367, False)
            ttnn_to_memory_config_368 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_25,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_25, False)
            ttnn_reshape_133 = ttnn.reshape(
                ttnn_to_memory_config_368,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_368, False)
            ttnn_matmul_127 = ttnn.matmul(
                ttnn_reshape_133,
                weights["model.layers.25.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_133, False)
            ttnn_add_50 = ttnn.add(
                ttnn_matmul_127,
                ttnn_add_49,
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
            ttnn.deallocate(ttnn_matmul_127, False)
            ttnn.deallocate(ttnn_add_49, False)

            return ttnn_add_50, cos, sin, attn_mask

        elif self.layer_idx == 26:
            # Map parameters to original variable names
            ttnn_to_memory_config_371 = hidden_states
            ttnn_add_51 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_53 = key_cache_idx
            args_54 = key_cache
            activation_54 = value_cache_idx
            args_55 = value_cache

            ttnn_rms_norm_52 = ttnn.rms_norm(
                ttnn_to_memory_config_371,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.26.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_371, False)
            ttnn_matmul_131 = ttnn.matmul(
                ttnn_rms_norm_52,
                weights["model.layers.26.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_52, False)
            ttnn_to_memory_config_372 = ttnn.to_memory_config(
                ttnn_matmul_131,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_131, False)
            ttnn_reshape_134 = ttnn.reshape(
                ttnn_to_memory_config_372,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_372, False)
            v_78, v_79, v_80 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_134,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_134, False)
            ttnn_to_memory_config_373 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_374 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_52 = ttnn.experimental.rotary_embedding(
                v_80,
                ttnn_to_memory_config_374,
                ttnn_to_memory_config_373,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_374, False)
            ttnn.deallocate(ttnn_to_memory_config_373, False)
            ttnn.deallocate(v_80, False)
            ttnn_slice_55 = ttnn.slice(
                ttnn_experimental_rotary_embedding_52,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_52, False)
            ttnn_reshape_135 = ttnn.reshape(
                ttnn_slice_55,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_55, False)
            ttnn_repeat_53 = ttnn.repeat(
                activation_53,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_53, False)
            ttnn_to_memory_config_375 = ttnn.to_memory_config(
                ttnn_reshape_135,
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
            ttnn.deallocate(ttnn_reshape_135, False)
            ttnn.experimental.paged_update_cache(
                args_54,
                ttnn_to_memory_config_375,
                update_idxs_tensor=ttnn_repeat_53,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_375, False)
            ttnn.deallocate(ttnn_repeat_53, False)
            ttnn_reshape_136 = ttnn.reshape(
                v_79,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_79, False)
            ttnn_repeat_54 = ttnn.repeat(
                activation_54,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_54, False)
            ttnn_to_memory_config_376 = ttnn.to_memory_config(
                ttnn_reshape_136,
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
            ttnn.deallocate(ttnn_reshape_136, False)
            ttnn.experimental.paged_update_cache(
                args_55,
                ttnn_to_memory_config_376,
                update_idxs_tensor=ttnn_repeat_54,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_376, False)
            ttnn.deallocate(ttnn_repeat_54, False)
            ttnn_to_memory_config_377 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_378 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_53 = ttnn.experimental.rotary_embedding(
                v_78,
                ttnn_to_memory_config_378,
                ttnn_to_memory_config_377,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_378, False)
            ttnn.deallocate(ttnn_to_memory_config_377, False)
            ttnn.deallocate(v_78, False)
            ttnn_slice_56 = ttnn.slice(
                ttnn_experimental_rotary_embedding_53,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_53, False)
            ttnn_reshape_137 = ttnn.reshape(
                ttnn_slice_56,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_56, False)
            ttnn_to_memory_config_379 = ttnn.to_memory_config(
                ttnn_reshape_137,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_137, False)
            ttnn_to_memory_config_380 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_26 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_379,
                    args_54,
                    args_55,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_380,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_380, False)
            ttnn.deallocate(ttnn_to_memory_config_379, False)
            ttnn_to_memory_config_381 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_26,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_26, False)
            ttnn_experimental_nlp_concat_heads_decode_26 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_381,
                    sub_core_grids=ttnn_to_memory_config_381.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_381, False)
            ttnn_to_memory_config_382 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_26,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_26, False)
            ttnn_reshape_138 = ttnn.reshape(
                ttnn_to_memory_config_382,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_382, False)
            ttnn_matmul_132 = ttnn.matmul(
                ttnn_reshape_138,
                weights["model.layers.26.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_138, False)
            ttnn_add_52 = ttnn.add(
                ttnn_matmul_132,
                ttnn_add_51,
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
            ttnn.deallocate(ttnn_matmul_132, False)
            ttnn.deallocate(ttnn_add_51, False)

            return ttnn_add_52, cos, sin, attn_mask

        elif self.layer_idx == 27:
            # Map parameters to original variable names
            ttnn_to_memory_config_385 = hidden_states
            ttnn_add_53 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_55 = key_cache_idx
            args_56 = key_cache
            activation_56 = value_cache_idx
            args_57 = value_cache

            ttnn_rms_norm_54 = ttnn.rms_norm(
                ttnn_to_memory_config_385,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.27.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_385, False)
            ttnn_matmul_136 = ttnn.matmul(
                ttnn_rms_norm_54,
                weights["model.layers.27.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_54, False)
            ttnn_to_memory_config_386 = ttnn.to_memory_config(
                ttnn_matmul_136,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_136, False)
            ttnn_reshape_139 = ttnn.reshape(
                ttnn_to_memory_config_386,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_386, False)
            v_81, v_82, v_83 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_139,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_139, False)
            ttnn_to_memory_config_387 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_388 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_54 = ttnn.experimental.rotary_embedding(
                v_83,
                ttnn_to_memory_config_388,
                ttnn_to_memory_config_387,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_388, False)
            ttnn.deallocate(ttnn_to_memory_config_387, False)
            ttnn.deallocate(v_83, False)
            ttnn_slice_57 = ttnn.slice(
                ttnn_experimental_rotary_embedding_54,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_54, False)
            ttnn_reshape_140 = ttnn.reshape(
                ttnn_slice_57,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_57, False)
            ttnn_repeat_55 = ttnn.repeat(
                activation_55,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_55, False)
            ttnn_to_memory_config_389 = ttnn.to_memory_config(
                ttnn_reshape_140,
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
            ttnn.deallocate(ttnn_reshape_140, False)
            ttnn.experimental.paged_update_cache(
                args_56,
                ttnn_to_memory_config_389,
                update_idxs_tensor=ttnn_repeat_55,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_389, False)
            ttnn.deallocate(ttnn_repeat_55, False)
            ttnn_reshape_141 = ttnn.reshape(
                v_82,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_82, False)
            ttnn_repeat_56 = ttnn.repeat(
                activation_56,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_56, False)
            ttnn_to_memory_config_390 = ttnn.to_memory_config(
                ttnn_reshape_141,
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
            ttnn.deallocate(ttnn_reshape_141, False)
            ttnn.experimental.paged_update_cache(
                args_57,
                ttnn_to_memory_config_390,
                update_idxs_tensor=ttnn_repeat_56,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_390, False)
            ttnn.deallocate(ttnn_repeat_56, False)
            ttnn_to_memory_config_391 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_392 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_55 = ttnn.experimental.rotary_embedding(
                v_81,
                ttnn_to_memory_config_392,
                ttnn_to_memory_config_391,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_392, False)
            ttnn.deallocate(ttnn_to_memory_config_391, False)
            ttnn.deallocate(v_81, False)
            ttnn_slice_58 = ttnn.slice(
                ttnn_experimental_rotary_embedding_55,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_55, False)
            ttnn_reshape_142 = ttnn.reshape(
                ttnn_slice_58,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_58, False)
            ttnn_to_memory_config_393 = ttnn.to_memory_config(
                ttnn_reshape_142,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_142, False)
            ttnn_to_memory_config_394 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_27 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_393,
                    args_56,
                    args_57,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_394,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_394, False)
            ttnn.deallocate(ttnn_to_memory_config_393, False)
            ttnn_to_memory_config_395 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_27,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_27, False)
            ttnn_experimental_nlp_concat_heads_decode_27 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_395,
                    sub_core_grids=ttnn_to_memory_config_395.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_395, False)
            ttnn_to_memory_config_396 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_27,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_27, False)
            ttnn_reshape_143 = ttnn.reshape(
                ttnn_to_memory_config_396,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_396, False)
            ttnn_matmul_137 = ttnn.matmul(
                ttnn_reshape_143,
                weights["model.layers.27.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_143, False)
            ttnn_add_54 = ttnn.add(
                ttnn_matmul_137,
                ttnn_add_53,
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
            ttnn.deallocate(ttnn_matmul_137, False)
            ttnn.deallocate(ttnn_add_53, False)

            return ttnn_add_54, cos, sin, attn_mask

        elif self.layer_idx == 28:
            # Map parameters to original variable names
            ttnn_to_memory_config_399 = hidden_states
            ttnn_add_55 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_57 = key_cache_idx
            args_58 = key_cache
            activation_58 = value_cache_idx
            args_59 = value_cache

            ttnn_rms_norm_56 = ttnn.rms_norm(
                ttnn_to_memory_config_399,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.28.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_399, False)
            ttnn_matmul_141 = ttnn.matmul(
                ttnn_rms_norm_56,
                weights["model.layers.28.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_56, False)
            ttnn_to_memory_config_400 = ttnn.to_memory_config(
                ttnn_matmul_141,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_141, False)
            ttnn_reshape_144 = ttnn.reshape(
                ttnn_to_memory_config_400,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_400, False)
            v_84, v_85, v_86 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_144,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_144, False)
            ttnn_to_memory_config_401 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_402 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_56 = ttnn.experimental.rotary_embedding(
                v_86,
                ttnn_to_memory_config_402,
                ttnn_to_memory_config_401,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_402, False)
            ttnn.deallocate(ttnn_to_memory_config_401, False)
            ttnn.deallocate(v_86, False)
            ttnn_slice_59 = ttnn.slice(
                ttnn_experimental_rotary_embedding_56,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_56, False)
            ttnn_reshape_145 = ttnn.reshape(
                ttnn_slice_59,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_59, False)
            ttnn_repeat_57 = ttnn.repeat(
                activation_57,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_57, False)
            ttnn_to_memory_config_403 = ttnn.to_memory_config(
                ttnn_reshape_145,
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
            ttnn.deallocate(ttnn_reshape_145, False)
            ttnn.experimental.paged_update_cache(
                args_58,
                ttnn_to_memory_config_403,
                update_idxs_tensor=ttnn_repeat_57,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_403, False)
            ttnn.deallocate(ttnn_repeat_57, False)
            ttnn_reshape_146 = ttnn.reshape(
                v_85,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_85, False)
            ttnn_repeat_58 = ttnn.repeat(
                activation_58,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_58, False)
            ttnn_to_memory_config_404 = ttnn.to_memory_config(
                ttnn_reshape_146,
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
            ttnn.deallocate(ttnn_reshape_146, False)
            ttnn.experimental.paged_update_cache(
                args_59,
                ttnn_to_memory_config_404,
                update_idxs_tensor=ttnn_repeat_58,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_404, False)
            ttnn.deallocate(ttnn_repeat_58, False)
            ttnn_to_memory_config_405 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_406 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_57 = ttnn.experimental.rotary_embedding(
                v_84,
                ttnn_to_memory_config_406,
                ttnn_to_memory_config_405,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_406, False)
            ttnn.deallocate(ttnn_to_memory_config_405, False)
            ttnn.deallocate(v_84, False)
            ttnn_slice_60 = ttnn.slice(
                ttnn_experimental_rotary_embedding_57,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_57, False)
            ttnn_reshape_147 = ttnn.reshape(
                ttnn_slice_60,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_60, False)
            ttnn_to_memory_config_407 = ttnn.to_memory_config(
                ttnn_reshape_147,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_147, False)
            ttnn_to_memory_config_408 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_28 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_407,
                    args_58,
                    args_59,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_408,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_408, False)
            ttnn.deallocate(ttnn_to_memory_config_407, False)
            ttnn_to_memory_config_409 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_28,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_28, False)
            ttnn_experimental_nlp_concat_heads_decode_28 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_409,
                    sub_core_grids=ttnn_to_memory_config_409.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_409, False)
            ttnn_to_memory_config_410 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_28,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_28, False)
            ttnn_reshape_148 = ttnn.reshape(
                ttnn_to_memory_config_410,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_410, False)
            ttnn_matmul_142 = ttnn.matmul(
                ttnn_reshape_148,
                weights["model.layers.28.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_148, False)
            ttnn_add_56 = ttnn.add(
                ttnn_matmul_142,
                ttnn_add_55,
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
            ttnn.deallocate(ttnn_matmul_142, False)
            ttnn.deallocate(ttnn_add_55, False)

            return ttnn_add_56, cos, sin, attn_mask

        elif self.layer_idx == 29:
            # Map parameters to original variable names
            ttnn_to_memory_config_413 = hidden_states
            ttnn_add_57 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_59 = key_cache_idx
            args_60 = key_cache
            activation_60 = value_cache_idx
            args_61 = value_cache

            ttnn_rms_norm_58 = ttnn.rms_norm(
                ttnn_to_memory_config_413,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.29.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_413, False)
            ttnn_matmul_146 = ttnn.matmul(
                ttnn_rms_norm_58,
                weights["model.layers.29.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_58, False)
            ttnn_to_memory_config_414 = ttnn.to_memory_config(
                ttnn_matmul_146,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_146, False)
            ttnn_reshape_149 = ttnn.reshape(
                ttnn_to_memory_config_414,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_414, False)
            v_87, v_88, v_89 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_149,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_149, False)
            ttnn_to_memory_config_415 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_416 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_58 = ttnn.experimental.rotary_embedding(
                v_89,
                ttnn_to_memory_config_416,
                ttnn_to_memory_config_415,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_416, False)
            ttnn.deallocate(ttnn_to_memory_config_415, False)
            ttnn.deallocate(v_89, False)
            ttnn_slice_61 = ttnn.slice(
                ttnn_experimental_rotary_embedding_58,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_58, False)
            ttnn_reshape_150 = ttnn.reshape(
                ttnn_slice_61,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_61, False)
            ttnn_repeat_59 = ttnn.repeat(
                activation_59,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_59, False)
            ttnn_to_memory_config_417 = ttnn.to_memory_config(
                ttnn_reshape_150,
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
            ttnn.deallocate(ttnn_reshape_150, False)
            ttnn.experimental.paged_update_cache(
                args_60,
                ttnn_to_memory_config_417,
                update_idxs_tensor=ttnn_repeat_59,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_417, False)
            ttnn.deallocate(ttnn_repeat_59, False)
            ttnn_reshape_151 = ttnn.reshape(
                v_88,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_88, False)
            ttnn_repeat_60 = ttnn.repeat(
                activation_60,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_60, False)
            ttnn_to_memory_config_418 = ttnn.to_memory_config(
                ttnn_reshape_151,
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
            ttnn.deallocate(ttnn_reshape_151, False)
            ttnn.experimental.paged_update_cache(
                args_61,
                ttnn_to_memory_config_418,
                update_idxs_tensor=ttnn_repeat_60,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_418, False)
            ttnn.deallocate(ttnn_repeat_60, False)
            ttnn_to_memory_config_419 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_420 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_59 = ttnn.experimental.rotary_embedding(
                v_87,
                ttnn_to_memory_config_420,
                ttnn_to_memory_config_419,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_420, False)
            ttnn.deallocate(ttnn_to_memory_config_419, False)
            ttnn.deallocate(v_87, False)
            ttnn_slice_62 = ttnn.slice(
                ttnn_experimental_rotary_embedding_59,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_59, False)
            ttnn_reshape_152 = ttnn.reshape(
                ttnn_slice_62,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_62, False)
            ttnn_to_memory_config_421 = ttnn.to_memory_config(
                ttnn_reshape_152,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_152, False)
            ttnn_to_memory_config_422 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_29 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_421,
                    args_60,
                    args_61,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_422,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_422, False)
            ttnn.deallocate(ttnn_to_memory_config_421, False)
            ttnn_to_memory_config_423 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_29,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_29, False)
            ttnn_experimental_nlp_concat_heads_decode_29 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_423,
                    sub_core_grids=ttnn_to_memory_config_423.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_423, False)
            ttnn_to_memory_config_424 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_29,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_29, False)
            ttnn_reshape_153 = ttnn.reshape(
                ttnn_to_memory_config_424,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_424, False)
            ttnn_matmul_147 = ttnn.matmul(
                ttnn_reshape_153,
                weights["model.layers.29.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_153, False)
            ttnn_add_58 = ttnn.add(
                ttnn_matmul_147,
                ttnn_add_57,
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
            ttnn.deallocate(ttnn_matmul_147, False)
            ttnn.deallocate(ttnn_add_57, False)

            return ttnn_add_58, cos, sin, attn_mask

        elif self.layer_idx == 30:
            # Map parameters to original variable names
            ttnn_to_memory_config_427 = hidden_states
            ttnn_add_59 = residual
            ttnn_typecast_550 = cos
            ttnn_typecast_551 = sin
            ttnn_repeat_2 = attn_mask
            activation_61 = key_cache_idx
            args_62 = key_cache
            activation_62 = value_cache_idx
            args_63 = value_cache

            ttnn_rms_norm_60 = ttnn.rms_norm(
                ttnn_to_memory_config_427,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.30.input_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_427, False)
            ttnn_matmul_151 = ttnn.matmul(
                ttnn_rms_norm_60,
                weights["model.layers.30.self_attn.qkv_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_60, False)
            ttnn_to_memory_config_428 = ttnn.to_memory_config(
                ttnn_matmul_151,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_151, False)
            ttnn_reshape_154 = ttnn.reshape(
                ttnn_to_memory_config_428,
                [32, 1, 6144],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_428, False)
            v_90, v_91, v_92 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_154,
                None,
                num_heads=32,
                num_kv_heads=8,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_154, False)
            ttnn_to_memory_config_429 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_430 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_60 = ttnn.experimental.rotary_embedding(
                v_92,
                ttnn_to_memory_config_430,
                ttnn_to_memory_config_429,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_430, False)
            ttnn.deallocate(ttnn_to_memory_config_429, False)
            ttnn.deallocate(v_92, False)
            ttnn_slice_63 = ttnn.slice(
                ttnn_experimental_rotary_embedding_60,
                [0, 0, 0, 0],
                [32, 8, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_60, False)
            ttnn_reshape_155 = ttnn.reshape(
                ttnn_slice_63,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_63, False)
            ttnn_repeat_61 = ttnn.repeat(
                activation_61,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_61, False)
            ttnn_to_memory_config_431 = ttnn.to_memory_config(
                ttnn_reshape_155,
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
            ttnn.deallocate(ttnn_reshape_155, False)
            ttnn.experimental.paged_update_cache(
                args_62,
                ttnn_to_memory_config_431,
                update_idxs_tensor=ttnn_repeat_61,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_431, False)
            ttnn.deallocate(ttnn_repeat_61, False)
            ttnn_reshape_156 = ttnn.reshape(
                v_91,
                [1, 32, 8, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(v_91, False)
            ttnn_repeat_62 = ttnn.repeat(
                activation_62,
                ttnn.Shape([32]),
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(activation_62, False)
            ttnn_to_memory_config_432 = ttnn.to_memory_config(
                ttnn_reshape_156,
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
            ttnn.deallocate(ttnn_reshape_156, False)
            ttnn.experimental.paged_update_cache(
                args_63,
                ttnn_to_memory_config_432,
                update_idxs_tensor=ttnn_repeat_62,
                share_cache=False,
                page_table=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_432, False)
            ttnn.deallocate(ttnn_repeat_62, False)
            ttnn_to_memory_config_433 = ttnn.to_memory_config(
                ttnn_typecast_551,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_to_memory_config_434 = ttnn.to_memory_config(
                ttnn_typecast_550,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn_experimental_rotary_embedding_61 = ttnn.experimental.rotary_embedding(
                v_90,
                ttnn_to_memory_config_434,
                ttnn_to_memory_config_433,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_to_memory_config_434, False)
            ttnn.deallocate(ttnn_to_memory_config_433, False)
            ttnn.deallocate(v_90, False)
            ttnn_slice_64 = ttnn.slice(
                ttnn_experimental_rotary_embedding_61,
                [0, 0, 0, 0],
                [32, 32, 1, 128],
                [1, 1, 1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_61, False)
            ttnn_reshape_157 = ttnn.reshape(
                ttnn_slice_64,
                [1, 32, 32, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_slice_64, False)
            ttnn_to_memory_config_435 = ttnn.to_memory_config(
                ttnn_reshape_157,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_157, False)
            ttnn_to_memory_config_436 = ttnn.to_memory_config(
                ttnn_repeat_2,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_transformer_scaled_dot_product_attention_decode_30 = (
                ttnn.transformer.scaled_dot_product_attention_decode(
                    ttnn_to_memory_config_435,
                    args_62,
                    args_63,
                    is_causal=False,
                    attn_mask=ttnn_to_memory_config_436,
                    cur_pos_tensor=None,
                    attention_sink=None,
                    scale=0.088388338685035706,
                    sliding_window_size=None,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                    ),
                )
            )
            ttnn.deallocate(ttnn_to_memory_config_436, False)
            ttnn.deallocate(ttnn_to_memory_config_435, False)
            ttnn_to_memory_config_437 = ttnn.to_memory_config(
                ttnn_transformer_scaled_dot_product_attention_decode_30,
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
            ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_30, False)
            ttnn_experimental_nlp_concat_heads_decode_30 = (
                ttnn.experimental.nlp_concat_heads_decode(
                    ttnn_to_memory_config_437,
                    sub_core_grids=ttnn_to_memory_config_437.memory_config().shard_spec.grid,
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
            ttnn.deallocate(ttnn_to_memory_config_437, False)
            ttnn_to_memory_config_438 = ttnn.to_memory_config(
                ttnn_experimental_nlp_concat_heads_decode_30,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_nlp_concat_heads_decode_30, False)
            ttnn_reshape_158 = ttnn.reshape(
                ttnn_to_memory_config_438,
                [32, 4096],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
                ),
            )
            ttnn.deallocate(ttnn_to_memory_config_438, False)
            ttnn_matmul_152 = ttnn.matmul(
                ttnn_reshape_158,
                weights["model.layers.30.self_attn.o_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_reshape_158, False)
            ttnn_add_60 = ttnn.add(
                ttnn_matmul_152,
                ttnn_add_59,
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
            ttnn.deallocate(ttnn_matmul_152, False)
            ttnn.deallocate(ttnn_add_59, False)

            return ttnn_add_60, cos, sin, attn_mask

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
            ttnn.deallocate(activation_63, False)
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
            ttnn.deallocate(activation_64, False)
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


class LlamaMLP(LightweightModule):
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx

    def forward(self, hidden_states, residual, weights, device):
        if self.layer_idx == 0:
            # Map parameters to original variable names
            ttnn_to_memory_config_19 = hidden_states
            ttnn_add_0 = residual

            ttnn_rms_norm_1 = ttnn.rms_norm(
                ttnn_to_memory_config_19,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.0.post_attention_layernorm.parametrizations.weight.original"
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
                weights["model.layers.0.mlp.gate_proj.parametrizations.weight.original"],
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
                weights["model.layers.0.mlp.up_proj.parametrizations.weight.original"],
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
                weights["model.layers.0.mlp.down_proj.parametrizations.weight.original"],
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

        elif self.layer_idx == 1:
            # Map parameters to original variable names
            ttnn_to_memory_config_33 = hidden_states
            ttnn_add_2 = residual

            ttnn_rms_norm_3 = ttnn.rms_norm(
                ttnn_to_memory_config_33,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.1.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_33, False)
            ttnn_matmul_8 = ttnn.matmul(
                ttnn_rms_norm_3,
                weights["model.layers.1.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_9 = ttnn.matmul(
                ttnn_rms_norm_3,
                weights["model.layers.1.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_3, False)
            ttnn_multiply_1 = ttnn.multiply(
                ttnn_matmul_8,
                ttnn_matmul_9,
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
            ttnn.deallocate(ttnn_matmul_9, False)
            ttnn.deallocate(ttnn_matmul_8, False)
            ttnn_to_memory_config_34 = ttnn.to_memory_config(
                ttnn_multiply_1,
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
            ttnn.deallocate(ttnn_multiply_1, False)
            ttnn_matmul_10 = ttnn.matmul(
                ttnn_to_memory_config_34,
                weights["model.layers.1.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_34, False)
            ttnn_add_3 = ttnn.add(
                ttnn_matmul_10,
                ttnn_add_2,
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
            ttnn.deallocate(ttnn_matmul_10, False)
            ttnn.deallocate(ttnn_add_2, False)

            return ttnn_add_3

        elif self.layer_idx == 2:
            # Map parameters to original variable names
            ttnn_to_memory_config_47 = hidden_states
            ttnn_add_4 = residual

            ttnn_rms_norm_5 = ttnn.rms_norm(
                ttnn_to_memory_config_47,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.2.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_47, False)
            ttnn_matmul_13 = ttnn.matmul(
                ttnn_rms_norm_5,
                weights["model.layers.2.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_14 = ttnn.matmul(
                ttnn_rms_norm_5,
                weights["model.layers.2.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_5, False)
            ttnn_multiply_2 = ttnn.multiply(
                ttnn_matmul_13,
                ttnn_matmul_14,
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
            ttnn.deallocate(ttnn_matmul_14, False)
            ttnn.deallocate(ttnn_matmul_13, False)
            ttnn_to_memory_config_48 = ttnn.to_memory_config(
                ttnn_multiply_2,
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
            ttnn.deallocate(ttnn_multiply_2, False)
            ttnn_matmul_15 = ttnn.matmul(
                ttnn_to_memory_config_48,
                weights["model.layers.2.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_48, False)
            ttnn_add_5 = ttnn.add(
                ttnn_matmul_15,
                ttnn_add_4,
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
            ttnn.deallocate(ttnn_matmul_15, False)
            ttnn.deallocate(ttnn_add_4, False)

            return ttnn_add_5

        elif self.layer_idx == 3:
            # Map parameters to original variable names
            ttnn_to_memory_config_61 = hidden_states
            ttnn_add_6 = residual

            ttnn_rms_norm_7 = ttnn.rms_norm(
                ttnn_to_memory_config_61,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.3.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_61, False)
            ttnn_matmul_18 = ttnn.matmul(
                ttnn_rms_norm_7,
                weights["model.layers.3.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_19 = ttnn.matmul(
                ttnn_rms_norm_7,
                weights["model.layers.3.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_7, False)
            ttnn_multiply_3 = ttnn.multiply(
                ttnn_matmul_18,
                ttnn_matmul_19,
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
            ttnn.deallocate(ttnn_matmul_19, False)
            ttnn.deallocate(ttnn_matmul_18, False)
            ttnn_to_memory_config_62 = ttnn.to_memory_config(
                ttnn_multiply_3,
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
            ttnn.deallocate(ttnn_multiply_3, False)
            ttnn_matmul_20 = ttnn.matmul(
                ttnn_to_memory_config_62,
                weights["model.layers.3.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_62, False)
            ttnn_add_7 = ttnn.add(
                ttnn_matmul_20,
                ttnn_add_6,
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
            ttnn.deallocate(ttnn_matmul_20, False)
            ttnn.deallocate(ttnn_add_6, False)

            return ttnn_add_7

        elif self.layer_idx == 4:
            # Map parameters to original variable names
            ttnn_to_memory_config_75 = hidden_states
            ttnn_add_8 = residual

            ttnn_rms_norm_9 = ttnn.rms_norm(
                ttnn_to_memory_config_75,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.4.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_75, False)
            ttnn_matmul_23 = ttnn.matmul(
                ttnn_rms_norm_9,
                weights["model.layers.4.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_24 = ttnn.matmul(
                ttnn_rms_norm_9,
                weights["model.layers.4.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_9, False)
            ttnn_multiply_4 = ttnn.multiply(
                ttnn_matmul_23,
                ttnn_matmul_24,
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
            ttnn.deallocate(ttnn_matmul_24, False)
            ttnn.deallocate(ttnn_matmul_23, False)
            ttnn_to_memory_config_76 = ttnn.to_memory_config(
                ttnn_multiply_4,
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
            ttnn.deallocate(ttnn_multiply_4, False)
            ttnn_matmul_25 = ttnn.matmul(
                ttnn_to_memory_config_76,
                weights["model.layers.4.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_76, False)
            ttnn_add_9 = ttnn.add(
                ttnn_matmul_25,
                ttnn_add_8,
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
            ttnn.deallocate(ttnn_matmul_25, False)
            ttnn.deallocate(ttnn_add_8, False)

            return ttnn_add_9

        elif self.layer_idx == 5:
            # Map parameters to original variable names
            ttnn_to_memory_config_89 = hidden_states
            ttnn_add_10 = residual

            ttnn_rms_norm_11 = ttnn.rms_norm(
                ttnn_to_memory_config_89,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.5.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_89, False)
            ttnn_matmul_28 = ttnn.matmul(
                ttnn_rms_norm_11,
                weights["model.layers.5.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_29 = ttnn.matmul(
                ttnn_rms_norm_11,
                weights["model.layers.5.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_11, False)
            ttnn_multiply_5 = ttnn.multiply(
                ttnn_matmul_28,
                ttnn_matmul_29,
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
            ttnn.deallocate(ttnn_matmul_29, False)
            ttnn.deallocate(ttnn_matmul_28, False)
            ttnn_to_memory_config_90 = ttnn.to_memory_config(
                ttnn_multiply_5,
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
            ttnn.deallocate(ttnn_multiply_5, False)
            ttnn_matmul_30 = ttnn.matmul(
                ttnn_to_memory_config_90,
                weights["model.layers.5.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_90, False)
            ttnn_add_11 = ttnn.add(
                ttnn_matmul_30,
                ttnn_add_10,
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
            ttnn.deallocate(ttnn_matmul_30, False)
            ttnn.deallocate(ttnn_add_10, False)

            return ttnn_add_11

        elif self.layer_idx == 6:
            # Map parameters to original variable names
            ttnn_to_memory_config_103 = hidden_states
            ttnn_add_12 = residual

            ttnn_rms_norm_13 = ttnn.rms_norm(
                ttnn_to_memory_config_103,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.6.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_103, False)
            ttnn_matmul_33 = ttnn.matmul(
                ttnn_rms_norm_13,
                weights["model.layers.6.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_34 = ttnn.matmul(
                ttnn_rms_norm_13,
                weights["model.layers.6.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_13, False)
            ttnn_multiply_6 = ttnn.multiply(
                ttnn_matmul_33,
                ttnn_matmul_34,
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
            ttnn.deallocate(ttnn_matmul_34, False)
            ttnn.deallocate(ttnn_matmul_33, False)
            ttnn_to_memory_config_104 = ttnn.to_memory_config(
                ttnn_multiply_6,
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
            ttnn.deallocate(ttnn_multiply_6, False)
            ttnn_matmul_35 = ttnn.matmul(
                ttnn_to_memory_config_104,
                weights["model.layers.6.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_104, False)
            ttnn_add_13 = ttnn.add(
                ttnn_matmul_35,
                ttnn_add_12,
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
            ttnn.deallocate(ttnn_matmul_35, False)
            ttnn.deallocate(ttnn_add_12, False)

            return ttnn_add_13

        elif self.layer_idx == 7:
            # Map parameters to original variable names
            ttnn_to_memory_config_117 = hidden_states
            ttnn_add_14 = residual

            ttnn_rms_norm_15 = ttnn.rms_norm(
                ttnn_to_memory_config_117,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.7.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_117, False)
            ttnn_matmul_38 = ttnn.matmul(
                ttnn_rms_norm_15,
                weights["model.layers.7.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_39 = ttnn.matmul(
                ttnn_rms_norm_15,
                weights["model.layers.7.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_15, False)
            ttnn_multiply_7 = ttnn.multiply(
                ttnn_matmul_38,
                ttnn_matmul_39,
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
            ttnn.deallocate(ttnn_matmul_39, False)
            ttnn.deallocate(ttnn_matmul_38, False)
            ttnn_to_memory_config_118 = ttnn.to_memory_config(
                ttnn_multiply_7,
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
            ttnn.deallocate(ttnn_multiply_7, False)
            ttnn_matmul_40 = ttnn.matmul(
                ttnn_to_memory_config_118,
                weights["model.layers.7.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_118, False)
            ttnn_add_15 = ttnn.add(
                ttnn_matmul_40,
                ttnn_add_14,
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
            ttnn.deallocate(ttnn_matmul_40, False)
            ttnn.deallocate(ttnn_add_14, False)

            return ttnn_add_15

        elif self.layer_idx == 8:
            # Map parameters to original variable names
            ttnn_to_memory_config_131 = hidden_states
            ttnn_add_16 = residual

            ttnn_rms_norm_17 = ttnn.rms_norm(
                ttnn_to_memory_config_131,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.8.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_131, False)
            ttnn_matmul_43 = ttnn.matmul(
                ttnn_rms_norm_17,
                weights["model.layers.8.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_44 = ttnn.matmul(
                ttnn_rms_norm_17,
                weights["model.layers.8.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_17, False)
            ttnn_multiply_8 = ttnn.multiply(
                ttnn_matmul_43,
                ttnn_matmul_44,
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
            ttnn.deallocate(ttnn_matmul_44, False)
            ttnn.deallocate(ttnn_matmul_43, False)
            ttnn_to_memory_config_132 = ttnn.to_memory_config(
                ttnn_multiply_8,
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
            ttnn.deallocate(ttnn_multiply_8, False)
            ttnn_matmul_45 = ttnn.matmul(
                ttnn_to_memory_config_132,
                weights["model.layers.8.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_132, False)
            ttnn_add_17 = ttnn.add(
                ttnn_matmul_45,
                ttnn_add_16,
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
            ttnn.deallocate(ttnn_matmul_45, False)
            ttnn.deallocate(ttnn_add_16, False)

            return ttnn_add_17

        elif self.layer_idx == 9:
            # Map parameters to original variable names
            ttnn_to_memory_config_145 = hidden_states
            ttnn_add_18 = residual

            ttnn_rms_norm_19 = ttnn.rms_norm(
                ttnn_to_memory_config_145,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.9.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_145, False)
            ttnn_matmul_48 = ttnn.matmul(
                ttnn_rms_norm_19,
                weights["model.layers.9.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_49 = ttnn.matmul(
                ttnn_rms_norm_19,
                weights["model.layers.9.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_19, False)
            ttnn_multiply_9 = ttnn.multiply(
                ttnn_matmul_48,
                ttnn_matmul_49,
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
            ttnn.deallocate(ttnn_matmul_49, False)
            ttnn.deallocate(ttnn_matmul_48, False)
            ttnn_to_memory_config_146 = ttnn.to_memory_config(
                ttnn_multiply_9,
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
            ttnn.deallocate(ttnn_multiply_9, False)
            ttnn_matmul_50 = ttnn.matmul(
                ttnn_to_memory_config_146,
                weights["model.layers.9.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_146, False)
            ttnn_add_19 = ttnn.add(
                ttnn_matmul_50,
                ttnn_add_18,
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
            ttnn.deallocate(ttnn_matmul_50, False)
            ttnn.deallocate(ttnn_add_18, False)

            return ttnn_add_19

        elif self.layer_idx == 10:
            # Map parameters to original variable names
            ttnn_to_memory_config_159 = hidden_states
            ttnn_add_20 = residual

            ttnn_rms_norm_21 = ttnn.rms_norm(
                ttnn_to_memory_config_159,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.10.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_159, False)
            ttnn_matmul_53 = ttnn.matmul(
                ttnn_rms_norm_21,
                weights["model.layers.10.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_54 = ttnn.matmul(
                ttnn_rms_norm_21,
                weights["model.layers.10.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_21, False)
            ttnn_multiply_10 = ttnn.multiply(
                ttnn_matmul_53,
                ttnn_matmul_54,
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
            ttnn.deallocate(ttnn_matmul_54, False)
            ttnn.deallocate(ttnn_matmul_53, False)
            ttnn_to_memory_config_160 = ttnn.to_memory_config(
                ttnn_multiply_10,
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
            ttnn.deallocate(ttnn_multiply_10, False)
            ttnn_matmul_55 = ttnn.matmul(
                ttnn_to_memory_config_160,
                weights["model.layers.10.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_160, False)
            ttnn_add_21 = ttnn.add(
                ttnn_matmul_55,
                ttnn_add_20,
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
            ttnn.deallocate(ttnn_matmul_55, False)
            ttnn.deallocate(ttnn_add_20, False)

            return ttnn_add_21

        elif self.layer_idx == 11:
            # Map parameters to original variable names
            ttnn_to_memory_config_173 = hidden_states
            ttnn_add_22 = residual

            ttnn_rms_norm_23 = ttnn.rms_norm(
                ttnn_to_memory_config_173,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.11.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_173, False)
            ttnn_matmul_58 = ttnn.matmul(
                ttnn_rms_norm_23,
                weights["model.layers.11.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_59 = ttnn.matmul(
                ttnn_rms_norm_23,
                weights["model.layers.11.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_23, False)
            ttnn_multiply_11 = ttnn.multiply(
                ttnn_matmul_58,
                ttnn_matmul_59,
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
            ttnn.deallocate(ttnn_matmul_59, False)
            ttnn.deallocate(ttnn_matmul_58, False)
            ttnn_to_memory_config_174 = ttnn.to_memory_config(
                ttnn_multiply_11,
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
            ttnn.deallocate(ttnn_multiply_11, False)
            ttnn_matmul_60 = ttnn.matmul(
                ttnn_to_memory_config_174,
                weights["model.layers.11.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_174, False)
            ttnn_add_23 = ttnn.add(
                ttnn_matmul_60,
                ttnn_add_22,
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
            ttnn.deallocate(ttnn_matmul_60, False)
            ttnn.deallocate(ttnn_add_22, False)

            return ttnn_add_23

        elif self.layer_idx == 12:
            # Map parameters to original variable names
            ttnn_to_memory_config_187 = hidden_states
            ttnn_add_24 = residual

            ttnn_rms_norm_25 = ttnn.rms_norm(
                ttnn_to_memory_config_187,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.12.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_187, False)
            ttnn_matmul_63 = ttnn.matmul(
                ttnn_rms_norm_25,
                weights["model.layers.12.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_64 = ttnn.matmul(
                ttnn_rms_norm_25,
                weights["model.layers.12.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_25, False)
            ttnn_multiply_12 = ttnn.multiply(
                ttnn_matmul_63,
                ttnn_matmul_64,
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
            ttnn.deallocate(ttnn_matmul_64, False)
            ttnn.deallocate(ttnn_matmul_63, False)
            ttnn_to_memory_config_188 = ttnn.to_memory_config(
                ttnn_multiply_12,
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
            ttnn.deallocate(ttnn_multiply_12, False)
            ttnn_matmul_65 = ttnn.matmul(
                ttnn_to_memory_config_188,
                weights["model.layers.12.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_188, False)
            ttnn_add_25 = ttnn.add(
                ttnn_matmul_65,
                ttnn_add_24,
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
            ttnn.deallocate(ttnn_matmul_65, False)
            ttnn.deallocate(ttnn_add_24, False)

            return ttnn_add_25

        elif self.layer_idx == 13:
            # Map parameters to original variable names
            ttnn_to_memory_config_201 = hidden_states
            ttnn_add_26 = residual

            ttnn_rms_norm_27 = ttnn.rms_norm(
                ttnn_to_memory_config_201,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.13.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_201, False)
            ttnn_matmul_68 = ttnn.matmul(
                ttnn_rms_norm_27,
                weights["model.layers.13.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_69 = ttnn.matmul(
                ttnn_rms_norm_27,
                weights["model.layers.13.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_27, False)
            ttnn_multiply_13 = ttnn.multiply(
                ttnn_matmul_68,
                ttnn_matmul_69,
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
            ttnn.deallocate(ttnn_matmul_69, False)
            ttnn.deallocate(ttnn_matmul_68, False)
            ttnn_to_memory_config_202 = ttnn.to_memory_config(
                ttnn_multiply_13,
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
            ttnn.deallocate(ttnn_multiply_13, False)
            ttnn_matmul_70 = ttnn.matmul(
                ttnn_to_memory_config_202,
                weights["model.layers.13.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_202, False)
            ttnn_add_27 = ttnn.add(
                ttnn_matmul_70,
                ttnn_add_26,
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
            ttnn.deallocate(ttnn_matmul_70, False)
            ttnn.deallocate(ttnn_add_26, False)

            return ttnn_add_27

        elif self.layer_idx == 14:
            # Map parameters to original variable names
            ttnn_to_memory_config_215 = hidden_states
            ttnn_add_28 = residual

            ttnn_rms_norm_29 = ttnn.rms_norm(
                ttnn_to_memory_config_215,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.14.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_215, False)
            ttnn_matmul_73 = ttnn.matmul(
                ttnn_rms_norm_29,
                weights["model.layers.14.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_74 = ttnn.matmul(
                ttnn_rms_norm_29,
                weights["model.layers.14.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_29, False)
            ttnn_multiply_14 = ttnn.multiply(
                ttnn_matmul_73,
                ttnn_matmul_74,
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
            ttnn.deallocate(ttnn_matmul_74, False)
            ttnn.deallocate(ttnn_matmul_73, False)
            ttnn_to_memory_config_216 = ttnn.to_memory_config(
                ttnn_multiply_14,
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
            ttnn.deallocate(ttnn_multiply_14, False)
            ttnn_matmul_75 = ttnn.matmul(
                ttnn_to_memory_config_216,
                weights["model.layers.14.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_216, False)
            ttnn_add_29 = ttnn.add(
                ttnn_matmul_75,
                ttnn_add_28,
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
            ttnn.deallocate(ttnn_matmul_75, False)
            ttnn.deallocate(ttnn_add_28, False)

            return ttnn_add_29

        elif self.layer_idx == 15:
            # Map parameters to original variable names
            ttnn_to_memory_config_229 = hidden_states
            ttnn_add_30 = residual

            ttnn_rms_norm_31 = ttnn.rms_norm(
                ttnn_to_memory_config_229,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.15.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_229, False)
            ttnn_matmul_78 = ttnn.matmul(
                ttnn_rms_norm_31,
                weights["model.layers.15.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_79 = ttnn.matmul(
                ttnn_rms_norm_31,
                weights["model.layers.15.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_31, False)
            ttnn_multiply_15 = ttnn.multiply(
                ttnn_matmul_78,
                ttnn_matmul_79,
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
            ttnn.deallocate(ttnn_matmul_79, False)
            ttnn.deallocate(ttnn_matmul_78, False)
            ttnn_to_memory_config_230 = ttnn.to_memory_config(
                ttnn_multiply_15,
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
            ttnn.deallocate(ttnn_multiply_15, False)
            ttnn_matmul_80 = ttnn.matmul(
                ttnn_to_memory_config_230,
                weights["model.layers.15.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_230, False)
            ttnn_add_31 = ttnn.add(
                ttnn_matmul_80,
                ttnn_add_30,
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
            ttnn.deallocate(ttnn_matmul_80, False)
            ttnn.deallocate(ttnn_add_30, False)

            return ttnn_add_31

        elif self.layer_idx == 16:
            # Map parameters to original variable names
            ttnn_to_memory_config_243 = hidden_states
            ttnn_add_32 = residual

            ttnn_rms_norm_33 = ttnn.rms_norm(
                ttnn_to_memory_config_243,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.16.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_243, False)
            ttnn_matmul_83 = ttnn.matmul(
                ttnn_rms_norm_33,
                weights["model.layers.16.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_84 = ttnn.matmul(
                ttnn_rms_norm_33,
                weights["model.layers.16.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_33, False)
            ttnn_multiply_16 = ttnn.multiply(
                ttnn_matmul_83,
                ttnn_matmul_84,
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
            ttnn.deallocate(ttnn_matmul_84, False)
            ttnn.deallocate(ttnn_matmul_83, False)
            ttnn_to_memory_config_244 = ttnn.to_memory_config(
                ttnn_multiply_16,
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
            ttnn.deallocate(ttnn_multiply_16, False)
            ttnn_matmul_85 = ttnn.matmul(
                ttnn_to_memory_config_244,
                weights["model.layers.16.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_244, False)
            ttnn_add_33 = ttnn.add(
                ttnn_matmul_85,
                ttnn_add_32,
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
            ttnn.deallocate(ttnn_matmul_85, False)
            ttnn.deallocate(ttnn_add_32, False)

            return ttnn_add_33

        elif self.layer_idx == 17:
            # Map parameters to original variable names
            ttnn_to_memory_config_257 = hidden_states
            ttnn_add_34 = residual

            ttnn_rms_norm_35 = ttnn.rms_norm(
                ttnn_to_memory_config_257,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.17.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_257, False)
            ttnn_matmul_88 = ttnn.matmul(
                ttnn_rms_norm_35,
                weights["model.layers.17.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_89 = ttnn.matmul(
                ttnn_rms_norm_35,
                weights["model.layers.17.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_35, False)
            ttnn_multiply_17 = ttnn.multiply(
                ttnn_matmul_88,
                ttnn_matmul_89,
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
            ttnn.deallocate(ttnn_matmul_89, False)
            ttnn.deallocate(ttnn_matmul_88, False)
            ttnn_to_memory_config_258 = ttnn.to_memory_config(
                ttnn_multiply_17,
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
            ttnn.deallocate(ttnn_multiply_17, False)
            ttnn_matmul_90 = ttnn.matmul(
                ttnn_to_memory_config_258,
                weights["model.layers.17.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_258, False)
            ttnn_add_35 = ttnn.add(
                ttnn_matmul_90,
                ttnn_add_34,
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
            ttnn.deallocate(ttnn_matmul_90, False)
            ttnn.deallocate(ttnn_add_34, False)

            return ttnn_add_35

        elif self.layer_idx == 18:
            # Map parameters to original variable names
            ttnn_to_memory_config_271 = hidden_states
            ttnn_add_36 = residual

            ttnn_rms_norm_37 = ttnn.rms_norm(
                ttnn_to_memory_config_271,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.18.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_271, False)
            ttnn_matmul_93 = ttnn.matmul(
                ttnn_rms_norm_37,
                weights["model.layers.18.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_94 = ttnn.matmul(
                ttnn_rms_norm_37,
                weights["model.layers.18.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_37, False)
            ttnn_multiply_18 = ttnn.multiply(
                ttnn_matmul_93,
                ttnn_matmul_94,
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
            ttnn.deallocate(ttnn_matmul_94, False)
            ttnn.deallocate(ttnn_matmul_93, False)
            ttnn_to_memory_config_272 = ttnn.to_memory_config(
                ttnn_multiply_18,
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
            ttnn.deallocate(ttnn_multiply_18, False)
            ttnn_matmul_95 = ttnn.matmul(
                ttnn_to_memory_config_272,
                weights["model.layers.18.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_272, False)
            ttnn_add_37 = ttnn.add(
                ttnn_matmul_95,
                ttnn_add_36,
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
            ttnn.deallocate(ttnn_matmul_95, False)
            ttnn.deallocate(ttnn_add_36, False)

            return ttnn_add_37

        elif self.layer_idx == 19:
            # Map parameters to original variable names
            ttnn_to_memory_config_285 = hidden_states
            ttnn_add_38 = residual

            ttnn_rms_norm_39 = ttnn.rms_norm(
                ttnn_to_memory_config_285,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.19.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_285, False)
            ttnn_matmul_98 = ttnn.matmul(
                ttnn_rms_norm_39,
                weights["model.layers.19.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_99 = ttnn.matmul(
                ttnn_rms_norm_39,
                weights["model.layers.19.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_39, False)
            ttnn_multiply_19 = ttnn.multiply(
                ttnn_matmul_98,
                ttnn_matmul_99,
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
            ttnn.deallocate(ttnn_matmul_99, False)
            ttnn.deallocate(ttnn_matmul_98, False)
            ttnn_to_memory_config_286 = ttnn.to_memory_config(
                ttnn_multiply_19,
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
            ttnn.deallocate(ttnn_multiply_19, False)
            ttnn_matmul_100 = ttnn.matmul(
                ttnn_to_memory_config_286,
                weights["model.layers.19.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_286, False)
            ttnn_add_39 = ttnn.add(
                ttnn_matmul_100,
                ttnn_add_38,
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
            ttnn.deallocate(ttnn_matmul_100, False)
            ttnn.deallocate(ttnn_add_38, False)

            return ttnn_add_39

        elif self.layer_idx == 20:
            # Map parameters to original variable names
            ttnn_to_memory_config_299 = hidden_states
            ttnn_add_40 = residual

            ttnn_rms_norm_41 = ttnn.rms_norm(
                ttnn_to_memory_config_299,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.20.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_299, False)
            ttnn_matmul_103 = ttnn.matmul(
                ttnn_rms_norm_41,
                weights["model.layers.20.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_104 = ttnn.matmul(
                ttnn_rms_norm_41,
                weights["model.layers.20.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_41, False)
            ttnn_multiply_20 = ttnn.multiply(
                ttnn_matmul_103,
                ttnn_matmul_104,
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
            ttnn.deallocate(ttnn_matmul_104, False)
            ttnn.deallocate(ttnn_matmul_103, False)
            ttnn_to_memory_config_300 = ttnn.to_memory_config(
                ttnn_multiply_20,
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
            ttnn.deallocate(ttnn_multiply_20, False)
            ttnn_matmul_105 = ttnn.matmul(
                ttnn_to_memory_config_300,
                weights["model.layers.20.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_300, False)
            ttnn_add_41 = ttnn.add(
                ttnn_matmul_105,
                ttnn_add_40,
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
            ttnn.deallocate(ttnn_matmul_105, False)
            ttnn.deallocate(ttnn_add_40, False)

            return ttnn_add_41

        elif self.layer_idx == 21:
            # Map parameters to original variable names
            ttnn_to_memory_config_313 = hidden_states
            ttnn_add_42 = residual

            ttnn_rms_norm_43 = ttnn.rms_norm(
                ttnn_to_memory_config_313,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.21.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_313, False)
            ttnn_matmul_108 = ttnn.matmul(
                ttnn_rms_norm_43,
                weights["model.layers.21.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_109 = ttnn.matmul(
                ttnn_rms_norm_43,
                weights["model.layers.21.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_43, False)
            ttnn_multiply_21 = ttnn.multiply(
                ttnn_matmul_108,
                ttnn_matmul_109,
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
            ttnn.deallocate(ttnn_matmul_109, False)
            ttnn.deallocate(ttnn_matmul_108, False)
            ttnn_to_memory_config_314 = ttnn.to_memory_config(
                ttnn_multiply_21,
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
            ttnn.deallocate(ttnn_multiply_21, False)
            ttnn_matmul_110 = ttnn.matmul(
                ttnn_to_memory_config_314,
                weights["model.layers.21.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_314, False)
            ttnn_add_43 = ttnn.add(
                ttnn_matmul_110,
                ttnn_add_42,
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
            ttnn.deallocate(ttnn_matmul_110, False)
            ttnn.deallocate(ttnn_add_42, False)

            return ttnn_add_43

        elif self.layer_idx == 22:
            # Map parameters to original variable names
            ttnn_to_memory_config_327 = hidden_states
            ttnn_add_44 = residual

            ttnn_rms_norm_45 = ttnn.rms_norm(
                ttnn_to_memory_config_327,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.22.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_327, False)
            ttnn_matmul_113 = ttnn.matmul(
                ttnn_rms_norm_45,
                weights["model.layers.22.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_114 = ttnn.matmul(
                ttnn_rms_norm_45,
                weights["model.layers.22.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_45, False)
            ttnn_multiply_22 = ttnn.multiply(
                ttnn_matmul_113,
                ttnn_matmul_114,
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
            ttnn.deallocate(ttnn_matmul_114, False)
            ttnn.deallocate(ttnn_matmul_113, False)
            ttnn_to_memory_config_328 = ttnn.to_memory_config(
                ttnn_multiply_22,
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
            ttnn.deallocate(ttnn_multiply_22, False)
            ttnn_matmul_115 = ttnn.matmul(
                ttnn_to_memory_config_328,
                weights["model.layers.22.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_328, False)
            ttnn_add_45 = ttnn.add(
                ttnn_matmul_115,
                ttnn_add_44,
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
            ttnn.deallocate(ttnn_matmul_115, False)
            ttnn.deallocate(ttnn_add_44, False)

            return ttnn_add_45

        elif self.layer_idx == 23:
            # Map parameters to original variable names
            ttnn_to_memory_config_341 = hidden_states
            ttnn_add_46 = residual

            ttnn_rms_norm_47 = ttnn.rms_norm(
                ttnn_to_memory_config_341,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.23.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_341, False)
            ttnn_matmul_118 = ttnn.matmul(
                ttnn_rms_norm_47,
                weights["model.layers.23.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_119 = ttnn.matmul(
                ttnn_rms_norm_47,
                weights["model.layers.23.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_47, False)
            ttnn_multiply_23 = ttnn.multiply(
                ttnn_matmul_118,
                ttnn_matmul_119,
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
            ttnn.deallocate(ttnn_matmul_119, False)
            ttnn.deallocate(ttnn_matmul_118, False)
            ttnn_to_memory_config_342 = ttnn.to_memory_config(
                ttnn_multiply_23,
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
            ttnn.deallocate(ttnn_multiply_23, False)
            ttnn_matmul_120 = ttnn.matmul(
                ttnn_to_memory_config_342,
                weights["model.layers.23.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_342, False)
            ttnn_add_47 = ttnn.add(
                ttnn_matmul_120,
                ttnn_add_46,
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
            ttnn.deallocate(ttnn_matmul_120, False)
            ttnn.deallocate(ttnn_add_46, False)

            return ttnn_add_47

        elif self.layer_idx == 24:
            # Map parameters to original variable names
            ttnn_to_memory_config_355 = hidden_states
            ttnn_add_48 = residual

            ttnn_rms_norm_49 = ttnn.rms_norm(
                ttnn_to_memory_config_355,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.24.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_355, False)
            ttnn_matmul_123 = ttnn.matmul(
                ttnn_rms_norm_49,
                weights["model.layers.24.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_124 = ttnn.matmul(
                ttnn_rms_norm_49,
                weights["model.layers.24.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_49, False)
            ttnn_multiply_24 = ttnn.multiply(
                ttnn_matmul_123,
                ttnn_matmul_124,
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
            ttnn.deallocate(ttnn_matmul_124, False)
            ttnn.deallocate(ttnn_matmul_123, False)
            ttnn_to_memory_config_356 = ttnn.to_memory_config(
                ttnn_multiply_24,
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
            ttnn.deallocate(ttnn_multiply_24, False)
            ttnn_matmul_125 = ttnn.matmul(
                ttnn_to_memory_config_356,
                weights["model.layers.24.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_356, False)
            ttnn_add_49 = ttnn.add(
                ttnn_matmul_125,
                ttnn_add_48,
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
            ttnn.deallocate(ttnn_matmul_125, False)
            ttnn.deallocate(ttnn_add_48, False)

            return ttnn_add_49

        elif self.layer_idx == 25:
            # Map parameters to original variable names
            ttnn_to_memory_config_369 = hidden_states
            ttnn_add_50 = residual

            ttnn_rms_norm_51 = ttnn.rms_norm(
                ttnn_to_memory_config_369,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.25.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_369, False)
            ttnn_matmul_128 = ttnn.matmul(
                ttnn_rms_norm_51,
                weights["model.layers.25.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_129 = ttnn.matmul(
                ttnn_rms_norm_51,
                weights["model.layers.25.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_51, False)
            ttnn_multiply_25 = ttnn.multiply(
                ttnn_matmul_128,
                ttnn_matmul_129,
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
            ttnn.deallocate(ttnn_matmul_129, False)
            ttnn.deallocate(ttnn_matmul_128, False)
            ttnn_to_memory_config_370 = ttnn.to_memory_config(
                ttnn_multiply_25,
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
            ttnn.deallocate(ttnn_multiply_25, False)
            ttnn_matmul_130 = ttnn.matmul(
                ttnn_to_memory_config_370,
                weights["model.layers.25.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_370, False)
            ttnn_add_51 = ttnn.add(
                ttnn_matmul_130,
                ttnn_add_50,
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
            ttnn.deallocate(ttnn_matmul_130, False)
            ttnn.deallocate(ttnn_add_50, False)

            return ttnn_add_51

        elif self.layer_idx == 26:
            # Map parameters to original variable names
            ttnn_to_memory_config_383 = hidden_states
            ttnn_add_52 = residual

            ttnn_rms_norm_53 = ttnn.rms_norm(
                ttnn_to_memory_config_383,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.26.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_383, False)
            ttnn_matmul_133 = ttnn.matmul(
                ttnn_rms_norm_53,
                weights["model.layers.26.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_134 = ttnn.matmul(
                ttnn_rms_norm_53,
                weights["model.layers.26.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_53, False)
            ttnn_multiply_26 = ttnn.multiply(
                ttnn_matmul_133,
                ttnn_matmul_134,
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
            ttnn.deallocate(ttnn_matmul_134, False)
            ttnn.deallocate(ttnn_matmul_133, False)
            ttnn_to_memory_config_384 = ttnn.to_memory_config(
                ttnn_multiply_26,
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
            ttnn.deallocate(ttnn_multiply_26, False)
            ttnn_matmul_135 = ttnn.matmul(
                ttnn_to_memory_config_384,
                weights["model.layers.26.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_384, False)
            ttnn_add_53 = ttnn.add(
                ttnn_matmul_135,
                ttnn_add_52,
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
            ttnn.deallocate(ttnn_matmul_135, False)
            ttnn.deallocate(ttnn_add_52, False)

            return ttnn_add_53

        elif self.layer_idx == 27:
            # Map parameters to original variable names
            ttnn_to_memory_config_397 = hidden_states
            ttnn_add_54 = residual

            ttnn_rms_norm_55 = ttnn.rms_norm(
                ttnn_to_memory_config_397,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.27.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_397, False)
            ttnn_matmul_138 = ttnn.matmul(
                ttnn_rms_norm_55,
                weights["model.layers.27.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_139 = ttnn.matmul(
                ttnn_rms_norm_55,
                weights["model.layers.27.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_55, False)
            ttnn_multiply_27 = ttnn.multiply(
                ttnn_matmul_138,
                ttnn_matmul_139,
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
            ttnn.deallocate(ttnn_matmul_139, False)
            ttnn.deallocate(ttnn_matmul_138, False)
            ttnn_to_memory_config_398 = ttnn.to_memory_config(
                ttnn_multiply_27,
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
            ttnn.deallocate(ttnn_multiply_27, False)
            ttnn_matmul_140 = ttnn.matmul(
                ttnn_to_memory_config_398,
                weights["model.layers.27.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_398, False)
            ttnn_add_55 = ttnn.add(
                ttnn_matmul_140,
                ttnn_add_54,
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
            ttnn.deallocate(ttnn_matmul_140, False)
            ttnn.deallocate(ttnn_add_54, False)

            return ttnn_add_55

        elif self.layer_idx == 28:
            # Map parameters to original variable names
            ttnn_to_memory_config_411 = hidden_states
            ttnn_add_56 = residual

            ttnn_rms_norm_57 = ttnn.rms_norm(
                ttnn_to_memory_config_411,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.28.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_411, False)
            ttnn_matmul_143 = ttnn.matmul(
                ttnn_rms_norm_57,
                weights["model.layers.28.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_144 = ttnn.matmul(
                ttnn_rms_norm_57,
                weights["model.layers.28.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_57, False)
            ttnn_multiply_28 = ttnn.multiply(
                ttnn_matmul_143,
                ttnn_matmul_144,
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
            ttnn.deallocate(ttnn_matmul_144, False)
            ttnn.deallocate(ttnn_matmul_143, False)
            ttnn_to_memory_config_412 = ttnn.to_memory_config(
                ttnn_multiply_28,
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
            ttnn.deallocate(ttnn_multiply_28, False)
            ttnn_matmul_145 = ttnn.matmul(
                ttnn_to_memory_config_412,
                weights["model.layers.28.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_412, False)
            ttnn_add_57 = ttnn.add(
                ttnn_matmul_145,
                ttnn_add_56,
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
            ttnn.deallocate(ttnn_matmul_145, False)
            ttnn.deallocate(ttnn_add_56, False)

            return ttnn_add_57

        elif self.layer_idx == 29:
            # Map parameters to original variable names
            ttnn_to_memory_config_425 = hidden_states
            ttnn_add_58 = residual

            ttnn_rms_norm_59 = ttnn.rms_norm(
                ttnn_to_memory_config_425,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.29.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_425, False)
            ttnn_matmul_148 = ttnn.matmul(
                ttnn_rms_norm_59,
                weights["model.layers.29.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_149 = ttnn.matmul(
                ttnn_rms_norm_59,
                weights["model.layers.29.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_59, False)
            ttnn_multiply_29 = ttnn.multiply(
                ttnn_matmul_148,
                ttnn_matmul_149,
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
            ttnn.deallocate(ttnn_matmul_149, False)
            ttnn.deallocate(ttnn_matmul_148, False)
            ttnn_to_memory_config_426 = ttnn.to_memory_config(
                ttnn_multiply_29,
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
            ttnn.deallocate(ttnn_multiply_29, False)
            ttnn_matmul_150 = ttnn.matmul(
                ttnn_to_memory_config_426,
                weights["model.layers.29.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_426, False)
            ttnn_add_59 = ttnn.add(
                ttnn_matmul_150,
                ttnn_add_58,
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
            ttnn.deallocate(ttnn_matmul_150, False)
            ttnn.deallocate(ttnn_add_58, False)

            return ttnn_add_59

        elif self.layer_idx == 30:
            # Map parameters to original variable names
            ttnn_to_memory_config_439 = hidden_states
            ttnn_add_60 = residual

            ttnn_rms_norm_61 = ttnn.rms_norm(
                ttnn_to_memory_config_439,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.30.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_439, False)
            ttnn_matmul_153 = ttnn.matmul(
                ttnn_rms_norm_61,
                weights["model.layers.30.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_154 = ttnn.matmul(
                ttnn_rms_norm_61,
                weights["model.layers.30.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_61, False)
            ttnn_multiply_30 = ttnn.multiply(
                ttnn_matmul_153,
                ttnn_matmul_154,
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
            ttnn.deallocate(ttnn_matmul_154, False)
            ttnn.deallocate(ttnn_matmul_153, False)
            ttnn_to_memory_config_440 = ttnn.to_memory_config(
                ttnn_multiply_30,
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
            ttnn.deallocate(ttnn_multiply_30, False)
            ttnn_matmul_155 = ttnn.matmul(
                ttnn_to_memory_config_440,
                weights["model.layers.30.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_440, False)
            ttnn_add_61 = ttnn.add(
                ttnn_matmul_155,
                ttnn_add_60,
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
            ttnn.deallocate(ttnn_matmul_155, False)
            ttnn.deallocate(ttnn_add_60, False)

            return ttnn_add_61

        elif self.layer_idx == 31:
            # Map parameters to original variable names
            ttnn_to_memory_config_453 = hidden_states
            ttnn_add_62 = residual

            ttnn_rms_norm_63 = ttnn.rms_norm(
                ttnn_to_memory_config_453,
                epsilon=9.9999997473787516e-06,
                weight=weights[
                    "model.layers.31.post_attention_layernorm.parametrizations.weight.original"
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
            ttnn.deallocate(ttnn_to_memory_config_453, False)
            ttnn_matmul_158 = ttnn.matmul(
                ttnn_rms_norm_63,
                weights["model.layers.31.mlp.gate_proj.parametrizations.weight.original"],
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
            ttnn_matmul_159 = ttnn.matmul(
                ttnn_rms_norm_63,
                weights["model.layers.31.mlp.up_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_rms_norm_63, False)
            ttnn_multiply_31 = ttnn.multiply(
                ttnn_matmul_158,
                ttnn_matmul_159,
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
            ttnn.deallocate(ttnn_matmul_159, False)
            ttnn.deallocate(ttnn_matmul_158, False)
            ttnn_to_memory_config_454 = ttnn.to_memory_config(
                ttnn_multiply_31,
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
            ttnn.deallocate(ttnn_multiply_31, False)
            ttnn_matmul_160 = ttnn.matmul(
                ttnn_to_memory_config_454,
                weights["model.layers.31.mlp.down_proj.parametrizations.weight.original"],
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
            ttnn.deallocate(ttnn_to_memory_config_454, False)
            ttnn_add_63 = ttnn.add(
                ttnn_matmul_160,
                ttnn_add_62,
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
            ttnn.deallocate(ttnn_matmul_160, False)
            ttnn.deallocate(ttnn_add_62, False)

            return ttnn_add_63

