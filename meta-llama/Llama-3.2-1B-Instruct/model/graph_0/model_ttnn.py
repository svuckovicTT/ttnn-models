import ttnn
import constants
import params


class LightweightModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ModelTTNN(LightweightModule):
    def __init__(self, device):
        self.device = device
        self.weights = params.load_weights_for__main_from_state_dict(device)
        self.weights = constants.run_consteval(self.weights, device)
        self.model = LlamaModel(device, self.weights)

    def forward(self, activations):
        args_1 = activations[0]
        args_0 = activations[1]
        args_2 = activations[2]
        args_3 = activations[3]
        args_4 = activations[4]
        args_5 = activations[5]
        args_6 = activations[6]
        args_7 = activations[7]
        args_8 = activations[8]
        args_9 = activations[9]
        args_10 = activations[10]
        args_11 = activations[11]
        args_12 = activations[12]
        args_13 = activations[13]
        args_14 = activations[14]
        args_15 = activations[15]
        args_16 = activations[16]
        args_17 = activations[17]
        args_18 = activations[18]
        args_19 = activations[19]
        args_20 = activations[20]
        args_21 = activations[21]
        args_22 = activations[22]
        args_23 = activations[23]
        args_24 = activations[24]
        args_25 = activations[25]
        args_26 = activations[26]
        args_27 = activations[27]
        args_28 = activations[28]
        args_29 = activations[29]
        args_30 = activations[30]
        args_31 = activations[31]
        args_32 = activations[32]
        args_33 = activations[33]
        args_34 = activations[34]
        args_35 = activations[35]
        args_36 = activations[36]
        args_37 = activations[37]
        args_38 = activations[38]
        args_39 = activations[39]
        args_40 = activations[40]
        args_41 = activations[41]
        args_42 = activations[42]
        args_43 = activations[43]
        args_44 = activations[44]
        args_45 = activations[45]
        args_46 = activations[46]
        args_47 = activations[47]
        args_48 = activations[48]
        hidden_states, layer_outputs = self.model(
            args_0,
            args_1,
            args_2,
            args_3,
            args_4,
            args_5,
            args_6,
            args_7,
            args_8,
            args_9,
            args_10,
            args_11,
            args_12,
            args_13,
            args_14,
            args_15,
            args_16,
            args_17,
            args_18,
            args_19,
            args_20,
            args_21,
            args_22,
            args_23,
            args_24,
            args_25,
            args_26,
            args_27,
            args_28,
            args_29,
            args_30,
            args_31,
            args_32,
            args_33,
            args_34,
            args_35,
            args_36,
            args_37,
            args_38,
            args_39,
            args_40,
            args_41,
            args_42,
            args_43,
            args_44,
            args_45,
            args_46,
            args_47,
            args_48,
        )
        ttnn_matmul_81 = ttnn.matmul(
            hidden_states,
            self.weights["lm_head.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(hidden_states, False)
        ttnn_reshape_36 = ttnn.reshape(
            ttnn_matmul_81,
            [32, 18, 128256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        return_list = []
        for a_K, a_V, ret in layer_outputs:
            return_list.append(a_K)
            return_list.append(a_V)
            return_list.append(ret)
        return_list.append(ttnn_matmul_81)
        return_list.append(ttnn_reshape_36)
        return return_list


class LlamaModel(LightweightModule):
    def __init__(self, device, weights):
        self.device = device
        self.weights = weights
        self.layer_0 = LlamaDecoderLayerFirst(device, weights)
        self.layers = [
            LlamaDecoderLayer(i, device, weights, is_last=(i == 15))
            for i in range(1, 16)
        ]

    def forward(
        self,
        args_0,
        args_1,
        args_2,
        args_3,
        args_4,
        args_5,
        args_6,
        args_7,
        args_8,
        args_9,
        args_10,
        args_11,
        args_12,
        args_13,
        args_14,
        args_15,
        args_16,
        args_17,
        args_18,
        args_19,
        args_20,
        args_21,
        args_22,
        args_23,
        args_24,
        args_25,
        args_26,
        args_27,
        args_28,
        args_29,
        args_30,
        args_31,
        args_32,
        args_33,
        args_34,
        args_35,
        args_36,
        args_37,
        args_38,
        args_39,
        args_40,
        args_41,
        args_42,
        args_43,
        args_44,
        args_45,
        args_46,
        args_47,
        args_48,
    ):
        var_0 = self.weights["const_seq_len_18"]
        ttnn_to_memory_config_0 = ttnn.to_memory_config(
            self.weights["iota_seq_18"],
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
        ttnn_add_0 = ttnn.add(
            args_1,
            ttnn_to_memory_config_0,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(
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
        ttnn.deallocate(ttnn_to_memory_config_0, False)
        ttnn_typecast_146 = ttnn.typecast(
            args_0,
            ttnn.DataType.UINT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(args_0, False)
        ttnn_reshape_0 = ttnn.reshape(
            ttnn_typecast_146,
            [576],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_146, False)
        ttnn_to_memory_config_1 = ttnn.to_memory_config(
            ttnn_reshape_0,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_0, False)
        ttnn_embedding_0 = ttnn.embedding(
            ttnn_to_memory_config_1,
            self.weights["model.embed_tokens.weight"],
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_to_memory_config_1, False)
        ttnn_to_memory_config_2 = ttnn.to_memory_config(
            ttnn_embedding_0,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 192],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn_add_3, cos, sin, causal_mask, ret0 = self.layer_0(
            ttnn_to_memory_config_2,
            ttnn_embedding_0,
            args_1,
            ttnn_add_0,
            var_0,
            args_2,
            args_3,
        )
        hidden_states = ttnn_add_3
        layer_outputs = [(args_2, args_3, ret0)]
        layer_args = [
            (args_4, args_5, args_6),
            (args_7, args_8, args_9),
            (args_10, args_11, args_12),
            (args_13, args_14, args_15),
            (args_16, args_17, args_18),
            (args_19, args_20, args_21),
            (args_22, args_23, args_24),
            (args_25, args_26, args_27),
            (args_28, args_29, args_30),
            (args_31, args_32, args_33),
            (args_34, args_35, args_36),
            (args_37, args_38, args_39),
            (args_40, args_41, args_42),
            (args_43, args_44, args_45),
            (args_46, args_47, args_48),
        ]
        for layer, (a_attn, a_K, a_V) in zip(self.layers, layer_args):
            hidden_states, ret = layer(
                hidden_states,
                cos,
                sin,
                causal_mask,
                a_attn,
                a_K,
                a_V,
                var_0,
            )
            layer_outputs.append((a_K, a_V, ret))
        ttnn_rms_norm_32 = ttnn.rms_norm(
            hidden_states,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["model.norm.weight"],
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 192],
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
        return ttnn_rms_norm_32, layer_outputs


class LlamaDecoderLayerFirst(LightweightModule):
    def __init__(self, device, weights):
        self.device = device
        self.weights = weights

    def forward(
        self,
        ttnn_to_memory_config_2,
        ttnn_embedding_0,
        args_1,
        ttnn_add_0,
        var_0,
        args_2,
        args_3,
    ):
        ttnn_rms_norm_0 = ttnn.rms_norm(
            ttnn_to_memory_config_2,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["model.layers.0.input_layernorm.weight"],
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 192],
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
        ttnn.deallocate(ttnn_to_memory_config_2, False)
        ttnn_matmul_0 = ttnn.matmul(
            ttnn_rms_norm_0,
            self.weights["model.layers.0.self_attn.qkv_proj.weight"],
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 288],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=3,
                out_block_h=2,
                out_block_w=9,
                per_core_M=2,
                per_core_N=9,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            ),
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_rms_norm_0, False)
        ttnn_reshape_1 = ttnn.reshape(
            ttnn_matmul_0,
            [32, 18, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_0, False)
        v_1, v_2, v_3 = ttnn.transformer.split_query_key_value_and_split_heads(
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
        ttnn_to_memory_config_3 = ttnn.to_memory_config(
            ttnn_add_0,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn_typecast_147 = ttnn.typecast(
            ttnn_to_memory_config_3,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_to_memory_config_3, False)
        ttnn_reshape_2 = ttnn.reshape(
            ttnn_typecast_147,
            [1, 1, 18],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_147, False)
        ttnn_matmul_1 = ttnn.matmul(
            self.weights["model.rotary_emb.inv_freq"],
            ttnn_reshape_2,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(
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
            dtype=ttnn.DataType.FLOAT32,
            program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=1,
                per_core_M=1,
                per_core_N=1,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
                gather_in0=False,
                hop_cores=ttnn.CoreRangeSet([]),
                num_global_cb_receivers=0,
                untilize_out=False,
            ),
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_reshape_2, False)
        ttnn_permute_0 = ttnn.permute(
            ttnn_matmul_1,
            [0, 2, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_matmul_1, False)
        ttnn_reshape_3 = ttnn.reshape(
            ttnn_permute_0,
            [1, 1, 18, 32],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_permute_0, False)
        ttnn_concat_16 = ttnn.concat(
            [ttnn_reshape_3, ttnn_reshape_3],
            3,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_3, False)
        ttnn_to_memory_config_4 = ttnn.to_memory_config(
            ttnn_concat_16,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
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
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                    ),
                    [32, 32],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_to_memory_config_4, False)
        ttnn_typecast_148 = ttnn.typecast(
            ttnn_cos_0,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                    ),
                    [32, 32],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_cos_0, False)
        ttnn_to_memory_config_5 = ttnn.to_memory_config(
            ttnn_concat_16,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                    ),
                    [32, 32],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_concat_16, False)
        ttnn_sin_0 = ttnn.sin(
            ttnn_to_memory_config_5,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                    ),
                    [32, 32],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_to_memory_config_5, False)
        ttnn_typecast_149 = ttnn.typecast(
            ttnn_sin_0,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                    ),
                    [32, 32],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_sin_0, False)
        ttnn_to_memory_config_6 = ttnn.to_memory_config(
            ttnn_typecast_149,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn_to_memory_config_7 = ttnn.to_memory_config(
            ttnn_typecast_148,
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
        ttnn_slice_48 = ttnn.slice(
            ttnn_experimental_rotary_embedding_0,
            [0, 0, 0, 0],
            [32, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_experimental_rotary_embedding_0, False)
        ttnn_slice_49 = ttnn.slice(
            ttnn_slice_48,
            [0, 0, 0, 0],
            [1, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_49, 0)
        ttnn.deallocate(ttnn_slice_49, False)
        ttnn_slice_50 = ttnn.slice(
            ttnn_slice_48,
            [1, 0, 0, 0],
            [2, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_50, 1)
        ttnn.deallocate(ttnn_slice_50, False)
        ttnn_slice_51 = ttnn.slice(
            ttnn_slice_48,
            [2, 0, 0, 0],
            [3, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_51, 2)
        ttnn.deallocate(ttnn_slice_51, False)
        ttnn_slice_52 = ttnn.slice(
            ttnn_slice_48,
            [3, 0, 0, 0],
            [4, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_52, 3)
        ttnn.deallocate(ttnn_slice_52, False)
        ttnn_slice_53 = ttnn.slice(
            ttnn_slice_48,
            [4, 0, 0, 0],
            [5, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_53, 4)
        ttnn.deallocate(ttnn_slice_53, False)
        ttnn_slice_54 = ttnn.slice(
            ttnn_slice_48,
            [5, 0, 0, 0],
            [6, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_54, 5)
        ttnn.deallocate(ttnn_slice_54, False)
        ttnn_slice_55 = ttnn.slice(
            ttnn_slice_48,
            [6, 0, 0, 0],
            [7, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_55, 6)
        ttnn.deallocate(ttnn_slice_55, False)
        ttnn_slice_56 = ttnn.slice(
            ttnn_slice_48,
            [7, 0, 0, 0],
            [8, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_56, 7)
        ttnn.deallocate(ttnn_slice_56, False)
        ttnn_slice_57 = ttnn.slice(
            ttnn_slice_48,
            [8, 0, 0, 0],
            [9, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_57, 8)
        ttnn.deallocate(ttnn_slice_57, False)
        ttnn_slice_58 = ttnn.slice(
            ttnn_slice_48,
            [9, 0, 0, 0],
            [10, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_58, 9)
        ttnn.deallocate(ttnn_slice_58, False)
        ttnn_slice_59 = ttnn.slice(
            ttnn_slice_48,
            [10, 0, 0, 0],
            [11, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_59, 10)
        ttnn.deallocate(ttnn_slice_59, False)
        ttnn_slice_60 = ttnn.slice(
            ttnn_slice_48,
            [11, 0, 0, 0],
            [12, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_60, 11)
        ttnn.deallocate(ttnn_slice_60, False)
        ttnn_slice_61 = ttnn.slice(
            ttnn_slice_48,
            [12, 0, 0, 0],
            [13, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_61, 12)
        ttnn.deallocate(ttnn_slice_61, False)
        ttnn_slice_62 = ttnn.slice(
            ttnn_slice_48,
            [13, 0, 0, 0],
            [14, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_62, 13)
        ttnn.deallocate(ttnn_slice_62, False)
        ttnn_slice_63 = ttnn.slice(
            ttnn_slice_48,
            [14, 0, 0, 0],
            [15, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_63, 14)
        ttnn.deallocate(ttnn_slice_63, False)
        ttnn_slice_64 = ttnn.slice(
            ttnn_slice_48,
            [15, 0, 0, 0],
            [16, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_64, 15)
        ttnn.deallocate(ttnn_slice_64, False)
        ttnn_slice_65 = ttnn.slice(
            ttnn_slice_48,
            [16, 0, 0, 0],
            [17, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_65, 16)
        ttnn.deallocate(ttnn_slice_65, False)
        ttnn_slice_66 = ttnn.slice(
            ttnn_slice_48,
            [17, 0, 0, 0],
            [18, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_66, 17)
        ttnn.deallocate(ttnn_slice_66, False)
        ttnn_slice_67 = ttnn.slice(
            ttnn_slice_48,
            [18, 0, 0, 0],
            [19, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_67, 18)
        ttnn.deallocate(ttnn_slice_67, False)
        ttnn_slice_68 = ttnn.slice(
            ttnn_slice_48,
            [19, 0, 0, 0],
            [20, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_68, 19)
        ttnn.deallocate(ttnn_slice_68, False)
        ttnn_slice_69 = ttnn.slice(
            ttnn_slice_48,
            [20, 0, 0, 0],
            [21, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_69, 20)
        ttnn.deallocate(ttnn_slice_69, False)
        ttnn_slice_70 = ttnn.slice(
            ttnn_slice_48,
            [21, 0, 0, 0],
            [22, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_70, 21)
        ttnn.deallocate(ttnn_slice_70, False)
        ttnn_slice_71 = ttnn.slice(
            ttnn_slice_48,
            [22, 0, 0, 0],
            [23, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_71, 22)
        ttnn.deallocate(ttnn_slice_71, False)
        ttnn_slice_72 = ttnn.slice(
            ttnn_slice_48,
            [23, 0, 0, 0],
            [24, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_72, 23)
        ttnn.deallocate(ttnn_slice_72, False)
        ttnn_slice_73 = ttnn.slice(
            ttnn_slice_48,
            [24, 0, 0, 0],
            [25, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_73, 24)
        ttnn.deallocate(ttnn_slice_73, False)
        ttnn_slice_74 = ttnn.slice(
            ttnn_slice_48,
            [25, 0, 0, 0],
            [26, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_74, 25)
        ttnn.deallocate(ttnn_slice_74, False)
        ttnn_slice_75 = ttnn.slice(
            ttnn_slice_48,
            [26, 0, 0, 0],
            [27, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_75, 26)
        ttnn.deallocate(ttnn_slice_75, False)
        ttnn_slice_76 = ttnn.slice(
            ttnn_slice_48,
            [27, 0, 0, 0],
            [28, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_76, 27)
        ttnn.deallocate(ttnn_slice_76, False)
        ttnn_slice_77 = ttnn.slice(
            ttnn_slice_48,
            [28, 0, 0, 0],
            [29, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_77, 28)
        ttnn.deallocate(ttnn_slice_77, False)
        ttnn_slice_78 = ttnn.slice(
            ttnn_slice_48,
            [29, 0, 0, 0],
            [30, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_78, 29)
        ttnn.deallocate(ttnn_slice_78, False)
        ttnn_slice_79 = ttnn.slice(
            ttnn_slice_48,
            [30, 0, 0, 0],
            [31, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_2, ttnn_slice_79, 30)
        ttnn.deallocate(ttnn_slice_79, False)
        ttnn_slice_80 = ttnn.slice(
            ttnn_slice_48,
            [31, 0, 0, 0],
            [32, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_slice_48, False)
        ttnn.fill_cache(args_2, ttnn_slice_80, 31)
        ttnn.deallocate(ttnn_slice_80, False)
        ttnn_slice_81 = ttnn.slice(
            v_3,
            [0, 0, 0, 0],
            [1, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_81, 0)
        ttnn.deallocate(ttnn_slice_81, False)
        ttnn_slice_82 = ttnn.slice(
            v_3,
            [1, 0, 0, 0],
            [2, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_82, 1)
        ttnn.deallocate(ttnn_slice_82, False)
        ttnn_slice_83 = ttnn.slice(
            v_3,
            [2, 0, 0, 0],
            [3, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_83, 2)
        ttnn.deallocate(ttnn_slice_83, False)
        ttnn_slice_84 = ttnn.slice(
            v_3,
            [3, 0, 0, 0],
            [4, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_84, 3)
        ttnn.deallocate(ttnn_slice_84, False)
        ttnn_slice_85 = ttnn.slice(
            v_3,
            [4, 0, 0, 0],
            [5, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_85, 4)
        ttnn.deallocate(ttnn_slice_85, False)
        ttnn_slice_86 = ttnn.slice(
            v_3,
            [5, 0, 0, 0],
            [6, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_86, 5)
        ttnn.deallocate(ttnn_slice_86, False)
        ttnn_slice_87 = ttnn.slice(
            v_3,
            [6, 0, 0, 0],
            [7, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_87, 6)
        ttnn.deallocate(ttnn_slice_87, False)
        ttnn_slice_88 = ttnn.slice(
            v_3,
            [7, 0, 0, 0],
            [8, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_88, 7)
        ttnn.deallocate(ttnn_slice_88, False)
        ttnn_slice_89 = ttnn.slice(
            v_3,
            [8, 0, 0, 0],
            [9, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_89, 8)
        ttnn.deallocate(ttnn_slice_89, False)
        ttnn_slice_90 = ttnn.slice(
            v_3,
            [9, 0, 0, 0],
            [10, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_90, 9)
        ttnn.deallocate(ttnn_slice_90, False)
        ttnn_slice_91 = ttnn.slice(
            v_3,
            [10, 0, 0, 0],
            [11, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_91, 10)
        ttnn.deallocate(ttnn_slice_91, False)
        ttnn_slice_92 = ttnn.slice(
            v_3,
            [11, 0, 0, 0],
            [12, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_92, 11)
        ttnn.deallocate(ttnn_slice_92, False)
        ttnn_slice_93 = ttnn.slice(
            v_3,
            [12, 0, 0, 0],
            [13, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_93, 12)
        ttnn.deallocate(ttnn_slice_93, False)
        ttnn_slice_94 = ttnn.slice(
            v_3,
            [13, 0, 0, 0],
            [14, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_94, 13)
        ttnn.deallocate(ttnn_slice_94, False)
        ttnn_slice_95 = ttnn.slice(
            v_3,
            [14, 0, 0, 0],
            [15, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_95, 14)
        ttnn.deallocate(ttnn_slice_95, False)
        ttnn_slice_96 = ttnn.slice(
            v_3,
            [15, 0, 0, 0],
            [16, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_96, 15)
        ttnn.deallocate(ttnn_slice_96, False)
        ttnn_slice_97 = ttnn.slice(
            v_3,
            [16, 0, 0, 0],
            [17, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_97, 16)
        ttnn.deallocate(ttnn_slice_97, False)
        ttnn_slice_98 = ttnn.slice(
            v_3,
            [17, 0, 0, 0],
            [18, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_98, 17)
        ttnn.deallocate(ttnn_slice_98, False)
        ttnn_slice_99 = ttnn.slice(
            v_3,
            [18, 0, 0, 0],
            [19, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_99, 18)
        ttnn.deallocate(ttnn_slice_99, False)
        ttnn_slice_100 = ttnn.slice(
            v_3,
            [19, 0, 0, 0],
            [20, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_100, 19)
        ttnn.deallocate(ttnn_slice_100, False)
        ttnn_slice_101 = ttnn.slice(
            v_3,
            [20, 0, 0, 0],
            [21, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_101, 20)
        ttnn.deallocate(ttnn_slice_101, False)
        ttnn_slice_102 = ttnn.slice(
            v_3,
            [21, 0, 0, 0],
            [22, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_102, 21)
        ttnn.deallocate(ttnn_slice_102, False)
        ttnn_slice_103 = ttnn.slice(
            v_3,
            [22, 0, 0, 0],
            [23, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_103, 22)
        ttnn.deallocate(ttnn_slice_103, False)
        ttnn_slice_104 = ttnn.slice(
            v_3,
            [23, 0, 0, 0],
            [24, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_104, 23)
        ttnn.deallocate(ttnn_slice_104, False)
        ttnn_slice_105 = ttnn.slice(
            v_3,
            [24, 0, 0, 0],
            [25, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_105, 24)
        ttnn.deallocate(ttnn_slice_105, False)
        ttnn_slice_106 = ttnn.slice(
            v_3,
            [25, 0, 0, 0],
            [26, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_106, 25)
        ttnn.deallocate(ttnn_slice_106, False)
        ttnn_slice_107 = ttnn.slice(
            v_3,
            [26, 0, 0, 0],
            [27, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_107, 26)
        ttnn.deallocate(ttnn_slice_107, False)
        ttnn_slice_108 = ttnn.slice(
            v_3,
            [27, 0, 0, 0],
            [28, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_108, 27)
        ttnn.deallocate(ttnn_slice_108, False)
        ttnn_slice_109 = ttnn.slice(
            v_3,
            [28, 0, 0, 0],
            [29, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_109, 28)
        ttnn.deallocate(ttnn_slice_109, False)
        ttnn_slice_110 = ttnn.slice(
            v_3,
            [29, 0, 0, 0],
            [30, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_110, 29)
        ttnn.deallocate(ttnn_slice_110, False)
        ttnn_slice_111 = ttnn.slice(
            v_3,
            [30, 0, 0, 0],
            [31, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_3, ttnn_slice_111, 30)
        ttnn.deallocate(ttnn_slice_111, False)
        ttnn_slice_112 = ttnn.slice(
            v_3,
            [31, 0, 0, 0],
            [32, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(v_3, False)
        ttnn.fill_cache(args_3, ttnn_slice_112, 31)
        ttnn.deallocate(ttnn_slice_112, False)
        ttnn_to_memory_config_8 = ttnn.to_memory_config(
            var_0,
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
        ttnn_add_1 = ttnn.add(
            args_1,
            ttnn_to_memory_config_8,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(
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
        ttnn.deallocate(ttnn_to_memory_config_8, False)
        ttnn.deallocate(args_1, False)
        ttnn_to_memory_config_9 = ttnn.to_memory_config(
            ttnn_add_1,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_1, False)
        ttnn_to_memory_config_10 = ttnn.to_memory_config(
            ttnn_typecast_149,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn_to_memory_config_11 = ttnn.to_memory_config(
            ttnn_typecast_148,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn_experimental_rotary_embedding_1 = ttnn.experimental.rotary_embedding(
            v_1,
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
        ttnn.deallocate(v_1, False)
        ttnn_slice_113 = ttnn.slice(
            ttnn_experimental_rotary_embedding_1,
            [0, 0, 0, 0],
            [32, 32, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_experimental_rotary_embedding_1, False)
        ttnn_reshape_4 = ttnn.reshape(
            ttnn_add_0,
            [1, 1, 18, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_add_0, False)
        ttnn_to_memory_config_12 = ttnn.to_memory_config(
            ttnn_reshape_4,
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
        ttnn.deallocate(ttnn_reshape_4, False)
        ttnn_to_memory_config_13 = ttnn.to_memory_config(
            self.weights["iota_kv_128"],
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
            ttnn_to_memory_config_12,
            ttnn_to_memory_config_13,
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
        ttnn.deallocate(ttnn_to_memory_config_13, False)
        ttnn.deallocate(ttnn_to_memory_config_12, False)
        ttnn_to_memory_config_14 = ttnn.to_memory_config(
            self.weights["const_attn_mask_neg_inf"],
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
        ttnn_to_memory_config_15 = ttnn.to_memory_config(
            self.weights["const_attn_mask_zero"],
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
            ttnn_to_memory_config_15,
            ttnn_to_memory_config_14,
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
        ttnn.deallocate(ttnn_to_memory_config_15, False)
        ttnn.deallocate(ttnn_to_memory_config_14, False)
        ttnn.deallocate(ttnn_ge_0, False)
        ttnn_to_memory_config_16 = ttnn.to_memory_config(
            ttnn_where_0,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_transformer_scaled_dot_product_attention_0 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_slice_113,
                args_2,
                args_3,
                attn_mask=ttnn_to_memory_config_16,
                is_causal=False,
                scale=0.1249999925494194,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn.deallocate(ttnn_to_memory_config_16, False)
        ttnn.deallocate(ttnn_slice_113, False)
        ttnn_transformer_concatenate_heads_0 = ttnn.transformer.concatenate_heads(
            ttnn_transformer_scaled_dot_product_attention_0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_0, False)
        ttnn_reshape_5 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_0,
            [576, 2048],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_transformer_concatenate_heads_0, False)
        ttnn_matmul_2 = ttnn.matmul(
            ttnn_reshape_5,
            self.weights["model.layers.0.self_attn.o_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 192],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=6,
                out_block_h=2,
                out_block_w=6,
                per_core_M=2,
                per_core_N=6,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            ),
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_reshape_5, False)
        ttnn_add_2 = ttnn.add(
            ttnn_matmul_2,
            ttnn_embedding_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 192],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_matmul_2, False)
        ttnn.deallocate(ttnn_embedding_0, False)
        ttnn_rms_norm_1 = ttnn.rms_norm(
            ttnn_add_2,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["model.layers.0.post_attention_layernorm.weight"],
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 192],
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
        ttnn_matmul_3 = ttnn.matmul(
            ttnn_rms_norm_1,
            self.weights["model.layers.0.mlp.gate_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 768],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=8,
                out_block_h=2,
                out_block_w=24,
                per_core_M=2,
                per_core_N=24,
                transpose_mcast=False,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
                fuse_batch=True,
            ),
            activation=None,
            compute_kernel_config=None,
        )
        ttnn_matmul_4 = ttnn.matmul(
            ttnn_rms_norm_1,
            self.weights["model.layers.0.mlp.up_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 768],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=8,
                out_block_h=2,
                out_block_w=24,
                per_core_M=2,
                per_core_N=24,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
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
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 768],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_matmul_4, False)
        ttnn.deallocate(ttnn_matmul_3, False)
        ttnn_matmul_5 = ttnn.matmul(
            ttnn_multiply_0,
            self.weights["model.layers.0.mlp.down_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 192],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=6,
                out_block_h=2,
                out_block_w=6,
                per_core_M=2,
                per_core_N=6,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            ),
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_0, False)
        ttnn_add_3 = ttnn.add(
            ttnn_matmul_5,
            ttnn_add_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 192],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_matmul_5, False)
        ttnn.deallocate(ttnn_add_2, False)
        return (
            ttnn_add_3,
            ttnn_typecast_148,
            ttnn_typecast_149,
            ttnn_where_0,
            ttnn_to_memory_config_9,
        )


class LlamaDecoderLayer(LightweightModule):
    def __init__(self, layer_idx, device, weights, is_last=False):
        self.layer_idx = layer_idx
        self.device = device
        self.weights = weights
        self.is_last = is_last

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        causal_mask,
        args_attn,
        args_K,
        args_V,
        var_0,
    ):
        ttnn_rms_norm_2 = ttnn.rms_norm(
            hidden_states,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["model.layers." + str(self.layer_idx) + ".input_layernorm.weight"],
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 192],
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
        ttnn_matmul_6 = ttnn.matmul(
            ttnn_rms_norm_2,
            self.weights["model.layers." + str(self.layer_idx) + ".self_attn.qkv_proj.weight"],
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 288],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=3,
                out_block_h=2,
                out_block_w=9,
                per_core_M=2,
                per_core_N=9,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            ),
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_rms_norm_2, False)
        ttnn_reshape_6 = ttnn.reshape(
            ttnn_matmul_6,
            [32, 18, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_6, False)
        v_4, v_5, v_6 = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn_reshape_6,
            None,
            num_heads=32,
            num_kv_heads=8,
            transpose_key=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_6, False)
        ttnn_to_memory_config_17 = ttnn.to_memory_config(
            sin,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn_to_memory_config_18 = ttnn.to_memory_config(
            cos,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn_experimental_rotary_embedding_2 = ttnn.experimental.rotary_embedding(
            v_5,
            ttnn_to_memory_config_18,
            ttnn_to_memory_config_17,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_to_memory_config_18, False)
        ttnn.deallocate(ttnn_to_memory_config_17, False)
        ttnn.deallocate(v_5, False)
        ttnn_slice_114 = ttnn.slice(
            ttnn_experimental_rotary_embedding_2,
            [0, 0, 0, 0],
            [32, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_experimental_rotary_embedding_2, False)
        ttnn_slice_115 = ttnn.slice(
            ttnn_slice_114,
            [0, 0, 0, 0],
            [1, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_115, 0)
        ttnn.deallocate(ttnn_slice_115, False)
        ttnn_slice_116 = ttnn.slice(
            ttnn_slice_114,
            [1, 0, 0, 0],
            [2, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_116, 1)
        ttnn.deallocate(ttnn_slice_116, False)
        ttnn_slice_117 = ttnn.slice(
            ttnn_slice_114,
            [2, 0, 0, 0],
            [3, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_117, 2)
        ttnn.deallocate(ttnn_slice_117, False)
        ttnn_slice_118 = ttnn.slice(
            ttnn_slice_114,
            [3, 0, 0, 0],
            [4, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_118, 3)
        ttnn.deallocate(ttnn_slice_118, False)
        ttnn_slice_119 = ttnn.slice(
            ttnn_slice_114,
            [4, 0, 0, 0],
            [5, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_119, 4)
        ttnn.deallocate(ttnn_slice_119, False)
        ttnn_slice_120 = ttnn.slice(
            ttnn_slice_114,
            [5, 0, 0, 0],
            [6, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_120, 5)
        ttnn.deallocate(ttnn_slice_120, False)
        ttnn_slice_121 = ttnn.slice(
            ttnn_slice_114,
            [6, 0, 0, 0],
            [7, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_121, 6)
        ttnn.deallocate(ttnn_slice_121, False)
        ttnn_slice_122 = ttnn.slice(
            ttnn_slice_114,
            [7, 0, 0, 0],
            [8, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_122, 7)
        ttnn.deallocate(ttnn_slice_122, False)
        ttnn_slice_123 = ttnn.slice(
            ttnn_slice_114,
            [8, 0, 0, 0],
            [9, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_123, 8)
        ttnn.deallocate(ttnn_slice_123, False)
        ttnn_slice_124 = ttnn.slice(
            ttnn_slice_114,
            [9, 0, 0, 0],
            [10, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_124, 9)
        ttnn.deallocate(ttnn_slice_124, False)
        ttnn_slice_125 = ttnn.slice(
            ttnn_slice_114,
            [10, 0, 0, 0],
            [11, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_125, 10)
        ttnn.deallocate(ttnn_slice_125, False)
        ttnn_slice_126 = ttnn.slice(
            ttnn_slice_114,
            [11, 0, 0, 0],
            [12, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_126, 11)
        ttnn.deallocate(ttnn_slice_126, False)
        ttnn_slice_127 = ttnn.slice(
            ttnn_slice_114,
            [12, 0, 0, 0],
            [13, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_127, 12)
        ttnn.deallocate(ttnn_slice_127, False)
        ttnn_slice_128 = ttnn.slice(
            ttnn_slice_114,
            [13, 0, 0, 0],
            [14, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_128, 13)
        ttnn.deallocate(ttnn_slice_128, False)
        ttnn_slice_129 = ttnn.slice(
            ttnn_slice_114,
            [14, 0, 0, 0],
            [15, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_129, 14)
        ttnn.deallocate(ttnn_slice_129, False)
        ttnn_slice_130 = ttnn.slice(
            ttnn_slice_114,
            [15, 0, 0, 0],
            [16, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_130, 15)
        ttnn.deallocate(ttnn_slice_130, False)
        ttnn_slice_131 = ttnn.slice(
            ttnn_slice_114,
            [16, 0, 0, 0],
            [17, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_131, 16)
        ttnn.deallocate(ttnn_slice_131, False)
        ttnn_slice_132 = ttnn.slice(
            ttnn_slice_114,
            [17, 0, 0, 0],
            [18, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_132, 17)
        ttnn.deallocate(ttnn_slice_132, False)
        ttnn_slice_133 = ttnn.slice(
            ttnn_slice_114,
            [18, 0, 0, 0],
            [19, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_133, 18)
        ttnn.deallocate(ttnn_slice_133, False)
        ttnn_slice_134 = ttnn.slice(
            ttnn_slice_114,
            [19, 0, 0, 0],
            [20, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_134, 19)
        ttnn.deallocate(ttnn_slice_134, False)
        ttnn_slice_135 = ttnn.slice(
            ttnn_slice_114,
            [20, 0, 0, 0],
            [21, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_135, 20)
        ttnn.deallocate(ttnn_slice_135, False)
        ttnn_slice_136 = ttnn.slice(
            ttnn_slice_114,
            [21, 0, 0, 0],
            [22, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_136, 21)
        ttnn.deallocate(ttnn_slice_136, False)
        ttnn_slice_137 = ttnn.slice(
            ttnn_slice_114,
            [22, 0, 0, 0],
            [23, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_137, 22)
        ttnn.deallocate(ttnn_slice_137, False)
        ttnn_slice_138 = ttnn.slice(
            ttnn_slice_114,
            [23, 0, 0, 0],
            [24, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_138, 23)
        ttnn.deallocate(ttnn_slice_138, False)
        ttnn_slice_139 = ttnn.slice(
            ttnn_slice_114,
            [24, 0, 0, 0],
            [25, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_139, 24)
        ttnn.deallocate(ttnn_slice_139, False)
        ttnn_slice_140 = ttnn.slice(
            ttnn_slice_114,
            [25, 0, 0, 0],
            [26, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_140, 25)
        ttnn.deallocate(ttnn_slice_140, False)
        ttnn_slice_141 = ttnn.slice(
            ttnn_slice_114,
            [26, 0, 0, 0],
            [27, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_141, 26)
        ttnn.deallocate(ttnn_slice_141, False)
        ttnn_slice_142 = ttnn.slice(
            ttnn_slice_114,
            [27, 0, 0, 0],
            [28, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_142, 27)
        ttnn.deallocate(ttnn_slice_142, False)
        ttnn_slice_143 = ttnn.slice(
            ttnn_slice_114,
            [28, 0, 0, 0],
            [29, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_143, 28)
        ttnn.deallocate(ttnn_slice_143, False)
        ttnn_slice_144 = ttnn.slice(
            ttnn_slice_114,
            [29, 0, 0, 0],
            [30, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_144, 29)
        ttnn.deallocate(ttnn_slice_144, False)
        ttnn_slice_145 = ttnn.slice(
            ttnn_slice_114,
            [30, 0, 0, 0],
            [31, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_K, ttnn_slice_145, 30)
        ttnn.deallocate(ttnn_slice_145, False)
        ttnn_slice_146 = ttnn.slice(
            ttnn_slice_114,
            [31, 0, 0, 0],
            [32, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_slice_114, False)
        ttnn.fill_cache(args_K, ttnn_slice_146, 31)
        ttnn.deallocate(ttnn_slice_146, False)
        ttnn_slice_147 = ttnn.slice(
            v_6,
            [0, 0, 0, 0],
            [1, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_147, 0)
        ttnn.deallocate(ttnn_slice_147, False)
        ttnn_slice_148 = ttnn.slice(
            v_6,
            [1, 0, 0, 0],
            [2, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_148, 1)
        ttnn.deallocate(ttnn_slice_148, False)
        ttnn_slice_149 = ttnn.slice(
            v_6,
            [2, 0, 0, 0],
            [3, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_149, 2)
        ttnn.deallocate(ttnn_slice_149, False)
        ttnn_slice_150 = ttnn.slice(
            v_6,
            [3, 0, 0, 0],
            [4, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_150, 3)
        ttnn.deallocate(ttnn_slice_150, False)
        ttnn_slice_151 = ttnn.slice(
            v_6,
            [4, 0, 0, 0],
            [5, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_151, 4)
        ttnn.deallocate(ttnn_slice_151, False)
        ttnn_slice_152 = ttnn.slice(
            v_6,
            [5, 0, 0, 0],
            [6, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_152, 5)
        ttnn.deallocate(ttnn_slice_152, False)
        ttnn_slice_153 = ttnn.slice(
            v_6,
            [6, 0, 0, 0],
            [7, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_153, 6)
        ttnn.deallocate(ttnn_slice_153, False)
        ttnn_slice_154 = ttnn.slice(
            v_6,
            [7, 0, 0, 0],
            [8, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_154, 7)
        ttnn.deallocate(ttnn_slice_154, False)
        ttnn_slice_155 = ttnn.slice(
            v_6,
            [8, 0, 0, 0],
            [9, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_155, 8)
        ttnn.deallocate(ttnn_slice_155, False)
        ttnn_slice_156 = ttnn.slice(
            v_6,
            [9, 0, 0, 0],
            [10, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_156, 9)
        ttnn.deallocate(ttnn_slice_156, False)
        ttnn_slice_157 = ttnn.slice(
            v_6,
            [10, 0, 0, 0],
            [11, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_157, 10)
        ttnn.deallocate(ttnn_slice_157, False)
        ttnn_slice_158 = ttnn.slice(
            v_6,
            [11, 0, 0, 0],
            [12, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_158, 11)
        ttnn.deallocate(ttnn_slice_158, False)
        ttnn_slice_159 = ttnn.slice(
            v_6,
            [12, 0, 0, 0],
            [13, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_159, 12)
        ttnn.deallocate(ttnn_slice_159, False)
        ttnn_slice_160 = ttnn.slice(
            v_6,
            [13, 0, 0, 0],
            [14, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_160, 13)
        ttnn.deallocate(ttnn_slice_160, False)
        ttnn_slice_161 = ttnn.slice(
            v_6,
            [14, 0, 0, 0],
            [15, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_161, 14)
        ttnn.deallocate(ttnn_slice_161, False)
        ttnn_slice_162 = ttnn.slice(
            v_6,
            [15, 0, 0, 0],
            [16, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_162, 15)
        ttnn.deallocate(ttnn_slice_162, False)
        ttnn_slice_163 = ttnn.slice(
            v_6,
            [16, 0, 0, 0],
            [17, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_163, 16)
        ttnn.deallocate(ttnn_slice_163, False)
        ttnn_slice_164 = ttnn.slice(
            v_6,
            [17, 0, 0, 0],
            [18, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_164, 17)
        ttnn.deallocate(ttnn_slice_164, False)
        ttnn_slice_165 = ttnn.slice(
            v_6,
            [18, 0, 0, 0],
            [19, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_165, 18)
        ttnn.deallocate(ttnn_slice_165, False)
        ttnn_slice_166 = ttnn.slice(
            v_6,
            [19, 0, 0, 0],
            [20, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_166, 19)
        ttnn.deallocate(ttnn_slice_166, False)
        ttnn_slice_167 = ttnn.slice(
            v_6,
            [20, 0, 0, 0],
            [21, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_167, 20)
        ttnn.deallocate(ttnn_slice_167, False)
        ttnn_slice_168 = ttnn.slice(
            v_6,
            [21, 0, 0, 0],
            [22, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_168, 21)
        ttnn.deallocate(ttnn_slice_168, False)
        ttnn_slice_169 = ttnn.slice(
            v_6,
            [22, 0, 0, 0],
            [23, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_169, 22)
        ttnn.deallocate(ttnn_slice_169, False)
        ttnn_slice_170 = ttnn.slice(
            v_6,
            [23, 0, 0, 0],
            [24, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_170, 23)
        ttnn.deallocate(ttnn_slice_170, False)
        ttnn_slice_171 = ttnn.slice(
            v_6,
            [24, 0, 0, 0],
            [25, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_171, 24)
        ttnn.deallocate(ttnn_slice_171, False)
        ttnn_slice_172 = ttnn.slice(
            v_6,
            [25, 0, 0, 0],
            [26, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_172, 25)
        ttnn.deallocate(ttnn_slice_172, False)
        ttnn_slice_173 = ttnn.slice(
            v_6,
            [26, 0, 0, 0],
            [27, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_173, 26)
        ttnn.deallocate(ttnn_slice_173, False)
        ttnn_slice_174 = ttnn.slice(
            v_6,
            [27, 0, 0, 0],
            [28, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_174, 27)
        ttnn.deallocate(ttnn_slice_174, False)
        ttnn_slice_175 = ttnn.slice(
            v_6,
            [28, 0, 0, 0],
            [29, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_175, 28)
        ttnn.deallocate(ttnn_slice_175, False)
        ttnn_slice_176 = ttnn.slice(
            v_6,
            [29, 0, 0, 0],
            [30, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_176, 29)
        ttnn.deallocate(ttnn_slice_176, False)
        ttnn_slice_177 = ttnn.slice(
            v_6,
            [30, 0, 0, 0],
            [31, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.fill_cache(args_V, ttnn_slice_177, 30)
        ttnn.deallocate(ttnn_slice_177, False)
        ttnn_slice_178 = ttnn.slice(
            v_6,
            [31, 0, 0, 0],
            [32, 8, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(v_6, False)
        ttnn.fill_cache(args_V, ttnn_slice_178, 31)
        ttnn.deallocate(ttnn_slice_178, False)
        ttnn_to_memory_config_19 = ttnn.to_memory_config(
            var_0,
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
        ttnn_add_4 = ttnn.add(
            args_attn,
            ttnn_to_memory_config_19,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(
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
        ttnn.deallocate(ttnn_to_memory_config_19, False)
        ttnn.deallocate(args_attn, False)
        ttnn_to_memory_config_20 = ttnn.to_memory_config(
            ttnn_add_4,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_4, False)
        ttnn_to_memory_config_21 = ttnn.to_memory_config(
            sin,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        if self.is_last:
            ttnn.deallocate(sin, False)
        ttnn_to_memory_config_22 = ttnn.to_memory_config(
            cos,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        if self.is_last:
            ttnn.deallocate(cos, False)
        ttnn_experimental_rotary_embedding_3 = ttnn.experimental.rotary_embedding(
            v_4,
            ttnn_to_memory_config_22,
            ttnn_to_memory_config_21,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_to_memory_config_22, False)
        ttnn.deallocate(ttnn_to_memory_config_21, False)
        ttnn.deallocate(v_4, False)
        ttnn_slice_179 = ttnn.slice(
            ttnn_experimental_rotary_embedding_3,
            [0, 0, 0, 0],
            [32, 32, 18, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_experimental_rotary_embedding_3, False)
        ttnn_to_memory_config_23 = ttnn.to_memory_config(
            causal_mask,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        if self.is_last:
            ttnn.deallocate(causal_mask, False)
        ttnn_transformer_scaled_dot_product_attention_1 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_slice_179,
                args_K,
                args_V,
                attn_mask=ttnn_to_memory_config_23,
                is_causal=False,
                scale=0.1249999925494194,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        )
        ttnn.deallocate(ttnn_to_memory_config_23, False)
        ttnn.deallocate(ttnn_slice_179, False)
        ttnn_transformer_concatenate_heads_1 = ttnn.transformer.concatenate_heads(
            ttnn_transformer_scaled_dot_product_attention_1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_1, False)
        ttnn_reshape_7 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_1,
            [576, 2048],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
            ),
        )
        ttnn.deallocate(ttnn_transformer_concatenate_heads_1, False)
        ttnn_matmul_7 = ttnn.matmul(
            ttnn_reshape_7,
            self.weights["model.layers." + str(self.layer_idx) + ".self_attn.o_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 192],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=6,
                out_block_h=2,
                out_block_w=6,
                per_core_M=2,
                per_core_N=6,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            ),
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_reshape_7, False)
        ttnn_add_5 = ttnn.add(
            ttnn_matmul_7,
            hidden_states,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 192],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_matmul_7, False)
        ttnn.deallocate(hidden_states, False)
        ttnn_rms_norm_3 = ttnn.rms_norm(
            ttnn_add_5,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["model.layers." + str(self.layer_idx) + ".post_attention_layernorm.weight"],
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 192],
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
        ttnn_matmul_8 = ttnn.matmul(
            ttnn_rms_norm_3,
            self.weights["model.layers." + str(self.layer_idx) + ".mlp.gate_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 768],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=8,
                out_block_h=2,
                out_block_w=24,
                per_core_M=2,
                per_core_N=24,
                transpose_mcast=False,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
                fuse_batch=True,
            ),
            activation=None,
            compute_kernel_config=None,
        )
        ttnn_matmul_9 = ttnn.matmul(
            ttnn_rms_norm_3,
            self.weights["model.layers." + str(self.layer_idx) + ".mlp.up_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 768],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=8,
                out_block_h=2,
                out_block_w=24,
                per_core_M=2,
                per_core_N=24,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
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
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 768],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_matmul_9, False)
        ttnn.deallocate(ttnn_matmul_8, False)
        ttnn_matmul_10 = ttnn.matmul(
            ttnn_multiply_1,
            self.weights["model.layers." + str(self.layer_idx) + ".mlp.down_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 192],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=6,
                out_block_h=2,
                out_block_w=6,
                per_core_M=2,
                per_core_N=6,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            ),
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_1, False)
        ttnn_add_6 = ttnn.add(
            ttnn_matmul_10,
            ttnn_add_5,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                    ),
                    [64, 192],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_matmul_10, False)
        ttnn.deallocate(ttnn_add_5, False)
        return ttnn_add_6, ttnn_to_memory_config_20
