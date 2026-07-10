import ttnn
import params
import consteval


class LightweightModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ModelTTNN(LightweightModule):
    def __init__(self, device):
        self.device = device
        self.weights = params.load_weights_for__main(device)
        self.weights = consteval.run_consteval(self.weights, device)
        self.transformer_blocks = [BriaFiboTransformerBlock(device, self.weights, i) for i in range(8)]
        self.single_transformer_blocks = [BriaFiboSingleTransformerBlock(device, self.weights, i) for i in range(38)]

    def forward(self, activations):
        timestep = activations[0]
        hidden_states = activations[1]
        encoder_hidden_states = activations[3]
        attention_mask = activations[4]
        img_ids = activations[5]
        txt_ids = activations[6]
        # 46 per-layer text-encoder features (one per transformer block)
        text_encoder_layers = [activations[2]] + [activations[i] for i in range(7, 52)]
        var_0 = self.weights["consteval.const_233"]
        var_1 = self.weights["consteval.const_238"]
        ttnn_to_layout_584 = ttnn.to_layout(
            encoder_hidden_states,
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_180 = ttnn.reshape(
            ttnn_to_layout_584,
            [90, 4096],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_584, False)
        ttnn_linear_0 = ttnn.linear(
            ttnn_reshape_180,
            self.weights["transformer.context_embedder.weight"],
            bias=self.weights["transformer.context_embedder.bias"],
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
        ttnn.deallocate(ttnn_reshape_180, False)
        ttnn_slice_0 = ttnn.slice(
            ttnn_linear_0,
            [0, 0],
            [90, 1536],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_linear_0, False)
        ttnn_reshape_181 = ttnn.reshape(
            ttnn_slice_0,
            [2, 45, 1536],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_0, False)
        ttnn_to_layout_585 = ttnn.to_layout(
            text_encoder_layers[0],
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_182 = ttnn.reshape(
            ttnn_to_layout_585,
            [90, 2048],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_585, False)
        ttnn_matmul_0 = ttnn.matmul(
            ttnn_reshape_182,
            self.weights["transformer.caption_projection.0.linear.weight"],
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
        ttnn.deallocate(ttnn_reshape_182, False)
        ttnn_reshape_183 = ttnn.reshape(
            ttnn_matmul_0,
            [2, 45, 1536],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_0, False)
        ttnn_concat_109 = ttnn.concat(
            [ttnn_reshape_181, ttnn_reshape_183],
            2,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_183, False)
        ttnn.deallocate(ttnn_reshape_181, False)
        ttnn_layer_norm_0 = ttnn.layer_norm(
            ttnn_concat_109,
            epsilon=9.9999999747524271e-07,
            weight=None,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_to_layout_586 = ttnn.to_layout(
            timestep,
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_114 = ttnn.typecast(
            ttnn_to_layout_586,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_586, False)
        ttnn_reshape_184 = ttnn.reshape(
            ttnn_typecast_114,
            [2, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_114, False)
        ttnn_multiply_0 = ttnn.multiply(
            ttnn_reshape_184,
            self.weights["consteval.const_0"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_184, False)
        ttnn_sin_0 = ttnn.sin(
            ttnn_multiply_0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_cos_0 = ttnn.cos(
            ttnn_multiply_0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_0, False)
        ttnn_concat_110 = ttnn.concat(
            [ttnn_cos_0, ttnn_sin_0],
            1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_cos_0, False)
        ttnn.deallocate(ttnn_sin_0, False)
        ttnn_linear_1 = ttnn.linear(
            ttnn_concat_110,
            self.weights["transformer.time_embed.timestep_embedder.linear_1.weight.fused.transformer.time_embed.timestep_embedder.linear_1.weight"],
            bias=self.weights["transformer.time_embed.timestep_embedder.linear_1.bias.fused.transformer.time_embed.timestep_embedder.linear_1.bias"],
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation="silu",
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_concat_110, False)
        ttnn_linear_2 = ttnn.linear(
            ttnn_linear_1,
            self.weights["transformer.time_embed.timestep_embedder.linear_2.weight.fused.transformer.time_embed.timestep_embedder.linear_2.weight"],
            bias=self.weights["transformer.time_embed.timestep_embedder.linear_2.bias.fused.transformer.time_embed.timestep_embedder.linear_2.bias"],
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation="silu",
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_linear_1, False)
        ttnn_matmul_1 = ttnn.matmul(
            ttnn_linear_2,
            self.weights["transformer.fused_norm_out_single_transformer_blocks_37_norm_36_35_34_33_32_31_30_29_28_27_26_25_24_23_22_21_20_19_18_17_16_15_14_13_12_11_10_9_8_7_6_5_4_3_2_1_0_transformer_blocks_norm1_norm1_context.linear.weight"],
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
        ttnn.deallocate(ttnn_linear_2, False)
        ttnn_slice_1 = ttnn.slice(
            ttnn_matmul_1,
            [0, 0],
            [2, 6144],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_2 = ttnn.slice(
            ttnn_matmul_1,
            [0, 6144],
            [2, 15360],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_3 = ttnn.slice(
            ttnn_matmul_1,
            [0, 15360],
            [2, 24576],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_4 = ttnn.slice(
            ttnn_matmul_1,
            [0, 24576],
            [2, 33792],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_5 = ttnn.slice(
            ttnn_matmul_1,
            [0, 33792],
            [2, 43008],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_6 = ttnn.slice(
            ttnn_matmul_1,
            [0, 43008],
            [2, 52224],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_7 = ttnn.slice(
            ttnn_matmul_1,
            [0, 52224],
            [2, 61440],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_8 = ttnn.slice(
            ttnn_matmul_1,
            [0, 61440],
            [2, 70656],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_9 = ttnn.slice(
            ttnn_matmul_1,
            [0, 70656],
            [2, 79872],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_10 = ttnn.slice(
            ttnn_matmul_1,
            [0, 79872],
            [2, 89088],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_11 = ttnn.slice(
            ttnn_matmul_1,
            [0, 89088],
            [2, 98304],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_12 = ttnn.slice(
            ttnn_matmul_1,
            [0, 98304],
            [2, 107520],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_13 = ttnn.slice(
            ttnn_matmul_1,
            [0, 107520],
            [2, 116736],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_14 = ttnn.slice(
            ttnn_matmul_1,
            [0, 116736],
            [2, 125952],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_15 = ttnn.slice(
            ttnn_matmul_1,
            [0, 125952],
            [2, 135168],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_16 = ttnn.slice(
            ttnn_matmul_1,
            [0, 135168],
            [2, 144384],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_17 = ttnn.slice(
            ttnn_matmul_1,
            [0, 144384],
            [2, 153600],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_18 = ttnn.slice(
            ttnn_matmul_1,
            [0, 153600],
            [2, 162816],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_19 = ttnn.slice(
            ttnn_matmul_1,
            [0, 162816],
            [2, 172032],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_20 = ttnn.slice(
            ttnn_matmul_1,
            [0, 172032],
            [2, 181248],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_21 = ttnn.slice(
            ttnn_matmul_1,
            [0, 181248],
            [2, 190464],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_22 = ttnn.slice(
            ttnn_matmul_1,
            [0, 190464],
            [2, 199680],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_23 = ttnn.slice(
            ttnn_matmul_1,
            [0, 199680],
            [2, 208896],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_24 = ttnn.slice(
            ttnn_matmul_1,
            [0, 208896],
            [2, 218112],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_25 = ttnn.slice(
            ttnn_matmul_1,
            [0, 218112],
            [2, 227328],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_26 = ttnn.slice(
            ttnn_matmul_1,
            [0, 227328],
            [2, 236544],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_27 = ttnn.slice(
            ttnn_matmul_1,
            [0, 236544],
            [2, 245760],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_28 = ttnn.slice(
            ttnn_matmul_1,
            [0, 245760],
            [2, 254976],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_29 = ttnn.slice(
            ttnn_matmul_1,
            [0, 254976],
            [2, 264192],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_30 = ttnn.slice(
            ttnn_matmul_1,
            [0, 264192],
            [2, 273408],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_31 = ttnn.slice(
            ttnn_matmul_1,
            [0, 273408],
            [2, 282624],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_32 = ttnn.slice(
            ttnn_matmul_1,
            [0, 282624],
            [2, 291840],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_33 = ttnn.slice(
            ttnn_matmul_1,
            [0, 291840],
            [2, 301056],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_34 = ttnn.slice(
            ttnn_matmul_1,
            [0, 301056],
            [2, 310272],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_35 = ttnn.slice(
            ttnn_matmul_1,
            [0, 310272],
            [2, 319488],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_36 = ttnn.slice(
            ttnn_matmul_1,
            [0, 319488],
            [2, 328704],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_37 = ttnn.slice(
            ttnn_matmul_1,
            [0, 328704],
            [2, 337920],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_38 = ttnn.slice(
            ttnn_matmul_1,
            [0, 337920],
            [2, 347136],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_39 = ttnn.slice(
            ttnn_matmul_1,
            [0, 347136],
            [2, 356352],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_40 = ttnn.slice(
            ttnn_matmul_1,
            [0, 356352],
            [2, 374784],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_41 = ttnn.slice(
            ttnn_matmul_1,
            [0, 374784],
            [2, 393216],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_42 = ttnn.slice(
            ttnn_matmul_1,
            [0, 393216],
            [2, 411648],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_43 = ttnn.slice(
            ttnn_matmul_1,
            [0, 411648],
            [2, 430080],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_44 = ttnn.slice(
            ttnn_matmul_1,
            [0, 430080],
            [2, 448512],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_45 = ttnn.slice(
            ttnn_matmul_1,
            [0, 448512],
            [2, 466944],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_46 = ttnn.slice(
            ttnn_matmul_1,
            [0, 466944],
            [2, 485376],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_47 = ttnn.slice(
            ttnn_matmul_1,
            [0, 485376],
            [2, 503808],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_48 = ttnn.slice(
            ttnn_matmul_1,
            [0, 503808],
            [2, 522240],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_49 = ttnn.slice(
            ttnn_matmul_1,
            [0, 522240],
            [2, 540672],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_50 = ttnn.slice(
            ttnn_matmul_1,
            [0, 540672],
            [2, 559104],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_51 = ttnn.slice(
            ttnn_matmul_1,
            [0, 559104],
            [2, 577536],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_52 = ttnn.slice(
            ttnn_matmul_1,
            [0, 577536],
            [2, 595968],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_53 = ttnn.slice(
            ttnn_matmul_1,
            [0, 595968],
            [2, 614400],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_54 = ttnn.slice(
            ttnn_matmul_1,
            [0, 614400],
            [2, 632832],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_55 = ttnn.slice(
            ttnn_matmul_1,
            [0, 632832],
            [2, 651264],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_1, False)
        ttnn_reshape_193, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_227, ttnn_slice_78, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_0, ttnn_typecast_116 = self.transformer_blocks[0](hidden_states, txt_ids, img_ids, attention_mask, text_encoder_layers[1], ttnn_concat_109, ttnn_layer_norm_0, ttnn_slice_54, ttnn_slice_55, var_0, var_1)
        ttnn_add_22, ttnn_reshape_273, ttnn_slice_103, ttnn_transformer_concatenate_heads_1, ttnn_typecast_125 = self.transformer_blocks[1](ttnn_reshape_227, ttnn_slice_78, ttnn_slice_53, var_1, ttnn_typecast_116, ttnn_transformer_concatenate_heads_0, ttnn_reshape_193, ttnn_slice_52, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[2])
        ttnn_add_42, ttnn_reshape_319, ttnn_slice_128, ttnn_transformer_concatenate_heads_2, ttnn_typecast_133 = self.transformer_blocks[2](ttnn_reshape_273, ttnn_slice_103, ttnn_slice_51, var_1, ttnn_typecast_125, ttnn_transformer_concatenate_heads_1, ttnn_add_22, ttnn_slice_50, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[3])
        ttnn_add_62, ttnn_reshape_365, ttnn_slice_153, ttnn_transformer_concatenate_heads_3, ttnn_typecast_141 = self.transformer_blocks[3](ttnn_reshape_319, ttnn_slice_128, ttnn_slice_49, var_1, ttnn_typecast_133, ttnn_transformer_concatenate_heads_2, ttnn_add_42, ttnn_slice_48, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[4])
        ttnn_add_82, ttnn_reshape_411, ttnn_slice_178, ttnn_transformer_concatenate_heads_4, ttnn_typecast_149 = self.transformer_blocks[4](ttnn_reshape_365, ttnn_slice_153, ttnn_slice_47, var_1, ttnn_typecast_141, ttnn_transformer_concatenate_heads_3, ttnn_add_62, ttnn_slice_46, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[5])
        ttnn_add_102, ttnn_reshape_457, ttnn_slice_203, ttnn_transformer_concatenate_heads_5, ttnn_typecast_157 = self.transformer_blocks[5](ttnn_reshape_411, ttnn_slice_178, ttnn_slice_45, var_1, ttnn_typecast_149, ttnn_transformer_concatenate_heads_4, ttnn_add_82, ttnn_slice_44, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[6])
        ttnn_add_122, ttnn_reshape_503, ttnn_slice_228, ttnn_transformer_concatenate_heads_6, ttnn_typecast_165 = self.transformer_blocks[6](ttnn_reshape_457, ttnn_slice_203, ttnn_slice_43, var_1, ttnn_typecast_157, ttnn_transformer_concatenate_heads_5, ttnn_add_102, ttnn_slice_42, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[7])
        ttnn_add_142, ttnn_reshape_549, ttnn_slice_253, ttnn_transformer_concatenate_heads_7, ttnn_typecast_173 = self.transformer_blocks[7](ttnn_reshape_503, ttnn_slice_228, ttnn_slice_41, var_1, ttnn_typecast_165, ttnn_transformer_concatenate_heads_6, ttnn_add_122, ttnn_slice_40, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[8])
        ttnn_add_166, ttnn_reshape_581, ttnn_slice_270 = self.single_transformer_blocks[0](text_encoder_layers[9], ttnn_add_142, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_549, ttnn_slice_253, ttnn_slice_39, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_7, ttnn_typecast_173, var_1)
        ttnn_add_173, ttnn_reshape_601, ttnn_slice_283 = self.single_transformer_blocks[1](ttnn_reshape_581, ttnn_slice_270, ttnn_add_166, ttnn_slice_38, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[10])
        ttnn_add_180, ttnn_reshape_621, ttnn_slice_296 = self.single_transformer_blocks[2](ttnn_reshape_601, ttnn_slice_283, ttnn_add_173, ttnn_slice_37, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[11])
        ttnn_add_187, ttnn_reshape_641, ttnn_slice_309 = self.single_transformer_blocks[3](ttnn_reshape_621, ttnn_slice_296, ttnn_add_180, ttnn_slice_36, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[12])
        ttnn_add_194, ttnn_reshape_661, ttnn_slice_322 = self.single_transformer_blocks[4](ttnn_reshape_641, ttnn_slice_309, ttnn_add_187, ttnn_slice_35, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[13])
        ttnn_add_201, ttnn_reshape_681, ttnn_slice_335 = self.single_transformer_blocks[5](ttnn_reshape_661, ttnn_slice_322, ttnn_add_194, ttnn_slice_34, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[14])
        ttnn_add_208, ttnn_reshape_701, ttnn_slice_348 = self.single_transformer_blocks[6](ttnn_reshape_681, ttnn_slice_335, ttnn_add_201, ttnn_slice_33, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[15])
        ttnn_add_215, ttnn_reshape_721, ttnn_slice_361 = self.single_transformer_blocks[7](ttnn_reshape_701, ttnn_slice_348, ttnn_add_208, ttnn_slice_32, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[16])
        ttnn_add_222, ttnn_reshape_741, ttnn_slice_374 = self.single_transformer_blocks[8](ttnn_reshape_721, ttnn_slice_361, ttnn_add_215, ttnn_slice_31, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[17])
        ttnn_add_229, ttnn_reshape_761, ttnn_slice_387 = self.single_transformer_blocks[9](ttnn_reshape_741, ttnn_slice_374, ttnn_add_222, ttnn_slice_30, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[18])
        ttnn_add_236, ttnn_reshape_781, ttnn_slice_400 = self.single_transformer_blocks[10](ttnn_reshape_761, ttnn_slice_387, ttnn_add_229, ttnn_slice_29, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[19])
        ttnn_add_243, ttnn_reshape_801, ttnn_slice_413 = self.single_transformer_blocks[11](ttnn_reshape_781, ttnn_slice_400, ttnn_add_236, ttnn_slice_28, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[20])
        ttnn_add_250, ttnn_reshape_821, ttnn_slice_426 = self.single_transformer_blocks[12](ttnn_reshape_801, ttnn_slice_413, ttnn_add_243, ttnn_slice_27, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[21])
        ttnn_add_257, ttnn_reshape_841, ttnn_slice_439 = self.single_transformer_blocks[13](ttnn_reshape_821, ttnn_slice_426, ttnn_add_250, ttnn_slice_26, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[22])
        ttnn_add_264, ttnn_reshape_861, ttnn_slice_452 = self.single_transformer_blocks[14](ttnn_reshape_841, ttnn_slice_439, ttnn_add_257, ttnn_slice_25, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[23])
        ttnn_add_271, ttnn_reshape_881, ttnn_slice_465 = self.single_transformer_blocks[15](ttnn_reshape_861, ttnn_slice_452, ttnn_add_264, ttnn_slice_24, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[24])
        ttnn_add_278, ttnn_reshape_901, ttnn_slice_478 = self.single_transformer_blocks[16](ttnn_reshape_881, ttnn_slice_465, ttnn_add_271, ttnn_slice_23, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[25])
        ttnn_add_285, ttnn_reshape_921, ttnn_slice_491 = self.single_transformer_blocks[17](ttnn_reshape_901, ttnn_slice_478, ttnn_add_278, ttnn_slice_22, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[26])
        ttnn_add_292, ttnn_reshape_941, ttnn_slice_504 = self.single_transformer_blocks[18](ttnn_reshape_921, ttnn_slice_491, ttnn_add_285, ttnn_slice_21, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[27])
        ttnn_add_299, ttnn_reshape_961, ttnn_slice_517 = self.single_transformer_blocks[19](ttnn_reshape_941, ttnn_slice_504, ttnn_add_292, ttnn_slice_20, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[28])
        ttnn_add_306, ttnn_reshape_981, ttnn_slice_530 = self.single_transformer_blocks[20](ttnn_reshape_961, ttnn_slice_517, ttnn_add_299, ttnn_slice_19, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[29])
        ttnn_add_313, ttnn_reshape_1001, ttnn_slice_543 = self.single_transformer_blocks[21](ttnn_reshape_981, ttnn_slice_530, ttnn_add_306, ttnn_slice_18, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[30])
        ttnn_add_320, ttnn_reshape_1021, ttnn_slice_556 = self.single_transformer_blocks[22](ttnn_reshape_1001, ttnn_slice_543, ttnn_add_313, ttnn_slice_17, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[31])
        ttnn_add_327, ttnn_reshape_1041, ttnn_slice_569 = self.single_transformer_blocks[23](ttnn_reshape_1021, ttnn_slice_556, ttnn_add_320, ttnn_slice_16, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[32])
        ttnn_add_334, ttnn_reshape_1061, ttnn_slice_582 = self.single_transformer_blocks[24](ttnn_reshape_1041, ttnn_slice_569, ttnn_add_327, ttnn_slice_15, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[33])
        ttnn_add_341, ttnn_reshape_1081, ttnn_slice_595 = self.single_transformer_blocks[25](ttnn_reshape_1061, ttnn_slice_582, ttnn_add_334, ttnn_slice_14, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[34])
        ttnn_add_348, ttnn_reshape_1101, ttnn_slice_608 = self.single_transformer_blocks[26](ttnn_reshape_1081, ttnn_slice_595, ttnn_add_341, ttnn_slice_13, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[35])
        ttnn_add_355, ttnn_reshape_1121, ttnn_slice_621 = self.single_transformer_blocks[27](ttnn_reshape_1101, ttnn_slice_608, ttnn_add_348, ttnn_slice_12, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[36])
        ttnn_add_362, ttnn_reshape_1141, ttnn_slice_634 = self.single_transformer_blocks[28](ttnn_reshape_1121, ttnn_slice_621, ttnn_add_355, ttnn_slice_11, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[37])
        ttnn_add_369, ttnn_reshape_1161, ttnn_slice_647 = self.single_transformer_blocks[29](ttnn_reshape_1141, ttnn_slice_634, ttnn_add_362, ttnn_slice_10, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[38])
        ttnn_add_376, ttnn_reshape_1181, ttnn_slice_660 = self.single_transformer_blocks[30](ttnn_reshape_1161, ttnn_slice_647, ttnn_add_369, ttnn_slice_9, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[39])
        ttnn_add_383, ttnn_reshape_1201, ttnn_slice_673 = self.single_transformer_blocks[31](ttnn_reshape_1181, ttnn_slice_660, ttnn_add_376, ttnn_slice_8, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[40])
        ttnn_add_390, ttnn_reshape_1221, ttnn_slice_686 = self.single_transformer_blocks[32](ttnn_reshape_1201, ttnn_slice_673, ttnn_add_383, ttnn_slice_7, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[41])
        ttnn_add_397, ttnn_reshape_1241, ttnn_slice_699 = self.single_transformer_blocks[33](ttnn_reshape_1221, ttnn_slice_686, ttnn_add_390, ttnn_slice_6, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[42])
        ttnn_add_404, ttnn_reshape_1261, ttnn_slice_712 = self.single_transformer_blocks[34](ttnn_reshape_1241, ttnn_slice_699, ttnn_add_397, ttnn_slice_5, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[43])
        ttnn_add_411, ttnn_reshape_1281, ttnn_slice_725 = self.single_transformer_blocks[35](ttnn_reshape_1261, ttnn_slice_712, ttnn_add_404, ttnn_slice_4, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[44])
        ttnn_add_418, ttnn_reshape_1301, ttnn_slice_738 = self.single_transformer_blocks[36](ttnn_reshape_1281, ttnn_slice_725, ttnn_add_411, ttnn_slice_3, var_1, ttnn_reshape_203, ttnn_reshape_209, ttnn_to_layout_590, text_encoder_layers[45])
        ttnn_layer_norm_70 = self.single_transformer_blocks[37](ttnn_add_418, ttnn_reshape_1301, ttnn_reshape_203, ttnn_reshape_209, ttnn_slice_2, ttnn_slice_738, ttnn_to_layout_590, var_1)
        ttnn_reshape_1321 = ttnn.reshape(
            ttnn_slice_1,
            [1, 1, 2, 6144],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_1, False)
        ttnn_reduce_scatter_124 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_1321,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_reshape_1321, False)
        ttnn_reshape_1322 = ttnn.reshape(
            ttnn_reduce_scatter_124,
            [2, 1536],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reduce_scatter_124, False)
        ttnn_all_gather_200 = ttnn.all_gather(
            input_tensor=ttnn_reshape_1322,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_1322, False)
        ttnn_add_426 = ttnn.add(
            ttnn_all_gather_200,
            self.weights["transformer.norm_out.linear.bias.f32"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_all_gather_200, False)
        ttnn_typecast_446 = ttnn.typecast(
            ttnn_add_426,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_426, False)
        ttnn_slice_752 = ttnn.slice(
            ttnn_typecast_446,
            [0, 0],
            [2, 3072],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_1323 = ttnn.reshape(
            ttnn_slice_752,
            [2, 1, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_752, False)
        ttnn_add_427 = ttnn.add(
            ttnn_reshape_1323,
            var_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_1323, False)
        ttnn_multiply_328 = ttnn.multiply(
            ttnn_layer_norm_70,
            ttnn_add_427,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_427, False)
        ttnn.deallocate(ttnn_layer_norm_70, False)
        ttnn_slice_753 = ttnn.slice(
            ttnn_typecast_446,
            [0, 3072],
            [2, 6144],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_446, False)
        ttnn_reshape_1324 = ttnn.reshape(
            ttnn_slice_753,
            [2, 1, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_753, False)
        ttnn_add_428 = ttnn.add(
            ttnn_multiply_328,
            ttnn_reshape_1324,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_1324, False)
        ttnn.deallocate(ttnn_multiply_328, False)
        ttnn_reshape_1325 = ttnn.reshape(
            ttnn_add_428,
            [8192, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_428, False)
        ttnn_linear_74 = ttnn.linear(
            ttnn_reshape_1325,
            self.weights["transformer.proj_out.weight"],
            bias=self.weights["transformer.proj_out.bias"],
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
        ttnn.deallocate(ttnn_reshape_1325, False)
        ttnn_reshape_1326 = ttnn.reshape(
            ttnn_linear_74,
            [2, 4096, 48],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_linear_74, False)
        return [ttnn_reshape_1326]


class BriaFiboTransformerBlock(LightweightModule):
    def __init__(self, device, weights, block_idx):
        self.device = device
        self.weights = weights
        self.block_idx = block_idx

    def forward(self, *args):
        # Blocks 1..7 are identical up to the block index and per-block inputs, so
        # they share this implementation (indexed by self.block_idx). Block 0 is the
        # first block (no previous-block tail, no caption projection) and stays
        # separate.
        idx = self.block_idx
        if idx == 0:
            return _tb_forward_0(self, *args)
        prev = idx - 1
        r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11 = args
        ttnn_matmul_19 = ttnn.matmul(
            r0,
            self.weights[f"transformer.caption_projection.{idx}.linear.weight"],
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
        ttnn.deallocate(r0, False)
        ttnn_reshape_366 = ttnn.reshape(
            ttnn_matmul_19,
            [2, 45, 1536],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_19, False)
        ttnn_concat_137 = ttnn.concat(
            [r1, ttnn_reshape_366],
            2,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_366, False)
        ttnn.deallocate(r1, False)
        ttnn_layer_norm_15 = ttnn.layer_norm(
            ttnn_concat_137,
            epsilon=9.9999999747524271e-07,
            weight=None,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_reshape_367 = ttnn.reshape(
            r2,
            [1, 1, 2, 18432],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(r2, False)
        ttnn_reduce_scatter_22 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_367,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_reshape_367, False)
        ttnn_reshape_368 = ttnn.reshape(
            ttnn_reduce_scatter_22,
            [2, 4608],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reduce_scatter_22, False)
        ttnn_all_gather_22 = ttnn.all_gather(
            input_tensor=ttnn_reshape_368,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_368, False)
        ttnn_add_74 = ttnn.add(
            ttnn_all_gather_22,
            self.weights[f"transformer.transformer_blocks.{idx}.norm1_context.linear.bias.f32"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_all_gather_22, False)
        ttnn_typecast_148 = ttnn.typecast(
            ttnn_add_74,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_74, False)
        ttnn_slice_154 = ttnn.slice(
            ttnn_typecast_148,
            [0, 6144],
            [2, 9216],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_369 = ttnn.reshape(
            ttnn_slice_154,
            [2, 1, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_154, False)
        ttnn_slice_155 = ttnn.slice(
            ttnn_typecast_148,
            [0, 3072],
            [2, 6144],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_370 = ttnn.reshape(
            ttnn_slice_155,
            [2, 1, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_155, False)
        ttnn_add_75 = ttnn.add(
            ttnn_reshape_370,
            r3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_370, False)
        ttnn_multiply_49 = ttnn.multiply(
            ttnn_layer_norm_15,
            ttnn_add_75,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_75, False)
        ttnn.deallocate(ttnn_layer_norm_15, False)
        ttnn_slice_156 = ttnn.slice(
            ttnn_typecast_148,
            [0, 0],
            [2, 3072],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_371 = ttnn.reshape(
            ttnn_slice_156,
            [2, 1, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_156, False)
        ttnn_add_76 = ttnn.add(
            ttnn_multiply_49,
            ttnn_reshape_371,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_371, False)
        ttnn.deallocate(ttnn_multiply_49, False)
        ttnn_reshape_372 = ttnn.reshape(
            ttnn_add_76,
            [90, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_76, False)
        ttnn_linear_19 = ttnn.linear(
            ttnn_reshape_372,
            self.weights[f"transformer.transformer_blocks.{idx}.attn.fused_add_q_proj_add_k_proj_add_v_proj.weight"],
            bias=self.weights[f"transformer.transformer_blocks.{idx}.attn.fused_add_q_proj_add_k_proj_add_v_proj.bias"],
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
        ttnn.deallocate(ttnn_reshape_372, False)
        ttnn_slice_157 = ttnn.slice(
            ttnn_linear_19,
            [0, 0],
            [90, 768],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_158 = ttnn.slice(
            ttnn_linear_19,
            [0, 768],
            [90, 1536],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_159 = ttnn.slice(
            ttnn_linear_19,
            [0, 1536],
            [90, 2304],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_linear_19, False)
        ttnn_reshape_373 = ttnn.reshape(
            ttnn_slice_157,
            [2, 45, 6, 128],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_157, False)
        ttnn_rms_norm_16 = ttnn.rms_norm(
            ttnn_reshape_373,
            epsilon=9.9999999747524271e-07,
            weight=self.weights[f"transformer.transformer_blocks.{idx}.attn.norm_added_q.weight"],
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
        ttnn.deallocate(ttnn_reshape_373, False)
        ttnn_slice_160 = ttnn.slice(
            r4,
            [0, 6144],
            [2, 9216],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_374 = ttnn.reshape(
            ttnn_slice_160,
            [2, 1, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_160, False)
        ttnn_slice_161 = ttnn.slice(
            r5,
            [0, 45, 0],
            [2, 4141, 768],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(r5, False)
        ttnn_reshape_375 = ttnn.reshape(
            ttnn_slice_161,
            [8192, 768],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_161, False)
        ttnn_matmul_20 = ttnn.matmul(
            ttnn_reshape_375,
            self.weights[f"transformer.transformer_blocks.{prev}.attn.to_out.0.weight"],
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
        ttnn.deallocate(ttnn_reshape_375, False)
        ttnn_reshape_376 = ttnn.reshape(
            ttnn_matmul_20,
            [1, 1, 8192, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_20, False)
        ttnn_reduce_scatter_23 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_376,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_reshape_376, False)
        ttnn_reshape_377 = ttnn.reshape(
            ttnn_reduce_scatter_23,
            [8192, 768],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reduce_scatter_23, False)
        ttnn_all_gather_23 = ttnn.all_gather(
            input_tensor=ttnn_reshape_377,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_377, False)
        ttnn_add_77 = ttnn.add(
            ttnn_all_gather_23,
            self.weights[f"transformer.transformer_blocks.{prev}.attn.to_out.0.bias.reshaped"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_all_gather_23, False)
        ttnn_reshape_378 = ttnn.reshape(
            ttnn_add_77,
            [2, 4096, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_77, False)
        ttnn_multiply_50 = ttnn.multiply(
            ttnn_reshape_374,
            ttnn_reshape_378,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_378, False)
        ttnn.deallocate(ttnn_reshape_374, False)
        ttnn_add_78 = ttnn.add(
            r6,
            ttnn_multiply_50,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_50, False)
        ttnn.deallocate(r6, False)
        ttnn_layer_norm_16 = ttnn.layer_norm(
            ttnn_add_78,
            epsilon=9.9999999747524271e-07,
            weight=None,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_slice_162 = ttnn.slice(
            r4,
            [0, 15360],
            [2, 18432],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_379 = ttnn.reshape(
            ttnn_slice_162,
            [2, 1, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_162, False)
        ttnn_slice_163 = ttnn.slice(
            r4,
            [0, 12288],
            [2, 15360],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_380 = ttnn.reshape(
            ttnn_slice_163,
            [2, 1, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_163, False)
        ttnn_add_79 = ttnn.add(
            ttnn_reshape_380,
            r3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_380, False)
        ttnn_multiply_51 = ttnn.multiply(
            ttnn_layer_norm_16,
            ttnn_add_79,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_79, False)
        ttnn.deallocate(ttnn_layer_norm_16, False)
        ttnn_slice_164 = ttnn.slice(
            r4,
            [0, 9216],
            [2, 12288],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(r4, False)
        ttnn_reshape_381 = ttnn.reshape(
            ttnn_slice_164,
            [2, 1, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_164, False)
        ttnn_add_80 = ttnn.add(
            ttnn_multiply_51,
            ttnn_reshape_381,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_381, False)
        ttnn.deallocate(ttnn_multiply_51, False)
        ttnn_reshape_382 = ttnn.reshape(
            ttnn_add_80,
            [8192, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_80, False)
        ttnn_linear_20 = ttnn.linear(
            ttnn_reshape_382,
            self.weights[f"transformer.transformer_blocks.{prev}.ff.net.0.proj.weight"],
            bias=self.weights[f"transformer.transformer_blocks.{prev}.ff.net.0.proj.bias"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation="gelu",
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_reshape_382, False)
        ttnn_matmul_21 = ttnn.matmul(
            ttnn_linear_20,
            self.weights[f"transformer.transformer_blocks.{prev}.ff.net.2.weight"],
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
        ttnn.deallocate(ttnn_linear_20, False)
        ttnn_reshape_383 = ttnn.reshape(
            ttnn_matmul_21,
            [1, 1, 8192, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_21, False)
        ttnn_reduce_scatter_24 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_383,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_reshape_383, False)
        ttnn_reshape_384 = ttnn.reshape(
            ttnn_reduce_scatter_24,
            [8192, 768],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reduce_scatter_24, False)
        ttnn_all_gather_24 = ttnn.all_gather(
            input_tensor=ttnn_reshape_384,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_384, False)
        ttnn_add_81 = ttnn.add(
            ttnn_all_gather_24,
            self.weights[f"transformer.transformer_blocks.{prev}.ff.net.2.bias.reshaped"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_all_gather_24, False)
        ttnn_reshape_385 = ttnn.reshape(
            ttnn_add_81,
            [2, 4096, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_81, False)
        ttnn_multiply_52 = ttnn.multiply(
            ttnn_reshape_379,
            ttnn_reshape_385,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_385, False)
        ttnn.deallocate(ttnn_reshape_379, False)
        ttnn_add_82 = ttnn.add(
            ttnn_add_78,
            ttnn_multiply_52,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_52, False)
        ttnn.deallocate(ttnn_add_78, False)
        ttnn_layer_norm_17 = ttnn.layer_norm(
            ttnn_add_82,
            epsilon=9.9999999747524271e-07,
            weight=None,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_reshape_386 = ttnn.reshape(
            r7,
            [1, 1, 2, 18432],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(r7, False)
        ttnn_reduce_scatter_25 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_386,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_reshape_386, False)
        ttnn_reshape_387 = ttnn.reshape(
            ttnn_reduce_scatter_25,
            [2, 4608],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reduce_scatter_25, False)
        ttnn_all_gather_25 = ttnn.all_gather(
            input_tensor=ttnn_reshape_387,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_387, False)
        ttnn_add_83 = ttnn.add(
            ttnn_all_gather_25,
            self.weights[f"transformer.transformer_blocks.{idx}.norm1.linear.bias.f32"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_all_gather_25, False)
        ttnn_typecast_149 = ttnn.typecast(
            ttnn_add_83,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_83, False)
        ttnn_slice_165 = ttnn.slice(
            ttnn_typecast_149,
            [0, 3072],
            [2, 6144],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_388 = ttnn.reshape(
            ttnn_slice_165,
            [2, 1, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_165, False)
        ttnn_add_84 = ttnn.add(
            ttnn_reshape_388,
            r3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_388, False)
        ttnn_multiply_53 = ttnn.multiply(
            ttnn_layer_norm_17,
            ttnn_add_84,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_84, False)
        ttnn.deallocate(ttnn_layer_norm_17, False)
        ttnn_slice_166 = ttnn.slice(
            ttnn_typecast_149,
            [0, 0],
            [2, 3072],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_389 = ttnn.reshape(
            ttnn_slice_166,
            [2, 1, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_166, False)
        ttnn_add_85 = ttnn.add(
            ttnn_multiply_53,
            ttnn_reshape_389,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_389, False)
        ttnn.deallocate(ttnn_multiply_53, False)
        ttnn_reshape_390 = ttnn.reshape(
            ttnn_add_85,
            [8192, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_85, False)
        ttnn_linear_21 = ttnn.linear(
            ttnn_reshape_390,
            self.weights[f"transformer.transformer_blocks.{idx}.attn.fused_to_q_to_k_to_v.weight"],
            bias=self.weights[f"transformer.transformer_blocks.{idx}.attn.fused_to_q_to_k_to_v.bias"],
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
        ttnn.deallocate(ttnn_reshape_390, False)
        ttnn_slice_167 = ttnn.slice(
            ttnn_linear_21,
            [0, 0],
            [8192, 768],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_168 = ttnn.slice(
            ttnn_linear_21,
            [0, 768],
            [8192, 1536],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_169 = ttnn.slice(
            ttnn_linear_21,
            [0, 1536],
            [8192, 2304],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_linear_21, False)
        ttnn_reshape_391 = ttnn.reshape(
            ttnn_slice_167,
            [2, 4096, 6, 128],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_167, False)
        ttnn_rms_norm_17 = ttnn.rms_norm(
            ttnn_reshape_391,
            epsilon=9.9999999747524271e-07,
            weight=self.weights[f"transformer.transformer_blocks.{idx}.attn.norm_q.weight"],
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
        ttnn.deallocate(ttnn_reshape_391, False)
        ttnn_concat_138 = ttnn.concat(
            [ttnn_rms_norm_16, ttnn_rms_norm_17],
            1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_rms_norm_17, False)
        ttnn.deallocate(ttnn_rms_norm_16, False)
        ttnn_typecast_150 = ttnn.typecast(
            ttnn_concat_138,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_277 = ttnn.permute(
            ttnn_typecast_150,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_150, False)
        ttnn_multiply_54 = ttnn.multiply(
            ttnn_permute_277,
            r8,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_277, False)
        ttnn_reshape_392 = ttnn.reshape(
            ttnn_concat_138,
            [2, 4141, 6, 64, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_concat_138, False)
        ttnn_slice_170 = ttnn.slice(
            ttnn_reshape_392,
            [0, 0, 0, 0, 1],
            [2, 4141, 6, 64, 2],
            [1, 1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_neg_8 = ttnn.neg(
            ttnn_slice_170,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_170, False)
        ttnn_slice_171 = ttnn.slice(
            ttnn_reshape_392,
            [0, 0, 0, 0, 0],
            [2, 4141, 6, 64, 1],
            [1, 1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_392, False)
        ttnn_concat_139 = ttnn.concat(
            [ttnn_neg_8, ttnn_slice_171],
            4,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_171, False)
        ttnn.deallocate(ttnn_neg_8, False)
        ttnn_typecast_151 = ttnn.typecast(
            ttnn_concat_139,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_concat_139, False)
        ttnn_reshape_393 = ttnn.reshape(
            ttnn_typecast_151,
            [2, 4141, 6, 128],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_151, False)
        ttnn_permute_278 = ttnn.permute(
            ttnn_reshape_393,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_393, False)
        ttnn_multiply_55 = ttnn.multiply(
            ttnn_permute_278,
            r9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_278, False)
        ttnn_add_86 = ttnn.add(
            ttnn_multiply_54,
            ttnn_multiply_55,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_55, False)
        ttnn.deallocate(ttnn_multiply_54, False)
        ttnn_typecast_152 = ttnn.typecast(
            ttnn_add_86,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_86, False)
        ttnn_reshape_394 = ttnn.reshape(
            ttnn_slice_158,
            [2, 45, 6, 128],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_158, False)
        ttnn_rms_norm_18 = ttnn.rms_norm(
            ttnn_reshape_394,
            epsilon=9.9999999747524271e-07,
            weight=self.weights[f"transformer.transformer_blocks.{idx}.attn.norm_added_k.weight"],
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
        ttnn.deallocate(ttnn_reshape_394, False)
        ttnn_reshape_395 = ttnn.reshape(
            ttnn_slice_168,
            [2, 4096, 6, 128],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_168, False)
        ttnn_rms_norm_19 = ttnn.rms_norm(
            ttnn_reshape_395,
            epsilon=9.9999999747524271e-07,
            weight=self.weights[f"transformer.transformer_blocks.{idx}.attn.norm_k.weight"],
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
        ttnn.deallocate(ttnn_reshape_395, False)
        ttnn_concat_140 = ttnn.concat(
            [ttnn_rms_norm_18, ttnn_rms_norm_19],
            1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_rms_norm_19, False)
        ttnn.deallocate(ttnn_rms_norm_18, False)
        ttnn_typecast_153 = ttnn.typecast(
            ttnn_concat_140,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_279 = ttnn.permute(
            ttnn_typecast_153,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_153, False)
        ttnn_multiply_56 = ttnn.multiply(
            ttnn_permute_279,
            r8,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_279, False)
        ttnn_reshape_396 = ttnn.reshape(
            ttnn_concat_140,
            [2, 4141, 6, 64, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_concat_140, False)
        ttnn_slice_172 = ttnn.slice(
            ttnn_reshape_396,
            [0, 0, 0, 0, 1],
            [2, 4141, 6, 64, 2],
            [1, 1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_neg_9 = ttnn.neg(
            ttnn_slice_172,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_172, False)
        ttnn_slice_173 = ttnn.slice(
            ttnn_reshape_396,
            [0, 0, 0, 0, 0],
            [2, 4141, 6, 64, 1],
            [1, 1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_396, False)
        ttnn_concat_141 = ttnn.concat(
            [ttnn_neg_9, ttnn_slice_173],
            4,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_173, False)
        ttnn.deallocate(ttnn_neg_9, False)
        ttnn_typecast_154 = ttnn.typecast(
            ttnn_concat_141,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_concat_141, False)
        ttnn_reshape_397 = ttnn.reshape(
            ttnn_typecast_154,
            [2, 4141, 6, 128],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_154, False)
        ttnn_permute_280 = ttnn.permute(
            ttnn_reshape_397,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_397, False)
        ttnn_multiply_57 = ttnn.multiply(
            ttnn_permute_280,
            r9,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_280, False)
        ttnn_add_87 = ttnn.add(
            ttnn_multiply_56,
            ttnn_multiply_57,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_57, False)
        ttnn.deallocate(ttnn_multiply_56, False)
        ttnn_typecast_155 = ttnn.typecast(
            ttnn_add_87,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_87, False)
        ttnn_reshape_398 = ttnn.reshape(
            ttnn_slice_159,
            [2, 45, 6, 128],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_159, False)
        ttnn_reshape_399 = ttnn.reshape(
            ttnn_slice_169,
            [2, 4096, 6, 128],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_169, False)
        ttnn_concat_142 = ttnn.concat(
            [ttnn_reshape_398, ttnn_reshape_399],
            1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_399, False)
        ttnn.deallocate(ttnn_reshape_398, False)
        ttnn_permute_281 = ttnn.permute(
            ttnn_concat_142,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_concat_142, False)
        ttnn_transformer_scaled_dot_product_attention_4 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_typecast_152,
                ttnn_typecast_155,
                ttnn_permute_281,
                attn_mask=r10,
                is_causal=False,
                scale=None,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
                program_config=None,
                compute_kernel_config=None,
                attention_sink=None,
            )
        )
        ttnn.deallocate(ttnn_permute_281, False)
        ttnn.deallocate(ttnn_typecast_155, False)
        ttnn.deallocate(ttnn_typecast_152, False)
        ttnn_transformer_concatenate_heads_4 = ttnn.transformer.concatenate_heads(
            ttnn_transformer_scaled_dot_product_attention_4,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_4, False)
        ttnn_slice_174 = ttnn.slice(
            ttnn_transformer_concatenate_heads_4,
            [0, 0, 0],
            [2, 45, 768],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_400 = ttnn.reshape(
            ttnn_slice_174,
            [90, 768],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_174, False)
        ttnn_matmul_22 = ttnn.matmul(
            ttnn_reshape_400,
            self.weights[f"transformer.transformer_blocks.{idx}.attn.to_add_out.weight"],
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
        ttnn.deallocate(ttnn_reshape_400, False)
        ttnn_reshape_401 = ttnn.reshape(
            ttnn_matmul_22,
            [1, 1, 90, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_22, False)
        ttnn_reduce_scatter_26 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_401,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_reshape_401, False)
        ttnn_reshape_402 = ttnn.reshape(
            ttnn_reduce_scatter_26,
            [90, 768],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reduce_scatter_26, False)
        ttnn_all_gather_26 = ttnn.all_gather(
            input_tensor=ttnn_reshape_402,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_402, False)
        ttnn_add_88 = ttnn.add(
            ttnn_all_gather_26,
            self.weights[f"transformer.transformer_blocks.{idx}.attn.to_add_out.bias.reshaped"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_all_gather_26, False)
        ttnn_reshape_403 = ttnn.reshape(
            ttnn_add_88,
            [2, 45, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_88, False)
        ttnn_multiply_58 = ttnn.multiply(
            ttnn_reshape_369,
            ttnn_reshape_403,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_403, False)
        ttnn.deallocate(ttnn_reshape_369, False)
        ttnn_add_89 = ttnn.add(
            ttnn_concat_137,
            ttnn_multiply_58,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_58, False)
        ttnn.deallocate(ttnn_concat_137, False)
        ttnn_layer_norm_18 = ttnn.layer_norm(
            ttnn_add_89,
            epsilon=9.9999999747524271e-07,
            weight=None,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_slice_175 = ttnn.slice(
            ttnn_typecast_148,
            [0, 15360],
            [2, 18432],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_404 = ttnn.reshape(
            ttnn_slice_175,
            [2, 1, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_175, False)
        ttnn_slice_176 = ttnn.slice(
            ttnn_typecast_148,
            [0, 12288],
            [2, 15360],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_405 = ttnn.reshape(
            ttnn_slice_176,
            [2, 1, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_176, False)
        ttnn_add_90 = ttnn.add(
            ttnn_reshape_405,
            r3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_405, False)
        ttnn_multiply_59 = ttnn.multiply(
            ttnn_layer_norm_18,
            ttnn_add_90,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_90, False)
        ttnn.deallocate(ttnn_layer_norm_18, False)
        ttnn_slice_177 = ttnn.slice(
            ttnn_typecast_148,
            [0, 9216],
            [2, 12288],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_148, False)
        ttnn_reshape_406 = ttnn.reshape(
            ttnn_slice_177,
            [2, 1, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_177, False)
        ttnn_add_91 = ttnn.add(
            ttnn_multiply_59,
            ttnn_reshape_406,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_406, False)
        ttnn.deallocate(ttnn_multiply_59, False)
        ttnn_reshape_407 = ttnn.reshape(
            ttnn_add_91,
            [90, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_91, False)
        ttnn_linear_22 = ttnn.linear(
            ttnn_reshape_407,
            self.weights[f"transformer.transformer_blocks.{idx}.ff_context.net.0.proj.weight"],
            bias=self.weights[f"transformer.transformer_blocks.{idx}.ff_context.net.0.proj.bias"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation="gelu",
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_reshape_407, False)
        ttnn_matmul_23 = ttnn.matmul(
            ttnn_linear_22,
            self.weights[f"transformer.transformer_blocks.{idx}.ff_context.net.2.weight"],
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
        ttnn.deallocate(ttnn_linear_22, False)
        ttnn_reshape_408 = ttnn.reshape(
            ttnn_matmul_23,
            [1, 1, 90, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_23, False)
        ttnn_reduce_scatter_27 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_408,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_reshape_408, False)
        ttnn_reshape_409 = ttnn.reshape(
            ttnn_reduce_scatter_27,
            [90, 768],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reduce_scatter_27, False)
        ttnn_all_gather_27 = ttnn.all_gather(
            input_tensor=ttnn_reshape_409,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_409, False)
        ttnn_add_92 = ttnn.add(
            ttnn_all_gather_27,
            self.weights[f"transformer.transformer_blocks.{idx}.ff_context.net.2.bias.reshaped"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_all_gather_27, False)
        ttnn_reshape_410 = ttnn.reshape(
            ttnn_add_92,
            [2, 45, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_92, False)
        ttnn_multiply_60 = ttnn.multiply(
            ttnn_reshape_404,
            ttnn_reshape_410,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_410, False)
        ttnn.deallocate(ttnn_reshape_404, False)
        ttnn_add_93 = ttnn.add(
            ttnn_add_89,
            ttnn_multiply_60,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_60, False)
        ttnn.deallocate(ttnn_add_89, False)
        ttnn_slice_178 = ttnn.slice(
            ttnn_add_93,
            [0, 0, 0],
            [2, 45, 1536],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_93, False)
        ttnn_to_layout_595 = ttnn.to_layout(
            r11,
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_411 = ttnn.reshape(
            ttnn_to_layout_595,
            [90, 2048],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_595, False)
        return ttnn_add_82, ttnn_reshape_411, ttnn_slice_178, ttnn_transformer_concatenate_heads_4, ttnn_typecast_149




def _tb_forward_0(self, hidden_states, txt_ids, img_ids, attention_mask, text_encoder_layer_1, ttnn_concat_109, ttnn_layer_norm_0, ttnn_slice_54, ttnn_slice_55, var_0, var_1):
    ttnn_reshape_185 = ttnn.reshape(
        ttnn_slice_55,
        [1, 1, 2, 18432],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_55, False)
    ttnn_reduce_scatter_0 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_185,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_185, False)
    ttnn_reshape_186 = ttnn.reshape(
        ttnn_reduce_scatter_0,
        [2, 4608],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_0, False)
    ttnn_all_gather_0 = ttnn.all_gather(
        input_tensor=ttnn_reshape_186,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_186, False)
    ttnn_add_0 = ttnn.add(
        ttnn_all_gather_0,
        self.weights["transformer.transformer_blocks.0.norm1_context.linear.bias.f32"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_0, False)
    ttnn_typecast_115 = ttnn.typecast(
        ttnn_add_0,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_0, False)
    ttnn_slice_56 = ttnn.slice(
        ttnn_typecast_115,
        [0, 6144],
        [2, 9216],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_187 = ttnn.reshape(
        ttnn_slice_56,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_56, False)
    ttnn_slice_57 = ttnn.slice(
        ttnn_typecast_115,
        [0, 3072],
        [2, 6144],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_188 = ttnn.reshape(
        ttnn_slice_57,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_57, False)
    ttnn_add_1 = ttnn.add(
        ttnn_reshape_188,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_188, False)
    ttnn_multiply_1 = ttnn.multiply(
        ttnn_layer_norm_0,
        ttnn_add_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_1, False)
    ttnn.deallocate(ttnn_layer_norm_0, False)
    ttnn_slice_58 = ttnn.slice(
        ttnn_typecast_115,
        [0, 0],
        [2, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_189 = ttnn.reshape(
        ttnn_slice_58,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_58, False)
    ttnn_add_2 = ttnn.add(
        ttnn_multiply_1,
        ttnn_reshape_189,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_189, False)
    ttnn.deallocate(ttnn_multiply_1, False)
    ttnn_reshape_190 = ttnn.reshape(
        ttnn_add_2,
        [90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_2, False)
    ttnn_linear_3 = ttnn.linear(
        ttnn_reshape_190,
        self.weights["transformer.transformer_blocks.0.attn.fused_add_q_proj_add_k_proj_add_v_proj.weight"],
        bias=self.weights["transformer.transformer_blocks.0.attn.fused_add_q_proj_add_k_proj_add_v_proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_190, False)
    ttnn_slice_59 = ttnn.slice(
        ttnn_linear_3,
        [0, 0],
        [90, 768],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_60 = ttnn.slice(
        ttnn_linear_3,
        [0, 768],
        [90, 1536],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_61 = ttnn.slice(
        ttnn_linear_3,
        [0, 1536],
        [90, 2304],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_3, False)
    ttnn_reshape_191 = ttnn.reshape(
        ttnn_slice_59,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_59, False)
    ttnn_rms_norm_0 = ttnn.rms_norm(
        ttnn_reshape_191,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.0.attn.norm_added_q.weight"],
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
    ttnn.deallocate(ttnn_reshape_191, False)
    ttnn_to_layout_587 = ttnn.to_layout(
        hidden_states,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_192 = ttnn.reshape(
        ttnn_to_layout_587,
        [8192, 48],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_587, False)
    ttnn_linear_4 = ttnn.linear(
        ttnn_reshape_192,
        self.weights["transformer.x_embedder.weight"],
        bias=self.weights["transformer.x_embedder.bias"],
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
    ttnn.deallocate(ttnn_reshape_192, False)
    ttnn_reshape_193 = ttnn.reshape(
        ttnn_linear_4,
        [2, 4096, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_4, False)
    ttnn_layer_norm_1 = ttnn.layer_norm(
        ttnn_reshape_193,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_reshape_194 = ttnn.reshape(
        ttnn_slice_54,
        [1, 1, 2, 18432],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_54, False)
    ttnn_reduce_scatter_1 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_194,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_194, False)
    ttnn_reshape_195 = ttnn.reshape(
        ttnn_reduce_scatter_1,
        [2, 4608],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_1, False)
    ttnn_all_gather_1 = ttnn.all_gather(
        input_tensor=ttnn_reshape_195,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_195, False)
    ttnn_add_3 = ttnn.add(
        ttnn_all_gather_1,
        self.weights["transformer.transformer_blocks.0.norm1.linear.bias.f32"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_1, False)
    ttnn_typecast_116 = ttnn.typecast(
        ttnn_add_3,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_3, False)
    ttnn_slice_62 = ttnn.slice(
        ttnn_typecast_116,
        [0, 3072],
        [2, 6144],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_196 = ttnn.reshape(
        ttnn_slice_62,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_62, False)
    ttnn_add_4 = ttnn.add(
        ttnn_reshape_196,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_196, False)
    ttnn_multiply_2 = ttnn.multiply(
        ttnn_layer_norm_1,
        ttnn_add_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_4, False)
    ttnn.deallocate(ttnn_layer_norm_1, False)
    ttnn_slice_63 = ttnn.slice(
        ttnn_typecast_116,
        [0, 0],
        [2, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_197 = ttnn.reshape(
        ttnn_slice_63,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_63, False)
    ttnn_add_5 = ttnn.add(
        ttnn_multiply_2,
        ttnn_reshape_197,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_197, False)
    ttnn.deallocate(ttnn_multiply_2, False)
    ttnn_reshape_198 = ttnn.reshape(
        ttnn_add_5,
        [8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_5, False)
    ttnn_linear_5 = ttnn.linear(
        ttnn_reshape_198,
        self.weights["transformer.transformer_blocks.0.attn.fused_to_q_to_k_to_v.weight"],
        bias=self.weights["transformer.transformer_blocks.0.attn.fused_to_q_to_k_to_v.bias"],
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
    ttnn.deallocate(ttnn_reshape_198, False)
    ttnn_slice_64 = ttnn.slice(
        ttnn_linear_5,
        [0, 0],
        [8192, 768],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_65 = ttnn.slice(
        ttnn_linear_5,
        [0, 768],
        [8192, 1536],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_66 = ttnn.slice(
        ttnn_linear_5,
        [0, 1536],
        [8192, 2304],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_5, False)
    ttnn_reshape_199 = ttnn.reshape(
        ttnn_slice_64,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_64, False)
    ttnn_rms_norm_1 = ttnn.rms_norm(
        ttnn_reshape_199,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.0.attn.norm_q.weight"],
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
    ttnn.deallocate(ttnn_reshape_199, False)
    ttnn_concat_111 = ttnn.concat(
        [ttnn_rms_norm_0, ttnn_rms_norm_1],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_1, False)
    ttnn.deallocate(ttnn_rms_norm_0, False)
    ttnn_typecast_117 = ttnn.typecast(
        ttnn_concat_111,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_257 = ttnn.permute(
        ttnn_typecast_117,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_117, False)
    ttnn_to_layout_588 = ttnn.to_layout(
        txt_ids,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_589 = ttnn.to_layout(
        img_ids,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_concat_112 = ttnn.concat(
        [ttnn_to_layout_588, ttnn_to_layout_589],
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_589, False)
    ttnn.deallocate(ttnn_to_layout_588, False)
    ttnn_typecast_118 = ttnn.typecast(
        ttnn_concat_112,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_112, False)
    ttnn_slice_67 = ttnn.slice(
        ttnn_typecast_118,
        [0, 0],
        [4141, 1],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_3 = ttnn.multiply(
        ttnn_slice_67,
        self.weights["consteval.const_176"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_67, False)
    ttnn_cos_1 = ttnn.cos(
        ttnn_multiply_3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_interleave_0 = ttnn.repeat_interleave(
        ttnn_cos_1,
        2,
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_cos_1, False)
    ttnn_slice_68 = ttnn.slice(
        ttnn_typecast_118,
        [0, 1],
        [4141, 2],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_multiply_4 = ttnn.multiply(
        ttnn_slice_68,
        var_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_68, False)
    ttnn_cos_2 = ttnn.cos(
        ttnn_multiply_4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_interleave_1 = ttnn.repeat_interleave(
        ttnn_cos_2,
        2,
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_cos_2, False)
    ttnn_slice_69 = ttnn.slice(
        ttnn_typecast_118,
        [0, 2],
        [4141, 3],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_118, False)
    ttnn_multiply_5 = ttnn.multiply(
        ttnn_slice_69,
        var_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_69, False)
    ttnn_cos_3 = ttnn.cos(
        ttnn_multiply_5,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_interleave_2 = ttnn.repeat_interleave(
        ttnn_cos_3,
        2,
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_cos_3, False)
    ttnn_reshape_200 = ttnn.reshape(
        ttnn_repeat_interleave_0,
        [1, 4141, 1, 16],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_repeat_interleave_0, False)
    ttnn_reshape_201 = ttnn.reshape(
        ttnn_repeat_interleave_1,
        [1, 4141, 1, 56],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_repeat_interleave_1, False)
    ttnn_reshape_202 = ttnn.reshape(
        ttnn_repeat_interleave_2,
        [1, 4141, 1, 56],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_repeat_interleave_2, False)
    ttnn_concat_113 = ttnn.concat(
        [ttnn_reshape_200, ttnn_reshape_201, ttnn_reshape_202],
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_202, False)
    ttnn.deallocate(ttnn_reshape_201, False)
    ttnn.deallocate(ttnn_reshape_200, False)
    ttnn_reshape_203 = ttnn.reshape(
        ttnn_concat_113,
        [1, 1, 4141, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_113, False)
    ttnn_multiply_6 = ttnn.multiply(
        ttnn_permute_257,
        ttnn_reshape_203,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_257, False)
    ttnn_reshape_204 = ttnn.reshape(
        ttnn_concat_111,
        [2, 4141, 6, 64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_111, False)
    ttnn_slice_70 = ttnn.slice(
        ttnn_reshape_204,
        [0, 0, 0, 0, 1],
        [2, 4141, 6, 64, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_0 = ttnn.neg(
        ttnn_slice_70,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_70, False)
    ttnn_slice_71 = ttnn.slice(
        ttnn_reshape_204,
        [0, 0, 0, 0, 0],
        [2, 4141, 6, 64, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_204, False)
    ttnn_concat_114 = ttnn.concat(
        [ttnn_neg_0, ttnn_slice_71],
        4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_71, False)
    ttnn.deallocate(ttnn_neg_0, False)
    ttnn_typecast_119 = ttnn.typecast(
        ttnn_concat_114,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_114, False)
    ttnn_reshape_205 = ttnn.reshape(
        ttnn_typecast_119,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_119, False)
    ttnn_permute_258 = ttnn.permute(
        ttnn_reshape_205,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_205, False)
    ttnn_sin_1 = ttnn.sin(
        ttnn_multiply_3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_3, False)
    ttnn_repeat_interleave_3 = ttnn.repeat_interleave(
        ttnn_sin_1,
        2,
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_sin_1, False)
    ttnn_sin_2 = ttnn.sin(
        ttnn_multiply_4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_4, False)
    ttnn_repeat_interleave_4 = ttnn.repeat_interleave(
        ttnn_sin_2,
        2,
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_sin_2, False)
    ttnn_sin_3 = ttnn.sin(
        ttnn_multiply_5,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_5, False)
    ttnn_repeat_interleave_5 = ttnn.repeat_interleave(
        ttnn_sin_3,
        2,
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_sin_3, False)
    ttnn_reshape_206 = ttnn.reshape(
        ttnn_repeat_interleave_3,
        [1, 4141, 1, 16],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_repeat_interleave_3, False)
    ttnn_reshape_207 = ttnn.reshape(
        ttnn_repeat_interleave_4,
        [1, 4141, 1, 56],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_repeat_interleave_4, False)
    ttnn_reshape_208 = ttnn.reshape(
        ttnn_repeat_interleave_5,
        [1, 4141, 1, 56],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_repeat_interleave_5, False)
    ttnn_concat_115 = ttnn.concat(
        [ttnn_reshape_206, ttnn_reshape_207, ttnn_reshape_208],
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_208, False)
    ttnn.deallocate(ttnn_reshape_207, False)
    ttnn.deallocate(ttnn_reshape_206, False)
    ttnn_reshape_209 = ttnn.reshape(
        ttnn_concat_115,
        [1, 1, 4141, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_115, False)
    ttnn_multiply_7 = ttnn.multiply(
        ttnn_permute_258,
        ttnn_reshape_209,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_258, False)
    ttnn_add_6 = ttnn.add(
        ttnn_multiply_6,
        ttnn_multiply_7,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_7, False)
    ttnn.deallocate(ttnn_multiply_6, False)
    ttnn_typecast_120 = ttnn.typecast(
        ttnn_add_6,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_6, False)
    ttnn_reshape_210 = ttnn.reshape(
        ttnn_slice_60,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_60, False)
    ttnn_rms_norm_2 = ttnn.rms_norm(
        ttnn_reshape_210,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.0.attn.norm_added_k.weight"],
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
    ttnn.deallocate(ttnn_reshape_210, False)
    ttnn_reshape_211 = ttnn.reshape(
        ttnn_slice_65,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_65, False)
    ttnn_rms_norm_3 = ttnn.rms_norm(
        ttnn_reshape_211,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.0.attn.norm_k.weight"],
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
    ttnn.deallocate(ttnn_reshape_211, False)
    ttnn_concat_116 = ttnn.concat(
        [ttnn_rms_norm_2, ttnn_rms_norm_3],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_3, False)
    ttnn.deallocate(ttnn_rms_norm_2, False)
    ttnn_typecast_121 = ttnn.typecast(
        ttnn_concat_116,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_259 = ttnn.permute(
        ttnn_typecast_121,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_121, False)
    ttnn_multiply_8 = ttnn.multiply(
        ttnn_permute_259,
        ttnn_reshape_203,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_259, False)
    ttnn_reshape_212 = ttnn.reshape(
        ttnn_concat_116,
        [2, 4141, 6, 64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_116, False)
    ttnn_slice_72 = ttnn.slice(
        ttnn_reshape_212,
        [0, 0, 0, 0, 1],
        [2, 4141, 6, 64, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_1 = ttnn.neg(
        ttnn_slice_72,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_72, False)
    ttnn_slice_73 = ttnn.slice(
        ttnn_reshape_212,
        [0, 0, 0, 0, 0],
        [2, 4141, 6, 64, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_212, False)
    ttnn_concat_117 = ttnn.concat(
        [ttnn_neg_1, ttnn_slice_73],
        4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_73, False)
    ttnn.deallocate(ttnn_neg_1, False)
    ttnn_typecast_122 = ttnn.typecast(
        ttnn_concat_117,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_117, False)
    ttnn_reshape_213 = ttnn.reshape(
        ttnn_typecast_122,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_122, False)
    ttnn_permute_260 = ttnn.permute(
        ttnn_reshape_213,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_213, False)
    ttnn_multiply_9 = ttnn.multiply(
        ttnn_permute_260,
        ttnn_reshape_209,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_260, False)
    ttnn_add_7 = ttnn.add(
        ttnn_multiply_8,
        ttnn_multiply_9,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_9, False)
    ttnn.deallocate(ttnn_multiply_8, False)
    ttnn_typecast_123 = ttnn.typecast(
        ttnn_add_7,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_7, False)
    ttnn_reshape_214 = ttnn.reshape(
        ttnn_slice_61,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_61, False)
    ttnn_reshape_215 = ttnn.reshape(
        ttnn_slice_66,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_66, False)
    ttnn_concat_118 = ttnn.concat(
        [ttnn_reshape_214, ttnn_reshape_215],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_215, False)
    ttnn.deallocate(ttnn_reshape_214, False)
    ttnn_permute_261 = ttnn.permute(
        ttnn_concat_118,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_concat_118, False)
    ttnn_to_layout_590 = ttnn.to_layout(
        attention_mask,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_transformer_scaled_dot_product_attention_0 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_typecast_120,
            ttnn_typecast_123,
            ttnn_permute_261,
            attn_mask=ttnn_to_layout_590,
            is_causal=False,
            scale=None,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
            compute_kernel_config=None,
            attention_sink=None,
        )
    )
    ttnn.deallocate(ttnn_permute_261, False)
    ttnn.deallocate(ttnn_typecast_123, False)
    ttnn.deallocate(ttnn_typecast_120, False)
    ttnn_transformer_concatenate_heads_0 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_0, False)
    ttnn_slice_74 = ttnn.slice(
        ttnn_transformer_concatenate_heads_0,
        [0, 0, 0],
        [2, 45, 768],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_216 = ttnn.reshape(
        ttnn_slice_74,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_74, False)
    ttnn_matmul_2 = ttnn.matmul(
        ttnn_reshape_216,
        self.weights["transformer.transformer_blocks.0.attn.to_add_out.weight"],
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
    ttnn.deallocate(ttnn_reshape_216, False)
    ttnn_reshape_217 = ttnn.reshape(
        ttnn_matmul_2,
        [1, 1, 90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_2, False)
    ttnn_reduce_scatter_2 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_217,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_217, False)
    ttnn_reshape_218 = ttnn.reshape(
        ttnn_reduce_scatter_2,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_2, False)
    ttnn_all_gather_2 = ttnn.all_gather(
        input_tensor=ttnn_reshape_218,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_218, False)
    ttnn_add_8 = ttnn.add(
        ttnn_all_gather_2,
        self.weights["transformer.transformer_blocks.0.attn.to_add_out.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_2, False)
    ttnn_reshape_219 = ttnn.reshape(
        ttnn_add_8,
        [2, 45, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_8, False)
    ttnn_multiply_10 = ttnn.multiply(
        ttnn_reshape_187,
        ttnn_reshape_219,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_219, False)
    ttnn.deallocate(ttnn_reshape_187, False)
    ttnn_add_9 = ttnn.add(
        ttnn_concat_109,
        ttnn_multiply_10,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_10, False)
    ttnn.deallocate(ttnn_concat_109, False)
    ttnn_layer_norm_2 = ttnn.layer_norm(
        ttnn_add_9,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_slice_75 = ttnn.slice(
        ttnn_typecast_115,
        [0, 15360],
        [2, 18432],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_220 = ttnn.reshape(
        ttnn_slice_75,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_75, False)
    ttnn_slice_76 = ttnn.slice(
        ttnn_typecast_115,
        [0, 12288],
        [2, 15360],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_221 = ttnn.reshape(
        ttnn_slice_76,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_76, False)
    ttnn_add_10 = ttnn.add(
        ttnn_reshape_221,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_221, False)
    ttnn_multiply_11 = ttnn.multiply(
        ttnn_layer_norm_2,
        ttnn_add_10,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_10, False)
    ttnn.deallocate(ttnn_layer_norm_2, False)
    ttnn_slice_77 = ttnn.slice(
        ttnn_typecast_115,
        [0, 9216],
        [2, 12288],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_115, False)
    ttnn_reshape_222 = ttnn.reshape(
        ttnn_slice_77,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_77, False)
    ttnn_add_11 = ttnn.add(
        ttnn_multiply_11,
        ttnn_reshape_222,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_222, False)
    ttnn.deallocate(ttnn_multiply_11, False)
    ttnn_reshape_223 = ttnn.reshape(
        ttnn_add_11,
        [90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_11, False)
    ttnn_linear_6 = ttnn.linear(
        ttnn_reshape_223,
        self.weights["transformer.transformer_blocks.0.ff_context.net.0.proj.weight"],
        bias=self.weights["transformer.transformer_blocks.0.ff_context.net.0.proj.bias"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation="gelu",
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_223, False)
    ttnn_matmul_3 = ttnn.matmul(
        ttnn_linear_6,
        self.weights["transformer.transformer_blocks.0.ff_context.net.2.weight"],
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
    ttnn.deallocate(ttnn_linear_6, False)
    ttnn_reshape_224 = ttnn.reshape(
        ttnn_matmul_3,
        [1, 1, 90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_3, False)
    ttnn_reduce_scatter_3 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_224,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_224, False)
    ttnn_reshape_225 = ttnn.reshape(
        ttnn_reduce_scatter_3,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_3, False)
    ttnn_all_gather_3 = ttnn.all_gather(
        input_tensor=ttnn_reshape_225,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_225, False)
    ttnn_add_12 = ttnn.add(
        ttnn_all_gather_3,
        self.weights["transformer.transformer_blocks.0.ff_context.net.2.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_3, False)
    ttnn_reshape_226 = ttnn.reshape(
        ttnn_add_12,
        [2, 45, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_12, False)
    ttnn_multiply_12 = ttnn.multiply(
        ttnn_reshape_220,
        ttnn_reshape_226,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_226, False)
    ttnn.deallocate(ttnn_reshape_220, False)
    ttnn_add_13 = ttnn.add(
        ttnn_add_9,
        ttnn_multiply_12,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_12, False)
    ttnn.deallocate(ttnn_add_9, False)
    ttnn_slice_78 = ttnn.slice(
        ttnn_add_13,
        [0, 0, 0],
        [2, 45, 1536],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_13, False)
    ttnn_to_layout_591 = ttnn.to_layout(
        text_encoder_layer_1,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_227 = ttnn.reshape(
        ttnn_to_layout_591,
        [90, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_591, False)
    return ttnn_reshape_193, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_227, ttnn_slice_78, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_0, ttnn_typecast_116




class BriaFiboSingleTransformerBlock(LightweightModule):
    def __init__(self, device, weights, block_idx):
        self.device = device
        self.weights = weights
        self.block_idx = block_idx

    def forward(self, *args):
        # Blocks 1..36 are identical up to the block index and per-block inputs, so
        # they share this one implementation (indexed by self.block_idx). Blocks 0 and
        # 37 are genuine structural exceptions kept as their own functions:
        #   - block 0 also finishes the last double block (its residual tail) and
        #     slices its own modulation inline, both woven through the body;
        #   - block 37 is the output block (final layer_norm, no next block / text layer).
        idx = self.block_idx
        if idx == 0:
            return _stb_forward_0(self, *args)
        if idx == 37:
            return _stb_forward_37(self, *args)
        idx8 = idx + 8  # caption_projection index = block_idx + 8
        # canonical inputs: r0 modulation-reshape, r1 hi-slice, r2 residual, r3 lo-slice,
        # r4 shared const, r5/r6 shared rope, r7 shared, r8 text-encoder layer
        r0, r1, r2, r3, r4, r5, r6, r7, r8 = args
        ttnn_matmul_53 = ttnn.matmul(
            r0,
            self.weights[f"transformer.caption_projection.{idx8}.linear.weight"],
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
        ttnn.deallocate(r0, False)
        ttnn_reshape_682 = ttnn.reshape(
            ttnn_matmul_53,
            [2, 45, 1536],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_53, False)
        ttnn_concat_191 = ttnn.concat(
            [r1, ttnn_reshape_682],
            2,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_682, False)
        ttnn.deallocate(r1, False)
        ttnn_slice_336 = ttnn.slice(
            r2,
            [0, 45, 0],
            [2, 4141, 3072],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(r2, False)
        ttnn_concat_192 = ttnn.concat(
            [ttnn_concat_191, ttnn_slice_336],
            1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_336, False)
        ttnn.deallocate(ttnn_concat_191, False)
        ttnn_layer_norm_38 = ttnn.layer_norm(
            ttnn_concat_192,
            epsilon=9.9999999747524271e-07,
            weight=None,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
        )
        ttnn_reshape_683 = ttnn.reshape(
            r3,
            [1, 1, 2, 9216],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(r3, False)
        ttnn_reduce_scatter_60 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_683,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_reshape_683, False)
        ttnn_reshape_684 = ttnn.reshape(
            ttnn_reduce_scatter_60,
            [2, 2304],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reduce_scatter_60, False)
        ttnn_all_gather_72 = ttnn.all_gather(
            input_tensor=ttnn_reshape_684,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_684, False)
        ttnn_add_202 = ttnn.add(
            ttnn_all_gather_72,
            self.weights[f"transformer.single_transformer_blocks.{idx}.norm.linear.bias.f32"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_all_gather_72, False)
        ttnn_typecast_222 = ttnn.typecast(
            ttnn_add_202,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_202, False)
        ttnn_slice_337 = ttnn.slice(
            ttnn_typecast_222,
            [0, 6144],
            [2, 9216],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_685 = ttnn.reshape(
            ttnn_slice_337,
            [2, 1, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_337, False)
        ttnn_slice_338 = ttnn.slice(
            ttnn_typecast_222,
            [0, 3072],
            [2, 6144],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_686 = ttnn.reshape(
            ttnn_slice_338,
            [2, 1, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_338, False)
        ttnn_add_203 = ttnn.add(
            ttnn_reshape_686,
            r4,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_686, False)
        ttnn_multiply_136 = ttnn.multiply(
            ttnn_layer_norm_38,
            ttnn_add_203,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_203, False)
        ttnn.deallocate(ttnn_layer_norm_38, False)
        ttnn_slice_339 = ttnn.slice(
            ttnn_typecast_222,
            [0, 0],
            [2, 3072],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_222, False)
        ttnn_reshape_687 = ttnn.reshape(
            ttnn_slice_339,
            [2, 1, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_339, False)
        ttnn_add_204 = ttnn.add(
            ttnn_multiply_136,
            ttnn_reshape_687,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_687, False)
        ttnn.deallocate(ttnn_multiply_136, False)
        ttnn_reshape_688 = ttnn.reshape(
            ttnn_add_204,
            [8282, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_204, False)
        ttnn_linear_42 = ttnn.linear(
            ttnn_reshape_688,
            self.weights[f"transformer.single_transformer_blocks.{idx}.fused_attn_to_q_to_k_to_v_proj_mlp.weight"],
            bias=self.weights[f"transformer.single_transformer_blocks.{idx}.fused_attn_to_q_to_k_to_v_proj_mlp.bias"],
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
        ttnn.deallocate(ttnn_reshape_688, False)
        ttnn_slice_340 = ttnn.slice(
            ttnn_linear_42,
            [0, 0],
            [8282, 768],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_341 = ttnn.slice(
            ttnn_linear_42,
            [0, 768],
            [8282, 1536],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_342 = ttnn.slice(
            ttnn_linear_42,
            [0, 1536],
            [8282, 2304],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_343 = ttnn.slice(
            ttnn_linear_42,
            [0, 2304],
            [8282, 5376],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_linear_42, False)
        ttnn_reshape_689 = ttnn.reshape(
            ttnn_slice_340,
            [2, 4141, 6, 128],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_340, False)
        ttnn_rms_norm_44 = ttnn.rms_norm(
            ttnn_reshape_689,
            epsilon=9.9999999747524271e-07,
            weight=self.weights[f"transformer.single_transformer_blocks.{idx}.attn.norm_q.weight"],
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
        ttnn.deallocate(ttnn_reshape_689, False)
        ttnn_typecast_223 = ttnn.typecast(
            ttnn_rms_norm_44,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_327 = ttnn.permute(
            ttnn_typecast_223,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_223, False)
        ttnn_multiply_137 = ttnn.multiply(
            ttnn_permute_327,
            r5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_327, False)
        ttnn_reshape_690 = ttnn.reshape(
            ttnn_rms_norm_44,
            [2, 4141, 6, 64, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_rms_norm_44, False)
        ttnn_slice_344 = ttnn.slice(
            ttnn_reshape_690,
            [0, 0, 0, 0, 1],
            [2, 4141, 6, 64, 2],
            [1, 1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_neg_28 = ttnn.neg(
            ttnn_slice_344,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_344, False)
        ttnn_slice_345 = ttnn.slice(
            ttnn_reshape_690,
            [0, 0, 0, 0, 0],
            [2, 4141, 6, 64, 1],
            [1, 1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_690, False)
        ttnn_concat_193 = ttnn.concat(
            [ttnn_neg_28, ttnn_slice_345],
            4,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_345, False)
        ttnn.deallocate(ttnn_neg_28, False)
        ttnn_typecast_224 = ttnn.typecast(
            ttnn_concat_193,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_concat_193, False)
        ttnn_reshape_691 = ttnn.reshape(
            ttnn_typecast_224,
            [2, 4141, 6, 128],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_224, False)
        ttnn_permute_328 = ttnn.permute(
            ttnn_reshape_691,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_691, False)
        ttnn_multiply_138 = ttnn.multiply(
            ttnn_permute_328,
            r6,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_328, False)
        ttnn_add_205 = ttnn.add(
            ttnn_multiply_137,
            ttnn_multiply_138,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_138, False)
        ttnn.deallocate(ttnn_multiply_137, False)
        ttnn_typecast_225 = ttnn.typecast(
            ttnn_add_205,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_205, False)
        ttnn_reshape_692 = ttnn.reshape(
            ttnn_slice_341,
            [2, 4141, 6, 128],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_341, False)
        ttnn_rms_norm_45 = ttnn.rms_norm(
            ttnn_reshape_692,
            epsilon=9.9999999747524271e-07,
            weight=self.weights[f"transformer.single_transformer_blocks.{idx}.attn.norm_k.weight"],
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
        ttnn.deallocate(ttnn_reshape_692, False)
        ttnn_typecast_226 = ttnn.typecast(
            ttnn_rms_norm_45,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_329 = ttnn.permute(
            ttnn_typecast_226,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_typecast_226, False)
        ttnn_multiply_139 = ttnn.multiply(
            ttnn_permute_329,
            r5,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_329, False)
        ttnn_reshape_693 = ttnn.reshape(
            ttnn_rms_norm_45,
            [2, 4141, 6, 64, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_rms_norm_45, False)
        ttnn_slice_346 = ttnn.slice(
            ttnn_reshape_693,
            [0, 0, 0, 0, 1],
            [2, 4141, 6, 64, 2],
            [1, 1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_neg_29 = ttnn.neg(
            ttnn_slice_346,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_346, False)
        ttnn_slice_347 = ttnn.slice(
            ttnn_reshape_693,
            [0, 0, 0, 0, 0],
            [2, 4141, 6, 64, 1],
            [1, 1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_693, False)
        ttnn_concat_194 = ttnn.concat(
            [ttnn_neg_29, ttnn_slice_347],
            4,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_347, False)
        ttnn.deallocate(ttnn_neg_29, False)
        ttnn_typecast_227 = ttnn.typecast(
            ttnn_concat_194,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_concat_194, False)
        ttnn_reshape_694 = ttnn.reshape(
            ttnn_typecast_227,
            [2, 4141, 6, 128],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_227, False)
        ttnn_permute_330 = ttnn.permute(
            ttnn_reshape_694,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_694, False)
        ttnn_multiply_140 = ttnn.multiply(
            ttnn_permute_330,
            r6,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_330, False)
        ttnn_add_206 = ttnn.add(
            ttnn_multiply_139,
            ttnn_multiply_140,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_140, False)
        ttnn.deallocate(ttnn_multiply_139, False)
        ttnn_typecast_228 = ttnn.typecast(
            ttnn_add_206,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_206, False)
        ttnn_reshape_695 = ttnn.reshape(
            ttnn_slice_342,
            [2, 4141, 6, 128],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_342, False)
        ttnn_permute_331 = ttnn.permute(
            ttnn_reshape_695,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_695, False)
        ttnn_transformer_scaled_dot_product_attention_14 = (
            ttnn.transformer.scaled_dot_product_attention(
                ttnn_typecast_225,
                ttnn_typecast_228,
                ttnn_permute_331,
                attn_mask=r7,
                is_causal=False,
                scale=None,
                sliding_window_size=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
                program_config=None,
                compute_kernel_config=None,
                attention_sink=None,
            )
        )
        ttnn.deallocate(ttnn_permute_331, False)
        ttnn.deallocate(ttnn_typecast_228, False)
        ttnn.deallocate(ttnn_typecast_225, False)
        ttnn_transformer_concatenate_heads_14 = ttnn.transformer.concatenate_heads(
            ttnn_transformer_scaled_dot_product_attention_14,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_14, False)
        ttnn_gelu_6 = ttnn.gelu(
            ttnn_slice_343,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_343, False)
        ttnn_reshape_696 = ttnn.reshape(
            ttnn_gelu_6,
            [2, 4141, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_gelu_6, False)
        ttnn_all_gather_73 = ttnn.all_gather(
            input_tensor=ttnn_transformer_concatenate_heads_14,
            dim=2,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_transformer_concatenate_heads_14, False)
        ttnn_all_gather_74 = ttnn.all_gather(
            input_tensor=ttnn_reshape_696,
            dim=2,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_696, False)
        ttnn_concat_195 = ttnn.concat(
            [ttnn_all_gather_73, ttnn_all_gather_74],
            2,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_all_gather_74, False)
        ttnn.deallocate(ttnn_all_gather_73, False)
        ttnn_to_layout_617 = ttnn.to_layout(
            ttnn_concat_195,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_concat_195, False)
        ttnn_mesh_partition_8 = ttnn.mesh_partition(
            input_tensor=ttnn_to_layout_617,
            dim=2,
            cluster_axis=1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_617, False)
        ttnn_to_layout_618 = ttnn.to_layout(
            ttnn_mesh_partition_8,
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_mesh_partition_8, False)
        ttnn_reshape_697 = ttnn.reshape(
            ttnn_to_layout_618,
            [8282, 3840],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_618, False)
        ttnn_matmul_54 = ttnn.matmul(
            ttnn_reshape_697,
            self.weights[f"transformer.single_transformer_blocks.{idx}.proj_out.weight"],
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
        ttnn.deallocate(ttnn_reshape_697, False)
        ttnn_reshape_698 = ttnn.reshape(
            ttnn_matmul_54,
            [1, 1, 8282, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_54, False)
        ttnn_reduce_scatter_61 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_698,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_reshape_698, False)
        ttnn_reshape_699 = ttnn.reshape(
            ttnn_reduce_scatter_61,
            [8282, 768],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reduce_scatter_61, False)
        ttnn_all_gather_75 = ttnn.all_gather(
            input_tensor=ttnn_reshape_699,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_699, False)
        ttnn_add_207 = ttnn.add(
            ttnn_all_gather_75,
            self.weights[f"transformer.single_transformer_blocks.{idx}.proj_out.bias.reshaped"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_all_gather_75, False)
        ttnn_reshape_700 = ttnn.reshape(
            ttnn_add_207,
            [2, 4141, 3072],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_207, False)
        ttnn_multiply_141 = ttnn.multiply(
            ttnn_reshape_685,
            ttnn_reshape_700,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_700, False)
        ttnn.deallocate(ttnn_reshape_685, False)
        ttnn_add_208 = ttnn.add(
            ttnn_concat_192,
            ttnn_multiply_141,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_141, False)
        ttnn.deallocate(ttnn_concat_192, False)
        ttnn_slice_348 = ttnn.slice(
            ttnn_add_208,
            [0, 0, 0],
            [2, 45, 1536],
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_to_layout_619 = ttnn.to_layout(
            r8,
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_701 = ttnn.reshape(
            ttnn_to_layout_619,
            [90, 2048],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_619, False)
        return ttnn_add_208, ttnn_reshape_701, ttnn_slice_348




def _stb_forward_0(self, text_encoder_layer_9, ttnn_add_142, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_549, ttnn_slice_253, ttnn_slice_39, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_7, ttnn_typecast_173, var_1):
    ttnn_matmul_39 = ttnn.matmul(
        ttnn_reshape_549,
        self.weights["transformer.caption_projection.8.linear.weight"],
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
    ttnn.deallocate(ttnn_reshape_549, False)
    ttnn_reshape_550 = ttnn.reshape(
        ttnn_matmul_39,
        [2, 45, 1536],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_39, False)
    ttnn_concat_161 = ttnn.concat(
        [ttnn_slice_253, ttnn_reshape_550],
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_550, False)
    ttnn.deallocate(ttnn_slice_253, False)
    ttnn_slice_254 = ttnn.slice(
        ttnn_typecast_173,
        [0, 6144],
        [2, 9216],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_551 = ttnn.reshape(
        ttnn_slice_254,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_254, False)
    ttnn_slice_255 = ttnn.slice(
        ttnn_transformer_concatenate_heads_7,
        [0, 45, 0],
        [2, 4141, 768],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_7, False)
    ttnn_reshape_552 = ttnn.reshape(
        ttnn_slice_255,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_255, False)
    ttnn_matmul_40 = ttnn.matmul(
        ttnn_reshape_552,
        self.weights["transformer.transformer_blocks.7.attn.to_out.0.weight"],
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
    ttnn.deallocate(ttnn_reshape_552, False)
    ttnn_reshape_553 = ttnn.reshape(
        ttnn_matmul_40,
        [1, 1, 8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_40, False)
    ttnn_reduce_scatter_46 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_553,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_553, False)
    ttnn_reshape_554 = ttnn.reshape(
        ttnn_reduce_scatter_46,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_46, False)
    ttnn_all_gather_46 = ttnn.all_gather(
        input_tensor=ttnn_reshape_554,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_554, False)
    ttnn_add_154 = ttnn.add(
        ttnn_all_gather_46,
        self.weights["transformer.transformer_blocks.7.attn.to_out.0.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_46, False)
    ttnn_reshape_555 = ttnn.reshape(
        ttnn_add_154,
        [2, 4096, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_154, False)
    ttnn_multiply_97 = ttnn.multiply(
        ttnn_reshape_551,
        ttnn_reshape_555,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_555, False)
    ttnn.deallocate(ttnn_reshape_551, False)
    ttnn_add_155 = ttnn.add(
        ttnn_add_142,
        ttnn_multiply_97,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_97, False)
    ttnn.deallocate(ttnn_add_142, False)
    ttnn_layer_norm_31 = ttnn.layer_norm(
        ttnn_add_155,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_slice_256 = ttnn.slice(
        ttnn_typecast_173,
        [0, 15360],
        [2, 18432],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_556 = ttnn.reshape(
        ttnn_slice_256,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_256, False)
    ttnn_slice_257 = ttnn.slice(
        ttnn_typecast_173,
        [0, 12288],
        [2, 15360],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_557 = ttnn.reshape(
        ttnn_slice_257,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_257, False)
    ttnn_add_156 = ttnn.add(
        ttnn_reshape_557,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_557, False)
    ttnn_multiply_98 = ttnn.multiply(
        ttnn_layer_norm_31,
        ttnn_add_156,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_156, False)
    ttnn.deallocate(ttnn_layer_norm_31, False)
    ttnn_slice_258 = ttnn.slice(
        ttnn_typecast_173,
        [0, 9216],
        [2, 12288],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_173, False)
    ttnn_reshape_558 = ttnn.reshape(
        ttnn_slice_258,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_258, False)
    ttnn_add_157 = ttnn.add(
        ttnn_multiply_98,
        ttnn_reshape_558,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_558, False)
    ttnn.deallocate(ttnn_multiply_98, False)
    ttnn_reshape_559 = ttnn.reshape(
        ttnn_add_157,
        [8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_157, False)
    ttnn_linear_35 = ttnn.linear(
        ttnn_reshape_559,
        self.weights["transformer.transformer_blocks.7.ff.net.0.proj.weight"],
        bias=self.weights["transformer.transformer_blocks.7.ff.net.0.proj.bias"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation="gelu",
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_559, False)
    ttnn_matmul_41 = ttnn.matmul(
        ttnn_linear_35,
        self.weights["transformer.transformer_blocks.7.ff.net.2.weight"],
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
    ttnn.deallocate(ttnn_linear_35, False)
    ttnn_reshape_560 = ttnn.reshape(
        ttnn_matmul_41,
        [1, 1, 8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_41, False)
    ttnn_reduce_scatter_47 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_560,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_560, False)
    ttnn_reshape_561 = ttnn.reshape(
        ttnn_reduce_scatter_47,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_47, False)
    ttnn_all_gather_47 = ttnn.all_gather(
        input_tensor=ttnn_reshape_561,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_561, False)
    ttnn_add_158 = ttnn.add(
        ttnn_all_gather_47,
        self.weights["transformer.transformer_blocks.7.ff.net.2.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_47, False)
    ttnn_reshape_562 = ttnn.reshape(
        ttnn_add_158,
        [2, 4096, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_158, False)
    ttnn_multiply_99 = ttnn.multiply(
        ttnn_reshape_556,
        ttnn_reshape_562,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_562, False)
    ttnn.deallocate(ttnn_reshape_556, False)
    ttnn_add_159 = ttnn.add(
        ttnn_add_155,
        ttnn_multiply_99,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_99, False)
    ttnn.deallocate(ttnn_add_155, False)
    ttnn_concat_162 = ttnn.concat(
        [ttnn_concat_161, ttnn_add_159],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_159, False)
    ttnn.deallocate(ttnn_concat_161, False)
    ttnn_layer_norm_32 = ttnn.layer_norm(
        ttnn_concat_162,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_reshape_563 = ttnn.reshape(
        ttnn_slice_39,
        [1, 1, 2, 9216],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_39, False)
    ttnn_reduce_scatter_48 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_563,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_563, False)
    ttnn_reshape_564 = ttnn.reshape(
        ttnn_reduce_scatter_48,
        [2, 2304],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_48, False)
    ttnn_all_gather_48 = ttnn.all_gather(
        input_tensor=ttnn_reshape_564,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_564, False)
    ttnn_add_160 = ttnn.add(
        ttnn_all_gather_48,
        self.weights["transformer.single_transformer_blocks.0.norm.linear.bias.f32"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_48, False)
    ttnn_typecast_180 = ttnn.typecast(
        ttnn_add_160,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_160, False)
    ttnn_slice_259 = ttnn.slice(
        ttnn_typecast_180,
        [0, 6144],
        [2, 9216],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_565 = ttnn.reshape(
        ttnn_slice_259,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_259, False)
    ttnn_slice_260 = ttnn.slice(
        ttnn_typecast_180,
        [0, 3072],
        [2, 6144],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_566 = ttnn.reshape(
        ttnn_slice_260,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_260, False)
    ttnn_add_161 = ttnn.add(
        ttnn_reshape_566,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_566, False)
    ttnn_multiply_100 = ttnn.multiply(
        ttnn_layer_norm_32,
        ttnn_add_161,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_161, False)
    ttnn.deallocate(ttnn_layer_norm_32, False)
    ttnn_slice_261 = ttnn.slice(
        ttnn_typecast_180,
        [0, 0],
        [2, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_180, False)
    ttnn_reshape_567 = ttnn.reshape(
        ttnn_slice_261,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_261, False)
    ttnn_add_162 = ttnn.add(
        ttnn_multiply_100,
        ttnn_reshape_567,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_567, False)
    ttnn.deallocate(ttnn_multiply_100, False)
    ttnn_reshape_568 = ttnn.reshape(
        ttnn_add_162,
        [8282, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_162, False)
    ttnn_linear_36 = ttnn.linear(
        ttnn_reshape_568,
        self.weights["transformer.single_transformer_blocks.0.fused_attn_to_q_to_k_to_v_proj_mlp.weight"],
        bias=self.weights["transformer.single_transformer_blocks.0.fused_attn_to_q_to_k_to_v_proj_mlp.bias"],
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
    ttnn.deallocate(ttnn_reshape_568, False)
    ttnn_slice_262 = ttnn.slice(
        ttnn_linear_36,
        [0, 0],
        [8282, 768],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_263 = ttnn.slice(
        ttnn_linear_36,
        [0, 768],
        [8282, 1536],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_264 = ttnn.slice(
        ttnn_linear_36,
        [0, 1536],
        [8282, 2304],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_265 = ttnn.slice(
        ttnn_linear_36,
        [0, 2304],
        [8282, 5376],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_36, False)
    ttnn_reshape_569 = ttnn.reshape(
        ttnn_slice_262,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_262, False)
    ttnn_rms_norm_32 = ttnn.rms_norm(
        ttnn_reshape_569,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.single_transformer_blocks.0.attn.norm_q.weight"],
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
    ttnn.deallocate(ttnn_reshape_569, False)
    ttnn_typecast_181 = ttnn.typecast(
        ttnn_rms_norm_32,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_297 = ttnn.permute(
        ttnn_typecast_181,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_181, False)
    ttnn_multiply_101 = ttnn.multiply(
        ttnn_permute_297,
        ttnn_reshape_203,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_297, False)
    ttnn_reshape_570 = ttnn.reshape(
        ttnn_rms_norm_32,
        [2, 4141, 6, 64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_32, False)
    ttnn_slice_266 = ttnn.slice(
        ttnn_reshape_570,
        [0, 0, 0, 0, 1],
        [2, 4141, 6, 64, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_16 = ttnn.neg(
        ttnn_slice_266,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_266, False)
    ttnn_slice_267 = ttnn.slice(
        ttnn_reshape_570,
        [0, 0, 0, 0, 0],
        [2, 4141, 6, 64, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_570, False)
    ttnn_concat_163 = ttnn.concat(
        [ttnn_neg_16, ttnn_slice_267],
        4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_267, False)
    ttnn.deallocate(ttnn_neg_16, False)
    ttnn_typecast_182 = ttnn.typecast(
        ttnn_concat_163,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_163, False)
    ttnn_reshape_571 = ttnn.reshape(
        ttnn_typecast_182,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_182, False)
    ttnn_permute_298 = ttnn.permute(
        ttnn_reshape_571,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_571, False)
    ttnn_multiply_102 = ttnn.multiply(
        ttnn_permute_298,
        ttnn_reshape_209,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_298, False)
    ttnn_add_163 = ttnn.add(
        ttnn_multiply_101,
        ttnn_multiply_102,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_102, False)
    ttnn.deallocate(ttnn_multiply_101, False)
    ttnn_typecast_183 = ttnn.typecast(
        ttnn_add_163,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_163, False)
    ttnn_reshape_572 = ttnn.reshape(
        ttnn_slice_263,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_263, False)
    ttnn_rms_norm_33 = ttnn.rms_norm(
        ttnn_reshape_572,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.single_transformer_blocks.0.attn.norm_k.weight"],
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
    ttnn.deallocate(ttnn_reshape_572, False)
    ttnn_typecast_184 = ttnn.typecast(
        ttnn_rms_norm_33,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_299 = ttnn.permute(
        ttnn_typecast_184,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_184, False)
    ttnn_multiply_103 = ttnn.multiply(
        ttnn_permute_299,
        ttnn_reshape_203,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_299, False)
    ttnn_reshape_573 = ttnn.reshape(
        ttnn_rms_norm_33,
        [2, 4141, 6, 64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_33, False)
    ttnn_slice_268 = ttnn.slice(
        ttnn_reshape_573,
        [0, 0, 0, 0, 1],
        [2, 4141, 6, 64, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_17 = ttnn.neg(
        ttnn_slice_268,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_268, False)
    ttnn_slice_269 = ttnn.slice(
        ttnn_reshape_573,
        [0, 0, 0, 0, 0],
        [2, 4141, 6, 64, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_573, False)
    ttnn_concat_164 = ttnn.concat(
        [ttnn_neg_17, ttnn_slice_269],
        4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_269, False)
    ttnn.deallocate(ttnn_neg_17, False)
    ttnn_typecast_185 = ttnn.typecast(
        ttnn_concat_164,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_164, False)
    ttnn_reshape_574 = ttnn.reshape(
        ttnn_typecast_185,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_185, False)
    ttnn_permute_300 = ttnn.permute(
        ttnn_reshape_574,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_574, False)
    ttnn_multiply_104 = ttnn.multiply(
        ttnn_permute_300,
        ttnn_reshape_209,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_300, False)
    ttnn_add_164 = ttnn.add(
        ttnn_multiply_103,
        ttnn_multiply_104,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_104, False)
    ttnn.deallocate(ttnn_multiply_103, False)
    ttnn_typecast_186 = ttnn.typecast(
        ttnn_add_164,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_164, False)
    ttnn_reshape_575 = ttnn.reshape(
        ttnn_slice_264,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_264, False)
    ttnn_permute_301 = ttnn.permute(
        ttnn_reshape_575,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_575, False)
    ttnn_transformer_scaled_dot_product_attention_8 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_typecast_183,
            ttnn_typecast_186,
            ttnn_permute_301,
            attn_mask=ttnn_to_layout_590,
            is_causal=False,
            scale=None,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
            compute_kernel_config=None,
            attention_sink=None,
        )
    )
    ttnn.deallocate(ttnn_permute_301, False)
    ttnn.deallocate(ttnn_typecast_186, False)
    ttnn.deallocate(ttnn_typecast_183, False)
    ttnn_transformer_concatenate_heads_8 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_8,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_8, False)
    ttnn_gelu_0 = ttnn.gelu(
        ttnn_slice_265,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_265, False)
    ttnn_reshape_576 = ttnn.reshape(
        ttnn_gelu_0,
        [2, 4141, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_gelu_0, False)
    ttnn_all_gather_49 = ttnn.all_gather(
        input_tensor=ttnn_transformer_concatenate_heads_8,
        dim=2,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_8, False)
    ttnn_all_gather_50 = ttnn.all_gather(
        input_tensor=ttnn_reshape_576,
        dim=2,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_576, False)
    ttnn_concat_165 = ttnn.concat(
        [ttnn_all_gather_49, ttnn_all_gather_50],
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_50, False)
    ttnn.deallocate(ttnn_all_gather_49, False)
    ttnn_to_layout_599 = ttnn.to_layout(
        ttnn_concat_165,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_165, False)
    ttnn_mesh_partition_2 = ttnn.mesh_partition(
        input_tensor=ttnn_to_layout_599,
        dim=2,
        cluster_axis=1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_599, False)
    ttnn_to_layout_600 = ttnn.to_layout(
        ttnn_mesh_partition_2,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mesh_partition_2, False)
    ttnn_reshape_577 = ttnn.reshape(
        ttnn_to_layout_600,
        [8282, 3840],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_600, False)
    ttnn_matmul_42 = ttnn.matmul(
        ttnn_reshape_577,
        self.weights["transformer.single_transformer_blocks.0.proj_out.weight"],
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
    ttnn.deallocate(ttnn_reshape_577, False)
    ttnn_reshape_578 = ttnn.reshape(
        ttnn_matmul_42,
        [1, 1, 8282, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_42, False)
    ttnn_reduce_scatter_49 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_578,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_578, False)
    ttnn_reshape_579 = ttnn.reshape(
        ttnn_reduce_scatter_49,
        [8282, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_49, False)
    ttnn_all_gather_51 = ttnn.all_gather(
        input_tensor=ttnn_reshape_579,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_579, False)
    ttnn_add_165 = ttnn.add(
        ttnn_all_gather_51,
        self.weights["transformer.single_transformer_blocks.0.proj_out.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_51, False)
    ttnn_reshape_580 = ttnn.reshape(
        ttnn_add_165,
        [2, 4141, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_165, False)
    ttnn_multiply_105 = ttnn.multiply(
        ttnn_reshape_565,
        ttnn_reshape_580,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_580, False)
    ttnn.deallocate(ttnn_reshape_565, False)
    ttnn_add_166 = ttnn.add(
        ttnn_concat_162,
        ttnn_multiply_105,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_105, False)
    ttnn.deallocate(ttnn_concat_162, False)
    ttnn_slice_270 = ttnn.slice(
        ttnn_add_166,
        [0, 0, 0],
        [2, 45, 1536],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_601 = ttnn.to_layout(
        text_encoder_layer_9,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_581 = ttnn.reshape(
        ttnn_to_layout_601,
        [90, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_601, False)
    return ttnn_add_166, ttnn_reshape_581, ttnn_slice_270


def _stb_forward_37(self, ttnn_add_418, ttnn_reshape_1301, ttnn_reshape_203, ttnn_reshape_209, ttnn_slice_2, ttnn_slice_738, ttnn_to_layout_590, var_1):
    ttnn_matmul_115 = ttnn.matmul(
        ttnn_reshape_1301,
        self.weights["transformer.caption_projection.45.linear.weight"],
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
    ttnn.deallocate(ttnn_reshape_1301, False)
    ttnn_reshape_1302 = ttnn.reshape(
        ttnn_matmul_115,
        [2, 45, 1536],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_115, False)
    ttnn_concat_346 = ttnn.concat(
        [ttnn_slice_738, ttnn_reshape_1302],
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_1302, False)
    ttnn.deallocate(ttnn_slice_738, False)
    ttnn_slice_739 = ttnn.slice(
        ttnn_add_418,
        [0, 45, 0],
        [2, 4141, 3072],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_418, False)
    ttnn_concat_347 = ttnn.concat(
        [ttnn_concat_346, ttnn_slice_739],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_739, False)
    ttnn.deallocate(ttnn_concat_346, False)
    ttnn_layer_norm_69 = ttnn.layer_norm(
        ttnn_concat_347,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_reshape_1303 = ttnn.reshape(
        ttnn_slice_2,
        [1, 1, 2, 9216],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_2, False)
    ttnn_reduce_scatter_122 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_1303,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_1303, False)
    ttnn_reshape_1304 = ttnn.reshape(
        ttnn_reduce_scatter_122,
        [2, 2304],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_122, False)
    ttnn_all_gather_196 = ttnn.all_gather(
        input_tensor=ttnn_reshape_1304,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_1304, False)
    ttnn_add_419 = ttnn.add(
        ttnn_all_gather_196,
        self.weights["transformer.single_transformer_blocks.37.norm.linear.bias.f32"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_196, False)
    ttnn_typecast_439 = ttnn.typecast(
        ttnn_add_419,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_419, False)
    ttnn_slice_740 = ttnn.slice(
        ttnn_typecast_439,
        [0, 6144],
        [2, 9216],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_1305 = ttnn.reshape(
        ttnn_slice_740,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_740, False)
    ttnn_slice_741 = ttnn.slice(
        ttnn_typecast_439,
        [0, 3072],
        [2, 6144],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_1306 = ttnn.reshape(
        ttnn_slice_741,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_741, False)
    ttnn_add_420 = ttnn.add(
        ttnn_reshape_1306,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_1306, False)
    ttnn_multiply_322 = ttnn.multiply(
        ttnn_layer_norm_69,
        ttnn_add_420,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_420, False)
    ttnn.deallocate(ttnn_layer_norm_69, False)
    ttnn_slice_742 = ttnn.slice(
        ttnn_typecast_439,
        [0, 0],
        [2, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_439, False)
    ttnn_reshape_1307 = ttnn.reshape(
        ttnn_slice_742,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_742, False)
    ttnn_add_421 = ttnn.add(
        ttnn_multiply_322,
        ttnn_reshape_1307,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_1307, False)
    ttnn.deallocate(ttnn_multiply_322, False)
    ttnn_reshape_1308 = ttnn.reshape(
        ttnn_add_421,
        [8282, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_421, False)
    ttnn_linear_73 = ttnn.linear(
        ttnn_reshape_1308,
        self.weights["transformer.single_transformer_blocks.37.fused_attn_to_q_to_k_to_v_proj_mlp.weight"],
        bias=self.weights["transformer.single_transformer_blocks.37.fused_attn_to_q_to_k_to_v_proj_mlp.bias"],
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
    ttnn.deallocate(ttnn_reshape_1308, False)
    ttnn_slice_743 = ttnn.slice(
        ttnn_linear_73,
        [0, 0],
        [8282, 768],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_744 = ttnn.slice(
        ttnn_linear_73,
        [0, 768],
        [8282, 1536],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_745 = ttnn.slice(
        ttnn_linear_73,
        [0, 1536],
        [8282, 2304],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_746 = ttnn.slice(
        ttnn_linear_73,
        [0, 2304],
        [8282, 5376],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_73, False)
    ttnn_reshape_1309 = ttnn.reshape(
        ttnn_slice_743,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_743, False)
    ttnn_rms_norm_106 = ttnn.rms_norm(
        ttnn_reshape_1309,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.single_transformer_blocks.37.attn.norm_q.weight"],
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
    ttnn.deallocate(ttnn_reshape_1309, False)
    ttnn_typecast_440 = ttnn.typecast(
        ttnn_rms_norm_106,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_482 = ttnn.permute(
        ttnn_typecast_440,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_440, False)
    ttnn_multiply_323 = ttnn.multiply(
        ttnn_permute_482,
        ttnn_reshape_203,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_482, False)
    ttnn_reshape_1310 = ttnn.reshape(
        ttnn_rms_norm_106,
        [2, 4141, 6, 64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_106, False)
    ttnn_slice_747 = ttnn.slice(
        ttnn_reshape_1310,
        [0, 0, 0, 0, 1],
        [2, 4141, 6, 64, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_90 = ttnn.neg(
        ttnn_slice_747,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_747, False)
    ttnn_slice_748 = ttnn.slice(
        ttnn_reshape_1310,
        [0, 0, 0, 0, 0],
        [2, 4141, 6, 64, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_1310, False)
    ttnn_concat_348 = ttnn.concat(
        [ttnn_neg_90, ttnn_slice_748],
        4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_748, False)
    ttnn.deallocate(ttnn_neg_90, False)
    ttnn_typecast_441 = ttnn.typecast(
        ttnn_concat_348,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_348, False)
    ttnn_reshape_1311 = ttnn.reshape(
        ttnn_typecast_441,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_441, False)
    ttnn_permute_483 = ttnn.permute(
        ttnn_reshape_1311,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_1311, False)
    ttnn_multiply_324 = ttnn.multiply(
        ttnn_permute_483,
        ttnn_reshape_209,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_483, False)
    ttnn_add_422 = ttnn.add(
        ttnn_multiply_323,
        ttnn_multiply_324,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_324, False)
    ttnn.deallocate(ttnn_multiply_323, False)
    ttnn_typecast_442 = ttnn.typecast(
        ttnn_add_422,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_422, False)
    ttnn_reshape_1312 = ttnn.reshape(
        ttnn_slice_744,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_744, False)
    ttnn_rms_norm_107 = ttnn.rms_norm(
        ttnn_reshape_1312,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.single_transformer_blocks.37.attn.norm_k.weight"],
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
    ttnn.deallocate(ttnn_reshape_1312, False)
    ttnn_typecast_443 = ttnn.typecast(
        ttnn_rms_norm_107,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_484 = ttnn.permute(
        ttnn_typecast_443,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_443, False)
    ttnn_multiply_325 = ttnn.multiply(
        ttnn_permute_484,
        ttnn_reshape_203,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_484, False)
    ttnn.deallocate(ttnn_reshape_203, False)
    ttnn_reshape_1313 = ttnn.reshape(
        ttnn_rms_norm_107,
        [2, 4141, 6, 64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_107, False)
    ttnn_slice_749 = ttnn.slice(
        ttnn_reshape_1313,
        [0, 0, 0, 0, 1],
        [2, 4141, 6, 64, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_91 = ttnn.neg(
        ttnn_slice_749,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_749, False)
    ttnn_slice_750 = ttnn.slice(
        ttnn_reshape_1313,
        [0, 0, 0, 0, 0],
        [2, 4141, 6, 64, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_1313, False)
    ttnn_concat_349 = ttnn.concat(
        [ttnn_neg_91, ttnn_slice_750],
        4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_750, False)
    ttnn.deallocate(ttnn_neg_91, False)
    ttnn_typecast_444 = ttnn.typecast(
        ttnn_concat_349,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_349, False)
    ttnn_reshape_1314 = ttnn.reshape(
        ttnn_typecast_444,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_444, False)
    ttnn_permute_485 = ttnn.permute(
        ttnn_reshape_1314,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_1314, False)
    ttnn_multiply_326 = ttnn.multiply(
        ttnn_permute_485,
        ttnn_reshape_209,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_485, False)
    ttnn.deallocate(ttnn_reshape_209, False)
    ttnn_add_423 = ttnn.add(
        ttnn_multiply_325,
        ttnn_multiply_326,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_326, False)
    ttnn.deallocate(ttnn_multiply_325, False)
    ttnn_typecast_445 = ttnn.typecast(
        ttnn_add_423,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_423, False)
    ttnn_reshape_1315 = ttnn.reshape(
        ttnn_slice_745,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_745, False)
    ttnn_permute_486 = ttnn.permute(
        ttnn_reshape_1315,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_1315, False)
    ttnn_transformer_scaled_dot_product_attention_45 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_typecast_442,
            ttnn_typecast_445,
            ttnn_permute_486,
            attn_mask=ttnn_to_layout_590,
            is_causal=False,
            scale=None,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
            compute_kernel_config=None,
            attention_sink=None,
        )
    )
    ttnn.deallocate(ttnn_permute_486, False)
    ttnn.deallocate(ttnn_typecast_445, False)
    ttnn.deallocate(ttnn_typecast_442, False)
    ttnn.deallocate(ttnn_to_layout_590, False)
    ttnn_transformer_concatenate_heads_45 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_45,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_45, False)
    ttnn_gelu_37 = ttnn.gelu(
        ttnn_slice_746,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_746, False)
    ttnn_reshape_1316 = ttnn.reshape(
        ttnn_gelu_37,
        [2, 4141, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_gelu_37, False)
    ttnn_all_gather_197 = ttnn.all_gather(
        input_tensor=ttnn_transformer_concatenate_heads_45,
        dim=2,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_45, False)
    ttnn_all_gather_198 = ttnn.all_gather(
        input_tensor=ttnn_reshape_1316,
        dim=2,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_1316, False)
    ttnn_concat_350 = ttnn.concat(
        [ttnn_all_gather_197, ttnn_all_gather_198],
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_198, False)
    ttnn.deallocate(ttnn_all_gather_197, False)
    ttnn_to_layout_710 = ttnn.to_layout(
        ttnn_concat_350,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_350, False)
    ttnn_mesh_partition_39 = ttnn.mesh_partition(
        input_tensor=ttnn_to_layout_710,
        dim=2,
        cluster_axis=1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_710, False)
    ttnn_to_layout_711 = ttnn.to_layout(
        ttnn_mesh_partition_39,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mesh_partition_39, False)
    ttnn_reshape_1317 = ttnn.reshape(
        ttnn_to_layout_711,
        [8282, 3840],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_711, False)
    ttnn_matmul_116 = ttnn.matmul(
        ttnn_reshape_1317,
        self.weights["transformer.single_transformer_blocks.37.proj_out.weight"],
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
    ttnn.deallocate(ttnn_reshape_1317, False)
    ttnn_reshape_1318 = ttnn.reshape(
        ttnn_matmul_116,
        [1, 1, 8282, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_116, False)
    ttnn_reduce_scatter_123 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_1318,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_1318, False)
    ttnn_reshape_1319 = ttnn.reshape(
        ttnn_reduce_scatter_123,
        [8282, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_123, False)
    ttnn_all_gather_199 = ttnn.all_gather(
        input_tensor=ttnn_reshape_1319,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_1319, False)
    ttnn_add_424 = ttnn.add(
        ttnn_all_gather_199,
        self.weights["transformer.single_transformer_blocks.37.proj_out.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_199, False)
    ttnn_reshape_1320 = ttnn.reshape(
        ttnn_add_424,
        [2, 4141, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_424, False)
    ttnn_multiply_327 = ttnn.multiply(
        ttnn_reshape_1305,
        ttnn_reshape_1320,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_1320, False)
    ttnn.deallocate(ttnn_reshape_1305, False)
    ttnn_add_425 = ttnn.add(
        ttnn_concat_347,
        ttnn_multiply_327,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_327, False)
    ttnn.deallocate(ttnn_concat_347, False)
    ttnn_slice_751 = ttnn.slice(
        ttnn_add_425,
        [0, 45, 0],
        [2, 4141, 3072],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_425, False)
    ttnn_layer_norm_70 = ttnn.layer_norm(
        ttnn_slice_751,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn.deallocate(ttnn_slice_751, False)
    return ttnn_layer_norm_70


