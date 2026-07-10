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
        ttnn_add_22, ttnn_reshape_273, ttnn_slice_103, ttnn_transformer_concatenate_heads_1, ttnn_typecast_125 = self.transformer_blocks[1](text_encoder_layers[2], ttnn_reshape_193, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_227, ttnn_slice_52, ttnn_slice_53, ttnn_slice_78, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_0, ttnn_typecast_116, var_1)
        ttnn_add_42, ttnn_reshape_319, ttnn_slice_128, ttnn_transformer_concatenate_heads_2, ttnn_typecast_133 = self.transformer_blocks[2](text_encoder_layers[3], ttnn_add_22, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_273, ttnn_slice_103, ttnn_slice_50, ttnn_slice_51, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_1, ttnn_typecast_125, var_1)
        ttnn_add_62, ttnn_reshape_365, ttnn_slice_153, ttnn_transformer_concatenate_heads_3, ttnn_typecast_141 = self.transformer_blocks[3](text_encoder_layers[4], ttnn_add_42, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_319, ttnn_slice_128, ttnn_slice_48, ttnn_slice_49, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_2, ttnn_typecast_133, var_1)
        ttnn_add_82, ttnn_reshape_411, ttnn_slice_178, ttnn_transformer_concatenate_heads_4, ttnn_typecast_149 = self.transformer_blocks[4](text_encoder_layers[5], ttnn_add_62, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_365, ttnn_slice_153, ttnn_slice_46, ttnn_slice_47, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_3, ttnn_typecast_141, var_1)
        ttnn_add_102, ttnn_reshape_457, ttnn_slice_203, ttnn_transformer_concatenate_heads_5, ttnn_typecast_157 = self.transformer_blocks[5](text_encoder_layers[6], ttnn_add_82, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_411, ttnn_slice_178, ttnn_slice_44, ttnn_slice_45, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_4, ttnn_typecast_149, var_1)
        ttnn_add_122, ttnn_reshape_503, ttnn_slice_228, ttnn_transformer_concatenate_heads_6, ttnn_typecast_165 = self.transformer_blocks[6](text_encoder_layers[7], ttnn_add_102, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_457, ttnn_slice_203, ttnn_slice_42, ttnn_slice_43, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_5, ttnn_typecast_157, var_1)
        ttnn_add_142, ttnn_reshape_549, ttnn_slice_253, ttnn_transformer_concatenate_heads_7, ttnn_typecast_173 = self.transformer_blocks[7](text_encoder_layers[8], ttnn_add_122, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_503, ttnn_slice_228, ttnn_slice_40, ttnn_slice_41, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_6, ttnn_typecast_165, var_1)
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
        return _TB_FORWARDS[self.block_idx](self, *args)


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


def _tb_forward_1(self, text_encoder_layer_2, ttnn_reshape_193, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_227, ttnn_slice_52, ttnn_slice_53, ttnn_slice_78, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_0, ttnn_typecast_116, var_1):
    ttnn_matmul_4 = ttnn.matmul(
        ttnn_reshape_227,
        self.weights["transformer.caption_projection.1.linear.weight"],
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
    ttnn.deallocate(ttnn_reshape_227, False)
    ttnn_reshape_228 = ttnn.reshape(
        ttnn_matmul_4,
        [2, 45, 1536],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_4, False)
    ttnn_concat_119 = ttnn.concat(
        [ttnn_slice_78, ttnn_reshape_228],
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_228, False)
    ttnn.deallocate(ttnn_slice_78, False)
    ttnn_layer_norm_3 = ttnn.layer_norm(
        ttnn_concat_119,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_reshape_229 = ttnn.reshape(
        ttnn_slice_53,
        [1, 1, 2, 18432],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_53, False)
    ttnn_reduce_scatter_4 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_229,
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
    ttnn.deallocate(ttnn_reshape_229, False)
    ttnn_reshape_230 = ttnn.reshape(
        ttnn_reduce_scatter_4,
        [2, 4608],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_4, False)
    ttnn_all_gather_4 = ttnn.all_gather(
        input_tensor=ttnn_reshape_230,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_230, False)
    ttnn_add_14 = ttnn.add(
        ttnn_all_gather_4,
        self.weights["transformer.transformer_blocks.1.norm1_context.linear.bias.f32"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_4, False)
    ttnn_typecast_124 = ttnn.typecast(
        ttnn_add_14,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_14, False)
    ttnn_slice_79 = ttnn.slice(
        ttnn_typecast_124,
        [0, 6144],
        [2, 9216],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_231 = ttnn.reshape(
        ttnn_slice_79,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_79, False)
    ttnn_slice_80 = ttnn.slice(
        ttnn_typecast_124,
        [0, 3072],
        [2, 6144],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_232 = ttnn.reshape(
        ttnn_slice_80,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_80, False)
    ttnn_add_15 = ttnn.add(
        ttnn_reshape_232,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_232, False)
    ttnn_multiply_13 = ttnn.multiply(
        ttnn_layer_norm_3,
        ttnn_add_15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_15, False)
    ttnn.deallocate(ttnn_layer_norm_3, False)
    ttnn_slice_81 = ttnn.slice(
        ttnn_typecast_124,
        [0, 0],
        [2, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_233 = ttnn.reshape(
        ttnn_slice_81,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_81, False)
    ttnn_add_16 = ttnn.add(
        ttnn_multiply_13,
        ttnn_reshape_233,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_233, False)
    ttnn.deallocate(ttnn_multiply_13, False)
    ttnn_reshape_234 = ttnn.reshape(
        ttnn_add_16,
        [90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_16, False)
    ttnn_linear_7 = ttnn.linear(
        ttnn_reshape_234,
        self.weights["transformer.transformer_blocks.1.attn.fused_add_q_proj_add_k_proj_add_v_proj.weight"],
        bias=self.weights["transformer.transformer_blocks.1.attn.fused_add_q_proj_add_k_proj_add_v_proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_234, False)
    ttnn_slice_82 = ttnn.slice(
        ttnn_linear_7,
        [0, 0],
        [90, 768],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_83 = ttnn.slice(
        ttnn_linear_7,
        [0, 768],
        [90, 1536],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_84 = ttnn.slice(
        ttnn_linear_7,
        [0, 1536],
        [90, 2304],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_7, False)
    ttnn_reshape_235 = ttnn.reshape(
        ttnn_slice_82,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_82, False)
    ttnn_rms_norm_4 = ttnn.rms_norm(
        ttnn_reshape_235,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.1.attn.norm_added_q.weight"],
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
    ttnn.deallocate(ttnn_reshape_235, False)
    ttnn_slice_85 = ttnn.slice(
        ttnn_typecast_116,
        [0, 6144],
        [2, 9216],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_236 = ttnn.reshape(
        ttnn_slice_85,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_85, False)
    ttnn_slice_86 = ttnn.slice(
        ttnn_transformer_concatenate_heads_0,
        [0, 45, 0],
        [2, 4141, 768],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_0, False)
    ttnn_reshape_237 = ttnn.reshape(
        ttnn_slice_86,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_86, False)
    ttnn_matmul_5 = ttnn.matmul(
        ttnn_reshape_237,
        self.weights["transformer.transformer_blocks.0.attn.to_out.0.weight"],
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
    ttnn.deallocate(ttnn_reshape_237, False)
    ttnn_reshape_238 = ttnn.reshape(
        ttnn_matmul_5,
        [1, 1, 8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_5, False)
    ttnn_reduce_scatter_5 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_238,
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
    ttnn.deallocate(ttnn_reshape_238, False)
    ttnn_reshape_239 = ttnn.reshape(
        ttnn_reduce_scatter_5,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_5, False)
    ttnn_all_gather_5 = ttnn.all_gather(
        input_tensor=ttnn_reshape_239,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_239, False)
    ttnn_add_17 = ttnn.add(
        ttnn_all_gather_5,
        self.weights["transformer.transformer_blocks.0.attn.to_out.0.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_5, False)
    ttnn_reshape_240 = ttnn.reshape(
        ttnn_add_17,
        [2, 4096, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_17, False)
    ttnn_multiply_14 = ttnn.multiply(
        ttnn_reshape_236,
        ttnn_reshape_240,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_240, False)
    ttnn.deallocate(ttnn_reshape_236, False)
    ttnn_add_18 = ttnn.add(
        ttnn_reshape_193,
        ttnn_multiply_14,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_14, False)
    ttnn.deallocate(ttnn_reshape_193, False)
    ttnn_layer_norm_4 = ttnn.layer_norm(
        ttnn_add_18,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_slice_87 = ttnn.slice(
        ttnn_typecast_116,
        [0, 15360],
        [2, 18432],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_241 = ttnn.reshape(
        ttnn_slice_87,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_87, False)
    ttnn_slice_88 = ttnn.slice(
        ttnn_typecast_116,
        [0, 12288],
        [2, 15360],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_242 = ttnn.reshape(
        ttnn_slice_88,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_88, False)
    ttnn_add_19 = ttnn.add(
        ttnn_reshape_242,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_242, False)
    ttnn_multiply_15 = ttnn.multiply(
        ttnn_layer_norm_4,
        ttnn_add_19,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_19, False)
    ttnn.deallocate(ttnn_layer_norm_4, False)
    ttnn_slice_89 = ttnn.slice(
        ttnn_typecast_116,
        [0, 9216],
        [2, 12288],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_116, False)
    ttnn_reshape_243 = ttnn.reshape(
        ttnn_slice_89,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_89, False)
    ttnn_add_20 = ttnn.add(
        ttnn_multiply_15,
        ttnn_reshape_243,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_243, False)
    ttnn.deallocate(ttnn_multiply_15, False)
    ttnn_reshape_244 = ttnn.reshape(
        ttnn_add_20,
        [8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_20, False)
    ttnn_linear_8 = ttnn.linear(
        ttnn_reshape_244,
        self.weights["transformer.transformer_blocks.0.ff.net.0.proj.weight"],
        bias=self.weights["transformer.transformer_blocks.0.ff.net.0.proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_244, False)
    ttnn_matmul_6 = ttnn.matmul(
        ttnn_linear_8,
        self.weights["transformer.transformer_blocks.0.ff.net.2.weight"],
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
    ttnn.deallocate(ttnn_linear_8, False)
    ttnn_reshape_245 = ttnn.reshape(
        ttnn_matmul_6,
        [1, 1, 8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_6, False)
    ttnn_reduce_scatter_6 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_245,
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
    ttnn.deallocate(ttnn_reshape_245, False)
    ttnn_reshape_246 = ttnn.reshape(
        ttnn_reduce_scatter_6,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_6, False)
    ttnn_all_gather_6 = ttnn.all_gather(
        input_tensor=ttnn_reshape_246,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_246, False)
    ttnn_add_21 = ttnn.add(
        ttnn_all_gather_6,
        self.weights["transformer.transformer_blocks.0.ff.net.2.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_6, False)
    ttnn_reshape_247 = ttnn.reshape(
        ttnn_add_21,
        [2, 4096, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_21, False)
    ttnn_multiply_16 = ttnn.multiply(
        ttnn_reshape_241,
        ttnn_reshape_247,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_247, False)
    ttnn.deallocate(ttnn_reshape_241, False)
    ttnn_add_22 = ttnn.add(
        ttnn_add_18,
        ttnn_multiply_16,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_16, False)
    ttnn.deallocate(ttnn_add_18, False)
    ttnn_layer_norm_5 = ttnn.layer_norm(
        ttnn_add_22,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_reshape_248 = ttnn.reshape(
        ttnn_slice_52,
        [1, 1, 2, 18432],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_52, False)
    ttnn_reduce_scatter_7 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_248,
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
    ttnn.deallocate(ttnn_reshape_248, False)
    ttnn_reshape_249 = ttnn.reshape(
        ttnn_reduce_scatter_7,
        [2, 4608],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_7, False)
    ttnn_all_gather_7 = ttnn.all_gather(
        input_tensor=ttnn_reshape_249,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_249, False)
    ttnn_add_23 = ttnn.add(
        ttnn_all_gather_7,
        self.weights["transformer.transformer_blocks.1.norm1.linear.bias.f32"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_7, False)
    ttnn_typecast_125 = ttnn.typecast(
        ttnn_add_23,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_23, False)
    ttnn_slice_90 = ttnn.slice(
        ttnn_typecast_125,
        [0, 3072],
        [2, 6144],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_250 = ttnn.reshape(
        ttnn_slice_90,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_90, False)
    ttnn_add_24 = ttnn.add(
        ttnn_reshape_250,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_250, False)
    ttnn_multiply_17 = ttnn.multiply(
        ttnn_layer_norm_5,
        ttnn_add_24,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_24, False)
    ttnn.deallocate(ttnn_layer_norm_5, False)
    ttnn_slice_91 = ttnn.slice(
        ttnn_typecast_125,
        [0, 0],
        [2, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_251 = ttnn.reshape(
        ttnn_slice_91,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_91, False)
    ttnn_add_25 = ttnn.add(
        ttnn_multiply_17,
        ttnn_reshape_251,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_251, False)
    ttnn.deallocate(ttnn_multiply_17, False)
    ttnn_reshape_252 = ttnn.reshape(
        ttnn_add_25,
        [8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_25, False)
    ttnn_linear_9 = ttnn.linear(
        ttnn_reshape_252,
        self.weights["transformer.transformer_blocks.1.attn.fused_to_q_to_k_to_v.weight"],
        bias=self.weights["transformer.transformer_blocks.1.attn.fused_to_q_to_k_to_v.bias"],
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
    ttnn.deallocate(ttnn_reshape_252, False)
    ttnn_slice_92 = ttnn.slice(
        ttnn_linear_9,
        [0, 0],
        [8192, 768],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_93 = ttnn.slice(
        ttnn_linear_9,
        [0, 768],
        [8192, 1536],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_94 = ttnn.slice(
        ttnn_linear_9,
        [0, 1536],
        [8192, 2304],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_9, False)
    ttnn_reshape_253 = ttnn.reshape(
        ttnn_slice_92,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_92, False)
    ttnn_rms_norm_5 = ttnn.rms_norm(
        ttnn_reshape_253,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.1.attn.norm_q.weight"],
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
    ttnn.deallocate(ttnn_reshape_253, False)
    ttnn_concat_120 = ttnn.concat(
        [ttnn_rms_norm_4, ttnn_rms_norm_5],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_5, False)
    ttnn.deallocate(ttnn_rms_norm_4, False)
    ttnn_typecast_126 = ttnn.typecast(
        ttnn_concat_120,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_262 = ttnn.permute(
        ttnn_typecast_126,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_126, False)
    ttnn_multiply_18 = ttnn.multiply(
        ttnn_permute_262,
        ttnn_reshape_203,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_262, False)
    ttnn_reshape_254 = ttnn.reshape(
        ttnn_concat_120,
        [2, 4141, 6, 64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_120, False)
    ttnn_slice_95 = ttnn.slice(
        ttnn_reshape_254,
        [0, 0, 0, 0, 1],
        [2, 4141, 6, 64, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_2 = ttnn.neg(
        ttnn_slice_95,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_95, False)
    ttnn_slice_96 = ttnn.slice(
        ttnn_reshape_254,
        [0, 0, 0, 0, 0],
        [2, 4141, 6, 64, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_254, False)
    ttnn_concat_121 = ttnn.concat(
        [ttnn_neg_2, ttnn_slice_96],
        4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_96, False)
    ttnn.deallocate(ttnn_neg_2, False)
    ttnn_typecast_127 = ttnn.typecast(
        ttnn_concat_121,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_121, False)
    ttnn_reshape_255 = ttnn.reshape(
        ttnn_typecast_127,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_127, False)
    ttnn_permute_263 = ttnn.permute(
        ttnn_reshape_255,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_255, False)
    ttnn_multiply_19 = ttnn.multiply(
        ttnn_permute_263,
        ttnn_reshape_209,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_263, False)
    ttnn_add_26 = ttnn.add(
        ttnn_multiply_18,
        ttnn_multiply_19,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_19, False)
    ttnn.deallocate(ttnn_multiply_18, False)
    ttnn_typecast_128 = ttnn.typecast(
        ttnn_add_26,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_26, False)
    ttnn_reshape_256 = ttnn.reshape(
        ttnn_slice_83,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_83, False)
    ttnn_rms_norm_6 = ttnn.rms_norm(
        ttnn_reshape_256,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.1.attn.norm_added_k.weight"],
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
    ttnn.deallocate(ttnn_reshape_256, False)
    ttnn_reshape_257 = ttnn.reshape(
        ttnn_slice_93,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_93, False)
    ttnn_rms_norm_7 = ttnn.rms_norm(
        ttnn_reshape_257,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.1.attn.norm_k.weight"],
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
    ttnn.deallocate(ttnn_reshape_257, False)
    ttnn_concat_122 = ttnn.concat(
        [ttnn_rms_norm_6, ttnn_rms_norm_7],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_7, False)
    ttnn.deallocate(ttnn_rms_norm_6, False)
    ttnn_typecast_129 = ttnn.typecast(
        ttnn_concat_122,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_264 = ttnn.permute(
        ttnn_typecast_129,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_129, False)
    ttnn_multiply_20 = ttnn.multiply(
        ttnn_permute_264,
        ttnn_reshape_203,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_264, False)
    ttnn_reshape_258 = ttnn.reshape(
        ttnn_concat_122,
        [2, 4141, 6, 64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_122, False)
    ttnn_slice_97 = ttnn.slice(
        ttnn_reshape_258,
        [0, 0, 0, 0, 1],
        [2, 4141, 6, 64, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_3 = ttnn.neg(
        ttnn_slice_97,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_97, False)
    ttnn_slice_98 = ttnn.slice(
        ttnn_reshape_258,
        [0, 0, 0, 0, 0],
        [2, 4141, 6, 64, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_258, False)
    ttnn_concat_123 = ttnn.concat(
        [ttnn_neg_3, ttnn_slice_98],
        4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_98, False)
    ttnn.deallocate(ttnn_neg_3, False)
    ttnn_typecast_130 = ttnn.typecast(
        ttnn_concat_123,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_123, False)
    ttnn_reshape_259 = ttnn.reshape(
        ttnn_typecast_130,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_130, False)
    ttnn_permute_265 = ttnn.permute(
        ttnn_reshape_259,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_259, False)
    ttnn_multiply_21 = ttnn.multiply(
        ttnn_permute_265,
        ttnn_reshape_209,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_265, False)
    ttnn_add_27 = ttnn.add(
        ttnn_multiply_20,
        ttnn_multiply_21,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_21, False)
    ttnn.deallocate(ttnn_multiply_20, False)
    ttnn_typecast_131 = ttnn.typecast(
        ttnn_add_27,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_27, False)
    ttnn_reshape_260 = ttnn.reshape(
        ttnn_slice_84,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_84, False)
    ttnn_reshape_261 = ttnn.reshape(
        ttnn_slice_94,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_94, False)
    ttnn_concat_124 = ttnn.concat(
        [ttnn_reshape_260, ttnn_reshape_261],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_261, False)
    ttnn.deallocate(ttnn_reshape_260, False)
    ttnn_permute_266 = ttnn.permute(
        ttnn_concat_124,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_concat_124, False)
    ttnn_transformer_scaled_dot_product_attention_1 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_typecast_128,
            ttnn_typecast_131,
            ttnn_permute_266,
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
    ttnn.deallocate(ttnn_permute_266, False)
    ttnn.deallocate(ttnn_typecast_131, False)
    ttnn.deallocate(ttnn_typecast_128, False)
    ttnn_transformer_concatenate_heads_1 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_1, False)
    ttnn_slice_99 = ttnn.slice(
        ttnn_transformer_concatenate_heads_1,
        [0, 0, 0],
        [2, 45, 768],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_262 = ttnn.reshape(
        ttnn_slice_99,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_99, False)
    ttnn_matmul_7 = ttnn.matmul(
        ttnn_reshape_262,
        self.weights["transformer.transformer_blocks.1.attn.to_add_out.weight"],
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
    ttnn.deallocate(ttnn_reshape_262, False)
    ttnn_reshape_263 = ttnn.reshape(
        ttnn_matmul_7,
        [1, 1, 90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_7, False)
    ttnn_reduce_scatter_8 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_263,
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
    ttnn.deallocate(ttnn_reshape_263, False)
    ttnn_reshape_264 = ttnn.reshape(
        ttnn_reduce_scatter_8,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_8, False)
    ttnn_all_gather_8 = ttnn.all_gather(
        input_tensor=ttnn_reshape_264,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_264, False)
    ttnn_add_28 = ttnn.add(
        ttnn_all_gather_8,
        self.weights["transformer.transformer_blocks.1.attn.to_add_out.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_8, False)
    ttnn_reshape_265 = ttnn.reshape(
        ttnn_add_28,
        [2, 45, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_28, False)
    ttnn_multiply_22 = ttnn.multiply(
        ttnn_reshape_231,
        ttnn_reshape_265,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_265, False)
    ttnn.deallocate(ttnn_reshape_231, False)
    ttnn_add_29 = ttnn.add(
        ttnn_concat_119,
        ttnn_multiply_22,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_22, False)
    ttnn.deallocate(ttnn_concat_119, False)
    ttnn_layer_norm_6 = ttnn.layer_norm(
        ttnn_add_29,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_slice_100 = ttnn.slice(
        ttnn_typecast_124,
        [0, 15360],
        [2, 18432],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_266 = ttnn.reshape(
        ttnn_slice_100,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_100, False)
    ttnn_slice_101 = ttnn.slice(
        ttnn_typecast_124,
        [0, 12288],
        [2, 15360],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_267 = ttnn.reshape(
        ttnn_slice_101,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_101, False)
    ttnn_add_30 = ttnn.add(
        ttnn_reshape_267,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_267, False)
    ttnn_multiply_23 = ttnn.multiply(
        ttnn_layer_norm_6,
        ttnn_add_30,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_30, False)
    ttnn.deallocate(ttnn_layer_norm_6, False)
    ttnn_slice_102 = ttnn.slice(
        ttnn_typecast_124,
        [0, 9216],
        [2, 12288],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_124, False)
    ttnn_reshape_268 = ttnn.reshape(
        ttnn_slice_102,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_102, False)
    ttnn_add_31 = ttnn.add(
        ttnn_multiply_23,
        ttnn_reshape_268,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_268, False)
    ttnn.deallocate(ttnn_multiply_23, False)
    ttnn_reshape_269 = ttnn.reshape(
        ttnn_add_31,
        [90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_31, False)
    ttnn_linear_10 = ttnn.linear(
        ttnn_reshape_269,
        self.weights["transformer.transformer_blocks.1.ff_context.net.0.proj.weight"],
        bias=self.weights["transformer.transformer_blocks.1.ff_context.net.0.proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_269, False)
    ttnn_matmul_8 = ttnn.matmul(
        ttnn_linear_10,
        self.weights["transformer.transformer_blocks.1.ff_context.net.2.weight"],
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
    ttnn.deallocate(ttnn_linear_10, False)
    ttnn_reshape_270 = ttnn.reshape(
        ttnn_matmul_8,
        [1, 1, 90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_8, False)
    ttnn_reduce_scatter_9 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_270,
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
    ttnn.deallocate(ttnn_reshape_270, False)
    ttnn_reshape_271 = ttnn.reshape(
        ttnn_reduce_scatter_9,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_9, False)
    ttnn_all_gather_9 = ttnn.all_gather(
        input_tensor=ttnn_reshape_271,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_271, False)
    ttnn_add_32 = ttnn.add(
        ttnn_all_gather_9,
        self.weights["transformer.transformer_blocks.1.ff_context.net.2.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_9, False)
    ttnn_reshape_272 = ttnn.reshape(
        ttnn_add_32,
        [2, 45, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_32, False)
    ttnn_multiply_24 = ttnn.multiply(
        ttnn_reshape_266,
        ttnn_reshape_272,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_272, False)
    ttnn.deallocate(ttnn_reshape_266, False)
    ttnn_add_33 = ttnn.add(
        ttnn_add_29,
        ttnn_multiply_24,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_24, False)
    ttnn.deallocate(ttnn_add_29, False)
    ttnn_slice_103 = ttnn.slice(
        ttnn_add_33,
        [0, 0, 0],
        [2, 45, 1536],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_33, False)
    ttnn_to_layout_592 = ttnn.to_layout(
        text_encoder_layer_2,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_273 = ttnn.reshape(
        ttnn_to_layout_592,
        [90, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_592, False)
    return ttnn_add_22, ttnn_reshape_273, ttnn_slice_103, ttnn_transformer_concatenate_heads_1, ttnn_typecast_125


def _tb_forward_2(self, text_encoder_layer_3, ttnn_add_22, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_273, ttnn_slice_103, ttnn_slice_50, ttnn_slice_51, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_1, ttnn_typecast_125, var_1):
    ttnn_matmul_9 = ttnn.matmul(
        ttnn_reshape_273,
        self.weights["transformer.caption_projection.2.linear.weight"],
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
    ttnn.deallocate(ttnn_reshape_273, False)
    ttnn_reshape_274 = ttnn.reshape(
        ttnn_matmul_9,
        [2, 45, 1536],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_9, False)
    ttnn_concat_125 = ttnn.concat(
        [ttnn_slice_103, ttnn_reshape_274],
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_274, False)
    ttnn.deallocate(ttnn_slice_103, False)
    ttnn_layer_norm_7 = ttnn.layer_norm(
        ttnn_concat_125,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_reshape_275 = ttnn.reshape(
        ttnn_slice_51,
        [1, 1, 2, 18432],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_51, False)
    ttnn_reduce_scatter_10 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_275,
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
    ttnn.deallocate(ttnn_reshape_275, False)
    ttnn_reshape_276 = ttnn.reshape(
        ttnn_reduce_scatter_10,
        [2, 4608],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_10, False)
    ttnn_all_gather_10 = ttnn.all_gather(
        input_tensor=ttnn_reshape_276,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_276, False)
    ttnn_add_34 = ttnn.add(
        ttnn_all_gather_10,
        self.weights["transformer.transformer_blocks.2.norm1_context.linear.bias.f32"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_10, False)
    ttnn_typecast_132 = ttnn.typecast(
        ttnn_add_34,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_34, False)
    ttnn_slice_104 = ttnn.slice(
        ttnn_typecast_132,
        [0, 6144],
        [2, 9216],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_277 = ttnn.reshape(
        ttnn_slice_104,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_104, False)
    ttnn_slice_105 = ttnn.slice(
        ttnn_typecast_132,
        [0, 3072],
        [2, 6144],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_278 = ttnn.reshape(
        ttnn_slice_105,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_105, False)
    ttnn_add_35 = ttnn.add(
        ttnn_reshape_278,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_278, False)
    ttnn_multiply_25 = ttnn.multiply(
        ttnn_layer_norm_7,
        ttnn_add_35,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_35, False)
    ttnn.deallocate(ttnn_layer_norm_7, False)
    ttnn_slice_106 = ttnn.slice(
        ttnn_typecast_132,
        [0, 0],
        [2, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_279 = ttnn.reshape(
        ttnn_slice_106,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_106, False)
    ttnn_add_36 = ttnn.add(
        ttnn_multiply_25,
        ttnn_reshape_279,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_279, False)
    ttnn.deallocate(ttnn_multiply_25, False)
    ttnn_reshape_280 = ttnn.reshape(
        ttnn_add_36,
        [90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_36, False)
    ttnn_linear_11 = ttnn.linear(
        ttnn_reshape_280,
        self.weights["transformer.transformer_blocks.2.attn.fused_add_q_proj_add_k_proj_add_v_proj.weight"],
        bias=self.weights["transformer.transformer_blocks.2.attn.fused_add_q_proj_add_k_proj_add_v_proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_280, False)
    ttnn_slice_107 = ttnn.slice(
        ttnn_linear_11,
        [0, 0],
        [90, 768],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_108 = ttnn.slice(
        ttnn_linear_11,
        [0, 768],
        [90, 1536],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_109 = ttnn.slice(
        ttnn_linear_11,
        [0, 1536],
        [90, 2304],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_11, False)
    ttnn_reshape_281 = ttnn.reshape(
        ttnn_slice_107,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_107, False)
    ttnn_rms_norm_8 = ttnn.rms_norm(
        ttnn_reshape_281,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.2.attn.norm_added_q.weight"],
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
    ttnn.deallocate(ttnn_reshape_281, False)
    ttnn_slice_110 = ttnn.slice(
        ttnn_typecast_125,
        [0, 6144],
        [2, 9216],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_282 = ttnn.reshape(
        ttnn_slice_110,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_110, False)
    ttnn_slice_111 = ttnn.slice(
        ttnn_transformer_concatenate_heads_1,
        [0, 45, 0],
        [2, 4141, 768],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_1, False)
    ttnn_reshape_283 = ttnn.reshape(
        ttnn_slice_111,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_111, False)
    ttnn_matmul_10 = ttnn.matmul(
        ttnn_reshape_283,
        self.weights["transformer.transformer_blocks.1.attn.to_out.0.weight"],
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
    ttnn.deallocate(ttnn_reshape_283, False)
    ttnn_reshape_284 = ttnn.reshape(
        ttnn_matmul_10,
        [1, 1, 8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_10, False)
    ttnn_reduce_scatter_11 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_284,
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
    ttnn.deallocate(ttnn_reshape_284, False)
    ttnn_reshape_285 = ttnn.reshape(
        ttnn_reduce_scatter_11,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_11, False)
    ttnn_all_gather_11 = ttnn.all_gather(
        input_tensor=ttnn_reshape_285,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_285, False)
    ttnn_add_37 = ttnn.add(
        ttnn_all_gather_11,
        self.weights["transformer.transformer_blocks.1.attn.to_out.0.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_11, False)
    ttnn_reshape_286 = ttnn.reshape(
        ttnn_add_37,
        [2, 4096, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_37, False)
    ttnn_multiply_26 = ttnn.multiply(
        ttnn_reshape_282,
        ttnn_reshape_286,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_286, False)
    ttnn.deallocate(ttnn_reshape_282, False)
    ttnn_add_38 = ttnn.add(
        ttnn_add_22,
        ttnn_multiply_26,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_26, False)
    ttnn.deallocate(ttnn_add_22, False)
    ttnn_layer_norm_8 = ttnn.layer_norm(
        ttnn_add_38,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_slice_112 = ttnn.slice(
        ttnn_typecast_125,
        [0, 15360],
        [2, 18432],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_287 = ttnn.reshape(
        ttnn_slice_112,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_112, False)
    ttnn_slice_113 = ttnn.slice(
        ttnn_typecast_125,
        [0, 12288],
        [2, 15360],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_288 = ttnn.reshape(
        ttnn_slice_113,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_113, False)
    ttnn_add_39 = ttnn.add(
        ttnn_reshape_288,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_288, False)
    ttnn_multiply_27 = ttnn.multiply(
        ttnn_layer_norm_8,
        ttnn_add_39,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_39, False)
    ttnn.deallocate(ttnn_layer_norm_8, False)
    ttnn_slice_114 = ttnn.slice(
        ttnn_typecast_125,
        [0, 9216],
        [2, 12288],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_125, False)
    ttnn_reshape_289 = ttnn.reshape(
        ttnn_slice_114,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_114, False)
    ttnn_add_40 = ttnn.add(
        ttnn_multiply_27,
        ttnn_reshape_289,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_289, False)
    ttnn.deallocate(ttnn_multiply_27, False)
    ttnn_reshape_290 = ttnn.reshape(
        ttnn_add_40,
        [8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_40, False)
    ttnn_linear_12 = ttnn.linear(
        ttnn_reshape_290,
        self.weights["transformer.transformer_blocks.1.ff.net.0.proj.weight"],
        bias=self.weights["transformer.transformer_blocks.1.ff.net.0.proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_290, False)
    ttnn_matmul_11 = ttnn.matmul(
        ttnn_linear_12,
        self.weights["transformer.transformer_blocks.1.ff.net.2.weight"],
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
    ttnn.deallocate(ttnn_linear_12, False)
    ttnn_reshape_291 = ttnn.reshape(
        ttnn_matmul_11,
        [1, 1, 8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_11, False)
    ttnn_reduce_scatter_12 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_291,
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
    ttnn.deallocate(ttnn_reshape_291, False)
    ttnn_reshape_292 = ttnn.reshape(
        ttnn_reduce_scatter_12,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_12, False)
    ttnn_all_gather_12 = ttnn.all_gather(
        input_tensor=ttnn_reshape_292,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_292, False)
    ttnn_add_41 = ttnn.add(
        ttnn_all_gather_12,
        self.weights["transformer.transformer_blocks.1.ff.net.2.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_12, False)
    ttnn_reshape_293 = ttnn.reshape(
        ttnn_add_41,
        [2, 4096, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_41, False)
    ttnn_multiply_28 = ttnn.multiply(
        ttnn_reshape_287,
        ttnn_reshape_293,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_293, False)
    ttnn.deallocate(ttnn_reshape_287, False)
    ttnn_add_42 = ttnn.add(
        ttnn_add_38,
        ttnn_multiply_28,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_28, False)
    ttnn.deallocate(ttnn_add_38, False)
    ttnn_layer_norm_9 = ttnn.layer_norm(
        ttnn_add_42,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_reshape_294 = ttnn.reshape(
        ttnn_slice_50,
        [1, 1, 2, 18432],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_50, False)
    ttnn_reduce_scatter_13 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_294,
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
    ttnn.deallocate(ttnn_reshape_294, False)
    ttnn_reshape_295 = ttnn.reshape(
        ttnn_reduce_scatter_13,
        [2, 4608],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_13, False)
    ttnn_all_gather_13 = ttnn.all_gather(
        input_tensor=ttnn_reshape_295,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_295, False)
    ttnn_add_43 = ttnn.add(
        ttnn_all_gather_13,
        self.weights["transformer.transformer_blocks.2.norm1.linear.bias.f32"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_13, False)
    ttnn_typecast_133 = ttnn.typecast(
        ttnn_add_43,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_43, False)
    ttnn_slice_115 = ttnn.slice(
        ttnn_typecast_133,
        [0, 3072],
        [2, 6144],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_296 = ttnn.reshape(
        ttnn_slice_115,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_115, False)
    ttnn_add_44 = ttnn.add(
        ttnn_reshape_296,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_296, False)
    ttnn_multiply_29 = ttnn.multiply(
        ttnn_layer_norm_9,
        ttnn_add_44,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_44, False)
    ttnn.deallocate(ttnn_layer_norm_9, False)
    ttnn_slice_116 = ttnn.slice(
        ttnn_typecast_133,
        [0, 0],
        [2, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_297 = ttnn.reshape(
        ttnn_slice_116,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_116, False)
    ttnn_add_45 = ttnn.add(
        ttnn_multiply_29,
        ttnn_reshape_297,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_297, False)
    ttnn.deallocate(ttnn_multiply_29, False)
    ttnn_reshape_298 = ttnn.reshape(
        ttnn_add_45,
        [8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_45, False)
    ttnn_linear_13 = ttnn.linear(
        ttnn_reshape_298,
        self.weights["transformer.transformer_blocks.2.attn.fused_to_q_to_k_to_v.weight"],
        bias=self.weights["transformer.transformer_blocks.2.attn.fused_to_q_to_k_to_v.bias"],
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
    ttnn.deallocate(ttnn_reshape_298, False)
    ttnn_slice_117 = ttnn.slice(
        ttnn_linear_13,
        [0, 0],
        [8192, 768],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_118 = ttnn.slice(
        ttnn_linear_13,
        [0, 768],
        [8192, 1536],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_119 = ttnn.slice(
        ttnn_linear_13,
        [0, 1536],
        [8192, 2304],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_13, False)
    ttnn_reshape_299 = ttnn.reshape(
        ttnn_slice_117,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_117, False)
    ttnn_rms_norm_9 = ttnn.rms_norm(
        ttnn_reshape_299,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.2.attn.norm_q.weight"],
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
    ttnn.deallocate(ttnn_reshape_299, False)
    ttnn_concat_126 = ttnn.concat(
        [ttnn_rms_norm_8, ttnn_rms_norm_9],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_9, False)
    ttnn.deallocate(ttnn_rms_norm_8, False)
    ttnn_typecast_134 = ttnn.typecast(
        ttnn_concat_126,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_267 = ttnn.permute(
        ttnn_typecast_134,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_134, False)
    ttnn_multiply_30 = ttnn.multiply(
        ttnn_permute_267,
        ttnn_reshape_203,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_267, False)
    ttnn_reshape_300 = ttnn.reshape(
        ttnn_concat_126,
        [2, 4141, 6, 64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_126, False)
    ttnn_slice_120 = ttnn.slice(
        ttnn_reshape_300,
        [0, 0, 0, 0, 1],
        [2, 4141, 6, 64, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_4 = ttnn.neg(
        ttnn_slice_120,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_120, False)
    ttnn_slice_121 = ttnn.slice(
        ttnn_reshape_300,
        [0, 0, 0, 0, 0],
        [2, 4141, 6, 64, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_300, False)
    ttnn_concat_127 = ttnn.concat(
        [ttnn_neg_4, ttnn_slice_121],
        4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_121, False)
    ttnn.deallocate(ttnn_neg_4, False)
    ttnn_typecast_135 = ttnn.typecast(
        ttnn_concat_127,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_127, False)
    ttnn_reshape_301 = ttnn.reshape(
        ttnn_typecast_135,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_135, False)
    ttnn_permute_268 = ttnn.permute(
        ttnn_reshape_301,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_301, False)
    ttnn_multiply_31 = ttnn.multiply(
        ttnn_permute_268,
        ttnn_reshape_209,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_268, False)
    ttnn_add_46 = ttnn.add(
        ttnn_multiply_30,
        ttnn_multiply_31,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_31, False)
    ttnn.deallocate(ttnn_multiply_30, False)
    ttnn_typecast_136 = ttnn.typecast(
        ttnn_add_46,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_46, False)
    ttnn_reshape_302 = ttnn.reshape(
        ttnn_slice_108,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_108, False)
    ttnn_rms_norm_10 = ttnn.rms_norm(
        ttnn_reshape_302,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.2.attn.norm_added_k.weight"],
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
    ttnn.deallocate(ttnn_reshape_302, False)
    ttnn_reshape_303 = ttnn.reshape(
        ttnn_slice_118,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_118, False)
    ttnn_rms_norm_11 = ttnn.rms_norm(
        ttnn_reshape_303,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.2.attn.norm_k.weight"],
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
    ttnn.deallocate(ttnn_reshape_303, False)
    ttnn_concat_128 = ttnn.concat(
        [ttnn_rms_norm_10, ttnn_rms_norm_11],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_11, False)
    ttnn.deallocate(ttnn_rms_norm_10, False)
    ttnn_typecast_137 = ttnn.typecast(
        ttnn_concat_128,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_269 = ttnn.permute(
        ttnn_typecast_137,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_137, False)
    ttnn_multiply_32 = ttnn.multiply(
        ttnn_permute_269,
        ttnn_reshape_203,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_269, False)
    ttnn_reshape_304 = ttnn.reshape(
        ttnn_concat_128,
        [2, 4141, 6, 64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_128, False)
    ttnn_slice_122 = ttnn.slice(
        ttnn_reshape_304,
        [0, 0, 0, 0, 1],
        [2, 4141, 6, 64, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_5 = ttnn.neg(
        ttnn_slice_122,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_122, False)
    ttnn_slice_123 = ttnn.slice(
        ttnn_reshape_304,
        [0, 0, 0, 0, 0],
        [2, 4141, 6, 64, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_304, False)
    ttnn_concat_129 = ttnn.concat(
        [ttnn_neg_5, ttnn_slice_123],
        4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_123, False)
    ttnn.deallocate(ttnn_neg_5, False)
    ttnn_typecast_138 = ttnn.typecast(
        ttnn_concat_129,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_129, False)
    ttnn_reshape_305 = ttnn.reshape(
        ttnn_typecast_138,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_138, False)
    ttnn_permute_270 = ttnn.permute(
        ttnn_reshape_305,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_305, False)
    ttnn_multiply_33 = ttnn.multiply(
        ttnn_permute_270,
        ttnn_reshape_209,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_270, False)
    ttnn_add_47 = ttnn.add(
        ttnn_multiply_32,
        ttnn_multiply_33,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_33, False)
    ttnn.deallocate(ttnn_multiply_32, False)
    ttnn_typecast_139 = ttnn.typecast(
        ttnn_add_47,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_47, False)
    ttnn_reshape_306 = ttnn.reshape(
        ttnn_slice_109,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_109, False)
    ttnn_reshape_307 = ttnn.reshape(
        ttnn_slice_119,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_119, False)
    ttnn_concat_130 = ttnn.concat(
        [ttnn_reshape_306, ttnn_reshape_307],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_307, False)
    ttnn.deallocate(ttnn_reshape_306, False)
    ttnn_permute_271 = ttnn.permute(
        ttnn_concat_130,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_concat_130, False)
    ttnn_transformer_scaled_dot_product_attention_2 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_typecast_136,
            ttnn_typecast_139,
            ttnn_permute_271,
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
    ttnn.deallocate(ttnn_permute_271, False)
    ttnn.deallocate(ttnn_typecast_139, False)
    ttnn.deallocate(ttnn_typecast_136, False)
    ttnn_transformer_concatenate_heads_2 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_2, False)
    ttnn_slice_124 = ttnn.slice(
        ttnn_transformer_concatenate_heads_2,
        [0, 0, 0],
        [2, 45, 768],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_308 = ttnn.reshape(
        ttnn_slice_124,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_124, False)
    ttnn_matmul_12 = ttnn.matmul(
        ttnn_reshape_308,
        self.weights["transformer.transformer_blocks.2.attn.to_add_out.weight"],
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
    ttnn.deallocate(ttnn_reshape_308, False)
    ttnn_reshape_309 = ttnn.reshape(
        ttnn_matmul_12,
        [1, 1, 90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_12, False)
    ttnn_reduce_scatter_14 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_309,
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
    ttnn.deallocate(ttnn_reshape_309, False)
    ttnn_reshape_310 = ttnn.reshape(
        ttnn_reduce_scatter_14,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_14, False)
    ttnn_all_gather_14 = ttnn.all_gather(
        input_tensor=ttnn_reshape_310,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_310, False)
    ttnn_add_48 = ttnn.add(
        ttnn_all_gather_14,
        self.weights["transformer.transformer_blocks.2.attn.to_add_out.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_14, False)
    ttnn_reshape_311 = ttnn.reshape(
        ttnn_add_48,
        [2, 45, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_48, False)
    ttnn_multiply_34 = ttnn.multiply(
        ttnn_reshape_277,
        ttnn_reshape_311,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_311, False)
    ttnn.deallocate(ttnn_reshape_277, False)
    ttnn_add_49 = ttnn.add(
        ttnn_concat_125,
        ttnn_multiply_34,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_34, False)
    ttnn.deallocate(ttnn_concat_125, False)
    ttnn_layer_norm_10 = ttnn.layer_norm(
        ttnn_add_49,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_slice_125 = ttnn.slice(
        ttnn_typecast_132,
        [0, 15360],
        [2, 18432],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_312 = ttnn.reshape(
        ttnn_slice_125,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_125, False)
    ttnn_slice_126 = ttnn.slice(
        ttnn_typecast_132,
        [0, 12288],
        [2, 15360],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_313 = ttnn.reshape(
        ttnn_slice_126,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_126, False)
    ttnn_add_50 = ttnn.add(
        ttnn_reshape_313,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_313, False)
    ttnn_multiply_35 = ttnn.multiply(
        ttnn_layer_norm_10,
        ttnn_add_50,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_50, False)
    ttnn.deallocate(ttnn_layer_norm_10, False)
    ttnn_slice_127 = ttnn.slice(
        ttnn_typecast_132,
        [0, 9216],
        [2, 12288],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_132, False)
    ttnn_reshape_314 = ttnn.reshape(
        ttnn_slice_127,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_127, False)
    ttnn_add_51 = ttnn.add(
        ttnn_multiply_35,
        ttnn_reshape_314,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_314, False)
    ttnn.deallocate(ttnn_multiply_35, False)
    ttnn_reshape_315 = ttnn.reshape(
        ttnn_add_51,
        [90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_51, False)
    ttnn_linear_14 = ttnn.linear(
        ttnn_reshape_315,
        self.weights["transformer.transformer_blocks.2.ff_context.net.0.proj.weight"],
        bias=self.weights["transformer.transformer_blocks.2.ff_context.net.0.proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_315, False)
    ttnn_matmul_13 = ttnn.matmul(
        ttnn_linear_14,
        self.weights["transformer.transformer_blocks.2.ff_context.net.2.weight"],
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
    ttnn.deallocate(ttnn_linear_14, False)
    ttnn_reshape_316 = ttnn.reshape(
        ttnn_matmul_13,
        [1, 1, 90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_13, False)
    ttnn_reduce_scatter_15 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_316,
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
    ttnn.deallocate(ttnn_reshape_316, False)
    ttnn_reshape_317 = ttnn.reshape(
        ttnn_reduce_scatter_15,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_15, False)
    ttnn_all_gather_15 = ttnn.all_gather(
        input_tensor=ttnn_reshape_317,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_317, False)
    ttnn_add_52 = ttnn.add(
        ttnn_all_gather_15,
        self.weights["transformer.transformer_blocks.2.ff_context.net.2.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_15, False)
    ttnn_reshape_318 = ttnn.reshape(
        ttnn_add_52,
        [2, 45, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_52, False)
    ttnn_multiply_36 = ttnn.multiply(
        ttnn_reshape_312,
        ttnn_reshape_318,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_318, False)
    ttnn.deallocate(ttnn_reshape_312, False)
    ttnn_add_53 = ttnn.add(
        ttnn_add_49,
        ttnn_multiply_36,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_36, False)
    ttnn.deallocate(ttnn_add_49, False)
    ttnn_slice_128 = ttnn.slice(
        ttnn_add_53,
        [0, 0, 0],
        [2, 45, 1536],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_53, False)
    ttnn_to_layout_593 = ttnn.to_layout(
        text_encoder_layer_3,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_319 = ttnn.reshape(
        ttnn_to_layout_593,
        [90, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_593, False)
    return ttnn_add_42, ttnn_reshape_319, ttnn_slice_128, ttnn_transformer_concatenate_heads_2, ttnn_typecast_133


def _tb_forward_3(self, text_encoder_layer_4, ttnn_add_42, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_319, ttnn_slice_128, ttnn_slice_48, ttnn_slice_49, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_2, ttnn_typecast_133, var_1):
    ttnn_matmul_14 = ttnn.matmul(
        ttnn_reshape_319,
        self.weights["transformer.caption_projection.3.linear.weight"],
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
    ttnn.deallocate(ttnn_reshape_319, False)
    ttnn_reshape_320 = ttnn.reshape(
        ttnn_matmul_14,
        [2, 45, 1536],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_14, False)
    ttnn_concat_131 = ttnn.concat(
        [ttnn_slice_128, ttnn_reshape_320],
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_320, False)
    ttnn.deallocate(ttnn_slice_128, False)
    ttnn_layer_norm_11 = ttnn.layer_norm(
        ttnn_concat_131,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_reshape_321 = ttnn.reshape(
        ttnn_slice_49,
        [1, 1, 2, 18432],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_49, False)
    ttnn_reduce_scatter_16 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_321,
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
    ttnn.deallocate(ttnn_reshape_321, False)
    ttnn_reshape_322 = ttnn.reshape(
        ttnn_reduce_scatter_16,
        [2, 4608],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_16, False)
    ttnn_all_gather_16 = ttnn.all_gather(
        input_tensor=ttnn_reshape_322,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_322, False)
    ttnn_add_54 = ttnn.add(
        ttnn_all_gather_16,
        self.weights["transformer.transformer_blocks.3.norm1_context.linear.bias.f32"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_16, False)
    ttnn_typecast_140 = ttnn.typecast(
        ttnn_add_54,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_54, False)
    ttnn_slice_129 = ttnn.slice(
        ttnn_typecast_140,
        [0, 6144],
        [2, 9216],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_323 = ttnn.reshape(
        ttnn_slice_129,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_129, False)
    ttnn_slice_130 = ttnn.slice(
        ttnn_typecast_140,
        [0, 3072],
        [2, 6144],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_324 = ttnn.reshape(
        ttnn_slice_130,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_130, False)
    ttnn_add_55 = ttnn.add(
        ttnn_reshape_324,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_324, False)
    ttnn_multiply_37 = ttnn.multiply(
        ttnn_layer_norm_11,
        ttnn_add_55,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_55, False)
    ttnn.deallocate(ttnn_layer_norm_11, False)
    ttnn_slice_131 = ttnn.slice(
        ttnn_typecast_140,
        [0, 0],
        [2, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_325 = ttnn.reshape(
        ttnn_slice_131,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_131, False)
    ttnn_add_56 = ttnn.add(
        ttnn_multiply_37,
        ttnn_reshape_325,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_325, False)
    ttnn.deallocate(ttnn_multiply_37, False)
    ttnn_reshape_326 = ttnn.reshape(
        ttnn_add_56,
        [90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_56, False)
    ttnn_linear_15 = ttnn.linear(
        ttnn_reshape_326,
        self.weights["transformer.transformer_blocks.3.attn.fused_add_q_proj_add_k_proj_add_v_proj.weight"],
        bias=self.weights["transformer.transformer_blocks.3.attn.fused_add_q_proj_add_k_proj_add_v_proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_326, False)
    ttnn_slice_132 = ttnn.slice(
        ttnn_linear_15,
        [0, 0],
        [90, 768],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_133 = ttnn.slice(
        ttnn_linear_15,
        [0, 768],
        [90, 1536],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_134 = ttnn.slice(
        ttnn_linear_15,
        [0, 1536],
        [90, 2304],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_15, False)
    ttnn_reshape_327 = ttnn.reshape(
        ttnn_slice_132,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_132, False)
    ttnn_rms_norm_12 = ttnn.rms_norm(
        ttnn_reshape_327,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.3.attn.norm_added_q.weight"],
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
    ttnn.deallocate(ttnn_reshape_327, False)
    ttnn_slice_135 = ttnn.slice(
        ttnn_typecast_133,
        [0, 6144],
        [2, 9216],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_328 = ttnn.reshape(
        ttnn_slice_135,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_135, False)
    ttnn_slice_136 = ttnn.slice(
        ttnn_transformer_concatenate_heads_2,
        [0, 45, 0],
        [2, 4141, 768],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_2, False)
    ttnn_reshape_329 = ttnn.reshape(
        ttnn_slice_136,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_136, False)
    ttnn_matmul_15 = ttnn.matmul(
        ttnn_reshape_329,
        self.weights["transformer.transformer_blocks.2.attn.to_out.0.weight"],
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
    ttnn.deallocate(ttnn_reshape_329, False)
    ttnn_reshape_330 = ttnn.reshape(
        ttnn_matmul_15,
        [1, 1, 8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_15, False)
    ttnn_reduce_scatter_17 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_330,
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
    ttnn.deallocate(ttnn_reshape_330, False)
    ttnn_reshape_331 = ttnn.reshape(
        ttnn_reduce_scatter_17,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_17, False)
    ttnn_all_gather_17 = ttnn.all_gather(
        input_tensor=ttnn_reshape_331,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_331, False)
    ttnn_add_57 = ttnn.add(
        ttnn_all_gather_17,
        self.weights["transformer.transformer_blocks.2.attn.to_out.0.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_17, False)
    ttnn_reshape_332 = ttnn.reshape(
        ttnn_add_57,
        [2, 4096, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_57, False)
    ttnn_multiply_38 = ttnn.multiply(
        ttnn_reshape_328,
        ttnn_reshape_332,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_332, False)
    ttnn.deallocate(ttnn_reshape_328, False)
    ttnn_add_58 = ttnn.add(
        ttnn_add_42,
        ttnn_multiply_38,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_38, False)
    ttnn.deallocate(ttnn_add_42, False)
    ttnn_layer_norm_12 = ttnn.layer_norm(
        ttnn_add_58,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_slice_137 = ttnn.slice(
        ttnn_typecast_133,
        [0, 15360],
        [2, 18432],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_333 = ttnn.reshape(
        ttnn_slice_137,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_137, False)
    ttnn_slice_138 = ttnn.slice(
        ttnn_typecast_133,
        [0, 12288],
        [2, 15360],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_334 = ttnn.reshape(
        ttnn_slice_138,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_138, False)
    ttnn_add_59 = ttnn.add(
        ttnn_reshape_334,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_334, False)
    ttnn_multiply_39 = ttnn.multiply(
        ttnn_layer_norm_12,
        ttnn_add_59,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_59, False)
    ttnn.deallocate(ttnn_layer_norm_12, False)
    ttnn_slice_139 = ttnn.slice(
        ttnn_typecast_133,
        [0, 9216],
        [2, 12288],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_133, False)
    ttnn_reshape_335 = ttnn.reshape(
        ttnn_slice_139,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_139, False)
    ttnn_add_60 = ttnn.add(
        ttnn_multiply_39,
        ttnn_reshape_335,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_335, False)
    ttnn.deallocate(ttnn_multiply_39, False)
    ttnn_reshape_336 = ttnn.reshape(
        ttnn_add_60,
        [8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_60, False)
    ttnn_linear_16 = ttnn.linear(
        ttnn_reshape_336,
        self.weights["transformer.transformer_blocks.2.ff.net.0.proj.weight"],
        bias=self.weights["transformer.transformer_blocks.2.ff.net.0.proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_336, False)
    ttnn_matmul_16 = ttnn.matmul(
        ttnn_linear_16,
        self.weights["transformer.transformer_blocks.2.ff.net.2.weight"],
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
    ttnn.deallocate(ttnn_linear_16, False)
    ttnn_reshape_337 = ttnn.reshape(
        ttnn_matmul_16,
        [1, 1, 8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_16, False)
    ttnn_reduce_scatter_18 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_337,
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
    ttnn.deallocate(ttnn_reshape_337, False)
    ttnn_reshape_338 = ttnn.reshape(
        ttnn_reduce_scatter_18,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_18, False)
    ttnn_all_gather_18 = ttnn.all_gather(
        input_tensor=ttnn_reshape_338,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_338, False)
    ttnn_add_61 = ttnn.add(
        ttnn_all_gather_18,
        self.weights["transformer.transformer_blocks.2.ff.net.2.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_18, False)
    ttnn_reshape_339 = ttnn.reshape(
        ttnn_add_61,
        [2, 4096, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_61, False)
    ttnn_multiply_40 = ttnn.multiply(
        ttnn_reshape_333,
        ttnn_reshape_339,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_339, False)
    ttnn.deallocate(ttnn_reshape_333, False)
    ttnn_add_62 = ttnn.add(
        ttnn_add_58,
        ttnn_multiply_40,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_40, False)
    ttnn.deallocate(ttnn_add_58, False)
    ttnn_layer_norm_13 = ttnn.layer_norm(
        ttnn_add_62,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_reshape_340 = ttnn.reshape(
        ttnn_slice_48,
        [1, 1, 2, 18432],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_48, False)
    ttnn_reduce_scatter_19 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_340,
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
    ttnn.deallocate(ttnn_reshape_340, False)
    ttnn_reshape_341 = ttnn.reshape(
        ttnn_reduce_scatter_19,
        [2, 4608],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_19, False)
    ttnn_all_gather_19 = ttnn.all_gather(
        input_tensor=ttnn_reshape_341,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_341, False)
    ttnn_add_63 = ttnn.add(
        ttnn_all_gather_19,
        self.weights["transformer.transformer_blocks.3.norm1.linear.bias.f32"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_19, False)
    ttnn_typecast_141 = ttnn.typecast(
        ttnn_add_63,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_63, False)
    ttnn_slice_140 = ttnn.slice(
        ttnn_typecast_141,
        [0, 3072],
        [2, 6144],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_342 = ttnn.reshape(
        ttnn_slice_140,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_140, False)
    ttnn_add_64 = ttnn.add(
        ttnn_reshape_342,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_342, False)
    ttnn_multiply_41 = ttnn.multiply(
        ttnn_layer_norm_13,
        ttnn_add_64,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_64, False)
    ttnn.deallocate(ttnn_layer_norm_13, False)
    ttnn_slice_141 = ttnn.slice(
        ttnn_typecast_141,
        [0, 0],
        [2, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_343 = ttnn.reshape(
        ttnn_slice_141,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_141, False)
    ttnn_add_65 = ttnn.add(
        ttnn_multiply_41,
        ttnn_reshape_343,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_343, False)
    ttnn.deallocate(ttnn_multiply_41, False)
    ttnn_reshape_344 = ttnn.reshape(
        ttnn_add_65,
        [8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_65, False)
    ttnn_linear_17 = ttnn.linear(
        ttnn_reshape_344,
        self.weights["transformer.transformer_blocks.3.attn.fused_to_q_to_k_to_v.weight"],
        bias=self.weights["transformer.transformer_blocks.3.attn.fused_to_q_to_k_to_v.bias"],
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
    ttnn.deallocate(ttnn_reshape_344, False)
    ttnn_slice_142 = ttnn.slice(
        ttnn_linear_17,
        [0, 0],
        [8192, 768],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_143 = ttnn.slice(
        ttnn_linear_17,
        [0, 768],
        [8192, 1536],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_144 = ttnn.slice(
        ttnn_linear_17,
        [0, 1536],
        [8192, 2304],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_17, False)
    ttnn_reshape_345 = ttnn.reshape(
        ttnn_slice_142,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_142, False)
    ttnn_rms_norm_13 = ttnn.rms_norm(
        ttnn_reshape_345,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.3.attn.norm_q.weight"],
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
    ttnn.deallocate(ttnn_reshape_345, False)
    ttnn_concat_132 = ttnn.concat(
        [ttnn_rms_norm_12, ttnn_rms_norm_13],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_13, False)
    ttnn.deallocate(ttnn_rms_norm_12, False)
    ttnn_typecast_142 = ttnn.typecast(
        ttnn_concat_132,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_272 = ttnn.permute(
        ttnn_typecast_142,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_142, False)
    ttnn_multiply_42 = ttnn.multiply(
        ttnn_permute_272,
        ttnn_reshape_203,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_272, False)
    ttnn_reshape_346 = ttnn.reshape(
        ttnn_concat_132,
        [2, 4141, 6, 64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_132, False)
    ttnn_slice_145 = ttnn.slice(
        ttnn_reshape_346,
        [0, 0, 0, 0, 1],
        [2, 4141, 6, 64, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_6 = ttnn.neg(
        ttnn_slice_145,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_145, False)
    ttnn_slice_146 = ttnn.slice(
        ttnn_reshape_346,
        [0, 0, 0, 0, 0],
        [2, 4141, 6, 64, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_346, False)
    ttnn_concat_133 = ttnn.concat(
        [ttnn_neg_6, ttnn_slice_146],
        4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_146, False)
    ttnn.deallocate(ttnn_neg_6, False)
    ttnn_typecast_143 = ttnn.typecast(
        ttnn_concat_133,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_133, False)
    ttnn_reshape_347 = ttnn.reshape(
        ttnn_typecast_143,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_143, False)
    ttnn_permute_273 = ttnn.permute(
        ttnn_reshape_347,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_347, False)
    ttnn_multiply_43 = ttnn.multiply(
        ttnn_permute_273,
        ttnn_reshape_209,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_273, False)
    ttnn_add_66 = ttnn.add(
        ttnn_multiply_42,
        ttnn_multiply_43,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_43, False)
    ttnn.deallocate(ttnn_multiply_42, False)
    ttnn_typecast_144 = ttnn.typecast(
        ttnn_add_66,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_66, False)
    ttnn_reshape_348 = ttnn.reshape(
        ttnn_slice_133,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_133, False)
    ttnn_rms_norm_14 = ttnn.rms_norm(
        ttnn_reshape_348,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.3.attn.norm_added_k.weight"],
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
    ttnn.deallocate(ttnn_reshape_348, False)
    ttnn_reshape_349 = ttnn.reshape(
        ttnn_slice_143,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_143, False)
    ttnn_rms_norm_15 = ttnn.rms_norm(
        ttnn_reshape_349,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.3.attn.norm_k.weight"],
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
    ttnn.deallocate(ttnn_reshape_349, False)
    ttnn_concat_134 = ttnn.concat(
        [ttnn_rms_norm_14, ttnn_rms_norm_15],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_15, False)
    ttnn.deallocate(ttnn_rms_norm_14, False)
    ttnn_typecast_145 = ttnn.typecast(
        ttnn_concat_134,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_274 = ttnn.permute(
        ttnn_typecast_145,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_145, False)
    ttnn_multiply_44 = ttnn.multiply(
        ttnn_permute_274,
        ttnn_reshape_203,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_274, False)
    ttnn_reshape_350 = ttnn.reshape(
        ttnn_concat_134,
        [2, 4141, 6, 64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_134, False)
    ttnn_slice_147 = ttnn.slice(
        ttnn_reshape_350,
        [0, 0, 0, 0, 1],
        [2, 4141, 6, 64, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_7 = ttnn.neg(
        ttnn_slice_147,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_147, False)
    ttnn_slice_148 = ttnn.slice(
        ttnn_reshape_350,
        [0, 0, 0, 0, 0],
        [2, 4141, 6, 64, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_350, False)
    ttnn_concat_135 = ttnn.concat(
        [ttnn_neg_7, ttnn_slice_148],
        4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_148, False)
    ttnn.deallocate(ttnn_neg_7, False)
    ttnn_typecast_146 = ttnn.typecast(
        ttnn_concat_135,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_135, False)
    ttnn_reshape_351 = ttnn.reshape(
        ttnn_typecast_146,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_146, False)
    ttnn_permute_275 = ttnn.permute(
        ttnn_reshape_351,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_351, False)
    ttnn_multiply_45 = ttnn.multiply(
        ttnn_permute_275,
        ttnn_reshape_209,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_275, False)
    ttnn_add_67 = ttnn.add(
        ttnn_multiply_44,
        ttnn_multiply_45,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_45, False)
    ttnn.deallocate(ttnn_multiply_44, False)
    ttnn_typecast_147 = ttnn.typecast(
        ttnn_add_67,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_67, False)
    ttnn_reshape_352 = ttnn.reshape(
        ttnn_slice_134,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_134, False)
    ttnn_reshape_353 = ttnn.reshape(
        ttnn_slice_144,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_144, False)
    ttnn_concat_136 = ttnn.concat(
        [ttnn_reshape_352, ttnn_reshape_353],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_353, False)
    ttnn.deallocate(ttnn_reshape_352, False)
    ttnn_permute_276 = ttnn.permute(
        ttnn_concat_136,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_concat_136, False)
    ttnn_transformer_scaled_dot_product_attention_3 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_typecast_144,
            ttnn_typecast_147,
            ttnn_permute_276,
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
    ttnn.deallocate(ttnn_permute_276, False)
    ttnn.deallocate(ttnn_typecast_147, False)
    ttnn.deallocate(ttnn_typecast_144, False)
    ttnn_transformer_concatenate_heads_3 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_3, False)
    ttnn_slice_149 = ttnn.slice(
        ttnn_transformer_concatenate_heads_3,
        [0, 0, 0],
        [2, 45, 768],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_354 = ttnn.reshape(
        ttnn_slice_149,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_149, False)
    ttnn_matmul_17 = ttnn.matmul(
        ttnn_reshape_354,
        self.weights["transformer.transformer_blocks.3.attn.to_add_out.weight"],
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
    ttnn.deallocate(ttnn_reshape_354, False)
    ttnn_reshape_355 = ttnn.reshape(
        ttnn_matmul_17,
        [1, 1, 90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_17, False)
    ttnn_reduce_scatter_20 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_355,
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
    ttnn.deallocate(ttnn_reshape_355, False)
    ttnn_reshape_356 = ttnn.reshape(
        ttnn_reduce_scatter_20,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_20, False)
    ttnn_all_gather_20 = ttnn.all_gather(
        input_tensor=ttnn_reshape_356,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_356, False)
    ttnn_add_68 = ttnn.add(
        ttnn_all_gather_20,
        self.weights["transformer.transformer_blocks.3.attn.to_add_out.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_20, False)
    ttnn_reshape_357 = ttnn.reshape(
        ttnn_add_68,
        [2, 45, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_68, False)
    ttnn_multiply_46 = ttnn.multiply(
        ttnn_reshape_323,
        ttnn_reshape_357,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_357, False)
    ttnn.deallocate(ttnn_reshape_323, False)
    ttnn_add_69 = ttnn.add(
        ttnn_concat_131,
        ttnn_multiply_46,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_46, False)
    ttnn.deallocate(ttnn_concat_131, False)
    ttnn_layer_norm_14 = ttnn.layer_norm(
        ttnn_add_69,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_slice_150 = ttnn.slice(
        ttnn_typecast_140,
        [0, 15360],
        [2, 18432],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_358 = ttnn.reshape(
        ttnn_slice_150,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_150, False)
    ttnn_slice_151 = ttnn.slice(
        ttnn_typecast_140,
        [0, 12288],
        [2, 15360],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_359 = ttnn.reshape(
        ttnn_slice_151,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_151, False)
    ttnn_add_70 = ttnn.add(
        ttnn_reshape_359,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_359, False)
    ttnn_multiply_47 = ttnn.multiply(
        ttnn_layer_norm_14,
        ttnn_add_70,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_70, False)
    ttnn.deallocate(ttnn_layer_norm_14, False)
    ttnn_slice_152 = ttnn.slice(
        ttnn_typecast_140,
        [0, 9216],
        [2, 12288],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_140, False)
    ttnn_reshape_360 = ttnn.reshape(
        ttnn_slice_152,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_152, False)
    ttnn_add_71 = ttnn.add(
        ttnn_multiply_47,
        ttnn_reshape_360,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_360, False)
    ttnn.deallocate(ttnn_multiply_47, False)
    ttnn_reshape_361 = ttnn.reshape(
        ttnn_add_71,
        [90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_71, False)
    ttnn_linear_18 = ttnn.linear(
        ttnn_reshape_361,
        self.weights["transformer.transformer_blocks.3.ff_context.net.0.proj.weight"],
        bias=self.weights["transformer.transformer_blocks.3.ff_context.net.0.proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_361, False)
    ttnn_matmul_18 = ttnn.matmul(
        ttnn_linear_18,
        self.weights["transformer.transformer_blocks.3.ff_context.net.2.weight"],
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
    ttnn.deallocate(ttnn_linear_18, False)
    ttnn_reshape_362 = ttnn.reshape(
        ttnn_matmul_18,
        [1, 1, 90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_18, False)
    ttnn_reduce_scatter_21 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_362,
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
    ttnn.deallocate(ttnn_reshape_362, False)
    ttnn_reshape_363 = ttnn.reshape(
        ttnn_reduce_scatter_21,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_21, False)
    ttnn_all_gather_21 = ttnn.all_gather(
        input_tensor=ttnn_reshape_363,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_363, False)
    ttnn_add_72 = ttnn.add(
        ttnn_all_gather_21,
        self.weights["transformer.transformer_blocks.3.ff_context.net.2.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_21, False)
    ttnn_reshape_364 = ttnn.reshape(
        ttnn_add_72,
        [2, 45, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_72, False)
    ttnn_multiply_48 = ttnn.multiply(
        ttnn_reshape_358,
        ttnn_reshape_364,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_364, False)
    ttnn.deallocate(ttnn_reshape_358, False)
    ttnn_add_73 = ttnn.add(
        ttnn_add_69,
        ttnn_multiply_48,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_48, False)
    ttnn.deallocate(ttnn_add_69, False)
    ttnn_slice_153 = ttnn.slice(
        ttnn_add_73,
        [0, 0, 0],
        [2, 45, 1536],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_73, False)
    ttnn_to_layout_594 = ttnn.to_layout(
        text_encoder_layer_4,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_365 = ttnn.reshape(
        ttnn_to_layout_594,
        [90, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_594, False)
    return ttnn_add_62, ttnn_reshape_365, ttnn_slice_153, ttnn_transformer_concatenate_heads_3, ttnn_typecast_141


def _tb_forward_4(self, text_encoder_layer_5, ttnn_add_62, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_365, ttnn_slice_153, ttnn_slice_46, ttnn_slice_47, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_3, ttnn_typecast_141, var_1):
    ttnn_matmul_19 = ttnn.matmul(
        ttnn_reshape_365,
        self.weights["transformer.caption_projection.4.linear.weight"],
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
    ttnn.deallocate(ttnn_reshape_365, False)
    ttnn_reshape_366 = ttnn.reshape(
        ttnn_matmul_19,
        [2, 45, 1536],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_19, False)
    ttnn_concat_137 = ttnn.concat(
        [ttnn_slice_153, ttnn_reshape_366],
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_366, False)
    ttnn.deallocate(ttnn_slice_153, False)
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
        ttnn_slice_47,
        [1, 1, 2, 18432],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_47, False)
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
        self.weights["transformer.transformer_blocks.4.norm1_context.linear.bias.f32"],
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
        var_1,
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
        self.weights["transformer.transformer_blocks.4.attn.fused_add_q_proj_add_k_proj_add_v_proj.weight"],
        bias=self.weights["transformer.transformer_blocks.4.attn.fused_add_q_proj_add_k_proj_add_v_proj.bias"],
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
        weight=self.weights["transformer.transformer_blocks.4.attn.norm_added_q.weight"],
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
        ttnn_typecast_141,
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
        ttnn_transformer_concatenate_heads_3,
        [0, 45, 0],
        [2, 4141, 768],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_3, False)
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
        self.weights["transformer.transformer_blocks.3.attn.to_out.0.weight"],
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
        self.weights["transformer.transformer_blocks.3.attn.to_out.0.bias.reshaped"],
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
        ttnn_add_62,
        ttnn_multiply_50,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_50, False)
    ttnn.deallocate(ttnn_add_62, False)
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
        ttnn_typecast_141,
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
        ttnn_typecast_141,
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
        var_1,
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
        ttnn_typecast_141,
        [0, 9216],
        [2, 12288],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_141, False)
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
        self.weights["transformer.transformer_blocks.3.ff.net.0.proj.weight"],
        bias=self.weights["transformer.transformer_blocks.3.ff.net.0.proj.bias"],
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
        self.weights["transformer.transformer_blocks.3.ff.net.2.weight"],
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
        self.weights["transformer.transformer_blocks.3.ff.net.2.bias.reshaped"],
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
        ttnn_slice_46,
        [1, 1, 2, 18432],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_46, False)
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
        self.weights["transformer.transformer_blocks.4.norm1.linear.bias.f32"],
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
        var_1,
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
        self.weights["transformer.transformer_blocks.4.attn.fused_to_q_to_k_to_v.weight"],
        bias=self.weights["transformer.transformer_blocks.4.attn.fused_to_q_to_k_to_v.bias"],
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
        weight=self.weights["transformer.transformer_blocks.4.attn.norm_q.weight"],
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
        ttnn_reshape_203,
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
        ttnn_reshape_209,
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
        weight=self.weights["transformer.transformer_blocks.4.attn.norm_added_k.weight"],
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
        weight=self.weights["transformer.transformer_blocks.4.attn.norm_k.weight"],
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
        ttnn_reshape_203,
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
        ttnn_reshape_209,
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
        self.weights["transformer.transformer_blocks.4.attn.to_add_out.weight"],
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
        self.weights["transformer.transformer_blocks.4.attn.to_add_out.bias.reshaped"],
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
        var_1,
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
        self.weights["transformer.transformer_blocks.4.ff_context.net.0.proj.weight"],
        bias=self.weights["transformer.transformer_blocks.4.ff_context.net.0.proj.bias"],
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
        self.weights["transformer.transformer_blocks.4.ff_context.net.2.weight"],
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
        self.weights["transformer.transformer_blocks.4.ff_context.net.2.bias.reshaped"],
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
        text_encoder_layer_5,
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


def _tb_forward_5(self, text_encoder_layer_6, ttnn_add_82, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_411, ttnn_slice_178, ttnn_slice_44, ttnn_slice_45, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_4, ttnn_typecast_149, var_1):
    ttnn_matmul_24 = ttnn.matmul(
        ttnn_reshape_411,
        self.weights["transformer.caption_projection.5.linear.weight"],
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
    ttnn.deallocate(ttnn_reshape_411, False)
    ttnn_reshape_412 = ttnn.reshape(
        ttnn_matmul_24,
        [2, 45, 1536],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_24, False)
    ttnn_concat_143 = ttnn.concat(
        [ttnn_slice_178, ttnn_reshape_412],
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_412, False)
    ttnn.deallocate(ttnn_slice_178, False)
    ttnn_layer_norm_19 = ttnn.layer_norm(
        ttnn_concat_143,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_reshape_413 = ttnn.reshape(
        ttnn_slice_45,
        [1, 1, 2, 18432],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_45, False)
    ttnn_reduce_scatter_28 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_413,
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
    ttnn.deallocate(ttnn_reshape_413, False)
    ttnn_reshape_414 = ttnn.reshape(
        ttnn_reduce_scatter_28,
        [2, 4608],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_28, False)
    ttnn_all_gather_28 = ttnn.all_gather(
        input_tensor=ttnn_reshape_414,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_414, False)
    ttnn_add_94 = ttnn.add(
        ttnn_all_gather_28,
        self.weights["transformer.transformer_blocks.5.norm1_context.linear.bias.f32"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_28, False)
    ttnn_typecast_156 = ttnn.typecast(
        ttnn_add_94,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_94, False)
    ttnn_slice_179 = ttnn.slice(
        ttnn_typecast_156,
        [0, 6144],
        [2, 9216],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_415 = ttnn.reshape(
        ttnn_slice_179,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_179, False)
    ttnn_slice_180 = ttnn.slice(
        ttnn_typecast_156,
        [0, 3072],
        [2, 6144],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_416 = ttnn.reshape(
        ttnn_slice_180,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_180, False)
    ttnn_add_95 = ttnn.add(
        ttnn_reshape_416,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_416, False)
    ttnn_multiply_61 = ttnn.multiply(
        ttnn_layer_norm_19,
        ttnn_add_95,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_95, False)
    ttnn.deallocate(ttnn_layer_norm_19, False)
    ttnn_slice_181 = ttnn.slice(
        ttnn_typecast_156,
        [0, 0],
        [2, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_417 = ttnn.reshape(
        ttnn_slice_181,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_181, False)
    ttnn_add_96 = ttnn.add(
        ttnn_multiply_61,
        ttnn_reshape_417,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_417, False)
    ttnn.deallocate(ttnn_multiply_61, False)
    ttnn_reshape_418 = ttnn.reshape(
        ttnn_add_96,
        [90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_96, False)
    ttnn_linear_23 = ttnn.linear(
        ttnn_reshape_418,
        self.weights["transformer.transformer_blocks.5.attn.fused_add_q_proj_add_k_proj_add_v_proj.weight"],
        bias=self.weights["transformer.transformer_blocks.5.attn.fused_add_q_proj_add_k_proj_add_v_proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_418, False)
    ttnn_slice_182 = ttnn.slice(
        ttnn_linear_23,
        [0, 0],
        [90, 768],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_183 = ttnn.slice(
        ttnn_linear_23,
        [0, 768],
        [90, 1536],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_184 = ttnn.slice(
        ttnn_linear_23,
        [0, 1536],
        [90, 2304],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_23, False)
    ttnn_reshape_419 = ttnn.reshape(
        ttnn_slice_182,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_182, False)
    ttnn_rms_norm_20 = ttnn.rms_norm(
        ttnn_reshape_419,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.5.attn.norm_added_q.weight"],
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
    ttnn.deallocate(ttnn_reshape_419, False)
    ttnn_slice_185 = ttnn.slice(
        ttnn_typecast_149,
        [0, 6144],
        [2, 9216],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_420 = ttnn.reshape(
        ttnn_slice_185,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_185, False)
    ttnn_slice_186 = ttnn.slice(
        ttnn_transformer_concatenate_heads_4,
        [0, 45, 0],
        [2, 4141, 768],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_4, False)
    ttnn_reshape_421 = ttnn.reshape(
        ttnn_slice_186,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_186, False)
    ttnn_matmul_25 = ttnn.matmul(
        ttnn_reshape_421,
        self.weights["transformer.transformer_blocks.4.attn.to_out.0.weight"],
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
    ttnn.deallocate(ttnn_reshape_421, False)
    ttnn_reshape_422 = ttnn.reshape(
        ttnn_matmul_25,
        [1, 1, 8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_25, False)
    ttnn_reduce_scatter_29 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_422,
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
    ttnn.deallocate(ttnn_reshape_422, False)
    ttnn_reshape_423 = ttnn.reshape(
        ttnn_reduce_scatter_29,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_29, False)
    ttnn_all_gather_29 = ttnn.all_gather(
        input_tensor=ttnn_reshape_423,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_423, False)
    ttnn_add_97 = ttnn.add(
        ttnn_all_gather_29,
        self.weights["transformer.transformer_blocks.4.attn.to_out.0.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_29, False)
    ttnn_reshape_424 = ttnn.reshape(
        ttnn_add_97,
        [2, 4096, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_97, False)
    ttnn_multiply_62 = ttnn.multiply(
        ttnn_reshape_420,
        ttnn_reshape_424,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_424, False)
    ttnn.deallocate(ttnn_reshape_420, False)
    ttnn_add_98 = ttnn.add(
        ttnn_add_82,
        ttnn_multiply_62,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_62, False)
    ttnn.deallocate(ttnn_add_82, False)
    ttnn_layer_norm_20 = ttnn.layer_norm(
        ttnn_add_98,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_slice_187 = ttnn.slice(
        ttnn_typecast_149,
        [0, 15360],
        [2, 18432],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_425 = ttnn.reshape(
        ttnn_slice_187,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_187, False)
    ttnn_slice_188 = ttnn.slice(
        ttnn_typecast_149,
        [0, 12288],
        [2, 15360],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_426 = ttnn.reshape(
        ttnn_slice_188,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_188, False)
    ttnn_add_99 = ttnn.add(
        ttnn_reshape_426,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_426, False)
    ttnn_multiply_63 = ttnn.multiply(
        ttnn_layer_norm_20,
        ttnn_add_99,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_99, False)
    ttnn.deallocate(ttnn_layer_norm_20, False)
    ttnn_slice_189 = ttnn.slice(
        ttnn_typecast_149,
        [0, 9216],
        [2, 12288],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_149, False)
    ttnn_reshape_427 = ttnn.reshape(
        ttnn_slice_189,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_189, False)
    ttnn_add_100 = ttnn.add(
        ttnn_multiply_63,
        ttnn_reshape_427,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_427, False)
    ttnn.deallocate(ttnn_multiply_63, False)
    ttnn_reshape_428 = ttnn.reshape(
        ttnn_add_100,
        [8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_100, False)
    ttnn_linear_24 = ttnn.linear(
        ttnn_reshape_428,
        self.weights["transformer.transformer_blocks.4.ff.net.0.proj.weight"],
        bias=self.weights["transformer.transformer_blocks.4.ff.net.0.proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_428, False)
    ttnn_matmul_26 = ttnn.matmul(
        ttnn_linear_24,
        self.weights["transformer.transformer_blocks.4.ff.net.2.weight"],
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
    ttnn.deallocate(ttnn_linear_24, False)
    ttnn_reshape_429 = ttnn.reshape(
        ttnn_matmul_26,
        [1, 1, 8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_26, False)
    ttnn_reduce_scatter_30 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_429,
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
    ttnn.deallocate(ttnn_reshape_429, False)
    ttnn_reshape_430 = ttnn.reshape(
        ttnn_reduce_scatter_30,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_30, False)
    ttnn_all_gather_30 = ttnn.all_gather(
        input_tensor=ttnn_reshape_430,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_430, False)
    ttnn_add_101 = ttnn.add(
        ttnn_all_gather_30,
        self.weights["transformer.transformer_blocks.4.ff.net.2.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_30, False)
    ttnn_reshape_431 = ttnn.reshape(
        ttnn_add_101,
        [2, 4096, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_101, False)
    ttnn_multiply_64 = ttnn.multiply(
        ttnn_reshape_425,
        ttnn_reshape_431,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_431, False)
    ttnn.deallocate(ttnn_reshape_425, False)
    ttnn_add_102 = ttnn.add(
        ttnn_add_98,
        ttnn_multiply_64,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_64, False)
    ttnn.deallocate(ttnn_add_98, False)
    ttnn_layer_norm_21 = ttnn.layer_norm(
        ttnn_add_102,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_reshape_432 = ttnn.reshape(
        ttnn_slice_44,
        [1, 1, 2, 18432],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_44, False)
    ttnn_reduce_scatter_31 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_432,
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
    ttnn.deallocate(ttnn_reshape_432, False)
    ttnn_reshape_433 = ttnn.reshape(
        ttnn_reduce_scatter_31,
        [2, 4608],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_31, False)
    ttnn_all_gather_31 = ttnn.all_gather(
        input_tensor=ttnn_reshape_433,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_433, False)
    ttnn_add_103 = ttnn.add(
        ttnn_all_gather_31,
        self.weights["transformer.transformer_blocks.5.norm1.linear.bias.f32"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_31, False)
    ttnn_typecast_157 = ttnn.typecast(
        ttnn_add_103,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_103, False)
    ttnn_slice_190 = ttnn.slice(
        ttnn_typecast_157,
        [0, 3072],
        [2, 6144],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_434 = ttnn.reshape(
        ttnn_slice_190,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_190, False)
    ttnn_add_104 = ttnn.add(
        ttnn_reshape_434,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_434, False)
    ttnn_multiply_65 = ttnn.multiply(
        ttnn_layer_norm_21,
        ttnn_add_104,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_104, False)
    ttnn.deallocate(ttnn_layer_norm_21, False)
    ttnn_slice_191 = ttnn.slice(
        ttnn_typecast_157,
        [0, 0],
        [2, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_435 = ttnn.reshape(
        ttnn_slice_191,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_191, False)
    ttnn_add_105 = ttnn.add(
        ttnn_multiply_65,
        ttnn_reshape_435,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_435, False)
    ttnn.deallocate(ttnn_multiply_65, False)
    ttnn_reshape_436 = ttnn.reshape(
        ttnn_add_105,
        [8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_105, False)
    ttnn_linear_25 = ttnn.linear(
        ttnn_reshape_436,
        self.weights["transformer.transformer_blocks.5.attn.fused_to_q_to_k_to_v.weight"],
        bias=self.weights["transformer.transformer_blocks.5.attn.fused_to_q_to_k_to_v.bias"],
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
    ttnn.deallocate(ttnn_reshape_436, False)
    ttnn_slice_192 = ttnn.slice(
        ttnn_linear_25,
        [0, 0],
        [8192, 768],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_193 = ttnn.slice(
        ttnn_linear_25,
        [0, 768],
        [8192, 1536],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_194 = ttnn.slice(
        ttnn_linear_25,
        [0, 1536],
        [8192, 2304],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_25, False)
    ttnn_reshape_437 = ttnn.reshape(
        ttnn_slice_192,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_192, False)
    ttnn_rms_norm_21 = ttnn.rms_norm(
        ttnn_reshape_437,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.5.attn.norm_q.weight"],
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
    ttnn.deallocate(ttnn_reshape_437, False)
    ttnn_concat_144 = ttnn.concat(
        [ttnn_rms_norm_20, ttnn_rms_norm_21],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_21, False)
    ttnn.deallocate(ttnn_rms_norm_20, False)
    ttnn_typecast_158 = ttnn.typecast(
        ttnn_concat_144,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_282 = ttnn.permute(
        ttnn_typecast_158,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_158, False)
    ttnn_multiply_66 = ttnn.multiply(
        ttnn_permute_282,
        ttnn_reshape_203,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_282, False)
    ttnn_reshape_438 = ttnn.reshape(
        ttnn_concat_144,
        [2, 4141, 6, 64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_144, False)
    ttnn_slice_195 = ttnn.slice(
        ttnn_reshape_438,
        [0, 0, 0, 0, 1],
        [2, 4141, 6, 64, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_10 = ttnn.neg(
        ttnn_slice_195,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_195, False)
    ttnn_slice_196 = ttnn.slice(
        ttnn_reshape_438,
        [0, 0, 0, 0, 0],
        [2, 4141, 6, 64, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_438, False)
    ttnn_concat_145 = ttnn.concat(
        [ttnn_neg_10, ttnn_slice_196],
        4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_196, False)
    ttnn.deallocate(ttnn_neg_10, False)
    ttnn_typecast_159 = ttnn.typecast(
        ttnn_concat_145,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_145, False)
    ttnn_reshape_439 = ttnn.reshape(
        ttnn_typecast_159,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_159, False)
    ttnn_permute_283 = ttnn.permute(
        ttnn_reshape_439,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_439, False)
    ttnn_multiply_67 = ttnn.multiply(
        ttnn_permute_283,
        ttnn_reshape_209,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_283, False)
    ttnn_add_106 = ttnn.add(
        ttnn_multiply_66,
        ttnn_multiply_67,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_67, False)
    ttnn.deallocate(ttnn_multiply_66, False)
    ttnn_typecast_160 = ttnn.typecast(
        ttnn_add_106,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_106, False)
    ttnn_reshape_440 = ttnn.reshape(
        ttnn_slice_183,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_183, False)
    ttnn_rms_norm_22 = ttnn.rms_norm(
        ttnn_reshape_440,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.5.attn.norm_added_k.weight"],
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
    ttnn.deallocate(ttnn_reshape_440, False)
    ttnn_reshape_441 = ttnn.reshape(
        ttnn_slice_193,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_193, False)
    ttnn_rms_norm_23 = ttnn.rms_norm(
        ttnn_reshape_441,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.5.attn.norm_k.weight"],
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
    ttnn.deallocate(ttnn_reshape_441, False)
    ttnn_concat_146 = ttnn.concat(
        [ttnn_rms_norm_22, ttnn_rms_norm_23],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_23, False)
    ttnn.deallocate(ttnn_rms_norm_22, False)
    ttnn_typecast_161 = ttnn.typecast(
        ttnn_concat_146,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_284 = ttnn.permute(
        ttnn_typecast_161,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_161, False)
    ttnn_multiply_68 = ttnn.multiply(
        ttnn_permute_284,
        ttnn_reshape_203,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_284, False)
    ttnn_reshape_442 = ttnn.reshape(
        ttnn_concat_146,
        [2, 4141, 6, 64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_146, False)
    ttnn_slice_197 = ttnn.slice(
        ttnn_reshape_442,
        [0, 0, 0, 0, 1],
        [2, 4141, 6, 64, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_11 = ttnn.neg(
        ttnn_slice_197,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_197, False)
    ttnn_slice_198 = ttnn.slice(
        ttnn_reshape_442,
        [0, 0, 0, 0, 0],
        [2, 4141, 6, 64, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_442, False)
    ttnn_concat_147 = ttnn.concat(
        [ttnn_neg_11, ttnn_slice_198],
        4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_198, False)
    ttnn.deallocate(ttnn_neg_11, False)
    ttnn_typecast_162 = ttnn.typecast(
        ttnn_concat_147,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_147, False)
    ttnn_reshape_443 = ttnn.reshape(
        ttnn_typecast_162,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_162, False)
    ttnn_permute_285 = ttnn.permute(
        ttnn_reshape_443,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_443, False)
    ttnn_multiply_69 = ttnn.multiply(
        ttnn_permute_285,
        ttnn_reshape_209,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_285, False)
    ttnn_add_107 = ttnn.add(
        ttnn_multiply_68,
        ttnn_multiply_69,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_69, False)
    ttnn.deallocate(ttnn_multiply_68, False)
    ttnn_typecast_163 = ttnn.typecast(
        ttnn_add_107,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_107, False)
    ttnn_reshape_444 = ttnn.reshape(
        ttnn_slice_184,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_184, False)
    ttnn_reshape_445 = ttnn.reshape(
        ttnn_slice_194,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_194, False)
    ttnn_concat_148 = ttnn.concat(
        [ttnn_reshape_444, ttnn_reshape_445],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_445, False)
    ttnn.deallocate(ttnn_reshape_444, False)
    ttnn_permute_286 = ttnn.permute(
        ttnn_concat_148,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_concat_148, False)
    ttnn_transformer_scaled_dot_product_attention_5 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_typecast_160,
            ttnn_typecast_163,
            ttnn_permute_286,
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
    ttnn.deallocate(ttnn_permute_286, False)
    ttnn.deallocate(ttnn_typecast_163, False)
    ttnn.deallocate(ttnn_typecast_160, False)
    ttnn_transformer_concatenate_heads_5 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_5,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_5, False)
    ttnn_slice_199 = ttnn.slice(
        ttnn_transformer_concatenate_heads_5,
        [0, 0, 0],
        [2, 45, 768],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_446 = ttnn.reshape(
        ttnn_slice_199,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_199, False)
    ttnn_matmul_27 = ttnn.matmul(
        ttnn_reshape_446,
        self.weights["transformer.transformer_blocks.5.attn.to_add_out.weight"],
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
    ttnn.deallocate(ttnn_reshape_446, False)
    ttnn_reshape_447 = ttnn.reshape(
        ttnn_matmul_27,
        [1, 1, 90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_27, False)
    ttnn_reduce_scatter_32 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_447,
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
    ttnn.deallocate(ttnn_reshape_447, False)
    ttnn_reshape_448 = ttnn.reshape(
        ttnn_reduce_scatter_32,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_32, False)
    ttnn_all_gather_32 = ttnn.all_gather(
        input_tensor=ttnn_reshape_448,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_448, False)
    ttnn_add_108 = ttnn.add(
        ttnn_all_gather_32,
        self.weights["transformer.transformer_blocks.5.attn.to_add_out.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_32, False)
    ttnn_reshape_449 = ttnn.reshape(
        ttnn_add_108,
        [2, 45, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_108, False)
    ttnn_multiply_70 = ttnn.multiply(
        ttnn_reshape_415,
        ttnn_reshape_449,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_449, False)
    ttnn.deallocate(ttnn_reshape_415, False)
    ttnn_add_109 = ttnn.add(
        ttnn_concat_143,
        ttnn_multiply_70,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_70, False)
    ttnn.deallocate(ttnn_concat_143, False)
    ttnn_layer_norm_22 = ttnn.layer_norm(
        ttnn_add_109,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_slice_200 = ttnn.slice(
        ttnn_typecast_156,
        [0, 15360],
        [2, 18432],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_450 = ttnn.reshape(
        ttnn_slice_200,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_200, False)
    ttnn_slice_201 = ttnn.slice(
        ttnn_typecast_156,
        [0, 12288],
        [2, 15360],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_451 = ttnn.reshape(
        ttnn_slice_201,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_201, False)
    ttnn_add_110 = ttnn.add(
        ttnn_reshape_451,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_451, False)
    ttnn_multiply_71 = ttnn.multiply(
        ttnn_layer_norm_22,
        ttnn_add_110,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_110, False)
    ttnn.deallocate(ttnn_layer_norm_22, False)
    ttnn_slice_202 = ttnn.slice(
        ttnn_typecast_156,
        [0, 9216],
        [2, 12288],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_156, False)
    ttnn_reshape_452 = ttnn.reshape(
        ttnn_slice_202,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_202, False)
    ttnn_add_111 = ttnn.add(
        ttnn_multiply_71,
        ttnn_reshape_452,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_452, False)
    ttnn.deallocate(ttnn_multiply_71, False)
    ttnn_reshape_453 = ttnn.reshape(
        ttnn_add_111,
        [90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_111, False)
    ttnn_linear_26 = ttnn.linear(
        ttnn_reshape_453,
        self.weights["transformer.transformer_blocks.5.ff_context.net.0.proj.weight"],
        bias=self.weights["transformer.transformer_blocks.5.ff_context.net.0.proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_453, False)
    ttnn_matmul_28 = ttnn.matmul(
        ttnn_linear_26,
        self.weights["transformer.transformer_blocks.5.ff_context.net.2.weight"],
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
    ttnn.deallocate(ttnn_linear_26, False)
    ttnn_reshape_454 = ttnn.reshape(
        ttnn_matmul_28,
        [1, 1, 90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_28, False)
    ttnn_reduce_scatter_33 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_454,
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
    ttnn.deallocate(ttnn_reshape_454, False)
    ttnn_reshape_455 = ttnn.reshape(
        ttnn_reduce_scatter_33,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_33, False)
    ttnn_all_gather_33 = ttnn.all_gather(
        input_tensor=ttnn_reshape_455,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_455, False)
    ttnn_add_112 = ttnn.add(
        ttnn_all_gather_33,
        self.weights["transformer.transformer_blocks.5.ff_context.net.2.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_33, False)
    ttnn_reshape_456 = ttnn.reshape(
        ttnn_add_112,
        [2, 45, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_112, False)
    ttnn_multiply_72 = ttnn.multiply(
        ttnn_reshape_450,
        ttnn_reshape_456,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_456, False)
    ttnn.deallocate(ttnn_reshape_450, False)
    ttnn_add_113 = ttnn.add(
        ttnn_add_109,
        ttnn_multiply_72,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_72, False)
    ttnn.deallocate(ttnn_add_109, False)
    ttnn_slice_203 = ttnn.slice(
        ttnn_add_113,
        [0, 0, 0],
        [2, 45, 1536],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_113, False)
    ttnn_to_layout_596 = ttnn.to_layout(
        text_encoder_layer_6,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_457 = ttnn.reshape(
        ttnn_to_layout_596,
        [90, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_596, False)
    return ttnn_add_102, ttnn_reshape_457, ttnn_slice_203, ttnn_transformer_concatenate_heads_5, ttnn_typecast_157


def _tb_forward_6(self, text_encoder_layer_7, ttnn_add_102, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_457, ttnn_slice_203, ttnn_slice_42, ttnn_slice_43, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_5, ttnn_typecast_157, var_1):
    ttnn_matmul_29 = ttnn.matmul(
        ttnn_reshape_457,
        self.weights["transformer.caption_projection.6.linear.weight"],
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
    ttnn.deallocate(ttnn_reshape_457, False)
    ttnn_reshape_458 = ttnn.reshape(
        ttnn_matmul_29,
        [2, 45, 1536],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_29, False)
    ttnn_concat_149 = ttnn.concat(
        [ttnn_slice_203, ttnn_reshape_458],
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_458, False)
    ttnn.deallocate(ttnn_slice_203, False)
    ttnn_layer_norm_23 = ttnn.layer_norm(
        ttnn_concat_149,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_reshape_459 = ttnn.reshape(
        ttnn_slice_43,
        [1, 1, 2, 18432],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_43, False)
    ttnn_reduce_scatter_34 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_459,
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
    ttnn.deallocate(ttnn_reshape_459, False)
    ttnn_reshape_460 = ttnn.reshape(
        ttnn_reduce_scatter_34,
        [2, 4608],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_34, False)
    ttnn_all_gather_34 = ttnn.all_gather(
        input_tensor=ttnn_reshape_460,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_460, False)
    ttnn_add_114 = ttnn.add(
        ttnn_all_gather_34,
        self.weights["transformer.transformer_blocks.6.norm1_context.linear.bias.f32"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_34, False)
    ttnn_typecast_164 = ttnn.typecast(
        ttnn_add_114,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_114, False)
    ttnn_slice_204 = ttnn.slice(
        ttnn_typecast_164,
        [0, 6144],
        [2, 9216],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_461 = ttnn.reshape(
        ttnn_slice_204,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_204, False)
    ttnn_slice_205 = ttnn.slice(
        ttnn_typecast_164,
        [0, 3072],
        [2, 6144],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_462 = ttnn.reshape(
        ttnn_slice_205,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_205, False)
    ttnn_add_115 = ttnn.add(
        ttnn_reshape_462,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_462, False)
    ttnn_multiply_73 = ttnn.multiply(
        ttnn_layer_norm_23,
        ttnn_add_115,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_115, False)
    ttnn.deallocate(ttnn_layer_norm_23, False)
    ttnn_slice_206 = ttnn.slice(
        ttnn_typecast_164,
        [0, 0],
        [2, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_463 = ttnn.reshape(
        ttnn_slice_206,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_206, False)
    ttnn_add_116 = ttnn.add(
        ttnn_multiply_73,
        ttnn_reshape_463,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_463, False)
    ttnn.deallocate(ttnn_multiply_73, False)
    ttnn_reshape_464 = ttnn.reshape(
        ttnn_add_116,
        [90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_116, False)
    ttnn_linear_27 = ttnn.linear(
        ttnn_reshape_464,
        self.weights["transformer.transformer_blocks.6.attn.fused_add_q_proj_add_k_proj_add_v_proj.weight"],
        bias=self.weights["transformer.transformer_blocks.6.attn.fused_add_q_proj_add_k_proj_add_v_proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_464, False)
    ttnn_slice_207 = ttnn.slice(
        ttnn_linear_27,
        [0, 0],
        [90, 768],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_208 = ttnn.slice(
        ttnn_linear_27,
        [0, 768],
        [90, 1536],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_209 = ttnn.slice(
        ttnn_linear_27,
        [0, 1536],
        [90, 2304],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_27, False)
    ttnn_reshape_465 = ttnn.reshape(
        ttnn_slice_207,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_207, False)
    ttnn_rms_norm_24 = ttnn.rms_norm(
        ttnn_reshape_465,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.6.attn.norm_added_q.weight"],
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
    ttnn.deallocate(ttnn_reshape_465, False)
    ttnn_slice_210 = ttnn.slice(
        ttnn_typecast_157,
        [0, 6144],
        [2, 9216],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_466 = ttnn.reshape(
        ttnn_slice_210,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_210, False)
    ttnn_slice_211 = ttnn.slice(
        ttnn_transformer_concatenate_heads_5,
        [0, 45, 0],
        [2, 4141, 768],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_5, False)
    ttnn_reshape_467 = ttnn.reshape(
        ttnn_slice_211,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_211, False)
    ttnn_matmul_30 = ttnn.matmul(
        ttnn_reshape_467,
        self.weights["transformer.transformer_blocks.5.attn.to_out.0.weight"],
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
    ttnn.deallocate(ttnn_reshape_467, False)
    ttnn_reshape_468 = ttnn.reshape(
        ttnn_matmul_30,
        [1, 1, 8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_30, False)
    ttnn_reduce_scatter_35 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_468,
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
    ttnn.deallocate(ttnn_reshape_468, False)
    ttnn_reshape_469 = ttnn.reshape(
        ttnn_reduce_scatter_35,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_35, False)
    ttnn_all_gather_35 = ttnn.all_gather(
        input_tensor=ttnn_reshape_469,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_469, False)
    ttnn_add_117 = ttnn.add(
        ttnn_all_gather_35,
        self.weights["transformer.transformer_blocks.5.attn.to_out.0.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_35, False)
    ttnn_reshape_470 = ttnn.reshape(
        ttnn_add_117,
        [2, 4096, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_117, False)
    ttnn_multiply_74 = ttnn.multiply(
        ttnn_reshape_466,
        ttnn_reshape_470,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_470, False)
    ttnn.deallocate(ttnn_reshape_466, False)
    ttnn_add_118 = ttnn.add(
        ttnn_add_102,
        ttnn_multiply_74,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_74, False)
    ttnn.deallocate(ttnn_add_102, False)
    ttnn_layer_norm_24 = ttnn.layer_norm(
        ttnn_add_118,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_slice_212 = ttnn.slice(
        ttnn_typecast_157,
        [0, 15360],
        [2, 18432],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_471 = ttnn.reshape(
        ttnn_slice_212,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_212, False)
    ttnn_slice_213 = ttnn.slice(
        ttnn_typecast_157,
        [0, 12288],
        [2, 15360],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_472 = ttnn.reshape(
        ttnn_slice_213,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_213, False)
    ttnn_add_119 = ttnn.add(
        ttnn_reshape_472,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_472, False)
    ttnn_multiply_75 = ttnn.multiply(
        ttnn_layer_norm_24,
        ttnn_add_119,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_119, False)
    ttnn.deallocate(ttnn_layer_norm_24, False)
    ttnn_slice_214 = ttnn.slice(
        ttnn_typecast_157,
        [0, 9216],
        [2, 12288],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_157, False)
    ttnn_reshape_473 = ttnn.reshape(
        ttnn_slice_214,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_214, False)
    ttnn_add_120 = ttnn.add(
        ttnn_multiply_75,
        ttnn_reshape_473,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_473, False)
    ttnn.deallocate(ttnn_multiply_75, False)
    ttnn_reshape_474 = ttnn.reshape(
        ttnn_add_120,
        [8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_120, False)
    ttnn_linear_28 = ttnn.linear(
        ttnn_reshape_474,
        self.weights["transformer.transformer_blocks.5.ff.net.0.proj.weight"],
        bias=self.weights["transformer.transformer_blocks.5.ff.net.0.proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_474, False)
    ttnn_matmul_31 = ttnn.matmul(
        ttnn_linear_28,
        self.weights["transformer.transformer_blocks.5.ff.net.2.weight"],
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
    ttnn.deallocate(ttnn_linear_28, False)
    ttnn_reshape_475 = ttnn.reshape(
        ttnn_matmul_31,
        [1, 1, 8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_31, False)
    ttnn_reduce_scatter_36 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_475,
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
    ttnn.deallocate(ttnn_reshape_475, False)
    ttnn_reshape_476 = ttnn.reshape(
        ttnn_reduce_scatter_36,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_36, False)
    ttnn_all_gather_36 = ttnn.all_gather(
        input_tensor=ttnn_reshape_476,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_476, False)
    ttnn_add_121 = ttnn.add(
        ttnn_all_gather_36,
        self.weights["transformer.transformer_blocks.5.ff.net.2.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_36, False)
    ttnn_reshape_477 = ttnn.reshape(
        ttnn_add_121,
        [2, 4096, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_121, False)
    ttnn_multiply_76 = ttnn.multiply(
        ttnn_reshape_471,
        ttnn_reshape_477,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_477, False)
    ttnn.deallocate(ttnn_reshape_471, False)
    ttnn_add_122 = ttnn.add(
        ttnn_add_118,
        ttnn_multiply_76,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_76, False)
    ttnn.deallocate(ttnn_add_118, False)
    ttnn_layer_norm_25 = ttnn.layer_norm(
        ttnn_add_122,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_reshape_478 = ttnn.reshape(
        ttnn_slice_42,
        [1, 1, 2, 18432],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_42, False)
    ttnn_reduce_scatter_37 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_478,
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
    ttnn.deallocate(ttnn_reshape_478, False)
    ttnn_reshape_479 = ttnn.reshape(
        ttnn_reduce_scatter_37,
        [2, 4608],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_37, False)
    ttnn_all_gather_37 = ttnn.all_gather(
        input_tensor=ttnn_reshape_479,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_479, False)
    ttnn_add_123 = ttnn.add(
        ttnn_all_gather_37,
        self.weights["transformer.transformer_blocks.6.norm1.linear.bias.f32"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_37, False)
    ttnn_typecast_165 = ttnn.typecast(
        ttnn_add_123,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_123, False)
    ttnn_slice_215 = ttnn.slice(
        ttnn_typecast_165,
        [0, 3072],
        [2, 6144],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_480 = ttnn.reshape(
        ttnn_slice_215,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_215, False)
    ttnn_add_124 = ttnn.add(
        ttnn_reshape_480,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_480, False)
    ttnn_multiply_77 = ttnn.multiply(
        ttnn_layer_norm_25,
        ttnn_add_124,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_124, False)
    ttnn.deallocate(ttnn_layer_norm_25, False)
    ttnn_slice_216 = ttnn.slice(
        ttnn_typecast_165,
        [0, 0],
        [2, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_481 = ttnn.reshape(
        ttnn_slice_216,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_216, False)
    ttnn_add_125 = ttnn.add(
        ttnn_multiply_77,
        ttnn_reshape_481,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_481, False)
    ttnn.deallocate(ttnn_multiply_77, False)
    ttnn_reshape_482 = ttnn.reshape(
        ttnn_add_125,
        [8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_125, False)
    ttnn_linear_29 = ttnn.linear(
        ttnn_reshape_482,
        self.weights["transformer.transformer_blocks.6.attn.fused_to_q_to_k_to_v.weight"],
        bias=self.weights["transformer.transformer_blocks.6.attn.fused_to_q_to_k_to_v.bias"],
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
    ttnn.deallocate(ttnn_reshape_482, False)
    ttnn_slice_217 = ttnn.slice(
        ttnn_linear_29,
        [0, 0],
        [8192, 768],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_218 = ttnn.slice(
        ttnn_linear_29,
        [0, 768],
        [8192, 1536],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_219 = ttnn.slice(
        ttnn_linear_29,
        [0, 1536],
        [8192, 2304],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_29, False)
    ttnn_reshape_483 = ttnn.reshape(
        ttnn_slice_217,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_217, False)
    ttnn_rms_norm_25 = ttnn.rms_norm(
        ttnn_reshape_483,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.6.attn.norm_q.weight"],
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
    ttnn.deallocate(ttnn_reshape_483, False)
    ttnn_concat_150 = ttnn.concat(
        [ttnn_rms_norm_24, ttnn_rms_norm_25],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_25, False)
    ttnn.deallocate(ttnn_rms_norm_24, False)
    ttnn_typecast_166 = ttnn.typecast(
        ttnn_concat_150,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_287 = ttnn.permute(
        ttnn_typecast_166,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_166, False)
    ttnn_multiply_78 = ttnn.multiply(
        ttnn_permute_287,
        ttnn_reshape_203,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_287, False)
    ttnn_reshape_484 = ttnn.reshape(
        ttnn_concat_150,
        [2, 4141, 6, 64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_150, False)
    ttnn_slice_220 = ttnn.slice(
        ttnn_reshape_484,
        [0, 0, 0, 0, 1],
        [2, 4141, 6, 64, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_12 = ttnn.neg(
        ttnn_slice_220,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_220, False)
    ttnn_slice_221 = ttnn.slice(
        ttnn_reshape_484,
        [0, 0, 0, 0, 0],
        [2, 4141, 6, 64, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_484, False)
    ttnn_concat_151 = ttnn.concat(
        [ttnn_neg_12, ttnn_slice_221],
        4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_221, False)
    ttnn.deallocate(ttnn_neg_12, False)
    ttnn_typecast_167 = ttnn.typecast(
        ttnn_concat_151,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_151, False)
    ttnn_reshape_485 = ttnn.reshape(
        ttnn_typecast_167,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_167, False)
    ttnn_permute_288 = ttnn.permute(
        ttnn_reshape_485,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_485, False)
    ttnn_multiply_79 = ttnn.multiply(
        ttnn_permute_288,
        ttnn_reshape_209,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_288, False)
    ttnn_add_126 = ttnn.add(
        ttnn_multiply_78,
        ttnn_multiply_79,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_79, False)
    ttnn.deallocate(ttnn_multiply_78, False)
    ttnn_typecast_168 = ttnn.typecast(
        ttnn_add_126,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_126, False)
    ttnn_reshape_486 = ttnn.reshape(
        ttnn_slice_208,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_208, False)
    ttnn_rms_norm_26 = ttnn.rms_norm(
        ttnn_reshape_486,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.6.attn.norm_added_k.weight"],
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
    ttnn.deallocate(ttnn_reshape_486, False)
    ttnn_reshape_487 = ttnn.reshape(
        ttnn_slice_218,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_218, False)
    ttnn_rms_norm_27 = ttnn.rms_norm(
        ttnn_reshape_487,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.6.attn.norm_k.weight"],
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
    ttnn.deallocate(ttnn_reshape_487, False)
    ttnn_concat_152 = ttnn.concat(
        [ttnn_rms_norm_26, ttnn_rms_norm_27],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_27, False)
    ttnn.deallocate(ttnn_rms_norm_26, False)
    ttnn_typecast_169 = ttnn.typecast(
        ttnn_concat_152,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_289 = ttnn.permute(
        ttnn_typecast_169,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_169, False)
    ttnn_multiply_80 = ttnn.multiply(
        ttnn_permute_289,
        ttnn_reshape_203,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_289, False)
    ttnn_reshape_488 = ttnn.reshape(
        ttnn_concat_152,
        [2, 4141, 6, 64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_152, False)
    ttnn_slice_222 = ttnn.slice(
        ttnn_reshape_488,
        [0, 0, 0, 0, 1],
        [2, 4141, 6, 64, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_13 = ttnn.neg(
        ttnn_slice_222,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_222, False)
    ttnn_slice_223 = ttnn.slice(
        ttnn_reshape_488,
        [0, 0, 0, 0, 0],
        [2, 4141, 6, 64, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_488, False)
    ttnn_concat_153 = ttnn.concat(
        [ttnn_neg_13, ttnn_slice_223],
        4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_223, False)
    ttnn.deallocate(ttnn_neg_13, False)
    ttnn_typecast_170 = ttnn.typecast(
        ttnn_concat_153,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_153, False)
    ttnn_reshape_489 = ttnn.reshape(
        ttnn_typecast_170,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_170, False)
    ttnn_permute_290 = ttnn.permute(
        ttnn_reshape_489,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_489, False)
    ttnn_multiply_81 = ttnn.multiply(
        ttnn_permute_290,
        ttnn_reshape_209,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_290, False)
    ttnn_add_127 = ttnn.add(
        ttnn_multiply_80,
        ttnn_multiply_81,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_81, False)
    ttnn.deallocate(ttnn_multiply_80, False)
    ttnn_typecast_171 = ttnn.typecast(
        ttnn_add_127,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_127, False)
    ttnn_reshape_490 = ttnn.reshape(
        ttnn_slice_209,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_209, False)
    ttnn_reshape_491 = ttnn.reshape(
        ttnn_slice_219,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_219, False)
    ttnn_concat_154 = ttnn.concat(
        [ttnn_reshape_490, ttnn_reshape_491],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_491, False)
    ttnn.deallocate(ttnn_reshape_490, False)
    ttnn_permute_291 = ttnn.permute(
        ttnn_concat_154,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_concat_154, False)
    ttnn_transformer_scaled_dot_product_attention_6 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_typecast_168,
            ttnn_typecast_171,
            ttnn_permute_291,
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
    ttnn.deallocate(ttnn_permute_291, False)
    ttnn.deallocate(ttnn_typecast_171, False)
    ttnn.deallocate(ttnn_typecast_168, False)
    ttnn_transformer_concatenate_heads_6 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_6,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_6, False)
    ttnn_slice_224 = ttnn.slice(
        ttnn_transformer_concatenate_heads_6,
        [0, 0, 0],
        [2, 45, 768],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_492 = ttnn.reshape(
        ttnn_slice_224,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_224, False)
    ttnn_matmul_32 = ttnn.matmul(
        ttnn_reshape_492,
        self.weights["transformer.transformer_blocks.6.attn.to_add_out.weight"],
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
    ttnn.deallocate(ttnn_reshape_492, False)
    ttnn_reshape_493 = ttnn.reshape(
        ttnn_matmul_32,
        [1, 1, 90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_32, False)
    ttnn_reduce_scatter_38 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_493,
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
    ttnn.deallocate(ttnn_reshape_493, False)
    ttnn_reshape_494 = ttnn.reshape(
        ttnn_reduce_scatter_38,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_38, False)
    ttnn_all_gather_38 = ttnn.all_gather(
        input_tensor=ttnn_reshape_494,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_494, False)
    ttnn_add_128 = ttnn.add(
        ttnn_all_gather_38,
        self.weights["transformer.transformer_blocks.6.attn.to_add_out.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_38, False)
    ttnn_reshape_495 = ttnn.reshape(
        ttnn_add_128,
        [2, 45, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_128, False)
    ttnn_multiply_82 = ttnn.multiply(
        ttnn_reshape_461,
        ttnn_reshape_495,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_495, False)
    ttnn.deallocate(ttnn_reshape_461, False)
    ttnn_add_129 = ttnn.add(
        ttnn_concat_149,
        ttnn_multiply_82,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_82, False)
    ttnn.deallocate(ttnn_concat_149, False)
    ttnn_layer_norm_26 = ttnn.layer_norm(
        ttnn_add_129,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_slice_225 = ttnn.slice(
        ttnn_typecast_164,
        [0, 15360],
        [2, 18432],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_496 = ttnn.reshape(
        ttnn_slice_225,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_225, False)
    ttnn_slice_226 = ttnn.slice(
        ttnn_typecast_164,
        [0, 12288],
        [2, 15360],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_497 = ttnn.reshape(
        ttnn_slice_226,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_226, False)
    ttnn_add_130 = ttnn.add(
        ttnn_reshape_497,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_497, False)
    ttnn_multiply_83 = ttnn.multiply(
        ttnn_layer_norm_26,
        ttnn_add_130,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_130, False)
    ttnn.deallocate(ttnn_layer_norm_26, False)
    ttnn_slice_227 = ttnn.slice(
        ttnn_typecast_164,
        [0, 9216],
        [2, 12288],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_164, False)
    ttnn_reshape_498 = ttnn.reshape(
        ttnn_slice_227,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_227, False)
    ttnn_add_131 = ttnn.add(
        ttnn_multiply_83,
        ttnn_reshape_498,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_498, False)
    ttnn.deallocate(ttnn_multiply_83, False)
    ttnn_reshape_499 = ttnn.reshape(
        ttnn_add_131,
        [90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_131, False)
    ttnn_linear_30 = ttnn.linear(
        ttnn_reshape_499,
        self.weights["transformer.transformer_blocks.6.ff_context.net.0.proj.weight"],
        bias=self.weights["transformer.transformer_blocks.6.ff_context.net.0.proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_499, False)
    ttnn_matmul_33 = ttnn.matmul(
        ttnn_linear_30,
        self.weights["transformer.transformer_blocks.6.ff_context.net.2.weight"],
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
    ttnn.deallocate(ttnn_linear_30, False)
    ttnn_reshape_500 = ttnn.reshape(
        ttnn_matmul_33,
        [1, 1, 90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_33, False)
    ttnn_reduce_scatter_39 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_500,
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
    ttnn.deallocate(ttnn_reshape_500, False)
    ttnn_reshape_501 = ttnn.reshape(
        ttnn_reduce_scatter_39,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_39, False)
    ttnn_all_gather_39 = ttnn.all_gather(
        input_tensor=ttnn_reshape_501,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_501, False)
    ttnn_add_132 = ttnn.add(
        ttnn_all_gather_39,
        self.weights["transformer.transformer_blocks.6.ff_context.net.2.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_39, False)
    ttnn_reshape_502 = ttnn.reshape(
        ttnn_add_132,
        [2, 45, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_132, False)
    ttnn_multiply_84 = ttnn.multiply(
        ttnn_reshape_496,
        ttnn_reshape_502,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_502, False)
    ttnn.deallocate(ttnn_reshape_496, False)
    ttnn_add_133 = ttnn.add(
        ttnn_add_129,
        ttnn_multiply_84,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_84, False)
    ttnn.deallocate(ttnn_add_129, False)
    ttnn_slice_228 = ttnn.slice(
        ttnn_add_133,
        [0, 0, 0],
        [2, 45, 1536],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_133, False)
    ttnn_to_layout_597 = ttnn.to_layout(
        text_encoder_layer_7,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_503 = ttnn.reshape(
        ttnn_to_layout_597,
        [90, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_597, False)
    return ttnn_add_122, ttnn_reshape_503, ttnn_slice_228, ttnn_transformer_concatenate_heads_6, ttnn_typecast_165


def _tb_forward_7(self, text_encoder_layer_8, ttnn_add_122, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_503, ttnn_slice_228, ttnn_slice_40, ttnn_slice_41, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_6, ttnn_typecast_165, var_1):
    ttnn_matmul_34 = ttnn.matmul(
        ttnn_reshape_503,
        self.weights["transformer.caption_projection.7.linear.weight"],
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
    ttnn.deallocate(ttnn_reshape_503, False)
    ttnn_reshape_504 = ttnn.reshape(
        ttnn_matmul_34,
        [2, 45, 1536],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_34, False)
    ttnn_concat_155 = ttnn.concat(
        [ttnn_slice_228, ttnn_reshape_504],
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_504, False)
    ttnn.deallocate(ttnn_slice_228, False)
    ttnn_layer_norm_27 = ttnn.layer_norm(
        ttnn_concat_155,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_reshape_505 = ttnn.reshape(
        ttnn_slice_41,
        [1, 1, 2, 18432],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_41, False)
    ttnn_reduce_scatter_40 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_505,
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
    ttnn.deallocate(ttnn_reshape_505, False)
    ttnn_reshape_506 = ttnn.reshape(
        ttnn_reduce_scatter_40,
        [2, 4608],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_40, False)
    ttnn_all_gather_40 = ttnn.all_gather(
        input_tensor=ttnn_reshape_506,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_506, False)
    ttnn_add_134 = ttnn.add(
        ttnn_all_gather_40,
        self.weights["transformer.transformer_blocks.7.norm1_context.linear.bias.f32"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_40, False)
    ttnn_typecast_172 = ttnn.typecast(
        ttnn_add_134,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_134, False)
    ttnn_slice_229 = ttnn.slice(
        ttnn_typecast_172,
        [0, 6144],
        [2, 9216],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_507 = ttnn.reshape(
        ttnn_slice_229,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_229, False)
    ttnn_slice_230 = ttnn.slice(
        ttnn_typecast_172,
        [0, 3072],
        [2, 6144],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_508 = ttnn.reshape(
        ttnn_slice_230,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_230, False)
    ttnn_add_135 = ttnn.add(
        ttnn_reshape_508,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_508, False)
    ttnn_multiply_85 = ttnn.multiply(
        ttnn_layer_norm_27,
        ttnn_add_135,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_135, False)
    ttnn.deallocate(ttnn_layer_norm_27, False)
    ttnn_slice_231 = ttnn.slice(
        ttnn_typecast_172,
        [0, 0],
        [2, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_509 = ttnn.reshape(
        ttnn_slice_231,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_231, False)
    ttnn_add_136 = ttnn.add(
        ttnn_multiply_85,
        ttnn_reshape_509,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_509, False)
    ttnn.deallocate(ttnn_multiply_85, False)
    ttnn_reshape_510 = ttnn.reshape(
        ttnn_add_136,
        [90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_136, False)
    ttnn_linear_31 = ttnn.linear(
        ttnn_reshape_510,
        self.weights["transformer.transformer_blocks.7.attn.fused_add_q_proj_add_k_proj_add_v_proj.weight"],
        bias=self.weights["transformer.transformer_blocks.7.attn.fused_add_q_proj_add_k_proj_add_v_proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_510, False)
    ttnn_slice_232 = ttnn.slice(
        ttnn_linear_31,
        [0, 0],
        [90, 768],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_233 = ttnn.slice(
        ttnn_linear_31,
        [0, 768],
        [90, 1536],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_234 = ttnn.slice(
        ttnn_linear_31,
        [0, 1536],
        [90, 2304],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_31, False)
    ttnn_reshape_511 = ttnn.reshape(
        ttnn_slice_232,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_232, False)
    ttnn_rms_norm_28 = ttnn.rms_norm(
        ttnn_reshape_511,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.7.attn.norm_added_q.weight"],
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
    ttnn.deallocate(ttnn_reshape_511, False)
    ttnn_slice_235 = ttnn.slice(
        ttnn_typecast_165,
        [0, 6144],
        [2, 9216],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_512 = ttnn.reshape(
        ttnn_slice_235,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_235, False)
    ttnn_slice_236 = ttnn.slice(
        ttnn_transformer_concatenate_heads_6,
        [0, 45, 0],
        [2, 4141, 768],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_6, False)
    ttnn_reshape_513 = ttnn.reshape(
        ttnn_slice_236,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_236, False)
    ttnn_matmul_35 = ttnn.matmul(
        ttnn_reshape_513,
        self.weights["transformer.transformer_blocks.6.attn.to_out.0.weight"],
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
    ttnn.deallocate(ttnn_reshape_513, False)
    ttnn_reshape_514 = ttnn.reshape(
        ttnn_matmul_35,
        [1, 1, 8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_35, False)
    ttnn_reduce_scatter_41 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_514,
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
    ttnn.deallocate(ttnn_reshape_514, False)
    ttnn_reshape_515 = ttnn.reshape(
        ttnn_reduce_scatter_41,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_41, False)
    ttnn_all_gather_41 = ttnn.all_gather(
        input_tensor=ttnn_reshape_515,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_515, False)
    ttnn_add_137 = ttnn.add(
        ttnn_all_gather_41,
        self.weights["transformer.transformer_blocks.6.attn.to_out.0.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_41, False)
    ttnn_reshape_516 = ttnn.reshape(
        ttnn_add_137,
        [2, 4096, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_137, False)
    ttnn_multiply_86 = ttnn.multiply(
        ttnn_reshape_512,
        ttnn_reshape_516,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_516, False)
    ttnn.deallocate(ttnn_reshape_512, False)
    ttnn_add_138 = ttnn.add(
        ttnn_add_122,
        ttnn_multiply_86,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_86, False)
    ttnn.deallocate(ttnn_add_122, False)
    ttnn_layer_norm_28 = ttnn.layer_norm(
        ttnn_add_138,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_slice_237 = ttnn.slice(
        ttnn_typecast_165,
        [0, 15360],
        [2, 18432],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_517 = ttnn.reshape(
        ttnn_slice_237,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_237, False)
    ttnn_slice_238 = ttnn.slice(
        ttnn_typecast_165,
        [0, 12288],
        [2, 15360],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_518 = ttnn.reshape(
        ttnn_slice_238,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_238, False)
    ttnn_add_139 = ttnn.add(
        ttnn_reshape_518,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_518, False)
    ttnn_multiply_87 = ttnn.multiply(
        ttnn_layer_norm_28,
        ttnn_add_139,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_139, False)
    ttnn.deallocate(ttnn_layer_norm_28, False)
    ttnn_slice_239 = ttnn.slice(
        ttnn_typecast_165,
        [0, 9216],
        [2, 12288],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_165, False)
    ttnn_reshape_519 = ttnn.reshape(
        ttnn_slice_239,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_239, False)
    ttnn_add_140 = ttnn.add(
        ttnn_multiply_87,
        ttnn_reshape_519,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_519, False)
    ttnn.deallocate(ttnn_multiply_87, False)
    ttnn_reshape_520 = ttnn.reshape(
        ttnn_add_140,
        [8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_140, False)
    ttnn_linear_32 = ttnn.linear(
        ttnn_reshape_520,
        self.weights["transformer.transformer_blocks.6.ff.net.0.proj.weight"],
        bias=self.weights["transformer.transformer_blocks.6.ff.net.0.proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_520, False)
    ttnn_matmul_36 = ttnn.matmul(
        ttnn_linear_32,
        self.weights["transformer.transformer_blocks.6.ff.net.2.weight"],
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
    ttnn.deallocate(ttnn_linear_32, False)
    ttnn_reshape_521 = ttnn.reshape(
        ttnn_matmul_36,
        [1, 1, 8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_36, False)
    ttnn_reduce_scatter_42 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_521,
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
    ttnn.deallocate(ttnn_reshape_521, False)
    ttnn_reshape_522 = ttnn.reshape(
        ttnn_reduce_scatter_42,
        [8192, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_42, False)
    ttnn_all_gather_42 = ttnn.all_gather(
        input_tensor=ttnn_reshape_522,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_522, False)
    ttnn_add_141 = ttnn.add(
        ttnn_all_gather_42,
        self.weights["transformer.transformer_blocks.6.ff.net.2.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_42, False)
    ttnn_reshape_523 = ttnn.reshape(
        ttnn_add_141,
        [2, 4096, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_141, False)
    ttnn_multiply_88 = ttnn.multiply(
        ttnn_reshape_517,
        ttnn_reshape_523,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_523, False)
    ttnn.deallocate(ttnn_reshape_517, False)
    ttnn_add_142 = ttnn.add(
        ttnn_add_138,
        ttnn_multiply_88,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_88, False)
    ttnn.deallocate(ttnn_add_138, False)
    ttnn_layer_norm_29 = ttnn.layer_norm(
        ttnn_add_142,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_reshape_524 = ttnn.reshape(
        ttnn_slice_40,
        [1, 1, 2, 18432],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_40, False)
    ttnn_reduce_scatter_43 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_524,
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
    ttnn.deallocate(ttnn_reshape_524, False)
    ttnn_reshape_525 = ttnn.reshape(
        ttnn_reduce_scatter_43,
        [2, 4608],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_43, False)
    ttnn_all_gather_43 = ttnn.all_gather(
        input_tensor=ttnn_reshape_525,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_525, False)
    ttnn_add_143 = ttnn.add(
        ttnn_all_gather_43,
        self.weights["transformer.transformer_blocks.7.norm1.linear.bias.f32"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_43, False)
    ttnn_typecast_173 = ttnn.typecast(
        ttnn_add_143,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_143, False)
    ttnn_slice_240 = ttnn.slice(
        ttnn_typecast_173,
        [0, 3072],
        [2, 6144],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_526 = ttnn.reshape(
        ttnn_slice_240,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_240, False)
    ttnn_add_144 = ttnn.add(
        ttnn_reshape_526,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_526, False)
    ttnn_multiply_89 = ttnn.multiply(
        ttnn_layer_norm_29,
        ttnn_add_144,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_144, False)
    ttnn.deallocate(ttnn_layer_norm_29, False)
    ttnn_slice_241 = ttnn.slice(
        ttnn_typecast_173,
        [0, 0],
        [2, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_527 = ttnn.reshape(
        ttnn_slice_241,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_241, False)
    ttnn_add_145 = ttnn.add(
        ttnn_multiply_89,
        ttnn_reshape_527,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_527, False)
    ttnn.deallocate(ttnn_multiply_89, False)
    ttnn_reshape_528 = ttnn.reshape(
        ttnn_add_145,
        [8192, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_145, False)
    ttnn_linear_33 = ttnn.linear(
        ttnn_reshape_528,
        self.weights["transformer.transformer_blocks.7.attn.fused_to_q_to_k_to_v.weight"],
        bias=self.weights["transformer.transformer_blocks.7.attn.fused_to_q_to_k_to_v.bias"],
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
    ttnn.deallocate(ttnn_reshape_528, False)
    ttnn_slice_242 = ttnn.slice(
        ttnn_linear_33,
        [0, 0],
        [8192, 768],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_243 = ttnn.slice(
        ttnn_linear_33,
        [0, 768],
        [8192, 1536],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_244 = ttnn.slice(
        ttnn_linear_33,
        [0, 1536],
        [8192, 2304],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_33, False)
    ttnn_reshape_529 = ttnn.reshape(
        ttnn_slice_242,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_242, False)
    ttnn_rms_norm_29 = ttnn.rms_norm(
        ttnn_reshape_529,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.7.attn.norm_q.weight"],
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
    ttnn.deallocate(ttnn_reshape_529, False)
    ttnn_concat_156 = ttnn.concat(
        [ttnn_rms_norm_28, ttnn_rms_norm_29],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_29, False)
    ttnn.deallocate(ttnn_rms_norm_28, False)
    ttnn_typecast_174 = ttnn.typecast(
        ttnn_concat_156,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_292 = ttnn.permute(
        ttnn_typecast_174,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_174, False)
    ttnn_multiply_90 = ttnn.multiply(
        ttnn_permute_292,
        ttnn_reshape_203,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_292, False)
    ttnn_reshape_530 = ttnn.reshape(
        ttnn_concat_156,
        [2, 4141, 6, 64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_156, False)
    ttnn_slice_245 = ttnn.slice(
        ttnn_reshape_530,
        [0, 0, 0, 0, 1],
        [2, 4141, 6, 64, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_14 = ttnn.neg(
        ttnn_slice_245,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_245, False)
    ttnn_slice_246 = ttnn.slice(
        ttnn_reshape_530,
        [0, 0, 0, 0, 0],
        [2, 4141, 6, 64, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_530, False)
    ttnn_concat_157 = ttnn.concat(
        [ttnn_neg_14, ttnn_slice_246],
        4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_246, False)
    ttnn.deallocate(ttnn_neg_14, False)
    ttnn_typecast_175 = ttnn.typecast(
        ttnn_concat_157,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_157, False)
    ttnn_reshape_531 = ttnn.reshape(
        ttnn_typecast_175,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_175, False)
    ttnn_permute_293 = ttnn.permute(
        ttnn_reshape_531,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_531, False)
    ttnn_multiply_91 = ttnn.multiply(
        ttnn_permute_293,
        ttnn_reshape_209,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_293, False)
    ttnn_add_146 = ttnn.add(
        ttnn_multiply_90,
        ttnn_multiply_91,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_91, False)
    ttnn.deallocate(ttnn_multiply_90, False)
    ttnn_typecast_176 = ttnn.typecast(
        ttnn_add_146,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_146, False)
    ttnn_reshape_532 = ttnn.reshape(
        ttnn_slice_233,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_233, False)
    ttnn_rms_norm_30 = ttnn.rms_norm(
        ttnn_reshape_532,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.7.attn.norm_added_k.weight"],
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
    ttnn.deallocate(ttnn_reshape_532, False)
    ttnn_reshape_533 = ttnn.reshape(
        ttnn_slice_243,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_243, False)
    ttnn_rms_norm_31 = ttnn.rms_norm(
        ttnn_reshape_533,
        epsilon=9.9999999747524271e-07,
        weight=self.weights["transformer.transformer_blocks.7.attn.norm_k.weight"],
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
    ttnn.deallocate(ttnn_reshape_533, False)
    ttnn_concat_158 = ttnn.concat(
        [ttnn_rms_norm_30, ttnn_rms_norm_31],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_31, False)
    ttnn.deallocate(ttnn_rms_norm_30, False)
    ttnn_typecast_177 = ttnn.typecast(
        ttnn_concat_158,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_294 = ttnn.permute(
        ttnn_typecast_177,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_typecast_177, False)
    ttnn_multiply_92 = ttnn.multiply(
        ttnn_permute_294,
        ttnn_reshape_203,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_294, False)
    ttnn_reshape_534 = ttnn.reshape(
        ttnn_concat_158,
        [2, 4141, 6, 64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_158, False)
    ttnn_slice_247 = ttnn.slice(
        ttnn_reshape_534,
        [0, 0, 0, 0, 1],
        [2, 4141, 6, 64, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_neg_15 = ttnn.neg(
        ttnn_slice_247,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_247, False)
    ttnn_slice_248 = ttnn.slice(
        ttnn_reshape_534,
        [0, 0, 0, 0, 0],
        [2, 4141, 6, 64, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_534, False)
    ttnn_concat_159 = ttnn.concat(
        [ttnn_neg_15, ttnn_slice_248],
        4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_248, False)
    ttnn.deallocate(ttnn_neg_15, False)
    ttnn_typecast_178 = ttnn.typecast(
        ttnn_concat_159,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_159, False)
    ttnn_reshape_535 = ttnn.reshape(
        ttnn_typecast_178,
        [2, 4141, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_178, False)
    ttnn_permute_295 = ttnn.permute(
        ttnn_reshape_535,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_535, False)
    ttnn_multiply_93 = ttnn.multiply(
        ttnn_permute_295,
        ttnn_reshape_209,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_295, False)
    ttnn_add_147 = ttnn.add(
        ttnn_multiply_92,
        ttnn_multiply_93,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_93, False)
    ttnn.deallocate(ttnn_multiply_92, False)
    ttnn_typecast_179 = ttnn.typecast(
        ttnn_add_147,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_147, False)
    ttnn_reshape_536 = ttnn.reshape(
        ttnn_slice_234,
        [2, 45, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_234, False)
    ttnn_reshape_537 = ttnn.reshape(
        ttnn_slice_244,
        [2, 4096, 6, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_244, False)
    ttnn_concat_160 = ttnn.concat(
        [ttnn_reshape_536, ttnn_reshape_537],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_537, False)
    ttnn.deallocate(ttnn_reshape_536, False)
    ttnn_permute_296 = ttnn.permute(
        ttnn_concat_160,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_concat_160, False)
    ttnn_transformer_scaled_dot_product_attention_7 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_typecast_176,
            ttnn_typecast_179,
            ttnn_permute_296,
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
    ttnn.deallocate(ttnn_permute_296, False)
    ttnn.deallocate(ttnn_typecast_179, False)
    ttnn.deallocate(ttnn_typecast_176, False)
    ttnn_transformer_concatenate_heads_7 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_7, False)
    ttnn_slice_249 = ttnn.slice(
        ttnn_transformer_concatenate_heads_7,
        [0, 0, 0],
        [2, 45, 768],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_538 = ttnn.reshape(
        ttnn_slice_249,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_249, False)
    ttnn_matmul_37 = ttnn.matmul(
        ttnn_reshape_538,
        self.weights["transformer.transformer_blocks.7.attn.to_add_out.weight"],
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
    ttnn.deallocate(ttnn_reshape_538, False)
    ttnn_reshape_539 = ttnn.reshape(
        ttnn_matmul_37,
        [1, 1, 90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_37, False)
    ttnn_reduce_scatter_44 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_539,
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
    ttnn.deallocate(ttnn_reshape_539, False)
    ttnn_reshape_540 = ttnn.reshape(
        ttnn_reduce_scatter_44,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_44, False)
    ttnn_all_gather_44 = ttnn.all_gather(
        input_tensor=ttnn_reshape_540,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_540, False)
    ttnn_add_148 = ttnn.add(
        ttnn_all_gather_44,
        self.weights["transformer.transformer_blocks.7.attn.to_add_out.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_44, False)
    ttnn_reshape_541 = ttnn.reshape(
        ttnn_add_148,
        [2, 45, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_148, False)
    ttnn_multiply_94 = ttnn.multiply(
        ttnn_reshape_507,
        ttnn_reshape_541,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_541, False)
    ttnn.deallocate(ttnn_reshape_507, False)
    ttnn_add_149 = ttnn.add(
        ttnn_concat_155,
        ttnn_multiply_94,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_94, False)
    ttnn.deallocate(ttnn_concat_155, False)
    ttnn_layer_norm_30 = ttnn.layer_norm(
        ttnn_add_149,
        epsilon=9.9999999747524271e-07,
        weight=None,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
    )
    ttnn_slice_250 = ttnn.slice(
        ttnn_typecast_172,
        [0, 15360],
        [2, 18432],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_542 = ttnn.reshape(
        ttnn_slice_250,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_250, False)
    ttnn_slice_251 = ttnn.slice(
        ttnn_typecast_172,
        [0, 12288],
        [2, 15360],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_543 = ttnn.reshape(
        ttnn_slice_251,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_251, False)
    ttnn_add_150 = ttnn.add(
        ttnn_reshape_543,
        var_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_543, False)
    ttnn_multiply_95 = ttnn.multiply(
        ttnn_layer_norm_30,
        ttnn_add_150,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_150, False)
    ttnn.deallocate(ttnn_layer_norm_30, False)
    ttnn_slice_252 = ttnn.slice(
        ttnn_typecast_172,
        [0, 9216],
        [2, 12288],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_172, False)
    ttnn_reshape_544 = ttnn.reshape(
        ttnn_slice_252,
        [2, 1, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_252, False)
    ttnn_add_151 = ttnn.add(
        ttnn_multiply_95,
        ttnn_reshape_544,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_544, False)
    ttnn.deallocate(ttnn_multiply_95, False)
    ttnn_reshape_545 = ttnn.reshape(
        ttnn_add_151,
        [90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_151, False)
    ttnn_linear_34 = ttnn.linear(
        ttnn_reshape_545,
        self.weights["transformer.transformer_blocks.7.ff_context.net.0.proj.weight"],
        bias=self.weights["transformer.transformer_blocks.7.ff_context.net.0.proj.bias"],
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
    ttnn.deallocate(ttnn_reshape_545, False)
    ttnn_matmul_38 = ttnn.matmul(
        ttnn_linear_34,
        self.weights["transformer.transformer_blocks.7.ff_context.net.2.weight"],
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
    ttnn.deallocate(ttnn_linear_34, False)
    ttnn_reshape_546 = ttnn.reshape(
        ttnn_matmul_38,
        [1, 1, 90, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_38, False)
    ttnn_reduce_scatter_45 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_546,
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
    ttnn.deallocate(ttnn_reshape_546, False)
    ttnn_reshape_547 = ttnn.reshape(
        ttnn_reduce_scatter_45,
        [90, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_45, False)
    ttnn_all_gather_45 = ttnn.all_gather(
        input_tensor=ttnn_reshape_547,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_547, False)
    ttnn_add_152 = ttnn.add(
        ttnn_all_gather_45,
        self.weights["transformer.transformer_blocks.7.ff_context.net.2.bias.reshaped"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_45, False)
    ttnn_reshape_548 = ttnn.reshape(
        ttnn_add_152,
        [2, 45, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_152, False)
    ttnn_multiply_96 = ttnn.multiply(
        ttnn_reshape_542,
        ttnn_reshape_548,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_548, False)
    ttnn.deallocate(ttnn_reshape_542, False)
    ttnn_add_153 = ttnn.add(
        ttnn_add_149,
        ttnn_multiply_96,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_96, False)
    ttnn.deallocate(ttnn_add_149, False)
    ttnn_slice_253 = ttnn.slice(
        ttnn_add_153,
        [0, 0, 0],
        [2, 45, 1536],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_153, False)
    ttnn_to_layout_598 = ttnn.to_layout(
        text_encoder_layer_8,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_549 = ttnn.reshape(
        ttnn_to_layout_598,
        [90, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_598, False)
    return ttnn_add_142, ttnn_reshape_549, ttnn_slice_253, ttnn_transformer_concatenate_heads_7, ttnn_typecast_173


_TB_FORWARDS = {
    0: _tb_forward_0,
    1: _tb_forward_1,
    2: _tb_forward_2,
    3: _tb_forward_3,
    4: _tb_forward_4,
    5: _tb_forward_5,
    6: _tb_forward_6,
    7: _tb_forward_7,
}


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


