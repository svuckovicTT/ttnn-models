"""Edge-case transformer blocks that differ from the uniform ones.

The first double block (block 0), the first single block (the double->single
transition), and the last single block (the output block) each do a
structurally different computation from the 43 uniform blocks, so they live
here and are dispatched from the block classes' forward().
"""
import ttnn


def double_block_0(self, hidden_states, txt_ids, img_ids, attention_mask, text_encoder_layer_1, ttnn_concat_109, ttnn_layer_norm_0, ttnn_slice_54, ttnn_slice_55, var_0, var_1):
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

def single_block_0(self, text_encoder_layer_9, ttnn_add_142, ttnn_reshape_203, ttnn_reshape_209, ttnn_reshape_549, ttnn_slice_253, ttnn_slice_39, ttnn_to_layout_590, ttnn_transformer_concatenate_heads_7, ttnn_typecast_173, var_1):
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

def single_block_37(self, ttnn_add_418, ttnn_reshape_1301, ttnn_reshape_203, ttnn_reshape_209, ttnn_slice_2, ttnn_slice_738, ttnn_to_layout_590, var_1):
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
