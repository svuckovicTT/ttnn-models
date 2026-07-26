import ttnn
import utils
import torch
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

    def forward(self, activations):
        device = self.device
        weights = self.weights
        var_0 = weights["model.layers.0.input_layernorm.parametrizations.weight.original"]
        primals_39 = activations[0]
        var_1 = weights["model.layers.1.input_layernorm.parametrizations.weight.original"]
        var_2 = weights[
            "model.layers.0.post_attention_layernorm.parametrizations.weight.original"
        ]
        var_3 = weights["model.norm.parametrizations.weight.original"]
        var_4 = weights[
            "model.layers.1.post_attention_layernorm.parametrizations.weight.original"
        ]
        var_5 = weights["moe_scatter_zeros"]
        var_6 = weights["causal_mask"]
        var_8 = weights["rotary_emb.cos"]
        var_9 = weights["rotary_emb.sin"]
        var_10 = weights["ones_scalar"]
        var_11 = weights["expert_index_offsets"]
        var_12 = weights["attn_scale"]
        var_13 = weights["rms_norm_epsilon"]
        var_14 = weights["sigmoid_scale"]
        ttnn_typecast_57 = ttnn.typecast(
            primals_39,
            ttnn.DataType.UINT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_0 = ttnn.reshape(
            ttnn_typecast_57,
            [17],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_57, False)
        ttnn_embedding_0 = ttnn.embedding(
            ttnn_reshape_0,
            weights["model.embed_tokens.parametrizations.weight.original"],
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_0, False)
        ttnn_reshape_1 = ttnn.reshape(
            ttnn_embedding_0,
            [1, 17, 2880],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_58 = ttnn.typecast(
            ttnn_reshape_1,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pow_0 = ttnn.pow(
            ttnn_typecast_58,
            2.0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_58, False)
        ttnn_mean_0 = ttnn.mean(
            ttnn_pow_0,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_pow_0, False)
        ttnn_add_0 = ttnn.add(
            ttnn_mean_0,
            var_13,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_mean_0, False)
        ttnn_rsqrt_0 = ttnn.rsqrt(
            ttnn_add_0,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_0, False)
        ttnn_rms_norm_0 = ttnn.rms_norm(
            ttnn_reshape_1,
            epsilon=9.9999997473787516e-06,
            weight=var_0,
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
        ttnn_reshape_2 = ttnn.reshape(
            ttnn_rms_norm_0,
            [17, 2880],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_linear_0 = ttnn.linear(
            ttnn_reshape_2,
            weights["model.layers.0.self_attn.qkv_proj.weight"],
            bias=weights["model.layers.0.self_attn.qkv_proj.bias"],
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
        ttnn.deallocate(ttnn_reshape_2, False)
        ttnn_reshape_3 = ttnn.reshape(
            ttnn_linear_0,
            [1, 17, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_linear_0, False)
        v_15, v_16, v_17 = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn_reshape_3,
            None,
            num_heads=16,
            num_kv_heads=2,
            transpose_key=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_3, False)
        ttnn_experimental_rotary_embedding_0 = ttnn.experimental.rotary_embedding(
            v_17,
            var_8,
            var_9,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=None,
        )
        ttnn.deallocate(v_17, False)
        ttnn_slice_12 = ttnn.slice(
            ttnn_experimental_rotary_embedding_0,
            [0, 0, 0, 0],
            [1, 2, 17, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_experimental_rotary_embedding_0, False)
        ttnn_experimental_rotary_embedding_1 = ttnn.experimental.rotary_embedding(
            v_15,
            var_8,
            var_9,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=None,
        )
        ttnn.deallocate(v_15, False)
        ttnn_slice_13 = ttnn.slice(
            ttnn_experimental_rotary_embedding_1,
            [0, 0, 0, 0],
            [1, 16, 17, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_experimental_rotary_embedding_1, False)
        ttnn_reshape_4 = ttnn.reshape(
            ttnn_slice_12,
            [1, 2, 1, 17, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_repeat_0 = ttnn.repeat(
            ttnn_reshape_4,
            ttnn.Shape([1, 1, 8, 1, 1]),
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_4, False)
        ttnn_reshape_5 = ttnn.reshape(
            ttnn_repeat_0,
            [1, 16, 17, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_0 = ttnn.permute(
            ttnn_reshape_5,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_5, False)
        ttnn_matmul_0 = ttnn.matmul(
            ttnn_slice_13,
            ttnn_permute_0,
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
        ttnn.deallocate(ttnn_permute_0, False)
        ttnn_multiply_0 = ttnn.multiply(
            ttnn_matmul_0,
            var_12,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_0, False)
        ttnn_add_1 = ttnn.add(
            ttnn_multiply_0,
            var_6,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_0, False)
        ttnn_concat_4 = ttnn.concat(
            [ttnn_add_1, weights["model.layers.0.self_attn.sinks_processed"]],
            3,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_1, False)
        ttnn_softmax_0 = ttnn.softmax(
            ttnn_concat_4,
            3,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=None,
            numeric_stable=True,
        )
        ttnn_slice_14 = ttnn.slice(
            ttnn_softmax_0,
            [0, 0, 0, 0],
            [1, 16, 17, 17],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_6 = ttnn.reshape(
            v_16,
            [1, 2, 1, 17, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_repeat_1 = ttnn.repeat(
            ttnn_reshape_6,
            ttnn.Shape([1, 1, 8, 1, 1]),
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_6, False)
        ttnn_reshape_7 = ttnn.reshape(
            ttnn_repeat_1,
            [1, 16, 17, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_1 = ttnn.matmul(
            ttnn_slice_14,
            ttnn_reshape_7,
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
        ttnn.deallocate(ttnn_reshape_7, False)
        ttnn_permute_1 = ttnn.permute(
            ttnn_matmul_1,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_transformer_concatenate_heads_0 = ttnn.transformer.concatenate_heads(
            ttnn_matmul_1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_1, False)
        ttnn_reshape_8 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_0,
            [17, 1024],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_transformer_concatenate_heads_0, False)
        ttnn_matmul_2 = ttnn.matmul(
            ttnn_reshape_8,
            weights["model.layers.0.self_attn.o_proj.weight_transposed"],
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
        ttnn.deallocate(ttnn_reshape_8, False)
        ttnn_reshape_9 = ttnn.reshape(
            ttnn_matmul_2,
            [1, 1, 17, 2880],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_2, False)
        ttnn_all_reduce_0 = ttnn.all_reduce(
            input_tensor=ttnn_reshape_9,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_9, False)
        ttnn_reshape_10 = ttnn.reshape(
            ttnn_all_reduce_0,
            [17, 2880],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_all_reduce_0, False)
        ttnn_add_2 = ttnn.add(
            ttnn_reshape_10,
            weights["model.layers.0.self_attn.o_proj.bias_reshaped"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_10, False)
        ttnn_add_3 = ttnn.add(
            ttnn_embedding_0,
            ttnn_add_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_2, False)
        ttnn.deallocate(ttnn_embedding_0, False)
        ttnn_reshape_11 = ttnn.reshape(
            ttnn_add_3,
            [1, 17, 2880],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_3, False)
        ttnn_typecast_59 = ttnn.typecast(
            ttnn_reshape_11,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pow_1 = ttnn.pow(
            ttnn_typecast_59,
            2.0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_59, False)
        ttnn_mean_1 = ttnn.mean(
            ttnn_pow_1,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_pow_1, False)
        ttnn_add_4 = ttnn.add(
            ttnn_mean_1,
            var_13,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_mean_1, False)
        ttnn_rsqrt_1 = ttnn.rsqrt(
            ttnn_add_4,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_4, False)
        ttnn_rms_norm_1 = ttnn.rms_norm(
            ttnn_reshape_11,
            epsilon=9.9999997473787516e-06,
            weight=var_2,
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
        ttnn_reshape_12 = ttnn.reshape(
            ttnn_rms_norm_1,
            [17, 2880],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_concat_5 = ttnn.concat(
            [
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
                ttnn_reshape_12,
            ],
            0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_13 = ttnn.reshape(
            ttnn_concat_5,
            [32, 17, 2880],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_3 = ttnn.matmul(
            ttnn_reshape_13,
            weights["model.layers.0.mlp.experts.gate_up_proj_bf8"],
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
        ttnn.deallocate(ttnn_reshape_13, False)
        ttnn_add_5 = ttnn.add(
            ttnn_matmul_3,
            weights["model.layers.0.mlp.experts.gate_up_proj_bias_reshaped"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_15 = ttnn.slice(
            ttnn_add_5,
            [0, 0, 1],
            [32, 17, 5760],
            [1, 1, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_clamp_0 = ttnn.clamp(
            ttnn_slice_15,
            -7.0,
            7.0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_15, False)
        ttnn_add_6 = ttnn.add(
            ttnn_clamp_0,
            var_10,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_clamp_0, False)
        ttnn_slice_16 = ttnn.slice(
            ttnn_add_5,
            [0, 0, 0],
            [32, 17, 5760],
            [1, 1, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_5, False)
        ttnn_clamp_1 = ttnn.clamp(
            ttnn_slice_16,
            float("-inf"),
            7.0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_16, False)
        ttnn_multiply_1 = ttnn.multiply(
            ttnn_clamp_1,
            var_14,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_sigmoid_0 = ttnn.sigmoid(
            ttnn_multiply_1,
            vector_mode=4,
            mode=ttnn.SigmoidMode.Accurate,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_1, False)
        ttnn_multiply_2 = ttnn.multiply(
            ttnn_clamp_1,
            ttnn_sigmoid_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_clamp_1, False)
        ttnn_multiply_3 = ttnn.multiply(
            ttnn_add_6,
            ttnn_multiply_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_2, False)
        ttnn.deallocate(ttnn_add_6, False)
        ttnn_matmul_4 = ttnn.matmul(
            ttnn_multiply_3,
            weights["model.layers.0.mlp.experts.down_proj_bf8"],
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
        ttnn_add_7 = ttnn.add(
            ttnn_matmul_4,
            weights["model.layers.0.mlp.experts.down_proj_bias_reshaped"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_60 = ttnn.typecast(
            ttnn_reshape_12,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_12, False)
        ttnn_linear_1 = ttnn.linear(
            ttnn_typecast_60,
            weights["model.layers.0.mlp.router.weight_transposed"],
            bias=weights["model.layers.0.mlp.router.bias_processed"],
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
        ttnn.deallocate(ttnn_typecast_60, False)
        ttnn_typecast_61 = ttnn.typecast(
            ttnn_linear_1,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_linear_1, False)
        v_18, v_19 = ttnn.topk(
            ttnn_typecast_61,
            4,
            1,
            True,
            True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(v_18, False)
        ttnn_typecast_62 = ttnn.typecast(
            v_19,
            ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(v_19, False)
        v_20, v_21 = ttnn.topk(
            ttnn_typecast_61,
            4,
            -1,
            True,
            True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_61, False)
        ttnn_typecast_63 = ttnn.typecast(
            v_21,
            ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(v_21, False)
        ttnn_softmax_1 = ttnn.softmax(
            v_20,
            1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=None,
            numeric_stable=True,
        )
        ttnn.deallocate(v_20, False)
        ttnn_reshape_14 = ttnn.reshape(
            ttnn_typecast_63,
            [17, 4, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_add_8 = ttnn.add(
            var_11,
            ttnn_reshape_14,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_14, False)
        ttnn_reshape_15 = ttnn.reshape(
            ttnn_add_8,
            [68],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_8, False)
        ttnn_reshape_16 = ttnn.reshape(
            ttnn_softmax_1,
            [68],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_to_layout_39 = ttnn.to_layout(
            ttnn_reshape_15,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_15, False)
        ttnn_to_layout_40 = ttnn.to_layout(
            ttnn_reshape_16,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_16, False)
        ttnn_scatter_0 = ttnn.scatter(
            input=var_5,
            dim=0,
            index=ttnn_to_layout_39,
            src=ttnn_to_layout_40,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            reduce=None,
        )
        ttnn.deallocate(ttnn_to_layout_40, False)
        ttnn.deallocate(ttnn_to_layout_39, False)
        ttnn_to_layout_41 = ttnn.to_layout(
            ttnn_scatter_0,
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_scatter_0, False)
        ttnn_reshape_17 = ttnn.reshape(
            ttnn_to_layout_41,
            [17, 32],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_41, False)
        ttnn_permute_2 = ttnn.permute(
            ttnn_reshape_17,
            [1, 0],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_18 = ttnn.reshape(
            ttnn_permute_2,
            [32, 17, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_2, False)
        ttnn_multiply_4 = ttnn.multiply(
            ttnn_add_7,
            ttnn_reshape_18,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_18, False)
        ttnn.deallocate(ttnn_add_7, False)
        ttnn_sum_0 = ttnn.sum(
            ttnn_multiply_4,
            [0],
            True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_4, False)
        ttnn_add_9 = ttnn.add(
            ttnn_reshape_11,
            ttnn_sum_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_sum_0, False)
        ttnn_typecast_64 = ttnn.typecast(
            ttnn_add_9,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pow_2 = ttnn.pow(
            ttnn_typecast_64,
            2.0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_64, False)
        ttnn_mean_2 = ttnn.mean(
            ttnn_pow_2,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_pow_2, False)
        ttnn_add_10 = ttnn.add(
            ttnn_mean_2,
            var_13,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_mean_2, False)
        ttnn_rsqrt_2 = ttnn.rsqrt(
            ttnn_add_10,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_10, False)
        ttnn_rms_norm_2 = ttnn.rms_norm(
            ttnn_add_9,
            epsilon=9.9999997473787516e-06,
            weight=var_1,
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
        ttnn_reshape_19 = ttnn.reshape(
            ttnn_rms_norm_2,
            [17, 2880],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_linear_2 = ttnn.linear(
            ttnn_reshape_19,
            weights["model.layers.1.self_attn.qkv_proj.weight"],
            bias=weights["model.layers.1.self_attn.qkv_proj.bias"],
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
        ttnn.deallocate(ttnn_reshape_19, False)
        ttnn_reshape_20 = ttnn.reshape(
            ttnn_linear_2,
            [1, 17, 1280],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_linear_2, False)
        v_22, v_23, v_24 = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn_reshape_20,
            None,
            num_heads=16,
            num_kv_heads=2,
            transpose_key=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_20, False)
        ttnn_experimental_rotary_embedding_2 = ttnn.experimental.rotary_embedding(
            v_24,
            var_8,
            var_9,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=None,
        )
        ttnn.deallocate(v_24, False)
        ttnn_slice_17 = ttnn.slice(
            ttnn_experimental_rotary_embedding_2,
            [0, 0, 0, 0],
            [1, 2, 17, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_experimental_rotary_embedding_2, False)
        ttnn_experimental_rotary_embedding_3 = ttnn.experimental.rotary_embedding(
            v_22,
            var_8,
            var_9,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=None,
        )
        ttnn.deallocate(v_22, False)
        ttnn_slice_18 = ttnn.slice(
            ttnn_experimental_rotary_embedding_3,
            [0, 0, 0, 0],
            [1, 16, 17, 64],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_experimental_rotary_embedding_3, False)
        ttnn_reshape_21 = ttnn.reshape(
            ttnn_slice_17,
            [1, 2, 1, 17, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_repeat_2 = ttnn.repeat(
            ttnn_reshape_21,
            ttnn.Shape([1, 1, 8, 1, 1]),
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_21, False)
        ttnn_reshape_22 = ttnn.reshape(
            ttnn_repeat_2,
            [1, 16, 17, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_permute_3 = ttnn.permute(
            ttnn_reshape_22,
            [0, 1, 3, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_22, False)
        ttnn_matmul_5 = ttnn.matmul(
            ttnn_slice_18,
            ttnn_permute_3,
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
        ttnn.deallocate(ttnn_permute_3, False)
        ttnn_multiply_5 = ttnn.multiply(
            ttnn_matmul_5,
            var_12,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_5, False)
        ttnn_add_11 = ttnn.add(
            ttnn_multiply_5,
            var_6,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_5, False)
        ttnn_concat_6 = ttnn.concat(
            [ttnn_add_11, weights["model.layers.1.self_attn.sinks_processed"]],
            3,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_11, False)
        ttnn_softmax_2 = ttnn.softmax(
            ttnn_concat_6,
            3,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=None,
            numeric_stable=True,
        )
        ttnn_slice_19 = ttnn.slice(
            ttnn_softmax_2,
            [0, 0, 0, 0],
            [1, 16, 17, 17],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_23 = ttnn.reshape(
            v_23,
            [1, 2, 1, 17, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_repeat_3 = ttnn.repeat(
            ttnn_reshape_23,
            ttnn.Shape([1, 1, 8, 1, 1]),
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_23, False)
        ttnn_reshape_24 = ttnn.reshape(
            ttnn_repeat_3,
            [1, 16, 17, 64],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_6 = ttnn.matmul(
            ttnn_slice_19,
            ttnn_reshape_24,
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
        ttnn.deallocate(ttnn_reshape_24, False)
        ttnn_permute_4 = ttnn.permute(
            ttnn_matmul_6,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_transformer_concatenate_heads_1 = ttnn.transformer.concatenate_heads(
            ttnn_matmul_6,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_6, False)
        ttnn_reshape_25 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_1,
            [17, 1024],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_transformer_concatenate_heads_1, False)
        ttnn_matmul_7 = ttnn.matmul(
            ttnn_reshape_25,
            weights["model.layers.1.self_attn.o_proj.weight_transposed"],
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
        ttnn.deallocate(ttnn_reshape_25, False)
        ttnn_reshape_26 = ttnn.reshape(
            ttnn_matmul_7,
            [1, 1, 17, 2880],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_7, False)
        ttnn_all_reduce_1 = ttnn.all_reduce(
            input_tensor=ttnn_reshape_26,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_26, False)
        ttnn_reshape_27 = ttnn.reshape(
            ttnn_all_reduce_1,
            [17, 2880],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_all_reduce_1, False)
        ttnn_add_12 = ttnn.add(
            ttnn_reshape_27,
            weights["model.layers.1.self_attn.o_proj.bias_reshaped"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_27, False)
        ttnn_reshape_28 = ttnn.reshape(
            ttnn_add_12,
            [1, 17, 2880],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_12, False)
        ttnn_add_13 = ttnn.add(
            ttnn_add_9,
            ttnn_reshape_28,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_28, False)
        ttnn_typecast_65 = ttnn.typecast(
            ttnn_add_13,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pow_3 = ttnn.pow(
            ttnn_typecast_65,
            2.0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_65, False)
        ttnn_mean_3 = ttnn.mean(
            ttnn_pow_3,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_pow_3, False)
        ttnn_add_14 = ttnn.add(
            ttnn_mean_3,
            var_13,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_mean_3, False)
        ttnn_rsqrt_3 = ttnn.rsqrt(
            ttnn_add_14,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_14, False)
        ttnn_rms_norm_3 = ttnn.rms_norm(
            ttnn_add_13,
            epsilon=9.9999997473787516e-06,
            weight=var_4,
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
        ttnn_reshape_29 = ttnn.reshape(
            ttnn_rms_norm_3,
            [17, 2880],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_concat_7 = ttnn.concat(
            [
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
                ttnn_reshape_29,
            ],
            0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_30 = ttnn.reshape(
            ttnn_concat_7,
            [32, 17, 2880],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_8 = ttnn.matmul(
            ttnn_reshape_30,
            weights["model.layers.1.mlp.experts.gate_up_proj_bf8"],
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
        ttnn.deallocate(ttnn_reshape_30, False)
        ttnn_add_15 = ttnn.add(
            ttnn_matmul_8,
            weights["model.layers.1.mlp.experts.gate_up_proj_bias_reshaped"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_20 = ttnn.slice(
            ttnn_add_15,
            [0, 0, 1],
            [32, 17, 5760],
            [1, 1, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_clamp_2 = ttnn.clamp(
            ttnn_slice_20,
            -7.0,
            7.0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_20, False)
        ttnn_add_16 = ttnn.add(
            ttnn_clamp_2,
            var_10,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_clamp_2, False)
        ttnn_slice_21 = ttnn.slice(
            ttnn_add_15,
            [0, 0, 0],
            [32, 17, 5760],
            [1, 1, 2],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_15, False)
        ttnn_clamp_3 = ttnn.clamp(
            ttnn_slice_21,
            float("-inf"),
            7.0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_21, False)
        ttnn_multiply_6 = ttnn.multiply(
            ttnn_clamp_3,
            var_14,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_sigmoid_1 = ttnn.sigmoid(
            ttnn_multiply_6,
            vector_mode=4,
            mode=ttnn.SigmoidMode.Accurate,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_6, False)
        ttnn_multiply_7 = ttnn.multiply(
            ttnn_clamp_3,
            ttnn_sigmoid_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_clamp_3, False)
        ttnn_multiply_8 = ttnn.multiply(
            ttnn_add_16,
            ttnn_multiply_7,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_7, False)
        ttnn.deallocate(ttnn_add_16, False)
        ttnn_matmul_9 = ttnn.matmul(
            ttnn_multiply_8,
            weights["model.layers.1.mlp.experts.down_proj_bf8"],
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
        ttnn_add_17 = ttnn.add(
            ttnn_matmul_9,
            weights["model.layers.1.mlp.experts.down_proj_bias_reshaped"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_66 = ttnn.typecast(
            ttnn_reshape_29,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_29, False)
        ttnn_linear_3 = ttnn.linear(
            ttnn_typecast_66,
            weights["model.layers.1.mlp.router.weight_transposed"],
            bias=weights["model.layers.1.mlp.router.bias_processed"],
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
        ttnn.deallocate(ttnn_typecast_66, False)
        ttnn_typecast_67 = ttnn.typecast(
            ttnn_linear_3,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_linear_3, False)
        v_25, v_26 = ttnn.topk(
            ttnn_typecast_67,
            4,
            1,
            True,
            True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(v_25, False)
        ttnn_typecast_68 = ttnn.typecast(
            v_26,
            ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(v_26, False)
        v_27, v_28 = ttnn.topk(
            ttnn_typecast_67,
            4,
            -1,
            True,
            True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_67, False)
        ttnn_typecast_69 = ttnn.typecast(
            v_28,
            ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(v_28, False)
        ttnn_softmax_3 = ttnn.softmax(
            v_27,
            1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=None,
            numeric_stable=True,
        )
        ttnn.deallocate(v_27, False)
        ttnn_reshape_31 = ttnn.reshape(
            ttnn_typecast_69,
            [17, 4, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_add_18 = ttnn.add(
            var_11,
            ttnn_reshape_31,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_31, False)
        ttnn_reshape_32 = ttnn.reshape(
            ttnn_add_18,
            [68],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_18, False)
        ttnn_reshape_33 = ttnn.reshape(
            ttnn_softmax_3,
            [68],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_to_layout_42 = ttnn.to_layout(
            ttnn_reshape_32,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_32, False)
        ttnn_to_layout_43 = ttnn.to_layout(
            ttnn_reshape_33,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_33, False)
        ttnn_scatter_1 = ttnn.scatter(
            input=var_5,
            dim=0,
            index=ttnn_to_layout_42,
            src=ttnn_to_layout_43,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            reduce=None,
        )
        ttnn.deallocate(ttnn_to_layout_43, False)
        ttnn.deallocate(ttnn_to_layout_42, False)
        ttnn_to_layout_44 = ttnn.to_layout(
            ttnn_scatter_1,
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_scatter_1, False)
        ttnn_reshape_34 = ttnn.reshape(
            ttnn_to_layout_44,
            [17, 32],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_44, False)
        ttnn_permute_5 = ttnn.permute(
            ttnn_reshape_34,
            [1, 0],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_35 = ttnn.reshape(
            ttnn_permute_5,
            [32, 17, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_5, False)
        ttnn_multiply_9 = ttnn.multiply(
            ttnn_add_17,
            ttnn_reshape_35,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_35, False)
        ttnn.deallocate(ttnn_add_17, False)
        ttnn_sum_1 = ttnn.sum(
            ttnn_multiply_9,
            [0],
            True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_9, False)
        ttnn_add_19 = ttnn.add(
            ttnn_add_13,
            ttnn_sum_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_sum_1, False)
        ttnn_typecast_70 = ttnn.typecast(
            ttnn_add_19,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_pow_4 = ttnn.pow(
            ttnn_typecast_70,
            2.0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_70, False)
        ttnn_mean_4 = ttnn.mean(
            ttnn_pow_4,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_pow_4, False)
        ttnn_add_20 = ttnn.add(
            ttnn_mean_4,
            var_13,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_mean_4, False)
        ttnn_rsqrt_4 = ttnn.rsqrt(
            ttnn_add_20,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_20, False)
        ttnn_rms_norm_4 = ttnn.rms_norm(
            ttnn_add_19,
            epsilon=9.9999997473787516e-06,
            weight=var_3,
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
        ttnn_reshape_36 = ttnn.reshape(
            ttnn_rms_norm_4,
            [17, 2880],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_matmul_10 = ttnn.matmul(
            ttnn_reshape_36,
            weights["lm_head.weight_transposed"],
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
        ttnn.deallocate(ttnn_reshape_36, False)
        ttnn_reshape_37 = ttnn.reshape(
            ttnn_matmul_10,
            [1, 17, 201088],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_10, False)
        ttnn_to_layout_45 = ttnn.to_layout(
            ttnn_concat_4,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_concat_4, False)
        ttnn_argmax_0 = ttnn.argmax(
            ttnn_to_layout_45,
            3,
            True,
            sub_core_grids=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_45, False)
        ttnn_to_layout_46 = ttnn.to_layout(
            ttnn_argmax_0,
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_argmax_0, False)
        ttnn_typecast_71 = ttnn.typecast(
            ttnn_to_layout_46,
            ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_46, False)
        ttnn_to_layout_47 = ttnn.to_layout(
            ttnn_concat_6,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_concat_6, False)
        ttnn_argmax_1 = ttnn.argmax(
            ttnn_to_layout_47,
            3,
            True,
            sub_core_grids=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_47, False)
        ttnn_to_layout_48 = ttnn.to_layout(
            ttnn_argmax_1,
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_argmax_1, False)
        ttnn_typecast_72 = ttnn.typecast(
            ttnn_to_layout_48,
            ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_48, False)
        ttnn_to_layout_49 = ttnn.to_layout(
            primals_39,
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(primals_39, False)
        return [
            v_16,
            ttnn_slice_12,
            v_23,
            ttnn_slice_17,
            ttnn_reshape_37,
            weights["model.layers.0.mlp.experts.gate_up_proj"],
            weights["model.layers.0.mlp.experts.gate_up_proj_bias"],
            weights["model.layers.0.mlp.experts.down_proj"],
            weights["model.layers.0.mlp.experts.down_proj_bias"],
            weights["model.layers.1.mlp.experts.gate_up_proj"],
            weights["model.layers.1.mlp.experts.gate_up_proj_bias"],
            weights["model.layers.1.mlp.experts.down_proj"],
            weights["model.layers.1.mlp.experts.down_proj_bias"],
            ttnn_to_layout_49,
            weights["rotary_emb.freqs"],
            ttnn_reshape_1,
            var_0,
            ttnn_rsqrt_0,
            ttnn_rms_norm_0,
            weights["model.layers.0.self_attn.q_proj.parametrizations.weight.original"],
            weights["model.layers.0.self_attn.k_proj.parametrizations.weight.original"],
            weights["model.layers.0.self_attn.v_proj.parametrizations.weight.original"],
            ttnn_slice_13,
            ttnn_repeat_0,
            ttnn_repeat_1,
            ttnn_typecast_71,
            ttnn_slice_14,
            ttnn_permute_1,
            weights["model.layers.0.self_attn.o_proj.parametrizations.weight.original"],
            ttnn_reshape_11,
            var_2,
            ttnn_rsqrt_1,
            ttnn_rms_norm_1,
            weights["model.layers.0.mlp.router.parametrizations.weight.original"],
            ttnn_typecast_62,
            ttnn_typecast_63,
            weights["moe_scatter_zeros_2d"],
            ttnn_reshape_17,
            ttnn_concat_5,
            ttnn_matmul_3,
            ttnn_sigmoid_0,
            ttnn_multiply_3,
            ttnn_matmul_4,
            ttnn_add_9,
            var_1,
            ttnn_rsqrt_2,
            ttnn_rms_norm_2,
            weights["model.layers.1.self_attn.q_proj.parametrizations.weight.original"],
            weights["model.layers.1.self_attn.k_proj.parametrizations.weight.original"],
            weights["model.layers.1.self_attn.v_proj.parametrizations.weight.original"],
            ttnn_slice_18,
            ttnn_repeat_2,
            ttnn_repeat_3,
            ttnn_typecast_72,
            ttnn_slice_19,
            ttnn_permute_4,
            weights["model.layers.1.self_attn.o_proj.parametrizations.weight.original"],
            ttnn_add_13,
            var_4,
            ttnn_rsqrt_3,
            ttnn_rms_norm_3,
            weights["model.layers.1.mlp.router.parametrizations.weight.original"],
            ttnn_typecast_68,
            ttnn_typecast_69,
            ttnn_reshape_34,
            ttnn_concat_7,
            ttnn_matmul_8,
            ttnn_sigmoid_1,
            ttnn_multiply_8,
            ttnn_matmul_9,
            ttnn_add_19,
            var_3,
            ttnn_rsqrt_4,
            ttnn_rms_norm_4,
            weights["lm_head.parametrizations.weight.original"],
            ttnn_softmax_3,
            ttnn_softmax_2,
            ttnn_softmax_1,
            ttnn_softmax_0,
        ]
