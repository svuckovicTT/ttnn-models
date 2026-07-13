"""TTNN forward model for the SmolLM3-3B codegen.

Holds the graph's forward model ``ModelTTNN`` -- a ``LightweightModule`` whose
``forward`` runs the graph's ttnn ops. Its weights (including the constant
tensors materialized by ``consteval.run_consteval``) are prepared in
``__init__``. ``ModelTTNN`` is instantiated and driven from ``main.py``.
"""
import ttnn
import params
import consteval


class LightweightModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ModelTTNN(LightweightModule):
    NO_ROPE_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]

    def __init__(self, device):
        self.device = device
        self.weights = params.load_weights_for__main_from_state_dict(device)
        self.weights = consteval.run_consteval(self.weights, device)
        self.layers = [
            DecoderLayer(self.weights, layer, use_rope=layer not in self.NO_ROPE_LAYERS)
            for layer in range(36)
        ]
        self.norm = RMSNorm(self.weights, "model.norm.weight")

    def forward(self, activations):
        args_0 = activations[0]
        args_1 = activations[1]
        activation_0 = activations[2]
        var_0 = self.weights["consteval.zeros_f32"]
        var_1 = self.weights["consteval.attn_scale_2d"]
        var_2 = self.weights["consteval.attn_scale_4d"]
        var_3 = self.weights["consteval.neg_inf_f32"]
        var_4 = [self.weights["model.rotary_emb.cos"], self.weights["model.rotary_emb.sin"]]
        var_5 = var_4[0]
        var_6 = var_4[1]
        var_7 = [self.weights["consteval.causal_mask_bool"], self.weights["consteval.causal_mask_index"]]
        ttnn_typecast_3 = ttnn.typecast(
            args_0,
            ttnn.DataType.UINT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(args_0, False)
        ttnn_reshape_1 = ttnn.reshape(
            ttnn_typecast_3,
            [4096],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_3, False)
        ttnn_embedding_0 = ttnn.embedding(
            ttnn_reshape_1,
            self.weights["model.embed_tokens.weight"],
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_1, False)
        ttnn_to_layout_109 = ttnn.to_layout(
            activation_0,
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_3 = ttnn.reshape(
            ttnn_to_layout_109,
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_109, False)
        ttnn_logical_and_0 = ttnn.logical_and(
            ttnn_reshape_3,
            self.weights["consteval.position_mask"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_3, False)
        ttnn_ne_0 = ttnn.ne(
            args_1,
            self.weights["consteval.zeros_i32"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_reshape_4 = ttnn.reshape(
            ttnn_ne_0,
            [4096, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_ne_0, False)
        ttnn_gather_0 = ttnn.gather(
            ttnn_reshape_4,
            0,
            var_7[1],
            sparse_grad=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_4, False)
        ttnn_where_0 = ttnn.where(
            var_7[0],
            self.weights["consteval.nan_col"],
            ttnn_gather_0,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_gather_0, False)
        ttnn_reshape_5 = ttnn.reshape(
            ttnn_where_0,
            [1, 1, 1, 4096],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_where_0, False)
        ttnn_logical_and_1 = ttnn.logical_and(
            ttnn_logical_and_0,
            ttnn_reshape_5,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_5, False)
        ttnn.deallocate(ttnn_logical_and_0, False)
        ttnn_where_1 = ttnn.where(
            ttnn_logical_and_1,
            self.weights["consteval.zeros_bf16"],
            self.weights["consteval.neg_inf_bf16"],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_logical_and_1, False)
        ttnn_typecast_6 = ttnn.typecast(
            ttnn_where_1,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_where_1, False)
        hidden = ttnn_embedding_0
        for layer in self.layers:
            hidden = layer(hidden, ttnn_typecast_6, var_5, var_6, var_0, var_1, var_2, var_3)
        ttnn_reshape_239 = ttnn.reshape(
            hidden,
            [1, 4096, 2048],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(hidden, False)
        output = self.norm(ttnn_reshape_239)
        ttnn.deallocate(ttnn_reshape_239, False)
        return [output]


class RMSNorm(LightweightModule):
    def __init__(self, weights, weight_key):
        self.weights = weights
        self.weight_key = weight_key

    def forward(self, x):
        result = ttnn.rms_norm(
            x,
            epsilon=9.9999999747524271e-07,
            weight=self.weights[self.weight_key],
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
        return result


class MLP(LightweightModule):
    def __init__(self, weights, layer):
        self.weights = weights
        self.layer = layer

    def forward(self, x):
        ttnn_matmul_11 = ttnn.matmul(
            x,
            self.weights[f"model.layers.{self.layer}.mlp.gate_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation="silu",
            compute_kernel_config=None,
        )
        ttnn_matmul_12 = ttnn.matmul(
            x,
            self.weights[f"model.layers.{self.layer}.mlp.up_proj.weight"],
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
        ttnn.deallocate(x, False)
        ttnn_multiply_6 = ttnn.multiply(
            ttnn_matmul_11,
            ttnn_matmul_12,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_12, False)
        ttnn.deallocate(ttnn_matmul_11, False)
        ttnn_matmul_13 = ttnn.matmul(
            ttnn_multiply_6,
            self.weights[f"model.layers.{self.layer}.mlp.down_proj.weight"],
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
        ttnn.deallocate(ttnn_multiply_6, False)
        ttnn_reshape_15 = ttnn.reshape(
            ttnn_matmul_13,
            [1, 1, 4096, 2048],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_13, False)
        ttnn_reduce_scatter_3 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_15,
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
        ttnn.deallocate(ttnn_reshape_15, False)
        ttnn_reshape_16 = ttnn.reshape(
            ttnn_reduce_scatter_3,
            [4096, 512],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reduce_scatter_3, False)
        ttnn_all_gather_3 = ttnn.all_gather(
            input_tensor=ttnn_reshape_16,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_16, False)
        return ttnn_all_gather_3


class Attention(LightweightModule):
    def __init__(self, weights, layer, use_rope):
        self.weights = weights
        self.layer = layer
        self.use_rope = use_rope

    def forward(self, x, mask, cos, sin, s0, s1, s2, s3):
        if self.use_rope:
            ttnn_matmul_7 = ttnn.matmul(
                x,
                self.weights[f"model.layers.{self.layer}.self_attn.qkv_proj.weight"],
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
            ttnn.deallocate(x, False)
            ttnn_reshape_11 = ttnn.reshape(
                ttnn_matmul_7,
                [1, 4096, 768],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_7, False)
            v_11, v_12, v_13 = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn_reshape_11,
                None,
                num_heads=4,
                num_kv_heads=1,
                transpose_key=False,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_11, False)
            ttnn_experimental_rotary_embedding_2 = ttnn.experimental.rotary_embedding(
                v_11,
                cos,
                sin,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(v_11, False)
            ttnn_typecast_10 = ttnn.typecast(
                ttnn_experimental_rotary_embedding_2,
                ttnn.DataType.FLOAT32,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_2, False)
            ttnn_multiply_4 = ttnn.multiply(
                ttnn_typecast_10,
                s2,
                dtype=ttnn.DataType.FLOAT32,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_typecast_10, False)
            ttnn_experimental_rotary_embedding_3 = ttnn.experimental.rotary_embedding(
                v_13,
                cos,
                sin,
                None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(v_13, False)
            ttnn_repeat_interleave_1 = ttnn.repeat_interleave(
                ttnn_experimental_rotary_embedding_3,
                4,
                1,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_experimental_rotary_embedding_3, False)
            ttnn_typecast_11 = ttnn.typecast(
                ttnn_repeat_interleave_1,
                ttnn.DataType.FLOAT32,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_repeat_interleave_1, False)
            ttnn_multiply_5 = ttnn.multiply(
                ttnn_typecast_11,
                s2,
                dtype=ttnn.DataType.FLOAT32,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_typecast_11, False)
            ttnn_matmul_8 = ttnn.matmul(
                ttnn_multiply_4,
                ttnn_multiply_5,
                transpose_a=False,
                transpose_b=True,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
                dtype=ttnn.DataType.FLOAT32,
                program_config=None,
                activation=None,
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_multiply_5, False)
            ttnn.deallocate(ttnn_multiply_4, False)
            ttnn_add_3 = ttnn.add(
                ttnn_matmul_8,
                mask,
                dtype=ttnn.DataType.FLOAT32,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_8, False)
            ttnn_eq_1 = ttnn.eq(
                ttnn_add_3,
                s3,
                dtype=ttnn.DataType.BFLOAT16,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_logical_not_2 = ttnn.logical_not(
                ttnn_eq_1,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_eq_1, False)
            ttnn_sum_1 = ttnn.sum(
                ttnn_logical_not_2,
                [3],
                True,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_logical_not_2, False)
            ttnn_logical_not_3 = ttnn.logical_not(
                ttnn_sum_1,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_sum_1, False)
            ttnn_softmax_1 = ttnn.softmax(
                ttnn_add_3,
                3,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
                compute_kernel_config=None,
                numeric_stable=True,
            )
            ttnn.deallocate(ttnn_add_3, False)
            ttnn_typecast_12 = ttnn.typecast(
                ttnn_logical_not_3,
                ttnn.DataType.FLOAT32,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_logical_not_3, False)
            ttnn_where_3 = ttnn.where(
                ttnn_typecast_12,
                s0,
                ttnn_softmax_1,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_typecast_12, False)
            ttnn.deallocate(ttnn_softmax_1, False)
            ttnn_typecast_13 = ttnn.typecast(
                v_12,
                ttnn.DataType.FLOAT32,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(v_12, False)
            ttnn_matmul_9 = ttnn.matmul(
                ttnn_where_3,
                ttnn_typecast_13,
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
            ttnn.deallocate(ttnn_typecast_13, False)
            ttnn.deallocate(ttnn_where_3, False)
            ttnn_typecast_14 = ttnn.typecast(
                ttnn_matmul_9,
                ttnn.DataType.BFLOAT16,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_9, False)
            ttnn_transformer_concatenate_heads_1 = ttnn.transformer.concatenate_heads(
                ttnn_typecast_14,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_typecast_14, False)
            ttnn_reshape_12 = ttnn.reshape(
                ttnn_transformer_concatenate_heads_1,
                [4096, 512],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_transformer_concatenate_heads_1, False)
            ttnn_matmul_10 = ttnn.matmul(
                ttnn_reshape_12,
                self.weights[f"model.layers.{self.layer}.self_attn.o_proj.weight"],
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
            ttnn.deallocate(ttnn_reshape_12, False)
            ttnn_reshape_13 = ttnn.reshape(
                ttnn_matmul_10,
                [1, 1, 4096, 2048],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_10, False)
            ttnn_reduce_scatter_2 = ttnn.reduce_scatter(
                input_tensor=ttnn_reshape_13,
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
            ttnn.deallocate(ttnn_reshape_13, False)
            ttnn_reshape_14 = ttnn.reshape(
                ttnn_reduce_scatter_2,
                [4096, 512],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reduce_scatter_2, False)
            ttnn_all_gather_2 = ttnn.all_gather(
                input_tensor=ttnn_reshape_14,
                dim=1,
                cluster_axis=1,
                subdevice_id=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
                num_links=None,
                topology=ttnn.Topology.Ring,
            )
            ttnn.deallocate(ttnn_reshape_14, False)
            return ttnn_all_gather_2
        else:
            ttnn_matmul_21 = ttnn.matmul(
                x,
                self.weights[f"model.layers.{self.layer}.self_attn.qkv_proj.weight"],
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
            ttnn.deallocate(x, False)
            ttnn_slice_0 = ttnn.slice(
                ttnn_matmul_21,
                [0, 0],
                [4096, 512],
                [1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_slice_1 = ttnn.slice(
                ttnn_matmul_21,
                [0, 512],
                [4096, 640],
                [1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_reshape_23 = ttnn.reshape(
                ttnn_slice_1,
                [1, 1, 4096, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_slice_1, False)
            ttnn_slice_2 = ttnn.slice(
                ttnn_matmul_21,
                [0, 640],
                [4096, 768],
                [1, 1],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_21, False)
            ttnn_reshape_24 = ttnn.reshape(
                ttnn_slice_2,
                [1, 1, 4096, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_slice_2, False)
            ttnn_permute_108 = ttnn.permute(
                ttnn_reshape_24,
                [0, 1, 3, 2],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
                pad_value=0.0,
            )
            ttnn.deallocate(ttnn_reshape_24, False)
            ttnn_typecast_20 = ttnn.typecast(
                ttnn_slice_0,
                ttnn.DataType.FLOAT32,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_slice_0, False)
            ttnn_multiply_10 = ttnn.multiply(
                ttnn_typecast_20,
                s1,
                dtype=ttnn.DataType.FLOAT32,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_typecast_20, False)
            ttnn_reshape_25 = ttnn.reshape(
                ttnn_multiply_10,
                [1, 4096, 4, 128],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_multiply_10, False)
            ttnn_permute_109 = ttnn.permute(
                ttnn_reshape_25,
                [0, 2, 1, 3],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
                pad_value=0.0,
            )
            ttnn.deallocate(ttnn_reshape_25, False)
            ttnn_typecast_21 = ttnn.typecast(
                ttnn_permute_108,
                ttnn.DataType.FLOAT32,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_permute_108, False)
            ttnn_multiply_11 = ttnn.multiply(
                ttnn_typecast_21,
                s2,
                dtype=ttnn.DataType.FLOAT32,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_typecast_21, False)
            ttnn_matmul_22 = ttnn.matmul(
                ttnn_permute_109,
                ttnn_multiply_11,
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
            ttnn.deallocate(ttnn_multiply_11, False)
            ttnn.deallocate(ttnn_permute_109, False)
            ttnn_add_9 = ttnn.add(
                ttnn_matmul_22,
                mask,
                dtype=ttnn.DataType.FLOAT32,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_22, False)
            ttnn_eq_3 = ttnn.eq(
                ttnn_add_9,
                s3,
                dtype=ttnn.DataType.BFLOAT16,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn_logical_not_6 = ttnn.logical_not(
                ttnn_eq_3,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_eq_3, False)
            ttnn_sum_3 = ttnn.sum(
                ttnn_logical_not_6,
                [3],
                True,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
                compute_kernel_config=None,
            )
            ttnn.deallocate(ttnn_logical_not_6, False)
            ttnn_logical_not_7 = ttnn.logical_not(
                ttnn_sum_3,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_sum_3, False)
            ttnn_softmax_3 = ttnn.softmax(
                ttnn_add_9,
                3,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
                compute_kernel_config=None,
                numeric_stable=True,
            )
            ttnn.deallocate(ttnn_add_9, False)
            ttnn_typecast_22 = ttnn.typecast(
                ttnn_logical_not_7,
                ttnn.DataType.FLOAT32,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_logical_not_7, False)
            ttnn_where_5 = ttnn.where(
                ttnn_typecast_22,
                s0,
                ttnn_softmax_3,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_typecast_22, False)
            ttnn.deallocate(ttnn_softmax_3, False)
            ttnn_typecast_23 = ttnn.typecast(
                ttnn_reshape_23,
                ttnn.DataType.FLOAT32,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reshape_23, False)
            ttnn_matmul_23 = ttnn.matmul(
                ttnn_where_5,
                ttnn_typecast_23,
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
            ttnn.deallocate(ttnn_typecast_23, False)
            ttnn.deallocate(ttnn_where_5, False)
            ttnn_typecast_24 = ttnn.typecast(
                ttnn_matmul_23,
                ttnn.DataType.BFLOAT16,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_23, False)
            ttnn_transformer_concatenate_heads_3 = ttnn.transformer.concatenate_heads(
                ttnn_typecast_24,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_typecast_24, False)
            ttnn_reshape_26 = ttnn.reshape(
                ttnn_transformer_concatenate_heads_3,
                [4096, 512],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_transformer_concatenate_heads_3, False)
            ttnn_matmul_24 = ttnn.matmul(
                ttnn_reshape_26,
                self.weights[f"model.layers.{self.layer}.self_attn.o_proj.weight"],
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
            ttnn.deallocate(ttnn_reshape_26, False)
            ttnn_reshape_27 = ttnn.reshape(
                ttnn_matmul_24,
                [1, 1, 4096, 2048],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_matmul_24, False)
            ttnn_reduce_scatter_6 = ttnn.reduce_scatter(
                input_tensor=ttnn_reshape_27,
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
            ttnn.deallocate(ttnn_reshape_27, False)
            ttnn_reshape_28 = ttnn.reshape(
                ttnn_reduce_scatter_6,
                [4096, 512],
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
            ttnn.deallocate(ttnn_reduce_scatter_6, False)
            ttnn_all_gather_6 = ttnn.all_gather(
                input_tensor=ttnn_reshape_28,
                dim=1,
                cluster_axis=1,
                subdevice_id=None,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
                num_links=None,
                topology=ttnn.Topology.Ring,
            )
            ttnn.deallocate(ttnn_reshape_28, False)
            return ttnn_all_gather_6


class DecoderLayer(LightweightModule):
    def __init__(self, weights, layer, use_rope):
        self.input_ln = RMSNorm(weights, f"model.layers.{layer}.input_layernorm.weight")
        self.self_attn = Attention(weights, layer, use_rope)
        self.post_ln = RMSNorm(weights, f"model.layers.{layer}.post_attention_layernorm.weight")
        self.mlp = MLP(weights, layer)

    def forward(self, hidden, mask, cos, sin, s0, s1, s2, s3):
        normed1 = self.input_ln(hidden)
        attn_out = self.self_attn(normed1, mask, cos, sin, s0, s1, s2, s3)
        h1 = ttnn.add(
            hidden,
            attn_out,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(attn_out, False)
        ttnn.deallocate(hidden, False)
        normed2 = self.post_ln(h1)
        mlp_out = self.mlp(normed2)
        h2 = ttnn.add(
            h1,
            mlp_out,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(mlp_out, False)
        ttnn.deallocate(h1, False)
        return h2
