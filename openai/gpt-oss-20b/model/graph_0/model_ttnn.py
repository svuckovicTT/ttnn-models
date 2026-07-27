import ttnn
import torch
import params
import consteval

DRAM_MEMORY_CONFIG = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)

WORMHOLE_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


class LightweightModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ModelTTNN(LightweightModule):
    def __init__(self, device):
        self.device = device
        self.weights = params.load_weights_for__main(device)
        self.weights = consteval.run_consteval(self.weights, device)

        self.layers = []
        for i in range(2):
            self.layers.append(GptOssDecoderLayer(device, self.weights, i))
        self.final_norm = GptOssRMSNorm(device, self.weights, "model.norm")
        self.lm_head = GptOssLMHead(device, self.weights)

    def forward(self, activations):
        device = self.device
        weights = self.weights
        primals_39 = activations[0]

        ttnn_typecast_57 = ttnn.typecast(
            primals_39,
            ttnn.DataType.UINT32,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn_reshape_0 = ttnn.reshape(ttnn_typecast_57, [17], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ttnn_typecast_57, False)
        ttnn_embedding_0 = ttnn.embedding(
            ttnn_reshape_0,
            weights["model.embed_tokens.weight"],
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_reshape_0, False)
        ttnn_reshape_1 = ttnn.reshape(ttnn_embedding_0, [1, 17, 2880], memory_config=DRAM_MEMORY_CONFIG)

        hidden_states = ttnn_reshape_1
        residual = ttnn_embedding_0

        layer_0_outputs = self.layers[0](hidden_states, residual)
        hidden_states_0 = layer_0_outputs["hidden_states"]
        residual_0 = layer_0_outputs["residual"]

        layer_1_outputs = self.layers[1](hidden_states_0, residual_0)
        hidden_states_1 = layer_1_outputs["hidden_states"]
        residual_1 = layer_1_outputs["residual"]

        norm_out, rsqrt_out = self.final_norm(hidden_states_1)
        logits = self.lm_head(norm_out)

        ttnn_to_layout_45 = ttnn.to_layout(
            layer_0_outputs["attn_concat"], ttnn.Layout.ROW_MAJOR, None,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(layer_0_outputs["attn_concat"], False)
        ttnn_argmax_0 = ttnn.argmax(
            ttnn_to_layout_45, 3, True, sub_core_grids=None,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_to_layout_45, False)
        ttnn_to_layout_46 = ttnn.to_layout(
            ttnn_argmax_0, ttnn.Layout.TILE, None,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_argmax_0, False)
        ttnn_typecast_71 = ttnn.typecast(ttnn_to_layout_46, ttnn.DataType.INT32, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ttnn_to_layout_46, False)

        ttnn_to_layout_47 = ttnn.to_layout(
            layer_1_outputs["attn_concat"], ttnn.Layout.ROW_MAJOR, None,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(layer_1_outputs["attn_concat"], False)
        ttnn_argmax_1 = ttnn.argmax(
            ttnn_to_layout_47, 3, True, sub_core_grids=None,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_to_layout_47, False)
        ttnn_to_layout_48 = ttnn.to_layout(
            ttnn_argmax_1, ttnn.Layout.TILE, None,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_argmax_1, False)
        ttnn_typecast_72 = ttnn.typecast(ttnn_to_layout_48, ttnn.DataType.INT32, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ttnn_to_layout_48, False)

        ttnn_to_layout_49 = ttnn.to_layout(
            primals_39, ttnn.Layout.TILE, None,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(primals_39, False)

        return [
            layer_0_outputs["v"],
            layer_0_outputs["k_rotated"],
            layer_1_outputs["v"],
            layer_1_outputs["k_rotated"],
            logits,
            weights["model.layers.0.mlp.experts.gate_up_proj"],
            weights["model.layers.0.mlp.experts.gate_up_proj_bias"],
            weights["model.layers.0.mlp.experts.down_proj"],
            weights["model.layers.0.mlp.experts.down_proj_bias"],
            weights["model.layers.1.mlp.experts.gate_up_proj"],
            weights["model.layers.1.mlp.experts.gate_up_proj_bias"],
            weights["model.layers.1.mlp.experts.down_proj"],
            weights["model.layers.1.mlp.experts.down_proj_bias"],
            ttnn_to_layout_49,
            weights["model.rotary_emb.freqs"],
            ttnn_reshape_1,
            layer_0_outputs["input_layernorm_weight"],
            layer_0_outputs["input_rsqrt"],
            layer_0_outputs["input_rms_norm"],
            weights["model.layers.0.self_attn.q_proj.parametrizations.weight.original"],
            weights["model.layers.0.self_attn.k_proj.parametrizations.weight.original"],
            weights["model.layers.0.self_attn.v_proj.parametrizations.weight.original"],
            layer_0_outputs["q_rotated"],
            layer_0_outputs["k_repeated"],
            layer_0_outputs["v_repeated"],
            ttnn_typecast_71,
            layer_0_outputs["attn_weights_sliced"],
            layer_0_outputs["attn_permute"],
            weights["model.layers.0.self_attn.o_proj.parametrizations.weight.original"],
            layer_0_outputs["post_attn_residual"],
            layer_0_outputs["post_attn_layernorm_weight"],
            layer_0_outputs["post_attn_rsqrt"],
            layer_0_outputs["post_attn_rms_norm"],
            weights["model.layers.0.mlp.router.parametrizations.weight.original"],
            layer_0_outputs["topk_indices_sorted"],
            layer_0_outputs["topk_indices"],
            weights["__consteval__.moe_weights_zeros"],
            layer_0_outputs["moe_reshape"],
            layer_0_outputs["expert_repeat"],
            layer_0_outputs["expert_gate_up_matmul"],
            layer_0_outputs["expert_sigmoid"],
            layer_0_outputs["expert_gate_activated"],
            layer_0_outputs["expert_down_matmul"],
            residual_0,
            layer_1_outputs["input_layernorm_weight"],
            layer_1_outputs["input_rsqrt"],
            layer_1_outputs["input_rms_norm"],
            weights["model.layers.1.self_attn.q_proj.parametrizations.weight.original"],
            weights["model.layers.1.self_attn.k_proj.parametrizations.weight.original"],
            weights["model.layers.1.self_attn.v_proj.parametrizations.weight.original"],
            layer_1_outputs["q_rotated"],
            layer_1_outputs["k_repeated"],
            layer_1_outputs["v_repeated"],
            ttnn_typecast_72,
            layer_1_outputs["attn_weights_sliced"],
            layer_1_outputs["attn_permute"],
            weights["model.layers.1.self_attn.o_proj.parametrizations.weight.original"],
            residual_1,
            layer_1_outputs["post_attn_layernorm_weight"],
            layer_1_outputs["post_attn_rsqrt"],
            layer_1_outputs["post_attn_rms_norm"],
            weights["model.layers.1.mlp.router.parametrizations.weight.original"],
            layer_1_outputs["topk_indices_sorted"],
            layer_1_outputs["topk_indices"],
            layer_1_outputs["moe_reshape"],
            layer_1_outputs["expert_repeat"],
            layer_1_outputs["expert_gate_up_matmul"],
            layer_1_outputs["expert_sigmoid"],
            layer_1_outputs["expert_gate_activated"],
            layer_1_outputs["expert_down_matmul"],
            hidden_states_1,
            weights["model.norm.parametrizations.weight.original"],
            rsqrt_out,
            norm_out,
            weights["lm_head.parametrizations.weight.original"],
            layer_1_outputs["router_softmax"],
            layer_1_outputs["attn_softmax"],
            layer_0_outputs["router_softmax"],
            layer_0_outputs["attn_softmax"],
        ]


class GptOssRMSNorm(LightweightModule):
    def __init__(self, device, weights, prefix):
        self.device = device
        self.weight = weights[f"{prefix}.parametrizations.weight.original"]

    def forward(self, x):
        ttnn_typecast = ttnn.typecast(x, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMORY_CONFIG)
        ttnn_pow = ttnn.pow(ttnn_typecast, 2.0, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ttnn_typecast, False)
        ttnn_mean = ttnn.mean(ttnn_pow, [2], True, memory_config=DRAM_MEMORY_CONFIG, compute_kernel_config=None)
        ttnn.deallocate(ttnn_pow, False)
        ttnn_add = ttnn.add(ttnn_mean, 9.9999997473787516e-06, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ttnn_mean, False)
        ttnn_rsqrt = ttnn.rsqrt(ttnn_add, fast_and_approximate_mode=False, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ttnn_add, False)
        ttnn_rms_norm = ttnn.rms_norm(
            x,
            epsilon=9.9999997473787516e-06,
            weight=self.weight,
            bias=None,
            residual_input_tensor=None,
            memory_config=DRAM_MEMORY_CONFIG,
            program_config=None,
            compute_kernel_config=WORMHOLE_COMPUTE_KERNEL_CONFIG,
        )
        return ttnn_rms_norm, ttnn_rsqrt


class GptOssAttention(LightweightModule):
    def __init__(self, device, weights, layer_idx):
        self.device = device
        self.layer_idx = layer_idx
        prefix = f"model.layers.{layer_idx}"
        self.qkv_weight = weights[f"{prefix}.self_attn.qkv_proj.weight"]
        self.qkv_bias = weights[f"{prefix}.self_attn.qkv_proj.bias"]
        self.o_proj_weight = weights[f"{prefix}.self_attn.o_proj.parametrizations.weight.original"]
        self.o_proj_bias = weights[f"{prefix}.self_attn.o_proj.bias"]
        self.sinks = weights[f"{prefix}.self_attn.sinks"]
        self.cos = weights["model.rotary_emb.cos"]
        self.sin = weights["model.rotary_emb.sin"]
        self.attn_scale = weights["__consteval__.attn_scale"]
        self.causal_mask = weights["__consteval__.causal_mask"]

    def forward(self, hidden_states_2d):
        ttnn_linear = ttnn.linear(
            hidden_states_2d,
            self.qkv_weight,
            bias=self.qkv_bias,
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(hidden_states_2d, False)
        ttnn_reshape_qkv = ttnn.reshape(ttnn_linear, [1, 17, 1280], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ttnn_linear, False)

        q, v, k = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn_reshape_qkv,
            None,
            num_heads=16,
            num_kv_heads=2,
            transpose_key=False,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_reshape_qkv, False)

        k_rotated_full = ttnn.experimental.rotary_embedding(
            k, self.cos, self.sin, None,
            memory_config=DRAM_MEMORY_CONFIG,
            compute_kernel_config=None,
        )
        ttnn.deallocate(k, False)
        k_rotated = ttnn.slice(k_rotated_full, [0, 0, 0, 0], [1, 2, 17, 64], [1, 1, 1, 1], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(k_rotated_full, False)

        q_rotated_full = ttnn.experimental.rotary_embedding(
            q, self.cos, self.sin, None,
            memory_config=DRAM_MEMORY_CONFIG,
            compute_kernel_config=None,
        )
        ttnn.deallocate(q, False)
        q_rotated = ttnn.slice(q_rotated_full, [0, 0, 0, 0], [1, 16, 17, 64], [1, 1, 1, 1], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_rotated_full, False)

        k_reshaped = ttnn.reshape(k_rotated, [1, 2, 1, 17, 64], memory_config=DRAM_MEMORY_CONFIG)
        k_repeated = ttnn.repeat(k_reshaped, ttnn.Shape([1, 1, 8, 1, 1]), memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(k_reshaped, False)
        k_expanded = ttnn.reshape(k_repeated, [1, 16, 17, 64], memory_config=DRAM_MEMORY_CONFIG)
        k_transposed = ttnn.permute(k_expanded, [0, 1, 3, 2], memory_config=DRAM_MEMORY_CONFIG, pad_value=0.0)
        ttnn.deallocate(k_expanded, False)

        attn_weights = ttnn.matmul(
            q_rotated, k_transposed,
            transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        ttnn.deallocate(k_transposed, False)
        attn_weights_scaled = ttnn.multiply(attn_weights, self.attn_scale, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_weights, False)
        attn_weights_masked = ttnn.add(attn_weights_scaled, self.causal_mask, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_weights_scaled, False)

        attn_concat = ttnn.concat([attn_weights_masked, self.sinks], 3, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_weights_masked, False)
        attn_softmax = ttnn.softmax(attn_concat, 3, memory_config=DRAM_MEMORY_CONFIG, compute_kernel_config=None, numeric_stable=True)
        attn_weights_sliced = ttnn.slice(attn_softmax, [0, 0, 0, 0], [1, 16, 17, 17], [1, 1, 1, 1], memory_config=DRAM_MEMORY_CONFIG)

        v_reshaped = ttnn.reshape(v, [1, 2, 1, 17, 64], memory_config=DRAM_MEMORY_CONFIG)
        v_repeated = ttnn.repeat(v_reshaped, ttnn.Shape([1, 1, 8, 1, 1]), memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(v_reshaped, False)
        v_expanded = ttnn.reshape(v_repeated, [1, 16, 17, 64], memory_config=DRAM_MEMORY_CONFIG)

        attn_output = ttnn.matmul(
            attn_weights_sliced, v_expanded,
            transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        ttnn.deallocate(v_expanded, False)
        attn_permute = ttnn.permute(attn_output, [0, 2, 1, 3], memory_config=DRAM_MEMORY_CONFIG, pad_value=0.0)
        concat_heads = ttnn.transformer.concatenate_heads(attn_output, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_output, False)

        reshaped_heads = ttnn.reshape(concat_heads, [17, 1024], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(concat_heads, False)
        o_proj_out = ttnn.matmul(
            reshaped_heads, self.o_proj_weight,
            transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        ttnn.deallocate(reshaped_heads, False)
        o_proj_reshaped = ttnn.reshape(o_proj_out, [1, 1, 17, 2880], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(o_proj_out, False)

        o_proj_reduced = ttnn.all_reduce(
            input_tensor=o_proj_reshaped,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=DRAM_MEMORY_CONFIG,
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(o_proj_reshaped, False)
        o_proj_2d = ttnn.reshape(o_proj_reduced, [17, 2880], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(o_proj_reduced, False)
        attn_out_with_bias = ttnn.add(o_proj_2d, self.o_proj_bias, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(o_proj_2d, False)

        return {
            "attn_output": attn_out_with_bias,
            "v": v,
            "k_rotated": k_rotated,
            "q_rotated": q_rotated,
            "k_repeated": k_repeated,
            "v_repeated": v_repeated,
            "attn_weights_sliced": attn_weights_sliced,
            "attn_permute": attn_permute,
            "attn_concat": attn_concat,
            "attn_softmax": attn_softmax,
        }


class GptOssMLP(LightweightModule):
    def __init__(self, device, weights, layer_idx):
        self.device = device
        prefix = f"model.layers.{layer_idx}"
        self.router_weight = weights[f"{prefix}.mlp.router.parametrizations.weight.original"]
        self.router_bias = weights[f"{prefix}.mlp.router.bias"]
        self.gate_up_proj = weights[f"{prefix}.mlp.experts.gate_up_proj"]
        self.gate_up_proj_bias = weights[f"{prefix}.mlp.experts.gate_up_proj_bias"]
        self.down_proj = weights[f"{prefix}.mlp.experts.down_proj"]
        self.down_proj_bias = weights[f"{prefix}.mlp.experts.down_proj_bias"]
        self.ones_scalar = weights["__consteval__.ones_scalar"]
        self.sigmoid_scale = weights["__consteval__.sigmoid_scale"]
        self.scatter_index = weights["__consteval__.scatter_index"]
        self.moe_scatter_zeros = weights["__consteval__.moe_scatter_zeros"]

    def forward(self, hidden_states_2d):
        ttnn_concat_experts = ttnn.concat(
            [hidden_states_2d] * 32,
            0,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn_reshape_experts = ttnn.reshape(ttnn_concat_experts, [32, 17, 2880], memory_config=DRAM_MEMORY_CONFIG)

        expert_gate_up_matmul = ttnn.matmul(
            ttnn_reshape_experts, self.gate_up_proj,
            transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_reshape_experts, False)
        gate_up_with_bias = ttnn.add(expert_gate_up_matmul, self.gate_up_proj_bias, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)

        gate_slice = ttnn.slice(gate_up_with_bias, [0, 0, 1], [32, 17, 5760], [1, 1, 2], memory_config=DRAM_MEMORY_CONFIG)
        gate_clamped = ttnn.clamp(gate_slice, -7.0, 7.0, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate_slice, False)
        gate_activated = ttnn.add(gate_clamped, self.ones_scalar, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate_clamped, False)

        up_slice = ttnn.slice(gate_up_with_bias, [0, 0, 0], [32, 17, 5760], [1, 1, 2], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate_up_with_bias, False)
        up_clamped = ttnn.clamp(up_slice, float("-inf"), 7.0, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(up_slice, False)

        up_scaled = ttnn.multiply(up_clamped, self.sigmoid_scale, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        expert_sigmoid = ttnn.sigmoid(up_scaled, vector_mode=4, mode=ttnn.SigmoidMode.Accurate, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(up_scaled, False)
        expert_gate_activated = ttnn.multiply(up_clamped, expert_sigmoid, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(up_clamped, False)
        expert_hidden = ttnn.multiply(gate_activated, expert_gate_activated, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(expert_gate_activated, False)
        ttnn.deallocate(gate_activated, False)

        expert_down_matmul = ttnn.matmul(
            expert_hidden, self.down_proj,
            transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        expert_down_with_bias = ttnn.add(expert_down_matmul, self.down_proj_bias, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)

        ttnn_typecast_router = ttnn.typecast(hidden_states_2d, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(hidden_states_2d, False)
        router_logits = ttnn.linear(
            ttnn_typecast_router, self.router_weight,
            bias=self.router_bias,
            transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.FLOAT32,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_typecast_router, False)
        router_bf16 = ttnn.typecast(router_logits, ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(router_logits, False)

        topk_vals_sorted, topk_indices_sorted = ttnn.topk(router_bf16, 4, 1, True, True, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(topk_vals_sorted, False)
        topk_indices_sorted_int = ttnn.typecast(topk_indices_sorted, ttnn.DataType.INT32, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(topk_indices_sorted, False)

        topk_vals, topk_indices = ttnn.topk(router_bf16, 4, -1, True, True, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(router_bf16, False)
        topk_indices_int = ttnn.typecast(topk_indices, ttnn.DataType.INT32, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(topk_indices, False)
        router_softmax = ttnn.softmax(topk_vals, 1, memory_config=DRAM_MEMORY_CONFIG, compute_kernel_config=None, numeric_stable=True)
        ttnn.deallocate(topk_vals, False)

        indices_reshaped = ttnn.reshape(topk_indices_int, [17, 4, 1], memory_config=DRAM_MEMORY_CONFIG)
        scatter_idx = ttnn.add(self.scatter_index, indices_reshaped, dtype=ttnn.DataType.INT32, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(indices_reshaped, False)
        scatter_idx_flat = ttnn.reshape(scatter_idx, [68], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(scatter_idx, False)
        softmax_flat = ttnn.reshape(router_softmax, [68], memory_config=DRAM_MEMORY_CONFIG)

        scatter_idx_rm = ttnn.to_layout(scatter_idx_flat, ttnn.Layout.ROW_MAJOR, None, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(scatter_idx_flat, False)
        softmax_rm = ttnn.to_layout(softmax_flat, ttnn.Layout.ROW_MAJOR, None, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(softmax_flat, False)

        moe_weights = ttnn.scatter(
            input=self.moe_scatter_zeros,
            dim=0,
            index=scatter_idx_rm,
            src=softmax_rm,
            memory_config=DRAM_MEMORY_CONFIG,
            reduce=None,
        )
        ttnn.deallocate(softmax_rm, False)
        ttnn.deallocate(scatter_idx_rm, False)

        moe_weights_tiled = ttnn.to_layout(moe_weights, ttnn.Layout.TILE, None, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(moe_weights, False)
        moe_reshape = ttnn.reshape(moe_weights_tiled, [17, 32], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(moe_weights_tiled, False)
        moe_permuted = ttnn.permute(moe_reshape, [1, 0], memory_config=DRAM_MEMORY_CONFIG, pad_value=0.0)
        moe_weights_3d = ttnn.reshape(moe_permuted, [32, 17, 1], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(moe_permuted, False)

        weighted_experts = ttnn.multiply(expert_down_with_bias, moe_weights_3d, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(moe_weights_3d, False)
        ttnn.deallocate(expert_down_with_bias, False)
        mlp_output = ttnn.sum(weighted_experts, [0], True, memory_config=DRAM_MEMORY_CONFIG, compute_kernel_config=None)
        ttnn.deallocate(weighted_experts, False)

        return {
            "mlp_output": mlp_output,
            "topk_indices_sorted": topk_indices_sorted_int,
            "topk_indices": topk_indices_int,
            "moe_reshape": moe_reshape,
            "expert_repeat": ttnn_concat_experts,
            "expert_gate_up_matmul": expert_gate_up_matmul,
            "expert_sigmoid": expert_sigmoid,
            "expert_gate_activated": expert_hidden,
            "expert_down_matmul": expert_down_matmul,
            "router_softmax": router_softmax,
        }


class GptOssDecoderLayer(LightweightModule):
    def __init__(self, device, weights, layer_idx):
        self.device = device
        self.layer_idx = layer_idx
        prefix = f"model.layers.{layer_idx}"
        self.input_layernorm = GptOssRMSNorm(device, weights, f"{prefix}.input_layernorm")
        self.self_attn = GptOssAttention(device, weights, layer_idx)
        self.post_attn_layernorm = GptOssRMSNorm(device, weights, f"{prefix}.post_attention_layernorm")
        self.mlp = GptOssMLP(device, weights, layer_idx)
        self.input_layernorm_weight = weights[f"{prefix}.input_layernorm.parametrizations.weight.original"]
        self.post_attn_layernorm_weight = weights[f"{prefix}.post_attention_layernorm.parametrizations.weight.original"]

    def forward(self, hidden_states, residual):
        input_rms_norm, input_rsqrt = self.input_layernorm(hidden_states)
        hidden_states_2d = ttnn.reshape(input_rms_norm, [17, 2880], memory_config=DRAM_MEMORY_CONFIG)

        attn_outputs = self.self_attn(hidden_states_2d)

        attn_3d = ttnn.reshape(attn_outputs["attn_output"], [1, 17, 2880], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_outputs["attn_output"], False)
        post_attn_residual = ttnn.add(residual, attn_3d, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_3d, False)
        ttnn.deallocate(residual, False)

        post_attn_hidden = ttnn.reshape(post_attn_residual, [1, 17, 2880], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(post_attn_residual, False)

        post_attn_rms_norm, post_attn_rsqrt = self.post_attn_layernorm(post_attn_hidden)
        post_attn_2d = ttnn.reshape(post_attn_rms_norm, [17, 2880], memory_config=DRAM_MEMORY_CONFIG)

        mlp_outputs = self.mlp(post_attn_2d)

        new_hidden = ttnn.add(post_attn_hidden, mlp_outputs["mlp_output"], dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mlp_outputs["mlp_output"], False)

        return {
            "hidden_states": new_hidden,
            "residual": new_hidden,
            "post_attn_residual": post_attn_hidden,
            "input_layernorm_weight": self.input_layernorm_weight,
            "input_rsqrt": input_rsqrt,
            "input_rms_norm": input_rms_norm,
            "post_attn_layernorm_weight": self.post_attn_layernorm_weight,
            "post_attn_rsqrt": post_attn_rsqrt,
            "post_attn_rms_norm": post_attn_rms_norm,
            "v": attn_outputs["v"],
            "k_rotated": attn_outputs["k_rotated"],
            "q_rotated": attn_outputs["q_rotated"],
            "k_repeated": attn_outputs["k_repeated"],
            "v_repeated": attn_outputs["v_repeated"],
            "attn_weights_sliced": attn_outputs["attn_weights_sliced"],
            "attn_permute": attn_outputs["attn_permute"],
            "attn_concat": attn_outputs["attn_concat"],
            "attn_softmax": attn_outputs["attn_softmax"],
            "topk_indices_sorted": mlp_outputs["topk_indices_sorted"],
            "topk_indices": mlp_outputs["topk_indices"],
            "moe_reshape": mlp_outputs["moe_reshape"],
            "expert_repeat": mlp_outputs["expert_repeat"],
            "expert_gate_up_matmul": mlp_outputs["expert_gate_up_matmul"],
            "expert_sigmoid": mlp_outputs["expert_sigmoid"],
            "expert_gate_activated": mlp_outputs["expert_gate_activated"],
            "expert_down_matmul": mlp_outputs["expert_down_matmul"],
            "router_softmax": mlp_outputs["router_softmax"],
        }


class GptOssLMHead(LightweightModule):
    def __init__(self, device, weights):
        self.device = device
        self.weight = weights["lm_head.parametrizations.weight.original"]

    def forward(self, hidden_states):
        hidden_2d = ttnn.reshape(hidden_states, [17, 2880], memory_config=DRAM_MEMORY_CONFIG)
        logits = ttnn.matmul(
            hidden_2d, self.weight,
            transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        ttnn.deallocate(hidden_2d, False)
        logits_3d = ttnn.reshape(logits, [1, 17, 201088], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(logits, False)
        return logits_3d
