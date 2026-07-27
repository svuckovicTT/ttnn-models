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


class GptOssAttention(LightweightModule):
    def __init__(self, weights, layer_idx):
        prefix = f"model.layers.{layer_idx}"
        self.qkv_weight = weights[f"{prefix}.self_attn.qkv_proj.weight"]
        self.qkv_bias = weights[f"{prefix}.self_attn.qkv_proj.bias"]
        self.o_proj_weight = weights[f"{prefix}.self_attn.o_proj.parametrizations.weight.original"]
        self.o_proj_bias = weights[f"{prefix}.self_attn.o_proj.bias"]
        self.sinks = weights[f"{prefix}.self_attn.sinks"]

    def forward(self, hidden_states, rotary_cos, rotary_sin, causal_mask, attn_scale):
        ttnn_reshape_2 = ttnn.reshape(hidden_states, [17, 2880], memory_config=DRAM_MEMORY_CONFIG)
        ttnn_linear = ttnn.linear(
            ttnn_reshape_2, self.qkv_weight, bias=self.qkv_bias,
            transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG, dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_reshape_2, False)
        ttnn_reshape_3 = ttnn.reshape(ttnn_linear, [1, 17, 1280], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ttnn_linear, False)

        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn_reshape_3, None, num_heads=16, num_kv_heads=2,
            transpose_key=False, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_reshape_3, False)

        rotary_key = ttnn.experimental.rotary_embedding(
            key, rotary_cos, rotary_sin, None,
            memory_config=DRAM_MEMORY_CONFIG, compute_kernel_config=None,
        )
        ttnn.deallocate(key, False)
        key_rotated = ttnn.slice(rotary_key, [0, 0, 0, 0], [1, 2, 17, 64], [1, 1, 1, 1], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(rotary_key, False)

        rotary_query = ttnn.experimental.rotary_embedding(
            query, rotary_cos, rotary_sin, None,
            memory_config=DRAM_MEMORY_CONFIG, compute_kernel_config=None,
        )
        ttnn.deallocate(query, False)
        query_rotated = ttnn.slice(rotary_query, [0, 0, 0, 0], [1, 16, 17, 64], [1, 1, 1, 1], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(rotary_query, False)

        # repeat_kv for key
        key_5d = ttnn.reshape(key_rotated, [1, 2, 1, 17, 64], memory_config=DRAM_MEMORY_CONFIG)
        key_repeated = ttnn.repeat(key_5d, ttnn.Shape([1, 1, 8, 1, 1]), memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(key_5d, False)
        key_expanded = ttnn.reshape(key_repeated, [1, 16, 17, 64], memory_config=DRAM_MEMORY_CONFIG)
        key_transposed = ttnn.permute(key_expanded, [0, 1, 3, 2], memory_config=DRAM_MEMORY_CONFIG, pad_value=0.0)
        ttnn.deallocate(key_expanded, False)

        # Q @ K^T * scale
        attn_weights = ttnn.matmul(
            query_rotated, key_transposed, transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG, dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        ttnn.deallocate(key_transposed, False)
        attn_weights = ttnn.multiply(attn_weights, attn_scale, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)

        # causal mask + sinks
        attn_weights = ttnn.add(attn_weights, causal_mask, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        attn_with_sinks = ttnn.concat([attn_weights, self.sinks], 3, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_weights, False)

        # softmax
        attn_probs = ttnn.softmax(attn_with_sinks, 3, memory_config=DRAM_MEMORY_CONFIG, compute_kernel_config=None, numeric_stable=True)
        attn_probs_sliced = ttnn.slice(attn_probs, [0, 0, 0, 0], [1, 16, 17, 17], [1, 1, 1, 1], memory_config=DRAM_MEMORY_CONFIG)

        # repeat_kv for value
        value_5d = ttnn.reshape(value, [1, 2, 1, 17, 64], memory_config=DRAM_MEMORY_CONFIG)
        value_repeated = ttnn.repeat(value_5d, ttnn.Shape([1, 1, 8, 1, 1]), memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(value_5d, False)
        value_expanded = ttnn.reshape(value_repeated, [1, 16, 17, 64], memory_config=DRAM_MEMORY_CONFIG)

        # attn @ V
        attn_output = ttnn.matmul(
            attn_probs_sliced, value_expanded, transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG, dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        ttnn.deallocate(value_expanded, False)

        attn_output_permuted = ttnn.permute(attn_output, [0, 2, 1, 3], memory_config=DRAM_MEMORY_CONFIG, pad_value=0.0)
        concat_heads = ttnn.transformer.concatenate_heads(attn_output, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_output, False)

        # o_proj
        reshaped = ttnn.reshape(concat_heads, [17, 1024], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(concat_heads, False)
        o_proj_out = ttnn.matmul(
            reshaped, self.o_proj_weight, transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG, dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        ttnn.deallocate(reshaped, False)

        o_proj_reshaped = ttnn.reshape(o_proj_out, [1, 1, 17, 2880], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(o_proj_out, False)
        all_reduced = ttnn.all_reduce(
            input_tensor=o_proj_reshaped, cluster_axis=1, subdevice_id=None,
            memory_config=DRAM_MEMORY_CONFIG, num_links=None, topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(o_proj_reshaped, False)
        flat = ttnn.reshape(all_reduced, [17, 2880], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(all_reduced, False)
        with_bias = ttnn.add(flat, self.o_proj_bias, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(flat, False)

        return (
            with_bias,
            value, key_rotated, query_rotated, key_repeated, value_repeated,
            attn_probs_sliced, attn_output_permuted, attn_with_sinks, attn_probs,
        )


class GptOssExperts(LightweightModule):
    def __init__(self, weights, layer_idx):
        prefix = f"model.layers.{layer_idx}"
        self.gate_up_proj = weights[f"{prefix}.mlp.experts.gate_up_proj"]
        self.gate_up_proj_bias = weights[f"{prefix}.mlp.experts.gate_up_proj_bias"]
        self.down_proj = weights[f"{prefix}.mlp.experts.down_proj"]
        self.down_proj_bias = weights[f"{prefix}.mlp.experts.down_proj_bias"]

    def forward(self, hidden_states, ones_scalar, sigmoid_scale):
        gate_up = ttnn.matmul(
            hidden_states, self.gate_up_proj, transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG, dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        ttnn.deallocate(hidden_states, False)
        gate_up = ttnn.add(gate_up, self.gate_up_proj_bias, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)

        # _apply_gate: split gate and up, apply SiLU-like gating
        gate = ttnn.slice(gate_up, [0, 0, 1], [32, 17, 5760], [1, 1, 2], memory_config=DRAM_MEMORY_CONFIG)
        gate_clamped = ttnn.clamp(gate, -7.0, 7.0, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate, False)
        gate_shifted = ttnn.add(gate_clamped, ones_scalar, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate_clamped, False)

        up = ttnn.slice(gate_up, [0, 0, 0], [32, 17, 5760], [1, 1, 2], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate_up, False)
        up_clamped = ttnn.clamp(up, float("-inf"), 7.0, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(up, False)

        scaled = ttnn.multiply(up_clamped, sigmoid_scale, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        sigmoid = ttnn.sigmoid(scaled, vector_mode=4, mode=ttnn.SigmoidMode.Accurate, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(scaled, False)
        activated = ttnn.multiply(up_clamped, sigmoid, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(up_clamped, False)
        gated = ttnn.multiply(gate_shifted, activated, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(activated, False)
        ttnn.deallocate(gate_shifted, False)

        # down_proj
        down = ttnn.matmul(
            gated, self.down_proj, transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG, dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        down_with_bias = ttnn.add(down, self.down_proj_bias, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)

        return down_with_bias, gate_up, down, sigmoid, gated


class GptOssTopKRouter(LightweightModule):
    def __init__(self, weights, layer_idx):
        prefix = f"model.layers.{layer_idx}"
        self.router_weight = weights[f"{prefix}.mlp.router.parametrizations.weight.original"]
        self.router_bias = weights[f"{prefix}.mlp.router.bias"]

    def forward(self, hidden_states_2d):
        hidden_fp32 = ttnn.typecast(hidden_states_2d, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(hidden_states_2d, False)
        router_logits = ttnn.linear(
            hidden_fp32, self.router_weight, bias=self.router_bias,
            transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG, dtype=ttnn.DataType.FLOAT32,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        ttnn.deallocate(hidden_fp32, False)
        router_logits_bf16 = ttnn.typecast(router_logits, ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(router_logits, False)

        _, top_indices_for_mark = ttnn.topk(router_logits_bf16, 4, 1, True, True, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(_, False)
        top_indices_int32 = ttnn.typecast(top_indices_for_mark, ttnn.DataType.INT32, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(top_indices_for_mark, False)

        top_values, top_indices_raw = ttnn.topk(router_logits_bf16, 4, -1, True, True, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(router_logits_bf16, False)
        top_indices_int = ttnn.typecast(top_indices_raw, ttnn.DataType.INT32, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(top_indices_raw, False)

        router_weights = ttnn.softmax(top_values, 1, memory_config=DRAM_MEMORY_CONFIG, compute_kernel_config=None, numeric_stable=True)
        ttnn.deallocate(top_values, False)

        return router_weights, top_indices_int, top_indices_int32


class GptOssMLP(LightweightModule):
    def __init__(self, weights, layer_idx):
        self.router = GptOssTopKRouter(weights, layer_idx)
        self.experts = GptOssExperts(weights, layer_idx)

    def forward(self, hidden_states, scatter_zeros, scatter_index, ones_scalar, sigmoid_scale):
        hidden_2d = ttnn.reshape(hidden_states, [17, 2880], memory_config=DRAM_MEMORY_CONFIG)

        # Replicate input for all 32 experts
        expert_input = ttnn.concat([hidden_2d] * 32, 0, memory_config=DRAM_MEMORY_CONFIG)
        expert_input_3d = ttnn.reshape(expert_input, [32, 17, 2880], memory_config=DRAM_MEMORY_CONFIG)

        # Expert computation
        expert_output, expert_gate_up, expert_down, expert_sigmoid, expert_gated = self.experts(
            expert_input_3d, ones_scalar, sigmoid_scale
        )

        # Router computation (uses hidden_2d which gets deallocated inside)
        router_weights, top_indices, top_indices_mark = self.router(hidden_2d)

        # Scatter routing weights
        indices_reshaped = ttnn.reshape(top_indices, [17, 4, 1], memory_config=DRAM_MEMORY_CONFIG)
        flat_indices = ttnn.add(scatter_index, indices_reshaped, dtype=ttnn.DataType.INT32, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(indices_reshaped, False)
        flat_indices = ttnn.reshape(flat_indices, [68], memory_config=DRAM_MEMORY_CONFIG)

        flat_weights = ttnn.reshape(router_weights, [68], memory_config=DRAM_MEMORY_CONFIG)

        flat_indices_rm = ttnn.to_layout(flat_indices, ttnn.Layout.ROW_MAJOR, None, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(flat_indices, False)
        flat_weights_rm = ttnn.to_layout(flat_weights, ttnn.Layout.ROW_MAJOR, None, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(flat_weights, False)

        scattered = ttnn.scatter(
            input=scatter_zeros, dim=0, index=flat_indices_rm, src=flat_weights_rm,
            memory_config=DRAM_MEMORY_CONFIG, reduce=None,
        )
        ttnn.deallocate(flat_weights_rm, False)
        ttnn.deallocate(flat_indices_rm, False)

        scattered_tile = ttnn.to_layout(scattered, ttnn.Layout.TILE, None, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(scattered, False)
        weight_matrix = ttnn.reshape(scattered_tile, [17, 32], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(scattered_tile, False)

        # Transpose and reshape for weighted sum
        weight_permuted = ttnn.permute(weight_matrix, [1, 0], memory_config=DRAM_MEMORY_CONFIG, pad_value=0.0)
        weight_3d = ttnn.reshape(weight_permuted, [32, 17, 1], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(weight_permuted, False)

        # Weighted sum of expert outputs
        weighted = ttnn.multiply(expert_output, weight_3d, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(weight_3d, False)
        ttnn.deallocate(expert_output, False)
        mlp_output = ttnn.sum(weighted, [0], True, memory_config=DRAM_MEMORY_CONFIG, compute_kernel_config=None)
        ttnn.deallocate(weighted, False)

        return (
            mlp_output,
            top_indices_mark, top_indices, weight_matrix,
            expert_input, expert_gate_up, expert_sigmoid, expert_gated, expert_down,
            router_weights,
        )


class GptOssRMSNorm(LightweightModule):
    def __init__(self, weight):
        self.weight = weight

    def forward(self, hidden_states, eps_tensor):
        hidden_fp32 = ttnn.typecast(hidden_states, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMORY_CONFIG)
        squared = ttnn.pow(hidden_fp32, 2.0, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(hidden_fp32, False)
        mean = ttnn.mean(squared, [2], True, memory_config=DRAM_MEMORY_CONFIG, compute_kernel_config=None)
        ttnn.deallocate(squared, False)
        with_eps = ttnn.add(mean, eps_tensor, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mean, False)
        rsqrt = ttnn.rsqrt(with_eps, fast_and_approximate_mode=False, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(with_eps, False)
        normed = ttnn.rms_norm(
            hidden_states, epsilon=9.9999997473787516e-06, weight=self.weight,
            bias=None, residual_input_tensor=None, memory_config=DRAM_MEMORY_CONFIG,
            program_config=None, compute_kernel_config=WORMHOLE_COMPUTE_KERNEL_CONFIG,
        )
        return normed, rsqrt


class GptOssDecoderLayer(LightweightModule):
    def __init__(self, weights, layer_idx):
        self.input_layernorm = GptOssRMSNorm(
            weights[f"model.layers.{layer_idx}.input_layernorm.parametrizations.weight.original"]
        )
        self.self_attn = GptOssAttention(weights, layer_idx)
        self.post_attention_layernorm = GptOssRMSNorm(
            weights[f"model.layers.{layer_idx}.post_attention_layernorm.parametrizations.weight.original"]
        )
        self.mlp = GptOssMLP(weights, layer_idx)

    def forward(self, hidden_states, residual_input, rotary_cos, rotary_sin, causal_mask, attn_scale,
                scatter_zeros, scatter_index, ones_scalar, sigmoid_scale, eps_tensor):
        # input_layernorm (on 3D hidden_states)
        normed, rsqrt_0 = self.input_layernorm(hidden_states, eps_tensor)

        # self_attn
        (attn_out, value, key_rotated, query_rotated, key_repeated, value_repeated,
         attn_probs_sliced, attn_output_permuted, attn_with_sinks, attn_probs) = self.self_attn(
            normed, rotary_cos, rotary_sin, causal_mask, attn_scale
        )

        # residual connection: residual_input (pre-reshape) + attn_out (2D)
        attn_residual = ttnn.add(residual_input, attn_out, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out, False)
        ttnn.deallocate(residual_input, False)

        post_attn_hidden = ttnn.reshape(attn_residual, [1, 17, 2880], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_residual, False)

        # post_attention_layernorm
        post_normed, rsqrt_1 = self.post_attention_layernorm(post_attn_hidden, eps_tensor)

        # mlp
        (mlp_output, top_indices_mark, top_indices, weight_matrix,
         expert_input, expert_gate_up, expert_sigmoid, expert_gated, expert_down,
         router_weights) = self.mlp(
            post_normed, scatter_zeros, scatter_index, ones_scalar, sigmoid_scale
        )

        # residual connection
        output = ttnn.add(post_attn_hidden, mlp_output, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mlp_output, False)

        return (
            output,
            post_attn_hidden, rsqrt_0, normed, rsqrt_1, post_normed,
            value, key_rotated, query_rotated, key_repeated, value_repeated,
            attn_probs_sliced, attn_output_permuted, attn_with_sinks, attn_probs,
            top_indices_mark, top_indices, weight_matrix,
            expert_input, expert_gate_up, expert_sigmoid, expert_gated, expert_down,
            router_weights,
        )


class ModelTTNN(LightweightModule):
    def __init__(self, device):
        self.device = device
        self.weights = params.load_weights_for__main(device)
        self.weights = consteval.run_consteval(self.weights, device)

        self.layers = [GptOssDecoderLayer(self.weights, i) for i in range(2)]
        self.final_norm = GptOssRMSNorm(self.weights["model.norm.parametrizations.weight.original"])

    def forward(self, activations):
        device = self.device
        weights = self.weights
        primals_39 = activations[0]

        # Shared constants
        causal_mask = weights["__consteval__.causal_mask"]
        rotary_cos = weights["model.rotary_emb.cos"]
        rotary_sin = weights["model.rotary_emb.sin"]
        scatter_zeros = weights["__consteval__.moe_scatter_zeros"]
        scatter_index = weights["__consteval__.scatter_index"]
        ones_scalar = weights["__consteval__.ones_scalar"]
        attn_scale = weights["__consteval__.attn_scale"]
        eps_tensor = weights["__consteval__.rms_norm_eps"]
        sigmoid_scale = weights["__consteval__.sigmoid_scale"]

        # Embedding
        input_ids = ttnn.typecast(primals_39, ttnn.DataType.UINT32, memory_config=DRAM_MEMORY_CONFIG)
        flat_ids = ttnn.reshape(input_ids, [17], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(input_ids, False)
        embedding = ttnn.embedding(
            flat_ids, weights["model.embed_tokens.weight"], padding_idx=None,
            layout=ttnn.Layout.TILE, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(flat_ids, False)
        embedding_reshaped = ttnn.reshape(embedding, [1, 17, 2880], memory_config=DRAM_MEMORY_CONFIG)

        # Decoder layers
        # For layer 0: hidden_states=[1,17,2880] for layernorm, residual_input=embedding (pre-reshape) for residual add
        # For layer N>0: hidden_states=prev_layer_output, residual_input=prev_layer_output for residual add
        hidden_states = embedding_reshaped
        residual_input = embedding  # layer 0 residual uses raw embedding
        layer_outputs = []
        layer_intermediates = []
        for layer in self.layers:
            (hidden_states,
             post_attn_hidden, rsqrt_0, normed, rsqrt_1, post_normed,
             value, key_rotated, query_rotated, key_repeated, value_repeated,
             attn_probs_sliced, attn_output_permuted, attn_with_sinks, attn_probs,
             top_indices_mark, top_indices, weight_matrix,
             expert_input, expert_gate_up, expert_sigmoid, expert_gated, expert_down,
             router_weights) = layer(
                hidden_states, residual_input, rotary_cos, rotary_sin, causal_mask, attn_scale,
                scatter_zeros, scatter_index, ones_scalar, sigmoid_scale, eps_tensor,
            )
            layer_outputs.append(hidden_states)
            residual_input = hidden_states  # next layer's residual uses this layer's output
            layer_intermediates.append((
                post_attn_hidden, rsqrt_0, normed, rsqrt_1, post_normed,
                value, key_rotated, query_rotated, key_repeated, value_repeated,
                attn_probs_sliced, attn_output_permuted, attn_with_sinks, attn_probs,
                top_indices_mark, top_indices, weight_matrix,
                expert_input, expert_gate_up, expert_sigmoid, expert_gated, expert_down,
                router_weights,
            ))

        # Final RMS norm
        final_normed, rsqrt_4 = self.final_norm(hidden_states, eps_tensor)

        # lm_head
        flat_normed = ttnn.reshape(final_normed, [17, 2880], memory_config=DRAM_MEMORY_CONFIG)
        logits = ttnn.matmul(
            flat_normed, weights["lm_head.parametrizations.weight.original"],
            transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG, dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        ttnn.deallocate(flat_normed, False)
        logits_reshaped = ttnn.reshape(logits, [1, 17, 201088], memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(logits, False)

        # Argmax on attention (from concat_4 / concat_6 of layers)
        l0 = layer_intermediates[0]
        l1 = layer_intermediates[1]

        attn_with_sinks_0 = l0[12]  # attn_with_sinks layer 0
        concat_4_rm = ttnn.to_layout(attn_with_sinks_0, ttnn.Layout.ROW_MAJOR, None, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_with_sinks_0, False)
        argmax_0 = ttnn.argmax(concat_4_rm, 3, True, sub_core_grids=None, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(concat_4_rm, False)
        argmax_0_tile = ttnn.to_layout(argmax_0, ttnn.Layout.TILE, None, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(argmax_0, False)
        typecast_71 = ttnn.typecast(argmax_0_tile, ttnn.DataType.INT32, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(argmax_0_tile, False)

        attn_with_sinks_1 = l1[12]  # attn_with_sinks layer 1
        concat_6_rm = ttnn.to_layout(attn_with_sinks_1, ttnn.Layout.ROW_MAJOR, None, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_with_sinks_1, False)
        argmax_1 = ttnn.argmax(concat_6_rm, 3, True, sub_core_grids=None, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(concat_6_rm, False)
        argmax_1_tile = ttnn.to_layout(argmax_1, ttnn.Layout.TILE, None, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(argmax_1, False)
        typecast_72 = ttnn.typecast(argmax_1_tile, ttnn.DataType.INT32, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(argmax_1_tile, False)

        primals_tile = ttnn.to_layout(primals_39, ttnn.Layout.TILE, None, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(primals_39, False)

        # Unpack layer intermediates for return
        # Layer 0
        (post_attn_hidden_0, rsqrt_0_0, normed_0, rsqrt_1_0, post_normed_0,
         value_0, key_rotated_0, query_rotated_0, key_repeated_0, value_repeated_0,
         attn_probs_sliced_0, attn_output_permuted_0, _, attn_probs_0,
         top_indices_mark_0, top_indices_0, weight_matrix_0,
         expert_input_0, expert_gate_up_0, expert_sigmoid_0, expert_gated_0, expert_down_0,
         router_weights_0) = l0

        # Layer 1
        (post_attn_hidden_1, rsqrt_0_1, normed_1, rsqrt_1_1, post_normed_1,
         value_1, key_rotated_1, query_rotated_1, key_repeated_1, value_repeated_1,
         attn_probs_sliced_1, attn_output_permuted_1, _, attn_probs_1,
         top_indices_mark_1, top_indices_1, weight_matrix_1,
         expert_input_1, expert_gate_up_1, expert_sigmoid_1, expert_gated_1, expert_down_1,
         router_weights_1) = l1

        return [
            value_0,
            key_rotated_0,
            value_1,
            key_rotated_1,
            logits_reshaped,
            weights["model.layers.0.mlp.experts.gate_up_proj"],
            weights["model.layers.0.mlp.experts.gate_up_proj_bias"],
            weights["model.layers.0.mlp.experts.down_proj"],
            weights["model.layers.0.mlp.experts.down_proj_bias"],
            weights["model.layers.1.mlp.experts.gate_up_proj"],
            weights["model.layers.1.mlp.experts.gate_up_proj_bias"],
            weights["model.layers.1.mlp.experts.down_proj"],
            weights["model.layers.1.mlp.experts.down_proj_bias"],
            primals_tile,
            weights["model.rotary_emb.freqs"],
            embedding_reshaped,
            weights["model.layers.0.input_layernorm.parametrizations.weight.original"],
            rsqrt_0_0,
            normed_0,
            weights["model.layers.0.self_attn.q_proj.parametrizations.weight.original"],
            weights["model.layers.0.self_attn.k_proj.parametrizations.weight.original"],
            weights["model.layers.0.self_attn.v_proj.parametrizations.weight.original"],
            query_rotated_0,
            key_repeated_0,
            value_repeated_0,
            typecast_71,
            attn_probs_sliced_0,
            attn_output_permuted_0,
            weights["model.layers.0.self_attn.o_proj.parametrizations.weight.original"],
            post_attn_hidden_0,
            weights["model.layers.0.post_attention_layernorm.parametrizations.weight.original"],
            rsqrt_1_0,
            post_normed_0,
            weights["model.layers.0.mlp.router.parametrizations.weight.original"],
            top_indices_mark_0,
            top_indices_0,
            weights["__consteval__.moe_weights_zeros"],
            weight_matrix_0,
            expert_input_0,
            expert_gate_up_0,
            expert_sigmoid_0,
            expert_gated_0,
            expert_down_0,
            layer_outputs[0],
            weights["model.layers.1.input_layernorm.parametrizations.weight.original"],
            rsqrt_0_1,
            normed_1,
            weights["model.layers.1.self_attn.q_proj.parametrizations.weight.original"],
            weights["model.layers.1.self_attn.k_proj.parametrizations.weight.original"],
            weights["model.layers.1.self_attn.v_proj.parametrizations.weight.original"],
            query_rotated_1,
            key_repeated_1,
            value_repeated_1,
            typecast_72,
            attn_probs_sliced_1,
            attn_output_permuted_1,
            weights["model.layers.1.self_attn.o_proj.parametrizations.weight.original"],
            post_attn_hidden_1,
            weights["model.layers.1.post_attention_layernorm.parametrizations.weight.original"],
            rsqrt_1_1,
            post_normed_1,
            weights["model.layers.1.mlp.router.parametrizations.weight.original"],
            top_indices_mark_1,
            top_indices_1,
            weight_matrix_1,
            expert_input_1,
            expert_gate_up_1,
            expert_sigmoid_1,
            expert_gated_1,
            expert_down_1,
            layer_outputs[1],
            weights["model.norm.parametrizations.weight.original"],
            rsqrt_4,
            final_normed,
            weights["lm_head.parametrizations.weight.original"],
            router_weights_1,
            attn_probs_1,
            router_weights_0,
            attn_probs_0,
        ]
