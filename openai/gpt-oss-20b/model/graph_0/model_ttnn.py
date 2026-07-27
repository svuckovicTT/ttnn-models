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
        self.layers = [
            GptOssDecoderLayer(device, self.weights, i) for i in range(2)
        ]

    def forward(self, activations):
        device = self.device
        weights = self.weights
        primals_39 = activations[0]

        ttnn_typecast_57 = ttnn.typecast(
            primals_39, ttnn.DataType.UINT32, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn_reshape_0 = ttnn.reshape(
            ttnn_typecast_57, [17], memory_config=DRAM_MEMORY_CONFIG,
        )
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
        ttnn_reshape_1 = ttnn.reshape(
            ttnn_embedding_0, [1, 17, 2880], memory_config=DRAM_MEMORY_CONFIG,
        )

        hidden_states = ttnn_reshape_1
        residual = ttnn_embedding_0

        layer_outputs = []
        for layer in self.layers:
            hidden_states, residual, layer_out = layer(hidden_states, residual)
            layer_outputs.append(layer_out)

        ttnn_typecast_final = ttnn.typecast(
            hidden_states, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn_pow_final = ttnn.pow(
            ttnn_typecast_final, 2.0, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_final, False)
        ttnn_mean_final = ttnn.mean(
            ttnn_pow_final, [2], True,
            memory_config=DRAM_MEMORY_CONFIG, compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_pow_final, False)
        ttnn_add_final = ttnn.add(
            ttnn_mean_final,
            weights["__consteval__.rms_norm_eps"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_mean_final, False)
        ttnn_rsqrt_final = ttnn.rsqrt(
            ttnn_add_final, fast_and_approximate_mode=False,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_final, False)
        ttnn_rms_norm_final = ttnn.rms_norm(
            hidden_states,
            epsilon=9.9999997473787516e-06,
            weight=weights["model.norm.parametrizations.weight.original"],
            bias=None,
            residual_input_tensor=None,
            memory_config=DRAM_MEMORY_CONFIG,
            program_config=None,
            compute_kernel_config=WORMHOLE_COMPUTE_KERNEL_CONFIG,
        )
        ttnn_reshape_lm = ttnn.reshape(
            ttnn_rms_norm_final, [17, 2880], memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn_matmul_lm = ttnn.matmul(
            ttnn_reshape_lm,
            weights["lm_head.parametrizations.weight.original"],
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_reshape_lm, False)
        ttnn_reshape_37 = ttnn.reshape(
            ttnn_matmul_lm, [1, 17, 201088], memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_matmul_lm, False)

        ttnn_to_layout_49 = ttnn.to_layout(
            primals_39, ttnn.Layout.TILE, None, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(primals_39, False)

        l0 = layer_outputs[0]
        l1 = layer_outputs[1]

        return [
            l0["v_value"],
            l0["k_rotated"],
            l1["v_value"],
            l1["k_rotated"],
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
            weights["model.rotary_emb.freqs"],
            ttnn_reshape_1,
            l0["input_layernorm_weight"],
            l0["rsqrt"],
            l0["rms_norm_input"],
            weights["model.layers.0.self_attn.q_proj.parametrizations.weight.original"],
            weights["model.layers.0.self_attn.k_proj.parametrizations.weight.original"],
            weights["model.layers.0.self_attn.v_proj.parametrizations.weight.original"],
            l0["q_rotated"],
            l0["k_repeated"],
            l0["v_repeated"],
            l0["argmax"],
            l0["attn_weights_sliced"],
            l0["attn_output_permuted"],
            weights["model.layers.0.self_attn.o_proj.parametrizations.weight.original"],
            l0["post_attn_residual"],
            l0["post_attn_layernorm_weight"],
            l0["post_attn_rsqrt"],
            l0["post_attn_rms_norm"],
            weights["model.layers.0.mlp.router.parametrizations.weight.original"],
            l0["topk_indices_dim1"],
            l0["topk_indices_dimn1"],
            weights["__consteval__.moe_weights_zeros"],
            l0["moe_scatter_weights"],
            l0["expert_input_concat"],
            l0["expert_gate_up_out"],
            l0["expert_sigmoid"],
            l0["expert_gated_output"],
            l0["expert_down_proj_out"],
            l0["layer_output"],
            l1["input_layernorm_weight"],
            l1["rsqrt"],
            l1["rms_norm_input"],
            weights["model.layers.1.self_attn.q_proj.parametrizations.weight.original"],
            weights["model.layers.1.self_attn.k_proj.parametrizations.weight.original"],
            weights["model.layers.1.self_attn.v_proj.parametrizations.weight.original"],
            l1["q_rotated"],
            l1["k_repeated"],
            l1["v_repeated"],
            l1["argmax"],
            l1["attn_weights_sliced"],
            l1["attn_output_permuted"],
            weights["model.layers.1.self_attn.o_proj.parametrizations.weight.original"],
            l1["post_attn_residual"],
            l1["post_attn_layernorm_weight"],
            l1["post_attn_rsqrt"],
            l1["post_attn_rms_norm"],
            weights["model.layers.1.mlp.router.parametrizations.weight.original"],
            l1["topk_indices_dim1"],
            l1["topk_indices_dimn1"],
            l1["moe_scatter_weights"],
            l1["expert_input_concat"],
            l1["expert_gate_up_out"],
            l1["expert_sigmoid"],
            l1["expert_gated_output"],
            l1["expert_down_proj_out"],
            hidden_states,
            weights["model.norm.parametrizations.weight.original"],
            ttnn_rsqrt_final,
            ttnn_rms_norm_final,
            weights["lm_head.parametrizations.weight.original"],
            l1["softmax_router"],
            l1["softmax_attn"],
            l0["softmax_router"],
            l0["softmax_attn"],
        ]


class GptOssDecoderLayer(LightweightModule):
    def __init__(self, device, weights, layer_idx):
        self.device = device
        self.weights = weights
        self.layer_idx = layer_idx
        self.prefix = f"model.layers.{layer_idx}"
        self.self_attn = GptOssAttention(device, weights, layer_idx)
        self.mlp = GptOssMLP(device, weights, layer_idx)

    def forward(self, hidden_states, residual):
        device = self.device
        weights = self.weights
        prefix = self.prefix
        eps = weights["__consteval__.rms_norm_eps"]

        input_layernorm_weight = weights[f"{prefix}.input_layernorm.parametrizations.weight.original"]

        ttnn_typecast_in = ttnn.typecast(
            hidden_states, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn_pow = ttnn.pow(
            ttnn_typecast_in, 2.0, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_in, False)
        ttnn_mean = ttnn.mean(
            ttnn_pow, [2], True,
            memory_config=DRAM_MEMORY_CONFIG, compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_pow, False)
        ttnn_add_eps = ttnn.add(
            ttnn_mean, eps, dtype=ttnn.DataType.FLOAT32,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_mean, False)
        rsqrt = ttnn.rsqrt(
            ttnn_add_eps, fast_and_approximate_mode=False,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_eps, False)
        rms_norm_input = ttnn.rms_norm(
            hidden_states,
            epsilon=9.9999997473787516e-06,
            weight=input_layernorm_weight,
            bias=None,
            residual_input_tensor=None,
            memory_config=DRAM_MEMORY_CONFIG,
            program_config=None,
            compute_kernel_config=WORMHOLE_COMPUTE_KERNEL_CONFIG,
        )

        attn_input = ttnn.reshape(
            rms_norm_input, [17, 2880], memory_config=DRAM_MEMORY_CONFIG,
        )

        attn_out, attn_intermediates = self.self_attn(attn_input)

        ttnn_add_bias = ttnn.add(
            attn_out,
            weights[f"{prefix}.self_attn.o_proj.bias"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_out, False)
        ttnn_add_res = ttnn.add(
            residual, ttnn_add_bias,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_bias, False)
        ttnn.deallocate(residual, False)
        post_attn_residual = ttnn.reshape(
            ttnn_add_res, [1, 17, 2880], memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_res, False)

        post_attn_layernorm_weight = weights[f"{prefix}.post_attention_layernorm.parametrizations.weight.original"]

        ttnn_typecast_post = ttnn.typecast(
            post_attn_residual, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn_pow_post = ttnn.pow(
            ttnn_typecast_post, 2.0, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_post, False)
        ttnn_mean_post = ttnn.mean(
            ttnn_pow_post, [2], True,
            memory_config=DRAM_MEMORY_CONFIG, compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_pow_post, False)
        ttnn_add_eps_post = ttnn.add(
            ttnn_mean_post, eps, dtype=ttnn.DataType.FLOAT32,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_mean_post, False)
        post_attn_rsqrt = ttnn.rsqrt(
            ttnn_add_eps_post, fast_and_approximate_mode=False,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_eps_post, False)
        post_attn_rms_norm = ttnn.rms_norm(
            post_attn_residual,
            epsilon=9.9999997473787516e-06,
            weight=post_attn_layernorm_weight,
            bias=None,
            residual_input_tensor=None,
            memory_config=DRAM_MEMORY_CONFIG,
            program_config=None,
            compute_kernel_config=WORMHOLE_COMPUTE_KERNEL_CONFIG,
        )

        mlp_input = ttnn.reshape(
            post_attn_rms_norm, [17, 2880], memory_config=DRAM_MEMORY_CONFIG,
        )

        mlp_output, mlp_intermediates = self.mlp(mlp_input)

        layer_output = ttnn.add(
            post_attn_residual, mlp_output,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(mlp_output, False)

        layer_out = {
            "input_layernorm_weight": input_layernorm_weight,
            "rsqrt": rsqrt,
            "rms_norm_input": rms_norm_input,
            "q_rotated": attn_intermediates["q_rotated"],
            "k_rotated": attn_intermediates["k_rotated"],
            "v_value": attn_intermediates["v_value"],
            "k_repeated": attn_intermediates["k_repeated"],
            "v_repeated": attn_intermediates["v_repeated"],
            "argmax": attn_intermediates["argmax"],
            "attn_weights_sliced": attn_intermediates["attn_weights_sliced"],
            "attn_output_permuted": attn_intermediates["attn_output_permuted"],
            "softmax_attn": attn_intermediates["softmax_attn"],
            "post_attn_residual": post_attn_residual,
            "post_attn_layernorm_weight": post_attn_layernorm_weight,
            "post_attn_rsqrt": post_attn_rsqrt,
            "post_attn_rms_norm": post_attn_rms_norm,
            "topk_indices_dim1": mlp_intermediates["topk_indices_dim1"],
            "topk_indices_dimn1": mlp_intermediates["topk_indices_dimn1"],
            "softmax_router": mlp_intermediates["softmax_router"],
            "moe_scatter_weights": mlp_intermediates["moe_scatter_weights"],
            "expert_input_concat": mlp_intermediates["expert_input_concat"],
            "expert_gate_up_out": mlp_intermediates["expert_gate_up_out"],
            "expert_sigmoid": mlp_intermediates["expert_sigmoid"],
            "expert_gated_output": mlp_intermediates["expert_gated_output"],
            "expert_down_proj_out": mlp_intermediates["expert_down_proj_out"],
            "layer_output": layer_output,
        }

        return layer_output, post_attn_residual, layer_out


class GptOssAttention(LightweightModule):
    def __init__(self, device, weights, layer_idx):
        self.device = device
        self.weights = weights
        self.prefix = f"model.layers.{layer_idx}"

    def forward(self, attn_input):
        device = self.device
        weights = self.weights
        prefix = self.prefix

        cos = weights["model.rotary_emb.cos"]
        sin = weights["model.rotary_emb.sin"]
        causal_mask = weights["__consteval__.causal_mask"]
        attn_scale = weights["__consteval__.attn_scale"]

        ttnn_linear_qkv = ttnn.linear(
            attn_input,
            weights[f"{prefix}.self_attn.qkv_proj.weight"],
            bias=weights[f"{prefix}.self_attn.qkv_proj.bias"],
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(attn_input, False)
        ttnn_reshape_qkv = ttnn.reshape(
            ttnn_linear_qkv, [1, 17, 1280], memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_linear_qkv, False)
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn_reshape_qkv,
            None,
            num_heads=16,
            num_kv_heads=2,
            transpose_key=False,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_reshape_qkv, False)

        k_rotated_full = ttnn.experimental.rotary_embedding(
            key, cos, sin, None,
            memory_config=DRAM_MEMORY_CONFIG, compute_kernel_config=None,
        )
        ttnn.deallocate(key, False)
        k_rotated = ttnn.slice(
            k_rotated_full,
            [0, 0, 0, 0], [1, 2, 17, 64], [1, 1, 1, 1],
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(k_rotated_full, False)

        q_rotated_full = ttnn.experimental.rotary_embedding(
            query, cos, sin, None,
            memory_config=DRAM_MEMORY_CONFIG, compute_kernel_config=None,
        )
        ttnn.deallocate(query, False)
        q_rotated = ttnn.slice(
            q_rotated_full,
            [0, 0, 0, 0], [1, 16, 17, 64], [1, 1, 1, 1],
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_rotated_full, False)

        k_reshaped = ttnn.reshape(
            k_rotated, [1, 2, 1, 17, 64], memory_config=DRAM_MEMORY_CONFIG,
        )
        k_repeated = ttnn.repeat(
            k_reshaped, ttnn.Shape([1, 1, 8, 1, 1]),
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(k_reshaped, False)
        k_expanded = ttnn.reshape(
            k_repeated, [1, 16, 17, 64], memory_config=DRAM_MEMORY_CONFIG,
        )
        k_transposed = ttnn.permute(
            k_expanded, [0, 1, 3, 2],
            memory_config=DRAM_MEMORY_CONFIG, pad_value=0.0,
        )
        ttnn.deallocate(k_expanded, False)

        attn_scores = ttnn.matmul(
            q_rotated, k_transposed,
            transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        ttnn.deallocate(k_transposed, False)
        attn_scores_scaled = ttnn.multiply(
            attn_scores, attn_scale,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_scores, False)
        attn_scores_masked = ttnn.add(
            attn_scores_scaled, causal_mask,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_scores_scaled, False)

        attn_with_sinks = ttnn.concat(
            [attn_scores_masked, weights[f"{prefix}.self_attn.sinks"]],
            3, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_scores_masked, False)
        softmax_attn = ttnn.softmax(
            attn_with_sinks, 3,
            memory_config=DRAM_MEMORY_CONFIG,
            compute_kernel_config=None, numeric_stable=True,
        )
        attn_weights_sliced = ttnn.slice(
            softmax_attn,
            [0, 0, 0, 0], [1, 16, 17, 17], [1, 1, 1, 1],
            memory_config=DRAM_MEMORY_CONFIG,
        )

        v_reshaped = ttnn.reshape(
            value, [1, 2, 1, 17, 64], memory_config=DRAM_MEMORY_CONFIG,
        )
        v_repeated = ttnn.repeat(
            v_reshaped, ttnn.Shape([1, 1, 8, 1, 1]),
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(v_reshaped, False)
        v_expanded = ttnn.reshape(
            v_repeated, [1, 16, 17, 64], memory_config=DRAM_MEMORY_CONFIG,
        )

        attn_output = ttnn.matmul(
            attn_weights_sliced, v_expanded,
            transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        ttnn.deallocate(v_expanded, False)
        attn_output_permuted = ttnn.permute(
            attn_output, [0, 2, 1, 3],
            memory_config=DRAM_MEMORY_CONFIG, pad_value=0.0,
        )
        concat_heads = ttnn.transformer.concatenate_heads(
            attn_output, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output, False)
        concat_heads_reshaped = ttnn.reshape(
            concat_heads, [17, 1024], memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(concat_heads, False)

        o_proj = ttnn.matmul(
            concat_heads_reshaped,
            weights[f"{prefix}.self_attn.o_proj.parametrizations.weight.original"],
            transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        ttnn.deallocate(concat_heads_reshaped, False)
        o_proj_reshaped = ttnn.reshape(
            o_proj, [1, 1, 17, 2880], memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(o_proj, False)
        o_proj_reduced = ttnn.all_reduce(
            input_tensor=o_proj_reshaped,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=DRAM_MEMORY_CONFIG,
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(o_proj_reshaped, False)
        attn_out = ttnn.reshape(
            o_proj_reduced, [17, 2880], memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(o_proj_reduced, False)

        argmax_layout = ttnn.to_layout(
            attn_with_sinks, ttnn.Layout.ROW_MAJOR, None,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_with_sinks, False)
        argmax = ttnn.argmax(
            argmax_layout, 3, True,
            sub_core_grids=None, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(argmax_layout, False)
        argmax_tiled = ttnn.to_layout(
            argmax, ttnn.Layout.TILE, None, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(argmax, False)
        argmax_int32 = ttnn.typecast(
            argmax_tiled, ttnn.DataType.INT32, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(argmax_tiled, False)

        intermediates = {
            "q_rotated": q_rotated,
            "k_rotated": k_rotated,
            "v_value": value,
            "k_repeated": k_repeated,
            "v_repeated": v_repeated,
            "argmax": argmax_int32,
            "attn_weights_sliced": attn_weights_sliced,
            "attn_output_permuted": attn_output_permuted,
            "softmax_attn": softmax_attn,
        }

        return attn_out, intermediates


class GptOssMLP(LightweightModule):
    def __init__(self, device, weights, layer_idx):
        self.device = device
        self.weights = weights
        self.prefix = f"model.layers.{layer_idx}"
        self.router = GptOssTopKRouter(device, weights, layer_idx)
        self.experts = GptOssExperts(device, weights, layer_idx)

    def forward(self, mlp_input):
        device = self.device
        weights = self.weights
        prefix = self.prefix

        expert_output, expert_intermediates = self.experts(mlp_input)

        router_input = ttnn.typecast(
            mlp_input, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(mlp_input, False)

        router_output, router_intermediates = self.router(router_input)

        ttnn.deallocate(router_input, False)

        ttnn_multiply_weighted = ttnn.multiply(
            expert_output, router_output,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(router_output, False)
        ttnn.deallocate(expert_output, False)
        mlp_output = ttnn.sum(
            ttnn_multiply_weighted, [0], True,
            memory_config=DRAM_MEMORY_CONFIG, compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_multiply_weighted, False)

        intermediates = {
            "topk_indices_dim1": router_intermediates["topk_indices_dim1"],
            "topk_indices_dimn1": router_intermediates["topk_indices_dimn1"],
            "softmax_router": router_intermediates["softmax_router"],
            "moe_scatter_weights": router_intermediates["moe_scatter_weights"],
            "expert_input_concat": expert_intermediates["expert_input_concat"],
            "expert_gate_up_out": expert_intermediates["expert_gate_up_out"],
            "expert_sigmoid": expert_intermediates["expert_sigmoid"],
            "expert_gated_output": expert_intermediates["expert_gated_output"],
            "expert_down_proj_out": expert_intermediates["expert_down_proj_out"],
        }

        return mlp_output, intermediates


class GptOssTopKRouter(LightweightModule):
    def __init__(self, device, weights, layer_idx):
        self.device = device
        self.weights = weights
        self.prefix = f"model.layers.{layer_idx}"

    def forward(self, router_input):
        device = self.device
        weights = self.weights
        prefix = self.prefix

        scatter_index = weights["__consteval__.scatter_index"]
        moe_scatter_zeros = weights["__consteval__.moe_scatter_zeros"]

        ttnn_linear_router = ttnn.linear(
            router_input,
            weights[f"{prefix}.mlp.router.parametrizations.weight.original"],
            bias=weights[f"{prefix}.mlp.router.bias"],
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn_typecast_router = ttnn.typecast(
            ttnn_linear_router, ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_linear_router, False)

        v_topk_vals_dim1, v_topk_idx_dim1 = ttnn.topk(
            ttnn_typecast_router, 4, 1, True, True,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(v_topk_vals_dim1, False)
        topk_indices_dim1 = ttnn.typecast(
            v_topk_idx_dim1, ttnn.DataType.INT32,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(v_topk_idx_dim1, False)

        v_topk_vals, v_topk_idx = ttnn.topk(
            ttnn_typecast_router, 4, -1, True, True,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_typecast_router, False)
        topk_indices_dimn1 = ttnn.typecast(
            v_topk_idx, ttnn.DataType.INT32, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(v_topk_idx, False)
        softmax_router = ttnn.softmax(
            v_topk_vals, 1,
            memory_config=DRAM_MEMORY_CONFIG,
            compute_kernel_config=None, numeric_stable=True,
        )
        ttnn.deallocate(v_topk_vals, False)

        ttnn_reshape_idx = ttnn.reshape(
            topk_indices_dimn1, [17, 4, 1], memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn_add_idx = ttnn.add(
            scatter_index, ttnn_reshape_idx,
            dtype=ttnn.DataType.INT32, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_reshape_idx, False)
        ttnn_flat_idx = ttnn.reshape(
            ttnn_add_idx, [68], memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_idx, False)
        ttnn_flat_weights = ttnn.reshape(
            softmax_router, [68], memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn_idx_rm = ttnn.to_layout(
            ttnn_flat_idx, ttnn.Layout.ROW_MAJOR, None,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_flat_idx, False)
        ttnn_weights_rm = ttnn.to_layout(
            ttnn_flat_weights, ttnn.Layout.ROW_MAJOR, None,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_flat_weights, False)
        ttnn_scatter = ttnn.scatter(
            input=moe_scatter_zeros,
            dim=0,
            index=ttnn_idx_rm,
            src=ttnn_weights_rm,
            memory_config=DRAM_MEMORY_CONFIG,
            reduce=None,
        )
        ttnn.deallocate(ttnn_weights_rm, False)
        ttnn.deallocate(ttnn_idx_rm, False)
        ttnn_scatter_tiled = ttnn.to_layout(
            ttnn_scatter, ttnn.Layout.TILE, None, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_scatter, False)
        moe_scatter_weights = ttnn.reshape(
            ttnn_scatter_tiled, [17, 32], memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_scatter_tiled, False)
        ttnn_permuted = ttnn.permute(
            moe_scatter_weights, [1, 0],
            memory_config=DRAM_MEMORY_CONFIG, pad_value=0.0,
        )
        router_output = ttnn.reshape(
            ttnn_permuted, [32, 17, 1], memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_permuted, False)

        intermediates = {
            "topk_indices_dim1": topk_indices_dim1,
            "topk_indices_dimn1": topk_indices_dimn1,
            "softmax_router": softmax_router,
            "moe_scatter_weights": moe_scatter_weights,
        }

        return router_output, intermediates


class GptOssExperts(LightweightModule):
    def __init__(self, device, weights, layer_idx):
        self.device = device
        self.weights = weights
        self.prefix = f"model.layers.{layer_idx}"

    def forward(self, mlp_input):
        device = self.device
        weights = self.weights
        prefix = self.prefix
        ones_scalar = weights["__consteval__.ones_scalar"]
        sigmoid_scale = weights["__consteval__.sigmoid_scale"]

        expert_input_concat = ttnn.concat(
            [mlp_input] * 32, 0, memory_config=DRAM_MEMORY_CONFIG,
        )
        expert_input = ttnn.reshape(
            expert_input_concat, [32, 17, 2880], memory_config=DRAM_MEMORY_CONFIG,
        )

        expert_gate_up_out = ttnn.matmul(
            expert_input,
            weights[f"{prefix}.mlp.experts.gate_up_proj"],
            transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        ttnn.deallocate(expert_input, False)
        ttnn_add_bias = ttnn.add(
            expert_gate_up_out,
            weights[f"{prefix}.mlp.experts.gate_up_proj_bias"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MEMORY_CONFIG,
        )

        gate_input = ttnn.slice(
            ttnn_add_bias,
            [0, 0, 1], [32, 17, 5760], [1, 1, 2],
            memory_config=DRAM_MEMORY_CONFIG,
        )
        gate_clamped = ttnn.clamp(
            gate_input, -7.0, 7.0, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate_input, False)
        gate_value = ttnn.add(
            gate_clamped, ones_scalar,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate_clamped, False)

        up_input = ttnn.slice(
            ttnn_add_bias,
            [0, 0, 0], [32, 17, 5760], [1, 1, 2],
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttnn_add_bias, False)
        up_clamped = ttnn.clamp(
            up_input, float("-inf"), 7.0, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(up_input, False)
        up_scaled = ttnn.multiply(
            up_clamped, sigmoid_scale,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG,
        )
        expert_sigmoid = ttnn.sigmoid(
            up_scaled,
            vector_mode=4,
            mode=ttnn.SigmoidMode.Accurate,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(up_scaled, False)
        expert_gated_output = ttnn.multiply(
            up_clamped, expert_sigmoid,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(up_clamped, False)
        expert_activated = ttnn.multiply(
            gate_value, expert_gated_output,
            dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(expert_gated_output, False)
        ttnn.deallocate(gate_value, False)

        expert_down_proj_out = ttnn.matmul(
            expert_activated,
            weights[f"{prefix}.mlp.experts.down_proj"],
            transpose_a=False, transpose_b=False,
            memory_config=DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None, activation=None, compute_kernel_config=None,
        )
        expert_output = ttnn.add(
            expert_down_proj_out,
            weights[f"{prefix}.mlp.experts.down_proj_bias"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MEMORY_CONFIG,
        )

        intermediates = {
            "expert_input_concat": expert_input_concat,
            "expert_gate_up_out": expert_gate_up_out,
            "expert_sigmoid": expert_sigmoid,
            "expert_gated_output": expert_activated,
            "expert_down_proj_out": expert_down_proj_out,
        }

        return expert_output, intermediates
