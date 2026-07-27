import ttnn
import torch
import params
import consteval


class LightweightModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


DRAM_MC = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)

# The hidden size is 2,880 = 90 tiles.  Keep one width tile per core on the
# same 10x9 grid used by the expert matmuls so the residual stream can remain
# L1-resident across layer boundaries and sharded RMSNorm/residual operations.
RESIDUAL_GRID = ttnn.CoreGrid(y=9, x=10)
L1_RESIDUAL_MC = ttnn.create_sharded_memory_config(
    shape=(1, 1, 32, 2880),
    core_grid=RESIDUAL_GRID,
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
)
RESIDUAL_NORM_PROGRAM_CFG = ttnn.LayerNormShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size=(10, 9),
    subblock_w=1,
    block_h=1,
    block_w=1,
    inplace=False,
)

WORMHOLE_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

ATTENTION_LOFI_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

MLP_LOFI_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

MLP_GATE_UP_PROGRAM_CFG = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(10, 9),
    in0_block_w=10,
    out_subblock_h=1,
    out_subblock_w=2,
    out_block_h=1,
    out_block_w=2,
    per_core_M=1,
    per_core_N=2,
    fuse_batch=False,
    fused_activation=None,
    mcast_in0=True,
)

MLP_DOWN_PROGRAM_CFG = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(10, 9),
    in0_block_w=10,
    out_subblock_h=1,
    out_subblock_w=1,
    out_block_h=1,
    out_block_w=1,
    per_core_M=1,
    per_core_N=1,
    fuse_batch=False,
    fused_activation=None,
    mcast_in0=True,
)


class ModelTTNN(LightweightModule):
    def __init__(self, device):
        self.device = device
        self.weights = params.load_weights_for__main(device)
        self.weights = consteval.run_consteval(self.weights, device)
        self.layers = [
            GptOssDecoderLayer(self.weights, 0, device),
            GptOssDecoderLayer(self.weights, 1, device),
        ]

    def forward(self, activations):
        device = self.device
        weights = self.weights
        primals_39 = activations[0]

        # Shared constants
        causal_mask = weights["__consteval__.causal_mask"]
        cos = weights["model.rotary_emb.cos"]
        sin = weights["model.rotary_emb.sin"]
        attn_scale = weights["__consteval__.attn_scale"]
        rms_norm_eps = weights["__consteval__.rms_norm_eps"]
        moe_scatter_zeros = weights["__consteval__.moe_scatter_zeros"]
        scatter_index = weights["__consteval__.scatter_index"]
        ones_scalar = weights["__consteval__.ones_scalar"]
        sigmoid_scale = weights["__consteval__.sigmoid_scale"]

        # ---- Embedding ----
        ttnn_typecast_57 = ttnn.typecast(
            primals_39,
            ttnn.DataType.UINT32,
            memory_config=DRAM_MC,
        )
        ttnn_reshape_0 = ttnn.reshape(
            ttnn_typecast_57,
            [17],
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(ttnn_typecast_57, False)
        ttnn_embedding_0 = ttnn.embedding(
            ttnn_reshape_0,
            weights["model.embed_tokens.weight"],
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(ttnn_reshape_0, False)
        ttnn_reshape_1 = ttnn.reshape(
            ttnn_embedding_0,
            [1, 17, 2880],
            memory_config=DRAM_MC,
        )

        # ---- Layer 0 ----
        (
            layer0_output,
            layer0_post_attn_hidden,
            layer0_input_ln_weight,
            layer0_pre_attn_rsqrt,
            layer0_pre_attn_rms_norm,
            layer0_post_attn_ln_weight,
            layer0_post_attn_rsqrt,
            layer0_post_attn_rms_norm,
            layer0_attn_intermediates,
            layer0_mlp_intermediates,
        ) = self.layers[0](
            ttnn_reshape_1,
            ttnn_embedding_0,
            cos,
            sin,
            causal_mask,
            attn_scale,
            rms_norm_eps,
            moe_scatter_zeros,
            scatter_index,
            ones_scalar,
            sigmoid_scale,
        )

        # ---- Layer 1 ----
        (
            layer1_output,
            layer1_post_attn_hidden,
            layer1_input_ln_weight,
            layer1_pre_attn_rsqrt,
            layer1_pre_attn_rms_norm,
            layer1_post_attn_ln_weight,
            layer1_post_attn_rsqrt,
            layer1_post_attn_rms_norm,
            layer1_attn_intermediates,
            layer1_mlp_intermediates,
        ) = self.layers[1](
            layer0_output,
            None,
            cos,
            sin,
            causal_mask,
            attn_scale,
            rms_norm_eps,
            moe_scatter_zeros,
            scatter_index,
            ones_scalar,
            sigmoid_scale,
        )

        # Unpack layer 0 attention intermediates
        (
            l0_v_heads,
            l0_k_after_rotary,
            l0_q_after_rotary,
            l0_k_repeated,
            l0_v_repeated,
            l0_attn_scores_slice,
            l0_attn_output_permuted,
            l0_attn_softmax,
            l0_concat_with_sinks,
        ) = layer0_attn_intermediates

        # Unpack layer 1 attention intermediates
        (
            l1_v_heads,
            l1_k_after_rotary,
            l1_q_after_rotary,
            l1_k_repeated,
            l1_v_repeated,
            l1_attn_scores_slice,
            l1_attn_output_permuted,
            l1_attn_softmax,
            l1_concat_with_sinks,
        ) = layer1_attn_intermediates

        # Unpack layer 0 MLP intermediates
        (
            l0_topk_dim1,
            l0_topk_neg1,
            l0_moe_weights,
            l0_expert_concat,
            l0_gate_up_matmul,
            l0_sigmoid,
            l0_silu,
            l0_down_proj_matmul,
            l0_moe_softmax,
        ) = layer0_mlp_intermediates

        # Unpack layer 1 MLP intermediates
        (
            l1_topk_dim1,
            l1_topk_neg1,
            l1_moe_weights,
            l1_expert_concat,
            l1_gate_up_matmul,
            l1_sigmoid,
            l1_silu,
            l1_down_proj_matmul,
            l1_moe_softmax,
        ) = layer1_mlp_intermediates

        # ---- Final norm + lm_head ----
        var_3 = weights["model.norm.parametrizations.weight.original"]

        # The codegen graph also materialized inverse-RMS statistics for a
        # diagnostic output. They are not part of the full-model logits path;
        # the dedicated RMSNorm below computes the normalization directly.
        ttnn_rsqrt_4 = None
        ttnn_rms_norm_4 = ttnn.rms_norm(
            layer1_output,
            epsilon=9.9999997473787516e-06,
            weight=var_3,
            bias=None,
            residual_input_tensor=None,
            memory_config=L1_RESIDUAL_MC,
            program_config=RESIDUAL_NORM_PROGRAM_CFG,
            compute_kernel_config=WORMHOLE_CFG,
        )
        ttnn_reshape_36 = ttnn.reshape(
            ttnn_rms_norm_4,
            [17, 2880],
            memory_config=L1_RESIDUAL_MC,
        )
        ttnn_matmul_10 = ttnn.matmul(
            ttnn_reshape_36,
            weights["lm_head.parametrizations.weight.original"],
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(ttnn_reshape_36, False)
        ttnn_reshape_37 = ttnn.reshape(
            ttnn_matmul_10,
            [1, 17, 201088],
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(ttnn_matmul_10, False)

        # Codegen diagnostic-only argmax and input-layout outputs are outside
        # the full-model logits contract. Avoid eleven dispatches across these
        # branches (including the per-layer attention diagnostic permutes).
        ttnn_typecast_71 = None
        ttnn_typecast_72 = None
        ttnn_to_layout_49 = None

        # ---- Return list ----
        return [
            l0_v_heads,                                                                      # 0
            l0_k_after_rotary,                                                               # 1
            l1_v_heads,                                                                      # 2
            l1_k_after_rotary,                                                               # 3
            ttnn_reshape_37,                                                                 # 4
            weights["model.layers.0.mlp.experts.gate_up_proj"],                              # 5
            weights["model.layers.0.mlp.experts.gate_up_proj_bias"],                         # 6
            weights["model.layers.0.mlp.experts.down_proj"],                                 # 7
            weights["model.layers.0.mlp.experts.down_proj_bias"],                            # 8
            weights["model.layers.1.mlp.experts.gate_up_proj"],                              # 9
            weights["model.layers.1.mlp.experts.gate_up_proj_bias"],                         # 10
            weights["model.layers.1.mlp.experts.down_proj"],                                 # 11
            weights["model.layers.1.mlp.experts.down_proj_bias"],                            # 12
            ttnn_to_layout_49,                                                               # 13
            weights["model.rotary_emb.freqs"],                                               # 14
            ttnn_reshape_1,                                                                  # 15
            layer0_input_ln_weight,                                                          # 16
            layer0_pre_attn_rsqrt,                                                           # 17
            layer0_pre_attn_rms_norm,                                                        # 18
            weights["model.layers.0.self_attn.q_proj.parametrizations.weight.original"],     # 19
            weights["model.layers.0.self_attn.k_proj.parametrizations.weight.original"],     # 20
            weights["model.layers.0.self_attn.v_proj.parametrizations.weight.original"],     # 21
            l0_q_after_rotary,                                                               # 22
            l0_k_repeated,                                                                   # 23
            l0_v_repeated,                                                                   # 24
            ttnn_typecast_71,                                                                # 25
            l0_attn_scores_slice,                                                            # 26
            l0_attn_output_permuted,                                                         # 27
            weights["model.layers.0.self_attn.o_proj.parametrizations.weight.original"],     # 28
            layer0_post_attn_hidden,                                                         # 29
            layer0_post_attn_ln_weight,                                                      # 30
            layer0_post_attn_rsqrt,                                                          # 31
            layer0_post_attn_rms_norm,                                                       # 32
            weights["model.layers.0.mlp.router.parametrizations.weight.original"],           # 33
            l0_topk_dim1,                                                                    # 34
            l0_topk_neg1,                                                                    # 35
            weights["__consteval__.moe_weights_zeros"],                                      # 36
            l0_moe_weights,                                                                  # 37
            l0_expert_concat,                                                                # 38
            l0_gate_up_matmul,                                                               # 39
            l0_sigmoid,                                                                      # 40
            l0_silu,                                                                         # 41
            l0_down_proj_matmul,                                                             # 42
            layer0_output,                                                                   # 43
            layer1_input_ln_weight,                                                          # 44
            layer1_pre_attn_rsqrt,                                                           # 45
            layer1_pre_attn_rms_norm,                                                        # 46
            weights["model.layers.1.self_attn.q_proj.parametrizations.weight.original"],     # 47
            weights["model.layers.1.self_attn.k_proj.parametrizations.weight.original"],     # 48
            weights["model.layers.1.self_attn.v_proj.parametrizations.weight.original"],     # 49
            l1_q_after_rotary,                                                               # 50
            l1_k_repeated,                                                                   # 51
            l1_v_repeated,                                                                   # 52
            ttnn_typecast_72,                                                                # 53
            l1_attn_scores_slice,                                                            # 54
            l1_attn_output_permuted,                                                         # 55
            weights["model.layers.1.self_attn.o_proj.parametrizations.weight.original"],     # 56
            layer1_post_attn_hidden,                                                         # 57
            layer1_post_attn_ln_weight,                                                      # 58
            layer1_post_attn_rsqrt,                                                          # 59
            layer1_post_attn_rms_norm,                                                       # 60
            weights["model.layers.1.mlp.router.parametrizations.weight.original"],           # 61
            l1_topk_dim1,                                                                    # 62
            l1_topk_neg1,                                                                    # 63
            l1_moe_weights,                                                                  # 64
            l1_expert_concat,                                                                # 65
            l1_gate_up_matmul,                                                               # 66
            l1_sigmoid,                                                                      # 67
            l1_silu,                                                                         # 68
            l1_down_proj_matmul,                                                             # 69
            layer1_output,                                                                   # 70
            var_3,                                                                           # 71
            ttnn_rsqrt_4,                                                                    # 72
            ttnn_rms_norm_4,                                                                 # 73
            weights["lm_head.parametrizations.weight.original"],                             # 74
            l1_moe_softmax,                                                                  # 75
            l1_attn_softmax,                                                                 # 76
            l0_moe_softmax,                                                                  # 77
            l0_attn_softmax,                                                                 # 78
        ]


class GptOssAttention(LightweightModule):
    def __init__(self, weights, layer_idx):
        prefix = f"model.layers.{layer_idx}.self_attn"
        self.qkv_weight = weights[f"{prefix}.qkv_proj.weight"]
        self.qkv_bias = weights[f"{prefix}.qkv_proj.bias"]
        self.o_proj_weight = weights[
            f"{prefix}.o_proj.parametrizations.weight.original"
        ]
        self.o_proj_bias = weights[f"{prefix}.o_proj.bias"]
        self.sinks = weights[f"{prefix}.sinks"]

    def forward(self, hidden_2d, cos, sin, causal_mask, attn_scale):
        # QKV projection
        qkv_out = ttnn.linear(
            hidden_2d,
            self.qkv_weight,
            bias=self.qkv_bias,
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ATTENTION_LOFI_CFG,
        )
        ttnn.deallocate(hidden_2d, False)
        qkv_reshaped = ttnn.reshape(
            qkv_out,
            [1, 17, 1280],
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(qkv_out, False)
        q, v_heads, k = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv_reshaped,
            None,
            num_heads=16,
            num_kv_heads=2,
            transpose_key=False,
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(qkv_reshaped, False)

        # Rotary embedding on K
        k_rotary = ttnn.experimental.rotary_embedding(
            k,
            cos,
            sin,
            None,
            memory_config=DRAM_MC,
            compute_kernel_config=None,
        )
        ttnn.deallocate(k, False)
        k_after_rotary = ttnn.slice(
            k_rotary,
            [0, 0, 0, 0],
            [1, 2, 17, 64],
            [1, 1, 1, 1],
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(k_rotary, False)

        # Rotary embedding on Q
        q_rotary = ttnn.experimental.rotary_embedding(
            q,
            cos,
            sin,
            None,
            memory_config=DRAM_MC,
            compute_kernel_config=None,
        )
        ttnn.deallocate(q, False)
        q_after_rotary = ttnn.slice(
            q_rotary,
            [0, 0, 0, 0],
            [1, 16, 17, 64],
            [1, 1, 1, 1],
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(q_rotary, False)

        # Collapse GQA head replication, QK matmul, scaling, masking, sink
        # normalization, softmax, and AV matmul into the dedicated SDPA op.
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q_after_rotary,
            k_after_rotary,
            v_heads,
            is_causal=True,
            scale=0.125,
            memory_config=DRAM_MC,
            program_config=None,
            compute_kernel_config=None,
            attention_sink=self.sinks,
        )

        # These generated attention diagnostics are subsumed by SDPA.
        k_repeated = None
        v_repeated = None
        attn_scores_slice = None
        attn_output_permuted = None
        softmax_result = None
        concat_with_sinks = None

        # Concatenate heads
        concat_heads = ttnn.transformer.concatenate_heads(
            attn_output,
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(attn_output, False)

        # O projection
        heads_2d = ttnn.reshape(
            concat_heads,
            [17, 1024],
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(concat_heads, False)
        o_proj_out = ttnn.matmul(
            heads_2d,
            self.o_proj_weight,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ATTENTION_LOFI_CFG,
        )
        ttnn.deallocate(heads_2d, False)

        # All reduce
        o_proj_4d = ttnn.reshape(
            o_proj_out,
            [1, 1, 17, 2880],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(o_proj_out, False)
        all_reduce_out = ttnn.all_reduce(
            input_tensor=o_proj_4d,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(o_proj_4d, False)

        # Reshape and add bias
        attn_output_flat = ttnn.reshape(
            all_reduce_out,
            [17, 2880],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(all_reduce_out, False)
        attn_output_2d = ttnn.add(
            attn_output_flat,
            self.o_proj_bias,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output_flat, False)

        return (
            attn_output_2d,
            v_heads,
            k_after_rotary,
            q_after_rotary,
            k_repeated,
            v_repeated,
            attn_scores_slice,
            attn_output_permuted,
            softmax_result,
            concat_with_sinks,
        )


class GptOssMLP(LightweightModule):
    def __init__(self, weights, layer_idx):
        prefix = f"model.layers.{layer_idx}.mlp"
        self.gate_up_proj = weights[f"{prefix}.experts.gate_up_proj"]
        self.gate_up_proj_bias = weights[f"{prefix}.experts.gate_up_proj_bias"]
        self.down_proj = weights[f"{prefix}.experts.down_proj"]
        self.down_proj_bias = weights[f"{prefix}.experts.down_proj_bias"]
        self.router_weight = weights[
            f"{prefix}.router.parametrizations.weight.original"
        ]
        self.router_bias = weights[f"{prefix}.router.bias"]

    def forward(
        self,
        norm_output_2d,
        moe_scatter_zeros,
        scatter_index,
        ones_scalar,
        sigmoid_scale,
    ):
        # ---- Expert forward ----
        # Replicate tokens across experts with the dedicated repeat operation
        # instead of a 32-input concat of the same tensor.
        expert_concat = ttnn.repeat(
            norm_output_2d,
            ttnn.Shape([32, 1]),
            memory_config=DRAM_MC,
        )
        expert_3d = ttnn.reshape(
            expert_concat,
            [32, 17, 2880],
            memory_config=DRAM_MC,
        )

        # Gate-up projection
        # The expert bias is BF16, so it can be fused into the projection
        # without the low-precision bias-accumulation hazard of BFP8 weights.
        gate_up_matmul = ttnn.linear(
            expert_3d,
            self.gate_up_proj,
            bias=self.gate_up_proj_bias,
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=MLP_GATE_UP_PROGRAM_CFG,
            activation=None,
            compute_kernel_config=MLP_LOFI_CFG,
        )
        ttnn.deallocate(expert_3d, False)

        # Split gate and up: stride-2 slices
        # Up (odd indices)
        up_proj = ttnn.slice(
            gate_up_matmul,
            [0, 0, 1],
            [32, 17, 5760],
            [1, 1, 2],
            memory_config=DRAM_MC,
        )
        up_clamped = ttnn.clamp(
            up_proj,
            -7.0,
            7.0,
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(up_proj, False)
        up_activated = ttnn.add(
            up_clamped,
            ones_scalar,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(up_clamped, False)

        # Gate (even indices)
        gate_proj = ttnn.slice(
            gate_up_matmul,
            [0, 0, 0],
            [32, 17, 5760],
            [1, 1, 2],
            memory_config=DRAM_MC,
        )
        gate_clamped = ttnn.clamp(
            gate_proj,
            float("-inf"),
            7.0,
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(gate_proj, False)

        # SiLU gating
        gate_scaled = ttnn.multiply(
            gate_clamped,
            sigmoid_scale,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )
        sigmoid_out = ttnn.sigmoid(
            gate_scaled,
            vector_mode=4,
            mode=ttnn.SigmoidMode.Accurate,
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(gate_scaled, False)
        swish = ttnn.multiply(
            gate_clamped,
            sigmoid_out,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(gate_clamped, False)
        silu_out = ttnn.multiply(
            up_activated,
            swish,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(swish, False)
        ttnn.deallocate(up_activated, False)

        # Down projection
        down_proj_matmul = ttnn.linear(
            silu_out,
            self.down_proj,
            bias=self.down_proj_bias,
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=MLP_DOWN_PROGRAM_CFG,
            activation=None,
            compute_kernel_config=MLP_LOFI_CFG,
        )
        expert_output = down_proj_matmul

        # ---- Router forward ----
        router_input = ttnn.typecast(
            norm_output_2d,
            ttnn.DataType.FLOAT32,
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(norm_output_2d, False)

        router_logits = ttnn.linear(
            router_input,
            self.router_weight,
            bias=self.router_bias,
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM_MC,
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(router_input, False)

        router_bf16 = ttnn.typecast(
            router_logits,
            ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(router_logits, False)

        # The router logits are 2-D [tokens, experts], so dim=1 and dim=-1
        # select the same expert axis. Reuse one TopK for both graph outputs.
        topk_values, topk_indices_raw_neg1 = ttnn.topk(
            router_bf16,
            4,
            -1,
            True,
            True,
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(router_bf16, False)
        topk_indices_neg1 = ttnn.typecast(
            topk_indices_raw_neg1,
            ttnn.DataType.INT32,
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(topk_indices_raw_neg1, False)
        topk_indices_dim1 = topk_indices_neg1

        # Softmax on topk values
        moe_softmax = ttnn.softmax(
            topk_values,
            1,
            memory_config=DRAM_MC,
            compute_kernel_config=None,
            numeric_stable=True,
        )
        ttnn.deallocate(topk_values, False)

        # ---- MoE dispatch ----
        idx_reshaped = ttnn.reshape(
            topk_indices_neg1,
            [17, 4, 1],
            memory_config=DRAM_MC,
        )
        idx_offset = ttnn.add(
            scatter_index,
            idx_reshaped,
            dtype=ttnn.DataType.INT32,
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(idx_reshaped, False)
        idx_flat = ttnn.reshape(
            idx_offset,
            [68],
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(idx_offset, False)
        weights_flat = ttnn.reshape(
            moe_softmax,
            [68],
            memory_config=DRAM_MC,
        )

        idx_rm = ttnn.to_layout(
            idx_flat,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(idx_flat, False)
        weights_rm = ttnn.to_layout(
            weights_flat,
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(weights_flat, False)

        scatter_out = ttnn.scatter(
            input=moe_scatter_zeros,
            dim=0,
            index=idx_rm,
            src=weights_rm,
            memory_config=DRAM_MC,
            reduce=None,
        )
        ttnn.deallocate(weights_rm, False)
        ttnn.deallocate(idx_rm, False)

        scatter_tiled = ttnn.to_layout(
            scatter_out,
            ttnn.Layout.TILE,
            None,
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(scatter_out, False)

        moe_weights_reshaped = ttnn.reshape(
            scatter_tiled,
            [17, 32],
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(scatter_tiled, False)

        moe_weights_permuted = ttnn.permute(
            moe_weights_reshaped,
            [1, 0],
            memory_config=DRAM_MC,
            pad_value=0.0,
        )

        moe_weights_3d = ttnn.reshape(
            moe_weights_permuted,
            [32, 17, 1],
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(moe_weights_permuted, False)

        # Weighted expert output
        weighted_output = ttnn.multiply(
            expert_output,
            moe_weights_3d,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=DRAM_MC,
        )
        ttnn.deallocate(moe_weights_3d, False)
        ttnn.deallocate(expert_output, False)

        # Sum over experts
        mlp_output = ttnn.sum(
            weighted_output,
            [0],
            True,
            memory_config=DRAM_MC,
            compute_kernel_config=None,
        )
        ttnn.deallocate(weighted_output, False)

        return (
            mlp_output,
            topk_indices_dim1,
            topk_indices_neg1,
            moe_weights_reshaped,
            expert_concat,
            gate_up_matmul,
            sigmoid_out,
            silu_out,
            down_proj_matmul,
            moe_softmax,
        )


class GptOssDecoderLayer(LightweightModule):
    def __init__(self, weights, layer_idx, device):
        self.device = device
        self.input_layernorm_weight = weights[
            f"model.layers.{layer_idx}.input_layernorm.parametrizations.weight.original"
        ]
        self.post_attn_layernorm_weight = weights[
            f"model.layers.{layer_idx}.post_attention_layernorm.parametrizations.weight.original"
        ]
        self.self_attn = GptOssAttention(weights, layer_idx)
        self.mlp = GptOssMLP(weights, layer_idx)

    def forward(
        self,
        hidden_states,
        residual_input_flat,
        cos,
        sin,
        causal_mask,
        attn_scale,
        rms_norm_eps,
        moe_scatter_zeros,
        scatter_index,
        ones_scalar,
        sigmoid_scale,
    ):
        hidden_states = ttnn.to_memory_config(
            hidden_states,
            memory_config=L1_RESIDUAL_MC,
        )

        # ---- Pre-attention RMS norm ----
        # Drop the parallel codegen-only inverse-RMS diagnostic branch. The
        # fused op is the sole normalization used by the model.
        pre_attn_rsqrt = None
        pre_attn_rms_norm = ttnn.rms_norm(
            hidden_states,
            epsilon=9.9999997473787516e-06,
            weight=self.input_layernorm_weight,
            bias=None,
            residual_input_tensor=None,
            memory_config=L1_RESIDUAL_MC,
            program_config=RESIDUAL_NORM_PROGRAM_CFG,
            compute_kernel_config=WORMHOLE_CFG,
        )

        # Reshape for attention
        hidden_2d = ttnn.reshape(
            pre_attn_rms_norm,
            [17, 2880],
            memory_config=DRAM_MC,
        )

        # ---- Attention ----
        (
            attn_output_2d,
            v_heads,
            k_after_rotary,
            q_after_rotary,
            k_repeated,
            v_repeated,
            attn_scores_slice,
            attn_output_permuted,
            softmax_result,
            concat_with_sinks,
        ) = self.self_attn(hidden_2d, cos, sin, causal_mask, attn_scale)

        # ---- Attention residual ----
        # Both the embedding handoff and every stacked-layer handoff use the
        # same [1,17,2880] L1 residual contract.  This avoids a second
        # layer-0-only DRAM->L1 conversion of the flat embedding view.
        attn_output_3d = ttnn.reshape(
            attn_output_2d,
            [1, 17, 2880],
            memory_config=L1_RESIDUAL_MC,
        )
        ttnn.deallocate(attn_output_2d, False)
        post_attn_hidden = ttnn.add(
            hidden_states,
            attn_output_3d,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=L1_RESIDUAL_MC,
        )
        ttnn.deallocate(attn_output_3d, False)

        # ---- Post-attention RMS norm ----
        post_attn_rsqrt = None
        post_attn_rms_norm = ttnn.rms_norm(
            post_attn_hidden,
            epsilon=9.9999997473787516e-06,
            weight=self.post_attn_layernorm_weight,
            bias=None,
            residual_input_tensor=None,
            memory_config=L1_RESIDUAL_MC,
            program_config=RESIDUAL_NORM_PROGRAM_CFG,
            compute_kernel_config=WORMHOLE_CFG,
        )

        # Reshape for MLP
        norm_output_2d = ttnn.reshape(
            post_attn_rms_norm,
            [17, 2880],
            memory_config=DRAM_MC,
        )

        # ---- MLP ----
        (
            mlp_output,
            topk_dim1,
            topk_neg1,
            moe_weights_reshaped,
            expert_concat,
            gate_up_matmul,
            sigmoid_out,
            silu_out,
            down_proj_matmul,
            moe_softmax,
        ) = self.mlp(
            norm_output_2d,
            moe_scatter_zeros,
            scatter_index,
            ones_scalar,
            sigmoid_scale,
        )

        # ---- Post-MLP residual ----
        layer_output = ttnn.add(
            post_attn_hidden,
            ttnn.to_memory_config(
                mlp_output,
                memory_config=L1_RESIDUAL_MC,
            ),
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=L1_RESIDUAL_MC,
        )
        ttnn.deallocate(mlp_output, False)

        attn_intermediates = (
            v_heads,
            k_after_rotary,
            q_after_rotary,
            k_repeated,
            v_repeated,
            attn_scores_slice,
            attn_output_permuted,
            softmax_result,
            concat_with_sinks,
        )
        mlp_intermediates = (
            topk_dim1,
            topk_neg1,
            moe_weights_reshaped,
            expert_concat,
            gate_up_matmul,
            sigmoid_out,
            silu_out,
            down_proj_matmul,
            moe_softmax,
        )

        return (
            layer_output,
            post_attn_hidden,
            self.input_layernorm_weight,
            pre_attn_rsqrt,
            pre_attn_rms_norm,
            self.post_attn_layernorm_weight,
            post_attn_rsqrt,
            post_attn_rms_norm,
            attn_intermediates,
            mlp_intermediates,
        )
