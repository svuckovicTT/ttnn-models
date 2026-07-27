# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import utils
import ttir_cpu
import torch

DRAM_MEMORY_CONFIG = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)


def _to_device(tensor, device):
    return ttnn.to_device(tensor, device=device, memory_config=DRAM_MEMORY_CONFIG)


def _to_device_tile_bf16(tensor, device):
    tensor = ttnn.to_layout(tensor, ttnn.Layout.TILE, None, memory_config=None)
    ttnn_prev = tensor
    tensor = _to_device(tensor, device)
    ttnn.deallocate(ttnn_prev, False)
    return tensor


def _transpose_to_bf8b(tensor, device, dtype=ttnn.DataType.BFLOAT8_B):
    def _permute_impl(arg_0):
        return ttir_cpu.permute(arg_0, [1, 0])

    t = ttnn.from_device(tensor)
    ttnn.deallocate(tensor, False)
    t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=None)
    prev = t
    t = ttnn.to_layout(t, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(prev, False)
    prev = t
    t = utils.execute_cpu_hoisted_function([t], _permute_impl, device)
    ttnn.deallocate(prev, False)
    prev = t
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(prev, False)
    prev = t
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(prev, False)
    prev = t
    t = _to_device(t, device)
    ttnn.deallocate(prev, False)
    prev = t
    t = ttnn.from_device(t)
    ttnn.deallocate(prev, False)
    prev = t
    t = ttnn.typecast(t, dtype, memory_config=None)
    ttnn.deallocate(prev, False)
    prev = t
    t = _to_device(t, device)
    ttnn.deallocate(prev, False)
    return t


def _typecast_to_bf8b(tensor, device):
    t = ttnn.from_device(tensor)
    ttnn.deallocate(tensor, False)
    prev = t
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(prev, False)
    prev = t
    t = _to_device(t, device)
    ttnn.deallocate(prev, False)
    return t


def _typecast_to_bf4b(tensor, device):
    t = ttnn.from_device(tensor)
    ttnn.deallocate(tensor, False)
    prev = t
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT4_B, memory_config=None)
    ttnn.deallocate(prev, False)
    prev = t
    t = _to_device(t, device)
    ttnn.deallocate(prev, False)
    return t


def _qkv_concat_to_bf8b(v_proj, k_proj, q_proj, device):
    def _permute_and_concat_impl(arg_v, arg_k, arg_q):
        pv = ttir_cpu.permute(arg_v, [1, 0])
        pk = ttir_cpu.permute(arg_k, [1, 0])
        pq = ttir_cpu.permute(arg_q, [1, 0])
        return ttir_cpu.concat([pv, pk, pq], dim=1)

    inputs = []
    for tensor in [v_proj, k_proj, q_proj]:
        t = ttnn.from_device(tensor)
        prev = t
        t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=None)
        ttnn.deallocate(prev, False)
        prev = t
        t = ttnn.to_layout(t, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
        ttnn.deallocate(prev, False)
        inputs.append(t)

    t = utils.execute_cpu_hoisted_function(inputs, _permute_and_concat_impl, device)
    for inp in inputs:
        ttnn.deallocate(inp, False)

    prev = t
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(prev, False)
    prev = t
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(prev, False)
    prev = t
    t = _to_device(t, device)
    ttnn.deallocate(prev, False)

    s0 = ttnn.slice(t, [0, 0], [2880, 128], [1, 1], memory_config=DRAM_MEMORY_CONFIG)
    s1 = ttnn.slice(t, [0, 128], [2880, 256], [1, 1], memory_config=DRAM_MEMORY_CONFIG)
    s2 = ttnn.slice(t, [0, 256], [2880, 1280], [1, 1], memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t, False)

    t = ttnn.concat([s2, s0, s1], 1, memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(s2, False)
    ttnn.deallocate(s1, False)
    ttnn.deallocate(s0, False)

    prev = t
    t = ttnn.from_device(t)
    ttnn.deallocate(prev, False)
    prev = t
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(prev, False)
    prev = t
    t = _to_device(t, device)
    ttnn.deallocate(prev, False)
    return t


def _reshape_expert_bias(tensor, shape, device):
    def _reshape_impl(arg_0):
        return ttir_cpu.reshape(arg_0, shape)

    t = ttnn.from_device(tensor)
    ttnn.deallocate(tensor, False)
    prev = t
    t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=None)
    ttnn.deallocate(prev, False)
    prev = t
    t = ttnn.to_layout(t, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(prev, False)
    prev = t
    t = utils.execute_cpu_hoisted_function([t], _reshape_impl, device)
    ttnn.deallocate(prev, False)
    prev = t
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(prev, False)
    t = _to_device_tile_bf16(t, device)
    return t


def _reshape_o_proj_bias(tensor, device):
    def _reshape_impl(arg_0):
        return ttir_cpu.reshape(arg_0, [1, 2880])

    t = ttnn.typecast(tensor, ttnn.DataType.FLOAT32, memory_config=None)
    prev = t
    t = utils.execute_cpu_hoisted_function([t], _reshape_impl, device)
    ttnn.deallocate(prev, False)
    prev = t
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(prev, False)
    t = _to_device_tile_bf16(t, device)
    return t


def _router_bias_passthrough(tensor, device):
    def _identity_impl(arg_0):
        return arg_0

    t = ttnn.typecast(tensor, ttnn.DataType.FLOAT32, memory_config=None)
    prev = t
    t = utils.execute_cpu_hoisted_function([t], _identity_impl, device)
    ttnn.deallocate(prev, False)
    t = _to_device_tile_bf16(t, device)
    return t


def _qkv_bias_concat(v_bias, k_bias, q_bias, device):
    def _concat_impl(arg_v, arg_k, arg_q):
        return ttir_cpu.concat([arg_v, arg_k, arg_q], dim=0)

    partitioned = []
    for bias in [v_bias, k_bias, q_bias]:
        t = _to_device(bias, device)
        prev = t
        t = ttnn.mesh_partition(
            input_tensor=t, dim=0, cluster_axis=1,
            memory_config=DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(prev, False)
        prev = t
        t = ttnn.from_device(t)
        ttnn.deallocate(prev, False)
        prev = t
        t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=None)
        ttnn.deallocate(prev, False)
        partitioned.append(t)

    t = utils.execute_cpu_hoisted_function(partitioned, _concat_impl, device)
    for p in partitioned:
        ttnn.deallocate(p, False)

    prev = t
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(prev, False)
    prev = t
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(prev, False)
    prev = t
    t = _to_device(t, device)
    ttnn.deallocate(prev, False)

    s0 = ttnn.slice(t, [0], [128], [1], memory_config=DRAM_MEMORY_CONFIG)
    s1 = ttnn.slice(t, [128], [256], [1], memory_config=DRAM_MEMORY_CONFIG)
    s2 = ttnn.slice(t, [256], [1280], [1], memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t, False)

    t = ttnn.concat([s2, s0, s1], 0, memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(s2, False)
    ttnn.deallocate(s1, False)
    ttnn.deallocate(s0, False)
    return t


def _sinks_broadcast(tensor, device):
    def _reshape_broadcast_impl(arg_0):
        t = ttir_cpu.reshape(arg_0, [1, 16, 1, 1])
        # SDPA applies its scale to both QK logits and attention sinks. GPT-OSS
        # sinks are already in post-scale logit space, so pre-divide by 0.125.
        inverse_attn_scale = ttir_cpu.full(
            shape=[1, 1, 1, 1], fill_value=8.0, dtype=torch.float32
        )
        return ttir_cpu.multiply(t, inverse_attn_scale)

    t = _to_device(tensor, device)
    prev = t
    t = ttnn.mesh_partition(
        input_tensor=t, dim=0, cluster_axis=1,
        memory_config=DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(prev, False)
    prev = t
    t = ttnn.from_device(t)
    ttnn.deallocate(prev, False)
    prev = t
    t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=None)
    ttnn.deallocate(prev, False)
    prev = t
    t = utils.execute_cpu_hoisted_function([t], _reshape_broadcast_impl, device)
    ttnn.deallocate(prev, False)
    prev = t
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(prev, False)
    t = _to_device_tile_bf16(t, device)
    return t


def _compute_rotary_embeddings(inv_freq, device):
    def _rotary_impl(arg_0):
        scale = ttir_cpu.full(shape=[1, 1, 1, 1], fill_value=1.34657359, dtype=torch.float32)
        positions = ttir_cpu.constant(
            shape=[1, 1, 17], dtype=torch.float32,
            data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        )
        reshaped = ttir_cpu.reshape(arg_0, [1, 32, 1])
        freqs = ttir_cpu.multiply(reshaped, positions)
        transposed = ttir_cpu.permute(freqs, [0, 2, 1])
        shaped = ttir_cpu.reshape(transposed, [1, 1, 17, 32])
        cos_val = ttir_cpu.multiply(ttir_cpu.cos(shaped), scale)
        sin_val = ttir_cpu.multiply(ttir_cpu.sin(shaped), scale)
        cos_repeated = ttir_cpu.concat([cos_val, cos_val], dim=3)
        sin_repeated = ttir_cpu.concat([sin_val, sin_val], dim=3)
        return freqs, cos_repeated, sin_repeated

    t = ttnn.typecast(inv_freq, ttnn.DataType.FLOAT32, memory_config=None)
    freqs, cos_val, sin_val = utils.execute_cpu_hoisted_function([t], _rotary_impl, device)
    ttnn.deallocate(t, False)

    cos_out = _to_device_tile_bf16(
        ttnn.typecast(cos_val, ttnn.DataType.BFLOAT16, memory_config=None), device
    )
    ttnn.deallocate(cos_val, False)

    sin_out = _to_device_tile_bf16(
        ttnn.typecast(sin_val, ttnn.DataType.BFLOAT16, memory_config=None), device
    )
    ttnn.deallocate(sin_val, False)

    freqs_out = _to_device_tile_bf16(freqs, device)

    return cos_out, sin_out, freqs_out


def _compute_causal_mask(device):
    def _causal_mask_impl():
        zeros = ttir_cpu.zeros(shape=[1, 1, 1, 1], dtype=torch.float32)
        rows = ttir_cpu.constant(
            shape=[1, 1, 17, 1], dtype=torch.int32,
            data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        )
        cols = ttir_cpu.constant(
            shape=[1, 1, 1, 17], dtype=torch.int32,
            data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        )
        neg_inf = ttir_cpu.full(
            shape=[1, 1, 1, 1], fill_value=-3.38953139e38, dtype=torch.float32,
        )
        mask = ttir_cpu.ge(rows, cols)
        return ttir_cpu.where(mask, zeros, neg_inf)

    t = utils.execute_cpu_hoisted_function([], _causal_mask_impl, device)
    prev = t
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(prev, False)
    t = _to_device_tile_bf16(t, device)
    return t


def _compute_scatter_index(device):
    def _scatter_index_impl():
        full_32 = ttir_cpu.full(shape=[1, 1, 1], fill_value=32, dtype=torch.int32)
        arange_17 = ttir_cpu.arange(0, 17, 1, arange_dimension=0, shape=[17], dtype=torch.int32)
        reshaped = ttir_cpu.reshape(arange_17, [17, 1, 1])
        multiplied = ttir_cpu.multiply(reshaped, full_32)
        return ttir_cpu.broadcast(multiplied, [17, 4, 1])

    t = utils.execute_cpu_hoisted_function([], _scatter_index_impl, device)
    t = _to_device_tile_bf16(t, device)
    return t


def run_consteval(weights, device):
    weights["model.embed_tokens.weight"] = _to_device(
        weights["model.embed_tokens.parametrizations.weight.original"], device
    )

    for i in range(2):
        prefix = f"model.layers.{i}"

        weights[f"{prefix}.self_attn.qkv_proj.weight"] = _qkv_concat_to_bf8b(
            weights[f"{prefix}.self_attn.v_proj.parametrizations.weight.original"],
            weights[f"{prefix}.self_attn.k_proj.parametrizations.weight.original"],
            weights[f"{prefix}.self_attn.q_proj.parametrizations.weight.original"],
            device,
        )

        weights[f"{prefix}.self_attn.qkv_proj.bias"] = _qkv_bias_concat(
            weights[f"{prefix}.self_attn.v_proj.bias"],
            weights[f"{prefix}.self_attn.k_proj.bias"],
            weights[f"{prefix}.self_attn.q_proj.bias"],
            device,
        )

        weights[f"{prefix}.self_attn.o_proj.parametrizations.weight.original"] = (
            _transpose_to_bf8b(
                weights[f"{prefix}.self_attn.o_proj.parametrizations.weight.original"],
                device,
            )
        )

        weights[f"{prefix}.self_attn.o_proj.bias"] = _reshape_o_proj_bias(
            weights[f"{prefix}.self_attn.o_proj.bias"], device
        )

        weights[f"{prefix}.self_attn.sinks"] = _sinks_broadcast(
            weights[f"{prefix}.self_attn.sinks"], device
        )

        weights[f"{prefix}.mlp.router.parametrizations.weight.original"] = (
            _transpose_to_bf8b(
                weights[f"{prefix}.mlp.router.parametrizations.weight.original"],
                device,
            )
        )

        weights[f"{prefix}.mlp.router.bias"] = _router_bias_passthrough(
            weights[f"{prefix}.mlp.router.bias"], device
        )

        weights[f"{prefix}.mlp.experts.gate_up_proj"] = _typecast_to_bf4b(
            weights[f"{prefix}.mlp.experts.gate_up_proj"], device
        )

        weights[f"{prefix}.mlp.experts.gate_up_proj_bias"] = _reshape_expert_bias(
            weights[f"{prefix}.mlp.experts.gate_up_proj_bias"],
            [32, 1, 5760], device,
        )

        weights[f"{prefix}.mlp.experts.down_proj"] = _typecast_to_bf8b(
            weights[f"{prefix}.mlp.experts.down_proj"], device
        )

        weights[f"{prefix}.mlp.experts.down_proj_bias"] = _reshape_expert_bias(
            weights[f"{prefix}.mlp.experts.down_proj_bias"],
            [32, 1, 2880], device,
        )

    weights["lm_head.parametrizations.weight.original"] = _transpose_to_bf8b(
        weights["lm_head.parametrizations.weight.original"], device
    )

    cos, sin, freqs = _compute_rotary_embeddings(
        weights["model.rotary_emb.inv_freq"], device
    )
    weights["model.rotary_emb.cos"] = cos
    weights["model.rotary_emb.sin"] = sin
    weights["model.rotary_emb.freqs"] = freqs

    weights["__consteval__.moe_scatter_zeros"] = ttnn.zeros(
        shape=ttnn.Shape([544]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=device,
        memory_config=DRAM_MEMORY_CONFIG,
    )

    weights["__consteval__.causal_mask"] = _compute_causal_mask(device)

    weights["__consteval__.ones_scalar"] = ttnn.ones(
        shape=ttnn.Shape([1, 1, 1]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MEMORY_CONFIG,
    )

    weights["__consteval__.scatter_index"] = _compute_scatter_index(device)

    weights["__consteval__.attn_scale"] = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=0.125,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MEMORY_CONFIG,
    )

    weights["__consteval__.rms_norm_eps"] = ttnn.full(
        shape=ttnn.Shape([1, 1, 1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MEMORY_CONFIG,
    )

    weights["__consteval__.moe_weights_zeros"] = ttnn.zeros(
        shape=ttnn.Shape([17, 32]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MEMORY_CONFIG,
    )

    weights["__consteval__.sigmoid_scale"] = ttnn.full(
        shape=ttnn.Shape([1, 1, 1]),
        fill_value=1.703125,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MEMORY_CONFIG,
    )

    return weights
