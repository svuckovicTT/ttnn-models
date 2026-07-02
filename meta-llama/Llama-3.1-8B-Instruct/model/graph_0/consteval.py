# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import ttir_cpu

DRAM_MEMCFG = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)


def _transpose_weight(tensor):
    t = ttnn.to_torch(tensor)
    t = ttir_cpu.permute(t, [1, 0])
    return ttnn.from_torch(t)


def _transpose_concat_qkv(q_tensor, k_tensor, v_tensor):
    q = ttir_cpu.permute(ttnn.to_torch(q_tensor), [1, 0])
    k = ttir_cpu.permute(ttnn.to_torch(k_tensor), [1, 0])
    v = ttir_cpu.permute(ttnn.to_torch(v_tensor), [1, 0])
    concatenated = ttir_cpu.concat([q, k, v], dim=1)
    return ttnn.from_torch(concatenated)


def _to_device_bf8b(device, tensor):
    t = ttnn.typecast(tensor, ttnn.DataType.FLOAT32, memory_config=None)
    t_transposed = _transpose_weight(t)
    ttnn.deallocate(t, False)
    t = ttnn.typecast(t_transposed, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(t_transposed, False)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t_tiled = t
    t = ttnn.to_device(t_tiled, device=device, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(t_tiled, False)
    t_dev = t
    t = ttnn.from_device(t_dev)
    ttnn.deallocate(t_dev, False)
    t_host = t
    t = ttnn.typecast(t_host, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(t_host, False)
    t_cast = t
    t = ttnn.to_device(t_cast, device=device, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(t_cast, False)
    return t


def _to_device_bf4b(device, tensor):
    t = ttnn.typecast(tensor, ttnn.DataType.FLOAT32, memory_config=None)
    t_transposed = _transpose_weight(t)
    ttnn.deallocate(t, False)
    t = ttnn.typecast(t_transposed, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(t_transposed, False)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t_tiled = t
    t = ttnn.to_device(t_tiled, device=device, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(t_tiled, False)
    t_dev = t
    t = ttnn.from_device(t_dev)
    ttnn.deallocate(t_dev, False)
    t_host = t
    t = ttnn.typecast(t_host, ttnn.DataType.BFLOAT4_B, memory_config=None)
    ttnn.deallocate(t_host, False)
    t_cast = t
    t = ttnn.to_device(t_cast, device=device, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(t_cast, False)
    return t


def _concat_qkv_to_device_bf8b(device, q_tensor, k_tensor, v_tensor):
    q = ttnn.typecast(q_tensor, ttnn.DataType.FLOAT32, memory_config=None)
    k = ttnn.typecast(k_tensor, ttnn.DataType.FLOAT32, memory_config=None)
    v = ttnn.typecast(v_tensor, ttnn.DataType.FLOAT32, memory_config=None)
    concatenated = _transpose_concat_qkv(q, k, v)
    ttnn.deallocate(v, False)
    ttnn.deallocate(k, False)
    ttnn.deallocate(q, False)
    t = ttnn.typecast(concatenated, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(concatenated, False)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t_tiled = t
    t = ttnn.to_device(t_tiled, device=device, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(t_tiled, False)
    t_dev = t
    t = ttnn.from_device(t_dev)
    ttnn.deallocate(t_dev, False)
    t_host = t
    t = ttnn.typecast(t_host, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(t_host, False)
    t_cast = t
    t = ttnn.to_device(t_cast, device=device, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(t_cast, False)
    return t


def _concat_qkv_reorder_to_device_bf8b(device, q_tensor, k_tensor, v_tensor):
    q = ttnn.typecast(q_tensor, ttnn.DataType.FLOAT32, memory_config=None)
    k = ttnn.typecast(k_tensor, ttnn.DataType.FLOAT32, memory_config=None)
    v = ttnn.typecast(v_tensor, ttnn.DataType.FLOAT32, memory_config=None)
    concatenated = _transpose_concat_qkv(q, k, v)
    ttnn.deallocate(v, False)
    ttnn.deallocate(k, False)
    ttnn.deallocate(q, False)
    t = ttnn.typecast(concatenated, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(concatenated, False)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t_tiled = t
    t = ttnn.to_device(t_tiled, device=device, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(t_tiled, False)
    q_slice = ttnn.slice(t, [0, 0], [4096, 4096], [1, 1], memory_config=DRAM_MEMCFG)
    k_slice = ttnn.slice(t, [0, 4096], [4096, 5120], [1, 1], memory_config=DRAM_MEMCFG)
    v_slice = ttnn.slice(t, [0, 5120], [4096, 6144], [1, 1], memory_config=DRAM_MEMCFG)
    ttnn.deallocate(t, False)
    t = ttnn.concat(
        [q_slice, v_slice, k_slice], 1, memory_config=DRAM_MEMCFG
    )
    ttnn.deallocate(v_slice, False)
    ttnn.deallocate(k_slice, False)
    ttnn.deallocate(q_slice, False)
    t_dev = t
    t = ttnn.from_device(t_dev)
    ttnn.deallocate(t_dev, False)
    t_host = t
    t = ttnn.typecast(t_host, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(t_host, False)
    t_cast = t
    t = ttnn.to_device(t_cast, device=device, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(t_cast, False)
    return t


def _reshape_inv_freq(device, tensor):
    t = ttnn.typecast(tensor, ttnn.DataType.FLOAT32, memory_config=None)
    t_torch = ttnn.to_torch(t)
    ttnn.deallocate(t, False)
    t_reshaped = ttir_cpu.reshape(t_torch, [1, 64, 1])
    t = ttnn.from_torch(t_reshaped)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t_tiled = t
    t = ttnn.to_device(t_tiled, device=device, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(t_tiled, False)
    return t


def run_consteval(weights, device):
    # Embed tokens: move to device
    weights["model.embed_tokens.parametrizations.weight.original"] = ttnn.to_device(
        weights["model.embed_tokens.parametrizations.weight.original"],
        device=device,
        memory_config=DRAM_MEMCFG,
    )

    # Per-layer weight transformations
    for i in range(32):
        prefix = f"model.layers.{i}"

        # self_attn projections
        o_key = f"{prefix}.self_attn.o_proj.parametrizations.weight.original"
        weights[o_key] = _to_device_bf8b(device, weights[o_key])

        q_key = f"{prefix}.self_attn.q_proj.parametrizations.weight.original"
        k_key = f"{prefix}.self_attn.k_proj.parametrizations.weight.original"
        v_key = f"{prefix}.self_attn.v_proj.parametrizations.weight.original"
        if i == 31:
            qkv = _concat_qkv_reorder_to_device_bf8b(
                device, weights[q_key], weights[k_key], weights[v_key]
            )
        else:
            qkv = _concat_qkv_to_device_bf8b(
                device, weights[q_key], weights[k_key], weights[v_key]
            )
        weights[f"{prefix}.self_attn.qkv_proj.parametrizations.weight.original"] = qkv
        del weights[q_key]
        del weights[k_key]
        del weights[v_key]

        # MLP projections
        down_key = f"{prefix}.mlp.down_proj.parametrizations.weight.original"
        weights[down_key] = _to_device_bf8b(device, weights[down_key])

        gate_key = f"{prefix}.mlp.gate_proj.parametrizations.weight.original"
        weights[gate_key] = _to_device_bf4b(device, weights[gate_key])

        up_key = f"{prefix}.mlp.up_proj.parametrizations.weight.original"
        weights[up_key] = _to_device_bf4b(device, weights[up_key])

    # LM head
    lm_key = "lm_head.parametrizations.weight.original"
    weights[lm_key] = _to_device_bf8b(device, weights[lm_key])

    # Rotary embedding inverse frequencies
    inv_freq_key = "model.rotary_emb.inv_freq"
    weights[inv_freq_key] = _reshape_inv_freq(device, weights[inv_freq_key])

    # Constant tensors
    weights["attention.neg_inf_scalar"] = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=float("-inf"),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MEMCFG,
    )

    weights["attention.position_ids"] = ttnn.Tensor(
        list(range(128)),
        [1, 1, 1, 128],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        device,
        memory_config=DRAM_MEMCFG,
    )

    weights["attention.zero_scalar"] = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=0.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MEMCFG,
    )

    return weights
