# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import ttir_cpu

DRAM_MEMORY_CONFIG = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)


def _transpose_weight(tensor):
    t = ttnn.to_torch(ttnn.typecast(tensor, ttnn.DataType.FLOAT32, memory_config=None))
    t = ttir_cpu.permute(t, [1, 0])
    return ttnn.from_torch(t)


def _transpose_and_cast(tensor, target_dtype, device):
    transposed = _transpose_weight(tensor)
    bf16 = ttnn.typecast(transposed, ttnn.DataType.BFLOAT16, memory_config=None)
    tiled = ttnn.to_layout(bf16, ttnn.Layout.TILE, None, memory_config=None)
    on_device = ttnn.to_device(tiled, device=device, memory_config=DRAM_MEMORY_CONFIG)
    on_host = ttnn.from_device(on_device)
    casted = ttnn.typecast(on_host, target_dtype, memory_config=None)
    return ttnn.to_device(casted, device=device, memory_config=DRAM_MEMORY_CONFIG)


def _transpose_concat_kvq_and_cast(k_weight, v_weight, q_weight, device):
    k_t = ttir_cpu.permute(ttnn.to_torch(ttnn.typecast(q_weight, ttnn.DataType.FLOAT32, memory_config=None)), [1, 0])
    v_t = ttir_cpu.permute(ttnn.to_torch(ttnn.typecast(v_weight, ttnn.DataType.FLOAT32, memory_config=None)), [1, 0])
    q_t = ttir_cpu.permute(ttnn.to_torch(ttnn.typecast(k_weight, ttnn.DataType.FLOAT32, memory_config=None)), [1, 0])
    concatenated = ttnn.from_torch(ttir_cpu.concat([k_t, v_t, q_t], dim=1))
    bf16 = ttnn.typecast(concatenated, ttnn.DataType.BFLOAT16, memory_config=None)
    tiled = ttnn.to_layout(bf16, ttnn.Layout.TILE, None, memory_config=None)
    on_device = ttnn.to_device(tiled, device=device, memory_config=DRAM_MEMORY_CONFIG)
    on_host = ttnn.from_device(on_device)
    casted = ttnn.typecast(on_host, ttnn.DataType.BFLOAT8_B, memory_config=None)
    return ttnn.to_device(casted, device=device, memory_config=DRAM_MEMORY_CONFIG)


def _transpose_concat_kvq_slice_reorder_and_cast(k_weight, v_weight, q_weight, device):
    k_t = ttir_cpu.permute(ttnn.to_torch(ttnn.typecast(q_weight, ttnn.DataType.FLOAT32, memory_config=None)), [1, 0])
    v_t = ttir_cpu.permute(ttnn.to_torch(ttnn.typecast(v_weight, ttnn.DataType.FLOAT32, memory_config=None)), [1, 0])
    q_t = ttir_cpu.permute(ttnn.to_torch(ttnn.typecast(k_weight, ttnn.DataType.FLOAT32, memory_config=None)), [1, 0])
    concatenated = ttnn.from_torch(ttir_cpu.concat([k_t, v_t, q_t], dim=1))
    bf16 = ttnn.typecast(concatenated, ttnn.DataType.BFLOAT16, memory_config=None)
    tiled = ttnn.to_layout(bf16, ttnn.Layout.TILE, None, memory_config=None)
    on_device = ttnn.to_device(tiled, device=device, memory_config=DRAM_MEMORY_CONFIG)
    s0 = ttnn.slice(on_device, [0, 0], [4096, 4096], [1, 1], memory_config=DRAM_MEMORY_CONFIG)
    s1 = ttnn.slice(on_device, [0, 4096], [4096, 5120], [1, 1], memory_config=DRAM_MEMORY_CONFIG)
    s2 = ttnn.slice(on_device, [0, 5120], [4096, 6144], [1, 1], memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(on_device, False)
    reordered = ttnn.concat([s0, s2, s1], 1, memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(s0, False)
    ttnn.deallocate(s1, False)
    ttnn.deallocate(s2, False)
    on_host = ttnn.from_device(reordered)
    ttnn.deallocate(reordered, False)
    casted = ttnn.typecast(on_host, ttnn.DataType.BFLOAT8_B, memory_config=None)
    return ttnn.to_device(casted, device=device, memory_config=DRAM_MEMORY_CONFIG)


def _reshape_inv_freq(tensor, device):
    t = ttnn.to_torch(ttnn.typecast(tensor, ttnn.DataType.FLOAT32, memory_config=None))
    reshaped = ttnn.from_torch(ttir_cpu.reshape(t, [1, 64, 1]))
    tiled = ttnn.to_layout(reshaped, ttnn.Layout.TILE, None, memory_config=None)
    return ttnn.to_device(tiled, device=device, memory_config=DRAM_MEMORY_CONFIG)


def run_consteval(weights, device):
    w = weights
    p = ".parametrizations.weight.original"

    # Embed tokens: move to device
    w[f"model.embed_tokens{p}"] = ttnn.to_device(
        w[f"model.embed_tokens{p}"], device=device, memory_config=DRAM_MEMORY_CONFIG
    )

    for i in range(32):
        # Self-attention projections: transpose -> BFLOAT8_B
        w[f"model.layers.{i}.self_attn.o_proj{p}"] = _transpose_and_cast(
            w[f"model.layers.{i}.self_attn.o_proj{p}"], ttnn.DataType.BFLOAT8_B, device
        )

        # QKV fused: transpose, concat [k,v,q] -> BFLOAT8_B
        if i == 31:
            w[f"model.layers.{i}.self_attn.qkv_proj{p}"] = (
                _transpose_concat_kvq_slice_reorder_and_cast(
                    w.pop(f"model.layers.{i}.self_attn.k_proj{p}"),
                    w.pop(f"model.layers.{i}.self_attn.v_proj{p}"),
                    w.pop(f"model.layers.{i}.self_attn.q_proj{p}"),
                    device,
                )
            )
        else:
            w[f"model.layers.{i}.self_attn.qkv_proj{p}"] = (
                _transpose_concat_kvq_and_cast(
                    w.pop(f"model.layers.{i}.self_attn.k_proj{p}"),
                    w.pop(f"model.layers.{i}.self_attn.v_proj{p}"),
                    w.pop(f"model.layers.{i}.self_attn.q_proj{p}"),
                    device,
                )
            )

        # MLP projections: transpose -> BFLOAT8_B for down_proj, BFLOAT4_B for gate/up
        w[f"model.layers.{i}.mlp.down_proj{p}"] = _transpose_and_cast(
            w[f"model.layers.{i}.mlp.down_proj{p}"], ttnn.DataType.BFLOAT8_B, device
        )
        w[f"model.layers.{i}.mlp.gate_proj{p}"] = _transpose_and_cast(
            w[f"model.layers.{i}.mlp.gate_proj{p}"], ttnn.DataType.BFLOAT4_B, device
        )
        w[f"model.layers.{i}.mlp.up_proj{p}"] = _transpose_and_cast(
            w[f"model.layers.{i}.mlp.up_proj{p}"], ttnn.DataType.BFLOAT4_B, device
        )

    # LM head: transpose -> BFLOAT8_B
    w[f"lm_head{p}"] = _transpose_and_cast(
        w[f"lm_head{p}"], ttnn.DataType.BFLOAT8_B, device
    )

    # Rotary embedding inv_freq: reshape to [1, 64, 1]
    w["model.rotary_emb.inv_freq"] = _reshape_inv_freq(
        w["model.rotary_emb.inv_freq"], device
    )

    # Constant tensors
    w["attn_mask_neg_inf"] = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=float("-inf"),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MEMORY_CONFIG,
    )

    w["attn_mask_zero"] = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=0.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MEMORY_CONFIG,
    )

    w["arange_128"] = ttnn.Tensor(
        list(range(128)),
        [1, 1, 1, 128],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        device,
        memory_config=DRAM_MEMORY_CONFIG,
    )

    return w
