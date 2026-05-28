# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Constant-evaluation pass: prepare model weights and runtime constants on device.

Run exactly once during :class:`model_ttnn.ModelTTNN` construction.  The pass
mutates the ``weights`` dictionary in place: per-layer projection weights are
typecast to BFLOAT8_B TILE on DRAM, Q/K/V projections are fused into a single
``self_attn.qkv_proj.weight`` per layer, and a handful of runtime constants
(attention-mask values, position-id seeds) are appended under descriptive keys.
"""

import ttnn
import ttir_cpu


_LAYER_INDICES = range(16)

_INTERLEAVED_DRAM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)


def _to_bfloat8_b_tile(weight, device):
    """Move a host BFLOAT16 row-major weight to BFLOAT8_B TILE on device DRAM."""
    on_device = ttnn.to_device(weight, device=device, memory_config=_INTERLEAVED_DRAM)
    tiled = ttnn.to_layout(
        on_device, ttnn.Layout.TILE, None, memory_config=_INTERLEAVED_DRAM
    )
    ttnn.deallocate(on_device, False)
    on_host = ttnn.from_device(tiled)
    ttnn.deallocate(tiled, False)
    bf8 = ttnn.typecast(on_host, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(on_host, False)
    result = ttnn.to_device(bf8, device=device, memory_config=_INTERLEAVED_DRAM)
    ttnn.deallocate(bf8, False)
    return result


def _fuse_qkv_to_bfloat8_b_tile(k_weight, v_weight, q_weight, device):
    """Fuse K/V/Q projection weights into a single BFLOAT8_B TILE weight on DRAM.

    Each weight is transposed (``permute([1, 0])``) and the three are
    concatenated along the output (column) dimension in ``[Q, K, V]`` order.
    """
    k_f32 = ttnn.typecast(k_weight, ttnn.DataType.FLOAT32, memory_config=None)
    v_f32 = ttnn.typecast(v_weight, ttnn.DataType.FLOAT32, memory_config=None)
    q_f32 = ttnn.typecast(q_weight, ttnn.DataType.FLOAT32, memory_config=None)

    q_t = ttir_cpu.permute(ttnn.to_torch(q_f32), [1, 0])
    k_t = ttir_cpu.permute(ttnn.to_torch(k_f32), [1, 0])
    v_t = ttir_cpu.permute(ttnn.to_torch(v_f32), [1, 0])
    fused = ttnn.from_torch(ttir_cpu.concat([q_t, k_t, v_t], dim=1))

    ttnn.deallocate(q_f32, False)
    ttnn.deallocate(v_f32, False)
    ttnn.deallocate(k_f32, False)

    fused_bf16 = ttnn.typecast(fused, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(fused, False)
    tiled = ttnn.to_layout(fused_bf16, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(fused_bf16, False)
    on_device = ttnn.to_device(tiled, device=device, memory_config=_INTERLEAVED_DRAM)
    ttnn.deallocate(tiled, False)
    on_host = ttnn.from_device(on_device)
    ttnn.deallocate(on_device, False)
    bf8 = ttnn.typecast(on_host, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(on_host, False)
    result = ttnn.to_device(bf8, device=device, memory_config=_INTERLEAVED_DRAM)
    ttnn.deallocate(bf8, False)
    return result


def _prepare_rotary_inv_freq(inv_freq, device):
    """Reshape rotary embedding inv_freq from [32] to [1, 32, 1] on device DRAM."""
    f32 = ttnn.typecast(inv_freq, ttnn.DataType.FLOAT32, memory_config=None)
    reshaped = ttnn.from_torch(ttir_cpu.reshape(ttnn.to_torch(f32), [1, 32, 1]))
    ttnn.deallocate(f32, False)
    tiled = ttnn.to_layout(reshaped, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(reshaped, False)
    on_device = ttnn.to_device(tiled, device=device, memory_config=_INTERLEAVED_DRAM)
    ttnn.deallocate(tiled, False)
    return on_device


def _embed_tokens_to_device(weight, device):
    """Move the (row-major BFLOAT16) embedding table to device DRAM."""
    return ttnn.to_device(weight, device=device, memory_config=_INTERLEAVED_DRAM)


def _make_int32_iota_tile(values, shape, device):
    """Create an INT32 TILE tensor on DRAM holding the given iota values."""
    return ttnn.Tensor(
        values,
        shape,
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        device,
        memory_config=_INTERLEAVED_DRAM,
    )


def _make_full_tile(shape, fill_value, dtype, device):
    """Create a TILE tensor on DRAM filled with a single scalar."""
    return ttnn.full(
        shape=ttnn.Shape(shape),
        fill_value=fill_value,
        dtype=dtype,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=_INTERLEAVED_DRAM,
    )


# Per-layer projection weights that get typecast to BFLOAT8_B TILE in place.
_PER_LAYER_BFLOAT8_PROJECTIONS = (
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "self_attn.o_proj",
)


def run_consteval(weights, device):
    """Prepare runtime constants and rewrite ``weights`` for the TTNN forward pass.

    Mutates and returns ``weights`` with these changes:
    - Each per-layer projection in ``_PER_LAYER_BFLOAT8_PROJECTIONS`` and
      ``lm_head.weight`` is replaced with its BFLOAT8_B TILE-on-DRAM version.
    - ``self_attn.{k,q,v}_proj.weight`` are removed and replaced with a single
      fused ``self_attn.qkv_proj.weight`` per layer.
    - ``model.embed_tokens.weight`` is moved to device DRAM.
    - ``model.rotary_emb.inv_freq`` is reshaped to ``[1, 32, 1]`` on device.
    - Five runtime constants are added under descriptive keys (see below).
    """
    for i in _LAYER_INDICES:
        for proj in _PER_LAYER_BFLOAT8_PROJECTIONS:
            key = f"model.layers.{i}.{proj}.weight"
            weights[key] = _to_bfloat8_b_tile(weights[key], device)

        k = weights.pop(f"model.layers.{i}.self_attn.k_proj.weight")
        v = weights.pop(f"model.layers.{i}.self_attn.v_proj.weight")
        q = weights.pop(f"model.layers.{i}.self_attn.q_proj.weight")
        weights[f"model.layers.{i}.self_attn.qkv_proj.weight"] = (
            _fuse_qkv_to_bfloat8_b_tile(k, v, q, device)
        )

    weights["lm_head.weight"] = _to_bfloat8_b_tile(weights["lm_head.weight"], device)
    weights["model.embed_tokens.weight"] = _embed_tokens_to_device(
        weights["model.embed_tokens.weight"], device
    )
    weights["model.rotary_emb.inv_freq"] = _prepare_rotary_inv_freq(
        weights["model.rotary_emb.inv_freq"], device
    )

    # Runtime constants used by the forward pass.
    weights["iota_seq_18"] = _make_int32_iota_tile(list(range(18)), [18], device)
    weights["iota_kv_128"] = _make_int32_iota_tile(
        list(range(128)), [1, 1, 1, 128], device
    )
    weights["const_seq_len_18"] = _make_full_tile(
        [1], 18, ttnn.DataType.INT32, device
    )
    weights["const_attn_mask_neg_inf"] = _make_full_tile(
        [1, 1, 1, 1], float("-inf"), ttnn.DataType.BFLOAT16, device
    )
    weights["const_attn_mask_zero"] = _make_full_tile(
        [1, 1, 1, 1], 0.0, ttnn.DataType.BFLOAT16, device
    )

    return weights
