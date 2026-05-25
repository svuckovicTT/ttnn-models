import ttnn
import ttir_cpu
import torch


NUM_LAYERS = 36
DRAM_MEMORY_CONFIG = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)
POSITIONS_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def _to_tile_bf16_dram(tensor, device):
    bf16 = ttnn.typecast(tensor, ttnn.DataType.BFLOAT16, memory_config=None)
    tiled = ttnn.to_layout(bf16, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(bf16, False)
    on_device = ttnn.to_device(
        tiled, device=device, memory_config=DRAM_MEMORY_CONFIG
    )
    ttnn.deallocate(tiled, False)
    return on_device


def _fuse_qkv_weights_cpu(q_proj_f32, k_proj_f32, v_proj_f32):
    """Concatenate the permuted [V, Q, K] projection weights along dim=1 on the CPU."""
    q_torch = ttnn.to_torch(q_proj_f32)
    k_torch = ttnn.to_torch(k_proj_f32)
    v_torch = ttnn.to_torch(v_proj_f32)
    q_perm = ttir_cpu.permute(q_torch, [1, 0])
    k_perm = ttir_cpu.permute(k_torch, [1, 0])
    v_perm = ttir_cpu.permute(v_torch, [1, 0])
    fused = ttir_cpu.concat([v_perm, q_perm, k_perm], dim=1)
    return ttnn.from_torch(fused)


def _build_fused_qkv_proj_weight(q_proj, k_proj, v_proj, device):
    """Fuse Q/K/V projection weights into a single TILE/BF16/DRAM tensor."""
    q_f32 = ttnn.typecast(q_proj, ttnn.DataType.FLOAT32, memory_config=None)
    k_f32 = ttnn.typecast(k_proj, ttnn.DataType.FLOAT32, memory_config=None)
    v_f32 = ttnn.typecast(v_proj, ttnn.DataType.FLOAT32, memory_config=None)
    fused = _fuse_qkv_weights_cpu(q_f32, k_f32, v_f32)
    ttnn.deallocate(v_f32, False)
    ttnn.deallocate(k_f32, False)
    ttnn.deallocate(q_f32, False)
    on_device = _to_tile_bf16_dram(fused, device)
    ttnn.deallocate(fused, False)
    return on_device


def _build_rotary_cos_sin_cpu(inv_freq):
    """Compute rotary cos/sin tables from inverse frequencies (CPU)."""
    inv_freq_torch = ttnn.to_torch(inv_freq)
    positions = ttir_cpu.constant(
        shape=[1, 1, 16], dtype=torch.float32, data=POSITIONS_16
    )
    angles = ttir_cpu.matmul(
        ttir_cpu.reshape(inv_freq_torch, [1, 64, 1]), positions
    )
    angles = ttir_cpu.reshape(ttir_cpu.permute(angles, [0, 2, 1]), [1, 1, 16, 64])
    angles_dup = ttir_cpu.concat([angles, angles], dim=3)
    cos = ttnn.from_torch(ttir_cpu.cos(angles_dup))
    sin = ttnn.from_torch(ttir_cpu.sin(angles_dup))
    return cos, sin


def _build_causal_mask_cpu():
    """Build the [1, 1, 16, 16] upper-triangular -inf causal attention mask on CPU."""
    row_idx = ttir_cpu.constant(
        shape=[1, 1, 16, 1], dtype=torch.int32, data=POSITIONS_16
    )
    col_idx = ttir_cpu.constant(
        shape=[1, 1, 1, 16], dtype=torch.int32, data=POSITIONS_16
    )
    neg_inf = ttir_cpu.full(
        shape=[1, 1, 1, 1], fill_value=float('-inf'), dtype=torch.float32
    )
    zero = ttir_cpu.full(
        shape=[1, 1, 1, 1], fill_value=0, dtype=torch.float32
    )
    mask = ttir_cpu.where(ttir_cpu.ge(row_idx, col_idx), zero, neg_inf)
    return ttnn.from_torch(mask)


def run_consteval(weights, device):
    """Run all constant-evaluation work and store the results in `weights`.

    Mutates and returns `weights`. Intended to be called exactly once per model
    instance from `ModelTTNN.__init__`.
    """
    # Embedding table: just move to DRAM
    weights["L__self___model_embed_tokens.weight"] = ttnn.to_device(
        weights["L__self___model_embed_tokens.weight"],
        device=device,
        memory_config=DRAM_MEMORY_CONFIG,
    )

    # Fused Q/K/V projection weights per transformer layer
    for layer_idx in range(NUM_LAYERS):
        q_key = f"L__self___model_layers_{layer_idx}_self_attn_q_proj.weight"
        k_key = f"L__self___model_layers_{layer_idx}_self_attn_k_proj.weight"
        v_key = f"L__self___model_layers_{layer_idx}_self_attn_v_proj.weight"
        weights[
            f"L__self___model_layers_{layer_idx}_self_attn_qkv_proj.weight"
        ] = _build_fused_qkv_proj_weight(
            weights[q_key], weights[k_key], weights[v_key], device
        )

    # Rotary embedding cos / sin tables
    cos_cpu, sin_cpu = _build_rotary_cos_sin_cpu(
        weights["L__self___model_rotary_emb_inv_freq"]
    )
    weights["L__self___model_rotary_emb_cos"] = _to_tile_bf16_dram(cos_cpu, device)
    ttnn.deallocate(cos_cpu, False)
    weights["L__self___model_rotary_emb_sin"] = _to_tile_bf16_dram(sin_cpu, device)
    ttnn.deallocate(sin_cpu, False)

    # Per-layer causal attention masks (one device tensor per layer)
    causal_mask_cpu = _build_causal_mask_cpu()
    for layer_idx in range(NUM_LAYERS):
        weights[
            f"L__self___model_layers_{layer_idx}_causal_mask"
        ] = _to_tile_bf16_dram(causal_mask_cpu, device)
    ttnn.deallocate(causal_mask_cpu, False)

    return weights
