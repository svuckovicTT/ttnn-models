import ttnn

DRAM_CONFIG = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)


def _transpose_weight_bf8b(weight, device):
    t = ttnn.to_device(weight, device=device, memory_config=DRAM_CONFIG)
    t_layout = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=DRAM_CONFIG)
    ttnn.deallocate(t, False)
    t_perm = ttnn.permute(t_layout, [1, 0], memory_config=DRAM_CONFIG, pad_value=0.0)
    ttnn.deallocate(t_layout, False)
    t_host = ttnn.from_device(t_perm)
    ttnn.deallocate(t_perm, False)
    t_cast = ttnn.typecast(t_host, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(t_host, False)
    result = ttnn.to_device(t_cast, device=device, memory_config=DRAM_CONFIG)
    ttnn.deallocate(t_cast, False)
    return result


def _transpose_weight_f32_bf8b(weight, device):
    t = ttnn.to_device(weight, device=device, memory_config=DRAM_CONFIG)
    t_layout = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=DRAM_CONFIG)
    ttnn.deallocate(t, False)
    t_perm = ttnn.permute(t_layout, [1, 0], memory_config=DRAM_CONFIG, pad_value=0.0)
    ttnn.deallocate(t_layout, False)
    t_f32 = ttnn.typecast(t_perm, ttnn.DataType.FLOAT32, memory_config=DRAM_CONFIG)
    ttnn.deallocate(t_perm, False)
    t_host = ttnn.from_device(t_f32)
    ttnn.deallocate(t_f32, False)
    t_cast = ttnn.typecast(t_host, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(t_host, False)
    result = ttnn.to_device(t_cast, device=device, memory_config=DRAM_CONFIG)
    ttnn.deallocate(t_cast, False)
    return result


def _concat_qkv_weights(k_weight, v_weight, q_weight, device):
    parts = []
    for w in [q_weight, k_weight, v_weight]:
        t = ttnn.to_device(w, device=device, memory_config=DRAM_CONFIG)
        t_layout = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=DRAM_CONFIG)
        ttnn.deallocate(t, False)
        t_perm = ttnn.permute(t_layout, [1, 0], memory_config=DRAM_CONFIG, pad_value=0.0)
        ttnn.deallocate(t_layout, False)
        parts.append(t_perm)
    result_concat = ttnn.concat(parts, 1, memory_config=DRAM_CONFIG)
    for p in parts:
        ttnn.deallocate(p, False)
    t_host = ttnn.from_device(result_concat)
    ttnn.deallocate(result_concat, False)
    t_cast = ttnn.typecast(t_host, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(t_host, False)
    result = ttnn.to_device(t_cast, device=device, memory_config=DRAM_CONFIG)
    ttnn.deallocate(t_cast, False)
    return result


def _concat_qkv_biases(k_bias, v_bias, q_bias, device):
    parts = []
    for b in [q_bias, k_bias, v_bias]:
        t = ttnn.to_device(b, device=device, memory_config=DRAM_CONFIG)
        t_layout = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=DRAM_CONFIG)
        ttnn.deallocate(t, False)
        parts.append(t_layout)
    result = ttnn.concat(parts, 0, memory_config=DRAM_CONFIG)
    for p in parts:
        ttnn.deallocate(p, False)
    return result


def _reshape_expert_bf8b(weight, shape, device):
    t = ttnn.to_device(weight, device=device, memory_config=DRAM_CONFIG)
    t_layout = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=DRAM_CONFIG)
    ttnn.deallocate(t, False)
    t_reshape = ttnn.reshape(t_layout, shape, memory_config=DRAM_CONFIG)
    ttnn.deallocate(t_layout, False)
    t_host = ttnn.from_device(t_reshape)
    ttnn.deallocate(t_reshape, False)
    t_cast = ttnn.typecast(t_host, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(t_host, False)
    result = ttnn.to_device(t_cast, device=device, memory_config=DRAM_CONFIG)
    ttnn.deallocate(t_cast, False)
    return result


def _to_device_tile_reshape_f32(weight, shape, device):
    t = ttnn.to_device(weight, device=device, memory_config=DRAM_CONFIG)
    t_layout = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=DRAM_CONFIG)
    ttnn.deallocate(t, False)
    t_reshape = ttnn.reshape(t_layout, shape, memory_config=DRAM_CONFIG)
    ttnn.deallocate(t_layout, False)
    result = ttnn.typecast(t_reshape, ttnn.DataType.FLOAT32, memory_config=DRAM_CONFIG)
    ttnn.deallocate(t_reshape, False)
    return result


def _allgather_reshape_row_major(init_fn, device):
    t = init_fn(
        shape=ttnn.Shape([16, 1]),
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_CONFIG,
    )
    t_gather = ttnn.all_gather(
        input_tensor=t,
        dim=0,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=DRAM_CONFIG,
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(t, False)
    t_reshape = ttnn.reshape(t_gather, [64], memory_config=DRAM_CONFIG)
    ttnn.deallocate(t_gather, False)
    result = ttnn.to_layout(
        t_reshape, ttnn.Layout.ROW_MAJOR, None, memory_config=DRAM_CONFIG
    )
    ttnn.deallocate(t_reshape, False)
    return result


def run_consteval(weights, device):
    weights["model.model.embed_tokens.weight.device"] = ttnn.to_device(
        weights["model.model.embed_tokens.weight"],
        device=device,
        memory_config=DRAM_CONFIG,
    )

    for i in range(4):
        pfx = f"model.model.layers.{i}.self_attn"
        weights[f"{pfx}.qkv_proj.weight"] = _concat_qkv_weights(
            weights[f"{pfx}.k_proj.weight"],
            weights[f"{pfx}.v_proj.weight"],
            weights[f"{pfx}.q_proj.weight"],
            device,
        )
        weights[f"{pfx}.qkv_proj.bias"] = _concat_qkv_biases(
            weights[f"{pfx}.k_proj.bias"],
            weights[f"{pfx}.v_proj.bias"],
            weights[f"{pfx}.q_proj.bias"],
            device,
        )

    import dram_matmul

    for i in range(4):
        key = f"model.model.layers.{i}.self_attn.o_proj.weight"
        wt = _transpose_weight_bf8b(weights[key], device)
        # DRAM width-shard the o_proj weight (k=1536, n=5120) so the decode matmul
        # can read it across all 12 DRAM banks in parallel (consteval -> no perf cost).
        weights[f"{key}.t"] = ttnn.to_memory_config(
            wt, dram_matmul.weight_dram_sharded_config(device, 1536, 5120)
        )
        ttnn.deallocate(wt, False)

    for i in range(3):
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            key = f"model.model.layers.{i}.mlp.{proj}.weight"
            weights[f"{key}.t"] = _transpose_weight_bf8b(weights[key], device)

    for proj in ["gate_proj", "up_proj", "down_proj"]:
        key = f"model.model.layers.3.mlp.shared_experts.{proj}.weight"
        weights[f"{key}.t"] = _transpose_weight_bf8b(weights[key], device)

    weights["model.lm_head.weight.t"] = _transpose_weight_bf8b(
        weights["model.lm_head.weight"], device
    )

    weights["model.model.layers.3.mlp.mlp.router.gate.weight.t"] = (
        _transpose_weight_f32_bf8b(
            weights["model.model.layers.3.mlp.mlp.router.gate.weight"], device
        )
    )

    for proj in ["gate_proj", "up_proj"]:
        key = f"model.model.layers.3.mlp.mlp.experts.{proj}"
        weights[f"{key}.reshaped"] = _reshape_expert_bf8b(
            weights[key], [1, 5, 5120, 1536], device
        )
    weights["model.model.layers.3.mlp.mlp.experts.down_proj.reshaped"] = (
        _reshape_expert_bf8b(
            weights["model.model.layers.3.mlp.mlp.experts.down_proj"],
            [1, 5, 1536, 5120],
            device,
        )
    )

    e_score_key = (
        "L__self___model_model_layers_3_mlp_mlp_router"
        "__route_fn___closure___0_cell_contents_e_score_correction_bias"
    )
    weights["consteval.e_score_correction_bias"] = _to_device_tile_reshape_f32(
        weights[e_score_key], [1, 160], device
    )

    weights["consteval.rotary_inv_freq"] = _to_device_tile_reshape_f32(
        weights["model.model.rotary_emb.inv_freq"], [1, 32, 1], device
    )

    t_u16 = ttnn.typecast(
        weights["model.model.layers.3.mlp.mlp.expert_mapping"],
        ttnn.DataType.UINT16,
        memory_config=None,
    )
    weights["consteval.expert_mapping_u16"] = ttnn.to_device(
        t_u16, device=device, memory_config=DRAM_CONFIG
    )
    ttnn.deallocate(t_u16, False)

    weights["consteval.scalar_zero_f32"] = ttnn.zeros(
        shape=ttnn.Shape([1, 1]),
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_CONFIG,
    )

    weights["consteval.scalar_one_i32"] = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=1,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_CONFIG,
    )

    weights["consteval.scalar_zero_bf16"] = ttnn.zeros(
        shape=ttnn.Shape([1, 1, 1, 1]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_CONFIG,
    )

    weights["consteval.neg_inf_bf16"] = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=-3.3895313892515355e38,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_CONFIG,
    )

    weights["consteval.head_dim_indices"] = ttnn.Tensor(
        list(range(128)),
        [1, 1, 1, 128],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        device,
        memory_config=DRAM_CONFIG,
    )

    weights["consteval.mesh_zeros"] = _allgather_reshape_row_major(ttnn.zeros, device)
    weights["consteval.mesh_ones"] = _allgather_reshape_row_major(ttnn.ones, device)

    t_arange = ttnn.arange(
        0,
        16,
        1,
        dtype=ttnn.DataType.INT32,
        device=device,
        layout=ttnn.Layout.TILE,
        memory_config=DRAM_CONFIG,
    )
    weights["consteval.batch_indices_i32"] = ttnn.reshape(
        t_arange, [16, 1, 1], memory_config=DRAM_CONFIG
    )
    ttnn.deallocate(t_arange, False)

    t_arange = ttnn.arange(
        0,
        16,
        1,
        dtype=ttnn.DataType.UINT32,
        device=device,
        layout=ttnn.Layout.TILE,
        memory_config=DRAM_CONFIG,
    )
    t_reshape = ttnn.reshape(t_arange, [16, 1, 1], memory_config=DRAM_CONFIG)
    ttnn.deallocate(t_arange, False)
    weights["consteval.moe_batch_indices"] = ttnn.repeat(
        t_reshape, ttnn.Shape([1, 8, 1]), memory_config=DRAM_CONFIG
    )
    ttnn.deallocate(t_reshape, False)

    weights["consteval.topk_scaling"] = ttnn.full(
        shape=ttnn.Shape([1, 1, 1]),
        fill_value=2.5,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_CONFIG,
    )

    weights["consteval.moe_epsilon"] = ttnn.full(
        shape=ttnn.Shape([1, 1]),
        fill_value=9.9999996826552254e-21,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_CONFIG,
    )

    weights["consteval.expert_indices"] = ttnn.Tensor(
        list(range(160)),
        [1, 1, 160],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        device,
        memory_config=DRAM_CONFIG,
    )

    weights["consteval.moe_constants"] = ttnn.Tensor(
        [160.0, 1.0],
        [2, 1],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        device,
        memory_config=DRAM_CONFIG,
    )

    return weights
