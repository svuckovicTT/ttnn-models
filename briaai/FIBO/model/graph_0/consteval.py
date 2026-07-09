import ttnn


DRAM_MEMCFG = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)


def _to_device_tiled(tensor, device):
    t = ttnn.to_device(tensor, device=device, memory_config=DRAM_MEMCFG)
    result = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(t, False)
    return result


def concat_biases(tensors, device):
    tiles = []
    for t in tensors:
        tiles.append(_to_device_tiled(t, device))
    result = ttnn.concat(tiles, 0, memory_config=DRAM_MEMCFG)
    for t in tiles:
        ttnn.deallocate(t, False)
    return result


def permute_and_concat_weights(tensors, device):
    permuted = []
    for t in tensors:
        tiled = _to_device_tiled(t, device)
        p = ttnn.permute(tiled, [1, 0], memory_config=DRAM_MEMCFG, pad_value=0.0)
        ttnn.deallocate(tiled, False)
        permuted.append(p)
    result = ttnn.concat(permuted, 1, memory_config=DRAM_MEMCFG)
    for p in permuted:
        ttnn.deallocate(p, False)
    return result


def reshape_bias(tensor, shape, device):
    tiled = _to_device_tiled(tensor, device)
    result = ttnn.reshape(tiled, shape, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(tiled, False)
    return result


def reshape_and_typecast_norm_bias(tensor, intermediate_shape, final_shape, device):
    tiled = _to_device_tiled(tensor, device)
    reshaped = ttnn.reshape(tiled, intermediate_shape, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(tiled, False)
    casted = ttnn.typecast(reshaped, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(reshaped, False)
    result = ttnn.reshape(casted, final_shape, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(casted, False)
    return result


def typecast_to_float32(tensor, device):
    tiled = _to_device_tiled(tensor, device)
    result = ttnn.typecast(tiled, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(tiled, False)
    return result


def permute_and_typecast_to_float32(tensor, device):
    tiled = _to_device_tiled(tensor, device)
    permuted = ttnn.permute(tiled, [1, 0], memory_config=DRAM_MEMCFG, pad_value=0.0)
    ttnn.deallocate(tiled, False)
    result = ttnn.typecast(permuted, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(permuted, False)
    return result


def mesh_partition_and_typecast(tensor, device):
    on_device = ttnn.to_device(tensor, device=device, memory_config=DRAM_MEMCFG)
    partitioned = ttnn.mesh_partition(
        input_tensor=on_device, dim=0, cluster_axis=1, memory_config=DRAM_MEMCFG,
    )
    ttnn.deallocate(on_device, False)
    tiled = ttnn.to_layout(partitioned, ttnn.Layout.TILE, None, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(partitioned, False)
    result = ttnn.typecast(tiled, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(tiled, False)
    return result


def mesh_partition_permute_and_typecast(tensor, device):
    on_device = ttnn.to_device(tensor, device=device, memory_config=DRAM_MEMCFG)
    partitioned = ttnn.mesh_partition(
        input_tensor=on_device, dim=0, cluster_axis=1, memory_config=DRAM_MEMCFG,
    )
    ttnn.deallocate(on_device, False)
    tiled = ttnn.to_layout(partitioned, ttnn.Layout.TILE, None, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(partitioned, False)
    permuted = ttnn.permute(tiled, [1, 0], memory_config=DRAM_MEMCFG, pad_value=0.0)
    ttnn.deallocate(tiled, False)
    result = ttnn.typecast(permuted, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMCFG)
    ttnn.deallocate(permuted, False)
    return result


def typecast_concat_norm_weights(weight_keys, weights, device):
    casted = []
    for key in weight_keys:
        tiled = _to_device_tiled(weights[key], device)
        c = ttnn.typecast(tiled, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMCFG)
        ttnn.deallocate(tiled, False)
        casted.append(c)
    result = ttnn.concat(casted, 1, memory_config=DRAM_MEMCFG)
    for c in casted:
        ttnn.deallocate(c, False)
    return result


def create_timestep_embedding_frequencies(device):
    return ttnn.Tensor(
        [
            1.0,
            0.9305720329284668,
            0.86596429347991943,
            0.80584216117858887,
            0.74989420175552368,
            0.69783055782318115,
            0.64938163757324219,
            0.60429638624191284,
            0.56234133243560791,
            0.52329909801483154,
            0.48696750402450562,
            0.45315837860107422,
            0.42169648408889771,
            0.39241895079612732,
            0.36517414450645447,
            0.33982083201408386,
            0.31622776389122009,
            0.29427272081375122,
            0.27384194731712341,
            0.25482964515686035,
            0.23713734745979309,
            0.22067341208457947,
            0.20535250008106232,
            0.19109529256820679,
            0.17782793939113617,
            0.16548170149326324,
            0.15399263799190521,
            0.14330124855041504,
            0.13335214555263519,
            0.12409376353025436,
            0.11547819525003433,
            0.10746076703071594,
            0.099999994039535522,
            0.093057207763195038,
            0.086596429347991943,
            0.080584220588207245,
            0.07498941570520401,
            0.069783061742782593,
            0.064938157796859741,
            0.060429636389017105,
            0.056234125047922134,
            0.052329909056425095,
            0.048696756362915039,
            0.045315831899642944,
            0.04216964915394783,
            0.039241891354322433,
            0.036517411470413208,
            0.033982079476118088,
            0.03162277489900589,
            0.029427273198962212,
            0.027384193614125252,
            0.025482967495918274,
            0.023713734000921249,
            0.022067340090870857,
            0.020535247400403023,
            0.019109528511762619,
            0.017782794311642647,
            0.016548173502087593,
            0.015399262309074402,
            0.014330124482512474,
            0.013335213996469975,
            0.01240937877446413,
            0.011547816917300224,
            0.010746076703071594,
            0.0099999997764825821,
            0.0093057211488485336,
            0.0086596440523862839,
            0.0080584203824400902,
            0.0074989409185945988,
            0.006978305522352457,
            0.0064938166178762913,
            0.0060429619625210762,
            0.0056234123185276985,
            0.0052329907193779945,
            0.0048696752637624741,
            0.0045315842144191265,
            0.0042169638909399509,
            0.0039241891354322433,
            0.0036517411936074495,
            0.0033982084132730961,
            0.0031622766982764006,
            0.002942726481705904,
            0.002738419221714139,
            0.0025482967030256987,
            0.0023713738191872835,
            0.002206733450293541,
            0.0020535246003419161,
            0.0019109528511762619,
            0.0017782794311642647,
            0.0016548173734918237,
            0.0015399261610582471,
            0.001433012424968183,
            0.0013335214462131262,
            0.0012409378541633487,
            0.0011547816684469581,
            0.0010746076004579663,
            0.00099999993108212948,
            0.00093057204503566027,
            0.0008659643935970962,
            0.00080584199167788029,
            0.00074989406857639551,
            0.00069783051731064916,
            0.00064938166178762913,
            0.00060429621953517199,
            0.00056234118528664112,
            0.00052329903701320291,
            0.00048696750309318304,
            0.00045315839815884829,
            0.00042169637163169682,
            0.00039241890772245824,
            0.00036517408443614841,
            0.00033982083550654352,
            0.00031622781534679234,
            0.00029427278786897659,
            0.00027384204440750182,
            0.00025482953060418367,
            0.00023713726841378957,
            0.00022067333338782191,
            0.0002053524658549577,
            0.00019109527056571096,
            0.00017782794020604342,
            0.00016548173152841628,
            0.00015399268886540085,
            0.00014330129488371313,
            0.00013335207768250257,
            0.00012409372720867395,
            0.00011547815665835515,
            0.00010746075713541359,
        ],
        [1, 128],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        device,
        memory_config=DRAM_MEMCFG,
    )


def create_rope_frequencies(device):
    return ttnn.Tensor(
        [
            1.0,
            0.31622776389122009,
            0.10000000149011612,
            0.031622778624296188,
            0.0099999997764825821,
            0.0031622778624296188,
            0.0010000000474974513,
            0.00031622778624296188,
        ],
        [1, 8],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        device,
        memory_config=DRAM_MEMCFG,
    )


def create_context_rope_frequencies(device):
    return ttnn.Tensor(
        [
            1.0,
            0.71968567371368408,
            0.51794743537902832,
            0.3727593719959259,
            0.26826956868171692,
            0.19306977093219757,
            0.13894954323768616,
            0.10000000149011612,
            0.071968555450439453,
            0.05179474875330925,
            0.037275936454534531,
            0.026826959103345871,
            0.019306976348161697,
            0.013894956558942795,
            0.0099999997764825821,
            0.0071968580596148968,
            0.0051794731989502907,
            0.0037275934591889381,
            0.002682696096599102,
            0.001930698286741972,
            0.0013894952135160565,
            0.0010000000474974513,
            0.00071968580596148968,
            0.00051794730825349689,
            0.00037275932845659554,
            0.00026826959219761193,
            0.00019306980539113283,
            0.00013894953008275479,
        ],
        [1, 28],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        device,
        memory_config=DRAM_MEMCFG,
    )


def create_ones_scalar(device):
    return ttnn.ones(
        shape=ttnn.Shape([1, 1, 1]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MEMCFG,
    )


def run_consteval(weights, device):
    # Timestep embedding frequencies
    weights["consteval.timestep_embedding_frequencies"] = create_timestep_embedding_frequencies(device)

    # RoPE frequencies
    weights["consteval.rope_frequencies"] = create_rope_frequencies(device)

    # Context RoPE frequencies
    weights["consteval.context_rope_frequencies"] = create_context_rope_frequencies(device)

    # Ones scalar
    weights["consteval.ones_scalar"] = create_ones_scalar(device)

    # --- single_transformer_blocks (0-37) ---
    for i in range(38):
        prefix = f"transformer.single_transformer_blocks.{i}"

        # Concat QKV + proj_mlp biases
        weights[f"{prefix}.fused_qkv_proj_mlp.bias"] = concat_biases(
            [
                weights[f"{prefix}.attn.to_q.bias"],
                weights[f"{prefix}.attn.to_k.bias"],
                weights[f"{prefix}.attn.to_v.bias"],
                weights[f"{prefix}.proj_mlp.bias"],
            ],
            device,
        )

        # Permute+concat QKV + proj_mlp weights
        weights[f"{prefix}.fused_qkv_proj_mlp.weight"] = permute_and_concat_weights(
            [
                weights[f"{prefix}.attn.to_q.weight"],
                weights[f"{prefix}.attn.to_k.weight"],
                weights[f"{prefix}.attn.to_v.weight"],
                weights[f"{prefix}.proj_mlp.weight"],
            ],
            device,
        )

        # Reshape proj_out bias
        weights[f"{prefix}.proj_out.bias"] = reshape_bias(
            weights[f"{prefix}.proj_out.bias"], [1, 3072], device,
        )

        # Reshape+typecast norm bias
        weights[f"{prefix}.norm.linear.bias"] = reshape_and_typecast_norm_bias(
            weights[f"{prefix}.norm.linear.bias"], [1, 1, 9216], [1, 9216], device,
        )

    # --- transformer_blocks (0-7) ---
    for i in range(8):
        prefix = f"transformer.transformer_blocks.{i}"

        # Concat main QKV biases
        weights[f"{prefix}.attn.fused_qkv.bias"] = concat_biases(
            [
                weights[f"{prefix}.attn.to_q.bias"],
                weights[f"{prefix}.attn.to_k.bias"],
                weights[f"{prefix}.attn.to_v.bias"],
            ],
            device,
        )

        # Concat context (add) QKV biases
        weights[f"{prefix}.attn.fused_add_qkv.bias"] = concat_biases(
            [
                weights[f"{prefix}.attn.add_q_proj.bias"],
                weights[f"{prefix}.attn.add_k_proj.bias"],
                weights[f"{prefix}.attn.add_v_proj.bias"],
            ],
            device,
        )

        # Permute+concat main QKV weights
        weights[f"{prefix}.attn.fused_qkv.weight"] = permute_and_concat_weights(
            [
                weights[f"{prefix}.attn.to_q.weight"],
                weights[f"{prefix}.attn.to_k.weight"],
                weights[f"{prefix}.attn.to_v.weight"],
            ],
            device,
        )

        # Permute+concat context (add) QKV weights
        weights[f"{prefix}.attn.fused_add_qkv.weight"] = permute_and_concat_weights(
            [
                weights[f"{prefix}.attn.add_q_proj.weight"],
                weights[f"{prefix}.attn.add_k_proj.weight"],
                weights[f"{prefix}.attn.add_v_proj.weight"],
            ],
            device,
        )

        # Reshape attention output biases
        weights[f"{prefix}.attn.to_out.0.bias"] = reshape_bias(
            weights[f"{prefix}.attn.to_out.0.bias"], [1, 3072], device,
        )
        weights[f"{prefix}.attn.to_add_out.bias"] = reshape_bias(
            weights[f"{prefix}.attn.to_add_out.bias"], [1, 3072], device,
        )

        # Reshape FF biases
        weights[f"{prefix}.ff.net.2.bias"] = reshape_bias(
            weights[f"{prefix}.ff.net.2.bias"], [1, 3072], device,
        )
        weights[f"{prefix}.ff_context.net.2.bias"] = reshape_bias(
            weights[f"{prefix}.ff_context.net.2.bias"], [1, 3072], device,
        )

        # Reshape+typecast norm biases
        weights[f"{prefix}.norm1.linear.bias"] = reshape_and_typecast_norm_bias(
            weights[f"{prefix}.norm1.linear.bias"], [1, 1, 18432], [1, 18432], device,
        )
        weights[f"{prefix}.norm1_context.linear.bias"] = reshape_and_typecast_norm_bias(
            weights[f"{prefix}.norm1_context.linear.bias"], [1, 1, 18432], [1, 18432], device,
        )

    # --- norm_out bias ---
    weights["transformer.norm_out.linear.bias"] = reshape_and_typecast_norm_bias(
        weights["transformer.norm_out.linear.bias"], [1, 1, 6144], [1, 6144], device,
    )

    # --- Time embedding ---
    weights["transformer.time_embed.timestep_embedder.linear_1.bias"] = typecast_to_float32(
        weights["transformer.time_embed.timestep_embedder.linear_1.bias"], device,
    )
    weights["transformer.time_embed.timestep_embedder.linear_1.weight"] = permute_and_typecast_to_float32(
        weights["transformer.time_embed.timestep_embedder.linear_1.weight"], device,
    )
    weights["transformer.time_embed.timestep_embedder.linear_2.weight"] = mesh_partition_permute_and_typecast(
        weights["transformer.time_embed.timestep_embedder.linear_2.weight"], device,
    )
    weights["transformer.time_embed.timestep_embedder.linear_2.bias"] = mesh_partition_and_typecast(
        weights["transformer.time_embed.timestep_embedder.linear_2.bias"], device,
    )

    # --- Fused normalization weights ---
    # Concatenate all norm weights typecast to FLOAT32 then concat(dim=1).
    # Order: tb.7-0 norm1_context, tb.0-7 norm1, stb.0-37 norm, norm_out
    norm_weight_keys = []
    for i in range(7, -1, -1):
        norm_weight_keys.append(f"transformer.transformer_blocks.{i}.norm1_context.linear.weight")
    for i in range(8):
        norm_weight_keys.append(f"transformer.transformer_blocks.{i}.norm1.linear.weight")
    for i in range(38):
        norm_weight_keys.append(f"transformer.single_transformer_blocks.{i}.norm.linear.weight")
    norm_weight_keys.append("transformer.norm_out.linear.weight")
    weights["consteval.fused_norm_weights"] = typecast_concat_norm_weights(
        norm_weight_keys, weights, device,
    )

    return weights
