import ttnn

DRAM_MEMORY_CONFIG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def _to_device_tile(tensor, device):
    t = ttnn.to_device(tensor, device=device, memory_config=DRAM_MEMORY_CONFIG)
    result = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t, False)
    return result


def _reshape_bias(tensor, device, shape):
    t = _to_device_tile(tensor, device)
    result = ttnn.reshape(t, shape, memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t, False)
    return result


def _reshape_typecast_bias(tensor, device, intermediate_shape, final_shape):
    t = _to_device_tile(tensor, device)
    t2 = ttnn.reshape(t, intermediate_shape, memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t, False)
    t3 = ttnn.typecast(t2, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t2, False)
    result = ttnn.reshape(t3, final_shape, memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t3, False)
    return result


def _typecast_to_float32(tensor, device):
    t = _to_device_tile(tensor, device)
    result = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t, False)
    return result


def _permute_typecast(tensor, device):
    t = _to_device_tile(tensor, device)
    t2 = ttnn.permute(t, [1, 0], memory_config=DRAM_MEMORY_CONFIG, pad_value=0.0)
    ttnn.deallocate(t, False)
    result = ttnn.typecast(t2, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t2, False)
    return result


def _mesh_partition_permute_typecast(tensor, device):
    t = ttnn.to_device(tensor, device=device, memory_config=DRAM_MEMORY_CONFIG)
    t2 = ttnn.mesh_partition(input_tensor=t, dim=0, cluster_axis=1, memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t, False)
    t3 = ttnn.to_layout(t2, ttnn.Layout.TILE, None, memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t2, False)
    t4 = ttnn.permute(t3, [1, 0], memory_config=DRAM_MEMORY_CONFIG, pad_value=0.0)
    ttnn.deallocate(t3, False)
    result = ttnn.typecast(t4, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t4, False)
    return result


def _mesh_partition_typecast(tensor, device):
    t = ttnn.to_device(tensor, device=device, memory_config=DRAM_MEMORY_CONFIG)
    t2 = ttnn.mesh_partition(input_tensor=t, dim=0, cluster_axis=1, memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t, False)
    t3 = ttnn.to_layout(t2, ttnn.Layout.TILE, None, memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t2, False)
    result = ttnn.typecast(t3, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t3, False)
    return result


def _concat_biases(tensors, device):
    tiled = [_to_device_tile(t, device) for t in tensors]
    result = ttnn.concat(tiled, 0, memory_config=DRAM_MEMORY_CONFIG)
    for t in tiled:
        ttnn.deallocate(t, False)
    return result


def _permute_concat_weights(tensors, device):
    permuted = []
    for tensor in tensors:
        t = _to_device_tile(tensor, device)
        p = ttnn.permute(t, [1, 0], memory_config=DRAM_MEMORY_CONFIG, pad_value=0.0)
        ttnn.deallocate(t, False)
        permuted.append(p)
    result = ttnn.concat(permuted, 1, memory_config=DRAM_MEMORY_CONFIG)
    for p in permuted:
        ttnn.deallocate(p, False)
    return result


def _permute_typecast_concat_weights(tensors, device):
    processed = []
    for tensor in tensors:
        t = _to_device_tile(tensor, device)
        p = ttnn.permute(t, [1, 0], memory_config=DRAM_MEMORY_CONFIG, pad_value=0.0)
        ttnn.deallocate(t, False)
        tc = ttnn.typecast(p, ttnn.DataType.FLOAT32, memory_config=DRAM_MEMORY_CONFIG)
        ttnn.deallocate(p, False)
        processed.append(tc)
    result = ttnn.concat(processed, 1, memory_config=DRAM_MEMORY_CONFIG)
    for tc in processed:
        ttnn.deallocate(tc, False)
    return result


def run_consteval(weights, device):
    weights["transformer.time_embed.timestep_frequencies_128"] = ttnn.Tensor(
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
        memory_config=DRAM_MEMORY_CONFIG,
    )

    weights["transformer.time_embed.timestep_frequencies_8"] = ttnn.Tensor(
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
        memory_config=DRAM_MEMORY_CONFIG,
    )

    weights["transformer.time_embed.timestep_frequencies_28"] = ttnn.Tensor(
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
        memory_config=DRAM_MEMORY_CONFIG,
    )

    weights["transformer.ones_constant"] = ttnn.ones(
        shape=ttnn.Shape([1, 1, 1]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MEMORY_CONFIG,
    )

    # Reshape proj_out biases for single_transformer_blocks
    for i in range(38):
        key = f"transformer.single_transformer_blocks.{i}.proj_out.bias"
        new_key = f"transformer.reshaped.single_transformer_blocks.{i}.proj_out.bias"
        weights[new_key] = _reshape_bias(weights[key], device, [1, 3072])

    # Reshape attn.to_out biases for transformer_blocks
    for i in range(8):
        key = f"transformer.transformer_blocks.{i}.attn.to_out.0.bias"
        new_key = f"transformer.reshaped.transformer_blocks.{i}.attn.to_out.0.bias"
        weights[new_key] = _reshape_bias(weights[key], device, [1, 3072])

    # Reshape attn.to_add_out biases for transformer_blocks
    for i in range(8):
        key = f"transformer.transformer_blocks.{i}.attn.to_add_out.bias"
        new_key = f"transformer.reshaped.transformer_blocks.{i}.attn.to_add_out.bias"
        weights[new_key] = _reshape_bias(weights[key], device, [1, 3072])

    # Reshape ff.net.2 biases for transformer_blocks
    for i in range(8):
        key = f"transformer.transformer_blocks.{i}.ff.net.2.bias"
        new_key = f"transformer.reshaped.transformer_blocks.{i}.ff.net.2.bias"
        weights[new_key] = _reshape_bias(weights[key], device, [1, 3072])

    # Reshape ff_context.net.2 biases for transformer_blocks
    for i in range(8):
        key = f"transformer.transformer_blocks.{i}.ff_context.net.2.bias"
        new_key = f"transformer.reshaped.transformer_blocks.{i}.ff_context.net.2.bias"
        weights[new_key] = _reshape_bias(weights[key], device, [1, 3072])

    # Reshape+typecast norm biases for single_transformer_blocks
    for i in range(38):
        key = f"transformer.single_transformer_blocks.{i}.norm.linear.bias"
        new_key = f"transformer.reshaped.single_transformer_blocks.{i}.norm.linear.bias"
        weights[new_key] = _reshape_typecast_bias(weights[key], device, [1, 1, 9216], [1, 9216])

    # Reshape+typecast norm1 biases for transformer_blocks
    for i in range(8):
        key = f"transformer.transformer_blocks.{i}.norm1.linear.bias"
        new_key = f"transformer.reshaped.transformer_blocks.{i}.norm1.linear.bias"
        weights[new_key] = _reshape_typecast_bias(weights[key], device, [1, 1, 18432], [1, 18432])

    # Reshape+typecast norm1_context biases for transformer_blocks
    for i in range(8):
        key = f"transformer.transformer_blocks.{i}.norm1_context.linear.bias"
        new_key = f"transformer.reshaped.transformer_blocks.{i}.norm1_context.linear.bias"
        weights[new_key] = _reshape_typecast_bias(weights[key], device, [1, 1, 18432], [1, 18432])

    # Reshape+typecast norm_out bias
    weights["transformer.reshaped.norm_out.linear.bias"] = _reshape_typecast_bias(
        weights["transformer.norm_out.linear.bias"], device, [1, 1, 6144], [1, 6144]
    )

    # Fuse QKV biases for single_transformer_blocks
    for i in range(38):
        weights[f"transformer.single_transformer_blocks.{i}.attn.fused_qkv_bias"] = _concat_biases(
            [
                weights[f"transformer.single_transformer_blocks.{i}.proj_mlp.bias"],
                weights[f"transformer.single_transformer_blocks.{i}.attn.to_v.bias"],
                weights[f"transformer.single_transformer_blocks.{i}.attn.to_k.bias"],
                weights[f"transformer.single_transformer_blocks.{i}.attn.to_q.bias"],
            ],
            device,
        )

    # Fuse QKV biases for transformer_blocks
    for i in range(8):
        weights[f"transformer.transformer_blocks.{i}.attn.fused_qkv_bias"] = _concat_biases(
            [
                weights[f"transformer.transformer_blocks.{i}.attn.to_v.bias"],
                weights[f"transformer.transformer_blocks.{i}.attn.to_k.bias"],
                weights[f"transformer.transformer_blocks.{i}.attn.to_q.bias"],
            ],
            device,
        )

    # Fuse add-QKV biases for transformer_blocks
    for i in range(8):
        weights[f"transformer.transformer_blocks.{i}.attn.fused_add_qkv_bias"] = _concat_biases(
            [
                weights[f"transformer.transformer_blocks.{i}.attn.add_v_proj.bias"],
                weights[f"transformer.transformer_blocks.{i}.attn.add_k_proj.bias"],
                weights[f"transformer.transformer_blocks.{i}.attn.add_q_proj.bias"],
            ],
            device,
        )

    # Fuse QKV weights for single_transformer_blocks
    for i in range(38):
        weights[f"transformer.single_transformer_blocks.{i}.attn.fused_qkv_weight"] = _permute_concat_weights(
            [
                weights[f"transformer.single_transformer_blocks.{i}.proj_mlp.weight"],
                weights[f"transformer.single_transformer_blocks.{i}.attn.to_v.weight"],
                weights[f"transformer.single_transformer_blocks.{i}.attn.to_k.weight"],
                weights[f"transformer.single_transformer_blocks.{i}.attn.to_q.weight"],
            ],
            device,
        )

    # Fuse QKV weights for transformer_blocks
    for i in range(8):
        weights[f"transformer.transformer_blocks.{i}.attn.fused_qkv_weight"] = _permute_concat_weights(
            [
                weights[f"transformer.transformer_blocks.{i}.attn.to_v.weight"],
                weights[f"transformer.transformer_blocks.{i}.attn.to_k.weight"],
                weights[f"transformer.transformer_blocks.{i}.attn.to_q.weight"],
            ],
            device,
        )

    # Fuse add-QKV weights for transformer_blocks
    for i in range(8):
        weights[f"transformer.transformer_blocks.{i}.attn.fused_add_qkv_weight"] = _permute_concat_weights(
            [
                weights[f"transformer.transformer_blocks.{i}.attn.add_v_proj.weight"],
                weights[f"transformer.transformer_blocks.{i}.attn.add_k_proj.weight"],
                weights[f"transformer.transformer_blocks.{i}.attn.add_q_proj.weight"],
            ],
            device,
        )

    # Typecast timestep embedder biases and weights
    weights["transformer.typecast.time_embed.timestep_embedder.linear_1.bias"] = _typecast_to_float32(
        weights["transformer.time_embed.timestep_embedder.linear_1.bias"], device
    )

    weights["transformer.permuted.time_embed.timestep_embedder.linear_1.weight"] = _permute_typecast(
        weights["transformer.time_embed.timestep_embedder.linear_1.weight"], device
    )

    weights["transformer.partitioned.time_embed.timestep_embedder.linear_2.weight"] = _mesh_partition_permute_typecast(
        weights["transformer.time_embed.timestep_embedder.linear_2.weight"], device
    )

    weights["transformer.partitioned.time_embed.timestep_embedder.linear_2.bias"] = _mesh_partition_typecast(
        weights["transformer.time_embed.timestep_embedder.linear_2.bias"], device
    )

    # Fuse all norm weights (permute + typecast + concat)
    weights["transformer.fused_norm_weight"] = _permute_typecast_concat_weights(
        [
            weights["transformer.norm_out.linear.weight"],
            weights["transformer.single_transformer_blocks.37.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.36.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.35.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.34.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.33.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.32.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.31.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.30.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.29.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.28.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.27.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.26.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.25.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.24.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.23.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.22.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.21.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.20.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.19.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.18.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.17.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.16.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.15.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.14.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.13.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.12.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.11.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.10.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.9.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.8.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.7.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.6.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.5.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.4.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.3.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.2.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.1.norm.linear.weight"],
            weights["transformer.single_transformer_blocks.0.norm.linear.weight"],
            weights["transformer.transformer_blocks.7.norm1.linear.weight"],
            weights["transformer.transformer_blocks.6.norm1.linear.weight"],
            weights["transformer.transformer_blocks.5.norm1.linear.weight"],
            weights["transformer.transformer_blocks.4.norm1.linear.weight"],
            weights["transformer.transformer_blocks.3.norm1.linear.weight"],
            weights["transformer.transformer_blocks.2.norm1.linear.weight"],
            weights["transformer.transformer_blocks.1.norm1.linear.weight"],
            weights["transformer.transformer_blocks.0.norm1.linear.weight"],
            weights["transformer.transformer_blocks.0.norm1_context.linear.weight"],
            weights["transformer.transformer_blocks.1.norm1_context.linear.weight"],
            weights["transformer.transformer_blocks.2.norm1_context.linear.weight"],
            weights["transformer.transformer_blocks.3.norm1_context.linear.weight"],
            weights["transformer.transformer_blocks.4.norm1_context.linear.weight"],
            weights["transformer.transformer_blocks.5.norm1_context.linear.weight"],
            weights["transformer.transformer_blocks.6.norm1_context.linear.weight"],
            weights["transformer.transformer_blocks.7.norm1_context.linear.weight"],
        ],
        device,
    )

    return weights
