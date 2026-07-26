import ttnn
import torch
import utils
import ttir_cpu

DRAM_MC = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)


def _to_device(tensor, device):
    return ttnn.to_device(tensor, device=device, memory_config=DRAM_MC)


def _cpu_permute_transpose(tensor, device):
    def impl(t):
        return ttir_cpu.permute(t, [1, 0])
    return utils.execute_cpu_hoisted_function([tensor], impl, device)


def _cpu_permute_concat_3(t0, t1, t2, device):
    def impl(a, b, c):
        return ttir_cpu.concat(
            [ttir_cpu.permute(a, [1, 0]),
             ttir_cpu.permute(b, [1, 0]),
             ttir_cpu.permute(c, [1, 0])],
            dim=1,
        )
    return utils.execute_cpu_hoisted_function([t0, t1, t2], impl, device)


def _cpu_concat_dim0(t0, t1, t2, device):
    def impl(a, b, c):
        return ttir_cpu.concat([a, b, c], dim=0)
    return utils.execute_cpu_hoisted_function([t0, t1, t2], impl, device)


def _cpu_reshape(tensor, shape, device):
    def impl(t):
        return ttir_cpu.reshape(t, shape)
    return utils.execute_cpu_hoisted_function([tensor], impl, device)


def _cpu_reshape_broadcast(tensor, device):
    def impl(t):
        reshaped = ttir_cpu.reshape(t, [1, 16, 1, 1])
        return ttir_cpu.broadcast(reshaped, [1, 16, 17, 1])
    return utils.execute_cpu_hoisted_function([tensor], impl, device)


def _cpu_identity(tensor, device):
    def impl(t):
        return t
    return utils.execute_cpu_hoisted_function([tensor], impl, device)


def transpose_weight_to_bf8(tensor, device):
    t = ttnn.from_device(tensor)
    t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=None)
    t = ttnn.to_layout(t, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    t = _cpu_permute_transpose(t, device)
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT16, memory_config=None)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t = _to_device(t, device)
    t = ttnn.from_device(t)
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT8_B, memory_config=None)
    return _to_device(t, device)


def transpose_weight_to_bf16(tensor, device):
    t = ttnn.from_device(tensor)
    t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=None)
    t = ttnn.to_layout(t, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    t = _cpu_permute_transpose(t, device)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t = _to_device(t, device)
    t = ttnn.from_device(t)
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT8_B, memory_config=None)
    return _to_device(t, device)


def qkv_concat_transpose_to_bf8(v_weight, k_weight, q_weight, device):
    inputs = []
    for w in [v_weight, k_weight, q_weight]:
        t = ttnn.from_device(w)
        t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=None)
        t = ttnn.to_layout(t, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
        inputs.append(t)
    t = _cpu_permute_concat_3(inputs[0], inputs[1], inputs[2], device)
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT16, memory_config=None)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t = _to_device(t, device)
    v_slice = ttnn.slice(t, [0, 0], [2880, 128], [1, 1], memory_config=DRAM_MC)
    k_slice = ttnn.slice(t, [0, 128], [2880, 256], [1, 1], memory_config=DRAM_MC)
    q_slice = ttnn.slice(t, [0, 256], [2880, 1280], [1, 1], memory_config=DRAM_MC)
    ttnn.deallocate(t, False)
    t = ttnn.concat([q_slice, v_slice, k_slice], 1, memory_config=DRAM_MC)
    ttnn.deallocate(q_slice, False)
    ttnn.deallocate(k_slice, False)
    ttnn.deallocate(v_slice, False)
    t = ttnn.from_device(t)
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT8_B, memory_config=None)
    return _to_device(t, device)


def typecast_to_bf8_device(tensor, device):
    t = ttnn.from_device(tensor)
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT8_B, memory_config=None)
    return _to_device(t, device)


def qkv_bias_concat(v_bias, k_bias, q_bias, device):
    parts = []
    for bias in [v_bias, k_bias, q_bias]:
        t = _to_device(bias, device)
        t = ttnn.mesh_partition(input_tensor=t, dim=0, cluster_axis=1, memory_config=DRAM_MC)
        t = ttnn.from_device(t)
        t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=None)
        parts.append(t)
    t = _cpu_concat_dim0(parts[0], parts[1], parts[2], device)
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT16, memory_config=None)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t = _to_device(t, device)
    v_slice = ttnn.slice(t, [0], [128], [1], memory_config=DRAM_MC)
    k_slice = ttnn.slice(t, [128], [256], [1], memory_config=DRAM_MC)
    q_slice = ttnn.slice(t, [256], [1280], [1], memory_config=DRAM_MC)
    ttnn.deallocate(t, False)
    t = ttnn.concat([q_slice, v_slice, k_slice], 0, memory_config=DRAM_MC)
    ttnn.deallocate(q_slice, False)
    ttnn.deallocate(k_slice, False)
    ttnn.deallocate(v_slice, False)
    return t


def sinks_to_device(tensor, device):
    t = _to_device(tensor, device)
    t = ttnn.mesh_partition(input_tensor=t, dim=0, cluster_axis=1, memory_config=DRAM_MC)
    t = ttnn.from_device(t)
    t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=None)
    t = _cpu_reshape_broadcast(t, device)
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT16, memory_config=None)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    return _to_device(t, device)


def o_proj_bias_reshape(tensor, device):
    t = ttnn.typecast(tensor, ttnn.DataType.FLOAT32, memory_config=None)
    t = _cpu_reshape(t, [1, 2880], device)
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT16, memory_config=None)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    return _to_device(t, device)


def expert_bias_reshape(tensor, shape, device):
    t = ttnn.from_device(tensor)
    t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=None)
    t = ttnn.to_layout(t, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    t = _cpu_reshape(t, shape, device)
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT16, memory_config=None)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    return _to_device(t, device)


def router_bias_to_device(tensor, device):
    t = ttnn.typecast(tensor, ttnn.DataType.FLOAT32, memory_config=None)
    t = _cpu_identity(t, device)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    return _to_device(t, device)


def build_causal_mask(device):
    def impl():
        zeros = ttir_cpu.zeros(shape=[1, 1, 1, 1], dtype=torch.float32)
        rows = ttir_cpu.constant(
            shape=[1, 1, 17, 1], dtype=torch.int32,
            data=list(range(17)),
        )
        cols = ttir_cpu.constant(
            shape=[1, 1, 1, 17], dtype=torch.int32,
            data=list(range(17)),
        )
        neg_inf = ttir_cpu.full(
            shape=[1, 1, 1, 1], fill_value=-3.38953139e38, dtype=torch.float32
        )
        mask = ttir_cpu.ge(rows, cols)
        return ttir_cpu.where(mask, zeros, neg_inf)
    t = utils.execute_cpu_hoisted_function([], impl, device)
    t = ttnn.typecast(t, ttnn.DataType.BFLOAT16, memory_config=None)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    return _to_device(t, device)


def build_rotary_emb(inv_freq_tensor, device):
    t = ttnn.typecast(inv_freq_tensor, ttnn.DataType.FLOAT32, memory_config=None)

    def impl(inv_freq):
        scale = ttir_cpu.full(shape=[1, 1, 1, 1], fill_value=1.34657359, dtype=torch.float32)
        positions = ttir_cpu.constant(
            shape=[1, 1, 17], dtype=torch.float32,
            data=list(range(17)),
        )
        freq = ttir_cpu.reshape(inv_freq, [1, 32, 1])
        freqs = ttir_cpu.multiply(freq, positions)
        freqs_t = ttir_cpu.permute(freqs, [0, 2, 1])
        freqs_4d = ttir_cpu.reshape(freqs_t, [1, 1, 17, 32])
        cos_val = ttir_cpu.multiply(ttir_cpu.cos(freqs_4d), scale)
        sin_val = ttir_cpu.multiply(ttir_cpu.sin(freqs_4d), scale)
        cos_rep = ttir_cpu.concat([cos_val, cos_val], dim=3)
        sin_rep = ttir_cpu.concat([sin_val, sin_val], dim=3)
        return freqs, cos_rep, sin_rep

    v_freqs, v_cos, v_sin = utils.execute_cpu_hoisted_function([t], impl, device)

    cos_t = ttnn.typecast(v_cos, ttnn.DataType.BFLOAT16, memory_config=None)
    cos_t = ttnn.to_layout(cos_t, ttnn.Layout.TILE, None, memory_config=None)
    cos_t = _to_device(cos_t, device)

    sin_t = ttnn.typecast(v_sin, ttnn.DataType.BFLOAT16, memory_config=None)
    sin_t = ttnn.to_layout(sin_t, ttnn.Layout.TILE, None, memory_config=None)
    sin_t = _to_device(sin_t, device)

    freqs_t = ttnn.to_layout(v_freqs, ttnn.Layout.TILE, None, memory_config=None)
    freqs_t = _to_device(freqs_t, device)

    return cos_t, sin_t, freqs_t


def build_expert_index_offsets(device):
    def impl():
        full_val = ttir_cpu.full(shape=[1, 1, 1], fill_value=32, dtype=torch.int32)
        arange_val = ttir_cpu.arange(0, 17, 1, arange_dimension=0, shape=[17], dtype=torch.int32)
        reshaped = ttir_cpu.reshape(arange_val, [17, 1, 1])
        multiplied = ttir_cpu.multiply(reshaped, full_val)
        return ttir_cpu.broadcast(multiplied, [17, 4, 1])
    t = utils.execute_cpu_hoisted_function([], impl, device)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    return _to_device(t, device)


def run_consteval(weights, device):
    w = weights

    w["model.embed_tokens.parametrizations.weight.original"] = _to_device(
        w["model.embed_tokens.parametrizations.weight.original"], device
    )

    for i in range(2):
        prefix = f"model.layers.{i}"
        w[f"{prefix}.self_attn.qkv_proj.weight"] = qkv_concat_transpose_to_bf8(
            w[f"{prefix}.self_attn.v_proj.parametrizations.weight.original"],
            w[f"{prefix}.self_attn.k_proj.parametrizations.weight.original"],
            w[f"{prefix}.self_attn.q_proj.parametrizations.weight.original"],
            device,
        )
        w[f"{prefix}.self_attn.o_proj.weight_transposed"] = transpose_weight_to_bf8(
            w[f"{prefix}.self_attn.o_proj.parametrizations.weight.original"], device
        )
        w[f"{prefix}.self_attn.qkv_proj.bias"] = qkv_bias_concat(
            w[f"{prefix}.self_attn.v_proj.bias"],
            w[f"{prefix}.self_attn.k_proj.bias"],
            w[f"{prefix}.self_attn.q_proj.bias"],
            device,
        )
        w[f"{prefix}.self_attn.o_proj.bias_reshaped"] = o_proj_bias_reshape(
            w[f"{prefix}.self_attn.o_proj.bias"], device
        )
        w[f"{prefix}.self_attn.sinks_processed"] = sinks_to_device(
            w[f"{prefix}.self_attn.sinks"], device
        )
        w[f"{prefix}.mlp.router.weight_transposed"] = transpose_weight_to_bf8(
            w[f"{prefix}.mlp.router.parametrizations.weight.original"], device
        )
        w[f"{prefix}.mlp.router.bias_processed"] = router_bias_to_device(
            w[f"{prefix}.mlp.router.bias"], device
        )
        w[f"{prefix}.mlp.experts.gate_up_proj_bf8"] = typecast_to_bf8_device(
            w[f"{prefix}.mlp.experts.gate_up_proj"], device
        )
        w[f"{prefix}.mlp.experts.gate_up_proj_bias_reshaped"] = expert_bias_reshape(
            w[f"{prefix}.mlp.experts.gate_up_proj_bias"], [32, 1, 5760], device
        )
        w[f"{prefix}.mlp.experts.down_proj_bf8"] = typecast_to_bf8_device(
            w[f"{prefix}.mlp.experts.down_proj"], device
        )
        w[f"{prefix}.mlp.experts.down_proj_bias_reshaped"] = expert_bias_reshape(
            w[f"{prefix}.mlp.experts.down_proj_bias"], [32, 1, 2880], device
        )

    w["lm_head.weight_transposed"] = transpose_weight_to_bf8(
        w["lm_head.parametrizations.weight.original"], device
    )

    w["moe_scatter_zeros"] = ttnn.zeros(
        shape=ttnn.Shape([544]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=device,
        memory_config=DRAM_MC,
    )
    w["causal_mask"] = build_causal_mask(device)
    cos_t, sin_t, freqs_t = build_rotary_emb(w["model.rotary_emb.inv_freq"], device)
    w["rotary_emb.cos"] = cos_t
    w["rotary_emb.sin"] = sin_t
    w["rotary_emb.freqs"] = freqs_t
    w["ones_scalar"] = ttnn.ones(
        shape=ttnn.Shape([1, 1, 1]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MC,
    )
    w["expert_index_offsets"] = build_expert_index_offsets(device)
    w["attn_scale"] = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=0.125,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MC,
    )
    w["rms_norm_epsilon"] = ttnn.full(
        shape=ttnn.Shape([1, 1, 1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MC,
    )
    w["sigmoid_scale"] = ttnn.full(
        shape=ttnn.Shape([1, 1, 1]),
        fill_value=1.703125,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MC,
    )
    w["moe_scatter_zeros_2d"] = ttnn.zeros(
        shape=ttnn.Shape([17, 32]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM_MC,
    )

    return w
