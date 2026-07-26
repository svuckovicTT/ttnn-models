# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import utils


_main_weights = {}


def load_weights_for__main():
    utils_DeviceGetter_get_device_34 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    global _main_weights
    utils_load_tensor_1 = utils.load_tensor(
        "./tensors/arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.0.self_attn.v_proj.bias"] = utils_load_tensor_1
    utils_load_tensor_2 = utils.load_tensor(
        "./tensors/arg1.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.0.self_attn.v_proj.parametrizations.weight.original"
    ] = utils_load_tensor_2
    utils_load_tensor_3 = utils.load_tensor(
        "./tensors/arg2.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.0.input_layernorm.parametrizations.weight.original"] = (
        utils_load_tensor_3
    )
    utils_load_tensor_4 = utils.load_tensor(
        "./tensors/arg4.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.embed_tokens.parametrizations.weight.original"] = (
        utils_load_tensor_4
    )
    utils_load_tensor_5 = utils.load_tensor(
        "./tensors/arg5.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.rotary_emb.inv_freq"] = utils_load_tensor_5
    utils_load_tensor_6 = utils.load_tensor(
        "./tensors/arg6.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.0.self_attn.k_proj.bias"] = utils_load_tensor_6
    utils_load_tensor_7 = utils.load_tensor(
        "./tensors/arg7.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.0.self_attn.k_proj.parametrizations.weight.original"
    ] = utils_load_tensor_7
    utils_load_tensor_8 = utils.load_tensor(
        "./tensors/arg8.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.1.self_attn.v_proj.bias"] = utils_load_tensor_8
    utils_load_tensor_9 = utils.load_tensor(
        "./tensors/arg9.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.1.self_attn.v_proj.parametrizations.weight.original"
    ] = utils_load_tensor_9
    utils_load_tensor_10 = utils.load_tensor(
        "./tensors/arg10.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.1.input_layernorm.parametrizations.weight.original"] = (
        utils_load_tensor_10
    )
    utils_load_tensor_11 = utils.load_tensor(
        "./tensors/arg11.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.0.mlp.router.bias"] = utils_load_tensor_11
    utils_load_tensor_12 = utils.load_tensor(
        "./tensors/arg12.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.0.mlp.router.parametrizations.weight.original"] = (
        utils_load_tensor_12
    )
    utils_load_tensor_13 = utils.load_tensor(
        "./tensors/arg13.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.0.post_attention_layernorm.parametrizations.weight.original"
    ] = utils_load_tensor_13
    utils_load_tensor_14 = utils.load_tensor(
        "./tensors/arg14.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.0.self_attn.o_proj.bias"] = utils_load_tensor_14
    utils_load_tensor_15 = utils.load_tensor(
        "./tensors/arg15.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.0.self_attn.o_proj.parametrizations.weight.original"
    ] = utils_load_tensor_15
    utils_load_tensor_16 = utils.load_tensor(
        "./tensors/arg16.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.0.self_attn.sinks"] = utils_load_tensor_16
    utils_load_tensor_17 = utils.load_tensor(
        "./tensors/arg17.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.0.self_attn.q_proj.bias"] = utils_load_tensor_17
    utils_load_tensor_18 = utils.load_tensor(
        "./tensors/arg18.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.0.self_attn.q_proj.parametrizations.weight.original"
    ] = utils_load_tensor_18
    utils_load_tensor_19 = utils.load_tensor(
        "./tensors/arg19.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.0.mlp.experts.down_proj_bias"] = utils_load_tensor_19
    utils_load_tensor_20 = utils.load_tensor(
        "./tensors/arg20.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.0.mlp.experts.down_proj"] = utils_load_tensor_20
    utils_load_tensor_21 = utils.load_tensor(
        "./tensors/arg21.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.0.mlp.experts.gate_up_proj_bias"] = utils_load_tensor_21
    utils_load_tensor_22 = utils.load_tensor(
        "./tensors/arg22.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.0.mlp.experts.gate_up_proj"] = utils_load_tensor_22
    utils_load_tensor_23 = utils.load_tensor(
        "./tensors/arg23.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.1.self_attn.k_proj.bias"] = utils_load_tensor_23
    utils_load_tensor_24 = utils.load_tensor(
        "./tensors/arg24.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.1.self_attn.k_proj.parametrizations.weight.original"
    ] = utils_load_tensor_24
    utils_load_tensor_25 = utils.load_tensor(
        "./tensors/arg25.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["lm_head.parametrizations.weight.original"] = utils_load_tensor_25
    utils_load_tensor_26 = utils.load_tensor(
        "./tensors/arg26.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.norm.parametrizations.weight.original"] = utils_load_tensor_26
    utils_load_tensor_27 = utils.load_tensor(
        "./tensors/arg27.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.1.mlp.router.bias"] = utils_load_tensor_27
    utils_load_tensor_28 = utils.load_tensor(
        "./tensors/arg28.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.1.mlp.router.parametrizations.weight.original"] = (
        utils_load_tensor_28
    )
    utils_load_tensor_29 = utils.load_tensor(
        "./tensors/arg29.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.1.post_attention_layernorm.parametrizations.weight.original"
    ] = utils_load_tensor_29
    utils_load_tensor_30 = utils.load_tensor(
        "./tensors/arg30.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.1.self_attn.o_proj.bias"] = utils_load_tensor_30
    utils_load_tensor_31 = utils.load_tensor(
        "./tensors/arg31.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.1.self_attn.o_proj.parametrizations.weight.original"
    ] = utils_load_tensor_31
    utils_load_tensor_32 = utils.load_tensor(
        "./tensors/arg32.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.1.self_attn.sinks"] = utils_load_tensor_32
    utils_load_tensor_33 = utils.load_tensor(
        "./tensors/arg33.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.1.self_attn.q_proj.bias"] = utils_load_tensor_33
    utils_load_tensor_34 = utils.load_tensor(
        "./tensors/arg34.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.1.self_attn.q_proj.parametrizations.weight.original"
    ] = utils_load_tensor_34
    utils_load_tensor_35 = utils.load_tensor(
        "./tensors/arg35.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.1.mlp.experts.down_proj_bias"] = utils_load_tensor_35
    utils_load_tensor_36 = utils.load_tensor(
        "./tensors/arg36.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.1.mlp.experts.down_proj"] = utils_load_tensor_36
    utils_load_tensor_37 = utils.load_tensor(
        "./tensors/arg37.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.1.mlp.experts.gate_up_proj_bias"] = utils_load_tensor_37
    utils_load_tensor_38 = utils.load_tensor(
        "./tensors/arg38.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.1.mlp.experts.gate_up_proj"] = utils_load_tensor_38
    return _main_weights


HOST_WEIGHTS = {
    "model.layers.0.self_attn.v_proj.bias",
    "model.layers.0.self_attn.k_proj.bias",
    "model.layers.0.self_attn.q_proj.bias",
    "model.layers.0.self_attn.o_proj.bias",
    "model.layers.0.self_attn.sinks",
    "model.layers.0.mlp.router.bias",
    "model.layers.1.self_attn.v_proj.bias",
    "model.layers.1.self_attn.k_proj.bias",
    "model.layers.1.self_attn.q_proj.bias",
    "model.layers.1.self_attn.o_proj.bias",
    "model.layers.1.self_attn.sinks",
    "model.layers.1.mlp.router.bias",
    "model.embed_tokens.parametrizations.weight.original",
    "model.rotary_emb.inv_freq",
}

DEVICE_WEIGHTS = {
    "model.layers.0.self_attn.v_proj.parametrizations.weight.original",
    "model.layers.0.self_attn.k_proj.parametrizations.weight.original",
    "model.layers.0.self_attn.q_proj.parametrizations.weight.original",
    "model.layers.0.self_attn.o_proj.parametrizations.weight.original",
    "model.layers.0.input_layernorm.parametrizations.weight.original",
    "model.layers.0.post_attention_layernorm.parametrizations.weight.original",
    "model.layers.0.mlp.router.parametrizations.weight.original",
    "model.layers.0.mlp.experts.gate_up_proj",
    "model.layers.0.mlp.experts.gate_up_proj_bias",
    "model.layers.0.mlp.experts.down_proj",
    "model.layers.0.mlp.experts.down_proj_bias",
    "model.layers.1.self_attn.v_proj.parametrizations.weight.original",
    "model.layers.1.self_attn.k_proj.parametrizations.weight.original",
    "model.layers.1.self_attn.q_proj.parametrizations.weight.original",
    "model.layers.1.self_attn.o_proj.parametrizations.weight.original",
    "model.layers.1.input_layernorm.parametrizations.weight.original",
    "model.layers.1.post_attention_layernorm.parametrizations.weight.original",
    "model.layers.1.mlp.router.parametrizations.weight.original",
    "model.layers.1.mlp.experts.gate_up_proj",
    "model.layers.1.mlp.experts.gate_up_proj_bias",
    "model.layers.1.mlp.experts.down_proj",
    "model.layers.1.mlp.experts.down_proj_bias",
    "lm_head.parametrizations.weight.original",
    "model.norm.parametrizations.weight.original",
}

ALL_WEIGHTS = HOST_WEIGHTS | DEVICE_WEIGHTS


def load_weights_for__main_from_state_dict():
    import model_pt

    device = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )

    model = model_pt.load_pytorch_model()
    sd = dict(model.state_dict())
    for name, buf in model.named_buffers():
        if name not in sd:
            sd[name] = buf

    weights = {}
    for key in ALL_WEIGHTS:
        pt_tensor = sd[key]
        ttnn_tensor = ttnn.from_torch(pt_tensor)

        if key in HOST_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)

        if key in DEVICE_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.TILE)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)
            ttnn_tensor = ttnn.to_device(ttnn_tensor, device, ttnn.DRAM_MEMORY_CONFIG)

        weights[key] = ttnn_tensor

    return weights
