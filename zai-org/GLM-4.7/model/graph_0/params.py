import ttnn
import utils


NORM_WEIGHTS = {
    f"model.model.layers.{i}.{s}"
    for i in range(4)
    for s in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.k_norm.weight",
        "self_attn.q_norm.weight",
    ]
} | {
    "model.model.norm.weight",
}

INT32_WEIGHTS = {
    "model.model.layers.3.mlp.mlp.expert_mapping",
}

_ATTN_WEIGHT_SUFFIXES = [
    "self_attn.k_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.q_proj.weight",
    "self_attn.o_proj.weight",
]

_DENSE_MLP_WEIGHT_SUFFIXES = [
    "mlp.down_proj.weight",
    "mlp.up_proj.weight",
    "mlp.gate_proj.weight",
]

ROW_MAJOR_BF16_WEIGHTS = (
    {
        f"model.model.layers.{i}.{s}"
        for i in range(4)
        for s in _ATTN_WEIGHT_SUFFIXES
    }
    | {
        f"model.model.layers.{i}.{s}"
        for i in range(3)
        for s in _DENSE_MLP_WEIGHT_SUFFIXES
    }
    | {
        "model.lm_head.weight",
        "model.model.embed_tokens.weight",
        "model.model.rotary_emb.inv_freq",
        "model.model.layers.3.mlp.shared_experts.down_proj.weight",
        "model.model.layers.3.mlp.shared_experts.up_proj.weight",
        "model.model.layers.3.mlp.shared_experts.gate_proj.weight",
        "model.model.layers.3.mlp.mlp.router.gate.weight",
        "model.model.layers.3.mlp.mlp.experts.down_proj",
        "model.model.layers.3.mlp.mlp.experts.up_proj",
        "model.model.layers.3.mlp.mlp.experts.gate_proj",
        "L__self___model_model_layers_3_mlp_mlp_router__route_fn___closure___0_cell_contents_e_score_correction_bias",
    }
)

ALL_WEIGHTS = NORM_WEIGHTS | INT32_WEIGHTS | ROW_MAJOR_BF16_WEIGHTS


_main_weights = {}


def load_weights_for__main():
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((4, 8))
    global _main_weights
    utils_load_tensor_14 = utils.load_tensor(
        "./tensors/arg0.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.model.layers.0.self_attn.k_norm.weight"] = utils_load_tensor_14
    utils_load_tensor_15 = utils.load_tensor(
        "./tensors/arg1.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.0.self_attn.k_proj.bias"] = utils_load_tensor_15
    utils_load_tensor_16 = utils.load_tensor(
        "./tensors/arg2.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.0.self_attn.k_proj.weight"] = utils_load_tensor_16
    utils_load_tensor_17 = utils.load_tensor(
        "./tensors/arg3.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.model.layers.0.input_layernorm.weight"] = utils_load_tensor_17
    utils_load_tensor_18 = utils.load_tensor(
        "./tensors/arg5.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.embed_tokens.weight"] = utils_load_tensor_18
    utils_load_tensor_19 = utils.load_tensor(
        "./tensors/arg7.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.rotary_emb.inv_freq"] = utils_load_tensor_19
    utils_load_tensor_20 = utils.load_tensor(
        "./tensors/arg10.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.0.self_attn.v_proj.bias"] = utils_load_tensor_20
    utils_load_tensor_21 = utils.load_tensor(
        "./tensors/arg11.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.0.self_attn.v_proj.weight"] = utils_load_tensor_21
    utils_load_tensor_22 = utils.load_tensor(
        "./tensors/arg13.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.model.layers.1.self_attn.k_norm.weight"] = utils_load_tensor_22
    utils_load_tensor_23 = utils.load_tensor(
        "./tensors/arg14.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.1.self_attn.k_proj.bias"] = utils_load_tensor_23
    utils_load_tensor_24 = utils.load_tensor(
        "./tensors/arg15.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.1.self_attn.k_proj.weight"] = utils_load_tensor_24
    utils_load_tensor_25 = utils.load_tensor(
        "./tensors/arg16.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.model.layers.1.input_layernorm.weight"] = utils_load_tensor_25
    utils_load_tensor_26 = utils.load_tensor(
        "./tensors/arg17.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.0.mlp.down_proj.weight"] = utils_load_tensor_26
    utils_load_tensor_27 = utils.load_tensor(
        "./tensors/arg18.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.0.mlp.up_proj.weight"] = utils_load_tensor_27
    utils_load_tensor_28 = utils.load_tensor(
        "./tensors/arg19.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.model.layers.0.post_attention_layernorm.weight"] = (
        utils_load_tensor_28
    )
    utils_load_tensor_29 = utils.load_tensor(
        "./tensors/arg20.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.0.self_attn.o_proj.weight"] = utils_load_tensor_29
    utils_load_tensor_30 = utils.load_tensor(
        "./tensors/arg21.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.model.layers.0.self_attn.q_norm.weight"] = utils_load_tensor_30
    utils_load_tensor_31 = utils.load_tensor(
        "./tensors/arg22.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.0.self_attn.q_proj.bias"] = utils_load_tensor_31
    utils_load_tensor_32 = utils.load_tensor(
        "./tensors/arg23.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.0.self_attn.q_proj.weight"] = utils_load_tensor_32
    utils_load_tensor_33 = utils.load_tensor(
        "./tensors/arg24.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.0.mlp.gate_proj.weight"] = utils_load_tensor_33
    utils_load_tensor_34 = utils.load_tensor(
        "./tensors/arg27.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.1.self_attn.v_proj.bias"] = utils_load_tensor_34
    utils_load_tensor_35 = utils.load_tensor(
        "./tensors/arg28.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.1.self_attn.v_proj.weight"] = utils_load_tensor_35
    utils_load_tensor_36 = utils.load_tensor(
        "./tensors/arg30.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.model.layers.2.self_attn.k_norm.weight"] = utils_load_tensor_36
    utils_load_tensor_37 = utils.load_tensor(
        "./tensors/arg31.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.2.self_attn.k_proj.bias"] = utils_load_tensor_37
    utils_load_tensor_38 = utils.load_tensor(
        "./tensors/arg32.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.2.self_attn.k_proj.weight"] = utils_load_tensor_38
    utils_load_tensor_39 = utils.load_tensor(
        "./tensors/arg33.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.model.layers.2.input_layernorm.weight"] = utils_load_tensor_39
    utils_load_tensor_40 = utils.load_tensor(
        "./tensors/arg34.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.1.mlp.down_proj.weight"] = utils_load_tensor_40
    utils_load_tensor_41 = utils.load_tensor(
        "./tensors/arg35.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.1.mlp.up_proj.weight"] = utils_load_tensor_41
    utils_load_tensor_42 = utils.load_tensor(
        "./tensors/arg36.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.model.layers.1.post_attention_layernorm.weight"] = (
        utils_load_tensor_42
    )
    utils_load_tensor_43 = utils.load_tensor(
        "./tensors/arg37.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.1.self_attn.o_proj.weight"] = utils_load_tensor_43
    utils_load_tensor_44 = utils.load_tensor(
        "./tensors/arg38.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.model.layers.1.self_attn.q_norm.weight"] = utils_load_tensor_44
    utils_load_tensor_45 = utils.load_tensor(
        "./tensors/arg39.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.1.self_attn.q_proj.bias"] = utils_load_tensor_45
    utils_load_tensor_46 = utils.load_tensor(
        "./tensors/arg40.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.1.self_attn.q_proj.weight"] = utils_load_tensor_46
    utils_load_tensor_47 = utils.load_tensor(
        "./tensors/arg41.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.1.mlp.gate_proj.weight"] = utils_load_tensor_47
    utils_load_tensor_48 = utils.load_tensor(
        "./tensors/arg44.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.2.self_attn.v_proj.bias"] = utils_load_tensor_48
    utils_load_tensor_49 = utils.load_tensor(
        "./tensors/arg45.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.2.self_attn.v_proj.weight"] = utils_load_tensor_49
    utils_load_tensor_50 = utils.load_tensor(
        "./tensors/arg47.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.model.layers.3.self_attn.k_norm.weight"] = utils_load_tensor_50
    utils_load_tensor_51 = utils.load_tensor(
        "./tensors/arg48.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.3.self_attn.k_proj.bias"] = utils_load_tensor_51
    utils_load_tensor_52 = utils.load_tensor(
        "./tensors/arg49.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.3.self_attn.k_proj.weight"] = utils_load_tensor_52
    utils_load_tensor_53 = utils.load_tensor(
        "./tensors/arg50.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.model.layers.3.input_layernorm.weight"] = utils_load_tensor_53
    utils_load_tensor_54 = utils.load_tensor(
        "./tensors/arg51.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.2.mlp.down_proj.weight"] = utils_load_tensor_54
    utils_load_tensor_55 = utils.load_tensor(
        "./tensors/arg52.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.2.mlp.up_proj.weight"] = utils_load_tensor_55
    utils_load_tensor_56 = utils.load_tensor(
        "./tensors/arg53.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.model.layers.2.post_attention_layernorm.weight"] = (
        utils_load_tensor_56
    )
    utils_load_tensor_57 = utils.load_tensor(
        "./tensors/arg54.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.2.self_attn.o_proj.weight"] = utils_load_tensor_57
    utils_load_tensor_58 = utils.load_tensor(
        "./tensors/arg55.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.model.layers.2.self_attn.q_norm.weight"] = utils_load_tensor_58
    utils_load_tensor_59 = utils.load_tensor(
        "./tensors/arg56.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.2.self_attn.q_proj.bias"] = utils_load_tensor_59
    utils_load_tensor_60 = utils.load_tensor(
        "./tensors/arg57.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.2.self_attn.q_proj.weight"] = utils_load_tensor_60
    utils_load_tensor_61 = utils.load_tensor(
        "./tensors/arg58.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.2.mlp.gate_proj.weight"] = utils_load_tensor_61
    utils_load_tensor_62 = utils.load_tensor(
        "./tensors/arg61.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.3.self_attn.v_proj.bias"] = utils_load_tensor_62
    utils_load_tensor_63 = utils.load_tensor(
        "./tensors/arg62.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.3.self_attn.v_proj.weight"] = utils_load_tensor_63
    utils_load_tensor_64 = utils.load_tensor(
        "./tensors/arg64.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.lm_head.weight"] = utils_load_tensor_64
    utils_load_tensor_65 = utils.load_tensor(
        "./tensors/arg65.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.model.norm.weight"] = utils_load_tensor_65
    utils_load_tensor_66 = utils.load_tensor(
        "./tensors/arg66.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.3.mlp.shared_experts.down_proj.weight"] = (
        utils_load_tensor_66
    )
    utils_load_tensor_67 = utils.load_tensor(
        "./tensors/arg67.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.3.mlp.shared_experts.up_proj.weight"] = (
        utils_load_tensor_67
    )
    utils_load_tensor_68 = utils.load_tensor(
        "./tensors/arg68.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.model.layers.3.post_attention_layernorm.weight"] = (
        utils_load_tensor_68
    )
    utils_load_tensor_69 = utils.load_tensor(
        "./tensors/arg69.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.3.self_attn.o_proj.weight"] = utils_load_tensor_69
    utils_load_tensor_70 = utils.load_tensor(
        "./tensors/arg70.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.model.layers.3.self_attn.q_norm.weight"] = utils_load_tensor_70
    utils_load_tensor_71 = utils.load_tensor(
        "./tensors/arg71.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.3.self_attn.q_proj.bias"] = utils_load_tensor_71
    utils_load_tensor_72 = utils.load_tensor(
        "./tensors/arg72.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.3.self_attn.q_proj.weight"] = utils_load_tensor_72
    utils_load_tensor_73 = utils.load_tensor(
        "./tensors/arg73.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.3.mlp.shared_experts.gate_proj.weight"] = (
        utils_load_tensor_73
    )
    utils_load_tensor_74 = utils.load_tensor(
        "./tensors/arg74.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "L__self___model_model_layers_3_mlp_mlp_router__route_fn___closure___0_cell_contents_e_score_correction_bias"
    ] = utils_load_tensor_74
    utils_load_tensor_75 = utils.load_tensor(
        "./tensors/arg75.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.3.mlp.mlp.router.gate.weight"] = (
        utils_load_tensor_75
    )
    utils_load_tensor_76 = utils.load_tensor(
        "./tensors/arg76.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        None,
        None,
    )
    _main_weights["model.model.layers.3.mlp.mlp.expert_mapping"] = utils_load_tensor_76
    utils_load_tensor_77 = utils.load_tensor(
        "./tensors/arg77.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.3.mlp.mlp.experts.down_proj"] = (
        utils_load_tensor_77
    )
    utils_load_tensor_78 = utils.load_tensor(
        "./tensors/arg78.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.3.mlp.mlp.experts.up_proj"] = utils_load_tensor_78
    utils_load_tensor_79 = utils.load_tensor(
        "./tensors/arg79.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.model.layers.3.mlp.mlp.experts.gate_proj"] = (
        utils_load_tensor_79
    )
    return _main_weights


def load_weights_for__main_from_state_dict():
    import model_pt

    model = model_pt.load_pytorch_model()

    sd = {}
    for k, v in model.state_dict().items():
        sd[f"model.{k}"] = v
    for name, buf in model.named_buffers():
        prefixed = f"model.{name}"
        if prefixed not in sd:
            sd[prefixed] = buf

    for prefix in [
        "model.model.layers.3.mlp.mlp.experts",
        "model.model.layers.3.mlp.experts",
    ]:
        gate_up_key = f"{prefix}.gate_up_proj"
        if gate_up_key in sd:
            gate_up = sd.pop(gate_up_key)
            half = gate_up.shape[1] // 2
            target = "model.model.layers.3.mlp.mlp.experts"
            sd[f"{target}.gate_proj"] = gate_up[:, :half, :].contiguous()
            sd[f"{target}.up_proj"] = gate_up[:, half:, :].contiguous()
            break

    _E_SCORE_MANGLED_KEY = (
        "L__self___model_model_layers_3_mlp_mlp_router"
        "__route_fn___closure___0_cell_contents_e_score_correction_bias"
    )
    for candidate in [
        "model.model.layers.3.mlp.mlp.router.e_score_correction_bias",
        "model.model.layers.3.mlp.router.e_score_correction_bias",
    ]:
        if candidate in sd:
            sd[_E_SCORE_MANGLED_KEY] = sd.pop(candidate)
            break

    for key in ALL_WEIGHTS:
        if key not in sd and ".mlp.mlp." in key:
            single_mlp_key = key.replace(".mlp.mlp.", ".mlp.", 1)
            if single_mlp_key in sd:
                sd[key] = sd.pop(single_mlp_key)

    device = utils.DeviceGetter.get_device((4, 8))

    weights = {}
    for key in ALL_WEIGHTS:
        pt_tensor = sd[key]
        ttnn_tensor = ttnn.from_torch(pt_tensor)

        if key in NORM_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.TILE)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)
            ttnn_tensor = ttnn.to_device(ttnn_tensor, device, ttnn.DRAM_MEMORY_CONFIG)
        elif key in INT32_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.INT32)
        else:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)

        weights[key] = ttnn_tensor

    return weights
