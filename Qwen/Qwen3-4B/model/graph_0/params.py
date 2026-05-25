# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import model_pt
import utils


_main_weights = {}


def load_weights_for__main():
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    global _main_weights
    utils_load_tensor_1 = utils.load_tensor(
        "./tensors/arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_0_self_attn_v_proj.weight"] = (
        utils_load_tensor_1
    )
    utils_load_tensor_2 = utils.load_tensor(
        "./tensors/arg1.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_0_input_layernorm_weight"] = (
        utils_load_tensor_2
    )
    utils_load_tensor_3 = utils.load_tensor(
        "./tensors/arg3.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_embed_tokens.weight"] = utils_load_tensor_3
    utils_load_tensor_4 = utils.load_tensor(
        "./tensors/arg4.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    _main_weights["L__self___model_rotary_emb_inv_freq"] = utils_load_tensor_4
    utils_load_tensor_5 = utils.load_tensor(
        "./tensors/arg5.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_0_self_attn_k_norm_weight"] = (
        utils_load_tensor_5
    )
    utils_load_tensor_6 = utils.load_tensor(
        "./tensors/arg6.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_0_self_attn_k_proj.weight"] = (
        utils_load_tensor_6
    )
    utils_load_tensor_7 = utils.load_tensor(
        "./tensors/arg7.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_1_self_attn_v_proj.weight"] = (
        utils_load_tensor_7
    )
    utils_load_tensor_8 = utils.load_tensor(
        "./tensors/arg8.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_1_input_layernorm_weight"] = (
        utils_load_tensor_8
    )
    utils_load_tensor_9 = utils.load_tensor(
        "./tensors/arg9.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_0_mlp_down_proj.weight"] = utils_load_tensor_9
    utils_load_tensor_10 = utils.load_tensor(
        "./tensors/arg10.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_0_mlp_up_proj.weight"] = utils_load_tensor_10
    utils_load_tensor_11 = utils.load_tensor(
        "./tensors/arg11.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_0_post_attention_layernorm_weight"] = (
        utils_load_tensor_11
    )
    utils_load_tensor_12 = utils.load_tensor(
        "./tensors/arg12.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_0_self_attn_o_proj.weight"] = (
        utils_load_tensor_12
    )
    utils_load_tensor_13 = utils.load_tensor(
        "./tensors/arg13.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_0_self_attn_q_norm_weight"] = (
        utils_load_tensor_13
    )
    utils_load_tensor_14 = utils.load_tensor(
        "./tensors/arg14.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_0_self_attn_q_proj.weight"] = (
        utils_load_tensor_14
    )
    utils_load_tensor_15 = utils.load_tensor(
        "./tensors/arg15.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_0_mlp_gate_proj.weight"] = (
        utils_load_tensor_15
    )
    utils_load_tensor_16 = utils.load_tensor(
        "./tensors/arg16.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_1_self_attn_k_norm_weight"] = (
        utils_load_tensor_16
    )
    utils_load_tensor_17 = utils.load_tensor(
        "./tensors/arg17.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_1_self_attn_k_proj.weight"] = (
        utils_load_tensor_17
    )
    utils_load_tensor_18 = utils.load_tensor(
        "./tensors/arg18.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_2_self_attn_v_proj.weight"] = (
        utils_load_tensor_18
    )
    utils_load_tensor_19 = utils.load_tensor(
        "./tensors/arg19.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_2_input_layernorm_weight"] = (
        utils_load_tensor_19
    )
    utils_load_tensor_20 = utils.load_tensor(
        "./tensors/arg20.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_1_mlp_down_proj.weight"] = (
        utils_load_tensor_20
    )
    utils_load_tensor_21 = utils.load_tensor(
        "./tensors/arg21.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_1_mlp_up_proj.weight"] = utils_load_tensor_21
    utils_load_tensor_22 = utils.load_tensor(
        "./tensors/arg22.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_1_post_attention_layernorm_weight"] = (
        utils_load_tensor_22
    )
    utils_load_tensor_23 = utils.load_tensor(
        "./tensors/arg23.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_1_self_attn_o_proj.weight"] = (
        utils_load_tensor_23
    )
    utils_load_tensor_24 = utils.load_tensor(
        "./tensors/arg24.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_1_self_attn_q_norm_weight"] = (
        utils_load_tensor_24
    )
    utils_load_tensor_25 = utils.load_tensor(
        "./tensors/arg25.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_1_self_attn_q_proj.weight"] = (
        utils_load_tensor_25
    )
    utils_load_tensor_26 = utils.load_tensor(
        "./tensors/arg26.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_1_mlp_gate_proj.weight"] = (
        utils_load_tensor_26
    )
    utils_load_tensor_27 = utils.load_tensor(
        "./tensors/arg27.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_2_self_attn_k_norm_weight"] = (
        utils_load_tensor_27
    )
    utils_load_tensor_28 = utils.load_tensor(
        "./tensors/arg28.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_2_self_attn_k_proj.weight"] = (
        utils_load_tensor_28
    )
    utils_load_tensor_29 = utils.load_tensor(
        "./tensors/arg29.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_3_self_attn_v_proj.weight"] = (
        utils_load_tensor_29
    )
    utils_load_tensor_30 = utils.load_tensor(
        "./tensors/arg30.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_3_input_layernorm_weight"] = (
        utils_load_tensor_30
    )
    utils_load_tensor_31 = utils.load_tensor(
        "./tensors/arg31.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_2_mlp_down_proj.weight"] = (
        utils_load_tensor_31
    )
    utils_load_tensor_32 = utils.load_tensor(
        "./tensors/arg32.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_2_mlp_up_proj.weight"] = utils_load_tensor_32
    utils_load_tensor_33 = utils.load_tensor(
        "./tensors/arg33.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_2_post_attention_layernorm_weight"] = (
        utils_load_tensor_33
    )
    utils_load_tensor_34 = utils.load_tensor(
        "./tensors/arg34.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_2_self_attn_o_proj.weight"] = (
        utils_load_tensor_34
    )
    utils_load_tensor_35 = utils.load_tensor(
        "./tensors/arg35.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_2_self_attn_q_norm_weight"] = (
        utils_load_tensor_35
    )
    utils_load_tensor_36 = utils.load_tensor(
        "./tensors/arg36.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_2_self_attn_q_proj.weight"] = (
        utils_load_tensor_36
    )
    utils_load_tensor_37 = utils.load_tensor(
        "./tensors/arg37.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_2_mlp_gate_proj.weight"] = (
        utils_load_tensor_37
    )
    utils_load_tensor_38 = utils.load_tensor(
        "./tensors/arg38.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_3_self_attn_k_norm_weight"] = (
        utils_load_tensor_38
    )
    utils_load_tensor_39 = utils.load_tensor(
        "./tensors/arg39.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_3_self_attn_k_proj.weight"] = (
        utils_load_tensor_39
    )
    utils_load_tensor_40 = utils.load_tensor(
        "./tensors/arg40.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_4_self_attn_v_proj.weight"] = (
        utils_load_tensor_40
    )
    utils_load_tensor_41 = utils.load_tensor(
        "./tensors/arg41.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_4_input_layernorm_weight"] = (
        utils_load_tensor_41
    )
    utils_load_tensor_42 = utils.load_tensor(
        "./tensors/arg42.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_3_mlp_down_proj.weight"] = (
        utils_load_tensor_42
    )
    utils_load_tensor_43 = utils.load_tensor(
        "./tensors/arg43.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_3_mlp_up_proj.weight"] = utils_load_tensor_43
    utils_load_tensor_44 = utils.load_tensor(
        "./tensors/arg44.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_3_post_attention_layernorm_weight"] = (
        utils_load_tensor_44
    )
    utils_load_tensor_45 = utils.load_tensor(
        "./tensors/arg45.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_3_self_attn_o_proj.weight"] = (
        utils_load_tensor_45
    )
    utils_load_tensor_46 = utils.load_tensor(
        "./tensors/arg46.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_3_self_attn_q_norm_weight"] = (
        utils_load_tensor_46
    )
    utils_load_tensor_47 = utils.load_tensor(
        "./tensors/arg47.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_3_self_attn_q_proj.weight"] = (
        utils_load_tensor_47
    )
    utils_load_tensor_48 = utils.load_tensor(
        "./tensors/arg48.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_3_mlp_gate_proj.weight"] = (
        utils_load_tensor_48
    )
    utils_load_tensor_49 = utils.load_tensor(
        "./tensors/arg49.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_4_self_attn_k_norm_weight"] = (
        utils_load_tensor_49
    )
    utils_load_tensor_50 = utils.load_tensor(
        "./tensors/arg50.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_4_self_attn_k_proj.weight"] = (
        utils_load_tensor_50
    )
    utils_load_tensor_51 = utils.load_tensor(
        "./tensors/arg51.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_5_self_attn_v_proj.weight"] = (
        utils_load_tensor_51
    )
    utils_load_tensor_52 = utils.load_tensor(
        "./tensors/arg52.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_5_input_layernorm_weight"] = (
        utils_load_tensor_52
    )
    utils_load_tensor_53 = utils.load_tensor(
        "./tensors/arg53.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_4_mlp_down_proj.weight"] = (
        utils_load_tensor_53
    )
    utils_load_tensor_54 = utils.load_tensor(
        "./tensors/arg54.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_4_mlp_up_proj.weight"] = utils_load_tensor_54
    utils_load_tensor_55 = utils.load_tensor(
        "./tensors/arg55.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_4_post_attention_layernorm_weight"] = (
        utils_load_tensor_55
    )
    utils_load_tensor_56 = utils.load_tensor(
        "./tensors/arg56.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_4_self_attn_o_proj.weight"] = (
        utils_load_tensor_56
    )
    utils_load_tensor_57 = utils.load_tensor(
        "./tensors/arg57.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_4_self_attn_q_norm_weight"] = (
        utils_load_tensor_57
    )
    utils_load_tensor_58 = utils.load_tensor(
        "./tensors/arg58.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_4_self_attn_q_proj.weight"] = (
        utils_load_tensor_58
    )
    utils_load_tensor_59 = utils.load_tensor(
        "./tensors/arg59.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_4_mlp_gate_proj.weight"] = (
        utils_load_tensor_59
    )
    utils_load_tensor_60 = utils.load_tensor(
        "./tensors/arg60.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_5_self_attn_k_norm_weight"] = (
        utils_load_tensor_60
    )
    utils_load_tensor_61 = utils.load_tensor(
        "./tensors/arg61.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_5_self_attn_k_proj.weight"] = (
        utils_load_tensor_61
    )
    utils_load_tensor_62 = utils.load_tensor(
        "./tensors/arg62.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_6_self_attn_v_proj.weight"] = (
        utils_load_tensor_62
    )
    utils_load_tensor_63 = utils.load_tensor(
        "./tensors/arg63.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_6_input_layernorm_weight"] = (
        utils_load_tensor_63
    )
    utils_load_tensor_64 = utils.load_tensor(
        "./tensors/arg64.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_5_mlp_down_proj.weight"] = (
        utils_load_tensor_64
    )
    utils_load_tensor_65 = utils.load_tensor(
        "./tensors/arg65.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_5_mlp_up_proj.weight"] = utils_load_tensor_65
    utils_load_tensor_66 = utils.load_tensor(
        "./tensors/arg66.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_5_post_attention_layernorm_weight"] = (
        utils_load_tensor_66
    )
    utils_load_tensor_67 = utils.load_tensor(
        "./tensors/arg67.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_5_self_attn_o_proj.weight"] = (
        utils_load_tensor_67
    )
    utils_load_tensor_68 = utils.load_tensor(
        "./tensors/arg68.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_5_self_attn_q_norm_weight"] = (
        utils_load_tensor_68
    )
    utils_load_tensor_69 = utils.load_tensor(
        "./tensors/arg69.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_5_self_attn_q_proj.weight"] = (
        utils_load_tensor_69
    )
    utils_load_tensor_70 = utils.load_tensor(
        "./tensors/arg70.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_5_mlp_gate_proj.weight"] = (
        utils_load_tensor_70
    )
    utils_load_tensor_71 = utils.load_tensor(
        "./tensors/arg71.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_6_self_attn_k_norm_weight"] = (
        utils_load_tensor_71
    )
    utils_load_tensor_72 = utils.load_tensor(
        "./tensors/arg72.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_6_self_attn_k_proj.weight"] = (
        utils_load_tensor_72
    )
    utils_load_tensor_73 = utils.load_tensor(
        "./tensors/arg73.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_7_self_attn_v_proj.weight"] = (
        utils_load_tensor_73
    )
    utils_load_tensor_74 = utils.load_tensor(
        "./tensors/arg74.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_7_input_layernorm_weight"] = (
        utils_load_tensor_74
    )
    utils_load_tensor_75 = utils.load_tensor(
        "./tensors/arg75.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_6_mlp_down_proj.weight"] = (
        utils_load_tensor_75
    )
    utils_load_tensor_76 = utils.load_tensor(
        "./tensors/arg76.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_6_mlp_up_proj.weight"] = utils_load_tensor_76
    utils_load_tensor_77 = utils.load_tensor(
        "./tensors/arg77.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_6_post_attention_layernorm_weight"] = (
        utils_load_tensor_77
    )
    utils_load_tensor_78 = utils.load_tensor(
        "./tensors/arg78.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_6_self_attn_o_proj.weight"] = (
        utils_load_tensor_78
    )
    utils_load_tensor_79 = utils.load_tensor(
        "./tensors/arg79.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_6_self_attn_q_norm_weight"] = (
        utils_load_tensor_79
    )
    utils_load_tensor_80 = utils.load_tensor(
        "./tensors/arg80.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_6_self_attn_q_proj.weight"] = (
        utils_load_tensor_80
    )
    utils_load_tensor_81 = utils.load_tensor(
        "./tensors/arg81.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_6_mlp_gate_proj.weight"] = (
        utils_load_tensor_81
    )
    utils_load_tensor_82 = utils.load_tensor(
        "./tensors/arg82.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_7_self_attn_k_norm_weight"] = (
        utils_load_tensor_82
    )
    utils_load_tensor_83 = utils.load_tensor(
        "./tensors/arg83.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_7_self_attn_k_proj.weight"] = (
        utils_load_tensor_83
    )
    utils_load_tensor_84 = utils.load_tensor(
        "./tensors/arg84.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_8_self_attn_v_proj.weight"] = (
        utils_load_tensor_84
    )
    utils_load_tensor_85 = utils.load_tensor(
        "./tensors/arg85.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_8_input_layernorm_weight"] = (
        utils_load_tensor_85
    )
    utils_load_tensor_86 = utils.load_tensor(
        "./tensors/arg86.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_7_mlp_down_proj.weight"] = (
        utils_load_tensor_86
    )
    utils_load_tensor_87 = utils.load_tensor(
        "./tensors/arg87.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_7_mlp_up_proj.weight"] = utils_load_tensor_87
    utils_load_tensor_88 = utils.load_tensor(
        "./tensors/arg88.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_7_post_attention_layernorm_weight"] = (
        utils_load_tensor_88
    )
    utils_load_tensor_89 = utils.load_tensor(
        "./tensors/arg89.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_7_self_attn_o_proj.weight"] = (
        utils_load_tensor_89
    )
    utils_load_tensor_90 = utils.load_tensor(
        "./tensors/arg90.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_7_self_attn_q_norm_weight"] = (
        utils_load_tensor_90
    )
    utils_load_tensor_91 = utils.load_tensor(
        "./tensors/arg91.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_7_self_attn_q_proj.weight"] = (
        utils_load_tensor_91
    )
    utils_load_tensor_92 = utils.load_tensor(
        "./tensors/arg92.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_7_mlp_gate_proj.weight"] = (
        utils_load_tensor_92
    )
    utils_load_tensor_93 = utils.load_tensor(
        "./tensors/arg93.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_8_self_attn_k_norm_weight"] = (
        utils_load_tensor_93
    )
    utils_load_tensor_94 = utils.load_tensor(
        "./tensors/arg94.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_8_self_attn_k_proj.weight"] = (
        utils_load_tensor_94
    )
    utils_load_tensor_95 = utils.load_tensor(
        "./tensors/arg95.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_9_self_attn_v_proj.weight"] = (
        utils_load_tensor_95
    )
    utils_load_tensor_96 = utils.load_tensor(
        "./tensors/arg96.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_9_input_layernorm_weight"] = (
        utils_load_tensor_96
    )
    utils_load_tensor_97 = utils.load_tensor(
        "./tensors/arg97.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_8_mlp_down_proj.weight"] = (
        utils_load_tensor_97
    )
    utils_load_tensor_98 = utils.load_tensor(
        "./tensors/arg98.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_8_mlp_up_proj.weight"] = utils_load_tensor_98
    utils_load_tensor_99 = utils.load_tensor(
        "./tensors/arg99.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_8_post_attention_layernorm_weight"] = (
        utils_load_tensor_99
    )
    utils_load_tensor_100 = utils.load_tensor(
        "./tensors/arg100.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_8_self_attn_o_proj.weight"] = (
        utils_load_tensor_100
    )
    utils_load_tensor_101 = utils.load_tensor(
        "./tensors/arg101.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_8_self_attn_q_norm_weight"] = (
        utils_load_tensor_101
    )
    utils_load_tensor_102 = utils.load_tensor(
        "./tensors/arg102.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_8_self_attn_q_proj.weight"] = (
        utils_load_tensor_102
    )
    utils_load_tensor_103 = utils.load_tensor(
        "./tensors/arg103.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_8_mlp_gate_proj.weight"] = (
        utils_load_tensor_103
    )
    utils_load_tensor_104 = utils.load_tensor(
        "./tensors/arg104.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_9_self_attn_k_norm_weight"] = (
        utils_load_tensor_104
    )
    utils_load_tensor_105 = utils.load_tensor(
        "./tensors/arg105.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_9_self_attn_k_proj.weight"] = (
        utils_load_tensor_105
    )
    utils_load_tensor_106 = utils.load_tensor(
        "./tensors/arg106.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_10_self_attn_v_proj.weight"] = (
        utils_load_tensor_106
    )
    utils_load_tensor_107 = utils.load_tensor(
        "./tensors/arg107.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_10_input_layernorm_weight"] = (
        utils_load_tensor_107
    )
    utils_load_tensor_108 = utils.load_tensor(
        "./tensors/arg108.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_9_mlp_down_proj.weight"] = (
        utils_load_tensor_108
    )
    utils_load_tensor_109 = utils.load_tensor(
        "./tensors/arg109.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_9_mlp_up_proj.weight"] = utils_load_tensor_109
    utils_load_tensor_110 = utils.load_tensor(
        "./tensors/arg110.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_9_post_attention_layernorm_weight"] = (
        utils_load_tensor_110
    )
    utils_load_tensor_111 = utils.load_tensor(
        "./tensors/arg111.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_9_self_attn_o_proj.weight"] = (
        utils_load_tensor_111
    )
    utils_load_tensor_112 = utils.load_tensor(
        "./tensors/arg112.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_9_self_attn_q_norm_weight"] = (
        utils_load_tensor_112
    )
    utils_load_tensor_113 = utils.load_tensor(
        "./tensors/arg113.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_9_self_attn_q_proj.weight"] = (
        utils_load_tensor_113
    )
    utils_load_tensor_114 = utils.load_tensor(
        "./tensors/arg114.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_9_mlp_gate_proj.weight"] = (
        utils_load_tensor_114
    )
    utils_load_tensor_115 = utils.load_tensor(
        "./tensors/arg115.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_10_self_attn_k_norm_weight"] = (
        utils_load_tensor_115
    )
    utils_load_tensor_116 = utils.load_tensor(
        "./tensors/arg116.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_10_self_attn_k_proj.weight"] = (
        utils_load_tensor_116
    )
    utils_load_tensor_117 = utils.load_tensor(
        "./tensors/arg117.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_11_self_attn_v_proj.weight"] = (
        utils_load_tensor_117
    )
    utils_load_tensor_118 = utils.load_tensor(
        "./tensors/arg118.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_11_input_layernorm_weight"] = (
        utils_load_tensor_118
    )
    utils_load_tensor_119 = utils.load_tensor(
        "./tensors/arg119.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_10_mlp_down_proj.weight"] = (
        utils_load_tensor_119
    )
    utils_load_tensor_120 = utils.load_tensor(
        "./tensors/arg120.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_10_mlp_up_proj.weight"] = (
        utils_load_tensor_120
    )
    utils_load_tensor_121 = utils.load_tensor(
        "./tensors/arg121.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_10_post_attention_layernorm_weight"] = (
        utils_load_tensor_121
    )
    utils_load_tensor_122 = utils.load_tensor(
        "./tensors/arg122.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_10_self_attn_o_proj.weight"] = (
        utils_load_tensor_122
    )
    utils_load_tensor_123 = utils.load_tensor(
        "./tensors/arg123.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_10_self_attn_q_norm_weight"] = (
        utils_load_tensor_123
    )
    utils_load_tensor_124 = utils.load_tensor(
        "./tensors/arg124.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_10_self_attn_q_proj.weight"] = (
        utils_load_tensor_124
    )
    utils_load_tensor_125 = utils.load_tensor(
        "./tensors/arg125.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_10_mlp_gate_proj.weight"] = (
        utils_load_tensor_125
    )
    utils_load_tensor_126 = utils.load_tensor(
        "./tensors/arg126.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_11_self_attn_k_norm_weight"] = (
        utils_load_tensor_126
    )
    utils_load_tensor_127 = utils.load_tensor(
        "./tensors/arg127.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_11_self_attn_k_proj.weight"] = (
        utils_load_tensor_127
    )
    utils_load_tensor_128 = utils.load_tensor(
        "./tensors/arg128.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_12_self_attn_v_proj.weight"] = (
        utils_load_tensor_128
    )
    utils_load_tensor_129 = utils.load_tensor(
        "./tensors/arg129.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_12_input_layernorm_weight"] = (
        utils_load_tensor_129
    )
    utils_load_tensor_130 = utils.load_tensor(
        "./tensors/arg130.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_11_mlp_down_proj.weight"] = (
        utils_load_tensor_130
    )
    utils_load_tensor_131 = utils.load_tensor(
        "./tensors/arg131.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_11_mlp_up_proj.weight"] = (
        utils_load_tensor_131
    )
    utils_load_tensor_132 = utils.load_tensor(
        "./tensors/arg132.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_11_post_attention_layernorm_weight"] = (
        utils_load_tensor_132
    )
    utils_load_tensor_133 = utils.load_tensor(
        "./tensors/arg133.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_11_self_attn_o_proj.weight"] = (
        utils_load_tensor_133
    )
    utils_load_tensor_134 = utils.load_tensor(
        "./tensors/arg134.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_11_self_attn_q_norm_weight"] = (
        utils_load_tensor_134
    )
    utils_load_tensor_135 = utils.load_tensor(
        "./tensors/arg135.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_11_self_attn_q_proj.weight"] = (
        utils_load_tensor_135
    )
    utils_load_tensor_136 = utils.load_tensor(
        "./tensors/arg136.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_11_mlp_gate_proj.weight"] = (
        utils_load_tensor_136
    )
    utils_load_tensor_137 = utils.load_tensor(
        "./tensors/arg137.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_12_self_attn_k_norm_weight"] = (
        utils_load_tensor_137
    )
    utils_load_tensor_138 = utils.load_tensor(
        "./tensors/arg138.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_12_self_attn_k_proj.weight"] = (
        utils_load_tensor_138
    )
    utils_load_tensor_139 = utils.load_tensor(
        "./tensors/arg139.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_13_self_attn_v_proj.weight"] = (
        utils_load_tensor_139
    )
    utils_load_tensor_140 = utils.load_tensor(
        "./tensors/arg140.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_13_input_layernorm_weight"] = (
        utils_load_tensor_140
    )
    utils_load_tensor_141 = utils.load_tensor(
        "./tensors/arg141.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_12_mlp_down_proj.weight"] = (
        utils_load_tensor_141
    )
    utils_load_tensor_142 = utils.load_tensor(
        "./tensors/arg142.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_12_mlp_up_proj.weight"] = (
        utils_load_tensor_142
    )
    utils_load_tensor_143 = utils.load_tensor(
        "./tensors/arg143.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_12_post_attention_layernorm_weight"] = (
        utils_load_tensor_143
    )
    utils_load_tensor_144 = utils.load_tensor(
        "./tensors/arg144.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_12_self_attn_o_proj.weight"] = (
        utils_load_tensor_144
    )
    utils_load_tensor_145 = utils.load_tensor(
        "./tensors/arg145.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_12_self_attn_q_norm_weight"] = (
        utils_load_tensor_145
    )
    utils_load_tensor_146 = utils.load_tensor(
        "./tensors/arg146.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_12_self_attn_q_proj.weight"] = (
        utils_load_tensor_146
    )
    utils_load_tensor_147 = utils.load_tensor(
        "./tensors/arg147.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_12_mlp_gate_proj.weight"] = (
        utils_load_tensor_147
    )
    utils_load_tensor_148 = utils.load_tensor(
        "./tensors/arg148.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_13_self_attn_k_norm_weight"] = (
        utils_load_tensor_148
    )
    utils_load_tensor_149 = utils.load_tensor(
        "./tensors/arg149.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_13_self_attn_k_proj.weight"] = (
        utils_load_tensor_149
    )
    utils_load_tensor_150 = utils.load_tensor(
        "./tensors/arg150.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_14_self_attn_v_proj.weight"] = (
        utils_load_tensor_150
    )
    utils_load_tensor_151 = utils.load_tensor(
        "./tensors/arg151.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_14_input_layernorm_weight"] = (
        utils_load_tensor_151
    )
    utils_load_tensor_152 = utils.load_tensor(
        "./tensors/arg152.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_13_mlp_down_proj.weight"] = (
        utils_load_tensor_152
    )
    utils_load_tensor_153 = utils.load_tensor(
        "./tensors/arg153.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_13_mlp_up_proj.weight"] = (
        utils_load_tensor_153
    )
    utils_load_tensor_154 = utils.load_tensor(
        "./tensors/arg154.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_13_post_attention_layernorm_weight"] = (
        utils_load_tensor_154
    )
    utils_load_tensor_155 = utils.load_tensor(
        "./tensors/arg155.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_13_self_attn_o_proj.weight"] = (
        utils_load_tensor_155
    )
    utils_load_tensor_156 = utils.load_tensor(
        "./tensors/arg156.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_13_self_attn_q_norm_weight"] = (
        utils_load_tensor_156
    )
    utils_load_tensor_157 = utils.load_tensor(
        "./tensors/arg157.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_13_self_attn_q_proj.weight"] = (
        utils_load_tensor_157
    )
    utils_load_tensor_158 = utils.load_tensor(
        "./tensors/arg158.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_13_mlp_gate_proj.weight"] = (
        utils_load_tensor_158
    )
    utils_load_tensor_159 = utils.load_tensor(
        "./tensors/arg159.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_14_self_attn_k_norm_weight"] = (
        utils_load_tensor_159
    )
    utils_load_tensor_160 = utils.load_tensor(
        "./tensors/arg160.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_14_self_attn_k_proj.weight"] = (
        utils_load_tensor_160
    )
    utils_load_tensor_161 = utils.load_tensor(
        "./tensors/arg161.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_15_self_attn_v_proj.weight"] = (
        utils_load_tensor_161
    )
    utils_load_tensor_162 = utils.load_tensor(
        "./tensors/arg162.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_15_input_layernorm_weight"] = (
        utils_load_tensor_162
    )
    utils_load_tensor_163 = utils.load_tensor(
        "./tensors/arg163.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_14_mlp_down_proj.weight"] = (
        utils_load_tensor_163
    )
    utils_load_tensor_164 = utils.load_tensor(
        "./tensors/arg164.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_14_mlp_up_proj.weight"] = (
        utils_load_tensor_164
    )
    utils_load_tensor_165 = utils.load_tensor(
        "./tensors/arg165.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_14_post_attention_layernorm_weight"] = (
        utils_load_tensor_165
    )
    utils_load_tensor_166 = utils.load_tensor(
        "./tensors/arg166.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_14_self_attn_o_proj.weight"] = (
        utils_load_tensor_166
    )
    utils_load_tensor_167 = utils.load_tensor(
        "./tensors/arg167.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_14_self_attn_q_norm_weight"] = (
        utils_load_tensor_167
    )
    utils_load_tensor_168 = utils.load_tensor(
        "./tensors/arg168.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_14_self_attn_q_proj.weight"] = (
        utils_load_tensor_168
    )
    utils_load_tensor_169 = utils.load_tensor(
        "./tensors/arg169.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_14_mlp_gate_proj.weight"] = (
        utils_load_tensor_169
    )
    utils_load_tensor_170 = utils.load_tensor(
        "./tensors/arg170.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_15_self_attn_k_norm_weight"] = (
        utils_load_tensor_170
    )
    utils_load_tensor_171 = utils.load_tensor(
        "./tensors/arg171.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_15_self_attn_k_proj.weight"] = (
        utils_load_tensor_171
    )
    utils_load_tensor_172 = utils.load_tensor(
        "./tensors/arg172.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_16_self_attn_v_proj.weight"] = (
        utils_load_tensor_172
    )
    utils_load_tensor_173 = utils.load_tensor(
        "./tensors/arg173.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_16_input_layernorm_weight"] = (
        utils_load_tensor_173
    )
    utils_load_tensor_174 = utils.load_tensor(
        "./tensors/arg174.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_15_mlp_down_proj.weight"] = (
        utils_load_tensor_174
    )
    utils_load_tensor_175 = utils.load_tensor(
        "./tensors/arg175.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_15_mlp_up_proj.weight"] = (
        utils_load_tensor_175
    )
    utils_load_tensor_176 = utils.load_tensor(
        "./tensors/arg176.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_15_post_attention_layernorm_weight"] = (
        utils_load_tensor_176
    )
    utils_load_tensor_177 = utils.load_tensor(
        "./tensors/arg177.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_15_self_attn_o_proj.weight"] = (
        utils_load_tensor_177
    )
    utils_load_tensor_178 = utils.load_tensor(
        "./tensors/arg178.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_15_self_attn_q_norm_weight"] = (
        utils_load_tensor_178
    )
    utils_load_tensor_179 = utils.load_tensor(
        "./tensors/arg179.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_15_self_attn_q_proj.weight"] = (
        utils_load_tensor_179
    )
    utils_load_tensor_180 = utils.load_tensor(
        "./tensors/arg180.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_15_mlp_gate_proj.weight"] = (
        utils_load_tensor_180
    )
    utils_load_tensor_181 = utils.load_tensor(
        "./tensors/arg181.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_16_self_attn_k_norm_weight"] = (
        utils_load_tensor_181
    )
    utils_load_tensor_182 = utils.load_tensor(
        "./tensors/arg182.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_16_self_attn_k_proj.weight"] = (
        utils_load_tensor_182
    )
    utils_load_tensor_183 = utils.load_tensor(
        "./tensors/arg183.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_17_self_attn_v_proj.weight"] = (
        utils_load_tensor_183
    )
    utils_load_tensor_184 = utils.load_tensor(
        "./tensors/arg184.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_17_input_layernorm_weight"] = (
        utils_load_tensor_184
    )
    utils_load_tensor_185 = utils.load_tensor(
        "./tensors/arg185.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_16_mlp_down_proj.weight"] = (
        utils_load_tensor_185
    )
    utils_load_tensor_186 = utils.load_tensor(
        "./tensors/arg186.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_16_mlp_up_proj.weight"] = (
        utils_load_tensor_186
    )
    utils_load_tensor_187 = utils.load_tensor(
        "./tensors/arg187.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_16_post_attention_layernorm_weight"] = (
        utils_load_tensor_187
    )
    utils_load_tensor_188 = utils.load_tensor(
        "./tensors/arg188.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_16_self_attn_o_proj.weight"] = (
        utils_load_tensor_188
    )
    utils_load_tensor_189 = utils.load_tensor(
        "./tensors/arg189.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_16_self_attn_q_norm_weight"] = (
        utils_load_tensor_189
    )
    utils_load_tensor_190 = utils.load_tensor(
        "./tensors/arg190.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_16_self_attn_q_proj.weight"] = (
        utils_load_tensor_190
    )
    utils_load_tensor_191 = utils.load_tensor(
        "./tensors/arg191.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_16_mlp_gate_proj.weight"] = (
        utils_load_tensor_191
    )
    utils_load_tensor_192 = utils.load_tensor(
        "./tensors/arg192.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_17_self_attn_k_norm_weight"] = (
        utils_load_tensor_192
    )
    utils_load_tensor_193 = utils.load_tensor(
        "./tensors/arg193.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_17_self_attn_k_proj.weight"] = (
        utils_load_tensor_193
    )
    utils_load_tensor_194 = utils.load_tensor(
        "./tensors/arg194.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_18_self_attn_v_proj.weight"] = (
        utils_load_tensor_194
    )
    utils_load_tensor_195 = utils.load_tensor(
        "./tensors/arg195.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_18_input_layernorm_weight"] = (
        utils_load_tensor_195
    )
    utils_load_tensor_196 = utils.load_tensor(
        "./tensors/arg196.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_17_mlp_down_proj.weight"] = (
        utils_load_tensor_196
    )
    utils_load_tensor_197 = utils.load_tensor(
        "./tensors/arg197.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_17_mlp_up_proj.weight"] = (
        utils_load_tensor_197
    )
    utils_load_tensor_198 = utils.load_tensor(
        "./tensors/arg198.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_17_post_attention_layernorm_weight"] = (
        utils_load_tensor_198
    )
    utils_load_tensor_199 = utils.load_tensor(
        "./tensors/arg199.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_17_self_attn_o_proj.weight"] = (
        utils_load_tensor_199
    )
    utils_load_tensor_200 = utils.load_tensor(
        "./tensors/arg200.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_17_self_attn_q_norm_weight"] = (
        utils_load_tensor_200
    )
    utils_load_tensor_201 = utils.load_tensor(
        "./tensors/arg201.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_17_self_attn_q_proj.weight"] = (
        utils_load_tensor_201
    )
    utils_load_tensor_202 = utils.load_tensor(
        "./tensors/arg202.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_17_mlp_gate_proj.weight"] = (
        utils_load_tensor_202
    )
    utils_load_tensor_203 = utils.load_tensor(
        "./tensors/arg203.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_18_self_attn_k_norm_weight"] = (
        utils_load_tensor_203
    )
    utils_load_tensor_204 = utils.load_tensor(
        "./tensors/arg204.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_18_self_attn_k_proj.weight"] = (
        utils_load_tensor_204
    )
    utils_load_tensor_205 = utils.load_tensor(
        "./tensors/arg205.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_19_self_attn_v_proj.weight"] = (
        utils_load_tensor_205
    )
    utils_load_tensor_206 = utils.load_tensor(
        "./tensors/arg206.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_19_input_layernorm_weight"] = (
        utils_load_tensor_206
    )
    utils_load_tensor_207 = utils.load_tensor(
        "./tensors/arg207.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_18_mlp_down_proj.weight"] = (
        utils_load_tensor_207
    )
    utils_load_tensor_208 = utils.load_tensor(
        "./tensors/arg208.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_18_mlp_up_proj.weight"] = (
        utils_load_tensor_208
    )
    utils_load_tensor_209 = utils.load_tensor(
        "./tensors/arg209.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_18_post_attention_layernorm_weight"] = (
        utils_load_tensor_209
    )
    utils_load_tensor_210 = utils.load_tensor(
        "./tensors/arg210.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_18_self_attn_o_proj.weight"] = (
        utils_load_tensor_210
    )
    utils_load_tensor_211 = utils.load_tensor(
        "./tensors/arg211.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_18_self_attn_q_norm_weight"] = (
        utils_load_tensor_211
    )
    utils_load_tensor_212 = utils.load_tensor(
        "./tensors/arg212.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_18_self_attn_q_proj.weight"] = (
        utils_load_tensor_212
    )
    utils_load_tensor_213 = utils.load_tensor(
        "./tensors/arg213.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_18_mlp_gate_proj.weight"] = (
        utils_load_tensor_213
    )
    utils_load_tensor_214 = utils.load_tensor(
        "./tensors/arg214.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_19_self_attn_k_norm_weight"] = (
        utils_load_tensor_214
    )
    utils_load_tensor_215 = utils.load_tensor(
        "./tensors/arg215.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_19_self_attn_k_proj.weight"] = (
        utils_load_tensor_215
    )
    utils_load_tensor_216 = utils.load_tensor(
        "./tensors/arg216.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_20_self_attn_v_proj.weight"] = (
        utils_load_tensor_216
    )
    utils_load_tensor_217 = utils.load_tensor(
        "./tensors/arg217.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_20_input_layernorm_weight"] = (
        utils_load_tensor_217
    )
    utils_load_tensor_218 = utils.load_tensor(
        "./tensors/arg218.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_19_mlp_down_proj.weight"] = (
        utils_load_tensor_218
    )
    utils_load_tensor_219 = utils.load_tensor(
        "./tensors/arg219.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_19_mlp_up_proj.weight"] = (
        utils_load_tensor_219
    )
    utils_load_tensor_220 = utils.load_tensor(
        "./tensors/arg220.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_19_post_attention_layernorm_weight"] = (
        utils_load_tensor_220
    )
    utils_load_tensor_221 = utils.load_tensor(
        "./tensors/arg221.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_19_self_attn_o_proj.weight"] = (
        utils_load_tensor_221
    )
    utils_load_tensor_222 = utils.load_tensor(
        "./tensors/arg222.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_19_self_attn_q_norm_weight"] = (
        utils_load_tensor_222
    )
    utils_load_tensor_223 = utils.load_tensor(
        "./tensors/arg223.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_19_self_attn_q_proj.weight"] = (
        utils_load_tensor_223
    )
    utils_load_tensor_224 = utils.load_tensor(
        "./tensors/arg224.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_19_mlp_gate_proj.weight"] = (
        utils_load_tensor_224
    )
    utils_load_tensor_225 = utils.load_tensor(
        "./tensors/arg225.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_20_self_attn_k_norm_weight"] = (
        utils_load_tensor_225
    )
    utils_load_tensor_226 = utils.load_tensor(
        "./tensors/arg226.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_20_self_attn_k_proj.weight"] = (
        utils_load_tensor_226
    )
    utils_load_tensor_227 = utils.load_tensor(
        "./tensors/arg227.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_21_self_attn_v_proj.weight"] = (
        utils_load_tensor_227
    )
    utils_load_tensor_228 = utils.load_tensor(
        "./tensors/arg228.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_21_input_layernorm_weight"] = (
        utils_load_tensor_228
    )
    utils_load_tensor_229 = utils.load_tensor(
        "./tensors/arg229.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_20_mlp_down_proj.weight"] = (
        utils_load_tensor_229
    )
    utils_load_tensor_230 = utils.load_tensor(
        "./tensors/arg230.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_20_mlp_up_proj.weight"] = (
        utils_load_tensor_230
    )
    utils_load_tensor_231 = utils.load_tensor(
        "./tensors/arg231.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_20_post_attention_layernorm_weight"] = (
        utils_load_tensor_231
    )
    utils_load_tensor_232 = utils.load_tensor(
        "./tensors/arg232.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_20_self_attn_o_proj.weight"] = (
        utils_load_tensor_232
    )
    utils_load_tensor_233 = utils.load_tensor(
        "./tensors/arg233.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_20_self_attn_q_norm_weight"] = (
        utils_load_tensor_233
    )
    utils_load_tensor_234 = utils.load_tensor(
        "./tensors/arg234.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_20_self_attn_q_proj.weight"] = (
        utils_load_tensor_234
    )
    utils_load_tensor_235 = utils.load_tensor(
        "./tensors/arg235.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_20_mlp_gate_proj.weight"] = (
        utils_load_tensor_235
    )
    utils_load_tensor_236 = utils.load_tensor(
        "./tensors/arg236.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_21_self_attn_k_norm_weight"] = (
        utils_load_tensor_236
    )
    utils_load_tensor_237 = utils.load_tensor(
        "./tensors/arg237.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_21_self_attn_k_proj.weight"] = (
        utils_load_tensor_237
    )
    utils_load_tensor_238 = utils.load_tensor(
        "./tensors/arg238.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_22_self_attn_v_proj.weight"] = (
        utils_load_tensor_238
    )
    utils_load_tensor_239 = utils.load_tensor(
        "./tensors/arg239.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_22_input_layernorm_weight"] = (
        utils_load_tensor_239
    )
    utils_load_tensor_240 = utils.load_tensor(
        "./tensors/arg240.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_21_mlp_down_proj.weight"] = (
        utils_load_tensor_240
    )
    utils_load_tensor_241 = utils.load_tensor(
        "./tensors/arg241.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_21_mlp_up_proj.weight"] = (
        utils_load_tensor_241
    )
    utils_load_tensor_242 = utils.load_tensor(
        "./tensors/arg242.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_21_post_attention_layernorm_weight"] = (
        utils_load_tensor_242
    )
    utils_load_tensor_243 = utils.load_tensor(
        "./tensors/arg243.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_21_self_attn_o_proj.weight"] = (
        utils_load_tensor_243
    )
    utils_load_tensor_244 = utils.load_tensor(
        "./tensors/arg244.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_21_self_attn_q_norm_weight"] = (
        utils_load_tensor_244
    )
    utils_load_tensor_245 = utils.load_tensor(
        "./tensors/arg245.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_21_self_attn_q_proj.weight"] = (
        utils_load_tensor_245
    )
    utils_load_tensor_246 = utils.load_tensor(
        "./tensors/arg246.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_21_mlp_gate_proj.weight"] = (
        utils_load_tensor_246
    )
    utils_load_tensor_247 = utils.load_tensor(
        "./tensors/arg247.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_22_self_attn_k_norm_weight"] = (
        utils_load_tensor_247
    )
    utils_load_tensor_248 = utils.load_tensor(
        "./tensors/arg248.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_22_self_attn_k_proj.weight"] = (
        utils_load_tensor_248
    )
    utils_load_tensor_249 = utils.load_tensor(
        "./tensors/arg249.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_23_self_attn_v_proj.weight"] = (
        utils_load_tensor_249
    )
    utils_load_tensor_250 = utils.load_tensor(
        "./tensors/arg250.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_23_input_layernorm_weight"] = (
        utils_load_tensor_250
    )
    utils_load_tensor_251 = utils.load_tensor(
        "./tensors/arg251.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_22_mlp_down_proj.weight"] = (
        utils_load_tensor_251
    )
    utils_load_tensor_252 = utils.load_tensor(
        "./tensors/arg252.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_22_mlp_up_proj.weight"] = (
        utils_load_tensor_252
    )
    utils_load_tensor_253 = utils.load_tensor(
        "./tensors/arg253.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_22_post_attention_layernorm_weight"] = (
        utils_load_tensor_253
    )
    utils_load_tensor_254 = utils.load_tensor(
        "./tensors/arg254.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_22_self_attn_o_proj.weight"] = (
        utils_load_tensor_254
    )
    utils_load_tensor_255 = utils.load_tensor(
        "./tensors/arg255.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_22_self_attn_q_norm_weight"] = (
        utils_load_tensor_255
    )
    utils_load_tensor_256 = utils.load_tensor(
        "./tensors/arg256.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_22_self_attn_q_proj.weight"] = (
        utils_load_tensor_256
    )
    utils_load_tensor_257 = utils.load_tensor(
        "./tensors/arg257.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_22_mlp_gate_proj.weight"] = (
        utils_load_tensor_257
    )
    utils_load_tensor_258 = utils.load_tensor(
        "./tensors/arg258.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_23_self_attn_k_norm_weight"] = (
        utils_load_tensor_258
    )
    utils_load_tensor_259 = utils.load_tensor(
        "./tensors/arg259.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_23_self_attn_k_proj.weight"] = (
        utils_load_tensor_259
    )
    utils_load_tensor_260 = utils.load_tensor(
        "./tensors/arg260.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_24_self_attn_v_proj.weight"] = (
        utils_load_tensor_260
    )
    utils_load_tensor_261 = utils.load_tensor(
        "./tensors/arg261.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_24_input_layernorm_weight"] = (
        utils_load_tensor_261
    )
    utils_load_tensor_262 = utils.load_tensor(
        "./tensors/arg262.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_23_mlp_down_proj.weight"] = (
        utils_load_tensor_262
    )
    utils_load_tensor_263 = utils.load_tensor(
        "./tensors/arg263.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_23_mlp_up_proj.weight"] = (
        utils_load_tensor_263
    )
    utils_load_tensor_264 = utils.load_tensor(
        "./tensors/arg264.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_23_post_attention_layernorm_weight"] = (
        utils_load_tensor_264
    )
    utils_load_tensor_265 = utils.load_tensor(
        "./tensors/arg265.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_23_self_attn_o_proj.weight"] = (
        utils_load_tensor_265
    )
    utils_load_tensor_266 = utils.load_tensor(
        "./tensors/arg266.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_23_self_attn_q_norm_weight"] = (
        utils_load_tensor_266
    )
    utils_load_tensor_267 = utils.load_tensor(
        "./tensors/arg267.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_23_self_attn_q_proj.weight"] = (
        utils_load_tensor_267
    )
    utils_load_tensor_268 = utils.load_tensor(
        "./tensors/arg268.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_23_mlp_gate_proj.weight"] = (
        utils_load_tensor_268
    )
    utils_load_tensor_269 = utils.load_tensor(
        "./tensors/arg269.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_24_self_attn_k_norm_weight"] = (
        utils_load_tensor_269
    )
    utils_load_tensor_270 = utils.load_tensor(
        "./tensors/arg270.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_24_self_attn_k_proj.weight"] = (
        utils_load_tensor_270
    )
    utils_load_tensor_271 = utils.load_tensor(
        "./tensors/arg271.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_25_self_attn_v_proj.weight"] = (
        utils_load_tensor_271
    )
    utils_load_tensor_272 = utils.load_tensor(
        "./tensors/arg272.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_25_input_layernorm_weight"] = (
        utils_load_tensor_272
    )
    utils_load_tensor_273 = utils.load_tensor(
        "./tensors/arg273.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_24_mlp_down_proj.weight"] = (
        utils_load_tensor_273
    )
    utils_load_tensor_274 = utils.load_tensor(
        "./tensors/arg274.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_24_mlp_up_proj.weight"] = (
        utils_load_tensor_274
    )
    utils_load_tensor_275 = utils.load_tensor(
        "./tensors/arg275.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_24_post_attention_layernorm_weight"] = (
        utils_load_tensor_275
    )
    utils_load_tensor_276 = utils.load_tensor(
        "./tensors/arg276.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_24_self_attn_o_proj.weight"] = (
        utils_load_tensor_276
    )
    utils_load_tensor_277 = utils.load_tensor(
        "./tensors/arg277.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_24_self_attn_q_norm_weight"] = (
        utils_load_tensor_277
    )
    utils_load_tensor_278 = utils.load_tensor(
        "./tensors/arg278.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_24_self_attn_q_proj.weight"] = (
        utils_load_tensor_278
    )
    utils_load_tensor_279 = utils.load_tensor(
        "./tensors/arg279.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_24_mlp_gate_proj.weight"] = (
        utils_load_tensor_279
    )
    utils_load_tensor_280 = utils.load_tensor(
        "./tensors/arg280.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_25_self_attn_k_norm_weight"] = (
        utils_load_tensor_280
    )
    utils_load_tensor_281 = utils.load_tensor(
        "./tensors/arg281.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_25_self_attn_k_proj.weight"] = (
        utils_load_tensor_281
    )
    utils_load_tensor_282 = utils.load_tensor(
        "./tensors/arg282.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_26_self_attn_v_proj.weight"] = (
        utils_load_tensor_282
    )
    utils_load_tensor_283 = utils.load_tensor(
        "./tensors/arg283.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_26_input_layernorm_weight"] = (
        utils_load_tensor_283
    )
    utils_load_tensor_284 = utils.load_tensor(
        "./tensors/arg284.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_25_mlp_down_proj.weight"] = (
        utils_load_tensor_284
    )
    utils_load_tensor_285 = utils.load_tensor(
        "./tensors/arg285.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_25_mlp_up_proj.weight"] = (
        utils_load_tensor_285
    )
    utils_load_tensor_286 = utils.load_tensor(
        "./tensors/arg286.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_25_post_attention_layernorm_weight"] = (
        utils_load_tensor_286
    )
    utils_load_tensor_287 = utils.load_tensor(
        "./tensors/arg287.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_25_self_attn_o_proj.weight"] = (
        utils_load_tensor_287
    )
    utils_load_tensor_288 = utils.load_tensor(
        "./tensors/arg288.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_25_self_attn_q_norm_weight"] = (
        utils_load_tensor_288
    )
    utils_load_tensor_289 = utils.load_tensor(
        "./tensors/arg289.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_25_self_attn_q_proj.weight"] = (
        utils_load_tensor_289
    )
    utils_load_tensor_290 = utils.load_tensor(
        "./tensors/arg290.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_25_mlp_gate_proj.weight"] = (
        utils_load_tensor_290
    )
    utils_load_tensor_291 = utils.load_tensor(
        "./tensors/arg291.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_26_self_attn_k_norm_weight"] = (
        utils_load_tensor_291
    )
    utils_load_tensor_292 = utils.load_tensor(
        "./tensors/arg292.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_26_self_attn_k_proj.weight"] = (
        utils_load_tensor_292
    )
    utils_load_tensor_293 = utils.load_tensor(
        "./tensors/arg293.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_27_self_attn_v_proj.weight"] = (
        utils_load_tensor_293
    )
    utils_load_tensor_294 = utils.load_tensor(
        "./tensors/arg294.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_27_input_layernorm_weight"] = (
        utils_load_tensor_294
    )
    utils_load_tensor_295 = utils.load_tensor(
        "./tensors/arg295.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_26_mlp_down_proj.weight"] = (
        utils_load_tensor_295
    )
    utils_load_tensor_296 = utils.load_tensor(
        "./tensors/arg296.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_26_mlp_up_proj.weight"] = (
        utils_load_tensor_296
    )
    utils_load_tensor_297 = utils.load_tensor(
        "./tensors/arg297.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_26_post_attention_layernorm_weight"] = (
        utils_load_tensor_297
    )
    utils_load_tensor_298 = utils.load_tensor(
        "./tensors/arg298.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_26_self_attn_o_proj.weight"] = (
        utils_load_tensor_298
    )
    utils_load_tensor_299 = utils.load_tensor(
        "./tensors/arg299.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_26_self_attn_q_norm_weight"] = (
        utils_load_tensor_299
    )
    utils_load_tensor_300 = utils.load_tensor(
        "./tensors/arg300.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_26_self_attn_q_proj.weight"] = (
        utils_load_tensor_300
    )
    utils_load_tensor_301 = utils.load_tensor(
        "./tensors/arg301.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_26_mlp_gate_proj.weight"] = (
        utils_load_tensor_301
    )
    utils_load_tensor_302 = utils.load_tensor(
        "./tensors/arg302.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_27_self_attn_k_norm_weight"] = (
        utils_load_tensor_302
    )
    utils_load_tensor_303 = utils.load_tensor(
        "./tensors/arg303.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_27_self_attn_k_proj.weight"] = (
        utils_load_tensor_303
    )
    utils_load_tensor_304 = utils.load_tensor(
        "./tensors/arg304.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_28_self_attn_v_proj.weight"] = (
        utils_load_tensor_304
    )
    utils_load_tensor_305 = utils.load_tensor(
        "./tensors/arg305.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_28_input_layernorm_weight"] = (
        utils_load_tensor_305
    )
    utils_load_tensor_306 = utils.load_tensor(
        "./tensors/arg306.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_27_mlp_down_proj.weight"] = (
        utils_load_tensor_306
    )
    utils_load_tensor_307 = utils.load_tensor(
        "./tensors/arg307.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_27_mlp_up_proj.weight"] = (
        utils_load_tensor_307
    )
    utils_load_tensor_308 = utils.load_tensor(
        "./tensors/arg308.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_27_post_attention_layernorm_weight"] = (
        utils_load_tensor_308
    )
    utils_load_tensor_309 = utils.load_tensor(
        "./tensors/arg309.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_27_self_attn_o_proj.weight"] = (
        utils_load_tensor_309
    )
    utils_load_tensor_310 = utils.load_tensor(
        "./tensors/arg310.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_27_self_attn_q_norm_weight"] = (
        utils_load_tensor_310
    )
    utils_load_tensor_311 = utils.load_tensor(
        "./tensors/arg311.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_27_self_attn_q_proj.weight"] = (
        utils_load_tensor_311
    )
    utils_load_tensor_312 = utils.load_tensor(
        "./tensors/arg312.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_27_mlp_gate_proj.weight"] = (
        utils_load_tensor_312
    )
    utils_load_tensor_313 = utils.load_tensor(
        "./tensors/arg313.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_28_self_attn_k_norm_weight"] = (
        utils_load_tensor_313
    )
    utils_load_tensor_314 = utils.load_tensor(
        "./tensors/arg314.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_28_self_attn_k_proj.weight"] = (
        utils_load_tensor_314
    )
    utils_load_tensor_315 = utils.load_tensor(
        "./tensors/arg315.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_29_self_attn_v_proj.weight"] = (
        utils_load_tensor_315
    )
    utils_load_tensor_316 = utils.load_tensor(
        "./tensors/arg316.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_29_input_layernorm_weight"] = (
        utils_load_tensor_316
    )
    utils_load_tensor_317 = utils.load_tensor(
        "./tensors/arg317.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_28_mlp_down_proj.weight"] = (
        utils_load_tensor_317
    )
    utils_load_tensor_318 = utils.load_tensor(
        "./tensors/arg318.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_28_mlp_up_proj.weight"] = (
        utils_load_tensor_318
    )
    utils_load_tensor_319 = utils.load_tensor(
        "./tensors/arg319.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_28_post_attention_layernorm_weight"] = (
        utils_load_tensor_319
    )
    utils_load_tensor_320 = utils.load_tensor(
        "./tensors/arg320.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_28_self_attn_o_proj.weight"] = (
        utils_load_tensor_320
    )
    utils_load_tensor_321 = utils.load_tensor(
        "./tensors/arg321.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_28_self_attn_q_norm_weight"] = (
        utils_load_tensor_321
    )
    utils_load_tensor_322 = utils.load_tensor(
        "./tensors/arg322.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_28_self_attn_q_proj.weight"] = (
        utils_load_tensor_322
    )
    utils_load_tensor_323 = utils.load_tensor(
        "./tensors/arg323.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_28_mlp_gate_proj.weight"] = (
        utils_load_tensor_323
    )
    utils_load_tensor_324 = utils.load_tensor(
        "./tensors/arg324.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_29_self_attn_k_norm_weight"] = (
        utils_load_tensor_324
    )
    utils_load_tensor_325 = utils.load_tensor(
        "./tensors/arg325.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_29_self_attn_k_proj.weight"] = (
        utils_load_tensor_325
    )
    utils_load_tensor_326 = utils.load_tensor(
        "./tensors/arg326.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_30_self_attn_v_proj.weight"] = (
        utils_load_tensor_326
    )
    utils_load_tensor_327 = utils.load_tensor(
        "./tensors/arg327.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_30_input_layernorm_weight"] = (
        utils_load_tensor_327
    )
    utils_load_tensor_328 = utils.load_tensor(
        "./tensors/arg328.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_29_mlp_down_proj.weight"] = (
        utils_load_tensor_328
    )
    utils_load_tensor_329 = utils.load_tensor(
        "./tensors/arg329.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_29_mlp_up_proj.weight"] = (
        utils_load_tensor_329
    )
    utils_load_tensor_330 = utils.load_tensor(
        "./tensors/arg330.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_29_post_attention_layernorm_weight"] = (
        utils_load_tensor_330
    )
    utils_load_tensor_331 = utils.load_tensor(
        "./tensors/arg331.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_29_self_attn_o_proj.weight"] = (
        utils_load_tensor_331
    )
    utils_load_tensor_332 = utils.load_tensor(
        "./tensors/arg332.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_29_self_attn_q_norm_weight"] = (
        utils_load_tensor_332
    )
    utils_load_tensor_333 = utils.load_tensor(
        "./tensors/arg333.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_29_self_attn_q_proj.weight"] = (
        utils_load_tensor_333
    )
    utils_load_tensor_334 = utils.load_tensor(
        "./tensors/arg334.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_29_mlp_gate_proj.weight"] = (
        utils_load_tensor_334
    )
    utils_load_tensor_335 = utils.load_tensor(
        "./tensors/arg335.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_30_self_attn_k_norm_weight"] = (
        utils_load_tensor_335
    )
    utils_load_tensor_336 = utils.load_tensor(
        "./tensors/arg336.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_30_self_attn_k_proj.weight"] = (
        utils_load_tensor_336
    )
    utils_load_tensor_337 = utils.load_tensor(
        "./tensors/arg337.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_31_self_attn_v_proj.weight"] = (
        utils_load_tensor_337
    )
    utils_load_tensor_338 = utils.load_tensor(
        "./tensors/arg338.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_31_input_layernorm_weight"] = (
        utils_load_tensor_338
    )
    utils_load_tensor_339 = utils.load_tensor(
        "./tensors/arg339.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_30_mlp_down_proj.weight"] = (
        utils_load_tensor_339
    )
    utils_load_tensor_340 = utils.load_tensor(
        "./tensors/arg340.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_30_mlp_up_proj.weight"] = (
        utils_load_tensor_340
    )
    utils_load_tensor_341 = utils.load_tensor(
        "./tensors/arg341.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_30_post_attention_layernorm_weight"] = (
        utils_load_tensor_341
    )
    utils_load_tensor_342 = utils.load_tensor(
        "./tensors/arg342.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_30_self_attn_o_proj.weight"] = (
        utils_load_tensor_342
    )
    utils_load_tensor_343 = utils.load_tensor(
        "./tensors/arg343.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_30_self_attn_q_norm_weight"] = (
        utils_load_tensor_343
    )
    utils_load_tensor_344 = utils.load_tensor(
        "./tensors/arg344.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_30_self_attn_q_proj.weight"] = (
        utils_load_tensor_344
    )
    utils_load_tensor_345 = utils.load_tensor(
        "./tensors/arg345.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_30_mlp_gate_proj.weight"] = (
        utils_load_tensor_345
    )
    utils_load_tensor_346 = utils.load_tensor(
        "./tensors/arg346.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_31_self_attn_k_norm_weight"] = (
        utils_load_tensor_346
    )
    utils_load_tensor_347 = utils.load_tensor(
        "./tensors/arg347.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_31_self_attn_k_proj.weight"] = (
        utils_load_tensor_347
    )
    utils_load_tensor_348 = utils.load_tensor(
        "./tensors/arg348.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_32_self_attn_v_proj.weight"] = (
        utils_load_tensor_348
    )
    utils_load_tensor_349 = utils.load_tensor(
        "./tensors/arg349.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_32_input_layernorm_weight"] = (
        utils_load_tensor_349
    )
    utils_load_tensor_350 = utils.load_tensor(
        "./tensors/arg350.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_31_mlp_down_proj.weight"] = (
        utils_load_tensor_350
    )
    utils_load_tensor_351 = utils.load_tensor(
        "./tensors/arg351.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_31_mlp_up_proj.weight"] = (
        utils_load_tensor_351
    )
    utils_load_tensor_352 = utils.load_tensor(
        "./tensors/arg352.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_31_post_attention_layernorm_weight"] = (
        utils_load_tensor_352
    )
    utils_load_tensor_353 = utils.load_tensor(
        "./tensors/arg353.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_31_self_attn_o_proj.weight"] = (
        utils_load_tensor_353
    )
    utils_load_tensor_354 = utils.load_tensor(
        "./tensors/arg354.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_31_self_attn_q_norm_weight"] = (
        utils_load_tensor_354
    )
    utils_load_tensor_355 = utils.load_tensor(
        "./tensors/arg355.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_31_self_attn_q_proj.weight"] = (
        utils_load_tensor_355
    )
    utils_load_tensor_356 = utils.load_tensor(
        "./tensors/arg356.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_31_mlp_gate_proj.weight"] = (
        utils_load_tensor_356
    )
    utils_load_tensor_357 = utils.load_tensor(
        "./tensors/arg357.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_32_self_attn_k_norm_weight"] = (
        utils_load_tensor_357
    )
    utils_load_tensor_358 = utils.load_tensor(
        "./tensors/arg358.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_32_self_attn_k_proj.weight"] = (
        utils_load_tensor_358
    )
    utils_load_tensor_359 = utils.load_tensor(
        "./tensors/arg359.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_33_self_attn_v_proj.weight"] = (
        utils_load_tensor_359
    )
    utils_load_tensor_360 = utils.load_tensor(
        "./tensors/arg360.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_33_input_layernorm_weight"] = (
        utils_load_tensor_360
    )
    utils_load_tensor_361 = utils.load_tensor(
        "./tensors/arg361.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_32_mlp_down_proj.weight"] = (
        utils_load_tensor_361
    )
    utils_load_tensor_362 = utils.load_tensor(
        "./tensors/arg362.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_32_mlp_up_proj.weight"] = (
        utils_load_tensor_362
    )
    utils_load_tensor_363 = utils.load_tensor(
        "./tensors/arg363.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_32_post_attention_layernorm_weight"] = (
        utils_load_tensor_363
    )
    utils_load_tensor_364 = utils.load_tensor(
        "./tensors/arg364.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_32_self_attn_o_proj.weight"] = (
        utils_load_tensor_364
    )
    utils_load_tensor_365 = utils.load_tensor(
        "./tensors/arg365.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_32_self_attn_q_norm_weight"] = (
        utils_load_tensor_365
    )
    utils_load_tensor_366 = utils.load_tensor(
        "./tensors/arg366.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_32_self_attn_q_proj.weight"] = (
        utils_load_tensor_366
    )
    utils_load_tensor_367 = utils.load_tensor(
        "./tensors/arg367.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_32_mlp_gate_proj.weight"] = (
        utils_load_tensor_367
    )
    utils_load_tensor_368 = utils.load_tensor(
        "./tensors/arg368.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_33_self_attn_k_norm_weight"] = (
        utils_load_tensor_368
    )
    utils_load_tensor_369 = utils.load_tensor(
        "./tensors/arg369.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_33_self_attn_k_proj.weight"] = (
        utils_load_tensor_369
    )
    utils_load_tensor_370 = utils.load_tensor(
        "./tensors/arg370.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_34_self_attn_v_proj.weight"] = (
        utils_load_tensor_370
    )
    utils_load_tensor_371 = utils.load_tensor(
        "./tensors/arg371.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_34_input_layernorm_weight"] = (
        utils_load_tensor_371
    )
    utils_load_tensor_372 = utils.load_tensor(
        "./tensors/arg372.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_33_mlp_down_proj.weight"] = (
        utils_load_tensor_372
    )
    utils_load_tensor_373 = utils.load_tensor(
        "./tensors/arg373.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_33_mlp_up_proj.weight"] = (
        utils_load_tensor_373
    )
    utils_load_tensor_374 = utils.load_tensor(
        "./tensors/arg374.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_33_post_attention_layernorm_weight"] = (
        utils_load_tensor_374
    )
    utils_load_tensor_375 = utils.load_tensor(
        "./tensors/arg375.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_33_self_attn_o_proj.weight"] = (
        utils_load_tensor_375
    )
    utils_load_tensor_376 = utils.load_tensor(
        "./tensors/arg376.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_33_self_attn_q_norm_weight"] = (
        utils_load_tensor_376
    )
    utils_load_tensor_377 = utils.load_tensor(
        "./tensors/arg377.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_33_self_attn_q_proj.weight"] = (
        utils_load_tensor_377
    )
    utils_load_tensor_378 = utils.load_tensor(
        "./tensors/arg378.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_33_mlp_gate_proj.weight"] = (
        utils_load_tensor_378
    )
    utils_load_tensor_379 = utils.load_tensor(
        "./tensors/arg379.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_34_self_attn_k_norm_weight"] = (
        utils_load_tensor_379
    )
    utils_load_tensor_380 = utils.load_tensor(
        "./tensors/arg380.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_34_self_attn_k_proj.weight"] = (
        utils_load_tensor_380
    )
    utils_load_tensor_381 = utils.load_tensor(
        "./tensors/arg381.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_35_self_attn_v_proj.weight"] = (
        utils_load_tensor_381
    )
    utils_load_tensor_382 = utils.load_tensor(
        "./tensors/arg382.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_35_input_layernorm_weight"] = (
        utils_load_tensor_382
    )
    utils_load_tensor_383 = utils.load_tensor(
        "./tensors/arg383.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_34_mlp_down_proj.weight"] = (
        utils_load_tensor_383
    )
    utils_load_tensor_384 = utils.load_tensor(
        "./tensors/arg384.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_34_mlp_up_proj.weight"] = (
        utils_load_tensor_384
    )
    utils_load_tensor_385 = utils.load_tensor(
        "./tensors/arg385.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_34_post_attention_layernorm_weight"] = (
        utils_load_tensor_385
    )
    utils_load_tensor_386 = utils.load_tensor(
        "./tensors/arg386.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_34_self_attn_o_proj.weight"] = (
        utils_load_tensor_386
    )
    utils_load_tensor_387 = utils.load_tensor(
        "./tensors/arg387.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_34_self_attn_q_norm_weight"] = (
        utils_load_tensor_387
    )
    utils_load_tensor_388 = utils.load_tensor(
        "./tensors/arg388.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_34_self_attn_q_proj.weight"] = (
        utils_load_tensor_388
    )
    utils_load_tensor_389 = utils.load_tensor(
        "./tensors/arg389.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_34_mlp_gate_proj.weight"] = (
        utils_load_tensor_389
    )
    utils_load_tensor_390 = utils.load_tensor(
        "./tensors/arg390.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_35_self_attn_k_norm_weight"] = (
        utils_load_tensor_390
    )
    utils_load_tensor_391 = utils.load_tensor(
        "./tensors/arg391.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_35_self_attn_k_proj.weight"] = (
        utils_load_tensor_391
    )
    utils_load_tensor_392 = utils.load_tensor(
        "./tensors/arg392.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___lm_head.weight"] = utils_load_tensor_392
    utils_load_tensor_393 = utils.load_tensor(
        "./tensors/arg393.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_norm_weight"] = utils_load_tensor_393
    utils_load_tensor_394 = utils.load_tensor(
        "./tensors/arg394.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_35_mlp_down_proj.weight"] = (
        utils_load_tensor_394
    )
    utils_load_tensor_395 = utils.load_tensor(
        "./tensors/arg395.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_35_mlp_up_proj.weight"] = (
        utils_load_tensor_395
    )
    utils_load_tensor_396 = utils.load_tensor(
        "./tensors/arg396.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_35_post_attention_layernorm_weight"] = (
        utils_load_tensor_396
    )
    utils_load_tensor_397 = utils.load_tensor(
        "./tensors/arg397.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_35_self_attn_o_proj.weight"] = (
        utils_load_tensor_397
    )
    utils_load_tensor_398 = utils.load_tensor(
        "./tensors/arg398.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_35_self_attn_q_norm_weight"] = (
        utils_load_tensor_398
    )
    utils_load_tensor_399 = utils.load_tensor(
        "./tensors/arg399.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["L__self___model_layers_35_self_attn_q_proj.weight"] = (
        utils_load_tensor_399
    )
    utils_load_tensor_400 = utils.load_tensor(
        "./tensors/arg400.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["L__self___model_layers_35_mlp_gate_proj.weight"] = (
        utils_load_tensor_400
    )
    return _main_weights


ROW_MAJOR_BFLOAT16_WEIGHTS = {
    "L__self___model_embed_tokens.weight",
    "L__self___model_layers_0_self_attn_k_proj.weight",
    "L__self___model_layers_0_self_attn_q_proj.weight",
    "L__self___model_layers_0_self_attn_v_proj.weight",
    "L__self___model_layers_10_self_attn_k_proj.weight",
    "L__self___model_layers_10_self_attn_q_proj.weight",
    "L__self___model_layers_10_self_attn_v_proj.weight",
    "L__self___model_layers_11_self_attn_k_proj.weight",
    "L__self___model_layers_11_self_attn_q_proj.weight",
    "L__self___model_layers_11_self_attn_v_proj.weight",
    "L__self___model_layers_12_self_attn_k_proj.weight",
    "L__self___model_layers_12_self_attn_q_proj.weight",
    "L__self___model_layers_12_self_attn_v_proj.weight",
    "L__self___model_layers_13_self_attn_k_proj.weight",
    "L__self___model_layers_13_self_attn_q_proj.weight",
    "L__self___model_layers_13_self_attn_v_proj.weight",
    "L__self___model_layers_14_self_attn_k_proj.weight",
    "L__self___model_layers_14_self_attn_q_proj.weight",
    "L__self___model_layers_14_self_attn_v_proj.weight",
    "L__self___model_layers_15_self_attn_k_proj.weight",
    "L__self___model_layers_15_self_attn_q_proj.weight",
    "L__self___model_layers_15_self_attn_v_proj.weight",
    "L__self___model_layers_16_self_attn_k_proj.weight",
    "L__self___model_layers_16_self_attn_q_proj.weight",
    "L__self___model_layers_16_self_attn_v_proj.weight",
    "L__self___model_layers_17_self_attn_k_proj.weight",
    "L__self___model_layers_17_self_attn_q_proj.weight",
    "L__self___model_layers_17_self_attn_v_proj.weight",
    "L__self___model_layers_18_self_attn_k_proj.weight",
    "L__self___model_layers_18_self_attn_q_proj.weight",
    "L__self___model_layers_18_self_attn_v_proj.weight",
    "L__self___model_layers_19_self_attn_k_proj.weight",
    "L__self___model_layers_19_self_attn_q_proj.weight",
    "L__self___model_layers_19_self_attn_v_proj.weight",
    "L__self___model_layers_1_self_attn_k_proj.weight",
    "L__self___model_layers_1_self_attn_q_proj.weight",
    "L__self___model_layers_1_self_attn_v_proj.weight",
    "L__self___model_layers_20_self_attn_k_proj.weight",
    "L__self___model_layers_20_self_attn_q_proj.weight",
    "L__self___model_layers_20_self_attn_v_proj.weight",
    "L__self___model_layers_21_self_attn_k_proj.weight",
    "L__self___model_layers_21_self_attn_q_proj.weight",
    "L__self___model_layers_21_self_attn_v_proj.weight",
    "L__self___model_layers_22_self_attn_k_proj.weight",
    "L__self___model_layers_22_self_attn_q_proj.weight",
    "L__self___model_layers_22_self_attn_v_proj.weight",
    "L__self___model_layers_23_self_attn_k_proj.weight",
    "L__self___model_layers_23_self_attn_q_proj.weight",
    "L__self___model_layers_23_self_attn_v_proj.weight",
    "L__self___model_layers_24_self_attn_k_proj.weight",
    "L__self___model_layers_24_self_attn_q_proj.weight",
    "L__self___model_layers_24_self_attn_v_proj.weight",
    "L__self___model_layers_25_self_attn_k_proj.weight",
    "L__self___model_layers_25_self_attn_q_proj.weight",
    "L__self___model_layers_25_self_attn_v_proj.weight",
    "L__self___model_layers_26_self_attn_k_proj.weight",
    "L__self___model_layers_26_self_attn_q_proj.weight",
    "L__self___model_layers_26_self_attn_v_proj.weight",
    "L__self___model_layers_27_self_attn_k_proj.weight",
    "L__self___model_layers_27_self_attn_q_proj.weight",
    "L__self___model_layers_27_self_attn_v_proj.weight",
    "L__self___model_layers_28_self_attn_k_proj.weight",
    "L__self___model_layers_28_self_attn_q_proj.weight",
    "L__self___model_layers_28_self_attn_v_proj.weight",
    "L__self___model_layers_29_self_attn_k_proj.weight",
    "L__self___model_layers_29_self_attn_q_proj.weight",
    "L__self___model_layers_29_self_attn_v_proj.weight",
    "L__self___model_layers_2_self_attn_k_proj.weight",
    "L__self___model_layers_2_self_attn_q_proj.weight",
    "L__self___model_layers_2_self_attn_v_proj.weight",
    "L__self___model_layers_30_self_attn_k_proj.weight",
    "L__self___model_layers_30_self_attn_q_proj.weight",
    "L__self___model_layers_30_self_attn_v_proj.weight",
    "L__self___model_layers_31_self_attn_k_proj.weight",
    "L__self___model_layers_31_self_attn_q_proj.weight",
    "L__self___model_layers_31_self_attn_v_proj.weight",
    "L__self___model_layers_32_self_attn_k_proj.weight",
    "L__self___model_layers_32_self_attn_q_proj.weight",
    "L__self___model_layers_32_self_attn_v_proj.weight",
    "L__self___model_layers_33_self_attn_k_proj.weight",
    "L__self___model_layers_33_self_attn_q_proj.weight",
    "L__self___model_layers_33_self_attn_v_proj.weight",
    "L__self___model_layers_34_self_attn_k_proj.weight",
    "L__self___model_layers_34_self_attn_q_proj.weight",
    "L__self___model_layers_34_self_attn_v_proj.weight",
    "L__self___model_layers_35_self_attn_k_proj.weight",
    "L__self___model_layers_35_self_attn_q_proj.weight",
    "L__self___model_layers_35_self_attn_v_proj.weight",
    "L__self___model_layers_3_self_attn_k_proj.weight",
    "L__self___model_layers_3_self_attn_q_proj.weight",
    "L__self___model_layers_3_self_attn_v_proj.weight",
    "L__self___model_layers_4_self_attn_k_proj.weight",
    "L__self___model_layers_4_self_attn_q_proj.weight",
    "L__self___model_layers_4_self_attn_v_proj.weight",
    "L__self___model_layers_5_self_attn_k_proj.weight",
    "L__self___model_layers_5_self_attn_q_proj.weight",
    "L__self___model_layers_5_self_attn_v_proj.weight",
    "L__self___model_layers_6_self_attn_k_proj.weight",
    "L__self___model_layers_6_self_attn_q_proj.weight",
    "L__self___model_layers_6_self_attn_v_proj.weight",
    "L__self___model_layers_7_self_attn_k_proj.weight",
    "L__self___model_layers_7_self_attn_q_proj.weight",
    "L__self___model_layers_7_self_attn_v_proj.weight",
    "L__self___model_layers_8_self_attn_k_proj.weight",
    "L__self___model_layers_8_self_attn_q_proj.weight",
    "L__self___model_layers_8_self_attn_v_proj.weight",
    "L__self___model_layers_9_self_attn_k_proj.weight",
    "L__self___model_layers_9_self_attn_q_proj.weight",
    "L__self___model_layers_9_self_attn_v_proj.weight",
}


TILE_BFLOAT16_WEIGHTS = {
    "L__self___lm_head.weight",
    "L__self___model_layers_0_input_layernorm_weight",
    "L__self___model_layers_0_mlp_down_proj.weight",
    "L__self___model_layers_0_mlp_gate_proj.weight",
    "L__self___model_layers_0_mlp_up_proj.weight",
    "L__self___model_layers_0_post_attention_layernorm_weight",
    "L__self___model_layers_0_self_attn_k_norm_weight",
    "L__self___model_layers_0_self_attn_o_proj.weight",
    "L__self___model_layers_0_self_attn_q_norm_weight",
    "L__self___model_layers_10_input_layernorm_weight",
    "L__self___model_layers_10_mlp_down_proj.weight",
    "L__self___model_layers_10_mlp_gate_proj.weight",
    "L__self___model_layers_10_mlp_up_proj.weight",
    "L__self___model_layers_10_post_attention_layernorm_weight",
    "L__self___model_layers_10_self_attn_k_norm_weight",
    "L__self___model_layers_10_self_attn_o_proj.weight",
    "L__self___model_layers_10_self_attn_q_norm_weight",
    "L__self___model_layers_11_input_layernorm_weight",
    "L__self___model_layers_11_mlp_down_proj.weight",
    "L__self___model_layers_11_mlp_gate_proj.weight",
    "L__self___model_layers_11_mlp_up_proj.weight",
    "L__self___model_layers_11_post_attention_layernorm_weight",
    "L__self___model_layers_11_self_attn_k_norm_weight",
    "L__self___model_layers_11_self_attn_o_proj.weight",
    "L__self___model_layers_11_self_attn_q_norm_weight",
    "L__self___model_layers_12_input_layernorm_weight",
    "L__self___model_layers_12_mlp_down_proj.weight",
    "L__self___model_layers_12_mlp_gate_proj.weight",
    "L__self___model_layers_12_mlp_up_proj.weight",
    "L__self___model_layers_12_post_attention_layernorm_weight",
    "L__self___model_layers_12_self_attn_k_norm_weight",
    "L__self___model_layers_12_self_attn_o_proj.weight",
    "L__self___model_layers_12_self_attn_q_norm_weight",
    "L__self___model_layers_13_input_layernorm_weight",
    "L__self___model_layers_13_mlp_down_proj.weight",
    "L__self___model_layers_13_mlp_gate_proj.weight",
    "L__self___model_layers_13_mlp_up_proj.weight",
    "L__self___model_layers_13_post_attention_layernorm_weight",
    "L__self___model_layers_13_self_attn_k_norm_weight",
    "L__self___model_layers_13_self_attn_o_proj.weight",
    "L__self___model_layers_13_self_attn_q_norm_weight",
    "L__self___model_layers_14_input_layernorm_weight",
    "L__self___model_layers_14_mlp_down_proj.weight",
    "L__self___model_layers_14_mlp_gate_proj.weight",
    "L__self___model_layers_14_mlp_up_proj.weight",
    "L__self___model_layers_14_post_attention_layernorm_weight",
    "L__self___model_layers_14_self_attn_k_norm_weight",
    "L__self___model_layers_14_self_attn_o_proj.weight",
    "L__self___model_layers_14_self_attn_q_norm_weight",
    "L__self___model_layers_15_input_layernorm_weight",
    "L__self___model_layers_15_mlp_down_proj.weight",
    "L__self___model_layers_15_mlp_gate_proj.weight",
    "L__self___model_layers_15_mlp_up_proj.weight",
    "L__self___model_layers_15_post_attention_layernorm_weight",
    "L__self___model_layers_15_self_attn_k_norm_weight",
    "L__self___model_layers_15_self_attn_o_proj.weight",
    "L__self___model_layers_15_self_attn_q_norm_weight",
    "L__self___model_layers_16_input_layernorm_weight",
    "L__self___model_layers_16_mlp_down_proj.weight",
    "L__self___model_layers_16_mlp_gate_proj.weight",
    "L__self___model_layers_16_mlp_up_proj.weight",
    "L__self___model_layers_16_post_attention_layernorm_weight",
    "L__self___model_layers_16_self_attn_k_norm_weight",
    "L__self___model_layers_16_self_attn_o_proj.weight",
    "L__self___model_layers_16_self_attn_q_norm_weight",
    "L__self___model_layers_17_input_layernorm_weight",
    "L__self___model_layers_17_mlp_down_proj.weight",
    "L__self___model_layers_17_mlp_gate_proj.weight",
    "L__self___model_layers_17_mlp_up_proj.weight",
    "L__self___model_layers_17_post_attention_layernorm_weight",
    "L__self___model_layers_17_self_attn_k_norm_weight",
    "L__self___model_layers_17_self_attn_o_proj.weight",
    "L__self___model_layers_17_self_attn_q_norm_weight",
    "L__self___model_layers_18_input_layernorm_weight",
    "L__self___model_layers_18_mlp_down_proj.weight",
    "L__self___model_layers_18_mlp_gate_proj.weight",
    "L__self___model_layers_18_mlp_up_proj.weight",
    "L__self___model_layers_18_post_attention_layernorm_weight",
    "L__self___model_layers_18_self_attn_k_norm_weight",
    "L__self___model_layers_18_self_attn_o_proj.weight",
    "L__self___model_layers_18_self_attn_q_norm_weight",
    "L__self___model_layers_19_input_layernorm_weight",
    "L__self___model_layers_19_mlp_down_proj.weight",
    "L__self___model_layers_19_mlp_gate_proj.weight",
    "L__self___model_layers_19_mlp_up_proj.weight",
    "L__self___model_layers_19_post_attention_layernorm_weight",
    "L__self___model_layers_19_self_attn_k_norm_weight",
    "L__self___model_layers_19_self_attn_o_proj.weight",
    "L__self___model_layers_19_self_attn_q_norm_weight",
    "L__self___model_layers_1_input_layernorm_weight",
    "L__self___model_layers_1_mlp_down_proj.weight",
    "L__self___model_layers_1_mlp_gate_proj.weight",
    "L__self___model_layers_1_mlp_up_proj.weight",
    "L__self___model_layers_1_post_attention_layernorm_weight",
    "L__self___model_layers_1_self_attn_k_norm_weight",
    "L__self___model_layers_1_self_attn_o_proj.weight",
    "L__self___model_layers_1_self_attn_q_norm_weight",
    "L__self___model_layers_20_input_layernorm_weight",
    "L__self___model_layers_20_mlp_down_proj.weight",
    "L__self___model_layers_20_mlp_gate_proj.weight",
    "L__self___model_layers_20_mlp_up_proj.weight",
    "L__self___model_layers_20_post_attention_layernorm_weight",
    "L__self___model_layers_20_self_attn_k_norm_weight",
    "L__self___model_layers_20_self_attn_o_proj.weight",
    "L__self___model_layers_20_self_attn_q_norm_weight",
    "L__self___model_layers_21_input_layernorm_weight",
    "L__self___model_layers_21_mlp_down_proj.weight",
    "L__self___model_layers_21_mlp_gate_proj.weight",
    "L__self___model_layers_21_mlp_up_proj.weight",
    "L__self___model_layers_21_post_attention_layernorm_weight",
    "L__self___model_layers_21_self_attn_k_norm_weight",
    "L__self___model_layers_21_self_attn_o_proj.weight",
    "L__self___model_layers_21_self_attn_q_norm_weight",
    "L__self___model_layers_22_input_layernorm_weight",
    "L__self___model_layers_22_mlp_down_proj.weight",
    "L__self___model_layers_22_mlp_gate_proj.weight",
    "L__self___model_layers_22_mlp_up_proj.weight",
    "L__self___model_layers_22_post_attention_layernorm_weight",
    "L__self___model_layers_22_self_attn_k_norm_weight",
    "L__self___model_layers_22_self_attn_o_proj.weight",
    "L__self___model_layers_22_self_attn_q_norm_weight",
    "L__self___model_layers_23_input_layernorm_weight",
    "L__self___model_layers_23_mlp_down_proj.weight",
    "L__self___model_layers_23_mlp_gate_proj.weight",
    "L__self___model_layers_23_mlp_up_proj.weight",
    "L__self___model_layers_23_post_attention_layernorm_weight",
    "L__self___model_layers_23_self_attn_k_norm_weight",
    "L__self___model_layers_23_self_attn_o_proj.weight",
    "L__self___model_layers_23_self_attn_q_norm_weight",
    "L__self___model_layers_24_input_layernorm_weight",
    "L__self___model_layers_24_mlp_down_proj.weight",
    "L__self___model_layers_24_mlp_gate_proj.weight",
    "L__self___model_layers_24_mlp_up_proj.weight",
    "L__self___model_layers_24_post_attention_layernorm_weight",
    "L__self___model_layers_24_self_attn_k_norm_weight",
    "L__self___model_layers_24_self_attn_o_proj.weight",
    "L__self___model_layers_24_self_attn_q_norm_weight",
    "L__self___model_layers_25_input_layernorm_weight",
    "L__self___model_layers_25_mlp_down_proj.weight",
    "L__self___model_layers_25_mlp_gate_proj.weight",
    "L__self___model_layers_25_mlp_up_proj.weight",
    "L__self___model_layers_25_post_attention_layernorm_weight",
    "L__self___model_layers_25_self_attn_k_norm_weight",
    "L__self___model_layers_25_self_attn_o_proj.weight",
    "L__self___model_layers_25_self_attn_q_norm_weight",
    "L__self___model_layers_26_input_layernorm_weight",
    "L__self___model_layers_26_mlp_down_proj.weight",
    "L__self___model_layers_26_mlp_gate_proj.weight",
    "L__self___model_layers_26_mlp_up_proj.weight",
    "L__self___model_layers_26_post_attention_layernorm_weight",
    "L__self___model_layers_26_self_attn_k_norm_weight",
    "L__self___model_layers_26_self_attn_o_proj.weight",
    "L__self___model_layers_26_self_attn_q_norm_weight",
    "L__self___model_layers_27_input_layernorm_weight",
    "L__self___model_layers_27_mlp_down_proj.weight",
    "L__self___model_layers_27_mlp_gate_proj.weight",
    "L__self___model_layers_27_mlp_up_proj.weight",
    "L__self___model_layers_27_post_attention_layernorm_weight",
    "L__self___model_layers_27_self_attn_k_norm_weight",
    "L__self___model_layers_27_self_attn_o_proj.weight",
    "L__self___model_layers_27_self_attn_q_norm_weight",
    "L__self___model_layers_28_input_layernorm_weight",
    "L__self___model_layers_28_mlp_down_proj.weight",
    "L__self___model_layers_28_mlp_gate_proj.weight",
    "L__self___model_layers_28_mlp_up_proj.weight",
    "L__self___model_layers_28_post_attention_layernorm_weight",
    "L__self___model_layers_28_self_attn_k_norm_weight",
    "L__self___model_layers_28_self_attn_o_proj.weight",
    "L__self___model_layers_28_self_attn_q_norm_weight",
    "L__self___model_layers_29_input_layernorm_weight",
    "L__self___model_layers_29_mlp_down_proj.weight",
    "L__self___model_layers_29_mlp_gate_proj.weight",
    "L__self___model_layers_29_mlp_up_proj.weight",
    "L__self___model_layers_29_post_attention_layernorm_weight",
    "L__self___model_layers_29_self_attn_k_norm_weight",
    "L__self___model_layers_29_self_attn_o_proj.weight",
    "L__self___model_layers_29_self_attn_q_norm_weight",
    "L__self___model_layers_2_input_layernorm_weight",
    "L__self___model_layers_2_mlp_down_proj.weight",
    "L__self___model_layers_2_mlp_gate_proj.weight",
    "L__self___model_layers_2_mlp_up_proj.weight",
    "L__self___model_layers_2_post_attention_layernorm_weight",
    "L__self___model_layers_2_self_attn_k_norm_weight",
    "L__self___model_layers_2_self_attn_o_proj.weight",
    "L__self___model_layers_2_self_attn_q_norm_weight",
    "L__self___model_layers_30_input_layernorm_weight",
    "L__self___model_layers_30_mlp_down_proj.weight",
    "L__self___model_layers_30_mlp_gate_proj.weight",
    "L__self___model_layers_30_mlp_up_proj.weight",
    "L__self___model_layers_30_post_attention_layernorm_weight",
    "L__self___model_layers_30_self_attn_k_norm_weight",
    "L__self___model_layers_30_self_attn_o_proj.weight",
    "L__self___model_layers_30_self_attn_q_norm_weight",
    "L__self___model_layers_31_input_layernorm_weight",
    "L__self___model_layers_31_mlp_down_proj.weight",
    "L__self___model_layers_31_mlp_gate_proj.weight",
    "L__self___model_layers_31_mlp_up_proj.weight",
    "L__self___model_layers_31_post_attention_layernorm_weight",
    "L__self___model_layers_31_self_attn_k_norm_weight",
    "L__self___model_layers_31_self_attn_o_proj.weight",
    "L__self___model_layers_31_self_attn_q_norm_weight",
    "L__self___model_layers_32_input_layernorm_weight",
    "L__self___model_layers_32_mlp_down_proj.weight",
    "L__self___model_layers_32_mlp_gate_proj.weight",
    "L__self___model_layers_32_mlp_up_proj.weight",
    "L__self___model_layers_32_post_attention_layernorm_weight",
    "L__self___model_layers_32_self_attn_k_norm_weight",
    "L__self___model_layers_32_self_attn_o_proj.weight",
    "L__self___model_layers_32_self_attn_q_norm_weight",
    "L__self___model_layers_33_input_layernorm_weight",
    "L__self___model_layers_33_mlp_down_proj.weight",
    "L__self___model_layers_33_mlp_gate_proj.weight",
    "L__self___model_layers_33_mlp_up_proj.weight",
    "L__self___model_layers_33_post_attention_layernorm_weight",
    "L__self___model_layers_33_self_attn_k_norm_weight",
    "L__self___model_layers_33_self_attn_o_proj.weight",
    "L__self___model_layers_33_self_attn_q_norm_weight",
    "L__self___model_layers_34_input_layernorm_weight",
    "L__self___model_layers_34_mlp_down_proj.weight",
    "L__self___model_layers_34_mlp_gate_proj.weight",
    "L__self___model_layers_34_mlp_up_proj.weight",
    "L__self___model_layers_34_post_attention_layernorm_weight",
    "L__self___model_layers_34_self_attn_k_norm_weight",
    "L__self___model_layers_34_self_attn_o_proj.weight",
    "L__self___model_layers_34_self_attn_q_norm_weight",
    "L__self___model_layers_35_input_layernorm_weight",
    "L__self___model_layers_35_mlp_down_proj.weight",
    "L__self___model_layers_35_mlp_gate_proj.weight",
    "L__self___model_layers_35_mlp_up_proj.weight",
    "L__self___model_layers_35_post_attention_layernorm_weight",
    "L__self___model_layers_35_self_attn_k_norm_weight",
    "L__self___model_layers_35_self_attn_o_proj.weight",
    "L__self___model_layers_35_self_attn_q_norm_weight",
    "L__self___model_layers_3_input_layernorm_weight",
    "L__self___model_layers_3_mlp_down_proj.weight",
    "L__self___model_layers_3_mlp_gate_proj.weight",
    "L__self___model_layers_3_mlp_up_proj.weight",
    "L__self___model_layers_3_post_attention_layernorm_weight",
    "L__self___model_layers_3_self_attn_k_norm_weight",
    "L__self___model_layers_3_self_attn_o_proj.weight",
    "L__self___model_layers_3_self_attn_q_norm_weight",
    "L__self___model_layers_4_input_layernorm_weight",
    "L__self___model_layers_4_mlp_down_proj.weight",
    "L__self___model_layers_4_mlp_gate_proj.weight",
    "L__self___model_layers_4_mlp_up_proj.weight",
    "L__self___model_layers_4_post_attention_layernorm_weight",
    "L__self___model_layers_4_self_attn_k_norm_weight",
    "L__self___model_layers_4_self_attn_o_proj.weight",
    "L__self___model_layers_4_self_attn_q_norm_weight",
    "L__self___model_layers_5_input_layernorm_weight",
    "L__self___model_layers_5_mlp_down_proj.weight",
    "L__self___model_layers_5_mlp_gate_proj.weight",
    "L__self___model_layers_5_mlp_up_proj.weight",
    "L__self___model_layers_5_post_attention_layernorm_weight",
    "L__self___model_layers_5_self_attn_k_norm_weight",
    "L__self___model_layers_5_self_attn_o_proj.weight",
    "L__self___model_layers_5_self_attn_q_norm_weight",
    "L__self___model_layers_6_input_layernorm_weight",
    "L__self___model_layers_6_mlp_down_proj.weight",
    "L__self___model_layers_6_mlp_gate_proj.weight",
    "L__self___model_layers_6_mlp_up_proj.weight",
    "L__self___model_layers_6_post_attention_layernorm_weight",
    "L__self___model_layers_6_self_attn_k_norm_weight",
    "L__self___model_layers_6_self_attn_o_proj.weight",
    "L__self___model_layers_6_self_attn_q_norm_weight",
    "L__self___model_layers_7_input_layernorm_weight",
    "L__self___model_layers_7_mlp_down_proj.weight",
    "L__self___model_layers_7_mlp_gate_proj.weight",
    "L__self___model_layers_7_mlp_up_proj.weight",
    "L__self___model_layers_7_post_attention_layernorm_weight",
    "L__self___model_layers_7_self_attn_k_norm_weight",
    "L__self___model_layers_7_self_attn_o_proj.weight",
    "L__self___model_layers_7_self_attn_q_norm_weight",
    "L__self___model_layers_8_input_layernorm_weight",
    "L__self___model_layers_8_mlp_down_proj.weight",
    "L__self___model_layers_8_mlp_gate_proj.weight",
    "L__self___model_layers_8_mlp_up_proj.weight",
    "L__self___model_layers_8_post_attention_layernorm_weight",
    "L__self___model_layers_8_self_attn_k_norm_weight",
    "L__self___model_layers_8_self_attn_o_proj.weight",
    "L__self___model_layers_8_self_attn_q_norm_weight",
    "L__self___model_layers_9_input_layernorm_weight",
    "L__self___model_layers_9_mlp_down_proj.weight",
    "L__self___model_layers_9_mlp_gate_proj.weight",
    "L__self___model_layers_9_mlp_up_proj.weight",
    "L__self___model_layers_9_post_attention_layernorm_weight",
    "L__self___model_layers_9_self_attn_k_norm_weight",
    "L__self___model_layers_9_self_attn_o_proj.weight",
    "L__self___model_layers_9_self_attn_q_norm_weight",
    "L__self___model_norm_weight",
}


ROW_MAJOR_FLOAT32_WEIGHTS = {
    "L__self___model_rotary_emb_inv_freq",
}


ALL_WEIGHTS = ROW_MAJOR_BFLOAT16_WEIGHTS | TILE_BFLOAT16_WEIGHTS | ROW_MAJOR_FLOAT32_WEIGHTS


def _state_dict_key(weight_name):
    name = weight_name[len("L__self___"):]
    if name.endswith("_weight"):
        name = name[: -len("_weight")] + ".weight"
    name = name.replace("model_layers_", "model.layers.")
    name = name.replace("model_embed_tokens", "model.embed_tokens")
    name = name.replace("model_norm", "model.norm")
    name = name.replace("model_rotary_emb_inv_freq", "model.rotary_emb.inv_freq")
    name = name.replace("_self_attn_", ".self_attn.")
    name = name.replace("_mlp_", ".mlp.")
    name = name.replace("_input_layernorm", ".input_layernorm")
    name = name.replace("_post_attention_layernorm", ".post_attention_layernorm")
    return name


def load_weights_for__main_from_state_dict():
    device = utils.DeviceGetter.get_device((1, 1))

    model = model_pt.load_pytorch_model()
    state_dict = dict(model.state_dict())
    for name, buf in model.named_buffers():
        if name not in state_dict:
            state_dict[name] = buf

    weights = {}
    for key in ALL_WEIGHTS:
        pt_tensor = state_dict[_state_dict_key(key)]
        ttnn_tensor = ttnn.from_torch(pt_tensor)

        if key in ROW_MAJOR_BFLOAT16_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)

        if key in TILE_BFLOAT16_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.TILE)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)
            ttnn_tensor = ttnn.to_device(ttnn_tensor, device, ttnn.DRAM_MEMORY_CONFIG)

        if key in ROW_MAJOR_FLOAT32_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.FLOAT32)

        weights[key] = ttnn_tensor

    return weights
