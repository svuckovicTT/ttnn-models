# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import model_pt
import utils


_NUM_LAYERS = 36


def _build_key_sets():
    row_major_bfloat16 = {"L__self___model_embed_tokens.weight"} | {
        f"L__self___model_layers_{i}_self_attn_{p}_proj.weight"
        for i in range(_NUM_LAYERS)
        for p in ("q", "k", "v")
    }
    row_major_float32 = {"L__self___model_rotary_emb_inv_freq"}
    tile_bfloat16_dram = {
        "L__self___lm_head.weight",
        "L__self___model_norm_weight",
    }
    for i in range(_NUM_LAYERS):
        tile_bfloat16_dram |= {
            f"L__self___model_layers_{i}_input_layernorm_weight",
            f"L__self___model_layers_{i}_post_attention_layernorm_weight",
            f"L__self___model_layers_{i}_self_attn_q_norm_weight",
            f"L__self___model_layers_{i}_self_attn_k_norm_weight",
            f"L__self___model_layers_{i}_self_attn_o_proj.weight",
            f"L__self___model_layers_{i}_mlp_gate_proj.weight",
            f"L__self___model_layers_{i}_mlp_up_proj.weight",
            f"L__self___model_layers_{i}_mlp_down_proj.weight",
        }
    return row_major_bfloat16, row_major_float32, tile_bfloat16_dram


ROW_MAJOR_BFLOAT16_KEYS, ROW_MAJOR_FLOAT32_KEYS, TILE_BFLOAT16_DRAM_KEYS = _build_key_sets()
ALL_WEIGHT_KEYS = ROW_MAJOR_BFLOAT16_KEYS | ROW_MAJOR_FLOAT32_KEYS | TILE_BFLOAT16_DRAM_KEYS


def _build_arg_to_key():
    # Replicates the arg index -> weight key ordering emitted by the original
    # auto-generated load_weights_for__main. Verified against main.py by spot check.
    mapping = {
        0: "L__self___model_layers_0_self_attn_v_proj.weight",
        1: "L__self___model_layers_0_input_layernorm_weight",
        3: "L__self___model_embed_tokens.weight",
        4: "L__self___model_rotary_emb_inv_freq",
        5: "L__self___model_layers_0_self_attn_k_norm_weight",
        6: "L__self___model_layers_0_self_attn_k_proj.weight",
        7: "L__self___model_layers_1_self_attn_v_proj.weight",
        8: "L__self___model_layers_1_input_layernorm_weight",
    }
    for n in range(35):
        base = 9 + 11 * n
        mapping[base + 0] = f"L__self___model_layers_{n}_mlp_down_proj.weight"
        mapping[base + 1] = f"L__self___model_layers_{n}_mlp_up_proj.weight"
        mapping[base + 2] = f"L__self___model_layers_{n}_post_attention_layernorm_weight"
        mapping[base + 3] = f"L__self___model_layers_{n}_self_attn_o_proj.weight"
        mapping[base + 4] = f"L__self___model_layers_{n}_self_attn_q_norm_weight"
        mapping[base + 5] = f"L__self___model_layers_{n}_self_attn_q_proj.weight"
        mapping[base + 6] = f"L__self___model_layers_{n}_mlp_gate_proj.weight"
        if n < 34:
            mapping[base + 7] = f"L__self___model_layers_{n + 1}_self_attn_k_norm_weight"
            mapping[base + 8] = f"L__self___model_layers_{n + 1}_self_attn_k_proj.weight"
            mapping[base + 9] = f"L__self___model_layers_{n + 2}_self_attn_v_proj.weight"
            mapping[base + 10] = f"L__self___model_layers_{n + 2}_input_layernorm_weight"
        else:
            mapping[base + 7] = "L__self___model_layers_35_self_attn_k_norm_weight"
            mapping[base + 8] = "L__self___model_layers_35_self_attn_k_proj.weight"
            mapping[base + 9] = "L__self___lm_head.weight"
            mapping[base + 10] = "L__self___model_norm_weight"
    mapping[394] = "L__self___model_layers_35_mlp_down_proj.weight"
    mapping[395] = "L__self___model_layers_35_mlp_up_proj.weight"
    mapping[396] = "L__self___model_layers_35_post_attention_layernorm_weight"
    mapping[397] = "L__self___model_layers_35_self_attn_o_proj.weight"
    mapping[398] = "L__self___model_layers_35_self_attn_q_norm_weight"
    mapping[399] = "L__self___model_layers_35_self_attn_q_proj.weight"
    mapping[400] = "L__self___model_layers_35_mlp_gate_proj.weight"
    return mapping


_ARG_TO_KEY = _build_arg_to_key()


def _build_dynamo_to_state_dict_key():
    mapping = {
        "L__self___model_embed_tokens.weight": "model.embed_tokens.weight",
        "L__self___model_norm_weight": "model.norm.weight",
        "L__self___model_rotary_emb_inv_freq": "model.rotary_emb.inv_freq",
        "L__self___lm_head.weight": "lm_head.weight",
    }
    for i in range(_NUM_LAYERS):
        for p in ("q", "k", "v", "o"):
            mapping[
                f"L__self___model_layers_{i}_self_attn_{p}_proj.weight"
            ] = f"model.layers.{i}.self_attn.{p}_proj.weight"
        for n in ("q", "k"):
            mapping[
                f"L__self___model_layers_{i}_self_attn_{n}_norm_weight"
            ] = f"model.layers.{i}.self_attn.{n}_norm.weight"
        for p in ("gate", "up", "down"):
            mapping[
                f"L__self___model_layers_{i}_mlp_{p}_proj.weight"
            ] = f"model.layers.{i}.mlp.{p}_proj.weight"
        mapping[
            f"L__self___model_layers_{i}_input_layernorm_weight"
        ] = f"model.layers.{i}.input_layernorm.weight"
        mapping[
            f"L__self___model_layers_{i}_post_attention_layernorm_weight"
        ] = f"model.layers.{i}.post_attention_layernorm.weight"
    return mapping


_DYNAMO_TO_STATE_DICT_KEY = _build_dynamo_to_state_dict_key()


_main_weights = {}


def load_weights_for__main():
    device = utils.DeviceGetter.get_device((1, 1))
    global _main_weights

    for arg_idx in sorted(_ARG_TO_KEY.keys()):
        key = _ARG_TO_KEY[arg_idx]
        path = f"./tensors/arg{arg_idx}.tensorbin"
        if key in ROW_MAJOR_BFLOAT16_KEYS:
            tensor = utils.load_tensor(
                path,
                ttnn.Layout.ROW_MAJOR,
                ttnn.DataType.BFLOAT16,
                None,
                None,
            )
        elif key in ROW_MAJOR_FLOAT32_KEYS:
            tensor = utils.load_tensor(
                path,
                ttnn.Layout.ROW_MAJOR,
                ttnn.DataType.FLOAT32,
                None,
                None,
            )
        else:
            tensor = utils.load_tensor(
                path,
                ttnn.Layout.TILE,
                ttnn.DataType.BFLOAT16,
                device,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        _main_weights[key] = tensor
    return _main_weights


def load_weights_for__main_from_state_dict():
    device = utils.DeviceGetter.get_device((1, 1))
    model = model_pt.load_pytorch_model()
    state_dict = model.state_dict()

    weights = {}
    for key in ALL_WEIGHT_KEYS:
        pt_tensor = state_dict[_DYNAMO_TO_STATE_DICT_KEY[key]]
        ttnn_tensor = ttnn.from_torch(pt_tensor)

        if key in ROW_MAJOR_BFLOAT16_KEYS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)

        if key in ROW_MAJOR_FLOAT32_KEYS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.FLOAT32)

        if key in TILE_BFLOAT16_DRAM_KEYS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.TILE)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)
            ttnn_tensor = ttnn.to_device(ttnn_tensor, device, ttnn.DRAM_MEMORY_CONFIG)

        weights[key] = ttnn_tensor
    return weights
