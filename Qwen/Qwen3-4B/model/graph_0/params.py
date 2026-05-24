# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import utils


_NUM_LAYERS = 36


def _build_key_to_arg_index():
    """Build a 400-entry mapping from FX-style weight key to arg<N>.tensorbin index.

    The order mirrors the original auto-generated load order: each entry's
    arg index is the same N used in the on-disk filename arg{N}.tensorbin.
    """
    mapping = {}

    # K=1..6: block-0 header (layer 0 v/input + global embed/rotary + layer 0 k_norm/k_proj)
    mapping["L__self___model_layers_0_self_attn_v_proj.weight"] = 0
    mapping["L__self___model_layers_0_input_layernorm_weight"] = 1
    mapping["L__self___model_embed_tokens.weight"] = 3
    mapping["L__self___model_rotary_emb_inv_freq"] = 4
    mapping["L__self___model_layers_0_self_attn_k_norm_weight"] = 5
    mapping["L__self___model_layers_0_self_attn_k_proj.weight"] = 6

    # K=7..391: 35 regular blocks of 11 entries each.
    # Block L (L=1..35) interleaves layer L's attn v/input/k_norm/k_proj
    # with layer L-1's mlp/o_proj/q_proj/q_norm/post_layernorm.
    for L in range(1, _NUM_LAYERS):
        prev = L - 1
        k = 7 + 11 * (L - 1)
        mapping[f"L__self___model_layers_{L}_self_attn_v_proj.weight"] = k
        mapping[f"L__self___model_layers_{L}_input_layernorm_weight"] = k + 1
        mapping[f"L__self___model_layers_{prev}_mlp_down_proj.weight"] = k + 2
        mapping[f"L__self___model_layers_{prev}_mlp_up_proj.weight"] = k + 3
        mapping[f"L__self___model_layers_{prev}_post_attention_layernorm_weight"] = k + 4
        mapping[f"L__self___model_layers_{prev}_self_attn_o_proj.weight"] = k + 5
        mapping[f"L__self___model_layers_{prev}_self_attn_q_norm_weight"] = k + 6
        mapping[f"L__self___model_layers_{prev}_self_attn_q_proj.weight"] = k + 7
        mapping[f"L__self___model_layers_{prev}_mlp_gate_proj.weight"] = k + 8
        mapping[f"L__self___model_layers_{L}_self_attn_k_norm_weight"] = k + 9
        mapping[f"L__self___model_layers_{L}_self_attn_k_proj.weight"] = k + 10

    # K=392..400: lm_head + model.norm, then the remaining mlp/attn weights of the last layer.
    last = _NUM_LAYERS - 1
    mapping["L__self___lm_head.weight"] = 392
    mapping["L__self___model_norm_weight"] = 393
    mapping[f"L__self___model_layers_{last}_mlp_down_proj.weight"] = 394
    mapping[f"L__self___model_layers_{last}_mlp_up_proj.weight"] = 395
    mapping[f"L__self___model_layers_{last}_post_attention_layernorm_weight"] = 396
    mapping[f"L__self___model_layers_{last}_self_attn_o_proj.weight"] = 397
    mapping[f"L__self___model_layers_{last}_self_attn_q_norm_weight"] = 398
    mapping[f"L__self___model_layers_{last}_self_attn_q_proj.weight"] = 399
    mapping[f"L__self___model_layers_{last}_mlp_gate_proj.weight"] = 400

    return mapping


def _build_fx_to_state_dict():
    """Map each FX-style weight key to its PyTorch state_dict counterpart."""
    mapping = {}
    for L in range(_NUM_LAYERS):
        mapping[f"L__self___model_layers_{L}_self_attn_q_proj.weight"] = f"model.layers.{L}.self_attn.q_proj.weight"
        mapping[f"L__self___model_layers_{L}_self_attn_k_proj.weight"] = f"model.layers.{L}.self_attn.k_proj.weight"
        mapping[f"L__self___model_layers_{L}_self_attn_v_proj.weight"] = f"model.layers.{L}.self_attn.v_proj.weight"
        mapping[f"L__self___model_layers_{L}_self_attn_o_proj.weight"] = f"model.layers.{L}.self_attn.o_proj.weight"
        mapping[f"L__self___model_layers_{L}_self_attn_q_norm_weight"] = f"model.layers.{L}.self_attn.q_norm.weight"
        mapping[f"L__self___model_layers_{L}_self_attn_k_norm_weight"] = f"model.layers.{L}.self_attn.k_norm.weight"
        mapping[f"L__self___model_layers_{L}_input_layernorm_weight"] = f"model.layers.{L}.input_layernorm.weight"
        mapping[f"L__self___model_layers_{L}_post_attention_layernorm_weight"] = f"model.layers.{L}.post_attention_layernorm.weight"
        mapping[f"L__self___model_layers_{L}_mlp_gate_proj.weight"] = f"model.layers.{L}.mlp.gate_proj.weight"
        mapping[f"L__self___model_layers_{L}_mlp_up_proj.weight"] = f"model.layers.{L}.mlp.up_proj.weight"
        mapping[f"L__self___model_layers_{L}_mlp_down_proj.weight"] = f"model.layers.{L}.mlp.down_proj.weight"

    mapping["L__self___model_embed_tokens.weight"] = "model.embed_tokens.weight"
    mapping["L__self___model_norm_weight"] = "model.norm.weight"
    mapping["L__self___model_rotary_emb_inv_freq"] = "model.rotary_emb.inv_freq"
    mapping["L__self___lm_head.weight"] = "lm_head.weight"

    return mapping


_KEY_TO_ARG_INDEX = _build_key_to_arg_index()
_FX_TO_STATE_DICT_KEY = _build_fx_to_state_dict()


# Host-resident, row-major, bfloat16: the embedding table and every q/k/v projection.
HOST_BFLOAT16_ROW_MAJOR_WEIGHTS = {
    key for key in _KEY_TO_ARG_INDEX
    if key == "L__self___model_embed_tokens.weight"
    or key.endswith("_self_attn_q_proj.weight")
    or key.endswith("_self_attn_k_proj.weight")
    or key.endswith("_self_attn_v_proj.weight")
}

# Host-resident, row-major, float32: the rotary embedding inverse-frequency buffer.
HOST_FLOAT32_ROW_MAJOR_WEIGHTS = {
    "L__self___model_rotary_emb_inv_freq",
}

# Device-resident, tile-layout, bfloat16, DRAM interleaved: everything else
# (layernorm scales, o_proj, mlp gate/up/down, q_norm/k_norm, lm_head).
DEVICE_BFLOAT16_TILE_WEIGHTS = (
    set(_KEY_TO_ARG_INDEX.keys())
    - HOST_BFLOAT16_ROW_MAJOR_WEIGHTS
    - HOST_FLOAT32_ROW_MAJOR_WEIGHTS
)


_main_weights = {}


def load_weights_for__main():
    """Load all 400 weight tensors for `_main` from on-disk arg<N>.tensorbin files."""
    global _main_weights
    device = utils.DeviceGetter.get_device((1, 1))

    for key, arg_index in _KEY_TO_ARG_INDEX.items():
        path = f"./tensors/arg{arg_index}.tensorbin"

        if key in HOST_BFLOAT16_ROW_MAJOR_WEIGHTS:
            _main_weights[key] = utils.load_tensor(
                path,
                ttnn.Layout.ROW_MAJOR,
                ttnn.DataType.BFLOAT16,
                None,
                None,
            )
        elif key in HOST_FLOAT32_ROW_MAJOR_WEIGHTS:
            _main_weights[key] = utils.load_tensor(
                path,
                ttnn.Layout.ROW_MAJOR,
                ttnn.DataType.FLOAT32,
                None,
                None,
            )
        elif key in DEVICE_BFLOAT16_TILE_WEIGHTS:
            _main_weights[key] = utils.load_tensor(
                path,
                ttnn.Layout.TILE,
                ttnn.DataType.BFLOAT16,
                device,
                ttnn.DRAM_MEMORY_CONFIG,
            )

    return _main_weights


def load_weights_for__main_from_state_dict(state_dict):
    """Build the same weight dict as `load_weights_for__main`, but sourced from a torch state_dict."""
    device = utils.DeviceGetter.get_device((1, 1))
    weights = {}

    for fx_key, sd_key in _FX_TO_STATE_DICT_KEY.items():
        pt_tensor = state_dict[sd_key]
        ttnn_tensor = ttnn.from_torch(pt_tensor)

        if fx_key in HOST_BFLOAT16_ROW_MAJOR_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)
        elif fx_key in HOST_FLOAT32_ROW_MAJOR_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.FLOAT32)
        elif fx_key in DEVICE_BFLOAT16_TILE_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.TILE)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)
            ttnn_tensor = ttnn.to_device(ttnn_tensor, device, ttnn.DRAM_MEMORY_CONFIG)

        weights[fx_key] = ttnn_tensor

    return weights
