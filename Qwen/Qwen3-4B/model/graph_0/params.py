# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn

import utils


_NUM_LAYERS = 36

# Weights placed on device with TILE layout, BFLOAT16 dtype, DRAM interleaved.
_TILE_BFLOAT16_DRAM_PER_LAYER_SUFFIXES = (
    "input_layernorm_weight",
    "post_attention_layernorm_weight",
    "self_attn_q_norm_weight",
    "self_attn_k_norm_weight",
    "self_attn_o_proj.weight",
    "mlp_gate_proj.weight",
    "mlp_up_proj.weight",
    "mlp_down_proj.weight",
)
TILE_BFLOAT16_DRAM_WEIGHTS = {
    "L__self___lm_head.weight",
    "L__self___model_norm_weight",
} | {
    f"L__self___model_layers_{i}_{s}"
    for i in range(_NUM_LAYERS)
    for s in _TILE_BFLOAT16_DRAM_PER_LAYER_SUFFIXES
}

# Weights kept on host with ROW_MAJOR layout, BFLOAT16 dtype.
_ROW_MAJOR_BFLOAT16_HOST_PER_LAYER_SUFFIXES = (
    "self_attn_q_proj.weight",
    "self_attn_k_proj.weight",
    "self_attn_v_proj.weight",
)
ROW_MAJOR_BFLOAT16_HOST_WEIGHTS = {
    "L__self___model_embed_tokens.weight",
} | {
    f"L__self___model_layers_{i}_{s}"
    for i in range(_NUM_LAYERS)
    for s in _ROW_MAJOR_BFLOAT16_HOST_PER_LAYER_SUFFIXES
}

# Weights kept on host with ROW_MAJOR layout, FLOAT32 dtype (rotary inv_freq).
ROW_MAJOR_FLOAT32_HOST_WEIGHTS = {"L__self___model_rotary_emb_inv_freq"}


def _build_key_to_arg_index():
    """Map each FX-style weight key to its on-disk ``./tensors/arg<N>.tensorbin`` index.

    The order reproduces exactly how the auto-generated ``load_weights_for__main``
    iterates over the tensors. arg2.tensorbin is reserved for the activation
    tensor, so it is intentionally skipped.
    """
    mapping = {}

    # arg0..arg6: layer 0 v_proj/input_layernorm, then global embed/inv_freq,
    # then layer 0 k_norm/k_proj. arg2 is reserved for the activation tensor.
    mapping["L__self___model_layers_0_self_attn_v_proj.weight"] = 0
    mapping["L__self___model_layers_0_input_layernorm_weight"] = 1
    mapping["L__self___model_embed_tokens.weight"] = 3
    mapping["L__self___model_rotary_emb_inv_freq"] = 4
    mapping["L__self___model_layers_0_self_attn_k_norm_weight"] = 5
    mapping["L__self___model_layers_0_self_attn_k_proj.weight"] = 6

    # arg7..arg391: 35 blocks of 11 entries each. Block L (L=1..35) interleaves
    # layer L's v/input/k_norm/k_proj with layer L-1's mlp/o_proj/q_proj/q_norm/post_layernorm.
    for layer in range(1, _NUM_LAYERS):
        prev = layer - 1
        k = 7 + 11 * (layer - 1)
        mapping[f"L__self___model_layers_{layer}_self_attn_v_proj.weight"] = k
        mapping[f"L__self___model_layers_{layer}_input_layernorm_weight"] = k + 1
        mapping[f"L__self___model_layers_{prev}_mlp_down_proj.weight"] = k + 2
        mapping[f"L__self___model_layers_{prev}_mlp_up_proj.weight"] = k + 3
        mapping[f"L__self___model_layers_{prev}_post_attention_layernorm_weight"] = k + 4
        mapping[f"L__self___model_layers_{prev}_self_attn_o_proj.weight"] = k + 5
        mapping[f"L__self___model_layers_{prev}_self_attn_q_norm_weight"] = k + 6
        mapping[f"L__self___model_layers_{prev}_self_attn_q_proj.weight"] = k + 7
        mapping[f"L__self___model_layers_{prev}_mlp_gate_proj.weight"] = k + 8
        mapping[f"L__self___model_layers_{layer}_self_attn_k_norm_weight"] = k + 9
        mapping[f"L__self___model_layers_{layer}_self_attn_k_proj.weight"] = k + 10

    # arg392..arg400: lm_head + final norm, then the remaining weights of the last layer.
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


_KEY_TO_ARG_INDEX = _build_key_to_arg_index()

_FX_TO_STATE_DICT_KEY = {
    "L__self___model_embed_tokens.weight": "model.embed_tokens.weight",
    "L__self___model_norm_weight": "model.norm.weight",
    "L__self___lm_head.weight": "lm_head.weight",
    "L__self___model_rotary_emb_inv_freq": "model.rotary_emb.inv_freq",
}
for _layer in range(_NUM_LAYERS):
    _FX_TO_STATE_DICT_KEY.update(
        {
            f"L__self___model_layers_{_layer}_self_attn_q_proj.weight": f"model.layers.{_layer}.self_attn.q_proj.weight",
            f"L__self___model_layers_{_layer}_self_attn_k_proj.weight": f"model.layers.{_layer}.self_attn.k_proj.weight",
            f"L__self___model_layers_{_layer}_self_attn_v_proj.weight": f"model.layers.{_layer}.self_attn.v_proj.weight",
            f"L__self___model_layers_{_layer}_self_attn_o_proj.weight": f"model.layers.{_layer}.self_attn.o_proj.weight",
            f"L__self___model_layers_{_layer}_self_attn_q_norm_weight": f"model.layers.{_layer}.self_attn.q_norm.weight",
            f"L__self___model_layers_{_layer}_self_attn_k_norm_weight": f"model.layers.{_layer}.self_attn.k_norm.weight",
            f"L__self___model_layers_{_layer}_input_layernorm_weight": f"model.layers.{_layer}.input_layernorm.weight",
            f"L__self___model_layers_{_layer}_post_attention_layernorm_weight": f"model.layers.{_layer}.post_attention_layernorm.weight",
            f"L__self___model_layers_{_layer}_mlp_gate_proj.weight": f"model.layers.{_layer}.mlp.gate_proj.weight",
            f"L__self___model_layers_{_layer}_mlp_up_proj.weight": f"model.layers.{_layer}.mlp.up_proj.weight",
            f"L__self___model_layers_{_layer}_mlp_down_proj.weight": f"model.layers.{_layer}.mlp.down_proj.weight",
        }
    )
del _layer


_main_weights = {}


def load_weights_for__main():
    """Load all 400 weight tensors for ``_main`` from ``./tensors/arg<N>.tensorbin`` files."""
    global _main_weights
    device = utils.DeviceGetter.get_device((1, 1))

    for key, arg_index in _KEY_TO_ARG_INDEX.items():
        path = f"./tensors/arg{arg_index}.tensorbin"

        if key in TILE_BFLOAT16_DRAM_WEIGHTS:
            _main_weights[key] = utils.load_tensor(
                path,
                ttnn.Layout.TILE,
                ttnn.DataType.BFLOAT16,
                device,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
                ),
            )
        elif key in ROW_MAJOR_BFLOAT16_HOST_WEIGHTS:
            _main_weights[key] = utils.load_tensor(
                path,
                ttnn.Layout.ROW_MAJOR,
                ttnn.DataType.BFLOAT16,
                None,
                None,
            )
        elif key in ROW_MAJOR_FLOAT32_HOST_WEIGHTS:
            _main_weights[key] = utils.load_tensor(
                path,
                ttnn.Layout.ROW_MAJOR,
                ttnn.DataType.FLOAT32,
                None,
                None,
            )

    return _main_weights


def load_weights_for__main_from_state_dict(state_dict):
    """Build the same weight dict as ``load_weights_for__main``, but sourced from a torch state_dict."""
    device = utils.DeviceGetter.get_device((1, 1))
    weights = {}

    for fx_key in _KEY_TO_ARG_INDEX:
        pt_tensor = state_dict[_FX_TO_STATE_DICT_KEY[fx_key]]
        ttnn_tensor = ttnn.from_torch(pt_tensor)

        if fx_key in TILE_BFLOAT16_DRAM_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.TILE)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)
            ttnn_tensor = ttnn.to_device(ttnn_tensor, device, ttnn.DRAM_MEMORY_CONFIG)
        elif fx_key in ROW_MAJOR_BFLOAT16_HOST_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)
        elif fx_key in ROW_MAJOR_FLOAT32_HOST_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.FLOAT32)

        weights[fx_key] = ttnn_tensor

    return weights
