# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn

import model_pt
import utils


NUM_LAYERS = 36


# ---------------------------------------------------------------------------
# Weight key groupings by shared TTNN property set.
#
# Three groups cover every weight loaded by this graph:
#   1. ROW_MAJOR + BFLOAT16, host-resident (no device transfer).
#   2. ROW_MAJOR + FLOAT32, host-resident.
#   3. TILE     + BFLOAT16, on-device in DRAM (interleaved).
# ---------------------------------------------------------------------------

ROW_MAJOR_HOST_BFLOAT16_WEIGHTS = {
    "L__self___model_embed_tokens.weight",
} | {
    f"L__self___model_layers_{i}_{suffix}"
    for i in range(NUM_LAYERS)
    for suffix in (
        "self_attn_q_proj.weight",
        "self_attn_k_proj.weight",
        "self_attn_v_proj.weight",
    )
}

ROW_MAJOR_HOST_FLOAT32_WEIGHTS = {
    "L__self___model_rotary_emb_inv_freq",
}

TILE_DEVICE_BFLOAT16_WEIGHTS = {
    "L__self___lm_head.weight",
    "L__self___model_norm_weight",
} | {
    f"L__self___model_layers_{i}_{suffix}"
    for i in range(NUM_LAYERS)
    for suffix in (
        "input_layernorm_weight",
        "post_attention_layernorm_weight",
        "self_attn_q_norm_weight",
        "self_attn_k_norm_weight",
        "self_attn_o_proj.weight",
        "mlp_gate_proj.weight",
        "mlp_up_proj.weight",
        "mlp_down_proj.weight",
    )
}

ALL_WEIGHTS = (
    ROW_MAJOR_HOST_BFLOAT16_WEIGHTS
    | ROW_MAJOR_HOST_FLOAT32_WEIGHTS
    | TILE_DEVICE_BFLOAT16_WEIGHTS
)


# ---------------------------------------------------------------------------
# Disk loader: original behavior, preserved.
#
# The auto-generated graph emits each weight as `argN.tensorbin` on disk. The
# mapping below is the original load order; arg 2 is skipped because it is the
# graph's input activations tensor.
# ---------------------------------------------------------------------------


def _build_key_to_arg_index():
    """Reconstruct the original key -> arg index mapping.

    The auto-generated load order is: 6 prefix entries, followed by 35 cycles
    of 11 entries each (one cycle per layer transition L -> L+1, for L in
    0..34), followed by a 9-entry tail that wraps up the final layer.
    """
    keys_in_load_order = [
        "L__self___model_layers_0_self_attn_v_proj.weight",
        "L__self___model_layers_0_input_layernorm_weight",
        "L__self___model_embed_tokens.weight",
        "L__self___model_rotary_emb_inv_freq",
        "L__self___model_layers_0_self_attn_k_norm_weight",
        "L__self___model_layers_0_self_attn_k_proj.weight",
    ]
    for layer in range(NUM_LAYERS - 1):
        nxt = layer + 1
        keys_in_load_order.extend(
            [
                f"L__self___model_layers_{nxt}_self_attn_v_proj.weight",
                f"L__self___model_layers_{nxt}_input_layernorm_weight",
                f"L__self___model_layers_{layer}_mlp_down_proj.weight",
                f"L__self___model_layers_{layer}_mlp_up_proj.weight",
                f"L__self___model_layers_{layer}_post_attention_layernorm_weight",
                f"L__self___model_layers_{layer}_self_attn_o_proj.weight",
                f"L__self___model_layers_{layer}_self_attn_q_norm_weight",
                f"L__self___model_layers_{layer}_self_attn_q_proj.weight",
                f"L__self___model_layers_{layer}_mlp_gate_proj.weight",
                f"L__self___model_layers_{nxt}_self_attn_k_norm_weight",
                f"L__self___model_layers_{nxt}_self_attn_k_proj.weight",
            ]
        )
    last = NUM_LAYERS - 1
    keys_in_load_order.extend(
        [
            "L__self___lm_head.weight",
            "L__self___model_norm_weight",
            f"L__self___model_layers_{last}_mlp_down_proj.weight",
            f"L__self___model_layers_{last}_mlp_up_proj.weight",
            f"L__self___model_layers_{last}_post_attention_layernorm_weight",
            f"L__self___model_layers_{last}_self_attn_o_proj.weight",
            f"L__self___model_layers_{last}_self_attn_q_norm_weight",
            f"L__self___model_layers_{last}_self_attn_q_proj.weight",
            f"L__self___model_layers_{last}_mlp_gate_proj.weight",
        ]
    )

    # arg 2 is reserved for the graph's input activations, so the weight
    # tensor at load position `i` lives at arg index `i` for i < 2 and `i + 1`
    # otherwise.
    return {
        key: (i if i < 2 else i + 1)
        for i, key in enumerate(keys_in_load_order)
    }


_KEY_TO_ARG_INDEX = _build_key_to_arg_index()


_main_weights = {}


def load_weights_for__main():
    device = utils.DeviceGetter.get_device((1, 1))
    global _main_weights

    for key in ALL_WEIGHTS:
        path = f"./tensors/arg{_KEY_TO_ARG_INDEX[key]}.tensorbin"

        if key in ROW_MAJOR_HOST_BFLOAT16_WEIGHTS:
            _main_weights[key] = utils.load_tensor(
                path,
                ttnn.Layout.ROW_MAJOR,
                ttnn.DataType.BFLOAT16,
                None,
                None,
            )
        elif key in ROW_MAJOR_HOST_FLOAT32_WEIGHTS:
            _main_weights[key] = utils.load_tensor(
                path,
                ttnn.Layout.ROW_MAJOR,
                ttnn.DataType.FLOAT32,
                None,
                None,
            )
        elif key in TILE_DEVICE_BFLOAT16_WEIGHTS:
            _main_weights[key] = utils.load_tensor(
                path,
                ttnn.Layout.TILE,
                ttnn.DataType.BFLOAT16,
                device,
                ttnn.DRAM_MEMORY_CONFIG,
            )

    return _main_weights


# ---------------------------------------------------------------------------
# State-dict loader: same TTNN tensors built from the golden PyTorch model's
# state_dict instead of from serialized tensor files.
# ---------------------------------------------------------------------------


_PER_LAYER_TTNN_TO_TORCH = {
    "input_layernorm_weight": "input_layernorm.weight",
    "post_attention_layernorm_weight": "post_attention_layernorm.weight",
    "self_attn_q_norm_weight": "self_attn.q_norm.weight",
    "self_attn_k_norm_weight": "self_attn.k_norm.weight",
    "self_attn_q_proj.weight": "self_attn.q_proj.weight",
    "self_attn_k_proj.weight": "self_attn.k_proj.weight",
    "self_attn_v_proj.weight": "self_attn.v_proj.weight",
    "self_attn_o_proj.weight": "self_attn.o_proj.weight",
    "mlp_gate_proj.weight": "mlp.gate_proj.weight",
    "mlp_up_proj.weight": "mlp.up_proj.weight",
    "mlp_down_proj.weight": "mlp.down_proj.weight",
}

_GLOBAL_TTNN_TO_TORCH = {
    "L__self___model_embed_tokens.weight": "model.embed_tokens.weight",
    "L__self___model_rotary_emb_inv_freq": "model.rotary_emb.inv_freq",
    "L__self___model_norm_weight": "model.norm.weight",
    "L__self___lm_head.weight": "lm_head.weight",
}


def _ttnn_key_to_torch_key(ttnn_key):
    if ttnn_key in _GLOBAL_TTNN_TO_TORCH:
        return _GLOBAL_TTNN_TO_TORCH[ttnn_key]

    prefix = "L__self___model_layers_"
    assert ttnn_key.startswith(prefix), f"Unrecognized weight key: {ttnn_key}"
    layer_str, _, suffix = ttnn_key[len(prefix):].partition("_")
    return f"model.layers.{layer_str}.{_PER_LAYER_TTNN_TO_TORCH[suffix]}"


def load_weights_for__main_from_state_dict():
    device = utils.DeviceGetter.get_device((1, 1))
    state_dict = model_pt.load_pytorch_model().state_dict()

    weights = {}
    for key in ALL_WEIGHTS:
        pt_tensor = state_dict[_ttnn_key_to_torch_key(key)]
        ttnn_tensor = ttnn.from_torch(pt_tensor)

        if key in ROW_MAJOR_HOST_BFLOAT16_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)
        elif key in ROW_MAJOR_HOST_FLOAT32_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.FLOAT32)
        elif key in TILE_DEVICE_BFLOAT16_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.TILE)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)
            ttnn_tensor = ttnn.to_device(ttnn_tensor, device, ttnn.DRAM_MEMORY_CONFIG)

        weights[key] = ttnn_tensor

    return weights
