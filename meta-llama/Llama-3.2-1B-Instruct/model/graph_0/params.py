# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import utils
import model_pt


_LAYER_INDICES = range(16)

# Layer-norm weights (and the final RMSNorm) are loaded onto the device in TILE
# layout. Everything else stays on host in ROW_MAJOR layout.
TILE_WEIGHTS = {
    "model.norm.weight",
    *(f"model.layers.{i}.input_layernorm.weight" for i in _LAYER_INDICES),
    *(f"model.layers.{i}.post_attention_layernorm.weight" for i in _LAYER_INDICES),
}

HOST_WEIGHTS = {
    "lm_head.weight",
    "model.embed_tokens.weight",
    "model.rotary_emb.inv_freq",
    *(
        f"model.layers.{i}.{name}"
        for i in _LAYER_INDICES
        for name in (
            "mlp.down_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        )
    ),
}

ALL_WEIGHTS = TILE_WEIGHTS | HOST_WEIGHTS


def load_weights_for__main_from_state_dict():
    device = utils.DeviceGetter.get_device((1, 1))

    model = model_pt.load_pytorch_model()
    state_dict = dict(model.state_dict())
    for name, buf in model.named_buffers():
        if name not in state_dict:
            state_dict[name] = buf

    weights = {}
    for key in ALL_WEIGHTS:
        ttnn_tensor = ttnn.from_torch(state_dict[key])

        if key in TILE_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.TILE)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)
            ttnn_tensor = ttnn.to_device(ttnn_tensor, device, ttnn.DRAM_MEMORY_CONFIG)

        if key in HOST_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)

        weights[key] = ttnn_tensor

    return weights
