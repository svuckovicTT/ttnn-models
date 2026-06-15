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

    # The traced graph names the MoE router submodule "router.gate", but the HF
    # state_dict names it just "gate" (transformers Glm4MoeMoE.gate).
    _ROUTER_WEIGHT_DISK_KEY = "model.model.layers.3.mlp.mlp.router.gate.weight"
    for candidate in [
        "model.model.layers.3.mlp.gate.weight",
        "model.model.layers.3.mlp.mlp.gate.weight",
    ]:
        if candidate in sd:
            sd[_ROUTER_WEIGHT_DISK_KEY] = sd.pop(candidate)
            break

    _E_SCORE_MANGLED_KEY = (
        "L__self___model_model_layers_3_mlp_mlp_router"
        "__route_fn___closure___0_cell_contents_e_score_correction_bias"
    )
    for candidate in [
        "model.model.layers.3.mlp.gate.e_score_correction_bias",
        "model.model.layers.3.mlp.mlp.gate.e_score_correction_bias",
    ]:
        if candidate in sd:
            sd[_E_SCORE_MANGLED_KEY] = sd.pop(candidate)
            break

    for key in ALL_WEIGHTS:
        if key not in sd and ".mlp.mlp." in key:
            single_mlp_key = key.replace(".mlp.mlp.", ".mlp.", 1)
            if single_mlp_key in sd:
                sd[key] = sd.pop(single_mlp_key)

    # expert_mapping is a routing constant baked into the traced graph (it maps
    # experts to mesh devices); it has no HF state_dict origin and is stored as a
    # multi-device tensor, so load it directly from the serialized constant the
    # disk loader uses rather than reconstructing it from a torch tensor.
    _EXPERT_MAPPING_KEY = "model.model.layers.3.mlp.mlp.expert_mapping"

    device = utils.DeviceGetter.get_device((4, 8))

    weights = {}
    for key in ALL_WEIGHTS:
        if key == _EXPERT_MAPPING_KEY:
            weights[key] = utils.load_tensor(
                "./tensors/arg76.tensorbin",
                ttnn.Layout.ROW_MAJOR,
                ttnn.DataType.INT32,
                None,
                None,
            )
            continue

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
