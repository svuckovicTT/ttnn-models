import torch
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


def load_weights_for__main_from_state_dict(device):
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

    # The disk tensorbins are distributed across the 4x8 mesh. To make these
    # weights usable as a drop-in replacement we must reproduce that exact
    # distribution rather than producing single-device tensors.
    MESH_ROWS, MESH_COLS = 4, 8

    # Replicated on every device: norms, embeddings, rotary freqs, the router
    # gate weight and its e_score correction bias.
    REPLICATED = NORM_WEIGHTS | {
        "model.model.embed_tokens.weight",
        "model.model.rotary_emb.inv_freq",
        "model.model.layers.3.mlp.mlp.router.gate.weight",
        _E_SCORE_MANGLED_KEY,
    }
    # Sharded along the mesh's 8-axis (replicated across the 4-axis) on tensor
    # dim 0 (output features) ...
    SHARD_DIM0 = (
        {"model.lm_head.weight"}
        | {
            f"model.model.layers.{i}.self_attn.{p}"
            for i in range(4)
            for p in [
                "q_proj.weight", "k_proj.weight", "v_proj.weight",
                "q_proj.bias", "k_proj.bias", "v_proj.bias",
            ]
        }
        | {
            f"model.model.layers.{i}.mlp.{p}.weight"
            for i in range(3)
            for p in ["gate_proj", "up_proj"]
        }
        | {
            "model.model.layers.3.mlp.shared_experts.gate_proj.weight",
            "model.model.layers.3.mlp.shared_experts.up_proj.weight",
        }
    )
    # ... or on tensor dim 1 (input features, for the down/out projections).
    SHARD_DIM1 = (
        {f"model.model.layers.{i}.self_attn.o_proj.weight" for i in range(4)}
        | {f"model.model.layers.{i}.mlp.down_proj.weight" for i in range(3)}
        | {"model.model.layers.3.mlp.shared_experts.down_proj.weight"}
    )
    # Experts are sharded across the whole mesh (one block of experts per device)
    # in column-major device order, and stored transposed on their last two dims.
    EXPERTS = {
        f"model.model.layers.3.mlp.mlp.experts.{p}"
        for p in ["gate_proj", "up_proj", "down_proj"]
    }

    def _arrange_experts(t):
        # (num_experts, A, B) -> transpose each expert -> reorder expert blocks
        # into physical (row-major) device order so a 1-D shard over the mesh
        # reproduces the codegen's column-major expert layout.
        t = t.transpose(-1, -2).contiguous()
        per = t.shape[0] // (MESH_ROWS * MESH_COLS)
        blocks = []
        for p in range(MESH_ROWS * MESH_COLS):
            block = (p % MESH_COLS) * MESH_ROWS + (p // MESH_COLS)
            blocks.append(t[per * block : per * block + per])
        return torch.cat(blocks, dim=0)

    weights = {}
    for key in ALL_WEIGHTS:
        if key in INT32_WEIGHTS:
            # expert_mapping is a routing constant with no HF state_dict origin.
            # It is a one-hot [1, 1, num_experts, num_devices] tensor assigning a
            # contiguous block of experts to each device, replicated on every
            # device. Fully determined by the model config and mesh, so we build
            # it at runtime instead of carrying a serialized constant.
            #
            # The physical expert placement is column-major: _arrange_experts()
            # reorders blocks so the 1-D ShardTensorToMesh(dim 0) lands expert
            # block B on device d = MESH_COLS*r + c where c*MESH_ROWS + r == B.
            # The dispatch/combine must therefore route expert e (block
            # e // experts_per_device) to that same device. Routing row-major
            # (device e // experts_per_device), as the old MoE sharding did,
            # sends tokens to the device holding a DIFFERENT block of experts and
            # caps decode PCC (~0.86 here). See tt-xla issue 5096: the correct
            # ("batch","model") layout = contiguous experts per device; we
            # reproduce it by inverting the column-major placement below.
            num_experts = model.config.n_routed_experts
            num_devices = MESH_ROWS * MESH_COLS
            assert num_experts % num_devices == 0, (
                f"expert_mapping assumes experts ({num_experts}) divide evenly "
                f"across devices ({num_devices})"
            )
            experts_per_device = num_experts // num_devices
            block = torch.arange(num_experts) // experts_per_device
            row = block % MESH_ROWS
            col = block // MESH_ROWS
            device_of_expert = row * MESH_COLS + col
            expert_mapping = torch.zeros(
                1, 1, num_experts, num_devices, dtype=torch.int32
            )
            expert_mapping[0, 0, torch.arange(num_experts), device_of_expert] = 1
            weights[key] = ttnn.from_torch(
                expert_mapping,
                dtype=ttnn.DataType.INT32,
                layout=ttnn.Layout.ROW_MAJOR,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
            continue

        pt_tensor = sd[key]

        if key in REPLICATED:
            mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        elif key in SHARD_DIM0:
            mesh_mapper = ttnn.ShardTensor2dMesh(
                device, (MESH_ROWS, MESH_COLS), (None, 0)
            )
        elif key in SHARD_DIM1:
            mesh_mapper = ttnn.ShardTensor2dMesh(
                device, (MESH_ROWS, MESH_COLS), (None, 1)
            )
        elif key in EXPERTS:
            pt_tensor = _arrange_experts(pt_tensor)
            mesh_mapper = ttnn.ShardTensorToMesh(device, 0)
        else:
            raise KeyError(f"No mesh distribution defined for weight '{key}'")

        ttnn_tensor = ttnn.from_torch(pt_tensor, mesh_mapper=mesh_mapper)

        if key in NORM_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.TILE)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)
            ttnn_tensor = ttnn.to_device(ttnn_tensor, device, ttnn.DRAM_MEMORY_CONFIG)
        else:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)

        weights[key] = ttnn_tensor

    return weights
