# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Weight loaders for the SmolLM3-3B TTNN codegen.

``load_weights_for__main_from_state_dict`` produces the weight dict `_main`
consumes, keyed by parameter name, by rebuilding the weights from the golden
PyTorch model's ``state_dict`` (``model_pt.load_pytorch_model``).

The graph runs tensor-parallel (Megatron-1D) on a (1, 4) mesh. Each weight is
placed with a fixed distribution that is part of the compiled graph's contract:

- Column-parallel projections (attention q/k/v, MLP gate/up) are sharded along
  their output dim (dim 0).
- Row-parallel projections (attention o, MLP down) are sharded along their
  contracting dim (dim 1).
- Everything else (embeddings, RMSNorm weights, rotary ``inv_freq``) is
  replicated -- every device holds an identical copy.

and with a fixed set of tensor properties (layout / dtype / placement):

- On-device weights (RMSNorm weights, o/gate/up/down projections): TILE layout,
  bfloat16, resident in DRAM.
- Host weights (embeddings, q/k/v projections): ROW_MAJOR layout, bfloat16.
- The rotary ``inv_freq`` buffer: ROW_MAJOR layout, float32, on host.
"""
import ttnn
import torch
import utils
import model_pt


# Number of decoder layers in SmolLM3-3B.
NUM_LAYERS = 36

# The graph runs tensor-parallel on a (1, 4) mesh.
MESH_SHAPE = (1, 4)
FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_1D_RING


def _per_layer(suffix):
    """Weight names ``model.layers.{i}.{suffix}`` for every decoder layer."""
    return {f"model.layers.{i}.{suffix}" for i in range(NUM_LAYERS)}


# --- Mesh distribution groups (Megatron-1D), matching the disk loader. -------
# Column-parallel: sharded along the weight's output dim (dim 0).
SHARD_DIM0_WEIGHTS = (
    _per_layer("self_attn.q_proj.weight")
    | _per_layer("self_attn.k_proj.weight")
    | _per_layer("self_attn.v_proj.weight")
    | _per_layer("mlp.gate_proj.weight")
    | _per_layer("mlp.up_proj.weight")
)
# Row-parallel: sharded along the weight's contracting dim (dim 1).
SHARD_DIM1_WEIGHTS = (
    _per_layer("self_attn.o_proj.weight")
    | _per_layer("mlp.down_proj.weight")
)
# Everything not in the two sets above is replicated across the mesh.

# --- Tensor-property groups (layout / dtype / placement). --------------------
# TILE layout, bfloat16, resident on device in DRAM.
TILE_DEVICE_BF16_WEIGHTS = (
    {"model.norm.weight"}
    | _per_layer("input_layernorm.weight")
    | _per_layer("post_attention_layernorm.weight")
    | _per_layer("self_attn.o_proj.weight")
    | _per_layer("mlp.gate_proj.weight")
    | _per_layer("mlp.up_proj.weight")
    | _per_layer("mlp.down_proj.weight")
)
# ROW_MAJOR layout, bfloat16, on host.
ROW_MAJOR_HOST_BF16_WEIGHTS = (
    {"model.embed_tokens.weight"}
    | _per_layer("self_attn.q_proj.weight")
    | _per_layer("self_attn.k_proj.weight")
    | _per_layer("self_attn.v_proj.weight")
)
# ROW_MAJOR layout, float32, on host (the rotary inv_freq buffer).
ROW_MAJOR_HOST_FP32_WEIGHTS = {"model.rotary_emb.inv_freq"}

ALL_WEIGHTS = (
    TILE_DEVICE_BF16_WEIGHTS
    | ROW_MAJOR_HOST_BF16_WEIGHTS
    | ROW_MAJOR_HOST_FP32_WEIGHTS
)


def load_weights_for__main_from_state_dict():
    """Rebuild the graph weights from the golden PyTorch model's state_dict.

    Returns the weight dict `_main` consumes, keyed by parameter name, with each
    weight's tensor properties (layout / dtype / placement) and mesh distribution
    (replicated vs sharded, and the shard dim) set per the compiled graph's
    contract. Weights are placed with the mesh mapper the graph expects -- a
    blanket replicate would break the compiled matmuls for the sharded
    projections.
    """
    device = utils.DeviceGetter.get_device(MESH_SHAPE, fabric_config=FABRIC_CONFIG)

    model = model_pt.load_pytorch_model()

    # model.state_dict() excludes non-persistent buffers (rotary_emb.inv_freq),
    # which the graph needs -- add them back from named_buffers().
    state_dict = dict(model.state_dict())
    for name, buf in model.named_buffers():
        if name not in state_dict:
            state_dict[name] = buf

    weights = {}
    for key in ALL_WEIGHTS:
        pt_tensor = state_dict[key]

        # Distribution: pick the mesh mapper matching the disk loader.
        if key in SHARD_DIM0_WEIGHTS:
            mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        elif key in SHARD_DIM1_WEIGHTS:
            mesh_mapper = ttnn.ShardTensorToMesh(device, dim=1)
        else:
            mesh_mapper = ttnn.ReplicateTensorToMesh(device)

        ttnn_tensor = ttnn.from_torch(pt_tensor, mesh_mapper=mesh_mapper)

        # Properties: apply layout, dtype, then device -- in that order.
        if key in TILE_DEVICE_BF16_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.TILE)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)
            ttnn_tensor = ttnn.to_device(ttnn_tensor, device, ttnn.DRAM_MEMORY_CONFIG)
        elif key in ROW_MAJOR_HOST_BF16_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)
        elif key in ROW_MAJOR_HOST_FP32_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.FLOAT32)

        weights[key] = ttnn_tensor

    return weights
