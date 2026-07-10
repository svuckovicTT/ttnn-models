# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re

import ttnn
import model_pt

# The FIBO transformer's weights follow a fully regular naming scheme, so instead
# of enumerating all 846 names we generate them from the module structure and
# derive each weight's (1, 4)-mesh placement from its name.

# Fields present on every block (index is filled in per block below).
_SINGLE_BLOCK_FIELDS = [
    "attn.norm_q.weight", "attn.norm_k.weight",
    "attn.to_q.weight", "attn.to_q.bias",
    "attn.to_k.weight", "attn.to_k.bias",
    "attn.to_v.weight", "attn.to_v.bias",
    "norm.linear.weight", "norm.linear.bias",
    "proj_mlp.weight", "proj_mlp.bias",
    "proj_out.weight", "proj_out.bias",
]
_DOUBLE_BLOCK_FIELDS = [
    "norm1.linear.weight", "norm1.linear.bias",
    "norm1_context.linear.weight", "norm1_context.linear.bias",
    "attn.norm_q.weight", "attn.norm_k.weight",
    "attn.norm_added_q.weight", "attn.norm_added_k.weight",
    "attn.to_q.weight", "attn.to_q.bias",
    "attn.to_k.weight", "attn.to_k.bias",
    "attn.to_v.weight", "attn.to_v.bias",
    "attn.add_q_proj.weight", "attn.add_q_proj.bias",
    "attn.add_k_proj.weight", "attn.add_k_proj.bias",
    "attn.add_v_proj.weight", "attn.add_v_proj.bias",
    "attn.to_out.0.weight", "attn.to_out.0.bias",
    "attn.to_add_out.weight", "attn.to_add_out.bias",
    "ff.net.0.proj.weight", "ff.net.0.proj.bias",
    "ff.net.2.weight", "ff.net.2.bias",
    "ff_context.net.0.proj.weight", "ff_context.net.0.proj.bias",
    "ff_context.net.2.weight", "ff_context.net.2.bias",
]
_SINGLETONS = [
    "transformer.context_embedder.weight", "transformer.context_embedder.bias",
    "transformer.x_embedder.weight", "transformer.x_embedder.bias",
    "transformer.time_embed.timestep_embedder.linear_1.weight",
    "transformer.time_embed.timestep_embedder.linear_1.bias",
    "transformer.time_embed.timestep_embedder.linear_2.weight",
    "transformer.time_embed.timestep_embedder.linear_2.bias",
    "transformer.norm_out.linear.weight", "transformer.norm_out.linear.bias",
    "transformer.proj_out.weight", "transformer.proj_out.bias",
]

_NUM_SINGLE_BLOCKS = 38
_NUM_DOUBLE_BLOCKS = 8
_NUM_CAPTION_PROJECTIONS = 46


def _weight_names():
    names = list(_SINGLETONS)
    names += [f"transformer.caption_projection.{i}.linear.weight"
              for i in range(_NUM_CAPTION_PROJECTIONS)]
    for i in range(_NUM_SINGLE_BLOCKS):
        names += [f"transformer.single_transformer_blocks.{i}.{f}"
                  for f in _SINGLE_BLOCK_FIELDS]
    for i in range(_NUM_DOUBLE_BLOCKS):
        names += [f"transformer.transformer_blocks.{i}.{f}"
                  for f in _DOUBLE_BLOCK_FIELDS]
    return names


# Tensor-parallel (Megatron) sharding across the (1, 4) mesh:
#   column-parallel linears (q/k/v, add_*_proj, proj_mlp, ff up-proj) split on dim 0
#     (both weight and bias); row-parallel linears (out-proj, ff down-proj,
#     modulation norms) split on dim 1 (weight only -- their bias is replicated).
#   Everything else is replicated.
_SHARD_DIM0 = re.compile(
    r"\.(to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj|proj_mlp)\.(weight|bias)$"
    r"|\.net\.\d+\.proj\.(weight|bias)$"
)
_SHARD_DIM1 = re.compile(
    r"\.to_out\.\d+\.weight$"
    r"|\.to_add_out\.weight$"
    r"|single_transformer_blocks\.\d+\.proj_out\.weight$"
    r"|\.net\.\d+\.weight$"
    r"|\.(norm|norm1|norm1_context|norm_out)\.linear\.weight$"
)

# Weights placed on device in TILE layout (matmul operands and attention-RMSNorm
# weights); everything else stays ROW_MAJOR on host.
_TILED = re.compile(
    r"^transformer\.(context_embedder|x_embedder|proj_out)\.(weight|bias)$"
    r"|caption_projection\.\d+\.linear\.weight$"
    r"|\.(norm_q|norm_k|norm_added_q|norm_added_k)\.weight$"
    r"|\.to_out\.\d+\.weight$"
    r"|\.to_add_out\.weight$"
    r"|single_transformer_blocks\.\d+\.proj_out\.weight$"
    r"|\.net\.\d+\.proj\.(weight|bias)$"
    r"|\.net\.\d+\.weight$"
)


def _shard_dim(name):
    if _SHARD_DIM0.search(name):
        return 0
    if _SHARD_DIM1.search(name):
        return 1
    return None


def load_weights_for__main(device):
    model = model_pt.load_pytorch_model()
    sd = dict(model.state_dict())
    for name, buf in model.named_buffers():
        sd.setdefault(name, buf)

    weights = {}
    for key in _weight_names():
        shard_dim = _shard_dim(key)
        if shard_dim is not None:
            mesh_mapper = ttnn.ShardTensorToMesh(device, dim=shard_dim)
        else:
            mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        tensor = ttnn.from_torch(sd[key], mesh_mapper=mesh_mapper)

        if _TILED.search(key):
            tensor = ttnn.to_layout(tensor, ttnn.Layout.TILE)
            tensor = ttnn.to_dtype(tensor, ttnn.DataType.BFLOAT16)
            tensor = ttnn.to_device(tensor, device, ttnn.DRAM_MEMORY_CONFIG)
        else:
            tensor = ttnn.to_layout(tensor, ttnn.Layout.ROW_MAJOR)
            tensor = ttnn.to_dtype(tensor, ttnn.DataType.BFLOAT16)

        weights[key] = tensor
    return weights
