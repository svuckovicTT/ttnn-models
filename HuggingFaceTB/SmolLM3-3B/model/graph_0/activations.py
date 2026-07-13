# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Graph activation (input) loader for the SmolLM3-3B TTNN codegen.

``load_inputs`` produces the graph inputs `_main` consumes, in order, by
rebuilding them from the golden PyTorch model's inputs (``model_pt.load_input``).

The graph runs tensor-parallel on a (1, 4) mesh. Its inputs, in the forward-arg
order read from the ``<input>`` args of ``func.func @main`` in ``ttnn.mlir``, are:

    arg217  ttir.name "args_0"  tensor<1x4096xsi32>  -> input_ids
    arg219  ttir.name "args_1"  tensor<1x4096xsi32>  -> attention_mask
    arg220  (no ttir.name)      tensor<bf16>         -> scalar bool (see below)

All three are replicated across the mesh (each device holds an identical copy).
"""
import ttnn
import torch
import model_pt


def _flatten_inputs(obj):
    """Flatten possibly-nested model inputs into a flat list of leaves.

    Fixed order at every level: direct tensors/scalars first, then the contents
    of nested lists/tuples, then the values of nested dicts.
    """
    if isinstance(obj, dict):
        items = list(obj.values())
    elif isinstance(obj, (list, tuple)):
        items = list(obj)
    else:
        items = [obj]

    leaves, nested_lists, nested_dicts = [], [], []
    for item in items:
        if isinstance(item, (list, tuple)):
            nested_lists.append(item)
        elif isinstance(item, dict):
            nested_dicts.append(item)
        else:
            leaves.append(item)

    flat = list(leaves)
    for item in nested_lists:
        flat.extend(_flatten_inputs(item))
    for item in nested_dicts:
        flat.extend(_flatten_inputs(item))
    return flat


def _to_graph_input(torch_tensor, mesh, layout, dtype):
    """Move a torch tensor onto the mesh with a graph input's properties.

    Placed replicated (matching the disk loader's per-input distribution), then
    the property conversions are applied explicitly in order: layout, dtype,
    device (DRAM). ``from_torch`` does not convert the dtype -- ``to_dtype`` does.
    """
    tensor = ttnn.from_torch(
        torch_tensor, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )
    tensor = ttnn.to_layout(tensor, layout)
    tensor = ttnn.to_dtype(tensor, dtype)
    tensor = ttnn.to_device(tensor, mesh, ttnn.DRAM_MEMORY_CONFIG)
    return tensor


def load_inputs(mesh):
    """Rebuild the graph inputs from the golden model's inputs.

    ``model_pt.load_input`` returns ``(input_ids, attention_mask)``; flatten it
    and reorder into the order ``_main`` consumes the inputs (the graph forward
    arg order read from the ``<input>`` ttir.names in ``ttnn.mlir``). The named
    inputs "args_N" map to flattened index N; do not assume the flattened order
    already matches. ``mesh`` is the open TTNN mesh device to place them on.
    """
    flat = _flatten_inputs(model_pt.load_input())
    input_ids = flat[0]  # ttir.name "args_0"
    attention_mask = flat[1]  # ttir.name "args_1"

    # Third graph input (arg220): a bf16 scalar with no ttir.name. The compiler
    # lifted ``torch.any`` over the attention mask's padding -- the "padding is
    # present" flag that is broadcast into the [1, 1, seq, seq] causal mask -- as
    # a graph input. It is not a member of load_input(), so derive it from the
    # golden attention mask here.
    padding_present = torch.any(attention_mask == 0)

    return [
        # args_0: input_ids -> ROW_MAJOR, INT32
        _to_graph_input(input_ids, mesh, ttnn.Layout.ROW_MAJOR, ttnn.DataType.INT32),
        # args_1: attention_mask -> ROW_MAJOR, INT32
        _to_graph_input(
            attention_mask, mesh, ttnn.Layout.ROW_MAJOR, ttnn.DataType.INT32
        ),
        # arg220: scalar padding-present flag -> ROW_MAJOR, BFLOAT16
        _to_graph_input(
            padding_present, mesh, ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16
        ),
    ]
