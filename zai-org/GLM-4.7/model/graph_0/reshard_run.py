"""Pure-ttnn reshard (no torch_xla). Override the broken single-shard weights in
main's weights dict with correctly-sharded multi-device HOST tensors built from
the full HF weights, then run _main and PCC vs the CPU golden.

Sharding is derived from the MLIR @main arg table (global vs local shape):
  factor 8  -> shard that dim across the model axis (mesh axis 1), replicate batch
  factor 32 -> shard that dim across BOTH axes (the EP-32 experts)
Each resharded weight's device-0 shard is verified against tensors/argN.tensorbin
(proven to be the correct device-0 slice) before running.
"""
import json
import re
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

import ttnn
import utils
import main

MODEL = "zai-org/GLM-4.7"
MESH = (4, 8)
GRAPH = "/home/svuckovic/_workspace/repos/project-alchemy/ttnn-models/zai-org/GLM-4.7/model/graph_0"

mesh = utils.DeviceGetter.get_device(MESH)


# --- 1. parse MLIR @main arg table: ttir.name -> (idx, global_shape, local_shape)
def parse_args():
    sig = re.search(r'func\.func @main\((.*?)\)\s*(->|\{)', open(f"{GRAPH}/ttnn.mlir").read(), re.S).group(1)
    out = {}
    for i, body in enumerate(re.split(r'%arg\d+:', sig)[1:]):
        g = re.search(r'tensor<([0-9x]+)[a-z]', body)
        l = re.search(r'local_shape = tensor<([0-9x]+)[a-z]', body)
        nm = re.search(r'ttir\.name = "([^"]+)"', body)
        if not (g and l and nm):
            continue
        gs = tuple(int(x) for x in g.group(1).rstrip('x').split('x'))
        ls = tuple(int(x) for x in l.group(1).rstrip('x').split('x'))
        out[nm.group(1)] = (i, gs, ls)
    return out


ARGS = parse_args()

# --- 2. HF weight resolver (lazy per-tensor; never loads a whole shard file)
_idx = json.load(open(hf_hub_download(MODEL, "model.safetensors.index.json")))["weight_map"]
_handles = {}
def hf_get(key):
    path = hf_hub_download(MODEL, _idx[key])
    if path not in _handles:
        _handles[path] = safe_open(path, framework="pt")
    return _handles[path].get_tensor(key)


def full_weight(ttir_name):
    hf = ttir_name[len("model."):]  # strip the wrapper's leading "model."
    m = re.search(r"layers\.(\d+)\.mlp\.mlp\.experts\.(\w+)$", ttir_name)
    if m:  # MoE experts: stack 160 per-expert HF weights, transpose last two dims
        layer, proj = m.group(1), m.group(2)
        ws = [hf_get(f"model.layers.{layer}.mlp.experts.{j}.{proj}.weight").to(torch.bfloat16)
              for j in range(160)]
        st = torch.stack(ws).transpose(1, 2).contiguous()
        del ws
        return st
    return hf_get(hf).to(torch.bfloat16).contiguous()


def make_mapper(g, l):
    sdims = [d for d in range(len(g)) if g[d] != l[d]]
    assert len(sdims) == 1, (g, l)
    d = sdims[0]
    factor = g[d] // l[d]
    if factor == 8:      # model axis only: replicate batch, shard dim d on model
        plc = [ttnn.PlacementReplicate(), ttnn.PlacementShard(d)]
        return ttnn.create_mesh_mapper(mesh, ttnn.MeshMapperConfig(plc, ttnn.MeshShape(*MESH)))
    elif factor == 32:   # EP-32: shard dim d across all 32 devices (flat, device order)
        return ttnn.ShardTensorToMesh(mesh, dim=d)
    raise ValueError(f"unexpected shard factor {factor} for {g}->{l}")


# --- 3. build weights dict: tensorbins for replicated, resharded HF for sharded
weights = main.load_weights_for__main()
n_over, n_bad = 0, 0
for ck in list(weights.keys()):
    if ck not in ARGS:
        continue
    idx, g, l = ARGS[ck]
    if g == l:
        continue  # replicated -> tensorbin already correct
    full = full_weight(ck)
    assert tuple(full.shape) == g, f"{ck}: HF shape {tuple(full.shape)} != codegen global {g}"
    rt = ttnn.from_torch(full, dtype=ttnn.DataType.BFLOAT16,
                         layout=ttnn.Layout.ROW_MAJOR, mesh_mapper=make_mapper(g, l))
    # verify device-0 shard == the saved tensorbin (proven correct device-0 slice)
    dev0 = ttnn.to_torch(ttnn.get_device_tensors(rt)[0]).to(torch.bfloat16)
    tb = ttnn.load_tensor(f"{GRAPH}/tensors/arg{idx}.tensorbin")
    if tb.layout != ttnn.Layout.ROW_MAJOR:
        tb = ttnn.to_layout(tb, ttnn.Layout.ROW_MAJOR)
    tb = ttnn.to_torch(tb).to(torch.bfloat16)
    ok = tuple(dev0.shape) == tuple(tb.shape) and torch.equal(dev0, tb)
    if not ok:
        n_bad += 1
        print(f"  MISMATCH {ck}: dev0{tuple(dev0.shape)} vs tb{tuple(tb.shape)} "
              f"equal={torch.equal(dev0, tb) if dev0.shape==tb.shape else 'shape'}", flush=True)
    weights[ck] = rt
    del full
    n_over += 1
print(f"overrode {n_over} sharded weights; device-0 mismatches: {n_bad}", flush=True)

# --- 4. run + PCC
outs = main._main(main.load_activations_for__main(), weights)
logits = ttnn.to_torch(ttnn.get_device_tensors(ttnn.from_device(outs[15]))[0]).float().cpu()
torch.save(logits, "/tmp/resharded_logits.pt")
print(f"resharded logits {tuple(logits.shape)} -> /tmp/resharded_logits.pt", flush=True)

try:
    golden = torch.load("/tmp/golden_logits.pt").float()
    x, y = logits.flatten(), golden.flatten()
    vx, vy = x - x.mean(), y - y.mean()
    pcc = float((vx @ vy) / (vx.norm() * vy.norm()))
    print(f"\n>>> PCC resharded-vs-golden = {pcc:.6f}   (was 0.013; target ~0.85)", flush=True)
except FileNotFoundError:
    print("golden not ready yet; saved resharded logits for later compare", flush=True)
