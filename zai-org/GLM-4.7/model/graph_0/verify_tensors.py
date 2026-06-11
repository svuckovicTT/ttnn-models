"""Verify whether the saved tensorbins are single device shards by comparing them
to the REAL HF weights (read straight from safetensors -> no transformers modeling
import -> no torchvision crash).

WEIGHT: arg2 = model.model.layers.0.self_attn.k_proj.weight
        MLIR global [1024,5120], local [128,5120], sharded ("model", None) x8.
INPUT:  arg4 = input_ids (args_1), MLIR global [64,1], local [16,1], batch-sharded.
"""
import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import ttnn

MODEL = "zai-org/GLM-4.7"


def load_bin(name):
    t = ttnn.load_tensor(f"tensors/{name}.tensorbin")
    if t.layout != ttnn.Layout.ROW_MAJOR:
        t = ttnn.to_layout(t, ttnn.Layout.ROW_MAJOR)
    return ttnn.to_torch(t)


# ---------- WEIGHT ----------
idx = hf_hub_download(MODEL, "model.safetensors.index.json")
wmap = json.load(open(idx))["weight_map"]
key = "model.layers.0.self_attn.k_proj.weight"
kproj = load_file(hf_hub_download(MODEL, wmap[key]))[key].to(torch.bfloat16)

arg2 = load_bin("arg2").to(torch.bfloat16)
print(f"HF {key}: {tuple(kproj.shape)}   saved arg2: {tuple(arg2.shape)}")

n_model = kproj.shape[0] // arg2.shape[0]  # 1024/128 = 8 model shards
shards = kproj.reshape(n_model, arg2.shape[0], arg2.shape[1])
print(f"\nDoes saved arg2 equal each of the {n_model} model-axis shards of the real weight?")
for i in range(n_model):
    md = (arg2.float() - shards[i].float()).abs().max().item()
    print(f"  arg2 == HF shard[{i}] (rows {i*128}:{(i+1)*128}) : "
          f"equal={torch.equal(arg2, shards[i])}  max_abs_diff={md:.4g}")

# are the 8 real shards actually distinct (i.e., does replicating shard 0 lose info)?
distinct = all(
    not torch.equal(shards[0], shards[i]) for i in range(1, n_model)
)
print(f"\nThe {n_model} real model-shards are pairwise-distinct vs shard0: {distinct}")
print("=> if arg2==shard0 AND shards distinct: only 1/{0} of k_proj was saved; "
      "replicating it puts shard0 on all devices.".format(n_model))

# ---------- INPUT ----------
arg4 = load_bin("arg4")
uniq = torch.unique(arg4)
print(f"\nINPUT arg4 (input_ids): shape={tuple(arg4.shape)} dtype={arg4.dtype} "
      f"n_unique_values={uniq.numel()} values={uniq.tolist()[:4]}")
print("  (global is [64,1]; saved is one batch-shard. batch is uniform -> "
      "replicating this shard reproduces the correct values, so inputs are harmless.)")
