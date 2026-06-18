import torch
import ttnn
import utils
import main as G


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


device = utils.DeviceGetter.get_device((4, 8))

# Load the 80 serialized decode inputs (arg0..arg79) as multi-device host tensors.
# PJRT normally prepares each input on-device with a per-input layout. forward()
# tilizes the inputs it consumes via to_layout, but passes the KV caches straight
# to paged_update_cache (which requires TILE). Tilize only those cache inputs.
_CACHE_INPUTS = {9, 12, 26, 29, 43, 46, 60, 63}
inp = []
for i in range(80):
    t = ttnn.load_tensor(f"tensors/arg{i}.tensorbin")
    if i in _CACHE_INPUTS and t.layout != ttnn.Layout.TILE:
        t = ttnn.to_layout(t, ttnn.Layout.TILE)
    inp.append(t)
print("loaded", len(inp), "input tensors")

outs = G.forward(inp, device)
print("forward returned", len(outs), "tensors")

# ttnn_all_gather_17: logits gathered over both mesh axes -> replicated [64,1,151552]
logits_t = outs[15]
dts = ttnn.get_device_tensors(logits_t)
print("num device tensors:", len(dts))
full = ttnn.to_torch(dts[0]).float()
print("FULL device logits:", tuple(full.shape))

golden = torch.load("../golden_logits.pt").float()
print("GOLDEN logits:", tuple(golden.shape))

print("PCC_FULL:", pcc(full, golden))

dev_tok = full.argmax(dim=-1).flatten()
gold_tok = golden.argmax(dim=-1).flatten()
agree = (dev_tok == gold_tok).float().mean().item()
print("TOP1_AGREEMENT:", agree)

# --- diagnostics: saturation vs structural ---
torch.save(full, "/tmp/dev_logits_v3.pt")
print(
    "DEV stats: min",
    float(full.min()),
    "max",
    float(full.max()),
    "mean",
    float(full.mean()),
    "std",
    float(full.std()),
)
print(
    "GOLD stats: min",
    float(golden.min()),
    "max",
    float(golden.max()),
    "mean",
    float(golden.mean()),
    "std",
    float(golden.std()),
)
fmax = float(full.max())
fmin = float(full.min())
sat = ((full >= fmax - 1e-3) | (full <= fmin + 1e-3)).float().mean().item()
print("DEV saturation fraction:", sat)
sa = full.flatten().double().sort().values
sb = golden.flatten().double().sort().values
print("SORTED PCC:", pcc(sa, sb))
# per-row PCC over first 4 rows
for r in range(4):
    print(f"row {r} PCC:", pcc(full[r], golden[r]))
