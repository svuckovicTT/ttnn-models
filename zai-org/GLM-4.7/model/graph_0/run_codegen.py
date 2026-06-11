"""Run the codegen graph (main._main) with the SAME saved tensors and PCC its
logits against the flatbuffer logits (/tmp/fb_logits.pt from run_fb.py).

Both paths consume identical ./tensors/argN.tensorbin, so this isolates
EmitPy(Python) vs runtime(flatbuffer):
  PCC high (~1.0) -> EmitPy matches runtime; both differ from XLA-live -> saved-tensor problem.
  PCC low         -> EmitPy execution bug -> bisect.
"""
import torch
import ttnn

import main  # codegen module (model_pt import is lazy; won't touch torchvision)


def pcc(x, y):
    x, y = x.flatten().float(), y.flatten().float()
    vx, vy = x - x.mean(), y - y.mean()
    denom = vx.norm() * vy.norm()
    return float("nan") if denom == 0 else float((vx @ vy) / denom)


acts = main.load_activations_for__main()
weights = main.load_weights_for__main()
outs = main._main(acts, weights)

host = ttnn.from_device(outs[15])
cg = ttnn.to_torch(ttnn.get_device_tensors(host)[0]).float().cpu()
torch.save(cg, "/tmp/codegen_logits.pt")
print(f"codegen logits: {tuple(cg.shape)}  -> saved /tmp/codegen_logits.pt", flush=True)

fb = torch.load("/tmp/fb_logits.pt").float()
print(f"fb logits:      {tuple(fb.shape)}")
if fb.shape == cg.shape:
    print(f"\n>>> PCC flatbuffer-vs-codegen = {pcc(fb, cg):.6f}")
    print(">>> high(~1.0) => saved-tensor problem ; low => EmitPy execution bug")
    # also report max abs diff + a couple sample rows
    d = (fb - cg).abs()
    print(f"    max_abs_diff={d.max().item():.4g}  mean_abs_diff={d.mean().item():.4g}")
