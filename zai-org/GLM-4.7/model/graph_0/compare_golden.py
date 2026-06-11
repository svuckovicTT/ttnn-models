"""Compute the CPU golden in a CLEAN process (no ttnn/ttrt, which corrupt the
torch dispatcher for torchvision) and PCC it against the flatbuffer logits saved
by run_fb.py (/tmp/fb_logits.pt)."""
import torch

import model_pt


def pcc(x, y):
    x, y = x.flatten().float(), y.flatten().float()
    vx, vy = x - x.mean(), y - y.mean()
    denom = vx.norm() * vy.norm()
    return float("nan") if denom == 0 else float((vx @ vy) / denom)


fb = torch.load("/tmp/fb_logits.pt").float()
golden = model_pt.run_pytorch_model().float().cpu()
torch.save(golden, "/tmp/golden_logits.pt")
print(f"fb={tuple(fb.shape)} golden={tuple(golden.shape)}")
if fb.shape == golden.shape:
    print(f">>> PCC flatbuffer-vs-golden = {pcc(fb, golden):.6f}")
    print(">>> codegen-vs-golden = 0.013 ; runtime(XLA)-vs-golden = 0.85")
