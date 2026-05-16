# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Run ResNet-50 on TT device via torch_xla and compare against CPU golden."""

import io
import os

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from datasets import load_dataset
from transformers import AutoImageProcessor, ResNetForImageClassification
from ttxla_tools import parse_executable


def load_input():
    dataset = load_dataset("huggingface/cats-image", split="test")
    image = dataset[0]["image"]
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    return processor(image, return_tensors="pt")["pixel_values"]


def run_resnet_cpu(pixel_values):
    print("Running ResNet-50 on CPU")
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50", torch_dtype=torch.bfloat16
    )
    model.eval()
    with torch.no_grad():
        return model(pixel_values=pixel_values.to(torch.bfloat16)).logits


def run_resnet_tt(pixel_values, cache_dir):
    print("Running ResNet-50 on TT")
    xr.initialize_cache(cache_dir)
    torch_xla.set_custom_compile_options({"optimization_level": 2})

    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50", torch_dtype=torch.bfloat16
    )
    model.eval()

    device = xm.xla_device()
    model = model.to(device)
    model.compile(backend="tt")

    pixel_values = pixel_values.to(torch.bfloat16).to(device)

    with torch.no_grad():
        output = model(pixel_values=pixel_values).logits

    output.to("cpu")

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(cache_dir) if os.path.isfile(os.path.join(cache_dir, f))]
    for i, fname in enumerate(files):
        with open(os.path.join(cache_dir, fname), "rb") as f:
            ttir_mlir, ttnn_mlir, flatbuffer_binary = parse_executable(io.BytesIO(f.read()))
        prefix = f"{output_dir}/resnet50_graph_{i}"
        with open(f"{prefix}_ttir.mlir", "w") as f:
            f.write(ttir_mlir)
        with open(f"{prefix}_ttnn.mlir", "w") as f:
            f.write(ttnn_mlir)
        with open(f"{prefix}.ttnn", "wb") as f:
            f.write(flatbuffer_binary)
    print(f"Compiled artifacts saved to {output_dir}/ ({len(files)} graphs)")

    return output


def calculate_pcc(x, y):
    x_flat, y_flat = x.flatten().float(), y.flatten().float()
    return torch.corrcoef(torch.stack([x_flat, y_flat]))[0, 1].item()


def calculate_pcc_original(x, y):
    # This function calculates the PCC between two torch tensors

    # Assert both are torch tensors
    assert isinstance(x, torch.Tensor), "x must be a torch tensor"
    assert isinstance(y, torch.Tensor), "y must be a torch tensor"

    if x.shape != y.shape:
        raise ValueError(
            f"Shapes of x and y must be the same, but got {x.shape} and {y.shape}"
        )

    # Calculate PCC
    x_flat, y_flat = x.flatten(), y.flatten()
    vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()

    return float("nan") if denom == 0 else ((vx @ vy) / denom).item()



def test_resnet50():
    xr.set_device_type("TT")

    cache_dir = os.path.join(os.path.dirname(__file__), "cachedir")

    pixel_values = load_input()
    cpu_output = run_resnet_cpu(pixel_values)
    tt_output = run_resnet_tt(pixel_values, cache_dir).cpu().float()
    cpu_output = cpu_output.float()

    pcc = calculate_pcc(tt_output, cpu_output)
    print(f"PCC: {pcc:.6f}")
    # assert pcc >= 0.99, f"PCC {pcc} is below threshold 0.99"

    pcc_original = calculate_pcc_original(tt_output, cpu_output)
    print(f"PCC original: {pcc_original:.6f}")
    assert pcc_original >= 0.99, f"PCC original {pcc_original} is below threshold 0.99"


if __name__ == "__main__":
    test_resnet50()
