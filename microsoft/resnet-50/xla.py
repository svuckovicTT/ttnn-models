# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates codegen for ResNet-50 from HuggingFace

import os
from pathlib import Path

import torch
import torch_xla
import torch_xla.runtime as xr
from datasets import load_dataset
from transformers import AutoImageProcessor, ResNetForImageClassification
from tt_torch import codegen_py

OUTPUT_DIR = str(Path(__file__).resolve().parent / "model")
COMPILE_OPTIONS = {
    "optimization_level": 2,
    "codegen_split_files": True,
}


def load_input():
    dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
    images = [sample["image"] for sample in dataset.take(8)]
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    x = processor(images, return_tensors="pt")["pixel_values"].to(torch.bfloat16)

    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")

    return x


def load_pytorch_model():
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model


def run_pytorch_model():
    model = load_pytorch_model()
    x = load_input()

    with torch.no_grad():
        output = model(x)

    return output.logits


def run_tt_model():
    device = torch_xla.device()

    torch_xla.set_custom_compile_options(COMPILE_OPTIONS)

    model = load_pytorch_model()
    model.compile(backend="tt", options={"tt_legacy_compile": True})
    model = model.to(device)
    x = load_input().to(device)

    with torch.no_grad():
        output = model(x)

    return output.logits.cpu()


def codegen_model():
    os.environ["XLA_HLO_DEBUG"] = "1"

    model = load_pytorch_model()
    x = load_input()

    codegen_py(
        model,
        x,
        export_path=OUTPUT_DIR,
        export_tensors=True,
        compiler_options=COMPILE_OPTIONS,
    )


def compare_pytorch_and_tt_runs():
    # Exact PCC is calculated during first run and manually set here
    exact_pcc = 0.97265625

    pt_output = run_pytorch_model()
    tt_output = run_tt_model()

    assert pt_output.shape == tt_output.shape, (
        f"shape mismatch: {pt_output.shape} vs {tt_output.shape}"
    )
    assert pt_output.dtype == tt_output.dtype, (
        f"dtype mismatch: {pt_output.dtype} vs {tt_output.dtype}"
    )
    pcc = calculate_pcc(pt_output, tt_output)
    print(f"PCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} is below threshold of {exact_pcc}"


def calculate_pcc(x, y):
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ResNet-50 codegen pipeline")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-pt", action="store_true", help="Run PyTorch model on CPU")
    mode.add_argument("--run-tt", action="store_true", help="Run model on TT hardware")
    mode.add_argument("--codegen", action="store_true", help="Generate TTNN code")
    mode.add_argument(
        "--golden", action="store_true", help="Compare PyTorch and TTNN runs"
    )

    args = parser.parse_args()

    if args.run_pt:
        run_pytorch_model()
    if args.run_tt:
        run_tt_model()
    if args.codegen:
        codegen_model()
    if args.golden:
        compare_pytorch_and_tt_runs()
