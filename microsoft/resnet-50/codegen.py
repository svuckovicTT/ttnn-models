# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates codegen for ResNet-50 from HuggingFace

import shutil
from pathlib import Path

import torch
import torch_xla.runtime as xr
from transformers import ResNetForImageClassification, AutoImageProcessor
from tt_torch import codegen_py
from datasets import load_dataset

OUTPUT_DIR = str(Path(__file__).resolve().parent / "model")


def main():
    # Set up XLA runtime for TT backend
    xr.set_device_type("TT")

    # Load ResNet-50 from HuggingFace
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50", torch_dtype=torch.bfloat16
    )
    model.eval()

    # Get input
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    x = processor(image, return_tensors="pt")["pixel_values"].to(torch.bfloat16)

    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")

    codegen_py(
        model,
        x,
        export_path=OUTPUT_DIR,
        export_tensors=True,
        compiler_options={
            "codegen_split_files": True,
            "optimization_level": 2,
        },
    )


def test_resnet_codegen():
    """Test that codegen for ResNet-50 creates the expected output folder."""
    output_dir = Path(OUTPUT_DIR)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    try:
        main()
        assert output_dir.exists(), (
            f"Expected output folder '{output_dir}' was not created"
        )
        assert output_dir.is_dir(), f"'{output_dir}' exists but is not a directory"
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    main()
