# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import ResNetForImageClassification, AutoImageProcessor
from datasets import load_dataset


def load_input():
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    x = processor(image, return_tensors="pt")["pixel_values"].to(torch.bfloat16)
    return x


def load_model():
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model


def main():
    model = load_model()
    x = load_input()

    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")

    with torch.no_grad():
        output = model(x)

    return output


if __name__ == "__main__":
    main()
