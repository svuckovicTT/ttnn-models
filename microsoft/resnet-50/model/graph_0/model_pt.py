# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU PyTorch reference implementation of ResNet-50 from HuggingFace."""

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, ResNetForImageClassification


def load_input():
    dataset = load_dataset("huggingface/cats-image", split="test")
    image = dataset[0]["image"]
    image = image.resize((224, 224))
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    return processor(image, return_tensors="pt")["pixel_values"].to(torch.bfloat16)


def load_model():
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model


def run():
    model = load_model()
    pixel_values = load_input()
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
    return logits


if __name__ == "__main__":
    logits = run()
    print(f"Logits shape: {logits.shape}")
    print(f"Predicted class: {logits.argmax(-1).item()}")
