import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, ResNetForImageClassification


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
