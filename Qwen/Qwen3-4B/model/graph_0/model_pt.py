# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_input():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    messages = [{"role": "user", "content": "Who are you?"}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    x = tokenizer(text, return_tensors="pt")["input_ids"]

    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")

    return x


def load_pytorch_model():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model


def run_pytorch_model():
    model = load_pytorch_model()
    x = load_input()

    with torch.no_grad():
        output = model(x)

    return output.logits
