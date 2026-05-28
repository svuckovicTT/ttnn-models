# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATA_FORMAT = torch.bfloat16
BATCH_SIZE = 32
INPUT_SEQUENCE_LENGTH = 128
DEFAULT_INPUT_PROMPT = (
    "Here is an exaustive list of the best practices for writing clean code:"
)


def load_pytorch_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DATA_FORMAT)

    if hasattr(model.config, "layer_types"):
        model.config.layer_types = ["full_attention"] * len(model.config.layer_types)
    if hasattr(model.config, "_experts_implementation"):
        model.config._experts_implementation = "dense"

    model.eval()
    return model


def load_input():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompts = [DEFAULT_INPUT_PROMPT] * BATCH_SIZE
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=INPUT_SEQUENCE_LENGTH,
        truncation=True,
    )
    input_ids = tokenized["input_ids"]

    model = load_pytorch_model()
    config = model.config
    head_dim = (
        config.head_dim
        if hasattr(config, "head_dim") and getattr(config, "head_dim")
        else config.hidden_size // config.num_attention_heads
    )
    num_key_value_heads = getattr(
        config, "num_key_value_heads", config.num_attention_heads
    )
    past_key_values = StaticCache(
        config=config,
        max_batch_size=BATCH_SIZE,
        max_cache_len=INPUT_SEQUENCE_LENGTH,
        device="cpu",
        dtype=DATA_FORMAT,
    )
    past_key_values.early_initialization(
        batch_size=BATCH_SIZE,
        num_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=DATA_FORMAT,
        device="cpu",
    )

    cache_position = torch.arange(0, input_ids.shape[1])

    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "cache_position": cache_position,
        "use_cache": True,
    }


def run_pytorch_model():
    model = load_pytorch_model()
    inputs = load_input()

    with torch.no_grad():
        output = model(**inputs)

    return output.logits
