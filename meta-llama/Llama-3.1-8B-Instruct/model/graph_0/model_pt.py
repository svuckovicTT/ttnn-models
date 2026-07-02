# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATA_FORMAT = torch.bfloat16
BATCH_SIZE = 32
INPUT_SEQUENCE_LENGTH = 128
DEFAULT_INPUT_PROMPT = (
    "Here is an exaustive list of the best practices for writing clean code:"
)


class LastTokenLogitsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        output = self.model(**kwargs)
        return output.logits[:, -1]


def load_pytorch_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DATA_FORMAT)

    if hasattr(model.config, "layer_types"):
        model.config.layer_types = ["full_attention"] * len(model.config.layer_types)
    if hasattr(model.config, "_experts_implementation"):
        model.config._experts_implementation = "dense"

    model.eval()
    return model


def load_input(model):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    prompts = [DEFAULT_INPUT_PROMPT] * BATCH_SIZE
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=INPUT_SEQUENCE_LENGTH,
        truncation=True,
    )
    input_ids = tokenized["input_ids"]

    config = model.config
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
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
    prefill_inputs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "cache_position": cache_position,
        "position_ids": cache_position.unsqueeze(0),
        "use_cache": True,
    }

    with torch.no_grad():
        prefill_logits = model(**prefill_inputs).logits[:, -1]
    next_token = prefill_logits.argmax(dim=-1, keepdim=True)
    cache_position = prefill_inputs["cache_position"][-1:] + 1

    return {
        "input_ids": next_token,
        "past_key_values": prefill_inputs["past_key_values"],
        "cache_position": cache_position,
        "position_ids": cache_position.unsqueeze(0),
        "use_cache": True,
    }


def run_pytorch_model():
    model = load_pytorch_model()
    decode_inputs = load_input(model)

    with torch.no_grad():
        decode_logits = model(**decode_inputs).logits[:, -1]

    print(f"[cpu] decode logits {tuple(decode_logits.shape)}")
    return decode_logits


if __name__ == "__main__":
    run_pytorch_model()
