# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache
from transformers.utils.quantization_config import Mxfp4Config

MODEL_ID = "openai/gpt-oss-20b"
DATA_FORMAT = torch.bfloat16
BATCH_SIZE = int(os.environ.get("BATCH", "1"))
INPUT_SEQUENCE_LENGTH = int(os.environ.get("ISL", "128"))
REDUCED_LAYERS = int(os.environ.get("REDUCED_LAYERS", "2"))
DEFAULT_INPUT_PROMPT = "Here is an exaustive list of the best practices for writing clean code:"


def _reduced_config():
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config.num_hidden_layers = REDUCED_LAYERS
    return config


def load_pytorch_model():
    config = _reduced_config()
    quantization_config = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, config=config, quantization_config=quantization_config,
        low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation="eager",
        torch_dtype=DATA_FORMAT,
    )
    if hasattr(model.config, "layer_types"):
        model.config.layer_types = ["full_attention"] * len(model.config.layer_types)
    model.eval()
    return model


def load_input():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenizer([DEFAULT_INPUT_PROMPT] * BATCH_SIZE, return_tensors="pt",
                          max_length=INPUT_SEQUENCE_LENGTH, truncation=True)
    input_ids = tokenized["input_ids"]
    config = _reduced_config()
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
    num_kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
    pkv = StaticCache(config=config, max_batch_size=BATCH_SIZE, max_cache_len=INPUT_SEQUENCE_LENGTH,
                      device="cpu", dtype=DATA_FORMAT)
    pkv.early_initialization(batch_size=BATCH_SIZE, num_heads=num_kv, head_dim=head_dim,
                             dtype=DATA_FORMAT, device="cpu")
    return {"input_ids": input_ids, "past_key_values": pkv,
            "cache_position": torch.arange(0, input_ids.shape[1]), "use_cache": True}


def run_pytorch_model():
    model = load_pytorch_model()
    with torch.no_grad():
        out = model(**load_input())
    o = out.logits[:, -1]
    print("PT_OUT_SHAPE", tuple(o.shape))
    return o


if __name__ == "__main__":
    run_pytorch_model()
