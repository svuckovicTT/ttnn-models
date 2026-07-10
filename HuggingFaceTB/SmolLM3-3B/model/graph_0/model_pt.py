import os

import torch
from transformers import AutoTokenizer, SmolLM3Model

MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
DATA_FORMAT = torch.bfloat16
BATCH_SIZE = 1
CONTEXT_LENGTH = int(os.environ.get("FIBO_TE_CONTEXT_LENGTH", "4096"))

BRINGUP_PROMPT = (
    '{"subject":"a hyper-detailed, ultra-fluffy owl in moonlit trees",'
    '"style_medium":"photograph","camera":"85mm prime, shallow depth of field",'
    '"lighting":"cool moonlight with subtle silver highlights"}'
)


class FiboTextEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state


def load_pytorch_model():
    model = SmolLM3Model.from_pretrained(MODEL_ID, torch_dtype=DATA_FORMAT)
    wrapper = FiboTextEncoderWrapper(model)
    wrapper.eval()
    return wrapper


def load_input():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        [BRINGUP_PROMPT] * BATCH_SIZE,
        return_tensors="pt",
        padding="max_length",
        max_length=CONTEXT_LENGTH,
        truncation=True,
    )
    return encoded["input_ids"], encoded["attention_mask"]


def run_pytorch_model():
    wrapper = load_pytorch_model()
    input_ids, attention_mask = load_input()

    with torch.no_grad():
        output = wrapper(input_ids, attention_mask)

    return output
