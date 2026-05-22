# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates codegen for Qwen/Qwen3-4B from HuggingFace

import os
from pathlib import Path

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import AutoModelForCausalLM, AutoTokenizer
from tt_torch import codegen_py

OUTPUT_DIR = str(Path(__file__).resolve().parent / "model")
COMPILE_OPTIONS = {
    "optimization_level": 2,
    "codegen_split_files": True,
}


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
    # Capture exact PCC from first --golden run and paste here.
    exact_pcc = None

    pt_output = run_pytorch_model()
    tt_output = run_tt_model()

    assert pt_output.shape == tt_output.shape, (
        f"shape mismatch: {pt_output.shape} vs {tt_output.shape}"
    )
    assert pt_output.dtype == tt_output.dtype, (
        f"dtype mismatch: {pt_output.dtype} vs {tt_output.dtype}"
    )
    x, y = pt_output.flatten().float(), tt_output.flatten().float()
    vx, vy = x - x.mean(), y - y.mean()
    pcc = ((vx @ vy) / (vx.norm() * vy.norm())).item()
    print(f"PCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match captured {exact_pcc}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen/Qwen3-4B codegen pipeline")

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
