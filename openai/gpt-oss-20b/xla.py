# SPDX-License-Identifier: Apache-2.0
### Reduced gpt-oss-20b xla.py for TESTING the full-model-check pipeline.
### Based on the benchmark-to-xla output (openai/gpt-oss-20b, test_gpt_oss_20b_tp_qb2),
### shrunk to REDUCED_LAYERS layers + OPT_LEVEL so codegen is fast and dodges the opt-2
### compile stall. Purpose: get a SMALL full multichip model + whole-model golden to
### genuinely exercise "prettify -> optimize with full-model PCC at every stage".
### Real MXFP4->bf16 weights, full TP4 mesh — NOT synthetic. Knobs via env:
###   REDUCED_LAYERS (default 2), OPT_LEVEL (default 1), BATCH (default 1), ISL (default 128)
import os
from pathlib import Path

import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache
from transformers.utils.quantization_config import Mxfp4Config
from tt_torch import TT_DENSE_EXPERTS_BACKEND_NAME, codegen_py
from tt_torch.sharding import sharding_constraint_hook
from tt_torch.weight_dtype import apply_weight_dtype_overrides

MODEL_ID = "openai/gpt-oss-20b"
DATA_FORMAT = torch.bfloat16
BATCH_SIZE = int(os.environ.get("BATCH", "1"))
INPUT_SEQUENCE_LENGTH = int(os.environ.get("ISL", "128"))
REDUCED_LAYERS = int(os.environ.get("REDUCED_LAYERS", "2"))
OPT_LEVEL = int(os.environ.get("OPT_LEVEL", "1"))
DEFAULT_INPUT_PROMPT = "Here is an exaustive list of the best practices for writing clean code:"
OUTPUT_DIR = str(Path(__file__).resolve().parent / "model_reduced")

WDT = os.environ.get("WEIGHT_DTYPE", "bfp_bf8")   # optimize knob: bfp_bf8 (baseline) vs bfp_bf4 (aggressive)
COMPILE_OPTIONS = {
    "optimization_level": OPT_LEVEL,
    "experimental_weight_dtype": WDT,
    "experimental_enable_permute_matmul_fusion": False,
    "experimental-kv-cache-dtype": WDT,
}
WEIGHT_DTYPE_OVERRIDES = {"default": WDT}


def _reduced_config():
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config.num_hidden_layers = REDUCED_LAYERS
    return config


def load_pytorch_model():
    config = _reduced_config()
    quantization_config = Mxfp4Config(dequantize=True)
    use_tt_dense = os.environ.get("EXPERTS_IMPL", "tt_dense") == "tt_dense"
    kw = {"experts_implementation": TT_DENSE_EXPERTS_BACKEND_NAME} if use_tt_dense else {}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, config=config, quantization_config=quantization_config,
        low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation="eager",
        torch_dtype=DATA_FORMAT, **kw,
    )
    if hasattr(model.config, "layer_types"):
        model.config.layer_types = ["full_attention"] * len(model.config.layer_types)
    if use_tt_dense and hasattr(model.config, "_experts_implementation"):
        model.config._experts_implementation = TT_DENSE_EXPERTS_BACKEND_NAME
    model.eval()
    return model


class LastTokenLogitsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        return self.model(**kwargs).logits[:, -1]


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


def _inputs_to_device(inputs, device):
    out = dict(inputs)
    for layer in out["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
        layer.cumulative_length.zero_()
        layer.cumulative_length = layer.cumulative_length.to(device)
        layer.device = device
    out["input_ids"] = out["input_ids"].to(device)
    out["cache_position"] = out["cache_position"].to(device)
    return out


def _mesh_config_fn(n):
    return (1, n), ("batch", "model")


def _shard_spec_fn(model):
    s = {}
    for layer in model.model.layers:
        s[layer.self_attn.q_proj.weight] = ("model", None)
        s[layer.self_attn.k_proj.weight] = ("model", None)
        s[layer.self_attn.v_proj.weight] = ("model", None)
        s[layer.self_attn.o_proj.weight] = (None, "model")
        s[layer.self_attn.sinks] = (None,)
        s[layer.mlp.router.weight] = (None, None)
        if os.environ.get("SHARD_EXPERTS", "1") == "1":
            # Experts sharding triggers the tt-mlir Shardy sdy.all_slice axis-bound bug.
            # SHARD_EXPERTS=0 replicates experts (attention still TP-sharded) to let codegen pass.
            s[layer.mlp.experts.gate_up_proj] = ("model", None, None)
            s[layer.mlp.experts.gate_up_proj_bias] = ("model", None)
            s[layer.mlp.experts.down_proj] = ("model", None, None)
            s[layer.mlp.experts.down_proj_bias] = ("model", None)
    return s


def _apply_tp_sharding(model):
    n = xr.global_runtime_device_count()
    mesh_shape, mesh_name = _mesh_config_fn(n)
    mesh = Mesh(list(range(n)), mesh_shape, mesh_name)
    for tensor, spec in _shard_spec_fn(model).items():
        xs.mark_sharding(tensor, mesh, spec)
    if hasattr(model, "lm_head") and model.lm_head is not None:
        model.lm_head.register_forward_hook(sharding_constraint_hook(model.lm_head, mesh, (None, None, None)))
    return mesh


def _shard_kv_cache(pkv, mesh):
    for layer in pkv.layers:
        xs.mark_sharding(layer.keys, mesh, (None, "model", None, None))
        xs.mark_sharding(layer.values, mesh, (None, "model", None, None))


def run_pytorch_model():
    model = load_pytorch_model()
    with torch.no_grad():
        out = model(**load_input())
    o = out.logits[:, -1]
    print("PT_OUT_SHAPE", tuple(o.shape))
    return o


def run_tt_model():
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    device = torch_xla.device()
    model = load_pytorch_model().to(device, dtype=DATA_FORMAT)
    mesh = _apply_tp_sharding(model)
    torch_xla.set_custom_compile_options(COMPILE_OPTIONS)
    apply_weight_dtype_overrides(model, WEIGHT_DTYPE_OVERRIDES)
    compiled = torch.compile(LastTokenLogitsWrapper(model), backend="tt")
    inputs = _inputs_to_device(load_input(), device)
    _shard_kv_cache(inputs["past_key_values"], mesh)
    with torch.no_grad():
        out = compiled(**inputs)
    return out.cpu()


def codegen_model():
    os.environ["XLA_HLO_DEBUG"] = "1"
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    device = torch_xla.device()
    model = load_pytorch_model().to(device, dtype=DATA_FORMAT)
    mesh = _apply_tp_sharding(model)
    apply_weight_dtype_overrides(model, WEIGHT_DTYPE_OVERRIDES)
    inputs = _inputs_to_device(load_input(), device)
    _shard_kv_cache(inputs["past_key_values"], mesh)
    codegen_py(model, export_path=OUTPUT_DIR, export_tensors=True,
               compiler_options=COMPILE_OPTIONS, **inputs)
    print("CODEGEN_DONE", OUTPUT_DIR)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--run-pt", action="store_true")
    m.add_argument("--run-tt", action="store_true")
    m.add_argument("--codegen", action="store_true")
    args = p.parse_args()
    print(f"[reduced xla] layers={REDUCED_LAYERS} opt={OPT_LEVEL} batch={BATCH_SIZE} isl={INPUT_SEQUENCE_LENGTH}")
    if args.run_pt:
        run_pytorch_model()
    if args.run_tt:
        run_tt_model()
    if args.codegen:
        codegen_model()
