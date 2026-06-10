# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Standalone xla.py reproducing tt-xla benchmark
### tests/benchmark/test_llms.py::test_glm_4_7_tp_galaxy_4_layers.
### Resolved model: zai-org/GLM-4.7 (HF) — GLM-4 MoE, first 4 decoder layers,
### built on the meta device and populated from HF safetensors shards.
### No imports from tt-xla/third_party/tt_forge_models or tt-xla/tests/benchmark/*.
###
### Hardware: Galaxy (galaxy-wh-6u -> wh-glx), 32 devices, mesh (4, 8). NOT retargeted.
### This script REQUIRES a 32-device Galaxy system. enable_sparse_mlp() and the
### TP shard specs assume num_devices == 32, so even the CPU golden path
### (--run-pt) constructs the model with a (4, 8) mesh and will fail to build on
### a non-Galaxy machine.
###
### Scope: reproduces the DECODE graph (single new token, seq_len=1), which is the
### graph the benchmark actually measures (it keeps only the decode perf metrics
### and checks first-decode PCC). The prefill is run on CPU only to populate the
### KV cache (advancing cumulative_length to prompt_len); the compiled/codegenned
### graph is the decode step alone, not prefill.
###
### Shard specs are the hidden-replicated _glm_4_7_shard_spec_fn passed explicitly
### by the test (TP-8 / DP-4 / EP-32): residual hidden kept replicated along the
### model axis so RMS norms reduce locally; embedding replicated; lm_head
### vocab-parallel; attention / dense MLP / shared experts col->row parallel;
### routed experts sharded across both model and batch axes.

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safetensors_load_file
from torch_xla.distributed.spmd import Mesh
from transformers import AutoTokenizer
from transformers.cache_utils import StaticCache

# tt_torch is a framework package — fair game to import.
from tt_torch.sharding import sharding_constraint_hook, sharding_constraint_tensor
from tt_torch.sparse_mlp import A2aSparseMLPWithSharedExperts, enable_sparse_mlp

MODEL_ID = "zai-org/GLM-4.7"
DATA_FORMAT = torch.bfloat16     # test_llms.py: DEFAULT_DATA_FORMAT = "bfloat16"
BATCH_SIZE = 64                  # test override (batch 128 hangs — tt-xla#4565)
INPUT_SEQUENCE_LENGTH = 128      # test_llms.py: DEFAULT_INPUT_SEQUENCE_LENGTH = 128
NUM_LAYERS = 4                   # test override (num_layers=4)
DEFAULT_INPUT_PROMPT = (
    "Here is an exaustive list of the best practices for writing clean code:"
)

# Shard specs from the test's explicit kwargs.
INPUT_OUTPUT_SHARDING_SPEC = ("batch", None)
KV_CACHE_SHARDING_SPEC = ("batch", None, None, None)

OUTPUT_DIR = str(Path(__file__).resolve().parent / "model")

# Compile options.
#   Per-test overrides: optimization_level=1, trace_enabled=False.
#   Benchmark-file DEFAULT_* (test_llms.py): experimental_weight_dtype="bfp_bf8",
#       experimental_enable_permute_matmul_fusion=False,
#       experimental-kv-cache-dtype="bfp_bf8".
#   Runtime-instrumentation keys (export_path/export_model_name/ttnn_perf_metrics_*)
#       dropped — xla.py produces no benchmark JSON.
#   No weight_dtype_overrides: test passes none and no mixed_precision_configs JSON
#       exists for GLM, so apply_weight_dtype_overrides is not called.
COMPILE_OPTIONS = {
    "optimization_level": 1,
    "experimental_weight_dtype": "bfp_bf8",
    "experimental_enable_permute_matmul_fusion": False,
    "experimental-kv-cache-dtype": "bfp_bf8",
    # "enable_trace": False,  # benchmark used trace_enabled=False
}


# ---------------------------------------------------------------------------
# Mesh + shard specs.
#   get_mesh_config: inlined from the GLM loader (model_loader param dropped).
#   glm_4_7_shard_spec: inlined from _glm_4_7_shard_spec_fn in test_llms.py
#     (the explicit shard_spec_fn= kwarg; hidden-replicated TP-8/DP-4/EP-32).
# ---------------------------------------------------------------------------
def get_mesh_config(num_devices: int):
    if num_devices == 32:
        mesh_shape = (4, 8)
    elif num_devices == 8:
        mesh_shape = (2, 4)
    else:
        raise ValueError(f"Unsupported number of devices: {num_devices}")
    return mesh_shape, ("batch", "model")


def glm_4_7_shard_spec(model):
    """Hidden-replicated sharding spec for GLM-4 on the 4x8 galaxy mesh.
    TP-8 : DP-4 : EP-32. Residual hidden kept replicated along the model axis so
    the RMS norms reduce locally instead of lowering to a distributed all_gather
    norm. Embedding replicated, lm_head vocab-parallel, attention / dense MLP /
    shared experts col->row parallel along model axis. Routed expert weights
    sharded across both model and batch axes (EP-32), matching DeepSeek V3.x."""
    shard_specs = {}

    shard_specs[model.model.embed_tokens.weight] = (None, None)
    shard_specs[model.model.norm.weight] = (None,)
    shard_specs[model.lm_head.weight] = ("model", None)

    for layer in model.model.layers:
        shard_specs[layer.input_layernorm.weight] = (None,)
        shard_specs[layer.post_attention_layernorm.weight] = (None,)

        attn = layer.self_attn
        shard_specs[attn.q_proj.weight] = ("model", None)
        shard_specs[attn.k_proj.weight] = ("model", None)
        shard_specs[attn.v_proj.weight] = ("model", None)
        shard_specs[attn.o_proj.weight] = (None, "model")

        if attn.q_proj.bias is not None:
            shard_specs[attn.q_proj.bias] = ("model",)
            shard_specs[attn.k_proj.bias] = ("model",)
            shard_specs[attn.v_proj.bias] = ("model",)

        if hasattr(attn, "q_norm"):
            shard_specs[attn.q_norm.weight] = (None,)
            shard_specs[attn.k_norm.weight] = (None,)

        mlp = layer.mlp

        if isinstance(mlp, A2aSparseMLPWithSharedExperts):
            inner = mlp.mlp  # A2aSparseMLP
            shard_specs[inner.router.gate.weight] = (None, None)
            shard_specs[inner.experts.gate_proj] = (("model", "batch"), None, None)
            shard_specs[inner.experts.up_proj] = (("model", "batch"), None, None)
            shard_specs[inner.experts.down_proj] = (("model", "batch"), None, None)

            shared = getattr(mlp, "shared_experts", None)
            if shared is not None:
                shard_specs[shared.gate_proj.weight] = ("model", None)
                shard_specs[shared.up_proj.weight] = ("model", None)
                shard_specs[shared.down_proj.weight] = (None, "model")

        else:
            shard_specs[mlp.gate_proj.weight] = ("model", None)
            shard_specs[mlp.up_proj.weight] = ("model", None)
            shard_specs[mlp.down_proj.weight] = (None, "model")

    return shard_specs


# ---------------------------------------------------------------------------
# Meta-device GLM-4 MoE loading — inlined from
# tt-xla/third_party/tt_forge_models/glm/causal_lm/pytorch/meta_loading.py
# ---------------------------------------------------------------------------
_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _resolve_hf_shards_for_layers(repo_id, n_layers, *, revision=None):
    """Local paths of safetensors shards holding the first n_layers decoder layers."""
    print(f"[meta_loading] Resolving shards for {repo_id} (first {n_layers} layer(s))")
    try:
        index_path = hf_hub_download(
            repo_id, "model.safetensors.index.json", revision=revision
        )
    except Exception:
        print("[meta_loading] No index.json; downloading single model.safetensors")
        return [hf_hub_download(repo_id, "model.safetensors", revision=revision)]

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    needed_shards = set()
    for ckpt_key, shard_name in weight_map.items():
        m = _LAYER_INDEX_RE.search(ckpt_key)
        if m and int(m.group(1)) >= n_layers:
            continue
        if ckpt_key.startswith("mtp.") or ".mtp." in ckpt_key:
            continue
        needed_shards.add(shard_name)

    sorted_shards = sorted(needed_shards)
    total = len(sorted_shards)
    print(f"[meta_loading] Need {total} shard(s) for first {n_layers} layer(s)")
    paths = []
    for i, s in enumerate(sorted_shards, start=1):
        print(f"[meta_loading] Downloading shard {i}/{total}: {s}")
        paths.append(hf_hub_download(repo_id, s, revision=revision))
    return paths


def _build_glm4_config(pretrained_model_name, num_layers):
    from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig

    config = Glm4MoeConfig.from_pretrained(pretrained_model_name)
    config.num_hidden_layers = num_layers
    config._attn_implementation = "eager"
    if hasattr(config, "num_nextn_predict_layers"):
        config.num_nextn_predict_layers = 0
    return config


def _load_glm4_state_dict(pretrained_model_name, n_layers, n_experts):
    """Load first n_layers from HF shards, fusing per-expert weights into 3-D tensors."""
    expert_re = re.compile(
        r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)"
        r"\.(gate_proj|up_proj|down_proj)\.weight$"
    )
    layer_re = re.compile(r"(?:^|\.)layers\.(\d+)\.")

    expert_tensors = {}
    state_dict = {}
    for shard_path in _resolve_hf_shards_for_layers(pretrained_model_name, n_layers):
        for k, t in safetensors_load_file(shard_path, device="cpu").items():
            m = layer_re.search(k)
            if m and int(m.group(1)) >= n_layers:
                continue
            t = t.to(torch.bfloat16)
            m = expert_re.match(k)
            if m:
                expert_tensors[(int(m.group(1)), int(m.group(2)), m.group(3))] = t
            else:
                state_dict[k] = t

    for layer_idx in sorted({k[0] for k in expert_tensors}):
        gate = torch.stack(
            [expert_tensors[(layer_idx, j, "gate_proj")] for j in range(n_experts)]
        )
        up = torch.stack(
            [expert_tensors[(layer_idx, j, "up_proj")] for j in range(n_experts)]
        )
        down = torch.stack(
            [expert_tensors[(layer_idx, j, "down_proj")] for j in range(n_experts)]
        )
        state_dict[f"model.layers.{layer_idx}.mlp.experts.gate_up_proj"] = torch.cat(
            [gate, up], dim=1
        ).contiguous()
        state_dict[
            f"model.layers.{layer_idx}.mlp.experts.down_proj"
        ] = down.contiguous()

    return state_dict


def _restore_glm4_remaining_meta_tensors(model, config):
    """Post-load fixups required after meta-device construction."""
    from transformers.models.glm4_moe.modeling_glm4_moe import (
        Glm4MoeRotaryEmbedding,
        Glm4MoeTopkRouter,
    )

    # inv_freq is not persisted in checkpoints; re-initialize on CPU.
    model.model.rotary_emb = Glm4MoeRotaryEmbedding(config, device="cpu")

    # bf16 rounding of e_score_correction_bias flips top-k expert selections.
    for module in model.modules():
        if (
            isinstance(module, Glm4MoeTopkRouter)
            and module.e_score_correction_bias.dtype != torch.float32
        ):
            module.e_score_correction_bias = module.e_score_correction_bias.to(
                torch.float32
            )


def load_model_from_checkpoint(pretrained_model_name, num_layers):
    """Build a GLM-4 MoE model on meta device and populate the first num_layers."""
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeForCausalLM

    config = _build_glm4_config(pretrained_model_name, num_layers)

    with torch.device("meta"):
        model = Glm4MoeForCausalLM(config)

    state_dict = _load_glm4_state_dict(
        pretrained_model_name, num_layers, config.n_routed_experts
    )
    model.load_state_dict(state_dict, strict=False, assign=True)

    _restore_glm4_remaining_meta_tensors(model, config)

    return model


# ---------------------------------------------------------------------------
# Model + inputs
# ---------------------------------------------------------------------------
def load_pytorch_model():
    """Mirrors GLM loader.load_model (GLM4 + num_layers branch) +
    setup_model_and_tokenizer patches from benchmarks/llm_benchmark.py."""
    # GLM-4 with num_layers -> meta-device load of the first NUM_LAYERS layers.
    model = load_model_from_checkpoint(MODEL_ID, NUM_LAYERS)
    model.eval()

    # Enable sparse MoE for GLM4 (loader.load_model). Needs the device mesh shape.
    num_devices = xr.global_runtime_device_count()
    mesh_shape, _ = get_mesh_config(num_devices)
    enable_sparse_mlp(model, mesh=mesh_shape, cluster_axis=0, config=model.config)

    # setup_model_and_tokenizer patches (benchmarks/llm_benchmark.py).
    if hasattr(model.config, "layer_types"):
        model.config.layer_types = ["full_attention"] * len(model.config.layer_types)
    if hasattr(model.config, "_experts_implementation"):
        # experts_implementation kwarg is None for this test -> DEFAULT "batched_mm".
        model.config._experts_implementation = "batched_mm"

    model.eval()
    return model


def default_read_logits_fn(output):
    # test_llms.py: default_read_logits_fn
    return output.logits


class LLMSamplingWrapper(torch.nn.Module):
    """VERBATIM copy of LLMSamplingWrapper from
    tt-xla/tests/benchmark/llm_utils/decode_utils.py — so the traced decode graph
    is byte-for-byte the benchmark's graph (same argmax, same next_token_ids /
    next_cache_position ops, same two sharding constraints, position_ids computed
    inside forward from cache_position, full replicated logits returned).

    Keeping token selection + cache-position increment inside the compiled graph
    is exactly what the benchmark does; reproducing it 1:1 is required to get the
    same IR (and the same on-device output layout that the host transfer needs)."""

    def __init__(
        self,
        model,
        read_logits_fn,
        return_logits: bool = True,
        mesh=None,
        output_sharding_spec=None,
    ):
        super().__init__()
        self.model = model
        self.read_logits_fn = read_logits_fn
        self.return_logits = return_logits
        self.mesh = mesh
        self.output_sharding_spec = output_sharding_spec

    def forward(self, input_ids, past_key_values, cache_position, use_cache=True):
        position_ids = cache_position.unsqueeze(0)
        output = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
        )
        logits = self.read_logits_fn(output)
        # Only take logits for last token in prefill.
        # This is a noop for decode.
        next_token_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
        next_token_ids_replicated = next_token_ids
        if self.mesh and self.output_sharding_spec:
            replicate_spec = tuple(None for _ in self.output_sharding_spec)
            next_token_ids = sharding_constraint_tensor(
                next_token_ids, self.mesh, self.output_sharding_spec
            )
            next_token_ids_replicated = sharding_constraint_tensor(
                next_token_ids, self.mesh, replicate_spec
            )
        next_cache_position = cache_position[-1:] + 1
        if self.return_logits:
            logits_out = logits
            if self.mesh and self.output_sharding_spec:
                replicate_spec = tuple(None for _ in range(logits_out.dim()))
                logits_out = sharding_constraint_tensor(
                    logits, self.mesh, replicate_spec
                )
            return (
                next_token_ids,
                next_token_ids_replicated,
                next_cache_position,
                logits_out,
            )
        return next_token_ids, next_token_ids_replicated, next_cache_position


def construct_inputs(config):
    """Mirrors construct_inputs() + init_static_cache() from benchmarks/
    llm_benchmark.py (StaticCache path). Returns EXACTLY the four keys the
    benchmark passes to the model — input_ids, past_key_values, cache_position,
    use_cache — with NO position_ids (LLMSamplingWrapper computes position_ids
    internally from cache_position)."""
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

    # init_static_cache(): StaticCache on CPU + early_initialization() to
    # pre-allocate backing tensors (moved to device explicitly later; tt-xla#1645).
    if hasattr(config, "head_dim") and getattr(config, "head_dim"):
        head_dim = config.head_dim
    else:
        head_dim = config.hidden_size // config.num_attention_heads
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


def cpu_prefill_to_decode_state(model):
    """Reproduce the benchmark's decode_only CPU prefill (benchmark iter 0):
    run ONE prefill step through LLMSamplingWrapper to populate the StaticCache
    (cumulative_length advanced to prompt_len) and produce the first decode token.

    Returns the post-prefill decode-state input dict — input_ids = next_token_0,
    cache_position = [prompt_len], past_key_values = the populated cache — exactly
    the state the benchmark snapshots as (first_decode_input_ids,
    decode_only_cache_position, decode_only_cache)."""
    input_args = construct_inputs(model.config)

    cpu_wrapper = LLMSamplingWrapper(
        model, default_read_logits_fn, return_logits=True
    )
    cpu_wrapper.eval()
    with torch.no_grad():
        next_token_ids, _, next_cache_position, _ = cpu_wrapper(**input_args)

    # generate_and_benchmark advances input_args after the prefill step.
    input_args["input_ids"] = next_token_ids
    input_args["cache_position"] = next_cache_position
    return input_args


def transfer_to_device(input_args, device):
    """Mirrors transfer_to_device() from benchmarks/llm_benchmark.py (StaticCache
    path — GLM has no MLA layers).

    ONE intentional deviation: we do NOT call cumulative_length.zero_(). The
    benchmark's helper zeroes it because it is also used for the fresh pre-prefill
    warmup cache, but here the cache is already prefill-populated and
    StaticLayer.update() (transformers cache_utils.py:327) derives the write index
    from cumulative_length, not the passed cache_position. Zeroing would make the
    device decode overwrite prefill[0] and lose context, producing a meaningless
    PCC vs the CPU golden (which keeps cumulative_length = prompt_len). This is a
    runtime-value choice only — it does NOT change the traced IR (cumulative_length
    is a static-address tensor input; the arange/add ops are identical either way)."""
    for layer in input_args["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
        # See docstring: preserve cumulative_length (benchmark zeroes it here).
        layer.cumulative_length = layer.cumulative_length.to(device)
        layer.device = device
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)
    return input_args


def _shard_kv_cache(past_key_values, mesh):
    """Mirrors _shard_kv_cache() — StaticCache path with the test's explicit spec."""
    for layer in past_key_values.layers:
        xs.mark_sharding(layer.keys, mesh, KV_CACHE_SHARDING_SPEC)
        xs.mark_sharding(layer.values, mesh, KV_CACHE_SHARDING_SPEC)


def _apply_tp_sharding(model):
    """Mark sharding on weights + lm_head all-gather hook. Returns the mesh.
    Must be called BEFORE any weight parametrization."""
    num_devices = xr.global_runtime_device_count()
    mesh_shape, mesh_name = get_mesh_config(num_devices)
    mesh = Mesh(np.array(range(num_devices)), mesh_shape, mesh_name)

    shard_specs = glm_4_7_shard_spec(model)
    if shard_specs is not None:
        for tensor, spec in shard_specs.items():
            xs.mark_sharding(tensor, mesh, spec)

    if hasattr(model, "lm_head") and model.lm_head is not None:
        hook = sharding_constraint_hook(model.lm_head, mesh, (None, None, None))
        model.lm_head.register_forward_hook(hook)

    return mesh


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def run_pytorch_model():
    # CPU reference for the first DECODE step (matches the benchmark's
    # cpu_decode_logits): prefill populates the cache (cumulative_length ->
    # prompt_len), then a single decode forward through the same wrapper. Returns
    # the full logits (element [3] of the wrapper's 4-tuple).
    model = load_pytorch_model()
    decode_args = cpu_prefill_to_decode_state(model)

    cpu_wrapper = LLMSamplingWrapper(model, default_read_logits_fn, return_logits=True)
    cpu_wrapper.eval()
    with torch.no_grad():
        _, _, _, logits = cpu_wrapper(**decode_args)

    return logits


def run_tt_model():
    # 1. SPMD before device — tensors created after get presharded annotations.
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    device = torch_xla.device()

    # 2. Load model (CPU) and populate the KV cache with a CPU prefill BEFORE
    #    moving to device — this keeps the prefill out of the compiled graph so
    #    only the DECODE graph is traced/run. (Benchmark also runs its CPU
    #    prefill after use_spmd().)
    model = load_pytorch_model()
    decode_args = cpu_prefill_to_decode_state(model)

    # 3. Move model to device WITH dtype (casts buffers, e.g. rotary inv_freq).
    model = model.to(device, dtype=DATA_FORMAT)

    # 4. Mark sharding on weights FIRST (no weight_dtype_overrides for GLM).
    mesh = _apply_tp_sharding(model)

    # 5. Compile options.
    torch_xla.set_custom_compile_options(COMPILE_OPTIONS)

    # 6. Verbatim LLMSamplingWrapper (mesh + output spec, as the benchmark's
    #    compiled_logits wrapper); compile the wrapper (not the raw model).
    wrapper = LLMSamplingWrapper(
        model,
        default_read_logits_fn,
        return_logits=True,
        mesh=mesh,
        output_sharding_spec=INPUT_OUTPUT_SHARDING_SPEC,
    )
    wrapper.eval()
    compiled = torch.compile(wrapper, backend="tt")

    # 7. Decode inputs: move to device (cumulative_length preserved), shard.
    inputs = transfer_to_device(decode_args, device)
    _shard_kv_cache(inputs["past_key_values"], mesh)
    xs.mark_sharding(inputs["input_ids"], mesh, INPUT_OUTPUT_SHARDING_SPEC)

    # 8. Decode forward. Take the replicated logits (4-tuple element [3]).
    with torch.no_grad():
        _, _, _, logits = compiled(**inputs)

    return logits.cpu()


def codegen_model():
    os.environ["XLA_HLO_DEBUG"] = "1"

    # Same TP init order + wrapper as run_tt_model, but with the codegen_py backend
    # so the sharded decode graph is emitted. We inline codegen_py's body (rather
    # than calling tt_torch.codegen_py) to control the to(device) / mark_sharding
    # ordering: mark_sharding must run after the tensors are on the XLA device and
    # before the forward, and codegen_py's own arg handling would drop the
    # non-tensor past_key_values / use_cache kwargs.
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    device = torch_xla.device()

    model = load_pytorch_model()
    # Populate the cache with a CPU prefill so codegen emits ONLY the decode graph.
    decode_args = cpu_prefill_to_decode_state(model)

    model = model.to(device, dtype=DATA_FORMAT)

    mesh = _apply_tp_sharding(model)

    options = {
        **COMPILE_OPTIONS,
        "backend": "codegen_py",
        "export_path": OUTPUT_DIR,
        "export_tensors": True,
    }
    torch_xla.set_custom_compile_options(options)

    wrapper = LLMSamplingWrapper(
        model,
        default_read_logits_fn,
        return_logits=True,
        mesh=mesh,
        output_sharding_spec=INPUT_OUTPUT_SHARDING_SPEC,
    )
    wrapper.eval()
    wrapper.compile(backend="tt", options={"tt_legacy_compile": True})

    inputs = transfer_to_device(decode_args, device)
    _shard_kv_cache(inputs["past_key_values"], mesh)
    xs.mark_sharding(inputs["input_ids"], mesh, INPUT_OUTPUT_SHARDING_SPEC)

    with torch.no_grad():
        wrapper(**inputs)

    xm.wait_device_ops()


def compute_pcc(golden_output, device_output):
    """VERBATIM copy of compute_pcc from tt-xla/tests/benchmark/utils.py — the
    exact PCC the benchmark's decode check uses (float32 cast, centered Pearson,
    clamped to [-1, 1])."""
    golden_flat = golden_output.to(torch.float32).flatten()
    device_flat = device_output.to(torch.float32).flatten()

    golden_centered = golden_flat - golden_flat.mean()
    device_centered = device_flat - device_flat.mean()
    denom = golden_centered.norm() * device_centered.norm()

    if denom == 0:
        if torch.allclose(golden_flat, device_flat, rtol=1e-2, atol=1e-2):
            return 1.0
        raise ValueError(
            "PCC computation failed: denominator is zero but tensors are not close"
        )

    pcc = ((golden_centered @ device_centered) / denom).item()
    return max(-1.0, min(1.0, pcc))


def compare_pytorch_and_tt_runs():
    # Capture exact PCC from first --golden run and paste here.
    # First-decode PCC: PT (CPU decode) vs TT (device decode), both reading the
    # same CPU-prefilled cache — exactly the benchmark's decode_only PCC check
    # (compute_pcc(device_decode_logits, cpu_decode_logits)). The test's
    # required_pcc for this 4-layer GLM-4.7 decode is 0.86.
    exact_pcc = 0.858482

    pt_output = run_pytorch_model()
    tt_output = run_tt_model()

    assert pt_output.shape == tt_output.shape, (
        f"shape mismatch: {pt_output.shape} vs {tt_output.shape}"
    )
    assert pt_output.dtype == tt_output.dtype, (
        f"dtype mismatch: {pt_output.dtype} vs {tt_output.dtype}"
    )
    pcc = compute_pcc(pt_output, tt_output)
    print(f"PCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="test_glm_4_7_tp_galaxy_4_layers codegen pipeline"
    )

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
