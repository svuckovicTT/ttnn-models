# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Re-emit GLM-4.7 4-layer decode with the WRONG or CORRECT MoE/kv shard specs,
from the SAME current tt-mlir, into SEPARATE output dirs so the prettified
``model/graph_0`` is never clobbered.

Two purposes:
  1. ``--golden`` confirms PCC for a given sharding. Running ``--variant correct
     --golden`` is the disambiguator: if it hits ~0.99 the gap was purely the
     baked-in specs (and the raw diff below is a valid port template); if it
     stays ~0.89 the gap is the tt-mlir uplift regression (issue #5209) and no
     porting will help until tt-mlir is fixed.
  2. ``--codegen`` emits the raw codegen_py package. Emitting BOTH variants and
     diffing ``model_wrong/graph_0/ttnn.mlir`` vs ``model_correct/graph_0/ttnn.mlir``
     isolates exactly what the sharding fix changes structurally — the template
     to port into the hand-prettified model_ttnn.py / params.py.

This reuses everything from the existing standalone ``xla.py`` and only overrides:
  - the routed-expert shard spec  ("model","batch") -> ("batch","model")
  - KV_CACHE_SHARDING_SPEC        ("batch",None,...) -> ("batch","model",None,None)
  - OUTPUT_DIR                    model/ -> model_{wrong,correct}/

Usage (on the 32-device galaxy box, tt-xla venv active):
  python xla_reemit.py --variant correct --golden
  python xla_reemit.py --variant correct --codegen
  python xla_reemit.py --variant wrong   --codegen
"""

import argparse
from pathlib import Path

import xla  # the existing standalone reproduction in this directory

_HERE = Path(__file__).resolve().parent

# PR #5240 corrected specs.
_CORRECT_EXPERT_AXIS = ("batch", "model")
_CORRECT_KV_CACHE_SHARDING_SPEC = ("batch", "model", None, None)


def _correct_glm_4_7_shard_spec(model):
    """Copy of xla.glm_4_7_shard_spec with the routed-expert compound axis fixed
    to ("batch","model") per PR #5240. Everything else is identical."""
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

        if isinstance(mlp, xla.A2aSparseMLPWithSharedExperts):
            inner = mlp.mlp  # A2aSparseMLP
            shard_specs[inner.router.gate.weight] = (None, None)
            shard_specs[inner.experts.gate_proj] = (_CORRECT_EXPERT_AXIS, None, None)
            shard_specs[inner.experts.up_proj] = (_CORRECT_EXPERT_AXIS, None, None)
            shard_specs[inner.experts.down_proj] = (_CORRECT_EXPERT_AXIS, None, None)

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


def _apply_variant(variant: str) -> str:
    """Monkeypatch the xla module for the chosen variant. Returns output dir."""
    if variant == "correct":
        xla.glm_4_7_shard_spec = _correct_glm_4_7_shard_spec
        xla.KV_CACHE_SHARDING_SPEC = _CORRECT_KV_CACHE_SHARDING_SPEC
        out_dir = str(_HERE / "model_correct")
    elif variant == "wrong":
        # Leave xla's originals untouched; only redirect output so we get a RAW
        # wrong emit (apples-to-apples with the correct one) without clobbering
        # the prettified model/graph_0.
        out_dir = str(_HERE / "model_wrong")
    else:
        raise ValueError(f"unknown variant: {variant}")

    xla.OUTPUT_DIR = out_dir
    return out_dir


def _golden():
    pt_output = xla.run_pytorch_model()
    tt_output = xla.run_tt_model()
    assert pt_output.shape == tt_output.shape, (
        f"shape mismatch: {pt_output.shape} vs {tt_output.shape}"
    )
    pcc = xla.compute_pcc(pt_output, tt_output)
    print(f"\n=== REEMIT PCC ({pcc:.6f}) ===")
    return pcc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=["wrong", "correct"],
        required=True,
        help="Which shard spec to use.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--golden", action="store_true", help="Run PT vs TT, print PCC")
    mode.add_argument("--codegen", action="store_true", help="Emit codegen_py package")
    mode.add_argument("--run-tt", action="store_true", help="Run on TT only")
    args = parser.parse_args()

    out_dir = _apply_variant(args.variant)
    print(f"[xla_reemit] variant={args.variant} output_dir={out_dir}")
    print(f"[xla_reemit] KV_CACHE_SHARDING_SPEC={xla.KV_CACHE_SHARDING_SPEC}")

    if args.golden:
        _golden()
    elif args.codegen:
        xla.codegen_model()
        print(f"[xla_reemit] codegen written under {out_dir}")
    elif args.run_tt:
        xla.run_tt_model()


if __name__ == "__main__":
    main()
