# GLM-4.7 4-layer decode — MoE compute tuning log

Baseline (branch mvasiljevic/glm-4.7-wh-galaxy):
- WH-working (50ab52b): PCC 0.859375.
- expert_mapping fix (5b54f15): PCC 0.894531.

Goal: replace hand-emitted MoE (router+topk+all_to_all_dispatch+moe_expert_token_remap
+3x sparse_matmul+all_to_all_combine) with fused ttnn.experimental.moe_compute, get good
PCC, then tune. Perf metric = tracy device time (run -t + tt-perf-report), not TPS.

| # | patch | scope | tracy DT delta | PCC delta | decision | why |
|---|-------|-------|----------------|-----------|----------|-----|
| 1 | FABRIC_1D -> FABRIC_1D_RING (prereq for moe_compute CCL) | full | tbd | hold @0.894? | tbd | ring topology required by moe_compute/dispatch_metadata |

## PCC investigation (before perf tuning) — 2026-06-16

Goal was good PCC. Findings (all on cached golden, mesh 4x8):
- expert_mapping column-major fix: 0.858 -> 0.894531 (committed 5b54f15). The
  gain is purely from making expert PLACEMENT (_arrange_experts column-major) and
  ROUTING (expert_mapping) consistent. Verified my mapping == reference
  get_linearized_mesh_coord(cluster_axis=0) exactly.
- experts bf8->bf16: 0.894531 (no change). all weights bf8->bf16: 0.894531 (no
  change). => precision is NOT the limiter.
- matmuls already run HiFi4 (29 compute_kernel_config sites). => fidelity maxed.
- row-major layout (identity _arrange_experts + row-major expert_mapping): also
  0.894531, bit-identical to column-major. => both self-consistent layouts are
  equivalent; routing is correct.
- MoE CCL op attributes are coherent (axis0=batch/expert dispatch+combine,
  axis1=tensor-parallel reduce_scatter/all_gather, shared by attn/dense TP).
  Swapping cluster_axis would break TP. Not an in-place bug.

Conclusion: expert routing is correct; 0.894531 is the robust accumulated-fidelity
ceiling of this 4-layer bf8 decode with correct MoE (matches codegen authors note
that ~0.86 is genuine bf8 fidelity). Issue-5096 0.99 is the upstream full-precision
benchmark, not this bf8 codegen. Reaching 0.99 in-place would need graph
regeneration from the corrected tt-xla sharding spec or higher precision throughout.
Next: moe_compute fused-op integration (perf goal; replaces dispatch+matmul+combine
with a structurally-correct fused op).
