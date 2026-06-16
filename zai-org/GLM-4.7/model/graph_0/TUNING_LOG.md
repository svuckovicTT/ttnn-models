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
