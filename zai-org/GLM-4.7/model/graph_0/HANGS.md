# moe_compute integration (smoke test) — 2026-06-17

- moe_compute path validated incrementally via standalone moe_compute_smoke.py (random GLM-shaped weights, no HF load). Confirmed working: bf4 weight packing, all_to_all_dispatch_metadata, moe_compute (returns 6 tensors, combine [8,16,5120]).
- HANG (deterministic): epilogue `reduce_scatter(cluster_axis=1)` immediately after moe_compute hangs (timeout 124) regardless of topology (Ring/Linear) or rank-3/rank-4 input. Hypothesis: moe_compute leaves combine semaphore / L1 / fabric busy; needs sync or dealloc of moe_compute's other 5 outputs before the cross-col CCL.
- CRASH (nondeterministic): after a reset+rerun, SIGBUS "Non-existent physical address" during weight from_torch (right after "packed w0w1"), before dispatch. Appears to be post-reset device instability from the repeated hang-kill-reset cycle, not a code bug (earlier runs passed this stage). Mitigation: reset again, rerun.
