# moe_compute integration (smoke test) — 2026-06-17

- moe_compute path validated incrementally via standalone moe_compute_smoke.py (random GLM-shaped weights, no HF load). Confirmed working: bf4 weight packing, all_to_all_dispatch_metadata, moe_compute (returns 6 tensors, combine [8,16,5120]).
- HANG (deterministic): epilogue `reduce_scatter(cluster_axis=1)` immediately after moe_compute hangs (timeout 124) regardless of topology (Ring/Linear) or rank-3/rank-4 input. Hypothesis: moe_compute leaves combine semaphore / L1 / fabric busy; needs sync or dealloc of moe_compute's other 5 outputs before the cross-col CCL.
- CRASH (nondeterministic): after a reset+rerun, SIGBUS "Non-existent physical address" during weight from_torch (right after "packed w0w1"), before dispatch. Appears to be post-reset device instability from the repeated hang-kill-reset cycle, not a code bug (earlier runs passed this stage). Mitigation: reset again, rerun.

# perf iter12 router-gate 1D matmul config — DEVICE FAULT — 2026-07-08

Applying `MatmulMultiCoreReuseMultiCast1DProgramConfig(grid=5x1, in0_block_w=20, per_core_N=1,
mcast_in0=True)` to the router-gate matmul (FP32 out, activation=sigmoid, [16,5120]x[5120,160])
faulted the device during the forward: repeated `TT_FATAL: Read unexpected run_mailbox value from
core 25-17`. The same 1D-mcast approach is fine for qkv (bf16, 7x8) and shared experts (bf16, 6x1)
— the router differs in FP32 output + sigmoid fused-activation + a narrow 5-wide grid. Prior to
this the single-device micro-bench of the same config couldn't even build the FP32
`bmm_large_block_zm_fused_bias_activation` kernel (NFS cache rename flake). Reverted; router left on
the default config. Not worth chasing (~2.8ms, x89) given the FP32 fused-activation-matmul path is
unstable here. Recovery: pkill + `tt-smi -glx_reset_auto`.
