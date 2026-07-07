# Llama-3.1-8B-Instruct Decode Graph — Optimization Summary

This document summarizes an agentic performance-optimization pass over the tt-forge-emitted TTNN single-token decode graph for `meta-llama/Llama-3.1-8B-Instruct`. The work was done in place on the emitted `graph_0` (32 decoder layers, batch size 32, one generated token per step) targeting a single Wormhole device with an 8x8 (64-core) compute grid. No forge or functional-decoder bringup stage was run; only the emitted graph and its harness were modified.

## Headline result (full model)

The warmed, trace-executed decode step for the full 32-layer model improved from a baseline of **0.0831 s/step (~385 TPS)** to a final **0.0791 s/step (~404 TPS)** while holding numerical correctness at **PCC 1.000000**. That is roughly a 4.8% latency reduction and about +19 tokens/s, with the final measurement reproduced across three consecutive trace-execute runs at 0.0791 s, 0.0791 s, and 0.0792 s.

| Metric | Baseline (legalized graph) | Final (optimized) |
| --- | --- | --- |
| Warmed traced-decode latency | 0.0831 s/step | 0.0791 s/step |
| Throughput | ~385 TPS | ~404 TPS |
| Correctness gate (PCC vs HF golden) | 1.000000 | 1.000000 |

Both numbers are for the identical default code path, so the comparison is apples-to-apples: the baseline is the emitted graph after only the minimal fixes needed to make it run legally on this device, and the final is that same graph with the kept optimizations applied.

## How performance is calculated

Performance is measured by `main.py`, which runs the model five times in a fixed sequence and reports each timing so that dispatch/compile overhead is cleanly separated from steady-state device execution. The first run is a compile run that populates the program cache, the second run captures a device trace, and the final three runs replay that captured trace. Only the trace-execute runs are the headline latency, because they represent warmed steady-state execution with host dispatch overhead removed. Each measurement wraps `ttnn.execute_trace` followed by `ttnn.synchronize_device` in a wall-clock timer, and throughput is computed as tokens divided by elapsed seconds, where tokens per run equals batch size times tokens per sample (32 x 1 = 32). A single 0.0791 s step therefore corresponds to about 404 TPS across the batch.

Correctness is gated in the same run: the last-token logits produced on device are compared against the Hugging Face PyTorch golden using PCC, and the harness asserts PCC is at or above 0.9921875. Every kept optimization was required to keep this gate passing, and the final graph passes it at 1.000000.

Device-level evidence was collected separately with a reduced single-layer profiling harness (`profile_reduced.py`, `num_layers=1`) that emits Tracy signposts around the decode region, after which `tt-perf-report` slices out the per-op device window between those signposts. This isolates per-operation cost without the noise of all 32 layers, and it is the source of the roofline and op-count numbers below. These reduced numbers are diagnostic only and are not the same as the full-model wall-clock latency above.

## Device-level change (reduced single-layer window)

On the reduced single-layer trace, the device window shrank from **8.435 ms across 71 ops** to **5.034 ms across 69 ops**, and modeled DRAM roofline utilization rose from **28.2% (81 GB/s)** to **47.2% (136 GB/s)**. The higher roofline utilization reflects that the QKV projection now moves its weights through DRAM in a width-sharded layout that keeps more bandwidth productive, and the two fewer ops come from removing a redundant logits reshape and copy at the tail of the graph.

## What was tried and kept

The following changes were validated to preserve PCC and were kept in the final graph.

The harness entrypoint was fixed so that `python3 main.py` actually runs the harness: the emitted file called a nonexistent `main()` at the bottom while the harness function was defined as `test_main()`. Without this the graph could not be exercised at all, so it is the precondition for every subsequent measurement.

The correctness gate was changed from an exact-equality assertion (`pcc == 0.9921875`) to a minimum-threshold assertion (`pcc >= 0.9921875`). Exact float equality is brittle and does not express the intended contract, which is that accuracy must not drop below the emitted tolerance; the final graph clears this comfortably at 1.000000.

The hard-coded shard and core-grid specifications emitted by tt-forge were legalized for the actual 8x8 compute grid so that the graph runs correctly on this device rather than the grid the emitter assumed. This is what defines the baseline path that the optimizations are compared against.

The terminal device-side logits reshape and a redundant second L1 copy of the logits were removed, and `main.py` now normalizes the logits shape on the host before the PCC comparison. This deletes real device work from the tail of the graph without changing the numbers that are compared.

The fused QKV weights were moved to a DRAM width-sharded layout and the QKV matmuls were switched to `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`. This is the primary compute-path optimization: it improves how the large QKV projection streams its weights from DRAM, which is what drives the roofline utilization from 28.2% to 47.2% on the reduced window.

## What was tried and rejected

Several candidates were implemented, measured, and reverted because they either regressed performance or broke a layout contract.

Reshaping directly from the returned DRAM logits tensor was slower than the host-side normalization that was kept, measuring about 0.0837 s versus the final 0.0791 s, so it was reverted.

Calling `nlp_create_qkv_heads_decode` directly on the sharded QKV output failed the downstream shape and rotary-embedding layout contracts, so the sharded QKV path was kept feeding the existing head-splitting sequence instead.

A DRAM-sharded candidate for the MLP down-projection regressed the reduced trace (about 0.0208 s on the reduced window) and was reverted, so the MLP down-projection was left on its original path.

## Validation

The final graph was additionally run under the device watcher (`TT_METAL_WATCHER=10`) and passed with PCC 1.000000 and no watcher fault, confirming the kept changes are not just faster but also free of device-safety violations. Watcher-enabled timings are slower by design and were not used as the headline numbers.

## Files in this change set

The optimization touches `model_ttnn.py` (the decode graph, including the DRAM-sharded QKV matmuls), `consteval.py` (the DRAM width-sharded weight memory-config helper and width-sharded QKV weight loading), and `main.py` (entrypoint fix, relaxed PCC gate, host-side logits normalization). The single-layer profiling scaffold used to gather device evidence is `profile_reduced.py`, and the collected `tt-perf-report` output lives under `perf_evidence/`.
