# Llama-3.1-8B-Instruct Decode Graph — Optimization Summary

This document summarizes an agentic performance-optimization pass over the tt-forge-emitted TTNN single-token decode graph for `meta-llama/Llama-3.1-8B-Instruct`. The work was done in place on the emitted `graph_0` (32 decoder layers, batch size 32, one generated token per step) targeting a single Wormhole device with an 8x8 (64-core) compute grid. No forge or functional-decoder bringup stage was run; only the emitted graph and its harness were modified.

## Headline result (full model)

The warmed, trace-executed decode step for the full 32-layer model improved from a baseline of **0.0831 s/step (~385 TPS)** to a final **0.0791 s/step (~404 TPS)** while holding numerical correctness at **PCC 1.000000**. That is roughly a 4.8% latency reduction and about +19 tokens/s, with the final measurement reproduced across three consecutive trace-execute runs at 0.0791 s, 0.0791 s, and 0.0792 s.

All throughput figures in this document are **batch-aggregate** across the batch of 32 concurrent sequences, since one decode step emits one token for each of the 32 sequences. The **per-user** rate is therefore the aggregate divided by 32, which is what a single sequence actually experiences: the step latency itself is the per-user token latency. In per-user tokens-per-second-per-user (t/s/u) terms the graph improved from about **12.0 t/s/u** (0.0831 s/token per sequence) to about **12.6 t/s/u** (0.0791 s/token per sequence).

| Metric | Baseline (legalized graph) | Final (optimized) |
| --- | --- | --- |
| Warmed traced-decode latency (per step = per-user token latency) | 0.0831 s | 0.0791 s |
| Batch-aggregate throughput (batch 32) | ~385 TPS | ~404 TPS |
| Per-user throughput | ~12.0 t/s/u | ~12.6 t/s/u |
| Correctness gate (PCC vs HF golden) | 1.000000 | 1.000000 |

Both numbers are for the identical default code path, so the comparison is apples-to-apples: the baseline is the emitted graph after only the minimal fixes needed to make it run legally on this device, and the final is that same graph with the kept optimizations applied.

## How performance is calculated

Performance is measured by `main.py`, which runs the model five times in a fixed sequence and reports each timing so that dispatch/compile overhead is cleanly separated from steady-state device execution. The first run is a compile run that populates the program cache, the second run captures a device trace, and the final three runs replay that captured trace. Only the trace-execute runs are the headline latency, because they represent warmed steady-state execution with host dispatch overhead removed. Each measurement wraps `ttnn.execute_trace` followed by `ttnn.synchronize_device` in a wall-clock timer, and throughput is computed as tokens divided by elapsed seconds, where tokens per run equals batch size times tokens per sample (32 x 1 = 32). A single 0.0791 s step therefore corresponds to about 404 TPS across the batch, or equivalently about 12.6 tokens/s per user for each of the 32 sequences.

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

Five candidates were implemented, measured, and reverted because they either failed to move the warmed trace number, regressed it, or broke a downstream layout contract. Each is listed below with the specific reason it did not survive.

1. Removing only the unused logits copy. As a first data-movement experiment the redundant logits copy was deleted on its own, but the warmed trace latency was identical before and after at 0.0831 s, so it produced no measurable gain while also narrowing the graph's output contract. It was reverted because a change that costs contract surface must at least pay for itself in latency, and this one did not. (This is distinct from the tail-reshape removal that was ultimately kept, which additionally deletes the terminal reshape op and normalizes the logits shape on the host.)

2. Reusing the DRAM logits tensor for the final reshape. The next variant tried to avoid allocating a second full L1 interleaved copy purely to reshape the logits, reshaping from the returned DRAM tensor instead. It passed PCC but was measurably slower on the warmed replays, about 0.0837 s versus the 0.0831 s corrected baseline it was compared against, so it was reverted in favor of the host-side normalization that was eventually kept.

3. Calling `nlp_create_qkv_heads_decode` directly on the sharded QKV matmul output. This was an attempt to remove a repeated QKV head-creation conversion seen in the profile by feeding the sharded matmul output straight into the decode head-split. It failed to compile because the matmul output rank did not satisfy the operation's required 4D shape contract, so the direct path was abandoned.

4. Pre-reshaping the sharded QKV output to `[1, 1, 32, 6144]` before head creation. As a follow-up to the previous item, the sharded output was first reshaped to a 4D shape so the head-split would accept it. It compiled far enough to produce heads, but the resulting dimension order no longer matched the emitted rotary-embedding path, which then saw a sequence length of 8 instead of 1. Repairing that would have required inserting extra reshapes and transposes downstream, which was not a clean win, so it was reverted and the sharded QKV output was left feeding the existing head-splitting sequence unchanged.

5. DRAM-sharding the MLP down-projection. The same DRAM width-sharded matmul treatment that helped QKV was tried on the MLP down-projection. On the reduced single-layer trace it did not improve replay (about 0.0208 s, essentially flat against the current reduced path) and it introduced a much slower compile path, so it was rejected without a full-model run and only the QKV DRAM-sharding was kept.

For context, the run also hit several pure bring-up blockers before any tuning began — a Hugging Face `401` on the gated model that was resolved by using locally cached weights, emitted shard/core grids referencing coordinates outside the device's 8x8 grid, and an MLP down-projection whose 7-tile-wide intermediate shard was not divisible by the emitted `in0_block_w=8`. These were corrected as part of legalizing the graph (see the kept changes above) rather than rejected as optimizations.

## Validation

The final graph was additionally run under the device watcher (`TT_METAL_WATCHER=10`) and passed with PCC 1.000000 and no watcher fault, confirming the kept changes are not just faster but also free of device-safety violations. Watcher-enabled timings are slower by design and were not used as the headline numbers.

## Files in this change set

The optimization touches `model_ttnn.py` (the decode graph, including the DRAM-sharded QKV matmuls), `consteval.py` (the DRAM width-sharded weight memory-config helper and width-sharded QKV weight loading), and `main.py` (entrypoint fix, relaxed PCC gate, host-side logits normalization). The single-layer profiling scaffold used to gather device evidence is `profile_reduced.py`, and the collected `tt-perf-report` output lives under `perf_evidence/`.
