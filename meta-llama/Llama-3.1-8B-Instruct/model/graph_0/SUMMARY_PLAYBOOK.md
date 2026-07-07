# How to Write an Optimization Summary (playbook for the next run)

This is a reusable guide for turning an agentic performance-optimization run into a summary like `OPTIMIZATION_SUMMARY.md`. It is written for the next graph, not as a repro of one specific run, so paths are given as patterns to resolve for whatever model and session you are documenting.

## Inputs to collect

- The agent session transcript, which is the primary source and the most complete record. For Codex CLI runs it is a JSONL rollout under `~/.codex/sessions/<YYYY>/<MM>/<DD>/rollout-*.jsonl`; pick the session whose `cwd` matches the graph directory and whose size/mtime matches the run you are documenting.
- The run's own deliverable log if the task wrote one (for example a `tuning_run.log`), used to cross-check the agent's final numbers.
- The on-disk code and its diff against the base branch, so every claim can be verified against what actually shipped: the model graph, the weight/consteval setup, the harness, and any profiling scaffold.
- The profiler evidence directory (for example `perf_evidence/`), which holds the `tt-perf-report` text, CSV, and console logs with the device-level numbers and roofline utilization.

## How to extract each part

- Parse the JSONL by `payload.type` and `payload.role`. The first real `user` message is the task/goal; the last `assistant` message is usually the final summary. Read both verbatim.
- For the list of rejected or failed trials, filter `assistant` messages for words like `revert`, `regress`, `slower`, `fail`, `reject`, and quote the log's own numbers rather than paraphrasing them.
- For the kept changes, run a diff of the changed files against the base branch and grep the graph for the specific ops or program configs that were introduced, so the "kept" list is verified in code and not just asserted in prose.
- For device numbers (per-op timings, op counts, roofline), read the profiler console logs and report text directly.
- For run counts and iterations per try, parse the transcript's `function_call` events for invocations of the full harness and the reduced profiling script, and take their timestamps.
- For timing, use the run timestamps to compute a wall-clock window per phase, and the session start/end for the total. Do not present these as device execution time; the transcript polls long runs at fixed intervals, so it does not record isolated per-run device time.

## What the summary should contain

- A headline full-model before/after: warmed traced-decode latency, throughput, and the correctness gate value, all for the identical default code path so the comparison is apples-to-apples.
- Throughput stated as batch-aggregate with the per-user (t/s/u) equivalent, since one decode step emits one token per sequence in the batch and "TPS" alone is ambiguous.
- A clear description of how performance is measured (the run sequence, which runs are the headline, how tokens-per-second is computed, and the correctness gate).
- The device-level change from the reduced or profiled window (per-op deltas, op-count change, roofline utilization).
- A "what was tried and kept" list and a "what was tried and rejected" list, each item with its specific, log-sourced reason.
- A scope section that records only what the logs state about candidate selection and where the search ended, plus a runs/iterations/timing table.
- A validation note (for example a watcher run) and a short list of the files in the change set.

## Rules

- Include only facts found in the logs or verifiable on disk, and prefer direct quotes for numbers and failure reasons.
- Never infer why something was or was not done; if the logs do not state a reason, say so explicitly.
- Always label wall-clock time as distinct from device execution time.
- Keep the full-model numbers and the reduced/profiled numbers clearly separated, and never present a single-layer reduced number as a full-model result.
- Leave any concrete, un-tried profiler recommendation documented as an open lead rather than implying it was ruled out.
