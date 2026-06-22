# Hangs log (GLM-4.7 attention tuning)

No hangs occurred during this tuning session.

The hang-risk changes were the CCL edits: the `all_reduce` fusion (#2) and the
`num_links=3` bumps on the o_proj `reduce_scatter`/`all_gather` (#6/#7). All ran to
completion within the normal time budget (no 99%-CPU wedge, no `kill -KILL`, no
`tt-smi -glx_reset_auto` needed). Every run was launched backgrounded with a
`timeout --signal=KILL` budget (900 s PCC / 1800 s tracy) per the hang discipline,
but the budget was never hit.

## Update (iter 12, profiling-only, not a model hang)

During iter 12's tracy run, the host-side profiler step that copies the 1.78 GB
`profile_log_device.csv` into the report dir **wedged on NFS** (`cp` at 0% CPU, dest already
full-size, ~13 min no progress). This is a host/NFS I/O stall, not a device hang. Killing the
wedged tracy then corrupted the profiler `.logs` state, so the next tracy failed with
`End marker found without a corresponding start marker` (profiler.cpp) and then a
`Read unexpected run_mailbox value from core 25-17` device error. Recovery that worked:
`rm -rf generated/profiler/.logs/*` + `tt-smi -glx_reset_auto` (via the tt-xla venv;
tt-smi is at `venv/bin/tt-smi`, not on PATH) + rerun. The clean rerun produced a valid CSV.
The iter-12 model code itself was fine throughout (its PCC run passed at 0.902344). Lesson:
prefer letting a wedged *profiler post-process* finish or time out rather than KILL mid-copy,
since the kill corrupts `.logs` and forces a device reset.
