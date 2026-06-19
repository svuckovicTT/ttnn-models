# Hangs log (GLM-4.7 attention tuning)

No hangs occurred during this tuning session.

The hang-risk changes were the CCL edits: the `all_reduce` fusion (#2) and the
`num_links=3` bumps on the o_proj `reduce_scatter`/`all_gather` (#6/#7). All ran to
completion within the normal time budget (no 99%-CPU wedge, no `kill -KILL`, no
`tt-smi -glx_reset_auto` needed). Every run was launched backgrounded with a
`timeout --signal=KILL` budget (900 s PCC / 1800 s tracy) per the hang discipline,
but the budget was never hit.
