# Unpacked-query-head FA3 comparison

This opt-in candidate tests whether packing query heads contributes to the
remaining tensor-utilization gap. It is not a promoted serving path.

The planner enumerates raw query tokens independently for each query head;
the scheduler maps each query head to its KV head. FA3 is deliberately unsplit
and writes directly to its output. FA2 retains its existing packed scheduling
and split reduction. The compact primary rejects split plans in this mode.
The loader requires `FUSED_PACKED_Q=off`, disables paged TMA, and uses a separate
`_unpacked` operator namespace. Default packed behavior is unchanged.

Both KV96 and KV128 compile and register locally. The host test proves exact
token/head coverage for query lengths512/17/129, GQA7 and four KV heads, with
both work ordering modes. It also checks that FA3 has no partial reductions
while FA2 split merges remain present. Existing packed-plan round-trip,
independent-role chunk and cluster-pair tests pass. Logs:
`fa3_unpacked_host_tests_20261014.txt`,
`fa3_unpacked_kv96_compile_20261014.log`, and
`fa3_unpacked_kv128_compile_20261014.log`.

At 03:30:06 +08:00 on 2026-10-14, PID653358 was verified live on the existing
H200, waiting for serving PID637372. It does not compile or use the GPU until
that predecessor exits and it acquires `a2_kernel.lock`. The runner then checks
for other serving/kernel processes and works only in private cloned trees.

`fa3_unpacked_kernel_20261014.sh` compares packed-auto, packed-unsplit,
unpacked-KV128 and unpacked-KV96, with matched round-robin-pair assignment.
There are16 numerical/graph/overlap gates and40 timings across fresh-prefill,
long-prefill, append+verify, FA3-only and irregular-length fixtures. The
packed-unsplit control separates query packing from split-KV differences.
No GPU correctness or performance result is available yet.

The patch is `fa3_unpacked_20261014.patch`. The source overlay is52,386 bytes,
SHA256 `7dd6207b1328d1d0b86ca87e06dcc763075aa856f73e2a43fc8abace9cb53846`.
Baseline driver and changed-source hashes were checked on the trial before
queuing; candidate hashes are checked after private-tree construction.
The main task document and its large TODOs are untouched.
