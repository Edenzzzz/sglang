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

## Completed GPU result

The run exited successfully at04:40:56 +08:00. All16 numerical/graph gates and
40 timed cases pass; raw-log hashes were rechecked during collection.
Two-repeat candidate graph times, in milliseconds:

| Fixture | Packed auto | Packed unsplit | Unpacked128 | Unpacked96 |
|---|---:|---:|---:|---:|
| Fresh512 +46 verify | 2.336943 | 2.335202 | 2.330976 | 2.332505 |
| Prefill2048 +8 verify | 4.762935 | 4.751546 | 4.751112 | 4.954453 |
| Append512 +46 verify | 3.238782 | 3.219771 | 3.233372 | 3.249014 |
| FA3-only8x1024 | 9.536257 | 9.546193 | 9.480100 | 9.904263 |
| Irregular lengths | 0.333568 | 1.249764 | 1.227718 | 1.309019 |

Unpacking does not close the FA3 gap. KV128 differences on the four larger
fixtures are below0.6%; KV96 is slower. Removing splitting on the irregular
short-query fixture costs approximately3.7x. Retain packed auto, and do not
promote this unsplit candidate into serving.

`fa3_unpacked_complete_20261014_audit.json` contains the16 gate records and40
timings. Full raw logs are in `paired_unpacked_complete_20261014.tar.gz`,
43,021 bytes, SHA256
`b42013039ba233e112489fc4058917c88dfaca81bf918ba28ed554bffbd451bf`.
