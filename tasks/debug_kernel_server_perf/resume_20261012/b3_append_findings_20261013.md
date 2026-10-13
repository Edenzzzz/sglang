# B3 agent-append KV-load diagnostic, 2026-10-13

The previously pending append fixtures ran from 13:09:42 to 13:12:04 +08:00 on
H200 trial `095f01a3e1585eea`. All three normal arms pass fp32 and CUDA-graph
replay correctness. All three zero-KV-load arms fail on every output row/head as
expected, while their unfused controls pass. All six logs pass the strict audit;
the runner exits zero only after those separate expected verdicts are verified.

| FA3 append fixture | Normal ms | Zero-KV diagnostic ms | Latency reduction |
|---|---:|---:|---:|
| 4 rows x q512, KV65536 | 6.1923 | 4.8123 | 22.29% |
| 8 rows x q1024, KV32768 | 11.9439 | 9.6627 | 19.10% |
| 2 rows x q256, KV131072 | 3.1040 | 2.4190 | 22.07% |

Configuration preserves Claude's queued experiment: GQA28/4, head dimension128,
FA3 q128 tile, SM split112, native FA2 co-launch, classes64/32/16, auto KV chunk,
work-budget splitting and backfill. Timing uses 50 graph replays and 10 warmups.
`fi_skip`, `merged_skip`, and `torch_ext_skip0/1` isolate these builds from serving.
Source hashes, environment, complete logs and per-case audits are preserved.

These append fixtures expose more load-removal headroom than the earlier 5–7%
mixed-fixture diagnostics. It would be incorrect to dismiss append-shape
multicast using only those earlier small percentages. Nevertheless, this removes
all KV global loads and deliberately changes the output. It is only a diagnostic
headroom estimate, not a measured multicast implementation, a predicted multicast
gain, or a serving speedup. No linear half-of-headroom claim is made.

The selected serving split uses stock FA3 plus nonpersistent FA2, whereas this
experiment uses the custom fused FA3 implementation. Transferring these numbers
to the selected serving backend requires a new measurement. Multicast must still
be implemented, output-gated and evaluated on the actual serving path; B3 is open.
The main checklist and its large todo items remain unchanged.

Local evidence: `b3_append_complete_20261013.tar.gz` and
`b3_append_complete_20261013_audit.json`. Remote source artifacts remain under
`/dev/shm/w/b3_append_resume_20261013`; the exact runner is
`b3_append_resume_20261013.sh`. The middle-range A3b run followed it on the same
exclusive GPU lock; the superseded Claude queue remains stopped.
