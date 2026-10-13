# FA3 work-distribution A/B

Nsight's low L2 hit rate on the gather primary motivated separating work
distribution from TMA. This test uses the same gather binary, page128, 112 CTAs,
no hardware cluster and no multicast in both arms. Only
`FUSED_FA3_CLUSTER_PAIR` changes: contiguous per-CTA slices versus distributing
consecutive work pairs round-robin across CTA pairs.

All 16 cases passed their seven numerical/stock/changed-Q graph checks,
completing at 01:09:45 +08:00 on 2026-10-14. Two-repeat mean graph times:

| Fixture | Contiguous (ms) | Round-robin pairs (ms) | Change |
|---|---:|---:|---:|
| FA3-only 8x1024 at KV32K | 11.115943 | 9.530803 | -14.26% |
| Prefill 2048 + eight verify | 4.809738 | 4.761919 | -0.99% |
| Append 512 + 46 verify | 3.237304 | 3.225806 | -0.36% |
| Varied odd queries/KV lengths | 0.452991 | 0.333458 | -26.39% |

The substantial gains on two fixtures occur without TMA. Therefore the earlier
gather-versus-TMA configuration gains cannot be credited solely to the load
instruction. Work assignment matters; a follow-up profile checks whether the
cache counters support the proposed locality mechanism. Neither this kernel
A/B nor that profile establishes a serving improvement.

`fa3_locality_complete_20261014.tar.gz` is 5,053 bytes, SHA256
`28e3f714516ff1bd79aa8ca663436806d53a04521096c84b5fa27aae888a1843`.
All 16 raw-log hashes match the adjacent completed audit JSON. The main task
document and production defaults remain unchanged.

## Cache mechanism checked

The follow-up gather-only profile uses the same verified FA3-only workload,
112 CTAs and no TMA. Changing only work distribution raises L2 hit rate from
11.92% to 94.86%; reported DRAM active cycles fall from 55.30% to 1.54% of
peak elapsed cycles. Tensor activity remains approximately 43.6%, so improved
locality does not explain the remaining tensor-utilization gap to stock FA3.
Profiler durations are not substituted for the normal A/B timings above.

`fa3_locality_ncu_complete_20261014.json` preserves the full counters and logs.
The remote report is 3,123,361 bytes, SHA256
`fe0e839ababc828690d237f004e5b6aecbd6daf238f2676e46a9445173a45001`.
The next distinct hypothesis is the stock kernel's KV tile96 versus the custom
tile128; enabling it requires correct split alignment and fresh numerical gates.
