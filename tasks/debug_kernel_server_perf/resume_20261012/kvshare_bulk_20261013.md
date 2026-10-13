# B3 asynchronous bulk DSMEM exchange

The first embedded KV-sharing implementation passed all 48 GPU cases but took
about three times as long as redundant gathering on long append shapes. This
follow-up replaces its per-vector remote loads with one asynchronous shared-to-
cluster bulk copy per producer CTA and K/V tile. It is still cooperative paged
KV sharing, not a global-memory TMA multicast instruction.

## Implementation and ordering

`FUSED_FA3_KV_SHARE_BULK=1` selects the new path inside the same compiled kernel.
It partitions gathering by physical shared-memory half, rather than assuming
that a logical head-dimension half is contiguous. Each CTA publishes its actual
pipeline-stage index, completes its local gather and arms a receive transaction
barrier. After the existing paired ready barrier, one thread pushes its half
into the peer's matching stage with `cp.async.bulk.shared::cluster.shared::cta`.

All producer threads wait for their received half, then participate in the
paired completion barrier. Consequently neither source buffer can be reused
until both copies have completed. Barrier parity persists across loads and
work items; unmatched work retains independent gathering. Existing consumer
pipeline commits, cluster exit synchronization and `tma_store_wait<0>()` remain.
The source-lifetime and 16-byte alignment requirements follow the
[NVIDIA PTX instruction specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk).

This deliberately retains the ready/completion handshakes to isolate the remote
copy change; it does not claim to remove synchronization overhead or implement
a fully asynchronous inter-CTA pipeline.

## Evidence and pending validation

Local CUDA compilation and operator registration pass. Disassembly contains
`UBLKCP.S.S`; this proves code generation, not execution. The existing C7510
WGMMA serialization warning remains in the build log.

The host layout test passes 112 cases using the actual FA3 K/V layouts: one,
two and three stages, every producer/peer stage pairing, and valid KV extents
1, 17, 127 and 128. It checks dense stage coverage, vector alignment, partition
ownership, distinct-stage exchange and zero-filled tails. This does not prove
device synchronization correct.

`kvshare_bulk_kernel_20261013.sh` is queued as PID 506950 from 19:59:41 +08:00
on 2026-10-13. It waits for the compact native-primary validation PID 501745,
then takes the existing exclusive GPU lock. It uses private
`fi_kvshare_bulk`, `merged_kvshare_bulk` and `torch_ext_kvshare_bulk` paths.
Three smoke gates require actual sharing/bulk counters for the selected arm and
zero bulk counters for controls. Then 54 cases compare redundant gathering,
the previous DSMEM exchange and bulk exchange within one binary. Second-repeat
arm order reverses. Numerical, graph-replay and unfused-control checks are
mandatory, including odd query/KV tails, append contexts 32/64/128K, mixed q4/q8
and FA2-only work. Timing runs prohibit device tracing.

The overlay archive is 52,974 bytes, SHA256
`09760f3a82cd5fdf9b307bb273add4265cdfc52d7bacd4328fd4c97d4792d952`.
The adjacent patch is relative to the first KV-sharing prototype. The source
overlay includes before/after manifests and the host layout test. No serving
source is promoted, and the main task document and big TODOs remain unchanged.

## Completed GPU result

All three smoke gates and all 54 timing cases passed, completing at 21:03:43
+08:00 on 2026-10-13. The bulk smoke reports one shared item and 58 actual bulk
copies on CTA0; both controls report zero bulk copies. Every timing case passes
fp32, graph-replay and unfused-control checks, including odd KV/query tails.

Two-repeat mean kernel times within the same compiled image:

| Fixture | Redundant (ms) | Previous exchange (ms) | Bulk exchange (ms) |
|---|---:|---:|---:|
| 4x512 append, KV64K | 7.91590 | 19.64375 | 12.67790 |
| 8x1024 append, KV32K | 15.20965 | 39.95030 | 24.83975 |
| 2x256 append, KV128K | 3.92595 | 9.88885 | 6.40695 |
| Mixed q4 verify | 0.08750 | 0.16920 | 0.11295 |
| Mixed q8 verify | 0.08985 | 0.16900 | 0.11225 |

Bulk exchange improves the previous exchange by 35–38% on the append fixtures,
but is still 60–63% slower than redundant gathering. Mixed fixtures remain
25–29% slower. FA2-only times remain approximately equal. Therefore this version
also fails the kernel-performance gate and is not promoted to serving.

The redundant control in this expanded binary is itself slower than the earlier
KV-sharing binary's control. Cross-build timing differences must not be
presented as an isolated bulk-copy effect. The current experiment also does not
separately measure the remaining gather drains, paired barriers or compiler
resource effects, and does not implement true global-memory TMA multicast.

`kvshare_bulk_complete_20261013.tar.gz` preserves the complete raw results,
30,605 bytes, SHA256
`7f430fa39192015cf58cf08c7446dc55eac002dd619dc4a9f7e675e2b62007fa`.
All 54 raw timing-log hashes match the adjacent completed audit JSON.
