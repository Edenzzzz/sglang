# FA3 KV96 geometry experiment

The verified stock profile uses query tile128, KV tile96 and two stages. The
custom kernel used KV128, with an assertion excluding96 because the planner
aligned split chunks to128. This experimental patch admits96 and aligns chunks
to the least common multiple of128, the FA3 KV tile and the page size. For KV96
on page1 or page128, split starts therefore align to384 tokens. Disabled
splitting stays disabled; invalid dimensions and int32 overflow reject.

The alignment preserves the FA2-compatible128 boundary as well as FA3 tile and
page boundaries. The plan format is unchanged. It can change split counts on
short-query cases, so those results must not be described as geometry-only.
Existing KV128/page1-or128 rounding remains unchanged.

Both CUDA builds register locally. The host suite passes 4,869 rounding cases,
disabled/invalid/overflow checks, odd-end split seams, existing plan round trips,
and pair-partition/merge checks. This is not proof of GPU correctness.

`fa3_kv_geometry_20261014.sh` runs as PID 620988 using private source, Python and
separate KV96/KV128 cache directories. Sixteen gates cross both geometries with
both work distributions, covering useful PDL overlap, boundaries around96/128,
varied split-KV rows and persistent multi-work CTAs. Forty interleaved timing
cases then compare the same five mixed/FA3-heavy fixtures. Every case keeps
fp32, stock-control and changed-input graph checks. TMA and hardware clustering
are disabled in every arm; this is a gather-kernel experiment.

Archive: `fa3_kv_geometry_overlay_20261014.tar.gz`, 46,904 bytes, SHA256
`873ec6751187c954c1110e76706d4484f96e0afc335fc7b27241aa9c787f9589`.
The adjacent patch, host-test output and separate compile logs preserve the
implementation. GPU and serving outcomes remain pending; no default or main
task-document item is changed.

## V1 failure and correction under test

V1 stopped at its first GPU numerical gate with nonfinite FA3 output; FA2 output
remained finite. Removing FA2, using a context divisible by both96 and128, and
separating O from V shared storage did not fix it. Short-context tests passed at
KV1 but exposed errors after changing Q at KV2 and above, so no timing result
from V1 is accepted.

The actual defect found in the packed-Q gather is a dimension mix-up: its
column loop used `get<1>(TileShape_QKD)` (the KV tile width), not the Q head
dimension. KV128 hid this because both values were128. KV96 left32 Q columns
unwritten. V2 bounds that loop by `HEAD_DIM_QK` and corrects the diagnostic label.
Two CPU coverage checks now require all Q columns independently of the KV tile.

A conversion-lifetime hypothesis was explicitly disproved: the active merged
`utils.cuh` already returns owning storage. The initially inspected bundled
header was not the effective compiled header. That attempted change is reverted
and is not part of V2. Register-layout ownership checks pass for KV64/96/128;
host gather/shuffle mapping checks also pass for all three widths.

`fa3_kv_geometry_v2_20261014.sh` runs as PID 627313 with new source/cache paths,
the original 16 gates and 40 timing cases. The new archive is 60,754 bytes,
SHA256 `f3e151628d99f1ffad69fb4ee3dc184916192c12a3dca736f64475bf81c68e94`.
The failed run and diagnostics remain separate. GPU validation of the Q-copy
correction is required before accepting any KV96 performance result.

## V2 completed: correct, not faster

All 16 GPU gates and 40 timing cases passed, completing at 02:31:17 +08:00 on
2026-10-14. This includes changed-Q graph replays, poisoned padding, split-KV
merges, boundary cases around96/128 and persistent multi-work CTAs. The packed-Q
column correction removes the demonstrated failure.

Two-repeat means with the same round-robin-pair distribution:

| Fixture | KV128 (ms) | KV96 (ms) | KV96 change |
|---|---:|---:|---:|
| Fresh 512 + 46 verify | 2.341762 | 2.345090 | +0.14% |
| Prefill 2048 + eight verify | 4.758218 | 4.949816 | +4.03% |
| Append 512 + 46 verify | 3.239743 | 3.266536 | +0.83% |
| FA3-only 8x1024 | 9.535453 | 9.898555 | +3.81% |
| Varied odd queries/KV | 0.333297 | 0.346780 | +4.05% |

KV96 is not promoted. The work-distribution improvement remains the useful
candidate, so KV128 with that distribution proceeds to a separate three-range
serving comparison. Copying stock's KV geometry alone did not reproduce its
tensor utilization or close the gap.

`fa3_kv_geometry_v2_complete_20261014.tar.gz` preserves all raw results,
19,381 bytes, SHA256
`518b1034681f1348059239b1dfa5e71d130cef4024c9581739efd80db5c4cf44`.
All 40 timing-log hashes match the completed audit; all 16 gate logs pass.
