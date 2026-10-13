# FA3 entry isolation experiment

The native FA2 bridge removes much of the FA2-family bandwidth gap, but long
prefills remain slower than the stock two-launch control. The FA3 primary still
uses a kernel that contains the inactive legacy FA2 body.

A dedicated FA3 entry now executes only the FA3 body, retaining its SM budget,
early PDL release, output-store waits and timestamp instrumentation. Selection is
opt-in with `FUSED_FA3_DEDICATED_PRIMARY=1` and applies only to the native-FA2 path.
The argument layout remains identical for a controlled comparison. Both variants
use the same force-inlined FA3 body; neither changes the attention arithmetic.

Local CUDA compilation and registration pass. Resource inspection of that single
build reports168 registers for both entries, with an8-byte stack frame for the
dedicated entry versus1,584 bytes for the combined entry. The preceding v3 build
reported1,536 bytes for the combined entry. These are compiled resource figures,
not proof of executed spills or a latency gain; GPU measurements are required.

Artifacts: `native_fa3_entry_20261013.patch`, the resource report, compile log and
`native_fa3_entry_overlay_20261013.tar.gz` (34,069 bytes), SHA256:
`e8f867d6d663ead28edf89851806a4ae098d32c6e7be8eabdc5d04242a03b1b9`.

`native_fa3_entry_ab_20261013.sh` is queued as PID493488 from19:08:48 +08:00,
behind the active three-range serving comparison PID487089. It uses a separate
`fi_native_pdl_v4` source tree and caches. Both entries must pass numerical and
useful-overlap gates before16 interleaved timing cases covering long prefill,
append, FA3-only and split-KV workloads. All timing cases require fp32, graph
replay and unfused-control checks. Existing running experiments and the main
checklist remain unchanged; this candidate is not promoted.

## Completed H200 comparison

Both numerical/useful-overlap gates and all 16 interleaved timing cases passed,
including fp32, CUDA-graph replay and unfused controls. The run completed at
20:40:26 +08:00 on 2026-10-13. Two-repeat mean times are:

| Fixture | Combined entry (ms) | Dedicated entry (ms) | Change | Unfused control (ms) |
|---|---:|---:|---:|---:|
| Prefill 2048 at KV64K + 8 verify rows | 6.04070 | 4.64970 | -23.03% | 4.05125 |
| Append 512 at KV48K + 46 verify rows | 2.58570 | 2.34775 | -9.20% | 2.27690 |
| FA3-only 8x1024 at KV32K | 12.12220 | 10.67335 | -11.95% | 7.45430 |
| Split-KV q32 + 8 verify rows | 0.20905 | 0.17020 | -18.58% | 0.98205 |

The dedicated entry is faster on all four measured fixtures, but still trails
the harness's unfused control on the three larger-query fixtures. The large
short-query advantage does not by itself establish a serving gain: the serving
control uses its own stock FA3 path and dispatcher. This is a candidate for
serving validation, not a replacement for that comparison.

`native_fa3_entry_complete_20261013.tar.gz` preserves the raw build, resources,
gates, timing logs and runner log: 13,254 bytes, SHA256
`8b213ee67606006f88c354f283a5f39bc9042e6a8c80901adfb84a5eaae2c9f3`.
All 16 raw timing-log hashes match the adjacent completed audit JSON.
