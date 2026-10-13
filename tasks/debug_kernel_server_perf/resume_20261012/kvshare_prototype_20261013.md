# B3 embedded DSMEM KV-sharing prototype, 2026-10-13

This is an implemented experimental FA3 path, not the earlier zero-load diagnostic
or the isolated gather microbenchmark. It passes the kernel gates but has not
passed serving gates and is not promoted. The main checklist remains unchanged.

## Implementation

The current custom FA3 mainloop gathers KV through arbitrary page-size-1 indices.
This variant retains that layout: two adjacent FA3 CTAs each gather half the head
dimension, then exchange the missing half through Hopper distributed shared
memory. It is cooperative software KV sharing, **not a TMA multicast instruction**.
The native FA2 executor still runs as the PDL dependent on the remaining SMs.

The planner distributes consecutive FA3 work pairs across two-CTA clusters before
constructing merge metadata. The device scheduler checks matching request/KV
offsets, head, extent and causal tile count; unmatched tails gather independently.
Each producer publishes its actual pipeline-stage index, so paired CTAs need not
have identical stage parity after unshared or skipped work. Ready/read-complete
cluster barriers order the exchange; a proxy fence precedes consumer visibility.

Cluster initialization and exit synchronize both CTAs. The existing output-barrier
arrival now addresses its own cluster rank rather than always rank zero. The
co-launch accepts this guarded two-CTA variant and retains `tma_store_wait<0>()`.
The warp-specialized FA3 body is forced inline to avoid crossing a function ABI
boundary while changing register allocations. Default serving sources are untouched.

## Validation and artifacts

- 1,032 host partition cases verify coverage, pair identity and odd tails; invalid
  cluster sizes reject, and reordered split/merge coordinates match the original.
- The complete pre-existing host-plan suite passes as well.
- Local CUDA compilation and operator registration pass with CUDA13.1 / Torch2.9.1.
  A remaining ptxas C7510 serialization warning is retained in the compile log;
  compilation is not evidence of GPU correctness or acceptable performance.
- The trial uses Torch2.14.1+cu130 and rebuilds from source; no local binary is deployed.
- Eight runner-parser checks cover actual sharing counters, bad outputs, inactive
  sharing, missing counters and accidental device printf in timing runs.

`kvshare_prototype_20261013.patch` contains the nine-file change relative to the
recovered remote baseline. `kvshare_source_overlay_20261013.tar.gz` contains the
exact sources and before/after SHA256 manifests (86,236 bytes), archive SHA256:
`50ac549166dfb2453428744bb77511d6fc542030ec1c8eb799af85cefb33ee55`.
The local build log is `kvshare_local_compile_20261013.log`; working source is
under `/dev/shm/kvshare_work_20261013`.

## Queued GPU test

`kvshare_kernel_20261013.sh` was queued as PID `412888` at 13:57:50 +08:00 behind
the active FA2 capped-auto validation PID `395036`. It verifies predecessor success
and baseline source hashes, takes the existing exclusive lock, then creates private
`fi_kvshare`, `up_kvshare`, `merged_kvshare` and `torch_ext_kvshare` paths.

Two numerical smoke gates first compare clustered redundant gathering with sharing
enabled. A device counter must be zero for the former and positive for the latter.
Only after both pass does the timing matrix run: eight fixtures, two repeats, three
arms (original co-launch, clustered redundant gather, clustered sharing), 48 cases.
Every timed case requires fp32 and graph-replay correctness plus a correct unfused
control; printf is disabled. Fixtures cover append contexts32/64/128K, odd query
tails, q4/q8 mixed verify, and FA2-only work. Kernel timeouts fail the run.

Comparison with the selected stock-FA3/native-FA2 serving split and real four-mode
serving validation are still required. Neither B3 nor the full goal
is complete merely because this prototype builds.

## GPU outcome

Both smoke gates passed on H200: the clustered redundant arm reported zero shared
items and the sharing arm reported one on CTA0. All 48 timing cases subsequently
passed fp32, graph-replay and unfused-control gates, finishing at 14:18:10 +08:00.
This includes the odd query tail, long append fixtures, q4/q8 mixed work and FA2-only
control. It is actual embedded kernel execution, not a zero-KV-load diagnostic.

The sharing path regresses performance substantially. For example, the first
4x512-query / KV65536 repeat takes18.2148 ms with sharing versus5.9367 ms with
clustered redundant gathering (original co-launch6.1796 ms in the second repeat).
Do not promote it or claim the earlier diagnostic headroom was realized. The
per-load drain, peer copies and ready/read-complete synchronization are candidates
for profiling; the current data does not isolate their individual costs.

The existing FA2 executor already places its PDL dependency wait after its work;
that hypothesis did not identify a bug, and no wait-position change was made.
Completed evidence uses `kvshare_kernel_complete_20261013`. A serving-output test
is still needed before claiming end-to-end integration, even though it is not a
performance candidate in its current form.
