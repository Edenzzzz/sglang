# B3 paged TMA multicast prototype

This is a new global-memory TMA path, not either DSMEM-exchange implementation.
It builds on the dedicated FA3 primary and compact native-FA2 PDL integration.
The experimental contract is bf16, head dimension 128, dense NHD physical cache
strides, 128-token pages, two-CTA clusters and page-aligned FA3 split starts.
Unsupported launch modes and page layouts reject rather than silently falling
back. Default serving sources and the main task document remain unchanged.

## Kernel and addressing

K/V tensor maps address the physical cache pool. Each logical tile looks up its
raw physical page before issuing `cp.async.bulk.tensor.3d...multicast::cluster`.
An eligible pair must have matching request, KV head, range and causal tile
count. The two producer leaders exchange their current pipeline-stage indices;
matching stages use mask 3 and one issuer. Otherwise each CTA loads independently
with its own single-bit mask. Unmatched work also remains independent.

The ready barrier has two arrivals, not 256. Its two metadata slots prevent a
fast producer from overwriting the preceding stage index before its peer has
consumed it. Each participating work item performs an even number of K/V loads,
so ready-barrier parity returns to its starting phase across work items.

Full tiles attach their transaction bytes directly to the existing pipeline's
full barrier; producer arrival counts and consumer release ordering remain in
force. There is no per-tile gather drain or received-tile wait before committing
a full tile. Partial KV tails use a separate transaction barrier, explicitly
zero invalid shared-memory rows, fence visibility, and synchronize the producers
before publishing the tile. This prevents poisoned padding from entering the
attention MMA. Cluster initialization/exit, own-rank output-barrier signaling
and `tma_store_wait<0>()` are retained.

The page-128 preparation exposed a page-1 assumption in the unified scheduler:
it added a token-valued split start directly to a page indptr. The prototype
now carries page size into the scheduler and adds `kv_start / page_size`.
Both its gather control and TMA variants include this correction; it is not
applied retroactively to the running page-1 serving comparison.

## Local evidence

Both the TMA and gather-control CUDA builds register successfully. The TMA build
contains an `sm_90a` cubin and generated tensor-multicast PTX. These are codegen
checks, not proof of successful GPU execution or performance. The generic-PTX
multicast advisory and a WGMMA serialization warning remain in the compile log.

Four CPU fixture tests pass, including page permutation, poisoned padding,
reference finiteness, a single-token oracle and deterministic seeds. Thirteen
page-alignment cases, the 1,032-case pair-partition coverage test, odd-tail/merge
checks and the existing host-plan suite pass. The GPU driver compares against
chunked fp32 attention, poisons output/partial buffers, changes Q values across
graph replays, and checks the unfused control separately.

## Queued GPU experiment

`tma_multicast_kernel_20261013.sh` is queued as PID 561859 after serving PID
546077. It waits for terminal status and exclusive access without restarting
that run. It verifies source hashes and creates private `fi_tma_multicast` and
`up_tma_multicast` trees plus separate gather/TMA extension caches.

Nine smoke gates cover gather, unicast TMA and multicast TMA with useful PDL
overlap, a poisoned partial page, and varied split-KV requests. Actual device
counters must distinguish unicast and multicast; the partial-page gate must
execute tail handling. Only then do 48 graph-timing cases run: eight fixtures,
three arms and two reversed-order repeats. Fixtures include fresh and appended
prefills, 32/64/128K contexts, q4/q8 native verification, odd extents and FA2-only
work. Timing prohibits device tracing. TMA versus multicast uses one binary and
the same paired schedule; the gather arm uses the original one-CTA geometry.

Archive: `tma_multicast_source_overlay_20261013.tar.gz`, 110,255 bytes, SHA256
`0c2b3edb5437bb98714494964b54b88d777498edd084dedeed635f776d834562`.
The adjacent patch is relative to the combined native-PDL candidate and includes
the new harness and tests. No multicast speedup or goal completion was inferred
from local compilation or host tests.

## Completed kernel experiment

All nine smoke gates and 48 timing cases completed at 23:13:39 +08:00 on
2026-10-13. The shared gate executes 58 multicast tiles on CTA0, while the
unicast control reports zero. Poisoned-tail handling, shuffled physical pages,
changed-Q graph replays, fp32 output checks and useful PDL overlap pass. The raw
logs also confirm that the named split/varied cases actually use multiple FA3
splits and nonempty merge work.

Two-repeat mean graph times, in milliseconds:

| Fixture | Gather | Unicast TMA | Multicast TMA |
|---|---:|---:|---:|
| Fresh 512 + mixed verify | 0.059482 | 0.057943 | 0.058779 |
| Prefill 2048 at KV64K + 8 verify | 4.738762 | 4.852550 | 4.932556 |
| Append 512 at KV48K + 46 verify | 3.197592 | 3.223039 | 3.217142 |
| FA3-only 8x1024 at KV32K | 11.005545 | 9.676730 | 9.658576 |
| FA3-only 2x256 at KV128K | 2.313225 | 2.402105 | 2.415525 |
| Split q32 at KV64K + 8 verify | 0.184291 | 0.190980 | 0.192855 |
| Varied odd queries/KV lengths | 0.456109 | 0.355876 | 0.356562 |

Unicast TMA improves the 8x1024 and varied-odd fixtures by 12.07% and 21.98%,
respectively, but regresses other attention-heavy fixtures by roughly 1–4%.
Multicast itself is within about -0.2% to +1.7% of unicast TMA: no clear benefit.
FA2-only timings are unchanged. There is no basis for globally promoting this
configuration or claiming that the requested serving target has been reached.

The native FA2 wrapper in this harness is graph-planned in every arm. Earlier
page-1 harness runs did not always use that planning mode, so cross-run timing
differences must not be attributed solely to page size. The completed audit
also retains the separate stock-wrapper control times.

`tma_multicast_kernel_complete_20261013.tar.gz` is 35,059 bytes, SHA256
`cdcaf1676e070626073466111f1d755bc8a312c88aa90e9a7b4f387214bba07d`.
All 48 timing-log hashes and all nine gate logs are preserved and checked.
Page-128 real-serving validation is recorded separately in
`tma_paged_serving_20261013.md`.
