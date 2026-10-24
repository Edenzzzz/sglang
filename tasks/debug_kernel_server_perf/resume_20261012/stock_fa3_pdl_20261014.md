# A: stock FlashAttention3 primary with native FA2 PDL

This implements the next primary path rather than further changing the simplified FA3
mainloop. It is experimental, not promoted; the main task outline and TODO states are unchanged.

## Implementation

`stock_fa3_pdl_20261014.cu` wraps the existing FlashAttention3 own-scheduler kernel with
early dependent-grid release and optional device timestamps. Thread count and shared storage
come from the stock kernel's own traits. The actual attention computation is unchanged.
The adapter offers original stock launch, serial wrapper, PDL wrapper and deferred-combine
phases. Native FA2 runs between the PDL primary and split-KV combination, so combination does
not serialize the independent FA2 launch. Final stores are fenced before recording completion.

The prototype accepts compact FA3 requests, page-1 NHD bf16 KV, head dimension128 and1–32
uniform KV splits. Caller-owned dense page tables, partials and counters are reusable; the
run path performs no allocation or device-to-host shape read. Production integration and
handling other layouts remain unproven. The existing split-capable implementation remains
available and unchanged.

Working directory: `/dev/shm/stock_fa3_pdl_work_20261014`. FlashAttention source is a clean
snapshot of commit `e9515d5dee6ade134a33d6020d38d01ef0596996`, not the original repository's
modified working file. The user's vendor-tree changes were preserved.

## Build controls

The vendor build uses `--use_fast_math`, `-DNDEBUG`, extended SM90 MMA shapes and SM90 GDC.
The migrated native loader did not supply those flags. The initial stock build without them
reports WGMMA serialization across a function boundary; the matching-release build does not.
That is a compiler observation, not proof of the cause of the serving-performance gap.

A release-flags build of the existing native primary is included as a fresh GPU comparator,
so an algorithm change is not credited for compiler-option differences. Its local dedicated
entry changes9304→8784 static instructions and two→zero call instructions; registers remain168
and the stack8 bytes. The stock original and PDL wrapper both compile at168 registers and
zero stack. No local CPU build substitutes for the pod's rebuild or GPU validation.

## Validation staged

The inherited-lock hook is installed after `mix128_c24/BASE_2`, under controller PID766826.
It does not interrupt a timed serving run. The GPU suite first runs varlen/split/head-ratio
checks against fp32, graph replays with newly randomized Q, NaN-poisoned outputs/partials and
invalid-metadata guards. Coupled eager/graph gates require correct native-FA2 output and
useful FA2 completion during the large-prefill PDL primary.

Then80 timing cases cross five fixtures,112/full-device SM budgets, two reversed-order
repeats and four modes: compiler-matched native primary, original stock FA3, serial stock
wrapper and PDL stock wrapper. Every accepted timing requires fp32/replay/unfused checks.
The scalar-metadata wrapper and compiler options are held fixed within each comparison.

Initial source overlay:631871 bytes, SHA256
`fdf7e6cedf773283321f1908130dadfbb4393c35ec3c638ae26fa785323caf8b`.
The follow-up overlay adds the compiler-matched native control and updated source manifest:
50922 bytes, SHA256 `ab6c7d9e2a262751664106f42c2195a6f5c60d9c3e5c4b61215876d760a187a3`.
Both archives are retained beside this note. The run log will be
`/dev/shm/w/stock_fa3_pdl_midpoint_20261014.log`; hook state will be
`/dev/shm/w/b1_kernel_concurrency_20261014/stock_fa3_pdl_midpoint_status.json`.

At preparation, local compilation passes; GPU correctness, overlap and performance are
pending. This does not close A or justify starting EDF work.

## GPU gates passed; serving integration prepared

The midpoint began after the20th B serving arm. Three GPU unit tests pass, covering18
varlen/split/head-ratio/launch combinations, changing-input graph replays and invalid-shape
guards. All four coupled eager/graph gates pass, including useful native-FA2 overlap during
the stock primary. The80-case performance matrix is still running; no serving gain is claimed.

The opt-in serving adapter is implemented in `stock_fa3_serving_20261014.patch`, targeting a
new private tree. It selects the stock path only when the existing FA3-only plan proves no
KV splits. Split plans keep the existing native primary/reduction. It reuses per-step buffers
for page tables, LSE, scratch and counters; none is allocated inside the per-layer dispatch.
The serial/PDL mode controls native FA2's launch attributes, and both contiguous views and
scattered row subsets retain the existing output placement. A CPU test exercises actual
extracted dispatch code for all three stock phases, both row layouts and the native path.

Serving-gate PID853799 waits for controller766826 and the completed stock kernel matrix.
It will run five fresh arms: plain split, compiler-matched native primary, stock serial,
stock PDL with exact same-input layer-reference checks, and stock PDL without diagnostics.
The inherited generation-prefix gate remains weak; the layer-reference gate, including
fp32 adjudication for disputed heads, is therefore required separately.

Serving overlay:54858 bytes, SHA256
`6d903fbf0d22078138b7fc8097d563e11c4e8196ff3a28000ac27533e4cb876e`.
Log `/dev/shm/w/stock_fa3_serving_gate_20261014.log`. No timed serving comparison follows
automatically until these gates and the complete kernel matrix are inspected.

## Kernel matrix complete

All80 timings complete17:01:39, after the GPU unit/overlap gates. Two-repeat means at112 SMs:

| Fixture | Native, matched release flags | Stock serial | Stock PDL |
|---|---:|---:|---:|
| prefill2048 +8 verify | 4.23725 ms | 3.66130 ms | 3.65010 ms |
| append512 +46 verify | 2.31320 ms | 2.20440 ms | 2.19255 ms |
| FA3-only8x1024 | 8.54635 ms | 7.23115 ms | 7.24435 ms |
| split q32 +8 verify | 0.16410 ms | 0.12700 ms | 0.12640 ms |
| prefill8192 +24 verify | 16.58765 ms | 14.42665 ms | 14.41115 ms |

Stock computation, not overlap alone, explains most of this improvement: PDL and the two
serial stock controls are usually within about1%. The native release-flags control is itself
faster than the earlier non-release build; that benefit must not be attributed to switching
mainloops. At132 SMs, stock PDL takes6.68210 ms on FA3-only and13.39605 ms on prefill8192,
but append512 is slightly worse than112 SMs. No universal full-SM setting is promoted.
Complete scalar audits: `stock_fa3_pdl_progress_20261014.json`.

The serving adapter now handles the tested1–32 uniform splits, not just unsplit plans:
it runs the primary, independent native FA2, then stock combination. Plans above32 retain
the existing primary/reduction. The actual extracted CPU dispatch test checks ordering and
row placement for13 stock/native cases, including split combination after FA2. This is still
not a substitute for the queued real-attention reference gate.

Replacement serving overlay:52863 bytes, SHA256
`d5259bce4821ca5c781f065001fb7393aaccdcb43d4280d9a71237060de1accd`.
The queued gate has not started and its future private tree does not yet exist. No live
serving source was changed; the active B experiment remains on its original private tree.

The complete kernel archive is local and verified: `stock_fa3_pdl_complete_20261014.tar.gz`,
65784 bytes,170 checked file hashes, SHA256
`419f29cb8ce27cc198371ace64647a102ae188ea79a0e12a2c25c15e6f97d454`.
The midpoint hook was removed17:15:58. At17:58:24, controller766826 is live on the22nd B
arm (`mix128_c48/CHAIN`); serving-gate PID853799 is still waiting for its terminal success.

## Serving gates complete; matched pilot running

All five serving gates finished20:15:19 with CUDA graphs,12 requests each, no errors or
fallbacks, and3401 stock-primary calls in each stock arm. The diagnostic arm passes the
same-input attention-reference checks with fp32 adjudication. Generation-prefix comparison
is weaker than full output equivalence and is not presented as bitwise equivalence.

The new private tree `up_stock_fa3_ab/python` permits the native primary to use all132 SMs.
Fresh full-SM control/reference gates passed20:32. PID886518 now runs
`stock_fa3_serving_ab_20261014.sh`: MTP, unfused four-mode, split, release-native,
stock serial/PDL at112/132 SMs, and a repeated split control. This is nine timed arms,
without EDF, fused chain epilogues or diagnostic synchronization.

The frozen workload is48 agent programs/96 rounds at24 active programs in32–64K, alongside
2048 natural ShareGPT requests at8 requests/s. It retains the gap-free finite-cohort limits
described in `b1_kernel_concurrency_20261014.md`; it is a kernel-isolation pilot, not the
complete realistic workload/rate sweep. Whole MTP-to-four-mode gains must not be credited
to PDL; the four-mode arms isolate the executor comparison.

Pilot source overlay:45690 bytes, SHA256
`c4d080d68ebef721c69fd2aab96b9ea2c70bdc605c83c16cbeb2453fb71ed231`.
Log `/dev/shm/w/stock_fa3_serving_ab_20261014.log`. No successor is queued, and neither
A/B closure nor the required serving improvement is claimed before the results land.

## Serving pilot complete: no general promotion

All nine timed arms finished21:30:28 with96 agent rounds and2048 chat requests each.
Request/output-length fingerprints, session identities,24-program peak, graph execution
and intended backend counters pass. Stock arms execute29101–29301 primary calls.
No errors, fallbacks or EDF runtime. Complete scalar/execution results are in
`stock_fa3_serving_ab_progress_20261014.json`.

Relative to the mean of the two fresh plain-split controls:

| Candidate | Tenant | Output tok/s | Mean TTFT | Mean ITL | P99 ITL |
|---|---|---:|---:|---:|---:|
| Release-native | Agent | +0.89% | -1.55% | -0.96% | +1.16% |
| Release-native | Chat | +0.88% | -1.13% | -0.92% | +5.33% |
| Stock serial112 | Agent | -0.96% | +5.65% | -13.76% | -2.97% |
| Stock serial112 | Chat | -0.97% | +4.34% | +2.59% | +1.38% |
| Stock PDL112 | Agent | -0.74% | +4.48% | -9.44% | -0.26% |
| Stock PDL112 | Chat | -1.56% | +2.62% | +1.77% | +0.90% |
| Stock serial132 | Agent | -0.53% | +0.87% | -0.01% | +6.97% |
| Stock serial132 | Chat | -0.37% | +0.79% | +0.59% | +2.02% |
| Stock PDL132 | Agent | +0.27% | +1.69% | -6.09% | +0.14% |
| Stock PDL132 | Chat | -0.37% | +0.43% | +0.31% | +5.05% |

The split-repeat throughput spreads are0.053% agent/0.292% chat; mean-ITL spreads are
2.127%/0.270%. These are two-run observed spreads, not confidence intervals. Agent strict
draft acceptance varies0.816–0.880 across the kernel arms; do not attribute the agent ITL
differences purely to a faster attention kernel. The112-SM stock variants trade agent ITL
against TTFT and the chat tenant;132 SMs do not produce a general benefit either. Keep the
plain split as the serving control. The microbenchmark improvement has not translated into
the required serving win, and this single-point pilot does not establish other contexts.

The five112-SM and two132-SM correctness gates are archived locally in
`stock_fa3_serving_gates_complete_20261014.tar.gz`:188032 bytes,36 verified file hashes,
SHA256 `09469f2009473384129bb6dd34502108646e7ffc9438f057bbff9802d3f36f00`.

Next diagnostic: `stock_fa3_serving_profile_20261014.sh` reruns plain split and stock PDL112
with the identical finite workload, capturing80 CPU/GPU-profiler steps30 seconds after the
two-client barrier. This measures where the tested executor spends time, not another EDF
policy. Profiling perturbs latency; its serving rates are not additional performance samples.
Step compositions are not guaranteed identical across arms, so profile totals alone do not
establish a causal kernel speedup. Kineto's estimated occupancy is not hardware occupancy.

Complete compact pilot archive is local: `stock_fa3_serving_ab_compact_complete_20261014.tar.gz`,
622931 bytes,51 verified hashes, SHA256
`4f62b2c9ee09695bc3c8170754e5554fb517d744fb1643d4c7b164cb345a2d53`.
It includes all nine audits, server/driver logs and program-event records; full per-request
client outputs remain on the trial with their hashes recorded in the audits.
