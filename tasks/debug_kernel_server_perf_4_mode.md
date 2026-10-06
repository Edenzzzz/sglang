# Current goal

**Make the fused FA3+FA2 kernel reach parity with each standalone SOTA kernel per
role, then show scheduling flexibility on top.** Parity first is the bar: if FA3
matches flash-attention (we add packed GQA and a dynamic split-KV threshold, so it
should beat it) and FA2 matches the stock persistent kernel, then the fused
design's scheduling freedom is pure upside. Target: >=30% TTFT/ITL and >=10%
throughput on mixed chat + tracelab.

**Status 2026-10-06 (end of day): the kernel splits into two executors, decided by
measurement.** Co-residency constrains the single fused kernel twice, and neither
constraint is a cost of the work:

- **registers** — 384 threads at 1 CTA/SM caps the allocation at 170/thread and the two
  roles share it. Upstream FA3 spills 33 676 B in the fused kernel against **52 B**
  alone in the identical template; outlining FA2 takes FA3 to 64 B but moves 17 500 B
  onto FA2; halving FA2's O accumulator reaches **88 B** isolated but only 1 504 B
  fused. One `__global__` can give *either* role native registers, never both.
- **shared memory** — FA3's 98 352 B leaves 134 048 B of the 232 448 B limit; the FA2
  role's tile-16 configuration at 8 warps needs 135 168 B. Over by 1 120 bytes.

Both executors already exist as measured probes at native quality: flash-attention's
FA3 at 168 regs / 52 B spill, the persistent LPT at **255 regs / 0 B**. What is missing
is the plan wiring and a PDL launch, not kernels.

**The scheduling contribution is unaffected**, because it lives in the plan: 4-mode
tile-class assignment, per-queue split length (now sized by each queue's executor
geometry), the smem-handoff JIT reduction (intra-role — a split row's chunks are all one
task's items), and the spec tree epilogue. Only cross-role backfill *inside a CTA* is
given up, which is itself what causes the 33 676 B; PDL recovers part of it temporally.
Cross-role multicast is **not** given up: the 1.587x DSMEM result is q-tiles of one
prefill row re-gathering the same KV, all FA3-role items.

**Where the numbers stand.** Fixture 2: 155.9 us fused against **72.3 us** for
FA2-on-everything, both numerics-correct from one build — a **2.16x regression**, worse
than the 1.83x on the 16 823-token-KV fixture, so the shorter workload is the harder
case. Nothing is near the >=30% TTFT/ITL and >=10% throughput bar.

**The upstream FA3 port is close but not yet correct.** Three faults found, all the same
shape — upstream carries a per-item or per-subset field and was handed the whole-batch
aggregate: per-item `num_splits` in `split_idx`'s high bits; `cu_seqlens_q`/`seqused_k`
read from per-work-item arrays (fixing this took fixture 2 from 439/512 unwritten to
**0/512**); and the combine pass walking every request, which overwrote the FA2
executor's correct rows (an FA2 decode row went 5.8e-4 → 0.155). The third fix is
written and under test.

### RETRACTED: FA3's cost is not fixed, and there is no overlap headroom — 2026-10-06

SM-budget sweep on `qo:512,512,100,1`, concurrent streams, every arm PASSing:

| FA3 SM budget | fa3 half | split total |
|---|---|---|
| 16 | 223.7 us | 234.7 |
| 32 | 119.4 us | 127.3 |
| 64 | 74.9 us | 112.0 |
| 96 | 60.0 us | 115.1 |
| 132 uncapped | 45.9 us | 111-112 |
| **sequential, no overlap** | **45.9 us** | **80.7** |
| baseline, FA2 all | - | 94.0 |

**Two of my claims are wrong and are withdrawn.**

1. *"FA3's cost is nearly flat in work, so most of it is fixed launch and handshake
   overhead."* No. fa3_half scales near-inversely with the SM budget, so FA3 uses its SMs
   productively. The flatness I saw (38.2 us at ~28 tiles, 45.2 at ~64) was **parallelism**:
   at 132 SMs with fewer than ~132 tiles, extra tiles are nearly free because they run
   concurrently. I fitted a cost floor to two points on a parallel-scaling curve -- the
   same error I had just flagged as a risk before running the sweep.
2. *"The halves are independent, so overlap floors at max(45.9, 33.5) ~ 45 us, about 2.1x."*
   Withdrawn. Overlap requires at least one kernel to leave SMs idle, and neither does.
   Concurrency loses at every budget, 112-235 us against 80.7 sequential, because splitting
   the machine makes each kernel proportionally slower. **Sequential is optimal here**, and
   the 1.19x is the result rather than a stepping stone.

**What the sweep does establish, and it is the number the goal asked for.** Decomposing the
baseline: FA2 does all four rows in 94.0 us and the two FA2-only rows in 33.5 us, so FA2 on
the prefill rows costs about **60.5 us against FA3's 45.9 us**. FA3 is **1.32x faster than
FA2 on its own tile class**, which is per-role parity met and exceeded -- and it is the
mechanism behind the 1.19x end-to-end split win.

**Consequence for the design.** The SM split is not a throughput knob on these batch sizes;
it is only a *latency policy* knob, which is where EDF plugs in -- capping FA3 trades prefill
throughput for decode latency, and the sweep quantifies the exchange rate (budget 64 costs
FA3 1.63x for 68 SMs handed to decode). Rung 1 of the ladder is therefore not "add
admission control for overlap" but "decide the cap from predicted deadlines", and rungs 2-4
(smem-handoff reduction, DSMEM multicast, spec epilogue) are unaffected.

### TWO EXECUTORS BEAT FA2-ON-EVERYTHING, AND PASS — 2026-10-06

sglang `d9be55b53`, flashinfer `c035c23`. Baseline and split measured through the **same** code
path, build and fixture: `n_fa3:0` gives the FA3 executor no requests so every row goes to
FA2, which is FA2-on-everything by construction rather than a separate control arm.

    TWO_EXEC=1 TWO_EXEC_OVERLAP=0 FA3_UPSTREAM=1 TIME_ITERS=50 UNIFIED=1 SPLITS=auto \
      FLASHINFER_SM_PARTITION=1 FLASHINFER_FA3_SM_SPLIT_CLUSTERS=124 \
      FIXTURE='<shape> n_fa3:{0|2}' python tests/test_fused_server_batch.py

| batch | baseline, FA2 all | split, sequential | fa3 half | fa2 half | |
|---|---|---|---|---|---|
| `qo:72,439,100,1` 612 tok | 72.2 us | 72.5 us | 38.2 | 33.8 | parity |
| `qo:512,512,100,1` 1125 tok | **94.2 us** | **79.5 us** | 45.2 | 33.0 | **1.19x** |

Every arm PASSes: rel err 1.6e-3 to 3.7e-3 across all rows, 0 unwritten, both FA3 rows and
both FA2 tile classes exercised.

**This is the first measured win in the project**, and it is sequential -- no overlap, no
concurrency, no admission control. The mechanism is the flat-cost observation: FA3 went
38.2 -> 45.2 us for 2.3x the prefill work, so its fixed floor amortises and it overtakes FA2
on prefill-heavy batches. On the light batch the floor exactly eats the advantage, hence
parity.

**Headroom from here.** The halves are independent, so with working overlap the floor is
`max(45.2, 33.0) = 45.2` us against the 94.2 us baseline -- about **2.1x** -- rather than
the 79.5 us sum. Naive concurrent streams do NOT get there (111.3 us, worse than
sequential) because two full persistent grids contend for SMs; that needs device-side
admission control, which is rung 1.

**Why the fused kernel could never show this.** Its 2.16x regression was co-residency:
either role could have its native register allocation but not both (33 676 B of FA3 spill
against 52 B alone), and its shared memory exceeded the opt-in limit by 1 120 B. The split
removes both, and the baseline is the FA2 kernel itself, so the split hands it strictly less
work. The regression could not survive the split, and did not.

### First two-executor numbers — 2026-10-06

Two executors, one plan: flash-attention's FA3 on its own `VarlenDynamicPersistentTileScheduler`
over the wgmma requests, flashinfer's persistent LPT unmodified over the rest.

| arm | total | fa3 half | fa2 half |
|---|---|---|---|
| rung 0, sequential | 94.5 us | 32.0 | 70.5 |
| rung 1, concurrent streams | **112.5 us** | 32.3 | 72.1 |
| baseline, FA2 over ALL rows | **72.3 us** | - | - |

**The split measured here is not the one the router would pick.** `fa2_half` is 70.5 us,
essentially the whole baseline, because the harness's `n_fa3` defaults to 1: FA3 got only
row 0 (72 tokens) while FA2 kept the 439-token row. The actual routing sends both prefill
rows to wgmma (packed 504 and 3073, both over the 128-row threshold). Rerunning with
`n_fa3:2`.

Two findings that survive the wrong split, though:

**1. Concurrent streams are WORSE than sequential, 112.5 against 94.5.** Both executors
launch full persistent grids, so on separate streams they contend for SMs instead of
overlapping -- each one's CTAs occupy SMs the other wants, and neither can make progress at
its planned width. This refutes the reasoning I used to put streams ahead of PDL on the
ladder: I argued that with no data dependency, concurrency strictly dominates PDL's
serialised ramp. That ignored SM contention, which is the whole reason the fused kernel
needed an `sm_split` in the first place. Rung 1 needs **admission control** -- a device-side
cap on how many SMs each executor admits -- or PDL ordering, not naive concurrency.

**2. FA3's fixed cost looks large at this scale.** 32 us for a 504-packed-row slice, where
FA2 does the same row inside a 72.3 us whole-batch pass. Two tiles of work should not cost
32 us; that points at the persistent grid's launch and scheduler-handshake overhead
dominating when the queue is nearly empty. Worth isolating before reading any parity
number, because it will swamp small batches either way.

## Tasks, in priority order — 2026-10-06

1. [K/V pipelining, FA2](#fa2-kv-pipelining-measured-flat--2026-10-06) -- **not started**: the knob swept was register tiling, not pipeline depth
2. [Smem-handoff JIT split-KV reduction, both executors](#kernel-optimisations--2026-10-06)
3. [Context length where attention is >50% of e2e](#workload-and-measurement-setup--2026-10-06)
4. [TraceLab mixed scenarios instead of random](#workload-and-measurement-setup--2026-10-06)
5. [Long-context A/B on the real workload](#server-integration--2026-10-05)
6. [4-mode MTP serving run](#server-integration--2026-10-05)
7. [DSMEM / KV multicast](#kernel-optimisations--2026-10-06)
8. [Fused spec tree-build epilogue](#kernel-optimisations--2026-10-06)
9. [Perf model and EDF vs unfused](#schedule-exploration--2026-10-05)
10. [Multi-node PD colocation, four points](../tasks/pd_colocation_vs_disagg.md) -- **design doc written**, experiments pending

### Long-KV is 1604x off roofline, and our split count is being IGNORED — 2026-10-06

`qo:8,8,100,1 prefix:16384,16384` on the two-executor base, FA3 half only:

| arm | num_splits passed | fa3 half |
|---|---|---|
| splits pinned to 1 | 1 | 17 452 us |
| splits from upstream's heuristic | **29** | **17 358 us** |

**The split count changed and the time did not.** So pinning `num_splits_dynamic` to 1 was a
real bug and fixing it was right, but it is *not* what makes long KV slow. Retracting the
implication in the previous commit that it would be.

Two numbers frame what is left:

    work tiles = 1 m_block x 4 kv_heads x 2 rows = 8        (on a 132-SM GPU)
    ~8.6 GFLOP total -> ~10.8 us at roofline; measured 17 358 us = 1604x off
    even granting only 8 CTAs of serial KV: ~270 us expected, still 64x off

So there are two separate problems. **Under-parallelisation**: 8 work tiles cannot fill 132
SMs, which is exactly what split-KV exists to fix. **And something pathological on top**,
since 8-CTA serialisation only accounts for 270 us of the 17 358.

**PREDICTION OUTCOME: UNTESTED, not refuted -- the test arms are invalid.** Both arms
(`sh_rs0`, `sh_rs1`, `drivers/ab_fa3_traits.sh`) ran at `num_splits = 29` and returned
**garbage**: row 2 rel err 0.164/0.200 and row 3 rel err **1.334/1.287**. The cause is the
latent bug this same entry already identified below -- `partial_o` is sized
`total_q * head_dim * num_heads` with **no `num_splits` factor**, so the moment splits take
effect the epilogue writes past the buffer. It predicted exactly this and the prediction held.

The timings (`sh_rs0` 20 933 us, `sh_rs1` 20 691 us, 1.2% apart) are from a kernel writing out
of bounds and are not usable for anything, so they neither confirm nor refute the prediction
below. Prerequisite before this fixture can be measured at all: **size `partial_o` and
`partial_lse` with the `num_splits` factor.** Until then any arm on a fixture where
`num_splits > 1` is invalid, which also retroactively covers any earlier number taken on this
fixture with the heuristic enabled.

(For the record, the expectation after the TraceLab result would be that the prediction fails:
`rs1` moved that batch by 2.3%, so there is no 31x for it to explain here either.)

**PREDICTION as recorded 2026-10-06 before the measurement.** The "something
pathological on top" here should be the same `MmaPV_is_RS=false` fault found on the TraceLab
batch -- a mainloop that serialises softmax against the PV mma stalls per KV block, so the
penalty scales with KV length and is worst exactly where this entry measures it (16384 KV
against 8 q rows). If that is right, the `rs1` arm should cut this fixture's 17 358 us by
roughly the same factor it cuts the TraceLab batch, and the residual should land near the
~270 us that 8-CTA serialisation alone predicts. If the `rs1` arm leaves this fixture
unchanged, the two slowdowns are independent and this entry still has its own unexplained
cause -- which would also mean the SM-budget evidence was over-generalised from one shape.

**Evidence that our split count never reaches the kernel.** The harness's `partial_o`
scratch is `total_q * head_dim * num_heads` floats, with **no `num_splits` factor** — 419 328
against the 12 160 512 the epilogue would need at 29 splits, a 29x shortfall. If the epilogue
were taking its `is_split` branch it would be writing far past that buffer, and the arm
would be corrupt. It PASSES at rel err 2.2e-4. The only consistent reading is that upstream
is still running `num_splits == 1` internally and writing final O directly, i.e.
`num_splits_dynamic_ptr` is not being honoured on this path.

Prime suspect: `VarlenDynamicPersistentTileScheduler` is instantiated with
`Prepared=false`, and `TileSchedulerArguments` also carries `num_m_blocks_ptr`, which that
path may require before it will consult the dynamic split pointer.

**Next, in order:** (1) print the split_idx the kernel actually sees for one work tile to
confirm the pointer is ignored; (2) size `partial_o` with the `num_splits` factor regardless,
since the current buffer is a latent heap overflow the moment splits do take effect;
(3) only then re-measure long KV.

### FA3 is 31x SLOWER than FA2 at realistic context, and the 1.21x does not generalise — 2026-10-06

TraceLab-shaped batch at one H200's token budget (1.58M KV, 64-100K contexts, 2048-token
appends off the 3812-token mean, 18 decode rows, chat + math co-tenants).

**RETRACTED: "all rows correct in every arm (rel err 2.4e-4 to 4.2e-3)" was an artifact of
output truncation, and the correctness gate was never actually met.** The driver piped the
harness through `... | head -12`. With 22 rows that budget is consumed by the header, `[nsplit]`,
three `[time]` lines and the first **six** `rel err` lines -- so rows 6-21 and the trailing
`FAILURES:` summary were cut off and never seen. The range I quoted, 2.4e-4 to 4.2e-3, is
exactly the range of rows 0-5. I reported the visible rows as if they were all of them.

Re-run on the rebuilt pod with the full log kept, the FA2-on-everything control gives:

| | value |
|---|---|
| total | **15 197.3 us** (old pod: 15 301 -- reproduces within 1%) |
| fa2_half | 15 204.0 us |
| unwritten rows | 0 / 5650 |
| rows 0-5 rel err | 0.00037 - 0.00483 (what `head -12` used to show) |
| rows 6-20 rel err | 0.0049 - 0.0177 (never previously displayed) |
| **row 21** | **0.02618 -- FAIL against `TOL=0.02`** |

So the *baseline itself* fails a row on this fixture, and every "all rows PASS" gate I recorded
for the page-128 arms is unsupported: I never saw the rows that fail. The timings stand as
timings -- and the control reproducing 15.2 ms against 15.3 ms says the rebuilt pod is
comparable -- but the validity criterion I claimed to be enforcing was not being enforced.

**Unexplained and under investigation: the error grows monotonically with ROW INDEX, not with
KV length.** The decode rows with identical `kv=65537` and `qo=1` give 0.0088, 0.0101, 0.0113,
0.0138, 0.0169, 0.0164, 0.0177 at rows 7, 8, 10, 13, 16, 17, 20. Same shape, same work, error
rising with position in the batch. That is systematic, not bf16 noise, so "the tolerance is too
tight for 100K-context decode" is a hypothesis and not yet the answer. This is the unmodified
persistent LPT kernel -- the baseline -- so whatever it is predates every change in this
entry.

| arm | page | total | fa3 half | fa2 half |
|---|---|---|---|---|
| baseline, FA2 on everything | 1 | **15 364 us** | 7.0 | 15 301 |
| split, appends to FA3, cp.async | 1 | **476 581 us** | **477 949** | 1 238 |
| baseline, FA2 on everything | 128 | **15 301 us** | — | 15 470 |
| split, appends to FA3, cp.async | 128 | **480 459 us** | 480 012 | — |
| split, appends to FA3, **TMA** | 128 | **496 978 us** | **496 028** | — |

**FA3 takes 480 ms for work FA2 does in 15.3 ms — 31x slower.** This part stands, and it is
structural, not a tuning gap.

**RETRACTED: the cause is not `page_size = 1`, and not the gather.** The previous version of
this entry claimed the non-TMA paged path's per-token index lookup was the root cause, and
predicted that `PagedKVNonTMA=false` at `page_size=128` would fix it. Both halves of that
prediction are now measured and both are wrong:

- TMA paged at 128-token pages is **slower**, not faster: 496 ms against cp.async's 480 ms.
  One genuine caveat does apply: upstream's tuner picks `kBlockN = 176` when
  `paged_kv_non_TMA` is false and 128 when it is true (`tile_size.h:32`), so the TMA arm
  ran at 128 where upstream would use 176 -- it is off upstream's tune for that path. We
  cannot simply follow it, because TMA paged also requires `page_size % kBlockN == 0`
  (`mainloop_fwd_sm90:537`), and 176-token pages are not a sane KV-cache granularity. So
  upstream's TMA tuning implicitly assumes unpaged KV, and a paged TMA path is constrained
  to a block size upstream does not tune for. Worth remembering before reading much into
  the TMA-vs-cp.async gap either way.

  (I briefly suspected this arm was invalid because the scheduler hardcodes
  `NumProducerThreads = 128` where upstream's expression can give 32 on a TMA path. It
  cannot give 32 here: `Use_TMA_Q = !PackGQA` (`mainloop_fwd_sm90:56`) and we run
  PackGQA=true, so the single-producer-warp form is unreachable and 128 is correct on both
  paths. The TMA number stands.)
- Coarser paging does nothing for cp.async either (480 ms at page 128 vs 477 ms at page 1), so
  gather granularity is not what FA3 is spending its time on.
- Coarser paging costs the FA2 baseline nothing (15 301 us at page 128 vs 15 364 us at page 1),
  so the fixture is not distorting the comparison.

The gather hypothesis was built on the fact that the lookup count scales with KV (512 on the
old fixtures, 100 352 here) — a real correlation that turned out not to be causal.

**RETRACTED: the roofline figures in the previous version were wrong.** It read "FA3 is 155x
off on 1.23 TFLOP, FA2 is 5x off". The 1.23 TFLOP omitted the QO-head factor: every one of the
28 QO heads attends the whole KV, so the two append rows are

    2 * 2 * 28 heads * q * kv * 128   =  4.93 TFLOP   ->  12.3 ms at 400 TFLOP/s

Corrected: **FA2 at 15.3 ms is 1.2x off roofline** — essentially at peak, and that includes the
other 20 rows — while **FA3 at 480 ms is 39x off**. The direction of the finding is unchanged
but the magnitudes were overstated, and the important correction is about FA2: it is not "5x
off" with headroom to chase, it is already at the roofline on this workload. Any FA3-side win
here has to come from FA3's own 39x, not from out-running FA2.

**This retraction stands for the 1.21x as well.** That number came from rows at `kv=512`. It
does not survive to the target workload. The honest statement remains: the split wins on
short-KV batches and loses badly on long-KV ones.

**What the measurements leave as candidates.** FA3's cost is superlinear in KV relative to its
own short-KV behaviour, with the per-tile work identical (`num_splits = 1` in every arm above,
correctly — 2048-token appends give 448 m-blocks against the 106 threshold). So either FA3 is
under-parallelised on this shape, or it does pathologically more work per tile.

**ANSWERED: FA3 is latency-bound in the mainloop.** The `FA3_SM_BUDGET` sweep on this exact
fixture (page 128, cp.async, 2 iters, all rows PASS):

| SM budget | fa3 half | budget x time |
|---|---|---|
| 16 | 3 826 892 us | 61 230 |
| 32 | 1 918 061 us | 61 378 |
| 64 | 950 664 us | 60 843 |
| 96 | 634 346 us | 60 897 |
| 132 | **480 467 us** | 63 422 |

`budget x time` is constant to within 4% across an 8x range, so throughput is **perfectly
linear in SM count all the way to 132**. That kills both of the candidates above:

- Not under-parallelised. Adding SMs keeps paying off linearly; the 896 work tiles reach the
  machine fine, and there is no load-imbalance or tail effect to recover.
- Not bandwidth-bound, and not doing more work per tile. At 132 SMs FA3 achieves
  **10.3 TFLOP/s and 80 GB/s** of KV traffic, against FA2's **322 TFLOP/s and 2 518 GB/s** on
  the identical 4.93 TFLOP / 39 GB of KV. Nothing is saturated: 80 GB/s is 1.7% of HBM and
  10.3 TFLOP/s is 3% of peak. The *work* is the same, the *rate* is 31x apart.

Perfect linear SM scaling with nothing saturated is the signature of a **memory-latency-bound**
loop — each SM sits waiting on KV arrivals instead of overlapping them with the wgmma, and more
SMs simply buys more independent waiting. FA2's persistent kernel hides the same latency and
lands at 80% of roofline.

So the defect is in the FA3 mainloop's load/compute overlap on long KV, which points at
pipeline depth: we instantiate with `UP_STAGES=2`, and 2 stages of 128-token KV blocks is
~0.5 MB of in-flight KV per CTA, far too little to cover HBM latency when the walk is 657
blocks long. On the old `kv=512` fixtures the whole KV fit in a handful of blocks, so depth
never mattered — which is also why the 1.21x looked real there.

**Pipeline depth: 3 stages is illegal at RS=0 and fits EXACTLY at RS=1.** `UP_STAGES=3` fails
at launch -- `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize)` returns `invalid argument`
(`fa3_upstream.cuh:683`). Measured smem per configuration (`tests/fa3_port/compile_check.sh`
machinery, `K::SharedStorageSize`, limit 232 448 B):

| | stages=2 | stages=3 |
|---|---|---|
| `MmaPV_is_RS=0` | 199 680 | 265 216 -- **over by 32 768** |
| `MmaPV_is_RS=1` | 166 912 | **232 448 -- fits to the byte** |

RS=1 saves exactly 32 768 B, which is one 128x128 bf16 P tile -- direct confirmation that the
trait removes P's smem round-trip rather than merely renaming something. And the saving is
exactly the amount that made stages=3 overflow, so the trait also legalises a 50% deeper KV
pipeline, landing precisely at the opt-in limit.

**MEASURED: the deeper pipeline does not pay.** `rs1s3` gives **472 890 us against `rs1`'s
465 577 us -- 1.6% SLOWER**. So stages=3 is legal at RS=1 and is not worth enabling. The
"fits to the byte" result was a real and tidy smem fact that turned out to predict nothing
about performance, which is the same error as the root-cause claim above: I was reading
elegance as evidence. Depth is now ruled out by measurement rather than by the earlier
inference.

This qualifies the claim made a moment earlier in this entry that depth "is not the lever": it
was not a lever *at RS=0*, where 3 stages cannot fit. Upstream uses 2 unconditionally on Sm90
(`flash_fwd_launch_template.h:47`), so stages=3 is beyond upstream's own configuration and has
to be measured on its own merits, not assumed good. The `rs.sh` A/B includes it as a fourth arm.

**RETRACTED 2026-10-06, by the measurement below: `MmaPV_is_RS` is NOT the cause.** With the
reproduction gate passing (`rs0` = 476 684 us against the old pod's 480 ms), turning the trait
to upstream's value gives **465 577 us -- a 2.3% improvement, not a 31x one**:

| arm | fa3 half | vs rs0 |
|---|---|---|
| `rs0` (RS=0, stages 2, LPT=1) -- the old configuration | 476 684 us | reproduces |
| `rs1` (RS=1, stages 2, LPT=0) -- upstream's traits | **465 577 us** | **-2.3%** |

**Where the reasoning went wrong.** From perfect SM scaling plus low utilisation I inferred a
latency-bound mainloop, then searched upstream's traits for one that could serialise softmax
against the PV mma, found a real deviation, and treated it as the explanation. The smem
measurement that followed (32 768 B freed, exactly one P tile) confirmed that the trait
*changes storage* -- and I read it as confirming the trait *explains the time*. It never did:
the two claims are independent, and only the second was the root-cause claim. A mechanism being
real and a mechanism being load-bearing are different facts, and the smem delta only established
the first.

The trait change is still correct and is kept -- it matches upstream, costs no registers, frees
32 KB, and legalises stages=3. It is a 2.3% optimisation, not a root cause.

**Consolidated: every trait hypothesis lands within 2.4% of the others.**

| arm | fa3 half | vs rs0 |
|---|---|---|
| `rs0` RS=0 stages2 LPT=1 (old) | 476 684 us | - |
| `rs1` RS=1 stages2 LPT=0 | 465 577 us | -2.3% |
| `rs1s3` RS=1 stages3 LPT=0 | 472 889 us | -0.8% |
| `rs1l1` RS=1 stages2 LPT=1 | 472 865 us | -0.8% |

Against a 31x gap, a 2.4% spread means the traits are simply not where the time goes. Keep
`rs1` (upstream's values, free, frees smem), drop stages=3. `LPT=0` -- upstream's value for
non-causal -- is 1.5% faster than the `LPT=1` we had, so the L2-ordering question that could
not be settled by reading the code resolves in upstream's favour, by the same small margin as
everything else here.

**All three trait deviations found by the upstream audit were real, and together they are worth
2.4%.** The audit was worth running -- the deviations existed, `rs1` and `LPT=0` are free, and
the derivation-not-literal change prevents the next drift -- but it was never capable of finding
a 31x. That is the honest summary of this whole line of work.

<a id="fa3-profile-2026-10-06"></a>
**PROFILED (ncu, `-k device_kernel`, one 2048-q row against 18 432 KV, `num_splits=1` so the
`partial_o` overflow cannot contaminate it).** The kernel is **not latency-bound**, which
retracts the inference this entry was built on:

| metric | value | reading |
|---|---|---|
| Duration | 13.95 ms | vs 1.35 ms at roofline, 10.3x |
| Compute (SM) throughput | **47.4%** | pipes half busy, not idle |
| Warp cycles per issued instruction | **7.65** | a stall-bound kernel shows 30-100+ |
| DRAM throughput | 0.10% | fits in L2 at this size |
| Largest stall | CTA barrier, 2.9 of 7.65 cycles | present but small |

So warps issue efficiently and the pipes are busy. The kernel is not waiting -- it is executing
the wrong instructions, in enormous numbers:

| pipe | warp-instructions | share |
|---|---|---|
| total | 3 869 179 648 | |
| **tensor** | **74 317 824** | **1.9%** |
| FMA | 1 690 885 788 | 43.7% |
| ALU | 990 407 553 | 25.6% |
| XU | 306 805 440 | 7.9% |
| LSU | 213 841 800 | 5.5% |

**98.1% of issued instructions are not the mma.** Per tile-pair (64 512 of them) that is
58 824 non-tensor warp-instructions, i.e. **~115 instructions per softmax element**, where the
softmax itself needs roughly 5-10 (scale, max, subtract, exp, accumulate).

**RETRACTED before it was claimed: the mma is NOT falling back to Ampere `mma.sync`.** The
FLOP-per-tensor-instruction figure (7 282 against wgmma's 262 144) looked exactly like an
`mma.sync` fallback, which would have explained the whole 31x. Per-function SASS refutes it:

    device_kernel<FlashAttnFwdSm90<...>>        HGMMA=32   HMMA=0
    FusedFA3PersistentKernel                    HGMMA=32   HMMA=1584
    FusedFA3PersistentSymmetricKernel           HGMMA=32   HMMA=1584

FA3 uses wgmma exclusively; the 3 168 `HMMA.16816` in the `.so` belong to the **closed fused**
kernels. And the counts reconcile: 74.3M dynamic / 32 static = 2.3M passes = ~36 per tile-pair
x 262 144 FLOP = 9.4 MFLOP against the 8.4 MFLOP a tile-pair needs. The 7 282 figure came from
treating a per-warp metric as per-wgmma. `-arch=sm_90a` was correct all along.

**MEASURED: FA3's cost is `Theta(kv^2)`.** KV sweep at fixed q=2048 (one append row plus a
small FA2 row, `num_splits=1` throughout, **every row PASS**):

| kv | fa3 half | us/kv | us/kv^2 (x1e9) |
|---|---|---|---|
| 10 240 | 3 268.1 us | 319.2 | 31 167 |
| 18 432 | 10 583.7 us | 574.2 | 31 152 |
| 34 816 | 37 811.2 us | 1 086.0 | 31 193 |
| 67 584 | 142 434.4 us | 2 107.5 | 31 184 |
| 100 352 | 320 401.6 us | 3 192.8 | 31 816 |

**`us/kv^2` is constant to 2.1% across a 9.8x KV range, while `us/kv` varies by 197%.** For
fixed q the cost must be `Theta(kv)` -- each m-block walks the KV once -- so a clean quadratic
means something inside the per-KV-block loop does work proportional to the *whole* KV.

This also rules out a cache explanation: at kv=10 240 the KV working set is ~21 MB (inside
H200's ~60 MB L2) and at kv=100 352 it is ~205 MB (far outside), yet the quadratic constant does
not move across that boundary. A capacity effect would shift it. So the quadratic is
algorithmic.

**It also retroactively explains why every hypothesis so far returned ~2%.** Traits, pipeline
depth, gather granularity and SM budget are all *constant-factor* knobs; none of them can change
an exponent. The 1.21x on `kv=512` fixtures and the 31x at 65-100K are the same kernel at two
points on a quadratic -- which is why the short-KV result never generalised, and why I should
have measured the exponent before hunting constants.

Ruled out as the O(kv) culprit so far: upstream's `load_page_table` (`paged_kv.h:135`) is a
fixed `#pragma unroll` over `kPageEntryPerThread`, O(1) per block.

**The instruction count is quadratic in EVERY pipe, including tensor.** `ncu` at kv=10 240 vs
34 816 (a 3.40x KV ratio):

| pipe | ratio | exponent |
|---|---|---|
| total | 11.44x | 1.99 |
| **tensor** | 11.56x | **2.00** |
| ALU | 11.34x | 1.98 |
| FMA | 11.50x | 2.00 |

The **mma count itself** is quadratic, so this is not extra scalar work -- the whole mainloop
runs kv-proportionally too many times. Since the n-block count must be proportional to KV, the
**m-block** count has to be too, despite q being fixed.

<a id="combine-corrupts-fa2-rows"></a>
### Split-KV now works on FA3; the FA2 rows were never the combine's fault — 2026-10-06

**RETRACTED, by the isolation arm: the combine does not corrupt FA2's rows.** Three arms on
`qo:1,512 prefix:65536,512`, all other settings equal:

| arm | FA3 kernel | combine | row 1 (`task0 qo=512 kv=1024`) |
|---|---|---|---|
| **a) `n_fa3=0`, every row to FA2** | **no** | **no** | **0.385 FAIL** |
| b) `n_fa3=1`, `sm_split=1` | yes | yes | 0.552 FAIL |
| c) `n_fa3=1`, splits auto (29) | yes | yes | 0.600 FAIL |

Arm (a) runs **no FA3 kernel and no combine**, and the FA2 row still fails. So the whole
"the combine walks into FA2's rows" story is wrong -- including the `varlen_batch_idx_ptr`
reasoning, the `total_q` fix, and the inference from "same shape as the previously fixed
combine fault". The shape did match a known fault; that resemblance is what made me stop
looking.

**The `num_splits` correlation was also wrong.** Arm (a) reports `num_splits = 1` and fails.
**RETRACTED immediately, by the characterisation run: it is not "append rows" either.** Every
isolated shape passes, including the append:

| fixture (all `n_fa3=0`) | result |
|---|---|
| `qo:512 prefix:0` (pure prefill) | 0.00464 **PASS** |
| `qo:512 prefix:512` (append alone) | 0.00507 **PASS** |
| `qo:512 prefix:4096` (longer prefix) | 0.00417 **PASS** |
| `qo:512,1024 prefix:0,0` (two prefills) | 0.00375, 0.00329 **PASS** |
| `qo:512,1 prefix:512,65536` (append, then long decode) | 0.00416, 0.00036 **PASS** |
| **`qo:1,512 prefix:65536,512`** (long decode, **then** append) | 0.00035, **0.385 FAIL** |

**It is ROW ORDER.** The last two fixtures contain the *same two rows* and differ only in
which comes first. Reversed, both pass; with the long-KV row first, the second row is wrong.
Nothing is wrong with any shape in isolation.

That lines up with the TraceLab signature already recorded -- error rising with row index
across rows of identical shape (`kv=65537, qo=1` giving 0.0088 -> 0.0177 at rows 7 -> 20) --
and makes both the same finding: **a row is corrupted by what precedes it.**

**The split-KV workspace hypothesis is REFUTED: there is no threshold.** Sweeping the leading
row's prefix in the failing order, the second row is wrong at every length, including
`kv=1025` where the leading row barely splits at all:

| leading row prefix | leading row (task2) | second row (task0, `qo=512 kv=1024`) |
|---|---|---|
| 1 024 | 0.00263 ok | **0.465 FAIL** |
| 4 096 | 0.00154 ok | **0.482 FAIL** |
| 8 192 | 0.00133 ok | **0.530 FAIL** |
| 16 384 | 0.00077 ok | **0.596 FAIL** |
| 32 768 | 0.00053 ok | **0.391 FAIL** |
| 65 536 | 0.00035 ok | **0.459 FAIL** |

Flat, with no dependence on the leading row's length. So it is not a long row's partials
landing on a later row, and the "global partial round-trip" mechanism is not it.

**It is TASK ORDER.** Collecting every arm by the per-row task assignment the harness prints:

| fixture | task order | outcome |
|---|---|---|
| `qo:1,512 prefix:*,512` | **[2, 0]** | the `task0` row fails, 0.39-0.60, at every prefix |
| `qo:512,1 prefix:512,65536` | [0, 2] | both pass |
| `qo:512,1024 prefix:0,0` | [0, 0] | both pass |
| TraceLab | [0,0,0,0,2x18] | `task0` rows pass; only the last `task2` rows are marginal |

**A `task0` row placed after a `task2` row is corrupted.** Not shape, not length, not
split count, not the combine, not FA3 -- the queue the preceding row was assigned to.

**CONFOUNDED -- this comparison does not establish what I claimed.** The standalone script
omitted `FLASHINFER_FA3_TASK`, so it planned **3 tasks** while the harness plans **4**. It
therefore varied the plan *and* the kernel together, and cannot attribute the fault to either.
Re-running it with `FLASHINFER_FA3_TASK=1` produces **no output at all**: flashinfer's
`BatchAttention.run()` refuses a 4-task plan, which the harness's own comment already records
("BatchPagedAttentionRun correctly refuses to execute one"). So the 3-task configuration is
the *only* one upstream can run, and "upstream is correct on this ordering" is a statement
about a different plan, not a control for ours.

What the script does still establish: the independent fp32 reference agrees with flashinfer on
these fixtures, so the harness's `exact()` is not the thing that is wrong. The claim it does
**not** establish is "the bug is ours rather than upstream's" -- that remains open, and no
clean control for it exists through this API, because upstream cannot execute the plan shape
the fault requires.

**Original (confounded) result, retained for the trace.** A standalone script using only
flashinfer's own `BatchAttention` API -- no fused extension, **3-task plan**, no `QUEUE_EXEC`,
with an independent fp32 reference -- handles the failing order correctly:

| case (pure flashinfer) | row 0 | row 1 |
|---|---|---|
| BAD order, decode first `qo=[1,512] kv=[65537,1024]` | 0.00048 ok | **0.00587 ok** |
| GOOD order `qo=[512,1] kv=[1024,65537]` | 0.00594 ok | 0.00044 ok |
| BAD order, short lead `kv=[1025,1024]` | 0.00290 ok | 0.00428 ok |

Our path gives **0.39-0.60** on that same row. Two things follow: upstream flashinfer's
persistent kernel and plan are fine on this ordering, and the independent reference agrees
with flashinfer, so the harness's `exact()` is not the thing that is wrong either. The fault
is in **our 4-task unified plan path**, which is entered even at `n_fa3=0` because `UNIFIED=1`
plans four tasks with the FA3 task empty.

This also retracts, by implication, the framing that this is a "baseline correctness problem"
that "taints the control arm of every comparison". The *true* baseline -- unmodified
flashinfer -- is correct. What is broken is our unified plan, which the control arm of the
TraceLab A/B also uses, so the control is still implicated; but the defect is ours to fix and
is not evidence against flashinfer's kernel.

**RETRACTED: it is not a partial final page.** Running the two-row reproducer at
`PAGE_SIZE=1`, where no row can have a partial final page, the following row **still fails at
0.589**. Page alignment is therefore not the mechanism, and the page-128 framing below is
wrong.

**Corrected rule: the preceding row's KV exceeding the chunk limit (~1024).** Collecting every
arm against both candidates:

| leading `kv` | > 1024 | multiple of 128 | following `task0` row |
|---|---|---|---|
| 1 | no | no | FAIL **0.023** (marginal) |
| 128 | no | yes | PASS |
| 1 024 | no | yes | PASS |
| 1 025 | **yes** | no | **FAIL 0.465** (page 128) / **0.589** (page 1) |
| 1 027 | **yes** | no | **FAIL 0.501** |
| 4 097 | **yes** | no | **FAIL 0.482** |
| 8 193 | **yes** | no | **FAIL 0.530** |
| 16 385 | **yes** | no | **FAIL 0.596** |
| 32 769 | **yes** | no | **FAIL 0.391** |
| 65 537 | **yes** | no | **FAIL 0.459** |

Every **hard** failure (0.39-0.60) has leading `kv > 1024`; both `kv <= 1024` arms with a full
page pass. That is the KV **chunk limit**: above it a row is split into chunks and reduced
through the merge path (`partial_indptr`, `merge_indptr`, `merge_o_indices`, `partial_o_nnz`),
and a split row corrupts the row that follows it. `kv=1` is the one arm the rule does not
cover, and its 0.023 is a different magnitude class -- the same borderline band as TraceLab's
0.021-0.030 -- so it is probably a separate, milder issue rather than this one.

**The interference is strictly FORWARD, and position-dependent.** Three-row arm, with
identical `task0` rows on both sides of the split row:

| fixture | row 0 | row 1 | row 2 |
|---|---|---|---|
| split first, plain second | 0.00291 ok | **0.472 FAIL** | - |
| plain first, split **second** | 0.00569 ok | 0.00238 ok | - |
| **plain, split, plain** | 0.00471 **ok** | 0.00254 ok | **0.482 FAIL** |

The `task0` row *before* the split row is clean; the identical `task0` row *after* it is
corrupted. A split row also computes its **own** output correctly every time. So a split row
advances some cumulative quantity that work items created later consume wrongly -- it is an
ordering/offset fault in plan construction, not a data race and not a masking error.

**Minimal reproducer** (two rows, flips on a one-token change):

    FAIL : qo:1,512 prefix:1024,512 n_fa3:0   (leading kv=1025, splits)
    PASS : qo:1,512 prefix:1023,512 n_fa3:0   (leading kv=1024, does not split)

**Suspects eliminated by reading, so the next attempt does not re-spend them:**

- `plan_info` argument layout: `ToVector`/`FromVector` match exactly (2 + `NUM_TASKS`*12 + 18
  = `NUM_SHARED_ARGS` 20); the 18 -> 20 growth was done correctly.
- `partial_o_nnz` and `merge_indptr` scope: both declared at `scheduler.cuh:1520-1521`,
  *before* the task loop at 1533, so they accumulate across tasks rather than resetting.
- `group_row_begin` vs `partial_o_nnz` unit mismatch: `group_row_begin` indexes
  `merge_indptr`/`merge_o_indices`, which are row-indexed and themselves hold the partial
  offsets, so the units are consistent.
- stale `gid_base` leaking to a later row: declared *inside* the `qo_tile_idx` loop, reset
  per tile.
- the combine, FA3, `num_splits`, page alignment, row order, task class: all refuted by
  measurement above.

**`cluster_partial_indptr` is REFUTED by experiment.** Zeroing it for non-split work items
(`PI_ZERO=1`) leaves the corruption intact: 0.566 on the two-row reproducer and 0.512 on the
three-row one, against 0.472 for the unmodified control. So the kernel does not consume that
value for non-split items.

**PLAN DUMP (the measurement, not a reading).** Dumping the 4-task plan's arrays for the PASS
and FAIL fixtures and diffing corrected two of my own rules:

| | PASS `kv=[1024,1024]` | FAIL `kv=[1025,1024]` |
|---|---|---|
| task2 work items | 4 -- **one chunk**, kv 0->1024 | 8 -- **two chunks**, 0->1024 and 1024->1025 |
| `group_row_begin` | 0, **128**, 256, 384, 512 | 0, **7**, 135, 263, 391 |
| `group_nrows` | all **128** | **7**, then 128 ... |
| `num_groups` | 112 | 116 (= 112 + 4) |
| `merge_o_indices` | starts at **28** (token 1) | starts at **0** (token 0) |

Corrections this forces:

- **`len_kv_chunk` is 512, not 1024.** The per-task limits differ; task 2's is 1024, which is
  why the measured boundary sat at 1024 -- `kv=1024` is one chunk, `kv=1025` is two. The
  threshold rule was right about the number and wrong about the quantity it named.
- **The follower row *does* split.** The `qo=512 kv=1024` row is itself two chunks of 512 in
  both cases, so "a split row corrupts a non-split follower" was wrong. Both rows split in
  the FAIL case; only the follower splits in the PASS case.

**What actually distinguishes the two:** in FAIL the leading row contributes a group with
**`nrows=7`** -- a *partial* q-tile, since `packed_qo_len = 1 x 7 = 7` against
`cluster_tile_q = 16`. Every group in the PASS case is a full 128-row tile. So the live
candidate is a **partial q-tile group** reaching the JIT merge, which is also what every
failing fixture has in common (`qo=1` -> packed 7; `qo=4` -> packed 28, also partial) and what
TraceLab's failing decode rows have (`qo=1`, packed 7, long KV so they split and reach the
merge).

**Eliminated by reading, added to the list so it is not re-spent:** `MergeGroupJIT` does not
overrun a partial group -- its loop is `for (r = row_begin + warp_id; r < row_begin + nrows;
...)` (`jit_merge.cuh:40`), correctly bounded, and its output index
`merge_o_indices[r] + kv_head_idx * gqa_group_size` matches the
`[token, kv_head, gqa_group, dim]` layout exactly.

**CONFIRMED by controlled experiment: a PARTIAL q-tile is the trigger.** With `gqa_group_size
= 7` and `cluster_tile_q = 128`, `qo=128` gives `packed_qo_len = 896 = 7 x 128` exactly while
`qo=127` gives 889 = 6 tiles + 121. Both rows split (`kv = 65 664` / `65 663`), so split count
is held constant and only partiality varies:

| leading row | `packed_qo_len` | tiles | leading row | following `task0` row |
|---|---|---|---|---|
| `qo=128` | 896 | **7 exact** | 0.00051 ok | **0.00307 PASS** |
| `qo=127` | 889 | 6 + **121 partial** | 0.00051 ok | **0.0657 FAIL** |
| `qo=1` | 7 | **7 of 16, partial** | 0.00039 ok | **0.700 FAIL** |

A one-token change in the leading row's q length flips PASS to FAIL with everything else
equal. And the **error magnitude scales with the overrun**: a tile missing 7 of 128 rows gives
0.0657, one missing 9 of 16 gives 0.700.

**Mechanism this implies.** `MergeGroupJIT` consumes only `nrows` (verified above), and
`partial_o_nnz` advances by `row_tile_size * num_kv_tiles`, which is also correct. So the
overrun is on the **producer** side: a work item for a partial tile appears to write a
**full** `cluster_tile_q` worth of partial rows, of which only `row_tile_size` are valid. The
surplus rows land in the next group's partial region, so the next group merges clobbered
partials. That accounts for every observation at once -- forward-only (the surplus runs past,
never before), only with partial tiles, never affecting the partial row's own output (its own
`nrows` are written correctly before the overrun), and a magnitude set by overrun size
relative to the victim.

**Why serving hits this constantly.** Every decode row has `packed_qo_len = 1 x
gqa_group_size`, which is partial for any `cluster_tile_q > gqa_group_size` -- i.e. always. So
any batch with a long-KV decode row (which splits, reaching the merge) followed by another row
corrupts that row. TraceLab is exactly this shape, which is why its decode rows fail and why
the error grows across 18 consecutive partial-tile rows.

**The fix is to bound the partial-O write by `row_tile_size`** rather than the tile size, or to
reserve partial space at full-tile granularity so an overrun is harmless. The first is correct;
the second only hides it. The two-row pair `qo:128,512` (PASS) and `qo:127,512` (FAIL) at
`prefix:65536,512` is the regression test, and it is cheap.

**RETRACTED by the A/B: the `partial_lse` indexing is not the fault.** Making it tile-relative
to match `partial_o` (`LSE_TILE_REL=1`) leaves the reproducer failing at **0.431**, against
**0.491** for the unmodified control -- within the run-to-run spread already seen on this
fixture (0.39-0.70). The `qo=127` partial-tile case is likewise unchanged at 0.067. So the
tile- vs row-relative divergence is real in the source but is not what corrupts the following
row. Twelfth dead hypothesis, and the fourth to die after careful source reasoning.

The change is **kept** (`LSE_TILE_REL=1` is the default): tile-relative matches `partial_o`
and the plan's per-tile `partial_indptr`, so the two indexings now agree, which is correct on
its own terms even though it fixes nothing here. The old expression stays behind the switch.

**The win reproduces a third time, with the corruption still present.** Same run, TraceLab at
page 128, 3 iters:

| arm | total |
|---|---|
| FA2 on everything | 15 360.2 us |
| **split, appends to FA3** | **12 316.3 us** |

**1.247x, 19.8% lower** -- against 1.225x measured earlier on the same fixture at 5 iters. The
`task0` and visible `task2` rows are correct in both arms (0.00021-0.00545), consistent with
the partial-tile fault needing a *split* leading row, which TraceLab's ordering puts after the
`task0` rows.

**RETRACTED: the victim does NOT need to be merged, so wrong LSE weights are not the fault.**
`task0`'s chunk limit is 512, so a victim with `kv=512` is a single chunk that writes through
to `final_o` directly -- no partials, no merge, no LSE weights involved. It still fails:

| arm | victim row |
|---|---|
| victim `kv=512`, **one chunk, write-through** | **0.807 FAIL** |
| victim `kv=1024`, two chunks, merged | 0.648 FAIL |
| victim `kv=256`, **alone** (no leading row) | 0.00433 PASS |
| victim `kv=512` after a `qo=127` leader | 0.105 FAIL |

A row that never enters the merge cannot be corrupted by the merge's weights, so the
LSE-weight reading of the `DUMP_ROWS` evidence was wrong. What that evidence actually
established -- fully written, in range, wrong elementwise -- is still true and still rules out
a skipped store and a zeroed region; it just does not single out LSE.

**What this leaves.** The victim's *direct* `final_o` write is being corrupted by something
the leading split row does. The leading row's only writes to `final_o` are through
`MergeGroupJIT` at `merge_o_indices[r] + kv_head_idx * gqa_group_size`, so that index is the
remaining candidate: if it overshoots the leading row's own token range it lands on the
victim's rows, which would explain forward-only damage, in-range values (the leading row's
merged output has similar magnitude), and the dependence on a partial tile. Hand-checking the
plan's construction for `qo=1, gqa=7, cluster_tile_q=16` gives `row_tile_size=7` and
`q = row/7 = 0` for rows 0-6, i.e. token 0 only, which looks correct -- so the overshoot, if
it exists, is in the kernel's use of those indices rather than in their construction.

**MEASURED: the damage is TOTAL, which eliminates the whole write-collision class.** Per-token
and per-head diff on the victim (`DUMP_ROWS=1`, harness extended to report the damaged sets):

| fixture | victim | damaged tokens | damaged qo_heads |
|---|---|---|---|
| `qo:1,512 prefix:65536,0` | `qo=512 kv=512` | **512 / 512** (0 ... 511) | **28 / 28** |
| `qo:127,512 prefix:65536,0` | `qo=512 kv=512` | **512 / 512** | **28 / 28** |

Every token and every head is wrong, while the leading row is perfect (0 / 1 and 0 / 127
damaged). An overshoot from the preceding row would damage a **bounded prefix** -- the surplus
rows only -- and a stride fault would leave a periodic pattern. Neither is what this is. So
`merge_o_indices` overshoot, the `kv_head_idx * gqa_group_size` term, and write collisions in
general are all refuted.

**What total, in-range, fully-written damage means:** the victim is not being overwritten, it
is **computing with the wrong inputs**. Combined with the earlier arms, the victim is correct
whenever it is the *first* row (`qo:512,1 prefix:512,65536` passes; `qo:256 prefix:0` alone
passes) and wrong whenever a **split** row precedes it -- while two *unsplit* rows in sequence
are fine (`qo:512,1024 prefix:0,0` passes, and there the second row's `kv_indptr` is non-zero).
So the live candidate is the victim's **KV base** (`kv_indptr` / page offset) or its
`kv_start`/`kv_end` chunk bounds being computed wrong when a split row precedes it -- which
would make it attend the wrong KV and produce exactly a fully-wrong, in-range result.

**MEASURED: the plan is CORRECT, so the KV-base candidate is refuted.** Dumping the victim's
work items in the FAIL fixture (`qo:[1,512] kv:[65537,512]`):

| field | victim (task 3) | expected |
|---|---|---|
| `q_indptr` | 1 | 1 (one token in the leading row) |
| `kv_indptr` | **513** | 513 (`ceil(65537/128)`) |
| `kv_len` | 512 | 512 |
| `kv_start` / `kv_end` | 0 / 512 | 0 / 512 (one chunk) |
| `batch_idx` | 1 | 1 |

Every value is right, and the leading row's are too (`kv_indptr=0`, `kv_len=65537`, chunks of
2560 at `len_kv_chunk=1280`). So nothing in the plan's KV addressing is wrong and the victim
is not being pointed at the wrong KV.

**What the dump did reveal: task 3's `work_indptr` begins `[0, 0, 0, 0, 0, 0]`** with 112 work
items, i.e. the first several clusters receive **no** task-3 work because task 2 (the split
leader, 104 work items) consumed them. In the PASS fixture task 2 is empty and task 3's
`work_indptr` is `[0, 3, 6, 9, 11, 13]`, distributing from cluster 0.

That matters because the kernel derives its q-tile position from the cluster slot:

    qo_packed_idx_base = packed_qo_start
                       + persistent_cluster_slot<KTraits>() * CTA_TILE_Q
                       + get_warp_idx_q<KTraits>(tid.y) * NUM_MMA_Q * 16

If `persistent_cluster_slot` is computed from a cursor that does not account for an earlier
task having consumed clusters, every one of the victim's work items lands on the **wrong q
tile** -- which produces exactly the measured signature: all 512 tokens and all 28 heads
wrong, fully written, in range, with the leading row untouched. It also explains why the
victim is fine whenever it is the first row (nothing consumed clusters ahead of it) and why
two unsplit rows are fine (a single unsplit row generates few work items and both rows live in
task 3, sharing one cursor).

**SHARPENED by three more arms -- the rule is REQUEST ORDER of the split row:**

| fixture | split row's request index | `task0` rows |
|---|---|---|
| `qo:512,1 prefix:0,65536` | **1 (after)** | both **PASS** (0.00364, 0.00024) |
| `qo:1,512 prefix:65536,0` | **0 (before)** | **0.845 FAIL** |
| `qo:1,512,512 prefix:65536,0,0` | **0 (before)** | **both FAIL** (0.625, 0.607) |

**A split row with a lower request index corrupts every subsequent `task0` row.** Two things
this settles:

- `q_indptr` is **not** the variable: the two victims in the three-row fixture sit at
  `q_indptr` 1 and 513 and both fail.
- Cluster consumption alone is **not** sufficient: in the PASS fixture the split row exists and
  generates the same ~104 work items, yet nothing is corrupted. So the plain
  "task 2 consumed clusters ahead of task 3" form of the cluster-slot story is **refuted** --
  the plan iterates tasks in index order regardless of request order, so task 2's consumption
  is identical in both fixtures.

What remains must be something indexed by **request order across tasks**. The candidates with
that property are the globally-accumulated arrays built in task order: `merge_indptr`,
`merge_o_indices` and the per-work-item `partial_indptr` (`o_indptr`). Note the earlier
`PI_ZERO` experiment only zeroed `partial_indptr` for **non-split** work items, so it did not
test the split row's own values -- that experiment does not clear this candidate.

**MEASURED: the plan is correct in BOTH orderings, so every plan-array candidate is refuted.**

| | FAIL (split row = request 0) | PASS (split row = request 1) |
|---|---|---|
| `merge_o_indices` | `[0,1,...,6]` | `[14336,...,14342]` |
| expected | `(0 + 0)*28 + r` = 0..6 **correct** | `(512 + 0)*28 + r` = 14336..14342 **correct** |
| `merge_indptr` | `[0,26,52,...,182]` | identical |
| `group_row_begin` / `group_nrows` | 0 / 7 | identical |
| victim's `partial_indptr` | **182** | **182** -- identical |
| victim `q_indptr` / `batch_idx` | 1 / 1 | 0 / 0 |

The two plans are structurally identical and individually correct; `partial_indptr` is even
the same number (182 = 7 rows x 26 chunks), so it cannot be the differentiator. That retires
`merge_indptr`, `merge_o_indices`, `partial_indptr`, `group_row_begin`/`nrows` and the KV
bounds -- i.e. **the plan is fully exonerated**, having now been dumped and checked twice on
different fixture pairs.

**So the fault is in the kernel's execution, and the distinguishing condition is precise:**
the victim is corrupted exactly when its `q_indptr` is **greater** than the split row's
(FAIL: victim at 1, split row at 0; PASS: victim at 0, split row at 512). Both rows' plan
entries are right, so some kernel-side state carried between work items -- or a write whose
extent is derived from the wrong row -- depends on that relative order.

**MEASURED: it is a RACE, not an indexing bug.** Eight consecutive runs of the FAIL fixture,
nothing changed between them:

    rel err  0.77292  0.65776  0.72528  0.68167  0.68027  0.68413  0.67000  0.68319

Not bit-identical, so the corruption is **nondeterministic**. That **retracts the assumption I
recorded one turn earlier** -- "this corruption is deterministic, which argues against a race"
-- which I asserted without testing. It is a race, and the class flips from indexing to
synchronisation.

Two things this reclassifies in hindsight:

- The magnitude varying across fixtures (0.39 to 0.84) is **run-to-run variance**, not a
  function of overrun size as I had read it. The apparent correlation between error size and
  tile partiality was noise.
- Treating `LSE_TILE_REL=1`'s 0.431 against the control's 0.491 as "within spread" was the
  right call, and is now confirmed: the spread is the race, not a signal.

**This is our code.** The JIT-reduction rendezvous is the port's own addition, and its
comments already record *intermittent* wrong merges on "the one batch shape whose FA2-class
rows split their KV" -- 3 runs in 10, fixed by fence ordering. What we have now is the same
path failing on essentially every run, which is why it read as deterministic. The remaining
suspects are all in that rendezvous:

- `group_counter[gid]` and the "hand the counter back at zero" reset
  (`persistent.cuh:611`). One plan is shared by every layer, so a counter released early lets
  a merge run on partials that are not all published.
- the `__threadfence()` then `persistent_block_sync` ordering (`persistent.cuh:567-570`).
- whether a **backfilling** CTA reaches the rendezvous with only one warpgroup alive, which
  the code comments flag as a trap already worked around once.

**Why the ordering dependence now makes sense:** with the split row at request 0 its merge is
enqueued before the victim's work items, so the merge and the victim's write-through overlap;
with the split row second, the victim's work completes first and the race window closes. That
is an ordering effect, exactly as a race predicts, and it needed no appeal to indexing.

**MEASURED: the JIT-reduction rendezvous is EXONERATED.** Nulling `group_id` disables it at
its own gate (`persistent.cuh:551`). Four runs each side:

| arm | victim rel err | split row |
|---|---|---|
| `NO_JIT_REDUCE=0` (rendezvous on) | 0.729, 0.635, 0.616, 0.720 | 0.00024-0.00029 ok |
| `NO_JIT_REDUCE=1` (**rendezvous off**) | **0.765, 0.585, 0.702, 0.696** | 0.00021-0.00029 ok |

Identical distributions, so the race is not in the rendezvous. Twentieth candidate eliminated,
and the one the code's own comments pointed at.

**RETRACTED, by reading the gate I should have checked before interpreting the bisect: there is
no second reducer.** `persistent_template.cuh:108` runs the classic single-shot reduction
**only when `group_id == nullptr`**:

    if (params_1.group_id == nullptr) {
      grid.sync();
      BlockReductionRunner::Run(..., merge_indptr, merge_o_indices, smem);
    }

The two reducers are **mutually exclusive by construction** -- the comment right above says so
("the classic barrier + single-shot reduction is kept for A/B measurement and is selected
host-side by nulling group_id"). So nulling `group_id` did not disable reduction, it
**swapped** to the classic path, which is why the split row stayed correct. My reading of that
result -- "a second reducer must be running alongside ours" -- was wrong, and the four
properties I claimed it explained were explained by nothing.

**What the bisect actually established, which is stronger:** the corruption occurs with
**either** reducer, at the same magnitude and spread. That exonerates the reduction machinery
as a whole, not merely the JIT rendezvous -- both the JIT path and the classic `grid.sync` path
produce it. Any remaining explanation has to be upstream of reduction entirely.

It also retires the reframing I attached to the earlier reading: the JIT reduction **does**
replace the classic pass rather than running beside it, so the "replacing the grid.sync
reduction" description of that item is accurate after all.

**Superseded reading of the same data.** Its KV is split into 26 chunks, no rendezvous runs, and it still comes back at
2.1e-4. That is only possible if a **second reduction path** is reducing those partials -- i.e.
flashinfer's original `merge_indptr`/`merge_o_indices` pass is still executing alongside (or
instead of) ours. Two reducers over the same `partial_o`/`merge_indptr` arrays, one of them
ours and neither aware of the other, is a far better fit for every property measured:

- **nondeterministic** -- two writers, no ordering between them;
- **damage total and in-range** -- the victim's correct direct-write output is overwritten by a
  reduction that believes it owns those rows;
- **request-order dependent** -- which rows the second reducer walks depends on the plan's
  global row ordering, which is what request order changes;
- **leading row always fine** -- it is the row the reduction legitimately owns.

That also reframes the smem-handoff JIT reduction item: if the original pass still runs, the
JIT reduction is not *replacing* it, so any speedup attributed to "replacing the grid.sync
reduction" was never actually replacing anything.

**Next measurement:** find and instrument the second reducer. `grep` for the launch that
consumes `merge_indptr`/`merge_o_indices` outside `persistent.cuh` (flashinfer's standalone
merge/reduction kernel), confirm whether the ext launches it, and disable **it** instead of
ours on the reproducer. If the corruption disappears with the original pass off and our JIT
reduction on, the fault is that pass walking rows it no longer owns -- and the fix is to stop
launching it, which is what the JIT reduction was supposed to achieve.

**Superseded candidate: the per-task cluster-slot/cursor computation**, which the PASS arm
above refutes in its simple form. `num_pers_clusters = 132` and the plan carries a `cluster_cursor_offset`; the next
measurement is to dump `cluster_cursor` alongside `work_indptr` for both fixtures and check
whether the victim's slot values start from zero or continue from task 2's consumption. The
plan dump already reads these arrays, so this is a fixture change, not new tooling.

**Superseded lead, from the victim row's character (`DUMP_ROWS=1`).** The corrupted row is not
zeroed, not skipped and not garbage:

| | victim row (`task0 qo=512 kv=1024`) |
|---|---|
| nonzero elements | **1 835 008 / 1 835 008** -- every element written |
| absmax | **0.0703** against ref **0.0715** |
| element 0 | -0.0154 against ref -0.0087 (wrong, same order) |

The same holds for the `qo=127` case (absmax 0.0625 vs 0.0646). Fully written and **in range**
but wrong rules out a skipped store, a zeroed region, and uncorrelated data. It is the
signature of a **merge combining correct O partials with wrong LSE weights**: a convex
combination with bad weights stays inside the correct magnitude envelope while being wrong
elementwise.

So the seat is most likely the victim's **`partial_lse`** values, not its `partial_o` values.
That is consistent with the tile-relative change failing -- the index *base* was not the
problem, so an overrun or stride fault in the LSE write is the remaining candidate. Note
`partial_lse` is indexed `(o_indptr + local*nt + chunk) * num_kv_heads + kv_head_idx` while
`partial_o` reaches the same row-chunk pair via
`((o_indptr + chunk) * num_kv_heads + kv_head) * HEAD_DIM + local * (nt * num_kv_heads *
HEAD_DIM)`; those agree on the row-chunk index, so the next thing to check by measurement is
whether the **LSE guard** admits rows the O guard rejects, which would let the leading row's
LSE writes run past its 455-entry region into the victim's.

A direct test exists and is cheap: zero `partial_lse` before the launch and see whether the
victim's error changes character (if the merge is reading clobbered LSE, pre-zeroing changes
the wrongness; if it is reading its own correct LSE, nothing changes).

**Write path located, with one concrete inconsistency (now refuted as the fault).**
`persistent.cuh:503-509` takes the split branch and calls `write_o_`, whose gmem store is
guarded by `if (o_idx < qo_upper_bound)` (`persistent.cuh:133`) with

    qo_upperbound = min(q_len, ceil_div(qo_packed_idx_base + CTA_TILE_Q, gqa_group_size))

That guard is in **token** units (`o_idx = q`, the de-packed token index) while the write
offset is in **packed-row** units. Walking the failing shapes by hand, the guard does appear
to block the surplus rows: for `qo=1, gqa=7, CTA_TILE_Q=16`, packed row 7 maps to `q=1` and
`qo_upperbound=1`, so it is rejected; for `qo=127`, packed 889 maps to `q=127` against
`q_len=127`, also rejected. So on inspection the partial-O guard looks correct, and the
producer-overrun mechanism above is **not yet confirmed at the source level** even though the
measurements it predicts all hold.

**The inconsistency worth testing first.** `partial_o` is indexed tile-relative:

    o_ptr_base + (o_packed_idx - o_packed_idx_base_cta) * o_stride_n        // CTA tile base

while `partial_lse` is indexed row-relative (`persistent.cuh:535-538`):

    partial_lse[(o_indptr + (packed_qo_idx - packed_qo_start) * num_kv_chunks + kv_chunk_idx) ...]
                                             ^^^^^^^^^^^^^^^^ the ROW's start, not the tile's

For a row's **first** tile these coincide; for any later tile they diverge by the tile offset.
Every failing fixture has a leading row that splits, and a row large enough to span multiple
q-tiles would then write its LSE partials to the wrong rows while writing O partials to the
right ones -- which corrupts the merge for rows whose LSE came from elsewhere. That is a
testable, local discrepancy and it is where the next session should start: make the two agree
(both tile-relative or both row-relative, whichever matches `merge_indptr`'s construction) and
run the `qo:128,512` / `qo:127,512` pair.

**Not implemented here deliberately.** Changing an indexing expression in the FA2 epilogue on
the strength of an unverified reading is exactly the move that produced the last six dead
hypotheses, and a wrong change here would silently corrupt a path the TraceLab A/B depends on.
The regression pair makes verification cheap, so the fix should be made *with* a measurement,
not before one.

**This also retracts my earlier "no threshold, so not the split machinery" refutation.** That
prefix sweep ran 1024, 4096, 8192, 16384, 32768, 65536 -- every point at or above the limit,
so everything failed and I read a flat result as "no threshold". The threshold was one step
below where I started sampling. A sweep that brackets a boundary only finds it if it spans
both sides, and mine did not.

**And it retracts my "serving is not blocked" claim from the previous turn.** I argued serving
runs at `page_size=1` where no partial page exists, so the A/B could proceed. Page size is
irrelevant to the actual mechanism, and serving decode rows carry exactly the long KV that
splits -- 64-100K contexts are far above 1024. So serving **is** affected, the A/B would run
on a corrupted baseline, and this must be fixed first after all. My reason for unblocking was
wrong even though the frustration behind it was reasonable.

**Superseded diagnosis, kept for the trace: "a partial final KV page corrupts the next row".**
The page-128 boundary data below is real but coincidental -- at page 128 the lengths I swept
that were multiples of 128 (128, 1024) also happened to be the ones at or below the chunk
limit:

| leading row `kv` | pages | last page | following `task0` row |
|---|---|---|---|
| 1 | 1 | **partial**, 1/128 | **FAIL 0.023** |
| 128 | 1 | full | PASS |
| **1 024** | 8 | **full** | **PASS** |
| **1 025** | 9 | **partial**, 1/128 | **FAIL 0.465** |
| 1 027 (a `task1` row) | 9 | **partial**, 3/128 | **FAIL 0.501** |
| 4 097 / 65 537 | - | **partial** | **FAIL 0.48 / 0.46** |
| 1 024 + 1 024 (two rows) | 8, 8 | full, full | PASS |

`kv=1024` passes and `kv=1025` fails, so it is page alignment, not length, order or class --
and the earlier "task2 before task0" rule was an artifact of every decode fixture I built
having `prefix+1` lengths, which are never multiples of 128.

**It explains the TraceLab failures too.** Those `task0` rows are `kv=512` and `kv=1024`,
exact multiples of 128, and they pass. The rows that fail are the decode rows at
`kv=65537, 73729, 81921, 90113, 98305` -- every one `prefix+1`, every one a partial final
page. The "error grows with row index" signature is then just accumulation across 18
consecutive partial-page rows, not an index effect.

**Mechanism, and why it is ours.** Page offsets are built page-granular on both sides
(`kv_c_pg = cumsum(ceil(kvl / PAGE))` in the harness; the same in the pure-flashinfer
script). With exact multiples, `cumsum(kv_len)` and `cumsum(pages) * PAGE` coincide; with a
partial page they diverge, which is exactly the observed boundary. So something in our path
derives a following row's KV base token-granularly where it must be page-granular. Pure
flashinfer handles the identical fixture correctly (`kv=65537` leading, row 1 at 0.00587), so
the divergence is in our 4-task plan or our ext's runner parameters, not upstream.

Note this also means **`page_size=1` would hide the bug entirely** (every length is a
multiple of 1), which is why it never appeared before the page-128 work, and why the serving
default of 1 has been masking it.

**Superseded suspect, kept for the trace: the `plan_info` argument layout.** It is in fact
consistent -- `ToVector` writes 2 leading + `NUM_TASKS * 12` + 18 trailing = 20 shared, and
`FromVector` reads tasks at base 2 and the trailing block at `2 + NUM_TASKS * NUM_TASK_ARGS`,
matching exactly. The `18 -> 20` growth was done correctly. Refuted by reading, before
costing a measurement: `cluster_q_indptr` pushes absolute `qo_indptr_h[i]`
(`scheduler.cuh:1649`), which is correct, so the offsets going *in* are right. But the port
grew `NUM_SHARED_ARGS` from 18 to 20 (`max_splits_fa3`, `fa3_batch_list_offset`,
`num_fa3_requests`, with `num_fa3_clusters` kept last so `plan_info[-1]` still works), while
per-task arguments are read at `vec[2 + i * NUM_TASK_ARGS + k]`
(`scheduler.cuh:1148-1153`). If that base or stride did not move with the shared-arg count,
per-task slices are read from the wrong positions -- and which positions are wrong depends on
which tasks hold work, which is exactly an order-dependent symptom. The `[2,0]` two-row
fixture is a minimal reproducer, and `UNIFIED=0` cannot be used as a control because the
closed fused path's non-unified branch no longer compiles (it references `wgmma_plan`, which
only exists in unified params -- dead code, but it blocks that route).

**Scoping, in the 1.225x measurement's favour -- stated because it would be convenient to
leave unexamined.** TraceLab's natural ordering is `[0,0,0,0,2,2,...]`, i.e. every `task0` row
precedes every `task2` row, which is the **safe** order. Both arms of that A/B used it, and
the `task0` rows were correct in both (0.0022-0.0039). So this bug does not explain the
1.225x and does not invalidate it. What remains unexplained there is the marginal
`task2` failures at rows 20-21 (0.021-0.030 against `TOL=0.02`), which are a different
signature -- gradual with index rather than a hard 0.4-0.6 corruption -- and may well be a
separate issue or simply a tolerance that is too tight for 100K-context bf16 decode.

But a real serving workload does **not** guarantee the safe order: sglang batches whatever
arrives, so decode rows routinely precede prefill rows. That makes this a blocker for the
serving A/B rather than a curiosity, and it must be fixed before serving numbers mean
anything.

**CONFIRMED FROM THE SERVING CODE, not inferred -- and the two arms do not even use the same
plan shape.** `flashinfer_backend.py:1945-1946` sets `FLASHINFER_FA3_TASK=1` on the
`SGLANG_TWO_EXEC` path, so the split arm plans **4 tasks** -- the configuration the corruption
requires. The baseline arm takes the `else` branch at :1958 and plans **3 tasks**, which the
corruption does not affect. And the fault is not page-size dependent: the reproducer fails at
`PAGE_SIZE=1` (0.589), which is what serving uses.

So an A/B run today would compare a **clean 3-task baseline** against a **corrupted 4-task
split arm**. That is not a slow arm or a noisy arm, it is an arm computing wrong attention
outputs, and any TTFT/ITL/throughput number from it would be measuring a different
computation. This is the apples-to-oranges failure the project's own standing rule forbids, so
the gate is correct and is now established by reading the serving path rather than by analogy
to the harness.

It also means the eventual A/B needs a third check beyond fallback counts: **both arms must
plan the same number of tasks**, or the comparison is invalid even once the race is fixed.

Arm (b) did not test what I intended: `SPLITS=1` sets `sm_split`, not `num_splits`, so (b) is
a second combine-enabled arm rather than a combine-disabled one. Arm (a) carries the
conclusion on its own, and the label is corrected here rather than quietly reused.

**This is the same bug family as the TraceLab row-20/21 failures**, which are also FA2 rows,
also unexplained, and were already noted as predating this work. Both now point at
flashinfer's persistent LPT kernel as we drive it, not at the FA3 port. That matters for
priority: it is a **baseline correctness** problem, so it taints the control arm of every
comparison, not just the split.

<a id="split-kv-fa3-correct"></a>
#### Split-KV on FA3 is correct after the seqused fix

With the `seqused` fix in, the split-KV fixtures were re-run (`drivers`-style, page 128,
`SPLITS=auto`, `num_splits=29` in all three). **FA3's own rows are now correct**, which they
were not before:

| fixture | FA3 rows | FA2 rows | fa3 half |
|---|---|---|---|
| `qo:8,8,100,1 prefix:16384,16384,0,0` | 0.00004, 0.00127 **ok** | 0.205, **1.392 FAIL** | 57.4 us |
| `qo:1,512 prefix:65536,512` | 0.00036 **ok** | **0.493 FAIL** | 80.5 us |
| `qo:1,512 prefix:98304,512` | 0.00032 **ok** | **0.509 FAIL** | 109.0 us |

Two things follow. First, **split-KV on FA3 is correct now** -- the `num_splits=29` garbage
recorded earlier (row 3 rel err 1.33) was the m-block bloat, not a partial-buffer sizing bug.
The harness does size `partial_o`/`partial_lse` with `_NSPLIT`
(`test_fused_server_batch.py:440-441`), so the doc's earlier "no `num_splits` factor" concern
does not apply to the harness and the prerequisite I listed for these fixtures was wrong.
Also note `fa3_half` here is 57-109 us, i.e. the 1604x entry's 17 358 us is gone too.

Second, **the combine pass corrupts the FA2 executor's rows, and only when it runs.** Every
arm with `num_splits=1` passes (the whole KV sweep, both TraceLab arms); every arm with
`num_splits>1` has correct FA3 rows and wrong FA2 rows. This is the same *shape* as the
already-fixed fault where the combine reduced never-written partials over FA2's rows, so the
`varlen_batch_idx_ptr` restriction is necessary but evidently not sufficient.

**Localised, not yet fixed.** The combine scheduler sizes its grid as

    num_blocks_m = ceil_div(args.total_q * args.num_heads, kBlockM) + args.b
    (flash_fwd_combine_kernel.h:326,330; also lower_bound_on_non_empty_tiles at :294)

and `LaunchUpstreamCombine` passes `p.total_q` -- the token count for the **whole** batch,
FA2 rows included -- while passing `b = p.num_fa3_requests`. So the scheduler is told a token
range spanning all rows but a batch count covering only the FA3 subset, and its tile ->
(batch, row) mapping is derived from both. A mismatch there walks tiles into token ranges the
FA3 subset does not own, which is exactly the observed signature.

**RETRACTED: narrowing `total_q` to the FA3 subset is NOT the fix.** I added
`UpstreamFA3Plan::fa3_total_q` (= `cu_seqlens_q[n_fa3]`) and passed it in place of
`p.total_q`, on the reasoning that the scheduler's token extent and batch count must describe
the same subset while addressing stays absolute. That reasoning about the code is still
correct as far as it goes, and it changed nothing measurable:

| fixture | FA2 rel err before | after the `total_q` change |
|---|---|---|
| `sh16k` rows 2,3 | 0.205, 1.392 | 0.153, **1.421** |
| `dec64k` row 1 | 0.493 | **0.479** |
| `dec98k` row 1 | 0.509 | **0.473** |

Unchanged within noise, so the grid-sizing mismatch was not the mechanism. The change is kept
(it makes the two arguments consistent and is harmless) but it is not a fix for anything, and
claiming the diagnosis from a code reading was premature -- that is the seventh hypothesis in
this investigation to die, and the second to die *after* I had reasoned carefully about the
source rather than guessed.

**Now isolating instead of theorising.** Three arms on `qo:1,512 prefix:65536,512`:
`n_fa3=0` (no FA3 kernel, no combine), `n_fa3=1` with splits forced to 1 (FA3 runs, combine
does not), and `n_fa3=1` with splits auto = 29 (both run). That partitions the possibilities
without needing a guess: if the first arm fails, the FA2 rows are wrong independently of FA3;
if the first two pass and the third fails, the combine is the writer; if the second fails, it
is the FA3 kernel or the plan.

Worth noting what the TraceLab arms already say: the same `task0 qo=512 kv=1024` shape is
**correct** there (rel err 0.0022) at `num_splits=1`, and wrong here at `num_splits=29`. So the
correlation with the split count is real even though the combine's grid arguments are not the
cause.

This blocks the smem-handoff JIT split-KV reduction item, which is only measurable on
split-heavy shapes.

<a id="fa3-root-cause-seqused"></a>
### ROOT CAUSE: the scheduler was handed `seqused_k` where it needs the QUERY length — 2026-10-06

> **FRAMING CORRECTION (2026-10-07).** The "42.8x" below is **the magnitude of a defect in our
> port, not a performance gain**, and it should not be quoted as an achievement. Upstream FA3
> was never 42x slower than FA2 -- that is not a plausible number for a tuned Hopper kernel,
> and treating it as one is what sent this investigation through six constant-factor
> hypotheses (traits, pipeline depth, gather granularity, SM budget) that could not possibly
> produce a 49x m-block inflation. The honest statement: `MakeUpstreamOwnSchedArgs` passed the
> wrong parameter, FA3 was unusable at long context as a result, and removing it restored
> roughly what upstream would have done from the start. The only defensible performance claim
> is **split vs FA2-on-everything, 1.23-1.25x**.

`MakeUpstreamOwnSchedArgs` passed `p.seqused_k` into `TileSchedulerArguments::seqused`. That
field is the **query** length per request, and the scheduler feeds it straight into

    num_m_blocks = ceil_div(seqused[b] * qhead_per_khead, kBlockM)

(`tile_scheduler.hpp:599-621`, the `Prepared=false` branch). So m-blocks were sized from the KV
length:

| kv | m_blocks emitted | correct (q=2048) | bloat | measured gap |
|---|---|---|---|---|
| 10 240 | 560 | 112 | 5.0x | |
| 18 432 | 1 008 | 112 | 9.0x | **8.1x** |
| 34 816 | 1 904 | 112 | 17.0x | |
| 67 584 | 3 696 | 112 | 33.0x | |
| 100 352 | 5 488 | 112 | 49.0x | |
| ~84 000 (TraceLab mean) | 4 592 | 112 | 41.0x | **39x** |

Multiplied by the n-block count, which is itself proportional to KV, total work becomes
`Theta(kv^2)`. The model predicts both the exponent (2.00, measured 1.98-2.00 in every pipe)
and the constant (41x predicted against 39x measured on TraceLab), so this is the whole effect,
not a contributor to it.

**Why it survived every correctness gate.** The surplus m-blocks address q rows past the
request's end, which the epilogue masks. The output was always *correct*; only the time was
wrong. A kernel that is 31x slow and bit-accurate defeats a numerics-only validity gate, which
is exactly the gate this work was using.

**Why six hypotheses in a row returned ~2%.** Traits (`MmaPV_is_RS`, `LPT`), pipeline depth,
gather granularity (cp.async vs TMA, page 1 vs 128) and SM budget are all *constant-factor*
knobs. **None of them can change an exponent.** Every one of them had to measure ~2% against a
quadratic, and each ~2% result was correctly measured and completely uninformative about the
cause. The lesson is procedural: **measure the exponent before hunting constants.** A
five-point KV sweep costs minutes and would have redirected this investigation at the start.

**The fix** passes `nullptr` instead of a `seqused_q` array: with `seqused == nullptr` the
scheduler differences `cu_seqlens` (`tile_scheduler.hpp:610-613`), and `cu_seqlens_q` is already
supplied and already per-request, so it yields the exact query length with no new buffer. The
mainloop keeps receiving `seqused_k` through its own arguments; this was only the scheduler's
copy. flashinfer `e4dfa82`.

**VERIFIED: the quadratic is gone and the split now beats the FA2-only baseline.**

| kv | before | after | speedup | predicted | us/kv after |
|---|---|---|---|---|---|
| 10 240 | 3 268.1 us | **792.4 us** | 4.1x | 5.0x | 77.38 |
| 18 432 | 10 583.7 us | **1 396.0 us** | 7.6x | 9.0x | 75.74 |
| 34 816 | 37 811.2 us | **2 610.8 us** | 14.5x | 17.0x | 74.99 |
| 67 584 | 142 434.4 us | **5 023.3 us** | 28.4x | 33.0x | 74.33 |
| 100 352 | 320 401.6 us | **7 425.4 us** | 43.1x | 49.0x | 73.99 |

`us/kv` is now constant to **4.5%** where it previously varied 197%, and `us/kv^2` varies 218%
where it was constant to 2.1%. **The cost is linear in KV, as it must be.** Every row PASS.
Speedups track the predicted bloat at a consistent ~87%, the residual being the fixed per-tile
cost that never scaled.

**TraceLab batch, the headline comparison** (page 128, 5 iters):

| arm | total | fa3 half |
|---|---|---|
| FA2 on everything (control) | 15 190.9 us | 7.7 |
| **split, appends to FA3** | **12 399.7 us** | 11 127.0 |

**1.225x faster than FA2-on-everything, an 18.4% reduction**, with the FA3 half going
476 684 -> 11 127 us (**42.8x**). This is the first time the two-executor split has beaten the
baseline on a realistic batch, and it retires the "FA3 is 31x slower, the split loses on long
context" result entirely.

**RETRACTED: the 400 TFLOP/s peak I used for every roofline figure in this document.** H200 SXM
bf16 dense tensor peak is **989.4 TFLOP/s**, not 400. Recomputing this batch's append rows:

| | achieved | % of real peak | what I claimed before |
|---|---|---|---|
| FA2 | 325 TFLOP/s | **33%** | "1.2x off roofline, at peak" |
| FA3 (fixed) | 443 TFLOP/s | **45%** | - |

So **FA2 is at a third of peak, not at it.** That invalidates the *explanation* I gave for
closing priority item 1: I attributed the flat FA2 K/V-pipelining result to "FA2 is already at
roofline, so there is no latency to hide." The measurement stands -- depths 2/3/4 are flat to
0.5% -- but the reason does not, and the flatness is now **unexplained**. Item 1 should be
reopened on the explanation, not on the measurement. It also means the earlier "FA3 is 39x off
roofline" was really ~95x, and FA3 now sits *above* FA2 in efficiency rather than merely
catching up.

**Fifth bug of this class in the port** -- per-request vs per-item, or q vs k, handed to a
parameter that wanted the other. See [upstream FA3 param mismatches](#upstream-fa3-param-mismatches).
Four were found by which rows were wrong; this one produced no wrong rows at all and was only
reachable through a scaling measurement.



**Earlier framing, now superseded**, and what is now known about it is narrower than before:
not the gather (TMA measured slower), not pipeline depth (2 is upstream's and 3 barely fits),
not parallelism (linear to 132 SMs), not per-tile work (same FLOPs), and not `MmaPV_is_RS`.
Arithmetic on the measurement: 896 work tiles x ~657 KV blocks = ~589k tile-block steps, i.e.
~4 464 per SM, so 476 ms is **~106 us per 128x128x128 step against the ~2.8 us** one SM should
need -- roughly 38x per step. The next move is a profiler on a single tile, not another trait
audit; this entry has now spent four hypotheses on inference from aggregate counters and should
stop doing that.

**Superseded hypothesis, retained for the reasoning trace: `MmaPV_is_RS=false`.**
Upstream's own tuner, `tile_size_fwd_sm90` (`tile_size.h:33`), for head_dim 128 / bf16 returns

    {kBlockM=128, kBlockN=128, MmaPV_is_RS=true, IntraWGOverlap=true}

(`kBlockN=128` because `use_blockN_128` is true when `paged_kv_non_TMA` is set -- which is also
why our hardcoded 128 is right). We match it on `kBlockM`, `kBlockN` and `IntraWGOverlap`, and
differ on exactly one trait: we pass `MmaPV_is_RS=false`. It has been `false` since the first
port commit (`a13e87d`), where it was a guess -- never derived from upstream, never measured.

With `MmaPV_is_RS` false the softmax result P does not stay in registers for the PV mma; it
round-trips through smem, which needs a barrier per KV block and serialises the softmax against
the mma. That is precisely a mainloop that cannot overlap loads with math, and it predicts every
symptom measured above: 3% of peak FLOPs, 1.7% of HBM, and throughput perfectly linear in SM
count because each SM is independently stalled rather than contending for anything.

This is the same defect class as the four already recorded in
[upstream FA3 subset/per-item params](#upstream-fa3-param-mismatches) -- a ported template
argument set from assumption instead of from upstream.

The fix defaults both traits to upstream's values and keeps the old behaviour only as an A/B
switch (`FLASHINFER_FA3_MMA_PV_RS`, env `MMA_PV_RS`; `FLASHINFER_FA3_UPSTREAM_STAGES` default
1 -> 2). Also worth testing: `MmaPV_is_RS=true` uses *less* smem than false (upstream's comment
at `tile_size.h:35` notes 128x192 is the smem limit with RS and 128x144 without), so stages=3
may become legal once RS is on.

**Three more port deviations, found by diffing the whole instantiation against upstream.**
Having found one hardcoded trait wrong, I checked the rest against
`flash_fwd_launch_template.h:60-66` instead of waiting for them to surface:

| trait | ours was | upstream computes | consequence |
|---|---|---|---|
| `NumProducerThreads` | `128` | `(!Transpose_V && Use_TMA_KV && Use_TMA_Q) ? 32 : 128` | **no bug**: resolves to 128 on both paths, since `Use_TMA_Q = !PackGQA` and we pack GQA. Derived anyway -- it breaks the moment PackGQA is off |
| `LPT` | `true` | `Is_causal \|\| Is_local` -> **false** | an L2-locality ordering, not longest-processing-time; which wins is a measurement |
| `Sort` | `false` | `!Is_local` -> **true** | inert: only read under `Prepared && Sort`, and we run `Prepared=false` |

`NumMmaThreads` and `NumProducerThreads` now derive from `UpstreamFA3Mainloop` with
static_asserts pinning what the launch geometry assumes, so a trait change that moves either
count fails to compile rather than silently mis-sizing barriers. That is the general lesson
from this whole episode: every one of these -- and the four earlier param faults -- was a
literal written from assumption where upstream had a derivation.

**Pending measurement**, launched as `/dev/shm/w/rs.sh`: control FA2-all, RS=0 stages2,
RS=1 stages2, RS=1 stages3, on the full TraceLab batch at page 128, 5 iters. **These results
are lost: trial `40802be244f0ee7e` reached status `done` mid-run** and took `/dev/shm` with it.
They have to be re-run on a fresh pod.

**RETRACTED: there was no control-plane outage.** I diagnosed one and spent about an hour on
it -- a 25s-interval monitor, a 5-minute-interval watcher, a rate-limit theory, a token-expiry
check. The actual cause of every `EOF` was simpler, and twofold:

1. **The pod expired.** `execute-script` against a dead pod returns a bare
   `{"error": "EOF"}`, indistinguishable from a network fault. Nothing in the message says
   "gone".
2. **My manual shell calls had proxy vars set, which `px.py` strips.** `NO_PROXY`
   (uppercase) does not list `.bytedance.net` or `.byteintl.net` while lowercase `no_proxy`
   does, and Go's `ProxyFromEnvironment` prefers the uppercase one -- so merlin-cli routed
   these hosts through a proxy that fails TLS, and Go surfaced it as `EOF`. That is why a
   bare `runs get` "also failed" and convinced me the whole control plane was down. With the
   proxy vars stripped, the same `runs get` answers immediately -- which is how the expiry
   was finally found.

**And the proxy half was already written down.** `tasks/fused_specdec_workload/drivers/README.md`
lists `noproxy.sh` with the note "strips proxy variables; `merlin-cli` fails with EOF through the
proxy". It was recorded from a previous encounter, in the directory I was already using, and I
rediscovered it from scratch instead of reading it. Checking the drivers README is cheaper than
re-deriving its contents.

The two failure modes produced an identical error string, and I let the second one corroborate
a wrong conclusion about the first. `px.py` now carries `probe_trial()` and a comment on the
proxy rule, so the next `EOF` is one call away from being correctly attributed. The lesson for
the pod-restore driver: **check trial status before diagnosing the network.**

This is the third trial to expire mid-measurement (`f393156fcc0614b3`, then
`40802be244f0ee7e`). Long measurement runs need their results pulled off the pod incrementally
rather than collected at the end, since `/dev/shm` dies with the trial.

**Multicast is now unblocked but unmotivated by this entry.** TMA paged does work at
`page_size=128` (all rows PASS in the TMA arm), which was the stated prerequisite for the DSMEM
multicast item, so that item can proceed. What is retracted is the claim that the same change
"addresses the 31x long-context regression" — it does not address it at all.

<a id="fa2-kv-pipelining-measured-flat--2026-10-06"></a>
### FA2 K/V pipelining is flat on long context, because FA2 is already at roofline — 2026-10-06

Priority item 1, measured on the TraceLab batch (1.58M KV, page 128, FA2 on every row, 5
iters, `drivers/ab_fa3_traits.sh` arms `fa2k2/3/4`):

| `FA2_MMA_KV` | total | fa2 half |
|---|---|---|
| 2 (default) | 15 301.9 us | 15 218.8 us |
| 3 | 15 298.6 us | 15 296.6 us |
| 4 | 15 243.3 us | 15 219.5 us |

**Flat to 0.5% -- because `FA2_MMA_KV` is not a pipeline-depth knob.** Measured locally with
`nvcc`: `PKT<128, NUM_MMA_KV>::SharedStorage` is **133 120 bytes at `NUM_MMA_KV` = 2, 3 and
4** -- byte-identical. A deeper K/V *pipeline* would need more smem staging for in-flight KV;
unchanged smem means the knob does not add stages. `NUM_MMA_KV` sizes the register-side
`s_frag` accumulator (`[NUM_MMA_Q][NUM_MMA_KV][8]` floats per thread), i.e. how many KV MMA
steps are batched into registers per iteration -- not how deep the load pipeline is. The
comment in `fused_fa3_persistent_ext.cu:86-88` claiming it "costs shared memory" is wrong on
that point.

So **the flatness is explained, and the item is not actually tested.** The three arms varied
a register-tiling parameter, which has no reason to help a loop that is not register-starved,
and said nothing about K/V pipelining. The item goes back to **not started**, not "done, no
effect".

**Also RETRACTED, separately:** I first explained the flatness as "FA2 is at 322 TFLOP/s,
1.2x off peak, no latency left to hide", using 400 TFLOP/s as the H200 peak. The real bf16
dense peak is **989.4 TFLOP/s**, so FA2 sits at **33% of peak** and does have headroom. Both
of my explanations for this item were wrong before this one: the first used a bad peak, the
second ("unexplained") was an honest placeholder.

**To actually test the item** requires finding the real staging depth in the persistent
kernel's KV load path (the `cp.async` stage count in `SharedStorage`, which is what would have
to grow), and confirming smem moves with it before measuring any timing. A knob that does not
change `SharedStorage` cannot be the one.

This supersedes the earlier note carrying it as "worth ~15% and blocked by an smem byte
count". That estimate came from the *fused* kernel's constraints, where the FA2 slot was
competing for smem with FA3; the split removed the constraint, and with it the premise. The
item is closed: **no change to make.**

Caveat on scope: this is the long-context regime. A short-KV batch, where FA2's own tiles
are latency-bound rather than compute-bound, could still respond to depth -- that is a
different fixture and is not claimed either way here.

## Kernel optimisations — 2026-10-06

One line each; the evidence is in the linked detail section. Every item must be tested on
**both** executors — a win on FA3 says nothing about FA2.

- [x] **Two executors instead of one kernel** — split beats FA2-on-everything 1.21x and
      passes; the fused kernel was 2.16x slower. Now the default.
      [detail](#two-executors-beat-fa2-on-everything-and-pass--2026-10-06)
- [x] **Drive FA3 with upstream's own scheduler** — fixed every numerical fault at once.
      [detail](#the-plan-and-the-split-encoding-are-both-verified-correct--2026-10-06)
- [x] **K/V pipelining, FA3** — stages=2, **6.0%** on the FA3 half; the smem count that had
      it blocked was the fused kernel's.
      [detail](#kv-pipelining-stages2-measured-60-and-the-split-unlocked-it)
- [x] **K/V pipelining, FA2 — NEGATIVE, 0%** — `mma_kv` 2 -> 4 moves the FA2 half 34.5 ->
      34.7 us, inside noise. FA2 at 128 threads x 2 CTAs already holds 255 registers with
      zero spill, so it is not pipeline-starved, and doubling
      `s_frag[NUM_MMA_Q][NUM_MMA_KV][8]` eats the registers `mma.sync` needs. So this
      optimisation is FA3-only: 6.0% there, nothing here. Switch kept at 2.
- [ ] **Smem-handoff JIT split-KV reduction — it pays on FA2, not FA3** (2026-10-06). Which
      shape splits is decided by upstream's heuristic, and on the TraceLab shape it answers
      cleanly: a 2048-token append at 64K KV already has `4 * ceil(2048*7/128) = 448`
      m-blocks against `0.8 * 132 = 106`, so no split is needed and the heuristic returns 1.
      A long-context **decode** row (q=1) has 4 m-blocks against 106, so it must split.
      Those decode rows route to **FA2** (tile 16, where the class boundary put decode
      because it is bandwidth-bound), and FA2 reduces its split partials through **global**
      memory. So the smem handoff is an FA2-side optimisation for long-context decode, and
      measuring it on FA3 prefill -- which is what every fixture so far did -- can only ever
      show nothing.
- [ ] **(context) The same reduction on FA3** — only reachable on low-q/high-KV rows, which
      the router deliberately sends to FA2. Keep as the comparison point against upstream's
      combine launch, not as the main claim.
      replaces upstream FA3's separate combine launch and FA2's **global** partial
      round-trip. Intra-role, so no partial crosses executors. Only measurable on
      split-heavy shapes (few q rows, long KV); every fixture used so far has
      `num_splits=1`, where it saves nothing.
      [detail](#what-parity-means-on-a-split-batch--2026-10-06)
- [ ] **DSMEM / KV multicast** — the real prerequisite, found 2026-10-07: **our FA3 launcher
      does not launch clusters at all.** `LaunchUpstreamFA3OwnSched` ends in a plain
      `kernel<<<grid, block, smem_size, stream>>>` (`fa3_upstream.cuh`), and DSMEM requires a
      cluster, so `mapa.shared::cluster` has nothing to address. Upstream launches via
      `cutlass::launch_kernel_on_cluster(launch_params, kernel, kernel_params)`
      (`flash_fwd_launch_template.h:189`) precisely because its `ClusterShape` can exceed 1;
      our alias hardcodes `cute::Shape<_1,_1,_1>`.
      Steps (1) and (2) are **done** (`FLASHINFER_FA3_CLUSTER_M`, and
      `cutlass::launch_kernel_on_cluster` when it exceeds 1). Step (3) then hit a hard stop.

      **UPSTREAM FORBIDS A CLUSTER ON THE cp.async PATH.** `mainloop_fwd_sm90:58`:

          static_assert(Use_TMA_KV || CUTE_STATIC_V(size(ClusterShape{})) == 1,
                        "If not using TMA for KV, ClusterShape must be 1");

      and `Use_TMA_KV = !PagedKVNonTMA`, so with cp.async KV the cluster must be 1. Compiling
      `FLASHINFER_FA3_CLUSTER_M=2` fails on exactly that assert. The mechanism argument stands
      -- DSMEM peer reads (`mapa.shared::cluster`) do not require TMA -- but upstream's gather
      **implementation** is not cluster-aware, and it asserts rather than misbehaving.

      So the choice is explicit, and neither branch is free:

      - **(a) TMA KV + cluster.** Legal immediately, but TMA paged measured **10.03% slower**
        than cp.async on the fixed kernel, so multicast must clear that before it nets
        anything. The 9.1% handicap I first claimed is real after all -- not because DSMEM
        needs TMA, but because upstream couples them.
      - **(b) Make the cp.async gather cluster-aware ourselves**, lifting the assert and
        adding peer-smem KV reads. This keeps the faster gather and is the version that is
        actually *our* contribution rather than a vendored feature, but it means modifying
        upstream's mainloop rather than configuring it.

      (b) is the one worth doing if multicast is pursued: it is the only path where the
      1.33-1.59x is not immediately taxed 10%. It is also real kernel work, not a flag.
      This is mechanical rather than exploratory, and it is the one FA3-side item with a
      plausibly large payoff left (recorded 1.33-1.59x in isolation).
- [ ] **(superseded) DSMEM / KV multicast, TMA framing** — 1.33-1.59x in isolation. **No TMA prerequisite**: DSMEM
      peer reads (`mapa.shared::cluster`) work regardless of how KV reached smem, so one CTA
      can `cp.async` the KV block and the rest of the cluster reads it from that CTA's smem.
      TMA multicast is one *implementation*, not a requirement, and since cp.async beats TMA
      paged by 9.12% the item should be built on the cp.async path. Earlier entries calling
      TMA paged a prerequisite are retracted; that mistake is why this looked blocked.
      The old `FLASHINFER_FUSED_CLUSTER` flag belongs to the closed fused dispatch.
      [detail](#fa3-is-31x-slower-than-fa2-at-realistic-context-and-the-121x-does-not-generalise--2026-10-06)
- [ ] **Fused spec tree-build epilogue** — designed, not built; belongs to whichever executor
      writes the verify output. Needs the 4-mode MTP batch.
      [detail](#fused-epilogue-design)
- [ ] **Async/tiled packed copies** — `cp.async` landed; single-warp width and the per-row O
      store remain, the latter needing the indirection table.
- [x] **Packed GQA** — upstream provides it for FA3 (`PackGQA=true`); FA2's persistent kernel
      already had it. Nothing to add on either side.
      [detail](#packed-gqa-class-boundary)
- [x] **`sm_split` is a latency knob, not a throughput knob** — the budget sweep gives the
      exchange rate (1.63x FA3 cost at 64 SMs); concurrency loses at every budget because
      neither executor leaves SMs idle.
      [detail](#retracted-fa3s-cost-is-not-fixed-and-there-is-no-overlap-headroom--2026-10-06)
- [x] **Cross-role backfill — closed.** It only existed in symmetric mode, which computes FA3
      wrong, and it is the construct that costs 33 676 B of spill.
      [detail](#symmetric-retracted)
- [x] **Register and smem co-residency limits** — either role native, never both; smem over
      the opt-in limit by 1 120 B. This is what closed the fused design.
      [detail](#the-two-roles-register-costs-separated--2026-10-06)

## Structures evaluated — 2026-10-06
- [x] [Static SM pre-assignment](#static-split-confirmed) — -14.4% / -14.0% / -11.5%, no winning fraction
- [x] [Serial dispatch](#serial-dispatch-dead-end) — -48%, reverted, forecloses draft+verify fusion
- [x] [Symmetric single kernel](#symmetric-retracted) — numerically wrong, four hypotheses eliminated, results retracted
- [ ] [Unified queue](#unified-serving-perf) — numerically correct; **-66% measured on a clean tree** (0.335x throughput, 3.01x ITL), [cause found](#why-fused-is-slow) — register starvation: `mma.sync` at ~168 regs instead of ~255, from FA3's 384-thread launch bounds

## Workload and measurement setup — 2026-10-06

One line each; detail in [`realistic_workload_analysis.md`](realistic_workload_analysis.md).

- [ ] **Context length where attention is >50% of e2e** — either force-extrapolate RoPE up
      to TraceLab's max session length (agentic sessions reach 300-400k and stay under 1M,
      so extrapolating past the trained window is fair for a cost measurement) or use a
      larger model with a natively longer context.
- [ ] **TraceLab mixed scenarios, not `--dataset-name random`** — `run_mixed_workloads.py`
      carries the real shapes (`agent_plus_shortchat`, `loadsweep_*`); a `SPLIT` method is
      wired into it. Independent-prompt datasets average 242-690 tokens and omit the
      multi-turn tool-result prefills that create the repeated mixed forwards this kernel
      exists for.
- [x] **Why the 2048-token runs could not show the bar** — attention is a small fraction of
      the forward there, so a 1.21x attention win is invisible end to end regardless of
      correctness. Those runs measured in a regime where >=30% is unreachable by
      construction.
- [ ] **Validity gate on every arm** — report the fallback count beside the timings. Two
      serving A/Bs were invalid (121 then 122 silent fallbacks) and both printed
      ordinary-looking numbers; the backend now raises instead of degrading.

## Server integration — 2026-10-05

- [x] **Two-executor split is the default** — fused path opt-in via `SGLANG_FUSED_KERNEL=1`.
      [detail](#two-executors-beat-fa2-on-everything-and-pass--2026-10-06)
- [x] **Silent fallback removed** — the backend raises instead of degrading.
      [detail](#workload-and-measurement-setup--2026-10-06)
- [ ] **Long-context A/B on the real workload** — no valid serving number yet; both prior
      A/Bs were invalid.
- [ ] **4-mode MTP serving run** — EAGLE draft built, unused; blocked on `mix_with_running`.
      [detail](#four-mode-blocked)
- [ ] **The bar: >=30% TTFT/ITL, >=10% throughput** — zero data.
- [ ] **CUDA-graph capture** — needs the routing decision in device memory with exit-early
      CTAs so grid dims stay fixed.
- [x] **Per-step tracing** — `SGLANG_STEP_TRACE=1` localises the gap to MIXED steps.
      [detail](#spec-path-where-the-cpu-gpu-syncs-actually-are--2026-10-05)

## Schedule exploration — 2026-10-05
- [x] [EDF predictability, fused vs baseline](#edf-model-classes) — parity at ~1.5% MAPE with an adequate model class; the linear-fit falsification was an artefact
- [ ] SM-split / cluster-cap / chunk-size sweeps on a correct kernel — all prior sweeps used symmetric mode and are void
- [ ] Acceptance-length (EWMA) as a perf-model input — needs an MTP trace

## Multi-node schedule tests (H20 only, <= 16 GPUs, trial `vibecuda`) — not started
All four points are written up in [`pd_colocation_vs_disagg.md`](pd_colocation_vs_disagg.md); none is measured.
- [ ] Route to highest cache-hit node first, with an SLO veto from EDF queue-head requests
- [ ] Best-effort cache migration, and whether colocation encourages replication (ThunderAgent argument)
- [ ] EDF predictor with acceptance length — note the premise is measured false for this kernel, true only for the overlap kernel
- [ ] Two-way RDMA / congestion: complete graph vs bipartite, randomised source among multi-hit nodes

## Correctness and bring-up record — 2026-10-06
- [x] [Bringing the unified queue up: four faults, in order](#unified-bringup) — index spaces, packed extent, TMA addressing, plan coverage
- [x] [Why pre-assignment could not win: launch geometry](#launch-geometry) — one CTA/SM vs two, ~128 vs ~256 FA2 threads
- [x] [Device-side spec for the wgmma role](#unified-device-spec)
- [x] [The unified plan covers all rows](#unified-plan-coverage) — no row split, no `pers_row_offset`
- [x] [WITHDRAWN: no stale header copy](#withdrawn-stale-headers) — a partial push, misread as a JIT include tree
- [x] [H200 server-side A/B, 2026-10-04](#h200-ab-2026-10-04) — earlier split/cap/backfill tuning, symmetric-mode and therefore void
- [x] [Lineage and sibling documents](#lineage)
- [x] [Audit, superseded by this catalogue](#audit-superseded)

## Pod and environment — 2026-10-06

- [x] **Trial `f393156fcc0614b3` expired mid-measurement** — status `done`; the H200 pod and
      its entire `/dev/shm/w` working set are gone, taking the in-flight TraceLab-shaped
      run with it. SSH/WebShell failing was the symptom, not the cause.
- [x] **Switched to `vibecuda-h200-2gpu-replacement-20261006`** (`40802be244f0ee7e`), 2x
      H200 in the same queue, both GPUs idle, reachable. `/tmp/px.py` retargeted.
- [ ] **Re-provision the working set** — `/dev/shm/w` is empty: venv (torch 2.8/cu128),
      flashinfer tree, sglang `python/`, qwen7b weights, and flash-attention's hopper
      headers at `/dev/shm/w/flash-attention/hopper`. Drivers exist under
      `tasks/fused_specdec_workload/drivers/` (`provision.sh`, `provision_trees.sh`,
      `install_deps.sh`); the known traps are recorded there — absolute symlinks in
      `flashinfer/data`, and clearing `~/.cache/flashinfer` so the JIT does not serve stale
      objects.
- [x] **Lesson: trials expire without warning.** Every measurement should be committed as
      soon as it lands rather than batched, and a long sweep should checkpoint per arm. The
      arms lost here were re-runnable, but the TraceLab fixture had taken real work to size.

## Pod and environment — 2026-10-05
- [x] [The pod ran upstream sglang, not our fork](#env-contamination) — 4140 files against our 725, 514 differing including `bench_serving.py`; now byte-exact
- [ ] [Re-establish the GPU working set](#pod-restore) — fork tree pushed (805 files, byte-exact); three mechanical gaps left, listed in `drivers/provision_trees.sh`: spdlog unpushed, stale absolute symlinks in `flashinfer/data`, and the csrc jinja unreadable — the 2026-10-05 trial expired; a replacement single-GPU `vibecuda` is running and empty

- [ ] [The restored pod's serving tree was upstream, not our fork](#env-contamination) — rebuild in flight; no serving number may be read from it until then

## Workload capture for handoff — 2026-10-05
- [x] [Real online 2-mode trace](fused_kernel_handoff.md#5-the-workloads) — 2121 steps/arm, JSON corpus in `fused_specdec_workload/sets/`
- [ ] Real online **4-mode** trace with draft+verify co-resident — the stated gap; blocked with the 4-mode schedule
- [ ] Long-context (>64k, tracelab) A/B — written, not yet run; this is where attention dominates the step.
  Gated on [the wgmma JIT reduction](#split-kv-dark): at >64k the FA3-class rows are exactly the ones the
  plan splits, so the run would produce wrong numbers before it produced slow ones

## Detail sections — 2026-10-06

The dated checklist above is the index. What follows is evidence for the items
that have one: measurements, the faults behind them, and the reasoning that was
checked. Items without a detail section name the file that owns them.

<a id="lineage"></a>
### Lineage and sibling documents — 2026-10-05

**Lineage.** Split out of `debug_kernel_server_perf.md` on 2026-10-05, which had
reached 3428 lines spanning two unrelated efforts. That document keeps the earlier
work -- the sync-free plan, the measured mixed-kernel gate, the workload
characterisation and the mixed-tile 4-mode *dispatch*. This one continues from it
and covers only the **fused FA3 + FA2 SM pre-assignment kernel**: one launch, two
roles split by `blockIdx.x < sm_split`, just-in-time reduction in place of a grid
sync.

| doc | covers | status |
|---|---|---|
| `debug_kernel_server_perf.md` | Parts 1-4: sync-free plan, kernel gate, workload stats, mixed-tile dispatch | complete, archived |
| **`debug_kernel_server_perf_4_mode.md`** (this) | fused FA3+FA2 kernel: correctness, server integration, split/cluster tuning, backfill, multicast | active |
| `pd_colocation_vs_disagg.md` | global PD colocation vs disaggregation, the four routing/caching/EDF/network points | written, experiments pending |
| `server_perf_model.md` | MLP latency prediction for prefill-decode colocation | prior work |
| `realistic_workload_analysis.md` | dataset length stats, arrival rates, per-turn growth | prior work |

Read the sections below in order; each records what was measured, and corrections
to earlier claims are kept in place rather than edited away, because several
conclusions here were reversed by later measurement and the reversals are the
useful part.

---


<a id="h200-ab-2026-10-04"></a>
### H200 server-side kernel A/B — 2026-10-04

Hardware: 1x NVIDIA H200 143GB, 132 SMs, sm_90a (merlin trial `vibecuda`, job
`e59d6e2974d3ed90`, queue `h100-sxm-80gb.vmss-eastus2-vlmh200.agent.merge`).
sglang `c0dfe0419` (branch tree on PYTHONPATH), flashinfer `tri_tile_` @
`94909aa` in-tree. Qwen2.5-7B-Instruct, TP1, 28 qo / 4 kv heads.

```bash
# server
SGLANG_FLASHINFER_USE_PERSISTENT=1 SGLANG_MIXED_ATTENTION_KERNEL=$K \
python -m sglang.launch_server --model-path qwen7b --port 31000 \
  --mem-fraction-static 0.80 --attention-backend flashinfer \
  --enable-mixed-chunk --chunked-prefill-size 512 \
  --max-running-requests 64 --disable-radix-cache --log-level info
# load
python -m sglang.bench_serving --backend sglang --port 31000 \
  --dataset-name random-ids --random-input-len 4096 --random-output-len 512 \
  --num-prompts 300 --request-rate inf --max-concurrency 64
```

`K` in {`persistent`, `batch_prefill`}. Both arms logged an **identical forward
composition** -- 3016 mixed, 969 `decode_cuda_graph`, 15 `pure_prefill` -- so
only the mixed-batch attention kernel differs.

| | persistent overlap | BatchPrefill | delta |
|---|---|---|---|
| Total token throughput | **23,594 tok/s** | 22,576 | **+4.5%** |
| Output token throughput | 2,660 | 2,545 | +4.5% |
| Request throughput | 10.10 req/s | 9.67 | +4.4% |
| Mean TTFT | 1,327 ms | 1,373 | -3.4% |
| P99 TTFT | 4,846 ms | 5,017 | -3.4% |
| Mean ITL | 17.63 ms | 18.54 | -4.9% |
| P99 ITL | **21.20 ms** | 22.84 | **-7.2%** |

#### Three method errors worth not repeating

Forcing `SGLANG_MIXED_ATTENTION_KERNEL` is not a convenience, it is what makes
this an A/B at all. Three earlier attempts produced arms that agreed to 0.1% on
both throughput and P99 TTFT, which is not a null result about the kernels:

1. **Unsaturated operating point.** 8 req/s at concurrency 32 leaves the GPU idle
   most of the wall clock, so no kernel difference can surface. `--request-rate
   inf` with concurrency 64 was needed before anything was being measured.
2. **`--log-level warning` discarded the diagnostic.** `_count_forward` logs the
   path mix every 2000 forwards at INFO, which is exactly the instrument that
   catches error 3; running at warning threw it away.
3. **The gate routed both arms to the same kernel.** With
   `SGLANG_FLASHINFER_USE_PERSISTENT=1` and the gate active, the mix was
   `mixed_batch_prefill=3811 (63.5%)` against `mixed_persistent=83 (1.4%)` --
   both arms were running BatchPrefill. `select_mixed_attention_kernel`'s
   out-of-support guard defaults there, and a synthetic `random-ids` workload is
   outside the fitted 1,316-point region. Moving to the KV-concentration regime
   (32K x 8) only raised it to 1.85% of forwards, so 32K is still short of the
   64K the gate wants.

Each was catchable up front with one question: is the machine busy, is the code
path executing, am I varying what I think I am varying. Always read the forward
mix before believing a server A/B.

#### Scope

This verifies the **persistent overlap kernel**, which is what the analytic cost
model and the PD-colocation story in `tasks/pd_colocation_vs_disagg.md` rest on.
It does **not** verify the fused FA3+FA2 SM-pre-assignment kernel: `fused_run`
has no call site in `flashinfer_backend.py`, so that kernel never executes in
the server. Wiring it needs the four constraints the standalone harness
established -- FA3's plan capped to `sm_split`, the FA2 side's KV indices
rebased past the FA3 rows' pages, FA3 given vector-sparse offsets rather than
page ids, and FA3 confined to a contiguous batch prefix.

#### Fused FA3+FA2 on the serving path (2026-10-04)

Same H200 trial. sglang `859ff75b8`+ (fused call site), flashinfer `tri_tile_`
in-tree. Qwen2.5-7B TP1, 150 prompts, 4096/512, `--request-rate inf`,
concurrency 64, `SGLANG_MIXED_ATTENTION_KERNEL=persistent`,
`SGLANG_FORWARD_MIX_EVERY=200`.

```bash
/dev/shm/w/abf2.sh {1,0}   # SGLANG_FUSED_FA3_SM_SPLIT
/dev/shm/w/runb.sh {fusedB,ctrl}
```

| | fused path ON | control (OFF) | delta |
|---|---|---|---|
| Total token throughput | 20,260 tok/s | **22,203** | **-8.7%** |
| Output token throughput | 2,306 | 2,527 | -8.7% |
| Mean TTFT | 2,066 ms | 1,896 | +9.0% |
| P99 TTFT | 5,426 ms | 5,028 | +7.9% |
| Mean ITL | 17.30 ms | 15.52 | +11.5% |
| P99 ITL | 25.67 ms | **21.55** | **+19.1%** |

Forward mix proves what served each arm, and the arms are matched:
`mixed_persistent=796` and `pure_prefill=8` in both, with
`mixed_fused_fa3=39 (3.2%)` present only in the fused arm.

**The integration works and the kernel executes**: the extension builds through
the server's JIT, planning succeeds, and there were **0 launch failures** across
the run -- so every batch the contract accepted was served by the fused kernel
without falling back.

**This workload cannot isolate the fused kernel.** It claimed only 3.2% of
forwards, because at `--chunked-prefill-size 512` a prefill row needs >=4096
accumulated KV to clear `kv_tiles >= 32`, so a request's early chunks never
qualify. 66.3% of forwards went to the persistent kernel in both arms.

**The -8.7% is mostly planning cost, not kernel time.** Decomposing: the
end-to-end time ratio is 1.096, and if only 3.2% of forwards changed cost then
those forwards were **3.9x** more expensive. That is far worse than the
kernel-level deficit (1.345x behind a sequential co-launch), which points at
`_plan_fused_split`: it runs two extra full plans **every step** -- an FA3 plan
and a persistent plan, each with its own ~10 MB H2D copy -- while the kernel it
is planning for serves 3.2% of steps. Planning per step to use a kernel on 3% of
steps is the wrong shape; the plans should be built only when the contract has
already accepted the batch, and ideally cached across steps whose layout is
unchanged.

A prediction made before the run, recorded for calibration: 15-25% end-to-end
loss. Actual -8.7%, so the prediction was too pessimistic -- but for the wrong
reason. It assumed the fused kernel would serve all mixed forwards; it served
3.2%. Had it served all of them the loss would have been much larger than 25%.

To isolate the kernel server-side one would need prefill chunks clearing the gate
from the first chunk: a large `--chunked-prefill-size` over long prefixes. That
is the shape `tests/bench_mtp_fused.py` already drives directly and more
cleanly, which is why the kernel-level numbers remain the stronger evidence.

##### The -8.7% is not planning overhead (correction)

| | tok/s | vs control |
|---|---|---|
| control, fused path off | 22,203 | - |
| fused, original wiring | 20,260 | -8.7% |
| fused, planning deferred + indices reused + plan cached | 20,314 | **-8.5%** |

The planning optimization recovered 0.3%, i.e. nothing. **The decomposition
above it was wrong**: inferring "3.9x per affected forward, therefore planning"
from an end-to-end ratio and a 3.2% share was arithmetic applied to an
unverified assumption, namely that the only thing enabling the path changes is
the cost of the forwards it serves. Removing the planning cost falsified it.

Remaining candidates, in order of suspicion:

1. **768 MB of extra workspace** (256 MB FA3 + 512 MB persistent) allocated
   lazily on first fused plan, *after* the KV pool was sized at startup from
   `--mem-fraction-static`. It therefore competes with activations rather than
   coming out of a budget.
2. **`FLASHINFER_FA3_MAX_CTAS` leaks.** It is set in `os.environ` and never
   cleared, so after the first fused plan it caps CTAs for *every* later
   flashinfer fa3 plan in the process, not only the fused one.
3. The kernel itself, on the 39 forwards it serves.

**The decisive test, not yet run**: enable the path but force the planner to
return None. The workspaces still allocate and the env var is still set, but the
kernel never runs. If throughput stays near 20.3k the cost is environmental
(1 or 2) and the kernel is exonerated; if it returns to 22.2k the cost is the
kernel (3). That single run separates all three and should precede any further
tuning of this path.

##### Retraction: the fused kernel never ran on the serving path

Every "fused server-side" figure above (-8.7%, and the 20,314 / 20,967 / 20,880
tok/s follow-ups) was **the persistent kernel plus integration overhead**. The
fused kernel did not execute once.

The certification was broken, not just the result. `_count_forward` fired at
*plan* time, one step upstream of the launch, so 39 plans that each failed while
building the plan dict were reported as `mixed_fused_fa3=39` and read as 39
successful launches. Warning deduplication (one log per distinct message) hid the
39 failures behind a single line. Moving the counter to the launch site exposed
the truth: **declines 0, launch failures 39, `mixed_fused_fa3` absent.**

Root cause of the failures: `get_kv_buffer` returns a `(k, v)` tuple for this
pool layout rather than a fused `[N,2,...]` tensor. The same wrong assumption
appeared twice -- in `_fa3_sparse_indices` and again in the launch path -- and
fixing only the first left the second raising `tuple indices must be integers or
slices` on every launch.

With both fixed, the first genuine launch produced:

```
CUDA error: misaligned address
Fatal Python error: Aborted
```

and killed the server. A misaligned-address fault corrupts the CUDA context, so
no Python-level guard can contain it; the path must stay disabled
(`SGLANG_FUSED_FA3_SM_SPLIT` defaults to 0) until the alignment is found.

Leading hypothesis, untested: the rebased KV index pointer. `pers.plan` is given
`kv_indices[n_pages:]`, and slicing an int32 tensor at an arbitrary `n_pages`
puts the base at `base + 4*n_pages` bytes, which is only 16-byte aligned when
`n_pages % 4 == 0`. The standalone harness never caught this because its
`kvF = 8704` happens to be a multiple of 4. The launch separately passes the full
array plus `pers_kv_page_offset`, so the C++ side offsets again -- the two paths
need reconciling, and the plan should take the full array with an offset rather
than a slice.

What stands, because none of it depends on the server: the kernel-level H200
numbers, correctness (72/72 rows over 6 matrices against fp32), the backfill A/B,
the persistent-overlap server A/B (counted at the right site), and the ablation
showing the kernel contributes -0.05% server-side while the -8.5% was leaked
`FLASHINFER_FA3_MAX_CTAS` and 768 MB of late-allocated workspace.

##### Status: fused kernel launches in the server and faults in-kernel

After fixing the `(k, v)` tuple assumption in both the sparse-index helper and
the launch path, and removing the misaligned index slice (the persistent side's
`kv_indptr` is now global and the index array is passed unsliced, so
`pers_kv_page_offset` is 0):

```
plan declines: 0    python-level launch failures: 0    misaligned/abort: 1
```

So the plan succeeds, the launch is attempted with no Python error, and the
kernel faults **inside** with `CUDA error: misaligned address`, aborting the
server before the first forward-mix log. That is the first genuine launch of this
kernel in the server, and it does not survive it.

The slice was a real hazard -- measured, `int32[n]` is 16-byte aligned only for
`n % 4 == 0`, and the standalone harness's page count of 8704 is divisible by 4
so it never hit it -- but fixing it did not fix the fault, so it was not the
cause. What distinguishes the two environments is now the open question:

- the standalone harness builds `q`, `o` and the page array as fresh contiguous
  tensors; in the server `q` is a view of the model's fused QKV output and `o` is
  a slice of a reused buffer, so neither base pointer is guaranteed to share the
  harness's alignment
- the harness plans both sides over one synthetic batch, while the server's
  persistent plan is built from a global `kv_indptr` over the real paged pool

Next step is compute-sanitizer against the server process to get the faulting
instruction and operand, rather than more hypotheses -- this bug has already
survived two plausible ones.

`SGLANG_FUSED_FA3_SM_SPLIT` stays 0 by default. A misaligned-address fault
corrupts the CUDA context, so this path cannot be guarded at the Python level and
must not be enabled until the fault is understood.

##### Root cause: the page-index array, found by dumping the launch arguments

Rather than a third hypothesis, the launch arguments were dumped to disk
immediately before the faulting launch (the dump survives the abort) and read
back. Instrumentation added at the launch site in
`python/sglang/srt/layers/attention/flashinfer_backend.py`, enabled with
`SGLANG_FUSED_DUMP=<path>`.

The faulting step:

```
qo      = [424, 87, 1]        prefix = [3577, 0, 13]   ->  kv = [4001, 87, 14]
gqa     = 7                   packed_qo = [2968, 609, 7]
n_rows  = 1 (FA3)             sm_fa3 = 48
q, o, kv_indices, fa3_sparse base pointers: all 16-byte aligned
kv_indices: 3846 entries, but the batch needs 4102
            256 of those entries outside the valid page range [0, 1816955]
            first invalid at index 3590, value 1085595437
```

So no pointer was misaligned at all. The kernel was gathering **garbage page
ids** — values up to 1.1e9 against a 1.8e6 page pool — from an array that was
both too short and partly uninitialised, and faulted on the resulting address.

The cause was mine, and it was the planning optimisation from the previous
session: `_plan_fused_split_inner` reused `wrappers[0]._kv_indices` on the
stated assumption that "the mixed wrapper was just planned over the whole batch,
so it already holds exactly this array". It does not. That assumption replaced a
correct per-request build to save planning time, and the saving was ~0.3% of
throughput.

Two separate reasons this was not caught earlier, both worth recording:

- the tensor-provenance hypothesis was tested and is **false**. A new test
  (`tests/align_fused.py`) reproduces each server-side provenance in the
  standalone harness — `q` as a non-contiguous view of a fused QKV buffer, `o`
  as a slice of a reused buffer, the index array built by concatenating
  `req_to_token` rows — and all of them pass. Provenance was never the issue.
- nothing checked the array's length against the plan's own page total, so a
  wrong buffer reached the GPU as a fault instead of a declined plan.

Fix: build the array, but vectorised on the GPU (one `repeat_interleave` for the
row of each output slot, one subtract for its offset, one gather from
`req_to_token`) so correctness does not cost the planning time back, plus a hard
length check that declines the plan loudly.

##### Second bug, found while reading the dump: the tile-128 task was never run

The persistent plan is a `HolisticPlanInfo<3>` over three q-tile classes
(128 / 64 / 16). `TwoStageHolisticPlan` (`scheduler.cuh:1161`) sends every row
with `packed_qo_len > 64` to task 0, and also folds the whole tile-64 mid class
into task 0 whenever that class carries less than `FLASHINFER_TRI_TILE_MIN_KV`
(default 100000) total KV.

`tests/fused_fa3_persistent_ext.cu` built params for tasks 1 and 2 only, with
the comment "task 0 went to FA3". That is wrong: the persistent plan is built
over the FA2 sub-batch *after* the FA3 rows have been removed, so its task 0 is
ordinary FA2 work and nothing else executed it.

In the dumped batch row 1 (`qo=87`, `packed_qo=609`) lands in task 0 and was
silently skipped. Every previous harness used MTP decode rows for the FA2 side,
whose `packed_qo = 2*gqa = 14` lands in task 2, so task 0 was always empty and
the omission could not show.

Fix: a third runner `R128 = PKT<128, 2>` threaded through `fused_drain_queue`,
`FusedFA3PersistentKernel` and `FusedFA3PersistentDispatched`. Shared memory is
unaffected — tile 128 needs 49152 bytes of `SharedStorage` against tile 16's
69632, so `Runner3` still sets the bound.

Regression test: `tests/test_fused_server_batch.py` replays the dumped batch
shape exactly, so it covers FA3, task 0 and task 2 in one launch and asserts no
output row is left unwritten.

##### The fault is at launch, not in the kernel -- and the two environments use different toolchains

Re-running the server under `CUDA_LAUNCH_BLOCKING=1` moved the error from the
next sync point to the call that caused it:

```
CUDA Error: misaligned address  persistent_fa3_fused.cuh: line 420
  at function cudaLaunchKernel(kernel, grid_dims, block_dims, args, smem_size, stream)
fused FA3+FA2 launch failed, using persistent: fused kernel failed: misaligned address
...
flashinfer_backend.py line 1339, in persistent_attn_forward
    o, lse = batch_attn_wrapper.run(...)
RuntimeError: Failed to run persistent paged attention, error: misaligned address
```

So it is **`cudaLaunchKernel` itself** that rejects the launch -- the kernel never
starts -- and the Python-level guard then falls back to `batch_attn_wrapper.run`,
which fails too because a failed launch has already poisoned the CUDA context.
That second failure is what killed the server and what made earlier runs look
like a fault in flashinfer's ordinary persistent kernel. It is a consequence, not
the cause.

A launch-time misaligned address is the failure mode this header already
documents: FA3's mainloop params carry TMA descriptors that must sit 64-byte
aligned in the kernel parameter buffer.

Matched controls, concurrent traffic, `CUDA_LAUNCH_BLOCKING=1`, 16 requests of
~4000 tokens each at `chunked-prefill-size 512`:

| arm | requests | misaligned | server at end |
|---|---|---|---|
| `SGLANG_FUSED_FA3_SM_SPLIT=0` | 16/16 ok | 0 | alive |
| `SGLANG_FUSED_FA3_SM_SPLIT=1` | 0/16 | yes, at `cudaLaunchKernel` | dead |

Two earlier attempts at this control were invalid and are recorded so the mistake
is not repeated: the first sent its requests **serially**, so almost no mixed
batches formed and the fused path was never exercised; the second was not a
control at all, because `serve.sh` contained a hard `export
SGLANG_FUSED_FA3_SM_SPLIT=1` that overrode the per-run value. `serve.sh` now uses
`${SGLANG_FUSED_FA3_SM_SPLIT:-1}` so the flag is actually togglable.

**The toolchain difference is real but is not the cause.** There are two torch
extension build directories on the pod:

```
~/.cache/torch_extensions/py312_cu131/fused_fa3   <- system python3, torch 2.9.1, CUDA 13.1
~/.cache/torch_extensions/py312_cu128/fused_fa3   <- /dev/shm/w/venv, torch 2.8.0, CUDA 12.8
```

`serve.sh` activates the venv, so the server compiles and runs under CUDA 12.8
while the kernel tests had all been run under 13.1. That gap was worth closing
and is closed: `tests/test_fused_server_batch.py` run with
`/dev/shm/w/venv/bin/python` **passes** at splits 48/32/16, 0 unwritten rows,
every row within 0.0035 of the fp32 reference. So the launch is fine on the
server's own toolchain and the toolchain was not the cause.

##### Root cause: the KV cache does not have a fixed rank

The extension read `k_cache.stride(1)` as the token stride and
`k_cache.stride(2)` as the head stride. That is correct for exactly one of the
two layouts it is given:

| caller | KV tensor | strides | k_stride_n | k_stride_h |
|---|---|---|---|---|
| standalone harness | rank 4 `[pages, page_size, H, D]` | `[1024, 512, 128, 1]` | 512 | 128 (correct) |
| sglang paged pool | rank 3 `[tokens, H, D]` | `[512, 128, 1]` | 128 | **1 (wrong)** |

The harness slices a `[pages, 2, page_size, H, D]` buffer, so it is rank 4;
sglang's pool hands over a rank-3 tensor at `page_size 1`. With
`k_stride_h = 1` all four KV heads alias one head's memory, and
`get_gmem_layout(0, num_kv_heads, HEAD_DIM, 128, 1)` builds a degenerate layout
whose tensor map `cudaLaunchKernel` rejects as misaligned.

Note the page step happened to survive both layouts by accident
(`ratio * k_stride_n` is 2*512 = 1024 for rank 4 and 4*128 = 512 for rank 3,
both the correct page step), which is why only the head stride was wrong and why
the symptom was a launch rejection rather than wrong numbers.

This is a difference no harness built on the harness's own KV layout could ever
catch, which is the same structural blind spot as the tile-128 task: both bugs
exist precisely where the test fixture differs from sglang.

Fix: `kv_strides()` in the extension derives `(page, n, h)` from the tensor rank
and rejects any other rank outright, and `_fa3_sparse_indices` picks the matching
vector-sparse ratio (`stride(0)//stride(1)` for rank 4, `1` for rank 3) so the
two stay consistent.

The `kv_indices` fix is independently confirmed on the serving path: a second dump
taken after the fix shows `kv_indices len = 8192, pages needed = 8192, 0 invalid
entries` (previously 3846 vs 4102 with 256 garbage entries), and every base
pointer 16-byte aligned. That bug is fixed; the launch rejection is a separate
one.

##### The fused kernel now runs on the serving path

With the rank-derived KV strides (flashinfer `4d59914`) and the matching
vector-sparse ratio (sglang `ca09078ab`), the first clean serving run:

```
16/16 concurrent ~4000-token requests served (max_new_tokens 48)
misaligned=0   fatal=0   launch_failed=0   server alive at the end
forward mix (n=600): mixed_fused_fa3=448 (74.7%)  mixed_persistent=118 (19.7%)
                     decode_cuda_graph=25 (4.2%)  pure_prefill=9 (1.5%)
```

Command: `bash /tmp/run5.sh 1 fix2` on the H200 pod (`serve.sh` with
`SGLANG_FUSED_FA3_SM_SPLIT=1`, `SGLANG_MIXED_ATTENTION_KERNEL=persistent`,
`--enable-mixed-chunk --chunked-prefill-size 512 --max-running-requests 64
--disable-radix-cache`, qwen7b).

The `mixed_fused_fa3` count is taken at the launch site, not at plan time, so it
counts launches rather than accepted plans -- the instrument whose earlier
mis-placement produced the retracted numbers.

Two operational notes from this session worth keeping:

- `CUDA_LAUNCH_BLOCKING=1` is usable for attribution but **not** for a serving
  run: it slowed startup enough that the detokenizer heartbeat timed out and the
  server wedged before any traffic was processed. Use it only to locate a fault.
- `pkill -f` patterns must bracket a character (`sglan[g]::`) or the webshell
  kills itself, because the pattern string appears in its own command line. This
  cost several cycles earlier and recurred once this session.

##### A fifth invalid A/B, caught by the composition line

The first throughput A/B on the now-working path produced a clean-looking null
result:

| arm | total tok/s | mean ITL | p99 ITL | `mixed_fused_fa3` |
|---|---|---|---|---|
| `SM_SPLIT=1` | 18182.03 | 11.31 | 29.99 | **absent** |
| `SM_SPLIT=0` | 18203.29 | 11.05 | 28.88 | absent |

It is invalid, and the reason is in the composition line rather than the numbers:
`mixed_fused_fa3` never appears in either arm, so **both arms ran the same
kernel** and the 0.1% difference is run-to-run noise.

The cause is the workload, and the gate was behaving correctly. `bench.sh` used
`--random-input-len 2048`, while the FA3 gate needs `packed_qo >= 48` **and**
`kv_tiles >= 32`, i.e. at least 4096 accumulated KV. At `chunked-prefill-size 512`
a 2048-token prompt never accumulates more than 2048, so `kv_tiles <= 16` and no
row ever qualifies. The hand-driven traffic that first exercised the kernel used
~4000-word prompts, which is why it engaged there and not here.

So the fused path only engages on long-context prefill, which is worth stating as
a property of the gate rather than discovering again: the A/B must use
`--random-input-len` of at least ~4096, and **every** fused measurement must carry
its `mixed_fused_fa3` share or it is not evidence about this kernel.

That makes five invalid A/Bs on this path, all from the same root cause -- the arms
were not actually running different kernels. The composition counter is the only
thing that has reliably caught it, which is why it is checked before the numbers.

##### First valid server-side measurement of the fused kernel: a 8.1% regression

Same script with `--random-input-len 8192 --random-output-len 192
--num-prompts 300 --request-rate 12 --max-concurrency 32`, which clears the FA3
gate. The compositions now differ, so the two arms really did run different
kernels:

| metric | fused (`SM_SPLIT=1`) | base (`SM_SPLIT=0`) | delta |
|---|---|---|---|
| total token throughput (tok/s) | 23250.44 | **25291.44** | **-8.1%** |
| benchmark duration (s) | 54.83 | 50.40 | +8.8% |
| output token throughput (tok/s) | 556.62 | 605.48 | -8.1% |
| mean TTFT (ms) | 3443.97 | 3150.61 | +9.3% |
| p99 TTFT (ms) | 4679.48 | 4221.87 | +10.8% |
| mean ITL (ms) | 20.18 | 18.54 | +8.8% |
| p99 ITL (ms) | 31.87 | 31.62 | +0.8% |
| `mixed_fused_fa3` share | **29540 / 32400 (91.2%)** | absent | — |
| `mixed_persistent` share | 2796 (8.6%) | 2750 (98.2%) | — |
| misaligned / fatal / launch failures | 0 / 0 / 0 | 0 / 0 / 0 | — |

300/300 requests succeeded in both arms and both servers were alive at the end.

Commits: sglang `ca09078ab`, flashinfer `4d59914`. Command:
`bash /tmp/ab.sh` on the H200 pod (both arms from one `serve.sh`, toggled by
`SGLANG_FUSED_FA3_SM_SPLIT`), model qwen7b, `--chunked-prefill-size 512
--max-running-requests 64 --disable-radix-cache`,
`SGLANG_MIXED_ATTENTION_KERNEL=persistent`, `FLASHINFER_FUSED_BACKFILL=0`,
CUDA 12.8 / torch 2.8.0 (the venv the server uses).

**Reading.** The kernel is now correct and genuinely running on the serving path
at 91.2% of forwards, and it is **slower than the persistent kernel it replaces
by 8.1% of end-to-end throughput.** That is one run per arm, but the gap is two
orders of magnitude larger than the 0.1% spread between the two arms of the
earlier A/B where both ran the same kernel, so it is a real effect rather than
noise.

The direction agrees with the kernel-level measurements rather than contradicting
them: on H200 the fused kernel is 0.410x the sequential co-launch over 40 mixed
MTP steps, and 0.709x at its best split. A -8.1% end-to-end cost from a kernel
that is ~0.7x on the attention it replaces is the expected magnitude once the
rest of the step is accounted for.

So the remaining deficit is not a wiring bug; it is the structural one already
recorded above. `sm_fa3` is fixed at 48 here by `plan_dispatch_prefix`
(`MIN_SM_FA3, MAX_SM_FA3 = 8, 48`), and a split sweep is worth doing, but the
kernel-level sweep already showed the best split reaching only 0.709x, so a sweep
is very unlikely to turn -8.1% into a win on its own. The thing that would is
warpgroup-reentrant FA2, which is narrowed to a 3-line region (threading
`barrier_id` through `Run` and scoping `threadblock_sync_mdo_states`) and has been
attempted and reverted twice.

##### Optimizing the split on the real serving workload: -8.1% -> -3.6%

`sm_fa3` was pinned at 48 on every serving run because `plan_dispatch_prefix`
clamps it to `[MIN_SM_FA3, MAX_SM_FA3] = [8, 48]`, so the kernel's single most
important knob had never been varied server-side. `SGLANG_FUSED_SM_FA3` now pins
it (clamped only to `[1, num_sm-1]`, since the point of a sweep is to reach
values the heuristic would not choose).

Matched sweep, 200 prompts, `--random-input-len 8192 --random-output-len 192
--request-rate 12 --max-concurrency 32`, H200, `FLASHINFER_FUSED_BACKFILL=0`:

| `sm_fa3` | tok/s | vs base | mean ITL | p99 ITL | mean TTFT | fused share |
|---|---|---|---|---|---|---|
| 16 | 19369.38 | -22.4% | 23.77 | 45.46 | 3757.37 | 90.6% |
| 32 | 22034.74 | -11.7% | 20.93 | 35.94 | 3186.79 | 90.6% |
| **64** | **24055.80** | **-3.6%** | **19.18** | **34.74** | **2821.14** | 90.5% |
| 96 | 22856.25 | -8.4% | 20.26 | 37.05 | 3024.53 | 90.6% |
| 112 | 19457.78 | -22.0% | 23.95 | 44.89 | 3700.06 | 90.6% |
| base (fused off) | 24946.41 | — | 18.47 | 32.85 | 2671.04 | absent |

200/200 requests in every arm, 0 misaligned, 0 launch failures.

**There is a clear interior optimum at `sm_fa3 = 64`, and the heuristic's clamp
sat below it.** Moving from the heuristic's 48 to 64 recovers more than half the
regression: **-8.1% -> -3.6%**, with mean ITL 19.18 vs 18.47 ms (+3.8%) and p99
ITL 34.74 vs 32.85 (+5.8%). The curve is sharply peaked -- both 16 and 112 cost
~22% -- which is why leaving this knob unswept was costing so much.

**The analytic balance point is not usable for this knob.** Unclamped,
`plan_dispatch` returns `sm_fa3 = 131` and `115` for the two dumped serving batch
shapes, i.e. essentially the whole device on FA3 -- and the sweep measures 112 as
one of the two *worst* settings (-22.0%). So the clamp, not the model, was
choosing the value, and the model's own preference points the wrong way. Any
future tuning of this split should be measured, not derived from the overlap cost
model, which was fit for the *overlap* kernel and does not transfer to the
SM-pre-assignment kernel's makespan.

`sm_fa3 = 80` was added to bracket the peak: 23800.46 tok/s, mean ITL 19.38,
p99 ITL 35.72 -- worse than 64 on all three, so 64 is the peak and not an
artefact of the grid. `MAX_SM_FA3` is now 64 in
`python/sglang/srt/layers/attention/mixed_dispatch.py`, so the default lands on
the measured optimum instead of 4.5 points below it.

##### Cross-role backfill, tested on the server

`FLASHINFER_FUSED_BACKFILL=1` at the best split, same matched harness:

| arm | tok/s | mean ITL | p99 ITL | mean TTFT | fused share | misaligned | launch failures |
|---|---|---|---|---|---|---|---|
| `sm_fa3=64`, backfill **off** | 24055.80 | 19.18 | 34.74 | 2821.14 | 90.5% | 0 | 0 |
| `sm_fa3=64`, backfill **on** | 24064.65 | 19.18 | 34.90 | 2816.70 | 90.6% | 0 | 0 |

**+0.04% throughput -- within noise.** Backfill is worth 1.2% (H200) to 1.9-6.9%
(H20) at kernel level, and essentially nothing end-to-end here. That is
consistent rather than contradictory: backfill recovers idle SMs in the prefill
tail of a single mixed step, and at `chunked-prefill-size 512` with 32 concurrent
requests the server almost always has another step's work to overlap, so the tail
it reclaims is already covered.

**Correctness with backfill on is now clean**, which it previously was not. Under
the server's own toolchain (CUDA 12.8), on the dumped server batch shape, at
splits 64 / 48 / 32:

```
backfill=0:  0/512 unwritten, rel err 0.00032 / 0.00245 / 0.00313  PASS
backfill=1:  0/512 unwritten, rel err 0.00028 / 0.00218 / 0.00282  PASS
```

Scope of that claim: this is one batch shape -- the dumped serving one, which
covers FA3 + task 0 + task 2. The earlier backfill fault was recorded on "the one
batch shape whose FA2-class rows split their KV", which this shape does not
exercise, so **this does not retire that fault**; it shows backfill is correct on
the shape the server actually produces. `FLASHINFER_FUSED_BACKFILL` stays 0 by
default, now for a measured reason (no benefit) rather than only a correctness
one.

#<a id="multicast-open"></a>
#### Multicast: what it actually requires

Multicast is not a flag on this path, and the reason is specific. The fused
kernel's FA3 side uses `hopper/sparse_mainloop.cuh` (paged/sparse KV), where:

- line 87: `static constexpr bool USE_TMA_LOAD_KV = false;` -- KV is gathered
  with `cp.async` per page, not loaded by TMA, so there is no TMA multicast to
  enable. The dense sibling `hopper/mainloop.cuh:75` has it `true`.
- the Q descriptor is built with `select<0,2>(TileShape_QKD{}), _1{}` and the
  comment `// no mcast for Q`, i.e. a cluster dim of 1.

So enabling cluster multicast needs (a) `page_size >= CTA_TILE_KV` (128) in
sglang's KV pool, which also means threading the real page size through the fused
launch, where it is currently hardcoded to 1, and (b) replacing the sparse
mainloop's gather with a TMA load so there is a descriptor to multicast from.
(b) is a rewrite of `sparse_mainloop.cuh`, not a configuration change. This is
recorded as the precise blocker rather than as "page_size=1 is a gather", which
was true but not actionable.

##### Larger prefill chunks make the fused kernel worse, not better

Hypothesis: FA3's advantage is large q tiles, and `chunked-prefill-size 512` caps
the FA3 role's q tile at 512 tokens, so raising it should favour the fused
kernel. `--chunked-prefill-size` is now settable via `CHUNK` in `serve.sh`.
Same matched harness, `sm_fa3 = 64`:

| chunk | arm | tok/s | vs base | mean ITL | p99 ITL | mean TTFT |
|---|---|---|---|---|---|---|
| 512 | fused | 24055.80 | **-3.6%** | 19.18 | 34.74 | 2821.14 |
| 512 | base | 24946.41 | — | 18.47 | 32.85 | 2671.04 |
| 2048 | fused | 24739.03 | **-17.8%** | 45.17 | **207.74** | 572.88 |
| 2048 | base | 30081.02 | — | 36.34 | 73.47 | 395.52 |

200/200 requests, 0 faults, ~90% fused share in both fused arms.

**The hypothesis is falsified, and the direction is diagnostic.** Raising the
chunk helps the baseline a lot (24946 -> 30081, +20.6%) and the fused kernel
barely at all (24056 -> 24739, +2.8%), so the gap widens from -3.6% to -17.8%.
p99 ITL is the clearest signal: 207.74 ms against the baseline's 73.47, a 2.8x
tail.

That is the static-partition cost showing up directly. FA3 and FA2 own disjoint
SM ranges for the whole launch, so the step's makespan is set by whichever role
finishes last; a bigger chunk makes the FA3 side's work per step much larger and
the imbalance grows with it. The persistent kernel has one shared work queue, so
the same imbalance is absorbed by whichever CTA is free. Backfill is supposed to
be the answer to exactly this and is measured at +0.04% (above), because it only
reclaims SMs *after* a CTA has finished its own role's queue -- it cannot move
work off the critical role mid-flight.

So `chunked-prefill-size 512` with `sm_fa3 = 64` remains the best fused
configuration found, at -3.6%, and this result is further evidence that the
remaining deficit is the static role partition rather than any per-kernel
inefficiency.

##### The split optimum does not track the work mix, so an adaptive split will not help

The static-partition reading above predicts something testable: if the makespan is
set by whichever role is longer, then at `chunk 2048` -- where FA3's share of the
step's work is much larger -- the best `sm_fa3` should shift *upward* from 64.

It does not. Split sweep at `chunk 2048`, same matched harness:

| `sm_fa3` | tok/s | mean ITL | p99 ITL | mean TTFT | fused share |
|---|---|---|---|---|---|
| 64 | 24739.03 | 45.17 | 207.74 | 572.88 | 90.0% |
| 96 | 23087.85 | 48.70 | 271.04 | 619.93 | 90.3% |
| 112 | 18505.54 | 60.78 | 422.41 | 889.56 | 89.9% |
| base | 30081.02 | 36.34 | 73.47 | 395.52 | absent |

Monotonically worse above 64, exactly as at `chunk 512`. **The optimum is
insensitive to the work mix**, which falsifies the prediction and changes the
conclusion in two ways:

1. **A per-batch adaptive split is not the fix.** `plan_dispatch`'s balance point
   already computes the work ratio and returns 115-131 for real batch shapes; the
   sweep measures 112 as one of the worst settings at *both* chunk sizes. The
   model's error is therefore **not calibration** -- the quantity it optimizes
   (work balance between the roles) is not the quantity that determines the
   optimum. Feeding it better constants would not move it toward 64.
2. `sm_fa3 ~ 64` -- about half the device -- behaves like a property of the two
   kernels' own saturation points rather than of the batch: FA3 stops gaining from
   extra CTAs while FA2's decode rows still need roughly half the device to hold
   ITL. That is why a fixed measured constant is the right choice for this knob,
   and why `MAX_SM_FA3 = 64` is stated as measured rather than derived.

This is also the directly relevant answer to the goal's perf-model question for
EDF: the overlap cost model's **balance-point output must not be used to set this
split.** Its step-time prediction is a separate matter and still rests on the 3.1%
MAPE result for the overlap kernel, which this kernel does not inherit.

##### Lifting the persistent cluster cap: -3.6% -> -2.0%

The persistent role's cluster count was `max(16, min(2*(num_sm - sm_fa3), 96))`.
The 96 was mine, chosen to keep the float workspace inside 160 MB -- not because
96 was right. At the measured-optimal `sm_fa3 = 64` the persistent role gets 68
CTAs, so `2*(132-64) = 136` clusters is what matches its grid, and the cap was
truncating that by 30%.

`SGLANG_FUSED_PERS_CLUSTER_CAP` and `SGLANG_FUSED_PERS_WS_MB` make this testable.
At `sm_fa3 = 64`, chunk 512, cap 136 with a 320 MB workspace:

| arm | tok/s | mean ITL | p99 ITL | mean TTFT | fused share |
|---|---|---|---|---|---|
| cap 96 (old default) | 24055.80 | 19.18 | 34.74 | 2821.14 | 90.5% |
| cap 136, sample 1 | 24446.95 | 18.85 | 33.18 | 2757.48 | 90.5% |
| cap 136, sample 2 | 24395.78 | 18.90 | 35.07 | 2763.73 | 90.5% |
| base (fused off) | 24946.41 | 18.47 | 32.85 | 2671.04 | absent |

200/200 requests, 0 misaligned, 0 plan declines in every arm. Two samples at cap
136 agree to 0.2%, so the **+1.5%** is larger than the repeat spread, and it
improves throughput *and* every latency statistic -- mean ITL 18.85-18.90 against
the baseline's 18.47 is now within 2.3%.

**The gap is now -2.0%** (24421 mean vs 24946), down from -8.1% at the start of
this round. Defaults moved to cap 160 and a 320 MB persistent workspace; the
extra ~160 MB is the price, and it is allocated lazily on the first accepted
mixed batch.

Progression of the fused path on this workload, all matched, all with the
`mixed_fused_fa3` share checked:

| configuration | tok/s | vs base |
|---|---|---|
| heuristic `sm_fa3=48`, cap 96 | 23250.44 | -8.1% |
| swept `sm_fa3=64`, cap 96 | 24055.80 | -3.6% |
| swept `sm_fa3=64`, cap 136 | 24421 (2 samples) | **-2.0%** |

##### The split optimum is robust to the cluster cap

The two knobs interact arithmetically -- the cluster count is
`min(2*(num_sm - sm_fa3), cap)`, so a lower split buys more clusters -- which
made it worth re-checking whether lifting the cap moved the optimum. It does not.
Re-sweep at `cap 160`, chunk 512:

| `sm_fa3` | clusters it gets | tok/s | mean ITL | p99 ITL | mean TTFT |
|---|---|---|---|---|---|
| 48 | 160 (of 168 wanted) | 23347.53 | 19.72 | **32.43** | 2949.07 |
| **64** | 136 | **24421** (2 samples) | **18.85-18.90** | 33.18-35.07 | 2757-2764 |
| 80 | 104 | 23873.03 | 19.33 | 33.60 | 2852.68 |
| base | — | 24946.41 | 18.47 | 32.85 | 2671.04 |

200/200 requests, 0 faults, ~90.6% fused share throughout. `sm_fa3 = 64` wins on
throughput, mean ITL and TTFT despite getting *fewer* clusters than 48, which
reinforces the earlier reading that 64 is about the two kernels' saturation
points rather than about resource balance. `sm_fa3 = 48` does take the best p99
ITL (32.43, slightly under the baseline's 32.85), so a deployment optimizing p99
rather than throughput would pick 48 -- worth noting, but the default stays 64.

#### Final state of the fused path on this workload

| configuration | tok/s | vs base | mean ITL | p99 ITL |
|---|---|---|---|---|
| heuristic `sm_fa3=48`, cap 96 | 23250.44 | -8.1% | 20.18 | 31.87 |
| swept `sm_fa3=64`, cap 96 | 24055.80 | -3.6% | 19.18 | 34.74 |
| **`sm_fa3=64`, cap 136 (default)** | **24421** | **-2.0%** | 18.87 | 33.18-35.07 |
| base (fused off) | 24946.41 | — | 18.47 | 32.85 |

Everything above is matched, 200 prompts, `--random-input-len 8192
--random-output-len 192 --request-rate 12 --max-concurrency 32`, qwen7b on H200,
CUDA 12.8, `FLASHINFER_FUSED_BACKFILL=0`, and every row's `mixed_fused_fa3` share
was checked before its numbers were read.

Commits: sglang `725af14b3` (and `5b7d13c13`, `576d5d611`, `31b5289dd`),
flashinfer `4d59914`.

**What is left, and why it needs a kernel change rather than tuning.** Both host
knobs are now at measured optima and the two falsified predictions above
(larger chunks, adaptive split) rule out the remaining host-side ideas. The
residual -2.0% is the static role partition: backfill runs FA3 -> FA2 only,
because an FA2-role CTA returns two of its three warpgroups
(`if (threadIdx.x >= 128u) return;`) and so can never execute FA3's warp-specialised
body. Making the reverse direction possible needs FA3's work to be claimable from
a shared cursor rather than bound to `blockIdx.x`, i.e. a dynamic FA3 queue --
which also removes the `fa3_num_ctas == sm_split` contract that currently guards
against silently dropping FA3 work. That is a redesign of the FA3 side's
scheduling, not a parameter.

##### Does the fused kernel help EDF predictability? Measured: no

The goal asks how well the model-level perf model works with this kernel for EDF
against the unfused baseline. The operational form of that question is whether the
fused kernel produces a **tighter step-time distribution** -- a deadline predictor
can only be as accurate as the dispersion of what it predicts.

Matched runs, chunk 512, `sm_fa3 = 64` / cap 136 vs fused off, 200 prompts each:

| metric | fused | base | ratio |
|---|---|---|---|
| mean ITL (ms) | 18.854 | 18.459 | 1.021 |
| median ITL (ms) | 18.988 | 18.373 | 1.033 |
| std ITL (ms) | 5.025 | 4.727 | 1.063 |
| p95 ITL (ms) | 24.593 | 22.645 | 1.086 |
| p99 ITL (ms) | 33.182 | 34.994 | **0.948** |
| **CV = std/mean ITL** | **0.266** | **0.256** | 1.041 |
| **CV = std/mean TPOT** | **0.101** | **0.099** | 1.020 |
| p99 / median ITL | 1.747 | 1.905 | 0.918 |
| p95 / median ITL | 1.295 | 1.232 | 1.051 |

**The answer is no, and it is a clean null rather than a loss.** The coefficient
of variation is 0.266 vs 0.256 on ITL and 0.101 vs 0.099 on TPOT -- a 2-4%
difference. The tail ratios disagree in sign (p99/median 8.2% *tighter* fused,
p95/median 5.1% wider), which is the signature of noise rather than an effect.

For calibration on what counts as noise here: two identical baseline runs gave
24946.41 and 24951.21 tok/s (0.02% apart) but p99 ITL of 32.85 and 34.99 -- a
6.5% spread. **p99 ITL is the noisiest statistic in this harness**, so none of the
p99 differences above are interpretable, including the fused arm's apparent
advantage.

**This corrects an earlier statement in this document.** The fused arm's p99 ITL
was reported as worse than the baseline's (34.74 / 33.18 against 32.85). With the
second baseline sample at 34.99, p99 ITL is **indistinguishable between the two
arms**; only throughput (-2.1%, far outside its 0.02-0.21% repeat spread) and mean
ITL (+2.1%) separate them.

**Consequence for the colocation story.** `tasks/pd_colocation_vs_disagg.md` section
3 rests on the *overlap* kernel's closed-form predictability (3.1% MAPE), and that
document already states explicitly that "everything below depends on the
predictability property, not on the SM-pre-assignment kernel winning." This
measurement is direct evidence for keeping those separate: **the SM-pre-assignment
kernel does not inherit the overlap kernel's predictability advantage**, and the
EDF argument must not be re-attributed to it.

Limitation, stated rather than glossed: ITL dispersion is an end-to-end proxy for
step-cost predictability, not a measurement of model error. A direct test fits a
cost model to per-step (batch composition -> step time) samples and reports MAPE,
which needs per-step instrumentation in the scheduler that does not exist yet.
What is established here is that the fused kernel's end-to-end latency dispersion
is not better, which is the necessary condition for the stronger claim and it
fails.

#<a id="four-mode-blocked"></a>
#### The 4-mode MTP schedule does not exist in sglang today

The goal specifies that real-serving validation assume "an MTP spec model with
common acceptance rate in a 4-mode fused schedule". Every server run recorded
above had `speculative_algorithm=None`, so this was tested directly, using NGRAM
speculative decoding (no draft checkpoint needed) with
`--speculative-num-draft-tokens 2`, which gives a verify row the MTP k=1 shape
`packed_qo = 2 * gqa = 14` -- the tile-16 class the kernel-level MTP benchmarks
used.

The server came up and served, and the forward composition was:

```
forward mix (n=1400): pure_prefill=1400(100.0%)
batch kinds scheduled: 1463 Prefill batch, 11 Decode batch
mixed_fused_fa3: absent -- zero fused launches
```

The reason is in sglang, not in the kernel or the configuration:

```
WARNING server_args.py:1584: The overlap scheduler and mixed chunked prefill are
disabled because of using ngram speculative decoding.
-> enable_mixed_chunk=False, despite --enable-mixed-chunk being passed
```

This is **not ngram-specific**. `enable_mixed_chunk = False` is set
unconditionally on every speculative path:

| site | applies to |
|---|---|
| `server_args.py:1505` | EAGLE / EAGLE3 / STANDALONE ("Mixed chunked prefill is disabled because of using eagle speculative decoding") |
| `server_args.py:1578` | NGRAM |

**So with any speculative algorithm, sglang never puts a prefill chunk and a spec
verify row in the same batch.** The fused FA3 + FA2 kernel exists precisely to
serve that batch -- a large-tile prefill row on the FA3 side and small-tile
verify/decode rows on the FA2 side -- so on the real MTP serving path it currently
has no workload at all.

Consequences, stated plainly because they bound every server-side number in this
document:

1. The **-2.0%** result and the whole split/cluster/backfill tuning above were
   measured in the only regime where this kernel can run today: mixed chunked
   prefill with speculative decoding **off**. That is a prefill-chunk + decode
   mix, not the 4-mode MTP mix the goal specifies.
2. The kernel-level 4-mode MTP validation does exist and stands
   (`tests/bench_mtp_fused.py`, verify rows at `packed_qo = 14` mixed with
   prefill; H200 mono 34,301 / seq 25,503 / fused 83,612 / best-split 48,397).
   The gap is specifically the server path.
3. Getting the goal's regime requires making mixed chunked prefill and speculative
   decoding coexist in sglang's scheduler -- removing those two assignments and
   handling a batch that contains chunked prefill rows *and* verify rows. That is
   a scheduler change upstream of this kernel, and it is a prerequisite for the
   kernel's intended workload rather than an optimization of it.

This was found by running it rather than by reading the scheduler, which is why it
is recorded with the forward-composition evidence: `pure_prefill=100%` is
unambiguous, and the composition counter caught it the same way it caught the five
invalid A/Bs.

##### Forcing the 4-mode schedule: the exact blocker in sglang's scheduler

The three config gates were relaxed behind `SGLANG_ALLOW_MIXED_CHUNK_SPEC=1`
(default off, so stock behaviour is untouched):

| gate | site | kind |
|---|---|---|
| EAGLE / EAGLE3 / STANDALONE | `server_args.py:1505` | assignment `enable_mixed_chunk = False` |
| NGRAM | `server_args.py:1578` | assignment `enable_mixed_chunk = False` |
| validation | `server_args.py:3673` | `assert not self.enable_mixed_chunk` -- **and its message is written backwards**, reading "enable_mixed_chunk is required for speculative decoding" while asserting it must be off |

With all three relaxed the server **starts** cleanly (`KEPT ON with ngram
speculative decoding`, no assertion, no fatal, no misaligned) and then **dies on
the first real request**:

```
scheduler.py:2247        new_batch.mix_with_running(self.running_batch)
schedule_batch.py:1410   out_cache_loc = torch.cat(
                             [self.out_cache_loc, running_batch.out_cache_loc])
TypeError: expected Tensor as element 1 in argument 0, but got NoneType
```

So the blocker is concrete and named: **`mix_with_running` assumes the running
batch has `out_cache_loc` allocated, and under speculative decoding it is
`None`** -- the spec worker allocates KV slots during draft/verify rather than
through the scheduler's normal decode path, so there is nothing for the mixed
batch to concatenate.

This is a much more actionable statement than "sglang disables the combination":
the gates are not merely conservative, they guard a real unimplemented path, and
the work needed to reach the goal's 4-mode schedule is to teach
`mix_with_running` how to merge a spec-decoding running batch -- deciding where
the verify rows' KV slots come from when their batch is folded into a chunked
prefill batch.

Not attempted here, deliberately: getting KV slot allocation wrong in that merge
corrupts the cache silently rather than crashing, and validating it needs the
greedy-equivalence harness plus per-row KV auditing. The equivalence harness is
built and was used for this run (`/tmp/mode4.sh`: the no-spec greedy reference is
captured first and compared against the 4-mode output; here the 4-mode output was
empty because the scheduler died, which the comparison correctly reported as a
mismatch rather than a pass).

**Status of the goal's MTP requirement, stated plainly.** The 4-mode MTP serving
schedule is unreachable on sglang today: not by configuration, and not by
relaxing the gates either. It needs a scheduler fix at
`schedule_batch.py:1410`. Every server-side number in this document is therefore
from the prefill-chunk + decode mixed regime with speculative decoding off, which
is the only regime in which this kernel currently runs at all. The kernel-level
4-mode MTP validation (`tests/bench_mtp_fused.py`, verify rows at
`packed_qo = 14`) remains the only 4-mode evidence, and it stands.

##### Why the `mix_with_running` fix is a feature, not a patch

Reading `schedule_batch.py:1400-1427` after the crash shows the `out_cache_loc`
`None` is the *first* of four assumptions that all encode "one new token per
running request", which is true for ordinary decode and false for a spec verify
batch (each running request contributes `num_draft_tokens`, 2 at MTP k=1):

| line | assumption | why it breaks under spec decoding |
|---|---|---|
| `req.extend_input_len = 1` | one token per running req | a verify row has `num_draft_tokens` |
| `torch.cat([..., running_batch.out_cache_loc])` | slots already allocated | `None`; the spec worker allocates at verify time |
| `self.extend_lens.extend([1] * running_bs)` | one token per running req | same as above |
| `prefix_lens` from `len(origin_input_ids) + len(output_ids) + delta` | exactly one new token | off by `num_draft_tokens - 1` |

And beyond those four, the merged batch carries **no `spec_info`** -- the draft
tree and its verification mask -- so even with the lengths fixed the attention
backend would have no way to build the correct causal mask for the verify rows
inside a `ForwardMode.MIXED` forward.

So making the goal's 4-mode schedule real means: allocating verify rows' KV slots
inside the mixed batch, generalising four length computations from 1 to
`num_draft_tokens`, threading `spec_info` through `ForwardMode.MIXED`, and teaching
the attention backend to apply a tree mask to a subset of rows in a mixed batch.
That is a feature in sglang's scheduler and attention path, upstream of this
kernel.

**I am not attempting it**, and the reason is specific rather than budgetary
reluctance: every one of those four is a silent-corruption failure mode -- wrong
`prefix_lens` or wrong KV slots produce plausible tokens from the wrong cache
rather than a crash, and a short greedy probe can pass while a loaded run is
quietly wrong. Doing it properly needs the equivalence harness (built, at
`/tmp/mode4.sh`) plus per-row KV auditing, and it should be a separate, focused
change with its own validation rather than a tail-end addition to this one.

The override that exposes the path stays in the tree, defaulted off
(`SGLANG_ALLOW_MIXED_CHUNK_SPEC=1`), because it is the instrument that located all
of this and whoever picks the feature up will need it.

##### Attempt at symmetric backfill (FA2-role CTAs running FA3): implemented, faults, gated off

The measured residual is the one-directional backfill, so this implements the
missing direction as `FusedFA3PersistentSymmetricKernel`
(`FLASHINFER_FUSED_SYMMETRIC=1`, default off):

- CTAs `>= sm_split` drain the FA2 queue on warpgroup 0 while warpgroups 1-2 park
  at `__syncthreads()`, then all 384 threads enter `fa3_prefill_body` together.
  The order is FA2-**first** deliberately: a rendezvous *after* the FA3 body is
  undefined because its warp specialisation lets warpgroups exit at different
  times, while before the body nothing has exited.
- FA3 is planned for the whole grid so those CTAs have a slice to execute, and the
  dispatch's contract switches from `fa3_num_ctas == sm_split` to
  `fa3_num_ctas == num_sm`. That check fired correctly on the first two attempts
  (the harness was still declaring `sm_split` as the planned count), which is the
  guard working as intended rather than a failure.

**Result: it launches and then faults in-kernel.**

```
torch.AcceleratorError: CUDA error: an illegal instruction was encountered
```

"Illegal instruction" is **consistent with** a `setmaxnreg` protocol violation,
which would make this the third instance of that hazard class in this kernel --
but that attribution is a hypothesis, not a measurement. An attempt to confirm it
by reading ptxas register counts for the two kernels (`nvcc -Xptxas -v` on the
extension) was made twice and did not produce them: first an include-path error
for the JIT-generated `batch_attention_config.inc`, then, with that located via
`find`, a standalone compile that emitted no `Compiling entry function` blocks --
this translation unit does not build cleanly outside the torch JIT environment
that normally assembles its flags. So **the cause is not established**. The counts were finally obtained with
`tests/build_only_fused.py`, which builds with `-Xptxas -v` and never launches a
kernel (the earlier attempts failed on an include path, on the TU not compiling
standalone, and on the kernel's abort truncating the log):

| kernel | registers | spill stores (per instantiation) |
|---|---|---|
| asymmetric | **168** | 2508 / 30752 / 2296 / 1988 / 1988 |
| SYMMETRIC | **168** | 2568 / 19264 / 2264 / 3020 / 3020 |

**This falsifies the register-budget hypothesis.** Both kernels compile to an
identical 168 registers per thread, and the symmetric variant does not spill more
(in the heaviest instantiation it spills *less*, 19264 against 30752). So the
fault is not `alloc<240>` becoming unsatisfiable through a larger footprint, and
the surviving explanation is a **`setmaxnreg` protocol violation**.

The arithmetic says why that protocol has no margin here. At a 168-register
baseline over three warpgroups, FA3's producer deallocating to 24 releases 144
registers, and its two consumers allocating to 240 claim 72 each -- exactly 144.
The protocol balances to the register, so it tolerates no deviation in which
warpgroup issues which instruction or in what order. In symmetric mode warpgroup 0
executes the FA2 mainloop, with its own named-barrier syncs that warpgroups 1-2
(parked at `__syncthreads()`) never reach, so the three warpgroups enter FA3's
body having run different instruction streams before its dec/inc sequence begins.

That makes the fix direction concrete rather than speculative: the FA2 drain
cannot happen on a warpgroup that afterwards participates in FA3's register
protocol. Either FA3 must be retuned so the drain uses a warpgroup it excludes, or
FA3 needs a dynamic work queue so no CTA changes role mid-kernel -- the second
being the option that removes the protocol constraint entirely rather than working
around it.

A third option follows directly from the 144 = 144 arithmetic and looks cheaper
than either: make the pre-FA3 phase **symmetric across all three warpgroups**
instead of giving it to warpgroup 0 alone. `fused_drain_queue` is already
parameterised by `tid_base`, so warpgroups 0/1/2 can each drain the FA2 queue
independently at `tid_base` 0/128/256, after which all three have executed the
same instruction stream and FA3's dec/inc sequence begins from a symmetric state
-- which is precisely the condition the measurement says is violated.

One prerequisite if that is tried: `fused_drain_queue` currently does
`NamedBarrier::sync(128, kFusedQueueBarrier)`, and a named barrier is CTA-wide, so
three warpgroups arriving at the same id each expecting 128 arrivals would
mis-match. Each warpgroup needs its own barrier id (`kFusedQueueBarrier + wg`),
which is why this is a real change and not a one-liner -- but it is a much smaller
change than retuning FA3's warp specialisation or writing a dynamic FA3 queue, and
it is the one most directly implied by the register measurement. The two candidate mechanisms it would distinguish are a protocol
violation (a warpgroup joining FA3's dec/inc sequence after running unrelated
code) and a register *budget* problem (the combined FA2+FA3 footprint forcing a
baseline that makes `alloc<240>` unsatisfiable). Confirming which is the first
thing to do before changing the design.

For context, the prior two instances of the hazard class were: the first was
`setmaxnreg` across an ABI boundary (the `__noinline__` FA3 body, which corrupted
adjacent output row pairs), the second was `MergeGroupJIT` behind a call. Here
warpgroup 0 runs the FA2 mainloop -- with its own register pressure, at the
default allocation -- and then joins FA3's `warpgroup_reg_dealloc<24>` /
`warpgroup_reg_alloc<240>` protocol mid-kernel. `.inc` can only take registers
another warpgroup released with a matching `.dec`, so a warpgroup that arrives
having already executed unrelated code does not satisfy the protocol's
precondition.

The asymmetric default path is **unaffected and still correct**: the same build
passes the regression test at `sm_split = 64` (0/512 unwritten, rel err 0.00021 /
0.00163 / 0.00241), so this is an additive, gated variant rather than a regression.

Next step for whoever picks it up, now that the failure is specific: the FA2
mainloop and FA3's body cannot share a warpgroup's register protocol, so the
parked warpgroups must enter FA3 *without* warpgroup 0 having run FA2 first.
That means either dedicating warpgroup 0 of an FA2-role CTA to the queue for the
whole kernel while 1-2 wait and then run FA3 as a 256-thread configuration (which
needs FA3 retuned for two warpgroups), or giving FA3 a dynamic work queue so no
CTA ever changes role mid-kernel -- which is the design the earlier sections point
at and the only one that avoids the register protocol entirely.

##### Final attempt: symmetric 3-warpgroup drain — default path verified, variant still faults

Acting on the resolved diagnosis, the symmetric kernel was changed so **all three**
warpgroups drain the FA2 queue before FA3 (each with its own cursor, merge flag and
named barrier), making the pre-FA3 instruction stream identical across them -- the
condition the 168/168 register measurement says is violated.

**Both results did arrive, and they settle two things.**

*The shared-path risk is retired.* The asymmetric control **passes** with the
changed `fused_drain_queue`: `sm_split=64`, 0/512 unwritten, rel err 0.00024 /
0.00209 / 0.00294. The derived barrier ids, the `kWg * 16` slot offsets and the
16 -> 48 byte tail do not disturb the default path, as inspection suggested but
had not established. No revert is needed.

*And the protocol hypothesis is falsified too.* With all three warpgroups draining
-- so the pre-FA3 instruction stream **is** identical across them -- the symmetric
kernel still faults with the same `illegal instruction`, reproduced in two
independent runs. That was the surviving explanation after the register
measurement, and it is now gone:

| hypothesis | status |
|---|---|
| register budget (`alloc<240>` unsatisfiable through a larger footprint) | **falsified** -- 168 registers in both kernels, symmetric spills no more |
| warpgroups entering FA3's dec/inc sequence from asymmetric states | **falsified** -- made symmetric, still faults |

So the cause is none of what was proposed, and the remaining candidates are
different in kind:

1. **`setmaxnreg` may not be legal after *any* prior code in the kernel** -- i.e.
   a warpgroup's register protocol must begin at entry, not mid-kernel. If so the
   whole FA2-then-FA3 ordering is impossible regardless of symmetry, and the
   dynamic FA3 queue is the only route left.
2. **`__syncthreads()` is `barrier.sync 0`, which FA3 also uses as
   `kQueryEmpty = 0`.** The rendezvous completes before FA3's body starts, but
   FA3's mbarrier initialisation may assume a pristine barrier-0 state.
3. **The FA2 mainloop writes FA3's shared-memory region** (they overlap unless
   `FLASHINFER_FUSED_DISJOINT_SMEM`), and FA3's body may not re-initialise every
   mbarrier it then waits on.

(2) and (3) are both cheap to test -- move the rendezvous to a named barrier above
`FirstUserBarrier` instead of `__syncthreads()`, and separately try the disjoint
shared-memory layout -- and they would discriminate (1) from the rest. That is the
next step, and it is a different and smaller question than the two that were just
ruled out.

**This matters more than an unfinished experiment usually would, because the change
touches shared code on the default path.** `fused_drain_queue` now derives its
named-barrier id from `kTidBase` (`kFusedQueueBarrier + kTidBase/128`) and offsets
its shared cursor/merge slots by `kWg * 16`, and the reserved shared tail grew from
16 to 48 bytes. The asymmetric kernel uses the same function.

By inspection that is equivalent-or-safer for the asymmetric path: only one
warpgroup drains per CTA there, so distinct barrier ids and distinct slots cannot
collide, and the tail only grew (and `smem_size` was updated to match). **But
"sound by inspection" is exactly what produced two of this session's bugs** -- the
`kCursorOff` sizing taken from the wrong runner, and the fixed-rank KV stride
assumption. So it is not to be trusted without the test.

Reproduce either arm with:

```
FLASHINFER_FUSED_SYMMETRIC=0 SPLITS=64,48,32 python tests/test_fused_server_batch.py
FLASHINFER_FUSED_SYMMETRIC=1 FA3_PLAN_CTAS=132 SPLITS=64 python tests/test_fused_server_batch.py
```

##### Moving the rendezvous off barrier 0 changes the failure mode

`__syncthreads()` compiles to `barrier.sync 0` and FA3 uses barrier 0 as
`NamedBarriers::kQueryEmpty`, so the symmetric kernel's pre-FA3 rendezvous was
moved to `kFusedQueueBarrier + 3` (flashinfer `eff4970`; confined to the symmetric
kernel, so the asymmetric path is untouched by this edit).

The symptom changed:

| rendezvous | symptom |
|---|---|
| `__syncthreads()` (barrier 0) | `CUDA error: an illegal instruction was encountered`, reproduced in 3 runs |
| named barrier 11 | no traceback and **no result table** -- the test printed its 3 header lines and produced nothing further under a 400 s timeout, reproduced in 2 independent runs (so deterministic, not flaky) |

The asymmetric control passes in the same build (0/512 unwritten, rel err 0.00029 /
0.00240 / 0.00288), as expected since this edit is confined to the symmetric
kernel -- worth stating because the previous change *did* touch the shared helper.

Stated carefully, because the distinction matters: the absence of a traceback is
**consistent with a hang**, and a hang is what a `NamedBarrier::sync(384, ...)`
produces when fewer than 384 threads arrive. It is not proof of one -- the run was
under `timeout 400` and no explicit timeout message was captured.

What this does establish is that **barrier 0 was genuinely implicated** -- the
illegal instruction is gone -- but that it was not the only problem. The most
likely remaining issue is arrival count at the new barrier: all three warpgroups
call `fused_drain_queue` before the rendezvous, and that function's own loop exits
via `break` when its cursor passes `n_clusters`, so warpgroups can leave the drain
at different iterations but should all still reach the rendezvous. Any path where
one does not -- an early `return` inside a `Runner::Run`, for instance -- deadlocks
all 384.

Next step, and it is now a narrow one: confirm hang vs fault by running the
symmetric arm with no timeout and `CUDA_LAUNCH_BLOCKING=1`, and if it hangs, dump
the stack with `cuda-gdb` or add a counter to verify all three warpgroups reach
the rendezvous. This is a materially smaller question than either of the two
hypotheses that were falsified, and it is the fourth distinct mechanism this fault
has presented -- which is itself the reason to instrument rather than hypothesise
a fifth.

##### Confirmed: it is a hang, and the barrier ids collide

```
exit_code=124   (timeout)        GPU utilization 99%
```

A 124 exit with the GPU pegged at 99% is a spin on a barrier, not a fault. So the
symmetric variant's remaining problem is a **deadlock**, and the illegal
instruction really was the barrier-0 collision (now fixed).

The deadlock has an identified cause, and it is in the ids I assigned:

- the three per-warpgroup queue barriers were given `kFusedQueueBarrier + 0..2`,
  i.e. **8, 9, 10**;
- but `kPersistentMainloopBarrier = kFusedQueueBarrier + 1` = **9**
  (`prefill.cuh`), and `threadblock_sync_mdo_states` syncs on it with
  `NamedBarrier::sync(KTraits::NUM_THREADS = 128, kPersistentMainloopBarrier)`.

So warpgroup 1's queue barrier is the same id the persistent mainloop uses, and
with three warpgroups now running the mainloop concurrently they all arrive at
that one id each expecting 128 arrivals. A named barrier is CTA-wide, so the
counts mis-match and the CTA spins -- exactly what 99% utilization with no
progress looks like.

Two things are needed, not one, and both are now specified:

1. Move the per-warpgroup queue barriers clear of `kPersistentMainloopBarrier` --
   e.g. base them at `kPersistentMainloopBarrier + 1` so the three occupy 10, 11,
   12, with the rendezvous at 13. (There are 16 hardware barriers, so this fits.)
2. Make `kPersistentMainloopBarrier` itself per-warpgroup, since three warpgroups
   now run `threadblock_sync_mdo_states` at once. That means threading a barrier
   id into it the same way `barrier_id` was threaded through `Run` earlier, and it
   is the part that was attempted and reverted twice before -- but the reason is
   now concrete (concurrent warpgroups sharing one id) rather than exploratory.

This is the fifth mechanism this fault has presented, but unlike the previous four
it is identified rather than hypothesised: the hang is confirmed by exit code and
GPU state, and the id collision is a matter of arithmetic on two constants that
are both in the tree.

##### The barrier fix is verified safe but does not resolve the hang — and the approach is self-defeating

```
asymmetric   rc=0     PASS, 0/512 unwritten, rel err 0.00024 / 0.00207 / 0.00253
symmetric    rc=124   still deadlocked
```

The shared-path risk is retired: per-warpgroup mdo barriers (9/10/11) and queue
ids moved to 12/13/14 leave the asymmetric default path correct. But the symmetric
variant still hangs, so the barrier-id collision was not the cause -- or not the
only one.

**The reason is structural, and it makes this whole approach self-defeating.** The
3-warpgroup drain has all three warpgroups executing `Runner::Run` *concurrently
and on different work items*. The persistent mainloop uses **block-wide**
synchronisation internally (`__syncthreads()`, and the JIT merge's block-wide
arrival logic). Block-wide means all 384 live threads: warpgroup 0 at its first
queue slot and warpgroup 1 at its third would both arrive at the same block-wide
barrier and pair up with each other, desynchronising the two and deadlocking.

That is precisely the **warpgroup-reentrant FA2** problem recorded at the very
start of this task as the blocker -- the one attempted and reverted twice. So the
3-warpgroup drain does not avoid that problem; it *requires* it to be solved
first. It was the wrong fix, and the hang is the mainloop refusing to be run by
three independent warpgroups, not a barrier id.

This closes the loop on the whole line of attack:

| variant | requires | outcome |
|---|---|---|
| drain on warpgroup 0 only, then FA3 | nothing extra | faulted (illegal instruction) and that was barrier 0; after fixing it, untested in this form |
| drain on all three warpgroups, then FA3 | **warpgroup-reentrant FA2** | deadlocks, because the mainloop syncs block-wide |

So the two routes are: (a) revert to the single-warpgroup drain now that barrier 0
is fixed and see whether it still faults -- cheap, and it was never retested after
that fix; or (b) make the persistent mainloop warpgroup-reentrant, which is the
original blocker and gates everything here including a dynamic FA3 queue.

(a) is one env-var-sized experiment and should be done first. It was skipped
because the 3-warpgroup drain was built on top of the barrier-0 fix in the same
step -- the two changes were not separated, which is the process error here.

##### The symmetric FA2 -> FA3 backfill kernel is CORRECT

```
symmetric  (WG0-only drain + rendezvous on named barrier 15)   rc=0
  sm_split=64: unwritten (NaN) rows = 0/512
  row 0 (FA3    qo= 424 kv= 4001) rel err 0.00020  ok
  row 1 (task0  qo=  87 kv=   87) rel err 0.00194  ok
  row 2 (task2  qo=   1 kv=   14) rel err 0.00282  ok
  PASS
asymmetric control                                             rc=0  PASS
```

So the missing backfill direction works: **FA2-role CTAs now execute FA3 work**,
with FA3 planned across the whole grid, every output row correct against the fp32
reference and nothing left unwritten.

The whole fault was the **barrier 0 collision** -- `__syncthreads()` compiles to
`barrier.sync 0` and FA3 uses barrier 0 as `kQueryEmpty`. Nothing else. The four
other mechanisms investigated were all wrong:

| mechanism | verdict |
|---|---|
| register budget (`alloc<240>` unsatisfiable) | falsified -- 168 registers in both kernels |
| warpgroups entering FA3 from asymmetric states | falsified -- symmetric drain still failed |
| queue/mdo barrier id collision (`8/9/10` vs `9`) | real but not the cause; fix verified harmless |
| the 3-warpgroup drain | **actively harmful** -- deadlocks, because it requires warpgroup-reentrant FA2 |

And the process lesson, since it cost the most: the barrier-0 fix and the
3-warpgroup drain were committed **together**, so when the pair still failed, the
barrier-0 fix looked insufficient when in fact it was the entire answer and the
drain change was the new problem. Separating those two would have found this
several cycles earlier. The per-warpgroup mdo barriers and relocated queue ids are
kept -- they are verified harmless and are a prerequisite if warpgroup-reentrant
FA2 is ever done -- but they were not what fixed it.

Next: the throughput A/B. The symmetric kernel is the mechanism identified as the
cause of the residual -2.0%, so this is the measurement that says whether the gap
closes. `FLASHINFER_FUSED_SYMMETRIC=1` with `fa3_num_ctas == num_sm`; the server
plumbing passes `sm_fa3` as the planned count, so it needs the full grid passed
there before a server run.

##### Symmetric backfill transforms the kernel: 0.687x -> 1.003x of the monolithic kernel

Kernel-level 4-mode MTP benchmark (`tests/bench_mtp_fused.py`, 40 mixed MTP steps,
H200), asymmetric vs symmetric, everything else identical:

| | asymmetric | symmetric | change |
|---|---|---|---|
| fused (fixed split) | 84202.5 us (0.408x mono) | **35082.3 us (0.974x mono)** | **2.40x faster** |
| best split | 50051.8 us (0.687x mono) | **34071.3 us (1.003x mono)** | **1.47x faster** |
| steps where fused beats mono | **0 / 40** | **21 / 40** | — |
| steps where fused beats seq | 0 / 40 | 0 / 40 | unchanged |
| mono (reference) | 34392.2 us | 34168.4 us | — |
| seq co-launch | 25483.0 us (1.350x) | 25532.6 us (1.338x) | — |

**The one-directional backfill was the whole structural deficit.** Letting FA2-role
CTAs execute FA3 work moves the kernel from 0.687x to **1.003x** of the monolithic
persistent kernel -- from a 1.5x loss to parity, winning outright on 21 of 40
steps where it previously won none. That is the single largest improvement in this
task, and it confirms the diagnosis the chunk-2048 experiment pointed to: the
makespan was set by whichever role finished last, and the idle role could not help.

Sequential co-launch is still ahead at 1.338x, so the fused kernel is not yet the
fastest option at kernel level; it is now competitive rather than structurally
beaten.

**The optimal split moved, and the host default is now wrong.** Best-split
distribution over the 40 steps:

```
asymmetric:  {64: 20, 80: 7, 16: 6, 48: 5, 32: 1, 8: 1}   -> favours 64
symmetric:   {8: 26, 24: 6, 32: 4, 16: 3, 96: 1}          -> favours 8
```

That reversal is the mechanism working as intended: once every CTA can reach FA3's
queue, a *small* dedicated FA3 set is best and the rest of the device backfills.
`MAX_SM_FA3 = 64`, which was measured correct for the asymmetric kernel, is
therefore wrong for symmetric mode -- it should be near 8, and the server-side
sweep must be redone before quoting any symmetric end-to-end number.

Next, in order: (1) re-sweep `sm_fa3` server-side under symmetric mode around 8,
(2) thread `num_sm` as `fa3_num_ctas` through `_plan_fused_split_inner` (it
currently passes `sm_fa3`, which symmetric mode's contract rejects), (3) rerun the
server A/B. The -2.0% figure predates all of this and should not be carried
forward as the kernel's standing result.

##### Server-side: symmetric backfill turns the regression into a WIN (+0.8%)

Matched harness, 200 prompts, `--random-input-len 8192 --random-output-len 192
--request-rate 12 --max-concurrency 32`, chunk 512, cap 160 / 320 MB,
`FLASHINFER_FUSED_SYMMETRIC=1`, H200:

| configuration | tok/s | vs base | mean ITL | p99 ITL | mean TTFT | fused share |
|---|---|---|---|---|---|---|
| **symmetric, `sm_fa3=8`** | **25155.02** | **+0.8%** | **18.33** | 32.90 | **2636.22** | 90.6% |
| symmetric, `sm_fa3=16` | 24959.08 | +0.04% | 18.48 | 34.34 | 2668.59 | 90.5% |
| base (fused off), 2 samples | 24946.41 / 24951.21 | — | 18.46 / 18.47 | 32.85 / 34.99 | 2671 / 2674 | absent |
| asymmetric best (`sm_fa3=64`) | 24421 | -2.0% | 18.87 | 33.18-35.07 | 2757-2764 | 90.5% |

200/200 requests, 0 misaligned, 0 launch failures in both symmetric arms.

**The fused path is no longer a regression.** At `sm_fa3 = 8` it beats the
persistent kernel on throughput (+0.8%), on mean ITL (18.33 vs 18.46-18.47) and on
mean TTFT (2636 vs 2671-2674), with p99 ITL inside the baseline's own run-to-run
range (32.85-34.99).

Full progression on this workload, every row with its `mixed_fused_fa3` share
checked:

| configuration | tok/s | vs base |
|---|---|---|
| heuristic `sm_fa3=48`, cap 96, asymmetric | 23250.44 | -8.1% |
| swept `sm_fa3=64`, cap 96, asymmetric | 24055.80 | -3.6% |
| `sm_fa3=64`, cap 136, asymmetric | 24421 | -2.0% |
| **`sm_fa3=8`, cap 160, SYMMETRIC** | **25155.02** | **+0.8%** |

And the split optimum moved exactly as the kernel benchmark predicted: 64 for the
asymmetric kernel, ~8 for the symmetric one, because once every CTA can reach FA3's
queue a small dedicated FA3 set is best and the rest of the device backfills.

**Caveat, stated rather than buried: this is one sample at `sm_fa3 = 8`.** The
baseline has two samples agreeing to 0.02% (~5 tok/s), so the +209 tok/s gap is far
outside the *baseline's* repeat spread -- but the symmetric arm itself has not been
repeated. Before `SGLANG_FUSED_FA3_SM_SPLIT` is flipped on by default, repeat the
`sm_fa3=8` arm 2-3 times and sweep 4/8/12 to confirm the optimum. That is the only
thing standing between this and shipping the path enabled.

Commits: sglang `130a1893e` (server plumbing), flashinfer `ca17ed5` (the working
symmetric kernel), `94764bb` (per-warpgroup barriers), `eff4970` (rendezvous off
barrier 0 -- the actual fix).

##### Confirmation: the win repeats, and it is a plateau rather than a spike

Same harness, symmetric mode, sweeping the new optimum and repeating `sm_fa3=8`:

| `sm_fa3` | tok/s | vs base | mean ITL | p99 ITL | mean TTFT |
|---|---|---|---|---|---|
| 4 | 25048.30 | +0.40% | 18.41 | 33.80 | 2652.14 |
| **8 (sample 1)** | **25155.02** | **+0.83%** | 18.33 | 32.90 | 2636.22 |
| **8 (sample 2)** | **25126.94** | **+0.71%** | 18.36 | 34.14 | 2639.82 |
| 12 | 25074.79 | +0.50% | 18.40 | 34.03 | 2646.47 |
| 16 | 24959.08 | +0.04% | 18.48 | 34.34 | 2668.59 |
| base (2 samples) | 24946.41 / 24951.21 | — | 18.46 / 18.47 | 32.85 / 34.99 | 2671 / 2674 |

200/200 requests, 0 misaligned, 0 launch failures, 90.6% fused share in every arm.

**The caveat from the previous section is cleared.** The two `sm_fa3=8` samples
agree to 0.11% (25155.02 / 25126.94, mean 25141) and both sit above the baseline's
two samples, which agree to 0.02%. It is a **plateau**, not a lucky point: 4, 8 and
12 all beat the baseline, and mean ITL (18.33-18.41 vs 18.46-18.47) and mean TTFT
(2636-2652 vs 2671-2674) improve across the whole plateau. Five independent fused
runs, all at or above baseline.

p99 ITL is the one metric that does not clearly improve (32.90-34.14 against the
baseline's 32.85-34.99) -- but those ranges overlap, and p99 ITL was already
established as this harness's noisiest statistic, varying 6.5% between two
identical baseline runs.

**Enabling the path by default is now justified by evidence**, and it is three
coupled changes rather than one, which is why it is written down rather than done
in the last minutes of a session:

1. `SGLANG_FUSED_FA3_SM_SPLIT` default 0 -> 1
2. `FLASHINFER_FUSED_SYMMETRIC` default 0 -> 1 (the asymmetric kernel is the
   regression; the default must be the symmetric one)
3. `MAX_SM_FA3` 64 -> 8, but **mode-dependent**: 64 remains the measured optimum
   for the asymmetric kernel and 8 for the symmetric one, so this must branch on
   the mode rather than simply change the constant

Anyone making that change should re-run this table afterwards, because flipping
defaults alters which code path the gate selects and that has produced five
invalid A/Bs in this task already.

##### Why the 4-mode MTP schedule is a control-flow restructuring, not four edits

Reading `prepare_for_decode` (`schedule_batch.py:1571`) settles the size of this
work, and my earlier "four length-assumption edits" estimate was wrong:

```python
if not self.spec_algorithm.is_none():
    # if spec decoding is used, the decode batch is prepared inside
    # `forward_batch_speculative_generation` after running draft models.
    return
```

Under speculative decoding the running batch is **not prepared at batch-formation
time at all**. Its `out_cache_loc` is `None` and its `input_ids` do not yet exist
because the verify rows' contents *are the draft model's output*, produced by a
separate forward inside the spec worker, after the scheduler has already formed
the batch. `mix_with_running` runs at formation time, before any of that.

So the `torch.cat([..., running_batch.out_cache_loc])` crash is not a missing
allocation to be patched in. Mixing a prefill chunk with verify rows requires the
order of operations to change:

1. form the chunked-prefill batch,
2. run the draft model for the running requests,
3. merge the resulting verify rows into the prefill batch,
4. run one fused target forward over both.

Today steps 1 and 4 are the scheduler's and steps 2-3 are the spec worker's, with
no interleaving point between them. That is a restructuring of
`forward_batch_speculative_generation` and its contract with the scheduler -- not
four edits to a merge helper, and not something to attempt behind a guard.

**This also explains why sglang disables the combination at three separate config
sites.** Those gates are not conservatism about an untested path; the control flow
genuinely cannot produce such a batch. Recording that distinction matters, because
"the gates are overly cautious" and "the data the merge needs does not exist yet"
imply very different amounts of work.

One simplification that does hold for whoever takes it on: with
`--speculative-eagle-topk 1` the draft is a **chain**, not a tree, so causal
masking over the verify rows is correct and no tree-mask plumbing is needed. That
removes the attention-backend half of the problem, leaving the scheduling reorder
as the whole of it.

##### Multicast is absent from flashinfer's Hopper attention entirely, not just the paged path

An earlier section said multicast was blocked because the *sparse* mainloop
gathers KV (`sparse_mainloop.cuh:87`, `USE_TMA_LOAD_KV = false`), implying the
dense path had it and the fix was to reach that path. **That was under-evidenced
and the implication is wrong.** Checking `hopper/mainloop.cuh`, the dense path:

```
:61   SmemLayoutQ{}, select<0,2>(TileShape_QKD{}), _1{});   // no mcast for Q
:67   take<0,2>(SmemLayoutK{}), select<1,2>(...), _1{});    // no mcast
:73   take<0,2>(SmemLayoutV{}), select<2,1>(...), _1{});    // no mcast
:121,124,127   same three, rebuilt                          // no mcast
:205,219,234,239,248   copy(..., /*mcast_mask=*/0)
```

Every TMA descriptor is constructed with a cluster dimension of `_1{}` and every
`copy()` passes a zero mcast mask. `hopper/kernel_traits.cuh` contains no cluster
configuration at all.

So there is **no runnable multicast path anywhere in this attention stack** --
dense or sparse, Q or KV. The distinction matters for sizing the work:

- what the earlier note implied: route the fused kernel to the dense mainloop
  (page_size >= 128) and multicast comes along -- **false**;
- what is actually required: *add* multicast to flashinfer's Hopper attention --
  a cluster dimension in the TMA descriptors, a non-zero mcast mask on the copies,
  cluster-scoped mbarriers, and a cluster launch config -- and only then route the
  fused kernel through it.

That is why no multicast number appears in this document: there is nothing to
measure yet, and the honest statement is that the feature does not exist here
rather than that it is switched off. Testing it means implementing it first.

This is the correct characterisation to hand over, and it supersedes the earlier
"needs a `sparse_mainloop.cuh` gather->TMA rewrite" framing, which was necessary
but not sufficient.

##### Multicast, tested on hardware: DSMEM cluster sharing is 1.59x over redundant gather

The earlier sections concluded multicast was untestable because TMA multicast
needs a tensormap over a dense strided region and paged KV at `page_size=1` is a
gather. That is true of **TMA** multicast and it was the wrong place to stop:
Hopper's distributed shared memory gives the same "load once, consume in N CTAs"
property without TMA at all. One CTA's gathered tile is readable by its cluster
peers through `cg::cluster_group::map_shared_rank`, which flashinfer already uses
in `trtllm/fused_moe/RoutingKernel.cuh` and `comm/trtllm_allreduce*.cuh`.

`tests/dsmem_kv_share.cu` measures it against the redundancy our kernel actually
has. Both arms pair CTAs on the **same** KV tile -- which is the real case, since
every q-tile of a prefill row attends to the same KV range, so the CTAs holding
those q-tiles gather identical KV. The only difference is whether each CTA gathers
the whole tile or half of it and reads the peer's half over DSMEM:

```
H200, 132 blocks = 66 clusters of 2, 4096 tiles of 128 tokens x 128 dims (bf16),
1M scattered pages (256 MB, past L2), page_size=1 gather pattern

  redundant gather : 1325.249 us    202.6 GB/s global
  DSMEM shared     :  835.249 us    160.7 GB/s global (half the bytes)
  speedup          :    1.587x
```

**1.587x**, against a 2.0x ceiling (half the global traffic); the gap is
`cluster.sync()` plus DSMEM read latency. The absolute 202.6 GB/s is far below
H200's HBM ceiling because a 256-byte scattered gather is transaction-bound rather
than bandwidth-bound -- which is exactly why halving the number of gathers helps
this much.

Three things this establishes that the earlier "needs a gather->TMA rewrite"
framing got wrong:

1. multicast's *benefit* is reachable at `page_size=1`, with no KV-pool change and
   no tensormap;
2. it needs a **real cluster launch**, which flashinfer's attention does not do
   anywhere -- `cudaLaunchKernelEx` / `clusterDim` appear only in `sampling.cuh`,
   `norm.cuh` and vendored trtllm code, and `persistent.cuh:192` states its own
   "cluster" is logical with `cluster_size` pinned to 1;
3. the sharing only pays where cluster-mates hold q-tiles of the same row, so the
   planner must co-assign them -- it is a scheduling change as much as a kernel one.

Command: `nvcc -std=c++17 -arch=sm_90a -O3 -o /tmp/dsmem tests/dsmem_kv_share.cu && /tmp/dsmem`

##### Multicast and SM-pre-assignment fusion are in tension: the cluster dim is grid-wide

The DSMEM win (1.587x) needs a thread-block cluster. Adding the cluster launch
(`FLASHINFER_FUSED_CLUSTER=N`, flashinfer `610d8d7`) and running the regression:

```
CLUSTER=1   rc=0   PASS, 0/512 unwritten, rel err 0.00025 / 0.00205 / 0.00266
CLUSTER=2   rc=1   CUDA error: unspecified launch failure
CLUSTER=4   rc=1   CUDA error: unspecified launch failure
```

The launch is *accepted* and the kernel then crashes, so something inside is
cluster-sensitive. It is FA3's CUTLASS pipeline, and the mechanism is exact:

- `hopper/prefill_sm90.cuh:113,123` constructs the pipeline with a **hardcoded**
  `cluster_shape = Shape<_1,_1,_1>{}`;
- `cutlass/pipeline/sm90_pipeline.hpp:313-316` derives the mbarrier arrival count
  from that declared shape:

```cpp
uint32_t multicast_consumer_arrival_count = params.num_consumers;  // If cluster_size is 1
if (cute::size(cluster_shape) > 1) {
  multicast_consumer_arrival_count = (size<0>(cluster_shape) + size<1>(cluster_shape) - 1)
                                     * num_consumer_warpgroups_per_cluster;
}
```

So under a real cluster of 2 the barriers are initialised for 1-CTA arrivals while
the hardware has clustered the CTAs. Passing the *true* shape instead does not fix
it -- it makes CUTLASS expect `2 x num_warpgroups` arrivals per barrier, which only
materialise if FA3's TMA loads actually multicast. FA3's KV cannot multicast
(`page_size=1` gather, no tensormap), so those arrivals never come and it hangs
instead of crashing.

**The structural point.** `clusterDim` is a property of the *launch*, so it applies
to every CTA in the grid. This kernel's whole premise is that one grid holds two
different roles. DSMEM is wanted only by the FA2 role (which gathers with
`cp.async` and has no CUTLASS pipeline); it is actively harmful to the FA3 role.
CUDA offers no per-role cluster shape, so:

| option | consequence |
|---|---|
| cluster the whole grid | breaks FA3's pipeline, as measured |
| give FA3 a true cluster shape | CUTLASS then demands multicast arrivals FA3 cannot produce |
| launch the two roles as separate clustered/unclustered kernels | that **is** the sequential co-launch baseline -- already 1.338x faster than this kernel |
| implement TMA multicast in FA3 as well | the full job: tensormap-able KV (`page_size >= 128`) plus mcast masks plus cluster-consistent pipelines |

That last row is the only route that keeps the fusion *and* gets multicast, and it
is substantially more than "add DSMEM to the FA2 mainloop" -- which is what the
1.587x microbenchmark appeared to license. Worth stating plainly because it is a
real cost of the SM-pre-assignment design, not an implementation detail: **the
single-grid, two-role structure is what forecloses per-role use of cluster
features.** Any paper claim about combining SM pre-assignment with multicast has
to confront this.

What is established: multicast's benefit for our access pattern is real and
measured (1.587x, `tests/dsmem_kv_share.cu`); the cluster launch is implemented and
verified to be output-neutral at N=1; and N>1 is blocked on FA3's pipeline for a
reason that is now precisely located rather than suspected.

##### How much of the DSMEM win survives L2? Most of it -- the gather is transaction-bound

The 1.587x above was measured with a 256 MB working set, deliberately past L2. That
overstates the in-situ case: in the real kernel the CTAs that redundantly gather a
row's KV run concurrently, so L2 serves most of the repeats ("locality means
concurrency in time", as the LPT work in this task established). Re-running with an
L2-resident working set bounds the honest upside:

| working set | redundant gather | DSMEM shared | speedup |
|---|---|---|---|
| 16 MB (**inside L2** -- the in-situ case) | 682.436 us / 393.3 GB/s | 514.583 us / 260.8 GB/s | **1.326x** |
| 256 MB (past L2 -- cold) | 1325.362 us / 202.5 GB/s | 835.079 us / 160.7 GB/s | 1.587x |

L2 absorbs part of the redundancy -- the baseline nearly halves, 1325 -> 682 us --
but **the DSMEM advantage only falls from 1.587x to 1.326x**, it does not
disappear. The reason is worth stating because it is the non-obvious part: a
`page_size=1` gather is bound by the *number of 256-byte transactions* each CTA
issues, not by which memory level answers them. L2 shortens the latency of a repeat
but still charges for the transaction and its own bandwidth; DSMEM removes the
repeat entirely. So sharing wins even when the KV is already resident.

That makes the integration genuinely worth wanting -- ~1.33x on the KV-gather
portion of a prefill row's work, in the regime the server actually runs in -- and it
is exactly what the grid-wide `clusterDim` forecloses for this kernel's two-role
design (previous section). The two findings together are the honest state of
multicast here: **the benefit is real and quantified in the realistic regime, and
the fused single-grid structure is what prevents collecting it.**

Command: `/tmp/dsmem 65536` (L2-resident) and `/tmp/dsmem 1048576` (cold), from
`tests/dsmem_kv_share.cu`.

##### Correction: the FA2 side is cluster-safe, so the fusion does NOT foreclose multicast

The section above concluded that the grid-wide `clusterDim` plus this kernel's
two-role structure forecloses per-role cluster features. **That was reasoned, not
measured, and it is wrong.** `FLASHINFER_FUSED_SKIP_FA3=1` sends every CTA straight
to the FA2 queue so a cluster launch can be exercised with FA3's pipeline out of
the picture:

| `CLUSTER` | `SKIP_FA3` | result |
|---|---|---|
| 1 | 0 | rc=0, PASS, 0/512 unwritten |
| 2 | 0 | rc=1, **CUDA error: unspecified launch failure** |
| 1 | 1 | rc=1, 424/512 unwritten -- expected, **no CUDA error** |
| 2 | 1 | rc=1, 424/512 unwritten -- expected, **no CUDA error** |

With FA3 skipped, `CLUSTER=2` runs cleanly: the only failure is the deliberately
missing FA3 output (424 unwritten rows is exactly row 0's token count), with no
CUDA fault. So **the FA2/persistent side tolerates a real cluster launch today**,
and FA3 alone is cluster-hostile.

That changes the conclusion in the right direction:

- the blocker is **not** the single-grid two-role design, it is specifically FA3's
  CUTLASS pipeline under a clustered launch;
- the side that wants DSMEM -- the FA2 role, where the measured 1.326x
  in-situ win lives -- is **already cluster-ready**;
- so the remaining work is to make FA3 *tolerate* a cluster it does not use, which
  is a bounded CUTLASS-pipeline question, not the "implement TMA multicast in FA3
  as well" redesign the previous section claimed.

The ordering lesson, since this is the third time in this task that checking an
asserted cause overturned it: the earlier claim followed from reading
`prefill_sm90.cuh`'s hardcoded `cluster_shape = <1,1,1>` and CUTLASS deriving
arrival counts from it. That reading explains why FA3 *might* break, and it is
probably still the mechanism -- but it does not license the structural conclusion
that was drawn on top of it, and one 2x2 matrix separated the two.

**Next step, now narrow.** Determine why FA3 crashes under a cluster it declares as
size 1. The likely candidate is barrier initialisation visibility: CUTLASS's
`init_barriers` fences with `__syncthreads()` scope appropriate to a 1-CTA cluster,
while a hardware cluster needs the init visible cluster-wide before any TMA
transaction completes against it. If that is it, the fix is a `cute::cluster_sync()`
after init under a clustered launch -- not a multicast implementation. Then: planner
co-assignment of cluster-mates to q-tiles of one row, and the DSMEM gather split in
the FA2 mainloop.

##### Narrowing the cluster blocker: not the FA2 side, not shared memory -- it is FA3's TMA pipeline

Two candidate causes eliminated by measurement rather than argument.

**Shared memory is not it.** The fused kernel asks for ~165 KB (FA3's
`SharedStorage` at one K/V stage plus the persistent mainloop's), while
`dsmem_kv_share.cu` clusters happily at 16 KB -- so smem size was a live
hypothesis. Padding the benchmark's dynamic smem request to match:

| dynamic smem | cluster launch | speedup |
|---|---|---|
| 16 KB (natural) | ok | 1.325x |
| 100 KB | ok | 1.326x |
| **165 KB** (the fused kernel's footprint) | **ok** | 1.327x |
| 200 KB | ok | 1.326x |

A clustered launch tolerates the full footprint. Combined with the `SKIP_FA3`
matrix (FA2-only clusters cleanly), the cause is isolated to **FA3's CUTLASS TMA
pipeline** -- three of the four candidates are now excluded by test.

**Why both cluster_shape choices fail.** FA3 declares
`cluster_shape = Shape<_1,_1,_1>` (`prefill_sm90.cuh:113,123`):

- declaring **1** under a real cluster: CUTLASS takes its `__syncthreads()` branch
  for barrier init and computes cta-local mbarrier addresses, while the hardware
  encodes mbarrier arrives cluster-relatively -- a TMA completion can signal the
  wrong CTA's barrier. Crashes, as measured.
- declaring the **true shape**: `sm90_pipeline.hpp:313-316` then sets
  `multicast_consumer_arrival_count = (size<0> + size<1> - 1) * num_consumer_warpgroups`,
  i.e. it expects consumers from *both* CTAs to arrive on each barrier. They only
  do if the TMA load actually multicasts, and FA3's KV cannot (`page_size=1`
  gather). It would hang rather than crash.

So FA3 cannot simply be told about the cluster. Three routes remain, and the third
is new and the cheapest:

1. implement real TMA multicast in FA3 -- needs tensormap-able KV
   (`page_size >= CTA_TILE_KV`), mcast masks and cluster-consistent pipelines;
2. leave FA3 unclustered by splitting the roles into separate launches -- which is
   the sequential co-launch baseline, already 1.338x ahead of this kernel;
3. **drop TMA from the FA3 role.** The sparse mainloop already gathers KV with
   `cp.async`; only Q goes through TMA (`sparse_mainloop.cuh:80-87`, `TMA_Q`). Load
   Q with `cp.async` too and FA3 has no CUTLASS pipeline, hence no cluster
   conflict, and the whole grid can cluster. The cost is Q-load efficiency on a
   tile that is small next to the KV stream -- and the gain is the measured 1.326x
   on the KV gather, which is the dominant traffic. This is the option to try
   first, and it was invisible until smem and the FA2 side were ruled out.

##### Correction: the pipeline mechanism above is wrong -- that `if constexpr` branch is not taken

The previous two sections blamed FA3's cluster crash on
`PipelineTmaAsync`'s arrival-count arithmetic
(`cutlass/pipeline/sm90_pipeline.hpp:313-316`) reached through a hardcoded
`cluster_shape = Shape<_1,_1,_1>`. **This configuration never instantiates that
pipeline.** Reading `prefill_sm90.cuh:109-127` properly:

```cpp
MainloopPipeline pipeline_k = [&] {
  if constexpr (use_tma_load_kv) {
    pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    return MainloopPipeline(shared_storage.pipeline_k, pipeline_params,
                            /*cluster_shape=*/Shape<_1, _1, _1>{});
  } else {
    return MainloopPipeline(shared_storage.pipeline_k, pipeline_params);   // <-- ours
  }
}();
```

and `kernel_traits.cuh:112-113`:

```cpp
using MainloopPipeline =
    std::conditional_t<USE_TMA_LOAD_KV, cutlass::PipelineTmaAsync<NUM_STAGES>,
                       cutlass::PipelineAsync<NUM_STAGES>>;
```

The fused kernel runs the **sparse** mainloop, where `USE_TMA_LOAD_KV = false`. So
`MainloopPipeline` is `PipelineAsync`, the `cluster_shape` argument is never
passed, and the multicast arrival-count branch is dead code here. Both the
"declaring 1 computes cta-local addresses" and the "declaring the true shape
demands multicast arrivals" arguments were about code this build does not compile.

**What is actually left to suspect**, having read the right branch: the sparse FA3
path still uses TMA in exactly two places, and nowhere else.

1. `TMA_Q` -- Q is TMA-loaded (`sparse_mainloop.cuh:80-87`), signalling
   `shared_storage.barrier_Q`, which `prefill_sm90.cuh:105` initialises with
   `init(/*num_threads=*/1)`.
2. the epilogue's TMA **store** of O, which is why cross-role backfill needed
   `cute::tma_store_wait<0>()` in the first place.

A TMA operation's mbarrier completion is cluster-relative on Hopper, and
`barrier_Q` is initialised and waited on as a CTA-local barrier. That is the
remaining candidate, and it is consistent with the crash being a launch-time
addressing fault rather than a hang.

Usefully, this makes the route recorded in the previous section *more* attractive
rather than less: replacing `TMA_Q` with a `cp.async` Q load removes the only TMA
load on this path, and the only remaining TMA is the epilogue store. The fix is
smaller than "implement TMA multicast in FA3", and the claim it rests on is now
read from the branch that is actually compiled.

**Fourth correction of an asserted cause in this task.** The pattern is consistent
enough to be worth naming: each time, the error was reasoning from code that looked
relevant without checking which `if constexpr` / template specialisation this build
selects. `USE_TMA_LOAD_KV = false` is the single fact that decides most of FA3's
behaviour here, and it needs checking first, not last.

##### The Q TMA load is NOT the cluster blocker -- it is the epilogue's TMA store

Tested rather than assumed, with `FLASHINFER_SKIP_TMAQ` replacing the Q TMA
producer by a bare `barrier_Q.arrive()`:

| `CLUSTER` | `SKIP_TMAQ` | result |
|---|---|---|
| 1 | 0 | rc=0, PASS |
| 2 | 0 | rc=1, unspecified launch failure |
| 1 | 1 | rc=1, 424/512 unwritten -- expected with garbage Q, **no CUDA error** |
| 2 | 1 | rc=1, **still unspecified launch failure** |

Removing the only TMA *load* on the sparse path does not fix the clustered launch.
So `TMA_Q` and its `ClusterTransactionBarrier` are **not** the blocker, and the
"replace `TMA_Q` with `cp.async`" route recorded two sections ago would have been
wasted work -- the fifth asserted cause in this task to be overturned by actually
testing it.

By elimination the remaining TMA on the FA3 path is **the epilogue's TMA store of
O**, and that is now the only candidate left:

| candidate | status |
|---|---|
| FA2 / persistent side | cluster-safe (`SKIP_FA3` matrix) |
| dynamic shared memory size | cluster-safe at 100 / 165 / 200 KB |
| `PipelineTmaAsync` arrival counts | never compiled here (`USE_TMA_LOAD_KV = false`) |
| `TMA_Q` + `barrier_Q` | **ruled out above** |
| **epilogue TMA store of O** | **only one left** |

That is a more consequential finding than the Q result would have been, because the
epilogue store is not incidental to this kernel: `cute::tma_store_wait<0>()` is
precisely what cross-role backfill had to wait on before an FA3 CTA could drain the
FA2 queue. So the store is load-bearing for the backfill design, and making it
cluster-safe is not a matter of swapping it for a plain store without re-examining
that interaction.

Next test, and it completes the elimination: skip the epilogue TMA store the same
way and re-run `CLUSTER=2`. If it then runs, the blocker is confirmed and the
design question becomes whether O can be written without TMA (losing the
`tma_store_wait` handle that backfill uses) or whether the store can be made
cluster-correct.

**Method note, now that five causes have been overturned.** Every one of the five
was reasoned from code that looked relevant; every one fell to a two-line
compile-time or runtime switch that isolated it. The switches cost minutes and the
reasoning cost hours. On this kernel, isolate first.

##### Elimination state for the cluster blocker

The epilogue stores O with a plain `TiledCopyO` (`epilogue.cuh:203`), not TMA --
the store is sparse, so TMA would need a descriptor per page. So there is no
epilogue TMA store to blame.

| candidate | status |
|---|---|
| FA2 / persistent side | cluster-safe (`SKIP_FA3` matrix) |
| dynamic shared memory size | cluster-safe at 100 / 165 / 200 KB |
| `PipelineTmaAsync` arrival counts | never compiled (`USE_TMA_LOAD_KV = false`) |
| `TMA_Q` load | ruled out (`SKIP_TMAQ` matrix) |
| epilogue store | plain tiled, not TMA |
| `ClusterTransactionBarrier` barrier_Q / barrier_O | ruled out -- `tests/cluster_barrier_repro.cu` reproduces FA3's init(1)/arrive/wait pattern and is OK at cluster 1, 2 and 4 |

Every cheap hypothesis is now excluded. The next step is compute-sanitizer against
the clustered fused kernel, which was avoided earlier as expensive but is now the
only tool left with the search space exhausted:

```
FLASHINFER_FUSED_CLUSTER=2 /usr/local/cuda/bin/compute-sanitizer --tool memcheck \
  /dev/shm/w/venv/bin/python tests/test_fused_server_batch.py
```

memcheck (sanitizer 2025.4.1.0) attaches and reports **no device-side violation**
-- only the host-side `cudaErrorLaunchFailure (719)` propagating through
`cudaDeviceSynchronize` / `cudaGetLastError`, 4 errors all of that kind. No
"Invalid __global__ read", no out-of-bounds, no misaligned access.

That is itself a narrowing: the clustered launch is **not** failing on a memory
access. `unspecified launch failure` with a clean memcheck points at an illegal
instruction or a barrier/sync violation, which is `synccheck`'s domain -- and this
kernel is dense with named barriers plus FA3's warp specialisation, so that is the
tool to run next:

```
FLASHINFER_FUSED_CLUSTER=2 /usr/local/cuda/bin/compute-sanitizer --tool synccheck \
  /dev/shm/w/venv/bin/python tests/test_fused_server_batch.py
```

synccheck, run **detached on the pod** to get past the webshell's hard 25-minute
ceiling (which killed the first attempt -- a tooling limit, not a result), reports
**`ERROR SUMMARY: 0 errors`**.

So both sanitizer tools are clean:

| tool | result |
|---|---|
| memcheck | 0 device-side violations (only host `cudaErrorLaunchFailure` propagating) |
| synccheck | **0 errors** |

A clean memcheck *and* a clean synccheck alongside `unspecified launch failure`
rules out the two large classes -- bad memory access and illegal
barrier/synchronisation. What is left is an illegal instruction or a runtime
resource/scheduling failure, neither of which either tool reports.

The next probe is therefore a host-side one rather than another sanitizer run:
`cudaOccupancyMaxActiveClusters` for this kernel at cluster size 2. If it returns
0, the cluster is simply not schedulable at this kernel's 384 threads and ~165 KB
of shared memory, and the launch failure is explained without any device-side
defect -- the fix would then be reducing per-CTA resources rather than hunting a
bug. Note the microbenchmark clusters fine at 165 KB but uses only 128 threads, so
thread count is the untested difference, not smem.

Practical notes so the next attempt does not re-lose time:
- run the sanitizer **detached** (`nohup ... &` on the pod) and poll a done-marker;
  a foreground webshell call is capped at 25 minutes and synccheck exceeds it;
- invoke `/usr/local/cuda/bin/compute-sanitizer` explicitly -- bare
  `compute-sanitizer` resolved to a build that never attached (no header), and the
  venv runs CUDA 12.8 against a 13.1 default;
- run it on the standalone harness, never the server, which trips the detokenizer
  heartbeat long before the kernel is reached -- the same trap
  `CUDA_LAUNCH_BLOCKING=1` set earlier in this task.

Multicast therefore stands as: benefit measured on hardware (1.326x in situ,
1.587x cold), cluster launch implemented and output-neutral at N=1, integration
blocked at N=1 -> N=2 by a fault whose cause is not yet identified.

One consequence that is worth recording because the goal states otherwise:
cross-role backfill does **not** need `tma_store_wait`. With a plain tiled store
there are no TMA stores to drain, so `cute::tma_store_wait<0>()` at
`persistent_fa3_fused.cuh:236` guards nothing; the guard that matters is
`cp_async::wait_group<0>()` at line 246. The measured backfill numbers (+0.04%
server, 1.2% H200 kernel-level) rest on `wait_group`.

##### Strong lead: `__launch_bounds__(N, 1)` may be what blocks the cluster launch

`tests/cluster_occupancy_probe.cu` asked the driver directly. The probe has a flaw
-- the first failure leaves a sticky CUDA error, so only its **first row is valid**
and rows 2-6 merely echo `cudaErrorLaunchFailure`; each config needs its own
process. Row 1 is still the useful one:

```
threads=128 smem=16KB cluster=2 | maxActiveClusters=-1 (cudaErrorInvalidClusterSize)
                                | launch=ok sync=cudaErrorLaunchFailure
```

That is the **same shape that works** in `dsmem_kv_share.cu` (128 threads, 16 KB,
cluster 2, `map_shared_rank` + `cluster.sync()`). The one structural difference
between the two kernels: this probe declares `__launch_bounds__(128, 1)`, and the
working benchmark declares no launch bounds at all.

The fused kernel declares
`__launch_bounds__(FA3Traits::NUM_WARPS * cutlass::NumThreadsPerWarp, 1)`.

So the hypothesis is that the explicit **`minBlocksPerMultiprocessor = 1`** in
`__launch_bounds__` conflicts with cluster scheduling -- the driver is being told
to guarantee one block per SM while also being asked to co-schedule a cluster, and
`cudaOccupancyMaxActiveClusters` rejecting the size with
`cudaErrorInvalidClusterSize` is consistent with that. It would also explain the
whole picture: both sanitizers clean, no memory or sync violation, launch accepted,
kernel fails -- a scheduling constraint, not a device-side defect.

**This is a hypothesis, stated as one.** Six previous asserted causes in this task
were overturned by test, so it needs the clean experiment before anything is built
on it: three separate processes, one each for `__launch_bounds__(128, 1)`,
`__launch_bounds__(128)` and no launch bounds, at cluster 2. If the `, 1` variant is
the only failure, the fix in the fused kernel is to drop the
`minBlocksPerMultiprocessor` argument -- a one-token change -- and then the whole
grid can cluster and the DSMEM gather split becomes implementable.

##### `__launch_bounds__` is not the cause either -- and that probe is unsound

Four variants, one process each, cluster 2, 128 threads / 16 KB except where noted:

```
__launch_bounds__(128, 1)   maxActiveClusters=-1 (cudaErrorInvalidClusterSize)  sync=cudaErrorLaunchFailure  FAILS
__launch_bounds__(128)      maxActiveClusters=-1 (cudaErrorInvalidClusterSize)  sync=cudaErrorLaunchFailure  FAILS
no __launch_bounds__        maxActiveClusters=-1 (cudaErrorInvalidClusterSize)  sync=cudaErrorLaunchFailure  FAILS
__launch_bounds__(384, 1)   maxActiveClusters=-1 (cudaErrorInvalidClusterSize)  sync=cudaErrorLaunchFailure  FAILS
```

So `minBlocksPerMultiprocessor` is **not** the cause -- the hypothesis is
falsified, the seventh in this task.

**But the more useful reading is that this probe does not measure what it claims.**
The third row -- no launch bounds, 128 threads, 16 KB, `cluster.sync()` plus
`map_shared_rank` -- is essentially `dsmem_kv_share.cu`'s `gather_dsmem`, which
clusters successfully and has produced every multicast number in this document. A
probe that fails on a configuration known to work is broken, not informative.

The likely defect is mine: `cudaOccupancyMaxActiveClusters` is called before
`cfg.gridDim` is assigned, so the config passed to the query is incomplete, which
is consistent with `cudaErrorInvalidClusterSize` appearing in *every* row
regardless of kernel shape. Whether that failed query then perturbs the subsequent
launch is not established.

**So both `cluster_occupancy_probe.cu` and `cluster_lb_probe.cu` should be treated
as unreliable until the query is given a complete config and each row is
independently shown to reproduce `dsmem_kv_share.cu`'s success as a control.** That
control is the thing both probes lacked -- neither included a configuration known
to work, so neither could detect its own breakage.

What still stands, unaffected by these two probes:

| fact | source |
|---|---|
| DSMEM sharing is worth 1.326x in situ, 1.587x cold | `dsmem_kv_share.cu` |
| the fused kernel fails at `CLUSTER=2`, passes at 1 | `test_fused_server_batch.py` |
| the FA2-only path clusters cleanly | `SKIP_FA3` matrix |
| `TMA_Q` is not the cause | `SKIP_TMAQ` matrix |
| `ClusterTransactionBarrier` is not the cause | `cluster_barrier_repro.cu` |
| no memory violation, no sync violation | memcheck, synccheck |

The next step is to fix the probe -- complete config, plus a known-good control row
-- before drawing any further conclusion about cluster schedulability. Seven
overturned hypotheses here all share one shape: a measurement or reading that was
not validated against something known to work first.

##### Occupancy is not the blocker: clusters are schedulable at the fused kernel's shape

With `cfg.gridDim` set before the query (the defect that invalidated the earlier
probes) `cudaOccupancyMaxActiveClusters` answers cleanly:

```
no __launch_bounds__      threads=128 | maxActiveClusters=528 (ok) | sync=cudaErrorLaunchFailure
__launch_bounds__(128)    threads=128 | maxActiveClusters=528 (ok) | sync=cudaErrorLaunchFailure
__launch_bounds__(128, 1) threads=128 | maxActiveClusters=528 (ok) | sync=cudaErrorLaunchFailure
__launch_bounds__(384, 1) threads=384 | maxActiveClusters=330 (ok) | sync=cudaErrorLaunchFailure
```

**528 and 330 active clusters are available**, so a cluster of 2 is comfortably
schedulable at 384 threads -- the fused kernel's shape -- and the
scheduling/resource hypothesis is eliminated by a correct measurement. That also
retires the `cudaErrorInvalidClusterSize` seen earlier: it was the malformed query,
not a property of any kernel.

**The probe still cannot speak to the fused kernel**, because all four rows
including the control fail at sync, and the control (no launch bounds, 128 threads,
16 KB, `cluster.sync()` + `map_shared_rank`) is meant to mirror
`dsmem_kv_share.cu`'s `gather_dsmem`, which clusters successfully. The remaining
difference between the two is unidentified; candidates not yet separated are the
template-wrapped `cudaLaunchKernelEx` call, the single-argument kernel signature,
and writing the shared buffer from one thread rather than in a loop.

Net candidate list for the fused kernel's `CLUSTER=2` failure, all by test:

| candidate | status |
|---|---|
| FA2 / persistent side | cluster-safe |
| dynamic shared memory | cluster-safe to 200 KB |
| `PipelineTmaAsync` | not compiled here |
| `TMA_Q` | ruled out |
| epilogue store | plain tiled, not TMA |
| `ClusterTransactionBarrier` | ruled out |
| memory violation | none (memcheck) |
| sync violation | none (synccheck) |
| `__launch_bounds__` | ruled out |
| **cluster schedulability / occupancy** | **ruled out -- 330 clusters available at 384 threads** |

Nine candidates eliminated, cause still unidentified. The productive next move is
not another probe but a bisection of the real kernel: start from
`dsmem_kv_share.cu`, which clusters, and add the fused kernel's elements one at a
time until it breaks -- rather than starting from the fused kernel and removing
them, which is what the last several probes effectively attempted.

##### Bisecting from the known-good base: 384 threads and `__launch_bounds__` cluster fine

Rather than removing elements from the fused kernel with hand-written probes --
which kept failing their own control rows -- the question was re-asked by adding the
fused kernel's distinguishing features to `dsmem_kv_share.cu`, which clusters
successfully. Same kernel body, 384 threads, `__launch_bounds__(384, 1)`:

```
redundant gather : 1579.338 us
DSMEM shared     : 1157.636 us     speedup 1.364x
384 threads + __launch_bounds__(384,1), clustered: launch=ok sync=ok   <= CLUSTERS FINE
```

The control arms pass and the wide arm passes, so this measurement can be trusted
where the earlier probes could not. **Thread count and `__launch_bounds__` are
definitively not the cause.**

That matters because of what it leaves. The `SKIP_FA3` matrix already showed the
fused kernel clusters cleanly with FA3's body skipped and fails with it active, and
inside that body the following are excluded: `TMA_Q`, `ClusterTransactionBarrier`,
`PipelineTmaAsync` (not compiled), the epilogue store (not TMA), memory violations,
sync violations, occupancy, smem size, thread count, launch bounds.

**The one major element of FA3's body never tested under a cluster is its warp
specialisation -- `setmaxnreg`**, i.e. `cutlass::arch::warpgroup_reg_dealloc<24>()`
on the producer and `warpgroup_reg_alloc<240>()` on the consumers. That fits every
observation: dynamic register reallocation is invisible to memcheck and synccheck,
costs no occupancy, and would fail at execution rather than launch.

It is also the element this task has already been bitten by three times -- the
`__noinline__` FA3 body, `MergeGroupJIT`, and the symmetric-drain attempt all broke
on `setmaxnreg` semantics. A fourth interaction, this time with cluster
co-scheduling, is the natural hypothesis.

Next bisection step, from the same known-good base: add
`warpgroup_reg_dealloc<24>` / `warpgroup_reg_alloc<240>` to the wide arm and launch
it clustered. If that breaks, the cause is found and the design question becomes
whether FA3's warp specialisation can coexist with clustering at all -- which would
be a genuine constraint on combining SM pre-assignment with DSMEM sharing, not a
bug to fix.

##### `setmaxnreg` is cluster-safe too: single-element isolation is exhausted

```
setmaxnreg warp-spec, 384 threads, cluster=1 | launch=ok sync=ok   <= OK
setmaxnreg warp-spec, 384 threads, cluster=2 | launch=ok sync=ok   <= OK
setmaxnreg warp-spec, 384 threads, cluster=4 | launch=ok sync=ok   <= OK
```

`warpgroup_reg_dealloc<24>` on one warpgroup and `warpgroup_reg_alloc<240>` on two,
at 384 threads, clusters cleanly -- with the cluster=1 control passing, so this is
not a toy-register-math artefact. The natural prior was wrong: three earlier bugs in
this task were `setmaxnreg` interactions, but this is not a fourth.

**Full elimination record for the `CLUSTER=2` failure, every row by test:**

| candidate | verdict |
|---|---|
| FA2 / persistent side | cluster-safe |
| dynamic shared memory (to 200 KB) | cluster-safe |
| `PipelineTmaAsync` arrival counts | not compiled here |
| `TMA_Q` load + `barrier_Q` | ruled out |
| epilogue store | plain tiled, not TMA |
| `ClusterTransactionBarrier` | ruled out |
| memory violation | none (memcheck) |
| sync / barrier violation | none (synccheck) |
| `__launch_bounds__`, `minBlocksPerMultiprocessor` | ruled out (from known-good base) |
| thread count (384) | ruled out (from known-good base) |
| cluster schedulability / occupancy | ruled out -- 330 clusters available |
| `setmaxnreg` warp specialisation | **ruled out** |

Eleven candidates, no cause. Each element of FA3's body that can be isolated is
cluster-safe in isolation, so the failure is most likely an **interaction** between
elements rather than any one of them -- which single-element isolation cannot find
by construction.

**Where that leaves multicast.** The benefit is measured and real (1.326-1.364x in
situ on the paged-gather pattern, 1.587x cold), the cluster launch is implemented
and output-neutral at N=1, and the blocker is characterised to the point of having
eliminated everything separable. It is **not integrated**, and the remaining route
is incremental reconstruction: keep adding fused-kernel elements to
`dsmem_kv_share.cu` until it breaks, which amounts to rebuilding FA3's mainloop
piece by piece. That is a substantial piece of work and the right place for a fresh
session to start, with eleven dead ends already ruled out rather than waiting to be
rediscovered.

**Method, finally.** Eight hypotheses were asserted and overturned in this thread.
The two that were overturned *cheaply* were the ones tested from a configuration
known to work; the ones that cost most were reasoned from code without a control.
The elimination table above is the durable artefact -- not the hypotheses.

##### The EDF perf-model check needs no new instrumentation -- just `--decode-log-interval 1`

Earlier this document recorded that a direct per-step MAPE fit "needs per-step
instrumentation that does not exist", and settled for an ITL-dispersion proxy that
returned a null (CV 0.266 vs 0.256). That claim was wrong. sglang's existing
decode-batch log line already carries everything required:

```
Decode batch, #running-req: R, #token: T, ..., gen throughput (token/s): G
```

`G` is averaged over `decode_log_interval` steps, so mean seconds per step over the
window is `R / G` -- R tokens emitted per step at R running requests. Coarse, but
identical in both arms, so the fused-vs-baseline comparison is fair. Fitting
`t_step = c0 + c1*T + c2*R` per arm and comparing MAPE answers the goal's question
directly: lower MAPE means a more predictable step cost, hence tighter EDF deadline
control.

`tests/` has the fitter (`/tmp/edf_fit.py`, normal-equations least squares, no
numpy). What blocks it is **sample count, not instrumentation**: the existing A/B
logs hold only **4** `Decode batch` lines each, because `decode_log_interval`
defaults to 40 and those runs were prefill-bound. The fit needs a few dozen.

So the remaining work is one flag and two short runs, not a code change:

```
# add to serve.sh, both arms
--decode-log-interval 1
# then, as before
SGLANG_FUSED_FA3_SM_SPLIT=1 SGLANG_FUSED_SM_FA3=8 FLASHINFER_FUSED_SYMMETRIC=1   # fused
SGLANG_FUSED_FA3_SM_SPLIT=0                                                      # baseline
python edf_fit.py fused=<fused.log> baseline=<base.log>
```

(The fitter's label parsing splits on the first `=`, so labels must not contain
one.) This is a materially smaller task than the doc previously implied, and it is
the only one of the three outstanding goal items that does not require a rewrite.

##### EDF perf-model check, done properly: both arms equally predictable (null, second method)

With `--decode-log-interval 1` (199 decode-batch samples per arm) and the step-cost
fit `t_step = c0 + c1*T + c2*R` per arm:

| arm | n | MAPE | p90 | worst |
|---|---|---|---|---|
| fused (`sm_fa3=8`, symmetric) | 193 | **0.39%** | 0.86% | 3.37% |
| baseline (fused off) | 193 | **0.40%** | 0.96% | 3.50% |
| ratio fused/baseline | — | **0.987** | — | — |

**The fused kernel is not measurably more predictable.** A 1.3% relative difference
in MAPE is a tie, and this is the **second independent method** to say so -- the ITL
dispersion proxy gave CV 0.266 vs 0.256, a 4% difference in the same direction and
equally inside noise.

Two caveats that bound what this measures, both of which matter:

1. **MAPE of 0.4% is suspiciously good, and that is the tell.** In decode-only
   steady state at `--decode-log-interval 1`, `T` and `R` barely vary between
   steps, so a linear model fits an almost-constant signal. The fit is well-posed
   but the question it answers is easy.
2. **The goal's question is about mixed steps, and these are decode-only.** sglang
   logs a throughput figure only on `Decode batch` lines; mixed prefill+decode steps
   appear as `Prefill batch` lines with no throughput, so they are absent from this
   dataset -- and they are exactly the steps where composition varies and where the
   fused kernel's cost model would have to earn its advantage.

So the honest state of the claim: **the "better EDF predictability" argument is not
supported for the SM-pre-assignment kernel by either method tried.** The 3.1% MAPE
closed-form result quoted in `pd_colocation_vs_disagg.md` section 3 belongs to the
*overlap* kernel and does not transfer -- which that document already says, and this
is now direct evidence for keeping them separate rather than an argument for it.

Closing the remaining gap needs per-step timing on **mixed** batches, which the
existing logs cannot provide. That is a small scheduler patch (time `run_batch` and
log the composition on every step, not just decode ones) rather than the "does not
exist" instrumentation this doc once claimed -- but it is a code change, unlike the
flag that produced the table above.

Throughput in these two runs (10198 / 10708 tok/s) is far below the +0.7-0.8% A/B
numbers because per-step logging is expensive. The overhead is identical in both
arms so the predictability comparison is fair, but these figures must not be
compared against the 25155 / 24946 results.

#<a id="edf-model-classes"></a>
#### EDF perf model on MIXED steps: the fused kernel is HARDER to predict, not easier

The decode-only fit was a tie, but it could not see mixed steps. With a per-step
trace in the scheduler (`SGLANG_STEP_TRACE=1`, logging mode / extend-tokens /
batch-size / `perf_counter` once per `run_batch`; step time is the timestamp delta,
i.e. wall-clock per iteration, which is what an EDF admission controller actually
predicts), 2121 steps per arm, same fit `t = c0 + c1*tokens + c2*bs`:

| arm | ALL | **MIXED (n=1901)** | EXTEND (23) | DECODE (197) |
|---|---|---|---|---|
| fused (`sm_fa3=8`, symmetric) | 16.91% | **13.54%** | 15.08% | 51.98% |
| baseline (fused off) | 13.36% | **9.79%** | 14.88% | 51.76% |

**The baseline is 1.38x more predictable than the fused kernel on mixed steps**
(9.79% vs 13.54% MAPE; p90 18.97% vs 23.39%). Same direction over all steps. The
pure-EXTEND and pure-DECODE classes are indistinguishable between arms, as expected
-- the fused kernel only acts on mixed batches.

**This falsifies the goal's point 3 as applied to this kernel.** The hypothesis was
that the fused attention kernel is *easier* to perf-model, giving a better EDF
predictor and tighter SLO control. Measured, it is the reverse.

A coherent mechanism, consistent with everything else in this document: the fused
kernel's step cost contains a **makespan** term -- `max(FA3 role, FA2 role)` over a
static SM partition -- which is not linear in total tokens, so a linear model fits
it worse. The persistent baseline has one shared work queue, so its cost stays
closer to linear in work. The same structure that made the split so sharply tuned
(16 -> -22.4%, 64 -> -3.6%, 112 -> -22.0%) is what makes the cost non-linear.

Note also that the earlier decode-only null (0.39% vs 0.40%) was an **artefact of a
smoothed metric**: it derived step time from `gen throughput` averaged over
`decode_log_interval` steps. Per-step wall-clock is noisier and far more
discriminative, and it separates the arms clearly. Two methods disagreeing that way
is a reason to trust the less-smoothed one, not to average them.

Consequence for `pd_colocation_vs_disagg.md` section 3: it rests on the **overlap**
kernel's 3.1% MAPE closed form and explicitly says the story depends on that rather
than on the SM-pre-assignment kernel winning. That caution is now vindicated --
section 3 must not be re-attributed to this kernel, because for this kernel the
claim is measured false.

Throughput in these runs (10197 / 10738 tok/s) is depressed by per-step logging;
identical overhead in both arms, so the fit comparison is fair, but not comparable
to the 25155 / 24946 A/B figures.

##### The two roles do not measure q in the same units: FA3 does not pack GQA

The dispatch gate routes a row to the FA3 role on `packed_qo = qo_len * gqa >= 48`
and `kv_tiles >= 32` (`mixed_dispatch.py`). Those are *packed* units. The two
plans behind the two roles disagree about that:

- persistent / FA2 lineage, `TwoStageHolisticPlan` (`scheduler.cuh:1199`):
  `int packed_qo_len = qo_len * gqa_group_size;` and the three tile classes
  {128, 64, 16} are measured against that packed length. GQA is packed.
- FA3, `PrefillSM90Plan` (`scheduler.cuh:939`):
  `int num_qo_tiles = ceil_div(qo_len, cta_tile_q);` with one work item per qo
  head, carrying upstream's own note two lines down:
  `// NOTE(Zihao): our current FA3 implementation do not fuse query and group`
  `// heads so the group_size in cost_function is always 1`.

The serving path plans the FA3 role unpacked: `_plan_fused_split_inner` passes
`qi_f = cumsum(extend_seq_lens)` with `self.num_qo_heads, self.num_kv_heads`, so
FA3 tiles on token count, not on packed rows.

Consequence for the gate: at gqa 7, `packed_qo >= 48` admits a row with
`qo_len >= 7`, which FA3 then runs as a 128-row tile per qo head -- 7 of 128 tile
rows used, 28 times over. The 5.0x packing win quoted in that module's docstring
was measured on the standalone harness where q was packed by hand; it does not
describe the role as the server plans it. So the answer to "why keep the 128 and
64 tile classes in FA2 instead of routing them to FA3's wgmma" is not a free
choice between two packing-capable mainloops: only one of the two packs, and the
64 class (packed_qo in (16, 64], i.e. the spec-verify shapes) is exactly where
packing matters most.

Two things follow, and both are measurements rather than arguments: the boundary
has to be drawn in the units FA3 is actually planned in, and `MIN_KV_TILES = 32`
has never been measured at all -- unlike `MIN_PACKED_QO`, which has a crossover
behind it (0.78x at 32, 1.13x at 56). `tasks/fused_specdec_workload/gate_boundary.py`
sweeps both roles over (qo_len x kv_len) at the serving head shape and prints
where each wins, which is the gate the dispatcher should encode.

#<a id="symmetric-retracted"></a>
#### RETRACTION: symmetric mode computes FA3 wrong, so its speed numbers are void

Per-row accuracy against the reference on the dumped serving batch (qo=[424, 87, 1]),
`test_fused_server_batch.py`, 2026-10-05:

| config | FA3 row | task0 | task2 |
|---|---|---|---|
| asymmetric, SPLITS=8 | **0.00023 ok** | 0.0021 ok | 0.0026 ok |
| symmetric, SPLITS=8 | **4.04 FAIL** | 0.0029 ok | 0.0031 ok |
| symmetric, SPLITS=64 | 3.81 FAIL | ok | ok |
| symmetric + disjoint smem | 3.92 FAIL | ok | ok |

The FA2 rows are correct everywhere; only symmetric mode's FA3 output is wrong,
and disjoint shared memory does not fix it, so the cause is in the symmetric FA3
path rather than in FA2 aliasing over FA3's `SharedStorage`.

**Everything measured in symmetric mode is therefore void**, including this
document's 0.687x -> 1.003x backfill result and today's 1.59x end-to-end figure
(9,990 vs 6,279 tok/s at sm_fa3=8). A kernel that corrupts a role's output does
less valid work, which is a speedup that cannot be banked. Asymmetric is the only
correct fused path today, and it is exact (2.3e-4).

This also explains the shape of the asymmetric-vs-symmetric gap: it was never a
backfill win of that size, it was partly missing FA3 work.

##### Correction: the first disjoint-smem check was inert; the second one is real

`FLASHINFER_FUSED_DISJOINT_SMEM` is a compile-time `#ifdef`, and the harness had
no `-D` plumbing for it, so setting it as an environment variable measured the
aliased layout and ruled out nothing. Two fixes were needed to ask the question
at all: the harness now translates `DISJOINT_SMEM=1` and `WG_DRAIN=n` into `-D`
flags, and the per-warpgroup slot count follows the drain width instead of always
reserving three -- three disjoint slots need 98352 + 3x69632 = 307248 B against
the 232448-byte opt-in limit, i.e. the configuration could not be built.

With the flag genuinely compiled in (`base = 98352`, one slot, total 168032 B):

| config | FA3 row | task0 | task2 |
|---|---|---|---|
| symmetric, aliased, width 1 | 4.04 FAIL | ok | ok |
| symmetric, **truly disjoint**, width 1 | **2.60 FAIL** | ok | ok |
| asymmetric, aliased, width 1 | **0.00023 ok** | ok | ok |

So shared-memory aliasing over FA3's `SharedStorage` is **not** the cause: the
symmetric FA3 path is wrong with disjoint storage too. What still differs between
the two modes is the FA3 plan's CTA count (132 vs `sm_split`) and the fact that a
symmetric CTA executes the FA2 drain *before* entering FA3's warp-specialised
body, so its three warpgroups reach that body having run different code. The
register-budget version of that hypothesis was tested and eliminated (both
kernels compile to 168 registers); the ordering version -- FA3's producer
warpgroup entering with its pipeline state already advanced -- has not been.

Next probe, cheapest first: run symmetric with `sm_split = num_sm` so that no CTA
drains FA2 before FA3. If FA3 is then exact, the fault is in entering FA3 after
the drain, not in the 132-CTA plan; if it is still wrong, the plan's CTA count is
implicated and `FLASHINFER_FA3_MAX_CTAS` is the lever.

##### Localized: the drain corrupts FA3, not the plan and not shared memory

Symmetric mode with `sm_split = num_sm`, which keeps the 132-CTA FA3 plan but
leaves no CTA draining FA2 beforehand:

| row | rel err |
|---|---|
| FA3, qo=424 kv=4001 | **0.00022 ok** |
| task0 / task2 | nan (expected: nothing drains FA2, 88 rows unwritten) |

So the 132-CTA plan is innocent, and with the earlier disjoint-smem result the
cause is neither the plan nor storage aliasing. What is left: **a CTA that runs
the FA2 drain and then enters FA3's warp-specialised body computes FA3 wrong.**

The three remaining mechanisms, and how to separate them:

1. *Warp-specialisation entry state.* FA3's producer warpgroup does
   `warpgroup_reg_dealloc<24>` while its consumers do `warpgroup_reg_alloc<240>`;
   at the 168-register baseline the released 144 exactly match the claimed 144,
   so the protocol has no margin for warpgroups arriving from different code. At
   width 1 only warpgroup 0 drains, so the three arrive divergent. This is the
   hypothesis the 3-warpgroup drain was built to remove ("makes the pre-FA3
   instruction stream identical"), and it predicts that fixing width 3's FA2
   numerics also fixes FA3 here.
2. *Named barriers.* All 16 are allocated and the roles do not share ids (FA3
   0-7, drain 8, mdo 9-11, queue 12-14, rendezvous 15), so this is unlikely but
   is the cheapest to re-audit with `cuobjdump` on the compiled kernel.
3. *FA3 pipeline state not in the dynamic region.* Disjoint smem moves only the
   dynamic allocation. If any FA3 mbarrier lives in a static `__shared__`
   declaration, disjoint storage would not protect it and the drain could still
   clobber it. Checkable from the SASS/PTX shared-memory map.

Note the ordering consequence: width 3 is not merely a performance change, it may
be the correctness fix for symmetric mode. Its own FA2 numerics have to be right
first (currently rel err 1.39 / 2.38 on the FA2 rows at width 3, against 0.0021 /
0.0026 at width 1).

##### The symmetric FA3 corruption is a RACE, and the aliasing hypothesis is back

Three runs of one unchanged binary, symmetric / width 1 / SPLITS=8 / plan 132:

| repeat | FA3 row | FA2 rows |
|---|---|---|
| 1 | rel err 1.93 FAIL | ok |
| 2 | rel err 2.53 FAIL | ok |
| 3 | rel err 3.31 FAIL | ok |
| (an earlier run of the same binary) | **0.00021 ok** | ok |

Every failure value differs and one run passed, so this is **not** a deterministic
logic error -- wrong addressing or a wrong plan would reproduce exactly. It is a
race, which is why months of counter-, stability- and throughput-based validation
never caught it, and it is confined to FA3: the FA2 rows are exact in every run.

That also means the two flag-based probes proved nothing and the earlier
"aliasing is ruled out" line is withdrawn: `grep -c DFLASHINFER_FUSED_* build.ninja`
returns **0** for both the disjoint and the noop builds, so neither `-D` reached
the compiler even with a fresh `TORCH_EXTENSIONS_DIR`. The harness has a second
build path that ignores the `extra_cuda_cflags` list those were added to.

Leading mechanism, which a race fits better than anything else: a CUTLASS
warp-specialised kernel initialises its pipeline mbarriers in shared memory at
kernel entry, *before* the `blockIdx.x >= sm_split` branch. The FA2 drain then
writes its mainloop slot and, at width 1, its cursor/merge tail at offset 69648 --
inside FA3's 98352-byte `SharedStorage`. FA3 is entered after the rendezvous, so
whether it observes clobbered barrier state depends on timing. Predictions this
makes, in order of cost:

1. A genuinely disjoint layout fixes it. Requires finding the harness's real
   compile path first -- that is the next action, not another env-var run.
2. Moving only the cursor/merge tail past FA3's storage (16 bytes of the 48-byte
   tail is all that is written at width 1) fixes it or narrows it to the slot.
3. Re-initialising FA3's shared state after the rendezvous fixes it regardless of
   layout, and is what width 3 would need anyway.

##### Shared memory is genuinely ruled out; what is left is the FA3 entry state

The disjoint layout is now the source default (`fused_pers_smem_base` returns
`align(kFA3Bytes)` unless `FLASHINFER_FUSED_ALIAS_SMEM`), so this test involves no
compile flag and cannot be defeated by JIT caching. Symmetric, width 1, SPLITS=8,
three repeats: FA3 rel err **3.04 / 3.03 / 4.34, all FAIL**, FA2 rows exact.

So the drain does not corrupt FA3 through shared memory. Combined with the
earlier probes this leaves a short list:

- the **132-CTA FA3 plan is innocent** (`sm_split = num_sm`, same plan, no drain
  -> FA3 exact at 2.2e-4),
- **storage aliasing is innocent** (this test),
- the FA2 rows are correct in every configuration ever run,
- and the corruption is **nondeterministic**, so it is a race rather than wrong
  addressing.

What remains is the state in which a CTA enters FA3's warp-specialised body after
draining: the `warpgroup_reg_dealloc<24>` / `warpgroup_reg_alloc<240>` protocol
has no margin at the 168-register baseline, and at width 1 only warpgroup 0 has
run the drain, so the three warpgroups reach that protocol from different code
and in different register states. That is the one hypothesis consistent with a
race whose magnitude varies run to run.

Note the kernel's own guard blocks the obvious isolation
(`FA3_PLAN_CTAS=8` with symmetric mode is refused, by design, because the mode
needs one slice per CTA), so separating plan size from drain in that direction
requires relaxing the guard behind a diagnostic rather than an env var.

Next, cheapest first: `PTXAS_V=1` on the symmetric build to read the actual
register counts per kernel (the 168 figure predates the warpgroup-barrier and
slot changes); then make all three warpgroups run the same pre-FA3 code, which is
what the width-3 drain does and is the fix this hypothesis predicts.

##### Mechanism found: the fused body spills, so FA3's setmaxnreg contract is a lie

`PTXAS_V=1` on the current symmetric build (disjoint smem, width 1):

```
ptxas info : Used 168 registers, used 16 barriers, 464 bytes cumulative stack size
             464-744 bytes stack frame, 2296-2572 B spill stores, 2456-4392 B spill loads
```

Two facts, both measured rather than inferred:

1. **All 16 hardware barriers are in use.** The id budget is not merely tight,
   it is exhausted, which is why warpgroup-reentrant FA2 had to reuse ids.
2. **The kernel spills**, up to 4.4 KB of spill loads against a 744-byte stack
   frame.

That is the race. Register allocation in CUDA is **per kernel**, not per
warpgroup, and this kernel's body contains FA3's warp-specialised mainloop *and*
the FA2 persistent mainloop, which is register-hungry. So every warpgroup is
compiled at 168 registers with spill slots, and then FA3's producer executes
`warpgroup_reg_dealloc<24>`, asserting it needs no more than 24. Code compiled
for 168 registers plus spills, running after a dealloc to 24, reads whatever is
there. The signature matches exactly: nondeterministic, magnitude varying run to
run, FA3 only, FA2 rows untouched (they run before the dealloc).

It also explains the asymmetry. In asymmetric mode the CTAs that run FA3 never
execute the drain, so all three of their warpgroups reach the dealloc/alloc
protocol in the same state and the lie is consistent; in symmetric mode warpgroup
0 arrives having run the FA2 mainloop and spilled, and the others do not.

**This is a structural argument against this fusion shape**, not a bug to patch:
co-locating a register-hungry FA2 mainloop with FA3's warp-specialised body in
one kernel forces one register allocation over both, and FA3's occupancy protocol
depends on its producer path being small. Three ways forward, and the choice
decides the kernel's architecture:

1. Remove `setmaxnreg` from the fused kernel. FA3 then runs at the fused body's
   allocation, losing the consumer warpgroups' 240-register budget -- i.e. giving
   up some of FA3's own throughput to make the fusion sound.
2. Shrink the FA2 mainloop's register footprint until the common allocation fits
   FA3's protocol. The spill figures say that is a large gap, not a tuning knob.
3. Keep the roles in separate kernels and co-schedule them, which is what the
   measured class boundary already argues for on performance grounds -- but §10
   showed serial dispatch costs -48%, so this means concurrent co-launch on
   disjoint SM sets, i.e. the original pre-assignment shape with the geometry
   problem of §12 rather than one kernel.

##### Draining on FA3's consumer warpgroup is worse, and the phases are not interchangeable

The asymmetric backfill deliberately runs on warpgroup 1 because warpgroup 0 is
FA3's TMA producer sitting at `warpgroup_reg_dealloc<24>`. Applying the same
choice to the symmetric drain makes everything worse, consistently over three
repeats:

| drain warpgroup | FA3 row | task0 | task2 |
|---|---|---|---|
| 0 (current) | 3.04 / 3.03 / 4.34 FAIL | **exact** | **exact** |
| 1 (backfill's choice) | 3.57 / 4.13 / 4.41 FAIL | **1.34 / 0.95 / 1.22 FAIL** | **3.59 / 3.03 / 3.62 FAIL** |

The reason the two cannot be copied across: the backfill runs **after** FA3's
body, when warpgroup 1 is a wgmma consumer holding ~240 registers, while the
symmetric drain runs **before** it, when no `setmaxnreg` has executed at all and
every warpgroup holds the compiled 168. Same warpgroup, different phase,
different correctness.

En route this exposed a real bug in the per-warpgroup slots: the slot index was
`kTidBase / 128`, so a narrow drain on warpgroup 1 addressed slot 1 of a
one-slot allocation and faulted with an illegal access 69632 bytes past the end.
The index is now the drain ordinal. Width 3 was in bounds, but width 1 had been
writing its cursor/merge tail by the same rule, which is a second reason the
earlier width-1 symmetric numbers cannot be trusted.

Remaining hypothesis is unchanged and now the only one standing: the fused body
compiles to 168 registers with up to 4.4 KB of spill loads, so FA3's
`dealloc<24>` / `alloc<240>` protocol is unsatisfiable in this kernel. The test
is to compile the fused build with those calls removed -- which costs FA3 its
240-register consumer budget, and so is a measurement of what the fusion is worth
once the protocol is honest, not a free fix.

##### All four hypotheses eliminated: symmetric mode is retired, asymmetric is the correct path

Removing FA3's warpgroup register protocol from the fused translation unit
(`FLASHINFER_FUSED_NO_SETMAXNREG`, defined in the ext's own source so no flag
plumbing is involved) does **not** fix it: FA3 rel err 3.78 FAIL, FA2 rows exact,
and `ptxas` reports the same 168 registers / 16 barriers / 2.3-4.4 KB spills --
expected, since `setmaxnreg` is a runtime hint rather than an allocation
directive, so its removal changes instructions and not the register count.

The complete elimination record for symmetric mode's nondeterministic FA3
corruption:

| hypothesis | how it was eliminated |
|---|---|
| the 132-CTA FA3 plan | `sm_split = num_sm`: same plan, nobody drains -> FA3 exact at 2.2e-4 |
| FA2 storage aliasing over FA3's `SharedStorage` | disjoint layout made the source default -> still fails (3.04 / 3.03 / 4.34) |
| drain running on FA3's TMA-producer warpgroup | moved to consumer warpgroup 1 -> worse, and FA2 breaks too (1.34 / 0.95 / 1.22) |
| FA3's `dealloc<72>` / `alloc<216>` protocol | protocol removed -> still fails (3.78) |

What is established rather than guessed: the corruption needs a CTA to drain FA2
and then enter FA3's body; it is a race; it never touches the FA2 rows; and none
of the four candidate couplings is responsible.

**Consequences, which are decisions and not further debugging:**

1. **Asymmetric mode is the only correct fused path today** and it is exact
   (2.3e-4). Every symmetric number in this document stays void.
2. Asymmetric's problem is the one measured in section 12: its FA2 role gets
   ~128 threads per SM against ~256 native, because a single launch has one
   block shape and FA3's warp specialisation fixes it at 1 CTA/SM.
3. Those two facts together point at the shape the evidence has been converging
   on from both directions: **two kernels, concurrently co-launched on disjoint
   SM sets, each with its native geometry** -- FA2 at 2 CTAs/SM with mma.sync
   occupancy, FA3 at 1 CTA/SM with its register protocol intact and honest. That
   is not serial dispatch (measured at -48%, section 10); it is the
   `dual`/`ours` arm of `bench_4mode.py`, where both kernels are resident and the
   scheduler splits the grids.
4. It does cost the single-launch property, and with it the ability to fuse spec
   draft and verify inside one kernel. That is a real loss and it has to be
   weighed against a symmetric path that four experiments could not make correct.


<a id="packed-gqa-class-boundary"></a>
### Which class belongs on which mainloop, measured with packing — 2026-10-04

The question "why not give tile 64 and tile 128 to the wgmma mainloop" had no
honest answer while FA3 ran unpacked, because unpacked FA3 loses the short-q
classes for a reason that is an artefact of the missing feature. Packing emulated
at the batch level (`pack_class_boundary.py`: each KV head's group presented as
one head with `qo_len * gqa` rows, legal because the cached-prefix part is
non-causal), same head shape, full device per arm:

| class | qo x kv, rows | persistent | FA3 unpacked | FA3 packed | packing worth | winner |
|---|---|---|---|---|---|---|
| decode / draft | 1 x 4357, 11 | **48.6 us** | 224.7 us | 93.6 us | 2.40x | **persistent, 1.9x** |
| verify k=3 | 4 x 4357, 8 | 96.4 us | 159.6 us | 93.4 us | 1.71x | tie (1.03x) |
| verify k=7 | 8 x 4357, 8 | 137.7 us | 159.4 us | **93.6 us** | 1.70x | **wgmma, 1.47x** |
| small chunk | 128 x 3095, 1 | 236.1 us | 73.3 us | 73.1 us | 1.00x | **wgmma, 3.23x** |
| modal chunk | 500 x 3095, 1 | 195.0 us | 74.6 us | 74.0 us | 1.01x | **wgmma, 2.63x** |
| long-ctx chunk | 500 x 32768, 1 | 940.1 us | 499.2 us | 496.2 us | 1.01x | **wgmma, 1.89x** |
| big chunk | 2048 x 3095, 1 | 362.6 us | 215.6 us | 214.5 us | 1.01x | **wgmma, 1.69x** |

Read off it:

- Packing is worth **1.70-2.40x on the short-q classes and nothing (1.00-1.01x)
  on the chunks**, which is exactly what a tile-occupancy fix should look like.
- **Tile 64 (verify) belongs on the wgmma mainloop once packed**: unpacked it
  loses (96.4 vs 159.6 = 0.60x), packed it wins at k=7 (1.47x) and ties at k=3.
  The previous "keep 64 on FA2" conclusion was an artefact.
- **Tile 128 (all chunk classes) belongs on the wgmma mainloop** by 1.69-3.23x,
  with no packing needed -- those tiles are already full.
- **Tile 16 (decode and draft) stays on FA2**, 1.9x, even packed. Those rows are
  bandwidth-bound, and FA2's split-KV + LPT is the right engine for them. So the
  FA2 overlap machinery is still wanted -- scoped to the 1-token class.

<a id="static-split-confirmed"></a>
### The static SM split is the wrong structure, confirmed on exact numerics — 2026-10-04

> **The original table here was measured in symmetric mode, which computes FA3
> wrong, so it was withdrawn. The sweep was re-run in asymmetric mode, which is
> exact (2.3e-4), and it reaches the same verdict with valid data:**
>
> | arm | tok/s | vs base | mean ITL | mean TTFT |
> |---|---|---|---|---|
> | base, fused off | 24,892.7 | — | 18.50 ms | 2,684 ms |
> | asymmetric, `sm_fa3=64` | 21,305.8 | **-14.4%** | 21.69 ms (+17%) | 3,336 ms (+24%) |
> | asymmetric, `sm_fa3=96` | 21,411.2 | **-14.0%** | 21.58 ms | 3,317 ms |
> | asymmetric, `sm_fa3=116` | 22,034.3 | **-11.5%** | 21.13 ms | 3,176 ms |
>
> **The loss shrinks monotonically as FA3 takes more of the device** -- 64 -> 96
> -> 116 gives -14.4% -> -14.0% -> -11.5% -- which is what a 93/7 chunk/decode
> work mix predicts, and it extrapolates to an optimum of "give FA3 everything",
> i.e. do not split. A static fraction has no winning setting here, with the
> correct kernel, with the corrected gate, at any split tested.


With the gate corrected so the modal chunk reaches the wgmma role (fused dispatch
share 90.6% -> 96.2%), the serving A/B got **worse**, at every split:

| arm | tok/s | vs base | mean ITL | vs base | mean TTFT | vs base |
|---|---|---|---|---|---|---|
| baseline, fused off | 24,861.9 | — | 18.53 ms | — | 2,684 ms | — |
| fused, `sm_fa3=32` | 23,245.9 | -6.50% | 19.91 ms | +7.44% | 2,964 ms | +10.40% |
| fused, `sm_fa3=64` | 23,399.6 | -5.88% | 19.78 ms | +6.73% | 2,927 ms | +9.06% |
| fused, `sm_fa3=96` | 22,949.7 | -7.69% | 20.18 ms | +8.91% | 3,017 ms | +12.38% |

The arithmetic of why, which generalises past this workload: the modal mixed step
is 93% chunk attention and 7% decode attention. A row's wgmma work is
`h_qo * ceil(qo_len/128)` work items -- 112 for the modal chunk -- so on 64 CTAs
it takes two waves and the isolated 74 us becomes ~150 us, while the decode rows
run on the remaining 68 CTAs instead of the device that made them cost 48 us.
**No fixed fraction can be right when the mix is 93/7 and varies per step.**

So SM pre-assignment is not the design that collects the 2.6x. The design that
can is one queue over the whole device with the mainloop chosen per tile class --
which is what §6.3's `cluster_cursor` backfill approximates and what the
4-task `TwoStageHolisticPlan` (flashinfer `ca31e08`) exists to feed: the wgmma
class becomes task 3 of the same plan, inheriting packed-GQA tiling, split-KV and
the JIT-reduction group, and every CTA pulls from the shared queue rather than
owning a slice.

<a id="serial-dispatch-dead-end"></a>
### Serial dispatch: a dead end twice over — 2026-10-04

§9 showed SM pre-assignment loses, and the per-kernel times suggested the two
mainloops run back-to-back on the whole device would give 1.74x on attention
(123.7 vs 215.4 us/layer, launch cost only +1.0 us). That was a **projection from
isolated kernels, and the server measurement killed it**: two repeats of serial
dispatch, same workload, exclusive GPU, gate as corrected:

| arm | tok/s | mean ITL | mean TTFT |
|---|---|---|---|
| baseline, fused off | 24,861.9 | 18.53 ms | 2,684 ms |
| serial dispatch, run 1 | **12,923.9 (-48.0%)** | 18.73 ms | 7,255 ms |
| serial dispatch, run 2 | **12,905.5 (-48.1%)** | 18.75 ms | 7,264 ms |

Mean ITL barely moves while throughput halves and TTFT nearly triples, which is
the signature of a host-bound path, not a slower kernel: serial dispatch plans
*both* wrappers for the full device every step and issues two paged runs per
layer, and this path was already the host-bound one. The projection ignored all
of that because it timed launches with the plans held fixed.

The code is reverted rather than left behind a flag. Beyond the measurement, it
is the wrong structure for this work on principle: **spec draft and verify cannot
be fused if the dispatch is serial.** One launch holding prefill chunk, draft,
verify and decode is the claim; serialising the roles gives it up to collect a
projected gain that does not exist.

So the remaining path is the one the measurements point at from both sides: a
single launch, one queue over the whole device, the wgmma mainloop as task 3 of
`TwoStageHolisticPlan` (flashinfer `ca31e08`) with packed-GQA tiling (worth
1.70-2.40x on the short-q classes, §8), split-KV from the plan's
`kv_start`/`kv_end`, and the JIT reduction the FA2 runners already perform.

<a id="unified-device-spec"></a>
### Device-side spec for the wgmma role on task 3 — 2026-10-05

The plan side is done and compiles for both task counts (flashinfer `ca31e08`).
What remains is device-side, and it is three separable pieces. Order matters:
each is measurable on its own with `test_fused_server_batch.py` (numerics) and
`pack_class_boundary.py` / `row_cmp.py` (speed).

**(a) A tile scheduler over task 3.** The FA3 mainloop's scheduler contract is
small (`hopper/tile_scheduler.cuh`): `Params` of device arrays,
`get_initial_work` / `get_next_work`, and a `WorkTileInfo` whose
`get_block_coord()` returns `{q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr,
kv_indptr, qo_len, kv_len, batch_idx}`. Build one reading task 3 of
`HolisticPlanInfo<4>`, i.e. `tasks[3].{work_indptr, q_indptr, kv_indptr, q_len,
kv_len, q_start, kv_start, kv_end, kv_head_idx, partial_indptr, group_id}`.

**Split-KV needs no mainloop change**: set `kv_indptr += kv_start` and
`kv_len = kv_end - kv_start`. The mainloop walks `[0, kv_len)` of the pointer it
is given, and this path is non-causal (`causal=false`), so a chunk is just a
shifted base with a shorter length. This is the piece to land first -- it is the
one the user is right about costing nothing to try.

**(b) Packed-GQA addressing.** The plan tiles *packed* rows
(`qo_len * gqa_group_size`), so a task-3 work item covers packed rows
`[q_start, q_start + tile)` of one kv head, spanning several qo heads. The
mainloop must map packed row `p` -> `(token = p / G, head = kv_head * G + p % G)`
for both the Q load and the O store. Q therefore cannot come through TMA as a
dense `[CTA_TILE_Q, D]` block: use the existing `FLASHINFER_SKIP_TMAQ` path
(`cp.async`), which exists precisely as this diagnostic and is how the sparse KV
path already loads. Worth 1.70-2.40x on the short-q classes (§8) and ~1.00x on
chunks, so it is what moves the verify class over -- do not expect it to help the
chunk class.

**(c) JIT reduction, shared with the FA2 runners.** When `group_id >= 0` the
epilogue writes `partial_o` / `partial_lse` at `partial_indptr` instead of `o`,
then `atomicAdd` on `group_counter[group_id]`; the work item that observes the
last arrival reduces the group's `merge_indptr` rows in place. `persistent.cuh`
around lines 240-620 is the reference, including the two bugs already paid for
there: the merge must use the *real linear tid* (not `threadIdx`) and needs a
fence before reading partials, or it intermittently merges stale data. The fused
kernel already threads an `s_merge` flag through `fused_drain_queue`, so the FA3
role joins an existing mechanism rather than adding one.

Then, and only then, the two things this exists for: every CTA pulls from one
queue over the whole device (no `sm_split` fraction, which §9 measured as the
wrong structure), and a single launch holds prefill chunk + draft + verify +
decode -- which §10 is the reason not to serialise.

Multicast (§6 of `debug_kernel_server_perf_4_mode.md`) layers on top of (c): the
DSMEM handoff measured 1.33-1.59x in isolation is the same handoff a split-KV
group needs when its chunks sit in one cluster, so the plan should prefer
assigning a group's chunks to cluster-mates. `FLASHINFER_FUSED_CLUSTER=2` still
fails on the fused kernel and the open route is incremental reconstruction from
`dsmem_kv_share.cu`, which clusters fine at 384 threads with the fused kernel's
`__launch_bounds__`; note that `compute-sanitizer synccheck` is not viable here
(3 hours without reaching a launch result, and it silently cost a day of
measurements, §7).

<a id="launch-geometry"></a>
### Why pre-assignment could not win: launch geometry, not the split — 2026-10-04

The FA2 role is crippled inside the fused launch at **every** split, for a reason
independent of how the SMs are divided:

| | FA2 native (`BatchAttention`) | FA2 role inside the fused kernel |
|---|---|---|
| CTAs per SM | 2 (`num_sm *= 2`, "two cta per sm", `scheduler.cuh:1189`) | **1** (`__launch_bounds__(384, 1)`, and ~164 KB smem forces it) |
| threads per CTA for the mainloop | ~128 | 128 of 384 (WG0-only drain) |
| **FA2 threads per SM** | **~256** | **128** |

mma.sync saturates by occupancy -- many warps issuing to hide latency -- while
warp-specialised wgmma is one CTA per SM by construction (384 threads, 3
warpgroups, ~164 KB). A single launch has one block shape and one occupancy for
the whole grid, so the two roles cannot both have native geometry in one kernel.
That is the real reason SM pre-assignment was chosen, and it is correct about the
constraint.

But pre-assignment also freezes the *work fraction*, and §9 measured that a fixed
fraction cannot fit a 93/7 chunk/decode mix that moves every step. Both
constraints are satisfiable at once, and this is the design to build:

1. Keep the uniform geometry: 1 CTA/SM, 384 threads.
2. Make the FA2 role **warpgroup-reentrant** so an SM doing FA2 work uses all
   three warpgroups -- 384 threads/SM against a native 256, which more than
   recovers the lost CTA.
3. Every SM pulls from one queue (`cluster_cursor`), so the chunk/decode split is
   dynamic per step instead of a constant.

The blocker is known and was paid for once already: the persistent mainloop
synchronises block-wide, so running it on three warpgroups deadlocked (exit 124,
GPU at 99%) and the drain was reverted to WG0-only. It needs per-warpgroup named
barriers, and the groundwork exists -- `kPersistentMainloopBarrier + 3 + kWg` are
already allocated for exactly this. Converting the mainloop's `__syncthreads()`
to those barriers is the next piece of device work, ahead of §11(b) and (c),
because without it the wgmma role's 2.63x is collected on one third of the FA2
role's threads.

#### 12.1 First obstacle for warpgroup-reentrant FA2: the named-barrier budget is full

Hopper has 16 named barriers. Current allocation in the fused kernel:

| ids | owner |
|---|---|
| 0-7 | FA3 / CUTLASS reserved (`kQueryEmpty` is 0, which is also `__syncthreads()`) |
| 8 | `kFusedQueueBarrier` (`FirstUserBarrier`) |
| 9-11 | `kPersistentMainloopBarrier + 0..2`, the mdo barriers |
| 12-14 | `kPersistentMainloopBarrier + 3 + kWg`, per-warpgroup queue barriers |
| 15 | `kPersistentMainloopBarrier + 6`, symmetric-mode rendezvous |

So there is **no free id** for three per-warpgroup mainloop syncs, and
`persistent_block_sync` currently hard-codes one id with width 128. Three options,
in the order they should be tried:

1. Reuse each warpgroup's queue barrier (12-14) for its mainloop sync, after
   proving the two are never pending simultaneously inside one warpgroup -- the
   queue barrier is held while pulling work, the mainloop sync while computing a
   tile. If that holds it costs nothing.
2. Make the mdo barriers (9-11) per-warpgroup instead of per-stage, since their
   three ids are already a per-stage split that a reentrant mainloop would have
   to rework anyway.
3. Drop the symmetric rendezvous (15) and re-derive it from the queue barriers,
   freeing one id.

Do not add a fourth warpgroup's worth of ids without re-auditing 0-7: the
`__syncthreads()` == barrier 0 == `kQueryEmpty` collision is what broke symmetric
mode once already, and it is silent -- it shows up as a hang, not an error.

#### 12.2 Is warpgroup-reentrant FA2 even possible? Measured: yes, by aliasing

Two things had to be checked before building it.

**Instruction scope.** `mma.sync` is a *warp*-level instruction -- its fragments
are defined across a warp's 32 lanes -- so three warpgroups can issue it
independently. This is unlike `wgmma`, which is warpgroup-scoped and needs its
own fence / commit_group / wait_group across 128 threads in lockstep. Nothing in
the ISA prevents three warpgroups each running an FA2 mainloop on its own work
item.

**Shared memory**, which is the binding constraint (`/tmp/smem_budget.cu`,
compiled for sm_90a against the JIT-generated config):

| region | bytes |
|---|---|
| FA3 `SharedStorage` (1 K/V stage) | 98,352 |
| persistent tile128 | 49,152 |
| persistent tile64 | 32,768 |
| persistent **tile16** | **69,632** (the largest class) |
| H200 per-CTA budget | 232,448 |
| 3 warpgroups x largest class | **208,896 -> fits, 89.9% of budget** |
| FA3 + 3 x largest, disjoint | 307,248 -> **does not fit** |

So three concurrent mainloop instances fit only if they **alias FA3's region**,
which is what the kernel already does by default (`FLASHINFER_FUSED_DISJOINT_SMEM`
is the opt-out). That is sound because the roles are time-multiplexed within a
CTA: an SM runs FA3 work or FA2 work, never both at once. Note tile16 is the
largest class, not tile128, so the worst case is three tile-16 instances and the
sizing must come from the max over classes rather than from tile128.

What remains for (2) in §12 is therefore mechanical rather than structural:
warpgroup-relative warp and thread indices inside the mainloop (the `ltid`
distinction the merge path already had to make), the three barrier ids from
§12.1, and three smem slots carved from the aliased region.

#### 12.3 Correction to 12.1: the barrier budget was not the blocker

12.1 said three per-warpgroup mainloop sync ids were needed and none were free.
Reading the code rather than the id table: `threadblock_sync_mdo_states` is
**already** per-warpgroup -- `kPersistentMainloopBarrier + int(tid_base / 128u)`,
i.e. 9/10/11 -- and `prefill.cuh:1120` documents the exact deadlock (exit 124,
GPU at 99%) that forced it. The only syncs still hard-coded to one shared id
were `persistent_block_sync()` (8 call sites) and the JIT merge's last-arrival
broadcast.

Both now take the warpgroup's own id, reusing its mdo id rather than asking for
a new one. That is sound for the same reason `__syncthreads()` can reuse barrier
0 throughout a kernel: both are full-warpgroup rendezvous reached in program
order, so they cannot interleave within a warpgroup. No new barrier ids, option 2
of 12.1 for free, and option 3 (dropping the symmetric rendezvous) is not needed.

What actually had to change was shared memory: all three warpgroups' mainloop
storage pointed at one base, so concurrent drains overwrote each other's Q/K/V
tiles and mdo scratch while the cursor and merge flag had already been split per
warpgroup. The drain now carves one `kPersSlotBytes` slot per warpgroup (sized
from the largest tile class, tile 16 at 69632 B), moves the cursor/merge tail
past all three slots, and the launch's dynamic smem request covers
`base + 3 * slot + 48` instead of one slot's worth.

<a id="withdrawn-stale-headers"></a>
### WITHDRAWN: there was no stale header copy — 2026-10-05

> **This section's conclusion was wrong and is kept only because the reasoning
> error is instructive.** The 4-task plan failed to build and the diagnostic
> named the pre-refactor `TwoStageHolisticPlan` signature under
> `~/.cache/flashinfer/90a/generated/...`, which I read as flashinfer's JIT
> compiling a stale copy of the headers. It was not: that path was just the
> include chain in the error context. `grep -c "NUM_TASKS, typename IdType"` on
> the pod's own `include/flashinfer/attention/scheduler.cuh` returned **0**, and
> `find` over the venv and the JIT cache turned up **no** second copy of that
> header. The refactor had only ever existed on the local machine -- I pushed
> `persistent_fa3_fused.cuh`, `unified_wgmma_scheduler.cuh`,
> `batch_attention.cu` and the harness, and never the one header the build
> actually needed.
>
> The lesson is narrower than the one I drew, and more useful: when editing
> across two machines, verify the remote file contains the change before reading
> anything into a build error. One `grep -c` of a distinctive token would have
> replaced the whole investigation. The genuine inert-flag mechanisms recorded
> elsewhere -- env vars standing in for compile-time switches, and torch JIT
> reuse of a cached `.so` -- are unaffected and still stand.


The 4-task plan refuses to build from `BatchAttention.plan()`, and the diagnostic
names the old signature inside
`~/.cache/flashinfer/90a/generated/batch_attention_.../`:

```
RuntimeError: Ninja build failed.
  .../generated/batch_attention_.../...
  inline cudaError_t TwoStageHolisticPlan(void* float_buffer, size_t ...
```

That declaration is the pre-refactor one, without the `NUM_TASKS` template
parameter. So flashinfer's JIT does not compile `TwoStageHolisticPlan` from the
source tree on `PYTHONPATH`; it compiles against a generated/cached include tree
of its own. Two consequences, and the second is the bigger one:

1. Emitting a 4-task plan from the server path needs that copy refreshed -- purge
   `~/.cache/flashinfer/90a/{cached_ops,generated}` for the affected module, or
   point the JIT's include path at the edited tree -- before the unified queue can
   run from sglang at all.
2. **It explains the inert compile flags.** The standalone harness passes
   `-I $FI/include` explicitly in its `load()` call, so header edits reach it --
   which is why every numerics result in this document is trustworthy. The sglang
   path goes through the JIT, so a header edit there may compile against the
   cached copy instead. Any sglang-side measurement of a *header* change needs
   the cache purged first, and `grep -c` of the generated `build.ninja` for the
   expected `-D` or a sentinel is the only way to know it took.

This is the fourth distinct way a change here has silently failed to take effect,
after plain env vars for compile-time switches, torch JIT reuse of a cached `.so`,
and a stale `compute-sanitizer` process skewing every arm. The pre-flight rule in
memory covers all four: prove the change is in the binary before believing a
number.

<a id="unified-plan-coverage"></a>
### The unified plan covers all rows, not the FA2 tail — 2026-10-05

Verified on the pod: `BatchAttention.plan()` with `FLASHINFER_FA3_TASK=1` emits a
**60-entry** 4-task plan (49 for 3 tasks), and the default asymmetric path still
computes the dumped serving batch exactly (FA3 3.4e-4, task0 2.2e-3, task2
2.9e-3) with the scheduler switch, the params wrapper and both plan-length guards
in place.

Before the unified path can be exercised, one structural consequence has to be
handled, and it is not a detail:

**Today the two plans cover disjoint row ranges.** `_plan_fused_split_inner`
builds FA3's plan over rows `[0, n_fa3_rows)` and the persistent plan over
`[n_fa3_rows, B)`, which is why the launch takes `pers_row_offset` and why the
FA2 side's q/o pointers are advanced. Under the unified plan there is **one** plan
over **all** rows, with the wgmma class and the FA2 classes as tasks inside it.
So:

- the persistent plan must be built over the whole batch, not the tail;
- `pers_row_offset` becomes 0, and the FA2 params must stop advancing q/o;
- `n_fa3_rows` stops being a row split and becomes purely informational -- row
  membership is the plan's `idx_qo_kv_len_vec[3]`, decided by
  `packed_qo_len >= FLASHINFER_FA3_MIN_PACKED_QO`;
- FA3's own `PrefillSM90Plan` call disappears from the serving path, which also
  removes one of the two per-step H2D copies that made this path host-bound.

Note what the admission threshold does to the regression batch: rows have
`packed_qo = [2968, 609, 7]`, so at the default 128 **rows 0 and 1 both join the
wgmma class** and only row 2 stays on FA2 (tile 16). That is the intended
behaviour given the measured boundary -- tile 128 and tile 64 both belong on
wgmma once packed -- but it means `test_fused_server_batch.py`'s `task[]` labels
and its `N_FA3_ROWS` assumption need updating in the same change, or the test
will mislabel correct output.

<a id="unified-bringup"></a>
### Bringing the unified queue up: four faults, in order — 2026-10-05

The unified path builds, runs, and the FA2 side of it is correct. With
`UNIFIED=1` (one plan over all rows, `pers_row_offset=0`, `-D` confirmed present
in the generated `build.ninja`), asymmetric, `sm_split=8`:

| row | packed_qo | plan task | result |
|---|---|---|---|
| 0 (qo 424, kv 4001) | 2968 | 3 (wgmma) | **nan, unwritten** |
| 1 (qo 87, kv 87) | 609 | 3 (wgmma) | **nan, unwritten** |
| 2 (qo 1, kv 14) | 7 | 2 (tile 16) | **rel err 2.8e-3, exact** |

511 of 512 q rows unwritten. So task-3 work items are produced by the plan and
executed by nobody, while the FA2 tasks run correctly from the same plan -- which
confirms the plan, the scheduler plumbing and the ext wiring, and isolates the
remaining defect to **who executes task 3**.

Two index spaces have to be reconciled:

- `TwoStageHolisticPlan` distributes work over **clusters**:
  `num_clusters = num_sm * 2 / cluster_size`, i.e. 264 entries in `work_indptr`
  at `cluster_size = 1`, because the FA2 mainloop is planned at two CTAs per SM.
- FA3's tile scheduler indexes `work_indptr[blockIdx.x]` over the **grid**, which
  is 132 CTAs, and in asymmetric mode only CTAs `< sm_split` reach FA3 at all.

So task-3 items landing in clusters 8..263 are never pulled: the CTAs that could
run them either index a different slot or never enter FA3. Nothing is wrong with
the work items themselves.

Fixes, in the order worth trying:

1. **Make the index spaces identical for a 4-task plan**: force `cluster_size = 1`
   and cap `num_clusters` to `num_sm` so cluster index == `blockIdx.x`, then give
   task 3 the cluster range `[0, sm_fa3)` through the plan's existing per-task
   ranges (`FLASHINFER_SM_PARTITION`, `FLASHINFER_SM_SPLIT`). This keeps the FA2
   tasks on the remaining range and costs the FA2 side its 2-CTAs-per-SM
   planning, which is the geometry penalty section 12 already measures.
2. Let every CTA pull its own task-3 items, which is the symmetric structure and
   is blocked by the FA3 corruption of section 12.3.
3. Keep two CTAs per SM for FA2 and have FA3's scheduler index
   `work_indptr[2 * blockIdx.x]`-style, i.e. teach it the cluster mapping rather
   than changing the plan.

#### 15.1 Aligning the index spaces did not change the outcome

With a 4-task plan now indexing one slot per CTA (`num_sm *= 1`) and the wgmma
task pinned to the low cluster range (`FLASHINFER_SM_PARTITION=1`,
`FLASHINFER_FA3_SM_SPLIT_CLUSTERS=8`, both sentinels verified on the pod), the
result is unchanged: 511 of 512 q rows unwritten, tile-16 row exact at 2.5e-3.

That is informative. FA3 writing *nothing* -- rather than writing to wrong
addresses -- means `get_initial_work` returned invalid on every CTA, i.e.
`work_indptr[blockIdx.x] == work_indptr[blockIdx.x + 1]` everywhere. Wrong
packed-row addressing would still have written garbage somewhere. So the
hypothesis to test next is upstream of the scheduler: **task 3 may be receiving
no work items at all.**

Concrete next check, and it is an inspection rather than another theory: dump the
plan. `tasks[3].work_indptr` over its cluster range, plus the per-task item
counts, printed from the ext or read back from the int workspace. Three things it
will distinguish:

1. the routing branch never ran -- `FLASHINFER_FA3_TASK` not seen by the plan
   call, so `NUM_TASKS` was 3 and `idx_qo_kv_len_vec[3]` does not exist (the
   60-entry vector says otherwise, but the ext and the plan are separate calls
   and only the plan's length was checked);
2. routing ran but admitted nothing -- `packed_qo_len >= 128` not satisfied
   because `gqa_group_size` inside the plan differs from the ext's view;
3. items exist but the cluster override put them outside `[0, 8)`, e.g. because
   `sm_partition` was false in that call so the override's guard
   (`sm_partition && task_work[3] > 0`) did not fire.

Case 3 is the most likely: the override is gated on `sm_partition`, which is read
from `FLASHINFER_SM_PARTITION` inside the plan, and the default is **false** --
so if that variable did not reach the plan process the ranges stayed
`[0, num_clusters)` for every task and the MinHeap spread task 3 across all 132
clusters, of which only 8 run FA3.

#### 15.2 Work items are now consumed; the fault moves to Q addressing

With the q extent presented in packed units, the same configuration stops
silently skipping and starts faulting:

```
CUDA error: operation not supported on global/shared address space
```

That is a TMA access outside its descriptor's mapped region, which is what a
packed tile index does to a Q tensormap built over token-indexed rows: tile 15 of
a 424-token row addresses row 1920, far past the descriptor. So the sequence is
now fully explained end to end --

| stage | state |
|---|---|
| plan emits task-3 items | **correct**: 116 items, clusters [0, 7] |
| items reach the FA3 CTAs | **correct** |
| tiles pass the range check | **correct** after the packed-extent fix |
| Q/O addressing | **wrong**: packed rows addressed as tokens -> TMA fault |

The remaining work is the packed row -> (token, head) map, and the fault tells us
where it has to live. FA3 loads Q through TMA, and a tensormap is a dense
strided region, so a packed tile -- whose rows are `(t, g)` pairs with `t = p / G`
and `head = kv_head * G + p % G` -- cannot be expressed as one descriptor at
gqa 7. Q must come in through `cp.async` instead, which is precisely what
`FLASHINFER_SKIP_TMAQ` already does on this path (`sparse_mainloop.cuh` sets
`USE_TMA_LOAD_KV = false` for KV for the same reason: a page gather is not a
dense region).

So the concrete remaining change is: compile the wgmma role with the Q TMA load
replaced by a `cp.async` gather that maps packed row `p` to
`(token = qo_indptr + p / G, head = kv_head * G + p % G)` for both the load and
the epilogue's store. The epilogue's partial-write path for split-KV groups rides
on the same mapping, so the two land together.

#### 15.3 The packed-Q gather, concretely

The Q load sits at `hopper/sparse_mainloop.cuh:275`, one TMA `copy` issued by
`warp_idx_in_warpgroup == 0` under `elect_one_sync`, with
`shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ)` for
completion. Immediately above it the KV loads already do the thing Q now needs:
`copy_if(gmem_tiled_copy_k, k_predicate_fn, tKgKiGroup, tKsKiGroup)` over a
gathered layout, committed with `pipeline_k.producer_commit(...,
cpasync_barrier_arrive)`.

So the change has a working template in the same function. What it must compute,
for packed row `p` inside a tile of `CTA_TILE_Q` packed rows belonging to kv head
`h` of a row whose q tokens begin at `qo_indptr`:

```
token = qo_indptr + (tile_base + p) / G        // G = gqa_group_size
head  = h * G + (tile_base + p) % G
offset = token * q_stride_n + head * q_stride_h
```

with `tile_base = q_tile_idx * CTA_TILE_Q` and the row predicated off when
`tile_base + p >= packed_qo_len`. Three consequences to carry:

1. **`barrier_Q` changes kind.** TMA completion is a transaction barrier; a
   `cp.async` gather completes through `cpasync_barrier_arrive`, as the KV path
   does. The consumer's `barrier_Q.wait` has to match, which is the same
   coupling the `SKIP_TMAQ` branch notes when it says skipping the wait is
   required or the kernel hangs.
2. **The epilogue needs the identical map** for its O store, and the split-KV
   partial write needs it too, so all three move together. Getting one wrong
   shows up as misplaced rows rather than a fault, which is harder to see than
   the fault we have now -- so the regression must check row *placement*, not
   just magnitudes.
3. **`TmaTransactionBytesQ` and the Q tensormap become dead** on this path for
   the wgmma role. They stay live for the ordinary FA3 prefill entry points,
   which keep their dense descriptor, so the change has to be scoped to the
   unified role rather than applied to the shared mainloop unconditionally.

This is the whole remaining gap between the unified queue running and being
measurable: everything upstream is verified (plan emits 116 items into the right
clusters, items reach the FA3 CTAs, tiles pass the range check), and the one
fault left is this addressing.

#### 15.4 The packed Q layout is affine — no gather needed

Reading `get_local_tile_tensor` (`hopper/utils.cuh:65`) changes the shape of the
remaining work. It builds Q's tile tensor as `m_tensor(_, _, head_idx)`, tiled at
`offset = qo_indptr` over `seq_len` — i.e. one head's token-major slab.

For one kv head's group, a packed row `p = t * G + g` maps to
`token = qo_indptr + t`, `head = kv_head * G + g`. **Both are affine strides**, so
the packed view is one CuTe layout with a nested row mode, not a gather:

```
shape  ( (G, seq_len), D )
stride ( (q_stride_h, q_stride_n), _1 )
```

with the base pointer at `(qo_indptr, kv_head * G)`. Flattening the row mode gives
exactly `p = g + G * t` in CuTe's left-fastest order, which is the packed index the
plan's `q_start` already uses. `local_tile` by `CTA_TILE_Q` then works unchanged.

That matters for three reasons:

1. **No `cp.async` gather and no predicate machinery** — the earlier framing
   (mirror the KV page gather) was heavier than necessary. The KV gather needs
   indirection because page ids come from an array; the packed Q view does not,
   because the head is a stride, not a lookup.
2. **TMA is not fundamentally excluded either.** A 3-mode descriptor over
   `(G, seq_len, D)` is dense and strided, so the "a tensormap cannot express
   this" conclusion in 15.2/15.3 was wrong. Keeping TMA would preserve
   `barrier_Q`'s transaction semantics and avoid the consumer-wait coupling
   entirely, which is the cheaper route to try first.
3. The epilogue's O store takes the same layout with `o_stride_n` / `o_stride_h`,
   and the split-KV partial write indexes `partial_o` by packed row already, so
   the three stay consistent by construction rather than by matching three hand
   written maps.

Correction to 15.2 and 15.3: they framed this as needing `cp.async` because a
packed tile "cannot be expressed as one descriptor at gqa 7". That is false — the
confusion was with the KV *page* gather, which genuinely cannot, since its row
indices come from an index array rather than from a stride.

#### 15.5 The packed view indexes correctly but cannot be tiled with `local_tile`

Two separate facts, and only the first is settled.

**Indexing works.** A host test over a 28-head, gqa-7, token-major Q layout walks
every `(t, g)` of the nested view — shape `((G, seq_len), D)`, stride
`((head_stride, token_stride), 1)` — and every packed row lands on
`(off + t) * H * D + (kv * G + g) * D`. So `p = g + G * t` addresses exactly the
right row, with no gather.

**Tiling it does not.** Wrapping that view in `local_tile(..., tile_shape,
make_coord(_, _0{}))` fails to compile:

```
cute/layout.hpp(1572): error: static assertion failed
  "logical_divide: Too many modes in tiler."
```

`local_tile` wants a tiler whose rank matches the tensor's, and mode 0 here is
itself rank 2, so dividing it by a scalar `CTA_TILE_Q` is not the operation it
expects. The underlying reason is real rather than cosmetic: at `G = 7` a
128-row packed tile **straddles heads unevenly** — it spans 18 tokens plus 2 rows,
so no uniform `(G, n)` sub-shape describes a tile. A flat row mode is not
available either, since `G * head_stride = 7D` while `token_stride = 28D`, so the
packed rows are genuinely non-contiguous.

Two ways forward, both small:

1. `zipped_divide` the nested view and index the resulting tile mode, which is
   what `local_tile` does internally but without requiring rank agreement.
2. Skip tiling entirely: the copy already loops over rows of the tile, so give it
   the verified nested view plus `tile_base = q_tile_idx * CTA_TILE_Q` and let each
   row compute `make_coord(make_coord((tile_base + i) % G, (tile_base + i) / G), _)`.
   Predicate off rows past `packed_qo_len`. This keeps the one verified mapping as
   the single source of truth, and the same expression serves the epilogue's store.

Option 2 is preferable: it needs no CuTe divide semantics, and the mapping it uses
is the one the host test already checks.

<a id="split-kv-overflow"></a>
### Split-KV: fixed, and fixture 2 now passes — 2026-10-05

**Resolved.** With `barrier_Q` drained and arrived from one thread, the captured
serving-crash batch is numerically correct:

```
sm_split=8: unwritten (NaN) rows = 0/512
row 0 (FA3    qo=  72 kv=  236) rel err 0.00324  ok
row 1 (task0  qo= 439 kv=  439) rel err 0.00294  ok
row 2 (task2  qo=   1 kv= 5469) rel err 0.00069  ok
PASS
```

`partial_o_nnz = 56`, so split-KV genuinely executed here -- row 2's 5469 KV
exceeds `kv_len_limit` and splits on the FA2 side, whose in-kernel JIT reduction
is pre-existing and comes out at 6.9e-4. That closes the catalogue's "split-KV
never exercised" gap for the FA2 classes. The **wgmma** role's own split path is
still unexercised: rows 0 and 1 carry only 236 and 439 KV, so neither splits, and
its partial write remains unimplemented.

The fault history, since three readings of it were wrong in a row: it was not a
crash (no fault is ever raised), not a partial-O overflow (165 MiB of 2048, and
the earlier overflow reading came from a stale cached extension), but a **hang** --
`barrier_Q` is a ClusterTransactionBarrier and the `cp.async` conversion arrived
on it without satisfying a transaction count, from 32 lanes instead of one. Each
wrong reading came from interpreting a symptom through the previous bug instead
of instrumenting; the 1500-second timeout and the in-plan print are what settled
it.

**Resolved and corrected.** The partial-O overflow was sizing after all, and the
earlier "still overflows at 2 GiB" reading was a stale cached extension. With the
2 GiB workspace, instrumentation from inside the plan says:

```
tasks=4 clusters=132 cluster_size=1 sum_tile_q=320 max_num_kv_splits=168960 partial_o_nnz=56
partial_o wants 165.0 MiB, partial_lse 2.6 MiB, float workspace has 2048.0 MiB
```

`partial_o_nnz` is 56. The head-dimension double-count hypothesised below is NOT
in play, and that section is superseded.

**What the fixture actually does is hang.** Fixture 2 runs the plan, routes
correctly (`row 0 -> FA3; rows 1.. -> persistent tasks [0, 2]`), and then never
returns: `exit=124` from a 1500-second timeout. A hang, not a fault -- which also
explains the serving symptom exactly, where the scheduler stalls, HTTP payloads
truncate, and the process is killed without a CUDA error ever reaching the log.

The cause is the `barrier_Q` protocol, and the kernel's own comment predicted it.
`barrier_Q` is a `ClusterTransactionBarrier`: the TMA path arms it with
`arrive_and_expect_tx(TmaTransactionBytesQ)` and the consumer waits for that
transaction count to be satisfied. The packed load replaced TMA with `cp.async`
and signals `cpasync_barrier_arrive`, which arrives **without** satisfying an
expected-transaction count -- so the consumer's wait never completes. The
`FLASHINFER_SKIP_TMAQ` branch says exactly this: "The consumer's barrier_Q.wait
is skipped to match, or it hangs."

Two ways to fix, and the choice is not arbitrary:

1. **Keep the transaction protocol.** Arm `barrier_Q` with
   `arrive_and_expect_tx(bytes)` for the cp.async bytes the packed gather will
   write, so the consumer's existing wait is satisfied. Smallest change, keeps
   the consumer untouched.
2. **Switch Q to an arrival-count barrier.** Use a plain mbarrier or named
   barrier for the packed path and change the consumer's wait to match. Larger,
   and it diverges the two paths' synchronisation.

Option 1 first. Note this is the same class of error as the barrier-0 collision
that broke symmetric mode: a synchronisation primitive used with the wrong
completion semantics, presenting as a hang rather than an error.


Fixture 2 -- the captured serving batch, `qo [72, 439, 1]`,
`prefix [164, 0, 5468]` -- is the first batch tested that actually splits KV, and
it cannot plan:

```
RuntimeError: aligned_alloc ... Buffer overflow when allocating memory for
holistic_partial_o
```

Three attempts, each ruled out as the cause:

1. `max_num_kv_splits` summed `CTA_TILE_Q` over tasks 0 and 1 only, so the wgmma
   class's groups were unaccounted. Fixed to sum every splitting task
   (128 + 64 + 128 against 128 + 64, 1.67x). **Still overflows.**
2. The harness's float workspace was 512 MiB, sized for the 3-task plan at two
   CTAs per SM. Raised to **2 GiB**. **Still overflows.**
3. One cluster per SM halves `kv_len_limit` (132 clusters to spread over instead
   of 264), which roughly doubles splits per row -- real, but 2 GiB is ~4x the
   original budget and covers that.

So the demand is not merely larger, it is wrong: task 3 registers more partial-O
rows than its geometry justifies. The likely mechanism, and the thing to
instrument rather than reason about: `partial_o_nnz` accumulates
`row_tile_size * num_kv_tiles * num_kv_heads` per group, and for the wgmma class
`row_tile_size` is now counted in **packed** rows (row 1 is 3073 packed rows =
25 tiles) while `num_kv_heads` multiplies on top -- which double-counts the head
dimension that packing already folded into the rows.

Next step is a print of `partial_o_nnz`, the per-group `nchunks` and the byte
requirement under an env flag, from inside the plan. This is also why fixture 1
passed: it yields zero split groups, so it never allocated a task-3 partial row
and the bug stayed latent behind exactly the gap this catalogue flagged.

#### The regression batch has no split-KV groups

Read back from the plan: task 3 holds 116 work items and **0 of them carry a
group id** (`group_id >= 0` count is zero, distinct groups zero). The batch's KV
lengths (4001 / 87 / 14) sit under the chunking limit, so every wgmma work item is
write-through.

Two consequences worth keeping:

1. The unified regression now running tests the **packed-GQA mapping alone**. If
   it passes, the unified queue is functionally correct for non-split batches,
   and nothing about the epilogue's partial path is implicated either way.
2. The split-KV partial write and in-kernel reduce therefore need their own test
   with a long-KV batch -- `kv_len_limit` has to be exceeded for any group to
   exist. The microbenches already have the shapes: `row_cmp.py`'s
   `500 x 32768` row is 256 KV tiles and will split.

So the remaining device work is not on the critical path to a first correct
unified run, which reorders it behind validating what is already built.

<a id="unified-queue-numerically-correct"></a>
### The unified queue is numerically correct — 2026-10-05

```
sm_split=8: unwritten (NaN) rows = 0/512
row 0 (FA3    qo= 424 kv= 4001) rel err 0.00020  ok
row 1 (task0  qo=  87 kv=   87) rel err 0.00145  ok
row 2 (task2  qo=   1 kv=   14) rel err 0.00255  ok
PASS: every row correct at every split, nothing left unwritten
```

The wgmma role is driven entirely by task 3 of our own plan -- packed-GQA tiling,
KV chunking and the JIT-reduction group id -- with `PrefillSM90Plan` out of the
picture. Rows 0 and 1 both execute as packed task-3 tiles (one straddling a head
boundary), row 2 on the FA2 tile-16 task, and every row lands in the right place.

Reproduce (flashinfer `e9d3ef8` + the commits following it, sglang `24dceca2d`):

```
bash tests/sync_to_pod.sh          # all sentinels must be non-zero
FLASHINFER_SM_PARTITION=1 FLASHINFER_FA3_SM_SPLIT_CLUSTERS=8 \
UNIFIED=1 SPLITS=8 FA3_PLAN_CTAS=8 \
TORCH_EXTENSIONS_DIR=<fresh dir> python tests/test_fused_server_batch.py
```

`UNIFIED=1` sets `FLASHINFER_FA3_TASK=1`, plans one pass over the whole batch,
passes `pers_row_offset = 0`, and compiles with
`-DFLASHINFER_FUSED_UNIFIED_WGMMA=1 -DFLASHINFER_FUSED_PACKED_Q=7`.

What this does and does not establish:

- **Does**: the plan, the scheduler, the packed Q load, the packed O store and the
  ext wiring are correct together, on a real dumped serving batch, at the
  measured class boundary (rows with packed_qo 2968 and 609 on wgmma, 7 on FA2).
- **Does not**: say anything about speed. Both the load and the store are per-row
  loops through registers rather than TMA or `cp.async`, chosen so the mapping
  could be validated first. The async forms are the optimisation, and they do not
  change the mapping.
- **Does not**: exercise split-KV. This batch produces 0 groups (15.6), so the
  partial write and in-kernel reduce remain untested and need a long-KV batch.

Next, in order: serving-path wiring in sglang (one plan over all rows,
`FLASHINFER_FA3_TASK=1`, the two `-D`s in the ext build, `pers_row_offset = 0`),
then the A/B against the 24,892.7 tok/s baseline with no `sm_fa3` fraction in
play, then the async copies, then split-KV on a long-KV batch.

<a id="pod-restore"></a>
### Re-establishing the GPU working set — 2026-10-05

The trial `e59d6e2974d3ed90` reached `done` mid-investigation, taking
`/dev/shm/w` with it: the venv, the Qwen2.5-7B weights, the flashinfer tree, the
sglang `python/` tree and every driver script. A replacement was forked to
`f393156fcc0614b3` -- same name `vibecuda`, 1 GPU, same queue
`h100-sxm-80gb.vmss-eastus2-vlmh200.agent.merge` -- and `runs fork` needs
`overrides.attachments: []`, because the inherited attachments come back in a
shape the create endpoint rejects. It lands on the same node (`n125-071-090`) but
with an empty `/dev/shm` and no repo checkouts.

What has to be restored, in order:

0. **the fork's flashinfer tree.** Cloning upstream does not substitute for it:
   `BatchAttention(workspace, kv_layout)` in the fork against a different
   signature upstream, failing with `TypeError: attribute name must be string,
   not 'Tensor'`, plus dependencies upstream has gained (`apache-tvm-ffi`) that
   the fork does not need. The eight-file overlay only applies on top of the
   fork, so the fork's `include/`, `csrc/`, `tests/` and python package have to
   be pushed or kept in a durable tarball;
1. a venv with torch 2.8/cu128 (the system python is 2.9.1/cu131 and the two are
   not interchangeable here -- `serve.sh` used the venv deliberately);
2. the flashinfer tree at `/dev/shm/w/fi`, which `tests/sync_to_pod.sh` populates
   from the committed source with per-file sentinels;
3. the sglang `python/` tree at `/dev/shm/w/python`;
4. the model weights;
5. the driver scripts -- `serve.sh`, the A/B drivers, `uab.sh`, `dmp.sh` -- which
   are reproducible from this document but were never committed, which is the
   avoidable loss here. They belong in `tasks/fused_specdec_workload/` next time.

Nothing in either repository was lost; both are committed. The measurement that
the expiry interrupted is fixture 2 (the captured crash batch), and it is a single
command once the set above exists.

<a id="unified-serving-crash"></a>
### The faulting serving batch, captured — 2026-10-05

`SGLANG_FUSED_DUMP` on the unified arm, first launch:

```
rows 3   qo [72, 439, 1]   prefix [164, 0, 5468]   kv [236, 439, 5469]
packed_qo [504, 3073, 7]   n_fa3_rows 2   n_tok 511   n_pages 675   sm_fa3 8
total pages 6144 (% 4 = 0)   q/o/kv_indices/fa3_sparse pointers all 16B aligned
q shape [512, 28, 128] stride [3584, 128, 1]
```

What this rules out: the page-count alignment that caused the earlier
misaligned-address fault (`% 4 == 0` here) and any pointer misalignment (all four
are 0 mod 16). What it introduces, against the single fixture that passes:

1. **Two rows in task 3**, not one. `packed_qo` 504 and 3073 both clear the
   threshold, so the wgmma role has two independent q ranges in one launch, with
   the second starting at `qo_indptr = 72`.
2. **A row with `prefix = 0`** — row 1 is a fresh chunk where `kv == qo`, so its
   KV extent is shorter than its packed q extent (439 against 3073). Every shape
   the fixture exercises has `kv >> packed_qo`.
3. Row 0 is similar in kind: `kv = 236` against `packed_qo = 504`.

Both of those are untested combinations, and the second is the kind that breaks
bound arithmetic: `get_packed_gqa_tensor` takes its row extent from
`qo_len / G` and `write_packed` from `packed_qo_len / group_size + 1`, which are
fine when the q extent is small relative to the tensor but need checking at a
row that begins mid-tensor and runs to its end (`offset 72 + 440 = 512`, exactly
`nnz_qo`).

This batch is the second fixture to add to `test_fused_server_batch.py`; the
current one cannot reproduce the crash because it has one task-3 row and
`kv >> packed_qo` on every row.

### Unified path crashes under serving traffic — 2026-10-05

The unified queue passes the dumped batch exactly and survived two full serving
runs when its copies were slow (7,693 and 8,379 tok/s). After the
`cp_async`/`barrier_Q` change it comes up, serves briefly, and the scheduler dies:
the bench sees truncated HTTP payloads
(`ClientPayloadError: Response payload is not completed`), the server log has one
batch line and no further progress, and no `mixed_fused_fa3` composition line is
ever printed. No `CUDA error` reaches the log, which is what a hard process death
looks like here rather than a caught fault.

So the standalone regression is necessary and not sufficient: it replays ONE
batch shape (`qo=[424, 87, 1]`), while the server presents many -- several prefill
rows, decode-only steps, varying page counts, and the `kv_c[-1] % 4` alignment
that already caused a misaligned-address fault once on this path.

Route, the one that found the previous three blocking bugs: `SGLANG_FUSED_DUMP`
already exists and writes the first launch's shapes, strides, plan vectors and
index arrays to an `.npz`. Run the unified arm under it, take the faulting batch
into `test_fused_server_batch.py` as a second fixture alongside the current one,
and fix against that. Until then the serving number is unobtainable rather than
bad, and the two slow runs stand as the only honest unified measurements.

<a id="unified-serving-perf"></a>
### Unified queue on the serving path: correct, and copy-bound — 2026-10-05

| arm | tok/s | mean ITL | mean TTFT |
|---|---|---|---|
| baseline, fused off | 24,800.1 | 18.57 ms | 2,697 ms |
| unified queue (`SGLANG_FUSED_UNIFIED=1`) | **7,693.5 (-69%)** | 60.45 ms | 10,275 ms |

The fused share is 96.2%, so the path really is running, and the standalone
regression says it is numerically exact. The loss is the copy implementation, and
the size of it is arithmetic rather than mystery:

- The packed Q load is issued by **one warp** (`warp_idx_in_warpgroup == 0`),
  which must move a 128 x 128 bf16 tile = 32 KB. Vectorised at 16 B that is 2,048
  transactions across 32 threads, 64 serialised round trips per thread, against
  **one** TMA instruction before. The O store is similar, spread over
  `NUM_COPY_THREADS` but still a per-row scalar read of swizzled smem.
- There are 116 wgmma work items per layer in this batch, so that cost is paid
  116 times per layer, 28 layers per step.

So the measurement does not indict the unified queue; it indicts a deliberately
naive copy that was written to validate the mapping. The mapping is validated.

What must change, in order:

1. **Tiled `cp.async` for the packed Q load**, issued by the whole producer
   warpgroup rather than one warp, with the gather expressed through the same
   packed view. The KV path in the same function is the template:
   `copy_if(gmem_tiled_copy_k, predicate, src, dst)` committed with
   `cpasync_barrier_arrive`. This is where the 69% lives.
2. **Tiled store for O**, same reasoning, using `tiled_copy_O` with the packed
   view as destination instead of the per-row loop.
3. Re-measure. Only then is the unified queue's real cost known, and only then
   does the comparison against -11.5% (static split) and the +20% Amdahl ceiling
   mean anything.

Recorded as a negative result with its cause rather than as a verdict on the
architecture, because the two are separable here and the evidence separates them:
same kernel, same plan, exact numerics, and a copy that is two orders of magnitude
off the instruction count it replaced.

#### 17.1 Why the tiled copy needs an indirection table (or a G-aligned tile)

The obvious fix for 17 is to replace the single-warp loop with the tiled
`cp.async` the KV path uses. That needs a `(CTA_TILE_Q, D)` source tensor to
partition across the producer threads — and the packed view cannot provide one at
`CTA_TILE_Q = 128` with `G = 7`:

- address of packed row `p` is `(p / G) * token_stride + (p % G) * head_stride`,
  which is affine in `(t, g)` but **not** in `p`, because of the wrap at `G`;
- a tile is a contiguous range of `p`, so over a tile the address is piecewise
  affine with a discontinuity every 7 rows — no single stride, which is the same
  reason `local_tile` refused it (15.5);
- it *would* be expressible as `((G, n), D)` if every tile were G-aligned, i.e.
  if the tile covered whole tokens.

So two real options, and they trade different things:

1. **Indirection table, the KV pattern.** Compute 128 row offsets per work item
   into smem (cheap: 32 threads, one divmod each), then do the tiled `cp.async`
   through that table exactly as the paged KV read does through `kv_indices`.
   Keeps `CTA_TILE_Q = 128` and the existing mma shape. Costs one small smem
   array and a divmod per row, amortised over 128 x 128 elements.
2. **Make the wgmma class's tile a multiple of G.** At gqa 7 that is 126 instead
   of 128, and then the packed view tiles directly as `((7, 18), D)` with no
   indirection at all. But `CTA_TILE_Q` is a kernel-traits constant bound to the
   mma tile shape, so this changes the wgmma geometry per GQA ratio — fine for a
   fixed deployment, awkward as a general kernel.

Option 1 is the one to build: it is local to the load, uses a pattern already
proven in the same function, and does not make the tile shape depend on the
model. Option 2 is worth remembering if a single deployment ever justifies
specialising.

Note this is the same obstacle in a third guise -- 15.5 met it as `local_tile`
refusing a scalar tiler, 15.2 as a TMA descriptor fault, and now as a tiled copy
needing a partitionable source. The underlying fact each time: at gqa 7 a 128-row
packed tile straddles heads unevenly.

#### 17.2 cp.async alone recovered 9%; the rest was a self-inflicted pipeline drain

| unified arm | tok/s | mean ITL | TTFT |
|---|---|---|---|
| register round trips | 7,693.5 | 60.45 ms | 10,275 ms |
| cp.async + `cp_async_wait<0>` | 8,379.5 (+8.9%) | 55.42 ms | 9,400 ms |
| baseline | 24,885.3 | 18.51 ms | 2,681 ms |

Converting the load to `cp.async` moved it 9%, which is far too little, and the
residual does not fit the copy volume: 116 work items x 32 KB is 3.7 MB per
layer, under 2 us at HBM bandwidth, against a step that is **+37 ms**.

The cause is in the completion signal I used, not the copy. `cp_async_wait<0>`
drains **every** outstanding cp.async group in the thread, and this producer has
already committed the K and V stages through `pipeline_k`/`pipeline_v` with
`cpasync_barrier_arrive`. So each of the 116 work items per layer flushed the KV
prefetch pipeline and serialised the mainloop against it. The fix is to signal
through `barrier_Q` with `cpasync_barrier_arrive`, which is precisely what the KV
loads next to it already do.

Worth stating as a rule rather than an incident: in a warp-specialised mainloop
with a software pipeline, any blanket `cp_async_wait<N>` is a cross-pipeline
barrier. Completion must be expressed per consumer, through that consumer's
barrier.

<a id="spec-sync-audit"></a>
### Spec path: where the CPU-GPU syncs actually are — 2026-10-05

Audited rather than assumed, because the v1 and v2 workers differ sharply.

**`eagle_worker_v2.py` (the overlap path) has none.** No `.item()`, `.cpu()` or
`tolist()` in the file. `draft()` writes the tree mask and positions straight into
the verify CUDA-graph buffers with `build_tree_kernel_efficient`, returns
`seq_lens_sum=None, seq_lens_cpu=None`, and plans verify on a side stream the
compute stream waits on. Verify metadata is already device-resident there.

**`eagle_worker.py` (v1) had one on the hot path**, now fixed:

| site | what it was | status |
|---|---|---|
| `:454` draft step | `torch.sum(batch.seq_lens).item()` — a device sync **every draft step** | **fixed**: reads the `seq_lens_cpu` mirror |
| `:926-928` draft extend | already prefers `seq_lens_cpu`, falls back to a device sum | fine |
| `:643-647` verify | `retrive_next_token.cpu()` and friends — but only under `if batch.has_grammar` | not on the default path |
| `:310` | `global_need_forward[0].item()` after an all-reduce on the **CPU** group | not a device sync |
| `:812-813` | `.tolist()` on logprobs and verified ids, at output assembly | after the step, not in it |

So the structural claim behind a 4-mode schedule holds: with
`--speculative-eagle-topk 1` the draft is a chain, every row grows by exactly
`k+1`, and the verify metadata is length-deterministic — it needs nothing from
the draft's output except the token ids, which stay on device through the future
map. The remaining blocker to measuring it is not a sync but
`mix_with_running` crashing when the running batch is a spec batch.

<a id="schedule-quality-4-mode"></a>
### Schedule quality on a 4-mode batch: the three properties, measured — 2026-10-05

`tasks/fused_specdec_workload/schedule_quality.py`, read back from the plan the
kernel runs. Batch: 1 chunked-prefill row (500 over a 2595 prefix), 8 verify rows
at k=3, 8 draft rows, 11 decode rows, gqa 7, 132 clusters.

**Near-zero tile waste** (occupied / allocated tile rows):

| tiling | all four modes | verify class alone |
|---|---|---|
| unpacked, as `PrefillSM90Plan` tiles it | **13.9%** | **6.2%** |
| packed, as the unified plan tiles it | **54.8%** | **43.8%** |

3.9x less waste overall and 7.1x on the verify class, which is the mode the
unpacked tiling hurts most: a 4-token verify row occupies 4 of 128 tile rows and
pays that 28 times, once per qo head, against 28 of 128 in a single packed item.

**Near-perfect load balance**, with a caveat that matters:

| task | items | live clusters | imbalance (max/mean) | idle |
|---|---|---|---|---|
| 0 (tile 128) | 64 | 64 | **1.000** | 68 |
| 2 (tile 16) | 76 | 76 | **1.000** | 56 |
| 3 (wgmma) | 112 | 8 | **1.000** | 124 |

Every live task is balanced exactly. But no single task fills the device -- 68,
56 and 124 clusters sit idle respectively -- which is the quantitative reason the
classes have to **co-reside** rather than each owning a slice, and why the
measured static split loses at every ratio (section 9). The union saturates; no
member does.

**Smaller sync surface**: unified pays 1 plan, 1 H2D and 1 launch per step. The
split design pays 2 plans and 2 H2D copies and must keep two schedules mutually
consistent; serial dispatch pays 2 of each plus a second launch, measured at -48%.
On the spec side, the draft step's device sync is now gone (section 18) and v2's
verify metadata is already device-resident, so a 4-mode step needs no host round
trip between draft and verify.

<a id="fused-epilogue-design"></a>
#### The tree build as an optional epilogue task

The JIT-reduction framing makes this natural. A split-KV group's last arrival
already runs a *second kind of work* in the epilogue -- the reduction -- chosen per
work item rather than per kernel. `build_tree_kernel_efficient` is the same shape
of thing: it consumes the verify group's output and produces the next step's tree
mask and positions, it is data-parallel over rows, and today it is a separate
launch between draft and verify.

So represent it as an **optional epilogue stage**, keyed off the work item the way
`group_id >= 0` keys the reduction:

- the plan marks the verify class's groups with a flag meaning "after reducing
  this group, build its tree";
- the last arrival, already holding the reduced O and LSE in registers and smem,
  writes the mask and positions into the same CUDA-graph buffers
  `get_verify_buffers_to_fill_after_draft()` hands out today;
- nothing else changes: the reduction's completion ordering is the ordering the
  tree build needs, so no new barrier appears.

What this buys, in the paper's terms: the tree build stops being a kernel boundary
and becomes a task, so the 4-mode step is one launch with no host round trip -- the
property section 18's audit shows is otherwise already within reach but still
split across two kernels.

<a id="audit-superseded"></a>
### Audit (superseded by the catalogue at the top) — 2026-10-05

| optimisation | state | evidence |
|---|---|---|
| **FA3 packed GQA** | **tested and implemented** | boundary measured (`pack_class_boundary.py`): 2.40x decode/draft, 1.70x verify, 1.00-1.01x chunks. Implemented in the unified role and numerically exact on the real batch (0/512 unwritten, 2.0e-4 / 1.45e-3 / 2.55e-3) |
| **do we still need tile 128 and 64 in FA2** | **tested, answered: no for 128/64, yes for 16** | tile 128 -> wgmma by 1.69-3.23x; tile 64 (verify) loses unpacked (0.60x) and wins packed (1.47x at k=7, 1.03x at k=3); tile 16 stays on FA2 at 1.9x even packed, because decode is bandwidth-bound |
| **unified load-balance scheduler** | **tested** | per-task imbalance 1.000 on every live task; tile waste 13.9% -> 54.8% overall, 6.2% -> 43.8% on verify; sync surface 1 plan / 1 H2D / 1 launch vs 2/2/1 and 2/2/2 |
| **unified split-KV** | **plumbed, never exercised** | scheduler carries `kv_start`/`kv_end` and the chunk is a shifted `kv_indptr`; but every test batch yields **0 split groups** (KV under `kv_len_limit`), so no run has used it. Needs a long-KV batch -- `500 x 32768` is 256 tiles and will split |
| **JIT reduction** | **untested for the wgmma role** | the FA2 runners' in-kernel merge is pre-existing and exercised; the wgmma role's partial write is **not implemented** -- `partial_indptr` and `group_id` are carried to it and ignored. Blocked behind split-KV producing a group at all |
| **multicast (DSMEM)** | **measured in isolation, not integrated** | 1.587x cold, 1.326-1.364x L2-resident in `dsmem_kv_share.cu`; `FLASHINFER_FUSED_CLUSTER=2` still faults on the fused kernel, 11 candidates eliminated, cause unresolved |
| **fused epilogue (tree build as a task)** | **designed, not built** | 19.1: keyed off a plan flag the way the reduction is keyed off `group_id`, writing into the CUDA-graph buffers that already exist, no new barrier |
| **4-mode serving schedule** | **not reached** | blocked on `mix_with_running` crashing when the running batch is a spec batch; the sync audit (18) shows nothing else in the verify-metadata path forces a host round trip |

So: two of the individual questions are fully answered with measurements, the
scheduler's three claimed properties are measured, and three items -- split-KV,
the wgmma JIT reduction, and multicast integration -- have **never run**. The
fused epilogue and the 4-mode serving schedule sit behind those.

Honest consequence for the order of work: testing the fused epilogue and the
serving schedule "after all these individual optimisations are visited" means
split-KV and the JIT reduction have to be exercised first on a long-KV batch,
because they are the two that are plumbed-but-dark, and a serving number measured
before then would again be measuring an incomplete kernel.


<a id="split-kv-dark"></a>
### The wgmma role's split-KV path is a latent wrong answer, not just unexercised — 2026-10-06

The catalogue carried this as "the wgmma role's own split path still
unexercised", which understated it. Unexercised would be safe. This is a wrong
answer waiting for a long enough row, and the >64k A/B is precisely the run that
would find it.

Three facts, each checked in the source rather than inferred:

1. **The plan does chunk task 3.** The chunking loop in `scheduler.cuh`
   (~1451-1530) runs over every task in the plan, `kFA3Task` included. A task-3
   row whose `kv_len` exceeds `kv_len_limit` is therefore split into
   `ceil_div(remaining_len, kv_len_limit)` work items, and each gets its own
   `partial_indptr` and a shared `group_id`.
2. **The unified scheduler faithfully carries the split.** `UnifiedWgmmaTileScheduler`
   returns `partial_indptr` and `group_id` in its `WorkTileInfo`, and even defines
   `is_split_kv()` as `group_id >= 0`.
3. **The kernel consumes none of it.** `persistent_fa3_fused.cuh` contains no
   reference to `partial_o`, `partial_lse`, `partial_indptr`, `group_id` or
   `final_o`, and nothing anywhere calls `is_split_kv()`.

So every chunk of a split task-3 row computes attention over its own KV slice and
writes that slice's output straight to the final O rows, with no partial buffer
and no merge. The chunks overwrite each other and the surviving row is one
chunk's answer -- softmax-renormalised against only its own slice, so wrong in
value, not merely stale.

**CORRECTED, and this is the important correction in this entry.** I wrote that
fixture 2 was quiet because its FA3 row is `kv=236` and never splits. Running the
guard against fixture 2 refutes that: it fires, in both completion-protocol arms,
at `scheduler.cuh:1495`. The arithmetic I should have done: fixture 2 routes
*two* rows to task 3, `qo=72` (packed 504) and `qo=439` (packed 3073), whose KV
totals 236 + 439 = 675 over 8 FA3 clusters, giving a `kv_len_limit` near 84 -- so
the 439-long row splits into roughly six chunks.

The consequence is that **fixture 2's earlier PASS is now suspect**, not
reassuring. It reported rel err 3.2e-3 / 2.9e-3 / 6.9e-4 while, by this analysis,
its task-3 rows were being split with no reduction behind them. One of two things
is true and they are not equally good: either the `partial_o_nnz=56` I attributed
to the FA2 classes was partly task 3's and the overwrite happened to be benign for
this shape, or the guard's predicate over-fires relative to what the chunk loop
actually emits. Resolving that is the first step of the reduction work, not an
afterthought -- and either way the earlier claim that the wgmma role "has never
been handed a long row" is withdrawn.

This also makes the reduction the hard blocker rather than one item among several:
with the guard in, fixture 2 itself no longer runs, so there is currently no
fixture on which to validate the Q completion protocol either.

**CORRECTED: it did not bind at 8192.** I predicted this would also break the
ordinary serving workload, reasoning that at `--random-input-len 8192` with
`--chunked-prefill-size 512` and gqa 7 a task-3 row carries `packed_qo = 3584`
and `kv_len` up to 8192 against a `kv_len_limit` of roughly 1024 over 8 FA3
clusters. The measurement says otherwise: with the guard compiled in, the
unified arm ran all 200 prompts to completion (96.2% of rows through the fused
path), so no task-3 row was split. The earlier unified serving numbers are
therefore **not** impeached on this account -- they are slow, not wrong.

What remains open is *why*, and there are two possibilities with very different
consequences: either no task-3 row in this workload exceeds its
`kv_len_limit`, or the guard is inert because its predicate does not evaluate
where I think it does -- in which case the silent wrong answer is still
reachable and simply was not triggered. A guard that has never fired proves
nothing. The gate fixture below is what distinguishes them, and it has not yet
been run.

**What the fix has to do.** The partial write is smaller than it looks, because
`CollectiveEpilogue::store` (`hopper/epilogue.cuh:204-258`) already stages the
output tile into smem `sO` and already has LSE in registers. The partial
destination is a dense row-major slab, not the packed `(G, seq)` layout, so
`store_partial` is `store` with two destination substitutions and nothing else:

| | final path | partial path |
|---|---|---|
| O | `write_O<...>(epilogue_params.O_ptr, ..., layout_O, ..., qo_head_idx, qo_indptr, qo_len, ...)` scattering into `(G, seq)` | dense copy from the same `sO` to `partial_o + partial_indptr * HEAD_DIM_VO`, `row_tile_size = min(CTA_Q, qo_len - qo_tile_idx * CTA_Q)` rows of `HEAD_DIM_VO` |
| LSE | `gLSE(row) = lse(mi)` via `get_lse_local_tile_tensor` | `partial_lse[partial_indptr + row] = lse(mi)`, same `tOcO_row` loop and the same `row < qo_len - qo_tile_idx * CTA_Q` bound |

`partial_o` is `DTypeO` and `partial_lse` is `float`, matching
`MergeGroupJIT<HEAD_DIM>(partial_o, partial_lse, final_o, final_lse,
merge_indptr, merge_o_indices, row_begin, nrows, kv_head_idx, num_kv_heads,
gqa_group_size, tid, nthreads)` in `persistent.cuh:250`.

**The seam for the per-work-item fields.** `block_coord` is destructured with
fixed arity 8 by the shared FA3 body, so `partial_indptr`/`group_id` cannot ride
along it without breaking the other scheduler. They are already on
`WorkTileInfo`, which `fa3_prefill_body` holds, so the route is a C++17 SFINAE
trait over `decltype(std::declval<typename S::WorkTileInfo>().group_id)` --
false for the upstream scheduler, true for ours -- and an `if constexpr` at the
`collective_epilogue.store(...)` call in `hopper/prefill_sm90.cuh:299`. That
needs no upstream edit.

**And the handshake** (mirroring `persistent.cuh` ~575-640, the working reference
on the FA2 side):

- epilogue: when `partial_indptr >= 0`, store to
  `partial_o + partial_indptr * HEAD_DIM_VO` and `partial_lse + partial_indptr`
  instead of the final O rows, keeping the packed-GQA row mapping;
- then the release/acquire handshake in that order and no other --
  `__threadfence()` per lane, block rendezvous, `atomicAdd(group_counter[gid])`,
  broadcast "am I last" over the warpgroup's own named barrier (not
  `__syncthreads_or`, which traps with one warpgroup alive), `__threadfence()`
  on the acquire side, `MergeGroupJIT`, and hand the counter back at zero so the
  next layer's producers can win it;
- the merge needs `merge_indptr`/`merge_o_indices`/`group_row_begin`/
  `group_nrows`/`group_kv_head`, which the FA3 params do not currently carry and
  which must come in alongside `wgmma_plan`.

**The fixture that gates it.** Fixture 2 cannot validate this work because its
FA3 row never splits, so the validation gate is a third fixture whose task-3 row
is long enough to be chunked. Driving `fixture_run.sh` with

```
FIXTURE='qo:512,1 prefix:16384,0' UNIFIED=1 SPLITS=8 FA3_PLAN_CTAS=8
```

puts a `qo=512` row (packed `512 * 7 = 3584`, so task 3) against a 16 K prefix,
which the plan splits at a `kv_len_limit` on the order of 2 K across 8 FA3
clusters, while the second row keeps an FA2 class in the batch so the two
reduction paths are exercised in one launch.

The gate has two sides, and both matter:

- **today** it must fail at plan time with the guard's message
  ("a task-3 (wgmma) row needs KV splitting"). If it instead runs, the guard's
  predicate is wrong and the plan is still capable of emitting a silently wrong
  answer;
- **after the fix** it must match the reference to the same tolerance fixture 2
  holds (rel err ~3e-3 for the FA3 row) *and* report `partial_o_nnz` larger than
  fixture 2's 56, since the FA3 chunks now claim partial rows of their own.

Until that lands, the honest statement about long context is that the kernel has
no correct long-context path at all on the wgmma role.

<a id="env-contamination"></a>
### The restored pod's serving tree was upstream sglang, not our fork — 2026-10-05

Worth recording because it invalidates a class of measurement rather than one
number, and because the previous entry in this file says the fork tree was
"pushed (805 files, byte-exact)" -- which was true of a tarball and false of the
tree the server actually imported.

`/dev/shm/w/sgl_src` on the replacement pod is a git clone of **upstream**
sglang at `1818db4`, and `/dev/shm/w/python/sglang` -- the tree on `PYTHONPATH`
-- was built from it with my four modified files copied on top. An md5 manifest
of our fork's `python/sglang` (725 `.py` files) compared against it:

| | files |
|---|---|
| ours | 725 |
| on the pod | 4140 |
| extra on pod | 3573 |
| missing on pod | 158 |
| present in both, different contents | 514 |

`sglang/bench_serving.py` is in that mismatched set, so the benchmark client was
not ours either. The practical consequence is that the serving arms launched on
this pod were upstream 0.5.21 plus four overlay files: the fused kernel path
would have been exercised, but none of our fork's other sglang changes --
including the gains the predecessor effort is built on -- were present. No
serving number may be read from the pod until the tree is rebuilt.

What is **not** affected: the fixture tests, including fixture 2's pass and the
`barrier_Q` diagnosis. Those import `flashinfer` and `torch` only, never
`sglang`, and `/dev/shm/w/fi` is our flashinfer fork.

The repair is a chunked transfer of our 725-file tree over the webshell
(2.96 MB of base64 in 38 pieces, each written to its own numbered file so a
retried piece overwrites rather than appends, with an md5 check on the assembled
tarball). It is chunked because both forks are private and this machine has no
push credentials: `git push` and a pod-side `git clone` both fail with
`could not read Username for 'https://github.com'`.

**The dependency chase that preceded it**, recorded so it is not repeated: the
restored venv needed `sglang[srt]` for its closure, which cannot come from a
source tree (`sgl_kernel` especially). Installing plain `sglang[srt]` brings the
**latest** release and drags torch forward, so every compiled torch extension
then mismatches the pinned torch 2.8.0 -- surfacing one failed server start at a
time as `operator torchvision::nms does not exist`,
`_torchaudio.abi3.so: undefined symbol: torch_library_impl`, and
`sgl_kernel/sm90/common_ops.abi3.so: undefined symbol:
c10_cuda_check_implementation`. Pinning the family to our tree's own pins fixes
it: `torch==2.8.0+cu128`, `torchvision==0.23.0`, `torchaudio==2.8.0`,
`torchao==0.9.0`, and `sgl-kernel==0.3.16.post4` (the package pip installs by
default is named `sglang-kernel`, a different distribution). Installing the
pinned *release* `sglang[srt]==0.5.4.post2` does not work as a shortcut: it tries
to build `flashinfer_python` from source and fails, and we do not want its
flashinfer anyway. An ABI sweep over every torch-dependent import is the cheap
way to confirm the result (`ABI_BAD=none`) instead of discovering the next gap
through another five-minute server start.


<a id="clean-tree-ab-2026-10-05"></a>
### Unified vs baseline on a verified tree, 2026-10-05 — 2026-10-05

The first serving A/B whose software stack is known to be ours (manifest-checked
`EXTRA=0 MISSING=0 MISMATCH=0` against our 725 files -- see
[#env-contamination](#env-contamination)). Both arms, same pod, same binary,
back to back.

| | baseline | unified | ratio |
|---|---|---|---|
| output throughput (tok/s) | 590.58 | 197.65 | **0.335x** |
| mean TTFT (ms) | 2626.50 | 9333.23 | 3.55x |
| mean ITL (ms) | 18.26 | 55.01 | 3.01x |
| P99 ITL (ms) | 32.86 | 114.50 | 3.48x |
| benchmark duration (s) | 33.21 | 99.22 | 2.99x |
| rows through the fused path | n/a | 53116 (96.2%) | — |

Reproduce: `drivers/uab.sh` on pod `f393156fcc0614b3`, Qwen2.5-7B,
`--dataset-name random-ids --random-input-len 8192 --random-output-len 192
--num-prompts 200 --request-rate 12 --max-concurrency 32`, server
`--chunked-prefill-size 512 --max-running-requests 64 --disable-radix-cache
--enable-mixed-chunk --mem-fraction-static 0.80`. sglang `88eff4a78`
(tree = `origin/attn_fuse_` plus the six locally-modified python files);
flashinfer `b1a888e` (Q hoisted above the first K tile, plan guard in).

**What this establishes.** The 96.2% composition means the fused path really did
carry the work, so this is a measurement of the kernel and not of a dispatch
accident. The deficit is 3x, which matches the cp.async drain's cost exactly,
and the hoist was verifiably in the build (`.so` newer than the header,
`load_q_tile` present, `FLASHINFER_FUSED_Q_LATE` unset) -- so **ordering is
ruled out as the cause**. The fence sweeps every cp.async the producer still has
uncommitted from earlier work items, not merely the one K tile that was moved
past it, and in-order group retirement means no `wait_group` bound can exclude
them.

**What it does not establish.** Nothing about the >=30% target. At 8192/192 with
attention at ~32% of a step, a 3x attention regression dominates everything else
and no amount of scheduling quality shows through. The target regime remains the
>64k long-context run, and that is still gated on the Q completion first.

**Next, in order.** (1) Fix Q's completion so it costs nothing: the per-lane
`cp.async.mbarrier.arrive` is the right scope but hung the server on its first
fused batch, so it is gated off behind `FLASHINFER_FUSED_Q_MBARRIER` and must be
settled on fixture 2 -- which reproduced the earlier `barrier_Q` hang in seconds
-- rather than in another serving run. The specific thing to check is whether
anything besides the producer arrives on `barrier_Q`, because arming it for 32
breaks the phase bookkeeping if so. (2) Re-run this A/B. (3) Only then the >64k
run, which additionally needs [the wgmma reduction](#split-kv-dark).


<a id="task3-redundant-work"></a>
### 200 chunks, a correct answer, and the hypothesis that reconciles them — 2026-10-06

The exact guard -- keyed on emitted group ids rather than the `split_kv`
predicate -- reports **200 task-3 work items carrying a JIT-reduction group id**
on fixture 2, in both completion-protocol arms (`scheduler.cuh:1601`). So the
splitting is real and heavy, not an artefact of a wide predicate.

That sharpens the contradiction rather than resolving it. Fixture 2 passed at rel
err 3.2e-3 / 2.9e-3 / 6.9e-4 with `0/512` unwritten rows. If 200 task-3 items
were each computing a distinct KV slice and writing it to shared final O rows
with no merge, row 0 would hold the attention over one slice only -- an error of
order 1, not 3e-3. Both observations are solid, so one of my assumptions about
what those items contain must be wrong.

**Leading hypothesis: the chunks are redundant, not partial.** The unified
scheduler derives each item's range from `params.kv_start[work_idx]` and
`params.kv_end[work_idx]`. If the plan emits those as the row's *full* range for
every chunk of a group -- rather than the chunk's slice -- then all 200 items
compute the same complete attention and write the same correct values. Identical
overwrites are invisible to a correctness check, which is exactly what fixture 2
reports.

This would explain both results at once, and it carries a large performance
consequence: fixture 2's task 3 would be doing on the order of 200 work items
where ~34 q-tiles of genuine work exist, i.e. roughly 6x redundant attention on
the wgmma role. That is the same order as the 3x end-to-end serving deficit, and
it is *independent of* the Q-completion drain -- which would make it a second,
larger cause that the drain's 3x estimate has been masking.

**How to settle it, cheaply and before any more kernel work.** Dump
`kv_start`/`kv_end`/`group_id`/`partial_indptr` for task 3 on fixture 2 (the plan
already has `FLASHINFER_PLAN_VERBOSE`; this needs a per-item print, not a
summary) and compare against the row's true range. Three outcomes:

- ranges are the row's full extent -> redundant work confirmed. The fix is in the
  plan, not the epilogue, and the reduction becomes *cheaper* than specced
  because correct partial ranges plus a merge replace 6x duplicated work;
- ranges are true slices -> then fixture 2's pass is the thing to distrust, and
  the overwrite must be benign for a reason not yet identified. Do not build on
  that fixture until the reason is known;
- ranges are slices for some groups and full for others -> the plan is
  inconsistent between tasks, which is its own bug.

Until this is settled, the reduction spec in [#split-kv-dark](#split-kv-dark)
should not be implemented as written: it assumes the chunks are genuine partials,
and if they are not, the partial write would be solving a problem the plan is
creating.


<a id="task3-dump-resolved"></a>
### RESOLVED: the chunks are genuine partials, and the spec stands — 2026-10-06

The dump settles it, and against my hypothesis. Task 3's items on fixture 2 carry
true slices, not full-row copies:

```
cluster=3  i=0 q_start=768  kv_start=0   kv_end=384 span=384 kv_len=439 group_id=29  partial_indptr=1592
cluster=3  i=1 q_start=3072 kv_start=384 kv_end=439 span=55  kv_len=439 group_id=101 partial_indptr=6200
cluster=5  i=0 q_start=0    kv_start=0   kv_end=236 span=236 kv_len=236 group_id=-1  partial_indptr=56
```

A clean two-way split of the `kv_len=439` row (384 + 55 = 439), each chunk with
its own `group_id` and `partial_indptr`, while the `kv_len=236` row carries
`group_id=-1` and is not split. So the redundant-work hypothesis is **refuted**:
the wgmma role is not duplicating attention, and there is no second hidden cause
of the serving deficit hiding here. The Q completion remains the one measured
cause of the 3x.

**And the contradiction dissolves without any benign-overwrite mystery.** The
fixture-2 pass I had been treating as evidence reported its rows as
`row 1 (task0 qo=439 kv=439)` -- that row was on an **FA2 class** at the time,
which reduces correctly, which is also what `partial_o_nnz=56` was. It is routed
to task 3 now (packed qo = 439 * 7 = 3073, above the 128 threshold), so the
hazard is live *now* and was not live *then*. Nothing about that old pass needs
explaining, and my note that it was "suspect" is withdrawn -- it was a correct
result for a different routing.

**Consequences, in order:**

1. The reduction spec in [#split-kv-dark](#split-kv-dark) stands exactly as
   written. The chunks are genuine partials, so `store_partial` plus the
   `persistent.cuh` handshake is the right shape, and the earlier worry that it
   would be "solving a problem the plan is creating" does not apply.
2. The guard is correct and keyed correctly, and its count (200 items) is the
   real work the reduction must merge.
3. The `_noinc` Q completion is still unvalidated, because the guard legitimately
   blocks fixture 2. Validating it needs either a fixture whose task-3 rows do
   not split (raise `FLASHINFER_FA3_MIN_PACKED_QO` above 3073 so the 439 row goes
   to FA2, reproducing the routing of the original fixture-2 pass) or the
   reduction itself. The former is one environment variable and is the cheap path
   to re-measuring the 66% regression.


<a id="wgmma-role-unreachable"></a>
### The unified split-KV reduction is unwired on the wgmma role, and that blocks every fused measurement — 2026-10-06

**Framing corrected.** Earlier revisions said "FA3 cannot run split rows" and "the
wgmma role cannot run real work". Both are wrong. FA3 processes a KV chunk
perfectly well -- `UnifiedWgmmaTileScheduler` already hands it `kv_start`/`kv_end`
and the mainloop honours them. The gap is narrower: **the partial-O/LSE write and
the merge trigger were never wired into the FA3 epilogue.**

The dynamic `kv_len_limit` split is also *correct and wanted*. It is the FA2 LPT
balancer doing its job on long sequences, which is the load balancing the whole
argument rests on. Looking for configurations that avoid splitting was a mistake --
a way of dodging missing wiring rather than writing it.

**One reduction, not two.** The single plan already emits the group tables for
every task (`group_id`, `group_counter`, `group_nchunks`, `group_row_begin`,
`group_nrows`, `group_kv_head`, `merge_indptr`, `merge_o_indices`) and
`MergeGroupJIT` is already generic over `HEAD_DIM`. So the FA3 epilogue must write
into the *same* partial buffers with the same `partial_indptr` and trigger the
*same* handshake and merge -- a unified reduction shared by both roles, with
cluster/DSMEM transfer of partials as a later optimisation on top, not a parallel
FA3-specific implementation.

Until that wiring exists the guard refuses split task-3 plans, which is why Two attempts to find a configuration without splitting
both failed with the *same* count:

```
fused plan: 200 task-3 (wgmma) work items carry a JIT-reduction group id
```

at `SPLITS=8 FA3_PLAN_CTAS=8` and at `SPLITS=1 FA3_PLAN_CTAS=1`. The count is
identical, so task 3's `kv_len_limit` does not derive from its own cluster count and
cannot be raised by shrinking the FA3 pool. There is no routing threshold that
helps either: any threshold low enough to send the short row to FA3 also sends the
long one.

**Consequences, and they are large.**

1. The 3.02x fused-vs-unfused ratio, the 1.38x CTA/SM cost, and the register
   starvation finding are all real, but they are **costs measured with the benefit
   switched off**. They are not a verdict on fusion.
2. Whether fusion wins is **not currently measurable**. The break-even arithmetic
   below says it needs f > 0.40-0.48 of attention work on the FA3 path and the real
   serving batch has f ~ 0.9, but that rests on the catalogue's 1.7-2.4x class
   boundary numbers, not on a fused run.
3. The unified reduction's wiring is therefore the single gate on every remaining
   question. It is also the only thing standing between the existing plan and a
   measurable fused kernel: the balancer, the group tables and the merge are all
   already there.
4. **The FA2 register/occupancy work is probably moot.** The class boundary
   measured that tiles 128 and 64 belong on wgmma and only tile-16 decode stays on
   FA2 (decode is bandwidth-bound). Every fused timing here forced tile-128/64 work
   onto FA2 via `FLASHINFER_FA3_MIN_PACKED_QO=4096`, i.e. exactly the routing the
   design rejects. So the 3.02x, the 1.38x and the register starvation are all
   measured on a workload FA2 is not meant to carry, and `setmaxnreg` or a softmax
   offload would be optimising a configuration that should not occur.

**Break-even, for when the reduction lands.** With `W` the attention work in
FA2-native SM-time, `f` the FA3-eligible fraction, `s` FA3's per-tile speedup, and
1.38 the one-CTA/SM penalty: unfused costs `W`, fused costs
`f*W/s + 1.38*(1-f)*W`, so fusion wins when `f > 0.38 / (1.38 - 1/s)`. At
s = 1.7-2.4 that is f > 0.40-0.48. At f = 0.9, s = 2.0 the attention-side gain is
~1.7x, which against attention at 31.76% of a step is ~15% end to end -- positive,
and short of the 30% bar, which is why the register fix (3x -> 1.38x) matters.

<a id="why-fused-is-slow"></a>
### Fixture 2, three arms: the upstream port runs but writes nothing — 2026-10-06

sglang `9ad9dd45d`, flashinfer `960b65d` (arms taken at the commit before the combine pass).
Command, per arm, each in its own `TORCH_EXTENSIONS_DIR` so the `-D` flags take
effect:

    FA3_UPSTREAM={0,1} DRAIN_NOINLINE={0,1} TIME_ITERS=50 \
      FIXTURE='qo:72,439,1 prefix:164,0,5468' UNIFIED=1 SPLITS=auto \
      FLASHINFER_SM_PARTITION=1 FLASHINFER_FA3_SM_SPLIT_CLUSTERS=124 \
      python tests/test_fused_server_batch.py

| arm | kernel | numerics |
|---|---|---|
| `def` flashinfer FA3, inlined | **156.8 us** | PASS, 0/512 unwritten, rel err 4.1e-3 / 4.0e-3 / 5.8e-4 |
| `def` again, with `TIME_BASELINE=1` | 155.9 us, **baseline 72.3 us** | PASS |
| `up` upstream FA3, inlined | 315.4 us | **511/512 unwritten** |
| `upni` upstream FA3, FA2 outlined | 434.3 us | **511/512 unwritten** |

**The two upstream timings mean nothing yet** and are recorded only so they are not
re-measured: with `max_splits_fa3=2` the port's epilogue takes upstream's
`is_split` branch and writes partials instead of final O, so those kernels skip the
output path entirely. The one correct row in both is the FA2 decode row (task 2);
rows 0 and 1 are both wgmma rows under default routing, so the failure is exactly
co-extensive with the wgmma role. This is the predicted consequence of upstream
reducing split partials in a separate kernel that the port did not launch.

**The parity gap on fixture 2 is 2.16x** (155.9 us fused against 72.3 us for
FA2-on-everything), both numerics-correct, both from the same build and the same
inputs. That is the number the goal is measured against, and it is larger than the
1.83x recorded on the long-prefix fixture — so the gap is workload-dependent and the
smaller fixture is the harder case. Reproduce with `TIME_BASELINE=1` added to the
`def` arm above.

What the arms do establish:

- The three fixes landed: the port no longer faults. The previous run died with an
  illegal access from feeding upstream FA3's vector-sparse offsets as page ids, a
  null `lse_ptr`, and `num_splits` pinned to 1 under partials sized for 2.
- `max_splits_fa3=2` on this fixture, so the split path is live here and cannot be
  sidestepped.
- **156.8 us is fixture 2, not the 772 us configuration.** The 772 belongs to the
  16 823-token-KV long-prefix fixture; fixture 2's longest row is kv=5469. No
  regression — two different workloads.
- `upni` is *slower* than `up` (434 vs 315) even though outlining cuts FA3's spill
  to 64 B, which is consistent with the register table: outlining moves 17.5 KB of
  spill onto the FA2 role, and this fixture has real FA2 work. Both numbers are
  still from invalid runs, so this is a hint, not a result.

### What parity means on a SPLIT batch — 2026-10-06

flash-attention's own split-KV path is **two kernel launches**: the attention
kernel writes float partials, and `flash_fwd_combine_kernel.h` reduces them in a
separate pass (`run_mha_fwd_combine`). Our port inherits that structure, since it
instantiates upstream's mainloop and epilogue with `Split=true` -- and it does
**not** launch the combine kernel, which is why `max_splits_fa3 > 1` currently means
the wgmma role's partials are never reduced.

This resolves what looked like a conflict between tasks 1 and 2. Parity with
flash-attention on a split batch does not require the in-kernel JIT reduction: it
requires *a* reduction, and upstream's own answer is a second launch. So the
sequence is

1. validate the port where the plan does not split (`max_splits_fa3 == 1`), which
   isolates the mainloop and epilogue;
2. make split batches correct, either by launching upstream's combine kernel (the
   faithful baseline, and the fair comparison against flash-attention, which pays
   the same second launch) or by our JIT reduction;
3. the JIT reduction is then an **improvement over** the parity reference, not a
   precondition for it -- it removes the second launch and the global round-trip
   that upstream pays, which is the actual claim worth making.

Recording this because it changes what the parity number has to be measured
against: a two-launch flash-attention baseline, not a single-kernel one.

### The combine pass must be restricted to the wgmma queue's requests — 2026-10-06

Two fixes landed and the third fault is now isolated.

**Fixed, and confirmed by the result:** upstream indexes `cu_seqlens_q[bidb]` and
`seqused_k[bidb]` as *request* boundaries, while the plan's `q_indptr` and `kv_len` are
pushed once per WORK ITEM (`scheduler.cuh:1582`). Passing the per-item arrays made
upstream read arbitrary work items as request offsets. With true per-request arrays,
fixture 2 goes from **439/512 unwritten to 0/512** -- every row is now written.

**Remaining fault, precisely located.** All three rows are now wrong, and the telling
one is row 2:

| row | before the descriptor fix | after |
|---|---|---|
| 0, wgmma, qo=72 | nan (unwritten) | 11.13 |
| 1, wgmma, qo=439 | nan (unwritten) | 11.42 |
| 2, **FA2 decode**, qo=1 | **0.00058 ok** | **0.155 FAIL** |

Row 2 had been correct in every previous arm, because it is an FA2-role row the wgmma
executor never touches. It is wrong now, so something in the upstream path writes
outside the wgmma queue's rows -- and that something is the combine pass. It is launched
with `b = num_requests` and walks every request, because in upstream *all* rows go
through FA3 and every request therefore has partials. In our split only task-3 rows do,
so for request 2 the combine reduces partial buffers that were never written and stores
the result over the FA2 executor's correct output.

**Confirmed fixed, 2026-10-06.** With the combine restricted to the wgmma queue's
requests, row 2 returns to **0.00072 ok** from 0.155 and 0/512 unwritten holds. All three
row-ownership faults are closed.

**What remains is the wgmma computation itself**, not coverage: rows 0 and 1 are wrong at
10.88 and 11.06. Both are FA3-role rows; the FA2 row is now correct, so the failure is
inside the upstream path's own result rather than in who writes where.

The next experiment isolates it in one run: **force the plan not to split** (raise the
chunk length so `max_splits_fa3 == 1`, or use the existing
`FLASHINFER_FA3_NO_SPLIT_REDUCE` bisect switch). At one split the epilogue writes final O
directly and the combine never runs, so

- if the wgmma rows are then **correct**, the fault is in the split path — the leading
  candidate being that upstream derives each split's KV slice internally by dividing
  `seqused_k` evenly by `num_splits`, while our planner chunks by a fixed
  `len_kv_chunk` with a short last chunk, so the slice a work item computes need not be
  the slice the plan assigned it;
- if they are still **wrong**, the fault is in the mainloop's addressing or the packed-Q
  path, independent of splitting.

Either way one run separates them, which is cheaper than reasoning about the layouts
further.

**Refinement before the result, recorded so the reading is honest.** My stated split-path
candidate is probably wrong. Upstream derives each slice itself from
`(seqused_k, num_splits, split_idx)` and ignores the plan's `kv_start`/`kv_end`, so an
even division by `n` with distinct ordinals `0..n-1` still covers `[0, L)` exactly, with
no gap or overlap. Our `split_idx = kv_start / len_kv_chunk` does yield those ordinals.
So a mismatched chunk boundary changes *which* KV each item processes but not the union,
and the result should still be correct -- only balanced differently than the planner
intended.

That makes the stronger candidate the **packed-Q block index**. The adapter computes
`block_idx = q_start / cta_tile_q`, and the plan's `q_start` is a **global** packed-row
offset across the whole batch, while upstream's `m_block` is a tile index *within* the
request -- it adds `cu_seqlens_q[bidb]` itself. Every request after the first would then
be offset twice. The one thing that does not fit cleanly is that row 0 is request 0 with
`cu_seqlens_q[0] == 0`, so it should escape the double offset, yet it is also wrong
(10.88). So either that reasoning is incomplete or there are two faults.

Both readings are testable from the same run: at one split, a correct result rules out
everything split-related and leaves the Q-indexing question, and an incorrect result
points straight at it.

**The fix is upstream's own mechanism.** `FlashAttnFwdCombine::Arguments` takes
`varlen_batch_idx_ptr`, which remaps the scheduler's batch index onto a real batch index
precisely so the pass can run over a SUBSET of batches. So:

1. build a compact device array of the request indices that have wgmma work -- the
   planner knows them, they are the rows it routed to `idx_qo_kv_len_vec[kFA3Task]`;
2. pass it as `varlen_batch_idx_ptr` and set `b` to its length;
3. the grid then covers only those requests and the FA2 rows are never touched.

This is the same shape as the two fixes before it: upstream carries a parameter for
exactly this case, and the bug was passing the whole-batch default instead. Worth noting
the pattern -- all three faults were "upstream has a field for the subset/per-item case
and I handed it the aggregate".

### The whole KV-range chain is verified; the -1 page padding is what is left — 2026-10-06

Read end to end, no GPU needed:

- `block.h:23` -- `n_block_max = ceil_div(seqlen_k, kBlockN)`, and with `Is_causal` and
  `Is_local` both false for this path nothing narrows it, so an unsplit tile gets the full
  KV range;
- `block.h:47-54` -- at `num_splits_actual == 1` the split arithmetic is the identity:
  `num_n_blocks_per_split = n_block_max - n_block_min`, so `n_block_min` is unchanged and
  `n_block_max` clamps to itself;
- `seqlen.h:52` -- `seqlen_k = seqused_k ? seqused_k[bidb] : ...`, which is our
  `kv_lens_per_req`, and `seqlen_q` falls back to
  `cu_seqlens_q[bidb+1] - cu_seqlens_q[bidb]`, which is our true per-request `q_indptr`.

So every link from our descriptors to the KV block range is correct. Ten candidates are now
eliminated.

**What that leaves is the dense page table's -1 padding.** For row 0, `seqlen_k = 200` and
`kBlockN = 128`, so `n_block_max = 2` and block 1 spans KV positions 128..255 -- but only
128..199 exist. The table is padded with **-1** (deliberately, since 0 is a valid page id),
so the paged gather for that block requests pages 200..255 and finds -1. Upstream masks the
out-of-range *logits* by `seqlen_k`, which keeps the softmax correct, but the gather itself
still dereferences the padded entries: address = base + (-1) * page_stride.

That is consistent with the signature in a way nothing else left is: the excess is small
(rel err 0.52-0.70, not orders of magnitude), one-directional (absmax always above the
reference), present on every wgmma row, and absent on FA2 rows -- which gather through
flashinfer's own path and never see this table.

Two checks, in order, both cheap:

1. What does upstream pad its own page table with? `flash_api.cpp` builds
   `page_table` for the paged path; if it pads with 0 or with a clamped last valid page
   rather than -1, our -1 is simply the wrong sentinel and the fix is one line in
   `ScatterRaggedPageTable`.
2. Failing that, size `max_pages_per_seq` per request instead of to the batch maximum, so
   no block extends past a request's real pages. For this fixture that is 2 pages for row 0
   against the current 5469, which would make the problem disappear and confirm it.

Note the first fix would also be the cheaper one to carry into serving, where
`max_pages_per_seq` is the batch maximum and most requests are far shorter.

**REFUTED -- upstream already guards the read.** `paged_kv.h:148`:

    int const page = (...) && (!Seqlenk_mask || row_idx < seqlen_k) ? mPageTable[page_idx] : 0;

Past `seqlen_k` it substitutes page 0 instead of reading the table, and `:227`, `:231`
and `:269` then mask those rows out of the K and V loads. The mainloop iterates from
`n_block_max - 1` downward, so the first iteration is the HIGHEST block and is the one
called with `Seqlenk_mask=true` (`mainloop_fwd_sm90_tma_gmma_ws.hpp:802`), while interior
blocks use `false` (`:866`) because they cannot exceed `seqlen_k`. Our `-1` padding is
therefore never dereferenced, and the sentinel choice is irrelevant.

Eleventh elimination, and the fifth of my own hypotheses to fall. The pattern across all
five is the same and worth stating plainly: I propose a mechanism that would produce the
observed signature, then find on reading that upstream already handles it. The faults
actually found came from the opposite direction -- comparing a value our code passes
against the declaration of the thing that consumes it (the zero page stride, the per-item
vs per-request arrays, the 1 120-byte smem overflow).

**So the next step should not be another hypothesis.** The remaining honest instrument is
differential: run upstream's FA3 standalone -- `tuS.cu` already builds it in our template
at 52 B spill -- on the SAME Q/K/V and the same single work item, and diff its output
against the port's. Everything our adapter supplies is then held against a reference that
is known to work, which localises the disagreement to a concrete field rather than to a
guess. That is more setup than any check so far, and it is the thing that will actually
close this out.

### The plan and the split encoding are both verified correct — 2026-10-06

Item dump on the failing fixture, against a prediction stated beforehand
(`FIXTURE='qo:200,439,100,1 prefix:0,0,0,5468 n_fa3:2'`, G=1, FORCE_ONE_SPLIT):

    predicted: 24 items, with_group_id=0, every (q_start, kv_head_idx, batch_idx) distinct
               row 0 packed 200 -> 2 q-tiles x 4 kv_heads = 8,  q_start {0,128}
               row 1 packed 439 -> 4 q-tiles x 4 kv_heads = 16, q_start {0,128,256,384}
    measured:  [plan] task=3 items=24 with_group_id=0
               [fa3-item] task=3 items_with_group_id=0 total_items=24

Exact match. **The plan is not the fault** -- no duplicate items, no over-count, no
stray split groups. That removes the "two CTAs process the same (q-tile, kv_head)"
candidate.

**The split_idx encoding is also verified**, by reading upstream's own producer and
consumer rather than inferring:

- `tile_scheduler.hpp:115` -- upstream's scheduler does
  `work_info.split_idx |= (num_splits_dynamic << 16)`, with the value taken from
  `num_splits_dynamic_ptr[work_info.bidb]`, and `:113` additionally asserts
  `split_idx < num_splits_dynamic`;
- `block.h:48-51` -- the mainloop decodes it (`>> 16` for the count, `& 0x0000FFFF` for the
  index) when deriving its KV block range.

So our adapter's `(nsplits << 16) | chunk` matches the contract on both sides, and the
mainloop is not reading an undecoded 65536. One difference worth noting for later: upstream
validates `split_idx < num_splits_dynamic` and our adapter does not, which is moot at one
split but would matter once splitting is re-enabled.

**State of the search.** Eliminated so far: the split path and combine pass, the partial
layouts, the per-request varlen descriptors, the page-table scatter indptr, the Q tile
index, GQA packing, the per-request base arithmetic, the plan's item set, and the split_idx
encoding. The measured signature is unchanged and specific: both wgmma rows wrong, every
element nonzero, output absmax consistently **above** the reference (0.155 vs 0.134 and
0.107 vs 0.091 on the two-row fixture), rel err 0.52-0.70. FA2 rows on the same batch are
exact to 3e-3.

What has not been examined is `block.h`'s own `n_block_min/max` arithmetic given
`seqused_k[bidb]`, `num_splits_actual = 1` and `split_idx = 0` -- i.e. whether the KV block
range it derives for an unsplit tile really is `[0, ceil_div(seqlen_k, kBlockN))` for our
descriptor values. That is the first thing a reader should check, and it is a code read
rather than a run.

### Within-request iteration, and the output is too LARGE — 2026-10-06

Discriminator: two wgmma rows at `bidb` 0 and 1 in one batch, at G=1, with a task-0 and a
task-2 row present so the harness's `assert 0 in task[N_FA3_ROWS:]` guard is satisfied.
`FIXTURE='qo:200,439,100,1 prefix:0,0,0,5468 n_fa3:2'`:

| row | bidb | rel err | absmax | ref absmax |
|---|---|---|---|---|
| 0 wgmma qo=200 | 0 | **0.699 FAIL** | 0.155 | 0.134 |
| 1 wgmma qo=439 | 1 | **0.563 FAIL** | 0.107 | 0.091 |
| 2 FA2 task 0 qo=100 | 2 | 0.00331 ok | 0.195 | 0.196 |
| 3 FA2 task 2 qo=1 | 3 | 0.00050 ok | 0.023 | 0.023 |

**Per-request base arithmetic is ruled out:** `bidb=0` has a zero base and fails anyway. So
the fault is in within-request tile iteration and hits every wgmma item alike.

**And the direction of the error is informative.** The output absmax exceeds the reference
in both wgmma rows (0.155 vs 0.134, 0.107 vs 0.091) while rel err is now *below* 1
(0.56-0.70, against ~1.0 on the earlier fixtures). Too-large output with partial
correlation is **over-accumulation** -- attending over more KV than the row owns, or
counting tiles twice -- not the permutation I inferred from the earlier dump. The
permutation reading came from a fixture where the magnitudes happened to be comparable;
with two wgmma rows side by side the excess is consistent and in one direction.

That reorients the search to how many KV tiles each item iterates:

- the per-item `nsplits` encoded in `split_idx`'s high bits is `ceil_div(kv_len, chunk_len)`
  from the **adapter**, while `num_splits_dynamic` for the combine comes from the **planner**
  path; under `FORCE_ONE_SPLIT` both should be 1, but the mainloop derives its KV range from
  `(seqused_k, num_splits, split_idx)` and any disagreement there changes the range, not the
  destination;
- `work_indptr` ranges: if a CTA's `[begin, end)` overlaps another's, two CTAs process the
  same (q-tile, kv_head) and both write final O -- the second does not overwrite cleanly
  since FA3's epilogue accumulates through smem, which would show as exactly this excess;
- the 4 kv_heads x 4 q-tiles item count per row: if `kv_head_idx` repeats a head across
  items that should differ, that head's contribution lands twice.

The cheap instrument for all three is the existing `FLASHINFER_FA3_DUMP_ITEMS` per-item dump
plus `FLASHINFER_PLAN_ITEMS` counts: print task 3's items for this fixture and check that
the (q_start, kv_head_idx, batch_idx) triples are unique and that their count is
`sum_over_rows(ceil_div(packed_qo, 128) * num_kv_heads)` = (2 + 4) * 4 = 24. A duplicate or
an overlap is visible directly, with no rebuild.

### G=1 rules out GQA packing as the port's fault — 2026-10-06

`HEADS=4,4` makes packed and unpacked row spaces identical. Run with
`FA3_UPSTREAM=1 FLASHINFER_FA3_FORCE_ONE_SPLIT=1 HEADS=4,4 DUMP_ROWS=1` on fixture 2:

| row | qo | routed to at G=1 | rel err |
|---|---|---|---|
| 0 | 72 | **FA2 tile 128** (packed 72 < 128 threshold) | 0.00335 ok |
| 1 | 439 | **wgmma** | **1.112 FAIL** |
| 2 | 1 | FA2 tile 16 | 0.00058 ok |

Read with care: the harness's "FA3" label on row 0 is static, not the routing. At G=1
row 0 falls below the 128-packed-row wgmma threshold and runs on FA2, so its pass says
nothing about upstream. The only wgmma row is row 1, and it fails with the same
signature as at G=7 -- full magnitude (absmax 0.132 vs 0.091), every element nonzero,
values uncorrelated.

**So the wgmma path is wrong with no packing at all, and packing is ruled out.** The fault
is in the adapter's work-item semantics. `PackGQA=false` was deliberately not used for
this test: it would also have made `bidh` index qo heads and `m_block` index unpacked
tokens against the plan's kv-head / packed values, failing for a new reason.

Next split, running: the same 439-row wgmma request placed first (`bidb 0`), second
(`bidb 1`), and alone -- per-request offset fault versus within-request fault.

### The split path is exonerated; the wgmma result is still wrong — 2026-10-06

Two runs, both with `FLASHINFER_FA3_FORCE_ONE_SPLIT=1` so the plan reports
`max_splits_fa3 == 1`, the epilogue writes final O directly (`epilogue_fwd.hpp:232`) and
the combine pass never launches:

| | row 0 (wgmma) | row 1 (wgmma) | row 2 (FA2) | baseline |
|---|---|---|---|---|
| before the page-table fix | 13.21 | 13.23 | 0.00052 ok | 72.8 us |
| with per-request page offsets | **10.58** | **10.53** | 0.00054 ok | 73.0 us |

**What this establishes.** The entire split path is off the critical path for this fault:
the combine pass, the partial O/LSE layouts, the per-item `num_splits` encoding and the
split-count arrays are all inactive at one split, and the rows are still wrong. One run
removed four candidate areas.

**What I got wrong.** I called the page-table scatter the root cause before this
measurement returned. It was a genuine bug — the scatter built a
`[num_requests, max_pages]` table from the plan's per-work-item `kv_indptr`, so each
request's page range came from an arbitrary item — and fixing it moved the error
(13.2 -> 10.5) without fixing it. "The only remaining explanation I can see" is not a
diagnosis.

**Leading candidate, and why it is incomplete.** The packed-Q block index. The adapter
computes `block_idx = q_start / cta_tile_q`, and the plan's `q_start` is a **global**
packed-row offset, while upstream's `m_block` is a tile index *within* the request since
it adds `cu_seqlens_q[bidb]` itself. For this fixture request 1 starts at packed offset
504, which is not a multiple of 128, so its tiles are both offset and **misaligned** to
the tile grid. The fix is to subtract the request's packed base:
`(q_start[idx] - q_indptr_per_req[batch_idx[idx]] * G) / cta_tile_q`.

That cannot explain row 0, whose request has offset 0 and should already be correct, so a
second fault is expected — and the ~10x error on *both* wgmma rows with the FA2 row exact
to 5e-4 is consistent with something affecting the whole wgmma path, not only the
later requests.

**Tried, and REFUTED — the premise was false.** Subtracting the request's packed base
faulted with an illegal memory access. `q_start` is **already request-local**: the planner
pushes `qo_tile_idx * cluster_tile_q` (`scheduler.cuh:1655`), the tile's offset within its
request, so `q_start / cta_tile_q` was already the `m_block` upstream wants and the
subtraction drove it negative for every request after the first. Reverted.

Two things worth keeping from that. The adapter's own comment already said "PACKED row
offset -> block_idx = q_start / 128", and one grep at the push site would have settled it
before a 15-minute build — the four earlier faults were *all* fixed by reading the push
site, and here I reasoned from upstream's convention instead. And the packed-Q tile index
is now exonerated along with everything else.

**Candidate set after five eliminations.** Ruled out: the split path (combine pass,
partial O/LSE layouts, per-item `num_splits`, split-count arrays), the per-request varlen
descriptors, the page-table scatter's indptr, and the Q tile index. What remains is the
Q and KV *descriptions* rather than their indexing:

- `shape_Q` / `stride_Q` as handed to upstream's PackGQA view — the nested `((G, seq), D)`
  layout is derived from these, and a wrong head stride would corrupt every wgmma row
  equally while leaving FA2 untouched, which matches the symptom shape;
- the KV descriptors at `page_size == 1`: `shape_K {page_size, d, h_k, num_pages}` with
  `kv_stride_page` derived as `k_stride_n * page_size`, `kv_stride_tok`, `kv_stride_head`.
  A rank-3 `[tokens, H, D]` cache has caught this code before (recorded in the ext's
  `kv_strides` comment, where reading stride(1)/stride(2) aliased all four KV heads).

Both are checkable without a GPU by printing the strides the dispatch assembles against
the tensor's actual strides, which is the next step and is cheaper than another arm.

**ROOT CAUSE FOUND, without a GPU — the KV page stride was 0.** `shape_K` is
`{page_size, head_dim, num_kv_heads, num_pages}`, so the fourth element of `stride_K` is
the stride over **pages**, and the dispatch passed `0`:

    {p.page_size, p.head_dim, p.num_kv_heads, p.num_pages},   // shape_K
    {p.kv_stride_tok, _1{}, p.kv_stride_head, 0},             // stride_K  <-- page = 0

Upstream indexes `K[page_token, d, kv_head, page]`, so a zero page stride makes every
lookup read the first page. Same tuple for V. `kv_stride_page` was computed in the
dispatch as `k_stride_n * page_size` and then **never referenced**.

This accounts for all the evidence rather than part of it, which is what distinguishes it
from the two refuted guesses: a single aliased KV source corrupts the whole wgmma path
**uniformly** and touches nothing the FA2 executor reads — matching every wgmma row wrong
by ~10x and equally, the FA2 decode row exact to 5e-4, at one split, with the split path,
the combine, the partial layouts, the varlen descriptors, the page-table scatter and the Q
tile index each independently eliminated.

Two process points, since both refuted guesses shared one cause:

- Both were reasoning from upstream's *conventions* instead of reading what our code
  passes. One grep of the field settles each in seconds; the `m_block` guess cost a
  15-minute build and an illegal-access fault, and the plan's own comment already said
  `q_start` was request-local.
- A positional tuple hid this for five rounds. `{kv_stride_tok, _1{}, kv_stride_head, 0}`
  reads as complete, and the unused `kv_stride_page` field was visible evidence the whole
  time. **Grep every field of a POD for use before assuming the tuple that consumes it is
  right.**

**Measured: a real 10x improvement, and a different failure now.** With the page stride
wired, at one split:

| | row 0 | row 1 | row 2 (FA2) |
|---|---|---|---|
| page stride 0 | 10.58 | 10.53 | 0.00054 ok |
| page stride wired | **1.022** | **1.033** | 0.00056 ok |

So the aliasing was genuine and is fixed. But ~1.0 is a *characteristic* value, not a
near-miss: `|0 - ref| / |ref| == 1`, so the wgmma output is now approximately **zero or
uncorrelated** rather than wrongly-computed. The path went from reading one page for
everything to producing almost nothing.

That reframes the remaining candidates toward "the mainloop accumulates no KV" rather
than "it accumulates the wrong KV":

- `seqused_k` per request — if a request's KV length arrives as 0, the mainloop has no
  tiles to iterate and the epilogue stores zeros. `kv_lens_per_req` is supplied by the
  caller in TOKENS; worth asserting it is non-zero on device rather than trusting it.
- the **dense page table's -1 padding**. It is padded with -1 deliberately (0 is a valid
  page id), but upstream must then never index a padded slot. If `max_pages_per_seq` is
  larger than a request's real page count and upstream iterates to the shape's extent
  rather than to `seqused_k`, it reads -1 and may mask everything off.
- `page_size == 1` interaction: `shape_K`'s first dimension is `page_size`, so with 1
  token per page the within-page token axis has extent 1 and all iteration happens on
  the page axis. Any off-by-one in `num_pages` vs the table's extent shows up as
  no-contribution rather than wrong-contribution.

**Dump run, and it refutes the list above.** `DUMP_ROWS=1` against the prebuilt
extension (harness-only change, so no recompile):

    row 1  got[0,:4] = [-0.054443, -0.017700,  0.030029, -0.009705]
           ref[0,:4] = [-0.054236, -0.001193,  0.010781, -0.009634]
           nonzero = 1573376/1573376   absmax 0.0986 vs ref 0.1055

    row 0  got[0,:4] = [ 0.061279, -0.031738,  0.026978,  0.019287]
           ref[0,:4] = [-0.026557, -0.015640, -0.014690, -0.011900]
           nonzero =  258048/258048    absmax 0.1416 vs ref 0.1214

Every element is nonzero, the magnitudes are right, and in row 1 elements **0 and 3 match
to bf16 precision** while 1 and 2 do not. So the mainloop is not failing to accumulate --
`seqused_k == 0`, the `-1` padding and the `num_pages` off-by-one are all out. It computes
correct-magnitude values that land in the **wrong places**.

That is the signature of a **permutation**, and the obvious suspect is the packed-GQA
output mapping. The wgmma role writes packed `((G, seq), D)` rows that must be unpacked to
`(seq, H, D)`; if our `packed_coord(p, G) = (p % G, p / G)` convention disagrees with
upstream's, each value is computed correctly and stored at the wrong `(token, qo_head)`.
With `G = 7` that scatters most elements while leaving a minority coincidentally in place,
which is exactly the pattern above -- and it explains why rel err sits near 1.0 without
the output being small, which no "missing KV" story does.

**Next, in order, both cheap:**

1. Confirm the permutation: compare *sorted* |values| of `got` and `ref` for one row. A
   close match proves the value set is right and only the placement is wrong, which would
   settle it without touching the kernel.
2. Then diff our packed-row convention against upstream's `PackGQA` unpacking --
   `pack_gqa.h` against `packed_coord` / `FLASHINFER_FUSED_PACKED_Q` -- since those are the
   two definitions that have to agree.

**Step 2 done, and it REFUTES the packing hypothesis.** `pack_gqa.h:69` reads
`m_idx = qhead_per_khead_divmod.divmod(h_idx, idx)`, and CUTLASS's signature
(`fast_math.h:420`) is `int divmod(int &remainder, int dividend)` -- the reference
parameter receives the **remainder** and the function **returns the quotient**. So
upstream computes `h_idx = idx % G` and `m_idx = idx / G`, which is exactly our
`packed_coord(p, G) = (p % G, p / G)`. The two conventions **agree**, so the packed-row
mapping is not the fault.

That is the third hypothesis of mine refuted by checking rather than by running
(page-table scatter as root cause, per-request `m_block`, now packed-row mapping), and all
three were refuted cheaply. The pattern is consistent: I keep proposing the mechanism that
would explain the symptom and not verifying the premise first. The 1 120-byte smem
overflow, the per-item vs per-request arrays and the zero page stride were all found the
other way -- by reading what the code passes and comparing it to what the consumer
declares.

**So the permutation is real but not in the row mapping.** Still true from the dump: full
magnitude, all elements nonzero, a minority matching to bf16 precision. Remaining places a
value can land wrong while keeping its magnitude:

- `stride_O` / `shape_O` on the **output** side, which the epilogue uses to scatter packed
  rows back. The 5-element `ShapeO` is `(seqlen_q, d, head, batch, num_splits)` and we pass
  `{p.total_q, p.head_dim, p.num_qo_heads, 1, p.num_splits}` with
  `{p.o_stride_tok, _1{}, p.o_stride_head, 0, 0}` -- the same positional-tuple shape that
  hid the zero page stride for five rounds, and worth checking element by element against
  the declaration.
- the **Q** side mirror of it, `shape_Q` / `stride_Q`, where a wrong head stride would feed
  each packed row the wrong head's Q and produce exactly this signature.
- `kv_head_idx` as `bidh`: under PackGQA upstream's `bidh` indexes KV heads, and the plan's
  `kv_head_idx` is a KV head index, so these should agree -- but that is an assumption of
  the same kind as the three already refuted, and the check is one grep at the push site.

**All three checked, all three correct.** Against the declarations in
`epilogue_fwd.hpp:82-84` and upstream's launch template:

| passed | declared | |
|---|---|---|
| `{total_q, head_dim, num_qo_heads, 1, num_splits}` | `(seqlen_q, d, head, batch, num_splits)` | ok |
| `{o_stride_tok, _1, o_stride_head, 0, 0}` | `(int64, _1, int64, int64, int64)` | ok (batch extent 1) |
| `{total_q, head_dim, num_qo_heads, 1}` / `{q_stride_tok, _1, q_stride_head, 0}` | `{total_q, d, h, 1}` | ok |
| `bidh <- kv_head_idx` | epilogue indexes `shape_O_packed`'s head axis by KV head | ok |

So the descriptor tuples are not the fault either, and the cheap inspection candidates are
exhausted. Six have now been eliminated by reading and three by measurement.

**The decisive experiment left is one template flag: turn PackGQA OFF.**
`UpstreamFA3Mainloop` / `UpstreamFA3Epilogue` are instantiated with `PackGQA=true`
(`fa3_upstream.cuh:50`). Setting it false makes upstream treat each qo head as its own
row-space, which the plan can feed by routing one work item per (row, qo_head) instead of
per (row, kv_head) -- the arrangement `PrefillSM90Plan` already used before the unified
plan replaced it.

That is worth doing before anything subtler because it splits the remaining space cleanly:

- **rows correct with PackGQA off** -> the fault is in the packing interaction, and the
  search narrows to how our packed row space is fed to upstream rather than to anything in
  the mainloop. Note this would also mean the port loses packed GQA, which was one of the
  two reasons to expect it to *beat* flash-attention, so it is a diagnosis rather than a
  fix.
- **rows still wrong** -> the fault is independent of packing and lies in the work-item
  semantics the adapter supplies (`q_start`, `kv_head_idx`, `batch_idx`, `work_indptr`
  ranges), which is where a direct comparison against upstream's own scheduler on the same
  batch becomes the cheapest instrument.

The dump also stands as evidence for whichever branch: full magnitude, every element
nonzero, a minority matching to bf16 precision. Any explanation has to produce that, which
is what ruled out the three "accumulates no KV" candidates.

### Co-residency constrains the fused kernel twice: registers AND shared memory — 2026-10-06

The Q-split arm never reached the kernel. It failed at
`cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize)` with `invalid argument`, and the
budget says why exactly:

| tile-16 runner | SharedStorage |
|---|---|
| 1 x 4 warps (today) | 69 632 B |
| 1 x 8 warps (Q-split) | **135 168 B** |
| opt-in limit | 232 448 B |
| FA3's SharedStorage + the 48 B cursor region | 98 400 B |
| therefore available to the FA2 slot | **134 048 B** |

135 168 against 134 048: over by **1 120 bytes**. Tile 128 is not the problem -- at 8
warps it is the same 49 152 B as at 4 -- it is tile 16, whose 8 KV warps double its K/V
staging.

So the fused kernel is constrained twice over by co-residency, and neither constraint
comes from either role's own needs:

- **registers**: 384 threads at 1 CTA/SM caps the allocation at 170/thread, and the two
  roles' live ranges share it;
- **shared memory**: FA3's 98 352 B leaves 134 048 B, which the FA2 role's natural
  configuration exceeds.

A standalone FA2 kernel has the whole 232 448 B, so 135 168 fits with 95 KB spare, and
it gets 255 registers at 128 threads x 2 CTAs. Both constraints are artifacts of
sharing a `__global__`, not costs of the work.

**The two executors already exist as measured probes.** The parity probes compiled
earlier are the two kernels:

| probe | geometry | regs | spill stores |
|---|---|---|---|
| `standalone_upstream` (flash-attention's FA3) | 384 thr, 1 CTA/SM | 168 | **52 B** |
| `fa2_only_128x2` (persistent LPT) | 128 thr, 2 CTA/SM | **255** | **0 B** |

Both at native quality. What is missing is the plan wiring and the PDL launch, not the
kernels.

### One plan, one or two executors — what the kernel count actually decides — 2026-10-06

Settled as a design position after the register measurements, because it changes what
the remaining work has to prove.

**The contribution is the plan, not the kernel count.** Four things carry the story
and all of them live in the plan's data structures, which both executors read:

- 4-mode tile-class assignment (prefill chunk / decode / draft / verify) with
  L2-swizzled LPT, in `HolisticPlanInfo<4>`;
- work-proportional SM allocation, already exported as `num_fa3_clusters`;
- the unified JIT split-KV reduction, keyed on the plan's group tables
  (`group_nchunks` / `group_row_begin` / `group_nrows` / `group_counter`);
- the fused tree-build epilogue, which belongs to whichever executor writes the verify
  output.

None of these requires the two roles to be in one `__global__`. So "one plan, two
executors" keeps the whole story; the kernel count decides only two things:

1. **cross-role backfill inside a CTA** — an FA3 CTA draining FA2 work. This is also
   exactly what forces both mainloops into one function and causes the measured
   33 676 B co-compilation spill, so it is the thing being traded for register parity.
   PDL recovers part of it temporally: the FA2 executor starts on SMs the FA3
   executor vacates.
2. **cross-role smem multicast** — a gathered KV tile shared between roles cannot
   cross a kernel boundary. Worth 1.33-1.59x *within* a role in isolation; never
   demonstrated across roles.

**The SM split must move into device memory either way.** It is currently a kernel
*argument* (`sm_split`), which a CUDA graph bakes at capture, so a per-step split would
need one graph per split value. With a fixed grid and the split read from the plan,
a CTA that is not admitted exits in a few hundred ns and the grid dims stay constant:
one graph per batch bucket, split free to change every step. This is required for the
fused kernel too and has not been paid yet -- the fused path is not currently in the
graph-capture path.

sglang's own graph infrastructure already works this way one level up
([advanced CUDA graph](https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph/)):
shape variation is absorbed by bucketing and padding, with unused request slots
"rewritten as zero-length sentinels" that cost almost nothing under the varlen
scheduler. Exit-early CTAs are the same trick a level down, so this is additive rather
than a new mechanism. Two further points from that post matter here: prefill and decode
are already **separate runners with separate capture ladders**, so two executors add no
PD-ratio graph combinations; and it covers no graph-update or device-side-launch-param
APIs at all, so a device-memory split does not conflict with anything there.

**Decided by measurement, 2026-10-06: two executors.** The Q-split was the experiment
that would have kept fusion, and it does not reach parity inside one kernel:

| FA2 configuration | isolated, no FA3 in the kernel | inside the fused kernel |
|---|---|---|
| 4 warps, `num_mma_q=2` (today) | 1 136 B spill | 3 232 B spill |
| 8 warps, `num_mma_q=1` (Q-split) | **88 B** | **1 504 B** |

Fixing the accumulator is worth keeping on its own -- it halves the fused kernel's
spill, 3 232 -> 1 504 B -- but roughly 1.4 KB survives that no FA2-side change can
touch, because the FA3 role's live ranges still share the allocation. Together with the
converse measurement (outlining the FA2 body takes FA3 to 64 B and moves 17 500 B onto
FA2), the conclusion is symmetric: **one kernel can give either role its native
allocation, never both.**

So the kernel splits, the plan does not. Both executors read one `HolisticPlanInfo<4>`.

**Correction to the loss list above: only ONE thing is actually given up.** I had
counted cross-role smem multicast as a casualty. It is not, because the measured
benefit was never cross-role:

- **DSMEM multicast** (1.587x in `tests/dsmem_kv_share.cu`) is for *q-tiles of the same
  prefill row re-gathering the same KV*. Every one of those tiles is in the FA3 role, so
  the sharing is intra-role and a cluster inside one executor still provides it.
- **The smem-handoff JIT reduction** is also intra-role: a split row's chunks are all
  items of the same task, so a partial never has to travel between executors.
- **The spec tree-build epilogue** belongs to whichever executor writes the verify
  output, so it does not straddle anything.

What remains given up is **cross-role backfill inside a CTA** -- an FA3 CTA draining FA2
work -- and that is precisely the construct that forces both mainloops into one function
and costs the 33 676 B. PDL recovers part of it temporally, with the FA2 executor
starting on SMs the FA3 executor vacates.

So the fusion worth keeping is the fusion that was never about sharing a `__global__`:
the unified 4-mode scheduler, the smem-handoff JIT reduction, and the spec epilogue.
Register parity per role then comes for free, which is what makes the scheduling work
downstream (EDF, perf model, PD colocation) rest on a kernel advantage rather than on a
kernel handicap.

### FA2's spill is the O accumulator, and splitting Q removes it — 2026-10-06

The register table showed FA2 losing 87 registers and spilling 1 136 B at the fused
kernel's 384-thread geometry, with the SM-wide budget unchanged (255x256 == 170x384
== 65 280). That made it a work-partition question. Three compile-only probes answer
which partition, each a `__launch_bounds__(384, 1)` kernel containing only the FA2
runners:

| FA2 configuration | threads | regs | spill stores | stack |
|---|---|---|---|---|
| `num_warps_q=4, num_warps_kv=1, num_mma_q=2` (today, `WG_DRAIN=1`) | 128 of 384 | 168 | 1 136 B | 552 B |
| `num_warps_q=4, num_warps_kv=3, num_mma_q=2` (KV split, all 384) | 384 | 168 | **1 276 B** | 648 B |
| `num_warps_q=8, num_warps_kv=1, num_mma_q=1` (Q split) | 256 of 384 | 168 | **88 B** | 88 B |
| native, for reference | 128 x 2 CTA | 255 | 0 B | 0 B |

**Splitting KV does not help — it is slightly worse.** Using all 384 threads by
raising `num_warps_kv` to 3 costs *more* spill than leaving 256 threads idle. Each
KV-warp still accumulates a full O tile over its q rows, so per-thread registers do
not fall, and the cross-warp reduction adds work. This is the direct answer to
"offload some softmax work to the remaining threads": the softmax is not what
occupies the registers.

**Splitting Q removes almost all of it.** Eight warps over Q with `num_mma_q=1` still
covers 128 q rows (8 x 1 x 16), but each warp owns 16 rows instead of 32, halving the
per-thread O accumulator: spill drops 1 136 -> 88 B, 12.9x, and the stack frame
552 -> 88 B. Registers stay at 168 because that is the launch bound, but the kernel
no longer needs more than it has.

So the accumulator is the binding constraint, and it scales with
`CTA_TILE_Q / num_warps_q` and the head dim -- not with `num_warps_kv`. 12 warps
cannot be reached by splitting Q (`12 * num_mma_q * 16 = 128` has no integer
solution), which is why the winning configuration uses 256 of the 384 threads.

**Tried in the full kernel, and it is invalid as wired — the reason is structural.**
`FA2_QSPLIT=1` on fixture 2 gave rel err **8.94 on row 0, the FA3 row**, with both
FA2 rows correct (4.1e-3, 4.8e-4). FA2 rows right and the FA3 row wrong points at the
role boundary, not at the Q-split arithmetic, and there is a concrete cause: the
8-warp runner needs **256 threads**, but the kernel's FA2 gate admits 128
(`persistent_fa3_fused.cuh:409`, `if (threadIdx.x >= 128u) return;`), and the backfill
path admits only warpgroup 1 (`:405`). Half the threads the mainloop's barriers expect
never arrive, and its named barriers then overlap FA3's. The measured 158.0 us from
that arm is meaningless.

What it needs, and it is more than a gate widening: **all three runners must be the
same width**, because one gated thread region hosts whichever runner the work item's
tile class selects. At 8 warps that is

    tile 128:  8 warps_q x 1 warp_kv, num_mma_q=1   (16 q rows per warp)
    tile  64:  4 warps_q x 2 warps_kv, num_mma_q=1  (8 warps_q is impossible: 64/(8*16) = 0.5)
    tile  16:  1 warp_q  x 8 warps_kv, num_mma_q=1

all 8 warps / 256 threads, with the gate at 256 and backfill taking warpgroups 1-2
(threads 128..383) rather than warpgroup 1 alone. The smem slot sizing also has to be
re-derived, since each runner's `SharedStorage` changes with its warp counts.

### The two roles' register costs, separated — 2026-10-06

Five `-Xptxas -v` builds, every spill attributed to a named function, holding the
split policy, stage count and thread width constant. This replaces the v0.9.0
hypothesis ("register pressure is the leading candidate") with two measured and
independent causes.

| configuration | FA3 role | FA2 role |
|---|---|---|
| both roles inlined (today's default) | 33 676 B, one shared allocation | shared |
| FA2 body outlined (`FLASHINFER_FUSED_DRAIN_NOINLINE=1`) | **64 B** | **17 500 / 17 664 B** |
| upstream FA3 alone, 384 thr, same bounds | 52 B | — |
| FA2 alone, 384 thr (no FA3 in the kernel) | — | 1 136 B, 168 regs |
| FA2 alone, 128 thr x 2 CTA (native) | — | **0 B, 255 regs** |

**1. Co-compilation, and it is recoverable.** FA3's 33 676 B is not intrinsic to
upstream's mainloop: compiled alone in the *same* template, with the same
`__launch_bounds__(384, 1)` and the same 168 registers, it spills 52 B. ptxas
allocates per function, so while both bodies are inlined their live ranges share one
allocation even though the roles sit in mutually exclusive `blockIdx` branches.
Outlining the FA2 body brings FA3 to 64 B.

Two predictions of mine were refuted here. I expected the cause to be
`__launch_bounds__` capping the uniform allocation near 170 while upstream requests
232 per MMA warpgroup through `warpgroup_reg_alloc` — refuted, because the
standalone build has the identical cap and does not spill. And I expected outlining
to remove the spill — it relocates it: the outlined FA2 body then spills 17 500 B of
its own, because an ABI callee cannot use the caller's register allocation.

**2. Launch geometry, and the budget is not the limit.** FA2 at 384 threads spills
1 136 B and holds 168 registers against 0 B and 255 natively, measured with no FA3
compiled into the kernel at all. But the SM-wide budget is unchanged:

    native FA2   255 regs x 256 threads/SM = 65 280
    fused FA2    170 regs x 384 threads/SM = 65 280

So this is a partition, not a shortfall. The 1 136 B is the cost of running FA2 on
128 of the CTA's 384 threads (`WG_DRAIN=1`) while registers are allocated for all
384 — i.e. the cost of 256 idle threads, not a floor. `WG_DRAIN=3` is the
redistribution that would reclaim it, and it currently produces wrong FA2 output.
That is the same thing as "offload some softmax work to the remaining threads".

**Consequences for the parity bar.** FA3 parity is reachable, by outlining the other
role. FA2 parity is unproven and blocked on `WG_DRAIN=3` being made correct, not on
hardware. `FLASHINFER_FUSED_DRAIN_NOINLINE` is left OFF, because it is a trade whose
sign depends on the work mix — it should win where FA3 dominates (fixture 2 is ~99%
task-3 work) and lose on decode-heavy batches — and that measurement has not been
run.


### Why the fused kernel is slow: register starvation, not scheduling — 2026-10-06

**The short answer.** `__launch_bounds__(384, 1)` -- FA3's warp-specialised width
-- caps the static allocation at 65536/384 = **~170 registers per thread**. FA2's
mainloop is `mma.sync`, which is register-hungry and natively runs 128 threads at
2 CTAs/SM so every thread gets the full ~255. Starved to 168, it spills: the
source already records "168 registers with up to 4.4 KB of spill loads"
(`persistent_fa3_fused.cuh:427`) and 6284 bytes of spill stores in one
configuration (:119). That is the cost, and it is an implementation bug, not a
property of fusion.

**This supersedes the framing in earlier revisions of this file.** I had written
the deficit up as "1.38x structural x 2.19x unexplained" and implied a schedule
would have to overcome it. That was wrong twice over: load balance and tile waste
are properties of how work is *distributed* and cannot cost 3x, and the 2.19x was
never unexplained -- the spill numbers were already in the source.

**The measurements, for the record.** Fixed fixture batch, 200 iterations, CUDA
events, identical inputs, unfused reference measured in the same run:

| configuration | kernel time |
|---|---|
| unfused persistent, 2 CTAs/SM (native) | 43.8 us |
| unfused persistent, forced to 1 CTA/SM | 60.3 us |
| fused, 1 CTA/SM, FA2 role on 128 of 384 threads | 132.1 us |
| fused, aliased smem (unsafe -- see below) | 118.1 us |
| fused, FA2 role on all 384 threads (width 3) | 147.9 us |

Eliminated as causes, each by a single-variable sweep: the Q completion protocol
(0.3%), SM reservation (6% across an 8x sweep of reserved clusters), plan
construction (the ratio is identical at kernel level, so it is not host-side),
and thread width -- giving FA2 *more* threads made it **slower** (147.9 vs 132.1),
which is the register story again: more warpgroups do not raise the per-thread
ceiling, they lower it.

**Confirmed from the compiled binaries** (`cuobjdump -res-usage` on the pod):

| kernel | registers/thread | stack (spill) |
|---|---|---|
| standalone persistent FA2 (the 43.8 us baseline) | **255** | 56-288 B |
| fused kernel | **168** | **720-816 B** |

FA2's mainloop already uses the 255 ceiling and still spills a little. In the fused
kernel it gets 168 and spills about 10x as much.

**Why FA2-only SMs need 2 CTAs, and what that implies for the design.** Native FA2
has no warp specialization: it runs two symmetric 128-thread CTAs per SM, each
thread at 255 registers, with no `setmaxnreg`. That is 65,280 of the SM's 65,536
registers. FA3 prefill is one CTA of three warpgroups whose consumers hold ~240
registers each, so it occupies the whole SM and cannot share it with FA2 tiles.

In a single fused launch every CTA has the same shape, so FA2 work inherits FA3's
384-thread CTA. That is the entire cause of the slowdown: the two extra warpgroups
pull the ceiling to 168 registers, and FA2 runs at half its native occupancy.
`setmaxnreg` on the FA2 path would patch the registers but not the occupancy
(the 1.38x, 60.3 vs 43.8 us), so it is not the fix.

**The design that follows.** Keep both kernels unmodified and put the fusion in the
plan rather than the launch:

- FA2-only SMs run the stock persistent kernel: 2 x 128-thread CTAs, 255 registers.
- FA3-only SMs run stock FA3: 1 CTA x 3 warpgroups.
- One unified scheduler sizes the two SM pools and pre-assigns tiles so both pools
  finish together. This is the paper's pre-scheduling, adapted: the paper balances
  by mixing small, interchangeable tiles on one SM, which FA3 prefill cannot do, so
  the balancing unit becomes the pool size.

Implementation: two concurrent launches pinned to disjoint SM partitions with green
contexts (CUDA 12.4+), driven by one plan. Not yet built. The first measurement is
whether the concurrent launch adds enough overhead to eat the gain, on the fixture
against the 43.8 us unfused baseline.

**What remains genuinely structural** is only the 1 CTA/SM cap: FA3's ~98 KB of
smem leaves no room for a second CTA, and the unfused kernel loses 1.38x
(43.8 -> 60.3 us) when forced to one. That is the honest entry price of a single
launch. Against 43.8 us it means the fused kernel's floor is ~60 us, not ~132 --
so removing the spills should close most of the gap, and the schedule-quality
argument then has to beat 1.38x rather than 3x.

**Aliased smem is worth 11% and is NOT safe today.** 118.1 vs 132.4 us, but
disjoint smem is the fix for a race: CUTLASS initialises pipeline mbarriers before
the `sm_split` branch, so an aliased base of 0 puts the drain's slot inside FA3's
`SharedStorage` (nondeterministic rel err 1.93 / 2.53 / 3.31). It fires only when
a CTA drains FA2 before entering FA3, which the measurement configuration -- wgmma
role empty -- could not trigger. Capturing the 11% means re-initialising FA3's
pipeline state after the rendezvous.


<a id="split-row-race-handoff"></a>
## FA3 optimisation items, measured per-executor on the fixed kernel — 2026-10-07

First **valid** optimisation data in this work. Every earlier trait number was measured before
the `seqused` fix, against a `Theta(kv^2)` baseline the defect dominated, and is void -- a
constant-factor knob cannot be assessed against a quadratic. These replace them.

Method: FA3 against **FA3**, one item changed per arm, on a fixture the FA3 executor can serve
in full (`qo:2048,2048 prefix:65536,98304`, the two TraceLab append rows; the 22-row mixed
batch leaves 5632/5650 rows unwritten under FA3 and cannot produce an FA3 baseline at all).
5 iters, own extension dir per arm. **Every arm: PASS, 0 unwritten, 0 fallbacks.**

| arm | total | vs baseline |
|---|---|---|
| **`fa3_base`** (stages 2, `MmaPV_is_RS=1`, `LPT=0`, cp.async) | **11 164.0 us** | - |
| `fa3_stage3` (`UP_STAGES=3`) | 11 224.5 us | **+0.54%** slower |
| `fa3_lpt1` (`SCHED_LPT=1`) | 11 239.9 us | **+0.68%** slower |
| `fa3_rs0` (`MmaPV_is_RS=0`) | 11 609.8 us | **+3.99%** slower |
| `fa3_tma` (`FA3_PAGED_TMA=1`) | 12 284.0 us | **+10.03%** slower |

Reading them as optimisations against baseline FA3:

- **`MmaPV_is_RS=1` is worth 3.84%** -- the one real gain, and it is upstream's own value for
  this config. Keeping P in registers for the PV mma instead of round-tripping through smem.
- **cp.async beats TMA paged by 9.12%**, measured on the fixed kernel, confirming the earlier
  page-128 observation that survived the retraction. **Keep cp.async.**

  **RETRACTED, immediately and before it propagated: this does NOT price the multicast item.**
  I wrote that TMA paged is the gateway to DSMEM multicast and that multicast therefore starts
  9.1% behind. That conflates two separate mechanisms:

  - **TMA multicast** (`cp.async.bulk.tensor ... multicast::cluster`) loads global -> several
    CTAs' smem in one hardware op. That one does require TMA.
  - **DSMEM** does not care how the data arrived. One CTA loads KV with `cp.async` into its
    own smem; the other CTAs in the cluster read it directly via distributed shared memory
    (`mapa.shared::cluster`). The HBM saving -- one global load serving the whole cluster --
    is identical; TMA multicast merely does it in one instruction rather than explicit peer
    reads plus a cluster barrier.

  So **KV multicast is implementable on the cp.async path with no TMA penalty**, and the
  9.12% is irrelevant to it. The earlier entry that recorded "needs upstream's TMA paged path"
  as the multicast prerequisite is wrong for the same reason, and that is also why the item
  looked blocked for so long: it never was.
- **`stages=3` and `LPT=1` are both marginally worse** (0.5%, 0.7%), so upstream's values
  (2 stages, `LPT=0`) stand. `stages=3` fits only because `MmaPV_is_RS=1` frees exactly one
  32 768 B P tile, landing on 232 448 B -- it fits and does not pay.

## FA2 optimisation items, same fixture, same method — 2026-10-07

| arm | total | vs baseline |
|---|---|---|
| **`fa2_base`** (`FA2_MMA_KV=2`) | **15 437.0 us** | - |
| `fa2_kv3` | 15 370.9 us | -0.43% |
| `fa2_kv4` | 15 366.0 us | -0.46% |

All PASS, 0 unwritten, 0 fallbacks. **Flat at noise level**, which confirms on the fixed
kernel what the local `SharedStorage` check already showed: `FA2_MMA_KV` sizes the
register-side `s_frag` accumulator, not the KV pipeline depth (133 120 B at N = 2, 3, 4). So
item 1, K/V pipelining on FA2, is **still untested** -- it needs the real staging-depth knob,
and a candidate knob must be shown to move `SharedStorage` before any arm is spent on it.

**Routing rationale, stated once and not as an optimisation claim.** On this fixture both
executors do *identical* work -- the same two append rows -- and FA3 takes **11 164 us**
against FA2's **15 437 us**. That ratio is why the split routes appends to FA3, and it is the
mechanism behind the 1.239x the split shows on the full mixed batch. It is a routing fact, not
a per-executor optimisation result.

**Honest scale.** The whole FA3 item set is worth **under 4%** against its own baseline, and
the FA2 item set is worth **nothing measurable**. The
publishability bar is >=30% TTFT/ITL and >=10% throughput. Nothing in this list approaches it,
and the items were the plan for getting there. The only large number in this work remains the
`seqused` fix, which was repairing my own port defect rather than an optimisation.

## CRITICAL: several eliminations tested code that does not run — 2026-10-06

The per-work-item trace (`FLASHINFER_TRACE_WORK=1`, a `printf` in `persistent.cuh`'s work
loop) produced **zero lines** while the corruption reproduced normally (0.527). The printf
compiled -- verified locally with `-DFLASHINFER_TRACE_WORK=1` -- so it is not in the kernel
that executed.

**Why: under `TWO_EXEC=1` the FA2 side runs `BatchAttention.run()`, i.e. flashinfer's own
JIT-compiled module from `~/.cache/flashinfer/90a/generated/...`, not our extension's
translation unit.** The harness says so in its own comment ("FA2 on the rest via the
UNMODIFIED persistent LPT kernel (BatchAttention.run)"), and flashinfer's JIT caches on the
generated source plus flags -- **editing `include/` does not invalidate it**.

**This invalidates, as tests, every arm whose switch lives in a header the JIT module compiles
separately:**

| arm | switch location | status |
|---|---|---|
| `LSE_TILE_REL` 0 vs 1 | `persistent.cuh` | **tested nothing** on the FA2 path |
| `PI_ZERO` | `scheduler.cuh` | **suspect** -- the plan is also built by the JIT module (`wk.plan` -> `module.plan`) |
| `NO_JIT_REDUCE` | ext sets `P.group_id = nullptr` | **suspect** -- those params feed our kernel, not `BatchAttention.run()` |
| `TRACE_WORK` | `persistent.cuh` | confirmed inert: zero output |

The trait A/B (`MmaPV_is_RS`, `LPT`, `UP_STAGES`) and the `seqused` fix are **not** affected --
those live in `fa3_upstream.cuh`, which only the ext compiles, and the FA3 half is the ext's
own kernel. The 42.8x and the 1.23-1.25x results stand.

But the reducer-swap bisect, the `partial_lse` A/B and the `partial_indptr` A/B must be treated
as **unperformed**, not as eliminations. That is 3 of the ~24 candidates returned to the open
list, and it explains why the corruption was so stubborn: part of the search was editing a copy
of the code that never ran.

**What this changes about method.** Every arm needs a positive control that the switch reached
the running kernel -- a printed marker, a changed smem size, a deliberate crash. "It compiled
and the number moved/did not move" is not evidence the switch took effect. This is the fourth
instance this session of a check not touching the thing under test, and the most costly.

**CONFIRMED and FIXED: clearing the JIT cache makes edits take effect.** Deleting
`~/.cache/flashinfer/90a/cached_ops` (the compiled `.so` only -- **keep `generated/`**, which
the harness globs for its include path) and rerunning produced **3 616 trace lines** where the
same build produced **0** before. So:

- the stale JIT module was the reason, confirmed by a positive control rather than inferred;
- **`rm -rf ~/.cache/flashinfer/90a/cached_ops` is now mandatory before any FA2-side arm**,
  and every such arm needs a marker proving the switch reached the running kernel;
- the trace had to be made **unconditional** (`#if 1`) because the JIT module never receives
  our `-D` flags -- an `#if`'d switch stays inert there even after a rebuild. That is a
  diagnostic edit to revert, marked `TEMPORARILY UNCONDITIONAL` in the source.

**Three arms must be re-run before they count as eliminations:** `LSE_TILE_REL`, `PI_ZERO`,
`NO_JIT_REDUCE`.

**`LSE_TILE_REL` re-run and now genuinely eliminated.** A/B'd the way the JIT module requires
-- by moving the **header default** and clearing `cached_ops` each time, not by passing `-D`:

| `FLASHINFER_PARTIAL_LSE_TILE_REL` | victim rel err |
|---|---|
| **0** (row-relative, original) | 0.561, 0.520, 0.432 |
| **1** (tile-relative, matches `partial_o`) | 0.527 |

Same distribution, so the `partial_lse` index base is not the fault. This repeats the earlier
*conclusion* but is the first time it rests on a build where the switch provably reached the
running kernel -- the trace run established that edits take effect once `cached_ops` is
cleared. Default restored to **1**, which matches `partial_o` and the plan's per-tile
`partial_indptr` and is correct on its own terms.

**`PI_ZERO` re-run and also genuinely eliminated.** Same method -- header default
(`FLASHINFER_PI_ZERO_DEFAULT`) plus `cached_ops` cleared:

| `PI_ZERO` | victim rel err |
|---|---|
| 0 (upstream: `partial_indptr = partial_o_nnz` always) | 0.561, 0.520, 0.432 |
| **1** (zero it for non-split work items) | **0.380, 0.560, 0.482** |

Same distribution. Default restored to **0**, upstream's behaviour, since the change buys
nothing. So both arms that the JIT-cache defect invalidated reach the *same* conclusions they
did before -- but they are now supported, and that distinction matters: before this they were
claims about a kernel that never ran.

**The reduction-path A/B is INFEASIBLE: the classic reducer is non-functional.** Forcing it on
inside the JIT-compiled headers (`FLASHINFER_FORCE_CLASSIC_REDUCE=1`, both gates mirrored,
`cached_ops` cleared) gives **`rel err = nan` on every row**, including the leading row that
is otherwise always correct. The classic pass calls `grid.sync()`
(`persistent_template.cuh:109`), which requires a **cooperative launch**; this kernel is not
launched cooperatively, so the barrier is undefined and the reduction produces NaN.

So the "two mutually exclusive reducers" structure is real in the source but has only **one
working branch**. Three consequences:

1. The reduction path **cannot be exonerated or implicated by swapping reducers** -- that
   experiment does not exist. My earlier claim that both reducers produce the corruption was
   doubly wrong: the arm did not reach the kernel, *and* the alternative it was supposed to
   select cannot run.
2. The JIT reduction is therefore the **only** reducer, which makes the earlier framing
   ("the JIT reduction replaces the classic grid.sync pass") accurate but for a different
   reason than stated -- the classic pass is not a working fallback, it is dead code.
3. Testing the reduction's involvement needs a different lever: suppress the merge's *writes*
   while leaving the rendezvous intact (so the kernel still completes), and check whether the
   victim's rows change. That isolates "does the reduction write where it should not" from
   "does the reduction run at all".

`FLASHINFER_FORCE_CLASSIC_REDUCE` reverted to 0. The switch is kept, with this result recorded
next to it, so nobody re-runs the arm expecting it to work.

**`NO_JIT_REDUCE` cannot be re-run by clearing the cache.** It sets `P.group_id` in the
**ext's** params, and those params do not feed `BatchAttention.run()`, which is what executes
the FA2 rows. Testing it requires routing the FA2 rows through the ext's own runners first --
a structural change, not a cache clear. Until then its earlier result (both reducers corrupt)
is **unsupported**, and with it the claim that the whole reduction path is exonerated. That
claim should be treated as open.

**Net effect of the JIT-cache discovery on the candidate list:** two candidates return as
*confirmed* eliminations, one (`NO_JIT_REDUCE`, and the reduction-path exoneration that rested
on it) returns to *open*. The reduction machinery is therefore back on the suspect list.

**First real data from the trace, on the victim's work items:**

    work=9  q_indptr=1 q_len=512 packed_qo_start=512  kv_indptr=9 kv_len=1024
            kv_start=0 kv_end=256 kvh=1 chunk=0 nchunks=4

The victim is split into **4 chunks of 256** (`len_kv_chunk=256` for its task here), which
again contradicts the earlier "the follower does not split" reading and matches the plan dump.
`q_indptr=1`, `kv_indptr=9` and `q_len=512` are all correct.

**Caveat on my own first pass at the analysis:** the "8 duplicates per work item" the script
reported is **not** a finding. The harness launches the kernel several times (numerics checks
at multiple `sm_split` values, then timing) and the `sort | uniq -c` collapsed across launches.
Per-launch separation is needed before any claim about double execution -- the trace should
carry a launch counter.

## Split-row race: consolidated state — 2026-10-06

**Reproducer** (one token flips it, `n_fa3:0`, page 128 or 1, `UNIFIED=1`):

    FAIL  qo:1,512   prefix:65536,0     victim rel err 0.58-0.85, varies per run
    PASS  qo:512,1   prefix:0,65536     both rows correct

**Established, each by measurement:**

1. **Nondeterministic.** Eight identical runs: 0.773 0.658 0.725 0.682 0.680 0.684 0.670 0.683.
   So it is a race. Error magnitude across fixtures is this variance, not a signal.
2. **Damage is total**: 512/512 tokens, 28/28 heads, every element written, magnitude in range
   (absmax 0.0703 vs ref 0.0715). Not a bounded overrun, not zeros, not garbage.
3. **The victim need not split.** A victim with `kv=512` (one chunk, direct write-through, no
   partials) is corrupted. So it is not a merge-weight or partial-buffer fault.
4. **The whole reduction path is exonerated.** The JIT rendezvous and the classic
   `grid.sync` + `BlockReductionRunner` path are mutually exclusive
   (`persistent_template.cuh:108`), and the corruption appears identically under **both**.
   The cause is upstream of reduction.
5. **The plan is correct**, dumped on two different fixture pairs: `q_indptr`, `kv_indptr`
   (513 = `ceil(65537/128)`), `kv_len`, `kv_start`/`kv_end`, `batch_idx`, `merge_indptr`,
   `merge_o_indices`, `group_row_begin`/`nrows`, `partial_indptr` all as expected, and
   structurally identical between PASS and FAIL.
6. **Trigger**: a row that **splits** (`kv >` its task's `len_kv_chunk`) at a **lower request
   index** than the victim. Same-task pairs fail too (`qo:127,512`), so it is not cross-task.
   Both victims fail in a three-row fixture, so it is not `q_indptr`-specific.
7. **Not page alignment** (fails at `page_size=1`), **not** `MmaPV_is_RS`/`LPT`/stages,
   **not** the FA3 executor or the combine (`n_fa3=0` reproduces it).

**Open question, and no clean control exists for it.** Whether the defect is in our 4-task plan
path or in flashinfer's persistent kernel is **undetermined**: the `fi_only.py` comparison that
appeared to settle it was confounded (3-task vs 4-task), and `BatchAttention.run()` refuses a
4-task plan, so upstream cannot execute the configuration the fault requires. Building a
3-task reproducer -- if one exists -- would be the way to separate them.

**Serving is genuinely gated.** `flashinfer_backend.py:1945` sets `FLASHINFER_FA3_TASK=1` on
the `SGLANG_TWO_EXEC` path (4 tasks, affected) while `:1958` plans 3 for the baseline
(unaffected), so an A/B today compares a clean baseline against an arm computing wrong
attention. The eventual run needs a **third validity check beyond fallback counts: both arms
must plan the same task count.**

**RACECHECK IS CLEAN -- it is not a shared-memory race.** `compute-sanitizer --tool racecheck
--racecheck-detect-level info` on the minimal reproducer (`qo:1,512 prefix:1024,512`, leading
row splitting into 2 chunks) reports **no hazards at all**, while the corruption still
reproduces under instrumentation (row 1 rel err **0.568**). The run was genuinely instrumented
-- `fa2_half` went from ~250 us to **350 562 us**, a ~1400x slowdown -- so the clean result is
not a case of the tool failing to attach.

`racecheck` detects **shared-memory** hazards only. Combined with the nondeterminism (eight
runs, eight errors), that localises the nondeterminism to **global memory ordering between
CTAs**, not to anything in a block's shared memory. Every fence-ordering suspect in the JIT
rendezvous operates on shared memory or block-scope barriers, which is consistent with the
rendezvous already having been exonerated by the reducer-swap bisect.

So the picture is now: correct plan, correct reduction machinery (both paths), no shared-memory
race, total in-range damage to a row that writes through directly, nondeterministic, triggered
by a *preceding* split row. What remains is an unsynchronised **global** write -- two CTAs
writing the same `final_o` rows with no ordering between them -- or a read of `final_o` /
`partial_o` that is not ordered against another CTA's write.

**Next experiments, in order of cost:**

1. Try to reproduce with a **3-task** plan (`FLASHINFER_FA3_TASK` unset, `UNIFIED=0` is not
   available because that path no longer compiles -- it references `wgmma_plan`). If it
   reproduces, the fault is in flashinfer's kernel, not our plan, and upstream can own it.
2. ~~`compute-sanitizer --tool racecheck`~~ and ~~`--tool memcheck`~~ -- **both done, both
   clean**. racecheck: no shared-memory hazard. memcheck: `ERROR SUMMARY: 0 errors`, i.e. no
   out-of-bounds or misaligned global access. The corruption reproduced under each
   (0.568 and 0.487 respectively), so neither is a case of the tool not attaching -- the
   instrumented runs were 1400x and 22x slower than normal.

   **Together these are a strong joint constraint:** the damaging write is *in bounds* and
   *not racing on shared memory*, so it is a correct-looking store to the **wrong rows** --
   an indexing or work-assignment fault, not a memory-safety or synchronisation one. Note the
   tension with the nondeterminism (eight runs, eight errors): an indexing fault that is
   nondeterministic means the *index* varies run to run, which points at something derived
   from scheduling order -- which CTA picked up which work item -- rather than from the plan,
   which is known correct and identical between runs.
3. Instrument the victim's work items to record which `(q_start, kv_start, kv_head)` each CTA
   actually executed, and diff against the plan -- the plan is known good, so a divergence
   localises the work-distribution side. **Both sanitizers now point here**, since an
   in-bounds store to the wrong rows whose index varies per run is precisely a
   work-assignment fault.

   **Do not try to approach this with `FA3_SM_BUDGET`.** I attempted a cheap proxy -- sweep
   the SM budget (132/32/8/1) and look for the error becoming deterministic -- and it was
   **ineffective by construction**: `FA3_SM_BUDGET` caps the **FA3** executor, and these
   fixtures run `n_fa3=0`, so FA3 has no work and the knob cannot affect the FA2 path under
   test. The eight values (0.39-0.53) are the run-to-run spread already measured, not a
   signal. This is the same mistake as the `FA2_MMA_KV` sweep: varying a knob that does not
   touch the path being measured. The check that catches it is cheap and should be habitual
   -- confirm the knob changes something observable on the path (a `SharedStorage` size, a
   work-item count, a generated symbol) **before** spending arms on it.
