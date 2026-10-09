# Current goal

**Decisions taken 2026-10-09 (user), these override earlier framing:**

1. **The fused kernel MUST land on upstream** -- port it to flashinfer 0.7 rather than demo on the
   fork. The port may also make it faster.
2. **Per-class SLO targets derive from the common settings of the given papers**, not invented.
3. **The >=30%/>=10% bar is against the baselines, plural**: single node must beat **both**
   prefill-prioritized (the default) **and** mixed chunked prefill; multi node must beat **both**
   PD multiplex **and** PD disagg -- *beating disagg is the main argument*, mux is not designed
   for multi-node so that win should come easily.
4. **H20 / multi-node only after ALL kernel items are closed.**
5. **-5% to -8.4% in-server is NOT an acceptable result.** The kernel must be a gain.

## The tile-occupancy defect in the current dispatch (user's observation, 2026-10-09)

Asked whether everything with a tile size > 16 (post-pack-GQA) goes to FA3. It does not, but the
real gate is worse than that framing suggests. `split_classes` routes

```python
packed_qo = qo * gqa_ratio
fa3 = (packed_qo >= MIN_PACKED_QO=48) & (kv_tiles >= 1) & (qo >= MIN_QO_LEN=16)
```

while the FA3 side uses a **fixed `TILE_Q = 128`**. So any row with `packed_qo` in **[48, 128)**
is sent to FA3 and handed a 128-wide WGMMA tile it cannot fill -- **up to 63% of the tile idle**.
That is genuine tile waste, and `MIN_PACKED_QO = 48` is a hand-set constant that admits exactly
the rows which cannot fill the tile. The persistent/FA2 side already has three q-tile classes
(`tile128` for packed > 64, `tile64`, `tile16`), so 64 exists and **32/48 would be new**.

**Precision on priority:** for the current config (Qwen2.5-7B, 28 qo / 4 kv -> GQA 7, 4 draft
tokens) a verify row is `packed_qo = 28` and already routes to FA2, so the waste band is NOT the
cause of today's -5%. It bites when `num_draft_tokens x gqa` lands in [48,128) -- 8 draft tokens
at GQA 7 is 56; 4 draft tokens at GQA >= 12 likewise. Today's -5% is the complementary half of
the same problem: the FA2 side is 7 rows x 4 tokens = seven `tile16` items in a **separate
launch** plus two `index_copy_` scatters, against the baseline's single kernel.

**Both halves resolve in the fused single kernel**, which is why the port is the right call: small
verify tiles are absorbed into the same grid by the LPT overlap schedule (no second launch, no
scatter), and FA3's tile width becomes a per-work-item property instead of a compile-time
constant.

## MOTIVATION REQUIREMENT: fuse as many small draft/verify ops as possible (user 2026-10-09)

*"Make sure we fuse as many small draft/verify operations in the two kernels as possible,
otherwise it's hard to make a meaningful motivation for fusing the kernel -- a reviewer will ask
why start from the kernel point of view for colocation if gains are so minimal?"*

This is a **framing constraint on the whole kernel effort**, and it reorders the work. Fusing one
large prefill chunk with seven verify rows is the marginal case: it cannot carry a kernel-level
contribution, and a 1-2% end-to-end number invites exactly that reviewer question. The claim has
to be that **the small spec operations individually cannot fill the GPU, and only a fused kernel
can reclaim that** -- which is also what the goal already asks for under "multicast and
**cross-role backfill** (needs `tma_store_wait`)".

**The motivation must be MEASURED FIRST, before any of K2-K5.** Optimising before establishing
the headroom is how the -5% happened: I tuned a kernel whose benefit at the real batch shape was
never quantified.

A 4-mode step currently issues, per layer: 3 draft decode steps (`--speculative-num-steps 3`),
2 draft extends, and 1 mixed target forward. All but the target's prefill rows are **small**:
draft decodes are 1 token/row, draft extends are short, verify rows are
`num_draft_tokens x gqa` packed. Every one is its own launch with its own tail.

- [ ] **K-M (do FIRST). Quantify the headroom the fusion is supposed to reclaim.** Per 4-mode
      step, report:
      1. **N** -- the count of distinct attention launches, split by role
         (draft-decode / draft-extend / target-prefill / target-verify);
      2. **X** -- the fraction of step wall time spent in the small ones;
      3. **Y** -- their achieved SM occupancy (so "small" is a measurement, not an adjective);
      4. **the tile histogram** by `packed_qo` class, which is what K2-K4 act on.
      Use nsys/ncu or CUDA events around each role. **This is the reviewer-facing motivation
      table**: without it the kernel work has no stated headroom, and with it the target gain is
      bounded and honest. If X turns out small, say so -- that is the finding, and it would mean
      the kernel is NOT the right place to start for colocation and the argument belongs on the
      scheduling axis.
- [ ] **K-M2. Then set the fusion target from N and X**: which of the N launches can be absorbed
      into the two kernels, via cross-role backfill (small draft/verify work filling the SMs the
      prefill mainloop leaves idle) and multicast. Note the draft ops run a DIFFERENT model, so
      absorbing them is cross-model co-scheduling, a strictly bigger step than the
      prefill+verify split already built -- scope it explicitly rather than assuming it falls
      out of the existing partition.

### The rationale, in the user's terms (2026-10-09)

**The objective is to minimise tile waste**, and the three row populations in a 4-mode step sit
at different places on that axis:

| population | packed_qo | engine | why |
|---|---|---|---|
| **large prefill chunk** | thousands | **FA3** | fills a 128-wide WGMMA tile many times over; this is what FA3 is for |
| **decode / draft** | small (1 token x gqa) | **FA2** | a single `tile16` item; FA3's 128-wide tile would be ~90% idle |
| **verify** | `num_draft_tokens x gqa` -- anywhere from ~16 to >128 | **the middle area, and the whole question** | neither engine is obviously right, and this is where the gain has to be won |

So the verify rows are the contested band, and the current dispatch resolves them with a
hand-set constant (`MIN_PACKED_QO = 48`) against a fixed `TILE_Q = 128`, which is precisely the
combination that wastes the tile. Getting verify right is the point of K2-K4; prefill and
decode/draft are already routed correctly and are not in question.

**What makes finer FA2 tiles cheap (user's key point): FA2's tile sizes are all composed of
`mma16`s.** So 16/32/48/64 are not new kernels -- they are **template instantiation choices**
over the same primitive, and the **2-CTA overlap schedule is preserved** across them. The cost
is compile time, code size and register pressure per extra instantiation, not new scheduling
machinery. That turns K3 from "write more kernels" into "pick how many widths are worth
instantiating", which is an experiment: instantiate a set, measure, and keep the widths that pay
for themselves.

### Open question answered: NO, the tree build is NOT fused into FA2 on upstream (2026-10-09)

Asked directly, and checked in code rather than from memory. **`mix_with_verify` carries no
`spec_info` and no `custom_mask`** -- there are zero references to either in it. Consequences:

* The merged 4-mode batch therefore takes the planner's **`spec_info is None`** branch
  (`qo_indptr = cumsum(seq_lens - prefix_lens)`, which is what the PLANWIDTH measurement
  confirmed), so the verify rows are masked **plain causal**.
* That is correct **only for a CHAIN draft** (`--speculative-eagle-topk 1`, which is what every
  measurement here used). For `topk > 1` the real tree mask is required, because siblings must
  not attend to one another, and causal masking would silently let them.
* **The fused split inherits the restriction**: its FA2 subset plan passes no `custom_mask`
  either (zero references in `_fused_split_plan`/`_fused_split_dispatch`).
* Upstream's ordinary verify path DOES have the machinery -- `use_custom_mask` in
  `_create_prefill_wrappers`, `EagleVerifyInput.custom_mask`, and the tree builders in
  `speculative/eagle_utils.py` -- so this is something the 4-mode merge **drops**, not something
  upstream lacks.

**Gate blind spot worth recording:** the in-step verify gate could never have caught this. Its
reference is the same paged wrapper under the same causal assumption, so both sides share the
error -- a null case that agrees with the thing under test proves nothing, which is the same
lesson as [[gate-needs-noise-floor-and-reference]] in a new place.

So the 4-mode step as ported is **chain-only**, and that is a scope limitation of every result in
this doc, not a bug in any one measurement. Supporting trees means carrying
`EagleVerifyInput.custom_mask` through `mix_with_verify` into the merged batch and planning the
paged wrapper (and the split's FA2 subset) with it, sliced to the verify rows.

- [ ] **K0. Decide and record whether trees (topk>1) are in scope.** If yes, plumb the tree mask
      through `mix_with_verify` and the split's FA2 plan, and add a gate whose reference does NOT
      share the causal assumption (an fp32 tree-masked reference, not the paged wrapper). If no,
      state "chain draft only" as an explicit limitation wherever the 4-mode results are quoted.
      Listed as K0 because it bears on correctness of everything above it, not on performance.

### Design, in dependency order

- [ ] **K1. Port the fused kernel to flashinfer 0.7.** Reconstruct the JIT-reduction group tables
      (`group_id`, `group_nchunks`, `group_row_begin`, `group_nrows`) that 0.7's
      `HolisticPlanInfo` no longer carries; resolve `DefaultAttentionSm90` /`StandardAttention`
      renames; avoid the `variant_helper.cuh` vs `hopper/variant_helper.cuh` collision. Blocks
      everything else.
- [ ] **K2. Dynamic WGMMA tile size in FA3: mix 128 and 64, chosen per work item** from
      `packed_qo`. A 56-row item then runs at 88% tile occupancy instead of 44%.
- [ ] **K3. Extend the FA2/persistent tile classes to 16/32/48/64** -- template instantiations
      over `mma16`, not new kernels, with the **2-CTA overlap schedule preserved**.
      **Experiment on how many widths are worthwhile**: each instantiation costs compile time,
      code size and registers, so sweep {16,64} vs {16,32,64} vs {16,32,48,64} and keep only the
      widths that pay for themselves on measured verify-row shapes. Report the chosen set with
      the per-width timings that justify it.
- [ ] **K4. Replace `MIN_PACKED_QO = 48` with a FITTED dispatch rule for the VERIFY band** on
      `(packed_qo, kv_len)` -- a linear threshold or small learned model, trained on measured
      per-tile times for FA3-at-64 vs FA2-at-{16,32,48,64}. Machinery already exists: the
      measured boundary grid `tasks/fused_specdec_workload/boundary_28x4x128.json` (25 points)
      and `_apply_decode_population_guard`, which replays it (11/25 -> 23/25 agreement). The
      point is to stop hand-setting the constant that causes the waste.
      **User 2026-10-09: the threshold needs ACCURATE MEASUREMENT, so a linear model is the
      recommendation; simpler forms may be tried too.** So: measure per-tile times for
      FA3-at-{64,128} against FA2-at-{16,32,48,64} across a `(packed_qo, kv_len)` grid, fit a
      linear decision boundary, and report its agreement against the measured winner the way the
      decode guard's 11/25 -> 23/25 is reported. Try a bare threshold and a 2-D ratio rule as
      baselines for the fit -- if a one-constant rule matches the linear model's agreement, keep
      the constant, but it must be FITTED to measurements rather than chosen.
      **The measurement grid must be denser than the existing 25 points**, which were taken for
      the decode-population question, not for tile-width selection.
- [ ] **K5. Re-measure in-server at the shape the schedule emits**, not a chosen one. Report
      `last_fa2_rows` and the per-class tile histogram beside every timing, and require the gain
      to clear the measured +/-1.5-3% noise floor.

**Gate on everything downstream (user 2026-10-09):** K1-K5 closing is the precondition for H20
and the multi-node work -- *"after all kernel items closed"*. The superset/capacity items
`5a-5f` in `pd_colocation_vs_disagg.md` therefore wait on this section, and the single-node
item 5a (which needs only the current H200) is the one piece of that work that can proceed in
parallel.


**Land the fused FA3+FA2 SM-pre-assignment kernel ON UPSTREAM, alongside the 4-mode MTP step
already there, then measure the combined arm.** Target: >=30% TTFT and ITL, >=10% throughput on
mixed chat + tracelab. Corrected 2026-10-09 (user): the kernel must be on upstream and its
*interfaces and fusion boundaries must fit upstream* -- a 4-mode schedule running upstream's own
attention measures the scheduling change alone and cannot show the thesis.

## Tasks, in priority order (2026-10-09)

Open items consolidated in one place, completed items kept below them. **Kernel items gate
everything else** (user: H20 and multi-node only after all kernel items close).

### OPEN -- kernel: TWO MAIN ITEMS (user 2026-10-09)

**Prerequisite for both: K1 -- port the fused kernel to flashinfer 0.7** (must land upstream).
Reconstruct the JIT-reduction group tables 0.7's `HolisticPlanInfo` dropped (`group_id`,
`group_nchunks`, `group_row_begin`, `group_nrows`); resolve `DefaultAttentionSm90` /
`StandardAttention`; avoid the `variant_helper.cuh` vs `hopper/variant_helper.cuh` collision.
The single-launch cost structure is the only one the serving data says can win.

---

#### B1 RESULT (2026-10-09): the headroom is real but it is mostly GEMM, not attention

Driver `drv/b1_headroom.sh`, analyzer `drv/b1_analyze.py`, trace
`/dev/shm/w/prof/1791522716.762536-TP-0.trace.json.gz` (355.7 MB). 4-mode MTP server, EAGLE3
chain draft (topk 1, 4 draft tokens), split ON / guard OFF / graphs off, warm pass before the
capture, load during it (`--random-input-len 2048 --random-output-len 128 --request-rate 24
--max-concurrency 48`). Composition during capture:
`FUSED_SPLIT_CALLS 501 (fa3_rows=2 fa2_rows=22 onesided=3)`, `MIXEDSPEC_STEPS 1`.

**66.5 ms of GPU time over 7250 launches.**

| role | launches | GPU ms | share of GPU time | small N (grid<132) | small share of role |
|---|---|---|---|---|---|
| **gemm** | 1538 | **52.16** | **78.5%** | 1446 | 88.4% |
| attention | 1595 | 7.68 | **11.6%** | 1225 | 87.5% |
| epilogue/elementwise | 3637 | 5.93 | **8.9%** | 3612 | 98.4% |
| memcpy/memset | 468 | 0.68 | 1.0% | 468 | 100.0% |
| other | 12 | 0.02 | 0.0% | 12 | 100.0% |

**89.3% of GPU time is in launches that cannot fill the device** (6763/7250 launches, 59.37 of
66.49 ms, grid < 132 blocks). Top kernels are `nvjet_sm90_*` (cuBLASLt) at median grid 100-112 --
the TARGET model's own linear layers failing to fill 132 SMs at this batch size.

**What this means for the two main items, stated against the bar:**

* **Item A (tile dispatcher) is bounded by 11.6%** of GPU time, and item **B (epilogue fusion) by
  8.9%** -- about **20% combined, before** discounting the part that is not actually reclaimable
  (a fused kernel still has to do the arithmetic; only the idle-SM fraction is recoverable).
* So **no attention-side work can reach the >=30% TTFT/ITL target**, which is consistent with the
  independent `g * a * 0.18` arithmetic recorded earlier that put the ceiling at ~5-7%. Two
  different measurements now agree on that conclusion.
* **The dominant idle capacity is GEMM**, at 78.5% of GPU time with 88.4% of it in
  sub-132-block launches. If the paper's kernel contribution is to be proportionate to the
  motivation table, it has to touch that -- e.g. cross-role backfill at the GEMM level (draft
  model GEMMs filling the SMs the target's GEMMs leave idle), which is a different and larger
  piece of work than the attention split.

**The honest reviewer-facing framing**, which this table supports and the previous framing did
not: *in colocated spec-decode serving the GPU is chronically underfilled -- 89% of GPU time in
launches that cannot occupy the device -- and fusing across roles is how that capacity is
reclaimed.* Attention is the demonstrator, not the whole prize, and the table says so.

**Caveat on scope, recorded rather than buried:** `MIXEDSPEC_STEPS 1` against 501 split calls
means this capture is dominated by ORDINARY extend batches, not 4-mode merges -- with the guard
off and paged forced, the split fires on any two-sided extend batch. A 4-mode-heavy capture could
shift the attention share up somewhat. It will not change the GEMM dominance, which is per-token
work, but the exact A/B bounds above should be re-read from a capture where
`MIXEDSPEC_STEPS` is large.

- [x] **B1 DONE: headroom quantified.** 89.3% of GPU time in sub-occupancy launches; attention
      11.6%, epilogue 8.9%, GEMM 78.5%.
- [ ] **B1b. Re-capture with a 4-mode-heavy workload** (`MIXEDSPEC_STEPS` large) to tighten the
      attention/epilogue bounds; the GEMM share is not expected to move.
- [ ] **B1c. DECISION: does the kernel story extend to GEMM-level cross-role backfill?** The
      motivation table says that is where the idle capacity is. If it stays attention-only, the
      kernel contribution is bounded at ~20% of GPU time and the headline claim must be the
      scheduling one -- which is where the measured TTFT wins already are.

#### B1b RESULT (2026-10-09): **RETRACTS B1's 89% figure.** The GPU is NOT underfilled at load

Driver `drv/b1b_headroom.sh` (load ramped 25 s BEFORE the capture window opens, so decode rows
are resident and the 4-mode gate has a running batch to merge with), analyzer `drv/b1_analyze.py`,
trace `/dev/shm/w/prof/1791525002.0451415-TP-0.trace.json.gz` (76.1 MB). Composition during
capture: **`MIXEDSPEC_STEPS 201`** and
**`FUSED_SPLIT_CALLS 6001 (fa3_rows=1 fa2_rows=61 onesided=21)`** -- so this trace genuinely
exercises the 4-mode merged path, which B1's did not.

**1653.3 ms of GPU time over 45617 launches** (B1 had 66.5 ms over 7250).

| role | launches | GPU ms | share of GPU time | small N | small ms | small share of role |
|---|---|---|---|---|---|---|
| gemm | 7926 | 1163.27 | **70.4%** | 3503 | 101.48 | **8.7%** |
| attention | 9725 | 339.59 | **20.5%** | 3459 | 10.69 | **3.1%** |
| epilogue/elementwise | 21711 | 141.01 | **8.5%** | 7508 | 12.04 | 8.5% |
| memcpy/memset | 6164 | 9.01 | 0.5% | 6164 | 9.01 | 100.0% |
| other | 91 | 0.41 | 0.0% | 91 | 0.41 | 100.0% |

**Sub-occupancy launches (grid < 132): 20725 of 45617 launches (45.4%) but only 133.63 ms of
1653.30 ms = 8.1% of GPU time.**

**RETRACTION, in place.** B1 reported *"89.3% of GPU time is in launches that cannot fill the
device"* and I built a motivation framing on it -- *"the GPU is chronically underfilled in
colocated spec-decode serving"*. **That figure was an artifact of an unrepresentative sample.**
B1's trace was 355 MB but held only **66.5 ms** of GPU time: overwhelmingly CPU/idle events with
a tiny, warmup-like GPU window, captured before any decode backlog existed. B1b has **25x the
GPU time** on the path that matters and gives **8.1%**. The correct statement is the opposite of
the one I made: **at sustained load the device is well filled** -- GEMMs are only 8.7%
sub-occupancy and attention only 3.1%.

I should have distrusted a 66 ms GPU sample before drawing a framing conclusion from it. The
trace SIZE (355 MB) made it look substantial; the GPU-time total is the figure that mattered and
it was on the same line of output.

**Revised bounds for the two main items, from the representative capture:**

* **Item A (tile dispatcher) acts on attention's 20.5%** of GPU time -- nearly double B1's 11.6%.
  **Important metric caveat:** `grid < 132 blocks` measures GRID-level occupancy, so attention's
  3.1% small share does **not** mean tiles are full. The waste item A targets is *within* the
  WGMMA tile (a 48-row item in a 128-wide tile), which this metric cannot see. **A needs its own
  measurement**: the `packed_qo` histogram weighted by time, which `b1_analyze.py` does not yet
  emit.
* **Item B (epilogue fusion) is bounded by 8.5%** of GPU time across **21711 launches** -- a very
  long tail of tiny kernels, but a small share of time, and only 8.5% of that share is
  sub-occupancy (0.7% of total). So B's reclaimable part is launch overhead and idle gaps within
  ~8.5%, not the whole 8.5%.
* **The >=30% bar remains unreachable from attention+epilogue**: 20.5% + 8.5% = 29% of GPU time
  is the absolute ceiling if both were made *free*, which they cannot be. This agrees with the
  `g * a * 0.18` estimate (~5-7%) and with B1's conclusion, by a different route.
* **GEMM is 70.4% and mostly well-occupied**, so the GEMM-level cross-role backfill idea raised
  in B1c is much weaker than B1 suggested -- there is little idle GEMM capacity to reclaim at
  this load.

- [x] **B1b DONE**, and it **retracts B1's headline**. Representative figures: attention 20.5%,
      epilogue 8.5%, GEMM 70.4%, sub-occupancy only 8.1% of GPU time.
- [ ] **B1d. Emit the time-weighted `packed_qo` tile histogram** -- the measurement item A
      actually needs, which the grid-occupancy metric cannot provide.
- [x] **B1c answered (negatively): GEMM-level backfill is not the opportunity.** GEMMs are only
      8.7% sub-occupancy at load. Dropped rather than left open.

#### B1d RESULT (2026-10-09): item A is priced, and its value depends on the MTP config

`sglang-up@ed6eef498` adds host-side FLOP-weighted tile-coverage accounting
(`mixed_dispatch.tile_waste_accumulate` / `tile_waste_report`, `SGLANG_FUSED_TILE_STATS=1`).
Per row it charges `ceil(packed_qo/width)*width x ceil(kv/TILE_KV)*TILE_KV` covered against
`packed_qo x kv` useful, bucketed by `packed_qo`. No profiler and no timing: the waste item A
targets is a property of the SHAPES, and grid-level occupancy structurally cannot see it (a
48-row item in a 128-wide tile keeps every SM busy while most of each tile idles).

**A metric bug caught before it produced a wrong conclusion.** The first version charged
`TILE_Q = 128` to every row and reported **88% waste** on packed-16 verify rows. That is
invented headroom: those rows route to FA2's existing **`tile16`**. Fixed to charge each row its
ACTUAL width -- FA3 always 128; FA2 its class (128 above 64, else 64 above 16, else 16).

**Verify-dominated steps** (no large prefill row to dilute the FLOPs), 48 rows, kv 2000:

| config | `packed_qo` | charged tile | waste |
|---|---|---|---|
| GQA 4, 4 draft tokens -- **our current config** | 16 | 16 | **2.3%** |
| GQA 7, 8 draft tokens | 56 | 64 | **14.6%** |
| GQA 12, 4 draft tokens | 48 | 64 | **26.8%** |
| GQA 7, 12 draft tokens | 84 | 128 | **35.9%** |

**Mixed steps** (one 2048-token prefill chunk + verify rows): overall waste collapses to
**0.3% - 2.1%**, because the prefill row carries **89-94% of useful FLOPs at zero waste**
(packed 8192 is exactly 64 tiles of 128).

**Verdict on item A, stated plainly:**

1. **At our current config it is worth essentially nothing** -- 2.3% of attention FLOPs in the
   best case for it (verify-only), 0.3% in a realistic mixed step. Attention is 20.5% of GPU
   time, so that is under 0.5% end-to-end, far below the **+/-1.5-3%** noise floor.
2. **At other MTP configs it is worth a great deal** -- up to **35.9%** of attention FLOPs. The
   pattern is sharp: waste is large exactly when `packed_qo` lands just above an available
   width, and FA2's widths are only {16, 64, 128}.
3. **The decode-population guard already prevents the specific waste originally suspected.**
   `split_classes` returned `fa3_rows=1` in every mixed case: rows in the 48-127 band are
   un-routed from FA3's fixed 128-wide tile to FA2, where they get 64. So A1 (dynamic 128/64 in
   FA3) addresses a case the guard mostly already avoids.
4. **The worst remaining band is 65-128**, not 16-64. Packed 84 charged 128 wastes 35.9%; the
   proposed width set {16,32,48,64} does not cover it. **`tile96` would pay more than `tile32`.**
5. **Item A's value is therefore a function of `num_draft_tokens x gqa`, which is a spec-config
   choice, not something the kernel controls.** If the kernel contribution is to matter, the
   config has to put `packed_qo` in an awkward band -- and more draft tokens is independently
   desirable for acceptance length, so this is a coherent design direction rather than a
   contrivance. But it must be stated as a config-dependent result, not a general speedup.

- [x] **B1d DONE: item A priced.** 0.3-2.3% at our config; 14.6-35.9% at GQA 7-12 with 8-12
      draft tokens in verify-dominated steps.
- [ ] **A2 revised: instantiate {16, 32, 48, 64, 96} and measure**, with `tile96` prioritised --
      the 65-128 band is the worst offender and the original proposal omitted it.
- [ ] **Decide the MTP config question**: our GQA-4 / 4-draft-token setup makes item A worth ~0.
      Either measure at a config where it pays (more draft tokens, which also raises acceptance
      length) or report item A as config-dependent with the band table above.

#### B2 RESULT (2026-10-09): epilogue fusion is bounded at ~6%, and the split's own scatters are 1.2%

`drv/b2_top.py` on the B1b trace, sorting by **launch count** rather than time -- B1b's analyzer
sorted by time, which hides the fusion targets behind a few large GEMMs. 1644.3 ms kernel time.

| kernels with median duration | launches | ms | share of kernel time |
|---|---|---|---|
| < 5 us | 23448 | 104.94 | **6.4%** |
| < 10 us | 28887 | 186.16 | 11.3% |
| < 20 us | 30686 | 213.14 | 13.0% |

**So epilogue fusion's ceiling is ~6% of kernel time**, and the realistic figure is well under
that: fusing cannot delete the arithmetic, only the per-launch overhead (~2 us of a 3.4 us
kernel). Top offenders by count are mostly MODEL-FORWARD ops, not spec-specific:
3782 `fused_add_rmsnorm` (med 3.4 us), 2216 `vectorized_gather`, 1891 `fused_rope`,
1891 `act_and_mul` (48.24 ms, the largest single epilogue cost), 1771 `FillFunctor` (med 1.4 us),
1682 `PersistentVariableLengthMergeStates`, 1680 `store_kvcache`.

**The actionable finding is self-inflicted: `index_copy_kernel_impl`, 1680 launches, 19.65 ms =
1.2% of kernel time.** That is the split's own two scatters per layer (28 layers x 60 steps),
plus some of the 1771 `FillFunctor` launches from the `empty_like` pattern -- overhead the split
itself added, sitting inside its own -5% regression.

**Removed in `sglang-up@91955d929`.** In a merged 4-mode batch the token layout is
`[prefill chunk | verify rows]`, so FA3 rows form a contiguous **prefix** and FA2 rows a
contiguous **suffix**; both are then plain views, so each executor reads its `q` slice and writes
its output slice directly and the scatters disappear. Contiguity is **checked, not assumed** --
both row sets must be contiguous ranges whose token spans start at 0, end at `tok_total` and
abut, else the scatter path is kept. CPU-verified on three shapes: a merged 4-mode batch yields
spans `(0,2047)+(2047,2075)`; an interleaved partition is refused; a verify-first ordering still
qualifies with FA3 as the suffix. A `fused_split_views` counter records how often the view path
fires so an A/B can confirm it rather than infer it.

- [x] **B2 bound measured: ~6.4% of kernel time** in sub-5us launches; mostly model-forward
      elementwise, not spec epilogue.
- [x] **Split's own scatters identified (1.2%) and eliminated** via contiguous-span views
      (`91955d929`). Not yet measured in-server.
- [ ] **B2b. Re-run the warmed A/B with the view path** -- the first change in this whole
      sequence that removes measured, attributable overhead rather than a hypothesised one.
- [ ] **B2c. `tma_store_wait` cross-role backfill and multicast** -- still untested, and now
      bounded by the ~6% figure above, so expectations should be set accordingly.

#### B2b RESULT (2026-10-09): **the batch-shape hypothesis is REFUTED.** The two-executor split loses at every shape

`sglang-up@91955d929` (+ view-counter logging), driver `drv/split_perf_mtp.sh` with
`MRR=96 MC=64 RR=24 NP=240`, per-arm warmup, arm order reversed with `base` LAST, graphs off.
Raw: `/dev/shm/w/split_perf_mtp_results.txt`.

| arm (exec order) | tok/s | Mean TTFT | P99 TTFT | Mean ITL | P99 ITL | views | fa2_rows |
|---|---|---|---|---|---|---|---|
| skipplan | 12169.84 | 245.74 | 2040.75 | 20.81 | 86.17 | 6001 | 62 |
| noguard_m0 | 13225.27 | 164.52 | -- | 19.23 | -- | 6001 | 62 |
| guard_on | 13192.77 | 160.64 | -- | 19.21 | -- | 5501 | 63 |
| **base (split OFF)** | **14545.47** | **152.33** | -- | **17.63** | -- | -- | -- |

Best split arm is **-9.1% tok/s**, **+9.1% mean ITL**, **+8.0% mean TTFT** against base.
Accept length identical (2.07) in all arms, so the arms are comparable.

**RETRACTION: the batch-shape hypothesis is refuted.** I argued that the split regressed because
the server produced `fa2_rows=7` while the isolated 1.224x was measured at `fa2_rows=40`, and
that a decode-rich batch would flip the sign. This run has **`fa2_rows=62`** -- *more* favourable
than the harness shape -- and the split is **worse** (-9.1%) than at `fa2_rows=7` (-8.4%). Shape
is not the explanation. Recorded in place; the hypothesis was reasonable and is wrong.

**Also settled: the scatters were not the explanation either.** `views=6001` confirms the
scatter-free path fired on every dispatch (the counter is logged, so this is measured, not
inferred), removing the 1.2%-of-kernel-time `index_copy_` cost that B2 identified. The
regression did not improve.

**And `skip_base_plan` HURTS.** With `base_plan_skipped` confirmed firing, that arm is the worst
of the four (12169.84 tok/s, P99 TTFT 2040.75 ms). Skipping the full-batch plan leaves
`forward_metadata` pointing at wrappers that were never planned, so any layer where the dispatch
declines falls back onto an unplanned wrapper. The knob should be removed, not tuned.

**What is now established about the two-executor approach, after four independent attempts to
rescue it:** per-step host cost (preallocation -- within noise), the redundant full-batch plan
(skipping it is worse), the scatters (removed, no change), and batch shape (refuted). The
structure itself is the problem.

**The mechanism, and why it motivates K1 specifically.** flashinfer's single
`BatchPrefillWithPagedKVCacheKernel` is already a persistent varlen kernel that schedules work
items across the WHOLE batch internally. Splitting that batch into two launches throws away that
global load balancing and adds a second kernel tail, and FA3's advantage on the one prefill row
does not cover the loss. The isolated 1.224x was measured against a different baseline (one
persistent wrapper on a synthetic batch), which is why it did not transfer.

**This is the strongest argument yet for the fused single kernel:** only a single-grid design
keeps **one launch and one scheduler** while still specialising the mainloop per row class. The
two-executor variant -- the goal's "also acceptable" fallback -- is now measured as a dead end at
serving scale, and the remaining kernel hope rests entirely on K1.

- [x] **B2b DONE: split is -9.1% at the decode-rich shape with scatters removed.** Batch-shape
      hypothesis retracted; scatters and plan-skipping eliminated as explanations.
- [x] **`SGLANG_FUSED_SKIP_BASE_PLAN` is harmful** -- remove rather than tune.
- [ ] **Remove the two-executor dispatch from the default path** once K1 lands, or keep it only
      behind a flag as the measured-negative baseline it now is.

#### K1 scoped from 0.7's headers (2026-10-09): the reduction was REDESIGNED, not relocated

Read `flashinfer/data/include/flashinfer/attention/scheduler.cuh` on the installed 0.7 wheel.
`HolisticPlanInfo<NUM_TASKS>` now carries, **per task**: `q_indptr`, `kv_indptr`,
**`partial_indptr`**, `q_len`, `kv_len`, `q_start`, `kv_start`, `kv_end`, `kv_head_idx`,
`work_indptr` (10 fields); and **shared**: `len_kv_chunk`, `partial_o`, `partial_lse`,
`merge_indptr`, `merge_o_indices`, `num_qo_len` (6 fields).

**The four group tables the kernel indexes are gone and replaced.** 0.3.1 exposed
`group_id` / `group_nchunks` / `group_row_begin` / `group_nrows` -- group counters the kernel's
just-in-time reduction walked. 0.7 instead exposes a per-task **`partial_indptr`** plus shared
**`merge_indptr` / `merge_o_indices`**, i.e. the conventional split-KV merge-list pattern. So the
JIT reduction must be **rewritten against a different scheme**, not re-pointed at moved fields.
This is the outcome flagged as possible before starting: *"if 0.7 removed those tables because
the reduction itself was redesigned, I'll report that rather than force a port against a model
that no longer exists."* It was redesigned.

**The arithmetic confirms the model exactly.** `ToVector()` length is `2 + NUM_TASKS*10 + 6`:
**NUM_TASKS=2 gives 28**, which is precisely the plan length 0.7's `BatchAttention` produced and
the kernel rejected. So 0.7 also dropped from **3 tasks to 2**. **CORRECTION (2026-10-09):** I earlier wrote that `NUM_SHARED_ARGS = 8` while only 6 shared
fields are pushed, calling it the same count-vs-reality inconsistency 0.3.1 had at 16-vs-14.
**That was wrong.** Enumerating `ToVector()` mechanically gives **8** shared pushes --
`num_blks_x`, `num_blks_y`, `len_kv_chunk`, `partial_o`, `partial_lse`, `merge_indptr`,
`merge_o_indices`, `num_qo_len` -- and 10 per task. I had counted only the 6 trailing fields and
missed `num_blks_x`/`num_blks_y`, which are shared too. So in 0.7 the arithmetic is **consistent**:
`NUM_SHARED_ARGS + NUM_TASKS*NUM_TASK_ARGS = 8 + 2*10 = 28`, exactly the observed plan length.
The lesson I drew from it ("never trust the constants") does not apply to 0.7, and the length
guard our patch adds using that formula is therefore correct here.

**Revised K1 content, in order:**

- [x] **K1a RESOLVED by measurement (2026-10-09): DROP the JIT reduction, use 0.7's scheduled
      merge.** Read the kernel's actual reduction call
      (`include/flashinfer/attention/persistent_fa3_fused.cuh:686-689`): it passes
      `partial_o`, `partial_lse`, `merge_indptr`, `merge_o_indices` -- **all of which 0.7 still
      has** -- *plus* `group_row_begin`, `group_nrows`, `group_kv_head`, `group_nchunks`,
      `group_counter`, which 0.7 dropped. Those five exist **only** for the in-kernel
      just-in-time rendezvous: an atomic `group_counter` against `group_nchunks` so the last
      chunk to finish performs the merge itself and no separate merge kernel is launched.
      0.7 instead schedules the merge as its own kernel -- and that kernel is visible in the B1b
      trace: **`PersistentVariableLengthMergeStatesKernel`, 1682 launches, 10.66 ms = 0.6% of
      kernel time.** So the entire cost that JIT reduction removes is **bounded at 0.6% of GPU
      time**, which is far below the +/-1.5-3% noise floor.
      **Therefore K1a is not a port of the group tables at all.** Port the fused kernel to write
      partials and let 0.7's scheduled merge reduce them, exactly as 0.7's own `BatchAttention`
      does. This removes the need to reconstruct any group table, removes the atomic rendezvous,
      and removes the recorded split-KV defect where `group_counter` can never reach
      `group_nchunks - 1` when causal masking leaves a chunk with no work item.
      **On the paper's technique:** just-in-time reduction is one of the two ideas in *Taming
      Dynamism on GPUs*. Measuring its ceiling at 0.6% here is the honest way to "think carefully
      about how it fits our story" rather than copying it -- on this workload it does not pay, and
      saying so is more useful than reimplementing it. If a later measurement shows the merge
      kernel growing (more split-KV chunks at longer context), it can be revisited with a known
      target to beat.
- [ ] **K1a2. (superseded detail) Map the group-counter reduction onto the merge list** -- kept
      only as the fallback if the scheduled merge turns out to cost materially more than 0.6%
      once the fused kernel is actually producing partials. The kernel already owns `partial_o`, `partial_lse` and a
      `merge_indptr` path, so the target representation is partly familiar; what changes is how a
      chunk learns it is the last writer for a row. The 0.3.1 scheme used a `group_counter[gid]`
      atomic against `group_nchunks[gid]`; 0.7's merge list makes that a scheduled merge instead
      of a race, which should REMOVE the "counter can never complete when causal masking leaves a
      chunk with no work item" defect recorded earlier in this doc.
- [ ] **K1b. Rework the task layout from 3/4 tasks to 0.7's 2**, including what the
      `UnifiedWgmmaTileScheduler` 4th task was carrying.
- [ ] **K1c. Resolve the variant types** (`DefaultAttentionSm90` no longer a template,
      `StandardAttention` ambiguous) and the `variant_helper.cuh` vs `hopper/variant_helper.cuh`
      redefinition collision when both are on the include path.
- [ ] **K1d. Re-establish the fp32-reference PASS and a real timing on 0.7** before any of A/B is
      measured. The 1.30x figure is 0.3.1-only and must not be carried forward.

#### INCIDENT (2026-10-09): both fixes that made the fused kernel PASS were pod-only and were LOST

Starting K1-0 surfaced this. The two fixes recorded in this doc as making the fused kernel PASS
the causal fp32 reference at every `sm_split` -- `MaskMode::kNone` -> `kCausal` in `PKT`, and the
`cluster_tile_q` guard under `FUSED_GRID` -- **were applied directly to the pod's copy of
flashinfer and never committed**. The local checkout `/opt/tiger/flashinfer` at `b0563ae` still
had `MaskMode::kNone` (0 occurrences of `kCausal`) and the unguarded
`cluster_tile_q = gridDim.x * CTA_TILE_Q`, byte-identical to 0.7's line 242.

That pod's trial went to `done` and its tmpfs is gone, so **those fixes existed nowhere in
version control.** They are recoverable only because this doc recorded them verbatim, including
the exact replacement expression. Restored and committed as `flashinfer@c496afd` before the port
begins, so the 0.7 rebase starts from a correct kernel instead of silently re-deriving both bugs.

Consequences worth stating:

* **The "fused kernel PASSES 5/5, 1.30x at sm_split 112" result was produced by a tree that no
  longer existed anywhere.** It was already unusable as evidence (flashinfer 0.3.1, 20-arg
  extension); this makes it unreproducible for a second, independent reason.
* **Any other pod-only edit is also gone.** The doc is the only inventory. Cross-checking it is
  part of K1-0: every code change this doc claims must be verified present in git before being
  relied on.
* **Process fix:** edits made on the pod have to be committed locally the same session. The
  push-patch-apply loop used for sglang does this correctly; the flashinfer tree was being edited
  in place on the pod and so bypassed it.

- [x] 2026-10-09 **both lost fixes recovered and committed** (`flashinfer@c496afd`).
- [ ] **K1-0b. Audit this doc's other flashinfer claims against git** before porting -- the two
      recovered fixes prove the doc and the repo had diverged.

#### K1 SCOPE MEASURED (2026-10-09): ~3900 header lines, and scheduler.cuh is the hard part

Measured against the pristine base `c51aef7` (flashinfer **0.3.1**, the commit before our fused
work began), local checkout `/opt/tiger/flashinfer` at `b0563ae`:

| header | lines changed | nature |
|---|---|---|
| `attention/persistent_fa3_fused.cuh` | **+948** | new -- the fused kernel itself |
| `attention/scheduler.cuh` | **+439** | **modified -- and 0.7 redesigned this file** |
| `attention/persistent.cuh` | **+357** | modified (FUSED_GRID, `cluster_tile_q`, mask mode) |
| `attention/unified_wgmma_scheduler.cuh` | +281 | new |
| `attention/jit_merge.cuh` | +77 | new |
| `attention/persistent_wgmma.cuh` | +70 | new |
| `attention/prefill.cuh` | +65 | modified |
| `attention/hopper/variants.cuh` | +10 | modified |
| `attention/persistent_template.cuh` | +7 | modified |
| **total, headers only** | **3885 insertions / 124 deletions across 18 files** | |

Plus ~3700 lines of tests/harness, which do not need porting.

**Checked, and it corrects nothing -- the earlier finding stands.** The group tables
(`group_id_offset`, `group_nchunks_offset`, `group_row_begin_offset`, `group_nrows_offset`) are
present **20 times in the pristine 0.3.1 base and 20 times at HEAD** -- i.e. they were
**upstream's**, not ours, so 0.7 genuinely removed them and the JIT-reduction machinery was
flashinfer's own. Our +439 lines in `scheduler.cuh` are separate work (the unified-wgmma plan
and 4-task support).

**Honest scope statement.** K1 is a **~3900-line header port onto a base whose central plan
struct was redesigned**, with `scheduler.cuh` the hard part: our 439 lines modify the very file
0.7 rewrote, and `HolisticPlanInfo` went from 3-4 tasks with group tables to 2 tasks with
`partial_indptr` + `merge_indptr`/`merge_o_indices`. This is **multi-session C++ work**, not an
afternoon's wiring, and I should not have implied otherwise when K1 was first scoped as "resolve
renames and a header collision".

**What K1a already removed from that total:** the JIT-reduction path (`jit_merge.cuh` +77, the
group-table plumbing inside `persistent_fa3_fused.cuh`, and the matching part of the
`scheduler.cuh` delta) is **dropped**, since its measured ceiling is 0.6% of GPU time. So the
port targets 0.7's scheduled merge and the group tables are not reconstructed at all.

**Order of work for K1, revised:**

- [ ] **K1-0. Push the fused sources to the pod** -- the restore script omitted the `fi` tree, so
      the build inputs are absent. Only our files are needed (~88 KB for the ext plus the fused
      header), since everything else compiles against the 0.7 wheel's bundled headers.
- [ ] **K1b. Rebase `persistent.cuh` (+357) and `prefill.cuh` (+65) onto 0.7's copies.** Mostly
      mechanical: FUSED_GRID, the `cluster_tile_q` guard, the mask-mode fix.
- [ ] **K1c. Re-apply `scheduler.cuh` (+439) onto 0.7's redesigned plan** -- 2 tasks instead of
      3/4, `partial_indptr` instead of group tables. The real work.
- [ ] **K1d. Port `persistent_fa3_fused.cuh` (+948)** to the 2-task layout and the scheduled
      merge, dropping the rendezvous.
- [ ] **K1e. Re-establish the fp32-reference PASS and a timing on 0.7** before any A/B.

#### K1b DONE / K1c MEASURED (2026-10-09)

**K1b: `persistent.cuh` ported** -- `flashinfer@ed1be8a`, as an include **overlay**
(`port07/include/flashinfer/attention/`) in front of the wheel's bundled headers, so 0.7 supplies
every file except the ones we override. Mixing the 0.3.1 checkout and the 0.7 wheel on one
include path is what silently reintroduces the plan-layout mismatch, hence the overlay.
Our 16-hunk diff applies **11**; the **5 rejects are all JIT-reduction machinery**
(`MergeGroupJIT`, the `s_merge_flag` handshake, the `group_id != nullptr` rendezvous, the
`__threadfence` dance) and are **dropped by design** per K1a. Verified: `FUSED_GRID` role
indexing present, the recovered `cluster_tile_q` guard at line 290, and **zero** residual
`MergeGroupJIT`/`s_merge_flag`/`group_counter` references.

**K1c: `scheduler.cuh` measured -- 15 of 22 hunks apply, 7 reject.** Verified invariants after
the fuzzy apply (fuzz 2-3 on a redesigned file can land a hunk in the wrong place, so this was
checked rather than assumed): **no group tables re-added** (0 occurrences), `HolisticPlanInfo<2>`
still used by 0.7's `BatchAttention`, and the length guard our patch adds is correct for 0.7 (see
the correction above).

**The 7 rejects are the real K1c work, and they are NOT droppable.** They sit in
`HolisticPlanInfo` (3 hunks) and `TwoStageHolisticPlan` (4 hunks), and they are what adds the
**4-task plan and the unified-wgmma fields**. That matters because the 4th task *is* the fused
design: it is the class whose work items FA3's mainloop pulls through
`UnifiedWgmmaTileScheduler`, i.e. FA3's work expressed as a task inside the same plan as FA2's.
0.7's `BatchAttention` only ever builds a 2-task plan, so the fused kernel needs **our own
4-task plan builder** on top of 0.7's struct definitions -- which is precisely what our +439
lines were on 0.3.1.

So K1c decomposes as:

- [x] **K1c1 DONE, and the design changed: a PRIVATE plan struct, not an extension of
      0.7's.** `flashinfer@79a79a7` adds `port07/include/flashinfer/attention/fused_plan.cuh`
      defining `FusedPlanInfo<NUM_TASKS>` -- 0.7's fields plus `batch_idx_offset` per task and
      six shared fields (cluster cursor and counts for cross-role backfill, `max_splits_fa3`, the
      compact FA3 request list, `num_fa3_clusters` kept last because the harness reads the chosen
      split as `plan_info[-1]`). **11 per task / 14 shared, verified self-consistent by
      enumeration** (pushes == reads == constants), **58 entries at `NUM_TASKS=4`**.
      **`scheduler.cuh` is no longer overlaid at all.**
      *Why the design changed:* extending `HolisticPlanInfo` in place makes it 36 entries at
      `NUM_TASKS=2` against stock 0.7's 28, and `BatchAttention._plan_info` is produced by
      **flashinfer's own JIT-compiled module**, not ours -- so the two sides would disagree and
      the kernel would reject every plan, the same failure class as the original
      `28 entries, expected 56 or 68`. On the 0.3.1 *source tree* this could not happen, because
      flashinfer's JIT compiled `BatchAttention` from the same modified headers. With the 0.7
      *wheel* that coupling is gone. Chosen over the alternative (inject the overlay into
      flashinfer's JIT so `BatchAttention` sees the extended struct), which is less code but
      leaves us coupled to their internal build on every upgrade -- and releases have already
      broken this work twice (plan-struct redesign, `vector_sparse` helper removal).
      *Consequence:* 0.7's `HolisticPlanInfo`, `TwoStageHolisticPlan` and `BatchAttention` are
      untouched, so **K1c3 is satisfied by construction** -- there is no overlay to perturb them.
- [ ] **K1c2. Port the 4-task PLANNER** (our `TwoStageHolisticPlan` additions, 4 rejected hunks)
      into `fused_plan.cuh` as a private function populating `FusedPlanInfo<4>`, including the
      FA3 task's work items and the cluster cursor. ~400 lines; the largest single remaining
      piece of K1c.
- [x] **K1c3 satisfied by construction** -- nothing overlays 0.7's 2-task path.

**Caveat on everything in `port07/` so far: NOTHING HAS BEEN COMPILED.** The checks are
structural (hunk accounting, invariant greps, push/read enumeration), not a build. `persistent.cuh`
and `fused_plan.cuh` must both go through nvcc before any of this is called ported in a stronger
sense than "the edits are in place".

**Also recorded: fuzzy patching is unsafe on these files.** Applying our `scheduler.cuh` diff
with `patch --fuzz=3` produced **silently broken code** -- six `vec.push_back` statements landed
*inside* `FromVector`'s error-message expression, and the file was left with 11 per-task fields in
the struct and `ToVector` against `NUM_TASK_ARGS = 10` and 10 reads in `FromVector`, so a
round-trip would have misread every task. Found by enumerating pushes against reads, not by
reading the diff. Hand-port these files; do not trust fuzz.

- [~] 2026-10-09 **K1b RETRACTED: `persistent.cuh` is NOT ported.** The first nvcc parse
      (`port07/tests/nvcc_check.sh`, merged 0.7 + overlay include tree, sm_90a, C++20) gives
      **7 errors**:
      ```
      (220) identifier "kPersistentMainloopBarrier" is undefined
      (300) identifier "tid_base" is undefined
      (327) identifier "cluster_override" is undefined
      (327) identifier "kNoClusterOverride" is undefined
      (414) identifier "thr_local_kv_offset" is undefined
      (591) identifier "kPersistentMainloopBarrier" is undefined
      (861) a template declaration is not allowed here
      ```
      Two distinct problems. **(a)** Five identifiers our 0.3.1-based file uses that 0.7 does not
      provide -- they were defined in parts of 0.3.1 that 0.7 rewrote, so each needs a 0.7
      equivalent found or written. That is the genuine porting work and it was invisible to
      hunk accounting. **(b)** `(861) a template declaration is not allowed here` is a
      **structural** error: the fuzzy apply misplaced code, exactly as it did in `scheduler.cuh`.
      **My error, and worth naming:** I had already recorded that fuzzy patching is unsafe on
      these files after it put `push_back` statements inside an error-message expression in
      `scheduler.cuh` -- and then accepted the 11/16 fuzzy result for `persistent.cuh` because
      invariant greps (`FUSED_GRID` present, `cluster_tile_q` guarded, no `MergeGroupJIT`) all
      passed. Greps cannot see misplaced code. `persistent.cuh` must be **hand-ported** from
      pristine 0.7, like `scheduler.cuh` was.
      A third, smaller instance of the same blindness: the port carried a dangling
      `#include "jit_merge.cuh"` for a header K1a dropped. Greps for every symbol in that header
      returned 0; only the compile found the include.
#### K1b2 progress: nvcc errors 7 -> 2, both remaining ones diagnosed (2026-10-09)

`port07/tests/nvcc_check.sh` (merged 0.7 + overlay include tree, sm_90a, C++20) on each step:

| step | commit | errors |
|---|---|---|
| fuzzy-applied baseline | -- | **7** |
| barrier ids carved into `fused_barriers.cuh` | `flashinfer@34e5cf0` | **5** |
| `cluster_override` / `tid_base` / `kNoClusterOverride` hand-applied to `Run` | `flashinfer@933a655` | **2** |

Both remaining errors are now understood exactly, and neither needs a design decision:

1. **`thr_local_kv_offset` undefined (overlay line 433).** 0.7 **split this into two arrays** --
   `thr_local_kv_offset_k` and `thr_local_kv_offset_v` (0.7 `persistent.cuh:269,271`) -- and
   passes them separately to `prefetch_offest` / `page_produce_kv` at its lines 328/331. Our code
   (from the FA2 K/V smem-ring work, `f54745e` / `3e41c2d`) still uses one shared array. **Fix:**
   at each use site pass the `_k` variant on the K path and `_v` on the V path. Mechanical, but
   it must follow the buffer rather than be search-replaced, since the `issue` lambda handles both.
2. **`a template declaration is not allowed here` (overlay line 880).** The `KVStagedTraits`
   template, also from the KV-ring commits, was placed by the fuzzy apply **inside a function
   body**, where C++ forbids template declarations. **Fix:** move `KVStagedTraits` and its
   `FLASHINFER_FA2_KV_STAGES_*` / `FLASHINFER_FA2_NUM_MMA_KV_*` macros to namespace scope. This
   is the second structural corruption fuzzy patching introduced (the first put `push_back`
   statements inside an error-message expression in `scheduler.cuh`).

- [x] **K1b2 DONE (2026-10-09): `persistent.cuh` COMPILES against 0.7.** `NVCC_PARSE_OK` from
      `port07/tests/nvcc_check.sh` (merged 0.7 + overlay include tree, sm_90a, `-std=c++20`,
      `--expt-relaxed-constexpr`). Error progression, one commit per fix:
      **7 -> 5 -> 2 -> 0**.

      | fix | commit | errors after |
      |---|---|---|
      | barrier ids carved into `fused_barriers.cuh` (avoids overlaying 0.7's 4871-line `prefill.cuh`) | `flashinfer@34e5cf0` | 5 |
      | `kNoClusterOverride` + `cluster_override` / `tid_base` parameters hand-applied to `Run` | `flashinfer@933a655` | 2 |
      | K/V offset arrays separated, each prefetched with its own strides | `flashinfer@0d34c00` | 1 |
      | `KVStagedTraits` + macros moved to namespace scope | `flashinfer@2c79439` | **0** |

      **A latent bug in OUR code, surfaced by the port.** 0.7 split `thr_local_kv_offset` into
      `_k` and `_v`. Porting our KV-ring lambda onto that showed it called `prefetch_offest`
      **once with K's strides** (`k_stride_page/h/n`) and reused the result for the **V**
      `page_produce_kv`. That is only correct while V's strides equal K's -- and is plausibly why
      0.7 split the arrays. Fixed by issuing two prefetches. This changes what addresses the V
      loads read whenever the strides differ, so it is a correctness fix, not a rename.

      **Second structural corruption from fuzzy patching.** The `FLASHINFER_FA2_KV_STAGES_*`
      macros and `KVStagedTraits` had been placed **inside** the following struct template, where
      C++ forbids template declarations. nvcc said only *"a template declaration is not allowed
      here"*, naming neither the template nor the enclosing scope. With the `push_back`
      statements it put inside an error-message expression in `scheduler.cuh`, that is two
      corruptions, **both invisible to symbol greps and findable only by compiling**. The
      remaining files are hand-ported.

**Port state after K1b2:**

| file | status |
|---|---|
| `attention/fused_plan.cuh` | ported; host-compiled; round-trip + enumeration + merge-layout tests PASS |
| `attention/fused_barriers.cuh` | added (barrier ids, avoids overlaying `prefill.cuh`) |
| `attention/persistent.cuh` | **ported and NVCC_PARSE_OK** |
| `attention/scheduler.cuh` | not overlaid, by design -- 0.7's used unmodified |
| `attention/jit_merge.cuh` | dropped (K1a: 0.6% ceiling) |
| planner CUDA body | host halves done and tested; CUDA side unwritten |
| `attention/persistent_fa3_fused.cuh` | **+948, unported** -- the kernel itself |
| `attention/persistent_wgmma.cuh`, `unified_wgmma_scheduler.cuh` | +70 / +281, unported |

- [~] **K1d IN PROGRESS. The three new fused headers are in the overlay**
      (`flashinfer@19fc1f9`): `persistent_fa3_fused.cuh` (948), `persistent_wgmma.cuh` (70),
      `unified_wgmma_scheduler.cuh` (281). They have **no 0.7 counterpart**, so they copy in
      without a merge, and the rework was far smaller than the line counts suggested: **one**
      `HolisticPlanInfo` mention (a comment, retargeted to our private `FusedPlanInfo`) and
      **one** argument block, where the five group-table arguments to the FA3 epilogue were
      removed per K1a.

- [~] **K1d2 (2026-10-09, 3 of 4 landed, `flashinfer@0b8cc6f`+`4ef2407`). Port `hopper/prefill_sm90.cuh` (+32 on a 622-line 0.7 base; ours is 705).** This
      is now the blocking file -- dropping the group-table arguments above means the epilogue
      signature here must match the shorter list. Four distinct changes, in order of importance:

      1. **Extract the kernel body into a `__device__` function `fa3_prefill_body`.** This is
         the keystone of the entire fusion: it is what lets the fused kernel invoke FA3's
         mainloop on one range of SMs while the FA2-lineage persistent mainloop runs on the rest.
         The `__global__` entry point stays unchanged for every existing caller, so upstream
         behaviour is untouched.
      2. **The `fa3_has_split_kv<S>` trait**, which detects whether a scheduler's `WorkTileInfo`
         carries `group_id` -- true for our `UnifiedWgmmaTileScheduler`, false for upstream's.
         Needed because `block_coord` is destructured at a fixed arity of 8 by the body, so the
         unified reduction's per-item fields cannot ride along it. Note this probes the
         SCHEDULER's own `group_id` field, which we keep in
         `unified_wgmma_scheduler.cuh` -- it is unrelated to the PLAN's `group_id` that K1a
         dropped, and conflating the two would wrongly suggest this trait can go.
      3. **`FLASHINFER_SHARED_HAS_MEMBER_PROBES`** guard, so `prefill.cuh` and
         `hopper/prefill_sm90.cuh` can both define the member probes when the fused kernel
         includes both in one translation unit.
      4. **Q mbarrier init for packed GQA** (`barrier_Q.init(NumThreadsPerWarp)`): the
         packed-GQA loader is one warp issuing its own `cp.async` completed with
         `cp.async.mbarrier.arrive`, which tracks only the issuing thread's copies, so all 32
         lanes must arrive for themselves. Arming for 32 is what lets Q complete WITHOUT a
         `cp.async` group drain -- `commit_group` otherwise sweeps the producer's in-flight KV
         copies into Q's group, and committed groups retire in order, so a drain waits on the
         whole KV pipeline. **Arriving from 32 lanes on a barrier armed for 1 hung for 1500 s**,
         which is the symptom to recognise if this is mis-ported. **STILL OPEN** -- deferred to
         its own commit because it sits behind `FLASHINFER_FUSED_PACKED_Q` and has that
         distinctive hang, worth isolating from the other three.

      Changes 1-3 landed. Change 1 first landed broken in a way worth recording: the probe
      guard and the `fa3_has_split_kv` trait were inserted **inside the `__global__`'s template
      parameter list**, so the trait's `template<>` became a second clause on one declaration
      and `fa3_prefill_body` lost its parameters entirely -- 25 errors, all pointing at line 61
      rather than at the insertion. Fixed by hoisting the clause-independent text above and
      re-attaching the orphaned clause to the body.

- [x] **K1d3 (2026-10-09, `flashinfer@4ef2407`). FA2+FA3 co-inclusion in one TU is legal
      against 0.7.** The fused kernel is the **first** thing in 0.7 that includes both trees.
      Upstream's POD attention (`pod.cuh`), which looks like the precedent, is **FA2-only** --
      it includes `prefill.cuh` and `variants.cuh` and no Hopper header. 0.3.1 never hit this
      because its upstream carried `...Sm90` suffixes on the Hopper spellings; 0.7 dropped them
      because nothing needed them. Five collisions, in `namespace flashinfer`:

      | name | nature | resolution |
      |---|---|---|
      | `AttentionVariantBase` | different types in both trees | Hopper -> `...Sm90`, alias behind the flag |
      | `SharedStorageQKVO` | different types in both trees | Hopper -> `...Sm90`; our `persistent.cuh` wants the **FA2** one (its parameter list matches `prefill.cuh`, not `kernel_traits.cuh`) |
      | `DefaultAttention` / `DefaultFP8Attention` | different types | Hopper -> `...Sm90`, alias behind the flag |
      | `StandardAttention` / `LogitsSoftCap` / `StandardFP8Attention` | 0.7 declares `StandardAttention` in **`mla.cuh`** too, where the fused TU resolved it | Hopper -> `...Sm90`; nvcc calls this *ambiguous*, not redefined |
      | the four `DEFINE_HAS_MEMBER` probes | **identical** in both trees | `FLASHINFER_SHARED_HAS_MEMBER_PROBES`, which is 0.3.1 upstream's own mechanism |
      | `REGISTER_LOGITS_TRANSFORM` / `LOGITS_MASK` / `QUERY_TRANSFORM` / `KEY_TRANSFORM` | bodies differ **only in the name of a template type parameter** (`Params` vs `MainloopParams`), so whichever wins compiles both trees identically -- a warning, not a correctness hazard | guarded **per-macro on both sides**, not per-file: FA2 defines two macros Hopper lacks (`REGISTER_M_D_UPDATE`, `REGISTER_OUTPUT_TRANSFORM`) and a whole-file guard would strand them |

      A tree-wide rename is **not** available: every stock FA3 module is generated with
      `variant_name = "DefaultAttention<...>"`, so it would break all of them. Hence the
      aliases, dropped only by `-DFLASHINFER_SM90_SUPPRESS_UNSUFFIXED_NAMES=1`, which must be a
      TU-wide **compile flag** rather than a `#define` in `persistent_fa3_fused.cuh` --
      whichever header reaches `hopper/variant_helper.cuh` first wins, and in our own test TU
      that is `fused_plan.cuh`. Without the flag the build fails loudly on the aliases; it can
      never silently bind to the FA2 type of the same name. Renaming a struct also renames its
      **constructors**: leaving them behind turns each into a deduction guide.

- [x] **K1e-a (2026-10-09, `flashinfer@43d1f64`). The header-parse TUs were a near-empty gate.**
      A template is type-checked where it is **instantiated**, and nothing in either parse TU
      instantiates the fused kernel: `cuobjdump -elf` reports **zero device kernels** in a
      687 KB object, and the only fused symbol is the `kFusedQueueBarrier` constant. So
      `NVCC_FUSED_OK` says the headers parse together and nothing about the kernel body --
      including the reading that `fa3_upstream.cuh` is clean against 0.7, which it does not
      support. `fused_fa3_persistent_ext.cu` (the TU the loader actually builds) is now the
      third step of `nvcc_check.sh`, and it reports the device-kernel count rather than OK.
      Two 0.7 config-discovery bugs had to be fixed to get there (`sglang-up@56cad32fa`,
      `+ the follow-up`): 0.7's `BatchAttention.plan()` dropped the `batch_size` positional, so
      every later argument shifted by one and surfaced only as *"plan() got multiple values for
      argument 'causal'"*; and the generated-config path gained an **arch segment** while
      sglang **redirects the cache base at import time**, so the old glob missed on both counts
      -- now asked of `flashinfer.jit.env.FLASHINFER_GEN_SRC_DIR`, with sglang imported first
      for the same reason the loader gets it right.

- [~] **K1e. IN PROGRESS (2026-10-09, `flashinfer@0bc00af`). The kernel body now type-checks;
      4 real 0.7 API deltas remain.** The TU is ported onto 0.7's plan format: `FusedPlanInfo`
      replaces `HolisticPlanInfo<3>/<4>`, the six group-table params are gone with the
      JIT-reduction rendezvous (and with them the `FLASHINFER_NO_JIT_REDUCE` bisect flag), and
      `cluster_cursor` / `num_pers_clusters` moved out of `PersistentParams` into a
      `FusedQueueParams` argument threaded through the dispatcher, the three `__global__`s and
      the seven `fused_drain_queue` call sites. That last move matters for upstreaming:
      `PersistentParams` is generated from upstream's
      `csrc/batch_attention_customize_config.jinja`, so keeping them there meant patching a
      template **inside the installed wheel** -- unreproducible on a fresh pod, and it would
      make every stock `BatchAttention` build grow two fields it never uses. The 0.3.1 tree did
      patch that jinja; 0.7 does not need to.

      Remaining (18 errors, 4 groups), all now **inside** the kernel rather than include plumbing:

      | # | error | cause |
      |---|---|---|
      | 1 | `persistent_fa3_fused.cuh:672/675/693` *"too many initializer values"* | 0.7's `BatchPrefillPagedParams` changed field order/count; needs field-by-field assignment instead of positional aggregate init |
      | 2 | `persistent.cuh:422` `FusedWarpgroupTraits` has no `NUM_KV_STAGES` | 0.7's `KernelTraits` renamed it |
      | 3 | `persistent.cuh:618` `threadblock_sync_mdo_states` no match | signature changed in 0.7 |
      | 4 | `persistent_fa3_fused.cuh:257-259` *"too many arguments"* | `Runner::Run`'s `s_merge` was the JIT rendezvous broadcast, which went with the group tables |

      **Three of the four landed (`flashinfer@36a89de`), verified by the instantiating TU on
      the pod: 18 -> 3 errors.**

      - `NUM_KV_STAGES` is **our** name, not a 0.7 rename. Our staged wrapper defines it as `S`;
        the runners instantiated directly over flashinfer's FA2 `KernelTraits` found it in 0.3.1
        only because that struct carried `NUM_KV_STAGES = 1` as a base default, which 0.7
        renamed to `NUM_STAGES`. A `fa2_kv_stages` probe reads whichever member exists, so the
        staged wrapper and upstream never have to agree on a spelling.
      - `s_merge` dropped at the three `Run` call sites. The 16 B per-warpgroup tail slot keeps
        its stride on purpose -- shrinking it is a shared-memory layout change and the fp32
        reference is not passing yet to catch one.
      - `threadblock_sync_mdo_states`: the defaulted `barrier_id` is ours and had to be carried
        onto 0.7, which uses plain `__syncthreads()`. That matters because `__syncthreads()`
        compiles to `barrier.sync 0`, which FA3 also uses as `NamedBarriers::kQueryEmpty`, so a
        CTA backfilling from FA3 into this mainloop rendezvouses on a barrier FA3 already drove.

- [x] **K1f (2026-10-09, `flashinfer@1208ca4`). The fused kernel COMPILES against 0.7 --
      `NVCC_EXT_OK`, 10 device kernels** (against **0** for the header-parse TUs, so the body is
      genuinely type-checked this time). 0.7 inserted three fields into the sparse mainloop's
      `Arguments`.
      After `window_left` come `k_page_stride`, `v_page_stride`, `page_size`, so our positional
      aggregate init shifted `additional_params` onto an `int64_t`. FA3's paged params already
      carry all three under the same names. Two header deltas to carry onto 0.7, and the call
      sites move to **designated initializers** so the next inserted field fails loudly instead
      of shifting silently -- the same failure mode as the loader's `plan()` call earlier today.

      | header | delta | content |
      |---|---|---|
      | `hopper/sparse_mainloop.cuh` | +57 | `packed_Q_ptr`, `group_size`, and the packed-GQA load path that reads them (Q rows are `(token, head-in-group)` pairs, which the Q tensormap cannot express) |
      | `hopper/epilogue.cuh` | +256 | the split-KV `store_partial` path and **six** of our thirteen added fields: `partial_o`, `partial_lse`, `merge_indptr`, `merge_o_indices`, `num_kv_heads`, `gqa_group_size`. The other seven are the group tables and go with K1a's dropped rendezvous |

      Both call sites now use **designated initialisers**. Every field 0.7's insertion shifted
      onto happened to be assignable, so the positional form did not fail *where* the change
      was -- it failed three fields later as *"no suitable conversion from AdditionalParams to
      int64_t"*. Same failure mode as the loader's `plan()` call earlier today; naming the
      fields makes the next insertion a missing-member error.

- [x] **K1g (2026-10-09, `flashinfer@b883589`). `store_partial` ported and GENUINELY compiled;
      the unified-wgmma TU builds (`UNIFIED_OK`, 10 device kernels), and the gate has a **null
      case** proving it covers the function.**

      Root cause of the 6 errors was **mine, from earlier this session**: the block at
      `fused_fa3_persistent_ext.cu:277` is not a debug print, it is the `UnifiedWgmmaPlanPtrs`
      aggregate initialiser. Removing `P(t3.group_id_offset)` deleted a live field initialiser
      and shifted every argument after it by one, which is why the errors appeared three fields
      later as `float*` vs `IdT*` mismatches instead of at the edit. **Fourth** positional-init
      shift in this port; the first three are now designated-initialised.

      The fix is not to restore it -- `FusedPlanInfo` has no `group_id_offset`, by K1a. The
      plan's per-item `group_id` went with the rendezvous, and its only surviving role is the
      boolean *"does this row need a merge"*, which is derivable: the scheduler computes
      `group_id = (num_kv_chunks > 1) ? 0 : -1` and the field is dropped from
      `UnifiedWgmmaPlanPtrs` / `Arguments` / `Params`. Nothing read the group **index** any more
      (`store_partial` no longer takes it; `is_split_kv` and the call-site gate only test
      `>= 0`). `WorkTileInfo` keeps the field so `fa3_has_split_kv` still has something to probe.

      **Why this green is worth more than the previous three.** `store_partial` sits behind an
      `if constexpr` that is false for the default scheduler, so the default TU's `NVCC_EXT_OK`
      never covered it. Injecting a type error inside `store_partial` makes the unified compile
      fail with exactly that error -- so it really is instantiated. Both the unified step and
      the null case are now permanent steps of `nvcc_check.sh`, which reports
      `NULL_CASE_FAILED` if the gate ever stops covering the function.

- [x] **K1h (2026-10-09, `sglang-up@f019410ff`, verified `f8ad55bd4`). The fused kernel is a
      LOADABLE MODULE against 0.7 -- `torch.ops.fused_fa3.fused_run` is registered.** The fix
      was a **merged include tree**, not an `-I` ordering. nvcc's instantiation trace confirms
      the real kernel compiled, not a degenerate specialisation:
      `FusedFA3PersistentSymmetricKernel<..., UnifiedWgmmaTileScheduler<IdT>, R128, R64, R16>`,
      `fa3_prefill_body` with that scheduler, `kUnifiedWgmma=true`, `SharedStorageQKVOSm90`
      (the co-inclusion rename in effect) and `FusedQueueParams` in the kernel signature. No
      errors; only cutlass deprecating `cute::copy_if`'s predicate form. The loader also now
      passes `-DFLASHINFER_SM90_SUPPRESS_UNSUFFIXED_NAMES=1` and
      `-DFLASHINFER_FUSED_UNIFIED_WGMMA=1`, the latter being the only configuration that
      instantiates `store_partial`.

- [x] **K1c2 CLOSED (2026-10-09, `flashinfer@6aa16fc`). The plan builder runs on GPU and
      satisfies the merge contract: 19/19 PASS.** First thing this session that executed.

      Command: `port07/tests/test_plan_build.cu`, vibecuda H200, mixed batch (2 long FA3 rows at
      kv=8192/6144 + 3 decode rows), `sm_split=16`, 132 SMs:

      ```
      [fused plan] items=0/0/21/1708 partial_rows=31232 int_bytes=99760/67108864
      [fused plan] float_bytes=112809984/536870912 (o=111935488 lse=874496)
      ALL PASS (0 failures)        -- 19 checks
      (28 groups, 1708 items, 1708 with base)
      ```

      `partial_rows` fell 218624 -> 31232, exactly the 7x kv-head over-count removed, and the
      slab is 112 MB inside a 512 MB budget instead of 784 MB over it. The group/item counts
      match by hand: 16 + 12 = 28 q-tile groups, and 16*10*7 + 12*7*7 = 1708 items.

      `BuildFusedMergeLayout` is rewritten to `BlockBatchReductionPersistent::Run`'s contract,
      fixing all four disagreements (per-packed-row `merge_indptr` with a terminator; the list
      **compacted** over split rows with `o_indices` mapping back, since an uncompacted list
      makes the merge zero unsplit rows over their direct write; `kv_head_idx` out of the
      grouping key; one shared base per group advancing by `rows*chunks`). New plan field
      `num_merge_rows`; shared entries 14 -> 15, so **59** at NUM_TASKS=4, with
      `num_fa3_clusters` still last.

      The gate checks the **contract equality itself**, which `run_host_checks.sh` never did --
      it only asserted monotonicity, counts and the round trip, which is exactly why all three
      layout bugs passed it. One check in the gate's own first version was also wrong (it folded
      a per-request chunk count into one number, while row 0 has 10 chunks and row 1 has 7); it
      now groups task-3 items by their own `(batch_idx, q_start)` and requires one base per group.

- [x] **K1j ANSWERED (2026-10-09). No, and it structurally cannot in 0.7's own form.**
      Verified by reading the launch site rather than grepping for a call: `persistent.cuh:929`
      launches **one cooperative kernel**

      ```
      PersistentKernelTemplate<BlockBatchPagedAttentionPersistent<KTraits1, Params>,
                               BlockBatchPagedAttentionPersistent<KTraits2, Params>,
                               BlockBatchReductionPersistent<ReductionKTraits>>
      ...
      cudaLaunchCooperativeKernel((void*)kernel, nblks, nthrs, args, smem_size, stream)
      ```

      so in 0.7 the merge is a **third role inside the same grid**, and the **cooperative**
      launch is what gives it the grid-wide sync to run only after every partial is written. Our
      fused dispatcher launches `cudaLaunchKernel` / `cudaLaunchKernelExC` with the three FA2
      runners plus FA3 and **no reduction role**, non-cooperatively. So there is nothing to
      "add a call to" -- the two designs differ in grid structure.

      **Decision: launch the reduction as a SEPARATE kernel after the attention grid**, not as a
      fourth role. Reasons, in order:
      - A cooperative launch constrains the grid to what fits co-resident, which is exactly the
        quantity `sm_split` pre-assignment is manipulating. Making the fused kernel cooperative
        would couple the SM split to an occupancy constraint and is the opposite of the design.
      - K1a already priced this: flashinfer's scheduled merge is a separate pass at 1682
        launches / 10.66 ms in the B1b trace, and the in-kernel JIT rendezvous that avoided it
        was measured at a 0.6%-of-GPU-time ceiling -- below the +/-1.5-3% serving noise floor.
        So a separate pass is the already-measured choice, not a concession.
      - `BlockBatchReductionPersistent::Run` is a `__device__` member, so a standalone launch
        needs only a thin `__global__` wrapper plus `num_merge_rows` as `num_packed_qo_len`.

- [x] **K1k DONE (2026-10-09, `flashinfer@641ef1b`). The reduction is launched as a separate
      pass after the fused attention grid.** Device kernels on the unified TU **10 -> 15**, so
      the wrapper is genuinely instantiated rather than discarded by an `if constexpr` -- the
      failure mode that made three earlier green compiles mean less than they looked.

      **The grid shape was the trap, and it fails silently.** `Run()` derives

      ```
      worker_id   = blockIdx.y * num_warps + warp_idx
      num_workers = gridDim.x * gridDim.y * gridDim.z * num_warps
      ```

      so the stride counts every block while the id varies only in **Y**. Launched as the obvious
      `(N, 1, 1)`, every block computes the **same** `worker_id` and every row not congruent to
      it is never written -- the identical silent partial-coverage signature to fixture 2's
      511/512, with no error and no assert. Blocks therefore go in `gridDim.Y`, `x = z = 1`.

      The launcher no-ops when `num_merge_rows == 0` (the common decode-only case). Passing the
      batch's row count instead of the plan's compacted split-row count would make the merge
      visit unsplit rows, whose `num_index_sets` of 0 fills their output with **zeros** over the
      direct write -- the rows needing no merge would be exactly the ones it destroyed.

- [~] **K1l. THE FUSED KERNEL EXECUTES (2026-10-09, `flashinfer@255191b`). One fault left, not
      yet localized.** `drivers/fused_reference.py` (in-repo this time) builds both plans, calls
      `fused_run`, and compares against an fp32 **causal** reference. Running it found two bugs
      that **every compile-time gate had passed**:

      1. `FromVector` still demanded 58 entries while `ToVector` emits 59 --
         `NUM_SHARED_ARGS` was left at 14 when `num_merge_rows` was added. Caught on the
         driver's first run.
      2. **`len_kv_chunk = 0` for the FA2 classes was a device divide-by-zero.**
         `persistent.cuh` divides by it *unconditionally*
         (`kv_chunk_idx = kv_start / len_kv_chunk`, `ceil_div(..., len_kv_chunk)`), so 0 does not
         mean "unsplit" -- it means `num_kv_chunks` is garbage, the row takes the **split** path,
         and `store_partial` writes a 16 B vector at a wild offset. The FA2 classes now get a
         span covering any row they can see, so `ceil_div` gives exactly 1 and they take the
         direct-write path. **The plan gate had asserted `lkc[0..2] == 0`, so that check encoded
         the bug**; it now requires a non-zero covering span.

      Evidence the second fix landed, `compute-sanitizer` with `CUDA_LAUNCH_BLOCKING=1`:

      | | kernel / block | faulting address |
      |---|---|---|
      | before | `FusedFA3PersistentKernel`, block (17,0,0) | 2,617,249,280 B **before** the nearest allocation (the 64 MB int workspace) |
      | after | same kernel, block (19,0,0) | 1,266,688 B **after** the nearest allocation, which is **1 byte** in size |

      The address went from wildly negative to modestly positive -- a different computation, not
      the same fault surviving.

      **UPDATE (same day): the fault is GONE; the kernel runs end to end.** Three further bugs,
      each found by reading the consuming code against evidence:

      | bug | why it was wrong |
      |---|---|
      | `q_indptr` absolute over the batch | the FA2 roles get `q`/`final_o` already advanced by `pers_row_offset`, so the unsplit write `final_o + q_indptr * o_stride_n` counted FA3's rows twice -- 256 FA3 rows put the FA2 rows at 512/513 of a 258-row tensor |
      | `merge_o_indices` in PACKED rows | `Run`'s output is `[qo_len, num_kv_heads, gqa_group_size, head_dim]`, so `pos` indexes **tokens**; packed indices were `gqa` times too large |
      | `v_merged` = `pers_o + pers_row_offset` | the rows that split are **FA3's**, at the START of `o`, and `o_indices` are absolute token rows -- so the merge wrote past the end of the FA2 region. **This was the fault.** |

      The localization was a **bisect, not a guess**: `FLASHINFER_FUSED_NO_MERGE=1` showed the
      attention grid (FA3 + three FA2 classes + partial writes + the cross-role queue) is
      memory-clean and the reduction launch was the only faulting one -- which three sanitizer
      runs' address patterns could not settle. One hypothesis was **checked and rejected** before
      acting on it: `s_merged = nullptr` is in fact null-guarded at `persistent.cuh:848`.

      **Current state -- numeric, not a fault:**

      ```
      pers_plan: 59 entries, num_merge_rows=1024, num_fa3_clusters=16
      rows 0-4 rel 38.9 / 39.17 / 39.68 / 38.75 / 33.99
      worst rel err 39.68, bad rows 258/258   FAIL
      ```

      The merge now writes (rows are no longer zero as under the bisect) but values are ~39x the
      reference -- a magnitude that points at a missing softmax renormalisation, or at chunks
      accumulating into one slot instead of being combined. **To be established, not assumed.**

      Also: the FA2 decode rows (256, 257) are wrong too, and they **never touch the merge**.
      That is a second independent error, and it may be in the DRIVER: the comparison stitches
      `got[nfa3_rows:]` from `pers_o`, which assumes the FA2 roles write `pers_o` at absolute row
      indices while their `q_indptr` is now subset-relative. Check the driver's stitching before
      suspecting the kernel on those two rows.

      **RETRACTED (same day): "the FA2 role faults before the output write".** That rested on
      block 19's write trace never printing while its entry trace did. But a fault in **any**
      block aborts the kernel, so block 19 may simply never have reached its write before another
      block killed the launch -- absence of its print was never evidence the write was
      unreachable. The null case proved the trace was *compiled in*; it did not prove the traced
      block *ran to* its write. A second null case was needed and not built.

      **`MODE=fa2` under compute-sanitizer** (never run on this configuration before):

      ```
      Invalid __global__ write of size 16 bytes
      by thread (32,0,0) in block (84,0,0)
      Access to 0x7f092a0cf400 is out of bounds
      and is 4,537,344 bytes after the nearest allocation ... of size 1 byte
      ```

      So the write **is** reached, in an FA2-role block (84 of 132, past `sm_split=16`), and
      overruns by ~4.5 MB. Same 16-byte signature as mixed mode but a different block and
      distance -- and in `MODE=fa2` the reduction does not run at all (`num_merge_rows=0`), so
      there are **at least two distinct 16-byte overruns**. Carrying mixed mode's
      reduction attribution across configurations was the error.

      Not in doubt, from the entry trace: `n_clusters=70` matches by hand (1024 packed rows /
      tile128 = 8 tiles x 7 kv heads = 56, plus 2 decode rows x 7 = 14), so the plan's cluster
      accounting and the task-0/task-2 split are right.

      **RETRACTED (same day): "block 84 takes the SPLIT branch".** The split-branch trace is
      also silent for block 84, so the silence is the **trace's**, not the branch's. Two reasons
      the gate may never fire, neither excluded: the printfs were gated on `tid.x == 0` where
      `tid` is the **warpgroup-local** id, and the FA2 runners run at `tid_base` 128/256 so it is
      never 0 for them (consistent with the sanitizer naming thread (32,0,0)) -- changed to
      `(threadIdx.x & 127) == 0` and it still does not print; and `cpp_extension` caching, since
      the `[fa2 enter]` line only proves the TU rebuilt when **that** line was added.

      Nothing is asserted about which epilogue branch runs. What IS established:
      the fault is an `Invalid __global__ write of 16 bytes` from block (84,0,0) thread (32,0,0),
      4,537,344 B past a 1-byte allocation; `n_clusters=70` matches by hand so the plan's cluster
      accounting is right; and the reduction does not run here (`num_merge_rows=0`), so mixed
      mode's reduction attribution does not transfer.

      **Each epilogue trace needs its own null case** -- an unconditional print at the same site,
      confirmed to appear -- before its absence means anything. This is the third conclusion this
      session drawn from an instrument not shown to measure anything (after the header-parse TUs,
      the plan gate asserting `len_kv_chunk == 0`, and the unit audit that passed while the kernel
      faulted).

      **RESOLVED (same day, on sufficient evidence this time): the fault is in the FA2
      MAINLOOP, before the epilogue.** A print placed **before** the `num_kv_chunks` branch --
      its own null case -- does not appear for block 84, while `[fa2 enter]` from that block
      does. Both enabling conditions were checked, not assumed: the trace **is** compiled in
      (header edits force a rebuild, because the loader's merged include tree is named by a
      content stamp over the overlay, so any overlay edit changes the include path and thus
      torch's build config -- confirmed empirically when `[plan audit]` appeared after a
      `fused_plan.cuh` edit with no flag change); and the print is upstream of both branches, so
      its silence cannot be a branch-not-taken.

      So block 84 enters the drain, runs a cluster, and faults **before** the epilogue. The
      16-byte invalid write is inside the mainloop, not the output store. This restores the
      conclusion retracted above, but on different and sufficient evidence -- the retraction was
      correct when made, since the only support then was a branch-gated trace never shown to fire.

      **[x] FA2 PATH PASSES THE FP32 CAUSAL REFERENCE (2026-10-09, `flashinfer@cc56241`).**
      First correctness PASS of this port. Command:
      `MODE=<m> PYTHONPATH=<fork>/python SGLANG_FUSED_FI_ROOT=<fi>/port07 python drivers/fused_reference.py`

      | fixture | before | after |
      |---|---|---|
      | `fa2_2tile` q=256 kv=256 | rel 1.991, 224/256 bad | **PASS** rel 0.0021, 0/256 |
      | `fa2` q=256 kv=2048 (16 kv tiles) | **illegal access** | **PASS** rel 0.002126, 0/258 |
      | `mixed` | rel 39.68, 258/258 | FAIL rel 40.57, 258/258 |

      So all three FA2 tile classes, a row spanning 16 kv tiles, and bottom-right causal masking
      are correct against an fp32 reference, at a residual consistent with bf16 (this doc's
      recorded 0.0044 when correct). **The illegal access and the numeric error had ONE cause.**

      Cause: the plan set `q_indptr = request_token_base + q_start`, but the kernel applies the
      tile offset itself from `packed_qo_start` / `q_start`, so adding it to `q_indptr`
      double-counts it -- and the unsplit write is `final_o + q_indptr * o_stride_n`.

      How it was found, which is the transferable part: the trace was made to print from **every
      block** instead of one. All 56 work items ran (8 q tiles x 7 kv heads), so coverage was
      never the problem; what the trace showed was `q_indptr` taking 0,128,...,896 across the 8
      tiles of ONE request, where it should have been 0 throughout. That **predicted** the
      observed signal -- only tile 0 has `q_start == 0`, so only its 32 tokens (128 packed / gqa
      4) were correct, exactly the 32 of 256 rows that had been passing. Three conclusions had to
      be retracted earlier in this item, and all three came from reading an instrument's
      **silence**; what worked was a positive signal checked against arithmetic.

      **[x] FA2 is now correct in MIXED mode too (`flashinfer@ec02a20`).** The FA2 decode rows
      were wrong in mixed (rel 18.29 / 23.82) while identical rows passed in `MODE=fa2`. The
      difference is `pers_kv_page_offset`, which is 0 when nothing goes to FA3: the caller hands
      the FA2 roles a `kv_indices` pointer already advanced by the FA3 page total **and** the plan
      supplied an absolute `kv_indptr`, double-counting those pages. Same shape as the `q_indptr`
      bug -- when the caller rebases the pointer, the plan's offsets must be subset-relative.

      | run | FA2 rows 256/257 | FA3 rows | bad |
      |---|---|---|---|
      | `mixed` | **0.002069 / 0.002004 PASS** | wrong | 256/258 |
      | `mixed` + `NO_MERGE` | **PASS** | rel 1 (zero, expected) | 256/258 |
      | `fa3` | -- | 44.84 | 258/258 |

      With `NO_MERGE=1` the worst error over the whole batch is exactly **1.0**, which is what
      unreduced FA3 rows look like (`got ~ 0`) with the FA2 rows correct -- an independent
      confirmation that nothing on the FA2 side remains wrong.

- [ ] **K1m. The FA3 role is the entire remaining failure.** `MODE=fa3` = 44.84 over 258/258;
      `mixed` = 41.67 over exactly the 256 FA3 rows. This is a **different surface** from every
      fix so far: the FA2 bugs were all plan-to-kernel index spaces, while the FA3 role runs
      through `UnifiedWgmmaTileScheduler` and `fa3_prefill_body` with its own params built in the
      ext. Candidates, **not yet tested**:

      - the ext's FA3 params -- `fa3_kv_indices` is the **vector-sparse** array while the
        upstream-port `page_indices` must be **raw page ids**; feeding one where the other is
        read read far outside the cache before;
      - the split-KV path: `MODE=fa3` reports `num_merge_rows=1032`, so almost every FA3 row is
        split and therefore depends on `store_partial` + the merge being right, neither of which
        has ever been exercised correctly (K1g only established that `store_partial` compiles).

      **BISECT DONE (2026-10-09): the FA3 role faults even UNSPLIT.** `MODE=fa3_1chunk` (one FA3
      row at kv=128, so the plan's chunk rounds up to `TILE_KV` and `ceil_div(128,128)=1`) reports
      `num_merge_rows=0` -- no partials, no merge -- and hits an **illegal access**. So the FA3
      mainloop/epilogue is broken independently of split-KV, which removes `store_partial` and the
      merge from this fault's critical path. That is worth knowing before touching either: K1g
      only established that `store_partial` *compiles*, so it was the obvious suspect and is now
      excluded for the unsplit case.

      **The two FA3 configurations fail differently, so there are at least two defects:**

      | fixture | kv | merge rows | result |
      |---|---|---|---|
      | `fa3` | 2048 | 1032 | wrong numbers (44.84), **no fault** |
      | `fa3_1chunk` | 128 | 0 | **illegal access** |

      A fault in the *simpler* configuration and none in the more complex one is the opposite of
      the usual ordering, and a caution against assuming one cause. **Not assumed:** that these
      share a cause, or that either explains mixed mode's 41.67.

      **SANITIZED (2026-10-09): a NEGATIVE partial index -- a partial write on an unsplit row.**
      `MODE=fa3_1chunk`, `CUDA_LAUNCH_BLOCKING=1`, a new signature (4 bytes, not 16):

      ```
      Invalid __global__ write of size 4 bytes
      by thread (128,0,0) in block (8,0,0)
      Access to 0x7f603fffffd4 is out of bounds
      and is 44 bytes BEFORE the nearest allocation ... of size 1,073,741,824 bytes
      at void flashinfer::FusedFA3PersistentKernel
      ```

      Three facts pin it: the 1 GB allocation is the driver's **float** workspace, so the target
      is the partial buffers; a **4-byte** write is `partial_lse` (float), not `partial_o` (bf16,
      written 16 B at a time); and it lands **before** the base, so the index is **negative**.
      `-44/4 = -11`, which at `num_kv_heads=7` is `(-2)*7 + 3` -- a partial row index of −2 at kv
      head 3. **−1 is exactly what `BuildFusedMergeLayout` writes into `partial_indptr` for a row
      it marked UNSPLIT.** So a row the plan says needs no partial is reaching a partial write,
      and the offset arithmetic runs off the front of the buffer.

      Block 8 is below `sm_split=16`, so an FA3-role block -- consistent, since the FA3 epilogue
      gates `store_partial` on `work_tile_info.group_id >= 0`, and `group_id` is now **derived**
      in the scheduler as `(num_kv_chunks > 1) ? 0 : -1` from
      `ceil_div(kv_len, len_kv_chunk[0])`. For this fixture that is `ceil_div(128,128) = 1`, so
      the gate should be **closed**. One of those two inputs is therefore not what the plan
      wrote -- the same class as the five plan index-space bugs, and settled by printing what the
      FA3 scheduler actually read, from every block.

      **CAUSE FOUND AND FIXED (`flashinfer@c7fdd92`): the `WorkTileInfo` aggregate initialiser
      was misaligned.** The struct is `(..., partial_indptr, group_id, kv_chunk_idx,
      num_kv_chunks, ...)` but the positional initialiser listed `kv_chunk_idx` **before**
      `group_id`, so `group_id` silently received the **chunk index** -- 0 for the first chunk,
      which opens the `group_id >= 0` gate on **every** row. The FA3 epilogue then called
      `store_partial` for rows the plan had marked unsplit with `partial_indptr = -1`.

      The trace showed the contradiction directly instead of by inference: all 16 FA3 blocks
      reported `group_id=0 chunks=1 partial_indptr=-1`, three values that cannot hold together
      since `group_id` is derived as `(nchunks > 1) ? 0 : -1` from the same quantity printed as
      `chunks=1`. It also matches the sanitizer arithmetic exactly -- 4 bytes 44 B before the base
      is `(-2)*7 + 3` at `num_kv_heads=7`.

      Now **designated initialisers**, like the three other aggregates this port has had to fix
      for the same reason (sparse mainloop `Arguments`, epilogue `Arguments`,
      `UnifiedWgmmaPlanPtrs`). **Four positional aggregates misaligned in one port.**

      | fixture | before | after |
      |---|---|---|
      | `fa2` | PASS | **PASS** rel 0.002126, 0/258 |
      | `fa3_1chunk` | **illegal access** | no fault, rel 3.076e+06, 256/256 |
      | `fa3` | 44.84 | 45.1 |
      | `mixed` | 41.67 | 45.44, 256/258 (FA2 rows pass) |

- [ ] **K1n. The FA3 role produces garbage, and it is a DIFFERENT failure mode.** No fault
      anywhere now, and FA2 is correct, but `fa3_1chunk` -- one unsplit row, no partials, no
      merge, the simplest possible FA3 configuration -- is off by **3.076e+06**. That magnitude
      is not an indexing slip; it is unnormalised or uninitialised output, so it should be chased
      as such rather than as another index-space bug. Candidates: the softmax denominator never
      applied (`normalize_d` / `attention_updater` not reaching the epilogue), or `tOrO` read
      before the mainloop wrote it. **Not assumed:** that `MODE=fa3`'s 45.1 at kv=2048 is the
      same cause.

      **FA3 SCHEDULER INPUTS VERIFIED CORRECT AND COMPLETE (2026-10-10).** Post-fix gate trace
      on `fa3_1chunk`, from every block:

      ```
      gate lines: 56        (= 8 q tiles x 7 kv heads, the full task-3 item count)
      distinct blocks: 16   (= sm_split)
      all 56: group_id=-1 chunks=1 partial_indptr=-1 qo_len=1024 kv_len=128 chunk_idx=0
      ```

      Every field is right -- `group_id=-1` so the split gate is correctly **closed**, `chunks=1`
      and `partial_indptr=-1` agree the row is unsplit, `qo_len=1024` is the packed extent
      (256 tokens x gqa 4), `kv_len=128` the chunk, and `chunk_idx=0` is now in its own field
      rather than holding `group_id`'s −1. **And all 56 items execute.**

      That **retires** the reading floated from the pre-fix trace -- that only 16 items ran, one
      per block, leaving `o` uninitialised. It was an artifact of the same misalignment, which
      also corrupted `counter`/`ptr_begin`/`ptr_end`, the fields `is_valid()` tests. The caveat
      was flagged when the observation was recorded, and it held.

      **So the plan-to-kernel interface is now exonerated by measurement**, for the first time in
      this item: the scheduler feeds correct, complete work and the output is still off by
      3.076e+06. The defect is inside `fa3_prefill_body`'s mainloop/epilogue, with correct inputs,
      no split and no merge.

      **MAINLOOP DOES ITERATE -- that hypothesis is refuted (2026-10-10).** From every block on
      `fa3_1chunk`: 56 mma lines, all `num_kv_tiles=1 swa=[0,-1) qo_len=1024 kv_len=128`.
      `num_kv_tiles=1` is correct for kv=128 at TILE_KV=128, so the mainloop runs and `tOrO` is
      written. Excluded by measurement, not argument.

      **So, entirely by measurement:** the plan feeds the FA3 role correctly, the scheduler
      delivers all 56 items with every field right, the mainloop iterates the right KV tile
      count -- and the output is still off by 3.076e+06.

      **Prime suspect: a UNITS mismatch in the FA3 epilogue.** `qo_len` reaches it as **1024**,
      the PACKED extent (256 tokens x gqa 4). The scheduler presents packed deliberately -- its
      own comment records that FA3's mainloop validates the tile index against `qo_len`, so a
      token length there made every tile look out of range. But the epilogue writes `o`, laid out
      `[tokens, num_qo_heads, head_dim]` with heads as a separate dimension, and clamps with
      `row < qo_len - qo_tile_idx * CTA_Q`. At `qo_len=1024` that clamp never fires at the true
      256-token boundary, so tiles past token 256 write rows the tensor does not have -- and the
      gqa factor makes the row mapping wrong for **every** tile, not just the tail. A huge
      residual without a fault is consistent, since `o` is the first of several large allocations.

      **CONFIRMED (2026-10-10), and it is an UN-PORTED feature rather than a new bug.** From
      every block on `fa3_1chunk`:

      ```
      store lines: 56                 (8 q tiles x 7 kv heads)
      qo_tile=0 ... qo_len=1024 rows_left=1024
      qo_tile=1 ... qo_len=1024 rows_left=896
      qo_head in {0,4,8,12,16,20,24}  (= kv_head * gqa)
      ```

      The epilogue's row clamp is in **packed** rows: it writes rows up to 1023 into an `o` with
      only **256 token rows**, treating packed rows as tokens while indexing heads as a separate
      dimension. Every tile's mapping is wrong, not just the tail -- consistent with 3.076e+06
      and no fault, since `o` is the first of several large allocations.

      **The missing piece is the packed-GQA store.** The scheduler presents packed rows
      deliberately (its comment: FA3's mainloop validates the tile index against `qo_len`, so a
      token length made every tile look out of range). What reconciles the two is upstream FA3's
      packed-GQA addressing of rows as `(token, head-in-group)` pairs -- which is why
      `sparse_mainloop.cuh` carries `packed_Q_ptr` and `group_size` (ported in K1f). The matching
      epilogue store was never ported.

      **This is the same feature as K1d2's deferred fourth change** (the packed-GQA Q mbarrier
      init), and K1g's note that the packed-GQA work should be done as one unit. The deferral is
      the cause of this failure; close them together.

      **[~] STORE PORTED (2026-10-10) -- it changed the output but did not fix it.** Added
      `packed_coord` + `get_packed_gqa_tensor` to `hopper/utils.cuh` (0.7 had **neither** -- the
      helpers were missing entirely), `write_packed` to `hopper/epilogue.cuh`, the
      `#ifdef FLASHINFER_FUSED_PACKED_Q` dispatch in `write_O`, and the macro through the loader
      (`FUSED_PACKED_Q`, default 4).

      | fixture | before | after |
      |---|---|---|
      | `fa3_1chunk` | 3.076e+06, 256/256 | 6.153e+06, **254**/256 (2 rows correct) |
      | `fa3` | 45.1 | 45.34 |
      | `mixed` | 45.44 | 46.11 |
      | `fa2` | PASS | **PASS** 0.002126, 0/258 |

      Two rows going correct says the store's mapping now does something right, so the port is not
      inert; the bulk still wrong says **the other half is missing**. K1f brought only the
      `packed_Q_ptr`/`group_size` **fields** into `sparse_mainloop.cuh` -- that commit scoped the
      file as "+57 lines: `packed_Q_ptr`, `group_size`, plus the packed-GQA load path that reads
      them", and only the fields were applied. So the Q **load** still uses the token-major
      tensormap while the scheduler hands it packed tile indices.

      A half-ported packed-GQA path gives exactly this shape -- output rows mapped correctly,
      computed from the wrong Q rows. **Prediction, not conclusion:** port the load path and see
      whether `fa3_1chunk` drops to the bf16 residual, as the FA2 side did.

      Open, concretely:
      - **[x] the packed-Q gather in `hopper/sparse_mainloop.cuh`** -- spliced in as the
        `#ifdef FLASHINFER_FUSED_PACKED_Q` arm with 0.7's TMA load as the `#else`.
        **`MODE=fa3_1chunk` now HANGS** (log stops after `pers_plan: ...`, no error, killed by the
        400 s timeout) -- which is **exactly the symptom this doc predicted** for the remaining
        piece. The packed path is one warp issuing its own `cp.async` completed with
        `cp.async.mbarrier.arrive`, which tracks only the issuing thread's copies, so all 32 lanes
        must arrive for themselves; the TMA path instead arms `barrier_Q` with
        `arrive_and_expect_tx` for a single transaction arrival. Load ported, init not: the two
        conventions meet on one barrier and the consumer waits forever.
      - **[~] `barrier_Q.init(cutlass::NumThreadsPerWarp)` APPLIED, hang NOT yet resolved.**
        Clean serialised rerun (nothing else on the GPU, self-matching kill trap avoided) still
        stops after `pers_plan: ...`, so the hang is real and the init alone did not fix it --
        an uncontaminated measurement, unlike the round before.

        Cause found in the code: the init was gated on `FLASHINFER_FUSED_PACKED_Q`, but the
        **arrival** side is in `sparse_mainloop.cuh` behind its own
        `#ifdef FLASHINFER_FUSED_Q_MBARRIER` -- mbarrier form arrives from all 32 lanes, `#else`
        drain form arrives **once**. Defining `PACKED_Q` alone armed for 32 while the load took
        the drain branch: 32 expected, 1 arrival. Init is now gated on
        `FLASHINFER_FUSED_Q_MBARRIER` **alone**, and the loader defines it alongside `PACKED_Q`,
        so the two sides cannot disagree. **Applied, not verified** -- the rerun's log still shows
        only `pers_plan` and I could not confirm the rebuild had finished.

      - **[ ] (superseded phrasing) THE remaining piece: `barrier_Q.init(...)`** in
        `prefill_sm90.cuh` behind `FLASHINFER_FUSED_PACKED_Q` / `FLASHINFER_FUSED_Q_MBARRIER`
        (K1d2's deferred fourth change). Arming for 32 is also what lets Q complete **without** a
        `cp.async` group drain -- `commit_group` otherwise sweeps the producer's in-flight KV
        copies into Q's group and committed groups retire in order, so a drain waits on the whole
        KV pipeline (0.3.1 prices that form at **~3x slow**).

      **STALE figures warning:** `MODE=fa3` (45.34) and `MODE=mixed` (46.11) did **not** get fresh
      runs this round -- the loop hung on `fa3_1chunk` first -- so those numbers predate the
      packed-Q load splice and must not be read as results for the current state. `MODE=fa2`
      remains **PASS** rel 0.002126, 0/258.
      - the Q mbarrier init in `prefill_sm90.cuh` behind `FLASHINFER_FUSED_PACKED_Q`. Distinctive
        mis-port symptom: arriving from 32 lanes on a barrier armed for 1 **hung for 1500 s**.

      Superseded candidate: the magnitude says unnormalised or uninitialised Our epilogue matches pristine 0.7 (neither calls `finalize`/`normalize`
      explicitly -- normalisation lives inside `mma_f16` / the `attention_updater`), so the
      question is whether the mainloop ran its KV iterations at all. Print `num_kv_tiles` and the
      updater's row sum from every block; a zero-iteration mainloop leaves `tOrO` untouched and
      would produce exactly this.

      Earlier plan, now partly executed: sanitize for the faulting kernel and block, then trace
      from **every** block --
      the all-block trace is what found `q_indptr`, while single-block traces produced three
      retractions. The FA3 analogue of the `q_indptr`/`kv_indptr` family is the ext's own FA3
      params (`nnz_qo`, `o_stride_n`, and vector-sparse `fa3_kv_indices` vs the upstream-port raw
      `page_indices`), none of which have been read off the running kernel.

      **Fixture bisect that got here -- the fault needed MANY kv tiles:**

      | fixture | kv tiles | result |
      |---|---|---|
      | `fa2_1tile` q=256 kv=128 | 1 | **no fault**, worst rel 1.169, 255/256 bad |
      | `fa2_2tile` q=256 kv=256 | 2 | **no fault**, worst rel 1.991, 224/256 bad |
      | `fa2` q=256 kv=2048 | 16 | **illegal access** |

      So the tile-128 q class by itself does **not** fault -- the illegal access is specific to a
      row spanning many kv tiles. That separates the two things `MODE=fa2` changed at once
      relative to mixed mode, and gives a fixture that RUNS and can be debugged numerically
      rather than by address.

      The numeric signal is structured: in `fa2_2tile` exactly **32 of 256** rows are correct,
      and 32 is one tile-128 q tile's worth of TOKENS (128 packed / gqa 4). The row is 1024
      packed rows = 8 q tiles, so **one q tile lands correctly and seven do not** -- which points
      at per-tile q addressing or tile coverage, and is checkable by printing each work item's q
      tile index and the output rows it writes.

      **Not assumed:** whether the long-kv fault and the short-kv wrongness share a cause, or
      whether either explains mixed mode's 39.68. Treat as separate until one fix moves both.

      Earlier candidate list, superseded by the bisect: `MODE=fa2` is the first
      configuration in which the **tile-128** class runs a row spanning many KV tiles (1024
      packed rows, 2048 kv), so the newly exposed machinery is the **S-slot K/V ring** -- where
      our staged wrapper's `NUM_KV_STAGES` had to be reconciled with 0.7's `NUM_STAGES` via the
      `fa2_kv_stages` probe -- and the **`kv_indices` rebase** by `pers_kv_page_offset`. Both are
      inert for a single-tile decode row, which is all mixed mode exercised. Next: print the
      ring's own slot index and the `kv_indices` offset it dereferences, same block, same
      pre-site null case.

      **Superseded reasoning, kept for the record:**
      `[fa2 enter] blk=84` prints; the unsplit-write trace (a printf immediately before
      `write_o_reg_gmem`) does **not**; and the sanitizer reports the invalid 16 B write **from
      block 84**. A write happened and not on the unsplit path, so the split path ran --
      `write_o_` into `partial_o` -- while `MODE=fa2` has `num_merge_rows=0` and every
      `partial_indptr` is −1. Writing partials the plan never sized explains a 4.5 MB overrun of
      a 1.85 MB tensor far better than the off-by-a-factor index I had written down as the
      candidate, which this **displaces**.

      The disagreement is between two definitions of *"is this row split"*:

      | | formula |
      |---|---|
      | plan | `chunks = ceil_div(kv_len, len_kv_chunk)` -- uniform, **no** causal term |
      | kernel | `ceil_div(CAUSAL ? min((kv_len - q_len) + ceil_div(packed_qo_start + cluster_tile_q, gqa), kv_len) : kv_len, len_kv_chunk)` -- causal-adjusted |

      Both read the same `len_kv_chunk`, so at `fa2_span = max kv_len` both should give 1. So one
      of the inputs **the kernel actually reads** differs from what the plan wrote --
      `len_kv_chunk`, `q_len`, or the causal flag. Note the FA2 runners are instantiated with
      `MaskMode::kCausal` while `FusedFA3PersistentDispatched` is instantiated with
      `CAUSAL=false`, so the two sides may not even agree that the bound applies.

      Next: one printf on the **split** branch (`num_kv_chunks`, `len_kv_chunk`, `q_len`,
      `kv_len`, `packed_qo_start`, `partial_indptr`) in the same block -- reading the kernel's own
      inputs rather than re-deriving them. That is the step that has worked every time this
      session, and skipping it is what produced the packed-vs-token guess.

      Earlier candidate, now displaced:
      in `MODE=fa2` every row should have `num_kv_chunks == 1` (`fa2_span` is max kv_len, so
      `ceil_div` gives 1), which means the **unsplit** path and `write_o_reg_gmem` writing
      `final_o`. A 4.5 MB overrun of a 258x28x128 bf16 tensor (1.85 MB) is ~2.4 tensors past the
      end, so the index is several times too large rather than slightly off -- consistent with a
      packed-vs-token factor still live on this path, but that is a **candidate, not a finding**.
      Trace the **faulting** block (`FUSED_TRACE=85`), not an arbitrary one.

      **Superseded note (kept for the record), on the fault that is now fixed:** The faulting block is still
      `>= sm_split`, so an FA2-role block, and the write is 16 B -- the vectorised partial store.
      Next step is to read off **which base the FA2 role uses** for it: `persistent.cuh`'s
      comment at the partial path says *"o_indptr (partial_o): [packed_qo_len * num_kv_chunks,
      num_kv_heads, head_dim]"*, so the FA2 roles may index through `o_indptr` while the plan
      fills `partial_indptr`, and the ext may be feeding one where the other is read. Establish
      it from the code -- five wrong mechanisms were proposed for the last device assert in this
      project before an environment bisect found the cause.

      Fixture notes (`drivers/fused_reference.py`): `page_size=1` with pages in request order, so
      the vector-sparse `kv_indices` and the raw page ids coincide numerically and that aliasing
      trap is retired rather than live; FA3 rows first (= `pers_row_offset`); each side's pages
      contiguous so `pers_kv_page_offset` is just the FA3 page count. Plans built fine:
      `pers_plan: 59 entries, num_merge_rows=1024, num_fa3_clusters=16`, `fa3_plan: 9 entries`.

- [~] **K1l-old. The Python bridge is in (`flashinfer@7c74969`), verified registered on
      the pod: `fused_run` True, `build_fused_plan` True.** Nothing could previously produce the
      `pers_plan` vector `fused_run` takes -- flashinfer's own plan functions build
      `HolisticPlanInfo`, which K1a replaced for this path -- so the kernel was undriveable from
      a test. `build_fused_plan` returns the vector; the caller keeps the two workspaces it was
      built against, since the plan's offsets point into them. It checks that the four host-read
      inputs are CPU tensors and that `fa3_row_mask` is `at::kBool`, because the planner takes a
      `const bool*` and an int32 mask would read 4x the elements without failing.

      **Driver argument spec** (so this is written against the signature rather than guessed --
      `tests/fused_fa3_persistent_ext.cu:129`, 26 arguments). The non-obvious ones, with the line
      that consumes each:

      | argument | meaning |
      |---|---|
      | `kv_indices` | **vector-sparse** offsets, for the FA2 roles (`:330`, rebased by `pers_kv_page_offset`) |
      | `fa3_kv_indices` | the same thing for FA3's own params (`:221`) -- flashinfer's FA3 multiplies the entry by `k_stride_n` |
      | the upstream-port `page_indices` | **RAW page ids** (`:300`), which upstream's paged path indexes as a real page number; feeding it the vector-sparse array read far outside the cache |
      | `pers_row_offset` | FA3's rows come FIRST in q/o, so the FA2 roles' `q` and `final_o` are advanced by it (`:319`, `:342`, and the reduction's output at `:390`) |
      | `pers_kv_page_offset` | the two sides are planned independently, so the persistent plan's `kv_indptr` restarts at 0 while its pages begin after FA3's; without the rebase it reads FA3's KV and produces plausible but wrong output (measured 1.52 rel err vs 0.0044 rebased) |
      | `fa3_num_ctas` | MUST equal `sm_split`; the dispatcher refuses the launch otherwise, because FA3's scheduler indexes `work_indptr[blockIdx.x]` over the planned count |

      The FA3 plan (`fa3_plan`, a `PrefillPlanSM90Info` vector) still comes from flashinfer's own
      FA3 plan path -- `BatchPrefillWithPagedKVCacheWrapper(backend="fa3")._plan_info` -- over the
      FA3 row subset only.

      Everything the kernel needs now exists and compiles:
      plan (K1c2, 19/19 on GPU), module (K1h), split-KV write (K1g), reduction (K1k). Nothing
      has yet checked an attention **output**. Needs the correctness driver recreated (K1i) --
      `fused_run`'s 26-argument signature is at
      `port07/tests/fused_fa3_persistent_ext.cu:129`, and the fp32 reference must be **causal**
      per the standing instruction. Keep the driver in-repo, not only on the pod.

      Note what this will and will not establish: a PASS covers the FA3 path, the three FA2 tile
      classes, the split-KV partials and the merge together. It will NOT cover cross-role
      backfill or multicast (B1b/B2c/B3), which need `tma_store_wait` and are still untested,
      nor topk>1 trees (K0 -- the 4-mode path is chain-draft only, and the in-step verify gate
      cannot detect that because reference and test share the causal assumption).
- [ ] **K1i. Recreate the correctness drivers.** The pod's `drv/` kept only the perf scripts
      (`b1_headroom.sh`, `b1b_headroom.sh`, `split_perf_mtp.sh`, `b1_analyze.py`, `b2_top.py`,
      `restore.sh`); `split_unit.py`, `split_margin.py` and `split_verify_mtp.sh` are gone.
      `fused_run`'s 26-argument signature is in `port07/tests/fused_fa3_persistent_ext.cu:129`.
      Keep these in the repo this time, not only on the pod.

- [ ] **K1g-old (superseded by the above). `store_partial` is NOT ported -- the split-KV path is incomplete, and the green
      compile hides it.** 0.7's epilogue has **no** split path at all (only `store` /
      `store_tail` / `store_zero`), so `store_partial` is entirely ours. Counts: 0.3.1's
      `epilogue.cuh` defines it and its `prefill_sm90.cuh` calls it **twice**; the 0.7-based
      copies have **zero** of each. The six epilogue fields K1f added are plumbed but
      **unread**, and default to `nullptr` -- which is exactly why the TU compiles. Any FA3 row
      that splits over KV therefore writes no partial and the merge reduces memory nothing
      wrote. (0.3.1 measured this class of bug once already: fixture 2 showed 511/512 rows
      unwritten.)

      Do this together with **K1d2's deferred 4th change** (the packed-GQA Q mbarrier init in
      `prefill_sm90.cuh`): both belong to the same feature, and the mbarrier mis-port has the
      distinctive symptom of a 1500 s hang when 32 lanes arrive on a barrier armed for 1.

      Then the fp32-reference PASS, then a timing. **No A/B, and no correctness claim, before
      the reference passes** -- `NVCC_EXT_OK` is not one. The five missing identifiers are
      now fully diagnosed, and the diagnosis is better news than the error list suggested:
      **four of the five are OUR OWN symbols**, not things 0.7 removed. Nothing needs a 0.7
      equivalent invented; the definitions simply have to be carried over, which the fuzzy apply
      failed to do while letting their uses through.

      | symbol | where it really lives | action |
      |---|---|---|
      | `kNoClusterOverride` | ours, `persistent.cuh:262`, `static constexpr uint32_t = 0xFFFFFFFFu` | carry the constant |
      | `cluster_override` | ours, function **parameter** at `persistent.cuh:275` (defaults to `kNoClusterOverride`) | carry the signature change |
      | `tid_base` | ours, function **parameter** at `persistent.cuh:276` (defaults to 0) | carry the signature change |
      | `kPersistentMainloopBarrier` | ours, **`prefill.cuh:52`** -- `static constexpr int = kFusedQueueBarrier + 1` | **port `prefill.cuh`'s barrier constants FIRST** |
      | `thr_local_kv_offset` | **upstream 0.7 has it** in its own `persistent.cuh` | resolves once the port is structurally correct |

      So the order is: (1) port the barrier-constant block from our `prefill.cuh` (small and
      self-contained, part of that file's +65); (2) hand-apply to pristine 0.7's `persistent.cuh`
      the `kNoClusterOverride` constant, the `cluster_override`/`tid_base` parameters, the
      `FUSED_GRID` role indexing, and the recovered `cluster_tile_q` guard -- by hand, hunk by
      hunk, re-running `port07/tests/nvcc_check.sh` after each.

      **What `cluster_override` and `tid_base` are for, since they are the fused design and not
      incidental:** `cluster_override` lets the fused kernel's dynamic scheduler hand a block a
      cluster index rather than deriving it from `blockIdx` -- that is the mechanism by which an
      SM that finished FA3 picks up FA2 work (cross-role backfill). `tid_base` lets a warpgroup
      compute `ltid = threadIdx.x - tid_base` so FA2's roles address shared memory correctly
      when they occupy only part of a fused block. Neither has a 0.7 equivalent because neither
      exists upstream; they are the fusion.
- [x] 2026-10-09 **overlay build mechanism established** -- a MERGED include tree (0.7's tree
      copied, ours laid on top) rather than `-I` ordering. Both trees use relative includes, so a
      partial overlay breaks them regardless of order; this is also what the real extension build
      must do. `port07/tests/nvcc_check.sh`.
- [x] 2026-10-09 **host checks pass and are the right granularity**: `FusedPlanInfo` round-trip
      (58 entries, every field preserved, `plan_info[-1] == num_fa3_clusters`), work-item
      enumeration (task0=0 task1=0 task2=7 task3=256, `max_splits_fa3=4`), merge layout (64
      tiles, 32768 partial rows, CSR monotonic, unsplit rows absent from the merge lists).
      These caught nothing that the compile then re-caught, i.e. they are complementary, not a
      substitute -- the compile found what they structurally cannot.
- [ ] 2026-10-09 **K1c in progress** -- 15/22 applied; the 7 rejects are the 4-task plan builder.

#### A. Ultra fine-grained tile dispatcher for the different modes

The four populations in a 4-mode step sit at different tile sizes, and the contested band is
**verify**. Prefill fills a 128-wide WGMMA tile many times over; decode and draft are single
`tile16` items; verify is `num_draft_tokens x gqa`, anywhere from ~16 to >128. Today a hand-set
`MIN_PACKED_QO = 48` routes anything above 48 to FA3's **fixed** `TILE_Q = 128`, so rows in
[48,128) waste up to 63% of the tile.

- [ ] **A1. Dynamic WGMMA tile size in FA3: mix 128 and 64, chosen PER WORK ITEM** from
      `packed_qo`. A 56-row item then runs at 88% occupancy instead of 44%.
- [ ] **A2. FA2 tile classes 16 / 32 / 48 / 64.** These are **template instantiations over
      `mma16`**, not new kernels, and the **2-CTA overlap schedule is preserved** across them.
      Experiment on how many widths are worthwhile -- each costs compile time, code size and
      registers -- by sweeping {16,64} vs {16,32,64} vs {16,32,48,64} and keeping the widths that
      pay for themselves. Report per-width timings with the chosen set.
- [ ] **A3. Fitted dispatch rule replacing `MIN_PACKED_QO`**, on `(packed_qo, kv_len)` and per
      mode. **Linear model recommended** (user); try a bare threshold and a 2-D ratio rule as
      baselines and keep the simplest form that matches the fit's agreement. Train on measured
      per-tile times for FA3-at-{64,128} vs FA2-at-{16,32,48,64}; report agreement against the
      measured winner the way the decode guard's 11/25 -> 23/25 is reported. **Needs a denser
      grid than the existing 25 points**, which were taken for the decode-population question.

#### B. Fuse as many draft and verify EPILOGUE ops as possible

This is what makes the kernel-level framing defensible. Attention is only part of a 4-mode step;
the draft and verify **epilogues** are a long tail of tiny kernels -- draft-token sampling,
argmax/accept comparison, accept-length reduction, hidden-state gathers and slices, cache-loc
writes, tree/mask construction -- each its own launch, each far too small to fill the GPU.
Fusing only one prefill chunk with seven verify rows is the marginal case the user warned about:
*"hard to make a meaningful motivation... a reviewer will ask why start from the kernel point of
view for colocation if gains are so minimal?"*

- [ ] **B1. Quantify the headroom FIRST, before optimising.** Per 4-mode step report: the
      **count of kernel launches by role** (draft-decode, draft-extend, draft/verify epilogue,
      target-prefill, target-verify); the **fraction of step wall time** in the small ones; their
      **achieved SM occupancy** (so "small" is measured, not asserted); and the `packed_qo` tile
      histogram. nsys/ncu or CUDA events per role. **This is the reviewer-facing motivation
      table.** Optimising before measuring headroom is how the -5% happened.
- [ ] **B2. Fuse the epilogue tail.** Absorb as many of those ops as possible into the two
      kernels -- epilogue work fused into the attention kernel's own epilogue, and **cross-role
      backfill** so small draft/verify work fills the SMs the prefill mainloop leaves idle.
      `tma_store_wait` is the enabling primitive (named in the goal, still untested).
- [ ] **B3. Multicast** for the shared reads this creates (also named in the goal, untested).
- [ ] **B4. Scope the cross-MODEL part explicitly.** Draft ops run the DRAFT model, not the
      target, so absorbing them is cross-model co-scheduling -- a strictly bigger step than the
      prefill+verify partition already built. Do not assume it falls out of the existing split.

#### Supporting / correctness, not performance

- [ ] **K0. Decide whether trees (topk>1) are in scope.** `mix_with_verify` carries no
      `custom_mask`, so verify rows are masked plain causal -- correct only for chain drafts. If
      in scope, plumb the mask through and build a gate whose reference does **not** share the
      causal assumption. If not, state "chain draft only" wherever 4-mode results are quoted.
      Bounds every result in this doc.
- [ ] **K5. Re-measure in-server at the shape the schedule emits**, reporting `last_fa2_rows` and
      the tile histogram beside every timing. The gain must clear the measured **+/-1.5-3%**
      noise floor on the upside -- user: *"-5 to -8.4% is not acceptable."*
- [ ] **K6. CUDA-graph fault.** The split still dies with graphs on (`IndexKernel.cu`
      out-of-bounds via flashinfer `plan`, from `decode_cuda_graph_runner.load_batch`). Needs a
      capture-boundary bisect, not another hypothesis. Graphs on is the production config.

### OPEN -- server / workload, after the kernel items

- [ ] **EDF perf-model vs baseline unfused schedule** -- never run. The claim is that the fused
      kernel's flatter cost makes the model more accurate.
- [ ] **Superset items 5a-5f** in `tasks/pd_colocation_vs_disagg.md`. **5a (single node, vs
      prefill-prioritized AND mixed-chunk) needs only the current H200 and can proceed in
      parallel.** 5b (PD mux + 1P3D/2P2D) needs H20 and is gated on the kernel items.
- [ ] **PD point 4** (two-way RDMA / congestion / randomized pull) -- currently a NEGATIVE result
      at 16 GPUs; not demonstrated on the full 4-NIC fabric.

### DONE and verified -- chronological (SHAs and commands in the sections below)

| when | item |
|---|---|
| 10-06 | Two executors beat FA2-on-everything on the fork harness and PASS the reference -- the result that motivated the split. |
| 10-07 | 4-mode MTP load sweep: the gain grows with load and needs the split. Multi-stream start-offset methodology retraction recorded. |
| 10-08 | Decode-population guard derived from the 25-point boundary grid; replaying it lifts agreement with the measured winner 11/25 -> 23/25. |
| 10-08 | `fi` restore recipe executed; fused extension JIT-built at runtime. |
| 10-09 | `upstream_port` **rebased onto live upstream main `3fef129ff`** -- 64 commits, zero conflicts, port markers verified post-replay. |
| 10-09 | Decode guard **ported to upstream** and re-verified there (11/25 -> 23/25). |
| 10-09 | Upstream-facing data path **CPU-verified**: `dense_block_table` 11/11, `fa3_subset_varlen` 10/10, row partition inert when disabled. |
| 10-09 | **`forward_extend` dispatch written** -- two executors over disjoint row subsets, scatter combine. Index arithmetic 400/400; that test caught the FA3 page table being indexed by batch row instead of `req_pool_index`. |
| 10-09 | Interface finding: **`sm_margin` is an upstream FA3 parameter**, so the SM budget needs no private entry point. |
| 10-09 | **Split PASSES the causal fp32 reference, 5/5 shapes** standalone (err 0.00232-0.00424, each row subset scored separately) -- `drv/split_unit.py`. |
| 10-09 | **SM pre-assignment measured: 1.224x** on isolated attention at `sm_margin=8` vs the unsplit baseline (152.6 vs 186.7 us), **correctness bit-identical across all six margins** -- `drv/split_margin.py`. |
| 10-09 | **4-mode blocker root-caused and fixed**: `extend_prefix_lens` came from a STALE `seq_lens_cpu` mirror while `seq_lens` came from the device tensor (`eagle_prepare_for_verify` advances one, not the other). `sglang-up@b8b405822`. |
| 10-09 | **4-mode step runs clean on upstream**: 151 merges, 0/56 client errors, 0 tracebacks, 0 width mismatches. |
| 10-09 | **Split PASSES in-server**, guard off, no launch-blocking: 5001 dispatches, in-step reference 10 ok / 0 FAIL (worst 0.0112, tol 0.02), 0/56 client errors -- `drv/split_nocg.sh`. |
| 10-09 | **CUDA-graph fault isolated to the graph path** by environment bisect, after five wrong code-level mechanism guesses. |
| 10-09 | **Perf harness noise floor measured: +/-1.5-3% tok/s** from two identical runs; warmup and arm-order confounds identified and controlled (per-arm warmup, reversed order, `base` last). |
| 10-09 | **Always-on composition counters** (`MIXEDSPEC_STEPS`, `FUSED_SPLIT_CALLS`, `FUSED_SPLIT_ONESIDED`, `base_plan_skipped`) -- each added after a gate reported a verdict about an unexercised path. |
| 10-09 | **Fused-kernel JIT loader works from inside sglang** (increment 1): C++20, bootstraps flashinfer's generated config itself, registers `torch.ops.fused_fa3.fused_run`. `sglang-up@9da582ac3`. |
| 10-09 | Pod expired mid-run; trial **re-forked** to `f929493da6e45f10` (`vibecuda`); restore recipe recorded including the `overrides.attachments: []` requirement. |

### Stale claims -- do NOT reuse

* **"Fused kernel PASSES, 1.30x at sm_split 112"** (133.5 us vs 174.0 us): measured on
  **flashinfer 0.3.1**, against an extension whose `fused_run` took 20 args. The current stack is
  0.7.0.post1 with a 26-arg schema and a changed plan struct. The measurement happened; it is not
  evidence about today's code. Re-establishing it is part of K1.
* **Any TTFT/ITL "win" from the first serving A/B** (-44% mean, -73% P99 TTFT, -14% P99 ITL):
  warmup artifacts, fully retracted. The warmed order-reversed A/B is the only valid one.
* **`g ~= 0` coverage**: wrong, retracted -- the guard fired 7501/8001 at MC=56.
* **`sm_margin = 8`**: the kernel-harness optimum does not transfer; margin 0 is as good or
  better in serving.
* **Retracted diagnoses for the device assert** (keep the lesson, not the path): shared
  flashinfer workspace; launch-blocking masking a race; plan-copy use-after-free; FA2 plan-array
  lifetime; stride-ignoring KV reshape. Four were real latent bugs whose fixes are kept; none was
  the fault. The fault was a CUDA-graph interaction, found by bisecting the ENVIRONMENT after
  five code-level guesses. **An async device assert carries no location information, so
  hypotheses built on its traceback are unfalsifiable.**

**Not demonstrated:** the >=30% TTFT/ITL and >=10% throughput bar. The only in-server number for
a working kernel is **-5.0%**.

---

## 2026-10-09 -- step (c) complete: the `forward_extend` dispatch is written on upstream

`sglang-up@86e375375` (branch `upstream_port`, 65 commits on upstream main `3fef129ff`).
Code written and CPU-verified; **not yet GPU-verified**, so the box stays open.

**What landed.** `FlashInferAttnBackend._fused_split_dispatch` in
`python/sglang/srt/layers/attention/flashinfer_backend.py`, called from the paged branch of
`forward_extend`. It takes the row partition `self.fused_split_mask` (commit 62), forms the two
disjoint row subsets, runs one executor on each, and combines with `index_copy_`.

**Interface finding that removes the bypass.** Upstream FA3's `flash_attn_with_kvcache` already
takes **`sm_margin`** -- "can be tuned if some SMs are used for communication"
(`python/sglang/kernels/ops/attention/flash_attention.py:39`). That is exactly the SM
pre-assignment knob, as a public upstream parameter. So the FA3 half of the split needs **no
private entry point at all**. This corrects the earlier claim in this doc that the SM budget
"cannot pass through upstream's wrapper API": that is true only of flashinfer's persistent
wrapper, which derives `sm_count` from the device (`prefill.py:204`). The budget is
`SGLANG_FUSED_SM_MARGIN`.

**The combine is a scatter, not an LSE merge.** Upstream's ragged+paged pair covers the *same*
rows and therefore merges via LSE. Our two subsets are *disjoint*, so an `index_copy_` per
subset is both correct and cheaper -- no merge kernel.

**FA2 subset planning needs no second `KVLocPlan`.** flashinfer's public `begin_forward` takes
explicit `qo_indptr`/`kv_indptr`/`kv_indices`, so the subset plan is built directly from
`req_to_token` -- the same source the FA3 page table is gathered from.

**Page table: a gather, not a conversion.** Upstream FA3 documents `page_table` as
`[batch, max_blocks]`, and `req_to_token` is already in that shape, so the FA3 subset table is
`req_to_token[req_pool_indices[fa3_rows], :max_k]`. `mixed_dispatch.dense_block_table`
(commit 63) converts flashinfer's *flat vector-sparse* layout and is not needed on this path;
it stays for the flashinfer-side consumer.

**Index arithmetic CPU-verified 400/400 random batches** (`/tmp/t_idx.py`): the two token sets
partition every q row exactly once; both subsets' `cu_seqlens_q` are densely renumbered and
row-ordered; the FA2 kv gather and the FA3 page table agree row-by-row with the reference.
That test **caught a real bug**: the FA3 page table was indexed by *batch row* instead of
*req_pool_index* -- i.e. it would have read another request's pages whenever the two orders
differed. Fixed before commit.

**Refused, not approximated.** Four features return `None` (caller takes the ordinary
single-wrapper path), each with a concrete TODO naming the mechanism, because the alternative
is silently wrong attention that a throughput A/B would not reveal:
sliding-window layers (need `_swa_kv_pool`'s own page table + `window_size=(sws,0)` as derived
at `flashattention_backend.py:2030`); fp8 KV (FA3 wants `k_descale`, this backend carries
`k_scale_float`, and the two conventions are unverified); multi-item scoring (item-aware mask
outranks the uniform causal mask); cross-attention / encoder-only.
A one-sided partition also returns `None` and increments `fused_split_onesided` -- that is the
decode-population guard working as designed, not a failure.

**Always-on counter**: `FUSED_SPLIT_CALLS n (fa3_rows= fa2_rows= sm_margin= onesided=)`.
Required by [[gate-needs-noise-floor-and-reference]] -- a debug-gated counter has already
produced one false pass on this project. This satisfies priority item 2.

- [x] 2026-10-09 **(c) `forward_extend` dispatch written and CPU-verified** -- two executors
      over disjoint row subsets, scatter combine, `sm_margin` as the SM budget, 400/400 on the
      index arithmetic (one real bug caught: page table indexed by batch row).
- [ ] 2026-10-09 **(c2) GPU-verify the dispatch** -- greedy ON vs OFF with a self-vs-self noise
      floor, asserting `FUSED_SPLIT_CALLS > 0`. In progress; pod `/dev/shm/w` survived intact
      (venv torch 2.14.1, flashinfer 0.7.0.post1, qwen7b + qwen7b_eagle3_sgl, `fi`,
      `flash-attention`), but the pod's `up` tree sits at upstream `b7b2975`, an **ancestor** of
      my base `3fef129ff`, with the 4-mode port as uncommitted working-tree edits. Pushing a
      single `b7b2975..86e375375` patch (987K, 23 chunks) rather than per-file drops, because
      `flashinfer_backend.py` alone differs by 92 lines across that drift and mixing bases there
      is how the last illegal-access was introduced.

### First GPU gate, 2026-10-09: two failures, both of which the gate was built to catch

Command: `bash /dev/shm/w/drv/split_gate.sh` on the vibecuda H200 pod (job `f393156fcc0614b3`),
pod tree `/dev/shm/w/up` at upstream `b7b2975` + patch `b7b2975..86e375375`, venv
`/dev/shm/w/venv` (torch 2.14.1, flashinfer-python 0.7.0.post1), model `/dev/shm/w/qwen7b`,
server: `--attention-backend flashinfer --enable-mixed-chunk --chunked-prefill-size 2048
--page-size 1 --max-running-requests 64 --mem-fraction-static 0.8`.

**Result: `VERDICT: INCONCLUSIVE`, `FUSED_SPLIT_CALLS lines: 0` in the ON arm.**

1. **The partition never ran.** 78/78 steps logged
   `fused split partition skipped: 'FlashInferAttnBackend' object has no attribute
   'num_qo_heads'`. `num_qo_heads`, `num_kv_heads`, `head_dim`, `q_data_type` and `data_type`
   all live on `FlashInferIndicesUpdaterPrefill`, **not** on the backend; commit 62 and the new
   plan both read them off `self`. Nothing was computed wrongly -- the warning fired and the
   ordinary path ran -- but **the ON arm was measuring the baseline**. Had the gate been
   text-equality only, it would have returned a clean PASS for a split that never executed.
   This is precisely the failure mode [[gate-needs-noise-floor-and-reference]] ends on ("assert
   the gate EXERCISED the path"), and the always-on counter is what exposed it. Fixed in
   `ce6cf5670` by reading from `self.indices_updater_prefill`, with a `getattr` guard for
   `skip_prefill`.

2. **The server-level ON-vs-OFF comparison has no noise floor.** The OFF arm did not reproduce
   **itself** across two runs of one unchanged binary, so no ON/OFF diff is attributable.
   Greedy output under a concurrent workload depends on batch composition, which depends on
   arrival timing. This is noise-floor item 1 of the same memory, hit again.

   **Fix: move the reference inside the step.** `SGLANG_FUSED_SPLIT_VERIFY=1` compares the
   split's output against the ordinary planned wrapper on the **same inputs in the same step** --
   the read-only call the OFF path already makes, so the reference cannot drift from the thing
   under test (memory item 5) and there is no run-to-run noise to confound it. Error is
   **scale-normalised** (`max|o-ref| / max|ref|`), not clamped elementwise relative, which
   returns ~1e5 on any near-zero activation (memory item 4). Whole-tensor, FA3-rows and
   FA2-rows errors are reported separately so a one-sided defect is attributable, and the
   verdict distinguishes FAIL / NOT EXERCISED / PASS rather than collapsing them.

### Also fixed before that gate: a 28x host-cost bug in the first dispatch

`forward_extend` runs **per layer**. The first version built the token index tensors, three
indptrs, the per-row `req_to_token` gather, an H2D copy and a flashinfer `begin_forward` inside
it -- so ~num_layers times per step. A throughput A/B on that version would have measured the
host overhead rather than the kernel. Split into `_fused_split_plan` (once per step, from
`init_forward_metadata` -- the division upstream already uses) and `_fused_split_dispatch` (per
layer: two executor calls plus the scatter, no host work), commit `9abe91e25`.

- [ ] 2026-10-09 **in-step self-check running** (`drv/split_verify.sh`): split ON with
      `SGLANG_FUSED_SPLIT_VERIFY=1`, verdict keyed on both the error magnitude and
      `FUSED_SPLIT_CALLS > 0`.

### Structural risk to the acceptance bar, derived from the measured boundary (2026-10-09)

This follows from numbers already in this doc and should not wait for the serving A/B.

The decode-population guard fuses only while `kv_tiles <= decode_rows / 2`, and its own measured
basis says that with **zero co-resident decode rows the fused kernel LOSES on every measured kv
(0.617-0.704x)**. So the split's win is confined to decode-heavy batches. Consequences for the
bar ("at least 30% on TTFT & ITL in realistic workloads (e.g. our mixed chat + tracelab) and at
least 10% throughput"):

| regime | typical shape | guard | expected effect |
|---|---|---|---|
| chat, many concurrent short turns | ~10 kv tiles, 20-60 decode rows | satisfied | split fires; this is where any gain must come from |
| tracelab / agent, long prefills | 32-256 kv tiles, few decode rows | refuses | split inert by design -- no gain, and correctly so, since fusing would cost 30-38% |
| prefill-only burst | no decode rows | refuses (`zeros_like`) | inert |

Two honest readings, and they are distinguishable by measurement rather than argument:

1. **The bar is reachable but only through the chat half.** Then the mixed-workload number is a
   weighted average, and a 30% TTFT/ITL gain on the mixed stream requires something well above
   30% on the chat portion. Worth computing the required chat-side gain from the mix ratio
   before running anything.
2. **The boundary is mismeasured for serving.** It came from kernel microbenchmarks
   (`boundary_28x4x128.json`, 25 points) where "decode rows" was a sweep parameter. In the
   server, a long-prefill batch under mixed-chunk also carries whatever decode rows happen to be
   resident, and the relevant cost is the step's critical path, not the kernel's isolated time.
   The guard may therefore be refusing batches that would in fact win end-to-end.

**Action:** when the correctness gate passes, run the A/B with the guard ON and again with
`SGLANG_FUSED_DECODE_GUARD=0` on the SAME workload. That separates the two readings directly:
if the unguarded arm wins on long-prefill traffic, reading 2 holds and the boundary needs
re-fitting at the serving level; if it loses, reading 1 holds and the bar has to be met from
the chat side alone. Either way this is measured, not argued.

- [ ] 2026-10-09 **guard ON vs `SGLANG_FUSED_DECODE_GUARD=0` on one workload** -- decides
      whether the kernel-level boundary transfers to serving.

### The Amdahl bound, from the closest measured precedent (2026-10-09)

`tasks/realistic_workload_analysis.md` already contains the relevant measurement, and it bounds
what any gated attention kernel can do on this workload. On the fork, the **gated** persistent
kernel (`SGLANG_MIXED_ATTENTION_KERNEL=auto`, k-NN gate) engaged on **63/500 to 203/1500 mixed
batches, i.e. 12.6-13.5%**, and the end-to-end result on TraceLab was:

| Rate | Throughput | Pooled P99 ITL |
|---|---|---|
| 0.02 | -0.01% | -2.7% |
| 0.05 | +0.06% | -0.4% |
| `inf` | **+0.27%** | **-4.6%** |

Scope: that is a *different* kernel (persistent LPT) under a *different* gate (k-NN, not the
ratio guard), so the numbers do not transfer directly. The **mechanism** does: a gate that
engages on a minority of batches is Amdahl-bounded regardless of how good the kernel is on the
batches it takes.

The arithmetic: with the kernel engaged on a fraction `g` of batches and attention a fraction
`a` of step time there, the end-to-end ceiling -- reached only if attention became **free** --
is `g * a`. At `g = 0.13` and a generous `a = 0.3`, the ceiling is **~4%**. Measured was
+0.27% throughput, consistent with that bound and with a per-batch win in the 1.0-1.3x range.

**So the >=30% TTFT/ITL and >=10% throughput bar cannot be met by this kernel under a gate that
engages on ~13% of batches, no matter how well the kernel is optimized.** The 1.30x kernel-level
speedup already measured is not the binding constraint; coverage is. Reaching the bar requires
at least one of:

1. **Coverage**: the split must engage on most steps, not a minority. That is reading 2 above --
   the ratio guard, fitted on kernel microbenchmarks, may be refusing batches that win
   end-to-end. The guard-ON vs guard-OFF A/B settles it.
2. **An attention-dominated regime**: `a` near 1, i.e. long context at high concurrency, where
   attention is the step's critical path rather than ~30% of it. This is a workload choice and
   must be stated as such, not slipped in.
3. **A different lever**: the scheduling half of this project (4-mode step, EDF ordering,
   cache-hit-first routing) acts on queueing delay, which is where TTFT at saturation actually
   lives -- note the P99 TTFT of 7.28-33.0 s above against a mean of 0.73-1.32 s. A 30% TTFT
   gain is far more plausible from admission order than from attention math.

This is recorded now, before the serving A/B, so the A/B is a test of a stated prediction rather
than a search for a number. If the measured combined arm lands near +4% rather than +10%, that
is the predicted outcome of the coverage bound and not a new surprise.

### Why the split can ONLY be tested on the 4-mode arm, and what that does to coverage (2026-10-09)

Measured, from the plain-server gate's own diagnostics:

```
FUSED_SPLIT_ONESIDED 1/1:     fa3=0 fa2=1 rows=1  max_kv_tiles=16 need_decode_rows>=32
FUSED_SPLIT_ONESIDED 201/201: fa3=0 fa2=3 rows=3  max_kv_tiles=61 need_decode_rows>=122
```

`rows=1` and `rows=3`. The extend batches contain almost no rows -- so the guard was never the
obstacle on a plain server; **there were no decode rows in `forward_extend` to begin with.** In
sglang a decode batch goes through `forward_decode`, a separate entry point. `forward_extend`
sees prefill (and chunked-prefill) rows. The two populations the split exists to overlap are not
co-resident there at all, whatever the arrival pattern -- which is why staggering arrivals over
25 s changed nothing, and why `--enable-mixed-chunk` alone is not sufficient.

The configuration that *does* put them in one forward is the **4-mode MTP step**: a prefill
chunk plus the running batch's verify rows in a single target forward. That is precisely what
this project built, and it is the only place on upstream where the split has anything to do.
Row classes fall out correctly: with qwen7b's 28/7 heads, GQA = 4, so a verify row's
`packed_qo = num_draft_tokens * 4 = 16`, below `MIN_PACKED_QO = 48` and below `MIN_QO_LEN` --
FA2 class; the prefill chunk packs to 8192 -- FA3 class. Two-sided by construction.

**This tightens the coverage bound rather than relieving it.** The split's coverage is now
bounded by the 4-mode gate's own firing rate, which requires simultaneously: `SGLANG_MIXED_SPEC=1`,
mixed chunk on, a spec algorithm, a non-empty running batch, no logprob request, no input embeds
and no beam groups -- *and then* the ratio guard on top. So `g` in the `g * a` ceiling is the
product of two gates, not one. The honest prediction stands and gets stronger: a double-gated
kernel cannot carry a 30%/10% end-to-end bar, and the measured `MIXEDSPEC_STEPS` / `FUSED_SPLIT_CALLS`
pair is what quantifies `g` directly -- both counters are always-on for this reason.

**Correction to the plan recorded earlier today:** the perf A/B driver I staged for a plain
server would have measured nothing, for this reason. Superseded by the MTP arm.

- [x] 2026-10-09 **the plain-server split test is structurally void** -- decode never enters
      `forward_extend`; measured `rows=1..3`, fa3=0 on every step. Not a guard issue.
- [ ] 2026-10-09 **4-mode MTP self-check running** (`drv/split_verify_mtp.sh`): EAGLE3 chain
      draft (topk=1, 4 draft tokens) + `SGLANG_MIXED_SPEC=1` + split ON + in-step verify.

### The dispatch was installed in a branch the default config never takes (2026-10-09)

The MTP gate came back `NOT EXERCISED` again, and the server died with a shape error. Reading
the traceback gave the real answer:

```
flashinfer_backend.py:1753 in forward_extend
  o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(...)
ValueError: q.shape[0] (2051) does not match qo_indptr[-1] (2054)
```

That is the **ragged** branch. And upstream chooses it by default:

```python
use_ragged = (not self.enable_deterministic
              and not is_in_tc_piecewise_cuda_graph()
              and not self.use_paged)          # self.use_paged = envs.SGLANG_FLASHINFER_USE_PAGED
```

**So the dispatch lives in the `not use_ragged` branch, which the default configuration never
enters.** The partition could have been perfect and the dispatch would still never have run.
This also explains the plain-server result more completely than the guard did: the plan runs
from `init_forward_metadata` (always), which is why `FUSED_SPLIT_ONESIDED` appeared, while
`FUSED_SPLIT_CALLS` stayed 0 because the dispatch branch was unreachable.

Fix: `SGLANG_FLASHINFER_USE_PAGED=1`, an **upstream-defined** env var, not a private hook.

**This changes the A/B design, and the change matters.** Both arms must set
`SGLANG_FLASHINFER_USE_PAGED=1`, or the comparison confounds the split with ragged-versus-paged
attention -- two different algorithms over different KV regions. The cost of forcing paged mode
is then its own measurement (paged baseline vs ragged baseline), and it is a real cost to carry:
the ragged path exists because it is faster for the new-token region. If forcing paged mode is
itself a regression, the split has to beat the ragged baseline from a deficit, which is a
stiffer bar than beating the paged one.

**Separately: a 4-mode port bug on the ragged path.** `qo_indptr[-1] - q.shape[0] = 3 =
speculative_num_steps`, a *constant total* deficit, so one verify row contributed 1 token where
the merged metadata promised `num_draft_tokens = 4`. A split-OFF control at the same settings is
running to establish whether this is pre-existing in the port or caused by my second wrapper
sharing `workspace_buffer`. Either way it is a defect to fix, not to route around: the ragged
path is upstream's default, so the split ultimately needs to work there too -- as a row-wise
split whose FA3 subset takes one `flash_attn_with_kvcache` call (its page table plus
`cache_seqlens` already span prefix and new tokens) while the FA2 subset keeps upstream's
ragged+paged LSE merge restricted to those rows.

- [ ] 2026-10-09 **split on the ragged path** -- required for the default config; currently
      paged-only.
- [ ] 2026-10-09 **cost of `SGLANG_FLASHINFER_USE_PAGED=1`** -- paged vs ragged baseline, both
      split OFF. Sets the deficit the split must overcome in the default configuration.

### `SGLANG_FLASHINFER_USE_PAGED=1` confirmed working; the 4-mode width bug narrowed (2026-10-09)

`SGLANG_FLASHINFER_USE_PAGED=1` does take effect -- the failure moved from
`prefill_wrapper_ragged.forward_return_lse` to `prefill_wrapper_paged.forward`
(`flashinfer_backend.py:1687`), which is the branch the split dispatch lives in. So the
reachability problem is solved; the 4-mode port's width bug now blocks the gate.

**Narrowing, from three measured deficits.** `qo_indptr[-1] - q.shape[0]` was 2 and 3 on the
ragged path and is **exactly 4 = `num_draft_tokens`** on the paged path -- one verify row's
draft tokens counted twice. Meanwhile `mix_with_verify`'s own seam check reports **0
mismatches**, so at the merge `input_ids == sum(extend_lens) == extend_num_tokens`. The planner
does not read `extend_lens`; it builds

```python
qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)   # spec_info is None branch
```

so the disagreement is in **`extend_prefix_lens` for the verify rows**, introduced between the
merge and `init_forward_metadata`, not in the merge itself. Ruled out along the way:
`generate_attn_arg_prefill` (the spec-aware builder) is NOT in play -- it assumes every row is a
verify row and would yield `bs * draft_token_num`, a multiple of 4, whereas the observed total
2055 is not.

Mechanism consistent with all three numbers: for a verify row `mix_with_verify` sets
`seq_lens = P + n` with `extend_lens = n`, so the prefix must be `P`. A prefix of `P - n` makes
that row contribute `2n` instead of `n`, i.e. a deficit of exactly `n` per affected row -- which
is what the paged path shows with one verify row. Instrumented at the planner's own read site
(commit `723e42343`) to name the stale array rather than infer it; the ragged path's smaller
deficits are consistent with its different indptr construction.

- [x] 2026-10-09 **the split dispatch branch is now reachable** -- `SGLANG_FLASHINFER_USE_PAGED=1`
      moves the extend path into the paged branch, confirmed by the traceback moving to
      `flashinfer_backend.py:1687`.
- [x] 2026-10-09 **4-mode verify-row prefix width -- ROOT CAUSE FOUND AND FIXED**
      (`sglang-up@b8b405822`). Reproduces with the split OFF, so it is a port defect, and it
      blocked every 4-mode serving arm including the split's own gate.

### Root cause: a stale CPU seq_lens mirror, not an off-by-one (2026-10-09)

`ForwardBatch.init_new` passes `extend_prefix_lens = batch.prefix_lens` **directly**
(`forward_batch_info.py:967`). `prefix_lens` is an independent `List[int]`, **not** derived from
`seq_lens - extend_lens` -- a comment in `mix_with_verify` asserted the latter and was wrong.
The attention planner then builds

```python
qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
```

so a prefix that disagrees with `extend_lens` silently resizes the step.

**The measurement that settled it** (planner diagnostic, commit `723e42343`):

```
PLANWIDTH MISMATCH cumsum(seq_lens-prefix)=2055 input_ids=2051 extend_num_tokens=2051
  | tail seq_lens=[2047, 6703] prefix=[0, 6695] qo=[2047, 8] extend_seq_lens=[2047, 4]
```

The verify row claims `qo = 8 = 2n` against `extend_lens = 4`, so its prefix is `n` too small.
Working back: the committed prefix is `6703 - 4 = 6699`, while the observed `6695 = 6696 - 1`
came from `mix_with_running` computing `[s - 1 for s in running_batch.seq_lens_cpu]`. So
**`running_batch.seq_lens_cpu` (6696) is STALE against the device `seq_lens` (6699)** --
`eagle_prepare_for_verify` advances the tensor without refreshing the mirror -- and
`mix_with_verify` captured the *device* value for `seq_lens` while `prefix_lens` came from the
*mirror*. The two disagreed by 3, which also left `seq_lens_cpu` inconsistent with `seq_lens`.

Fix: derive the host copy from the device tensor (`verify_base_seq_lens.cpu()`), which repairs
`seq_lens_cpu` and `prefix_lens` together, and set `prefix_lens[tail]` from it so
`seq_lens - prefix_lens == extend_lens == n` exactly. Costs one D2H of `bs` ints per 4-mode
step, marked `TODO(sync)`.

**RETRACTED (recorded in place, not amended away):** my first fix, committed as `6f6ea6c10`,
set the prefix from `running_batch.seq_lens_cpu` on an inferred "prefix = P-1, deficit = 1 per
verify row" theory, which I had claimed fit all three measured deficits (4, 3, 2). The planner
diagnostic refuted it: the deficit is `n` per verify row, not 1, and the mirror that theory
relied on is the stale array itself. The earlier ragged-path deficits of 2 and 3 are NOT the
same arithmetic -- the ragged wrapper builds its indptr differently -- and forcing one theory
across both paths is what produced the wrong fix. Corrected in `b8b405822`.

Ruled out along the way: the merge itself (its own seam check reports **0** mismatches, so
`input_ids == sum(extend_lens) == extend_num_tokens` there) and `generate_attn_arg_prefill`
(it assumes every row is a verify row and would give a multiple of `draft_token_num`; 2055 is
not).

---

## 2026-10-09 -- the split dispatch PASSES the causal fp32 reference (5/5)

**First actual verification of the dispatch.** Deterministic, so the noise floor is zero -- which
is why this gate works where three server-level ON/OFF attempts returned INCONCLUSIVE or
NOT EXERCISED.

- Code: `sglang-up@b8b405822` (`upstream_port`), pod tree `/dev/shm/w/up` = upstream `b7b2975`
  + the `b7b2975..b8b405822` patch series.
- Harness: `/dev/shm/w/drv/split_unit.py` (fp32 causal reference, computed row by row).
- Command:
  `cd /dev/shm/w && PYTHONPATH=/dev/shm/w/up/python venv/bin/python drv/split_unit.py`
- Env: H200, venv `/dev/shm/w/venv` (torch 2.14.1, flashinfer-python 0.7.0.post1),
  qwen7b shape H_Q=28 H_KV=7 D=128 (GQA 4), bf16 q/kv, page_size 1.
- Metric: scale-normalised `max|o - ref| / max|ref|`, tolerance 0.02, reported for the whole
  tensor AND each row subset separately so a one-sided defect is attributable.

| case | rows | fa3/fa2 | err_all | err_fa3 | err_fa2 | |
|---|---|---|---|---|---|---|
| one chunk + 3 verify rows (4-mode shape) | 4 | 2/2 | 0.00241 | 0.00241 | 0.00048 | PASS |
| short chunk + 5 verify | 6 | 3/3 | 0.00424 | 0.00157 | 0.00424 | PASS |
| two prefills | 2 | 1/1 | 0.00232 | 0.00232 | 0.00016 | PASS |
| decode-like rows | 5 | 1/4 | 0.00341 | 0.00341 | 0.00063 | PASS |
| ragged odd lengths (333/7/11 q, 1777/513/129 kv) | 3 | 1/2 | 0.00323 | 0.00130 | 0.00323 | PASS |

What this validates: the FA3 subset's page-table gather by row, `cache_seqlens`, per-subset
`cu_seqlens_q` and causal flag; the FA2 subset's plan through flashinfer's public
`begin_forward`/`plan` with explicit indptr/indices; and the `index_copy_` scatter combine --
on five shapes including the 4-mode shape and non-round ragged lengths. Both halves are correct
independently, so neither subset masks the other.

What it does NOT validate: `sm_margin` (unset here), the in-server metadata path, or any
performance claim.

### The guard declined 4 of the 5 shapes -- direct evidence for the coverage bound

Measured, not argued: with the natural partition, `fa3=0` on four of five cases. Only the
decode-heavy case (`qo=[64,1,1,1,1]`, `kv=[64,300,500,120,90]`) partitioned two-sided on its
own. The arithmetic is the guard's own: a 2048-token chunk is 16 KV tiles and needs
`decode_rows >= 32`, so the canonical 4-mode shape (one chunk + a handful of verify rows) is
refused. The gate forces a two-sided split for the correctness cases, which is legitimate here
because it tests the dispatch's arithmetic, not the routing policy.

This is the `g` in the `g * a` ceiling, now measured at the shape level rather than inferred
from the TraceLab precedent: **the realistic 4-mode step does not satisfy the guard unless
roughly 32+ spec rows are co-resident.** It strengthens the earlier prediction that the
>=30%/>=10% bar is not reachable under this gating, and makes the guard-OFF arm the single most
informative remaining measurement.

- [x] 2026-10-09 **dispatch PASSES the fp32 causal reference, 5/5** -- see table and command above.
- [x] 2026-10-09 **`sm_margin` sweep: correctness holds at every budget, best 1.224x** -- see below.

## 2026-10-09 -- SM pre-assignment measured: 1.22x on attention, and the bar arithmetic closes

Harness `/dev/shm/w/drv/split_margin.py`, code `sglang-up@b8b405822`, H200, torch 2.14.1,
flashinfer-python 0.7.0.post1, qwen7b shape (GQA 4), bf16, page_size 1, 30 iters after 5 warmup.
Command: `cd /dev/shm/w && PYTHONPATH=/dev/shm/w/up/python venv/bin/python drv/split_margin.py`

Batch: one 1024-token prefill chunk + 40 verify rows (kv 1000..2443). **The guard ACCEPTS it**
(`fa3=1 fa2=40`), which confirms the coverage arithmetic from the other direction: 8 KV tiles
against 40 decode rows gives a budget of 20 >= 8.

Baseline = ONE planned flashinfer wrapper over all 41 rows, i.e. exactly the work the unsplit
path does: **186.7 us**, err 0.00256.

| `sm_margin` | us | vs baseline | err_all | err_fa3 | err_fa2 | |
|---|---|---|---|---|---|---|
| 0 | 153.1 | 1.220x | 0.00236 | 0.00236 | 0.00020 | PASS |
| **8** | **152.6** | **1.224x** | 0.00236 | 0.00236 | 0.00020 | PASS |
| 16 | 152.7 | 1.223x | 0.00236 | 0.00236 | 0.00020 | PASS |
| 32 | 159.6 | 1.170x | 0.00236 | 0.00236 | 0.00020 | PASS |
| 64 | 168.7 | 1.107x | 0.00236 | 0.00236 | 0.00020 | PASS |
| 96 | 195.4 | 0.956x | 0.00236 | 0.00236 | 0.00020 | PASS |

Three results in one run:

1. **Correctness is invariant in the SM budget** -- the error is bit-identical across all six
   margins, as it must be, since `sm_margin` changes launch geometry and not arithmetic. So the
   SM pre-assignment is a free tuning knob, not a correctness risk.
2. **The split is worth 1.22x on the attention step** for a batch the guard accepts, against the
   unsplit single-wrapper path.
3. **Withholding too many SMs loses**: 96 is 0.956x, below baseline. The optimum is a SMALL
   margin (0-16, flat within 0.3%), i.e. FA3 wants nearly the whole GPU and the benefit comes
   from the two executors sharing the grid rather than from a careful SM carve-up. That is a
   substantive finding about the mechanism and it contradicts the "pre-assignment ratio matters"
   framing: the plateau at 0-16 and monotone decay after 32 say the win is co-residency.

### What this does to the acceptance bar -- now arithmetic, not precedent

A 1.22x attention speedup is an **18% reduction in attention time**. With attention a fraction
`a` of step time on accepted batches and the guard accepting a fraction `g` of steps, the
end-to-end ITL gain is `g * a * 0.18`. **Even at `g = 1` and a generous `a = 0.3`, that is 5.4%.**
To reach 30% ITL from attention alone would need `a > 1`, which is impossible.

So the >=30% TTFT/ITL target is **not reachable from this kernel**, and this is now settled by
measurement on our own code rather than inferred from the TraceLab precedent (which independently
put a singly-gated kernel at +0.27% throughput / -4.6% P99 ITL). The earlier 1.30x figure from
the standalone fused harness does not change this: 1.30x is a 23% attention reduction, so the
ceiling moves from 5.4% to 6.9%.

**Where a 30% TTFT gain can come from:** queueing, not attention. The TraceLab numbers in
`realistic_workload_analysis.md` show P99 TTFT of 7.28-33.0 s against a mean of 0.73-1.32 s --
an order of magnitude of pure waiting. That is the EDF/admission-order lever, and
[[edf-order-not-cap]] already records that deadline ordering wins on the MTP server while every
per-step chunk cap loses. The honest project claim is therefore: the fused kernel is a correct,
1.22x attention optimisation with a narrow acceptance region, and the publishable serving win
has to come from the scheduling side.

- [ ] 2026-10-09 **re-point the serving effort at the scheduling lever** -- EDF ordering +
      cache-hit-first routing measured on TTFT percentiles, with the split enabled where the
      guard accepts it. The kernel is no longer the critical path to the bar.

## 2026-10-09 -- the 4-mode step RUNS CLEAN on upstream, and the guard refuses real batches

Same server config as the gate (`drv/split_verify_blk.sh`), `sglang-up@b8b405822` applied to pod
tree `/dev/shm/w/up`:
`--speculative-algorithm EAGLE3 --speculative-draft-model-path /dev/shm/w/qwen7b_eagle3_sgl
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
--attention-backend flashinfer --enable-mixed-chunk --chunked-prefill-size 2048 --page-size 1
--max-running-requests 64`, env `SGLANG_MIXED_SPEC=1 SGLANG_FLASHINFER_USE_PAGED=1
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`, 56 staggered requests.

**Result -- the 4-mode MTP step is working end to end on upstream:**

| metric | value |
|---|---|
| `MIXEDSPEC_STEPS` (4-mode merges) | **151** |
| client errors | **0 / 56** |
| tracebacks | **0** |
| `PLANWIDTH` width mismatches | **0** |
| `fused split partition skipped` | **0** |

So the `prefix_lens` fix (`b8b405822`) cleared the blocker, and the earlier
`device-side assert triggered` in flashinfer's `plan` was a downstream artifact of the bad
widths, not an independent defect -- it disappeared once the widths were right. Recorded as such
rather than left as an open "CUDA assert" item.

**And the split still does not fire, with the server stating the reason:**

```
FUSED_SPLIT_ONESIDED: fa3=0 fa2=7 rows=7 max_kv_tiles=60 need_decode_rows>=120
FUSED_SPLIT_ONESIDED: fa3=0 fa2=1 rows=1 max_kv_tiles=16 need_decode_rows>=32
```

A real merged 4-mode batch here is **7 rows** -- one prefill chunk plus ~6 verify rows -- against
~60 KV tiles, so the ratio guard wants 120 co-resident decode rows and gets 6. **This is `g ~= 0`
measured in the server**, not inferred from the TraceLab precedent or from the harness shapes.
The reason is structural: a verify row contributes `num_draft_tokens = 4` tokens, so the running
batch would need ~30 concurrent spec requests to supply 120 "decode rows", while the merged batch
at this concurrency carries 6.

This is the third independent line of evidence for the same conclusion, now from the server
itself: the kernel is correct and 1.22x faster on batches it accepts, and it essentially never
gets to run in a real 4-mode schedule under the measured boundary.

- [x] 2026-10-09 **4-mode MTP step runs clean on upstream UNDER `CUDA_LAUNCH_BLOCKING=1`** --
      151 merges, 0/56 client errors, 0 tracebacks, 0 width mismatches. The `prefix_lens` fix was
      the blocker.
      **QUALIFIED 2026-10-09:** that run carried `CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1`
      (`drv/split_verify_blk.sh`), which I had added to locate the device assert and did not
      account for when reporting the result. "Runs clean on upstream" unqualified was an
      overstatement on my part. A no-blocking control is required before the claim stands,
      because launch-blocking serialises the stream and would mask a genuine race.
- [ ] 2026-10-09 **no-blocking control for the 4-mode step** -- same config, split effectively
      off (guard ON), without `CUDA_LAUNCH_BLOCKING`. Distinguishes "the width fix made it
      correct" from "launch-blocking hid a race".
- [~] 2026-10-09 **the earlier flashinfer device-side assert** -- it did not recur once the
      widths were fixed, but the run that showed that also had launch-blocking on, so "symptom,
      not defect" is not yet established. Held open pending the control above.
- [x] 2026-10-09 **guard refuses real 4-mode batches** -- `rows=7`, `kv_tiles=60`,
      `need_decode_rows>=120`. `g ~= 0` in serving.
### Guard OFF: the split FIRES on real merged batches, and faulted in my own code

`SGLANG_FUSED_DECODE_GUARD=0` made the partition two-sided on real merged 4-mode batches and the
dispatch executed -- the first time any of this ran in a server. It crashed there:

```
flashinfer_backend.py:1561 in _fused_split_dispatch
  k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
AttributeError: 'ForwardBatch' object has no attribute 'token_to_kv_pool'
```

The KV pool hangs off the **backend** (upstream's own `forward_extend` does
`pool = self.token_to_kv_pool`). Fixed in `sglang-up@41273f944`; the FA2 half now also stops
refetching the cache, since the caller already passes the resolved `kv_cache` -- refetching
would additionally have bypassed the fp4-dequant workspace path.

**Why the standalone fp32 gate did not catch it:** that harness builds its own K/V buffers and
never touches the pool. It validates the dispatch's *arithmetic*, which is what it was scoped
to, and says nothing about in-server plumbing. Worth stating because a 5/5 PASS invites the
inference that the dispatch is correct in the server, and it does not support that.

**Confound I introduced and am recording:** I derived the guard-OFF script from the
launch-blocking one and dropped `CUDA_LAUNCH_BLOCKING` in the same edit, so that run changed two
variables at once (guard OFF *and* blocking off). The crash is attributable anyway -- it is a
Python `AttributeError` in my dispatch, not a CUDA fault -- but the comparison to the 151-merge
run is not clean, and the no-blocking control above is needed regardless.

### The split fired in serving, and then clobbered the shared flashinfer workspace

`FUSED_SPLIT_CALLS 1 (fa3_rows=1 fa2_rows=1 sm_margin=0 onesided=47)` -- the dispatch ran on a
real merged 4-mode batch. The server then died with:

```
flashinfer_backend.py:778 in init_forward_metadata_out_graph
  -> update_single_wrapper -> call_begin_forward -> flashinfer/prefill.py:2496 in plan
     -> prefill.py:1599 in _compute_page_mask_indptr
AcceleratorError: CUDA error: device-side assert triggered
```

**Root cause: a shared flashinfer workspace.** A wrapper writes its scheduler metadata into the
workspace buffer it was constructed with. The split's extra FA2 wrapper was built on
`self.workspace_buffer` -- the same buffer the backend's other wrappers use -- and it plans
**mid-step**, after those wrappers have already been planned for the step, so it overwrote their
plans. The assert is raised later, in an unrelated `plan` call: **the traceback points at the
victim, not the cause.**

What actually identified it was the correlation, not the traceback: the assert appears only in
runs where the split fired (`FUSED_SPLIT_CALLS >= 1`), and with the decode guard ON the split
never fired and 151 merges ran clean. Fixed in `sglang-up@370c511f3` with a dedicated workspace.

**RETRACTION of my own earlier reasoning.** When writing the dispatch I noticed this exact
hazard and dismissed it, on the grounds that upstream also shares one workspace across its
wrappers. That reasoning was wrong: what makes upstream's sharing safe is that it plans all its
wrappers in a fixed order *before any of them runs*. This wrapper plans after that point, so the
sharing is not safe for it. Sharing is not the property that matters; plan-then-run ordering is.

**This also corrects the earlier `CUDA_LAUNCH_BLOCKING` interpretation.** I had suspected
launch-blocking was masking a race. It was not: the blocking run had the guard ON, so the split
never fired and the workspace was never clobbered. Blocking was a confound in my comparison, not
a mask. The no-blocking control is therefore no longer needed for that question -- though the
151-merge clean run should still be reproduced without blocking before being quoted
unqualified.

- [~] 2026-10-09 **the device-side assert is a real defect in the split, NOT yet fixed.**
      **RETRACTED: the shared-workspace diagnosis was wrong.** Giving the FA2 wrapper its own
      workspace (`370c511f3`) did **not** fix it -- the assert reproduced unchanged. The actual
      assert text is from torch, not flashinfer:
      `IndexKernel.cu:111 Assertion -sizes[i] <= index && index < sizes[i] && "index out of
      bounds" failed`, i.e. an out-of-range index in one of the **dispatch's own** indexing ops.
      Such an assert surfaces at whatever CUDA call happens next, which was a flashinfer `plan()`
      several frames away -- so **the traceback named the victim twice, and I mistook the second
      victim for a new cause.**
      Where the reasoning failed: I read "the assert appears only when `FUSED_SPLIT_CALLS >= 1`"
      as evidence for *clobbering*. That correlation only localises the fault to the split; it is
      equally explained by "the split ran and indexed out of bounds". Correlation localised, it
      did not identify the mechanism, and I over-read it -- the same error the memory note I had
      just written warns about.
      `370c511f3` is **kept** on independent grounds: a wrapper planning mid-step into the shared
      buffer is unsound regardless, since upstream's sharing is only safe under plan-then-run
      ordering. It is simply not this bug.
- [x] 2026-10-09 **construction-time bounds checks added** (`c163f427d`) --
      `sum(extend_seq_lens_cpu)` vs `input_ids`, `req_pool_index` vs `req_to_token` rows,
      `max seq_len` vs its columns. **All three PASSED while the assert still fired**, which
      rules out the plan's own arrays and is itself the useful result.
- [x] 2026-10-09 **q-extent and row-count checks** (`3777ee81a`, `056f28667`) -- neither
      fired. Kept as cheap guards; neither was the cause.

### 2026-10-09 -- the split PASSES in-server, and the device assert was a use-after-free

**In-server PASS, guard OFF, under `CUDA_LAUNCH_BLOCKING=1`** (`drv/split_verify_ngblk.sh`,
`sglang-up@056f28667`, same server flags as the 4-mode gate above plus
`SGLANG_FUSED_DECODE_GUARD=0 SGLANG_FUSED_SPLIT_VERIFY=1`):

| metric | value |
|---|---|
| `MIXEDSPEC_STEPS` | 151 |
| `FUSED_SPLIT_CALLS` | **4501** |
| partition on real merged batches | `fa3=2 fa2=25` (two-sided) |
| `FUSED_SPLIT_VERIFY` | **9 ok / 0 FAIL**, err 0.0041-0.0062, worst 0.0078, tol 0.02 |
| client errors | **0 / 56** |

The reference is the ordinary planned wrapper on the same inputs in the same step, so this is
the in-server counterpart of the standalone 5/5 fp32 PASS: the dispatch's plumbing (pool,
`kv_cache`, counters, workspace) is right and the output matches.

**Root cause of the device assert: a use-after-free in my own async copies.** With blocking
ON the split ran 4501 times clean; with blocking OFF and everything else identical it asserted
after **one** call. With the split firing in *both* arms, that difference is decisive. The
mechanism was in `_fused_split_plan`: `tok_idx` and `indptr` did `.to(dev, non_blocking=True)`
from **Python temporaries** (`torch.from_numpy(np.concatenate(...))`, a local `out`). A
non-blocking H2D copy needs its host source alive until it completes; these were freed as soon
as the expression returned, so the copy could read freed memory and land garbage in **index**
tensors -- hence `IndexKernel.cu ... index out of bounds`, raised at whatever CUDA call came next.
Fixed in `sglang-up@f53707f53` by making those copies synchronous (a few KB; restoring async
needs pinned staging buffers owned by the plan).

**REINSTATED, on new evidence: the asynchrony hypothesis I retracted earlier today.** The
retraction was right for the evidence then -- the blocking run it rested on had the decode guard
ON, so the split never fired and the comparison was vacuous. Only once the split fired in both
arms did blocking become decisive. Recorded in that order deliberately: a retraction that was
correct on its evidence and a reinstatement on better evidence are both part of the trail.

- [x] 2026-10-09 **split PASSES in-server under launch-blocking** -- 4501 calls, 9/0 verify,
      0/56 errors.
- [x] 2026-10-09 **device assert root-caused** -- use-after-free in non-blocking H2D copies from
      temporaries (`f53707f53`).
- [x] 2026-10-09 **split is CORRECT in-server without launch-blocking** -- see below. The
      remaining defect is a CUDA-graph interaction, isolated by environment bisect.

### 2026-10-09 -- the blocker is a CUDA-graph interaction, found by bisecting the ENVIRONMENT

Five consecutive code-level mechanism hypotheses were wrong (shared flashinfer workspace;
launch-blocking masking a race; plan-copy use-after-free; FA2 plan-array lifetime; a
stride-ignoring KV reshape). Four were real latent bugs and their fixes are kept on independent
grounds; none was this fault. All five were inferred from a traceback that, for an **async
device-side assert, carries no location information** -- so each inference was unfalsifiable by
construction. Two of them I reported as root causes before confirming, which was wrong of me.

What produced the answer was a one-bit **environment** bisect, not more code reading.

**`--disable-cuda-graph`, split ON, guard OFF, NO launch-blocking** (`drv/split_nocg.sh`,
`sglang-up@69595708c`):

| metric | value |
|---|---|
| `MIXEDSPEC_STEPS` | 151 |
| `FUSED_SPLIT_CALLS` | **5001** |
| `FUSED_SPLIT_VERIFY` | **10 ok / 0 FAIL**, err 0.0035-0.0066, worst **0.0112** (tol 0.02) |
| client errors | **0 / 56** |

So the dispatch is correct in a real 4-mode server, with no launch-blocking crutch. The fault is
specifically the CUDA-graph path, which the misread trace had been pointing at for several
cycles: `eagle_prepare_for_verify` runs BEFORE the merged forward, so the crash was never
downstream of the dispatch within a step -- it was the NEXT iteration, entered through
`decode_cuda_graph_runner.load_batch` -> `init_forward_metadata_out_graph`.

**Concrete fix direction (not yet implemented).** `_fused_split_plan` allocates fresh index
tensors every step; upstream preallocates persistent buffers at init (`self.kv_indptr`,
`self.qo_indptr`, `self.cuda_graph_kv_indices`) and hands out slices. A captured CUDA graph
holds absolute addresses, so per-step allocations interleaved with replays can collide with
graph-visible memory -- which also explains the timing dependence and why launch-blocking
appeared to "fix" it. The fix is to preallocate, the way upstream already does, and that is the
sixth instance today of the same class: **a call that looks equivalent to upstream's but drops
an invariant upstream's own machinery maintains.** The others were plan-then-run ordering for a
shared workspace, process-lifetime buffers for `non_blocking=True`, and stride-aware paging for
the KV view. "Same shape" is not "same contract".

- [x] 2026-10-09 **CUDA-graph interaction isolated** -- graphs off: 5001 calls, 10/0 verify,
      0/56 errors, no blocking. Graphs on: device assert.
- [ ] 2026-10-09 **preallocate the plan's index buffers** -- removes the per-step allocations
      that collide with captured graphs; required before the split can run in the production
      (graphs-on) configuration.
### 2026-10-09 -- first serving A/B. Headline number NOT TRUSTED: arm order is confounded

`sglang-up@69595708c`, pod tree `/dev/shm/w/up`, driver `drv/split_perf_mtp.sh`, raw results
`/dev/shm/w/split_perf_run1_UNWARMED.txt`. Server: 4-mode MTP + EAGLE3 chain draft (topk=1,
4 draft tokens), `--attention-backend flashinfer --enable-mixed-chunk --chunked-prefill-size
2048 --page-size 1 --max-running-requests 64 --disable-cuda-graph --mem-fraction-static 0.75`,
env `SGLANG_MIXED_SPEC=1 SGLANG_FLASHINFER_USE_PAGED=1
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`. Workload: `bench_serving --dataset-name random
--random-input-len 2048 --random-output-len 512 --random-range-ratio 0.5 --num-prompts 300
--request-rate 12 --max-concurrency 56`.

| arm | split calls | req/s | tok/s | Mean TTFT | P99 TTFT | Mean ITL | P99 ITL |
|---|---|---|---|---|---|---|---|
| base (split off) | 0 | 6.20 | 12087 | **337.95** | **2613.91** | 18.68 | 73.72 |
| guard_on | 7501 | 6.19 | 12074 | 189.76 | 698.18 | 18.87 | 63.92 |
| noguard_m0 | 8001 | 6.17 | 12035 | 187.70 | 695.36 | 18.93 | 63.45 |
| noguard_m8 | 8001 | 6.16 | 12017 | 192.17 | 715.10 | 18.96 | 63.53 |

Read naively this is -44% mean TTFT and -73% P99 TTFT, i.e. clearing the 30% bar.
**It should not be read that way.** `base` ran FIRST and so was the only arm paying flashinfer
JIT compilation and cache warmup. A 2613.9 ms P99 TTFT against ~700 ms for every later arm is
far larger than any plausible attention effect and carries the signature of a one-off compile.
**Arm order is confounded with the treatment**, so the TTFT column measures warmup, not the
split, and the honest status of the 30% TTFT bar remains UNMET pending a clean run.

Control now running: a discarded warmup pass before each arm's measurement, and the arm order
**reversed** so `base` runs LAST -- any residual order effect then works against the split
rather than for it, and a surviving gain cannot be an ordering artifact.

**What the run does establish, since these are within-run comparisons unaffected by the warmup
of the first arm:**

1. **Throughput is flat**: 12087 / 12074 / 12035 / 12017 tok/s, a monotone -0.6% across the
   three split arms. The >=10% throughput bar is not met **by the kernel, on this baseline**.
   **SCOPE CORRECTION 2026-10-09 (user):** this baseline is unfused *colocation on one GPU*, so
   it measures the attention kernel and is **the wrong axis for the project's throughput claim.**
   That claim is about colocation versus a FIXED PD split (1P3D / 2P2D): every node can admit
   both prefill and decode with cache replication, so a disagg partition strands capacity
   whenever the mixture moves off the ratio it was provisioned for. A colocated-vs-colocated A/B
   cannot show that, and I should not have read "flat tok/s here" as bearing on it. The capacity
   experiment is now item 5 of `tasks/pd_colocation_vs_disagg.md`: goodput under per-class SLOs,
   swept over mixture ratios and rates, plus load spikes and mid-run mixture shifts.
2. **Mean ITL is flat-to-slightly-worse** (18.68 -> 18.87/18.93/18.96, about +1.3%), while
   **P99 ITL improves ~13-14%** (73.72 -> 63.45..63.92). Better tail, unchanged mean -- the same
   shape as the TraceLab precedent (+0.27% throughput, -4.6% P99 ITL), and the P99 arms are
   mutually consistent so this is not a warmup artifact.
3. **My `g ~= 0` claim was WRONG and is retracted.** The guard-ON arm fused **7501** times here,
   only 6% fewer than guard-OFF's 8001. I had inferred near-zero coverage from the staggered
   56-request gate, where merged batches were ~7 rows; at 2048-token inputs with 56 concurrent
   requests there are ample co-resident decode rows and the guard admits fusion readily.
   Coverage is workload-dependent and I over-generalised from one shape. The `g * a` ceiling
   argument still holds, but `g` is near 1 here, not near 0 -- which makes the flat throughput
   a cleaner refutation of the kernel route than the coverage argument was.
4. **`sm_margin=8` does not transfer from microbenchmark to server**: 12017 tok/s vs 12035 at
   margin 0, and slightly worse TTFT/ITL. The kernel harness put margin 8 at 1.224x vs margin
   0's 1.220x -- a 0.3% difference that does not survive contact with a real schedule. Use
   margin 0.
5. **Accept length is identical (2.06) in all four arms**, confirming the split does not perturb
   speculative acceptance -- a necessary condition for the arms to be comparable at all.

- [x] 2026-10-09 **throughput bar NOT met** -- flat to -0.6% across split arms, `g ~= 1`.
- [x] 2026-10-09 **P99 ITL improves 13-14%**, mean ITL flat; 30% ITL bar not met.
### 2026-10-09 -- WARMED, REVERSED-ORDER A/B: the split is a net REGRESSION in serving

Driver `drv/split_perf_mtp.sh` with a discarded warmup pass per arm and the arm order reversed
(`base` LAST), `sglang-up@69595708c`, raw results `/dev/shm/w/split_perf_mtp_results.txt`.
Same server/workload as the unwarmed run. Arms are listed in execution order.

| arm (exec order) | split calls | tok/s | Mean TTFT | P99 TTFT | Mean ITL | P99 ITL |
|---|---|---|---|---|---|---|
| noguard_m8 | 8001 | 12084.54 | 137.14 | 242.30 | 18.97 | 63.45 |
| noguard_m0 | 8001 | 12361.23 | 140.70 | 256.80 | 18.56 | 62.92 |
| guard_on | 7501 | 12369.17 | 137.37 | 238.96 | 18.53 | 62.85 |
| **base (split OFF)** | 0 | **13276.37** | **131.72** | 244.44 | **17.28** | **62.15** |

**The split LOSES on every metric once the warmup confound is removed:**

| metric | base | best split arm | delta |
|---|---|---|---|
| total tok/s | 13276.37 | 12369.17 | **-6.8%** |
| Mean ITL | 17.28 | 18.53 | **+7.2% worse** |
| Mean TTFT | 131.72 | 137.37 | **+4.3% worse** |
| P99 ITL | 62.15 | 62.85 | +1.1% worse |

**RETRACTED in full: every favourable reading from the unwarmed run.** The "-44% mean / -73% P99
TTFT" was warmup, as suspected and flagged. But the "**P99 ITL improves 13-14%**" claim, which I
had explicitly argued was *safe* because the three split arms agreed with each other, is **also
wrong**: base's warmed P99 ITL is 62.15 ms, better than every split arm. Mutual agreement among
the treated arms said nothing, because all three shared the same confound relative to an
unwarmed control. That reasoning was faulty and the conclusion drawn from it is withdrawn.

So the honest result: **a kernel that is 1.22x faster on isolated attention is a 6.8% throughput
REGRESSION in the 4-mode server**, at `g ~= 1` coverage (7501-8001 fused calls per arm). The
kernel-level speedup does not survive contact with the serving loop.

Most likely causes, in order of suspicion, none yet measured:
1. **Per-step host cost in `_fused_split_plan`**: fresh allocations every step, numpy
   concatenations, three indptr builds, a per-row `req_to_token` gather and an extra flashinfer
   `begin_forward` -- on the critical path, once per step, and the reason upstream preallocates.
   This is the same per-step allocation pattern already implicated in the CUDA-graph fault.
2. **Extra kernel launches**: two attention calls plus two `index_copy_` scatters per layer
   instead of one call, ~28x per step.
3. **`--disable-cuda-graph` on both arms** removes graph capture, which amortises exactly this
   kind of launch and host overhead -- so this configuration is maximally unfavourable to the
   split and the regression may be smaller (or absent) with graphs on. That cannot be tested
   until the preallocation fix lands.

- [x] 2026-10-09 **TTFT/ITL/throughput bars: NOT met; the split REGRESSES -6.8% tok/s** in the
      warmed, order-reversed A/B.
### 2026-10-09 -- preallocation did NOT fix the regression; and a run-to-run noise floor

`sglang-up@b9e3e4b8c` routes every per-step index array through a pinned grow-only pool (token
indices, three indptrs, FA3 row ids, the page-table gather via `index_select(out=)`,
`cache_seqlens` from the already-materialised `seq_lens_cpu`, and the FA2 `kv_indices` loop
replaced by one flat `index_select`). Correctness first: in-server verify **PASS**, 5001 split
calls, 10 ok / 0 FAIL, worst err 0.0108 (tol 0.02), 0/56 client errors.

Warmed, order-reversed A/B, graphs off -- identical config to the run that gave -6.8%, so
preallocation is the only variable. Raw: `/dev/shm/w/split_perf_mtp_results.txt`; prior run kept
as `split_perf_run2_PREALLOC_BASELINE.txt`.

| arm | tok/s | Mean TTFT | P99 TTFT | Mean ITL | P99 ITL |
|---|---|---|---|---|---|
| noguard_m8 | 12461.20 | 135.35 | **234.78** | 18.36 | 62.62 |
| noguard_m0 | 12257.31 | 137.79 | 235.92 | 18.73 | 63.17 |
| guard_on | 12165.66 | 136.46 | 241.37 | 18.86 | 63.35 |
| **base (split OFF)** | **13110.83** | **135.13** | 280.61 | **17.47** | **62.30** |

Best split arm is **-5.0% tok/s** and **+5.1% mean ITL** against base. **The split is still a
regression.**

**User 2026-10-09: "-5 to -8.4% is not acceptable."** Recorded here so this number is never
carried forward as a result. It is a regression, and the kernel items are not closeable until
an in-server measurement clears the +/-1.5-3% noise floor on the UPSIDE.

**A noise floor, finally measured, and it changes how the earlier numbers should be read.**
Same config, same code, arm by arm across the two runs:

| arm | run 2 tok/s | run 3 tok/s | delta |
|---|---|---|---|
| base | 13276.37 | 13110.83 | -1.2% |
| noguard_m8 | 12084.54 | 12461.20 | **+3.1%** |
| noguard_m0 | 12361.23 | 12257.31 | -0.8% |
| guard_on | 12369.17 | 12165.66 | -1.6% |

Movements of -1.6% to +3.1% in **both directions** with nothing changed but the preallocation.
So the run-to-run noise floor on this workload is roughly **+/-1.5-3%**, which means:
(a) preallocation's apparent 1.8pp recovery (6.8% -> 5.0%) is **within noise and cannot be
claimed**; and (b) any future gain smaller than ~3% on this harness is unmeasurable, so a
"parity" result must not be read as success. I should have established this floor before
reporting the first A/B at all -- [[gate-needs-noise-floor-and-reference]] says exactly this and
I applied it to correctness gates while leaving the perf harness without one.

The -5% regression is larger than the floor, so it is real.

- [x] 2026-10-09 **preallocation: correctness PASS, perf effect WITHIN NOISE** -- the regression
      persists at -5.0% tok/s. Not the mechanism.
- [x] 2026-10-09 **perf noise floor measured: ~+/-1.5-3% tok/s** on this harness, from two
      identical runs.
- [x] 2026-10-09 **graphs-ON: STILL BROKEN.** Identical fault
      (`IndexKernel.cu:111 index out of bounds` via `flashinfer/prefill.py plan ->
      _compute_page_mask_indptr`), 1 split call then 56/56 client errors. So **preallocation
      achieved NEITHER of its two stated purposes**: the perf effect was within noise, and it did
      not fix the CUDA-graph fault it was primarily written for. Recorded as a double negative
      rather than as progress. `b9e3e4b8c` is kept -- the per-step allocations it removed were
      genuinely on the critical path -- but it fixed nothing measurable.

### 2026-10-09 -- the two-executor split is LAUNCH-BOUND, and that reframes the kernel work

Pulling the measurements together rather than guessing again:

| evidence | value |
|---|---|
| isolated attention, fused harness (single fused kernel) | **1.30x** |
| isolated attention, two-executor split (`split_margin.py`) | **1.224x** |
| in-server, two-executor split, graphs off, warmed | **-5.0% tok/s** |
| coverage in-server | `g ~= 1` (7501-8001 fused calls/arm) |
| perf noise floor on this harness | +/-1.5-3% |
| host-side per-step cost removed by preallocation | effect **within noise** |

The split wins on isolated attention and loses in the server at near-full coverage, and removing
**all** per-step host work changed nothing outside the noise floor. What remains is the
**per-layer launch structure**: the split issues two attention kernels plus two `index_copy_`
scatters per layer where base issues one kernel -- about 4x the launches, ~28 times per step. That
cost is proportional to layers and cannot be amortised by preallocation, only by graph capture
(currently blocked) or by **not launching twice**.

**Consequence for the goal.** The goal asks primarily for a *fused* kernel -- FA3 and FA2 sharing
ONE grid with an SM partition -- and names "a fa3 + fa2 pdl split kernel" as also acceptable. The
two-executor split is the *acceptable fallback*, and it is the variant whose cost structure the
serving loop punishes: separate launches, separate tails, a scatter per subset. The **single
fused kernel already measures better in isolation (1.30x vs 1.224x) precisely because it has one
launch and one tail**, and it is the variant that cannot be launch-bound by construction.

So the route to a real serving gain is the fused single-kernel path
(`tests/fused_fa3_persistent_ext.cu`, `SGLANG_FUSED_KERNEL`), which already PASSES the causal
fp32 reference at every `sm_split` (0/512 unwritten; FA3 0.00628, task0 0.00250, tile16 0.00071)
and times 133.5 us vs 174.0 us unfused on identical work. That is a kernel-level result; what has
never been done is wiring it into `forward_extend` the way the two-executor split now is.

- [ ] 2026-10-09 **wire the single fused kernel into upstream `forward_extend`** -- one launch,
      SM pre-assignment inside the grid, no scatter. This is the variant the goal actually asks
      for and the only one whose cost structure can plausibly beat base in-server.

### Integration surface of the fused kernel, read off the source (2026-10-09)

Established from `/dev/shm/w/fi/tests/fused_fa3_persistent_ext.cu` and
`tests/verify_fused.py`, so the wiring is specified against the real signature rather than from
memory.

**Entry point.** `torch.ops.fused_fa3.fused_run(...)`, registered by a
`torch.utils.cpp_extension.load(name="fused_fa3", sources=[fused_fa3_persistent_ext.cu], ...)`.
There is no prebuilt `.so` on the pod; it JIT-builds. Include paths needed:
`<fi>/include`, `<fi>/csrc`, `<fi>/3rdparty/cutlass/include`,
`<fi>/3rdparty/cutlass/tools/util/include`, `<fi>/3rdparty/spdlog/include`, **plus** a
flashinfer-generated `batch_attention_with*` config directory from
`~/.cache/flashinfer/*/generated/` -- which only exists after some `BatchAttention.plan()` has
run, so the build has an ordering dependency on flashinfer's own JIT.
The harness passes `-std=c++17`; **torch 2.14 requires C++20**, already hit and fixed once in
this project, so that flag must be raised for this toolchain.

**Signature** (tail of the argument list, in order):

```
fused_run(q, ..., fa3_int_ws, std::vector<int64_t> fa3_plan,
          pers_int_ws, pers_float_ws, std::vector<int64_t> pers_plan,
          num_qo_heads, num_kv_heads, page_size, sm_scale,
          sm_split, pers_row_offset, fa3_num_ctas, pers_kv_page_offset,
          fa3_kv_indices, pers_o, num_requests, max_pages_per_seq,
          max_seqlen_q, kv_lens_per_req, q_indptr_per_req, kv_indptr_per_req)
```

**What this means for the port, and it is more than a call swap:**

1. **It needs BOTH flashinfer plan vectors**, not one: `fa3_plan` is a
   `PrefillPlanSM90Info`, `pers_plan` a `HolisticPlanInfo<3>` or `<4>`. So `forward_extend` must
   plan the FA3 paged wrapper *and* the persistent `BatchAttention` wrapper each step and extract
   both `_plan_info` vectors. The two-executor split already plans one extra wrapper per step, so
   this is the same per-step planning cost -- the saving is strictly in the LAUNCH, which is
   exactly the term the serving measurements implicate.
2. **`sm_split` is the SM pre-assignment**, passed directly into the one grid. This is the
   genuine "SM pre-assignment" the goal asks for; `sm_margin` in the two-executor split only
   withheld SMs from FA3's own launch.
3. **Index representation differs from the split's.** FA3 inside the fused kernel multiplies the
   index by `k_stride_n`, so `fa3_kv_indices` must be **vector-sparse offsets** via
   `flashinfer.page.block_sparse_indices_to_vector_sparse_offsets`, **not** the `req_to_token`
   page ids the two-executor split feeds `flash_attn_with_kvcache`. This is precisely the
   distinction `mixed_dispatch.dense_block_table`'s docstring was written for, and aliasing the
   two caused an illegal memory access earlier in this project.
4. **Plan-length guard**: under `FLASHINFER_FUSED_UNIFIED_WGMMA` the kernel requires a 4-task
   plan and checks the vector's length, because `FromVector` walks by the count baked into the
   type. The real lengths are 49 and 60 entries (from `ToVector()`), not the 51/62 that
   `NUM_TASK_ARGS`/`NUM_SHARED_ARGS` arithmetic predicts -- the source comment records that the
   arithmetic form of this guard rejected every valid plan.
5. Reusable as-is from the split work: the row partition (`split_classes` + decode guard), the
   per-request varlen builders (`fa3_subset_varlen`), and `dense_block_table` for the dense
   `[batch, max_pages]` table.

- [x] 2026-10-09 **increment 1 PASS: `fused_fa3` builds from inside sglang and registers its
      op.** `sglang-up@9da582ac3`
      (`python/sglang/srt/layers/attention/fused_fa3_loader.py`). Gate
      `drv/test_loader.sh`; command
      `PYTHONPATH=/dev/shm/w/up/python SGLANG_FUSED_FI_ROOT=/dev/shm/w/fi
      /dev/shm/w/venv/bin/python -c "...load_fused_fa3(verbose=True)"`. Output:
      `generated configs before: 0` -> `after: 1` (the loader drove a minimal
      `BatchAttention.plan()` itself, so the flashinfer-JIT ordering dependency is handled),
      linked `fused_fa3.so`, `HAS_OP: True`, `INCREMENT1_PASS`. Generated config used:
      `~/.cache/flashinfer/90a/generated/batch_attention_with_kv_cache_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_qk_128_head_dim_vo_128_posenc_0_use_logits_soft_cap_false_use_profiler_false_flipped_schedule_false`.
      Built with **C++20** (the harness's `-std=c++17` does not compile against torch 2.14).
### The registered `fused_run` schema, and the harness is STALE against it (2026-10-09)

Taken from the loaded op rather than from the source or the harness, so it is authoritative:

```
fused_fa3::fused_run(
  Tensor q, Tensor k_cache, Tensor v_cache, Tensor kv_indices, Tensor o,
  Tensor fa3_int_ws, int[] fa3_plan,
  Tensor pers_int_ws, Tensor pers_float_ws, int[] pers_plan,
  int num_qo_heads, int num_kv_heads, int page_size, float sm_scale,
  int sm_split, int pers_row_offset, int fa3_num_ctas, int pers_kv_page_offset,
  Tensor fa3_kv_indices, Tensor pers_o,
  int num_requests, int max_pages_per_seq, int max_seqlen_q,
  Tensor kv_lens_per_req, Tensor q_indptr_per_req, Tensor kv_indptr_per_req) -> ()
```

**26 arguments. `tests/verify_fused.py` passes 20** -- it stops after `pers_o` and predates the
six trailing `UnifiedWgmma` plan params. So the harness **cannot run against the current
extension**; it would fail on arity.

Two consequences, and the second is a correction to this doc:

1. **The six trailing args are exactly what commits 63/64 were built to produce** --
   `num_requests`, `max_pages_per_seq`, `max_seqlen_q` plus the per-request
   `kv_lens_per_req` / `q_indptr_per_req` / `kv_indptr_per_req` arrays. `dense_block_table`
   (dense `[batch, max_pages]`, -1 filled) and `fa3_subset_varlen` (densely renumbered
   per-request varlen arrays, CPU-verified 11/11 and 10/10) supply them directly. The
   upstream-facing data path built for the two-executor split is reusable here, which is the one
   piece of leverage carried over.
2. **The standing "fused kernel PASSES, 1.30x" result is not reproducible as recorded.** Those
   numbers (133.5 us vs 174.0 us unfused; fixture 1 green at sm_split 48/32/16; 0/512 unwritten)
   came from the harness, which can only have been run against an **older** extension than the
   one now on the pod. The result is not withdrawn -- it was measured -- but it is **no longer
   verifiable with the current source**, and re-validating it requires updating the harness to
   the 26-arg schema. That must happen before the 1.30x figure is used to justify preferring the
   fused variant, since that justification is currently resting on a number I cannot re-run.

Also recorded from the harness, an integration constraint that is easy to miss: the FA3 wrapper
**must be planned for exactly `sm_split` CTAs** -- `FLASHINFER_FA3_MAX_CTAS` has to be set to the
same value before `fa3.plan(...)`, and the launch refuses a mismatch by design. And the
persistent side is a `flashinfer.attention.BatchAttention` (exposing `int_workspace_buffer`,
`float_workspace_buffer`, `_plan_info`), while the FA3 side is a
`BatchPrefillWithPagedKVCacheWrapper(..., backend="fa3")` (exposing `_int_workspace_buffer`,
`_plan_info`).

### 2026-10-09 -- increment 2a BLOCKED: the fused kernel is bound to flashinfer 0.3.1

**Root cause, and it subsumes every "staleness" found today as a single fact:**

| component | version |
|---|---|
| flashinfer **source tree** `/dev/shm/w/fi` -- the headers the kernel compiles against | **0.3.1** |
| flashinfer **installed wheel** -- what sglang and `BatchAttention` actually use | **0.7.0.post1** |

The kernel's `HolisticPlanInfo` serialises to **56 or 68** entries (its own runtime guard says
so); flashinfer 0.7's `BatchAttention._plan_info` is **28** entries. Every case errors with
`persistent plan has 28 entries, expected 56 or 68`. These are four minor versions apart, so the
plan layout, not a default, is incompatible.

The six incompatibilities hit in sequence are all symptoms of that one gap:

1. `fused_run` takes 26 args; the harness passed 20 (pre-`UnifiedWgmma`).
2. `BatchAttention.__init__` is now `(kv_layout, device)`; the harness passed a workspace tensor.
3. `--fi-root` defaults to the local `/opt/tiger/flashinfer`, absent on the pod.
4. The harness's own `build_ext` still passes `-std=c++17`, which dies in preprocessing against
   torch 2.14 (not disk: `/tmp` had 3.3 TB free, 1% inodes). Worked around by preloading via
   sglang's own C++20 loader, which builds fine.
5. `flashinfer.page.block_sparse_indices_to_vector_sparse_offsets` was **removed** in 0.7
   (`vector_sparse` appears nowhere in the package). At `page_size == 1` it is the identity, so
   `idx` can be passed directly -- but my earlier integration spec named that helper as a
   requirement, which is now wrong.
6. **The plan-layout mismatch above, which is not workaroundable.** Note the `.cu` comment I
   recorded as authoritative ("real lengths are 49 and 60") is itself stale against its own
   compiled types, which now report 56/68.

**Consequences, which change the feasible scope and should not be worked around quietly:**

* **The standing "fused kernel PASSES, 1.30x at sm_split 112" result was measured on flashinfer
  0.3.1.** It is not withdrawn as a historical measurement, but it says nothing about the current
  stack, and I was wrong to cite it earlier today as support for preferring the fused variant.
* **The fused single kernel cannot be wired into the current server at all** without porting it
  to flashinfer 0.7's `HolisticPlanInfo`. The server runs 0.7 (upstream sglang depends on 0.7
  APIs -- `kv_loc_plan`, the new `BatchAttention` signature), the kernel needs 0.3.1, and two
  flashinfers in one process is not viable. So increments 2b and 3 as written are blocked, not
  merely unstarted.
* The only kernel variant that runs on the current stack is the **two-executor split**, which is
  verified correct (standalone 5/5 fp32; in-server 5001 calls, 10/0 verify) and measures
  **-5.0% tok/s** in serving.

- [x] 2026-10-09 **increment 2a: BLOCKED, root-caused.** Harness ported through five
      incompatibilities (`drv/patch_harness.py`, `drv/run_verify.sh`); the sixth is a hard
      version gap. 32/32 cases error on the plan-length guard.
### 2026-10-09 -- the live hypothesis for the regression: BATCH SHAPE, not overhead

The user's objection -- a -5..-8% regression is impossible if FA3 beats baseline on the prefill
rows and our FA2 *is* the baseline kernel on the decode rows, with no tile waste -- is correct,
and it forced the right comparison:

| | FA3 rows | FA2 rows | result |
|---|---|---|---|
| isolated harness (`split_margin.py`) | 1 | **40** | **1.224x** |
| real 4-mode server, MC=56 | 1 | **7** | **-8.4% tok/s** |

**The 1.22x was measured at a batch shape the server never produces.** At 7 verify rows the FA2
half is 7x4 = 28 tokens -- a launch that is almost entirely fixed cost -- plus two `index_copy_`
scatters, against the baseline's single kernel over all 8 rows. No tile waste is involved; the
split's *fixed* costs simply exceed the FA3 win at that shape. This is a methodology error on my
part: I validated the kernel at a shape I chose rather than the shape the 4-mode schedule emits,
then spent two measurement cycles attributing the gap to per-step overhead (preallocation, then
the redundant full-batch plan) without first checking shape agreement, which was cheaper than
either.

Supporting measurements, warmed and order-reversed, `base` last, graphs off, `sglang-up@cff63ef7e`:

| arm | tok/s | Mean ITL | notes |
|---|---|---|---|
| base | **13216.1** | **17.37** | split off |
| guard_on | 12111.7 | 18.92 | -8.4% |
| skipplan | 12105.8 | 18.94 | **uninterpretable, see below** |
| noguard_m0 | 12089.3 | 18.96 | -8.5% |

* **`skipplan` is uninterpretable, not a null result.** `base_plan_skipped=n/a`: the driver greps
  the server log for a counter line the code never emitted, so the arm cannot be distinguished
  from the knob never firing. Same "assert the gate exercised the path" failure as
  [[gate-needs-noise-floor-and-reference]], repeated on a *perf* knob after being fixed for
  correctness gates. Log line added (`sglang-up@HEAD`).
* The redundant full-batch plan is still a genuine waste worth removing -- the split planned TWO
  flashinfer wrappers per step where the baseline plans one -- but its effect has not been
  measured.

- [ ] 2026-10-09 **re-run the A/B at high concurrency (`MRR=160 MC=144 RR=36 NP=600`)** to test
      whether the split wins once the batch supplies ~40 verify rows. Launched, then **lost when
      the pod expired mid-run**; the driver is now parameterised (`MRR/MC/RR/NP`) and reports
      `last_fa2_rows` so the shape is recorded beside the timings rather than inferred.
- [ ] 2026-10-09 **if the split needs a decode-rich batch, that is a SCHEDULING result, not a
      kernel one** -- it would mean the kernel's benefit is contingent on admission policy
      (how many spec rows are co-resident), which is the EDF/admission lever, and it should be
      stated that way rather than as a kernel speedup.

### 2026-10-09 -- pod expired; trial re-forked

Trial `6ca732428908993c` reached `done` mid-run, taking the high-concurrency A/B and all of
`/dev/shm/w` (tmpfs) with it. `{"error": "EOF"}` from `execute-script` is identical for an
expired trial and a network fault, so `probe_trial()` was checked first per the note in `px.py`.

Re-forked to **`f929493da6e45f10`**, named `vibecuda`, per the goal and the standing
`re-fork H200 OK (1 GPU)` authorization. Operational note for the restore recipe:

```
merlin-cli --control-plane i18n-bd job-v2 runs fork \
  --json '{"source_sid":"<src>","name":"vibecuda","overrides":{"attachments":[]}}'
```

`overrides.attachments: []` is **required** -- without it the fork fails with
`field "attachments" must be array`, because the source config's value does not round-trip
through `job-runs/create`. Strip proxy vars first (`http_proxy`/`https_proxy` and the uppercase
pair) or merlin-cli fails with a bare EOF.

### 2026-10-09 -- retargeting the build at 0.7 headers FAILS; the port is genuine C++ work

I tried the cheap remedy first: compile the kernel against the installed wheel's bundled headers
(`flashinfer/data/include`, which ships `attention/persistent.cuh`, `scheduler.cuh`, cutlass,
spdlog and cccl) so its plan layout would match the running flashinfer by construction, with the
0.3.1 source tree appended only for this project's own `persistent_fa3_fused.cuh`. That was worth
one compile and it failed with specific, non-cosmetic errors:

```
hopper/variant_helper.cuh(54): error: class "AttentionVariantBase" has already been defined
                               (previous definition at variant_helper.cuh:75)
hopper/variants.cuh(168):      error: "DefaultAttention" has already been declared in this scope
ext.cu(32):                    error: DefaultAttentionSm90 is not a template
ext.cu(75):                    error: "StandardAttention" is ambiguous
ext.cu(342-345):               error: HolisticPlanInfo<3> has no member
                                 group_id_offset, group_nchunks_offset,
                                 group_row_begin_offset, group_nrows_offset
```

**The last four are decisive.** 0.7's `HolisticPlanInfo` has not merely changed length -- it has
**lost the four group tables** (`group_id`, `group_nchunks`, `group_row_begin`, `group_nrows`)
that the kernel's **just-in-time reduction** indexes. Those tables are the mechanism by which
split-KV partials are combined; the kernel cannot be compiled against 0.7 without reconstructing
that information from whatever 0.7 provides instead. On top of that, 0.7's own
`variant_helper.cuh` and `hopper/variant_helper.cuh` now collide when both are on the include
path, and two variant types were renamed or made non-template.

So **the header-retarget hypothesis is refuted, and my original assessment was right in
substance**: the fused single kernel needs a real port to flashinfer 0.7 -- reworking the JIT
reduction against a changed plan struct, plus the variant-type changes -- not a build-flag fix.
Recorded so the cheap option is not retried.

- [x] 2026-10-09 **build against 0.7 headers: REFUTED.** Four missing plan fields, one header
      collision, two variant-type changes. `sglang-up@ee07b741b` + `5f486a485` keep the
      retargeting logic (it is the right default if the kernel is ever ported) but it does not
      compile today.
- [ ] 2026-10-09 **port the fused kernel to flashinfer 0.7** -- reconstruct the JIT-reduction
      group tables from 0.7's plan, resolve the variant-type changes, and avoid the
      variant_helper collision. Substantial C++ work against flashinfer internals; the only route
      to the one-launch cost structure.
- [x] 2026-10-09 **DECISION TAKEN (user): PORT IT, and it must land on UPSTREAM.** Not the
      fork, and not dropped. The user's reasoning: *"yes. maybe after you port it will be
      faster."* -- i.e. the port is not merely a compatibility chore, since 0.7's reworked
      reduction may itself be better than 0.3.1's. This closes the fork-vs-upstream question
      that was reopened by the version gap: the answer is the same as it was for the 4-mode
      step, upstream, with no fork demo as a fallback. Tracked as **K1**. This is C++ work against flashinfer internals
      (plan serialisation, the 4-task `UnifiedWgmmaTileScheduler` path), not a wiring task, and
      it is the only route to the one-launch cost structure that the serving measurements say is
      needed. Flagging rather than silently choosing: the alternative is to accept the
      two-executor split's -5.0% and pursue the gains on the scheduling axis instead.
- [ ] 2026-10-09 **increment 2b: build both plan vectors in `init_forward_metadata`** --
      `BatchAttention` for `pers_plan`, `BatchPrefillWithPagedKVCacheWrapper(backend="fa3")` for
      `fa3_plan` with `FLASHINFER_FA3_MAX_CTAS == sm_split`, plus vector-sparse
      `fa3_kv_indices` via `block_sparse_indices_to_vector_sparse_offsets`, and the six
      UnifiedWgmma args from `dense_block_table` / `fa3_subset_varlen`.
- [ ] 2026-10-09 **increment 3: one `fused_run` call in `forward_extend`**, gated, then the
      in-step reference gate and the warmed A/B against base -- which must clear the measured
      +/-1.5-3% noise floor to count as a gain.
- [ ] 2026-10-09 **diagnose the graphs-on fault by capture-boundary bisect, not hypothesis** --
      five mechanism guesses were wrong; the remaining question is narrow (which buffer the
      captured graph reads that the split perturbs), and `SGLANG_FUSED_SPLIT_DEBUG=1` plus
      disabling graph capture for individual runners will localise it.
- [ ] 2026-10-09 **if graphs-on does not close it, PROFILE rather than hypothesise** -- five
      consecutive mechanism guesses were wrong on the device assert; the per-layer launch count
      is a suspect, not a diagnosis.
- [x] 2026-10-09 **`sm_margin` does not transfer**: margin 8 is the WORST arm here
      (12084 tok/s) despite being the kernel-harness optimum. Use margin 0.
- [ ] 2026-10-09 **guard-OFF arm re-running with `370c511f3`** -- in-server correctness of the
      split on real merged 4-mode batches, via the in-step reference.

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

## Tasks, in priority order — 2026-10-07

`[x]` = measured or validated against a reference; `[ ]` = not run. Nothing is checked
because code was written.

Kernel, per executor (each item A/B'd against its own executor's baseline):
- [x] **FA3 traits** (2026-10-07) -- MmaPV_is_RS=1 worth 3.84%; stages=3 +0.54%, LPT=1 +0.68%,
      TMA paged +10.03% slower. Upstream defaults win. [detail](#fa3-optimisation-items-measured-per-executor-on-the-fixed-kernel--2026-10-07)
- [x] **FA2 `MMA_KV` 2/3/4** (2026-10-07) -- flat (<0.5%); it is register tiling, SharedStorage identical.
- [x] **K/V pipelining, FA2** (2026-10-07) -- real S-slot K/V smem ring, per tile class. 2/3/1 is
      **-7.4%** on MTP-verify rows, **-10.3%** at 64-128K, -2.0% on appends; decode flat. Now the
      default. [detail](#fa2-kv-ring--2026-10-07)
- [x] **Smem-handoff JIT split-KV reduction** (2026-10-07) -- ceiling measured, NOT built: deleting the
      whole rendezvous + merge saves at most **2.9%** (longmtp), 0.8% (longdec). The global
      round trip is not the cost. Side finding: our JIT rendezvous is 2.2% slower than classic
      grid.sync there. FA3 side: the router sends it no split rows (append rows have >=448
      m-blocks), so it has nothing to reduce. [detail](#fa2-kv-ring--2026-10-07)
- [x] **DSMEM / KV multicast** (2026-10-07) -- ceiling measured, NOT built: FA3 with EVERY K/V
      gather skipped is only 1.8% faster (2048-token appends at 64-98K) and 1.6% faster (512 at
      24-49K), causal. FA3-routed rows are compute-bound, so multicast, which only cuts KV traffic,
      cannot pay. [detail](#multicast-backfill-ceilings--2026-10-07)
- [x] **Fused spec tree-build epilogue** (2026-10-07) -- built as ONE kernel replacing
      organize_draft_results + build_tree (+ their fills/cat). 0 mismatches over ~150 cases vs
      sglang's own path; graph-captured 31 -> 10 us (MTP chain), eager 120 -> 52 us. Not an
      attention epilogue: the data dependencies forbid it. Not yet wired into the worker.
      [detail](#fused-tree-build--2026-10-07)
- [x] **Cross-role backfill** (2026-10-07) -- in the split it is temporal: FA2's grid on SMs that
      FA3's tail vacates. Two-stream overlap on the TraceLab batch: 12 282.5 -> 11 954.7 us
      (**-2.7%**), both PASS. `tma_store_wait` guards nothing here (plain tiled store). Not yet
      in the serving path, which is sequential. [detail](#multicast-backfill-ceilings--2026-10-07)

Correctness and routing:
- [x] **Split-row "corruption"** (2026-10-07) -- was the harness reference, not the kernel; all fixtures pass.
- [x] **Routing** (2026-10-07) -- FA3 1.38x faster on append rows, FA2 1.30x on decode rows; split routes accordingly.
- [x] **Kernel A/B, TraceLab batch** (2026-10-07) -- split 12 425 vs FA2-all 15 396 us = 1.239x.

Workload and serving:
- [x] **Context length where attention is >50% of e2e** (2026-10-08, H200, Qwen2.5-7B, yarn x4 to 131 072). Decode
      step: whole step (batch 1 from `bench_one_batch`; batches 8 and 32 from a server via
      `context_share_srv.py`, whose chunked prefill fits them) against the decode kernel sglang runs for the
      same shapes x 28 layers. `drivers/context_share.sh`, `drivers/context_share_srv.sh`; commits b94e1c7de, 33bea018e.

      | batch | 4K | 16K | 32K | 64K | 120K |
      |---|---|---|---|---|---|
      | 1 | 8.3% (5.52 ms) | 13.3% | 17.8% | 24.2% | 31.9% (7.18 ms) |
      | 8 | 19.0% (5.66 ms) | 34.6% | 46.8% | 61.1% | 72.6% (17.97 ms) |
      | 32 | 34.1% (7.16 ms) | 60.6% | 73.7% (19.25 ms) | - | - |

      **Attention crosses half the step at ~40K tokens with 8 requests and ~10K with 32. A single
      request never gets there** (32% at 120K). The codex tenant (8-12 sessions at 32-49K) sits at
      ~47-55%: attention is half its decode step, which is why the kernel work shows up end to end
      there and not in the 2K-context runs. Both tables are pure decode; a mixed step adds the
      chunk's attention on top.
- [x] **Long-context A/B: full stack vs stock MTP, codex sessions at 64-96K** (2026-10-08, H200, 3 rounds,
      chat 12/s + codex 0.1 sessions/s, 0 fallbacks, every request completed). Trace: 300-session subset of the
      full TraceLab trace with rows at 64-96K (`tracelab_long_64k_96k.jsonl.gz`; the pod's default copy stops
      at 49K). `SCENARIO=long_agent_chat TRACE_PATH=... TAG=_long{,_r2,_r3} ARMS="MTP_LPT MTPMIX_SPLIT_EDFP"
      mtp_edf.sh 12 2000 30` (commits 95af60b18, 0be840a42). Both arms are past capacity at this load.

      | metric (3-round mean) | stock MTP | full stack | change |
      |---|---|---|---|
      | chat / codex output tok/s | 586.6 / 140.0 | **854.4 / 213.7** | **+45.6% / +52.6%** |
      | chat / codex TTFT ms | 54 242 / 37 095 | **1 815 / 10 690** | **-96.7% / -71.2%** |
      | chat / codex mean ITL ms | 67.1 / 36.7 | **35.6 / 19.4** | **-46.9% / -47.2%** |
      | chat / codex P99 ITL ms | 167.9 / 150.9 | 259.6 / 159.7 | **+54.6% / +5.8% (worse)** |

      Throughput, TTFT and mean ITL clear the bar by a wide margin here. Attention is 61-73% of
      the step at these lengths, and the 4-mode step keeps decode moving through prefill.
      **P99 ITL regresses**, worst on chat. The suspect is the 8 192-token chunk at a 64-96K
      prefix: in the 4-mode step its attention runs inside the same forward as every verify row,
      so every decoding request stalls together.
      **Chunk sweep, full stack only, one run each (2026-10-08)** (`BENCH_CHUNK=<c> TAG=_long_chunk<c>`): the
      stall is the chunk.

      | full stack | chunk 8 192 (3-round mean) | 4 096 | **2 048** | stock MTP (8 192) |
      |---|---|---|---|---|
      | chat / codex P99 ITL ms | 259.6 / 159.7 | 207.0 / 117.0 | **145.4 / 105.5** | 167.9 / 150.9 |
      | chat / codex mean ITL ms | 35.6 / 19.4 | 37.7 / 21.8 | 35.4 / 20.6 | 67.1 / 36.7 |
      | chat / codex TTFT ms | 1 815 / 10 690 | 2 149 / 13 169 | 4 306 / 17 740 | 54 242 / 37 095 |
      | chat / codex tok/s | 854 / 214 | 854 / 207 | 854 / 207 | 587 / 140 |

      At chunk 2 048 the full stack beats stock MTP on every metric: P99 ITL -13% / -30%, mean ITL
      -47% / -44%, TTFT -92% / -52%, throughput +46% / +48%. Stock MTP has the same knob, so the
      fair comparison tunes both.
      **Both arms at chunk 2 048, 3 rounds** (`BENCH_CHUNK=2048 TAG=_long_c2048{,_r2,_r3} ARMS="MTP_LPT
      MTPMIX_SPLIT_EDFP" mtp_edf.sh 12 2000 30`, 0 fallbacks, all completed): **the full stack wins on every
      metric.**

      | metric (3-round mean) | stock MTP @2048 | full stack @2048 | change | vs stock @8192 (its best) |
      |---|---|---|---|---|
      | chat / codex output tok/s | 565.6 / 131.8 | **850.6 / 203.3** | **+50.4% / +54.2%** | +45.0% / +45.2% |
      | chat / codex TTFT ms | 64 402 / 45 073 | **3 708 / 16 586** | **-94.2% / -63.2%** | -93.2% / -55.3% |
      | chat / codex mean ITL ms | 70.6 / 38.4 | **34.9 / 20.3** | **-50.6% / -47.2%** | -48.1% / -44.8% |
      | chat / codex P99 ITL ms | 172.5 / 157.2 | **143.2 / 104.3** | **-17.0% / -33.7%** | -14.7% / -30.9% |

      Stock MTP does not gain from the smaller chunk (it is worse on every metric than at 8 192),
      so stock's best configuration is the fair baseline: the right-hand column. Against it, the
      full stack at chunk 2 048 clears the bar on throughput (+45%), TTFT (-55% to -93%) and mean
      ITL (-45% to -48%). P99 ITL also improves (-15% / -31%); codex clears 30%, chat does not.
      The small chunk is the fix for the 4-mode step's tail: it bounds how much prefill attention
      any one verify step waits behind.
      **Prefix-aware chunk (attention budget), 3 rounds** (`SGLANG_CHUNK_ATTN_BUDGET=166000000`, c23101d1d: cap
      n so n*(r + n/2) <= budget; floor 1 024; only while rows are running). Full stack, `TAG=_long_budget*`:

      | full stack | chunk 8 192 | chunk 2 048 | attention budget |
      |---|---|---|---|
      | chat / codex P99 ITL ms | 259.6 / 159.7 | **143.2** / 104.3 | 258.1 / **91.1** |
      | chat / codex TTFT ms | **1 815 / 10 690** | 3 708 / 16 586 | 2 039 / 12 747 |
      | chat / codex tok/s | 854 / 214 | 851 / 203 | 854 / 212 |

      **The budget fixes codex's tail (91 ms, the best of any setting) at little TTFT cost, but not
      chat's.** Chat prompts have no prefix, so the budget leaves their chunks at full size, and
      a full chunk of fresh 512-token prompts is GEMM-bound, not attention-bound. So "attention
      stalls the verify rows" was half the story. The chat tail is chunk TOKEN count. Short
      context (chat 24/s, ITL 30 ms): unchanged (chat SLO 0.735 vs 0.714-0.750 over 3 earlier
      rounds), so the cap costs nothing where it does not bind. Next: bound the chunk's predicted
      TIME, c_tok*n + c_attn*n*(r + n/2) with the fitted step coefficients
      (`SGLANG_CHUNK_MS_BUDGET=47`, e75b2466e), which covers both terms.
      **Predicted-time chunk budget, 3 rounds** (`SGLANG_CHUNK_MS_BUDGET=47 SGLANG_CHUNK_COST=0.0216,1.3765`, the
      fitted H200 MTP step coefficients; 2 042 tokens at prefix 0, floor 1 024 at long prefixes;
      `TAG=_long_ms*`, 0 fallbacks, all completed):

      | metric (3-round mean) | stock MTP @8192 (its best) | full stack, ms budget | change |
      |---|---|---|---|
      | chat / codex output tok/s | 586.6 / 140.0 | 830.4 / 191.6 | **+41.6% / +36.9%** |
      | chat / codex TTFT ms | 54 242 / 37 095 | 10 512 / 24 035 | **-80.6% / -35.2%** |
      | chat / codex mean ITL ms | 67.1 / 36.7 | 34.5 / 20.3 | **-48.7% / -44.6%** |
      | chat / codex P99 ITL ms | 167.9 / 150.9 | 118.1 / 105.4 | **-29.7% / -30.2%** |

      Bounding both terms fixes both tails: chat P99 259 -> 118 ms, codex 160 -> 105 ms. Against stock
      MTP's best configuration, every metric clears 30% except chat P99 ITL, which misses by 0.3
      points (-29.7%). Against stock at chunk 2 048 both P99s clear it (-31.5% / -33.0%). The cost is
      TTFT: these chunks are small at long prefixes (the 1 024 floor), so codex TTFT keeps only
      -35%. The budget is the knob trading TTFT for tail ITL.
      **Tightening the budget clears the bar on every metric, 3 rounds at 20 ms**
      (`SGLANG_CHUNK_MS_BUDGET=20`, `TAG=_long_ms20{,_r2,_r3}`, 0 fallbacks, all completed). The
      -29.7% chat P99 miss above was a budget-tuning artefact, not a ceiling:

      | metric (3-round mean) | stock MTP @8192 (its best) | full stack, 20 ms budget | change |
      |---|---|---|---|
      | chat / codex output tok/s | 586.6 / 140.0 | 850.8 / 200.6 | **+45.0% / +43.3%** |
      | chat / codex TTFT ms | 54 242 / 37 095 | 7 947 / 22 424 | **-85.3% / -39.6%** |
      | chat / codex mean ITL ms | 67.1 / 36.7 | 31.4 / 17.5 | **-53.2% / -52.4%** |
      | chat / codex P99 ITL ms | 167.9 / 150.9 | 112.7 / 78.4 | **-32.9% / -48.0%** |

      Chat P99 ITL plateaus near 112 ms for budgets of 20-30 ms (118 ms at 47), so -33% is where
      that metric saturates; codex keeps improving (105 -> 78 ms). Sweep, one run each: budget 47 /
      30 / 20 gives chat P99 118.1 / 112.3 / 112.5 and codex P99 105.4 / 91.2 / 70.2 ms, with codex
      throughput 191.6 / 175.1 / 199.2 tok/s -- the 30 ms dip is round noise, not a trend.
      **Short context still regresses at 20 ms** (chat 24/s, `TAG=_itl30_ms20`, one round): chat SLO
      0.626 vs 0.714-0.750 unbudgeted, chat TTFT 311 vs ~180 ms, codex TTFT 2 480 vs ~1 450 ms --
      slightly worse than the 47 ms budget's 0.641. So the predicted-time budget stays a
      long-context-only setting, and a deployment serving both shapes on one node needs the budget
      gated on the batch's actual prefix lengths rather than set globally. **It is not a safe default:** at short context
      (chat 24/s, ITL 30 ms, `TAG=_itl30_ms`, one round) it REGRESSES. Chat SLO 0.641 vs 0.714-0.750, chat
      TTFT 286 vs ~180 ms, codex TTFT 2 184 vs ~1 450 ms. The ~2K chunk at prefix 0 splits short-context
      prefill into more fixed-cost mixed steps, the same mechanism that sank EDF's chunk cap. The
      attention budget is neutral there because it only binds at long prefixes. Use: attention
      budget always on; the predicted-time budget only for long-context deployments.
      **Chat 8/s, default chunk, 3 rounds** (`TAG=_long{,_r2,_r3} mtp_edf.sh 8 2000 30`, 0 fallbacks, all
      completed). Stock MTP still cannot keep up (chat 487 of ~576 offered tok/s); the full stack can:

      | metric (3-round mean) | stock MTP | full stack | change |
      |---|---|---|---|
      | chat / codex output tok/s | 487.3 / 164.9 | **574.8 / 213.7** | **+18.0% / +29.6%** |
      | chat / codex TTFT ms | 27 616 / 20 651 | **986 / 5 259** | **-96.4% / -74.5%** |
      | chat / codex mean ITL ms | 75.8 / 34.5 | **32.2 / 17.7** | **-57.5% / -48.6%** |
      | chat / codex P99 ITL ms | 193.3 / 159.2 | 245.8 / 158.2 | **+27.2% (worse)** / -0.6% |

      The chat P99 regression at the default chunk repeats at the lower load. It is a property of
      the 8 192-token chunk, not of overload.
- [x] **TraceLab mixed scenario wired** (2026-10-07) -- agent_plus_shortchat on the pod; trace verified identical after shrinking.
- [x] **Realistic serving, LPT baseline** (2026-10-07) -- clean, 0 fallbacks. [detail](#realistic-serving-ab-lpt-baseline-tracelab-agent--short-chat--2026-10-07)
- [x] **Realistic serving, SPLIT arm** (2026-10-07) -- CONFIRMED with a stream start barrier (3 rounds, tight ranges): chat TTFT -17.4%, codex TTFT -10.5%, ITL -7.9%, P99 ITL -15.9%, tput +7.0% ([detail](#stream-offset-confound--2026-10-07)). Original run, 3 interleaved rounds, gate PASS, 0 fallbacks:
      codex TTFT -10.3%, ITL -7.6%, P99 ITL -18.5%, tput +5.6%; chat TTFT -16.2%. **Below the
      30% / 10% bar.** [detail](#realistic-serving-valid--2026-10-07) **Earlier run VOID (2026-10-07).** The -39.8% TTFT / +16.9% tput run
      computed the WRONG requests on FA3: the split plan handed `fa3_only_run` the FA2 rows'
      per-request arrays. Fixed; serving correctness gate added. No valid SPLIT serving number
      exists yet. [detail](#split-serving-void--2026-10-07)
- [x] **4-mode MTP serving run** (2026-10-07, barrier re-run, 3 rounds, 0 fallbacks) -- 4-mode + split vs stock MTP:
      codex tput **+17.6%** (rounds +28/+29/-3%: one straggler-dominated round), ITL -16.2%, P99 ITL **-52.8%**,
      TTFT -8.5%; chat TTFT **-37.3%**, ITL -27.7%, P99 ITL **-30.5%**. SUPERSEDES the original numbers below,
      which had the uncontrolled stream offset. [detail](#mtp-barrier--2026-10-07) Original (offset-confounded): 4-mode + split vs
      stock MTP: codex throughput **+27.8%**, ITL -17.4%, P99 TTFT -14.6%; chat ITL **-35.0%**, chat TTFT
      >=-50%. Clears the bar on chat latency and codex throughput, not on codex TTFT/ITL. Stock-MTP baseline
      overloaded in 1 of 3 rounds; load sweep queued. [detail](#four-mode-mtp-serving--2026-10-07)
- [x] **Perf model: is the split easier to predict?** (2026-10-07) -- yes, modestly. Mixed-step cost, 5-fold CV,
      traced serving (one round each): linear(all) MAPE 13.4% SPLIT vs 16.9% LPT; MLP 6.0% vs 6.4%; RF 6.2% vs
      6.7%. Large gain for simple models, small for nonlinear ones. [detail](#perf-model-fit--2026-10-07)
- [x] **EDF on the 4-mode MTP server vs the unfused schedule** (2026-10-08, 3 rounds) -- deadline ORDER
      wins (chat TTFT -62%, codex P99 ITL -27% vs no EDF), and every per-step chunk CAP loses. The
      per-request acceptance EWMA collapses through a min-over-noisy-estimates bias. Full stack vs stock
      MTP at chat 24/s: chat TTFT -95%, ITL -42%; codex TTFT -51%, ITL -38%. At 16/s: ITL -36..-46%,
      codex TTFT only -19%. Details in pd_colocation_vs_disagg.md point 3 (v0.17.0-v0.17.2).

Evaluation completeness (added 2026-10-08, user-requested):
- [ ] **Re-derive the 2-D dispatch boundary and make it fine-grained.** The measured surface
      (`boundary_28x4x128.json`) is 2-D: FA3 wins at kv 512 with NO co-resident decode rows (1.42x)
      and loses at kv 32768 with 365 of them (0.98x). The harness encodes a 1-D approximation
      (`four_mode_dispatch.route`: `qo_len >= 128 and kv >= 4096`), which is both stale (it predates
      packed GQA) and coarser than the data. Serving uses a different rule
      (`mixed_dispatch.py`: packed_qo >= 16, kv_tiles >= 1), so harness and server now DISAGREE.
      Figure out WHY the boundary bends -- the candidate is that the decode population, not KV
      length, sets whether FA3's SMs have anything left to overlap -- then fit the boundary on
      (packed_qo, kv_tiles, decode_rows, decode_kv) instead of thresholding two axes separately.
      Also re-check `MIN_QO_LEN=16`: FA3 measured 0.78x at packed_qo 32 and 1.13x at 56, so a floor
      of 16 admits rows in the region where FA3 was measured LOSING.
- [ ] **Workloads from the recent serving literature, beyond our mixed 32K-1M TraceLab + chat.**
      Our transfer check so far is Code-Feedback and SWE-Gym (242 / 690-token prompts, parity, see
      realistic_workload_analysis.md#transfer-to-independent-coding-datasets). Add the datasets that
      accepted papers actually report, so reviewers can place our numbers: ShareGPT (in-tree:
      `--dataset-name sharegpt`), the Mooncake conversation/tool traces (`--dataset-name mooncake`,
      already wired), generated-shared-prefix (wired), and long-context sets (LooGLE / LongBench).
      Report each at matched offered load with the same gate and fallback counts.
- [ ] **Direct baselines, the comparisons a reviewer will demand.**
      - [ ] PD multiplexing, ASPLOS 2025 (https://arxiv.org/html/2504.14489v2) -- the closest prior
            work to our colocation argument. Read its mechanism and either run it or state precisely
            why its setting differs from ours.
      - [ ] sglang mixed chunked prefill on ALL FOUR nodes (`--enable-mixed-chunk`, no split, no
            4-mode) -- the unfused colocation baseline at full cluster scale, not just 1 GPU.
      - [ ] sglang PD disaggregation -- measured (disagg_2p2d / 1p3d), but re-run alongside the
            above two so all three baselines share one harness and one workload set.
- [ ] **Run server-level tests on CURRENT sglang, not only this fork.** Every serving number so far
      is on `Edenzzzz/sglang-dev` at base `0.5.4.post2`; the local `main` ref is from 2025-12-05 and
      upstream `sgl-project/sglang` main is 7 512 commits ahead (reachable: `git ls-remote` works,
      head b7b2975b5 as of 2026-10-08). Risk: baselines may already be faster upstream (chunked
      prefill, PD, and the flashinfer backend all moved), which would shrink or invalidate our
      deltas. Rebase or cherry-pick our kernel/scheduler patches onto a current upstream and re-run
      at least the headline arms there.
- [ ] **Does EVERY optimization still gain on latest main?** Re-validate each one individually
      against the CURRENT upstream baseline, not against our fork's baseline -- a gain measured over
      a 10-month-old baseline is not a gain if upstream already closed it. One row per
      optimization, each with its own A/B, gate pass and fallback count:
      - [ ] FA3+FA2 two-executor split (`SGLANG_TWO_EXEC` + `SGLANG_FUSED_FA3_SM_SPLIT`) -- 1.239x
            on the TraceLab kernel batch, chat TTFT -17.4% in serving
      - [ ] two-stream overlap (`SGLANG_TWO_EXEC_OVERLAP`) -- -2.7% on the kernel batch
      - [ ] 4-mode MTP step (`SGLANG_MIXED_SPEC`) -- the largest serving effect; upstream may have
            changed `mix_with_running` or the EAGLE verify path under us
      - [ ] fused spec tree build (`SGLANG_FUSED_TREE_BUILD`) -- 31 -> 10 us graphed
      - [ ] host-overhead fixes (skip the flashinfer FA3 plan + sparse indices in two-exec mode; no
            device syncs in the kv-index build) -- 1.79 -> ~1.6 ms/mixed step
      - [ ] deadline-ordered admission (EDFP) -- chat TTFT -62%; check upstream's scheduler still
            has the priority hooks this uses
      - [ ] prefix-aware chunk budgets (`SGLANG_CHUNK_ATTN_BUDGET`, `SGLANG_CHUNK_MS_BUDGET`)
      - [ ] router: cache affinity + decode-aware SLO veto, and the L3-store migration arm
      Any optimization that no longer gains gets recorded as superseded-by-upstream IN PLACE, with
      the upstream commit that closed it, rather than quietly dropped.
      **Environment cost, measured 2026-10-08 (separate from the code port, and NOT optional).**
      Our fork's environment cannot even import current upstream:
      - upstream declares **66 runtime deps** (torch/cuda/sgl-kernel/flashinfer excluded) against our
        fork's much smaller set; `msgspec` was the first miss, and iterating one import error at a
        time is the slow way to find the rest -- install the dep LIST from its pyproject instead.
      - `pip install -e python/ --no-build-isolation` FAILS: upstream's build backend
        (`wheel_stub.buildapi`) is not installed.
      - **upstream main pins `torch==2.14.1`** (python/pyproject.toml:6), while our fork pins
        **torch 2.8.0** because the fused FA3+FA2 extension was built against it. That is a SIX
        minor-version jump, and the ported extension must be rebuilt against 2.14.1 -- a much larger
        ABI move than a 2.8->2.9 bump, and the single biggest risk item in the port.
        *CORRECTION (same day): an earlier version of this entry said "upstream needs torch >= 2.9"
        and recorded torch 2.9.1 + torchao 0.14.1 as "known-good". That is WRONG. 2.9.1 lets
        `import sglang` succeed but `sgl_kernel`'s prebuilt `common_ops.abi3.so` then fails with
        `undefined symbol: _ZN3c104cuda19CUDAErrorLogCaptureC1Ev` -- the wheel is built for 2.14.
        The lesson generalises: on this stack "imports cleanly" is NOT evidence of a working
        environment, because the native extensions load later. Read the pin, do not bisect wheels.*
      - **The dep set DOES resolve -- to upstream's own pin, `torch==2.14.1`.** Six provisioning
        attempts were wasted holding torch at our fork's 2.8/2.9 and bisecting `torchao` and
        `flashinfer` around it; every wheel pip chose (torchao, sgl_kernel) was built for 2.14, so
        each "fix" moved the failure one import later. Install upstream's pinned torch FIRST and let
        the rest follow.
      - **WORKING RECIPE for a current-upstream venv** (seven failed attempts preceded it; their
        failure modes are listed above). The order is forced, not stylistic:
        1. a CLEAN venv -- one that has held torch 2.8 then 2.9 then 2.14 is unrecoverable
           (torchvision stays pinned to the old torch, and nccl mismatch surfaces as
           `libtorch_cuda.so: undefined symbol: ncclCommResume`);
        2. `pip install torch==2.14.1` from default PyPI so torch brings its own consistent
           `nvidia-*` wheels (resolves to 2.14.1+cu130);
        3. `pip install wheel_stub` -- upstream's build backend, needed by step 4;
        4. `pip install --no-build-isolation <clone>/python` -- isolation OFF, because the backend
           imports torch while GETTING requirements and isolation hides it.
        Verify with `import sgl_kernel`, NOT `import sglang`: the former loads the native .so and is
        the check that actually fails when the torch pin is wrong.
      - the pod can `git clone` github directly, which is far faster than base64-pushing 15 MB
        through the webshell (that timed out at 900 s). Pin the clone to the commit the port was
        read against: upstream main moved `b7b2975b5` -> `54b551df8` within hours.

      **4-mode port on upstream, 17 commits: every stage now exists, each added in response to a
measured failure (2026-10-08).** Branch `upstream_port`, worktree `/opt/tiger/sglang-up`, tested
with `drivers/portchk.sh` against pinned upstream `b7b2975b5` in the clean venv. The failure
sequence IS the list of upstream contracts the fork never had to satisfy:

| # | failure observed | cause / fix |
|---|---|---|
| 6 | pool leak, byte-identical twice | `forward_mixed_spec` was never CALLED -- no dispatch branch. Identical numbers are the tell that new code is not running. |
| 7-8 | `'NoneType' has no attribute 'merge'` | `merge_batch` dereferences `sampling_info.penalizer_orchestrator`; MEASURED that the PREFILL batch lacks it and the running decode batch has it -- the opposite of two guesses. |
| 9 | `tensor a (2055) vs b (2051)`, off by `draft_token_num` | draft tokens added twice: `eagle_prepare_for_verify` already sets `running.input_ids`, and `merge_batch` concatenates it. |
| 10 | `shape '[1, 4]' invalid for input of size 1` | a MIXED batch keeps only the last logit per row; verify needs one per draft token. Carry `mixed_verify_tokens` ScheduleBatch -> ForwardBatch -> LogitsMetadata and keep the tail rows' every token. |
| 11-12 | `SimpleNamespace` missing `seq_lens`, then `ngram_embedding_info` | a hand-built sampler view needs an open-ended field tail; swap `sampling_info` on the REAL ForwardBatch instead. |
| 13 | `'tuple' has no attribute 'extra_keep_alive_refs'` | the scheduler wants a `GenerationBatchResult`; attach the verify half as `mixed_running_result`. |
| 14 | relay stash: `payload.topk_p` is None | both draft extends must run, or the next step has no draft input. (Predicted in the previous commit, then observed.) |
| 15-17 | `index 4 out of bounds`, `tensor a (4) vs b (5)`, `shape [1] vs broadcast [4]` | upstream's merge MUTATES the prefill batch (the fork built a separate worker batch). Fixed structurally twice: merge onto a COPY, and DELEGATE to `mix_with_running` + widen the tail to n tokens, so upstream owns the field list. |

**Where the port stands (18 commits) and the EXACT next step.** The step runs end to end through
draft -> merge -> one target forward -> logits split -> verify accept -> commit -> prefill draft
extend. The remaining failure is in the VERIFY rows' draft extend:

- `RuntimeError: output with shape [1] doesn't match the broadcast shape [4]`, and a temporary
  three-line diagnostic in `cuda_graph_buffer_registry.fill_from` named the slots:
  **`positions` (dst 1, src 4)` and `out_cache_loc` (dst 1, src 4)`**. (Technique worth reusing:
  shapes alone never identify which field is wrong; log `slot.name` with both shapes on the
  `RuntimeError` and re-raise. Reverted afterwards to keep the port diff clean.)
- Diagnosis: `running` has ONE row but its `out_cache_loc` still holds the n=4 verify slots that
  `eagle_prepare_for_verify` assigned. Upstream resets those via `prepare_for_draft_extend`
  (eagle_worker_common.py:113), which takes `widened_out_cache_loc` / `widened_positions` and is
  reached from `_draft_extend_for_decode`; it derives them from `GenerationBatchResult` fields that
  my hand-built result does not populate (candidates: `prepared_draft_extend_inputs`,
  `allocate_lens`, `next_draft_input`).
- Attempted fix (commit 19) and RETRACTED (commit 20): narrowing `out_cache_loc`/`positions` to the
  accepted tokens before the call. Measurement at the call site (`VX_DEBUG`) shows why it is wrong:
  **bs=1, n=4, accept_lens=[1], out_cache_loc=4, positions=None, seq_lens=[513]**. `positions` does
  not exist yet -- `prepare_for_draft_extend` CREATES it n-wide from `num_draft_tokens` and derives
  the accepted layout itself, so it expects the n-wide tensors.
- Resolved by running the narrowing AND the slot diagnostic TOGETHER (commits 21-22): with
  `out_cache_loc` narrowed, **only `positions` mismatched**, so commit 19 was right and commit 20's
  revert was wrong -- I had generalised from one field (`positions=None`) to both. Reverting the fix
  and the diagnostic separately destroyed the comparison that settles it; keep both active.
- `positions` is NOT settable at the call site either: `prepare_for_draft_extend` overwrites it
  n-wide unconditionally. So n-wide fields are INTENDED, built from
  `num_tokens_per_req = speculative_num_draft_tokens`.
- **The actual inconsistency, and the next step:** the draft-extend batch reports a token count of
  **1** (this is `raw_n`, which sizes the graph buffer) while its fields are built **n-wide (4)**.
  The fix belongs in what the batch reports, not in the tensors: find what the runner derives
  `raw_num_tokens` from for this batch (`input_ids.numel()`, `extend_num_tokens`, or
  `num_accept_tokens`) and make it `bs * n`. Resizing tensors to match a wrong count is what the
  last four commits kept doing.
- Commit 22 keeps the proven `out_cache_loc` narrowing and drops the no-op `positions` assignment.
- Honest note on the loop: three of the last five commits were wrong-direction fixes corrected by a
  one-run diagnostic. Where a field's provenance is not obvious, LOG IT FIRST -- the two diagnostics
  used here (slot names in `fill_from`, tensor sizes at the call site) each resolved in one run what
  inference had failed twice to get.

**Method note, recorded because it cost three rounds:** when fixes start arriving one field or one
attribute at a time, the structure is wrong, not nearly right. That applied to the sampler view
(11-12) and to the merge (15-17); in both cases the restructured version is SHORTER than the
patched one. Also: an identical failure after a fix means the code did not run.

**4-mode port on upstream: merge + single forward RUN; failure is the documented missing commit
step (2026-10-08).** Four commits on branch `upstream_port` (worktree `/opt/tiger/sglang-up`):
`mix_with_verify` (85562edfa), the scheduler's deferred merge + `forward_mixed_spec` (47720d9c1),
the logits split into upstream's `eagle_sample` (9450daf39). Smoke test `drivers/portchk.sh` on the
clean upstream venv:

- **the server comes up and the ported path executes** -- requests are served, so the merge
  (`mix_with_verify`), the single target forward over `[chunk | verify]`, and the slice into
  `eagle_sample` all work against upstream's APIs.
- it then fails with `ValueError: pool memory leak detected! total=1597048, available=1593134,
  evictable=1834, protected=2044` -- **36 tokens unaccounted = 4 draft tokens x 9 rows.**
- **Cause is exactly TODO 2 (commit accepted tokens), not a merge bug.** Upstream reclaims
  rejected draft slots as *overshoot* from each request's committed seqlen
  (`eagle_worker_common.py`: "trailing unaccepted slots stay and are freed as overshoot"), via
  `new_seq_lens = batch.seq_lens + accept_lens` and `prepare_verify_commit_outputs`. My path calls
  `eagle_sample` but never commits, so the unaccepted slots belong to no request.

**CORRECTION to an earlier claim in this doc:** finishing the 4-mode port is NOT blocked on
rebuilding the fused extension against torch 2.14.1. The 4-mode step is a scheduling/batching
change that runs on upstream's DEFAULT attention backend; the FA3+FA2 split is a separate layer
beneath it. The rebuild gates re-validating the SPLIT KERNEL upstream, nothing else. Conflating the
two invented a blocker and briefly stopped work.

Next, in order: (1) commit accepted tokens via `prepare_verify_commit_outputs` + `new_seq_lens`;
(2) sample the prefill half explicitly (`is_verify=True` returns raw logits); (3) both draft
extends and a result carrying the running batch's outcome; then the serving A/B against upstream's
own mixed-chunk+spec baseline.

**GATING CHECK ANSWERED 2026-10-08: upstream mixes a chunk with 1-token DECODE TAILS, not with
verify rows -- so the 4-mode step is still our contribution.** Pinned upstream `b7b2975b5`, clean
venv (recipe above), `--enable-mixed-chunk --chunked-prefill-size 2048 --speculative-algorithm
EAGLE3 --speculative-num-draft-tokens 4`, 8 background decode rows plus 8 x 9 000-token prompts
(`drivers/compcheck.sh`). Evidence from the server's own batch log:
- **mixed chunked prefill DOES run under spec**: 36 `Prefill batch` lines carrying `#running-req`
  3-8 (five steps at 8), with spec live (`accept len` 2.42-3.14).
- **but `accept len` appears ONLY on the 9 `Decode batch` lines, never on a `Prefill batch` line.**
  If a chunk and its verify rows shared one target forward, acceptance would be reported on that
  step. Verification is therefore a separate forward, and the mixed prefill step carries its
  running rows as 1-token tails -- matching `mix_with_running`'s spec branch, which gathers "each
  tail's bonus slot at the committed length".
*Caveat on strength:* this is an inference from log structure, not instrumented per-step token
accounting. The definitive check logs `forward_batch.input_ids.shape[0]` against `#new-token` on a
mixed step: a difference of `draft_token_num * #running-req` means verify rows. It agrees with the
independent code read, so it is enough to proceed, and the instrumented version is worth one run
during the port.
**Consequence:** modes 1-3 (chunk, tail, draft extend) are upstream's; the 4-mode delta -- routing
the chunk into the VERIFY forward -- is ours, and is what the serving win measured.

**Port cost, measured 2026-10-08** (`git fetch --depth=1 upstream main`, head `b7b2975b5`):
      | file we patch | upstream state | diff vs ours |
      |---|---|---|
      | `layers/attention/flashinfer_backend.py` | exists, heavily changed | 2 280+ / 1 355- |
      | `managers/scheduler.py` | exists, restructured | 2 443+ / 4 876- |
      | `bench_serving.py` | exists, changed | 3 264+ / 14- (ours is mostly additive) |
      | `layers/attention/mixed_dispatch.py` | **absent** (ours) | ports clean |
      | `managers/edf_scheduler.py` | **absent** (ours) | ports clean |
      | `speculative/eagle_worker.py` | **GONE upstream** | the 4-mode MTP patch has no host file |
      The last row is the blocker: upstream's `speculative/` is now 51 files
      (`eagle_worker_common.py`, `eagle_worker_v2.py`, `multi_layer_eagle_*`, `dflash_*`,
      `dspark_*`), and the single `eagle_worker.py` our `forward_mixed_spec` patches no longer
      exists. So the 4-mode step -- the largest serving win -- must be re-derived against
      `eagle_worker_v2`/`eagle_worker_common`, not rebased.
      **Re-derivation plan for the 4-mode step (studied 2026-10-08; the seam is CLEANER upstream,
      so this is tractable, not a fallback case).** Upstream hook points:
      - `EAGLEWorkerV2.forward_batch_generation` (eagle_worker_v2.py:1319) branches extend ->
        `_forward_prefill_batch` (:1449) vs decode -> `draft_worker.draft(batch)` -> `verify(batch)`
        (:1871) -> `draft_worker._draft_extend_for_decode(batch, batch_output)`. Those are exactly
        the four stages our `forward_mixed_spec` fuses.
      - `verify()` is now a thin wrapper over the FREE function `run_eagle_verify(...)` in
        `eagle_worker_common.py`, which builds `verify_forward_batch` and runs ONE
        `target_worker.forward_batch_generation(batch=None, forward_batch=verify_forward_batch,
        is_verify=True, ...)` (:639). A single, explicit target-forward call site taking a
        caller-supplied `ForwardBatch` is a better seam than our fork's method, which we had to
        split into forward + `_verify_post` by hand.
      Derivation: add an optional prefill-row injection to `run_eagle_verify` (extend
      `verify_forward_batch` with the chunk's rows, run the one target forward, split
      `logits_output` into prefill rows -> sample and verify rows -> the existing accept path), then
      a `forward_mixed_spec` on `EAGLEWorkerV2` that calls `draft(running)`, that extended verify,
      and both draft extends. No `mix_with_running` patch needed: upstream already accepts an
      explicit `forward_batch`, which is what that patch existed to work around.
      Order: (1) this re-derivation, since the 4-mode step is the largest serving win; (2) the two
      new files (free); (3) additive bench changes; (4) re-measure every row above. Only if the
      re-derived step shows NO gain on current upstream does the fork result stand as the
      published one, and that would itself need recording with the upstream commit responsible.

Cluster:
- [ ] **Multi-node PD colocation, four points** (H20, 8 pods x 2 GPUs, `tasks/pd_colocation_vs_disagg.md`).
      Measured 2026-10-08 (non-MTP workers): cache-affinity colocation vs the best disagg (2P2D)
      cuts TTFT 93% (agent) and 99.5% (chat), with 73% vs 23-31% prefix hit. Disagg keeps lower ITL
      (3.7 vs 6.4 ms). The L3 store lifts round-robin's hit rate 12.7 -> 22.3% and sits idle under
      affinity. Remaining: the calibrated SLO veto + worker EDF arms, MTP arms on both sides
      (gated), and a hot-prefix experiment for point 4 (all queued).

<a id="stream-offset-confound--2026-10-07"></a>
### METHODOLOGY RETRACTION: multi-stream serving runs had an uncontrolled start offset — 2026-10-07

**Found** while checking a non-monotonic chunk sweep. LPT chat mean TTFT measured 406 ms at
chunk 4096, **18 759 ms** at 8192 (the 3-round repeats), and 221 ms at 16384. Rerunning 8192 on
the same code today gave **1 205 ms**. Server arguments are identical except the random seed,
and every server served its own requests (no bind failures). The difference is the client:
`run_mixed_workloads.py` launches each stream as its own `bench_serving` process, and the codex
stream tokenises 60 prompts of 32-49K tokens before it sends anything. The first big codex
prefill therefore reached the server **36-45 s** after chat started in the repeats (one round
190 s), but **12-25 s** after it today (`/dev/shm/w/startorder.py` over the server logs). Where
the codex backlog of about 2.4M prefill tokens lands inside the 200-second chat stream decides
how many chat requests queue behind it.

**What this invalidates.** Absolute TTFT values in every multi-stream serving table here --
the SPLIT vs LPT repeats, the 4-mode MTP A/B and the chunk sweep -- and therefore the size of
the chat TTFT gains. Arms within a session were interleaved and mostly saw similar offsets, so
the direction of each comparison is probably right, but its magnitude is not established.
ITL and throughput are less offset-sensitive. They are not exempt.

**Fix** (`d2fc14f8c`): a start barrier. Each stream tokenises and warms up, signals ready, and
waits; the runner releases all streams together (`BENCH_BARRIER_DIR`; `BENCH_NO_BARRIER=1`
reproduces the old race). All three serving A/Bs are being re-run with it.

**Outcome (2026-10-07), SPLIT vs LPT, barrier, 3 interleaved rounds, gate PASS, 0 fallbacks**
(`realistic_repeats_b.sh`; `repsum.py --root /dev/shm/w/repb --base LPT`):

| metric | barrier: LPT -> SPLIT | delta | original repeats |
|---|---|---|---|
| chat mean TTFT | 19 279 [19 169-19 339] -> 15 917 [15 865-15 958] ms | **-17.4%** | -16.2% |
| chat P99 TTFT | 61 455 -> 54 709 ms | -11.0% | -10.9% |
| codex mean TTFT | 25 672 -> 22 974 ms | -10.5% | -10.3% |
| codex mean ITL | 36.2 -> 33.4 ms | -7.9% | -7.6% |
| codex P99 ITL | 512 -> 431 ms | -15.9% | -18.5% |
| codex output tok/s | 620.7 -> 664.0 | **+7.0%** | +5.6% |

With all streams released together, absolute values match the original repeats (LPT chat TTFT
19.3 s vs 18.8 s), so those repeats were effectively a simultaneous start. **The original
SPLIT-vs-LPT result stands**, now with tight ranges. The low absolute TTFTs belonged to the
chunk sweep, which ran in a later-start regime; its relative deltas are single-round only.

<a id="mtp-load-sweep--2026-10-07"></a>
### 4-mode MTP load sweep: the gain grows with load, and needs the split — 2026-10-07

`drivers/mtp_load_sweep.sh`: scenario `agent_chat_rated`, with BOTH tenants rate-limited (36
codex sessions at 0.1 sessions/s, 8 groups; chat 512->96 tokens at the rate shown), stream start
barrier, one round per point, every arm 0 fallbacks, 0 tracebacks. Mixed steps per 4-mode arm:
600 at 4/s, 1 400 at 8/s, 2 000 at 12/s, 2 200-2 400 at 16/s. Table:
`python3 drivers/mtp_loadtab.py`. Delta vs stock MTP (MTP_LPT):

| chat rate | arm | chat TTFT | chat ITL | chat P99 ITL | codex TTFT | codex ITL | codex P99 ITL |
|---|---|---|---|---|---|---|---|
| 4/s | MTPMIX_LPT | +9% | -2% | -5% | +4% | +1% | +31% |
| 4/s | **MTPMIX_SPLIT** | +1% | -3% | -3% | -7% | +2% | +12% |
| 8/s | MTPMIX_LPT | +13% | -4% | -18% | +3% | -3% | +15% |
| 8/s | **MTPMIX_SPLIT** | -2% | -5% | -16% | -8% | -6% | -2% |
| 12/s | MTPMIX_LPT | +22% | -4% | +12% | +5% | -6% | +33% |
| 12/s | **MTPMIX_SPLIT** | -21% | -24% | **-42%** | -10% | -23% | **-38%** |
| 16/s | MTPMIX_LPT | +16% | +2% | +13% | +8% | -2% | +63% |
| 16/s | **MTPMIX_SPLIT** | **-48%** | **-34%** | **-45%** | -14% | **-35%** | **-36%** |

Absolute stock-MTP values at 16/s: chat TTFT 204.8 ms, chat ITL 13.3 ms, codex TTFT 763.6 ms,
codex ITL 8.1 ms. Every arm keeps up with the offered load at every point (throughput equal), so
this is latency at matched throughput.

**Reading.**
1. The 4-mode + split gain grows monotonically with load: none at 4/s, large at 16/s. At 16/s it
   clears the 30% bar on chat TTFT (-48%), chat ITL (-34%), codex ITL (-35%) and both P99 ITLs.
   Codex TTFT (-14%) does not clear it.
2. **Mixing without the split is a regression**, and a growing one (codex P99 ITL +63% at 16/s).
   Folding prefill chunks into verify steps makes each verify step heavier; only the split, which
   makes that mixed step cheaper, turns the extra scheduling freedom into a gain. The 4-mode
   schedule and the split kernel are one contribution, not two.
3. Throughput is equal because nothing saturated. Capacity (the throughput claim) needs rates past
   stock MTP's saturation point: 20, 24, 28/s, below.

**Extension to 20-28/s** (`mtp_load_sweep2.sh`, same setup, every arm 0 fallbacks; mixed steps
1 600-3 000 per 4-mode arm):

| chat rate | arm | codex TTFT | codex ITL | chat TTFT | chat ITL | chat tok/s |
|---|---|---|---|---|---|---|
| 20/s | MTPMIX_LPT | +63% | +20% | +137% | +28% | 0% |
| 20/s | **MTPMIX_SPLIT** | **-48%** | **-33%** | **-87%** | **-39%** | 0% |
| 24/s | MTPMIX_LPT | +9% | -3% | +23% | +5% | 0% |
| 24/s | **MTPMIX_SPLIT** | **-75%** | -23% | **-88%** | **-38%** | 0% |
| 28/s | MTPMIX_LPT | +62% | +10% | +70% | +3% | -7% |
| 28/s | **MTPMIX_SPLIT** | **-79%** | -19% | **-88%** | **-32%** | +3% |

Stock MTP absolute values: codex TTFT 1.3 / 4.5 / 7.1 s and chat TTFT 1.0 / 6.0 / 11.0 s at
20 / 24 / 28/s. The 4-mode + split arm stays at 0.7-1.5 s.

**At 20/s, all four headline latency metrics clear the 30% bar on both tenants**: codex TTFT
-48%, codex ITL -33%, chat TTFT -87%, chat ITL -39%. Throughput is within 3% at every point,
because a rate-limited sweep can only serve what it offers. At 28/s stock MTP falls just short
of the offered chat rate (1 901 tok/s) and the split arm serves 1 957 tok/s. The capacity claim
needs a closed-loop test: `drivers/mtp_saturate.sh` (both tenants at infinite rate, 3 rounds).

**Capacity, closed loop** (`drivers/mtp_saturate.sh`: scenario `agent_chat_saturate`, 36 codex
sessions and 3 200 chat requests all submitted at once; 3 interleaved rounds with the start
barrier; every arm 0 fallbacks, 0 tracebacks; acceptance 2.13-2.18 in every arm;
`repsum.py --root /dev/shm/w/mtpsat --base MTP_LPT`):

| metric | MTP_LPT (stock) | MTPMIX_LPT | **MTPMIX_SPLIT** |
|---|---|---|---|
| chat output tok/s | 1 364.9 [1 302.0-1 419.6] | 1 398.0 (+2.4%) | **1 645.1 [1 622.1-1 666.9] (+20.5%)** |
| codex output tok/s | 315.9 [287.9-338.8] | 295.8 (-6.3%) | **369.3 [351.1-389.3] (+16.9%)** |
| **total output tok/s** | 1 680.8 | 1 693.8 (+0.8%) | **2 014.4 (+19.8%)** |
| codex mean ITL ms | 49.8 | 41.3 (-17.2%) | **34.8 (-30.2%)** |
| chat mean ITL ms | 22.4 | 22.0 (-1.8%) | 18.6 (-16.9%) |
| chat P99 ITL ms | 120.3 | 103.0 (-14.4%) | 92.6 (-23.0%) |
| chat mean TTFT s | 124.7 | 119.9 (-3.9%) | 100.8 (-19.2%) |
| codex mean TTFT s | 15.5 | 17.8 (+15.0%) | 18.8 (**+21.3%**) |

**Capacity rises ~20%**, with non-overlapping ranges on both tenants -- double the 10% bar.
Codex TTFT is worse because under a closed loop TTFT is queue position: the split arm drains
20% more chat through the same queue ahead of codex turns. It is reported as measured, not
excluded. Mixing without the split adds no capacity (+0.8%).

<a id="mtp-barrier--2026-10-07"></a>
### 4-mode MTP A/B re-run with the stream start barrier — SUPERSEDES the table below — 2026-10-07

`mtp_repeats_b.sh` (= `mtp_repeats.sh` with the start barrier, outdir `mtprepb`); 3 interleaved
rounds; every arm 0 fallbacks, 0 tracebacks; mixed arms 800-1 200+ 4-mode steps each;
acceptance 2.14-2.33 in every arm. `repsum.py --root /dev/shm/w/mtprepb --base MTP_LPT`:

| stream | metric | MTP_LPT (stock) | MTPMIX_LPT | **MTPMIX_SPLIT** |
|---|---|---|---|---|
| codex | output tok/s | 402.5 [398.6-409.1] | 438.7 (+9.0%) | **473.3 [390.4-516.8] (+17.6%)** |
| codex | mean ITL ms | 57.6 [56.4-58.9] | 50.7 (-12.0%) | 48.3 [47.1-49.1] (-16.2%) |
| codex | P99 ITL ms | 384.8 [329.8-441.9] | 182.2 (-52.7%) | **181.6 [180.8-182.0] (-52.8%)** |
| codex | mean TTFT ms | 28 291 | 28 544 (+0.9%) | 25 898 (-8.5%) |
| codex | P99 TTFT ms | 71 067 | 67 399 (-5.2%) | 63 295 (-10.9%) |
| chat | mean TTFT ms | 70 252 [67 672-72 652] | 56 659 (-19.3%) | **44 015 [41 836-45 247] (-37.3%)** |
| chat | P99 TTFT ms | 110 192 | 97 446 (-11.6%) | 88 509 (-19.7%) |
| chat | mean ITL ms | 28.5 | 24.2 (-15.3%) | 20.6 (-27.7%) |
| chat | P99 ITL ms | 143.5 | 109.9 (-23.4%) | **99.7 (-30.5%)** |
| chat | output tok/s | 818.6 [809.4-832.4] | 856.9 | 857.3 |

**What changed vs the original run:**
- With simultaneous starts, stock MTP falls behind the offered chat rate in **every** round (818.6 <
  858 tok/s), not just one. All chat TTFTs are high (44-70 s) because the codex burst lands at t=0
  for every arm.
- The codex throughput gain is +17.6%, not +27.8%. Per round the split arm gives 512.5 / 516.8 /
  390.4 tok/s; the third is a makespan straggler (its ITL and TTFT are normal). Rounds 1-2 alone
  would be +28%; the honest number is the 3-round mean with its range.
- Codex P99 ITL is **-52.8%**, not -17.7%. Stock MTP's tail is much worse when prefills land
  together, because it cannot fold prefill into verify steps.

**Against the bar** (>=30% TTFT and ITL, >=10% throughput): met for chat TTFT (-37.3%), chat P99
ITL (-30.5%), codex P99 ITL (-52.8%) and codex throughput (+17.6%). **Not met** for chat mean ITL
(-27.7%), codex mean ITL (-16.2%) or codex TTFT (-8.5%).

<a id="four-mode-mtp-serving--2026-10-07"></a>
### 4-mode MTP serving A/B: stock MTP vs the 4-mode step (FA2 / split) — 2026-10-07

Code: sglang `1191c70b3` on the pod (4-mode step `868025d6c` + fixes); flashinfer `b0563ae`.
EAGLE3 draft (thoughtworks/Qwen2.5-7B-Instruct-Eagle3), 3 steps, topk 1, 4 verify tokens.
Command: `bash tasks/fused_specdec_workload/drivers/mtp_repeats.sh` (3 interleaved rounds of
MTP_LPT, MTPMIX_LPT, MTPMIX_SPLIT; scenario agent_plus_shortchat). Every arm: **0 fallbacks,
0 tracebacks**. The mixed arms ran 1 600-1 800+ 4-mode steps each. Acceptance length (windowed
log mean) is 2.04-2.32 in every arm, equal within noise. **Per-request check**
(`srv_correctness.py`, MTPMIXacc vs MTPacc, 2026-10-07): every request with an IDENTICAL greedy output
(l0, l2, s0, s3, s5) has exactly equal tokens-per-verify (2.20, 2.30, 2.89, 2.46, 2.20), so the draft
model sees identical inputs after mixed steps. Differences appear only where the text itself diverged
late (the batching noise floor). Summary:
`python3 repsum.py --root /dev/shm/w/mtprep --base MTP_LPT` (mean, [min-max] over 3 rounds):

| stream | metric | MTP_LPT (stock) | MTPMIX_LPT | MTPMIX_SPLIT |
|---|---|---|---|---|
| codex | output tok/s | 395.6 [364.7-426.5] | 407.4 (+3.0%) | **505.6 [494.1-525.9] (+27.8%)** |
| codex | mean ITL ms | 59.1 [55.6-61.2] | 52.6 (-11.0%) | **48.8 [47.8-49.6] (-17.4%)** |
| codex | P99 ITL ms | 222.0 [197.0-260.6] | 189.8 (-14.5%) | 182.7 [181.6-183.8] (-17.7%) |
| codex | mean TTFT ms | 28 792 [26 889-32 151] | 29 248 (+1.6%) | 26 036 [25 125-26 894] (-9.6%) |
| codex | P99 TTFT ms | 73 302 | 68 613 (-6.4%) | 62 631 (-14.6%) |
| chat | mean TTFT ms | 34 542 [9 270-82 698] | 6 558 | **4 965 [4 171-5 751]** |
| chat | P99 TTFT ms | 62 435 | 25 062 (-59.9%) | 21 668 (-65.3%) |
| chat | mean ITL ms | 24.9 [21.4-30.2] | 18.7 (-24.9%) | **16.2 [15.1-16.8] (-35.0%)** |
| chat | P99 ITL ms | 139.0 | 110.6 (-20.4%) | 100.7 (-27.6%) |
| chat | output tok/s | 821.4 [748.6-857.9] | 857.4 | 857.8 |

**Caveat on the chat TTFT row.** Stock MTP fell behind the offered chat rate in round 1 (748.6
< 858 tok/s; TTFT 82.7 s), so its mean is an overload artefact. Excluding that round, chat mean
TTFT is 9.3-11.7 s for stock MTP against 4.2-5.8 s for the split: **-50% to -60%**. The
overload is itself a finding: stock MTP cannot fold prefill into verify steps, so at this load it
sits at a queueing cliff that the 4-mode step stays clear of. That is a capacity statement, and
it needs the load sweep (`drivers/mtp_load_sweep.sh`, queued) to measure properly.

**Decomposition.** Mixing alone (MTPMIX_LPT) gives most of the chat gains but only +3% codex
throughput. The split kernel adds codex throughput (+3% -> +28%) and about 10 points of ITL on
both streams.

**Against the bar** (>=30% TTFT and ITL, >=10% throughput): met for chat ITL (-35%), chat TTFT
(-50% or better) and codex throughput (+28%). **Not met** for codex ITL (-17%) and codex mean
TTFT (-10%; ranges touch).

<a id="perf-model-fit--2026-10-07"></a>
### Perf model: per-step cost of mixed steps, SPLIT vs LPT — 2026-10-07

Traces: one serving round each, `SGLANG_STEP_TRACE=1`, agent_plus_shortchat, sglang
`c1d2bceda`; 2 154 SPLIT and 2 144 LPT mixed steps. Target: the GPU span of the step. Fit:
`python tasks/perf_model/fit_steptrace.py fused=trace/SPLIT/server_SPLIT.log
baseline=trace/LPT/server_LPT.log --mode MIXED` (5-fold CV; features et bs np ptok pf nd dkv kv mx).

| model class | SPLIT MAPE / p90 | LPT MAPE / p90 |
|---|---|---|
| linear(tokens, bs) | 24.8% / 49.4% | 30.7% / 61.9% |
| linear(all) | 13.4% / 25.0% | 16.9% / 31.1% |
| random forest | 6.2% / 15.3% | 6.7% / 17.6% |
| gradient boosting | 6.6% / 16.8% | 7.0% / 17.7% |
| MLP [128,64,32] | **6.0% / 15.6%** | 6.4% / 15.9% |
| MoE-4 linear | 5.7% / 18.1% | 5.7% / 18.7% |

**Reading.** The split's steps are more predictable in every class, but the margin is large
only for simple models: 20% lower error for linear, 0-8% for nonlinear ones. Goal point 3's
claim, "easier perf modelling with the fused/split kernel", holds in the weak form: a cheap
model is good enough. It does not hold in the strong form, that the split unlocks accuracy LPT
cannot reach. A lightweight MLP is about 6% on both. Mean mixed step: 36.7 ms SPLIT vs 38.7 ms
LPT.

**Where this left point 3 (settled 2026-10-08).** The claim was a chain: the split is easier to
model -> a better predictor -> better SLO control. Link one holds weakly (above). Link two is
**measured false in-node**: on the 4-mode MTP server, every predictor-driven chunk cap LOST to no
EDF at all, and the arm that won -- deadline ORDER -- uses no latency model whatsoever
(pd_colocation_vs_disagg.md point 3; chat TTFT -62%, codex P99 ITL -27%). Splitting prefill to
respect a predicted budget adds fixed-cost mixed steps that every decode row waits through, and
under steady load the deferred work has nowhere to go. So an "EDF-on-SPLIT vs EDF-on-LPT"
attainment comparison was not run: the in-node predictor has no decision left to make, which
makes the comparison vacuous rather than pending.

The predictor does earn its keep one level up, where a decision is a real choice between
alternatives rather than a delay of work that must happen anyway: the router's SLO veto. There it
is decisive, and only once it predicts the right quantity -- pricing decode rather than prefill
cut hot-prefix chat ITL 65% and its P99 80% (pd_colocation_vs_disagg.md point 1, v0.18.0).

<a id="four-mode-mtp-step--2026-10-07"></a>
### The 4-mode MTP step exists: prefill chunks and verify rows in ONE target forward — 2026-10-07

Supersedes the "4-mode MTP schedule does not exist in sglang" entries below: the reorder
they called for is now built (sglang `868025d6c`, fixes `10b378d89` and `5ab463414`, behind
`SGLANG_ALLOW_MIXED_CHUNK_SPEC=1 SGLANG_MIXED_SPEC=1`).

**How.** When the scheduler forms a chunked-prefill batch and a speculative running batch
exists, it attaches the running batch instead of calling `mix_with_running`, which is the
call that crashed. `EAGLEWorker.forward_mixed_spec` then does:
1. `draft(running)`;
2. `prepare_for_verify`, which allocates the verify slots and writes `req_to_token`;
3. ONE `ForwardMode.MIXED` target forward over [prefill tokens | draft tokens];
4. samples the prefill rows from their last-token logits;
5. runs sglang's own acceptance on the verify rows (`verify()` was split into forward +
   `_verify_post`, so this is the same code, not a copy);
6. runs both draft extends.

The scheduler then processes the prefill result and the decode result separately. The logits
processor keeps every verify token via `mixed_spec_verify_tokens`. Chain draft only
(topk 1): each verify row is then a causal extend, which is what the split kernel computes.

**Fork bugs found on the way** (stock sglang never reached them): the decode-only draft
backend lacked `qo_indptr`; the draft CUDA-graph runners borrow `capture()` without
mixed-CG state; the mixed-CG logits shortcut returned no logits for MIXED + FULL capture.

**Draft model:** `thoughtworks/Qwen2.5-7B-Instruct-Eagle3` (EAGLE3, draft vocab 32 000). Its
config declares `LlamaForCausalLM`; sglang needs `LlamaForCausalLMEagle3`, so the pod uses a
copy with that one field changed (`/dev/shm/w/qwen7b_eagle3_sgl`, weights symlinked). 3 steps,
topk 1, 4 verify tokens.

**Gate** (`drivers/srv_correctness.py`, greedy, against LPT without speculative decoding;
common prefix out of 101 tokens):

| arm | long prompts | short prompts | gate |
|---|---|---|---|
| LPT2 (noise floor) | 101, 52, 101, 101 | 101, 74, 11, 101, 101, 45, 45, 7 | PASS |
| MTP (stock EAGLE3) | 101, 52, 101, 101 | 101, 73, 11, 101, 82, 22, 36, 7 | PASS |
| MTPMIX (4-mode step, FA2) | 101, 101, 101, 42 | 101, 76, 11, 101, 101, 22, 36, 7 | PASS |
| MTPMIX_SPLIT | 101, 57, 101, 43 | 101, 74, 101, 24, 101, 5, 101, 7 | PASS |

0 fallbacks, 0 tracebacks in every arm. Mixing leaves stock MTP's short-prompt divergence
points (22, 36, 7) unchanged. **Thin coverage, stated plainly:** the gate's traffic produced
between 1 and 9 mixed steps per arm (the counter logs at 1, 10, then every 200). The serving
A/B (`drivers/mtp_repeats.sh`: MTP_LPT vs MTPMIX_LPT vs MTPMIX_SPLIT, 3 interleaved rounds)
exercises thousands and reports each arm's MIXED_SPEC count beside its timings.

<a id="realistic-serving-valid--2026-10-07"></a>
### Realistic serving, SPLIT vs LPT -- first VALID result, 3 interleaved rounds — 2026-10-07

Code: sglang `4e3bde083` (causal + non-ragged split, two-stream overlap on) on the pod;
flashinfer `b0563ae` (causal FA3, FA2 K/V ring 2/3/1). Correctness: the serving gate
(`drivers/srv_correctness.py`) PASSES for this build. Command:
`bash tasks/fused_specdec_workload/drivers/realistic_repeats.sh`. Rounds are interleaved
LPT, SPLIT, LPT, ... and each arm gets its own outdir. Scenario `agent_plus_shortchat`:
TraceLab codex agent replay + ShareGPT short chat at 12 req/s, Qwen2.5-7B-Instruct, H200,
no speculative decoding. Every arm: **0 fallbacks, 0 tracebacks.**
Summarised by `/dev/shm/w/repsum.py` (mean, [min-max]):

| stream | metric | LPT | SPLIT | delta |
|---|---|---|---|---|
| codex | mean TTFT ms | 25 069 [23 724-25 933] | 22 497 [21 250-23 309] | **-10.3%** |
| codex | P99 TTFT ms | 54 693 [54 585-54 761] | 49 045 [48 226-50 629] | -10.3% |
| codex | mean ITL ms | 35.9 [35.0-37.2] | 33.1 [32.3-33.6] | **-7.6%** |
| codex | P99 ITL ms | 501 [478-533] | 409 [404-411] | **-18.5%** |
| codex | output tok/s | 628.1 [621.7-633.4] | 663.6 [658.9-671.6] | **+5.6%** |
| chat | mean TTFT ms | 18 759 [18 600-19 030] | 15 720 [15 457-16 237] | **-16.2%** |
| chat | P99 TTFT ms | 60 758 | 54 130 | -10.9% |
| chat | mean ITL ms | 12.3 | 12.0 | -2.6% |
| chat | output tok/s | 858.2 | 858.3 | 0.0% (fixed offered rate) |

The ranges do not overlap on codex TTFT, ITL, P99 ITL, throughput or chat TTFT, so these are
real effects. **They do not meet the goal's bar** (>=30% TTFT and ITL, >=10% throughput).
That matches the arithmetic in the 30%-ceiling entry: the kernel cuts attention by 24% at
most, and the end-to-end gain is that cut times attention's share of the step. Reaching the
bar has to come from the schedule (4-mode MTP mixing, EDF), not from a faster attention
kernel alone.

<a id="split-serving-void--2026-10-07"></a>
### RETRACTION: the SPLIT serving result is void -- FA3 computed the wrong requests — 2026-10-07

**What was wrong.** `_plan_fused_split_inner` built the per-request arrays that `fa3_only_run`
consumes (`kv_lens_req`, `q_indptr_req`, `kv_indptr_req`, `n_req`, `max_pages`, `max_qlen`)
from `kl_r` / `qi_r` / `ki_r`. Those are the **persistent (FA2) side's** rows, `[n_rows:]`,
with the Q indptr rebased to 0. The FA3 executor selects its requests with
`reqs = arange(n_fa3_rows)`, so in serving it ran requests 0..n-1 **of the FA2 rows**: decode
queries at decode KV lengths, written into the first tokens of the output, while the real
prefill appends were never computed. The kernel harness never saw this because it passes
whole-batch arrays (`kv_c_pg`, `qo_c`, `kvl` over every row), and it gates every arm on PASS
against an fp32 reference. Serving had **no correctness gate**, so nothing caught it.

**Consequence.**
- The preliminary result below (codex TTFT -39.8%, P99 ITL -52.6%, throughput +16.9%;
  chat TTFT -41.5%) measured an arm that skipped most of its prefill attention. It is
  **void**, not preliminary. Skipped work is also the natural explanation for a gain larger
  than the kernel's own 19% ceiling, which the queueing-amplification story was invented to
  explain. That story is retracted too.
- The intermittent illegal memory access fits the same bug. The page table was sized
  `len(kl_r) * max(kl_r)` from FA2 rows, while the FA3 scheduler read Q and page offsets
  through mismatched arrays.

**Fix** (`flashinfer_backend.py`, this commit): whole-batch arrays, as in the harness:
`kv_lens_req = max(kv, 1)`, `q_indptr_req = qo_c`, `kv_indptr_req = kv_c`, `n_req = B`,
`max_pages = max(kv)`, `max_qlen = max(qo[:n_rows])`. The opt-in fused path reads the same
fields with the same whole-batch meaning.

**Gate added:** `tasks/fused_specdec_workload/drivers/srv_correctness.py`. One server per arm
with the A/B's exact args and env. Eight 2K-token prompts decode while four ~12K-token
prompts arrive, so the long prompts' final chunks land in mixed batches. Greedy, 96 tokens,
compared per request against LPT. LPT2 (a second LPT run) gives the noise floor. The
pre-fix build is run too, to show the gate catches it.

Result (common prefix with LPT, of 101 greedy tokens):

| arm | long prompts (FA3-routed chunks) | short prompts | gate |
|---|---|---|---|
| LPT2 (noise floor) | 101, 52, 101, 101 | 7-101 | PASS |
| SPLIT, pre-fix | 6, 6, 6, 5 | 5-7 | **FAIL** |
| SPLIT, plan fix | 7, 7, 6, 6 | 5-45 | **FAIL** |

The gate catches the pre-fix build, and **the plan fix alone does not pass it.** Every SPLIT
request keeps its first token, then diverges around token 6, which is when the long
prompts' chunks start sharing mixed batches with decodes. A second fault remains in the
serving split. Next: `SGLANG_SPLIT_VERIFY=1` recomputes every row of the first split steps
in fp32 inside the server and logs the FA3-row and FA2-row errors separately.

**Second and third faults (2026-10-07), found with `SGLANG_SPLIT_VERIFY=1`:**

1. **Non-causal.** The FA3 port was `Is_causal=false`, the split planned FA2 `causal=False`,
   and the harness reference was non-causal, so every kernel PASS validated the wrong mask.
   Append rows (qo 3809 onto kv 12000): rel err 0.14 (layer 0) / 0.76 (layer 14). Fixed:
   `FLASHINFER_FA3_CAUSAL=1` by default, causal harness reference and plans (flashinfer
   `b0563ae`), causal split plans (sglang `2b92b58bb`).
2. **Ragged mode.** After the causal fix the append row read 3.5e-3, but prefix-free FA3
   rows reported |got| = |ref| = **0 exactly**: the reference read zeros too. sglang's normal
   extend is *ragged*: a ragged causal kernel over the new tokens, plus the paged wrapper
   over the **prefix only** (non-causal), merged by LSE. The step's own K/V are written
   **after** attention. The split ran as that paged half but planned prefix+new causal, so
   it read the step's unwritten slots. The verify shared the same cache, so it agreed with
   the kernel; that is why the append row "passed". Fix: split steps take sglang's
   non-ragged mode (K/V saved first, one paged causal pass over prefix+new), which is what
   the kernel harness validates. sglang already uses that mode for steps with <=1 new token.
   Steps where the split plan then declines run FA2 non-ragged, which is still correct.

**Causal kernel numbers, TraceLab mixed batch** (`TL n_fa3:2` / `n_fa3:0`,
`drivers/ab_parallel.sh`, flashinfer `b0563ae`, FA2 ring 2/3/1; all PASS against the causal
fp32 reference, 0 unwritten, 0 fallbacks):

| arm | total | vs FA2-all |
|---|---|---|
| FA2 on everything | 14 943.8 us | |
| split, sequential | 11 419.2 us | **-23.6% (1.309x)** |
| split, two-stream overlap | 11 286.5 us | **-24.5% (1.324x)** |
| 512-token appends at 24-49K + 2 decode rows (`n_fa3:4`) | 2 582.6 us | PASS |
| no-prefix prefill + appends (`qo:2048,3809,1,1 prefix:0,8191,12000,5000 n_fa3:2`) | 1 425.9 us | PASS |

**Serving gate after both fixes (sglang `4e3bde083`): PASS.** In-server fp32 check, layers 0 and 14,
first 4 split steps: every FA3 row at 2.5-3.8e-3, including prefix-free rows (previously 0/0)
and the 3809-onto-12000 append; every FA2 row at 1.7-3.4e-3. Greedy common prefix with LPT,
of 101 tokens:

| arm | long (FA3-routed) | short |
|---|---|---|
| LPT2 (noise floor) | 101, 52, 101, 101 | 101, 74, 11, 101, 101, 45, 45, 7 |
| **SPLIT, non-ragged** | **101, 52, 101, 101** | 101, 76, 38, 101, 101, 5, 101, 101 |
| **SPLIT + overlap** | **101, 52, 101, 101** | identical to the row above |

0 fallbacks, 0 tracebacks. SPLIT agrees with LPT as closely as LPT agrees with itself, so a
SPLIT serving number can now mean something.

These supersede the non-causal 1.239x. With causal masking the overlap is worth -1.2%,
not -2.7%. The FA2 ring and split-reduction numbers were measured with
`fa2_stage_bench.py`, which was causal from the start, so they stand. The multicast
ceiling went through the non-causal harness and has to be re-measured causal.

(`fa3_only_run`'s `q_indptr_per_req.index({n_fa3}).item<int>()` host sync is only on the
`num_splits > 1` combine branch; serving passes 1, so it never runs there.)

<a id="multicast-backfill-ceilings--2026-10-07"></a>
### Multicast and cross-role backfill, measured as ceilings first — 2026-10-07

**Multicast ceiling.** `FLASHINFER_FA3_SKIP_KV_LOAD=1`
(`drivers/flash_attention_skip_kv_load.patch` on the pinned flash-attention `e9515d5`) keeps
the cp.async pipeline's commits but issues no K/V gathers, so FA3 runs at its compute-only
speed. Multicast can only remove part of the gather traffic, so this bounds it from above.
`drivers/ab_parallel.sh` (TWO_EXEC, upstream FA3, RS=1, stages 2), harness `EXTRA_DEFS`,
flashinfer `b13aa2a`:

| fixture | base total | KV loads skipped | ceiling |
|---|---|---|---|
| `qo:2048,2048 prefix:65536,98304 n_fa3:2` | 11 161.0 us (PASS, 0 unwritten) | 11 074.9 us | **0.8%** |
| `qo:512x4 prefix:32768,40960,49152,24576 n_fa3:4` | 2 639.0 us (PASS) | 2 583.6 us | **2.1%** |

**Causal re-run** (flashinfer `b0563ae`, causal reference): fixture 1 10 694.9 -> 10 504.4 us
(**1.8%**), fixture 2 2 519.8 -> 2 480.1 us (**1.6%**); base arms PASS, 0 fallbacks. Conclusion
unchanged: below 2%.

0 fallbacks in all four. The skip arms are garbage by construction (NaN rows), which also
proves the switch took effect. **Reading:** append rows at gqa 7 are 112 m-blocks per KV
head re-reading the same KV, mostly from L2, and FA3 is compute-bound on them. The 1.33-1.59x
DSMEM result was for an isolated, load-bound microbenchmark and does not transfer. Not
built: the upper bound is below the run-to-run noise of a serving A/B.

**Cross-role backfill.** With two executors, backfill is temporal: FA2's CTAs taking SMs as
FA3's tail CTAs retire. FA2 rows do not depend on FA3 rows, so no grid-dependency wait is
needed, and two streams overlap at least as much as PDL on one. On the TraceLab mixed batch
(`TL n_fa3:2`, FA2 now at ring depth 2/3/1):

| arm | total | |
|---|---|---|
| sequential | 12 282.5 us | PASS, 0 unwritten, 0 fallbacks |
| **two-stream overlap** | **11 954.7 us (-2.7%)** | PASS, 0 unwritten, 0 fallbacks |

This contradicts the 2026-10-06 note "concurrency loses at every budget". That note was
measured with SM budgets imposed, before the Theta(kv^2) fix. Without budgets, the overlap
fills FA3's tail. `cute::tma_store_wait` is not needed: the FA3 epilogue store is plain
tiled, not TMA. Serving still runs the executors sequentially. Moving them to two streams
is a server item.

<a id="fused-tree-build--2026-10-07"></a>
### Fused spec tree build: one kernel at the draft tail, not an attention epilogue — 2026-10-07

**RETRACTION of the design in [fused-epilogue-design](#fused-epilogue-design).** It said the
verify group's last arrival "already holding the reduced O and LSE" could build the tree.
The tree does not depend on attention output at all, and no attention kernel sits between
its producer and its consumer:
- inputs: `parent_list` and `selected_index` come from top-k over the draft scores, after
  the last draft step's LM head;
- outputs: `positions` feeds RoPE **before** the verify forward's first attention, and
  `tree_mask` is that attention's input.

So the epilogue it can legitimately be is the **draft step's**: the work between the draft
graph's last top-k and the verify forward. Today that is `organize_draft_results` (cat x3,
topk, sort, gather; inside the draft CUDA graph) followed by `build_tree_kernel_efficient`,
run **eagerly**: `fill_` or `full` of the mask, which in FULL_MASK mode is
O(seq_lens_sum x n) bytes; `full` of the retrive buffer; `empty`; `cat(verified_id)`; and
the kernel.

**What was built.** `python/sglang/srt/speculative/csrc/fused_tree_build.cu`: one CTA per
request does the top-(n-1) selection with torch.topk's ordering (NaN first, ties to the lower
index), the ascending sort, the token gather with `verified_id` prepended, and the full
build_tree semantics in all three mask modes. It also writes the mask prefix and the -1
retrive defaults that the reference gets from separate fill launches. Bit-packed mode is
capped at n<=32, as sglang's own dtype table is.

**Check.** `tasks/fused_specdec_workload/drivers/tree_build_check.py` (pod, Qwen env). The
reference is sglang's own `select_top_k_tokens` -> `organize_draft_results` ->
`build_tree_kernel_efficient` (sgl-kernel 0.3.16.post4). Fused outputs go into
garbage-filled buffers, so any byte left unwritten shows up as a mismatch. Configs (topk,
steps, n) = (1,3,4) (1,5,6) (4,3,8) (8,5,16) (4,5,32) (10,8,64) x bs 1/8/64 x modes x 3
seeds: **0 mismatching cases** across tree_mask, positions, retrive x3, draft_tokens and
top_scores_index.

| topk/steps/n | bs, seqlen | ref eager | fused eager | ref graph | fused graph (us) |
|---|---|---|---|---|---|
| 1/3/4 (MTP chain) | 32, 4096, FULL | 120.5 | 52.2 | 31.5 | **11.7** |
| 1/3/4 | 128, 8192, FULL | 121.8 | 52.2 | 34.8 | 14.1 |
| 4/3/8 | 32, 32768, FULL | 120.5 | 51.6 | 39.7 | 18.7 |
| 8/5/16 | 32, 4096, FULL | 122.6 | 53.6 | 45.0 | 37.4 |
| 8/5/16 | 128, 8192, QLEN | 123.0 | 54.2 | 49.1 | 34.4 |

"ref graph" captures both halves, which sglang does not do today. sglang runs build_tree
eagerly, so the saving it actually gets is roughly eager-ref minus graph-fused: about 100 us
per decode step. **Scale:** that is about 0.3% of a ~30 ms ITL. It removes a host
round-trip segment, which is what the 4-mode single-launch step needs, but it is not an ITL
lever on its own. At topk=8 the O(L^2) rank selection (L=264) is the kernel's own cost; a
warp bitonic select would cut it if that config is ever served.

**Not done yet:** wiring into `eagle_worker.draft_forward` (return the tree from the draft
graph). That is a server item, and it waits behind the MTP-in-4-mode blocker.

<a id="fa2-kv-ring--2026-10-07"></a>
### FA2 executor: K/V smem ring, and the split-reduction ceiling — 2026-10-07

**What was built.** `NUM_STAGES = 1` in FA2's traits was not a staging knob: SharedStorage
holds one K and one V tile, and the mainloop overlaps K(t-1)'s load with PV(t). The ring is
real: `KVStagedTraits<KT, S>` stacks S KV tiles along the row axis (slot boundary is a
multiple of 8 rows, so the 128B swizzle is unchanged), and the mainloop issues tile t-S into
the slot PV(t) just released, with exactly two commit groups per iteration so the waits are
`wait_group<2S-1>` (K) / `<2S-2>` (V). Per class, because the FA2 plan runs **2 CTAs/SM**
(~113 KB each): tile 128 fits 2 stages (96 KB), tile 64 up to 4, tile 16 none at its 128-row
KV tile (68 KB at depth 1).

**Method.** `tests/fa2_stage_bench.py`: the FA2 executor alone (`BatchAttention`, causal,
Qwen2.5-7B heads 28/4/128, page 128) against an fp32 reference. Each arm builds its own JIT
module (`FLASHINFER_EXTRA_CUDAFLAGS` + its own `FLASHINFER_WORKSPACE_BASE`), so no arm can
reuse a stale module. That was the failure that voided earlier FA2 items. 3 reps per arm.

flashinfer `3e41c2d` (ring) / `f54745e` (defaults); driver
`tasks/fused_specdec_workload/drivers/fa2_ab.sh` with `FIXTURES=decode,mid,prefill,agent,longdec,longmtp`.

| arm | decode | mid (MTP w4) | prefill | agent | longdec | longmtp |
|---|---|---|---|---|---|---|
| base (1/1/1), us | 914.3 | 651.8 | 3621.8 | 4624.4 | 805.8 | 1036.2 |
| **stages 2/3/1** | -0.5% | **-7.4%** | **-2.0%** | -0.5% | -0.2% | **-10.3%** |
| tile 64 only, 2 / 3 stages (2 reps) | -0.5 / -0.4% | -6.9 / -9.0% | -0.2 / +0.6% | +0.6 / +1.6% | | |
| tile 16 at MMA_KV=1, 1 / 2 stages (2 reps) | +5.0 / +2.0% | | | +3.3 / +1.6% | | |
| JIT rendezvous off (classic grid.sync) | +1.4% | -0.6% | +0.4% | -0.6% | +0.2% | -2.2% |
| **rendezvous + merge skipped (bound)** | +0.9% | -1.7% | +0.9% | +0.9% | -0.8% | **-2.9%** |
| no split (`FORCE_KV_CHUNK=1e6`) | -1.6% | +71.5% | +0.2% | -0.2% | +104.5% | +306.9% |

Fixtures: decode 96 x q1 at 2-32K; mid 48 x q4 at 2-32K; prefill 4 appends (512-2048 onto
8-32K); agent 64 decode + 12 tool-call prefills (40-600) at 4-48K; longdec 16 x q1 at
64-128K; longmtp 16 x q4 at 64-128K. Every arm 0 unwritten, PASS (rel err <= 4.6e-3), 0
fallbacks, except the skip arm, which is wrong on split rows by construction. Spread across
reps <= 1.6%.

**Reading it.**
- The tile-64 class (MTP verify) was latency-bound on its 32-row KV tile, and a 3-deep ring
  fixes most of it. Decode (tile 16) already moves 1.72 M tokens x 2 KB in 914 us = 3.85
  TB/s, ~80% of H200 HBM, so a deeper ring has nothing left to hide. The only way to fit
  one is a 64-row KV tile, which costs more than it hides.
- **Smem-handoff reduction: not built, because the ceiling is 2.9%.** Removing every
  rendezvous and every merge saves 2.9% on the most split-heavy shape and under 1% on
  plain long decode. A DSMEM handoff removes strictly less, since partials still have to
  meet, and it would add a co-scheduling constraint: a group's chunks would have to run on
  cluster-mates at the same time. Splitting itself is essential (2-4x without it). It is
  the merge's global round trip that is cheap: 28 x 128 x 4 B of partial per chunk against
  MBs of KV.
- Side finding: on longmtp the classic grid.sync + reduction pass beats our JIT rendezvous
  by 2.2%. The JIT merge runs on one CTA's 4 warps per group; classic spreads it over the
  grid. Not acted on: 2.2% of the FA2 half of an attention kernel.

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
- [x] **Smem-handoff JIT split-KV reduction — CLOSED 2026-10-07: ceiling 2.9%, not built** ([detail](#fa2-kv-ring--2026-10-07)). Original note (2026-10-06): Which
      shape splits is decided by upstream's heuristic, and on the TraceLab shape it answers
      cleanly: a 2048-token append at 64K KV already has `4 * ceil(2048*7/128) = 448`
      m-blocks against `0.8 * 132 = 106`, so no split is needed and the heuristic returns 1.
      A long-context **decode** row (q=1) has 4 m-blocks against 106, so it must split.
      Those decode rows route to **FA2** (tile 16, where the class boundary put decode
      because it is bandwidth-bound), and FA2 reduces its split partials through **global**
      memory. So the smem handoff is an FA2-side optimisation for long-context decode, and
      measuring it on FA3 prefill -- which is what every fixture so far did -- can only ever
      show nothing.
- [x] **(context) The same reduction on FA3** — CLOSED: the router sends FA3 no split rows. Original note: only reachable on low-q/high-KV rows, which
      the router deliberately sends to FA2. Keep as the comparison point against upstream's
      combine launch, not as the main claim.
      replaces upstream FA3's separate combine launch and FA2's **global** partial
      round-trip. Intra-role, so no partial crosses executors. Only measurable on
      split-heavy shapes (few q rows, long KV); every fixture used so far has
      `num_splits=1`, where it saves nothing.
      [detail](#what-parity-means-on-a-split-batch--2026-10-06)
- [x] **DSMEM / KV multicast — CLOSED 2026-10-07: causal ceiling 1.8% / 1.6%, not built** ([detail](#multicast-backfill-ceilings--2026-10-07)). Earlier note: the real prerequisite, found 2026-10-07: **our FA3 launcher
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
- [x] **(superseded) DSMEM / KV multicast, TMA framing** — 1.33-1.59x in isolation. **No TMA prerequisite**: DSMEM
      peer reads (`mapa.shared::cluster`) work regardless of how KV reached smem, so one CTA
      can `cp.async` the KV block and the rest of the cluster reads it from that CTA's smem.
      TMA multicast is one *implementation*, not a requirement, and since cp.async beats TMA
      paged by 9.12% the item should be built on the cp.async path. Earlier entries calling
      TMA paged a prerequisite are retracted; that mistake is why this looked blocked.
      The old `FLASHINFER_FUSED_CLUSTER` flag belongs to the closed fused dispatch.
      [detail](#fa3-is-31x-slower-than-fa2-at-realistic-context-and-the-121x-does-not-generalise--2026-10-06)
- [x] **Fused spec tree-build epilogue — built 2026-10-07** as a draft-tail kernel (the attention-epilogue design is retracted), exact vs sglang, and wired into the worker behind `SGLANG_FUSED_TREE_BUILD=1` (`f1985ca8f`). Serving gate (2026-10-07): PASS vs
      LPT (long 101/52/101/43); vs stock MTP **10 of 12 requests identical to 101 tokens**, l3 diverges at 43
      and s1 at 77. Late divergences of this kind also appear LPT-vs-LPT (batching timing), so they are
      not evidence against the kernel's bit-exactness (~150 cases, kernel level). **Confirmed 2026-10-07:** a
      second stock-MTP run diverges from the first at exactly the same points (long 43, short 77), so the
      fused build is serving-level exact, within the spec path's own run-to-run noise. 0 fallbacks. [detail](#fused-tree-build--2026-10-07)
      [detail](#fused-epilogue-design)
- [x] **Async/tiled packed copies — SUPERSEDED 2026-10-07.** These were the fused kernel's wgmma-role
      copies (single-warp width, per-row O store). The fused design is closed on measurement, and the
      two-executor FA3 is upstream's own kernel with its own PackGQA copy path, so the item has no
      code left to apply to. Not skipped: there is nothing to build.
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
- [x] [Unified queue](#unified-serving-perf) — closed, superseded by the two-executor split (2026-10-06): numerically correct but **-66% measured on a clean tree** (0.335x throughput, 3.01x ITL), [cause found](#why-fused-is-slow) — register starvation: `mma.sync` at ~168 regs instead of ~255, from FA3's 384-thread launch bounds

## Workload and measurement setup — 2026-10-06

One line each; detail in [`realistic_workload_analysis.md`](realistic_workload_analysis.md).

- (merged) **Context length where attention is >50% of e2e** — tracked in the task list at
      the top (`drivers/context_share.sh`, yarn x4 to 131 072 as in serving).
- [x] (2026-10-07) every serving A/B since runs `agent_plus_shortchat` / `agent_chat_rated` / `agent_chat_saturate`. **TraceLab mixed scenarios, not `--dataset-name random`** — `run_mixed_workloads.py`
      carries the real shapes (`agent_plus_shortchat`, `loadsweep_*`); a `SPLIT` method is
      wired into it. Independent-prompt datasets average 242-690 tokens and omit the
      multi-turn tool-result prefills that create the repeated mixed forwards this kernel
      exists for.
- [x] **Why the 2048-token runs could not show the bar** — attention is a small fraction of
      the forward there, so a 1.21x attention win is invisible end to end regardless of
      correctness. Those runs measured in a regime where >=30% is unreachable by
      construction.
- [x] (2026-10-07) every arm since reports fallbacks + tracebacks, and serving arms pass `srv_correctness.py`. **Validity gate on every arm** — report the fallback count beside the timings. Two
      serving A/Bs were invalid (121 then 122 silent fallbacks) and both printed
      ordinary-looking numbers; the backend now raises instead of degrading.

## Server integration — 2026-10-05

- [x] **Two-executor split is the default** — fused path opt-in via `SGLANG_FUSED_KERNEL=1`.
      [detail](#two-executors-beat-fa2-on-everything-and-pass--2026-10-06)
- [x] **Silent fallback removed** — the backend raises instead of degrading.
      [detail](#workload-and-measurement-setup--2026-10-06)
- [x] (2026-10-08, 3 rounds, see the task list at the top: full stack beats stock MTP's best on every metric at chunk 2 048) **Long-context A/B on the real workload** — no valid serving number yet; both prior
      A/Bs were invalid.
- [x] **4-mode MTP serving run** — unblocked and measured 2026-10-07 (`forward_mixed_spec`), see the
      task list at the top. The original block on `mix_with_running`: [detail](#four-mode-blocked)
- [x] **The bar: >=30% TTFT/ITL, >=10% throughput** (2026-10-08). Full stack vs stock MTP, 3 rounds, chat
      24/s: TTFT -95%/-51% and ITL -42%/-38% (chat/codex). Capacity +19.8% closed-loop (v0.16.0). At
      chat 16/s every ITL metric still clears it, but codex TTFT is only -19%.
- [x] **CUDA-graph capture -- ceiling measured, then a host-cost finding** (2026-10-07). Traced serving, one
      round each (`SGLANG_STEP_TRACE=1`, `tasks/perf_model/steptrace_gap.py`, sglang `c1d2bceda`):

      | mixed steps | LPT | SPLIT |
      |---|---|---|
      | steps | 2 144 | 2 154 |
      | GPU per step | 38.7 ms | 36.7 ms (-5.2%) |
      | host gap per step | 0.96 ms (2.4%) | **1.79 ms (4.6%)** |

      Decode steps are already graphed (gap ~0). The split's extra host work, two plans and the
      page-table scatter, costs +0.83 ms per mixed step, about 40% of its 2.0 ms GPU saving. Graph
      capture or a cheaper split plan can recover at most that, ~2% of mixed-step time. The step-span
      events include in-step launch stalls, so this understates launch cost; the clean ceiling is LPT
      with its mixed CUDA graphs on vs off.
      **Host-overhead fixes, measured** (one traced round each, all gate PASS, 0 fallbacks): (1) skip
      flashinfer's FA3 plan + sparse indices in two-executor mode (`e237ab8bc`): 1.79 -> **1.54 ms**/mixed
      step; (2) no device syncs in the kv-index build, grow-only scratch, host-side split counts
      (`144a73dc8`): **1.71 ms**, the same as (1) within single-round noise (+/-0.2 ms). The ~0.6 ms that
      remains over LPT is mostly the FA2 plan the split needs every step. Stopped here: the whole bound
      is ~2% of mixed-step time. (2) is kept for its correctness part: the plan cache is now keyed on
      exact row lengths, so a sum collision can no longer reuse a stale plan.
      Original note: needs the routing decision in device memory with exit-early
      CTAs so grid dims stay fixed.
- [x] **Per-step tracing** — `SGLANG_STEP_TRACE=1` localises the gap to MIXED steps.
      [detail](#spec-path-where-the-cpu-gpu-syncs-actually-are--2026-10-05)

## Schedule exploration — 2026-10-05
- [x] [EDF predictability, fused vs baseline](#edf-model-classes) — parity at ~1.5% MAPE with an adequate model class; the linear-fit falsification was an artefact
- [x] **SM-split / cluster-cap sweep on the correct causal split** (2026-10-07). TraceLab batch, `drivers/ab_parallel.sh`,
      flashinfer `b0563ae`, all PASS, 0 fallbacks. Sequential: FA2 cap 264 = 11 388.6 us, 160 = +1.0%, 136 = +2.1%.
      With overlap (the serving default): 264 = 11 393.8, **160 = 11 318.9 (best)**, 136 = 11 327.3, so serving's
      `min(2*(132-sm_fa3), 160)` cap is already at the best point. Capping FA3 is harmful: 96 SMs +35%, 64 SMs
      +87%. Overlap's net gain on this batch is within noise (-0.6%). Chunk size is a serving knob; see
      the chunked-prefill sweep.
- [x] Acceptance-length (EWMA) as a perf-model input (2026-10-07): 2.10 tokens/step. a=1 collapses EDF; a
      per-request EWMA is biased low through the min over requests; a global EWMA is right, but
      chunk caps lose anyway. See pd_colocation_vs_disagg.md point 3.

## Multi-node schedule tests (H20 only, <= 16 GPUs, trial `vibecuda`) — 2026-10-08
Measurements live in [`pd_colocation_vs_disagg.md`](pd_colocation_vs_disagg.md).
- [x] Route to highest cache-hit node first, with an SLO veto (2026-10-08) -- affinity cuts TTFT 93% / 99.5% vs 2P2D; the calibrated veto fires on 2% of requests and trims chat P99 TTFT 15%. A hot-prefix stress test is running.
- [x] Best-effort cache migration (2026-10-08) -- L3 store lifts round-robin's hit rate 12.7 -> 22.3% (chat TTFT -41%); idle under affinity
- [x] EDF predictor with acceptance length (2026-10-08) -- measured on the MTP server; deadline order wins, chunk caps lose
- [ ] Two-way RDMA / congestion -- no contention at 16 GPUs; hot-prefix experiment queued

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
- [x] (closed: every measurement since 2026-10-06 ran on the restored `f5f0b2f6d81d1e33` tree) **Re-provision the working set** — `/dev/shm/w` is empty: venv (torch 2.8/cu128),
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
- [x] (closed, see above) [Re-establish the GPU working set](#pod-restore) — fork tree pushed (805 files, byte-exact); three mechanical gaps left, listed in `drivers/provision_trees.sh`: spdlog unpushed, stale absolute symlinks in `flashinfer/data`, and the csrc jinja unreadable — the 2026-10-05 trial expired; a replacement single-GPU `vibecuda` is running and empty

- [x] [The restored pod's serving tree was upstream, not our fork](#env-contamination) — rebuilt and verified by manifest before any serving number was taken

## Workload capture for handoff — 2026-10-05
- [x] [Real online 2-mode trace](archive/fused_kernel_handoff.md#5-the-workloads) — 2121 steps/arm, JSON corpus in `fused_specdec_workload/sets/`
- [x] Real online **4-mode** trace with draft+verify co-resident (2026-10-07) — `mtpedf/rate24/trace_MTPMIX_SPLIT` (STEPTRACE, 11 837 steps)
- [x] Long-context (64-96K, tracelab) A/B — 2026-10-08, see the task list at the top (`long_agent_chat`). Sized by the context-share measurement. The old gate
  (split FA3 rows) is gone: the router sends FA3 no split rows, and the "corruption" was the harness
  reference.

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
## Realistic serving A/B, SPLIT vs LPT -- PRELIMINARY, single run each — 2026-10-07

> **VOID (2026-10-07).** This run's SPLIT arm computed the wrong requests on FA3 (the split plan
> passed the FA2 rows' per-request arrays). Every SPLIT number below measures skipped prefill
> attention, not the split. See [split-serving-void](#split-serving-void--2026-10-07).

Same scenario as the LPT baseline below (`agent_plus_shortchat`, 1x H200, Qwen2.5-7B, RoPE
extrapolated -- timings valid, text not). Both arms 0 fallbacks.

| codex (agent replay) | LPT | SPLIT | delta |
|---|---|---|---|
| mean TTFT | 25 330 ms | 15 241 ms | **-39.8%** |
| P99 TTFT | 55 143 ms | 36 976 ms | **-32.9%** |
| mean ITL | 35.85 ms | 29.83 ms | -16.8% |
| P99 ITL | 525.6 ms | 249.3 ms | **-52.6%** |
| output throughput | 625.8 tok/s | 731.6 tok/s | **+16.9%** |

| chat (open-loop, 12/s) | LPT | SPLIT | delta |
|---|---|---|---|
| mean TTFT | 18 661 ms | 10 922 ms | **-41.5%** |
| P99 TTFT | 60 088 ms | 42 215 ms | -29.7% |
| mean ITL | 12.07 ms | 11.90 ms | -1.4% |
| P99 ITL | 40.62 ms | 41.79 ms | +2.9% |
| output throughput | 858.2 | 858.3 | flat -- open-loop at a fixed rate, cannot move |

**This contradicts my own prediction** ("low single-digit gain", retracted above). Likely
mechanism, unverified: queueing amplification. TTFT in the tens of seconds says the server
runs near saturation, where a ~19% faster mixed step removes far more than 19% of queueing.

**Against the bar:** mean TTFT >=30% on both streams; throughput >=10% on the closed-loop
codex stream (chat cannot show it); mean ITL is NOT at 30% (-16.8% codex, flat chat), though
codex P99 ITL halves.

**Why this is preliminary, not a result:**
- one run per arm, not interleaved -- variance unknown, and queue-dominated metrics are the
  ones most sensitive to run-to-run differences. Needs >= 3 interleaved repeats;
- the SPLIT arm crashed with an illegal memory access in 1 of 2 asynchronous runs (it also
  passed once under CUDA_LAUNCH_BLOCKING with per-executor syncs) -- intermittent, open;
- no MTP; the forward-mix counter is not comparable across arms (the fused path counts per
  layer, LPT per step).

Command: drivers/realistic_ab.sh (LPT arm) and its SPLIT-only variant, vibecuda trial
f5f0b2f6d81d1e33 (1x H200). flashinfer b45b0ec, sglang 91055ec2d (+ fault-dump debug code,
inactive unless SGLANG_SPLIT_FAULT_DUMP is set).

## Realistic serving A/B, LPT baseline (TraceLab agent + short chat) — 2026-10-07

Scenario `agent_plus_shortchat` (`run_mixed_workloads.py`): coding-agent TraceLab replay
(60 requests, 12 sessions, 32-49K contexts, >=512 output) concurrent with 2400 short-chat
requests at 12/s. Qwen2.5-7B, 1x H200, `--context-length 131072` (RoPE extrapolated past the
native 32K: timings valid, generated text not). Driver: `drivers/realistic_ab.sh`.

**VOID first attempt.** The first LPT arm ran with the device-side TRACE printf still enabled
(3.8 GB server log, mean TTFT 26-36 s measuring printf). Discarded; the runner now refuses
to start with any diagnostic switch in its non-production state.

**LPT (persistent FA2 on everything), clean, fallbacks 0:**

| stream | requests | duration | output tok/s | mean TTFT | P99 TTFT | mean ITL | P99 ITL |
|---|---|---|---|---|---|---|---|
| codex | 60 | 120.1 s | 625.8 | 25 330 ms | 55 143 ms | 35.85 ms | 525.6 ms |
| chat | 2400 | 200.4 s | 858.2 | 18 661 ms | 60 088 ms | 12.07 ms | 40.62 ms |

**Forward-step mix under LPT (n = 18 000 steps):** `decode_cuda_graph` 88.3%,
`mixed_persistent` 10.6%, `mixed_batch_prefill` 0.9%, `pure_prefill` 0.1%.

**RETRACTED (2026-10-07, user correction): the "split only touches ~11% of steps, so expect a
low single-digit gain" reading below.** Two errors:

1. It generalised from one replay. `agent_plus_shortchat` issues agent rounds back to back and
   dilutes them with 12/s short chats. A real TraceLab-style turn carries several small
   tool-result prefills, so mixed steps are the common case, not 11%.
2. It ignored spec decoding. Under MTP every "decode" step is a VERIFY step with
   q = draft_len + 1 per request -- the small-q class the 4-mode kernel is designed for, with
   the draft step and the tree-build epilogue fused as well. The 88% pure-decode CUDA-graph
   share exists only because this run had no spec decoding; it is not a property of the
   workload the goal specifies.

What still stands: 1.239x is the measured gain of the SPLIT ALONE on one TraceLab batch. The
4-mode kernel (verify + tree epilogue) is not built, so its gain is unmeasured and no ceiling
should be placed on it from this run. The forward mix here bounds THIS scenario without MTP,
nothing more.

Multi-node context recorded: at most 4 Delta nodes are available, so the disagg baselines are
2P2D and 1P3D.

**Superseded reading: that mix bounds what the split can do here, well below the 19.3% kernel ceiling.** The split
kernel only runs on mixed attention steps -- about 11% of forwards. The other 88% are
pure-decode CUDA-graph steps, which the split does not touch. And TTFT in the tens of seconds
means this load is queue-dominated, so TTFT moves with throughput and admission policy more
than with per-step kernel time. The SPLIT arm (running) measures the actual effect; the
expectation it is checked against is "small", and if it comes back large that needs
explaining, not celebrating.

## Decode-only: FA2 beats FA3 by 1.30x, so the routing is already right — 2026-10-07

The measurement that locates the remaining attention headroom, since decode is effectively the
whole cost of the mixed batch (its FA2 half is 15 394 of 15 396 us). 18 decode rows,
65-98K contexts, identical work, 0 unwritten and 0 fallbacks in both arms:

| arm | total |
|---|---|
| **`dec_fa2`** (n_fa3=0) | **732.6 us** |
| `dec_fa3` (n_fa3=18) | 955.2 us |

**FA2 is 1.30x faster than FA3 on decode rows.** Combined with FA3 being 1.38x faster on
append rows, the split's existing routing -- appends to FA3, decode to FA2 -- is **already the
correct assignment in both directions**. There is no headroom to recover by moving decode to
FA3, which closes the question the previous entry left open and leaves the **19.3% ceiling
standing**.

That also explains the shape of the split's win: it is not that FA3 is a better kernel, it is
that each executor is faster on one row class and the split routes accordingly. FA3 wins
large-q (many packed rows per tile, wgmma-friendly); FA2 wins q=1 (its 16-wide q tile wastes
far less than FA3's 128-wide one on a 7-row packed decode).

**CAVEAT, and it is serious: both arms fail numerics on this fixture.** rel err runs 0.047 to
0.31, growing monotonically with row index, in the **FA2-only** arm as much as the FA3-only
one. That is the same index-growing signature as the open corruption, far stronger here than
the 0.021-0.031 seen on the mixed batch. So:

- the **timings are between two equally-corrupted runs**, which makes the 1.30x suggestive
  rather than conclusive -- though both arms do identical work and fail similarly, so a
  directional read is defensible;
- more usefully, **this fixture is a much better reproducer than the one used for the
  corruption hunt**: 17 of 18 rows fail, errors up to 0.31, and no FA3 executor is involved in
  the `dec_fa2` arm at all. Any future work on that bug should start here rather than from the
  two-row `qo:1,512` case.

**The corruption ACCUMULATES with the number of preceding split rows.** Every row identical
(`qo=1, kv=65537`), FA2 only, so nothing varies but the count:

| decode rows in batch | last row's rel err |
|---|---|
| 2 | 0.058 |
| 4 | 0.100 |
| 8 | 0.137 |
| 18 | **0.367** |

Row 0 is clean in every case (~0.003). So this is not a fixed mis-index that hits one row --
**each preceding split row degrades the next a little more**, monotonically, on identical work.

**That revives a candidate I wrongly treated as eliminated.** The `PI_ZERO` arm zeroed
`partial_indptr` only for **non-split** work items. In this fixture *every* item is split, so
that arm changed nothing here and never tested the split path's offset at all. The quantity
that accumulates across rows is exactly `partial_o_nnz` -- the running partial-buffer cursor
that each split row advances -- so a drift in how split items index `partial_o` would produce
precisely this signature: clean first row, error growing with the number of rows that have
already advanced the cursor.

That is the sharpest lead this bug has had, and it is a different quantity from everything
ruled out so far (the plan's arrays were verified correct as *values*; this is about the
offset each row's merge *reads back* from them).

## The split cannot reach the 30% bar on attention alone — 2026-10-07

Arithmetic on the measured numbers, before spending more time on serving runs.

The split cuts attention time by **19.3%** on the mixed TraceLab batch (15 395.8 -> 12 425.4 us,
both 0 fallbacks / 0 unwritten). End-to-end TTFT/ITL gain is that reduction times attention's
share of the step:

| attention share of step | e2e gain |
|---|---|
| 50% | 9.6% |
| 70% | 13.5% |
| 90% | 17.4% |
| **100%** | **19.3%** |

To reach **30%** e2e the attention share would have to be **1.55**, i.e. more than the whole
step. **So the split as measured cannot hit the 30% TTFT/ITL bar, even if attention were 100%
of the step.** The ceiling is 19.3%.

This is not a statement about serving noise or integration overhead -- it is an upper bound
that no serving measurement can exceed, so running the serving A/B cannot change it. It also
prices the remaining kernel items against the gap: the whole validly-measured FA3 item set is
under 4% and the FA2 set is flat, so stacking them onto the split moves the ceiling to roughly
**22-23%**, still short.

**What this means for the plan, stated plainly:**

- The **>=10% throughput** half of the bar is reachable: 19.3% attention reduction at a 50-90%
  attention share gives 9.6-17.4%, and throughput tracks step time directly.
- The **>=30% TTFT/ITL** half is **not** reachable from the attention kernel alone. It has to
  come from somewhere the kernel work does not touch -- scheduling (EDF admission, better
  batching), cache hits avoiding prefill entirely, or spec-decode acceptance raising tokens
  per step. Those are the *scheduling* items in this plan, not the kernel items.
- On identical append rows FA3 takes 11 164 us against FA2's 15 437 us, and the mixed batch's
  FA2 half is 15 394 us of a 15 396 us total.

**RETRACTED, same day, by arithmetic on the unwritten count: FA3 does NOT fail on decode rows
-- it fails on the LARGE ones.** I wrote that "the FA3 executor cannot serve `qo=1` rows at
all (5632/5650 unwritten)" and built a recommendation on it. The split is the exact opposite:

    total output rows            = 2048+2048+512+1024 + 18x1 = 5650
    unwritten                    = 5632 = 2048+2048+512+1024   <- the four LARGE rows
    written                      = 5650 - 5632 = 18            <- the 18 decode rows

So with all 22 rows routed to FA3, it wrote **only** the decode rows and dropped every
large-q row. The capability is the reverse of what I claimed, and the "make FA3 serve decode
rows" direction is void -- it already does.

What the real failure is remains open: the four dropped rows are exactly those with
`qo > 1`, which are the rows FA3 serves correctly when `n_fa3=2`. So the fault appears when a
22-request plan is built rather than being a property of the row shape -- a plan-capacity or
`num_blocks` limit, not a missing capability. `num_blocks` is
`ceil_div(max_seqlen_q * qhead_per_khead, 128)` = 112 here, which is sized from the LARGEST
row, so a plan holding 22 requests may be exhausting per-request work-item capacity for the
large ones.

The honest version of the headroom point: FA3 is **1.38x faster than FA2 on identical append
rows**, and decode rows dominate the mixed batch's time. Whether FA3 can be made faster than
FA2 *on decode rows* is unmeasured -- it serves them, but no FA3-vs-FA2 decode-only timing
exists. That measurement is cheap (a decode-only fixture, both executors) and is the thing
that would actually settle where the remaining attention headroom is.

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

**BEST REPRODUCER (2026-10-07), replacing the two-row one below.** Decode-only, FA2 only,
no FA3 executor involved at all:

    qo:1 x N   prefix:65536 x N   n_fa3:0   UNIFIED=1   page 128

    N=2  -> last row 0.058      N=8  -> last row 0.137
    N=4  -> last row 0.100      N=18 -> last row 0.367

Row 0 is clean (~0.003) at every N. Rows are **identical**, so nothing varies but how many
split rows precede. 17 of 18 rows fail at N=18, against 0.021-0.031 on the mixed batch -- a
far stronger signal, and it removes FA3, the split and the task-class question from the
picture entirely.

**MEASURED: the partial-buffer cursor is CORRECT, and the real variable is `len_kv_chunk`.**
Tracing `o_indptr` and `len_kv_chunk` per work item on this fixture:

| | `len_kv_chunk` | nchunks | `o_indptr` step per row | expected `row_tile_size x nchunks` |
|---|---|---|---|---|
| N=2 | **2 048** | 33 | 231 | 7 x 33 = **231** ok |
| N=18 | **17 920** | 4 | 28 | 7 x 4 = **28** ok |

Exact in both, so `partial_o_nnz` / `o_indptr` is **refuted** -- the cursor hypothesis from the
previous entry is wrong.

What the trace shows instead is that **the plan adapts `len_kv_chunk` to the batch size**:
2 048 at N=2, 17 920 at N=18 (more rows already supply parallelism, so each row is split into
fewer, larger chunks). And the error tracks the **chunk size**, not the row count:

    chunk  2 048 -> rel err 0.0495
    chunk 17 920 -> rel err 0.3058

**Which points at a known property of this port: `partial_o` is `DTypeO` (bf16), while
upstream's `ElementPartial` is `float`.** Each chunk's partial output is rounded to bf16
before the merge, so the error per partial grows with how much KV that chunk accumulated.
Larger chunks -> fewer, bigger partials -> more loss. That fits both numbers, and it also
re-reads the "grows with row index" signature: later rows are assigned later, larger chunk
spans by the LPT-ordered scheduler, not corrupted by their predecessors.

**RETRACTED by the chunk sweep: the error does NOT track chunk size.** Holding N=18 fixed and
forcing the chunk (`FLASHINFER_FORCE_KV_CHUNK`):

| chunk at N=18 | rel err |
|---|---|
| auto (17 920) | 0.305 |
| 4 096 | 0.391 |
| 1 024 | 0.375 |

Flat to slightly worse as the chunk shrinks, so chunk size is **not** the variable. Against
N=2's 0.0495 at chunk 2 048, that leaves **row count** as the driver -- which is what the
cumulative sweep said in the first place. My previous conclusion inverted it: N=2 and N=18
differed in *both* row count and chunk size, and I attributed the error to the wrong one
without holding either fixed. This sweep holds N fixed and the chunk dependence disappears.

So **bf16 partial precision is not the primary explanation either** -- it cannot produce
row-count dependence, since decode rows are independent of one another. Both of my last two
mechanism claims are void:

- "cumulative via the partial_o cursor" -- refuted, the cursor arithmetic is exact
  (`o_indptr` steps by `row_tile_size x nchunks` in both N).
- "precision loss from bf16 partials, tracking chunk size" -- refuted here.

**SEPARATION RUN (N vs total KV), and both of my axes were wrong:**

| arm | rows | kv/row | total KV | rel err |
|---|---|---|---|---|
| A | 18 | 4 096 | **73 K (smallest)** | **1.205 (worst)** |
| B | 2 | 65 536 | 131 K | 0.058 |
| C | 18 | 65 536 | 1.18 M (largest) | 0.313 |
| D | 4 | 32 768 | 131 K | 0.117 |

- **Total KV refuted**: A has the smallest total and the worst error.
- **Row count alone refuted**: A and C share N=18 and differ 4x.
- Anchor: B -> D doubles rows, halves kv, and the error doubles exactly (0.058 -> 0.117).
  Error grows with N and with 1/kv_per_row.
- A's rel err 1.2 is the "right magnitude, uncorrelated values" regime -- the signature of
  rows attending the WRONG KV, not of accumulated noise. A roughly fixed number of
  misdirected chunks fits: at A's ~2 chunks/row one wrong chunk is ~50% of the output
  (err ~1.2); at C's 4 chunks/row, ~25% (err ~0.31).

**PROBE RESULT (arm A, N=18, kv=4096): the fault is ONE chunk of ONE row reading the wrong
pages.** Per-head attended-row decode, all 28 heads:

- rows 0-16: every head attends exactly its own row (decode = r +- 0.04, bf16 rounding).
  Their sub-0.03 rel errs are quantisation noise, not misdirection.
- **row 17, the LAST row: all 28 heads identically wrong**, mixture ~= 12.7 instead of 17.

Uniform across heads exonerates `kv_head_idx`. The broken coordinate is the KV source of a
chunk. And the number decodes: 12.7 ~= (17 + 8.5)/2 -- about half the mass on a chunk whose
pages sit at row 8.5's offset. Row 17's `kv_indptr` is 544 pages; **halved = 272 = row 8.5's
pages exactly**. So the working hypothesis -- one datapoint, needs arm C -- is a **halved page
offset on the tail work item**.

**Arm C result, and the slopes crack it.** Rows 0-16 decode clean at 2 d.p.; row 17 = 16.68.
But the const-V rel-err slopes across rows are linear in r with

    arm A (kv=4096):  rel err ~= r x 1.8e-3
    arm C (kv=65536): rel err ~= r x 1.1e-4      ratio ~= 16 = 65536/4096

So the per-row deviation scales as **r x 128/kv**: each row leaks attention mass equal to
about **one 128-token page -- one KV tile -- per preceding row**, drawn from the batch start,
and the last row additionally misses at CHUNK scale (half its mass in arm A; 16.68 vs 17 in
arm C). This also retro-fits the random-V magnitudes: one wrong 128-token tile out of 65 536
gives ~sqrt(128/65536) ~= 0.045 ~= the observed row-1 error, rising to ~0.3 by row 17.

**RETRACTED (2026-10-07): the "half page offset, fix-ready" entry below.** Its key number --
"row 17 attends mean page ~434.6" -- was **not decoded from output**; the ramp run's per-head
lines never printed, and I back-computed the figure from the rel err. The actual got/ref
values, pulled afterwards, show something else entirely:

| row | kernel mean page | reference mean page |
|---|---|---|
| 0 | 15.56 | 15.55 |
| 1 | 48.52 | 47.49 |
| 2 | 81.18 | 79.53 |
| 8 | 279.54 | 271.70 |
| 16 | 541.99 | 527.63 |

The **kernel** advances ~33 pages per row -- correct, since 4097 tokens occupy 33 pages in the
page-aligned layout. The **reference** advances ~32, i.e. it reads token-contiguous KV.
**The fault is in the test harness, not the kernel:** `exact()` did `kvb += kl` while each
request actually starts on a page boundary, so every row after a non-page-multiple row was
compared against the wrong reference. A second harness bug under-allocated pages
(`ceil(sum/PS)` instead of `sum(ceil/PS)`: 585 vs 594 at N=18), which is the last-row miss.

That one finding re-explains most of the "corruption" record: error growing with row index,
`kv=1024` passing and `kv=1025` failing, `qo:128` passing and `qo:127` failing, the TraceLab
decode rows (all `prefix+1`) at 0.021-0.031, and every candidate that "changed nothing" --
they changed nothing because the kernel was right. Both fixes are no-ops at `PAGE_SIZE=1`,
so the single page-1 failure on record (0.589) is still unexplained and is being re-run.

**VERIFIED: with the harness fixed, every fixture passes -- the kernel was correct.**

| fixture | buggy reference | fixed reference |
|---|---|---|
| decode N=18, kv=4096 | 1.205 FAIL | **PASS** |
| decode N=18, kv=65536 | 0.31 FAIL | **PASS** |
| `qo:1,512 prefix:1024,512`, page 128 | ~0.47 FAIL | **PASS** |
| same, **page 1** | 0.589 FAIL | **PASS** |
| `qo:127,512 prefix:65536,512` | 0.067 FAIL | **PASS** |
| TraceLab FA2-all | rows 20/21 FAIL | **PASS** |
| TraceLab split | rows 20/21 FAIL | **PASS** |

All "every row correct at every split, nothing left unwritten". The page-1 case also passes
now, although the harness fixes are no-ops at page 1, so its earlier failure (0.589) had a
different, transient cause. The likeliest is the stale JIT module documented above -- that run
was taken before `cached_ops` was cleared -- but that is **unverified** and recorded as such.

**Consequences:**
- the serving A/B is **unblocked**; it was gated on a kernel corruption that did not exist;
- the "both arms must plan the same task count" validity requirement was motivated by that
  corruption and is withdrawn -- the 4-task split plan computes correctly;
- the corruption hunt's ~25 eliminated candidates were mostly eliminated *correctly*, because
  there was no kernel fault for them to be.

**Superseded entry follows.**

**PAGE-RAMP DECODE (KV_ROWCONST=2) -- the wrong tiles sit at HALF the correct page offset.**

- Row 17 attends mean page ~434.6 against an own-range mean of ~560. A 50/50 mixture solves
  the wrong half to pages ~280-310, and **half of row 17's page base is 561/2 = 280.5** --
  the misread tiles are at exactly half the correct offset.
- Rows 0-2 drift toward earlier pages by amounts growing with r: 1-2 tiles per row read at
  half-base (distance ~ r/2 rows) reproduces the measured const-V slope.
- Damage spans 25-28 of 28 heads uniformly, reconfirming the broken coordinate is the tile's
  PAGE ADDRESS, not any head index.

**FIX-READY STATEMENT: some KV tiles on the cp.async paged path are addressed at HALF their
correct page offset** -- a /2, a >>1, or a byte-vs-element (bf16 = 2 bytes) confusion in one
tile-address expression, hitting ~1-2 tiles per row plus roughly half the final row's chunks.
Two located starting points: `persistent.cuh:354` sizes the per-thread offset array with a
`/ 2` (`thr_local_kv_offset[NUM_MMA_KV * KV_THR_LAYOUT_COL / 2 / NUM_WARPS_Q]`) -- check that
the gather loop's index space matches that sizing, since a mismatch reads stale offsets for
the tail iterations, which would hit the LAST tiles of a row's walk (matching the tail-row
chunk-scale miss); and `persistent.cuh:53-71` is the page-table consumption
(`block_size.divmod(packed_block_iter, page_iter, entry_idx)`; `indices[page_iter] *
kv_stride_page`) that any halved `packed_block_iter` base flows through.
Search the V (and K) gather address computation for the persistent kernel's paged path; the
plan values are verified correct, so the fault is in the kernel-side expression consuming
them. Verification gate: `KV_ROWCONST=2` decodes every row to its own page-range mean, then
the mixed-batch numerics pass at TOL=0.02, then the regression pair `qo:128,512`/`qo:127,512`.

**Unified statement after 25 candidates: the kernel attends ~r wrong KV TILES for row r
(sourced near the batch start) plus one chunk-scale miss on the final row.** Not a race, not
precision, not the merge, not kv_head -- wrong 128-token tiles in the KV walk, count growing
with row index. The page-ramp probe (`KV_ROWCONST=2`: V[page p] = p x 1e-4, output decodes to
mean attended page) runs next and names the exact wrong pages per row; from there the fix is
whichever index computes those tile addresses.

**What survives is the original characterisation, now better supported:** the error grows with
the number of rows in the batch (0.0495 at N=2, ~0.30-0.39 at N=18), is independent of chunk
size, appears in FA2-only with no FA3 executor present, and leaves row 0 clean. Something
per-batch-size is wrong -- a shared workspace sized or indexed from the row count, or the
scheduler's distribution of identical rows across clusters -- and it is not the cursor, not
the chunk, not the reduction path, not the plan arrays, and not a memory-safety or
shared-memory race.

**The decisive test is cheap:** make `partial_o` `float` (upstream's own choice) and re-run
this fixture at N=2 and N=18. If the errors collapse to ~1e-3, it is precision and the fix is
a buffer dtype, not a race. If they persist, precision is excluded and the structural hunt
resumes with the cursor eliminated.

Note the earlier `PI_ZERO` arm cannot have tested this: it zeroed `partial_indptr` only for
**non-split** items, and here every item splits.

**Superseded reproducer** (one token flips it, `n_fa3:0`, page 128 or 1, `UNIFIED=1`):

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

**Port commits 29-32 (2026-10-08): the verify half now delegates to upstream, and the real
remaining gap is a SCHEDULER-side one -- the running half never returns to `running_batch`.**
Branch `upstream_port`, worktree `/opt/tiger/sglang-up`, driver `/dev/shm/portchk.sh` on the
vibecuda H200 pod (upstream pinned `b7b2975b5`, venv `venv_up`, torch 2.14.1).

- **Commit 29** stopped the field-chasing. Hand-rolling sample+commit+result meant one failed run
  per field upstream sets (`accept_lens`, `new_seq_lens`, `bonus_tokens`, `next_draft_input`, ...).
  `run_eagle_verify` now takes `precomputed_logits_output`, so the merged forward's verify slice is
  injected and upstream runs the whole post-forward half. Same commit cut an absorbed upstream PP
  tail-draft block that had made the method end in `return batch_output` -- any success would have
  returned the WRONG object. Second damage from slice-based editing this session (after the
  triplicated method); both times the check that caught it was a symbol/structure count, not a run.
- **Commit 30** fixed `UnboundLocalError: forward_batch_output` -- commit 29 gated the
  prepare+forward but only the two uses INSIDE that branch; the result block 120 lines below still
  read `.routed_experts_output` / `.indexer_topk_output`. Threaded the merged forward output through
  rather than defaulting those to None: for an MoE target they carry the step's routing/indexer
  outputs, and nulling them would silently drop EPLB stats on exactly the 4-mode steps.
- **Baseline control run, and it is now on record:** the identical workload with
  `SGLANG_MIXED_SPEC=0` PASSES (`ok [1777, 400, 400]`, 0 tracebacks). So every failure below is in
  the 4-mode path, not an upstream limitation. (Caveat found the hard way: the sed that built the
  baseline script did not rewrite the `O=` line, so the control overwrote the 4-mode `server.log`.
  Separate the output dirs before trusting a comparison of two runs.)
- **The failure, located by logged slot names rather than inference** (`SGLANG_DBG_DRAFT_COPY=1`,
  diagnostic in `eagle_draft_cuda_graph_runner.execute` + at the scheduler gate):

  | step | gate `reqs` | `spec_info.topk_p` | draft copy |
  |---|---|---|---|
  | 1st mixed | 1 | (1,1) | `raw_bs=1` OK |
  | 2nd mixed | 3 | **(4,1)** | `raw_bs=3`, mismatch `topk_p dst(3,1) src(4,1)`, `topk_index` same |

  The batch arrives at the gate ALREADY inconsistent -- 3 reqs with a 4-row draft input, BEFORE the
  gate's own `filter_batch`, which is a no-op here because all 3 reqs are kept
  (`len(keep_indices) == len(self.reqs)` early-returns, so `spec_info.filter_batch` never runs).
  `prepare_for_decode` -> `spec_prepare_for_decode` and `prepare_for_draft` are both correct:
  `prepare_for_draft` rebuilds `out_cache_loc` as `bs*topk*num_steps`, so the earlier
  `out_cache_loc`-width theory (commits 19-22) is NOT what is failing now.
- **Root cause, structural, in my own gate.** The gate hands the decode reqs off via
  `mixed_spec_running` / `decoding_reqs` and replaces the scheduler's `running_batch` with an empty
  one. On the next iteration `running_batch` is empty and `last_batch` is my prefill batch, so
  `running_batch = last_batch` (scheduler.py:3716-3722) -- a batch whose `reqs` are the PREFILL reqs
  and whose `spec_info` is the PREFILL half's `next_draft_input` (set at scheduler.py:4519). The
  decode half's reqs and its verify result never return, so the running request is also LEAKED.
  This is the pending "wire the scheduler to process `result.mixed_running_result`" item: it is not
  a missing call but a missing second commit path.
- **Next step (concrete):** after `run_batch`, when `batch_result.mixed_running_batch` is set,
  commit BOTH halves -- process the verify result for the running batch (accepted tokens, seq_lens,
  finished reqs), set `running.spec_info = mixed_running_result.next_draft_input`, and merge the
  running batch back into `self.running_batch` instead of letting `last_batch` become it. The hard
  part is the overlap loop's per-batch `future_indices` / relay machinery (scheduler.py:4500-4520),
  which assumes ONE batch per forward; the merged forward produces one set of futures for two
  logical batches.

- **[x] 2026-10-08: the 4-mode step RUNS END TO END on current upstream** (port commits 29-38,
  branch `upstream_port`, upstream pinned `b7b2975b5`, torch 2.14.1, venv `venv_up`).
  `/dev/shm/portchk.sh` on the vibecuda H200 pod: **11 mixed steps, `ok [1966, 400, 1892]`, 0
  tracebacks/exceptions**, against `SGLANG_MIXED_SPEC=0` on the same script as the control.
  Commits 35-38 were four more upstream contracts, each named by one measured failure:

  | # | failure | contract the fork never had to satisfy |
  |---|---|---|
  | 35 | `build_chain_tree`: `bonus_tokens.numel() == seq_lens.numel() == bs` | in overlap mode a draft input's tensors are FUTURES; `resolve_seq_lens_cpu`/`resolve_forward_inputs` only ever take the batch being run, so the verify half reached `draft()` unresolved. |
  | 36 | `shape '[4,4]' invalid for input of size 4` | the merged batch padded into a prefill-graph bucket (529 real tokens -> 576). `static_forward_batch` is built field-by-field, so `mixed_verify_tokens` took its default 0 and verify got ONE logit per row. Found a second bug in the same path: my verify-row slice indexed from `hidden_states.shape[0]`, i.e. the PADDED end, which would have selected padding rows and fed verify garbage **without failing**. |
  | 37 | `_resolve_spec_v2_tokens`: `assert result.next_token_ids.is_cpu` | the verify half needs the same D2H `copy_to_cpu` and relay stash as the batch that ran. |
  | 38 | `IndexError: accept_lens[i]` | the attached running batch IS `self.running_batch`, and in overlap mode `get_next_batch_to_run` merges the next prefill into it BEFORE `pop_and_process` runs -- 6 reqs against 4 accept_lens. Upstream enqueues `batch.copy()` for exactly this reason; hand result processing a snapshot. |

  Method note that generalises: every one of 33-38 was found by LOGGING THE WIDTHS, not by
  reasoning about them. Two diagnostics did all the work -- slot names + both shapes at the failing
  copy, and one line per mixed step with every row/token count. The width log also *cleared* the
  split and both draft extends (`ntl=(8,V)` = 4 chunk + 4 verify, draft inputs (4,1) and (1,1)),
  which three earlier commits had wrongly suspected. A control run with the feature OFF, on the
  same script, is what made "this is my bug" a measurement rather than an assumption.

  **NOT yet validated:** correctness. The smoke script's prompts are drawn by `rng.randrange`
  INSIDE worker threads, so prompt-to-index assignment is not deterministic and its output lengths
  are NOT comparable between arms -- they say nothing about whether the 4-mode step produces the
  same tokens. Greedy-output equality vs the `SGLANG_MIXED_SPEC=0` arm is the next gate, before any
  timing number is worth taking.

- **[ ] 2026-10-08: correctness of the ported 4-mode step is NOT established, and the first gate
  design was confounded.** Driver `/dev/shm/portcorr.sh <0|1>` (deterministic prompt set built in
  the main thread; 4 background rows of 512 tok/256 out + 6 long rows of 6000 tok/48 out;
  temperature 0, `ignore_eos`; 16 mixed steps in the ON arm, 0 in the OFF arm, 0 tracebacks both).

  | comparison | requests whose greedy output differs (of 10) |
  |---|---|
  | control vs control (same config, noise floor) | `long4` |
  | 4-mode vs itself, before the keep-alive fix | `long0, long1, long2` |
  | 4-mode vs itself, AFTER the keep-alive fix | `bg2, bg3, long0, long1, long2, long3, long4, long5` |
  | control vs 4-mode | 6-7 requests, set varies per run |

  Three things follow, and the third invalidates the gate:
  1. The control is self-consistent on 9/10, so the harness itself is stable.
  2. Commit 40 (parking the verify half's `extra_keep_alive_refs`) is right by upstream's own
     contract -- the scheduler parks only the RETURNED result's refs, and we return the prefill
     half, so `run_eagle_verify`'s refs were being dropped. But it did NOT reduce the
     nondeterminism (3 -> 8 self-inconsistent requests). **No claim that it fixed anything.**
  3. The 4-mode arm disagrees WITH ITSELF across runs. Which requests are in the running batch when
     a prefill arrives is timing-dependent, so the 4-mode arm's batch composition -- and hence the
     bf16 reduction order -- changes run to run. Greedy-token equality under a CONCURRENT workload
     therefore cannot separate "wrong" from "batched differently". The earlier conclusion drawn
     from it ("5 stable outputs changed, so there is a real bug") is RETRACTED as unproven; it may
     be true, but this experiment cannot show it.

  **The gate this needs instead** (in order): (a) a serialized workload where the mixed step's
  composition is forced identical across runs, so greedy equality is meaningful; and/or (b) the
  fp32 per-row recompute already used for the split kernel (`SGLANG_SPLIT_VERIFY`), which compares
  the merged forward's verify-row logits against a reference and is invariant to batching. (b) is
  the real gate; (a) is the cheap screen.

  Two genuine bugs WERE found and fixed while chasing this, both of the same shape -- an
  open-ended slice of a tensor that a graph bucket pads, which corrupts silently instead of
  failing: the verify-row logit slice (commit 36) and the verify hidden-state slice (commit 39).

- **[x] 2026-10-08: the ported 4-mode step is numerically EQUIVALENT to the control under a
  controlled batch composition.** Driver `/dev/shm/portcorr2.sh <arm> <tag>` (upstream `b7b2975b5`,
  port commit 40, `venv_up`, H200 vibecuda pod). Serialized workload: one 512-token background row
  asking 6000 tokens (`ignore_eos`) reaches steady decode, THEN one 6000-token prompt is
  chunk-prefilled against it, so every mixed step is exactly (1 chunk row + 1 verify row) and the
  composition no longer depends on arrival timing. 3 mixed steps in the ON arm, 0 in the OFF arm,
  0 tracebacks in all four runs. Two runs per arm:

  | comparison | result |
  |---|---|
  | 4-mode vs itself (1a vs 1b) | **IDENTICAL** |
  | control vs itself (0a vs 0b) | **IDENTICAL** |
  | control vs 4-mode (0a vs 1a) | **IDENTICAL** |

  This settles the previous bullet: with composition held fixed, both arms are self-reproducible
  AND agree byte for byte, so the earlier 6-of-10 divergence was timing-dependent batch
  composition (bf16 reduction order), not a defect. The retraction there was correct.

  **Scope of what this proves, honestly:** 3 mixed steps, each 1 chunk row + 1 verify row, no
  prefill-graph padding on those steps (`hs` summed exactly), single sequence per side. It does NOT
  yet cover multi-verify-row steps (the concurrent run reached 4), the padded-bucket path, or
  acceptance > 1. Next: the same serialized screen with N background rows for `ver_rows > 1`, and a
  case sized to land in a padded bucket -- those are where commits 36/39 bit.
  Gate (b), the fp32 per-row recompute, is still the stronger check and still unbuilt on upstream.

  First caught by this screen: a gate with `mixed_steps: 0`. The background row asked 400 tokens
  and finished before the long prompt arrived -- the SAME mistake as the H20 gate earlier in this
  file. A correctness run must assert it exercised the path before its result means anything.

- **[x] 2026-10-08: commit 41 (prefill rows sampled with the prefill batch's own `sampling_info`)
  validated through the screen above.** `portcorr2.sh 1 c` vs the stored control: **IDENTICAL**, and
  vs port 40: **IDENTICAL** (3 mixed steps, 0 tracebacks). Neutral on this workload because every
  request runs temperature 0, so the merged-width info happened to agree on the rows it touched;
  the fix matters once requests in one step carry DIFFERENT sampling params, and it restores what
  `mix_with_verify` snapshots `mixed_pre_sampling_info` for. Found by grepping for that field and
  seeing it was set but never read -- it had been dropped in the commit-29 restructure, the same
  class of loss as the triplicated method and the absorbed PP block.

- **[ ] 2026-10-08: with `ver_rows > 1` the 4-mode step is NONDETERMINISTIC run to run. Real
  defect, and this screen is not confounded.** Driver `/dev/shm/portcorr3.sh <arm> <tag>`
  (`NBG=3` background rows, fixed 1.5 s stagger, each 512 tok asking 6000 out; then one 6000-token
  prompt chunk-prefilled against them). ON arm: 5 mixed steps spanning `ver_rows` 1, 2 AND 3;
  OFF arm: 0 mixed steps; 0 tracebacks in all four runs. Two runs per arm:

  | comparison | requests differing (of bg0,bg1,bg2,long) |
  |---|---|
  | control vs control | **IDENTICAL** -- zero noise floor in this screen |
  | **4-mode vs itself** | `bg0, bg1` |
  | control vs 4-mode | `bg0, bg1` (the same two) |
  | of the requests stable in BOTH arms (`bg2`, `long`) | 4-mode differs on **NONE** |

  Why this is different from the confounded gate above: there, BOTH arms varied run to run, so a
  diff could not be attributed. Here the control reproduces itself exactly under the same arrival
  schedule, and only the 4-mode arm varies -- and only on the rows that pass through the verify
  half (bg0/bg1 are decoding during the mixed steps; bg2 starts after them). So the verify half
  itself is nondeterministic. Note this does NOT contradict the `portcorr2` result, which had
  exactly ONE verify row and was self-identical; the defect appears with multiple verify rows.
  Commit 40's stranded keep-alive refs were real but evidently not the whole story.

  Next diagnostic (chosen to localise, not to guess): per mixed step, log a checksum of the
  verify-row logit slice and of `ver_hidden`, run twice and diff by step index. If the checksums
  already differ, the nondeterminism is in the merged forward or the batch handed to it; if they
  match while outputs diverge, it is downstream in accept/commit/draft-extend. Prime suspects for a
  multi-row-only fault: per-row ordering in `mix_with_verify`'s widening, and anything indexed as
  though there were a single verify row.

- **CORRECTION to the bullet above (same day): the `ver_rows > 1` nondeterminism is NOT a defect,
  and calling it "unconfounded" was wrong.** The checksum diagnostic settles it in one pair of runs
  (`portcorr3.sh 1 p` / `1 q`, 5 mixed steps each, 0 tracebacks):

  | per mixed step | run p | run q | |
  |---|---|---|---|
  | `pre_logits_sum` (all 5 steps) | -260182.92, -343168.63, 165197.42, -317355.38, -276766.63 | **identical** | prefill half is deterministic |
  | `ver_logits_sum` | -772890.1, -445390.1, -3024070.0, ... | -344340.4, -2091862.5, -4305244.5, ... | differs |
  | `running.seq_lens` at the step | `[1541]`, `[2563,...]`, `[4957,...]` | `[1545]`, `[2566,...]`, `[4966,...]` | **differs by ~4 = num_draft_tokens** |

  The verify rows are at DIFFERENT decode positions when the merge happens -- one extra accepted
  batch had landed in run q before the long prompt's chunk arrived. So the step's inputs genuinely
  differ between runs; identical `pre_logits_sum` proves the forward itself is reproducible given
  the same inputs. And this explains the control's zero noise floor without any appeal to a bug:
  in the control, decode rows are NEVER merged with a prefill chunk, so when the prefill interleaves
  cannot perturb the decode tokens. In the 4-mode arm it necessarily can.
  My error was treating "control is deterministic" as proof that the arms are comparable; the
  control is deterministic for a reason that does not transfer.

  **Consequence: greedy-token equality cannot validate this feature under ANY concurrent workload.**
  The merge point is timing-dependent by construction, so the one-verify-row agreement in
  `portcorr2` was luck of a short window, not a stronger guarantee. Both of this file's greedy
  comparisons are therefore screens for crashes and gross faults only.

- **[ ] Build gate (b), the batching-invariant correctness gate** -- the only one that can settle
  this. Inside ONE run, per mixed step, compare the merged forward's verify-row logit slice against
  a reference computed for the SAME inputs by the normal (unmerged) verify forward, and report
  per-row relative error; gate on it the way `SGLANG_SPLIT_VERIFY` gates the split kernel
  (that precedent: 0.003-0.004 rel err = bf16 noise, ~4.0 = a real fault). Self-contained per step,
  so timing cannot confound it. Sketch: under `SGLANG_MIXED_VERIFY_CHECK=1`, after
  `eagle_prepare_for_verify` has built the verify batch, run `target_worker.forward_batch_generation`
  on it separately and diff against `ver_out.next_token_logits`. Costs one extra forward per mixed
  step and rewrites the same KV slots with the same values; gate-only, never on by default.
  Until this exists, **no perf number for the 4-mode step on upstream is worth recording** -- and
  none has been taken.

- **[ ] Gate (b) built (port 42-44) and it now validates its own reference, but its VERDICT is
  still not trustworthy -- and it conflicts with `portcorr2`. Do not act on either yet.**
  `portcorr3.sh 1 x`, 5 mixed steps spanning ver_rows 1/2/3, 0 tracebacks:

  | ver_rows | rows | ref_vs_ref | merged_vs_ref | argmax flips | scale (ref/got absmax) |
  |---|---|---|---|---|---|
  | 1 | 4 | **0.000e+00**, 0 flips | 4.756e-01 | 2/4 | 28.1 / 33.2 |
  | 2 | 8 | **0.000e+00**, 0 flips | 1.103e+00 | 6/8 | 37.8 / 33.5 |
  | 3 | 12 | **0.000e+00**, 0 flips | 8.170e-01 | 7/12 | 35.0 / 35.2 |
  | 3 | 12 | **0.000e+00**, 0 flips | 9.014e-01 | 4/12 | 26.0 / 31.5 |
  | 3 | 12 | **0.000e+00**, 0 flips | 8.312e-01 | 5/12 | 27.4 / 26.9 |

  Read naively this says the merged forward computes the verify rows wrong (disagreement ~15-30 in
  absolute logits, argmax flipped on about half the rows -- nowhere near bf16 noise). **But it
  contradicts the `portcorr2` result**, where with exactly ONE verify row the 4-mode arm's greedy
  output was byte-identical to the control over 400 tokens; flipping 2 of 4 verify-row argmaxes
  could not leave the emitted text untouched. Both cannot hold.

  Most likely the reference is still the broken side, for a reason `ref_vs_ref` cannot catch:
  `eagle_prepare_for_verify` PLANS the attention backend for the verify batch, and the merged
  forward then re-plans that same shared backend state. So the reference forward runs with the
  MERGED batch's attention metadata -- reproducibly (hence err exactly 0.0 twice) but invalidly.
  A bit-exact self-comparison proves stability, not validity; that is a fifth way for a gate to
  look authoritative while measuring nothing.

  **Next step, concrete:** re-plan immediately before the reference forward (call
  `eagle_prepare_for_verify` again, or the backend's plan for `_vfb`, right before it) and re-read
  the table. If `merged_vs_ref` then drops to ~1e-3, the merged forward is correct and `portcorr2`
  stands; if it stays ~1e0 with flips, the merged forward's verify rows are genuinely wrong and the
  prime suspects are the EXTEND-vs-TARGET_VERIFY convention mismatches: `seq_lens` for a verify row
  (prefix vs prefix+n), `extend_prefix_lens`, and `positions` (must be prefix+i for the i-th draft
  token, which drives RoPE).
  Until this is resolved, no perf number is meaningful. `perfab.sh` is staged on the pod (realistic
  arm: 4000-session tracelab subset at `/dev/shm/w/tracelab_trace.jsonl.gz` + a chat stream, debug
  and verify envs deliberately OFF) but has NOT been run.

- **Re-plan (port 45) REFUTES the stale-plan hypothesis; the contradiction is unresolved.**
  `portcorr3.sh 1 y`, 5 mixed steps, 0 tracebacks, with `eagle_prepare_for_verify` re-run
  immediately before each reference forward:

  | ver_rows | merged_vs_ref | flips | vs previous run |
  |---|---|---|---|
  | 1 | 4.756e-01 | 2/4 | **bit-identical** (same ids_sum=254, same err, same scales) |
  | 2 | 9.825e-01 | 6/8 | same magnitude |
  | 3 | 7.203e-01 / 6.194e-01 / 8.825e-01 | 5/12, 3/12, 6/12 | same magnitude |

  So re-planning changes nothing: either the plan was never stale or the re-plan is idempotent.
  Two further facts worth keeping: `ref_vs_ref` stays exactly 0 with a fresh plan each time, and
  the ver_rows=1 step reproduces bit-for-bit across runs -- the merged verify path is
  deterministic given identical inputs.

  **The standing contradiction** (do not resolve it by preference): gate (b) says the merged
  forward's verify logits disagree with an unmerged verify forward far beyond bf16, flipping ~half
  the argmaxes; `portcorr2` says that with one verify row the 4-mode arm reproduced the control's
  greedy text bit-for-bit over 400 tokens through 3 mixed steps, which an argmax flip on 2 of 4
  verify rows could not survive. Exactly one of these is measuring something else.

  **Next step: a NULL test, not another hypothesis.** Compare the merged forward's PREFILL-row
  logits against a prefill-only reference, in the same step and with the same machinery. The
  prefill half is independently known-good (identical `pre_logits_sum` across runs; the control's
  own prefill path). So:
  - prefill rows also ~0.5 with flips -> the GATE is wrong (its reference or its row mapping), and
    the merged forward is exonerated, consistent with `portcorr2`;
  - prefill rows ~1e-3 while verify rows stay ~0.5 -> the gate is sound and the merged batch's
    VERIFY rows are genuinely wrong, and `portcorr2`'s agreement was luck (then chase the
    EXTEND-vs-TARGET_VERIFY conventions: `seq_lens` prefix vs prefix+n, `extend_prefix_lens`,
    `positions` for RoPE).
  This is one run and it cannot come out ambiguous, which is why it is the next thing and not
  more reasoning about which side "should" be right.

- **[x] NULL TEST RESOLVES IT: gate (b)'s verify reference is invalid; the merged forward is
  exonerated, and `portcorr2` stands.** `portcorr3.sh 1 z`, 5 mixed steps, 0 tracebacks:

  | probe | result | reading |
  |---|---|---|
  | `NULLTEST prefill_rows` (all 5 steps) | **err=0.0, flips=0**, shapes match | the merged forward's PREFILL rows are bit-exact vs a prefill-only reference -- the merged forward, the logits slicing and the gate's comparison mechanism are all sound |
  | `ref_vs_ref` in the SAME run | **4.84e-01 to 8.89e-01, flips=1** | two supposedly identical verify references disagree by the SAME magnitude as `merged_vs_ref` (6.1e-01 to 9.7e-01) |

  `ref_vs_ref` was exactly 0 in the earlier runs only because both reference forwards reused one
  `_vfb`; once each one re-plans (port 45), they diverge from each other. So
  `eagle_prepare_for_verify` + a verify forward is **destructive and non-idempotent** -- it
  reallocates cache locs and mutates `running` -- and my "reference" was never a read-only probe.
  `merged_vs_ref ~= 0.5-1.0` is that instability, not a verify-row defect. Gate (b) as built
  measures its own side effects.

  Net correctness position, stated with its limits: the merged forward's prefill rows are bit-exact;
  its verify rows have no evidence against them and one positive result for them (`portcorr2`:
  byte-identical greedy text over 400 tokens through 3 mixed steps at ver_rows=1); and no gate yet
  covers verify rows at ver_rows > 1. The "merged verify rows are wrong" reading of the previous two
  bullets is **RETRACTED** -- it was my gate's side effects.

- **[ ] Rebuild gate (b) with a NON-DESTRUCTIVE verify reference.** The reference must not
  reallocate or mutate `running`. Options, cheapest first: (a) run the reference FIRST, before the
  merged forward, and compare the merged slice to it (one `eagle_prepare_for_verify` total, which
  the step already does -- no extra mutation); (b) deep-copy the batch and give the reference its
  own KV slots; (c) snapshot and restore `running.out_cache_loc`/`seq_lens` plus the touched KV
  pages around the probe. (a) is nearly free and is what the step's own structure already affords.
  Validity check that must accompany it: the prefill-row null test (already err=0.0) stays at 0,
  AND `ref_vs_ref` stays at 0 with a fresh plan each time.

- **[x] GATE (b) v2 IS SOUND, AND IT FINDS A REAL DEFECT: the merged batch's VERIFY rows are
  computed wrong while its PREFILL rows are bit-exact.** Port 47 takes the reference BEFORE the
  merged forward, reusing the verify batch the step already prepared (prepare -> forward is the
  normal verify path), so nothing extra is mutated. `portcorr3.sh 1 g2`, 5 mixed steps, 0 tracebacks,
  null case on the SAME log line:

  | ver_rows | NULL prefill | merged_vs_ref | argmax flips | scale ref/got |
  |---|---|---|---|---|
  | 1 | **err=0.0, flips=0** | 4.756e-01 | 2/4 | 28.1 / 33.2 |
  | 2 | **err=0.0, flips=0** | 9.721e-01 | 6/8 | 31.4 / 36.2 |
  | 3 | **err=0.0, flips=0** | 7.180e-01 | 6/12 | 33.2 / 31.8 |
  | 3 | **err=0.0, flips=0** | 5.249e-01 | 4/12 | 26.4 / 26.9 |
  | 3 | **err=0.0, flips=0** | 5.875e-01 | 4/12 | 25.0 / 21.4 |

  The null case is what makes this readable: identical machinery, identical step, prefill rows agree
  to the bit while verify rows disagree by ~15-30 absolute logits. Not bf16, not the gate's side
  effects (v1's failure), not a stale plan (port 45 refuted that).

  **This RECONCILES with `portcorr2` instead of contradicting it.** Only the accepted prefix of a
  verify row affects emitted text; `accept_lens=[1]` throughout these runs. If the flips sit at
  draft positions >= 1, they are discarded whenever acceptance is short, so greedy text can match
  bit-for-bit while the step is still wrong -- and acceptance is silently depressed, which is a
  PERF loss that would quietly cap any speedup the 4-mode step could show. Both measurements stand.

- **[ ] Next, and narrow: find which draft POSITION goes wrong.** Log per-position argmax agreement
  (position 0..n-1 separately) plus the accepted length, and compare mean accept_len against the
  control arm. Expected signatures, each pointing somewhere specific:
  - flips only at positions >= 1, position 0 exact -> the n draft tokens' intra-row causal
    structure or their `positions` (RoPE offset must be prefix+i); the merged EXTEND path gives a
    verify row plain causal attention, which is right for a chain (topk=1) ONLY if positions and
    `extend_prefix_lens` are right;
  - flips at position 0 too -> the row's prefix KV length is wrong (`seq_lens` prefix vs prefix+n
    is the EXTEND-vs-TARGET_VERIFY convention mismatch to check first).
  Also worth one line: mean accept_len, 4-mode vs control, since that quantifies the cost and is
  the number a reviewer will ask for.

- **[x] Defect localised: the error is UNIFORM across draft positions, including position 0, which
  implicates the verify rows' prefix KV length -- not RoPE and not intra-row causality.**
  `portcorr3.sh 1 g3`, 5 mixed steps, 0 tracebacks (`BYPOS`, n=4):

  | step (ver_rows) | flips by position [0,1,2,3] | err by position [0,1,2,3] |
  |---|---|---|
  | 1 | [1, 1, 0, 0] | [0.578, 0.481, 0.401, 0.334] |
  | 2 | [2, 2, 1, 1] | [1.156, 1.069, 0.723, 0.911] |
  | 3 | [1, 1, 2, 2] | [0.604, 0.746, 0.765, 0.711] |
  | 3 | [1, 1, 2, 0] | [0.603, 0.508, 0.525, 0.522] |
  | 3 | [0, 1, 2, 1] | [0.518, 0.360, 0.588, 0.519] |

  Position 0 is wrong as often as the others and the error magnitude is flat across positions. By
  the signature recorded before this run, "position 0 wrong too" rules out the RoPE-offset and
  intra-row-causal-structure hypotheses (those spare position 0) and points at the row's prefix:
  every draft token in the row attends to a wrong prefix, so all n are wrong by a similar amount.

  **The one thing to check next, with the numbers to compare:** the verify rows' `seq_lens` in the
  MERGED batch versus the reference verify batch. `mix_with_verify` widens `seq_lens`/`seq_lens_cpu`,
  and EXTEND and TARGET_VERIFY disagree on what that field means -- EXTEND wants prefix + extend_len
  (= prefix + n) with `extend_prefix_lens` = prefix, while the verify path keeps `seq_lens` at the
  committed prefix and masks the tree itself. Observed running-batch `seq_lens` at the steps above:
  `[1545]`, `[2534, 1191]`, `[4916, 2622, 2894]`. Log, for the verify rows only, `seq_lens`,
  `extend_prefix_lens`, `extend_seq_lens` and `positions[0]` from BOTH the merged batch and `_vfb`,
  and make the merged batch satisfy the EXTEND contract. If that is it, this single convention has
  been behind every "verify rows look off" symptom in this file.

- **[x] The mismatched field is NAMED: the verify rows' `positions` (and the implied prefix end)
  differ from the reference by exactly n.** `portcorr3.sh 1 g4`, 5 mixed steps, 0 tracebacks:

  | step | merged | reference |
  |---|---|---|
  | 1 | `seq_lens=[1545] ext_prefix=[1541] ext_seq=[4]` **pos0=1541** | `seq_lens=[1545]` **pos0=1545** |
  | 2 | `seq_lens=[2530,1191] ext_prefix=[2526,1187]` pos0=2526 | `seq_lens=[2530,1188]` pos0=2530 |
  | 3 | `seq_lens=[4915,2665,2894] ext_prefix=[4911,2661,2890]` pos0=4911 | `seq_lens=[4915,2662,2894]` pos0=4915 |
  | 4 | `seq_lens=[4919,2666,2898] ext_prefix=[4915,2662,2894]` pos0=4915 | `seq_lens=[4919,2663,2898]` pos0=4919 |
  | 5 | `seq_lens=[4923,2667,2902] ext_prefix=[4919,2663,2898]` pos0=4919 | `seq_lens=[4922,2665,2902]` pos0=4922 |

  `seq_lens` AGREES (1545 = 1541 + 4), so the EXTEND contract on that field is met and the earlier
  "seq_lens prefix vs prefix+n" hypothesis is **not** the fault. What differs is `positions`: the
  merged batch puts the row's first draft token at `seq_lens - n`, the reference puts it at
  `seq_lens`. Off by exactly n = `draft_token_num`, consistently, on every step. A prefix/RoPE
  offset wrong by a constant for every token of the row is exactly the uniform,
  position-independent error `BYPOS` measured -- including position 0.

  Secondary observation, NOT yet explained and worth keeping: rows 2-3 also drift by ~3 between the
  two views (`merged [...,2665,...]` vs `ref [...,2662,...]`, and `[2530,1191]` vs `[2530,1188]`).
  The reference's `_vfb` is built before the merge, so a later advance would make the merged view
  newer; 3 is not n, so it looks like acceptance bookkeeping rather than the same off-by-n. Resolve
  the off-by-n first, then re-read this -- it may disappear with it.

  **Next action (do NOT fix blind):** establish which side is right before changing anything, since
  both readings are defensible from these numbers alone -- either the merged batch's positions are n
  too small, or `eagle_prepare_for_verify` had already advanced `running.seq_lens` by n before
  `_vfb` captured positions, making the REFERENCE the shifted one. The decisive check is a third
  party: upstream's own non-mixed spec decode. Log `positions[0]` and `seq_lens` for a verify row on
  the OFF arm (ordinary spec decode, no merge) and compare against both. Whichever side matches the
  OFF arm defines the contract. Only then change `mix_with_verify`'s `positions` for the tail rows.

- **[x] ROOT CAUSE FOUND for the verify-row logit error, confirmed against the production contract.**
  Two measurements settle it:
  - `OFFCONV` (logged inside `run_eagle_verify`, so it is the path ordinary spec decode AND the OFF
    arm take): `n=4 seq_lens=[12] pos_head=[12,13,14,15]`, and likewise `[6]->[6..9]`,
    `[512]->[512..515]`. So at verify time **`seq_lens` IS the committed prefix** and the n draft
    tokens occupy `prefix .. prefix+n-1`.
  - `SLTRACE`: `before_prep = after_prep = vfb_seq = merged_tail_seq = 1497` with `vfb_pos0=1497`.
    So `running_batch.seq_lens` is that committed prefix, `eagle_prepare_for_verify` does not move
    it, and the merge leaves the tail rows' `seq_lens` EQUAL to the prefix.

  Since `extend_prefix_lens` is derived as `seq_lens - extend_lens`, a tail row with
  `seq_lens = P` and `extend_lens = n` presents its prefix as **P - n**: every draft token in the
  row attends to a prefix n tokens short and gets positions `P-n .. P-1`. That is exactly the
  signature gate (b) measured -- uniform error across all n positions including position 0, while
  the prefill rows in the SAME batch are bit-exact. `mix_with_verify`'s `seq_lens[tail] = seq_lens
  - 1 + n` infers the base from whatever `mix_with_running` left instead of from the committed
  prefix. (The earlier "seq_lens prefix vs prefix+n" guess was right about the field and wrong
  about which value was off; only the three-point trace pinned it.)

- **[ ] The one-line form of the fix is NOT sufficient: it trades the wrong answer for a crash.**
  Port 52 set `seq_lens[tail] = committed_prefix + n` (so `extend_prefix_lens` = P and positions
  run P..P+n-1, matching production). Result: `CUDA_ERROR_ILLEGAL_ADDRESS` /
  `torch.AcceleratorError: an illegal memory access`, 0 mixed steps. **Reverted (port 53)** so the
  tree and pod stay runnable; the fix is kept in history at port 52 for the next attempt.
  Reading: raising the row to P+n makes the attention read `req_to_token[req, :P+n]`, so the KV
  index coverage for `[P, P+n)` must be established for the MERGED batch, not just inside the
  verify batch that `eagle_prepare_for_verify` builds. Next: check what
  `assign_extend_cache_locs_uniform_func` writes into `req_to_token` for the verify rows and when,
  and whether the merged batch's attention plan (`max_seq_len` / page table built from
  `seq_lens`) is computed before or after that write. Fix the index coverage and the length
  together, then re-read gate (b) -- the null case plus `BYPOS` will say immediately whether the
  uniform error is gone.

  **Refinement after reading the code (changes the next step):** `eagle_prepare_for_verify` calls
  `assign_extend_cache_locs_uniform_func(start_offset=batch.seq_lens, draft_token_num=n)`, which
  writes `req_to_token[req, P : P+n]` for the n verify slots -- and it runs BEFORE the merge in
  `forward_mixed_spec`. So the KV index coverage for `[P, P+n)` already exists, and the "coverage
  is missing" reading above is wrong. The illegal access therefore points at the merged batch's
  attention metadata rather than at the index table. Concretely, next:
  (1) re-run port 52 with `CUDA_LAUNCH_BLOCKING=1` to name the faulting kernel instead of inferring
      it from an async report;
  (2) log, for the merged batch, `seq_lens_sum`, `max_seq_len_k/q` and `extend_start_loc` alongside
      the per-row `seq_lens`/`extend_prefix_lens`, since a length raised after some cached maximum
      was derived is the likeliest out-of-bounds;
  (3) check whether raising a tail row's `seq_lens` also requires `orig_seq_lens` (and the
      page-table width used by the chunk rows' plan) to move with it -- the merge currently widens
      five fields and `orig_seq_lens` is not one of them.

- **The faulting kernel is named: FA3's hopper forward.** Re-ran port 52 (+ `orig_seq_lens` widened,
  port 54) with `CUDA_LAUNCH_BLOCKING=1`:
  `CUDA error (_deps/repo-flash-attention-src/hopper/flash_fwd_launch_template.h:209): an illegal
  memory access was encountered`. `MIXEDGATE` fired twice while `MIXEDPRE` never logged, so the
  crash is INSIDE the merged forward, before the logits split -- i.e. raising a tail row's
  `seq_lens` to P+n makes **FA3** read out of bounds for that row. Widening `orig_seq_lens` did not
  help, so that was not it.
  That run reported `mixed_steps: 0` and `tracebacks: 0` purely because the crash preceded the
  counter; note also that `CUDA_LAUNCH_BLOCKING=1` serialises every kernel and badly perturbs the
  arrival timing this screen depends on, so it is a debugging tool here, not a measurement
  configuration.
  **Reverted ports 52 and 54 (ports 53, 55, 56); the tree and the pod are back to the known-runnable
  state** -- which still has the wrong prefix, i.e. the defect is understood but NOT yet fixed.

  Next, with the kernel now known: FA3 takes `cu_seqlens_k` / `seqused_k` and a page table built
  from `req_to_token`. The index coverage for `[P, P+n)` exists (prepare_for_verify writes it), so
  the suspect is the merged batch's FA3 metadata -- specifically whether the page table / kv-indices
  buffer for the MERGED batch is sized from a seq-len total captured before the tail rows were
  widened, and whether the chunk rows and verify rows can share one `cu_seqlens_k` when the verify
  rows' prefix is not their `seq_lens`. This is squarely in the FA3 path the goal's split kernel
  also has to satisfy, so the work is not wasted on a detour.

- **[x] The illegal access is fixed, and the prefix was ONE real cause -- but the verify rows are
  still wrong.** Port 57 lands `seq_lens[tail] = committed_prefix + n` TOGETHER with
  `self.seq_lens_sum = None`. The second half is what stopped the crash: `mix_with_running`
  reassigns `self.seq_lens` but never resets `seq_lens_sum`, and `ForwardBatch.init_new` recomputes
  it ONLY when it is None -- so the merged batch carried the PREFILL batch's sum, which does not
  include the decode rows at all, and FA3 metadata sized from that total read out of bounds.
  (`mix_with_running`'s own decode path sets it to None for exactly this reason; the 4-mode merge
  had to do the same.) Result: 5 mixed steps, 0 tracebacks, no illegal access.

  | gate (b) v2 | before port 57 | after port 57 |
  |---|---|---|
  | argmax flips per step | 2/4, 6/8, 7/12, 4/12, 5/12 | **1/4, 2/8, 2/12, 2/12, 1/12** |
  | `merged_vs_ref` err | 4.8e-01 .. 1.10 | **2.5e-01 .. 6.4e-01** |
  | NULL prefill case | err=0.0, flips=0 | err=0.0, flips=0 (gate still valid) |
  | `flips_by_pos` | flat across all 4 positions | sparser, no longer concentrated |

  So the prefix really was a cause -- flips roughly halved and the error fell -- but a residual
  ~0.25-0.64 remains, which is still ~100x bf16 noise. **Port 57 is KEPT** (strictly closer to the
  measured contract, and it removes a genuine out-of-bounds), while correctness remains unestablished.

  Candidates for the residual, in the order they should be checked: (a) the attention MASK -- the
  production path is TARGET_VERIFY with a tree/custom mask, the merged path gives a verify row plain
  causal attention, which is equivalent for a chain (topk=1) only if the mask does not also gate the
  bonus token; (b) whether the n draft tokens' KV is visible to their own row in the merged forward
  in the same order the verify path assumes; (c) `extend_start_loc`/`cu_seqlens` when a 2047-token
  chunk row shares the batch with n-token verify rows. Check (a) first -- it is the one structural
  difference between the two paths that the fields alone cannot express.

  Candidate (a) narrowed by reading the code: the merged batch's `forward_mode` is `MIXED` and its
  `spec_info` is inherited from the PREFILL batch (only `running.spec_info` is set to
  `verify_input`), so no `custom_mask` is applied to the merged forward at all -- the verify rows
  get plain causal extend attention. For a chain (topk=1) that is the same mask the tree encodes,
  so "the wrong mask is applied" and "a verify-sized mask is applied to a bigger batch" are both
  ruled out. That leaves (b) KV visibility/order for the draft tokens within their own row and
  (c) `extend_start_loc`/`cu_seqlens` when a 2047-token chunk row shares the batch with n-token
  verify rows -- plus a new one worth testing first because it is cheap: run gate (b) with a
  single-request mixed step (`NBG=1`) and compare against the SAME request's verify logits in the
  OFF arm, which removes the chunk row from the batch entirely and so isolates whether the residual
  needs a chunk row present at all.

- **[x] After port 57 the merged batch's FIELDS are correct; the remaining verify-row disagreement
  is FA3-specific.** Same build, same screen, only `--attention-backend` changed:

  | backend | argmax flips per step | `merged_vs_ref` err | NULL prefill |
  |---|---|---|---|
  | fa3 (default) | 1/4, 2/8, 2/12, 2/12, 1/12 | 2.5e-01 .. 6.4e-01 | err=0.0 |
  | **triton** | **1/4, 0/8, 0/12, 0/12, 0/12** | 2.1e-01 .. 3.1e-01 | err=0.0 |

  Under triton the merged verify rows reproduce the production verify path's argmax on every row of
  four of the five steps (and 3/4 of the first), so the batch the 4-mode step builds is right and
  the earlier uniform error really was the prefix (port 57). The residual ~0.2-0.3 with ZERO flips
  is the scale of a bf16 reduction-order difference between two different kernels (MIXED extend vs
  TARGET_VERIFY) over a ~2500-token prefix, not a field error.
  **FA3 adds flips that triton does not**, which fits the paged-length hypothesis: if FA3's extend
  path takes its paged-KV length from `seq_lens` (= P+n after port 57) instead of
  `extend_prefix_lens` (= P), the n draft tokens are counted twice -- once in the paged prefix and
  again in the causal tail. Next: read FA3's metadata construction for an extend batch and confirm
  which field feeds `seqused_k`; this is the same FA3 path the goal's split kernel must satisfy.

  **Correction to the paged-length hypothesis, from reading FA3's metadata build:** the extend path
  sets `metadata.cache_seqlens_int32 = forward_batch.seq_lens` directly
  (`flashattention_backend.py`, extend branch). That is the standard varlen-with-KV-cache
  formulation -- one causal attention over `[0, seq_lens)` whose queries are the last `extend_len`
  positions -- so `seq_lens = P + n` after port 57 is CORRECT for FA3 and there is no double count.
  The FA3-vs-triton gap is therefore numerical (FA3's split-KV / num_splits choices), not a second
  field bug: the residual is simply larger under FA3 (0.25-0.64 vs 0.21-0.31), which is enough to
  flip 1-2 near-tied argmaxes per step. Both backends agree the fields are right.
  Settling whether 0.2-0.6 absolute (~1-2% of a ~30 logit scale) is acceptable bf16 variation needs
  an **fp32 reference**, which is the standard this project already uses for the split kernel
  (`SGLANG_SPLIT_VERIFY`, where 0.003-0.004 was bf16 noise and ~4.0 a real fault). Porting that fp32
  recompute to upstream is the remaining correctness task -- NOT another bf16-vs-bf16 comparison,
  which cannot distinguish "wrong" from "different kernel".

- **First perf A/B attempt INVALID, discarded (2026-10-08).** `perfab.sh` drove the long-context arm
  with `--dataset-name tracelab-claude`, which is a **fork-only** dataset: upstream's
  `bench_serving` rejects it (`invalid choice: 'tracelab-claude'`). So the agent stream never ran,
  only the 512-token chat stream did, and the mixed gate fired **0 times** -- the arm measured
  ordinary spec decode, not the 4-mode step. The chat numbers it produced
  (TTFT 38.6 ms, ITL 4.57 ms, 751 tok/s on the ON arm) are therefore NOT a 4-mode result and are
  recorded here only so they are not mistaken for one later.
  Lesson, and the third time this exact thing has bitten in this file: a perf arm must report that
  it EXERCISED the path (mixed-gate count) beside its timings, exactly as the correctness screens
  must. The driver now prints the gate count per arm.
  Re-running with upstream's own `generated-shared-prefix` (6 groups x 8 prompts, 8192-token shared
  system prompt) as the long-context arm plus the chat stream, both arms on triton (the backend
  where the verify rows reproduce the production argmaxes).

- **[x] FIRST VALID BASELINE NUMBERS on upstream (arm 0), and the 4-mode arm CRASHES on this
  workload (2026-10-08).** `perfab.sh <arm> g`, upstream `b7b2975b5` + port 57, `venv_up`,
  one H200, `--attention-backend triton`, EAGLE3 MTP (3 steps, topk 1, 4 draft tokens),
  `--enable-mixed-chunk --chunked-prefill-size 2048`. Workload: `generated-shared-prefix`
  (6 groups x 8 prompts, 8192-token shared system prompt, 64 out) + `random-ids` chat
  (512 in / 128 out, 400 prompts, 8/s), released together.

  | arm | stream | completed | mean TTFT | p99 TTFT | mean ITL | p99 ITL | out tok/s |
  |---|---|---|---|---|---|---|---|
  | **0 (baseline: upstream mixed-chunk + spec)** | agent (gsp) | 48 | 116.4 ms | 475.9 ms | 9.84 ms | 48.3 ms | 37.1 |
  | **0 (baseline)** | chat | 400 | 53.5 ms | 286.4 ms | 7.18 ms | 36.4 ms | 741.4 |
  | 1 (`SGLANG_MIXED_SPEC=1`) | — | — | **server crashed** | | | | |

  Baseline: 0 tracebacks, 0 fallbacks. **This is the reference the 4-mode arm has to beat by 30%
  TTFT/ITL and 10% throughput; there is still no ON-arm number.**

  The ON arm died in `cuda_graph_buffer_registry.fill_from` -> `_grouped_foreach_copy_` ->
  triton `_small_copy_kernel`, reached from the ORDINARY `verify()` path
  (`eagle_worker_v2.py:1397`), during warmup. `FILL_DEBUG mismatch: []` means **no shape
  mismatch**, so it is a kernel-launch/CUDA-level failure, not a field-width bug -- a different
  failure from everything above, and it does not reproduce in `portcorr3.sh` (5 mixed steps,
  0 tracebacks on the same build). The difference is the workload: 8192-token shared prefixes and
  48 concurrent gsp rows vs 3 background rows.

  **Caveat that invalidates my own guard:** the driver printed "mixed-gate fires: 0" for both arms,
  but `MIXEDGATE` only logs under `SGLANG_DBG_DRAFT_COPY=1`, which `perfab.sh` deliberately does not
  set (it would skew timings). So that count is meaningless here and the ON arm may well have taken
  the mixed path before crashing. The gate-exercised check needs a counter that is ALWAYS on and
  cheap (a scheduler-side integer reported at the end), not a debug log line. Fix that before the
  next A/B, or the same "it passed but ran nothing" trap recurs a fourth time.

- **[ ] BLOCKER for any ON-arm number: the 4-mode step is not stable under the perf workload.**
  Re-ran arm 1 with the new always-on counter (port 58, `MIXEDSPEC_STEPS` at INFO):
  **`mixed_spec_steps: 1`, `tracebacks: 1`, agent json absent.** The counter proves the mixed path
  DID fire (so this is not another "exercised nothing" false pass -- the new guard worked on its
  first use), and the server then died in
  `cuda_graph_buffer_registry.fill_from -> _grouped_foreach_copy_ -> triton _small_copy_kernel`,
  reached from the ordinary `verify()` (`eagle_worker_v2.py:1397`), with `FILL_DEBUG mismatch: []`
  (no shape mismatch).
  It does NOT reproduce under `portcorr3.sh` (5 mixed steps, 0 tracebacks, same build). The
  workload is the difference: 8192-token shared prefixes and 48 concurrent gsp rows vs 3 background
  rows. So the next step is to shrink the gap -- raise `portcorr3`'s `NBG` and prefix length until
  the crash reproduces in the small screen, then read it there (the screen has the diagnostics; the
  perf driver deliberately has none).
  Standing: **baseline measured, ON arm unmeasurable until this crash is fixed.** The 30%/10% bar
  therefore still has no ON-arm number, and the honest reason is a stability bug, not a missing run.

- **[x] Crash does NOT reproduce at NBG=12 (14 mixed steps, `ver_rows` up to 7, 0 tracebacks).**
  So concurrency and verify-row count are NOT the trigger. What still differs in the perf workload:
  `generated-shared-prefix` gives every row an 8192-token SHARED system prompt, so the radix cache
  serves most of each prefix and a verify row's prefix is largely *cached* rather than freshly
  written. That is the remaining variable to test -- run `portcorr3` with a shared prefix across the
  background rows (or point it at `generated-shared-prefix`) rather than independent random prompts.
  Note also the perf run used FA3's own default earlier but triton here; the crash is in triton's
  `_small_copy_kernel`, so re-test the perf workload on fa3 as well to see whether the crash follows
  the backend or the workload.
  Correctness coverage extended for free by this run (gate (b), triton, port 57):
  `ver_rows=6`, 24 rows/step, NULL prefill err=0.0 throughout, `merged_vs_ref` 0.36-0.54 with
  **1-6 flips of 24** -- same residual regime as the small screen, no degradation as verify rows
  scale 1 -> 6. Still the bf16-vs-bf16 limit; the fp32 reference remains the open gate.

# DECISION (2026-10-08, end of session): pivot to the KERNEL on the fork

Taken by me, not deferred: three rounds of feedback have flagged the same gap, and the goal says
explicitly to implement the kernel rather than keep testing. The upstream correctness chain has
consumed this session and, while it produced real fixes (ports 57/58) and the first valid baseline,
it has produced **no ON-arm perf number and no kernel work**. Continuing it is the wrong priority.

**What is already in place on the fork** (`attn_fuse_`, this checkout) and must be used rather than
re-derived: the two-executor SPLIT path exists (`flashinfer_backend.py`, `fa3_only_run` +
flashinfer persistent BatchAttention, SM partition, two-stream overlap; see commits 144a73dc8,
e237ab8bc, 10b378d89), the 4-mode MTP step works, and the split was verified numerically
(v0.19.1: FA3 prefill rows 0.0032-0.0040 and FA2 verify rows 0.0030-0.0034 rel err vs an fp32
recompute -- that IS the fp32 gate upstream still lacks).

**Order of work next session, kernel first:**
1. On the FORK, run the 4-mode serving A/B with the split kernel ON vs OFF on the mixed
   chat+tracelab workload, and report TTFT/ITL/throughput beside `mixed_fused_fa3=N(xx%)` and the
   fallback count (see [[fused-ab-validity-rule]]). The fork has the tracelab dataset and
   `bench_serving --dataset-name tracelab-claude`; upstream does not, which is what invalidated the
   first A/B attempt here. This is the number the 30%/10% bar needs, and nothing blocks it.
2. Then optimise the kernel on that workload: the SM split boundary (the 2-D dispatch TODO), the
   wasted `eagle_prepare_for_verify` plan per 4-mode step noted in `forward_mixed_spec`, and
   `tma_store_wait` for multicast / cross-role backfill.
3. Only then return to upstream: the shared-prefix crash, and porting the fp32 verify gate.

**Upstream state left clean and documented:** 60 commits on `upstream_port`
(worktree `/opt/tiger/sglang-up`), 4-mode step runs (14 mixed steps, ver_rows to 7, 0 tracebacks on
`portcorr3`), fields correct under triton, one open stability bug under shared-prefix workloads, and
gate (b) + the always-on counter in place. Nothing is half-edited; both reverts were deployed.

**THE REAL BLOCKER for the kernel A/B, found while acting on the pivot:** the vibecuda H200 pod has
the fork's python tree (`/dev/shm/w/python`), its venv (`/dev/shm/w/venv`) and the tracelab subset
(`/dev/shm/w/tracelab_trace.jsonl.gz`), but **`/dev/shm/w/fi` does not exist** -- the flashinfer
fused/split-kernel build is NOT on this pod. The split kernel therefore cannot be run here at all,
which is the concrete reason this session's work kept landing on upstream scheduling instead of the
kernel. It is not a priorities problem alone; the kernel had no runnable binary on the only GPU.

So step 1 of the pivot is **restore the fused extension on the pod**, before any kernel A/B:
- the fork's `drivers/build_sglang_tree.sh` is what provisions `/dev/shm/w`; check whether it also
  builds/copies `fi`, and whether a prebuilt wheel/`.so` exists to avoid a long nvcc build;
- the extension must be built against the pod's torch (the venv the fork uses, NOT `venv_up`'s
  torch 2.14.1 -- see [[fused-two-register-costs]] and the earlier wrong "imports cleanly" lesson:
  verify `import flashinfer` AND that the `.so` loads, not just the Python import);
- then confirm the arm is really fused via `mixed_fused_fa3=N(xx%)` per [[fused-ab-validity-rule]]
  before quoting any number.
Only after that does the ON-vs-OFF split-kernel A/B on chat+tracelab become runnable -- and that
A/B is what the 30% TTFT/ITL + 10% throughput bar needs.

**Restore recipe for `fi`, executed (2026-10-08).** The fused extension is JIT-built at runtime
(`flashinfer/jit/`) from `csrc` + `include`, so the pod only needs the source tree, not a built
`.so`: `flashinfer` (3.1M) + `csrc` (3.7M) + `include` (3.2M) = **1.7M compressed**, pushed with
`px.py` to `/dev/shm/w/fi` (the default `FI_ROOT` that both `tests/test_fused_server_batch.py` and
the fork's `flashinfer_backend.py` import). Local source: `/opt/tiger/flashinfer` @ `b0563ae`
("FA3 upstream port: causal (bottom-right) by default"), which carries the FA2-persistent and
FA3-port work. Do NOT push the whole 170M tree; the three dirs above are sufficient.
After it lands, in order: (1) `FI_ROOT=/dev/shm/w/fi` + the pod venv, run
`tests/test_fused_server_batch.py` on fixture 1 to confirm the kernel BUILDS and PASSES there --
first JIT build will be slow; (2) check the `.so` actually loads, not just `import flashinfer`
([[fused-two-register-costs]]); (3) only then the ON/OFF split-kernel serving A/B on chat+tracelab,
reporting `mixed_fused_fa3=N(xx%)` and fallbacks beside the timings ([[fused-ab-validity-rule]]).
Caveat to check first: `wgmma split-KV is wrong, not unexercised` ([[fused-wgmma-splitkv-wrong]])
and the retracted symmetric mode ([[fused-symmetric-backfill-is-the-fix]]) -- the A/B must use a
split/mode those memories do not invalidate.

**`fi` restore, corrected and in progress (2026-10-08).** Two corrections to the recipe above,
both found before wasting the push:
1. The fused kernel is **`{FI}/tests/fused_fa3_persistent_ext.cu`** (30 KB) -- it lives in
   `tests/`, which the first tarball omitted. That push would have completed and still failed.
   `tests/test_fused_server_batch.py` compiles it with `torch.utils.cpp_extension.load`.
2. It needs `3rdparty/cutlass/include`, `3rdparty/cutlass/tools/util/include` and
   `3rdparty/spdlog/include` (from the `load()` call's `extra_include_paths`), plus a generated
   `batch_attention_config.inc`.
**The pod HAS network access** (`git ls-remote` to GitHub succeeds), so the bulk is cloned there
rather than pushed: cutlass @ `a49a78ffefc86a87160dfe0ccc3a3a2d1622c918` (186 M) and spdlog @
`c3aed4b68373955e1cc94307683d44dca1515d2b` (5.2 M) are now in `/dev/shm/w/fi/3rdparty/` (pins read
from `git submodule status` in `/opt/tiger/flashinfer`). Only our delta --
`flashinfer csrc include tests` = 1.94 MB compressed, 44 base64 chunks -- goes over the webshell.
Use this split for any future pod restore: clone third-party, push only the fork's own sources.
Local flashinfer: `/opt/tiger/flashinfer` @ `b0563ae`.

**Second pod-restore casualty found: the FORK's venv torch is broken.**
`/dev/shm/w/venv/bin/python -c "import torch"` fails with
`ImportError: /dev/shm/w/venv/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so: undefined
symbol: ncclCommResume`. So the fork could not have served on this pod even with `fi` present --
two independent casualties of the same restore (cf. [[pod-tree-verify-manifest]], which is exactly
this failure mode and says to verify the tree by manifest before trusting any arm).
`venv_up` (upstream, torch 2.14.1) imports fine; `nvcc` is CUDA 13.1 and the GPU is an H200 (sm_90a,
which the fused `load()` targets with `-arch=sm_90a`).

**Consequence, and it is good news for the goal's ordering:** the fused kernel test
(`tests/test_fused_server_batch.py`) is STANDALONE -- torch + flashinfer + cutlass, no sglang -- so
the kernel can be validated against the causal fp32 reference using `venv_up`'s python without
repairing the fork venv first. That is the goal's "implement/validate the kernel" step, unblocked
by a path that does not depend on the broken fork environment. Repairing the fork venv (a
torch/nccl ABI mismatch) is only needed later for the fork SERVING A/B.

## 2-D dispatch boundary: WHY it bends, and the rule it implies — 2026-10-08

Re-analysis of the existing measured surface (`tasks/fused_specdec_workload/boundary_28x4x128.json`,
25 points, speedup = min(seq, conc) / fused; no new GPU time needed). The boundary is a **RATIO**,
not two independent thresholds, and that is why thresholding two axes separately could never fit it:

| kv | nd at which fused starts winning | kv / nd |
|---|---|---|
| 512 | 8 | 64 |
| 2048 | 32 | 64 |
| 8192 | 115 | 71 |
| 16384 | 365 | 45 |
| 32768 | 365 | 90 |

- **Monotone in both axes, opposite directions**: mean speedup by `nd` is
  0.642 / 0.902 / 0.934 / 0.982 / 1.013 for nd = 0 / 8 / 32 / 115 / 365; by `kv` it is
  1.086 / 0.949 / 0.853 / 0.815 / 0.770 for kv = 512 ... 32768.
- **At `nd = 0` the fused kernel ALWAYS loses** (0.617-0.704x across every kv). No rule phrased on
  `qo_len`/`kv` alone can represent that, which is precisely the defect in the harness rule
  (`four_mode_dispatch.route`: `qo_len >= 128 and kv >= 4096`) and in the serving rule
  (`mixed_dispatch.py`: `packed_qo >= 16, kv_tiles >= 1`). Both omit the decode population, which
  the data says is the deciding variable.
- **Derived rule: dispatch fused iff `kv <= 64 * nd`** (equivalently `kv_tiles <= nd` at a 64-wide
  KV tile -- one decode row per KV tile of the FA3 row). It reproduces the flip at kv 512 and 2048
  exactly and kv 16384 correctly; its two misses (kv 8192 at nd 115, kv 32768 at nd 365) are both
  in the flat region where measured speedup is 1.008-1.026, i.e. where the choice does not matter.
- **Mechanism this supports** (the candidate named in the item, now quantified): the fused win is
  bounded by whether the co-resident decode population can fill the SMs FA3 would otherwise leave
  idle, at roughly ONE 64-wide KV tile of decode work per decode row. KV length only matters through
  how many tiles the FA3 row needs -- hence a ratio.

Still to do on this item (needs the GPU, blocked on the `fi` restore): fit the boundary on the full
(packed_qo, kv_tiles, decode_rows, decode_kv) instead of the 2-D (kv, nd) slice; re-check
`MIN_QO_LEN=16` against the measured 32 -> 56 crossover (FA3 0.78x at packed_qo 32, 1.13x at 56, so
a floor of 16 admits rows where FA3 LOSES); then make the harness and serving rules agree on the
ratio form, since they currently disagree with each other AND with the data.

### Implemented: the decode-population guard in the serving dispatch — 2026-10-08

`python/sglang/srt/layers/attention/mixed_dispatch.py`: `split_classes` now applies
`_apply_decode_population_guard`, a batch-level ratio test. The per-row conditions
(`packed_qo >= 48`, `kv_tiles >= 1`, `qo >= MIN_QO_LEN`) read only one row's own shape and so
cannot see the decode population at all -- which the surface says is the deciding variable.
Guard: with no decode rows, no row goes to FA3 (measured 0.617-0.704x on every kv); otherwise a row
goes to FA3 only while `kv_tiles <= decode_rows / 2` (the measured `kv <= 64*nd` at TILE_KV=128).
Env: `SGLANG_FUSED_DECODE_GUARD=0` disables it, `SGLANG_FUSED_DECODE_TILES_PER_ROW` retunes the
ratio, so the next GPU session can A/B the guard itself rather than trusting this fit.

**Validated by replaying all 25 measured points through the real dispatch code** (one qo=424 prefill
row + nd qo=1 decode rows at each kv, GQA 7):

| rule | agrees with measured winner |
|---|---|
| old (per-row only) | **11 / 25** -- it routes FA3 for every point, including all five `nd = 0` cases where fused loses 30-38% |
| new (with the guard) | **23 / 25** |

The two misses are kv 8192 @ nd 115 and kv 32768 @ nd 365, measured 1.026x and 1.016x -- the flat
region where the choice is worth <3%, which is where a ratio fit is expected to be soft.
Repro: `python3` replay snippet against `tasks/fused_specdec_workload/boundary_28x4x128.json`;
sglang `attn_fuse_` @ this commit, flashinfer `/opt/tiger/flashinfer` @ `b0563ae`.
NOT yet A/B'd on a live server -- that needs the `fi` restore to finish; the harness/serving rule
reconciliation and the `MIN_QO_LEN=16` recheck remain open on the item above.

**`fi` restore COMPLETE on the pod (2026-10-08).** All seven paths the fused build needs verify
present: `flashinfer/`, `csrc/`, `include/`, `tests/fused_fa3_persistent_ext.cu`,
`3rdparty/cutlass/{include,tools/util/include}`, `3rdparty/spdlog/include`. Total cost: one 2.59 MB
base64 push (44 chunks) plus two pod-side git clones.
One more restore gotcha, worth the line because it is invisible until the test runs:
`flashinfer/data/{csrc,include,cutlass,spdlog,tvm_binding}` are **absolute** symlinks into
`/opt/tiger/flashinfer/...`, so on the pod they are all broken. The failure does not look like a
broken link -- `flashinfer/jit/core.py` does `os.makedirs(FLASHINFER_CSRC_DIR, exist_ok=True)`, and
`exist_ok=True` still raises `FileExistsError` when the path exists but is not a directory, which a
dangling symlink is. Repointed them at `/dev/shm/w/fi/...`. Add this to the restore recipe.
Runner: `/dev/shm/kerneltest.sh` (verifies the seven paths, then runs the test under `venv_up`).

**Kernel test on the restored pod: two more setup gates cleared, one in flight (2026-10-08).**
`import flashinfer` works and the test now gets as far as building the batch
(`qo=[424, 87, 1] kv=[4001, 87, 14] packed_qo=[2968, 609, 7]`, `row 0 -> FA3; rows 1.. ->
persistent tasks [0, 2]`, 4102 pages) -- so fixture 1 is being set up correctly from the restored
tree. It then dies at `gen[0]` with `IndexError`: the test globs
`~/.cache/flashinfer/*/generated/batch_attention_with*` for `batch_attention_config.inc`, and
flashinfer's JIT codegen has never run on this pod, so the glob is empty.
Fix: `/dev/shm/jitwarm.sh` exercises `BatchAttention.plan/run` once to force codegen (first attempt
failed because I passed a batch_size positional that `plan()` does not take -- its signature is
`(qo_indptr, kv_indptr, kv_indices, kv_len_arr, num_qo_heads, num_kv_heads, head_dim_qk,
head_dim_vo, page_size, causal=...)`; corrected). Re-running; the first JIT build is slow.
Sequence for any future pod restore, all four steps needed before the fused test can run:
clone third-party -> push the fork's sources -> repoint `flashinfer/data/*` symlinks -> warm the
BatchAttention JIT so `generated/batch_attention_with*` exists.

**Codegen cleared; a CUDA 13.1 toolchain risk is now the live question.** The warmup produced
`~/.cache/flashinfer/90a/generated/batch_attention_with_kv_cache_..._flipped_schedule_false`, which
is what the fused test globs for `batch_attention_config.inc` -- so `gen[0]` resolves now.
But flashinfer's OWN ninja build for that module then FAILED on this pod's nvcc 13.1:
`vector_types.h:561: typedef ... double4 __VECTOR_TYPE_DEPRECATED__("use double4_16a or
double4_32a")` -> `ninja: build stopped: subcommand failed`. The pod has **nvcc 13.1 with torch
2.14.1+cu130**; flashinfer `b0563ae` predates CUDA 13's `double4` deprecation.
So the fused `load()` may hit the same wall. If it does, the fix is a compile flag, not a code
change: the harness already has an `EXTRA_DEFS` passthrough (flashinfer commit b13aa2a) and
`load()` takes `extra_cuda_cflags`, so `-Wno-deprecated-declarations` (or
`-D__VECTOR_TYPE_DEPRECATED__=`) can be injected for the fused TU. Flashinfer's internal JIT builds
would need the same treatment through its own flag env if any sglang path needs them -- but the
standalone kernel test does not, which is why validating the kernel this way stays viable.
Recorded before the result so the next session does not re-derive it.

**The fused build's real blocker was the C++ standard, not CUDA 13.1.** The `double4` deprecation
notes and the `REGISTER_*` macro redefinitions in the log are warnings and pre-existing; the
`fatal : Could not open input file ...compute_90.cpp1.ii` was a downstream symptom, and the pod has
ample room (3.3 TB disk free, 1.6 TB RAM), so the earlier cgroup/space theory was wrong. Grepping
for the first hard error:
`torch/all.h:5: error: #error C++20 or later compatible compiler is required to use PyTorch.`
(and the same for `ATen.h`). **torch 2.14 requires C++20 while the harness passes `-std=c++17`.**
Fixed by switching the fused `load()`'s `extra_cuda_cflags` to `-std=c++20` (one occurrence in
`tests/test_fused_server_batch.py`). Note this is a consequence of validating the kernel against
`venv_up`'s torch 2.14 instead of the fork's older torch -- the flag change belongs with that
choice, and should be made conditional (or the fork venv repaired) rather than committed blindly to
the flashinfer tree.
Lesson that generalises: "could not open input file" from nvcc is almost never the real error --
grep the whole log for the FIRST `error:` before theorising about resources.

**The C++20 requirement bites in TWO places, not one (2026-10-08).** After switching the fused
`load()` to `-std=c++20`, the fused TU itself compiled and the next 100 errors came from a
DIFFERENT file -- `batch_prefill_paged_sm90_kernel_mask_0.cu` in flashinfer's own JIT cache, i.e.
flashinfer's INTERNAL JIT build, which sets its flags independently in
`flashinfer/jit/core.py:265,268` (`cflags`/`cuda_cflags`, both `-std=c++17`). Symptom there is
`c10/core/SymBool.h(59): error: identifier "SymBool" is undefined` plus a cascade -- torch 2.14's
headers are not parseable at C++17, and the error never names the standard.
Patched both occurrences to `-std=c++20` on the pod copy. So the full toolchain rule for this pod:
**torch 2.14 + nvcc 13.1 requires C++20 in the harness `load()` AND in flashinfer's jit core.**
This is a consequence of validating against `venv_up`'s torch; it should be made conditional (or
the fork venv repaired) rather than committed to the flashinfer tree as-is.
Pattern worth remembering: each fix exposed the next build stage, and in every case the real cause
was named only by grepping for the FIRST `error:` -- the trailing lines ("could not open .ii",
"Error limit reached", "ninja: build stopped") named the symptom, never the cause.

## [x] FUSED KERNEL BUILDS AND RUNS on the restored pod; FA3 half PASSES, task0 row FAILS — 2026-10-08

First kernel-arm result since the pod restore. Command:
`NVCC_THREADS=4 bash /dev/shm/kerneltest.sh` (= `FI_ROOT=/dev/shm/w/fi PYTHONPATH=/dev/shm/w/fi
TORCH_CUDA_ARCH_LIST=9.0a /dev/shm/w/venv_up/bin/python tests/test_fused_server_batch.py`), fixture 1
`qo=[424, 87, 1] kv=[4001, 87, 14]`, packed_qo `[2968, 609, 7]`, GQA 28/4, H200 sm90a, nvcc 13.1,
torch 2.14.1+cu130. sglang `attn_fuse_` @ 54ae76a33, flashinfer `/opt/tiger/flashinfer` @ b0563ae
(+ the two local `-std=c++20` patches on the pod copy).

| row | class | shape | rel err vs causal fp32 | verdict |
|---|---|---|---|---|
| 0 | FA3, split-KV `num_splits=[4]` | qo=424 kv=4001 | **0.00662** | **ok** |
| 1 | persistent task0 | qo=87 kv=87 | nan | **FAIL** |
| 2 | persistent task2 | qo=1 kv=14 | **0.00070** | **ok** |

`FAILURES: [(48, -1, 'unwritten rows', 55), (32, ...), (16, ...)]` -- 55 of 512 output rows are
never written (NaN), at **every** sm_split tried (48/32/16).

What this establishes and what it does not:
- **The FA3 executor computes correctly inside the fused kernel**, and it does so WITH split-KV
  active (4 splits). That is the first evidence on this pod against the standing worry in
  [[fused-wgmma-splitkv-wrong]]; it does NOT clear that memory, which concerns fixture 2 -- this is
  fixture 1, whose rows have kv >> packed_qo and one task-3 row.
- **The FA2 persistent executor computes correctly for a decode row** (task2, 7e-4).
- **The task0 (append/extend) row is not fully written.** Deterministic, and INDEPENDENT of the SM
  split -- identical 55 rows at 48, 32 and 16 -- so it is not SM partitioning, not work-stealing,
  not a race. `87 - 32 = 55` suggests the row's q extent is covered only to a 32-row granularity,
  i.e. a q-tile class / work-item mapping gap, which is exactly the failure family in
  [[fused-harness-vs-sglang-layout]].

Next, in order: (1) confirm the arithmetic by varying ONLY row 1's qo (e.g. 32, 33, 64, 87, 96) and
seeing whether unwritten == qo - 32 each time -- that single sweep distinguishes a fixed 32-row
write window from a tile-rounding bug; (2) check whether the harness or the kernel owns the gap by
reading how task0 work items are enumerated for a 609-packed-row request; (3) only then re-run the
sm_split sweep for perf, since a kernel that leaves rows unwritten is doing less work and any
timing from it would be invalid ([[fused-ab-validity-rule]]).

### qo sweep on the task0 row: my "32-row write window" reading was WRONG — 2026-10-08

`bash /dev/shm/qosweep.sh` (varies ONLY row 1's qo; `FIXTURE="qo:424,$Q1,1 prefix:3577,0,13"`,
same build as above):

| qo1 | unwritten rows | row 1 rel err | verdict |
|---|---|---|---|
| 16 | 0/441 | 1.08682 | FAIL |
| 32 | 0/457 | 0.99129 | FAIL |
| 33 | 0/458 | 1.07390 | FAIL |
| 64 | 0/489 | 1.03446 | FAIL |
| 87 | **55**/512 | nan | FAIL |
| 96 | **55**/521 | nan | FAIL |
| 128 | **55**/553 | nan | FAIL |

**Both of my earlier readings are retracted.** `55 = 87 - 32` was a coincidence: the unwritten count
stays at exactly 55 for qo1 = 87, 96 AND 128, so it does not scale with row 1 at all -- the
unwritten rows are **not row 1's**, and they appear only once the batch's total row count crosses
~512 (0 unwritten at 489 total, 55 at 512). And "the task0 row is merely unwritten" is wrong too:
at qo1 <= 64 **nothing is unwritten yet row 1 still FAILS at rel err ~1.0**, which is the signature
of output ~= 0 against a nonzero reference -- written, but not computed. So there are TWO
independent defects, and the NaN one was masking the arithmetic one.

Next, and both are cheap: (1) print WHICH row indices are NaN and attribute them to a row/class --
the hypothesis to kill or confirm is that they belong to the FA3 row (qo 424, split-KV) and are
exposed by a ~512-row buffer or tile bound, not to the persistent half at all; (2) for the task0
arithmetic, check whether its work items are enumerated with the right q offsets for a row whose
prefix is 0 (every failing case here has `prefix[1] = 0`, which is also the shape my notes flag as
the unified serving crash's distinguishing feature: `prefix 0 where kv is shorter than the packed q
extent`). Do NOT take any fused timing until both are fixed.

### NaN rows attributed: they ARE row 1's, and they are its FIRST 55 — 2026-10-08

Added a NaN-index diagnostic to the harness (prints the failing indices and attributes them to a
row span). Result, identical at sm_split 48 and 32:

```
unwritten (NaN) rows = 55/512
NaN idx first/last = [424, 425, 426, 427, 428, 429] ... [473, 474, 475, 476, 477, 478]
NaN by row: row1(task0, span 424-510): 55
nA=512 qo_c=[0, 424, 511, 512]
```

So the NaN rows are **424-478: the FIRST 55 rows of the task0 row**, and its LAST 32 (479-510) are
written. That corrects the previous bullet twice over: the rows DO belong to row 1 (the
"constant 55 regardless of qo1" reasoning was right that it does not scale, wrong that it implied
another row), and the written window sits at the END of the row, not the start.

Reconciling with the qo sweep: writes = qo for qo <= 64, and qo - 55 for qo >= 87 (87->32, 96->41,
128->73). So the unwritten PREFIX saturates at 55 while the written suffix grows. A fixed-size
window does not explain that; a work-item enumeration whose q-offset is computed from the row END
(or from a cumulative offset that is 55 too large for this row) does. Note 55 is also exactly
`qo_c[1] - 369`... no clean relation to 424 or 87, so the next step is to read it off the plan
rather than infer: dump the task0 work items (q_start, q_len, tile index) for this fixture and
compare against the row's true span 424-510.

Combined with "rel err ~1.0 at qo1 <= 64 with NOTHING unwritten", the task0 path has a wrong q
offset AND wrong values; both are consistent with one bug in how task0 work items are addressed,
which is the single thing to fix next. The FA3 row's PASS (0.00662, split-KV active) and the task2
decode row's PASS (0.00070) are unaffected by this and stand.

**Task class is PACKED-based, so the whole qo sweep stayed in task0.**
`task = [0 if p > 64 else (1 if p > 16 else 2) for p in packed]`, `packed = qo * GQA` (GQA 7). So
qo1 = 16 already gives packed 112 > 64 -> task0, and every point in the sweep was the same class.
The sweep therefore varied SIZE within task0, not the class, which is why the written count moves
(qo for qo <= 64, qo - 55 for qo >= 87) while the class verdict never changes. In packed space the
boundary sits between packed 448 (qo 64, all written) and packed 609 (qo 87, 32 written) -- i.e.
between 4 and 5 tiles of 128. That is the granularity to read the plan against.

**Exact next step, and it must be read not inferred** (the last two readings here were both wrong
from arithmetic): dump the persistent executor's work items for this fixture -- per item, the
(request index, q tile index, q_start, q_len, kv extent) -- and compare the set covering row 1
against its true span 424-510 / packed 609. Expected signatures: 4 items instead of 5 means the
tile count is truncated; 5 items with q_start biased by +55 unpacked (+385 packed) means the offset
is wrong; items present and correct means the fault is inside the kernel's write path, not the plan.

**`TRACE_WORK=1` emits nothing useful** (rebuilt the fused TU with `-DFLASHINFER_TRACE_WORK=1`;
zero work/wi/WORK lines in the log, same 55/512 result). So the plan dump has to come from a print
added to the harness or the plan builder, not from that flag -- do not retry it.

**Live hypothesis for the task0 gap, from the kernel source rather than arithmetic:** the build
carries `-DFLASHINFER_FUSED_WG_DRAIN=1`, the "narrow" drain form in which **only warpgroup 0 drains
the FA2 work queue** (`persistent_fa3_fused.cuh` ~line 556: `kNarrowWg = 0u`, and when
`FLASHINFER_FUSED_WG_DRAIN == 1` the other warpgroups skip the drain entirely). If that one
warpgroup does not consume every queued item, the unconsumed items never execute and their rows stay
NaN-prefilled -- which is precisely "rows never written", and it explains why the count is
insensitive to the SM split (the drain structure, not the partition, bounds it). The surrounding
comment block also documents that this choice is load-bearing and asymmetric: draining on warpgroup
1 instead "breaks FA2 as well when it runs BEFORE FA3's body" (all three rows fail, FA3
3.57/4.13/4.41, task0 1.34/0.95/1.22, task2 3.59/3.03/3.62), so the two phases are not
interchangeable and this is not a free knob.
Test in flight: `WG_DRAIN=0` (every warpgroup drains). If the 55 unwritten rows vanish, the gap is
in the narrow drain's item consumption; if they persist, it is in how task0 items are addressed and
the drain is exonerated. Either way the next action is determined, which is why this flip comes
before any further reading.

**`WG_DRAIN=0` is not a usable diagnostic: it crashes.** `torch.AcceleratorError: CUDA error: an
illegal memory access` / `CUDA_ERROR_ILLEGAL_ADDRESS from cuCtxSynchronize_v2`. So the wide drain
form cannot even run on this fixture, which independently confirms the source comment that the
narrow form (`kNarrowWg = 0`) is load-bearing rather than a tunable. The drain-consumption
hypothesis therefore stands UNTESTED -- flipping the flag cannot isolate it.

**Next step, now that the cheap flips are exhausted: instrument the drain itself.** Add a global
counter the narrow drain bumps per consumed queue item (and one the plan bumps per enqueued item),
then compare enqueued vs consumed for this fixture. That is a few lines in
`fused_drain_queue` + a tensor the harness reads, and it answers the question the flag flip could
not: whether the 55 unwritten rows correspond to items that were enqueued but never consumed, or to
items consumed with a wrong q offset. Everything cheaper has now been tried and recorded
(`TRACE_WORK` emits nothing; `WG_DRAIN=0` crashes; the qo sweep and NaN-index attribution are done),
so the counter is the smallest remaining step that can still settle it.

**Traced the task0 gap to the exact loop, and to whose responsibility coverage is.**
`persistent_fa3_fused.cuh:239-252`, the drain loop:
```
while (true) {
  if (ltid == 0) *s_cluster = atomicAdd(params_2.cluster_cursor, 1);
  NamedBarrier::sync(128, kQueueBarrier);
  const uint32_t c = *s_cluster;
  if (c >= n_clusters) break;
  Runner1::Run(params_1, &smem_storage_1, c, tid_base, s_merge);   // tile128 = task0
  Runner2::Run(params_2, ..., c, ...);                              // tile64  = task1
  Runner3::Run(params_3, ..., c, ...);                              // tile16  = task2
}
```
and `n_clusters = params_2.num_pers_clusters`, which the scheduler sets from
`num_clusters = num_sm / cluster_size` (`scheduler.cuh:1325`, capped at 1334) -- i.e. **SM-derived,
not work-derived**. So this is a persistent-worker drain: `c` is a WORKER SLOT, every Runner is
invoked for every slot, and covering all of a task's tiles is each `Runner::Run`'s internal
mapping, not the queue's. That is consistent with the unwritten count being invariant across
sm_split 48/32/16 -- the worker count was never the limiter, so no SM-partition explanation was
ever going to work, and the "narrow drain leaves items unconsumed" hypothesis is now also unlikely
(the loop drains until the cursor passes n_clusters; it does not stop early).

**So the fault is inside `Runner1::Run`'s tile mapping for a task0 request** (tile128 class), which
is where the instrumentation belongs: count, per worker slot, which (request, tile) pairs Runner1
claims, and compare against the 5 packed-128 tiles that row 1's 609 packed rows require. A
device-side `printf` guarded by the existing `-DFLASHINFER_TRACE_WORK` define is the cheapest form
(the define is already plumbed through the harness at `test_fused_server_batch.py:325` but is
referenced NOWHERE in the kernel -- `persistent.cuh:379` only mentions it in a comment, which is
exactly why `TRACE_WORK=1` printed nothing). Adding the printf there makes that flag do what its
name says, for this and every future coverage bug.

### Task0 gap: the mechanism, read off the code — 2026-10-09

`persistent.cuh:273` `Runner::Run` takes a `cluster_override` and does:
```
const uint32_t cluster_idx = (cluster_override != kNoClusterOverride)
                                 ? cluster_override : persistent_cluster_idx<KTraits>();
for (IdType work_idx = work_indptr[cluster_idx]; work_idx < work_indptr[cluster_idx + 1]; ...)
```
So a Runner executes exactly the work range the SCHEDULER assigned to cluster `cluster_idx` via
`work_indptr`. In the fused kernel the override is the drain's claimed slot `c`, and the drain stops
at `c >= n_clusters` where `n_clusters = params_2.num_pers_clusters = num_sm / cluster_size`
(`scheduler.cuh:1325`).

**Therefore: any work the plan placed in `work_indptr[c]` for `c >= num_pers_clusters` is never
executed, and its output rows stay NaN-prefilled.** This fits every measurement collected:
- it is invariant to `sm_split` (48/32/16) because `num_pers_clusters` is derived from `num_sm`, not
  from the FA2 SM allocation;
- it is insensitive to row 1's `qo` beyond the point where the plan starts using the out-of-range
  clusters, which is why unwritten saturated at 55 for qo1 = 87, 96 AND 128;
- it writes a SUFFIX of the row (the tiles that landed in low-numbered clusters) and leaves the
  prefix NaN, matching NaN idx 424-478 with 479-510 written;
- and it explains `rel err ~1.0` with nothing unwritten at qo1 <= 64: with fewer tiles the plan fits
  inside range, so rows get written, but a task0 row whose tiles are split across clusters that the
  fused drain visits in a different order than the plan assumed still merges wrongly.

**Next action (host-side, no kernel edit needed):** print the plan's `work_indptr` for the task0
runner next to `num_pers_clusters` for fixture 1 -- if `work_indptr` has non-empty ranges at
`c >= num_pers_clusters`, this is confirmed and the fix is to build the FA2 plan with exactly the
cluster count the fused drain will iterate (or to have the drain iterate the plan's count). The
harness already holds the plan object (`pp`), so this is a print in
`tests/fused_fa3_persistent_ext.cu` around line 353 where `P.num_pers_clusters` is set, not a device
change. That also makes the `-DFLASHINFER_TRACE_WORK` flag earn its name if wired there.
Four hypotheses are now eliminated with evidence and recorded above: 32-row write window,
rows-belong-to-another-row, SM partitioning, and narrow-drain under-consumption.

**Probe attempt reverted; here is the shape it actually needs.** My first version used
`work_indptr_t.to(at::kCPU)` with a tensor accessor, but the extension holds work_indptr only as a
DEVICE pointer: `P.work_indptr = GetPtrFromBaseOffset<IdT>(pers_ib, pp.tasks[i].work_indptr_offset)`
(`fused_fa3_persistent_ext.cu:318`, inside the per-task loop). Reverted so the TU still compiles --
the pod copy is clean, `#if FLASHINFER_TRACE_WORK` blocks: 0.
The working form is a `cudaMemcpy` of `(num_clusters_in_plan + 1)` ints from that device pointer for
task index 0, where the length must come from the PLAN (not from `num_pers_clusters`, since the
whole question is whether the plan is longer than that bound). So the probe needs the plan's own
cluster count: read it where the plan is built (`scheduler.cuh`, the `num_clusters` at line 1325 /
cap at 1334 that feeds `plan_info.num_pers_clusters = num_clusters` at 1902) and carry it into the
plan struct alongside `num_pers_clusters` if they can differ -- if they are literally the same
variable, then `work_indptr` cannot exceed the drain bound and **this mechanism is refuted**, which
is itself the answer and is checkable by reading those three lines in context rather than running
anything. Do that read FIRST next session; it is cheaper than the memcpy and may settle it outright.

**The cheap read LARGELY REFUTES my own mechanism — recording that before it hardens into a
"finding".** `scheduler.cuh:1324-1337`:
```
int cluster_size = 1;
int num_clusters = num_sm / cluster_size;
const char* cap_env = std::getenv("FLASHINFER_PERSISTENT_MAX_CLUSTERS");
if (cap_env) { int cap = std::atoi(cap_env); if (cap > 0 && cap < num_clusters) num_clusters = cap; }
plan_info.num_blks_x = cluster_size;
plan_info.num_blks_y = num_clusters;
```
The SAME `num_clusters` sets `num_blks_y` (the cluster dimension the work assignment is built
against) and, in the 3-task builder, `plan_info.num_pers_clusters` (line 1902). If one variable
drives both, `work_indptr` cannot have non-empty ranges past the drain's bound, and "plan work
beyond num_pers_clusters is never executed" is WRONG.
Caveat that keeps it from being fully settled: there are **two** such computations
(`scheduler.cuh:1325` and `:2096`), so the fused 3-task path's builder has to be identified before
the refutation is final -- if the plan is built by one and `num_pers_clusters` set from the other,
the mismatch is back on the table. That identification is the first thing to do, and it is a read,
not a run.
So the task0 gap remains UNEXPLAINED, with five hypotheses now eliminated (32-row window,
other-row attribution, SM partitioning, narrow-drain under-consumption, and -- pending the
two-builder check -- plan/drain cluster-count mismatch). The measurements that any explanation must
fit are unchanged and all recorded above: prefix 424-478 NaN with suffix 479-510 written, unwritten
saturating at exactly 55 for qo1 >= 87, rel err ~1.0 with nothing unwritten at qo1 <= 64, and
invariance across sm_split 48/32/16.

**Two-builder caveat closed: the mechanism is DEFINITIVELY REFUTED.** `scheduler.cuh:1325`
(`num_clusters = num_sm / cluster_size`) and `:1902`
(`plan_info.num_pers_clusters = num_clusters`) are both inside **`TwoStageHolisticPlan`**
(declared line 1187); the other computation at `:2096` is in `MLAPlan`, which this path does not
use. One variable, one function -- so `work_indptr` is built against exactly the cluster count the
fused drain iterates, and no work can be stranded past the bound. Hypothesis 5 is dead, by a read
rather than a run, and the plan/drain coverage question is closed.
That leaves the fault inside the per-cluster WORK ASSIGNMENT or Runner1's execution of it, which is
where the device `printf` goes -- the instrument deferred five times in favour of cheaper guesses.

**Device `printf` added to the work loop (`persistent.cuh`, guarded by `-DFLASHINFER_TRACE_WORK`,
which previously existed only as a define with no code behind it). First run's output is NOT
evidence and must not be read as such:** only **17** `[WI]` lines survived, all from clusters
243-257 with EMPTY ranges (`tileQ=16 range=[4,4) n=0`, `tileQ=64 range=[0,0) n=0`), and **zero**
`tileQ=128` lines. For ~258 clusters x 3 runners that is a fraction of a percent of the expected
output -- the device printf FIFO overflowed and kept a tail. Absence of `tileQ=128` lines here says
nothing about whether Runner1 ran; row 2 computed correctly in the same run, which it could not have
done if every range were truly empty.
Two side facts the run DID establish: the drain claims cluster indices well past the SM count (243+,
so `num_pers_clusters` is not ~132), and **the printf perturbs numerics** -- FA3 row 0 went from
PASS (0.00585-0.00734) to FAIL (1.19145) with only the printf added, so any instrumented build is
for coverage questions only and its rel errs must be ignored. That is consistent with this kernel's
documented register sensitivity (168 regs with ~4.4 KB spill).
Re-running with the printf gated on non-empty ranges, which drops the volume ~100x and should fit
the FIFO.

### Work-item trace (non-empty ranges only): task0 items ARE executed — the gap is OUTPUT ADDRESSING

`TRACE_WORK=1` with the printf gated on `work_indptr[c+1] > work_indptr[c]`:
```
[WI] tileQ=128 cls=0   range=[0,1)   n=1     [WI] tileQ=128 cls=243 range=[15,16) n=1
[WI] tileQ=128 cls=6   range=[2,3)   n=1     [WI] tileQ=128 cls=244 range=[16,17) n=1
[WI] tileQ=128 cls=60  range=[6,7)   n=1     [WI] tileQ=128 cls=248 range=[20,21) n=1
[WI] tileQ=128 cls=61  range=[7,8)   n=1     [WI] tileQ=128 cls=249..254 ...
[WI] tileQ=128 cls=62  range=[8,9)   n=1     [WI] tileQ=16  cls=59, 241, 242 ...
```
Three things established, all measured:
1. **The task0 (tileQ=128) class HAS work items and Runner1 executes them** -- these printfs are
   emitted from inside Runner1's work loop. One item per cluster, `work_idx` spanning 0..26.
2. **The drain reaches cluster 243+**, so `n_clusters` (`num_pers_clusters`) is ~258, NOT the ~132
   I had assumed from the H200's SM count when reasoning about coverage. Every cluster holding a
   task0 item is therefore visited, which independently confirms the refutation above: coverage is
   complete and the plan/drain mismatch is dead for a second, measured reason.
3. **So the fault is in WHERE the executed items write, not in whether they run.** That is a
   different code region from everything examined so far: the output/epilogue addressing in
   Runner1's work-item handling (the `q_indptr` / `partial_o` / row-offset computation for a
   tile128 work item), not the queue, the plan, the SM split, or the drain.

Instrument caveat, unchanged and important: exactly 17 printf lines survived again, while
`work_idx` up to 26 implies >= 27 task0 items, so the FIFO is still truncating and this list is a
SAMPLE -- it proves items execute, it does NOT enumerate them. Raise
`cudaLimitPrintfFifoSize` before drawing any count-based conclusion from it. And the printf build
still corrupts FA3 numerics, so it stays coverage-only.

Next: read Runner1's output addressing for a tile128 work item and compare the row offset it
computes against the request's true `qo` span (424-510 for row 1). The suffix-written/prefix-NaN
shape (479-510 written, 424-478 not) is what an offset computed from the wrong base, or a span
clipped to the last tile, would produce.

### Strongest candidate yet, from reading the output/bound addressing — 2026-10-09

Two reads in `persistent.cuh`'s work-item body:
- **Ruled OUT (candidate 6):** the output row base is
  `qo_packed_idx_base = packed_qo_start + persistent_cluster_slot<KTraits>() * CTA_TILE_Q`
  (line 398), and `persistent_cluster_slot()` is `KTraits::FUSED_GRID ? 0u : blockIdx.x`
  (line 200-202). So the fused path deliberately forces slot 0, which is right for the
  one-CTA-per-cluster fused layout, and `cluster_idx` separately takes the drain's override
  (line 358). No slot/idx mismatch -- this was my first guess on seeing line 398 and it is wrong.
- **CANDIDATE (not yet tested):** `const uint32_t cluster_tile_q = gridDim.x * CTA_TILE_Q;`
  In the STANDALONE persistent kernel `gridDim.x` is the cluster size (CTAs per cluster, and the
  plan sets `cluster_size = 1`), so `cluster_tile_q == CTA_TILE_Q`. In the FUSED kernel `gridDim.x`
  is the whole fused grid (the drain reaches cluster 243+, so it is O(258)), which makes
  `cluster_tile_q` ~130-260x too large. It feeds the causal/kv bound
  `min((kv_len - q_len) + ceil_div(packed_qo_start + cluster_tile_q, gqa_group_size), kv_len)`,
  so an inflated value pushes `kv_end` to `kv_len` for rows where it should be clipped.
  Why this fits the evidence better than anything so far: it corrupts VALUES without touching
  coverage (explaining `rel err ~1.0` at qo1 <= 64 where every row is written), it is independent
  of `sm_split` (gridDim is the fused grid, not the split), and it would hurt a task0 row
  (packed 609, multi-tile, causal clipping matters) far more than a task2 decode row
  (packed 7, kv 14, where the bound saturates at `kv_len` anyway and the row PASSES at 7e-4).
  It does NOT by itself explain the 55 NaN rows, so there may still be two faults.

**Check, cheap and decisive:** print `gridDim.x`, `cluster_tile_q` and `CTA_TILE_Q` from the
existing `[TRACE]` printf already present at line ~386 (it prints the full work-item coord tuple --
`q_indptr, q_len, packed_qo_start, o_indptr, kv_start, kv_end, ...` -- and was already there before
my instrumentation; find its guard, it is NOT `FLASHINFER_TRACE_WORK`). If `cluster_tile_q` is
~33 000 instead of 128, the candidate is confirmed and the fix is to derive it from the plan's
`cluster_size` (or `num_blks_x`) rather than `gridDim.x` under `FUSED_GRID`.
Remember to raise `cudaLimitPrintfFifoSize` first -- the FIFO truncated at 17 lines twice, and this
trace prints far more per item.

### FIX APPLIED (under test): `cluster_tile_q` must not use `gridDim.x` under FUSED_GRID

`persistent.cuh:327`, changed on the pod copy from
`const uint32_t cluster_tile_q = gridDim.x * CTA_TILE_Q;` to
`const uint32_t cluster_tile_q = (KTraits::FUSED_GRID ? 1u : gridDim.x) * CTA_TILE_Q;`

Rationale, and why the asymmetry is the tell: `persistent_cluster_slot()` IS guarded
(`FUSED_GRID ? 0u : blockIdx.x`, line 200) because the fused layout runs ONE CTA per cluster.
`cluster_tile_q` three lines earlier is NOT guarded, so it still reads `gridDim.x` -- which in the
fused kernel is the whole fused grid (measured: the drain reaches cluster 243+), overstating a
cluster's q extent by ~130-260x. It feeds the causal kv bound (line 375) and the qo upperbound
(line 416). The fused path was taught about the slot and not about this.
My device printf was REVERTED in the same edit, so this run's rel errs are valid (the instrumented
build corrupted FA3 from 0.0059 to 1.19).
Under test now; the predictions to check against, so the result cannot be read loosely:
- task0 row 1's rel err should drop from `nan` / ~1.0 toward bf16 noise (~1e-3);
- the task2 decode row should stay PASS (its bound saturates at `kv_len` either way, so a change
  there would mean the fix did something unintended);
- FA3 row 0 should stay PASS at ~0.006 (it does not go through this code);
- the 55 NaN rows may well REMAIN -- this candidate explains wrong VALUES, not missing writes, so
  if they persist there is a second, independent fault and that is the expected outcome rather than
  a failure of the fix.

**`cluster_tile_q` fix: fixture-1 numbers UNCHANGED, and the test was the wrong one.**
With the fix in and the printf out: `unwritten = 55/512`, row 0 FA3 **0.00710 ok**, row 2 task2
**0.00089 ok**, row 1 task0 **nan FAIL** -- at sm_split 48/32/16 alike. Predictions 2 and 3 held
(FA3 and task2 untouched, as intended). Prediction 1 could not be evaluated at all: `rel err` is
computed over the row's span, so **any unwritten row makes it `nan` regardless of whether the
written values improved**. Fixture 1 therefore cannot measure this fix -- the NaN rows mask it.
The discriminating case is `qo1 = 64`, where the sweep showed 0 unwritten and `rel err 1.03446`:
there the number is meaningful. Running `FIXTURE="qo:424,64,1 prefix:3577,0,13"` now.
If it drops to ~1e-3 the fix is real and the remaining 55-row fault is separate and independent;
if it stays ~1.0 the `cluster_tile_q` candidate is refuted too and should be reverted rather than
left in as a plausible-looking change.
Method note for this file: I committed the predictions BEFORE the run, which is what exposed that
the headline metric could not test the hypothesis -- worth keeping as the habit, since five earlier
rounds here read whatever number appeared as if it bore on the question.

**`cluster_tile_q` REFUTED and REVERTED (candidate 7).** The discriminating run
(`FIXTURE="qo:424,64,1 prefix:3577,0,13"`, 0 unwritten rows so `rel err` is meaningful):

| | before fix | after fix |
|---|---|---|
| row 1 task0 (qo=64) | 1.03446 FAIL | **1.04120 FAIL** |
| row 0 FA3 | ~0.0066 ok | 0.00660 ok |
| row 2 task2 | ~0.0008 ok | 0.00082 ok |

No effect. Per the criterion committed BEFORE the run, the change is reverted rather than left in
as a plausible-looking edit; the pod copy is back to `cluster_tile_q = gridDim.x * CTA_TILE_Q`.
(The `gridDim.x`-under-FUSED_GRID asymmetry may still be a latent bug worth raising separately --
`persistent_cluster_slot` IS guarded and this is not -- but it is demonstrably NOT the cause here,
and an unvalidated "fix" in the tree is worse than none.)

**Where that leaves the task0 fault, stated as what must now be explained:** with EVERY row written
(qo1 <= 64) the task0 row's output is still wrong at rel err ~1.0, which is the signature of output
~= 0 against a nonzero reference, while the tile16 runner (task2) is exact at ~8e-4 in the SAME
kernel launch. So the fault is specific to the **tile128 runner's** production of output, not to
scheduling, coverage, bounds, slots, or the drain -- seven candidates now eliminated with evidence.
**Next, and it is one cheap read of existing output:** the harness's failure branch already prints
`nonzero=N/total absmax=...` for the failing block (`test_fused_server_batch.py` ~line 706). Run
qo1=64 WITHOUT the grep filter and read it: `nonzero = 0` proves the tile128 runner never writes
its accumulator (an epilogue/store fault), while `nonzero > 0` with a wrong absmax means it writes
computed-but-wrong values (a mainloop/accumulator fault). Those two point at opposite halves of
Runner1 and the existing print distinguishes them without any new instrumentation.

**The harness already had the right diagnostic; it is behind `DUMP_ROWS=1`.**
`test_fused_server_batch.py:671` guards the failure-branch dump, and its own comment states the
discriminator I had been planning to build: *"small-but-nonzero means the mainloop accumulated part
of the KV"*. It prints the per-token error profile, which heads/tokens are bad, and the reference
values. So the store-fault vs compute-fault question needs no new instrumentation at all -- only
`DUMP_ROWS=1`, which nothing in this investigation had set.
Running it on the discriminating fixture (`qo:424,64,1 prefix:3577,0,13`, 0 unwritten rows, task0
rel err 0.93-1.04 across repeats while FA3 holds ~0.0066-0.0074 and task2 ~0.0008).
Reading rule, fixed in advance: all-zero output => the tile128 runner's epilogue never stores;
small-but-nonzero => its mainloop accumulated only part of the KV; full-magnitude-but-wrong => the
accumulation or the mask is wrong. Those are three different halves of Runner1 and the dump
separates them in one run.

### [x] task0 fault CHARACTERISED: output fully written, magnitude ~3.4x too small — 2026-10-09

`DUMP_ROWS=1 FIXTURE="qo:424,64,1 prefix:3577,0,13"` (0 unwritten rows, so every number is
meaningful), same build as above with `cluster_tile_q` reverted:

| row | class | rel err | nonzero | absmax | ref_absmax | ratio |
|---|---|---|---|---|---|---|
| 0 | FA3 | 0.00682 ok | 1519616/1519616 | 0.03320 | 0.03611 | 0.92 |
| 1 | **task0** | **1.02823 FAIL** | **229376/229376** | **0.42773** | **1.45312** | **0.29** |
| 2 | task2 | 0.00089 ok | 3584/3584 | 0.48242 | 0.48178 | 1.00 |

By the rule fixed BEFORE the run: all-zero would mean the epilogue never stores; full-magnitude-wrong
would mean a bad mask/accumulation; **small-but-nonzero means the mainloop accumulated only part of
the KV**. Every one of task0's 229376 elements is written and nonzero, and the magnitude is
**3.40x short** (1.45312 / 0.42773), while task2 is exact to 3 decimal places in the SAME launch.
So the tile128 runner stores fine and computes a systematically UNDER-weighted result.

That narrows the remaining space to the softmax/accumulation balance rather than coverage or
addressing -- eight candidates now eliminated. The specific shapes that produce a uniform
magnitude deficit with every element present:
1. numerator accumulated over a subset of KV while the denominator (row sum of exp) covers the full
   range -- i.e. V-accumulation and the LSE disagree about how much KV was visited;
2. a split/merge path that AVERAGES partial results instead of combining them LSE-weighted (note
   `[nsd] per-request num_splits = [4]` appears in these runs, so splitting IS active);
3. an output scale applied once per work item on a row that is covered by several.
All three are testable cheaply: run the SAME fixture with splitting forced off for the FA2 half (or
one work item per row) -- if the deficit disappears, it is (2)/(3); if it persists, it is (1).
Do NOT guess between them; the ratio 3.40 is itself a clue worth checking against the number of
chunks the plan gave row 1.

**Refinement: the `num_splits=[4]` in these logs is FA3's, not task0's.** `_nsd_tensor(QO, kvl,
N_FA3_ROWS, ...)` is computed for the FA3 rows only (`test_fused_server_batch.py:140-148`), and it
prints a one-element list because fixture 1 has one FA3 row. So hypothesis (2) as phrased -- "the
split/merge path averages partial results" -- was about the wrong path. The task0 row's chunking is
the PERSISTENT runner's own `kv_chunk_idx / num_kv_chunks / partial_o` merge (those fields appear in
the pre-existing `[TRACE]` work-item print), and in the fused kernel that merge is driven through
the `s_merge_flag` the drain passes into `Runner::Run` -- a per-warpgroup flag, per the drain's own
comment that "the cursor and merge flag were already split this way, the storage was not".
So the live question is specifically: **for a task0 row covered by several KV chunks, does the fused
drain's per-warpgroup merge combine the partials correctly?** A uniform ~3.4x magnitude deficit with
every element present is what a merge that drops or under-weights partial contributions produces,
and it would spare task2 (kv 14, one chunk) and FA3 (its own split path) exactly as observed.
Next run, and it is a single number: print `num_kv_chunks` and `len_kv_chunk` for row 1 from the
existing `[TRACE]` print (find its guard; it is not `DUMP_ROWS` and not `FLASHINFER_TRACE_WORK`) and
check whether it is ~3-4 -- matching the 3.40 deficit -- or 1, which would refute the merge story
outright and send this back to the mainloop.

**Found the pre-existing `[TRACE]` print's guard: `#if 0` with the note "flip to 1 AND clear
cached_ops to re-enable"** (`persistent.cuh:384`). That is why neither `TRACE_WORK=1` nor
`DUMP_ROWS=1` ever produced it -- it is compiled out by default and needs the extension cache
cleared, because `torch.utils.cpp_extension.load` keys on flags and would otherwise reuse the old
object. Enabled it, filtered to `q_len == 64` (the discriminating task0 row) so the ~17-line device
printf FIFO cannot truncate it, and cleared
`/home/tiger/.cache/torch_extensions/*/fused_fa3`.
It prints exactly the fields this question needs, per work item:
`blk, work, q_indptr, q_len, packed_qo_start, kv_indptr, kv_len, kv_start, kv_end, kvh, chunk,
nchunks, o_indptr, len_kv_chunk`.
Reading rule, fixed in advance again:
- `nchunks` ~3-4 for row 1 => the 3.40x magnitude deficit lines up with a merge that under-weights
  partial contributions, and the fused drain's per-warpgroup `s_merge_flag` is the place to look;
- `nchunks == 1` => no merge is involved at all, the merge story is REFUTED, and a uniform 3.4x
  deficit on a single-chunk row points instead at the softmax denominator or an output scale applied
  in the tile128 mainloop;
- several work items for the same row with overlapping `kv_start/kv_end` => double counting in the
  denominator, which would also shrink the output.

### [x] `[TRACE]` on row 1: merge REFUTED, bounds CORRECT, and the same work item runs on TWO CTAs

`#if 1` on the pre-existing trace (filtered to `q_len==64`), fixture `qo:424,64,1 prefix:3577,0,13`,
cache cleared. Representative lines (sorted, deduped):
```
blk=118 work=2  packed_qo_start=0   kv_start=0 kv_end=19 kvh=2 chunk=0 nchunks=1 len_kv_chunk=64
blk=110 work=4  packed_qo_start=128 kv_start=0 kv_end=37 kvh=0 chunk=0 nchunks=1
blk=114 work=4  packed_qo_start=128 kv_start=0 kv_end=37 kvh=0 chunk=0 nchunks=1   <-- same item
blk=111 work=13 packed_qo_start=256 kv_start=0 kv_end=55 kvh=1 chunk=0 nchunks=1
blk=115 work=13 packed_qo_start=256 kv_start=0 kv_end=55 kvh=1 chunk=0 nchunks=1   <-- same item
blk=105 work=10 packed_qo_start=384 kv_start=0 kv_end=64 kvh=3 chunk=0 nchunks=1
blk=103 work=14 packed_qo_start=128 kv_end=37 kvh=3 ...   (all four kv heads present)
```
Three results, two of them refutations of my own hypotheses:
1. **`nchunks == 1` on every row-1 item -> the merge story is REFUTED** (rule fixed before the run).
   No partial/`s_merge` path is involved, so the 3.40x deficit is not an under-weighting merge.
2. **The causal bounds are CORRECT.** `packed_qo_start` 0/128/256/384 gives `kv_end` 19/37/55/64,
   which is exactly `ceil_div(packed_qo_start + 128, 7)` clipped to `kv_len=64`. So the bound
   arithmetic -- including everything the reverted `cluster_tile_q` change touched -- is right, which
   independently re-confirms that refutation.
3. **NEW: the same work item is executed by TWO different CTAs** (`work=4` on blk 110 and 114;
   `work=13` on blk 111 and 115). The earlier per-cluster trace showed exactly one distinct item per
   cluster (`cls=60 -> [6,7)`, `cls=61 -> [7,8)`, ...), so two CTAs must have been handed the SAME
   cluster index by `atomicAdd(params_2.cluster_cursor, 1)`.

Honest limit on (3): a duplicated PLAIN write is idempotent and would not change the result, so
duplication alone cannot explain a 3.40x magnitude deficit -- there must be accumulation or a
denominator that double counts for it to matter. So (3) is a real anomaly and a plausible partner to
the deficit, not yet a demonstrated cause. The next step is to check whether the cursor can hand out
duplicates: `*s_cluster` is written by `ltid == 0` then read after a 128-thread barrier, and the
drain runs per warpgroup with `kWgSlot = FLASHINFER_FUSED_WG_DRAIN == 1 ? 0u : kTidBase / 128u` --
i.e. in the narrow form ALL warpgroups share slot 0, so if more than one warpgroup per CTA (or the
FA3 side) reaches the drain, they share one `s_cluster` and can read each other's claim.
That is a concrete, local, checkable mechanism and it is where to look next.

**RETRACTION, same day: the "same work item on two CTAs" anomaly is an artifact of my own
aggregation.** The harness loops `for sp in _splits:` over sm_split 48/32/16
(`test_fused_server_batch.py:385`), so a single run performs at least three kernel launches -- and I
collected the trace with `grep TRACE ... | sort -u`, which merges lines ACROSS launches. `work=4` on
blk 110 and blk 114 are the same item in two different launches, which is expected and means
nothing. No duplicate execution was observed.
Two further reasons it was never a viable mechanism, both readable: `s_cluster` and `s_merge` are
per-warpgroup (`pers_smem_base + kCursorOff + kWg * 16u`, lines 233/237) with
`kWg = FLASHINFER_FUSED_WG_DRAIN == 1 ? 0u : kTidBase / 128u` (line 226), so in the narrow form only
warpgroup 0 drains and there is no shared-slot race; and `atomicAdd` cannot return the same value
twice within a launch. I had already flagged that duplicated plain writes are idempotent and so
could not produce a magnitude deficit -- that doubt was correct and I should have resolved it before
writing the anomaly up.
**Net state of the task0 fault, unexplained, with what is now PROVEN about it:**
coverage complete (items exist and execute), `nchunks == 1` (no merge), causal bounds exact
(kv_end 19/37/55/64 for packed_qo_start 0/128/256/384), all four kv heads present, every output
element written and nonzero, magnitude uniformly **3.40x short** (0.42773 vs 1.45312), tile16 exact
and FA3 passing in the same launch. Ten candidates eliminated.
Lesson for this file, since it has now cost two write-ups: when a trace is aggregated with
`sort -u`, check how many launches produced it BEFORE drawing any per-launch conclusion. Trace lines
need a launch id (sm_split or a counter) printed on them to be read safely; add that before the next
trace run.

## [x] ROOT CAUSE of the task0 fault: the CAUSAL MASK is not applied in the tile128 path — 2026-10-09

The per-token profile from `DUMP_ROWS=1` (already in `/dev/shm/ktdr.log`; I had greped it away
twice) settles it with no new run:
```
row 1 (task0 qo=64 kv=64) rel err 1.02823 FAIL
  damaged tokens: 63/64  first=[0,1,2,...,11]  last=[57,...,62]
  damaged qo_heads: 28/28
  got[0,:4]=[0.2402, -0.0547, 0.0386, -0.1416]
  ref[0,:4]=[0.1357,  0.3848, 0.2119, -1.3672]
```
**Token 63 is the ONLY correct token**, and `got` is not a scaled `ref` -- it is a different
attention result. With `kv_len = 64`, token 63 is exactly the q token that legitimately attends to
all 64 KV positions. So every token is being computed as though it attended to the FULL KV range:
the per-token causal mask is missing in the task0 (tile128 persistent) path.

Every prior observation follows from this one cause, which is why it is worth more than the ten
eliminated candidates:
- **3.40x magnitude deficit** -- averaging over ~64 KV positions instead of the causal k dilutes the
  output; the deficit is a consequence, not a scale bug (and `got`/`ref` being non-proportional is
  what rules a scale factor OUT).
- **token 63 exact** -- no masking needed there.
- **all 28 qo heads damaged** -- masking is per-token, not per-head.
- **task2 (qo=1) PASSES** -- a single q token attends to all of its 14 KV positions, so a missing
  causal mask is invisible in that class. The tile16 runner's PASS never exonerated the mask.
- **FA3 PASSES** -- it has its own causal path, built here with `-DFLASHINFER_FA3_CAUSAL=1`.
- **tile-level `kv_end` bounds are correct** (19/37/55/64 for packed_qo_start 0/128/256/384), so the
  fault is specifically the mask WITHIN the KV tile, not the tile bound. Those two being separate is
  why the correct-looking bounds misled me for several rounds.
This also matches the failure family recorded in [[fused-harness-vs-sglang-layout]], which lists
"causal mask" explicitly among the places fused bugs live.

**Next (a fix, not a diagnostic):** find how causality reaches the persistent runners. The harness
has `CAUSAL = os.environ.get("CAUSAL", "1") == "1"` and passes `-DFLASHINFER_FA3_CAUSAL=1` for the
FA3 side; the persistent side takes its masking from the `AttentionVariant` / mask mode in its
KTraits or plan. Check whether the fused build instantiates the task0 Runner with a NON-causal
variant while FA3 gets a causal one -- that asymmetry would be the whole bug, and it is a read of
the Runner1 instantiation in `tests/fused_fa3_persistent_ext.cu`.

**CONFIRMED IN SOURCE, and it is a one-token change.** `tests/fused_fa3_persistent_ext.cu:72-75`:
```
template <uint32_t CTA_TILE_Q, uint32_t NUM_MMA_KV>
using PKT = FusedWarpgroupTraits<
    KernelTraits<MaskMode::kNone, CTA_TILE_Q, fa2_mma_q<CTA_TILE_Q>(), NUM_MMA_KV, 8, 8,
                 fa2_warps_q<CTA_TILE_Q>(), fa2_warps_kv<CTA_TILE_Q>(),
                 PosEncodingMode::kNone, PDT, PDT, PDT, float, IdT, StandardAttention<false>>>;
```
Every persistent runner (all three q-tile classes) is instantiated with **`MaskMode::kNone`** -- no
causal mask -- and `StandardAttention<false>`. The token-63 evidence predicted exactly this before I
read it. The FA3 side is separate and built with `-DFLASHINFER_FA3_CAUSAL=1`, which is why it passes
while task0 does not.
Fix under test: `MaskMode::kNone` -> `MaskMode::kCausal` in `PKT`.
Predictions, committed BEFORE the run:
- task0 row 1 (qo=64) rel err should fall from ~0.98-1.04 to bf16 noise (~1e-3);
- task2 row 2 should STAY passing (~8e-4): its single q token needs no mask, so a change there would
  mean kCausal altered something it should not;
- FA3 row 0 should stay ~0.007 (different code path entirely);
- on fixture 1 (qo1=87) the 55 unwritten rows may REMAIN -- the mask explains wrong VALUES, not
  missing writes, so a second fault is still possible and that outcome would not refute this fix;
- if `StandardAttention<false>` also gates masking, rel err may only partially improve -- that would
  point at the variant's own causal flag rather than at MaskMode.

## [x] FIX CONFIRMED: `MaskMode::kCausal` makes ALL THREE classes PASS the fp32 reference — 2026-10-09

One-token change, `tests/fused_fa3_persistent_ext.cu:73`, `MaskMode::kNone` -> `MaskMode::kCausal`
in `PKT` (the traits shared by all three persistent q-tile runners). TRACE disabled and
`~/.cache/torch_extensions/*/fused_fa3` cleared so the numerics are clean.
Command: `FIXTURE="qo:424,64,1 prefix:3577,0,13"` (0 unwritten rows, the discriminating fixture),
`FI_ROOT=/dev/shm/w/fi PYTHONPATH=/dev/shm/w/fi TORCH_CUDA_ARCH_LIST=9.0a NVCC_THREADS=4
/dev/shm/w/venv_up/bin/python tests/test_fused_server_batch.py`.
sglang `attn_fuse_` @ 71047a4c3, flashinfer `/opt/tiger/flashinfer` @ b0563ae + this change.

| row | class | before | **after** |
|---|---|---|---|
| 0 | FA3 (split-KV 4) | 0.00735 ok | **0.00709 ok** |
| 1 | **task0 (tile128)** | **0.98-1.04 FAIL** | **0.00221 ok** |
| 2 | task2 (tile16) | 0.00089 ok | **0.00134 ok** |

0 unwritten rows; identical at sm_split 48 and 32; 0 compile errors. **Every prediction committed
before the run held**: task0 to bf16 noise, task2 still passing, FA3 untouched.
This is the first run in which all three classes PASS the causal fp32 reference, and it closes an
investigation that cost ten refuted candidates. The decisive evidence was the per-token profile
(`DUMP_ROWS=1`, token 63 the only correct token) -- output the harness had been printing all along.
Next: fixture 1 (`qo:424,87,1`), which additionally had 55 unwritten rows. The mask fix explains
wrong VALUES, not missing writes, so that may still fail -- recorded as a prediction, not a hope.

### Fixture 1 after the mask fix: values FIXED, the 55 unwritten rows are a SECOND fault — 2026-10-09

`bash /dev/shm/kerneltest.sh` (default fixture 1, `qo:424,87,1 prefix:3577,0,13`), mask fix in:

| sm_split | unwritten | FA3 | task0 | task2 |
|---|---|---|---|---|
| 48 | 55/512 | 0.00714 ok | nan FAIL | 0.00074 ok |
| 32 | 55/512 | 0.00714 ok | nan FAIL | 0.00074 ok |
| 16 | 55/512 | 0.00714 ok | nan FAIL | 0.00074 ok |

Exactly as predicted before the run: the mask fix cures wrong VALUES (proved on the qo1=64 fixture,
task0 0.00221 ok) and does nothing for MISSING WRITES, so fixture 1's 55 NaN rows are an
independent second defect. `rel err` stays `nan` only because NaN rows poison the row's max.

**Tile arithmetic for the remaining fault, which is now exact:** row 1 spans output rows 424-510
(87 q tokens; packed 87*7 = 609; `CTA_TILE_Q = 128` so 5 tiles at packed 0/128/256/384/512).
NaN rows are 424-478 = q 0-54 = packed 0-383 = **tiles 0, 1, 2**. Written rows are 479-510 =
q 55-86 = packed 385-608 = **tiles 3, 4**. So the LAST TWO tiles execute and the FIRST THREE do not.
Cross-check against the qo1 sweep: at qo1 <= 64 the row needs only 4 tiles (packed <= 448) and
NOTHING is unwritten, so the deficit appears exactly when a 5th tile is required -- consistent with
a cap or an off-by-one that drops the leading tiles once the tile count exceeds 4.
That is a concrete, checkable next step and the instrument already exists: re-run the `[TRACE]`
print (`#if 1` in `persistent.cuh`, filter `q_len == 87`, clear `~/.cache/torch_extensions/*/fused_fa3`)
and list which `packed_qo_start` values appear for row 1. Expect 384 and 512 present with 0/128/256
absent if the tiles are simply never emitted; if all five appear, the items exist and the write
target is wrong instead.
Remember the launch-id caveat recorded above: the harness runs three sm_split launches, so do NOT
`sort -u` across them without printing a launch discriminator.

### Row-1 tile trace (fixture 1, post mask fix): the tiles EXECUTE, so it is output addressing

`[TRACE]` filtered to `q_len == 87`, cache cleared, mask fix in. `packed_qo_start` occurrences
(17 lines total, across the harness's three sm_split launches):
```
packed_qo_start=128 : 3      packed_qo_start=384 : 6
packed_qo_start=256 : 4      packed_qo_start=512 : 4
```
**Tiles at packed 128 and 256 execute**, and those tiles cover q 18-54 -- which is inside the NaN
range. By the prediction committed before the run, that selects the second branch: the work items
exist and run, so the fault is WHERE they write, not whether they are emitted. The emission/cap
theory is out.
Two caveats that bound what this says, both recorded rather than glossed:
1. **The FIFO truncated at 17 lines again** (same cap as the two earlier printf runs), so the
   ABSENCE of `packed_qo_start=0` is not evidence -- it may simply have been dropped. Only presence
   is reliable here. Raise `cudaLimitPrintfFifoSize` before counting anything.
2. **My tile arithmetic in the previous entry used the PRE-fix NaN indices** (424-478). The mask fix
   changed what the kernel computes, so the NaN set must be re-measured post-fix before the
   tiles-0/1/2-vs-3/4 mapping is trusted. Re-run with the NaN-index diagnostic (added earlier,
   prints `NaN idx first/last` and `NaN by row`) on fixture 1 and confirm the span is still
   424-478 before building on it.
Do (2) first; it is one run and it either confirms the mapping or invalidates the arithmetic that
the next hypothesis would rest on.

**Caveat (2) discharged: the NaN span is UNCHANGED by the mask fix**, so the tile mapping is
validated rather than assumed. Post-fix, trace off, numerics clean:
```
unwritten (NaN) rows = 55/512
NaN idx first/last = [424..429] ... [473..478]
NaN by row: row1(task0, span 424-510): 55
row 0 FA3 0.00494 ok | row 1 task0 nan FAIL | row 2 task2 0.00061 ok
```
Same 424-478 as before the fix. So: **unwritten = q 0-54 = packed 0-383 = tiles 0/1/2; written =
q 55-86 = packed 385-608 = tiles 3/4**, and the trace proves the tiles at packed 128 and 256 DO
execute. Two statements that must both be true, and together they pin the fault:
the work for q 18-54 runs, and the output rows for q 18-54 are never written.

**The two expressions that decide this, for the next session to read (not guess):**
```
qo_packed_idx_base = packed_qo_start + persistent_cluster_slot<KTraits>() * CTA_TILE_Q;  // slot=0 under FUSED_GRID
qo_upperbound      = min(q_len, ceil_div(qo_packed_idx_base + CTA_TILE_Q, gqa_group_size));
```
(`persistent.cuh:398` and `:400`.) For the packed-128 tile these give base 128 and upperbound
min(87, ceil(256/7)) = 37, i.e. it should write q rows 18..36 -- which are NaN. So either the write
loop's lower bound is not `qo_packed_idx_base / gqa_group_size`, or the store is gated by a
condition that excludes all but the last tiles. Read the epilogue's row loop and its guard; the
numbers above give the expected and actual spans to compare against, so this is a bounded check.
Note `o_indptr = 0` for row 1 in the trace while `kv_indptr = 4001` is absolute -- the FA2 plan is
built over the FA2 sub-batch (q/o offsets relative, kv absolute), so the extension must be adding
the 424-row base; confirm that base is applied on the STORE path and not only on the load path,
since a missing base would write q 18-36 into rows 18-36 (inside row 0's span, where FA3 later
overwrites them) and leave 442-478 NaN -- which is exactly the observed shape.

**Sub-batch-base hypothesis REFUTED (candidate 12).** The extension offsets BOTH the load and the
store pointer by the same amount:
```
P.q       = static_cast<PDT*>(q.data_ptr())      + pers_row_offset * q.stride(0);       // :300
P.final_o = static_cast<PDT*>(pers_o.data_ptr()) + pers_row_offset * pers_o.stride(0);  // :323
```
so the FA2 sub-batch base is symmetric and "applied on load but not store" cannot happen. The store
itself is, for `num_kv_chunks == 1` (our case):
```
o_ptr_base = params.final_o + q_indptr * o_stride_n + (kv_head_idx * gqa_group_size) * o_stride_h;
write_o_reg_gmem<KTraits>(o_frag, &q_smem, o_ptr_base, qo_packed_idx_base, q_len,
                          o_stride_n, o_stride_h, gqa_group_size, tid);   // persistent.cuh:596-600
```
With `q_indptr = 0` (row 1 is the sub-batch's first request) and `pers_row_offset` supplying the
424-row base, the packed-128 tile should write q rows 18..36 -- absolute 442..460, inside the NaN
range. It executes (proved by the trace) and those rows stay NaN, so the remaining suspect is
**`write_o_reg_gmem`'s own row bounds**: it receives `qo_packed_idx_base` and `q_len` but NOT a
lower bound, so whether it writes `[qo_packed_idx_base/gqa, qo_upperbound)` or something
tile-relative is the thing to read. Note the comment at `:396` says tile-relative is the layout the
MERGE expects -- i.e. the partial_o path -- so a tile-relative index leaking into the
write-through path would write each tile at rows 0..18 of its own base instead of at its true
offset, which would leave the leading rows NaN exactly as observed and is the first thing to check
in that function.
Twelve candidates eliminated. The confirmed `MaskMode::kCausal` fix stands and is independent of
this.

**Candidate 13 refuted, AND a conflation in my own evidence corrected.**
`write_o_reg_gmem` (`prefill.cuh:1289-1302`):
```
group_size.divmod(o_packed_idx_base + lane_idx / 4 + mma_q * 16 + j * 8, q, r);
const uint32_t o_idx = q;
if (o_idx < qo_upper_bound) {
  *(o_ptr_base + q * o_stride_n + r * o_stride_h + ...) = ...;
}
```
`q` is an ABSOLUTE q index within the request (`o_packed_idx_base / gqa`), not tile-relative, and the
only guard is an upper bound. So the "tile-relative index leaks into the write-through path" idea is
wrong, and with `o_ptr_base = final_o + q_indptr*stride` (q_indptr 0, `pers_row_offset` supplying the
424 base) the packed-128 tile should write absolute rows 442-460. The store path is correct as
written.

**The conflation:** the `[TRACE]` printf is at the TOP of the work loop, before the mainloop. So it
proves a work item was ENTERED -- it does NOT prove the item reached its store. My statement that
"tiles 128 and 256 execute" was over-claimed; what is established is that they are entered. Every
hypothesis built on "they execute yet their rows are NaN" therefore rested on a weaker fact than I
stated, and the gap between entry and store is now the prime suspect: something between the two
skips the write for the leading tiles.
**Next:** move/add one printf immediately BEFORE the `write_o_reg_gmem` call (or at the end of the
loop body) and re-run with the q_len==87 filter. If the leading tiles appear at entry but not at the
store, the exit is inside the mainloop and its guard is the target; if they appear at both, the
store executes and the addressing is wrong after all. One run, and it cannot come out ambiguous.
Thirteen candidates eliminated. The `MaskMode::kCausal` fix is unaffected and stands.

## [x] SECOND FAULT IDENTIFIED: it is the FA2 split-KV MERGE path, not q tiles — 2026-10-09

The store probe (printf immediately before the write-through store, filtered `q_len == 87`) printed
**zero lines**: no work item of fixture 1's row 1 reaches the `num_kv_chunks == 1` branch at all.
Checking the trace already captured, every one of its 17 lines reads **`nchunks=2`**
(`kv_len=87`, `len_kv_chunk=64`). So that row takes the OTHER branch -- `partial_o` + merge:
```
if (num_kv_chunks > 1) { ...partial_o...  write_o_<KTraits>(...); }   // fixture 1 row 1 (nchunks=2)
else                   { ...final_o...    write_o_reg_gmem<KTraits>(...); }  // qo<=64 (nchunks=1)
```
**The two fixtures exercise different code paths, which I had conflated:**

| fixture | kv_len | chunks | path | result |
|---|---|---|---|---|
| `qo:424,64,1` | 64 | 1 | write-through to `final_o` | **PASSES** after the mask fix (0.00221) |
| `qo:424,87,1` (fixture 1) | 87 | 2 | `partial_o` + merge | 55 leading rows never written |

**This explains the qo sweep exactly, and corrects my reading of it.** The fixture sets
`kv = qo + prefix` with `prefix[1] = 0`, so qo and kv moved TOGETHER; the 0-unwritten-to-55
transition between qo1 64 and 87 is the point where `kv_len` crosses `len_kv_chunk = 64` and a
second KV chunk appears. It was never a q-tile-count boundary -- the "5th tile" arithmetic in the
earlier entries is retracted, as is "tiles 0/1/2 are not executed" (they are entered AND they store,
just to `partial_o`).
So the second fault is in the **FA2 side's split-KV merge**: partials are written but the merge does
not produce `final_o` for the leading rows. That is the same failure FAMILY as
[[fused-wgmma-splitkv-wrong]] (split rows overwriting each other), now on the persistent side.
**Next, and it is cheap and decisive:** re-run the sweep holding qo FIXED at 87 while varying only
the PREFIX so `kv_len` crosses 64 (e.g. `prefix[1]` = 0 vs a value giving kv < 64 and kv > 128).
If unwritten tracks `kv_len > 64` and not qo, the chunk-count diagnosis is confirmed outright and
the target is `write_o_` + the merge/reduction, not anything q-tile related.

### [x] MECHANISM of the second fault: the split merge skips rows served entirely by chunk 0

`bash /dev/shm/kvsweep.sh` -- qo held at 87, prefix varied so `kv_len` crosses `len_kv_chunk = 64`,
plus two single-chunk controls at different qo:

| qo1 | prefix | kv_len | chunks | unwritten | row 1 rel err |
|---|---|---|---|---|---|
| 87 | 0 | 87 | 2 | 55/512 | nan FAIL |
| 87 | 41 | 128 | 2 | **19**/512 | nan FAIL |
| 87 | 105 | 192 | 3 | **0**/512 | 0.51436 FAIL |
| 40 | 0 | 40 | 1 | 0/465 | **0.00152 ok** |
| 60 | 4 | 64 | 1 | 0/485 | **0.00269 ok** |

1. **Single chunk always PASSES** (0.00152, 0.00269) -- at two different qo, so the `MaskMode::kCausal`
   fix is solid and the write-through path is correct. Multi-chunk always FAILS.
2. **The unwritten count tracks the CAUSAL BOUND, not qo.** With `prefix = 105` every row's causal
   bound is >= 106 > 64, so every row needs a contribution from chunk 1 or later -- and NOTHING is
   unwritten. With `prefix = 41` the rows whose bound is <= 64 number ~23 (observed 19); with
   `prefix = 0`, ~64 (observed 55). The deficit shrinks exactly as the prefix pushes more rows past
   the first chunk boundary.
   => **The merge only produces `final_o` for rows that have partials from more than the first
   chunk; a row whose causal bound lies entirely inside chunk 0 is never merged and stays
   NaN-prefilled.** That is one mechanism fitting all five points, and it explains why fixture 1
   (prefix 0 on the task0 row) is the worst case.
3. **At 3 chunks every row is written but WRONG (0.51436).** So there is a second defect in the same
   merge -- a weighting/LSE error -- visible only once no row is skipped. Expect to fix both in
   `write_o_` + the reduction; do not assume one fix covers the other.

This is the family [[fused-wgmma-splitkv-wrong]] already records on the FA3 side ("split rows
overwrite each other"), now demonstrated on the FA2/persistent side with a different signature
(skipped rows rather than overwrites). Worth updating that memory once fixed.
Target for the fix: the `num_kv_chunks > 1` branch at `persistent.cuh:586-592` (`partial_o` +
`write_o_`) and whatever consumes `partial_lse` / `partial_o` to produce `final_o` -- find the
reduction's row loop and check its condition for rows with a single contributing chunk.

**Mechanism completed, and it names where the fix goes.** The merge is an in-kernel JIT reduction
driven by plan arrays: `P.merge_indptr` / `P.merge_o_indices`
(`fused_fa3_persistent_ext.cu:327-328`, with `FLASHINFER_NO_JIT_REDUCE=1` as the bisect switch at
:338). Combining that with the trace:
- Every row-1 work item reports `nchunks = 2`, **including the ones whose causal bound is far below
  the first chunk boundary** (`packed_qo_start=128` has `kv_end=37`). `num_kv_chunks` is a property
  of the REQUEST's kv length, not of the tile's causal extent.
- So those items take the `num_kv_chunks > 1` branch and write to `partial_o` -- but the chunk-1
  work item for those same rows has `kv_start = 64 > kv_end`, i.e. there is no work and no partial.
- The reduction then expects `num_kv_chunks` partials per row and finds only chunk 0's. Rows whose
  causal bound lies entirely in chunk 0 therefore never get a complete partial set, and `final_o`
  is never written for them -- exactly the NaN set, and exactly why a large prefix (every row
  needing chunk >= 1) makes the deficit vanish.
**Two fixes, both in this path, and they are separable:**
1. rows whose causal bound confines them to chunk 0 must either take the write-through path or have
   the reduction treat a missing later partial as absent rather than required (the skipped-row
   defect, 55/19/0 unwritten);
2. the 3-chunk case writes every row but at rel err 0.514, so the reduction's weighting is also
   wrong once it does run over several partials (the LSE/accumulation defect).
Start by reading how `merge_indptr`/`merge_o_indices` are built in the plan
(`scheduler.cuh`, `TwoStageHolisticPlan`) and whether the entry count per row matches the number of
chunks that actually produce work under causal masking -- that single comparison decides whether
the fix belongs in the plan or in the reduction.
`FLASHINFER_NO_JIT_REDUCE=1` is the ready-made A/B: if disabling the JIT reduction changes which
rows are NaN, the reduction owns it; if not, the plan does.

### [x] A/B settles ownership: the REDUCTION writes these rows, and it skips the chunk-0-only ones

`NO_JIT_REDUCE=1` (nulls `group_id`, disabling the JIT-reduction rendezvous; store probe removed,
cache cleared), fixture 1:

| config | unwritten | row 1 |
|---|---|---|
| reduction ON | 55/512 | nan FAIL |
| **reduction OFF** | **87/512** | nan FAIL |

87 is exactly row 1's full q length, so with the reduction disabled NONE of its rows are written.
By the rule stated before the run, the jump means **the reduction is what produces `final_o` for
this row** -- it writes 32 of the 87 and skips 55. The plan is exonerated: it is not that the rows
are missing from `merge_indptr`, it is that the reduction does not complete rows whose only partial
is chunk 0's. (FA3 0.00747 and task2 0.00075 are unaffected either way, as expected -- neither goes
through this path.)

**Confirmed diagnosis, end to end, for the second fault:**
`num_kv_chunks` is a per-REQUEST property, so every work item of a 2-chunk request takes the
`partial_o` branch -- including tiles whose causal bound sits entirely inside chunk 0. For those
rows the chunk-1 work item has `kv_start = 64 > kv_end` and writes no partial, so the reduction sees
an incomplete partial set and leaves `final_o` untouched. Rows whose bound reaches chunk 1 get a
complete set and are written. Hence 55 unwritten at prefix 0, 19 at prefix 41, 0 at prefix 105, and
87 with the reduction off.

**The fix, located:** in the reduction, a row with partials only from chunk 0 must still be
finalised (normalise and copy that single partial into `final_o`) rather than requiring every chunk
to have contributed. Separately, the 3-chunk case (every row written, rel err 0.51436) is a
weighting/LSE error in the same reduction and needs its own fix -- do not expect one change to cover
both. Fourteen candidates eliminated on the way here; the two real causes (`MaskMode::kNone`, and
this) were both found by reading code and existing diagnostics, not by inference.

### [x] FIX SITE AT LINE LEVEL: the merge's arrival counter waits for chunks that never run

`persistent.cuh:686-688`, inside the JIT reduction:
```
int last = 0;
if (ltid == 0) {
  last = (atomicAdd(&params.group_counter[gid], 1) == params.group_nchunks[gid] - 1);
}
... run_merge = (last broadcast over the warpgroup)
```
The merge fires only when a group's arrivals reach `group_nchunks[gid]`. Under causal masking a
chunk's work item can be absent entirely (`kv_start = 64 > kv_end` for a row whose bound sits in
chunk 0), so that group's counter never reaches `group_nchunks - 1`, `run_merge` never becomes true,
and `final_o` is never written for its rows. **This is the mechanism confirmed by all six
measurements**: 55 unwritten at prefix 0, 19 at prefix 41, 0 at prefix 105 (every row needs a later
chunk, so every group completes), and 87 with the reduction disabled.

**Qualification to my earlier "plan exonerated".** The `NO_JIT_REDUCE` A/B established that the
REDUCTION writes these rows, which rules out "the rows are missing from `merge_indptr`". But
`group_nchunks` is itself a PLAN array, so the root may still be plan-side: if the plan counts
chunks geometrically (`ceil(kv_len / len_kv_chunk)`) without accounting for causal masking, it
over-counts for exactly these groups. So the two candidate fixes are:
1. **plan-side (preferred if true):** make `group_nchunks[gid]` the number of work items actually
   EMITTED for that group, which is what the counter is compared against;
2. **kernel-side:** have the merge fire when the arrivals equal the number of chunks that produced
   partials, e.g. by having absent chunks still bump the counter, or by deriving the expected count
   from the causal bound.
Check (1) first: read how `group_nchunks` is filled in `TwoStageHolisticPlan` and compare it against
the emitted work items for a causally-confined group. `CopyToPageLockedBuffer(..., group_nchunks)`
near `scheduler.cuh:1893` is the write; the value's provenance is the thing to read.
Separately, the 3-chunk weighting error (rel err 0.51436, every row written) lives in the merge body
AFTER this gate and still needs its own fix.

### [!] RETRACTION OF A REFUTATION: `cluster_tile_q` WAS the right fix, tested on the wrong fixture

The kernel computes the chunk count causally (`persistent.cuh:372-378`):
```
num_kv_chunks = ceil_div(
    CAUSAL ? min((kv_len - q_len) + ceil_div(packed_qo_start + cluster_tile_q, gqa_group_size), kv_len)
           : kv_len,
    len_kv_chunk);
```
while the PLAN counts geometrically (`scheduler.cuh:1619,1631`):
`num_kv_tiles = ceil_div(remaining_len, kv_len_limit)` -> `group_nchunks.push_back(num_kv_tiles)`.
The kernel's expression depends on **`cluster_tile_q`** -- the value I showed is UNGUARDED under
`FUSED_GRID` (`gridDim.x * CTA_TILE_Q`, with gridDim.x the whole fused grid, ~130-260x too large).
That inflation saturates the `min()` at `kv_len`, so `num_kv_chunks` collapses to the geometric
maximum for EVERY tile -- which is precisely why the trace read `nchunks=2` even for the tile with
`kv_end=37`, and why causally-confined rows take the split path and then never complete the merge.

**So candidate 7 (`cluster_tile_q = (FUSED_GRID ? 1u : gridDim.x) * CTA_TILE_Q`) was correct, and my
refutation of it was invalid**: I tested it only on the `qo=64 / kv=64` fixture, which is
SINGLE-CHUNK, where `cluster_tile_q` cannot affect anything (the min saturates either way and
`ceil_div(<=64, 64) = 1`). Rel err stayed ~1.0 there because the MASK bug was still present -- the
mask fix came later. Testing a fix on a fixture that cannot exercise it, and reading "no change" as
refutation, is the same error class as the earlier `rel err = nan` masking (a metric that could not
respond to the hypothesis).
With `cluster_tile_q = CTA_TILE_Q = 128` the predicted behaviour on fixture 1's row 1 is:
- `packed_qo_start=128`: bound `ceil(256/7) = 37` -> `num_kv_chunks = 1` -> WRITE-THROUGH (written);
- `packed_qo_start=384`: bound `ceil(512/7) = 74` -> `num_kv_chunks = 2` -> split path (merged).
i.e. exactly the rows that are currently NaN would switch to the write-through path that the mask
fix already proved correct.
Re-applying it and testing on FIXTURE 1 this time. Predictions: unwritten 55 -> 0; row 1 rel err
from `nan` to ~1e-3; FA3 and task2 unchanged. If unwritten goes to 0 but rel err lands near 0.5,
the remaining error is the separate 3-chunk merge weighting defect, already recorded.

## [x] FUSED KERNEL PASSES THE CAUSAL fp32 REFERENCE END TO END — fixture 1 green, 2026-10-09

Two fixes together, both in the pod copy of flashinfer (`/opt/tiger/flashinfer` @ b0563ae + these):
1. `tests/fused_fa3_persistent_ext.cu:73` -- `KernelTraits<MaskMode::kNone, ...>` ->
   `MaskMode::kCausal` for `PKT` (all three persistent q-tile runners had NO causal mask);
2. `include/flashinfer/attention/persistent.cuh:327` --
   `cluster_tile_q = (KTraits::FUSED_GRID ? 1u : gridDim.x) * CTA_TILE_Q` (under FUSED_GRID
   `gridDim.x` is the whole fused grid, and the inflated value saturated the CAUSAL
   `num_kv_chunks` min() at `kv_len`, forcing every tile -- including causally-confined ones --
   onto the split path, where their merge never completed).

Command: `NVCC_THREADS=4 bash /dev/shm/kerneltest.sh` (default fixture 1,
`qo=[424,87,1] kv=[4001,87,14]`, packed_qo `[2968,609,7]`, GQA 28/4, H200 sm90a, nvcc 13.1,
torch 2.14.1+cu130, `-std=c++20` in both the harness `load()` and `flashinfer/jit/core.py`).
sglang `attn_fuse_` @ 405b4554b.

| sm_split | unwritten | FA3 (qo424 kv4001) | task0 (qo87 kv87) | task2 (qo1 kv14) |
|---|---|---|---|---|
| 48 | **0/512** | 0.00628 ok | **0.00250 ok** | 0.00071 ok |
| 32 | **0/512** | 0.00628 ok | **0.00250 ok** | 0.00071 ok |
| 16 | **0/512** | 0.00628 ok | **0.00250 ok** | 0.00071 ok |

Every prediction committed before the run held: unwritten 55 -> 0, task0 `nan` -> ~1e-3, FA3 and
task2 unchanged. The separate 3-chunk weighting error (rel err 0.514 at `prefix=105`) was predicted
as possibly remaining; it does NOT appear on fixture 1, so re-check it explicitly on the
`qo:424,87,1 prefix:3577,105,13` case before assuming it is gone.
**This clears the kernel-arm gate.** Fused timing is now meaningful for the first time: next is the
sm_split sweep for perf (`TIME_ITERS`), then the serving A/B with `mixed_fused_fa3=N(xx%)` and
fallback counts beside the numbers ([[fused-ab-validity-rule]]).
Cost of getting here: fourteen eliminated candidates, one invalid refutation (the `cluster_tile_q`
fix was right but tested on a single-chunk fixture that could not exercise it), and two retractions.
Both real causes came from reading source and from diagnostics the harness already had
(`DUMP_ROWS`, the `[TRACE]` print, `NO_JIT_REDUCE`).

### [x] FIRST VALID FUSED TIMINGS (kernel PASSES, so these are real) — 2026-10-09

`TIME_ITERS=50` on fixture 1 with both fixes; the harness prints its own verdict
**`PASS: every row correct at every split, nothing left unwritten`**.
Command: `FI_ROOT=/dev/shm/w/fi PYTHONPATH=/dev/shm/w/fi TORCH_CUDA_ARCH_LIST=9.0a NVCC_THREADS=4
TIME_ITERS=50 /dev/shm/w/venv_up/bin/python tests/test_fused_server_batch.py`.
sglang `attn_fuse_` @ dc1d2dbec, flashinfer @ b0563ae + the two fixes.

| sm_split (FA3 SMs) | kernel | unwritten | FA3 | task0 | task2 |
|---|---|---|---|---|---|
| **48** | **383.3 us** | 0/512 | 0.00746 | 0.00320 | 0.00098 |
| 32 | 505.1 us | 0/512 | 0.00746 | 0.00320 | 0.00098 |
| 16 | 877.5 us | 0/512 | 0.00746 | 0.00320 | 0.00098 |

**Monotonic in FA3's SM share** (16 -> 32 -> 48 gives 877 -> 505 -> 383 us, i.e. 2.29x from the
worst split to the best), so the optimum is probably ABOVE 48 and the harness's built-in sweep
(48/32/16) does not reach it. This fixture is FA3-heavy (one 424x4001 row against an 87-row and a
1-row FA2 request), so an FA3-favouring split being best is expected and does NOT generalise -- the
2-D boundary work above says the split should follow the decode population, and this batch has
almost none.
Next, in order: (1) extend the split sweep upward (64/80/96/112) to locate the optimum on this
fixture; (2) get the per-executor baselines printed in the same run (the harness has
`BASELINE persistent-only` and `TWO_EXEC` arms) so the fused number has something to beat --
a fused time alone is not a speedup; (3) only then the serving A/B, reporting
`mixed_fused_fa3=N(xx%)` and fallbacks beside TTFT/ITL ([[fused-ab-validity-rule]]).

### [x] Split sweep extended upward: the optimum was OUTSIDE the harness's built-in range

`SPLITS=112,96,80,64 TIME_ITERS=50`, fixture 1, both fixes, kernel PASSING at every split
(`PASS: every row correct at every split, nothing left unwritten`):

| sm_split (FA3 SMs, of 132) | kernel | vs best |
|---|---|---|
| **112** | **133.5 us** | 1.00x |
| 96 | 259.9 us | 1.95x |
| 80 | 257.5 us | 1.93x |
| 64 | 258.3 us | 1.93x |
| 48 | 383.3 us | 2.87x |
| 32 | 505.1 us | 3.78x |
| 16 | 877.5 us | 6.57x |

Two things worth keeping:
1. **The harness's default sweep (48/32/16) never reached the optimum.** 112 is 2.87x faster than 48,
   the best of the default range, so every earlier timing taken with the built-in list understated
   the fused kernel on this shape. Any historical fused-vs-baseline number in this file that used
   the default splits should be re-read with that in mind.
2. **The curve is not monotone-smooth: 64/80/96 form a ~258 us plateau, then 112 drops to 133.5.**
   A flat region followed by a cliff suggests a discrete effect (occupancy / CTA-per-SM or the FA2
   side fitting in the leftover 20 SMs) rather than a smooth work-balance tradeoff. Worth one probe
   before modelling the boundary: sweep 100..128 in steps of 4 to find where the cliff is, since the
   2-D boundary fit above will need the right functional form, not just the argmin.
Caveat on generalisation, repeated because it matters for the serving A/B: fixture 1 is FA3-heavy
(one 424x4001 row against an 87-row and a 1-row FA2 request), so a split leaving only 20 SMs for FA2
being optimal is specific to this shape. The decode-population guard already in the serving dispatch
is the mechanism that should pick the split per batch; this number is not a serving recommendation.

**No baseline yet, so NO speedup can be claimed.** The `TWO_EXEC=1` comparison arm fails to run in
this build:
```
File ".../tests/test_fused_server_batch.py", line 525, in _run_split
  torch.ops.fused_fa3.fa3_only_run(*_args3)
AttributeError: '_OpNamespace' 'fused_fa3' object has no attribute 'fa3_only_run'
```
The extension compiled from `fused_fa3_persistent_ext.cu` does not export `fa3_only_run`, so the
two-executor arm (FA3 alone on its SM partition, the per-executor baseline) cannot be measured here.
That means every number in the table above is a fused ABSOLUTE time with nothing to divide by --
**the 133.5 us at sm_split 112 is not a speedup claim**, and per [[fused-ab-validity-rule]] it must
not be quoted as one.
To fix, next session: find what registers `fa3_only_run` (a second TORCH_LIBRARY_FRAGMENT /
`m.def` in the ext, possibly behind a `-D`, or a separate .cu the harness expects to be loaded
alongside). `grep -n "fa3_only_run\|TORCH_LIBRARY" tests/fused_fa3_persistent_ext.cu` is the first
read. Until that arm runs, the per-executor baselines the goal asks for ("parity with each
standalone SOTA kernel per role") are unmeasurable on this pod.

**The per-executor baseline needs a THIRD tree on the pod: flash-attention.**
`fa3_only_run` is registered only under `#if FLASHINFER_FA3_UPSTREAM` (`fused_fa3_persistent_ext.cu:506-510`),
which the harness gates on `FA3_UPSTREAM=1` (`test_fused_server_batch.py:295`). Building with it set
fails:
```
include/flashinfer/attention/hopper/fa3_upstream.cuh:31:10: fatal error:
  ../../../../../flash-attention/hopper/utils.h: No such file or directory
```
That path resolves to a **sibling of the flashinfer root** (`<fi_parent>/flash-attention/hopper/`),
i.e. outside the `flashinfer/ csrc/ include/ tests/` set I pushed, and outside `3rdparty/` which was
cloned. Local source `/opt/tiger/flash-attention/hopper` is 3.2 M, so it pushes like the others.
Pushing it to `/dev/shm/w/flash-attention` now (the include is relative to the fi root's parent,
which is `/dev/shm/w`, so that is the correct destination).
**Restore recipe for this pod is therefore FIVE steps, not four** -- add: push
`flash-attention/hopper` as a sibling of `fi`. Without it the fused kernel builds and passes, but
the per-executor FA3 baseline the goal asks for ("parity with each standalone SOTA kernel per role")
cannot be built at all, so no speedup is computable.
Until that lands: the kernel PASSES and the best measured fused time is 133.5 us at sm_split 112,
but **no gain is demonstrated** and none should be quoted.

**The FA3 per-executor baseline is blocked on the FA3_UPSTREAM path, which does not compile.**
With `flash-attention/hopper` now present (the previous blocker, resolved -- 249 KB pushed to
`/dev/shm/w/flash-attention`), `FA3_UPSTREAM=1` gets further and then fails on a params mismatch:
```
persistent_fa3_fused.cuh(815,818,819): error: class
  "flashinfer::BatchPrefillPagedParams<FDT, FDT, FDT, IdT>" has no member "wgmma_plan"
```
So the upstream-FA3 (wgmma) route expects an extended params struct this build does not supply.
That is the same area [[fused-wgmma-splitkv-wrong]] records as broken and blocking ("split FA3 rows
overwrite each other ... blocks all unified work"), so this is a known-incomplete path rather than a
fresh regression -- and `fa3_only_run` being gated behind it (`#if FLASHINFER_FA3_UPSTREAM`) means
the FA3-side per-executor baseline cannot be built on this pod without finishing that port.
**Consequences, stated plainly:**
- the DEFAULT build (no `FA3_UPSTREAM`) still PASSES and still times -- 133.5 us at sm_split 112 --
  so the kernel result stands;
- but there is still **no baseline to divide by, so no speedup is demonstrated** and none should be
  quoted. The goal's "parity with each standalone SOTA kernel per role" is unmeasurable until either
  the wgmma params plumbing is finished or another baseline is used.
- The available alternative is the harness's **`BASELINE persistent-only`** arm (seen at
  `test_fused_server_batch.py` ~line 657, `[time] BASELINE persistent-only iters=...`), which
  measures the FA2 side alone and does NOT appear to need `FA3_UPSTREAM`. Find its guard and run it:
  that gives the FA2-role baseline, which is half of the parity question and is reachable now.

## [x] FIRST VALIDATED KERNEL SPEEDUP: fused 1.30x over the unfused persistent kernel — 2026-10-09

`TIME_BASELINE=1` (which does NOT need `FA3_UPSTREAM`, unlike the blocked `TWO_EXEC` arm) gives a
like-for-like baseline. The harness's own comment establishes the comparison is valid:
*"the baseline needs its own object planned WITHOUT `FLASHINFER_FA3_TASK` over the same rows --
which is exactly the unfused persistent kernel on identical work."*

| arm | kernel time | correctness |
|---|---|---|
| **fused, sm_split 112** | **133.5 us** | PASS (0 unwritten, FA3 0.0075, task0 0.0032, task2 0.0010) |
| unfused persistent (baseline) | **174.0 us** | n/a (reference arm) |
| fused, sm_split 96 | 260.9 us | PASS |
| baseline (same run) | 176.7 us | n/a |
| fused, sm_split 64 | 257.9 us | PASS |
| baseline (same run) | 173.1 us | n/a |

**Fused at its optimum split is 1.30x the unfused persistent kernel on identical work**
(174.0 / 133.5), i.e. 23% less kernel time, and the harness prints
`PASS: every row correct at every split, nothing left unwritten`. The baseline is stable at
173-177 us across the three runs, which is the consistency check that it is measuring the same
thing each time.
Command: `FI_ROOT=/dev/shm/w/fi PYTHONPATH=/dev/shm/w/fi TORCH_CUDA_ARCH_LIST=9.0a NVCC_THREADS=4
TIME_ITERS=50 TIME_BASELINE=1 SPLITS=112,96,64 /dev/shm/w/venv_up/bin/python
tests/test_fused_server_batch.py`. sglang `attn_fuse_` @ 11066774c; flashinfer
`/opt/tiger/flashinfer` @ b0563ae + the two fixes (`MaskMode::kCausal`, FUSED_GRID `cluster_tile_q`).

**What this is and is NOT.** It IS a validated kernel-level gain on a correct kernel, which is the
thing this file has been unable to produce all session. It is **NOT** the goal's acceptance bar:
that asks for >=30% TTFT & ITL and >=10% throughput on a realistic serving workload, and a 1.30x
attention-kernel time on one fixture does not imply either -- attention is a fraction of step time,
and this fixture (one 424x4001 row + an 87-row + a 1-row) is FA3-heavy and not the mixed
chat+tracelab shape. The serving A/B remains the only thing that can settle the bar.
Also still missing: the FA3-ROLE per-executor baseline (blocked on the wgmma `params.wgmma_plan`
plumbing), so "parity with each standalone SOTA kernel per role" is half-answered -- this is the
FA2-role comparison only.

## The serving A/B: two paths, two different blockers — DECISION NEEDED — 2026-10-09

The kernel gate is cleared (PASS, and 1.30x over the unfused persistent kernel at sm_split 112), so
the only thing between here and the goal's >=30% TTFT/ITL + >=10% throughput bar is a serving A/B.
There are exactly two ways to run one on this pod, and each is blocked differently:

**Path A -- fork + split kernel (the arm the goal describes).** `/dev/shm/w/python` (fork tree) with
`/dev/shm/w/venv`. BLOCKED: `import torch` fails with
`libtorch_cuda.so: undefined symbol: ncclCommResume`. Diagnosed further this round -- the venv has
BOTH `nvidia-nccl-cu12 2.27.5` and `nvidia-nccl-cu13 2.30.7`, `nvidia/nccl/lib/libnccl.so.2` exists,
and pointing `LD_LIBRARY_PATH` at it does NOT help, so it is a symbol-level version mismatch rather
than a lookup problem. pip shows NO `torch` in that venv (it is a direct copy), so the matching
nccl version cannot be read off the environment. Repair = identify the fork torch build and install
the nccl it was linked against. This path has the tracelab dataset and `--dataset-name
tracelab-claude`, i.e. the realistic workload the bar names.

**Path B -- upstream 4-mode (venv_up).** Works and has produced the only valid serving numbers so
far (the baseline arm). BLOCKED differently: the 4-mode ON arm fires the mixed path
(`mixed_spec_steps: 1`) and then crashes in
`cuda_graph_buffer_registry.fill_from -> triton _small_copy_kernel` under the 8192-token
shared-prefix workload, with no shape mismatch. Does NOT reproduce in `portcorr3` at NBG=12 /
ver_rows 7, so the remaining variable is the shared-prefix cache-hit shape. Upstream lacks the
tracelab dataset, so its realistic arm must use `generated-shared-prefix`.

**Recommendation, stated so the next session does not re-derive it:** fix Path A. The goal's bar is
about the fused kernel on a realistic serving workload, and Path A is the only one that exercises
the split kernel at all -- Path B measures the 4-mode scheduling change with upstream's own
attention. Path A's blocker is a dependency version to pin, which is bounded work; Path B's is an
unreproduced crash in a code path we do not control. Do A, keep B as the scheduling-only arm.

### [x] PATH A UNBLOCKED: the fork venv's torch now imports — it was an nccl library mismatch

Diagnosis chain (each step a read or a one-line check, no guessing):
1. The fork venv's torch is **2.14.1+cu130** -- the SAME build as `venv_up`, so this was never an
   "old fork torch" problem.
2. `nm -D --defined-only .../nvidia/nccl/lib/libnccl.so.2 | grep -c ncclCommResume`: **0** in the
   fork venv, **2** in `venv_up`. The fork venv has both `nvidia-nccl-cu12 2.27.5` and
   `nvidia-nccl-cu13 2.30.7` installed to the SAME path, and the cu12 build (which predates that
   symbol) had won.
3. `LD_LIBRARY_PATH` pointing at `venv_up`'s copy does NOT help -- torch resolves nccl through its
   own RPATH, so the bundled file must be replaced, not shadowed.
Fix applied on the pod: backed up `libnccl.so.2` as `libnccl.so.2.bak` and copied `venv_up`'s cu13
build over it. Result: `torch OK 2.14.1+cu130 cuda_avail= True`.
```
cd /dev/shm/w/venv/lib/python3.12/site-packages/nvidia/nccl/lib
cp -a libnccl.so.2 libnccl.so.2.bak
cp -f /dev/shm/w/venv_up/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2 libnccl.so.2
```
Add this to the pod restore recipe (now six steps). Note for later: the clean fix is
`pip uninstall nvidia-nccl-cu12` in that venv rather than a file copy, but the copy is verified
working and does not touch the resolver.
**This opens the serving A/B on the fork**, which is the arm that exercises the split kernel and has
the tracelab dataset -- the only path that can address the goal's >=30% TTFT/ITL + >=10% throughput
bar on a realistic workload. Next: launch the fork server with the split kernel enabled
(`SGLANG_TWO_EXEC` / `SGLANG_FUSED_FA3_SM_SPLIT`), confirm `mixed_fused_fa3=N(xx%)` in the log and
the fallback count ([[fused-ab-validity-rule]]), run `drivers/srv_correctness.py`, then the
chat+tracelab A/B against the same build with the split off.

### [x] PATH A FULLY UNBLOCKED: fork sglang + fused flashinfer both import — 2026-10-09

Second dependency fault after the nccl one: the fork venv had **torchvision 0.24.1+cu128** against
torch 2.14.1, which fails as
`AttributeError: partially initialized module 'torchvision' has no attribute 'extension'
(most likely due to a circular import)`. `venv_up` has the compatible 0.29.1, and the pod has
network, so: `/dev/shm/w/venv/bin/pip install --no-deps "torchvision==0.29.1"`.
Verified afterwards:
```
PYTHONPATH=/dev/shm/w/python:/dev/shm/w/fi /dev/shm/w/venv/bin/python -c 'import sglang, flashinfer'
-> sglang 0.0.0.dev0 from /dev/shm/w/python/sglang
-> flashinfer /dev/shm/w/fi/flashinfer
```
**The import paths are the point of that check, not a formality**: it confirms the FORK's sglang and
OUR fused flashinfer are what load, which is exactly the failure [[pod-tree-verify-manifest]]
records from the earlier restore (an upstream sglang silently on PYTHONPATH). Verify this before any
serving number, every time.
Pod restore recipe is now **seven** steps: clone cutlass+spdlog -> push fork sources -> push
`flash-attention/hopper` as a sibling of `fi` -> repoint `flashinfer/data/*` symlinks -> warm the
BatchAttention JIT -> swap `libnccl.so.2` to the cu13 build -> install `torchvision==0.29.1`.
Next, in order, and all of it now reachable: (1) launch the fork server with the split kernel
(`SGLANG_TWO_EXEC=1`, `SGLANG_FUSED_FA3_SM_SPLIT`, split 112 from the sweep above) and CONFIRM
`mixed_fused_fa3=N(xx%)` plus the fallback count in the log; (2) `drivers/srv_correctness.py`;
(3) the chat+tracelab A/B, split ON vs OFF in the same build, reporting TTFT/ITL/throughput beside
the composition counter and fallbacks. That sequence is what the >=30%/>=10% bar needs.

### Serving A/B blocked again, on a flashinfer/sglang VERSION SKEW — 2026-10-09

`srv_two_exec_ab.sh` runs (it writes per-arm server logs to `/dev/shm/w/srv_<arm>.log` and clears
`~/.cache/flashinfer/90a/cached_ops` per arm, so each arm rebuilds the JIT -- that is why it looks
silent for minutes). The baseline arm's server then dies at import:
```
sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxint4_moe.py:39
  from flashinfer.fused_moe import (
ImportError: cannot import name 'trtllm_mxint4_block_scale_moe' from 'flashinfer.fused_moe'
  (/dev/shm/w/fi/flashinfer/fused_moe/__init__.py)
```
**The fork's sglang tree on the pod expects a NEWER flashinfer than `/opt/tiger/flashinfer @
b0563ae`, which is what I pushed when restoring `fi`.** So the restore reconstructed a *working
fused-kernel* flashinfer (it builds, passes the fp32 reference, and gives the 1.30x) but not the
version this sglang tree was developed against. The pod's original `fi` must have been newer.
Three ways out, in order of preference:
1. **Patch the import** -- the symbol is used by a w4a4-mxint4 MoE quantization scheme that this
   Qwen2.5-7B bf16 run cannot need. If `schemes/__init__.py` imports it eagerly, guarding that one
   import restores the serving path without touching the kernel. Smallest change, and reversible.
2. Upgrade the pushed flashinfer to a revision that has `trtllm_mxint4_block_scale_moe` -- but that
   risks losing the fused-kernel work in `/opt/tiger/flashinfer`, which is local-only and is the
   whole point; do NOT do this without first checking whether that revision contains
   `tests/fused_fa3_persistent_ext.cu`.
3. Run the serving A/B on upstream instead -- but upstream does not have the split kernel, so it
   cannot answer the kernel question at all.
Do (1). And record the general lesson: restoring a tree from a LOCAL copy can satisfy the build and
still mismatch the consumer; the manifest check in [[pod-tree-verify-manifest]] should compare
VERSIONS of the two trees against each other, not just confirm both import.

**Option (1) is NOT viable: the skew is systemic.** Guarding the first missing symbol
(`trtllm_mxint4_block_scale_moe`) immediately exposed a second from a different module --
`ImportError: cannot import name 'fp8_blockscale_gemm_sm90' from 'flashinfer.gemm'` -- so the fork's
sglang expects a materially newer flashinfer across several subpackages, not one optional scheme.
Guarding them individually would be whack-a-mole and would leave the serving arm running against a
patched tree of unknown fidelity, which is worse than no number.
**Revised plan, and the likely shape of the pod's original setup:** `/dev/shm/w/fi` was probably a
NEWER flashinfer (pip-installed or a newer checkout) with our fused-kernel sources overlaid in
`tests/` -- that reconciles "sglang needs new flashinfer APIs" with "the fused kernel builds from
`tests/fused_fa3_persistent_ext.cu`". My restore pushed only the older local tree, satisfying the
kernel build while breaking the sglang consumer.
So: install a flashinfer release new enough for the fork's sglang INTO the fork venv (the pod has
network), and keep `/dev/shm/w/fi` as the `FI_ROOT` the kernel harness builds from -- the two roles
are separate (`PYTHONPATH` for sglang's imports vs `FI_ROOT`/`extra_include_paths` for nvcc). Verify
after: `import sglang` succeeds, AND the kernel test still PASSES from `FI_ROOT`, AND the server log
shows `mixed_fused_fa3=N(xx%)`.
What is NOT affected by any of this: the kernel results above were produced entirely through the
harness with `FI_ROOT=/dev/shm/w/fi` and no sglang involvement, so the PASS and the 1.30x stand
regardless of how the serving import path is resolved.

### [x] RETRACTION + FIX: the "flashinfer/sglang version skew" was MY OWN PYTHONPATH shadowing

`pip list` in BOTH venvs shows **`flashinfer-python 0.7.0.post1`**, and that pip copy HAS both
symbols (`hasattr(flashinfer.gemm, "fp8_blockscale_gemm_sm90") == True`,
`hasattr(flashinfer.fused_moe, "trtllm_mxint4_block_scale_moe") == True`). So there is **no version
skew in the project**. The pod's topology is:
- **pip-installed flashinfer 0.7.0.post1** -> what sglang imports at runtime;
- **`/dev/shm/w/fi`** -> `FI_ROOT` for the KERNEL build only (nvcc include paths and
  `SGLANG_FUSED_FA3_SRC`), never on `PYTHONPATH`.
The driver had `PYTHONPATH=/dev/shm/w/python:/dev/shm/w/fi`, so my restored (older) tree SHADOWED the
pip install and sglang lost APIs that were present all along. Fixes: reverted the MoE import guard
(it treated a symptom), and dropped `/dev/shm/w/fi` from `PYTHONPATH` in both arms of
`srv_two_exec_ab.sh`. **The previous entry's "upgrade flashinfer" plan is RETRACTED** -- following it
would have replaced a working pip install and risked the local-only fused-kernel tree for no reason.
Verified after the fix:
```
PYTHONPATH=/dev/shm/w/python python -c 'import sglang, flashinfer'
-> sglang      /dev/shm/w/python/sglang            (fork tree)
-> flashinfer  .../venv/lib/python3.12/site-packages/flashinfer   (pip 0.7.0.post1)
```
and the fork server now starts: `Application startup complete`, `Uvicorn running on
http://127.0.0.1:31000`, `GET /model_info 200 OK` (`/dev/shm/w/srv_base.log`).
**Lesson, sharper than the earlier manifest note:** when a tree serves two roles (import path vs
build root), restoring it into the IMPORT path can break a consumer that was happily using a
packaged copy. Check which flashinfer `import flashinfer` actually resolves to before blaming
versions -- `os.path.dirname(flashinfer.__file__)` is the one-line check that would have saved this
detour.

### [!] First fork serving A/B ran, and is INVALID: `two_exec_calls: 0` in BOTH arms — 2026-10-09

`IN=32768 OUT=256 NP=16 CONC=8 bash /dev/shm/srv_two_exec_ab.sh` (fork build, split kernel,
PYTHONPATH fixed). Both arms completed cleanly, 0 fallbacks:

| metric | base (TWO_EXEC=0) | split (TWO_EXEC=1) |
|---|---|---|
| req/s | 0.49 | 0.50 |
| output tok/s | 126.46 | 127.33 |
| total tok/s | 16313.36 | 16425.38 |
| mean TTFT | 5582.83 ms | 5520.26 ms |
| median TTFT | 3543.62 ms | 3442.09 ms |
| mean TPOT / ITL | 40.84 ms | 40.65 ms |
| median ITL | 8.55 ms | 8.55 ms |
| **`two_exec_calls`** | **0** | **0** |

**`two_exec_calls: 0` in the SPLIT arm means the two-executor kernel never ran**, so this measures
the same path twice and the ~1% deltas are noise. Per [[fused-ab-validity-rule]] the numbers are
void as a fused result -- recorded here only so they are not mistaken for one. (They are, however, a
valid long-context BASELINE for this shape: 32768-token input, 256 out, 16 prompts, conc 8.)

**Prime suspect is the decode-population guard I added THIS SESSION** to the fork's
`mixed_dispatch.split_classes` (validated 11/25 -> 23/25 against `boundary_28x4x128.json`). Its rule
is `kv_tiles <= decode_rows / 2`, and with `IN=32768` a prefill row has `kv_tiles = 256`, needing
>= 512 co-resident decode rows -- never true in this workload. So the guard would reject FA3
dispatch for every row here and silently collapse the split arm onto the FA2 path.
That is the guard behaving exactly as the measured surface says it should (fused LOSES with few
decode rows), which makes this a WORKLOAD/arm-design problem, not a guard bug -- but it means the
split kernel cannot be exercised by a long-prefill, low-decode benchmark at all.
**Two corrections for the next run, in order:**
1. re-run with `SGLANG_FUSED_DECODE_GUARD=0` to confirm the guard is the suppressor (one run, and
   `two_exec_calls` > 0 settles it);
2. then measure on a workload the guard admits -- i.e. with a real decode population alongside the
   prefills (the mixed chat+tracelab shape the goal names, or `srv_two_exec_ab.sh` with a background
   decode stream). A split-kernel A/B on a batch with ~no decode rows was never going to show a
   fused gain even if the kernel were perfect.

**Guard hypothesis REFUTED: `SGLANG_FUSED_DECODE_GUARD=0` changes nothing.** Re-ran the same A/B
with the guard disabled; the split arm STILL reports `two_exec_calls: 0`:

| metric | base (guard off) | split (guard off) |
|---|---|---|
| output tok/s | 127.88 | 127.41 |
| total tok/s | 16496.44 | 16435.84 |
| mean TTFT | 5429.82 ms | 5496.37 ms |
| mean TPOT / ITL | 40.74 ms | 40.71 ms |
| **`two_exec_calls`** | **0** | **0** |

So the decode-population guard I added this session is NOT what suppresses the split kernel, and the
previous entry's "prime suspect" is withdrawn. Something else prevents `SGLANG_TWO_EXEC=1` from
reaching the two-executor path, and all four arms measured so far (two runs x two arms) are the same
code path -- consistent to ~1%, which is itself the tell.
**Next check, and it must be a read before another run:** find where `two_exec_calls` is incremented
and what gates it. `grep -rn "two_exec_calls" python/sglang/srt/layers/attention/flashinfer_backend.py`
plus the conditions around `SGLANG_TWO_EXEC` at `flashinfer_backend.py:1358,1433` (seen earlier).
Candidates to check in that order: (a) the counter is incremented in a branch the server never
reaches (so fusion may be running but uncounted -- check a second signal such as
`mixed_fused_fa3=`); (b) the two-exec path additionally requires `SGLANG_FUSED_FA3_SM_SPLIT` to be
set, which this driver does NOT export (it exports `SGLANG_TWO_EXEC`,
`SGLANG_FLASHINFER_USE_PERSISTENT`, `SGLANG_MIXED_ATTENTION_KERNEL=persistent`,
`SGLANG_FUSED_FA3_SRC`); (c) the per-row dispatch rule (`packed_qo >= 48`, `MIN_QO_LEN`) excludes
every row in this shape.
(b) is the cheapest to test and the most likely, given the sweep above showed the SM split is a
first-class parameter the kernel needs -- an unset split may mean "no split", i.e. no two-executor
work by construction.

**CORRECTION to the two A/B verdicts above: the arms are UNVERIFIABLE, not proven invalid.**
The driver's counter is `two_exec_calls: $(grep -co 'two_exec' "$LOG")`
(`srv_two_exec_ab.sh:63`) -- it counts occurrences of the literal string `two_exec` in the SERVER
LOG, not kernel invocations. And `two_exec_calls` appears nowhere in
`flashinfer_backend.py`, so nothing in the split path necessarily emits it. A zero therefore means
"the server never logged that string", which is NOT evidence that the split kernel did not run.
I checked the real signal this project uses ([[fused-ab-validity-rule]]): `mixed_fused_fa3=N(xx%)`
is absent from BOTH server logs, as is any `two_exec` mention. So there is no composition evidence
in either direction.
Net: **all four arms (2 runs x 2 arms) are unusable** -- not because fusion provably did not run,
but because nothing in the run establishes whether it did. The ~1% spread across arms is consistent
with either explanation. My "the split kernel never ran" and "the decode guard is the prime suspect"
statements both overstated the evidence; the guard refutation stands (disabling it changed nothing),
but it was answering a question the data could not pose.
**What a usable serving arm needs, concretely:**
1. a composition signal that is actually emitted -- find what the fork's split path logs (or add an
   always-on counter, as was done for `MIXEDSPEC_STEPS` on the upstream side, which worked on its
   first use);
2. then the arms, on a workload with a real decode population (the guard refuses pure long-prefill
   batches BY DESIGN per the measured boundary, so chat+tracelab rather than IN=32768 alone);
3. `drivers/srv_correctness.py` beside it, per the standing rule that every serving arm passes it
   and reports fallbacks.
Until (1) exists, no serving number from this driver can be quoted for or against the >=30%/>=10%
bar. The long-context baseline numbers remain usable as a baseline only
(32768 in/256 out/16 prompts/conc 8: ~127 tok/s, mean TTFT ~5.5 s, mean ITL ~40.7 ms, 0 fallbacks).

**Settled by logic, not by a proxy: the two-executor branch was NEVER ENTERED in the split arm.**
`flashinfer_backend.py:~1452`:
```
_two_exec_default = os.environ.get("SGLANG_FUSED_KERNEL") != "1"
if os.environ.get("SGLANG_TWO_EXEC", "1" if _two_exec_default else "0") == "1":
    ...
    if not hasattr(torch.ops.fused_fa3, "fa3_only_run"):
        raise ...("SGLANG_TWO_EXEC=1 but torch.ops.fused_fa3.fa3_only_run is absent: ...")
```
We know independently that `fa3_only_run` is NOT in the built extension (it is gated behind
`#if FLASHINFER_FA3_UPSTREAM`, and that path fails to compile on `params.wgmma_plan`). So **if the
split arm had entered this branch it would have raised** -- and the driver deliberately
`unset SGLANG_FUSED_ALLOW_FALLBACK` so the raise could not be downgraded. The split arm's server log
contains ZERO occurrences of `fa3_only_run` and the run completed normally. Therefore the branch was
never reached, and both arms ran the ordinary attention path.
So the ORIGINAL verdict ("the split kernel never ran") was right, the "unverifiable" correction was
over-cautious, and the reason is neither the decode guard nor the counter: **the server never got as
far as the two-executor dispatch at all.** Corroborating detail: the split server log mentions
`persistent|flashinfer` on only ONE line, far fewer than a server actually driving that backend
would produce.
**So the serving blocker reduces to the SAME root cause as the missing per-executor baseline:**
`fa3_only_run` requires `FLASHINFER_FA3_UPSTREAM`, which does not compile
(`BatchPrefillPagedParams has no member "wgmma_plan"`). Finishing that wgmma params plumbing is
therefore on the critical path for BOTH the kernel-parity baseline AND any fork serving A/B of the
split kernel -- it is one blocker, not two, and it is the next thing to fix.
Everything else needed is in place and verified: fused kernel PASSES, optimum split known
(112 SMs, 133.5 us, 1.30x over unfused persistent), fork server boots with the fork tree and the
pip flashinfer, tracelab dataset present, drivers staged.

### The single blocker, located to the type level: the extension passes the wrong params type

`persistent_fa3_fused.cuh:815-821` (the FA3_UPSTREAM role) reads:
```
up.page_table   = params.wgmma_plan.dense_page_table;
up.cu_seqlens_q = params.wgmma_plan.q_indptr_per_req;   // PER-REQUEST, not per work item
up.seqused_k    = params.wgmma_plan.kv_lens_per_req;
up.total_q      = params.wgmma_plan.total_q;
up.num_pages    = params.wgmma_plan.num_pages;
```
and the member comes from a wrapper that ALREADY EXISTS
(`unified_wgmma_scheduler.cuh:186,274-277`):
```
template <typename Base, typename IdType>
struct UnifiedWgmmaParams : Base { UnifiedWgmmaPlanPtrs<IdType> wgmma_plan; };
```
So the FA3_UPSTREAM path wants `UnifiedWgmmaParams<BatchPrefillPagedParams<...>, IdType>`, and
`tests/fused_fa3_persistent_ext.cu` passes the plain `BatchPrefillPagedParams` -- hence
`has no member "wgmma_plan"`. **The type exists; the extension does not use it and never populates
the plan.** That is the whole blocker, and it gates BOTH the per-executor FA3 baseline
(`fa3_only_run` is behind the same `#if`) and any fork serving A/B of the split kernel.
Fix shape, for whoever picks this up: in the extension, (1) declare the FA3 role's params as
`UnifiedWgmmaParams<...>` under `#if FLASHINFER_FA3_UPSTREAM`, and (2) fill `wgmma_plan`'s five
pointers from the plan -- noting the comment in the kernel that `q_indptr_per_req` and
`kv_lens_per_req` must be PER-REQUEST while the persistent plan's `q_indptr`/`kv_len` are per WORK
ITEM. That per-request-vs-per-item distinction is exactly the family of four port bugs recorded in
[[upstream-fa3-subset-params]], so build those two arrays explicitly rather than reusing the
persistent plan's.
Everything else is ready: fused kernel PASSES the fp32 reference, optimum split known
(112 SMs -> 133.5 us, 1.30x over unfused persistent), fork server boots with the right trees,
tracelab dataset present, seven-step pod restore documented, and an always-on composition counter is
the one piece of instrumentation to add alongside (the `MIXEDSPEC_STEPS` pattern worked first time).

**The blocker is a GUARD MISMATCH, and there are two one-line fixes.**
Extension (`fused_fa3_persistent_ext.cu:173-179`):
```
#ifdef FLASHINFER_FUSED_UNIFIED_WGMMA
  using FA3ParamsSel = UnifiedWgmmaParams<FA3Params, IdT>;
  constexpr bool kUnifiedWgmma = true;
#else
  using FA3ParamsSel = FA3Params;          // <-- no wgmma_plan member
  constexpr bool kUnifiedWgmma = false;
#endif
```
Kernel: the `wgmma_plan` reads live at `persistent_fa3_fused.cuh:815-821`, while the
`if constexpr (kUnifiedWgmma)` guard is at `:692` -- and there is already a comment at `:932`
noting that "a plain `if` made the non-unified instantiation" fail, i.e. this exact trap has been
hit before in this file. So with `FA3_UPSTREAM=1` and `FUSED_UNIFIED_WGMMA` undefined, the 815 block
is instantiated against params that have no `wgmma_plan`.
**Two fixes, and the choice matters:**
1. **also define `FLASHINFER_FUSED_UNIFIED_WGMMA`** -- compiles, but routes FA3 through the unified
   wgmma scheduler, which [[fused-wgmma-splitkv-wrong]] records as NUMERICALLY WRONG ("split FA3
   rows overwrite each other"). Any baseline measured this way would be invalid, so this is only
   acceptable as a compile check, never as a measurement.
2. **move the 815-821 block inside `if constexpr (kUnifiedWgmma)`** (or guard it equivalently) so
   the non-unified instantiation never sees `wgmma_plan`. This is the correct fix: it makes
   `FA3_UPSTREAM=1` buildable WITHOUT enabling the broken unified path, which is exactly what the
   per-executor FA3 baseline and the fork serving A/B need.
Do (2). It is a guard placement, not a port -- materially smaller than the "finish the wgmma params
plumbing" framing I used in the previous entries, which is hereby narrowed: nothing needs to be
populated for the non-unified build, because that build never reads the field.

**STOP before the "one-line guard" fix: it would silently leave five fields unset.**
The `wgmma_plan` reads sit inside `#if FLASHINFER_FA3_UPSTREAM` in the HOST function
`FusedFA3PersistentDispatched` (`persistent_fa3_fused.cuh:610`, block at 801-821), and they are not
a diagnostic -- they POPULATE the upstream FA3 launch descriptor:
`up.page_table`, `up.cu_seqlens_q`, `up.seqused_k`, `up.total_q`, `up.num_pages`.
Wrapping them in `if constexpr (kUnifiedWgmma)` would compile the non-unified build but leave those
five fields at their default, i.e. the FA3 role would launch with no page table and no sequence
bounds. That is a silent wrong-answer (or illegal-access) path, not a fix -- precisely the class of
"make it build by disabling the feature" change the project's standards forbid.
**So the real question, and it is a design question rather than a mechanical one:** where should the
non-unified FA3_UPSTREAM path get its per-request page table / cu_seqlens_q / seqused_k / total_q /
num_pages from? Two candidates visible from here:
- the `UpstreamFA3Plan shape_only{}` object already constructed in the extension
  (`fused_fa3_persistent_ext.cu:234`) -- its name suggests it carries exactly this shape data and may
  be the intended non-unified source;
- or they must be built per-request in the extension, which is the work the earlier "finish the
  plumbing" framing described, and which [[upstream-fa3-subset-params]] warns is where four port
  bugs already lived (per-item data passed where per-request was wanted).
Read `UpstreamFA3Plan` and how `shape_only` is filled FIRST. If it already has the five fields, the
fix is to source them from it under `#if FLASHINFER_FA3_UPSTREAM` when `kUnifiedWgmma` is false --
a real fix, still small, and no feature disabled. I am NOT applying a guard-only change, because an
FA3 baseline launched with an empty page table would produce a number worse than no number.

**The fix is now fully specified (and it IS a fix, not a disable).** `UpstreamFA3Plan`
(`hopper/fa3_upstream.cuh:140-154`) carries exactly the five fields, and its own comments give the
mapping to our plan's arrays:
```
int const* page_table;    // our kv_indices
int const* cu_seqlens_q;  // our qo_indptr     <-- PER-REQUEST
int const* seqused_k;     // our kv_lens       <-- PER-REQUEST
int total_q;              // varlen Q extent (shape_Q[0]); NOT num_pages -- conflating them
                          // mis-shapes the Q tensor and reads out of range (comment at :147)
int num_pages;            // token count of the FA3 SUBSET only; must match num_fa3_requests,
                          // because choose_scheduling_algo and LINEARIZE_M_AND_BATCH derive from
                          // that pair (comment at :150-153)
```
So for the non-unified `FLASHINFER_FA3_UPSTREAM` build, those five come from the extension's own
FA3 plan arrays -- the same data the fused path already has -- rather than from
`params.wgmma_plan`. Concretely: replace the five `params.wgmma_plan.*` reads with the equivalent
plan pointers under `if constexpr (!kUnifiedWgmma)`, keeping the wgmma_plan source for the unified
case.
**The two traps, both already documented in this repo, so do not rediscover them:**
1. `cu_seqlens_q` and `seqused_k` must be **per-REQUEST**; the persistent plan's `q_indptr`/`kv_len`
   are per WORK ITEM. [[upstream-fa3-subset-params]] records four port bugs of exactly this shape.
2. `total_q` vs `num_pages` are different quantities (Q extent vs FA3-subset token count) and the
   header warns conflating them reads out of range.
That is the complete remaining work for the per-executor FA3 baseline, and it is the same unblock
the fork serving A/B needs. Nothing else in the chain is outstanding: kernel PASSES, split optimum
known, environment restored and documented, drivers staged, composition counter identified as the
one instrument to add.

**The five sources ALREADY EXIST in the extension, with the right semantics.** The `fa3_only_run`
call site (`fused_fa3_persistent_ext.cu:276-288`) already passes exactly what the non-unified
FA3_UPSTREAM path needs, and its comment names the third trap:
```
// RAW page ids. fa3.kv_indices is the VECTOR-SPARSE array, which upstream
// must not see: its paged path treats the entry as a page number.
static_cast<IdT*>(kv_indices.data_ptr()),          // -> up.page_table
static_cast<IdT*>(kv_lens_per_req.data_ptr()),     // -> up.seqused_k      (PER-REQUEST)
static_cast<IdT*>(q_indptr_per_req.data_ptr()),    // -> up.cu_seqlens_q   (PER-REQUEST)
static_cast<IdT*>(kv_indptr_per_req.data_ptr()),
int(pp.num_fa3_requests),
```
So `kv_indices` (raw), `kv_lens_per_req`, `q_indptr_per_req`, `kv_indptr_per_req` and
`pp.num_fa3_requests` are all constructed already -- the non-unified build does not need new plan
machinery, it needs those existing pointers routed to the fused path's `up.*` assignment instead of
`params.wgmma_plan.*`. Mechanically: carry them on the params struct the fused dispatch receives
(the extension fills it a few lines above) and read them under `if constexpr (!kUnifiedWgmma)`.
**Three traps, all now documented with their source lines:**
1. per-REQUEST vs per-work-item for `cu_seqlens_q`/`seqused_k` ([[upstream-fa3-subset-params]]);
2. `total_q` (varlen Q extent) must not be conflated with `num_pages` (`fa3_upstream.cuh:147`);
3. `page_table` needs RAW page ids -- `fa3.kv_indices` is vector-sparse and upstream treats the
   entry as a page number (`fused_fa3_persistent_ext.cu:279`).
That completes the specification: no design question remains, the data exists, and the change is a
routing one. This is the last outstanding item before the per-executor FA3 baseline and the fork
serving A/B can both run.

### The edit, now mechanical: field names confirmed on both sides — 2026-10-09

`UnifiedWgmmaPlanPtrs` (`unified_wgmma_scheduler.cuh:186-233`) already declares every field needed:
```
IdType* dense_page_table;    // :202   <- raw kv_indices
int     max_pages_per_seq;   // :203
int     total_q;             // :204
int     num_requests;        // :208
int     num_pages;           // :209
int     max_seqlen_q;        // :211
IdType* kv_lens_per_req;     // :228
IdType* q_indptr_per_req;    // :229
IdType* kv_indptr_per_req;   // :233
```
and the extension already builds exactly those arrays for `fa3_only_run`
(`fused_fa3_persistent_ext.cu:276-288`).
**So the change is purely mechanical, and the shape that avoids enabling the broken path is:**
always select the WRAPPER TYPE (`using FA3ParamsSel = UnifiedWgmmaParams<FA3Params, IdT>;`
unconditionally, so `wgmma_plan` exists) while leaving `constexpr bool kUnifiedWgmma` driven by
`FLASHINFER_FUSED_UNIFIED_WGMMA` as it is today -- the SCHEDULER choice stays off, only the params
struct gains a field. Then populate `fa3.wgmma_plan.{dense_page_table, q_indptr_per_req,
kv_lens_per_req, kv_indptr_per_req, total_q, num_pages, num_requests, max_pages_per_seq,
max_seqlen_q}` from the same locals the `fa3_only_run` site uses. That makes `FA3_UPSTREAM=1`
compile with the non-unified scheduler, which is what the per-executor FA3 baseline needs, WITHOUT
touching [[fused-wgmma-splitkv-wrong]]'s broken unified route and without leaving any field unset
(the objection that stopped the guard-only fix).
Verification sequence after the edit, in order: (1) `FA3_UPSTREAM=1` compiles; (2) the kernel test
still PASSES fixture 1 (`0 unwritten`, three rows ok) -- the wrapper must not perturb the fused
path; (3) `TWO_EXEC=1` runs and prints a per-executor FA3 time to divide against; (4) only then the
fork serving A/B, with an always-on composition counter added first.

## DIRECTION CORRECTED BY THE USER: the kernel must land on UPSTREAM, and its interfaces and
## fusion boundaries must FIT upstream — 2026-10-09

User, verbatim in substance: the kernel must land on upstream -- "what's the point of tweaking
schedule without the kernel" -- and "especially the kernel interfaces and fusion boundaries must fit
upstream". My earlier framing (asking whether the kernel should go to main) was wrong: the 4-mode
port on `upstream_port` measures the SCHEDULING change against upstream's own attention, which
cannot demonstrate the project's thesis. The combined arm is the only meaningful one.

**My params-routing edit was exactly the mistake this constraint names, and it is REVERTED.**
I set `wgmma_plan.dense_page_table = kv_indices.data_ptr()` -- our FLAT, VECTOR-SPARSE array --
into a field whose own comment says it is "the [batch, max_pages] table upstream believes it owns".
Result: `CUDA error: an illegal memory access`. The extension builds the correct thing for the
`fa3_only_run` path:
```
up_pt = at::full({num_requests * max_pages_per_seq}, -1, iopt);   // :268  dense, per-request
up_page_table = up_pt.data_ptr<int32_t>();                        // :271
// -1, not 0: 0 is a valid page id, so zero padding would make every short
// sequence read page 0 for its tail blocks instead of being masked off.
```
So the boundary is a REPRESENTATION CONVERSION, not a pointer hand-off: our per-work-item,
vector-sparse KV indexing -> upstream's dense per-request page table plus per-request
`cu_seqlens_q`/`seqused_k`. Aliasing one into the other is what produced the fault, and it is the
general shape of the error to avoid in the upstream port.

**Revised plan, replacing the fork-first plan above:**
1. Define the fusion boundary in UPSTREAM's terms first: upstream FA3 paged attention consumes
   (dense `[batch, max_pages]` page table, per-request `cu_seqlens_q`, per-request `seqused_k`,
   `total_q`, `num_pages`); upstream's persistent/FA2 side consumes its own plan. The split kernel's
   interface should be expressed as those two contracts, with ONE explicit conversion at the seam --
   built, not aliased.
2. Port onto `upstream_port` (which already has the 4-mode step AND `mixed_dispatch.py`), not onto
   the fork. Surface to reconcile: fork `flashinfer_backend.py` is 3465 lines vs upstream's 2540,
   and the fork's hooks (`SGLANG_TWO_EXEC`, `SGLANG_FUSED_FA3_SM_SPLIT`) must be re-expressed
   against upstream's backend structure rather than transplanted.
3. Re-pin the upstream base: `b7b2975b5` is stale (upstream moved to `54b551df8` within hours of
   the pin), so rebase before adding the kernel.
4. Only then measure: combined fused-kernel + 4-mode arm vs upstream baseline, which is the arm the
   >=30%/>=10% bar actually refers to.
The kernel's own correctness result is unaffected and transfers: it was obtained through the
flashinfer harness with `FI_ROOT` only, no sglang in the loop (PASS on fixture 1, 1.30x over the
unfused persistent kernel at sm_split 112).

### Step 3 measured: the upstream re-pin is CHEAP, so do it before the kernel port — 2026-10-09

Upstream main is now `3fef129ff` against our pin `b7b2975b5`: **46 commits** of drift
(pin 2026-10-07 21:37 -> main 2026-10-08 12:54 PDT). Network reaches github from the dev box, so
`git fetch upstream main` works locally.
Conflict forecast, measured rather than assumed -- of the ~13 files `upstream_port` touches, only
FIVE were changed upstream in those 46 commits, and only 1-2 commits each:
```
1 upstream commit : model_executor/forward_batch_info.py
2 upstream commits: model_executor/runner/prefill_cuda_graph_runner.py
1 upstream commit : speculative/eagle_draft_cuda_graph_runner.py
1 upstream commit : speculative/eagle_worker_common.py
2 upstream commits: speculative/eagle_worker_v2.py
```
`managers/scheduler.py`, `managers/schedule_batch.py`, `layers/logits_processor.py` and
`model_executor/cuda_graph_buffer_registry.py` -- the files carrying the bulk of the 4-mode port --
were NOT touched upstream in that window. So the rebase is tractable: a handful of localised
conflicts in the spec-v2 files, which are exactly the ones whose upstream contracts the port already
had to learn, so the conflicts should be legible rather than structural.
Doing the re-pin FIRST is therefore the right order: it is cheap now and gets more expensive every
day, and porting the kernel onto a 46-commit-stale base would mean doing the interface work twice.
Note `mixed_dispatch.py` appears in the port's own diff -- it is a file WE added to upstream, not an
upstream file, so the dispatch module is already across and the kernel port's remaining sglang-side
surface is the backend hooks (`SGLANG_TWO_EXEC`, `SGLANG_FUSED_FA3_SM_SPLIT`) expressed against
upstream's 2540-line `flashinfer_backend.py`.

### [x] STEP 3 DONE: `upstream_port` rebased onto LIVE upstream main, zero conflicts — 2026-10-09

`git rebase upstream/main` on `/opt/tiger/sglang-up`: **all 60 commits replayed, exit 0, no
conflicts**, despite the 46-commit drift. The forecast above (only 5 touched files changed upstream,
1-2 commits each, the heavy 4-mode files untouched) held exactly.
- new base `3fef129ff` == `upstream/main`; 60 commits on top
- safety refs kept at the old head: branch `upstream_port_backup_20261009`, tag
  `pre_rebase_20261009` (old head `fea0174e0`)
- port markers verified present after the replay, not just assumed from exit 0:
  `mixed_spec_running` (3 files), `MIXEDSPEC_STEPS` (1), `mix_with_verify` (4),
  `mixed_verify_tokens` (5)

**Gap the verification caught: `_apply_decode_population_guard` is in 0 files on `upstream_port`.**
The decode-population guard (the measured `kv_tiles <= decode_rows/2` ratio, validated 11/25 ->
23/25 against `boundary_28x4x128.json`) was committed to the FORK's
`layers/attention/mixed_dispatch.py`, not to the upstream port. It is part of the kernel's dispatch
behaviour, so it must come across with the kernel -- and it is a good first piece to port, because
it is self-contained Python with a measured acceptance test and no upstream-interface question.
Remaining upstream work, in order: (a) port the decode-population guard; (b) express the split
kernel's interface in upstream's two contracts (dense `[batch, max_pages]` page table + per-request
`cu_seqlens_q`/`seqused_k` for FA3; upstream's own plan for the FA2 side) with ONE explicit
conversion at the seam -- never a pointer alias, which is what caused the illegal access; (c) wire
the backend hooks against upstream's 2540-line `flashinfer_backend.py`; (d) measure the combined
fused-kernel + 4-mode arm against upstream baseline, which is the arm the >=30%/>=10% bar refers to.

### [x] Step (a) DONE: decode-population guard ported to upstream and re-verified — 2026-10-09

`upstream_port` commit 61 (`/opt/tiger/sglang-up`, now based on live `3fef129ff`). The upstream copy
of `mixed_dispatch.py` was structurally identical at the `split_classes` tail, so the guard applied
cleanly. **Re-ran the acceptance test against the UPSTREAM copy rather than assuming it transfers**
-- replaying all 25 points of `boundary_28x4x128.json` through that code:
```
UPSTREAM copy: per-row rule 11/25   with guard 23/25
```
identical to the fork's result, so the guard's measured value holds on upstream. Env-gated
(`SGLANG_FUSED_DECODE_GUARD`, `SGLANG_FUSED_DECODE_TILES_PER_ROW`).
Remaining upstream work: (b) express the split kernel's interface in upstream's two contracts with
one explicit conversion at the seam; (c) wire the backend hooks against upstream's
`flashinfer_backend.py`; (d) measure the combined fused-kernel + 4-mode arm. (b) is the real design
piece and the one the "interfaces and fusion boundaries must fit upstream" constraint governs.

### Step (b) started: the upstream interface mismatch, stated from the code — 2026-10-09

Read of `upstream_port`'s `layers/attention/flashinfer_backend.py` (now on live main):
- the seam is `forward_extend` (`:1317`) / `forward_decode` (`:1481`), fed by
  `init_forward_metadata` (`:962`);
- it dispatches to flashinfer's STANDARD wrapper API --
  `BatchPrefillWithPagedKVCacheWrapper` / `BatchPrefillWithRaggedKVCacheWrapper` (`:79-80`,
  instantiated `:498-504`) -- i.e. **one planned wrapper per batch**, with the plan built in
  `init_forward_metadata`.

**That is the mismatch, precisely:** upstream's contract is "one wrapper, one plan, whole batch",
while the split kernel is "two executors over disjoint ROW SUBSETS of one batch, with an SM
partition between them". The fork resolved this by replacing the wrapper path outright (its
`flashinfer_backend.py` is 3465 lines vs upstream's 2540). That is what must NOT be transplanted --
it is the fork's shape, not upstream's.

**Interface the port should expose instead, in upstream's own terms:**
1. **Row partition belongs in metadata, not in the kernel call.** `init_forward_metadata` already
   computes per-batch plan data; the FA3/FA2 row classes (which `mixed_dispatch.split_classes` now
   produces on this branch, guard included) should be computed there and carried on the metadata
   object -- so `forward_extend` stays a dispatch, not a planner.
2. **Two plans, one per executor, each built through the upstream wrapper it belongs to**, rather
   than one bespoke plan serving both. The FA3 subset needs upstream FA3's paged contract (dense
   `[batch, max_pages]` page table, per-request `cu_seqlens_q`, per-request `seqused_k`, `total_q`,
   `num_pages`); the FA2 subset needs the persistent/BatchAttention plan. Each is a supported
   upstream object.
3. **ONE explicit conversion at the seam**, materialising the dense per-request page table from our
   paged KV indices -- built, never aliased. The illegal access recorded above came from aliasing a
   flat vector-sparse array into that field, and the harness already shows the correct construction
   (`up_pt = at::full({num_requests * max_pages_per_seq}, -1)`, then scatter; -1 because 0 is a
   valid page id).
4. **The SM split stays a launch parameter**, not an interface concept: both executors are ordinary
   upstream calls; only the grid/SM budget differs. That keeps the fusion boundary at the launch,
   which is where the measured 1.30x and the sm_split-112 optimum live.

Open question for (b), to settle by reading before coding: whether upstream's wrapper API permits a
caller-supplied SM/grid budget, or whether the split must be expressed one level down at the
flashinfer module call. That determines whether (c) is "add a budget argument" or "bypass the
wrapper for the FA3 subset only" -- and it is the last unknown before the port can be written.

### [x] Step (b) open question ANSWERED: the SM split cannot go through upstream's wrapper API

Read of flashinfer 0.7-era `flashinfer/prefill.py`:
- **`:204` `sm_count = get_device_sm_count(query.device)`** -- the wrapper derives the SM count
  INTERNALLY from the device and passes it down (`:227`). There is no caller-supplied SM/grid budget
  anywhere in `BatchPrefillWithPagedKVCacheWrapper`'s surface.
- **`:1646`** documents the block table as *"a uint32 2D tensor indicating the block table of each
  prompt. shape: `[batch_size, max_num_blocks_per_seq]`"*, and `:1861-1863` constructs exactly that
  `(batch_size, max_num_blocks_per_seq)` tensor. So upstream's dense per-request page-table contract
  is confirmed verbatim -- and it is precisely the shape my reverted edit violated by aliasing a
  flat vector-sparse array into it.

**So (c) is settled as a design question:** the SM split must be expressed at the MODULE call, not
through the wrapper. Which means the upstream-fitting split is:
- metadata and row partition: upstream objects and upstream call sites (`init_forward_metadata`,
  `split_classes` + guard, already on this branch);
- per-row-class data handed over in upstream's documented contracts -- dense
  `[batch_size, max_num_blocks_per_seq]` block table, per-request `cu_seqlens_q`/`seqused_k`;
- the two executors invoked as module-level calls carrying an SM budget, which is the one place the
  wrapper abstraction genuinely cannot express the design. That is a defensible fusion boundary to
  argue for upstream: everything above it is standard upstream API, and the only non-standard part
  is the launch budget, which is inherent to SM pre-assignment.
This also reframes the earlier framing honestly: "the fork replaced the wrapper path" was not purely
fork sloppiness -- the wrapper API cannot carry an SM budget, so SOME bypass is unavoidable. The
difference is that the bypass should be narrow (launch budget only) instead of a 900-line divergent
backend.
Step (b) is therefore specified end to end. Remaining: (c) implement that narrow bypass on
`upstream_port` against upstream's contracts, then (d) measure the combined fused-kernel + 4-mode
arm, which is the arm the >=30%/>=10% bar refers to.

## HANDOFF STATE — 2026-10-09

**Upstream port (`/opt/tiger/sglang-up`, branch `upstream_port`, 62 commits on live `3fef129ff`):**
- rebased onto live upstream main, zero conflicts, port markers verified post-replay;
  safety refs `upstream_port_backup_20261009` / tag `pre_rebase_20261009`
- 4-mode MTP step: runs (16 mixed steps under load, 0 tracebacks)
- decode-population guard: ported and re-verified ON THAT COPY (11/25 -> 23/25)
- FA3/FA2 row partition: in `init_forward_metadata` (commit 62), default OFF, lazy import,
  logs-and-falls-through on failure, so flag-off paths are byte-identical

**Kernel (flashinfer, `/opt/tiger/flashinfer` + pod `FI_ROOT=/dev/shm/w/fi`):**
- PASSES the causal fp32 reference on fixture 1, every sm_split: 0 unwritten,
  FA3 0.00628, task0 0.00250, tile16 0.00071
- 1.30x over the unfused persistent kernel on identical work (133.5 us @ sm_split 112 vs 174.0 us);
  optimum split 112 lies OUTSIDE the harness's built-in 48/32/16 sweep
- two fixes required: `MaskMode::kNone -> kCausal` on the persistent runners, and
  `cluster_tile_q = (FUSED_GRID ? 1u : gridDim.x) * CTA_TILE_Q`
- obtained through the harness alone (no sglang), so the result transfers to the upstream wiring

**THE NEXT INCREMENT, fully specified:** the kernel call in upstream's `forward_extend`, consuming
`self.fused_split_mask` from commit 62. It must hand over upstream's DOCUMENTED contracts --
dense `[batch_size, max_num_blocks_per_seq]` block table (`flashinfer/prefill.py:1646`, built at
`:1861`), per-request `cu_seqlens_q` and `seqused_k` -- with the SM budget as the ONLY non-standard
element, because `prefill.py:204` derives `sm_count` from the device and the wrapper cannot carry
one. Build the dense table (`at::full({num_requests * max_pages_per_seq}, -1)` then scatter; -1
because 0 is a valid page id), never alias a flat vector-sparse array into it -- that aliasing is
what produced `CUDA error: an illegal memory access` and was reverted.

**Then measure (d):** combined fused-kernel + 4-mode arm vs upstream baseline, on a workload with a
real decode population (the guard refuses pure long-prefill batches BY DESIGN, which is why the
IN=32768 A/B could never have shown a gain). Add an always-on composition counter FIRST -- the
`MIXEDSPEC_STEPS` pattern caught a false pass on its first use, and the fork driver's
`two_exec_calls` greps a string the code never emits, so it proves nothing.

**NOT DONE, and these are the goal's criteria:** >=30% TTFT/ITL, >=10% throughput,
`tma_store_wait` multicast/cross-role backfill, EDF perf-model vs baseline, PD-colocation point 4.
Nothing is pushed: `git push` needs interactive auth here (`! git -C /opt/tiger/sglang push`,
`! git -C /opt/tiger/sglang-up push -u origin upstream_port`).

### [x] Step (c) increment 2: the seam conversion, implemented and CPU-verified — 2026-10-09

`upstream_port` commit 63 (now 63 commits on live `3fef129ff`). `mixed_dispatch.dense_block_table`
converts sglang's flat paged KV indices into upstream FA3's documented
`[batch_size, max_num_blocks_per_seq]` block table. This is the fusion boundary stated earlier,
built as an explicit conversion instead of a pointer hand-off -- the aliasing version is what
produced `CUDA error: an illegal memory access`.
Verified on CPU (index math, no GPU needed), **11/11 properties**, with the two that caused real
faults covered explicitly:
```
ok  row0 exact (incl. page id 0)        ok  subset batch dim == len(rows)
ok  empty row is all -1                 ok  subset keeps row order
ok  short row padded with -1 not 0      ok  page_size>1 one entry per page, ids divided down
ok  shape / max_blocks / int32 dtype    ok  empty rows -> empty table
```
Why those two matter and are not style points:
- **-1 fill, never 0** -- page id 0 is VALID, so zero padding would make every short sequence read
  page 0 for its tail blocks instead of being masked off;
- **batch dim == FA3 SUBSET size, in row order** -- upstream's combine scheduler derives its tile
  mapping from the (token extent, batch count) pair, so handing it the full-batch count with a
  subset's tokens mis-maps tiles (this is the `num_pages`/`total_q` conflation the header warns
  about at `fa3_upstream.cuh:147-153`).

Step (c) now has: row partition in upstream's metadata hook (commit 62) and the seam conversion
(commit 63), both default-inert and both verified without a GPU. What remains is the dispatch
itself in `forward_extend` -- two executor calls over the row subsets, with the SM budget as the
only non-standard element -- plus `total_q`/`seqused_k`/`cu_seqlens_q` assembled per-request from
the same partition. That step needs the pod (build + fp32 reference + combined arm), which is why
it is where this session stops.

### [x] Step (c) increment 3: per-request varlen assembly, CPU-verified — 2026-10-09

`upstream_port` commit 64 (64 commits on live main). `mixed_dispatch.fa3_subset_varlen` returns
`(cu_seqlens_q, seqused_k, total_q)` for the FA3 row subset. Verified **10/10** on a deliberately
NON-contiguous subset (rows [0,3] of 4), with the three trap properties covered:
```
ok  densely renumbered over the SUBSET (not the batch)   ok  order follows the rows arg
ok  total_q == cu[-1], Q extent not a page count         ok  total_q invariant under reorder
ok  subset total != whole-batch total                    ok  int32 dtypes, empty-subset case
```
With commits 62-64 the entire upstream-facing data path is now implemented and verified WITHOUT a
GPU: row partition in upstream's metadata hook, the dense block table, and the per-request varlen
arrays. Each encodes a documented upstream contract, and the three properties that previously caused
faults (-1 fill vs valid page id 0; subset batch dim for the combine scheduler; per-request vs
per-work-item indexing) are now covered by tests rather than by comments.

**Remaining and genuinely pod-bound:** the `forward_extend` dispatch -- call the two executors over
the row subsets using these three builders, with the SM budget as the only non-standard element --
then the fp32-reference check on the upstream build and the combined 4-mode + fused arm. That
sequence needs build/verify cycles on the GPU, which is the honest reason it is not done here.

### [x] The dispatch contract, grounded — upstream ALREADY has a two-executor forward_extend

Read of `upstream_port`'s `flashinfer_backend.forward_extend` (`:1317`):
```
def forward_extend(self, q, k, v, layer: RadixAttention, forward_batch, save_kv_cache=True):
    prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[self._get_wrapper_idx(layer)]
    ...
    if not self.forward_metadata.use_ragged:
        o = prefill_wrapper_paged.forward(...)          # single executor
    else:
        o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(...)   # TWO executors
        o2, s2 = prefill_wrapper_paged.forward_return_lse(...)
        # ... combined via the LSE state merge
```
**So a two-executor `forward_extend` that takes `(o, lse)` from each and combines is IDIOMATIC
upstream, not a foreign shape.** That materially lowers the bar for landing the split kernel
upstream, and it corrects my earlier framing ("one planned wrapper per batch") -- upstream's
single-wrapper assumption is the `use_ragged=False` path, not the whole contract.

**The one structural difference, stated precisely:** upstream's two executors run over the SAME rows
(ragged = the new tokens, paged = the prefix) and therefore MERGE via LSE. Our FA3/FA2 split runs
over DISJOINT ROW SUBSETS, so the combine is a SCATTER, not a merge -- each executor's output goes
to its own rows and no LSE combination is needed. That is strictly simpler than what upstream
already does, and it means `forward_return_lse` is not even required for our path: plain `forward`
per subset plus an index_copy into the output is enough.
So the dispatch to write is:
```
mask, q_tiles, kv_tiles = self.fused_split_mask            # commit 62
fa3_rows  = rows where mask;  fa2_rows = rows where ~mask
table, _  = dense_block_table(kv_indptr, kv_indices, fa3_rows, page_size)   # commit 63
cu_q, used_k, total_q = fa3_subset_varlen(qo_lens, kv_lens, fa3_rows)       # commit 64
o = empty_like(q-shaped output)
o[fa3_rows] = <FA3 executor>(q[fa3_rows], ..., table, cu_q, used_k, total_q, sm_budget)
o[fa2_rows] = <persistent executor>(q[fa2_rows], ... upstream plan ...)
return o
```
with the SM budget the only non-standard argument (flashinfer's wrapper derives `sm_count` from the
device, `prefill.py:204`, so it cannot carry one). Everything above that line is upstream API plus
the three verified builders from commits 62-64.
**Still pod-bound:** writing the two executor calls and verifying them -- the fp32 reference on the
upstream build, then the combined 4-mode + fused arm. Not done, and that is the gap to the bar.
