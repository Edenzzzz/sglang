# Changelog

## v3.83.0 — Paired run confirms the chunk-cap fix, with revised figures (2026-10-29)
- Both arms under identical conditions, three repeats each, supersedes the after-only pass: below-1.0x 13/28 ->
  9/28 (not 8/28), median 1.008 -> 1.161 (not 1.169), realistic chunk 0.975/0.981/0.955/0.937 ->
  1.011/1.059/1.003/1.027, toy chunk bs64 0.910 -> 1.231 and 0.745 -> 1.091, long-tail 1.161-6.813 -> 1.681-7.192.
- Precision differs by family and only one family is quotable per capture: the agent-dominant and realistic-chunk
  rows that carry the fix have 0.4-4% spreads and reproduce within 0.5%, while the nine remaining sub-1.0x
  sharegpt captures move 10-20% between runs with 12-78% intra-run spreads. No individual sharegpt ratio may be
  quoted to better than +-15% -- consistent with sub-millisecond steps where launch and plan jitter dominate.
- Paper 6.2 updated to the paired figures.

## v3.82.0 — The floor fires and the overshoot is gone, but the gap survives it (2026-10-29)
- Hoisted floor at alpha 6: chat TPOT 46.3 -> 57.0 ms, landing on L0's 57.1, mean TTFT 8.26 -> 6.41 s, loose
  attainment 50.3 -> 56.7%. So the floor does spend the slack, as predicted.
- Tight attainment did not move (39.4% against L0's 61.5%) because we now miss BOTH targets: TPOT 57.0 is over the
  50 ms target and TTFT 6.41 s is still 2.8x the baseline's 2.28 s. The rule surrendered its TPOT advantage
  without buying back enough TTFT, so it is now strictly worse than the baseline on both axes.
- This EXCLUDES the floor as the explanation -- it fires, it behaves correctly, and the gap survives it. What
  remains is the budget's magnitude: the model over-pricing the chunk (fitted with dkv identically zero) and the
  minimum over rows letting one outlier set it. A larger constant would only tune the floor toward the baseline's
  chunk size, which is the baseline with extra steps.

## v3.81.0 — Sub-1x shapes fixed: flashinfer's FA2 chunk rule degenerates on all-long batches (2026-10-29)
- flashinfer unified_dispatch_20261024 d9cd7e6. Below-1.0x captures 13/28 -> 8/28, median 1.008 -> 1.169,
  correctness 28/28 PASS, two independent after-runs within 0.5%.
- My routing hypothesis was refuted by per-role decomposition: our FA3 role is NOT sglang's FA3 call. It is +1.7%
  on a large prefill and +25..+40% on narrow long-KV rows, so the all-FA3 limit sits ABOVE the baseline and
  routing alone could never reach 1.0x.
- Actual cause: flashinfer's kv_len_limit is total work / clusters, so on a batch whose rows are all long it lands
  at the longest row, nothing splits, and one row runs alone on one CTA (260 items for 256 row-heads, 7.77 ms
  against a 3.9 ms work bound). This also explains the non-monotonicity -- the chunk grows with total work until
  it stops splitting. Fix caps the chunk at a quarter of the keys one slot must walk, plus an empty-queue launch
  skip.
- Supersedes today's 0.937-0.972x realistic-chunk result: those shapes are now 1.008-1.056x, and the agent
  -dominant regressions (0.745-0.910x) are now 1.120-1.219x.
- Still below 1.0x only on chat-dominant/uniform captures: 0.2-1.6 ms steps with ~800-key KV where there is
  nothing to split, both roles are individually behind on tiny work, and the mixed step equals the SUM of our
  parts because PDL buys no overlap at that size.

## v3.80.0 — Two defects in the chunk rule, and a zero-GPU discriminator (2026-10-29)
- The baselines refute the "cluster is overloaded" explanation: pp 57.8%, mc 53.8%, L0 61.5% at the same alpha 6,
  so 54-62% is attainable and our 41.1% gives up ~20 points. budget_ms = -11731 is evidence about our accounting,
  not the cluster.
- The tell is an overshoot: we land chat TPOT at 46.3 ms against a 50 ms TARGET while paying 8.26 s of TTFT
  against L0's 2.28 s, admitting 12% fewer agent rounds and carrying 30% less SLO-attaining throughput. The rule
  does not protect TPOT, it overshoots past the target and spends prefill capacity on headroom the SLO never asked
  for. Correctness criterion sharpened: admit until projected mean TPOT REACHES the target, neither exceeding nor
  falling short.
- Two compounding causes: (1) structural -- the minimum over rows lets one outlier veto, and a row can escape the
  demote test (which fires on history, spent > s*L) while vetoing through the projection term (L-o-a)*rate;
  (2) predictive -- the model was fitted with dkv identically zero, so its decode terms, the ones that matter at
  overload, are fitted to noise and over-price the chunk.
- Discriminator needing no GPU: refit on the 11541 clean decode steps from the alpha-3 traced arm and re-run. TPOT
  near 49-50 means the gap was prediction; TPOT still near 46 means the residual is structural.
- Withdrawn: "a better model would optimise the wrong objective" -- it rested on a starving rule and on assuming
  the model was sound.

## v3.79.0 — The gain's mechanism: padding wastes compute, not bandwidth (2026-10-29)
- Unifies every kernel measurement of the day into one falsifiable claim. Padding a one-query decode input into a
  128-row tile does not change KV traffic -- the same bytes stream either way -- it wastes the MMA. So the waste is
  free exactly when it hides behind memory traffic: bandwidth-bound long-KV inputs absorb it, compute-bound
  short-KV inputs do not. That predicts the measured split (1.15-7.44x spread-KV, 0.937-0.972x prefill-dominated)
  rather than describing it after the fact.
- Hardware consequence, against the intuition that faster memory erodes a kernel gain: machine balance is ~295
  FLOP/byte on H100 against ~206 on H200, and a lower balance makes a given shape more likely compute bound, so
  H200 puts MORE of the step in the regime where tile waste costs time. Capacity agrees independently -- 141 GB
  holds larger mixed batches, hence more narrow-width work beside each chunk.
- Caveat recorded: this is machine-balance reasoning, not a paired measurement, and composition dominates hardware
  by an order of magnitude (1:2 to 67:1 work ratio swung 7.44x to 0.94x). A kernel ratio must be re-measured on new
  silicon rather than assumed to carry.

## v3.78.0 — The floor fix failed because of where I put it; hoisted (2026-10-29)
- Measured: alpha 6 with the floor deployed gives chat 40.9% against 41.1% without it, and zero=2255 of calls=5501
  -- still 41%, unchanged. The floor never fired.
- Cause: the `elif not meets:` branch sat inside `if deadline is not None and deadline > now:`. A prefill whose
  deadline has already passed has deadline <= now, so the block is skipped -- the common case under overload and
  exactly the one the floor exists to serve. The rescue was inside a guard excluding whoever needs rescuing.
- The same run shows budget_ms = -11731.3, so the budget is deeply negative rather than merely small.
- apply_floor_hoist_20261029.py moves an unconditional minimum out of the deadline guard, and logs latefloor=,
  which the first patch computed but never printed -- which is why the failure had to be inferred from zero rather
  than read directly. Deployed and verified on python_slo_life on all four nodes.
- The alpha-3 point of the first run is unusable: its tree was patched mid-driver.

## v3.77.0 — The failed merge experiment rules out DSMEM/global-barrier merge fusion (2026-10-29)
- Deleting the barrier and the merge tail outright moved the step -0.1% to +0.2%, inside the sub-1% spread. So
  fusing the reduction into FA3 behind a global barrier, or handing partials between cluster CTAs over distributed
  shared memory, cannot be worth more than deleting them. Declined on evidence; the evidence exists only because
  the optimization failed.
- By elimination the fixed 0.4-1.0 ms is the second launch's existence or our FA2 kernel's cost on long-KV rows
  against FA3 handling the same rows. The latter is the only candidate consistent with everything measured: the
  penalty scales with the number of long rows, vanishes where rows are short and heterogeneous, and survives every
  merge-side change. Cost-aware role routing is the fix, and its worst case degrades to the baseline's assignment
  plus PDL rather than below it.
- Policy note: pdmux's floor and SLOs-Serve's admission control are the same property in different components, not
  competing choices. A zero chunk is sound only when admission guarantees it is transient -- theirs is, ours was a
  steady state. The design uses both: the floor as a local safety net, router-side SLO-checked admission as the
  global policy, which is point 1 of the PD-colocation points.

## v3.76.0 — pdmux structurally cannot allocate zero SMs; read from the implementation (2026-10-29)
- Settled by reading python/sglang/srt/multiplex/pdmux_context.py rather than the paper, which is silent on it.
  divide_sm() enumerates partitions as [x for x in range(min_per_part, total_sms - min_per_part + 1, multiple)
  if x >= total_sms - x and total_sms - x >= 16], appended as (prefill SMs, decode SMs). On Hopper min_per_part is
  8 with multiple 8, so prefill always takes at least half and decode always keeps at least 16: neither role can
  ever be allocated zero.
- That reinstates the original claim and withdraws yesterday's retraction of it. Both of my readings of the paper
  were unfounded because the paper does not address the question; the code does. Our runs' 40-56 of 78 SMs for
  prefill are an instance of the constraint rather than a coincidence.
- Consequence for the fix: there are TWO differences from the systems that run this policy successfully, not one.
  pdmux bounds the resource below; SLOs-Serve resolves infeasibility at admission. Our rule did neither -- it drove
  the chunk to zero and re-litigated feasibility every step.

## v3.75.0 — The merge-tail guard is a no-op: the barrier was not the fixed cost (2026-10-29)
- Three repeats per arm: -0.1%, +0.1%, +0.1%, +0.2% against the pre-patch build, all inside the sub-1% spread. The
  grid-dependency barrier was NOT the 0.4-1.0 ms fixed penalty, so that hypothesis is refuted by its own
  measurement.
- Why the reasoning failed: the barrier sits at the very END of the FA2 kernel, where those CTAs have already
  finished their tiles, and under PDL FA3's CTAs are already resident and running. Waiting there delays those CTAs'
  retirement, not FA3's progress. "Holding SMs that FA3 needs" assumed FA3 was still waiting to be scheduled.
- The guard is kept -- free, gated correct at 10/10, and it removes a barrier that does nothing -- but the fixed
  cost remains unexplained. Remaining candidates: the second launch itself, and our FA2 kernel's cost on these row
  shapes against FA3 handling the same rows.

## v3.74.0 — Read the papers: the difference is an admission decision, not a floor (2026-10-29)
- Checked against tasks/papers/. Two earlier notes were wrong: first that pdmux does not use a latency-budgeted
  allocation, then that its knob cannot reach zero.
- MuxWise p8: "the remaining compute resources are allocated to the prefill", at 16-SM granularity -- decode takes
  what the model says it needs and prefill the remainder, the same decision as ours with SMs as the resource.
  Nothing in the paper establishes a nonzero floor, so that claim is withdrawn; what holds is only that our own
  pdmux runs gave prefill 40-56 of 78 SMs.
- SLOs-Serve p7/p8/p11: late requests consume SURPLUS budget, a request that cannot meet its SLO is ROUTED TO THE
  NEXT REPLICA, and under high load requests are REJECTED. So the infeasible case is resolved at admission.
- The real distinction: both systems decide once whether they can serve a request and then serve what they
  admitted. We have no admission decision -- the router admits everything and the chunk rule re-decides against it
  every step, which is how an admitted prefill got nothing 41% of the time. A floor bounds the damage; the policy
  needs the infeasible case resolved at admission (reroute or defer), which is point 1 of the four PD-colocation
  points.

## v3.73.0 — Correction: pdmux runs the same policy, which is why ours is the broken one (2026-10-29)
- An earlier note claimed pdmux does not use a latency-budgeted allocation. Wrong: pdmux sizes the prefill SM
  allocation from a performance model of what decode needs to hold its SLO -- the same decision as our chunk rule,
  with SMs as the resource rather than chunk tokens.
- That makes pdmux the strongest evidence that the policy is sound and our implementation is not. pdmux's knob
  CANNOT reach zero: it moves prefill onto fewer SMs, never onto none. Ours drives the chunk to zero on 41% of
  steps -- the same policy with an unbounded knob.
- Two systems running this policy without the pathology, plus SLOs-Serve 4.1's surplus rule, give one principle:
  the resource granted to prefill must be REDUCED, never withdrawn. That also fixes the floor's value -- the
  analogue of pdmux's minimum SM group, the smallest allocation that still makes progress, rather than a constant
  picked by hand.

## v3.72.0 — The budget's minimum-over-rows is the defect; reject only when all capacity is needed (2026-10-29)
- The model is not over-predicting. A chunk's attention costs chunk x prefix, so at a ~100K prefix a 2048-token
  chunk really does cost about 600 ms, and the baseline's chat TPOT p99 of 401 ms shows it absorbs such steps. The
  decision is what is wrong: the SLO is a MEAN, so one 600 ms step adds 600/n ms to a row's mean -- about 1.5 ms at
  a 400-token output -- which is absorbable.
- The structural defect is that the budget is a MINIMUM over resident rows' slack, so one row in trouble vetoes
  prefill for the whole step. SLOs-Serve section 4.1 instead prioritises requests that can still meet their
  deadlines and serves late ones from surplus capacity; ours has neither half. The fix is to price the step against
  savable rows, admit prefill from the surplus, and reject only when the savable rows need all of it.
- Two causes eliminated by controls rather than argument: a fixed 2048-token chunk never consults the budget, which
  is why c2048 tracks L0; and since c2048 shares our router, tree and policy, routing to loaded nodes cannot be the
  cause. pdmux does not face this -- it partitions SMs and has no latency-budgeted chunk.
- Recorded a numerical inconsistency to chase first: under the life form a row 10 tokens into a 400-token output at
  14 ms against a 50 ms target should grant 14 428 ms, yet logged budgets at alpha 6 are 40-188 ms while
  nolife=729 says the lifetime branch was taken. Next step is to log which row supplies the minimum rather than
  reason about it.

## v3.71.0 — The SLO-chunk "loss" is a fifth bug: a zero chunk on 41% of calls (2026-10-29)
- The user pushed back on the loss being real. It is not. The arms ran the tree with all four earlier fixes present,
  but the rule's own counters show it returns a chunk of ZERO on 2674 of 6501 calls (41%), i.e. it admits no prefill
  at all on four steps in ten.
- Cause: the TTFT floor's guard is `if meets and floor > out`, where `meets` means the head prefill can still make
  its deadline. Under overload almost nothing can, so no floor is granted and the chunk stays zero -- self
  reinforcing, since a prefill denied service can never stop being late. Same error as the demote bug fixed earlier:
  treating "already missed" as grounds to withhold service rather than grounds not to block others.
- It explains every symptom together: TPOT better by 11-28% (decode unobstructed), TTFT 3.5-4x worse and attainment
  19-33 points down (prefill starved), fewer rounds admitted, and insensitivity to the cap across a 16x range
  because the cap never bound -- `zero` did.
- WITHDRAWN: that the chunk layer is a loss; that budget-sizing cannot work on this workload; and that a better
  step model would optimise the wrong objective. Fix is to floor the earliest-deadline prefill regardless of
  whether it can still meet its deadline, then re-run alpha 3 and alpha 6.

## v3.70.0 — Skip the FA3 merge tail and its barrier when there is nothing to reduce (2026-10-29)
- Implemented, not just measured. The FA3 split partials are reduced by the FA2 kernel, which in the static path
  ends with cudaGridDependencySynchronize() then UnifiedMergeFA3Tail. Under PDL that barrier is where a CTA waits
  for the previous kernel, so FA2 CTAs launched early, ran their tiles, then blocked HOLDING THEIR SMs until the
  whole FA3 grid finished -- SMs that FA3's own CTAs could use on a prefill-dominated step. That is contention
  rather than latency and fits the measured fixed 0.38-1.02 ms better than a bare barrier.
- The realistic-chunk plans report fa3_merge_reqs = 0, so the tail had nothing to reduce and the barrier was paid
  for nothing. Guarding on num_fa3_merge_reqs > 0 lets those CTAs retire and release their SMs, and also covers the
  CTAs a cluster-sized persistent grid leaves with no items at all, which previously still waited.
- Correctness gate: 10 of 10 captures PASS, 0 failures, including all six toy-chunk captures where fa3merge=1 --
  the control showing the split path is byte-for-byte unchanged where partials exist. Timing in progress against a
  0.937-0.972x baseline whose spreads are under 1%.
- Implementation-level; deliberately not described in the paper.

## v3.69.0 — Realistic chunk + agent-dominant rows: 2.8-6.3% slower, spreads under 1% (2026-10-29)
- 4096-token chunk over a 65536-token prefix with 75% of rows at 64K-200K, three repeats: 0.972x, 0.957x, 0.956x
  and 0.937x against sglang's single attention call, and 0.896-0.949x against v3. Every spread under 1%, so this is
  settled for this composition. All four captures passed the FP32 gate first.
- The penalty is a FIXED 0.38-1.02 ms on a 13-16 ms step, growing only mildly with batch, so it is the two-launch
  structure plus the merge rather than the tile mechanism. The reason it earns nothing here: a 4096-token chunk
  over a 65K prefix does about 67x the attention work of 32 decode rows at 126K, so the step is overwhelmingly one
  prefill row's attention, which both arms give to FA3 identically.
- The three sweeps now bracket the mechanism by prefill-to-decode work ratio rather than contradicting one another:
  about 1:2 gives 1.15-7.44x (10% agent) and 2.12x-0.74x by batch (75% agent); about 67:1 gives 0.937-0.972x.
- Both queued merge-placement optimizations target the fixed cost, and this is the capture set to measure them on.
- Caveat carried rather than buried: the trace-sampled 1.86-2.92x is a statement about the real distribution of
  that ratio, and those captures cannot currently be regenerated on this trial.

## v3.68.0 — The decode-KV tracer works: 11541 decode steps, none with dkv=0 (2026-10-29)
- The alpha-3 traced point launched fresh workers against the patched tree and records real KV lengths, against
  6117 of 6117 zeroes in the arm that ran before the patch. Sample: DECODE bs=8 nd=8 dkv=163904 kv=163904
  gpu=14.8003 -- eight decode rows holding 164K KV tokens in a 14.8 ms step, the term the decode-step model had
  been fitted without.
- Point 3's premise is testable for the first time: refit on these traces and measure the decode-step signed bias
  on a chronological split. The three readings that rested on the dead input -- 29.0% MAPE, -13.0% bias, and the
  nonlinear fit being worse at -19.9% -- were all measured against a zero feature and remain withdrawn until this
  refit lands.

## v3.67.0 — Deferred KV commit for rejected draft tokens: declined with arithmetic (2026-10-29)
- Two independent grounds, no GPU time. The traffic is negligible: 2 048 B per token of K and V, two rejected
  tokens per row, 48 rows, is 192 KiB per step -- 1.9 to 9.4 MiB/s, about 0.00005% of the H20's ~4 TB/s. And the
  dependency forbids the useful version: verify attends over the draft tokens themselves, so their KV must be
  resident during verify and the commit cannot be deferred past its use. What remains is a scratch write plus a
  copy of the accepted entries, which is a wash at best.
- Completes the set of fused-step mechanisms evaluated on measured grounds rather than listed as future work:
  cross-role KV grouping (0 shared pages, <= 1.3% of step time) declined; deferred KV commit declined; draft/verify
  pipelining worth building (17-28% ceiling) but blocked on predicting accepted length per step.

## v3.66.0 — The alpha-6 traced point is void: 6117 of 6117 decode steps carry dkv=0 (2026-10-29)
- Confirmed rather than inferred. That arm launched at 14:37, before the corrected tracer patch went on at about
  15:00, so every decode step in it records a zero KV length -- the defect the trace exists to measure, for the
  second time, exactly as the chain's unconditional DKV_DEPLOYED implied.
- The alpha-3 point is queued behind the pdmux smoke and will launch fresh workers against the patched tree, so
  point 3 waits on that single arm rather than on re-running both.
- Also recorded: those decode lines read et=16384 (extend_num_tokens) on steps with np=0 and no prefill, so that
  field is stale from an earlier extend step rather than reset per step. No fitted coefficient depends on it -- it
  is not among the step model's columns -- but any future feature set reaching for `et` would read a stale value.

## v3.65.0 — Agent-dominant timing is parity (median 1.09x), and the chunk size is why (2026-10-29)
- Three repeats, six reachable captures, spreads 0-4%: 2.12x and 1.71x at batch 16, 1.20x and 0.99x at 32, 0.88x
  and 0.74x at 64 for two- and four-mode. Overall median 1.09x, monotonically declining with batch. On the
  90%-chat set the same build measured 1.15-7.44x, so correcting the composition removed almost all the gain.
- The cause is not the mix defeating the mechanism: these generated steps carry a 512-token chunk on a 4096-token
  context, so with 75% of rows at 64K-200K essentially all work is long-KV decode and verify input -- the regime a
  single FA3 call is built for, with almost no tile padding to remove. Our narrower classes walk the same long KV
  less efficiently as the number of long rows grows.
- The sets vary row KV spread while holding the other determinant of the mechanism, the chunk's share of the step,
  fixed at an unrealistic value. The trace-sampled 1.86-2.92x is unaffected: those are real serving steps where the
  chunk carries thousands of tokens and both roles are genuinely present.
- Next run: agent-dominant rows WITH realistic chunk sizes (2048-16384 tokens over a long prefix), which is what a
  four-mode serving step is. Until then no agent-dominant ratio may be quoted as the kernel's gain on our workload.

## v3.64.0 — The decode-KV tracer fix never applied; the chain claimed it had (2026-10-29)
- Caught before the traced arm took the cluster lock. The patch anchors on a multi-line `_sl = (...)` block that
  the deployed tracer does not contain -- it reads `seq_lens = ... if ... else []` on one line -- so the assertion
  fired on all four nodes. But the chain printed DKV_DEPLOYED unconditionally and queued the arm, which would have
  recorded dkv=0 on every decode step for the second time and surfaced only in the fit afterwards.
- apply_steptrace_decode_kv_v2_20261029.py targets the deployed text with the same shape the verify-rows patch uses
  for the running batch. Applied and verified on both trees on all four nodes by marker count and compile():
  dkv=1 verify=1 COMPILES everywhere.
- Lessons recorded: a deployment step must verify a marker rather than trust an exit code or a following echo, and
  a patch anchor is a property of the tree it targets, so it must be re-read against the deployed revision before
  being chained into a long run.

## v3.63.0 — Agent-dominant correctness gate: every reachable size exact, one unreachable size fails (2026-10-29)
- The 75%-agent set passes the FP32 gate at batches 16, 32 and 64 in both two- and four-mode (worst 0.0035 against
  FP32, 0.0047 against sglang's own call).
- Batch 128 produces non-finite output. The first reading -- that merge_groups=0 meant partials went unreduced --
  was wrong: the plan dump shows fa2_chunk 448512 against a longest KV of 199670, so FA2 declined to split and
  there was nothing to reduce. The real point is that this step needs 14801057 KV tokens, 2.21x the server's
  6696000-token pool, so it cannot occur in our configuration; batch 64 sits at 0.89x and is exact.
- Both batch-128 captures moved to synth_ag/_unreachable/ so the timing sweep does not spend three repeats on a
  non-finite arm and no result can be quoted from it. The cause of the NaN in an unsplit FA2 pass over ~200K tokens
  is recorded as an unchased limit beyond the pool, with the arithmetic that bounds it.
- Harness lesson recorded: the generator takes batch sizes from a plan list and knows nothing about the KV pool, so
  generated sets must be checked against POOL_KV_TOKENS before being measured.

## v3.62.0 — Three repeats: spread is 0-16%, and two of my own methodology claims were wrong (2026-10-29)
- Three repeats at 200 iterations, keyed by (set, capture): heterogeneous 90/10 mix 1.15-7.44x median 3.64x over
  sglang's single attention call; uniform short input 0.60-1.27x median 0.86x. Within-configuration spread is
  0-16%, usually under 10%.
- CORRECTION 1: the "40% pass-to-pass variation" reported earlier was an artefact of my aggregation, which keyed on
  capture filename -- and the two-mode and four-mode sets reuse filenames, so it averaged two different workloads
  and called their difference variance. The pessimistic "no single-pass number is trustworthy to a third of its
  size" does not apply to this harness, though it does stand for the v3 arm in the earlier knob sweep where the
  8-34% spread was measured correctly.
- CORRECTION 2: 40-iteration runs overstate our arm by up to 40% (longtail four-mode batch 64: 0.774 at 40
  iterations against 1.086 +-2% at 200, with the only changed knob provably inert for four-mode work). The first
  kernel table in the task doc, taken at 40 iterations, is superseded and no 40-iteration number should be quoted.
- All of this is still the 90%-chat mix; the agent-dominant sweep is queued and is what belongs in the paper.

## v3.61.0 — The generated mix was inverted: agent inputs must dominate (2026-10-29)
- The `longtail` set is 90% chat inputs with 10% agent inputs of 64K-200K. The evaluated workload has agent inputs
  dominating at over 70% (user). The share of wide-KV inputs is exactly what decides how much tile waste there is
  to remove, so every ratio measured on `longtail` is measured on a composition we do not claim -- understating the
  heterogeneous case and over-weighting the uniform one.
- add_agent_dominant_20261029.py adds `agentmix`: 75% agent inputs of 64K-200K, 25% ShareGPT, identical in every
  other respect to `longtail` so the two are directly comparable. Generated median KV is 117K-162K against
  `longtail`'s ~1K.
- Queued ahead of the default-grid control, with a correctness gate first, since a 75%-agent mix carries far more
  split-KV work than anything gated so far. Three repeats.

## v3.60.0 — The corrected cap sweep is complete: seven caps, all 19-33 points down (2026-10-29)
- cap16000 lands at chat 37.8% / agent 64.2%, flat against cap8000's 37.7% / 62.9% and still 23.7 points of chat
  attainment below the control's 61.5%. Acceptance 1.926, failed=0.
- Seven caps from 1000 to 16000 ms -- a sixteen-fold range -- all sit 19-33 points below the control while
  improving mean TPOT 11-28%, non-monotonically. At 16000 ms the cap exceeds any plausible step time, so what
  still binds is the budget rather than the cap, which is the direct evidence that the loss comes from sizing the
  chunk from a latency budget on this workload rather than from a mistuned cap.
- This closes the question the sweep was run to answer and supports the earlier decision to report the chunk layer
  with its measured regime rather than as a general mechanism.

## v3.59.0 — Full generated-capture sweep: 3.92x median on heterogeneous input, 0.83x on uniform (2026-10-29)
- All four generated sets at the tuned 16-row grid, 200 iterations, unfiltered output. Heterogeneous input
  1.13-6.70x median 3.92x over sglang's single attention call; uniform short input 0.57-1.36x median 0.83x, i.e.
  about 17% SLOWER. Against v3 the dispatcher loses almost everywhere (median 0.69x).
- The uniform-input loss is recorded rather than softened, with the two properties of these generated sets that
  decide what it means: the prefill chunk is a fixed 512 tokens on a 4096-token context, against 2048-16384 tokens
  over 32K-256K in serving, so FA3's share is small and the plan plus second launch are amortised over very little
  work; and batches of 64-1024 sit far above the 48-request cap speculative decoding imposes. The reading is that
  the gain comes from tile-width heterogeneity, and a step with neither heterogeneity nor a chunk large enough to
  amortise the plan gives the mechanism nothing to earn.
- Three repeats of the sweep are running. Pass-to-pass variation has been seen at 13% and once at 40% on a point a
  provably inert knob could not have moved, so no row is quoted without bars.

## v3.58.0 — Compiling the in-launch merge out deadlocks the kernel (2026-10-29)
- UNIFIED_SKIP_MERGE=1 does not produce a timing probe; it hangs. The GPU sits at 100% utilisation while the host
  sleeps on one thread -- a host blocked in a CUDA sync behind a kernel that never completes. Two attempts hung,
  one for 16 minutes.
- The FA2 launch's synchronisation depends on the merge completing, so removing the merge leaves the ticketed
  handshake unsatisfied. The merge is structurally integral to the one-launch design rather than a stage bolted on
  to it, which is a point worth making in the paper about why the fusion lives in the plan and the merge.
- Measuring the merge's cost therefore needs a build whose merge body is a no-op that still signals the handshake,
  which is a kernel change rather than a flag. Candidate 2 for the v3 gap is closed by this route.
- Harness mistake recorded: the first attempt piped the driver's output through grep, which hid both the hang and
  any error. A probe's output must not be filtered until it is known to terminate.

## v3.57.0 — Draft/verify pipelining ceiling is 17-28%; the blocker is a data dependency (2026-10-29)
- Computed the draft share of decode step time from the fitted step model, which separates the speculative terms
  (nd*S for the S=3 draft forwards, nd*T and dkv*T for verify), so no GPU time was needed. Draft is 22.7% at 8
  decode inputs rising to 28.3% at 48, and falling to 16.8% at 120K context because verify's KV reads scale with
  context while the draft's do not.
- That is the ceiling on pipelining, and it is more than an order of magnitude above the cross-role KV reuse
  ceiling of 1.3%, so this mechanism is worth considering on size where that one is not.
- The blocker is a true data dependency: verify needs the draft's tokens within a step, so pipelining is only
  available across steps, and the draft for step N+1 needs step N's accepted prefix. Breaking it requires drafting
  from a predicted accepted prefix, and at the measured acceptance of about 2.0 of 4 tokens the accepted length is
  uncertain enough that wasted draft work eats into the ceiling.
- Caveat recorded with the numbers: the model's 25% all-step error makes the shares indicative of magnitude, though
  the ordering follows from which terms carry KV length rather than from the fit's precision.

## v3.56.0 — Cross-role KV page overlap is zero on the realistic replay (2026-10-29)
- Settles the number that gates cross-role shared-prefix KV reuse, with no GPU time, by reading the workload
  generator instead of instrumenting a step -- a tracer copying per-step page sets would perturb the step times
  the traced arm exists to measure.
- Two roles share a page only if they share a token prefix. Across sessions the replay draws each session's prompt
  independently from the full vocabulary, so two sessions agree on their first token with probability 6.6e-6 and a
  page needs every token to match: zero pages. Within a session rounds reuse tokens verbatim, but round N+1 is
  issued only after N finishes, so the prefilling request is never from the same session as a resident decode
  input: zero pages.
- Consequence recorded as a design decision rather than future work: the mechanism has nothing to reuse on the
  workload the headline results use, and even on a fan-out workload built to maximise page overlap it is bounded
  at <= 1.3% of step time, because compiling the prefill role's KV loads out entirely is worth only 0.1-1.3%.
- tasks/perf_model/cross_role_page_overlap.py states the arithmetic.

## v3.55.0 — The pdmux fix broke CUDA graph capture; guarded (2026-10-29)
- The spec-metadata fix forced a plan on the resolved per-stream backend whenever pdmux was on. Planning copies
  plan data host-to-device and CUDA graph capture rejects an unpinned host source, so the draft model's capture
  died with "Cannot copy between CPU and CUDA tensors during CUDA graph capture". The readiness mark had previously
  made the forward skip planning during capture, which is why the fault appeared only after the fix.
- apply_pdmux_capture_guard_20261029.py adds `not torch.cuda.is_current_stream_capturing()`. Capture has its own
  planning path -- the decode graph runner plans each per-stream backend as it captures -- so the forced plan
  belongs only on the eager forward. Deployed and compiling on all four nodes.
- Recorded that "0/16 workers healthy after 40 min" has now had three distinct causes: an import-time SyntaxError
  in our own patch, a backend/metadata mismatch, and an illegal copy inside graph capture. The symptom carries no
  information; each cause was found only by reading a worker log.

## v3.54.0 — The 16-row class's grid is under-tuned; v3's own timing is noisy (2026-10-29)
- Sweeping UNIFIED_FA2_CTAS_PER_SM16 over 1/2/4/8 on the heterogeneous two-mode captures: four resident CTAs per SM
  beats today's default by 17% / 12% / 5% / 2% as the batch grows, which is the expected shape -- the 16-row
  class's grid matters most when there is least other work to hide an under-filled launch behind. The default
  should move. It narrows the gap to v3 from 0.63x to 0.92x at batch 16 but does not close it (0.71-0.84x above).
- Recorded a methodological limit that applies to every small-batch number in this harness: v3's own time varies
  8-34% between passes, and the knob under test cannot affect v3, so that spread is run-to-run noise. The gap to v3
  is real because it reproduces in every pass, but its magnitude at batches 16 and 64 is known only to about a
  third of its own size. The first table's 0.61x should be read with that spread.
- The FA3-split hypothesis stays refuted; the in-launch merge is built as a separate extension and chained to bench
  as a timing probe only, since output is incomplete with the merge compiled out.

## v3.53.0 — A6's FA3 KV split is why it loses to v3 on heterogeneous steps (2026-10-29)
- Confirmed the regression at 200 iterations with the workspace pre-grown: A6 runs at 0.61-0.83x of v3 on the
  heterogeneous two-mode captures, while both beat sglang's single attention call by 2.0-7.0x. Correctness is
  unaffected (18/18 FP32 gate).
- Cause read off the plan: the planner splits the FA3 chunk until FA3 alone has `fa3_ctas` items, hardcoded to the
  SM count. On a 4096-token context that splits a small chunk three ways (chunk 1664, 336 items, 12-56 merge
  groups, ~0.5 MB of partials) to avoid idle SMs -- ignoring the 188-612 FA2 items running concurrently under PDL,
  which already fill the machine. The policy was tuned on 32K-256K contexts where splitting is necessary.
- apply_fa3_cta_target_20261029.py exposes UNIFIED_FA3_CTAS without changing any default; a 132/64/16/1 sweep is
  measuring how much of the gap the split accounts for, with the 132 arm reproducing the baseline as its control.

## v3.52.0 — Kernel gains measured on mixed input: 1.23-7.51x where input is heterogeneous (2026-10-29)
- The A6 unified dispatcher now builds and runs on the H200 trial. The build needed flash-attention's hopper
  headers, which were absent; the register gate had never revealed it because that gate compiles flashinfer's own
  FA3 mainloop rather than the upstream one.
- Correctness: 18 of 18 generated mixed-step captures PASS the FP32 gate, worst error 0.0030-0.0036 against an FP32
  reference and 0.0045-0.0048 against sglang's FA3 call. Four-mode steps place verify inputs on the 32-row class
  and two-mode steps place decode inputs on the 16-row class, with FA3 partials merged in-launch.
- Timing against the single attention call sglang runs today: 1.23-7.51x (median 3.15x) on heterogeneous input
  (90% ShareGPT plus 10% agent inputs of 64K-200K), and 0.75-1.42x (median 1.20x) on uniform short input. The gain
  tracks heterogeneity, which is the mechanism stated as a prediction and then confirmed -- where every input has
  nearly the same width there is no tile waste to remove.
- Three caveats recorded rather than glossed: A6 loses to v3 on three heterogeneous rows (0.36-0.78x), which
  contradicts the standing "A6 cannot lose to v3" expectation and is a defect to chase; the host plan costs 5-10 ms
  against a 0.6-3.8 ms kernel in this harness; and batches of 256-1024 are above our serving regime, which caps
  max_running_requests at 48 under speculative decoding.
- Paper: C1 restated as a unified tile dispatch kernel implementation that maximizes compute efficiency by removing
  tile waste for mixed input, and the abstract and contributions now name all four inputs (prefill, decode, draft,
  verify) instead of three, dropping the kernel-internal word "rows".

## v3.51.0 — Item 18's register story was measured against the wrong baseline (2026-10-29)
- The user asked why the guest-free arm spills at all, since a 16-row tile is cheaper in registers than a 64-row
  one. Both halves of that were right, and chasing it found the real baseline error.
- "Guest compiled out" is FLASHINFER_FUSED_DRAIN_NOOP=1: it deletes the three Runner::Run call sites but keeps
  replace mode's CTA shape and its explicit warpgroup_reg_dealloc<24> / warpgroup_reg_alloc<240>. With no guest the
  spill is 16 B, i.e. nothing; adding the 16-row guest costs 148 B more, which is small for exactly the reason
  given -- a 16-row tile is cheap.
- The correction: quoting 164 B against the guest-free 16 B framed the guest as a cost to justify. Against the
  STOCK q64 path (same TU, bodies out, no piggyback) the piggyback build is strictly BETTER: 168 registers against
  144, and 164 B of spill against 1560 B with a 536 B stack frame. Same kernel, same tile, same 256-thread CTA, so
  it is single-axis. There is no register case against item 18 to answer.
- Also killed: that 168 is a hardware cap here. At 256 threads the launch-bound cap is 256; 168 is a chosen
  allocation. The 168 = 65536/384 coincidence belongs to the q128 arm only.
- Shared memory measured on the H200: FA3 at q64 147 536 B + the 16-row class 69 632 B = 217 168 B against the
  232 448 B opt-in maximum, so it fits with 6.6% headroom. Supersedes the estimated 218 160 B.
- Both remaining item-18 questions are now the only ones: is it correct, and is it faster.

## v3.50.0 — All 48 arms re-scored; the SLO-budgeted chunk is a negative result (2026-10-29)
- Rather than convert old numbers into the new metric set, every surviving arm's raw per-request records were
  re-scored, so each reported figure is measured in the metric it is reported in. Comparability was checked, not
  assumed: equal chat request counts, failed=0, and durations within 1016-1083 s (alpha 6) / 1804-2080 s (alpha 3).
  Raw lines kept at tasks/paper_figures/data/rescore_means_20261029.txt.
- Against conditional disaggregation (two baseline repeats per load): alpha 3 +40.4/+43.4 points chat and
  +29.0/+30.2 agent attainment, mean TPOT -69..-79%, mean TTFT -79..-86%, and 65-82% more agent rounds issued;
  alpha 6 +8.3/+13.1 chat and +25.8/+30.6 agent, TPOT -30..-49%, 73-78% more rounds.
- Against sglang's own colocated schedules: chat attainment +7.7 points at alpha 3 and +4.0/+8.0 at alpha 6, chat
  mean TPOT -30%/-7%/-3%, agent mean TPOT -18%/-15%/-14%, chat mean TTFT -42..-63%. The agent-attainment
  differences are inside the ~5-point noise floor and are reported as such; the latency means are the solid part.
- **The SLO-budgeted chunk is withdrawn from the contribution.** Against L0 (same tree and kernel, differing only
  in using sglang's chunk policy) at alpha 6 it improves mean TPOT 19-23% and costs 20.4 points of chat attainment,
  15.6 of agent, a 3.6-3.9x worse mean TTFT and 11.8% fewer agent rounds issued. The whole cap sweep agrees. This
  reverses the 2026-10-28 reading, which was taken on the token rate and the TPOT tail: the tail claim stands and
  was the wrong question, because attainment requires both targets. The four earlier bugs in the rule were real and
  are fixed; this is measured with all four fixed, so it is the policy and not the implementation.
- Paper: the metrics section now defines the three reported quantities and says why no rate is reported, and C3's
  chunk component is stated as a negative result rather than scoped to overload.

## v3.49.0 — pdmux is incompatible with speculative decoding on this base; fixed (2026-10-29)
- Root cause, read off the deployed tree rather than guessed: pdmux holds one attention backend per SM group, but
  `ForwardBatch.needs_forward_metadata_init()` tracks planning readiness per BATCH while metadata lives per backend
  OBJECT. Speculative decoding pre-plans the main or draft backend and marks the batch ready, so the decode forward
  runs the model under a per-stream backend nobody planned and the first attention layer dereferences None. FA3's
  `swa_out_cache_loc` and flashinfer's `decode_wrappers` are the same fault, which is why switching backends never
  helped -- all three pdmux attempts died at launch.
- `apply_pdmux_spec_metadata.py`: under pdmux, plan whenever the resolved backend is not the object the readiness
  mark could refer to. Sound because a pre-planner never plans a per-stream decode backend, so planning one cannot
  disturb pre-planned draft metadata. Deliberately stronger than a `forward_metadata is None` test, which from the
  second step on would pass on the previous step's stale plan and corrupt silently rather than crash. Inert outside
  pdmux. Deployed and compiling on all four nodes; a smoke arm is queued.
- The baseline is fixed rather than run without MTP: pdmux is what the single-node bar is judged against, and
  running it unspeculated while our arm speculates would hold its configuration below ours.
- The scorer no longer exits on an arm whose client predates the start-time field; it reports UNSCORABLE and
  continues. That abort had discarded 25 of 48 arms in a re-scoring pass.

## v3.48.0 — Attainment and mean latency with error bars; no goodput (2026-10-29)
- Reporting changed on the user's instruction: every serving result is SLO attainment plus mean TTFT and mean TPOT,
  each carried by a p10-p99 spread. The SLO-attaining token rate is dropped. It is a third axis moving with
  attainment, latency and the request mix at once, so it made tables harder to read without answering what the
  first two leave open; where capacity itself is the question, the capacity curve answers it. TPOT leads, since we
  claim it matters more than TTFT.
- `tracelab_windows.py` emits the new line; `superset_report.py` reads it and still reads pre-change logs, which
  carry a MEDIAN and a rate. Those are marked `*`, and the reader now REFUSES to compare a median against a mean
  anywhere -- in the superset verdict, in the "would beat ours" note and in the publishability bar. The mean of a
  right-skewed latency distribution sits well above its median, so mixing them manufactures differences between
  arms measured either side of the change; the first mixed table produced two such spurious findings.
- The bar's throughput half is no longer judged from this table, which carries no rate; it is read off the capacity
  curve. The latency half is judged on mean TPOT, and attainment gains are stated in points as well as relatively,
  because a metric that saturates near 100 makes a small absolute gain look large in relative terms.
- `test_tracelab_scorer.py` (new, 9 checks, all passing) builds records whose mean TTFT is 5.49 s against a 0.10 s
  median, so a scorer reporting the wrong statistic cannot pass, and checks the scorer-to-reader round trip.

## v3.47.1 — pdmux's third failure was ours: an import above `from __future__` (2026-10-29)
- The per-tenant acceptance patch prepended `import os as _os` to `metrics_reporter.py`, above its
  `from __future__ import annotations`. Python requires every `__future__` import to come first, so the module
  raised SyntaxError and every scheduler on four trees across all four nodes died before opening a port. The driver
  reported "only 0/16 workers healthy after 40 min", indistinguishable at that level from an RDMA or cluster fault,
  and two pdmux retries had already been spent on the attention backend and the memory fraction.
- The repair tool's first gate used `ast.parse`, which does not apply the `__future__` ordering rule at all: it
  reported "already-ok" for all four broken files, so the first repair pass changed nothing and printed success.
  Validation of patched Python now uses `compile()`. Tool kept as `fix_metrics_import.py`.
- The patch now inserts after the last `__future__` line and compiles the result before writing.
- No recorded result is contaminated: every arm that ran before the patch has acceptance ~2.0, full request counts
  and failed=0, checked individually for the demotefix and capsweep arms at both loads. Only the two pdmux arms
  died, and pdmux remains unmeasured -- a failed arm fills no row.
- The scorer no longer aborts on an arm with no records; it reports NO_RECORDS and continues, after one dead arm's
  empty directory stopped a 40-arm re-scoring pass at arm six.

## v3.47.0 — Item 18's register result reproduces bit-for-bit on the H200 (2026-10-29)
- All four arms match the CPU box exactly: q128 control 168 reg / 33 424 B, q128 guest-out 168 / 0 B, replace +
  16-row guest 168 / 164 B, replace guest-out 168 / 16 B. So the toolchains agree and the variant's marginal cost
  of 148 B is confirmed on the target hardware rather than inferred from a CPU-only box.
- Two wrong invocations failed loudly instead of producing numbers, which is what makes the result trustworthy:
  FUSED_PIGGYBACK_REPLACE alone hits an #error because it is a MODE of the piggyback path, and at q128 four
  static_asserts fire on tile, CTA width, register distribution and the shared-memory sum.
- Recorded a number NOT to quote: the default class set at q64 gives 4 432 B, which matches neither the 164 B
  (16-row guest only) nor the 21 008 B (all three classes) arm. The guest hosts one tile class, so DRAIN_ONLY16 is
  the configuration the mechanism needs. Reading 4 432 B as a toolchain discrepancy would have repeated the dkv
  error -- treating a number as a property of the code when it is a property of the inputs.
- Outstanding on this item: the correctness fixture and timing, both of which need a GPU run and can now run here.

## v3.46.0 — H200 trial provisioned and verified; the ladder's real blocker is the capture set (2026-10-29)
- Trial 3f9e6a55ef7ae47e running since 07:20 with all four GPUs loaded so it cannot be reaped again. Verified by
  name: sglang at 3fef129ff8, torch 2.14.1+cu130 on 4 GPUs, flashinfer/sgl_kernel/numpy importing, Qwen2.5-7B plus
  the EAGLE3 head, the serving overlay's EDF chunk chain (2 markers, so OUR tree), port07's fused FA3 header, 4
  prebuilt .so libraries, 92 bench scripts, and fi -> port07.
- Recorded the two traps that cost about 1.5 h: the 100 s exec cap turning a source build into `rc=1` with the
  relay's JSON as output, and px.push's (local, staging, dest-DIR) signature plus bool return, where passing None
  appended 137 chunks to a file named `None` and concatenated with a prior attempt's bytes.
- Corrected the sequence for item 1: the ladder is NOT next, because the capture set was lost with the previous
  trial. Captures must be regenerated by serving first, and the capture script's TraceLab dataset is the one bundle
  entry excluded to halve the push, so it is transferring separately.

## v3.45.0 — Scope C3's chunk claim to overload, per the alpha-3/alpha-6 pair (2026-10-29)
- C3 presented the predicted chunk size as an unqualified part of the worker policy. The measured pair says
  otherwise: against a fixed-chunk control on the same tree it cuts the chat mean-TPOT tail 46% past capacity and
  worsens it about 20% at capacity, because a budget that rarely binds lets the chunk run to its ceiling and
  lengthens the step.
- The paper now claims it as an overload mechanism rather than a general one, with the reason stated. That is
  narrower than before and is what the evidence supports.

## v3.44.0 — alpha-3 no-regression check: approximately met, but the rule's sign flips with load (2026-10-29)
- alpha 3, demote fix, stepcap 2000, accept 1.987, failed=0: chat 88.3% / 1417 tok/s, TPOT p50/p90 13.4/35.7 ms;
  agent 92.2% / 2374 tok/s. Against rung L0 (91.2% / 94.3% / 2493, 10.8/29.8) attainment is -2.9 and -2.1 points,
  both inside the ~5-point noise floor, so there is no collapse and the no-regression requirement is approximately
  met. Goodput is -4.8% and chat TPOT p90 is 20% WORSE, the latter outside the 3 ms floor.
- The sign flips between loads: at alpha 6 the rule cuts the tail 46% against the fixed-2048 control; at alpha 3 it
  worsens it. Mechanically, at capacity the budget rarely binds so the chunk runs to its 16384 ceiling and steps
  get longer than sglang's own chunk policy allows.
- So the rule is an OVERLOAD mechanism rather than a general one -- narrower than C3 currently claims, and the
  paper should say so. It also means the corrected sweep must cover alpha 3 too, and that one global cap may not
  serve both loads, which is the argument for the per-tenant form over a constant.

## v3.43.0 — Rule out the device-name hypothesis for the stock-PD transfer failure (2026-10-29)
- mn_node.sh:281 computes IB=mlx5_1..mlx5_4 and all four are present and ACTIVE on the pods, so the decode side
  does receive a valid --disaggregation-ib-device. The earlier reading -- that the MOONCAKE_PROTOCOL/DEVICE exports
  being guarded by `[ "$MODE" != decode ]` starves the decode side -- does not hold: those exports configure the
  hierarchical-cache store, while the disagg path is configured separately at line 284 with its own device.
- That leaves cross-pod bootstrap reachability as the live candidate, and it matches the signature: an unreachable
  bootstrap port fails per request rather than at startup, so servers report ready, thousands of individual
  transfers fail, and no acceptance line is ever written. The check must be run FROM a decode pod against the
  prefill pods' --disaggregation-bootstrap-port, not inferred.

## v3.42.0 — RDMA is healthy, so the stock-PD transfer failure is configuration, not hardware (2026-10-29)
- The pods expose five InfiniBand devices (mlx5_0..mlx5_4), every port in state `4: ACTIVE`, with 20 RDMA/NCCL_IB
  environment variables set. So the fabric is present and the KV-transfer failure is not a missing-RDMA problem.
- That narrows it to configuration: (a) Mooncake's device/interface selection -- its master log carries no protocol
  line for these arms, where a working transfer names one; (b) cross-pod bootstrap-port reachability, which would
  give a per-request connect failure rather than a startup error, matching the observed 2819 transfer-failed count.
- Revises yesterday's conclusion: stock PD is probably fixable here rather than impossible, so the paper decision
  changes from "report that it could not be run on this cluster" to holding the comparison open. The cheapest test
  is one short arm with Mooncake's device and protocol pinned explicitly.

## v3.41.0 — Stock PD is void on clean upstream too: the tree was never the cause (2026-10-29)
- Both arms fail the driver's own gates: alpha 3 chat 4.3% with failed=1728, alpha 6 chat 13.2% with failed=145,
  accept=NONE on both, TTFT 195-327 s, TPOT 2.7-3.0 ms. Identical signature to r1.
- So the up_clean switch changed nothing and the spec_info-assert hypothesis is refuted. The r1 diagnosis stands
  (2819 transfer-failed errors on the decode pod) and is now attributable to the environment: Mooncake's KV
  transfer does not work on this job. Next step is checking RDMA availability and bootstrap-port reachability, not
  another tree or parameter.
- Consequence: stock PD may not be measurable on this cluster. The honest options are to report only against
  conditional disaggregation (valid, failed=0, both loads) and say so, or to obtain a job where the transfer works.
  The pre-correction numbers are not an option -- mis-configured model and, on this evidence, the same broken
  transfer.
- The conditional-disagg arms here are valid (accept 1.992/2.017) and repeat r1, but alpha-3 agent goodput differs
  15% between repeats (905 vs 771) against a stated 4% noise floor, so a third repeat is needed before quoting.

## v3.40.0 — The per-tenant acceptance patch targeted a file no pod has (2026-10-29)
- It patched `managers/scheduler_metrics_mixin.py`, which is where the acceptance line lives in THIS checkout. On
  the deployed trees (sglang 3fef129) that file does not exist at all -- all three carry the line in
  `managers/scheduler_components/metrics_reporter.py:979`. So the patch committed earlier as "ready to deploy"
  would have silently failed to apply on every pod.
- Fixed to try both paths, require that the chosen file actually contains the acceptance log line, and assert
  loudly when neither is present instead of reporting success. Found by deploying it and reading the errors rather
  than trusting the earlier dry-run, which had only ever run against the local layout.
- Lesson consistent with tonight's others: a patch verified against the local checkout proves nothing about the
  deployed tree.

## v3.39.0 — Withdraw 5.2.2's decode row: it was fitted on an absent feature (2026-10-29)
- The per-step-type table reported decode-step errors of 34-38% (linear), 18-23% (MLP) and 21-25% (GBT), and the
  text read that spread as the linear model's missing nonlinearity. Both are withdrawn: the trace records each
  row's KV length from a host copy of the sequence lengths that the decode path does not populate, and the
  fallback recorded zero, so all 73 770 decode steps carry a zero KV length against 10 189 of 14 162 mixed steps
  carrying a real one.
- So that row measured models predicting a decode step from its row count alone, without the quantity that
  dominates decode attention cost, and the separation between model families there is an artefact of fitting
  different functions to a dead input.
- The merged-step row stands and is the one the chunk rule depends on, since it sizes a chunk for a merged step.
  No decode-step or TPOT prediction-accuracy claim appears until the tracer fix is deployed and the row re-measured.

## v3.38.0 — Stage the decode-KV tracer deploy and a traced arm for point 3 (2026-10-29)
- `/tmp/deploy_dkv_and_trace.sh` chained after the queue drains: patches both trees an arm can run on
  (h20py/python and python_slo_life), asserts each still parses, then runs one short traced arm with
  SGLANG_STEP_TRACE=1 so decode steps finally carry dkv.
- Deliberately NOT applied mid-chain. A running server has already imported scheduler.py so patching cannot
  disturb it, but the next arm launches fresh workers and would pick the patch up, leaving two arms in one driver
  with different instrumentation -- the same class of mistake as patching a shared tree while arms are in flight.
- After that trace lands: refit with fit_step_mlp_20261028.py and re-measure the decode-step signed bias with
  tpot_bias.py on a chronological split. Only a bias surviving that is evidence against the perf-model premise;
  until then point 3 stays UNMEASURED rather than refuted.

## v3.37.0 — Fix the tracer so decode steps record their KV length (2026-10-29)
- `tasks/pd_colocation/apply_steptrace_decode_kv_20261029.py`: where `seq_lens_cpu` is absent -- which it is on the
  decode path -- the tracer fell back to an EMPTY list, so every row's KV length read as 0 and `dkv` accumulated 0
  on all 73 770 decode steps. It now falls back to the requests themselves (prompt + output tokens is exactly a
  decode row's KV length), which adds no device synchronisation and so does not perturb the measured path.
- Matched by regex on the captured indentation rather than a hard-coded column: the first version hard-coded one
  and failed against a fixture built from the deployed tracer block, which is the silent-failure shape that has
  cost arms repeatedly. Verified to apply, parse and be idempotent on that fixture.
- This is what unblocks point 3 of the four-points doc. Order after applying: re-trace one arm with
  SGLANG_STEP_TRACE=1, confirm DECODE lines carry dkv > 0, refit, then re-measure the decode-step SIGNED bias on a
  chronological split. Only a bias surviving that is evidence against the perf-model premise.

## v3.36.0 — The decode bias is a DEAD INPUT: dkv is zero on every decode step (2026-10-29)
- Counted over the alpha-3 traces: DECODE 73 770 steps with 0 having dkv > 0; EXTEND 14 162 steps with 10 189
  having dkv > 0 (up to 100 375). The tracer records the decode rows' KV length on mixed/prefill steps and never on
  decode-only steps, so the step model's dkv terms contribute nothing there and decode step time was predicted from
  the row COUNT alone -- without the dominant term in decode attention cost.
- That explains the 29% MAPE and -13% bias. Control: the same correction fitted on a RANDOM half gives a held-out
  median bias of +0.1% against -12.3% chronologically, so the model is unbiased in the median within one
  distribution and the chronological gap is covariate shift (GPU time per decode step grows 1.31x across the run).
- Two earlier conclusions withdrawn: the MLP being worse (-19.9%) was a nonlinear fit to noise in a feature set
  with the key input zeroed, and "more model does not fix this" was wrong -- no model can fix an always-zero
  feature.
- Point 3 of the four-points doc goes from REFUTED to UNMEASURED, which is the honest state. Fix order: populate
  dkv on decode steps, re-trace one arm, refit, re-measure the signed bias on a chronological split.

## v3.35.0 — Refresh the four-points evidence table in pd_colocation_vs_disagg.md (2026-10-29)
- The table was dated 2026-10-21 and predated every 16-GPU TraceLab replay arm. Rewritten with what each point now
  rests on:
  - Point 1 gains the strongest evidence yet: +40.7 points chat attainment, 2.75x agent SLO goodput and -85% chat
    mean-TPOT p90 against conditional disaggregation at alpha 3, both arms failed=0 at matched acceptance.
  - Point 3's premise is REFUTED on the quantity that matters: decode-step prediction carries a -13.0% signed bias
    no scalar calibration removes and a nonlinear fit makes worse (-19.9%), so predicted TPOT is ~13% optimistic.
    It must come out of the contribution list until a better-featured model is measured.
  - Point 4 becomes a timescale argument supported by measurement: the P/D demand ratio moves 2.2x inside a surge
    and 3.4-4.2x within a run, while re-partitioning takes tens of minutes.
  - Point 2 unchanged.
- Records that two arms the older sections lean on are VOID, and that both failures flattered us: stock PD (2819
  KV-transfer failures) and pdmux (every worker failed to launch on an FA3-backend incompatibility).

## v3.34.0 — The demote fix works, and it invalidates the flat sweep (2026-10-29)
- alpha 6, stepcap 2000, accept 2.010, failed=0. Excluding past-rescue rows from the budget minimum against the
  identical arm with them clamping: chat attainment 35.3 -> 41.1% (+5.8 pts), agent 56.4 -> 63.5% (+7.1), agent SLO
  goodput 1150 -> 1463 tok/s (+27%), chat TTFT p50 4.00 -> 2.04 s (-49%), at a 12.5% worse chat TPOT tail (55.0 ->
  61.9 ms). The intended trade: the clamp was buying tail by starving prefill. The death spiral is broken.
- The stepcap sweep was flat BECAUSE of this bug -- with most rows pinning the step to ~90 ms the cap could not
  bind, so 600-3000 ms all looked alike. It must be redone, and at HIGHER values, since the fix moved the arm
  toward throughput and the remaining gap to the fixed-2048 control is attainment and goodput, not tail.
- Against that control the rule is still behind on attainment (41.1 vs 60.5%) and goodput (1463 vs 1976) and well
  ahead on the tail (61.9 vs 114.6 ms, -46%). It now trades along a real curve instead of collapsing.
- `mn_capsweep2_16g_20261029.sh` queued: caps 4000 / 8000 / 16000 ms at alpha 6, chained after pdmux.

## v3.33.0 — pdmux root-caused: FA3's backend is incompatible with it, not a memory limit (2026-10-29)
- The arm directory has no worker log at all, only bindfail1..5 for every GPU, and the bindfail log ends
  `AttributeError: 'NoneType' object has no attribute 'swa_out_cache_loc'` at
  `swa_loc=self.forward_metadata.swa_out_cache_loc`. So forward_metadata is None inside the FA3 backend on the
  pdmux path: pdmux runs decode and layer-split prefill on separate SM-group streams and the per-forward metadata
  is not initialised there.
- MEM_FRAC was never the cause, which is why 0.70 and 0.55 failed identically. Two parameter guesses cost two arms
  when the bindfail log would have said this the first time -- the lesson is to read bindfail logs before changing
  a parameter.
- `mn_pdmux_backend_16g_20261029.sh` queued: pdmux on sglang's default backend (PDMUX_ATTN, default flashinfer)
  instead of the forced ATTN_BACKEND=fa3 every other arm uses. Fair for a baseline whose point is the SM
  partitioning rather than the attention kernel, and the paper will say pdmux is measured on its own backend.

## v3.32.0 — Apply the demote fix and launch its arm at both loads (2026-10-29)
- A row that cannot reach its target is now EXCLUDED from the budget minimum rather than given the tightest bound
  in it. The old behaviour made overload self-reinforcing, and the flat stepcap sweep is the evidence: across
  600-3000 ms the chunk barely moved because at alpha 6 most rows are behind, so nearly every step was pinned to
  the bare ~90 ms allowance whatever the cap said.
- Offline test extended to lock the fix in rather than the bug: the past-rescue row must NOT clamp the others, and
  a batch of ONLY past-rescue rows must still leave the tail cap in force so something still bounds the step.
  9/9 pass.
- Deployed to all four pods with `cp -aL` (the main tree is a symlink; `cp -a` would patch it), verified identical
  md5 6e1e53057f110ad4 and that the result parses.
- `mn_demotefix_16g_20261029.sh` launched at alpha 6 and alpha 3 with stepcap=2000, the best of the swept caps.
  alpha 6 tests whether the death spiral is gone; alpha 3 tests the no-regression requirement against rung L0.

## v3.31.0 — The stepcap sweep is flat: the demote branch binds, not the cap (2026-10-29)
- alpha 6, caps 600/1000/1500/2000/3000 ms, all failed=0: chat TPOT p90 stays 55.0-56.1 ms and chat TTFT p90 stays
  31-42 s across the whole range, with attainment 23.6-35.3% against the fixed-2048 control's 60.5%. A 5x change
  in the cap moves nothing, so stepcap is not the binding constraint and the registered plan ("sweep the value")
  is refuted -- there is no value to find.
- What binds is the `demote` branch, the fourth instance of the same error: the budget is a MIN over rows, and any
  row that cannot reach its target gets the bare per-step allowance (~90 ms). At alpha 6 that is most rows, so
  nearly every step is pinned there whatever the cap.
- It is a death spiral: a row falls behind, clamps the step, the chunk collapses, prefill starves, more rows fall
  behind. The mechanism meant to protect latency under overload is disabled BY overload.
- Fix identified and one line: a row that has already missed cannot be rescued, so it must be EXCLUDED from the
  minimum rather than given the tightest bound in it -- which is what "demote" always implied.
- Honest paper statement until that is measured: the rule protects tail TPOT (-51% against a matched control) and
  cannot yet be operated at overload without starving prefill. Mechanism validated, policy not.
- pdmux failed again at both loads (accept=NONE), so MEM_FRAC was not the cause. Two parameter guesses have failed;
  next step is reading the worker logs, not a third guess.

## v3.30.0 — Fix the abstract edit: no bold, and no dangling clause (2026-10-29)
- The previous commit put a bold span into the abstract, contradicting the standing instruction that abstracts are
  not typeset with emphasis, and left "57% more under shifting prefill-decode demand" dangling from the clause it
  replaced.
- Rewritten as two sentences: 2.75x the agent SLO goodput of conditional disaggregation at 40.7 points higher chat
  attainment and 85% lower chat mean-TPOT at the tail; then the shifting-demand and mixed-chunk figures. No bold
  spans remain in the abstract.

## v3.29.0 — The abstract understated the conditional-disaggregation margin (2026-10-28)
- It quoted +12-37% agent SLO goodput, which is the FAN-OUT figure, while the realistic replay measures 2.75x at
  40.7 points higher chat attainment. Fan-out gives every program of a task the same prefix, so the baseline's
  local-append rule keeps the work local and it nearly matches us; that workload understates the comparison.
- The abstract now leads with the replay number. The fan-out figures stay in 6.4, where the contrast between the
  two workloads is the argument for why per-session history is what separates the designs.

## v3.28.0 — Consistency check after tonight's edits: one duplicate figure number fixed (2026-10-28)
- Ran the renumbering tool in dry-run over both sequences after ~25 separate edits to the draft. Tables are clean
  (identity mapping 1-21, every reference resolves) and all seventeen numbered sections are present.
- Figures had two "Figure 25": section 6.5's per-tenant attainment figure and the per-contribution leave-one-out
  figure I inserted into section 7. Renumbered section 7's block up by one, so figures now run 1-29 with every
  reference resolving.
- Draft is 30 114 words, up from 24 342, because the annotated reference list came back (3 435) and tonight added
  the conditional-disaggregation table, the motivation rewrite and the baseline lists. Body text is about 26.7k.

## v3.27.0 — Paper 6.4 leads the disaggregation comparison with the replay result (2026-10-28)
- Added the alpha-3 conditional-disaggregation table: chat attainment 91.2% against 50.5% (+40.7 points), agent
  94.3% against 64.9%, agent SLO goodput 2493 against 905 (2.75x), chat mean-TPOT p90 29.8 against 193.8 ms
  (-85%), agent p90 38.0 against 188.2. Both arms completed every request at matched acceptance.
- States why this dwarfs the fan-out margin, because that contrast is the argument rather than a caveat: fan-out
  gives every program of a task the same prefix, so a local-append rule keeps the work local and the baseline
  nearly matches us; in the replay each session carries its own 119K-token history, so the append is local but the
  history must still be where the decode is. The cost 3.4 predicts is visible only on the workload with
  per-session state -- the one agentic serving actually produces.

## v3.26.0 — Conditional disaggregation at alpha 3: valid, and our margin is far larger on the replay (2026-10-28)
- cond_p1 at alpha 3, accept 2.003, failed=0 on both tenants (passing the gates the stock-PD arm failed): chat
  50.5% / 665 tok/s with TPOT p50/p90 44.8/193.8 ms; agent 64.9% / 905 tok/s, not_issued 6148.
- Against rung L0 on the identical workload and cluster (91.2% / 94.3% / 2493 tok/s, chat TPOT 10.8/29.8 ms):
  +40.7 points chat attainment, +29.4 agent, 2.75x agent SLO goodput, chat mean-TPOT p90 -85%.
- Why this dwarfs the fan-out margin (+12-37%): fan-out gives every program of a task the same prefix, so
  conditional disaggregation's local-append rule keeps most work local and it nearly matches us. On the TraceLab
  replay each session carries its own long history (median 119K cached tokens), so appends are local but the
  prefix must still be where the decode is -- exactly the mechanism motivation 1 claims. This is the workload that
  separates the designs, and it is the realistic one.
- Caveat recorded: the acceptance mean rests on n=15 samples against ~1000-5800 elsewhere. failed=0 and 14 400 /
  6 924 requests served mean the score stands, but 2.003 is thin and must not be used to argue about acceptance.

## v3.25.0 — Queue stock PD on clean upstream, the documented configuration (2026-10-28)
- `tasks/pd_colocation/mn_pdre_clean_16g_20261028.sh`: the r1 arms re-run with SGL_PY=up_clean/python instead of
  up/python. mn_node.sh's header states stock PD requires clean upstream because the serving overlay's spec-v2
  changes trip disagg prefill's spec_info identity assert, and r1 used the overlaid tree.
- The script carries the r1 evidence and, more importantly, two pre-trust checks in its header, because a score
  alone cannot distinguish a working arm from r1's: `accept=` must not be NONE (a failing transfer should still
  leave acceptance lines, so NONE means nothing decoded or MTP is absent, and an arm without MTP is not the
  baseline), and failed / lost_after_failure must both be 0.
- Written as a new file and chained last, so no driver with arms in flight is patched.

## v3.24.0 — Broken PD arm root-caused: the KV transfer fails, not the rope override (2026-10-28)
- Decode pod logged 2819 x "transfer failed for request" plus KVTransferError(bootstrap_room=...) aborts, while
  the prefill pod's Mooncake store set up and warmed successfully and all workers carry rope_theta 1000000. So
  prefill computes the KV and cannot hand it to decode -- that is the failed=1890 / lost_after_failure=7470. The
  arm measured a broken transfer path, and the rope override is exonerated.
- Next in order: (1) switch the driver from SGL_PY=up/python to up_clean/python, which mn_node.sh's own header
  says stock PD requires because the serving overlay trips disagg prefill's spec_info assert -- the documented
  configuration and the cheapest hypothesis; (2) if transfers still fail, check decode-to-prefill bootstrap port
  reachability and whether RDMA is available on THIS job, since the published disagg numbers came from a different
  one.
- Noted what the root cause does NOT explain: accept=NONE. A failing transfer should still leave acceptance lines
  from whatever decoded, so either almost nothing decoded or stock MTP is not running on this tree -- check for an
  MTP startup line before trusting a re-run.

## v3.23.0 — The rope-corrected stock-PD arm is BROKEN, and its numbers flatter us (2026-10-28)
- alpha 3 stock disagg reported chat 4.9% / 78 tok/s with TTFT p50/p90 189.98/311.20 s, failed=1890,
  lost_after_failure=7470, TPOT 2.9/5.9 ms and accept=NONE. Four signs it is broken rather than slow: thousands of
  failed and lost requests; no acceptance line at all despite stock MTP being configured; TTFT three orders of
  magnitude above target; and implausibly FAST TPOT, the signature of requests that emit almost nothing before
  dying.
- Flagged prominently because this is the most dangerous failure shape yet: the broken numbers are hugely
  favourable to us, and "stock PD reaches 4.9%" would pass a casual read into the paper. New rule recorded: an arm
  with non-zero failed / lost_after_failure, or accept=NONE while MTP is configured, is void whatever its score.
- Diagnosis order recorded: decode-side OOM and mooncake KV-transfer errors first; then two driver-specific
  candidates -- the ROPE_THETA override now passed to the upstream tree (a malformed override killed all 16 workers
  once already), and that mn_node.sh documents stock PD needing CLEAN upstream (up_clean) while this driver passes
  SGL_PY=up/python.
- Consequence: the disaggregation headline has no valid measurement at either load, since the pre-correction arms
  carry the dropped-rope_theta caveat. This is now the biggest hole in the evaluation.

## v3.22.0 — Motivation 2 restated as a timescale mismatch, with both sources of evidence (2026-10-28)
- The outline's second motivation said "the ratio is a static configuration and the workload is not", which asserts
  rigidity without saying why it is fatal. Restated as the control-loop argument: the uncached-input-to-output
  ratio moves at minute scale while re-partitioning roles takes tens of minutes, so a static split is a loop whose
  actuator is an order of magnitude slower than its signal.
- Carries both kinds of evidence, each labelled: our own in-run measurement (median ratio halves inside the chat
  surge, 38.4 vs 85.1; 3.4-4.2x spread with no surge, from pd_ratio_20261028.py) and independent trace analysis at
  fleet scale (peak-to-mean up to 4.7x at minute timescales, median 24.5x hourly within a day). Ours is causal and
  in-run; theirs establishes generality.

## v3.21.0 — Crossflow's methodology read from the paper: rate-only load axis, trace-analysis evidence (2026-10-28)
- Verified rather than recalled: "A load factor alpha divides every timestamp offset from the first arrival by
  alpha, increasing rate without changing the trace's temporal shape." So its load axis is only the rate and it
  does not independently vary the P:D mix, as suspected.
- Its P/D-variation evidence is trace analysis on the SAME quantity we measured (uncached input to output ratio):
  peak-to-mean up to 4.7x at minute timescales, and a median 24.5x hourly span within a single day over 2,959
  hours of TraceLab. And it frames rigidity as a timescale mismatch -- the ratio moves at minute scale "while
  reassigning a replica takes tens of minutes" -- which is sharper than our phrasing and worth adopting.
- How to use it without borrowing: cite theirs as independent evidence the phenomenon is general, report ours as
  in-run and causal (median ratio halves inside our surge window, 38.4 vs 85.1; 3.4-4.2x spread with no surge),
  and keep the claim ours -- a per-step bound corrects at step granularity, faster than the disturbance.
- Tooling note: pypdf works via `pip install --target` plus PYTHONPATH; a plain install silently lands elsewhere.
  8 of 11 downloads were truncated, and a truncated PDF still has a valid %PDF header -- verify by extracting
  page 1. Re-download with retries running.

## v3.20.0 — All 11 reference PDFs on disk; text extraction still missing (2026-10-28)
- `/opt/tiger/papers/` holds 11 arXiv PDFs, outside the repo so no binaries are committed. Each was verified by its
  %PDF magic bytes, since a wrong id returns an HTML error page with exit status 0.
- pdftotext is absent and `pip install pypdf` reports success while the module stays unimportable, so bulk grep is
  not available. The Read tool reads PDFs natively by page range, which is the reliable path but costs context.
- Recorded what to read first and why: Crossflow's load axis (our note says request rate via timestamp
  compression, which would mean it does not independently vary the P:D mix either -- relevant to how our 2.2x
  ratio shift is compared against its setup), plus SLOs-Serve's DP, DeFT's KV-guided grouping and POD's overlap
  structure, since the paper makes a differentiating claim against each and currently cites all four from my own
  summaries rather than from the papers.

## v3.19.0 — Motivation 2 measured: the P/D demand ratio shifts 2.2x within minutes (2026-10-28)
- `tasks/paper_figures/pd_ratio_20261028.py`, zero GPU cost, from the demand-shift figure's own data: the
  prefill-to-decode demand ratio has median 82.4 on the colocated no-surge arm and spans 4.2x over the run; on the
  surge arm its median is 38.4 inside the surge window against 85.1 outside -- a 2.2x shift -- with 37x between
  its extremes.
- So motivation 2 upgrades from "no static ratio is good enough" to the direct claim: the right ratio changes by
  more than 2x within minutes, so a partition of nodes fixed before the run cannot track it.
- Also recorded: rate scaling alone cannot show this, because compressing timestamps by alpha preserves the
  trace's P:D proportion. The variation comes from the trace's structure plus the surge arms, which move the mix in
  opposite directions. And conditional disaggregation sees a smaller swing (1.55x) because it prefills short
  appends locally -- consistent with the mechanism motivation 1 rests on.
- 8 of 11 arXiv reference PDFs downloaded to /opt/tiger/papers (outside the repo, so no binaries are committed).

## v3.18.0 — Restore the outline's one-line goal (2026-10-28)
- Reverted the goal bullet to its original one-liner (meet per-tenant SLOs while maximizing throughput for mixed
  agentic + chat workloads, without a fixed prefill:decode node ratio). The outline is a scaffold; the expanded
  argument -- same control as disaggregation, continuous knob instead of a static partition, therefore highest
  throughput at fixed SLOs -- stays in the task doc and in the body sections where it is argued and measured.

## v3.17.0 — "Superset" means the same control as disaggregation plus a continuous knob (2026-10-28)
- Per the user, the superset claim is about expressiveness against disaggregation, not about dominating every
  baseline on every metric. Rewrote the goal around it: (a) the same control -- disaggregation protects decode by
  isolating it on its own nodes, and a per-step latency bound buys the same protection in software, demonstrated by
  stepcap=600 cutting chat tail TPOT 51% and agent 43%; (b) a continuous knob instead of a static integer
  partition, so disaggregation's behaviour is the LIMIT of our knob rather than a different design; (c) therefore
  the highest throughput at fixed SLOs, since resources not needed to hold latency serve prefill instead of being
  stranded on the wrong side of a fixed split.
- Measured support exists at both extremes (600 ms: control without throughput; 2048: throughput without control)
  and under the spike (stock PD 32-46 tok/s against our 1053). What is missing is the middle of the curve, which
  the capsweep supplies: a setting that holds the tail while keeping attainment at or above the constant's.

## v3.16.0 — State the goal, and restate "superset of all baselines" as a frontier claim (2026-10-28)
- The outline's goal bullet now says it: meet per-tenant SLOs WITHOUT disaggregation, push the
  throughput/interactivity frontier (higher attainment and lower mean TPOT at fixed throughput; more load at fixed
  attainment), and stay flexible because the knob is a latency bound solved against a measured step model rather
  than a token constant.
- Story audit: the motivation chain is consistent and each link is measured. But "a superset of all baselines" is
  NOT supported -- no configuration of ours dominates every baseline on every metric (at alpha 6 we beat mixed
  chunk on all, but lose both tails to prefill-prioritized; the chunk rule at 600 ms wins tail TPOT 51% while
  losing 37 points of attainment). Restated as a claim about the policy SPACE: loosening the bound approximates
  mixed chunk, tightening it approximates prefill-prioritized, so sweeping it should ENVELOPE their operating
  points. That is what the capsweep measures and what a static policy cannot claim.
- Recorded the fallback: if the frontier does not envelope them, a single per-step bound is not expressive enough
  and the next lever is the per-tenant form, not a tighter constant.
- Recorded for motivation 2: plot the P/D demand RATIO over time from the data fig23_demand_shift.py already
  extracts. Our two spike arms move the ratio in opposite directions, which a load factor cannot do.

## v3.15.0 — Claim 3 is already measured; trim the outline's kernel design bullet (2026-10-28)
- Claim 3 ("provisioning for either phase wastes the other's throughput") is supported once stated as the
  consequence rather than the mechanism: the spike arm gives stock PD at its best swept ratio 32-46 tok/s of agent
  SLO goodput and 8-9% chat attainment against 1053 tok/s and 93.2% for colocation (Table 17). The outline now
  says that, with the figure. What remains unmeasured is only the mechanism -- idle capacity on one side -- so the
  goodput collapse is the finding and idle capacity the explanation.
- Only claim 2 still overstates; it stays written as the sweep result until per-window re-scoring shows the
  leading ratio changes over a run.
- Trimmed the outline's kernel design bullet, which had carried executor and CTA-level detail that 5.1 already
  covers. It now states the plan, the tile classes, the balancing and the single reduction, plus the one
  architectural point (both executors stay stock at their own warp layouts; the fusion is in the plan and the
  reduction, not the instruction stream), and points to 5.1 for the mechanism.

## v3.14.0 — Audit: does our evidence support the rewritten motivation? Two of three (2026-10-28)
- Claim 1, repeated KV transfer: WELL SUPPORTED. Median 875 uncached tokens over a median 119K cached prefix, and
  conditional disaggregation -- which prefills appends locally and skips the re-ship -- matches us to 64 programs
  (94.6% vs 94.2%) while stock PD falls to 31.2% on the same cluster. One mechanism differs, so the gap isolates it.
- Claim 2, rigidity of the P:D ratio: MOSTLY SUPPORTED. We show demand swinging 100K-500K uncached tokens/s within
  minutes and that no swept ratio holds any load point at >=90%. But that is "no static ratio works", not "the best
  ratio changes over time". Cheap fix with no new runs: re-score the existing stock-PD arms in 60 s windows and
  report which ratio leads per window.
- Claim 3, over-provisioning wastes throughput: WEAKEST, currently an assertion. No per-role utilisation trace
  exists showing one side idle while the other saturates. This is the claim a reviewer will want a figure for.
- Until 2 and 3 close, claims 2 and 3 should be written as the sweep result rather than as statements about
  rigidity and waste.

## v3.13.0 — Motivation is the agentic workload's shape, not the tenant mix (2026-10-28)
- The motivation had been narrowed to "a cluster serves chat and agents at once", which is a complication rather
  than the reason. Rewritten around the workload: an agent session is long decode interleaved with small prefills
  -- a median of under a thousand uncached tokens appended to a prefix of tens to hundreds of thousands the
  previous round already computed -- and three consequences make the fixed role split the wrong shape for it:
  the KV cache is transferred over and over, scaling with session history rather than with the tokens the round
  added; the prefill-to-decode ratio is a static configuration while the demand swings within minutes as tool
  calls land; and provisioning for either phase wastes the other's throughput, since idle capacity cannot be lent
  across the role boundary.
- The multi-class mix (agentic sessions beside chat turns) is now what makes all three worse, because no deployed
  policy knows which class is waiting -- a second-order point rather than the headline.

## v3.12.0 — Replace the "prefill-width tiles" shorthand with what it means (2026-10-28)
- "rows stop paying prefill-width tiles" assumed the reader already knows which work has small query tiles and
  that the chunk's tiles are the large ones. Now stated: the small-query-tile work -- decode rows, draft steps and
  speculative verify rows -- stops paying the cost of the prefill chunk's large query tiles.
- Same shorthand replaced in the artifact notes. No instance of "prefill-width" remains.

## v3.11.0 — The outline's motivation states the motivation, not just the baselines (2026-10-28)
- The motivation bullet listed four existing systems and their flaws without ever saying what problem the paper is
  solving. It now leads with it: one cluster serving an agentic coding product and a chat product at once, whose
  prompt lengths and latency targets differ by an order of magnitude, where every deployed answer meets one of
  throughput or latency and gives up the other, and none knows which tenant is waiting.
- Each of the four failure modes now names the quantity it costs (throughput; the running requests'
  time-between-tokens; the network and the agent's round latency; throughput and capture memory), so the list reads
  as evidence for the motivation rather than as a roster of systems.

## v3.10.0 — Baselines belong in the evaluation sections only (2026-10-28)
- Removed the four-baseline list from 5.1. Baselines are listed at the head of each evaluation section -- 6.2, 6.3
  and 6.4 -- and nowhere else; 5.1 is the design and should not carry a comparison list.
- What stays in 5.1 is design rationale rather than a baseline list: the role split is named in one sentence as
  what the plan must improve on, followed by the three properties of it the single plan removes (two plans, one
  query tile per FA2 launch, a separate merge).
- 6.2's list carries all four kernel baselines, with two-stream overlap as its own entry.

## v3.09.0 — 5.1's baseline block lists all four designs; two-stream gets its own bullet (2026-10-28)
- 5.1's bulleted baseline block had been scoped to naive FA3 + FA2 alone, so sglang's single varlen FA3 call and
  POD's one-FA2-kernel structure were named in 6.2 but missing where the design is introduced. It now lists all
  four: sglang's single call, POD's structure, naive serial (the fair baseline), and naive on two streams.
- Two-stream overlap is now its own baseline in both 5.1 and 6.2 rather than a clause inside the naive bullet. It
  is a distinct design -- the cheapest attempt at overlapping the roles, needing no plan -- and it bounds what
  launch-level overlap can be worth.
- The three properties of the naive split that C1 improves on (two plans, one query tile per FA2 launch, a separate
  merge) are kept as a separate list, so baselines and the gap they leave are not interleaved.

## v3.08.0 — Our own earlier build is not a baseline (2026-10-28)
- Removed "our own earlier two-plan build" from the baseline list at the head of 6.2. A system's own superseded
  version is not a comparison anyone else can check, and listing it beside external baselines inflates the
  apparent contribution the way the role-split factors would.
- Its row stays in the normalized table, marked *(ours, superseded)*, with a sentence saying why it is there --
  evidence that the one-plan structure was the right call -- and why it is not in the baseline list. Deleting the
  row outright would have discarded a measured data point for the wrong reason.

## v3.07.0 — 5.1's baseline paragraph becomes bullets (2026-10-28)
- The "Baselines for this design" blob packed three separable facts into one run of prose. Split into bullets:
  two kernels with two plans each merging its own partials; one query tile for the whole FA2 launch, chosen from
  the rows' average packed length, so it cannot give two rows in one launch different tiles; and a separate merge
  pass because neither launch can reduce the other's partials.
- Followed by the two parts of C1 that are mandatory rather than optional, also as bullets: one plan for the step,
  and one reduction of both executors' partials inside the FA2 launch.

## v3.06.0 — List each stage's baselines at the head of its section (2026-10-28)
- 6.2: naive FA3 + stock FA2 (serial and two-stream, the fair baseline our 4-9% claim is measured against),
  sglang's single varlen FA3 call, POD-Attention's one-FA2-kernel structure, and our earlier two-plan build.
- 6.3: sglang prefill-prioritized and mixed chunk on FA3 (each with and without stock EAGLE3 MTP) and
  PD-Multiplexing, all with sglang's own chunk policy -- with pdmux named as the most informative, since it attains
  our latency protection structurally.
- 6.4: stock PD disaggregation with the P:D ratio swept (headline), conditional disaggregation at Dynamo's
  defaults, sglang's own colocated schedules on the same cluster, and the fixed-2048 chunk flagged as an internal
  control rather than a published system.
- Each list says WHY the baseline is there, so a reader can see which comparison carries which claim.

## v3.05.0 — Highlight the outline scaffold so it is skimmable (2026-10-28)
- Bolded 27 anchors in "Paper Outline (10/24)": the six top-level headings (goal, motivation, design, distributed
  inference, evaluation, story), each motivation item's subject, the design and evaluation subjects, and every
  stage's `claim:` / `measured` label, with `figures:` italicised so it reads as the lower tier it is.
- Two outline lines were also corrected while highlighting, since they stated the wrong mechanism and the body text
  had already been fixed: prefill-prioritized now says an arriving prefill takes the step's budget before the
  decode rows, so requests already generating stall (not prefill queuing behind prefill), and pd multiplexing now
  notes it attains the same latency protection as ours structurally, by partitioning SMs.
- Synced to Drive.

## v3.04.0 — Queue the stepcap sweep: fill the interior of the frontier we already bracket (2026-10-28)
- `tasks/pd_colocation/mn_capsweep_16g_20261028.sh`: stepcap at 1000 / 1500 / 2000 / 3000 ms, alpha 6, chained
  last. The measured arm says the mechanism works (-51% chat tail TPOT) and the 600 ms value starves prefill
  (TTFT p90 41.65 s, attainment 23.6% against the constant's 60.5%), so this sweeps the one number rather than
  redesigning the budget again.
- Read as a Pareto curve of chat TPOT p90 against chat attainment, which doubles as C4's throughput-latency
  frontier. The script records the falsifiable reading: the rule earns its place only if some value beats the
  constant on tail TPOT without giving up attainment; if none does, the finding is that a per-step latency bound
  cannot buy tail TPOT here without starving prefill, and the next lever is capping by the tightest RESIDENT
  tenant's target rather than a global constant.

## v3.03.0 — Queue the pdmux retry at both loads with MEM_FRAC lowered (2026-10-28)
- `tasks/pd_colocation/mn_pdmux_retry_16g_20261028.sh`: PD multiplexing at alpha 3 and 6 with MEM_FRAC 0.70 -> 0.55.
  pdmux captures CUDA graphs per SM group, so graph memory multiplies and the 0.70 carried over from the 8-GPU
  driver is the suspected cause of the arm that started workers and served nothing.
- The driver carries the reason this arm matters most: pdmux has the same SLO control as ours but no kernel and no
  overlap schedule, attaining latency protection structurally by partitioning SMs, so it is the serving-level
  leave-one-out for C1+C2. Registered expectation in the script: close on tail TPOT, behind on throughput; if it
  matches on both, C1 and C2 do not survive at the serving level.
- Chained last, behind the PD re-measurement and the alpha-3 chunk retry.

## v3.02.0 — pdmux reframed as the strongest baseline, which makes its failed arm blocking (2026-10-28)
- Per the user: pdmux has "same SLO control, but no kernel/overlap schedule". It gives decode and layer-split
  prefill their own SM groups, so an arriving prefill cannot block a running decode row -- it attains the TPOT
  protection our chunk rule pursues by sizing, but structurally. So it is the one baseline that already solves the
  problem C3 solves, and the comparison answers the reviewer's question: if partitioning SMs meets the SLO goal,
  what do the kernel and four-mode step add?
- It is the clean serving-level leave-one-out for C1+C2, the counterpart of the fixed-2048 arm (which isolates
  chunk sizing while holding the kernel constant). The two bracket the contributions from both sides, which is why
  pdmux is definite and 2048 potential.
- Expected shape registered in advance: close to us on tail TPOT, behind on throughput (no overlap scheduling, and
  partitioning costs occupancy on both sides; prior ~10%). If it matches us on BOTH, the kernel and four-mode
  contributions do not survive at the serving level and the story must change -- so this is the highest-value arm
  to get right, and it is currently unmeasured at both loads.

## v3.01.0 — Baseline status: sglang's three policies definite, fixed-2048 potential (2026-10-28)
- 6.1 now states it plainly: prefill-prioritized, mixed chunk and PD multiplexing stay in every table. The fixed
  2048-token chunk is a potential baseline, first to be cut if space is tight -- it is a control rather than a
  system anyone publishes, though internally it is the best one we have (one variable differs, acceptance matched
  to 0.5%).
- Consequence flagged as blocking: pdmux is a DEFINITE baseline and is currently unmeasured at both loads -- its
  alpha-6 arm failed with no score and no acceptance lines, and alpha 3 was never run. Fix before re-running:
  check for capture-time OOM and lower MEM_FRAC from the 0.70 carried over from the 8-GPU driver, since PD
  multiplexing captures CUDA graphs per SM group.

## v3.00.0 — The 2048 arm is the baseline for the TPOT-control study (2026-10-28)
- Recorded its three roles: the leave-one-out control for chunk sizing (identical tree, kernel, four-mode step,
  router and EDF -- only the chunk source differs), the throughput-side endpoint of the stepcap frontier, and the
  ad-hoc policy the rule must beat, which is not a straw man since at alpha 3 it matches rung L0 inside the noise.
- The decisive advantage over the sglang comparisons: acceptance matches to within 0.5% (2.023 vs 2.013), where
  every sglang arm sits at 1.765-1.795 against our 1.982-2.001. So the -51% chat / -43% agent tail-TPOT result
  against c2048 carries no acceptance caveat, and the TPOT-control study is reported against it with the sglang
  arms as the external comparison.

## v2.99.0 — The SLO chunk rule is measured: -51% tail TPOT, but the bound is far too tight (2026-10-28)
- First clean measurement of the policy (alpha 6, accept 2.023), against the fixed-2048 arm from the same tree:
  chat TPOT p90 55.8 vs 114.6 ms (**-51%**), agent 59.2 vs 103.9 (**-43%**). The mechanism works and the user's
  expectation of a large tail-TPOT win is confirmed.
- But a 600 ms ceiling starves prefill: chat TTFT p90 41.65 s against 5.91, chat attainment 60.5 -> 23.6%, agent
  SLO goodput 1976 -> 900 tok/s, 8161 agent rounds never issued.
- Against the prediction registered before the run (TPOT p90 <= 95 ms AND attainment above 61.8%): first met with
  room, second not. So the registered reading applies -- the knob works, the value is wrong -- and the next step is
  a sweep of `stepcap` (1000/1500/2000/3000 ms), not another budget redesign. Two points on one curve now exist,
  the bound is a monotonic latency, and the sweep doubles as C4's throughput-latency frontier.
- Second lever if needed: cap by the tightest RESIDENT tenant's tail target instead of a global constant, so
  agent-only steps may run long while steps carrying chat rows stay short. No new model.

## v2.98.1 — Refit the step MLP: a nonlinear model on the same features makes the decode bias worse
- `fit_step_mlp_20261028.py` (new, saved artifact): held-out decode bias **-19.9%** against the linear model's
  -13.0%, with train bias +6.8% -- a 27-point swing across the chronological split, i.e. distribution shift rather
  than model capacity.
- So the router must not be given an MLP on these five features. The missing information is likely the decode
  batch's KV-length distribution (only its mean is logged) and the resident tenant mix. TPOT prediction carries a
  load-dependent 13-20% bias and no accuracy claim can be made until such a feature is logged.

## v2.98.0 — The MLP the paper credits has no saved artifact (2026-10-28)
- Checked all three checkpoints under tasks/perf_model/ (model.pt, lpt_model.pt, dmc_model.pt): identical five
  features ['n_p','r_p','r_d','decode_bs','overlap'] from train_perf_model_mlp.py, with `overlap` at ~1e-8 sd in
  every one, i.e. constant in training and a dead input. None is the "MLP with acceptance" that paper 5.2.2 credits
  at 3.4-4.0% on merged steps, and no script there saves such a model -- fit_steptrace.py fits one in memory and
  discards it.
- Two consequences. The plan to give the router an MLP for decode-step time needs that model refit and SAVED first
  (zero GPU cost, but work rather than a lookup). And 5.2.2 quotes a figure from a fit with no artifact, so it is
  not reproducible: refit and save, or drop the column.
- Recorded a do-not: none of the three existing checkpoints may be wired into the chunk rule or the router, since
  their feature semantics are unconfirmed and one input is dead.

## v2.97.0 — Queue the alpha-3 chunk-rule retry; the driver fix is confirmed working (2026-10-28)
- The health-poll fix takes effect per arm, since each arm is a fresh `python3 -u mn_run.py`. Confirmed on the
  next arm: `a6_tail_r1_ours` reported "16 workers healthy after 156 s" and launched its router -- the exact step
  that crashed with IndexError on the alpha-3 arm.
- `tasks/pd_colocation/mn_tailchunk_a3retry_20261028.sh`: re-runs ONLY our alpha-3 arm (`_tail_r2`), since c2048 at
  alpha 3 already scored (90.6% / 93.8% / 2473 tok/s). Chained behind the PD re-measurement so it does not contend
  for the cluster.
- After this the chunk rule will have been attempted five times: three invalidated by budget bugs now fixed and
  covered by an offline test, one by this driver crash, and two arms (alpha 3 retry and alpha 6) pending.

## v2.96.0 — Root cause of the failed chunk arm: a driver crash, not the policy (2026-10-28)
- `mn_run.py:153` indexed `[l for l in out.splitlines() if l.startswith("UP=")][-1]`. The worker health poll
  returned output with no UP= line, so the list was empty, IndexError aborted the arm before the router launched,
  and the arm directory ended up with worker logs but no router.log, no mn_bench.log and no bench_*.log. The
  scheduler never crashed -- the server logged "fired up and ready to roll" -- and nothing was ever served.
- Fixed: a missing UP= line now logs, sleeps and retries, letting the existing 40-minute timeout decide, which is
  how every other transient in that loop already behaves. One bad poll cost a 30-minute cluster slot.
- So the SLO-budgeted chunk is STILL untested: four attempts, three invalidated by budget bugs since fixed and
  this one by a driver crash unrelated to the rule.

## v2.95.0 — The 2048 constant matches rung L0; our chunk arm failed to serve (2026-10-28)
- `c2048` (our tree, A6, four-mode, router, EDF, fixed 2048 chunk), accept 1.994, alpha 3: chat 90.6% / 1451 tok/s,
  TPOT p50/p90 11.2/31.6 ms; agent 93.8% / 2473 tok/s, TPOT p50/p90 18.0/41.1 ms. That lands on top of rung L0
  (91.2% / 93.8% / 2493, 10.8/29.8) with every difference inside the noise floor -- so the constant is a strong
  baseline and the rule must beat a configuration that already matches sglang's own chunk policy.
- The `:life` + `stepcap=600` arm produced accept=NONE and no score. The patched scheduler did NOT crash: the
  worker log ends "fired up and ready to roll" with no traceback, and no acceptance line means nothing was served
  -- a client-side or bench-start failure for that arm alone, since c2048 ran right after on the same cluster.
- Diagnostics recorded in order (bench logs and the unconditional MNBENCHDONE marker; whether all 16 workers
  reached healthy; whether the new SLO_CHUNK spec parses in the DEPLOYED tree, since the config is parsed lazily
  on first call and a bad field would raise inside the scheduler loop after the server reports ready).
- The SLO-budgeted chunk still has no measurement: four attempts, three invalidated by bugs since fixed and this
  one by an arm that never served. Untested, not disproven.

## v2.94.0 — A single calibration constant does not fix the decode bias: it is load-dependent (2026-10-28)
- Fitted one multiplicative factor on the chronologically first half of 73 774 decode steps (x1.1602) and validated
  on the second: held-out bias -13.3%, unchanged from the uncalibrated -13.0%. Train bias +15.0% against a
  median-fitted 0.0%. So the bias is NOT stationary -- the superposed windows shift the load mix across the run and
  the error is load-dependent, which no scalar correction can remove.
- This answers the MLP question from the other direction. The linear model is missing a load-varying feature, which
  is the regime where the MLP already measures better on decode steps (18-23% against 34-38%). So an MLP earns its
  keep for DECODE-step time in the router's TPOT prediction, not for chunk sizing, where a threshold tolerates
  error ~1:1 and the linear model suffices. The "simple methods suffice" conclusion stands only for the chunk.
- Next, no GPU time: evaluate the step MLP's decode-step SIGNED bias on the same chronological split. Small bias
  means the router should use it; a similar -13% means the problem is a missing feature (decode batch composition,
  KV-length distribution) rather than model capacity.

## v2.93.0 — The decode-step error is biased -13%, so predicted TPOT is systematically optimistic (2026-10-28)
- `tasks/perf_model/tpot_bias.py` on 73 774 decode-only steps: per-step MAPE 29.0% (p90 49.6%), per-step SIGNED
  bias **-13.0%** (median -23.1%), sd 33.7%.
- The signed error is the statistic that matters, because a request's TPOT is a mean over its decode steps:
  independent error shrinks as 1/sqrt(n) (33.7% -> 1.5% at 500 tokens) but bias does not. So predicted TPOT is
  about 13% optimistic, which is the worst direction -- the router and SLO check admit requests expecting better
  TPOT than they will get, and the chunk rule's budget is too permissive because the decode base it subtracts is
  under-estimated.
- Fix is a per-step-type multiplicative calibration, fitted on a training split and validated on held-out data;
  fitting on all the data and reporting zero bias would be circular. Until then no TPOT-accuracy claim goes in the
  paper and 5.2.2's 2.0% aggregate must not be read as a TPOT number.

## v2.92.0 — TPOT prediction accuracy is not established; say so and separate the two predictions (2026-10-28)
- The 2.0% held-out step-time MAPE is an aggregate. The per-type split already in paper 5.2.2 shows decode steps at
  34-38% (linear), 18-23% (MLP), 21-25% (GBT). TPOT is almost entirely decode steps, so 2.0% must not be quoted
  for TPOT: the aggregate looks good because long prefill steps dominate an error-weighted mean.
- Separated the two predictions. The chunk rule needs the time of one MIXED step (4.7% on 10,185 merged steps) and
  its output is a threshold, tolerating error ~1:1. The router's TPOT prediction sums many future DECODE steps,
  where the 18-38% error lives and compounds. Validating one does not validate the other.
- Recorded the measurement that closes it, with no new GPU time: predicted versus measured PER-REQUEST TPOT from
  the traces on disk, MAPE and p90 signed error, split by tenant. Never measured end to end.
- Recorded why not the alternatives: a DP over future steps is unnecessary (the budget is exact arithmetic) as well
  as the method we are avoiding; a direct chunk regressor trains on step-model labels, so it inherits that error
  plus its own, must learn arithmetic we compute exactly, and is single-purpose where the step model is reused.

## v2.91.0 — Correct the head-of-line-blocking mechanism, and flag survivor bias in the TPOT tails (2026-10-28)
- 3.3 described the wrong mechanism: prefill queuing behind prefill (a TTFT effect). The real one, per the user, is
  incoming prefill blocking RUNNING decode -- sglang serves the waiting queue before the decode rows, so an
  arriving prefill stretches the inter-token gaps of requests already generating, and the damage lands on their
  TPOT. Rewritten to say that, and to name it as damage inflicted by a request that has produced no token yet.
- That makes pp's better measured TPOT tail (94.9 vs our 105.2 ms p90) suspect: pp left 7088 agent rounds unissued
  against our 6699 and served 5984 against 6373, at 57.8% chat attainment against 61.8%. A request never admitted
  contributes no TPOT sample, so pp's percentiles are over a smaller, more favourable population with fewer rows
  per step.
- Action recorded, no re-run needed: re-score both arms over the intersection of sessions both issued and report
  TPOT conditioned on admission. If pp's advantage vanishes on a common population it is an artefact of serving
  less; if it survives, our longer steps really do cost the tail.

## v2.90.0 — Register the tail-TPOT prediction before the arm lands (2026-10-28)
- Our own tail should improve a lot: L1's chat TPOT p90 of 105.2 ms was measured with no bound on step length at
  all, and stepcap=600 is that bound.
- But pp's head-of-line blocking lands on TTFT, not TPOT. pp serializes prefill into its own steps and runs stock
  MTP, so its steps stay short and its TPOT p90 of 94.9 ms is the strength its policy buys -- paid for with chat
  TTFT p90 of 9.02 s against our 3.42 s. The structural win against pp is TTFT and attainment; on tail TPOT the
  result to expect is parity-or-better, not a rout. Against mixed chunk, which shares our mechanism and also has no
  bound (102.0 ms), a large tail win IS expected.
- Falsifiable form recorded: chat TPOT p90 <= 95 ms with attainment above 61.8% means the rule works; above 100 ms
  means the 600 ms cap is too loose and the next step is a sweep on that number, not another budget redesign.

## v2.89.0 — Report TPOT only, not inter-token percentiles (2026-10-28)
- `tracelab_windows.py` no longer prints the ITL cell. TPOT p50/p90/p99 is the latency metric: a per-request mean
  is what the SLO is written against and what a user feels, while an inter-token percentile is taken over every gap
  in the run, is dominated by single long steps, and is not comparable to a TPOT percentile. Reporting both side by
  side invited reading one as the other, which is exactly the confusion that arose over the alpha-6 arms.
- Table 14's inter-token row becomes the chat mean-TPOT p99 row, marked [pending] until a run reports it -- the
  honest state, since the old number was an ITL percentile and cannot be relabelled as a TPOT one.

## v2.88.0 — The alpha-6 tail loss is an arm without the chunk layer, and tail TPOT is the metric (2026-10-28)
- Stated plainly what the earlier entries implied but did not say: L1 is a decomposition rung carrying sglang's
  default chunk policy, so the mechanism that protects TPOT -- the chunk sizing -- is NOT in the arm that loses the
  tail to prefill-prioritized. Those numbers measure our longer fused steps with no latency bound at all, which is
  what stepcap=600 supplies in the arm now running. The comparison must not be reported as "our system loses the
  tail".
- Separated the two tail metrics: TPOT is a mean over a request's tokens so a long step is amortized (chat p50
  35.2 vs pp's 35.6 ms -- we are ahead; p90 105.2 vs 94.9, +10.9%), while ITL p99 is a percentile over individual
  gaps so one folded step lands in it (999 vs 503 ms, +99%). Tail TPOT is the user-facing quantity and we lose it
  by 11%, not 2x.
- `tracelab_windows.py` now reports TPOT p50/p90/p99, so the tail of the mean is visible without reading ITL, with
  a comment on why tail TPOT is read first.

## v2.87.0 — Remove the internal publishability target from the paper (2026-10-28)
- Dropped all seven "**Bar.**" / "**Bar status.**" paragraphs and every remaining reference to the 30% latency /
  10% throughput target. That target is our own decision rule for whether to keep pursuing a direction; it is not
  a claim about the system, and a reader has no use for it.
- Results that were framed against it are restated as what they are: e.g. "chat mean TPOT falls 29-32% and the
  chat TTFT tail 79%, agent mean TPOT 19.5%, combined SLO goodput +5.2%" instead of "the bar is partially met".
- The one surviving use of "bar" is Figure 20's error bars, which is a different thing and was left alone.

## v2.86.0 — alpha-6 ladder complete; pdmux failed and is recorded as failed (2026-10-28)
- `MN_OVERLOAD_DONE` with 4 of 5 arms scored. pdmux launched but produced `SCORE (none)` and `accept=NONE`:
  no records, no acceptance lines, nothing served. Recorded as a failed arm rather than dropped, since it is one of
  the three single-node baselines the user named.
- First cause to check: PD multiplexing captures CUDA graphs per SM group, multiplying graph memory, and
  MEM_FRAC=0.70 was carried over from the 8-GPU driver -- a capture-time OOM gives exactly this signature. Second:
  the pdmux path disables overlap scheduling and may reject this driver's EDF_UNCACHED / HC_PREFETCH env.
- pdmux is therefore unmeasured at both load factors on this cluster.

## v2.85.0 — No bold in the abstract; measured the abstract/intro redundancy (2026-10-28)
- Removed the bold spans from the abstract: abstracts are not typeset with emphasis. Bolding stays in the
  introduction, section 6 and the ablations, where a reader skims for numbers.
- Measured the redundancy the draft was suspected of: all 9 numeric results in the abstract also appear in the
  introduction, i.e. the abstract is a strict subset (abstract-only numbers: 0) and the introduction restates it
  before adding 17 more. Repeating one or two headline figures is conventional; restating the whole abstract is
  not, and the introduction's results paragraph should instead organise by baseline and point forward to section 6.
  Flagged, not yet rewritten.

## v2.84.0 — Abstract rewritten: 606 -> ~290 words, key results in it (2026-10-28)
- The abstract was 606 words and carried no headline number in its first half. Rewritten to four paragraphs:
  the scenario (agentic coding and chat on one cluster, lengths and targets an order of magnitude apart), why both
  deployed answers fail (disaggregation re-ships the KV every round; colocated schedules compute a one-token row at
  the chunk's 128-row tile and block a chat turn behind a 64K prompt), what we do, and the results.
- Results now in the abstract, bolded: 4.0-8.7% over the fair kernel baseline; on four nodes of four GPUs,
  3.7-14x the agent SLO goodput of disaggregation at its best ratio (93.6% against 13.1% chat attainment), +12-37%
  over conditional disaggregation and +57% under shifting demand, +14.2% and +8.0 points over sglang's mixed chunk,
  -32% chat mean TPOT and -79% chat TTFT tail against prefill-prioritized, and no requests lost when a quarter of
  the cluster fails.
- The prefill-prioritized goodput margin is still excluded, pending the acceptance question.

## v2.83.0 — Highlight the headline results so the paper is skimmable (2026-10-28)
- Bolded the numbers a reader should find without reading prose, in the abstract and the introduction's results
  paragraph: the kernel's 4.0-8.7% over the fair baseline, single-node chat attainment +38-74 points and agent
  attainment 13-19% -> 44-50%, GLM throughput +27-28%, the cluster headline (93.6% chat attainment at 1100 tok/s
  agent goodput), +12-37% over conditional disaggregation and +21-24% under a spike, the shifting-demand result
  (+57%, +90% under a chat surge, +14-21 points chat attainment), the fault-tolerance result, and the Llama margins.
- Applied idempotently (each string is bolded once and skipped if already bold), so the pass can be re-run as more
  results land. Synced to Drive.

## v2.82.0 — C4: the mixed-traffic evaluation becomes its own contribution (2026-10-28)
- New contribution item for the scenario, which is compelling on its own: an agentic coding product and a chat
  product sharing GPUs, as Codex and ChatGPT traffic do -- TraceLab coding sessions on real timestamps (10 s TTFT /
  80 ms TPOT) concurrent with ShareGPT chat (2 s / 50 ms) on four nodes of four GPUs.
- States the measured margins against both baseline families: 3.7-14x agent SLO goodput and up to 80 points of
  chat attainment over stock PD at its best ratio; +12-37% goodput over conditional disaggregation (+57% on the
  shifting-demand replay, +90% under a chat surge); +14.2% goodput and +8.0 points chat attainment over sglang's
  mixed chunk at overload, and -32% chat mean TPOT with a -79% TTFT tail against prefill-prioritized at capacity.
- Framed as a throughput-latency frontier (higher attainment and lower mean TPOT at fixed throughput, more load at
  fixed attainment), with the Pareto curve marked pending, and the prefill-prioritized goodput margin explicitly
  withheld pending the acceptance question.

## v2.81.0 — C3 stated concisely, with the workload it is evaluated on (2026-10-28)
- C3's worker paragraph tightened (it also had a dangling "A" and a number-agreement error), and the dropped
  mechanisms compressed to one clause.
- C3 now names the setting that motivates it: a cluster serving several kinds of request at once with different
  lengths and different SLOs -- TraceLab's agentic CODING sessions replayed on their real arrival timestamps
  (rounds carrying tens to hundreds of thousands of cached tokens) mixed with ShareGPT chat turns of a few hundred
  tokens under a much tighter target. Recorded for accuracy: the cluster mix is TraceLab coding + ShareGPT chat;
  there is no separate code stream, and math/coding standalone sets are separate single-workload runs.

## v2.80.0 — Add the fixed-2048 baseline to the queued chunk arm (2026-10-28)
- `mn_tailchunk_16g_20261028.sh` gains a `c2048` arm at both loads: identical tree, A6 kernel, four-mode step,
  router and EDF ordering, differing only in that the chunk is a fixed 2048 tokens instead of solved from the step
  model. That is a clean leave-one-out on the chunk-sizing decision, and it is the bar the rule must clear -- a
  model-driven chunk has to beat a constant to justify its complexity, and beating sglang while losing to the
  constant would kill the contribution.
- Added before the driver started, replaced atomically with mv, syntax-checked.

## v2.79.0 — Fixed chunk 2048 stays as a baseline, not as our policy (2026-10-28)
- The 6.1 marker conflated two things. Corrected: for OUR arms the constant is being replaced, because a constant
  cannot be justified across loads; as a BASELINE the fixed 2048-token chunk stays, since it is the simple ad-hoc
  policy a deployed system actually uses and a model-driven scheduler has to beat it to justify its complexity.
  Marked as removable only if the final arm set makes it redundant beside sglang's own chunk policy.

## v2.78.0 — Motivation sections now carry data from the settings we evaluate (2026-10-28)
- 3.3 gains the cluster-scale measurement of head-of-line blocking, in section 6.4's own setting: on 16 H20s under
  the superposed TraceLab replay plus chat at alpha 6, sglang's prefill-prioritized schedule holds chat TTFT p90 at
  9.02 s against a 2 s target with 7088 agent rounds never issued, while its agent SLO goodput is within 9% of the
  best arm. The policy fails only at latency, and at both scales -- the single-node burst (4.5-13.9 s on one H200,
  section 6.3's setting) and the cluster.
- 3.4 gains the end-to-end cost of mixed chunk in the same setting: lowest agent SLO goodput of every arm measured
  (1926 tok/s against 2012 for prefill-prioritized and 2199 for ours), worst inter-token tails (chat p99 1246 ms,
  agent 1301 ms), chat attainment 53.8%. The tile-level waste derived in that section is therefore not a
  microbenchmark artefact: the policy that pays it is the slowest of the three schedules on the paper's workload.

## v2.77.0 — Mixed chunk at alpha 6 reframes the ITL-tail finding (2026-10-28)
- mc (accept 1.765): chat 53.8% / 684 tok/s, TTFT p90 8.19 s, TPOT p90 102.0 ms, ITL p99 1245.9 ms; agent 77.4% /
  1926 tok/s, TPOT p90 108.4 ms, ITL p99 1301.3 ms.
- The chat ITL p99 ordering is pp 503 < ours 999 < mc 1246 ms (agent 514 < 920 < 1301). Mixed chunk folds prefill
  into decode steps, as our four-mode step does, and has the worst tail of the three; prefill-prioritized
  serializes prefill and has the best. So the tail cost is the price of mixing, we pay less of it than sglang's own
  mixed chunk, and pp buys its tail by giving up throughput and TTFT.
- L1 vs mc wins every metric: chat attainment +8.0 pts, agent +2.9, agent goodput +14.2% (clears the 10% bar),
  chat TTFT p90 -58%, agent TPOT p50/p90 -13%/-9.8%, chat ITL p99 -20%, agent ITL p99 -29%.
- Acceptance caveat hardens: both sglang arms land at 1.765-1.795, both ours at 1.982-2.001. A per-arm fluke is now
  unlikely and a systematic tree or step-structure difference likely -- which is what the per-tenant logging will
  separate, and until then the goodput margins stay unclaimed.

## v2.76.0 — 3.3: sglang's OOM-driven chunk back-off protects liveness, not SLOs (2026-10-28)
- Added the caveat a reviewer would otherwise raise: sglang DOES shrink the prefill it admits, lowering its
  new-token ratio and ultimately retracting running requests when admitting more would exhaust the KV pool. That
  mechanism is driven by memory, not time -- it guarantees a running request can produce its next token at all
  (liveness), not when the token arrives (the SLO).
- Why the distinction matters for the claim: a server with ample KV headroom never triggers it and still misses
  every target under the measured burst, and one that does trigger it has already abandoned the retracted
  requests' deadlines. So it is not a weak form of SLO control we are strengthening; it is a different guarantee,
  and the latency axis is unmanaged.

## v2.75.0 — 3.3 explains sglang's prefill-prioritized policy instead of pre-empting our solution (2026-10-28)
- The section argued about EDF and demotion, which are C3's mechanisms and belong in section 5; a motivation
  section should only establish that the deployed policy fails. Rewritten to state what the policy IS (fill each
  step's token budget from the waiting queue in arrival order, chunks capped at the configured size, decode gets
  the remainder), why it is chosen (eager prefill keeps the decode batch large, which amortizes the weight read, so
  throughput is high), and why it cannot bound latency (arrival order plus a fixed chunk ignore both how long a
  request occupies the prefill side and how urgent it is).
- The failure is named as head-of-line blocking and quantified: a 64K prompt holds the prefill side for tens of
  consecutive steps, and the measured agent burst holds chat TTFT at 4.5-13.9 s against a 2 s target while
  throughput is unharmed. Also notes why a larger chunk cannot fix it -- it shortens the chunk queue but lengthens
  each step, moving the cost onto the resident rows' inter-token latency.

## v2.74.0 — 3.4 now explains WHY mixed chunk falls behind, from the tile composition (2026-10-28)
- Table 3's reading explained why OUR gain is small but never why the baseline loses. Added the arithmetic, which
  closes the gap quantitatively: on the replay's median merged step (2048-token chunk, 119K cached prefix, 14
  one-token decode rows) the rows need 14 x 7 = 98 query rows of work, 4.6% of the chunk -- but padded to a 128-row
  tile each they become 1792, 87% of the chunk. The step goes from 2146 row-equivalents to 3840, a factor of 1.79x,
  against the 1.86x measured for sglang's shipped single call on the 128K four-mode captures. Mixed chunk loses
  because it computes 98 rows of work as 1792, not because the shape is hard.
- The opening paragraph is now labelled "What sglang's mixed chunk actually runs", so the baseline is named as the
  starting point rather than reading as preamble.

## v2.73.0 — Section 7 leads with per-contribution leave-one-out; negatives to one line each (2026-10-28)
- The ablation now opens with the figure SLOs-Serve's Figure 14 does: remove one contribution at a time from the
  full system. New table collects C1, C2, C3-worker (ordering and chunk separately), C3-router and the pool, each
  with the cost of removing it, and a figure spec for the bars at load factors 3 and 6. The kernel-internal
  ablations are demoted to supporting detail.
- Every kernel negative that does not change a claim is now one line with its number: persistent executor
  (0.987-1.006x), ready-queue merge (0.982-1.005x), DSMEM cluster reduction (0.597-0.993x, gang scheduling costs
  1-11% even with nothing to reduce), KV multicast (1.3-1.5x in isolation, 0.76-0.77x in the step, lockstep pairing
  alone costing 6%), L2 ordering (1.08x on a 48K shared prefix, -29% elsewhere), intra-CTA hosting (implementable
  at 148 B of spill, bounded at 1.1-2.3%).
- Tables renumbered for the insertion.

## v2.72.0 — Third chunk-rule bug found offline: :qcap starves prefill under load (2026-10-28)
- The user's position that a chunk-sizing regression "can't go wrong -- there must be a bug" is right, and this is
  the third of the same family. `:qcap` caps a step by the tightest WAITING request's remaining TTFT slack, but a
  queued request's deadline says when it must be ADMITTED, not what one step may cost. With a deep queue there is
  always a request due in tens of milliseconds, so the cap collapses the chunk exactly under load.
- Measured offline in `test_slo_chunk_rule.py` (new queue cases): chunk 4753 tokens at a 5 s or 1 s nearest
  deadline, 2264 at 0.3 s, **518 at 0.1 s**. It had already bound 13% of steps (qcapped=927 of 7001) while the far
  tighter per-step budget was also active, so with the lifetime budget fixed it becomes the binding constraint.
- Removed from the queued tail-bounded arm, which had not yet started; the tail target (`stepcap=600`, the chat p99
  inter-token SLO) is now the only per-step bound. Replaced atomically with mv rather than edited in place.
- Restating the record: every measured SLO-chunk regression predates these fixes. The `:life` arm's own log proves
  its lifetime branch never executed (budget_ms ~90, capped 7001/7001), so no arm has yet tested the policy.

## v2.71.0 — POD-first kernel argument, references restored, chunk-2048 results marked (2026-10-28)
- 3.4 restructured into the order the story needs: POD first (its design is FA2-only, and FA2 is the problem on
  Hopper -- measured at 3.45-3.69x our attention time, while moving the chunk to FA3 is worth 1.76-7.7x), then why
  FA3 and FA2 cannot be fused (incompatible warp layouts, one block size, one register budget; co-compiled 2.16x
  slower against two launches 1.21x faster), then the one overlap the layouts do permit -- a 16-row mma.sync tile
  beside a 64-row wgmma tile in one CTA -- with its 1.1-2.3% ceiling and a forward reference to section 7.
- The full annotated reference list is back in the paper (12 entries, four bullets each); the split-off task-doc
  copy is removed so there is one source of truth.
- Section 6.1 carries a standing TO BE REPLACED marker: every serving result in section 6 used a fixed 2048-token
  chunk because the predicted-chunk rule was not working when they ran. No submitted number will rest on a chunk
  constant.
- Status of the SLO-budgeted chunk, stated plainly: no positive result yet. All three arms so far are below rung
  L0 (2259, similar, 2323 against 2493 tok/s), but none tested the policy -- two used the wrong budget form and
  :life never executed its lifetime branch. The queued tail-bounded arm is the first on the fixed tree.

## v2.70.0 — Per-tenant acceptance logging, written but deliberately not deployed yet (2026-10-28)
- `unified/apply_accept_per_tenant_20261028.py`: splits the acceptance mean by tenant class on the same prompt
  threshold the SLO machinery uses, emitting `ACCEPT_TENANT chat=<mean>/<n> agent=<mean>/<n>` beside the existing
  line. Off unless `SGLANG_ACCEPT_PER_TENANT` is set, so no other arm's output changes. Dry-run verified on a copy:
  parses, inserts once, idempotent on a second application.
- This is what separates the two surviving explanations for pp's 1.795 against our 1.982: matching per-tenant
  acceptance means the gap is request mix and the +9.3% goodput margin is real; differing means the merged step
  changes what the draft sees, which is a finding about C2 rather than about the schedule.
- NOT deployed: `mc` and `pdmux` are running against those trees now, and patching a tree mid-chain invalidates
  whatever is in flight with no visible error. It goes out after the ladder finishes.

## v2.69.0 — The acceptance gap is not a draft-config difference (2026-10-28)
- Both alpha-6 arms resolve identical speculative settings: EAGLE3, eagle_topk 1, num_draft_tokens 4, num_steps 3,
  accept thresholds 1.0, attention_mode prefill. So pp's 1.795 against our 1.982 is not a configuration artifact.
- Remaining causes are the request mix (pp issued 5984 agent rounds against 6373, chat 7200 in both; acceptance
  rises with cached context) or a genuine effect of the merged step. This run's logs cannot separate them, because
  acceptance is a single mean over all requests.
- The +9.3% goodput margin stays held back. The fix is instrumentation, not a re-run: bucket `accept len` by tenant
  class so per-tenant acceptance can be compared. Matching per-tenant acceptance would make the margin real;
  differing would be a finding about C2 itself.

## v2.68.0 — pp's low acceptance is NOT the rope defect, and is now unexplained (2026-10-28)
- Checked the pp arm's worker log: the YaRN override carries `rope_theta 1000000`, so the arm runs the correct
  Qwen2.5 and the 1.795 acceptance is not the 2026-10-28 defect.
- That makes the 9.4% acceptance gap against our 1.982 unexplained, which is worse for the claim than a known
  cause: a schedule should not move acceptance, so the candidates are a configuration difference between stock MTP
  and our four-mode step, a different request mix (pp issued 5984 agent rounds against our 6373), or a genuine
  effect of the merged step -- only the last would be ours.
- The +9.3% agent goodput margin stays unusable until one is established, since the acceptance deficit alone costs
  pp ~9% more forwards per output token. The acceptance-independent results stand: chat TTFT p90 -62%, and the
  inter-token tails we lose (chat p99 +99%, agent +79%).
- Next: diff the two arms' resolved speculative configuration from their worker logs before re-running anything.

## v2.67.0 — alpha 6 vs sglang prefill-prioritized: we win the means, lose the tails, bar not met (2026-10-28)
- pp at alpha 6 (accept 1.795): chat 57.8% / 685 tok/s, TTFT p90 9.02 s, TPOT p90 94.9 ms, ITL p99 502.5 ms; agent
  79.9% / 2012 tok/s, TPOT p50/p90 45.6/104.2, ITL p99 514.2 ms.
- L1 vs pp: agent goodput +9.3%, chat TTFT p90 -62%, agent TPOT p50/p90 -11%/-6.1%, chat attainment +4.0 pts --
  but chat ITL p99 +99% and agent ITL p99 +79%, chat TPOT p90 +10.9%. The inter-token tail is a genuine
  regression and ours by construction (a prefill chunk folded into the verify forward lengthens the step), the
  alpha-3 signature amplified ~4x. The tail-bounded chunk arm (stepcap=600) is queued precisely for this.
- Bar NOT met at alpha 6: TPOT attainment +0.4..+4.0 pts against 30%, goodput +9.3% against 10%.
- **The +9.3% is not usable yet:** pp's acceptance 1.795 is below our own 1.8 validity gate and ~9.4% under ours,
  which costs it about 9% more forwards per output token -- the same order as the gap claimed. Must check the pp
  workers' effective rope base before quoting it.

## v2.66.0 — EDF ordering flips sign at overload: +5.7% agent goodput, -24% chat TTFT tail (2026-10-28)
- alpha 6 L1 vs L0 on 16 H20s (`mn_overload_decomp_16g_20261028.sh`, log /tmp/h20/mn_overload_r1.log, accept
  1.982 / 2.001): agent SLO goodput 2199 vs 2080 (+5.7%), chat TTFT p90 3.42 vs 4.50 s (-24%), agent TPOT p90 97.8
  vs 103.5 ms, agent ITL p99 919.8 vs 958.7, agent attainment +1.2 pts, 89 fewer rounds unissued; chat TPOT p90
  1.5% worse, chat attainment level.
- Outside the noise floor (~5 pts, ~4%, ~3 ms): goodput, chat TTFT p90, agent TPOT p90. The rest is inside it.
- At alpha 3 the same comparison was uniformly, marginally negative. So deadline ordering costs nothing at capacity
  and pays a little under overload -- the shape the goal asks for, but far from the 30%/10% bar alone. C3's case
  rests on the chunk layer above it, which the tail-bounded arm tests next.
- alpha 6 is confirmed genuine overload: L0 chat attainment 91.2% -> 61.5%, TPOT p90 29.8 -> 103.6 ms, ITL p99 ~1 s,
  6788 agent rounds never issued.

## v2.65.0 — Draft-share instrumentation root-caused: it wrapped a class the arm never built (2026-10-28)
- Three attempts to measure the draft share of decode wall time produced zero reports while logging
  "instrumentation active". Cause, found offline: `_create_eagle_worker` (speculative/spec_info.py:264) returns
  `EAGLEWorkerV2` only when `enable_overlap=True` and the v1 `EAGLEWorker` otherwise, and both define
  `draft_forward` and `verify`. The patch wrapped v2 alone, so an arm without the overlap scheduler instrumented a
  class that was never constructed -- a silent failure, since the module-scope patch still installs.
- `apply_draft_share_events_20261028.py` now wraps `verify` on every speculative worker class the tree defines,
  asserts at least one was wrapped, logs which, and probes `TpModelWorker.__init__` to log the live worker's class.
  A zero-report run is diagnosable from one log line.

## v2.64.0 — Item 18 built by replacing a consumer warpgroup (2026-10-28)
- Item 18's register wall is not a wall, and 1c's "vise" is released. Replacing one of FA3's consumer warpgroups
  with the 16-row row mainloop (flashinfer `FUSED_PIGGYBACK_REPLACE=1`, commit `0b4b342`) holds
  `FusedFA3PersistentKernel` at 168 registers with 164 B of spill stores and no C7512 `wgmma`-serialized warning,
  against 16 B for the same kernel with the guest compiled out and 33 424 B for the default q128 build.
- 240 registers was never reachable: ptxas names `min(launch-bound cap, setmaxnreg value)`, and the cap at 384
  threads with one CTA per SM is 65 536/384 -> 168. Measured on a standalone kernel, and `-maxrregcount` cannot
  lift it. The 64 512-register figure is the physical distribution, which the replace build restores exactly.
- The spill was a second inlined copy of the row mainloop on the FA3 role's backfill path, not the guest body;
  that backfill is also unsound in this mode, since the guest holds the CTA's only row-mainloop shared-memory slot.
- Shared memory, by compile-time `sizeof`: the launch asks 218 160 B of the 232 448 B opt-in maximum with
  `NUM_STAGES` untouched at 2.
- Still never run on a GPU: no correctness or timing number is claimed. `tasks/debug_kernel_server_perf_4_mode.md`
  item 18 bullet 1e records the arms, the SASS check (396 `HMMA` + 48 `GMMA` in one kernel) and what the H200 run
  needs.

## v2.63.0 — A more accurate step model would not change the chunk (2026-10-28)
- `chunk_predictor.py --model-error e` scales the prediction and re-solves: the chosen chunk moves ~1:1 with model
  error (median 2 304 at -5%, 2 176 at 0, 2 048 at +5%). Swapping the ten-term linear model (4.7% MAPE) for the
  step MLP (3.4-4.0% on merged steps) would move the median chunk ~1.3%, an order of magnitude below the serving
  noise floor.
- So the simple model is sufficient on the merits: sizing a chunk is a threshold question (where the step time
  crosses the budget), and thresholds tolerate error that point predictions do not. A direct regression head to the
  max safe chunk is unnecessary for the same reason. Written into paper 5.2.3.

## v2.62.0 — Cross-role shared-prefix KV reuse closed on fan-out too (2026-10-28)
- `tasks/perf_model/fanout_role_overlap.py`: the gating number for the one kernel mechanism that only C2 makes
  possible (both roles in one step, so the plan could group by KV page instead of by request).
- Fan-out has the spatial sharing the TraceLab replay lacks, by construction: every program of a task starts from
  the same L-token task prefix. Page overlap ceiling **98.6%** of the prefilling program's KV reads (94.6% at the
  Llama 32K configuration), given a resident sibling.
- **It is still closed, by the measurement that closed multicast:** compiling FA3's KV loads out entirely makes the
  step only 0.1-1.3% faster, because they are hidden behind compute and hit in L2. So cross-role dedup is worth
  <= 1.3% at any overlap. The high overlap shows the workload has the right structure, on a step whose KV traffic
  is not the bottleneck -- it would need a much smaller model or far less bandwidth per flop to matter.

## v2.61.0 — No lab bookkeeping in the paper (2026-10-28)
- New CLAUDE.md rule: the draft mentions a failed experiment, bug, misconfiguration, superseded run or pending
  re-measurement only when it clarifies why we landed on a core method. The FA2/FA3 fusion failing on warp layout
  qualifies -- it is why the design is two launches sharing one plan. A baseline's misconfigured rope base does
  not; that belongs in the task docs.
- Removed from the draft accordingly: the model-fidelity caveat paragraph in §6.4, "Status of the measurements in
  section 6" (which runs are pending a re-measurement), the superseded single-node configuration result, two
  "an earlier H20 measurement supersedes" notes, and the hint at what a broken model's acceptance looks like.
- Kept, reframed from version history into design rationale: why the slack bank is uncapped.
- Tables renumbered 1-21 to close the gap the deletions left.

## v2.60.0 — Paper: readable for a first external read (2026-10-28)
- §5.1 renamed "Fused Kernel for Fine-Grained Tile Dispatch": the design is no longer a persistent kernel, and the
  old title named a mechanism that section 7 shows does not win.
- Length: 30 169 -> ~25 000 words. The annotated bibliography (3 435 words) moves to `tasks/paper_references.md`,
  where the four-bullet reading notes belong; in the submission those entries are ordinary citations. §3.4 cut from
  2 112 to ~1 000 words and §5.1 from 2 660 to ~1 500, mostly by deleting failed-experiment detail that section 7
  already reports with its numbers (persistent executor, DSMEM cluster reduction, L2 ordering, multicast) and by
  cutting "Portability" from two paragraphs to three sentences.
- Deleted a duplicate: §3.4's normalized-latency table and §6.2's speedup table were the same measurements twice.
  The normalized form (ours as a row, 1.00) is kept, in §6.2 where the evaluation belongs.
- Deleted the retracted intra-CTA shared-memory/register table and its three paragraphs of reading; the mechanism
  is now one honest paragraph in §7 with its measured ceiling (1.1-2.3% at 128K) and the replace-a-consumer build
  marked [pending].
- Numbering: tables renumbered 1-22 and figures 1-28 into reading order (the draft had Table 9a three pages before
  Table 9, and a dangling Table 2b and Figure 25b). `/tmp/renumber_tables.py` does this and refuses to write while
  any reference does not resolve.

## v2.59.0 — Offline test for the chunk rule; smaller tables in the docx sync (2026-10-28)
- `tasks/perf_model/test_slo_chunk_rule.py`: lifts the injected `_slo_budget_chunk` out of the patch and runs it
  against synthetic rows, 8/8 passing. The decisive pair is rows one token in: 751 tokens under the old fallback
  against 4 753 with the lifetime projection. The rule has now been wrong twice in ways no serving metric could
  distinguish from "the policy does not help", because the budget is a minimum over rows and one mishandled row
  clamps the batch; this makes that class of bug cost seconds instead of a 16-GPU arm.
- `tasks/paper_sync_gdrive.py`: table text is set to 8 pt with tight cell padding. The script forced every table to
  100% page width while leaving cell text at body size, so the four seven-column tables wrapped in every cell and
  ran most of a page. The table check now also verifies that every run in every table carries the smaller size
  (27 tables, 27 bordered, 27 resized), so a silent revert to body text fails the sync.

## v2.58.0 — The chunk-rule regression was a guard bug, not the policy (2026-10-28)
- The `:life` arm finished and its own log shows the lifetime budget NEVER engaged: `budget_ms=92.6 ... capped=7001
  of calls=7001`, i.e. every step was clamped to the ~90 ms per-step allowance. Score chat 86.3% / agent 91.2% /
  2 323 tok/s against rung L0's 91.2 / 94.3 / 2 493, so it repeated the earlier arm.
- Two guard bugs found and fixed in `apply_slo_budget_chunk_20261026.py`:
  1. A row with no output token yet (`o == 0`) has no mean TPOT at all -- its latency so far is TTFT -- but the
     demotion test `spent > s_r * max(o, 1)` fired on it after one ordinary step, and since the budget is a MIN
     over resident rows, one just-admitted row clamped every step. Such rows now contribute no bound.
  2. The demotion test used the INSTANTANEOUS mean. A row three tokens in whose first step waited behind a long
     prefill chunk fails it on every early step, so the same clamp returned one token later. Under `:life` the test
     now asks whether the row can still reach its target BY COMPLETION (`spent > s_r * L`).
- The rule's log now prints `fresh=`, `life=`, `nolife=`, `demoted=`, the learned rate and the per-tenant output
  length, so a silent fall-through can no longer be mistaken for "the rule did not help".
- Chunk-size prediction is standard practice (SLOs-Serve, FlowPrefill, PD-Multiplexing); the regression was ours.
- Also: `cp -a` of the symlinked `$W/python` wrote the patch into the main serving tree. Reverted by structure on
  all four pods and re-verified marker by marker; `python_slo_life` is now a real directory. Recorded in memory.

## v2.57.0 — Queue the tail-bounded chunk arm at both loads (2026-10-28)
- `tasks/pd_colocation/mn_tailchunk_16g_20261028.sh`: the mean-TPOT chunk rule with the inter-token tail target as
  its second bound (`stepcap=600`, the chat p99 target, in place of 2000 ms), at alpha 3 and alpha 6, chained
  after the overload decomposition. Only our arm runs; the L0 / L1 / pp / mc / pdmux references come from the
  decomposition drivers on the same cluster and window set.
- Also adds the H200 boot helper (`unified/h200_boot_20261028.sh`) that keeps all four GPUs loaded while the
  re-forked trial provisions, since idleness is what got the previous trial reaped.

## v2.56.0 — Which budget the SLO chunk rule should use: two of the three forms are dead ends (2026-10-28)
- `chunk_predictor.py` gains `--rule {step,life,amort,tail}` and `--out-len`, and replays the 10,185 recorded
  merged steps under each. `amort` (remaining slack over remaining output) is *identical* to the per-step form,
  because its minimum is attained at a freshly admitted row where it equals target x tokens by construction -- so
  the structural reason the deployed rule regressed is not fixable by amortizing.
- `life` (projected mean TPOT at completion) is the correct reading of a mean SLO and useless as a bound: it grants
  16.3 s and saturates the chunk maximum on 100% of steps. Its usable band is about 1.5 ms wide (a 14.5-16 ms
  target against a 13.9 ms achieved rate). The running `:life` arm with `stepcap=2000` therefore measures an
  effectively uncapped chunk.
- The bound that binds is the inter-token tail target, also stated by the workload: `--rule tail` at 600 ms gives a
  median chunk of 2 176 tokens while admitting more than the run used on 65% of steps and less on 35%.
- Paper 5.2.3 rewritten: the "banked slack raises the chunk" table is deleted (it described a form the cluster
  measurement contradicts) and replaced by the measured three-form comparison and the tail bound.

## v2.55.0 — Paper: one kernel claim everywhere, and C3's worker policy is no longer [pending] (2026-10-28)
- The abstract, the introduction's results paragraph, the outline bullet and the conclusion all quoted the
  superseded H20 figure (1.43-1.51x over sglang's single FA3 call) with the fair baseline left [pending], while
  §6.2 already carried the H200 ladder. All four now lead with the claim the user asked for -- 4.0-8.7% over naive
  FA3 + stock FA2 on Qwen2.5-7B, 2.8-5.8% on Llama-3.3-70B, 2.0-2.8% on GLM-4.5-Air -- and state the 1.86-2.92x
  against sglang's call as mostly the role split, which is not ours.
- Table 4c's C3 row: the worker policy's own contribution was [pending] and is now the measured L1-vs-L0 rung.

## v2.54.0 — Queue the overload point (alpha 6), which no run on this cluster had reached (2026-10-28)
- `tasks/pd_colocation/mn_overload_decomp_16g_20261028.sh`: the alpha = 6 decomposition (L0, L1, pp, mc, pdmux) on
  the 16 H20s, chained after the :life arm. Every earlier driver declared `for A in 3 6` and was stopped or
  re-scoped before leaving alpha 3, so the second half of the node-level goal -- better TPOT attainment under high
  load -- is entirely unmeasured. Deadline ordering is neutral at capacity by construction, so this is where it
  earns its place or does not.
- `superset_report.py` now reads any sweep tag (superset / decomp / overload / life) and takes `--ours <arm>`, so
  the decomposition rungs can be judged against the baselines in one table.

## v2.53.0 — H200 trial re-forked after an under-utilisation kill; paper Table 9c (2026-10-28)
- The authorized 4-GPU H200 trial `bf7d041fb8313d8a` was killed at 2026-10-28 04:02 UTC with `exit_code -77777`,
  "资源利用率不达标被查杀" -- resource under-utilisation, because it sat idle while the overnight work ran on the H20
  cluster. Read the reason with `merlin-cli job list-trial-exit-info --json '{"trial_id":"304747380"}'`; `job-v2 runs
  get` only reports `trial_status: killed`. Under-utilisation is the one reason the user's rule says to re-fork rather
  than fall back to H20, so the replacement trial `3f9e6a55ef7ae47e` (vibecuda, 4x H200) is queued, and the GPUs are
  to be kept loaded.
- Fork body recorded in the task doc: the current CLI (0.2.298) rejects the older `source_sid` / `name` / `overrides`
  field names and requires `roles[].queue_name`.
- Paper 6.4 gains Table 9c, the scheduling decomposition (sglang prefill-prioritized / L0 / L1) with the noise floor
  stated, and §7 "Result: schedule" now carries the end-to-end confirmation that deadline ordering alone is neutral
  and the per-step SLO chunk is a regression, with its structural cause.

## v2.52.0 — Decomposition L1: EDF ordering is neutral at alpha 3 (2026-10-28)
- L1 (L0 + EDF, sglang's default chunk), acceptance 1.993: chat 90.9% / agent 93.9% / 2481 tok/s, chat TPOT p50/p90
  11.4/32.0 ms, agent TPOT p90 40.4, chat ITL p99 185.5. Against L0 (FCFS: 91.2 / 94.3 / 2493, 10.8/29.8, 38.0,
  162.9) every difference is inside the noise floor -- but all of them point the same way (slightly worse), and both
  p99 ITL tails grow. EDF alone neither helps nor hurts at this load, which satisfies the "no regression at low load"
  goal; its case rests on overload (alpha 6, where the old cluster measured +7.5% goodput).
- Attribution at alpha 3 is now: tree + A6 + four-mode + router = the entire gain over sglang prefill-prioritized;
  EDF neutral; SLO-budgeted chunk a large regression. The :life redesign is running.

## v2.51.0 — Paper 6.4: Table 9a, our stack against sglang's own colocated schedule on one cluster (2026-10-28)
- New first result in 6.4, before the disaggregation tables: L0 vs sglang prefill-prioritized, same 16 H20s, same
  superposed TraceLab + chat at load factor 3, both arms checked by draft acceptance (1.99 / 1.84).
- Reported with every metric, including the one we lose (chat p99 ITL +25%), the bar stated as partial (chat TPOT and
  TTFT clear 30%; agent TPOT -19.5%; combined goodput +5.2%), and the earlier scheduler layer's regression and its
  diagnosed cause stated plainly, with the redesigned budget marked [pending].

## v2.50.0 — Decomposition L0: our stack beats sglang prefill-prioritized on every metric; the chunk rule is the loser (2026-10-28)
- L0 = our tree (H20 lineage), A6 kernel, four-mode MTP step, our cache router, with sglang's FCFS and sglang's own
  default chunk; acceptance 1.993. vs sglang prefill-prioritized at alpha 3: chat 91.2 vs 83.2% (+8.0 pts), agent
  94.3 vs 93.2%, agent tok/s 2493 vs 2448; chat TPOT p90 29.8 vs 41.9 ms (-28.9%), agent TPOT p90 38.0 vs 47.2
  (-19.5%), chat TTFT p90 0.60 vs 2.81 s (-79%). Superset holds for L0.
- Bar PARTLY met: chat TPOT p90 and chat TTFT clear 30%; agent TPOT p90 (-19.5%) and throughput (combined goodput
  +5.2%; agent +1.8%, chat +11.5%) do not; chat p99 ITL is worse than pp (162.9 vs 130.8 ms).
- Confirms the diagnosis: adding the SLO-budgeted chunk turns this win into a loss (r5/r6). The gain over sglang comes
  from tree + kernel + four-mode + router; our scheduler layer currently subtracts. L1 (+EDF) and the :life redesign
  run next.

## v2.49.0 — Lifetime-mean budget (:life) for the SLO-budgeted chunk; queued after the decomposition (2026-10-28)
- Fix for the diagnosed root cause (the budget is a MIN over resident rows, and a just-admitted row has banked
  nothing, so it pins every step and the mean-TPOT rule degrades to a per-step bound). ":life" budgets each row by its
  projected mean TPOT at COMPLETION: step <= s_r * L - spent - remaining * rate, with L the tenant's expected output
  length and rate the row's own measured ms per token.
- No tuned prior: L is the running mean of output lengths of requests that actually FINISHED (learned online as rows
  leave the batch), never below what the row has produced; until 8 have finished the old bound applies. For scale,
  TraceLab agent rounds have median 218 / mean 516 output tokens.
- Applied to a copy of our arm's scheduler: parses. Built into python_slo_life on all four pods; mn_life_16g_20261028.sh
  (qcap + stepcap=2000 + life, alpha 3) chained to run after the decomposition, against r5's prefill-prioritized.

## v2.48.0 — qcap does not fix the loss; the real cause is the chunk rule starving prefill (2026-10-28)
- r6 ours (qcap + stepcap=2000): 84.6% chat / 89.3% agent / 2237 tok/s, chat TPOT p90 43.9, agent TTFT p90 9.52 s.
  The cap works (capped 8001/8001, none at 16,384; qcapped 490) but TPOT p90 did not improve (43.6 -> 43.9): the
  ~6 s stall diagnosis was WRONG.
- Real loss: agent TTFT p90 9.5 s vs sglang prefill-prioritized's 3.4 s. The rule's typical chunk is 147-333 tokens,
  so long agent prefills crawl and miss the 10 s agent TTFT SLO, which is the agent attainment pp wins on. pp protects
  TPOT about as well (41.9 vs 43.9 ms p90) without throttling prefill.
- Root cause from the rule's structure: the budget is a MIN over resident rows, and a just-admitted row has banked
  nothing, so one fresh row pins every step to its bare per-step allowance (~100 ms). Under steady admission there is
  nearly always a fresh row, so the "mean TPOT" rule degenerates to a per-step ITL bound.
- Next: decomposition L0/L1 vs pp to confirm the chunk rule is the losing layer; redesign the budget so a fresh row's
  first steps are not binding.

## v2.47.0 — Sweep reordered: the qcap-fixed arm runs next against prefill-prioritized (2026-10-28)
- Stopped sweep r5 after its pp arm (local drivers by PID, mn_kill.sh on all pods, 16 GPUs free in 15 s), and the
  two chained follow-ups that keyed on its completion marker.
- mn_superset_v2_16g_20261028.sh (r6): ours = the qcap-fixed arm (python_slo_qcap,
  SLO_CHUNK=...:demote:ttft:qcap:stepcap=2000), first at each load; mc and pdmux at alpha 3; ours, pp, mc, pdmux at
  alpha 6. pp at alpha 3 is reused from r5 (same cluster). The PD re-measure is chained after it in /tmp/run_v2.sh,
  each stage behind the GPU-memory preflight.

## v2.46.0 — First superset verdict: VIOLATED at alpha 3 -- sglang's default prefill-prioritized wins (2026-10-28)
- Same cluster and load, superset_report.py: sglang prefill-prioritized 83.2% chat / 93.2% agent / 2448 tok/s,
  TPOT p90 41.9 / 47.2 ms, against ours (uncapped SLO chunk) 84.4 / 90.3 / 2259, 43.6 / 48.8 ms. pp wins agent
  attainment (+2.9 pts), goodput (+8.4%) and both TPOT tails; chat is a tie. Bar NOT MET (TPOT p90 -3.4%, goodput
  -7.7% vs the best valid baseline).
- Not a broken baseline: pp is valid and its acceptance (1.837) is LOWER than ours (2.012), yet it still wins.
- Treated as a defect per the superset rule. Diagnosed cause: the uncapped rule's ~6 s single-step stalls. The sweep
  is reordered so the qcap-fixed arm runs next against pp, instead of running two more baselines against an arm
  already known to be defective.

## v2.45.0 — PD re-measure queued; full evaluation pipeline now chained end to end (2026-10-28)
- mn_pd_remeasure_16g_20261028.sh: stock PD disaggregation (disagg_p2, sglang PD router, upstream tree) and
  conditional disaggregation (cond_p1, Dynamo defaults, our tree and router) at alpha 3 and 6. Re-measured because
  every earlier disagg_p* number ran the upstream tree with the rope-base defect; ROPE_THETA now reaches it as a
  top-level override field.
- Pipeline now runs unattended in order: superset sweep r5 (ours, pp, mc, pdmux at alpha 3 and 6) -> qcap follow-up
  (ours with the per-step cap) -> PD re-measure. Each stage starts only after the previous finishes, behind the
  GPU-memory preflight. superset_report.py judges the superset claim and the bar from the logs.

## v2.44.0 — Per-step cap for the SLO-budgeted chunk (qcap / stepcap), built and queued (2026-10-28)
- Fix for the defect the r5 ours arm exposed (unlimited banking spent ~6 s of slack in one 16,384-token step).
  apply_slo_budget_chunk_20261026.py gains ":qcap" -- bound one step's budget by the remaining TTFT slack of the
  tightest WAITING request, so no step is longer than the most urgent queued request can afford -- and
  ":stepcap=MS", a hard ceiling for an empty queue. Both derive from SLO deadlines, not a tuned chunk constant; the
  mean-TPOT banking for resident rows is unchanged. SLO_CHUNK log lines now report qcapped=.
- Validated by applying it to a copy of the A6 bundle's scheduler.py (the one our arm runs): parses, and the SLO
  hook, the cap and late-demote EDF are all present.
- Built into a SEPARATE tree python_slo_qcap on all four pods, leaving python_slo untouched for the sweep's alpha-6
  ours arm. mn_qcap_16g_20261028.sh (ours + qcap + stepcap=2000, the chat TTFT SLO, at alpha 3 and 6) is chained to
  start when the superset sweep finishes. superset_report.py reads both logs and shows ours_qcap as its own row.

## v2.43.0 — First valid "ours" result on the new cluster, and a chunk-rule defect it exposes (2026-10-28)
- Superset sweep r5, ours at alpha 3 (EDF + SLO-budgeted chunk, A6, cache router): chat 84.4% / agent 90.3% /
  2259 tok/s; chat TPOT p50/p90 15.5/43.6 ms; agent 21.4/48.8 ms; chat TTFT p90 2.80 s. Valid: acceptance 2.012;
  A6 and SLO_CHUNK both active in the worker logs.
- Worse than our own earlier best config (cache + FCFS, chunk 2048: 90.6/94.2/2471, chat TPOT p90 31.9, chat TTFT
  p90 0.67 s) beyond the noise floor. Cause CONFIRMED from the rule's log: 93% of steps are capped (chunks down to
  256), but ~7% run the full 16,384-token chunk, with budgets up to 5,867 ms -- unlimited banking lets ~6 s of banked
  slack be spent as ONE stall, which lands in the p90 TPOT tail and in chat TTFT.
- Fix recorded: keep the mean-TPOT budget but cap what one step may spend, preferably by a cap derived from the chat
  TTFT SLO (an SLO, not a tuned constant). Baselines for the superset verdict still running.

## v2.42.0 — Step-model claim corrected: the linear model matches the MLP on merged steps (2026-10-28)
- Re-ran fit_4mode_step.py --target gpu --pool on the four alpha-3 STEPTRACE logs (@3:4), held-out second halves.
  Merged steps: lin4mode 2.8-4.2%, overlap 2.4-4.4%, GBT 2.8-3.6%, MLP 3.4-4.2%, MLP with acceptance 3.4-4.0% (the
  paper's figure reproduces). Decode-only: lin4mode 34-38% vs MLP 18-23%.
- The paper said "a linear model does not suffice" with the acceptance MLP as the model of choice. That is wrong for
  merged steps -- all five models are within ~1 point there; the MLP wins only on decode-only steps, which the chunk
  rule never sizes. Table gains a linear column and the text now says so. (Per-worker, non-pooled fits made the MLP
  look worse still, 5.8-14.9%, which is why the pooled protocol is the one reported.)
- This supports the user's "simple methods" framing: the closed-form linear model the chunk rule uses is as accurate
  as the learned one on the steps it acts on.

## v2.41.0 — smoke9 PASSES end to end; superset sweep running with our arm first (2026-10-28)
- smoke9: agent completed 652 rounds and chat 300, 0 tracebacks -- the first fully working smoke arm on this
  cluster. The gate's post-smoke GPU check aborted on CUDA-context release lag (all 16 GPUs read 0 MiB moments later);
  the sweep wrapper now retries that check for up to 3 min.
- Router: the calibration's --decode-grid points at a file from the OLD cluster, and Router.__init__ loads it
  unconditionally, so our arm's router would have crashed at startup. Our arm uses the plain `cache` policy, which
  routes on prefix hit and in-flight count only and never reads the perf model the grid feeds (router.py choose()),
  so the flag is stripped -- behaviour unchanged.
- mn_superset_oursfirst_16g_20261028.sh: same arms, OUR arm first at each load, so a failure of its never-launched
  serving tree (H20 lineage + A6 + SLO-chunk patch) shows in ~30 min, not after ~90 min of baselines. Restarted
  cleanly (mn_kill.sh, all GPUs 0 MiB). Running as r5.

## v2.40.0 — The "ours" arm's serving tree rebuilt on the correct lineage before the sweep reached it (2026-10-28)
- Found while smoke9 ran: the stage-2 push used the REPO tree as the serving tree, which has no A6 hooks and no
  late-demote EDF. Every prior cluster result ran the H20 serving tree (upstream 3fef129 + 31-file overlay
  up_overlay_h20_20261024.tgz + A6 bundle files). KERNEL=a6 would have died at launch, and python_slo carried the
  wrong scheduler lineage -- the sweep's "ours" arm would not have been our system.
- Rebuilt on all four pods: h20py = upstream + overlay; A6 backend/scheduler copied in under the deploy guard (the
  overlay's scheduler.py md5 f8fbbdb95e9b is exactly the guard's expected base); A6 executor modules into $W/a6;
  python_slo = h20py + SLO-chunk patch; $W/python -> h20py. Verified: A6 hooks, late demote, SLO patch, RoPE fix
  present, all touched files parse.
- The baseline-only smoke arm could not catch this. Server-side dry-run (import the serving tree) noted as needed.

## v2.39.0 — Agent client dry-runs clean; the slim trace reproduces the old replay round for round (2026-10-28)
- smoke8: the slim trace arrived intact (md5 d55fcf85dd87, 629 s). The agent client then failed with
  "DatasetRow.__init__() got an unexpected keyword argument 'session_id'": upstream's DatasetRow lacks the
  session-replay fields our TraceLab loader passes. Added warmup_only, session_id, round_index, inter_round_delay_s
  (all defaulted, so other loaders are unchanged) on all four pods.
- Instead of finding the next gap with another 15-minute smoke run, dry-ran the agent client on pod 0: its own
  cli_main and argument parser, run_benchmark stubbed to call get_dataset and exit. DRYRUN_OK in 38 s: 13,353 rows,
  324 sessions, 281 warmups.
- Independent fidelity check of the slim trace: 13,353 - 281 = 13,072 real rounds, which is exactly the old
  cluster's best-config arm with the FULL trace (11,424 issued + 1,648 not issued = 13,072).
- Overlay README now lists the DatasetRow extension and the dry-run as deploy steps 6 and 7. smoke9 launched.

## v2.38.0 — superset_report.py: judge the sweep by the agreed rules the moment it lands (2026-10-28)
- tasks/pd_colocation/superset_report.py reads the sweep log's SCORE / ACCEPT lines and prints, per load: every arm's
  chat and agent attainment, agent goodput and TPOT p50/p90; an acceptance VALIDITY gate (default 1.8; the
  rope-base-broken baseline measured 1.42 against a clean 1.99); the SUPERSET verdict listing every metric where a
  valid baseline beats us; and the 30% TPOT / 10% throughput bar against the BEST valid baseline per metric.
- An invalid baseline that would have beaten us is called out explicitly with "RE-RUN it with a fixed model rather
  than drop it" -- silently excluding a winning baseline would make the comparison non-credible.
- Tested on a synthetic log in the drivers' exact format, including a deliberately broken baseline: gate, superset
  check and bar all fire as intended.

## v2.37.0 — Slim windowed trace: 4.6 MB, field- and span-equivalent, ~10 min to the pods (2026-10-28)
- The 22 MB windowed trace was still uploading at ~0.5%/min (~2 h). Larger exec chunks are not an option: 400 KB
  chunks are rejected (ok=False), and 60 KB costs ~6.5 s per call.
- tracelab.py reads only eight record fields and, from timing_events, only the min and max timestamp
  (_record_span_ms). make_windowed_trace_20261028.py --slim keeps exactly those, with the earliest and latest event
  per round: 4,633,585 bytes (md5 d55fcf85dd87), 158,849 records, ZERO mismatches in kept fields or per-record span.
- Superseded uploads stopped by PID; chain8 pushes the slim file, requires the md5, then runs smoke8 -> superset r4.

## v2.36.0 — Paper sections 3.1 and 6.2 filled from measured data; stale H20 kernel figures replaced (2026-10-28)
- 3.1 step census (87,932 steps of the cluster replay): chunk-plus-rows steps are 11.6% of steps but 67.2% of GPU
  time; rows-only 83.9% / 27.6%; prefill-only 4.5% / 5.1%. Attention share of a long-context mixed step stated as an
  arithmetic estimate (1.08e14 FLOP = 0.73 s on H20 vs a 0.727 s step); the per-layer split of Figure 3 stays pending.
- 6.2: the H20-era "1.43-1.51x faster than sglang's FA3 call" replaced by the H200 ladder (1.86-2.92x vs sglang's
  call, 4.0-8.7% vs naive FA3 + FA2); Llama-3.3-70B and GLM-4.5-Air now from the H200 replay (1.028-1.058x and
  1.020-1.028x vs naive). Bar status judged: not met by the kernel alone (2-9% vs the strongest baseline); the 30% bar
  must be met at serving level.
- Pending markers 75 -> 70; the rest are cluster results awaiting the sweep, and two figures.

## v2.35.0 — Windowed TraceLab cut: same replay, 2.4x smaller, so it can actually reach the pods (2026-10-28)
- The full release (357,161 rounds, 54 MB) was pushing at ~16% per 25 min, i.e. ~2.5 h, through the cluster's
  base64-over-exec channel. tasks/pd_colocation/make_windowed_trace_20261028.py keeps only the whole sessions the
  windowed replay can select: 689 of 4,265 sessions, 158,849 rounds, 22.3 MB (md5 42f498f35197).
- Exact by construction, and checked: tracelab.py measures each window start in seconds after the trace's FIRST
  round, so the session holding the earliest event is always kept -- dropping it would silently shift every window
  onto different hours. Verified the origin is unchanged and that all 35,543 in-window rounds are present.
- Pushing to pod 0; chain7 launches smoke7 (then superset r4) only if the arrival md5 matches.

## v2.34.1 — H200 trial killed; item 18 closed at compile time; its paper section cut to match its (zero) gain (2026-10-28)
- H200 trial bf7d041fb8313d8a: trial_status killed, completed 2026-10-28 04:02 UTC, no stop reason recorded. Not
  re-forked: the reason (idle underutilization vs preemption) is unknown, H200 is scarce, and all remaining work is
  multi-node on H20. Kernel results stand; the agent's ptxas numbers agree bit-for-bit between the dev box and the
  trial on the default-build control.
- Item 18 (q64 x q16 intra-CTA overlap) closed by the kernel worker at compile time. Premise refuted: q64 shrinks the
  CTA (384 -> 256 threads) rather than freeing a warpgroup. The guest spills the chunk's registers at either tile
  (7.5 KB vs 0). Shared memory decides it: both roles live at once need 234,544 B at q128 vs the 232,448 B limit, so
  it launches only at q64, where ptxas serializes the chunk's wgmma (C7512) even with no rows present. Built behind
  FUSED_Q64_PIGGYBACK=1 (flashinfer 03d3759), never run.
- Paper 5.1's account of it cut from five paragraphs to one, per the instruction that writing be proportional to
  gains. Duplicate changelog heading v2.32.0 renumbered v2.32.1.

## v2.34.0 — the piggyback's two constraints point opposite ways: only q64 can launch, and q64 is where wgmma serializes (2026-10-28)
- The corrected mechanism now has numbers. `FUSED_Q64_PIGGYBACK=1` with the CTA widened by one warpgroup compiles to
  **128 reg / 4 460 B** spill at the 128-row chunk tile (512 threads) and **144 reg / 3 716 B + C7512** at 64 rows
  (384 threads). Widening the CTA lowers the spill against hosting at the same width (7 528 / 5 448 B) because the
  guest's live range hangs off a thread-index branch that returns, but at 512 threads the allocation falls to 128
  registers per thread, below the 168 FA3's consumers need to stay spill-free.
- `unified/probe_fused_smem_sizes_20261028.cu` reports the shared-memory demand at compile time, by reading the sizes
  out of a deliberately failed template instantiation (no GPU, no runtime). The piggyback is the only configuration
  where both roles' shared memory is live at once, so the request is their SUM: FA3 163 920 B at q128 / 147 536 B at
  q64, rows 69 632 B, giving **234 544 B at q128 — over the 232 448 B opt-in maximum by 2 096 B, so the CTA cannot be
  created** — against 218 160 B at q64, which fits with 14 288 B spare. The q128 figure independently reproduces the
  number already recorded for the co-resident two-stage layout, which checks the probe.
- **So the two constraints point opposite ways: the only launchable tile is the one that costs the chunk its `wgmma`
  pipelining.** That, not the spill count, is the reason item 18 is closed; paper 5.1 and Table 2b now say so.
- Toolchain cross-check, which is what lets the arms be read as one table: the H200 trial dropped its ssh listener
  mid-session, so the piggyback arms were compiled on the dev box. `unified/gen_fi07_config_20261028.py` renders
  flashinfer 0.7's `batch_attention_config.inc` there (`PersistentParams` is defined in that generated header, and
  the box's cache held a 0.3.1 render, which fails with 54 "no member" errors). The shared arm — default build at
  q128 — gives 168 reg / 33 424 B on both hosts, so the two agree bit for bit; that run is also the regression
  control showing the piggyback edits leave the default build unchanged.

## v2.33.0 — lin4mode 25% vs 4.7% resolved: same model, different step types (2026-10-28)
- Per-type error of the lin4mode step model on the 87,932-step replay: merged 4.7% (5.0% held out), prefill 6.2%,
  decode 29.0%, all steps 25.2% (24.0% held out). The paper's "25%" is the all-step figure, dominated by decode-only
  steps (84% of steps, 10-20 ms each). The chunk rule only sizes chunk-carrying steps, where it is 4.7-5.0%.
- Paper 5.2.2 rewritten to give the per-type split and say which figure governs the chunk rule; the separate 45%
  launch-to-launch-gap claim kept (an intermediate edit had dropped it).

## v2.32.0 — smoke6: last missing piece is the TraceLab trace file itself (2026-10-28)
- smoke6 again completed chat end to end (300 requests, mean TTFT 53.6 ms, TPOT 3.4 ms, 0 tracebacks; the dataset
  registration resolved). The agent client then failed with FileNotFoundError on
  /dev/shm/w/workloads/tracelab/syfi_coding_trace.jsonl.gz: provisioning creates that directory and never fills it.
- The only local file I had pushed before (/tmp/h200prov/tracelab.jsonl.gz) is the LENGTH-ONLY variant: no
  timing_events. Pushing it would have silently turned the timestamp replay (AGENT_MODE=tracelab_ts) into a plain
  length replay. Not used.
- Located the real full release at workloads/tracelab/syfi_coding_trace.jsonl.gz in the repo checkout: 357,161
  rounds, md5 465ccfb7d0d9, per-round timing_events, spanning 253.4 days. Two other local copies are older subsets
  (135K and 140K rounds, different md5s). Verified all 48 window starts in tracelab_top_hours_20261026.json fall fully
  inside the trace's span. Pushing it to pod 0 (the bench client's pod), with an md5 check on arrival.

## v2.32.1 — item 18's gating compile fails: q64 does not free a warpgroup and costs FA3 its wgmma pipelining (2026-10-28)
- `unified/gate_q64_registers_20261028.py` compiles the co-compiled fused translation unit (FA3's wgmma mainloop and
  the FA2 `mma.sync` runners in one ptxas allocation) at FA3 tile 128 and 64 and reads registers and spill bytes. It
  drives `nvcc -c` directly rather than `cpp_extension.load`, which keys its cache on the extension name and serves a
  stale `.so` when only `-D` flags change.
- Result, H200 toolchain, `FusedFA3PersistentKernel` spill STORES: FA3 alone is **168 reg / 0 B** at q128 against
  **144 reg / 1 560 B** at q64; adding only the 16-row runner costs **7 528 B** (q128) / 5 448 B (q64); all three FA2
  classes cost 33 424 B / 19 840 B. `ptxas` emits C7512 "wgmma.mma_async instructions are serialized due to
  insufficient register resources" at q64 in every build and never at q128, and `--maxrregcount=240` does not move it.
- Root cause of the proposal's premise being wrong: both FA3 implementations derive the consumer warpgroup count from
  the tile (`NUM_WARPS = ((CTA_Q/64)+1)*4`; upstream's `AtomLayoutQK = Shape<Int<kBlockM/64>,_1,_1>`), so a 64-row tile
  HALVES the CTA to 256 threads rather than freeing a slot in a 384-thread one, and each consumer owns 64 query rows at
  either tile, so per-thread register pressure is unchanged.
- The corrected mechanism is implemented behind `FUSED_Q64_PIGGYBACK=1` (flashinfer `persistent_fa3_fused.cuh`,
  `fused_barriers.cuh`): the CTA is launched one warpgroup wider than FA3's thread count, that warpgroup drains only the
  16-row tile class concurrently with FA3's mainloop, shared memory is forced disjoint (aliasing is only sound when the
  roles do not overlap in time) and the FA2 named-barrier ids move past CUTLASS's `FirstUserBarrier` (arrivals are
  counted, not tagged, so a shared id releases one role's barrier on the other's arrivals). Defaults are byte-for-byte
  unchanged at `=0`; **compiled, not measured** — the H200 trial dropped its ssh listener before the probe arms ran.
- Item 18 is recorded as CLOSED as proposed, with the measured upper bound (1.1–2.3% of the step at 128K, 4.5–10.2% at
  32K) noted so no gain is claimed above it. Its value to the paper is as evidence for the 5.1 claim that POD-style
  intra-SM overlap does not carry to Hopper's warp-specialised FA3.

## v2.31.0 — smoke5: chat client end to end, teardown and done-marker both work; dataset registered (2026-10-28)
- smoke5 is the first run with all three fixes live. Chat completed end to end (300 requests, mean TTFT 54.2 ms,
  mean TPOT 3.4 ms, p99 ITL 6.5 ms, 0 tracebacks). The bench reported `MNBENCHDONE status=1` at once instead of
  holding the cluster 90 min, and mn_kill.sh returned all 16 GPUs to 0 MiB.
- The agent client failed with "ValueError: Unknown dataset: tracelab-claude": tracelab.py was copied into
  datasets/ but never added to DATASET_MAPPING in datasets/__init__.py. Registered on all four pods; resolves to
  TraceLabClaudeDataset on each.
- The overlay README now carries a five-step deploy checklist including the registration, which it omitted.
- smoke6 launched behind the GPU preflight; superset r4 follows if both agent.json and chat.json appear.

## v2.30.0 — True root cause of the orphaned servers: mn_kill.sh was never deployed (2026-10-28)
- The GPU preflight caught it: after smoke4's "normal" teardown, all 16 GPUs were still at 86.7 GB, and gate5
  aborted instead of OOMing. mn_run.py's teardown is `bash .../mn_kill.sh`, and that script was never pushed to the
  re-provisioned cluster, so EVERY teardown failed with "exit status 2" (printed just before MNRUNDONE and ignored).
  This corrects the earlier diagnosis that blamed only kill -9 on local drivers.
- mn_summary.py and calibrate_h20.py were missing too: the stage-2 bundle was hand-picked. All three deployed; the
  rule is now to deploy every script the drivers reference (grep the drivers for tasks/pd_colocation/*.{sh,py}).
- mn_kill.sh verified: KILL_RC=0 on all four pods, all 16 GPUs free in ~30 s. Its header documents the identical
  86 GB orphan failure from 2026-10-16. gate5 relaunched (PREFLIGHT_OK).

## v2.29.0 — mn_bench.sh always writes its done-marker; a crashed client no longer holds the cluster 90 min (2026-10-28)
- mn_bench.sh ran `[ "$wait_status" -eq 0 ] || exit "$wait_status"` BEFORE `echo MNBENCHDONE`, so any client failure
  exited without the marker. mn_run.py polls for MNBENCHDONE for up to 5400 s before its teardown, so a client that
  crashed in seconds looked identical to one still running and held all 16 GPUs for the full 90 minutes. smoke4 hit
  exactly this (client died at the startup barrier, "released: 0/2 ready").
- Now writes `MNBENCHDONE status=<n>` unconditionally, then exits non-zero on failure. The failure is still reported
  -- in the marker and the exit status -- but no longer masked as "running". Deployed to all four pods.
- smoke4 released by hand-writing the marker into its pod log so mn_run.py took its NORMAL teardown path, rather than
  kill -9 on the driver (which is what orphaned 16 servers earlier).

## v2.28.0 — Bench client installed the documented way; servers now come up clean (2026-10-28)
- smoke4: 16 workers healthy in 94 s with no retries, so the GPU-preflight fix holds and the server path is clean.
  The client then failed on sglang.benchmark.session_replay: the overlay's serving.py imports sibling modules.
- drivers/serving/bench_overlay_20261026/README says to deploy the overlay into UPSTREAM's tree: serving.py,
  session_replay.py, session_concurrency.py into $W/up/python/sglang/benchmark/ and tracelab.py into its datasets/.
  It is written against upstream's srt (imports sglang.srt.utils.network), so grafting it onto our tree cannot work.
  Installed that way on all four pods; import verified on each, with --dataset-name tracelab-claude present.
- mn_bench.sh BENCH_PY default back to $W/up/python, with a comment stating why. Servers still launch via SGL_PY.
- smoke5 + sweep queued behind gate4 rather than killing smoke4: killing a driver mid-run is what orphaned 16 servers.

## v2.27.0 — Cluster blocker resolved: my own orphaned servers held every GPU (2026-10-28)
- CORRECTION: the "launch-path regression" diagnosis was wrong. worker_r0_g0.log was missing only because each
  failed retry is renamed to bindfailN.log; all five smoke3 attempts died of CUDA OOM ("Process 4061538 has
  84.60 GiB"). All 16 GPUs held ~86.7 GB each by servers up to 3.2 h old: the superset r2/r3 and smoke2 server sets,
  orphaned when I stopped their local drivers with kill -9 (which skips mn_run.py's teardown).
- Cleared by PID on every pod (22 + 12 + 12 + 12 processes); all 16 GPUs at 0 MiB.
- gate4.sh adds a GPU-memory preflight that aborts naming busy GPUs, before the smoke arm and before the sweep.
- Kernel item 18 (q64 x q16 intra-CTA overlap) delegated to a separate worker with the measured upper bound
  (1.1-2.3% at 128K) as its sanity check; the cluster path stays on the main thread.

## v2.26.0 — Item 18's upper bound measured per model: 1.1-2.3% at 128K (2026-10-28)
- bench_q64_piggyback_ub_20261028.py: if every row tile rode free inside a chunk CTA the gain equals the share of
  attention time rows cost today. Chunk 2048 (32 q64 tiles) beside 32 rows, H200.
  Qwen2.5-7B 4.50/5.22% at 32K and 1.07/1.33% at 128K (2-mode/4-mode); Llama-3.3-70B 10.20/9.51% and 2.33/2.26%;
  GLM-4.5-Air 6.67/6.37% and 1.75/1.61%.
- Capacity is NOT the constraint (32 q64 tiles vs 32 row tiles, each chunk tile far heavier) -- rows are simply
  cheap: 0.08-0.41 ms against a 7-23 ms chunk.
- At 128K the ceiling is below the 2.2-6.1% PDL already delivers and at/below the serving noise floor. Building it
  is justified as EVIDENCE for the architectural claim, not as a claimable gain. Paper 5.1 carries the bound and
  states that this is why we do not pursue fusion further.

## v2.25.0 — Session end state: cluster arms blocked on a launch-path regression (2026-10-28)
- smoke3's workers never write worker_r0_g0.log: mn_node.sh exits before the launch redirect. smoke2 DID reach
  "16 workers healthy after 482 s" and failed only at the bench client, so the server path can work and this is a
  regression from an edit between the two runs (mn_bench.sh BENCH_PY, client-only and expected inert; or the
  sglang/benchmark copy into ourpy/python and python_slo).
- Next diagnostic recorded in the task doc: run mn_node.sh by hand on pod 0 with the smoke env and capture stderr
  directly -- mn_run.py discards it, which is why three consecutive faults were invisible in the driver log.
- Blocked behind it: the corrected arm set, the alpha 3 / alpha 6 sweep, and the re-measured
  colocation-vs-disaggregation comparison. The kernel results do not depend on it and are complete.

## v2.24.0 — Kernel results presented as one normalized table with ours as a row (2026-10-28)
- User: our method must be a ROW, not the reference frame, in POD Figure 1's convention, across several workloads.
  Tables 4 and 5 merged into ONE table: methods as rows (ours included at 1.00), four workload columns
  (32K/128K x two-mode/four-mode), values = attention time divided by ours, so larger is slower.
- Ordering: sglang's single FA3 call 1.86-2.92, POD's structure 3.45-3.71, our v3 1.34-1.92, naive role split serial
  1.04-1.09, naive on two streams 1.04-1.07, ours 1.00. The two designs a reader expects to be strong are the
  weakest, for the same reason: a tile or an instruction chosen for the wrong work.
- POD repeat: 3.49x / 3.71x against 3.45x / 3.69x, reproducible to better than 1%. Four-mode POD not run; the
  ordering should not change because the gap is in the prefill half's instruction.

## v2.23.0 — Tables 4 and 5 reordered so the comparison direction is unambiguous (2026-10-28)
- The columns were being misread as absolute speed. Each cell is OUR time divided into the baseline's, so larger =
  that baseline is slower; the header now says so and the rows run strongest baseline first.
- Ordering (two-mode, 128K): naive role split on two streams 1.042x, serial 1.044x, our v3 1.769x, sglang's single
  FA3 call 2.921x, POD's structure 3.45x. So sglang's single call and POD are the WEAKEST arms, not the strongest.
- Stated the reason explicitly: sglang's call picks one query tile for every row, so a one-token decode row is
  computed at the chunk's tile width; POD's prefill half issues mma.sync where FA3 issues wgmma. Both are the same
  failure -- a tile or instruction chosen for the wrong work -- which is what the dispatcher avoids. The strongest
  baseline is the naive role split, which we beat by only 4-9%.
- POD rows added to both tables with the measured factors.

## v2.22.0 — POD measured on Hopper instead of called not-implementable (2026-10-28)
- User correction: POD is FA2-based, so it RUNS on Hopper and should be measured, not ruled out.
  bench_pod_proxy_20261028.py times POD's structure (one stock flashinfer FA2 call covering the chunk AND the rows)
  against ours (stock FA3 chunk + stock FA2 rows) on identical shapes: 24.29 ms vs 7.03 ms at a 128K prefix with a
  2048-token chunk (ours 3.45x faster), and 6.15 vs 1.67 ms at 32K (3.69x).
- This is an UPPER bound for POD on Hopper: its prefill half issues mma.sync where ours issues wgmma, and any
  CTA interleaving POD adds sits on top of that same kernel, so it cannot recover the instruction gap.
- Table 4's POD row now carries the measured factors rather than "not implementable"; the rationale paragraph cites
  both context points.

## v2.21.0 — Kernel section condensed to match its gains; Tables 4 and 5 added (2026-10-28)
- User: writing on the kernel items must be proportional to their gains. The POD-on-Hopper argument is cut from
  five numbered paragraphs to one, keeping the hardware rationale (wgmma/tcgen05 CTAs sized to fill the SM) and all
  the measured consequences inline, and the section now ends by saying explicitly that it is kept short in
  proportion to a 4-9% gain so the paper's space goes to the schedule.
- Table 4 (two-mode) and Table 5 (four-mode) give our dispatcher against every alternative we could run: sglang's
  single FA3 call (2.107x / 2.921x two-mode; 2.647x / 1.861x four-mode), the naive role split serial (1.079 / 1.044;
  1.087 / 1.058), the same on two streams (1.040 / 1.042; 1.070 / 1.056), our earlier v3 build, and POD-style
  single-kernel fusion marked not implementable on Hopper with the reasons above it.
- The q64-with-q16 exception is stated as the register-wall consequence with its 94.8% reach, reported as reach
  only, no gain claimed.

## v2.20.0 — Item 18's causal chain stated precisely: the register wall is why q64 and not q128 (2026-10-28)
- The point, in one line: q128 cannot share a CTA with q16 because its TWO consumer warpgroups at ~168 registers
  each already consume the register budget (the measured 1,436 B spill, 2-6x slowdown); q64 uses ONE consumer
  warpgroup and leaves room for a register-light q16 warpgroup (16 accumulator regs/thread at head_dim 128). So
  q64 x q16 is the only intra-CTA overlap Hopper permits.
- It is immune to the SM-saturation argument because the CTA is already resident and the warpgroup already
  allocated -- unlike two-stream concurrency, which needs spare SMs and finds 4.5% at best, harmful at chunk 2048.
- Paper 5.1 now carries the register arithmetic as the argument rather than asserting the mechanism; the task doc
  carries the one-sentence form at the head of the item.

## v2.19.0 — TWO RETRACTIONS on item 18, and its reach is ~95%, not 1.15% (2026-10-28)
- RETRACTED: the two-stream sweep does NOT test item 18. Item 18 is INTRA-CTA (the FA3 CTA is already resident, its
  second warpgroup already allocated, so rows ride inside it and SM saturation is irrelevant). The sweep also varied
  the CHUNK, i.e. the number of tiles, while flash-attention chose its tile internally -- it never varied the tile
  size. The "q64 hypothesis refuted" claim was wrong on both counts and is withdrawn. What the sweep does establish
  is the SM-saturation evidence for paper 5.1 reason (4): 4.5% at best, 1.006x (harmful) at a 2048-token chunk.
- RETRACTED: the 1.15% reach figure assumed only a chunk REMAINDER runs at q64. If the whole chunk runs at q64,
  every tile frees a warpgroup. Recounted over 10,185 merged steps: 94.75% of row items hostable (127,384 of
  134,443), 91.1% of steps fit their rows entirely, median 31 freed warpgroups against 13 rows.
- Consequence: item 18 is untested and now the most valuable open kernel idea, not a curiosity. Next steps are the
  compile test (does q64 clear the 1,436 B spill?) then an implementation issuing mma.sync from the freed warpgroup.

## v2.18.0 — MEASURED the tile-16-with-FA3 overlap gain; q64 hypothesis refuted (2026-10-28)
- bench_q64_overlap_20261028.py on the H200: chunk/rows concurrency over a 131K prefix. streams/serial = 0.966x at
  chunk 64, 0.955x at 128, 0.973x at 256, and 1.006x (harmful) at 2048; gap-to-ideal recovered 13 / 48 / 41 / 0%.
- The q64 hypothesis is refuted as stated: chunk 128 overlaps BETTER than chunk 64. The deciding variable is the
  absolute size of the FA3 work, not the query tile -- at chunk 64 FA3 already splits the 131K KV across enough CTAs
  to fill the GPU, and at 2048 it fills every SM.
- Absolute ceiling 4.5% of the step at best, 0% at the 2048-token chunk real steps use (row work is ~1% of serial
  time there). Independently consistent with the 1.763 vs 1.759x two-stream result on real captures.
- Both bounds on the mechanism are now measured by different mechanisms: launch-level concurrency <=4.5% (this run)
  and intra-CTA reach 1.15% of row items (plan-only count). The conclusion holds either way. Paper 5.1 reason (4)
  carries the sweep.

## v2.17.0 — The hardware reason POD-style overlap cannot work on Hopper+ (2026-10-28)
- Strengthened reason (1) of the POD argument in paper 5.1 with the architectural cause the user supplied: Ampere's
  mma.sync is warp-level over a 16x8x16 tile, so many warps from several CTAs coexist on an SM and a kernel really
  can interleave a large prefill tile with small decode tiles -- POD's premise is SM slack. Hopper's wgmma and
  Blackwell's tcgen05.mma make the opposite trade: warpgroup/CTA-level instructions over large tiles, driven by warp
  specialisation (producer TMA warp feeding a deep SMEM pipeline, consumer warpgroups issuing back-to-back MMAs). A
  single FA3 CTA is sized to FILL the SM (384 threads x ~168 registers is the whole register budget, plus the K/V
  pipeline's shared memory), so one CTA -- in some configurations one instruction -- consumes the SM's capacity by
  design. The slack POD exploits does not exist and is getting scarcer per generation.
- Conclusion extended to be forward-looking: as the MMA unit grows and moves behind warp specialisation and tensor
  memory, intra-SM co-residency of two differently shaped roles gets harder, so designs depending on spare SM
  capacity should age badly while whole-tile dispatch to the right executor should not.

## v2.16.0 — Item 18 bounded by a plan-only count: 1.15% of row items could move (2026-10-28)
- Counted over the 87,932-step alpha-3 replay: 83,959 steps carry rows; only 9,516 of them (11.3%) also have a q64
  chunk remainder, so 9,516 of 828,705 row items (1.15%) could move into a freed warpgroup, and even where capacity
  exists only 7.7% of that step's rows fit (p90 16.7%).
- The binding constraint is the ratio of q64 chunk tiles to row tiles, not whether the overlap works: one q64 tile
  frees one warpgroup hosting one 16-row tile, while a row-carrying step has many. Agrees independently with
  Table 3's 1.2% of query tiles.
- Verdict: not worth building for throughput. Its value is as EVIDENCE for the architectural claim -- measuring the
  only intra-SM overlap Hopper permits at 1.15% is what makes "POD-style overlap does not carry to Hopper" a
  quantified statement. Paper 5.1 updated with all three figures; no serving gain claimed.

## v2.15.0 — Negative result promoted to a claim: POD-style overlap does not carry to Hopper (2026-10-28)
- User's framing, written into paper 5.1 with the full evidence chain: POD-Attention's fusion needs both roles
  issuing Ampere mma.sync from one kernel body, and on Hopper that fails four ways we MEASURED, not assumed.
  (1) The prefill role must leave FA2: the role split is worth 1.76-7.7x, and running rows on FA3 costs 15-25 points
  of agent ITL. (2) wgmma cannot interleave operand shapes within a warpgroup. (3) The co-compiled kernel we built
  needed FA3's register protocol disabled, spilled every class (1,436 B) and ran 2-6x slower. (4) Overlap therefore
  reduces to launch level: PDL 1.4-6.2%, two streams nothing, and six further mechanisms at or below parity.
- The one intra-SM overlap Hopper permits is FA3 q64 (one consumer warpgroup) + mma.sync 16-row rows in the freed
  warpgroup -- different warpgroups may issue different instruction classes. Table 3 bounds it at 1.2% of tiles,
  since 64-row tiles arise only as chunk remainders. Reported as a mechanism with a ceiling, not a technique.
- Architectural conclusion stated: on Hopper and later the gain comes from giving each role the executor its tile
  shape deserves, not from fusing the roles. This is why C1 is a dispatcher and a plan, not a fused mainloop.

## v2.14.0 — Table 3 (measured tile mix) and the tile-64 boundary correction (2026-10-28)
- Table 3 added to paper 5.1: measured share of query tiles per class on the mixed long-context TraceLab replay
  (87,932 steps, median 119K prefix). Qwen/Llama: 16=69.3%, 32=13.5%, 48=0.1%, 64=1.2%, 128=16.0%; GLM swaps 32 and
  48 because only the grouped-query ratio changes. 16-row tiles are 69% of tiles but the 16% of 128-row chunk tiles
  carry essentially all the arithmetic at a 119K prefix -- the central tension, and why row padding is worth per
  cent not a factor.
- CORRECTION (user): Table 2 claimed the 64 class as "FA2, or FA3 above 64" as if settled. 64 rows is wgmma's native
  tile (one consumer warpgroup) while FA2 builds it from four 16-row blocks, and the measured basis (A3: FA2 wins
  verify-width 28-48; A5.1: FA3 q64 wins short-KV few-row) never tested 64 itself on both executors. Table 2 now
  marks it the one unmeasured boundary. This also simplifies item 18: if 64-row work belongs on FA3 q64, the spare
  warpgroup is the natural home for it rather than a special mode.
- Bench client fixed: our overlay (drivers/serving/bench_overlay_20261026/serving.py, the only one with
  --dataset-name tracelab-claude and --cache-report) installed as sglang/benchmark/serving.py in ourpy/python and
  python_slo on all four pods; mn_bench.sh BENCH_PY now defaults to our tree. Upstream's benchmark/ is unusable
  here (no tracelab dataset, and it breaks on sglang.srt.utils.network against our srt).

## v2.13.0 — Table 2: tile class -> model x mode mapping, read from the tile's side (2026-10-28)
- Replaced the per-work-item list with a table keyed BY TILE CLASS (16 / 32 / 48 / 64 / 128), naming for each the
  executor, what lands there, and which model and which of the four modes produce it -- so a reader can follow the
  fusion without reconstructing it. 16: all decode rows and all draft steps (7 / 8 / 12 packed rows). 32: Qwen and
  Llama verify (28 / 32). 48: GLM verify (48, exact; the class exists for GLM). 64: chunk remainder tiles and the
  warpgroup-freeing case of item 18. 128: the body of every prefill chunk, all models.
- Added the two consequences explicitly: the three models' decode rows share one tile while verify rows span 28-48
  (hence the model ordering in 6.2), and the classes are unevenly used -- the 128 class carries nearly all the
  arithmetic, which is why removing row padding is worth per cent rather than a factor.

## v2.12.0 — Tile-mapping table in the paper; q64 regime comes from our chunk rule (2026-10-28)
- Added a work-item -> tile table to paper section 5.1: decode rows are 7 / 8 / 12 packed rows (Qwen / Llama-70B /
  GLM) and all need the 16-row tile, while verify rows are 28 / 32 / 48 and need 32 / 32 / 48-64; the prefill chunk
  follows the step's chunk size. This is the padding C1 removes and it explains the model ordering in 6.2.
- Measured the trace to test the user's point that agent tool-call prefills coexist with rows at long context:
  appends are median 875 uncached tokens (p10/p90 199 / 7 119) over a median 119K prefix, and only 0.4% of rounds
  append <=64 tokens (0.5% at >=100K prefix). So the workload alone almost always gives several q128 tiles.
- Correction recorded: my "short-to-medium context only" restriction on item 18 was wrong in its reasoning. The q64
  regime is created by the SLO-budgeted CHUNK RULE (192-16384 tokens depending on slack), not by append lengths --
  so the tile mix is a function of C3's chunk sizing, coupling the two contributions. Item 18's plan-only test must
  use the chunk sizes the rule picks, not a fixed 2048.

## v2.11.0 — Item 18 extended: tile-16 offload from FA2 to FA3 q64 CTAs (2026-10-28)
- User's extension: the warpgroup freed by running the FA3 role at q64 is extra tile-16 CAPACITY sized by the number
  of q64 chunk tiles, so the plan can offload tile-16 rows from the FA2 executor into FA3 CTAs
  (rows_to_FA3 = min(spare_slots, pending tile-16 rows)). Goal: maximise tensor-core utilisation, minimise tile
  waste. The two executors' capacity becomes fungible instead of fixed at launch.
- This is the first mechanism whose novelty rests on the UNIFIED PLAN rather than tile sizing: it needs one plan
  that sees both roles and co-assigns across them. PDL cannot (capacity fixed per launch), POD cannot (FA2-only),
  DeFT does not (one call, one role).
- Honest bound from A5.3: co-running the roles measured 18% where they balance, but <=2% of serving GPU time on
  TraceLab is in that regime. Claim scoped to short-to-medium context with many rows.
- Two cheap tests recorded, in order: (1) compile-only ptxas registers/spills at q64 (the first co-compiled attempt
  spilled 1,436 B); (2) plan-only count from the trace of how many tile-16 rows could move, bounding the gain
  before any kernel work.

## v2.10.0 — New kernel item 18: FA3 q64 co-scheduled with an mma.sync 16-row warpgroup (2026-10-28)
- User's proposal, recorded as kernel item 18 and the most promising untried overlap. The Hopper restriction is
  per-warpgroup wgmma shape, not wgmma-vs-mma.sync across warpgroups: FA3 at q64 uses one consumer warpgroup
  instead of two, freeing a slot that can issue mma.sync for 16-row rows in the same CTA. q128 rows keep the
  separate-launch PDL path.
- Nothing is lost by taking q64 for the FA3 role: A5.1 measured q64 winning only on short-KV few-row items, which
  route to FA2 anyway, and q128 is never slower on FA3's prefill/append chunks.
- It attacks the measured blocker directly: the co-compiled persistent kernel spilled every class (1,436 B) because
  FA3's warpgroup protocol balances at 168 registers and cannot hold with the FA2 mainloop co-compiled. Register
  arithmetic: q128 ~64K (whole SM budget), q64 ~43K, plus a register-light 16-row mma.sync warpgroup (~16
  accumulator regs/thread) at ~10K => ~53K.
- Bounded by role imbalance like rows 3 and 7: expect gains at short-to-medium context with many rows, ~nothing at
  long context. First test is a COMPILE (ptxas -v registers and spills), no GPU time.

## v2.09.0 — Kernel items consolidated into one results table (2026-10-28)
- Added "Kernel items: one table, every measured result" at the top of section A: 17 mechanisms, each with its
  measured effect, verdict and evidence pointer, plus four end-to-end checks of the shipped dispatcher and the four
  facts that establish the ceiling is real rather than a missed optimisation.
- The top-of-doc deliverables section now points at that table. The evidence had been split across three places,
  which is how cross-role backfill came to be described as "untested" in one section while A5.6 recorded it as
  shipped and measured at 2.2-6.1%.

## v2.08.0 — Two more self-inflicted faults in the smoke runs (2026-10-28)
- Both smoke attempts reused RUN_TAG=_smoke1, so they shared one directory and interleaved logs, making the failure
  read as a contradiction (ls showed worker_r0_g0.log while grep reported it missing; bindfail5 logs came from port
  contention between the overlapping attempts). The smoke driver now takes SMOKE_N for a distinct tag per attempt.
- The GPU-clearing pkill raced a live launch: supervise_r0_g0.log records "mn_node.sh: line 308: 43273 Killed".
  Clearing GPUs must complete and be confirmed before a launch, never interleaved with one.
- Verified the override block is NOT at fault: running the real mn_node.sh block on pod 0 emits
  {"rope_scaling":{...},"rope_theta":1000000} and python3 is present, so the BAD_OVERRIDE_JSON guard did not stop
  the launch.

## v2.07.0 — CORRECTION: cross-role backfill with tma_store_wait ships in A6 and is measured (2026-10-28)
- An earlier entry this session called cross-role backfill "untested" and bounded it analytically. That was wrong,
  written without re-reading the A5.6 record. The shipped A6 kernel DOES do cross-role backfill: the FA3 launch
  triggers its PDL dependent at kernel entry and drains its TMA stores with tma_store_wait before exit, so FA2 CTAs
  fill each SM an FA3 CTA frees. Measured worth 2.2-6.1% of the step (PDL-off ablation, H200, 96 captures).
- The in-kernel variant (an FA3 CTA taking FA2 work itself) was also built and measured: correct and worth 1.4-5.5x
  over no backfill, but that co-compiled path ran 2-6x slower, so co-launch shipped.
- With multicast measured at 0.76x (A5.5), the goal's "test multicast and cross-role backfill (needs
  tma_store_wait)" is DONE with numbers. The ceiling analysis stands as the reason no FURTHER backfill work pays.

## v2.06.0 — rope_theta was nested in the wrong place; smoke arm caught it (2026-10-28)
- Qwen2.5 carries rope_theta as a TOP-LEVEL config field, a sibling of rope_scaling. My override nested it inside
  rope_scaling, which corrupted the context-length derivation (worker logged "derived context_length (2048)") and
  the server then died with CUDA OOM at startup. Correct form: {"rope_scaling":{...},"rope_theta":1000000};
  verified by parsing and checking top-level keys, redeployed to all four pods, smoke arm relaunched.
- This is the fourth deployment fault in a row (invalid override JSON, missing sglang/benchmark, client/server tree
  mismatch, rope_theta misplaced). Each was found only by running one short arm, and the smoke gate found this one
  in five minutes instead of a 40-minute sweep.

## v2.05.0 — Smoke-gated sweep launch (2026-10-28)
- /tmp/smoke_then_sweep.sh waits for the smoke arm, verifies it wrote at least one result JSON, and only then
  launches the superset sweep as r4; otherwise it reports SMOKE_FAIL and points at the pod-side bench_agent.log.
  This closes the failure mode that cost sweeps r1-r3: a run proceeding for 40 minutes with healthy workers and a
  live router while every bench client was dead, invisible in the driver log.

## v2.04.0 — Draft-share measurement unresolved after three attempts; de-prioritised (2026-10-28)
- The CUDA-event wrapper installs ("DRAFT_SHARE instrumentation active") and verify is defined on EAGLEWorkerV2 in
  the tree being run, but zero reports appear even at a threshold of 20 verify calls, so the instrumented method is
  not on the executed path for that arm; the native arm also crashed behind the driver's grep filter.
- Left OPEN and NOT claimed. Diagnostics recorded for whoever resumes: log type(self.spec_worker) at startup instead
  of assuming EAGLEWorkerV2 (frozen_kv_mtp_worker_v2.py and multi_layer_eagle_worker_v2.py define their own
  draft_forward), and widen the driver's output filter so crashes are visible.
- De-prioritised because the number is only a CEILING for an unbuilt idea, and only 0.2% of decode wall time lies
  outside the traced forward, so a large payoff is already unlikely. H200 freed for critical-path work.

## v2.03.0 — Client/server tree separation; smoke arm before sweeps (2026-10-28)
- Third deployment blocker: with upstream's benchmark/ copied into our tree, the client died on
  ModuleNotFoundError: No module named 'sglang.srt.utils.network' -- upstream's benchmark client imports modules our
  serving tree predates. Grafting the package was the wrong fix.
- Proper fix: mn_bench.sh now runs the CLIENT against a self-consistent upstream checkout via
  PYTHONPATH=${BENCH_PY:-$W/up/python}:$W/fi, while servers still launch from SGL_PY (our tree or python_slo). No
  arm's serving code changes. Deployed and verified on all four pods.
- A short smoke arm (one baseline, alpha 12, 180 s horizon) now runs before any sweep. Three separate deployment
  faults (invalid rope override JSON, missing sglang/benchmark, client/server tree mismatch) each burned a full
  sweep attempt and would all have been caught by it.
- pkill self-match hit a third time: a pattern naming the driver matched the shell issuing the command and killed it
  mid-script, so a heredoc never ran. Kill by PID only.

## v2.02.0 — Second superset blocker: our tree lacks sglang/benchmark/ (2026-10-28)
- Run r2 produced nothing for 40 minutes with healthy workers and a live router: every bench client died with
  ModuleNotFoundError: No module named 'sglang.benchmark.serving'. Our pushed tree carries only the deprecation
  shim (sglang/bench_serving.py); the repo has no python/sglang/benchmark/serving.py at all, so the stage-2 push
  could never have included it. Upstream's up/python is the only copy on the pods.
- Fixed by copying up/python/sglang/benchmark into ourpy/python/sglang and python_slo/sglang on all four pods
  (verified, including --cache-report). Sweep relaunched as r3.
- Lesson recorded: after re-provisioning, run ONE short arm end-to-end before queueing a sweep -- the driver log
  showed nothing because the failure was inside bench_agent.log on the pod.

## v2.01.0 — Deferred KV commit closed; draft-share threshold lowered (2026-10-28)
- Deferred KV commit for rejected draft tokens is CLOSED with no GPU time: sglang already compacts only the
  accepted slots after verify (fill_accepted_out_cache_loc, eagle_info_v2.py:405), and the draft tokens' KV writes
  are required by the computation rather than wasted -- the chain's later positions attend to earlier draft tokens
  inside the same verify forward, and acceptance is only known after that forward produces logits.
- Draft-share measurement: the wrapper installed correctly but reported only every 200 verify calls, which the short
  arm never reached; threshold lowered to 20 and relaunched. Two earlier failures recorded: make_root re-copies the
  tree from $SRC each run (so patching a previous root does nothing), and the report threshold.
- Paper: 5.2.3 (SLO-budgeted chunk) and 5.1 (three-model kernel table plus the four bounding measurements) now
  carry measured results instead of placeholders.

## v2.00.0 — Fixed the invalid override JSON that killed every worker (2026-10-28)
- My ROPE_THETA edit was wrong: bash strips the quotes inside ${VAR:+...}, so the override became
  `,rope_theta:1000000` (unquoted key) and every worker died with JSONDecodeError at column 86. The first superset
  run stalled 12 minutes and produced no arm.
- mn_node.sh now builds the override in steps and validates it with json.loads before use, failing loudly with
  BAD_OVERRIDE_JSON. Verified valid with and without ROPE_THETA; redeployed to all four pods with the guard string
  checked on each, and the sweep relaunched as r2.
- Process lessons recorded: pkill -f matched and killed the shell running the command containing the pattern, so a
  later tar never ran and a STALE tarball was pushed while push reported success; pgrep -cf counts itself, making
  clean pods look occupied; a `b"""` typo for `b="""` skipped a doc edit while the commit still succeeded.

## v1.99.0 — Cross-role backfill bounded and closed on evidence (2026-10-28)
- The goal's untested item is now bounded from the 96-capture H200 results instead of being built blind:
  PDL recovers only 1.4% (32K four-mode) to 6.2% (two-mode) over the non-overlapped serial two-call arm, and two
  separate streams recover nothing (1.763x vs 1.759x). Two independent overlap mechanisms find nothing left.
- Structural reason: backfill can only hide the SMALLER role, and the roles are extremely unbalanced -- the chunk's
  attention approaches 100% of the step as context grows (0.73 s of FLOPs for a 2048-token chunk over 128K on H20),
  which is also why the A6-vs-stock serving gain falls 1.147x -> 1.006x. The serial/2 (<=49%) bound only holds for
  equal-cost roles, which never occurs here.
- Closed: cross-role backfill cannot exceed low single digits on this workload and PDL already collects most of it;
  tma_store_wait would add persistent-kernel structure measured at 0.987-1.006x. Interesting only for balanced
  roles (short context, many rows).
- Draft-share CUDA-event instrumentation written (apply_draft_share_events_20261028.py, module-scope wrapper, off
  unless SGLANG_DRAFT_SHARE=1), validated on a throwaway copy, and running on the H200 via
  a6_draftshare_h200_20261028.sh. The first attempt measured nothing: that driver's make_root re-copies the tree
  from $SRC every run, so the patch must be applied to $SRC/python/sglang/srt, not to a previous run's root. Check
  for "DRAFT_SHARE instrumentation active" in server.log before trusting such a run.

## v1.98.0 — The superset claim, and the valid single-node baselines (2026-10-28)
- User's two deliverables recorded at the top of the task doc: (1) kernel gains from fusing tile classes in one
  forward (measured 2-9% over the fair baseline across three models); (2) scheduling as a SUPERSET of every
  baseline via simple SLO control + a latency model, read as domination at every load on the curve.
- Consequence: FCFS beating us at alpha 3 was a VIOLATION of the superset property (caused by chunk parking, now
  dropped, and the fixed chunk, now adaptive), not a finding about FCFS. The chunk-2048 + FCFS arm is not a valid
  baseline at all -- it was our chunk control with sglang's ordering.
- Valid single-node baselines, each with sglang's own chunk policy: default prefill-prioritized (MIXED_CHUNK=0),
  mixed chunk (MIXED_CHUNK=1), PD multiplexing (PDMUX=1, no overlap scheduling, ~10% lower throughput expected).
- mn_superset_16g_20261028.sh launched on the new cluster: those three baselines against ours (A6 + four-mode +
  EDF coupled to the SLO chunk at 16384), alpha 3 and alpha 6, acceptance logged per arm.
- A6 kernel libraries deployed to all four pods (5 .so + 2 shims) from the local bundle.

## v1.97.0 — Draft-share ceiling inconclusive from traces; no host headroom at decode (2026-10-28)
- Tried to bound draft/verify pipelining at zero GPU cost: on 18,075 consecutive decode-step pairs the wall gap
  between target forwards is p50 30.08 ms against a traced forward GPU time of p50 30.98 ms, so `gpu=` already
  covers the draft forwards and they cannot be separated from the trace. The ceiling needs in-step CUDA events.
- Useful negative result: only 0.2% (p90 1.7%) of decode wall time is outside the traced forward, so the overlap
  scheduler hides all host work. No launch-overhead/host-side time exists to recover at decode -- consistent with
  vLLM PR 61869's launch-count reduction measuring below our noise floor.

## v1.96.0 — New cluster fully re-provisioned (2026-10-28)
- Stage 1 on all 4 pods: venv/torch/sgl_kernel/flashinfer, Qwen2.5-7B + EAGLE3, ShareGPT, TraceLab.
- Stage 2 pushed from the repo: our serving tree (ourpy/python -> python), port07 kernel sources, drivers/serving,
  step model + scorers under perf/, and python_slo = our tree + the chunk patch. Verified on every pod, with the
  patch carrying bank="inf" and :ttft. up/python remains untouched upstream for the BASELINE arm.
- A6 libraries restored from the local bundle a6_bundle_20261025.tgz (H20 sm90); not rebuildable quickly on a fresh
  pod. Recorded fallback: if they cannot be restored, run the chunk/EDF arms with the kernel held at KERNEL=lpt on
  both sides -- cleaner for C3, since C1 is 1.4% at cluster level (inside the noise floor).

## v1.95.0 — Cross-role KV reuse gated out on the session replay (zero GPU cost) (2026-10-28)
- The gating measurement is answered structurally: (1) tracelab.py draws each session's tokens independently from
  the global RNG, so concurrent sessions share NO prefix -- our 92.7% hit is temporal (round N+1 reusing round N),
  not spatial; (2) a session's rounds are sequential, so the session being prefilled has no decode rows of its own
  in that step. Cross-role page overlap is therefore ~0 by construction on the TraceLab replay.
- Scope: the mechanism is viable only on fan-out / branching workloads (siblings of one parent concurrent, one
  prefilling over the parent prefix while another decodes) -- drivers/serving/fanout_agents.py, the ThunderAgent
  setting of goal point 2. Must not be claimed on the session replay.
- Also explains the weak version's behaviour: KV-position ordering gave 1.08x on a synthetic 48K shared prefix and
  -29% on the trace, because the trace has no shared prefix to order around.
- Corrected arm driver committed (mn_chunk_vs_sglang_20261028.sh): baseline = upstream tree with sglang's own
  schedule and chunk policy (CHUNK and SLO_CHUNK unset); ours = EDF coupled to the SLO chunk, launched at 16384 so
  the rule decides. Both alpha 3 and alpha 6, acceptance logged per arm.

## v1.94.0 — SLO-budgeted chunk rule defined and validated offline, no DP (2026-10-28)
- tasks/perf_model/chunk_predictor.py: budget = (1-h) * min_r [s_r*(o_r+a) - E_r]; chunk = largest c with
  T_step(c) <= budget, by bisection over the fitted step model. Three lines plus 16 model evaluations (~2 us); no
  dynamic program, nothing borrowed.
- Validated on 10,185 merged steps of the alpha-3 replay: step-model MAPE 4.7% (p90 7.5%) on the chunk actually
  used, so the fused step is predictable enough to size from.
- The fixed 2048 chunk's predicted step is 5.1x the per-step allowance (p90 8.4x) -- the TPOT damage removed.
- The rule is adaptive both ways: rows bank 36.1 ms/token (13.9 ms observed against a 50 ms target), so chunk p50
  goes 192 -> 1600 -> 3072 -> 7168 -> 16384 tokens as a row emits 0, 10, 20, 50, 200 tokens. It admits far MORE
  than 2048 in steady state and clamps only when rows are fresh.
- Confirms why bank=4 starved (slochunkh: TPOT p90 64 ms but 7.9% attainment, 107 tok/s): it turned a mean-TPOT
  rule into a per-step ITL bound. Unlimited banking is the correct reading of mean TPOT.

## v1.93.0 — Llama-70B ladder completes the three-model kernel picture; H20 job ended (2026-10-28)
- Llama-3.3-70B, 38 captures, FP32-checked: A6 is 1.028-1.049x over the naive role split (2.8-4.9%), while the role
  split alone is 6.7-7.7x of sglang's single FA3 call.
- Three-model summary of C1 against the fair baseline: Qwen 4.0-8.7%, Llama-70B 2.8-5.8%, GLM 2.0-2.8%. Every large
  figure (1.8x-7.7x) belongs to the role split, which is not our contribution. The worse sglang's call serves a
  model's verify rows, the larger the role-split gain and the smaller our marginal gain.
- DeFT (arXiv 2404.00242) added to the reference list: KV-guided grouping saves 73-99% KV IO but only for a tree
  within ONE call; our cross-role/cross-request grouping is the differentiator and exists only because of C2.
- NEW H20 CLUSTER: 8d89d1880003f89d, named vibecuda, forked from the finished job (needs overrides.attachments=[]).
  Same 4x4 H20 layout; needs re-provisioning before arms can run.
- Diagnosis note: a finished job makes merlin-cli report "webshell relay redirected to SSO", which looks like an
  auth failure but is not -- check job status first; the ssh proxy reports it honestly.
- H20 job 2f608c1b1a67cf09 finished at 06:29 CST (status done). The last gap-source arm and the corrected stock
  baseline died with it; /dev/shm on those pods is gone. All measured results are already in the docs. Multi-node
  work needs a fresh H20 allocation (vibecuda, <= 16 GPUs). The H200 trial is unaffected.

## v1.91.0 — Draft/verify pipelining queued as the next kernel item; why FCFS is TPOT-neutral (2026-10-28)
- User's kernel idea recorded: overlap the draft head of the current step with the verify of another group. The
  draft is 3 sequential one-layer forwards per step, latency-bound rather than FLOP-bound, so it is the only
  remaining item that deletes serial time instead of shaving row-side work. Requires a two-group staggered batch,
  since draft(N+1) depends on verify(N) within a group. First measure the draft share of decode wall time; that is
  the ceiling. Acceptance to protect: 3.35-3.55 at 128K.
- CORRECTION (user): the FCFS arm was designed wrong -- both arms were given OUR fixed 2048 chunk, so ordering was
  the only difference and the TPOT result was guaranteed. Three rules recorded: never give the baseline our chunk
  control (baseline = sglang prefill-prioritized + sglang chunk policy); our node schedule is COUPLED with the
  chunk rule and must not be ablated against it; and no fixed chunk in our arm either (at serving start, many
  prefills with few decodes make a 2048 cap throttle prefill and starve the decode batch).
- SLO-budgeted chunk without a DP: learn it from state we already log (step features + per-row remaining TPOT slack
  + SLO class), either by direct regression to the max safe chunk or by binary-searching the validated step MLP;
  labels come from existing traces, so no new GPU time.
- Llama-70B ladder finished (LADDER_llama70b_DONE); A6 is 6.7x sglang's call on mix32_two, role split dominant.
- BLOCKER: pods 1-3 need `merlin-cli login --control-plane cn-seed --force` (WebShell SSO expired); pod 0 still
  reachable over ssh. Cluster runs cannot launch until that is refreshed.

## v1.90.0 — GLM ladder: the tile-class hypothesis is refuted (2026-10-28)
- GLM-4.5-Air, 43 captures, FP32-checked: A6 is 1.020-1.023x over the naive role split (serial), i.e. 2.0-2.8% --
  LESS than Qwen's 4.0-8.7%, on the model the 64-row tile class was designed for.
- Cause: GLM gains 5.3-6.0x from the role split alone because FA3 tiles its 48-row verify rows badly, but stock
  flashinfer FA2 already tiles them well. The earlier 1.38x for the 64-row class was measured against FA3, the
  wrong baseline.
- Across three models C1 is 2-9% over a fair baseline; every large figure belongs to the role split, not to us.
  Only cross-role backfill (tma_store_wait) remains untested, and it is the only idea that targets beta.

## v1.89.0 — SLO chunk works but was mis-tuned; cluster push gotcha (2026-10-28)
- alpha 6 chunk arms rescored on TPOT (user: TPOT matters more than TTFT): the SLO-budgeted chunk cuts chat TPOT p90
  from 214 ms (fixed 2048) to 64 ms and agent p90 from 165 to 79 ms, but attainment collapses to 7.9% and goodput to
  107 tok/s -- it was run with the old bank=4 default, which turns the mean-TPOT rule into a per-step ITL bound and
  starves admission. The mechanism is sound; the operating point is wrong. The patch now defaults to unlimited
  banking, which is the correct reading of mean TPOT. Sweep at alpha 6 on the cache router is the next step.
- Earlier "chunk size contributes nothing" was an alpha-3 artifact: at low load there is nothing to protect.
- At alpha 6 deadline ordering is TPOT-NEUTRAL (chat p90 111 vs 109 ms, agent 98 vs 102); its gains are goodput
  (+7.5%) and TTFT tails. Ordering reorders who waits; it does not protect the rate of admitted requests.
- Cluster gotcha: Cluster() without nodes= sends every push/run to one default instance (not even worker-0..3), so
  two "corrected" stock re-runs silently used the old mn_node.sh. ROPE_THETA also had to be added to mn_run.py's
  explicit env string. ropefix_r3 is the first run that actually carries the fix.

## v1.88.0 — Overload (alpha 6) point complete (2026-10-28)
- Deadline ordering separates from FCFS only under overload: agent 78.7% / 2102 tok/s vs 76.2% / 1956 (+7.5%
  goodput, +2.5 points). At alpha 3 the two are identical and at 128K single-node EDF costs 8.5% of TPOT, so
  C3-worker is an overload mechanism, not a general win.
- Conditional disaggregation collapses faster than colocation under overload: agent 53.8% / 725 tok/s against
  76-79% / 1956-2102, and 9264 rounds never issued. That arm runs our router on our tree, so it is RoPE-clean.
- The stock colocated alpha-6 row (66.7%, 1292) carries the wrong RoPE base and is not comparable; re-run queued.

## v1.87.0 — Cluster decomposition complete: the router is the one contribution that measures (2026-10-28)
- All four arms on our tree, acceptance 1.983-1.997 (internally valid). Our cache router with a load tie-break
  against sglang's cache_aware: chat attainment 90.6% vs 83.3% (+7.3 points) and chat TTFT p90 0.67 s vs 3.24 s
  (-79%), at identical agent goodput (2471 vs 2482) and acceptance.
- C1 1.4%, C2 0%, chunk size 0% at cluster scale -- all within the noise floor.
- The winning router is the PLAIN cache router with a load tie-break, not the SLO-checked cache_slo router, which
  loses badly on session-affine traces (65.1-68.5% chat) by moving requests off their cached prefix.

## v1.86.0 — Single-node attribution complete (2026-10-28)
- Cumulative on one tree, two repeats: stock 97.3 ms TPOT at 128K -> 48.0 with the four-mode step (-51%) -> 41.3
  with the A6 kernel (-14%) -> 44.8 with EDF and preemption (+8.5% WORSE). At 32K: 22.9 -> 14.2 (-38%) -> 14.2 (0%)
  -> 14.3 (~0%).
- C2 (four-mode) carries the single-node result, as its 8192-token gate predicts. C1 is 0% at 32K and at most -14%
  at 128K, and that is confounded by a +11% draft-acceptance difference (3.27 vs 2.94 tokens/verify) caused by the
  kernel's numerics; at 32K acceptance matches and C1's TPOT effect is zero.
- C3-worker costs 8.5% of TPOT at 128K, matching the cluster's preemption finding; it helps only under overload.

## v1.85.0 — INVALID BASELINE: upstream arms ran Qwen with rope base 1e4, not 1e6 (2026-10-28)
- mn_node.sh's YaRN override replaces the rope_scaling dict; upstream sglang drops rope_theta with it and falls back
  to 10000, while Qwen2.5-7B needs 1e6. Our tree preserves it, so every up_clean arm served a mis-configured model.
- Measured: mean draft acceptance 1.420 (stock colocated, upstream tree) vs 1.991 (ours). 1.40x on acceptance alone
  covers the +25% throughput / -45% TPOT we reported over stock colocated sglang. That claim is WITHDRAWN.
- This explains the puzzle of the last several hours: C1, C2, chunk and router all measured zero inside our tree
  while the margin over upstream looked large. The margin was the baseline's broken RoPE, not a contribution.
- NOT affected: single-node H200 A/Bs (one tree for both arms), the kernel ladder, all ours-vs-ours ablations.
- Fix without patching upstream: mn_node.sh now appends "rope_theta":$ROPE_THETA to the override when ROPE_THETA is
  set. mn_stockfix_16g_20261028.sh re-runs the stock baseline with ROPE_THETA=1000000 plus our arm as a control.
- All past up_clean arms are suspect, including the PD-disaggregation baselines; check acceptance before quoting.

## v1.84.0 — C2's contradiction resolved: it is gated on uncached prefill length (2026-10-28)
- The four-mode step gives -38% TPOT on one node and zero on the cluster because of its gate
  (SGLANG_MIXED_SPEC_MIN_CTX=8192). At 92.7% prefix hit a cluster agent round carries only ~1085 uncached tokens,
  7.5x below the gate, so it rarely fires; the single-node stages (32K/128K uncached) always clear it.
- C2 therefore pays when prefills are long and cold and is inert once cache affinity shrinks them. This is a real
  tension between our contributions: cache-first routing removes the condition that makes the four-mode step pay.
- Supersedes two earlier wrong readings: the Amdahl explanation, and "C2 duplicates upstream's MIXED step" (_mtps
  has no mixing at all; it is not upstream's spec-aware mixed step).
- dchunk arm: upstream's default chunk matches CHUNK=2048 (2499 vs 2471 tok/s), so chunk size contributes nothing.
- alpha 6 overload: EDF beats FCFS by +7.5% agent goodput (2102 vs 1956) and +2.5 points, the first regime where
  the worker scheduler separates; at alpha 3 they are identical.
- Single-node kernel alone (both arms four-mode, no worker scheduler): TPOT unchanged, throughput +6.9%.

## v1.83.0 — Single-node 128K: -54% TPOT and +11% throughput vs stock sglang (2026-10-28)
- mix128 stage: our full stack 44.8 ms TPOT / 34.4 ITL / 95.7 worst-decile / 54.4 tok/s against stock MTP + stock
  FA3 at 97.3 / 61.0 / 231.4 / 48.8. Clears both bars at once (>=30% latency, >=10% throughput) and is larger than
  the -38% at 32K, so the margin grows with context. Fused path verified (2478 composition events).
- No-speculation rows isolate the kernel alone: -6.7% TPOT at 32K, -2.8% at 128K -- small and shrinking with
  context, consistent with the capture ladder and the cluster step-time analysis.
- Still bundles C1+C2+C3worker; the no-ours pass separating them is running.

## v1.82.0 — Single-node A/B on H200: -38% TPOT, -57% tail vs stock sglang (2026-10-28)
- mix32, Qwen2.5-7B, two repeats, A6 composition verified: four-mode + A6 gives TPOT 14.3 ms / ITL 15.0 / worst
  decile 32.4 against stock MTP + stock FA3 at 22.9 / 20.6 / 75.9, equal throughput. Matches the cluster direction.
- CORRECTION: the -38% bundles C1 + C2 + C3-worker, not two contributions -- `--ours` also turns on EDF
  scheduling, uncached deadlines, chunk preemption and deadline-sized chunks. It is an end-to-end stage-2 headline
  and attributes nothing. a6_noours_singlenode_h200_20261028.sh repeats it without --ours to isolate C1 + C2.
- Does NOT separate C1 from C2 either: the control removes both, and the harness's "two" mode disables the draft model
  entirely (no speculation), so those rows are not a stock-MTP arm. An earlier C1xC2 interaction reading is
  withdrawn. The no-speculation rows isolate the kernel at -6.7% TPOT, consistent with 4-9% on captures.

## v1.81.0 — C2 (four-mode step) contributes zero at the cluster workload (2026-10-28)
- At the best configuration (cache router + FCFS), removing the four-mode step changes nothing: 90.6/93.8/2492 tok/s
  against 90.6/94.2/2471 with it, identical within the noise floor and nominally faster without. The hypothesis that
  C2 carried the -45% TPOT over stock colocated sglang is falsified.
- CORRECTION to that entry's first explanation: not an Amdahl bound. Merged steps are 11.6% of steps but 67.2% of
  GPU time, so C2 applies to two thirds of GPU time and still changes nothing. The real cause is that upstream's
  spec-aware MIXED step already merges decode tails into prefill steps -- C2 duplicates a baseline capability.
- C2 cannot be claimed as a gain here; it needs a merged-step-dominated workload or demotion to an enabling
  mechanism. Chunk size and router remain the candidates for the +25% / -45%; arms dchunk and sglrt running.

## v1.80.0 — Kernel ladder on 4x H200 vs the mandated naive FA3 + stock FA2 baseline (2026-10-28)
- 96 trace-sampled captures, FP32-checked, 20 graph replays. A6 is 1.86-2.92x sglang's single varlen FA3 call, but
  the naive role split (stock FA3 + stock flashinfer FA2) is itself 1.76-2.80x. **A6's own gain over that fair
  baseline is 4.0-8.7%** -- the honest C1 claim; the ~2x figure is not ours to quote.
- Two-stream naive overlap buys nothing over serial (1.763 vs 1.759x), bounding launch-level overlap and confirming
  the large-tile/small-tile overlap is gone once the chunk runs on FA3.
- A6 beats the earlier two-plan v3 by 1.34-1.92x; v3 retired as an arm.
- Round-robin routing collapses the cluster (chat 11.4%, agent 10.3%, 49 tok/s), so cache affinity is worth ~50x --
  but stock sglang's own cache_aware router already achieves it.

## v1.79.0 — The disaggregation gap is a cache gap (not ours); our gain holds at equal hit rate (2026-10-28)
- Stock upstream sglang, colocated behind its own cache_aware router, reaches 91.8% prefix hit vs our 92.7%. So the
  8x prefill-work advantage over conditional disaggregation (11.2% hit) belongs to the LAYOUT and sglang's shipped
  routing, not to us. Reported as a finding; the cache hit is no longer claimed as a contribution.
- Against that strongest baseline at equal hit rate: +25% agent SLO goodput (2471 vs 1976 tok/s), -45% TPOT
  (11.8 vs 21.6 ms), -37% mean ITL, chat misses halved, 11,424 rounds served vs 9,656. Clears the >=10% throughput
  and >=30% latency bars against an unhandicapped baseline.
- -45% TPOT >> C1's 4% of step time, so C2 (four-mode step) is the likely carrier; its earlier within-noise verdict
  came from an SLO-router arm where only ~11% of steps merged. Clean decomposition at the best config queued.
- Found: the client's cached_tokens is 0 in every arm, so the agent >200K TTFT allowance never fires (pessimistic
  for long rounds, equally in all arms). Worker-log hit rates unaffected.

## v1.78.0 — Where C1's gain actually is; POD's FA2-only limitation (2026-10-27)
- Step-for-step A6 vs stock FA3 on the same replay (A6 validity confirmed: 70,701 a6_unified calls vs 0): speedup
  FALLS with chunk prefix, 1.147x below 8K to 1.006x above 128K. The step is attention-dominated (a 2048-token chunk
  over 128K is 1.08e14 FLOP = 0.73 s at H20's 148 TFLOP/s, matching the 727 ms measured), but both arms run that
  chunk on FA3 identically. C1 optimises the rows only, so it is nearly inert where the chunk dominates.
- Recorded: PDL is not the persistent-FA2 large-tile/small-tile overlap; that idea left when the chunk moved to FA3.
  MELD's JIT reduction pays on many-partial tasks; our steps have <= 0.9% merge. Draft/verify cannot be fused
  (sequential dependency).
- Related work: POD-Attention's tile overlap is FA2-only and cannot carry to FA3 / FA4 on Hopper / Blackwell.

## v1.77.0 — Demotion never ran; noise floor measured; best-config arm 1 (2026-10-27)
- CORRECTION: the demote code is gated behind `SGLANG_EDF_PREEMPT_CHUNK=1`, so both decomposition arms (PREEMPT=0)
  ran it never -- zero EDF_LATE_DEMOTE lines. The two arms were identical, so their gap (chat 68.8 vs 73.8, goodput
  1912 vs 1986) is the **run-to-run noise floor: ~5 points chat, ~4% goodput**. "Ordering + demotion beats FCFS" is
  withdrawn; parking's ~25% cost survives (outside the floor). Deadline rescoring onto uncached tokens is behind the
  same gate, so both arms ordered by full-prompt-length priority.
- Best config arm 1 (cache routing + FCFS): 90.6% / 94.2% / 2471 tok/s, within noise of cache + EDF (91.4 / 94.3 /
  2514). The router, not the worker policy, is the whole cluster-level effect.
- `apply_edf_model_demote_20261027.py` drafts the user's design (never preempt; demote on model-predicted TTFT
  miss) but must be moved out of the gated method before use.

## v1.76.0 — vLLM PR 61869 reviewed; C1 leave-one-out is within noise (2026-10-27)
- PR 61869 is post-forward sampling work (fused draft-sampling kernel, post-forward CUDA graph), not an attention
  epilogue, and its "fused Markov loop" is a host loop over per-position kernels. Its gains (-37 us on a 4 ms step)
  are below our noise floor at 20-40 ms steps, and its graph is gated off any step containing prefill. Two items
  worth taking: the batched host-state-to-device copy (answers our merged-step read-back) and the self-resetting
  arrival counter (needed for A6 graph capture, where a host plan upload cannot re-zero it).
- Second without-C1 repeat: 1504 tok/s vs 1388 (r1), against 1501-1763 for the full stack. The ranges overlap, so
  C1's cluster-level leave-one-out effect is within noise and the earlier 7.5-21% range is withdrawn; C1's evidence
  stays the kernel microbenchmarks and the single-node long-context runs.

## v1.75.3 — Paper: C3's worker level is ordering + demotion; parking reported as dropped (2026-10-27)
- Sections 1, 5.2.3 and Table 4c now state the mechanism as deadline ordering plus demotion, with preemption and
  parking described as built, measured and dropped (a quarter of agent goodput for the chat TTFT tail alone).

## v1.75.2 — Decomposition arm 2: C3's worker level survives as ordering + demotion (2026-10-27)
- Ordering + demotion, no preemption: chat 73.8%, agent 84.7%, 1986 tok/s -- matches FCFS on chat and beats it on
  agent attainment and goodput, and beats the shipped ordering + parking configuration on everything.
- Demotion recovers what plain ordering lost (68.8 -> 73.8 chat) at no throughput cost. Chunk preemption/parking is
  the mechanism to drop.
- A fourth arm (cache routing + ordering + demotion) is added to the best-configuration driver.

## v1.75.1 — Decomposition arm 1: preemption/parking is the cost, not deadline ordering (2026-10-27)
- EDF ordering with preemption and parking off: 1912 tok/s, level with FCFS (1922) and well above the full stack
  (1501-1763). Chat attainment 68.8% vs 73.9% (FCFS).
- Preemption/parking is the only thing that buys the chat TTFT tail (p90 1.03-1.19 s with it, 3.03-4.10 s without),
  and it costs ~25% of goodput to do so.

## v1.75.0 — Ablation arm 4: removing C3's router wins by a wide margin (2026-10-27)
- Plain cache-affinity routing (no SLO check) reaches chat 91.4%, agent 94.3%, 2514 tok/s against 65.1-68.5% /
  72.6-78.7% / 1501-1763 for the full stack: +23..+26 points chat, +43..+67% goodput, with every latency column
  better.
- Mechanism: on session-affine agent traces each session has one cache holder, and a wide TTFT prediction moves
  requests off it for nothing. The opposite held on the fan-out workload, where prefixes are replicated.
- Consequence: the reported "ours" is not our best measured configuration. `mn_bestcfg_16g_20261027.sh` is queued:
  cache routing + FCFS workers (a combination no ablation arm covered), a repeat of the best measured arm, and
  conditional disaggregation in the same session. No claim is restated until it runs.

## v1.74.3 — Paper Table 4c carries the measured leave-one-out arms (2026-10-27)
- C1: -3.4..-6.8 points chat and -7.5..-21% agent goodput when removed (range not yet separated by the repeats).
- C2: within the repeat spread on the cluster replay (~11% merged steps there); its value is the single-node
  long-context case.
- C3 worker level: removing it *wins* on the replay (73.9% vs 65.1-68.5% chat), stated plainly with the TTFT-tail vs
  TPOT trade and the note that demotion was never active in those arms.

## v1.74.2 — Ablation arm 3: removing C3's worker level WINS on the trace replay (2026-10-27)
- FCFS admission (no deadline order, demotion, parking or reservation) beats the full stack at alpha 3: chat 73.9%
  vs 65.1-68.5%, agent 83.2% vs 72.6-78.7%, 1922 vs 1501-1763 tok/s, outside the repeat spread.
- The machinery buys a better TTFT tail (chat p90 1.03-1.19 vs 3.03 s) and pays for it in TPOT (p50 27-30 vs
  21.5 ms); under a 2 s + 50 ms SLO that trade loses. C3's worker level must not be claimed as a cluster-level gain;
  its evidence is the single-node agent burst.
- Correction: demotion was never enabled in any H20 multi-node arm (`mn_node.sh` had no `LATE_DEMOTE` until today),
  so the arm removed ordering + preemption/parking + reservation only. `LATE_DEMOTE` is now plumbed through.
- Decomposition queued ahead of the chunk A/B: ordering only, ordering + demotion, and a second without-C1 repeat.

## v1.74.1 — Ablation arm 2: C2 is within repeat noise at alpha 3 (2026-10-27)
- Without C2 (stock MTP steps instead of the merged 4-mode step): chat 69.1%, agent 77.4%, 1638 tok/s, inside the
  full-stack repeat spread (1501 and 1763). Only ~11% of steps are merged here, so the effect is diluted; C2's
  measured value is at long context on one node, and its paper claim is enabling speculation inside colocation.

## v1.74.0 — Closed-form chunk sizing from the kernel's cost law; first ablation arm (2026-10-27)
- Design recorded: the fused kernel's `max + beta x min` law is invertible, so the chunk solves in closed form, while
  the split-kernel baseline (beta = 2.19 when fit) forces a search. The balance-point chunk grows with decode load
  (162 tok at 4 rows -> 761 at 48), the opposite of a cap; acceptance widens it through the `s x a` allowance. H20
  beta = 0.796 makes the discount ~10%, so the claim belongs on the H200.
- Ablation arm 1 (without C1): chat 61.7%, agent 70.1%, 1388 tok/s vs 1501-1763 for the full stack, i.e. C1 is worth
  7.5-21% of agent goodput; the full-stack repeats differ by 17%, so a second noC1 repeat is needed.

## v1.73.2 — Metric audit and the capacity-reservation design (2026-10-27)
- Crossflow scores ITL (mean, plus a 120 ms p99 constraint); SLOs-Serve, JITServe and our scorer use mean TPOT. We
  keep mean TPOT for attainment and now also report mean and p99 ITL per tenant (`tracelab_windows.py`), since
  Crossflow is the closest workload comparison.
- Recorded the published answers to "how do you reserve prefill capacity under a mean SLO": SLOs-Serve's per-class
  rate reservation `sum(n_l / s_l)` plus soft admission control, and JITServe's per-token deadline with a predicted
  output length. These define the next chunk-control iteration.

## v1.73.1 — The ad-hoc chunk rule is retired (2026-10-27)
- v5 at alpha 3 made every metric worse than fixed 2048, including chat TPOT (45.0 vs 30.0 ms p50), with TTFT p50
  80.7 s. At that load fixed chunks already met the 50 ms target, so no throttling was warranted.
- Cause: the rule bounds every step by one TPOT allowance, while the SLO is a request's mean TPOT and only 12% of
  steps carry a prefill chunk. v5's banking cap removed the amortisation; v4 removed the bound instead.
- Correction: the SLO is mean TPOT (`sum(itls)/(n-1)`), and sglang's `edf_compute_max_chunk_size` is itself a
  per-step ITL bound, so it carries the same mismatch. The fix is the cumulative budget `s x (o + a) - E` with
  unlimited banking (v4's formula plus v5's tokens-per-step and demotion fixes, minus the cap). Stub: a row at
  30 ms/token against a 50 ms target now earns full chunks. A/B queued (v7, alpha 3 first).

## v1.73.0 — Use the existing SLO chunk solver; H20 constants for the analytic model (2026-10-27)
- Inventory: `edf_compute_max_chunk_size()` already binary-searches the largest chunk that fits the tightest row's
  slack, behind `SGLANG_EDF_LATENCY_MODEL`, which our runs left empty (hence the fixed 2048 chunk). Predictors:
  trained MLP (.pt), `linear:<json>`, or `analytic` (the closed-form overlap model built for the fused kernel).
- `perf_model/fit_overlap_params_from_steptrace.py` fits the analytic model's seven per-GPU constants from STEPTRACE
  logs. On 87,932 corrected H20 steps: held-out 4.2% MAPE on merged steps (the ones the solver prices), beta 0.796.
  Parameters in `perf_model/overlap_params_h20_qwen7b_20261027.json`.

## v1.72.6 — Chunk-budget audit against recorded steps (2026-10-27)
- `perf_model/chunk_budget_audit.py`: on 87,932 alpha-3 steps, decode-only steps (8-52 ms) never exceed the chat
  per-step allowance (100 ms), while a 2048-token chunk costs 446 ms, 4.5 allowances, on 98.1% of merged steps. The
  ~300-token chunk the rule picks is what the allowance buys, and it is what cut chat TPOT p50 from 72 to 47 ms.
- So the rule computes the right number; its prefill rate is 35% lower, which is ruinous only because alpha 6 is
  3.5-5x past prefill capacity. At overload the correct lever is request-level admission (EDF + demotion), not a
  per-step cap.

## v1.72.5 — Budgeted chunk: TTFT floor option (2026-10-27)
- `apply_slo_budget_chunk_20261026.py` adds `:ttft`: the earliest-deadline prefill that can still meet its deadline
  gets at least the chunk that finishes its remaining tokens by then. A late, or impossible, prefill gets none.
- Stub check, 20K tokens: 4 s left -> 576, 10-30 s left -> 256, 2 s left (infeasible) and late -> the TPOT budget
  (58).
- Tree `/dev/shm/w/python_slo2` on all pods. Driver `mn_slo_chunk_v6_16g_20261027.sh` is queued after the
  leave-one-out ablation.

## v1.72.4 — Budgeted chunk v5 at alpha 6: TPOT protected, prefill starved (2026-10-27)
- Chat TPOT p50 47 ms (vs 72) but chat TTFT p50 74 s (vs 0.40 s); attainment 7.9 / 15.0% vs 31.8 / 50.0%. A
  TPOT-only budget starves prefill far past capacity. Next: a TTFT floor from the earliest feasible deadline.

## v1.72.3 — SLO-budgeted chunk: tokens-per-step and demotion bugs fixed (2026-10-27)
- At alpha 6 the v4 budgeted chunk was worse than fixed 2048 on every metric (chat 27.0 vs 31.8%, agent goodput 647
  vs 890 tok/s). Its budgets reached seconds or infinity:
  - tokens per step was measured across several steps;
  - demotion dropped late rows from the bound.
- Now tokens per step come from the acceptance EWMA, late rows bound the step at their per-step allowance, and banked
  slack is capped at 4 steps. Rerun as v5.

## v1.72.2 — Step model on corrected traces (2026-10-27)
- Corrected H20 traces (alpha 3 trace replay, four workers, held-out): MLP with acceptance predicts merged prefill +
  verify steps at 3.4-4.0% and prefill at 4.6-5.5%; decode-only steps (10-20 ms) at 18-23%. Paper section 5.2.2 is
  updated, with the stock comparison pending.
- The leave-one-out arm without C1 is now traced, which provides that comparison on the same schedule.

## v1.72.1 — Chunk test on a GPU-time step model (2026-10-27)
- The corrected trace fits poorly on the gap target (lin4mode 45% MAPE, unphysical coefficients). The GPU-time fit is
  sensible (25% in-sample). The alpha-6 arm on the gap model was stopped; the budgeted-chunk arms rerun on the
  GPU-only model (driver v4, `tasks/perf_model/step_qwen_h20_a6_gpu_20261027.json`).

## v1.72.0 — Router policy cache_slo_h: placement, with SLO enforcement left to the worker (2026-10-27)
- `pd_colocation/router.py --policy cache_slo_h --budget-chunk <model>:<h>:<chat_ms>:<agent_ms>:<agent_tokens>:<chunk_max>`.
  - TTFT is priced under the worker's SLO-budgeted chunk, simulated with the same step model.
  - Nodes rank by feasibility, then cache hit, then added lateness to later-deadline queued requests, then load.
  - With no feasible node: least total lateness. No resident or own-TPOT veto; the router never holds a request.
- `build_parser()` factored out. `test_router_slo_h.py`: 7 / 7 pass.

## v1.71.6 — STEPTRACE fix, second attempt (2026-10-27)
- The merge happens inside the worker after `_steptrace_begin`. The tracer's own `mixed_spec_running` block skipped
  the running rows when there was no host `seq_lens_cpu`; their KV now comes from the requests. The ineffective v1 fix
  is reverted.
- Chunk-size driver v3 started; a background check confirms merged steps are traced with decode rows.

## v1.71.5 — Paper: step-model per-type claim withdrawn pending a corrected tracer (2026-10-27)
- Table 3's step model came from the same tracer, so the four-mode merged steps were logged in "prefill" (ours 8.7% vs
  7.5%), not in "mixed".
- Withdrawn wherever it appeared: "mixed steps are more predictable (5.9 vs 7.7%)" (stage-1 summary, section 5.2.2,
  the MuxWise reference entry).
- The overall 2.0% vs 1.9% stands. Table 3's caption states what each row holds. A re-fit is pending.

## v1.71.4 — STEPTRACE counts verify rows as decode rows; chunk test restarted (2026-10-27)
- The step tracer classified rows only by extend length, so a merged step's 4-token verify rows were logged as prefill
  rows. `apply_steptrace_verify_rows_20261027.py` fixes it; it is applied on all H20 pods, and the chunk-size driver
  is rerun (v2).
- The traced alpha-3 arm serves as a second repeat of ours: chat 68.5%, agent 78.7%, 1763 tok/s.

## v1.71.3 — Demand shift: stock PD 2 + 2 at alpha 6 rerun (2026-10-27)
- Chat 6.2%, agent 4.9%, 23 tok/s, 1096 rounds issued. Table 10d is complete.

## v1.71.2 — Paper: C3 is SLO control without disaggregation (user, 2026-10-27)
- C3 now covers both levels of SLO control:
  - per-worker deadline scheduling;
  - the SLO-checked router: routing as admission priced in SLO terms, with four checks.
- This replaces the phase isolation that disaggregation buys with fixed roles. The pool stays credited to sglang /
  Mooncake. The abstract, sections 1, 4.1, 5.2.1, 6.1 (Table 4c) and 7 are updated.

## v1.71.1 — Paper: three contributions; the router is a supporting mechanism (user, 2026-10-27)
- C4 is dropped as a headline contribution:
  - the pool is sglang's (hierarchical cache over Mooncake), and the router's isolated gain so far is moderate;
  - the router stays, labelled as a supporting mechanism (sections 1, 5.2.1, Table 4c);
  - every C4 reference is rewritten (abstract, motivation, design, sections 5.4 / 6.5 / 7);
  - it would be re-promoted only if the trace-replay leave-one-out shows a large gain.

## v1.71.0 — Contributions stated precisely; gains tied to contributions; leave-one-out ablation queued (2026-10-27)
- Paper section 1:
  - C3 and C4 are now the mechanisms as built;
  - "what we reuse" states that the KV pool is sglang's hierarchical cache over Mooncake (ours is only the router's
    pull pricing), and that the executors are stock FA3 / FA2.
- Section 6.1: an attribution paragraph and Table 4c (contribution -> how it is removed -> measured cost), with the
  trace-replay leave-one-out cells pending.
- `pd_colocation/mn_ablation_contrib_16g_20261027.sh` runs the full stack minus C1 / C2 / C3 / C4 on the trace replay
  at alpha 3. Queued after the chunk-size test.

## v1.70.1 — Paper: results vs our two-plan build removed (user, 2026-10-27)
- Removed the "vs our two-plan build [95% CI]" column from Table 5, and every result quoted against that build: the
  stage-1 summary, the intro, the H200 and H20 kernel paragraphs, the kernel bar status, Figure 13's arms, section 7's
  plan-and-merge result, the DSMEM C = 4 row, FA2-rows-alone timing, GLM's 64-row class (now stated as 1.38x between
  the two placements) and the KV-split comparison (now stated as an ordering). The build's description as a baseline
  is gone too (sections 5.1 and 6.1, the artifact notes).

## v1.70.0 — Fault tolerance at 64 programs (2026-10-27)
- One node of four killed at 300 s, 64 programs + 16 chat/s:
  - ours holds (agent 95.9 -> 93.4 -> 97.3%, chat 95.8%, -11% goodput);
  - conditional disaggregation holds (-8..-19%);
  - stock PD losing a prefill node never recovers (0% after the kill, -64%).
- Table 10c and task doc.

## v1.69.6 — Paper section 6.4 bar status includes the trace replay (2026-10-27)

## v1.69.4 — Demand shift v2 complete (2026-10-27)
- Surge, stock PD 3 + 1: chat 2.2 / 0.0%, agent 2.7%, 9 tok/s (corrected in v1.69.5; the first entry was written before the score was read). Table 10d is complete except stock PD 2 + 2's agent side at alpha 6 (rerun queued).

## v1.69.3 — Demand shift, surge: stock PD 2 + 2 row (2026-10-27)
- Chat 17.2 / 11.1%, agent 4.7%, 13 tok/s goodput; Table 10d and task doc.

## v1.69.2 — Abstract, intro and story carry the demand-shift and fault results (2026-10-27)

## v1.69.1 — Paper Figure 23 (demand shift over time) (2026-10-27)
- `tasks/paper_figures/fig23_demand_shift.py`: issued prefill / decode demand and per-tenant attainment in 30 s
  windows, alpha 3 + chat surge, ours vs conditional disaggregation (data under `paper_figures/data/demand_shift/`).
  Conditional disaggregation's prefill side falls behind from about 1100 s and never recovers; ours recovers within a
  minute of the surge.

## v1.69.0 — Paper section 6.4: demand-shift results (Table 10d) (2026-10-27)
- Superposed TraceLab replay on 16 H20s.
  - Alpha 3, ours vs conditional disaggregation: chat +14 points, agent +8.7 points, agent goodput +57%.
  - Alpha 3 + chat surge: base chat 57.4 vs 36.5%, surge chat 46.2 vs 27.1%, agent goodput +90%, chat TTFT p90 1.2 vs
    34.9 s.
  - Alpha 6: ours +24% agent goodput, chat behind.
  - Stock PD collapses at both loads.
  - Stock PD surge rows and the 2 + 2 alpha-6 agent rerun are pending.

## v1.68.4 — Demand shift at alpha 3: stock PD rows (2026-10-27)
- Stock PD 2 + 2 / 3 + 1 at alpha 3: chat attainment 9.9 / 2.2%, agent SLO goodput 49 / 7 tok/s, median TTFT about 170-200 s; many requests cut by the stock PD router.

## v1.68.3 — H200 trial renamed vibecuda (2026-10-27)
- The 4-GPU H200 trial is now `bf7d041fb8313d8a`, named vibecuda as required; the misnamed vibecuda4 request was stopped (it never ran).

## v1.68.2 — Demand-shift replay results so far (2026-10-27)
- Superposed TraceLab replay on 16 H20s, alpha 3, ours vs conditional disaggregation: chat attainment 65.1 vs 50.9%,
  agent attainment 72.6 vs 63.9%, agent SLO goodput 1501 vs 956 tok/s (+57%), chat TTFT p90 1.12 vs 8.74 s.
- Alpha 6: ours +24% agent goodput, chat behind (TPOT p50 72 vs 28 ms). Stock PD collapses at alpha 6.
- Task doc item (a').

## v1.68.1 — Trace replay survives a failed round (2026-10-27)
- In the v2 demand-shift run, the stock PD 2 + 2 arm at alpha 6 lost its whole agent client at 01:09: one round failed
  (the stock PD router cut a response mid-stream), and the timestamp replay raised on any failure.
- Now a failed round ends only its own session. Its later rounds come back as "not issued: failed predecessor", the
  audit counts failed sessions and lost rounds, and the scorer reports `lost_after_failure` separately from the
  horizon backlog. Test added; all replay tests pass.
- The stuck arm was killed. Its rerun (`_tl2_a6_r1b`) heads the queued chunk-size driver.

## v1.68.0 — SLO-budgeted chunk on H20; trace scorer counts one-token replies by TTFT (2026-10-27)
- `apply_slo_budget_chunk_20261026.py` applies to our serving tree: it hooks after `_edf_limit_chunk`, takes
  `running_batch` as a parameter, and reads the prefix from the tree's own fields. Applied to a tree copy
  (`/dev/shm/w/python_slo`) on all four H20 pods; `mn_run.py` / `mn_node.sh` forward `SLO_CHUNK`.
- `pd_colocation/mn_slo_chunk_16g_20261027.sh`: step trace, lin4mode fit, then the SLO-budgeted chunk on the trace
  replay (alpha 6 / 3 / surge). Queued after the 64-program fault test.
- `tracelab_windows.py`: trace rounds with a one-token reply are judged on TTFT alone; they were counted as failures
  before. Demand shift at alpha 6 (v2, fixed 900 s horizon), ours vs conditional disaggregation: agent goodput 890 vs
  720 tok/s, rounds issued 4818 vs 3850, agent TTFT p90 13.0 vs 73.1 s; chat attainment 31.8 vs 57.2% (chat TPOT p50
  72 vs 28 ms).

## v1.67.2 — Paper: demand-shift setup updated to the v2 protocol (2026-10-26)
- Section 6.4, Table 4b and Figure 23 describe alpha 3 / 6, the fixed horizon, and the 300 s surge at alpha 3.

## v1.67.1 — Trace replay over a fixed horizon (2026-10-26)
- Session replay: `BENCH_REPLAY_HORIZON_S`. A round whose issue time falls past the horizon is not issued, nor are its
  session's later rounds; each gets a failed output marked "not issued". The scorer drops these from the tenant rows
  and reports them as backlog. Every arm is then measured over the same wall-clock window. Test: replay horizon
  check, ALL PASS.
- Without it, ours' alpha = 6 replay was still half done after 22 min (completion-dependent rounds stretch under
  overload), and stock PD would have exceeded mn_run's 90-min limit. The demand-shift driver was restarted as v2
  (`mn_tracelab_shift_v2_16g_20261026.sh`): horizon 1.5x the scaled trace hour; alpha 6 and 3, then a chat surge at
  alpha 3.

## v1.67.0 — Full-stack fault tolerance measured (2026-10-26)
- Qwen2.5-7B on 16 H20s, 32 programs + chat 8/s, one node killed at 300 s. Ours holds both tenants (agent 95.0 ->
  95.5 -> 96.8%, chat 98.7%, -9% goodput). Conditional disaggregation: -16% goodput. Stock PD 2 + 2 with a prefill node
  killed: agent 50.8 -> 21.8 -> 29.6%, chat 30%, -1/3 goodput; with a decode node killed it is level. Paper section
  6.5, Table 10c; task doc.

## v1.66.5 — Paper Figure 16a (single-node attainment) (2026-10-26)
- `tasks/paper_figures/fig16a_single_node.py`: Table 7b as grouped bars, three SLO panels, stock vs ours; embedded in
  section 6.3.

## v1.66.4 — Fault test at a chosen load (2026-10-26)
- `pd_colocation/mn_fault_load_16g_20261026.sh`: the full-stack fault test with `FAULT_LOAD=C:rate` and `FAULT_ARMS`.
  At 32 programs both colocated systems ride through a node kill (ours agent attainment 95.0 -> 95.5 -> 96.8%,
  conditional disaggregation 94.1 -> 93.2 -> 98.7%, no agent requests lost), so a 64-program point is queued behind
  the demand-shift run.

## v1.66.3 — H200 kernel-ladder chain (2026-10-26)
- `unified/a6_h200x4_kernel_ladder_20261026.sh`: on the 4-H200 trial, recaptures the trace-sampled steps; re-benches
  sglang FA3, the role split and v3; builds and checks A6 np64; benches naive FA3 + stock FA2 (serial and two
  streams, pip flashinfer in its own process); reports time-weighted speedups over every baseline.

## v1.66.2 — Paper Figure 22 (cache thrashing) (2026-10-26)
- `tasks/paper_figures/fig22_thrashing.py`: cached fraction, agent tok/s and agent TTFT against working-set size,
  ours vs stock PD 2 + 2 on 16 H20s; embedded in section 6.4.

## v1.66.1 — Paper Figure 2 (P:D ratio sweep) (2026-10-26)
- `tasks/paper_figures/fig02_pd_sweep.py`: stock PD chat attainment and agent SLO goodput against prefill pods
  (16 and 32 H20s), with ours as a reference line; embedded in section 2.3.

## v1.66.0 — Paper Figure 19 (capacity at SLO) from measured stage-3 data; fault scorer (2026-10-26)
- `tasks/paper_figures/fig19_capacity.py`: chat attainment and agent SLO goodput vs agent programs on 16 H20s, for
  Qwen2.5-7B and Llama-3.3-70B, comparing ours, conditional disaggregation and stock PD 2 + 2 / 3 + 1 (mean of two
  repeats with a span bar; validated categorical palette; de-collided direct labels). The data are the saved
  paper-SLO score lines (`tasks/paper_figures/data/`). Embedded in the draft; the Drive sync now resolves images.
- `tasks/perf_model/fault_windows.py`: paper-SLO attainment, losses and recovery time around a node kill.

## v1.65.1 — Paper: kernel and scheduling policy explained step by step; evaluation settings table (2026-10-26)
- Section 5.1 opens with what a mixed step asks of attention (packed rows, tile fill), the plan's four decisions
  (tile class, KV split, LPT placement, merge schedule), the two launches with one in-launch merge, and a worked
  Qwen2.5-7B example.
- Section 5.3 states the four scheduling mechanisms as built: deadlines from uncached work (single-node 0.5 s / 10 s;
  multi-node 2 s + 1 s per 5000 tokens), deadline order with demotion of missed prefills (single node), parking a long
  chunked prefill, and reserving budget for earlier deadlines.
- Section 6.1 Table 4b lists every stage's workload, load points, SLOs and hardware. Section 6.4 adds the trace-replay
  demand-shift experiment (pending), and section 6.5 the full-stack fault setup (pending).

## v1.65.0 — SLO-budgeted prefill chunk patch (2026-10-26)
- `unified/apply_slo_budget_chunk_20261026.py`: per step, the largest prefill chunk whose predicted step time keeps
  every running decode row within its mean-TPOT slack (per tenant, headroom h). With no decode rows it uses the
  server's full chunk size. Optional `demote` excludes rows already past their SLO. It hooks the chunk variable
  passed to PrefillAdder, so it composes with the EDF patches; a zero chunk skips prefill for that step.
- `perf_model/fit_4mode_step.py --export-lin4mode`: writes the lin4mode step model (gap / gpu targets, S / T) that the
  patch reads.

## v1.64.0 — Superposed TraceLab replay for the P/D demand-shift experiment (2026-10-26)
- TraceLab loader (`drivers/serving/bench_overlay_20261026/tracelab.py`): `--tracelab-window-starts s1,s2,...`
  superposes several one-hour trace segments on one clock, each timed from its own start, with session ids suffixed
  `@k`. One trace hour holds only 3-6 concurrent sessions, too few for 16 GPUs. Tests: superposed segments equal
  their standalone replays (`test_tracelab_window.py`, ALL PASS on the full release).
- Bench client: `request_start_epoch` and `cached_tokens` per request, so separate client processes share one time
  axis.
- `mn_bench.sh` `WINDOW_STARTS`; `mn_run.py` forwards the TraceLab / window / math knobs to the bench (they were
  dropped before, so the timestamp mode never ran multi-node).
- `tasks/perf_model/tracelab_windows.py`: per-tenant paper-SLO attainment and SLO goodput for trace-replay arms, plus a
  30 s series of offered prefill / decode demand and attainment.
- `pd_colocation/mn_tracelab_shift_16g_20261026.sh`: 24 superposed busiest hours (at least one hour apart), alpha
  4 / 6 / 8 plus a chat surge, with ours, conditional disaggregation, and stock PD 2 + 2 / 3 + 1 on 16 H20s.
  Queued behind the fault-tolerance run.

## v1.63.1 — Paper sync: tables render as real bordered tables (2026-10-26)
- `tasks/paper_sync_gdrive.py`: pandoc's tables referenced an undefined "Table" style, so they showed borderless, like
  text. Every table now gets grid borders at full width, with a shaded bold header row. The script fails if the docx
  table count differs from the markdown's. Verified through a temporary Google Docs conversion: 13 tables, all with
  solid borders, with the same row and cell counts as the markdown.

## v1.63.0 — Llama-3.3-70B stage 3 second repeat; full-stack fault-tolerance driver (2026-10-26)
- Llama-3.3-70B stage 3 r2 reproduces r1. Against stock PD's best split: +3..+23 points chat attainment at -2..+7%
  agent goodput. Against conditional disaggregation: +4..+94% agent goodput at -0.3..-7.1 points chat attainment.
  Table 10b now has both repeats, and the claims of "most goodput at every load" are corrected.
- `pd_colocation/mn_fault_fullstack_16g_20261026.sh`: full-stack fault tolerance on 16 H20s (one pod killed at 300 s;
  ours, conditional disaggregation, stock PD prefill-role and decode-role kills).

## v1.62.2 — Paper draft synced to Google Drive; 5.2.3 chunk-size framing corrected (2026-10-26)
- `tasks/paper_sync_gdrive.py` compiles `tasks/paper_outline.md` to .docx with pandoc, using the user's outline docx
  as the style reference, and uploads it to Drive as `attn_fuse_draft_2026.docx` (updated in place). Credentials stay
  outside the repo (`~/.config/attn_fuse/`). A local post-commit hook runs it whenever a commit touches the draft.
- Section 5.2.3: fixed 2048-token chunks are the configuration behind every measured result. The SLO-budgeted chunk
  replaces them only if it wins on prefill-heavy loads with no regression on decode-heavy and mixed loads (pending
  validation).

## v1.62.1 — Paper 5.2.3: SLO-budgeted chunk size (pending validation); 4-H200 trial (2026-10-26)
- Section 5.2.3 states the chunk rule: the largest chunk whose predicted step keeps every decode row within its
  mean-TPOT slack, with headroom h, and unbounded with no decodes. It is marked [pending validation]; the measured
  results still use fixed 2048-token chunks. The backlog cap moves to "what did not work".
- Task doc: the chunk-size test plan sits beside the naive FA3 + FA2 baseline. One 4-H200 trial (`d4b20cad59170ebd`)
  for the kernel and single-node stages; the queued 1-GPU trial was stopped.

## v1.62.0 — Paper draft: router design in detail, conclusion, stale parts cleaned (2026-10-26)
- Section 5.2.1 now describes the router as built: block-chained prefix index plus a pool mirror, the in-flight table,
  calibrated prefill curve and decode grid, online EWMAs, and optional piggybacked node load. It gives the TTFT / TPOT
  prediction step by step (pool pull, prefill cost T(n) - T(n - u), EDF queue wait, decode rows ahead), the four
  feasibility checks, the choice order, and measured accuracy and ablations.
- Kernel claim restated against naive FA3 + stock FA2 (pending on the H200); v3 is now "our two-plan build", an
  ablation of the one plan and merge. Tile classes are 16 / 32 / 64 everywhere (48 was stale).
- Portability (FA4 + FA2 on Blackwell) folded into section 5.1; the standalone 5.6 is removed. The conclusion now
  covers each stage, the lessons and the limitations.
- Stale parts fixed: Llama-3.3-70B stage 2 / 3 results, paper SLOs, final-configuration numbers in the abstract and
  intro, EDF deadline rule (2 s + 1 s per 5000 uncached tokens), chunk cap dropped, CUDA-graph root cause, router
  ablation results, internal jargon in References.

## v1.61.0 — Paper draft as prose; DSMEM cluster reduction measured; workload/SLO matrix (2026-10-25)
- Paper draft (`tasks/paper_outline.md`) rewritten from bullets into conference prose. Every evaluation stage gives its
  claim, bar, setup, results and figure captions; internal shorthand moved to an artifact-notes list. Headline numbers
  reconciled with the task doc: 1.027-1.047x vs v3 and 1.43-1.51x vs sglang FA3; JIT ready-queue 0.997-1.005x vs its
  own ticketed loop.
- DSMEM-cluster JIT reduction: exact at C = 2 / 4 / 8 / 16 on 96 H200 captures, but 0.60-0.70x / 0.93x / 0.97x of the
  ticketed in-launch merge. The plan's chunk cap (KV chunk >= kv_max / C) leaves the longest request too few splits.
  The merge-free bound caps any reduction scheme at 0.9% on these captures.
- FA2 L2 order (`UNIFIED_FA2_ORDER=kvpos`, flashinfer `f2703e4`): KV-position-major, for shared-prefix rows; measured
  on captures plus synthetic sets (running).
- Synthetic MELD-shaped steps (`unified/gen_synth_meld_20261025.py`): ShareGPT / math / long-tail / shared-prefix
  decode and verify rows at batch 16-1024; every executor arm plus native v3. The native harness's float workspace
  can now grow to 4 GB.
- Per-capture plan shape and FA2 balance tool (`unified/plan_balance_20261025.py`).
- Serving workload/SLO matrix in S, taken from JITServe / SLOs-Serve: chat 2 s + 50/100 ms; agent 10 s, 20 s above 200K
  tokens; TraceLab replay, mixed stream with agent share > 50%, standalone sets.
- CLAUDE.md: the draft is written for human readers; deadline 2026-10-30.

## v1.60.0 — 64-row FA2 class for GLM; A5.5 closed; stage-3 A6 worker mode (2026-10-25)

- A6 NP executor: a 64-row class (v3's BatchPrefill 64-row tile) in its own instantiation for plans with 64-row
  items. GLM-4.5-Air four-mode stage 1 goes from 0.724x to 1.000x of native v3. Qwen is unchanged.
- GLM-4.5-Air stage 1 pipeline: trace-sampled capture with NEXTN MTP, TP4 on H20.
- A5.5: FA3 zero-KV-load ceiling on the flash-attention role is 0.1-1.3% of the step (the loads were verified
  removed), so FA3 multicast is not built.
- Stage 3: `KERNEL=a6` worker mode in `mn_node.sh`, plus `mn_deploy_a6_20261025.sh` to push the A6 serving kernel to
  the H20 pods.

## v1.59.0 — Persistent FA2 executor at parity; plan-chosen instantiations (2026-10-25)

- Persistent overlap + LPT FA2 kernel: residency was register-bound at 2 CTAs/SM. The plan now picks one of two
  instantiations: 16-row-only plans get 3 CTAs/SM with the wide runners compiled out; plans with 32-row items get
  2 CTAs/SM with the 32-row class at MMA_KV 4 (v3's Q32 depth). Each plan also gets its own dynamic smem, and an
  undersized launch traps.
- FA2 rows (H20): 5.37 / 6.66 ms vs v3's FA2 launch 5.46 / 7.17. Stage 1 on the H200: 1.024-1.031x vs v3, within
  1% of the NP executor, which stays the default.
- A5.5: FA3 zero-KV-load ceiling driver on the flash-attention role (`a5_fa3_skipkv_ceiling_h20_20261025.sh`), with
  a timing-bound mode in the stage-1 bench.

## v1.58.0 — S4 step model on A6; persistent FA2 timeline; batched merge rows (2026-10-25)

- S4: the step model re-fitted on A6 serving traces (H200) predicts held-out steps at 2.0% MAPE vs 1.9% for stock's
  single FA3 call, and mixed steps better (5.9 vs 7.7%).
- `UNIFIED_TIMELINE=1` logs one globaltimer record per persistent-loop item (`timeline_fa2_rows_20261025.py`). It
  shows the persistent kernel's merge gap is the merge items' own latency (6-7 us for 4 rows) and the ready queue's
  pull cost (~2 us per pull vs 0.7 us).
- `UnifiedMergeRowCG`: one merge row helper for both FA2 executors, online max per chunk group and 4 chunks in flight
  per batch (was a max pass plus one dependent load round per chunk).
- Ready-queue polls use `__ldcg`; the priority threshold only runs when `UNIFIED_READY_FRAC > 0`.

## v1.57.0 — A6 beats native v3 on the H200; serving integration of the unified dispatcher (2026-10-25)

- A6 stage 1 on the re-forked H200 (96 fresh captures): 1.027-1.047x native v3 and 1.43-1.51x sglang's FA3 call
  under `geometry_fa3`, all FP32-exact. On H20 it is level with v3.
- What got it there:
  - Stock flash-attention for FA3 at the plan's split, which removed a uniform 1-2% scheduler loss.
  - v3's BatchPrefill body as the FA2 executor, with a chunk-parallel in-launch merge.
  - A fixed race: the embedded body triggered the next PDL launch before the counters were reset
    (`suppress_pdl_trigger`, atomic resets, ticket trap).
- Measured negatives: SM pre-assignment (a per-step oracle gains < 0.1%), interleaved merge items, last-arriver merge
  on four-mode, deeper KV tiles.
- Serving: `SGLANG_A6_UNIFIED=1` in `prepare_native_pdl` builds one unified plan per forward batch. The pair driver
  takes `--native-env`. Stage 2 (full stack vs stock, with v3 and A6 as the kernel) and Llama-3.3-70B stage 1 (TP4,
  H20) are set up.
- H200 restore: `prov_h200_20261024.sh` plus `fix_serving_src_20261025.sh` (serving-protocol client,
  `mn_bench_pair.sh`).
- Docs: A6 regrouped by requirement / feature; CLAUDE.md: group by item, never chronologically.

## v1.56.0 — Fused unified reduction; v3's FA2 body as a second executor; multicast ceiling (2026-10-24)

- A6, mandatory part: FA3's split rows are now reduced by plan-scheduled merge items in the FA2 launch, the same
  mechanism as FA2's groups (no tail loop). FP32-exact on all 96 captures.
- A6: `UNIFIED_FA2_NP=1` runs v3's own BatchPrefill body under the unified plan; v3's Q32 layout fix and
  independent PDL are ported into port07 `prefill.cuh`.
  - Retracted the same day: the "np" builds never received `-DUNIFIED_FA2_NP` (the build script forwards a fixed
    flag list), so their numbers measured the persistent kernel. The flag is now forwarded, and drivers fail if
    the nvcc line lacks it.
  - The persistent kernel's gap sits in few-row long-KV steps (0.71-0.74x of v3 there).
  - Next: the non-persistent executor with ticketed merge items (deadlock-free by ticket order).
- A5.5: shared-KV ceiling. FA2 under 1% (FA2 multicast not built). FA3 99% from one prefill row's q tiles, which
  are already L2 hits.
- New plan policy `geometry_fa3` (v3's FA3 split, fixed FA2 split); stage-1 run queued.

## v1.55.0 — A6 requirements recorded; FA2 parity measured against v3's own FA2 launch (2026-10-24)

- A6 requirements, per the user:
  - Mandatory: the unified plan, and the split-KV reduction in the FA2 epilogue with no merge launches.
  - The persistent overlap + LPT FA2 schedule stays the intended executor, demoted only with measured causes.
  - Optional: DSMEM JIT handoff and multicast.
  - Stages 2-3 run on v3 overnight.
  - The paper shows all three models (maybe two later: Llama-3.3-70B and Qwen2.5-7B).
- FA2 parity harness: new `v3rows` arm (v3's exact FA2 row launch). On identical rows ours is 6-13% slower
  (4.74 / 4.33 ms v3 vs 5.03 / 4.98 ms ours). Matching v3's KV depth or split does not close the gap.
- New plan options (flashinfer): `UNIFIED_SPLIT_POLICY=geometry` (v3's split) and `UNIFIED_MERGE_MODE=dsmem`
  grouping (host-checked; no kernel uses it yet).
- Reference lists now carry four bullets per paper (Core method/contributions added); the pd doc's entries are
  completed. Four claims the sources contradict were corrected: Dynamo's threshold is 2048 / 0.7 and its planner
  has a 5 s mode; semi-PD partitions dynamically; the delta-KV RFC is an open proposal.
- MELD paper PDF filed as `tasks/archive/meld_taming_dynamism_cross_sm_fusion_sosp26.pdf`.

## v1.54.0 — Plan-scheduled merge items; unified level with native v3; docs deduplicated (2026-10-24)

- A6: FA2 merge groups are cut into row ranges and queued after every compute item. Our FA2 rows go from 0.86x to
  1.10x stock BatchAttention, and stage 1 on 96 captures reaches 0.97-1.00x native v3 (level, not better).
- A6 negatives: the L2-swizzled FA3 tile order and a deeper KV tile on the 16-row class (MMA_KV 4 spills; 8 exceeds
  the shared-memory limit) do not help. Build knobs `UNIFIED_MMA_KV16` / `UNIFIED_MMA_KV32` (flashinfer `b544bb9`).
- A6 restated: the baseline is native v3, which runs the same FA3 kernel and an already-tuned FA2 path. Our runner
  reaches parity with v3's FA2 optimizations first, then the overlap + LPT schedule is measured on top. Open items:
  MELD-style DSMEM JIT reduction and FA3 split/pulling.
- Task doc deduplicated:
  - Multicast is one item (A5.5, reopened for FA3 and FA2 under the unified plan; B3 is merged into it), and
    backfill is one item (A5.6; B2 is merged into it).
  - Superseded A6 sub-items are condensed, and the retracted session-rate EDF sweeps are reduced to their one
    surviving finding.
  - The stale MuxWise and Dynamo items are closed, and S0's models match the 2026-10-24 decision.

## v1.53.0 — Unified dispatcher stage 1: correct on 96 captures, 2-5% behind native v3 (2026-10-24)

- Unified tile dispatcher (A6) runs end to end on the trace-sampled captures with three KV-split policies.
  - Correctness: FP32 worst 0.0067.
  - vs sglang's FA3 call: 1.08-1.52x.
  - vs native v3 and the role split: 0.89-0.98x, i.e. not yet better. The fixed policy is best; flashinfer's
    persistent-LPT rule is worst.
- Fix: the merges skip -inf partial slots. 0 x stale NaN produced one NaN row of a causal FA3 prefill.

## v1.52.0 — Unified tile dispatcher built and correct on captures; FA2 executor speed work (2026-10-24)

- Unified dispatcher (A6), flashinfer branch `unified_dispatch_20261024`:
  - `UnifiedTilePlan`: one plan per step with LPT placement, JIT merge groups and an FA3 work-based split.
  - `unified_dispatch_ext.cu`: FA3 launched from task 3 of the plan, PDL-chained to an FA2-only persistent kernel
    with the merge in its epilogue and an FA3 tail merge.
- FP32 gate passes on real captures: all-FA2 0.0037, all-FA3 0.0037 (bit-identical to sglang's FA3 call), mixed
  0.0058.
- Speed is not there yet:
  - FA3 side: 0.070 ms vs sglang's 0.135 after the work-based split.
  - FA2 kernel: 0.25 ms vs native v3's 0.080 for the whole step. Cause traced to a single-CTA serial merge (two-pass
    merge halved it) and to the 48 runner's spills (1 444 B; 16-20 B with it compiled out).
- Outline:
  - 5.1 explains the fused-kernel failure as a warp-layout mismatch, with the measured numbers.
  - Each eval section lists its graphs and metrics.
  - 6.5 gains the prefill-down / decode-down fault-tolerance experiment, also queued in `pd_colocation_vs_disagg.md`.

## v1.51.0 — C1 restated as the unified tile dispatcher + merge schedule; kernel items unchecked (2026-10-24)

- C1 is one CPU-side plan that assigns every tile (FA2 16/32/48, FA3 > 48) and the split-KV merge schedule (JIT
  reduction by the last partial, no merge kernel). Running FA3 and FA2 side by side is not the contribution.
- Task file: the "Where it stands" block is removed; the current goal is rewritten; A and B are unchecked (never
  built); A6 now specifies the unified plan.
- `paper_outline.md` follows the original docx outline format. CLAUDE.md records it as the exception to the task-doc
  format.

## v1.50.0 — Outline rewritten per stage; kernel goes back to the user's tile rule + LPT; Llama-70B provisioning (2026-10-24)

- `tasks/paper_outline.md` rewritten: current goal; per-stage claim / rationale / baselines / results for the three
  models. The old outline is archived as `tasks/archive/paper_outline_pre_rewrite_20261024.md`.
- Kernel task reopened. The serving FA2 rows run a non-persistent kernel with a 64 tile and no LPT schedule, against
  the rule (FA2 16/32/48, >=64 to FA3).
- Llama-3.3-70B provisioning for H20 pods 0-3 (H200-built native kernel libraries, identical ABI);
  `mn_cap16_cond_p1_20261024.sh` (conditional-disagg 1-prefill-pod split).

## v1.49.0 — 16-GPU sweep vs conditional disagg: ~17% more load at SLO (2026-10-24)

- Conditional disagg (best split, 2 prefill pods) vs ours. Chat attainment is equal through 64 programs; at 96/128
  ours leads by 4.8/11.7 points. Agent SLO goodput is +14..+37% from 48 programs. Interpolated load at 90% chat
  attainment: ~91 vs ~78 programs.
- The stage-3 claim is restated against this baseline. A 1-prefill-pod split is queued
  (`mn_cap16_cond_p1_20261024.sh`).

## v1.48.0 — Conditional disagg matches ours at 64 programs on 16 GPUs (2026-10-24)

- cond_p2 at 64 programs: chat attainment 94.6% vs ours 94.2% (stock PD 31.2%); agent SLO goodput 845 vs 964 tok/s.
  Only 0.5% of requests were prefilled remotely, so most of ours' gain over stock PD is the per-round KV re-ship that
  Dynamo-style conditional disaggregation already avoids. `mn_cap16_cond_high_20261024.sh` queues the knee (96/128).

## v1.47.0 — Disagg + EDF changes nothing; conditional-disaggregation (Dynamo) baseline (2026-10-24)

- 16 GPUs, disagg 6+2 with deadline order on the prefill pods: chat attainment 92.5 / 70.7 / 46.7 / 30.5% at
  16/32/48/64 programs, the same as FCFS. The limit is the per-round KV re-ship: at L64 the decode pods hold 52-55
  requests in `#transfer-req`.
- `router.py --policy cond_disagg`: Dynamo's conditional disaggregation on the same colocated workers as ours. Long
  uncached prompts prefill remotely, with KV delivered through the pool by write-through; short appends prefill
  locally.
  - `mn_run.py` adds `cond_p<P>` arms; `mn_fanout_8gpu_20261018.sh` adds a `cond_*` case.
  - `cond_disagg_check.py` checks validity; `test_router_cond_disagg.py` is the unit test.
  - `mn_cap16_cond_disagg_20261024.sh` runs the sweep.

## v1.46.0 — 32-GPU capacity table; disagg + EDF prefill baseline (2026-10-24)

- 32 GPUs: ours holds >= 96% chat attainment through 128 programs; best disagg split 83.7% at 32, 26.7% at 128.
  Ours' agent SLO goodput rises 569 -> 1910 tok/s.
- `apply_edf_prefill_role_20261024.py`: EDF deadlines on PD prefill-role workers (order only), so disagg can run
  with the same deadline order as ours. `mn_fanout_8gpu_20261018.sh` gains `DISAGG_SGL_PY`;
  `mn_cap16_disagg_edf_20261024.sh` reruns the 16-GPU load grid with disagg 6+2 + EDF.

## v1.45.0 — v3 single-node full stack: attainment bar met at 32-64K; no single-GPU capacity point at 128-256K (2026-10-24)

- 32-64K, two repeats: chat strict attainment 49-50% -> 59-65%, relaxed 50% -> 83-85%, agent 45-52% -> 77-85%; chat
  TTFT -96%; agent tput +1.7% (two) / -8.5% (four); chat TPOT +35-56%.
- 128-256K at agent concurrency 1 / 2 / 4: neither system reaches 90% for either tenant. Agent attainment caps at 50%
  (cold first turns cannot prefill in 10 s), and chat strict stays <= 46%. Ours still gains attainment at C=4.
- Outline 6.3 C2d rewritten on v3; C2c marked superseded for the full stack.

## v1.44.0 — Demote-on-miss adopted; v3 single-node run; 16-GPU capacity point (2026-10-24)

- 128-256K screen: demoting a prefill only after its deadline passes (`SGLANG_EDF_LATE_DEMOTE_TOK_S=1e12`) beats v2's
  6000 tok/s feasibility test on every axis. Four-mode agent tput +1.4% (v2 -6.4%); chat strict attainment 26.7%
  (v2 18%).
- `ours_full_stack_v3_h200_20261024.sh`: both stages + agent-concurrency capacity sweep (C=1, 2).
- 16 GPUs: ours 89.2% / 72.6% chat attainment at 96 / 128 programs, so ours' capacity is 64-96 programs vs disagg's
  16-48. 32-GPU phase A: ours 99.1% vs best disagg split 55.9% at 64 programs.

## v1.43.0 — v2 single-node full stack: two-mode wins attainment everywhere, four-mode does not (2026-10-24)

- Two repeats at 128-256K (chat 4/2/1 req/s) and 32-64K. Every cell: chat TTFT -95..-99%, agent TPOT -12..-55%,
  agent attainment up or equal. But chat TPOT +26..+84%, four-mode agent throughput -6..-16%, and four-mode at
  128-256K loses chat strict attainment. The single-node bar is not met as a whole.
- Diagnosis: the 6000 tok/s demotion rate demotes every cold 128-256K agent on arrival. The chat-rate sweep is the
  wrong load axis, since agents alone saturate one GPU. Demote-on-miss screen launched
  (`edf_demote_on_miss_h200_20261024.sh`).

## v1.42.0 — Late demotion fixes single-node chat TTFT under overload; v2 full-stack run (2026-10-24)

- Screen at 32-64K: per-tenant deadlines alone change nothing; adding `SGLANG_EDF_LATE_DEMOTE=1` cuts chat TTFT
  96-98% and lifts chat attainment at 1 s/30 ms from 50% to 84-86%. Strict 20 ms chat TPOT stays bound by decoding
  beside continuous long-context prefill. Four-mode agent throughput falls 15%.
- `ours_full_stack_v2_h200_20261024.sh`: both stages + chat-rate sweep with the fix, two repeats each.

## v1.41.0 — 32-64K full stack misses the bar; EDF deadline mismatch + overload demotion under screen (2026-10-24)

- 32-64K full stack vs stock on FA3: agent tput -3.4% / +5.0%, chat attainment flat ~50% in both arms. All agent
  requests land in the first 60 s; ours' EDF ran a 2 s chat deadline (scored SLO 0.5 s) with no overload handling.
- `apply_edf_late_demote_20261024.py`: `SGLANG_EDF_LATE_DEMOTE=1` ranks prefills that can no longer meet their
  deadline behind every feasible one (deadline order kept among the demoted); retracted decodes never demoted.
- Harness `--ours-env KEY=VALUE`; audit `--repeats`; `edf_overload_screen_h200_20261024.sh` screens per-tenant
  deadlines (0.5 s / 10 s) with and without demotion. 128-256K rate sweep stopped (it used the 2 s deadline).

## v1.40.0 — Full single-node stack vs stock sglang on FA3, 128-256K (2026-10-24)

- H200, paired and audited: ours (deadline order + chunk preemption + deadline-sized chunks + FA3 + FA2 v3 + 4-mode)
  vs stock sglang mixed chunk on FA3 (stock EAGLE3 MTP in four-mode). Agent tok/s +12%. Chat TTFT -69% (four-mode) /
  -79% (two-mode). Agent TPOT -35%; agent ITL -31% (four) / -12% (two). Chat mean TPOT +17% / +30%, with a growing p90
  tail. Past both systems' capacity, so the attainment ratios are not headlined; a chat-rate sweep is queued.
- The audit requires a four-mode merge exactly when `SGLANG_MIXED_SPEC=1`, and zero merges otherwise. Before this, it
  rejected the valid stock-MTP control.
- `ours_full_stack_h200_20261024.sh` gains `ONLY=` / `AUDIT_ONLY=`; new `ours_full_stack_rate_h200_20261024.sh`
  (chat 1 and 2 req/s).

## v1.39.0 — Same-session: FA3 + FA2 rows win agent decode latency; role split is the ablation (2026-10-24)

- 128-256K, H200, paired vs FA3, same session and tree: FA3 + FA2 native v3 gives +10.7/+11.6% agent throughput and
  cuts agent TPOT 20-24% / ITL 17-21%; the role-split FA3 gives the same throughput but agent TPOT -9.0/+6.6%, ITL
  -1.5/-5.7%. The FA2 rows are worth ~15-25 points of agent decode latency in serving, invisible to the per-step
  benchmark. The fused FA3 + FA2 kernel is the one to ship.

## v1.38.0 — Fair multi-node result: ours sustains >= 2-4x the load of best-tuned disagg at SLO (2026-10-24)

- 16 H20s (8 pods x 2), both on FA3, disagg's P:D swept at worker granularity with the KV pool on, load swept.
  Best-tuned disagg (6+2) holds >= 90% chat attainment only at 16 agent programs (92.8%; 67.8% at 32); ours holds
  94.2% at 64. A lower bound -- ours' capacity is not yet reached; 96 / 128 programs queued.
- At equal light load the systems are close (99.5% vs 92.8%); the gap is saturation: disagg's agent throughput
  plateaus near 360-410 tok/s while ours scales to 991. Ours' agent ITL p99 stays worse (109-134 vs 29-31 ms).

## v1.37.0 — Role-split FA3 kernel in the native executor; stage-2 serving rerun with it (2026-10-23)

- `SGLANG_ROLE_SPLIT_FA3=1` in `native_fa3_fa2_pdl_20261023.py`: the mixed step runs as two FA3 launches (prefill
  rows on a side stream, decode/verify rows with num_splits 32), no FA2 planning. On 48 trace-sampled four-mode
  captures (H200): 1.249x / 1.389x vs sglang's FA3 call at 32-64K / 128-256K, 1.00x vs the fixed role-split FA3 it
  reimplements, 0.99x vs a per-step oracle; FP32 worst 0.0046 (vs 0.0075 with FA2 rows). Off by default.
- `run_native_serving_pair_20261023.py --role-split`: sets the switch explicitly for the candidate arm (the harness
  strips every inherited SGLANG_* key, so an exported variable would never have reached the server).
- `rolesplit_serving_h200_20261023.sh`: paired FA3 vs role-split serving, 128-256K and 32-64K, two- and four-mode,
  two repeats, audited -- running.
- Caught before measuring: a script run from a directory holding the unpatched module imports that copy ahead of
  PYTHONPATH, so the first role-split bench would have timed the old FA2 path.

## v1.36.0 — Stage 1 on trace-proportional captures: role separation is the contribution (2026-10-23)

- 96 real mixed serving steps sampled in proportion to the trace (two/four-mode, 32-64K and 128-256K), H200, v3
  policy, all FP32-checked. Native is 1.23-1.57x faster than sglang's single FA3 call (every CI lower bound > 1.09),
  but a pure-FA3 role split with one fixed setting is 1-2% faster than native: the gain is splitting the mixed step
  by role, and the FA2 rows do not pay on the H200.
- Resolves the stage-2 inconsistency (128-256K four-mode attention 1.37x faster, consistent with +10.6% throughput).
- Fixes bench_captured_mixed's expected-record count for the added v3 arms (the data was complete; only the final
  assertion was wrong).

## v1.35.0 — Native kernel passes the downstream quality gate; stage 1 reweighted by real traces (2026-10-23)

- Quality gate (H200, Qwen2.5-7B): GSM8K and 32-60K passkey, native vs FA3 run twice, two- and four-mode. Native is
  within FA3's own run-to-run spread on every task (GSM8K 76.50 vs 76.57/76.57, 76.19 vs 76.88/75.44), passkey 100%,
  native path confirmed firing. PASS on all four checks.
- Stage-1 kernel speedups reweighted by each workload's census of real mixed steps: 0.975-1.13x of mixed-step
  attention time, not the 1.5-1.8x per-bucket headlines (those buckets are <1% of steps).
- Audited Codex's Qwen2.5 v3 common-workload run: all checks pass; +5.8% agent throughput, -9.1% ITL (two-mode).

## v1.34.0 — Outline restructured by stage; two failed multi-node arms diagnosed as invalid (2026-10-23)

- Paper outline section 6 now runs kernel -> single node -> multi-node with the baseline and claim at each stage.
  Multi-node goodput ratios (7-44x) are withdrawn as headlines: they came from an overload point with a pod-granular
  P:D sweep and no KV pool for disagg. A capacity-at-SLO sweep (8 pods x 2 GPUs, P:D swept 1-7, pool on both sides)
  is running.
- Predictor value: a 1.25x pessimistic step model matches the calibrated one (1092 vs 1094 tok/s), so on the
  overestimating side accuracy does not change SLO control at this load. The 0.75x arm and the kernel-on MTP arm are
  invalid runs (stream cuts / a silent chat client, 0 worker tracebacks, cause unknown); OOM kills in dmesg predate
  both and are unrelated.

## v1.33.0 — Clean multi-node table complete: MTP widens the gap to 19x / 44x (2026-10-22)

- 16 H20s, every arm on upstream's FA3 backend, ours with no split, disagg swept over 2+2 and 3+1, no-spec and MTP.
  With MTP vs disagg's better ratio: chat SLO attainment 92.8% vs 4.9% (19x), agent SLO goodput 1099 vs 25 tok/s
  (44x), raw throughput 5.2x, agent TTFT -98%. Ours' chat TPOT p90 9.5 ms also beats disagg 3+1's 13.8 ms.
- Speculation widens the gap because disagg ships draft hidden states to decode pods every round on top of
  re-prefilling the session, while colocation verifies where the KV already is.

## v1.32.2 — Correction: most of the C=16 single-node deficit was a tuning artifact (2026-10-22)

- Ours no MTP at C=16 with deadline chunk 1024:4096 instead of 1024:2048: agent attainment 34.4% -> 48.4% (stock
  50.5-55.9%), chat 55.3% vs 57.0% (parity), agent TTFT 6.72 -> 5.04 s, chat TTFT 0.95 s (2x better than stock).
- So v1.32.1's "monotone degradation, single-node question settled" was premature: it swept load while holding a
  chunk size suited only to light load. C=24 needs the same retest. Unchanged: ours does not clear the bar on one
  GPU at any load measured, and multi-node is far stronger.

## v1.32.1 — Single-node question settled: the advantage is light-load only (2026-10-22)

- Agent concurrency sweep complete (8 / 16 / 24, every arm on FA3, n=45 / 93 / 140). Ours goes from parity with
  stock at C=8 to about half its attainment at C=24: agent 17.9% vs 30.7%, chat 26.4% vs 29.8%, agent TTFT 13.5 s
  vs 4.9 s. Ours + MTP collapses to 11.0% chat attainment and 25.5 s agent TTFT.
- The degradation is monotone in load, so all three load directions are now closed (light load has no headroom,
  saturation favours stock, chat-heavy is bounded by ours' chat TPOT tail). The single-node contribution is a
  light-load latency/throughput trade; the quantitative case rests on multi-node.

## v1.32.0 — BAR MET on multi-node, confound-free, with no kernel work (2026-10-22)

- 16 H20s, no spec, every arm on upstream's FA3 backend, ours with KERNEL=lpt (no split at all), disagg swept over
  2+2 and 3+1. vs disagg's best ratio: chat SLO attainment 96.5% vs 23.3% (+314%), agent SLO goodput 939 vs 129
  tok/s (7.3x), raw agent throughput +179%, agent TTFT -95%, chat TTFT -99.5%. Requirement is >=30% attainment and
  >=10% throughput.
- Ours' chat TPOT p90 (11.5 ms) also beats disagg 3+1 (13.3 ms), so ITL favours ours at this scale.
- The advantage is entirely the colocated schedule plus the cache-first SLO router; the fused kernel contributes
  predictability (point 3) and ~2.5% on the single-GPU schedule, not this.

## v1.31.0 — Multi-node confound removed: +297% with NO kernel work (2026-10-22)

- 16 H20s, no spec, every arm on upstream's FA3 backend with no split planning (KERNEL=lpt): ours 962.7 vs disagg
  2+2 242.3 agent tok/s, agent TTFT 0.89 vs 27.57 s, cached fraction 0.97 vs 0.76. Stronger than the confounded
  version, and attributable entirely to the colocated schedule plus the cache-first SLO router.
- Disagg retains the better ITL tail (14.4 vs 110.1 ms p99).

## v1.30.2 — Saturation does not favour our schedule on one GPU (2026-10-22)

- Agent concurrency 16, every arm on FA3, n=93 agent requests. Stock degrades as predicted (agent attainment 84 ->
  50.5-55.9%, chat 80 -> 57.0%), but ours at chunk 2048 is worse on everything: agent 34.4%, chat 55.2%, agent TTFT
  6.72 s vs stock's 3.16-3.55 s. EDF's deferral of loose-deadline agents compounds under saturation and chat gains
  nothing because it is starved by agent prefill volume regardless of ordering.
- So the single-GPU advantage is a LIGHT-load phenomenon, the opposite of multi-node where the lead grows with load
  and cache pressure. The difference is the routing dimension, which one GPU does not have.
- Queued the last single-node test: ours at its best chunk (4096 / 8192) under C=16.

## v1.30.1 — The 4-mode MTP arm was mis-tuned: chunk 4096 recovers most of the throughput gap (2026-10-22)

- Ours + MTP at deadline chunk 1024:4096 instead of 1024:2048: agent throughput -3.8% vs stock + MTP (was -12.5%),
  chat attainment 77.9% vs 77.0% (now ahead), chat TTFT -77%, agent TTFT -17%, 30 ms-ITL agent attainment +21%,
  parity on the loose agent SLO. One repeat; repeats and a 8192 point still needed.
- The 4-mode step rides a prefill chunk with the verify forward, so it wants a LARGER chunk than the no-MTP
  schedule -- the opposite of how every earlier run configured it.

## v1.30.0 — Point 2 holds across the whole thrashing curve (2026-10-22)

- 16 H20s, both on FA3, working set swept 1M -> 4M -> 8M live KV tokens at fixed agent program concurrency. Ours:
  -10.6% throughput, -3.1 points cached fraction, agent TTFT 0.60 -> 0.68 s. Disagg 2+2: -55%, -28.5 points, agent
  TTFT 7.0 -> 22.6 s. Lead +91% -> +266% -> +277%; chat TTFT 76 ms vs 17.8 s at the last point.
- Degradation is monotone in both systems and tracks the cached fraction, so the re-prefill mechanism is measured
  rather than inferred. Disagg retains the better ITL tail throughout.

## v1.29.3 — Five-schedule FA3 table complete (10/10 arms) (2026-10-22)

- Final arms in. Throughput reproduces to within 0.3% across repeats for every arm: stock mixed chunk 141.8/141.8,
  stock prefill-prioritized 142.1/141.7, stock + MTP 161.2/161.5, ours no MTP 133.9/134.2, ours + MTP 140.6/141.5.
  So the throughput deficits (-5.5% no MTP, -12.5% with MTP) are real, not noise, while agent attainment is noisy
  (ours no MTP 86.7/95.6%).
- At the 30 ms ITL target ours + MTP leads stock + MTP 55.6/51.1% vs 46.7/48.9% in both repeats.

## v1.29.2 — Five-schedule FA3 table: bar not met at this load; attainment is SLO-threshold dependent (2026-10-22)

- Every arm on sglang's FA3 backend so the schedule is the only variable. Ours no MTP: +6.5% agent attainment,
  -2.5% chat attainment, -5% throughput vs stock prefill-prioritized. Ours + MTP behind stock + MTP on primary
  attainments. Bar (>=30% / >=10%) not met at chat 8 req/s with agent concurrency 8.
- Under the tighter 30 ms ITL target the ordering flips: ours + MTP 55.6% vs stock + MTP 46.7-48.9%. The result is
  SLO-threshold dependent and should be reported that way.
- Ours' agent attainment spans 86.7-95.6% across two repeats of one configuration (n=45, one request = 2.2 points),
  so earlier single-repeat headlines were over-precise.

## v1.29.1 — Point 2 supported: cache thrashing separates colocation from disagg (2026-10-22)

- 16 H20s, both on FA3, working set swept by distinct agent tasks at fixed program concurrency. From 16 to 64 tasks
  ours loses 4.5% agent throughput and 1.4 points of cached fraction; disagg 2+2 loses 50% and 21 points, with agent
  TTFT 3x and chat TTFT 2.4x worse. Ours' lead widens from +91% to +266%.
- The cached fraction is the re-prefill proxy: disagg's prefill pods lose an agent session's prefix between rounds
  and recompute it. 128-task point (past cluster capacity) running.

## v1.29.0 — Point 3 supported: the fused step is the most predictable one measured (2026-10-22)

- With the 4-mode schedule actually traced and fit (the earlier arms logged no STEPTRACE lines), ours + MTP on the
  fused backend predicts GPU time at 0.4% MAPE on decode steps and 3.2% on prefill-bearing ones, vs stock + MTP on
  FA3 at 1.5% / 5.7% -- 3.8x and 1.8x better. This reverses the 2026-10-11 "premise inverted" finding, which rested
  on a flashinfer-backend baseline.
- The same schedule on upstream's FA3 backend is 2.7-4.5x worse to model, so the fused split is what buys the
  predictability. The kernel's measured value is SLO-control predictability, not throughput (+2.5%).

## v1.28.2 — The 4-mode step runs on upstream's FA3 backend; no port needed (2026-10-22)

- MIXEDSPEC_STEPS 851, accept 2.05, 0 tracebacks: our 4-mode MTP schedule runs on sglang's own FA3 backend, so the
  fused-split backend was never load-bearing for the schedule.
- Backend choice is schedule-dependent: ours + MTP is +2.9% on upstream FA3 (140.6 vs 136.6), ours without MTP is
  +2.5% on our patched backend (137.3 vs 133.9). So the kernel work is worth ~2.5% for our own schedule, not nothing.
- Best agent attainment to date: ours no MTP on FA3, 95.6% vs stock's 84.4% (+13.3%), at -5.7% throughput. Bar still
  unmet on single node; the deficit is the schedule's throughput cost, not the backend's.

## v1.28.1 — Single-node table queued on the FA3 footing (2026-10-22)

- `h200_fa3_table_20261022.sh`: five schedules (stock mixed chunk, stock prefill-prioritized, stock + MTP, ours no
  MTP, ours + MTP 4-mode), every arm on sglang's FA3 backend, two repeats, per-tenant attainment. With the backend
  held fixed the schedule is the only variable, which is the footing the bar should have been judged on all along.

## v1.28.0 — The backend gap is sglang's flashinfer backend class (2026-10-22)

- Control arm, same schedule and load, only the backend differs: stock on the FA3 backend 7.48 ms / 161.2 tok/s,
  stock on the flashinfer backend with NO split at all 11.98 ms / 81.1 tok/s. The flashinfer backend is half the
  throughput on this workload -- which is why upstream made FA3 the Hopper default -- and our split was built
  inside it. Our FA3 patches recover 81 -> 147; the residual ~9% is that class's own machinery.
- Our scheduling is in scheduler.py, not the backend, so it should run on the FA3 backend with no split and no
  port. Measuring (`h200_ours_on_fa3_20261022.sh`).

## v1.27.2 — Dispatch fastpath: correct but not the cost; backend-class control queued (2026-10-22)

- `apply_split_fastpath.py` gate passes (0 failures, worst 0.0166, both paths exercised) and leaves serving
  unchanged at 146.9 tok/s, so the per-layer output copy and view rebuilds were not the 114 us/layer.
- Queued the control that was missing all along: stock's schedule on the flashinfer backend with no split at all.
  It decides whether the remaining cost is our split code or the backend class, and so whether the port is needed.

## v1.27.1 — The backend gap is per-layer dispatch overhead, not attention (2026-10-22)

- Ruled out with measurement: the split itself (forcing one FA3 call per extend step changes nothing) and the page
  table's row stride (-4..+4% across 2.1x-28.4x stride ratios, all within fp32 reference).
- Located: the slow bin's steps carry 164 prefill tokens, 37 prefix and zero decode rows, so the 3.2 ms gap is
  ~114 us/layer of dispatch work on launch-bound steps. `apply_split_fastpath.py` drops the per-layer output copy
  and allocation on single-call steps and caches the paged K/V views. Gate and A/B running.

## v1.27.0 — PD-colocation doc: evidence state and standing rewritten to current measurement (2026-10-21)

- The four-point evidence table and the standing summary were dated 2026-10-10 and contradicted
  everything measured since (they claimed no serving data, points 2 and 4 unmeasured, and a
  long-fixed blocker). Rewritten with both systems on FA3: points 1 and 2 now rest on multi-node
  measurement, point 4's pull rule is refuted at <= 16 H20s, point 3's premise stays inverted.
- Recorded plainly: multi-node clears the bar (+56% to 3.5x agent throughput vs disagg at its best
  ratio), single node does not (-3.4% throughput vs stock on FA3), ours' ITL tails are worse than
  disagg's, and the kernel contribution reduces to the per-step split decision.

## v1.26.5 — Backend gap decomposed; best single-node ours (2026-10-21)

- Skipping the unused per-step flashinfer plan: +1.7% (counter-verified). With draft extend on FA3 the backend gap
  is +2% / +1.7% / ~9% unaccounted; a step trace of both backends is queued to split GPU time from host gap.
- Best ours on single node, two repeats: no MTP 137.3 / 137.4 tok/s (stock on FA3 142), agent attainment +9 points,
  chat TTFT -78%, chat attainment -1.5 points. With MTP 135.4 / 136.3 vs stock 161.5. Bar not met.

## v1.26.4 — GLM-4.5-Air single node; a TP4 hang; an invalid knob test caught (2026-10-21)

- GLM-4.5-Air on one TP4 H20 pod: ours + MTP is -15% agent throughput vs stock + MTP at quarter load and wins only
  chat TTFT; at half load stock + MTP collapses but ours no-MTP is the better arm. The 4-mode schedule does not pay
  on this model.
- A forward step hung in ours no-MTP (600 s watchdog, TP1), not reproduced in 4 reruns. Left open with a soak planned.
- The first SKIP_BASE_PLAN run measured nothing: the pod carried the pre-flag launcher, so the knob never fired while
  the arm still produced a plausible number. The script now verifies the pod exports it before arming.

## v1.26.3 — Rate sweep vs stock on FA3; MTP gap located; draft extend on FA3 (2026-10-21)

- Chat 8-16 req/s: ours +9..+14% agent attainment and -74..-80% chat TTFT, but -2.5..-4 points chat attainment and
  -3..-5% throughput (chat TPOT tail). The bar is not met.
- MTP gap of 161 vs 131 tok/s: ~12% is our backend (stock's own schedule on our backend: 142.5), ~8% is the 4-mode
  schedule. `apply_draft_fa3.py` (`DRAFT_FA3=1`): the draft model's draft extend on FA3. Queued.

## v1.26.2 — GLM-4.5-Air output gate passes; launcher supports NEXTN and no YaRN (2026-10-21)

- One H20 pod at TP4: ours agrees with stock within stock's own run-to-run noise. Acceptance 2.5-2.75.
- `mn_node.sh`: `DRAFT_DIR=nextn` (the model's own MTP layer), `YARN_FACTOR=0`. `mn_h20_single_20261021.sh`
  single-pod harness. The split policy changes nothing in serving (ours no MTP 135.8 tok/s, chat 78.5%).

## v1.26.1 — Qwen3-8B with MTP on 16 H20s, both on FA3 (2026-10-21)

- Ours 448 vs disagg 105 / 128 tok/s (2 + 2 / 3 + 1); agent TTFT 1.7 vs 24-31 s. Ours' agent ITL p99 is 340 ms vs
  30-58 ms for disagg.

## v1.26.0 — Per-step split-or-one-call policy; SM partition negative (2026-10-21)

- Per-step SM partition lost 9% in serving: serving steps are chunk-dominated, so reserving SMs for the rows slowed
  the chunk. On serving-shaped steps the split beats one FA3 call by 20-77% when a short chunk sits beside long
  rows, and loses 0-21% beside a long chunk. Shared SMs beat a reserved partition.
- `apply_split_policy.py` (`SPLIT_POLICY=1`): split only when the chunk is short; otherwise one FA3 call over all
  rows. Every extend step now runs on FA3. Gate and A/B running.

## v1.25.6 — Plain decode on FA3 in serving (2026-10-21)

- Ours without MTP, every attention call on FA3: 137 vs 142 tok/s for stock on FA3, agent attainment +7..+9
  points, chat attainment -1.5 points, chat TTFT -78%. Gate: 0 failures, worst error 0.011.

## v1.25.5 — 32 H20s at 2x load, both on FA3 (2026-10-21)

- Ours 1172 vs disagg 5 + 3 444 tok/s agent throughput (+164%); agent TTFT 0.72 vs 12.25 s. Disagg holds the lower
  ITL tail.

## v1.25.4 — Per-step SM partition for the co-run (2026-10-21)

- `apply_split_auto_margin.py` (`SPLIT_MARGIN_AUTO=1`): the chunk / rows SM split follows the step's predicted
  attention times. Every earlier serving co-run had margin 0, so both launches shared all SMs. A/B queued.

## v1.25.3 — Verify on FA3 in serving; co-run vs FA3 stock with margin to 112 (2026-10-21)

- MTP verify-only steps on FA3 (gate 0 failures in 55K calls, captured in CUDA graphs): ours + MTP throughput
  +32% (98.5 -> 129.4 / 131.9 tok/s) and chat attainment 61 -> 72%. Still behind stock + MTP on FA3 (161 tok/s,
  77%).
- Kernel: co-run vs one stock FA3 call -4..-25% where the rows' and chunk's times differ, ~0 where they match. The
  best SM margin ranges from 16 to 80, so it must follow the step's composition.

## v1.25.2 — 32 H20s with disagg on FA3, 1x load (2026-10-21)

- Ours vs disagg 5 + 3 at 1x load, where disagg is not collapsed: agent throughput +56%, agent TTFT -84%. Disagg holds
  the lower ITL tail.

## v1.25.1 — Fair-stock single-node rep 1; plain decode on FA3 (2026-10-21)

- On FA3 (its Hopper default), stock beats ours: chat attainment 80% vs ours 73% without MTP, and 77% vs 61% with
  MTP. Rows on FA3 moved ours little. The earlier single-node table is superseded as a bar result.
- `apply_decode_fa3.py` (`DECODE_FA3=1`): plain decode steps on FA3, so every attention call matches stock and only
  the mixed step differs. Gate and A/B queued (`h200_decode_fa3_20261021.sh`); H20 sweep relaunched with disagg on FA3.

## v1.25.0 — Rows and verify on FA3; stock baseline moves to FA3; page-128 split bug (2026-10-21)

- Stock sglang's default attention backend on Hopper is FA3, but every arm ran with `--attention-backend flashinfer`
  forced. `ATTN_BACKEND=fa3` restores the default, and serving reruns against it are running. Against FA3 stock, our
  sequential split gains nothing (-8..+9%); the co-run gains 4-31% except with 48 rows beside a short chunk.
- `apply_split_rows_fa3.py` (`SPLIT_ROWS_FA3=1`): split decode / verify rows on FA3 instead of FA2 BatchPrefill
  (48 rows: 0.9 vs ~8 ms per layer at any KV layout). Serving gate: 0 failures in 12.5K split calls, worst 0.0167.
  `apply_verify_fa3.py` (`VERIFY_FA3=1`): MTP verify-only steps on FA3, CUDA-graph safe; gate queued.
- Fix: the split's FA3 page table was per token, which is wrong at page_size > 1 (rel err 12.6 at page 128). The
  page-128 serving result for ours is invalid.

## v1.24.4 — Qwen3-8B single node; Qwen2.5-7B multi-node MTP (2026-10-21)

- Qwen3-8B half load: ours no-MTP throughput +16%, chat TTFT -86%, TPOT -27% vs prefill-prioritized; its EAGLE3 head
  accepts 1.8 and MTP is a loss for every system. Qwen2.5-7B MTP at 16 / 32 GPUs: disagg overloaded at every ratio.

## v1.24.3 — Single-node table with chat template (2026-10-21)

- Qwen2.5-7B, real head (accept 2.02): ours + MTP chat attainment 62.7% vs 44.0% prefill-prioritized + MTP (+42%),
  agent ~10x, throughput +33%. Ours without the KV pool at 32 GPUs: 2092 vs 2153 tok/s (the pool is not the gain).

## v1.24.2 — Fusion microbench, 2-mode / 4-mode, three models (2026-10-21)

- Split alone 0-6% faster than unfused FA2 per mixed step; co-run 8-15% with 16 rows, 3-6% with 48. The 34% serving
  ablation gap is left to decompose.

## v1.24.1 — Chat template knob; 32-GPU sweep and chat-templated single-node rerun launched (2026-10-21)

- `mn_bench.sh` `CHAT_TEMPLATE=1` (ShareGPT chat through the model's chat template); `mn_scale32_20261021.sh`,
  `h200_chattemplate_20261021.sh`. Larger-model runs move to GLM-4.5-Air (less KV per token, built-in MTP layer).

## v1.24.0 — 32-GPU P:D ratio sweep setup (2026-10-21)

- User authorized up to 8 x 4 H20. `mn_run.py` `disagg_p<P>` arms (any prefill:decode pod split); 32-GPU Qwen-7B
  provisioning (`mn_provision_32g_20261021.sh`); 70B per-request results archived (`h20_70b_requests_20261021.npz`).

## v1.23.8 — One-GPU chat 8 req/s accepted as a limitation (2026-10-21)

- +28% chat SLO attainment (63.0% vs 49.3%) recorded as the one point below the +30% bar, per the user.

## v1.23.7 — FA3 page-size kernel bench and step traces (2026-10-21)

- FA3 is 9-17% faster at page 128 than page 1 on deep-context chunks (paged TMA vs per-token gather), but decode
  steps slow 5% at page 128 in serving; total GPU time equal, page 1 stays.

## v1.23.6 — EDF base deadline at the chat SLO: no change (2026-10-21)

- TTFT_SLO_MS=500 with rows + admission: chat attainment 62.6% (two repeats), TTFT misses unchanged at 33%.

## v1.23.5 — Slack chunk cap on top of rows + admission: negative (2026-10-21)

- Chat TPOT misses 30% -> 4%, but prefill starves (chat attainment 43%); a short-prompt floor does not help.

## v1.23.4 — Rows + admission without EDF: negative (2026-10-21)

- FCFS + rows + SLO admission: chat attainment 22%, agent 8% (head-of-line blocking behind agent prefills).

## v1.23.3 — Llama-3.3-70B, 4 replicas: ours vs PD disagg (2026-10-21)

- 4 x 4 H20, TP4, MTP at acceptance 2.1, fan-out + chat, two repeats: chat attainment 77-89% vs 16-18% (2 s + 50 ms),
  agent goodput 59-69 vs 4-5 tok/s, agent TTFT 2.5 vs 41-43 s.
- `mn_node.sh` `WORKER_UP_S` (startup window); post-provisioning installs libssl-dev and sets the draft to bf16.

## v1.23.2 — Llama-3.3-70B on 4 x 4 H20: provisioned, gate PASS (2026-10-21)

- Job f3b4627e0864b06f provisioned from the archived serving tree (manifest verified on all pods); upstream from a
  `git archive` because GitHub resets the pods' clones. TP4 output gate PASS (ours 41/96 vs noise floor 42/96 exact).
- The trained EAGLE3 head accepts 1.3 tokens per verify on raw prompts; the 70B A/B fixes acceptance at 2.1.
- `mn_run.py` calibrate passes TP / GPUS_PER_POD / DRAFT_DIR / SIM_ACC; `POOL_KV_TOKENS` knob for the router.

## v1.23.1 — H20 results archived; 70B job provisioning (2026-10-20)

- `tasks/pd_colocation/h20_requests_20261020.npz`: per-request TTFT / TPOT / output length / start of all 413 streams
  run on H20 job bb854a10b8954223 (`tasks/perf_model/pack_requests.py`), kept before the job is replaced.
- `tree_20261020.tgz` + `tree_manifest_20261020.txt`: that job's serving tree (32 files vs upstream 3fef129);
  `mn_provision_70b_20261021.sh` rebuilds it and verifies the manifest, then fetches Llama-3.3-70B + its EAGLE3 head.
- `mn_node.sh`: any TP size (one TP worker per pod, one RDMA NIC per rank).

## v1.23.0 — Fault tolerance; acceptance 3.0; page 128 (2026-10-20)

- `router.py`: a worker that refuses or breaks a connection is marked dead and later requests are rerouted
  (`test_router_dead_worker.py`); `mn_run.py` `FAULT_AT` / `FAULT_POD` fault injection, `DISAGG_HEALTH` ejection for the
  stock router, and a bench-launch check (a lost launch call had left an arm idle for its timeout).
- 8 H20s, kill 2 of 8 GPUs at 300 s: ours agent attainment 96 -> 98%, chat 85% (91% unfaulted); disagg chat 9-15%
  (26% unfaulted).
- H200 acceptance 3.0 (simulated): chat attainment +26% over prefill-prioritized + MTP, agent 2.5x. Page 128: negative.

## v1.22.4 — MTP ablation, CUDA graphs off, chat 6 req/s (2026-10-20)

- H200, two repeats: without the split kernel the MTP stack loses 34% agent throughput (agent / chat attainment 11% /
  37%); without EDF agent attainment falls to 16-20% (from 47%); the deadline chunk is neutral at the standard mix;
  CUDA graphs off costs ours ~20%, stock MTP ~4%.
- Chat 6 req/s rerun after the one-chunked-request fix: chat attainment 69.3% vs 55.7% prefill-prioritized + MTP,
  chat TTFT -69%, TPOT -30%, agent attainment 2.6x, throughput +29%.

## v1.22.3 — Attainment vs chat rate; one-chunked-request fix on the H200 tree (2026-10-20)

- Four repeats of deadline-ordered rows + SLO admission at chat 8 req/s: chat attainment 63.0% vs 49.3% (+28%).
  Attainment-vs-rate sweep: chat 2 / 4 req/s +14% / +16% attainment, chat TTFT -76% / -75%, TPOT -51% / -39%,
  agent throughput +30% / +29% over prefill-prioritized stock + MTP.
- `apply_edf_one_chunked_req.py`: at most one unfinished chunked request under every policy, on both admission paths
  (the re-forked H200 tree lacked the 10-17 `edf_one_chunked_req.patch`; the ignore-EOS path was unguarded everywhere).

## v1.22.2 — 14B with simulated MTP; semi-PD retry (2026-10-20)

- 8 H20s, 14B half load, MTP with a random-weight head and acceptance fixed at 2.1: ours chat attainment 73-77% vs
  disagg 0.1-1.1%, agent throughput 117-121 vs 15 tok/s, agent TTFT 3.2-3.5 s vs 106-108 s.
- semi-PD: the pip CUDA 12.4 wheels give headers and libs but no nvcc driver; still needs a CUDA 12 toolkit install.

## v1.22.1 — SLO-checked admission for deadline-ordered rows (2026-10-20)

- `apply_decode_rows_admit.py`: admit a request only while the resident rows' mean-TPOT SLO load (measured step
  interval / (SLO x tokens per verify)) fits util x K row slots. Without it (cap 96) chat TPOT reached 57.6 ms; with it,
  rep 1 chat attainment 65.0% vs 49.1% for prefill-prioritized stock + MTP (+32%), agent attainment 46.7% vs 8.9%.

## v1.22.0 — Deadline-ordered decode rows; simulated-MTP 14B; MTP ablation arms (2026-10-20)

- `apply_decode_rows_edf.py`: with MTP, up to `--max-running-requests` requests stay resident while each step runs the
  K (= speculative graph batch) rows whose next token is due soonest; the rest sit out with KV and draft state kept.
  Output gate PASS against its same-batch-size null (41 vs 43 of 160 exact, `gate_decode_rows_20261020.sh`).
- `make_sim_eagle3_head.py` + `DRAFT_DIR` / `SIM_ACC` harness knobs: random-weight EAGLE3 head of the target's shape
  with simulated acceptance (`SGLANG_SIMULATE_ACC_LEN`), used for Qwen2.5-14B MTP on H20.
- `mn_h200_nospec_20261018.sh`: MTP ablation arms (`ours_mtp_unfused`, `_noedf`, `_nodc`); runners
  `h200_decode_rows_20261020.sh`, `h200_mtp_ablation_20261020.sh`, `mn_14b_mtp_8gpu_20261020.sh`.

## v1.21.16 — 14B at half load (2026-10-20)

- 8 H20s, agent C=16 + chat 4 req/s, two repeats: ours chat attainment 80% (0.5 s + 20 ms) vs disagg 9-17%, agent
  goodput 133-143 vs 0-1 tok/s, agent TTFT 2.2 s vs 110-118 s.

## v1.21.15 — Outline: eval models re-sized for TP4 / TP8 nodes (2026-10-20)

- `tasks/paper_outline.md` / `.html` 6.1: one TP8 / TP4 instance per node; Llama-3.3-70B-Instruct and GLM-4.5-Air
  (replacing Qwen3-30B-A3B), with KV capacity per node for 8 / 4 x H200 / H100; 4 replicas; context forced to 262K
  via `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` for the long mix.

## v1.21.14 — Outline: proposed eval models, HTML version (2026-10-20)

- `tasks/paper_outline.md` 6.1: Llama-3.3-70B-Instruct (dense) and Qwen3-30B-A3B-Instruct-2507 (MoE) for 4 x 8
  H200 / H100, KV capacity per TP layout, MTP simulated via `SGLANG_SIMULATE_ACC_LEN`.
- `tasks/paper_outline.html`: the outline as a standalone page for review.

## v1.21.13 — Paper outline rewritten (2026-10-20)

- `tasks/paper_outline.md`: 10/20 outline in the 11/21 format plus drafted sections (abstract through conclusion),
  four main contributions in the design section, main workloads highlighted in the evaluation, results to date.

## v1.21.12 — ## Baselines in the main task doc (2026-10-20)

- `debug_kernel_server_perf_4_mode.md` gains a ## Baselines section (settings, baselines, priority, current status),
  pointing at the full tables in `realistic_workload_analysis.md` and `pd_colocation_vs_disagg.md`.

## v1.21.11 — Second model: Qwen2.5-14B (2026-10-20)

- 8 H20s, fan-out, two repeats: ours agent 186-216 tok/s, chat attainment 45-51% (0.5 s + 20 ms), agent goodput
  102-131 tok/s; PD disagg agent TTFT ~290 s, agent goodput 0, chat 11-13%.

## v1.21.10 — MTP cap 64 also worse for every arm (2026-10-20)

- Stock MTP 36.6 tok/s, prefill-prioritized MTP 37.1, ours MTP 60.4 (vs 102 at 48); 48 is each system's best cap.

## v1.21.9 — Second model (Qwen2.5-14B) and MTP cap sweep (2026-10-20)

- `MODEL_DIR` knob (worker, bench tokenizer, fan-out driver); calibration writes a per-model decode grid (`CAL_GRID`)
  and log; `mn_fanout_8gpu_20261018.sh` takes `RA_LOG`. `mn_14b_8gpu_20261020.sh`: 14B calibration, then ours vs PD
  disagg on the fan-out, two repeats (downloads running).
- `h200_mtp_capsweep_20261020.sh`: ours + MTP at caps 64 / 80, both stock MTP baselines at 64, each at its best.

## v1.21.8 — MTP cap 128 is worse for every arm (2026-10-20)

- Stock MTP 32.8 tok/s / TTFT 32 s, prefill-prioritized MTP 31.6 / 41 s, ours MTP 57.0 / 9.6 s (TPOT 40 ms vs 18.6 at 48):
  bigger verify batches slow every step. The cap-48 comparison stands.

## v1.21.7 — Mixture shift: ours absorbs doubled agent load (2026-10-20)

- 8 H20s, two repeats: ours 97-100% agent attainment through the shift, agent goodput 718-786 tok/s (x2), chat ~84%;
  PD disagg 21-31% agent, 18-22% chat, TTFT 26-49 s.

## v1.21.6 — MTP arms were capped at 48 running requests (2026-10-20)

- sglang's speculative default `max_running_requests = 48` capped every MTP arm (ours and both stocks); 36-68% of their
  chats missed on TTFT from the queue it builds. All MTP arms rerun with `--max-running-requests 128
  --cuda-graph-max-bs 128` (`h200_mtp_cap128_20261020.sh`).

## v1.21.5 — Chunk trade: negative (2026-10-20)

- Deadline chunk 512 / 1024 leaves chat attainment unchanged (69.3% no MTP, 57.1% MTP) and costs agent attainment
  (80%, 29%); the single-node chat gap to prefill-prioritized stock is structural at this load.

## v1.21.4 — Single-node slack decode-first: negative; chunk trade (2026-10-20)

- Per-row slack on one H200 starves prefill (agent 50 tok/s, TTFT 30 s; caps mostly 0 because some chat row is
  always behind). `ours_mtp_dc` arm (env chunks); `h200_chunk_trade_20261020.sh` (deadline chunk 512 / 256) running.

## v1.21.3 — Mixture-shift harness; single-node slack arms (2026-10-20)

- `fanout_agents.py --start-delay`; `mn_bench.sh FO_SPIKE=delay:conc:duration` adds a second fan-out driver on the same
  task prefixes (own seed); both drivers' PIDs waited on. `spike_windows.py` scores the second driver.
  `mn_shift_8gpu_20261020.sh` (agent load x2 for 240 s, ours vs disagg) running.
- `h200_slack_20261020.sh`: per-row slack decode-first single-node (chat 20 ms / agents 80 ms), targeting chat
  attainment vs prefill-prioritized stock; H200 tree rebuilt with the current patch.

## v1.21.2 — Load spike: ours absorbs a 3x chat spike (2026-10-20)

- 8 H20s, two repeats: agent attainment 99-100% through the spike, spike-stream chat 88-91% (TTFT 0.13 s); PD disagg
  36-57% agent, 19-22% chat, 14-31 s TTFT in every window.

## v1.21.1 — 5e load spike harness (2026-10-20)

- `mn_bench.sh CHAT_SPIKE=delay:rate:n`: a second chat stream on top of the base load after `delay` s (own seed, own
  output). `spike_windows.py`: agent attainment / goodput pre / spike / first 60 s / post; base and spike chat
  attainment. `mn_spike_8gpu_20261020.sh` (ours vs PD disagg, chat 8 -> 24 req/s for 240 s) running.

## v1.21.0 — Six-repeat single-node result (2026-10-20)

- vs stock mixed chunk: MTP pair clears the attainment bar (chat +72%, agent 15x, +55% throughput); no MTP chat +29.7%.
  vs prefill-prioritized: chat attainment +16.5% (no MTP) / +20% (MTP), short of +30%. Stock slow mode in 5/6 repeats.
  `bench_reps_summary.py`.

## v1.20.99 — Decode-first on H20 fan-out: every variant lowers attainment (2026-10-20)

- Per-row slack: 142.9 tok/s, TTFT 16.3 s, chat TTFT p99 214 s -- the measured decode step (~20 ms) is at the chat SLO,
  so rows bank no slack and prefill stops. Per-step measured 306-382 tok/s. The uncontrolled SLO-checked router
  (551 tok/s goodput, chat attainment 91%) remains the multi-node configuration vs disagg (92, 26%).

## v1.20.98 — Per-row slack decode-first budget (2026-10-20)

- Per-step decode-first lowered H20 fan-out attainment (chat 91% -> 65%, agent goodput 551 -> 262 tok/s): the SLO is
  a request's mean TPOT and was not binding. `apply_edf_itl_slo_chunk.py` `:slack` mode budgets per running row
  (x <= s*(o + acc) - E), per-class SLO via a `cls<tokens>_<ms>` token; unit-checked. Arms queued.

## v1.20.97 — Pooled MTP step-model fit (2026-10-20)

- Closed-form 4-mode linear model: ours 12-13% vs stock MTP 24-27% on prefill-bearing steps; GBT ~3% on both.
  Outline C3 narrowed to "admits a simple model at half the error".

## v1.20.96 — Measured decode-step time for decode-first control (2026-10-20)

- The grid-fit decode model over-predicted H20 fan-out decode steps ~2-3x (shared-prefix rows; 4 rows at 68K
  depth priced at the full 20 ms budget vs ~12 ms measured ITL), so the fixed rule capped prefill to ~10 tokens/step.
- `apply_edf_itl_slo_chunk.py`: `ITL_SLO=...:measured` uses the node's measured decode-step EWMA (sglang's
  `decode_moment_totals` deltas) as the decode base. `apply_load_stat_4mode.py`: `decode_step_ms` in LoadStat.
  `router.py`: decode budget from the reported step; own-ITL anchored to it (model gives only the growth ratio).
  CPU test extended. Arms `mn_slo_measured_8gpu_20261020.sh` queued after the A/B.

## v1.20.95 — Decode-first rule: prefill share under mixed chunk (2026-10-20)

- Root cause of both SLO-control collapses (35-41K errors): under mixed chunk `PrefillAdder` charges one token per
  running decode row against the chunk, so a ~100-token prefill cap beside 50+ decode rows left a zero / negative
  prefill share, the in-progress chunked request was scheduled with a zero-token extend, and flashinfer's planner
  aborted every worker. The rule now returns cap + decode rows (cap compared with chunk - rows; no prefill below one
  page). Both earlier SLO-control runs are void; rep 1 is rerun after the A/B (`mn_slo_control_r1b_20261020.sh`).

## v1.20.94 — 4-mode step-model search: per-step predictability claim not supported (2026-10-20)

- Serving traces, best model per schedule (MLP / GBT): every schedule ~3-8% on mixed / prefill steps; ours beats
  pdmux (5.9% vs 7.5% light load) but not stock mixed chunk (4.2%). The 10-18 gap was pdmux's linear form. Outline C3
  demoted to an enabler of SLO control. `fit_4mode_step.py` gains pdmux sg / lay features.

## v1.20.93 — Decode-first rule: skip prefill below one page; direct load publisher (2026-10-20)

- `apply_edf_itl_slo_chunk.py`: when less than one KV page fits beside the running decodes, no prefill that step
  (returns 0, the scheduler's existing skip). The 1-token floor rounded to a zero-token extend and crashed every H20
  worker in flashinfer's planner (`kv_len[0]-1 should be non-negative`), collapsing the first SLO-control arm
  (41 127 errors) -- that run is void. A queued rerun script was launched with a malformed wait loop and two rogue
  copies ran concurrently with disagg rep 1; all three overlapping runs are void, the cluster was cleared, and the A/B
  was relaunched with only the valid rep-1 reference kept (`SKIP_REF1=1`); `mn_slo_control_r1b_20261020.sh` unused.
- `apply_load_stat_4mode.py`: `SGLANG_LOAD_PUB_DIRECT` binds the LoadStat socket directly (upstream binds it only
  beside a KV-events publisher and refuses `--load-publish-endpoint` without `--kv-events-config`).
- `apply_steptrace_pdmux.py` applied on H200 (the first pdmux trace emitted no lines); pdmux traces queued.

## v1.20.92 — Node-reported load and multi-node SLO control (2026-10-20)

- `apply_load_stat_4mode.py`: append decode rows, decode KV, acceptance and the earliest-deadline prefill queue
  (ms to deadline, uncached, prefix) to sglang's router-facing `LoadStat` (ZMQ PUB, change-triggered + 1 s heartbeat).
- `router.py --load-endpoints`: SUB thread keeps each node's latest report; decode state, own-ITL rows and the
  earlier-deadline drain come from it when fresh (< `--load-max-age-s`), else from router bookkeeping. CPU test
  `test_router_load_sub.py` passes.
- `apply_edf_itl_slo_chunk.py`: 7-field form with a live decode base c0 + c_bs*rows + c_kv*KV (H20 fit: decode grid
  4.1% MAPE, cold prefill 2.4%). `LOAD_PUB` knob (bootstrap port). A/B `mn_slo_control_8gpu_20261020.sh` running.

## v1.20.91 — Correction: stock baseline is bimodal (2026-10-19)

- Stock rep 3 matches the 10-18 numbers; reps 1-2 were a slow mode (start-up queueing behind the first agent
  prefills). Vs rep 3, ours without MTP misses agent TTFT (+17%). Doc and outline corrected; repeats 4-6 queued.

## v1.20.90 — Baselines delineated by SLO control; 4-mode step-model fitter (2026-10-19)

- Baselines table: single node without SLO control vs prefill-prioritized and mixed chunk (required), with SLO control
  vs pdmux (nice to have); multi-node with SLO control vs PD disagg / semi-PD (the main claim). `MIXED_CHUNK=0` knob,
  `stock_pp` / `stock_mtp_pp` arms.
- `fit_4mode_step.py`: lin5 (pdmux form), lin7, lin4mode (draft / verify terms), overlap max+beta*min, MLP, GBT;
  held-out MAPE per step kind, pooled across speculative configs. Synthetic check reaches the 2.5% noise floor.
- `h200_pp_and_trace_20261019.sh` queued: prefill-prioritized arms, STEPTRACE runs at two loads and two MTP configs.

## v1.20.89 — Router: prefill time under the node's decode load (2026-10-19)

- `router.py --slo-chunk itl_ms:a:b`: when workers serve running decodes first (`apply_edf_itl_slo_chunk.py`), a
  node's prefill rate is what the ITL budget leaves after its decode step; TTFT and queue drain are walked chunk by
  chunk under that load, and a node whose decodes fill the budget predicts infinite TTFT, so the SLO check routes away.

## v1.20.88 — SLO-sized prefill chunk (2026-10-19)

- `apply_edf_itl_slo_chunk.py`: largest chunk keeping running decodes within their ITL SLO (step model base + a*c +
  b*c*(prefix + c/2) <= slo * tokens_per_step), no exception (user): no late-prefill override, no floor above the
  budget; `ITL_SLO` knob.
  H200 A/B `h200_itl_slo_chunk_20261019.sh` running.

## v1.20.87 — H200 per-tenant rerun of the standard mix (2026-10-19)

- Two repeats on `dadf94f0e1da7238`: ours no MTP vs stock +44% / -35% / -32%, chat -94% / -45%; ours + MTP vs stock
  MTP +60.5% / -38.9% / -32.9%, chat -77.8% / -28.1%. Per tenant: chat attainment 1.3x / 1.8-1.9x, agent goodput
  2.3-2.8x / ~16x. Per-request records committed. Stock slower than on the 10-18 pod: open.

## v1.20.86 — Point 4 measured as proposed: pull-source choice neutral (2026-10-19)

- 8 H20s, one NIC, 45 s rotation: off 531 tok/s agent goodput, first 492, random 477, least_latency 487. Per-page
  latency equal across sources (0.011-0.019 ms); copies cost 7-10%; one source per page stays the default.

## v1.20.85 — KV pull-source choice among per-worker copies (2026-10-19)

- `apply_kv_copy_source.py`: pulled prefixes are re-exported as per-worker copies on the puller's segment; a pull
  picks its source by `SGLANG_KV_COPY_POLICY` (off / first / random / least_latency); own copy read first. `KV_COPY`
  knob in `mn_node.sh` / `mn_run.py`. CPU fake-store test passes. A/B: `mn_kv_copy_source_20261019.sh`.
- `h200_per_tenant_20261019.sh`, `bench_per_tenant.py`: single-node per-tenant rerun on H200 `dadf94f0e1da7238`.

## v1.20.84 — Paper outline; point-4 random-pull correction (2026-10-19)

- `tasks/paper_outline.md`: the 11/21 draft's sections mapped to measured results, contributions C1-C4, ranked gaps.
- Point 4 correction: `cache_rand` randomized the serving node among equal-hit holders, not the KV pull source;
  the "randomized pull refuted" conclusions are retracted and the items unchecked.

## v1.20.83 — Per-tenant fan-out scoring (2026-10-19)

- `fanout_per_tenant.py`: chat SLO attainment 91-96% (ours) vs 25-33% (disagg) heavy, 97-99% vs 43-54% light;
  agent goodput 3.0-7.1x. Mean chat ITL is worse, per-request chat TPOT stays under 20 ms for >= 90%.

## v1.20.82 — FA3 prefill-chunk splits: negative, no lever (2026-10-19)

- Forced num_splits / pack_gqa within +-2% of auto at 4K-120K depth; the mixed step is attention-FLOP-bound
  (~540 TFLOP/s), so the standard-mix chat ITL is not limited by FA3 launch shape.

## v1.20.81 — Router time-based draining: negative, off by default (2026-10-19)

- 25% of workers drained per 20 / 8 s window: agent 521 / 503 tok/s (vs 555), chat ITL 12.1 ms (vs 10.8). Draining
  concentrates prefill on the open workers.

## v1.20.80 — Router soft node roles: negative, off by default (2026-10-19)

- Long-prefill pool 6 / 4: chat ITL 9.8-9.9 ms (vs 10.8), agents -9..-13% throughput. Routing cannot make chat decode
  prefill-free at this load; disagg's decode pods are by construction.

## v1.20.79 — Router per-class (chat) ITL SLO: negative, off by default (2026-10-19)

- 12 / 9 ms chat SLO: agent 505 / 461 tok/s (vs 555), chat ITL unchanged (10.5 ms). No prefill-free node exists at this
  load, so the check only moves agents off their cached prefixes.

## v1.20.78 — Point 4: random holder choice loses on the narrowed fabric with fast-rotating fan-out (2026-10-19)

- 1 NIC, 45 s rotation: load-ordered 545 tok/s / TPOT 13.3 ms / goodput 531 vs random 526 / 15.3 / 506; chat TTFT 157
  vs 364 ms. Links are not the constraint at <= 16 H20s (load-ordered within 2% of the full-fabric run).

## v1.20.77 — Router: load-adaptive ITL SLO (K = 2.5) tracks the hand-tuned optimum at both loads (2026-10-19)

- `router.py --itl-slo-auto K`: SLO = K x observed per-token decode time. K = 2.5: heavy 555 tok/s / TPOT 12.1 ms /
  goodput 545 (fixed best ~560 / 12.0 / ~552); light 298 tok/s / 9.6 ms (fixed best 8.9-9.2 ms).

## v1.20.76 — PD 2b: 30 ms ITL SLO repeats; the best SLO depends on load (2026-10-19)

- Heavy rep 2: 569 tok/s / 0.64 s / TPOT 12.0 ms, goodput 562 tok/s. Light: 30 ms costs TPOT vs 15 ms (10.0 vs 8.9-9.2
  ms). Settled vs PD disagg: goodput 5.0-7.1x (heavy), 3.0-3.5x (light).

## v1.20.75 — PD 2b: router ITL SLO plateau at 20-30 ms (2026-10-19)

- Heavy load: ~560 agent tok/s, TTFT 0.66-0.73 s, TPOT 12.0 ms, goodput ~550 tok/s at 20 / 25 / 30 ms; 30 ms is best
  for chat (TTFT 254 ms, ITL 9.5 ms, p99 42 ms) and becomes the default.

## v1.20.74 — PD 2b: router ITL SLO 20 ms beats 15 ms; agent goodput 5.0-7.0x PD disagg (2026-10-19)

- Heavy natural-text load: 562 tok/s, TTFT 0.73 s, TPOT 12.0 ms; goodput 552 vs disagg 79-111 tok/s. Light-load
  repeat 2 reproduces repeat 1.

## v1.20.73 — PD 2b light-load repeat: goodput 3.0-3.5x disagg; replicas follow the hot set (2026-10-19)

- Ours (ITL SLO 15 ms) vs disagg: agent 294 vs 124 tok/s, TTFT 0.69 vs 10.6 s, TPOT p50 6.49 vs 6.61 ms (mean 9.2 vs
  6.1). Hot prefixes replicate to 3-7 of 8 workers and follow popularity; 76 of 2106 routings made a new copy.

## v1.20.72 — PD 2b re-prefill: disagg recomputes ~3x the share of its input that colocation does (2026-10-19)

- Heavy natural-text load: ours 3.4-3.5% of input tokens recomputed (2.5% with MTP), disagg 10.2-10.4% (16.7% with
  MTP). `fanout_reprefill.py`.

## v1.20.71 — PD 2b heavy-load repeat: agent goodput 4.4-6.2x PD disagg over two repeats (2026-10-19)

- Rep 2: ours 503 tok/s / 0.89 s / TPOT 16.9 ms, goodput 490 / 483 tok/s; disagg 170 / 17.9 s / 5.7 ms, 79 / 100.

## v1.20.70 — PD 2b: router ITL SLO with full router args; agent goodput 4.4-5.8x PD disagg (2026-10-19)

- Fixed the fan-out script dropping router args for no-MTP arms. ITL SLO 15 ms: agent 505 tok/s, TTFT 0.92 s, TPOT
  16.0 ms; goodput 489-495 vs disagg 86-111 tok/s. Agent TPOT still 3x disagg's.

## v1.20.69 — PD 2b heavy load on natural text; agent SLO goodput 3.2-5.3x PD disagg (2026-10-18)

- 8 H20s, 32 programs + chat 8 req/s: ours (no MTP) vs disagg (no spec) agent +162% / TTFT -93% / TPOT 4x worse,
  chat TTFT -99%. Agent SLO goodput (10 s TTFT + 80 ms TPOT): 457 vs 86 tok/s (98% vs 48% attainment).

## v1.20.68 — PD 2b: agent TPOT gap vs disagg is mixed-step interference, not the decode kernel (2026-10-18)

- Stock kernel with the persistent path on / off gives agent TPOT 12.9 / 12.1 ms (ours 10.0, disagg 6.3). The earlier
  "decode step is slower" reading is corrected in place: chat prefill load was constant across those points.

## v1.20.67 — PD 2b on natural text; per-token TPOT replaces the chunk-gap ITL for MTP arms (2026-10-18)

- Fan-out driver: `--text-source` (ShareGPT slices); the chunk-gap "ITL" overstated MTP arms 2-3x, TPOT recomputed.
- 8 H20s, light load: ours + MTP vs disagg + MTP agent +137% / -92% / TPOT -16%, chat -99% / -34%; ours without MTP vs
  disagg without: +117% / -93% / TPOT 2x worse (decode isolation at half the throughput).

## v1.20.66 — PD 2b fan-out at a lighter load: TTFT / throughput win, agent ITL loses to disagg (2026-10-18)

- 16 programs + chat 4 req/s on 8 H20s: ours vs PD disagg (no spec) agent +130% / -92% TTFT, chat TTFT -99%, but agent
  ITL 31.6 vs 6.1 ms (disagg's isolated decode pods). Disagg + MTP: agent ITL 23 ms, worse than without.

## v1.20.65 — Long mix, no-MTP arm, three matched repeats: on the bar's edge (2026-10-18)

- Slope 250: agent TTFT -30.9% / ITL -28.4%; slope 400: -29.2% / -30.2% vs stock without spec (means of 3). Chat
  -78..-85% / -38..-40%, agent throughput +39..+41%.

## v1.20.64 — PD 2b fan-out workload on 8 H20s: colocation + replication vs PD disagg (2026-10-18)

- Ours vs PD disagg (stock MTP): agent throughput 11.5x, TTFT -98.5%, ITL -68%; chat TTFT -99.8%, ITL -88%. Ours vs
  affinity-only colocation: agent +25% / -20% (p99 -66%) / -45%. Single repeat. `mn_fanout_8gpu_20261018.sh`.

## v1.20.63 — Long-context mix vs all single-node baselines: MTP arm clears all six vs stock + MTP again (2026-10-18)

- 32-250K TraceLab + chat 2 req/s: ours + MTP vs stock + MTP agent +54% / -34% / -49%, chat +23% / -90% / -42%
  (second repeat); ours without MTP vs stock without spec agent +46% / -16% / -36%, chat +18% / -95% / -45%.

## v1.20.62 — No-MTP arm: agent-TTFT floor and deadline-slope x chunk frontier (2026-10-18)

- Unloaded agent TTFT 1.30 s (stock) / 1.20 s (ours): loaded TTFT is mostly queueing. Flatter EDF slopes reach agent
  TTFT -24.3% vs stock (slope 250, 2048/8192) but agent ITL falls to -19.8%; best ITL -31.2% costs TTFT +15.6%.

## v1.20.61 — Output gate PASS for the FA2-overlapped arms; kernel-numerics null case (2026-10-18)

- No-MTP overlapped split diverges from stock FlashInfer exactly where stock FA3 does (token 11); MTP arm matches the
  chunk-size null case. `gate_h200_ovl_20261018.sh`.

## v1.20.60 — No-MTP chunk sweep with FA2 overlap; overlap neutral on the MTP arm (2026-10-18)

- vs stock mixed chunk without spec: 2048/8192 agent +24.8% / -10.4% / -24.0%, chat -87.5% / -29.6%; 1024/2048
  agent +25.7% / +15.6% / -31.2%, chat -85.6% / -36.0%. Agent TTFT and ITL trade along one frontier.

## v1.20.59 — Single-node A/B against the bar's baselines; FA2 overlap measured for the first time (2026-10-18)

- Ours without MTP (FA2 overlapped) vs stock mixed chunk without spec: agent +25.7% / +15.6% / -31.2%, chat -85.6% /
  -36.0%; agent TTFT the one miss. Overlap vs sequential split: +10% agent tput, -12% / -16% agent / chat ITL.
- Ours + MTP vs stock + MTP: agent +39.5% / -26.9% / -28.8%, chat -68.6% / -20.9%. `SPLIT_OVERLAP` knob.

## v1.20.58 — 32-250K mix at chat 2 req/s: 4-mode MTP clears all six vs stock MTP (2026-10-18)

- Ours agent +53% / -37.5% / -44%, chat +23% / -86% / -45.5%; with 2x deadline above 200K agent +55% / -37.8% / -46%,
  chat +23% / -89% / -48%. Single repeat; baseline A/B (stock mixed chunk, stock + MTP, pdmux) queued.

## v1.20.57 — 2x deadline above 200K on a 32-250K mix at chat 4 req/s: five of six metrics clear (2026-10-18)

- vs stock MTP: agent +41% / -39.1% / -33%, chat +35% / -80.4% / -26.7% (chat ITL misses); SLO attainment agent
  30.6% vs 11.1%, chat 50.7% vs 0.2%. Single repeat; stock is past capacity at this rate.

## v1.20.56 — Long-prefill deadline multiplier (2x above 200K uncached tokens) (2026-10-18)

- `apply_edf_long_deadline.py` + `LONG_DEADLINE`; harness knobs `CTX_LEN`, `YARN_FACTOR`, overridable workload knobs.
- 32-250K TraceLab + chat (over capacity): mean agent TTFT unchanged, < 128K rounds 25-37% faster, > 195K slower.

## v1.20.55 — Reference-list format; JITServe added; mixed-SLO setting item (2026-10-18)

- CLAUDE.md: every reference gets summary / relation / what-we-learn, the last only where it strengthens our core
  contributions. The 4-mode doc's reference list rewritten to that format; JITServe (NSDI '26) added.
- S: mixed-SLO setting item -- service goodput by request type, tiered TTFT SLO (2x above 200K uncached tokens).

## v1.20.54 — Stock MTP decodes without speculation under interleaved prefill (2026-10-18)

- Per-chunk probe: stock MTP advances ~40% of its decode steps one token at a time (no verify) while chunked prefill
  runs (1.66 tokens per step vs ours 2.08), which is why its per-request tokens-per-verify read 3.0. Item closed.

## v1.20.53 — Per-request acceptance halves TPOT prediction error on the 4-mode schedule (2026-10-18)

- `tpot_acc_probe.py`: TPOT = k / acc_i, MAPE 7.0% vs 12.7% with a constant acceptance (point 3).
- Stock MTP's per-request `spec_verify_ct` undercounts verifies under interleaved prefill (2.96 vs 2.13 unloaded);
  acceptance is equal on both schedules (2.09 vs 2.13), so the 4-mode step does not degrade drafting.

## v1.20.52 — S4 on the 4-mode fused schedule: mixed-step model 6x more accurate than stock MTP (2026-10-18)

- `fit_tpot_acc.py`: steps carrying a prefill chunk 7.8 / 8.5% MAPE (4-mode) vs 49.5 / 44.2% (stock MTP), two traces.
- `apply_steptrace_dacc.py`: per-step acceptance delta in STEPTRACE; too sparse (reporter batches its counters), so the
  acceptance-input test stays open. Bar rep 4: agent TTFT -28.1%, chat ITL -24.1%.

## v1.20.51 — FP8 linear A/B; the split now serves identity-scaled KV layers (2026-10-18)

- `apply_split_unit_kv_scale.py`: under `--quantization fp8` (BF16 KV) the split refused every layer for its unit KV
  scale and fell back silently; it now refuses only non-unit scales or an fp8 KV cache.
- Both arms FP8: agent +36% / -31.2% / -25.9%, chat -76.5% / -26.7% vs stock MTP + FP8 (exploratory, numerics change).

## v1.20.50 — Mixed-step profile; checkpointing every 8192 tokens does not shorten steps (2026-10-18)

- `prof/prof_mixed_20261018.py`, `prof/gpu_gaps.py`, `prof/sync_sites.py`: torch-profile mixed steps on the serving
  config, GPU idle gaps and the call sites behind them.
- `apply_checkpoint_every.py` + `CKPT_EVERY`: gate PASS; serving unchanged (mixed step 44.2 vs 44.0 ms).

## v1.20.49 — Steeper EDF deadline slope: chat TTFT ~130 ms, agent mean TTFT worse (2026-10-18)

- `TTFT_SCALE_MS` knob. 5x slope: agent median TTFT halves but p99 doubles, mean -21..-26% (vs -29%); chat TTFT -98.7%.
  2x slope: agent TTFT -26.2%, chat TTFT -86.8%; default slope stays best for the bar.

## v1.20.48 — Traced 4-mode MTP A/B: host overhead <= 4%, mixed-step model twice as accurate as stock (2026-10-18)

- Rep 3 of the best H200 setting: agent +45.5% / -29.0% / -35.7%, chat -71.5% / -23.8% vs stock MTP.
- `steptrace_util.py`: GPU idle after eager 4-mode mixed steps is 3.2 ms each (8.8 s of 297 s GPU time).
- Step model on mixed steps: MAPE 26.7% (4-mode) vs 53.5% (stock MTP). `STEP_TRACE` knob in mn_node.sh / mn_run.py.

## v1.20.47 — Step-time budget on the 4-mode chunk: negative (2026-10-18)

- `apply_edf_step_budget.py` + `STEP_BUDGET` knob (mn_node.sh / mn_run.py / mn_h200_fourmode): caps the chunk to a
  predicted mixed-step time over its prefix. Budget 50 ms on 1024/2048: agent TTFT -21.8% (vs -29.3%), chat ITL
  unchanged (-23.5%). A first 60 / 80 ms sweep did not see depth (stale `prefix_len`), recorded as such.

## v1.20.46 — H200 best 4-mode MTP setting repeated, with per-request error bars (2026-10-18)

- Rep 2: agent +42% / -28.9% / -32.0%, chat TTFT -71.8% / ITL -22.8% vs stock MTP; two-rep means agent +47% / -29.3% /
  -36%, chat -72.7% / -25%. Agent p1 tok/s/user rises 4-5x. `mn_ab_figure.py` takes `MN_CP`.

## v1.20.45 — H200 output gate PASS for the best 4-mode MTP serving config (2026-10-18)

- `gate_h200_fourmode_20261018.sh`: config arm diverges from stock MTP exactly where stock MTP at the same chunk size
  does (null case); split fired (>= 1501 calls), 0 tracebacks / fallbacks. Serving arms' counter: >= 66001 split calls.
- `srv_correctness.py`: `SRV_ARGS` (per-invocation server flags for any arm) and a mixed-composition step count.

## v1.20.44 — H200 4-mode MTP with a 2-step / 3-token draft: chat ITL gap unchanged (2026-10-18)

- Both arms on 2 draft steps / 3 verify tokens: agent +45% / -30.0% / -30.5%, chat TTFT -71% / ITL -23% vs stock MTP.
  Chat ITL does not move with draft depth in either arm; the 3-step / 4-token setting stays best.

## v1.20.43 — H200 4-mode MTP chunk pairs: chat ITL floor ~12 ms set by the MTP decode step (2026-10-18)

- Best 4-mode MTP setting (deadline chunk 1024/2048) vs stock MTP: agent +52% / -29.7% / -40.7%, chat TTFT -73.7% /
  ITL -26.7%. Chat ITL is 12.1-12.7 ms at every chunk setting -- the spec-decode step, not prefill, bounds it.

## v1.20.42 — H200 re-forked; 4-mode MTP schedule reaches the bar except chat ITL (2026-10-18)

- Re-forked H200 (`d6a80befdd4e4da5`) provisioned from the H20 recipe (`mn_provision_h200_20261018.sh`); harness gained
  MN_CP / GPUS_PER_POD / SM_FA3 and an explicit sglang-router metrics port.
- 4-mode MTP + EDF (uncached deadlines) + deadline-driven chunk (1024 / 4096) vs stock MTP, agent C=8 + chat 8 req/s:
  agent +48% / -29.9% / -31.0%, chat TTFT -71.7% / ITL -22.4%.

## v1.20.41 — 4-mode MTP: per-step deadline chunk choice stays on the same TTFT / ITL frontier (2026-10-18)

- `apply_edf_deadline_chunk.py`: the TTFT-driven per-step chunk degenerates to the large chunk under load (agent ITL
  -2..-11% vs stock + MTP). With chunk 512, the 4-mode MTP schedule clears 10 of 12 metrics; agent TTFT (-21%) misses.

## v1.20.40 — 4-mode MTP chunk sweep on 8 H20s: agent TTFT / ITL frontier, no single chunk clears both (2026-10-18)

- Chunk 512: everything clears vs stock + MTP except agent TTFT (-21%): agent +37% / -21% / -45%, chat +47% / -87% / -61%.

## v1.20.39 — 4-mode MTP schedule on 8 H20s: TTFT and throughput clear vs stock + MTP, ITL does not (2026-10-18)

- EAGLE3 MTP in the 4-mode fused step + EDF with uncached deadlines: chunk 2048 agent +20% / -33% / -5%, chat +23% /
  -84% / -19%; chunk 1024 agent +15% / -31% / -13%, chat +22% / -87% / -28%. Mixed steps (~100 ms on H20) bound ITL.
- The H200 trial 095f01a3e1585eea has ended; further H200 work needs a re-fork and a rebuild of the serving trees.

## v1.20.38 — PD point 4 scaling inside the cap: colocation keeps cache locality, disagg loses it (2026-10-18)

- 2 vs 4 H20 pods at fixed per-GPU load (`mn_scaling_20261018.sh`): prefill cache hit rises with nodes for colocation
  (0.34 -> 0.39) and halves for PD disaggregation (0.125 -> 0.055). Ours vs disagg agent throughput 5.1x / 2.6x, TTFT
  -90% / -80%. All arms clean on every pod.

## v1.20.37 — H20 config: repeat 2 holds (every value clears vs stock + MTP, smallest -39%); output gate PASS (2026-10-17)

## v1.20.36 — per-step SM sharing with TTFT deadlines + ITL budget: all 12 vs stock + MTP by >= 39% on 8 H20s (2026-10-17)

- `apply_pdmux_itl_budget.py`: a deadline-late prefill batch shares the GPU only while the decode-interval EWMA is under
  SGLANG_PDMUX_ITL_BUDGET_MS. At 24 ms, 16 / 32 req/s: agent +73% / -41% / -65%, +45% / -39% / -47%; chat +46% / -88%
  / -88%, +64% / -79% / -60% vs stock colocation + MTP. vs stock non-speculative colocation: ITL -36..-57%, TTFT worse.

## v1.20.35 — per-step SM sharing from EDF deadlines (PD item 5) (2026-10-17)

- `apply_pdmux_deadline_share.py`: share the whole GPU for a prefill batch only when the deadline-ordered queue would
  miss a TTFT deadline under the SM split. On 8 H20s it beats stock non-speculative colocation on every metric at 16
  and 32 req/s with no tuned threshold (agent +7..+14% / -13..-17% / -16..-24%), short of 30%.

## v1.20.34 — extra repeats: H200 C=8 agent -31.4% / -31.2% (4 reps), H20 32 req/s agent ITL -31.6% (3 reps) (2026-10-17)

## v1.20.33 — 8 H20s: pdmux + EDF + 48K backlog sharing clears all 12 bar means vs stock colocation + MTP (2026-10-17)

- Two repeats, both clean: 16 req/s agent +63% / -46% / -63%, chat +46% / -86% / -86%; 32 req/s agent +56% / -56% /
  -30.9% (per rep -32.8 / -29.0%), chat +67% / -86% / -51%. Against stock colocation without speculation the agent
  pair stays short of 30% (TTFT floor ~5.9 s on this cohort).

## v1.20.32 — H20 TP=2: lowers TTFT, raises ITL for every scheduler; pdmux + TP=2 hangs (2026-10-17)

- One TP=2 worker per pod: stock agent TTFT 7.18 -> 4.76 s but ITL 27.9 -> 45.1 ms; ours (EDF, uncached deadlines)
  at TP=2 is level with stock TP=2. pdmux + TP=2 hangs (GPUs at 100%, no progress). The best stock setting differs per
  metric, and our best H20 config sits between them on both.

## v1.20.31 — H20 agent-TTFT floor; SLO-check routing over pdmux workers negative (2026-10-17)

- Zero-queueing bound from the calibrated H20 prefill curve: mean agent TTFT >= ~5.9 s on this cohort; ours 5.91 s,
  stock non-speculative colocation 7.18 s, so its -30% target (5.03 s) is below the compute floor. Load-aware routing
  over the pdmux workers gives no agent TTFT gain over cache affinity.

## v1.20.30 — H20: backlog-triggered SM sharing in pdmux (2026-10-17)

- `apply_pdmux_backlog_share.py` (`SGLANG_PDMUX_SHARE_BACKLOG_TOKENS`): a split-prefill batch runs on the whole GPU
  (pdmux group 0) when the uncached prompt backlog reaches N tokens. On 8 H20s at N=32K it beats stock
  non-speculative colocation on every metric (16 req/s: agent +14% / -18% / -21%, chat TTFT -55% / ITL -24%; 32 req/s:
  agent +9% / -11% / -22%, chat +15% / -39% / -36%) but not by 30%. A 14-SM decode group (prefill-heavy manual
  divisions) fails cuBLAS during graph capture, so prefill is capped at 56 of 78 SMs under the SM split.

## v1.20.29 — pdmux + EDF on 8 H20s: ITL -41..-88%, but TTFT loses to stock non-speculative colocation (2026-10-17)

- Multi-node arm `coloc_cache_edf_pdmux` (pdmux on every colocated worker, uncached EDF deadlines, divisor 30,
  MEM_FRAC 0.70 -- 0.85 OOMed on unchunked agent prefills). vs stock coloc + MTP: everything clears except agent TTFT
  (-11% / -27%). vs stock coloc without speculation: ITL -41..-61%, TTFT 1.9-3.2x worse, throughput -1..-13% -- on H20
  prefill compute is the bottleneck and pdmux's decode SM share comes out of it.

## v1.20.28 — EDF step model on the final schedule: decode ITL predictable to 9% vs 35-117% unfused (2026-10-17)

- STEPTRACE under PD multiplexing (`apply_steptrace_pdmux.py`) and `tasks/perf_model/fit_pdmux_step.py`. Held-out MAPE
  of the decode interval: pdmux + EDF 8.9% at C=8 and C=16 vs stock chunked LPT 74-117% and stock MTP 35-52%; decode
  step GPU time 4-7% vs 7.5-67%. Prefill-split time (TTFT side) stays hard to predict under pdmux (p90 62-95%).

## v1.20.27 — single-GPU bar met in the mean: pdmux + EDF (uncached deadlines, divisor 30) (2026-10-17)

- Three repeats + output gate PASS vs stock MTP: C=8 agent +29% / -31.7% / -31.1%, chat +25% / -73% / -55%; C=16 agent
  +33% / -32% / -40%, chat +32% / -57% / -55%. 5 of 6 C=8 agent per-rep values clear (one ITL at -29.6%).

## v1.20.26 — EDF deadlines on uncached tokens: pdmux + EDF clears all 12 bar means vs stock MTP (2026-10-17)

- `apply_edf_uncached.py` (`SGLANG_EDF_UNCACHED_DEADLINE=1`): the TTFT deadline scales with prompt tokens not already
  in the radix tree (read-only match at enqueue), so cheap cache-hit agent turns are not queued as cold prompts.
- With pdmux decode_bs_divisor 27, three repeats + output gate PASS: C=8 agent +28% / -30.1% / -31.6%, chat +25% / -71%
  / -55%; C=16 agent +33% / -32% / -40%, chat +32% / -57% / -55%. C=8 agent TTFT clears by 0.1 point only (per rep
  -31.1 / -29.4 / -29.7%); divisor 30 next. `srv_correctness.py` gained `SRV_EXTRA_ARGS` for pdmux arms.

## v1.20.25 — pdmux + EDF: gate PASS; decode-SM share and two-tier SLO are trade-offs (2026-10-17)

- Output gate: pdmux + EDF diverges from stock LPT exactly as the pdmux null case (short 1 / long 11; floor 163 / 96).
- More decode SMs (decode_bs_divisor 18 / 12) buy agent ITL (-37..-48%) at the cost of agent TTFT (-20..-26%); a flat
  agent TTFT SLO does not move agent TTFT and costs chat TTFT. Bar vs stock MTP: 11/12 at C=8 and C=16; C=8 agent ITL
  -28.6% is the one miss, on a TTFT/ITL frontier at about (-30%, -29%).

## v1.20.24 — PD multiplexing + worker-side EDF clears 11 of 12 bar metrics vs stock MTP (2026-10-17)

- H200, frozen 32-64K TraceLab cohort at agent C = 8 / 16 + ShareGPT 8 req/s, two repeats (spread <=2%): pdmux + EDF vs
  stock MTP -- C=8 agent +29% tput / -31% TTFT / -29% ITL, chat +25% / -84% / -54%; C=16 agent +32% / -32% / -41%,
  chat +32% / -66% / -55%. EDF on top of pdmux cuts chat TTFT 64% / 30% with ITL and throughput unchanged.
  `srv_correctness.py` gained `_EDF` arms; output gate and a decode-SM-share sweep queued.

## v1.20.23 — A5 closed: FA3 at state of the art; co-run bounded at 1.2% of serving GPU time (2026-10-17)

- Serving bound for FA3 || FA2 co-running from TraceLab step traces: 1.2% of GPU time (mixed steps are prefill-
  dominated, min/max p50 0.03). Multicast: ClusterM=2 deadlocks without a cluster-aware causal scheduler, and NCU shows
  98% L2 hit / 72% tensor activity, identical to sgl-kernel FA3 -- not load-bound. q-tile choice, split reduction,
  locality and in-kernel backfill closed by measured bounds. A1 re-closed.

## v1.20.22 — FA3 role at parity with upstream FA3; SM co-run pays -18% on append+verify (2026-10-17)

- Final FA3 role build: flash-attention mainloop, paged-TMA KV, `-DNDEBUG`, fast math, upstream's dynamic scheduler
  -- matches or beats sgl-kernel FA3 (prefill 2048 @ 64K 3.19 vs 3.27 ms). Split-KV on that path is exact after
  `fused_primary_finish` stopped re-merging over flash-attention's combine. Packed GQA is not a cost.
- A5.3: FA3 on B SMs co-run with stock FA2 verify rows vs the serving split (sequential): -18% on append 512 @ 48K +
  46 verify rows, tie or loss where one role dominates -- B must be chosen per step. Correction recorded in place: the
  1.1x residual was a per-SM gap, not the SM-count ratio.

## v1.20.21 — FA3 role on flash-attention's mainloop; -DNDEBUG closes most of the FA3 gap (2026-10-17)

- A5.0d: `FLASHINFER_FA3_UPSTREAM` builds and runs again (loader switch, flash-attention headers, dynamic smem now
  covers flash-attention's SharedStorage -- the launch had sized it from flashinfer's traits). All arms exact.
- A5.0e: the FA3 extension was compiled without `-DNDEBUG`; device asserts ran per tile. With it, prefill 2048 @ 64K
  4.68 -> 3.61 ms on 112 CTAs (sgl-kernel upstream FA3 3.32 ms on 132), chat 2048 0.168 -> 0.122 ms. ~57 us of fixed
  per-call cost remains on short items in our compact path.

## v1.20.20 — our FA3 role was flashinfer's paged mainloop, not upstream's (2026-10-17)

- A5.0c NCU: same occupancy, registers and smem as upstream FA3 and identical math instruction counts (HGMMA, MUFU,
  FMUL/FFMA/FADD), but 2x total instructions -- ~790M extra LOP3/PRMT/IMAD/SHFL/SHF/LEA plus cp.async KV gather.
  The profiled kernel is `DedicatedFA3PrimaryKernel<SparseCollectiveMainloop<...>>`: the upstream-mainloop role
  (`FLASHINFER_FA3_UPSTREAM`) was never enabled after the 0.7 port, so every FA3 measurement since used flashinfer's
  paged mainloop. A5.0d: loader switch `FA3_UPSTREAM`, flash-attention hopper headers on the pod; it now compiles and
  faults at run time (illegal address), under compute-sanitizer.

## v1.20.19 — pdmux + MTP payoff is negative; fast math worth 4-6% on our FA3 (2026-10-17)

- MTP + `--enable-pdmux` (race fixed, all arms clean) vs plain pdmux at C=8/16: throughput -24%, TTFT +61..+117%,
  chat ITL 2x; acceptance equals our MTP arm (2.12), so the earlier smoke-test acceptance gap was noise. Plain pdmux
  remains the strongest ITL baseline; ours keeps chat TTFT and ties throughput.
- A5.0b: building our FA3 with `--use_fast_math` (as upstream does) is 2.3-6.0% faster on every fixture, all exact;
  the remaining gap to upstream FA3 (1.37x long, 2-2.95x short) goes to an NCU comparison (A5.0c).

## v1.20.18 — pdmux + MTP cross-stream race fixed; FA3 gap located vs upstream FA3 (2026-10-17)

- pdmux + EAGLE3 MTP (patch item 8): the draft extend after a split prefill (prefill stream) shared the global
  flashinfer workspace and eager kv-index buffers with the concurrent decode step, giving an illegal address at C=16.
  It now runs on the draft runner's pdmux prefill backend (private workspace). Output gate re-run: PASS, unchanged
  short 1 / long 11. MTP_PDMUX runs at mem fraction 0.78 (unchunked 64K prefill OOMed at 0.85).
- A5 (new, user-requested FA3 technique study). A5.0: on FA3-only fixtures our persistent FA3 is 1.45x slower than
  upstream sgl-kernel FA3 on long items and 2.3-2.9x on chat-size ones, at 112 and 132 CTAs alike (SM budget is not
  the cause); our split-KV merge path costs 0.64 ms where it fires. A1 reopened: the winner map shows q64 up to 24%
  faster on short-KV few-row items and 1.34x slower in median elsewhere (the old "no difference" line was wrong).

## v1.20.17 — PD multiplexing runs under EAGLE3 MTP (eager), first end-to-end (2026-10-17)

- `tasks/debug_kernel_server_perf/pdmux_mtp_20261017/apply_pdmux_mtp.py` (tree `up_pdmux_mtp`): split-prefill support for
  the speculative path -- target capture + EAGLE3 aux across splits (llama, qwen2), draft extend after the last split
  as a normal EXTEND with input_ids restored, EagleDraftInput installed before decode, target verify on the per-stream
  decode backend, eager pdmux decode override limited to the target runner. Smoke: 24 requests served, 0 tracebacks,
  20/24 identical to stock MTP over 128 greedy tokens, acceptance 3.03-3.13. Next: srv_correctness, CUDA graphs, A/B.
- CUDA graphs work under pdmux + MTP. srv_correctness (PDMUX arm added): MTP_PDMUX vs MTP short 1 / long 11, PASS by
  rule; null case LPT_PDMUX vs LPT (no speculation, unpatched paths) gives exactly the same, so the divergence is
  pdmux's own numerics and the integration adds none. Payoff A/B running (`pdmux_mtp_ab_20261017.sh`).

## v1.20.16 — MuxWise baseline (sglang --enable-pdmux) beats our single-GPU config on ITL (2026-10-17)

- H200, frozen 32-64K cohort at agent C = 8 / 16 + ShareGPT 8 req/s, one session: PD multiplexing ties ours on
  throughput and agent ITL, has chat ITL 13.7-13.8 ms vs ours 24-25 ms (P99 32 vs 128-131 ms) and a far better
  slow-user tail; ours wins only chat TTFT (-49% at C=8, -7% at C=16). Ours still beats stock MTP clearly. 8 arms rc=0,
  0 tracebacks / fallbacks. Recorded as a negative result in the main doc's direct-baselines item.
- EAGLE3 MTP + `--enable-pdmux` fails in this tree: draft CUDA-graph capture (`decode_wrappers` None), and eagerly
  the first draft step in `event_loop_pdmux` (`topk1.py` assert) -- an integration gap, recorded in the same item.

## v1.20.15 — Pool pulls fail in the worker's host tier (host_capacity), not in the pool (2026-10-17)

- Prefetch diagnostic with `--enable-metrics`: the pool held 796K of the requested tokens but only 349K were
  prefetched; all deferred (553K) and unfulfilled (608K) tokens are `host_capacity` -- write-through keeps the L2
  host pool 100% full and sglang defers L3 staging rather than evicting L2. `HC_HOST_MODE` sets
  `--hicache-host-memory-mode`; buffer-only A/B queued (`mn_bufferonly_8gpu_20261017.sh`). `WORKER_EXTRA`,
  `SCRAPE_METRICS` knobs for diagnostics. H200: MuxWise-style `--enable-pdmux` baseline on the concurrency axis running
  (`tasks/debug_kernel_server_perf/baselines_20261017/pdmux_conc_20261017.sh`).
- Buffer-only A/B x2: pulls land 38% (was 18%) but the SLO check + transfer still loses to cache affinity (agent
  throughput -10..-11%, TTFT +8..+14%); buffer-only also costs cache affinity its L2 hits. Spread table + figure.

## v1.20.14 — Measured TTFT predictor error; conformal-style margin for the SLO check (2026-10-17)

- Hot mix, 13 200 requests: measured / predicted TTFT median 1.04-1.12, p80 1.34-1.58, p90 2.25-2.36 where a feasible
  node existed (table in item 7). `router.py --ttft-quantile q`: feasibility at the q-quantile of that ratio over a
  sliding window per prompt-length class. A/B queued (`mn_ttft_margin_8gpu_20261017.sh`, q 0 vs 0.8, x2).
- Measured x2 on the hot mix: q = 0.8 vs 0 within noise on every metric (margin binds, +10-20% own-TTFT rejections,
  but reps differ more than arms). Kept off by default. Spread figure + table in item 7.

## v1.20.13 — Overload: SLO check with cache fallback ties cache affinity; pool pulls do not land (2026-10-17)

- 192 x 16K shared-prefix mix, x2: SLO check + cache fallback = cache affinity within noise (chat 1 773 vs 1 737 tok/s,
  TTFT 27.6 vs 27.5 s): 95% of decisions had no feasible node and went to the top-hit node. The collapse of the
  most-slack fallback is gone. Learned pull realisation fell to 0.0 even with write_through: predicted pool pulls
  do not land -- open. Spread figure + per-rep table in the PD doc.
- Cause found: the pool was full (BatchPut "insufficient space" at 93-94% of 192 GB; one-off agent contexts written
  under write_through). `router.py --pool-write-policy write_through_selective` mirrors sglang's default (a block
  enters the pool after its second hit on a worker); `MC_EVICT_HIGH` sets the master's eviction watermark. A/B queued
  (`mn_selective_pool_8gpu_20261017.sh`).
- Selective pool, rep 1: no failed writes, pool 81% full, arms within noise; pulls 27% of predicted. `router.py`:
  the realisation EWMA learns from the unscaled prediction (`pull_raw`); learned from the scaled one it locked at
  0.0 once predictions fell below the prefetch threshold.
- Selective-pool A/B x2 (spread table + figure): cache affinity, SLO check, SLO check + transfer tie within noise at
  16 chat req/s on the 192-prefix mix -- past capacity, so placement has nothing to choose. 0 tracebacks / fallbacks.

## v1.20.12 — Resident ITL check measured: chat ITL -32% over the plain SLO check, agent TTFT +26% (2026-10-17)

- 8 H20, hot 4 x 16K shared-prefix chat + agent C=16, x2, clean grid and cache fallback: SLO check + resident ITL
  check vs cache affinity -- chat ITL -58% (P99 -33%), TTFT -88%, throughput +17%; agent TTFT +32%, throughput -23%.
  Over the plain SLO check: chat ITL -32%, agent TTFT +26%. 0 tracebacks / 0 fallbacks.
- Spread backfilled for every 2026-10-17 routing A/B (6 figures + CSVs, per-rep tok/s/user p1..p99, TTFT, TPOT tables in
  item 7); `mn_ab_figure.py` takes single-run arms. Standing rule: every serving result records its spread.
- `tasks/pd_colocation/mn_ab_figure.py`: per-arm, per-rep spread of a routing A/B (tok/s/user p1/p10/median/p90/p99,
  per-rep TTFT / TPOT); resident A/B figure and per-rep table in the PD doc.

## v1.20.11 — Router SLO check on the clean grid decides; resident-ITL charge measured as an overlap (2026-10-17)

- Clean-grid hot-mix A/B (rep 1): the SLO check now decides most routes (no feasible node 14%, was 45-63%) and keeps
  chat TTFT -89% / ITL -32% / throughput +20% vs cache affinity (agent throughput -11%, ITL +21%). The resident ITL
  check rejected 7 680 nodes without helping agent ITL: its per-step charge assumed prefill and decode serialise.
- `calibrate_h20.py` measures decode interference of a cold 32K prefill (`--chunk-interference`); `router.py` scales
  the resident charge by it. Measured 1.07-1.12: prefill chunk and decode serialise (decoder ITL 12-34 -> 380-426 ms
  beside a cold 32K prefill), so the charge was right and the "overlap" explanation is retracted; the most-slack
  fallback, which ignores ITL, is why the check did not help. Rerun with `--no-feasible-fallback cache`.

## v1.20.10 — Router TPOT prediction counted queued requests as decoding; SLO-check results re-read (2026-10-17)

- `router.py predict()`: rows decoding beside a request = those decoding now + those queued ahead of it (earlier
  deadline), not every in-flight request. The old count read the decode grid at 40-60 rows (~210-250 ms vs
  35-59 ms measured), so ~15K node rejections per hot-mix run came from an inflated own-ITL and 56-63% of decisions
  had no feasible node: earlier SLO-check results mostly measured the most-slack fallback (doc corrected in place).
  Resident-ITL A/B rerun on the fixed router with a cache-affinity control.
- `calibrate_h20.py`: decode ITL timed on a second, fully cached pass; the first pass put the other requests'
  prefill chunks inside the timed steps (210 ms at 32 x 16K vs 35-84 ms in serving) -- the grid's unexplained knee.

## v1.20.9 — SLO check protects requests already decoding on the target node; "veto" wording removed (2026-10-17)

- `router.py --check-residents`: a node is also rejected if admitting the request would push any request already
  decoding there past its ITL SLO (time spent so far + remaining steps at the step time with the new request + one
  prefill chunk per step during its prefill); decisions log the rejecting check. A/B queued
  (`mn_resident_itl_8gpu_20261017.sh`).
- Router pool model: learns `pull_realized` (EWMA of pulled / predicted tokens) and scales predicted pulls by it;
  bandwidth only from complete pulls, only upward (incomplete pulls drove it to 0 GB/s). The mirror predicted 19.5K
  pulled tokens where 1.6K landed on the 192-prefix mix (chat TTFT 234 s): sglang's default write_through_selective
  writes a prefix to the pool after its second hit. `HC_WRITE` sets the workers' `--hicache-write-policy`.
- `router.py --no-feasible-fallback cache`: when no node is predicted to meet the SLO, take the top-hit node instead
  of the most-slack one (the latter scattered requests off their prefixes under overload: chat TTFT 28 -> 73 s on
  the 192-prefix mix). A/B queued (`mn_overload_fallback_8gpu_20261017.sh`).
- Docs, CHANGELOG, router and scripts no longer use "veto": the router's SLO-feasibility check is called that; the
  decision-log field `vetoed` is now `rejected_nodes` (mn_audit reads both). Scripts renamed
  `mn_slo_check_8gpu_20261017.sh`, `mn_pool_slo_check_8gpu_20261017.sh`.

## v1.20.8 — Hot-prefix SLO check A/B: chat TTFT -90% vs cache affinity; transfer term untested by 4 groups (2026-10-17)

- 8 H20, 4 x 16K shared-prefix chat at 16 req/s + agent C=16, pool on, x2: the SLO check beats cache affinity on chat
  throughput +21%, TTFT -90%, ITL -34% at -5..-8% agent throughput, +16% agent ITL. With 4 prefixes the transfer
  term had only 37 transferable moves (prediction 81% of the actual pull); 192-prefix A/B queued.

## v1.20.7 — EDF reservation could chunk two requests at once (scheduler assert), fixed (2026-10-17)

- `edf_one_chunked_req.patch`: `_select_prefill_admission` refused to chunk a second request while one is in
  flight only under `shortest-prefill-first`; EDF's earlier-deadline reservation also caps the in-progress chunk,
  so a waiting request got chunked beside it and `assert self.chunked_req is None` killed the worker (8-GPU
  hot-prefix arm). The guard now applies whenever a chunked request is in flight. Hot-prefix A/B rerunning.

## v1.20.6 — Pool-on SLO check A/B: the transfer term never fires under best_effort prefetch (2026-10-17)

- 8 H20, agent C=16 + chat 8 req/s, mooncake L3 pool on: under the default `best_effort` prefetch the SLO check sends
  requests to idle non-holders, where no pull lands, so the transfer term predicts 0 and the two SLO-check arms run the
  same logic; their 8% agent-TTFT and 1.35 vs 15.9 s chat-P99 gap is run-to-run noise. Predictor ~30% optimistic,
  agent error p10/p90 -7.7/+9.6 s. Next: `timeout` prefetch on workers + router, with repeats.
- `mn_node.sh` / `mn_run.py`: `HC_PREFETCH` sets the workers' `--hicache-storage-prefetch-policy`; timeout-policy
  A/B queued (`mn_pool_timeout_8gpu_20261017.sh`, cache vs SLO check + transfer term, interleaved x2).
- `router.py --prefill-curve`: prefill time from the calibrated cold-TTFT curve (T(ctx) - T(ctx - uncached))
  instead of one blended rate; `calibrate_h20.py` prints it.

## v1.20.5 — Router SLO check on the final stack: wins at C=32, neutral at C=16 (2026-10-17)

- 8 H20, same-session A/B (`mn_slo_check_8gpu_20261017.sh`): at agent C=32 + chat 8 req/s the SLO check beats cache
  affinity by +24% agent throughput, -23% agent ITL, -32% chat ITL (+10% TTFT); vs stock colocation every bar
  metric clears except agent TTFT. At C=16 it is neutral-to-worse (no pool: a moved request recomputes).
- `mn_node.sh`: with the L3 pool on, cap the HiCache host pool (`--hicache-size 48`) and the mooncake segment
  (24 GB): the defaults (~130 GB + 32 GB per worker, two per 232 GB pod) OOM-killed every worker at startup.
- `mn_run.py`: mooncake master from the venv (`$W/venv/bin/mooncake_master`; the old ~/.local path does not
  exist on the re-provisioned pods, so no master ran and every worker failed "setup Mooncake store"); an arm
  now aborts if the master is not running.
- Provisioning installs `libssl-dev`: HiCache's native page hash JIT-compiles against `openssl/sha.h`; without it
  every pool-on worker died at startup ("Failed to load HiCache native hash extension").

## v1.20.4 — PD doc: inter-node SLO-feasibility prediction as a major open item (2026-10-17)

- `pd_colocation_vs_disagg.md` item 7: predicting TTFT/ITL on a node from its hit length and load for
  the routing SLO check -- current predictor, its errors by sign (best-effort KV pull pessimistic; blended
  prefill rate, decode interference, future arrivals, stale hits optimistic), open design questions,
  sub-items.
- `router.py`: transfer-aware TTFT prediction for the SLO check (`--pool-kv-tokens`, `--prefetch-policy`,
  pull-bandwidth EWMA from `cached_tokens`), pool mirror with HiCache-compatible chained block keys,
  `ttft_obs` logged per request. Pool-on A/B queued (`mn_pool_slo_check_8gpu_20261017.sh`).

## v1.20.3 — H20 8-GPU chat request-rate sweep at reference-list rates (2026-10-16)

- 1 / 2 / 4 / 8 chat req/s per GPU at agent C=16, stock coloc | ours | stock PD disagg, with the throughput vs
  tok/s/user figure (p1/p10/p90/p99 error bars). Ours beats stock colocation on throughput (+6..+25%) and TTFT at
  every rate; >=30% ITL only at 1 req/s/GPU; agent TTFT never clears -30%. Disagg: worst TTFT and throughput,
  lowest TPOT. 8 H20s saturate between 1 and 2 chat req/s per GPU with this agent tenant.

## v1.20.2 — Interactivity error bars on the throughput vs tok/s/user figures (2026-10-16)

- `eval_curves.py` reports per-request tok/s/user p1 / p10 / p90 / p99 next to the median;
  `plot_eval.py` and `eval_curves.py --plot` draw them as horizontal error bars (median marker,
  p10-p90 bar, p1-p99 whisker); `plot_eval.py` plots arms beyond the three S1 x S2 names.
- Regenerated S1 x S2 and both EDF rate-sweep figures (`tasks/debug_kernel_server_perf/userpct_20261016/`).
- `eval_curves.py`: `burst_frac`; requests that stream all tokens at once (< 1 ms decode per token, 17-29%
  of stock PD-disagg requests) are excluded from tok/s/user; spec-count sums tolerate `None`. Figures and
  CSVs force-added (`*.csv` / `*.png` are gitignored; the previous commit linked missing files).
- `tasks/pd_colocation/mn_sweep_figure.py`: the H20 chat-rate sweep -> eval_curves layout -> figure.
- `mn_node.sh`: claims a torch.distributed port and passes `--nccl-port` (sglang's default http port +
  randint(100, 1000) landed on other workers' claimed ports and hung init); the supervisor restarts a worker
  whose "Init parallel" has not ended after ~60 s instead of 360 s.
- PD doc: 8-GPU C=32 our-router rows rerun with the fixed router; old rows retracted (router-capped).
  Ours vs stock colocation at C=32: agent +19% throughput, -4% TTFT, -23% ITL; chat +13% throughput, -65% TTFT.

## v1.20.1 — Multi-node teardown kills supervisor loops (2026-10-16)

- `tasks/pd_colocation/mn_kill.sh` now kills `mn_node.sh` supervisor shells before workers. A surviving
  supervisor from an aborted arm relaunched its worker seconds after teardown; that orphan held 86 GB on
  pod 0 GPU 1, so every PD-disagg prefill worker there OOMed (logged as "bindfail").
- `mn_node.sh`: worker ports picked below the ephemeral range under a lock; `mn_run.py`: 40-min health deadline.
- `router.py`: connector `limit=0` (aiohttp's default capped it at 100 concurrent upstream connections,
  binding at C=32 + chat 8 req/s), keep-alive 3 s, retry also on `ClientOSError` (an EPIPE aborted the C=32
  stock-MTP agent client). Both 8-GPU C=32 our-router rows marked not final; rerun queued.
- PD doc: stock PD-disagg 8-GPU C=16 measured (clean upstream): ours is 3.1x agent throughput, -80% agent
  TTFT, -98.7% chat TTFT; disagg keeps lower ITL. `mn_audit.py` takes `NPODS`.

## v1.20.0 — S3 complete in scope; load + context gate for the 4-mode step (2026-10-16)

- S3 standalone (ShareGPT, GSM8K, MATH-500, HumanEval, Code-Feedback, WebShop agentic web browsing):
  no gain from the 4-mode step on short prompts; LongBench rate points retracted (past capacity).
- `mixed_spec_ctx_queue_gate.patch`: mix prefill into the verify step only if prefill context >= 8192
  or >= 4 requests wait. Keeps the full cohort gain, ShareGPT TTFT -45% at MTP throughput, WebShop
  throughput loss -10% -> -7%. Promoted as the default dispatch.
- H20: 8-GPU vibecuda job (4 pods x 2) provisioned via the ghfast.top GitHub proxy and non-xet HF mirror.

## v1.20.0 — Final single-GPU config clears the bar on the mixed workload near capacity (2026-10-16)

- Split + EDF + floor + uncapped chunk preemption (park gate PASS) vs MTP, cohort C=8 + chat 4/s:
  agent tput +44% / TTFT -35% / ITL -31%, chat tput +11% / TTFT -83% / ITL -39%. Under overload
  (C=16 or chat 8/s) agent TTFT (-19..-27%) and chat ITL (-16..-19%) miss; P99 ITL -72..-86%.

## v1.19.11 — Uncapped EDF preemption: TraceLab TTFT -39..-43% vs MTP (2026-10-16)

- Fixed-code preemption, no cap (shortest remaining work first): C=8 TTFT -39/-43%, tput +35%,
  ITL -23/-24%; C=4 TTFT -43%, tput +28%, ITL -17%. TTFT and throughput clear the bar; ITL does not.

## v1.19.9 — In-server prefill rate: FA3 split +30% over MTP's FA2 on TraceLab (2026-10-16)

- Step-traced TraceLab C=4: prefill 476-507 TFLOP/s (split + EDF) vs 351-401 (MTP) at every prefix
  bucket up to >=256K; GPU-busy -19%. Mean TTFT on full-length TraceLab is bounded near this gain on
  one GPU [retracted 2026-10-16: EDF chunk preemption reaches -32..-40% TTFT on one GPU]; agent ITL
  SLO 80 ms not promoted (no consistent gain).

## v1.19.10 — EDF chunk preemption: TraceLab C=4 TTFT -39% vs MTP (2026-10-16)

- Preemption removes the cache-hit TTFT tail (C=4 <1K-uncached mean 3.1 s -> 0.3 s); C=4 TTFT
  -39/-40% and tput +27% vs MTP, ITL -20%. C=8: TTFT -15%, ITL -29%, tput +32%. Uncapped (SRPT)
  preemption queued; forced park gate pending.

## v1.19.9 — EDF chunk preemption for the agent-TTFT miss (2026-10-16)

- Root cause: a cache-hit round needing more than one step's budget waits behind a cold long
  prompt's whole chunked prefill. `edf_preempt_chunk.patch` parks the chunked prompt (prefix kept
  in the radix tree) for a short earlier-deadline request; gated A/B queued.
- H20 8x2 vibecuda re-allocated (job 1c90eac0a6848b48) now that A and B are closed.

## v1.19.8 — A3 closed: fitted dispatch = default at guard 0; all kernel items closed (2026-10-16)

- Clean A/B at guard 0 (gates pass; cohort C=16 x2, TraceLab C=8): A3B within noise of DEFAULT on
  every metric. The earlier A3b TTFT gains were the decode-guard removal. A and B are both closed.

## v1.19.8 — Same-queue confirmation: EDF + reservation vs MTP (2026-10-15)

- 24 arms, 2 interleaved reps (spread 0-2% on throughput/ITL). EDF + deadline reservation vs MTP:
  throughput +26..+34%, P99 ITL -68..-88% everywhere; misses mean TTFT on TraceLab (-4/-5%), cohort
  agent TTFT (-28%) and chat / TraceLab-C4 mean ITL (-24/-27%). Bar not met on all metrics.
- Short-request reservation not promoted (trades agent throughput/TTFT for chat TTFT -93%).

## v1.19.7 — EDF reservation was inert: waiting requests were never prefix-matched (2026-10-15)

- Both reservation variants fired ~once per run: the FCFS/priority policy never prefix-matches
  waiting requests on this radix cache, so their uncached work read as their full length.
  `edf_reserve_matched.patch` matches them inside the reservation; rerun queued.

## v1.19.7 — EDF reservation measured; short-request reservation mode 2 (2026-10-15)

- Earlier-deadline reservation vs same-queue EDF control: mean TTFT -38% (TraceLab C=4), -28% (C=8);
  short appends -66% / -44%; ITL +13%. Fires rarely at C=8 (long prompts already overdue).
- `edf_reserve_short_mode2.patch` (`SGLANG_EDF_RESERVE=2`): also reserve for short later-deadline
  requests; queued in the same-queue confirmation run.

## v1.19.6 — A4 closed by measurement (2026-10-15)

- Codex's paired native-PDL serving (no uniform gain) and unpacked-query FA3 (<0.6%) complete; with
  every candidate measured against the split, A4 closes: verify rows run on stock FA2 in the split.

## v1.19.6 — Concurrency sweep complete (2026-10-15)

- 18 arms (TraceLab C=2/4/8, cohort C=8/16/24 + chat 8 req/s), all clean with peak = C. Throughput
  +22..+34%, P99 ITL -64..-88% above C=2; agent mean TTFT and chat mean ITL miss the 30% bar.

## v1.19.5 — Program-concurrency sweep complete (2026-10-15)

- 18 arms, peak = C every arm. Throughput +22..+34%, TPOT and P99 ITL clear the bar at every loaded
  point; agent TTFT (-21..-28% cohort, worse on TraceLab) and chat mean ITL (-17..-27%) miss.
  Chat at 8 req/s overloads every cohort point; a below-capacity chat rate is next.

## v1.19.4 — TraceLab concurrency sweep C=2/4; EDF chunk reservation for earlier-deadline requests (2026-10-15)

- Full-length TraceLab at program concurrency 2 and 4: split / EDF clear throughput and ITL at
  C=4 vs MTP; mean TTFT is +14-15% from head-of-line blocking of cache-hit rounds behind a chunked
  prefill.
- `edf_reserve_earlier.patch` (opt-in `SGLANG_EDF_RESERVE_EARLIER`): reserve chunk budget for
  waiting requests whose uncached-work deadline precedes the chunked request's; A/B queued.

## v1.19.5 — Concurrency sweep C=2/4; EDF earlier-deadline reservation queued (2026-10-15)

- Full-length TraceLab by program concurrency: at C=4 the split clears throughput (+26-29%), ITL
  (-31..-34%) and P99 ITL (-67..-73%) vs MTP; mean TTFT +14% comes only from short appends blocked
  behind long prompts' remaining chunks (cold >=32K prefills improve 20-26%).
- `edf_reserve_earlier_deadlines.patch` (`SGLANG_EDF_RESERVE=1`): EDF reserves step budget for
  earlier-deadline waiting requests ahead of the chunked request; A/B queued.

## v1.19.4 — S3 re-queued on request-rate axis behind the concurrency sweep (2026-10-15)

- `s3_standalone_final_20261015.sh` waits for `concurrency_sweep_20261015.sh`; Mooncake points are
  reported as achieved req/s (timestamp compression, Crossflow's protocol), never session rates.

## v1.19.3 — Agent load is program concurrency only; EDF load points retracted (2026-10-15)

- Session-start rates (sessions/s) are dropped as a load axis (user): the full-length TraceLab
  0.005/0.02 sessions/s run is cancelled (0.005/s averaged 1.3-1.5 requests in flight).
- The EDF trees lacked the program-concurrency replay, so the EDF frozen-cohort runs released all
  48 programs (31-35 in flight, not 8/16/24), and the v1.19.0/1.19.1 rate sweeps drove the agent
  tenant at 0.1 sessions/s. Their "clears the bar" claims are retracted pending
  `concurrency_sweep_20261015.sh` (replay ported: `edf_floor_session_concurrency.patch`).
- S3 queue cancelled (Mooncake replay must move to concurrency too).

## v1.19.2 — S3 standalone workloads queued on the final config (2026-10-15)

- `s3_standalone_final_20261015.sh`: MTP | split static 3072 | split + EDF 50 ms + floor 2048 on
  ShareGPT, GSM8K, MATH-500, HumanEval, Code-Feedback, Mooncake, LongBench v2; queued after S2b.

## v1.19.1 — EDF sweep repeats and mix128 confirm; two-tier TTFT SLO rejected (2026-10-15)

- mix32 12/16 req/s repeats reproduce the v1.19.0 gains; mix128 16/s clears the bar for both
  tenants (chat tput +59%, TTFT -83%; agent tput +47%, TTFT -33..-41%). Remaining gap: agent TTFT
  at mix128 12/s (-18%) and borderline at mix32 12/s (-31% mean).
- Two-tier TTFT deadlines (flat agent/chat SLOs) only trade TTFT between tenants at 12/s and lose
  for both at 16/s; not promoted.

## v1.19.0 — Split + EDF dynamic chunk with a prefill floor clears the bar vs MTP at and past capacity (2026-10-15)

- S1 mix32 rate sweep (chat 8/12/16/s + 32-64K agent sessions): split kernel + EDF chunk budget
  (refit guard-0 step model, per-request acceptance EWMA, 2048-token prefill floor) vs stock MTP:
  at 12/s chat TTFT -85% / TPOT -63%, agent tput +57% / TTFT -35% / TPOT -54%; at 16/s chat tput
  +12% / TTFT -99% / TPOT -71%, agent tput +108% / TTFT -90% / TPOT -59%. Both tenants clear
  >=30% TTFT/ITL and >=10% throughput at 12 and 16/s. Chat SLO attainment 0.975-0.988 vs MTP
  0.119-0.755. One run per arm; repeats pending.

## v1.18.12 — Guard-0 step-model refit; EDF dynamic chunk starves prefill past saturation (2026-10-15)

- Refit with the decode guard off: held-out MAPE split 10.1% vs unfused 10.7% (goal point 3).
- EDF dynamic chunk on the overloaded cohort: meets the ITL target (19.9 / 16.2 ms) but cuts
  prefill to ~1.3K / ~300 tokens per step, so TTFT and throughput fall (30 ms: worse than MTP).
  Opt-in prefill floor added; near-capacity A/B queued.

## v1.18.11 — Static chunks cannot clear both bars; true LM-head epilogue bounded (2026-10-15)

- Chunk sweep repeats: split at 2048 and 3072 each miss one bar (2048: agent TTFT -25%; 3072:
  chat ITL -29.1%, agent TTFT -27.4%); MTP control repeats within 1%. EDF dynamic chunk queued.
- B: a real GEMM-epilogue argmax could save at most 19-235 us per step (0.2-1%), below the serving
  noise floor; not built.

## v1.18.10 — Prefill chunk is the chat-ITL lever on the split with guard 0 (2026-10-15)

- Picked up the Codex kernel worker's thread (its last job exited 02:06; GPU taken per the user).
  Its Q32 FA2 serving A/B: within noise, not promoted.
- Chunk sweep on the frozen 32-64K cohort: split at chunk 2048 vs MTP (best at 8192) gives chat
  +34% tput / -43% TTFT / -34% mean ITL and agent +38% / -28% / -56%; only agent TTFT misses the
  bar. MTP does not gain from small chunks. Supersedes the 2026-10-12 "chunk is not the lever"
  finding, which was measured with the decode guard on.

## v1.18.9 — Codex kernel worker's 2026-10-13/14 results integrated; B3 closed (2026-10-14)

- B3 closed by measurement: true TMA multicast (page 128, 2-CTA clusters) implemented and gated by
  the Codex kernel worker; -0.2..+1.7% vs unicast TMA. DSMEM KV sharing (two versions) loses.
- A3b on the serving split complete (12 arms): large mean gains but not promoted (SLO/fairness
  trade-offs, acceptance confounds).
- A4 candidates (native-PDL FA2, dedicated FA3 entry, compact planning, auto4096 cap, KV96, SM
  budget, one-warp producer): none beats the plain split in serving; FA3 work-pair locality
  (-14%/-26% kernel) is in serving test. KV96 work fixed a real packed-Q gather bug.
- Overlap A/B: no gain beyond noise; the plain two-launch split remains the serving kernel.
- eval_curves: relative-SLO reference directories named <mix>_unloaded_<stream> now resolve
  (Codex fix + test).

## v1.18.8 — Serving kernel switches to the two-launch FA3 + stock-FA2 split (2026-10-13)

- Split dispatch fix: the two-launch split was planned but never run in serving (ragged path);
  ragged is now off whenever a two-sided split plan exists. Gate wrappers fail on the run's own
  status (the earlier split "PASS" was a swallowed failure; recorded in place).
- Split vs fused co-launch vs stock MTP at mix32/mix128 x chat 16/24/s, race-fixed tree, no
  barrier: the split wins every point; vs stock it clears >=30% TTFT/TPOT and >=10% throughput at
  3 of 4 points for both tenants (mix128 24/s chat +56% tput, -55% TTFT, -45% TPOT).
- FA2 executor tuning: KV chunk 4096 best (2.52 -> 1.84 ms) but still ~20% behind stock FA2; a
  fixed chunk floor is rejected (small-KV regressions).

## v1.18.7 — B1 headroom measured; verify-row bandwidth is the kernel lever (2026-10-13)

- B1: small ops are ~3% of serving kernel time in every arm (attention 75-80%, GEMM 18-22%);
  epilogue fusion cannot approach the bar. Kineto occupancy estimates are unusable.
- A4: at serving verify shapes the co-launched persistent FA2 executor reaches 38-52% of HBM
  bandwidth, stock BatchPrefill FA2 63-83%; the two-launch split beats the fused co-launch on
  every fixture (1.53 vs 2.57 ms on chunk+verify). Split-vs-fused serving A/B queued.
- Per-context-range attribution of the S1 x S2 gains (scheduling vs kernel).
- Dynamo comparison made optional (conditional disaggregation lacks an sglang backend).

## v1.18.6 — Graphs-on 4-mode crash fix validated; full-length replay fits the KV pool (2026-10-13)

- Race fix PASS: unfixed tree without the barrier crashes (index assert); fixed tree without the
  barrier runs both 4-mode arms twice (1651-1751 mixed steps each) and the synchronous path with 0
  tracebacks, output gate PASS. The mixed-step barrier is retired for all later 4-mode arms.
- S2b v2 aborted: 24 sessions overflowed the 1.82M-token KV pool (14.4M tokens prefilled vs 5.8M
  uncached). v3 replays 8 seeded sessions (combined peak 1.37M; p10/50/90/99 38K/123K/223K/390K).
- B1 headroom profile moved ahead of the S2b resume.

## v1.18.5 — Graphs-on crash root cause; A3 serving loss explained; S3/S4 drivers (2026-10-13)

- Graphs-on 4-mode crash: the verify half's `future_map.publish(new_seq_lens)` and relay stash ran
  on the schedule stream, unordered against the forward stream producing those tensors (stale
  seq_lens -> out-of-range indices next step). Fixed by issuing both on the forward stream; the
  synchronous (overlap-off) path gained the 4-mode handoff it never had, which retracts the
  "overlap off does not cure" bisect arm. `_fs_upload` waits on the previous copy's event before
  rewriting its pinned buffer. Validation (null case, gate, serving without barrier) queued.
- A3: the fitted work floor rejected chat prefill chunks, making 65% of mixed steps one-sided (7%
  under the default rule). A3b scopes the floor to the fitted regime; gated A/B queued.
- Relative SLOs in `eval_curves.py` from unloaded single-tenant runs (the lowest swept rate is
  already loaded: mix128 chat median TTFT 38 s); `STREAMS` filter in the workload driver.
- S3 standalone sweep and S4 EDF step-model drivers (STEPTRACE re-ported) queued.
- Pod: a pushed tarball replaced the `validated_upstream -> up` symlink with a directory for six
  minutes; restored before any server launched from it.

## v1.18.4 — B3 multicast ceiling on the current kernel; full-length TraceLab v2 (2026-10-13)

- B3: skipping every FA3 K/V global load (`FLASHINFER_FA3_SKIP_KV_LOAD`) saves 4.9-6.6% on mixed
  serving fixtures and 16-20% on short-append FA3 rows against 16-64K of context; a 2-CTA multicast
  can reach at most half of that. Agent-append shapes queued (`b3_append_ceiling_20261013.sh`).
- S2b: v1 (120 sessions) aborted after it queued 16.8M prompt tokens. v2 replays 24 sessions x 10
  rounds whose context distribution matches the release (new `--tracelab-selection-seed`, seed
  187), at 0.01/0.02/0.05 sessions/s (`tracelab_full_v2_20261013.sh`).

## v1.18.3 — Evaluation matrix results, full-length TraceLab, PD points 5-6, doc cleanup (2026-10-13)

- **S1 x S2 sweep** (chat 4-24/s + TraceLab agent 0.1/s, graphs on, 0 fallbacks/tracebacks):
  arms equal below capacity; under load fused 4-mode vs stock MTP gives TTFT -31..-79%,
  throughput +11..+36%, TPOT -19..-30% (mix32 and mix64); 4-mode P99 ITL is worse at light load.
  mix128 running.
- **Full-length TraceLab:** release statistics (context p50 ~124K, a quarter of Claude rounds
  >= 256K, outputs ~200 tokens); loader replays both providers, first rounds and round-weighted
  sessions (replayed median 120-126K); YaRN x32 driver queued.
- **PD point 5:** disagg re-ships session KV every agent round (p50 285 ms per round for 7B at
  200 Gb/s; 3.1 PB vs 107 TB append-only over the release); Dynamo's conditional disaggregation
  avoids most of it, stated plainly. **PD point 6:** Dynamo's router, conditional disaggregation,
  planner and KVBM compared with ours point by point; hyperlinked references.
- **Docs:** main checklist cut from 19,147 to 163 lines (full log archived verbatim); reference
  list with a note per paper, semi-PD and Dynamo added.

## v1.18.2 — A3 map on the fixed kernel, A1 closed, B backfill measured, graphs-on serving (2026-10-12)

- **Pod re-forked** to `095f01a3e1585eea` (trial expired); serving tree rebuilt from the fork's
  upstream base plus the serving diff; the rebuilt kernel reproduces the old pod's verdicts and
  timings within 1%.
- **A3:** winner map on the fixed kernel, 216/216 arms exact. FA2 wins every verify-width row;
  fitted rule (FA3 iff packed >= 112 and FA3 work >= 3.5M) agrees on 69/72 cells (worst loss 6%)
  vs 55/72 (98%) for the old rule -- but **loses in serving** (-3.5% chat throughput, +29% TTFT),
  because a batch it makes one-sided runs the stock wrapper. Refit open.
- **A1 closed by measurement:** FA3 q64 never beats q128 beyond noise.
- **B backfill:** co-launch hardware backfill worth up to 2.9x when FA2 work exceeds the slots
  beside FA3; in-kernel `tma_store_wait` backfill is correct and worth 1.4-5.5x over none but
  that path is 2-6x slower overall.
- **Graphs on:** the output gate passes, but both 4-mode arms crashed under load (async index
  assert). Bisected to work a mixed step leaves in flight (sync after mixed steps cures it; after
  non-mixed steps does not; overlap scheduler, flashinfer plan staging and the in-step graph are
  refuted). Root cause open; measured with a mixed-step barrier, fused 4-mode vs stock MTP:
  **TTFT -56% / -58%, throughput +20% / +25%, mean ITL -21% / -13%** (chat / codex).
- **S tooling:** `eval_curves.py` (Pareto + SLO attainment), env-driven mixture/standalone
  scenarios, 32K-256K TraceLab extract, S3 datasets prepared.

## v1.18.1 — Fused co-launch beats stock MTP on TTFT and throughput in realistic serving (2026-10-12)

Class set 64/32/16 plus `FUSED_FA2_KV_CHUNK=auto` (FA2 chunk sized to the CTA slots FA2 can
occupy; the first build left `len_kv_chunk` at "no split", failed every split row and the q4 gate,
and its serving run was voided). Fixed build: 10 kernel fixtures exact, beats `BatchAttention` on
every mixed verify fixture; q4 serving-output gate PASS.

Realistic A/B (chat 24/s + TraceLab codex 0.1/s, EAGLE3 q4, no graphs, 2 fused rounds, 0
fallbacks): fused vs stock MTP **TTFT -55% / -58%, throughput +18% / +21%, mean ITL -19% / -11%**
(chat / codex); vs the unfused 4-mode arm throughput +7% / +6%, TTFT -33% / -37%. Mean ITL misses
the 30% bar (P99 tail ~111 ms unchanged) and no arm runs CUDA graphs yet.
Evidence: `tasks/debug_kernel_server_perf/resume_20261012/colaunch_20261012.md`.

## v1.18.0 — FA2 leaves the co-compiled kernel (PDL co-launch); tile 48 refuted (2026-10-12)

`FLASHINFER_FUSED_COLAUNCH=1` (flashinfer `persistent_fa3_fused.cuh`): the FA3 grid runs alone on
`sm_split` CTAs and `FA2ExecutorKernel` -- same runners, plan and queue at native 128x2 geometry --
is its programmatic dependent (FA3 `griddepcontrol.launch_dependents` at entry, FA2
`griddepcontrol.wait` before exit). FA2 CTAs fill the SMs FA3 does not hold and then the ones it
vacates. All gate arms exact against the fp32 causal reference and on graph replay; 10 race
repeats exact.

- 48/32/16 classes: 3.3x on the A2 mixed fixture (0.508 -> 0.156 ms), 3.8-6.5x on FA2 rows; the
  executor still spills (648 B stack).
- **Tile 48 is the spill**: 1 q-warp x 4 kv-warps, `num_mma_q=3`. 64/32/16 builds at 96 B and is
  1.6-2.2x faster at every q8 shape; the co-launched kernel then beats `BatchAttention` on the A2
  mixed fixture (0.096 vs 0.106 ms) and is 1.55x off the unfused two-launch control. The A2
  class-48 premise is refuted in place; the tail dispatcher stays.

Evidence: `tasks/debug_kernel_server_perf/resume_20261012/colaunch_20261012.md`
(`colaunch_gate_20261012.sh`, `colaunch_tiles_20261012.sh`, `colaunch_persistent_fa3_fused.patch`).

## v1.17.1 — Post-RoPE-fix realistic A/B: the 4-mode schedule wins, the fused kernel gives it back (2026-10-12)

First serving timing on the fixed model (`agent_chat_rated`: chat 24/s + TraceLab codex 0.1/s,
EAGLE3 chain q4, no CUDA graphs, one round, 0 fallbacks/tracebacks in every arm). All arms are
saturated, so TTFT tracks capacity.

- Unfused 4-mode (`MTPMIX_LPT`) vs stock MTP: TTFT **-32%** (chat and codex), throughput
  **+10% / +13%**, mean ITL -13% (short of the 30% bar), P99 ITL flat.
- Fused 4-mode (`MTPMIXFUSED`, >=43,601 fused calls, A2 tail on) vs unfused 4-mode: throughput
  -11% / -13%, TTFT +32% / +29%; median chat ITL lower (16.9 vs 20.3 ms). The fused kernel is
  slower per call than the attention it replaces, consistent with the harness gap to `BatchAttention`.
- Acceptance on the real workload: 2.13-2.22 tokens per verify, equal across the 4-mode arms.

Serving-output gate at q4: PASS for the unfused, split and fused 4-mode arms (long prompts agree
on >= 11 tokens; 0 fallbacks). Round 2 reproduces round 1 (fused chat 1251 vs split 1410 tok/s).

**Cause, and a retraction:** the FA2 role is ~21x slower inside the fused kernel than in
`BatchAttention` on a decode-shaped row (345 vs 15 us per call, both exact): the kernel compiles at
168 registers with a 4200-byte stack vs 255 / ~100 B, so the co-compiled FA2 body spills in its
inner loop. The 2026-10-10 claim that the "FA2 22x" was a fixture artifact is retracted in place.
The A3 winner map (108 arms, all exact) therefore measures the defect, not the dispatch boundary.
Fix under test: `FLASHINFER_FUSED_COLAUNCH=1` -- FA3 grid on `sm_split` CTAs plus
`FA2ExecutorKernel` at native geometry as its PDL dependent (`colaunch_persistent_fa3_fused.patch`).
Evidence: `tasks/debug_kernel_server_perf/resume_20261012/postfix_realistic_ab_20261012.md`
(runner `postfix_realistic_ab_20261012.sh`, sha256 144dc5d4...).

## v1.17.0 — A2 q8 serving gate PASSES after fixing a shared RoPE bug; tail dispatch is bit-identical (2026-10-12)

**The handed-off q8 run failed its gate, and the cause was the serving stack.** Stock LPT is not a
reliable control: archived repeats diverge on long prompts (R1 vs R3 at `[16,72,33,96]`). An
independent HF reference (bf16/fp32, with and without YaRN) showed both stock and fused wrong on
3 of 4 long prompts by up to 24 logits.

**Root cause:** under transformers 5.x `rope_theta` lives inside `rope_parameters`; sglang's
`--json-model-override-args` loop replaced the rope dict with the YaRN-only override, and
`get_rope_config` fell back to 10000 (Qwen2.5 = 1e6). Bisection: bare server 4/4, + override 0/4,
reproducing stock LPT's tokens exactly. Every serving driver (`srv_correctness.py`,
`run_mixed_workloads.py`) passes that override, so **all serving correctness verdicts and serving
timings before this fix used a corrupted model** and are retracted in place.

**Fix** (sglang-up `srt/utils/hf_transformers/config.py`, upstream bug): a rope override that omits
`rope_theta` inherits the config's; explicit values win. `test/manual/test_rope_override_theta.py`
passes 3/3 and fails on the pre-fix file with `10000 != 1000000.0`.

**Post-fix:** all five stock configs (up to YaRN + chunk512 + mixed-chunk + background decodes +
EAGLE3 q8) match the reference 4/4. The q8 A2 gate passes (`status=0`), and **TAIL1 is
bit-identical to TAIL0 on all 12 requests** while the plan moves 384/0/0/112 → 192/0/192/112.
Every fused first token is the reference argmax or within one bf16 ulp (s1: 0.067-logit tie).
Per-draft acceptance is equal across arms; the fused arm's extra verify calls are accounting
(stock mixed-chunk decodes rows without speculation). Acceptance on these slice prompts is ~0.16
and did not depend on the bug; it must be measured on the real chat+TraceLab workload.

Correctness only — no serving performance claim. Evidence:
`tasks/debug_kernel_server_perf/resume_20261012/a2_serving_ropefix_20261012.md`.

## v1.16.0 — RESOLVED: the 4-mode gate never fires; the fused path is unmeasurable here (2026-10-11)

Serialised run, `SGLANG_DBG_DRAFT_COPY=1`, server to completion:

| artifact | value |
|---|---|
| server log | **9474 lines** (full run) |
| **`MIXEDGATE` lines** | **0** |
| `mixedspec` | **0** |
| prefill with `#running-req > 0` | **4195** |

The `MIXEDGATE` print is **inside** the gate body (`scheduler.py:4150`, after the `if` at `:4141`), so
**0 lines ⇒ the condition is false** — not mixing-without-logging. That resolves the earlier
ambiguity: "zero mixed steps" was right, now confirmed by the gate's own instrumentation rather than
by `SGLANG_LOG_MIXED_COMPOSITION`, which the fork doesn't read.

Four of seven clauses are demonstrably satisfied (`SGLANG_MIXED_SPEC`, `is_mixed_chunk`,
`spec_algorithm` not none, `running_batch` non-empty ×4195). The failing one is `return_logprob`,
`input_embeds`, `beam_group` — or `:4141` is never reached on this scheduler path.

**Terminal finding for the serving thread:** the 0.7 port's production behaviour is **unmeasurable on
this pod as configured**, not merely unmeasured. Eight attempts, seven blockers (six provisioning, one
self-inflicted).

**Next, and it is a code change not a measurement:** log each clause's value immediately before
`:4141`. Every artifact-level approach is exhausted — the gate's own print was the last.

## v1.15.0 — every gate condition satisfied yet no mixed steps logged; MIXEDGATE is the next step (2026-10-11)

Walked `scheduler.py:4141` against the driver: `SGLANG_MIXED_SPEC=1` **set and read**;
`--enable-mixed-chunk` **passed** (`:328`); EAGLE3 args **added** (`method.startswith("MTP")` matches
`MTPMIX_SPLIT`); `running_batch` non-empty **4158 times**. All satisfied — and yet
`MIXEDSPEC_STEPS`/`mixedspec` appears **0 times** in a completed trace run.

Two possibilities I **cannot** distinguish from current artifacts: (1) mixing genuinely doesn't
happen for a reason outside these conditions; (2) mixing happens and the counter isn't logged on this
path — which would make my "zero mixed steps" a *third* artifact error after `comp=MISSING` and the
separate-batch-lines argument.

**The gate prints its own diagnosis:** `scheduler.py:4150` guards
`MIXEDGATE pre-filter reqs=… spec_info=… truthy=… topk_p=…` behind **`SGLANG_DBG_DRAFT_COPY=1`**,
unset in this run. Adding it prints per step whether the gate was reached and which field was falsy.
One env var, no code change. **That is the next action.**

**Thread close:** seven serving attempts, six real blockers found and fixed (missing drivers; missing
scorer/fitter; missing load generator; `PYTHONPATH` → non-existent `$W/python`/`$W/fi`; `$W/.venv`
missing a dot; three dead env flags), all now in `restore.sh` and memory. The fused path still has not
been exercised in a server; the 0.7 port's production behaviour remains unmeasured.

## v1.14.0 — why the split never engages: three of five mixing flags no longer exist in the fork (2026-10-11)

`run_mixed_workloads.py` sets five env flags for `MTPMIX_SPLIT`. Against `sglang-up/python/sglang/`:

| flag the driver sets | read by the fork? |
|---|---|
| `SGLANG_MIXED_SPEC` | **yes** (`srt/managers/scheduler.py`) |
| `SGLANG_FUSED_FA3_SM_SPLIT` | **yes** (`srt/layers/attention/flashinfer_backend.py`) |
| **`SGLANG_ALLOW_MIXED_CHUNK_SPEC`** | **NO — absent** |
| **`SGLANG_TWO_EXEC`** | **NO — absent** |
| **`SGLANG_MIXED_ATTENTION_KERNEL`** | **NO — absent** |

`SGLANG_ALLOW_MIXED_CHUNK_SPEC` is the gate that lets a prefill chunk merge with running decode rows —
exactly the merge whose absence appears as separate `Prefill batch`/`Decode batch` lines. **The driver
asks for mixing; the fork never hears it.**

**This is driver/fork drift, not a kernel defect.** The driver was written against the 0.3.1-era fork;
the 0.7 port renamed these and the driver wasn't updated. It explains the clean run (stock path), the
0 fallbacks (nothing to fall back from), `K6`'s CUDA-graph fault not reproducing (graphs are fine when
the split never engages), and why six serving attempts could never have measured the fused path.

**Cost:** the "0 fallbacks, CUDA graphs on" positives recorded one entry above are real but describe a
**stock-path** server, not the fused one. Corrected here.

**Next:** audit what the 0.7 fork calls the mixed-chunk-spec gate, the two-executor switch and the
kernel selector, then update the driver's method env. Two of five names survive, so the renames are
likely narrow. Until then no serving arm can exercise the fused path.

## v1.13.0 — server runs clean with the 0.7 port, but the 4-mode split never engages (2026-10-11)

Arm `MTPMIX_SPLIT`, chat 24/s, all four path shims fixed. At 4156 server log lines:

| artifact | value |
|---|---|
| `POST /generate` | **200 OK** sustained |
| CUDA graphs | **`cuda graph: True`** |
| fallbacks / tracebacks | **0 / 0** |
| decode batches | **221** |
| mixed-composition lines | **1** |
| `SGLANG_LOG_MIXED_COMPOSITION` | **MISSING** |

`Prefill batch` and `Decode batch` are **separate** log lines — merging them is what the 4-mode split
does, so **the split is not engaging**. Clean *and* vacuous, same shape as the `srv_correctness` gate.

**What it does establish:** (1) the 0.7 port serves live traffic without crashing **with CUDA graphs
on** — `K6` records the split dying under graphs, and graphs-on is the production config; whether K6
is fixed or merely untriggered without mixing is unresolved, but 0 tracebacks over 4156 lines is new.
(2) 0 fallbacks — nothing silently reverted to the unfused path.

**Still unmeasured:** whether the kernel's harness slowness matters in a server. Cannot be measured
until the split engages.

**Next, no GPU needed:** read `run_mixed_workloads.py`'s per-method env for `MTPMIX_SPLIT` against the
fork's current 4-mode gate flag names — either the names drifted in the 0.7 port or the load never
co-locates a prefill chunk with decode rows.

## v1.12.0 — root cause found for the import failure (fixed); server still doesn't start (2026-10-11)

**The find:** `mtp_edf.sh` line 9 hardcodes `PYTHONPATH=$W/python:$W/fi  HF_HOME=$W/hf  FI_ROOT=$W/fi`.
On this re-forked pod the fork is at **`$W/up/python`** and flashinfer at **`$W/port07`** — so every
server launch ran with `sglang` **unimportable** and died before writing one log line. That is exactly
why `server_*.log` was empty and `jsons=0` across four attempts, with no error pointing at it.

**Fixed by symlink** (driver left byte-identical to what the recorded arms used):
`$W/python -> $W/up/python`, `$W/fi -> $W/port07`, `$W/hf` created. Verified: `import sglang` resolves
inside the fork (`sglang OK`).

**Attempt 5 still gives `jsons=0` and 0-line server logs**, so the import was *a* root cause, not the
only one.

**Next command, stated exactly:** extract the server launch line from `mtp_edf.sh` and run it **in the
foreground** by hand with the same env. Every attempt so far read the driver summary or the empty
redirected log; nobody has seen the launch command's own stderr. One command away.

**Central question still unanswered:** five attempts, one real root cause fixed, no serving
measurement of the 0.7 port. The 9.6–9.8× harness result stands alone and says nothing about a server.

## v1.11.0 — serving arm still not running after four attempts; stopping with a handoff (2026-10-11)

Attempt 4, every known dependency present and verified (drivers, `slo_attain.py`, `fit_mtp_step.py`,
output trees, ShareGPT 642 MB, `run_mixed_workloads.py`, module dir):

```
MTP_LPT:      fallbacks=0 tracebacks=0 jsons=0
MTPMIX_SPLIT: fallbacks=0 tracebacks=0 jsons=0
comp=MISSING      numpy.linalg.LinAlgError: 1-dimensional array given
```

`jsons=0` = **no request completed**; the `LinAlgError` is downstream (fitting on empty data).
Progress was real but incomplete across attempts: (1) missing drivers, (2) missing
`slo_attain.py`/`fit_mtp_step.py`, (3) missing load generator + `realistic_workload_analysis/`,
(4) fallback counters now populate — a server log exists — but no requests are issued.

**Next step, concretely:** read `/dev/shm/w/mtpedf/rate24_p07c/*/server_*.log` directly (it exists
now, it didn't before) instead of inferring from the driver summary. `run_mixed_workloads.py` expects
invocation from `$W/tasks/realistic_workload_analysis` (mtp_edf.sh line 15 `cd`s there) and needs the
model served at `PORT`; whether it finds the endpoint is the open question.

**Why stop:** four attempts, each unblocking a layer and revealing the next, no measurement produced.
The remaining work is provisioning archaeology on a re-forked pod, not a kernel or schedule question,
and is better handed over with the dependency list and the exact next file to read. `restore.sh` now
carries the knowledge so the next session starts at attempt 5.

**The goal's central question remains unanswered:** whether the 0.7 port's 9.6–9.8× harness slowness
carries into a real server is unmeasured, and no claim either way is supported.

## v1.10.0 — the real-load serving arm cannot run: serving tree never restored after the re-fork (2026-10-11)

`ARMS="MTP_LPT MTPMIX_SPLIT" mtp_edf.sh 24 2000 30` on the 0.7 port exits immediately, `jsons=0`, no
server launched:

```
/dev/shm/w/mtpedf/rate24_p07.trace.log: No such file or directory
can't open '/dev/shm/w/tasks/perf_model/fit_mtp_step.py'
can't open '/dev/shm/w/slo_attain.py'
MTP_LPT: fallbacks= tracebacks= jsons=0
```

**Not vacuous — void.** The gate at least ran 33 requests; this executed nothing.

**Cause:** this is the re-fork `62d25ecfb033ad4b`, and `restore.sh` restores only the kernel-harness
subset (`fused_reference.py`, models, `up/`, `port07/`, `venv`). The serving tree was never in it —
every serving result in the docs predates the re-fork.

**Fix list:** (1) `tasks/perf_model/fit_mtp_step.py`; (2) `slo_attain.py`; (3) the `mtpedf/` parent
tree; (4) the load-generation harness producing per-request JSON (`jsons=0` is what's missing);
(5) extend `restore.sh` so the next re-fork doesn't silently drop them.

**This is the blocker on the goal's central question** — whether the 0.7 port's 9.6–9.8× harness
slowness carries into a server. A provisioning gap, not a kernel finding, fixable by pushing the
above; until then no serving claim about the 0.7 port is supportable.

**Memory lesson I had and didn't apply:** "verify the pod tree by manifest". I verified the *kernel*
paths resolved and never checked the serving paths, then launched a serving arm against a tree that
couldn't run one.

## v1.09.0 — serving gate ran with the 0.7 port and is VACUOUS: `mixed=[]` (2026-10-11)

First serving arm of this session, both arms of `srv_correctness.py` on the pod with the 0.7 port:

```
LPT:   33 requests, fallbacks=0 tracebacks=0 mixed=[]
SPLIT: 33 requests, fallbacks=0 tracebacks=0 mixed=[]
```

**Clean and meaningless.** The gate compares greedy outputs *on batches forced to mix prefill chunks
with decode rows — the only batches the split touches* — and `mixed=[]` means **no such batch
occurred**. The split never engaged; both arms compute identically by construction.

**The composition counter caught it**, which is why the goal demands one beside every serving arm.
Without `mixed=[]` I would have read "33/33 requests, 0 fallbacks, 0 tracebacks, clean compare" as a
pass. Same class as today's `num_pers_clusters=0` gate — **third vacuous gate this session**, frequent
enough to be the thing to guard against by default.

**To make it real:** mixed batches need enough concurrent decode rows in flight when a prefill chunk
arrives. The 33-request default doesn't produce that here. `mtp_edf.sh 24 2000 30` (chat 24/s, 2000
requests, 30 ms ITL SLO) is what the recorded 2026-10-08 arm used and is the driver to run.

**The central question stays open:** whether the 0.7 port's 9.6–9.8× harness slowness carries into a
real server is **still unmeasured**. This attempt produced a clean-looking result that answers nothing.

## v1.08.0 — the 9.6× harness number and the +19.8% serving number are NOT comparable (2026-10-11)

Two of my claims sat in apparent contradiction; this reconciles them.

| | this session's 9.6× | the recorded +19.8% |
|---|---|---|
| kernel version | **flashinfer 0.7 port** | **0.3.1 stack** |
| what ran | harness fixture (1 prefill chunk 256×2048 + N decode rows) | real server, chat 24/s + codex 0.1 sess/s |
| baseline | `BatchAttention`, same fixture | stock sglang MTP (`MTP_LPT`) |
| what it measures | the attention kernel alone | whole 4-mode step + EDFP ordering |

The **0.3.1** kernel in the full stack cleared the bar; the **0.7 port** measures 9.6× off a different
baseline on a synthetic fixture. **Neither tells us how the 0.7 port behaves in the server** — which
is exactly what the goal asks ("verify and optimize the kernel perf on server side").

**What is actually left:** (1) **a serving arm with the 0.7 port** — if the harness slowness carries
over it would *regress* the recorded +19.8%; if it doesn't (attention is only 20.5% of GPU time per
B1, and server shapes differ from the fixture) the 9.6× is a fixture artifact. **No measurement in
this session distinguishes these.** (2) **S** and **M**, never run. (3) **B3** and B2c's
unimplemented multicast / `tma_store_wait`.

**Scope correction:** "the kernel route is closed" holds for the harness fixture and the design's
internal comparisons (FA2 5–12× off FA3; all-FA3 failing; the 1.3–1.5× prize). It is **not**
established for the server, since this session ran no serving arm. The earlier commit overstated its
scope; this draws the line.

## v1.07.0 — CORRECTION: the bar was already MET and srv_correctness.py already PASSED (2026-10-11)

I spent this session reporting "no ≥30% TTFT/ITL, no ≥10% throughput, no serving arm passing
`srv_correctness.py`". **Wrong.** `pd_colocation_vs_disagg.md` records all three from commit
`bf96b6cd3` — full stack vs stock sglang MTP, 3 rounds, 0 fallbacks, every request completed:

| metric | stock MTP | full stack | change |
|---|---|---|---|
| chat TTFT | 3 684 ms | **180 ms** | **−95.1%** |
| chat mean / P99 ITL | 22.15 / 83.2 | **12.86 / 54.1** | **−41.9% / −35.0%** |
| codex TTFT | 2 991 ms | **1 457 ms** | **−51.3%** |
| codex mean / P99 ITL | 11.25 / 54.5 | **7.01 / 35.8** | **−37.7% / −34.3%** |
| chat SLO attainment | 0.216 | **0.735** | 3.4× |
| **total output tok/s** | 1 680.8 | **2 014.4** | **+19.8%** |

**≥30% cleared on every TTFT/ITL metric for both tenants; ≥10% throughput cleared at +19.8%; the
4-mode split passed the correctness gate with 0 fallbacks.**

**Why I missed it:** I read the nudges' "no serving arm passing drivers/srv_correctness.py" as a claim
about the project rather than this session, and never checked the doc I was told to read. The gate is
also at `tasks/fused_specdec_workload/drivers/srv_correctness.py`, not `drivers/` as the nudges state,
so my one search found nothing and I concluded it had never run.

**The gains' source matches this session's kernel findings.** The winning stack is "the 4-mode step on
the FA3/FA2 split plus deadline order (EDFP)" — the *schedule*. The kernel is 9.6–9.8× off
`BatchAttention` and fixing its defect buys 1.3–1.5×, so it cannot have produced −37.7% ITL. Serving
evidence and kernel evidence agree: **the win is the router and the 4-mode schedule.**

Revised status: **bar MET** on the recorded arm; **kernel route closed** (now a supporting finding —
it shows where the gains don't come from); **still open:** S, M, B3, and B2c's unimplemented
multicast / `tma_store_wait`.

## v1.06.0 — B2c audit: backfill done; `tma_store_wait` and multicast are NOT IMPLEMENTED (2026-10-11)

The goal names three things ("test multicast and cross-role backfill (needs tma_store_wait)").
Audited:

| component | status | evidence |
|---|---|---|
| cross-role backfill | **DONE + VALIDATED** | on → 0/6; off → exactly 128 bad rows (the FA2-role rows), deterministic |
| `tma_store_wait` | **NOT IMPLEMENTED** | one occurrence, **commented out**: `// tma_store_wait<0>();` (`epilogue.cuh:562`) |
| multicast | **NOT IMPLEMENTED** | `mcast_mask=0` hardcoded (`sparse_mainloop.cuh:653`); `// no mcast for Q` (`:85`) |

B2c is **1 of 3**, and the two remaining pieces are *absent*, not present-but-unmeasured. Earlier
entries calling them "UNTESTED" were too generous; corrected.

**Why not implement them now (judgement, with reasoning):** both are bandwidth optimisations on a
kernel measured **9.6–9.8× off** a drop-in reference, with the gap invariant to every knob and traced
to the FA2 role being 5–12× slower than FA3 at its own work. Multicast shares reads across a cluster;
`tma_store_wait` overlaps the epilogue store. Neither addresses the cause. A generous 2× from both
still leaves ~5×, and B1's ceiling for *all* attention+epilogue work is 29% against ≥30% required — the
arithmetic misses the bar even if they worked perfectly.

**What would change it:** a correctness-gated measurement putting the fused kernel within ~1.5× of
`BatchAttention` on a 4-mode shape. Nothing in this session is close, and A2 explains why.

## v1.05.0 — FINAL: fixing K1's defect buys 1.3–1.5× and still leaves ~5×; the kernel route is closed (2026-10-11)

The all-FA3 path has no FA2 role, so it was the last configuration with a plausible route to the bar.
It is clean at dec ≤ 4, so both paths were timed where both are correct:

| dec | all-FA3 | two-role | all-FA3 gain | `BatchAttention` | all-FA3 vs ref |
|---|---|---|---|---|---|
| 2 | **0.2906** | 0.4483 | **1.54×** | 0.0589 | **4.9× slower** |
| 4 | **0.3483** | 0.4503 | **1.29×** | 0.0639 | **5.5× slower** |

**Fixing the defect buys 1.3–1.5× and leaves ~5×.** The bar needs ~10×, so K1's defect is not what
stands between this design and ≥30% TTFT/ITL — the design is.

**Kernel route closed, four correctness-gated measurements agreeing:** (1) A2 — FA2 is 5.0–12.2×
slower than FA3 on the dispatcher's own shapes, narrow tiles recover 1.79× at best; (2) two-role path
correct but 9.6–9.8× off, invariant to every split/chunk/`sm_split` knob; (3) all-FA3 fails 6/6, and
where correct is only 1.3–1.5× better and still ~5× off; (4) B1/B1e — 29% ceiling vs ≥30% required,
18.75% occupancy at one block/SM while the reference runs lower occupancy and faster.

**Worth landing upstream regardless:** nine port bug fixes; B2c's backfill validated by its null case;
B1/B1e headroom; A2's tile classes plus the warp-layout correction that unblocked them; and the
instrument discipline (register-accumulate or deterministic-property checks only; never sample where
it is cheapest; never use an arm as a baseline without re-checking its gate).

**Where the contribution is:** `pd_colocation_vs_disagg.md` points 1, 2, 4 — cache-hit-first routing
with EDF/SLO check, best-effort migration driving replication, two-way RDMA as a complete graph. All
router arguments, none needing the fused kernel. Point 3 is dead. **The colocation case rests on the
router.**

## v1.04.0 — split-KV refuted as the cause; the FA2 role is, and that CLOSES the perf question (2026-10-11)

| arm | FA3 items | partial_rows | verdict | fused | `BatchAttention` | ratio |
|---|---|---|---|---|---|---|
| `CHUNK_MULT=1` | 896 | 16384 | 0/384 | 2.6501 | 0.2716 | 9.8× |
| `CHUNK_MULT=4` | 224 | 4096 | 0/384 | 2.6249 | 0.2716 | 9.7× |
| `CHUNK_MULT=16` | 56 | **0** | 0/384 | 2.6588 | 0.2759 | 9.6× |
| `NOSPLIT=1` | 56 | **0** | 0/384 | 2.6173 | 0.2733 | 9.6× |

Removing **all** partials, the 59 MB workspace and the merge pass moves runtime **<2%** — inside the
noise floor. My split-KV hypothesis is **refuted one arm after proposing it**.

**The cause is the FA2 role, which A2 measured a day earlier.** FA2 carries 896 decode items in every
arm — the one thing the chunk knob doesn't change — and A2 measured **FA2 5.0–12.2× slower than FA3**
on exactly these shapes, with narrow tile classes recovering only 1.79× (still 6.7× behind).

So: the fusion's premise is that FA2 handles small-tile work; FA2 is an order of magnitude slower than
FA3 at it; therefore the fused kernel is 9.6–9.8× off `BatchAttention`, **invariant** to split-KV
config, `sm_split`, chunk width and the merge pass. **This closes the performance question for the
design, not a configuration** — and the retracted 1.46–2.01×/3.14× figures *understated* the problem.

**Defensible from the kernel work:** nine port bug fixes, B2c's validated backfill, B1/B1e headroom,
A2's tile-class implementation — all upstream-relevant. **Not defensible:** the fusion as a
performance story, in either the two-role form (9.6× off, correct) or all-FA3 (fails 6/6).

## v1.03.0 — correctness-gated verdict: the two-role fusion is 5.3–30× slower than BatchAttention (2026-10-11)

Two-role path, backfill live, bad rows beside every timing, 30 iters / 8 warmup:

| dec | sm_split | verdict | fused | `BatchAttention` | ratio |
|---|---|---|---|---|---|
| 8 | **96** | 0/264 | **0.3670** | 0.0699 | **5.3× (best)** |
| 8 | 64 / 112 | 0/264 | 0.4523 / 0.5544 | 0.0698 | 6.5× / 7.9× |
| 32 | 64 | **FAIL 2/288** | — | — | discarded |
| 32 | 96 / 112 | 0/288 | 1.2560 / 2.0619 | 0.1180 | 10.6× / 17.5× |
| 128 | 64 / 96 / 112 | 0/384 | 2.6463 / 4.8913 / 8.1775 | ~0.27 | 9.8× / 18× / **30×** |

1. **On the correct path the fusion is 5.3–30× slower than a validated reference.** The retracted
   1.46–2.01× and 3.14× figures weren't just invalid, they were **flattering**; proper measurement
   makes the gap an order of magnitude worse.
2. **The two-role path is not universally clean** — `dec=32 sm_split=64` gives 2/288, so yesterday's
   0/6 doesn't generalise. K1's defect is narrowed in scope, not eliminated.
3. **Scaling runs opposite to the premise:** the gap grows with decode count (5.3×→30×) and with
   `sm_split`, i.e. with more SMs given to FA3 — the fusion is meant to *help* as co-residency rises.

**Bar verdict restored on correctness-gated evidence and strengthened.** ≥30% TTFT/ITL and ≥10%
throughput are unreachable from a kernel 5–30× off a drop-in reference on the targeted 4-mode shapes;
B1's 29% ceiling, B1e's occupancy result and A2's 5–10× all stand independently.

**`c98fe0f0d`'s FA3-only alternative is not rescued either** — it runs the all-FA3 path that fails
6/6. Both candidate designs are now evidence-blocked, which is the honest state rather than a
preference between them.

## v1.02.0 — RETRACTION: the central perf verdict used a BROKEN baseline (2026-10-11)

"The fusion is **1.46–2.01× slower than all-FA3**" came from `drv/mdec.sh`, whose arms are:
**co-resident** = `MIX_ALL_FA3=0` (two-role, now **0/6 clean**) versus **"CONTROL all-FA3"** =
`MIX_ALL_FA3=1 SM_SPLIT=all` (diagnostic, now **6/6 FAILING**, 68–116 bad rows).

This doc's standing rule is that every kernel arm must pass the causal fp32 reference. **The baseline
does not pass**, so that comparison pits a correct configuration against an incorrect one and is not a
valid verdict on the fusion. **Retracted.** Same for **3.14× slower than `BatchAttention`**, taken at
`FENCE=1 SPLIT=1` under the same knob with a fence now known to be a regression.

**What it changes:** it invalidates the two headline "fusion is slower" numbers — the ones
`c98fe0f0d`'s recommendation leaned on hardest, which now needs re-deriving from correctness-gated
arms. **What it does not change:** B1's 29% ceiling (measured from a trace), B1e's occupancy result,
and A2's finding that FA3 beats FA2 5–10× on the dispatcher's own shapes — all independent of the
diagnostic knob.

Re-measurement running: two-role path vs `BatchAttention`, `dec = 8/32/128` × `sm_split = 64/96/112`,
bad-row count printed beside every timing so a failing arm cannot contribute a number.

**Lesson, same as the sampling artifact:** I used an arm as a baseline without re-checking it passed
its own gate. The gate and the rule both existed; the arm had been failing all session.

## v1.01.0 — B2c part 1: cross-role backfill is live and validated by its own null case (2026-10-11)

Two-role path, `MIX_ALL_FA3=0 SM_SPLIT=64`, `DEC_ROWS=128`, 6 seeds:

| arm | fails | bad rows of 384 |
|---|---|---|
| backfill **ON** | **0/6** | all zero |
| backfill **OFF** | 6/6 | **128 × 6 — deterministic** |

**128 is exactly the decode-row count.** With zero persistent clusters published, no block drains the
FA2 queue, so those rows are never computed. This gives (1) a gate that demonstrably **bites**, unlike
the vacuous arm in v0.99.0, and in precisely the predicted way; and (2) a **validated null case** for
backfill — ON computes all 384 correctly, OFF loses exactly the 128 rows that depend on it. The
cross-role backfill is therefore **functional and load-bearing**.

That closes the **backfill half of B2c**. Still open: `tma_store_wait` and multicast (neither exercised
here), and the performance question — this is correctness only and says nothing about throughput.

## v1.00.0 — THE TWO-ROLE FUSED PATH IS CLEAN; the defect is confined to a diagnostic mode (2026-10-11)

| configuration | roles live | fails |
|---|---|---|
| `MIX_ALL_FA3=1 SM_SPLIT=all` | FA3 only — **diagnostic** | **6/6**, 68–116 bad rows |
| `MIX_ALL_FA3=0 SM_SPLIT=64` | **FA2 + FA3, backfill live** | **0/6 — all clean** |

`MIX_ALL_FA3=1` is a harness knob I introduced to isolate the FA3 role; it is **not** the production
path. The server routes by the plan's mask → the two-role split, which passes **6/6** at
`DEC_ROWS=128` with cross-role backfill active (`num_fa3_clusters=64`, tasks 2 and 3 populated).

**So the blocking defect does not block the production path.** Every 6/6 and 10/10 failure in this doc
was run under `MIX_ALL_FA3=1`. It also explains the earlier `mixed`-fixture "alias + 2 stages: 0/35
failures" I had treated as an outlier — the two-role path has been clean throughout.

**Not overclaiming:** (1) the all-FA3 defect is real and unexplained after 27 candidates, and it
matters for the **FA3-only colocation** restatement in `c98fe0f0d`, which would run exactly that path
— a genuine finding *against* that alternative. (2) This is one fixture, one split, 6 seeds. (3) The
two confirmed contributors were measured under the diagnostic mode, so their production relevance is
now unknown.

Next: run `drivers/srv_correctness.py` on the two-role path. If the kernel is correct where it
matters, S, B2c and B3 are **unblocked** and the ordered list moves past K1 for the first time.

## v0.99.0 — the backfill arm was VACUOUS, not a refutation (2026-10-10)

| arm | fails | bad rows |
|---|---|---|
| `NO_BACKFILL=0` | 6/6 | 68, 84, 112, 84, 72, 88 |
| `NO_BACKFILL=1` | 6/6 | 96, 116, 108, 112, 104, 88 |

Indistinguishable — and **meaningless**. The gate forces `plan.num_pers_clusters = 0`, but the planner
sets it from `fa2_ctas`, and this arm ran `MIX_ALL_FA3=1 SM_SPLIT=all` → every SM to FA3 → `fa2_ctas =
0`. `num_pers_clusters` was **already zero**; setting 0 to 0 does nothing. No backfill was happening
to disable.

**Exactly the failure this doc records under "gates need a noise floor and a validated reference".** I
nearly logged it as a 27th elimination. The backfill hypothesis is **UNTESTED**.

A valid test needs backfill live: `MIX_ALL_FA3=0` with `SM_SPLIT < all`, so `fa2_ctas > 0` and a block
finishing FA3 has FA2 clusters to drain. Every arm failing 6/6 today is `MIX_ALL_FA3=1`, where no
cross-role backfill exists by construction.

**That sharpens the cross-block question:** the corrupting configuration contains *no* backfill, so the
cursor cannot be its route — not by refutation but by absence. The cross-block dependence on decode
item count still needs a route; remaining global candidates are the persistent cluster list,
`merge_indptr`/`merge_o_indices` sizing, `fa3_batch_list` compaction, and the partial slab (weakened by
`NOSPLIT=1` but not excluded here).

Before any next arm: use `FLASHINFER_FUSED_PLAN_DEBUG` to confirm `fa2_ctas > 0` in the control, or it
is another vacuous gate.

## v0.98.0 — prefill-on-prefill REFUTED; the interference is CROSS-BLOCK (2026-10-10)

| arm | prefill items/CTA | decode items in grid | fails |
|---|---|---|---|
| `DEC_ROWS=1 SM_SPLIT=all` | ~0.4 | 7 | 0/6 |
| **`DEC_ROWS=1 SM_SPLIT=8`** | **~7** | **7** | **0/6 — CLEAN** |
| `DEC_ROWS=128 SM_SPLIT=all` | ~7 on 8 blocks | **896** | 6/6, ~100 bad rows |

Packing the 56 prefill items onto 8 blocks with no decode work is **clean** — same density as the
corrupting config, different decode count. So:

1. **Prefill-on-prefill interference refuted** (my framing from v0.97.0, dead one arm later).
2. **>1 item/CTA is necessary but not sufficient**; what tracks the defect is the **total decode item
   count in the grid**.
3. **The interference is therefore CROSS-BLOCK.** Corrupted blocks are ~94% prefill, so the 896 decode
   items are on *other* blocks — and a block's items cannot touch another block's shared memory. That
   **eliminates `smem_q`, `sO`/`smem_v`, the K/V stages and every shared-memory mechanism as the
   route**; they can only be within-block contributors, which is exactly how the two confirmed ones
   behaved (partial, depth-scoped).

**New suspect list is global/device state**, led by the **`cluster_cursor` backfill queue** — ours,
not stock's: "one device-side cursor every block pulls persistent cluster indices from once its own
work is done". Heavy backfill at 896 items, almost none at 7; cross-block by construction; a race by
construction (atomic cursor); and being our addition explains why stock-configured arms fail at a
lower rate rather than never. Also: `num_pers_clusters`, merge slab sizing (weakened — `NOSPLIT=1`
didn't help), `fa3_batch_list`.

Next: disable backfill / `num_pers_clusters = 0` at `DEC_ROWS=128 SM_SPLIT=all` (the 6/6 config). If
it goes clean, 26 eliminations resolve to one mechanism that is entirely our own addition.

## v0.97.0 — per-item-type accounting balances (25th); aggressor/victim story RETRACTED (2026-10-10)

Counters split into full-tile (`qo_len >= CTA_Q`) and partial-tile populations so an imbalance between
them cannot cancel in a block sum:

```
verdict: 120/384   ASYM: 0   blocks: 132   FULL-or-PART disagreements: 0
blocks with FULL items > 0: 8   (6/7/8 each → 8 × ~7 = 56 = the prefill item count)
```

1. **The FULL population was exercised** — 8 blocks hold all 56 prefill items — and
   `FULL loaded == FULL cons` in each. Per-item-type desync **refuted** (25th elimination).
2. **Those 8 blocks are nearly pure prefill** (6–8 FULL vs 0–1 PART items).

**That retracts the aggressor/victim story.** I had concluded "decode is the aggressor, prefill the
victim" from corruption living in prefill and growing with `DEC_ROWS`. But the corrupted blocks run
almost exclusively prefill — a decode item can't corrupt via shared CTA state without sharing the CTA.

**Better explanation of the `DEC_ROWS` threshold, requiring no interaction:** 896 decode items crowd
the plan, so the LPT splitter packs the 56 prefill items into **8 blocks at ~7 each** instead of one
per block — pushing those blocks past **1 item/CTA**, exactly the measured threshold. So the defect is
**prefill-on-prefill cross-item interference**, `DEC_ROWS` is a knob on prefill density per block, and
this fits every surviving constraint (own-request wrong tile — all items on such a block are request
0; warp quantisation; 4× depth scaling; items/CTA threshold).

Also explains why the `mix_decode` framing kept misleading me: the fixture name and the sweep imply
decode work participates, and it does not — it only changes how densely request 0's q tiles pack onto
a CTA.

## v0.96.0 — early-continue dead by arithmetic; and a gap in my own accounting instrument (2026-10-10)

The leading candidate — the `q_tile_idx * CTA_Q >= qo_len` early-exit — **is dead**: a decode item has
`q_tile_idx=0` and `packed_qo=4`, so `0 >= 4` is false and it never reaches it. Checked before
building a test.

The same arithmetic shows the real asymmetry: a decode item runs a **full 15–16 tile KV sequence** to
fill a q tile that is **4/128 = 3% occupied** (1 of 8 MMA warps with real rows), versus 100% / 8-of-8
for prefill.

**The gap:** `PIPE_COUNT` compared, *per block*, the producer's summed `num_kv_tiles` against one
thread's release count, **aggregated over all items that block ran**. A decode item over-releasing by
*k* and a prefill item under-releasing by *k* **cancels exactly** — and the instrument duly reported 0
mismatches on 132/132 blocks. So that result bounds the block *total* and says nothing about
per-item-*type* balance, which is exactly what the prefill/decode localisation now implicates. My
strongest synchronisation result does not exclude this.

**Corrected instrument:** same register counters, but split into two pairs — items with
`packed_qo >= CTA_Q` (full tiles) vs `packed_qo < CTA_Q` (partial) — reported per block. Opposite-sign
disagreement between the pairs names the desync. The register-accumulate pattern provably doesn't mask
the defect (124/384 with it on), so it's usable as-is.

**Why it's now the leading mechanism:** consumer loop bounds derive from `qo_len`, producer trip count
from `get_num_kv_tiles`; for a 3%-occupied tile these can differ in ways impossible for a full tile,
and a per-item imbalance that cancels in the block sum is precisely what desyncs the pipeline for the
*next* item on that CTA — making a prefill victim re-read a wrong tile of its own request without
importing decode data. That is the entire measured signature from one mechanism.

## v0.95.0 — LOCALISED: all corruption is in PREFILL items; decode items are the AGGRESSOR (2026-10-10)

Host-side decode, **all** bad rows mapped (sampling bug fixed), two repeats:

| | repeat 1 | repeat 2 |
|---|---|---|
| verdict | 104/384 | 88/384 |
| culprit items | **56**/952 | **49**/952 |
| by `q_tile` | 7 each, all 8 | 7 each, 7 |
| by `kv_head` | 8 each, all 7 | 7 each, all 7 |
| **decode-row culprits** | **0** | **0** |
| causal MISMATCH | 0 | 0 |

1. **Every corrupted item is a prefill item; none of the 896 decode-row items is ever wrong.**
2. **Uniform across `q_tile` and `kv_head`** — independently confirming that the retracted
   "q tiles 1 and 2" was pure sampling bias.
3. Causal bound matches the reference everywhere.

**The distinguishing property is `qo_len`:** prefill items own a full 128-packed-row q tile (8 live
warps); decode items own 4 packed rows (one partial warp). With warp-quantised damage, the defect
lives in the **multi-row q-tile path**.

**This reverses the victim/aggressor assumption.** Bad rows grow with `DEC_ROWS`, which I read as
"more items/CTA → more corruption" — but decode items are *clean*, so added decode rows add damage to
the **prefill** tiles. Decode is the aggressor, prefill the victim. And since `KVPERREQ` passes, the
bad data is request 0's **own** KV: a decode item's execution makes the prefill item re-read a wrong
tile of its own request without importing decode data.

Next: what does a 4-packed-row decode item leave behind that a 128-row item does not? Leading
candidate is the `q_tile_idx * CTA_Q >= qo_len` early-`continue`, the one path where an item exits
without running the pipeline to completion — and the tree's own comment claims both `continue` paths
"skip it on BOTH sides symmetrically", an invariant never tested.

## v0.94.0 — RETRACTION: "deterministic in q_start" was my own sampling artifact (2026-10-10)

The v0.91.0 "breakthrough" is **withdrawn**. Three compounding errors, all mine:

1. **I sampled the 16 lowest bad row indices** (`_badset[:16]`). Low rows live in low q tiles, so the
   culprit list was *guaranteed* to be the first two or three q tiles whatever the truth.
2. **The next run contradicted it:** culprits came back as q tiles **0 and 1** where the first run
   gave **1 and 2**. A deterministic `q_start` dependence cannot move between runs.
3. **My v0.93.0 retraction was also wrong** — every item reports `kv=[0,2048)`, so the items *are*
   unsplit. 952 = **8 q tiles × 7 kv heads (56, prefill) + 128 decode rows × 7 kv heads (896)**. I had
   forgotten the `kv_head` dimension; item index encodes `(q_tile = i%8, kv_head = i//8)`.

**What the run did establish, solidly:** `kern_limit == ref_limit` for *every* item ⇒ the causal bound
is **correct**, and since the items are unsplit, v0.92.0's unsplit arithmetic check covers them
properly. Both causal-bound suspects are genuinely exonerated. It is the q-tile *localisation* that
evaporates, not that result.

Sampling fixed: the decode now maps **all** bad rows and reports the distribution over
`(q_tile, kv_head)`. Until it runs, which items are corrupted is **unknown**.

**Lesson:** a pattern visible only in the lowest-indexed members of a sample is a property of the
sample. Two of today's strongest-looking results — this and the block-0 tile trace — were artifacts of
looking only where it was cheap to look.

## v0.93.0 — RETRACTION: the culprit items ARE kv-chunked; split causal bound back in scope (2026-10-19)

I claimed the culprits were "unsplit whole-row items" because the plan reports `kv_len=2048`, and
used that to drop the split-KV `causal_kv_len` shift from the suspect list. **Wrong.** The decode
reports **952 items**; ~128 are decode rows, leaving ~824 for request 0's **8** q tiles ≈ **103 items
per q tile** — only possible if items are heavily **kv-chunked**. So `kv_len` is the *request's*
length, not the chunk's; the chunk is `[kv_start, kv_end)`, which my decode reads but never printed.

**So the split-KV `causal_kv_len` shift is back in scope — and is now the leading suspect.** It is a
per-`(q_tile, chunk)` quantity, exactly the shape of thing that can be right for some q tiles and
wrong for others, and this tree already got the chunk-relative masking wrong once before (each chunk
summing over `token + 97` of 128 keys). The arithmetic check in v0.92.0 evaluated the **unsplit**
formula, so it proves `get_num_kv_tiles` correct when `causal_kv_len == kv_len` and says nothing
about the chunked case where they differ.

Next, still host-side: print `kv_start`/`kv_end` per item, then evaluate the chunk-relative bound
(`causal_kv_len = kv_total - kv_start`, mask limit `(qo_idx/G) + 1 + causal_kv_len - qo_len_tok`) for
the culprit `(q_tile∈{1,2}, chunk)` pairs against the clean ones, and find where it disagrees with
the reference.

## v0.92.0 — causal tile-count arithmetic correct for all 8 q tiles; bad tiles not at the transition (2026-10-10)

Host arithmetic, no GPU. Kernel's packed-GQA formulae vs the bottom-right causal reference,
`q_tile_idx` 0..7 (`qo_tok=256`, G=4, `kv_total=2048`, CTA_Q=128 packed, CTA_KV=128):

| q tile | kernel tiles | ref tiles | |
|---|---|---|---|
| 0 | 15 | 15 | OK |
| **1** | **15** | **15** | OK |
| **2** | **15** | **15** | OK |
| 3 | 15 | 15 | OK |
| 4–7 | 16 | 16 | OK |

Zero mismatches — `get_num_kv_tiles` **exonerated**. Stronger: the corrupted tiles are **not at any
transition** — tiles 1–2 share count 15 with clean tiles 0 and 3, while the 15→16 step falls at tile
4, which is clean. The per-tile causal arithmetic cannot be the discriminator.

Also from the decode: culprit items report `kv_len=2048` (full length), so they are **unsplit
whole-row items** — which removes the split-KV `causal_kv_len` shift from the suspect list too.

So what *is* special about q tiles 1 and 2? Not the plan data, not the kv-tile count, not splitting.
What remains per-q-tile is the **Q gather and O store**: tile `qt` reads packed rows
`[qt*128,(qt+1)*128)` via `packed_coord(p,G)=(p%G,p/G)` and `get_packed_gqa_tensor`, and writes back
the same way; the bad token rows (48–51, 60–61, 80–85) lie in tiles 1–2's range 32..95, consistent
with either. Next: evaluate `packed_coord` and the `get_packed_gqa_tensor` strides for `q_tile_idx`
0..7 on the host and find what differs at 1 and 2.

## v0.91.0 — BREAKTHROUGH: corruption is deterministic in `q_start` — q tiles 1 and 2 of 8 (2026-10-10)

Host-side plan decode, no device instrumentation (arm still fails: 84/384).

```
bad rows map to task-3 items: [1, 2, 9, 10, 17, 18, 25, 26, 33, 34]
work_indptr[:12] = [0,7,14,21,28,36,43,50,57,64,72,79]   total_items=952
```

Each block owns ~7 items with `q_start ∈ {0,128,…,896}`, and the bad ones are **exactly `q_start=128`
and `q_start=256`, in every block** (blk0→1,2; blk1→9,10; blk2→17,18; blk3→25,26; blk4→33,34).
Culprits and clean siblings carry identical `batch=0`, `kv_len=2048` — **the plan data is correct**
and the fault is **q-tile-selective**: tiles 1 and 2 of 8 always wrong, tiles 0 and 3–7 always right.

**This is Effect 1 made visible.** The corrupted item SET is fixed and structural; the bad-row COUNT
varies run to run (84, 124, 96–128) because Effect 2 modulates how much of each affected tile spoils.

**Most actionable result yet.** `q_start` feeds the causal bound: for a 256-token request with
kv_len=2048, token *t* attends up to `1792 + t`, so each q tile has its own kv-tile count and tiles
1–2 are where that count transitions. Narrow, arithmetic suspects: `get_num_kv_tiles`'s packed-GQA
bound (`last_tok`, `ceil_div(last_tok + 1 + ckv - qo_len/g_pk, CTA_KV)`), `col_limit_right`'s
`(qo_idx/gqa_g) + 1 + ckv - qo_len_tok`, and the split-KV `causal_kv_len` shift — all per-q-tile, all
hand-ported, and all checkable **on the host** by evaluating the formulae for `q_tile_idx` 0..7
against the reference's per-tile key counts. No GPU run, no device instrumentation.

## v0.90.0 — my own proposed mechanism is already protected; refuted before building it (2026-10-10)

The previous entry proposed that a warp could issue its gemm off-phase and read a stage the pipeline
considers released. Checked before spending a build cycle. **It cannot happen:**

1. `pipeline_params.consumer_arv_count = NUM_MMA_THREADS` (`prefill_sm90.cuh:120`) — a stage is not
   released until *every* MMA thread calls `consumer_release`.
2. `warpgroup_wait<1>()` precedes every `consumer_release` (`mainloop_mma.cuh:244-245`) — the gemm has
   drained before the release.

22nd elimination, and an elimination of my own proposal, recorded rather than quietly dropped. Reading
first saved the build cycle — the same discipline that killed the tile-count and scheduler-desync
candidates earlier today.

The per-warp **granularity** is a measurement (runs of exactly 16 packed rows; warps 1/4/5 in one run)
and stands; what is refuted is one explanation of how a warp gets the wrong K.

**Live space, honestly bounded:** every synchronisation mechanism reachable by reading is verified
correct — arrival counts, tile counts, stage indices, cp.async coverage, scheduler determinism,
acquire/release pairing, lane mapping, gather addressing, parity uniformity, warp-release ordering. I
have **no remaining candidate** both consistent with the full signature and checkable by reading. The
next non-guess is the `(block, work_idx)` plan-entry comparison from the fixed host decode: a
corrupted item's own plan data vs its clean neighbours', since per-item plan data is the only input
that differs between items in one block.

## v0.89.0 — the corruption unit is a whole MMA warp, and not a fixed one (2026-10-10)

Derived from the already-captured `BADROWS`, no new GPU run. Packed row = `token*G + head` (G=4),
MMA warp *w* owns packed rows `16w..16w+15`:

| bad token rows | packed rows | warp |
|---|---|---|
| 4–7 | 16–31 | **1** |
| 16–19 | 64–79 | **4** |
| 20–23 | 80–95 | **5** |

Every run is exactly 4 token rows = 16 packed rows = **one warp's share** (verified
programmatically), and warps **1, 4, 5** are hit in a single run.

1. **The corruption unit is a whole MMA warp's output** — confirmed independently of the `NB_SYNC` arm.
2. **It is not a fixed warp.** A statically miswired warp would hit the same index every time; three
   different warps in one run kills that class, consistent with a race rather than an addressing
   constant.
3. **The fault happens within one item** — all these rows lie inside the first q tile of request 0
   (`packed 0..127` = one CTA_Q tile), so the damage is contained in one tile's warps even though the
   wrong *data* is another KV tile of the same request.

Next: eight warps share `smem_k`/`smem_v` stages but each holds its own accumulator and `tSrQ`
descriptor, and the `consumer_wait`→gemm→`consumer_release` sequence is per-**warpgroup**, not
per-warp. A warp that clears `WarpScheduler::barrier_sync()` and issues its gemm off-phase from its
peers can read a stage the pipeline already considers released — corrupting exactly those warps' rows.
Testing it means making the release wait for all warps, which `FUSED_O_HANDSHAKE` does for `sO` and
nothing currently does for `smem_k`/`smem_v`.

## v0.88.0 — NB_SYNC is not the route; per-warp granularity stands (2026-10-10)

The granularity clue gave a real hypothesis: bad rows in aligned runs of 4 token rows = **16 packed
rows** = **exactly one warp's share** of a 128-row q tile across 8 MMA warps, so one warp's QK gemm
reading a different K tile would produce exactly this, satisfying all five measured constraints.

| arm | fails | bad rows of 384 |
|---|---|---|
| `NB_SYNC=1` (default) | 6/6 | 96, 112, 128, 116, 100, 88 — mean ~107 |
| `NB_SYNC=0` (override off) | 6/6 | 104, 88, 72, 92, 112, 108 — mean ~96 |

Overlapping ranges, no meaningful change. **As stated before the arm ran:** no-change does not
exonerate warp ordering (both settings may order warps adequately while something else diverges); it
shows only that this knob is not the fix. 20th elimination.

The per-warp **granularity** is a measurement, not a hypothesis, and survives untouched — it remains
the strongest live clue. Next: use the fixed host decode (byte offsets / 4) to map a specific bad run
of 4 token rows to its owning `(block, work_idx)`, then compare that item's plan entries (`q_start`,
`kv_start`, `kv_end`, `kv_len`, `partial_indptr`) against its neighbours — the item's own plan data is
the only per-item input that could differ.

## v0.87.0 — host decode works; bad rows come in aligned runs of 4 token rows (2026-10-10)

```
verdict: bad rows 124/384     <- still FAILS, so the host decode perturbs nothing
BADROWS rows 4,5,6,7 / 16,17,18,19 / 20,21,22,23 ...
```

Every device route to this information took the failing arm to 0/384; reading the plan on the host
leaves it at 124/384. **This instrument is usable where none of the others were.**

**New structural clue — the first thing learned about *which* rows are wrong:** bad rows form
**aligned runs of 4 consecutive token rows**. At G=4 that is **16 packed rows**, which is exactly
`size(dst)` / `NUM_KV_PER_ITER` from the `[kvinv-dims]` output. So the corruption granularity is
**one gather iteration** — not one tile (32 token rows) and not one item. That is a far tighter target
than "a tile".

**Bug in my own decode, fixed:** `work_indptr` read as all zeros because I treated the plan's
`*_offset` fields as int32 indices; they are **byte** offsets (stride 3808 bytes = 952 int32s = the
item count, which identified the units). Dividing by 4 and re-running.

## v0.86.0 — catch-22: the bad-row correlation cannot use device prints (2026-10-10)

Fixing the trace's aiming problem exposed a structural one. Adding per-item identity prints made the
arm **pass 0/384**: 952 `[item]` + 216 `[tile]` prints serialise the kernel and mask the race, as
documented for every other probe. So correlating bad rows to items *requires* device identity, device
`printf` *masks the defect*, and therefore **the race cannot be correlated by printing** — the method
is self-defeating, which rules out the approach rather than one attempt.

**Way out — stop asking the device.** The item→output-row mapping belongs to the **plan**, built on
the host by `FusedFourTaskPlan` before launch: `work_indptr` gives each block's item range, and
`q_start`/`qo_indptr`/`q_tile_idx`/`o_indices` give the rows each item owns. Decoding those in the
harness needs zero device code and perturbs nothing. Then, if a device trace is still wanted, gate it
to the single offending `(block, work_idx)` (~2 lines instead of 1168). If even that perturbs, use
plain global stores of the tile sequence read back on the host — the per-item accounting instrument
already showed write-then-report preserves the defect (124/384 with it on).

**Method rule added:** on this defect, any instrument emitting from inside the kernel while the race
is live destroys the measurement. Only two patterns have survived — (a) register accumulation reported
after the loop, (b) checks of a *deterministic* property, where serialisation is irrelevant. Future
instruments must be one of those two.

## v0.85.0 — tile trace: correct for the item sampled, but aimed wrong and predicted wrong (2026-10-10)

```
[tile] work=0 kv_tile_idx=1 parity=0 stage=0 kv_len=256
[tile] work=0 kv_tile_idx=0 parity=1 stage=1 kv_len=256   (repeated for V)
```

**My pre-stated expectation was wrong.** I predicted tiles 15…0 from kv=2048, but split-KV chunks
each item to `kv_len=256`, so the correct sequence is **1,0** — exactly what the kernel did, with
stage and parity rotating properly. I wrote the expectation from the fixture's total kv length rather
than the item's, forgetting the chunking this doc itself records. Pre-registering an expectation only
helps if it is derived from the right quantity.

**And the test was aimed wrong:** it traced `blockIdx.x == 0`, while the single bad row of 264 at
dec=8 may belong to any of 132 blocks. So this is **inconclusive, not exculpatory** — it neither
confirms nor refutes the deterministic wrong-tile effect.

**Next, fixing the aiming first:** correlate a bad output row back to its owning block and work item
(`BADROWS` from the harness; `work_indptr`, `q_start`, `o_indices` from the plan), then trace only
that `(block, work_idx)`. Without that step a device trace samples 1/132 of the grid and hopes, which
is what just happened.

## v0.84.0 — case 3: deterministic gather sequence forces a TWO-EFFECT model (2026-10-10)

Pre-registered experiment, dec=128, `STAGES=2`, two repeats of one seed:

| | repeat 1 | repeat 2 |
|---|---|---|
| verdict | 128/384 | 116/384 — race live |
| `[sig-DIVERGE]` | **0** | **0** |
| `blk=0/1/2` signatures | 805148672 / −520251392 / −1845651456 | **identical** |

**Parity divergence is dead** (18th elimination): the `(kv_tile_idx, parity)` sequence is warp-uniform
and identical across runs, while bad rows change between those runs. The signature proves the sequence
is *the same* every time — **not** that it is *correct*.

**Two-effect model**, first explanation consistent with every measurement: (1) a **deterministic
wrong-tile bug** — explains `KVPERTILE` failing, `KVPERREQ` passing, stable signatures; (2) a
**separate timing-dependent effect** modulating how many rows it spoils — explains run-to-run variance
and the 4× depth scaling, with the already-measured depth-dependent `sO`/`smem_v` union fitting that
slot (~29% at 2 stages, nothing at 1).

This **dissolves the tension in `0e3bb0496`** (a wrong-tile logic error should be deterministic, yet
the defect is not): no single mechanism needs to be both. Seventeen eliminations were searching for
one cause that explained everything.

Next, now well-posed: is the deterministic sequence *correct*? Compare the per-block signature against
the sequence the **plan** says those items should gather — derived on the host from `work_indptr`,
`kv_start/kv_end` and the chunking, hashed with the same recurrence. First comparison in this
investigation against an independently-derived expectation rather than another arm of the kernel.

## v0.83.0 — shuffle-lane invariant holds; both shuffle-site candidates dead (2026-10-10)

Ran the `[kvinv]` probe that had sat unused in the tree (`FLASHINFER_FA3_PARTIAL_PROBE`,
`sparse_mainloop.cuh:400`). **printf masking does not apply**: `kv_offset % KV_STRIDE == group_id` is
deterministic arithmetic, not the race, so serialisation cannot hide a violation — unlike every
earlier probe, where it destroyed the signal.

```
[kvinv-dims] CTA_KV=128 NUM_KV_PER_ITER=16 KV_STRIDE=8 NUM_GROUPS=8
             THREADS_PER_GROUP=16 NUM_ITERS_PER_GROUP=16 size(dst)=16
[kvinv-ok] lines: 3528      <- probe RAN;  [kvinv-bad] violations: 0
```

1. **Lane mapping correct** — `src_thread` fetches the intended lane for every checked pair.
2. **"Silent page 0" path dead**, proven by the dims: `NUM_ITERS_PER_GROUP == THREADS_PER_GROUP == 16`
   so `thread_in_group < 16` always holds and `k < 16` always holds; the zeroing branch can fire only
   via the `kv_len` predicate, where the reader's `guard` matches the source lane's validity exactly.

Seventeenth elimination, both by arithmetic rather than a timing-sensitive run. The arm's 0/264
verdict reflects `FA3_PROBE=1` serialising away the race, which is irrelevant to a static invariant.

Remaining: the `parity` double-buffer — case 1 (timing-dependent divergence) or case 3 (parity fine,
wrong tile arrives otherwise) from the pre-registered list, separated by per-thread
`(kv_tile_idx, parity)` signatures in registers, printed after the loop.

## v0.82.0 — positively localised: the wrong KV *tile* is loaded; row addressing exonerated (2026-10-10)

`KVPERTILE=1` (pages identical within each `CTA_KV` tile, distinct across; boundaries aligned to the
kernel's own `FA3_CTA_KV`), dec=128, `STAGES=2`, 6 seeds:

| arm | fails | bad rows of 384 |
|---|---|---|
| control, all distinct | 6/6 | 100, 104, 116, 112, 96, 124 — mean ~109 |
| **identical within tile** | **6/6** | 120, 112, 104, 104, 120, 112 — mean ~112 |

Unchanged — identical rows inside a tile would have *hidden* a wrong-row error and hid nothing. So
**row/lane addressing is exonerated** (agreeing with the hand-check that `src_thread` always lands in
the caller's own warp) and **the tile index is what is wrong**: the kernel works on a different kv
tile of its own request.

Surviving suspect: **`my_kv_offset[parity]`**, the 2-entry rolling buffer (`sparse_mainloop.cuh:339`,
`parity ^= 1`). A wrong parity loads a different tile of the same request — the measured signature —
it alone explains the 4× depth scaling, and it is a race by construction since the parity toggles
across the prefetch/consume handoff.

All five measured constraints are satisfied by it: a race; wrong kv tile; same request; >1 item/CTA;
~4× depth scaling. Sixteenth elimination, and the best-aimed state the investigation has reached.

## v0.81.0 — RETRACTION: the wrong rows are WITHIN the request, not from another one (2026-10-10)

I read the `KVPERREQ` discriminator backwards and repeated the error across five commits
(`683dc9141`, `09cc03410`, `776c924c0`, `c0462732a`, `664d758b2`).

`KVPERREQ=1` = pages identical **within** a request, distinct **across**. So a wrong-row read inside
the same request returns the **right** value (invisible), while a read from another request returns a
wrong value (visible). **That arm passed 0/8** — a cross-request read would have failed it.

| arm | result | proves |
|---|---|---|
| all identical (`KVCONST`) | PASS | any wrong-row read invisible |
| identical within request (`KVPERREQ`) | **PASS** | **no cross-request reads** |
| all distinct (control) | FAIL 8/8 | wrong rows **within** the request |

So every "CTAs read another request's K/V" statement is **retracted**. The supported claim: the
consumer reduces the **wrong kv rows of its own request**.

Unaffected: it is a race; needs >1 item/CTA; scales 4× with depth; invisible to serialising
instruments but not register counting; all fifteen structural eliminations; both contributors and
their depth-scoping.

**Redirected:** a wrong row within the request means a wrong *index*, not stale foreign data — so the
gather's addressing arithmetic is now the suspect list, which I had set aside believing the data came
from elsewhere: `__shfl_sync(FULL_MASK, my_kv_offset[parity], src_thread)` (takes **another lane's**
page offset = another row of the same request; undefined under lane divergence), the 2-entry
`my_kv_offset[parity]` rolling buffer (wrong parity ⇒ different tile, same request), and
`kv_base_idx = kv_tile_idx * CTA_KV` with the coordinate tensor's `kv_offset`. All three give
wrong-row-same-request, are data-dependent, and are depth-dependent via the parity buffer.

## v0.80.0 — the sO/smem_v union is depth-dependent, not the route; candidates exhausted (2026-10-10)

dec=128, `STAGES=1`, 6 seeds, same build set:

| arm | fails | bad rows of 384 |
|---|---|---|
| `SPLIT_SMEM_O=0` (control) | 6/6 | 16, 28, 20, 20, 12, 15 — mean ~19 |
| `SPLIT_SMEM_O=1` (un-aliased) | **6/6** | 23, 16, 16, 20, 6, 16 — mean ~16 |

Cross-referenced with the 2-stage result this is a finding, not just a negative: un-aliasing `sO`
cuts ~29% at **two** stages (102→73) and **nothing** at one (19→16). The union only matters when the
pipeline has depth — with two stages the producer runs ahead and overwrites the `sO` the epilogue is
still reading; at one stage it cannot. So it is a real contributor *at depth* and **not the route**:
with both known factors removed, ~16 bad rows of 384 remain.

**Structural candidates are exhausted.** Fifteen eliminated, every synchronisation mechanism verified
correct at per-thread granularity, both confirmed contributors shown partial and depth-scoped. Any
remaining explanation must satisfy: it is a race; it reads another request's K/V; it needs >1 item
per CTA; it survives at one stage with no `sO` aliasing.

**Next step (pre-committed before this arm ran):** dump *which* kv rows a corrupted output row
actually consumed, built on the register-counting pattern that does not mask the defect — per item,
record a cheap signature of the reduced `kv_idx` values in registers, print per block after the loop,
compare against what the plan says that item should have consumed. `KVPERREQ` already proves the rows
belong to a different request, so the dump need only identify which one.

## v0.79.0 — FA2/FA3 region aliasing exonerated (14th eliminated) (2026-10-10)

dec=128, `STAGES=1` (the only depth where disjoint fits — 234544 B vs a 232448 max at two stages),
6 seeds each, both arms producing real verdicts rather than OOM:

| arm | fails | bad rows of 384 |
|---|---|---|
| `ALIAS_SMEM=1` (default) | 6/6 | 16, 12, 8, 12, 8, 48 — mean ~17 |
| `ALIAS_SMEM=0` (disjoint) | **6/6** | 28, 32, 20, 28, 32, 24 — mean ~27 |

Disjoint is **not** clean and is slightly worse, so the layout suspicion recorded in v0.78.0 is
wrong and nothing writing FA2's region is the route.

**Two distinct things in this tree are called "aliasing"** and must not be conflated:
`FUSED_SPLIT_SMEM_O` aliases `sO` with `smem_v` *inside FA3's own* `SharedStorage` — a **confirmed
contributor, ~⅓** — while `FLASHINFER_FUSED_ALIAS_SMEM` overlays *FA2's* `SharedStorage` on FA3's
region, and that one is now **exonerated**.

## v0.78.0 — per-thread barrier arrivals are symmetric; suspicion moves to the smem layout (2026-10-10)

Each of the 128 producer threads compared its own `producer_commit` count to its own intended loads,
printing only on disagreement (silence = pass). All 7 commit sites, on the branch that compiles:

```
verdict: bad rows 124/384      <- instrument again PRESERVES the defect
ASYMMETRIC THREADS: 0
loaded == commit_k == commit_v on every block (108, 109, 110, 112, 124, 128)
```

Thirteenth hypothesis eliminated. **Synchronisation is now verified correct at every reachable
level**: item tile counts, per-thread arrivals, stage indexing (`STAGES=1` can't shift and still
fails), cp.async coverage, scheduler determinism, acquire/release pairing — and a race still reads
another request's K/V.

Suspicion therefore moves from the protocol to the **shared-memory layout**: `ALIAS_SMEM=1` (default)
overlays FA2's `SharedStorage` on FA3's region, which is the only reason the 2-stage co-resident
config fits (163920 vs 234544 against a 232448 max). Every block here takes the FA3 role so FA2
should never write there — but anything that does corrupts K/V stages while leaving all counts
consistent. Testable as one bit at `STAGES=1` only, since disjoint doesn't fit at two stages.

Also fixed two bugs in the instrument itself before trusting it: the counters had gone to the
never-instantiated `MULTIITEMSCORING` branch (all threads would have reported false asymmetry), and a
near-committed hypothesis about the `else` branch's shorter arg list was checked and dropped —
`load()` reads `work_tile_info` and calls `fused_causal_kv_len` itself.

## v0.77.0 — it IS a race (first clean within-build test); accounting limitation recorded (2026-10-10)

Same build dir, same seed, 6 repeats at dec=128 baseline: **96 100 88 88 92 84**. Varying ⇒ **a
race**, not an addressing bug. Every earlier comparison crossed builds (seed 1 → 100 in one dir, 116
in another), which proves nothing since register allocation differs; this is the first within-build
repeat. Independently, `my_kv_offset[2]` is declared *inside* `load()` (`sparse_mainloop.cuh:339`),
so it is fresh per item and cannot carry a stale page across an item boundary.

**Limitation of my own accounting, so "0 mismatches" is not over-read:** the instrument counted the
producer's *intended* loads (`num_kv_tiles`, once per item) against consumer releases *from one
thread* — not actual per-thread `producer_commit` arrivals. A thread diverging through
`use_predicate` or the `kv_tile_idx == swa_begin` special case would arrive a different number of
times and the accounting would still read 0 mismatches. So item-level bookkeeping is bounded;
**per-thread arrival symmetry on the KV barriers is unmeasured** and is the next target — an
over-arrival releases the consumer early (race, no deadlock), exactly the observed signature.

Standing state: a race, in the K/V path, reading another request's data, requiring >1 item/CTA,
scaling 4× with pipeline depth, invisible to every serialising instrument but not to register
counting. Eleven hypotheses eliminated; two contributors confirmed (`sO`/`smem_v` ~⅓, unfenced
`smem_q` ~⅓, the latter a regression that must not ship).

## v0.76.0 — A2 CLOSED: the fine-grained tile dispatcher's premise is false (2026-10-10)

Narrow FA2 classes `{48,32,16}` vs default `{128,64,16}`, single-request arms, all **PASSING** the
causal fp32 reference:

| packed_qo | FA2 `{128,64,16}` | FA2 `{48,32,16}` | effect | FA3 | FA3 vs best FA2 |
|---|---|---|---|---|---|
| 16 | 0.1409 | **0.1330** | 1.06× faster | 0.0268 | **5.0×** |
| 32 | 0.3978 | **0.2219** | **1.79× faster** | 0.0329 | **6.7×** |
| 48 | 0.3952 | 0.4900 | 1.24× slower | 0.0389 | **10.2×** |
| 64 | 0.3948 | 0.4964 | 1.26× slower | 0.0457 | **8.6×** |

The narrow classes **work** — tile 32 is a real 1.79× win, the first traction the dispatcher idea has
shown, unlocked by correcting the tree's wrong "tile 48 needs 12 warps" claim. Tiles 48/64 regress
structurally: 48 resolves to 1q×8kv (`num_mma_q=3`) which loses to 64's 4q×2kv, and `packed_qo=64`
then needs two 48-wide tiles.

**But item A's premise fails.** The hypothesis was FA2 owning {16,32,48} with FA3 taking ≥64; FA3 is
**5.0–10.2× faster than the best FA2 config at every one of those sizes**. No crossover exists, so no
threshold (A3) or dynamic WGMMA tile (A1) can make the dispatcher pay. The honest form of item A is
"route everything to FA3", which is not a dispatcher. Third independent measurement today pointing
the same way, after B1e and the A3 four-axis sweep — and measured where the kernel is provably
correct, so K1's open defect doesn't bear on it.

Also lands the per-item pipeline accounting instrument (`PIPE_COUNT`): producer's intended loads vs
consumer's actual consumes, counted in registers during the race and printed once per block after
the loop, since every in-loop instrument masked the defect.

## v0.75.0 — A2/A3 on correct arms: FA3 wins 5–12× at small tiles; four more mechanisms ruled out (2026-10-10)

**A2/A3, every arm PASSING the fp32 reference.** The defect needs >1 item/CTA and `band1` is a
single request, so these arms are valid while K1 is open (unlike the pre-2026-10-10 A3 arms, whose
numerics failed). FA2 `{128,64,16}`, kv=2048:

| packed_qo | FA2 | FA3 | FA3 advantage | `BatchAttention` |
|---|---|---|---|---|
| 16 | 0.1409 | 0.0271 | 5.2× | 0.0202 |
| 32 | 0.3978 | 0.0326 | 12.2× | 0.0203 |
| 48 | 0.3952 | 0.0393 | 10.1× | 0.0205 |
| 64 | 0.3948 | 0.0456 | 8.7× | 0.0193 |

The user's `{16,32,48}` hypothesis is **refuted again on correctness-gated arms**: FA3 owns the
narrow tiles by 5–12× where the tile-occupancy argument predicts FA2 should. And `BatchAttention`
beats our FA3 side **1.3–2.4× at every size** on shapes where we are provably correct — the fusion's
problem restated without reference to the race.

**Four mechanisms ruled out by reading, no GPU time:** producer/consumer tile-count disagreement
(same call, same args, `:247`/`:325`); scheduler desync (deterministic per block, `is_producer`
ignored); acquire/release imbalance (balanced per item — `count` K and `count` V both sides, every
wait matched); cp.async coverage on the KV commit (`producer_arv_count = NUM_COPY_THREADS` = 128 and
all 128 issuing threads commit, so completion implies all copies landed — the strongest candidate,
and still correct as written).

Next must be the empirical instrument: per-item tiles **loaded vs consumed**, accumulated in
registers and printed at block exit only, so it cannot serialise away the signal the way the printf
probe, racecheck and memcheck each did.

## v0.74.0 — LOCALISED: CTAs read another request's K/V via the cross-item pipeline (2026-10-10)

First mechanism with **positive** evidence instead of elimination. `KVPERREQ=1` (pages identical
within a request, distinct across) at dec=128, baseline, warm dir:

| arm | fails | bad rows of 384 |
|---|---|---|
| distinct pages everywhere | 8/8 | 72–116 |
| **identical within request** | **0/8** | **all zero** |
| all pages identical | 0/8 | all zero |

**The stale rows come from another request** — cross-item leakage in the K/V path. Not Q (fenced,
didn't fix), not the output path, not the partial slab (refuted), not addressing within a request.

**Mechanism from source:** `load_tail` runs **once after the whole persistent loop**
(`prefill_sm90.cuh:285`) while `smem_pipe_write_k/v` and `smem_pipe_read_k/v` live *outside* it. One
item consuming fewer stages than were loaded shifts the pipeline **permanently**, so every later
consumer reads a stage filled for a different request. Stock is immune — one item per CTA leaves no
successor to inherit the shift. It retro-explains the handshake (112→56, halved not fixed: a
rendezvous realigns timing, not stage indices), the items/CTA threshold, the data dependence, and
the serialisation-masking.

**Ruled out by reading, no GPU time spent:** producer/consumer tile-count disagreement (both sites
call the same `get_num_kv_tiles` with the same args, `:247` and `:325`); scheduler desync
(`UnifiedWgmmaTileScheduler` is deterministic per block, `is_producer` ignored, broadcast/prefetch
are no-ops).

Next: **count** tiles loaded vs consumed per item on both sides before attempting a fix — resetting
`PipelineState` without its mbarrier phases is the bookkeeping that deadlocked the `barrier_O`
attempt.

## v0.73.0 — the baseline defect is KV ADDRESSING; the fence is a regression (2026-10-10)

| arm, dec=128, `KVCONST=1` (identical kv page **values**) | fails | bad rows of 384 |
|---|---|---|
| `FENCE=0 SPLIT=0` baseline | **0/8** | all zero — **PASS** |
| `FENCE=1 SPLIT=1` | 4/8 + 4 cold-build | 4, 12, 12, 12 (12, 16, 12 on re-run) |

**Premise correction:** I built this to test K/V pipeline parity, assuming `KVCONST=1` equalised kv
*lengths*. It does not — it sets `k_cache[:] = k_cache[0:1]`, identical page *values*, lengths
untouched. **Pipeline parity remains untested**; the "eleventh refutation" is withdrawn.

Read correctly it is sharper. (1) **The baseline defect is a KV-addressing error**: clean 8/8 with
identical pages, 10/10 failures with distinct ones ⇒ CTAs read the *wrong kv rows*, the stale-K-tile
signature — the first positive identification of the defect's class rather than another elimination.
(2) **`FENCE=1 SPLIT=1` adds a different error**: it fails where addressing errors are invisible by
construction and where stock is clean, so it is a **regression**, not merely an incomplete fix. "A
third real contributor" holds only for the dec=128 bad-row count and must carry this caveat; it must
not ship.

**NOVERDICT explained** (two wrong guesses, one wrongly dismissed): warm runs take 9–10 s and
previously-NOVERDICT seeds return verdicts 3/3 once built; NOVERDICTs sit on the leading runs of
*fresh* dirs (warm 1/10, fresh 3/10, fresh 4/8 at seeds 1–4). They are **cold-build timeouts** — the
full TU build exceeds the 600 s cap and ninja progresses incrementally across kills. I had dismissed
this using 122 s read from a single ninja *edge* rather than the whole build. So cold-dir counts are
"of the runs that completed", and per-arm NOVERDICT rates say nothing about the kernel.

## v0.72.0 — split-KV slab refuted as the residual; pipeline parity is next (2026-10-10)

The 48 remaining bad rows at dec=128 are **not** the split-KV partial slab. Both fixes so far are
shared-memory reuse, so the slab — where each split item writes
`(indptr[packed_qo_idx] + off) * num_kv_heads + kv_head_idx`, and whose pressure scales with the
same axis the threshold sweep varied — was the natural next candidate, and the only one that could
also explain the bimodal knee. `FUSED_NOSPLIT=1` compiles it out and changed nothing: 8/8 fails at
36–82 bad rows (mean ~53) against 9/10 at 32–60 (mean ~48) with split on. **Tenth hypothesis
eliminated.**

Next and last enumerable piece of cross-item state: **K/V pipeline parity**.
`smem_pipe_write_k/v` and `smem_pipe_read_k/v` are passed by reference across work items, and stock
runs one item per CTA. An item ending on an odd stage leaves the next item's producer and consumer
disagreeing on stage parity, so the consumer reads a stale K tile. Tested with `KVCONST=1` (equal kv
lengths ⇒ equal tile counts ⇒ identical parity after every item), plus a null case on the baseline
since constant kv also changes the workload.

## v0.71.0 — B1e: achieved occupancy measured, and it undercuts the fusion motivation (2026-10-10)

ncu with kernels attributed by name (`mix_decode`, `MIX_ALL_FA3=1`, `SM_SPLIT=all`, dec 8 and 128):

| kernel | achieved occupancy | blocks/SM (reg / smem) | waves/SM |
|---|---|---|---|
| `FusedFA3PersistentKernel` (ours) | **18.75%** | **1 / 1** | 1.00 |
| `FusedStateReductionKernel` | 41.64% | 8 / 13 | 1.75 |
| `PersistentKernelTemplate<BlockBatchPagedAttentionPersistent…>` (`BatchAttention`) | **12.49%** | 2 / 2 | 1.00 |

Our kernel is pinned to **one block per SM by registers and shared memory simultaneously** — 18.75%
of peak warps is exactly a 384-thread block (12 of 64 warps), identical at dec=128. So nothing can
be co-scheduled *within* an SM; the role split is purely spatial across SMs. Structural, not tuning.

**The reference runs at lower occupancy (12.49%) and is 3.14× faster.** Achieved occupancy therefore
does not predict performance on this workload, and "the GPU is underfilled, so fuse the small
kernels" fails on its own measurement. B1b had already cut that claim from 89.3% to 8.1% of GPU
time; B1e removes the warp-fill version too.

**Method note:** the first attempt profiled every kernel in the process and returned
`softmax_warp_forward` (9183 invocations), `arange`, `CatArrayBatchedCopy` — the harness's fp32
*reference* path, not the kernel — after 11 minutes at 1% GPU. Those numbers are not kernel
occupancy. Fixed with `-k regex:(Fused|Persistent|persistent) --launch-count 24`.

## v0.70.0 — Q fence is a third real contributor, not a fix, and 3.14× slower (2026-10-10)

Dedicated `barrier_Qdone`, 10 seeds/arm at `DEC_ROWS=128`, macros verified in `build.ninja`:

| arm | numeric fails | bad rows of 384 | no verdict |
|---|---|---|---|
| `FENCE=0 SPLIT=0` | 10/10 | 64–116, mean ~102 | 0/10 |
| `FENCE=0 SPLIT=1` | 7 | 60–86, mean ~73 | 3/10 |
| `FENCE=1 SPLIT=1` | 9 | 32–60, mean ~48 | 1/10 |

Counting a missing verdict as a failure, **every arm fails 10/10 at dec=128**. Bad rows fall ~⅓ per
step (102→73→48), so both mechanisms are real contributors and the source reading was directionally
right — but neither nor both is sufficient; something beyond `smem_q`/`sO` reuse still corrupts at
high items/CTA.

At `DEC_ROWS=32` the combined arm is the **first correct configuration** in this project on a mixed
4-mode shape (PASS 0/288 where baseline FAILs 2/288) — and it is **3.14× slower** than
`BatchAttention` (0.3703 ms vs 0.1181 ms), the predicted cost of serialising producer and consumer
across items, and the same reason every serialising instrument masked the defect.

**Unattributed:** 4 of 30 runs printed no verdict. Re-running those seeds gave ordinary finite
failures (82/71/45 bad rows) — no hang, no non-finite, no reproduction. My two guesses ("hang", then
"non-finite") are both unsupported and recorded as unattributed.

**Methodological retraction:** `SEED` sets input data, not the race — the same seed yields different
outcomes across runs. A per-arm NOVERDICT *rate* therefore reflects 10 random draws, not the arm, so
the 3/10-vs-1/10 gap carries no weight, and this session's earlier claim that "the fence introduces
an intermittent hang" was wrong on that basis and is retracted.

## v0.69.0 — barrier_O cannot carry the Q invariant; dedicated barrier_Qdone (2026-10-10)

Reusing `barrier_O` for the cross-item Q fence **deadlocked**: GPU at 100% for 17 minutes on a
microsecond kernel. Diagnosing it settled what that barrier means. The consumer runs
`mma(0), epi(0), mma(1), epi(1), …`, so an arrival at the *start* of `mma(N)` means "epi(N−1) is
done" — which is what licenses the producer to refill `smem_v`, the buffer `sO` aliases. Therefore
`barrier_O` is an **`sO`/`smem_v` handshake**, never a Q handshake (the Q load sits above it); its
arrival/wait are paired with a one-item offset and the arrival is guarded by `work_idx != 0`, so
relocating it left a wait with no matching arrival. Moving it would also have *weakened* the output
side, placing the arrival before `epi(N−1)`. One barrier cannot carry two invariants — which is also
why un-aliasing `sO` measured as a separate 45% contributor rather than a duplicate.

Reimplemented with a dedicated `barrier_Qdone` (one arrival per item, unconditional, after the
consumer's last `sQ` read; producer waits `(work_idx-1)%2` before refilling `smem_q`, guarded at
item 0). `barrier_O` is left byte-identical to stock, verified by diff. Also added `FUSED_FA2_W*`
single-sourcing in the loader so the compile-time runner widths cannot drift from the planner's
`FUSED_FA2_TILES` — a silent mis-route otherwise — with host-side validation mirroring the device
warp-layout constraint.

## v0.68.0 — fence control arm in; fence arm retracted as a build artifact; A2 unblocked (2026-10-10)

**Control (`FENCE=0`, fresh build dir, macro verified at nvcc):** dec=8 fails 3/10; dec=128 fails
**10/10** at 64–116 bad rows of 384 — reproducing the historical figure in a clean tree, so the A/B
is fence-vs-no-fence rather than cached-vs-fresh.

**RETRACTED: the first `FENCE=1` "10/10 fails" was an artifact.** The hoisted `barrier_O.wait` was
inserted into `get_num_kv_tiles`, where `shared_storage` and `work_idx` are not in scope, so the arm
never compiled; and the harness scored a *missing* verdict as a failure (`${t#bad rows }` is empty,
and `[ "$v" != "0" ]` is true for empty). Same family as the NaN gate recorded earlier: a comparison
that cannot tell "measured nothing" from "measured zero". Re-patched into `load()` above every
`smem_q` write; the script now checks the build first and reports `NOVERDICT` separately. The fence
hypothesis is **untested**.

**A2 unblocked — a documented constraint was wrong.** The tree recorded "tile 48 needs 3 q-warps and
8/3 is not integral — tile 48 requires a 6- or 12-warp layout". That assumed
`warps_q == CTA_TILE_Q/16`; nothing requires it, because `num_mma_q` absorbs the leftover factor.
Choosing the largest *divisor* of `kFa2Warps` whose q-width divides the tile makes all of
{16,32,48,64,96,128} expressible at the existing 8 warps — 48 as 1q×8kv with `num_mma_q=3`, 96 as
2q×4kv with `num_mma_q=3`. The old helper failed only because `kFa2Warps/3` truncated to 2, giving a
silent 6-warp layout against the required 8. So the user's {16,32,48} hypothesis needs **no** change
to `FusedPlanInfo<4>`: three widths, three FA2 task slots. Runner widths are now
`FUSED_FA2_W0/W1/W2` with `static_assert`s on both layout invariants; defaults {128,64,16} unchanged.

## v0.67.0 — items-per-CTA threshold measured; the cross-item Q hazard named (2026-10-10)

**Threshold sweep (5 seeds/point, all-FA3, `SM_SPLIT=all`):** clean 15/15 below 1 work item per CTA
(DEC_ROWS 1/2/4), first failures exactly where the ratio crosses 1 (dec=8), then a ~100× explosion
at dec=128 (112/112/84/92/96 bad rows of 384, 5/5). So >1 item/CTA is **necessary**, but severity is
**bimodal** rather than a smooth function of the ratio — 7.9 items/CTA at dec=64 yields one bad row.
That residual is logged as open, not explained away.

**Mechanism, read off the source (two verified premises):** the producer refills `smem_q` *before*
`barrier_O.wait` (`sparse_mainloop.cuh`), and the consumer arrives on `barrier_O` after only the
*first* QK gemm is issued (`mainloop_mma.cuh`, `wg_wait=-1`) — while `TiledMmaQK` is
`GMMA::ss_op_selector` (`kernel_traits.cuh:97`), so Q stays in shared memory and is re-read by every
later KV tile. The sole cross-item fence is thus placed too late on the producer side and raised too
early on the consumer side. This is **stock 0.7 code**: correct for one item per CTA, a hazard for
many. It independently predicts the stock-arm failures (5.6 vs 128), the instrument-masking, and
puts the defect in the same class as the confirmed `sO`/`smem_v` contributor (45%) — the same bug on
the output buffer rather than the input.

New knob `FUSED_Q_REUSE_FENCE` / `Q_REUSE_FENCE` hoists the producer wait above the Q load and moves
the consumer arrival after its last `sQ` read. Default off; it serializes producer and consumer
across items, so it is expected to cost time. A/B in flight.

## v0.66.0 — memcheck clean; instruments exhausted; recommendation to park the fusion (2026-10-10)

**`memcheck`: 0 errors** on the arm that fails 10/10 — and the run *passed* under the sanitizer
(0/384), the slowdown masking it as `FA3_PROBE`'s printf did. So: **no out-of-bounds access**, and
timing-dependence reconfirmed.

**Instrument status is complete and unhelpful:** `racecheck` can't discriminate (flags stock's own
correct kernel), `memcheck` is clean, and anything slow enough to observe the kernel hides the bug.

**Tally on this defect: nine hypotheses eliminated, one contributor confirmed** (`sO`/`smem_v`
aliasing, ~45%, `FUSED_SPLIT_SMEM_O`). Eliminated: `kQueryEmpty` symmetry, `work_idx` symmetry,
Q-completion mechanism, scheduler advance logic, the store-completion handshake (it made a clean arm
dirty), same-output-row collisions, per-request plan arrays, group crossings, out-of-bounds access.

**Recommendation (unchanged by this result): park the two-role fusion.** Three independent blockers:
the fusion is net-negative on realistic 4-mode batches (1.46–2.01× slower than all-FA3); B1's
measured ceiling is 29%, below the ≥30% bar; and `BatchAttention` already does the whole batch in
one validated-correct persistent kernel, 1.11–4.9× faster than ours everywhere. Restate the project
on **FA3-only colocation** — one kernel, linear in KV, no `sm_split` parameter, no cross-role reuse.
PD-colocation points 1/2/4 don't depend on the fusion and point 3 is stronger without it.

## v0.65.0 — separation done: the race tracks WORK-ITEM count, not request count (2026-10-10)

Two harness fixes were prerequisites:
- **`ORACLE=ba`** — use the validated `BatchAttention` where the fp32 reference is unaffordable.
  Cross-validated on a fixture both can run: fp32 0.003303 vs BA 0.004407, **both 0/1024**.
- **`FWS` 1 GB → 8 GB.** Everything past qo≈2048 had been failing with `FusedFourTaskPlan failed:
  invalid argument` — the **plan builder**, not the kernel. At qo=2048/kv=8192 the planner makes 64
  chunks × 64 q tiles, so `partial_o` alone needs **~939 MB**. A harness limit that had masqueraded
  as a kernel ceiling across several sweeps.

**Result — one request, rising scale:**

| fixture | merge rows | items/CTA | result |
|---|---|---|---|
| qo=2048, kv=8192, 1 request | 8192 | ~217 | **PASS 0/2048** |
| qo=4096, kv=8192, 1 request | 16384 | ~434 | **1 bad row/4096**, rel 0.2989 |

**A single request corrupts at sufficient scale**, so request count is not necessary. The defect
scales with work-item count — i.e. with how often a CTA reuses `sQ`/`sO` — which **rules out the
per-request plan arrays** and leaves cross-item reuse as the mechanism, agreeing with the
four-corner bisect.

**Separate finding:** the partial slab grows as O(`packed_qo` × `num_kv_chunks`) — ~1 GB of partials
for one 2048-token chunk at 8K context. A real serving concern independent of correctness, and it
explains why the chunk-size sweep gave 4.5×.

## v0.64.0 — racecheck ran: 17 hazards, but it also flags stock's correct kernel (2026-10-10)

`compute-sanitizer --tool racecheck` on the simplest failing arm (stock gather + stock store,
`DEC_ROWS=32`, 30/288 bad under the sanitizer): **17 hazards, 17 errors.**

| kernel | mentions in hazard blocks |
|---|---|
| `FusedFA3PersistentKernel` (ours) | 136 |
| `BlockBatchPagedAttentionPersistent` (FA2 runners) | 408 |
| **stock `BatchPrefillPersistentTileScheduler`** (`CTA_KV=96`) | **35** |

The third row is decisive against the tool: that is flashinfer's own FA3 kernel, which this harness
runs as the validated null case and which **passes** at 0 bad rows. **racecheck does not model
`mbarrier`/TMA ordering**, so on CUTLASS Hopper warp-specialised kernels it reports the async
pipeline's traffic as unsynchronised — false positives by construction.

**So the 17 hazards must not be chased as 17 bugs.** Recorded so the next attempt doesn't spend a
turn on them.

**Better next experiment — bisect by disabling reuse, not by hypothesis:** partition `work_indptr`
one-item-per-CTA at the same batch shape (~3-line planner change). Corruption vanishing confirms
cross-item reuse as the mechanism without needing to identify the buffer; persisting exonerates
reuse entirely. That is what the last six hypotheses were each trying to approximate indirectly.

## v0.63.0 — the multi-item-per-CTA path is racy in ALL corners, baseline included (2026-10-10)

Four corners × 10 runs, G=1, 129 requests, ~6.9 work items per CTA, handshake OFF:

| gather | store | runs with bad rows | mean bad/run | worst |
|---|---|---|---|---|
| packed | packed (**default**) | **10/10** | **128.0** | 144 |
| packed | stock | 10/10 | 19.4 | 36 |
| stock | packed | 6/10 | 20.7 | 48 |
| **stock** | **stock** | **7/10** | **5.6** | 16 |

**Supersedes two earlier single-run claims** — "stock is CLEAN 0/384" and "the defect is OURS".
Both were n=1 on a configuration whose failure rate is a few percent.

- The path is racy in **all four corners**, so reverting the packed additions would not give a
  correct kernel.
- Each packed path amplifies the baseline ~3.5×; **together 23×**, strongly superlinear — they
  interact, consistent with `sQ` and `sO` sharing the aliased smem region and both being reused
  per work item.

**Consequence:** a real 4-mode batch (chunk + tens to hundreds of decode rows) *is* the multi-item
configuration, and the port is incorrect there in every configuration tested. No serving arm and no
realistic-shape timing is meaningful until the baseline race is root-caused.

**Next:** root-cause the *baseline* race first (simplest reproducer), and start by establishing
whether any 0.7 configuration actually runs multiple work items through one CTA — our persistent
multi-item loop may be a configuration its epilogue and `barrier_Q` phase handling have never seen.

## v0.62.0 — decode-heavy arm: the fusion is NET NEGATIVE on realistic shapes (2026-10-10)

The +5.4% in v0.61.0 was measured with **2 decode tokens**. I predicted the margin would grow with
more co-resident FA2 work. **It inverts.** One prefill chunk (256×2048) + N decode rows (1×2048):

| decode rows | co-resident (best split) | all-FA3 control | fusion |
|---|---|---|---|
| 2 | 0.2426 @128 | 0.2557 | +5.4% |
| **8** | 0.3665 @96 | **0.2512** | **1.46× SLOWER** |
| **32** | 0.7371 @64 | **0.3670** | **2.01× SLOWER** |

Best `sm_split` moves down as decode grows (128→96→64) and the mis-set penalty explodes (8.68 ms at
dec=32/split=128, 23× that shape's optimum).

**Why:** decode routed to FA2 lands on an executor 3.7–5× slower per row, so FA2 becomes the
critical path while FA3 finishes its chunk and waits. More decode ⇒ monotonically worse.

**The +5.4% is retracted as representative** — it is the best case (almost no FA2 work to be slow),
not the typical one.

**Project-level conclusion: the two-role FA3+FA2 fusion as implemented is net-negative on realistic
4-mode batches.** Not a tuning problem — at dec=32 *no* `sm_split` reaches the control. The fusion
can only pay if the FA2 role nears FA3 parity, and it is currently 1.1×–22× off `BatchAttention`
(its own source family). **Fixing the FA2 role is a precondition for the fusion mattering at all.**

**New bug:** all-FA3 with 129 requests → 92/384 bad rows (every other arm 0 bad). The FA3 path
breaks at high request count; no prior fixture had more than 3 requests. Invalidates the dec=128
comparison; suspect per-request plan arrays or the merge layout at scale.

## v0.61.0 — THE FUSION QUESTION: co-residency is worth +5.4% (2026-10-10)

Never measured directly before. Same rows (`qo=[256,1,1]`, `kv=[2048,300,512]`), both arms 0 bad
rows, 40 iters:

| config | ms |
|---|---|
| co-resident (`mixed`) @ `SM_SPLIT=96` | 0.3228 |
| @ 112 / 120 | 0.2604 / 0.2579 |
| **@ 128 (best)** | **0.2426** |
| **FA3-only, all SMs** (same rows) | **0.2557** |

**Fusion beats routing-everything-to-FA3 by 1.054× (+5.4%).** Co-residency pays — answering the
question A3's sweep raised — but by 5.4%, not 30%.

**Consequence for the bar:** attention is 20.5% of step time (B1b), so +5.4% on attention is
**~1.1% end-to-end** — two orders below the ≥30% target and inside the ±1.5–3% serving noise floor,
i.e. **not measurable in a serving A/B**.

**Caveat / next arm:** this batch has only 2 decode tokens against a 256-token prefill chunk, so the
FA2 role has almost nothing to overlap. Fusion's value should grow with co-resident FA2 work. **A
decode-heavy mix (one chunk against 32–128 decode rows) is the arm that could change the answer**
and is now the highest-value measurement left.

Also: co-resident beats FA3-only *only* at `sm_split=128`; at 96 it is worse (0.3228). The fusion's
win is entirely contingent on SM pre-assignment — which is the mechanism, not a refinement.

## v0.60.0 — A3 measured on all three axes: route to FA3 everywhere (2026-10-10)

Pod `f929493da6e45f10` expired mid-session; re-forked as **`62d25ecfb033ad4b`** and restored
(`drivers/restore.sh`, now committed — it had only ever existed on the pod's tmpfs).

**Third axis (batched decode rows), qo=1, kv=2048, all arms 0 bad rows:**

| rows | FA2 ms | FA3 ms | FA3 wins by | `BatchAttention` | FA3/BA |
|---|---|---|---|---|---|
| 1 | 0.1431 | **0.0267** | 5.36× | 0.0207 | **1.29×** |
| 8 | 0.1978 | **0.0709** | 2.79× | 0.0296 | 2.40× |
| 32 | 0.7200 | **0.1761** | 4.09× | 0.0786 | 2.24× |
| 128 | 2.6674 | **0.6311** | 4.23× | 0.2456 | 2.57× |

FA2 never overtakes FA3. My row-packing explanation for the doc's "decode-only: FA2 wins by 1.30×"
is **refuted**.

**A3 now spans `packed_qo` 4–128, kv 128–8192, rows 1–128 — and the answer is "route to FA3"
everywhere.** `MIN_PACKED_QO = 48` has no measured support anywhere in that space and costs
1.2×–12× below it.

**Open and not dropped:** the earlier "decode-only: FA2 beats FA3 by 1.30×" is now contradicted by
three axes. One of the two results is wrong and the dispatch rule depends on which, so it must be
re-run before A3 is wired into `split_classes`.

On these shapes FA3 is **1.29×–2.57×** off `BatchAttention` vs 4.9× on `mixed` — the verify/decode
band is the closest ground, and split-KV (78% of kernel time) is what stands between it and parity.

## v0.59.0 — baseline validated; we have NO win on any fixture (2026-10-10)

`BatchAttention` gated against the same fp32 causal reference as the fused arm, every fixture:
**rel 0.0018–0.0031, 0 bad rows — VALID BASELINE.** It is not computing something cheaper.

| fixture | fused ms | `BatchAttention` ms | fused/BA |
|---|---|---|---|
| `fa3_1chunk` | 0.0213 | 0.0192 | **1.11× slower** |
| `fa3_multitile` | 0.0428 | 0.0195 | 2.19× |
| `mixed` | 0.2433 | 0.0494 | **4.92×** |
| `fa2` | 1.1036 | 0.0494 | **22.36×** |

**Bottom line: the fused kernel is slower than flashinfer's own single persistent kernel on every
fixture, by 1.11× to 22×. There is no win anywhere.** The `fa3_1chunk` "1.51× faster" reported
earlier was an artifact of the weaker two-launch baseline (which runs at 0.59–0.90× of
`BatchAttention`).

**Implied priority:** fix the FA2 path (22×, and decode-only steps are FA2-only — the likely
explanation of the unexplained in-server −5% to −8.4%), then the split-KV path on `mixed` (4.92×),
before any of A/B/S/M. Tile-dispatch tuning while the FA2 role is 22× off would be measuring noise
on top of a much larger defect.

## v0.58.0 — the baseline was wrong: `BatchAttention` is the real comparator (2026-10-10)

Every speedup in this project compared against "unfused" = **two `BatchPrefillWithPagedKVCacheWrapper`
launches**. But our FA2 runners are ported from flashinfer's **persistent `BatchAttention`**, which
does the whole batch in **one** launch. It had never been timed.

| fixture | fused | 2-launch "unfused" | **`BatchAttention`** | fused vs BA |
|---|---|---|---|---|
| `fa2` @ 16 | 1.0961 | 0.0549 | **0.0499** | **22.0×** |
| `mixed` @ 128 | 0.2430 | 0.0730 | **0.0493** | **4.9×** |

`BatchAttention` is **0.675×** the two-launch arm on `mixed`, so the baseline I was using is 48%
slower than the real one and every recorded ratio is optimistic.

**Consequences:** (1) the gap is 4.9× on `mixed` and 22× on FA2-only, not 3.4×/20×; (2) the
`fa3_1chunk` 1.51× win must be re-measured against `BatchAttention` before being cited; (3) it
reframes the premise — the thing to beat is already a single persistent kernel, so the contribution
must be the SM pre-assignment and role specialisation *within* one kernel, measured against
`BatchAttention`. The ≥30%/≥10% bar should be restated against this comparator.

**Caveat:** `BatchAttention`'s output was not validated against the fp32 reference — it ran as a
timing arm only. It needs the same correctness gate before any published number.

## v0.57.0 — race CLOSED (alias + 2 stages, now default); FA2-only path found 20x slow (2026-10-10)

### Fixed — the intermittent race

`FLASHINFER_FUSED_ALIAS_SMEM` + `FUSED_FA3_STAGES=2`, both now loader defaults. `mixed` @
`SM_SPLIT=128` (co-resident), 35 runs per arm:

| arm | failures / 35 | fused ms |
|---|---|---|
| **alias + 2 stages (new default)** | **0 / 35** | **0.2426** |
| disjoint + 1 stage (old default) | 3 / 35 | 0.2486 |

Aliasing puts FA2's storage at offset 0 so `smem = max(FA3, FA2) = 163920` rather than
`FA3 + FA2 = 234544`, which is what lets 2 stages launch under the 232448 B limit. Stock 0.7 relies
on double buffering to make the `smem_v`/`smem_o` union safe; our 1-stage port had removed that
invariant. Settles the tree's two contradictory records in favour of "aliasing is sound".

All six fixtures pass on the new defaults. `fa3_1chunk` is **1.51× faster** than unfused
(0.0203 vs 0.0306 ms).

### New defect — the FA2-only path is ~20× slower than unfused

Never previously timed. `MODE=fa2`: **1.10 ms** at `SM_SPLIT` 1/4/16/64, **13.34 ms** at 128 (FA2
starved to 4 SMs — a 12× cliff, mirroring the earlier `sm_split=16` misallocation on `mixed`), vs
**0.055 ms** unfused. Two problems: the cliff, and a 20× floor at the best split.

**Decode-only steps are FA2-only**, so this would dominate any end-to-end serving number and is a
strong candidate for the earlier in-server −5% to −8.4%. Must be understood before any serving A/B.

## v0.56.0 — `STAGES=1` was never valid; and the obvious enabler for `STAGES=2` is wrong (2026-10-10)

**Reframing.** Stock 0.7 has the same `union { smem_v; smem_o; }` but runs **2 stages**, and
double buffering is what makes the union safe — the producer's next V load hits the *other* stage.
Our port set `STAGES=1` to fit smem and removed that invariant. So `STAGES=2` (0/35 vs 3/35) is not
a mitigation hiding a race; **`STAGES=1` was an invalid configuration.**

**Failed attempt, kept because the failure is the useful part.** The 69,632 B reservation blocking
`STAGES=2` is commented as the scheduler's "cluster slot", so I gated it off on the grounds that
cross-role backfill is unused → **`mixed` 25 failures / 25**. It is *not* a backfill queue: it is
where the **FA2 runners' SharedStorage lives**, placed past FA3's region. Reverted; verified back
to 1/10.

**Arithmetic that failure establishes:** co-residency needs FA3's layout *plus* the FA2 region past
it — `164864 + 69632 + 48 = 234544` vs a `232448` max. **`STAGES=2` and co-residency are
incompatible with the current layout**, which is why the clean STAGES=2 result only holds at
`SM_SPLIT=all`.

**The right fix, with numbers, not implemented:** FA2-role blocks should address storage from
**offset 0**. A block is one role, never both, so the layouts can alias across roles →
`smem_size = max(FA3, FA2) = 163920`, ~68 KB spare, `STAGES=2` everywhere. The past-FA3 placement
is only needed when one block does both — i.e. the cross-role drain (B2c). So this is a
prerequisite for B2c too, and the smallest change that closes the race where it matters.

## v0.55.0 — race ROOT-CAUSED: the `sO`/`smem_v` union, O side only (2026-10-10)

Slab fingerprint across runs (`MODE=fa3` @ `SM_SPLIT=all`, same seed and build):

```
lse_sum = 768993.500000  (IDENTICAL every run)
o_sum   = 1467.978 / 1468.258 / 1468.412  (DIFFERS every run)
```

`partial_lse` is deterministic, `partial_o` is not. `partial_lse` is written straight from
registers; `partial_o` is written **from `sO` in shared memory**, and `sO` is **unioned with
`smem_v`** — the buffer the producer warps refill for the next work item.

**This single mechanism explains every prior observation:** the single-warp gate on the O write is
load-bearing because it keeps the `sO` read window short, and parallelising it across all MMA
threads widened the window and took failures from 1/10 to 8/10.

**`tma_store_wait` is the missing piece** — and the goal names it. `store_tail()` is
`{ // tma_store_wait<0>(); }`, commented out. 0.7's `store()` writes O by TMA and relies on that
wait to drain the store before `sO`/`smem_v` is reused; our `store_partial` uses plain vectorised
stores with **no completion handshake**, so nothing orders the producer's next V load against the
consumer's `sO` read.

**Fix, not yet implemented:** a store-completion handshake before the producer may refill
`smem_v` — either a named-barrier arrival after the global write, or route the partial O write
through TMA and restore `tma_store_wait<0>()`. The latter matches 0.7 and is the same primitive
B2c's cross-role backfill needs.

### Also added

Harness: bad-row indices (`BADROWS`), a whole-slab degenerate scan, and a slab fingerprint. The
bad row is always a **q-tile boundary** (first row of a tile). The earlier degenerate-NaN
hypothesis was **refuted** — `nonfinite_O_rows = 0`, and the 3584 `-inf` LSE entries are the
correct identity for chunks beyond the causal bound.

## v0.54.0 — the gap is the split-KV path (78%); unsplit is already 1.45× FASTER (2026-10-10)

The cost scales with kv chunk count, not with anything fixed. `SM_SPLIT=all`, 30 iters:

| fixture | kv | chunks | fused ms | unfused ms | ratio |
|---|---|---|---|---|---|
| `fa3_1chunk` | 128 | 1 | **0.0223** | 0.0324 | **1.45× — +31% faster** |
| `fa3_multitile` | 256 | 2 | 0.0433 | 0.0314 | 0.72× |
| `fa3` | 2048 | 16 | 0.2574 | 0.0384 | 0.149× |

Direct confirmation — `MODE=fa3` with the split switched off: **0.0578 ms** vs 0.2586 ms split, so
the partial write plus merge is **0.2008 ms = 78% of the kernel's time** (unfused does the whole
problem in 0.0386 ms). Without it the fused kernel is within 1.5× of baseline while using *less*
parallelism, and on one chunk it already beats baseline by 31%.

**First positive signal in the project**: the fusion works; one bounded path is the whole deficit.
It also explains why the earlier prime suspects measured small — gather ~1.5%, pipeline depth ~1%.

### Attempted and REVERTED — parallelising `store_partial`'s O write

It writes 32 KB from a single warp (64 dependent 16 B stores/thread). Spreading it over all
`NUM_MMA_THREADS` gave **−12%** on `mixed` (0.2470 → 0.2183) but took `MODE=fa3` from **1 failure
in 10 to 8 in 10** (1–7 bad rows, rel 0.15–1.21). Reverted; revert re-verified at 1/10. The
single-warp gate is load-bearing — likely protecting `sO`'s lifetime, since `sO` is **unioned with
`smem_v`** and the producer warps may refill it for the next work item while the consumer reads.
The `store_partial_zero` uint4 vectorisation was kept (not implicated).

### Next

Split the 0.2 ms between `store_partial` and the merge kernel — unmeasured so far — then optimise
whichever dominates while respecting the sO lifetime.

## v0.53.0 — STAGES=2 runs and is no faster; the smem blocker was the backfill slot (2026-10-10)

### Measured — what actually blocked the second pipeline stage

| STAGES | `sizeof(SharedStorage)` | FINAL request | device max | launches? |
|---|---|---|---|---|
| 1 | 98352 | 169008 = 99328 + 69632 + 48 | 232448 | yes |
| 2 | 163920 | **234544** = 164864 + 69632 + 48 | 232448 | **no, by 2096 B** |

FA3's own 2-stage storage fits with **68 KB spare**. The overflow came from the **cross-role
backfill drain slot** (`max(r1,r2,r3) = 69632`) added **unconditionally** past the FA3 base — which
is why `SM_SPLIT=all` gave the same number. "STAGES=2 does not fit" was wrong twice (blamed first
on the fusion's reservation, then on a traits limit); both retracted.

Gated the slot on `fa2_can_run`: at `sm_split == nsm` every block is FA3-role, nothing drains
through FA2 storage. STAGES=1 drops 169008 → **98352** (70 KB freed); STAGES=2 launches.

### Measured — the stage count does not matter

`MODE=fa3`, `SM_SPLIT=all`: STAGES=1 → 0.2590 ms, STAGES=2 → **0.2563 ms**, both PASS 0.003279.
A 1% difference, so the single-stage pipeline is **not** the cause of the gap. With the packed-Q
gather at ~1.5%, both code-level explanations are now measured and small.

### Fixed — my own A/B had three H2D copies per iteration in the fused arm

`kv_len_t.to(DEV)`, `qo_indptr.to(DEV)`, `kv_indptr.to(DEV)` were inside the timed closure, so the
fused arm paid transfers the baseline never pays. Hoisting them: `mixed` 0.2919 → **0.2470**,
`fa3` 0.3053 → 0.2590. **15–18% of the measured cost was harness artifact.**

### Open — fixed per-call overhead is now the lead

Fused is 0.2470–0.2590 across two quite different workloads while unfused moves 0.0380–0.0771. A
cost that does not scale with work is fixed overhead. Next suspect: the ext allocates its
upstream-port workspace (`up_partial_o`, `up_partial_lse`, `up_page_table`) **inside** `fused_run`,
once per call inside the timed loop.

## v0.52.0 — configurable FA3 tile shape; the 2nd pipeline stage is still unreachable (2026-10-10)

`FA3_CTA_Q` / `FA3_CTA_KV` added so the second K/V stage can be bought by shrinking a tile.
`MODE=mixed`, `SM_SPLIT=128`, `SEED=0`:

| CTA_KV | STAGES | result |
|---|---|---|
| 128 | 1 | PASS 0.003279, **0.2952 ms** (baseline) |
| 64 | 1 | **NON-FINITE: 590976 / 924672 nan** |
| 64 | 2 | **NON-FINITE: 640128 / 924672 nan** |
| 128 | 2 | **cannot launch** ("invalid argument", smem) |

Both routes to a 2-stage pipeline are blocked, so the leading hypothesis for the 4× gap remains
untested. **`CTA_KV=64` failing is a new correctness bug** — something hard-assumes 128. Not the
planner's chunk alignment (chunks round to 128, which 64 divides), and the KV gather's derivation
looks sound at 64 on paper, so it needs a probe.

### Fixed — a parameter-order bug the call site could not reveal

`AttentionKernelTraits` is `<USE_TMA_LOAD_KV, HEAD_DIM_QK, HEAD_DIM_VO, CTA_Q, CTA_KV, NUM_STAGES,
…>`, and the original call site was `<false, 128, 128, 128, 128, STAGES, …>` — all four identical,
so the ordering was unverifiable from the call site. The first attempt set `HEAD_DIM_VO=64` and
tripped `static_assert(HEAD_DIM_QK == HEAD_DIM_VO)`. Same hazard as the positional aggregate
initialisers fixed earlier; the order is now read off `kernel_traits.cuh` and recorded at the call
site.

The non-finite guard added in v0.51.0 caught all three failing arms here immediately — this
morning's harness would have called them PASS.

## v0.51.0 — harness was silently passing NaN; and the Q gather is exonerated (2026-10-10)

### Fixed — the correctness harness passed non-finite output

NaN comparisons are all False, so a NaN output scored `rel = nan`, failed the `rel > tol` test and
came back as `bad rows 0 ... PASS`. Caught because `SKIP_PACKED_Q=1` — which removes the Q load and
*must* produce garbage — reported `PASS 0.002069` while its output fingerprint was `absmax=nan,
sum=nan`. **Every verdict this harness printed was unsound for any arm producing non-finite
output.** Now an explicit non-finite guard fails before any per-row norm; null-cased both ways
(`SKIP=1` → `NON-FINITE OUTPUT: 730112 of 924672`; `SKIP=0` → PASS; normal arm 6/6 after).

Also added an output fingerprint (`absmax`/`sum`/`nonzero`) so two builds can be compared for
"did this change anything at all" — the question that resolved this in one run.

### Measured — the packed-Q gather is NOT the bottleneck

With the skip confirmed effective: **0.2927 ms (normal) vs 0.2884 ms (gather removed)** at
`SM_SPLIT=128` → **~1.5%**. Caveat: the skip arm computes on NaNs and NaN arithmetic need not be
equally fast, but the loads are structurally gone. This was the *prime* suspect for the 4× gap
(it replaced TMA with a per-row `cp.async`), so the gap lives elsewhere.

Remaining suspects in order: `FA3_STAGES=1` (single-stage K/V pipeline vs the baseline's 2 — no
prefetch overlap, the kind of factor that costs multiples; `STAGES=2` cannot launch at these traits,
so the traits need revisiting); the scalar per-row O stores in `store_partial`/`write_packed`; and
the baseline being flashinfer's own tuned FA3 at full occupancy.

## v0.50.0 — fully-masked split chunks now write their partial slot (2026-10-10)

One of (at least) two nondeterminism sources found and fixed. Residual race remains at 1-in-20.

### Fixed

A split item whose chunk lies entirely beyond the causal diagonal hits `num_kv_tiles <= 0` and
took `store_zero` + `continue`, so it **never wrote its partial slot**. The merge counts the chunk
regardless — `EnumerateFusedWorkItems` chunks uniformly with no causal adjustment, by design, so
the count matches what sized the slab and built `merge_indptr`. The reduction therefore read
`partial_lse` at an untouched slot (0.0 in a freshly zeroed workspace) and folded a zero O in with
weight `2^(0 - lse_true)` — a ~1% dilution at a true LSE near 6.6, measured as rel **0.03311** on
row 0. `store_zero` was doubly wrong: it writes the final O, not the partial slab, through a
token-indexed mapping that is also wrong on the packed path.

New `store_partial_zero` writes the slot with **-inf LSE** (the merge's identity) and zero O.

**Why row 0:** under bottom-right causal with kv=2048 / qo=256, token 0 sees the *fewest* keys
(1793 of 2048), so it is the row most likely to own a fully-masked chunk. The two FA2 rows were
always clean.

**Evidence the fix is real** — and it is not the pass count: before, passing runs scattered over
rel 0.003273 / 0.004544 / 0.004985 / 0.003337 / 0.005239 across splits; after, **every passing run
returns an identical 0.003279 at every split**. Deterministic-when-correct is what removing an
uninitialised read looks like.

### Still open

Residual race: 1 failure in 20 runs, 1 bad row, rel 0.2151, still in `mixed` (co-resident). Ruled
out: `store_partial`'s barrier accounting is identical to 0.7's `store()`. Remaining candidates:
cross-role backfill (`fused_drain_queue` / device `cluster_cursor`) handing one work item to two
blocks, or ordering between the FA2 roles' writes and the shared FA3 partial slab. Next probe:
count executions per work item on device and assert exactly one.

## v0.49.0 — RETRACTION: an intermittent race; and the first performance numbers (2026-10-10)

### Retracted — "kernel correctness complete" (v0.48.0) is withdrawn

`mixed` produces **1 bad row of 258 at rel 0.16–0.35 in roughly 2 of ~25 runs**, against a 0.004
noise floor. It is a **race**, not data- or split-dependence:

- `SM_SPLIT=32`, `SEED=0`: FAIL 0.1604 one run, **PASS 0.004005** another — same build, same seed.
- `SM_SPLIT=24`, `SEED=0`: PASS in one sweep, FAIL 0.2904 in the next.

It appears in `mixed` (FA3+FA2 **co-resident**) and once in `fa3`. Suspect shared state between
roles — smem, a named barrier, or ordering between the partial writes and the reduction launch.

**Cause of the bad claim:** completion was declared from **single runs** of an **unseeded**
fixture, and `sm_split` — the kernel's defining parameter — was never swept for correctness. Now
fixed: `SEED=` (default 0) and a numeric `SM_SPLIT`.

### First timing A/B (`TIME_AB=1`)

One fused launch vs the unfused baseline of two separate launches. Timing runs only after the
correctness verdict and is skipped for any arm with a bad row. `MODE=mixed`, `SEED=0`, 30 iters:

| `sm_split` | fused ms | unfused ms | ratio |
|---|---|---|---|
| 8 | 2.088 | 0.069 | 0.033× |
| 16 (old default) | 1.825 | 0.069 | 0.038× |
| 64 | 0.502 | 0.069 | 0.134× |
| 96 | 0.375 | 0.069 | 0.181× |
| 116 | 0.314 | 0.069 | 0.226× |
| 128 | **0.296** | 0.069 | **0.242×** |

The default split cost **6.2×**: FA3 held all the work (258 tokens, kv 2048) on 12% of the SMs
while 116 SMs served 2 FA2 tokens. Even at the best split the kernel is **~4× slower than
unfused**.

Known causes, all correctness-first code already documented as slow: the packed-Q gather replaces
TMA with a per-row `cp.async` gather; `FA3_STAGES=1` (single-stage K/V pipeline); `store_partial`
and `write_packed` both write O with scalar per-row loops.

### Next

(a) Find the race — it invalidates every measurement until fixed. (b) Restore TMA/vectorised paths
for the packed Q load and O store. (c) Revisit `FA3_STAGES=2`. (d) Re-run the A/B and sweep.

## v0.48.0 — KERNEL CORRECTNESS COMPLETE: every fixture passes with split-KV on (2026-10-10)

| fixture | split | result |
|---|---|---|
| `fa3_split` 32 tok / kv 2048, 16 chunks | yes | **PASS 0.00325, 0/32** |
| `fa3_1chunk` | yes | **PASS 0.002071, 0/128** |
| `fa3` 3 req, kv 2048/300/512 | yes | **PASS 0.004888, 0/258** |
| `mixed` FA3+FA2 co-resident | yes | **PASS 0.003273, 0/258** |
| `fa2` | n/a | **PASS 0.002126, 0/258** |
| `fa3` unsplit (control) | no | PASS 0.002113 |

### Fixed — the split-KV causal bound is now absolute

Threaded as a quantity **separate** from `kv_len`, because `kv_len` is simultaneously the
iteration bound (`get_num_kv_tiles`, `kv_idx >= kv_len`) — widening it would make each chunk read
into later chunks' keys.

- **`unified_wgmma_scheduler.cuh`**: `WorkTileInfo::causal_kv_len = kv_len[work_idx] - kv_begin`.
- **`hopper/utils.cuh`**: `fused_causal_kv_len()` + a `fused_has_causal_kv_len` detector —
  *detected, not required*, since upstream's scheduler has no such field and `prefill_sm90.cuh`
  serves both. Returns `kv_len` when absent, so non-split and upstream paths are unchanged.
- **`sparse_mainloop.cuh`**: `get_num_kv_tiles(..., causal_kv_len = -1)`, and `load()` passes the
  same value — load and mma must agree on the tile count or the pipeline deadlocks.
- **`mainloop_mma.cuh`**: `col_limit_right` / `col_limit_left`.
- **`prefill_sm90.cuh`**: four call sites.

`min(kv_len, col_limit_right)` then handles all three cases with no extra logic: chunks before the
diagonal are unmasked, the diagonal chunk masks correctly, chunks beyond it are fully masked.

### K1n closed — five real bugs in four families

The unassigned `k_page_stride`/`v_page_stride`; a `CAUSAL` default resting on a measurement taken
while that bug made all keys identical; packed-GQA breaking every token-indexed causal bound (the
mask, and `get_num_kv_tiles`, where it *deleted* output); `merge_o_indices` units; and the
chunk-relative causal bound. Plus latent positional-aggregate initialiser bugs.

### Next

Remove the debug probes and take the first timing A/B. The ≥30% TTFT/ITL and ≥10% throughput bars
remain entirely unmeasured.

## v0.47.0 — split-KV root cause: the causal mask is chunk-relative (2026-10-10)

Found by dumping `partial_o` / `partial_lse` for known merge rows against a Python per-chunk
reference. `partial_o` is a *normalized* per-chunk output, so the comparison needs no LSE
convention — that is the split that reading could not make.

**The LSE identifies the key count exactly.** `log2(97) = 6.5999` vs measured `lse_got = 6.60012`
(token 0), and `log2(98) = 6.6147` vs `6.61413` (token 1). Each chunk sums over `token + 97` keys
instead of 128, and `O_rel` is 0.51–0.68.

**Why 97:** `col_limit_right = (qo_idx/G) + 1 + kv_len - qo_len/G` with the *chunk's* `kv_len=128`
and `qo_len/G=32` gives `token + 97`. The scheduler hands each split item a base shifted to its
chunk and `kv_len = kv_chunk`, commented *"Legal because this path is non-causal"* — but `CAUSAL`
is a **compile-time parameter applied to every item**, so once `FA3_CAUSAL=1` became necessary for
the unsplit path, every split chunk began masking against a chunk-relative bound.

**The merge is innocent of the value error.** `merge_o_indices[:6] = [0,1,2,3,28,29]` is exactly
right under the fixed units, `merge_indptr` strides by `chunks=16`, and the slot arithmetic
resolves correctly. It was faithfully combining partials that were already wrong.

### Still open — the fix, a design item not a one-liner

The causal bound must be absolute: for chunk `[kv0,kv1)` with total `kv_total` and `qo_tokens`, row
`token` may attend up to `kv_total - qo_tokens + token - kv0` within the chunk. Either thread the
chunk base and total kv through `BlockCoord`, or have the planner skip chunks beyond the diagonal
and mask only the diagonal chunk (which must agree with the `splits` count the partial slab was
sized from). Whatever carries the bound must stay in TOKEN units while `qo_idx` is a packed row.

## v0.46.1 — split-KV part 2 narrowed: no double write, residual is in the reduction (2026-10-10)

No code change; four measurements and two code reads that bound the remaining defect.

- **`FLASHINFER_FUSED_NO_MERGE=1` gives 0/32 and 0/256 rows non-zero.** There is **no double
  write**: the attention grid writes only partials for split rows, and every output row comes from
  the reduction. The direct-write path is exonerated for split rows.
- `store_partial`'s O slot and its LSE slot **both match** the merge's slot formula exactly, with
  `partial_indptr == group_base` and `merge_indptr[row] == group_base + r*chunks` by construction.
- Row counts check out: `fa3`'s `num_merge_rows=1032` = 8 tiles x 128 packed rows + 4 + 4 for the
  two single-token requests (kv 300 -> 3 chunks, kv 512 -> 4 chunks).
- The reduction launch wraps 0.7's `BlockBatchReductionPersistent::Run`, which self-covers via a
  grid-stride loop, so grid sizing cannot explain missing rows — only slowness.

Remaining: the **contents** of `partial_o` / `partial_lse` and the reduction's combination of them.
`partial_lse` is the leading suspect — a wrong LSE gives correctly-addressed rows with wrong
weights, exactly the residual's shape (all rows present, values off by O(1), `fa3_split` at 16
chunks, and 1 of 32 rows passing as a weighting error would leave where one chunk dominates).

## v0.46.0 — split-KV part 1: `merge_o_indices` units fixed (2026-10-10)

Every row is now written when split-KV is enabled; the merged **values** are still wrong, so this
is partial progress, not a pass.

| fixture | rows written before | after | rel err after |
|---|---|---|---|
| `fa3_split` | 2/32 | **32/32** | 0.5605, 31/32 bad |
| `fa3_multitile` | 6/256 | 128/256 | 8.134 |
| `fa3` | — | 130/258 | 11.43 |
| `mixed` | — | 128/256 | 11.43 |
| `fa2` | — | **PASS 0.002126** | unaffected |

### Fixed

- **`fused_plan.cuh` `BuildFusedMergeLayout`**: `merge_o_indices` was pushing the **token index
  alone**. Read off 0.7's own reduction: `BlockBatchReductionPersistent::Run` resolves its
  destination as `o_indices[packed_qo_idx] + kv_head_idx * gqa_group_size` into a `v_merged`
  viewed as `[qo_len, num_kv_heads, gqa_group_size, head_dim]`, so the value must be a flat row
  index over (token, qo_head): `(token_base + token) * (num_kv_heads * G) + head_in_group`. The
  old value was unscaled by `num_qo_heads` and dropped `head_in_group`, collapsing four packed
  rows of one token onto one destination.

  It explains the prior symptom exactly: values 0–31 plus `kv_head*G <= 24` land in flat rows
  0–55 of an 896-row space — token 0 and token 1 only, and the fixture reported precisely
  `2/32 rows non-zero`.

### Still open — split-KV part 2, the merged values

All rows are written and `fa2` is unaffected, so this is the reduction arithmetic or the partials
it reads. Next: `store_partial`'s slab offset vs the merge's slot formula (ordering and per-group
base), `partial_lse` (a wrong LSE gives right addresses with wrong weights — the exact shape of
the residual), and why `fa3_multitile` writes only 128 of 256 token rows.

## v0.45.0 — every fixture passes with split-KV off, `mixed` included (2026-10-10)

The fused 4-role kernel passes the causal fp32 reference on **every** fixture when split-KV is
disabled — including `mixed` at `SM_SPLIT=default`, i.e. FA3 and FA2 roles co-resident on one
grid, which has never passed before.

| fixture | split | result |
|---|---|---|
| `fa3_split` 32 tok / kv 2048 (null case) | yes | FAIL 1.326, 2/32 rows written |
| `fa3_split` | no | **PASS 0.002109, 32/32** |
| `fa3_multitile` 256 tok, 8 q tiles | no | **PASS 0.002109, 256/256** |
| `fa3` 3 requests, kv 2048/300/512 | no | **PASS 0.002113, 258/258** |
| `mixed` FA3+FA2 co-resident | no | **PASS 0.002113, 0/258** |
| `fa2` | n/a | **PASS 0.002126, 0/258** |

### Added

- **`fused_plan.cuh`**: `FUSED_NOSPLIT=1` forces the plan's `kv_chunk` to 0, which
  `EnumerateFusedWorkItems` reads as "one item spans the whole kv range" — no partials, no merge —
  at whatever sizes the fixture uses. This isolation did not exist: `fa3_1chunk` was the only
  unsplit fixture and also the smallest, so "unsplit" and "short kv, one q tile" were inseparable.
  Diagnostic only (it changes the plan), though it is a *correct* configuration rather than a
  disabled feature.
- **`drivers/fused_reference.py`**: `fa3_multitile` (8 q tiles) and `fa3_split` (1 q tile, long kv)
  fixtures; guarded a diagnostic that `IndexError`ed on any fixture with fewer than 33 tokens,
  killing the run before the measurement it was meant to explain.

### Still open — the only remaining kernel-correctness item

Split-KV + merge. `num_merge_rows` is *correct* (128 = per-head packed count, matching the
documented contract), but only 2 of 32 token rows come back non-zero and those two are wrong — so
most rows are never written, pointing at `merge_o_indices` and `store_partial`'s slab offset rather
than the reduction arithmetic.

## v0.44.0 — FA3 passes in the PRODUCTION config (packed-GQA, G=4) (2026-10-10)

`MODE=fa3_1chunk` at `FUSED_PACKED_Q=4`, 28 qo / 7 kv heads: **rel 0.002071, 0/128, PASS**.
`fa2` and the G=1 arms unchanged — no regression.

### Fixed — packed-GQA rows break every token-indexed causal bound

Our unified FA3 role tiles over PACKED rows, `(token, head-in-group)` pairs, because FA3's
mainloop validates the tile index against `qo_len`. 0.7's causal arithmetic is token-indexed, so
both the row index and `qo_len` are G times too large on the q axis. The token index needs an
integer division, so no choice of `qo_len` makes the stock expressions right.

- **`mainloop_mma.cuh`** (now in the overlay; stock md5 `c82eff57` verified against the pod before
  patching): `col_limit_right` / `col_limit_left`, the per-row mask bound.
- **`sparse_mainloop.cuh` `get_num_kv_tiles`**: the causal kv-tile count — and this is why most
  output was *missing* rather than wrong. At G=4 with 128 tokens (512 packed rows, 4 q tiles) the
  stock bound gives `(0+1)*128 + 128 - 512 = -256` for q tile 0, so `num_kv_tiles <= 0` and the
  body exits early **without storing**. Only q tile 3 survived, so 32 of 128 token rows were
  written and 96 stayed zero.

Both reduce to 0.7's expression exactly at G=1, which is why the G=1 arms passed either way.

- **`unified_wgmma_scheduler.cuh`** (latent, no measured symptom): both terminal returns were
  still positional — 13 and 14 values into a 15-field `WorkTileInfo`, so `counter` received
  `ptr_begin`, `ptr_begin` received `ptr_end`, and `num_kv_chunks` received `counter+1`. The
  comment in `load()` documents this exact hazard; the designated-initialiser conversion had been
  applied to `load()` only.

### Retracted in place

v0.42.0 concluded "the packed gather and packed store are BOTH EXONERATED" because all three arms
failed at the same magnitude. **That inference was invalid**: all three were dominated by the
page-stride bug, which made them equal regardless of their differences. With it fixed the same A/B
separates cleanly — G=1 passes, G=4 fails — and the packed path was implicated after all.
Comparing arms that share a dominant fault measures the fault, not the difference.

### Still open

Multi-chunk `fa3` (2.643) and `mixed` (2.066, 256/258). Single chunk passes in every configuration
and `fa2` is unaffected, so what remains is the split-KV / merge path (`num_kv_chunks > 1`,
`store_partial`, `merge_indptr` / `merge_o_indices`, the reduction). `SPLIT=all` (2.643) and
`SPLIT=default` (6.185) differ, so there is a fusion-dependent component on top.

## v0.43.0 — FA3 PASSES the causal fp32 reference for the first time (2026-10-10)

`sglang@b48fa7d2`, `flashinfer@ce514ae`, H200 pod `f929493da6e45f10`. Single-tile arm
`MODE=fa3_1chunk`: **rel 0.002292, 0/128, PASS**.

| step | fa3_1chunk | fa3 (multi-tile) |
|---|---|---|
| before | 11.8, 127/128 | 48.93, 258/258 |
| + page strides assigned | 1.001, 127/128 | 1.593, 258/258 |
| + `FLASHINFER_FUSED_FA3_CAUSAL=1` | **0.002292, 0/128 PASS** | 4.856, 258/258 |

### Fixes

- **`fused_fa3_persistent_ext.cu`**: assign `fa3.k_page_stride` / `fa3.v_page_stride`. They were
  never set, and `FA3ParamsSel fa3{}` zero-initialises them. 0.7's `sparse_mainloop` computes
  `page_idx * k_page_stride + entry_idx * k_stride_n`, so a zero page stride sent every page to
  offset 0 and, at `page_size=1` (`entry_idx` always 0), **every key read page 0**. The FA2
  persistent params already set their equivalents; only the FA3 selection block was missed.
- **`fused_fa3_loader.py`**: `FLASHINFER_FUSED_FA3_CAUSAL` is now settable via `FA3_CAUSAL`. Its
  default of 0 came from a *vacuous* measurement — "CAUSAL=1 is worse" was taken while the page
  stride bug made every key identical, a configuration in which causal and non-causal cannot be
  distinguished. The reference is bottom-right causal and the stock wrapper is `causal=True`.
- **`fused_fa3_persistent_ext.cu`**: corrected the stale kv-index convention comment. 0.3.1 wanted
  vector-sparse offsets; 0.7 wants raw page ids plus the page stride. They coincide at
  `page_size=1`, so this fixture cannot see the difference — flagged as a TODO before any
  `page_size>1` arm.
- **`drivers/fused_reference.py`**: `KVCONST=1` (all kv pages identical) and `GQA1=1` (G=1)
  discriminators.

### Retracted in place

The whole "PV accumulator duplicates rows / every output row equals reference row 0 / identity
matches 1/128" investigation was a **consequence of the page-stride bug**, not a finding about the
MMA: with every key reading page 0, all rows attend to identical keys, so every row's output is
legitimately V[0] — exactly reference row 0 under bottom-right causal. The accumulator was correct
on the input it was given. The "duplicated Q input" reframing is withdrawn too; Q was never
duplicated. `tOrO(0)==tOrO(2)` was measured correctly, but the inference from it to a broken MMA
was wrong.

### Still open

The multi-tile `fa3` fixture fails at 4.856, 258/258. Single tile passes, so what remains is the
multi-tile / split-KV / merge path (`num_kv_chunks`, `store_partial`, `merge_indptr` /
`merge_o_indices`, the reduction), not the mainloop.

## v0.42.0 — K1n: packed gather and packed store EXONERATED (2026-10-10)

`sglang@d1f78a58`, overlay `/dev/shm/w/port07`, H200 pod `f929493da6e45f10`.

**The A/B.** `FUSED_PACKED_Q=off` now omits `-DFLASHINFER_FUSED_PACKED_Q` entirely instead of
defining it to a value. Both the Q load and the O store are `#ifdef`-gated on that macro, so
omitting it reverts FA3 to **stock 0.7 TMA load + `write_tiled`** while still running through our
extracted `fa3_prefill_body` and `UnifiedWgmmaTileScheduler`. Valid only at **G=1** (`GQA1=1` ->
HQ=HKV=7), where packed rows and tokens coincide.

| arm | GQA1 | PACKED_Q | MBARRIER | rel err | bad rows |
|---|---|---|---|---|---|
| g1stock — stock TMA load + `write_tiled` | 1 | off | 0 | **48.93** | 258/258 |
| g1packed — our gather + `write_packed`   | 1 | 1   | 1 | **47.62** | 258/258 |
| g4packed — our gather + `write_packed`   | 0 | 4   | 1 | **45.24** | 258/258 |

All three fail at the same magnitude, so the FA3 defect is **independent of** the packed gather,
the packed store, the Q-barrier arrival mode **and** the group size. With the earlier
`SM_SPLIT=all` A/B (bit-identical output) it is independent of the fusion too. What remains is
`fa3_prefill_body` or the scheduler/params handed to it. This retires the gather addressing, its
coverage, `write_packed` and the arrival mode as candidates — the failure survives compiling all
of them out.

### Fixes

- **`prefill_sm90.cuh`**: `barrier_Q` arming was gated on `#if defined(FLASHINFER_FUSED_Q_MBARRIER)`
  while the arrival side tests the macro's **value**. Once the flag became settable, `-D...=0` was
  still *defined*, so the drain form's single arrival met a barrier armed for 32 and the kernel
  hung (GPU pinned at 100% for 440–470 s on a microsecond fixture). Now
  `#if FLASHINFER_FUSED_Q_MBARRIER && defined(FLASHINFER_FUSED_PACKED_Q)` — value-tested, and
  requiring the packed path, since the 32-lane arrival exists only there. The in-code comment had
  warned about exactly this count mismatch while the code it guarded could not express it.
- **`fused_fa3_loader.py`**: `FUSED_PACKED_Q` and `FA3_MBARRIER` are now switchable, with `off`
  omitting the macro rather than zeroing it.
- **`drivers/fused_reference.py`**: `GQA1=1` collapses the fixture to G=1.

### Retracted in place

The Q-load `#else` arm's comment claims it is *"Correct and ~3x slow"*. Its slowness may be a real
0.3.1-era measurement; its **correctness was never measured on this stack** — FA3 has never
passed, so there was nothing to measure it against — and as first run it **hung**. It is not a
known-good reference.

## v0.41.0 — FA3's illegal access fixed; FA2 still passes (2026-10-10)

`flashinfer@c7fdd92`, `ec02a20`. `MODE=<m> ... python drivers/fused_reference.py`:

| fixture | before | after |
|---|---|---|
| `fa2` | PASS | **PASS** rel 0.002126, 0/258 |
| `fa3_1chunk` | **illegal access** | no fault, rel 3.076e+06 |
| `fa3` | 44.84 | 45.1 |
| `mixed` | 41.67 | 45.44, 256/258 (FA2 rows pass) |

- **`WorkTileInfo`'s positional initialiser was misaligned**: the struct is
  `(partial_indptr, group_id, kv_chunk_idx, num_kv_chunks)` but the initialiser listed
  `kv_chunk_idx` before `group_id`, so `group_id` received the chunk index — 0 on the first chunk,
  opening the `group_id >= 0` gate on every row. The FA3 epilogue then wrote partials for rows the
  plan marked unsplit (`partial_indptr = -1`), running off the front of the float workspace. Now
  designated initialisers — the **fourth** positional aggregate this port has had to fix the same
  way.
- `kv_indptr` was double-counting the FA3 pages (the caller rebases the pointer, so the plan's
  offsets must be subset-relative), which is why FA2 rows were wrong in `mixed` while identical
  rows passed in `fa2`. Fixed; FA2 is now correct in both.
- Method note: the diagnosis came from printing the kernel's own values from **every** block and
  checking them against arithmetic — all 16 FA3 blocks reported the impossible triple
  `group_id=0 chunks=1 partial_indptr=-1`, and `-44/4 = (-2)*7 + 3` matched the sanitizer exactly.
  Single-block traces and reasoning from silence produced three retractions earlier in this item.
- Open: the FA3 role is off by 3.076e+06 on the simplest unsplit fixture — a magnitude indicating
  unnormalised/uninitialised output, not an index slip. Still no performance number.

## v0.40.0 — the FA2 path passes the fp32 causal reference (2026-10-09)

`flashinfer@cc56241`. First correctness result of the flashinfer 0.7 port.
`MODE=<m> ... python drivers/fused_reference.py`:

| fixture | before | after |
|---|---|---|
| `fa2_2tile` q=256 kv=256 | rel 1.991, 224/256 bad | **PASS** rel 0.0021, 0/256 |
| `fa2` q=256 kv=2048 | **illegal access** | **PASS** rel 0.002126, 0/258 |
| `mixed` | rel 39.68 | FAIL rel 40.57 |

- All three FA2 tile classes, a row spanning 16 KV tiles, and bottom-right causal masking are
  correct at a bf16-consistent residual. The illegal access and the numeric error had **one**
  cause: the plan added `q_start` into `q_indptr`, double-counting a tile offset the kernel
  applies itself, against an unsplit write of `final_o + q_indptr * o_stride_n`.
- Also fixed on the way: `num_pers_clusters` was in item space rather than CTA-slot space, which
  silently dropped 26 of 56 tile-128 work items; and `len_kv_chunk = 0` for the FA2 classes was a
  device divide-by-zero, since `persistent.cuh` divides by it unconditionally.
- Method note: three conclusions in this item had to be **retracted**, all from reading an
  instrument's silence (a branch-gated trace never shown to fire, twice; and a unit audit written
  from the same understanding that produced the bugs). What worked was a trace printing from every
  block, giving a positive signal that could be checked against arithmetic — the `q_indptr`
  sequence 0,128,…,896 predicted exactly the 32-of-256 passing rows already observed.
- `MODE=mixed` remains FAIL, now isolated to the FA3 role and/or the merge. No performance number
  yet, and the ≥30%/≥10% bar is still untested.

## v0.39.0 — the plan builder runs and satisfies the merge contract (2026-10-09)

`flashinfer@6aa16fc`. `port07/tests/test_plan_build.cu` on the vibecuda H200: **19/19 PASS**.
First executed result of this port.

- `FusedFourTaskPlan`'s device half is written: chunk choice (one per FA3 SM, rounded up to
  `TILE_KV`), work-item enumeration, merge layout, per-task `work_indptr` (FA3 over exactly
  `sm_split` CTAs), a bump allocator over page-locked staging, and one H2D copy.
- `BuildFusedMergeLayout` rewritten to `BlockBatchReductionPersistent::Run`'s contract — 0.7's
  merge for the *holistic* path. Reading `cascade.cuh` instead produced two claims that had to be
  retracted; that kernel takes no `o_indices`, this one does. Four disagreements fixed:
  per-packed-row `merge_indptr`, the list compacted over split rows, `kv_head_idx` out of the
  grouping key, and one shared base per group. `partial_rows` 218624 → 31232 (the 7× head
  over-count), slab 112 MB inside budget instead of 784 MB over.
- The subtlest of those would not have crashed: giving each chunk its own base is the same slab
  size in a different permutation, so chunk 0 of row r collided with chunk r of row 0 and the
  merge would have reduced the wrong partials silently.
- The gate checks the contract equality itself. `run_host_checks.sh` passed throughout all three
  bugs because it only asserted monotonicity, counts and the round trip.
- Plan is now **59** entries (15 shared + 4×11); `num_merge_rows` added; `num_fa3_clusters` still
  last for `plan_info[-1]`.
- Still no attention output: whether the dispatcher launches the merge is unverified (K1j), and
  the fp32 reference has not run.

## v0.38.0 — the fused kernel is a loadable module on flashinfer 0.7 (2026-10-09)

`torch.ops.fused_fa3.fused_run` registers. `sglang-up@f019410ff`, `flashinfer@b883589`.

- **Merged include tree, not an `-I` ordering.** The loader put the wheel first and the source
  tree last — right for 0.3.1, where the trees were disjoint; wrong for the 0.7 overlay, which
  deliberately replaces same-named headers. Reversing the order fails too, because both trees
  use relative includes that resolve against the including file. The loader now builds what
  `nvcc_check.sh` builds: the wheel's tree with the overlay laid on top, content-stamped and
  cached.
- `store_partial` is ported and **genuinely compiled**. It sits behind an `if constexpr` that is
  false for the default scheduler, so three earlier green compiles never covered it. The gate
  now has a **null case** — a type error injected inside the function must break the unified
  build — and both steps are permanent in `nvcc_check.sh`.
- The plan's per-item `group_id` is gone for good: its only surviving role was the boolean "does
  this row need a merge", now derived as `num_kv_chunks > 1`. Removing its initializer without
  removing the field was the **fourth** positional-aggregate shift in this port, and the one
  that produced six type errors three fields away from the edit.
- **Nothing has executed.** The remaining blocker is not the harness but `FusedFourTaskPlan`'s
  CUDA body (K1c2): `fused_run` needs a `pers_plan` and nothing can produce one yet. No
  correctness claim and no A/B until the fp32 reference passes.

## v0.37.0 — FA2+FA3 co-inclusion, and the fused kernel body type-checks on 0.7 (2026-10-09)

The instantiating TU (`fused_fa3_persistent_ext.cu`, what the loader actually builds) now
compiles far enough that nvcc type-checks the kernel itself. `flashinfer@4ef2407`,
`43d1f64`, `0bc00af`; `sglang-up@56cad32fa` and its follow-up.

- **The previous compile gate was nearly empty.** A template is type-checked where it is
  instantiated, and nothing in the header-parse TUs instantiates the fused kernel:
  `cuobjdump -elf` reports **zero device kernels** in a 687 KB object, the only fused symbol
  being a constant. `NVCC_FUSED_OK` therefore said the headers parse together and nothing about
  the kernel body — including the reading that `fa3_upstream.cuh` was clean against 0.7, which
  it did not support. The instantiating TU is now the third step of `nvcc_check.sh` and reports
  the device-kernel count rather than OK/FAILED.
- **The fused kernel is the first thing in 0.7 to include both the FA2-lineage and Hopper
  trees.** Upstream's POD attention, which looks like the precedent, is FA2-only. 0.3.1 never
  hit this because its upstream carried `...Sm90` suffixes that 0.7 dropped. Five name
  collisions resolved by restoring those suffixes with unsuffixed aliases behind
  `-DFLASHINFER_SM90_SUPPRESS_UNSUFFIXED_NAMES=1`; a tree-wide rename is unavailable because
  every stock FA3 module is generated with `variant_name = "DefaultAttention<...>"`. The four
  `DEFINE_HAS_MEMBER` probes are identical in both trees and use upstream's own
  `FLASHINFER_SHARED_HAS_MEMBER_PROBES` guard; the four `REGISTER_*` macros differ only in the
  name of a template type parameter, so they are guarded per-macro on both sides rather than
  per-file (FA2 defines two macros Hopper lacks).
- **`cluster_cursor` / `num_pers_clusters` no longer live in `PersistentParams`.** That struct is
  generated from upstream's `batch_attention_customize_config.jinja`, so carrying them there
  meant patching a template inside the installed wheel — unreproducible on a fresh pod, and it
  would make every stock `BatchAttention` build grow two fields it never uses. They are now a
  `FusedQueueParams` argument. The 0.3.1 tree patched that jinja; 0.7 does not need to.
- 0.7 plan format in the TU: `FusedPlanInfo` replaces `HolisticPlanInfo`, and the six group
  tables are gone with the JIT-reduction rendezvous, replaced by the scheduled merge over
  `partial_indptr` / `merge_indptr` / `merge_o_indices`.
- Two 0.7 config-discovery fixes in the loader: `BatchAttention.plan()` dropped the `batch_size`
  positional (every later argument shifted by one, surfacing only as *"plan() got multiple
  values for argument 'causal'"*), and the generated-config path gained an arch segment while
  sglang redirects the cache base at import time. Both now asked of flashinfer rather than
  globbed.
- Three of the four 0.7 API deltas inside the kernel are closed (`36a89de`), verified by the
  instantiating TU: **18 → 3 errors**. `NUM_KV_STAGES` turned out to be our own name that 0.3.1's
  FA2 `KernelTraits` supplied a default for and 0.7's does not (now a `fa2_kv_stages` probe);
  `Runner::Run`'s `s_merge` went with the rendezvous; and `threadblock_sync_mdo_states`'s
  defaulted `barrier_id` is ours and had to be carried onto 0.7, which matters because
  `__syncthreads()` compiles to `barrier.sync 0` — the same hardware barrier FA3 uses as
  `kQueryEmpty`.
- **The fused kernel compiles against flashinfer 0.7** (`1208ca4`): `NVCC_EXT_OK`, **10 device
  kernels** in the object. The last three errors were one cause — 0.7 *inserted*
  `k_page_stride` / `v_page_stride` / `page_size` into the sparse mainloop's `Arguments` after
  `window_left`, and because every field it shifted onto happened to be assignable, the
  positional init failed three fields later as *"no suitable conversion from AdditionalParams to
  int64_t"* rather than where the change was. Both call sites now use designated initializers.
- **This is not a correctness result, and the split-KV path is incomplete.** `store_partial` is
  entirely ours (0.7's epilogue has only `store` / `store_tail` / `store_zero`) and is **not
  ported**: 0.3.1 defines it and calls it twice, the 0.7-based copies have zero of each. The six
  epilogue fields added here are plumbed but unread and default to `nullptr`, which is precisely
  why the TU compiles — any FA3 row that splits over KV writes no partial and the merge reduces
  memory nothing wrote. That is K1g, to be done with K1d2's deferred packed-GQA mbarrier change.
  **No A/B, and no correctness claim, before the fp32 reference passes.**

## v0.36.0 — persistent.cuh compiles against flashinfer 0.7 (2026-10-09)

`NVCC_PARSE_OK` from `port07/tests/nvcc_check.sh` (merged 0.7 + overlay include tree, sm_90a,
C++20). First file of the fused-kernel port to build against the current stack.

- Error progression 7 → 5 → 2 → 0, one commit per fix: barrier ids into their own header
  (`34e5cf0`), `cluster_override`/`tid_base`/`kNoClusterOverride` hand-applied to `Run`
  (`933a655`), K/V offset arrays separated (`0d34c00`), `KVStagedTraits` moved to namespace scope
  (`2c79439`).
- **Latent bug in our own code, surfaced by the port**: our KV-ring lambda prefetched offsets
  once with K's strides and reused them for the V produce — correct only while V's strides equal
  K's, and plausibly why 0.7 split the array into `_k`/`_v`. Now two prefetches. Correctness fix,
  not a rename.
- **Second structural corruption from fuzzy patching**: macros and `KVStagedTraits` had been
  placed inside a struct template where C++ forbids template declarations. Both corruptions were
  invisible to symbol greps and found only by compiling; remaining files are hand-ported.
- Also host-verified this session: `FusedPlanInfo` round-trip (58 entries), work-item enumeration
  (task3=256, `max_splits_fa3=4`), merge layout (64 tiles / 32768 partial rows, CSR monotonic,
  unsplit rows absent).
- Still unported: `persistent_fa3_fused.cuh` (+948, the kernel), the two wgmma headers, and the
  planner's CUDA body. Bar remains unmet — attention+epilogue caps at 29% of GPU time.

## v0.35.0 — B2b: the two-executor split is a dead end; batch-shape hypothesis retracted (2026-10-09)

`sglang-up@91955d929`, `drv/split_perf_mtp.sh MRR=96 MC=64 RR=24 NP=240`, per-arm warmup, arm
order reversed with `base` last, graphs off.

- Best split arm **−9.1% tok/s** (13225.3 vs 14545.5), **+9.1% mean ITL**, **+8.0% mean TTFT**.
  Accept length identical at 2.07 across arms.
- **Batch-shape hypothesis RETRACTED**: this run has `fa2_rows=62`, *more* favourable than the
  `fa2_rows=40` where the isolated 1.224× was measured, and the split is worse than it was at
  `fa2_rows=7`. Shape is not the explanation.
- **Scatters eliminated as an explanation**: `views=6001` (logged, not inferred) confirms the
  scatter-free path fired on every dispatch, removing B2's measured 1.2%-of-kernel-time
  `index_copy_` cost, with no improvement.
- **`SGLANG_FUSED_SKIP_BASE_PLAN` is harmful** — that arm is worst (12169.8 tok/s, P99 TTFT
  2040.8 ms). It leaves `forward_metadata` pointing at unplanned wrappers. Remove, don't tune.
- Mechanism: flashinfer's single `BatchPrefillWithPagedKVCacheKernel` already schedules work
  items across the whole varlen batch; splitting into two launches discards that global balancing
  and adds a second tail. The isolated 1.224× was against a different baseline, so it never
  transferred.
- Consequence: the two-executor variant (the goal's "also acceptable" fallback) is measured as a
  dead end at serving scale. Only a single-grid fused kernel keeps one launch and one scheduler
  while specialising per row class — so the remaining kernel hope rests entirely on K1.

## v0.34.0 — B1b RETRACTS B1: the GPU is not underfilled at load (2026-10-09)

Driver `drv/b1b_headroom.sh` (load ramped before the capture window so decode rows are resident),
`MIXEDSPEC_STEPS 201`, `FUSED_SPLIT_CALLS 6001 (fa3_rows=1 fa2_rows=61)`, 1653.3 ms GPU time over
45617 launches — 25x B1's GPU sample, on the 4-mode path B1's trace never exercised.

- **Sub-occupancy launches are 8.1% of GPU time, not 89.3%.** B1's figure came from a trace that
  was 355 MB but held only 66.5 ms of GPU time — mostly CPU/idle events in a warmup-like window
  captured before any decode backlog existed. The motivation framing built on it ("the GPU is
  chronically underfilled") is **retracted**; at sustained load the device is well filled.
- Representative role split: **gemm 70.4%** (only 8.7% sub-occupancy), **attention 20.5%** (3.1%),
  **epilogue 8.5%** across 21711 launches, memcpy 0.5%.
- Item A's ceiling rises to attention's 20.5%; item B is bounded by 8.5%. Their sum, 29%, is the
  absolute ceiling if both were made free — so the ≥30% bar stays unreachable from attention and
  epilogue, agreeing with the `g*a*0.18` ~5–7% estimate by a different route.
- **B1c answered negatively and dropped**: GEMM-level cross-role backfill is not the opportunity,
  since GEMMs are only 8.7% sub-occupancy at load.
- New item B1d: emit the time-weighted `packed_qo` histogram. Grid-level occupancy cannot see
  *within*-tile waste, which is exactly what item A targets.

## v0.33.0 — B1: the fusion headroom is 89% sub-occupancy, but mostly GEMM (2026-10-09)

Driver `drv/b1_headroom.sh` + `drv/b1_analyze.py` on a 4-mode MTP server (EAGLE3 chain draft,
split ON, guard OFF, graphs off), 355.7 MB torch-profiler trace, 66.5 ms GPU time / 7250 launches.

- **89.3% of GPU time is in launches with grid < 132 blocks** — they cannot fill the H200
  regardless of inner-loop speed. 6763 of 7250 launches.
- Role split: **gemm 78.5%**, attention **11.6%**, epilogue/elementwise **8.9%**, memcpy 1.0%.
  Top kernels are cuBLASLt `nvjet_sm90_*` at median grid 100–112.
- So the attention-side tile dispatcher is bounded by 11.6% of GPU time and epilogue fusion by
  8.9% — ~20% combined before discounting the non-reclaimable part. **No attention-side work can
  reach the ≥30% target**, agreeing with the independent `g*a*0.18` ceiling of ~5–7%.
- The defensible motivation is therefore "the GPU is chronically underfilled in colocated
  spec-decode serving and cross-role fusion reclaims it", with attention as the demonstrator —
  and the table shows the dominant idle capacity is GEMM, which the current scope does not touch.
- Scope caveat: `MIXEDSPEC_STEPS 1` vs 501 split calls means the capture is dominated by ordinary
  extend batches, not 4-mode merges; a 4-mode-heavy re-capture is queued (B1b).

## v0.32.0 — Fused single kernel is bound to flashinfer 0.3.1; split is the only runnable variant (2026-10-09)

- Root cause of six consecutive harness/extension incompatibilities: the flashinfer SOURCE tree
  the kernel compiles against is **0.3.1**, while the installed wheel sglang uses is
  **0.7.0.post1**. The kernel's `HolisticPlanInfo` is 56/68 entries; 0.7's `BatchAttention`
  emits 28. All 32 harness cases error on the kernel's own plan-length guard.
- **The "1.30x fused" result was measured on 0.3.1** and says nothing about the current stack. It
  stands as history; citing it as support for the fused variant today was wrong.
- The fused single kernel therefore **cannot be wired into the current server** without porting
  it to 0.7's plan serialisation — C++ work against flashinfer internals, not integration.
  Increments 2b/3 are blocked, not unstarted.
- Increment 1 did pass: `fused_fa3` compiles from inside sglang with C++20 and registers
  `torch.ops.fused_fa3.fused_run` (`sglang-up@9da582ac3`).
- Runnable variant remains the two-executor split: correct (standalone 5/5 fp32; in-server 5001
  calls, 10/0 verify) and **−5.0% tok/s** in serving, with a measured ±1.5–3% noise floor.

## v0.31.0 — Fused split PASSES in-server; the device assert was a use-after-free (2026-10-09)

`sglang-up@056f28667` for the PASS, `f53707f53` for the fix. Driver `drv/split_verify_ngblk.sh`
(4-mode MTP server, EAGLE3 chain draft topk=1 / 4 draft tokens, `SGLANG_MIXED_SPEC=1
SGLANG_FLASHINFER_USE_PAGED=1 SGLANG_FUSED_DECODE_GUARD=0 SGLANG_FUSED_SPLIT_VERIFY=1
CUDA_LAUNCH_BLOCKING=1`).

- In-server: 151 4-mode merges, **4501 split calls**, two-sided partitions on real merged batches
  (`fa3=2 fa2=25`), in-step reference **9 ok / 0 FAIL** (err 0.0041–0.0062, worst 0.0078, tol
  0.02), **0/56** client errors.
- Root cause of the device-side assert: `_fused_split_plan` did `non_blocking=True` H2D copies
  from Python temporaries, a use-after-free that wrote garbage into index tensors. Blocking ON →
  4501 clean calls; blocking OFF → assert after one. Fixed by making the copies synchronous.
- Four in-server bugs in the dispatch, all invisible to the standalone fp32 gate (which never
  builds a backend): pool read off `ForwardBatch`, lazily created counter read before its
  increment, mid-step wrapper on the shared workspace, and the async copies above.
- Retractions recorded in place: the shared-workspace diagnosis (not the cause; the fix is kept
  on its own merits) and the earlier retraction of the asynchrony hypothesis (right for its
  evidence, overturned once the split fired in both arms).
- No-blocking confirmation run pending; the claim is not quoted unqualified until it passes.

## v0.30.0 — SM pre-assignment measured: 1.22x attention, and the bar arithmetic closes (2026-10-09)

`sglang-up@b8b405822`, harness `/dev/shm/w/drv/split_margin.py`, H200, torch 2.14.1,
flashinfer-python 0.7.0.post1, qwen7b shape (GQA 4), bf16, page_size 1, 30 iters / 5 warmup.
Batch: one 1024-token chunk + 40 verify rows — a shape the guard ACCEPTS (`fa3=1 fa2=40`).

- Baseline (one planned wrapper over all 41 rows, i.e. the unsplit path): 186.7 us.
- Best split: **152.6 us at `sm_margin=8` → 1.224x**. Flat within 0.3% over margins 0–16, then
  monotone decay; `sm_margin=96` is 0.956x, i.e. **below baseline**.
- Correctness is bit-identical across all six margins (err 0.00236), as it must be — `sm_margin`
  changes launch geometry, not arithmetic. The SM budget is a free tuning knob.
- Mechanism finding: the optimum is a SMALL margin, so FA3 wants nearly the whole GPU and the
  win comes from the two executors sharing the grid, not from a careful SM carve-up. This
  contradicts the "pre-assignment ratio matters" framing.
- **Bar arithmetic, now measured rather than inferred:** 1.22x attention = 18% less attention
  time, so end-to-end ITL gain is `g * a * 0.18`. At `g=1` and a generous `a=0.3` that is 5.4%;
  reaching 30% from attention alone would need `a > 1`. The ≥30% TTFT/ITL target is therefore not
  reachable from this kernel. A 30% TTFT win has to come from queueing (P99 TTFT 7.3–33 s vs a
  0.73–1.32 s mean), i.e. the EDF/admission-order lever.

## v0.29.0 — Fused split dispatch PASSES the causal fp32 reference, 5/5 (2026-10-09)

`sglang-up@b8b405822`. Harness `/dev/shm/w/drv/split_unit.py`; command
`cd /dev/shm/w && PYTHONPATH=/dev/shm/w/up/python venv/bin/python drv/split_unit.py`;
H200, torch 2.14.1, flashinfer-python 0.7.0.post1, qwen7b shape (GQA 4), bf16, page_size 1.

- Scale-normalised error 0.00232–0.00424 against a row-by-row fp32 causal reference, tolerance
  0.02, across five shapes including the 4-mode shape and non-round ragged lengths. Each row
  subset is scored separately, so neither half masks the other.
- Validates the FA3 subset's page-table gather, `cache_seqlens` and per-subset `cu_seqlens_q`;
  the FA2 subset's plan through flashinfer's public API with explicit indptr/indices; and the
  `index_copy_` scatter combine. Does NOT validate `sm_margin` or any perf claim.
- Deterministic, so the noise floor is zero — which is why this gate yields a verdict where
  three server-level ON/OFF attempts returned INCONCLUSIVE or NOT EXERCISED.
- **Coverage measured at the shape level:** the decode-population guard declined 4 of the 5
  shapes (`fa3=0`). A 2048-token chunk is 16 KV tiles and needs ≥32 co-resident decode rows, so
  the canonical 4-mode step is refused. This strengthens the prediction that the ≥30%/≥10% bar
  is unreachable under the current gating and makes the guard-OFF arm the most informative
  remaining measurement.

## v0.28.1 — 4-mode root cause: a stale CPU seq_lens mirror (2026-10-09)

`sglang-up@b8b405822`. Found while trying to gate the fused split, which could not run until
this was fixed. Reproduces with the split OFF, so it is a port defect, and it blocked every
4-mode serving arm.

- `ForwardBatch.init_new` passes `extend_prefix_lens = batch.prefix_lens` directly; that field is
  an independent list, NOT derived from `seq_lens - extend_lens` as a comment in
  `mix_with_verify` claimed. The planner builds `qo_indptr = cumsum(seq_lens - prefix_lens)`, so
  a disagreeing prefix silently resizes the step and surfaced as flashinfer's
  `q.shape[0] (2051) does not match qo_indptr[-1] (2055)` several frames away.
- Measured: `tail seq_lens=[2047, 6703] prefix=[0, 6695] qo=[2047, 8] extend_seq_lens=[2047, 4]`.
  `running_batch.seq_lens_cpu` (6696) is stale against the device `seq_lens` (6699) --
  `eagle_prepare_for_verify` advances the tensor without refreshing the mirror -- so `seq_lens`
  came from the tensor and `prefix_lens` from the mirror.
- Fix: derive the host copy from the device tensor, repairing `seq_lens_cpu` and `prefix_lens`
  together. One D2H of `bs` ints per 4-mode step, marked `TODO(sync)`.
- **Retraction:** the first fix (`6f6ea6c10`) acted on an inferred "deficit = 1 per verify row"
  theory and used the stale mirror. The deficit is `n` per row; the ragged path's smaller
  deficits are a different construction, and forcing one theory across both paths produced the
  wrong fix.
- Also: `SGLANG_FLASHINFER_USE_PAGED=1` is required for the split dispatch to be reachable at
  all — upstream defaults `use_ragged` True, and the dispatch lives in the paged branch.

## v0.28.0 — forward_extend dispatch: the fused FA3+FA2 split is wired on upstream (2026-10-09)

`sglang-up@86e375375` (`upstream_port`, 65 commits on upstream main `3fef129ff`).

- `FlashInferAttnBackend._fused_split_dispatch` runs the two executors over the disjoint row
  subsets and combines them with `index_copy_` -- a scatter, not an LSE merge, because the subsets
  are disjoint (unlike upstream's ragged+paged pair, which covers the same rows).
- **The SM budget needs no private entry point.** Upstream FA3's `flash_attn_with_kvcache` already
  exposes `sm_margin` ("can be tuned if some SMs are used for communication"). This corrects the
  earlier claim that the SM budget cannot cross upstream's API: that holds only for flashinfer's
  persistent wrapper, which derives `sm_count` from the device.
- FA2's subset is planned through flashinfer's public `begin_forward` with explicit
  indptr/indices, so no second `KVLocPlan` is needed; the FA3 page table is a gather from
  `req_to_token`, which is already in upstream's documented `[batch, max_blocks]` shape.
- Index arithmetic CPU-verified 400/400 random batches. The test caught a real bug: the FA3 page
  table was indexed by batch row instead of `req_pool_index`, so it would have read another
  request's pages whenever the two orders differed.
- Four features are refused rather than approximated, each with a concrete TODO naming the
  mechanism (sliding-window, fp8 KV descale convention, multi-item scoring, cross-attention) --
  silently wrong attention would not show up in a throughput A/B.
- Always-on `FUSED_SPLIT_CALLS` counter; default OFF via `SGLANG_FUSED_FA3_SM_SPLIT`.
- Not yet GPU-verified; the correctness gate (greedy ON vs OFF with an OFF-vs-OFF noise floor) is
  running on the pod.

## v0.27.4 — Dispatch contract grounded: upstream already has a two-executor forward_extend (2026-10-09)

- `forward_extend`'s ragged path already runs TWO executors and combines their `(o, lse)` returns,
  so a multi-executor forward_extend is idiomatic upstream. Corrects the earlier "one planned
  wrapper per batch" framing -- that is only the `use_ragged=False` path.
- Our split differs in a way that is strictly SIMPLER: the two executors cover disjoint row subsets,
  so the combine is a scatter, not an LSE merge, and `forward_return_lse` is not needed.
- With commits 62-64 (row partition, dense_block_table, fa3_subset_varlen) the full data path is
  implemented and CPU-verified; the remaining work is the two executor calls plus GPU verification.

## v0.27.3 — Seam conversion landed on upstream, CPU-verified (2026-10-09)

- `mixed_dispatch.dense_block_table` converts sglang's flat paged KV indices into upstream FA3's
  documented `[batch_size, max_num_blocks_per_seq]` block table -- the fusion boundary built as an
  explicit conversion, since aliasing the two representations caused an illegal memory access.
- 11/11 CPU-verified properties, including the two load-bearing ones: -1 fill (page id 0 is valid,
  so zero padding would make short sequences read page 0), and a batch dimension equal to the FA3
  subset size in row order (upstream's combine scheduler maps tiles from the token-extent/batch-count
  pair).
- `upstream_port` is now 63 commits on live main, with the row partition (62) and seam conversion
  (63) both default-inert and verified without a GPU.

## v0.27.2 — Upstream fusion boundary specified end to end (2026-10-09)

- Answered the last design unknown from flashinfer's code: `prefill.py:204` derives `sm_count` from
  the device, so upstream's wrapper API cannot carry a caller-supplied SM budget -- the SM split must
  be expressed at the module call. Upstream's dense block-table contract is confirmed verbatim at
  `:1646` / `:1861` as `[batch_size, max_num_blocks_per_seq]`.
- So the upstream-fitting boundary is: standard upstream API for metadata, row partition and the
  per-class data contracts, with a NARROW bypass for the launch budget only -- not the fork's
  900-line divergent backend. Also noted honestly: some bypass is unavoidable, so the fork's
  replacement was not purely sloppiness.
- Step (a) (decode guard) is ported and re-verified upstream (11/25 -> 23/25 on that copy).

## v0.27.1 — upstream_port rebased onto live upstream main, zero conflicts (2026-10-09)

- All 60 commits replayed onto `3fef129ff` (was pinned 46 commits back at `b7b2975b5`) with no
  conflicts; backup branch `upstream_port_backup_20261009` and tag `pre_rebase_20261009` retained.
  Port markers verified present after replay rather than inferred from the exit code.
- Gap found by that verification: the decode-population guard lives only on the fork, not on
  `upstream_port`. It ports first -- self-contained Python with a measured acceptance test
  (11/25 -> 23/25 against the boundary surface).

## v0.27.0 — Direction corrected: kernel must land on upstream with upstream-shaped interfaces (2026-10-09)

- Per the user: the kernel must land on upstream, and its interfaces and fusion boundaries must FIT
  upstream. The 4-mode-only port cannot demonstrate the thesis; the combined arm is the only
  meaningful one. The earlier fork-first plan is superseded.
- My params-routing edit is REVERTED: it aliased our flat vector-sparse `kv_indices` into
  `wgmma_plan.dense_page_table`, which upstream expects as a dense `[batch, max_pages]` table
  (-1-filled, since 0 is a valid page id) -- producing an illegal memory access. The seam is a
  representation CONVERSION, not a pointer hand-off.
- Revised plan recorded: express the split kernel's interface in upstream's two contracts with one
  explicit conversion, port onto `upstream_port` (which already carries the 4-mode step and
  `mixed_dispatch.py`), re-pin the stale base, then measure the combined arm.

## v0.26.2 — Remaining edit reduced to mechanical, field names confirmed (2026-10-09)

- `UnifiedWgmmaPlanPtrs` already declares all nine fields the non-unified FA3_UPSTREAM path needs
  (dense_page_table, q/kv_indptr_per_req, kv_lens_per_req, total_q, num_pages, num_requests,
  max_pages_per_seq, max_seqlen_q), and the extension already builds those arrays for
  `fa3_only_run`.
- Shape of the change: select the wrapper TYPE unconditionally so `wgmma_plan` exists, keep
  `kUnifiedWgmma` gated as-is so the broken unified scheduler stays off, and populate the plan from
  the existing locals. No field left unset, no feature disabled.

## v0.26.1 — Remaining work fully specified (2026-10-09)

- `UpstreamFA3Plan` (`hopper/fa3_upstream.cuh:140`) carries the five fields the non-unified
  FA3_UPSTREAM build needs, with in-header comments mapping them to our plan's arrays
  (page_table/kv_indices, cu_seqlens_q/qo_indptr, seqused_k/kv_lens, total_q, num_pages). So the
  fix is to source them from the extension's FA3 plan under `if constexpr (!kUnifiedWgmma)` --
  a real fix, no feature disabled.
- Two documented traps flagged: cu_seqlens_q/seqused_k must be PER-REQUEST (the persistent plan's
  are per work item -- the known four-bug family), and total_q must not be conflated with num_pages.

## v0.26.0 — Blocker narrowed from "finish the wgmma port" to a guard placement (2026-10-09)

- The extension selects `UnifiedWgmmaParams` only under `FLASHINFER_FUSED_UNIFIED_WGMMA`, but the
  kernel's `wgmma_plan` reads (`persistent_fa3_fused.cuh:815-821`) sit outside the
  `if constexpr (kUnifiedWgmma)` guard at `:692`, so `FA3_UPSTREAM=1` instantiates them against
  params lacking the member. A comment at `:932` shows this trap was hit before in this file.
- Correct fix is to move those reads inside the `if constexpr` guard, making `FA3_UPSTREAM=1`
  buildable WITHOUT enabling the unified wgmma path that is recorded as numerically wrong. Nothing
  needs populating for the non-unified build. This unblocks both the per-executor FA3 baseline and
  the fork serving A/B.

## v0.25.9 — Blocker located to the type level (2026-10-09)

- The FA3_UPSTREAM role reads `params.wgmma_plan`, supplied by an EXISTING wrapper
  `UnifiedWgmmaParams<Base, IdType>` (`unified_wgmma_scheduler.cuh:274`). The extension passes the
  plain `BatchPrefillPagedParams`, hence the compile error. The type exists and is simply unused.
- Fix shape recorded: declare the FA3 role's params as the wrapper under the `#if`, and populate
  `wgmma_plan`'s five pointers with PER-REQUEST q_indptr/kv_lens (not the persistent plan's
  per-work-item arrays -- the known port-bug family).
- This one defect gates both the per-executor FA3 baseline and the fork serving A/B.

## v0.25.8 — Serving blocker reduced to ONE root cause: the wgmma params plumbing (2026-10-09)

- The two-executor branch RAISES if `fa3_only_run` is absent, and the driver unsets the fallback
  escape, so a completed split arm with no such error proves the branch was never entered. Both
  arms ran the ordinary attention path; the earlier "unverifiable" correction was over-cautious.
- `fa3_only_run` is gated behind `FLASHINFER_FA3_UPSTREAM`, which fails to compile on
  `BatchPrefillPagedParams has no member "wgmma_plan"`. That single defect blocks BOTH the
  per-executor kernel baseline AND any fork serving A/B of the split kernel -- one blocker, not two.
- Everything else is verified in place: kernel PASSES, optimum split 112 (133.5 us, 1.30x), fork
  server boots correctly, dataset and drivers staged.

## v0.25.7 — Serving A/B verdicts corrected: unverifiable, not invalid (2026-10-09)

- The driver's `two_exec_calls` is `grep -co 'two_exec'` over the server log, and that string is
  emitted nowhere in `flashinfer_backend.py`, so a zero proves nothing about execution. The real
  composition signal (`mixed_fused_fa3=`) is absent from both arms' logs too.
- So all four measured arms are UNUSABLE for the fused question -- no composition evidence either
  way -- rather than demonstrating that fusion did not run. Both earlier verdicts are corrected in
  place; the decode-guard refutation stands but answered an unanswerable question.
- A usable serving arm needs an always-on composition counter first (the `MIXEDSPEC_STEPS` approach
  worked immediately upstream), then a workload with a real decode population.

## v0.25.6 — Decode-guard hypothesis refuted; split kernel still not reached in serving (2026-10-09)

- Re-ran the A/B with `SGLANG_FUSED_DECODE_GUARD=0`: the split arm still reports
  `two_exec_calls: 0`, so the guard added this session is NOT what suppresses the split kernel.
  Prime-suspect claim withdrawn. All four arms measured (2 runs x 2 arms) are the same path,
  consistent to ~1%.
- Next: read where `two_exec_calls` is incremented and what gates `SGLANG_TWO_EXEC`; the leading
  candidate is that the driver never exports `SGLANG_FUSED_FA3_SM_SPLIT`, so there is no SM split
  and hence no two-executor work by construction.

## v0.25.5 — First fork serving A/B ran but is INVALID (composition counter 0) (2026-10-09)

- Both arms completed (0 fallbacks) with near-identical numbers, but `two_exec_calls: 0` in the
  SPLIT arm: the two-executor kernel never ran, so the A/B measured one path twice. Void as a fused
  result per the project's validity rule; kept as a long-context baseline for the shape
  (32768 in / 256 out / 16 prompts / conc 8: 126.5 tok/s, mean TTFT 5582 ms, mean ITL 40.8 ms).
- Prime suspect: the decode-population guard added this session (`kv_tiles <= decode_rows/2`). At
  IN=32768 a prefill row has 256 kv_tiles and would need >=512 decode rows, so the guard correctly
  refuses fusion -- meaning a long-prefill/low-decode benchmark cannot exercise the split kernel.
  Next: confirm with `SGLANG_FUSED_DECODE_GUARD=0`, then measure on a workload with a real decode
  population.

## v0.25.4 — Serving path unblocked; the "version skew" was PYTHONPATH shadowing (2026-10-09)

- RETRACTED the version-skew diagnosis. Both venvs already have `flashinfer-python 0.7.0.post1` with
  the symbols sglang needs; the driver's `PYTHONPATH=/dev/shm/w/python:/dev/shm/w/fi` let the older
  restored tree shadow the pip install. Dropped `fi` from `PYTHONPATH` (it is only needed as
  `FI_ROOT` for the kernel build) and reverted the symptom-treating import guard.
- The fork server with the split kernel now starts (`Application startup complete`, Uvicorn on
  :31000), so the serving A/B can run.

## v0.25.3 — Serving A/B blocked on a flashinfer/sglang version skew (2026-10-09)

- With the venv repaired, `srv_two_exec_ab.sh` runs but the server dies at import:
  `cannot import name 'trtllm_mxint4_block_scale_moe' from 'flashinfer.fused_moe'`. The fork's
  sglang tree expects a newer flashinfer than `/opt/tiger/flashinfer @ b0563ae`, which is what the
  `fi` restore pushed.
- Preferred fix is to guard that one eager import (a w4a4-mxint4 MoE scheme a bf16 Qwen2.5-7B run
  cannot need) rather than upgrade flashinfer, since the fused-kernel work is local-only.
- Lesson: a tree restored from a local copy can build correctly and still mismatch its consumer;
  verify the two trees' versions against each other, not just that both import.

## v0.25.2 — Fork serving environment fully restored (2026-10-09)

- Second dependency fault fixed: `torchvision 0.24.1+cu128` against torch 2.14.1 caused a circular
  import; installed the matching `0.29.1`. With the earlier nccl swap, the fork venv now imports
  both the fork's sglang (`/dev/shm/w/python/sglang`) and our fused flashinfer (`/dev/shm/w/fi`).
- Pod restore is a seven-step recipe, all documented. The fork serving A/B -- the only arm that
  exercises the split kernel on the realistic tracelab workload -- is now reachable.

## v0.25.1 — Fork venv repaired: nccl library mismatch, serving A/B path unblocked (2026-10-09)

- The fork venv's `import torch` failure (`undefined symbol: ncclCommResume`) was an nccl build
  conflict: both `nvidia-nccl-cu12 2.27.5` and `cu13 2.30.7` install to the same path and the cu12
  build (lacking that symbol) had won. `LD_LIBRARY_PATH` cannot fix it -- torch resolves nccl via
  RPATH -- so the bundled `libnccl.so.2` was replaced with the cu13 build (original kept as .bak).
- `torch OK 2.14.1+cu130, cuda_avail=True`. This unblocks the fork serving A/B, the only arm that
  exercises the split kernel and has the tracelab dataset.

## v0.25.0 — First validated kernel speedup: fused 1.30x over the unfused persistent kernel (2026-10-09)

- `TIME_BASELINE=1` provides a like-for-like baseline (the harness plans a separate 3-task object
  over the SAME rows without `FLASHINFER_FA3_TASK` -- the unfused persistent kernel on identical
  work). Fused at sm_split 112: **133.5 us** vs baseline **174.0 us** = **1.30x**, 23% less kernel
  time, with the kernel PASSING the causal fp32 reference at every split.
- Baseline stable at 173-177 us across three runs. Fixture is FA3-heavy, so this does not
  generalise to the mixed serving workload, and it does NOT satisfy the >=30% TTFT/ITL or >=10%
  throughput bar -- only the serving A/B can.
- The FA3-role per-executor baseline is still blocked on the wgmma `params.wgmma_plan` plumbing.

## v0.24.3 — flash-attention tree restored; FA3 baseline blocked on the wgmma params path (2026-10-09)

- Pushed `flash-attention/hopper` (249 KB) as a sibling of the flashinfer root, clearing the
  `fa3_upstream.cuh -> ../../../../../flash-attention/hopper/utils.h` build failure. The pod restore
  recipe is five steps, not four.
- `FA3_UPSTREAM=1` now fails later, on `BatchPrefillPagedParams has no member "wgmma_plan"` -- the
  upstream-FA3/wgmma route is incomplete in this build, and `fa3_only_run` is gated behind it. So the
  FA3 per-executor baseline is unbuildable here and **no speedup is demonstrated**; the default
  build still PASSES and times at 133.5 us (sm_split 112).

## v0.24.2 — Split optimum found (sm_split 112, 133.5 us); baseline arm still missing (2026-10-09)

- Extending the sweep past the harness's built-in 48/32/16 found the optimum at sm_split 112:
  133.5 us vs 383.3 us at 48 (2.87x), with a ~258 us plateau at 64/80/96 and a cliff at 112.
  Every split PASSES the reference.
- The `TWO_EXEC=1` per-executor baseline arm does not run: the extension does not export
  `fa3_only_run`. So the fused times have nothing to divide by and **no speedup is claimed**.

## v0.24.1 — First valid fused kernel timings (2026-10-09)

- With the kernel passing, `TIME_ITERS=50` on fixture 1 gives 383.3 us at sm_split 48, 505.1 us at
  32, 877.5 us at 16 (0 unwritten, every row within bf16 noise at every split; the harness prints
  `PASS: every row correct at every split`).
- Monotonic in FA3's SM share, so the optimum is likely above the harness's 48-SM ceiling. This
  fixture is FA3-heavy, so the FA3-favouring split is expected and does not generalise.
- Per-executor baselines are still needed before any of this can be called a speedup.

## v0.24.0 — FUSED FA3+FA2 KERNEL PASSES THE CAUSAL fp32 REFERENCE (2026-10-09)

- Fixture 1 is green at sm_split 48/32/16: **0 unwritten rows**, FA3 0.00628, task0 **0.00250**,
  tile16 0.00071 -- all three classes within bf16 noise of the causal fp32 reference.
- Two fixes: (1) the persistent runners' `KernelTraits` used `MaskMode::kNone`, so the FA2 side
  applied no causal mask; (2) `cluster_tile_q` used `gridDim.x` unguarded under `FUSED_GRID`, and
  since the kernel's CAUSAL `num_kv_chunks` depends on it, the inflated value forced every tile onto
  the split-KV path where causally-confined rows never completed their merge.
- One invalid refutation corrected in place: fix (2) had been tested only on a single-chunk fixture
  that could not exercise it, and "no change" was wrongly read as refutation.
- Fused timing is meaningful for the first time; the serving A/B is the next step.

## v0.23.4 — Split-KV defect attributed to the reduction, with the fix located (2026-10-09)

- `NO_JIT_REDUCE=1` A/B: unwritten rows jump 55 -> 87 (row 1's full length) when the reduction is
  disabled, so the reduction is what writes these rows and the plan is exonerated. It completes 32
  rows and skips 55 -- precisely the rows whose only partial is chunk 0's.
- Fix located: the reduction must finalise a row whose partials come from a single chunk instead of
  requiring every chunk to contribute. The separate 3-chunk weighting error (rel err 0.514) needs
  its own fix in the same reduction.

## v0.23.3 — Split-KV merge mechanism pinned on the FA2 side (2026-10-09)

- kv-length sweep at fixed qo: single-chunk rows PASS at two different qo (0.00152, 0.00269);
  every multi-chunk row FAILS. The unwritten count tracks the CAUSAL BOUND, not qo -- with a prefix
  large enough that every row needs a chunk beyond the first, nothing is unwritten.
- Mechanism: the merge produces `final_o` only for rows with partials beyond chunk 0; a row whose
  causal bound lies entirely in chunk 0 is never merged and stays NaN. At 3 chunks every row is
  written but wrong (0.514), so the same merge also has a weighting error -- two defects, one path.

## v0.23.2 — Second fused fault identified as the FA2 split-KV merge (2026-10-09)

- A store probe showed NO work item of fixture 1's failing row reaches the write-through store, and
  its trace reads `nchunks=2`: that row takes the `partial_o` + merge path, while the `qo=64`
  fixture (`nchunks=1`) takes write-through and PASSES after the MaskMode fix.
- So the 55 unwritten rows are a split-KV merge defect on the persistent side, not a q-tile
  coverage problem. The earlier "5th tile" arithmetic is retracted: the sweep's 0-to-55 transition
  tracks `kv_len` crossing `len_kv_chunk = 64`, because the fixture moves qo and kv together.

## v0.23.1 — Fused kernel PASSES the causal fp32 reference on all three classes (2026-10-09)

- Fix: the persistent runners' shared `KernelTraits` used `MaskMode::kNone`, so the FA2 side applied
  no causal mask. Changed to `MaskMode::kCausal`. task0 rel err **0.98-1.04 FAIL -> 0.00221 ok**,
  with FA3 (0.00709) and tile16 (0.00134) unchanged and 0 unwritten rows, at sm_split 48 and 32.
- First run where every class passes the reference. Found from the per-token profile already
  available behind `DUMP_ROWS=1`: token 63 of 64 was the sole correct token, i.e. every token was
  attending to the full KV range.

## v0.23.0 — Root cause of the fused task0 failure: missing causal mask in the tile128 path (2026-10-09)

- The failing task0 row has exactly ONE correct token -- the last one, which is the only q token
  that legitimately attends to all of its KV. Every other token is computed against the full KV
  range, so the per-token causal mask is not applied in the tile128 persistent path.
- This single cause explains the whole evidence set: the 3.40x magnitude deficit (dilution, not
  scaling -- got is not proportional to ref), all 28 heads damaged, the tile16 row passing (its
  single q token needs no mask), FA3 passing (separate causal path), and the tile-level kv_end
  bounds being provably correct while the in-tile mask is not.
- Ten earlier candidates were eliminated and recorded in place, including one attempted fix
  (`cluster_tile_q`, refuted and reverted) and one retracted "duplicate work item" anomaly that was
  an artifact of aggregating traces across the harness's three sm_split launches.

## v0.22.1 — task0 fused fault characterised; cluster_tile_q candidate refuted (2026-10-09)

- The fused kernel's failing task0 row writes EVERY element (229376/229376 nonzero) but at
  **3.40x too small a magnitude** (absmax 0.428 vs reference 1.453), while the tile16 row is exact
  and FA3 passes in the same launch. So the tile128 runner's epilogue is fine and its
  softmax/accumulation balance is not -- the fault is an under-weighted result, not a missing write.
- The `cluster_tile_q = gridDim.x * CTA_TILE_Q` candidate (unguarded under FUSED_GRID, unlike
  `persistent_cluster_slot`) was tested on the discriminating fixture and **refuted** (rel err
  1.03446 -> 1.04120), then reverted rather than left in the tree.
- Harness note: the store-vs-compute discriminator already existed behind `DUMP_ROWS=1`; and an
  instrumented (device-printf) build corrupts FA3 numerics (0.0059 -> 1.19), so coverage runs and
  numerics runs must stay separate.

## v0.22.0 — Fused FA3+FA2 kernel builds and runs again; FA3 half validated (2026-10-08)

- Restored the flashinfer tree on the vibecuda pod (it was absent, so the split kernel had no
  runnable binary at all) and got the fused TU compiling: five sequential blockers, the last being
  that torch 2.14 needs **C++20** in both the harness `load()` and `flashinfer/jit/core.py`.
- First kernel-arm result on fixture 1: the **FA3 executor PASSES** the causal fp32 reference at
  rel err 0.00662 with split-KV active (4 splits), and a persistent decode row passes at 0.00070.
- Open kernel bug: the persistent **task0** row (qo=87) leaves **55/512 output rows unwritten**
  (NaN), identically at sm_split 48/32/16 -- so it is a q-tile/work-item coverage gap, not an SM
  partitioning or race effect. No fused timing is valid until this is fixed.

## v0.21.0 — Dispatch boundary is a ratio, and the decode-population guard (2026-10-08)

- Re-analysed the measured 2-D boundary (`boundary_28x4x128.json`): the fused win is set by a RATIO,
  `kv <= 64 * decode_rows` (one 64-wide KV tile of decode work per decode row), not by thresholds on
  `qo_len`/`kv`. With no co-resident decode rows the fused kernel loses on every measured kv
  (0.617-0.704x), which no per-row rule can express.
- `mixed_dispatch.split_classes` now applies a batch-level decode-population guard
  (`SGLANG_FUSED_DECODE_GUARD`, `SGLANG_FUSED_DECODE_TILES_PER_ROW`). Replaying all 25 measured
  points through the real dispatch code: agreement with the measured winner rises from **11/25 to
  23/25**, and the `nd = 0` losing region is no longer routed to FA3.
- Pod restore: found TWO casualties of the earlier restore -- the flashinfer tree (`/dev/shm/w/fi`)
  was absent, so the split kernel had no runnable binary, and the fork venv's torch is broken
  (nccl ABI). Recorded a restore split that clones third-party on the pod (it has network) and
  pushes only the fork's ~2 MB of sources.

## v0.20.1 — Correctness evidence for the ported 4-mode step, and two invalid gates retired (2026-10-08)

- The merged forward's PREFILL rows are bit-exact against a prefill-only reference (err 0.0, 0
  argmax flips, all 5 mixed steps), which validates the merged batch, the logits split and the
  comparison machinery at once.
- Greedy-output equality was retired as a gate for this feature: the merge point is
  timing-dependent, so the verify rows sit at different decode positions each run (identical
  `pre_logits_sum` with `seq_lens` differing by `num_draft_tokens`). Under a FIXED composition the
  4-mode arm is byte-identical to the control.
- Gate (b), comparing the merged verify slice against an unmerged verify forward, was found to
  measure its own side effects: `eagle_prepare_for_verify` plus a verify forward is destructive and
  non-idempotent, so two identical references disagree as much as merged-vs-reference. Needs a
  non-destructive reference (run it before the merged forward).
- Port commits 39-46: bounded the verify hidden-state slice (padding rows were reaching the draft
  extend), parked the verify half's keep-alive refs, and restored the prefill rows' own
  `sampling_info`, which had been dropped in an earlier restructure.

## v0.20.0 — The 4-mode step runs end to end on current upstream (2026-10-08)

- Branch `upstream_port` (worktree `/opt/tiger/sglang-up`) against upstream pinned `b7b2975b5`,
  torch 2.14.1: `portchk.sh` reaches **11 mixed steps with 0 tracebacks**, where a
  `SGLANG_MIXED_SPEC=0` control on the same script also passes. Correctness (greedy-output
  equality) is NOT yet gated -- the smoke script's prompts are drawn inside worker threads, so its
  output lengths are not comparable between arms.
- The verify half now delegates to upstream's own `run_eagle_verify` via a new
  `precomputed_logits_output` / `precomputed_forward_output` pair, instead of hand-rolling sample +
  commit + result (which cost one failed run per field upstream sets).
- The scheduler commits BOTH halves of a 4-mode step: the gate keeps `running_batch` (emptying it
  let the prefill batch take the decode rows' place and leaked a request), the verify half gets its
  own future wiring, relay stash and D2H copy, and result processing receives a snapshot batch
  because overlap merges the next prefill into the live one first.
- Two latent bugs fixed on the way, both of which would have corrupted output silently rather than
  crashing: the dispatch branch dropped the scheduler's `on_publish`, so a 4-mode step published no
  futures at all; and the verify-row logit slice indexed from the padded end of `hidden_states`
  under a prefill-graph bucket, selecting padding rows instead of verify rows.

## v0.19.1 — The 4-mode split verified numerically; MTP results confirmed (2026-10-08)

- `SGLANG_SPLIT_VERIFY=1` fp32 recompute on a real mixed step: FA3 prefill rows 0.0032-0.0040 and
  FA2 **verify** rows 0.0030-0.0034 relative error -- bf16 noise. The 5-token greedy divergence from
  stock MTP is argmax flips from that noise, not a fault, so the MTP serving numbers stand.
- Min-common-prefix is a weak cross-kernel discriminator (it only catches gross faults, which show
  rel err ~4.0); the fp32 per-row check is the real gate. New driver `verify_gate.sh`.
- The H20 gate's earlier one-mixed-step pass was the GATE's fault: 96-token background rows
  finished before the long prefills arrived. Fixed -> 10 mixed steps, one with 2 prefill + 11
  verify rows.

## v0.19.0 — Randomized source selection refuted under contention (2026-10-08)

- Point 4's proposed algorithm ("pull at random among nodes whose hit lengths are within a
  threshold") LOSES on every metric against ordering by in-flight load, even on a deliberately
  narrowed 1-NIC fabric built to favour it: chat mean ITL +24%, chat P99 ITL +63%, agent P99 ITL
  +83%, chat TTFT +15%. Load, not link bandwidth, is the binding constraint at 16 GPUs.
- Narrowing the fabric DOES create real contention (tails 2.3-3.2x worse at unchanged means), so
  the test was valid; randomization simply discards the load signal.
- New router policy `cache_slo_rand` exists for this test and should NOT be a default.
- Two earlier experiment designs were discarded before use and are recorded: a partially-pinned
  fabric (sequential push raced the arm launch) and an asymmetric pair (worker EDF on one side).

## v0.18.1 — Long context at a 20 ms chunk budget clears 30% on every metric (2026-10-08)

- 3 rounds, 64-96K TraceLab + chat 12/s, vs stock MTP at its own best chunk: throughput +45.0% /
  +43.3%, TTFT -85.3% / -39.6%, mean ITL -53.2% / -52.4%, **P99 ITL -32.9% / -48.0%** (chat /
  codex). The earlier -29.7% chat P99 miss at a 47 ms budget was budget tuning, not a ceiling.
- Chat P99 ITL saturates near 112 ms for 20-30 ms budgets; codex keeps improving to 78 ms.

## v0.18.0 — The SLO check prices decode, not just prefill (2026-10-08)

- Hot-prefix workload (4 groups sharing 16K prompts, H20 16 GPUs): a TTFT-only SLO check never fired on
  an overloaded cache holder, because its prefill stays fast. The SLO check now prices the decode
  slowdown a request would cause: chat mean ITL **-65%** (23.4 -> 8.3 ms), P99 **-80%** (224 -> 44 ms)
  for a 25% chat TTFT cost. Colocation keeps a ~70x TTFT advantage over disagg 2P2D.
- `calibrate_h20.py` measures decode ITL over context x concurrency (was 256-token context only,
  fitting an unusably small 0.065 ms/seq). The router interpolates that grid (`--decode-grid`); two
  closed forms were rejected and the rejections are recorded in the code.
- RESOLVED (2026-10-08): the 15x ITL jump by 524K resident tokens is prefill interference in the
  calibration (all requests submitted at once, so 16K prefills share steps with decode), not a
  capacity or bandwidth effect: a server held at 32x16K logs pure-decode steps at ~18.6 ms, 7x
  faster, with no retraction. The grid stays valid for routing as a loaded-worker cost, but it is
  not a decode-only model.

## v0.17.6 — Prefix-aware chunk sizing (2026-10-08)

- `SGLANG_CHUNK_ATTN_BUDGET` caps a chunk's attention work n*(r + n/2) while rows are running. At
  64-96K it fixes codex's tail (P99 ITL 91 ms, the best of any setting) at little TTFT cost, and it
  is neutral at short context. It does not fix chat's tail: fresh-prompt chunks are GEMM-bound.
- `SGLANG_CHUNK_MS_BUDGET` + `SGLANG_CHUNK_COST` cap the chunk's predicted time with the fitted step
  coefficients. At 64-96K, 3 rounds, vs stock MTP's best: throughput +42/+37%, TTFT -81/-35%,
  mean ITL -49/-45%, P99 ITL -29.7/-30.2% (chat/codex). It regresses at short context (chat SLO
  0.641 vs ~0.735), so it is a long-context setting only.

## v0.17.5 — Long context, both arms tuned: full stack wins on every metric (2026-10-08)

- Chunk 2 048, 3 rounds, 64-96K TraceLab + chat 12/s. Against stock MTP at its own best chunk
  (8 192): throughput +45% / +45%, TTFT -93% / -55%, mean ITL -48% / -45%, P99 ITL -15% / -31%
  (chat / codex). The chunk-8 192 P99 regression is fixed by the chunk size; stock MTP does not
  benefit from the smaller chunk.
- Chat 8/s at the default chunk, 3 rounds: throughput +18% / +30%, TTFT -96% / -75%, mean ITL
  -58% / -49%; chat P99 ITL +27% (the chunk effect again).
- H20: the colocated-MTP "regression" is retracted. The router's `min()` tie-break sent every
  hit-0 request to workers 0/1 and affinity pinned sessions there. The tie-break is now
  randomized. With even load, colocation + MTP + 4-mode keeps TTFT and cuts agent ITL 47%.

## v0.17.4 — Long-context (64-96K) A/B: throughput +46/+53%, P99 ITL regresses (2026-10-08)

- Full stack vs stock MTP, TraceLab sessions at 64-96K + chat 12/s, 3 rounds: output throughput
  +45.6% (chat) / +52.6% (codex), TTFT -96.7% / -71.2%, mean ITL -46.9% / -47.2%. P99 ITL is
  worse: +54.6% chat, +5.8% codex. Chunk sweep and below-saturation rounds are queued.
- `long_agent_chat` scenario; `mtp_edf.sh` honours `TRACE_PATH` and `SCENARIO`.

## v0.17.3 — Attention share of a decode step vs context length (2026-10-08)

- H200, Qwen2.5-7B, yarn to 131K: attention exceeds half of a decode step at ~40K tokens with 8
  requests (72.6% at 120K) and ~10K with 32 (73.7% at 32K). A single request never does (31.9% at
  120K). The codex tenant (8-12 sessions at 32-49K) is ~47-55%. Drivers `context_share.sh`
  (bench_one_batch, batch 1) and `context_share_srv.sh` (server, batches 8/32: bench_one_batch's
  one-shot prefill OOMs there).

## v0.17.2 — Full stack vs stock MTP: every TTFT/ITL metric >=30% better, 3 rounds (2026-10-08)

- 4-mode split + deadline order vs stock sglang MTP at chat 24/s + codex: chat TTFT -95%,
  chat ITL -42% (P99 -35%), codex TTFT -51%, codex ITL -38% (P99 -34%). Chat SLO attainment and
  goodput 3.4x. Output throughput is equal, since both arms serve the offered rate; the capacity
  gain is the v0.16.0 closed-loop +19.8%. Commands and SHAs are in pd_colocation_vs_disagg.md
  point 3.

## v0.17.1 — EDF on the MTP server: order by deadline, do not cap chunks (2026-10-07)

- Ablations at ITL SLO 30 ms (H200, chat 24/s; `mtp_edf.sh` with `ARMS=`). Every per-step
  chunk-cap variant loses to no EDF: global EWMA, FCFS + cap, and the P90 slow-step allowance
  (`SGLANG_EDF_ITL_QUANTILE`). Splitting prefill adds fixed-cost mixed steps that every decode row
  waits through. Per-request EWMA collapses because the binding request's estimate is ~1.15 vs a
  global ~1.6-1.9 (`EDF_BIND`).
- **Deadline ordering alone (EDFP), 3 rounds:** chat TTFT -62%, codex P99 ITL -27%, chat P99 ITL
  -10%, chat SLO attainment and goodput +8.6% (ranges disjoint); codex mean TTFT +43%, inside its
  length-scaled SLO.

## v0.17.0 — EDF on the MTP server: acceptance length in the ITL budget (2026-10-07)

- `edf_scheduler.py`: the per-step ITL budget is `ITL_SLO * tokens/step`, with three estimators
  of tokens/step (`SGLANG_EDF_ACCEPT_MODE`): `none` (a=1), `global`, or per-request `req` EWMA
  (`SGLANG_EDF_ACCEPT_ALPHA`). The estimators are scored online against delivered tokens
  (`EDF_ACCEPT` log; `SGLANG_EDF_ACCEPT_TRACE=1` scores them without EDF). Also added: an
  `EDF_BIND` log and a `linear:<json>` step model (`LinearStepPredictor`).
- STEPTRACE counts a 4-mode step's verify rows as decode rows. New:
  `tasks/perf_model/fit_mtp_step.py` (held-out step MAPE 21.9% vs 98.6% for a constant), and
  drivers `mtp_edf.sh` and `slo_attain.py` (with `--scaled`, the length-scaled TTFT SLO EDF
  schedules against).
- Measured (H200, chat 24/s, ITL SLO 50 ms): mean tokens/step is 2.10. EDF with a=1 collapses
  (chat SLO attainment 0.117, TTFT 41 s); the global EWMA restores it to 0.866. That still does
  not beat no-EDF (0.876), because this SLO barely binds. The per-request EWMA has a lower
  estimator error but worse attainment (0.696).

## v0.16.1 — multi-node runs: 1024-fd limit was dropping requests (2026-10-07)

- **Retraction:** every earlier H20 multi-node arm ran with the pods' soft `ulimit -n` of 1024.
  Under the 3200-request chat burst the router ran out of sockets ("error sending request").
  For the stock sglang_router this then opened its circuit breaker, which rejected ~5.7k
  requests ("No available prefill"). disagg_2p2d served only 163/384 agent and 2265/3200 chat
  requests. Those numbers, and the coloc arm numbers from the same harness, are void pending
  re-runs.
- `mn_run.py`, `mn_bench.sh` and `mn_node.sh` raise `ulimit -n` to 1048576. Stock-router arms
  run with the circuit breaker and health ejection off, and the PD arms use
  `--prefill-policy cache_aware --decode-policy power_of_two`.

## v0.16.0 — 4-mode MTP clears the bar: latency under load, ~20% capacity (2026-10-07)

- **MTP load sweep** (chat 4-28/s, both tenants rate-limited, barrier, 0 fallbacks): the
  4-mode + split gain grows with load. At 20/s all four headline latency metrics beat stock MTP
  by >=30% on both tenants (codex TTFT -48%, ITL -33%; chat TTFT -87%, ITL -39%). Mixing without
  the split regresses at every load.
- **MTP capacity, closed loop** (3 rounds): total output throughput **+19.8%** (chat +20.5%,
  codex +16.9%, non-overlapping ranges); codex ITL -30%. Codex TTFT is +21% under saturation
  (queue position), reported as measured.
- H20 multi-node study: cache-affinity routing beats round-robin by 8.4x on agent TTFT and
  ~90x on chat TTFT (72.9% worker-reported cache hit). sglang's own `cache_aware` router is a
  strong baseline: better chat TTFT tails, worse ITL tails than ours. The SLO check needs
  calibrated H20 constants (0 SLO-check moves with placeholders).

## v0.15.1 — Stream start barrier; serving results re-measured (2026-10-07)

- **Methodology fix:** multi-stream serving runs had an uncontrolled stream start offset (12-45 s,
  set by how long each `bench_serving` took to tokenise). It swung chat TTFT 15x between
  sessions. Fixed with a start barrier (`d2fc14f8c`). All serving A/Bs re-run with it.
- **SPLIT vs LPT: confirmed** (3 rounds, tight ranges): chat TTFT -17.4%, codex TTFT -10.5%,
  ITL -7.9%, P99 ITL -15.9%, throughput +7.0%. Essentially the original result.
- **4-mode MTP: CORRECTED.** v0.15.0's figures were offset-confounded. Barrier re-run, 4-mode +
  split vs stock MTP: codex throughput **+17.6%** (not +27.8%), codex P99 ITL **-52.8%**, ITL
  -16.2%, TTFT -8.5%; chat TTFT **-37.3%**, P99 ITL -30.5%, ITL -27.7%.
- Kernel catalogue closed: chunk sweep, host-overhead fixes (~1.6 ms per mixed step, stopped at a
  ~2% bound), the plan cache keyed on exact row lengths.
- H20 multi-node study started: 8 pods x 2 H20 (16 GPUs, trial vibecuda); ablatable router;
  per-pod port probing with bind-race supervisor; session-replay workload.

## v0.15.0 — 4-mode MTP serving result; kernel catalogue closed on measurement (2026-10-07)

- **4-mode MTP serving A/B** (3 interleaved rounds, gate PASS, 0 fallbacks; `drivers/mtp_repeats.sh`):
  4-mode step + split vs stock sglang MTP -- codex throughput +27.8%, ITL -17.4%, P99 TTFT -14.6%;
  chat ITL -35.0%, chat TTFT -50% or better (stock MTP overloaded in one round). Mixing alone
  gives most of the chat gain; the split adds codex throughput.
- 4-mode MTP gate PASSES (stock, mixed-FA2, mixed-split); fork bugs fixed (`10b378d89`, `5ab463414`).
- Fused tree build wired into the EAGLE worker (`f1985ca8f`, `SGLANG_FUSED_TREE_BUILD=1`), gate PASS.
- Kernel catalogue: SM/cluster-cap sweep (serving's cap already optimal; FA3 budget must stay full),
  CUDA-graph ceiling via step traces (split host gap +0.83 ms/mixed step -> FA3 plan skipped in
  two-executor mode, `e237ab8bc`), packed copies superseded, stale boxes closed.
- Perf model: split mixed steps modestly easier to predict (MLP 6.0% vs 6.4% MAPE; linear 13.4% vs
  16.9%).

## v0.14.0 — SPLIT serving made correct; first valid serving result; 4-mode MTP step (2026-10-07)

- **Three correctness faults in the serving split, fixed:** (1) FA3 was handed the FA2 rows'
  per-request arrays (`748f44553`); (2) both executors ran non-causal, and so did the harness
  reference (flashinfer `b0563ae`, sglang `2b92b58bb`); (3) in ragged mode the split read the
  step's not-yet-written K/V. Split steps now use non-ragged mode (`4e3bde083`). The earlier
  -40% TTFT result is **void**.
- **Serving correctness gate** (`drivers/srv_correctness.py`): greedy outputs against LPT, with
  LPT-vs-LPT as the noise floor. It catches the pre-fix build and PASSES the fixed one. Also
  added: `SGLANG_SPLIT_VERIFY=1`, an in-server fp32 recompute per executor.
- **Causal kernel:** split 1.309x over FA2-on-everything on the TraceLab batch, 1.324x with
  two-stream overlap (now the serving default). Multicast ceiling 1.8% / 1.6% causal.
- **First valid serving A/B** (3 interleaved rounds, 0 fallbacks): codex TTFT -10.3%, ITL
  -7.6%, P99 ITL -18.5%, throughput +5.6%; chat TTFT -16.2%. Below the 30% / 10% bar.
- **4-mode MTP step** (`868025d6c`, `SGLANG_MIXED_SPEC=1`): prefill chunk rows and EAGLE
  verify rows in ONE target forward. Untested; gate queued. Draft:
  thoughtworks/Qwen2.5-7B-Instruct-Eagle3, topk 1.

## v0.13.0 — Kernel items closed: FA2 K/V ring, split-reduction ceiling, fused tree build (2026-10-07)

- **FA2 K/V smem ring** (flashinfer `3e41c2d`, defaults `f54745e`): an S-slot ring per tile
  class (`FLASHINFER_FA2_KV_STAGES_{1,2,3}`), now 2/3/1 by default. MTP-verify rows -7.4%,
  64-128K verify -10.3%, appends -2.0%, decode flat (already ~80% of HBM). 3 reps, all PASS,
  0 fallbacks. `drivers/fa2_ab.sh` + `tests/fa2_stage_bench.py`.
- **Smem-handoff split reduction: ceiling measured, not built.** Skipping every rendezvous and
  merge saves <=2.9%; splitting itself is worth 2-4x. Classic grid.sync beats our JIT
  rendezvous by 2.2% on long verify.
- **Fused draft tree build** (`speculative/csrc/fused_tree_build.cu`): organize_draft_results +
  build_tree in one kernel, 0 mismatches vs sglang over ~150 cases; 31 -> 10 us graph-captured
  (MTP chain). The doc's "attention epilogue" design is retracted: the data dependencies forbid
  it.
- Tooling: `FLASHINFER_EXTRA_CUDAFLAGS` (per-arm -D for flashinfer's JIT), harness `EXTRA_DEFS`,
  `FLASHINFER_SPLIT_MERGE_SKIP` and `FLASHINFER_FA3_SKIP_KV_LOAD` timing-bound switches
  (default off; the latter is a patch to the pinned flash-attention, in
  `drivers/flash_attention_skip_kv_load.patch`).
- Serving repeats (realistic_repeats.sh) stopped mid-round-1 to close kernel items first.

## v0.12.0 — First realistic serving comparison: SPLIT vs LPT, preliminary (2026-10-07)

TraceLab coding-agent replay plus 12/s short chat on 1x H200, both arms 0 fallbacks, single
run each:

- codex: mean TTFT -39.8%, P99 TTFT -32.9%, mean ITL -16.8%, P99 ITL -52.6%, throughput +16.9%
- chat: mean TTFT -41.5%, P99 TTFT -29.7%, ITL flat, throughput flat (open-loop fixed rate)

Contradicts my earlier "low single-digit gain" prediction, which is retracted. Likely
mechanism is queueing amplification near saturation (unverified). Preliminary: no repeats,
an intermittent illegal memory access in the SPLIT arm (1 of 2 async runs), no MTP.

Getting here took three serving-path fixes that no kernel-level test could see: the
two-exec build dropped -DFLASHINFER_FUSED_UNIFIED_WGMMA=1 (one env var gated a runtime and a
compile decision), sglang's fa3_only_run call lacked the num_splits argument, and the pod's
dependency chain needed eight fixes. Also: priority list rewritten as a dated checklist, and
ab_parallel.sh builds A/B arms concurrently.

## v0.11.0 — The "split-row corruption" was a test-harness bug; per-executor items measured; the 30% ceiling (2026-10-07)

**The kernel was correct.** The long-running "split-row corruption" was two bugs in the test
harness, both introduced with its `PAGE_SIZE` support: the reference `exact()` read KV
token-contiguously (`kvb += kl`) while the paged layout starts each request on a page
boundary, and `NPAGES` was `ceil(sum/PS)` instead of `sum(ceil/PS)`. A page-ramp probe
(`KV_ROWCONST=2`, V[page p] = p x 1e-4) exposed it: the kernel's decoded mean page advances
~33 per row (correct for 4097 tokens), the reference ~32. With both fixed (flashinfer
`333902e`), every fixture on record passes -- decode N=18 at kv=4096 and 65536, the two-row
reproducer at page 128 and page 1, `qo:127`, and both TraceLab arms.

This retracts a long chain of kernel-side diagnoses (row order, partial pages, chunk limits,
partial-tile overrun, partial_lse indexing, races, the partial_o cursor, bf16 partials, a
"half page offset"), several of which were reported as confirmed. The last of those rested on
a number back-computed from a rel err rather than decoded from output. The serving A/B is
unblocked.

**Per-executor optimisation items, measured on the fixed kernel** (each executor against its
own baseline; all PASS, 0 unwritten, 0 fallbacks):

| FA3 vs FA3 | | FA2 vs FA2 | |
|---|---|---|---|
| `MmaPV_is_RS=0` | +3.99% slower | `MMA_KV=3` | -0.43% |
| `stages=3` | +0.54% slower | `MMA_KV=4` | -0.46% |
| `LPT=1` | +0.68% slower | | |
| TMA paged | +10.03% slower | | |

Upstream's defaults win every FA3 comparison; `FA2_MMA_KV` is register tiling, not pipeline
depth (SharedStorage identical at 2/3/4), so FA2 pipelining remains untested.

**Routing is already optimal both ways:** FA3 is 1.38x faster on identical append rows, FA2
1.30x faster on identical decode rows, and the split routes exactly that way. On the mixed
TraceLab batch the split cuts attention 19.3% (15 395.8 -> 12 425.4 us, 1.239x).

**That 19.3% is a ceiling on TTFT/ITL from attention**: even at 100% attention share, e2e
gain cannot exceed it. >=10% throughput is reachable; >=30% TTFT/ITL has to come from
scheduling (EDF admission, cache-hit routing, spec acceptance), not from this kernel.

**Multicast:** cluster launch implemented (`FLASHINFER_FA3_CLUSTER_M`,
`launch_kernel_on_cluster`), but upstream asserts `Use_TMA_KV || ClusterShape == 1`, so it
needs either TMA KV (measured 10% slower) or a cluster-aware cp.async gather of our own.

Also retracted this release: "42.8x" as a performance gain (it is the size of a defect in our
own port), and the claim that KV multicast requires TMA (DSMEM peer reads do not; upstream's
implementation couples them).

## v0.10.2 — The split-row corruption is a RACE in our JIT reduction; nineteen candidates eliminated (2026-10-06)

**The split now beats FA2-on-everything, reproduced three times.** TraceLab at page 128:
12 316.3 us against 15 360.2 us = **1.247x**, after 12 399.7 / 15 190.9 = 1.225x earlier on
the same fixture. Kernel-batch time, **not serving** -- the serving A/B has still never run.

**The remaining blocker is a race, not an indexing bug.** Eight consecutive runs of the
two-row reproducer, nothing changed between them:

    rel err  0.77292  0.65776  0.72528  0.68167  0.68027  0.68413  0.67000  0.68319

Nondeterministic. This **retracts** the previous entry's assertion that it was deterministic
(stated without testing), and reclassifies two earlier readings: the error magnitude varying
across fixtures (0.39-0.84) is run-to-run variance, not a function of overrun size, so the
apparent correlation with tile partiality was noise.

The suspect is the port's own **JIT-reduction rendezvous** (`group_counter`, the
`__threadfence()` / `persistent_block_sync` ordering, the counter zero-reset), whose comments
already record *intermittent* wrong merges -- 3 runs in 10 -- on exactly "the one batch shape
whose FA2-class rows split their KV". It now fails on nearly every run, which is why it read as
deterministic and sent the investigation after indexing for a dozen candidates.

**What the reproducer is:** `qo:1,512 prefix:65536,0` fails; `qo:512,1 prefix:0,65536` passes.
A split row with a **lower request index** corrupts every later `task0` row. Damage is total
(512/512 tokens, 28/28 heads), fully written and in range.

**Eliminated, with the measurement that did it:**

| candidate | eliminated by |
|---|---|
| partial final KV page | fails at `page_size=1` too |
| the chunk limit being 1024 | plan dump: `len_kv_chunk` is 512, per-task |
| follower must be unsplit | plan dump: the follower splits in both cases |
| the combine pass | `n_fa3=0` arm: no FA3, no combine, still fails |
| `partial_indptr` for unsplit items | `PI_ZERO=1` changes nothing |
| `partial_lse` row- vs tile-relative | `LSE_TILE_REL=1` changes nothing |
| wrong LSE weights | victim with `kv=512` never merges, still fails |
| write collisions / `merge_o_indices` overshoot | damage is total, not a bounded prefix |
| the victim's KV base | plan dump: `kv_indptr=513` is correct |
| per-task cluster-slot/cursor | the PASS ordering has identical cluster consumption |
| every plan array | dumped twice on different fixture pairs, correct both times |
| upstream flashinfer | pure `BatchAttention` handles the failing order at 0.00587 |

**Priority item 1 reopened then demoted to not started.** `FA2_MMA_KV` does not change
`PKT<128,N>::SharedStorage` -- 133 120 bytes at N = 2, 3 and 4 -- so it sizes the register-side
`s_frag` accumulator, not pipeline depth. The flat 15 218.8 / 15 296.6 / 15 219.5 us sweep
therefore measured a register-tiling knob and said nothing about K/V pipelining. Also retracts
the earlier explanation that FA2 was "at roofline": at the real 989.4 TFLOP/s peak it is at
**33%**, not 83%.

**The four-point PD colocation doc is written** (`tasks/pd_colocation_vs_disagg.md`), covering
cache-hit-first routing with an EDF/SLO check, best-effort cache migration and the replication
argument, an EDF predictor with EWMA acceptance length, and complete-vs-bipartite RDMA with
random-among-ties source selection. Every claim is labelled measured or predicted and each
point carries the measurement that would falsify it. None of the four is measured yet.

**Still no serving data** against the >=30% TTFT/ITL and >=10% throughput bar. Multicast,
cross-role backfill, the 4-mode MTP run and the EDF comparison are unstarted.

## v0.10.3 — Framing correction: the 42.8x was our bug's size, not a gain (2026-10-07)

Upstream FA3 was never 42x slower than FA2; that is not a plausible figure for a tuned Hopper
kernel. The 476 ms came entirely from our adapter -- `MakeUpstreamOwnSchedArgs` passing
`seqused_k` where the scheduler wants the query length, inflating m-blocks up to 49x and making
the cost `Theta(kv^2)`. So **"FA3's half 42.8x faster" measures the defect, not an
optimisation**, and v0.10.1/v0.10.2 quote it as a headline result where they should not.

Treating 476 ms as a believable FA3 number is also what cost six hypotheses: traits, pipeline
depth, gather granularity and SM budget are all constant factors and none can produce a 49x
work inflation. The exponent should have been measured first, and an implausible baseline
should have been read as a port bug immediately.

Post-fix sanity check supports that: FA3 now reaches 443 TFLOP/s (45% of H200's 989 peak)
against FA2's 325 (33%), which is the expected ordering for a wgmma kernel against an
Ampere-style `mma.sync` one.

**The only defensible performance claim is split vs FA2-on-everything, 1.23-1.25x** (kernel
batch, not serving). There is still no baseline-FA3 arm; the three-arm driver
(`drivers/ab_three_arm.sh`, sglang `1d8897736`) exists to supply it, and it is also the test
for whether further adapter defects remain -- a 49x inflation went undetected because every
row stayed bit-correct, the surplus m-blocks landing on masked q rows.

## v0.10.1 — ROOT CAUSE: the scheduler was handed `seqused_k` where it needs the query length; the split now beats FA2 (2026-10-06)

**FA3's long-context slowness was `Theta(kv^2)`, from one wrong argument.**
`MakeUpstreamOwnSchedArgs` passed `p.seqused_k` into `TileSchedulerArguments::seqused`,
which is the **query** length per request: the scheduler computes
`num_m_blocks = ceil_div(seqused[b] * qhead_per_khead, kBlockM)` from it
(`tile_scheduler.hpp:599-621`, `Prepared=false`). So m-blocks were sized from KV --
112 correct at q=2048 against **5 488** emitted at kv=100 352, a 49x bloat -- and
multiplied by an n-block count itself proportional to KV, the work became quadratic.

Fixed by passing `nullptr`: the scheduler then differences `cu_seqlens`, and `cu_seqlens_q`
is already supplied per-request, so it yields the exact query length with no new buffer.

**Verified.** Same fixtures before and after, every row PASS:

| kv | before | after | speedup | predicted |
|---|---|---|---|---|
| 10 240 | 3 268.1 us | **792.4 us** | 4.1x | 5.0x |
| 18 432 | 10 583.7 us | **1 396.0 us** | 7.6x | 9.0x |
| 34 816 | 37 811.2 us | **2 610.8 us** | 14.5x | 17.0x |
| 67 584 | 142 434.4 us | **5 023.3 us** | 28.4x | 33.0x |
| 100 352 | 320 401.6 us | **7 425.4 us** | 43.1x | 49.0x |

`us/kv` constant to 4.5% (was a 197% spread); `us/kv^2` now varies 218% (was 2.1%). Linear,
as it must be. Speedups track predicted bloat at ~87% across a 10x range.

**TraceLab batch: the split finally wins.** FA2-on-everything 15 190.9 us against the split's
**12 399.7 us** -- **1.225x, 18.4% lower** -- with the FA3 half going 476 684 -> 11 127 us
(**42.8x**). This retires the v0.9.9/v0.10.0 finding that FA3 is 31x slower and the split
loses on long context.

**Why it hid behind every gate.** The surplus m-blocks address q rows past each request's end,
which the epilogue masks, so output was always bit-correct and only the clock was wrong. A
numerics-only validity gate cannot see this, and that was the gate in use.

**Why six hypotheses each returned ~2%.** `MmaPV_is_RS`, `LPT`, pipeline depth, cp.async vs
TMA, page 1 vs 128, SM budget -- all constant-factor knobs, and **none can change an
exponent**. Each was correctly measured and uninformative. Procedural lesson: measure the
exponent before hunting constants; a five-point KV sweep costs minutes.

**RETRACTED: the 400 TFLOP/s peak behind every roofline figure in the docs.** H200 SXM bf16
dense peak is **989.4 TFLOP/s**. Recomputed, FA2 achieves 325 TFLOP/s = **33% of peak**, not
the "1.2x off roofline, essentially at peak" claimed in v0.10.0; fixed FA3 achieves
443 TFLOP/s = 45%, above FA2. The earlier "39x off roofline" was ~95x.

Consequence: **priority item 1 is reopened on its explanation.** The flat FA2 K/V-pipelining
measurement (depths 2/3/4 within 0.5%) stands, but it was explained by FA2 being at roofline,
which is false at 33% of peak. The flatness is unexplained.

This is the fifth bug of one class in this port -- per-request vs per-item, or q vs k, given
to a parameter wanting the other. The other four were found by which rows were wrong; this one
produced no wrong rows and was reachable only through a scaling measurement.

flashinfer `e4dfa82`, sglang `7749120b2`.
Command: `/dev/shm/w/verify.sh` on vibecuda trial `444d5bc154bd9050`.

## v0.10.0 — The FA3 trait audit is worth 2.4%, not 31x; FA2 pipelining is flat; two gates were never gating (2026-10-06)

Ten-arm A/B on a rebuilt pod (trial `444d5bc154bd9050`), `drivers/ab_fa3_traits.sh`,
TraceLab batch (1.58M KV, page 128, 5 iters). **Both reproduction gates pass**: the
FA2-all control gives 15 197 us against the old pod's 15 301, and `rs0` (the old
configuration) gives 476 684 us against 480 459 -- so the rebuilt pod is comparable and
every delta below is attributable.

**RETRACTED: `MmaPV_is_RS` is not the cause of FA3's 31x.** It is worth 2.3%.

| arm | fa3 half | vs `rs0` |
|---|---|---|
| `rs0` RS=0 stages2 LPT=1 (old) | 476 684 us | - |
| `rs1` RS=1 stages2 LPT=0 | 465 577 us | -2.3% |
| `rs1s3` RS=1 stages3 LPT=0 | 472 889 us | -0.8% |
| `rs1l1` RS=1 stages2 LPT=1 | 472 865 us | -0.8% |

All three deviations the upstream audit found were real and together are worth **2.4%**.
Kept: `rs1` and `LPT=0` (free, and `rs1` frees 32 KB). Dropped: stages=3, which is 1.6%
*slower* despite fitting the 232 448 B limit exactly.

How the claim went wrong: from perfect SM scaling plus low utilisation I inferred a
latency-bound mainloop, found a genuine trait deviation that could serialise softmax
against the PV mma, and treated it as the explanation. The smem measurement that followed
(32 768 B freed = exactly one P tile) confirmed the trait changes *storage*, and I read it
as confirming the trait explains the *time*. Independent claims; only the second was the
root-cause claim. The same error repeated with stages=3 -- "fits to the byte" was elegant
and predicted nothing.

**The 31x is unexplained, and now tightly bounded.** Ruled out: the gather (TMA slower),
pipeline depth on both executors, parallelism (linear to 132 SMs), per-tile work (identical
FLOPs), and every upstream trait divergence. What remains is per-iteration: ~589k
tile-block steps, ~4 464 per SM, so 476 ms is **~106 us per 128x128x128 step against ~2.8 us
expected**, invariant across nine arms with a stable control. That is a profiler question,
not a fifth trait hypothesis.

**Priority item 1 closed, no change: FA2 K/V pipelining is flat.** `FA2_MMA_KV` 2/3/4 give
15 218.8 / 15 296.6 / 15 219.5 us -- within 0.5%. FA2 runs this batch at 322 TFLOP/s, 1.2x
off peak, so there is no latency to hide. This supersedes the standing "worth ~15%, blocked
by an smem byte count" estimate, which came from the fused kernel where FA2's slot competed
with FA3 for smem; the split removed the constraint and nobody rechecked the premise.

**RETRACTED: the "all rows PASS" validity gate was output truncation.** The page-128 arms'
recorded "rel err 2.4e-4 to 4.2e-3, all rows correct" came from a driver piping through
`| head -12`, which on a 22-row fixture shows only rows 0-5 -- exactly that range. Rows 6-21
and the `FAILURES:` summary were never printed. With full logs kept, **row 21 fails in every
arm** (control 0.0262, `rs0` 0.0296, `rs1` 0.0344, `rs1s3` 0.0308) against `TOL=0.02`, and
the error tracks **row index, not KV length** -- rows 7/8/10/13/16/17/20 are all
`kv=65537, qo=1` yet give 0.0088 -> 0.0177. This is the unmodified persistent LPT kernel, so
it predates every change here, and it is unexplained.

**The 16384-KV prediction arms are invalid, so that prediction is untested.** Both ran at
`num_splits=29` and returned garbage (row 3 rel err 1.33), because `partial_o` is sized with
no `num_splits` factor -- the latent overflow the doc had already identified. Fixing that
sizing is a prerequisite for any fixture where `num_splits > 1`.

Five pod-restore defects were fixed and committed as drivers rather than left in /tmp:
dangling absolute `flashinfer/data` symlinks, the empty JIT generated-sources cache,
`python -I` ignoring PYTHONPATH, `provision_trees.sh` having never worked, and `px.py`
itself having survived three trial expiries only in /tmp.

## v0.9.9 — FA3's long-context 31x traced to one wrong template argument; two retractions (2026-10-06)

**Root cause of the 31x: `MmaPV_is_RS=false`.** Upstream's own tuner for head_dim 128 /
bf16 returns `{kBlockM=128, kBlockN=128, MmaPV_is_RS=true, IntraWGOverlap=true}`
(`tile_size.h:33`). We matched three of the four and passed `MmaPV_is_RS=false`, set as a
guess in the first port commit and never measured. With it false the softmax result P
round-trips through smem with a barrier per KV block rather than staying in registers for
the PV mma, serialising softmax against the mma. Now defaulted to upstream's value, with
the old behaviour kept only as an A/B switch (`MMA_PV_RS`).

The diagnosis came from an `FA3_SM_BUDGET` sweep on the TraceLab batch:

| SM budget | 16 | 32 | 64 | 96 | 132 |
|---|---|---|---|---|---|
| fa3 half (us) | 3 826 892 | 1 918 061 | 950 664 | 634 346 | **480 467** |
| budget x time | 61 230 | 61 378 | 60 843 | 60 897 | 63 422 |

Constant within 4% over an 8x range, so throughput is perfectly linear in SM count to 132
with nothing saturated -- at 132 SMs FA3 hits 10.3 TFLOP/s and 80 GB/s against FA2's
322 TFLOP/s and 2 518 GB/s on the identical 4.93 TFLOP / 39 GB. That rules out both
standing hypotheses (under-parallelised; more work per tile) and leaves a latency-bound
mainloop, which is what the wrong trait produces.

`FLASHINFER_FA3_UPSTREAM_STAGES` also goes 1 -> 2, upstream's unconditional Sm90 value.
The 1-stage default was there to fit beside the FA2 slot in the fused kernel, which is
closed. 3 stages does not fit at all -- it fails
`cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` against the 232 448 B limit -- so depth
was never the lever.

**Retraction 1: the 31x is not the gather, and not `page_size = 1`.** v0.9.8's entry
claimed upstream's non-TMA paged gather cost one index lookup per KV token and predicted
`PagedKVNonTMA=false` at `page_size=128` would fix it. Measured, all rows PASS, 0
fallbacks: FA2-all baseline 15 301 us; split cp.async 480 459 us; split **TMA** 496 978 us.
TMA paged is *slower*, and coarser paging does nothing for cp.async either (480 ms at page
128 vs 477 ms at page 1). Coarser paging costs the FA2 baseline nothing, so the fixture is
not at fault. The multicast item's TMA prerequisite is now met in passing, but it must pay
back TMA's own 16 ms deficit before it can show a win.

**Retraction 2: the roofline figures were wrong.** "155x off on 1.23 TFLOP" omitted the
QO-head factor -- all 28 heads attend the whole KV, so the two append rows are 4.93 TFLOP,
12.3 ms at 400 TFLOP/s. Corrected: **FA2 at 15.3 ms is 1.2x off roofline**, not 5x, and
FA3 is 39x off, not 155x. The material part is FA2: it is already at the roofline on this
workload, so there is no FA2-side headroom to chase and any win must come from FA3's own
39x.

## v0.9.8 — Port's row-ownership faults all closed; a self-inflicted baseline regression caught (2026-10-06)

**The upstream FA3 port no longer loses or overwrites rows.** Three faults, all the same
shape -- upstream carries a per-item or per-subset field and was handed the whole-batch
aggregate:

1. per-item `num_splits` in `split_idx`'s top 16 bits;
2. `cu_seqlens_q` / `seqused_k` read from the plan's per-WORK-ITEM arrays instead of
   per-request ones (fixing this took fixture 2 from 439/512 unwritten to **0/512**);
3. the combine pass walking every request, which reduced never-written partials over the
   FA2 executor's rows (an FA2 decode row went 5.8e-4 -> 0.155, and back to **0.00072**
   once restricted via upstream's `varlen_batch_idx_ptr`).

Each produced plausible numbers rather than an error, and two were located only by which
rows failed. Remaining: both wgmma rows are still wrong (10.88, 11.06) while the FA2 row
is correct, so the fault is now inside the upstream path's own result. The isolating
experiment is recorded -- force `max_splits_fa3 == 1` so the epilogue writes final O and
the combine never runs.

**A regression I introduced, and the control arm caught it.** v0.9.7's executor-aware
split denominator is reverted:

| | fused | baseline |
|---|---|---|
| before | 155.9 us | **72.3 us** |
| with the change | 156.3 us | **134.6 us** |

Dividing `kv_len_limit` by `clusters * ctas_per_sm` splits the FA2 queues twice as
finely; each extra split costs a partial write and a reduction pass that outweighs the
added concurrency. The fused path did not move, so the fused/baseline ratio would have
read as "improved from 2.16x to 1.16x" purely because the baseline got worse. That is
exactly what the control arm is for. Kept behind `FLASHINFER_SPLIT_BY_EXEC_SLOTS=1` for
reproducibility only; `QUEUE_EXEC` stays, since the explicit queue -> executor mapping is
not implicated.

## v0.9.7 — Per-queue split length follows the executor; planning cost is not the bottleneck (2026-10-06)

The scheduler's per-queue split length now accounts for which executor drains that
queue. The FA2 persistent kernel runs 128 threads at 2 CTAs/SM and the wgmma executor
384 threads at 1 CTA/SM, so the same SM count gives them different concurrency;
`kv_len_limit` was being divided by a bare cluster count, under-splitting the FA2 queues
by 2x against their real slot count. `QUEUE_EXEC` also makes the queue -> executor
mapping explicit instead of leaving task 3 implicitly special.

The rest of the "unified scheduler" asked for already existed: tile classes are routed
to queues by packed row count (>= 128 fills a wgmma tile, else 128/64/16), each queue
gets its own `cluster_len_kv_chunk`, and LPT min-heap assignment runs inside each queue.

**A device-side scheduler is not justified by the evidence.** Measured earlier:

| | tok/s | vs control |
|---|---|---|
| control, fused off | 22 203 | - |
| fused, original wiring | 20 260 | -8.7% |
| fused, planning deferred + indices reused + plan cached | 20 314 | **-8.5%** |

Removing essentially all host-side planning recovered 0.3%. That measurement had two
plans per step serving a kernel used on 3.2% of forwards, so the single unified plan's
per-step cost is still un-isolated -- which is the thing to measure before writing a GPU
scheduler, rather than assuming either way.

## v0.9.6 — One kernel can give either role its native registers, never both (2026-10-06)

The Q-split was the experiment that would have kept single-kernel fusion. It does not:

| FA2 configuration | isolated | inside the fused kernel |
|---|---|---|
| 4 warps, `num_mma_q=2` (today) | 1 136 B spill | 3 232 B spill |
| 8 warps, `num_mma_q=1` | **88 B** | **1 504 B** |

Halving the per-thread O accumulator is worth keeping -- it takes the fused kernel from
3 232 to 1 504 B -- but ~1.4 KB survives that no FA2-side change can reach, because the
FA3 role's live ranges share the allocation. With the converse already measured
(outlining FA2 puts FA3 at 64 B and moves 17 500 B onto FA2), the result is symmetric:
one `__global__` can give either role its native allocation, not both.

Decision: **the kernel splits, the plan does not.** The 4-mode tile assignment, the
work-proportional SM allocation, the unified JIT split-KV reduction and the fused
tree-build epilogue all live in `HolisticPlanInfo<4>`, which both executors read, so
none of them depends on the roles sharing a `__global__`. What is given up is
cross-role backfill inside a CTA -- itself the cause of the co-compilation spill -- and
cross-role smem multicast.

Graph count does not multiply, provided the split lives in device memory and both
executors launch fixed grids with unadmitted CTAs exiting early. This is required for
the fused kernel too: `sm_split` is a kernel argument today, which a graph would bake
at capture. sglang's graph layer already uses the same bucket-and-pad idea a level up
and treats prefill and decode as separate capture ladders, so two executors add no
PD-ratio combinations.

Also fixed here: the combine pass's per-request split count was derived from the plan's
`kv_indptr`, which is pushed once per WORK ITEM (`scheduler.cuh:1583`), so differencing
adjacent entries subtracted two unrelated items instead of bounding a request's pages.
That is what left 439/512 rows unwritten with the combine otherwise correctly wired.
Per-request KV lengths now come from the caller.

## v0.9.5 — FA2's spill is the O accumulator; the parity gap is 2.16x (2026-10-06)

Two measurements and one fix.

**The gap, on one build.** Fixture 2: 155.9 us fused against 72.3 us for
FA2-on-everything, both numerics-correct from the same build and inputs. 2.16x, which
is larger than the 1.83x recorded on the 16 823-token-KV fixture -- so the gap is
workload-dependent and the *shorter* fixture is the harder case. Worth knowing,
because the split sweep that produced the 1.83x ran on the easier one.

**What FA2's share of it is.** Three compile-only probes, each a 384-thread kernel
containing only the FA2 runners:

| FA2 configuration | threads used | spill stores |
|---|---|---|
| `4 warps_q, num_mma_q=2` (today) | 128 of 384 | 1 136 B |
| `4 warps_q, 3 warps_kv` (all threads busy) | 384 | **1 276 B** |
| `8 warps_q, num_mma_q=1` | 256 of 384 | **88 B** |
| native `128 x 2 CTA`, for reference | 256 | 0 B |

Occupying all 384 threads by splitting KV is *slightly worse* than leaving 256 idle:
every KV-warp still accumulates a full O tile over its q rows, so per-thread registers
do not fall and the cross-warp reduction adds work. That is a direct answer to
"offload some softmax work to the remaining threads" -- the softmax is not what holds
the registers. Splitting **Q** is: 8 warps with `num_mma_q=1` covers the same 128 rows
with 16 per warp instead of 32, halving the accumulator, and spill falls 12.9x to
within 88 B of native. Available as `FLASHINFER_FA2_QSPLIT=1`; tile 128 only, since
tile 64 already has 16 rows per warp and neither 64 nor 16 divides into 8 warps.

**Fix.** The combine launch failed with "invalid argument" because its grid is
`ceil_div(seqlen_q * num_heads, kBlockM)` and I passed `seqlen_q = 0`, making grid.x
zero. Upstream documents the field as the varlen *maximum* q length
(`flash_fwd_combine_kernel.h:292`); it now travels on the plan from both call sites.

Correctness and timing of the Q-split configuration inside the full fused kernel are
not yet measured -- the probes are compile-only.

## v0.9.4 — The port's split-KV path: upstream's combine pass, and two silent-wrong traps (2026-10-06)

Fixture 2 measured the predicted failure: with `max_splits_fa3=2`, both upstream arms
left **511 of 512 rows unwritten**, the single correct row being the FA2 decode row.
The port instantiates upstream with `Split=true`, so its epilogue takes the
`is_split` branch (`epilogue_fwd.hpp:232`) and writes partials instead of final O,
and flash-attention reduces those in a *separate* kernel
(`flash_fwd_combine_kernel.h`) that the port never launched. It does now, guarded on
`num_splits > 1` because at 1 the epilogue writes final O directly -- the same guard
upstream's own `flash_api` uses.

Two traps found while wiring it. Both would have produced plausible wrong numbers
rather than an error, which is the only reason they are worth writing down:

1. **Per-item `num_splits` lives in the top 16 bits of `split_idx`.** Upstream packs
   it there and masks the index to the low 16 (`epilogue_fwd.hpp:226-231`). Leaving
   the high bits zero makes every row fall back to the plan-wide maximum, so a row
   that did *not* split writes one partial, still takes the `is_split` branch, and
   the combine pass averages it against uninitialised memory. The adapter now encodes
   each row's own chunk count, which is what the field exists for.
2. **The combine pass needs the same per REQUEST**, through
   `num_splits_dynamic_ptr`; `nullptr` means "use the maximum for everyone" with the
   same consequence. A small kernel fills it from `kv_indptr` and `len_kv_chunk` --
   the same two quantities the epilogue derives from, so writer and reader cannot
   disagree.

The partials also now use upstream's own varlen layout,
`out_accum[num_splits, num_heads, total_q, head_dim]` (`flash_api.cpp:1102`), instead
of the equally dense but differently ordered layout I first wrote; the combine kernel
is written against upstream's, so mine would have reduced the wrong elements. Strides
and element counts come from shared helpers and the ext sizes its buffers from those
same helpers. Two sizing bugs fixed on the way: the O-partial split stride omitted
the `num_heads` factor, and O's strides were being taken from Q.

Also recorded: fixture 2's `def` arm is 156.8 us, and that is **not** a regression
against the 772 us in the status section -- the 772 belongs to the 16 823-token-KV
long-prefix fixture, while fixture 2's longest row is kv=5469. Two workloads, not a
change.

The default path builds unchanged throughout (0 errors).

## v0.9.3 — The two roles cannot both have their native register allocation in one launch (2026-10-06)

Five builds, `-Xptxas -v`, attributing every spill count to a named function. This
supersedes v0.9.2's conclusion, which was right that co-compilation causes FA3's
spill and wrong to imply outlining removes it -- it moves it.

| configuration | FA3 role | FA2 role |
|---|---|---|
| both roles inlined (today's default) | 33 676 B, shared allocation | shared |
| FA2 body outlined (`FLASHINFER_FUSED_DRAIN_NOINLINE=1`) | **64 B** | **17 500 / 17 664 B** |
| upstream FA3 alone, 384 thr | 52 B | — |
| FA2 alone, 384 thr, 168 regs | — | 1 136 B |
| FA2 alone, 128 thr x 2 CTA, 255 regs | — | **0 B** |

Two independent costs, now separated:

1. **Co-compilation.** FA3's 33 676 B is not intrinsic. Move the FA2 body out of the
   function and FA3 spills 64 B -- the standalone figure. ptxas allocates per
   function, so while both roles are inlined their live ranges share one
   allocation even though the roles sit in mutually exclusive `blockIdx` branches.

2. **Launch geometry.** Outlining does not make the spill disappear; the FA2 body
   then spills 17 500 B in its own ABI frame, because an ABI callee cannot use the
   caller's allocation. And FA2's floor inside *any* 384-thread launch is 1 136 B at
   168 registers, against 0 B at 255 registers in its native 128x2 geometry --
   measured with no FA3 in the kernel at all, so this is the geometry alone.

The arithmetic behind the second: 384 threads at 1 CTA/SM leaves 65536/384 = 170
registers per thread, and FA2's mainloop uses 255 when it can, which requires at
most 256 threads per SM. One launch has one blockDim, so the FA2 role cannot hold
255 registers per thread while the FA3 role needs 384 threads in one CTA for its
warp specialisation.

**Correction, same day.** I first concluded from that that FA2 parity is
unreachable in one launch. That does not follow, and the arithmetic says so:

    native FA2   255 regs x 256 threads/SM = 65 280
    fused FA2    170 regs x 384 threads/SM = 65 280

The SM-wide register budget is *identical*. What changes is the partition, not the
total, so the deficit is only real if FA2's work cannot be spread across three
warpgroups instead of two CTAs. That makes it an engineering question about work
distribution, not a hardware limit -- and `FLASHINFER_FUSED_WG_DRAIN=3` is exactly
that redistribution, currently producing wrong FA2 output. The honest status is that
FA2 parity is unproven and blocked on WG_DRAIN=3 being made correct, which is also
what "offload some softmax work to the remaining threads" asks for.

What the rows above *do* establish: the parity bar is not reachable by scheduling,
stage count or split policy, all held constant across them. FA3 parity is reachable
by outlining the other role. FA2's penalty at `WG_DRAIN=1` is 1 136 B of spill and
87 registers per thread, and that figure is the cost of leaving 256 of 384 threads
idle, not a floor.

`FLASHINFER_FUSED_DRAIN_NOINLINE` is kept as a switch rather than turned on,
because it is a trade whose sign depends on the work mix: it should win where FA3
dominates (fixture 2 is ~99% task-3 work) and lose on decode-heavy batches. Which
way it goes is a measurement, not a choice, and has not been run yet.

## v0.9.2 — Measured: single-kernel co-compilation, not the register cap, is what starves FA3 (2026-10-06)

The register-pressure hypothesis from v0.9.0 is confirmed in outline and wrong in
its mechanism, and the corrected version is sharper.

`-Xptxas -v` on three builds of the same kernel, all `__launch_bounds__(384, 1)`:

| build | regs | stack | spill stores | spill loads |
|---|---|---|---|---|
| upstream FA3 alone, our template | 168 | 32 B | 52 B | 40 B |
| upstream FA3 + the three FA2 runners | 168 | 2 976 B | 33 676 B | 37 852 B |
| flashinfer FA3 + the three FA2 runners | 168 | 848 B | 3 232 B | 5 028 B |

I had predicted that upstream would be starved because it asks for 232 registers per
MMA warpgroup through `warpgroup_reg_alloc` while `__launch_bounds__(384, 1)` caps
the uniform allocation near 170, and that the fix was to make the FA2 role
participate in the same register protocol. That prediction is refuted: upstream
compiled **alone** in exactly the same template, with the same bounds and the same
168 registers, spills 52 bytes. The cap is not the problem.

What costs the 33 KB is that both roles are one function. ptxas allocates registers
per kernel, so the FA2 mainloop's live ranges and upstream's share a single
allocation, and at equal register count the fused build spills 648x more than the
standalone. The default path shows the same effect an order of magnitude smaller
(3 232 B), which is why it was visible but not fatal there.

This is a property of single-kernel fusion itself, not of the split policy, the
thread width, or the stage count -- none of which differ between the two rows that
matter. It is also why ptxas emits C7510 (wgmma serialized across a function
boundary) only on the upstream path.

The consequence for the goal is direct: per-role parity cannot be reached by tuning
the split while both roles share one register allocation. Either the roles must be
separated in a way that gives each its own allocation, or the FA2 role's live ranges
must be cut down. No parity number should be quoted until this is resolved, and the
1.83x gap already measured is most likely this.

## v0.9.1 — The upstream FA3 port compiles, and the planner exports its split count (2026-10-06)

The port's dispatch now instantiates end to end with `FLASHINFER_FA3_UPSTREAM=1`:
a TU that takes the address of `FusedFA3PersistentDispatched` compiles the whole
upstream kernel body with 0 errors, and the default path compiles from the same TU
unchanged. Previous commits in this port claimed "compile-checked" on code that the
upstream scheduler's `if constexpr` false branch never instantiated, so everything
here is verified by forcing instantiation rather than by the header parsing.

Every params field name in the dispatch had been guessed, and every guess was wrong
in the same way: the KV pointers and strides were read off `params_1`, the FA2
role's `PersistentParams`, as `k_data`/`v_data`/`kv_stride_n`. Those members do not
exist (they are `k`/`v`/`k_stride_n`), but the deeper error is that the wgmma role
was reading the other role's params at all -- `FA3Params` carries its own
`k_ptr`/`v_ptr`/`k_stride_n`/`k_stride_h`/`kv_indices`/`page_size`, and taking them
from the FA2 role would have been correct only because this harness happens to hand
both roles the same cache.

`max_splits_fa3` is now exported through `plan_info`. Upstream sizes its float
partial O/LSE as `total_q * num_splits`, so the buffers cannot be sized by a
conservative bound -- the obvious one, pages per sequence, asks for
`8192 * 32 * 8192 * 4` bytes of LSE alone at a 8192-token batch. The planner knows
the exact value, because it is the `num_kv_tiles` it just chose, and nothing
downstream can recompute it. It is placed *before* `num_fa3_clusters` in the
serialised vector so that stays last: the harness reads the chosen split as
`plan_info[-1]`, and appending would have handed it the split count instead -- a
plausible small integer that would have planned FA3 for the wrong CTA count with no
error anywhere.

The export is guarded by `NUM_TASKS > kFA3Task`. I first wrote it unguarded on the
assumption that the enclosing block was already that guard; it is a bare scope, and
`task_max_kv_splits` has `NUM_TASKS` entries, so every `HolisticPlanInfo<3>` plan
would have read index 3 out of range.

The three workspace buffers are allocated in the ext, and `num_requests` /
`max_pages_per_seq` are threaded from both call sites (the harness and the serving
backend). The dense page table is filled with -1, not 0, because 0 is a valid page
id and zero padding would make every short sequence read page 0 for its tail blocks
instead of being masked.

Still unvalidated at runtime: this machine has no GPU, so no number from the
upstream path means anything yet.

## v0.9.0 — Register pressure is the leading candidate for the 2.19x (2026-10-05)

Grounded in ptxas numbers already in `persistent_fa3_fused.cuh`, not in new
reasoning. The fused kernel is `__launch_bounds__(384, 1)` because that is FA3's
warp-specialised width, capping ~170 registers per thread; the standalone persistent
kernel at 128 threads x 2 CTAs/SM has roughly 256. The file records the consequence:
the fused body compiles to 168 registers with up to 4.4 KB of spill loads, and 6284
bytes of spill stores in one configuration.

It fits every measurement the other candidates failed. It is grid-independent (SM
reservation 6%), protocol-independent (Q completion 0.3%), predicts that more
warpgroups are worse (width 3: 147.9 vs 132.4 us, since extra warpgroups do not
raise the ceiling), and survives at equal threads per SM -- the baseline forced to 1
CTA/SM has the same 128 FA2 threads per SM and is still 2.19x faster.

The test needs no GPU: compile the persistent mainloop under `__launch_bounds__(128,
2)` against `(384, 1)` with `-Xptxas -v` and compare registers and spills. If it
holds, the design implication is sharp -- one launch forces one set of launch bounds
on both roles, the wider warp specialisation sets them, and SM pre-assignment inside
a single kernel cannot be made free.


## v0.8.9 — The serving fallback is strict by default (2026-10-05)

A fused planning failure used to degrade to the persistent path with a warning. That
is how the width-3 arm produced an apparent win -- 505.85 tok/s against 397.88 --
while the fused launch was actually being rejected and roughly half the steps ran on
the faster unfused kernel (composition 1897 at 47.4% against 53116 at 96.2%). The
warning did not help, because the number was still produced and believed.

`_plan_fused_split` now raises, naming the cause and saying that a run with a
fallback compares a mixture rather than the fused path.
`SGLANG_FUSED_ALLOW_FALLBACK=1` restores degradation for keeping a server alive, and
never for measurement.


## v0.8.8 — The penalty factorises: 1.38x structural, 2.19x unexplained (2026-10-05)

`scheduler.cuh:1199` gives the 3-task plan 2 CTAs/SM and the fused 4-task plan one,
since FA3's ~98 KB of smem makes two impossible. Forcing the baseline to one CTA/SM
isolates it: 43.8 us at two, 60.3 us at one, against the fused kernel's 132.1 us. So
3.02x = 1.38x (one CTA/SM) x 2.19x (the fused kernel itself).

The 1.38x is the structural entry price of co-residency with FA3's smem and is not
addressable by scheduling at any quality. The 2.19x remains uncaused, with the Q
protocol (0.3%), SM reservation (6%), thread width (negative), plan construction
(none) and now CTA count (1.38x) all measured and eliminated.

It also corrects an earlier framing error here: I called the launch-geometry finding
"refuted" by the width result, conflating threads within a CTA (more is worse --
warpgroups contend on one queue cursor) with CTAs per SM (worth 1.38x -- independent
contexts). This file used "launch geometry" for both; they are different mechanisms.


## v0.8.7 — Aliased smem: the first measured win, 11% (2026-10-05)

With width held at 1 and smem the only variable: aliased 118.1 us against disjoint
132.4 us, baseline 43.8-44.0 us in the same runs. 11% faster but NOT free, and the default must NOT flip --
corrected after reading persistent_fa3_fused.cuh:78-93. Disjoint is the fix for a
documented race: CUTLASS initialises pipeline mbarriers before the sm_split branch,
so an aliased base of 0 puts the drain's slot inside FA3's SharedStorage, which gave
nondeterministic rel err 1.93 / 2.53 / 3.31 -- and it fires only when a CTA drains
FA2 before entering FA3. My run had the wgmma role EMPTY, so nothing entered FA3
after a drain and the race could not appear. Capturing the 11% means removing the
race (re-initialising FA3 pipeline state after the rendezvous), not setting a flag.

It does not explain the 3x. 2.68x remains. Measured contributions on identical work
now read: SM reservation ~6% over an 8x sweep, smem layout 11%, residual ~2.7x
uncaused.

The queue protocol is the primary suspect, from two independent results: adding
warpgroups made the kernel slower (147.9 vs 132.4 us), which is contention rather
than capacity, and the penalty barely moves with the number of reserved SMs, which
fits a serialised hand-off. Next measurement is the per-task profiler events in
`persistent.cuh` -- queue acquire versus mainloop, per role.


## v0.8.6 — The 3x is intrinsic to the fused kernel (2026-10-05)

Sweeping sm_split with the wgmma role empty and plan_ctas matched at every point:
129.7 / 130.5 / 132.2 / 137.4 us at 2 / 4 / 8 / 16 reserved clusters, against a
42.6-43.9 us unfused baseline measured in the same runs. An 8x change in reserved
clusters costs 6%, so SM reservation is not the cause and the confound flagged last
entry is real but small -- my hypothesis that it explained the 3.01x is refuted.

What stands: the fused kernel is ~3x slower than the unfused persistent kernel on
identical work (3.02-3.13x across configurations, matching the 3.00x serving ITL
ratio), and that penalty belongs to the fused kernel itself. The SM split, the Q
completion protocol, thread width, and plan construction have each now been measured
and eliminated.

This reframes the effort: the scheduling argument is about how work is distributed
and cannot be worth 3x. No schedule reaches the 30% target while the fused path
starts a factor of three behind. Next, and never yet measured here: FA2 occupancy
inside the fused kernel (1 CTA/SM where it natively runs 2), the element-wise
packed-GQA O store, and the queue protocol -- which the width-3 result hinted at,
since adding warpgroups made it slower, as a contended queue would.


## v0.8.5 — The kernel-level ratio: 3.01x, and the confound in it (2026-10-05)

With a separate 3-task plan as the reference, the fused kernel runs 132.2 us against
the unfused persistent kernel's 43.9 us on identical work -- 3.01x, matching the
serving ITL ratio of 3.00 to within noise. That settles host versus device: the cost
is in the kernel, and one fused plan replacing two is not the problem.

The confound: this configuration routes every row to an FA2 class, so the wgmma role
has no work, yet sm_split=8 still reserves 8 clusters for it. Idle reserved SMs are
part of that 3.01x and are a property of static pre-assignment, not of fusion. This
also qualifies the earlier decomposition -- the "fused FA2 path costs 1.50x" arm had
the same 8 idle clusters, so that figure is reservation waste plus something else,
not the intrinsic FA2-path cost I presented it as.

An sm_split sweep on the fixed batch separates the two, and the harness now has both
the timing loop and the reference to do it.


## v0.8.4 — More threads are slower; width 3 is numerically fine (2026-10-05)

Fixed-batch kernel timing settles the thread-count question against my hypothesis:
width 1 (FA2 role on 128 of 384 threads, disjoint smem) runs 132.4 us; width 3 (all
384, aliased smem) runs 147.9 us. Giving the FA2 role every thread makes the kernel
12% slower, so the launch-geometry framing is wrong as a performance explanation and
the warpgroup-reentrant path is a dead end for perf.

Separately it corrects a standing catalogue claim: width 3's numerics are CORRECT
with aliased smem (rel err 3.5e-3 / 3.8e-3 / 6.0e-4, all rows pass). The recorded
"FA2 numerics wrong at width 3" was almost certainly the disjoint-smem config -- the
same condition that makes the launch fail with 'invalid argument', since three
warpgroups need 208,896 of 232,448 B and only fit aliased. The fault was
configuration, not the reentrant logic.

The 1.50x fused-FA2 cost remains unexplained, now with its most plausible mechanism
eliminated, and 132.4 us still lacks an unfused reference.


## v0.8.3 — The width-3 probe is void by its own validity check (2026-10-05)

Giving the fused FA2 role all 384 threads (`SGLANG_FUSED_WG_DRAIN=3`, numerically
wrong, perf probe only) appeared to recover most of the 1.50x: 505.85 tok/s / 21.47
ms ITL against width 1's 397.88 / 27.41. The composition counter refutes it --
1897 (47.4%) against 53116 (96.2%), so the arm pushed a twenty-eighth as much work
through the fused path. Corrupt FA2 output ends sequences early, fewer decode steps
follow, and the benchmark finishes faster by doing less; TTFT dropping to 3255.95 ms
is not what a decode-side thread-count fix looks like.

The thread-count attribution is therefore unproven and serving is the wrong
instrument for it -- a wrong-numerics arm can always shift composition and buy an
apparent win. It needs a kernel-level timing on a fixed fixture batch instead. The
1.50x itself stands; its cause does not.


## v0.8.2 — The 66% decomposes into two causes (2026-10-05)

Routing every row to the fused kernel's FA2 roles, leaving the wgmma role idle,
splits the deficit: baseline 590.58 tok/s / 18.26 ms ITL, fused-FA2-only 397.88 /
27.41 (1.50x ITL), fused-with-wgmma 198.16 / 54.86 (3.00x ITL).

Two roughly equal multiplicative costs, not one. The first is the launch geometry
already measured in this effort and never connected to this regression: 1 CTA/SM at
384 threads where FA2 natively runs 2 CTAs/SM, ~128 of 384 threads doing FA2 work.
The second, a further 2.00x, is isolated to the wgmma role and still uncaused;
the element-wise packed-GQA O store and 200 task-3 items against ~34 q-tiles of real
work are the measurable candidates.

Also noted: `mixed_fused_fa3=53116(96.2%)` is identical across all three arms. It
records the dispatch decision, not which internal role ran, and must not be read as
evidence the wgmma role executed.


## v0.8.1 — The Q drain is exonerated; the 66% has no known cause (2026-10-05)

Replacing the cp.async group drain with the validated per-lane `_noinc` mbarrier
arrival moves serving by 0.3%: 198.16 tok/s against 197.65, ITL 54.86 against
55.01, identical 96.2% fused composition, baseline 590.58 / 18.26. The drain was
not the cause of the 3x.

The attribution was mine and it was wrong. I had written that the deficit "matches
the cp.async drain's cost exactly", carrying a ~3x ratio from a different
configuration into this workload as though it had been measured here. It had not.
The 66% regression has no identified cause.

What is settled: the Q completion protocol. `_noinc` is correct (fixture 2, rel err
3.3e-3, 0/512 unwritten), needs no drain, costs nothing, and can stay on. Next step
is a profile rather than another mechanism argument -- three mechanism hypotheses
have now each been refuted by measurement (static split, redundant chunks, the
drain).


## v0.8.0 — The chunks are real partials; hypothesis refuted (2026-10-05)

The per-item dump shows task 3's chunks are genuine slices (`span=384` + `span=55`
against `kv_len=439`, each with its own group id and partial_indptr), not full-row
copies. So the redundant-work hypothesis is refuted and there is no second hidden
cause of the serving deficit -- the Q completion remains the one measured cause.

It also dissolves the contradiction: the old fixture-2 pass reported that row as
`task0`, an FA2 class that reduces correctly, and it is routed to task 3 only now
that packed qo = 3073 exceeds the threshold. So the old pass was correct for a
different routing, and calling it "suspect" is withdrawn.

Net: the reduction spec stands as written, the guard and its 200-item count are
correct, and the cheap path to re-measuring the regression is to raise
`FLASHINFER_FA3_MIN_PACKED_QO` above 3073 so the split row goes to FA2 -- which
reproduces the original routing and unblocks validating `_noinc` against the drain.


## v0.7.9 — 200 split chunks and a correct answer (2026-10-05)

The exact guard reports 200 task-3 work items carrying JIT-reduction group ids on
fixture 2, in both arms. Fixture 2 nevertheless passed at rel err 3.2e-3 with 0/512
unwritten rows, and 200 unmerged distinct slices sharing final O rows cannot produce
that. Leading hypothesis: the plan emits each chunk's `kv_start`/`kv_end` as the
row's full range, so the chunks are redundant rather than partial -- identical
overwrites, invisible to a correctness check, and roughly 6x duplicated attention on
the wgmma role. That is the same order as the 3x serving deficit and independent of
the Q drain, so it may be a second and larger cause the drain estimate was masking.

Consequence for the plan of record: do not implement the reduction as specced until
this is settled by dumping task 3's per-item ranges, because the spec assumes the
chunks are genuine partials.


## v0.7.8 — The guard fires on fixture 2 (2026-10-05)

Ran the plan guard against fixture 2 to validate Q's completion protocol and got
a more useful answer than expected: the guard fires, in both arms, at
`scheduler.cuh:1495`. Fixture 2 routes two rows to task 3 (`qo=72` packed 504 and
`qo=439` packed 3073), 675 KV over 8 clusters gives a `kv_len_limit` near 84, and
the 439-long row splits about six ways. So my claim that fixture 2's FA3 row is
`kv=236` and never splits was wrong, and its earlier PASS at rel err 3.2e-3 is
suspect rather than reassuring -- either the overwrite was benign for that shape
or the guard over-fires relative to the chunk loop. Resolving which is step one of
the reduction work.

Net effect: the wgmma JIT reduction is now the hard blocker. With the guard in,
fixture 2 does not run, so there is no fixture left on which to settle the Q
completion protocol (`_noinc` vs drain) either.


## v0.7.7 — The first valid unified serving number: -66% (2026-10-05)

Measured on a tree verified byte-exact against our fork, both arms back to back:
baseline 590.58 output tok/s against unified 197.65 (0.335x), mean ITL 18.26 against
55.01 (3.01x), mean TTFT 2626.50 against 9333.23, with 96.2% of rows through the
fused path -- so this measures the kernel, not a dispatch accident. Full table,
commits and command in `tasks/debug_kernel_server_perf_4_mode.md`.

The 3x matches the packed-GQA Q drain exactly, and the Q-before-K hoist was
verifiably compiled in, which rules out ordering: the fence sweeps every cp.async
the producer has uncommitted from earlier work items, and groups retire in order.
The per-lane `cp.async.mbarrier.arrive` that has the right scope hung the server on
its first fused batch and is now gated off behind `FLASHINFER_FUSED_Q_MBARRIER`,
to be settled on fixture 2 rather than in another serving run.


## v0.7.6 — Two findings that invalidate measurements, not numbers (2026-10-05)

### The wgmma role's split-KV path is a latent wrong answer

Recorded as "unexercised" before, which was too kind. `scheduler.cuh`'s chunking
loop covers every task including `kFA3Task`, the unified scheduler carries
`partial_indptr`/`group_id` per work item, and `persistent_fa3_fused.cuh`
references neither -- nor anything else about partials -- so each chunk of a
split task-3 row writes its own slice straight to the final O rows and the
chunks overwrite each other. Fixture 2 is quiet about it only because its
FA3-class row is `kv=236` and never splits.

Corrected after measuring: I expected this to break the 8192 workload too, but
the unified arm ran all 200 prompts with the guard compiled in, so nothing
split there and the earlier unified numbers are slow rather than wrong. Whether
that is because no task-3 row exceeds its `kv_len_limit` or because the guard is
inert is still open -- the gate fixture decides it, and a guard that has never
fired proves nothing.

### The restored pod's serving tree was upstream sglang, not our fork

`/dev/shm/w/sgl_src` is a clone of upstream at `1818db4`, and the tree on
`PYTHONPATH` was built from it with four of our files copied on top. Against an
md5 manifest of our fork's 725 `.py` files: 4140 files on the pod, 3573 extra,
158 missing, 514 present in both with different contents -- `bench_serving.py`
among them. Fixtures are unaffected (they import `flashinfer` and `torch`, never
`sglang`), so fixture 2's pass and the `barrier_Q` diagnosis stand. Both forks
are private and this machine has no push credentials, so the repair is a chunked
webshell transfer of the real tree. Details, and the torch-ABI dependency chase
that preceded it, in `tasks/debug_kernel_server_perf_4_mode.md`.

## v0.7.5 — Q's drain stops draining K/V (2026-10-05)

### The packed-GQA Q load is now issued before the first K tile

The packed-GQA Q load on the sparse path finishes with a cp.async group drain
(`cp_async_fence` + `cp_async_wait<0>`) and then a single-thread arrive on
`barrier_Q`. That drain is what made the correct unified kernel ~3x slow
(8,379 against 24,885 tok/s), and the reason was mis-stated in the code: the
comment claimed narrowing the wait needed the KV pipeline's outstanding group
count.

There is no such count to learn. The KV copies signal completion through
`cpasync_barrier_arrive`, the mbarrier mechanism, and are never committed to a
cp.async group at all -- so `cp.async.commit_group` sweeps them, still
uncommitted, into the very group it closes for Q. Committed groups also retire
in order, so no `wait_group N` bound can retire Q's group while an older group
is outstanding. Both facts point the same way: the only way for Q's drain to
cover Q alone is for nothing older to be in flight when it runs.

So the Q load moves above the last-k-tile load, as a `load_q_tile` lambda
called from one of two sites (`FLASHINFER_FUSED_Q_LATE` restores the old
ordering for A/B). A per-lane `cpasync_barrier_arrive` on `barrier_Q` was the
other candidate and cannot work: that instruction tracks only the cp.async of
its own issuing thread, so all 32 lanes would have to arrive on a barrier whose
arrival count is 1 -- which is exactly the 1500 s hang already paid for once.

Compile-checked in three configurations (plain, packed-GQA, and `Q_LATE`).
Not yet measured on device: the serving A/B is what decides whether the ~3x
comes back, and it is reported in
`tasks/debug_kernel_server_perf_4_mode.md`.

### Fix: the lambda no longer captures structured bindings

Hoisting the block into a lambda made it capture `q_tile_idx`, `kv_head_idx`,
`qo_indptr`, `qo_len` and the Q tensors, all of which come from structured
bindings -- capturing those is C++20 and this builds as C++17 (nvcc only warns).
They are passed as init-captures over plain aliases instead, keeping the
original names so the body is unchanged.

### Provisioning: install sglang's dependency closure from the released package

Hand-listing dependencies cost several server-start cycles (pynvml, IPython,
`sgl_kernel`, one at a time), and `sgl_kernel` cannot come from a source tree.
`drivers/install_deps.sh` installs `sglang[srt]` for the closure and then
re-pins `torch==2.8.0+cu128`, which is mandatory rather than cosmetic: that
install drags torch forward, the fused extension needs 2.8.0, and a drifted
torch would quietly make measurements incomparable.

## v0.7.4 — One Scheduler for Both Roles (2026-10-05)

### Refactor: the wgmma role becomes a task in our own plan, not `PrefillSM90Plan`

The FA3 role was planned by flashinfer's `PrefillSM90Plan`, which measurement
showed cannot serve this kernel: it emits one work item per (q-tile x qo head x
row) carrying the **full** `kv_len`, with no partial-O/LSE buffers and no merge
map. Concretely, as the server planned it, the wgmma role had

- **no GQA packing** — `num_qo_tiles = ceil_div(qo_len, cta_tile_q)`
  (`scheduler.cuh:939`) plus upstream's own note that FA3 "do[es] not fuse query
  and group heads", against `qo_len * gqa_group_size` in the persistent plan
  (`scheduler.cuh:1199`);
- **no KV splitting** — one CTA walks a whole KV range.

At the measured serving shapes (median 11 decode rows totalling 47,935 KV, one
500-token prefill chunk over a 2,595-token prefix) that leaves the role almost no
parallel width, which is why fusing it changed a mixed step by -0.21% and
end-to-end throughput by -0.65%: parity was the *expected* result, not a tuning
failure.

`TwoStageHolisticPlan` is now generic in its task count. A 4-task plan adds the
wgmma class as task 3, which therefore inherits, with no new machinery:

- packed-GQA tiling (every class in this plan tiles packed rows),
- split-KV via the existing `kv_len_limit` chunk loop,
- a JIT-reduction group (`group_id`, `partial_indptr`, `merge_indptr`,
  `group_counter`) so partials are merged in-kernel by whichever work item
  arrives last, with no grid barrier and no second launch.

A 3-task `plan_info` selects exactly the previous behaviour, so `BatchAttention`
cannot silently drop task-3 work. Admission is `packed_qo_len >=
FLASHINFER_FA3_MIN_PACKED_QO` (default 128 = fills one packed tile), tunable for
the boundary sweep rather than asserted. Both instantiations compile for sm_90a.

### Measured: the symmetric flag is worth 1.59x on the serving path
At `sm_fa3=8`, asymmetric (8 CTAs own all FA3 work) vs symmetric: 6,279 vs
9,990 tok/s, mean ITL 73.05 vs 46.08 ms, p99 ITL 170.84 vs 63.44 ms, on identical
batch shapes (p10/p50 step time matched to 2%; only the tail moved). Previously
this was a kernel-level number (0.687x -> 1.003x of monolithic) only.

### Measurement hygiene: a stale diagnostic invalidated a day of numbers
A detached `compute-sanitizer synccheck` from the CLUSTER=2 investigation was
still resident after 2h54m at 100% CPU, holding 2,760 MiB. Every arm measured
alongside it ran 2.45x slow (fused 9,990 vs 24,431 tok/s on the same command) and
*stayed consistent with its own control*, so no A/B looked wrong. Killing it
restored yesterday's scale. The measurement driver now refuses to start while
`nvidia-smi --query-compute-apps` is non-empty, and synccheck is recorded as
non-viable on this kernel (3 hours without reaching a launch result).

### Per-step trace carries features and GPU time
`SGLANG_STEP_TRACE=1` now logs per-step row classes (prefill rows, chunk tokens,
prefix, decode rows, decode KV, total KV, max chunk) and the step's own GPU span
from CUDA events read a step later, so nothing synchronises on the measured path.
Cost: 0.02% (24,668 vs 24,431 tok/s traced/untraced), which corrects the earlier
claim that these runs were "depressed by per-step logging".

## v0.7.3 — Hopper 4-Mode Fused Kernel: Multicast Tested, Doc Split (2026-10-05)

### Docs: split by effort, with lineage
`tasks/debug_kernel_server_perf.md` had reached 3428 lines spanning two unrelated
efforts. Split:

| doc | covers |
|---|---|
| `debug_kernel_server_perf.md` (1783 lines, closed) | Parts 1-4: sync-free plan, measured mixed-kernel gate, workload stats, mixed-tile 4-mode *dispatch* |
| `debug_kernel_server_perf_4_mode.md` (new, active) | the fused FA3 + FA2 SM pre-assignment **kernel**: correctness, server integration, split/cluster-cap tuning, backfill, multicast |

Each carries a lineage header pointing at the other, and the old file ends with a
forward pointer. Corrections to earlier claims are kept in place in both rather
than edited away — several conclusions were reversed by later measurement and the
reversals are the useful part.

### Measured: multicast tested on hardware (flashinfer `4eb41c6`, `88fc805`)
Multicast had been written off as needing a `sparse_mainloop.cuh` gather→TMA
rewrite. That conflated multicast with *TMA* multicast: Hopper DSMEM gives the same
load-once-consume-in-N-CTAs property with no tensormap and no KV-pool change, via
`cg::cluster_group::map_shared_rank` (already used in-tree by
`trtllm/fused_moe/RoutingKernel.cuh`).

`tests/dsmem_kv_share.cu`, built to match the redundancy this kernel has — CTAs
paired on the same KV tile, since every q-tile of a prefill row re-gathers the same
KV:

| working set | redundant gather | DSMEM shared | speedup |
|---|---|---|---|
| 16 MB (**L2-resident, in-situ**) | 682.436 us | 514.583 us | **1.326x** |
| 256 MB (cold) | 1325.362 us | 835.079 us | 1.587x |

L2 halves the baseline but costs DSMEM only 1.587x → 1.326x, because a
`page_size=1` gather is bound by *transaction count*, not by which level answers
it. L2 shortens a repeat; DSMEM removes it.

### Added: real cluster launch, opt-in (flashinfer `610d8d7`, `986270d`)
`FLASHINFER_FUSED_CLUSTER=N` switches the fused dispatch to `cudaLaunchKernelExC`
with a real `clusterDim`, default 1 so the shipped path is byte-identical. It
refuses rather than reshapes when N does not divide the SM count, since rounding
the grid would break the CTA count FA3's contract checks.

Isolation matrix via `FLASHINFER_FUSED_SKIP_FA3=1`:

```
CLUSTER=1 SKIP_FA3=0   rc=0  PASS
CLUSTER=2 SKIP_FA3=0   rc=1  CUDA error: unspecified launch failure
CLUSTER=1 SKIP_FA3=1   rc=1  424/512 unwritten (expected), no CUDA error
CLUSTER=2 SKIP_FA3=1   rc=1  424/512 unwritten (expected), no CUDA error
```

**The FA2 side is cluster-safe; FA3 alone is cluster-hostile.** This overturns an
earlier conclusion in this work that the single-grid two-role design forecloses
per-role cluster features — the side that wants DSMEM is already cluster-ready, and
the remaining work is making FA3 tolerate a cluster it does not use.

### Resource correction
Two H20 trials had been allocated for the PD-colocation experiments before the
single-GPU kernel work finished, and both sat idle for the session:
`81d861acb7b7c964` (4 nodes x 8 = **32 GPUs**, over the 16-GPU cap) and
`a1edeb8114365af4` (8 x 2 = 16 GPUs, within cap). Both now `STOPPED`. The
multi-node work is gated behind the kernel work by the goal's own ordering, so
allocating ahead of it was wrong regardless of the cap.

### Still open
4-mode MTP **serving** schedule unreachable (`forward_batch_speculative_generation`
would need reordering against the scheduler), so every server number remains
spec-decoding-off. Multicast measured but not integrated. Best server
configuration is the confirmed +0.7-0.8% win from v0.7.2.

## v0.7.2 — The Fused Kernel Reaches the Serving Path: Three Bugs, and a Measured 8.1% Regression (2026-10-04)

### Fixed: the page-index array handed to the fused kernel was wrong
The previous session's planning optimisation reused `wrappers[0]._kv_indices` on
the assumption that the mixed wrapper, planned over the whole batch moments
earlier, already held exactly the page-id array the launch needs. It does not.

Found by dumping the launch arguments to disk immediately before the faulting
launch (`SGLANG_FUSED_DUMP=<path>`, written at the launch site so it survives the
abort) rather than by a third hypothesis. On the faulting step:

```
qo = [424, 87, 1]   prefix = [3577, 0, 13]   ->  kv = [4001, 87, 14]
kv_indices: 3846 entries, batch needs 4102
            256 entries outside the valid page range [0, 1816955]
            first invalid at index 3590, value 1085595437
q, o, kv_indices, fa3_sparse base pointers: all 16-byte aligned
```

So nothing was misaligned. The kernel was gathering garbage page ids — up to
1.1e9 against a 1.8e6 page pool — from an array both too short and partly
uninitialised, and faulted on the resulting address. The optimisation it replaced
was worth ~0.3% of throughput.

- `_fused_kv_indices` now builds the array vectorised on the GPU (one
  `repeat_interleave` for each output slot's row, one subtract for its offset,
  one gather from `req_to_token`), so correctness does not cost the planning time
  back, and no per-request Python loop returns.
- It is checked against the plan's own page total before launch. The absence of
  that check is why a wrong buffer reached the GPU as a fault rather than a
  declined plan.

### Fixed: the tile-128 task was never executed (flashinfer `b43e50b`)
The fused extension built persistent params for tasks 1 and 2 only, on the
comment "task 0 went to FA3". The persistent plan is built over the FA2
sub-batch *after* the FA3 rows are removed, so its task 0 is ordinary FA2 work:
`TwoStageHolisticPlan` sends every row with `packed_qo_len > 64` there, and folds
the whole tile-64 mid class in too whenever that class carries less than
`FLASHINFER_TRI_TILE_MIN_KV` total KV. Nothing executed it.

No harness could have caught it: they all use MTP decode rows for the FA2 side,
whose `packed_qo = 2 * gqa = 14` lands in task 2, so task 0 was always empty. In
the dumped batch row 1 (`packed_qo = 609`) landed in task 0 and was skipped.

Adds `R128 = PKT<128, 2>` threaded through `fused_drain_queue`,
`FusedFA3PersistentKernel` and `FusedFA3PersistentDispatched`. Shared memory is
unchanged — tile 128 needs 49152 bytes against tile 16's 69632, so `Runner3`
still sets the bound.

### Measured (H200, flashinfer `b43e50b`, sglang this commit)
`tests/test_fused_server_batch.py`, which replays the dumped batch shape verbatim
so one launch covers FA3, task 0 and task 2:

| kernel | unwritten rows | FA3 row | task-0 row | task-16 row |
|---|---|---|---|---|
| pre-fix | **87 / 512** at every split | 0.00029 | **nan** | 0.00293 |
| fixed | 0 / 512 | 0.00023 | 0.00196 | 0.00271 |

87 is exactly row 1's token count — the task-0 row. Relative error is against an
fp32 reference, at `sm_split` 48 / 32 / 16.

Command: `FI_ROOT=/dev/shm/w/fi python3 tests/test_fused_server_batch.py`

### Ruled out
`tests/align_fused.py` reproduces each server-side tensor provenance in the
standalone harness — `q` as a non-contiguous view of a fused QKV buffer, `o` as a
slice of a reused buffer, the index array built by concatenating `req_to_token`
rows. All pass, so provenance was never the cause, contrary to the hypothesis
recorded in the previous session.

### Fixed: the KV cache does not have a fixed rank (flashinfer `4d59914`)
The extension read `k_cache.stride(1)` as the token stride and `stride(2)` as the
head stride. The standalone harness slices a `[pages, 2, page_size, H, D]` buffer
and so passes **rank 4**; sglang's pool passes **rank 3** `[tokens, H, D]`. On the
dumped batch that gave `k_stride_h = 1`, so all four KV heads aliased one head's
memory and `get_gmem_layout` built a degenerate layout whose tensor map made
`cudaLaunchKernel` **itself** fail with "misaligned address". The failed launch
poisoned the CUDA context, so the next unrelated attention call failed too and
killed the server — which is what earlier runs mistook for a fault inside
flashinfer's own persistent kernel.

The page step survived both layouts by accident (`ratio * k_stride_n` is
2*512 for rank 4 and 4*128 for rank 3, both correct), which is why only the head
stride was wrong and the symptom was a launch rejection rather than wrong numbers.

`kv_strides()` now derives `(page, n, h)` from the rank and rejects any other
rank; `_fa3_sparse_indices` picks the matching vector-sparse ratio.

Ruled out by measurement, not argument: tensor provenance (`tests/align_fused.py`)
and the CUDA 12.8 vs 13.1 toolchain split (`tests/test_fused_server_batch.py`
passes under the server's own venv python).

### Measured: the fused kernel runs server-side, and costs 8.1%
First valid server-side A/B. `--random-input-len 8192 --random-output-len 192
--num-prompts 300 --request-rate 12 --max-concurrency 32`, qwen7b on H200,
CUDA 12.8 / torch 2.8.0, sglang this commit + flashinfer `4d59914`:

| metric | fused | base | delta |
|---|---|---|---|
| total token throughput (tok/s) | 23250.44 | **25291.44** | **-8.1%** |
| mean TTFT (ms) | 3443.97 | 3150.61 | +9.3% |
| mean ITL (ms) | 20.18 | 18.54 | +8.8% |
| p99 ITL (ms) | 31.87 | 31.62 | +0.8% |
| `mixed_fused_fa3` share | 29540/32400 (**91.2%**) | absent | — |

300/300 requests succeeded in both arms, 0 misaligned, 0 launch failures. The
direction agrees with the kernel-level result (0.410x the sequential co-launch,
0.709x at the best split on H200), so the remaining deficit is the structural one,
not wiring.

An earlier A/B at `--random-input-len 2048` showed a null result (18182 vs 18203
tok/s) and was **invalid**: at that length no row clears the FA3 gate's
`kv_tiles >= 32` (>= 4096 KV) requirement, so `mixed_fused_fa3` was absent from
both arms and they ran the same kernel. Every fused measurement must carry its
`mixed_fused_fa3` share or it is not evidence about this kernel.

### Optimized: sweeping the SM split recovers more than half the regression
`sm_fa3` was pinned at 48 on every serving run by `plan_dispatch_prefix`'s
`MAX_SM_FA3`, so the kernel's main knob had never been varied server-side.
`SGLANG_FUSED_SM_FA3` now pins it. Matched sweep, 200 prompts, input 8192, H200:

| `sm_fa3` | tok/s | vs base | mean ITL | p99 ITL |
|---|---|---|---|---|
| 16 | 19369.38 | -22.4% | 23.77 | 45.46 |
| 32 | 22034.74 | -11.7% | 20.93 | 35.94 |
| **64** | **24055.80** | **-3.6%** | 19.18 | 34.74 |
| 80 | 23800.46 | -4.6% | 19.38 | 35.72 |
| 96 | 22856.25 | -8.4% | 20.26 | 37.05 |
| 112 | 19457.78 | -22.0% | 23.95 | 44.89 |
| base (fused off) | 24946.41 | — | 18.47 | 32.85 |

200/200 requests in every arm, 0 faults, fused share ~90.6% throughout. The curve
is sharply peaked at 64, and the old ceiling of 48 sat below it: **-8.1% -> -3.6%**.
`MAX_SM_FA3` is now 64 so the default lands on the measured optimum.

**The analytic balance point is not usable for this knob.** Unclamped,
`plan_dispatch` returns 115-131 for real serving batch shapes — nearly the whole
device on FA3 — and the sweep measures 112 as one of the two *worst* settings.
The clamp, not the model, was choosing the value. That cost model was fit for the
overlap kernel and does not transfer to this kernel's makespan.

### Tested: cross-role backfill on the server
`FLASHINFER_FUSED_BACKFILL=1` at `sm_fa3=64`: 24064.65 vs 24055.80 tok/s,
**+0.04% — within noise**, against 1.2% (H200) / 1.9-6.9% (H20) at kernel level.
Consistent rather than contradictory: backfill reclaims the prefill tail of one
mixed step, and at 32 concurrent requests the server already has other work to
overlap it with.

Correctness with backfill on is now clean under the server's own toolchain on the
dumped server batch shape, at splits 64/48/32: 0/512 unwritten, rel err
0.00028/0.00218/0.00282 (vs 0.00032/0.00245/0.00313 with it off). Scope: that is
one shape, and it is not the one the earlier backfill fault was recorded on, so it
does not retire that fault. The default stays off, now for a measured reason.

### Multicast: the precise blocker
Not a flag. The fused kernel's FA3 side uses `hopper/sparse_mainloop.cuh`, where
`USE_TMA_LOAD_KV = false` (line 87) — KV is gathered with `cp.async`, so there is
no TMA descriptor to multicast from — and the Q descriptor is built with `_1{}`
and the comment "no mcast for Q". Enabling it needs `page_size >= CTA_TILE_KV`
(128) threaded through the fused launch (currently hardcoded to 1) **and** the
sparse mainloop's gather replaced by a TMA load. The second is a rewrite, not
configuration.

### Still open
Best fused configuration is -3.6% against the persistent kernel, so
`SGLANG_FUSED_FA3_SM_SPLIT` stays 0 by default. Closing the last 3.6% needs
warpgroup-reentrant FA2; the split is now tuned and backfill is measured at zero.

## v0.7.1 — Split-KV and GQA Packing for FA3; Gate on Tile-Grid Quantities (2026-09-30)

### Measured
- **FA3 split-KV: 1.75x over FA3 as-is and 1.38x over the persistent kernel**
  (507 -> 290 us, S=2, one 1024x32768 row at 8 q heads). Width, not KV traffic,
  was FA3's problem, and it is a property of flashinfer's sm90 planner having no
  KV split rather than of wgmma. Emulated via batch expansion + `merge_states`.
- **GQA packing: 5.0x on the verify class** (2541 -> 511 us, 58 rows @ kv 11000),
  moving it from 0.23x to 1.13x against the persistent kernel at packed_qo 56;
  still 0.78x at packed_qo 32. Nothing on large prefill (509.8 vs 507.2 us).
- **CUDA graphs: the fused path is strictly better.** At fixed tokens and batch
  size, one captured fused graph replays correctly across p/d ratios 1/32 -> 8/25,
  while FA3's `batch_size`/`qo_indptr` length change with the ratio and are baked
  into the captured launch. Sequential needs a capture per (n_prefill, n_decode)
  pair. This reverses the earlier parity claim.

### Rewritten: mixed_dispatch gate
- Routes on the two tile-grid quantities the measurements identify:
  `packed_qo_len` vs `MIN_PACKED_QO = 48` (tile occupancy) and `kv_tiles` vs
  `MIN_KV_TILES = 32`. Work is counted in (q tile x kv tile) pairs.
- The two kernels get **disjoint SM sets** at the makespan balance point; FA3's
  KV-split S is the smallest that covers the SMs it was given (returns S=2 for
  the (8,1,128) chunk, the measured optimum).
- A class folds into the persistent kernel unless one SM's worth of it is at
  least half the makespan the rest of the device needs anyway -- no separate SM
  set that cannot be kept busy.
- Latency 2.15 us on decode-only batches (short-circuits on `max(qo)`) and
  13.8-14.7 us when an FA3 row is present, once per forward.

### Corrections
- The gqa-7 verify figure was first reported as 2.72x; it packed one GQA group
  while the persistent kernel processed all 4, so FA3 did a quarter of the work.
  Matched-work value is 1.13x.

## v0.7.0 — 4-Mode Dispatch: Separate Kernels, Scheduler-Partitioned Grids (2026-09-30)

### New: per-tile-class kernel dispatch with a grid-partitioning scheduler
- `python/sglang/srt/layers/attention/mixed_dispatch.py` (new): picks per batch
  between `fused` (one persistent launch), `seq` (FA3 then persistent) and `conc`
  (both co-launched at a split grid), and returns the grid split in SMs.
- flashinfer (`tri_tile_`): `FLASHINFER_FA3_MAX_CTAS` caps `PrefillSM90Plan`'s CTA
  count and `FLASHINFER_PERSISTENT_MAX_CLUSTERS` caps the holistic planner's grid,
  each rebalancing its own work over what it owns. Only the grid shrinks, so graph
  capture still sees one shape per bucket — green contexts were rejected because
  captures would multiply by the number of groups.
- `plan_4mode_dispatch` in `flashinfer_backend.py` now delegates to the measured
  scheduler; the unsupported `DISPATCH_MIN_MONOLITHIC_US` guard and the fixed
  `DISPATCH_CLASS_KERNEL` table are removed (both are contradicted by measurement).

### Results (held out on captured sets; see tasks/debug_kernel_server_perf.md)
- vs sequential (vLLM-style): **1.015x** overall, **1.054x** (h200 TP8) and
  **1.147x** (gb200 TP16); parity on the small-model long-prefix set, where the
  scheduler correctly picks `seq` and becomes the baseline.
- vs a fixed fused megakernel: **1.383x**. vs dual-stream: **1.000x** (no gain).
- Graph compatibility is **not** better: only 27 of 41 shape buckets resolve to a
  single mode, so parity (41 graphs) costs most of the gain and the full gain needs
  55 captures.

### Key measurements
- FA3's wgmma mainloop is **1.86x** the persistent kernel's per-CTA compute rate,
  but it reloads KV once per q head and its width is `h_qo * ceil(qo_len/128)` —
  only 64 CTAs at the (8,1) TP8 shard. At 8 q heads per rank the persistent kernel
  therefore **beats FA3 on pure prefill** (401us vs 507us at 256 KV tiles).
- Co-launch does overlap (up to **1.65x** at a 64/68 split), but over-provisioning
  the cooperative kernel stalls it behind FA3 (2286us at 88/44 vs 1470us).
- Boundary fitted on 50 cells, **50/50** correctly classified; within 5% of the
  per-batch oracle on 26/28 held-out batches.

### Doc hygiene
- `tasks/archive_h100_era.md` (new): 581 lines of H100/Qwen3-8B results, closed
  investigations and v3-scheduler notes moved out of `debug_kernel_server_perf.md`,
  which is reorganised into four story parts (gate -> workload -> kernels ->
  dispatch) and shrinks 2,184 -> ~1,540 lines. The superseded 4-mode *estimate*
  section collapses from 125 to 27 lines.
- Corrected the FA2/FA3/persistent planner table: it had given FA3
  `cta_tile_q = 64` and a `total_kv_lens/num_clusters` KV split, citing lines that
  are actually inside `TwoStageHolisticPlan` (our planner). FA3 is
  `PrefillSM90Plan`: `cta_tile_q = 128`, **no KV split**, no group-head fusion.

### Fixes
- Routing tested KV length alone and sent 58 spec-verify rows to FA3, **14x slower**:
  FA3 does not fuse group heads, so the test must be on `qo_len`, not `packed_qo_len`.
- Split accounting mixed units — the persistent planner sets `num_sm *= 2`, so a
  "96/36 split" subscribed 114 of 132 SMs. This invalidated four earlier `conc` wins.
- Out-of-bounds write in the per-task SM apportionment's remainder redistribution
  (`task_hi` could exceed `num_clusters`).

## v0.6.0 — Real-Workload Capture, Hopper Scheduling Study, Gate v3 (2026-09-19 → 09-22)

### Workload capture (new)
- `SGLANG_LOG_MIXED_COMPOSITION=<path>` records every mixed batch's real composition
  from the gate: per-row decode KV, chunk tokens, the **true attended prefix** (the
  scheduler's `#cached-token` is 0 for continued chunks), and the gate's decision.
  Wired into `run_mixed_workloads.py`, plus four `loadsweep_{hi,med,lo,min}` scenarios.
- Captured 3,493 mixed batches across a high→low co-tenancy load sweep. The realized
  shape: chunk size barely moves with load (p50 395–446) because it is the leftover
  token budget, while decode rows go 46 → 2 and the gate's persistent share 77% → 23%.
  Prefixes are fresh 73–96% of the time.
- **The sweep grid was badly mis-sampled**: 91.4% of real chunks fall in 128–512 tokens,
  a bin the grid covers with one point, while five of its seven chunk values sit where
  7% of traffic lands.

### Corrections (2026-09-23)
- The captured ~400-token "chunks" are **not chunks**: the distribution is bimodal and the
  small mode is the chat tenant's whole prompt (uniform[256,512], prefilled in one shot
  since 512 ≪ chunked_prefill_size=8192). It is load-invariant because it is a scenario
  parameter, not a leftover token budget; EDF was off (`fcfs`). The gate-v3 gain is
  therefore established for this workload's prompt-length mix, not in general.
- **Scope limit**: a decode row's `packed_qo_len` is `1 × gqa_ratio`, and the dual-tile
  premise only exists at ratio ≤ 16. At ratio 64–128 (MLA: Kimi-K2, DeepSeek-V3/V4)
  decode rows already sit in the large tile class — no heterogeneity to exploit and no
  16→128 flip for a chunk to cause — and those models use separate MLA backends that
  never call `select_mixed_attention_kernel`. The win region is low-GQA-ratio models
  (Llama/Qwen dense, 7–16) on the standard flashinfer backend.

### Kernel replay on real compositions (`sweep_from_capture.py`)
- Time-weighted over the captured distribution: FA2 1.00×, **FA3 0.67×**, persistent
  **2.60×**, deployed v2 gate 1.90×. FA3 loses to FA2 at every decode-row count
  (0.28–0.84×) — its fixed `cta_tile_q = 64` pads a GQA-7 decode row ninefold, which
  costs more than its better planner saves. Porting to FA3 is not the Hopper win.
- Gate decline diagnosis over 3,520 batches: 55.1% accepted, **27.0% declined by the
  out-of-support guard**, 17.9% non-unanimous — the grid mismatch, costing 14.2% of
  total FA2 kernel time.

### Gate v3 (deployed)
- Rebuilt `mixed_kernel_gate_h800_gqa7.json` with 60 measured serving compositions
  folded into the grid (1,376 points). On **held-out** buckets: accept 63.1% → 98.1%,
  speedup vs FA2 **2.17× → 3.50×**, 99.6% of oracle. A 1.61× kernel-policy gain with
  no kernel change.
- Serving validation (single run, `loadsweep_med`): live gate acceptance 34.1% → 89.3%,
  100% retokenization, **node throughput +0.0%** — the kernel-policy gain does not
  translate into node throughput. Chat tail improves (mean ITL −10.3%, P99 −43.1%);
  codex P99 +7.2% / TTFT +8.3%, inside the documented ±6% saturation run-to-run band,
  so neither direction is established without a paired repeat.
- Doing this at k=5 would have cost the zero-regression property (worst-case LOO
  0.971×); **k=7 restores worst-case LOO to exactly 1.000×** with no loss of held-out
  gain. `build_mixed_kernel_gate_table.py` gains `--k` / `--label`.

### Hopper instruction-level study (`tasks/debug_kernel_server_perf/wgmma/`)
- Five microbenchmarks, marginal cycles, validated at 96% of H800 peak bf16:
  wgmma shape mixing does **not** overlap within a warpgroup (32.00 + 20.00 = 52.00,
  order irrelevant) **or across warpgroups** on one SM (52.10) — the tensor core is a
  serial per-SM resource. exp2 does **not** hide under an in-flight wgmma (+50.5 cyc/8
  ops), but **hides 73% of the MMA when placed on a different warpgroup** (FA3's
  ping-pong premise, quantified). Small tiles absorb 8 FFMA ops at **zero** cost where
  large tiles pay 3.5 — cost is ≈ `16 + N/4`, so `n16` runs at 38% of peak.
- Closes one β hypothesis: the non-overlappable 0.53 is not the MMA pipe.
- Two measurement traps documented: ptxas C7515 (non-wgmma writes to accumulators)
  silently serializes the pipeline, and any runtime branch in the issue loop inflates
  the baseline 3× — both produced false "no overlap" readings in earlier drafts.

## v0.5.1 — Task-doc cleanup: invalidated results removed, story-order restructure (2026-08-13)

- `tasks/debug_kernel_server_perf.md`: deleted the sections marked superseded/
  invalidated (2026-07-24 serving results; Historical H800 WL1–6/EDF tables that
  compared contaminated DMC baselines). Reordered: current story (sync-free plan +
  gate + cost model + conclusion) and Open TODOs first; honest H100-era kernel/server
  results and closed investigations behind them as labeled history. Open TODOs
  rebuilt: EDF-analytic A/B marked done (multiturn_slo_clean), remaining suite
  scenarios added.
- `tasks/realistic_workload_analysis.md`: removed the superseded headline table,
  contaminated offered-load sweep, and pre-fix sweet-spot table; rewrote Scope
  (gate + LPT_EDF methods) and Final conclusions to the current story; added an
  evaluation-hygiene section (baseline contamination, warmup, queue-vs-service TTFT
  decomposition, rate=inf confound, episode defense) and mixed-workload repro
  commands.
- `tasks/edf_scheduling.md`: validity note (pre-fix DMC arms; policy conclusions
  stand, kernel deltas do not); task #117 closed by the measured gate.
- `tasks/server_perf_model.md`: status note — analytic model supersedes the MLP for
  chunk control; MLP remains for the router `/get_load` signal only.
- `tasks/README.md`: updated to the current document map.
- External-validity subsection for the gate sweeps (are isolated-kernel benchmarks
  representative of peak serving?): measured from the mixed-tenancy captures that exactly
  one kernel is resident 93.25% of serving wall (≥2 only 0.36%), so the bench's
  one-kernel-alone condition is the serving condition; serving kernel times (BP
  0.007–16.6 ms, persistent 0.055–2.26 ms) sit inside the swept envelope (BP 0.019–28.6 ms,
  PA 0.019–25.7 ms, decode KV 2K–6.3M); and BP's p90 kernel time is 0.267 ms in the gated
  arm vs 0.906 ms in DMC, i.e. the predicted routing is visible end to end. Documents the
  gaps too — contiguous `arange` KV pages vs a fragmented radix allocator, sustained-load
  clocks, flushed vs dirty L2, graph replay, shared GPUs.
- New WGMMA microbenchmark (`tasks/debug_kernel_server_perf/wgmma/`) answering whether
  different tile sizes overlap or get reordered: they do not. Marginal cost is exactly
  additive (m64n64k16 32.00 + m64n16k16 20.00 = 52.00 cycles/iter), issue order is
  irrelevant, and two warpgroups on one SM give 52.10 — the tensor-core pipe is a serial
  per-SM resource. Validated at 96% of H800 peak bf16 for the large tile. Closes one
  hypothesis for the β TODO (overlap cannot come from MMA-with-MMA, only MMA-with-memory)
  and quantifies a 2.5× per-unit-work penalty for small query tiles (cost ≈ 16 + N/4
  cycles, so n16 runs at 38% of peak).
- Mechanism arithmetic re-verified against flashinfer source, fixing one error: the
  persistent planner's split was documented as `kv_len_limit ~ total/264`, but 264 is
  fa2's `max_grid_size` (2 × 132 SMs) — the persistent side computes
  `f(total_kv_lens × num_kv_heads / num_clusters)` with `num_clusters = num_sm = 132`,
  i.e. ≈ total/33 here, halved again for the prefill tile class. The two BatchPrefill
  numbers turn out exact rather than approximate for this config: the split budget is
  exactly 66 units/KV head (`max_grid_size / num_kv_heads` = 264/4) and a 1,024-token
  chunk's q-tiles are exactly 56 of them (`ceil(1024 × 7 / 128)`), so splitting the eight
  decode rows two ways (+16) would exceed the budget and the search stops splitting
  entirely. Also notes that the tile decision keys on the batch *mean* only in the
  non-graph path (what the sweeps use) and on `max_qo_len` under CUDA graphs, and flags
  that the 0.366 ms decode-only datapoint is not reproducible from saved artifacts.
- **Corrected an overstatement**: "BatchPrefill's cost surface has cliffs no smooth model
  can track" was too strong. New `fit_learned_cost_model.py` fits MLP and GBDT to both
  kernels on the same 1,316 sweep compositions, features and model class (5-fold CV × 3
  seeds): an MLP reaches 9.9% MAPE on BatchPrefill, beating the 13.3% of the analytic
  parts-based fit, so part of that gap was the functional form. What survives is the
  relative claim — BP is 1.7–3× harder than the persistent kernel (GBDT 10.1% vs 3.3%,
  worst case 103% vs 63%) — plus two practical points: adding fa2's own tile-decision
  variables recovers only 10.1%→8.6%, consistent with the residual living in the
  split-budget search over the per-row KV vector; and fitting BP needs ~1,300 measured
  compositions per GPU versus 35 for the analytic law, reintroducing exactly the
  profiling campaign it removes. Docstring in `overlap_cost_model.py` updated to match.
- Recorded why the kernel layer exists when only `predict()` is consumed: it is the
  interior of the one model in use, not a second model, and it is what makes the model
  training-free (35 kernel measurements vs the replaced MLP's 25,444 profiled configs),
  what makes the fused-vs-split predictability contrast visible at all (the step-level
  `c0 + c1·tokens` term masks it), and what an overlap-aware chunk controller would need.
  That controller is now an Open TODO rather than only prose.
- Clarified that the analytic cost model is **kernel-level** (its per-layer attention law
  is fit on 35 kernel measurements: T_d 2.04% MAPE, T_p 3.47%, composition law 3.10%
  in-sample and newly 3.22%/9.7% leave-one-out, so the one-parameter law is not overfit);
  only the `c0 + c1·tokens` step wrapper is server-level, and those two constants are
  derived from hardware specs rather than fit, with a single serving anchor. Records the
  two scope limits: persistent kernel only (no BatchPrefill equivalent by design), and a
  fit grid that sits above the realized operating point of the SLO runs it drives.
- Batch-mixture representativeness measured from the gated runs' scheduler logs and
  recorded as a known weakness: the sweeps are a Cartesian grid, but realized mixed
  batches form a narrow correlated cluster per workload (win scenario: 386-token chunks
  p90 532 against 46 decode rows @189K KV; TraceLab: 8,182-token chunks against 10 rows
  @904K). Chunk size and decode count are anti-correlated in serving because the chunk is
  the leftover budget, so the product grid spends points on combinations the scheduler
  cannot emit, and samples the actual operating region with two points. Notes that
  leave-one-out 1.000 is therefore generalization within the grid, not to serving.
  New TODO: rebuild the table by importance-sampling logged compositions.
- Removed the "DMC baselines were silently persistent" post-mortem now that every result
  it invalidated is gone from the docs; its one live invariant (baseline runners must set
  `SGLANG_FLASHINFER_USE_PERSISTENT=0` — it defaults on) is promoted to Setup as a rule.
- Cost-surface and win-region sections unified into one: adds a sweep-provenance table
  (479 uniform + 477 fresh → 956 merged, + 360 blended = the deployed 1,316 points) with
  every file checked in, and shows where the headline multiple comes from — at a fixed
  fresh 2K chunk, BatchPrefill is flat (2.97 → 5.05 ms) while the decode pool grows 64×
  and the persistent kernel scales with the work (0.105 → 4.274 ms), so the speedup runs
  28.2× → 1.2×. Corrects "up to 20×" to the per-sweep maxima (21.0× uniform, 28.2×
  merged) and records the loss region from the same data (162/956 below 1.0×, worst
  0.57× at 1-4 tiny rows). New data files: `gate_sweep_blend_qwen25_h800.json` (was only
  in an ephemeral scratchpad, though it is 360 of the deployed table's points) and
  `probe_fresh_qwen25_h800.json`.
- Profiling post-mortem added to the overhead diagnosis, mined from the mixed-tenancy
  nsys captures. It leads with the conclusion — **our plan path does no D2H at all**
  (host tensors only; verified by `cudaDeviceSynchronize` appearing once in 279 s, so
  neither flashinfer's device-tensor branch nor the `seq_lens_cpu is None` fallback ever
  fires) — and presents the rest as the mechanism behind the old 56.5 ms, measured on
  the copies that remain elsewhere in the stack. Findings: `non_blocking=True` is not
  honored for D2H into pageable memory, and the
  cost is a pipeline drain rather than a copy. Within back-to-back groups of pageable
  D2H copies the **first** costs 5–8 ms (median) and the rest ~14 µs — 560–800× apart
  at identical size and destination — so the old plan() stall was paid by the first
  `.to("cpu")`, not by the following `torch.cuda.synchronize()`. Steady-state window
  analysis shows the host blocked 74.6% of wall in both arms while the GPU stays ~94%
  busy (GPU-idle-while-host-blocked 0.64% of wall), so such readbacks hurt only when
  they remove run-ahead the GPU needs. Two live optimizations filed: flashinfer's stock
  `plan()` still does pageable D2H for the BatchPrefill path (baseline pays it every
  mixed step), and sglang's `fast_decode_plan` remains shadowed by the class-level
  `begin_forward = plan` alias.

## v0.5.0 — Mixed-Tenancy Win, Overlap Attribution, Analytic Cost Model (2026-08-04 → 2026-08-06)

### Mixed-tenancy workloads (`run_mixed_workloads.py`)
- New multi-tenant suite (coding agent + chat on one node, plus long-CoT / RAG / control
  scenarios). `agent_plus_shortchat` vs honest BatchPrefill baseline: coding tenant
  **+36.5% throughput**, chat tenant **−42% mean ITL / −23% P99 / −37% TTFT**;
  `shortchat_only` control at parity with the gate declining persistent (6/2,566).
- One −19% LPT trial archived as `run1_regressed/` (storm-phase retraction-cascade
  signature); unreproduced across repeat + nsys trials with a stable baseline.

### Gate v2 (KV concentration + out-of-support guard)
- Root cause of the win region: few long-context decode rows starve BatchPrefill of
  query tiles (8×64K = 8.2x vs 128×4K = 1.02x at equal aggregate KV); one chunk flips
  its CTA tile and exhausts the KV-split budget (3.125 ms plateau for any chunk ≥1K).
- Two k-NN blind spots fixed (fresh-prefix chunks; blended pools) via 837 new sampled
  compositions and a KV-concentration feature (Σkv²/Σkv). Merged table 1,316 points,
  k=5 unanimous, label ≥1.10: leave-one-out worst regression 1.000, capture 81%.

### Overlap attribution + analytic cost model
- Same-kernel ablation: temporal PD overlap contributes up to **1.26×** in the balanced
  band (T_p ≈ T_d); heterogeneity-robust scheduling supplies the rest.
- `SGLANG_EDF_LATENCY_MODEL=analytic`: `T_attn = max + 0.53·min` (3.1% MAPE / 9% max,
  7 physical constants, no training) drops into the EDF chunk controller; BP's best
  parts-based fit is 13.3% / 43% with fitted β=2.19 (super-additive) — the fused kernel
  is what makes mixed-step cost predictable enough for SLO-aware chunking.

### Docs
- Superseded/contaminated 2026-07-24 serving tables flagged in place; conclusions and
  TODOs consolidated to current results.

## v0.4.0 — Sync-Free Persistent Plan + Per-Batch Mixed Kernel Gate (2026-07-30)

### Validation (TraceLab decode-heavy N=100 screens + WL1 Qwen3-8B, honest baselines)
- 10 sessions (old +14.5% pooled-P99 regression point): parity on every axis across paired
  runs at the matched cap; pooled P99 ITL −0.7% to −6.2% better than true DMC.
- 16 sessions: +0.8% mean throughput across paired runs, pooled P99 ITL −1.3% to −4.1%,
  request-max P99 up to −23%; cap 6656 option reaches −14.6% pooled P99 ITL / +0.77% tput.
- WL1 (Qwen3-8B, rates 8 and 2): parity or slightly better on all metrics; 100%
  retokenization on every accepted run.
- Startup prewarm of both persistent module variants added after a mid-replay JIT build
  froze all nine sessions for 18.3 s in the first g10 run (flashinfer `c6d2a00`,
  sglang `313b97611`).

### Sync-free BatchAttention planning (flashinfer `f4d2a80` + backend host metadata)
- `BatchAttention.plan()` no longer calls `torch.cuda.synchronize()`. The backend now
  builds `qo_indptr`/`kv_indptr`/`kv_len` on the host from `seq_lens_cpu`/`extend_prefix_lens_cpu`
  (NumPy cumsum) and passes CPU tensors straight to `plan()`; flashinfer double-buffers its
  pinned int-workspace staging with CUDA-event fencing so removing the sync is race-free.
  Measured: with ~40ms of queued GPU work, plan wall time drops 56.5ms → 0.47ms. This was the
  main serving overhead of the persistent kernel (it broke overlap-scheduler CPU run-ahead on
  every mixed step); previous naive removal attempts hung because of the pinned-buffer race.
- `persistent_attn_forward` reuses grown `out`/`lse` buffers instead of allocating two tensors
  per layer per step.
- Removed a dead `call_begin_forward` branch that would have routed BatchPrefill wrappers
  into a BatchAttention-signature `plan()` call when `use_persistent=1`.

### Per-batch mixed-attention kernel gate (scaffolding)
- New env `SGLANG_MIXED_ATTENTION_KERNEL` = `auto` (default) / `persistent` / `batch_prefill`.
  For each mixed batch, `select_mixed_attention_kernel()` chooses persistent BatchAttention or
  BatchPrefill from CPU-side batch composition; forward dispatch now keys on the planned
  wrapper type instead of the global env flag. Cost-model constants pending the H800 gate
  sweep (see tasks/debug_kernel_server_perf.md).

### DMC baselines were silently persistent
- `SGLANG_FLASHINFER_USE_PERSISTENT` defaults to 1 and no DMC launcher overrode it, so every
  H800 "DMC_CG" baseline since 2025-11-14 ran the persistent kernel for mixed batches. July
  TraceLab LPT-vs-DMC deltas compared chunk caps/EDF guards on the same kernel. Both
  `run_gain_search.sh` and `run_edf_sweep.sh` now set `SGLANG_FLASHINFER_USE_PERSISTENT=0`
  for baseline methods.

### H800 kernel measurements (serving-matched non-causal prefix composition, Qwen2.5-7B heads)
- Persistent kernel run() CPU dispatch is cheaper than BatchPrefill (16µs vs 26µs) —
  cooperative launch is not a CPU-side cost.
- 479-case sweep: persistent wins up to 20× at decode-dominant compositions but loses up to
  0.55× when decode work is tiny and the chunk is compute-bound; 87/186 raw losses are
  recovered by choosing the right flipped schedule (big chunks + long prefix want non-flipped).

## v0.3.3 — Nebius-SWE Workload + Chunk-Control Fall-back Fix (2026-04-09)

### Heavier Benchmark Workload
- Added `nebius-swe-traj` and `swe-bench-pro` datasets to `bench_serving_3rdparty.py` and the bench_serving CLI choices.
- `nebius-swe-traj` loads `nebius/SWE-agent-trajectories` (multi-turn SWE agent traces) via streaming, samples a random AI-turn boundary per trajectory. Median input ~9.7K tokens — the only prefill-heavy real-world dataset surveyed in `tasks/realistic_workload_analysis.md`. Code-Feedback (200-token median input) was too lightweight for chunk control to matter; nebius is the right stress workload.
- `swe-bench-pro` loads `ScaleAI/SWE-bench_Pro` test split — 731 single-step Q/A pairs, longer than swe-gym but still single-shot, kept available but not the headline workload.

### Chunk-Control Fall-back Fix (`edf_scheduler.py`)
- **Bug**: when any decode req's slack went negative (`min_slack_ms ≤ 0`), `edf_compute_max_chunk_size` fell back to the **full** `default_chunk_size` (8192). At saturation this triggered immediately, so chunk control was effectively disabled — first nebius EDF run gave P99 ITL = 297ms vs 297ms baseline (zero improvement).
- **Fix**: throttle to `MIN_CHUNK = 512` instead of falling back to default. Already-violated requests can't be saved by relaxing the chunk; the not-yet-violated ones can be, by capping added prefill latency to ~10-20ms.
- **Impact** on nebius-swe-traj 1k sweep: DMC/LPT_CG P99 ITL drops from ~290ms → ~135ms across all rates; per-req P99 ITL ≤ 150ms attainment goes from 0.6-22% to **85-100%**. Throughput rises 1212→1290 tok/s (faster KV drainage = higher effective concurrency).

### New SLO Metric (`analyze_results.py`)
- Added `itl_p99_attain_pct`: per-request P99 ITL ≤ ITL_SLO (Sarathi-Serve-style). Strict max-ITL gives ≈0% on long-decode workloads (any single outlier kills the score, masking real differences); per-req P99 filters single-step noise but still penalizes consistent prefill-blocking. TPOT (mean ITL) is too lenient (>95% on every run).
- Added `nebius_swe` to the per-workload SLO table (5s TTFT / 150ms ITL — same as SWE-Gym).

### New Sweep Scripts + Pareto Plot
- `tasks/edf_benchmark/run_nebius_baseline_1k.sh` and `run_nebius_edf_1k.sh`: 1000 prompts, rates 1/2/4/8/12/inf (lower than other workloads because 9K-token prefills are 30× heavier than ShareGPT), output fixed to 1024 tok via `--sharegpt-output-len`. Both scripts toggle `gpu_keepalive` per benchmark per project rules.
- `tasks/edf_benchmark/plot_pareto_nebius.py`: side-by-side baseline median ITL + EDF SLO attainment plot. Output: `tasks/edf_benchmark/pareto_nebius.png`.

### Findings
- Chunk control eliminates the historical PP advantage on decode tail latency on prefill-heavy workloads. DMC/LPT_CG now match PP at 91-100% vs 95-100% per-req P99 ITL attainment.
- TTFT remains the binding constraint above rate 2 — saturation queues are fundamental and no scheduling policy can hide them. EDF + chunk control protects the decode SLO at the cost of TTFT for late requests, which is the correct trade-off for interactive serving.

## v0.3.2 — 1k-prompt EDF Sweep + schedule_policy Admission Fix (2026-04-07)

### schedule_policy Bug Fix (overlap scheduler crash root cause)
- `add_chunked_req` and `add_one_req` could admit a chunked-prefill req with `extend_input_len ≤ 0` (full prefix hit on a freshly chunked req). In MIXED mode the sampler then produced N-1 sampled tokens for a batch of N, crashing the overlap scheduler with `RuntimeError: expanded size (N) must match existing size (N-1)`.
- Fix: when `extend_input_len ≤ 0`, drop one prefix block (`prefix_indices[:-1]`), recompute `extend_input_len`, and refuse admission if still ≤ 0. This is a real root-cause fix — no `--disable-overlap-schedule` workaround.
- Files: `python/sglang/srt/managers/schedule_policy.py` (`add_chunked_req`, `add_one_req`).

### 1000-Prompt EDF Sweep with 6 Pareto Curves
- Re-ran the full 3 methods × 3 workloads × 6 rates (8/16/24/32/48/inf) sweep at **1000 prompts** per workload, both baseline (FCFS) and EDF + SLO-aware chunk control.
- Baseline tables now report median TTFT and median ITL alongside P99 (per user request).
- Switched the EDF Pareto y-axis from TPOT (mean ITL ≤ SLO) to **strict max-ITL ≤ SLO**: TPOT stays > 99% across every run because median ITL is 25-30ms vs 100-150ms SLO, masking real differentiation. Strict max-ITL is what chunk control actually protects.
- 6 Pareto curves: `tasks/edf_benchmark/pareto_baseline_median_itl.png`, `tasks/edf_benchmark/pareto_edf_slo_attainment.png`.
- Key result: EDF + chunk control gives 2-10× tail-ITL SLO attainment improvement for DMC/LPT_CG vs PP at saturation on ShareGPT and SWE-Gym.
- Scripts: `tasks/edf_benchmark/run_baseline_1k.sh`, `run_edf_1k.sh`, `analyze_results.py`, `plot_pareto_6.py`.
- Results: `tasks/edf_benchmark/results_baseline_1k/`, `results_edf_1k/`.

### Project Rule
- Added to `CLAUDE.md`: **Never compromise by falling back silently** — if a feature crashes, find the root cause; don't disable it to work around the problem.

## v0.3.1 — EDF Benchmark Sweep + Chunk Control Fix (2026-04-07)

### EDF Chunk Control Bug Fix
- `edf_compute_max_chunk_size` was using `req.last_tic` as previous decode time, but `last_tic` is only updated when `enable_metrics=True`. With metrics off it equals request creation time, slack went negative, prefills starved (smoke test: 579 vs 2135 tok/s).
- Fix: track `_edf_last_step_time` per req, updated in `process_batch_result_decode`. New reqs fall back to `now` (full ITL_SLO budget).

### Full EDF Benchmark Sweep (3 methods × 3 workloads × 5 rates)
- Ran PP / DMC / LPT_CG with EDF priority + SLO-aware chunk control on ShareGPT, Code-Feedback, SWE-Gym at rates 4/8/12/16/24
- Per-workload SLOs: ShareGPT (2s/100ms), Code-Feedback (3s/150ms), SWE-Gym (5s/150ms)
- TTFT attainment is 100% across all 45 runs — ITL is the binding SLO
- PP dominates Code-Feedback (100% at rate ≥ 12); mixed batches (DMC/LPT_CG) win on ShareGPT at saturation
- Generated two Pareto curves: throughput vs ITL P99, and SLO attainment vs request rate
- Scripts: `tasks/edf_benchmark/run_edf_sweep.sh`, `analyze_results.py`, `plot_pareto.py`
- Results in `tasks/edf_benchmark/results_edf/`

## v0.3.0 — EDF Scheduling + SLO-Aware Chunk Control (2026-04-06)

### EDF Scheduling
- Implemented Earliest Deadline First scheduling: priority = TTFT deadline (arrival + SLO)
- TTFT SLO scales with input length (base + extra for >4K tokens)
- Enable via `SGLANG_EDF_SCHEDULING=1` + `--enable-priority-scheduling --schedule-low-priority-values-first`

### SLO-Aware Chunk Size Control
- Binary search for max prefill chunk size that fits within ITL SLO slack
- Uses trained MLP latency model to predict server step latency
- Configured via `SGLANG_EDF_LATENCY_MODEL=path/to/model.pt`
- Fixed clock mismatch: all EDF code uses `time.monotonic()` consistently

### Latency Prediction Models
- Retrained both LPT and DMC server-level latency models (25K+ configs each)
- LPT model: 2.4% MAPE, 3.26ms MAE (tasks/perf_model/lpt_model.pt)
- DMC model: 2.3% MAPE, 3.04ms MAE (tasks/perf_model/dmc_model.pt)

### Benchmark Status
- Created benchmark scripts for PP vs LPT_CG vs DMC with EDF on 3 workloads
- Blocked by pre-existing `req_to_token_pool memory leak` bug (available_size > total_size)

## v0.2.1 — CG25: data-driven mixed CG size optimization (2026-04-01)

### Mixed CG Size Optimization
- Instrumented mixed CG replay to collect (raw_tokens, padded_tokens) per workload
- Greedy selection of 6 additional capture sizes based on real padding waste: +3584 (28% of waste), +2816 (7%), +2304 (5%), +640, +1408, +1920
- CG25 (25 sizes, ~1.79GB) reduces padding from 11% to 6.5%, improves all 6 workloads +0.3-1.4% over CG19
- LPT+CG25 beats trueDMC on all workloads: +18.9% (WL1), +3.6-8.0% (WL2-4), +1.9-2.1% (WL5/6)
- Updated task MD: replaced v3 with LPT in tables, updated stale CG12-era results with CG25 data

## v0.2.0 — Mixed Batch CUDA Graphs + LPT + Server Benchmarks (2026-03-27 → 2026-03-29)

### Mixed Batch CUDA Graph Support
- Implemented mixed batch (prefill+decode) CUDA graph capture and replay for persistent kernel (BatchAttention)
- LPT (dynamic scheduler) compatible with CUDA graphs — activates at runtime when plan() detects mixed batch
- 6 bugs fixed: missing kv_indices copy, plan_info[-1] wrong field, module auto-switch breaking CG, padding overflow in replay, shared indptr buffer too small, LPT pointer initialization for CG

### Server Benchmarks (H100 80GB, Qwen3-8B)
- v3+LPT+CG beats prefill-prio by +41-87% throughput on long prefill (12k-16k input)
- HOL blocking crossover at ~8k input tokens for prefill-prio (max ITL 21-25s vs v3's 230ms)
- v3 matches default-mc within ±0.1% — plan() overhead offsets CG savings

### LPT Load-Balance Profiling
- Grid.sync() wait reduction: -15% to -39% max wait on KV-splitting workloads
- Kernel speedup: 7-16% on mixed workloads

### Overlap Profiler
- Fixed shape mismatch for persistent v3 compatibility (disable CG during profiling)

## v0.1.0 — Persistent Kernel v3 + Decode Fix (2026-03-26)

### Decode Overhead Fix
- Decode steps use regular BatchDecodeWithPagedKVCacheWrapper instead of BatchAttention
- Eliminated 0.5-4% throughput regression vs default-mc
