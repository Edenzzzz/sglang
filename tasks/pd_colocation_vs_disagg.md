# Current goal

Make the case for **PD colocation with an EDF/SLO-aware router** against **PD disaggregation**,
for a 4x8 H100/H200 cluster serving agentic traffic (TraceLab-shaped: 64-100K contexts, ~3.8K
appends, ~480-token outputs, mixed with chat and math).

Tasks, in priority order (status 2026-10-09; evidence in the sections below).

---

## [2026-10-12] SERVING EVIDENCE USED A CORRUPTED MODEL -- re-measurement in progress

Every serving run behind this doc passed the YaRN `--json-model-override-args`, which under
transformers 5.x reset Qwen2.5's `rope_theta` from 1e6 to 1e4 (sglang's override replaced the rope
dict). Fixed in sglang's config loader; details and bisection in
`debug_kernel_server_perf/resume_20261012/a2_serving_ropefix_20261012.md`.

What that does and does not touch, point by point:

1. **Cache-hit-first routing with an EDF/SLO veto** -- the policy argument is unaffected; any
   serving latency used to motivate it was measured on the corrupted model and must be re-run.
2. **Best-effort migration / replication** -- unaffected in logic (no serving numbers depend on
   the model's outputs).
3. **EDF predictor accuracy** -- the step model (`heldout_mape=21.9%`) and the EDF A/Bs were fitted
   and measured on the corrupted model. Step *latency* depends weakly on outputs, but acceptance
   length and request lengths do not, so the fit and its SLO results are re-measured before use.
   The 2026-10-11 conclusion that the fused kernel is harder to model (kernel harness, no model
   outputs involved) stands.
4. **Two-way RDMA / complete graph** -- unaffected in logic.

The kernel-harness numbers in the 2026-10-11 section below (fused vs `BatchAttention`) do not
involve the served model and are unaffected. The headline serving comparison was re-run on the
fixed model (`agent_chat_rated`, chat 24/s + TraceLab codex 0.1/s, one round, no graphs): the
**unfused** 4-mode schedule beats stock MTP by TTFT -32% and throughput +10%/+13% (mean ITL -13%),
and the co-compiled fused kernel lost 11-13% throughput against it. After moving FA2 out of the
co-compiled kernel (PDL co-launch, classes 64/32/16, auto FA2 chunk) the fused 4-mode arm beats
stock MTP by TTFT -55% / -58% and throughput +18% / +21% (chat / codex), mean ITL -19% / -11%, and
the unfused 4-mode arm by +7% / +6% throughput (2 rounds, no CUDA graphs)
(`debug_kernel_server_perf/resume_20261012/colaunch_20261012.md`). The EDF step model (point 3)
must be re-fitted on this kernel before its accuracy claim is tested.

---

## [2026-10-11] CORRECTED KERNEL EVIDENCE -- supersedes the 2026-10-10 section below

The 2026-10-10 update to this doc leaned on two kernel figures that are now **RETRACTED**, because
both compared against an arm that fails the causal fp32 reference (`MIX_ALL_FA3=1`, a diagnostic
configuration, failing 6/6 at 68-116 bad rows of 384):

* ~~"the fusion is 1.46-2.01x slower than all-FA3 on realistic batches"~~ -- the baseline was broken;
* ~~"3.14x slower than `BatchAttention`"~~ -- measured under the same knob, with a fence since shown
  to be a regression.

**The correctness-gated re-measurement is worse for the fusion, not better.** Two-role path
(`MIX_ALL_FA3=0`), backfill live, bad rows beside every timing:

| dec | sm_split | verdict | fused | `BatchAttention` | ratio |
|---|---|---|---|---|---|
| 8 | 96 | 0/264 | 0.3670 ms | 0.0699 ms | **5.3x slower (best case)** |
| 32 | 96 | 0/288 | 1.2560 ms | 0.1180 ms | 10.6x |
| 128 | 112 | 0/384 | 8.1775 ms | 0.2740 ms | **30.0x** |

**What this means for each of the four points, stated against corrected evidence:**

1. **Route to highest cache-hit node, EDF/SLO veto** -- **UNAFFECTED.** This is a router policy and
   does not depend on the fused kernel. The argument in section 1 stands on its own.
2. **Best-effort cache migration / replication** -- **UNAFFECTED**, same reason. Section 2 stands.
3. **"Our EDF predictor is more accurate because the fused kernel is easier to model"** --
   **THE PREMISE IS DEAD, and now for a better-evidenced reason than the 2026-10-10 entry gave.**
   That entry said the premise was inverted by a "23x `sm_split` cliff". The corrected measurement
   shows something simpler and more damaging: latency varies by **5.3x to 30x across `sm_split` and
   decode count on correct arms** (0.367 -> 8.18 ms). A kernel whose runtime moves by 22x across its
   own configuration space is *harder* to model than the unfused baseline, not easier. **Point 3
   must be dropped or re-argued without the kernel.**
4. **Two-way RDMA / complete graph** -- **UNAFFECTED.** A network-topology argument, independent of
   the attention kernel.

**So three of the four points survive intact and one is dead.** The surviving three are all
*scheduling and networking* arguments that never needed the fused kernel; the one that depended on
it is the one that fails. That is a cleaner statement of this doc's thesis than the kernel-first
framing: **the colocation case rests on the router, not on the attention kernel.**

**And the FA3-only alternative does not rescue point 3 either** -- it would run the all-FA3 path,
which fails 6/6 on the fp32 reference. Both kernel designs are currently evidence-blocked, so point
3 has no working kernel to stand on in either form.

---


**The thesis in one line:** our policy space (4-mode colocation + EDF/SLO router + tunable P/D
admission ratio) is a **strict superset** of every baseline's fixed policy, so we should match
the best baseline at any static workload and beat all of them when the workload moves.
Baselines are two-tier: **single node** = prefill-prioritized (the default) and mixed chunked
prefill; **multi node** = PD multiplex and PD disagg (1P3D, 2P2D).

**Item 5 carries that claim and is the weakest-evidenced part of the thesis:** nothing measured
so far varies the workload mixture, shifts it mid-run, or compares against any baseline on
goodput. Items 1-3 measure routing, cache replication and EDF ordering against 2P2D at a single
static mixture.

- [x] **1. Cache-hit-first routing with an SLO veto** (2026-10-08, H20 16 GPUs) — cache-affinity
      colocation vs the best disagg (2P2D): agent TTFT -93% (21.6 -> 1.6 s), chat TTFT -99.5%,
      prefix hit 73% vs 23-31%. Disagg keeps lower ITL (3.7 vs 6.4 ms). The calibrated veto fires on 2% of
      requests and trims chat P99 TTFT 15%.
      Hot-prefix stress (4 shared 16K prompts piled on their holders): a TTFT-only veto barely
      fires, because a cache holder's PREFILL stays fast while its DECODE degrades. Pricing the
      decode slowdown (`--decode-grid`) cuts chat ITL **-65%** and P99 **-80%** for a 25% chat TTFT
      cost (v0.18.0). Sub-items of this point:
      - [x] route to the highest-hit node first (affinity beats rr and disagg by 8-99x on TTFT)
      - [x] veto on predicted SLO violation, counting only EARLIER-deadline in-flight requests
            (`router.py feasible()`: it sums the prefill ahead of this request's deadline, and
            re-checks that admitting it does not push later-deadline requests past their own)
      - [x] the veto must price the term the overload appears in (TPOT, not just TTFT)
      - [x] load balancing: tie-break on hit bucket, then in-flight load, then random (a bare
            `min()` created a 2-worker hotspot; retracted and fixed, daaedef6a)
- [x] **2. Best-effort cache migration / replication** (2026-10-08) — a cluster-wide mooncake L3
      store (write-through, best-effort prefetch) lifts round-robin's hit rate 12.7% -> 22.3% (chat
      TTFT -41%, P99 ITL -83%). Under affinity it sits idle (16 pulls): affinity already serves 73%
      locally, so migration pays only where routing must place a request elsewhere.
- [x] **3. EDF with an EWMA acceptance length** (2026-10-08, H200) — measured 2.10 tokens/step. a=1
      collapses EDF; a per-request EWMA is biased low by the min over requests; a global EWMA is
      right, but every per-step chunk cap loses. **Deadline ORDER wins**: chat TTFT -62%, codex
      P99 ITL -27%. Full stack vs stock MTP: TTFT -95%/-51%, ITL -42%/-38% (chat/codex, 24/s, 3 rounds).
- [ ] **7. Inter-node routing: predict whether a request meets its TTFT and ITL SLO on node i, given
      i's estimated hit length x_i and current load** (added 2026-10-17 on the user's request; needs
      more design thinking before more code). This is the decision behind point 1's veto: when the
      highest-hit node is too hot, route to the highest-hit node predicted to still meet the SLO.
      **Current predictor** (`router.py` `predict()` / `feasible()`, policy `cache_slo`; router-side
      bookkeeping only, no worker query):
      `TTFT_i = (Σ prefill_left of in-flight requests on i with deadline <= this one + n - x_i) / prefill_tok_s`;
      `TPOT_i = grid(running_i + 1, mean ctx incl. this prompt) / acceptance EWMA_i`; feasible iff both
      meet the SLO and no later-deadline request queued on i is pushed past its own deadline.
      **Known errors, by sign:**
      - pessimistic: the global KV pool (HiCache + mooncake L3) pulls the missing prefix
        best-effort while the request waits in the queue, so the effective hit at start is
        `x_i + min(x_pool - x_i, W_i * B_src / bytes_per_token)`, not `x_i`. Scale on H20 / Qwen-7B:
        a 32K prefix is ~1.8 GB of KV, ~0.1 s at ~20 GB/s RDMA, vs ~6 s to recompute at 5K tok/s.
        With multi-second queue waits the pull almost always finishes, so ignoring it overstates a
        non-holder's TTFT by up to the whole recompute time. (Exact only with the pool off, as in
        every H20 run so far.)
      - optimistic: one blended prefill rate (5 012 tok/s); calibrated 8.3K tok/s at 4K falling to
        4.4K at 48K (`/tmp/h20/mn_slo_calibrate.log`), so the longest agent prefills -- the ones that
        decide agent TTFT -- are under-predicted.
      - optimistic: decode interference -- colocated steps mix <= 2048-token chunks with decode rows
        (4-mode), so prefill drains slower when many rows decode; ignored.
      - optimistic: future earlier-deadline arrivals preempt the queue under EDF (and chunk parking),
        so the current queue is a lower bound on the work ahead.
      - optimistic: x_i is the router's hash estimate; the worker may have evicted those blocks.
      **Open design questions:**
      - target: a calibrated (unbiased) prediction plus an explicit margin, or a quantile (e.g. P90)
        prediction? The cost is asymmetric -- a false veto loses the hit (recompute elsewhere,
        cache replication as a side effect), a false accept misses an SLO.
      - queue drain as a sum of fitted step times (`fit_mtp_step.py`: chunk tokens, prefix ctx,
        decode rows, decode KV, acceptance) over ceil(ahead / chunk) steps -- point 3's claim that the
        fused/split kernel is easy to model is tested here.
      - arrival-aware wait: add the expected earlier-deadline work arriving during the wait (rate x
        work of tight-deadline classes, e.g. chat).
      - transfer term: source bandwidth from an EWMA of observed pulls per source (no router query);
        sglang's prefetch policies (best_effort / wait_complete / timeout) change what "best effort"
        guarantees; source eviction during the wait; host-to-device copy contention.
      - ITL over the request's lifetime, not one step: resident rows finish and others arrive; the
        ITL SLO applies to every resident row, and all share one step time.
      - staleness: router-side bookkeeping only sees first-token and finish events; piggybacking
        worker queue state on response headers is cheaper than a per-decision query.
      - interaction with point 2 (routing to a non-holder IS replication) and point 4 (which holder to
        pull from).
      **Sub-items:**
      - [ ] measure the current predictor's error: router-logged `ttft_pred` vs measured TTFT per
            request, bucketed by load and prompt length (data: `mn_slo_veto_8gpu_20261017.sh`)
      - [ ] step-model queue drain (with decode interference) replacing the constant prefill rate
      - [ ] transfer-aware effective hit; requires an arm with the L3 pool on. **Implemented
            2026-10-17 (`router.py`, not yet measured):** the router mirrors the pool's contents
            (write-through on each finished request) with the same chained block keys as HiCache's
            page keys, takes `missing = pool hit - local hit`, and predicts the pulled tokens from
            the predicted queue wait and the workers' prefetch policy (best_effort: what arrives
            during the wait; timeout: wait up to 1 s + 0.25 s/Ktok; wait_complete: all). Pull
            bandwidth is an EWMA learned from finished requests (`cached_tokens` includes
            prefetched pages, minus the router's local-hit estimate, over the observed queue
            wait); a complete pull only raises it. The router now logs `ttft_obs` next to
            `ttft_pred`. Unit check: a 32K-prefix request whose holder has a 12 s queue -- without
            the pool the veto picks a non-holder at a predicted 9.5 s (3 s queue + 6.5 s recompute);
            with it, 3.0 s (the 1.9 GB pull fits inside the wait). A/B queued:
            `mn_pool_veto_8gpu_20261017.sh` (pool on: affinity | veto without | veto with the
            transfer term, agent C=16, chat 8 req/s).
            The index answers "radix tree?": it IS one, flattened at 256-token blocks (each key
            chains its parent's), so matching is the radix walk; it loses up to 255 tokens per
            match and evicts per block rather than per leaf.
      - [ ] margin / quantile choice set from the measured error distribution
      - [ ] A/B: cache affinity | current veto | + step model | + transfer term | + margin, on
            goodput / SLO attainment at the 8-GPU H20 points (agent C=16/32, chat 8 req/s)
- [ ] **5. THE SUPERSET CLAIM: adapt to any workload the baselines are each tuned for.**
      Added 2026-10-09 on the user's correction; **this is the capacity/throughput axis and
      nothing measured so far tests it.**

      **The claim.** Our method is 4-mode colocation plus an EDF/SLO router with a tunable
      **P/D admission ratio**. Each baseline is one fixed point in that policy space, so our
      policy space is a strict superset of all of them, and the EDF predictor plus per-class
      SLOs choose the operating point *per workload and per step* instead of at deploy time.

      | baseline | scope | the fixed choice it makes | our method reduces to it by |
      |---|---|---|---|
      | **prefill-prioritized** (sglang DEFAULT) | single node | prefill always preempts decode | admission ratio -> prefill absolute priority |
      | **mixed chunked prefill** | single node | always co-admit a prefill chunk with decode | admitting a chunk every step |
      | **PD multiplex** (`arxiv 2504.14489v2`) | multi node | time-share P and D on every node | alternating admission in time |
      | **PD disagg 1P3D** | multi node | 1 node pure-prefill, 3 pure-decode | per-node ratio -> {1.0} u {0.0 x3} |
      | **PD disagg 2P2D** | multi node | 2 pure-prefill, 2 pure-decode | per-node ratio -> {1.0 x2} u {0.0 x2} |

      **Why this is the right framing:** each baseline is optimal for *some* mixture and wrong
      for others — prefill-priority wins when prefill is scarce and starves decode when it is
      not; 2P2D is right at its provisioned ratio and strands half the cluster when traffic
      turns decode-heavy. A fixed policy cannot follow the workload; a tunable ratio can.

      **How the claim is falsified, stated up front so the experiment can fail honestly:**
      the superset argument predicts (a) at *every* static mixture we are within noise of the
      **best** baseline for that mixture, and (b) at mixtures that **shift mid-run** we are
      strictly better than all of them. If there is a static mixture where we lose to a
      baseline by more than run-to-run noise, the superset claim fails there — a fixed policy
      has less bookkeeping and that overhead is real, as the 6.8% kernel regression in
      `debug_kernel_server_perf_4_mode.md` already shows for a different mechanism.

      - [ ] **5a. SINGLE-NODE baselines** — vs **prefill-prioritized** (the default) and
            **mixed chunked prefill**. 1 GPU; runnable on the current H200 pod.
            **User 2026-10-09: must be better over BOTH**, not merely over whichever of the two
            is weaker at a given mixture. Each is optimal for some mixture, so "better over both"
            is the whole content of the superset claim at single-node scale.
      - [ ] **5b. MULTI-NODE baselines** — vs **PD multiplex** and **PD disagg (1P3D and 2P2D)**
            at equal total GPU count. Per the standing authorization: **H20**, trial named
            **vibecuda**, <= 16 GPUs (4 nodes x 4, or <= 8 nodes x 2).
            **User 2026-10-09: we must beat BOTH, and beating PD DISAGG is the main argument.**
            PD multiplex is not designed for multi-node, so a win there should come easily and
            carries correspondingly less weight -- do not present it as the headline.
            **User 2026-10-09: H20 is gated on ALL kernel items being closed first** (K1-K5 in
            `debug_kernel_server_perf_4_mode.md`). Do not allocate multi-node capacity before
            then.
      - [ ] **5c. Metric is GOODPUT under per-class SLOs, not raw tok/s.** A TTFT target for
            interactive chat and a TPOT/ITL target for agentic streams; a request counts only if
            it meets its OWN class's SLO. Raw throughput is what made the colocated-vs-colocated
            kernel A/B look flat and is the wrong number for a capacity claim.
            **User 2026-10-09: DERIVE the targets from the common settings of the papers already
            given, do not invent them.** Primary source is `arxiv 2504.08784`; note its arXiv PDF
            does not text-extract (compressed streams only), so use the ar5iv/HTML rendering.
            Cross-check against `arxiv 2504.14489v2` (PD multiplex) and ThunderAgent, and record
            which paper each number came from next to the number.
      - [ ] **5d. STATIC sweep: mixture ratio x request rate.** Prefill-heavy -> decode-heavy at
            several rates; report the **capacity frontier** (max sustainable rate per mixture),
            not a single operating point. This is the test of prediction (a): the best baseline
            changes across the sweep, and we must track it everywhere. Mixtures blended from the
            datasets already in use (TraceLab agentic replay + chat + math), with the blend
            recorded so it is reproducible.
      - [ ] **5e. DYNAMIC: mid-run mixture shift and load spike.** The core of the argument and
            the test of prediction (b): step-change the mixture (prefill share down, decode share
            up) and inject a rate spike; measure SLO attainment **through** the transition and
            time-to-recover, with the transition visible in the time series rather than averaged
            away. A fixed split cannot re-provision; we re-balance per step. This is also where a
            null result would most damage the thesis, so it gets run most carefully.
      - [ ] **5f. Read `arxiv 2504.08784`** for its SLO definitions, goodput metric and how it
            constructs and shifts mixtures. The arxiv PDF does not text-extract (compressed
            streams only); use the ar5iv/HTML rendering.

- [ ] **4. Two-way RDMA / congestion, randomized source selection** — **not demonstrated on the
      full 4-NIC fabric, and reported as such.** No congestion exists to relieve at 16 GPUs on 4
      hosts: random tie-break among near-equal hits is neutral, the L3 store moved ~236 GB without
      measurable contention, and disagg's 8 prefill GPUs replicate 4 hot prefixes to 82% hit. The
      store also COSTS ~19% chat ITL once the veto spreads load. Sub-items:
      - [x] randomized pull among near-equal-hit nodes (`cache_rand`): neutral at 16 GPUs
      - [x] the same on a NARROWED fabric (`ONE_NIC=1`) -- **randomized selection LOSES even under
            deliberate contention.** Measured below; this is a negative result for the proposed
            algorithm, in the regime built to favour it.
      - [ ] congestion modelled from router-observed past link latency (the alternative in the
            goal). NOT built, and the measurements argue against building it here: with no
            contention to observe there is nothing to fit, and a per-request router query would add
            latency to every decision. Revisit only if the narrowed-fabric run shows an effect.
      - [ ] scaling with node count (the actual claim) -- needs more nodes than the 16-GPU cap
      - [ ] **Simulated-node topology (user proposal, 2026-10-08): treat each GPU as a node and
            shape its interconnect.** 2x8 or 1x8 H200, one TP1 worker per GPU = 8 independent cache
            holders instead of 16 workers paired on 4 host NICs. More holders means hit lengths tie
            more often, which is the only case the random-pull policy differs at all.
            *User authorized H200 for this despite the standing "H20 only / H200 is scarce" note in
            the goal; record it as an explicit exception.*
            **Caveat that decides the implementation: `tc qdisc ... netem` cannot shape RoCE.**
            RDMA bypasses the kernel network stack, so delay/rate rules on the interface do not
            touch mooncake's RDMA transport. Two workable knobs instead:
            (a) run the store over TCP (`MOONCAKE_PROTOCOL=tcp`), which puts transfers back in the
                kernel path where `tc qdisc add dev <if> root netem delay <d> rate <r>` applies
                per node -- this is the way to emulate cross-node RDMA latency AND bandwidth;
            (b) keep RDMA and vary the NIC count as a coarse bandwidth divisor (`ONE_NIC=1`,
                measured: tails 2.3-3.2x worse at unchanged means), which is real contention but
                only one point on the curve.
            Prefer (a) for a latency/bandwidth sweep, and cross-check one point against (b) so the
            TCP emulation is anchored to a real RDMA measurement.
- [x] **MTP on every role** (2026-10-08) — colocation + MTP + 4-mode keeps TTFT and cuts agent ITL
      47%, and vs disagg + MTP cuts TTFT 94% / 99.6%. (An apparent regression was a router
      tie-break hotspot, retracted and fixed.) The fixed-router rerun confirms it: agent ITL -46%
      (7.2 -> 3.9 ms) at unchanged TTFT, and the 4-mode split edges stock MTP on every metric.
      - [x] the H20 serving gate now forces real mixed steps (2026-10-08). Cause of the weak pass
            was the GATE, not the kernel: background rows asked for 96 output tokens and finished
            decoding before the long prefills arrived, so nothing mixed. Fixed (1 200-token
            background rows with `ignore_eos`, staggered long prompts): **10 mixed steps, one with
            2 prefill + 11 verify rows**, vs 1 before. Both comparisons PASS, 0 fallbacks.
      - [x] **The weak pass margin is bf16 noise, not a fault -- RESOLVED 2026-10-08, MTP numbers
            stand.** Split-vs-stock min common prefix is 5 (short) / 15 (long) against a
            stock-vs-stock floor of 285 / 101, identically on H200 and H20, so it is systematic.
            `SGLANG_SPLIT_VERIFY=1` (in-server fp32 recompute of every row of every split step from
            the paged cache, `drivers/verify_gate.sh`) gives the answer:

            | rows | side | qo | kv | rel err vs fp32 |
            |---|---|---|---|---|
            | 0-3 | FA3 (prefill chunks) | 2 048 | 2 048 | 0.0032-0.0040 |
            | 4 | FA2 (**verify rows**) | 4 | 2 052 | 0.0030-0.0034 |

            0.3-0.4% is bf16 accumulation noise, and `|got|` tracks `|ref|` (2.8906 vs 2.8921). Row
            4 at `qo=4` is a VERIFY row beside 2 048-token prefill chunks, so a real 4-mode step was
            verified on both sides. A 0.3% logit difference flips a greedy argmax whenever the top
            two are close, after which sequences separate completely -- which is why the floor is
            high (identical kernels -> identical numerics) and why min-common-prefix is a WEAK
            discriminator across kernel paths. The fp32 per-row check, not prefix length, is the
            real gate; prefix length only catches gross faults like the voided per-request-array bug
            (which showed rel err ~4.0, three orders of magnitude worse).
            *Caveat:* this run logged 1 mixed step (4 verified split steps), because the fp32
            recompute slows the server enough to shift the gate's overlap timing. The step count
            came from the separate strengthened-gate run (10 mixed steps); use that for coverage and
            this for numerics.

**Labels.** Every quantitative claim below is either *measured* (with the run and commit) or
*predicted*. The design sections that predate the measurements say so where the measurements
contradicted them.

---

## [!] WHAT TODAY'S KERNEL MEASUREMENTS TAKE AWAY FROM THE SCHEDULING ARGUMENT — 2026-10-10

Four results from `debug_kernel_server_perf_4_mode.md` bear directly on the sections below, and
**two of them cut against the argument as written**. Recorded here rather than left in the kernel
doc, because the scheduling case leans on them.

**1. The two-role fusion is NET NEGATIVE on realistic 4-mode batches.** One prefill chunk plus N
decode rows, against routing everything to FA3 (same rows, no fusion):

| decode rows | co-resident (best `sm_split`) | all-FA3 | fusion |
|---|---|---|---|
| 2 | 0.2426 | 0.2557 | +5.4% |
| 8 | 0.3665 | **0.2512** | **1.46x SLOWER** |
| 32 | 0.7371 | **0.3670** | **2.01x SLOWER** |

The +5.4% was a 2-decode-token artifact. So **"one colocated batch through one fused attention
kernel" is currently slower than one colocated batch through FA3 alone** -- the scheduling story
does not need the two-role fusion, and is actively hurt by it at realistic decode counts.

**2. Point 3's premise is INVERTED: the fused kernel makes cost modelling HARDER, not easier.**
Section 3 below argues a colocated step is predictable because it is "one batch through one
attention implementation whose cost is linear in total KV". Measured today: the **optimal
`sm_split` moves with batch composition** (128 at 2 decode rows, 96 at 8, 64 at 32) and the
penalty for getting it wrong is **not** smooth -- at dec=32 with `sm_split=128` the kernel takes
**8.68 ms against 0.367 ms at that shape's own optimum, a 23x cliff**. A predictor over
`(#prefill, #decode, sum KV)` must therefore also model the split, and the target has a 23x
discontinuity in it. That is the opposite of "a smooth low-variance function".
**Point 3 should be rewritten to claim predictability for the UNFUSED colocated step** (FA3-only,
one kernel, linear in KV -- which still holds and is still better than disagg's two-stage
pipeline), not for the fused one.

**3. The baseline is stronger than this doc assumes.** flashinfer's own persistent
`BatchAttention` does the whole mixed batch in ONE kernel and is **validated correct** here
(0 bad rows vs the fp32 reference). Our fused kernel is **1.11x-4.9x slower than it** on every
fixture. So "colocation gives us one kernel instead of two" is not a differentiator -- the
baseline already has one kernel. The differentiator has to be the *schedule*.

**4. The split-KV partial slab is O(packed_qo x num_kv_chunks)** -- ~1 GB for a single 2048-token
chunk at 8K context. At production context lengths that workspace competes with the KV cache
budget, which is a **scheduling** constraint (it reduces the cache a colocated node can hold, and
cache capacity is exactly what points 1 and 2 trade on). Not previously accounted for anywhere
below.

**Gating, unchanged in kind but worse in degree than the 2026-10-06 note below:** the kernel is
incorrect on multi-item-per-CTA batches -- i.e. a prefill chunk plus tens of decode rows, which is
every realistic 4-mode step -- in *every* configuration tested. So no serving arm, single- or
multi-node, is measurable today. The scheduling arguments below stand or fall on their own logic
and on the H20 measurements already recorded; **none of them has fresh kernel support as of
2026-10-10.**

---

## What the kernel work licenses us to assume — 2026-10-06

The scheduling argument below leans on one property: **attention cost on a colocated batch is
predictable from the batch's shape**. Two measured results support that, and one undermines it.

- *Measured.* After the `seqused` fix, FA3's cost is **linear in KV** at fixed q:
  `us/kv` constant to 4.5% across a 9.8x range (10 240 -> 100 352). Before the fix it was
  `Theta(kv^2)`. A linear, low-variance cost is what makes an MLP-style predictor viable at
  all. See [the root cause entry](debug_kernel_server_perf_4_mode.md#fa3-root-cause-seqused).
- *Measured.* The two-executor split beats FA2-on-everything on a TraceLab-shaped batch by
  **1.225x (18.4%)**, 12 399.7 us against 15 190.9 us. Kernel-batch time, **not** serving.
- *Against.* A row whose KV exceeds the ~1024 chunk limit **corrupts later rows** in the
  current plan. Until that is fixed, no serving measurement is trustworthy, including anything
  below. This is the gating item.

---

## Multi-node, program concurrency, MTP everywhere (H20, 8 pods x 2 GPUs) — 2026-10-16

The 2026-10-08 runs below drove agents by a session-start rate (an axis since dropped: agent load is
program concurrency only) and ran stock FA2 without speculation. Re-run on job `1c90eac0a6848b48`
(trial 432235269, vibecuda, 8x2 H20) after A and B closed: every worker serves EAGLE3 MTP; agents are
the TraceLab 32-64K cohort (192 sessions x 2 rounds) at program concurrency AGENT_CONC, plus ShareGPT
chat at CHAT_RATE. Provisioning `tasks/pd_colocation/mn_provision_20261016.sh` (upstream 3fef129 + the
H200 serving tree's 26-file overlay + port07); driver `tasks/pd_colocation/mn_matrix_20261016.sh`.
- [x] **C=64 + chat 16 req/s (overloaded: stock chat mean TTFT 43 s, median 95 ms), all requests
  completed, 0 tracebacks / fallbacks:**
  | arm | agent tput / TTFT mean, med / TPOT / ITL / P99 ITL | chat tput / TTFT mean, med / TPOT / ITL |
  |---|---|---|
  | stock MTP, cache-affinity router | 671 / 21.9 s, 8.0 s / 47.4 / 27.2 / 360 | 3305 / 43.0 s, 95 ms / 15.1 / 13.5 |
  | **ours** (split + EDF + uncapped preemption, chunk 2048), same router | 767 / 18.0 s, 8.1 s / 29.8 / 21.5 / 149 | 3305 / 26.5 s, 82 ms / 17.3 / 13.8 |
  Ours vs stock colocation: agent tput +14%, TTFT -18%, TPOT -37%, ITL -21%, P99 ITL -59%; chat
  TTFT -38%, TPOT +15%, throughput = offered load. Mixed at this overload point; the sglang
  cache-aware-router arm failed to start (sglang_router 0.3.2 cannot bind `::`; fixed, rerun queued),
  disagg 2p2d and the lighter point (C=32, chat 8 req/s) are running.

## Multi-node measurements with the final single-GPU stack (H20) — 2026-10-16

Workload: TraceLab 32-64K agent sessions (2 rounds each) at program concurrency C (ThunderAgent) +
ShareGPT chat at a fixed request rate; EAGLE3 MTP on every worker. Arms: stock sglang colocation
(sglang_router cache_aware, stock MTP) | our cache-affinity router + stock MTP | **ours: our router +
4-mode MTP on the split + worker EDF (deadline order) + chunk preemption + load/context gate** | stock
PD disaggregation (sglang_router pd, stock MTP). Driver `tasks/pd_colocation/mn_run.py` + `mn_bench.sh`.

- [x] **16 GPUs (8 pods x 2, job 1c90eac0a6848b48, measured before the user cut H20 to 8 GPUs), C=32 +
  chat 8 req/s**, ours vs stock colocation: agent throughput +14% (929 vs 815 tok/s), TTFT -12%, ITL
  -19%; chat TTFT -51%. Disagg never started (prefill workers died on import, see below).
- [~] **8 GPUs (4 pods x 2, job bb854a10b8954223), `mn_matrix_8gpu_20261016.sh`, all reported arms 0
  tracebacks / fallbacks.** Out tok/s / mean TTFT / mean ITL / P99 ITL:
  | point | arm | agent | chat |
  |---|---|---|---|
  | C=16, chat 4/s | stock coloc | 343 / 8.3 s / 19.3 / 227 | 798 / 1697 ms / 19.9 / 219 |
  | C=16, chat 4/s | our router, stock MTP | 320 / 7.8 s / 19.5 / 173 | 798 / 1429 / 17.5 / 80 |
  | C=16, chat 4/s | **ours** | **420 / 6.9 s / 14.2 / 185** | 798 / **287** / 16.5 / 347 |
  | C=16, chat 4/s | stock PD disagg (2 prefill + 2 decode pods, clean upstream) | 137 / 34.3 s / 10.8 / 34 | 796 / 21 546 / 5.0 / 25 |
  | C=32, chat 8/s | stock coloc | 221 / 12.1 s / 97.4 / 1815 | 1461 / 7310 / 185 / 2265 |
  | ~~C=32, chat 8/s~~ | ~~our router, stock MTP~~ (router capped at 100 connections) | ~~(agent stream lost)~~ | ~~1658 / 35.1 s / 30.3 / 421~~ |
  | ~~C=32, chat 8/s~~ | ~~ours~~ (router capped at 100 connections) | ~~391 / 12.6 s / 32.2 / 315~~ | ~~1658 / 19.7 s / 28.3 / 425~~ |
  | C=32, chat 8/s | our router, stock MTP (fixed router) | 221 / 13.8 s / 95.0 / 1767 | 1433 / 10.1 s / 189 / 2294 |
  | C=32, chat 8/s | **ours** (fixed router) | **263 / 11.6 s / 74.6 / 391** | **1657 / 2.54 s / 135 / 651** |
  | C=32, chat 8/s | stock PD disagg (clean upstream) | 152 / 74.5 s / 13.6 / 51 | 1075 / 78.3 s / 6.5 / 43 |
  Ours vs stock colocation: C=16 agent throughput +22%, TTFT -17%, TPOT -59%, ITL -26%, chat TTFT -83%,
  chat ITL -17% (chat P99 ITL +58%). **C=32 (fixed router, 2026-10-16):** agent throughput +19%, TTFT -4%,
  ITL -23%, P99 ITL -78%; chat throughput +13% (serves the full offered 8 req/s; stock 1461 of ~1650
  tok/s), TTFT -65%, ITL -27%, P99 ITL -71%. Bar misses at C=32: agent TTFT (-4%) and agent/chat ITL
  (-23 / -27% vs -30%). ~~C=32 agent throughput +77%, ITL -67%, P99 ITL -83%, chat throughput +14%,
  chat ITL -85%, but agent TTFT +4% and chat mean TTFT +169% (median 85 vs 2957 ms: a starved tail).~~
  **Retracted:** measured behind our router's 100-connection cap, which queued chat inside the router
  (fewer requests on the GPUs, hence the low ITL and the starved chat tail). Rerun audit: 8 workers 0
  tracebacks / 0 fallbacks, 256-394 requests each; composition per worker FUSED_SPLIT_CALLS 6.5K-11.5K,
  MIXEDSPEC_STEPS 201-401, MIXEDSPEC_GATED >= 201, EDF_PARK firing. Command:
  `bash tasks/pd_colocation/mn_rerun_c32_cache_mtps_20261016.sh` (router.py `limit=0`). ~~The C=32 "our router, stock MTP" agent client died after warmup without output
  (silent kill; pods run at half the memory request to get scheduled).~~ **Retracted 2026-10-16:** not a
  kill. One upstream POST in our `router.py` hit EPIPE on a stale keep-alive connection (the retry caught
  only `ServerDisconnectedError`), and the concurrency replay aborts the client on any failed round.
  Also found: `router.py`'s aiohttp session used the default connector, **capped at 100 concurrent
  upstream connections**; at C=32 + chat 8 req/s ~200+ requests are in flight, so both our-router C=32
  rows above were queued inside the router -- the suspected cause of the chat starved tail. Both C=32
  our-router rows are therefore not final; router fixed (`limit=0`, keep-alive 3 s, retry on
  `ClientOSError`) and both arms rerun (`mn_rerun_c32_cache_mtps_20261016.sh`). C=16 (~2 chat in flight)
  is unaffected.
  **Ours vs stock PD disagg at C=16:** agent throughput 3.1x (420 vs 137 tok/s), agent TTFT -80%, chat
  TTFT -98.7% (0.29 vs 21.5 s); disagg keeps lower ITL (agent 10.8 vs 14.2 ms, chat 5.0 vs 16.5 ms,
  P99 34 vs 185 ms). Disagg is prefill-bound: its 2 prefill pods hit 16.6% of prompt tokens (862K /
  5.19M, `mn_audit.py`), 8 workers 0 tracebacks / 0 fallbacks, 302-344 requests each. Command:
  `bash tasks/pd_colocation/mn_disagg_rerun_8gpu_20261016.sh` (SGL_PY=up_clean, KERNEL=lpt).
  - [x] **Chat request-rate sweep at reference-list rates (2026-10-16, 23:5x):** the matrix's chat 4 / 8 req/s
    is 0.5 / 1 req/s per GPU, at or below the bottom of the papers' per-GPU rates (SLOs-Serve 1.5-4.5 per
    A100, semi-PD 11-16 per A100 for Llama3-8B ShareGPT, DistServe / Sarathi-Serve 0.5-10). Swept 1 / 2 /
    4 / 8 req/s per GPU (8-64 total; plus the matrix's 4 req/s point) at agent C=16, stock coloc | ours |
    stock PD disagg. Figure [`h20x8_chatsweep_c16.png`](pd_colocation/figures_20261016/h20x8_chatsweep_c16.png)
    (throughput vs tok/s/user: median marker, p10-p90 bar, p1-p99 whisker; SLO attainment, TTFT, TPOT vs
    chat rate; data [`h20x8_chatsweep_c16.csv`](pd_colocation/figures_20261016/h20x8_chatsweep_c16.csv)).
    Every arm completed every request; all 14 audited arms 8 workers 0 tracebacks / 0 fallbacks; ours'
    composition per worker at 8 / 16 / 32 / 64 req/s: FUSED_SPLIT_CALLS >= 3K / 3.5K / 10K / 15.5K,
    MIXEDSPEC_STEPS >= 101 / 101 / 401 / 651, EDF_PARK firing. Prefill cache hit: ours 44-68%, stock coloc
    25-39%, disagg 5-13%.
    | chat req/s (per GPU) | ours vs stock coloc: agent tput / TTFT / ITL | chat tput / TTFT / ITL |
    |---|---|---|
    | 4 (0.5) | +22% / -17% / -26% | = (offered) / -83% / -17% |
    | 8 (1) | +20% / -3% / -32% | = (offered) / -81% / -30% |
    | 16 (2) | +6% / -9% / -12% | +12% / -75% / -7% |
    | 32 (4) | +12% / -11% / -17% | +10% / -40% / -1% |
    | 64 (8) | +25% / -16% / -28% | +20% / -30% / -6% |
    **Reading.** (1) 8 H20s with this agent tenant saturate between 1 and 2 chat req/s per GPU, so the
    papers' upper per-GPU rates (their ShareGPT-only workloads on A100) are deep overload here; above 8
    req/s every arm queues (TTFT 10-144 s) and colocated chat TPOT falls again past 16 req/s because
    fewer requests decode at once. (2) Ours beats stock colocation on throughput (+6..+25%) and TTFT at
    every rate; the >=30% ITL bar is met only at 8 req/s (both tenants) and 64 req/s agent (-28%, just
    short); agent TTFT never clears -30%. (3) Stock PD disagg has the worst TTFT (21-144 s) and the
    lowest throughput at every rate, but the lowest TPOT and a far better slow-user tail at low load
    (chat p1 47 vs 4 tok/s/user at 4 req/s): its prefill pods throttle admission, so its decode pods are
    lightly loaded. Command: `bash tasks/pd_colocation/mn_chat_ratesweep_8gpu_20261016.sh`, then
    `PYTHONPATH=<matplotlib> python tasks/pd_colocation/mn_sweep_figure.py tasks/pd_colocation/figures_20261016/h20x8_chatsweep_c16`;
    audit `SID=bb854a10b8954223 NPODS=4 python tasks/pd_colocation/mn_audit.py <arm>_8g_chatsweep_c16_r<R>`.
  - [x] **Stock PD-disagg ITL/TPOT are understated by burst delivery (2026-10-16):** 17.5-29.1% of its
    requests stream every token at once at the end (ITLs 0.0 s; never on colocated workers, 0 of 1 200),
    so their decode time sits in TTFT and adds zero-gap ITLs. Excluding those requests, disagg mean ITL /
    TPOT: C=16 chat 5.5 / 6.2 ms (reported 5.0 / 5.2), agent 11.4 / 11.2 (10.8 / 8.4); C=32 chat 7.4 / 8.4
    (6.5 / 6.0), agent 14.1 / 12.6 (13.6 / 10.3). Disagg still has the lower ITL; its TPOT edge shrinks.
    `eval_curves.py` now reports `burst_frac` and excludes burst requests from tok/s/user. Per-request
    spread at chat 4 req/s, C=16 (`mn_sweep_figure.py`): chat tok/s/user p1 / median / p99 -- ours 4.3 /
    243 / 394, stock coloc 4.8 / 239 / 371, stock disagg 47 / 207 / 685: colocation's slow-user tail
    (p1 ~4-5 tok/s/user) is much worse than disagg's.
  **Stock PD disagg at C=32** (same script): saturated on prefill -- prefill hit 10.0% (1.05M / 10.5M
  tok), agent and chat mean TTFT 74.5 / 78.3 s, agent 152 tok/s, chat 1075 tok/s (offered ~1 650); 8
  workers 0 tracebacks / 0 fallbacks, 617-677 requests each. Ours (fixed router) vs disagg at C=32: agent
  throughput 1.7x (263 vs 152), agent TTFT -84%, chat throughput +54%, chat TTFT -97% (2.5 vs 78.3 s);
  disagg keeps far lower ITL (agent 14.1 vs 74.6 ms, chat 7.4 vs 135 ms, burst-excluded).
  **PD disaggregation baseline (history):** its prefill workers failed on import at C=16 -- mooncake links
  `libcudart.so.12`, the venv is cu130 -- fixed by installing `nvidia-cuda-runtime-cu12` on all pods;
  then a cross-GPU port collision (ports now claimed under a lock), the serving overlay's spec-v2 assert in
  disagg prefill (baseline now runs clean upstream), and orphan workers relaunched by surviving
  `mn_node.sh` supervisors that OOMed new prefill workers (`mn_kill.sh` now kills supervisors first).

## Multi-node measurements (H20, 8 pods x 2 GPUs = 16 workers) — 2026-10-08

Setup: merlin job `c2fb3965f36cd15f` (cn-seed, trial vibecuda), 8 pods x 2 H20, one
Qwen2.5-7B-Instruct TP1 worker per GPU (persistent FA2 kernel in every arm, so arms differ only
in placement and routing). Pods 0/5/6/7 share one physical host. Workload
(`tasks/pd_colocation/mn_bench.sh`, start barrier): 64 TraceLab sessions x 6 turns at
0.5 sessions/s, replaying recorded inter-turn gaps x0.1, where each turn extends the previous
prompt (16-49K tokens); plus random-ids chat at 16 req/s (3 200 requests, 512 -> 128 tokens).
Driver: `SESSIONS=64 TURNS=6 SESSION_RATE=0.5 SLOWDOWN=0.1 CHAT_RATE=16 CHAT_N=3200 python
tasks/pd_colocation/mn_run.py <arms>`; audit: `mn_audit.py` (every pod's worker logs).
"Worker-reported hit" is the sum of sglang's own `meta_info.cached_tokens` over the sum of
`prompt_tokens`, recorded per request by the router.

**Re-run with the fd limit fixed (2026-10-08).** The first table (2026-10-07) ran with the pods' soft
`ulimit -n` of 1024. Under the chat burst the stock router ran out of sockets, its circuit breaker
opened, and it rejected most requests: disagg_2p2d served 163/384 agent and 2 265/3 200 chat. That
table is withdrawn. The re-run raises the limit to the hard limit (`ulimit -n $(ulimit -Hn)`, verified
1 024 768 in the router process) and gives disaggregation its best stock configuration: circuit
breaker off, health ejection off, `--prefill-policy cache_aware --decode-policy power_of_two`
(commits a8bd07337, b6ac0c2ad). Command:
`python mn_run.py disagg_2p2d disagg_1p3d coloc_cache coloc_rr coloc_rand` with the knobs above.
Audit: `mn_audit.py <arms>`. 0 tracebacks and 0 fallbacks on all 16 workers in every arm.

| arm | completed agent / chat | agent TTFT ms mean / P99 | agent ITL ms | chat TTFT ms mean / P99 | chat ITL ms | chat out tok/s | prefix hit | requests/worker |
|---|---|---|---|---|---|---|---|---|
| disagg_2p2d (8 prefill + 8 decode GPUs) | 384 / 3 173 | 21 581 / 77 164 | 5.8 | 20 147 / 77 131 | **3.7** | 152.5 | 23.3% | 19-885 |
| disagg_1p3d (4 prefill + 12 decode GPUs) | 384 / 3 176 | 49 257 / 97 041 | **5.2** | 50 301 / 94 592 | **3.5** | 152.6 | 30.8% | 19-936 |
| coloc_rr (round-robin) | 384 / 3 200 | 11 702 / 61 465 | 22.4 | 8 783 / 68 555 | 25.5 | 1 338.2 | 12.7% | 225-226 |
| **coloc_cache** (affinity, load tie-break) | 384 / 3 200 | **1 596 / 8 371** | 7.2 | **101 / 1 539** | 6.4 | **1 481.6** | **72.9%** | 88-329 |
| coloc_rand (affinity, load then random tie-break) | 384 / 3 200 | **1 530 / 8 190** | 7.5 | 121 / 2 613 | 6.3 | 1 481.1 | 72.8% | 111-307 |

Prefix hit: for coloc, the worker-reported `cached_tokens / prompt_tokens` summed over requests;
for disagg, the prefill workers' own logged cache hit. The disagg arms' failed chat requests
(27 and 24) were all streams cut off mid-response, consistent with aborted KV transfers.

**Reading.**
- **Against the best disaggregated layout, cache-affinity colocation cuts mean TTFT 93% for the
  agent (21.6 s -> 1.6 s) and 99.5% for chat (20.1 s -> 0.10 s).** It serves the full offered chat
  rate (1 482 vs 153 output tok/s over the run) and completes every request.
- **Disagg is prefill-bound here, and its cache-aware prefill routing finds little cache.** It hits
  23-31% of prompt tokens against colocation's 73%. With 4-8 prefill GPUs serving all 64 sessions'
  appends, each turn's prefix is on a prefill node only if the last turn ran there and survived
  eviction. Colocation can place a turn on any of 16 GPUs: point 1's flexibility, measured.
- **Disagg keeps the lower ITL** (3.5-5.8 ms vs 6.3-7.5 ms). Its decode GPUs never run prefill.
  This is the price of colocation on this kernel stack, and it is what the 4-mode kernel and
  deadline ordering attack on a single node (point 3: -38% to -42% ITL vs stock MTP).
- **Routing is most of colocation's win.** Round-robin colocation is only 1.8-2.3x better than
  disagg on TTFT and 4-7x worse on ITL. With 12.7% hit, every node recomputes session prefixes, and
  that prefill stalls decode everywhere.
- **Random choice among near-equal hits costs nothing at this scale** (agent TTFT -4%, chat TTFT
  +20%, ITL equal). It does not help yet either: 16 workers on 4 hosts do not congest links (point 4).
  The earlier purely random tie-break, which sent hit-0 chat to busy workers, is fixed by breaking
  ties on in-flight load first.

**Shared L3 prefix store: cross-node pulls on a complete graph (points 2 and 4), 2026-10-08.**
`<arm>_hc` arms add a cluster-wide mooncake store. One `mooncake_master` runs on pod 0. Every
worker contributes 32 GB of host DRAM over RDMA on the back-end NICs and runs
`--enable-hierarchical-cache --hicache-storage-backend mooncake` (write-through, best-effort
prefetch). Any worker can therefore pull a prefix that any other worker computed. Commands:
`python mn_run.py coloc_rr_hc coloc_cache_hc` (commits 5eae86c26, a8465ecc7). 0 tracebacks and
0 fallbacks.

| arm | completed agent / chat | agent TTFT ms mean / P99 | chat TTFT ms mean / P99 | chat ITL ms mean / P99 | worker-reported hit | store gets (batched) |
|---|---|---|---|---|---|---|
| coloc_rr | 384 / 3 200 | 11 702 / 61 465 | 8 783 / 68 555 | 25.5 / 909.5 | 12.7% | - |
| coloc_rr_hc | 384 / 3 199 | 9 982 / 37 938 | 5 163 / 38 004 | 20.7 / 153.6 | **22.3%** | 12 649 (3.2 M items) |
| coloc_cache | 384 / 3 200 | 1 596 / 8 371 | 101 / 1 539 | 6.4 / 11.8 | 72.9% | - |
| coloc_cache_hc | 384 / 3 200 | 1 631 / 8 224 | 122 / 1 549 | 6.3 / 11.7 | 72.9% | 16 |

- **Migration works where routing cannot keep a prefix local.** Under round-robin the store
  lifts the hit rate from 12.7% to 22.3% by pulling prefixes other nodes computed. Chat TTFT
  falls 41%, chat P99 ITL 83%, and agent P99 TTFT 38%. Replication happens as a side effect of
  serving (write-through), with no migration step and no knowledge of the holder.
- **With affinity routing the store is idle (16 pulls in the whole run) and changes nothing.**
  Affinity already serves 73% of tokens locally. This is the intended ordering: route to the
  holder first (point 1), and pull from the store only when the router must place a request
  elsewhere (an SLO veto or an overloaded holder, point 2).
- **Not yet shown: the congestion advantage (point 4).** At 16 GPUs on 4 hosts the store moved
  ~236 GB without measurable link contention, so random vs load-aware source choice cannot
  differ yet. That needs either more nodes than the 16-GPU cap allows, or a deliberately
  hot-prefix workload where many nodes pull one session's prefix at once. The bipartite
  disagg design forces exactly that on its few prefill nodes.
- The single failed chat request in coloc_rr_hc was a worker dropping a reused keep-alive
  connection (`ServerDisconnectedError`). The router now retries once before streaming starts
  (a8465ecc7), as the stock router does.

**The SLO veto, calibrated, with worker-side EDF (point 1), 2026-10-08.** The router's
`PerfModel` was fitted on one H20 worker (`mn_run.py calibrate`, `calibrate_h20.py`): prefill
9.0K tok/s at 4-8K falling to 4.9K at 49K, so 5 531 tok/s over 16-49K, and decode ITL 5.05 ms +
0.065 ms per running sequence. The router uses the same length-scaled TTFT deadline as the
workers (2 s + 1 s per 5K tokens beyond 4 096). Workers run deadline-ordered admission with no
chunk cap (`_edf` suffix; the configuration that won on the H200). Command:
`ROUTER_ARGS="--prefill-tok-s 5531 --decode-step-ms 5.05 --decode-ms-per-seq 0.065 --ttft-slo-ms 2000"
python mn_run.py coloc_cache_edf coloc_slo_edf` (commit bf96b6cd3). 0 tracebacks and 0 fallbacks;
every request completed.

| arm | agent TTFT ms mean / P99 | chat TTFT ms mean / P99 | ITL ms agent / chat | vetoes |
|---|---|---|---|---|
| coloc_cache | 1 596 / 8 371 | 101 / 1 539 | 7.2 / 6.4 | - |
| coloc_cache_edf (worker EDF) | 1 611 / 8 102 | 117 / 2 269 | 7.4 / 6.4 | - |
| **coloc_slo_edf** (veto + worker EDF) | **1 582** / 8 132 | 108 / **1 313** | 7.2 / 6.4 | 71 of 3 587 (2.0%) |

- **The veto fires once calibrated** (the uncalibrated run vetoed nothing). It moves 2% of
  requests off their highest-hit node and trims the chat TTFT tail (P99 -15% vs affinity alone),
  with means unchanged. At this load affinity rarely pushes a worker past its SLO, so there is
  little to correct. The veto is a tail guard, and its value should grow with hotspots, which the
  HOT experiment below creates.

**Hot-prefix stress (points 1, 2, 4), 2026-10-08.** `HOT=1`: the chat tenant becomes 4 groups
sharing 16K-token system prompts (`generated-shared-prefix`, 512-token questions, 16/s, 3 200
requests); the agent tenant is unchanged. Command: `HOT=1 RUN_TAG=_hot ROUTER_ARGS="<calibrated>
--ttft-slo-ms 2000" python mn_run.py coloc_cache coloc_slo_edf coloc_slo_edf_hc disagg_2p2d`
(commits 498d60155, daaedef6a; the cache tie-break was already randomized). 0 tracebacks, 0
fallbacks, every request completed.

| arm | agent TTFT ms mean / P99 | chat TTFT ms mean / P99 | ITL ms agent / chat (chat P99) | prefix hit | vetoes | max requests/worker |
|---|---|---|---|---|---|---|
| coloc_cache_hot | 1 673 / 8 602 | 340 / 2 263 | 7.1 / 23.4 (224) | 93.5% | - | 803 |
| coloc_slo_edf_hot | 1 605 / 8 408 | 347 / 3 290 | 7.8 / 19.1 (230) | 93.5% | 96 (2.7%) | 808 |
| coloc_slo_edf_hc_hot | 1 648 / 8 189 | 377 / 3 414 | 7.5 / **18.9 (200)** | 93.4% | 87 (2.4%) | 802 |
| disagg_2p2d_hot | 27 776 / 76 129 | 29 513 / 85 682 | 6.3 / **3.8 (30.5)** | 81.9% | - | 861 |

- **Colocation still wins TTFT by an order of magnitude** (agent -94%, chat -99%), even when the
  workload is designed to create hotspots.
- **CORRECTED 2026-10-08 (see the decode-aware veto below): the "veto is too weak" reading in
  this bullet held only while the veto priced TTFT alone. Pricing the decode slowdown cuts chat
  mean ITL to 8.3 ms and P99 to 44 ms.** Original reading: **Disagg wins chat ITL by 5x here.** A hot group's decode concentrates on its holder in
  colocation (~800 requests on the busiest worker), while disagg spreads decode over 8 dedicated
  GPUs. This is colocation's real weakness, and the veto is too weak to fix it: it moves only
  ~2.5% of requests, because the calibrated model predicts TTFT, and TTFT stays fine on a holder
  whose prefix is cached. The overload shows up as ITL, which the veto does not price. The veto
  plus store trims chat ITL 19% (23.4 -> 18.9 ms) and P99 to 200 ms, and does not close the gap.
- **What point 4 needs and does not yet have:** a veto that prices the decode (TPOT) slowdown a
  request causes on its holder. The router's TPOT model exists (`decode_step_ms`,
  `decode_ms_per_seq`), but with the calibrated slope (0.065 ms/seq) it never crosses the
  100 ms ITL budget. Calibration measured ITL vs concurrency at short contexts, while the holders'
  ITL is driven by 16K-prefix decode rows plus their prefill.
- **Disagg's prefill pods hit 82%** because 4 hot prefixes replicate across 8 prefill GPUs, so the
  bipartite graph is not the bottleneck at 16 GPUs either. No congestion effect is measurable
  within the cap.

**Narrowing the fabric makes congestion measurable within the GPU cap (point 4), 2026-10-08.**
`ONE_NIC=1` pins every pod's store traffic to mlx5_1 instead of spreading over mlx5_1..4, cutting
aggregate cross-node bandwidth ~4x (verified per worker from `/proc/<pid>/environ`:
`MOONCAKE_DEVICE=mlx5_1`, `MC_MS_AUTO_DISC=0`). Same workload, same arm, 4 NICs vs 1:

| coloc_rr_hc (the heaviest puller, ~12.6K batched gets) | 4 NICs | 1 NIC |
|---|---|---|
| agent TTFT ms mean / P99 | 9 982 / 37 938 | 9 861 / 34 510 |
| chat TTFT ms mean / P99 | 5 163 / 38 004 | 4 787 / 31 849 |
| agent ITL ms mean / **P99** | 20.1 / **110.7** | 21.9 / **358.1** |
| chat ITL ms mean / **P99** | 20.7 / **153.6** | 21.5 / **351.4** |

**Tails degrade 2.3-3.2x while means and TTFT are unchanged** -- the signature of link contention,
not compute: a pull that queues behind others stalls one decode step badly rather than slowing
every step. This is the regime point 4's argument is about, and it did not exist on the full
fabric. The source-selection comparison (`coloc_cache_hc_1nic` deterministic best-holder vs
`coloc_rand_hc_1nic` randomized among near-equal holders) runs on this narrowed fabric.
*Caveat on scope:* this shows randomized selection can be tested, not that colocation scales
better with node count. One NIC at 16 GPUs emulates per-link pressure, not a larger cluster's
topology.

**That pair could not isolate source selection -- design flaw, found on the second arm.** With
affinity the narrowed fabric changes nothing (`coloc_cache_hc_1nic`: agent TTFT 1 589 ms, chat P99
ITL 11.8 ms, against 1 631 ms / 11.7 ms on 4 NICs), because affinity keeps 93% of tokens local and
the store is idle. So the only arm that congests (`rr`) ignores cache hits entirely, while the
arms that select on hits never pull. Randomized source selection has no choice to make in either.
The isolating configuration needs both at once: several workers holding the SAME prefix (so hit
lengths genuinely tie) and local misses that pull cross-node. `HOT=1 ONE_NIC=1` with the
decode-aware veto does that -- the veto spreads each shared-prefix group over several workers, which
both replicates the prefix and forces the spread requests to pull. New policy `cache_slo_rand`
(c87aa83b9) is `cache_slo` with the load term dropped inside a hit bucket, i.e. the goal's "pick
randomly among near-equal holders"; it differs from `cache_slo` only when holders really tie.
Running: `HOT=1 ONE_NIC=1 mn_run.py coloc_slo_edf_hc coloc_slo_rand_hc`.

**Randomized source selection under contention: measured, and it LOSES (2026-10-08).** The
isolating pair, both arms identical except the policy -- hot shared prefixes, `ONE_NIC=1`,
decode-aware veto (`--decode-grid`, 12 points), L3 store, worker EDF on both sides. Command:
`HOT=1 ONE_NIC=1 RUN_TAG=_hotnic ROUTER_ARGS="<grid> --ttft-slo-ms 2000 --itl-slo-ms 40"
mn_run.py coloc_slo_edf_hc` then `coloc_slo_rand_edf_hc` (commits c87aa83b9, f6eab46b3).
0 tracebacks, 0 fallbacks, all requests completed in both.

| arm | chat ITL mean / P99 ms | chat TTFT mean / P99 ms | agent ITL mean / P99 ms | agent TTFT ms |
|---|---|---|---|---|
| `coloc_slo_edf_hc_hotnic` (load-ordered within a hit bucket) | **8.9 / 66.5** | **484 / 6 196** | **9.5 / 52.2** | **1 985** |
| `coloc_slo_rand_edf_hc_hotnic` (random within a hit bucket) | 11.0 / 108.1 | 558 / 6 512 | 10.4 / 95.5 | 2 099 |

**Randomizing among near-equal holders is worse on every metric**: chat mean ITL +24%, chat P99 ITL
+63%, agent P99 ITL +83%, chat TTFT +15%. **Why, and why this is the third time:** dropping the
load term is what the policy DOES -- and load, not link bandwidth, is the binding constraint even
on a 4x-narrowed fabric. A holder chosen at random may already be decoding a dozen 16K rows, and
that costs far more than the pull it saves. The same mechanism appeared twice before at full
fabric width (`coloc_rand`, where purely random choice sent hit-0 chat to busy workers, and the
`min()` tie-break hotspot). The fix each time was to order by in-flight load first and use
randomness only to break what remains.

**Point 4's algorithm, as proposed, is therefore refuted at this scale**: "when several nodes have
hit lengths within a threshold, pull from one at random" is dominated by "pull from the least
loaded of them". Randomness is not harmless here; it discards the signal that matters. What the
goal's underlying intuition gets RIGHT is that concentration is the risk -- but the measured cure
is the load-aware tie-break plus the decode-aware veto (chat ITL -65%, P99 -80%, v0.18.0), not
randomization. A congestion-dominated regime where randomization wins would need link cost to
exceed holder-compute cost; one NIC at 16 GPUs does not reach it, and the simulated-node sweep
below is the way to look for it.

Arm 3 of the flawed pair closes it as predicted:Arm 3 of the flawed pair closes it as predicted: `coloc_rand_hc_1nic` (randomized tie-break,
narrowed fabric) measures agent TTFT 1 611 ms / ITL 7.3 ms / P99 13.2 ms and chat 134 ms / 6.5 /
12.2 -- indistinguishable from `coloc_cache_hc_1nic`'s 1 589 / 7.3 / 13.5 and 148 / 6.5 / 11.8.
With affinity keeping 93% of tokens local, neither policy pulls, so the tie-break has nothing to
spread and the narrowed fabric nothing to congest. Both arms: 0 tracebacks, 0 fallbacks, all
requests completed.

A first attempt at this run was DISCARDED before use: the node script was pushed to the pods in
the background while the arms launched, and the push is sequential, so the last pod's workers
started on the old 4-NIC assignment (2 of 16 workers). The identical script mtime on every pod
was misleading -- `tar` preserves mtimes, so it showed the local file's time, not extraction time.
Later arms would have been fully pinned, making the A/B inconsistent.

**The veto has to price TPOT, not TTFT (point 1, resolved 2026-10-08).** The hot-prefix holders'
overload shows up as slow decode, not slow prefill, so a TTFT-only veto never fired on them. Two
fixes, in order:

1. *Calibration.* `calibrate_h20.py` measured decode ITL at 256-token context only, fitting
   0.065 ms/seq. Re-measured at 256 / 4K / 16K context x 1 / 8 / 32 / 64 concurrency, ITL is
   **flat to ~131K resident tokens** (9.1 ms at 32x4K, 8.3 ms at 8x16K) and then jumps to
   **136.6 ms at 524K** (32x16K) and 347.8 ms at 64x16K.
   **CORRECTED 2026-10-08: that knee is PREFILL INTERFERENCE in the measurement, not a capacity or
   bandwidth effect.** (The earlier reading in this bullet -- "a capacity effect, eviction/
   retraction or an attention-path change, root cause not yet found" -- was wrong.) The calibration
   submits all N requests at once, so with 16K prompts the later requests' prefill chunks share
   steps with the earlier ones' decode; at 256-token context prefill is trivial and no
   interference appears. Checked directly: a server held at 32 concurrent requests and 543K
   resident tokens logs **pure-decode** steps at 1 716 tok/s = **~18.6 ms per step**, with token
   usage 0.44 and CUDA graphs on -- 7x faster than the 136 ms the grid records for that shape, and
   no retraction. (The confirming script's own summary was lost: the chain's `KILL` sweep matches
   `launch_server` and killed its server mid-run. The server log carries the numbers.)
   **This does not invalidate the grid for routing**, which needs "what ITL will a request see on
   this worker", and a worker serving a hot prefix really is running prefill beside decode. It
   does mean the grid is a *loaded-worker* cost, not a pure decode cost, so it should not be reused
   as a decode-only model.
2. *Model form.* No closed form fitted that surface: a 3-term linear fit returned a NEGATIVE base
   cost (-4.64 ms), and `max(flat, kv/bw)` overpredicted the middle of the grid 3-4x. The router
   now **interpolates the measured grid** (`--decode-grid`, bilinear, KV-scaled past the corner;
   reproduces every measured point). 16 rows x 16K predicts 51 ms where the old slope predicted
   ~6 ms. Commits 7010f40fe, f6eab46b3.

| hot-prefix arm | chat ITL mean / P99 ms | chat TTFT mean / P99 ms | agent ITL mean ms | vetoes |
|---|---|---|---|---|
| coloc_cache_hot (affinity) | 23.4 / 224 | **340** / 2 263 | **7.1** | - |
| coloc_slo_edf_hot (TTFT-only veto) | 19.1 / 230 | 347 / 3 290 | 7.8 | 96 (2.7%) |
| **coloc_slo_edf_hot2 (decode-aware veto, ITL SLO 40 ms)** | **8.3 / 44.3** | 424 / 5 161 | 8.7 | **192 (5.4%)**, 20 least-bad |
| coloc_slo_edf_hc_hot2 (decode-aware veto + L3 store) | 9.9 / 80.0 | 468 / 6 304 | 9.8 | - |
| disagg_2p2d_hot | 3.8 / 30.5 | 29 513 / 85 682 | 6.3 | - |

**Pricing the decode cost cuts chat mean ITL 65% (23.4 -> 8.3 ms) and P99 80% (224 -> 44 ms)**,
closing most of the gap to disaggregation, for a 25% chat TTFT cost (340 -> 424 ms) and slightly
worse agent ITL. Colocation keeps a ~70x TTFT advantage. This is point 1's mechanism working as
designed: route to the cache holder first, and veto it only when the load it would carry is
predicted to break the SLO -- with the SLO term that the overload actually appears in.

**The L3 store does not help once the veto spreads the load, and costs ~19% chat mean ITL / 80%
P99** (8.3 -> 9.9 ms, 44 -> 80 ms; agent ITL 8.7 -> 9.8 ms). Its write-through traffic and
prefetches compete with decode on the same GPUs, while the vetoed requests already replicate the
hot prefixes by recomputing them. Together with the earlier result that the store is idle under
affinity routing (16 pulls), the rule is: **the store pays only where routing CANNOT place a
request near its prefix** -- round-robin, or a cold cluster -- and is a cost where the router is
already doing that job.

**Point 4, honest conclusion.** The congestion argument is **not demonstrated within the 16-GPU
cap**, in either direction:
- Random choice among near-equal cache holders costs nothing and gains nothing (coloc_rand).
- Disagg's own prefill pods reached 81.9-82% prefix hit on the hot workload, because 4 hot
  prefixes replicate across 8 prefill GPUs. The bipartite graph is not a bottleneck at this size.
- The L3 store moved ~236 GB between nodes with no measurable link contention.
The prediction that colocation's complete graph scales better as nodes increase is therefore
**untested, not supported**: at 16 GPUs on 4 hosts there is no congestion to relieve. Testing it
needs either more nodes than the cap allows or an artificially narrowed interconnect.
- **Worker-side EDF does nothing here** (agent +1%, chat +16 ms, within noise). With 16 workers,
  each sees a light queue, and admission order matters only when work queues up. On a single
  saturated H200 it cut chat TTFT 62% (point 3).

**MTP on every worker (`_mtp`), 2026-10-08.** *Correction, same day: the "colocation regresses
with MTP" reading below is RETRACTED. The cause was a router tie-break hotspot, not MTP; see the
resolution after the table.*
EAGLE3 (3 steps, topk 1, 4 verify tokens) runs on every role. Colocated workers also run the
4-mode step on the FA3/FA2 split (`KERNEL=split`). Before any MTP arm, the H20 serving gate
(`h20_gate.sh`) passed both comparisons with 0 fallbacks, weakly: the split's run had only ONE
mixed step, and one short prompt agreed with stock MTP for 5 tokens (noise floor 101). Commands:
`KERNEL=split python mn_run.py disagg_2p2d_mtp coloc_cache_mtp coloc_cache_edf_mtp` (6561caf51).

| arm | completed agent / chat | agent TTFT ms mean / P99 | chat TTFT ms mean / P99 | ITL ms agent / chat | prefix hit | requests/worker |
|---|---|---|---|---|---|---|
| disagg_2p2d (no MTP) | 384 / 3 173 | 21 581 / 77 164 | 20 147 / 77 131 | 5.8 / 3.7 | 23.3% | 19-885 |
| disagg_2p2d_mtp | 384 / 3 178 | 27 423 / 77 995 | 26 470 / 78 841 | **2.9 / 4.1** | 20.5% | 22-919 |
| coloc_cache (no MTP) | 384 / 3 200 | **1 596** / 8 371 | **101** / 1 539 | 7.2 / 6.4 | 72.9% | 88-329 |
| coloc_cache_mtp | 384 / 3 200 | 18 456 / 142 986 | 42 361 / 142 796 | 14.5 / 24.3 | 72.9% | 90-**812** |

- MTP halves disagg's agent ITL, as expected, and worsens its TTFT: the prefill pods also run
  the draft.
- **Colocation with MTP + 4-mode is ~12x worse on agent TTFT and 2-4x worse on ITL than without
  MTP.** Cache hit is unchanged (72.9%), and per-worker queues stay tiny (max 9 waiting). Chat
  TTFT is 37-93 s over the first 60% of requests, then recovers. Agent TTFT stays at 12-23 s for
  the whole 30-minute run, against a flat 1.4-1.8 s without MTP. One worker takes 812 requests
  (329 max without MTP).
- **Resolution: router hotspot, not MTP.** The `cache` policy broke load ties with a bare
  `min()`, which returns the LOWEST-INDEX worker. Every session's first turn and every chat
  request ties at hit 0, so ties went to workers 0 and 1, and affinity then pinned those sessions
  there: workers 1 and 0 took 811 and 565 requests (61 and 91 agent turns, vs at most 36
  per worker otherwise). The next arm, coloc_cache_edf_mtp, ran the same code with the same bias.
  It happened to spread evenly (max 341 requests, 37 agent turns per worker) and is healthy:

  | arm | agent TTFT ms mean / P99 | chat TTFT ms mean / P99 | ITL ms agent / chat | max requests/worker |
  |---|---|---|---|---|
  | coloc_cache (no MTP) | 1 596 / 8 371 | 101 / 1 539 | 7.2 / 6.4 | 328 |
  | coloc_cache_mtp (hotspot) | 18 456 / 142 986 | 42 361 / 142 796 | 14.5 / 24.3 | 811 |
  | **coloc_cache_edf_mtp** | **1 561 / 8 067** | 114 / 1 382 | **3.8** / 5.7 | 341 |
  | disagg_2p2d_mtp | 27 423 / 77 995 | 26 470 / 78 841 | 2.9 / 4.1 | 919 |

  **Rerun on the fixed router (`RUN_TAG=_fixtie`, 7010f40fe), plus stock MTP without the 4-mode step
  (`_mtps`):**

  | arm | agent TTFT ms mean / P99 | chat TTFT ms mean / P99 | ITL ms agent / chat |
  |---|---|---|---|
  | coloc_cache (no MTP) | 1 596 / 8 371 | **101** / 1 539 | 7.2 / 6.4 |
  | coloc_cache_mtps (stock MTP) | 1 629 / 8 719 | 135 / 3 586 | 4.2 / 5.8 |
  | **coloc_cache_mtp (4-mode, fixed router)** | **1 569** / 8 516 | 122 / 2 607 | **3.9 / 5.7** |
  | coloc_cache_edf_mtp (4-mode + worker EDF) | 1 561 / **8 067** | 114 / **1 382** | 3.8 / 5.7 |

  MTP cuts agent ITL **46%** (7.2 -> 3.9 ms) at unchanged TTFT, and the 4-mode split is slightly
  ahead of stock MTP on every metric (agent TTFT -3.7%, agent ITL -7%, chat P99 TTFT -27%). Worker
  EDF adds a little more in the chat TTFT tail. With MTP + the 4-mode split, colocation keeps its
  TTFT and cuts agent ITL 46% against colocation without MTP. Against disagg with MTP on both sides: TTFT -94% (agent) and
  -99.6% (chat). Disagg's dedicated decode GPUs still give lower ITL (2.9 / 4.1 vs 3.8 / 5.7 ms).
  The tie-break is now randomized among equally loaded workers. The stock-MTP arm and a rerun of
  coloc_cache_mtp run on the fixed router.


## Evidence state of the four points — 2026-10-10

Added because the sections below carry 17 "measured" markers and **no** markers for the claims
that are argument only, so a reader cannot tell which is which. Nothing here changes the
arguments; it labels them.

| point | mechanism exists in sglang | measured | argument only |
|---|---|---|---|
| 1. highest-hit-first with an EDF/SLO veto | yes — radix/prefix cache + router hit lengths; EDF ordering is **measured** on the MTP server (`edf-order-not-cap`) | the EDF *ordering* win, and the linear-in-KV step cost the veto's predictor needs | that routing hit-first **beats** disagg's prefill-first on TTFT: no multi-node run |
| 2. best-effort cache migration | yes — hier cache + dist cache pool | nothing | the whole replication advantage, incl. the ThunderAgent thrashing comparison |
| 3. EDF predictor with acceptance length | yes — EWMA of accept length is already tracked | step cost linear in total KV, on one GPU, 4-mode MTP (section 3) | a matched fused/unfused EDFG A/B now runs cleanly, but shows no ITL/throughput advantage and does not establish better predictor accuracy |
| 4. two-way RDMA, complete vs bipartite graph | partially — transfer paths exist, congestion-aware pull selection does not | nothing | the entire scaling claim; `O(N^2)` vs `O(N_p*N_d)` is topology arithmetic, not a measurement |

So **one** of the four points has supporting measurement (3's cost model, plus 1's EDF
ordering), and it is single-GPU. Points 2 and 4 are unmeasured end to end. The multi-node runs
that would test them are gated on the kernel items per the standing instruction, and separately
on the FA3 defect, since a serving A/B today would measure a corrupted FA3 path.

**This labelling does not weaken the doc's purpose** — the four points are a design argument the
goal asked to be written down, and they are written down. It records which parts a reviewer can
challenge on data and which on reasoning.

## 1. Route to the highest cache-hit node first, with an EDF/SLO veto

**The disagg baseline.** PD disaggregation picks a *prefill* node first, because prefill and
decode live on different pools. Prefix cache locality is then a second-order concern: the
request goes where prefill capacity is, and whatever prefix that node happens to hold is luck.
With agentic traffic this is the wrong order of preference, because the prefix is enormous
(64-100K) relative to the new tokens (~3.8K), so a hit is worth far more than a marginally
emptier queue.

**The colocated policy.** Rank candidate nodes by prefix-hit length, then **veto** by SLO
feasibility rather than by load:

    candidates = sort(nodes, key = -hit_len(node, request))
    for n in candidates:
        if feasible(n, request):  route(request, n); return
    route(request, argmin_over_nodes(predicted_slo_slack))   # least-bad

`feasible(n, r)` must account for more than the node's current occupancy. Under EDF, admitting
`r` can only delay requests whose deadline is **later** than `r`'s; the ones that can be hurt
by `r` are those with *tighter* deadlines already queued ahead of it. So:

    feasible(n, r):
        head = {q in queue(n) : deadline(q) < deadline(r)}
        # r is served after `head` under EDF, so r's own completion is pushed by all of head
        t_r = now + sum(predicted_service_time(q) for q in head) + predicted_service_time(r, n)
        if t_r > deadline(r): return False
        # and admitting r must not push any LATER-deadline request past its own deadline
        for q in queue(n) with deadline(q) > deadline(r):
            if predicted_completion(q | r admitted) > deadline(q): return False
        return True

**Why this is a structural advantage, not a tuning difference.** In disagg the prefill pool and
decode pool are scheduled separately, so an SLO decision at admission cannot trade prefill
delay against decode delay -- they are different machines with different queues. Colocation
puts both on one node, so the router is choosing within a single queue discipline and can
express "take the cache hit, pay a little TTFT, still make the deadline". That trade is simply
not available to a disagg router.

**Risk to measure, not assume.** Cache-hit-first concentrates load on whichever nodes hold
popular prefixes. The SLO veto is the only thing preventing hotspotting, so the experiment must
report the **hit-rate vs p99-TTFT frontier**, not just mean TTFT. If the veto fires often, the
policy degenerates toward load-balancing and the advantage shrinks -- that is the honest
failure mode and it should be looked for.

## 2. Best-effort cache migration, and why colocation replicates better

When the veto rejects the best-hit node, we route to a worse-hit node **and** pull the missing
prefix to it, using sglang's hierarchical cache plus the distributed cache pool. The pull is
best-effort: if it does not complete before the request is scheduled, the node recomputes the
tail, so correctness never depends on the transfer and a slow link degrades latency rather
than breaking the request.

**The replication argument.** Every migration leaves a copy. Under cache-hit-first routing with
an SLO veto, popular prefixes are therefore *pulled toward wherever load forces requests to
go*, so replica count grows with demand for that prefix. This is the behaviour ThunderAgent
argues for in agentic serving, where the same long session prefix is revisited repeatedly and a
single-copy cache thrashes: the copy is evicted between turns and re-prefilled from scratch,
over and over.

Disagg has a weaker version of this. Its prefill pool holds the prefixes, and decode nodes do
not accumulate them, so replication is bounded by the prefill pool's capacity and the pool is
the thing that must be scaled to reduce thrashing. Colocation makes every node a potential
replica host, so the effective cache is the whole cluster's memory rather than one pool's.

**What must be measured.** Replica count per popular prefix over time, re-prefill rate (tokens
recomputed because of a miss), and whether migration traffic displaces useful work. The claim
"colocation replicates better" is a prediction; the number that would falsify it is a
re-prefill rate no better than disagg's at equal memory.

## 3. A more accurate EDF predictor, with acceptance length as an input

EDF only works if `predicted_service_time` is accurate; a scheduler with a bad predictor
misses deadlines it could have met and rejects requests it could have taken.

**Why colocation plus one attention kernel helps.** A colocated step is one batch through one
attention implementation whose cost is now **linear in total KV** (*measured*, above).

> **[!] CORRECTION 2026-10-10 — this premise holds for the UNFUSED colocated step, not the fused
> one.** With the two-role fused kernel the optimal `sm_split` moves with batch composition
> (128 / 96 / 64 at 2 / 8 / 32 decode rows) and a mis-set split costs **23x** (8.68 ms vs 0.367 ms
> at dec=32). A predictor would have to model the split too, over a target with a 23x
> discontinuity. The claim should rest on FA3-only colocation — one kernel, linear in KV, no split
> parameter — which is both simpler to model AND currently faster than the fusion (1.46–2.01x).
> See the 2026-10-10 section at the top of this doc. Features
that are known at admission:

    TTFT  ~ f(sum of append lengths, sum of context lengths, #prefill rows, #decode rows)
    TPOT  ~ g(#decode rows, sum of context lengths, predicted acceptance length)

A small MLP (two hidden layers, tens of units) over those features is sufficient; the point is
not model capacity but that the target is a smooth low-variance function. In disagg the same
prediction spans two machines, two queues and a KV transfer, so its variance is dominated by
cross-node effects the router does not observe.

**Acceptance length as a feature, and why it is fair.** With an MTP draft model the step cost
depends on how many drafted tokens are accepted. Online, the draft model is **fixed**, so
acceptance length is a slowly-varying property of the traffic, and an EWMA over recent accepts
estimates it without any oracle:

    a_hat <- (1 - alpha) * a_hat + alpha * a_observed        # per model, optionally per workload class
    TPOT_hat = g(..., a_hat)

This is legitimate because nothing about the *future* is assumed -- only that the recent past
predicts the next step, which is exactly the assumption every production scheduler already
makes about arrival rates. It should be reported with the EWMA's own error bars, and the
ablation that matters is `g` with and without `a_hat`.

**Validation plan.** Fit on one traffic mix, test on another (held-out ratio of chat to
TraceLab). Report predictor MAPE, then the thing that actually matters: SLO attainment under
EDF with this predictor against EDF with a naive linear model and against the unfused baseline
schedule. A better MAPE that does not improve attainment is not a result.

### Measured on the 4-mode MTP server — 2026-10-07 (H200, one GPU)

Setup: Qwen2.5-7B + EAGLE3 draft (3 steps, topk 1, 4 verify tokens), 4-mode step on the FA3/FA2
split (`MTPMIX_SPLIT`), `agent_chat_rated` at chat 24/s + codex 0.1 sessions/s, start barrier.
Driver `tasks/fused_specdec_workload/drivers/mtp_edf.sh 24 2000 50`.
Traced run: `SGLANG_STEP_TRACE=1 SGLANG_EDF_ACCEPT_TRACE=1`, 0 fallbacks, 0 tracebacks.

- [x] **Tokens per decode row per step are 2.10 on average, not 1.** bench_serving (and a user)
      measure ITL per token: a chunk's gap divided by its tokens. A step-time budget of
      `ITL_SLO` (a=1) is therefore about 2.1x too tight under MTP.
- [x] **Online estimators, scored against the tokens each step then emitted** (175 000
      row-steps, mean absolute error in tokens): a=1 **1.10**, global EWMA (alpha 0.2) **1.00**,
      per-request EWMA **0.89**. Per-step acceptance is noisy (1-4 tokens), so no estimator gets
      close to zero, but the per-request EWMA removes the 2.1x bias and is 19% closer than a=1.
- [x] **Step model** (`tasks/perf_model/fit_mtp_step.py`: linear in chunk tokens, chunk
      attention work, verify rows and verify KV; fit on the first half of 11 837 back-to-back
      steps, scored on the second): held-out MAPE **21.9%**, against **98.6%** for a constant
      (mixed steps 14.8%, pure-decode steps 23.4%).
- [x] **EDF attainment at a non-binding ITL SLO (2026-10-07).** A request attains its SLO when its
      TTFT and its P90 per-token ITL both meet target. ITL SLO 50 ms. The TTFT SLO is the one EDF
      schedules against: 2 s + 1 s per 5 000 prompt tokens beyond 4 096
      (`slo_attain.py --scaled`). Under a flat 2 s TTFT SLO, codex looks worse for every arm
      (baseline 0.861). One round per arm; 0 fallbacks and 0 tracebacks in every arm.

      | arm (tokens/step estimate) | chat SLO | chat goodput /s | chat TTFT ms | chat P99 ITL ms | codex SLO | codex TTFT ms | codex P99 ITL ms |
      |---|---|---|---|---|---|---|---|
      | no EDF | **0.876** | **20.62** | **446** | 59.4 | **1.000** | **978** | 49.6 |
      | EDF, a=1 | 0.117 | 2.07 | 40 883 | 56.8 | 0.111 | 55 815 | 69.3 |
      | EDF, global EWMA | 0.866 | 20.37 | 637 | 64.6 | 0.861 | 3 399 | **36.6** |
      | EDF, per-request EWMA | 0.696 | 16.36 | 1 264 | **53.5** | 0.778 | 4 468 | 47.3 |

      Reading:
      - **Without the acceptance term, EDF collapses on an MTP server.** a=1 spends the 50 ms
        per-token budget on one token per step. It refuses almost every prefill chunk, and chat TTFT
        reaches 41 s. The global EWMA restores attainment from 0.117 to 0.866. That is the
        ablation the design asked for, and the gain is large.
      - **EDF does not beat no-EDF at this SLO.** Without EDF, 93-100% of requests already meet
        the 50 ms ITL target, so the controller has almost no violations to prevent. It pays in
        TTFT and buys ITL headroom that the metric does not reward (codex P99 ITL 49.6 -> 36.6 ms).
      - **The lower-error estimator schedules worse.** Per-request EWMA has the lowest MAE (0.84
        vs 0.95 global in its own run), yet attainment is lower. The chunk budget is a MINIMUM
        over ~40 running requests' slack, and a minimum over noisy per-request estimates picks
        whichever request just saw a run of rejected drafts. The global estimate has no such
        order-statistic bias. Being checked with an `EDF_BIND` log (binding request's estimate vs
        global). This is the design section's warning in concrete form: better MAPE that does not
        improve attainment is not a result.
- [x] **Binding ITL SLO (30 ms per token) — the strict per-step controller loses (2026-10-07).**
      `TAG=_itl30 STEP_MODEL=/dev/shm/w/mtpedf/rate24/mtp_step.json mtp_edf.sh 24 2000 30`
      (commit 937792218). 0 fallbacks and 0 tracebacks in every arm. Scored with
      `slo_attain.py --scaled --itl 30`.

      | arm | chat SLO | chat ITL ok | chat TTFT ms | codex SLO | queue mean | running mean |
      |---|---|---|---|---|---|---|
      | no EDF | **0.706** | **0.723** | **433** | **0.972** | 3 | 16.3 |
      | EDF, a=1 | 0.019 | 0.575 | 163 063 | 0.028 | 1 345 | 20.3 |
      | EDF, global EWMA | 0.359 | 0.409 | 1 637 | 0.417 | 23 | 21.3 |
      | EDF, per-request EWMA | 0.016 | 0.478 | 90 187 | 0.028 | 234 | 24.3 |

      - **The order-statistic bias is confirmed.** `EDF_BIND` logs the tokens/step estimate of
        the request that sets the budget. Under the per-request EWMA it averages 1.12-1.19,
        while the global mean is 1.57-1.91. The minimum over ~20 running requests' noisy
        estimates is close to 1, so the arm behaves like a=1 and collapses the same way. Under
        the global EWMA the two are equal by construction (2.04).
      - **Even the global EWMA makes ITL worse than no EDF** (ITL attainment 0.409 vs 0.723),
        with a short queue (mean 23). So the cause is not queueing. The running batch grows
        (21.3 vs 16.3 rows) at about the same token throughput, and mixed steps rise 19%
        (3 800 vs 3 200). The per-step cap spreads the same prefill over more mixed steps. Each
        pays a fixed cost and slows every decode row, while the cap only sees this step's slack
        and ignores that deferred prefill still has to run later.
- [x] **Ablations: deadline ORDER helps; the per-step chunk CAP is what hurts (2026-10-07).**
      Same load, same SLO (ITL 30 ms, scaled TTFT), same command with `ARMS=...`; one round each.

      | arm | priority | chunk cap | chat SLO | chat ITL ok | chat TTFT ms | chat P99 ITL ms | codex SLO | codex P99 ITL ms |
      |---|---|---|---|---|---|---|---|---|
      | no EDF | FCFS | none | 0.706 | 0.723 | 433 | 58.8 | 0.972 | 43.5 |
      | **EDFP** | deadline | none | **0.750** | **0.750** | **178** | **53.6** | **1.000** | **28.9** |
      | EDFGC | FCFS | global EWMA | 0.381 | 0.499 | 2 657 | 58.4 | 0.861 | 43.3 |
      | EDFG | deadline | global EWMA | 0.359 | 0.409 | 1 637 | 56.5 | 0.417 | 52.3 |
      | EDFGQ | deadline | global EWMA, P90 allowance | 0.301 | 0.357 | 2 227 | 57.8 | 0.389 | 53.3 |

      - **Deadline ordering alone wins on every metric**: chat TTFT -59%, codex P99 ITL -34%,
        chat SLO 0.706 -> 0.750, codex 0.972 -> 1.000.
      - [x] **Replicated, 3 rounds** (`rate24_itl30{,_r2,_r3}`; rounds 2-3 used
        `ARMS="MTPMIX_SPLIT MTPMIX_SPLIT_EDFP"`, commit 863fb3212; 0 fallbacks and 0 tracebacks in all
        six runs). Mean [min-max] over the rounds:

        | metric | no EDF | EDFP (deadline order) | change |
        |---|---|---|---|
        | chat SLO attainment | 0.677 [0.658-0.706] | **0.735** [0.714-0.750] | +8.6%, ranges disjoint |
        | chat goodput /s | 15.93 | **17.31** | +8.6% |
        | chat mean TTFT ms | 479 [433-534] | **180** [172-191] | **-62%** |
        | chat P99 ITL ms | 60.3 | **54.1** | -10% |
        | codex P99 ITL ms | 49.2 [43.5-54.8] | **35.8** [28.9-39.7] | **-27%** |
        | codex mean TTFT ms | 1 018 | 1 457 | +43%, still inside its length-scaled SLO |
        | codex SLO attainment | 0.981 | **1.000** | |

        Codex pays in TTFT for chat's TTFT. That is the trade the SLO definition licenses: a
        40K-token prompt is allowed ~9 s.
      - [x] **Full stack vs stock sglang MTP, 3 rounds (2026-10-08).** The stack is the 4-mode step
        on the FA3/FA2 split plus deadline order (EDFP). Stock is `MTP_LPT`: same EAGLE3 draft, no
        prefill/verify mixing, FCFS. Same load (chat 24/s + codex 0.1 sessions/s), barrier. Stock
        rounds: `TAG=_itl30{,_r2,_r3} ARMS=MTP_LPT mtp_edf.sh 24 2000 30` (commit bf96b6cd3).
        0 fallbacks and 0 tracebacks; every request completed in every run. Means:

        | metric | stock MTP | full stack | change |
        |---|---|---|---|
        | chat TTFT ms | 3 684 | **180** | **-95.1%** |
        | chat mean ITL ms | 22.15 | **12.86** | **-41.9%** |
        | chat P99 ITL ms | 83.2 | **54.1** | **-35.0%** |
        | codex TTFT ms | 2 991 | **1 457** | **-51.3%** |
        | codex mean ITL ms | 11.25 | **7.01** | **-37.7%** |
        | codex P99 ITL ms | 54.5 | **35.8** | **-34.3%** |
        | chat SLO attainment (TTFT scaled, ITL 30 ms) | 0.216 | **0.735** | 3.4x |
        | chat goodput /s | 5.08 | **17.31** | 3.4x |
        | output tok/s (chat / codex) | 1 686.7 / 126.1 | 1 691.2 / 125.8 | equal: both serve the offered rate |

        Every TTFT and ITL metric clears the 30% bar on both tenants. Throughput cannot differ at
        a fixed offered rate below both arms' capacity. The throughput claim is the closed-loop
        capacity run (debug_kernel_server_perf_4_mode.md, v0.16.0: +19.8% total). Correctness: the
        4-mode split passed `drivers/srv_correctness.py`. EDFP changes only admission order, not
        the computation.
      - [x] **Same comparison at a moderate load, chat 16/s, 3 rounds (2026-10-08).**
        `TAG=_itl30{,_r2,_r3} ARMS="MTP_LPT MTPMIX_SPLIT_EDFP" mtp_edf.sh 16 2000 30`. All requests
        completed, 0 fallbacks. Means:

        | metric | stock MTP | full stack | change |
        |---|---|---|---|
        | chat TTFT ms | 256 | **104** | **-59%** |
        | chat mean / P99 ITL ms | 13.95 / 82.7 | **8.95 / 44.3** | **-36% / -46%** |
        | codex TTFT ms | 824 | 667 | **-19% (below the 30% bar)** |
        | codex mean / P99 ITL ms | 8.53 / 48.1 | **5.44 / 27.1** | **-36% / -44%** |
        | chat SLO attainment | 0.740 | **0.907** | |

        At this load stock MTP is not yet queueing codex, so there is less TTFT to remove. Codex
        TTFT misses the bar here and clears it at 24/s (-51%). Every ITL metric clears it at both
        loads.
      - **Any per-step chunk cap loses, with either ordering.** Splitting a chunk into k pieces
        adds k-1 mixed steps. Each costs ~7 ms plus 0.32 ms per running row (`mtp_step.json`) and
        delays every decode row. The deferred prefill still has to run, and under steady load
        there is no idle step to defer it into.
      - **Gating the cap on a P90 slow-step allowance does not rescue it.** More than 10% of steps
        already exceed 30 ms per token, so most requests use up their allowance early. From then
        on the cap applies, and it creates more slow steps.
      - **What this means for point 3.** The predictor does not earn its keep by throttling chunks
        inside a node. The accurate step model and acceptance EWMA matter where a decision is a
        real choice between alternatives, not a delay of work that must happen anyway: the
        router's admission/SLO veto across nodes (point 1). The in-node scheduler should order by
        deadline and stay work-conserving.

### Router support for point 3 (2026-10-14)

`tasks/pd_colocation/router.py` now predicts per-token TPOT under speculative decoding: each
worker keeps an EWMA of acceptance length from finished requests' `meta_info`
(`completion_tokens / spec_verify_ct`, `--accept-ewma-alpha`), and the SLO veto divides the
step-time prediction by it; `--step-model` loads the linear MTP step model from
`tasks/perf_model/fit_mtp_step.py` (incl. its `c_acc` term) -- the S4 refit on the split kernel
is what will feed it. Defaults leave the old per-step behaviour unchanged. Checks:
`router_accept_test.py` (EWMA and TPOT arithmetic, PASS) and `router_test.py` (fake workers;
same routing behaviour as the committed router on all four policies).

## 4. Two-way RDMA: a complete graph instead of a bipartite one

**The topology argument.** Disagg's KV traffic is structurally **bipartite**: prefill nodes
produce, decode nodes consume, and the cut between the two pools carries all of it. Colocation
makes every node both producer and consumer, so the transfer graph is **complete** -- any node
can serve any other. For the same total bytes, a complete graph spreads them over
`O(N^2)` links instead of `O(N_p * N_d)` across one cut, and it scales with `N` instead of with
the smaller pool.

This also removes the "few nodes hold the hot segment" failure mode: in disagg, a popular
prefix lives in the prefill pool and every decode node needing it pulls across the same cut.

**Congestion control, and a deliberate preference for the dumb algorithm.** When several nodes
hold hits within a threshold of each other, pick one **uniformly at random**:

    hits = [n for n in nodes if hit_len(n) >= (1 - eps) * max_hit_len]
    source = uniform_random(hits)

Randomisation needs no state, adds no latency, and spreads load across sources -- the standard
power-of-choices argument from load-balancing. Compared with modelling link congestion, it is
far more robust for this workload:

- a congestion model needs per-link state for links we **rarely use**, and those estimates are
  the stalest exactly when we suddenly need them;
- querying a central router adds a round trip on the critical path of a cache pull, which is
  latency we are trying to save;
- classic congestion-aware routing (ECMP's hashing, CONGA's leaf-to-leaf load sensing,
  HULA/Hedera-style adaptive schemes) works because switches observe link utilisation
  continuously and locally; our router observes neither continuously nor locally.

So the plan is: **random-among-ties first**, and treat congestion-aware source selection as an
ablation that must *beat* random to justify its state and its round trip. "Random is within
noise of the modelled policy" is a perfectly good outcome and should be reported as such.

**What must be measured.** Per-link utilisation spread (max/mean), p99 cache-pull latency, and
whether aggregate attainment degrades more slowly with `N` than disagg's -- the scaling claim
is the real one, and it needs at least 4 nodes to say anything, hence the H20 budget (<=16 GPUs,
4x4 or 8x2, trial `vibecuda`).

---

## 5. Agentic append rounds make disagg re-ship the session's KV every round — 2026-10-13

(User point, 2026-10-12.) An agent session is many short *append* rounds against a long, growing
context: tool result appended, a few hundred output tokens, repeat. Under disaggregation each
round is a new request: the prefill node computes the appended tokens and the decode node must
receive the request's KV before its first decode step. Stock sglang PD frees a request's KV on
the decode node when it finishes, so every round ships the **whole context**; even a decode
node that kept the session prefix would still ship every round's **append**. Under colocation
the session's KV stays on the node that decodes it (cache-hit-first routing, point 1), so a
round moves **nothing** and only computes its append.

**Measured from the full TraceLab release** (353,598 follow-up rounds in 4,109 sessions; rounds
per session p50 18, p90 163, p99 1209; context p50/p90/p99 124K/256K/823K tokens; append
p50/p90/p99 860/6.5K/105K tokens). Transfer time per round at line rate, no contention:

| model (KV bytes/token) | link | disagg, full KV per round p50 / p90 / p99 | disagg, append only p50 / p90 / p99 | colocation |
|---|---|---|---|---|
| Qwen2.5-7B (57.3 KB) | 200 Gb/s | 285 / 587 / 1887 ms | 2.0 / 14.8 / 240 ms | 0 |
| Qwen2.5-7B (57.3 KB) | 400 Gb/s | 143 / 294 / 944 ms | 1.0 / 7.4 / 120 ms | 0 |
| 70B GQA-8 (320 KB) | 200 Gb/s | 1.63 / 3.36 / 10.8 s | 11 / 85 / 1373 ms | 0 |
| 70B GQA-8 (320 KB) | 400 Gb/s | 0.81 / 1.68 / 5.39 s | 6 / 42 / 687 ms | 0 |

Over the whole release that is 3.12 PB moved (full KV) against 107 TB (append only) for a 7B
model -- 29x -- and colocation moves neither.

**Where it shows up:** in **TTFT** of every round (the transfer precedes the first decode
step), and in **ITL of the requests already decoding** on the receiving node, because each
incoming transfer competes with them for the NIC and for HBM write bandwidth. Disagg's usual
mitigation -- layer-wise streaming overlapped with prefill -- hides transfer behind *prefill
compute*, but an append round has almost none (median 860 tokens), so there is nothing to hide
it behind. This complements points 1 (keep the session where its cache is) and 2 (when a session
must move, it moves once, not every round).

**Prior art that already blunts this, stated plainly.** NVIDIA Dynamo's *conditional
disaggregation* prefills locally on the decode worker when the uncached part of the prompt is
small (default effective-ISL threshold 4096 tokens) or remote prefill is loaded; with the median
append at 860 tokens and the session prefix resident on its decode worker, most agent rounds
then move no KV at all, and a proposed delta-KV RFC would cut the remaining remote rounds to
append-only transfer (the 1-15 ms column above, not 143-587 ms). semi-PD avoids transfer
entirely by splitting SMs inside one GPU over a shared KV store. So point 5 is decisive against
stock sglang PD and against Dynamo's remote-prefill path; against Dynamo with conditional
disaggregation the advantage that remains is (a) making the local prefill + speculative decode
co-execution efficient -- that is our 4-mode step and fused kernel, measured at +20-36%
throughput and -31..-79% TTFT over stock MTP under load -- (b) symmetric nodes instead of fixed
prefill/decode pools, so routing and cache replication (points 1-2) can use every node, and
(c) EDF/SLO-aware admission (point 3). Dynamo with conditional disaggregation must be run as a
baseline rather than argued around.

**What must be measured (H20, <= 16 GPUs, trial `vibecuda`):** full-length TraceLab replay
(round-weighted session sampling, both providers) on sglang PD (e.g. 2P2D) vs colocation with
cache-hit-first routing, at matched rates; report per-round transferred bytes, round TTFT, ITL
of co-resident decodes on decode nodes, and NIC utilisation. Not run yet. The single-node
full-length run (S2, H200) measures the colocation side of this first.

---

## 6. NVIDIA Dynamo's scheduling policy vs ours, in detail — 2026-10-13

Sources: Dynamo docs -- [KV router routing concepts][dyn-router], [conditional
disaggregation][dyn-cd] (v1.5.1, experimental), [planner][dyn-planner] (and the older [SLA
planner][dyn-sla]), [KVBM design][dyn-kvbm] / [overall architecture][dyn-arch], and the [delta-KV
transfer RFC][dyn-rfc]; full list under [References](#references).

**How Dynamo schedules, layer by layer:**

| layer | how Dynamo decides |
|---|---|
| **Router** (per request) | Picks the worker with the lowest block-count cost: pending prefill blocks minus a prefix-hit credit, plus active decode blocks, plus an optional per-request charge. Cache hits are just a load discount. There is no SLO or deadline term anywhere, and no time model. |
| **Queueing** | Weighted round-robin across request classes, then the engine's own first-come order. No deadline ordering. |
| **Conditional disaggregation** (experimental) | Separate prefill and decode pools. A request prefills locally on its decode worker only below static thresholds: uncached prompt under 2048 tokens and under 70% of the prompt, or when the prefill workers are busy. The local prefill then runs in the engine's ordinary mixed batch. It doesn't support SGLang (vLLM and TensorRT-LLM only). |
| **Planner** | Rescales the number of prefill and decode replicas about every 3 minutes to meet TTFT/ITL targets. The older SLA planner works only with disaggregated pools. |
| **KV placement** | KV is offloaded across memory tiers and fetched point to point when needed. Copies of hot prefixes exist only where routing and transfers happened to leave them. |

**Where ours differs:**

- **Routing:** we rank nodes by cache hit, then veto any node where a step-time model plus the
  earlier-deadline requests queued there predict a missed SLO (point 1). Dynamo can send a request
  to a cache-hot worker that will miss its deadline.
- **Load model:** we predict step time from a model that includes speculative acceptance length
  (point 3). Dynamo counts KV blocks, which ignore context length and acceptance.
- **Prefill placement:** every node prefills and decodes, and each step co-runs prefill chunks
  with speculative verify work using the fused kernel. That is exactly the local path Dynamo
  leaves to an ordinary batch, and the one we measured at TTFT -31% to -79% and throughput +11%
  to +36% versus stock MTP under load.
- **Reacting to shifts:** the prefill/decode balance is re-chosen every step, against Dynamo's
  roughly 3-minute replica rescaling.
- **Cache:** we deliberately replicate hot prefixes through migration (point 2) and spread pulls
  across near-equal holders (point 4). Dynamo's copies are incidental.

**Worth adopting from Dynamo:** hit credits weighted by memory tier, a decaying estimate of
prefill load, and output-length hints for agent rounds.

**Baseline:** a fair comparison must run on vLLM, since Dynamo's conditional disaggregation
doesn't support SGLang: conditional disaggregation tuned to the full-length TraceLab distribution,
plus Dynamo's router in aggregated mode as a routing-only control. Both are H20 work. Per the user
(2026-10-13) this comparison is **optional** because it lacks an SGLang backend; sglang PD
disaggregation is the required disaggregation baseline.

---

## Honest summary of standing

The kernel-level result that motivates all of this (**1.225x on a TraceLab-shaped batch**) is
real but is batch time, not serving. The publishability bar is **>=30% TTFT/ITL and >=10%
throughput on a realistic workload**, and there is **no serving data at all** yet. The gating
blocker is the split-row corruption in point 0 above, because serving batches routinely contain
rows above the chunk limit, so a serving A/B today would measure a corrupted baseline.

Order of work: fix the corruption, run the serving A/B, then the four experiments above in the
listed order.

---

## References

Systems and papers this doc argues against or borrows from. *(sent)* marks the ones the user sent;
the kernel-side reference list is at the end of `debug_kernel_server_perf_4_mode.md`.

- **NVIDIA Dynamo** -- the main baseline for points 1, 5 and 6.
  - [KV router: routing concepts][dyn-router] -- note: block-count cost with a prefix-hit credit,
    DRR across policy classes, no SLO/deadline term (point 6).
  - [Conditional disaggregation][dyn-cd] -- note: local prefill on the decode worker below a static
    uncached-ISL threshold (2048, ratio 0.7) or when prefill is busy; vLLM/TRT-LLM only (points 5, 6).
  - [Planner][dyn-planner] and the older [SLA-based planner][dyn-sla] -- note: minute-scale P/D
    replica autoscaling from profiled performance and load prediction (point 6).
  - [KVBM design][dyn-kvbm], [overall architecture][dyn-arch] -- note: G1-G4 KV tiers, NIXL
    point-to-point transfer, incidental rather than deliberate replication (points 2, 6).
  - [Delta-KV transfer RFC][dyn-rfc] -- note: would cut remote agent rounds to append-only
    transfer (point 5).
- **semi-PD: Towards Efficient LLM Serving via Phase-Wise Disaggregated Computation and Unified
  Storage** -- K. Hong et al., [arXiv:2504.19867][semipd] ([code][semipd-code]) -- note: SM-level
  prefill/decode split inside one GPU over a shared KV store; static ratio, relaunch to change.
- **Towards High-Goodput LLM Serving with Prefill-decode Multiplexing (MuxWise)** -- Y. Chen,
  W. Cui et al., [arXiv:2504.14489][muxwise] *(sent)* -- note: intra-GPU P/D multiplexing with an
  SLO-aware dispatcher; direct colocation baseline.
- **Crossflow: Prefill–Decode Elasticity for Agentic LLM Serving** -- Y. Xu et al.,
  [arXiv:2609.27085v1](https://arxiv.org/abs/2609.27085v1) *(sent)* -- note: direct elastic-P/D
  baseline for points 1–4, not merely static disaggregation. It retains decode homes and lends
  bounded local-prefill capacity through leases and SLO feedback; our symmetric placement,
  replication and network benefits remain hypotheses requiring matched comparison.
  [Detailed analysis](debug_kernel_server_perf/resume_20261012/crossflow_analysis_20261014.md).
- **ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System** -- H. Kang et al.,
  [arXiv:2602.13692][thunder] *(sent)* -- note: KV-cache thrashing in agentic serving; motivates the
  replication argument (point 2).
- **TraceLab: Characterizing Coding Agent Workloads for LLM Serving** -- K. Zhu et al.,
  [arXiv:2606.30560][tracelab] ([dataset][tracelab-gh]) *(sent)* -- note: the agentic workload and
  the per-round transfer numbers of point 5.
- **SLOs-Serve: Optimized Serving of Multi-SLO LLMs** -- S. Chen et al., [arXiv:2504.08784][slos]
  *(sent)* -- note: per-application SLOs and mixture x load evaluation (point 3, experiments 5c/5d).

[dyn-router]: https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/routing-concepts
[dyn-cd]: https://docs.nvidia.com/dynamo/advanced-customizations/conditional-disaggregation
[dyn-planner]: https://docs.nvidia.com/dynamo/v1.5.0/knowledge-base/modular-components/planner/overview
[dyn-sla]: https://docs.nvidia.com/dynamo/v-0-8-1/components/planner/sla-based-planner
[dyn-kvbm]: https://docs.nvidia.com/dynamo/v-0-9-1/design-docs/kvbm-design
[dyn-arch]: https://docs.nvidia.com/dynamo/v1.1.1/design-docs/overall-architecture
[dyn-rfc]: https://github.com/ai-dynamo/dynamo/discussions/7861
[semipd]: https://arxiv.org/abs/2504.19867
[semipd-code]: https://github.com/infinigence/Semi-PD
[muxwise]: https://arxiv.org/abs/2504.14489
[thunder]: https://arxiv.org/abs/2602.13692
[tracelab]: https://arxiv.org/abs/2606.30560
[tracelab-gh]: https://github.com/uw-syfi/TraceLab
[slos]: https://arxiv.org/abs/2504.08784
