# Realistic Workload Analysis for the Persistent Overlap Kernel

## Scope

This document contains the current, correctness-valid evaluation of persistent
prefill/decode overlap. Historical B200 scans, graph-incompatible runs,
median-only latency tables, synthetic prioritization notes, and superseded
arrival-rate surveys were removed; they remain available in Git history.

The paper-facing comparison has three methods (plus one variant):

- **PP:** prefill-prioritized scheduling with unchanged BatchPrefill.
- **DMC:** default mixed-chunk scheduling with unchanged BatchPrefill
  (`SGLANG_FLASHINFER_USE_PERSISTENT=0` set explicitly — the flag defaults on).
- **LPT:** mixed scheduling with the gated persistent overlap kernel
  (`SGLANG_MIXED_ATTENTION_KERNEL=auto`, measured k-NN gate) at the matched
  8,192-token chunk cap; 6,656 only where marked as the tail-optimized point.
- **LPT_EDF:** LPT plus the SLO-aware chunk controller driven by the analytic
  overlap cost model (`SGLANG_EDF_LATENCY_MODEL=analytic`,
  `SGLANG_EDF_PLAN_OVERHEAD_MS=0`).

Ordinary decode CUDA graphs are enabled for every method. Piecewise EXTEND
graphs are not combined with decode graphs because that combination is
output-incorrect at N=400. LPT has no minimum prefill chunk. Every reported
point retokenizes within 2% of its requested output.

The story order below: pure long-context replay (parity — the zero-regression
check), the occupancy scan that locates where the kernel matters, why
independent-prompt datasets cannot show it, and the mixed-tenancy scenarios
where it wins end to end, including SLO control with the analytic cost model.

## TraceLab causal replay: the zero-regression check

The headline setting is one H800 with Qwen2.5-7B-Instruct, 400 causal rounds
from ten real Claude coding sessions, 64K–100K contexts, mean 3,812-token
append, mean 481-token output, and recorded think/tool gaps compressed 100x.
All ten sessions are admitted together (`--request-rate inf`). Warmup uses
same-length but token-disjoint prompts, so it exercises kernels and memory
allocation without pre-populating the timed sessions' radix prefixes. The
server seed is fixed to one.

The replay preserves session identity, consecutive round ordering, prefix
growth, tool-result prefills, output lengths, causal dependencies, and the
relative timing of inter-round gaps. It is not a bag of independent prompts.

### Offered-load sweep (honest baselines, 2026-07-31)

For causal TraceLab replay, `--request-rate` is the Poisson arrival rate of
**new sessions**; later rounds remain causal. The sweep uses N=400 with fixed
server seed one, identical sampled rounds, ordinary decode CUDA graphs for
both methods, and token-disjoint warmup. The rates below are sessions/s;
`inf` admits all ten sessions together.

Earlier versions of this sweep (in git history) compared two servers that
both ran the persistent kernel (the DMC launcher never disabled
`SGLANG_FLASHINFER_USE_PERSISTENT`) and both paid the plan()-synchronize
stall. The current sweep ran with sglang
`f158e4b85` / flashinfer `c6d2a00`: true BatchPrefill DMC versus gated LPT
(`SGLANG_MIXED_ATTENTION_KERNEL=auto`, matched 8,192 cap), raw runs in
`h800_qwen25_tracelab_claude_pareto_gated_n400/`. The honest baseline is
itself 6.1–10.7% faster than the contaminated one.

| Rate | Method | Total tok/s | Mean / P99 TTFT | Mean / pooled P99 ITL |
|-----:|--------|------------:|----------------:|----------------------:|
| 0.02 | true DMC | 28,907 | **725 ms** / 7.87 s | 17.83 / 96.12 ms |
| 0.02 | gated LPT | 28,903 | 758 ms / **7.28 s** | **17.73 / 93.54 ms** |
| 0.05 | true DMC | 30,775 | 892 ms / 9.92 s | **18.25** / 98.99 ms |
| 0.05 | gated LPT | **30,794** | **816 ms / 9.55 s** | 18.38 / **98.59 ms** |
| `inf` | true DMC | 31,558 | **1.29 s / 31.0 s** | 18.33 / 100.41 ms |
| `inf` | gated LPT | **31,643 (+0.27%)** | 1.32 s / 33.0 s | **18.25 / 95.83 ms (−4.6%)** |

The gated build is parity or better on throughput and pooled P99 ITL at every
rate (P99 ITL −2.7%/−0.4%/−4.6%); saturation TTFT differences are within the
±6% run-to-run band measured in paired repeats. The gate selected the
persistent kernel for 63/500 (0.02) to 203/1500 (inf) mixed batches. The
figures below are regenerated from this run.

Current corrected TraceLab figures:

- [Capacity versus mean ITL](realistic_workload_analysis/tracelab_current_n400/figures/pareto_capacity_vs_mean_itl.png)
- [Pooled P99 ITL versus session rate](realistic_workload_analysis/tracelab_current_n400/figures/p99_itl_vs_rate.png)
- [TTFT SLO attainment](realistic_workload_analysis/tracelab_current_n400/figures/slo_ttft_attainment_vs_rate.png)
- [Per-request P99-ITL SLO attainment](realistic_workload_analysis/tracelab_current_n400/figures/slo_itl_attainment_vs_rate.png)

The compact source is
[`summary.json`](realistic_workload_analysis/tracelab_current_n400/summary.json);
multi-megabyte per-request JSON and server logs remain local benchmark
artifacts.

## Workload sweet-spot scan

The kernel needs simultaneous long decode and intermittent prefill, not merely
long contexts: long decode makes attention dominant, but persistence helps end
to end only while an intermittent append arrives against enough active decodes
to form repeated mixed forwards. The screen varies real causal session counts
(N=100, output ≥512 tokens, fixed seed, disjoint warmup, 64K–100K contexts).
Configured session count is not itself a controlled offered-load coordinate —
a twelve-session sample reached only 1.57 observed concurrency because its
real windows contained long tool/think gaps. A pre-fix version of this scan
(git history) ran against the contaminated baseline and misplaced the win
boundary at "concurrency ≈ 11".

### Gated-stack sweet-spot rescan (2026-07-31, honest baselines)

The scan was repeated at g12/g14/g18/g20 with the fixed stack (sync-free
plan, kernel gate, prewarm; sglang `f158e4b85`, flashinfer `c6d2a00`),
true-BatchPrefill DMC, gated LPT at the tail-optimized 6,656 cap, plus the
earlier validated g10/g16 screens. Results dirs:
`h800_tracelab_decodeheavy512_gated_g*_n100/`.

| Groups | Obs. conc | Mean ITL regime | LPT tput | pooled P99 ITL | reqmax P99 | worst ITL (LPT / DMC) |
|-------:|----------:|-----------------|---------:|---------------:|-----------:|----------------------:|
| 12 | 1.4 | 17.8 ms (light) | −0.07% | **+72%** (82 vs 47 ms) | −14% | 0.88 / 1.02 s |
| 10 | 3.2 | 17.5 ms (light) | −0.11% | +11.6% | −18% | 0.89 / 1.08 s |
| 14 | 2.7 | 44 ms (heavy) | −0.20% | **−15.7%** | −14% | 0.87 / 1.02 s |
| 18 | 8.2 | 52 ms (heavy) | **+0.64%** | **−16.0%** | **−44%** | **0.89 s / 53 s** |
| 16 | 10.9 | 43 ms (heavy) | **+0.77%** | **−14.6%** | −12% | 1.05 / 32 s |
| 20 | 14.5 | 47 ms (heavy) | −0.39% | **−15.7%** | **−67%** | **0.89 s / 82 s** |

The discriminator is the **mixed-step interference regime**, not raw
concurrency: whenever the sample keeps mixed chunks colliding with decodes
(mean ITL ≈ 43–52 ms; g14/g16/g18/g20), the gated stack delivers a uniform
−15 to −16% pooled P99 ITL at parity-to-+0.8% throughput, cuts request-max
tails by up to 67%, and replaces DMC's 32–82-second worst stalls with
sub-second worst gaps. In light-mixing samples (mean ITL ≈ 17.5 ms;
g10/g12), the 6,656 cap costs pooled P99 (absolute 35–82 ms, still under
the 150 ms SLO) while request-max stays 14–18% better; the matched 8,192
cap restores strict parity there (validated at g10: pooled P99 −6.2%).

The sweet spot is therefore g16–g18 (concurrency ~8–11 under heavy mixing):
simultaneous throughput, pooled-tail, and worst-stall wins. Notably the win
region is much broader than the pre-fix scan suggested — g14 (conc 2.7) and
g20 (conc 14.5) also win the tail decisively; the old "only wins near
concurrency eleven" boundary was mostly plan-sync overhead and the
contaminated baseline. Gate picks stay modest everywhere (1–4% of mixed
batches persistent at long context), so the tail gains come from the gate
keeping BatchPrefill where it is faster while the cap shapes interference.

## Transfer to independent coding datasets

Matched N=400 sweeps use Qwen3-8B and ordinary decode CUDA graphs for every
method (Code-Feedback 4–24 req/s plus saturation; SWE-Gym 0.5–4 req/s plus
saturation; raw runs in `tasks/edf_benchmark/results_valid_n400/`). DMC and
LPT land within fractions of a percent of each other on throughput and mean
ITL at every rate — parity, not a win, and expectedly so.

These datasets contain independent prompts averaging only about 242 and 690
tokens. They omit the multi-turn intermittent tool-result prefills that create
repeated mixed forwards in TraceLab, so the persistent kernel is active during
too small a fraction of server time to materially shift the full curve.

## Mixed-tenancy workloads (2026-08-05): the win scenario, end to end

`run_mixed_workloads.py --scenario agent_plus_shortchat`: one H800 serving a
coding-agent tenant (12 causal TraceLab sessions @32–48K, rounds issued
back-to-back, 60 requests) beside a chat tenant (2,400 fresh 512-token
prompts at 12 req/s, 96-token outputs). True-BatchPrefill DMC baseline vs the
gated persistent path (merged 1,316-point table with the KV-concentration
feature). Identical token volumes, 100% retokenization on every stream.

| Tenant | Metric | true DMC | gated LPT | delta |
|--------|--------|---------:|----------:|------:|
| codex  | total tok/s | 13,809 | **18,844** | **+36.5%** |
| codex  | wall time   | 188 s | **138 s** | −27% |
| codex  | mean ITL    | 43.4 ms | **40.5 ms** | −6.7% |
| chat   | mean ITL    | 27.3 ms | **15.8 ms** | **−42.2%** |
| chat   | P99 ITL     | 80.1 ms | **61.7 ms** | −23.0% |
| chat   | mean TTFT   | 38.2 s | **24.1 s** | −37% |

Mechanism, confirmed by full-run nsys captures of both methods: mixed
forwards are 14–30% of all forwards in this scenario, and the gated path
spends 112 s of mixed-step wall time where DMC spends 157 s (DMC's worst
mixed step stalls 13.9 s; the persistent kernel's per-instance max is
0.9 ms). The coding tenant converts that directly into throughput; the chat
tenant stops waiting behind slow mixed steps.

**In-band multi-turn + SLO control (2026-08-11).** `agent_plus_multiturn`
(2K follow-up turns on 8K shared prefixes over the coding pool) drives 49% of
forwards mixed — the balanced band. At deliberately saturated load (all arms
queue 18–77 s of TTFT), adding the closed-form-model EDF controller
(`LPT_EDF`: `SGLANG_EDF_LATENCY_MODEL=analytic`, ITL SLO 100 ms) pins P99 ITL
almost exactly at the budget it was told to spend — codex 497→113 ms (−77%),
turns 258→107 ms (−59%), turns per-request SLO attainment 1.3%→54.8% — at the
expected cost of TTFT/throughput for the deferred prefill (the correct
tradeoff beyond capacity; no controller can also absorb an infeasible queue).
The controller's guard is set to 0 for the analytic model, which already
carries fixed overheads in `c0`; the adaptive 5–10 ms guard predates the
plan-sync fix and would double-count. The feasible-load variant
(`agent_plus_multiturn_feasible`) replicates the controller property at a
second operating point — pooled P99 ITL pinned at 104–105 ms against the
100 ms budget (codex −77%, turns −51%; codex request-max P99 645→150 ms) at
identical node-wide throughput (23,319 vs 23,317 tok/s) — but tenant-level
TTFT/attainment comparisons there are confounded: the codex tenant runs at
rate=inf, and with an infinite-backlog prefill source present, FCFS
admission ordering rather than compute dominates the other tenant's latency
(LPT's turns TTFT moves 0.7→6.6 s at identical node throughput and identical
turns tput). The clean experiment (`multiturn_slo_clean`, both streams rate-limited,
2026-08-11) closes the loop: with identical arrivals and node throughput
matched to four digits across all three arms (9,368/9,369/9,368 tok/s),
the gated kernel alone improves codex pooled P99 ITL −28.3% (81→58 ms,
turns −8.1%) at parity elsewhere, and adding the analytic-model controller
pins turns pooled P99 at **100.24 ms against the 100 ms budget** while
cutting request-max P99 ITL **−60% on both tenants** (519/522→208/208 ms),
with unchanged throughput. Across the three operating points the achieved
pooled P99 converges onto the configured budget (113 → 105 → 100.2 ms) as
load moves from saturated to feasible — the controller delivers exactly the
tail it is asked for whenever the load permits, which is the operational
meaning of a 3%-accurate cost model. A recurring
plain-LPT turns anomaly (request-max 1.4 s, retok 98.9%) matches the
retraction-cascade signature — third sighting.

The `shortchat_only` control (3,000 fresh 512-token prompts at 16 req/s, no
long-context tenant) lands at parity — −0.02% throughput, −0.45% mean ITL —
with the gate selecting persistent for only 6 of 2,566 mixed batches. The
gate takes the win where the composition supports it and declines everywhere
else, which is the zero-regression contract.

One LPT trial (archived in `h800_mw_agent_plus_shortchat/run1_regressed/`)
showed a −19% episode whose damage was concentrated in the first two
arrival-deciles of the codex prefill storm (chat ITL 98.7 ms early vs
15–16 ms steady-state) with 98.5% retokenization, consistent with a
KV-retraction cascade compounding at fixed arrival rate. The identical
configuration reproduces the win in repeat trials and under nsys; the
baseline is stable across all trials. Multi-trial runs are the suite's
default defense against episode-vs-property confusion.

- [Capacity versus mean ITL](edf_benchmark/current_policy_n400/pareto_capacity_vs_mean_itl.png)
- [Pooled P99 ITL versus offered rate](edf_benchmark/current_policy_n400/p99_itl_vs_rate.png)
- [TTFT SLO attainment](edf_benchmark/current_policy_n400/slo_ttft_attainment_vs_rate.png)
- [Per-request P99-ITL SLO attainment](edf_benchmark/current_policy_n400/slo_itl_attainment_vs_rate.png)

TTFT and ITL attainment are kept separate. The continuous P99-ITL and
capacity/interactivity curves are the primary evidence; SLO attainment is a
threshold-sensitive diagnostic. Pooled P99 ITL is accompanied by
request-maximum ITL because pooled token percentiles hide rare multi-second
stalls.

## Which attention kernel wins, and does our workload answer that? (2026-09-27)

Full kernel-level detail (planner structure, isolated-composition
measurements, FA3 levers tried) is consolidated in
`tasks/debug_kernel_server_perf.md` → "Ours vs FA3: the complete picture"; this
section carries the serving-level verdict only.

Kernel-level replay of the realized mixed-batch distribution
(`sweep_from_capture.py`, time-weighted over 2,845 real batches) gives
**FA2 1.00× / FA3 0.67× / persistent 2.60×**. That reads as a decisive win for
the persistent kernel, but it is a statement about *this* workload, and the
workload has a bias that matters.

**The bias: our captured chunks do almost no prefill work** — though *not*
because the cache is missing. Those runs hit the radix cache **79.7%** in token
terms with peak KV occupancy 0.61 (no eviction). The per-row distribution is
bimodal: the coding tenant's continuations hit heavily but are few rows (3 of
43 sampled mixed batches carry `cached-token > 512`, and they hold 3.8M of the
cached tokens), while the chat tenant's fresh random prompts — which have no
reusable prefix by construction — dominate the row count. Hence attended prefix
2 tokens at p50, **94% of rows below 512**, and a median chunk doing ~550
token-pairs of attention, ~**30,000×** less than a 2,048-token chunk against an
8K prefix. It is a property of the scenario's tenant mix, not of caching and
not of serving in general. Splitting the same replay by attended prefix shows
the comparison inverting as prefill work appears:

| attended prefix | weight | FA3/FA2 | persistent/FA2 | persistent/FA3 |
|-----------------|-------:|--------:|---------------:|---------------:|
| < 512 (fresh) | 2,793 | 0.63× | **3.95×** | **6.26×** |
| ≥ 8K | 52 | 0.95× | 1.02× | **1.08×** |

In the long-prefix band all three kernels are within 8% of each other — but it
carries only 2% of the weight here, so the headline 2.60× is set almost
entirely by the fresh-prefix band. Measured directly on prefill-dominant
batches, **FA3 is 1.35–1.54× faster than FA2 while the persistent kernel is
0.89–1.00× of FA2** (`tasks/debug_kernel_server_perf.md`, "FA3 on
prefill-dominant batches"). So the ordering genuinely reverses by regime:

- **decode-heavy, fresh/short chunks** → persistent wins, by a lot (3.2–5.1×)
- **prefill-dominant, long attended prefix** → FA3 wins (1.35–1.54×), ours is
  at parity-or-worse than FA2
- **crossover** around a handful of decode rows against a real chunk

**Verdict, measured on a workload that does have prefix reuse.** The above is
settled by capturing `agent_plus_multiturn_feasible` (multi-turn on 8K shared
prefixes + agent sessions, at the SLO-feasible operating point) and replaying
it: 409 mixed batches with **attended prefix p50 8,592, 56% ≥ 8K, chunk p50
2,598, 18.6M token-pairs of prefill work per chunk** — 34,000× the fresh
capture — alongside a live decode population (rows p50 12, p90 47) and KV
concentration 28.0K. Time-weighted over 365 replayed batches:

| policy | total kernel ms | vs FA2 |
|--------|----------------:|-------:|
| FA2 | 958,499 | 1.00× |
| **FA3** | 1,410,339 | **0.68×** |
| **persistent** | 713,424 | **1.34×** |
| deployed gate | 813,294 | 1.18× |
| oracle per-batch best-of-3 | 664,749 | **1.44×** |

**The persistent kernel still wins overall — but by 1.34×, half the 2.60× it
showed on the fresh-prefix workload.** Adding real prefill work closes most of
the gap, exactly as expected. FA3 remains 0.68× in aggregate because these
batches still carry a decode population.

The decode-row split is where the answer actually lives:

| split | weight | FA3/FA2 | persistent/FA2 | persistent/FA3 |
|-------|-------:|--------:|---------------:|---------------:|
| 1–3 decode rows | 74 | **1.39×** | 0.98× | **0.70×** |
| 4–15 rows | 127 | 1.30× | **1.48×** | 1.14× |
| 16–47 rows | 164 | 0.55× | **1.34×** | 2.43× |
| prefix < 512 | 76 | 0.52× | 3.26× | 6.22× |
| prefix 512–8K | 83 | 0.53× | 1.43× | 2.68× |
| prefix ≥ 8K | 206 | 0.78× | 1.20× | 1.55× |

**FA3 wins outright on prefill-dominant batches** — at 1–3 decode rows it is
1.39× over FA2 and **1.43× over ours** (persistent/FA3 = 0.70×). The crossover
is around **four decode rows**; above it ours wins, decisively so past 16.
FA3's aggregate loss is therefore not a statement that FA3 is a weak kernel, it
is that this workload spends most of its time in batches with many decode rows.

Two consequences. First, **the honest claim is regime-scoped**: ours wins the
mixed-batch regime a busy colocated node actually occupies, FA3 wins the
prefill-dominant corner. Second, **a three-way gate is worth building**: the
oracle best-of-3 is 1.44× against 1.34× for always-persistent, so ~7% is left
on the table by not considering FA3 at all — and the deployed gate only
realizes 1.18×, so there is more headroom in selection than in any single
kernel.

**Does our workload meet the "realistic mixed workload" bar?** The co-tenancy
capture alone does not; the prefix-reuse capture does. Both are needed:

- *Realistic in composition* — co-tenancy of a coding agent with chat is a real
  deployment, the request stream is a real trace, and the decode-side mixture
  (46 rows, high KV concentration) is what a busy node looks like.
- *Not realistic in prefix reuse per row* — 94% of prefill rows attend <512
  tokens (the chat tenant has no reusable prefix by construction), which is
  exactly the regime where FA3 cannot earn its tile. Note this is a row-mix
  effect, not a cache failure: token-level hit rate is 79.7%. A production node with radix-cache hits or multi-turn traffic
  sits much further right on the prefix axis. This is the same grid-vs-realized
  sampling problem found earlier, now on the prefix axis rather than the chunk
  axis.
- *Fixed by the second capture* — `agent_plus_multiturn_feasible` supplies the
  missing half (prefix p50 8,592, 56% ≥ 8K, SLO-feasible arrival rates), and it
  is the one that should be quoted for kernel-selection claims. Reporting only
  the fresh-prefix number overstates the persistent kernel by ~2× (2.60×
  vs 1.34×).

**Under common SLOs the kernel choice matters less than these ratios suggest.**
The gate-v3 A/B is the cleanest evidence: a **1.90× → 3.50×** improvement in
kernel policy on mixed batches produced **+0.0% node throughput**, moving only
tails (chat P99 ITL −43.1%; codex P99 +7.2%, inside the ±6% run-to-run band).
Mixed batches are a minority of forwards and attention a fraction of step time,
so kernel wins show up in the ITL tail — where an ITL SLO is actually spent —
rather than in capacity. A kernel-selection decision aimed at an SLO should
therefore be made on tail latency in the regime the node actually occupies, not
on aggregate kernel speedups.

## Reproduction

```bash
# Headline replay
bash tasks/realistic_workload_analysis/run_tracelab_claude.sh

# Offered session-rate sweep
bash tasks/realistic_workload_analysis/run_tracelab_pareto.sh

# Decode-heavy peak-load screen (real causal TraceLab sessions)
TRACELAB_MIN_OUTPUT_LEN=512 TRACELAB_NUM_GROUPS=16 \
  NUM_PROMPTS=100 LPT_CHUNKED_PREFILL_SIZE=6656 \
  bash tasks/realistic_workload_analysis/run_tracelab_claude.sh

# Analyze and plot accepted results
python tasks/edf_benchmark/analyze_results.py RESULTS_DIR
python tasks/edf_benchmark/plot_serving_tradeoffs.py SUMMARY.json OUTDIR

# Queue versus first-token service decomposition
python tasks/realistic_workload_analysis/analyze_request_timing.py \
  RESULTS_DIR/server_DMC_CG.log RESULTS_DIR/server_LPT.log

# Mixed-tenancy scenarios (win scenario, controls, SLO A/Bs)
python tasks/realistic_workload_analysis/run_mixed_workloads.py \
  --scenario agent_plus_shortchat            # or shortchat_only,
                                             # agent_plus_multiturn{,_feasible},
                                             # multiturn_slo_clean
python tasks/realistic_workload_analysis/summarize_mixed_workloads.py OUTDIR \
  --baseline DMC_CG --method LPT             # or --method LPT_EDF
```

## Final conclusions

1. **Zero regression where the kernel cannot win.** Pure long-context
   TraceLab replay — every chunk attends a 32–100K prefix — is
   BatchPrefill's territory; the gated stack holds parity-or-better on
   throughput and pooled P99 ITL at every offered rate, and the
   `shortchat_only` control lands at parity with the gate declining
   persistent on 2,560/2,566 mixed batches. The gate, not scheduling
   restraint, is what makes the comparison safe by construction.
2. **The win scenario is co-tenancy**: fresh, short-prefix prefills meeting
   a long-context decode pool (chat + coding agent on one node). Codex
   tenant +36.5% throughput, chat tenant −42% mean ITL / −37% TTFT at
   identical token volumes; nsys attributes it to mixed-step wall time
   (157 s → 112 s; worst single mixed step 13.9 s → 0.9 ms).
3. **Predictability is the fused kernel's second product.** With the
   analytic cost model (`T_attn = max(T_p,T_d) + 0.53·min`, 3.1% MAPE) the
   EDF controller spends an ITL budget exactly: across three operating
   points the achieved pooled P99 converges onto the configured 100 ms
   budget (113 → 105 → 100.24 ms) as load moves from saturated to feasible,
   with request-max P99 −60% on both tenants at unchanged throughput.
4. **Occupancy, not context length, decides.** The kernel matters only
   while mixed forwards recur: heavy-mixing regimes in the sweet-spot
   rescan give a uniform −15 to −16% pooled P99 ITL and replace 32–82 s
   worst stalls with sub-second ones, while independent short-prompt
   datasets are structurally at parity (too few mixed forwards).

### Evaluation-hygiene lessons

- **Baseline contamination**: `SGLANG_FLASHINFER_USE_PERSISTENT` defaults
  on; every baseline launcher must disable it explicitly. The honest
  BatchPrefill baseline is itself 6.1–10.7% faster than the contaminated
  one, so contamination understated both sides.
- **Exact-prefix warmup** was the primary early evaluation bug: cache
  residency and eviction order dominated first-cohort admission and
  flipped apparent TTFT winners. Warmup is now length-matched but
  token-disjoint from every timed prefix.
- **A kernel speedup and a TTFT speedup are not interchangeable.**
  Request-timing decomposition (queue delay vs admission-to-first-token
  service) showed service time unchanged while queue time moved ~9–11%:
  near saturation, small service or completion-order changes drain the
  queue nonlinearly and the sign is phase-sensitive. Attribute TTFT
  changes to queue vs service before crediting the kernel.
- **Pooled token-level P99 hides rare multi-second stalls**; request-max
  P99 ITL is reported alongside it everywhere. A tenant at
  `--request-rate inf` makes FCFS admission ordering dominate the other
  tenant's latency, confounding attainment comparisons — clean SLO A/Bs
  keep every stream rate-limited.
- **Fair-baseline rules**: no minimum prefill chunk was introduced; DMC
  retains its stock kernel, 8,192-token cap, and CUDA graphs. Multi-trial
  runs are the default defense against episode-vs-property confusion
  (the −19% retraction-cascade episode reproduced nowhere in repeats).

## Context length for the kernel A/B — 2026-10-06

**Decision (user, 2026-10-06): the A/B must run at a context length where attention is
>50% of end-to-end time, by one of two routes.**

1. **Force-extrapolate RoPE** up to TraceLab's maximum session length. Normal agentic
   sessions (Claude/Codex-style) reach 300-400k tokens and stay under 1M, so extrapolating
   beyond the trained window is a fair simulation of the real regime even though
   perplexity is not meaningful there -- this measurement is about attention cost, not
   output quality.
2. **Or use a larger model with a natively longer context**, if extrapolation muddies the
   result.

**Why this matters, and why earlier runs could not show the bar.** At
`--random-input-len 2048` attention is a small fraction of the forward, so an attention-only
speedup is invisible end to end: a 1.21x kernel win cannot produce a >=30% TTFT/ITL gain
there no matter how correct it is. Those runs were measuring in a regime where the bar is
unreachable by construction. At 64k+ attention dominates, and at 300-400k it is >50% of
e2e, where the same kernel win translates directly.

**Also decided: use the TraceLab-derived mixed scenarios, not `--dataset-name random`.**
`run_mixed_workloads.py` has the real shapes (`agent_plus_shortchat`, `loadsweep_*`), whose
multi-turn tool-result prefills create the repeated mixed forwards this kernel exists for.
Independent-prompt random datasets omit exactly that structure -- this doc already records
that they average 242-690 tokens and cannot show the effect. A `SPLIT` method is now wired
into that driver.

