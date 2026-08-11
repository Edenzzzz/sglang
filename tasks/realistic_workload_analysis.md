# Realistic Workload Analysis for the Persistent Overlap Kernel

## Scope

This document contains the current, correctness-valid evaluation of persistent
prefill/decode overlap. Historical B200 scans, graph-incompatible runs,
median-only latency tables, synthetic prioritization notes, and superseded
arrival-rate surveys were removed; they remain available in Git history.

The paper-facing comparison has three methods:

- **PP:** prefill-prioritized scheduling with unchanged BatchPrefill.
- **DMC:** default mixed-chunk scheduling with unchanged BatchPrefill.
- **LPT:** mixed scheduling with the persistent overlap kernel, adaptive work
  ordering, and a 7,168-token maximum prefill chunk.

Ordinary decode CUDA graphs are enabled for every method. Piecewise EXTEND
graphs are not combined with decode graphs because that combination is
output-incorrect at N=400. LPT has no minimum prefill chunk. Every reported
point retokenizes within 2% of its requested output.

## Headline: faithful TraceLab at peak load

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

(Superseded 2026-07-30: the DMC row silently ran the persistent kernel;
see the gated rerun below and the mixed-tenancy section for current results.)

| Method | Total tok/s | Mean / P99 TTFT | pooled P99 ITL |
|--------|------------:|----------------:|---------------:|
| DMC + decode CG | 28,965 | 2,583 / 64,253 ms | **93.56 ms** |
| LPT + persistent, 7,168 cap | **29,093** | **2,459 / 57,708 ms** | 97.13 ms |

LPT is 0.44% faster, reduces mean/P99 TTFT by 4.8%/10.2%, and increases pooled
P99 ITL by 3.8%. This corrected result does **not** yet support the former
claim of higher throughput at the same ITL SLO.

The earlier 31,186 versus 30,424 tok/s result reused each timed session's exact
prefix during warmup. Cache residency and eviction order then changed the
first timed cohort's admission phase: in a fixed-seed reproduction, eight LPT
requests shared a 24.16-second queue stall while the corresponding DMC cohort
waited 2.75 seconds. Reversing the phase also reversed the apparent TTFT
winner. The former +2.50% throughput, −41.0% mean TTFT, and −59.8% P99 TTFT
numbers are retained in Git history as the observation that triggered this
profile, but are superseded as paper evidence.

## Corrected TraceLab offered-load sweep

For causal TraceLab replay, `--request-rate` is the Poisson arrival rate of
**new sessions**; later rounds remain causal. The sweep was rerun at N=400
with fixed server seed one, identical sampled rounds, ordinary decode CUDA
graphs for both methods, and token-disjoint warmup. The rates below are
sessions/s; `inf` admits all ten sessions together.

| Rate | Method | Total tok/s | Mean / P99 TTFT | Mean / pooled P99 ITL |
|-----:|--------|------------:|----------------:|----------------------:|
| 0.02 | DMC | 27,198 | 1,390 / 26,249 ms | **17.55 / 91.10 ms** |
| 0.02 | LPT, 6,656 cap | **27,205** | **1,385 / 20,758 ms** | 17.67 / 93.86 ms |
| 0.05 | DMC | **28,995** | **1,530 / 26,257 ms** | **18.14 / 95.31 ms** |
| 0.05 | LPT, 6,656 cap | 28,897 | 1,562 / 26,526 ms | 18.23 / 113.32 ms |
| `inf` | DMC | **28,507** | **3,081 / 83,938 ms** | **18.08 / 92.65 ms** |
| `inf` | LPT, 6,656 cap | 28,188 | 3,385 / 95,939 ms | 18.11 / 98.92 ms |

This rejects the promising N=100 cap screen as a paper result. At N=400,
6,656 is throughput-neutral at 0.02 sessions/s and loses throughput and P99
ITL at 0.05 and saturation. The trace's first 100 selected rounds average 666
output tokens; the full 400 average only 481, so the shorter screen spends a
larger fraction of server time in the decode-heavy overlap regime.

### Gated rerun with honest baselines (2026-07-31, supersedes the table above)

The table above compared two servers that both ran the persistent kernel
(the DMC launcher never disabled `SGLANG_FLASHINFER_USE_PERSISTENT`) and both
paid the plan()-synchronize stall. The sweep was rerun with sglang
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
long contexts. A real-trace screen therefore required output length at least
512 tokens and varied independent causal sessions. It used N=100 only for
workload selection, the same fixed seed/disjoint warmup, 64K–100K contexts,
and a 6,656-token LPT cap.

| Sessions | Observed concurrency | Mean output / append | LPT throughput | Mean / P99 TTFT | Mean / pooled P99 ITL |
|---------:|---------------------:|---------------------:|---------------:|----------------:|----------------------:|
| 10 | 3.2 | 2,875 / 5,016 | −0.08% | +3.5% / +2.1% | −0.1% / **+14.5%** |
| 16 | 10.9 | 2,013 / 3,383 | **+2.14%** | **−7.2% / −10.3%** | **−0.25% / −14.0%** |
| 20 | 14.1 | 2,682 / 5,126 | **+0.60%** | +0.6% / +4.5% | **−0.1% / −14.1%** |

Sixteen sessions is the observed sweet spot: LPT improves capacity, TTFT, and
ITL together. It is nevertheless a stress point—absolute pooled P99 ITL is
642 ms—so it demonstrates where the kernel benefit becomes visible, not yet a
paper-ready interactive-SLO claim. Ten sessions leave too few overlapping
requests; twenty sessions are beyond the TTFT knee. A twelve-session sample
reached only 1.57 observed concurrency because its real windows contained long
tool/think gaps, showing that configured session count is not itself a
controlled offered-load coordinate.

The mechanism is workload occupancy. Long decode makes attention dominant,
but persistence helps end to end only while an intermittent append arrives
against enough active decodes to form repeated mixed forwards. At low
occupancy the current implementation still selects persistence for every
mixed batch, paying planning/fixed costs without enough overlap. A
server-level BatchPrefill-versus-persistent cost gate is the next required
optimization.

The exact screen values are in
[`tracelab_sweetspot_screen_summary.json`](realistic_workload_analysis/tracelab_sweetspot_screen_summary.json).

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
method. Code-Feedback spans 4–24 req/s plus saturation; SWE-Gym spans
0.5–4 req/s plus saturation.

- **Code-Feedback:** LPT modestly shifts the capacity/interactivity frontier
  around the load knee. At 16 req/s it is 0.4% faster and has 1.3% lower mean
  ITL than DMC; at 24 req/s it is 0.28% faster with 1.4% lower mean ITL.
- **SWE-Gym:** DMC and LPT are essentially tied on mean-ITL/capacity. At
  4 req/s, LPT improves request-max P99 ITL (136.64 versus 145.77 ms), while
  pooled P99 ITL is slightly worse.

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
plan-sync fix and would double-count. A feasible-load variant
(`agent_plus_multiturn_feasible`) measures attainment at matched throughput.

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
```

## Final conclusions and optimizations

1. The persistent kernel's isolated mixed-forward advantage is real, but the
   corrected broad N=400 TraceLab sweep does not move the DMC frontier. The
   best observed workload point is the 16-session decode-heavy screen:
   +2.14% throughput, −7.2%/−10.3% mean/P99 TTFT, and −0.25%/−14.0%
   mean/P99 ITL. Its absolute tail is overloaded, so this is a mechanism and
   workload-boundary result, not yet the headline paper claim.
2. Exact-prefix warmup was the primary evaluation bug. It made cache eviction
   and initial-cohort admission dominate mean and P99 TTFT. Warmup is now
   length-matched but token-disjoint from every timed prefix.
3. A kernel speedup and a TTFT speedup are not numerically interchangeable.
   TTFT includes admission delay and waiting behind earlier mixed chunks;
   faster mixed forwards can drain that queue nonlinearly near saturation.
   The matched request-timing profile below quantifies this amplification.
4. The main implementation fixes were CUDA-graph correctness isolation,
   ordinary decode graphs for all methods, removal of persistent decode
   overhead, adaptive mixed-work ordering, and shorter LPT-only chunk caps.
   Cap tuning alone does not generalize: 6,656 wins the N=100 screen but loses
   the broad N=400 sweep.
5. No minimum prefill chunk was introduced. DMC retains its stock BatchPrefill
   kernel and 8,192-token cap, so the comparison does not improve the baseline
   by silently changing its kernel or disabling its CUDA graph.
6. The current policy always selects persistence for a mixed batch. The load
   scan shows why that is insufficient: low-occupancy mixed batches regress
   P99 ITL, while concurrency around eleven exposes a simultaneous throughput
   and tail benefit. The next controller should compare predicted full-forward
   cost for BatchPrefill and persistence using decode count/context, append
   work, prefix hits, and P/D ratio, then fall back to DMC when the predicted
   advantage does not cover planning and interference cost.
   **Done (2026-07-30):** a measured k-NN gate now makes this per-batch
   choice (`SGLANG_MIXED_ATTENTION_KERNEL=auto`), the plan()-synchronize
   stall is removed, and the screens were re-run against a *true*
   BatchPrefill baseline (the old DMC baselines silently ran the persistent
   kernel — see `tasks/debug_kernel_server_perf.md`). Result: the 10-session
   regression is eliminated (pooled P99 ITL −6.2% at the matched cap) and
   the 16-session point keeps −14.6% pooled P99 ITL at cap 6656 or −23%
   request-max P99 at the default cap.

### Why TTFT improves more than raw kernel time

The request logger now records first-token time and directly decomposes:

`TTFT = admission/queue delay + chunked-prefill service to the first token`.

The compact machine-readable result is
[`tracelab_claude_disjoint_warmup_profile_summary.json`](realistic_workload_analysis/tracelab_claude_disjoint_warmup_profile_summary.json).

| Method | Queue mean / P99 | admission-to-first-token mean / P99 | server TTFT mean / P99 |
|--------|-----------------:|------------------------------------:|-----------------------:|
| DMC | 1,415 / 58,123 ms | **998 / 16,243 ms** | 2,413 / 62,606 ms |
| LPT | **1,289 / 51,747 ms** | 999 / **16,040 ms** | **2,288 / 56,441 ms** |

Admission-to-first-token service is effectively unchanged: 998 versus 999 ms
on average. The corrected TTFT improvement comes almost entirely from queue
time, which falls 8.9% at the mean and 11.0% at P99. Faster mixed forwards can
finish earlier work and shift admission timing, but this is queue-drainage
amplification, not a 60%-faster attention kernel. Near the KV/utilization
knee, a small service-time or completion-order change can affect every request
waiting behind it, and the sign is sensitive to cache and arrival phase.

This is the reviewer-safe conclusion: the old 41%/59.8% TTFT reductions were
not reproducible after removing prefix-warmup contamination. The clean
4.8%/10.2% reductions are consistent with an indirect queueing effect, while
the remaining ITL regression must be closed before making the headline
same-SLO claim.
