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

## Offered-load sweep status

For causal TraceLab replay, `--request-rate` is the Poisson arrival rate of
**new sessions**; later rounds remain causal. The previous TraceLab sweep used
the same exact-prefix warmup and is therefore removed from the current
figures. It must be rerun with disjoint warmup before making a rate-dependent
frontier claim. The independent Code-Feedback and SWE-Gym sweeps are
unaffected.

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

## Paper-facing figures

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

# Analyze and plot accepted results
python tasks/edf_benchmark/analyze_results.py RESULTS_DIR
python tasks/edf_benchmark/plot_serving_tradeoffs.py SUMMARY.json OUTDIR

# Queue versus first-token service decomposition
python tasks/realistic_workload_analysis/analyze_request_timing.py \
  RESULTS_DIR/server_DMC_CG.log RESULTS_DIR/server_LPT.log
```

## Final conclusions and optimizations

1. The persistent kernel's isolated mixed-forward advantage is real, but the
   clean end-to-end TraceLab gain is currently only 0.44% and comes with 3.8%
   worse pooled P99 ITL. A same-SLO frontier claim requires more optimization
   and repeated disjoint-warmup runs.
2. Exact-prefix warmup was the primary evaluation bug. It made cache eviction
   and initial-cohort admission dominate mean and P99 TTFT. Warmup is now
   length-matched but token-disjoint from every timed prefix.
3. A kernel speedup and a TTFT speedup are not numerically interchangeable.
   TTFT includes admission delay and waiting behind earlier mixed chunks;
   faster mixed forwards can drain that queue nonlinearly near saturation.
   The matched request-timing profile below quantifies this amplification.
4. The main implementation fixes were CUDA-graph correctness isolation,
   ordinary decode graphs for all methods, removal of persistent decode
   overhead, adaptive mixed-work ordering, and the 7,168-token LPT cap.
5. No minimum prefill chunk was introduced. DMC retains its stock BatchPrefill
   kernel and 8,192-token cap, so the comparison does not improve the baseline
   by silently changing its kernel or disabling its CUDA graph.

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
