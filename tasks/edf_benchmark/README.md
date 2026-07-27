# Serving frontier evaluation

## Headline result and setting

The primary result and full experimental setting are documented first in
[`../realistic_workload_analysis.md`](../realistic_workload_analysis.md):
H800, Qwen2.5-7B-Instruct, 400 causal rounds from ten real TraceLab sessions,
64K–100K contexts, 100x-compressed gaps, and all ten sessions admitted
together (`--request-rate inf`). LPT reaches 31,186 token/s, 2.50% above DMC,
while reducing P99 TTFT from 31.66 to 12.73 seconds and preserving pooled P99
ITL parity. That is the headline; the sweeps below test robustness as offered
load changes and transfer to independent coding datasets.

## Paper-facing figures

Run:

```bash
python tasks/edf_benchmark/analyze_results.py RESULTS_DIR
python tasks/edf_benchmark/plot_serving_tradeoffs.py RESULTS_DIR/summary.json
```

The plotting script emits four separate figures:

- [`slo_ttft_attainment_vs_rate.png`](current_policy_n400/slo_ttft_attainment_vs_rate.png):
  length-aware TTFT SLO attainment.
- [`slo_itl_attainment_vs_rate.png`](current_policy_n400/slo_itl_attainment_vs_rate.png):
  fraction of requests whose own P99 ITL
  meets the ITL SLO.
- [`p99_itl_vs_rate.png`](current_policy_n400/p99_itl_vs_rate.png):
  pooled token-level P99 ITL.
- [`pareto_capacity_vs_mean_itl.png`](current_policy_n400/pareto_capacity_vs_mean_itl.png):
  achieved total-token throughput against
  mean ITL, the capacity/interactivity view.

TTFT and ITL attainment are intentionally not multiplied or combined. The
continuous P99-ITL and capacity/interactivity curves are the primary frontier
evidence; attainment is a threshold-sensitive diagnostic. Pooled P99 ITL can
hide rarer request stalls, so the JSON summary also reports median and P99
request-maximum ITL.

## Matched configurations

- PP: unchanged `BatchPrefill`, ordinary decode CUDA graph.
- DMC: unchanged mixed `BatchPrefill`, ordinary decode CUDA graph.
- LPT: persistent mixed kernel, ordinary decode CUDA graph, adaptive work
  ordering, and the validated 7,168-token maximum prefill chunk. There is no
  minimum prefill chunk.
- Every coding-dataset point uses N=400 and ten warmup requests.

Code-Feedback uses offered rates 4, 6, 8, 10, 12, 14, 16, 20, 24, and
saturation. SWE-Gym has a much lower request-capacity knee because its outputs
are longer, so its informative grid is 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, and
saturation.

TraceLab uses offered **session** arrival rate rather than breaking causal
round ordering. The sweep rates are 0.01, 0.02, 0.05 sessions/s and
saturation; later requests in each real session still wait for the preceding
response and the scaled recorded think/tool delay. Achieved request and token
throughput are measured in every run.

The completed sweep shows why the headline must retain its load setting.
TraceLab LPT is tied with DMC at 0.01 sessions/s, improves P99 TTFT by 35.6%
at 0.02, and produces its 2.50% throughput gain when all sessions are offered
together. Code-Feedback shows a modest outward LPT shift around its knee.
SWE-Gym leaves DMC and LPT essentially tied: its independent requests have
short prompts and do not preserve intermittent tool-result prefills, so few
steps exercise persistent prefill/decode overlap.

## Why older figures were removed

The former checked-in figures mixed TTFT and ITL into one attainment number,
used median ITL where tail stalls matter, included correctness-invalid CUDA
graph runs, or compared old remote/N=1K experiments. They are retained in Git
history but are not valid paper evidence.
