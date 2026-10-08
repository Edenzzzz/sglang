# Serving frontier evaluation

## Headline result and setting

The primary result and full experimental setting are documented first in
[`../realistic_workload_analysis.md`](../realistic_workload_analysis.md):
H800, Qwen2.5-7B-Instruct, 400 causal rounds from ten real TraceLab sessions,
64K–100K contexts, 100x-compressed gaps, and all ten sessions admitted
together (`--request-rate inf`). Warmup is length-matched but token-disjoint
from timed prefixes. In the corrected run, LPT is 0.44% above DMC, reduces
mean/P99 TTFT by 4.8%/10.2%, and has 3.8% worse pooled P99 ITL. The former
2.50%/ITL-parity result used exact timed prefixes during warmup and is
superseded.

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

The old TraceLab rate sweep used exact-prefix warmup and has been removed from
the canonical summary and figures. It must be rerun with disjoint warmup.
Code-Feedback shows a modest outward LPT shift around its knee. SWE-Gym leaves
DMC and LPT essentially tied because its short independent prompts expose few
persistent prefill/decode overlap steps.

## Why older figures were removed

The former checked-in figures mixed TTFT and ITL into one attainment number,
used median ITL where tail stalls matter, included correctness-invalid CUDA
graph runs, or compared old remote/N=1K experiments. They are retained in Git
history but are not valid paper evidence.
