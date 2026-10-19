# tasks/ — index

Active docs at the top level; superseded ones in `archive/`. Each doc opens with a dated
`# Current goal` checklist, per CLAUDE.md: `[x]` means measured against a reference, `[ ]` means
not run. Retractions stay in place, marked, rather than being deleted.

## Active

| doc | what it owns | status |
|---|---|---|
| `debug_kernel_server_perf_4_mode.md` | **Main task file.** FA3+FA2 two-executor split, the 4-mode MTP step, kernel A/Bs, serving A/Bs, perf model, chunk sizing | 2026-10-12: A1/A2 pass kernel gates; greedy verify epilogue is integrated and improves TTFT; wider-draft serving, ITL/throughput, multicast and EDF remain open |
| `pd_colocation_vs_disagg.md` | Multi-node scheduling: the four PD-colocation points, H20 16-GPU measurements | points 1-3 measured; point 4 (congestion) open, narrowed-fabric test in progress |
| `realistic_workload_analysis.md` | Workload shapes: TraceLab stats, arrival rates, offered-load sweeps, dataset transfer | reference, stable |
| `server_perf_model.md` | MLP latency prediction for prefill/decode colocation | reference; superseded in part by the fitted MTP step model (`perf_model/mtp_step_h200.json`) |
| `paper_outline.md` | Story flow of the OSDI draft (`AttentionFuse Outline OSDI 25'.docx`) mapped to measured results; core contributions C1-C4 and the evidence gaps | 2026-10-19: outline drafted; gaps 1-6 open |

## Code and results

| dir | contents |
|---|---|
| `fused_specdec_workload/` | kernel harness (`bench_4mode.py`, `four_mode_dispatch.py`, boundary sweeps) and `drivers/` — the serving gate (`srv_correctness.py`), A/B and sweep drivers |
| `pd_colocation/` | multi-node harness: `router.py` (rr / cache / cache_rand / cache_slo / cache_slo_rand), `mn_run.py`, `mn_node.sh`, `mn_bench.sh`, `calibrate_h20.py`, audit scripts |
| `perf_model/` | step-trace fitters and checked-in fitted models |
| `realistic_workload_analysis/` | `run_mixed_workloads.py` scenarios and result trees |
| `debug_kernel_server_perf/`, `edf_benchmark/` | result trees from earlier eras; kept for reproduction |

## Archive

`archive/debug_kernel_server_perf.md` (predecessor main file, H100/H800 era),
`archive/archive_h100_era.md`, `archive/edf_scheduling.md`,
`archive/fused_kernel_handoff.md`, `archive/README_full_graph.MD`, `archive/ralph_loop_cmd.txt`.
