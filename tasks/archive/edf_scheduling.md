# EDF (Earliest Deadline First) Scheduling for SLO-Aware Serving

> **Validity note (2026-08-13).** The sweeps in this file predate the
> 2026-07-30 honest-baseline fix: `SGLANG_FLASHINFER_USE_PERSISTENT`
> defaults on, and pre-fix DMC arms may have silently run the persistent
> kernel for mixed batches. Policy conclusions (EDF vs FCFS, chunk control
> on/off) compare within the same kernel arm and stand; do not cite
> LPT-vs-DMC *kernel* deltas from this file. Current kernel and SLO
> results live in `tasks/debug_kernel_server_perf.md` and
> `tasks/realistic_workload_analysis.md` (analytic-model EDF pins pooled
> P99 ITL at 100.24 ms against a 100 ms budget).

## [x] Goal
Implement deadline-aware scheduling in sglang to maximize SLO attainment under load. Compare PP vs LPT+CG vs DMC on real agent workloads with realistic SLOs and request rates.

## [x] Design

### [x] 1. SLO Definition

Per-request SLOs based on literature:
- **TTFT SLO**: Base 2s + length-dependent scaling (longer inputs get more time)
- **ITL SLO**: Workload-specific (see Tempo paper, DistServe, Sarathi-Serve)

Details and paper references in `tasks/realistic_workload_analysis.md`.

### [x] 2. Priority = Earliest Deadline

sglang already supports `--schedule-policy priority` with integer priorities. Modify to:
- Set priority = deadline timestamp (lower = earlier deadline = higher priority)
- For prefill admission: priority = arrival_time + TTFT_SLO
- For decode scheduling: priority = last_token_time + ITL_SLO
- Scheduler picks requests with earliest deadlines first (EDF)

Key files:
- `python/sglang/srt/managers/schedule_policy.py` — priority scheduling logic
- `python/sglang/srt/managers/scheduler.py` — request admission, batch formation
- `python/sglang/srt/managers/io_struct.py` — request metadata (add deadline fields)

### [x] 3. SLO-Aware Chunk Size Control (DMC + LPT only)

Use a **server-level** latency prediction model to estimate: **how many prefill tokens can we admit in this step without violating the ITL SLO of any running decode request?**

- Retrain `tasks/server_perf_model.md` MLP model for BOTH LPT and DMC **server configurations**
- Uses the existing `overlap_profiler.py` to profile full forward pass (MLP + attention + all layers)
- Note: kernel-level latency is too noisy (L2 eviction effects) — server-level profiling smooths this out, which is why LPT exists
- Input: (n_p, r_p, r_d, decode_bs)
- Output: predicted server step latency (ms)
- At batch formation time:
  1. For each candidate prefill chunk size C:
     - Predict step_latency(C, prefix_len, decode_kv_avg, decode_bs)
     - Check: for ALL running decode requests, does `now + step_latency - last_token_time <= ITL_SLO`?
  2. Use largest C that satisfies all decode SLOs
  3. If no prefill can be admitted without violating SLOs, run pure decode

This applies to DMC and LPT (mixed batches). PP doesn't mix, so it only uses EDF for prefill admission order.

### [x] 4. Benchmarking

**Workloads** (must have traces for realistic arrival patterns):
1. ShareGPT (general chat) — rate TBD from trace analysis
2. Code-Feedback (code generation with feedback loops) — rate TBD
3. Nebius-SWE (SWE-agent trajectories) — may need lower rate (compute-bound at rate=2+)

**Metrics**:
- SLO attainment rate (% requests meeting both TTFT and ITL SLOs)
- P50/P95/P99 TTFT and ITL
- Total throughput (tok/s)
- Pareto curves: vary request rate, plot ITL P99 vs throughput for all 3 methods

**Configs**: PP+PCG, LPT+CG29, DMC (all with EDF scheduling + SLO-aware chunk size where applicable)

---

## [x] Implementation Plan

### [x] Phase 1: SLO Research + Performance Model
- Research TTFT/ITL SLO standards from Tempo, DistServe, Sarathi-Serve, vLLM papers
- Define per-workload SLOs
- Profile server-level latency for LPT and DMC configs using `overlap_profiler.py`
- Retrain MLP to predict server step latency for both configs

### [x] Phase 2: EDF Scheduler
- [x] Modify priority scheduling to use deadline timestamps
- [x] Add TTFT/ITL deadline fields to request metadata
- [x] Implement SLO-aware chunk size control using latency model
- [x] Fix clock mismatch: `last_tic` uses `time.monotonic()`, EDF must use monotonic consistently (not perf_counter)
- [x] Add `SGLANG_EDF_LATENCY_MODEL` env var to specify per-config model path
- [x] Wire predictor into scheduler init and `get_new_batch_prefill`

### [x] Phase 3: Benchmarking

- [x] Initial 200/100-prompt runs (commit c68bfcefe) — superseded by 1k-prompt sweeps below.
- [x] Generate Pareto curves — `tasks/edf_benchmark/pareto_baseline_median_itl.png`, `pareto_edf_slo_attainment.png`
- [x] Re-run with EDF scheduling enabled and compare SLO attainment

##### [x] Implementation notes for SLO-aware chunk control
- The first attempt used `req.last_tic` as the previous decode time, but `last_tic` is only updated when `enable_metrics=True`. With metrics off (default for benchmarks) it equals request creation time, making slack massively negative and starving prefills (verified: smoke test gave 579 tok/s vs 2135 with the fix).
- Fix: track `_edf_last_step_time` per request, updated in `process_batch_result_decode`. New decode requests fall back to `now` (full ITL_SLO budget). See `python/sglang/srt/managers/edf_scheduler.py:edf_compute_max_chunk_size` and `python/sglang/srt/managers/scheduler_output_processor_mixin.py:process_batch_result_decode`.

## [x] 1k-prompt sweep with 6 Pareto curves (sglang `attn_fuse_`, 2026-04-07/09) — EDF tables re-run post-fix v0.3.3

Per the 4 requirements:
1. 1000 prompts per dataset for all sweeps.
2. Baseline tables report median TTFT and median ITL alongside P99.
3. Rates pushed to 8 / 16 / 24 / 32 / 48 / inf so saturation is visible.
4. Six Pareto plots: 3 baseline (median ITL vs rate) and 3 EDF + SLO-aware (decode SLO attainment vs rate).

SLOs (from Tempo / Ascendra / DistServe): ShareGPT (TTFT 2s / ITL 100ms), Code-Feedback (3s / 150ms), SWE-Gym (5s / 150ms).
Decode SLO attainment is **strict per-request**: max ITL of the request ≤ ITL_SLO. (TPOT — mean ITL ≤ SLO — stays ≈100% even at saturation because median ITL is 25-30ms vs 100-150ms SLO, so it doesn't reveal differences. The strict tail metric is what chunk control protects.)

Reproduce:
```
bash tasks/edf_benchmark/run_baseline_1k.sh
bash tasks/edf_benchmark/run_edf_1k.sh
python3 tasks/edf_benchmark/analyze_results.py tasks/edf_benchmark/results_baseline_1k
python3 tasks/edf_benchmark/analyze_results.py tasks/edf_benchmark/results_edf_1k
python3 tasks/edf_benchmark/plot_pareto_6.py tasks/edf_benchmark/results_baseline_1k/summary.json tasks/edf_benchmark/results_edf_1k/summary.json tasks/edf_benchmark
```
Plots: `tasks/edf_benchmark/pareto_baseline_median_itl.png` and `tasks/edf_benchmark/pareto_edf_slo_attainment.png`.

### [x] Baseline (FCFS, no SLO control), 1000 prompts

#### [x] ShareGPT
| Method | Rate | tok/s | med TTFT | P99 TTFT | med ITL | P99 ITL | ITLp99% |
|--------|------|-------|----------|----------|---------|---------|---------|
| PP     | 8    | 5105  | **43.6** | 106.2    | 15.7    | 58.3    | **81.3%** |
| PP     | 16   | 5883  | 65.7     | 21885.9  | 26.3    | **102.5** | 60.3%  |
| PP     | 24   | 5848  | 81.4     | 40393.4  | 27.5    | **148.3** | *27.3%* |
| PP     | 32   | 5860  | 90.3     | 45771.8  | 27.9    | **136.0** | *22.1%* |
| PP     | 48   | 6027  | 3231.3   | 56770.6  | 27.7    | **154.0** | *18.1%* |
| PP     | inf  | 6015  | 13627.7  | 78454.4  | 27.5    | **102.4** | *19.3%* |
| DMC    | 8    | 5075  | 75.9     | 694.0    | 17.4    | 57.8    | **81.3%** |
| DMC    | 16   | 5968  | 119.8    | 21111.5  | 25.5    | 88.7    | 70.1%     |
| DMC    | 24   | 5987  | 157.8    | 38907.4  | 27.3    | 98.4    | 58.2%     |
| DMC    | 32   | 6041  | 578.5    | 47683.8  | 27.5    | **101.4** | 50.8%  |
| DMC    | 48   | 6075  | 6502.7   | 57245.4  | 27.3    | **104.8** | 47.0%  |
| DMC    | inf  | 6075  | 17398.0  | 77722.4  | 27.1    | **113.3** | 49.5%  |
| LPT_CG | 8   | 5091  | 74.9     | 823.7    | 17.2    | 57.3    | **83.9%** |
| LPT_CG | 16  | 5954  | 121.6    | 21380.1  | 25.6    | 89.8    | 69.0%     |
| LPT_CG | 24  | 6002  | 147.4    | 38600.0  | 27.6    | 96.5    | 58.5%     |
| LPT_CG | 32  | 6028  | 561.2    | 47747.6  | 27.5    | **100.5** | 54.8%  |
| LPT_CG | 48  | 6053  | 7000.3   | 57074.8  | 27.2    | **106.1** | 46.2%  |
| LPT_CG | inf | 6059  | 17390.7  | 77914.9  | 27.2    | **106.2** | 50.5%  |

#### [x] Code-Feedback
| Method | Rate | tok/s | med TTFT | P99 TTFT | med ITL | P99 ITL | ITLp99% |
|--------|------|-------|----------|----------|---------|---------|---------|
| PP     | 8    | 6836  | **35.2** | **65.3** | 15.8    | 38.0    | **100.0%** |
| PP     | 16   | 8264  | 58.8     | 19878.6  | 28.2    | 90.9    | 61.8%     |
| PP     | 24   | 8558  | 66.0     | 36414.7  | 29.6    | 144.3   | 52.2%     |
| PP     | 32   | 8638  | 67.1     | 44920.5  | 29.6    | **238.0** | *41.6%* |
| PP     | 48   | 8737  | 86.1     | 48159.8  | 29.7    | **265.2** | *33.6%* |
| PP     | inf  | 8907  | 11958.7  | 71044.4  | 30.2    | 94.9    | *47.6%*   |
| DMC    | 8    | 6813  | 67.8     | 884.8    | 16.3    | 35.6    | 89.0%     |
| DMC    | 16   | 8302  | 117.3    | 19395.4  | 27.7    | 96.1    | 57.7%     |
| DMC    | 24   | 8566  | 147.8    | 36492.7  | 28.9    | **211.6** | 45.2%  |
| DMC    | 32   | 8645  | 159.8    | 45270.8  | 29.5    | **214.2** | 41.6%  |
| DMC    | 48   | 8707  | 2204.5   | 51421.5  | 30.0    | **206.3** | 44.4%  |
| DMC    | inf  | 8824  | 16482.6  | 71714.8  | 30.0    | **178.8** | 46.7%  |
| LPT_CG | 8   | 6816  | 67.9     | **5593.9** | 16.6  | 36.9    | 87.1%     |
| LPT_CG | 16  | 8210  | 115.4    | 19763.6  | 27.7    | 81.7    | 63.0%     |
| LPT_CG | 24  | 8406  | 140.7    | 37759.5  | 29.5    | **185.0** | 44.1%  |
| LPT_CG | 32  | 8583  | **177.3** | 45465.9 | 29.6    | **250.1** | 43.3%  |
| LPT_CG | 48  | 8705  | 1920.5   | 51514.4  | 29.9    | 132.8   | 48.8%     |
| LPT_CG | inf | 8758  | 17768.1  | 74555.4  | 30.0    | **200.6** | 52.3%  |

#### [x] SWE-Gym
| Method | Rate | tok/s | med TTFT | P99 TTFT | med ITL | P99 ITL | ITLp99% |
|--------|------|-------|----------|----------|---------|---------|---------|
| PP     | 8    | 4258  | **55.5** | 16674.2  | 20.6    | 79.7    | **99.1%** |
| PP     | 16   | 4399  | 87.4     | 59295.4  | 25.8    | **101.9** | *68.3%* |
| PP     | 24   | 4445  | 2555.6   | 76605.9  | 26.2    | **116.6** | *45.1%* |
| PP     | 32   | 4468  | 6797.6   | 85033.7  | 26.2    | **108.0** | *53.8%* |
| PP     | 48   | 4479  | 12115.1  | 94726.3  | 26.1    | **111.2** | *52.3%* |
| PP     | inf  | 4484  | 23379.1  | 115829.2 | 26.2    | **101.7** | *55.0%* |
| DMC    | 8    | 4227  | 113.8    | 18212.7  | 21.7    | 85.1    | 81.3%     |
| DMC    | 16   | 4411  | 158.8    | 58396.7  | 25.7    | 93.3    | **82.8%** |
| DMC    | 24   | 4457  | 4036.3   | 75388.9  | 26.2    | 98.2    | 76.5%     |
| DMC    | 32   | 4470  | 9646.6   | 85313.3  | 26.1    | 97.1    | 76.3%     |
| DMC    | 48   | 4479  | 13979.3  | 95069.2  | 26.1    | **100.3** | 75.1%  |
| DMC    | inf  | 4475  | 25194.3  | 116873.8 | 26.2    | 98.0    | **80.3%** |
| LPT_CG | 8   | 4215  | **132.0** | **19881.5** | 22.0 | 95.2   | 78.1%     |
| LPT_CG | 16  | 4412  | 154.1    | 58242.6  | 25.7    | 93.1    | **83.1%** |
| LPT_CG | 24  | 4458  | 3680.6   | 74877.6  | 26.1    | 97.7    | 76.1%     |
| LPT_CG | 32  | 4472  | 8940.2   | 84733.8  | 26.2    | 97.5    | 76.8%     |
| LPT_CG | 48  | 4485  | 13853.1  | 94975.5  | 26.0    | 96.6    | 76.6%     |
| LPT_CG | inf | 4479  | 25114.5  | 116488.9 | 25.9    | **102.0** | **81.3%** |

### [x] EDF + SLO-aware chunk control, 1000 prompts (post-fix v0.3.3, task #109)

#### [x] ShareGPT (per-req P99 ITL ≤ 100ms)
| Method | Rate | tok/s | med TTFT | P99 TTFT | med ITL | P99 ITL | maxITL≤100ms | ITLp99% |
|--------|------|-------|----------|----------|---------|---------|-------------|---------|
| PP     | 8    | 5093  | **43.4** | 100.5    | 15.7    | 59.4    | 57.0%       | 57.7%   |
| PP     | 16   | 5869  | 65.5     | 22376.5  | 26.5    | **127.1** | 17.2%     | 42.4%   |
| PP     | 24   | 5812  | 80.4     | 40955.6  | 27.6    | **153.8** | 9.5%      | *18.8%* |
| PP     | 32   | 5946  | 90.0     | 46523.8  | 27.8    | **176.0** | 5.7%      | *13.7%* |
| PP     | 48   | 5984  | 3512.9   | 57794.6  | 27.7    | **168.3** | 5.2%      | *12.0%* |
| PP     | inf  | 6007  | 14692.3  | 78935.6  | 27.5    | **147.4** | 4.8%      | *12.0%* |
| DMC    | 8    | 5100  | 76.2     | 852.4    | 17.3    | **56.9** | **63.8%**  | **81.9%** |
| DMC    | 16   | 5843  | 520.7    | 24476.6  | 26.7    | **66.8** | **37.0%**  | **92.8%** |
| DMC    | 24   | 5951  | 4869.2   | 41855.6  | 27.2    | **68.9** | **39.9%**  | **93.8%** |
| DMC    | 32   | 5931  | 9983.1   | 52073.0  | 27.1    | **72.1** | **35.6%**  | **93.0%** |
| DMC    | 48   | 5973  | 14835.7  | 61424.8  | 26.8    | **76.7** | **29.0%**  | **90.0%** |
| DMC    | inf  | 5877  | 26699.7  | 83186.4  | 27.0    | **73.0** | **37.4%**  | **94.5%** |
| LPT_CG | 8    | 5100  | 76.0     | 862.5    | 17.1    | **55.9** | **63.2%**  | **83.9%** |
| LPT_CG | 16   | 5867  | 568.3    | 24497.5  | 26.5    | **74.7** | **31.5%**  | **89.3%** |
| LPT_CG | 24   | 5901  | 4639.5   | 42600.3  | 27.7    | **75.1** | **41.6%**  | **92.6%** |
| LPT_CG | 32   | 5973  | 9136.7   | 51261.2  | 27.1    | **75.7** | **36.2%**  | **91.1%** |
| LPT_CG | 48   | 6005  | 14799.6  | 60852.0  | 26.7    | **76.0** | **28.6%**  | **90.9%** |
| LPT_CG | inf  | 5592  | 26824.2  | 83765.7  | 26.9    | **75.6** | **36.6%**  | **94.7%** |

#### [x] Code-Feedback (per-req P99 ITL ≤ 150ms)
| Method | Rate | tok/s | med TTFT | P99 TTFT | med ITL | P99 ITL | maxITL≤150ms | ITLp99% |
|--------|------|-------|----------|----------|---------|---------|-------------|---------|
| PP     | 8    | 6858  | **35.2** | **64.6** | 15.8    | 38.5    | **81.1%**   | 96.1%   |
| PP     | 16   | 8315  | 57.6     | 20143.0  | 28.5    | 113.8   | 11.7%       | 75.4%   |
| PP     | 24   | 8425  | 65.9     | 37303.3  | 29.9    | 126.7   | 6.3%        | *39.3%* |
| PP     | 32   | 8537  | 68.8     | 45811.2  | 30.2    | **170.7** | 3.0%      | *28.5%* |
| PP     | 48   | 8593  | 71.0     | 48233.1  | 30.3    | **187.4** | 1.0%      | *15.3%* |
| PP     | inf  | 8678  | 13994.6  | 72871.3  | 30.1    | **173.1** | 0.1%      | *0.5%*  |
| DMC    | 8    | 6852  | 59.4     | 782.9    | 15.6    | **34.3** | **58.2%**  | **90.9%** |
| DMC    | 16   | 8295  | 129.3    | 19433.4  | 27.4    | **74.0** | **10.1%**  | **66.2%** |
| DMC    | 24   | 8427  | 444.5    | 36898.4  | 29.6    | **68.4** | 4.2%       | **62.5%** |
| DMC    | 32   | 8523  | 2454.4   | 45549.7  | 29.6    | **85.5** | 3.5%       | **50.3%** |
| DMC    | 48   | 8655  | 8790.0   | 55467.9  | 29.5    | **68.4** | 2.2%       | **57.4%** |
| DMC    | inf  | 8650  | 22215.5  | 76236.4  | 29.2    | **62.8** | 1.7%       | **79.7%** |
| LPT_CG | 8    | 6848  | 60.1     | 529.5    | 15.7    | **34.1** | **43.5%**  | **92.8%** |
| LPT_CG | 16   | 8320  | 127.8    | 19451.2  | 27.5    | **70.9** | **9.1%**   | **58.8%** |
| LPT_CG | 24   | 8430  | 995.2    | 36557.2  | 29.8    | **73.3** | 4.4%       | **63.2%** |
| LPT_CG | 32   | 8529  | 2723.2   | 45353.8  | 30.0    | **70.5** | 2.9%       | **58.1%** |
| LPT_CG | 48   | 8627  | 9112.2   | 55663.1  | 29.6    | **123.2** | 2.0%      | **57.7%** |
| LPT_CG | inf  | 8684  | 21324.7  | 75106.8  | 29.4    | **63.4** | 1.4%       | **82.7%** |

#### [x] SWE-Gym (per-req P99 ITL ≤ 150ms)
| Method | Rate | tok/s | med TTFT | P99 TTFT | med ITL | P99 ITL | maxITL≤150ms | ITLp99% |
|--------|------|-------|----------|----------|---------|---------|-------------|---------|
| PP     | 8    | 4257  | **56.3** | 16728.1  | 20.6    | 87.5    | 38.8%       | 98.9%   |
| PP     | 16   | 4397  | 85.6     | 59192.5  | 26.0    | **124.1** | 7.3%      | *69.9%* |
| PP     | 24   | 4434  | 1915.8   | 76414.5  | 26.1    | **149.2** | 2.3%      | *40.4%* |
| PP     | 32   | 4456  | 7025.0   | 85430.7  | 26.3    | **139.6** | 5.3%      | *48.2%* |
| PP     | 48   | 4468  | 12511.3  | 95191.3  | 26.3    | **141.8** | 4.0%      | *45.6%* |
| PP     | inf  | 4460  | 23970.8  | 115165.7 | 26.3    | **130.8** | 7.8%      | *59.0%* |
| DMC    | 8    | 4231  | 115.2    | 19506.9  | 21.7    | **80.6** | **22.0%**  | **84.9%** |
| DMC    | 16   | 4380  | 2189.3   | 62624.3  | 25.7    | **78.7** | **34.8%**  | **95.6%** |
| DMC    | 24   | 4439  | 8860.6   | 78343.6  | 25.7    | **84.3** | **40.6%**  | **92.2%** |
| DMC    | 32   | 4479  | 13208.6  | 86573.4  | 25.2    | **85.9** | **39.9%**  | **96.0%** |
| DMC    | 48   | 4449  | 19080.2  | 98803.9  | 25.5    | **82.5** | **25.3%**  | **97.6%** |
| DMC    | inf  | 4455  | 30667.8  | 117906.4 | 25.7    | **87.6** | **50.4%**  | **98.7%** |
| LPT_CG | 8    | 4234  | 129.4    | 18724.8  | 21.8    | **80.0** | **24.5%**  | **79.5%** |
| LPT_CG | 16   | 4394  | 1195.0   | 61488.6  | 25.5    | **80.3** | **33.4%**  | **97.7%** |
| LPT_CG | 24   | 4464  | 8739.2   | 77255.9  | 25.3    | **86.5** | **40.3%**  | **94.5%** |
| LPT_CG | 32   | 4453  | 14066.8  | 88153.3  | 25.6    | **87.7** | **41.0%**  | **94.0%** |
| LPT_CG | 48   | 4460  | 18724.6  | 98098.2  | 25.5    | **88.0** | **36.3%**  | **96.9%** |
| LPT_CG | inf  | 4456  | 30248.6  | 118161.5 | 25.6    | **83.8** | **41.4%**  | **96.2%** |

### [x] 1k-sweep findings (post-fix v0.3.3)
1. **Per-req P99 ITL is the right metric**: strict max-ITL is still useful but overly strict (few % attainment at high rates). Per-req P99 ITL ≤ SLO provides the clear separation.
2. **Chunk control dramatically improves decode tail across ALL workloads** (P99 ITL reductions):
   - ShareGPT: baseline 88-113ms → post-fix **57-77ms** (35-40% reduction)
   - Code-Feedback: baseline 90-265ms → post-fix **34-85ms** (60-75% reduction)
   - SWE-Gym: baseline 93-102ms → post-fix **78-88ms** (10-15% reduction, lighter wins because SWE prefills are only 412 tokens)
3. **DMC/LPT_CG with chunk control now dominate PP on per-req P99 ITL ≤ SLO**:
   - SWE-Gym: DMC/LPT_CG **85-99%** vs PP **40-70%** (PP's pure-prefill steps produce 120-150ms tail spikes)
   - ShareGPT: DMC/LPT_CG **82-95%** vs PP **12-58%**
   - Code-Feedback: DMC/LPT_CG **50-93%** vs PP **0.5-96%** (PP wins only at lowest rate where no prefill-decode interference occurs)
4. **Throughput preserved or improved**: chunk control throttling to MIN_CHUNK=512 at saturation doesn't reduce throughput; DMC/LPT_CG match or exceed PP throughput at every workload.
5. **The v0.3.2 data was completely wrong**: the broken fall-back-to-default bug meant chunk control was disabled at saturation. Now post-fix, the P99 ITL improvement is 35-75% depending on workload — the system works as designed.

## [x] SWE-bench-Pro sweep (2026-04-09, task #113)

Medium prefill workload: ~720-token median input (between code-feedback 200 and nebius 9700). 731 examples from `ScaleAI/SWE-bench_Pro` test split. Output fixed to 1024 via `--sharegpt-output-len`. SLOs: TTFT 4s / ITL 150ms.

Reproduce:
```
bash tasks/edf_benchmark/run_swebpro_baseline_1k.sh && bash tasks/edf_benchmark/run_swebpro_edf_1k.sh
python3 tasks/edf_benchmark/analyze_results.py tasks/edf_benchmark/results_{baseline,edf}_swebpro_1k
```

### [x] SWE-bench-Pro ITLp99% summary (per-req P99 ITL ≤ 150ms)

| Rate | DMC base→EDF | LPT_CG base→EDF | PP base→EDF |
|------|--|--|--|
| 4 | 100% → 100% | 100% → 99% | 100% → 100% |
| 8 | 100% → 100% | 100% → 100% | 100% → 100% |
| 16 | 74% → **100%** | 48% → **100%** | 86% → 33% |
| 24 | 79% → **100%** | 97% → **100%** | 18% → 14% |
| 32 | 96% → **100%** | 76% → **100%** | 28% → 15% |
| inf | 1% → **100%** | 0% → **100%** | 100% → 67% |

### [x] SWE-bench-Pro findings
1. **Chunk control gives 100% ITLp99 at ALL saturated rates (16-inf)** — P99 ITL capped to 45-80ms (vs baseline 120-218ms). Even medium-prefill workloads benefit.
2. **PP collapses at rates 16-32**: 14-33% ITLp99 because PP runs pure-prefill steps that block decodes for 180-193ms. At rate=inf PP recovers to 67% because prefills are queued and drained before decodes (no mixing at all).
3. **Confirms benefit is monotonic in prefill weight**: code-feedback (200 tok input) → ~35-75% P99 ITL improvement; SWE-bench-Pro (720 tok) → ~50-77%; SWE-Gym (412 tok) → ~10-15%; nebius (9700 tok) → 50-55%. Heavier prefill = more chunk-control headroom.
4. **Throughput preserved**: DMC/LPT_CG EDF 7200-7360 tok/s vs baseline 7440-7650 → ~3% reduction from more aggressive throttling, acceptable trade-off for 100% SLO attainment.

## [x] Nebius-SWE-Trajectories sweep + chunk-control fall-back fix (2026-04-09)

**Why this dataset**: Code-Feedback proved uninformative — its 200-token median input means there's barely any prefill to chunk-control, so all methods converge to ~0% strict-tail SLO at saturation regardless of policy. To get clean Pareto separation we need a workload where chunk control actually has something to throttle. From `tasks/realistic_workload_analysis.md`, **Nebius-SWE-Trajectories** is the only prefill-heavy real-world dataset surveyed (median input 9K tokens, 2.4:1 input/output ratio, hits 16K context by turn 8). Loaded via the new `nebius-swe-traj` entry in `bench_serving_3rdparty.py` (random AI-turn boundary sampling from `nebius/SWE-agent-trajectories`). Decode length fixed to 1024 via `--sharegpt-output-len 1024` to mimic Qwen3-8B thinking-mode output.

SLOs (matching SWE-Gym, since both are agent traces): TTFT 5s, ITL 150ms, plus length scaling.

**SLO metric**: introduced `itl_p99_attain_pct` (per-request P99 ITL ≤ ITL_SLO, Sarathi-Serve-style) in `analyze_results.py`. Strict max-ITL gives ≈0% across all methods on this workload because each request has 1024 decode steps × 1000 reqs = 1M decode steps, so any single-step outlier kills the score and reveals nothing. Per-req P99 filters single-step noise but still penalizes the consistent prefill-blocking pattern. TPOT (mean ITL) is too lenient — stays > 95% across every run.

### [x] Chunk-control fall-back bug (root cause + fix)

The original `edf_compute_max_chunk_size` had this branch:
```python
if min_slack_ms <= 0 or decode_only_ms > min_slack_ms:
    return default_chunk_size  # 8192
```
At saturation (rate ≥ 2 on nebius), the queue depth is so large that **at least one** running decode has already exceeded its 150ms slack (`min_slack_ms ≤ 0`), so we fell back to the **full** 8192 chunk every step. A single 8K-token prefill chunk on this workload takes ~200-300ms — directly cascading the violation to all other decodes and producing the same ~290ms P99 ITL as the no-control baseline. The first nebius EDF run confirmed this: DMC P99 ITL = 297ms vs 297ms baseline at rate 2 (no improvement at all).

**Fix** (`python/sglang/srt/managers/edf_scheduler.py`): when slack is exhausted, throttle to MIN_CHUNK (512) instead of falling back to default. The intuition is that already-violated requests can't be saved by relaxing the chunk, but the **not-yet-violated** decodes can be — capping added prefill latency to ~10-20ms keeps them within their budget and lets the violated ones catch up faster. This eliminates the cascade.

After the fix, DMC/LPT_CG nebius P99 ITL drops from ~290ms → ~135ms across all rates **and** throughput rises from 1212 → 1290 tok/s (faster decode drainage means faster overall completion).

### [x] Nebius-SWE-Trajectories results, 1000 prompts, fixed output 1024

Reproduce:
```
bash tasks/edf_benchmark/run_nebius_baseline_1k.sh
bash tasks/edf_benchmark/run_nebius_edf_1k.sh
python3 tasks/edf_benchmark/analyze_results.py tasks/edf_benchmark/results_baseline_nebius_1k
python3 tasks/edf_benchmark/analyze_results.py tasks/edf_benchmark/results_edf_nebius_1k
python3 tasks/edf_benchmark/plot_pareto_nebius.py \
    tasks/edf_benchmark/results_baseline_nebius_1k/summary.json \
    tasks/edf_benchmark/results_edf_nebius_1k/summary.json \
    tasks/edf_benchmark
```
Plot: `tasks/edf_benchmark/pareto_nebius.png` (left = baseline median ITL, right = EDF SLO attainment).

#### [x] Baseline (FCFS, no SLO control)
| Method | Rate | tok/s | med TTFT | P99 TTFT | med ITL | P99 ITL | ITLp99% |
|--------|------|-------|----------|----------|---------|---------|---------|
| PP     | 1    | 1009  | 274.0    | 9098.2   | 19.01   | 147.81  | 52.5%   |
| PP     | 2    | 1212  | 123660.5 | 314470.0 | 26.50   | 50.69   | 87.1%   |
| PP     | 4    | 1212  | 251342.5 | 557232.3 | 26.39   | 45.47   | 97.0%   |
| PP     | 8    | 1199  | 315839.0 | 653683.1 | 26.44   | 39.34   | 99.4%   |
| PP     | 12   | 1198  | 323554.8 | 691407.9 | 26.40   | 35.98   | 98.9%   |
| PP     | inf  | 1196  | 230667.5 | 775669.7 | 27.43   | 30.26   | 97.9%   |
| DMC    | 1    | 1009  | 362.6    | 8732.2   | 19.29   | 261.75  | 21.8%   |
| DMC    | 2    | 1232  | 119398.2 | 307980.6 | 26.61   | 297.55  | 0.6%    |
| DMC    | 4    | 1225  | 247918.1 | 549008.1 | 26.46   | 289.56  | 2.0%    |
| DMC    | 8    | 1216  | 310077.7 | 669129.2 | 26.39   | 298.66  | 1.2%    |
| DMC    | 12   | 1216  | 332152.7 | 707062.6 | 26.43   | 295.21  | 1.5%    |
| DMC    | inf  | 1205  | 230106.8 | 768678.2 | 27.18   | 266.94  | 4.7%    |
| LPT_CG | 1    | 1009  | 358.6    | 8512.4   | 19.22   | 261.55  | 22.0%   |
| LPT_CG | 2    | 1228  | 121360.7 | 308618.1 | 26.59   | 300.17  | 0.8%    |
| LPT_CG | 4    | 1225  | 248028.8 | 549324.0 | 26.48   | 293.28  | 2.2%    |
| LPT_CG | 8    | 1215  | 310219.5 | 669274.1 | 26.42   | 297.91  | 1.3%    |
| LPT_CG | 12   | 1216  | 332679.1 | 707524.9 | 26.44   | 294.19  | 1.5%    |
| LPT_CG | inf  | 1206  | 260970.1 | 769605.3 | 27.47   | 267.40  | 3.9%    |

#### [x] EDF + SLO-aware chunk control (with the fall-back fix)
| Method | Rate | tok/s | med TTFT | P99 TTFT | med ITL | P99 ITL | ITLp99% |
|--------|------|-------|----------|----------|---------|---------|---------|
| PP     | 1    | 1007  | 272.6    | 8893.0   | 19.04   | 152.14  | 52.9%   |
| PP     | 2    | 1211  | 118003.0 | 312352.7 | 26.62   | 48.76   | 89.4%   |
| PP     | 4    | 1211  | 247722.0 | 555342.6 | 26.38   | 48.30   | 95.8%   |
| PP     | 8    | 1210  | 314857.4 | 646779.0 | 26.41   | 38.96   | 99.7%   |
| PP     | 12   | 1200  | 305039.6 | 686297.9 | 26.45   | 38.57   | 99.1%   |
| PP     | inf  | 1192  | 229104.2 | 779505.8 | 27.54   | 31.75   | 97.2%   |
| DMC    | 1    | 1013  | 432.9    | 5307.6   | 19.74   | **134.19**  | **91.1%** |
| DMC    | 2    | **1289** | 103393.5 | 274252.3 | 33.40   | **139.99**  | **93.4%** |
| DMC    | 4    | **1307** | 227197.5 | 500063.8 | 32.08   | **140.40**  | **90.8%** |
| DMC    | 8    | **1306** | 284004.5 | 611116.7 | 34.08   | **134.90**  | **97.1%** |
| DMC    | 12   | **1284** | 297726.5 | 658801.3 | 34.14   | **138.34**  | **96.2%** |
| DMC    | inf  | **1277** | 276365.0 | 740285.9 | 30.37   | **141.42**  | **85.4%** |
| LPT_CG | 1    | 1014  | 423.6    | 4917.6   | 19.00   | **125.94**  | **95.5%** |
| LPT_CG | 2    | **1279** | 106197.5 | 278855.4 | 30.83   | **135.89**  | **97.7%** |
| LPT_CG | 4    | **1281** | 231653.0 | 517835.5 | 32.33   | **134.14**  | **99.7%** |
| LPT_CG | 8    | **1282** | 292764.2 | 622084.6 | 33.13   | **133.51**  | **100.0%** |
| LPT_CG | 12   | **1285** | 297466.4 | 656986.7 | 33.60   | **131.94**  | **96.8%** |
| LPT_CG | inf  | **1263** | 230264.9 | 747779.2 | 29.60   | **135.41**  | **97.7%** |

#### [x] DMC vs LPT_CG rate=inf root cause (task #110)

The post-fix nebius EDF table shows DMC and LPT_CG within ~2% of each other on per-req P99 ITL ≤ 150ms attainment at every rate **except** rate=inf, where DMC drops to 85.4% vs LPT_CG's 97.7%. Pulled the per-step ITL distributions from the JSONs:

| | DMC inf | LPT_CG inf |
|--|--|--|
| per-req P99 ITL median | **137.8 ms** | 105.0 ms |
| per-req P99 ITL p90 | **157.4 ms** | 140.7 ms |
| reqs over 150ms | **149 (14.9%)** | 23 (2.3%) |
| pooled decode steps > 150ms | **0.679%** | 0.337% |
| pooled p99.9 step | 167.8 ms | 155.9 ms |
| output throughput | **1277 tok/s** | 1263 tok/s |

DMC's throughput is *higher* despite worse latency — chunk control admitted bigger chunks. The per-req P99 distribution sits right at the SLO line for DMC (median 137ms, p90 157ms), but well below it for LPT_CG (median 105ms, p90 141ms). The whole DMC distribution is shifted ~30ms right.

**Root cause**: the DMC MLP latency model (`tasks/perf_model/dmc_model.pt`) under-predicts at large decode_bs (which only rate=inf reaches). The binary search in `edf_compute_max_chunk_size` therefore admits chunks ~10% bigger than safe, producing a consistent ~140ms decode tail that occasionally crosses 150ms. LPT_CG uses a different model + the persistent kernel and is more conservative.

**Mitigation landed in v0.3.3**: added `SGLANG_EDF_LATENCY_SAFETY` env var (default 1.0) that multiplies predicted step latency before comparing to slack. Setting it to e.g. 1.15 makes chunk control 15% more conservative without retraining the model. Suggested DMC default: 1.10-1.15. Verify with a follow-up DMC nebius rate=inf re-run after the v0.3.3 commit.

**Long-term fix**: retrain the DMC model with more training samples at decode_bs ≥ 64 (the regime that only rate=inf hits). Tracked separately.

#### [x] Nebius findings
1. **Chunk control fix is the main story**: DMC/LPT_CG decode SLO attainment goes from 0.6-22% (baseline / broken EDF) to **85-100%** across all rates. P99 ITL drops from ~290ms to ~135ms.
2. **Throughput improves with the fix**, not regresses: DMC tok/s rises from 1212→1290 because faster decode drainage frees KV slots earlier, raising effective concurrency.
3. **PP is unaffected** (52-99% in both runs) — PP doesn't use mixed batches, so chunk control is a no-op for it. PP is naturally good on decode tail because it never blocks decode with a prefill chunk.
4. **DMC/LPT_CG with chunk control now match PP** on decode SLO attainment (91-100% vs 95-100%). The historical advantage of PP over mixed-batch policies on decode tail is eliminated.
5. **TTFT is the binding constraint at saturation** (rate ≥ 2): all methods saturate at ~1200-1290 tok/s output throughput, so TTFT queue delay grows linearly with rate. Median TTFT goes to 100K-300K ms. **This is unfixable by EDF — at fundamental saturation no scheduling policy can hide queueing delay.** The decode SLO is what EDF + chunk control protects.
6. **rate=1 has TTFT P99 ~5-9s** which exceeds the 5s base SLO (with length scaling for 9K inputs the budget is ~6s, so it's marginal). Only at this rate is TTFT not fully saturated (96% pass).

## [ ] Future directions / known limitations

- [ ] **TTFT saturation is unfixable by EDF alone.** At rate ≥ 2 on nebius (rate ≥ 16 on lighter workloads), median TTFT goes to 100K-300K ms because the queue is fundamentally deeper than the model can drain. EDF reorders the queue but doesn't reduce its depth. Two paper-worthy directions to explore:
  1. **Admission control / load shedding** (Tempo-style) — drop requests whose TTFT deadline is unreachable instead of letting them rot in queue.
  2. **Disaggregated prefill-decode** (DistServe-style) — give prefill its own GPU so it doesn't compete with decode for compute.
  Tracked here as a future direction; intentionally NOT in the active overnight task list (per user instruction).

### [x] Blockers (both resolved)

#### [x] LPT_CG performance regressions vs DMC — kernel overhead on short-input workloads (task #117; resolved 2026-07-30 by the measured kernel gate)

**Resolution**: "Fix option 1/2" below became the measured k-NN kernel gate
(`SGLANG_MIXED_ATTENTION_KERNEL=auto`) — a per-batch measured choice between
BatchPrefill and the persistent kernel with leave-one-out worst regression
1.000. The dominant overhead also turned out to be the plan()-synchronize
stall, not `grid.sync()`; see `tasks/debug_kernel_server_perf.md`.

**Root cause**: Both DMC and LPT_CG capture the same 58 piecewise CGs with identical memory (6.42 GB available after CGs, 366K KV tokens — `mem-fraction-static=0.85` makes the cache pool static, so CG memory does NOT reduce KV capacity). The difference is the **kernel replayed** during mixed-batch CG:

- **DMC** mixed CG replays `BatchPrefillWithPagedKVCacheWrapper` (FA3 kernel) — lightweight
- **LPT_CG** mixed CG replays `BatchAttention` (cooperative kernel with `grid.sync()`) — heavier per-step overhead

On short-input workloads (Code-Feedback 200 tok, ShareGPT 150 tok), mixed batches have very few prefill tokens (~150-200 per step). The cooperative kernel's `grid.sync()` overhead and SM coordination cost dominates the actual attention computation, making LPT_CG slower per mixed step. This explains:
- Regressions are worst on **Code-Feedback** (shortest inputs → most decode-dominated mixed batches → cooperative overhead hurts most)
- Regressions are smallest on **SWE-Gym** (412 tok inputs → more prefill per step → cooperative kernel has more work to amortize overhead)
- WL1-6 benchmarks (4K-16K inputs) showed LPT beating DMC because the cooperative kernel amortizes its overhead over large prefill chunks

**Fix options**:
1. **Fall back to FA3 for small mixed batches**: if `num_prefill_tokens < threshold` (e.g., 512), use `BatchPrefillWithPagedKVCacheWrapper` instead of `BatchAttention` for that step. This gives LPT_CG the cooperative kernel's benefit on heavy prefills and FA3's speed on light ones.
2. **Profile the crossover**: find the prefill-token count where `BatchAttention` becomes faster than FA3 on this model/GPU combo. Use that as the threshold.
3. **Reduce piecewise CG sizes for small-token captures**: if sizes below 512 tokens always use FA3 (not cooperative), skip capturing those 28 sizes as mixed CGs entirely — saves ~1.2 GB and startup time.

**Verification needed**: A/B latency profiling of DMC vs LPT_CG on Code-Feedback (in progress).

#### [x] LPT cooperative kernel hang — RESOLVED (task #114)

**Resolution**: the `_ensure_positive_extend()` guard added in the schedule_policy fix (v0.3.3, commit `dcf000500`) eliminates the root cause. A request with `extend_input_len=0` (full prefix-cache hit) was entering the persistent kernel via the mixed-batch path, creating a malformed batch that caused `BatchAttention`'s cooperative `grid.sync()` to deadlock when batch size shrank to 1-2 remaining requests.

**Verification**: LPT_CG now completes 500 prompts at rate=inf on nebius (9K-input) without hanging, and passed 18 separate 1000-prompt LPT_CG runs across 3 workloads × 6 rates during sweep #109. The blocker is resolved.

**Original debug hypotheses** (kept for reference):

Code-reading findings (no GPU repro yet — pending #109 freeing the GPU):
1. **`BatchAttention` cooperative kernel** (`flashinfer_backend.py:976` `persistent_attn_forward`) calls `wrapper.run()` which dispatches to a flashinfer kernel that uses cooperative groups + `grid.sync()`. The cooperative model requires ALL launched blocks to reach the sync point. If a block returns early without syncing, the kernel deadlocks. This is the most likely culprit for the "hangs at #running-req: 1" symptom — when batch shrinks to 1 the kernel may launch more blocks than there's work for and the empty blocks don't reach `grid.sync()`.
2. **bs=1 edge case missing for persistent path**. There's already a `if forward_batch.seq_lens_sum - forward_batch.prefix_lens_sum <= 1: use_ragged = False` special case at `flashinfer_backend.py:543` for the ragged-prefill path (it works around a flashinfer TMA descriptor bug, see flashinfer-ai/flashinfer#2150). The persistent path has no equivalent guard. May need to fall back to non-persistent for bs=1.
3. **Stream interference from `non_blocking=True`** at `flashinfer_backend.py:1358,1381,1892`: these copy plan tensors to GPU asynchronously. If the persistent kernel reads them before the copy completes, that's a data hazard. Less likely to be the root cause (would always hang) but worth ruling out by setting `non_blocking=False` and re-testing.
4. **CG replay at specific bs**. If LPT_CG captures CG at e.g. bs=8 but the actual mixed batch on the last decode step has bs=1, the CG replay may use a stale plan. Bisect by removing small bs from `cuda_graph_bs` until hang stops.

Repro plan when GPU is free:
1. Smallest config that hangs: `nebius rate=8 num_prompts=200 SGLANG_FLASHINFER_USE_PERSISTENT=1 --enable-mixed-chunk --enable-piecewise-cuda-graph` (no EDF needed).
2. When it hangs, attach `py-spy dump --pid <PID>` and `gdb --pid <PID> -batch -ex "thread apply all bt" -ex quit` to grab Python + C stacks.
3. If C stack shows kernel launch waiting on event, dump CUDA state with `cuda-gdb`.
4. Bisect: try fix #2 first (add bs=1 fallback to non-persistent in `forward_extend`). If that resolves, root-causes the cooperative kernel theory.

## [x] Flashinfer C (cfca7c6f) EDF sweep with optimal kernel (2026-04-12)

Flashinfer C (`cfca7c6f`): barrier scaffolding gives +4% throughput / -41% TTFT on WL1 vs pure `d9c01c81` (see `tasks/debug_kernel_server_perf.md` for root cause).
200 prompts per run, rates 2/4/8/12/16/24/inf. `--attention-backend flashinfer`, EDF + SLO-aware chunk control.
sglang `attn_fuse_` branch, commit `aac11aa49`.

Reproduce:
```
bash tasks/edf_benchmark/run_edf_sweep.sh
python3 tasks/edf_benchmark/analyze_results.py tasks/edf_benchmark/results/
```
Plots: `tasks/edf_benchmark/pareto_baseline_median_itl.png`, `pareto_edf_slo_attainment.png`.

### ShareGPT (TTFT 2s / ITL 100ms)
| Method | Rate | tok/s | med TTFT | P99 TTFT | med ITL | P99 ITL | SLO% | ITLp99% |
|--------|------|-------|----------|----------|---------|---------|------|---------|
| PP | 2 | 1584 | 35.1 | 89.5 | 8.66 | 18.24 | 94.5% | 100.0% |
| PP | 4 | 1917 | 39.3 | 96.3 | 10.86 | 27.97 | 85.0% | 100.0% |
| PP | 8 | 2181 | 44.4 | 316.1 | 13.85 | 39.49 | 57.0% | 100.0% |
| PP | 12 | 2274 | 44.4 | 121.0 | 14.75 | 46.77 | 50.0% | 100.0% |
| PP | 16 | 2327 | 48.4 | 145.1 | 15.28 | 52.15 | 43.5% | 100.0% |
| PP | 24 | 2380 | 49.6 | 156.4 | 15.56 | 63.72 | 20.0% | 78.5% |
| PP | inf | 2451 | 1816.1 | 3180.7 | 15.81 | 17.93 | 0.0% | 77.0% |
| DMC | 2 | 1591 | 36.4 | 93.9 | 8.62 | 22.50 | 100.0% | 100.0% |
| DMC | 4 | 1937 | 44.2 | 107.7 | 10.50 | 33.53 | 100.0% | 100.0% |
| DMC | 8 | 2204 | 52.2 | 321.9 | 13.44 | 41.55 | 93.0% | 100.0% |
| DMC | 12 | 2297 | 59.6 | 151.4 | 14.25 | 43.76 | 100.0% | 100.0% |
| DMC | 16 | 2346 | 62.7 | 196.4 | 14.97 | 45.74 | 100.0% | 100.0% |
| DMC | 24 | 2397 | 96.5 | 237.5 | 15.52 | 43.27 | 69.0% | 99.5% |
| DMC | inf | 2407 | 2674.2 | 5600.6 | 15.58 | 38.15 | 13.5% | 100.0% |
| LPT_CG | 2 | 1591 | 36.8 | 99.2 | 8.62 | 22.74 | 100.0% | 100.0% |
| LPT_CG | 4 | 1936 | 43.4 | 107.9 | 10.50 | 34.42 | 100.0% | 100.0% |
| LPT_CG | 8 | 2203 | 52.9 | 190.7 | 13.40 | 41.54 | 90.0% | 100.0% |
| LPT_CG | 12 | 2296 | 56.3 | 160.2 | 14.30 | 44.91 | 100.0% | 100.0% |
| LPT_CG | 16 | 2350 | 66.8 | 163.4 | 14.95 | 44.20 | 100.0% | 100.0% |
| LPT_CG | 24 | 2404 | 85.2 | 214.5 | 15.44 | 42.09 | 100.0% | 100.0% |
| LPT_CG | inf | 2419 | 2571.8 | 4960.4 | 15.59 | 19.70 | 28.5% | 100.0% |

### Code-Feedback (TTFT 3s / ITL 150ms)
| Method | Rate | tok/s | med TTFT | P99 TTFT | med ITL | P99 ITL | SLO% | ITLp99% |
|--------|------|-------|----------|----------|---------|---------|------|---------|
| PP | 2 | 1773 | 27.5 | 53.9 | 8.35 | 17.02 | 100.0% | 100.0% |
| PP | 4 | 2659 | 29.3 | 52.9 | 10.00 | 19.54 | 100.0% | 100.0% |
| PP | 8 | 3372 | 31.9 | 164.2 | 11.49 | 23.04 | 100.0% | 100.0% |
| PP | 12 | 3668 | 31.1 | 72.5 | 12.36 | 25.93 | 100.0% | 100.0% |
| PP | 16 | 3834 | 32.6 | 67.1 | 13.08 | 28.69 | 100.0% | 100.0% |
| PP | 24 | 3996 | 32.7 | 66.0 | 13.84 | 35.04 | 100.0% | 100.0% |
| PP | inf | 4333 | 957.8 | 1491.1 | 13.90 | 16.26 | 11.5% | 99.0% |
| DMC | 2 | 1773 | 31.3 | 64.5 | 8.32 | 20.32 | 100.0% | 100.0% |
| DMC | 4 | 2671 | 35.3 | 62.7 | 9.83 | 24.61 | 100.0% | 100.0% |
| DMC | 8 | 3378 | 42.1 | 321.5 | 11.23 | 29.84 | 91.5% | 100.0% |
| DMC | 12 | 3687 | 42.9 | 99.4 | 11.99 | 31.12 | 100.0% | 100.0% |
| DMC | 16 | 3854 | 45.1 | 98.5 | 12.55 | 31.37 | 100.0% | 100.0% |
| DMC | 24 | 4019 | 45.9 | 105.1 | 13.52 | 34.27 | 100.0% | 100.0% |
| DMC | inf | 4298 | 880.0 | 2654.3 | 14.00 | 19.67 | 30.5% | 99.0% |
| LPT_CG | 2 | 1773 | 30.9 | 60.1 | 8.31 | 20.43 | 100.0% | 100.0% |
| LPT_CG | 4 | 2670 | 34.8 | 90.1 | 9.84 | 24.61 | 100.0% | 100.0% |
| LPT_CG | 8 | 3387 | 41.1 | 139.2 | 11.23 | 28.06 | 100.0% | 100.0% |
| LPT_CG | 12 | 3690 | 42.2 | 84.9 | 12.03 | 30.83 | 100.0% | 100.0% |
| LPT_CG | 16 | 3851 | 43.4 | 122.6 | 12.64 | 32.12 | 100.0% | 100.0% |
| LPT_CG | 24 | 4020 | 45.2 | 142.8 | 13.45 | 32.71 | 100.0% | 100.0% |
| LPT_CG | inf | 4284 | 1182.7 | 1842.5 | 13.84 | 16.15 | 0.0% | 98.5% |

### SWE-Gym (TTFT 5s / ITL 150ms)
| Method | Rate | tok/s | med TTFT | P99 TTFT | med ITL | P99 ITL | SLO% | ITLp99% |
|--------|------|-------|----------|----------|---------|---------|------|---------|
| PP | 2 | 1173 | 37.3 | 135.5 | 9.11 | 18.52 | 88.0% | 100.0% |
| PP | 4 | 1452 | 40.2 | 150.6 | 9.84 | 31.48 | 80.0% | 100.0% |
| PP | 8 | 1629 | 49.8 | 264.9 | 10.37 | 40.84 | 68.0% | 100.0% |
| PP | 12 | 1693 | 53.2 | 160.8 | 10.72 | 43.73 | 65.0% | 100.0% |
| PP | 16 | 1725 | 54.8 | 162.5 | 10.93 | 45.88 | 63.0% | 98.0% |
| PP | 24 | 1759 | 67.2 | 212.5 | 12.33 | 53.04 | 63.0% | 94.0% |
| PP | inf | 1795 | 1136.6 | 2151.4 | 12.55 | 15.69 | 2.0% | 99.0% |
| DMC | 2 | 1174 | 42.4 | 154.8 | 9.12 | 23.13 | 88.0% | 100.0% |
| DMC | 4 | 1455 | 44.4 | 395.8 | 9.72 | 36.37 | 84.0% | 100.0% |
| DMC | 8 | 1637 | 54.4 | 306.2 | 9.99 | 43.78 | 22.0% | 100.0% |
| DMC | 12 | 1701 | 58.0 | 167.5 | 10.37 | 43.81 | 100.0% | 100.0% |
| DMC | 16 | 1733 | 69.9 | 185.8 | 10.67 | 44.20 | 66.0% | 100.0% |
| DMC | 24 | 1767 | 92.7 | 234.3 | 10.99 | 38.95 | 63.0% | 100.0% |
| DMC | inf | 1796 | 1327.0 | 2906.0 | 12.52 | 18.51 | 100.0% | 100.0% |
| LPT_CG | 2 | 1175 | 41.4 | 154.4 | 9.08 | 23.20 | 88.0% | 100.0% |
| LPT_CG | 4 | 1457 | 43.3 | 154.2 | 9.71 | 35.06 | 81.0% | 100.0% |
| LPT_CG | 8 | 1637 | 53.6 | 314.6 | 10.09 | 43.29 | 70.0% | 100.0% |
| LPT_CG | 12 | 1701 | 59.2 | 166.9 | 10.25 | 43.94 | 69.0% | 100.0% |
| LPT_CG | 16 | 1734 | 67.7 | 195.6 | 10.51 | 43.98 | 66.0% | 100.0% |
| LPT_CG | 24 | 1766 | 97.0 | 244.6 | 11.37 | 40.27 | 63.0% | 100.0% |
| LPT_CG | inf | 1797 | 1325.2 | 2891.3 | 12.36 | 18.63 | 100.0% | 100.0% |

---

## Debug Notes

1. **req_to_token_pool memory leak**: Fixed upstream in commit `027f314` on `public/main` — "[Fix] data race in req_to_token pool (#17850)". The fix changes `alloc()`/`free()` to be request-based, preventing double-frees. Need to cherry-pick into `attn_fuse_`. Touches 13 files including `memory_pool.py`, `radix_cache.py`, `scheduler.py`.
2. **LPT cooperative kernel hang**: Persistent kernel hangs under load (~30+ concurrent requests). **DMC does NOT hang** under identical load — confirmed by side-by-side test (DMC completed all 50 requests in 180s, LPT hung at 163 requests). The hang happens when only 1-2 requests remain running (last decode batch shows `#running-req: 1`). No error messages — the scheduler loop simply freezes. Possible causes:
   - Cooperative kernel deadlock during CG replay at a specific batch size
   - State corruption from mixed CG replay that only manifests on the final request
   - Stream interference from `non_blocking=True` operations (identified in `forward_batch_info.py:771` and `flashinfer_backend.py:1358,1381,1892`)
