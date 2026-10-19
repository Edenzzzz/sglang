# Server Performance Model for Prefill-Decode Colocation

## Goal

After optimizing persistent v3 performance and studying the workloads in/out lengths, our goal is to run full **prefill-decode colocation** at large scale instead of PD disaggregation.

People generally believe disaggregation is necessary to meet tight SLO requirements in large scale serving because of HOL: long prefill can block decode, and chunked prefill/mixed chunk hurts throughput due to repeated kernel launches and wasted compute due to padding, respectively. However, disaggregation wastes compute resources on decode nodes (cannot fully use tensor cores), is less fault-tolerant (prefill failures propagate to decode nodes), and less flexible (demand peak/in-out length changes).

Our method aims to achieve an optimal pareto curve: high throughput, stable decode SLO attainment, and flexible deployment. To do this, we use a profiler (`python/sglang/srt/model_executor/overlap_profiler.py`, supports POD and persistent, but might need some fixes) to profile the server-level latency when facing a load totaling `n_p` new prefill tokens with `r_p` prefill prefix length (already cached), `r_d` decode KV length and `decode_bs`. After getting the data on comprehensive sequence lengths, we train an MLP (`perf_model/train_perf_model_mlp.py`) to predict the runtime of each node so as to determine how many decode and prefill tokens to admit for this batch, based on the corresponding P/D SLO. This allows flexibly serving mixed workloads and fully utilizing compute on all nodes with mixed batch.

> **Status (2026-08).** For the fused persistent path, the trained-MLP chunk
> controller is superseded by the closed-form analytic model
> (`python/sglang/srt/managers/overlap_cost_model.py`,
> `SGLANG_EDF_LATENCY_MODEL=analytic`):
> `T_step = c0 + c1·tokens + n_layers·(max(T_p,T_d) + β·min)`, seven physical
> constants fit once from microbenchmarks, 3.1% MAPE, no per-workload
> profiling. The MLP checkpoints remain relevant only for the router
> `/get_load` cost signal below. Note the same closed form does *not* exist
> for the split-kernel baseline (best parts-based fit 13.3% MAPE / 43%
> worst), which is part of the paper story.

## TODOs

- [ ] **One step model for the 4-mode mixed step, shared by router and worker (user, 2026-10-19).** Today three
  different models are in play and none is the joint model: the H20 router uses lookup tables (cold-prefill curve,
  decode ITL grid, one `--chunk-interference` scalar); the S4 predictability numbers (mixed MAPE 26.7% ours vs 53.5%
  stock MTP, decode 3.6% vs 2.7%) come from pdmux's 5-term linear form fit offline (`fit_pdmux_step.py`); the worker's
  decode-first chunk rule (`apply_edf_itl_slo_chunk.py`) uses a hand-set subset with a fixed decode base. The MLPs
  below are from the persistent-kernel era and were never fit on 4-mode steps. Plan:
  - data: STEPTRACE (`STEP_TRACE=1`) from 4-mode and stock-MTP serving on H200 at several loads and chunk sizes;
    per step P new tokens, P prefix, #P requests, decode rows, decode KV, acceptance, GPU time and launch-to-launch gap;
  - candidates, fit on the first half and scored on the second (MAPE and p90, mixed and decode-only separately):
    (1) pdmux 5-term linear, (2) + MTP terms (verify tokens = rows x draft tokens, acceptance), (3) overlap form
    max(T_p, T_d) + beta * min (FA2 runs beside FA3; the persistent-kernel analytic model reached 3.1%), (4) MLP on
    the same features, (5) a tree ensemble;
  - deploy the winner in both places: router TTFT / ITL prediction from node-reported state (rows, sum KV,
    acceptance, prefill queue by deadline, free KV) piggybacked on responses with an idle heartbeat, and the
    worker's decode-first chunk rule (decode base from the live batch instead of a constant).

- [ ] **Wire overlap_profiler into scheduler** [DEPRIORITIZED 2026-08: the
  analytic model removed the chunk-control dependency on profiling; still
  needed only to validate the router cost signal]: `ProfilingRunner` exists in `overlap_profiler.py` but is NOT integrated into the server. Options: (a) add `FlushCacheReqInput`-style special request type + API endpoint + handler in `scheduler.py`, (b) add `--run-profiler` flag to `launch_server` that runs profiling during scheduler init before entering event loop, (c) standalone script that creates `ModelRunner` directly (like tests). Option (b) is simplest. The profiler requires `--enable-piecewise-cuda-graph`.
- [ ] **Fix profiler shape mismatch with persistent kernel**: Calling `/profile_seq_lens` on v3+PCG server crashes with `RuntimeError: output with shape [1] doesn't match the broadcast shape [389]`. The `ProfilingRunner.profile_one_seq_config` creates a ForwardBatch that's missing some fields the persistent kernel needs (possibly `extend_seq_lens` or `prefix_lens` tensor shapes). Need to debug the exact line and fix the ForwardBatch construction.
- [x] **Profile with persistent v3 + CG + PCG (old)**: Profiler ran 22,837/25,444 configs (90%) before server crash. Data: `Qwen_Qwen3-8B_6b6dccb5_20260329_220215_profile.csv`. Trained MLP (`tasks/perf_model/model.pt`): 3.4% MAPE (now superseded)
- [x] **Retrained LPT model** (commit 9544552f, 2026-04-06): Full 25,444 configs profiled with latest LPT kernel (static fallback + extra CG sizes). Data: `Qwen_Qwen3-8B_9544552f_20260406_005503_profile.csv`. Trained MLP (`tasks/perf_model/lpt_model.pt`):
  - **MAPE: 2.4%**, MAE: 3.26ms, P90 error: 7.32ms
  - Features: n_p, r_p, r_d, decode_bs, overlap
  - Target range: 14.34ms to 1394.86ms
  - Architecture: 3-layer MLP [128, 64, 32] with log1p target transform
  - Training: 200 epochs, Adam lr=1e-3, cosine schedule
  - Command: `python tasks/perf_model/train_perf_model_mlp.py --csv Qwen_Qwen3-8B_9544552f_20260406_005503_profile.csv --output tasks/perf_model/lpt_model.pt --epochs 200`
- [x] **DMC model** (commit 9544552f, 2026-04-06): Full 25,241 configs profiled with DMC (no persistent). Data: `Qwen_Qwen3-8B_9544552f_20260405_185729_profile.csv`. Trained MLP (`tasks/perf_model/dmc_model.pt`):
  - **MAPE: 2.3%**, MAE: 3.04ms, P90 error: 6.29ms
  - Features: n_p, r_p, r_d, decode_bs, overlap
  - Target range: 20.43ms to 1354.95ms
  - Training: 200 epochs, command: `python tasks/perf_model/train_perf_model_mlp.py --csv Qwen_Qwen3-8B_9544552f_20260405_185729_profile.csv --output tasks/perf_model/dmc_model.pt --epochs 200`
- [ ] Validate MLP predictions against actual server latency on representative workloads from `tasks/realistic_workload_analysis.md`
- [x] Use trained models for SLO-aware chunk size control (integrated in `edf_scheduler.py` via `SGLANG_EDF_LATENCY_MODEL` env var)

### 2026-10-12 fused-vs-unfused EDF check

The checked-in MTP linear step model (`heldout_mape=21.9%`) was exercised with
global acceptance EWMA on matched fused and unfused `loadsweep_min` arms. Both
completed 601 mixed steps with zero fallbacks/tracebacks. Fused mean TTFT was
2.9%/2.3% lower for chat/codex, but mean ITL was 0.6%/0.7% higher and codex
output throughput was 1.4% lower. This validates the comparison path, not the
claim that fusion makes EDF easier or better; the broader MLP validation TODO
remains open.

The EDF predictor interface now accepts an `acceptance_length` argument. The
online EWMA binding estimate is passed into chunk-budget predictions, while old
MLP/analytic models remain compatible; `fit_mtp_step.py` also accepts optional
`acc=<EWMA>` STEPTRACE fields and emits `c_acc` when present. This makes the
requested acceptance-length feature explicit without treating the old
21.9%-MAPE checkpoint as retrained evidence.

### Production target and router integration audit (2026-07-23)

The LPT/DMC checkpoints already time `model_runner.forward`, so their target
includes the full GPU model forward rather than attention alone. The profiler
still initializes attention metadata before timing, forcibly disables CUDA
graphs, and stops before sampling/output readiness. Its saved `overlap` feature
also has zero variance. The reported 2.4%/2.3% MAPE therefore validates the
non-graph GPU-forward domain, not production token-ready server cost. The next
calibration dataset must time the real graph/planner path through token
readiness; fixed overhead should remain only an initial prior.

Public upstream `main` at `b98a577f` retains the cache-aware imbalance gate:
when both absolute and relative thresholds trigger it selects minimum
`worker.load()`, otherwise it uses prefix affinity (and minimum load on a
miss). `worker.load()` is the number of active requests, not token or predicted
load. A model-based router should preserve that gate but compare incremental
predicted server cost inside the permitted imbalance region. `SelectWorkerInfo`
already carries tokenized input. `/get_load` now reports decode batch size,
aggregate decode KV, remaining expected decode tokens, pending prefill tokens,
aggregate prefix hits, and `predicted_forward_ms`. The prediction is the
checkpoint's full-model cost for the next admitted forward: pending prefill is
capped at the server chunk size, while prefix and decode KV use per-request
averages. It is explicitly `null` without a configured checkpoint and is not
presented as an unvalidated queue-drain estimate.

An H800 live check with the trained LPT checkpoint reported 291.87 ms for a
mixed snapshot with 42,478 pending prefill tokens, one decode row, and 1,062
decode KV tokens. This correctly predicts only the next 8,192-token admitted
chunk; an idle server reports zero rather than the MLP's meaningless
zero-feature intercept. `/get_load` also falls back to the
dispatched `last_batch` during overlap scheduling, rather than incorrectly
reporting zero active decodes, and no longer classifies fresh waiting requests
(whose `extend_input_len` is initially zero) as decode.

The scheduler now publishes this snapshot over its existing tokenizer IPC at
the server's 100 ms load-watch interval; `/get_load` serves the latest
per-DP-rank copy without sending a control request back through the saturated
scheduler. In a matched H800 N=20 peak-load run, 30 consecutive reads took
0.78–1.13 ms and their timestamps advanced at roughly 100 ms, replacing the
previous six-second synchronous delay. Total throughput was 2,921 tok/s versus
2,959 tok/s in the preceding no-publication smoke run (-1.3%, within run
noise), with exact output retokenization.

The signal is fresh enough for routing, but is still only one-forward cost:
the router's existing monitor defaults to 30-second polls and only updates
power-of-two policies. The router now parses the structured signal, including
its snapshot timestamp and age, while retaining compatibility with legacy
token-only workers. For multi-DP workers it uses the maximum predicted forward
cost and the oldest timestamp so neither load nor staleness is understated.
These fields are diagnostic only. Incremental request/queue cost must be
validated before replacing request-count load inside the existing
cache-imbalance guard.

### 2026-10-12 fused-vs-unfused EDF check

The checked-in MTP step model (`heldout_mape=21.9%`) was exercised with global
acceptance EWMA on matched fused and unfused `loadsweep_min` arms. Both completed
601 mixed steps with zero fallbacks/tracebacks. Fused mean TTFT was 2.9%/2.3%
lower for chat/codex, but mean ITL was 0.6%/0.7% higher and codex output
throughput was 1.4% lower. This validates the comparison path, not the claim
that fusion makes EDF easier or better; the broader MLP validation TODO remains
open.

The EDF predictor interface now accepts `acceptance_length`; the online EWMA
binding estimate is passed into chunk-budget predictions. `fit_mtp_step.py`
accepts optional `acc=<EWMA>` STEPTRACE fields and emits `c_acc`, while old
five-coefficient models remain compatible. Regression coverage is in
`test/manual/benchmark/test_mtp_step_acceptance_model.py`.
The scheduler's STEPTRACE records the current lifetime acceptance EWMA as
`acc=...`, so new calibration runs can actually fit the added coefficient.
