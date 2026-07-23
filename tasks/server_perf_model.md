# Server Performance Model for Prefill-Decode Colocation

## Goal

After optimizing persistent v3 performance and studying the workloads in/out lengths, our goal is to run full **prefill-decode colocation** at large scale instead of PD disaggregation.

People generally believe disaggregation is necessary to meet tight SLO requirements in large scale serving because of HOL: long prefill can block decode, and chunked prefill/mixed chunk hurts throughput due to repeated kernel launches and wasted compute due to padding, respectively. However, disaggregation wastes compute resources on decode nodes (cannot fully use tensor cores), is less fault-tolerant (prefill failures propagate to decode nodes), and less flexible (demand peak/in-out length changes).

Our method aims to achieve an optimal pareto curve: high throughput, stable decode SLO attainment, and flexible deployment. To do this, we use a profiler (`python/sglang/srt/model_executor/overlap_profiler.py`, supports POD and persistent, but might need some fixes) to profile the server-level latency when facing a load totaling `n_p` new prefill tokens with `r_p` prefill prefix length (already cached), `r_d` decode KV length and `decode_bs`. After getting the data on comprehensive sequence lengths, we train an MLP (`perf_model/train_perf_model_mlp.py`) to predict the runtime of each node so as to determine how many decode and prefill tokens to admit for this batch, based on the corresponding P/D SLO. This allows flexibly serving mixed workloads and fully utilizing compute on all nodes with mixed batch.

## TODOs

- [ ] **Wire overlap_profiler into scheduler**: `ProfilingRunner` exists in `overlap_profiler.py` but is NOT integrated into the server. Options: (a) add `FlushCacheReqInput`-style special request type + API endpoint + handler in `scheduler.py`, (b) add `--run-profiler` flag to `launch_server` that runs profiling during scheduler init before entering event loop, (c) standalone script that creates `ModelRunner` directly (like tests). Option (b) is simplest. The profiler requires `--enable-piecewise-cuda-graph`.
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

This signal is not yet safe to install in the router: under that same
saturated smoke run a synchronous `/get_load` response was delayed by about
six seconds, while the router's existing monitor defaults to 30-second polls
and only updates power-of-two policies. Production routing should publish the
snapshot asynchronously and validate incremental request/queue cost before
replacing request-count load inside the existing cache-imbalance guard.
