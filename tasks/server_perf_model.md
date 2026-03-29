# Server Performance Model for Prefill-Decode Colocation

## Goal

After optimizing persistent v3 performance and studying the workloads in/out lengths, our goal is to run full **prefill-decode colocation** at large scale instead of PD disaggregation.

People generally believe disaggregation is necessary to meet tight SLO requirements in large scale serving because of HOL: long prefill can block decode, and chunked prefill/mixed chunk hurts throughput due to repeated kernel launches and wasted compute due to padding, respectively. However, disaggregation wastes compute resources on decode nodes (cannot fully use tensor cores), is less fault-tolerant (prefill failures propagate to decode nodes), and less flexible (demand peak/in-out length changes).

Our method aims to achieve an optimal pareto curve: high throughput, stable decode SLO attainment, and flexible deployment. To do this, we use a profiler (`python/sglang/srt/model_executor/overlap_profiler.py`, supports POD and persistent, but might need some fixes) to profile the server-level latency when facing a load totaling `n_p` new prefill tokens with `r_p` prefill prefix length (already cached), `r_d` decode KV length and `decode_bs`. After getting the data on comprehensive sequence lengths, we train an MLP (`perf_model/train_perf_model_mlp.py`) to predict the runtime of each node so as to determine how many decode and prefill tokens to admit for this batch, based on the corresponding P/D SLO. This allows flexibly serving mixed workloads and fully utilizing compute on all nodes with mixed batch.

## TODOs

- [ ] **Wire overlap_profiler into scheduler**: `ProfilingRunner` exists in `overlap_profiler.py` but is NOT integrated into the server. Options: (a) add `FlushCacheReqInput`-style special request type + API endpoint + handler in `scheduler.py`, (b) add `--run-profiler` flag to `launch_server` that runs profiling during scheduler init before entering event loop, (c) standalone script that creates `ModelRunner` directly (like tests). Option (b) is simplest. The profiler requires `--enable-piecewise-cuda-graph`.
- [ ] **Fix profiler shape mismatch with persistent kernel**: Calling `/profile_seq_lens` on v3+PCG server crashes with `RuntimeError: output with shape [1] doesn't match the broadcast shape [389]`. The `ProfilingRunner.profile_one_seq_config` creates a ForwardBatch that's missing some fields the persistent kernel needs (possibly `extend_seq_lens` or `prefix_lens` tensor shapes). Need to debug the exact line and fix the ForwardBatch construction.
- [ ] **Profile with persistent v3 + CG + PCG**: After fixing the shape error, run profiler with full v3 stack. 25,444 configs. Train MLP and evaluate accuracy.
- [ ] Validate MLP predictions against actual server latency on representative workloads from `tasks/realistic_workload_analysis.md`
- [ ] Use trained model to determine optimal prefill/decode admission policy given P/D SLO targets
