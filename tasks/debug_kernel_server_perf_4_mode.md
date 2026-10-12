# Current goal

Fused FA3+FA2 SM pre-assignment attention for a 4-mode MTP (EAGLE3) serving schedule, optimised
on real serving workloads, then the EDF model and the PD-colocation argument
(`tasks/pd_colocation_vs_disagg.md`). Bar: >=30% TTFT and ITL, >=10% throughput on realistic
chat + TraceLab workloads. Single H200 trial `095f01a3e1585eea` (`vibecuda`); H20 multi-node only
after the kernel items close (<=16 GPUs).

**Where it stands (2026-10-13):** under load the fused 4-mode arm beats stock MTP by TTFT
-31..-79%, throughput +11..+36%, per-token latency -19..-30%; below capacity the arms are equal.
TTFT and throughput clear the bar wherever the server is loaded; per-token latency clears it only
at some points. Graphs-on 4-mode runs need a mixed-step barrier whose root cause is open.

## Tasks, in priority order

A and B are the two kernel items; S is the server/workload evaluation the user specified.
Supporting gates are prerequisites, not substitutes. `[x]` = measured, `[~]` = partly, `[ ]` = not run.

### [~] A. Ultra fine-grained tile dispatcher

- [x] **A-fix, FA2 out of the co-compiled kernel (2026-10-12):** FA2 spilled inside the fused
  kernel (168 regs, 4200 B stack, 21x slower per item than `BatchAttention`); now FA3 grid on
  `sm_split` CTAs + `FA2ExecutorKernel` at native geometry as its PDL dependent
  (`FLASHINFER_FUSED_COLAUNCH=1`). 3.3-6.5x on FA2 rows, every gate exact; still 1.2-2x behind the
  non-persistent unfused two-launch control. ([evidence](debug_kernel_server_perf/resume_20261012/colaunch_20261012.md))
- [x] **A1, FA3 q64 vs q128 per work item (2026-10-12):** closed by measurement -- q64 never beats
  q128 beyond noise and loses up to 2.3x; nothing to select. Not built for that reason.
- [x] **A2, FA2 tile classes (2026-10-12):** 48/32/16 refuted (tile 48 = 1 q-warp x 4 kv-warps,
  3x accumulator, spills; 1.6-2.2x slower at q8); **64/32/16** adopted (96 B stack). Per-item tail
  dispatch kept (bit-identical outputs). FA2 split-KV chunk = `auto` (sized to the FA2 CTA slots;
  matches or beats the best fixed chunk on every fixture). ([evidence](debug_kernel_server_perf/resume_20261012/colaunch_20261012.md))
- [~] **A3, fitted FA2/FA3 dispatch rule (2026-10-12):** winner map on the fixed kernel, 216/216
  exact: FA2 wins every verify-width row; rule "FA3 iff packed_qo >= 112 and FA3 work >= 3.5M"
  agrees on 69/72 cells (worst loss 6%) vs 55/72 (98%) for the serving rule -- **but loses in
  serving** (-3.5% throughput, +29% TTFT) because batches it makes one-sided run the stock path.
  Next: refit with the one-sided path priced in, or route one-sided batches through the co-launch.
  ([table](debug_kernel_server_perf/resume_20261012/a3_winner_map/a3_v2_table_20261012.txt))

### [~] B. Fuse draft/verify epilogue ops

- [~] **B1, small-op tail:** greedy EAGLE verify has an opt-in fused argmax/acceptance/length
  epilogue; sampling, gathers, cache writes and tree build remain separate.
- [x] **B2, cross-role backfill (2026-10-12):** co-launch hardware backfill worth up to 2.9x when FA2
  work exceeds the slots beside FA3, nothing on small mixes; in-kernel `tma_store_wait` backfill is
  correct and worth 1.4-5.5x over none, but that whole path is 2-6x slower. Co-launch ships.
  ([evidence](debug_kernel_server_perf/resume_20261012/colaunch_20261012.md))
- [ ] **B3, multicast:** not implemented (`mcast_mask=0`); needs page-size-128 paged TMA. A
  2026-10-07 ceiling on the old two-executor FA3 (every K/V gather skipped) was only 1.6-1.8%
  faster -- FA3-routed rows are compute-bound -- and FA2 verify rows have no shared reads (one
  q-tile per row, disjoint split-KV chunks). Re-measuring that ceiling on the current kernel
  before building anything: `FLASHINFER_FA3_SKIP_KV_LOAD` diagnostic
  (`fa3_skip_kv_load.patch`), `b3_multicast_ceiling_20261013.sh`, queued after the full-length run.

### [ ] S. Server / workload evaluation

Arms everywhere: stock MTP, unfused 4-mode, fused 4-mode (co-launch, 64/32/16, auto chunk);
EAGLE3 chain q4; CUDA graphs on; 4-mode arms with the measured mixed-step barrier (see gates);
each arm reports fallbacks and composition counters. Analyzer: `drivers/serving/eval_curves.py`.

- [~] **S1 x S2, rate sweep x context mixtures (running since 2026-10-12 19:22,
  `eval_matrix_s12_20261012.sh`):** chat 4-24/s + TraceLab agent at 0.1 sessions/s; per-tenant
  SLOs chat 2 s / 50 ms TPOT, agent 10 s / 80 ms. 0 fallbacks, 0 tracebacks so far.
  - [x] **mix32 (32-64K):** arms equal at 4-8/s (4-mode P99 ITL worse: a verify rides with a
    chunk). From 12/s fused vs stock: TTFT -47..-79%, throughput +11..+25%, TPOT -19..-28%, SLO
    attainment up to 2x (agent 0.83 vs 0.47 at 16/s); fused beats unfused 4-mode from 16/s.
  - [x] **mix64 (64-128K):** already saturated at 4/s (the agent tenant alone fills the GPU). At
    12-20/s fused vs stock: chat throughput +15..+36%, TTFT -31..-35%, TPOT -19..-30%; agent
    throughput +18..+26%, TPOT -11..-25%; fused beats unfused by 9-14% chat throughput at 16-20/s.
  - [ ] **mix128 (128-256K, YaRN x8):** running.
  - [ ] Pareto (throughput vs tok/s/user) and SLO-vs-rate figures, once the sweep completes.
- [ ] **S2b, full-length TraceLab (user, 2026-10-13):** the release as it is -- Claude Code context
  p50/p90/p99 133K/483K/923K (25% >= 256K), Codex 120K/204K/233K, outputs p50 ~200 tokens; the
  32-64K slice is ~16% of rounds. Loader now replays both providers, first rounds, and
  round-weighted sessions (replayed median 120-126K vs the release's 124K);
  `tracelab_full_env` scenario at YaRN x32 (1M tokens). Queued after the sweep.
  ([patch](debug_kernel_server_perf/resume_20261012/tracelab_full_replay.patch))
- [ ] **S3, standalone workloads:** ShareGPT, GSM8K, MATH-500, HumanEval, Code-Feedback, Mooncake
  tool-agent/conversation, LongBench v2 -- prepared on the pod, not run.
- [ ] **S4, EDF step model** re-fitted on the fused kernel and compared with the unfused schedule
  (goal point 3); earlier fits used the RoPE-corrupted model.
- [x] **Chunk size is not the ITL lever at saturation (2026-10-12):** chunk 8192/4096/2048 leaves
  both arms unchanged (prefill per step is many ~512-token prompts).
- [ ] **Direct baselines:** MuxWise (arXiv 2504.14489), semi-PD, NVIDIA Dynamo conditional
  disaggregation, sglang mixed-chunk and sglang PD -- see References.

### [~] Supporting gates

- [x] **RoPE root cause (2026-10-12):** the YaRN override dropped `rope_theta` (1e6 -> 1e4) under
  transformers 5.x; fixed in sglang's config loader with a regression test. All serving numbers
  before 2026-10-12 are void. ([evidence](debug_kernel_server_perf/resume_20261012/a2_serving_ropefix_20261012.md))
- [x] **Serving-output gate (q4 and q8; graphs off and on):** every 4-mode arm agrees with stock on
  the long prompts; first tokens are the HF reference argmax or within one bf16 ulp.
- [~] **Graphs-on 4-mode crash (2026-10-12):** both 4-mode arms crash under load (async index
  assert). Bisect: a device sync after mixed steps (or launch-blocking) cures it; phase syncs,
  no in-step graph, overlap off, synchronous flashinfer plans and a sync after non-mixed steps do
  not. A mixed step leaves GPU work in flight that the next step's host side races. **Root cause
  open**; measurements use the mixed-step barrier, which can only slow the 4-mode arms. Latent race
  of the same class found: `_fs_upload` rewrites a pinned host buffer an async copy may still read.
- [x] **Pod re-forked (2026-10-12):** `f096519a4d17aa1c` expired; `095f01a3e1585eea` rebuilt from the
  fork's upstream base + serving diff, reproducing the old pod's kernel results within 1%.

---

## Evidence index

- Per-result evidence (commands, SHAs, tables, artifacts): `tasks/debug_kernel_server_perf/resume_20261012/`
  -- `colaunch_20261012.md` (A-fix, class set, KV chunk, serving), `postfix_realistic_ab_20261012.md`
  (post-RoPE A/B, FA2 spill diagnosis), `a2_serving_ropefix_20261012.md` (RoPE root cause),
  `a3_winner_map/` (A3 tables); earlier days under `resume_20261011/`.
- Full working log to 2026-10-12, verbatim (history, retractions, bring-up):
  [`archive/debug_kernel_server_perf_4_mode_log_to_20261012.md`](archive/debug_kernel_server_perf_4_mode_log_to_20261012.md).
- Scheduling argument (PD colocation vs disaggregation, points 1-5): `tasks/pd_colocation_vs_disagg.md`.

---

## References

Papers and systems this work draws on. *(sent)* marks the ones the user sent.

- **Taming Dynamism on GPUs: Cross-SM Kernel Fusion via SM Cooperation and Just-in-Time Reduction**
  *(sent; PDF)* -- note: technique. Taken: one persistent launch whose CTAs pull heterogeneous,
  dynamically sized work from a device-side queue (our four-task plan and atomic cluster cursor,
  and cross-role backfill); just-in-time reduction of split-KV partials (built, measured at <=0.6%
  of GPU time, replaced by flashinfer 0.7's scheduled merge); its task representation as the
  template for an optional epilogue task (EAGLE tree build, item B). Not taken, by measurement:
  co-compiling roles with different register needs into one body -- our FA2 role spilled 21x
  there, so FA2 runs as a PDL-co-launched executor and backfill is done by the hardware scheduler.
- **SLOs-Serve: Optimized Serving of Multi-SLO LLMs** -- S. Chen, Z. Jia, S. Khan, A. Krishnamurthy,
  P. B. Gibbons, [arXiv:2504.08784](https://arxiv.org/abs/2504.08784) *(sent)* -- note: setting.
  Application- and stage-specific SLOs, several application mixtures (summarization, coding,
  chat, tool calling, reasoning) crossed with load, capacity measured under SLOs (goodput / SLO
  attainment): the source of S1-S3's per-tenant SLOs and mixture x rate sweeps. Its chunked-prefill
  planner that co-plans speculative decoding is the closest scheduling analogue of our 4-mode step.
- **Towards High-Goodput LLM Serving with Prefill-decode Multiplexing (MuxWise)** -- Y. Chen,
  W. Cui et al., [arXiv:2504.14489](https://arxiv.org/abs/2504.14489) *(sent; "the ASPLOS PD
  multiplexing paper")* -- note: baseline. Intra-GPU prefill/decode multiplexing with an SLO-aware
  dispatcher and a contention-tolerant estimator; goodput under SLO as the metric. Direct
  colocation baseline (not yet run); its estimator is the comparison point for our EDF step model.
- **ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System** -- H. Kang et al.,
  [arXiv:2602.13692](https://arxiv.org/abs/2602.13692) *(sent)* -- note: argument and setting. KV-cache
  thrashing in agentic serving (a session's cache evicted during a tool call and re-prefilled, up
  to 7.14x end-to-end): motivates PD point 2 (migration leaves replicas where demand is) and
  evaluating on multi-turn agent sessions rather than independent prompts.
- **TraceLab: Characterizing Coding Agent Workloads for LLM Serving** -- K. Zhu, M. Jacob, C. Ma,
  Y. Pan, S. Wang, A. Krishnamurthy, B. Kasikci, [arXiv:2606.30560](https://arxiv.org/abs/2606.30560);
  dataset `syfi_coding_trace.jsonl.gz` ([github](https://github.com/uw-syfi/TraceLab)) *(sent)* --
  note: workload. The coding-agent tenant in every mixed run (causal replay of prefix / append /
  output tokens per round); its statistics set the mixtures and the full-length replay (context
  p50 ~124K, a quarter of Claude rounds >= 256K, short outputs, high prefix hit) and PD point 5.
- **semi-PD: Towards Efficient LLM Serving via Phase-Wise Disaggregated Computation and Unified
  Storage** -- K. Hong et al., [arXiv:2504.19867](https://arxiv.org/abs/2504.19867)
  ([code](https://github.com/infinigence/Semi-PD)) -- note: closest intra-GPU design. Splits each
  GPU's SMs between a prefill and a decode process over one shared KV store, so there is no KV
  transfer and no weight replication. Difference from ours: its SM ratio is a deployment-level
  setting changed by relaunching (criticised as static in arXiv:2504.19516), while our split is
  decided per step inside one forward, co-runs prefill chunks with MTP verify rows, and lets the
  hardware backfill FA2 work onto SMs FA3 vacates.
- **NVIDIA Dynamo -- KV router, conditional disaggregation, planner, KVBM**
  ([conditional disagg](https://docs.nvidia.com/dynamo/advanced-customizations/conditional-disaggregation);
  [router](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/routing-concepts);
  [planner](https://docs.nvidia.com/dynamo/v1.5.0/knowledge-base/modular-components/planner/overview);
  [delta-KV RFC](https://github.com/ai-dynamo/dynamo/discussions/7861)). Full policy-by-policy
  comparison: `pd_colocation_vs_disagg.md` point 6 (block-count cost router with no SLO term,
  static local-prefill thresholds, minute-scale P/D autoscaling, incidental replication) -- note: closest
  multi-node baseline for agentic rounds. The KV-aware router prefills locally on the decode worker
  when the uncached part of the prompt is small (default effective-ISL threshold 4096 tokens) or
  remote prefill is loaded, so most agent append rounds avoid the KV transfer that PD point 5
  measures; the RFC would cut remote rounds to append-only transfer. What remains ours: making that
  local prefill + speculative decode co-execution efficient (4-mode step + fused kernel, S1:
  +20-25% throughput, -50-80% TTFT vs stock MTP under load), symmetric nodes instead of fixed
  prefill/decode pools (routing and replication free to use every node), and EDF/SLO-aware
  admission. Must be run as a baseline, not argued around.
