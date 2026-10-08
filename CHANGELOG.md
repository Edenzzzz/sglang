# Changelog

## v0.25.4 — Serving path unblocked; the "version skew" was PYTHONPATH shadowing (2026-10-09)

- RETRACTED the version-skew diagnosis. Both venvs already have `flashinfer-python 0.7.0.post1` with
  the symbols sglang needs; the driver's `PYTHONPATH=/dev/shm/w/python:/dev/shm/w/fi` let the older
  restored tree shadow the pip install. Dropped `fi` from `PYTHONPATH` (it is only needed as
  `FI_ROOT` for the kernel build) and reverted the symptom-treating import guard.
- The fork server with the split kernel now starts (`Application startup complete`, Uvicorn on
  :31000), so the serving A/B can run.

## v0.25.3 — Serving A/B blocked on a flashinfer/sglang version skew (2026-10-09)

- With the venv repaired, `srv_two_exec_ab.sh` runs but the server dies at import:
  `cannot import name 'trtllm_mxint4_block_scale_moe' from 'flashinfer.fused_moe'`. The fork's
  sglang tree expects a newer flashinfer than `/opt/tiger/flashinfer @ b0563ae`, which is what the
  `fi` restore pushed.
- Preferred fix is to guard that one eager import (a w4a4-mxint4 MoE scheme a bf16 Qwen2.5-7B run
  cannot need) rather than upgrade flashinfer, since the fused-kernel work is local-only.
- Lesson: a tree restored from a local copy can build correctly and still mismatch its consumer;
  verify the two trees' versions against each other, not just that both import.

## v0.25.2 — Fork serving environment fully restored (2026-10-09)

- Second dependency fault fixed: `torchvision 0.24.1+cu128` against torch 2.14.1 caused a circular
  import; installed the matching `0.29.1`. With the earlier nccl swap, the fork venv now imports
  both the fork's sglang (`/dev/shm/w/python/sglang`) and our fused flashinfer (`/dev/shm/w/fi`).
- Pod restore is a seven-step recipe, all documented. The fork serving A/B -- the only arm that
  exercises the split kernel on the realistic tracelab workload -- is now reachable.

## v0.25.1 — Fork venv repaired: nccl library mismatch, serving A/B path unblocked (2026-10-09)

- The fork venv's `import torch` failure (`undefined symbol: ncclCommResume`) was an nccl build
  conflict: both `nvidia-nccl-cu12 2.27.5` and `cu13 2.30.7` install to the same path and the cu12
  build (lacking that symbol) had won. `LD_LIBRARY_PATH` cannot fix it -- torch resolves nccl via
  RPATH -- so the bundled `libnccl.so.2` was replaced with the cu13 build (original kept as .bak).
- `torch OK 2.14.1+cu130, cuda_avail=True`. This unblocks the fork serving A/B, the only arm that
  exercises the split kernel and has the tracelab dataset.

## v0.25.0 — First validated kernel speedup: fused 1.30x over the unfused persistent kernel (2026-10-09)

- `TIME_BASELINE=1` provides a like-for-like baseline (the harness plans a separate 3-task object
  over the SAME rows without `FLASHINFER_FA3_TASK` -- the unfused persistent kernel on identical
  work). Fused at sm_split 112: **133.5 us** vs baseline **174.0 us** = **1.30x**, 23% less kernel
  time, with the kernel PASSING the causal fp32 reference at every split.
- Baseline stable at 173-177 us across three runs. Fixture is FA3-heavy, so this does not
  generalise to the mixed serving workload, and it does NOT satisfy the >=30% TTFT/ITL or >=10%
  throughput bar -- only the serving A/B can.
- The FA3-role per-executor baseline is still blocked on the wgmma `params.wgmma_plan` plumbing.

## v0.24.3 — flash-attention tree restored; FA3 baseline blocked on the wgmma params path (2026-10-09)

- Pushed `flash-attention/hopper` (249 KB) as a sibling of the flashinfer root, clearing the
  `fa3_upstream.cuh -> ../../../../../flash-attention/hopper/utils.h` build failure. The pod restore
  recipe is five steps, not four.
- `FA3_UPSTREAM=1` now fails later, on `BatchPrefillPagedParams has no member "wgmma_plan"` -- the
  upstream-FA3/wgmma route is incomplete in this build, and `fa3_only_run` is gated behind it. So the
  FA3 per-executor baseline is unbuildable here and **no speedup is demonstrated**; the default
  build still PASSES and times at 133.5 us (sm_split 112).

## v0.24.2 — Split optimum found (sm_split 112, 133.5 us); baseline arm still missing (2026-10-09)

- Extending the sweep past the harness's built-in 48/32/16 found the optimum at sm_split 112:
  133.5 us vs 383.3 us at 48 (2.87x), with a ~258 us plateau at 64/80/96 and a cliff at 112.
  Every split PASSES the reference.
- The `TWO_EXEC=1` per-executor baseline arm does not run: the extension does not export
  `fa3_only_run`. So the fused times have nothing to divide by and **no speedup is claimed**.

## v0.24.1 — First valid fused kernel timings (2026-10-09)

- With the kernel passing, `TIME_ITERS=50` on fixture 1 gives 383.3 us at sm_split 48, 505.1 us at
  32, 877.5 us at 16 (0 unwritten, every row within bf16 noise at every split; the harness prints
  `PASS: every row correct at every split`).
- Monotonic in FA3's SM share, so the optimum is likely above the harness's 48-SM ceiling. This
  fixture is FA3-heavy, so the FA3-favouring split is expected and does not generalise.
- Per-executor baselines are still needed before any of this can be called a speedup.

## v0.24.0 — FUSED FA3+FA2 KERNEL PASSES THE CAUSAL fp32 REFERENCE (2026-10-09)

- Fixture 1 is green at sm_split 48/32/16: **0 unwritten rows**, FA3 0.00628, task0 **0.00250**,
  tile16 0.00071 -- all three classes within bf16 noise of the causal fp32 reference.
- Two fixes: (1) the persistent runners' `KernelTraits` used `MaskMode::kNone`, so the FA2 side
  applied no causal mask; (2) `cluster_tile_q` used `gridDim.x` unguarded under `FUSED_GRID`, and
  since the kernel's CAUSAL `num_kv_chunks` depends on it, the inflated value forced every tile onto
  the split-KV path where causally-confined rows never completed their merge.
- One invalid refutation corrected in place: fix (2) had been tested only on a single-chunk fixture
  that could not exercise it, and "no change" was wrongly read as refutation.
- Fused timing is meaningful for the first time; the serving A/B is the next step.

## v0.23.4 — Split-KV defect attributed to the reduction, with the fix located (2026-10-09)

- `NO_JIT_REDUCE=1` A/B: unwritten rows jump 55 -> 87 (row 1's full length) when the reduction is
  disabled, so the reduction is what writes these rows and the plan is exonerated. It completes 32
  rows and skips 55 -- precisely the rows whose only partial is chunk 0's.
- Fix located: the reduction must finalise a row whose partials come from a single chunk instead of
  requiring every chunk to contribute. The separate 3-chunk weighting error (rel err 0.514) needs
  its own fix in the same reduction.

## v0.23.3 — Split-KV merge mechanism pinned on the FA2 side (2026-10-09)

- kv-length sweep at fixed qo: single-chunk rows PASS at two different qo (0.00152, 0.00269);
  every multi-chunk row FAILS. The unwritten count tracks the CAUSAL BOUND, not qo -- with a prefix
  large enough that every row needs a chunk beyond the first, nothing is unwritten.
- Mechanism: the merge produces `final_o` only for rows with partials beyond chunk 0; a row whose
  causal bound lies entirely in chunk 0 is never merged and stays NaN. At 3 chunks every row is
  written but wrong (0.514), so the same merge also has a weighting error -- two defects, one path.

## v0.23.2 — Second fused fault identified as the FA2 split-KV merge (2026-10-09)

- A store probe showed NO work item of fixture 1's failing row reaches the write-through store, and
  its trace reads `nchunks=2`: that row takes the `partial_o` + merge path, while the `qo=64`
  fixture (`nchunks=1`) takes write-through and PASSES after the MaskMode fix.
- So the 55 unwritten rows are a split-KV merge defect on the persistent side, not a q-tile
  coverage problem. The earlier "5th tile" arithmetic is retracted: the sweep's 0-to-55 transition
  tracks `kv_len` crossing `len_kv_chunk = 64`, because the fixture moves qo and kv together.

## v0.23.1 — Fused kernel PASSES the causal fp32 reference on all three classes (2026-10-09)

- Fix: the persistent runners' shared `KernelTraits` used `MaskMode::kNone`, so the FA2 side applied
  no causal mask. Changed to `MaskMode::kCausal`. task0 rel err **0.98-1.04 FAIL -> 0.00221 ok**,
  with FA3 (0.00709) and tile16 (0.00134) unchanged and 0 unwritten rows, at sm_split 48 and 32.
- First run where every class passes the reference. Found from the per-token profile already
  available behind `DUMP_ROWS=1`: token 63 of 64 was the sole correct token, i.e. every token was
  attending to the full KV range.

## v0.23.0 — Root cause of the fused task0 failure: missing causal mask in the tile128 path (2026-10-09)

- The failing task0 row has exactly ONE correct token -- the last one, which is the only q token
  that legitimately attends to all of its KV. Every other token is computed against the full KV
  range, so the per-token causal mask is not applied in the tile128 persistent path.
- This single cause explains the whole evidence set: the 3.40x magnitude deficit (dilution, not
  scaling -- got is not proportional to ref), all 28 heads damaged, the tile16 row passing (its
  single q token needs no mask), FA3 passing (separate causal path), and the tile-level kv_end
  bounds being provably correct while the in-tile mask is not.
- Ten earlier candidates were eliminated and recorded in place, including one attempted fix
  (`cluster_tile_q`, refuted and reverted) and one retracted "duplicate work item" anomaly that was
  an artifact of aggregating traces across the harness's three sm_split launches.

## v0.22.1 — task0 fused fault characterised; cluster_tile_q candidate refuted (2026-10-09)

- The fused kernel's failing task0 row writes EVERY element (229376/229376 nonzero) but at
  **3.40x too small a magnitude** (absmax 0.428 vs reference 1.453), while the tile16 row is exact
  and FA3 passes in the same launch. So the tile128 runner's epilogue is fine and its
  softmax/accumulation balance is not -- the fault is an under-weighted result, not a missing write.
- The `cluster_tile_q = gridDim.x * CTA_TILE_Q` candidate (unguarded under FUSED_GRID, unlike
  `persistent_cluster_slot`) was tested on the discriminating fixture and **refuted** (rel err
  1.03446 -> 1.04120), then reverted rather than left in the tree.
- Harness note: the store-vs-compute discriminator already existed behind `DUMP_ROWS=1`; and an
  instrumented (device-printf) build corrupts FA3 numerics (0.0059 -> 1.19), so coverage runs and
  numerics runs must stay separate.

## v0.22.0 — Fused FA3+FA2 kernel builds and runs again; FA3 half validated (2026-10-08)

- Restored the flashinfer tree on the vibecuda pod (it was absent, so the split kernel had no
  runnable binary at all) and got the fused TU compiling: five sequential blockers, the last being
  that torch 2.14 needs **C++20** in both the harness `load()` and `flashinfer/jit/core.py`.
- First kernel-arm result on fixture 1: the **FA3 executor PASSES** the causal fp32 reference at
  rel err 0.00662 with split-KV active (4 splits), and a persistent decode row passes at 0.00070.
- Open kernel bug: the persistent **task0** row (qo=87) leaves **55/512 output rows unwritten**
  (NaN), identically at sm_split 48/32/16 -- so it is a q-tile/work-item coverage gap, not an SM
  partitioning or race effect. No fused timing is valid until this is fixed.

## v0.21.0 — Dispatch boundary is a ratio, and the decode-population guard (2026-10-08)

- Re-analysed the measured 2-D boundary (`boundary_28x4x128.json`): the fused win is set by a RATIO,
  `kv <= 64 * decode_rows` (one 64-wide KV tile of decode work per decode row), not by thresholds on
  `qo_len`/`kv`. With no co-resident decode rows the fused kernel loses on every measured kv
  (0.617-0.704x), which no per-row rule can express.
- `mixed_dispatch.split_classes` now applies a batch-level decode-population guard
  (`SGLANG_FUSED_DECODE_GUARD`, `SGLANG_FUSED_DECODE_TILES_PER_ROW`). Replaying all 25 measured
  points through the real dispatch code: agreement with the measured winner rises from **11/25 to
  23/25**, and the `nd = 0` losing region is no longer routed to FA3.
- Pod restore: found TWO casualties of the earlier restore -- the flashinfer tree (`/dev/shm/w/fi`)
  was absent, so the split kernel had no runnable binary, and the fork venv's torch is broken
  (nccl ABI). Recorded a restore split that clones third-party on the pod (it has network) and
  pushes only the fork's ~2 MB of sources.

## v0.20.1 — Correctness evidence for the ported 4-mode step, and two invalid gates retired (2026-10-08)

- The merged forward's PREFILL rows are bit-exact against a prefill-only reference (err 0.0, 0
  argmax flips, all 5 mixed steps), which validates the merged batch, the logits split and the
  comparison machinery at once.
- Greedy-output equality was retired as a gate for this feature: the merge point is
  timing-dependent, so the verify rows sit at different decode positions each run (identical
  `pre_logits_sum` with `seq_lens` differing by `num_draft_tokens`). Under a FIXED composition the
  4-mode arm is byte-identical to the control.
- Gate (b), comparing the merged verify slice against an unmerged verify forward, was found to
  measure its own side effects: `eagle_prepare_for_verify` plus a verify forward is destructive and
  non-idempotent, so two identical references disagree as much as merged-vs-reference. Needs a
  non-destructive reference (run it before the merged forward).
- Port commits 39-46: bounded the verify hidden-state slice (padding rows were reaching the draft
  extend), parked the verify half's keep-alive refs, and restored the prefill rows' own
  `sampling_info`, which had been dropped in an earlier restructure.

## v0.20.0 — The 4-mode step runs end to end on current upstream (2026-10-08)

- Branch `upstream_port` (worktree `/opt/tiger/sglang-up`) against upstream pinned `b7b2975b5`,
  torch 2.14.1: `portchk.sh` reaches **11 mixed steps with 0 tracebacks**, where a
  `SGLANG_MIXED_SPEC=0` control on the same script also passes. Correctness (greedy-output
  equality) is NOT yet gated -- the smoke script's prompts are drawn inside worker threads, so its
  output lengths are not comparable between arms.
- The verify half now delegates to upstream's own `run_eagle_verify` via a new
  `precomputed_logits_output` / `precomputed_forward_output` pair, instead of hand-rolling sample +
  commit + result (which cost one failed run per field upstream sets).
- The scheduler commits BOTH halves of a 4-mode step: the gate keeps `running_batch` (emptying it
  let the prefill batch take the decode rows' place and leaked a request), the verify half gets its
  own future wiring, relay stash and D2H copy, and result processing receives a snapshot batch
  because overlap merges the next prefill into the live one first.
- Two latent bugs fixed on the way, both of which would have corrupted output silently rather than
  crashing: the dispatch branch dropped the scheduler's `on_publish`, so a 4-mode step published no
  futures at all; and the verify-row logit slice indexed from the padded end of `hidden_states`
  under a prefill-graph bucket, selecting padding rows instead of verify rows.

## v0.19.1 — The 4-mode split verified numerically; MTP results confirmed (2026-10-08)

- `SGLANG_SPLIT_VERIFY=1` fp32 recompute on a real mixed step: FA3 prefill rows 0.0032-0.0040 and
  FA2 **verify** rows 0.0030-0.0034 relative error -- bf16 noise. The 5-token greedy divergence from
  stock MTP is argmax flips from that noise, not a fault, so the MTP serving numbers stand.
- Min-common-prefix is a weak cross-kernel discriminator (it only catches gross faults, which show
  rel err ~4.0); the fp32 per-row check is the real gate. New driver `verify_gate.sh`.
- The H20 gate's earlier one-mixed-step pass was the GATE's fault: 96-token background rows
  finished before the long prefills arrived. Fixed -> 10 mixed steps, one with 2 prefill + 11
  verify rows.

## v0.19.0 — Randomized source selection refuted under contention (2026-10-08)

- Point 4's proposed algorithm ("pull at random among nodes whose hit lengths are within a
  threshold") LOSES on every metric against ordering by in-flight load, even on a deliberately
  narrowed 1-NIC fabric built to favour it: chat mean ITL +24%, chat P99 ITL +63%, agent P99 ITL
  +83%, chat TTFT +15%. Load, not link bandwidth, is the binding constraint at 16 GPUs.
- Narrowing the fabric DOES create real contention (tails 2.3-3.2x worse at unchanged means), so
  the test was valid; randomization simply discards the load signal.
- New router policy `cache_slo_rand` exists for this test and should NOT be a default.
- Two earlier experiment designs were discarded before use and are recorded: a partially-pinned
  fabric (sequential push raced the arm launch) and an asymmetric pair (worker EDF on one side).

## v0.18.1 — Long context at a 20 ms chunk budget clears 30% on every metric (2026-10-08)

- 3 rounds, 64-96K TraceLab + chat 12/s, vs stock MTP at its own best chunk: throughput +45.0% /
  +43.3%, TTFT -85.3% / -39.6%, mean ITL -53.2% / -52.4%, **P99 ITL -32.9% / -48.0%** (chat /
  codex). The earlier -29.7% chat P99 miss at a 47 ms budget was budget tuning, not a ceiling.
- Chat P99 ITL saturates near 112 ms for 20-30 ms budgets; codex keeps improving to 78 ms.

## v0.18.0 — The SLO veto prices decode, not just prefill (2026-10-08)

- Hot-prefix workload (4 groups sharing 16K prompts, H20 16 GPUs): a TTFT-only veto never fired on
  an overloaded cache holder, because its prefill stays fast. The veto now prices the decode
  slowdown a request would cause: chat mean ITL **-65%** (23.4 -> 8.3 ms), P99 **-80%** (224 -> 44 ms)
  for a 25% chat TTFT cost. Colocation keeps a ~70x TTFT advantage over disagg 2P2D.
- `calibrate_h20.py` measures decode ITL over context x concurrency (was 256-token context only,
  fitting an unusably small 0.065 ms/seq). The router interpolates that grid (`--decode-grid`); two
  closed forms were rejected and the rejections are recorded in the code.
- RESOLVED (2026-10-08): the 15x ITL jump by 524K resident tokens is prefill interference in the
  calibration (all requests submitted at once, so 16K prefills share steps with decode), not a
  capacity or bandwidth effect: a server held at 32x16K logs pure-decode steps at ~18.6 ms, 7x
  faster, with no retraction. The grid stays valid for routing as a loaded-worker cost, but it is
  not a decode-only model.

## v0.17.6 — Prefix-aware chunk sizing (2026-10-08)

- `SGLANG_CHUNK_ATTN_BUDGET` caps a chunk's attention work n*(r + n/2) while rows are running. At
  64-96K it fixes codex's tail (P99 ITL 91 ms, the best of any setting) at little TTFT cost, and it
  is neutral at short context. It does not fix chat's tail: fresh-prompt chunks are GEMM-bound.
- `SGLANG_CHUNK_MS_BUDGET` + `SGLANG_CHUNK_COST` cap the chunk's predicted time with the fitted step
  coefficients. At 64-96K, 3 rounds, vs stock MTP's best: throughput +42/+37%, TTFT -81/-35%,
  mean ITL -49/-45%, P99 ITL -29.7/-30.2% (chat/codex). It regresses at short context (chat SLO
  0.641 vs ~0.735), so it is a long-context setting only.

## v0.17.5 — Long context, both arms tuned: full stack wins on every metric (2026-10-08)

- Chunk 2 048, 3 rounds, 64-96K TraceLab + chat 12/s. Against stock MTP at its own best chunk
  (8 192): throughput +45% / +45%, TTFT -93% / -55%, mean ITL -48% / -45%, P99 ITL -15% / -31%
  (chat / codex). The chunk-8 192 P99 regression is fixed by the chunk size; stock MTP does not
  benefit from the smaller chunk.
- Chat 8/s at the default chunk, 3 rounds: throughput +18% / +30%, TTFT -96% / -75%, mean ITL
  -58% / -49%; chat P99 ITL +27% (the chunk effect again).
- H20: the colocated-MTP "regression" is retracted. The router's `min()` tie-break sent every
  hit-0 request to workers 0/1 and affinity pinned sessions there. The tie-break is now
  randomized. With even load, colocation + MTP + 4-mode keeps TTFT and cuts agent ITL 47%.

## v0.17.4 — Long-context (64-96K) A/B: throughput +46/+53%, P99 ITL regresses (2026-10-08)

- Full stack vs stock MTP, TraceLab sessions at 64-96K + chat 12/s, 3 rounds: output throughput
  +45.6% (chat) / +52.6% (codex), TTFT -96.7% / -71.2%, mean ITL -46.9% / -47.2%. P99 ITL is
  worse: +54.6% chat, +5.8% codex. Chunk sweep and below-saturation rounds are queued.
- `long_agent_chat` scenario; `mtp_edf.sh` honours `TRACE_PATH` and `SCENARIO`.

## v0.17.3 — Attention share of a decode step vs context length (2026-10-08)

- H200, Qwen2.5-7B, yarn to 131K: attention exceeds half of a decode step at ~40K tokens with 8
  requests (72.6% at 120K) and ~10K with 32 (73.7% at 32K). A single request never does (31.9% at
  120K). The codex tenant (8-12 sessions at 32-49K) is ~47-55%. Drivers `context_share.sh`
  (bench_one_batch, batch 1) and `context_share_srv.sh` (server, batches 8/32: bench_one_batch's
  one-shot prefill OOMs there).

## v0.17.2 — Full stack vs stock MTP: every TTFT/ITL metric >=30% better, 3 rounds (2026-10-08)

- 4-mode split + deadline order vs stock sglang MTP at chat 24/s + codex: chat TTFT -95%,
  chat ITL -42% (P99 -35%), codex TTFT -51%, codex ITL -38% (P99 -34%). Chat SLO attainment and
  goodput 3.4x. Output throughput is equal, since both arms serve the offered rate; the capacity
  gain is the v0.16.0 closed-loop +19.8%. Commands and SHAs are in pd_colocation_vs_disagg.md
  point 3.

## v0.17.1 — EDF on the MTP server: order by deadline, do not cap chunks (2026-10-07)

- Ablations at ITL SLO 30 ms (H200, chat 24/s; `mtp_edf.sh` with `ARMS=`). Every per-step
  chunk-cap variant loses to no EDF: global EWMA, FCFS + cap, and the P90 slow-step allowance
  (`SGLANG_EDF_ITL_QUANTILE`). Splitting prefill adds fixed-cost mixed steps that every decode row
  waits through. Per-request EWMA collapses because the binding request's estimate is ~1.15 vs a
  global ~1.6-1.9 (`EDF_BIND`).
- **Deadline ordering alone (EDFP), 3 rounds:** chat TTFT -62%, codex P99 ITL -27%, chat P99 ITL
  -10%, chat SLO attainment and goodput +8.6% (ranges disjoint); codex mean TTFT +43%, inside its
  length-scaled SLO.

## v0.17.0 — EDF on the MTP server: acceptance length in the ITL budget (2026-10-07)

- `edf_scheduler.py`: the per-step ITL budget is `ITL_SLO * tokens/step`, with three estimators
  of tokens/step (`SGLANG_EDF_ACCEPT_MODE`): `none` (a=1), `global`, or per-request `req` EWMA
  (`SGLANG_EDF_ACCEPT_ALPHA`). The estimators are scored online against delivered tokens
  (`EDF_ACCEPT` log; `SGLANG_EDF_ACCEPT_TRACE=1` scores them without EDF). Also added: an
  `EDF_BIND` log and a `linear:<json>` step model (`LinearStepPredictor`).
- STEPTRACE counts a 4-mode step's verify rows as decode rows. New:
  `tasks/perf_model/fit_mtp_step.py` (held-out step MAPE 21.9% vs 98.6% for a constant), and
  drivers `mtp_edf.sh` and `slo_attain.py` (with `--scaled`, the length-scaled TTFT SLO EDF
  schedules against).
- Measured (H200, chat 24/s, ITL SLO 50 ms): mean tokens/step is 2.10. EDF with a=1 collapses
  (chat SLO attainment 0.117, TTFT 41 s); the global EWMA restores it to 0.866. That still does
  not beat no-EDF (0.876), because this SLO barely binds. The per-request EWMA has a lower
  estimator error but worse attainment (0.696).

## v0.16.1 — multi-node runs: 1024-fd limit was dropping requests (2026-10-07)

- **Retraction:** every earlier H20 multi-node arm ran with the pods' soft `ulimit -n` of 1024.
  Under the 3200-request chat burst the router ran out of sockets ("error sending request").
  For the stock sglang_router this then opened its circuit breaker, which rejected ~5.7k
  requests ("No available prefill"). disagg_2p2d served only 163/384 agent and 2265/3200 chat
  requests. Those numbers, and the coloc arm numbers from the same harness, are void pending
  re-runs.
- `mn_run.py`, `mn_bench.sh` and `mn_node.sh` raise `ulimit -n` to 1048576. Stock-router arms
  run with the circuit breaker and health ejection off, and the PD arms use
  `--prefill-policy cache_aware --decode-policy power_of_two`.

## v0.16.0 — 4-mode MTP clears the bar: latency under load, ~20% capacity (2026-10-07)

- **MTP load sweep** (chat 4-28/s, both tenants rate-limited, barrier, 0 fallbacks): the
  4-mode + split gain grows with load. At 20/s all four headline latency metrics beat stock MTP
  by >=30% on both tenants (codex TTFT -48%, ITL -33%; chat TTFT -87%, ITL -39%). Mixing without
  the split regresses at every load.
- **MTP capacity, closed loop** (3 rounds): total output throughput **+19.8%** (chat +20.5%,
  codex +16.9%, non-overlapping ranges); codex ITL -30%. Codex TTFT is +21% under saturation
  (queue position), reported as measured.
- H20 multi-node study: cache-affinity routing beats round-robin by 8.4x on agent TTFT and
  ~90x on chat TTFT (72.9% worker-reported cache hit). sglang's own `cache_aware` router is a
  strong baseline: better chat TTFT tails, worse ITL tails than ours. The SLO veto needs
  calibrated H20 constants (0 vetoes with placeholders).

## v0.15.1 — Stream start barrier; serving results re-measured (2026-10-07)

- **Methodology fix:** multi-stream serving runs had an uncontrolled stream start offset (12-45 s,
  set by how long each `bench_serving` took to tokenise). It swung chat TTFT 15x between
  sessions. Fixed with a start barrier (`d2fc14f8c`). All serving A/Bs re-run with it.
- **SPLIT vs LPT: confirmed** (3 rounds, tight ranges): chat TTFT -17.4%, codex TTFT -10.5%,
  ITL -7.9%, P99 ITL -15.9%, throughput +7.0%. Essentially the original result.
- **4-mode MTP: CORRECTED.** v0.15.0's figures were offset-confounded. Barrier re-run, 4-mode +
  split vs stock MTP: codex throughput **+17.6%** (not +27.8%), codex P99 ITL **-52.8%**, ITL
  -16.2%, TTFT -8.5%; chat TTFT **-37.3%**, P99 ITL -30.5%, ITL -27.7%.
- Kernel catalogue closed: chunk sweep, host-overhead fixes (~1.6 ms per mixed step, stopped at a
  ~2% bound), the plan cache keyed on exact row lengths.
- H20 multi-node study started: 8 pods x 2 H20 (16 GPUs, trial vibecuda); ablatable router;
  per-pod port probing with bind-race supervisor; session-replay workload.

## v0.15.0 — 4-mode MTP serving result; kernel catalogue closed on measurement (2026-10-07)

- **4-mode MTP serving A/B** (3 interleaved rounds, gate PASS, 0 fallbacks; `drivers/mtp_repeats.sh`):
  4-mode step + split vs stock sglang MTP -- codex throughput +27.8%, ITL -17.4%, P99 TTFT -14.6%;
  chat ITL -35.0%, chat TTFT -50% or better (stock MTP overloaded in one round). Mixing alone
  gives most of the chat gain; the split adds codex throughput.
- 4-mode MTP gate PASSES (stock, mixed-FA2, mixed-split); fork bugs fixed (`10b378d89`, `5ab463414`).
- Fused tree build wired into the EAGLE worker (`f1985ca8f`, `SGLANG_FUSED_TREE_BUILD=1`), gate PASS.
- Kernel catalogue: SM/cluster-cap sweep (serving's cap already optimal; FA3 budget must stay full),
  CUDA-graph ceiling via step traces (split host gap +0.83 ms/mixed step -> FA3 plan skipped in
  two-executor mode, `e237ab8bc`), packed copies superseded, stale boxes closed.
- Perf model: split mixed steps modestly easier to predict (MLP 6.0% vs 6.4% MAPE; linear 13.4% vs
  16.9%).

## v0.14.0 — SPLIT serving made correct; first valid serving result; 4-mode MTP step (2026-10-07)

- **Three correctness faults in the serving split, fixed:** (1) FA3 was handed the FA2 rows'
  per-request arrays (`748f44553`); (2) both executors ran non-causal, and so did the harness
  reference (flashinfer `b0563ae`, sglang `2b92b58bb`); (3) in ragged mode the split read the
  step's not-yet-written K/V. Split steps now use non-ragged mode (`4e3bde083`). The earlier
  -40% TTFT result is **void**.
- **Serving correctness gate** (`drivers/srv_correctness.py`): greedy outputs against LPT, with
  LPT-vs-LPT as the noise floor. It catches the pre-fix build and PASSES the fixed one. Also
  added: `SGLANG_SPLIT_VERIFY=1`, an in-server fp32 recompute per executor.
- **Causal kernel:** split 1.309x over FA2-on-everything on the TraceLab batch, 1.324x with
  two-stream overlap (now the serving default). Multicast ceiling 1.8% / 1.6% causal.
- **First valid serving A/B** (3 interleaved rounds, 0 fallbacks): codex TTFT -10.3%, ITL
  -7.6%, P99 ITL -18.5%, throughput +5.6%; chat TTFT -16.2%. Below the 30% / 10% bar.
- **4-mode MTP step** (`868025d6c`, `SGLANG_MIXED_SPEC=1`): prefill chunk rows and EAGLE
  verify rows in ONE target forward. Untested; gate queued. Draft:
  thoughtworks/Qwen2.5-7B-Instruct-Eagle3, topk 1.

## v0.13.0 — Kernel items closed: FA2 K/V ring, split-reduction ceiling, fused tree build (2026-10-07)

- **FA2 K/V smem ring** (flashinfer `3e41c2d`, defaults `f54745e`): an S-slot ring per tile
  class (`FLASHINFER_FA2_KV_STAGES_{1,2,3}`), now 2/3/1 by default. MTP-verify rows -7.4%,
  64-128K verify -10.3%, appends -2.0%, decode flat (already ~80% of HBM). 3 reps, all PASS,
  0 fallbacks. `drivers/fa2_ab.sh` + `tests/fa2_stage_bench.py`.
- **Smem-handoff split reduction: ceiling measured, not built.** Skipping every rendezvous and
  merge saves <=2.9%; splitting itself is worth 2-4x. Classic grid.sync beats our JIT
  rendezvous by 2.2% on long verify.
- **Fused draft tree build** (`speculative/csrc/fused_tree_build.cu`): organize_draft_results +
  build_tree in one kernel, 0 mismatches vs sglang over ~150 cases; 31 -> 10 us graph-captured
  (MTP chain). The doc's "attention epilogue" design is retracted: the data dependencies forbid
  it.
- Tooling: `FLASHINFER_EXTRA_CUDAFLAGS` (per-arm -D for flashinfer's JIT), harness `EXTRA_DEFS`,
  `FLASHINFER_SPLIT_MERGE_SKIP` and `FLASHINFER_FA3_SKIP_KV_LOAD` timing-bound switches
  (default off; the latter is a patch to the pinned flash-attention, in
  `drivers/flash_attention_skip_kv_load.patch`).
- Serving repeats (realistic_repeats.sh) stopped mid-round-1 to close kernel items first.

## v0.12.0 — First realistic serving comparison: SPLIT vs LPT, preliminary (2026-10-07)

TraceLab coding-agent replay plus 12/s short chat on 1x H200, both arms 0 fallbacks, single
run each:

- codex: mean TTFT -39.8%, P99 TTFT -32.9%, mean ITL -16.8%, P99 ITL -52.6%, throughput +16.9%
- chat: mean TTFT -41.5%, P99 TTFT -29.7%, ITL flat, throughput flat (open-loop fixed rate)

Contradicts my earlier "low single-digit gain" prediction, which is retracted. Likely
mechanism is queueing amplification near saturation (unverified). Preliminary: no repeats,
an intermittent illegal memory access in the SPLIT arm (1 of 2 async runs), no MTP.

Getting here took three serving-path fixes that no kernel-level test could see: the
two-exec build dropped -DFLASHINFER_FUSED_UNIFIED_WGMMA=1 (one env var gated a runtime and a
compile decision), sglang's fa3_only_run call lacked the num_splits argument, and the pod's
dependency chain needed eight fixes. Also: priority list rewritten as a dated checklist, and
ab_parallel.sh builds A/B arms concurrently.

## v0.11.0 — The "split-row corruption" was a test-harness bug; per-executor items measured; the 30% ceiling (2026-10-07)

**The kernel was correct.** The long-running "split-row corruption" was two bugs in the test
harness, both introduced with its `PAGE_SIZE` support: the reference `exact()` read KV
token-contiguously (`kvb += kl`) while the paged layout starts each request on a page
boundary, and `NPAGES` was `ceil(sum/PS)` instead of `sum(ceil/PS)`. A page-ramp probe
(`KV_ROWCONST=2`, V[page p] = p x 1e-4) exposed it: the kernel's decoded mean page advances
~33 per row (correct for 4097 tokens), the reference ~32. With both fixed (flashinfer
`333902e`), every fixture on record passes -- decode N=18 at kv=4096 and 65536, the two-row
reproducer at page 128 and page 1, `qo:127`, and both TraceLab arms.

This retracts a long chain of kernel-side diagnoses (row order, partial pages, chunk limits,
partial-tile overrun, partial_lse indexing, races, the partial_o cursor, bf16 partials, a
"half page offset"), several of which were reported as confirmed. The last of those rested on
a number back-computed from a rel err rather than decoded from output. The serving A/B is
unblocked.

**Per-executor optimisation items, measured on the fixed kernel** (each executor against its
own baseline; all PASS, 0 unwritten, 0 fallbacks):

| FA3 vs FA3 | | FA2 vs FA2 | |
|---|---|---|---|
| `MmaPV_is_RS=0` | +3.99% slower | `MMA_KV=3` | -0.43% |
| `stages=3` | +0.54% slower | `MMA_KV=4` | -0.46% |
| `LPT=1` | +0.68% slower | | |
| TMA paged | +10.03% slower | | |

Upstream's defaults win every FA3 comparison; `FA2_MMA_KV` is register tiling, not pipeline
depth (SharedStorage identical at 2/3/4), so FA2 pipelining remains untested.

**Routing is already optimal both ways:** FA3 is 1.38x faster on identical append rows, FA2
1.30x faster on identical decode rows, and the split routes exactly that way. On the mixed
TraceLab batch the split cuts attention 19.3% (15 395.8 -> 12 425.4 us, 1.239x).

**That 19.3% is a ceiling on TTFT/ITL from attention**: even at 100% attention share, e2e
gain cannot exceed it. >=10% throughput is reachable; >=30% TTFT/ITL has to come from
scheduling (EDF admission, cache-hit routing, spec acceptance), not from this kernel.

**Multicast:** cluster launch implemented (`FLASHINFER_FA3_CLUSTER_M`,
`launch_kernel_on_cluster`), but upstream asserts `Use_TMA_KV || ClusterShape == 1`, so it
needs either TMA KV (measured 10% slower) or a cluster-aware cp.async gather of our own.

Also retracted this release: "42.8x" as a performance gain (it is the size of a defect in our
own port), and the claim that KV multicast requires TMA (DSMEM peer reads do not; upstream's
implementation couples them).

## v0.10.2 — The split-row corruption is a RACE in our JIT reduction; nineteen candidates eliminated (2026-10-06)

**The split now beats FA2-on-everything, reproduced three times.** TraceLab at page 128:
12 316.3 us against 15 360.2 us = **1.247x**, after 12 399.7 / 15 190.9 = 1.225x earlier on
the same fixture. Kernel-batch time, **not serving** -- the serving A/B has still never run.

**The remaining blocker is a race, not an indexing bug.** Eight consecutive runs of the
two-row reproducer, nothing changed between them:

    rel err  0.77292  0.65776  0.72528  0.68167  0.68027  0.68413  0.67000  0.68319

Nondeterministic. This **retracts** the previous entry's assertion that it was deterministic
(stated without testing), and reclassifies two earlier readings: the error magnitude varying
across fixtures (0.39-0.84) is run-to-run variance, not a function of overrun size, so the
apparent correlation with tile partiality was noise.

The suspect is the port's own **JIT-reduction rendezvous** (`group_counter`, the
`__threadfence()` / `persistent_block_sync` ordering, the counter zero-reset), whose comments
already record *intermittent* wrong merges -- 3 runs in 10 -- on exactly "the one batch shape
whose FA2-class rows split their KV". It now fails on nearly every run, which is why it read as
deterministic and sent the investigation after indexing for a dozen candidates.

**What the reproducer is:** `qo:1,512 prefix:65536,0` fails; `qo:512,1 prefix:0,65536` passes.
A split row with a **lower request index** corrupts every later `task0` row. Damage is total
(512/512 tokens, 28/28 heads), fully written and in range.

**Eliminated, with the measurement that did it:**

| candidate | eliminated by |
|---|---|
| partial final KV page | fails at `page_size=1` too |
| the chunk limit being 1024 | plan dump: `len_kv_chunk` is 512, per-task |
| follower must be unsplit | plan dump: the follower splits in both cases |
| the combine pass | `n_fa3=0` arm: no FA3, no combine, still fails |
| `partial_indptr` for unsplit items | `PI_ZERO=1` changes nothing |
| `partial_lse` row- vs tile-relative | `LSE_TILE_REL=1` changes nothing |
| wrong LSE weights | victim with `kv=512` never merges, still fails |
| write collisions / `merge_o_indices` overshoot | damage is total, not a bounded prefix |
| the victim's KV base | plan dump: `kv_indptr=513` is correct |
| per-task cluster-slot/cursor | the PASS ordering has identical cluster consumption |
| every plan array | dumped twice on different fixture pairs, correct both times |
| upstream flashinfer | pure `BatchAttention` handles the failing order at 0.00587 |

**Priority item 1 reopened then demoted to not started.** `FA2_MMA_KV` does not change
`PKT<128,N>::SharedStorage` -- 133 120 bytes at N = 2, 3 and 4 -- so it sizes the register-side
`s_frag` accumulator, not pipeline depth. The flat 15 218.8 / 15 296.6 / 15 219.5 us sweep
therefore measured a register-tiling knob and said nothing about K/V pipelining. Also retracts
the earlier explanation that FA2 was "at roofline": at the real 989.4 TFLOP/s peak it is at
**33%**, not 83%.

**The four-point PD colocation doc is written** (`tasks/pd_colocation_vs_disagg.md`), covering
cache-hit-first routing with an EDF/SLO veto, best-effort cache migration and the replication
argument, an EDF predictor with EWMA acceptance length, and complete-vs-bipartite RDMA with
random-among-ties source selection. Every claim is labelled measured or predicted and each
point carries the measurement that would falsify it. None of the four is measured yet.

**Still no serving data** against the >=30% TTFT/ITL and >=10% throughput bar. Multicast,
cross-role backfill, the 4-mode MTP run and the EDF comparison are unstarted.

## v0.10.3 — Framing correction: the 42.8x was our bug's size, not a gain (2026-10-07)

Upstream FA3 was never 42x slower than FA2; that is not a plausible figure for a tuned Hopper
kernel. The 476 ms came entirely from our adapter -- `MakeUpstreamOwnSchedArgs` passing
`seqused_k` where the scheduler wants the query length, inflating m-blocks up to 49x and making
the cost `Theta(kv^2)`. So **"FA3's half 42.8x faster" measures the defect, not an
optimisation**, and v0.10.1/v0.10.2 quote it as a headline result where they should not.

Treating 476 ms as a believable FA3 number is also what cost six hypotheses: traits, pipeline
depth, gather granularity and SM budget are all constant factors and none can produce a 49x
work inflation. The exponent should have been measured first, and an implausible baseline
should have been read as a port bug immediately.

Post-fix sanity check supports that: FA3 now reaches 443 TFLOP/s (45% of H200's 989 peak)
against FA2's 325 (33%), which is the expected ordering for a wgmma kernel against an
Ampere-style `mma.sync` one.

**The only defensible performance claim is split vs FA2-on-everything, 1.23-1.25x** (kernel
batch, not serving). There is still no baseline-FA3 arm; the three-arm driver
(`drivers/ab_three_arm.sh`, sglang `1d8897736`) exists to supply it, and it is also the test
for whether further adapter defects remain -- a 49x inflation went undetected because every
row stayed bit-correct, the surplus m-blocks landing on masked q rows.

## v0.10.1 — ROOT CAUSE: the scheduler was handed `seqused_k` where it needs the query length; the split now beats FA2 (2026-10-06)

**FA3's long-context slowness was `Theta(kv^2)`, from one wrong argument.**
`MakeUpstreamOwnSchedArgs` passed `p.seqused_k` into `TileSchedulerArguments::seqused`,
which is the **query** length per request: the scheduler computes
`num_m_blocks = ceil_div(seqused[b] * qhead_per_khead, kBlockM)` from it
(`tile_scheduler.hpp:599-621`, `Prepared=false`). So m-blocks were sized from KV --
112 correct at q=2048 against **5 488** emitted at kv=100 352, a 49x bloat -- and
multiplied by an n-block count itself proportional to KV, the work became quadratic.

Fixed by passing `nullptr`: the scheduler then differences `cu_seqlens`, and `cu_seqlens_q`
is already supplied per-request, so it yields the exact query length with no new buffer.

**Verified.** Same fixtures before and after, every row PASS:

| kv | before | after | speedup | predicted |
|---|---|---|---|---|
| 10 240 | 3 268.1 us | **792.4 us** | 4.1x | 5.0x |
| 18 432 | 10 583.7 us | **1 396.0 us** | 7.6x | 9.0x |
| 34 816 | 37 811.2 us | **2 610.8 us** | 14.5x | 17.0x |
| 67 584 | 142 434.4 us | **5 023.3 us** | 28.4x | 33.0x |
| 100 352 | 320 401.6 us | **7 425.4 us** | 43.1x | 49.0x |

`us/kv` constant to 4.5% (was a 197% spread); `us/kv^2` now varies 218% (was 2.1%). Linear,
as it must be. Speedups track predicted bloat at ~87% across a 10x range.

**TraceLab batch: the split finally wins.** FA2-on-everything 15 190.9 us against the split's
**12 399.7 us** -- **1.225x, 18.4% lower** -- with the FA3 half going 476 684 -> 11 127 us
(**42.8x**). This retires the v0.9.9/v0.10.0 finding that FA3 is 31x slower and the split
loses on long context.

**Why it hid behind every gate.** The surplus m-blocks address q rows past each request's end,
which the epilogue masks, so output was always bit-correct and only the clock was wrong. A
numerics-only validity gate cannot see this, and that was the gate in use.

**Why six hypotheses each returned ~2%.** `MmaPV_is_RS`, `LPT`, pipeline depth, cp.async vs
TMA, page 1 vs 128, SM budget -- all constant-factor knobs, and **none can change an
exponent**. Each was correctly measured and uninformative. Procedural lesson: measure the
exponent before hunting constants; a five-point KV sweep costs minutes.

**RETRACTED: the 400 TFLOP/s peak behind every roofline figure in the docs.** H200 SXM bf16
dense peak is **989.4 TFLOP/s**. Recomputed, FA2 achieves 325 TFLOP/s = **33% of peak**, not
the "1.2x off roofline, essentially at peak" claimed in v0.10.0; fixed FA3 achieves
443 TFLOP/s = 45%, above FA2. The earlier "39x off roofline" was ~95x.

Consequence: **priority item 1 is reopened on its explanation.** The flat FA2 K/V-pipelining
measurement (depths 2/3/4 within 0.5%) stands, but it was explained by FA2 being at roofline,
which is false at 33% of peak. The flatness is unexplained.

This is the fifth bug of one class in this port -- per-request vs per-item, or q vs k, given
to a parameter wanting the other. The other four were found by which rows were wrong; this one
produced no wrong rows and was reachable only through a scaling measurement.

flashinfer `e4dfa82`, sglang `7749120b2`.
Command: `/dev/shm/w/verify.sh` on vibecuda trial `444d5bc154bd9050`.

## v0.10.0 — The FA3 trait audit is worth 2.4%, not 31x; FA2 pipelining is flat; two gates were never gating (2026-10-06)

Ten-arm A/B on a rebuilt pod (trial `444d5bc154bd9050`), `drivers/ab_fa3_traits.sh`,
TraceLab batch (1.58M KV, page 128, 5 iters). **Both reproduction gates pass**: the
FA2-all control gives 15 197 us against the old pod's 15 301, and `rs0` (the old
configuration) gives 476 684 us against 480 459 -- so the rebuilt pod is comparable and
every delta below is attributable.

**RETRACTED: `MmaPV_is_RS` is not the cause of FA3's 31x.** It is worth 2.3%.

| arm | fa3 half | vs `rs0` |
|---|---|---|
| `rs0` RS=0 stages2 LPT=1 (old) | 476 684 us | - |
| `rs1` RS=1 stages2 LPT=0 | 465 577 us | -2.3% |
| `rs1s3` RS=1 stages3 LPT=0 | 472 889 us | -0.8% |
| `rs1l1` RS=1 stages2 LPT=1 | 472 865 us | -0.8% |

All three deviations the upstream audit found were real and together are worth **2.4%**.
Kept: `rs1` and `LPT=0` (free, and `rs1` frees 32 KB). Dropped: stages=3, which is 1.6%
*slower* despite fitting the 232 448 B limit exactly.

How the claim went wrong: from perfect SM scaling plus low utilisation I inferred a
latency-bound mainloop, found a genuine trait deviation that could serialise softmax
against the PV mma, and treated it as the explanation. The smem measurement that followed
(32 768 B freed = exactly one P tile) confirmed the trait changes *storage*, and I read it
as confirming the trait explains the *time*. Independent claims; only the second was the
root-cause claim. The same error repeated with stages=3 -- "fits to the byte" was elegant
and predicted nothing.

**The 31x is unexplained, and now tightly bounded.** Ruled out: the gather (TMA slower),
pipeline depth on both executors, parallelism (linear to 132 SMs), per-tile work (identical
FLOPs), and every upstream trait divergence. What remains is per-iteration: ~589k
tile-block steps, ~4 464 per SM, so 476 ms is **~106 us per 128x128x128 step against ~2.8 us
expected**, invariant across nine arms with a stable control. That is a profiler question,
not a fifth trait hypothesis.

**Priority item 1 closed, no change: FA2 K/V pipelining is flat.** `FA2_MMA_KV` 2/3/4 give
15 218.8 / 15 296.6 / 15 219.5 us -- within 0.5%. FA2 runs this batch at 322 TFLOP/s, 1.2x
off peak, so there is no latency to hide. This supersedes the standing "worth ~15%, blocked
by an smem byte count" estimate, which came from the fused kernel where FA2's slot competed
with FA3 for smem; the split removed the constraint and nobody rechecked the premise.

**RETRACTED: the "all rows PASS" validity gate was output truncation.** The page-128 arms'
recorded "rel err 2.4e-4 to 4.2e-3, all rows correct" came from a driver piping through
`| head -12`, which on a 22-row fixture shows only rows 0-5 -- exactly that range. Rows 6-21
and the `FAILURES:` summary were never printed. With full logs kept, **row 21 fails in every
arm** (control 0.0262, `rs0` 0.0296, `rs1` 0.0344, `rs1s3` 0.0308) against `TOL=0.02`, and
the error tracks **row index, not KV length** -- rows 7/8/10/13/16/17/20 are all
`kv=65537, qo=1` yet give 0.0088 -> 0.0177. This is the unmodified persistent LPT kernel, so
it predates every change here, and it is unexplained.

**The 16384-KV prediction arms are invalid, so that prediction is untested.** Both ran at
`num_splits=29` and returned garbage (row 3 rel err 1.33), because `partial_o` is sized with
no `num_splits` factor -- the latent overflow the doc had already identified. Fixing that
sizing is a prerequisite for any fixture where `num_splits > 1`.

Five pod-restore defects were fixed and committed as drivers rather than left in /tmp:
dangling absolute `flashinfer/data` symlinks, the empty JIT generated-sources cache,
`python -I` ignoring PYTHONPATH, `provision_trees.sh` having never worked, and `px.py`
itself having survived three trial expiries only in /tmp.

## v0.9.9 — FA3's long-context 31x traced to one wrong template argument; two retractions (2026-10-06)

**Root cause of the 31x: `MmaPV_is_RS=false`.** Upstream's own tuner for head_dim 128 /
bf16 returns `{kBlockM=128, kBlockN=128, MmaPV_is_RS=true, IntraWGOverlap=true}`
(`tile_size.h:33`). We matched three of the four and passed `MmaPV_is_RS=false`, set as a
guess in the first port commit and never measured. With it false the softmax result P
round-trips through smem with a barrier per KV block rather than staying in registers for
the PV mma, serialising softmax against the mma. Now defaulted to upstream's value, with
the old behaviour kept only as an A/B switch (`MMA_PV_RS`).

The diagnosis came from an `FA3_SM_BUDGET` sweep on the TraceLab batch:

| SM budget | 16 | 32 | 64 | 96 | 132 |
|---|---|---|---|---|---|
| fa3 half (us) | 3 826 892 | 1 918 061 | 950 664 | 634 346 | **480 467** |
| budget x time | 61 230 | 61 378 | 60 843 | 60 897 | 63 422 |

Constant within 4% over an 8x range, so throughput is perfectly linear in SM count to 132
with nothing saturated -- at 132 SMs FA3 hits 10.3 TFLOP/s and 80 GB/s against FA2's
322 TFLOP/s and 2 518 GB/s on the identical 4.93 TFLOP / 39 GB. That rules out both
standing hypotheses (under-parallelised; more work per tile) and leaves a latency-bound
mainloop, which is what the wrong trait produces.

`FLASHINFER_FA3_UPSTREAM_STAGES` also goes 1 -> 2, upstream's unconditional Sm90 value.
The 1-stage default was there to fit beside the FA2 slot in the fused kernel, which is
closed. 3 stages does not fit at all -- it fails
`cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` against the 232 448 B limit -- so depth
was never the lever.

**Retraction 1: the 31x is not the gather, and not `page_size = 1`.** v0.9.8's entry
claimed upstream's non-TMA paged gather cost one index lookup per KV token and predicted
`PagedKVNonTMA=false` at `page_size=128` would fix it. Measured, all rows PASS, 0
fallbacks: FA2-all baseline 15 301 us; split cp.async 480 459 us; split **TMA** 496 978 us.
TMA paged is *slower*, and coarser paging does nothing for cp.async either (480 ms at page
128 vs 477 ms at page 1). Coarser paging costs the FA2 baseline nothing, so the fixture is
not at fault. The multicast item's TMA prerequisite is now met in passing, but it must pay
back TMA's own 16 ms deficit before it can show a win.

**Retraction 2: the roofline figures were wrong.** "155x off on 1.23 TFLOP" omitted the
QO-head factor -- all 28 heads attend the whole KV, so the two append rows are 4.93 TFLOP,
12.3 ms at 400 TFLOP/s. Corrected: **FA2 at 15.3 ms is 1.2x off roofline**, not 5x, and
FA3 is 39x off, not 155x. The material part is FA2: it is already at the roofline on this
workload, so there is no FA2-side headroom to chase and any win must come from FA3's own
39x.

## v0.9.8 — Port's row-ownership faults all closed; a self-inflicted baseline regression caught (2026-10-06)

**The upstream FA3 port no longer loses or overwrites rows.** Three faults, all the same
shape -- upstream carries a per-item or per-subset field and was handed the whole-batch
aggregate:

1. per-item `num_splits` in `split_idx`'s top 16 bits;
2. `cu_seqlens_q` / `seqused_k` read from the plan's per-WORK-ITEM arrays instead of
   per-request ones (fixing this took fixture 2 from 439/512 unwritten to **0/512**);
3. the combine pass walking every request, which reduced never-written partials over the
   FA2 executor's rows (an FA2 decode row went 5.8e-4 -> 0.155, and back to **0.00072**
   once restricted via upstream's `varlen_batch_idx_ptr`).

Each produced plausible numbers rather than an error, and two were located only by which
rows failed. Remaining: both wgmma rows are still wrong (10.88, 11.06) while the FA2 row
is correct, so the fault is now inside the upstream path's own result. The isolating
experiment is recorded -- force `max_splits_fa3 == 1` so the epilogue writes final O and
the combine never runs.

**A regression I introduced, and the control arm caught it.** v0.9.7's executor-aware
split denominator is reverted:

| | fused | baseline |
|---|---|---|
| before | 155.9 us | **72.3 us** |
| with the change | 156.3 us | **134.6 us** |

Dividing `kv_len_limit` by `clusters * ctas_per_sm` splits the FA2 queues twice as
finely; each extra split costs a partial write and a reduction pass that outweighs the
added concurrency. The fused path did not move, so the fused/baseline ratio would have
read as "improved from 2.16x to 1.16x" purely because the baseline got worse. That is
exactly what the control arm is for. Kept behind `FLASHINFER_SPLIT_BY_EXEC_SLOTS=1` for
reproducibility only; `QUEUE_EXEC` stays, since the explicit queue -> executor mapping is
not implicated.

## v0.9.7 — Per-queue split length follows the executor; planning cost is not the bottleneck (2026-10-06)

The scheduler's per-queue split length now accounts for which executor drains that
queue. The FA2 persistent kernel runs 128 threads at 2 CTAs/SM and the wgmma executor
384 threads at 1 CTA/SM, so the same SM count gives them different concurrency;
`kv_len_limit` was being divided by a bare cluster count, under-splitting the FA2 queues
by 2x against their real slot count. `QUEUE_EXEC` also makes the queue -> executor
mapping explicit instead of leaving task 3 implicitly special.

The rest of the "unified scheduler" asked for already existed: tile classes are routed
to queues by packed row count (>= 128 fills a wgmma tile, else 128/64/16), each queue
gets its own `cluster_len_kv_chunk`, and LPT min-heap assignment runs inside each queue.

**A device-side scheduler is not justified by the evidence.** Measured earlier:

| | tok/s | vs control |
|---|---|---|
| control, fused off | 22 203 | - |
| fused, original wiring | 20 260 | -8.7% |
| fused, planning deferred + indices reused + plan cached | 20 314 | **-8.5%** |

Removing essentially all host-side planning recovered 0.3%. That measurement had two
plans per step serving a kernel used on 3.2% of forwards, so the single unified plan's
per-step cost is still un-isolated -- which is the thing to measure before writing a GPU
scheduler, rather than assuming either way.

## v0.9.6 — One kernel can give either role its native registers, never both (2026-10-06)

The Q-split was the experiment that would have kept single-kernel fusion. It does not:

| FA2 configuration | isolated | inside the fused kernel |
|---|---|---|
| 4 warps, `num_mma_q=2` (today) | 1 136 B spill | 3 232 B spill |
| 8 warps, `num_mma_q=1` | **88 B** | **1 504 B** |

Halving the per-thread O accumulator is worth keeping -- it takes the fused kernel from
3 232 to 1 504 B -- but ~1.4 KB survives that no FA2-side change can reach, because the
FA3 role's live ranges share the allocation. With the converse already measured
(outlining FA2 puts FA3 at 64 B and moves 17 500 B onto FA2), the result is symmetric:
one `__global__` can give either role its native allocation, not both.

Decision: **the kernel splits, the plan does not.** The 4-mode tile assignment, the
work-proportional SM allocation, the unified JIT split-KV reduction and the fused
tree-build epilogue all live in `HolisticPlanInfo<4>`, which both executors read, so
none of them depends on the roles sharing a `__global__`. What is given up is
cross-role backfill inside a CTA -- itself the cause of the co-compilation spill -- and
cross-role smem multicast.

Graph count does not multiply, provided the split lives in device memory and both
executors launch fixed grids with unadmitted CTAs exiting early. This is required for
the fused kernel too: `sm_split` is a kernel argument today, which a graph would bake
at capture. sglang's graph layer already uses the same bucket-and-pad idea a level up
and treats prefill and decode as separate capture ladders, so two executors add no
PD-ratio combinations.

Also fixed here: the combine pass's per-request split count was derived from the plan's
`kv_indptr`, which is pushed once per WORK ITEM (`scheduler.cuh:1583`), so differencing
adjacent entries subtracted two unrelated items instead of bounding a request's pages.
That is what left 439/512 rows unwritten with the combine otherwise correctly wired.
Per-request KV lengths now come from the caller.

## v0.9.5 — FA2's spill is the O accumulator; the parity gap is 2.16x (2026-10-06)

Two measurements and one fix.

**The gap, on one build.** Fixture 2: 155.9 us fused against 72.3 us for
FA2-on-everything, both numerics-correct from the same build and inputs. 2.16x, which
is larger than the 1.83x recorded on the 16 823-token-KV fixture -- so the gap is
workload-dependent and the *shorter* fixture is the harder case. Worth knowing,
because the split sweep that produced the 1.83x ran on the easier one.

**What FA2's share of it is.** Three compile-only probes, each a 384-thread kernel
containing only the FA2 runners:

| FA2 configuration | threads used | spill stores |
|---|---|---|
| `4 warps_q, num_mma_q=2` (today) | 128 of 384 | 1 136 B |
| `4 warps_q, 3 warps_kv` (all threads busy) | 384 | **1 276 B** |
| `8 warps_q, num_mma_q=1` | 256 of 384 | **88 B** |
| native `128 x 2 CTA`, for reference | 256 | 0 B |

Occupying all 384 threads by splitting KV is *slightly worse* than leaving 256 idle:
every KV-warp still accumulates a full O tile over its q rows, so per-thread registers
do not fall and the cross-warp reduction adds work. That is a direct answer to
"offload some softmax work to the remaining threads" -- the softmax is not what holds
the registers. Splitting **Q** is: 8 warps with `num_mma_q=1` covers the same 128 rows
with 16 per warp instead of 32, halving the accumulator, and spill falls 12.9x to
within 88 B of native. Available as `FLASHINFER_FA2_QSPLIT=1`; tile 128 only, since
tile 64 already has 16 rows per warp and neither 64 nor 16 divides into 8 warps.

**Fix.** The combine launch failed with "invalid argument" because its grid is
`ceil_div(seqlen_q * num_heads, kBlockM)` and I passed `seqlen_q = 0`, making grid.x
zero. Upstream documents the field as the varlen *maximum* q length
(`flash_fwd_combine_kernel.h:292`); it now travels on the plan from both call sites.

Correctness and timing of the Q-split configuration inside the full fused kernel are
not yet measured -- the probes are compile-only.

## v0.9.4 — The port's split-KV path: upstream's combine pass, and two silent-wrong traps (2026-10-06)

Fixture 2 measured the predicted failure: with `max_splits_fa3=2`, both upstream arms
left **511 of 512 rows unwritten**, the single correct row being the FA2 decode row.
The port instantiates upstream with `Split=true`, so its epilogue takes the
`is_split` branch (`epilogue_fwd.hpp:232`) and writes partials instead of final O,
and flash-attention reduces those in a *separate* kernel
(`flash_fwd_combine_kernel.h`) that the port never launched. It does now, guarded on
`num_splits > 1` because at 1 the epilogue writes final O directly -- the same guard
upstream's own `flash_api` uses.

Two traps found while wiring it. Both would have produced plausible wrong numbers
rather than an error, which is the only reason they are worth writing down:

1. **Per-item `num_splits` lives in the top 16 bits of `split_idx`.** Upstream packs
   it there and masks the index to the low 16 (`epilogue_fwd.hpp:226-231`). Leaving
   the high bits zero makes every row fall back to the plan-wide maximum, so a row
   that did *not* split writes one partial, still takes the `is_split` branch, and
   the combine pass averages it against uninitialised memory. The adapter now encodes
   each row's own chunk count, which is what the field exists for.
2. **The combine pass needs the same per REQUEST**, through
   `num_splits_dynamic_ptr`; `nullptr` means "use the maximum for everyone" with the
   same consequence. A small kernel fills it from `kv_indptr` and `len_kv_chunk` --
   the same two quantities the epilogue derives from, so writer and reader cannot
   disagree.

The partials also now use upstream's own varlen layout,
`out_accum[num_splits, num_heads, total_q, head_dim]` (`flash_api.cpp:1102`), instead
of the equally dense but differently ordered layout I first wrote; the combine kernel
is written against upstream's, so mine would have reduced the wrong elements. Strides
and element counts come from shared helpers and the ext sizes its buffers from those
same helpers. Two sizing bugs fixed on the way: the O-partial split stride omitted
the `num_heads` factor, and O's strides were being taken from Q.

Also recorded: fixture 2's `def` arm is 156.8 us, and that is **not** a regression
against the 772 us in the status section -- the 772 belongs to the 16 823-token-KV
long-prefix fixture, while fixture 2's longest row is kv=5469. Two workloads, not a
change.

The default path builds unchanged throughout (0 errors).

## v0.9.3 — The two roles cannot both have their native register allocation in one launch (2026-10-06)

Five builds, `-Xptxas -v`, attributing every spill count to a named function. This
supersedes v0.9.2's conclusion, which was right that co-compilation causes FA3's
spill and wrong to imply outlining removes it -- it moves it.

| configuration | FA3 role | FA2 role |
|---|---|---|
| both roles inlined (today's default) | 33 676 B, shared allocation | shared |
| FA2 body outlined (`FLASHINFER_FUSED_DRAIN_NOINLINE=1`) | **64 B** | **17 500 / 17 664 B** |
| upstream FA3 alone, 384 thr | 52 B | — |
| FA2 alone, 384 thr, 168 regs | — | 1 136 B |
| FA2 alone, 128 thr x 2 CTA, 255 regs | — | **0 B** |

Two independent costs, now separated:

1. **Co-compilation.** FA3's 33 676 B is not intrinsic. Move the FA2 body out of the
   function and FA3 spills 64 B -- the standalone figure. ptxas allocates per
   function, so while both roles are inlined their live ranges share one
   allocation even though the roles sit in mutually exclusive `blockIdx` branches.

2. **Launch geometry.** Outlining does not make the spill disappear; the FA2 body
   then spills 17 500 B in its own ABI frame, because an ABI callee cannot use the
   caller's allocation. And FA2's floor inside *any* 384-thread launch is 1 136 B at
   168 registers, against 0 B at 255 registers in its native 128x2 geometry --
   measured with no FA3 in the kernel at all, so this is the geometry alone.

The arithmetic behind the second: 384 threads at 1 CTA/SM leaves 65536/384 = 170
registers per thread, and FA2's mainloop uses 255 when it can, which requires at
most 256 threads per SM. One launch has one blockDim, so the FA2 role cannot hold
255 registers per thread while the FA3 role needs 384 threads in one CTA for its
warp specialisation.

**Correction, same day.** I first concluded from that that FA2 parity is
unreachable in one launch. That does not follow, and the arithmetic says so:

    native FA2   255 regs x 256 threads/SM = 65 280
    fused FA2    170 regs x 384 threads/SM = 65 280

The SM-wide register budget is *identical*. What changes is the partition, not the
total, so the deficit is only real if FA2's work cannot be spread across three
warpgroups instead of two CTAs. That makes it an engineering question about work
distribution, not a hardware limit -- and `FLASHINFER_FUSED_WG_DRAIN=3` is exactly
that redistribution, currently producing wrong FA2 output. The honest status is that
FA2 parity is unproven and blocked on WG_DRAIN=3 being made correct, which is also
what "offload some softmax work to the remaining threads" asks for.

What the rows above *do* establish: the parity bar is not reachable by scheduling,
stage count or split policy, all held constant across them. FA3 parity is reachable
by outlining the other role. FA2's penalty at `WG_DRAIN=1` is 1 136 B of spill and
87 registers per thread, and that figure is the cost of leaving 256 of 384 threads
idle, not a floor.

`FLASHINFER_FUSED_DRAIN_NOINLINE` is kept as a switch rather than turned on,
because it is a trade whose sign depends on the work mix: it should win where FA3
dominates (fixture 2 is ~99% task-3 work) and lose on decode-heavy batches. Which
way it goes is a measurement, not a choice, and has not been run yet.

## v0.9.2 — Measured: single-kernel co-compilation, not the register cap, is what starves FA3 (2026-10-06)

The register-pressure hypothesis from v0.9.0 is confirmed in outline and wrong in
its mechanism, and the corrected version is sharper.

`-Xptxas -v` on three builds of the same kernel, all `__launch_bounds__(384, 1)`:

| build | regs | stack | spill stores | spill loads |
|---|---|---|---|---|
| upstream FA3 alone, our template | 168 | 32 B | 52 B | 40 B |
| upstream FA3 + the three FA2 runners | 168 | 2 976 B | 33 676 B | 37 852 B |
| flashinfer FA3 + the three FA2 runners | 168 | 848 B | 3 232 B | 5 028 B |

I had predicted that upstream would be starved because it asks for 232 registers per
MMA warpgroup through `warpgroup_reg_alloc` while `__launch_bounds__(384, 1)` caps
the uniform allocation near 170, and that the fix was to make the FA2 role
participate in the same register protocol. That prediction is refuted: upstream
compiled **alone** in exactly the same template, with the same bounds and the same
168 registers, spills 52 bytes. The cap is not the problem.

What costs the 33 KB is that both roles are one function. ptxas allocates registers
per kernel, so the FA2 mainloop's live ranges and upstream's share a single
allocation, and at equal register count the fused build spills 648x more than the
standalone. The default path shows the same effect an order of magnitude smaller
(3 232 B), which is why it was visible but not fatal there.

This is a property of single-kernel fusion itself, not of the split policy, the
thread width, or the stage count -- none of which differ between the two rows that
matter. It is also why ptxas emits C7510 (wgmma serialized across a function
boundary) only on the upstream path.

The consequence for the goal is direct: per-role parity cannot be reached by tuning
the split while both roles share one register allocation. Either the roles must be
separated in a way that gives each its own allocation, or the FA2 role's live ranges
must be cut down. No parity number should be quoted until this is resolved, and the
1.83x gap already measured is most likely this.

## v0.9.1 — The upstream FA3 port compiles, and the planner exports its split count (2026-10-06)

The port's dispatch now instantiates end to end with `FLASHINFER_FA3_UPSTREAM=1`:
a TU that takes the address of `FusedFA3PersistentDispatched` compiles the whole
upstream kernel body with 0 errors, and the default path compiles from the same TU
unchanged. Previous commits in this port claimed "compile-checked" on code that the
upstream scheduler's `if constexpr` false branch never instantiated, so everything
here is verified by forcing instantiation rather than by the header parsing.

Every params field name in the dispatch had been guessed, and every guess was wrong
in the same way: the KV pointers and strides were read off `params_1`, the FA2
role's `PersistentParams`, as `k_data`/`v_data`/`kv_stride_n`. Those members do not
exist (they are `k`/`v`/`k_stride_n`), but the deeper error is that the wgmma role
was reading the other role's params at all -- `FA3Params` carries its own
`k_ptr`/`v_ptr`/`k_stride_n`/`k_stride_h`/`kv_indices`/`page_size`, and taking them
from the FA2 role would have been correct only because this harness happens to hand
both roles the same cache.

`max_splits_fa3` is now exported through `plan_info`. Upstream sizes its float
partial O/LSE as `total_q * num_splits`, so the buffers cannot be sized by a
conservative bound -- the obvious one, pages per sequence, asks for
`8192 * 32 * 8192 * 4` bytes of LSE alone at a 8192-token batch. The planner knows
the exact value, because it is the `num_kv_tiles` it just chose, and nothing
downstream can recompute it. It is placed *before* `num_fa3_clusters` in the
serialised vector so that stays last: the harness reads the chosen split as
`plan_info[-1]`, and appending would have handed it the split count instead -- a
plausible small integer that would have planned FA3 for the wrong CTA count with no
error anywhere.

The export is guarded by `NUM_TASKS > kFA3Task`. I first wrote it unguarded on the
assumption that the enclosing block was already that guard; it is a bare scope, and
`task_max_kv_splits` has `NUM_TASKS` entries, so every `HolisticPlanInfo<3>` plan
would have read index 3 out of range.

The three workspace buffers are allocated in the ext, and `num_requests` /
`max_pages_per_seq` are threaded from both call sites (the harness and the serving
backend). The dense page table is filled with -1, not 0, because 0 is a valid page
id and zero padding would make every short sequence read page 0 for its tail blocks
instead of being masked.

Still unvalidated at runtime: this machine has no GPU, so no number from the
upstream path means anything yet.

## v0.9.0 — Register pressure is the leading candidate for the 2.19x (2026-10-05)

Grounded in ptxas numbers already in `persistent_fa3_fused.cuh`, not in new
reasoning. The fused kernel is `__launch_bounds__(384, 1)` because that is FA3's
warp-specialised width, capping ~170 registers per thread; the standalone persistent
kernel at 128 threads x 2 CTAs/SM has roughly 256. The file records the consequence:
the fused body compiles to 168 registers with up to 4.4 KB of spill loads, and 6284
bytes of spill stores in one configuration.

It fits every measurement the other candidates failed. It is grid-independent (SM
reservation 6%), protocol-independent (Q completion 0.3%), predicts that more
warpgroups are worse (width 3: 147.9 vs 132.4 us, since extra warpgroups do not
raise the ceiling), and survives at equal threads per SM -- the baseline forced to 1
CTA/SM has the same 128 FA2 threads per SM and is still 2.19x faster.

The test needs no GPU: compile the persistent mainloop under `__launch_bounds__(128,
2)` against `(384, 1)` with `-Xptxas -v` and compare registers and spills. If it
holds, the design implication is sharp -- one launch forces one set of launch bounds
on both roles, the wider warp specialisation sets them, and SM pre-assignment inside
a single kernel cannot be made free.


## v0.8.9 — The serving fallback is strict by default (2026-10-05)

A fused planning failure used to degrade to the persistent path with a warning. That
is how the width-3 arm produced an apparent win -- 505.85 tok/s against 397.88 --
while the fused launch was actually being rejected and roughly half the steps ran on
the faster unfused kernel (composition 1897 at 47.4% against 53116 at 96.2%). The
warning did not help, because the number was still produced and believed.

`_plan_fused_split` now raises, naming the cause and saying that a run with a
fallback compares a mixture rather than the fused path.
`SGLANG_FUSED_ALLOW_FALLBACK=1` restores degradation for keeping a server alive, and
never for measurement.


## v0.8.8 — The penalty factorises: 1.38x structural, 2.19x unexplained (2026-10-05)

`scheduler.cuh:1199` gives the 3-task plan 2 CTAs/SM and the fused 4-task plan one,
since FA3's ~98 KB of smem makes two impossible. Forcing the baseline to one CTA/SM
isolates it: 43.8 us at two, 60.3 us at one, against the fused kernel's 132.1 us. So
3.02x = 1.38x (one CTA/SM) x 2.19x (the fused kernel itself).

The 1.38x is the structural entry price of co-residency with FA3's smem and is not
addressable by scheduling at any quality. The 2.19x remains uncaused, with the Q
protocol (0.3%), SM reservation (6%), thread width (negative), plan construction
(none) and now CTA count (1.38x) all measured and eliminated.

It also corrects an earlier framing error here: I called the launch-geometry finding
"refuted" by the width result, conflating threads within a CTA (more is worse --
warpgroups contend on one queue cursor) with CTAs per SM (worth 1.38x -- independent
contexts). This file used "launch geometry" for both; they are different mechanisms.


## v0.8.7 — Aliased smem: the first measured win, 11% (2026-10-05)

With width held at 1 and smem the only variable: aliased 118.1 us against disjoint
132.4 us, baseline 43.8-44.0 us in the same runs. 11% faster but NOT free, and the default must NOT flip --
corrected after reading persistent_fa3_fused.cuh:78-93. Disjoint is the fix for a
documented race: CUTLASS initialises pipeline mbarriers before the sm_split branch,
so an aliased base of 0 puts the drain's slot inside FA3's SharedStorage, which gave
nondeterministic rel err 1.93 / 2.53 / 3.31 -- and it fires only when a CTA drains
FA2 before entering FA3. My run had the wgmma role EMPTY, so nothing entered FA3
after a drain and the race could not appear. Capturing the 11% means removing the
race (re-initialising FA3 pipeline state after the rendezvous), not setting a flag.

It does not explain the 3x. 2.68x remains. Measured contributions on identical work
now read: SM reservation ~6% over an 8x sweep, smem layout 11%, residual ~2.7x
uncaused.

The queue protocol is the primary suspect, from two independent results: adding
warpgroups made the kernel slower (147.9 vs 132.4 us), which is contention rather
than capacity, and the penalty barely moves with the number of reserved SMs, which
fits a serialised hand-off. Next measurement is the per-task profiler events in
`persistent.cuh` -- queue acquire versus mainloop, per role.


## v0.8.6 — The 3x is intrinsic to the fused kernel (2026-10-05)

Sweeping sm_split with the wgmma role empty and plan_ctas matched at every point:
129.7 / 130.5 / 132.2 / 137.4 us at 2 / 4 / 8 / 16 reserved clusters, against a
42.6-43.9 us unfused baseline measured in the same runs. An 8x change in reserved
clusters costs 6%, so SM reservation is not the cause and the confound flagged last
entry is real but small -- my hypothesis that it explained the 3.01x is refuted.

What stands: the fused kernel is ~3x slower than the unfused persistent kernel on
identical work (3.02-3.13x across configurations, matching the 3.00x serving ITL
ratio), and that penalty belongs to the fused kernel itself. The SM split, the Q
completion protocol, thread width, and plan construction have each now been measured
and eliminated.

This reframes the effort: the scheduling argument is about how work is distributed
and cannot be worth 3x. No schedule reaches the 30% target while the fused path
starts a factor of three behind. Next, and never yet measured here: FA2 occupancy
inside the fused kernel (1 CTA/SM where it natively runs 2), the element-wise
packed-GQA O store, and the queue protocol -- which the width-3 result hinted at,
since adding warpgroups made it slower, as a contended queue would.


## v0.8.5 — The kernel-level ratio: 3.01x, and the confound in it (2026-10-05)

With a separate 3-task plan as the reference, the fused kernel runs 132.2 us against
the unfused persistent kernel's 43.9 us on identical work -- 3.01x, matching the
serving ITL ratio of 3.00 to within noise. That settles host versus device: the cost
is in the kernel, and one fused plan replacing two is not the problem.

The confound: this configuration routes every row to an FA2 class, so the wgmma role
has no work, yet sm_split=8 still reserves 8 clusters for it. Idle reserved SMs are
part of that 3.01x and are a property of static pre-assignment, not of fusion. This
also qualifies the earlier decomposition -- the "fused FA2 path costs 1.50x" arm had
the same 8 idle clusters, so that figure is reservation waste plus something else,
not the intrinsic FA2-path cost I presented it as.

An sm_split sweep on the fixed batch separates the two, and the harness now has both
the timing loop and the reference to do it.


## v0.8.4 — More threads are slower; width 3 is numerically fine (2026-10-05)

Fixed-batch kernel timing settles the thread-count question against my hypothesis:
width 1 (FA2 role on 128 of 384 threads, disjoint smem) runs 132.4 us; width 3 (all
384, aliased smem) runs 147.9 us. Giving the FA2 role every thread makes the kernel
12% slower, so the launch-geometry framing is wrong as a performance explanation and
the warpgroup-reentrant path is a dead end for perf.

Separately it corrects a standing catalogue claim: width 3's numerics are CORRECT
with aliased smem (rel err 3.5e-3 / 3.8e-3 / 6.0e-4, all rows pass). The recorded
"FA2 numerics wrong at width 3" was almost certainly the disjoint-smem config -- the
same condition that makes the launch fail with 'invalid argument', since three
warpgroups need 208,896 of 232,448 B and only fit aliased. The fault was
configuration, not the reentrant logic.

The 1.50x fused-FA2 cost remains unexplained, now with its most plausible mechanism
eliminated, and 132.4 us still lacks an unfused reference.


## v0.8.3 — The width-3 probe is void by its own validity check (2026-10-05)

Giving the fused FA2 role all 384 threads (`SGLANG_FUSED_WG_DRAIN=3`, numerically
wrong, perf probe only) appeared to recover most of the 1.50x: 505.85 tok/s / 21.47
ms ITL against width 1's 397.88 / 27.41. The composition counter refutes it --
1897 (47.4%) against 53116 (96.2%), so the arm pushed a twenty-eighth as much work
through the fused path. Corrupt FA2 output ends sequences early, fewer decode steps
follow, and the benchmark finishes faster by doing less; TTFT dropping to 3255.95 ms
is not what a decode-side thread-count fix looks like.

The thread-count attribution is therefore unproven and serving is the wrong
instrument for it -- a wrong-numerics arm can always shift composition and buy an
apparent win. It needs a kernel-level timing on a fixed fixture batch instead. The
1.50x itself stands; its cause does not.


## v0.8.2 — The 66% decomposes into two causes (2026-10-05)

Routing every row to the fused kernel's FA2 roles, leaving the wgmma role idle,
splits the deficit: baseline 590.58 tok/s / 18.26 ms ITL, fused-FA2-only 397.88 /
27.41 (1.50x ITL), fused-with-wgmma 198.16 / 54.86 (3.00x ITL).

Two roughly equal multiplicative costs, not one. The first is the launch geometry
already measured in this effort and never connected to this regression: 1 CTA/SM at
384 threads where FA2 natively runs 2 CTAs/SM, ~128 of 384 threads doing FA2 work.
The second, a further 2.00x, is isolated to the wgmma role and still uncaused;
the element-wise packed-GQA O store and 200 task-3 items against ~34 q-tiles of real
work are the measurable candidates.

Also noted: `mixed_fused_fa3=53116(96.2%)` is identical across all three arms. It
records the dispatch decision, not which internal role ran, and must not be read as
evidence the wgmma role executed.


## v0.8.1 — The Q drain is exonerated; the 66% has no known cause (2026-10-05)

Replacing the cp.async group drain with the validated per-lane `_noinc` mbarrier
arrival moves serving by 0.3%: 198.16 tok/s against 197.65, ITL 54.86 against
55.01, identical 96.2% fused composition, baseline 590.58 / 18.26. The drain was
not the cause of the 3x.

The attribution was mine and it was wrong. I had written that the deficit "matches
the cp.async drain's cost exactly", carrying a ~3x ratio from a different
configuration into this workload as though it had been measured here. It had not.
The 66% regression has no identified cause.

What is settled: the Q completion protocol. `_noinc` is correct (fixture 2, rel err
3.3e-3, 0/512 unwritten), needs no drain, costs nothing, and can stay on. Next step
is a profile rather than another mechanism argument -- three mechanism hypotheses
have now each been refuted by measurement (static split, redundant chunks, the
drain).


## v0.8.0 — The chunks are real partials; hypothesis refuted (2026-10-05)

The per-item dump shows task 3's chunks are genuine slices (`span=384` + `span=55`
against `kv_len=439`, each with its own group id and partial_indptr), not full-row
copies. So the redundant-work hypothesis is refuted and there is no second hidden
cause of the serving deficit -- the Q completion remains the one measured cause.

It also dissolves the contradiction: the old fixture-2 pass reported that row as
`task0`, an FA2 class that reduces correctly, and it is routed to task 3 only now
that packed qo = 3073 exceeds the threshold. So the old pass was correct for a
different routing, and calling it "suspect" is withdrawn.

Net: the reduction spec stands as written, the guard and its 200-item count are
correct, and the cheap path to re-measuring the regression is to raise
`FLASHINFER_FA3_MIN_PACKED_QO` above 3073 so the split row goes to FA2 -- which
reproduces the original routing and unblocks validating `_noinc` against the drain.


## v0.7.9 — 200 split chunks and a correct answer (2026-10-05)

The exact guard reports 200 task-3 work items carrying JIT-reduction group ids on
fixture 2, in both arms. Fixture 2 nevertheless passed at rel err 3.2e-3 with 0/512
unwritten rows, and 200 unmerged distinct slices sharing final O rows cannot produce
that. Leading hypothesis: the plan emits each chunk's `kv_start`/`kv_end` as the
row's full range, so the chunks are redundant rather than partial -- identical
overwrites, invisible to a correctness check, and roughly 6x duplicated attention on
the wgmma role. That is the same order as the 3x serving deficit and independent of
the Q drain, so it may be a second and larger cause the drain estimate was masking.

Consequence for the plan of record: do not implement the reduction as specced until
this is settled by dumping task 3's per-item ranges, because the spec assumes the
chunks are genuine partials.


## v0.7.8 — The guard fires on fixture 2 (2026-10-05)

Ran the plan guard against fixture 2 to validate Q's completion protocol and got
a more useful answer than expected: the guard fires, in both arms, at
`scheduler.cuh:1495`. Fixture 2 routes two rows to task 3 (`qo=72` packed 504 and
`qo=439` packed 3073), 675 KV over 8 clusters gives a `kv_len_limit` near 84, and
the 439-long row splits about six ways. So my claim that fixture 2's FA3 row is
`kv=236` and never splits was wrong, and its earlier PASS at rel err 3.2e-3 is
suspect rather than reassuring -- either the overwrite was benign for that shape
or the guard over-fires relative to the chunk loop. Resolving which is step one of
the reduction work.

Net effect: the wgmma JIT reduction is now the hard blocker. With the guard in,
fixture 2 does not run, so there is no fixture left on which to settle the Q
completion protocol (`_noinc` vs drain) either.


## v0.7.7 — The first valid unified serving number: -66% (2026-10-05)

Measured on a tree verified byte-exact against our fork, both arms back to back:
baseline 590.58 output tok/s against unified 197.65 (0.335x), mean ITL 18.26 against
55.01 (3.01x), mean TTFT 2626.50 against 9333.23, with 96.2% of rows through the
fused path -- so this measures the kernel, not a dispatch accident. Full table,
commits and command in `tasks/debug_kernel_server_perf_4_mode.md`.

The 3x matches the packed-GQA Q drain exactly, and the Q-before-K hoist was
verifiably compiled in, which rules out ordering: the fence sweeps every cp.async
the producer has uncommitted from earlier work items, and groups retire in order.
The per-lane `cp.async.mbarrier.arrive` that has the right scope hung the server on
its first fused batch and is now gated off behind `FLASHINFER_FUSED_Q_MBARRIER`,
to be settled on fixture 2 rather than in another serving run.


## v0.7.6 — Two findings that invalidate measurements, not numbers (2026-10-05)

### The wgmma role's split-KV path is a latent wrong answer

Recorded as "unexercised" before, which was too kind. `scheduler.cuh`'s chunking
loop covers every task including `kFA3Task`, the unified scheduler carries
`partial_indptr`/`group_id` per work item, and `persistent_fa3_fused.cuh`
references neither -- nor anything else about partials -- so each chunk of a
split task-3 row writes its own slice straight to the final O rows and the
chunks overwrite each other. Fixture 2 is quiet about it only because its
FA3-class row is `kv=236` and never splits.

Corrected after measuring: I expected this to break the 8192 workload too, but
the unified arm ran all 200 prompts with the guard compiled in, so nothing
split there and the earlier unified numbers are slow rather than wrong. Whether
that is because no task-3 row exceeds its `kv_len_limit` or because the guard is
inert is still open -- the gate fixture decides it, and a guard that has never
fired proves nothing.

### The restored pod's serving tree was upstream sglang, not our fork

`/dev/shm/w/sgl_src` is a clone of upstream at `1818db4`, and the tree on
`PYTHONPATH` was built from it with four of our files copied on top. Against an
md5 manifest of our fork's 725 `.py` files: 4140 files on the pod, 3573 extra,
158 missing, 514 present in both with different contents -- `bench_serving.py`
among them. Fixtures are unaffected (they import `flashinfer` and `torch`, never
`sglang`), so fixture 2's pass and the `barrier_Q` diagnosis stand. Both forks
are private and this machine has no push credentials, so the repair is a chunked
webshell transfer of the real tree. Details, and the torch-ABI dependency chase
that preceded it, in `tasks/debug_kernel_server_perf_4_mode.md`.

## v0.7.5 — Q's drain stops draining K/V (2026-10-05)

### The packed-GQA Q load is now issued before the first K tile

The packed-GQA Q load on the sparse path finishes with a cp.async group drain
(`cp_async_fence` + `cp_async_wait<0>`) and then a single-thread arrive on
`barrier_Q`. That drain is what made the correct unified kernel ~3x slow
(8,379 against 24,885 tok/s), and the reason was mis-stated in the code: the
comment claimed narrowing the wait needed the KV pipeline's outstanding group
count.

There is no such count to learn. The KV copies signal completion through
`cpasync_barrier_arrive`, the mbarrier mechanism, and are never committed to a
cp.async group at all -- so `cp.async.commit_group` sweeps them, still
uncommitted, into the very group it closes for Q. Committed groups also retire
in order, so no `wait_group N` bound can retire Q's group while an older group
is outstanding. Both facts point the same way: the only way for Q's drain to
cover Q alone is for nothing older to be in flight when it runs.

So the Q load moves above the last-k-tile load, as a `load_q_tile` lambda
called from one of two sites (`FLASHINFER_FUSED_Q_LATE` restores the old
ordering for A/B). A per-lane `cpasync_barrier_arrive` on `barrier_Q` was the
other candidate and cannot work: that instruction tracks only the cp.async of
its own issuing thread, so all 32 lanes would have to arrive on a barrier whose
arrival count is 1 -- which is exactly the 1500 s hang already paid for once.

Compile-checked in three configurations (plain, packed-GQA, and `Q_LATE`).
Not yet measured on device: the serving A/B is what decides whether the ~3x
comes back, and it is reported in
`tasks/debug_kernel_server_perf_4_mode.md`.

### Fix: the lambda no longer captures structured bindings

Hoisting the block into a lambda made it capture `q_tile_idx`, `kv_head_idx`,
`qo_indptr`, `qo_len` and the Q tensors, all of which come from structured
bindings -- capturing those is C++20 and this builds as C++17 (nvcc only warns).
They are passed as init-captures over plain aliases instead, keeping the
original names so the body is unchanged.

### Provisioning: install sglang's dependency closure from the released package

Hand-listing dependencies cost several server-start cycles (pynvml, IPython,
`sgl_kernel`, one at a time), and `sgl_kernel` cannot come from a source tree.
`drivers/install_deps.sh` installs `sglang[srt]` for the closure and then
re-pins `torch==2.8.0+cu128`, which is mandatory rather than cosmetic: that
install drags torch forward, the fused extension needs 2.8.0, and a drifted
torch would quietly make measurements incomparable.

## v0.7.4 — One Scheduler for Both Roles (2026-10-05)

### Refactor: the wgmma role becomes a task in our own plan, not `PrefillSM90Plan`

The FA3 role was planned by flashinfer's `PrefillSM90Plan`, which measurement
showed cannot serve this kernel: it emits one work item per (q-tile x qo head x
row) carrying the **full** `kv_len`, with no partial-O/LSE buffers and no merge
map. Concretely, as the server planned it, the wgmma role had

- **no GQA packing** — `num_qo_tiles = ceil_div(qo_len, cta_tile_q)`
  (`scheduler.cuh:939`) plus upstream's own note that FA3 "do[es] not fuse query
  and group heads", against `qo_len * gqa_group_size` in the persistent plan
  (`scheduler.cuh:1199`);
- **no KV splitting** — one CTA walks a whole KV range.

At the measured serving shapes (median 11 decode rows totalling 47,935 KV, one
500-token prefill chunk over a 2,595-token prefix) that leaves the role almost no
parallel width, which is why fusing it changed a mixed step by -0.21% and
end-to-end throughput by -0.65%: parity was the *expected* result, not a tuning
failure.

`TwoStageHolisticPlan` is now generic in its task count. A 4-task plan adds the
wgmma class as task 3, which therefore inherits, with no new machinery:

- packed-GQA tiling (every class in this plan tiles packed rows),
- split-KV via the existing `kv_len_limit` chunk loop,
- a JIT-reduction group (`group_id`, `partial_indptr`, `merge_indptr`,
  `group_counter`) so partials are merged in-kernel by whichever work item
  arrives last, with no grid barrier and no second launch.

A 3-task `plan_info` selects exactly the previous behaviour, so `BatchAttention`
cannot silently drop task-3 work. Admission is `packed_qo_len >=
FLASHINFER_FA3_MIN_PACKED_QO` (default 128 = fills one packed tile), tunable for
the boundary sweep rather than asserted. Both instantiations compile for sm_90a.

### Measured: the symmetric flag is worth 1.59x on the serving path
At `sm_fa3=8`, asymmetric (8 CTAs own all FA3 work) vs symmetric: 6,279 vs
9,990 tok/s, mean ITL 73.05 vs 46.08 ms, p99 ITL 170.84 vs 63.44 ms, on identical
batch shapes (p10/p50 step time matched to 2%; only the tail moved). Previously
this was a kernel-level number (0.687x -> 1.003x of monolithic) only.

### Measurement hygiene: a stale diagnostic invalidated a day of numbers
A detached `compute-sanitizer synccheck` from the CLUSTER=2 investigation was
still resident after 2h54m at 100% CPU, holding 2,760 MiB. Every arm measured
alongside it ran 2.45x slow (fused 9,990 vs 24,431 tok/s on the same command) and
*stayed consistent with its own control*, so no A/B looked wrong. Killing it
restored yesterday's scale. The measurement driver now refuses to start while
`nvidia-smi --query-compute-apps` is non-empty, and synccheck is recorded as
non-viable on this kernel (3 hours without reaching a launch result).

### Per-step trace carries features and GPU time
`SGLANG_STEP_TRACE=1` now logs per-step row classes (prefill rows, chunk tokens,
prefix, decode rows, decode KV, total KV, max chunk) and the step's own GPU span
from CUDA events read a step later, so nothing synchronises on the measured path.
Cost: 0.02% (24,668 vs 24,431 tok/s traced/untraced), which corrects the earlier
claim that these runs were "depressed by per-step logging".

## v0.7.3 — Hopper 4-Mode Fused Kernel: Multicast Tested, Doc Split (2026-10-05)

### Docs: split by effort, with lineage
`tasks/debug_kernel_server_perf.md` had reached 3428 lines spanning two unrelated
efforts. Split:

| doc | covers |
|---|---|
| `debug_kernel_server_perf.md` (1783 lines, closed) | Parts 1-4: sync-free plan, measured mixed-kernel gate, workload stats, mixed-tile 4-mode *dispatch* |
| `debug_kernel_server_perf_4_mode.md` (new, active) | the fused FA3 + FA2 SM pre-assignment **kernel**: correctness, server integration, split/cluster-cap tuning, backfill, multicast |

Each carries a lineage header pointing at the other, and the old file ends with a
forward pointer. Corrections to earlier claims are kept in place in both rather
than edited away — several conclusions were reversed by later measurement and the
reversals are the useful part.

### Measured: multicast tested on hardware (flashinfer `4eb41c6`, `88fc805`)
Multicast had been written off as needing a `sparse_mainloop.cuh` gather→TMA
rewrite. That conflated multicast with *TMA* multicast: Hopper DSMEM gives the same
load-once-consume-in-N-CTAs property with no tensormap and no KV-pool change, via
`cg::cluster_group::map_shared_rank` (already used in-tree by
`trtllm/fused_moe/RoutingKernel.cuh`).

`tests/dsmem_kv_share.cu`, built to match the redundancy this kernel has — CTAs
paired on the same KV tile, since every q-tile of a prefill row re-gathers the same
KV:

| working set | redundant gather | DSMEM shared | speedup |
|---|---|---|---|
| 16 MB (**L2-resident, in-situ**) | 682.436 us | 514.583 us | **1.326x** |
| 256 MB (cold) | 1325.362 us | 835.079 us | 1.587x |

L2 halves the baseline but costs DSMEM only 1.587x → 1.326x, because a
`page_size=1` gather is bound by *transaction count*, not by which level answers
it. L2 shortens a repeat; DSMEM removes it.

### Added: real cluster launch, opt-in (flashinfer `610d8d7`, `986270d`)
`FLASHINFER_FUSED_CLUSTER=N` switches the fused dispatch to `cudaLaunchKernelExC`
with a real `clusterDim`, default 1 so the shipped path is byte-identical. It
refuses rather than reshapes when N does not divide the SM count, since rounding
the grid would break the CTA count FA3's contract checks.

Isolation matrix via `FLASHINFER_FUSED_SKIP_FA3=1`:

```
CLUSTER=1 SKIP_FA3=0   rc=0  PASS
CLUSTER=2 SKIP_FA3=0   rc=1  CUDA error: unspecified launch failure
CLUSTER=1 SKIP_FA3=1   rc=1  424/512 unwritten (expected), no CUDA error
CLUSTER=2 SKIP_FA3=1   rc=1  424/512 unwritten (expected), no CUDA error
```

**The FA2 side is cluster-safe; FA3 alone is cluster-hostile.** This overturns an
earlier conclusion in this work that the single-grid two-role design forecloses
per-role cluster features — the side that wants DSMEM is already cluster-ready, and
the remaining work is making FA3 tolerate a cluster it does not use.

### Resource correction
Two H20 trials had been allocated for the PD-colocation experiments before the
single-GPU kernel work finished, and both sat idle for the session:
`81d861acb7b7c964` (4 nodes x 8 = **32 GPUs**, over the 16-GPU cap) and
`a1edeb8114365af4` (8 x 2 = 16 GPUs, within cap). Both now `STOPPED`. The
multi-node work is gated behind the kernel work by the goal's own ordering, so
allocating ahead of it was wrong regardless of the cap.

### Still open
4-mode MTP **serving** schedule unreachable (`forward_batch_speculative_generation`
would need reordering against the scheduler), so every server number remains
spec-decoding-off. Multicast measured but not integrated. Best server
configuration is the confirmed +0.7-0.8% win from v0.7.2.

## v0.7.2 — The Fused Kernel Reaches the Serving Path: Three Bugs, and a Measured 8.1% Regression (2026-10-04)

### Fixed: the page-index array handed to the fused kernel was wrong
The previous session's planning optimisation reused `wrappers[0]._kv_indices` on
the assumption that the mixed wrapper, planned over the whole batch moments
earlier, already held exactly the page-id array the launch needs. It does not.

Found by dumping the launch arguments to disk immediately before the faulting
launch (`SGLANG_FUSED_DUMP=<path>`, written at the launch site so it survives the
abort) rather than by a third hypothesis. On the faulting step:

```
qo = [424, 87, 1]   prefix = [3577, 0, 13]   ->  kv = [4001, 87, 14]
kv_indices: 3846 entries, batch needs 4102
            256 entries outside the valid page range [0, 1816955]
            first invalid at index 3590, value 1085595437
q, o, kv_indices, fa3_sparse base pointers: all 16-byte aligned
```

So nothing was misaligned. The kernel was gathering garbage page ids — up to
1.1e9 against a 1.8e6 page pool — from an array both too short and partly
uninitialised, and faulted on the resulting address. The optimisation it replaced
was worth ~0.3% of throughput.

- `_fused_kv_indices` now builds the array vectorised on the GPU (one
  `repeat_interleave` for each output slot's row, one subtract for its offset,
  one gather from `req_to_token`), so correctness does not cost the planning time
  back, and no per-request Python loop returns.
- It is checked against the plan's own page total before launch. The absence of
  that check is why a wrong buffer reached the GPU as a fault rather than a
  declined plan.

### Fixed: the tile-128 task was never executed (flashinfer `b43e50b`)
The fused extension built persistent params for tasks 1 and 2 only, on the
comment "task 0 went to FA3". The persistent plan is built over the FA2
sub-batch *after* the FA3 rows are removed, so its task 0 is ordinary FA2 work:
`TwoStageHolisticPlan` sends every row with `packed_qo_len > 64` there, and folds
the whole tile-64 mid class in too whenever that class carries less than
`FLASHINFER_TRI_TILE_MIN_KV` total KV. Nothing executed it.

No harness could have caught it: they all use MTP decode rows for the FA2 side,
whose `packed_qo = 2 * gqa = 14` lands in task 2, so task 0 was always empty. In
the dumped batch row 1 (`packed_qo = 609`) landed in task 0 and was skipped.

Adds `R128 = PKT<128, 2>` threaded through `fused_drain_queue`,
`FusedFA3PersistentKernel` and `FusedFA3PersistentDispatched`. Shared memory is
unchanged — tile 128 needs 49152 bytes against tile 16's 69632, so `Runner3`
still sets the bound.

### Measured (H200, flashinfer `b43e50b`, sglang this commit)
`tests/test_fused_server_batch.py`, which replays the dumped batch shape verbatim
so one launch covers FA3, task 0 and task 2:

| kernel | unwritten rows | FA3 row | task-0 row | task-16 row |
|---|---|---|---|---|
| pre-fix | **87 / 512** at every split | 0.00029 | **nan** | 0.00293 |
| fixed | 0 / 512 | 0.00023 | 0.00196 | 0.00271 |

87 is exactly row 1's token count — the task-0 row. Relative error is against an
fp32 reference, at `sm_split` 48 / 32 / 16.

Command: `FI_ROOT=/dev/shm/w/fi python3 tests/test_fused_server_batch.py`

### Ruled out
`tests/align_fused.py` reproduces each server-side tensor provenance in the
standalone harness — `q` as a non-contiguous view of a fused QKV buffer, `o` as a
slice of a reused buffer, the index array built by concatenating `req_to_token`
rows. All pass, so provenance was never the cause, contrary to the hypothesis
recorded in the previous session.

### Fixed: the KV cache does not have a fixed rank (flashinfer `4d59914`)
The extension read `k_cache.stride(1)` as the token stride and `stride(2)` as the
head stride. The standalone harness slices a `[pages, 2, page_size, H, D]` buffer
and so passes **rank 4**; sglang's pool passes **rank 3** `[tokens, H, D]`. On the
dumped batch that gave `k_stride_h = 1`, so all four KV heads aliased one head's
memory and `get_gmem_layout` built a degenerate layout whose tensor map made
`cudaLaunchKernel` **itself** fail with "misaligned address". The failed launch
poisoned the CUDA context, so the next unrelated attention call failed too and
killed the server — which is what earlier runs mistook for a fault inside
flashinfer's own persistent kernel.

The page step survived both layouts by accident (`ratio * k_stride_n` is
2*512 for rank 4 and 4*128 for rank 3, both correct), which is why only the head
stride was wrong and the symptom was a launch rejection rather than wrong numbers.

`kv_strides()` now derives `(page, n, h)` from the rank and rejects any other
rank; `_fa3_sparse_indices` picks the matching vector-sparse ratio.

Ruled out by measurement, not argument: tensor provenance (`tests/align_fused.py`)
and the CUDA 12.8 vs 13.1 toolchain split (`tests/test_fused_server_batch.py`
passes under the server's own venv python).

### Measured: the fused kernel runs server-side, and costs 8.1%
First valid server-side A/B. `--random-input-len 8192 --random-output-len 192
--num-prompts 300 --request-rate 12 --max-concurrency 32`, qwen7b on H200,
CUDA 12.8 / torch 2.8.0, sglang this commit + flashinfer `4d59914`:

| metric | fused | base | delta |
|---|---|---|---|
| total token throughput (tok/s) | 23250.44 | **25291.44** | **-8.1%** |
| mean TTFT (ms) | 3443.97 | 3150.61 | +9.3% |
| mean ITL (ms) | 20.18 | 18.54 | +8.8% |
| p99 ITL (ms) | 31.87 | 31.62 | +0.8% |
| `mixed_fused_fa3` share | 29540/32400 (**91.2%**) | absent | — |

300/300 requests succeeded in both arms, 0 misaligned, 0 launch failures. The
direction agrees with the kernel-level result (0.410x the sequential co-launch,
0.709x at the best split on H200), so the remaining deficit is the structural one,
not wiring.

An earlier A/B at `--random-input-len 2048` showed a null result (18182 vs 18203
tok/s) and was **invalid**: at that length no row clears the FA3 gate's
`kv_tiles >= 32` (>= 4096 KV) requirement, so `mixed_fused_fa3` was absent from
both arms and they ran the same kernel. Every fused measurement must carry its
`mixed_fused_fa3` share or it is not evidence about this kernel.

### Optimized: sweeping the SM split recovers more than half the regression
`sm_fa3` was pinned at 48 on every serving run by `plan_dispatch_prefix`'s
`MAX_SM_FA3`, so the kernel's main knob had never been varied server-side.
`SGLANG_FUSED_SM_FA3` now pins it. Matched sweep, 200 prompts, input 8192, H200:

| `sm_fa3` | tok/s | vs base | mean ITL | p99 ITL |
|---|---|---|---|---|
| 16 | 19369.38 | -22.4% | 23.77 | 45.46 |
| 32 | 22034.74 | -11.7% | 20.93 | 35.94 |
| **64** | **24055.80** | **-3.6%** | 19.18 | 34.74 |
| 80 | 23800.46 | -4.6% | 19.38 | 35.72 |
| 96 | 22856.25 | -8.4% | 20.26 | 37.05 |
| 112 | 19457.78 | -22.0% | 23.95 | 44.89 |
| base (fused off) | 24946.41 | — | 18.47 | 32.85 |

200/200 requests in every arm, 0 faults, fused share ~90.6% throughout. The curve
is sharply peaked at 64, and the old ceiling of 48 sat below it: **-8.1% -> -3.6%**.
`MAX_SM_FA3` is now 64 so the default lands on the measured optimum.

**The analytic balance point is not usable for this knob.** Unclamped,
`plan_dispatch` returns 115-131 for real serving batch shapes — nearly the whole
device on FA3 — and the sweep measures 112 as one of the two *worst* settings.
The clamp, not the model, was choosing the value. That cost model was fit for the
overlap kernel and does not transfer to this kernel's makespan.

### Tested: cross-role backfill on the server
`FLASHINFER_FUSED_BACKFILL=1` at `sm_fa3=64`: 24064.65 vs 24055.80 tok/s,
**+0.04% — within noise**, against 1.2% (H200) / 1.9-6.9% (H20) at kernel level.
Consistent rather than contradictory: backfill reclaims the prefill tail of one
mixed step, and at 32 concurrent requests the server already has other work to
overlap it with.

Correctness with backfill on is now clean under the server's own toolchain on the
dumped server batch shape, at splits 64/48/32: 0/512 unwritten, rel err
0.00028/0.00218/0.00282 (vs 0.00032/0.00245/0.00313 with it off). Scope: that is
one shape, and it is not the one the earlier backfill fault was recorded on, so it
does not retire that fault. The default stays off, now for a measured reason.

### Multicast: the precise blocker
Not a flag. The fused kernel's FA3 side uses `hopper/sparse_mainloop.cuh`, where
`USE_TMA_LOAD_KV = false` (line 87) — KV is gathered with `cp.async`, so there is
no TMA descriptor to multicast from — and the Q descriptor is built with `_1{}`
and the comment "no mcast for Q". Enabling it needs `page_size >= CTA_TILE_KV`
(128) threaded through the fused launch (currently hardcoded to 1) **and** the
sparse mainloop's gather replaced by a TMA load. The second is a rewrite, not
configuration.

### Still open
Best fused configuration is -3.6% against the persistent kernel, so
`SGLANG_FUSED_FA3_SM_SPLIT` stays 0 by default. Closing the last 3.6% needs
warpgroup-reentrant FA2; the split is now tuned and backfill is measured at zero.

## v0.7.1 — Split-KV and GQA Packing for FA3; Gate on Tile-Grid Quantities (2026-09-30)

### Measured
- **FA3 split-KV: 1.75x over FA3 as-is and 1.38x over the persistent kernel**
  (507 -> 290 us, S=2, one 1024x32768 row at 8 q heads). Width, not KV traffic,
  was FA3's problem, and it is a property of flashinfer's sm90 planner having no
  KV split rather than of wgmma. Emulated via batch expansion + `merge_states`.
- **GQA packing: 5.0x on the verify class** (2541 -> 511 us, 58 rows @ kv 11000),
  moving it from 0.23x to 1.13x against the persistent kernel at packed_qo 56;
  still 0.78x at packed_qo 32. Nothing on large prefill (509.8 vs 507.2 us).
- **CUDA graphs: the fused path is strictly better.** At fixed tokens and batch
  size, one captured fused graph replays correctly across p/d ratios 1/32 -> 8/25,
  while FA3's `batch_size`/`qo_indptr` length change with the ratio and are baked
  into the captured launch. Sequential needs a capture per (n_prefill, n_decode)
  pair. This reverses the earlier parity claim.

### Rewritten: mixed_dispatch gate
- Routes on the two tile-grid quantities the measurements identify:
  `packed_qo_len` vs `MIN_PACKED_QO = 48` (tile occupancy) and `kv_tiles` vs
  `MIN_KV_TILES = 32`. Work is counted in (q tile x kv tile) pairs.
- The two kernels get **disjoint SM sets** at the makespan balance point; FA3's
  KV-split S is the smallest that covers the SMs it was given (returns S=2 for
  the (8,1,128) chunk, the measured optimum).
- A class folds into the persistent kernel unless one SM's worth of it is at
  least half the makespan the rest of the device needs anyway -- no separate SM
  set that cannot be kept busy.
- Latency 2.15 us on decode-only batches (short-circuits on `max(qo)`) and
  13.8-14.7 us when an FA3 row is present, once per forward.

### Corrections
- The gqa-7 verify figure was first reported as 2.72x; it packed one GQA group
  while the persistent kernel processed all 4, so FA3 did a quarter of the work.
  Matched-work value is 1.13x.

## v0.7.0 — 4-Mode Dispatch: Separate Kernels, Scheduler-Partitioned Grids (2026-09-30)

### New: per-tile-class kernel dispatch with a grid-partitioning scheduler
- `python/sglang/srt/layers/attention/mixed_dispatch.py` (new): picks per batch
  between `fused` (one persistent launch), `seq` (FA3 then persistent) and `conc`
  (both co-launched at a split grid), and returns the grid split in SMs.
- flashinfer (`tri_tile_`): `FLASHINFER_FA3_MAX_CTAS` caps `PrefillSM90Plan`'s CTA
  count and `FLASHINFER_PERSISTENT_MAX_CLUSTERS` caps the holistic planner's grid,
  each rebalancing its own work over what it owns. Only the grid shrinks, so graph
  capture still sees one shape per bucket — green contexts were rejected because
  captures would multiply by the number of groups.
- `plan_4mode_dispatch` in `flashinfer_backend.py` now delegates to the measured
  scheduler; the unsupported `DISPATCH_MIN_MONOLITHIC_US` guard and the fixed
  `DISPATCH_CLASS_KERNEL` table are removed (both are contradicted by measurement).

### Results (held out on captured sets; see tasks/debug_kernel_server_perf.md)
- vs sequential (vLLM-style): **1.015x** overall, **1.054x** (h200 TP8) and
  **1.147x** (gb200 TP16); parity on the small-model long-prefix set, where the
  scheduler correctly picks `seq` and becomes the baseline.
- vs a fixed fused megakernel: **1.383x**. vs dual-stream: **1.000x** (no gain).
- Graph compatibility is **not** better: only 27 of 41 shape buckets resolve to a
  single mode, so parity (41 graphs) costs most of the gain and the full gain needs
  55 captures.

### Key measurements
- FA3's wgmma mainloop is **1.86x** the persistent kernel's per-CTA compute rate,
  but it reloads KV once per q head and its width is `h_qo * ceil(qo_len/128)` —
  only 64 CTAs at the (8,1) TP8 shard. At 8 q heads per rank the persistent kernel
  therefore **beats FA3 on pure prefill** (401us vs 507us at 256 KV tiles).
- Co-launch does overlap (up to **1.65x** at a 64/68 split), but over-provisioning
  the cooperative kernel stalls it behind FA3 (2286us at 88/44 vs 1470us).
- Boundary fitted on 50 cells, **50/50** correctly classified; within 5% of the
  per-batch oracle on 26/28 held-out batches.

### Doc hygiene
- `tasks/archive_h100_era.md` (new): 581 lines of H100/Qwen3-8B results, closed
  investigations and v3-scheduler notes moved out of `debug_kernel_server_perf.md`,
  which is reorganised into four story parts (gate -> workload -> kernels ->
  dispatch) and shrinks 2,184 -> ~1,540 lines. The superseded 4-mode *estimate*
  section collapses from 125 to 27 lines.
- Corrected the FA2/FA3/persistent planner table: it had given FA3
  `cta_tile_q = 64` and a `total_kv_lens/num_clusters` KV split, citing lines that
  are actually inside `TwoStageHolisticPlan` (our planner). FA3 is
  `PrefillSM90Plan`: `cta_tile_q = 128`, **no KV split**, no group-head fusion.

### Fixes
- Routing tested KV length alone and sent 58 spec-verify rows to FA3, **14x slower**:
  FA3 does not fuse group heads, so the test must be on `qo_len`, not `packed_qo_len`.
- Split accounting mixed units — the persistent planner sets `num_sm *= 2`, so a
  "96/36 split" subscribed 114 of 132 SMs. This invalidated four earlier `conc` wins.
- Out-of-bounds write in the per-task SM apportionment's remainder redistribution
  (`task_hi` could exceed `num_clusters`).

## v0.6.0 — Real-Workload Capture, Hopper Scheduling Study, Gate v3 (2026-09-19 → 09-22)

### Workload capture (new)
- `SGLANG_LOG_MIXED_COMPOSITION=<path>` records every mixed batch's real composition
  from the gate: per-row decode KV, chunk tokens, the **true attended prefix** (the
  scheduler's `#cached-token` is 0 for continued chunks), and the gate's decision.
  Wired into `run_mixed_workloads.py`, plus four `loadsweep_{hi,med,lo,min}` scenarios.
- Captured 3,493 mixed batches across a high→low co-tenancy load sweep. The realized
  shape: chunk size barely moves with load (p50 395–446) because it is the leftover
  token budget, while decode rows go 46 → 2 and the gate's persistent share 77% → 23%.
  Prefixes are fresh 73–96% of the time.
- **The sweep grid was badly mis-sampled**: 91.4% of real chunks fall in 128–512 tokens,
  a bin the grid covers with one point, while five of its seven chunk values sit where
  7% of traffic lands.

### Corrections (2026-09-23)
- The captured ~400-token "chunks" are **not chunks**: the distribution is bimodal and the
  small mode is the chat tenant's whole prompt (uniform[256,512], prefilled in one shot
  since 512 ≪ chunked_prefill_size=8192). It is load-invariant because it is a scenario
  parameter, not a leftover token budget; EDF was off (`fcfs`). The gate-v3 gain is
  therefore established for this workload's prompt-length mix, not in general.
- **Scope limit**: a decode row's `packed_qo_len` is `1 × gqa_ratio`, and the dual-tile
  premise only exists at ratio ≤ 16. At ratio 64–128 (MLA: Kimi-K2, DeepSeek-V3/V4)
  decode rows already sit in the large tile class — no heterogeneity to exploit and no
  16→128 flip for a chunk to cause — and those models use separate MLA backends that
  never call `select_mixed_attention_kernel`. The win region is low-GQA-ratio models
  (Llama/Qwen dense, 7–16) on the standard flashinfer backend.

### Kernel replay on real compositions (`sweep_from_capture.py`)
- Time-weighted over the captured distribution: FA2 1.00×, **FA3 0.67×**, persistent
  **2.60×**, deployed v2 gate 1.90×. FA3 loses to FA2 at every decode-row count
  (0.28–0.84×) — its fixed `cta_tile_q = 64` pads a GQA-7 decode row ninefold, which
  costs more than its better planner saves. Porting to FA3 is not the Hopper win.
- Gate decline diagnosis over 3,520 batches: 55.1% accepted, **27.0% declined by the
  out-of-support guard**, 17.9% non-unanimous — the grid mismatch, costing 14.2% of
  total FA2 kernel time.

### Gate v3 (deployed)
- Rebuilt `mixed_kernel_gate_h800_gqa7.json` with 60 measured serving compositions
  folded into the grid (1,376 points). On **held-out** buckets: accept 63.1% → 98.1%,
  speedup vs FA2 **2.17× → 3.50×**, 99.6% of oracle. A 1.61× kernel-policy gain with
  no kernel change.
- Serving validation (single run, `loadsweep_med`): live gate acceptance 34.1% → 89.3%,
  100% retokenization, **node throughput +0.0%** — the kernel-policy gain does not
  translate into node throughput. Chat tail improves (mean ITL −10.3%, P99 −43.1%);
  codex P99 +7.2% / TTFT +8.3%, inside the documented ±6% saturation run-to-run band,
  so neither direction is established without a paired repeat.
- Doing this at k=5 would have cost the zero-regression property (worst-case LOO
  0.971×); **k=7 restores worst-case LOO to exactly 1.000×** with no loss of held-out
  gain. `build_mixed_kernel_gate_table.py` gains `--k` / `--label`.

### Hopper instruction-level study (`tasks/debug_kernel_server_perf/wgmma/`)
- Five microbenchmarks, marginal cycles, validated at 96% of H800 peak bf16:
  wgmma shape mixing does **not** overlap within a warpgroup (32.00 + 20.00 = 52.00,
  order irrelevant) **or across warpgroups** on one SM (52.10) — the tensor core is a
  serial per-SM resource. exp2 does **not** hide under an in-flight wgmma (+50.5 cyc/8
  ops), but **hides 73% of the MMA when placed on a different warpgroup** (FA3's
  ping-pong premise, quantified). Small tiles absorb 8 FFMA ops at **zero** cost where
  large tiles pay 3.5 — cost is ≈ `16 + N/4`, so `n16` runs at 38% of peak.
- Closes one β hypothesis: the non-overlappable 0.53 is not the MMA pipe.
- Two measurement traps documented: ptxas C7515 (non-wgmma writes to accumulators)
  silently serializes the pipeline, and any runtime branch in the issue loop inflates
  the baseline 3× — both produced false "no overlap" readings in earlier drafts.

## v0.5.1 — Task-doc cleanup: invalidated results removed, story-order restructure (2026-08-13)

- `tasks/debug_kernel_server_perf.md`: deleted the sections marked superseded/
  invalidated (2026-07-24 serving results; Historical H800 WL1–6/EDF tables that
  compared contaminated DMC baselines). Reordered: current story (sync-free plan +
  gate + cost model + conclusion) and Open TODOs first; honest H100-era kernel/server
  results and closed investigations behind them as labeled history. Open TODOs
  rebuilt: EDF-analytic A/B marked done (multiturn_slo_clean), remaining suite
  scenarios added.
- `tasks/realistic_workload_analysis.md`: removed the superseded headline table,
  contaminated offered-load sweep, and pre-fix sweet-spot table; rewrote Scope
  (gate + LPT_EDF methods) and Final conclusions to the current story; added an
  evaluation-hygiene section (baseline contamination, warmup, queue-vs-service TTFT
  decomposition, rate=inf confound, episode defense) and mixed-workload repro
  commands.
- `tasks/edf_scheduling.md`: validity note (pre-fix DMC arms; policy conclusions
  stand, kernel deltas do not); task #117 closed by the measured gate.
- `tasks/server_perf_model.md`: status note — analytic model supersedes the MLP for
  chunk control; MLP remains for the router `/get_load` signal only.
- `tasks/README.md`: updated to the current document map.
- External-validity subsection for the gate sweeps (are isolated-kernel benchmarks
  representative of peak serving?): measured from the mixed-tenancy captures that exactly
  one kernel is resident 93.25% of serving wall (≥2 only 0.36%), so the bench's
  one-kernel-alone condition is the serving condition; serving kernel times (BP
  0.007–16.6 ms, persistent 0.055–2.26 ms) sit inside the swept envelope (BP 0.019–28.6 ms,
  PA 0.019–25.7 ms, decode KV 2K–6.3M); and BP's p90 kernel time is 0.267 ms in the gated
  arm vs 0.906 ms in DMC, i.e. the predicted routing is visible end to end. Documents the
  gaps too — contiguous `arange` KV pages vs a fragmented radix allocator, sustained-load
  clocks, flushed vs dirty L2, graph replay, shared GPUs.
- New WGMMA microbenchmark (`tasks/debug_kernel_server_perf/wgmma/`) answering whether
  different tile sizes overlap or get reordered: they do not. Marginal cost is exactly
  additive (m64n64k16 32.00 + m64n16k16 20.00 = 52.00 cycles/iter), issue order is
  irrelevant, and two warpgroups on one SM give 52.10 — the tensor-core pipe is a serial
  per-SM resource. Validated at 96% of H800 peak bf16 for the large tile. Closes one
  hypothesis for the β TODO (overlap cannot come from MMA-with-MMA, only MMA-with-memory)
  and quantifies a 2.5× per-unit-work penalty for small query tiles (cost ≈ 16 + N/4
  cycles, so n16 runs at 38% of peak).
- Mechanism arithmetic re-verified against flashinfer source, fixing one error: the
  persistent planner's split was documented as `kv_len_limit ~ total/264`, but 264 is
  fa2's `max_grid_size` (2 × 132 SMs) — the persistent side computes
  `f(total_kv_lens × num_kv_heads / num_clusters)` with `num_clusters = num_sm = 132`,
  i.e. ≈ total/33 here, halved again for the prefill tile class. The two BatchPrefill
  numbers turn out exact rather than approximate for this config: the split budget is
  exactly 66 units/KV head (`max_grid_size / num_kv_heads` = 264/4) and a 1,024-token
  chunk's q-tiles are exactly 56 of them (`ceil(1024 × 7 / 128)`), so splitting the eight
  decode rows two ways (+16) would exceed the budget and the search stops splitting
  entirely. Also notes that the tile decision keys on the batch *mean* only in the
  non-graph path (what the sweeps use) and on `max_qo_len` under CUDA graphs, and flags
  that the 0.366 ms decode-only datapoint is not reproducible from saved artifacts.
- **Corrected an overstatement**: "BatchPrefill's cost surface has cliffs no smooth model
  can track" was too strong. New `fit_learned_cost_model.py` fits MLP and GBDT to both
  kernels on the same 1,316 sweep compositions, features and model class (5-fold CV × 3
  seeds): an MLP reaches 9.9% MAPE on BatchPrefill, beating the 13.3% of the analytic
  parts-based fit, so part of that gap was the functional form. What survives is the
  relative claim — BP is 1.7–3× harder than the persistent kernel (GBDT 10.1% vs 3.3%,
  worst case 103% vs 63%) — plus two practical points: adding fa2's own tile-decision
  variables recovers only 10.1%→8.6%, consistent with the residual living in the
  split-budget search over the per-row KV vector; and fitting BP needs ~1,300 measured
  compositions per GPU versus 35 for the analytic law, reintroducing exactly the
  profiling campaign it removes. Docstring in `overlap_cost_model.py` updated to match.
- Recorded why the kernel layer exists when only `predict()` is consumed: it is the
  interior of the one model in use, not a second model, and it is what makes the model
  training-free (35 kernel measurements vs the replaced MLP's 25,444 profiled configs),
  what makes the fused-vs-split predictability contrast visible at all (the step-level
  `c0 + c1·tokens` term masks it), and what an overlap-aware chunk controller would need.
  That controller is now an Open TODO rather than only prose.
- Clarified that the analytic cost model is **kernel-level** (its per-layer attention law
  is fit on 35 kernel measurements: T_d 2.04% MAPE, T_p 3.47%, composition law 3.10%
  in-sample and newly 3.22%/9.7% leave-one-out, so the one-parameter law is not overfit);
  only the `c0 + c1·tokens` step wrapper is server-level, and those two constants are
  derived from hardware specs rather than fit, with a single serving anchor. Records the
  two scope limits: persistent kernel only (no BatchPrefill equivalent by design), and a
  fit grid that sits above the realized operating point of the SLO runs it drives.
- Batch-mixture representativeness measured from the gated runs' scheduler logs and
  recorded as a known weakness: the sweeps are a Cartesian grid, but realized mixed
  batches form a narrow correlated cluster per workload (win scenario: 386-token chunks
  p90 532 against 46 decode rows @189K KV; TraceLab: 8,182-token chunks against 10 rows
  @904K). Chunk size and decode count are anti-correlated in serving because the chunk is
  the leftover budget, so the product grid spends points on combinations the scheduler
  cannot emit, and samples the actual operating region with two points. Notes that
  leave-one-out 1.000 is therefore generalization within the grid, not to serving.
  New TODO: rebuild the table by importance-sampling logged compositions.
- Removed the "DMC baselines were silently persistent" post-mortem now that every result
  it invalidated is gone from the docs; its one live invariant (baseline runners must set
  `SGLANG_FLASHINFER_USE_PERSISTENT=0` — it defaults on) is promoted to Setup as a rule.
- Cost-surface and win-region sections unified into one: adds a sweep-provenance table
  (479 uniform + 477 fresh → 956 merged, + 360 blended = the deployed 1,316 points) with
  every file checked in, and shows where the headline multiple comes from — at a fixed
  fresh 2K chunk, BatchPrefill is flat (2.97 → 5.05 ms) while the decode pool grows 64×
  and the persistent kernel scales with the work (0.105 → 4.274 ms), so the speedup runs
  28.2× → 1.2×. Corrects "up to 20×" to the per-sweep maxima (21.0× uniform, 28.2×
  merged) and records the loss region from the same data (162/956 below 1.0×, worst
  0.57× at 1-4 tiny rows). New data files: `gate_sweep_blend_qwen25_h800.json` (was only
  in an ephemeral scratchpad, though it is 360 of the deployed table's points) and
  `probe_fresh_qwen25_h800.json`.
- Profiling post-mortem added to the overhead diagnosis, mined from the mixed-tenancy
  nsys captures. It leads with the conclusion — **our plan path does no D2H at all**
  (host tensors only; verified by `cudaDeviceSynchronize` appearing once in 279 s, so
  neither flashinfer's device-tensor branch nor the `seq_lens_cpu is None` fallback ever
  fires) — and presents the rest as the mechanism behind the old 56.5 ms, measured on
  the copies that remain elsewhere in the stack. Findings: `non_blocking=True` is not
  honored for D2H into pageable memory, and the
  cost is a pipeline drain rather than a copy. Within back-to-back groups of pageable
  D2H copies the **first** costs 5–8 ms (median) and the rest ~14 µs — 560–800× apart
  at identical size and destination — so the old plan() stall was paid by the first
  `.to("cpu")`, not by the following `torch.cuda.synchronize()`. Steady-state window
  analysis shows the host blocked 74.6% of wall in both arms while the GPU stays ~94%
  busy (GPU-idle-while-host-blocked 0.64% of wall), so such readbacks hurt only when
  they remove run-ahead the GPU needs. Two live optimizations filed: flashinfer's stock
  `plan()` still does pageable D2H for the BatchPrefill path (baseline pays it every
  mixed step), and sglang's `fast_decode_plan` remains shadowed by the class-level
  `begin_forward = plan` alias.

## v0.5.0 — Mixed-Tenancy Win, Overlap Attribution, Analytic Cost Model (2026-08-04 → 2026-08-06)

### Mixed-tenancy workloads (`run_mixed_workloads.py`)
- New multi-tenant suite (coding agent + chat on one node, plus long-CoT / RAG / control
  scenarios). `agent_plus_shortchat` vs honest BatchPrefill baseline: coding tenant
  **+36.5% throughput**, chat tenant **−42% mean ITL / −23% P99 / −37% TTFT**;
  `shortchat_only` control at parity with the gate declining persistent (6/2,566).
- One −19% LPT trial archived as `run1_regressed/` (storm-phase retraction-cascade
  signature); unreproduced across repeat + nsys trials with a stable baseline.

### Gate v2 (KV concentration + out-of-support guard)
- Root cause of the win region: few long-context decode rows starve BatchPrefill of
  query tiles (8×64K = 8.2x vs 128×4K = 1.02x at equal aggregate KV); one chunk flips
  its CTA tile and exhausts the KV-split budget (3.125 ms plateau for any chunk ≥1K).
- Two k-NN blind spots fixed (fresh-prefix chunks; blended pools) via 837 new sampled
  compositions and a KV-concentration feature (Σkv²/Σkv). Merged table 1,316 points,
  k=5 unanimous, label ≥1.10: leave-one-out worst regression 1.000, capture 81%.

### Overlap attribution + analytic cost model
- Same-kernel ablation: temporal PD overlap contributes up to **1.26×** in the balanced
  band (T_p ≈ T_d); heterogeneity-robust scheduling supplies the rest.
- `SGLANG_EDF_LATENCY_MODEL=analytic`: `T_attn = max + 0.53·min` (3.1% MAPE / 9% max,
  7 physical constants, no training) drops into the EDF chunk controller; BP's best
  parts-based fit is 13.3% / 43% with fitted β=2.19 (super-additive) — the fused kernel
  is what makes mixed-step cost predictable enough for SLO-aware chunking.

### Docs
- Superseded/contaminated 2026-07-24 serving tables flagged in place; conclusions and
  TODOs consolidated to current results.

## v0.4.0 — Sync-Free Persistent Plan + Per-Batch Mixed Kernel Gate (2026-07-30)

### Validation (TraceLab decode-heavy N=100 screens + WL1 Qwen3-8B, honest baselines)
- 10 sessions (old +14.5% pooled-P99 regression point): parity on every axis across paired
  runs at the matched cap; pooled P99 ITL −0.7% to −6.2% better than true DMC.
- 16 sessions: +0.8% mean throughput across paired runs, pooled P99 ITL −1.3% to −4.1%,
  request-max P99 up to −23%; cap 6656 option reaches −14.6% pooled P99 ITL / +0.77% tput.
- WL1 (Qwen3-8B, rates 8 and 2): parity or slightly better on all metrics; 100%
  retokenization on every accepted run.
- Startup prewarm of both persistent module variants added after a mid-replay JIT build
  froze all nine sessions for 18.3 s in the first g10 run (flashinfer `c6d2a00`,
  sglang `313b97611`).

### Sync-free BatchAttention planning (flashinfer `f4d2a80` + backend host metadata)
- `BatchAttention.plan()` no longer calls `torch.cuda.synchronize()`. The backend now
  builds `qo_indptr`/`kv_indptr`/`kv_len` on the host from `seq_lens_cpu`/`extend_prefix_lens_cpu`
  (NumPy cumsum) and passes CPU tensors straight to `plan()`; flashinfer double-buffers its
  pinned int-workspace staging with CUDA-event fencing so removing the sync is race-free.
  Measured: with ~40ms of queued GPU work, plan wall time drops 56.5ms → 0.47ms. This was the
  main serving overhead of the persistent kernel (it broke overlap-scheduler CPU run-ahead on
  every mixed step); previous naive removal attempts hung because of the pinned-buffer race.
- `persistent_attn_forward` reuses grown `out`/`lse` buffers instead of allocating two tensors
  per layer per step.
- Removed a dead `call_begin_forward` branch that would have routed BatchPrefill wrappers
  into a BatchAttention-signature `plan()` call when `use_persistent=1`.

### Per-batch mixed-attention kernel gate (scaffolding)
- New env `SGLANG_MIXED_ATTENTION_KERNEL` = `auto` (default) / `persistent` / `batch_prefill`.
  For each mixed batch, `select_mixed_attention_kernel()` chooses persistent BatchAttention or
  BatchPrefill from CPU-side batch composition; forward dispatch now keys on the planned
  wrapper type instead of the global env flag. Cost-model constants pending the H800 gate
  sweep (see tasks/debug_kernel_server_perf.md).

### DMC baselines were silently persistent
- `SGLANG_FLASHINFER_USE_PERSISTENT` defaults to 1 and no DMC launcher overrode it, so every
  H800 "DMC_CG" baseline since 2025-11-14 ran the persistent kernel for mixed batches. July
  TraceLab LPT-vs-DMC deltas compared chunk caps/EDF guards on the same kernel. Both
  `run_gain_search.sh` and `run_edf_sweep.sh` now set `SGLANG_FLASHINFER_USE_PERSISTENT=0`
  for baseline methods.

### H800 kernel measurements (serving-matched non-causal prefix composition, Qwen2.5-7B heads)
- Persistent kernel run() CPU dispatch is cheaper than BatchPrefill (16µs vs 26µs) —
  cooperative launch is not a CPU-side cost.
- 479-case sweep: persistent wins up to 20× at decode-dominant compositions but loses up to
  0.55× when decode work is tiny and the chunk is compute-bound; 87/186 raw losses are
  recovered by choosing the right flipped schedule (big chunks + long prefix want non-flipped).

## v0.3.3 — Nebius-SWE Workload + Chunk-Control Fall-back Fix (2026-04-09)

### Heavier Benchmark Workload
- Added `nebius-swe-traj` and `swe-bench-pro` datasets to `bench_serving_3rdparty.py` and the bench_serving CLI choices.
- `nebius-swe-traj` loads `nebius/SWE-agent-trajectories` (multi-turn SWE agent traces) via streaming, samples a random AI-turn boundary per trajectory. Median input ~9.7K tokens — the only prefill-heavy real-world dataset surveyed in `tasks/realistic_workload_analysis.md`. Code-Feedback (200-token median input) was too lightweight for chunk control to matter; nebius is the right stress workload.
- `swe-bench-pro` loads `ScaleAI/SWE-bench_Pro` test split — 731 single-step Q/A pairs, longer than swe-gym but still single-shot, kept available but not the headline workload.

### Chunk-Control Fall-back Fix (`edf_scheduler.py`)
- **Bug**: when any decode req's slack went negative (`min_slack_ms ≤ 0`), `edf_compute_max_chunk_size` fell back to the **full** `default_chunk_size` (8192). At saturation this triggered immediately, so chunk control was effectively disabled — first nebius EDF run gave P99 ITL = 297ms vs 297ms baseline (zero improvement).
- **Fix**: throttle to `MIN_CHUNK = 512` instead of falling back to default. Already-violated requests can't be saved by relaxing the chunk; the not-yet-violated ones can be, by capping added prefill latency to ~10-20ms.
- **Impact** on nebius-swe-traj 1k sweep: DMC/LPT_CG P99 ITL drops from ~290ms → ~135ms across all rates; per-req P99 ITL ≤ 150ms attainment goes from 0.6-22% to **85-100%**. Throughput rises 1212→1290 tok/s (faster KV drainage = higher effective concurrency).

### New SLO Metric (`analyze_results.py`)
- Added `itl_p99_attain_pct`: per-request P99 ITL ≤ ITL_SLO (Sarathi-Serve-style). Strict max-ITL gives ≈0% on long-decode workloads (any single outlier kills the score, masking real differences); per-req P99 filters single-step noise but still penalizes consistent prefill-blocking. TPOT (mean ITL) is too lenient (>95% on every run).
- Added `nebius_swe` to the per-workload SLO table (5s TTFT / 150ms ITL — same as SWE-Gym).

### New Sweep Scripts + Pareto Plot
- `tasks/edf_benchmark/run_nebius_baseline_1k.sh` and `run_nebius_edf_1k.sh`: 1000 prompts, rates 1/2/4/8/12/inf (lower than other workloads because 9K-token prefills are 30× heavier than ShareGPT), output fixed to 1024 tok via `--sharegpt-output-len`. Both scripts toggle `gpu_keepalive` per benchmark per project rules.
- `tasks/edf_benchmark/plot_pareto_nebius.py`: side-by-side baseline median ITL + EDF SLO attainment plot. Output: `tasks/edf_benchmark/pareto_nebius.png`.

### Findings
- Chunk control eliminates the historical PP advantage on decode tail latency on prefill-heavy workloads. DMC/LPT_CG now match PP at 91-100% vs 95-100% per-req P99 ITL attainment.
- TTFT remains the binding constraint above rate 2 — saturation queues are fundamental and no scheduling policy can hide them. EDF + chunk control protects the decode SLO at the cost of TTFT for late requests, which is the correct trade-off for interactive serving.

## v0.3.2 — 1k-prompt EDF Sweep + schedule_policy Admission Fix (2026-04-07)

### schedule_policy Bug Fix (overlap scheduler crash root cause)
- `add_chunked_req` and `add_one_req` could admit a chunked-prefill req with `extend_input_len ≤ 0` (full prefix hit on a freshly chunked req). In MIXED mode the sampler then produced N-1 sampled tokens for a batch of N, crashing the overlap scheduler with `RuntimeError: expanded size (N) must match existing size (N-1)`.
- Fix: when `extend_input_len ≤ 0`, drop one prefix block (`prefix_indices[:-1]`), recompute `extend_input_len`, and refuse admission if still ≤ 0. This is a real root-cause fix — no `--disable-overlap-schedule` workaround.
- Files: `python/sglang/srt/managers/schedule_policy.py` (`add_chunked_req`, `add_one_req`).

### 1000-Prompt EDF Sweep with 6 Pareto Curves
- Re-ran the full 3 methods × 3 workloads × 6 rates (8/16/24/32/48/inf) sweep at **1000 prompts** per workload, both baseline (FCFS) and EDF + SLO-aware chunk control.
- Baseline tables now report median TTFT and median ITL alongside P99 (per user request).
- Switched the EDF Pareto y-axis from TPOT (mean ITL ≤ SLO) to **strict max-ITL ≤ SLO**: TPOT stays > 99% across every run because median ITL is 25-30ms vs 100-150ms SLO, masking real differentiation. Strict max-ITL is what chunk control actually protects.
- 6 Pareto curves: `tasks/edf_benchmark/pareto_baseline_median_itl.png`, `tasks/edf_benchmark/pareto_edf_slo_attainment.png`.
- Key result: EDF + chunk control gives 2-10× tail-ITL SLO attainment improvement for DMC/LPT_CG vs PP at saturation on ShareGPT and SWE-Gym.
- Scripts: `tasks/edf_benchmark/run_baseline_1k.sh`, `run_edf_1k.sh`, `analyze_results.py`, `plot_pareto_6.py`.
- Results: `tasks/edf_benchmark/results_baseline_1k/`, `results_edf_1k/`.

### Project Rule
- Added to `CLAUDE.md`: **Never compromise by falling back silently** — if a feature crashes, find the root cause; don't disable it to work around the problem.

## v0.3.1 — EDF Benchmark Sweep + Chunk Control Fix (2026-04-07)

### EDF Chunk Control Bug Fix
- `edf_compute_max_chunk_size` was using `req.last_tic` as previous decode time, but `last_tic` is only updated when `enable_metrics=True`. With metrics off it equals request creation time, slack went negative, prefills starved (smoke test: 579 vs 2135 tok/s).
- Fix: track `_edf_last_step_time` per req, updated in `process_batch_result_decode`. New reqs fall back to `now` (full ITL_SLO budget).

### Full EDF Benchmark Sweep (3 methods × 3 workloads × 5 rates)
- Ran PP / DMC / LPT_CG with EDF priority + SLO-aware chunk control on ShareGPT, Code-Feedback, SWE-Gym at rates 4/8/12/16/24
- Per-workload SLOs: ShareGPT (2s/100ms), Code-Feedback (3s/150ms), SWE-Gym (5s/150ms)
- TTFT attainment is 100% across all 45 runs — ITL is the binding SLO
- PP dominates Code-Feedback (100% at rate ≥ 12); mixed batches (DMC/LPT_CG) win on ShareGPT at saturation
- Generated two Pareto curves: throughput vs ITL P99, and SLO attainment vs request rate
- Scripts: `tasks/edf_benchmark/run_edf_sweep.sh`, `analyze_results.py`, `plot_pareto.py`
- Results in `tasks/edf_benchmark/results_edf/`

## v0.3.0 — EDF Scheduling + SLO-Aware Chunk Control (2026-04-06)

### EDF Scheduling
- Implemented Earliest Deadline First scheduling: priority = TTFT deadline (arrival + SLO)
- TTFT SLO scales with input length (base + extra for >4K tokens)
- Enable via `SGLANG_EDF_SCHEDULING=1` + `--enable-priority-scheduling --schedule-low-priority-values-first`

### SLO-Aware Chunk Size Control
- Binary search for max prefill chunk size that fits within ITL SLO slack
- Uses trained MLP latency model to predict server step latency
- Configured via `SGLANG_EDF_LATENCY_MODEL=path/to/model.pt`
- Fixed clock mismatch: all EDF code uses `time.monotonic()` consistently

### Latency Prediction Models
- Retrained both LPT and DMC server-level latency models (25K+ configs each)
- LPT model: 2.4% MAPE, 3.26ms MAE (tasks/perf_model/lpt_model.pt)
- DMC model: 2.3% MAPE, 3.04ms MAE (tasks/perf_model/dmc_model.pt)

### Benchmark Status
- Created benchmark scripts for PP vs LPT_CG vs DMC with EDF on 3 workloads
- Blocked by pre-existing `req_to_token_pool memory leak` bug (available_size > total_size)

## v0.2.1 — CG25: data-driven mixed CG size optimization (2026-04-01)

### Mixed CG Size Optimization
- Instrumented mixed CG replay to collect (raw_tokens, padded_tokens) per workload
- Greedy selection of 6 additional capture sizes based on real padding waste: +3584 (28% of waste), +2816 (7%), +2304 (5%), +640, +1408, +1920
- CG25 (25 sizes, ~1.79GB) reduces padding from 11% to 6.5%, improves all 6 workloads +0.3-1.4% over CG19
- LPT+CG25 beats trueDMC on all workloads: +18.9% (WL1), +3.6-8.0% (WL2-4), +1.9-2.1% (WL5/6)
- Updated task MD: replaced v3 with LPT in tables, updated stale CG12-era results with CG25 data

## v0.2.0 — Mixed Batch CUDA Graphs + LPT + Server Benchmarks (2026-03-27 → 2026-03-29)

### Mixed Batch CUDA Graph Support
- Implemented mixed batch (prefill+decode) CUDA graph capture and replay for persistent kernel (BatchAttention)
- LPT (dynamic scheduler) compatible with CUDA graphs — activates at runtime when plan() detects mixed batch
- 6 bugs fixed: missing kv_indices copy, plan_info[-1] wrong field, module auto-switch breaking CG, padding overflow in replay, shared indptr buffer too small, LPT pointer initialization for CG

### Server Benchmarks (H100 80GB, Qwen3-8B)
- v3+LPT+CG beats prefill-prio by +41-87% throughput on long prefill (12k-16k input)
- HOL blocking crossover at ~8k input tokens for prefill-prio (max ITL 21-25s vs v3's 230ms)
- v3 matches default-mc within ±0.1% — plan() overhead offsets CG savings

### LPT Load-Balance Profiling
- Grid.sync() wait reduction: -15% to -39% max wait on KV-splitting workloads
- Kernel speedup: 7-16% on mixed workloads

### Overlap Profiler
- Fixed shape mismatch for persistent v3 compatibility (disable CG during profiling)

## v0.1.0 — Persistent Kernel v3 + Decode Fix (2026-03-26)

### Decode Overhead Fix
- Decode steps use regular BatchDecodeWithPagedKVCacheWrapper instead of BatchAttention
- Eliminated 0.5-4% throughput regression vs default-mc
