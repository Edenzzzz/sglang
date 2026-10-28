# Current goal

Fused FA3 + FA2 attention for the 4-mode MTP (EAGLE3) serving schedule, built to the user's tile rule and optimised on
real serving steps, then the EDF step model and the PD-colocation argument (`pd_colocation_vs_disagg.md`).
- **Bar:** >= 30% on TTFT and ITL (or their SLO attainment) and >= 10% throughput on realistic chat + TraceLab
  workloads.
- **Models:** Qwen2.5-7B, Llama-3.3-70B (next), GLM-4.5-Air.
- **Multi-node H20 cluster (2026-10-28): `8d89d1880003f89d`, named `vibecuda`**, forked from the finished
  `2f608c1b1a67cf09` with `overrides={"name":"vibecuda","attachments":[]}` (the fork fails without the explicit
  empty attachments array). Same 4 pods x 4 H20 configuration, within the user's <= 16 GPU limit.
  - The previous job ended at 06:29 CST on 2026-10-28 (`status: done`, `trial_status: finished`), taking its
    `/dev/shm/w` with it. **Note for diagnosis: when pods become unreachable and merlin-cli reports "webshell relay
    redirected to SSO", CHECK THE JOB STATUS FIRST** -- a finished job produces exactly that error, and the login
    token may be perfectly valid. The ssh proxy gives the honest message ("job run ... is not running").
  - **Re-provisioning done (2026-10-28 08:00).** Stage 1 (`mn_provision_20261016.sh`, all 4 pods in parallel):
    venv, torch, sgl_kernel, flashinfer, Qwen2.5-7B (4 shards), the EAGLE3 head, ShareGPT, TraceLab. Stage 2 (pushed
    with `Cluster.push`, base64 over exec, ~8 MB of base64 at 60 KB a chunk): our serving tree from the repo
    (`/dev/shm/w/ourpy/python`, symlinked as `python`), `port07` (fused-kernel sources), `drivers/serving`, the step
    model and scorers under `perf/`, and `python_slo` = our tree + `apply_slo_budget_chunk_20261026.py`.
    - Verified on all four pods: `python=1 up=1 slo=1 model=1 scorer=1`, and the chunk patch carries the corrected
      defaults (`bank=` default `"inf"`, `:ttft` supported). `up/python` is the untouched upstream tree, which is
      what the BASELINE arm must use.
    - `port07` holds only kernel sources (include/, tests/), not a Python package -- flashinfer itself comes from
      pip, so a missing `$W/fi/flashinfer` is expected and not an error.
    - The five A6 kernel libraries had to come from the local bundle `/tmp/a6_bundle_20261025/a6_bundle.tgz`
      (built for H20 sm90 on the previous cluster); they are not rebuildable in reasonable time on a fresh pod.
      `mn_node.sh KERNEL=a6` expects them at `/opt/tiger/w/a6lib` with the python shims in `$W/a6`.
    - **If the A6 libraries ever cannot be restored, the corrected chunk/EDF arm set should still run with the
      kernel held CONSTANT at `KERNEL=lpt` on both sides.** That is in fact the cleaner experiment for C3: C1
      measures 1.4% at cluster level (inside the noise floor), so holding the kernel fixed isolates the schedule and
      chunk contribution without confounding it, and avoids attributing a scheduling gain to the kernel.
  - (superseded) Re-provisioning needed on the new job before any arm: the serving tree, workloads, models, the SLO-chunk tree
    (`/dev/shm/w/python_slo`) and the step model, plus `tasks/pd_colocation/` with today's `mn_node.sh`
    (`ROPE_THETA`) and `mn_run.py` (`ROPE_THETA` in the env string).
  - Push files with `Cluster(SID, cp, nodes=[f"worker-{i}" for i in range(NPODS)])`; without `nodes=` every call
    goes to one default instance that is not even one of the workers.
- **Hardware (2026-10-26, user):** ONE 4-GPU H200 trial for the kernel and single-node stages, `bf7d041fb8313d8a`
  (vibecuda, trial 304747380; replaced the misnamed vibecuda4 `d4b20cad59170ebd` on 2026-10-27 after 6.5 h queued; forked from `5b256eb857aa98d9` with gpu=4 / cpu 96 / 876544 MB and
  `overrides.attachments=[]`). The queued 1-GPU fork `5b256eb857aa98d9` was stopped. Llama-3.3-70B and GLM-4.5-Air TP4
  single-node runs move here from H20. If the trial is killed for any reason other than underutilization (e.g. a
  high-priority preemption), fall back to the H20 pods.
  - Re-provision: `unified/prov_h200_20261024.sh` from `/tmp/h200r2/bundle.tar.gz`.
  - Lost with the earlier trial `8574c727fab6ffbd`, to be regenerated: the 96 trace-sampled captures, the synthetic
    and standalone sets, every A6 build, and experiment 4 (sync-free merge), whose result was never read.
- **Kernel claim baseline (user, 2026-10-26):** gains are measured on 4-mode inputs against NAIVE FA3 + FA2: stock
  flash-attention FA3 on the prefill rows plus STOCK flashinfer FA2 (`BatchPrefillWithPagedKVCacheWrapper`,
  backend fa2, its own plan / tile choice / split-KV merge), serial and two-stream
  (`unified/bench_naive_fa3_fa2_20261026.py`, run with the pip flashinfer 0.7.0.post1, not port07). The v3 build is
  our own earlier design and must not be the baseline. Microbenchmarks and single-node runs on the H200 only (the H20
  is compute-curtailed). Plan on the re-forked H200:
  - Qwen captures are regenerated.
  - Llama / GLM TP4 rank-0 captures are replayed from the H20 copies.
  - Arms on the same captures: sglang's single FA3 call, naive FA3 + FA2 (serial / streams), v3, A6.
- **To test alongside the naive FA3 + FA2 baseline (user, 2026-10-26), on the H200:**
  - [ ] **Router / local-scheduler split (user, 2026-10-27): router `cache_slo_h` policy.**
    - Problem: the measured router (`cache_slo`) applies four hard vetoes joined by AND: own TTFT, own TPOT,
      later-deadline queued requests, resident TPOT. It overlaps with the worker. A resident-TPOT veto priced from
      coarse router bookkeeping mostly moves requests off their cached prefix: earlier it bought -32% chat ITL for +26%
      agent TTFT.
    - Design: the worker enforces resident SLOs with the SLO-budgeted chunk; the router places.
      1. TTFT priced under the worker's own budget rule, by simulating its chunk choice with the workers' step model
         (`--budget-chunk`);
      2. cache hit;
      3. lateness added to later-deadline queued requests (ranking, not a veto);
      4. load.
    - No feasible node: least total lateness, cache breaking ties. The router never holds a request, since the worker's
      EDF queue and demotion already do.
    - Implemented: `pd_colocation/router.py` (`--policy cache_slo_h`, `build_parser()` factored out for tests).
    - Unit checks: `test_router_slo_h.py`, 7 / 7 pass; `test_router_cond_disagg.py` passes.
      `test_router_dead_worker.py` and `test_router_loaded_prefill.py` fail identically on the pre-change router (stale
      tests).
    - Finding from the simulation: under the worker's budget rule a cache holder is rarely too slow. Rows at their
      target keep earning about s x a of budget per step, and rows past it are demoted, so moving a request pays mainly
      for long uncached prompts or deep EDF queues.
    - [ ] A/B on the trace replay, alpha 3 and 6, with SLO-budgeted workers: `cache_slo` (four vetoes, most-slack
          fallback) vs `cache_slo_h`. This waits for the user's go-ahead and for the chunk-size results.
  - [x] **Single-node A/B on the H200 (2026-10-28): our full stack vs stock sglang, 32K stage.**
    `unified/a6_c1c2_singlenode_h200_20261028.sh` on trial `bf7d041fb8313d8a`, log `/opt/tiger/w/c1c2_singlenode.log`;
    Qwen2.5-7B TP1, agent C=8 + chat 8 req/s, two repeats per arm in opposite order, A6 composition counter 1144
    (non-zero, so the fused path really ran).
    | arm (mean of 2 repeats) | TPOT ms | ITL ms | worst-decile TPOT ms | tok/s |
    |---|---:|---:|---:|---:|
    | **32K stage** | | | | |
    | our full stack | **14.3** | **15.0** | **32.4** | 137.5 |
    | stock MTP + stock FA3 (control) | 22.9 | 20.6 | 75.9 | 138.5 |
    | no spec decoding + A6 | 23.7 | 22.9 | 72.5 | 140.5 |
    | no spec decoding + stock FA3 | 25.4 | 25.5 | 77.9 | 131.9 |
    | **128K stage** | | | | |
    | our full stack | **44.8** | **34.4** | **95.7** | **54.4** |
    | stock MTP + stock FA3 (control) | 97.3 | 61.0 | 231.4 | 48.8 |
    | no spec decoding + A6 | 86.3 | 67.9 | 218.5 | 53.8 |
    | no spec decoding + stock FA3 | 88.8 | 70.2 | 218.1 | 49.0 |
    - Ours vs the stock control: **-38% TPOT / -27% ITL / -57% tail at 32K (equal throughput), and -54% TPOT /
      -44% ITL / -59% tail / +11% throughput at 128K.** The 128K point clears BOTH publishability bars at once, and
      the advantage grows with context. A6 composition counters 1144 (32K) and 2478 (128K) confirm the fused path ran.
    - The no-speculation rows isolate the kernel with no MTP and no worker scheduler: -6.7% TPOT at 32K, -2.8% at
      128K, with +7-10% throughput. Small, and shrinking with context, exactly as the capture ladder and the
      cluster step-time analysis predict.
    - [x] **SINGLE-NODE ATTRIBUTION, complete (2026-10-28).** Combining the `--ours` and no-`--ours` passes gives a
      cumulative decomposition on one tree, two repeats per arm, fused path verified (composition 1091 / 2333):
      | arm, added cumulatively | 32K TPOT | step | 128K TPOT | step |
      |---|---:|---|---:|---|
      | stock MTP + stock FA3, FCFS | 22.9 ms | - | 97.3 ms | - |
      | + four-mode step (C2) | 14.2 | **-38%** | 48.0 | **-51%** |
      | + A6 unified kernel (C1) | 14.2 | 0% | 41.3 | **-14%** |
      | + EDF and chunk preemption (C3-worker) | 14.3 | ~0% | 44.8 | **+8.5% WORSE** |
      - **C2 carries the single-node result**, as the gate analysis predicts: these stages carry 32K-128K uncached
        prefill, far above the 8192 gate, so the four-mode step fires on every chunk.
      - **C1 is 0% at 32K and at most -14% at 128K**, and that -14% is **confounded**: the A6 arm also shows higher
        draft acceptance at 128K (3.27 against 2.94 tokens per verify, +11%), while at 32K acceptance matches
        (3.53 / 3.54). Acceptance feeds straight into TPOT, so part of the 128K gain is numerics, not kernel speed
        (FP32 worst relative error 0.0083 is enough to flip marginal draft tokens). Do not quote -14% as pure kernel.
      - **C3-worker costs 8.5% of TPOT at 128K** on one node, matching the cluster result that chunk preemption /
        parking costs goodput. The worker scheduler should be presented as useful only under overload (alpha 6:
        +7.5% goodput), not as a general win.
      - Throughput: C1 alone +6.9% at 32K and +8.4% at 128K (the latter carrying the same acceptance confound).
    - **CORRECTION (same day): the -38% bundles THREE contributions, not two.** With `--ours` the native arm also
      enables the worker scheduler -- `SGLANG_EDF_SCHEDULING`, uncached-token deadlines, `SGLANG_EDF_PREEMPT_CHUNK=1`
      (chunk preemption / parking) and deadline-sized chunks (run_native_serving_pair_20261023.py:86-95) -- while the
      control is plain stock sglang. So the row is **our full single-node stack vs stock sglang (C1 + C2 +
      C3-worker)**, and it includes the preemption that costs ~25% of goodput at cluster scale. It is a valid
      end-to-end stage-2 headline and attributes nothing.
      - To separate them: `unified/a6_noours_singlenode_h200_20261028.sh` repeats both stages WITHOUT `--ours`, so
        both sides run sglang's FCFS admission and the pass measures **C1 + C2 alone**; the difference against this
        pass then measures **C3-worker** on one node. Queued behind the mix128 stage on the same GPU lock.
    - **This does NOT separate C1 from C2 either:** the harness's `four`-mode control removes BOTH (stock MTP and the
      stock FA3 call), and its `two` mode disables the draft model entirely (`draft_model=... if mode == "four" else None`,
      run_native_serving_pair_20261023.py:128), so the `two` rows are a no-speculation baseline, not "stock MTP".
      An earlier reading of these rows as a C1 x C2 interaction was wrong and is withdrawn.
    - The `two` rows do isolate the kernel without speculation: **-6.7% TPOT, -10% ITL**, consistent with C1's
      4-9% on captures.
    - Still needed to separate them on one node: a four-mode + stock-FA3 arm. mix128 stage running.
  - [x] **KERNEL LADDER ON THE 4-H200 TRIAL vs NAIVE FA3 + STOCK FA2 (2026-10-28, the user's mandated baseline).**
    Trial `bf7d041fb8313d8a` (vibecuda, 4x H200), `unified/a6_h200x4_kernel_ladder_20261026.sh`, log
    `/opt/tiger/w/kernel_ladder.log`; 96 trace-sampled Qwen2.5-7B captures (24 per cell) from real 4-mode MTP
    serving; A6 = `u_np64` (FA2 classes 64/32/16, stock FA3 for >64 packed rows, one plan, ticketed in-launch
    merge); every arm FP32-checked (worst head relative error 0.0083) and graph-replayed 20x.
    | arm | mix32_two | mix32_four | mix128_two | mix128_four |
    |---|---|---|---|---|
    | A6 vs sglang's single varlen FA3 call | 2.107x | 2.647x | 2.921x | 1.861x |
    | naive FA3 + stock FA2, serial, vs that call | 1.954x | 2.436x | 2.796x | 1.759x |
    | naive FA3 + stock FA2, two streams, vs that call | 2.026x | 2.474x | 2.804x | 1.763x |
    | **A6 vs naive serial** | **1.079x** | **1.087x** | **1.044x** | **1.058x** |
    | **A6 vs naive two-stream** | **1.040x** | **1.070x** | **1.042x** | **1.056x** |
    | A6 vs our earlier two-plan v3 | 1.343x | 1.924x | 1.769x | 1.366x |
    | A6 vs fa3_dual_split32 | 1.440x | 1.846x | 1.920x | 1.355x |
    - **The honest kernel claim is 4-9% over a fair baseline, not ~2x.** Splitting the single varlen call by role is
      worth 1.76-2.80x and is the obvious engineering step; our dispatcher adds **4.0-8.7%** on top of it. The ~2x
      figure is only against sglang's shipped call and must not be quoted as our contribution.
    - **Two-stream overlap buys nothing over serial** (1.763x vs 1.759x on mix128_four; 2.026 vs 1.954 at mix32_two
      is the only cell above noise). This independently confirms that the large-tile / small-tile overlap is absent
      once the chunk goes to FA3, and bounds what PDL can be worth.
    - A6 beats our earlier v3 design by 1.34-1.92x, so the one-plan + in-launch-merge build is the right one to
      carry forward; v3 is retired as an arm.
    - Consistent with serving: 4-9% on captures, 1.04x whole-step in the cluster replay, and falling with context.
  - [x] **SLO-BUDGETED CHUNK: rule defined and validated offline (2026-10-28), no DP, no fixed cap.**
    `tasks/perf_model/chunk_predictor.py`, validated on the 10,185 merged steps of the alpha-3 replay traces
    (`/tmp/pfix/st3/w*.log`) with the fitted step model `step_qwen_h20_a6_gpu_20261027.json`. The rule:

        budget_ms = (1 - h) * min over running rows r of [ s_r * (o_r + a) - E_r ]
        chunk     = max { c : T_step(c, prefix, nd, dkv) <= budget_ms }     (bisection, 16 model evaluations)

    s_r = row r's mean-TPOT target, o_r tokens emitted, E_r time spent, a = acceptance EWMA, h = headroom.
    - **Step model accuracy on the chunk the run actually used: MAPE 4.7%, p90 7.5%** -- the fused kernel's step is
      predictable enough to size the chunk from, which is the premise of the whole mechanism.
    - **What the fixed 2048 chunk costs: its predicted step is 5.1x the per-step allowance (p90 8.4x).** That is
      the TPOT damage the rule removes, and it is why a constant chunk cannot be defended.
    - **The rule is adaptive in BOTH directions** (rows decode at 13.9 ms against a 50 ms target, so each emitted
      token banks 36.1 ms of slack):
      | tokens a row has emitted | banked slack | chunk p50 | chunk p90 | steps allowing >= 2048 |
      |---|---:|---:|---:|---:|
      | 0 (just admitted) | 0 ms | 192 | 320 | 0% |
      | 10 | 361 ms | 1600 | 2624 | 31% |
      | 20 | 722 ms | 3072 | 4800 | 84% |
      | 50 | 1806 ms | 7168 | 10816 | 100% |
      | 200 | 7224 ms | 16384 | 16384 | 100% |
      So it admits FAR MORE than 2048 in steady state (up to the server maximum) and clamps only when rows are
      fresh and have no slack -- answering both halves of the user's objection: no throttling when there is nothing
      to protect, real protection when there is.
    - **Why banking must be unlimited:** the SLO is a request's MEAN TPOT. With no banked slack the rule picks
      64-320 tokens and starves prefill, which is exactly the measured `slochunkh` arm (chat TPOT p90 64 ms but
      7.9% attainment, 107 tok/s). The `bank=4` default that produced it turned a mean-TPOT rule into a per-step
      ITL bound. Unlimited banking is the correct reading and is now the default.
    - Simplicity is the point (user): three lines of arithmetic plus a bisection over a 10-term linear model
      (~2 us), no dynamic program, nothing borrowed from SLOs-Serve.
    - Still to run on the cluster: the arm set with this rule against sglang's own schedule and chunk policy.
  - [ ] **Chunk predictor, optional refinement: a learned head instead of bisection.**
    - The user does not want SLOs-Serve's dynamic program (it would read as borrowing their method). Instead predict
      the chunk from state we already have, using the step model that the fused kernel makes easy to fit.
    - Inputs, all already logged per step in STEPTRACE: chunk tokens, prefix length, decode rows and their KV
      lengths, batch size, acceptance EWMA; **plus** each running row's remaining mean-TPOT slack
      (`s_r * (o_r + a) - E_r`, already computed inside apply_slo_budget_chunk_20261026.py) and its SLO class.
    - Target: the largest chunk whose predicted step keeps every row inside its slack. Two options:
      1. **Direct regression** to the max safe chunk -- one forward pass, no search, no DP.
      2. **Reuse the validated step-time MLP and binary-search it** (6 evaluations), which is what
         `edf_compute_max_chunk_size` already does and needs no new model.
      Option 2 is preferred unless the search cost shows up, because that MLP is already validated to 3.4-4.0% on
      merged steps; option 1 only removes a search we can already afford. Either way it is a learned predictor over
      our own fused step, not a borrowed formula.
    - Labels come from traces we already hold, so training needs no new GPU time.
  - [ ] **KERNEL INNOVATION 1 (2026-10-28, user approved): cross-role shared-prefix KV reuse.**
    - In our fan-out agent workload a parent session spawns children sharing its prefix, and the stack runs at 92.7%
      prefix hit. So within ONE four-mode step the prefill chunk (a child's new round) and the decode/verify rows
      (its siblings) often read the SAME prefix KV pages, and today each role streams them from HBM independently.
    - Both roles are in one step under one plan, so the plan can detect the overlap (the radix tree already reports
      shared prefix length) and group the work by KV page rather than by request, fetching shared pages once.
    - **Not available to a disaggregated or separate-call design**: there the roles are in different kernels, often
      on different nodes. It is a direct consequence of C2 putting both roles in one step.
    - **DeFT (arXiv 2404.00242) validates the mechanism** (KV-guided grouping, 73-99% KV IO saved) but only for a
      tree inside ONE generation call. Our novelty must be the CROSS-ROLE and CROSS-REQUEST grouping; do not adopt
      tree attention. See the reference entry.
    - Weak version already measured: KV-position ordering gave 1.08x on a 48K shared prefix and -29% elsewhere, so
      any version must be gated on measured sharing.
    - **[x] GATING MEASUREMENT ANSWERED (2026-10-28), at zero GPU cost: the overlap is ~0 on our main workload,
      for two structural reasons, so this item is NOT buildable against the TraceLab replay.**
      1. *The replay has no spatial sharing.* `drivers/serving/bench_overlay_20261026/tracelab.py` synthesises each
         session's tokens with `draw(n)` from the global RNG (it is a length/session trace, not real agent text), so
         two concurrent sessions share no prefix. Our 92.7% prefix hit is entirely TEMPORAL -- round N+1 reusing
         round N within one session -- not spatial across concurrent requests.
      2. *A session cannot occupy both roles at once.* Rounds within a session are sequential, so the session whose
         chunk is prefilling has no decode rows of its own in that step. Cross-ROLE sharing therefore requires two
         DIFFERENT requests that share a prefix and are concurrent -- which (1) says does not happen here.
      - **Consequence:** cross-role shared-prefix reuse is viable only on fan-out / branching workloads, where
        siblings of one parent are concurrent (one child prefilling over the parent prefix while another decodes
        over it). We have that workload in `drivers/serving/fanout_agents.py`, and it is the ThunderAgent-style
        setting the goal's point 2 describes. Any claim for this mechanism must be made on fan-out, with the page
        overlap reported, and must NOT be claimed on the session replay.
      - This also explains why the weak version (KV-position ordering) gave 1.08x only on a synthetic 48K shared
        prefix and -29% on the trace: the trace has no shared prefix to order around.
      - DeFT's 73-99% KV IO saving comes from trees inside ONE call, where sharing is near-total by construction.
        Our setting only reaches that regime under fan-out.
  - [ ] **KERNEL INNOVATION 3 (2026-10-28): deferred KV commit for rejected draft tokens.**
    - Verify computes and writes paged KV for all 4 draft tokens, but acceptance is ~2.0 on the cluster (3.35-3.55
      at 128K single node), so a large share of verify KV writes are discarded. Holding the draft chain's KV in a
      scratch buffer and committing only accepted entries saves that write bandwidth. Speculation-specific; check
      first whether sglang already elides these writes.
  - [ ] **KERNEL INNOVATION 2 (user, 2026-10-28): pipeline the draft head against the verify forward.**
    - The user's observation: the baseline runs the draft and the verify as SEPARATE attention calls, so can the
      draft of the current step overlap the verify of the previous batch?
    - Why this is the best remaining item. EAGLE3 runs 3 sequential draft forwards of a ONE-layer model per step and
      then one 28-layer verify. In arithmetic the draft is ~3% of the step, but each draft forward is a tiny launch
      at one token per request, so it is latency-bound and costs far more wall-clock than its FLOPs. **It is the only
      remaining idea that deletes SERIAL time instead of shaving row-side work** -- which is exactly why every other
      mechanism (DSMEM, persistent, ready-queue JIT, multicast, L2 order, two-stream) measured zero: none of them
      changed what the step waits on. It also attacks the same quantity as cross-role backfill, beta = 0.796.
    - The dependency, and how to break it: draft(N+1) for a request depends on verify(N) for that request, so they
      cannot overlap within one group. Split the running batch into two staggered groups: group A's verify runs
      concurrently with group B's draft, then the roles swap -- a two-stage pipeline in which the draft head rides
      in the verify's shadow. This is specific to the four-mode step, not borrowed from another paper.
    - Costs to measure, not assume: a staggered batch halves each group's size (worse GPU efficiency per launch, and
      CUDA-graph capture must cover both phases), and acceptance may shift because a group now drafts against a
      slightly older target state. Measured acceptance to protect: 3.35-3.55 at 128K (rate 0.78-0.85).
    - **Zero-cost attempt at the ceiling (2026-10-28): INCONCLUSIVE for the draft share, but it rules out host-side
      headroom.** On 18,075 consecutive decode-step pairs from the alpha-3 traces, the wall gap between successive
      target forwards is p50 30.08 ms while the traced `gpu=` of the forward itself is p50 30.98 ms. So `gpu=`
      already spans the whole step (draft forwards included) and the two cannot be separated from the trace.
      - What it DOES establish: only **0.2% (p90 1.7%) of decode wall time lies outside the traced forward**, so the
        overlap scheduler hides essentially all host work. There is no launch-overhead or host-side time to recover
        at decode steps, which independently explains why the vLLM PR 61869 style launch-count reduction measured
        below our noise floor.
      - The draft share therefore needs in-step instrumentation (CUDA events around the 3 draft forwards, or nsys)
        on the H200. Until that number exists this item stays open and unbuilt -- it must not be claimed.
  - [x] **Llama-3.3-70B kernel ladder on the H200 (2026-10-28), completing the three-model set.**
    `unified/a6_ladder_model_h200_20261028.sh /opt/tiger/w/trace_sampled_llama70b_h20_20261025 llama70b`, log
    `/opt/tiger/w/ladder_llama.log`; 38 rank-0 captures replayed from the H20, FP32 worst error 0.0059, 0 failures.
    | Llama-70B cell | A6 vs naive serial | A6 vs naive streams | A6 vs sglang's single FA3 call |
    |---|---:|---:|---:|
    | mix32, 4-mode | **1.049x** | 1.058x | 7.749x |
    | mix32, 2-mode | **1.028x** | 1.037x | 6.724x |
    - **Three-model summary of C1 against the fair baseline (the naive role split):**
      | model | A6 vs naive role split | role split vs sglang's call (NOT ours) |
      |---|---|---|
      | Qwen2.5-7B | 1.040-1.087x (4.0-8.7%) | 1.76-2.80x |
      | Llama-3.3-70B | 1.028-1.058x (2.8-5.8%) | ~6.4-7.4x |
      | GLM-4.5-Air | 1.020-1.028x (2.0-2.8%) | 5.3-6.0x |
    - **C1 is 2-9% across every model tested, and the large figures (1.8x to 7.7x) all belong to the role split.**
      The pattern is consistent: the worse sglang's single varlen call serves a model's verify rows, the bigger the
      role-split gain and the SMALLER our marginal gain, because stock flashinfer FA2 already tiles those rows well.
  - [x] **GLM-4.5-Air kernel ladder on the H200 (2026-10-28): the tile-class hypothesis is REFUTED.**
    `unified/a6_ladder_model_h200_20261028.sh /opt/tiger/w/trace_sampled_glm_h20_20261025 glm`, log
    `/opt/tiger/w/ladder_glm.log`; 43 rank-0 captures copied from the H20 (24 four-mode, 19 two-mode), FP32 worst
    relative error 0.0064, 20 graph replays, 0 correctness failures.
    | GLM cell | A6 vs naive serial | A6 vs naive streams | A6 vs sglang's single FA3 call |
    |---|---:|---:|---:|
    | mix32, 4-mode | **1.020x** | 1.028x | 6.029x |
    | mix32, 2-mode | **1.023x** | 1.025x | 5.328x |
    - GLM was the best candidate for a larger kernel claim: its verify rows pack to 48 queries, which is what the
      FA2 64-row tile class was built for, and an earlier H20 measurement showed 1.38x for that class. **It is in
      fact where our dispatcher matters LEAST: 2.0-2.8% over the naive role split, against 4.0-8.7% on Qwen.**
    - Why: GLM gains **5.3-6.0x** from the role split alone (Qwen 1.76-2.80x), because FA3's tiles serve 48-row
      verify rows badly. But stock flashinfer FA2 already picks a good tile for them, so our classes add ~2%. The
      earlier 1.38x was measured against FA3, not against a competent FA2 -- the wrong baseline made the tile
      classes look like the mechanism when the role split was.
    - **Conclusion across three models: C1 is 2-9% over a fair baseline, and every large number (1.8x, 2.9x, 6.0x)
      belongs to the role split, which is not our contribution.** No further kernel item has produced a significant
      gain; the remaining untested idea is cross-role backfill (needs `tma_store_wait`), which is the only one that
      attacks beta (0.796 on H20) rather than the row-side work.
  - **Where C1's gain actually is, measured step-for-step (2026-10-27, user challenged the 4% figure).**
    - Method: STEPTRACE from two arms of the same alpha-3 replay, same schedule and router, matched by step shape.
      A6 arm `..._steptrace3_r1` (validity check: 70,701 `NATIVE_PDL_CALL ... a6_unified=1`), stock arm
      `..._abl_r1_noC1` (`KERNEL=lpt` on `ATTN_BACKEND=fa3`, zero a6 calls). Whole-step GPU time, merged steps with
      a 1024-2048 token chunk, bucketed by the chunk's prefix depth:
      | chunk prefix | A6 | stock FA3 | speedup | n (A6 / stock) |
      |---|---|---|---|---|
      | < 8K | 286 ms | 329 ms | **1.147x** | 296 / 275 |
      | 8-32K | 358 | 374 | 1.047x | 3355 / 3104 |
      | 32-64K | 513 | 545 | 1.062x | 3151 / 2979 |
      | 64-128K | 727 | 746 | 1.026x | 2231 / 2183 |
      | > 128K | 682 | 686 | 1.006x | 75 / 77 |
      All steps weighted: 1.041x (merged 1.055x, decode 1.011x, prefill 1.004x).
    - **The step IS attention-dominated, which confirms the objection rather than contradicting it.** A 2048-token
      chunk over a 128K prefix costs 2 x (2 x 2048 x 131072 x 128) x 28 heads x 28 layers = 1.08e14 FLOP; at H20's
      ~148 TFLOP/s bf16 that is 0.73 s, against the measured 727 ms step. Attention is ~100% of that step.
    - **So the small gain is not a small attention share; it is that both arms compute the dominant term
      identically.** Both send the prefill chunk to FA3. C1 only changes how the decode / verify ROWS are computed,
      and at 128K the chunk outweighs the rows by orders of magnitude. The speedup therefore FALLS with context
      (1.147x -> 1.006x), which is the signature of a row-side optimisation, not of a general attention speedup.
    - Consequences:
      - C1 pays where rows dominate: decode-heavy, high concurrency, short-to-medium context. It is nearly inert on
        the long-context chunk-dominated steps that define the TraceLab agent workload.
      - The 96-capture set that produced 1.43-1.51x (H200) over-represents row-heavy shapes relative to the trace.
        On H20 the same build is 1.10-1.23x on captures, and 1.04x in serving.
      - A real kernel win on this workload must come from the CHUNK's own FA3 attention, or from genuine
        large-tile / small-tile overlap so the rows ride inside the chunk's 727 ms instead of adding to it.
    - **We do not have that overlap.** PDL only lets the FA2 launch begin as FA3 drains; it is not the persistent-FA2
      design that interleaved a 128-row prefill tile with 16-row row tiles on the same SMs. Once the chunk moved to
      FA3 that idea left the system, and the fitted beta = 0.796 on H20 measures how little is hidden.
    - Why MELD's JIT reduction pays for them and not for us: their tasks carry many split partials, while our
      trace-sampled steps are typically one chunk plus a couple of verify rows, so the merge is <= 0.9% of the step.
      On synthetic decode-heavy batches of 64-256 rows it is 6-13%, so the mechanism is not useless -- our steps lack
      the structure. Draft-and-verify fusion is impossible: the three draft forwards are sequentially dependent.
  - **vLLM PR 61869 reviewed (user, 2026-10-27): orthogonal to our epilogue, two small things worth taking.**
    - What it is: "DSpark draft on the specialized decode, with a fused Markov loop and a post-forward CUDA graph",
      DeepSeek-V4.1-Flash only, behind `decode_backend=latency_specialized`. Open, `needs-rebase`, no approving
      review, depends on PR 61868. Despite the name there is **no persistent sampling loop over steps**: the driver
      is still a Python `for i in range(n_spec)` calling one fused kernel per draft position
      (`spec_decode/dspark/fused_sample.py::_markov_sample_kernel`), which keeps the bias GEMM and folds the other
      ~9 launches per position into one.
    - Every mechanism sits **after the model forward** (LM head, rejection sampler, request-state update, draft input
      prep) or inside the draft head. Nothing touches attention tiles, split-KV, the merge or PDL. It would sit
      downstream of our two launches, not inside them.
    - It attacks host launch overhead, the same class as our merged-step read-back regression, but the magnitude does
      not transfer: their post-forward CUDA graph is -37 us on a 4,075 us GB300 TP4 batch-1 step (0.9%), and their
      headline 6.4% comes from a step with ~1.5 ms of host loop. Our steps are 20-40 ms, so the same region is
      ~0.1-0.2%, below our 2-3% noise floor -- the same Amdahl bound that killed the argmax-into-LM-head idea.
    - Their graph is also gated off exactly our step: `_step_eligible` returns false on `input_batch.has_prefill` and
      requires `num_scheduled_tokens == num_steps + 1` everywhere, i.e. uniform pure decode only. Our 4-mode merged
      step would never qualify.
    - **Worth taking (two items, both small and concrete):**
      - `_copy_step_tensors_kernel` + `_reading_copies`: batch every per-step host-backed value into device buffers in
        one launch and repoint readers at the copies, so no stage reads host state mid-step. This is a direct answer
        to the merged-step host read-back that cost 6-7% on short-prompt ShareGPT, and it needs no CUDA graph.
      - The self-resetting arrival counter (`atomic_xchg` to 0 by the finishing block). Our shipped non-persistent
        path has **no** per-launch memset to remove (the plan stages the counter as zero and copies it), so this is a
        no-op today -- but it is exactly what the pending "make A6 graph-capturable" item needs, since a replayed
        graph cannot rely on a host-side plan upload to re-zero the counter. The only `cudaMemsetAsync` we have is
        `persistent_fa3_fused.cuh:1007` (the persistent variant's cluster cursor), which that pattern would remove.
    - **Negative lesson on persistence, from their own numbers:** the genuinely persistent, cooperative, cross-rank
      sharded Markov kernel bought -8 to -20 us, while the plain per-position fused kernel bought -93 us -- launch
      count beat persistence by ~5x, at far less complexity and without their spin-wait-in-a-graph hang risk. That
      agrees with our persistent-LPT ablation (0.987-0.998x) rather than contradicting it.
    - Nothing here changes a paper claim; at most a one-line related-work note that concurrent vLLM work attacks the
      same host-overhead class in the sampling stage, which C1 does not claim.
  - **Closed-form chunk sizing from the fused kernel's cost law (2026-10-27, user: can we avoid SLOs-Serve's DP and
    tie this to our own contributions?).**
    - Why they need a DP and we do not: SLOs-Serve searches token allocations over `(n_1 ... n_L)` SLO-class states
      because its mixed-step cost is not invertible. Ours is, and only because of C1: the fused kernel obeys
      `T_attn = max(T_p, T_d) + beta x min(T_p, T_d)`, a quadratic in the chunk, solvable in closed form. The same
      law fit to the split-kernel baseline gives beta = 2.19 (its mixed cost is super-additive in its parts), so a
      split-kernel system has no closed-form controller and must search. "The kernel turns chunk sizing from a search
      into one equation" is the C1 -> C3 claim.
    - Balance-point chunk `c*`: the largest chunk whose attention still fits inside the decode pass (`T_p = T_d`), so
      it rides in its shadow and costs only `beta` of its own time. With the fitted H20 constants, 30K KV per row,
      32K prefix:
      | decode rows | decode-only step | c* | step at c* | step at 2048 |
      |---|---|---|---|---|
      | 4 | 27.7 ms | 162 tok | 55.9 ms | 427 ms |
      | 16 | 44.9 ms | 326 tok | 101.8 ms | 441 ms |
      | 48 | 91.0 ms | 761 tok | 223.8 ms | 478 ms |
      The policy is the opposite of a cap: **the busier the decode pool, the larger the free chunk** (162 -> 761
      tokens), because a longer decode pass casts a longer shadow. Every per-step cap we have tried shrinks exactly
      when this says to grow, which is consistent with "EDF: order, don't cap".
    - Acceptance enters as the per-step allowance `s x a`, so MTP at a ~ 2 doubles each row's budget per step and
      widens the admissible chunk: a direct C2 tie-in, with `a` already tracked per worker as an EWMA.
    - **Honest caveat:** the fitted H20 beta is 0.796, so only ~20% of the chunk's attention hides. Prefill
      efficiency at `c*` is 5.7 tokens per ms of added step time vs 5.1 at 2048, about 10%, not a step change. The
      published H800 beta is 0.529, where the discount is much larger. Present the mechanism on H20 and measure the
      claim on the H200 kernel stage; do not headline 10%.
  - **Which latency metric each system scores (2026-10-27, user asked whether to switch to ITL).**
    - Ours, SLOs-Serve and JITServe score a request's **mean TPOT** (`sum(itls) / (output_len - 1)`). SLOs-Serve uses
      TPOT 50 / 100 ms with TTFT 3x / 5x the zero-load latency; JITServe states it as a per-token deadline, token i on
      time if done by `TTFT_SLO + i x TBT_SLO`, which is the same mean constraint expressed as a deadline.
    - **Crossflow scores ITL**: it reports mean ITL and carries a 120 ms p99-ITL constraint (which our analysis notes
      comes from its Figure 3 characterization, not a per-dataset SLO). Its headline weakness is a mean-ITL
      regression of 20.9-34.3% on GLM-5.2 TraceLab under decode-local prefill, which is the cost our fused step is
      meant to remove.
    - Decision: keep mean TPOT for attainment (it is what our SLO numbers were taken from), and **also report mean and
      p99 ITL per tenant**, since Crossflow is the closest workload comparison and ITL is its currency. Added to
      `perf_model/tracelab_windows.py` (ITL percentiles over every inter-token gap) and deployed on pod 0.
    - Do not switch the controller to a per-step ITL bound: that is strictly stricter than what we claim, and v5
      measured its cost (chat 65.1 -> 12.7% at alpha 3).
  - **How to reserve capacity under a mean-TPOT SLO (2026-10-27).** A per-step bound is easy but wrong; the mean's
    slack depends on the unknown remaining output length. Two published answers:
    - **SLOs-Serve** turns TPOT into a **rate reservation**: a token budget per time unit, with the DP tracking
      accepted request counts per TPOT class `(n_1 ... n_L)`. Decode needs `sum(n_l / s_l)` tokens/s; prefill gets the
      residual capacity. It pairs this with **soft admission control**: admitted requests are guaranteed their SLOs,
      and requests whose SLOs are unattainable are routed to another replica or served best effort.
    - **JITServe** keeps a per-token deadline and avoids greedy spending with a predicted upper-bound output length
      (quantile regression forest, refined every ~50 tokens), giving each request just enough bandwidth to meet its
      SLO just in time.
    - Design for the next iteration: (1) decode rate reservation `sum(n_l / s_l)` from the step model, prefill takes
      the residual, so the chunk grows when few rows decode and shrinks when many; (2) when the reservation exceeds
      capacity, the lever is admission, not throttling, which our router (placement) and demotion (best effort)
      already provide; (3) the per-token deadline `arrival + TTFT_SLO + i x s` unifies TTFT and TPOT on one line, so
      the router and the worker price the same quantity. The queued v7 arm uses the cumulative-credit form and is a
      data point on greedy spending, not the final design.
  - **Inventory of predictors and the chunk solver (2026-10-27, user asked; the fixed 2048 chunk is not a design
    choice but an unset env var).** Our tree already contains the SLO-aware chunk-size solver:
    - `managers/edf_scheduler.py` `edf_compute_max_chunk_size()`: binary-searches the largest chunk whose predicted
      step fits the tightest running row's remaining slack (`ITL_SLO_MS x tokens_per_step - elapsed`), with a
      safety margin, plan overhead, and a guard that returns 0 when decode alone already exceeds the slack. This is
      exactly "solve for the largest chunk that does not violate the SLO".
    - It is wired into `get_new_batch_prefill` and gated on `SGLANG_EDF_LATENCY_MODEL`, which our multi-node runs set
      to the empty string. That is why every arm so far ran a fixed 2048 chunk.
    - Three interchangeable predictors (same `predict(n_p, r_p, decode_avg_kv, decode_bs)` contract):
      - `.pt` -> `LatencyPredictor`, the trained MLP (5 features, evaluated in NumPy, ~us per call);
      - `linear:<json>` -> `LinearStepPredictor`, from `perf_model/fit_mtp_step.py` (includes a `c_acc` term);
      - `analytic` -> `AnalyticOverlapPredictor` (`managers/overlap_cost_model.py`), the closed form built for the
        fused kernel: `T_attn = max(T_p, T_d) + beta x min(T_p, T_d)`, constants per GPU via
        `SGLANG_OVERLAP_COST_PARAMS`.
    - The router's model is separate and coarser (calibrated cold-prefill curve + decode grid + EWMAs); it predicts
      placement, not steps.
    - **The analytic model is the one tied to C1**: its docstring records that the split-kernel baseline's mixed cost
      is super-additive in its parts (up to 8.1x; fitting max + beta x min gives beta = 2.19), so no closed-form
      controller can invert it, while the fused kernel obeys the law to 3.1% MAPE. The fused kernel is what makes
      SLO-aware chunk sizing possible, which is the C1 -> C3 link the paper should make.
    - [x] **H20 constants fit from real steps** (2026-10-27, `perf_model/fit_overlap_params_from_steptrace.py` on the
      87,932 corrected alpha-3 steps, first half fit / second half scored;
      `perf_model/overlap_params_h20_qwen7b_20261027.json`): held-out MAPE **4.2% on merged steps** (p90 8.8%), the
      steps the chunk solver actually prices; prefill-only 39.9%, decode-only 55.0% (a mix of MTP draft and verify
      steps that one roofline does not separate), all 48.5%. beta = 0.796 on H20 vs 0.529 on H800.
    - [ ] Next: run the existing solver with `SGLANG_EDF_LATENCY_MODEL=analytic` +
      `SGLANG_OVERLAP_COST_PARAMS=<h20 json>` against fixed 2048 at alpha 3, and retire the ad-hoc
      `apply_slo_budget_chunk_20261026.py` (it duplicates the solver with a worse model and no decode-only guard).
      The decode-only guard needs the measured decode step time rather than the 55%-error prediction.
  - [ ] **SLO-budgeted prefill chunk size from the EDF step model.**
    - Code (2026-10-26, unit-checked on a stub scheduler, not yet served):
      - `unified/apply_slo_budget_chunk_20261026.py`, enabled by
        `SGLANG_SLO_CHUNK="<model.json>:<h>:<chat_ms>:<agent_ms>:<agent_prompt_tokens>[:demote]"`;
      - the step model is `perf_model/fit_4mode_step.py --export-lin4mode` (lin4mode on the gap target);
      - the server runs with `--chunked-prefill-size 16384` so that the chunk can grow.
      - Unit check (toy coefficients):
        - no decode rows: 16384;
        - 8 fresh chat rows on a 30K prefix: 1716;
        - 4 agent rows at 80 ms: 10373;
        - one row 10 s behind: 0 without `demote`, 1709 with it.
    - Order on the H200: collect A6 STEPTRACE at the Table 7b loads, export the model, then run the arms below.
    - Bug (2026-10-27): the step tracer logged a 4-mode step's 4-token verify rows as prefill rows (nd = 0 on all
      ~2300 merged steps per worker), so fit_4mode_step.py refused to fit and the v1 chunk driver stopped.
      - Root cause: the merge runs inside the speculative worker after `_steptrace_begin`. The tracer reads the running
        rows from `batch.mixed_spec_running`, but skipped them because the serving path keeps no host `seq_lens_cpu`.
      - The first fix keyed on `mixed_verify_rows` (set only on the merged object inside the worker) and changed
        nothing: the v2 re-trace still had 0 mixed rows, and it was stopped.
      - `pd_colocation/apply_steptrace_verify_rows_20261027.py` now takes the running rows' KV from their requests. It
        is applied to both trees on all pods (v1 reverted), and the run continues as
        `mn_slo_chunk_v3_16g_20261027.sh`.
      - Fixed tracer, verified: about 2,500 merged steps per worker are logged as mixed (nd > 0).
      - Fit (2026-10-27, `fit_4mode_step.py --target gap --export-lin4mode`, pod 0's 4 workers, trace
        `coloc_slo_edf_mtp_hc_tl2_a3_steptrace3_r1`; log `/tmp/h20/mn_slo_chunk_16g_v3_fit.log`):
        - The gap target (launch to launch, including host waits under the overlap scheduler) is not fittable by
          lin4mode. Held-out decode 43-59%, prefill 46-52%, mixed 18-24% MAPE. Its coefficients are unphysical
          (c_tok = -0.0005 ms/token, c_mixed = 53 ms).
        - The GPU-time target fits better: 25% in-sample MAPE, and physically sensible coefficients (decode-only
          step 21.7 ms at 8 rows x 30K KV; a 2048-token chunk on a 32K prefix 435 ms).
        - The alpha-6 arm on the gap model was stopped. The arms rerun on the GPU-only model
          (`tasks/perf_model/step_qwen_h20_a6_gpu_20261027.json`) as `mn_slo_chunk_v4_16g_20261027.sh`.
        - [x] Nonlinear fits on the corrected traces (2026-10-27, local, `fit_4mode_step.py --target gpu --pool` on
              the STEPTRACE / MIXEDSPEC lines of pod 0's four workers, copied to `/tmp/pfix/st3/`). Held-out errors,
              pooled across the four workers, per worker:
              - MLP with acceptance: mixed (merged) 3.4-4.0%, prefill 4.6-5.5%, decode 18-23%;
              - MLP: mixed 3.4-4.2%, prefill 5.5-6.3%, decode 21-23%;
              - GBT: mixed 2.8-3.6%, prefill 6.9-7.8%, decode 21-25%.
        - [ ] Stock comparison on the same schedule: the leave-one-out arm without C1 is now traced (`STEP_TRACE=1`).
    - [x] v4, alpha 6, budgeted chunk on the GPU-time model (2026-10-27, `mn_slo_chunk_v4_16g_20261027.sh`, log
          `/tmp/h20/mn_slo_chunk_16g_v4.log`). **Worse than fixed 2048 on every metric**:
          - chat 27.0% vs 31.8%, chat TPOT p50 88.6 vs 72 ms;
          - agent 44.8% vs 50.0%, agent SLO goodput 647 vs 890 tok/s.
          Cause, read from the SLO_CHUNK lines: budgets of 243 ms, 3.7 s and infinity, and median chunk 8414 tokens.
          - Tokens per step came from the rows' progress between calls. Calls skip decode-only steps, so it read 8.8
            at acceptance ~2.
          - Demotion dropped late rows from the bound, so at overload, when every row is late, the budget was infinite.
          Fixed in `apply_slo_budget_chunk_20261026.py`:
          - tokens per step = EWMA of the scheduler's accepted tokens per verify forward;
          - demoted rows still bound the step at s_r x a;
          - banked slack is capped at 4 steps (`bank=K`).
          Stub check: all rows late -> chunk 398 (was 16384); large banked slack -> 2530. The alpha-3 v4 arm was
          stopped and all three arms rerun as v5 (`mn_slo_chunk_v5_16g_20261027.sh`).
    - [x] **v5, alpha 3: the rule is wrong, not just mis-levered** (2026-10-27). Fixed 2048 vs v5:
          - chat attainment 65.1% -> **12.7%**, chat TTFT p50 0.14 -> **80.7 s**;
          - chat TPOT p50 30.0 -> **45.0 ms**, i.e. the budget made the metric it protects *worse*;
          - agent 72.6% -> 26.1%, goodput 1501 -> 194 tok/s.
          **Diagnosis.** At alpha 3 under fixed chunks chat TPOT p50 is already 30 ms against a 50 ms target, so
          nothing needed throttling; the rule still cut the chunk about 7x. The SLO is a request's *mean* TPOT, but
          the rule enforces a *per-step* bound as if every step were merged. Merged steps are only 12% of steps
          (10,185 of 87,932), so "every step within one allowance" is far stricter than "mean within target". The
          banking term existed to amortise that, and v5's `bank=4` cap destroyed it; v4 left banking uncapped but
          mispriced tokens per step and dropped the bound for late rows. Both directions therefore failed.
          The surge arm was stopped.
          **Correction (user, 2026-10-27): the SLO is mean TPOT, so the fix is the cumulative budget, not sglang's
          solver.** Checked on both sides:
          - our scorer uses `sum(itls) / (output_len - 1)`, a request's mean;
          - `edf_compute_max_chunk_size` uses `slack = ITL_SLO_MS x a - elapsed_since_last_step`, a per-step (ITL)
            bound, which its own comment calls "the strict per-step controller" at the default
            `SGLANG_EDF_ITL_QUANTILE=0`. It carries the same mismatch, so adopting it unchanged would repeat the
            failure; its quantile knob is a tail approximation of the mean, worth testing separately.
          The cumulative form `s_r x (o_r + a) - E_r` is the one that matches a mean SLO, and that was v4's formula.
          v4 failed on two unrelated bugs (tokens per step measured across several steps, and demotion dropping late
          rows from the minimum), which v5 fixed while adding the `bank=4` cap that re-imposed a per-step bound.
          Default banking is now unlimited. Stub check at a 50 ms target: a row at target gets 491 tokens, a row
          decoding at 30 ms (the measured alpha-3 condition) gets 15,246 (so full chunks, matching fixed 2048), a row
          at 70 ms gets 491, and the v5 cap would have given 3,079. Deployed as `/dev/shm/w/python_slo3`; the A/B
          (`mn_slo_chunk_v7_16g_20261027.sh`, alpha 3 first) is queued after the ablation.
    - [x] v5, alpha 6 (2026-10-27, log `/tmp/h20/mn_slo_chunk_16g_v5_a6.log`): **TPOT is protected, but prefill
          starves.**
          - Chat TPOT p50 / p90: 47 / 64 ms (fixed 2048: 72 / 214); agent TPOT p50 54 ms (71).
          - Chat TTFT p50: 74 s (0.40 s); agent TTFT p50: 55 s (1.1 s).
          - Attainment: chat 7.9% (31.8%), agent 15.0% (50.0%); agent goodput 107 tok/s (890).
          - Far past capacity, a TPOT-only budget leaves too little prefill and the backlog grows without bound. This
            matches the earlier finding that every per-step chunk cap loses.
    - [x] **Leave-one-out, arm 1 of 4: without C1** (2026-10-27, `mn_ablation_contrib_16g_20261027.sh`, log
          `/tmp/h20/mn_ablation_contrib_16g_r1.log`; KERNEL=lpt on ATTN_BACKEND=fa3, i.e. sglang's single varlen FA3
          call, with our 4-mode schedule, deadline order and router unchanged; STEP_TRACE=1 for the step-model
          comparison). Chat 61.7% ; agent 70.1% ; 1388 tok/s ; 8653 rounds issued.
          Full stack at alpha 3: 65.1% / 72.6% / 1501 / 8995 (r1) and 68.5% / 78.7% / 1763 / 9665 (r2, the traced
          arm). So removing C1 costs 3.4-6.8 points of chat attainment, 2.5-8.6 points of agent attainment and
          7.5-21% of agent SLO goodput. **Second repeat (2026-10-27): 63.4% / 73.2% / 1504 tok/s / 8,901 rounds.**
          With noC1 at 1388-1504 and the full stack at 1501-1763, the ranges overlap, so **C1's leave-one-out effect
          on this cluster workload is within noise** and the 7.5-21% range is withdrawn. C1's evidence is the kernel
          microbenchmarks (1.43-1.51x sglang's FA3 call) and the single-node long-context runs (+10.7% agent
          throughput, -24% TPOT at 128-256K), not this replay.
    - [x] **Leave-one-out, arm 2 of 4: without C2** (2026-10-27; `coloc_slo_edf_mtps_hc`, i.e. stock MTP steps
          instead of the merged 4-mode step, A6 kernel / deadline order / router unchanged). Chat 69.1% ; agent
          77.4% ; 1638 tok/s ; 9443 rounds issued.
          **This is inside the full-stack repeat spread** (65.1 / 72.6 / 1501 and 68.5 / 78.7 / 1763): better than
          r1, worse than r2 on every column. So on this cluster replay at alpha 3, merging the prefill chunk into
          the verify forward is worth about nothing, and the honest statement is "no measurable effect here".
          Consistent with the step census: only ~11% of steps are merged (~2,500 of ~22,000 per worker), because the
          mix gate needs >= 8192 context or >= 4 queued requests, so C2 is diluted. Its measured value is at long
          context on one node (128-256K: agent tok/s +8.1% four-mode, TPOT -39%) and its paper claim is that prior
          colocation designs cannot run speculative decoding at all (POD-Attention, pdmux), not that merging beats
          stock MTP on every workload. Note the arm still runs speculation, so this is "merged vs separate verify
          step", not "spec vs no spec".
    - [x] **Leave-one-out, arm 3 of 4: without C3's worker level -- REMOVING IT WINS** (2026-10-27;
          `coloc_slo_mtp_hc`, i.e. sglang FCFS admission on every worker, with the A6 kernel, the 4-mode step and the
          SLO-checked router unchanged). Chat **73.9%** ; agent **83.2%** ; **1922 tok/s** ; 10,022 rounds issued.
          Against the full stack (65.1 / 72.6 / 1501 / 8995 and 68.5 / 78.7 / 1763 / 9665) that is **+5.4 to +8.8
          points of chat attainment, +4.5 to +10.6 points of agent attainment and +9 to +28% agent SLO goodput**,
          i.e. clearly outside the repeat spread.
          - Where the trade sits: FCFS gives better TPOT (chat p50 21.5 vs 27.3-30.0 ms, ITL mean 35.8 vs ~51 ms) and
            a worse TTFT tail (chat p90 3.03 vs 1.03-1.19 s) at the same median TTFT (0.11 vs 0.13-0.15 s). Under a
            2 s TTFT + 50 ms TPOT SLO that trade favours FCFS, so our deadline machinery is spending TPOT to protect
            a TTFT tail that the SLO does not charge for.
          - This is the same shape as the 2026-10-20 single-node finding that every strict decode-first variant
            lowered attainment below the uncontrolled arm.
          - **Do not report C3's worker level as a cluster-level gain.** Its measured value is the single-node agent
            burst (chat TTFT -96..-98% from demotion), a different workload. On this replay it costs.
          - **Correction after checking `mn_node.sh`: demotion was never active in any H20 multi-node arm.** The
            file had no `SGLANG_EDF_LATE_DEMOTE` until 2026-10-27, though the scheduler carries the code. So C3's
            worker level here is EDF ordering + chunk preemption/parking + budget reservation, and the mechanism
            with the single-node evidence has never been tested multi-node. `LATE_DEMOTE=1` is now plumbed through
            `mn_node.sh` / `mn_run.py`.
          - The arm removes three mechanisms at once, so it does not say which one costs; preemption/parking is the
            prime suspect (per-step work, and it can thrash a long chunked prefill).
    - [x] **Leave-one-out, arm 4 of 4: without C3's router -- REMOVING IT WINS BY A LARGE MARGIN** (2026-10-27;
          `coloc_cache_edf_mtp_hc`, i.e. router policy `cache` (highest prefix hit, load tie-break, no TTFT/TPOT
          check, no pull pricing), with the A6 kernel, the 4-mode step and worker EDF unchanged).
          Chat **91.4%** ; agent **94.3%** ; **2514 tok/s** ; 11,440 rounds issued (1,632 not).
          Against the full stack (65.1 / 72.6 / 1501 / 8995 and 68.5 / 78.7 / 1763 / 9665): **+23 to +26 points of
          chat attainment, +16 to +22 points of agent attainment, +43 to +67% agent SLO goodput**, and every latency
          column improves (chat TTFT p50 0.07 vs 0.13-0.15 s, p90 0.59 vs 1.03-1.19 s; chat TPOT p50 11.6 vs
          27.3-30.0 ms; ITL mean 19.0 vs ~51 ms).
          - Mechanism: on this workload each session has exactly one cache holder, and the SLO check moves a request
            off its holder whenever it predicts a miss. The prediction is wide (measured / predicted TTFT p90 2.3x),
            so it moves requests that did not need moving, and each move recomputes a long prefix. Plain affinity
            keeps every session on its holder.
          - This does not contradict the earlier 8-GPU result where the SLO check won (+24% agent throughput at
            C=32): that was the fan-out workload, where a prefix is replicated across many nodes and moving a request
            is cheap. **The router's value is workload-dependent, and on session-affine agent traces it is negative.**
          - **Consequence for the paper: the configuration we have been reporting as "ours" is not our best measured
            configuration.** A6 + 4-mode + cache-affinity routing + FCFS workers beats it on every column. Table 10d's
            "ours" row at alpha 3 (65.1% / 1501) should be read as the full stack, with the best configuration
            reported beside it; against conditional disaggregation (50.9% / 956) the best configuration is 91.4% /
            2514, i.e. +40 points of chat attainment and 2.6x the agent goodput.
          - Follow-up queued below: the best configuration against conditional disaggregation and stock PD at the
            same point, so the comparison is best-vs-best rather than our-full-stack-vs-theirs.
    - [x] **Where the lead over disaggregation comes from: it is a prefix-cache gap, and it is not ours**
          (2026-10-28, `mn_gapsource_16g_20261027.sh`, log `/tmp/h20/mn_gapsource_16g_r1.log`; cache hit counted from
          the workers' `Prefill batch ... #cached-token` lines, since the client's `cached_tokens` field reads 0 in
          every arm -- a scorer bug noted below).
          | arm (alpha 3) | chat | agent | agent tok/s | TPOT p50 | ITL mean | prefix hit | rounds issued |
          |---|---|---|---|---|---|---|---|
          | stock upstream sglang, colocated, `cache_aware` router | 81.1% | 88.7% | 1976 | 21.6 ms | 28.3 ms | **91.8%** | 9656 |
          | ours: A6 + 4-mode + cache router + FCFS | **90.6%** | **94.2%** | **2471** | **11.8 ms** | **17.7 ms** | 92.7% | 11424 |
          | conditional disaggregation (Dynamo defaults) | 54.6% | 65.2% | 1031 | 41.9 ms | 66.4 ms | **11.2%** | 7332 |
          - **The disaggregation gap is a cache gap.** Conditional disaggregation recomputes nearly every prompt
            (11.2% hit, 21.3M new tokens) because a round's prefill lands on a worker that does not hold that
            session's prefix; colocation with session affinity keeps it where 92% of the prompt already is, doing
            ~8x less prefill work for the same offered load. That is the whole 2471 vs 1031 tok/s.
          - **But it is NOT our contribution: stock sglang reaches 91.8% with its own `cache_aware` router.** This is
            reported as a finding about the layout (colocation beats disaggregation on session-affine agentic
            traces, with sglang's shipped routing), not as a contribution of ours. Do not claim the cache hit.
          - **Our contribution is what we add at the same hit rate.** Against the strongest baseline (stock
            colocated, 91.8% vs our 92.7% -- the same prefill work): **+25% agent SLO goodput, -45% TPOT, -37% mean
            ITL**, chat SLO misses halved (18.9% -> 9.4%), 11,424 rounds served against 9,656. This clears the
            publishability bar (>= 10% throughput, >= 30% on a latency metric) against a baseline that is not
            handicapped.
          - **Attribution (2026-10-28, `mn_bestdecomp_16g_20261028.sh`, log `/tmp/h20/mn_bestdecomp_16g_r1.log`).**
            The hypothesis that C2 carries the -45% TPOT is **FALSIFIED**. At the best configuration (cache router +
            FCFS workers), removing the four-mode step changes nothing:
            | arm | chat | agent | agent tok/s | TPOT p50 chat/agent | ITL mean chat/agent |
            |---|---|---|---|---|---|
            | full (A6 + four-mode) | 90.6% | 94.2% | 2471 | 11.8 / 17.5 ms | 17.7 / 22.3 ms |
            | noC2 (A6 + stock MTP) | 90.6% | 93.8% | 2492 | 11.0 / 18.2 ms | 17.2 / 22.8 ms |
            Identical within the noise floor, and noC2 is nominally faster.
            **RESOLUTION (2026-10-28, supersedes both explanations below).** The same toggle gives -38% TPOT on one
            node and zero on the cluster, and the reason is the four-mode step's **load/context gate**:
            `SGLANG_MIXED_SPEC_MIN_CTX=8192` (or >= 4 queued requests). `_mtp` sets
            `SGLANG_ALLOW_MIXED_CHUNK_SPEC=1 SGLANG_MIXED_SPEC=1`; `_mtps` does not (mn_node.sh:251), so both the
            cluster noC2 arm and the single-node `--ours` control disable the same mechanism.
            - On the cluster the stack reaches 92.7% prefix hit, so a typical agent round carries only
              **12.4M new tokens / 11,424 rounds = ~1085 uncached tokens**, about **7.5x BELOW the 8192 gate**. The
              four-mode step therefore rarely fires, and removing it costs nothing.
            - On the single node the stages are 32K and 128K of uncached prefill, always above the gate, so it fires
              on every chunk and is worth -38% TPOT (22.9 -> 14.2 ms at 32K with the kernel held at stock FA3).
            - **So C2's value scales with how much UNCACHED prefill a step carries.** It pays when prefills are long
              and cold (new sessions, long documents, low hit rate) and is inert when cache affinity has already
              reduced the prefill to ~1K tokens. This is a real tension between our own contributions: C3's
              cache-first routing removes the condition that makes C2 valuable. Both the claim and the tension
              belong in the paper; neither is a defect.
            - Consequence for the earlier entries: C2 is NOT "duplicating an upstream capability" (that reading was
              wrong -- `_mtps` has no mixing at all, it is not upstream's spec-aware MIXED step), and the null is NOT
              an Amdahl effect. Both superseded.
            **Superseded explanation (kept for the record): this is NOT an Amdahl bound.** Merged steps are 11.6% of steps
            but **67.2% of GPU time** (decode-only 83.9% of steps / 27.6% of time; prefill-only 4.5% / 5.1%), so the
            four-mode step applies to two thirds of the GPU time and still changes nothing.
            The real reason is that **upstream already does the equivalent**: sglang's spec-aware MIXED step merges
            decode tails into a prefill step, so our four-mode step duplicates a capability the baseline has. C2's
            measurable contribution over upstream at this workload is ZERO. It must be presented as an enabling
            mechanism for the fused kernel's row classes, not as a source of gain, unless a workload is found where
            it beats upstream's MIXED step.
            **noC1 at the same best configuration (2026-10-28): also zero.** Replacing A6 with sglang's single
            varlen FA3 call gives chat 89.2% / agent 94.0% / **2436 tok/s** against 90.6 / 94.2 / 2471 -- a 1.4%
            goodput difference, inside the noise floor.
            **So at cluster scale C1 and C2 both measure zero, yet the stack still beats stock colocated sglang by
            +25% goodput and -45% TPOT.** The remaining candidates are the chunk size (2048 vs upstream default),
            our router against sglang's `cache_aware`, and the 32-file delta between our serving tree and upstream.
            **COMPLETE (2026-10-28).** All four arms on our tree with acceptance 1.983-1.997 (so internally valid,
            unlike the withdrawn stock comparison):
            | arm | chat attain | chat TTFT p90 | agent attain | agent tok/s |
            |---|---:|---:|---:|---:|
            | full stack (our cache router + load tie-break) | **90.6%** | **0.67 s** | 94.2% | 2471 |
            | sglang's `cache_aware` router instead (`sglrt`) | 83.3% | 3.24 s | 94.3% | 2482 |
            | noC1 | 89.2% | 0.86 s | 94.0% | 2436 |
            | noC2 | 90.6% | 0.67 s | 93.8% | 2492 |
            | upstream default chunk (`dchunk`) | 90.1% | 0.64 s | 94.4% | 2499 |
            - **The router is the only cluster-level contribution that measures: +7.3 points of chat attainment and
              a 79% lower chat TTFT tail (3.24 -> 0.67 s), at identical agent goodput and acceptance.** The
              mechanism is the load tie-break among workers whose prefix hit ties; sglang's `cache_aware` picks on
              hit length alone and piles chat onto whichever worker holds the prefix.
            - C1 (1.4%), C2 (0%) and chunk size (0%) all measure zero at cluster scale, for the reasons recorded
              above (row-side kernel gain; the four-mode gate against ~1085 uncached tokens per round).
            - Note this is the PLAIN cache router with a load tie-break, not the SLO-checked `cache_slo` router,
              which loses on these session-affine traces (65.1-68.5% chat). The SLO checks move requests off their
              cached prefix; the cheap tie-break does not.
            (superseded plan) Arms `dchunk` and `sglrt` were the decisive ones: if both land inside the noise, the
            margin belongs to the tree delta and must be traced to a specific mechanism before anything is claimed.
            - **Overload point, alpha 6 (`gap_r1`), complete:**
              | arm | chat | agent | agent tok/s | not issued |
              |---|---:|---:|---:|---:|
              | ours, cache routing + FCFS | 58.0% | 76.2% | 1956 | 6946 |
              | ours, cache routing + EDF | 57.6% | **78.7%** | **2102** | 6913 |
              | stock colocated sglang (upstream tree) | 60.3% | 66.7% | 1292 | 7756 |
              | conditional disaggregation | 58.2% | 53.8% | 725 | 9264 |
              - **Deadline ordering finally separates under overload: +7.5% agent goodput and +2.5 points over
                FCFS**, against nothing at alpha 3.
              - **[!] THE FCFS ARM WAS DESIGNED WRONG (user, 2026-10-28). Do not reuse this comparison.** Both arms
                were given OUR `CHUNK=2048`, so ordering was the only difference, which guaranteed the TPOT result
                and measured nothing. Measured anyway, for the record: chat TPOT p90 109.0 (FCFS) against 111.2 ms
                (EDF), FCFS worse on TTFT (chat p90 8.52 against 7.18 s).
                - **Rule: never give the baseline our chunk control.** The baseline is sglang's own
                  prefill-prioritized (or mixed-chunk) schedule with sglang's own chunk policy.
                - **Rule: our node-level schedule is COUPLED with the chunk rule.** EDF ordering and the
                  SLO-budgeted chunk are one mechanism and must be evaluated together, never as separate ablations
                  of each other.
                - **Rule: no fixed chunk in OUR arm either.** A constant 2048 is an ad-hoc hack with a clean
                  counterexample: at the start of serving, many prefills arrive with few running decodes, there is
                  nothing to protect, and the cap throttles prefill -- starving the decode batch that would have
                  formed and costing throughput. The earlier "chunk size contributes nothing" result at alpha 3 is
                  just what a harmless constant looks like at one load; it does not generalise.
                - Corrected arm set for the next cluster run: baseline = sglang prefill-prioritized + sglang chunk
                  policy; ours = EDF + SLO-budgeted chunk (coupled), swept at alpha 3 and alpha 6. This is C3-worker's regime and the only load where it pays;
                at 128K single-node it costs 8.5% of TPOT and at alpha 3 it is inside the noise.
              - Conditional disaggregation degrades much faster than colocation under overload (53.8% agent against
                76-79%), and issues 9264 fewer rounds: the re-shipped prefill competes with decode exactly when
                there is no headroom. This arm runs OUR router on OUR tree, so it is not affected by the RoPE
                defect and the comparison stands.
              - The stock colocated row (66.7%, 1292 tok/s) carries the wrong RoPE base and is NOT comparable;
                `mn_stockfix_16g_20261028.sh` re-measures it.
          - Remaining differences between the two rows, to be separated: A6 kernel, four-mode step, chunk 2048 vs
            upstream default, our router's load tie-break vs sglang's `cache_aware`.
    - [!!] **INVALID BASELINE: the stock colocated arm ran Qwen2.5-7B with the WRONG RoPE base (2026-10-28).**
          `mn_node.sh` always passes `--json-model-override-args '{"rope_scaling":{...}}'` (YaRN to 126976).
          Upstream sglang REPLACES the whole `rope_scaling` dict, dropping the model's `rope_theta`, so
          `get_rope_config` falls back to **10000 while Qwen2.5-7B needs 1e6**. Our tree preserves it
          (`utils/hf_transformers/config.py`, the only functional difference in that file), so every arm on
          `SGL_PY=/dev/shm/w/up_clean/python` -- the stock colocated baseline -- served a mis-configured model.
          Measured consequence, mean draft acceptance length over all decode batches:
          | arm | tree | mean accept len |
          |---|---|---|
          | stock colocated (`gap_r1_stock`) | upstream `up_clean` | **1.420** |
          | ours (`best_r1_cacheFCFS`) | our tree | **1.991** |
          | ours noC2 (`bdec_r1_noC2`) | our tree | 1.995 |
          With MTP, served throughput scales with acceptance length: 1.991 / 1.420 = **1.40x**, which alone covers
          the **+25% throughput and -45% TPOT we reported over stock colocated sglang**. That comparison is
          **WITHDRAWN** until re-measured; so is the "+25% at equal cache hit" framing built on it.
          - This also explains why C1, C2, the chunk size and the router all measured zero inside our own tree while
            the margin over upstream looked large: the margin was never ours, it was the baseline's broken RoPE.
          - **Not affected:** the single-node H200 A/Bs (the harness copies ONE tree for both arms, so both have the
            fix), the kernel ladder (same tree, same model), and every ours-vs-ours leave-one-out.
          - Fix, no upstream patch needed: carry the base frequency inside the override dict. `mn_node.sh` now
            appends `"rope_theta":$ROPE_THETA` when `ROPE_THETA` is set, so both trees build the same RoPE.
            `mn_stockfix_16g_20261028.sh` re-runs the stock baseline with `ROPE_THETA=1000000` plus our arm as a
            control; log `/tmp/h20/mn_stockfix_16g_r1.log`.
          - **Every past upstream-tree arm is suspect**, including the PD-disaggregation arms
            (`disagg_p*_mtps_*`). Verified 2026-10-28: those arms do pass the same YaRN override and do run
            `SGL_PY=/dev/shm/w/up_clean/python` (mn_tracelab_shift_v2_16g_20261026.sh:50), so they carry the same
            wrong RoPE base. Their decode-side acceptance could not be read from here -- only pod 0's worker logs
            are collected to the head node, and in `disagg_p3` pod 0 holds prefill workers with no decode batches;
            reading it needs an ssh to each decode pod.
            - The disaggregation comparison is **less** distorted than the stock-colocation one, because its gap was
              dominated by an 8x prefix-cache difference (11.2% against ~92%) rather than by acceptance. But the
              acceptance defect pushes the same direction, so the PD numbers overstate our margin by an unknown
              amount and must be re-run with `ROPE_THETA` before they appear in the paper.
            - Arms confirmed CLEAN (acceptance 1.98-2.00, our tree): every `bdec_r1_*`, `best_r1_cacheFCFS`,
              `coloc_slo_edf_mtp_hc_tl2_a3_r1`, `cond_p1_edf_mtp_hc_tl2_a3_best_r1_cond`. Conditional
              disaggregation runs OUR router on our tree, so that arm is valid.
    - [!] **Deployment gotcha that invalidated two re-runs (2026-10-28).** `px_multi.Cluster(sid, cp)` without
          `nodes=` omits `pod_name`, so merlin-cli sends EVERY `run`/`push` to one default instance -- which on this
          job is not even one of `worker-0..3`. Two "corrected" stock re-runs (`ropefix_r1`, `ropefix_r2`) therefore
          ran with the OLD `mn_node.sh` and simply reproduced the original arm (acceptance 1.368 against 1.420).
          Always construct it as `mn_run.py:18` does: `Cluster(SID, cp, nodes=[f"worker-{i}" for i in range(NPODS)])`,
          then verify with `hostname` that the four pods are distinct before trusting a push.
          Also note the ssh entry point (`...worker_0...@merlin-ssh-proxy-cn`) is pod 0, but the merlin-cli DEFAULT
          instance is a different container: files and scorers visible over ssh are not necessarily on the pods.
          `ROPE_THETA` additionally had to be added to `mn_run.py`'s explicit env string (it is not inherited).
          `ropefix_r3` is the first run that actually carries the fix.
    - [!] **Scorer bug: the client's `cached_tokens` is 0 for every request in every arm**, so the agent tenant's
          >200K-uncached TTFT allowance (20 s instead of 10 s) never fires and `uncached` equals the full prompt.
          Attainment is therefore scored slightly PESSIMISTICALLY for long rounds, in every arm equally. Fix before
          the final tables; the worker-log hit rates above are unaffected.
    - [x] **Best configuration, arm 1: cache routing + FCFS workers** (2026-10-27,
          `mn_bestcfg_16g_20261027.sh`, log `/tmp/h20/mn_bestcfg_16g_r1.log`): chat **90.6%** ; agent **94.2%** ;
          **2471 tok/s** ; 11,424 rounds issued; chat TTFT p50/p90 0.07 / 0.67 s, TPOT p50 11.8 ms.
          Against cache routing + worker EDF (the ablation's arm 4: 91.4% / 94.3% / 2514): the two differ by 0.8
          points and 1.7%, **inside the noise floor**. So on this workload the worker policy barely matters and the
          router is the whole effect: cache affinity 90.6-91.4% chat against 65.1-68.5% for the SLO-checked router.
    - [ ] **Best-configuration comparison, remaining arms** (`mn_bestcfg_16g_20261027.sh`, 3 arms, alpha 3), after the
          decomposition:
          - `cacheFCFS`: A6 + 4-mode + `cache` routing + FCFS workers. **No ablation arm covered this combination**
            -- arm 3 removed the worker level with the SLO router still on, arm 4 removed the router with worker EDF
            still on. It is the predicted best.
          - `cacheEDF`: a repeat of arm 4 (the best measured so far), for confidence on 91.4% / 2514 tok/s.
          - `cond`: conditional disaggregation in the same session at Dynamo's defaults. Note it routes through
            `cond_disagg_route`, which never applied our four-check SLO test, so it was never handicapped by the
            mechanism that cost us; the earlier comparison was unfair against us, not for us.
          Until this runs, no best-vs-best claim is restated in the paper.
    - [x] **Decomposition arm 1: EDF ordering only (`PREEMPT=0`)** (2026-10-27,
          `mn_c3worker_decomp_16g_20261027.sh`, log `/tmp/h20/mn_c3worker_decomp_16g_r1.log`): chat 68.8% ; agent
          81.7% ; **1912 tok/s** ; 10,017 rounds issued.
          Placed against the others at alpha 3:
          | arm | chat | agent | goodput | chat TTFT p50/p90 | chat TPOT p50 |
          |---|---|---|---|---|---|
          | full stack (ordering + preemption/parking) | 65.1-68.5% | 72.6-78.7% | 1501-1763 | 0.13-0.15 / 1.03-1.19 s | 27-30 ms |
          | ordering only | 68.8% | 81.7% | 1912 | 0.12 / 4.10 s | 23.7 ms |
          | FCFS | 73.9% | 83.2% | 1922 | 0.11 / 3.03 s | 21.5 ms |
          - **Chunk preemption and parking cost essentially all of the goodput gap**: dropping them recovers
            1501-1763 -> 1912 tok/s, level with FCFS (1922). Deadline ordering itself is nearly free on throughput.
          - What preemption buys is the chat TTFT tail, and it is the only thing that buys it: p90 1.03-1.19 s with
            it, 3.03-4.10 s without (FCFS and ordering-only alike). Under a 2 s TTFT SLO that tail decides roughly
            10% of chat requests, but the TPOT it costs (27-30 vs 21.5-23.7 ms) decides more.
          - So the mechanism is not "EDF is wrong" but "parking a long chunked prefill is too expensive here". That
            is a sharper and more defensible statement for the paper than the wholesale arm-3 result.
    - [!] **CORRECTION (2026-10-27): demotion never ran in either decomposition arm.** The demote code sits inside
          `_edf_rescore_and_park`, which returns at its first line unless `SGLANG_EDF_PREEMPT_CHUNK=1`; both arms ran
          `PREEMPT=0`. Worker logs confirm it: **zero** `EDF_LATE_DEMOTE` lines in `c3d_r1_ord` and `c3d_r1_demote`.
          The two arms were therefore mechanically identical, and their difference is run-to-run variance:
          | two identical configurations | chat | agent | goodput |
          |---|---|---|---|
          | `ord` | 68.8% | 81.7% | 1912 |
          | `demote` | 73.8% | 84.7% | 1986 |
          **So the noise floor on this workload is about 5 points of chat attainment and 4% of agent goodput.** Every
          arm needs >= 2 repeats before it is quoted, and several earlier effects sit at or below this floor.
          - **Withdrawn:** "ordering + demotion beats FCFS". Demotion did not run, and ordering vs FCFS (1912, 1986
            vs 1922) is inside the floor.
          - **Survives:** chunk preemption / parking costs about 25% of goodput -- two no-parking arms (1912, 1986)
            against two with it (1501, 1763), a gap well outside the floor.
          - **Still untested multi-node:** demotion itself; its evidence remains the single-node burst.
          - Second defect in the same block: the deadline **rescoring onto uncached tokens** is also behind that early
            return, so with `PREEMPT=0` both arms ordered by the enqueue priority, which uses the full prompt length.
            A cached agent round was ranked like a cold prompt in both.
          - Fix required before retesting: lift demotion and rescoring out of the parking gate.
            `pd_colocation/apply_edf_model_demote_20261027.py` (model-based demote-on-predicted-miss, user's design:
            never preempt, walk the remaining chunks through the step model under the current decode load and demote
            if the prediction misses) currently inserts into the same gated method and must be restructured first.
            It only evaluates the model forward, so the trained MLP works directly; inversion is needed only for
            chunk sizing, which `edf_compute_max_chunk_size` does by binary search over the same predictor.
    - [~] **Decomposition arm 2 (as run): identical to arm 1, see the correction above**
          (2026-10-27; first multi-node test of demotion). Chat **73.8%** ; agent **84.7%** ; **1986 tok/s** ;
          10,275 rounds issued. Worker policies at alpha 3, SLO router on throughout:
          | worker policy | chat | agent | goodput | chat TTFT p90 | chat TPOT p50 |
          |---|---|---|---|---|---|
          | ordering + preemption/parking (what we shipped) | 65.1-68.5% | 72.6-78.7% | 1501-1763 | 1.03-1.19 s | 27-30 ms |
          | ordering only | 68.8% | 81.7% | 1912 | 4.10 s | 23.7 ms |
          | FCFS | 73.9% | 83.2% | 1922 | 3.03 s | 21.5 ms |
          | **ordering + demotion** | **73.8%** | **84.7%** | **1986** | 2.92 s | 21.7 ms |
          - Demotion recovers exactly what plain ordering lost (chat 68.8 -> 73.8) and costs nothing: it matches FCFS
            on chat and beats it on agent attainment (+1.5 points) and goodput (+3.3%).
          - **So C3's worker level is deadline ordering + demotion, and chunk preemption / parking is a mechanism we
            tried and should drop.** The single-node evidence for demotion (chat TTFT -96..-98% under a burst) and
            this multi-node result agree; the wholesale arm-3 loss was parking's, not ordering's.
          - Still open: all four arms ran with the SLO router on, which arm 4 showed is itself costly. The best
            candidate is cache routing + ordering + demotion, added as a fourth arm to the best-configuration driver.
    - [ ] **C3-worker decomposition, arm 3 (second without-C1 repeat)** (`mn_c3worker_decomp_16g_20261027.sh`, 3 arms, same alpha-3 replay):
          - `ord`: EDF ordering only (`PREEMPT=0`), isolating preemption/parking;
          - `demote`: ordering + demotion, still no preemption -- the first multi-node test of the single-node
            mechanism;
          - `noC1b`: a second without-C1 repeat, to tighten that arm's 7.5-21% goodput range.
          It runs before the chunk A/B, since the C3 finding matters more to the paper.
    - [x] **Audit of the rule against recorded steps** (2026-10-27, `perf_model/chunk_budget_audit.py` on the four
          alpha-3 worker traces, 87,932 steps; user asked whether the implementation is wrong):
          - Decode-only steps cost 8-52 ms (p50 27.7, p90 44.9) and **never** exceed the chat allowance
            (50 ms x acceptance 2.0 = 100 ms per step). So the rule's floor is never the binding term: there is
            48-92 ms of room for a chunk at 1-28 rows.
          - A 2048-token chunk makes the step 446 ms (p90 742): **4.5 chat allowances**, and 98.1% of merged steps
            exceed the allowance. Every resident row waits 4.5x its own SLO budget for one such step. This is the
            mechanism behind chat TPOT p50 72 ms / p90 214 ms under fixed chunks.
          - The chunk the allowance actually buys is 60 ms / (0.099 + 2.908 x 32768 / 1e6) ms per token ~ 300 tokens,
            which is what the rule chose (logged 235-297) and what cut chat TPOT p50 to 47 ms.
          - **The arithmetic is right; the lever is wrong under overload.** Per worker, fixed chunks prefill at
            2048 / 446 ms ~ 4.6K tok/s and the budget at ~300 / 100 ms ~ 3.0K tok/s (-35%). At alpha 6 the replay
            offers ~260K uncached tok/s against 16 x 4.6K = 73K, so the cluster is 3.5-5x past prefill capacity
            either way and a 35% cut turns a deep queue into a much deeper one (TTFT p50 0.4 -> 74 s).
          - At overload the right lever is request-level admission (serve fewer sessions fully), which EDF plus
            demotion already does; a per-step chunk cap keeps every request in the system and misses TTFT for all of
            them. This is the same conclusion as "EDF: order, don't cap".
          - The informative point is therefore alpha 3, where the cluster is near capacity (running), and the TTFT
            floor (v6), which makes the rule serve both SLOs instead of TPOT alone.
        - [ ] The chunk budget still uses lin4mode on GPU time (25%). The patch takes only a linear model; a nonlinear
              model in the patch (or a per-step lookup) is the next step if the chunk arms show mispricing.
      Earlier 4-mode STEPTRACE fits (the 2.0% MAPE in the paper's Table 3 came from an H200 log) must be checked for
      the same misclassification before the paper quotes them.
    - 2026-10-27, also on 16 H20s (queued, `pd_colocation/mn_slo_chunk_16g_20261027.sh`, log
      `/tmp/h20/mn_slo_chunk_16g_r1.log`).
      - Why there: at alpha 6 in the demand-shift replay, ours beats conditional disaggregation on agent goodput
        (890 vs 720 tok/s), but its chat TPOT p50 is 72 ms vs 28 ms with fixed 2048-token chunks.
      - Steps:
        1. ours at alpha 3 with STEP_TRACE;
        2. fit lin4mode on pod 0's workers and copy it to every pod;
        3. ours with `SLO_CHUNK=<model>:0.1:50:80:24000:demote`, `CHUNK=16384` and `SGL_PY=/dev/shm/w/python_slo`
           (the patch applied to a copy of the tree), at alpha 6, alpha 3 and the alpha-3 surge.
      - The patch now handles both scheduler shapes:
        - in our serving tree it runs after `_edf_limit_chunk`, where a zero chunk already means no prefill;
        - in the upstream shape it hooks PrefillAdder;
        - the prefix is read from `kv_committed_len` / `prefix_len` / `_edf_cached_len`.
    - Problem: every serving arm prefills in fixed 2048-token chunks (`CHUNK=2048`, `--chunked-prefill-size`). That
      wastes the step when prefill load is high and decode load is low, e.g. cold 128-256K agent turns with few rows
      decoding.
    - Rule, per step:
      - c* = the largest chunk whose predicted step time keeps every running decode row within its SLO, with headroom
        h: step(c, rows, KV, prefix, acceptance) <= (1 - h) x min_r budget_r;
      - budget_r is row r's remaining mean-TPOT slack, s_r x (tokens_r + acceptance) - elapsed_r, with s_r its tenant's
        SLO (chat 50 ms, agent 80 ms);
      - with no decode rows, c* is bounded only by memory and the activation buffers.
    - Model: the 4-mode step fit (`tasks/perf_model/fit_4mode_step.py`, 2.0% held-out MAPE), refit on A6 steps that
      cover chunks above 2048.
    - Router: price TTFT with the same rule (`router.py --slo-chunk` already walks a prefill chunk by chunk under a
      node's decode load).
    - Prior evidence, to beat:
      - the 8-H20 decode-first rule (`apply_edf_itl_slo_chunk.py`: a 20 ms per-step budget, and its per-row slack
        variant) only ever shrank chunks below 2048, and lost to no control (chat attainment 91.4% -> 65.1%);
      - every per-step cap so far lost (memory: EDF order, don't cap).
      - This rule differs: it grows chunks when slack allows, against per-tenant 50 / 80 ms SLOs instead of 20 ms.
    - Arms:
      - fixed 2048 (today);
      - fixed 8192 and 16384;
      - the SLO-budgeted rule at h = 0 / 10 / 20%.
    - Workloads:
      - prefill-heavy, low decode: 128-256K agents at concurrency 1-2;
      - the Table 7b loads: 128-256K at concurrency 4 + chat 4/s, and 32-64K at 8 + 8/s;
      - ShareGPT 16 / 32 req/s.
    - Record:
      - per-tenant attainment, TTFT, and TPOT p50 / p90 / p99;
      - throughput;
      - the distribution of chosen chunk sizes;
      - predicted vs measured step time.
    - Order: Qwen first, then Llama / GLM.
    - CUDA graphs: chunks above the captured token buckets run without graphs, so record graph coverage per arm.
- **Hardware (before):** H200 job `8574c727fab6ffbd` (vibecuda; restored with `unified/prov_h200_20261024.sh` after
  `34c7bd503d461dc1` ended) for stages 1-2; H20 job `2f608c1b1a67cf09` for multi-node and, meanwhile, kernel runs.

Tasks, in priority order:

- [ ] **A6. Unified fine-grained tile dispatcher + merge schedule (THE kernel contribution).**
      One CPU plan per step routes every row of the mixed batch to FA3 or an FA2 tile class, splits KV, and fixes the
      merge schedule. FA3 and FA2 run as two PDL-chained launches. The FA2 launch reduces every split row, FA2's and
      FA3's, so there is no merge launch. Judged against native v3, which runs the same FA3 kernel: v3's FA2 path is our
      tuned BatchPrefill (Q32, FP32 partials, direct BF16 output, geometry_v3 split).
      Code: flashinfer-private `unified_dispatch_20261024` (port07: `fused_plan.cuh`, `unified_merge.cuh`,
      `prefill.cuh`, `tests/unified_dispatch_ext.cu`); sglang `tasks/debug_kernel_server_perf/resume_20261012/unified/`.

      **Current result: A6 beats native v3 on the H200.** Job `8574c727fab6ffbd`, 132 SMs, 96 fresh trace-sampled
      4-mode MTP captures; flashinfer `cda2acf`; `unified/a6_h200_chain_20261025.sh` (capture, v3 re-bench, build,
      check, stage 1). Every arm passes FP32 + 20 graph replays on all 96 captures (worst 0.0082).
      | stage 1, time-weighted [95% CI] | 32-64K two | 32-64K four | 128-256K two | 128-256K four |
      |---|---|---|---|---|
      | A6 `geometry_fa3` vs native v3 | **1.033x** [1.006-1.051] | **1.047x** [1.035-1.057] | **1.027x** [1.018-1.036] | **1.027x** [1.018-1.041] |
      | A6 `geometry_fa3` vs sglang's FA3 call | 1.48x | 1.43x | 1.51x | 1.44x |
      | A6 `geometry_fa3` vs role-split FA3 (dual split32) | 1.015x | 1.020x | 1.014x | 1.018x |
      | A6 `geometry` vs native v3 | 1.013x | 1.041x | 1.025x | 1.025x |
      | native v3 vs sglang's FA3 call (re-bench) | | 1.36x | 1.47x | 1.40x |
      **Llama-3.3-70B, TP4 on H20 pod 0 (rank 0: 16 q / 2 kv heads), 32-64K** (`unified/a6_llama70b_h20_20261025.sh`;
      38 captures, 14 two-mode + 24 four-mode; all FP32-exact):
      | | 32-64K two | 32-64K four |
      |---|---|---|
      | A6 `geometry_fa3` vs native v3 | **1.009x** [1.004-1.018] | **1.011x** [1.005-1.020] |
      | A6 vs sglang's FA3 call | 1.38x | 1.61x |
      | native v3 vs sglang's FA3 call | 1.37x | 1.59x |
      The 128-256K mixture does not fit Llama's 128K context without YaRN.
      H20 Qwen2.5-7B (job `2f608c1b1a67cf09`, 78 SMs, 92 captures): level with v3 (0.998-1.008x), 1.10-1.23x sglang's FA3 call.
      There FA3 dominates the step and the FA2 path's 3-10% saving (5.30 vs 5.46 ms two-mode, 6.48 vs 7.16 ms
      four-mode FA2 rows) hides under it. H200 FA2-rows two-mode: 5.49 vs 5.02 ms for v3. The four-mode FA2-rows
      harness stops on a capture whose v3 row set differs from its `q_len <= 4` rows; the harness needs fixing, not the
      kernel.

      **Stage 2: serving on the H200** (`unified/a6_stage2_h200_20261025.sh`, root `/opt/tiger/w/a6_stage2_h200_20261025`;
      Qwen2.5-7B + EAGLE3; full stack = EDF with per-tenant deadlines, demote-on-miss and the 4-mode MTP step, against
      stock sglang on FA3; native v3 or A6 as the kernel; two opposite-order repeats; all four roots pass the audit;
      A6 active in 1,883 / 1,265 logged call lines). Relative change against stock, A6 / v3 as the kernel:
      | workload | metric | four-mode | two-mode |
      |---|---|---|---|
      | 128-256K, agent conc 4, chat 4 req/s | chat TTFT | -98% / -98% | -99% / -99% |
      | | agent TPOT | -42% / -38% | -21% / -23% |
      | | agent output tok/s | +4.0% / +4.0% | +10.4% / +10.5% |
      | | agent TTFT | +17% / +31% | -8% / -10% |
      | | chat TPOT | +65% / +72% | +49% / +49% |
      | 32-64K, agent conc 8, chat 8 req/s | chat TTFT | -95% / -92% | -95% / -97% |
      | | agent TPOT | -31% / -40% | -11% / -17% |
      | | agent output tok/s | -9.3% / -10.6% | +0.9% / +0.8% |
      | | agent TTFT | +14% / +27% | -7% / -5% |
      SLO attainment, stock -> ours:
      | workload | chat at 2 s / 50 ms | agent at 10 s / 80 ms |
      |---|---|---|
      | 32-64K | 54-57% -> 95-98% | about +10 points in four-mode, flat in two-mode |
      | 128-256K | 18-21% -> 58-63% | 12-19% -> 37-50% |
      - Data note: the H200 restore copied the raw TraceLab file as `tracelab_unique.jsonl.gz`, where 2026-10-23 used
        the deduplicated set (`deduplicate_trace_rounds_20261023.py`). Arms within this stage share the data, so they
        compare with each other. Deduplication removes 514 of 357,161 records (0.14%), so the two sets are effectively
        the same workload. Runs from S4 on use the deduplicated file.
      - [ ] **Rerun without the deadline chunk cap** (2026-10-26, `unified/a6_stage2_h200_nodc_20261026.sh`, root
            `/opt/tiger/w/a6_stage2_h200_nodc_20261026`; A6 only, serving library `serving_np64_20261026`, FA3_MIN=65).
            On Llama-70B the cap cost throughput (+2-3% -> +9-10% without it). **Done 2026-10-26** (two repeats; A6
            active, 1080 call lines). Ours vs stock, with the capped run in parentheses:
            | workload | mode | agent tok/s | agent TPOT | agent TTFT | chat TTFT | chat TPOT (mean) |
            |---|---|---|---|---|---|---|
            | 128-256K, conc 4 + chat 4/s | four | **+8.1%** (+4.0%) | -39% | +12% | -98% | +68% |
            | | two | **+10.6%** (+10.4%) | -36% | +1% | -98% | +51% |
            | 32-64K, conc 8 + chat 8/s | four | -3.6% (-9.3%) | -44% | +15% | -96% | +48% |
            | | two | +2.7% (+0.9%) | -20% | +2% | -94% | +38% |
            Attainment under the paper SLOs, stock -> ours:
            | workload | mode | chat 2 s / 50 ms | chat 2 s / 100 ms | chat e2e mean | agent 10 s (20 s > 200K) / 80 ms | agent SLO goodput |
            |---|---|---|---|---|---|---|
            | 128-256K | four | 21.0 -> 63.1% | 21.0 -> 94.9% | 27.1 -> 8.1 s | 18.8 -> 43.8% | 16-24 -> 41 tok/s |
            | | two | 18.8 -> 62.8% | 18.9 -> 92.6% | 25.6 -> 8.3 s | 12.5-18.8 -> 50.0% | 14-22 -> 37 tok/s |
            | 32-64K | four | 56.4 -> 97.2% | 56.5 -> 99.6% | 5.8 -> 2.8 s | 70.0-73.3 -> 80.0% | 74-84 -> 85-86 tok/s |
            | | two | 54.3 -> 95.4% | 54.4 -> 99.8% | 5.7 -> 2.8 s | 83.3 -> 83.3-86.7% (loose 100 ms: 90-93 -> 83-87%) | 101-107 -> 88-90 tok/s |
            - Chat attainment (+38 to +74 points) and chat e2e latency (-51 to -70%) clear the 30% bar everywhere.
            - Agent SLO goodput rises 70-190% on 128-256K and is level on 32-64K four-mode.
            - Agent SLO goodput falls 13% on 32-64K two-mode, where deadline order gives chat priority over
              agents that stock had already been serving well.
            - Raw agent throughput clears 10% only on 128-256K two-mode (+10.6%).
            - Sample sizes are small (16 / 30 agent requests per arm). Scripts: `/dev/shm/w/ub/cta.py`
              (`drivers/serving/chat_tail_attainment_20261026.py`), `/dev/shm/w/ub/aatt.py`.
      - **Latency and attainment clear the >= 30% bar against stock on both workloads.**
      - **Raw throughput does not clear the >= 10% bar.** Agent output is -9% to +10.5%, and chat output is pinned at
        the offered rate. SLO goodput does clear it: chat 2 s / 50 ms goodput is ~1,600 vs ~850 tok/s on 32-64K and
        520-540 vs 140-160 tok/s on 128-256K.
      - [x] Throughput at higher agent load (`unified/a6_load_h200_20261025.sh`, 128-256K four-mode, agent concurrency
            8, A6, two repeats). The system is saturated: agent TTFT ~60-64 s in both arms. Agent output is the same
            for ours and stock (49.9 / 50.8 vs 51.0 / 50.8 tok/s). With prefill-bound saturation, neither the kernel
            nor MTP adds single-node throughput, so the >= 10% throughput claim has to come from the multi-node
            schedule (colocation vs PD disaggregation, `pd_colocation_vs_disagg.md`). Concurrency 16 was not run; the
            four-mode-only pair also tripped the audit's both-modes check.
      - [x] S4: step model on the final kernel vs stock's single FA3 call (H200 `8574c727fab6ffbd`, mix32 four-mode,
            agent C=8 + chat 8 req/s, both arms on the same four-mode schedule, A6 active in 407 logged calls; fit on
            repeat 1, scored on repeat 2; `unified/a6_s4_steptrace_h200_20261025.sh`, sglang `a21e5b345`, flashinfer
            `21be81d`). The result file's scope string says H20; the run was on the H200.
            | held-out MAPE, primary model (MLP + causal acceptance) | all | decode | mixed | prefill |
            |---|---|---|---|---|
            | stock single FA3 call (10 673 steps in repeat 1) | 1.87% | 1.17% | 7.66% | 7.54% |
            | A6 (FA3 + FA2, one plan) | 2.03% | 1.42% | **5.92%** | 8.71% |
            - A6's step is as predictable as stock's, so the EDF predictor carries over unchanged. The mixed steps EDF
              prices before admitting a prefill are more predictable (5.9 vs 7.7%). Without acceptance (MLP alone):
              1.79 vs 1.88% all, 6.87 vs 8.08% mixed. GBT is 1.5-1.7% all for both.
            - The same pairs, kernel effect only: agent tok/s 110.6 / 110.0 vs 109.0 / 107.2 (+2%), agent TPOT
              12.93 / 12.94 vs 14.24 / 13.53 ms (-6%), chat TTFT 5.13 / 5.17 vs 5.77 / 5.67 s (-10%).
      - **A6 vs v3:** in both four-mode workloads A6 degrades agent TTFT much less against stock (+17% vs +31%;
        +14% vs +27%) and degrades chat TPOT / ITL less; agent TPOT gains a few points less; two-mode is level. This is
        consistent with a 3-5% faster attention step.

      **Requirements (user, 2026-10-24)**
      - [x] (Mandatory) One unified CPU plan per step for FA3 and FA2.
      - [x] (Mandatory) Fused unified reduction: FA2 and FA3 split partials reduced inside the FA2 launch by the plan's
            merge schedule, no merge launch.
      - [x] A6 beats native v3 on stage 1: H200 1.027-1.047x under `geometry_fa3` (H20: level).
      - [x] FA2 executor: persistent overlap + LPT optimized and compared like for like (2026-10-25, section 3).
            - FA2 rows alone it beats v3's FA2 launch on the H20 (1.018x / 1.076x).
            - In the H200 step it ties the v3-body executor (0.991-1.006x) and beats v3 by 2.4-3.1%.
            - NP stays the default for measured reasons (section 3, Decision).
      - [x] (Optional) JIT reduction (MELD), 2026-10-25.
            - Built as event-driven merge release on the persistent loop (ready queue). Measured 0.982-1.005x vs merges
              queued after compute: the polling slows the remaining compute (section 3).
            - [x] Where the unified merge sits in the full step (H200, 96 captures, FA2 timeline + the stock FA3 grid's
                  start / end; `unified/timeline_step_20261025.py`). Merge time after both FA3 and FA2 compute are done,
                  i.e. what any faster merge could remove:
                  | set | step (sum) | exposed merge | share |
                  |---|---|---|---|
                  | 32-64K two | 11.19 ms | 0.068 ms | 0.6% |
                  | 32-64K four | 14.42 ms | 0.146 ms | 1.0% |
                  | 128-256K two | 31.54 ms | 0.113 ms | 0.4% |
                  | 128-256K four | 43.79 ms | 0.093 ms | 0.2% |
                  - FA3-dominated steps (most of the time): every merge item finishes before FA3 ends, so 0 us is
                    exposed.
                  - FA2-dominated steps (verify / decode rows beside a tiny prefill, FA3 8-58 us): the merge is the
                    step's tail, 2.5-8.2 us of 63-265 us (3-5% of those steps).
                  - FA3 split no row in any capture under `geometry_fa3`, so the unified merge never had FA3 rows to
                    reduce here.
            - The DSMEM-handoff form is not built, for two measured reasons (the ceiling above is 0.2-1.0% of the step):
              - it can remove at most the merge tail, ~5 us per step (timeline);
              - putting a split group in one cluster caps its splits at the cluster size, and pairing CTAs in lockstep
                alone cost 6% on FA3 (A5.5).
      - [x] (Optional) Multicast for FA3 and FA2 (A5.5): FA2 bounded at <1% shared loads; FA3 built, exact, 0.76-0.77x.
            - FA2: under 1% of its KV loads are shared, so multicast is not built.
            - FA3: zero KV loads would make the step only 0.1-1.3% faster; the built 2-CTA exchange runs 0.76-0.77x
              (pairing alone 0.93-0.94x), details in A5.5.
            - [x] Build FA3 multicast under the unified plan anyway (user, 2026-10-25): the plan can pair tiles on a
                  cluster, which upstream's scheduler could not. Measure it against the <= 0.7% bound.
                  - Constraint: serving uses page size 1, so FA3's KV loads are per-row `cp.async` gathers through the
                    page table, and Hopper TMA cannot gather. Upstream flash-attention disables cluster multicast for
                    causal / varlen / paged / split calls, i.e. for every serving FA3 call. So multicast here is a 2-CTA
                    cluster exchange: each CTA gathers half of each KV tile and pushes it to its peer over DSMEM.
                  - [x] Step 1, tile-delivery microbenchmark (`unified/kv_exchange_bench_20261025.{cu,py}`, H200,
                        66 pairs, 128-row bf16 tiles of head dim 128, page-1 gather):
                        - the full tile gathered per CTA, as FA3 does;
                        - vs each CTA gathering a contiguous 16 KB half and one thread bulk-copying it into the
                          peer's stage (`cp.async.bulk.shared::cluster` with `complete_tx`), overlapped with the next
                          gather, with stage reuse guarded by remote consumer releases.
                        The exchanged tiles match the gathered rows exactly.
                        | delivered GB/s (L2-hot / cold) | full gather | half + push, 2 stages | half + push, 4 stages |
                        |---|---|---|---|
                        | 64 tiles | 3886 / 2053 | 2479 / 1825 | **5069 / 3074** (1.30x / 1.50x) |
                        | 256 tiles | 4043 / 2115 | 2519 / 1859 | **5412 / 3216** (1.34x / 1.52x) |
                        - The full gather does not speed up with depth: it is bound by issuing per-row page lookups and
                          `cp.async`s, which halving the rows per CTA halves. The DSMEM push is cheap once it overlaps
                          (4 stages).
                        - So the exchange can deliver FA3's KV tiles ~1.3x faster even when L2-hot. Its step-level
                          value stays bounded by the zero-KV-load ceiling (0.1-1.3%).
                        - Bugs on the way: a first version copied half of each 256 B row, so the pushes never met the
                          barrier's expected bytes and hung. A 16 B-per-thread push without overlap ran 2.7-3.7x slower.
                  - [x] Step 2, integration into flash-attention (`unified/fa3_kv_exchange.patch` on flash-attention
                        e9515d5, `unified/a5_fa3_exchange_h200_20261025.sh`, H200, 96 captures, A6 NP executor):
                        - a pair-aware varlen scheduler: rank 0 takes the pair index and mails it to rank 1 over DSMEM;
                        - both CTAs walk the pair's union causal n-range;
                        - cluster transaction pipelines for K / V;
                        - the half-gather + bulk-push producer, finalized one tile late so the push overlaps the next
                          gather;
                        - a cluster launch.
                        All 96 captures pass FP32 + 20 graph replays in both exchange builds, and the device check finds
                        each half stage contiguous. Bugs on the way: the producer sync used a named-barrier id past
                        hardware id 15, which cutlass offsets again, so it wrapped onto FA3's own barrier and trapped
                        intermittently; and an upstream assert forbade clusters on the paged path.
                        | stage 1 vs control: same tree, no exchange (32-64K two / four, 128-256K two / four) | |
                        |---|---|
                        | control vs the serving FA3 library | 1.001 / 1.002 / 1.003 / 1.001x |
                        | 2-CTA pairing + cluster pipelines, full gather per CTA (no push) | 0.931 / 0.939 / 0.943 / 0.940x |
                        | pairing + half gather + DSMEM push (the exchange) | 0.762 / 0.762 / 0.774 / 0.769x |
                        - Pairing alone costs ~6%: lockstep pairs, half as many independent tiles for FA3's scheduler,
                          and cross-CTA stage releases. That is more than the whole zero-KV-load ceiling (0.1-1.3%), so
                          no multicast built on this pairing can win.
                        - The push adds ~17%: with FA3's 2 K + 2 V stages, finalizing each tile (wait, producer sync,
                          push) delays the tile's arrival, which the 4-stage microbenchmark could hide and FA3 cannot.
                        - Outcome: built, correct, measured negative. Not used.
      - [x] Partials FP32 or BF16 if the FP32 gate passes: ours are BF16, worst error 0.0081.
      - [ ] Serving (stages 2-3) only after the kernel is resolved; v3 is the kernel until A6 beats it.
      - [ ] All three models swept and shown; the paper may keep only Llama-3.3-70B and Qwen2.5-7B.

      **1. Unified plan** (`UnifiedTilePlan`)
      - Routing: FA2 classes 16 / 32 (48 pending, item 6); rows above 32 packed queries go to FA3. FA2-64 was removed:
        on GLM's 48-row verify rows FA3 beats it by 1.02-1.59x (`pd_colocation/results_20261023/glm_captured/row_controls.json`).
      - Host test: 44 real serving geometries, exact coverage, class routing, group and queue invariants, with a
        corrupted-entry null case (`port07/tests/test_unified_plan.cpp`). Plan time 55-97 us mean, 105-378 us worst.
      - KV split policies (`UNIFIED_SPLIT_POLICY`), stage 1 vs v3:
        | policy | FA3 split | FA2 split | result |
        |---|---|---|---|
        | `geometry` (used) | v3's geometry_v3 count | v3's `row_split_size` | best with the final executor |
        | `geometry_fa3` | v3's | fixed | within 0.5% of `geometry` |
        | `fixed` | work / 2 CTAs | power of two >= work / CTAs | under-fills few-row long-KV steps (8192 where v3 uses 4096) |
        | `flashinfer`, `dynamic` | persistent-LPT `kv_len_limit` / makespan search | same | 4-8% behind `fixed` (H200, 2026-10-24) |

      **2. FA3 executor**
      - [x] Stock flash-attention with its own tile scheduler, launched as v3 does, with the plan's split count
            (`UNIFIED_FA3_STOCK=1`, flashinfer `848610b`). This removed a uniform 1-2% loss: our adapter
            (`UpstreamFA3SchedulerAdapter`) ran the same kernel 1.3% slower on FA3-bound steps (per-launch profile,
            `profile_step_20261025.py`).
      - [x] L2-swizzled tile order for the adapter: no change (within the CI, H200).

      **3. FA2 executor**
      - [x] v3's BatchPrefill body, non-persistent, both classes in one launch (`UNIFIED_FA2_NP=1`). v3's Q32 layout
            (2x2 warps, V2 merge-union barrier) and `independent_pdl` are ported into port07's `prefill.cuh`.
      - [ ] Persistent overlap + LPT kernel (the intended executor, being optimized).
        - KV depth is not its gap. Its 16-row class uses MMA_KV 2, as v3's 16-row kernel does.
        - Raising it to MMA_KV 4 widens the per-stage KV tile to 256 keys (1 q warp x 4 kv warps x 16 x 4), i.e.
          128 KB of K+V shared memory. That leaves one CTA per SM, and FA2 rows lose 26-36%.
        - [x] Brought to parity with the v3-body executor's fixes (chunk-parallel merge items, stock FA3, atomic
              resets; flashinfer `a79b31b`, `unified/unified_h20_persistent_20261025.sh`; FP32 + 20 graph replays pass on
              all 92):
              | executor, `geometry` | FA2 rows two-mode | FA2 rows four-mode | stage 1 vs v3 (32-64K two / four, 128-256K two / four) |
              |---|---|---|---|
              | v3's FA2 launch | 5.46 ms | 7.17 ms | |
              | persistent overlap + LPT | 5.82 ms (0.94x) | 6.92 ms (1.035x) | 0.998 / 1.005 / 0.999 / 1.003x |
              | v3 body (non-persistent) | 5.31 ms (1.03x) | 6.49 ms (1.10x) | 0.998 / 1.008 / 0.999 / 1.002x |
              With like-for-like fixes the persistent kernel beats v3 on four-mode FA2 rows and matches it on stage 1.
              On FA2 rows it is still 7-10% behind the v3 body.
        - [x] **The remaining 7-10% is the merge, not the runner** (`unified/unified_h20_compute_bound_20261025.sh`;
              merge-free bounds, output wrong):
              | compute only | two-mode | four-mode |
              |---|---|---|
              | persistent overlap + LPT | 4.72 ms (1.16x v3's FA2 launch) | 6.19 ms (1.16x) |
              | v3 body | 5.11 ms (1.07x) | 6.24 ms (1.15x) |
              - The persistent runner + schedule is the faster FA2 executor: 8% ahead of the v3 body on two-mode
                decode rows.
              - Its merge costs 1.10 / 0.73 ms, against 0.20 / 0.25 ms for the v3 body. Per step (47 rows at 58K KV):
                20 us against 2.4 us.
              - Cause: merge items sit after every compute item in the one LPT queue, so they run as a tail of about 2-3
                dependent rounds once compute has drained. The v3 body's merge CTAs are separate blocks that the
                hardware starts while the compute tail runs.
              - Measured negatives:
                - merge-item size (1-32 rows): 4 stays best;
                - interleaving merge items into the queue: 6.14-6.80 / 7.60-9.37 ms, because it breaks LPT order;
                - last-arriver merge in the runner epilogue: 5.92 / 7.97 ms.
        - [ ] **Ready-queue merge** (`UNIFIED_MERGE_MODE=ready`, `unified/unified_h20_persistent_ready_20261025.sh`).
              - The member that completes a group publishes that group's merge items into a separate ready queue.
              - CTAs drain published merges (claimed by CAS) before pulling more LPT compute, so merges run as soon as
                their group is complete, with no group spinning, and the compute order is kept.
              - Producers only reserve and write codes, the data-driven hand-off of MELD's JIT reduction.
              - Status: hangs intermittently. Each per-set check reached its 20-minute limit after 5-9 captures, all
                of which passed, and the capture it stalled on passes when run alone. So it is a race in the claim /
                publish protocol, not a deterministic bug.
              - Root cause, found with a watchdog build that prints the queue state and traps
                (`unified/unified_h20_persistent_ready_debug_20261025.sh`):
                - Every hang lost exactly one group: `head=22 tail=22 total=24`, all compute pulled. It reproduced with
                  PDL off (`UNIFIED_PDL=0`), so it is not launch overlap.
                - In the runner's check-in hook, every thread wrote `unified_last = 0` while thread 0 still had to read
                  the flag to decide whether to publish. When another warp's write landed first, the group was never
                  published.
                - Fixed (flashinfer `adcaaef`): the flag is read into a register after the sync. The JIT branch used the
                  same pattern and was fixed the same way.
              - [x] 2026-10-25 After the fix (`unified/unified_h20_persistent_ready2_20261025.sh`): FP32 + 20 graph
                    replays pass on all 92, the watchdog never fires, and 3 FA2-rows repeats agree within 0.01 ms.
                    | persistent merge | FA2 rows two-mode | FA2 rows four-mode | stage 1 vs v3 (32-64K two / four, 128-256K two / four) |
                    |---|---|---|---|
                    | merge items at the queue tail | 5.82 ms | 6.92 ms | 0.998 / 1.005 / 0.999 / 1.003x |
                    | ready queue | 5.70 ms | 7.66 ms | 0.996 / 1.001 / 1.002 / **1.006x** [1.005-1.008] |
                    - Two-mode improves.
                    - Four-mode gets worse: a 28-row group yields 7 merge items, and taking them ahead of compute
                      delays the longest LPT compute items that set the step's tail.
                    - [x] Merge-priority threshold (`UNIFIED_READY_FRAC`, flashinfer `21be81d`,
                          `unified/unified_h20_persistent_readyfrac_20261025.sh`): every value, 0 to 1, gives
                          5.85-5.87 / 8.26-8.32 ms, slower than the plain ready queue, including 0, which should
                          match it. The one change was an extra L2 atomic read of the compute cursor per pull, so a
                          pull's atomic count is itself a first-order cost of this loop (four L2 atomics per pull). FP32
                          check passes at 0.75.
                    - [x] Polls as coherent L2 loads (`__ldcg`), atomics only for claims
                          (`unified/unified_h20_persistent_readyfrac2_20261025.sh`):
                          - 5.81-5.84 / 8.12-8.15 ms at threshold 0 / 0.75 / 1. Better than with atomic polls, still
                            slower than the plain ready queue, and the threshold value barely matters.
                          - The threshold check is off by default now (it runs only when `UNIFIED_READY_FRAC > 0`).
                          - FP32 + graph checks pass and the watchdog never fires.
        - [x] **Per-item timeline** (`UNIFIED_TIMELINE=1`, globaltimer per item; flashinfer `6a608d9`,
              `unified/unified_h20_persistent_timeline_20261025.sh`, `timeline_fa2_rows_20261025.py`):
              - Most steps are 1-4 compute rounds of 31-43 us items (144-680 items on 156 CTAs), then one merge round.
              - A 4-row merge item takes 6-7 us. Its rows have ~18 chunks, so the time is a chain of dependent round
                trips (plan indices, partial loads, fence, counter atomic), not bandwidth.
              - Ready queue: a pull costs ~2 us against 0.7 us at the queue tail, and per-CTA idle time doubles on
                four-mode (1.04 vs 0.56 ms summed). This is why the ready queue loses on four-mode.
              - Long steps (128-256K, 46 rows): item times spread from small to 159 us. The longest item, not the
                merge, sets the step.
        - [x] **Batched online-max merge rows** (`UnifiedMergeRowCG`, both executors; flashinfer `0ab3886`,
              `unified/unified_h20_merge_unroll_20261025.sh`). Each chunk group keeps its own max and has 4 chunks in
              flight, with no separate max pass. FP32 + 20 graph replays pass on all 92 for every arm.
              | FA2 rows (two / four-mode) | before | after | stage 1 vs v3 after (32-64K two / four, 128-256K two / four) |
              |---|---|---|---|
              | v3's FA2 launch | 5.47 / 7.17 ms | | |
              | persistent, merge items at the queue tail | 5.82 / 6.92 | **5.42 / 6.98** | 0.998 / 1.006 / 1.000 / 1.008x |
              | persistent, ready queue | 5.70 / 7.66 | 5.62 / 7.66 | 0.996 / 1.000 / 0.999 / 1.002x |
              | v3 body (NP) | 5.31 / 6.49 | **5.23 / 6.42** | 0.999 / 1.008 / 1.000 / 1.006x |
              - The persistent kernel now beats v3's FA2 launch on both modes (1.009x / 1.027x). It still trails the
                NP executor by 4% / 9%, and the gap holds on every four-mode capture, including 1-row steps.
              - A merge item still takes ~6 us, so the gain does not come from the merge loop; the build's register
                allocation also changed. The open cause is the runner configuration (next item).
        - [ ] **Runner configuration vs v3's** (H20; `unified/unified_h20_persistent_occ_20261025.sh`,
              `unified/unified_h20_persistent_smem_20261025.sh` with flashinfer `d5944d7`; FA2 rows two / four-mode):
              | 32-row KV depth | 2 CTAs/SM | 3 | 4 |
              |---|---|---|---|
              | MMA_KV 2 (64-key tile) | 5.42 / 6.99 ms | 5.41 / 6.98 | 5.41 / 6.99 |
              | MMA_KV 4 (128 keys, v3's Q32) | 5.71 / 6.66 | 5.72 / 6.66 | 5.71 / 6.65 |
              - MMA_KV 4 on the 32-row class gives four-mode 6.65 ms (1.077x v3's FA2 launch) but two-mode 5.71 ms, so
                neither depth wins both. FP32 + 20 graph replays pass on all 92 at MMA_KV 4. Stage 1 vs v3:
                0.997 / 1.006 / 0.998 / 1.003x.
              - The CTA count does nothing: the kernel holds **2 CTAs per SM at every smem size** (occupancy query:
                128 threads, 68 KB for the 16-row class, 40 / 72 KB for the 32-row class at MMA_KV 2 / 4).
                `__launch_bounds__(128, 1)` leaves the compiler up to 255 registers, and that, not shared memory,
                caps residency (3 x 68 KB fits in 228 KB).
              - Per-plan dynamic smem (16-row-only plans launch with the 16-row class's size; the kernel traps on
                an undersized launch) is built but changes nothing while registers bind.
              - [x] `UNIFIED_FA2_MIN_BLOCKS=3` (3 CTAs per SM confirmed by the occupancy query;
                    `unified/unified_h20_persistent_minblocks_20261025.sh`): MMA_KV32 2 gives 5.36 / 7.89 ms, MMA_KV32 4
                    gives 5.57 / 7.81 ms (spills 76 / 192 B). Two-mode improves to **1.019x** v3's FA2 launch; four-mode
                    loses 13-18% (its 32-row items need the registers). FP32 + 20 graph replays pass on all 92. Stage 1
                    vs v3: 1.000 / 1.007 / 0.999 / 1.006x.
              - [x] **Plan-chosen instantiation** (flashinfer `22d8d03`, `unified/unified_h20_persistent_dual_20261025.sh`):
                    16-row-only plans launch a 3-CTA/SM instantiation with the wide runners compiled out (with them
                    compiled in it ran 5.55 ms); plans with 32-row items launch the 2-CTA build at MMA_KV32 4. The
                    executor sizes the LPT grid to match (`UNIFIED_FA2_CTAS_PER_SM16=3`). FP32 + 20 graph replays
                    pass on all 92.
                    | FA2 rows | two-mode | four-mode |
                    |---|---|---|
                    | v3's FA2 launch | 5.46 ms | 7.17 ms |
                    | persistent overlap + LPT, plan-chosen | **5.37 (1.018x)** | **6.66 (1.076x)** |
                    | v3 body (NP) | 5.23 (1.044x) | 6.42 (1.117x) |
                    Stage 1 vs v3 (H20): 1.000 / 1.006 / 0.999 / 1.005x. On the H20 the FA2 launch runs under FA3, so
                    the FA2 gain does not reach the step.
              - Remaining gap to NP (2.6% / 3.7%): the persistent kernel's compute alone is faster than NP's (merge-free
                bounds, above), so the gap is in the merge round trips and per-pull cost (timeline).
              - [x] **H200 stage 1, persistent (plan-chosen) vs NP under the same plan** (job `8574c727fab6ffbd`, 96
                    fresh captures, `geometry_fa3`, flashinfer `22d8d03`, `unified/a6_h200_pers_20261025.sh`). FP32 + 20
                    graph replays pass on all 96 for both.
                    | stage 1 (32-64K two / four, 128-256K two / four) | vs native v3 | vs sglang FA3 |
                    |---|---|---|
                    | NP executor, batched merge | 1.036 / 1.041 / 1.023 / 1.027x | 1.48 / 1.42 / 1.50 / 1.44x |
                    | persistent overlap + LPT, plan-chosen | 1.028 / 1.031 / 1.029 / 1.024x | 1.47 / 1.41 / 1.51 / 1.44x |
                    | persistent vs NP | 0.993 / **0.991** [0.984-0.997] / 1.006 / 0.997x | |
                    - Both beat v3 by 2.3-4.1%. The persistent kernel is within 1% of NP on every set: NP wins on
                      32-64K four-mode (CI below 1), and the persistent kernel is slightly ahead on 128-256K two-mode.
                    - FA2 rows alone, two-mode: persistent 5.34 ms, NP 5.56, v3's FA2 launch 5.02. On the H200 v3's
                      launch is faster alone, but inside the step both A6 executors win, because the step is FA3-bound
                      and A6 drops v3's merge launch. The four-mode FA2-rows harness stops on these captures: v3 sends
                      a different row set to its FA2 launch (harness limit, not a kernel failure).
              - [ ] **The gap to NP is unexplained (unchecked 2026-10-25, user).**
                - Both executors run the same merge (plan-scheduled merge items, `UnifiedMergeRowCG`, tickets after
                  compute), so "the merge" cannot be the difference. The "persistent compute alone is faster" bound
                  was taken in different modes for the two executors, on an older persistent build.
                - Candidates: who runs a merge item (a resident hardware-launched CTA vs a persistent CTA after its
                  compute item), residency (2-3 vs NP's CTAs per SM), and per-pull queue atomics.
                - [x] Same per-item timeline in both executors, same 96 H200 captures, same ticketed merge items
                      (`unified/a6_h200_timeline_np_vs_pers_20261025.sh`, flashinfer `22d8d03` + NP timeline):
                      | in-kernel span, summed | NP | persistent |
                      |---|---|---|---|
                      | two-mode | 5.38 ms | **5.23 ms** |
                      | four-mode | 4.44 ms | **4.38 ms** |
                      | tail after the last compute item (two / four) | 0.27 / 0.28 ms | 0.23 / 0.23 ms |
                      | merge item duration | 4-6 us | 4-6 us |
                      - The merge is the same, and the persistent schedule is not behind.
                      - The per-item compute time differs, e.g. 45.9 vs 51.8 us on a 46-row step. The two executors
                        run different tile code: NP runs v3's BatchPrefill body, the persistent kernel runs
                        flashinfer's persistent runner at a capped register budget.
                      - Both walk the same plan queue. It is one FA2 queue, longest first; NP's tickets follow it in
                        launch order. So the schedule (LPT order, splits, merge items, PDL overlap) is common to both,
                        and the earlier comparisons mixed dispatch with tile code.
                - [x] **Dispatch, tile code and schedule separated** (H200, 96 captures, geometry_fa3, flashinfer
                      `905660d`, `unified/a6_h200_dispatch_schedule_20261025.sh`). FP32 + 20 graph replays pass on all 96
                      for the new build.
                      | stage 1 vs the NP executor (32-64K two / four, 128-256K two / four) | |
                      |---|---|
                      | NP rerun (noise floor) | 0.998 / 1.001 / 1.003 / 1.000x |
                      | same BatchPrefill body on a persistent LPT pull loop (2 CTAs/SM) | 0.987 / 0.998 / 0.996 / 0.993x |
                      | enumeration order instead of longest first | 0.993 / 0.996 / 1.001 / 1.001x |
                      | FA2 launch serial after FA3 (PDL off) | **0.939 / 0.967 / 0.975 / 0.978x** |
                      | flashinfer's persistent runner vs the persistent BatchPrefill loop | 1.005 / 0.993 / 1.011 / 1.004x |
                      - Overlapping FA2 with FA3 is worth 2.2-6.1% of the step. That is the schedule's measurable
                        contribution, larger than A6's whole lead over v3 (2.3-4.1%).
                      - Longest-first order is worth at most 0.7%, inside the noise on three of the four sets.
                      - Persistent dispatch gains nothing over hardware-launched CTAs at equal tile code (-0.2..-1.3%).
                - [x] **Persistent loop + JIT reduction** (user, 2026-10-25; flashinfer `bfa7e2a`,
                      `unified/a6_h200_npp_jit_20261025.sh`, H200, 96 captures).
                      - The member that completes a split group publishes the group's merge items into a ready queue,
                        and CTAs take published merges before further compute (claimed by CAS, `__ldcg` polls).
                      - FP32 + 20 graph replays pass on all 96; the watchdog never fires.
                      | stage 1 (32-64K two / four, 128-256K two / four) | |
                      |---|---|
                      | JIT (ready) vs merge items after compute (ticket), same persistent loop | 0.997 / **0.982** / 1.003 / 1.005x |
                      | JIT vs the hardware-launched NP executor | 0.982 / 0.979 / 0.997 / 0.995x |
                      | timeline, summed over captures (two / four-mode) | ticket | JIT |
                      |---|---|---|
                      | tail after the last compute item | 0.24 / 0.28 ms | 0.37 / 0.77 ms |
                      | pull time | 0.05 / 0.05 ms | 0.15 / 0.27 ms |
                      | compute | 3.72 / 3.16 ms | 3.92 / 3.41 ms |
                      - Measured reason it loses: idle CTAs polling the ready queue add atomics and L2 traffic that slow
                        the compute still running, and the tail grows instead of shrinking.
                      - Its ceiling is small: it can only remove the ticket-mode tail, ~5 us per step, a few percent of
                        the FA2 rows, and most of that runs under FA3 in the full step.
                      - Outcome: neither persistence nor JIT release beats hardware-launched CTAs at equal tile code.
                        The kernel's schedule contribution is the plan's FA2 / FA3 overlap (2.2-6.1% of the step), plus
                        the plan and the in-launch merge.
              - **Decision so far** (the NP executor is the default):
                - It leads or ties on stage 1 on both GPUs: H20 +0.0-0.3%; H200 +0.3..+0.9% on three sets and -0.6% on
                  one.
                - The persistent kernel's compute alone is faster, but it loses that lead to its merge round trips
                  (~6 us per merge item, a chain of dependent global accesses) and its per-pull queue atomics.
                - It also needs two register-targeted instantiations per plan to reach parity.
                - It remains the measured alternative executor (`UNIFIED_FA2_NP=0`).
      - [x] Per-row tile classes in four-mode (2026-10-25, closed by construction): in the 4-mode MTP schedule every
            running request verifies 4 tokens, and two-mode steps carry only q=1 rows. A step never mixes decode and
            verify rows, so per-row classes inside a step have nothing to separate.

      **4. Fused in-launch reduction**
      - [x] Plan-scheduled merge items: each group is cut into 4-row items. A CTA takes a ticket; compute tickets come
            first, so a merge item waits only on members that are already running.
      - [x] FA3 split rows as merge items of the same schedule, gated on the FA3 grid (no combine launch).
      - [x] Chunk-parallel merge: 8 chunk groups x 4 column groups per warp, 16-byte loads, three shuffle rounds. The
            serial per-row loop had made our merge cost 21-28 us per step against v3's 3-8 us merge kernel. Our compute
            alone is faster than v3's (20.6 vs 32.0 us on a 3-row step: BF16 partials).
      - Measured negatives:
        - Last-arriver merge with no tickets: worse on four-mode (one CTA reduces a 28-row group).
        - Interleaving merge items behind their group's compute (`UNIFIED_MERGE_LAG`): slower at every lag.

      **5. SM pre-assignment** (measured negative, `unified_h20_smbudget_20261025.sh`, `sm_oracle_20261025.py`)
      - [x] FA3 grid capped at 70 / 62 / 54 / 46 of 78 SMs, FA2 starting on the free SMs:
        - Fixed caps lose: 0.86-0.89x of v3 at 70 SMs, 0.60-0.62x at 46.
        - Even a per-step oracle over the caps gains under 0.1% over full-SM FA3 plus PDL co-run (0.999 / 1.008 /
          0.999 / 1.002x). The steps a smaller cap wins are short.

      **6. Open**
      - [ ] **Rows of 33-64 packed queries (GLM-4.5-Air's 4-token verify rows pack 48)** (unchecked 2026-10-25, user).
            - The FA3 arm below ran FA3 at its 128-row tile, which pads 48-row rows 62%, so the FA2-64 win was never
              ablated against FA3 at its 64-row tile (wgmma's smallest M). FA3 cannot run 48; FA2 can.
            - Deferred (user, 2026-10-25): Qwen and Llama come first, and GLM is the only model with these rows. The run
              already in flight is left to finish and recorded; no further work on it until the Qwen / Llama kernel
              items are done.
            - [ ] Ablation (`unified/glm_tile_ablation_h20_20261025.sh`, H20 pod 4, 43 GLM captures, running), with the
                  verify rows alone and stage 1 per arm:
                  - FA3 q tile 128;
                  - FA3 q tile 64 (`FLASHINFER_FA3_BLOCK_M=64` in the stock FA3 build);
                  - FA2 64-row class;
                  - FA2 48-row class (1q x 4kv, 3 MMAs per warp).
            - [x] GLM stage 1 with these rows on FA3 (`UNIFIED_FA3_MIN=33`; H20 pod 0, TP4, rank-0 attention, 43
                  trace-sampled captures with NEXTN MTP; `unified/a6_glm_h20_20261025.sh`, flashinfer `22d8d03`).
                  FP32 + 20 graph replays pass on all 43.
                  | stage 1, GLM-4.5-Air | 32-64K two | 32-64K four |
                  |---|---|---|
                  | native v3 vs sglang FA3 | 1.28x | 1.41x |
                  | A6 vs native v3 | 1.005x | **0.724x** [0.540-0.830] |
                  - A6 loses 28% on four-mode because it sends the 48-row verify rows to FA3, where v3 runs them on
                    FA2 BatchPrefill's 64-row tile.
                  - The earlier row control (FA3 1.02-1.59x faster than a 64-row FA2 class) measured our old persistent
                    FA2 runner, not v3's BatchPrefill tile.
            - [x] **A 64-row class in the NP executor** (flashinfer `396b04b`, `unified/a6_np64_glm_h20_20261025.sh`):
                  v3's BatchPrefill 64-row tile, 4q x 1kv warps, NUM_MMA_KV 8 (flashinfer's dispatch choice). It is
                  launched as a separate instantiation for plans with 64-row items (that one spills 88 B; the 16 /
                  32-row instantiation is unchanged). With `UNIFIED_FA3_MIN=65`, rows route as in v3. FP32 + 20 graph
                  replays pass on all 43 GLM and all 92 Qwen captures.
                  | stage 1 vs native v3 | 32-64K two | 32-64K four | 128-256K two | 128-256K four |
                  |---|---|---|---|---|
                  | GLM-4.5-Air, 48-row rows on FA3 (before) | 1.005x | 0.724x | | |
                  | GLM-4.5-Air, 64-row NP class | 0.998x | **1.000x** | | |
                  | Qwen2.5-7B, same build (regression) | 0.999x | 1.003x | 0.998x | 0.999x |
                  vs sglang FA3 with the 64-row class: GLM 1.28x / 1.41x; Qwen 1.19 / 1.22 / 1.19 / 1.09x.
            - The persistent kernel's co-compiled 48 runner spilled every class (1 436 B); NP avoids this by
              instantiating per plan.
      - [x] **Open kernel gaps (2026-10-25, user): all closed 2026-10-26.** New defaults: tile-rounded FA2 chunk. Every
            other item is measured, with its mechanism, below.
        - [ ] **DSMEM-cluster JIT reduction** (`UNIFIED_MERGE_MODE=dsmem`, `UnifiedFA2NPClusterKernel`, flashinfer
              `f62b92a`, `unified/a6_h200_dsmem_jit_20261025.sh`, H200, 96 captures). It is exact: FP32 + 20 graph
              replays pass on all captures at C=2 and C=4. It is slow:
              FP32 + 20 graph replays pass on all 96 captures at every C.
              | stage 1, vs ticketed merge items | 32-64K two | 32-64K four | 128-256K two | 128-256K four |
              |---|---|---|---|---|
              | C=2 | 0.597x | 0.667x | 0.622x | 0.703x |
              | C=8 | 0.925x | 0.927x | 0.913x | 0.940x |
              | C=16 (non-portable) | 0.967x | 0.984x | 0.967x | 0.978x |
              vs native v3: C=2 0.62-0.72x, C=4 0.81-0.87x, C=8 0.93-0.97x, C=16 0.989-1.025x.
              - Cause (from the plan): to fit a split group in one cluster, the plan raises the FA2 chunk to
                >= kv_max / C. At C=2 the longest request runs as two chunks and bounds the step. The loss shrinks
                monotonically as C relaxes the cap, but it never passes the ticketed merge. The cluster barrier also
                holds every CTA of a task until its slowest entry is done.
              - Ceiling: with every merge, fence and atomic removed, the step is at most 0.9% faster on these
                captures (merge-free bound), so no reduction scheme can win more than that here. Synthetic
                MELD-shaped batches (`unified/a6_h200_synth_meld_20261025.sh`, running) test where the reduction
                costs more.
              - [x] **v2, split count independent of C** (flashinfer `24c0545`; the old cap is
                    `UNIFIED_DSMEM_CAP_CHUNK=1`; `unified/a6_h200_dsmem2_20261026.sh`, H200, 2026-10-26).
                    - Design: the chunk comes from the normal policy. Groups of <= C members reduce inside their
                      cluster after the cluster barrier; a member of a larger group checks in on the global counter,
                      and the last arrival reduces the whole group (no waits).
                    - Gate: FP32 + 20 graph replays pass at C = 4 and 8 on all 96 captures and six synthetic sets.
                    - Results vs the ticketed merge, same build:
                      | set | C = 4 | C = 8 | C = 16 |
                      |---|---|---|---|
                      | captures (32-64K two / four, 128-256K two / four) | 0.993 / 0.988 / 0.992 / 0.993x | 0.974-0.990x | 0.955-0.983x |
                      | ShareGPT two, bs 512 / 1024 (zero split groups) | 0.994 / 0.970x | 0.964 / 0.935x | 0.928 / 0.892x |
                      | ShareGPT four | 0.946-0.973x | 0.929-0.954x | 0.900-0.930x |
                      | shared 4K prefix | 0.71-0.90x | 0.70-0.89x | 0.69-0.86x |
                      | 48K shared prefix | 0.38-0.76x | 0.38-0.77x | 0.38-0.77x |
                      | long tail / math | 0.74-1.02x | 0.73-1.01x | 0.75-0.97x |
                    - v1's chunk coupling is gone (captures 0.60-0.87x -> 0.988-0.993x at C <= 4), but two costs
                      remain:
                      1. **Gang scheduling.** A cluster's CTAs launch together and hold their SMs until the slowest
                         member ends. With items of different lengths in one task this breaks longest-first balance;
                         it costs 1-11% even with nothing to reduce (ShareGPT two, bs 512 / 1024), growing with C.
                      2. **Groups larger than a cluster** are reduced by one CTA. The plan-scheduled merge spreads a
                         big group's rows over many CTAs; the last-arriver reduction serializes them (48K shared
                         prefix, 0.38-0.77x).
                    - **Outcome: closed as a measured negative.** On these workloads the in-launch, plan-scheduled
                      merge beats both cluster designs. A cluster reduction would need tasks of equal-cost items and
                      a parallel reduction for big groups, which is what the ticketed merge already provides without
                      gang scheduling.
        - [x] **Synthetic MELD-shaped steps (2026-10-25, H200, Qwen2.5-7B geometry, every arm FP32-gated;
              `unified/gen_synth_meld_20261025.py`, `unified/a6_h200_synth_meld_20261025.sh`, flashinfer `f62b92a`).**
              One 512-token prefill chunk plus bs decode (two) or verify (four) rows. Values are A6-default time /
              arm time (> 1 = the arm is faster):
              | set, mode | bs | native v3 | merge-free bound | persistent loop, same tile | persistent runner (3 CTAs/SM decode) | DSMEM C=8 |
              |---|---|---|---|---|---|---|
              | ShareGPT, two | 64 / 256 / 1024 | 1.01 / 0.95 / 0.94 | 1.13 / 1.06 / 1.01 | 0.93 / 0.95 / 0.98 | 1.03 / **1.12** / **1.08** | -- |
              | ShareGPT, four | 64 / 128 / 256 | 0.91 / 0.85 / 0.83 | 0.85 / 0.86 / 0.74 | 0.91 / 0.97 / 0.89 | 0.91 / 1.00 / 0.93 | -- |
              | math, two | 16 / 64 / 256 | 1.04 / 1.05 / 0.92 | 1.06 / 1.05 / 1.01 | 0.97 / 0.95 / 0.97 | 0.98 / **1.12** / 1.03 | -- |
              | shared 4K prefix, two | 64 / 128 / 256 | 0.97 / 0.96 / 0.91 | 1.07 / 1.10 / 1.08 | 0.91 / 0.93 / 0.92 | 1.11 / 1.05 / **1.14** | -- |
              | long tail (10% at 64-200K), four | 16 / 64 / 128 | 1.04 / 1.13 / 1.02 | 0.82 / 0.71 / 0.75 | 0.91 / 0.89 / 0.90 | 0.96 / 0.94 / 0.99 | 0.40 / 0.78 / 0.67 |
              - **Merge cost:** up to 13% of the step on two-mode decode batches (bs 64; 6-10% at bs 128-256), against
                <= 0.9% on the captures. This is where a cheaper reduction pays.
              - **The persistent runner wins at large decode batch (1.08-1.14x), but not because of CTA launch
                overhead.** The persistent pull loop on the SAME tile code is slower (0.89-0.98x), so hardware CTA
                launch of thousands of CTAs is not the cost. The win is occupancy: its decode-only instantiation runs
                3 CTAs/SM; the NP kernel runs 2 (register-bound, `UNIFIED_NPP per_sm=2`).
                -> decode-only NP instantiation built (flashinfer `64f40c7`, below).
              - **Load imbalance exists at large batch, and the split policy causes it** (per-capture FA2 balance,
                `plan_balance_20261025.py`). The FA2 chunk grows with batch (2048 at bs 64, 32768 at bs 1024), so long
                rows go unsplit. Four-mode ShareGPT bs 1024: one 32K row runs 786 us against a balanced 634 us
                (1.24x). bs 64: span 1.35-1.47x the balanced bound (tail).
                -> tile-rounded chunk built (flashinfer `5797024`, below).
              - A6 beats v3 by up to 1.21x on four-mode ShareGPT (bs 256), 1.15x on shared prefix and 1.09x on math
                two-mode at bs 256. v3 is faster on long-tail four-mode at bs 64 (A6 0.89x) and on math two-mode at
                bs 16 / 64 (A6 0.96 / 0.95x).
              - **The four-mode merge-free bound was a code-generation artifact (resolved 2026-10-26).**
                - Interleaved control (`unified/a6_h200_nosync_control_20261026.sh`, both builds, graph replay, two
                  repeats): no-sync is 0.71-0.86x on four-mode ShareGPT / long tail. This holds even at bs 512 / 1024,
                  where the plan has zero split groups and zero merge items, so the merge path never runs.
                - Cause: compiling the check-in branch out changes how ptxas allocates the 32-row body. The NP kernel
                  is 255 registers + 16 B spill with merges vs 209 registers, no spill without; both fit 2 CTAs/SM,
                  but the 209-register schedule runs the four-mode body slower.
                - Valid four-mode bound: the per-item timeline pair, whose builds match in codegen (40-44 B spill
                  each; `unified/a6_h200_nosync_probe_20261025.sh`). Merges cost 0.8% of the step on 32-64K
                  four-mode captures (14.43 vs 14.31 ms) and ~0 on synthetic four-mode.
                - Two-mode is unaffected (bs 512 / 1024 with no merges: 0.99-1.00x), so the 4-13% two-mode merge cost
                  at bs 64-256 is real.
                - Lesson: a bound built by deleting code is only valid if the kernel's register allocation is
                  unchanged; check ptxas output first.
              - The baseline pass on four-mode bs >= 512 overflowed the native harness's 512 MB workspace; it is
                rerunning with growth to 4 GB (`unified/a6_h200_synth_base_rerun_20261025.sh`).
        - [x] **Decode-only NP instantiation** (flashinfer `64f40c7`: 32-row body compiled out,
              `__launch_bounds__(128, 3)`, 16-row smem, chosen by the plan when every FA2 item is 16-row; 28 B spill;
              `unified/a6_h200_np16_20261025.sh`, H200, 2026-10-26).
              - Gate: FP32 + 20 graph replays pass on all 96 captures and all synthetic two-mode sets; 3 CTAs/SM
                for decode-only plans (`UNIFIED_NP_OCC variant=only16 blocks/SM=3`), 2 for mixed.
              - Captures, on vs off (same build): 1.000 / 0.995 / 1.000 / 1.002x. A6 vs v3 with it:
                1.025-1.050x.
              - Synthetic decode, on vs off: shared 4K prefix 1.00 / 1.08 / 1.15x (bs 64-256); ShareGPT 1.05 / 1.01
                / 0.92 / 1.03 / 1.06x (bs 64-1024); math 1.00 / 0.95 / 0.93 / 0.87x; long tail 0.93-0.99x; 48K
                shared prefix 0.91-0.97x. Geometric mean ~0.98x.
              - The persistent runner (also 3 CTAs/SM) still beats it on math by 1.15x. So occupancy alone does
                not reproduce the runner's long-KV decode advantage; its tile code (persistent.cuh body, no spill)
                matters.
              - Outcome: opt-in only (`UNIFIED_NP_ONLY16=1`, flashinfer `ffdd579`). The chunk-rounding run was built
                before this change, so both of its arms have it on.
              - [ ] Plan-chosen executor per step (persistent runner for decode-only plans, NP for mixed): the
                    runner wins two-mode decode batches >= 64 rows by 3-14% on synthetic steps but is level on the
                    captures and loses on four-mode (0.91-1.00x). It needs both executors in one build. Deferred
                    until a serving workload shows large decode-only batches.
        - [x] **Tile-rounded FA2 chunk, now the default** (flashinfer `5797024`, default since `73d2502`: the per-slot
              share rounded up to 256 keys, not a power of two; `UNIFIED_FA2_CHUNK_ROUND=pow2` = ablation;
              `unified/a6_h200_chunk_round_20261025.sh`, H200, 2026-10-26, same build both arms). FP32 + 20 graph
              replays pass (mix32_four, mix128_two, ShareGPT two / four, long-tail four).
              | set | tile vs power of two |
              |---|---|
              | 96 captures | 1.006 / 1.003 / 1.004 / 1.004x (32-64K four / two, 128-256K two / four) |
              | 48K shared prefix four, bs 16 / 32 / 64 / 128 | **1.221 / 1.279** / 1.083 / 1.015x |
              | 48K shared prefix two | 1.170 / 1.194 / 1.034 / 0.998x |
              | long tail four | 1.020 / 1.062 / **1.208** / 1.120x |
              | math four / two | 1.000-1.140x / 1.005-1.101x |
              | ShareGPT four, bs 64-1024 | 0.981-1.000x, bs 1024 **1.112x** |
              | ShareGPT two | 1.058 / 0.979 / **1.183** / 1.036 / 1.000x |
              | shared 4K prefix, long tail two | 0.994-1.034x |
              - Balance after the change (four-mode ShareGPT): FA2 span 1.16x its balanced bound, down from 1.24x at
                bs 1024. The two-mode bound uses 2 CTAs/SM while that build ran 3, so its ratio < 1 is not meaningful.
              - [ ] Rebuild the serving library (`/opt/tiger/w/a6lib/unified_dispatch_ext.so`) with this default
                    and redeploy to the H20 pods before the next serving runs. Record the build SHA in each arm's
                    protocol.
                    - [x] H200 build of the serving configuration (np64: 3 FA2 classes 64 / 32 / 16, NP + in-launch
                          merge, stock FA3; flashinfer `24c0545`, `fused_plan.cuh` md5 `ac0ce78c83a1`,
                          `unified_dispatch_ext.cu` `a7e28a43983e`; `unified/a6_h200_serving_lib_20261026.sh`).
                          - FP32 + 20 graph replays pass with FA3_MIN=65 on 96 captures + 4 synthetic sets.
                          - vs native v3: 1.042 / 1.035 / 1.028 / 1.030x (32-64K four / two, 128-256K four / two).
                          - tile vs power of two, same build: 1.000-1.005x.
                    - [x] H20 build from the same source on pod 4 (`unified/pod4_srv_build_20261026.sh`, library md5
                          `bc43aedeb9ed`). FP32 + 20 graph replays pass on all 43 GLM-4.5-Air captures (24 four, 19
                          two); vs native v3, GLM two-mode 1.006x.
                          - The GLM four-mode timing pass stopped in the v3 baseline's setup ("not native-split
                            eligible" for one capture), a harness limit of the v3 control, not the new library.
                          - Build fixes on the pod: add `native_q32_check_20261023` to the loader path, and copy the
                            flash-attention tree the H200 used (md5 `0d9da1edeb00`; pod 4 had `76b466e00f7b`).
                    - [x] Deployed to pods 0-3 (2026-10-26, `pd_colocation/mn_deploy_a6lib_20261026.sh`; md5
                          `bc43aedeb9ed` everywhere; previous `c76894e92792` kept as `.np64_20261025`).
                    - [x] Llama-3.3-70B gate on pod 0 (`unified/pod0_llama_gate_20261026.sh`): FP32 + 20 graph replays
                          pass on all 38 captures. Deployed lib vs native v3: four-mode **1.017x** [1.012-1.027], two-mode
                          **1.009x** [1.005-1.019]. Before tile rounding, Llama on H20 was level with v3.
        - [x] **Why MELD (SOSP '26) reports large JIT gains and we see none** (answered 2026-10-26). MELD's GQA comparison is decode-only
              at bs 128-1024 against FlashInfer 0.6, whose combine runs as a separate kernel (37% of the time in its
              FlashMLA example), and it gains most at small batch from fine-grained balance. A6 already merges inside
              the launch and balances by LPT, so MELD's "decouple" step is already in our baseline. Its "cooperate"
              step (DSMEM) can only win the merge cost, which is <= 0.9% on captures but up to 13% on large decode
              batches (above). Compare its workloads (split counts,
              batch sizes, KV spread, separate-merge-launch baseline) with ours. Measure ours per capture with
              `unified/plan_balance_20261025.py` (`a6_h200_plan_balance_20261025.sh`, queued behind the DSMEM run):
              batch size, FA2 KV min / median / max, chunk size, split groups, items, and FA2 span vs the two balance
              floors (sum of work / CTA slots, and the longest item).
        - [x] **Splits and load imbalance on four-mode steps** (answered 2026-10-26).
              - Every one of the 96 captures has FA2 split groups.
              - On captures, FA2 shares SMs with FA3, so the FA2 span (1.13-1.61x its balance bound) is not pure
                imbalance.
              - On synthetic large batches, imbalance is real and comes from the power-of-two chunk; fixed by tile
                rounding (above).
              - Original question: is it true that no FA2 split groups occur, and FA3
              split rows never occur under `geometry_fa3`? Is a four-mode step ever imbalanced? Answer from the same
              per-capture run, then build synthetic steps that are imbalanced (long-tailed KV) to show where the
              reduction matters.
        - [ ] **L2 order for FA2** (the old `HolisticPlanInfo<4>` design was "L2-swizzled LPT"; flashinfer `34cfdf8`
              recorded that its host-side section ordering never crossed the L2 budget, and the balance gain came from
              the per-section KV limit).
          - [x] **KV-position-major order** (`UNIFIED_FA2_ORDER=kvpos`, flashinfer `f2703e4`;
                `unified/a6_h200_kvpos_20261025.sh`; H200, same build, vs LPT; FP32 gate passes everywhere):
                | set | result |
                |---|---|
                | 96 captures | 1.000 / 1.003 / 0.998 / 1.000x |
                | 48K shared prefix (exceeds L2), four-mode, bs 16 / 32 / 64 | 1.016 / 1.014 / **1.081x** |
                | 48K shared prefix, two-mode (rows unsplit, order = LPT) | 1.00-1.03x |
                | 4K shared prefix (fits L2) | 0.995-1.012x |
                | long tail two-mode | 1.017-1.032x |
                | ShareGPT split batches, bs 64-256 | **0.71-0.88x** (breaks longest-first: long chunks land in the tail) |
                L2 grouping helps only where a shared prefix exceeds L2 and its rows are split, and it must not
                break longest-first order.
          - [x] **lptpos** (flashinfer `e422c9d`: longest first, equal-cost chunks by KV position;
                `unified/a6_h200_lptpos_20261026.sh`): level everywhere (captures 0.998-1.002x, synthetic
                0.981-1.019x). **L2 order closed:** grouping by KV position helps only by breaking longest-first
                order, which costs more elsewhere. The default stays LPT.
        - [x] **CTA count / launch overhead at large batch** (answered 2026-10-26): none measurable. The persistent pull
              loop on the same tile code is 0.89-0.98x of the hardware grid up to 4096 items; the runner's
              large-batch edge is occupancy plus its tile code (above). Original question: the NP grid launches one CTA per item (thousands at
              large batch). Synthetic sweep at bs 64-4096, decode-only and verify-only rows, short and long KV: NP
              grid vs the persistent loop on the same BatchPrefill tile code (`h200_npp`), isolating launch / prologue
              cost from tile code.
        - [x] Paper 5.1 / 7 rewritten from these results (2026-10-26). The pushed text claims there is no locality for a swizzle;
              that claim is withdrawn until the swizzle is measured.
      - [x] **Plan time on serving shapes, overlapped with the previous step.**
        - [x] 2026-10-25 Found and fixed a serving stall (`unified/plan_overlap_20261025.py`,
              `a6_plan_overlap_h200_20261025.sh`, H200, 24 captures).
              - The A6 serving plan made five blocking host-to-device copies per mixed step. A PyTorch `.to(device)`
                from CPU memory synchronizes the stream, so building the next step's plan waited for the previous
                step's kernels to drain.
              - With ~20 attention calls queued, the old plan took 1.5-161 ms of host time (median 4.8 ms), i.e. the
                queued GPU time. It is now built from the FA3 backend's device metadata (`cu_seqlens_q`,
                `cache_seqlens_int32`), checked against the host lengths on the first three steps, with
                double-buffered pinned staging guarded by per-buffer events, as v3 does.
              - It now takes 0.4-1.1 ms (median 0.71 ms) whatever is queued. Outputs are bit-identical on all 24, and
                FP32 + 20 graph replays pass on all 96.
              - Stage 2's A6 numbers were measured with the stall.
        - [x] Serving A/B, old vs new plan path (`unified/a6_plan_serving_ab_h200_20261025.sh`, H200, stage 2's short
              set, four-mode, two repeats each, 449 / 453 A6 calls logged; stock FA3 reference 137-138 agent tok/s in
              both roots):
              | A6 arm | agent tok/s | agent TPOT | agent token-weighted ITL |
              |---|---|---|---|
              | old plan (blocking copies) | 124.8 / 123.6 | 14.05 / 13.46 ms | 12.0 / 11.9 ms |
              | new plan (device metadata) | 124.6 / 125.0 | 13.97 / 14.01 ms | 11.1 / 12.4 ms |
              - No measurable serving effect: in this server the next step's host work does not run far ahead of the
                GPU on mixed steps, so the stall had little to wait for.
              - The fix stays, since it removes a dependence on that, and stage 2's A6 numbers stand.
      - [x] Serving executor in the FA3 backend (`SGLANG_A6_UNIFIED=1`) and stage 2 pairs (above).

      **Bugs found on the way (all fixed)**
      - Merges read unwritten partials: a split with no visible key has LSE -inf and an unwritten O slot. Merges skip
        -inf slots.
      - A build switch (`UNIFIED_FA2_NP`) never reached nvcc, so two runs measured the persistent kernel. Results
        retracted; every driver now fails unless each switch is on the nvcc line.
      - Stock FA3 takes page-1 KV as `[tokens, kv_heads, d]`; the executor passes a 3-D view.
      - Cross-launch race: the embedded BatchPrefill body triggered the next PDL launch
        (`griddepcontrol.launch_dependents`) before our epilogue had reset the ticket and group counters, so two launches
        of one plan overlapped. Fixed with `suppress_pdl_trigger`, atomic resets and a trap on out-of-range tickets.
        Every NP FA2-rows number taken before the fix is superseded by the table above.
      - Lost H200: trial `34c7bd503d461dc1` ended on 2026-10-24 with its captures and builds. Re-forked as
        `8574c727fab6ffbd` and restored with `unified/prov_h200_20261024.sh`.
- [ ] 2026-10-24 **Stages 2-3 on v3, after the kernel items** (user, 2026-10-24): native v3 is the kernel for these
      arms until A6 beats v3 on stage 1; then A6 replaces it. All three models are swept and shown (the paper may later keep only
      Llama-3.3-70B and Qwen2.5-7B); every model's results are recorded.
- [ ] 2026-10-24 **Llama-3.3-70B, stages 1-3** (provisioning on H20 pods 0-3: `pd_colocation/mn_provision_70b_20261024.sh`).
  - [x] Stage 1: 1.009 / 1.011x native v3 on two- / four-mode (H20, TP4, 32-64K), 1.38 / 1.61x sglang FA3.
  - [ ] Stage 2 (H20 pod 4, TP4, `unified/a6_stage2_llama_h20_20261025.sh`; both arms hold the EAGLE3 head at 3.0
        tokens per verify via `--common-env SGLANG_SIMULATE_ACC_LEN=3.0`, since the trained head accepts ~1.3 here):
    - [x] 2026-10-25 agent C=4 + chat 2/s (`cap_a6`), two repeats; far above capacity (TTFT 40-55 s in both arms):
      | Llama-3.3-70B | agent tok/s | chat tok/s | agent TPOT | agent TTFT | chat TTFT | chat TPOT |
      |---|---|---|---|---|---|---|
      | four-mode, stock | 16.7 / 16.7 | 324 / 323 | 127 / 107 ms | 42.3 / 43.6 s | 46.0 / 46.4 s | 97 ms |
      | four-mode, ours | 17.0 / 17.0 | 333 / 332 | 58 / 64 ms | 53.6 / 55.6 s | 7.1 / 6.2 s | 119-120 ms |
      | two-mode, stock | 16.2 / 16.2 | 320 / 319 | 97 / 112 ms | 45.0 / 43.5 s | 46.6 / 46.4 s | 98-102 ms |
      | two-mode, ours | 17.5 / 17.5 | 349 / 347 | 64 / 73 ms | 45.1 / 43.9 s | 8.8 / 10.1 s | 130-135 ms |
      - Four-mode: chat TTFT -85%, agent TPOT -45..-54%, throughput +2-3%. Costs: agent TTFT +23-28%, chat TPOT +20%.
      - SLO attainment is near zero in both arms at this load.
    - [x] **The same load without the deadline chunk cap** (`over_a6_nodc`, 2026-10-26, `--deadline-chunk=`, two
          repeats). Ours vs stock:
          | mode | agent tok/s | agent TPOT | agent worst-10-token TPOT | agent TTFT | chat tok/s | chat TTFT | chat TPOT | chat att 2 s / 100 ms | chat e2e mean |
          |---|---|---|---|---|---|---|---|---|---|
          | four | **+8.9%** | -46% | -64% | +12% | **+9.9%** | -84% | +31% | 27-29 vs 0% | 30 vs 67 s |
          | two | +7.3% | -38% | -35% | +6% | +5.5% | -73% | +38% | 20-23 vs 0% | 35-39 vs 66 s |
          - Throughput rises from +2-3% (capped) to +9-10% (four-mode).
          - Chat attainment at 2 s / 100 ms goes from 0% to 20-29%, and chat e2e latency halves.
          - Chat mean TPOT rises for the reason found at low load: the first-gap tail.
    - [x] agent C=2 + chat 0.5/s (`low_a6`, 2026-10-26, two repeats, 12 agent requests per arm, same driver): ours vs
          stock, change in the mean:
          | mode | agent tok/s | agent TTFT | agent TPOT | agent worst-10-token TPOT | chat TTFT | chat TPOT | chat ITL |
          |---|---|---|---|---|---|---|---|
          | four | -4.7% | +24% | -36% | -51% | -89% | +38% | +45% |
          | two | +3.3% | +1.6% | -11% | -37% | -90% | +41% | +27% |
          - Near capacity the deadline order still moves chat prefills ahead (chat TTFT -89%) and steadies agent
            decode, but chat decode gets ~40% slower.
          - [x] **Chat TPOT "regression" resolved (2026-10-26): it is a tail, and attainment improves.**
                - Two causes:
                  1. The deadline chunk cap (`SGLANG_EDF_DEADLINE_CHUNK=1024:2048:6000`, Qwen-calibrated pace) gave
                     47% more prefill batches (570 vs 387).
                  2. Deadline order serves chat prefill first, so the wait for one agent-chunk step (~1 s on 70B
                     TP4) moves from chat TTFT into chat's first inter-token gap. 48 chat requests have a single
                     > 1 s gap vs 22 in stock, and those carry 70% of the summed TPOT.
                - Median chat TPOT is better in ours (16.0 vs 17.6 ms two-mode; 6.1 vs 6.2 four-mode).
                - Rerun with the cap off (`low_a6_nodc`, `--deadline-chunk=`, two repeats): prefill batches 404 vs
                  387. Ours vs stock:
                  | mode | agent tok/s | agent TPOT | chat TTFT | chat mean TPOT | chat att 2 s / 50 ms | chat att 2 s / 100 ms | chat e2e mean |
                  |---|---|---|---|---|---|---|---|
                  | four | +3.8% | -39% | -86% | +13% | 67.2-67.8 vs 66.1% | 73.9 vs 66.7% | 10.1 vs 12.4 s |
                  | two | +3.6% | -17% | -83% | +57% | 67.2-67.8 vs 63.9-64.4% | 72.2-73.3 vs 67.2% | 12.9 vs 14.3 s |
                  (capped run: four 65.6 vs 66.1% / 73.3 vs 66.7%, e2e 10.7 vs 12.3 s; two 67.8 vs 64.4-65.6% /
                  75.0 vs 67.2%)
                - Decision: later single-node runs drop the cap (`--deadline-chunk=`). The paper reports
                  per-tenant attainment and tail ITL, not mean TPOT.
                - Tools: `/tmp/chat_tail.py`, `/tmp/chat_att.py` (copied to `drivers/serving/chat_tail_attainment_20261026.py`).
          - Agent sample is small (12 requests per arm); per-request distributions and error bars are pending.
- [ ] **Stage 3 with the full stack** (kernel + 4-mode MTP) vs stock PD with the P:D ratio swept, at higher load and
      with mixture spikes (`pd_colocation_vs_disagg.md`).
      - [x] 2026-10-25 A6 on the multi-node workers: `KERNEL=a6` in `mn_node.sh` (FA3 backend, the libraries under
            /opt/tiger since /dev/shm is noexec on the H20 pods), deployed to pods 0-3 by `mn_deploy_a6_20261025.sh`
            (applied only where the base files match the checked md5). Smoke test on one worker passed: a 62K-token
            prompt plus 12 concurrent short requests under 4-mode MTP, about 900 A6 calls (`a6_unified=1`), no
            tracebacks.
      - [x] Run (`pd_colocation/mn_stage3_a6_16g_20261025.sh`; Qwen2.5-7B, 16 H20 = 4 pods x 4, router step model
            calibrated on an A6 + MTP worker; flashinfer `396b04b` np64 build; one repeat per point). Chat attainment
            (0.5 s + 20 ms) / agent SLO goodput (10 s + 80 ms, tok/s):
            | agent programs + chat/s | ours (full stack) | conditional disagg (2048 / 0.7) | stock PD 2 + 2 | stock PD 3 + 1 |
            |---|---|---|---|---|
            | 32 + 8 | 97.4% / 601 | 97.8% / 527 | 40.1% / 107 | 53.2% / 162 |
            | 64 + 16 | 93.6% / 1100 | 96.0% / 821 | 13.4% / 52 | 13.1% / 80 |
            | 96 + 24 | 90.0% / 1403 | 90.5% / 1136 | 10.0% / 15 | 0.0% / 4 |
            | 128 + 32 | 79.1% / 1564 | 71.4% / 1143 | 0.3% / 6 | 0.1% / 6 |
            | 64 + 16 + 32-program spike | 93.2% / 1053 | 94.4% / 873 | 8.8% / 32 | 8.0% / 46 |
            - Stock PD with 1 prefill pod: 1.6% / 2 at 64 + 16.
            - Against conditional disaggregation: agent SLO goodput +14 / +34 / +24 / +37% and +21% under the spike;
              chat attainment within 2.4 points through 96 programs and +7.7 at 128.
            - Against stock PD: every point, by 3.7-260x in agent SLO goodput.
            - [x] **Repeat r2 (2026-10-26, `R=r2 BEST_SPLITS="3 2"`, log `/tmp/h20/mn_stage3_a6_16g_r2.log`).** Ours vs
                  conditional: 97.3% / 611 vs 97.7% / 547 (32); 93.8% / 1093 vs 95.5% / 932 (64); 90.3% / 1463 vs
                  89.3% / 1181 (96); 77.1% / 1532 vs 69.4% / 1212 (128); spike 93.7% / 1064 vs 93.9% / 855.
                  - Agent SLO goodput vs conditional, r1 / r2: +14 / +12, +34 / +17, +24 / +24, +37 / +26% at
                    32 / 64 / 96 / 128 programs; +21 / +24% under the spike. Stock PD collapses again (0-58% chat).
                  - Paper SLO matrix (`PAPER` line, chat 2 s + 50 ms, agent 10 s + 80 ms): ours 95.5-98.9% chat and
                    96.8-98.4% agent attainment at every load; conditional 90.4-99.6% / 91.3-98.4%.
            - [x] Error bars from per-request distributions (2026-10-26, `perf_model/fanout_tok_s_user_20261026.py`, arrays
                  in pod 0 `/dev/shm/w/mn/stage3_tok_s_user_20261026.json`). Per-request decode speed, tok/s/user,
                  p10 / p50 / p90, r1 ; r2:
                  | load | tenant | ours | conditional disagg |
                  |---|---|---|---|
                  | 64 + 16 | agent | 74 / 139 / 219 ; 72 / 136 / 216 | 25 / 74 / 199 ; 41 / 104 / 198 |
                  | 64 + 16 | chat | 111 / 267 / 370 ; 104 / 273 / 373 | 135 / 286 / 373 ; 112 / 267 / 364 |
                  | 128 + 32 | agent | 23 / 48 / 112 ; 22 / 45 / 114 | 15 / 36 / 111 ; 17 / 36 / 110 |
                  | 128 + 32 | chat | 40 / 138 / 315 ; 38 / 136 / 317 | 22 / 118 / 300 ; 21 / 105 / 292 |
                  | spike | agent | 54 / 120 / 208 ; 52 / 121 / 212 | 30 / 78 / 193 ; 29 / 76 / 195 |
                  - Stock PD's p99 (up to 250K tok/s/user) is a measurement artifact: tokens arrive in bulk after the
                    KV transfer, so the per-request decode interval collapses. Its medians (55-183) are the usable
                    statistic.
                  - [ ] Figure: CDF of agent / chat tok/s/user per system at 64 and 128 programs (paper Figure 20).
- [x] **Stage 3, Llama-3.3-70B, full stack, 16 H20** (2026-10-26, two repeats; `pd_colocation/mn_stage3_llama_16g_20261026.sh`,
      r2 as `R=r2 BEST_SPLITS="2 3"`; logs `/tmp/h20/mn_stage3_llama_16g.log`, `/tmp/h20/mn_stage3_llama_16g_r2.log`).
  - Setup:
    - 4 TP4 workers, one per pod; trained EAGLE3 head with real acceptance on the natural-text fan-out; deployed lib
      `bc43aedeb9ed`;
    - load: programs C = 4 / 8 / 12 / 16 with chat C/4 req/s, on 32K prefixes over 8 tasks;
    - P:D split searched at C = 8 (2 + 2: 98 tok/s, 3 + 1: 94, 1 + 3: 87), then the spike (+4 programs).
  - Score: paper SLOs (chat 2 s + 50 ms; agent 10 s, 20 s > 200K, + 80 ms), `fanout_per_tenant.py` PAPER line.
  - Results, chat attainment ; agent SLO goodput tok/s, r1 / r2:
    | C + chat/s | ours | conditional disagg | stock PD 2 + 2 | stock PD 3 + 1 | stock PD 1 + 3 (r1) |
    |---|---|---|---|---|---|
    | 4 + 1 | 98.1 / 98.0% ; 57 / 58 | 98.4 / 98.6% ; 54 / 56 | 91.4 / 90.8% ; 55 / 56 | 92.7 / 92.2% ; 53 / 55 | 84.0% ; 54 |
    | 8 + 2 | 95.4 / 96.8% ; 105 / 107 | 98.2 / 98.2% ; 83 / 79 | 85.0 / 85.1% ; 98 / 98 | 92.3 / 91.0% ; 94 / 102 | 67.7% ; 87 |
    | 12 + 3 | 93.4 / 94.3% ; 139 / 139 | 97.7 / 97.7% ; 99 / 101 | 79.6 / 78.6% ; 136 / 134 | 83.1 / 81.7% ; 122 / 120 | 27.1% ; 110 |
    | 16 + 4 | 90.8 / 90.5% ; 165 / 159 | 97.9 / 97.6% ; 85 / 97 | 68.2 / 66.6% ; 160 / 163 | 74.9 / 67.2% ; 123 / 106 | 1.3% ; 93 |
    | spike | 95.8 / 96.3% ; 101 / 100 | 98.5 / 97.9% ; 70 / 79 | 86.9 / 84.4% ; 101 / 101 | 89.9 / 89.6% ; 99 / 96 | 63.9% ; 92 |
  - vs stock PD's best split per load: chat attainment +3..+23 points; agent goodput -2..+7%.
  - vs conditional disaggregation: agent goodput +4..+94%; chat attainment -0.3..-7.1 points.
  - Why Llama differs from Qwen: on 4 TP4 replicas the router has 4 choices instead of 16, and a 70B prefill chunk
    on a colocated node holds every co-resident chat decode for one long step.
  - Bugs found and fixed:
    - `sglang_router` rejects `power_of_two` with one decode worker, so it now uses `round_robin` there;
    - one 2 + 2 arm lost its pod-0 ports and was rerun.
- [ ] 2026-10-24 **Conditional-disagg comparison re-run at Dynamo's defaults** (2048 tokens, 0.7 of the prompt; the
      earlier run used 4096), queued behind the stage-3 baselines (`pd_colocation_vs_disagg.md`).
- [x] **Fault tolerance with the full stack, 32 programs** (2026-10-26, one repeat; `pd_colocation/mn_fault_fullstack_16g_20261026.sh`,
      log `/tmp/h20/mn_fault_fullstack_16g_r1.log`; scored by `perf_model/fault_windows.py --series` after copying
      `fault_t.txt` from the killed pod, and by `fanout_per_tenant.py` (paper SLOs)).
  - Results: agent attainment before -> first 60 s after the kill -> rest of run; overall chat attainment ; agent SLO
    goodput (the no-fault reference at 32 + 8 is from Table 10, r1 / r2):
    | arm (node killed) | agent windows | chat (2 s + 50 ms) | agent goodput | no-fault goodput |
    |---|---|---|---|---|
    | ours (node 1) | 95.0 -> 95.5 -> 96.8%, 0 lost | 98.7% | 552 tok/s | 601 / 611 |
    | conditional disagg (node 1, decode) | 94.1 -> 93.2 -> 98.7%, 0 lost | 99.4% | 449 tok/s | 527 / 547 |
    | stock PD 2 + 2, prefill node 1 | 50.8 -> 21.8 -> 29.6%, 1 lost; chat 45.7 -> 19.7 -> 23.1%, 5 lost | 30.0% | 64 tok/s | 101 / 87 |
    | stock PD 2 + 2, decode node 2 | 52.7 -> 43.8 -> 50.9%, 1 lost; chat 46.3 -> 33.9 -> 49.3% | 47.6% | 107 tok/s | 101 / 87 |
  - Losing a quarter of the cluster costs ours 9% of agent goodput, with no attainment dip for either tenant.
    Conditional disaggregation loses 16%. Stock PD loses about a third when a prefill node dies (agent attainment
    halves and does not recover), and is level when a decode node dies: its bottleneck is the prefill / transfer
    side, not decode.
- [x] **Fault at 64 programs** (2026-10-27, `FAULT_LOAD=64:16 FAULT_ARMS="ours cond pd_p1" bash
      pd_colocation/mn_fault_load_16g_20261026.sh`, log `/tmp/h20/mn_fault_L64_16g_r1.log`; same scoring).
  - Agent windows (before -> first 60 s -> after) ; chat ; agent goodput (no-fault reference from Table 10):
    - ours: 95.9 -> 93.4 -> 97.3%, 0 lost ; 95.8% ; 974 tok/s (1093-1100);
    - conditional disaggregation: 93.4 -> 94.1 -> 95.4% ; 98.4% ; 756 (821-932);
    - stock PD 2 + 2, prefill node 1: 25.6 -> 3.1 -> 0.0%, never recovers ; 6.0% ; 19 (52-55).
- [ ] **GLM-4.5-Air, stages 1-3** (stage 1 at parity with v3 after the 64-row class, A6 section 6).
  - [ ] Stage 2 (H20 pod 4, TP4, NEXTN MTP, mix32: agent C=8 + chat 8 req/s;
        `unified/a6_stage2_glm_h20_20261025.sh`).
    - [x] 2026-10-25 First attempt. Repeat 1, four-mode, ours vs stock:
      - agent tok/s 26.9 vs 22.6 (+19%), agent TPOT 58 vs 107 ms (-46%), agent TTFT 52 vs 61 s;
      - chat TTFT 19.6 vs 97.6 s. The load is far above GLM-Air's capacity here.
      - Repeat 2's ours arm hung, with every TP rank stuck to the 600 s watchdog.
    - [x] Cause: **TP ranks diverged.** The EDF code read each rank's own `time.monotonic()` (arrival stamps, late
          demotion, deadline-sized chunks, ITL slack), so a deadline near "now" could be judged differently per rank.
          At the hang, rank 0 ran a batch of 2 while ranks 1-2 ran 4, and the collectives deadlocked. Stage 2 on the
          H200 was TP1, so it never showed.
    - [x] Fix (`pd_colocation/apply_edf_tp_clock_20261025.py`, applied to the serving tree): rank 0 reads the clock
          once per scheduling iteration and ships it inside the request broadcast it already makes. Every EDF decision
          on every rank uses that value (`edf_now()`). TP = 1 is unchanged.
    - [x] **Rerun with the fix** (H20 pod 4, sglang serving tree + `apply_edf_tp_clock_20261025.py`, A6 build np64,
          flashinfer `396b04b`). All 8 arms completed with no hang; 12 120 A6 call lines logged, acceptance 3.0-3.1
          tokens per verify. Ours (EDF + 4-mode MTP on A6) vs stock sglang on FA3, per repeat:
          | GLM-4.5-Air, mix32 | agent tok/s | chat tok/s | agent TPOT | chat TTFT | chat attain. 2 s + 50 ms | agent attain. 10 s + 80 ms |
          |---|---|---|---|---|---|---|
          | four-mode, ours | 26.4 / 27.4 | 677 / 706 | 56.9 / 55.5 ms | 32.1 / 16.0 s | 65.4 / 75.5% | 36.7 / 43.3% |
          | four-mode, stock | 22.3 / 23.1 | 575 / 595 | 110.4 / 94.6 ms | 99.7 / 93.4 s | 0.0 / 0.0% | 20.0 / 20.0% |
          | two-mode, ours | 25.1 / 25.7 | | 84.9 / 76.6 ms | 34.9 / 34.0 s | 59.7 / 63.5% | 23.3 / 30.0% |
          | two-mode, stock | 22.4 / 22.6 | | 102.8 / 106.4 ms | 95.1 / 94.3 s | 0.0 / 0.0% | 23.3 / 23.3% |
          - Four-mode: agent and chat throughput +18-19%, agent TPOT -41..-48%, chat TTFT -68..-83%. Both bars
            (>= 30% latency, >= 10% throughput) are met for GLM on one node.
          - The load is far above GLM-Air's capacity on 4 H20s (TTFT tens of seconds in both arms), so this is the
            overloaded end of the curve.
    - [x] **Final configuration, same load** (2026-10-26, `short_a6_nodc`, `--deadline-chunk=`, deployed lib
          `bc43aedeb9ed`, two repeats; attainment r1 / r2). Ours vs stock:
          | mode | agent tok/s | chat tok/s | agent TPOT | agent TTFT | chat TTFT | chat att 2 s / 100 ms | agent att 10 s / 80 ms | chat e2e |
          |---|---|---|---|---|---|---|---|---|
          | four | **+28%** | **+27%** | -32% | -23% | -71% | 0 -> 76.7 / 77.5% | 20 -> 30.0 / 33.3% | 111 / 110 -> 40 / 36 s |
          | two | +17% | +15% | -37% | -13% | -55% | 0 -> 65.7 / 63.2% | 23 -> 33.3 / 43.3% | 105 / 108 -> 55 / 56 s |
          With the cap, four-mode throughput was +18-19%.
    - [x] **A load at capacity** (2026-10-26, `cap_a6_nodc`: agent C=4 + chat 2/s, final configuration, two
          repeats, attainment r1 / r2, 14 agent requests per arm). Ours vs stock:
          | mode | agent tok/s | agent TPOT | chat TTFT | chat att 2 s / 100 ms | chat e2e | agent att 10 s / 80 ms | agent SLO goodput |
          |---|---|---|---|---|---|---|---|
          | four | **+18%** | -18% | -92% | 51.5 / 49.4 -> 82.5 / 84.2% | 13.4 / 13.9 -> 7.9 / 8.2 s | 28.6 / 42.9 -> 42.9 / 50.0% | 8 / 17 -> 18 / 24 tok/s |
          | two | +9% | -20% | -88% | 57.9 / 57.9 -> 82.1 / 81.0% | 12.7 / 12.7 -> 9.4 / 9.7 s | 42.9 / 35.7 -> 57.1 / 57.1% | 14 / 14 -> 24 / 28 tok/s |
- [x] **S4 step model re-fitted on the final kernel (A6)** (2026-10-25): held-out MAPE 2.0% vs stock's 1.9%, mixed steps
  5.9% vs 7.7%; details under A6 stage 2.

## Baselines

Per stage, all on sglang's FA3 backend (its Hopper default):
- **Kernel:** sglang's single varlen FA3 call; the best single fixed FA3 configuration; a per-step oracle over all FA3
  variants.
- **Single node:** sglang mixed chunk on FA3, with stock EAGLE3 MTP in four-mode; same frozen tree, audited
  opposite-order pairs, two repeats.
- **Multi-node:** sglang PD disaggregation (mooncake) with the P:D ratio swept at worker granularity and the KV pool
  on. Reference: the conditional-disagg policy (`router.py cond_disagg`; Dynamo lists SGLang as unsupported).
- pdmux is a baseline only.
- Results, commands and commits: `debug_kernel_server_perf/resume_20261012/native_kernel_serving_attribution_20261023.md`
  (single node) and `pd_colocation_vs_disagg.md` (multi-node).

## Tasks, in priority order

A and B are the two kernel items; S is the server/workload evaluation the user specified.
Supporting gates are prerequisites, not substitutes. `[x]` = measured, `[~]` = partly, `[ ]` = not run.

### [ ] A. Ultra fine-grained tile dispatcher (unchecked 2026-10-24: never built -- the unified plan + merge schedule is A6)

- [x] **A-fix, FA2 out of the co-compiled kernel (2026-10-12):** FA2 spilled inside the fused
  kernel (168 regs, 4200 B stack, 21x slower per item than `BatchAttention`); now FA3 grid on
  `sm_split` CTAs + `FA2ExecutorKernel` at native geometry as its PDL dependent
  (`FLASHINFER_FUSED_COLAUNCH=1`). 3.3-6.5x on FA2 rows, every gate exact; still 1.2-2x behind the
  non-persistent unfused two-launch control. ([evidence](debug_kernel_server_perf/resume_20261012/colaunch_20261012.md))
- [x] **A1, FA3 q64 vs q128 per work item (2026-10-12; reopened and re-closed 2026-10-17, see A5.1):** *Correction in place:* the
  old line "q64 never beats q128 beyond noise, loses up to 2.3x" misread the winner map. On the fixed
  kernel (`a3_winner_map/a3_v2_table_20261012.txt`, 72 cells, 216/216 arms exact) q64 is slower on 64
  cells (median 1.34x, worst 2.98x at kv 64K x 16 rows x qo 16: half the query rows per loaded KV block,
  so twice the KV traffic on memory-bound items) and faster by >3% on 3 cells, up to 24% (kv 1024, 1 row,
  qo 16 -- short-KV, few-row items). A per-item q-tile choice therefore has a real, narrow win region;
  reopened as part of the FA3 technique study (S-kernel, below), to be measured on serving-shaped items.
- [x] **A2, FA2 tile classes (2026-10-12; the open 48-tile and 64-removal work is in A6):** the adoption of 64/32/16 broke the user's rule (FA2 for 16/32/48 only, >= 64 to FA3) instead of fixing the 48 layout. Original finding: 48/32/16 refuted (tile 48 = 1 q-warp x 4 kv-warps,
  3x accumulator, spills; 1.6-2.2x slower at q8); **64/32/16** adopted (96 B stack). Per-item tail
  dispatch kept (bit-identical outputs). FA2 split-KV chunk = `auto` (sized to the FA2 CTA slots;
  matches or beats the best fixed chunk on every fixture). ([evidence](debug_kernel_server_perf/resume_20261012/colaunch_20261012.md))
- [x] **A3, fitted FA2/FA3 dispatch rule (2026-10-12; closed 2026-10-16, see the guard-0 A/B below):** winner map on the fixed kernel, 216/216
  exact: FA2 wins every verify-width row; rule "FA3 iff packed_qo >= 112 and FA3 work >= 3.5M"
  agrees on 69/72 cells (worst loss 6%) vs 55/72 (98%) for the serving rule -- **but loses in
  serving** (-3.5% throughput, +29% TTFT). Cause, from the split counters (2026-10-13): the work
  floor was fitted on band rows only (qo <= 32) but also rejected 512-token chat prefill chunks
  (3584 packed x 512 kv = 1.8M), so 1039/1601 split plans went one-sided onto the stock path vs
  122/1801 under the default rule. Routing one-sided batches through the co-launch instead would
  not help: on one-sided band shapes the fused kernel is 1.0-2.2x slower than `BatchAttention`.
  ([table](debug_kernel_server_perf/resume_20261012/a3_winner_map/a3_v2_table_20261012.txt))
  - [x] **A3b on the serving split (Codex kernel worker, 2026-10-13, all 12 arms):** DEFAULT vs A3B
    on `MTPMIX_SPLIT`, interleaved twice at mix32/mix64/mix128 x chat 24/s + agent 0.1/s; both rules
    pass the q4 output gate (long prefix 52), 0 fallbacks / tracebacks. A3B vs DEFAULT: mix32 chat
    TTFT -26.7%, agent tput +5.4% / TTFT -25.5%; mix64 chat TTFT -14.1%, agent tput +11.2%;
    mix128 chat tput +6.1% / TTFT -28.2% / P99 ITL -33.3%, agent tput +10.9%. **Not promoted as a
    universal switch:** A3B replaces the whole rule (incl. the decode-population guard), mean chat
    interactivity falls (mix128 -16.9%), agent fixed-SLO attainment drops at mix128 (0.19 vs
    0.44/0.31), acceptance differs between arms, and only 16-24 agent requests per arm.
    ([findings](debug_kernel_server_perf/resume_20261012/a3b_split_findings_20261013.md), `a3b_split_serving_20261013.sh`,
    `a3b_split_midrange_20261013.sh`.) My older `a3b_serving_ab_20261013.sh` is superseded.
  - [x] **A3 at guard 0, clean A/B (2026-10-16 00:19-01:13, all arms clean, peak = C):** the A3b test above compared A3B against
    DEFAULT with the decode guard ON, and A3B drops the guard, so its gains are not attributable to
    the fitted thresholds; guard 0 is now the default. Both rules at guard 0 on the split (chunk
    3072, no EDF): q4 gate per rule, interleaved DEFAULT/A3B x2 on the cohort at C=16 + chat 8 req/s,
    one pair on TraceLab C=8 (`a3_guard0_20261015.sh`). Both rules pass the q4 gate (long prefix 52
    vs the MTP_LPT_Q4G floor). **A3B = DEFAULT within noise:** cohort C=16, two-repeat means, A3B vs
    DEFAULT: agent tput +0.3%, TTFT -0.5%, TPOT -0.3%, ITL -3.5% (repeat spread 1.4%), P99 ITL -12.6%
    (spread 6.3%); chat tput -0.1%, TTFT -2.2%, TPOT -1.8%, ITL -0.8%; acceptance 2.13 in all four.
    TraceLab C=8: tput 0.0%, mean TTFT -3.8% (single runs vary up to 17%), median -0.5%, TPOT +1.3%,
    ITL +2.1%, P99 ITL +9.6%. The 2026-10-13 A3b gains (TTFT -25..-28%) came from dropping the decode
    guard, now the default, not from the fitted thresholds. Split calls 48.0-51.5K (cohort),
    35.5K (TraceLab) in both rules. **A3 closed:** keep the default rule at guard 0.
    Command: `bash a3_guard0_20261015.sh` (pod `/dev/shm/w`), table `a3tab.py a3_guard0_20261015`.

- [x] **A4, FA2 verify-row bandwidth (2026-10-13; closed by measurement 2026-10-15):** serving runs
  verify rows on stock FA2 BatchPrefill (63-83% of HBM) inside the two-launch split, and every fused /
  PDL / entry / locality / unpacked candidate measured no serving gain over it (sub-items below). The
  persistent co-launched FA2 executor stays ~20% behind stock (a mainloop gap) -- relevant only if the
  fused co-launch is revived. attention is 75-80% of serving kernel time,
  dominated by memory-bound verify rows. At serving verify shapes (q=4, GQA 28/4, all arms exact,
  `fa2_verify_bw_20261013.sh`), KV bytes / time per layer:
  | fixture | fused co-launch | unfused: FA3 + stock BatchPrefill FA2 | BatchAttention |
  |---|---|---|---|
  | 46 rows x 16K | 0.850 ms, 1.82 TB/s (38%) | **0.509 ms, 3.03 TB/s (63%)** | 0.831 ms |
  | 46 rows x 48K | 2.523 ms, 1.84 TB/s (38%) | **1.511 ms, 3.06 TB/s (64%)** | 2.446 ms |
  | 16 rows x 128K | 1.732 ms, 2.48 TB/s (52%) | **1.083 ms, 3.97 TB/s (83%)** | 1.985 ms |
  | 512-chunk + 46 x 48K | 2.567 ms | **1.532 ms** | 2.294 ms |
  The co-launched FA2 executor inherits the persistent BatchAttention design and reaches 38-52%
  of HBM; stock non-persistent BatchPrefill reaches 63-83%. On verify rows the fused path is
  ~1.65x slower than it need be -- the A-fix colaunch doc's "1.2-2x behind the two-launch
  control" is this. Serving's `MTPMIX_SPLIT` arm is exactly that two-launch split (FA3 via
  `flash_attn_with_kvcache` + `BatchPrefillWithPagedKVCacheWrapper` fa2). Next: gated serving A/B
  split vs fused at mix32 / mix128 x 16 / 24/s (`split_vs_fused_20261013.sh`, queued; the gate now
  requires split calls > 0, `split_gate_counter.patch`); then a PDL split (stock FA2 launched as
  the FA3 grid's dependent on the SMs FA3 leaves) and an FA2 executor whose KV pipeline matches
  BatchPrefill's.
  - [x] **Split A/B v1 invalid (2026-10-13 06:5x, recorded):** `MTPMIX_SPLIT` planned 2201
    two-sided splits but ran **zero** -- forward_extend dispatches the split on the paged path
    only, and ragged was forced off only under FUSED_KERNEL_EXEC, so the two-launch split fell
    through to stock BatchPrefill every layer. Its gate "PASS" was false: the driver raised "zero
    split calls" but the gate subshell returned the compare's status. Fixed: ragged off whenever a
    two-sided split plan exists (`split_dispatch_paged.patch`, in `up_racefix`); gate subshells in
    the split and A3b scripts now fail on the run's own status. A/B aborted, re-queued as v2.
  - [x] **FA2 executor tuning (2026-10-13, all exact):** CTAs/SM 2x vs 3x: no change (2.52 vs
    2.52 ms on 46 x 48K) -- not MLP-bound. KV chunk is the lever: auto 2.52 ms, 1024 2.01 ms,
    4096 1.84 ms (unfused control 1.49-1.52 ms); same on chunk+verify (2.55 / 2.03 / 1.84 ms). The
    auto rule splits long verify rows too finely. Larger chunks: 8192 1.92 ms, 16384 2.16 ms,
    65536 (no split; 184 items for 264 CTAs) 2.53 ms -- **optimum 4096: 1.84 ms, still ~20% behind
    the two-launch control (1.49-1.51 ms)**, a gap in the persistent mainloop itself, not its
    configuration (`fa2_exec_tune_20261013.sh`, `fa2_exec_tune2_20261013.sh`). Next: the auto
    rule needs a floor near 4096 for long verify rows (re-check on the A2 fixtures before
    changing it); the serving decision waits on split v2 (gate PASS, long prompts agree on 52
    tokens, split calls now required > 0; running).
  - [x] **Split v2, first point (2026-10-13 07:14; superseded by "Split v2 complete" below):**
    mix32 chat 16/s + agent 0.1/s, race-fixed tree, no barrier, both arms 0 tracebacks / 0
    fallbacks; split ran 80,001 split calls, fused 66,901 fused-kernel calls.
    | metric | split (FA3 + stock FA2 BatchPrefill) | fused co-launch | stock MTP (S1) |
    |---|---|---|---|
    | chat TTFT mean | **689 ms** | 4932 ms | 24740 ms |
    | chat TPOT mean | **14.5 ms** | 28.1 ms | 39.0 ms |
    | chat ITL p99 | 108 ms | 149 ms | 138 ms |
    | agent TTFT mean | **1941 ms** | 5102 ms | 14500 ms |
    | agent TPOT mean | **11.3 ms** | 16.0 ms | 21.9 ms |
    | agent output tok/s | **308** | 205 | 162 |
    **mix32 24/s (07:30, both arms clean; split 103,501 split calls, fused 80,801 fused calls):**
    | metric | split | fused | stock MTP (S1) | split vs stock |
    |---|---|---|---|---|
    | chat output tok/s | **1641** | 1489 | 1226 | +34% |
    | chat TTFT mean | **8684 ms** | 33662 ms | 63512 ms | -86% |
    | chat TPOT mean | **20.7 ms** | 27.7 ms | 34.8 ms | -40% |
    | agent output tok/s | **212** | 162 | 130 | +64% |
    | agent TTFT mean | **8768 ms** | 22597 ms | 43663 ms | -80% |
    | agent TPOT mean | **13.4 ms** | 16.7 ms | 23.8 ms | -44% |
    At both mix32 points the split clears the bar vs stock MTP on TTFT, TPOT and throughput for
    both tenants.
  - [x] **Split v2 complete -> serving kernel = the two-launch split (2026-10-13 08:00):** all 8 arms
    clean (0 tracebacks, 0 fallbacks; split 23.5K-103.5K split calls, fused 23.0K-80.8K fused
    calls). Split vs stock MTP (S1 numbers for stock):
    | point | chat tput | chat TTFT | chat TPOT | agent tput | agent TTFT | agent TPOT |
    |---|---|---|---|---|---|---|
    | mix32 16/s | = (offered load) | -97% | -63% | +90% | -87% | -48% |
    | mix32 24/s | +34% | -86% | -40% | +64% | -80% | -44% |
    | mix128 16/s | +31% | -54% | -33% | +29% | -26% | -26% |
    | mix128 24/s | +56% | -55% | -45% | +46% | -37% | -32% |
    The bar is met everywhere except the agent tenant at mix128 16/s (-26%). Split vs fused
    co-launch: mix32 chat TTFT -74..-86%, TPOT -25..-48%, agent tput +31..+50%; mix128 chat TTFT
    -14..-18%, tput +5..+6%, agent TPOT +1..+9% (fused better there). The gap shrinks with context
    because long prefill rows (FA3 in both arms) dominate; the split's edge is the verify rows
    (stock FA2 at 63-83% of HBM vs the executor's 38-52%). Data:
    [`split_vs_fused_v2_20261013.csv`](debug_kernel_server_perf/resume_20261012/split_vs_fused_v2_20261013.csv).
    **Overlap A/B queued (2026-10-13 08:1x):** `SGLANG_SPLIT_OVERLAP=1` runs FA3 on a side stream
    holding all but `SGLANG_FUSED_SM_MARGIN` SMs while stock FA2 runs on the main stream
    (`split_overlap.patch`; logged as `overlapped=` in FUSED_SPLIT_CALLS). Gate on the m16 arm, then
    split / overlap m16 / m32 at mix32 and mix128 x 24/s (`split_overlap_20261013.sh`). The queue is
    paused after the full-length replay's stock arm (`stop_v3.sh`) so the replay's and S3's 4-mode
    arms use the winner.
  - [x] **Overlap A/B: no gain, overlap stays off (2026-10-13 08:36-09:22):** gate PASS on the m16
    arm (long prompts agree on 52 tokens; 3001/3001 split calls overlapped). All 6 serving arms
    clean, overlap arms 100% overlapped (e.g. 35,501/35,501 at mix128).
    | 24/s | chat TTFT split / m16 / m32 | chat TPOT | agent tok/s |
    |---|---|---|---|
    | mix32 | 8944 / 8614 / 8675 ms | 20.9 / 20.4 / 20.8 ms | 211 / 215 / 212 |
    | mix128 | 68130 / 68108 / 68064 ms | 31.2 / 31.4 / 31.3 ms | 80.6 / 80.6 / 80.6 |
    mix32 differences are inside the +/-1.5-3% noise floor; mix128 within 0.3%. In these steps one
    role dominates (verify FA2 at mix32, prefill FA3 at mix128), so concurrent execution has
    little to hide. The serving kernel stays the plain two-launch split.
    ([csv](debug_kernel_server_perf/resume_20261012/split_overlap_20261013.csv),
    `split_overlap_20261013.sh`, `split_overlap.patch`)
    Originally planned next (A4): overlap the split's two launches (FA3 on a side stream with `sm_margin`, stock FA2
    on the SMs it leaves) and re-run the S1 x S2 sweep with the split as the 4-mode arm.
  - [x] **Chunk floor rejected (2026-10-13, all exact):** a fixed 4096 floor wrecks small-KV cases
    (1 row x 16K: auto 0.028 ms vs 4096 0.099 ms; 16 x 4K q8: 0.094 vs 0.163 ms) while helping
    long verify rows; and the two-launch control beats the fused executor on every A2 fixture too
    (e.g. 16 x 16K q8: 0.156 vs 0.323 ms). With the split as the serving kernel the auto rule is
    moot for serving (`fa2_chunk_floor_20261013.sh`).
    Chat throughput at 16/s equals the offered load in all arms (~1136 tok/s). The barrier-free fused arm
    (4932 ms TTFT) is within 7% of S1's fused-with-barrier (5276 ms). Command:
    `INHERIT_LOCK=1 bash split_vs_fused_20261013.sh` (root `split_vs_fused_v2_20261013`), analysed
    with `drivers/serving/eval_curves.py --slo '{"chat":[2000,50],"codex":[10000,80]}'`.

  - [x] **Kernel candidates vs the split (Codex kernel worker, 2026-10-13/14), none promoted:**
    - native nonpersistent FA2 as the PDL dependent of an SM-limited FA3 grid: v1 failed (planner
      reused the FA2 chunk for FA3 -> 512/512 nonfinite), fixed in v2 (0/544 bad); closes much of
      the FA2 gap (46 q4 x 48K: 1.495 vs stock two-launch 1.527 ms) but not the FA3-dominated one
      (prefill2048+8: 5.872 vs 3.950 ms) ([proto](debug_kernel_server_perf/resume_20261012/native_pdl_prototype_20261013.md)).
      Serving v3 with split-KV FA3: no mean gain (mix32 chat TTFT +9.8%)
      ([v3](debug_kernel_server_perf/resume_20261012/native_pdl_v3_serving_20261013.md)).
    - dedicated FA3 entry: -9..-23% kernel time vs the combined entry, still behind the unfused
      control on large-query fixtures ([entry](debug_kernel_server_perf/resume_20261012/native_fa3_entry_20261013.md)); compact planning
      validated ([compact](debug_kernel_server_perf/resume_20261012/native_pdl_compact_20261013.md)); combined serving: no general win
      (mix64 agent +9.7% is acceptance-confounded, 2.753 vs 3.177)
      ([combined](debug_kernel_server_perf/resume_20261012/native_pdl_combined_20261013.md)).
    - `auto4096` cap (min(auto, 4096)), opt-in: -20..-27% on long verify fixtures, +9..+18% serving
      tput vs the uncapped co-launch -- moot for the split, which uses stock FA2
      ([cap](debug_kernel_server_perf/resume_20261012/fa2_auto4096_candidate_20261013.md)).
    - FA3 diagnosis: NCU shows custom FA3 at ~43% tensor activity vs stock 65% (stock: KV96, 2
      stages, 132 CTAs) ([ncu](debug_kernel_server_perf/resume_20261012/fa3_ncu_20261014.md)); round-robin work pairs raise L2 hit
      11.9% -> 94.9% and cut kernel time -14% / -26% on two fixtures
      ([locality](debug_kernel_server_perf/resume_20261012/fa3_locality_20261014.md)); KV96 is +0.1..+4% slower than KV128 but exposed a
      real packed-Q bug (Q gather looped over the KV width, leaving 32 Q columns unwritten), fixed
      ([kv96](debug_kernel_server_perf/resume_20261012/fa3_kv96_prototype_20261014.md)); SM budget: keep 112
      ([budget](debug_kernel_server_perf/resume_20261012/tma_sm_budget_20261013.md)); one-warp TMA producer: no gain
      ([warp](debug_kernel_server_perf/resume_20261012/tma_warp_prototype_20261014.md)).
    - [x] paired serving split vs native PDL + dedicated entry + round-robin pairs (complete
      2026-10-14 04:4x, two repeats): no uniform gain -- 32-64K and 64-128K within +/-4% on every
      metric; 128-256K agent ITL -12.4% but agent TTFT +6.3%, chat tput -1.9%; split stays
      ([paired](debug_kernel_server_perf/resume_20261012/native_pdl_paired_serving_20261014.md)).
    - [x] unpacked query heads for FA3 (complete 2026-10-14 04:40, 16 gates + 40 timings pass):
      KV128 within 0.6% of packed auto on the four large fixtures, KV96 slower, unsplit 3.7x slower
      on irregular short queries; packed auto retained
      ([unpacked](debug_kernel_server_perf/resume_20261012/fa3_unpacked_20261014.md)).

- [x] **A5, FA3 technique study on serving-shaped items (user, 2026-10-17: "test these techniques more for fa3 ...
  otherwise it's hard to make a story around the kernel stuff if we just port it without any optimizations").**
  **Outcome (2026-10-17):** the custom FA3 role now runs flash-attention's own kernel (paged TMA, `-DNDEBUG`, fast
  math, dynamic scheduler) and matches sgl-kernel FA3 instruction for instruction; every earlier FA3 gap was the build
  (flashinfer's mainloop, asserts compiled in, cp.async gather). Of the techniques, only SM co-running FA3 with FA2
  buys anything (-18% on balanced append + verify steps), and its serving ceiling on TraceLab is 1.2% of GPU time
  because long-context prefill dominates every mixed step. The attention kernel is at the state of the art and is
  not where the serving bar can be won. (The co-launched FA2 executor's A4 gap was also measured without
  `-DNDEBUG`; moot while serving runs stock FA2.)
  Fixtures = FA3 items replayed from serving step traces (TraceLab append 256-1024 on 32-131K, chat prefill
  chunks, split-KV long rows), every arm PASS vs the causal fp32 reference, reported against upstream FA3
  (what serving runs) and the unfused two-launch step. In order:
  - [x] A5.0 mainloop gap (closed 2026-10-17): it was the build, not the mainloop design -- flashinfer's paged mainloop
    instead of flash-attention's, no `-DNDEBUG`, and cp.async page gather instead of paged TMA.
    - [x] **Parity (2026-10-17 13:53-13:58, all runs 11-12/11-12 vs fp32 causal):** sgl-kernel FA3 with
      `pack_gqa=True` forced is no slower (prefill 2048 @ 64K 3.24 vs 3.23 ms), so packed GQA is not the gap. Paged
      TMA KV (`FA3_PAGED_TMA=1`) with `-DNDEBUG` -- earlier measured slower only because the device asserts were in --
      makes our flash-attention build on its dynamic scheduler **match or beat sgl-kernel FA3**: prefill 2048 @ 64K
      3.19 vs 3.27, agent 4 x 512 @ 64K 3.32 vs 3.35, chat 2048 0.0606 vs 0.0601 ms. Final FA3 role build:
      flash-attention mainloop, paged TMA, `-DNDEBUG`, fast math. (Our static-list path at 112: 3.44 / 3.58 /
      0.117 ms -- the dynamic scheduler stays.) Mixed co-run table rerunning on this build
      (`a5_mixed_corun_tma_20261017.sh`).
    - [x] **A5.0f full table + scheduler (2026-10-17 13:26-13:50; every run 11/11 vs fp32 causal):** flash-attention
      mainloop + `-DNDEBUG` + fast math (`a5_fa3_upstream_table_20261017.sh`, two reversed repeats, spread <=0.6%):
      long items within 1.07-1.12x of sgl-kernel FA3 (append 512 @ 48K 0.627 vs 0.586, agent 4 x 512 @ 64K 3.74 vs
      3.34, prefill 2048 @ 64K 3.61 vs 3.31 ms); flashinfer's mainloop with `-DNDEBUG` gains only ~4%, so the
      flash-attention mainloop is the base. **Split-KV fix:** the two split fixtures were nonfinite --
      `fused_primary_finish` re-merged the plan's split rows from flashinfer's (unwritten) partial buffer after the
      dispatch had already run flash-attention's combine; patch item 2 in `fi_a5_patch.py`, now exact (append @ 132
      splits=2, 16 x 16 @ 1K). **Scheduler** (`a5_fa3_budget_20261017.sh`): our build on upstream's own dynamic
      persistent scheduler capped at 112 SMs (`fa3_only_run`, `FA3_SM_BUDGET=112`) vs our static per-CTA lists at
      112, ms: 8 x 1024 @ 32K 7.42 vs 9.05; chat 2048 0.074 vs 0.121; chat 4 x 512 0.040 vs 0.046; 16 x 16 @ 1K
      0.023 vs 0.041; long items tie (3.60 vs 3.63); static wins only append 512 @ 48K (0.648 vs 0.703, one tile per
      CTA). Dynamic at 112 vs sgl-kernel at 132: 1.10-1.24x. *Correction (2026-10-17 13:52, from A5.3 below): I read
      this as the SM-count ratio ("parity at equal SMs") -- wrong for the long items: prefill 2048 @ 64K is 448
      equal tiles, 4 waves at both 112 and 132, and sgl-kernel FA3 capped at 112 SMs is no slower; our 3.60 ms there
      is a real 1.10x per-SM gap.* So the FA3 role should run flash-attention's kernel on its dynamic scheduler over the
      FA3 SM budget; the remaining question is what the other SMs buy when FA2 runs on them concurrently (A5.3),
      which was last measured on the flashinfer mainloop without `-DNDEBUG`.
    - [x] **A5.0d FA3 role on flash-attention's mainloop (2026-10-17 12:55-13:16, every run 11/11 checks vs fp32
      causal reference):** `FA3_UPSTREAM=1` now builds (loader switch, hopper headers beside the merged include
      tree) and runs after one fix -- the launch sized dynamic smem from flashinfer's FA3 traits (163 920 B) while
      the body is flash-attention's kernel (compute-sanitizer: out-of-range shared address); `fi_a5_patch.py` takes
      the max with `UpstreamFA3Kernel::SharedStorageSize`. Upstream mainloop removed the LOP3/PRMT overhead but
      was no faster (prefill 2048 @ 64K 4.71 ms): consumers spin on mbarriers (NANOSLEEP 10M, SYNCS 31M). Paged-TMA
      KV (`FA3_PAGED_TMA=1`) is slower still (5.22 ms). Control arm `fa3only` (our build of flash-attention's
      kernel on upstream's own scheduler, 132 SMs): 4.65 ms = our candidate, so the scheduler adapter was not the
      long-item cost -- the build was.
    - [x] **A5.0e `-DNDEBUG` is the long-item gap (2026-10-17 13:21-13:25, 11/11 checks):** our FA3 extension was
      built without `-DNDEBUG`, so every device `assert` in flash-attention's scheduler / paged-KV / mainloop ran
      per tile (the ISETP 24M vs 6M, BRA 36M vs 10M excess). With it (upstream mainloop, fast math, 112 CTAs):
      prefill 2048 @ 64K 4.68 -> **3.61 ms** (sgl-kernel upstream 3.32 on 132 SMs: 1.09x), chat 2048 0.168 ->
      0.122 ms; `fa3only` 3.59 / 0.065 ms. Upstream's other build flags (GDC, extended MMA shapes, trace level 0)
      add nothing. Left: ~57 us fixed on short items in our compact path (0.122 vs fa3only 0.065 ms). The same
      extension compiles the co-launched FA2 executor, whose A4 "~20% mainloop gap" to stock FA2 was measured
      without `-DNDEBUG` too -- to re-measure. Full fixture table running (`a5_fa3_upstream_table_20261017.sh`).
      Scripts: `a5_fa3_upstream_try_20261017.sh` (UPS / PTMA / XN switches), `extra_nvcc_patch.py`,
      `upstream_patch.py`, `fi_a5_patch.py` (pod trees `up_a5`, `fi_a5`).
    - [x] **A5.0c NCU, ours vs upstream FA3 (2026-10-17 12:47, one launch each, both fixtures PASS):** same block
      (384), registers (168), smem (164 vs 167 KB) and occupancy (18.7% warps), but ours issues **2x the
      instructions** for the same work: prefill 2048 @ 64K 1.67G (fast math) vs 0.88G, chat 2048 34M vs 18M. The
      math is identical -- HGMMA 28.93M, MUFU 119.4M, FMUL / FFMA / FADD / FMNMX / F2FP all within 1% -- and the excess
      is ~790M integer/permute/shuffle instructions: LOP3 203M, PRMT 123M (one per score element), IMAD 111M (vs
      15M), SHFL 69M (vs 7M), SHF 65M (vs 5M), LEA 65M (vs 6M), IADD3 62M, plus 29M LDGSTS (cp.async KV gather; upstream
      uses TMA). **Root cause: the profiled kernel is `DedicatedFA3PrimaryKernel<SparseCollectiveMainloop<
      BatchPrefillPagedParams ...>>` -- flashinfer's own sparse/paged hopper mainloop, not flash-attention's.** The
      upstream-mainloop role (`hopper/fa3_upstream.cuh`, `FLASHINFER_FA3_UPSTREAM`) was never enabled by the loader
      after the 0.7 port (its last attempt failed on a missing flash-attention tree, then a `wgmma_plan` params
      mismatch, archive log 2026-10-09). Every FA3 number since -- NCU 43%, KV96, locality, unpacked, multicast,
      SM budget -- measured flashinfer's mainloop. Next: A5.0d, build the role on the upstream mainloop.
      `bash a5_fa3_ncu_20261017.sh`; `python ncu_compare.py <dir> <fixture>`; per-opcode via `ncu --import <rep>
      --page raw --csv --print-metric-instances details --metrics sass__inst_executed_per_opcode`.
    - [x] **A5.0b `--use_fast_math` (2026-10-17 12:39-12:46, ON/OFF interleaved x2, 112 CTAs, all 28 runs 9/9):**
      ON/OFF = 0.940-0.977 on every fixture (spread <=0.4%), e.g. prefill 2048 @ 64K 4.628 -> 4.411 ms, chat 2048
      0.171 -> 0.161 ms. Real but small: ours/upstream is still 1.37-1.38x on long items and 2.0-2.95x on short
      ones. Adopt fast math (upstream's own build flag); the gap is elsewhere -> A5.0c NCU, ours vs upstream.
      `bash a5_fa3_fastmath_20261017.sh` (tree `up_a5` = `up_tma_multicast` + `fastmath_patch.py`).
    - [x] **Gap located (2026-10-17 12:16-12:23, H200, all 28 runs 9/9 checks vs fp32 causal reference, repeat
      spread <=3.1%):** that NCU compare was confounded (ours on 112 CTAs vs flashinfer FA3 on 132, and flashinfer
      FA3 is not what serving runs). FA3-only fixtures, ms, two-repeat means:
      | fixture | ours 112 | ours 132 | flashinfer FA3 | upstream FA3 (serving) | ours/upstream |
      |---|---|---|---|---|---|
      | 8 x 1024 @ 32K | 11.03 | 10.25 | 7.42 | 5.99 | 1.71x (132) |
      | chat 2048 fresh | 0.171 | 0.173 | 0.081 | 0.060 | 2.88x |
      | chat 4 x 512 fresh | 0.069 | 0.073 | 0.038 | 0.032 | 2.32x |
      | append 512 @ 48K | 0.856 | 1.493 | 0.682 | 0.586 | 1.46x (112) |
      | agent 4 x 512 @ 64K | 4.75 | 4.77 | 4.11 | 3.28 | 1.45x |
      | prefill 2048 @ 64K | 4.65 | 4.65 | 3.96 | 3.21 | 1.45x |
      | 16 x 16 @ 1K | 0.073 | 0.045 | 0.071 | 0.024 | 1.86x |
      Three separate costs, none of them the SM budget: (1) ~1.45x mainloop on long compute-bound items;
      (2) ~0.1 ms fixed per call, dominant on chat-size items; (3) our split-KV merge path: append 512 @ 48K
      plans 112 tiles of 48K keys (one wave at 112 CTAs); at 132 it splits in two and gets 0.64 ms *slower*.
      Flashinfer FA3 is itself 1.16-1.35x behind upstream. Packed vs unpacked Q (Codex, <0.6%) and TMA vs
      gather KV (A5.5) were already ruled out. Our build lacks `--use_fast_math`, which upstream FA3 uses
      (softmax exp2 throughput bounds FA3 at head dim 128): A5.0b tests it next.
      Commands: `bash a5_fa3_gap_20261017.sh` (pod `/dev/shm/w`, harness
      `drivers/a5_fa3_reference_20261017.py` = tma_multicast_reference + an upstream sgl-kernel arm), both in
      `tasks/debug_kernel_server_perf/a5_fa3_20261017/`; binaries = tma_sm_budget GATHER arm (tree
      `up_tma_multicast`, `fi_tma_multicast`); sglang `ad2d19bab`, flashinfer `1a73e67`.
  - [x] A5.1 per-item q-tile choice (closed by bound, 2026-10-17): q64 wins only on short-KV few-row items (A1, up to
    24%), and those rows are verify-width and route to FA2 (A3: FA2 wins every verify-width row); FA3 rows are
    prefill/append chunks where q128 is never slower. No serving item reaches the win region.
  - [x] A5.2 split-KV reduction (2026-10-17): on the parity FA3 role the split path is exact (flash-attention's
    combine; the double merge fixed), but splitting itself loses where it fires (append 512 @ 48K at 132 CTAs: 0.679
    ms split vs 0.627 unsplit at 112), so the planner should not split FA3 rows; with no splits there is nothing to
    reduce, and the FA2 side's merge is stock BatchPrefill's. A faster reduction (JIT / DSMEM) only pays on a path
    the scheduler should avoid -- not built.
  - [x] A5.3 FA3 co-run with FA2 on the remaining SMs (2026-10-17): -18% where roles balance, <=2% serving bound.
    - [x] **Co-run vs serving split on mixed fixtures (2026-10-17 13:45-13:51, 30 runs, each 13/13 vs fp32 causal,
      repeat spread <=4.3% except chat 6.6%):** arms per run -- *split* (serving: sgl-kernel FA3 on all SMs, then
      stock FA2 BatchPrefill), *corun_sgl* (sgl-kernel FA3 with `sm_margin` = 132-B on one stream || stock FA2 on
      another), *corun_dyn* (our flash-attention build on its dynamic scheduler capped at B || stock FA2), *ours*
      (compact FA3 role, static lists + native FA2 as PDL dependent). Best B per fixture, vs split:
      | fixture (FA3 rows + FA2 q=4 rows) | split ms | ours | corun_sgl | corun_dyn |
      |---|---|---|---|---|
      | fresh 512 + 46 @ 48K | 2.335 | -0.2% (112) | +9.4% (96) | -0.8% (112) |
      | append 512 @ 48K + 46 @ 48K | 3.208 | -6.9% (120) | -16.7% (120) | **-17.9% (120)** |
      | prefill 2048 @ 64K + 8 @ 16K | 3.417 | +9.5% (120) | -3.3% (120) | +11.9% (120) |
      | chat 2048 + 32 @ 8K | 0.282 | +20.3% (120) | -1.7% (120) | +3.0% (112) |
      | agent 2 x 512 @ 64K + 24 @ 32K | 2.497 | +4.7% (112) | +0.1% (120) | +5.4% (120) |
      **Co-running pays when the roles take comparable time and one cannot fill the GPU alone** (append: a 0.6 ms
      FA3 wave hides under ~2.6 ms of bandwidth-bound verify rows, -18%); where one role dominates, sequential
      split ties or wins, and B=96 loses everywhere but append. So B must be chosen per step, "no co-run"
      included -- a dispatch decision (A2/A3), not a constant. corun_dyn trails corun_sgl wherever FA3 dominates:
      the remaining 1.1x per-SM gap of our build. Our static-list path (ours) loses on chat/agent (+5..+20%): its
      static per-CTA lists, as in A5.0f. Script `a5_mixed_corun_20261017.sh` (harness arms in
      `a5_fa3_reference_20261017.py`).
    - [x] **Same table on the parity build (paged TMA, 2026-10-17 13:59-14:04, 30 runs 13/13, spread <=5.7%):**
      corun_dyn (our FA3 role, dynamic scheduler on B SMs || stock FA2) vs split: append 512 @ 48K + 46 verify
      **-18.4%** (B 120); agent 2 x 512 @ 64K + 24 verify -1.8% (112); chat 2048 + 32 verify -1.2% (112); fresh 512 +
      46 verify -0.5%; prefill 2048 @ 64K + 8 verify +1.0% (corun_sgl -3.1%). corun_dyn now beats corun_sgl on every
      fixture but prefill; the static-list path never wins (up to +27% on short items) and is retired for the
      dynamic scheduler. Serving decision: co-run when both roles carry real work (an agent append beside many
      verify rows), split otherwise -- to be wired into the serving split path and measured end to end.
      `bash a5_mixed_corun_tma_20261017.sh`.
    - [x] **Serving bound for co-running (2026-10-17 14:2x, from the TraceLab C=4 step traces of 2026-10-16):** per
      mixed step (prefill rows + decode/verify rows), FA3 time = causal prefill FLOPs at 500 TFLOP/s, FA2 time =
      verify-row KV bytes at 3 TB/s; an ideal co-run saves min of the two. Our arm (EDF + deadline reservation,
      10,629 steps, 1,784 mixed): modelled attention is 79% of mixed-step time, but the roles are badly
      unbalanced -- min/max p10/p50/p90 0.01/0.03/0.18 -- so a perfect co-run saves **1.2% of all GPU time (2.1% of
      mixed-step time)**; stock MTP 0.4%. A long-context prefill chunk dwarfs the verify rows beside it, which is
      also why the 2026-10-13 serving overlap A/B found nothing. **Co-running cannot move serving metrics on agentic
      workloads; FA3 prefill speed is the lever**, and those items already run at ~580-600 TFLOP/s (prefill 2048
      @ 64K: 1.9 TFLOP per layer in 3.19 ms) with our role at parity with upstream. `corun_bound_from_steptrace.py`
      on `steptrace_tracelab_20261016/rep1/tracelab_c4/{MTP,EDF_RESV}/server_*.log`.
  - [x] A5.4 L2 locality (closed by measurement, 2026-10-17): on the parity build L2 hit rate is already 98% for both
    our role and sgl-kernel FA3 (NCU, prefill 2048 @ 64K / agent 4 x 512 @ 64K); the 2026-10-14 pairing gain was on
    the flashinfer mainloop's 12% hit rate.
  - [x] A5.5 KV multicast, FA3 and FA2 (closed by measured bounds 2026-10-25: FA2 shares <1% of its KV loads, FA3's
    zero-load step ceiling is 0.1-1.3%; reopened 2026-10-24 because the earlier tests below ran on the flashinfer mainloop or
    deadlocked on the flash-attention role, never under the unified plan). Prior B3 is merged here.
    - [x] 2026-10-24 **Shared-KV ceiling on the 96 captures** (`unified/kv_share_ceiling_20261024.py`, CPU only, page
      ids from each capture's page table; rows routed as in stage 1). Fraction of each role's KV page reads, per kv
      head, that another tile of the same step also reads (upper bound on load bytes, no causal trimming):
      | set | FA2: total shared | FA3: across rows | FA3: among one row's q tiles |
      |---|---|---|---|
      | 32-64K two | 0.83% | 0.12% | 99.06% |
      | 32-64K four | 0.92% | 0.07% | 99.09% |
      | 128-256K two | 0.42% | 0.04% | 99.09% |
      | 128-256K four | 0.31% | 0.05% | 99.10% |
      - **FA2 multicast is not built:** under 1% of FA2's KV loads are shared. Every FA2 row has one tile per kv head
        (rows > 32 packed go to FA3), and rows barely share prefixes.
      - **FA3:** nearly all KV is re-read by the same prefill row's other q tiles, which is the multicast case. Those
        re-reads are already L2 hits (98% on the parity build), and FA3 there is not load-bound (72% tensor-pipe
        activity). So multicast would save L2-to-SM traffic, not HBM traffic.
    - [x] 2026-10-25 **Zero-KV-load ceiling on the flash-attention role** (H20, 92 trace-sampled captures, A6 with the
      NP executor; `unified/a5_fa3_skipkv_ceiling_h20_20261025.sh`, `load_stock_fa3_variant.py`, flash-attention
      e9515d5 + `fa3_skip_kv_load_flash_attention.patch`, flashinfer `22d8d03`).
      - `stock_fa3_pdl` was rebuilt twice from one tree: a control, and a build with the paged K/V cp.async loads
        compiled out (the pipeline still commits).
      - The control matches the serving library (0.997-1.000x).
      - The skip build really removes the loads: its FA3 rows come out non-finite while its FA2 rows match the
        control exactly.
      | stage 1, zero KV loads vs control | 32-64K two | 32-64K four | 128-256K two | 128-256K four |
      |---|---|---|---|---|
      | step speedup (bound) | 1.013x | 1.003x | 1.003x | 1.001x |
      - Free KV loads would make the step at most 1.3% faster. A 2-CTA cluster removes at most half of the shared
        loads, so FA3 multicast is bounded at ~0.7%.
      - FA3 multicast under the unified plan is therefore not built. It would also have to pair tiles in lockstep
        on each cluster, which deadlocked upstream (below).
      - The flashinfer-mainloop figure of 19-22% (below) measured a load-bound mainloop that serving no longer runs.
    - [x] 2026-10-24 FA2 multicast (leader CTA loads with cp.async, then sends to its cluster peers over DSMEM):
      not built. The ceiling above leaves under 1% of FA2's KV loads to share.
    - Earlier results (2026-10-13..17):
    - Ceiling (2026-10-13, flashinfer mainloop, `b3_multicast_ceiling_20261013.sh` + diagnostic
      `fa3_skip_kv_load.patch`, [raw](debug_kernel_server_perf/resume_20261012/b3_multicast_ceiling_20261013.txt)). Zero-filling every FA3 K/V load saves
      4.9-6.6% on serving mixes, 16-20% on 32-token appends at 16-64K, and 19-22% on agent appends (4 x q512 @ 64K
      6.192 -> 4.812 ms, [append](debug_kernel_server_perf/resume_20261012/b3_append_findings_20261013.md)). A 2-CTA cluster removes at most half of the
      shared loads, so multicast is bounded at ~3% on serving mixes.
    - True TMA multicast on the flashinfer mainloop (Codex, 2026-10-13): `cp.async.bulk.tensor` over 2-CTA clusters
      at page 128, 9 gates + 48 timing cases pass, serving integration exact on 5 EAGLE3 arms. **-0.2..+1.7% vs
      unicast TMA**, within noise ([multicast](debug_kernel_server_perf/resume_20261012/tma_multicast_prototype_20261013.md),
      [paged serving](debug_kernel_server_perf/resume_20261012/tma_paged_serving_20261013.md)). Software KV sharing over DSMEM loses: 18.21 vs 5.94 ms
      redundant on 4 x 512 @ 64K, bulk exchange 60-63% slower ([dsmem](debug_kernel_server_perf/resume_20261012/kvshare_prototype_20261013.md),
      [bulk](debug_kernel_server_perf/resume_20261012/kvshare_bulk_20261013.md)).
    - flash-attention role (2026-10-17): ClusterM=2 with paged TMA deadlocks on the first call (GPU 100%, no progress
      for 8 min). Multicast needs both CTAs of a cluster on paired tiles in lockstep, which neither our lists nor
      upstream's varlen scheduler provide; flash-attention's own dispatcher disables clusters for causal / varlen /
      split. On the parity build NCU shows 98% L2 hit and 72% tensor-pipe activity, instruction-for-instruction
      equal to sgl-kernel FA3 (881.5M vs 881.2M on prefill 2048 @ 64K), so K/V loads are not the bound.
      `a5_fa3_ncu_parity_20261017.sh`.
    - Outcome: multicast is not pursued.
  - [x] A5.6 cross-role backfill. On A6 (2026-10-25) it is built and measured:
    - The FA3 launch triggers its PDL dependent at kernel entry (`griddepcontrol.launch_dependents` in
      `StockFA3PdlPrimary`) and drains its TMA stores (`tma_store_wait`) before exit. So FA2 CTAs fill each SM an FA3
      CTA frees, and FA2 rows run beside FA3's tail.
    - Worth 2.2-6.1% of the step: the PDL-off ablation, H200, 96 captures (A6 section 3, `a6_h200_dispatch_schedule_20261025.sh`).
    - In-kernel backfill (an FA3 CTA taking FA2 work itself) needs the co-compiled kernel, measured and dropped below.
    Earlier history: 2026-10-12, closed by bound 2026-10-17; the only backfill item, B2 merged here
    2026-10-24). On the flashinfer-mainloop fused path, co-launch hardware backfill was worth up to 2.9x when FA2 work
    exceeds the slots beside FA3 and nothing on small mixes; in-kernel `tma_store_wait` backfill was correct and worth
    1.4-5.5x over none, but that whole path was 2-6x slower, so co-launch shipped
    ([evidence](debug_kernel_server_perf/resume_20261012/colaunch_20261012.md)). Backfill only matters when FA3 and
    FA2 run concurrently, and the serving bound for any co-run is 1.2% of GPU time on TraceLab (A5.3). Not rebuilt on
    the flash-attention role.

### [x] B. Fuse draft/verify epilogue ops (bounded at ~4% of kernel time on A6, 2026-10-25, not built; backfill lives in A5.6 and multicast in A5.5)

- [x] **B1 re-bound on A6 (2026-10-25)** (H200, one stage-2 pair, mix32 four-mode, agent C=8 + chat 8/s; torch
  profiler over 60 steps of each arm 150 s into its run; `unified/a6_epilogue_bound_h200_20261025.sh`,
  `drivers/serving/parse_kernel_tail.py` with A6's kernel names added to the attention bucket):
  | arm | GPU busy of span | GEMM | attention | index / copy | elementwise / norm | sampling / verify | other |
  |---|---|---|---|---|---|---|---|
  | stock FA3 | 72.1% | 81.1% | 11.3% | 3.6% | 2.4% | 0.6% | 1.0% |
  | ours, A6 | 84.4% | 80.5% | 12.0% | 3.3% | 2.7% | 0.6% | 0.9% |
  - The draft / verify epilogue lives in the sampling / verify bucket (0.6%) and part of index / copy (<= 3.3%). The
    elementwise / norm kernels are the model's own layer ops (act-and-mul, RoPE, norms), not the epilogue.
  - So fusing the epilogue is bounded at ~4% of kernel time on A6, against GEMM's 80%; not built.
  - In this window ours keeps the GPU busier (84 vs 72% of the span).
- [x] **B1, fusion headroom (2026-10-13, measured):** 60 serving steps per arm profiled (torch
  profiler, GPU activity), mix32 chat 16/s + agent 0.1/s, race-fixed tree, no barrier
  (`b1_tail_profile_20261013.sh`, `drivers/serving/parse_kernel_tail.py`).
  | arm | span / GPU busy | attention | GEMM | small ops (index+elementwise+sampling+other) |
  |---|---|---|---|---|
  | fused 4-mode | 4147 ms / 87.2% | 2739 ms (74.9%) | 809 ms (22.1%) | 2.9% (29.6K launches) |
  | unfused 4-mode | 4895 ms / 91.1% | 3574 ms (80.1%) | 783 ms (17.6%) | 2.4% (27.6K launches) |
  | stock MTP | 3635 ms / 90.8% | 2500 ms (75.7%) | 718 ms (21.7%) | 2.7% (24.4K launches) |
  **Epilogue fusion is bounded at ~3% of kernel time** (+ the 9-13% GPU-idle share, part of it
  host-side), far below the bar; B1's further fusion is not worth building. Kineto's
  "est. achieved occupancy" reads 0.4-1% for attention/GEMM (persistent / graph-launched
  kernels) and is not usable -- real occupancy needs Nsight Compute. packed_qo histogram (fused,
  5202 plans): 82,833 verify rows with packed_qo <= 32 carry 4.4% of useful FLOPs at 56.8%
  in-tile waste; 11,141 rows >= 512 carry 95.6% at 1.0%; overall waste 6.3%. Verify rows are
  memory-bound, so their FLOP waste is not the cost -- their KV bandwidth is (A4). Unfused vs fused
  over the same 60 steps: attention 3574 vs 2739 ms (-23%), span -15%. Per context range (chat 16/s + agent 0.1/s, 60 steps each,
  `b1_tail_profile_mix128_20261013.sh`; steps are sampled from live traffic, so step mixes differ
  between arms -- per-call times are the cleaner comparison):
  | range | arm | span | attention | small ops | main attention kernels (avg per call) |
  |---|---|---|---|---|---|
  | mix64 | fused | 16.2 s | 13.3 s (84.7%) | ~1.4% | BatchPrefill 7.20 ms (decode-only steps), FA2Executor 5.34 ms (mixed) |
  | mix64 | unfused 4-mode | 17.7 s | 14.2 s (82.7%) | ~1.5% | BatchPrefill 7.22 ms (all steps) |
  | mix64 | stock | 22.5 s | 17.7 s (80.0%) | ~1.6% | BatchPrefill 9.44 ms |
  | mix128 | fused | 3.6 s | 2.2 s (70.1%) | ~3.9% | BatchPrefill 1.29 ms (decode-only), FA2Executor 0.83 ms (mixed) |
  | mix128 | unfused 4-mode | 7.4 s | 6.0 s (86.1%) | ~1.8% | BatchPrefill 3.15 ms (all steps, long prefill rows on FA2) |
  | mix128 | stock | 6.8 s | 5.8 s (88.6%) | ~1.4% | BatchPrefill 3.14 ms |
  Small ops stay at 1.4-3.9% at every range, so B1's bound holds everywhere. At mix128 the unfused
  merged batch runs its 128-256K prefill rows through FA2 BatchPrefill (3.15 ms/call); the fused
  step sends them to FA3 and the verify rows to the FA2 executor (0.83 ms/call) -- the kernel-side
  mechanism behind the per-range attribution above. Tile waste falls with context (mix32 6.3%,
  mix64 2.7%, mix128 0.6%) because long prefill rows dominate useful work.
  - [x] **True GEMM-epilogue fusion bounded (2026-10-15, user: "fuse them if you can get a real
    gain over PDL"):** the existing fused verify/chain epilogues are separate Triton launches, not
    epilogues. The largest candidate for a real epilogue, the target argmax inside the LM-head GEMM
    (Qwen2.5-7B, vocab 151936), can save at most the argmax pass plus the logits round trip:
    19 / 72 / 235 us at 48 / 184 / 1024 verify tokens (LM-head GEMM 270 / 338 / 1717 us), i.e.
    ~0.2-1% of a 20-40 ms serving step -- below the 2-3% serving noise floor, so no measurable
    gain over PDL chaining; not built. bf16-logit argmax disagrees with an fp32 reference on 2-3%
    of rows (ties), so an fp32-accumulator epilogue would be more exact, not faster.
    (`b1_lmhead_epilogue_ceiling_20261015.py`, log `b1_lmhead_ceiling_20261015.log`)
### [ ] S. Server / workload evaluation

- [ ] **Paper workload and SLO matrix (2026-10-25, user: SLOs from the papers' common settings; TTFT deadline x2 for
      prefills > 200K; agent share > 50% in the mix; 16 GPUs = 4 pods x TP4 first, then 32; Qwen2.5-7B and
      Llama-3.3-70B first, then GLM).**
  - SLOs per class (sources: JITServe, SLOs-Serve section 6; Crossflow states no evaluation SLO):
    | class | TTFT | TPOT (mean per request) | source |
    |---|---|---|---|
    | chat / ShareGPT / math / code | 2 s; also 3x / 5x the unloaded TTFT | 50 ms tight / 100 ms loose | JITServe; SLOs-Serve |
    | web agent (WebShop) | 20 s end to end x stages | -- | JITServe compound |
    | TraceLab agent round | 10 s if <= 200K uncached tokens, 20 s above | 80 ms tight / 100 ms loose | JITServe deadline-sensitive; user rule for > 200K |
    | every class | SLO scale swept 0.8-1.5x | | JITServe |
  - Decisions on the conflicts in our scoring:
    - Chat 0.5 s / 20 ms had no source and is unreachable for Llama-70B (chat TPOT 97-135 ms in both arms). It is
      replaced by the table above.
    - Llama-3.3-70B uses the relative form (3x / 5x unloaded TTFT and TPOT), which needs the unloaded reference runs.
    - TPOT is the per-request mean (DistServe / SLOs-Serve); p99 ITL is reported beside it, never mixed into
      attainment.
    - Scheduler deadlines (EDF / router) are kept separate from evaluation SLOs, and both are stated.
    - The > 200K tier is applied in one scorer for every figure.
  - [ ] (a') **Demand shift on the superposed TraceLab replay, 16 H20s, full stack** (2026-10-27, running;
        `pd_colocation/mn_tracelab_shift_v2_16g_20261026.sh`, log `/tmp/h20/mn_tracelab_shift_16g_r1.log`; scorer
        `perf_model/tracelab_windows.py --windows 30`; client overlay and replay `drivers/serving/bench_overlay_20261026/`).
    - Setup:
      - the 24 busiest non-overlapping trace hours superposed on one clock; ShareGPT chat at 8 req/s;
      - each arm measured over 1.5x the scaled trace hour; rounds past it are not issued and count as backlog.
    - Paper SLOs, one repeat. Chat 2 s + 50 ms (2 s + 100 ms) ; agent 10 s + 80 ms attainment ; agent SLO goodput ;
      agent rounds issued (not issued):
      | alpha | ours | conditional disagg | stock PD 2 + 2 | stock PD 3 + 1 |
      |---|---|---|---|---|
      | 6 (overload) | 31.8 (56.8)% ; 50.0% ; 890 tok/s ; 4818 (8254) | 57.2 (67.6)% ; 51.1% ; 720 ; 3850 (9222) | rerun `_r1b`: 6.2 (6.2)% ; 4.9% ; 23 ; 1096 (11757, plus 219 lost); the first run's agent client aborted (chat 8.2%) | 3.6 (3.6)% ; 1.9% ; 6 ; 1227 (10594, plus 1251 lost after failed rounds) |
      | 3 | 65.1 (81.2)% ; 72.6% ; 1501 tok/s ; 8995 (4077) | 50.9 (68.6)% ; 63.9% ; 956 ; 7264 (5808) | 9.9 (9.9)% ; 11.6% ; 49 ; 1457 (4903, plus 6712 lost after failed rounds; 3568 chat requests failed) | 2.2 (2.2)% ; 1.8% ; 7 ; 1971 (7162, plus 3939 lost) |
    - alpha 3 + chat surge (base / surge chat):
      - ours 57.4 / 46.2% ; 69.3% ; 1266 tok/s ; 8554;
      - conditional 36.5 / 27.1% ; 53.8% ; 668 ; 6081;
      - stock PD 2 + 2: 17.2 / 11.1% ; 4.7% ; 13 ; 1169 (4769 failed chat requests, 10677 agent rounds lost after
        failed rounds);
      - stock PD 3 + 1: 2.2 / 0.0% ; 2.7% ; 9 ; 1894 (8020 not issued, 3158 lost after failed rounds).
    - v2 driver complete 2026-10-27 08:5x; the stock PD 2 + 2 agent side at alpha 6 is rerun in the chunk driver.
    - Second repeat of ours at alpha 3 (the traced arm `_tl2_a3_steptrace_r1`, fixed 2048 chunk): chat 68.5 (84.3)% ;
      agent 78.7% ; 1763 tok/s ; 9665 issued (3407 not). This is above r1 (65.1% ; 72.6% ; 1501), so the gap to
      conditional disaggregation holds.
    - alpha 3, ours vs conditional disaggregation:
      - chat attainment +14 points (+28% relative), chat TTFT p90 1.12 vs 8.74 s;
      - agent attainment +8.7 points, agent SLO goodput +57%, rounds issued +24%, agent TTFT p90 8.2 vs 21.2 s.
    - alpha 6: ours issues 25% more rounds and carries 24% more agent goodput, but its chat TPOT p50 is 72 vs 28 ms,
      which the SLO-budgeted chunk test addresses.
    - Both stock PD splits collapse at alpha 6 (median TTFT about 200 s for both tenants).
  - [ ] (a) TraceLab replay as Crossflow does. **Tooling built and tested offline (2026-10-26, 31/31 checks,
        `drivers/serving/test_tracelab_window.py`; overlay `drivers/serving/bench_overlay_20261026/`):**
        - `AGENT_MODE=tracelab_ts WINDOW_START=<s> ALPHA=<a> WINDOW_ROUNDS=segment` in `mn_bench.sh`;
        - natural-text prompts from ShareGPT by default;
        - math tenant via `MATH_RATE`.
        Decisions: `segment` rounds, Crossflow's one-hour segment (session mode replays for days on this trace).
        The trace is sparse (<= 0.17 calls/s per hour), so alpha ~20-100 is needed to reach capacity. Candidate
        windows: 19537200 (24 sessions / 519 rounds), 14868000 (16 / 369, most uncached input), 19692000 (11 / 602).
        Not yet run end to end on a pod. Spec: one-hour windows of `syfi_coding_trace` (deduplicated), session starts at
        the trace's own timestamps / alpha, alpha swept around capacity. Report measured in-flight programs, LLM
        calls/s, uncached-input and output tok/s per point. Missing:
        - the timestamp-scaled replay (`session_replay.py` draws Poisson starts);
        - natural-text prompts (random vocabulary gives MTP near-zero acceptance).
  - [ ] (b) Mixed: TraceLab agents (program concurrency C) + chat + math, agent 60% / chat 30% / math 10% of LLM
        calls/s. Sweep 4-5 C points from under capacity to overloaded around 90% attainment, plus the program and chat
        spikes. Missing: a third stream in `mn_bench.sh` (only one `SECOND`); the stage-3 agent tenant is synthetic
        fan-out today.
  - [ ] (c) **Running 2026-10-26 on the H200** (`unified/a6_standalone_h200_20261026.sh`: ShareGPT 16 / 32 req/s,
        MATH-500 16, GSM8K 32, Code-Feedback 16; final config vs stock, four / two-mode, two repeats; harness
        `--standalone` mode, `apply_bench_pair_standalone_20261026.py`). Datasets were prepared on the H200
        (`/dev/shm/w/workloads/standalone`). WebShop is not yet prepared.
        - [x] ShareGPT 16 req/s: level in both modes (chat attainment 99.7-100% both arms; tok/s within 0.3%).
        - [x] ShareGPT 32 req/s two-mode: level (tok/s 6283-6285 vs 6273-6286); ours 98% vs 94-95% at 1 s / 100 ms.
        - [ ] **ShareGPT 32 req/s four-mode: REGRESSION, root-caused.**
              - Ours 5909-5983 vs 6401-6418 tok/s (-7%); TTFT 3.3-4.2 s vs 0.16 s; attainment 92% vs 100%.
              - Cause: `SGLANG_MIXED_SPEC_MIN_QUEUE=4` re-enables the 4-mode mixed step for short prefills whenever
                4+ requests wait. The prefill CUDA-graph runner refuses MIXED batches, so 2979 of 3247 prefill steps
                ran eager (stock MTP: 0 of 3190).
              - This is a feedback loop: lower throughput, longer queue, more mixing; queue 114 vs 2.7.
              - The queue trigger was justified by long-context runs, where steps are GPU-bound and the eager cost
                is hidden.
              - [ ] Control with the queue trigger off (prefills >= 8192 tokens still mix):
                    `unified/a6_standalone_noq_h200_20261026.sh` (queued).
              - [x] **The trigger's value on the mixed workload** (GLM-4.5-Air overloaded four-mode, pod 4,
                    `short_a6_nodc_noq`, 2026-10-26, two repeats).
                    - With the trigger off, agent throughput vs stock is +14% (25.3 vs 22.1 tok/s, r2), against +28%
                      with it on.
                    - Chat attainment at 2 s / 100 ms: 70.6-74.5% vs 76.7-77.5% with it.
                    - Eager prefill steps: 563 of 1306 vs 1078 of 1084.
                    - So the trigger pays when long agent rows are running and costs when every row is short.
              - [ ] **Root fix (user, 2026-10-26): the 4-mode merged step must replay CUDA graphs under
                    `--enable-mixed-chunk`.**
                    - Plain mixed-chunk batches already replay: two-mode ShareGPT runs 3472 of 3473 prefill steps on
                      graphs in both arms.
                    - The eager steps are only the merged steps (prefill chunk riding in the verify forward):
                      2979 of 3247.
                    - Diagnosis running: `SGLANG_PCG_REASON=1` counts which eligibility check of
                      `PrefillCudaGraphRunner` refuses each batch (`unified/apply_pcg_reason_20261026.py`,
                      `unified/a6_pcg_diag_h200_20261026.sh`).
                    - Then implement the replay for that path. The long-row trigger below is a stopgap, not the
                      fix; its ShareGPT test was cancelled for this.
                    - Design requirements (user, 2026-10-26):
                      - Mixed steps run under CUDA graphs.
                      - Sweep the captured sizes and padding to balance speed against memory, so large batches can
                        still be admitted.
                      - Pad with request padding as sglang does (zero-length sentinel request slots).
                      - Since prefill, decode and mixed steps use the same kernels, capture one shared graph set and
                        change only how each graph's batch is filled.
                    - Current code: the default prefill backend ("breakable") already captures the transformer body
                      per token-count bucket. Attention metadata, attention and the logits tail run outside the
                      graph, so a body graph is independent of request composition. The "full" backend adds
                      request-slot padding (zero-length sentinels up to a captured slot count). Decode / verify use
                      a separate per-bs graph set (`decode_cuda_graph_runner`, attention inside).
                    - Diagnostic run 1 failed at server start: the patch put `import os` above
                      `from __future__`. Fixed with a `compile()` check.
                    - **Run 2 corrects the diagnosis** (ShareGPT 32 req/s four-mode):
                      - The target's extend forwards DO replay: 3369 replays vs 3302 prefill steps.
                      - The batch log's "cuda graph: False" on merged steps is hard-coded
                        (`forward_mixed_spec` returns `can_run_cuda_graph=False`), not a measurement. My earlier
                        "2979 eager steps" counted that flag.
                      - The real refusals are ~9000 other forwards (~3 per step). All fail the hidden-state check:
                        `capture_hidden_mode` of the graph < requested.
                      - Suspect: the draft model's extend in the merged step requests FULL, while the breakable EAGLE
                        draft captures LAST.
                      - Runs 3-5:
                        - The refusals are the DRAFT model's prefill-graph runner (captures LAST), on
                          `DRAFT_EXTEND_V2` forwards requesting FULL, bs 45-48: the verify rows' draft extend falling
                          through to eager.
                        - The draft-extend graph runner logged NO refusals, and neither arm logged a draft-extend
                          capture, so that path is eager in both arms. The refusal counter ran only in ours
                          (`--ours-env`), so "stock: 0 refusals" was not evidence. The draft path does not explain
                          the gap.
                    - **Explanation consistent with all runs:**
                      - Stock MTP runs verify as its own step on the decode CUDA graph (whole forward incl.
                        attention, one launch), and a short prefill on the breakable prefill graph.
                      - The 4-mode step merges both into one forward on the breakable prefill graph, where attention
                        (and its metadata, and the A6 plan) runs eager per layer.
                      - On short-prompt, high-rate ShareGPT the extra CPU launch work per step is the -7%. On long
                        contexts GPU time hides it.
                    - **Plan (user's design):**
                      - One shared graph set keyed by token-count bucket, attention inside the graph, request-slot
                        padding with zero-length sentinel requests, attention metadata refreshed outside the graph
                        (the shape of sglang's "full" prefill backend). It serves decode, verify, prefill and mixed
                        steps; only the batch fill changes.
                      - A6 must be graph-capturable: the plan writes static device buffers (the device-metadata path
                        already does), and launch grids are fixed. Stock FA3's persistent scheduler is fixed-grid.
                        The FA2 NP kernel launches one CTA per item, so it needs a max grid with early exit. The
                        persistent FA2 executor is graph-friendly as built.
                      - Sweep captured buckets / slot counts against memory so large batches still fit.
                      - Experiment 1 progress:
                        - Upstream asserts "Mixed chunk prefill requires the breakable prefill CUDA graph backend"
                          (no reason given). Lifted behind `SGLANG_FULL_CG_MIXED=1`
                          (`unified/apply_full_cg_mixed_20261026.py`), 64 request slots.
                        - Stock MTP under the full backend: 6412 tok/s (breakable: 6401-6418).
                        - The 4-mode arm crashed on merged steps: the full backend's static logits buffer had one row
                          per request, but a merged step keeps every verify token's logits
                          (`eagle_sample: shape [5, 4] invalid for 5`). Fixed: rows = batch_size - verify_rows +
                          verify_tokens.
                        - Still failing with the same error: the logits processor computed the right rows
                          (`mvt=16`, extend lens `[199, 4, 4, 4, 4]`, 17 rows), but `_trim_logits_output` cut the
                          full backend's output to one row per request (`raw_bs`). Fixed: trim to the per-step logits
                          row count recorded at load.
                        - [x] **Experiment 1 result** (ShareGPT 32 req/s four-mode, full backend both arms, our 4-mode
                              schedule on stock FA3 attention, one repeat):
                              - **Correct:** greedy outputs agree with the breakable runs at the noise floor (44-45%
                                identical vs 45-47% between two breakable repeats; common prefix 62-64% vs 63-65%).
                              - Target forwards replay full graphs in both arms (~3000 replays; no refusals).
                              - **Still -6.2%** (6025 vs 6424 tok/s; TTFT 3.7 s vs 0.13 s).
                              - So graph coverage of the target's merged step is NOT the cause.
                        - [x] **Experiment 2** (`unified/a6_fullcg_exp2_h200_20261026.sh`): ours with SGLANG_MIXED_SPEC=0
                              (EDF + stock MTP steps), same full backend: 6393 vs 6426 tok/s (-0.5%), TTFT 67 vs 134 ms.
                              - The deadline scheduler is not the cost; **the merged step is**.
                              - With the merge on, the same prefill work (1.20M tokens in ~3200 batches) leaves 46
                                running vs 35 and a queue of 105 vs 2, and TPOT is 8.0 vs 5.7 ms.
                              - Both paths run the same forwards, and every forward replays a graph (no refusals in
                                either arm under the full backend). So the time is lost outside the GPU forward
                                work: suspect host-side cost or broken CPU / GPU overlap in the merged path.
                        - [x] **Experiment 3, root cause found** (`SGLANG_STEP_TRACE=1` in both arms;
                              `perf_model/steptrace_wall_vs_gpu_20261026.py`):
                              | arm | step type | n | GPU ms | wall ms |
                              |---|---|---|---|---|
                              | ours | merged (prefill + verify) | 3213 | 22.0 | **27.5** |
                              | ours | verify | 5780 | 8.2 | 8.2 |
                              | stock | MIXED (prefill + decode tails, upstream) | 3266 | 14.2 | 11.6 |
                              | stock | verify | 10066 | 7.3 | 8.7 |
                              - GPU work matches (22.0 vs 14.2 + 7.3 = 21.5 ms). But the merged step's wall time
                                exceeds its GPU time by 5.5 ms, while stock's steps overlap the next launch.
                              - Cause: `mix_with_verify` calls `running_batch.seq_lens.cpu()` (the running rows'
                                committed prefix). Under the overlap scheduler that waits for the GPU queue to drain,
                                every merged step. The code's own TODO(sync) notes it.
                        - [ ] **Fix: sync-free merge** (`SGLANG_MIX_SYNC_FREE=1`, `unified/apply_sync_free_mix_20261026.py`).
                              - The host copies of the verify rows' prefix / seq_lens become an upper bound (mirror +
                                n; the device leads the mirror by at most one step's acceptance).
                              - The exact prefix stays on the device, and `ForwardBatch.init_new` writes it into the
                                verify rows of `extend_prefix_lens` before positions are computed.
                              - The stock FA3 path reads the host values only for maxima.
                              - A6 needs one more change: its compact KV index list is built from host lengths
                                (`torch.cat(page_table[i, :kv[i]])`) while its indptr comes from device lengths, so it
                                must gather on the device (`create_flashinfer_kv_indices_triton`). Its plan tolerates
                                upper bounds because the kernels clamp chunks to the device KV length.
                              - Experiment 4 (stock FA3, full graphs, step trace): running.
                        - Correctness gate: `drivers/serving/compare_outputs_20261026.py` (greedy-output agreement vs
                          the breakable runs, two breakable repeats as the noise floor).
                      - [ ] Experiment 1 (`unified/a6_fullcg_exp1_h200_20261026.sh`): the full prefill
                            backend (`--cuda-graph-backend-prefill full`, via the harness's new `--server-arg`) in BOTH
                            arms, our 4-mode schedule on stock FA3 attention, ShareGPT 32 req/s four-mode, refusal
                            counters in both arms. The FA3 backend already has full-graph prefill metadata
                            (`_init_full_cg_prefill_metadata`). With topk = 1, a verify row is a plain 4-token causal
                            extend, so a merged step has a prefill batch's attention shape.
                      - Steps: (1) check the full backend's support for the FA3 attention backend and for
                        target-verify (spec) batches; (2) make A6 capturable; (3) enable the merged 4-mode step;
                        (4) sweep buckets and padding; (5) ShareGPT 32 req/s four-mode + the mixed workload.
              - [ ] **Stopgap: `SGLANG_MIXED_SPEC_QUEUE_LONG_ROWS=1`** (`unified/apply_mixed_spec_queue_long_20261026.py`,
                    opt-in). The queue trigger needs a running row with context >= `MIN_CTX`. Under test:
                    - [x] GLM overloaded four-mode (pod 4, `short_a6_nodc_qlong`, two repeats): agent 27.6 / 28.9 vs
                          21.8 / 21.6 tok/s (**+27 / +34%**, vs +28% with the plain trigger). Chat attainment at
                          2 s / 100 ms is **81.2-82.4%** (plain trigger 76.7-77.5%), chat e2e 28-31 s (36-40 s).
                          Prefill steps: 739 eager / 397 graph (plain trigger 1078 / 6). The rule keeps the gain and
                          improves chat.
                    - ShareGPT 32 req/s four-mode (`unified/a6_standalone_qlong_h200_20261026.sh`, H200): must
                      remove the regression.
                    If both hold, it becomes part of the final configuration.
        - [x] MATH-500 16 / GSM8K 32 / Code-Feedback 16 req/s (two repeats each, `unified/a6_standalone_resume_20261026.sh`,
              summary `/dev/shm/w/ub/st_sum.py`): ours vs stock tok/s -0.0..+0.2% in both modes, chat attainment at
              2 s / 50 ms 100% for both, TTFT 20-24 vs 19-30 ms. Level, as expected for short prompts below
              capacity.
        Standalone: ShareGPT, GSM8K / MATH-500, HumanEval / Code-Feedback, WebShop (`prep_standalone.py`,
        `prep_webshop_agentic.py`); Poisson rate around capacity; WebShop at C = 8 / 16 / 32 / 64. This is the
        no-regression case (one H200 so far: ShareGPT at saturation +3% throughput, -15% TTFT).
  - [ ] Figures: per-tenant SLO attainment vs load; goodput at 90% attainment; tightest SLO scale met; throughput vs
        tok/s/user Pareto with p10-p90; TTFT and ITL (mean, p99) side by side; recovery time after a load step; share
        of requests with > 200 / 500 ms stalls; results by uncached-prompt bucket.
  - [ ] Unloaded reference runs per model (for the relative SLOs).

- [ ] **S0b, EVERY single-node comparison must be redone on sglang's FA3 backend (2026-10-22).** The backend-class
  control (`tasks/realistic_workload_analysis.md`, "THE COST IS THE FLASHINFER BACKEND CLASS") measured the same stock
  schedule at 161.2 agent tok/s on the FA3 backend and 81.1 on the flashinfer backend with no split at all. Our split
  was built inside the flashinfer backend, which upstream deprecated on Hopper for that regression; our FA3 patches
  recover 81 -> 147 but cannot reach 161 from inside it. Our scheduling (EDF, deadline chunk, decode-row cap, SLO
  admission, 4-mode MTP) lives in `scheduler.py`, so it runs on the FA3 backend with no split and no port -- making
  the schedule, not the backend, the thing under test. The definitive table is
  `bash tasks/pd_colocation/h200_fa3_table_20261022.sh`: five schedules, every arm on FA3, two repeats, per-tenant
  attainment.
  **Result (2026-10-22, COMPLETE, 10 of 10 arms; chat 8 req/s, agent program concurrency 8, chat template, real head,
  every arm 0 tracebacks / 0 fallbacks). BAR NOT MET at this load.** rep1 / rep2, primary SLOs (agent 10 s + 80 ms,
  chat 0.5 s + 20 ms):
  | arm | agent att. | chat att. | agent tok/s | agent att. @30 ms ITL |
  |---|---|---|---|---|
  | stock mixed chunk | 84.4 / 84.4% | 79.9 / 79.8% | 141.8 / 141.8 | 33.3 / 33.3% |
  | stock prefill-prioritized | 86.7 / 84.4% | 80.2 / 80.3% | 142.1 / 141.7 | 31.1 / 33.3% |
  | stock prefill-prioritized + MTP | 66.7 / 68.9% | 77.0 / 77.0% | 161.2 / 161.5 | 46.7 / 48.9% |
  | ours no MTP | 86.7 / **95.6%** | 78.2 / 78.0% | 133.9 / 134.2 | 28.9 / 24.4% |
  | ours + MTP (4-mode) | 62.2 / 64.4% | 75.6 / 75.5% | 140.6 / 141.5 | **55.6 / 51.1%** |
  Readings, in order of how much they matter:
  - **The bar is not met here.** Ours no MTP is +6.5% agent attainment over stock prefill-prioritized (91.2% vs 85.6%
    on rep means), -2.5% chat attainment and -5.5% throughput (134.1 vs 141.9 mean); ours + MTP is behind stock + MTP
    on both primary attainments (-6 and -1.5 points) at -12.5% throughput (141.1 vs 161.4). Nothing approaches
    +30% / +10%. Both repeats agree to within 0.3% on throughput for every arm, so the throughput deficits are real
    and not noise -- unlike the agent-attainment numbers below.
  - **Ours' agent attainment is noisier than the effect I had been quoting**: 86.7% and 95.6% across two repeats of
    the same configuration. With n=45 agent requests one request is 2.2 points, so a 9-point spread is 4 requests.
    Any single-node agent-attainment claim needs more repeats than two, and the earlier "+9 points" and "95.6%"
    headlines should be read as 86.7-95.6%.
  - **Under the tighter ITL SLO the ordering flips in our favour**: at agent 10 s + 30 ms, ours + MTP is 55.6% vs
    stock + MTP's 46.7-48.9% (+14-19%), and both MTP arms beat both no-MTP arms (46-56% vs 24-33%) because
    speculation halves ITL. The 80 ms target is loose enough that it rewards stock's throughput instead; the 30 ms
    target is where a scheduler that protects ITL shows up. Worth stating in the paper as SLO-dependence rather than
    quoting one threshold.
  - **On RAW TTFT / ITL the bar is met on three of six metrics against sglang's actual default, and which baseline
    you pick decides the verdict.** Ours no MTP vs stock mixed chunk (the default; rep1):
    | metric | stock mixed chunk | ours no MTP | delta |
    |---|---|---|---|
    | chat TTFT mean | 579.5 ms | 132.8 ms | **-77%** |
    | agent ITL p99 | 417.8 ms | 120.2 ms | **-71%** |
    | chat ITL p99 | 293.1 ms | 111.9 ms | **-62%** |
    | agent TTFT mean | 2356 ms | 3692 ms | +57% |
    | chat ITL mean | 13.1 ms | 14.5 ms | +11% |
    | agent throughput | 141.8 tok/s | 133.9 tok/s | -5% |
    But against stock PREFILL-PRIORITIZED (not the default -- it needs mixed chunk off) ours' chat ITL p99 is 111.9
    vs **27.3 ms**, 4x worse, because that schedule never mixes a prefill chunk into chat decode. So no schedule
    dominates: stock mixed chunk has good agent TTFT and bad ITL tails, stock prefill-prioritized has excellent ITL
    tails and poor chat TTFT, ours has the best chat TTFT (4x) and the best agent ITL tail while paying in agent TTFT
    and chat ITL tail. **Our schedule moves latency from chat TTFT to agent TTFT, and from the agent ITL tail to the
    chat ITL tail** -- a redistribution, not a uniform win, and the paper has to say which tenant it is protecting.
  - Consistent with the diagnosis that this load does not saturate stock (chat throughput = offered rate in every
    arm, nothing dropped). The agent-concurrency sweep (`h200_agentload_20261022.sh`, C=16 / 24) is the test that
    can still clear the bar on throughput; if it does not, the single-node claim should be dropped in favour of the
    multi-node one.

- [ ] **S0, three-model evaluation flow (user, 2026-10-21; models revised 2026-10-24): kernel microbench -> single
  node -> 16 GPUs -> 32 GPUs, real acceptance only, chat prompts through the chat template, disagg at its best P:D
  ratio.** Models (2026-10-24): Qwen2.5-7B + trained EAGLE3 head (thoughtworks), Llama-3.3-70B first, GLM-4.5-Air +
  its own MTP layer; Qwen3-8B is dropped. Single node = one H200 (7B) or one 4-GPU H20 pod at TP4; multi-node = H20
  job 2f608c1b1a67cf09 (8 x 4) at 4 and 8 pods. The kernel stage for every model is A6.
  - [ ] **Kernel microbench (2026-10-21, idle H200, 132 SMs; every arm in every case within 4.6e-3 of an fp32 causal
    reference). Reopened: the unfused baseline was the wrong stock.** One mixed attention step per layer: a prefill chunk
    (1024 @ 16K / 49K, 2048 @ 32K) + 16 or 48 rows at 4-64K KV; 2-mode rows = 1 decode token, 4-mode rows = 4 verify
    tokens; Qwen2.5-7B heads unless noted.
    - [x] **Baseline (2026-10-21).** Upstream's default backend on Hopper for MHA models with no spec or topk 1 is FA3
      (`get_default_attn_backend`, `arg_groups/model_override_base.py:296`, citing a flashinfer regression on Hopper);
      `mn_node.sh` forced `--attention-backend flashinfer` on every arm, stock included. So stock's mixed step is ONE
      varlen FA3 call over chunk + rows, not FA2 BatchPrefill. `ATTN_BACKEND=fa3` now selects it.
    - [x] **Rows set the step on FA2 BatchPrefill, at any KV layout (2026-10-21; `rows_kernels_20261021.py`,
      `fusion_decompose_20261021.py` -> `fusion_decompose_20261021.txt`, `rows_layout_check_20261021.py` part 1).**
      48 rows: FA2 BatchPrefill 7.4-8.2 ms vs FA3 0.90-0.93 ms, FA2 BatchDecode (q1) 0.98-1.04 ms, persistent FA2
      (`flashinfer.BatchAttention`) 2.14 ms (q4) / 1.00 ms (q1), KV-read floor 0.78 ms; the same with KV slots
      scattered per token, in 16-token runs, or contiguous. The chunk alone is close on either kernel (FA3 0.49-3.38 vs
      FA2 0.48-3.85 ms).
    - [x] **Against flashinfer stock (2026-10-21; `fusion_microbench_20261021.py`, raw
      `fusion_microbench_20261021.txt` and `fusion_microbench_rows3_20261021.txt`).** Split with FA2 rows: -6..+7%;
      co-run with FA2 rows: -0.4..-17%. Rows on FA3 (`apply_split_rows_fa3.py`): -49..-85% sequential, -57..-89%
      co-run, all three models (Qwen3-8B 32 / 8 heads, GLM-4.5-Air TP4 24 / 2). This is FA3 vs FA2 BatchPrefill on the
      rows, not fusion.
    - [x] **Against FA3 stock (2026-10-21; `python tasks/debug_kernel_server_perf/rows_layout_check_20261021.py`, raw
      `rows_layout_check_20261021.txt`; sglang 6d3f344af).** Split sequential (FA3 chunk, then FA3 rows): -5..+20% --
      no gain, a second launch over the same work. Co-run (chunk with sm_margin m, rows on the m SMs left), best m of
      8..112, contig | scatter KV:
      | case | 16 x q4 | 48 x q4 | 48 x q1 |
      |---|---|---|---|
      | 1024 @ 16K | -24.9 / -24.6% (m48) | -10.2 / -11.0% (m80) | -11.0 / -8.5% (m64-80) |
      | 1024 @ 49K | -3.6 / -4.7% (m16) | +2.6 / -0.1% (m48) | +0.4 / -0.1% (m48) |
      | 2048 @ 32K | -11.0 / -13.4% (m16) | -8.2 / -7.0% (m32) | -9.2 / -7.8% (m32) |
      The kernel gain left over FA3 stock is the co-run alone: 4-25%, none where the chunk's own FA3 time (49K prefix)
      matches the rows'. The best m moves from 16 to 80 SMs with the rows' share of the work, so a fixed margin loses
      most of it -- the margin has to follow the step's composition.
    - [x] **Per-step SM partition in serving -- negative (2026-10-21; `apply_split_auto_margin.py`,
      `h200_auto_margin_20261021.sh`, rep 1, pod md5 b7b4b6c4).** m from a cost model fit on the sweep above:
      ours no MTP 125.2 vs 137.0 tok/s at margin 0, chat attainment 75.6 vs 78.4%, chat TPOT p90 62 vs 50 ms; + MTP
      130.9 vs 129.4 / 131.9 (no change). Logged steps show why: serving's mixed steps are a deep deadline chunk
      (predicted 7-8 ms per layer) beside rows worth ~0.2 ms, so m = 8 took SMs from the step-setting chunk.
      Rep 2 stopped.
    - [x] **Serving-shaped mixed steps vs one stock FA3 call (2026-10-21; `python tasks/debug_kernel_server_perf/
      serving_mix_corun_20261021.py`, raw `serving_mix_corun_20261021.txt`; chunk C @ P + A agent rows at 32-64K KV
      + S chat rows at 0.5-4K, q4 or q1; all 20 steps within 4.2e-3 of fp32).**
      | chunk | split | co-run, shared SMs (m0) | co-run, rows on 8 SMs | best partition |
      |---|---|---|---|---|
      | 512 @ 4K | -63..-76% | -64..-77% | -12..+151% | -67..-79% |
      | 1024 @ 16K | -31..+1% | -8..-38% | -10..+158% | -10..-33% |
      | 2048 @ 32K | +11..+21% | +8..+11% | +7..+26% | +3..+8% |
      | 4096 @ 32K | 0..+14% | -3..+9% | +4..+17% | +5..+17% |
      | 6000 @ 49K | -2..-4% | -2..-3% | +1..+3% | +1..+3% |
      One varlen FA3 call sized for a short chunk is slow on long-context rows (0.8 ms floor with a 512-token
      chunk vs 0.14-0.30 ms split) -- the split wins there; beside a long chunk the rows hide and the second launch
      only adds. A reserved partition never beats shared SMs by more than 3 points and loses badly when rows are
      long. So the per-step decision is whether to split, not where: `apply_split_policy.py` splits below a
      predicted chunk time of 1 ms per layer, otherwise runs the whole step as one FA3 call (and every one-sided
      step, which still ran flashinfer's prefill, now runs FA3). Serving: `h200_split_policy_20261021.sh`, running.
    - [ ] The 2-mode / 4-mode table for all three models against FA3 stock under the policy.
    - [x] Serving against stock on FA3: see `tasks/realistic_workload_analysis.md`, "Stock on its own Hopper default".
  - [ ] Single node: Qwen2.5-7B on the FA3 backend is S0b; GLM-4.5-Air on one H20 pod below; Llama-3.3-70B in the
    current goal.
    - [x] **GLM-4.5-Air output gate (2026-10-21; one H20 pod at TP4, `bash tasks/pd_colocation/gate_glm_20261021.sh` on
      pod 0 of job 2f608c1b1a67cf09, pods' flashinfer_backend.py md5 31b6d37c; own MTP layer via NEXTN, no YaRN; every
      arm on FA3): PASS.** 96 ShareGPT + 8 x 32K TraceLab prompts at once, greedy, 128 tokens. Stock rerun vs stock (noise
      floor): short 16/96 exact, mean common prefix 238 of 567 chars; long 0/8, 219 of 600. Ours vs stock: short 20/96,
      230; long 1/8, 187. 0 tracebacks in all arms; ours 1001 split calls, 151 4-mode steps, verify on FA3 captured;
      accept len 2.54 / 2.61 (stock) and 2.75 (ours). Serving table below.
    - [ ] **GLM-4.5-Air single node, one H20 pod at TP4 (2026-10-21, `bash tasks/pd_colocation/mn_glm_single_20261021.sh`;
      chat template, own MTP layer; stock on FA3, ours with the split policy). At the standard mix every system is
      overloaded** (stock + MTP chat TTFT 136 s, 772 of ~1640 chat tok/s offered), so quarter load (agents C=2 over 6
      sessions, chat 2 req/s) and half load (C=4 over 12, chat 4 req/s), one repeat each. agent tok/s | agent TTFT |
      chat TTFT | chat TPOT:
      | arm | quarter load | half load |
      |---|---|---|
      | stock prefill-prioritized + MTP, FA3 | 43.1 \| 4.3 s \| 0.81 s \| 25.1 ms | 23.8 \| 24.4 s \| 22.9 s \| 50.1 ms |
      | stock mixed chunk, FA3, no MTP | 35.2 \| 4.2 s \| 0.80 s \| 29.9 ms | 31.1 \| 6.3 s \| 2.85 s \| 88.8 ms |
      | stock prefill-prioritized, FA3, no MTP | 35.6 \| 4.8 s \| 0.89 s \| 35.0 ms | 32.1 \| 6.5 s \| 2.93 s \| 106.8 ms |
      | ours + MTP (rows + admission, policy) | 36.6 \| 6.2 s \| 0.37 s \| 25.8 ms | 25.0 \| 12.6 s \| 4.50 s \| 98.1 ms |
      | ours no MTP (policy, decode on FA3) | HUNG (below) | 30.6 \| 8.8 s \| 1.41 s \| 85.4 ms |
      Ours + MTP loses to stock + MTP at quarter load (-15% agent tok/s) and wins only chat TTFT (-54%); at half load
      stock + MTP collapses (chat TTFT 22.9 s) while ours + MTP holds 4.5 s, but ours no-MTP is the better arm there and
      stock no-MTP still leads it on throughput. On this model and GPU the 4-mode schedule does not pay. Needs repeats.
      - [ ] **A forward step hung in ours no-MTP at quarter load (`_glm_L2_ct_pol_r1`): "Scheduler watchdog timeout
        (self.watchdog_timeout=600.0, self.soft=False)" on TP1, then SIGQUIT; 0 tracebacks, no CUDA error in the log.**
        Not reproduced in 4 further runs at the same load (`mn_glm_hang_bisect_20261021.sh`: the same config twice, once
        without DECODE_FA3, once without SPLIT_POLICY -- all 4 completed, 0 tracebacks), so it is rarer than 1 in 4 and
        those bisect arms prove nothing yet. **Why the first hang left no stack: sglang shells out to `py-spy dump`
        when the watchdog fires and py-spy was not installed on the pod** ("Pyspy failed ... py-spy: not found" for all
        four TP ranks), so the one diagnostic that would have named the stuck frame produced nothing. py-spy is now in
        the pod venv, `mn_node.sh` puts `$W/venv/bin` on the worker's PATH, and `WATCHDOG` is a knob. Soak running:
        `bash tasks/pd_colocation/mn_glm_hang_soak_20261022.sh` -- 8 runs of the arm that hung at the same load with
        `WATCHDOG=180`, each log scanned for the watchdog line and any py-spy dump. **All 8 runs clean: zero watchdog
        hits, zero tracebacks, all 8 agent streams and chat streams completing** (run 8: agent 12/12, chat 600/600).
        With the 4 earlier bisect runs that is 12 consecutive clean runs of the configuration that hung, so the fault
        is rarer than 1 in 12 at this load and a soak cannot settle it.
        **Remedy rather than more soaking (2026-10-22):** the hang lives in our flashinfer-backend split path, and the
        H200 work showed our 4-mode schedule runs on sglang's OWN FA3 backend (MIXEDSPEC_STEPS 851, accept 2.05) and is
        *faster* there with MTP (140.6 vs 136.6 tok/s) -- that backend contains none of our split code. So the same two
        GLM arms are being rerun with `KERNEL=lpt ATTN_BACKEND=fa3` at both loads, 2 repeats of the arm that hung
        (`mn_glm_fa3_20261022.sh`, WATCHDOG=180 and py-spy still armed). If they are clean and competitive, the fix is
        to run this schedule on the FA3 backend -- which we want on performance grounds anyway -- and the flashinfer
        split path can be retired instead of debugged.
  - [ ] 32 GPUs, Qwen2.5-7B no spec, ours vs disagg at 2 / 4 / 5 / 6 of 8 prefill pods (`mn_scale32_20261021.sh`,
    running); then 16 / 32 GPUs with MTP at real acceptance for each model.

Arms in S1-S3 (2026-10-12..16, flashinfer backend, before S0b): stock MTP, unfused 4-mode, fused 4-mode (co-launch, 64/32/16, auto chunk);
EAGLE3 chain q4; CUDA graphs on; 4-mode arms with the measured mixed-step barrier (see gates);
each arm reports fallbacks and composition counters. Analyzer: `drivers/serving/eval_curves.py`.

- [x] **S1 x S2, rate sweep x context mixtures (2026-10-12 19:22 -> 2026-10-13 02:57, 54 runs,
  `eval_matrix_s12_20261012.sh`):** chat 4-24/s + TraceLab agent at 0.1 sessions/s; per-tenant
  SLOs chat 2 s / 50 ms TPOT, agent 10 s / 80 ms. 0 fallbacks, 0 tracebacks in all 54 runs.
  - [x] **mix32 (32-64K):** arms equal at 4-8/s (4-mode P99 ITL worse: a verify rides with a
    chunk). From 12/s fused vs stock: TTFT -47..-79%, throughput +11..+25%, TPOT -19..-28%, SLO
    attainment up to 2x (agent 0.83 vs 0.47 at 16/s); fused beats unfused 4-mode from 16/s.
  - [x] **mix64 (64-128K):** already saturated at 4/s (the agent tenant alone fills the GPU). At
    12-20/s fused vs stock: chat throughput +15..+36%, TTFT -31..-35%, TPOT -19..-30%; agent
    throughput +18..+26%, TPOT -11..-25%; fused beats unfused by 9-14% chat throughput at 16-20/s.
  - [x] **mix128 (128-256K, YaRN x8), 2026-10-13:** the largest gains -- fused vs stock at
    4-24/s: chat throughput +11..+49%, TTFT -35..-46%, TPOT -22..-41%; agent throughput
    +10..+38%, TPOT -11..-31%, TTFT -8..-30%. At chat 24/s every chat metric clears the bar
    (+49% / -46% / -41%) and the agent reaches -30% TTFT / -31% TPOT. Fused beats unfused 4-mode
    at every rate.
  - [x] **Where the speedup comes from, per context range (2026-10-13):** each stock->fused change
    split into scheduling (stock MTP -> unfused 4-mode) and kernel (unfused 4-mode -> fused), from
    `eval_s12_v2_20261012.csv`. Acceptance is not a source: tokens/verify is within 1-2% across the
    three arms everywhere (chat ~2.03-2.08, agent ~3.1-3.7; stock marginally highest).
    | range | scheduling share | kernel share |
    |---|---|---|
    | mix32 (32-64K) | carries the onset: at 12/s chat TTFT -46% / TPOT -34% all scheduling; at 20-24/s TTFT -25..-32% | appears from 16/s: TTFT -29..-45%, TPOT -9..-15%, throughput +8..+11% at 20-24/s |
    | mix64 (64-128K) | steady: TTFT -18..-24%, TPOT -14..-20%, throughput +8..+20% | grows with load: TTFT -9..-21%, TPOT -5..-17%, throughput +3..+19% (chat 24/s: +19% kernel vs +13% scheduling) |
    | mix128 (128-256K) | small and for the agent zero or negative (agent TTFT +15..-9%) | dominant: chat TTFT -27..-33%, throughput +17..+25% at 8-24/s; almost all agent TTFT/throughput gain |
    Mechanism. *Scheduling:* stock sglang keeps mixed chunk on for EAGLE3 but degrades running
    requests to 1-token decodes in a mixed step (no speculation), or runs verify in separate steps;
    the 4-mode step keeps them speculating beside the prefill chunk. That pays most where steps
    are short and many requests are decoding (mix32 onset) and saturates once the prefill chunk
    dominates step time. *Kernel:* unfused 4-mode runs the whole merged batch through one FA2
    BatchPrefill, so long-context prefill rows run on the FA2 path; the fused kernel sends them to
    FA3 (WGMMA) on 112 SMs and co-launches FA2 for the verify rows. Attention's share of the step
    grows with context (B1 at mix32: 75-80% of kernel time; fused attention 2739 ms vs unfused 3574
    ms over 60 profiled steps, -23%), so the kernel's share of the gain grows with context: second
    to scheduling at 32-64K, comparable at 64-128K, dominant at 128-256K. Below capacity
    (mix32/mix64 at 4-8/s) neither applies and the arms are equal or 4-mode is slightly worse on
    TPOT (a verify riding with a chunk). Per-range kernel attribution at 128K+ is being profiled
    (`b1_tail_profile_mix128_20261013.sh`).
  - [x] **Figures (2026-10-13):** [`eval_s12_v2_20261012.png`](debug_kernel_server_perf/resume_20261012/eval_s12_v2_20261012.png) -- per
    mixture and tenant: throughput vs median tok/s/user, SLO attainment, mean TTFT and TPOT vs chat
    rate (data [`eval_s12_v2_20261012.csv`](debug_kernel_server_perf/resume_20261012/eval_s12_v2_20261012.csv), `drivers/serving/plot_eval.py`).
    **With interactivity spread (2026-10-16):** [`eval_s12_v2_20261012_userpct.png`](debug_kernel_server_perf/userpct_20261016/eval_s12_v2_20261012_userpct.png)
    -- the tok/s/user axis now shows each point's per-request distribution: marker = median request,
    thick bar p10-p90, whisker p1-p99 (p1/p10 are the slow-user tail, i.e. the latency P99/P90 users).
    Medians reproduce the committed CSV exactly. Same for the EDF rate sweeps:
    [`edf_ratesweep_20261015_userpct.png`](debug_kernel_server_perf/userpct_20261016/edf_ratesweep_20261015_userpct.png),
    [`edf_ratesweep2_20261015_userpct.png`](debug_kernel_server_perf/userpct_20261016/edf_ratesweep2_20261015_userpct.png)
    (their agent tenant ran on the retracted session-rate axis; the chat x-axis is a request rate).
    Requests whose tokens all arrive in one burst at the end (< 1 ms decode per token) are excluded
    from tok/s/user and counted in `burst_frac`: 0 here (one request in one row), but 17-29% of stock
    PD-disagg requests on H20 (`pd_colocation_vs_disagg.md`).
    Command, CPU only on the pod's JSONs: `eval_curves.py --root /dev/shm/w/<sweep> --slo
    '{"chat":[2000,50],"codex":[10000,80]}' --csv <sweep>_userpct.csv; plot_eval.py <csv> <png>`.
    Chat rates vs the reference list (per GPU): this sweep's 4-24 req/s on one H200 for a 7B model
    sits inside semi-PD's Llama3-8B ShareGPT range (11-16 req/s per A100) and above SLOs-Serve's
    1.5-4.5 req/s per A100 (OPT-7B/13B); DistServe / Sarathi-Serve use 0.5-10 req/s.
    Fused is lowest on TTFT and TPOT at nearly every loaded point of all three mixtures. Two
    limits of this sweep, to fix in the next pass: (1) the agent-tenant Pareto panels are not
    meaningful -- the swept rate is the chat rate, the agent rate is fixed, so agent throughput
    falls as chat load rises; the agent tenant needs its own rate sweep (the full-length run does
    this); (2) a fixed 2 s chat TTFT SLO is met by no arm at 64-128K and above, so SLO attainment
    stops discriminating there -- SLOs should scale with each workload's unloaded latency
    (SLOs-Serve style, e.g. 5x / 10x).
  - [ ] **Mixed-SLO setting (2026-10-18, user; JITServe in References):** score each tenant by service goodput
    (tokens on time) next to means: chat latency-sensitive (TTFT 2 s, TBT 100 ms), agent rounds deadline-sensitive
    with a tiered TTFT SLO -- doubled above 200K uncached tokens (user); sweep a common SLO scale 0.8-1.5x. In the
    scheduler, the same tier as an EDF deadline multiplier (`SGLANG_EDF_LONG_DEADLINE=tokens:scale`), tested on a
    workload that contains > 200K prompts.
    - [x] **First test, over capacity (2026-10-18, H200 `d6a80befdd4e4da5`):** TraceLab cohort 32-250K (YaRN x8,
      context 262144), agent C=6, 18 sessions x 2 rounds (36 rounds, 4 above 200K, max 222,761 tokens) + ShareGPT
      8 req/s; all arms clean (0 tracebacks / fallbacks, >= 73001 split calls in ours). Every arm is past capacity
      (chat TTFT >= 32 s), so these are not operating-point results. vs stock MTP (agent 37.4 tok/s / 52.8 s / 47.8 ms,
      chat 1073 tok/s / 69.2 s / 34.8 ms), ours: agent +32% / -22% / -55%, chat +31% / -53% / -29%; ours with
      `LONG_DEADLINE=200000:2`: agent +33% / -21.5% / -59%, chat +32% / -52% / -29%. The doubled deadline leaves mean
      agent TTFT unchanged (41.5 vs 41.3 s) and moves it between lengths, as intended: < 64K 20.6 -> 13.0 s, 64-128K
      34.4 -> 25.7 s, 128-195K 51.0 -> 51.5 s, > 195K 61.6 -> 92.8 s; median 41.2 -> 28.5 s, p99 100 -> 142 s. Next:
      the same cohort at a chat rate near capacity, scored by per-tier SLO attainment. `YARN_FACTOR=8.0
      CTX_LEN=262144 MIN_CTX=32768 MAX_CTX=250000 AGENT_CONC=6 SESSIONS=18 SPEC_TAG=_long CHUNKS=2048
      DEADLINE_CHUNK=1024:2048:6000 bash tasks/pd_colocation/mn_h200_fourmode_20261018.sh`, then the same with
      `SKIP_BASE=1 LONG_DEADLINE=200000:2`; sglang `be923db1f` + this commit.
    - [x] **Chat 4 req/s, same cohort (2026-10-18; 1200 chat requests; all arms clean, >= 52001 split calls):**
      stock MTP is still past capacity (chat 590 tok/s served, TTFT 48.7 s); single repeat.
      | metric vs stock MTP (agent 42.4 tok/s / 40.0 s / 50.3 ms; chat 590 tok/s / 48.7 s / 61.4 ms) | ours | ours + 2x above 200K |
      |---|---|---|
      | agent throughput / TTFT / ITL | +41% / -27.5% / -40% | +41% / **-39.1%** / -33% |
      | chat throughput / TTFT / ITL | +35% / -63% / -32.4% | +35% / **-80.4%** / -26.7% |
      | agent TTFT SLO attainment (10 s; 20 s for >= 200K) | 25.0% (stock 11.1%) | 30.6% |
      | chat TTFT SLO attainment (2 s) | 20.6% (stock 0.2%) | 50.7% |
      Agent TTFT by prompt length, stock / ours / ours + 2x: < 64K 24.4 / 10.6 / 8.3 s, 64-128K 44.7 / 20.7 / 14.4 s,
      128-195K 45.2 / 37.4 / 31.1 s, > 195K 46.3 / 51.6 / 51.2 s. With the doubled deadline five of six metrics clear
      the bar on this long-context mix; chat ITL (-26.7%) misses -- the long prefills it defers still run as mixed steps
      chat decode rides through. Without it chat ITL clears (-32.4%) and agent TTFT misses (-27.5%). Next: repeat, a
      lower chat rate where stock MTP is below capacity, and the output gate for the `LONG_DEADLINE` arm (ordering
      only; the config's gate passed). `CHAT_RATE=4 CHAT_N=1200 SPEC_TAG=_long_r4` + the command above.
    - [x] **Chat 2 req/s, same cohort (2026-10-18; 600 chat requests; all arms clean):** both our arms clear all six
      bar metrics vs stock MTP (agent 46.0 tok/s / 28.8 s / 68.8 ms, chat 329 tok/s / 36.6 s / 103 ms; stock still
      serves below the offered ~407 chat tok/s, ours serves all of it): ours agent +53% / -37.5% / -44%, chat +23% /
      -86% / -45.5%; ours + 2x above 200K agent +55% / -37.8% / -46%, chat +23% / -89% / -48%. SLO attainment (agent 10
      s, 20 s >= 200K; chat 2 s): stock 22.2% / 0.2%, ours 52.8% / 54.3%, ours + 2x 55.6% / 70.8%. Agent TTFT by length,
      stock / ours / ours + 2x: < 64K 19.2 / 6.0 / 2.2 s, 64-128K 33.0 / 13.9 / 9.9 s, 128-195K 29.9 / 22.8 / 23.1 s,
      > 195K 38.5 / 33.1 / 46.6 s. Single repeat. These are vs stock MTP; the bar's single-node baselines (stock mixed
      chunk without MTP, stock + MTP for the MTP arm, pdmux) are in `realistic_workload_analysis.md` ?lines.
      `CHAT_RATE=2 CHAT_N=600 SPEC_TAG=_long_r2` + the command above.
  - [ ] **Relative SLOs (2026-10-13):** `eval_curves.py --slo-relative K_TTFT,K_TPOT --ref-root`
    scales each tenant's SLO from its median TTFT/TPOT running ALONE on stock MTP. The swept runs'
    lowest rate cannot be the reference (mix128 chat already waits 38 s at 4/s; tried and
    discarded). Unloaded runs, chat 0.5/s and agent 0.02 sessions/s per mixture,
    `unloaded_ref_20261013.sh` (driver gets a `STREAMS` filter, `streams_filter.patch`) -- queued.
- [ ] **S2b, full-length TraceLab (user, 2026-10-13):** the release as it is -- Claude Code context
  p50/p90/p99 133K/483K/923K (25% >= 256K), Codex 120K/204K/233K, outputs p50 ~200 tokens; the
  32-64K slice is ~16% of rounds. Loader replays both providers, first rounds, round-weighted
  sessions; `tracelab_full_env` at YaRN x32 (1M tokens), agent tenant alone, rate = session starts/s.
  - [x] **v1 aborted (2026-10-13 03:16):** 120 sessions x 2 rounds at 0.2-0.8/s queued 16.8M prompt
    tokens at ~2K tok/s -- one 1M-token session alone is ~400 s of prefill; could not finish.
  - [x] **v2 aborted (2026-10-13 04:54):** 24 sessions x 10 rounds, seed 187 (loader-checked p10/50/90/99
    40K/127K/235K/869K). The stock arm hit its 90-min timeout having prefilled **14.4M** tokens for a
    replay with 5.8M uncached tokens: the KV pool (1,823,088 tokens) cannot hold 24 sessions reaching
    869K, so prefixes were evicted between rounds (usage 0.99, no retractions) and rounds re-prefilled.
    No data. Lesson: a single-GPU replay must keep the sessions' combined peak context in the pool.
  - [x] **v3 -- superseded (2026-10-18 note):** its replay ran as P1 of the 2026-10-15 concurrency sweep below (same
    8 x 30 rounds, seed 1014, C = 2 / 4 / 8, all arms clean); the session-rate form here was retracted. Original plan:
    8 sessions x 30 rounds, seed 1014 --
    combined peak 1.37M tokens; loader-checked context p10/50/90/99 38K/123K/223K/390K vs the
    release's 38K/124K/256K/822K. The >400K tail cannot be co-resident with other sessions on one GPU
    (excluded by KV capacity, not by a length filter) and belongs to the multi-GPU stage. YaRN x16
    (524288), session rates 0.005 and 0.02/s, three arms, 4-mode arms on the verdict's tree
    (`tracelab_full_v2b_20261013.sh`, root `tracelab_full_v3_20261013`).
  - [x] **Session-rate axis retracted (2026-10-15):** `tracelab_full_final_20261015.sh` swept
    "0.005 / 0.02 sessions/s" (Poisson session starts, rounds back to back), an axis no reference
    uses (Crossflow: req/s; ThunderAgent: concurrency). It controls neither: at 0.005/s the 8 sessions averaged 1.3-1.5 requests in flight (MTP / split), a
    near-idle server. Cancelled by the user after 2 of 6 arms; those numbers are not a result.
    Agent load is program concurrency only (ThunderAgent; references).
  - [x] **Concurrency sweep (2026-10-15 12:36-15:56, all 18 arms clean, peak = C every arm):** same 8x30-round seed-1014 replay, program concurrency
    C = 2/4/8 (KV pool caps C at p99 390K context), arms MTP | split static 3072 | split + EDF +
    floor 2048; then the frozen 32-64K cohort at C = 8/16/24 + ShareGPT 8 req/s (replaces the
    retracted EDF load points). Running since 12:36: `concurrency_sweep_20261015.sh`, root
    `concurrency_sweep_20261015`; each arm logs `PROGRAM_CONCURRENCY` (peak must equal C).
    Partial (TraceLab, peak = C and mean in-flight 1.99-2.00 / 3.76-3.79 in every arm, 0 tracebacks /
    fallbacks): **C=2** split / EDF vs MTP: tput +7.7% / +13.9%, mean TTFT +4% / -12%, TPOT -23% /
    -22%, mean ITL -19% / -13%, P99 ITL 28 -> 85 / 88 ms. **C=4**: tput +29% / +26%, mean TTFT
    **+15% / +14%** (median -22% / -17%, P99 -25% / -27%), TPOT -29% / -31%, mean ITL -34% / -31%,
    P99 ITL -67% / -73%. TTFT by uncached tokens at C=4: cache-hit rounds (<1K uncached) keep a
    170-200 ms median in all arms but their mean rises 1.1 s (MTP) -> 1.7 s (split) / 3.1 s (EDF):
    head-of-line blocking -- an in-progress chunked prefill takes the whole 2-3K budget every step
    and EDF never ranks waiting requests against it. Fix under test: `edf_reserve_earlier.patch`
    (`SGLANG_EDF_RESERVE_EARLIER`); its A/B `edf_reserve_20261015.sh` was cancelled at 16:3x as a
    duplicate of the parallel session's `edf_reserve_ab_20261015.sh` (`edf_reserve_earlier_deadlines.patch`).
    That variant ranks waiting requests by their enqueue deadline from TOTAL prompt length and fired
    once in the whole C=4 run (`EDF_RESERVE calls=1`); a cache-hit round of a long session is never
    earlier than the chunked prompt under that deadline. `edf_reserve_earlier.patch` rescores on
    uncached work instead.
    **Both variants were inert (2026-10-15 18:1x):** mine also fired ~once per run (TraceLab C=4,
    17:50-18:05). Cause, found in the server log: the EDF arms run the FCFS/priority policy, which
    prefix-matches waiting requests only when the radix cache `supports_fast_match_prefix()` (it does
    not here), so `num_matched_prefix_tokens` stays 0 and a 99%-cached 27K round is charged 27K
    tokens > the 2K budget -> the reservation loop breaks. Log: 88 s stretch with `#queue-req: 1`,
    every step `#new-seq: 1` (the cold chunked prompt), token usage <= 0.41 (not memory). The TTFT
    tail is a handful of cache-hit rounds (<600 uncached) waiting 60-96 s -- also in MTP (one at
    96 s). Single C=4 runs are too noisy to judge (two identical EDF controls: mean TTFT 3.9 vs
    4.6 s). Fix: match the first 16 waiting requests inside the reservation
    (`edf_reserve_matched.patch`, tree `up_edf_reserve2`); rerun with C=4 twice + same-tree control,
    C=8, cohort C=16 (`edf_reserve_matched_20261015.sh`), queued behind the A3 A/B and S3.
    **Still inert (23:13-23:20, TraceLab C=4):** <20 firings in 6 min with matching, so unmatched
    prefixes are not the whole cause; run stopped. One diagnostic arm queued
    (`edf_reserve_diag_20261015.sh`, tree `up_edf_reserve3`): logs the head waiting request's
    length, matched prefix, arrival and deadline against the chunked request's.
    **Root cause found (2026-10-16 01:3x, server-log trace of the 42-61 s outliers):** the outlier IS
    in the waiting queue (`#queue-req: 1` for 61 s) behind a cold ~390K prompt prefilled 2,048 tokens
    per step; it is a cache-hit round with 2,377 uncached tokens -- more than the whole step budget,
    and sglang allows one chunked request, so no budget reservation can admit it (the diagnostic arm:
    20 reservations in 15 min; MTP shows the same pattern, one 96 s outlier). Fix: EDF over the chunked
    request -- rescore waiting deadlines on uncached work and park the chunked prompt (prefix kept in
    the radix tree, row released as a retraction, re-queued next step) when a short earlier-deadline
    request waits (`edf_preempt_chunk.patch`, `SGLANG_EDF_PREEMPT_CHUNK`, tree `up_edf_preempt`).
    A/B with the EDF serving gate, interleaved in one queue: `edf_preempt_20261016.sh`.
    Gates (01:32-01:35): EDF control and EDF+preempt both PASS (long prefix 52) **but parked 0
    times** -- the standard gate sends short prompts first, so it cannot exercise parking. Forced
    gate queued: long prompts first, short ones while they are chunked (chunk 256), stock MTP in the
    same order as the order-only null case, park arm must park >= 1
    (`edf_park_gate_20261016.sh`, `drivers/serving/srv_correctness_edf_park_20261016.py`).
    **Serving, TraceLab (2026-10-16 01:35-03:40, all arms clean, peak = C, one tree/queue):**
    | C | arm | tput | mean TTFT | <1K-uncached TTFT mean / med | mean ITL | TPOT |
    |---|---|---|---|---|---|---|
    | 4 | MTP | 93.8 | 4178 ms | 1040 / 201 | 29.9 | 26.7 |
    | 4 | EDF control (x2) | 116.6 / 114.0 | 3941 / 4823 | 3107 / 2254 | 22.1 / 20.6 | 20.7 / 21.3 |
    | 4 | EDF + preempt (x2) | 119.4 / 119.1 | **2558 / 2518** | **326 / 284** | 24.0 / 24.1 | 23.3 / 24.4 |
    | 8 | MTP | 94.7 | 6068 | 2003 / 311 | 59.7 | 77.6 |
    | 8 | EDF control | 124.1 | 7771 | 6166 / 294 | 37.0 | 37.3 |
    | 8 | EDF + preempt | 125.2 | 5136 | 1762 / 405 | 42.2 | 49.6 |
    Parks: ~20-40 per run (urgent 197-546 uncached tokens behind 89-110K-token remainders). **C=4:
    preempt vs MTP TTFT -39/-40%, tput +27%, ITL -20% (control -28..-31%); repeat TTFT within 2%
    (control 22% apart).** C=8: TTFT only -15% vs MTP (-34% vs control), ITL -29%, tput +32%,
    TPOT -36%; mean TTFT is now set by the 17 cold 64K+ prompts (39 s vs MTP 45 s) and the 16-64K
    bin got worse (12.0 vs 9.7 s: over the 16K cap they cannot preempt a 390K prompt). Preemption
    raises ITL 2-5 ms over the control (hit rounds decode instead of queueing). Next: uncapped
    preemption = shortest-remaining-work-first (`edf_preempt_srpt_20261016.sh`), queued after the
    park gate. Correctness of parking pending the forced gate.
    **Bug found and fixed (04:05):** the cohort C=16 preempt arm hung after its first park (no server
    log for 18 min): parked requests were re-queued inside `_edf_rescore_and_park`, which runs after
    `get_new_batch_prefill`'s early returns, so a parked request that was the only remaining work was
    stranded. Re-queue moved to the top of `get_new_batch_prefill` (`edf_preempt_chunk_v2.patch`, tree
    `up_edf_preempt` patched in place before the park gate and the SRPT runs). The TraceLab preempt
    numbers above ran v1 (constant arrivals there always re-triggered the re-queue; every request
    completed), so they stand provisionally; the SRPT run re-measures the 16K-cap arm on v2. The
    cohort C=16 preempt point is void (rc 143, killed).
    **Forced park gate v1 (04:39-04:43, `edf_park_gate_20261016.sh`):** order-only null cases PASS (stock
    MTP long-first: long prefix 11; EDF control long-first: 96), but the park arm parked 0 times and
    failed its must-park check (rc 1; outputs still PASS, long 96): EDF already serves the short
    requests ahead of the WAITING long prompts, and the in-progress long had < 2x the short work left.
    v2 queued: one long, then the shorts 0.1 s later, then the other longs.
    **Preempt on the fixed code (v2), TraceLab C=8, first pair (04:48-05:16):** 16K cap: TTFT 4113 ms
    (**-32% vs MTP 6068**; v1 gave 5136 -- the stranded-request bug cost ~1 s), ITL 44.6 (-25%), tput
    125.8 (+33%); uncapped (SRPT): TTFT 3963 (**-35%**), ITL 44.8 (-25%), tput 127.3 (+34%), 16-64K bin
    6.1 -> 4.1 s. TTFT now clears at C=8; ITL does not (needs <= 41.8 ms). Mechanism: EDF runs ~1900
    mixed steps vs MTP's 670 (2048 vs 8192 prefill tokens per step at 100-390K context), and the EDF
    budget asks for less than the 2048 floor (`requested=8192 admitted=2050`); preemption adds decode
    concurrency (5.35-5.46 vs 4.98 running per decode step). One GPU sits on a TTFT/ITL frontier here
    (floor 1024: ITL -8%, TTFT +19%); the remaining lever is placement across nodes (PD point 1).
    **Uncapped vs 16K-cap preemption, complete (`edf_preempt_srpt_20261016.sh`, 04:48-06:32, all arms
    clean, peak = C, parks 20-60 per run; same-queue MTP at C=8: TTFT 6538 ms, ITL 58.8, tput 93.5):**
    | arm | TTFT | ITL | TPOT | tput | vs MTP (TTFT / ITL / tput) |
    |---|---|---|---|---|---|
    | C=8 16K cap (x2) | 4113 / 5016 | 44.6 / 42.9 | 54.4 / 48.8 | 125.8 / 125.4 | -37/-23% / -24/-27% / +35/+34% |
    | C=8 uncapped (x2) | **3963 / 3751** | 44.8 / 45.5 | 50.4 / 50.3 | 127.3 / 125.9 | **-39/-43%** / -24/-23% / +36/+35% |
    | C=4 uncapped (x2) | **2369 / 2378** | 24.9 / 25.0 | 32.1 / 32.1 | 120.7 / 119.8 | **-43%** / -17% / +28% (MTP from the 01:35 queue) |
    Uncapped preemption (shortest remaining work first among prompts) is adopted: better and steadier
    TTFT than the 16K cap. **Bar status, full-length TraceLab: TTFT and throughput clear at C=4 and
    C=8; ITL misses (-17% / -23..-24%), and C=4 TPOT is worse than MTP (+20%)** -- more requests decode
    beside 2048-token chunks at 100-390K context.
    **Forced park gate v2 PASS (06:32-06:36, `edf_park_gate_v2_20261016.sh` = `edf_park_gate_20261016.sh`
    with the v2 order):** the park arm parked (logged: chunked remainder 10,464, urgent 2,048 tokens,
    slack 1.96 s) and agrees with the stock MTP_LPT_Q4G reference on long prefix 11 -- the same as stock
    MTP in this order (11, the order-only noise floor; the EDF control without parking: 96); 0
    tracebacks / fallbacks. Parked prefills resume correctly; the preemption serving arms are gated.
    **Final single-GPU config vs MTP, mixed workload (2026-10-16 06:38-07:27, `mixed_final_20261016.sh`;
    split + EDF 50 ms + floor 2048 + uncapped preemption; frozen 32-64K cohort at program concurrency C
    + natural ShareGPT chat; all 8 arms clean, peak = C, parks 20-120; acceptance within 0.02):**
    | point | tenant | tput | mean TTFT | TPOT | mean ITL | P99 ITL |
    |---|---|---|---|---|---|---|
    | C=8, chat 4/s | agent | **+44%** | **-35%** | -38% | **-31%** | -85% |
    | C=8, chat 4/s | chat | **+11%** | **-83%** | -42% | **-39%** | -74% |
    | C=16, chat 4/s | agent | +40% | -23% | -58% | -46% | -86% |
    | C=16, chat 4/s | chat | +12% | -62% | -42% | -39% | -75% |
    | C=8, chat 8/s | agent | +22% | -19% | -43% | -28% | -83% |
    | C=8, chat 8/s | chat | +20% | -57% | -13% | -16% | -73% |
    | C=16, chat 8/s | agent | +27% | -27% | -52% | -34% | -81% |
    | C=16, chat 8/s | chat | +25% | -44% | -20% | -19% | -72% |
    **C=8 + chat 4/s clears the bar on every metric for both tenants** (chat throughput is near the
    offered load). Misses elsewhere: agent mean TTFT at C=16 / chat 8 (-19..-27%; MTP agent TTFT is
    18-44 s, i.e. overload, where the agent's own prefill backlog dominates) and chat mean ITL / TPOT
    under chat 8/s overload (-13..-20%). P99 ITL drops 72-86% everywhere. Data
    [`mixed_final_20261016.txt`](debug_kernel_server_perf/resume_20261012/mixed_final_20261016.txt),
    table `drivers/serving/mixtab.py mixed_final_20261016` (pod `/dev/shm/w`).
    **Complete table, % vs MTP** (tput / mean TTFT / median TTFT / TPOT / mean ITL / P99 ITL):
    | point | split static 3072 | split + EDF + floor |
    |---|---|---|
    | TraceLab C=2 | +8 / +4 / -19 / -23 / -19 / +209 | +14 / -12 / -20 / -22 / -13 / +217 |
    | TraceLab C=4 | +29 / +15 / -22 / -29 / -34 / -67 | +26 / +14 / -17 / -31 / -31 / -73 |
    | TraceLab C=8 | +32 / +7 / -37 / -49 / -33 / -80 | +34 / +28 / -26 / -50 / -38 / -82 |
    | cohort C=8 agent | +24 / -23 / -24 / -37 / -23 / -81 | +25 / -22 / -25 / -42 / -27 / -82 |
    | cohort C=8 chat | +22 / -38 / -35 / -18 / -19 / -66 | +23 / **-75** / -74 / -19 / -17 / -73 |
    | cohort C=16 agent | +30 / -28 / -26 / -53 / -36 / -84 | +30 / -27 / -27 / -55 / -39 / -86 |
    | cohort C=16 chat | +28 / -37 / -35 / -25 / -25 / -66 | +29 / -55 / -53 / -25 / -24 / -73 |
    | cohort C=24 agent | +32 / -25 / -13 / -72 / -52 / -83 | +30 / -21 / -5 / -77 / -60 / -88 |
    | cohort C=24 chat | +29 / -39 / -42 / -25 / -27 / -64 | +29 / -48 / -50 / -22 / -20 / -69 |
    Mean in-flight agent requests: TraceLab 1.99-2.00 / 3.76-3.79 / 7.07-7.13; cohort 7.3-7.7 /
    13.6-14.9 / 19.2-21.0. Acceptance 2.12-2.16 (cohort), 2.89-3.02 (TraceLab) in every arm.
    **Verdict:** throughput (+22..+34%), TPOT and P99 ITL clear the bar at every loaded point; agent
    TTFT misses (-21..-28% on the cohort, worse than MTP on TraceLab from head-of-line blocking);
    chat mean ITL misses (-17..-27%). **Load caveat:** chat at 8 req/s alone exceeds capacity at
    every C (MTP chat TTFT 24-74 s), so every cohort point is overloaded; a below-capacity chat
    rate is still missing. Data [`concurrency_sweep_20261015.csv`](debug_kernel_server_perf/resume_20261012/concurrency_sweep_20261015.csv),
    `bash concurrency_sweep_20261015.sh` (pod `/dev/shm/w`), table `drivers/serving/conc_table.py
    concurrency_sweep_20261015`; sglang `118c5985b` + pod trees `up_edf_floor` (+ ported replay).

- [ ] **S3, standalone workloads:** ShareGPT (4-32/s), GSM8K, MATH-500, HumanEval, Code-Feedback
  (8-32/s), Mooncake conversation / tool-agent (recorded timestamps at 0.5-2x speed), LongBench v2
  (0.25-1/s, documents that do not fit 131072 dropped); three arms each, YaRN x4.
  `s3_standalone_20261013.sh` (`s3_standalone_knobs.patch`: Mooncake slowdown and LongBench length
  filter were not plumbed) -- queued last.
    - [x] **C = 2 / 4 done (14:13):** all 6 arms clean, peak = C, mean in-flight 1.99-2.00 / 3.76-3.79.
      | C | arm | out tok/s | TTFT mean / p50 / p99 | TPOT | ITL mean / p99 |
      |---|---|---|---|---|---|
      | 2 | MTP | 100.3 | 3846 / 341 / 96.5K | 10.9 | 10.0 / 27.6 |
      | 2 | split 3072 | 108.0 (+8%) | 4007 / 277 / 102.8K | 8.4 | 8.1 / 85.5 |
      | 2 | EDF+floor | 114.3 (+14%) | 3383 (-12%) / 273 / 94.3K | 8.5 (-22%) | 8.7 (-13%) / 87.6 |
      | 4 | MTP | 93.2 | 3466 / 422 / 96.1K | 30.1 | 31.8 / 590 |
      | 4 | split 3072 | 120.5 (+29%) | 3995 (+15%) / 330 / 72.2K | 21.4 (-29%) | 21.1 (-34%) / 194 (-67%) |
      | 4 | EDF+floor | 117.9 (+26%) | 3945 (+14%) / 349 / 70.0K | 20.8 (-31%) | 21.8 (-31%) / 160 (-73%) |
      At C = 4 the split clears throughput, ITL and P99 ITL; mean TTFT misses. **TTFT by uncached
      prompt length (C = 4):** short appends < 4K (185 of 240 rounds) MTP 926 ms -> split 1994 /
      EDF 2344; 4-32K 2435 -> 3618 / 2407; cold >= 32K (19 rounds) 30.1 s -> 24.2 / 22.4 s (-20 /
      -26%). The kernel helps the long prefills; the loss is head-of-line blocking: sglang continues
      the in-progress chunked prefill with the whole step budget, so a short append waits for every
      remaining chunk of a long prompt, and 2-3K chunks hold that slot longer than MTP's 8192.
      EDF's floor binds at every logged step (admitted 2048 of 8192); the step model's prediction is
      physical (8192 tokens at a 100K prefix ~ 670 ms of attention vs ~145 ms of budget at 50 ms x
      ~2.9 tokens/step), so this is the SLO, not a model error.
      (`ttft_breakdown.py` on the pod; per-arm `*_agent.json` under `concurrency_sweep_20261015/`.)
    - [x] **C = 8 done (15:00):** all arms clean, peak 8, mean in-flight 7.07-7.13.
      | arm | out tok/s | TTFT mean / p50 / p99 | TPOT | ITL mean / p99 | TTFT <4K / 4-32K / >=32K uncached |
      |---|---|---|---|---|---|
      | MTP | 93.7 | 6133 / 651 / 100.9K | 74.0 | 59.8 / 1335 | 2179 / 7166 / 42678 |
      | split 3072 | 124.1 (+32%) | 6580 (+7%) / 411 / 82.2K | 37.7 (-49%) | 40.1 (-33%) / 271 (-80%) | 4312 / 4536 / 32534 |
      | EDF+floor | 125.2 (+34%) | 7846 (+28%) / 483 / 72.1K | 36.6 (-50%) | 36.8 (-38%) / 236 (-82%) | 6157 / 3681 / 32180 |
      Throughput, TPOT, ITL and P99 ITL clear the bar; mean TTFT does not, for the same reason as
      C = 4 and more so: short appends (185/240 rounds) wait 2-2.8x longer behind long prompts'
      chunks, while 4-32K and cold prefills improve 37-49% and 24-25%. EDF's 2048 floor makes the
      blocking worse than static 3072. The reservation A/B below is the direct test.
    - [x] **Cohort C = 8 + ShareGPT 8 req/s (15:18):** all arms clean, peak 8 programs, mean
      7.29-7.67; acceptance 2.12-2.15 in every arm. Out tok/s / TTFT mean / TPOT / ITL mean / P99 ITL:
      | arm | agent | chat |
      |---|---|---|
      | MTP | 110.3 / 18992 / 29.6 / 20.7 / 413 | 1288 / 23560 / 30.0 / 30.4 / 476 |
      | split 3072 | +24% / -23% / -37% / -23% / -81% | +22% / -38% / -18% / -19% / -66% |
      | EDF+floor | +25% / -22% / -42% / -27% / -82% | +23% / **-75%** / -19% / -17% / -73% |
      Chat is still queueing (MTP chat TTFT 23.6 s at 8 req/s), so this point is past chat capacity.
      Misses: agent TTFT (-22%), agent ITL (-23..-27%), chat mean ITL / TPOT (-17..-19%).
    - [x] **Cohort C = 16 + ShareGPT 8 req/s (15:37):** all arms clean, peak 16, mean 13.6-14.9.
      | arm | agent | chat |
      |---|---|---|
      | MTP | 106.9 / 43548 / 44.9 / 27.4 / 501 | 1207 / 52023 / 31.5 / 31.4 / 471 |
      | split 3072 | +30% / -28% / -53% / -36% / -84% | +28% / -37% / -25% / -25% / -66% |
      | EDF+floor | +30% / -27% / -55% / -39% / -86% | +29% / -55% / -25% / -24% / -73% |
      Clears throughput, agent TPOT/ITL/P99 ITL, chat TTFT and P99 ITL; misses agent TTFT (-27%)
      and chat mean ITL (-24%). Both arms deeply overloaded (MTP TTFT 44-52 s).
    - [x] **Cohort C = 24 + ShareGPT 8 req/s (15:56):** all arms clean, peak 24, mean 19.2-21.0.
      | arm | agent | chat |
      |---|---|---|
      | MTP | 108.1 / 58964 / 87.6 / 42.6 / 597 | 1208 / 73577 / 29.3 / 28.8 / 406 |
      | split 3072 | +32% / -25% / -72% / -52% / -83% | +29% / -39% / -25% / -27% / -64% |
      | EDF+floor | +30% / -21% / -77% / -60% / -88% | +29% / -48% / -22% / -20% / -69% |
      **Sweep summary (all 18 arms clean, every peak = C):** throughput +22..+34% and P99 ITL
      -64..-88% everywhere above C = 2; agent TPOT/ITL clear the bar from C = 4 (TraceLab) and C = 16
      (cohort). Two metrics miss at every loaded point: **agent mean TTFT** (cohort -21..-28%;
      TraceLab +7..+28%, short appends blocked behind long prompts' chunks) and **chat mean ITL**
      (-17..-27%). The reservation and floor A/Bs below target exactly these.
    - [x] **EDF earlier-deadline reservation (2026-10-15, done 17:06):** under EDF, cap the chunked
      request's share so waiting requests with an earlier TTFT deadline are admitted in the same
      step (`edf_reserve_earlier_deadlines.patch`, tree `up_edf_resv`, `SGLANG_EDF_RESERVE=1`).
      A/B vs a same-queue EDF control at TraceLab C = 4 / 8 and cohort C = 16 + chat 8 req/s:
      `edf_reserve_ab_20261015.sh`; S3 now waits for it.
      **TraceLab C = 4 (16:26):** both arms clean. Reservation vs same-queue control: mean TTFT
      4633 -> 2882 ms (-38%); short appends <4K 2670 -> 917 ms (-66%, back to MTP's 926); cold
      >=32K 28.8 -> 23.5 s; 4-32K 1991 -> 2074; throughput 115.2 -> 118.2; mean ITL 20.8 -> 23.5 ms
      (+13%, short requests join the decode batch sooner). The control repeated 17% above the
      sweep's EDF arm in mean TTFT, so cross-queue comparisons are not trusted:
      `confirm_resv_20261015.sh` reruns MTP / split 3072 / EDF + reservation interleaved twice in one
      queue (TraceLab C = 4 / 8, cohort C = 16 + chat 8 req/s) before any verdict; S3 behind it.
      **TraceLab C = 8 (16:55):** mean TTFT 7961 -> 5759 ms (-28%), short appends 6561 -> 3673 (-44%),
      ITL 36.6 -> 41.3 ms (+13%), throughput +2%; still above the sweep's MTP for short appends
      (2179). The reservation fired < 200 times: at C = 8 the long prompts are already past their own
      deadline, so EDF correctly ranks them first and nothing earlier waits. **Short-request reservation (mode 2)**
      (`edf_reserve_short_mode2.patch`, tree `up_edf_resv2`, `SGLANG_EDF_RESERVE=2`) also reserves
      for later-deadline requests whose whole prefill fits in half the step, and beside a chunked
      request admits only the reserved ones (an unreserved request ahead would end admission).
      Not smoke-tested separately; added as a fourth arm of the confirmation run.
      **Cohort C = 16 + chat 8 req/s (17:06):** no effect, within noise -- agent tput / TTFT / ITL
      138.6 / 32.0 s / 16.7 -> 139.9 / 31.6 s / 16.8, chat 1549 / 23.8 s / 24.2 -> 1559 / 23.1 s /
      23.8. Fired < 200 times; here TTFT is queueing past capacity (32 s), which reordering within a
      step cannot remove. Deadline reservation helps only where long prompts block short ones (TraceLab).
    - [x] **Reservation + lower floor (done 17:50):** chat mean ITL misses (-24% at cohort C=16)
      because the 2048 floor binds every step; floor 1024 and 512 with the reservation on, TraceLab
      C = 8 and cohort C = 16 (`edf_floor_resv_20261015.sh`). S3 moved behind it.
      **Done 17:50, all 4 arms clean.** TraceLab C = 8, floor 2048 / 1024 / 512 (all with
      reservation): mean TTFT 5759 / 9505 / 13220 ms, short appends 3673 / 6810 / 10486, mean ITL
      41.3 / 33.6 / 30.3, P99 ITL 282 / 178 / 136, tput 126.3 / 122.3 / --. Cohort C = 16 vs the
      sweep's MTP (cross-queue), agent and chat tput / TTFT / TPOT / ITL / P99 ITL:
      | floor | agent | chat |
      |---|---|---|
      | 2048 | +31% / -28% / -55% / -39% / -85% | +29% / -56% / -25% / -24% / -73% |
      | 1024 | +23% / -17% / -64% / -49% / -88% | +23% / -44% / **-30% / -30%** / -83% |
      | 512 | +20% / -11% / -68% / -54% / -90% | +20% / -39% / -33% / -33% / -84% |
      The floor is a straight TTFT-vs-ITL dial: lowering it buys chat mean ITL (the -30% bar is
      crossed at 1024) and pays in agent TTFT and throughput. No single floor clears both chat ITL
      and agent TTFT at this overloaded point. Floor 2048 stays the default; the confirmation run
      keeps it.
    - [x] **Same-queue confirmation (done 23:13, `confirm_resv_20261015.sh`):** MTP | split
      3072 | EDF + deadline reservation | EDF + short-request reservation, interleaved x2, TraceLab C = 4 / 8 and cohort
      C = 16. **Rep 1, TraceLab C = 4 (same queue, vs MTP):** split tput +23% / mean TTFT +28% / ITL
      -33%; EDF+reservation +28% / **-2%** / -28% (P99 ITL -70%); EDF+short-request reservation +22% / +2% / -25% (short
      appends 761 ms vs MTP 945, but 4-32K requests 4597 vs 2427 and cold 28.9 s). **Why mean TTFT
      stays near MTP:** the 19 cold >= 32K prompts (159K uncached on average, 190 PFLOP) are 45-69% of
      all TTFT; they reach 331 (MTP) / 396 (split) / 446 (EDF+reservation) TFLOP/s including queueing,
      vs a ~700 TFLOP/s FA3 floor of 14.3 s per prompt (`cold_bound.py`). -30% mean TTFT needs cold
      prefill near ~16 s, i.e. larger chunks, which the 50 ms ITL budget forbids (floor 2048 binds).
      **Rep 1 complete (20:42), all 12 arms clean**, vs same-queue MTP (tput / mean TTFT / mean ITL /
      P99 ITL):
      | point | split 3072 | EDF + deadline reservation | EDF + short-request reservation |
      |---|---|---|---|
      | TraceLab C=4 | +23% / +28% / -33% / -65% | +28% / -2% / -28% / -70% | +22% / +2% / -25% / -61% |
      | TraceLab C=8 | +32% / +15% / -33% / -78% | +33% / +4% / -33% / -79% | +35% / -8% / -31% / -79% |
      | cohort C=16 agent | +30% / -28% / -37% / -84% | +31% / -27% / -42% / -88% | +12% / -2% / -45% / -86% |
      | cohort C=16 chat | +29% / -37% / -25% / -66% | +30% / -56% / -24% / -73% | +17% / **-93%** / -9% / -76% |
      Short-request reservation reserved 4,200+ times on the cohort and serves chat almost at once (TTFT 51.8 -> 3.8 s)
      but starves agent prefill; at TraceLab C=8 it gives the best mean TTFT (-8%) yet short appends
      still wait 2780 vs MTP 1903 ms (its half-step cap excludes 1-4K tool-output appends). Deadline reservation is
      the balanced arm.
      **Final, both reps (23:13), all 24 arms clean, every peak = C.** Rep-to-rep spread is 0-2% on
      throughput / ITL and 0-13% on mean TTFT (`confirm_aggregate.py confirm_resv_20261015`). Mean of
      2 reps vs same-queue MTP (tput / mean TTFT / mean ITL / P99 ITL):
      | point | split 3072 | **EDF + deadline reservation** | EDF + short-request reservation |
      |---|---|---|---|
      | TraceLab C=4 | +21 / +37 / -33 / -67% | +26 / -4 / -27 / -68% | +24 / -10 / -24 / -62% |
      | TraceLab C=8 | +33 / +9 / -33 / -79% | +34 / -5 / -31 / -78% | +35 / -17 / -29 / -78% |
      | cohort C=16 agent | +30 / -28 / -37 / -84% | +32 / -28 / -43 / -88% | +13 / -3 / -46 / -86% |
      | cohort C=16 chat | +28 / -37 / -25 / -66% | +30 / -57 / -24 / -73% | +17 / -93 / -9 / -75% |
      **Verdict vs the bar (>=30% TTFT and ITL, >=10% throughput):** deadline reservation clears throughput
      (+26..+34%) and P99 ITL (-68..-88%) at every point, mean ITL at TraceLab C=8 and the cohort agent
      tenant, and chat TTFT; it misses mean TTFT on TraceLab (-4 / -5%; cold >=32K prefills are half
      of all TTFT and compute-bound, see above), cohort agent TTFT (-28%), and mean ITL for TraceLab C=4
      (-27%) and chat (-24%). **The bar is not met on all metrics.** Short-request reservation is not promoted: it trades
      agent throughput and TTFT for chat TTFT. Static split 3072 is dominated by deadline reservation.
    - [x] **Agent ITL SLO 80 ms (done 00:19, not promoted):** TraceLab is agent-only and the
      agent tenant's evaluation SLO is 80 ms; EDF + reservation with `ITL_SLO_MS=80`, TraceLab C = 4 /
      8, two reps (`edf_slo80_20261015.sh`). S3 behind it.
      Rep 1-2 TraceLab C=4: mean TTFT 3319 ms (-5% vs MTP), cold 27.2 s (368 TFLOP/s vs 446 at 50 ms),
      admitted chunk 2049-2654: the larger budget barely enlarges chunks and does not speed cold prefill.
      **Done (00:19), 4 arms clean.** Mean TTFT rep1 / rep2: C=4 3319 / 2570 ms (mean -16% vs MTP),
      C=8 5740 / 7605 ms (mean +9%); throughput 117-124 tok/s and ITL equal to the 50 ms arms. The rep
      spread (25-30%) exceeds any effect; **not promoted** -- chunk size does not limit TTFT, consistent
      with prefill being near compute-bound (below).
    - [x] **Is TraceLab TTFT compute-bound? (2026-10-16, traced 04:39):** the SM-cap hypothesis does not apply --
      the split launches upstream FA3 with `sm_margin` 0 (all SMs); `SGLANG_FUSED_SM_FA3` only sizes
      the fused single-launch path. In-server prefill rate from the existing cohort step traces
      (`steptrace_prefill_rate.py`, single-prefill steps incl. co-scheduled decode time): chunk 2048
      ~500 TFLOP/s, chunk 8192 ~570 TFLOP/s, near the H200 FA3 ceiling (~650-750). Prefill is close to
      compute-bound; small chunks cost ~12-15%; the rest of cold TTFT is cold prompts queueing behind
      each other. TraceLab C=4 step trace (MTP | EDF + deadline reservation) queued:
      `steptrace_tracelab_20261016.sh`, after the output gate; S3 behind it.
      **TraceLab C=4 traced (04:39), both arms clean.** In-server prefill rate on single-prefill
      steps, prefix <32K / 32-128K / 128-256K / >=256K (TFLOP/s, co-scheduled decode time included):
      MTP 401 / 373 / 359 / 351 (FA2 BatchPrefill, chunk 8192); EDF + deadline reservation 507 / 476 /
      484 / 494 (FA3 prefill rows, chunk 2048) -- **+30% prefill rate from the kernel, holding at
      >= 256K prefixes**. Total GPU-busy time 1029 -> 831 s (-19%); this traced pair's mean TTFT 3437
      -> 2843 ms (-17%; one rep, tracing on). The split reaches ~70% of the ~700 TFLOP/s FA3 ceiling;
      the rest is the 2048 chunk and the decode/verify rows sharing each step. Mean TTFT gains less
      than the prefill rate because TTFT also contains cold prompts queueing behind each other and
      the decode share of every step (which is what buys the ITL gain). ~~Single-GPU mean TTFT on
      full-length TraceLab is therefore bounded near the prefill-rate gain (~-20..-30%) at best; the
      30% bar there needs more prefill capacity (multi-node, PD points) rather than scheduling.~~
      **Retracted (2026-10-16 06:0x):** the parallel session's EDF chunk preemption (park the chunked
      long prompt, serve uncached-short waiting rounds; `edf_preempt_chunk_v2.patch`) reaches TraceLab
      mean TTFT -39/-40% (C=4) and -32/-35% (C=8, 16K cap / SRPT) vs MTP on one GPU. The prefill-rate
      measurement stands; the inference from it was wrong -- most of the remaining TTFT was blocking,
      not compute. The TraceLab miss is now mean ITL (-20..-25%).
    - [x] **Serving-output gate, reservation trees (2026-10-16 01:33, `resv_gate_20261015.sh`):**
      Codex's EDF gate driver (`drivers/srv_correctness_edf_20261014.py`, BENCH_CHUNK 512, 1200
      decode tokens, guard 0), all arms 0 tracebacks / fallbacks / width mismatches, graphs on.
      MTP repeat vs MTP (noise floor): min common prefix short 15 / long 39, PASS. Split + EDF +
      deadline reservation: short 1 / long 11, PASS by the rule (long >= 8); short-request
      reservation: identical 1 / 11. No reservation fired on the gate workload, so both test split +
      EDF. **Not a clean pass:** far below the floor, and earlier split-only gates kept long prefixes
      of 52-96. EDF changes batch composition, and greedy bf16 outputs depend on it, so the MTP repeat
      (same schedule) is an optimistic floor -- control queued: split-only, EDF-only (stock FA2 4-mode)
      and split + EDF, each vs a fresh MTP (`gate_control_20261016.sh`).
      **Control (04:48), all arms clean:** EDF-only (stock FA2 4-mode) short 18 / long 39 = the MTP floor;
      split-only 1 / 11; split + EDF 1 / 11. The early divergence is the split kernel path, not EDF.
      This matches the Codex role-isolation finding (`stock_fa3_dispatch_20261014.md`): the generation
      difference localizes to the FA3 primary, whose split-KV reduction differs from unsplit by up to
      4.2e-3 per-head relative error, while the attention checks against fp32 pass. Same-input check
      on this tree queued (`split_verify_20261016.sh`: `SGLANG_FUSED_SPLIT_VERIFY=1`, every split step
      recomputed with the stock FA2 wrapper, FAIL above 0.02 scale-normalised error).
      **Same-input check (06:36-06:38), both arms clean:** split-only and split + EDF, every split step
      recomputed with the stock FA2 wrapper on the same inputs: **0 FAIL lines**, sampled error 0.0059
      (FA3 rows 0.0059, FA2 rows 0.0029-0.0059), worst 0.0112 < 0.02. The split's attention agrees with
      stock within tolerance; the earlier greedy divergence (1 / 11 with this driver; the canonical
      `srv_correctness.py` driver gives long 52 on the same split code) is bf16 FA3-vs-FA2 numerics
      flipping near-tie tokens, not a wrong result. **Gate closed: PASS** for both reservation trees
      (no reservation fired on the gate workload; their only change is scheduling).
  - [x] **Final-config arms (2026-10-15, run 2026-10-16):** MTP | split static 3072 | split + EDF 50 ms +
    floor 2048, tree `up_edf_floor`, guard 0, no barrier (`s3_standalone_final_20261015.sh`, root
    `s3_standalone_final_20261015`; yields to `codex_needs_gpu`). Replaces the never-run
    2026-10-13 arms. Re-queued 13:1x behind `concurrency_sweep_20261015.sh`; load axis is request
    rate only (Mooncake: recorded per-request timestamps compressed, achieved req/s reported).
    **ShareGPT (2026-10-16 07:27-08:08, 12 arms clean):** req/s achieved / out tok/s / mean TTFT /
    mean ITL / P99 ITL, acceptance 2.04-2.09 in every arm:
    | rate | MTP | split 3072 | split + EDF + floor |
    |---|---|---|---|
    | 4 | 3.99 / 847 / 43 ms / 4.17 / 13.5 | 3.99 / 847 / 49 / 4.16 / 15.8 | 3.99 / 847 / 48 / 4.16 / 16.9 |
    | 8 | 7.99 / 1633 / 48 / 4.68 / 18.5 | 7.99 / 1634 / 55 / 4.76 / 24.9 | 8.00 / 1635 / 57 / 4.80 / 25.0 |
    | 16 | 15.01 / 3074 / 95 / 6.18 / 31.2 | 15.02 / 3077 / 72 / 6.20 / 33.9 | 14.93 / 3059 / 76 / 6.21 / 34.0 |
    | 32 | 24.03 / 5049 / 17.3 s / 9.10 / 44.1 | 24.73 / 5198 / 14.7 s / 8.75 / 47.3 | 22.92 / 4818 / 20.2 s / 9.47 / 43.9 |
    **No gain on short-prompt ShareGPT:** below capacity all arms are equal within a few ms (the split
    adds ~6 ms TTFT and some P99 ITL); at saturation (32/s) the split gives +3% throughput and -15%
    TTFT, while EDF + floor loses (-5% throughput, +16% TTFT: the 2048 floor and 50 ms budget throttle
    prefill when every request is short). The 4-mode gains are specific to long-context mixing
    (agent / TraceLab / mixtures), as the B1 and step-trace evidence predicts.
    **GSM8K / MATH-500 / HumanEval / Code-Feedback (08:08-09:21, 36 arms, 0 tracebacks / fallbacks):**
    dataset size caps N (GSM8K 1319, MATH-500 500, HumanEval 164 prompts), so their higher-rate points
    exit rc=1 on the harness's "did not complete all N" check after serving every available prompt --
    all three arms see the same requests. MTP vs split vs EDF+floor, mean TTFT ms / mean ITL / P99 ITL:
    | point | MTP | split 3072 | EDF + floor |
    |---|---|---|---|
    | GSM8K 8 | 30 / 3.38 / 15.0 | 38 / 3.44 / 11.8 | 38 / 3.43 / 11.7 |
    | GSM8K 32 (31.0 req/s) | 36 / 4.99 / 18.4 | 51 / 5.34 / 25.0 | 50 / 5.33 / 24.8 |
    | MATH-500 32 (27-28 req/s) | 91 / 6.16 / 21.7 | 104 / 6.24 / 29.7 | 222 / 6.58 / 28.6 |
    | HumanEval 32 | 37 / 5.30 / 18.2 | 53 / 5.94 / 26.4 | 53 / 5.80 / 25.9 |
    | Code-Feedback 16 | 63 / 5.65 / 22.2 | 77 / 5.79 / 28.6 | 82 / 5.82 / 28.7 |
    | Code-Feedback 32 (saturated, 18 req/s) | 29.1 s / 6.75 / 25.1 | 27.5 s / 6.55 / 32.4 | 31.5 s / 6.90 / 31.9 |
    Throughput is equal (within 1-2%) everywhere below saturation. **On short prompts the 4-mode
    split is slightly worse:** +8..+16 ms TTFT, +1..+7% mean ITL, +30..+45% P99 ITL (prefill now shares
    the step with verify rows); at saturation (Code-Feedback 32) it gains +2% throughput and -5% TTFT.
    EDF + floor matches the split or is worse. The gains measured on the cohort / TraceLab come from
    long-context prefill mixed with decode; a production deployment should enable the 4-mode step only
    when such prefill is present (a dispatch decision per step, not a fallback).
    **Mooncake conversation / tool-agent (09:21-10:37, 18 arms clean; recorded per-request timestamps
    at 0.5 / 1 / 2x speed, 600 requests, ~3.7K-token prompts):** req/s achieved / mean TTFT / mean ITL /
    P99 ITL, MTP vs split vs EDF + floor:
    | trace, speed | MTP | split 3072 | EDF + floor |
    |---|---|---|---|
    | conversation 0.5x | 1.49 / 606 ms / 6.1 / 13 | 1.49 / 524 / 6.3 / 41 | 1.49 / **452** / 6.6 / 46 |
    | conversation 1x | 2.98 / 634 / 8.6 / 72 | 2.98 / 545 / 8.3 / 81 | 2.98 / **472** / 9.0 / 68 |
    | conversation 2x | 4.26 / 15.6 s / 28.2 / 236 | 4.63 / 9.5 s / 24.8 / 118 | **4.76 / 8.3 s / 24.6 / 96** |
    | tool-agent 0.5x | 2.73 / 589 / 6.6 / 13 | 2.73 / 505 / 6.7 / 45 | 2.73 / **422** / 7.0 / 64 |
    | tool-agent 1x | 5.36 / 624 / 9.0 / 147 | 5.35 / 523 / 8.7 / 83 | 5.36 / **430** / 9.5 / 70 |
    | tool-agent 2x | 6.70 / 7.5 s / 27.4 / 231 | 8.12 / 3.8 s / 23.0 / 113 | **8.13 / 3.1 s / 23.1 / 91** |
    **Medium-length prompts show the gain:** EDF + floor cuts TTFT 25-31% below capacity (ITL +5..+8%);
    at 2x, where MTP falls behind the trace, EDF gives throughput +12% (conversation) / +21% (tool-agent),
    TTFT -47% / -58%, P99 ITL -59% / -60%, mean ITL -13% / -16%. Throughput and TTFT clear the bar at
    2x; mean ITL does not.
    **LongBench v2 -- retracted (2026-10-16 12:11):** the rate grid 0.25 / 0.5 / 1 req/s was carried
    over unreviewed from the 2026-10-13 script. LongBench v2 prompts average ~57K tokens (11.39M input
    tokens / 200 requests) and one H200 prefills ~14.5K tok/s, so capacity is ~0.25 req/s: every point
    was at or past saturation (MTP mean TTFT 57 s at 0.25/s, 241 s at 0.5/s with 0.25 achieved). These
    points measure queue growth, not serving latency; not reported. Stopped after 0.25 and 0.5 (the
    1 req/s arms were killed). The references set load around measured capacity -- Crossflow by request
    rate (timestamp compression), ThunderAgent by concurrency; a long-document set without timestamps
    belongs on program concurrency (documents in flight), not req/s.
    **Scope (user, 2026-10-16):** S3 is ShareGPT, math reasoning (GSM8K, MATH-500), coding (HumanEval,
    Code-Feedback) and agentic web browsing only; LongBench and Mooncake leave S3 (the Mooncake rows above
    stay as recorded). **Web browsing (12:26-12:59):** THUDM/AgentInstruct WebShop, 351 multi-turn
    trajectories (4-10 turns, mean 4.7), converted by `drivers/serving/prep_webshop_agentic.py` and replayed
    round by round with the server's own replies (`sglang-oai-chat`, 64 output tokens per turn); load is
    program concurrency C = 8 / 16 / 32 / 64. Arms MTP | split 3072 | EDF + floor | EDF + floor + chunk
    preemption, tree `up_edf_preempt` (`s3_webshop_agentic_20261016.sh`, runner `run_agentic_standalone.py`).
    **WebShop done (12:26-12:59, 16 arms, 0 tracebacks / fallbacks, 0 parks; acceptance 1.78 in every
    arm).** Every arm completes all 1642 turns of the 351 conversations; rc=1 is only the runner's
    "requests == conversations" check (the bench counts turns). Out tok/s / mean TTFT ms / P99 TTFT /
    mean ITL / P99 ITL:
    | C | MTP | split 3072 | EDF + floor | EDF + preempt |
    |---|---|---|---|---|
    | 8 | 1246 / 40 / 70 / 5.93 / 24.9 | 1162 / 47 / 82 / 6.29 / 30.9 | 1156 / 48 / 86 / 6.31 / 31.5 | 1158 / 48 / 83 / 6.29 / 31.5 |
    | 16 | 1916 / 47 / 92 / 7.77 / 29.4 | 1732 / 59 / 120 / 8.50 / 36.0 | 1724 / 61 / 112 / 8.49 / 36.3 | 1714 / 60 / 105 / 8.57 / 36.3 |
    | 32 | 2577 / 61 / 201 / 11.4 / 41.5 | 2324 / 83 / 235 / 12.4 / 47.4 | 2390 / 81 / 328 / 12.1 / 46.9 | 2383 / 82 / 273 / 12.1 / 47.4 |
    | 64 | 2995 / 375 / 736 / 14.8 / 60.5 | 3148 / 380 / 821 / 13.6 / 64.0 | 2765 / 432 / 888 / 15.7 / 60.7 | 2697 / 447 / 908 / 16.0 / 64.4 |
    **Web browsing loses on one GPU at C <= 32:** throughput -7..-10%, TTFT +18..+36%, ITL +6..+9% for
    every 4-mode arm -- WebShop observations are short (no long prefill to co-schedule; preemption never
    parks), the same regime as the other short-prompt sets. At C=64 the split gains +5% throughput and -8%
    ITL; EDF arms lose there. Consistent across S3: the 4-mode step pays off only where long-context
    prefill shares the GPU with decode (cohort, TraceLab, Mooncake 2x).
    - [x] **4-mode context gate (2026-10-16 13:12-14:33, `mixed_gate_ab_20261016.sh`, 12 arms clean):**
      `SGLANG_MIXED_SPEC_MIN_CTX` (`mixed_spec_ctx_gate.patch`, tree `up_edf_gate` = `up_edf_preempt` +
      gate): a step mixes its prefill into the verify step only if some prefill row's prefix + chunk >=
      the gate, else the prefill runs alone as in MTP; `MIXEDSPEC_GATED` counts gated steps. MTP vs
      EDF + floor + preemption ungated vs gated at 8192 (out tok/s / mean TTFT / mean ITL / P99 ITL):
      | point | MTP | ungated | gate 8K | gated / mixed steps |
      |---|---|---|---|---|
      | WebShop C=16 | 1908 / 47 ms / 7.80 / 29.7 | 1696 / 60 / 8.66 / 36.4 | 1788 / 42 / 8.46 / 43.9 | 1201 / 0 |
      | ShareGPT 16 req/s | 3058 / 97 / 6.25 / 32.0 | 3078 / 68 / 6.26 / 34.5 | 3070 / **54** / 6.41 / 32.8 | 1601 / 1 |
      | cohort C=16 agent | 106.8 / 43.3 s / 28.0 / 503 | **137.3 / 31.6 s / 17.6 / 92** | 126.5 / 34.7 s / 19.6 / 104 | 1401 / 1151 |
      | cohort C=16 chat | 1219 / 51.8 s / 31.4 / 473 | **1528 / 28.4 s / 25.0 / 129** | 1404 / 34.4 s / 28.4 / 142 | |
      | TraceLab C=8 | 94.8 / 6461 / 58.6 / 1276 | 127.4 / 4214 / 43.8 / 310 | 126.8 / 4089 / 44.3 / 293 | 1 / 1801 |
      **Not promoted.** On short prompts it removes most of the mixed-step cost (ShareGPT: throughput
      equal to MTP, TTFT -44%; WebShop: throughput -6% instead of -11%), and TraceLab never gates.
      But on the overloaded cohort, gating the chat-only prefill steps stalls decode during them:
      agent throughput -8%, chat ITL 25.0 -> 28.4 ms vs ungated. Prompt length is the wrong signal --
      mixing pays when the server is loaded and costs at moderate load; a load-based gate (queue depth /
      running count) is the next candidate. Ungated EDF + preemption vs MTP on these runs: TraceLab C=8
      tput +34%, TTFT -35%, ITL -25%, P99 ITL -76%; cohort agent +29% / -27% / -37%, chat +25% / -45% /
      -20%.
    - [x] **Load + context gate -- promoted (2026-10-16 14:38-15:13, `mixed_gateq_ab_20261016.sh`, 9 arms
      clean):** `SGLANG_MIXED_SPEC_MIN_QUEUE` (`mixed_spec_ctx_queue_gate.patch`, tree `up_edf_gateq`)
      re-enables mixing whenever >= Q requests wait; gate = context 8192 OR queue 4. Same-queue, vs MTP
      and ungated EDF + preemption (out tok/s / mean TTFT / mean ITL / P99 ITL):
      | point | MTP | ungated | gate ctx 8K + queue 4 | gated / mixed steps |
      |---|---|---|---|---|
      | WebShop C=16 | 1919 / 46 ms / 7.76 / 29.5 | 1723 / 61 / 8.52 / 36.0 | 1786 / **43** / 8.48 / 43.4 | 1401 / 0 |
      | ShareGPT 16 req/s | 3091 / 95 / 6.21 / 30.9 | 3076 / 68 / 6.29 / 34.1 | 3087 / **52** / 6.28 / 33.5 | 1601 / 1 |
      | cohort C=16 agent | 107.8 / 42.9 s / 27.8 / 505 | 139.5 / 30.8 s / 17.6 / 91 | **139.2 / 30.9 s / 17.6 / 91** | 1 / 2651 |
      | cohort C=16 chat | 1228 / 51.2 s / 31.0 / 472 | 1542 / 27.0 s / 24.7 / 129 | **1549 / 27.1 s / 24.6 / 129** | |
      Matches or beats the ungated arm at every point: the overloaded cohort keeps the full gain (the
      queue clause keeps mixing on), ShareGPT keeps MTP throughput with TTFT -45%, WebShop's loss vs MTP
      shrinks from -10% to -7% throughput (ITL still +9%: short-prompt agent traffic is not fully
      recovered). **Default dispatch from here: mix iff prefill context >= 8192 or >= 4 requests wait.**
      **Output gate (15:27-15:31, `gateq_gate_20261016.sh`, 3 arms clean, graphs on):** MTP repeat vs MTP
      short 15 / long 39 (floor); split + EDF + preemption + gate short **37** / long 18, PASS (gate fired,
      FUSED_SPLIT_CALLS 1001). Short prompts now prefill alone as in MTP, so they agree longer than the
      floor; the long-prompt gap is the FA3-vs-FA2 bf16 difference already adjudicated by the same-input
      check above.

- [ ] **S4, EDF step model** re-fitted on the fused kernel and compared with the unfused schedule
  - [x] **Step model on the 4-mode fused schedule vs stock MTP (2026-10-18, H200 `d6a80befdd4e4da5`, two traced A/Bs,
    each both arms clean, >= 66001 split calls):** same linear form (c0, prefill tokens, attention work, rows, KV), fit
    on the first half of each trace, held-out MAPE (`tasks/perf_model/fit_tpot_acc.py`):
    | step kind (GPU time) | 4-mode MTP, trace 1 / 2 | stock MTP, trace 1 / 2 |
    |---|---|---|
    | decode-only | 3.6% / 3.6% (p90 7.3 / 7.1%) | 2.7% / 2.7% (p90 5.2 / 5.3%) |
    | carries a prefill chunk | **7.8% / 8.5%** (p90 15.8 / 18.5%) | 49.5% / 44.2% (p90 89.4 / 84.3%) |
    | TPOT per accepted token (interval / acceptance) | 12.8% / 12.9% | 5.9% / 6.5% |
    The step EDF has to predict before it admits a prefill -- one that carries a chunk -- is ~6x more predictable on the
    4-mode schedule (point 3). The per-token interval is less predictable on ours (12.9% vs 6.5%) because ours
    alternates 9 ms decode and 44 ms mixed steps where stock's back-to-back steps are mostly 8 ms decode. Traces
    `STEP_TRACE=1 SPEC_TAG={_trace,_dacc} CHUNKS=2048 DEADLINE_CHUNK=1024:2048:6000 bash
    tasks/pd_colocation/mn_h200_fourmode_20261018.sh`; the second A/B is also bar rep 4: agent +46.5% / -28.1% /
    -34.2%, chat -71.0% / -24.1% vs its own stock MTP.
  - [x] **Acceptance length as a TPOT input (point 3), per request (2026-10-18):** `tasks/perf_model/tpot_acc_probe.py`
    streams 600 ShareGPT first turns (6 req/s Poisson, <= 256 output tokens) while 4 threads keep 40K-token prefills
    running, and reads each response's `completion_tokens / spec_verify_ct`. Fit on the first half by arrival, test on
    the second. **4-mode schedule:** TPOT 42.1 ms mean, acceptance 2.07 (sd 0.30, p10-p90 1.78-2.42; decode log 2.05),
    corr(TPOT, 1/acc) 0.88; TPOT = k / acc_i (the steady state of a per-request EWMA) MAPE **7.0%** (p90 17.4%) vs a
    constant acceptance 12.7% (p90 26.7%); adding running requests at arrival 6.8%. The predicted acceptance length
    halves the TPOT error EDF would plan with. **Stock MTP arm not usable:** its per-request acceptance reads 2.97
    (sd 1.05) while its own decode log reports 1.87, and corr(TPOT, 1/acc) is -0.70 -- stock's `spec_verify_ct`
    undercounts verifies on some path; open check below. `python tpot_acc_probe.py /dev/shm/w/probe_ours ours` (and
    without `ours`) on pod `d6a80befdd4e4da5`; sglang `af3a39efc` + this commit.
  - [x] **Stock MTP's per-request acceptance under load is a counter artifact; the 4-mode schedule loses no
    acceptance (2026-10-18).** Same probe without the background prefills (`BG=0 N=300`): chat acceptance stock 2.13
    / ours 2.09 (sd 0.30 both, corr(TPOT, 1/acc) 0.90 / 0.89). With them: ours 2.08 (unchanged), stock 2.96 at every
    output length (verifies per token 0.36 vs ours 0.49-0.60), while both servers' decode logs read 2.06 / 2.04. So
    stock's `spec_verify_ct` (batch_result_processor's per-request increment) misses verifies when decode interleaves
    with chunked prefill steps; acceptance itself is equal on the two schedules, and the mixed step does not degrade
    drafting. The missed increment path in stock is not located (it does not affect any bar metric).
  - [x] **The stock-MTP "undercount" is non-speculative decode steps, not a counter bug (2026-10-18).** Per-chunk
    probe (`tpot_acc_probe.py` with chunk records, N=200, 4 background 40K prefills): ours emits ~123 token-adding
    stream chunks per 256-token request at 2.08 tokens per chunk (= its acceptance, 2.06); stock MTP emits ~154 at
    1.66 tokens per chunk, ~105 of them single-token, against ~85 verifies. Under interleaved chunked prefill stock MTP
    advances its decode rows in plain one-token steps (no draft / verify) for ~40% of their steps; those tokens count
    in `completion_tokens` but not `spec_verify_ct`, so its tokens-per-verify reads 3.0. Ruled out on the way:
    retraction (counter set only in `Req.__init__`, no retractions logged) and the decode path (increments once per
    verify). This is part of why the 4-mode step wins on stock MTP's long-prefill load: ours keeps speculating through
    every mixed step (2.06 tokens per step) where stock falls back to 1. `N=200 python tpot_acc_probe.py
    /dev/shm/w/probe_{stock,ours}_chunks [ours]` on pod `d6a80befdd4e4da5`.
  - [x] **Acceptance length from the step trace -- not testable (superseded by the per-request probe above).** `acc=` is the server's
    cumulative mean, and the per-step delta (`apply_steptrace_dacc.py`, `dacc=`) is zero on most steps because the
    metrics reporter accumulates accept counts only at its decode-log interval; with either signal the EWMA input does
    not change TPOT MAPE (12.9 vs 12.6% ours, 6.5 vs 6.5% stock), which says nothing yet. Needs per-request accepted
    tokens per verify from the batch result (EAGLE accept lengths) logged per step, then a per-request EWMA.
  - [x] **Step model on the final schedule (PD multiplexing + EDF, uncached deadlines, divisor 30) vs the stock
    (unfused) schedules (2026-10-17 18:50-19:26, 6 traced arms rc=0, 0 tracebacks / 0 fallbacks):** STEPTRACE now
    covers pdmux's event loop (`apply_steptrace_pdmux.py`: decode and split-prefill forwards traced on their own
    streams, with the SM group `sg=` and split layers `lay=`). Lightweight linear models, fit on the first half of
    each trace, held-out MAPE on the second (`tasks/perf_model/fit_pdmux_step.py`):
    | what EDF predicts | pdmux + EDF | stock chunked LPT | stock MTP |
    |---|---|---|---|
    | decode interval a request sees (ITL), C=8 / 16 | **8.9% / 8.9%** (p90 15 / 21%) | all steps 117% / 74% | all steps 52% / 35% |
    | decode-only step GPU time, C=8 / 16 | **6.7% / 4.0%** | 17.5% / 7.5% | 66.8% / 32.3% |
    | prefill work (pdmux: one split forward; stock: mixed step), GPU, C=8 / 16 | 22.5% / 246% (p90 62 / 95%) | 56.7% / 52.5% | 43.4% / 46.8% |
    pdmux decode model: per SM group, c0 + c_bs*rows + c_kv*KV (21 coefficients); stock: fit_mtp_step.py's form
    (c0, prefill tokens, attention work, rows, KV). **With decode on its own SM group, the per-token interval is
    predictable to ~9%; on the unfused schedules prefill chunks interleave with decode and the same kind of model
    misses by 35-117%** -- the user's point-3 thesis (a schedule that is easy to model gives better SLO control)
    holds for ITL on this schedule, not for the fused-kernel step (whose co-run bound is 1.2%). TTFT-side prediction
    is the weak part: a split prefill shares the GPU with a varying decode load (p90 62-95%). The project fitter
    (`fit_mtp_step.py`, launch-to-launch target) gives >100% on the stock traces under the overlap scheduler, so the
    2026-10-14 10.1 / 10.7% figures (4-mode MTP server, other workload) are not comparable to these. Output:
    [`pdmux_step_model_20261017.txt`](perf_model/pdmux_step_model_20261017.txt); traces `REP=1 bash
    steptrace_pdmux_20261017.sh` (pod `/dev/shm/w`).
  (goal point 3); earlier fits used the RoPE-corrupted model and are void (incl. the uncommitted
  2026-10-12 05:10 "fused-vs-unfused EDF check" in `server_perf_model.md`). The STEPTRACE emitter
  did not survive the re-port; ported to the overlap path with the 4-mode verify rows counted
  (`steptrace_port.patch`). `s4_edf_20261013.sh`, on the race-fixed tree without the barrier (a
  per-step sync would distort the modelled step times), runs only if the race fix passes: traces
  on both 4-mode arms (mix32 8/16/24/s, mix128 12/s), one linear model per arm (held-out MAPE),
  then EDF-R with each arm's own model vs no EDF at mix32 16/24/s -- queued after S3.

  - [x] **Guard-0 refit + EDF dynamic chunk, overloaded cohort (Claude, 2026-10-15; builds on the
    Codex kernel worker's EDF runtime / budget fix / mode-aware fitter):** step traces with the
    decode guard off (split and unfused, chunk 8192 and 2048; 5.4-6.6K rows each), fitted with
    `fit_step_matrix_piecewise` (piecewise-relative): held-out MAPE **split 10.1% vs unfused
    10.7%** (goal point 3: the split is slightly easier to predict; Codex's guard-on fit was
    13.9% vs 16.3%). Acceptance stays out of the step model because the traced value is the
    lifetime average (2.12-2.19, not identifiable); it enters the EDF budget instead
    (slack = ITL_SLO x per-request acceptance EWMA - time since the row's last token).
    Serving on the frozen 32-64K cohort, all arms in Codex's `up_edf_budgetfix` tree:
    | arm (2 runs where shown) | chat tput / TTFT / mean ITL / P99 | agent tput / TTFT / mean ITL |
    |---|---|---|
    | MTP | 928 / 179 s / 32.2 / 486 ms | 83 / 121 s / 93 ms |
    | split static 3072 (x2) | 1240 / 125 s / 22.0 / 168 ms | 116 / 95 s / 29 ms |
    | EDF itl 50 ms | 1074 / 149 s / 19.9 / 75 ms | 105 / 115 s / 18 ms |
    | EDF itl 30 ms (x2) | 849 / 211 s / 16.2 / 50 ms | 80 / 148 s / 13 ms |
    EDF meets its ITL target but admits only ~1.3K (50 ms) / ~300 (30 ms) prefill tokens per step:
    this cohort is past saturation in every arm (TTFT 40-200 s), every decode row is late, and a
    slack-sized chunk starves prefill -- at 30 ms TTFT is worse than stock MTP. Confirms
    `EDF: order, don't cap` on the current stack. **Caveat:** all absolute numbers in this tree
    are ~30% slower than the same frozen cohort in the chunk-sweep tree, MTP included (identical
    server args and requests; GPU unthrottled; scheduler host-bound at ~100% CPU) -- compare only
    within a run. ([script](debug_kernel_server_perf/resume_20261012/edf_dynchunk_20261015.sh))
    **Correction (2026-10-15 12:40):** the EDF trees (`up_edf_budgetfix`, `up_edf_floor`) lacked the
    program-concurrency replay (`BENCH_SESSION_CONCURRENCY` silently ignored; no `PROGRAM_CONCURRENCY`
    line in any bench log), so every program was released at once: measured mean in-flight
    31.4-34.8 in every arm, not the stated 24 / 8 / 16 programs. Arms within a run stay matched, but
    the load labels are wrong and "overloaded at any program concurrency" was never tested.
    Rerun with the replay ported (`edf_floor_session_concurrency.patch`): `concurrency_sweep_20261015.sh`.
  - [x] **EDF with a prefill floor (2026-10-15 07:20-08:01):** `SGLANG_EDF_PREFILL_FLOOR`
    (`edf_prefill_floor.patch`, tree `up_edf_floor`), MTP / static 3072 / EDF 50 ms + floor 2048,
    all arms clean. Both lighter points turned out still past saturation (chat TTFT 95-158 s): the
    frozen cohort's ~4.7M agent prefill tokens keep it overloaded at any program concurrency. The
    floor binds (admitted prefill = 2048 every logged step), so EDF acts as chunk 2048 + deadline
    order. EDF+floor vs static 3072: equal-or-better on everything except agent TTFT -- 8p/4/s: chat
    TTFT -10%, chat ITL -10%, chat P99 ITL -27%, agent ITL -16% (tput, agent TTFT within noise);
    16p/6/s: chat TTFT -6%, P99 ITL -24%, agent ITL -21%, agent TTFT +5%. Vs MTP: 8p chat
    +38% / -35% / -42% (tput/TTFT/ITL), agent +41% / **-16%** / -74%; 16p chat +34% / -30% / -34%,
    agent +41% / **-15%** / -78%. Agent TTFT stays the gap (queueing behind its own prefill
    backlog). Next: across capacity on the S1 mixture. **Retracted (2026-10-15 12:40):** "8p" and
    "16p" ran with all 48 programs released (mean in-flight 33.1-35.5), see the correction above.
  - [x] **EDF + floor on the session-rate axis -- retracted (2026-10-15).** Two sweeps (mix32 and mix128, chat 8-16/s,
    agents at 0.1 sessions/s) drove the agent tenant by session starts, not program concurrency, so agent load was
    uncontrolled and the numbers are not evidence for the bar. Superseded by the concurrency sweep in S2b (cohort
    C = 8 / 16 / 24 + chat 8 req/s). One finding survives as a within-run comparison: EDF's TTFT deadline grows with
    prompt length (+1 s per 5K tokens past 4096), so a 200K agent prompt gets ~40 s and always waits while the
    evaluation scores agents against 10 s. Data `edf_ratesweep_20261015.csv`, `edf_ratesweep2_20261015.csv`
    (`debug_kernel_server_perf/resume_20261012/`).
  - [x] **Two-tier TTFT SLO in EDF -- not promoted (2026-10-15 10:39-11:20; same session-rate axis, the
    rejection stands as a within-run comparison):** prompts >= 16K get
    the flat agent SLO (10 s), shorter ones the chat SLO (2 s) (`edf_two_tier_slo.patch`, opt-in
    `SGLANG_EDF_LONG_PROMPT_TOKENS`), EDF+floor with vs without, same run, all arms clean.
    mix32 12/s: no effect (32-64K prompts already had ~8-14 s deadlines; agent TTFT is its own
    prefill time). mix128 12/s: agent TTFT 26.0 -> 21.3 s (-18%, P99 -37%) but chat TTFT
    16.7 -> 22.5 s (+35%) -- it moves TTFT between tenants. mix128 16/s: worse for both (chat
    tput -8%, chat TTFT +73%, agent TTFT +11%). Standard length-scaled deadlines stay. This run's
    standard-deadline control at mix128 16/s reproduces sweep 2 with margin: agent TTFT 26.1 s vs
    MTP 44.2 s (-41%). Data [`edf_tier_20261015.csv`](debug_kernel_server_perf/resume_20261012/edf_tier_20261015.csv),
    `edf_tier_20261015.sh`.
- [x] **Chunk size: not the lever with the decode guard on (2026-10-12); IS the ITL lever with guard 0 (2026-10-15, supersedes):** chunk 8192/4096/2048 leaves
  both arms unchanged (prefill per step is many ~512-token prompts).
  - **2026-10-15 update (guard 0, split serving kernel):** with long prefills now routed to FA3
    (Codex found the decode-population guard kept every 32K+ prefill off FA3 in timed runs), the
    mixed-step chunk sets step time and hence ITL. Frozen 32-64K cohort (24 agent programs +
    natural ShareGPT 8/s), all arms audited; noise floor from the 8192 repeat: 2-3%.
    | vs MTP (8192) | split 8192 (mean of 2) | split 4096 | split 2048 |
    |---|---|---|---|
    | chat tput / TTFT / mean ITL | +31% / -43% / -26% | +32% / -43% / -28% | **+34% / -43% / -34%** |
    | agent tput / TTFT / mean ITL | +34% / -31% / -33% | +34% / -30% / -42% | **+38% / -28% / -56%** |
    MTP itself does not gain from small chunks (2048: chat ITL 29.40 vs 29.15 ms, tput -8%, TTFT
    +20%), so 8192 is its best setting and the comparison is fair. At chunk 2048 every metric for
    both tenants clears the bar except **agent TTFT (-27.6%)**; at 4096 agent TTFT clears (-30.2%)
    but chat ITL does not (-28.3%).
    **With repeats (sweep 2b, two runs per arm; MTP repeats within 1%):** 2048 gives chat
    +32% / -40% / -32% and agent +35% / **-25%** / -57%; 3072 gives chat +31% / -41% / **-29.1%**
    and agent +33% / **-27.4%** / -49%. No static chunk clears both chat ITL and agent TTFT: a
    smaller chunk protects decode ITL but delays agent prefills. Next: EDF dynamic chunk (prefill
    sized to the tightest decode slack each step), `edf_dynchunk_20261015.sh`, running.
    ([table](debug_kernel_server_perf/resume_20261012/chunk_itl_sweep_20261015.txt),
    `chunk_itl_sweep_20261015.sh`)
- [ ] **Direct baselines:** MuxWise (arXiv 2504.14489), semi-PD, NVIDIA Dynamo conditional
  disaggregation and sglang PD -- see References.
  - [x] **sglang mixed-chunk is already the stock arm (2026-10-13):** upstream keeps
    `--enable-mixed-chunk` on for EAGLE3 and degrades running requests to plain 1-token decodes
    in a mixed step. The S1 `MTP_LPT` server log shows `enable_mixed_chunk: True`, no disable
    warning, and 3151 prefill batches carrying running requests (mix32 16/s). So every "stock
    MTP" number is stock mixed-chunk + MTP; the 4-mode step's difference is that those requests
    keep speculating (verify rows) instead of dropping to single-token decode.
  - [x] **MuxWise baseline measured on the program-concurrency axis (Claude, 2026-10-17 08:51-09:40; H200 idle --
    Codex's owner PID992767 gone, lock free, noted in the coordination file).** Frozen 32-64K TraceLab cohort (48
    programs / 96 rounds) at agent C = 8 / 16 + natural ShareGPT 8 req/s, tree `up_edf_floor`, one session, one run per
    arm; all 8 arms rc=0, 0 tracebacks, 0 fallbacks; PDMUX server log `enable_pdmux: True`; ours FUSED_SPLIT_CALLS
    67 501 / 68 001; every agent arm logged `concurrency_limit` = C. Out tok/s / mean TTFT / mean ITL / P99 ITL, and
    tok/s/user p1 / p50 / p99 ([summary](debug_kernel_server_perf/baselines_20261017/pdmux_conc_20261017_summary.txt)):
    | C | arm | agent | chat | chat tok/s/user p1 / p50 / p99 |
    |---|---|---|---|---|
    | 8 | stock non-spec (LPT) | 123 / 13.2 s / 25.7 / 466 | 1 498 / 14.1 s / 26.3 / 465 | 7 / 52 / 156 |
    | 8 | **PD multiplexing (`--enable-pdmux`)** | 139 / 13.1 s / 15.7 / 39 | 1 612 / 11.3 s / **13.8 / 32** | **35 / 71** / 130 |
    | 8 | stock MTP | 110 / 18.8 s / 21.0 / 415 | 1 291 / 23.5 s / 30.3 / 476 | 8 / 42 / 162 |
    | 8 | ours (split + EDF 50 ms + floor 2048) | 139 / 14.7 s / 15.4 / 72 | 1 575 / **5.8 s** / 25.0 / 131 | 18 / 46 / 144 |
    | 16 | stock non-spec (LPT) | 127 / 29.8 s / 32.1 / 528 | 1 482 / 33.3 s / 24.6 / 456 | 7 / 59 / 155 |
    | 16 | **PD multiplexing** | 142 / 29.0 s / 16.9 / 41 | 1 609 / 25.3 s / **13.7 / 32** | **37 / 71** / 130 |
    | 16 | stock MTP | 107 / 43.4 s / 27.6 / 500 | 1 222 / 52.0 s / 31.3 / 472 | 8 / 40 / 158 |
    | 16 | ours | 139 / 31.8 s / 16.9 / 75 | 1 555 / **23.5 s** / 24.0 / 128 | 17 / 48 / 129 |
    **Measured negative result: PD multiplexing beats our final single-GPU config on ITL.** Ours vs PD multiplexing --
    C=8: throughput tie (agent 139 / 139, chat -2%), chat TTFT **-49%**, agent ITL tie, but chat ITL **+81%** (25.0 vs
    13.8 ms, P99 4x), agent TTFT +12%; C=16: agent / chat throughput -2% / -3%, chat TTFT -7%, agent TTFT +9%, agent
    ITL tie, chat ITL **+75%** (P99 4x). Its decode runs on its own SMs (GreenContext split) every step, so decode
    rows never wait behind a prefill chunk; ours mixes a 2048-token chunk into decode steps, and a mixed step costs the
    chunk plus the decode (measured on H20: decoders' ITL 12-34 -> 380-426 ms beside a cold 32K prefill). Its slow-user
    tail is far better too (chat p1 35-37 vs 17-18 tok/s/user). Against stock MTP ours still wins (C=16: agent
    throughput +30%, agent TTFT -27%, chat TTFT -55%, chat ITL -23%), but against this direct baseline the >=30% ITL
    bar is not met, and only chat TTFT clears it (C=8). PD multiplexing is non-speculative; the same SM split under
    MTP's verify rows is the open combination. **Tried directly (09:48-09:51, `pdmux_mtp_conc_20261017.sh`,
    `pdmux_mtp_eager_20261017.sh`): stock EAGLE3 chain MTP + `--enable-pdmux` does not run in this tree.** The only
    explicit rejection is for UNO tree mode, and the server does enable PD multiplexing (8 stream groups, prefill/decode
    SMs 132/0 ... 0/132), but (1) with CUDA graphs it dies capturing the EAGLE3 draft graphs (`AttributeError: 'NoneType'
    object has no attribute 'decode_wrappers'` -- the draft graph runner hard-codes `enable_pdmux = False` while the
    pdmux attention backends are per stream group), and (2) as a labelled eager diagnostic (`--disable-cuda-graph`) it
    reaches `event_loop_pdmux` and fails on the first draft step (`AssertionError` in
    `speculative/topk1.py:95 draft_topk1_postprocess`): pdmux's event loop runs split prefill batches and decode on
    separate stream groups, and EAGLE's draft/verify state assumes the standard loop. Combining them is scheduler
    integration work (drive the EAGLE worker's draft and verify from the decode stream group while prefill runs on the
    other), not a flag.
    - [ ] **PD multiplexing under EAGLE3 MTP -- scoped (2026-10-17, from `multiplex/multiplexing_mixin.py
      event_loop_pdmux`).** The loop runs the decode batch on the decode SM group with `run_batch`, advances a
      layer-split prefill (`split_index` / `split_forward_count`, a token budget's worth of layers per iteration) on
      the prefill SM group, and merges finished prefill requests into the running batch. Needed: (1) the EAGLE worker's
      `forward_batch_generation` must honour the split for extend batches (it runs the whole target forward today);
      (2) EAGLE3 reads hidden states from 3 target layers, so the split prefill must keep those across iterations and run
      the draft extend when the last layer finishes; (3) `merge_batch` must carry `EagleDraftInput` -- the eager
      `topk1.py` assert is consistent with a merged batch lacking draft state; (4) draft CUDA graphs per stream group
      (or eager draft, at a measured cost). Gates: srv_correctness against the stock-LPT floor, then the same cohort A/B
      vs `--enable-pdmux` and vs ours. This is where the SM-pre-assignment idea meets decode ITL: the measured gap to
      PD multiplexing is chat ITL +75-81%.
      **Pieces (1)-(3) implemented (2026-10-17, eager path; not yet measured):**
      `tasks/debug_kernel_server_perf/pdmux_mtp_20261017/apply_pdmux_mtp.py` on tree `up_pdmux_mtp` (copy of
      `up_edf_floor`): `TpModelWorker.forward_batch_split_prefill` takes a capture mode; `llama.forward_split_prefill`
      keeps the EAGLE3 aux hidden states on the ForwardBatch across splits and hands them to the logits processor at
      the last layer; `EAGLEWorkerV2.forward_batch_split_prefill` runs the target slice with `CaptureHiddenMode.FULL`
      and, after the last slice, `_draft_extend_for_prefill` exactly as `_forward_prefill_batch` does; the scheduler's
      pdmux branch routes split prefills through it when speculation is on. Compiles. Gate running:
      `smoke_pdmux_mtp.sh` (stock MTP vs MTP + pdmux, both eager, 24 concurrent requests incl. 8 x 12-24K prompts,
      greedy 128 tokens, per-request common prefix).
      **Eager path works end to end (2026-10-17 11:17, six fix rounds, all in `apply_pdmux_mtp.py`):** beyond (1)-(3) it
      needed (4b) the draft extend to run as a normal EXTEND (it inherited SPLIT_PREFILL and was routed to the draft
      model's split forward), (4c) the draft extend's `EagleDraftInput` installed on the batch before it joins decode,
      (5) the eager pdmux per-stream decode-backend override restricted to the target runner (the draft decode runs on
      the multi-step draft backends `draft()` planned; the override left `forward_metadata` None), (6) the same EAGLE3
      aux capture in `qwen2.forward_split_prefill` (the target is Qwen2), (7) target verify planned on and run under
      the per-stream decode backend (as an extend it re-planned the split prefill's `attn_backend` between slices:
      "q.shape[0] (19885) does not match qo_indptr[-1] (12)"), (8) `input_ids` restored from the split ForwardBatch for
      the draft extend. **Smoke gate (`bash /dev/shm/w/smoke_pdmux_mtp.sh`): MTP + pdmux served all 24 requests, 0
      tracebacks; 20 / 24 identical to stock MTP over 128 greedy tokens (median common prefix 128, worst 8);
      acceptance length 3.03-3.13, 17 split prefill batches beside decode.** The worst-case prefix is within the bf16
      batching noise measured before (split vs stock min common prefix 5 / 15) but this is not the formal gate:
      `srv_correctness.py` against the stock-LPT floor comes next, then (4) CUDA graphs, then the cohort A/B.
      **CUDA graphs work too (11:3x, `smoke_pdmux_mtp_graphs.sh`):** with graphs on in both servers MTP + pdmux served
      24 / 24, 0 tracebacks, decode on graphs (`cuda graph: True`); 18 / 24 identical to stock MTP over 128 tokens --
      five long prompts diverge at 48-99 tokens, one short at token 2 -- and acceptance length 3.07 vs 3.31 (-7%). Under
      greedy speculation the output is fixed by the target's verify, so divergence is target numerics (pdmux decodes on
      a different SM count, so flashinfer's split-KV partition differs); the acceptance drop is the open question.
      **Formal gate (`bash /dev/shm/w/gate_pdmux_mtp.sh`, srv_correctness, graphs on; the driver gained a PDMUX arm):
      MTP_PDMUX PASS by the rule** (long prompts agree on their first 8 tokens; 0 tracebacks, 0 fallbacks), but far
      from the floor: stock MTP repeat vs MTP is fully deterministic (min common prefix short 1 200 / long 96), MTP_PDMUX
      short 1 / long 11. Per request: every first token matches; 6 / 8 long prompts identical over 96 tokens (l1, l5
      diverge at 18 / 11); the 1 200-token background rows -- which decode on the decode SM group while pdmux resizes it
      as prefills come and go, changing the kernels' partition -- diverge at 1-723, one identical; several identical
      outputs show different acceptance, i.e. the draft saw differently rounded inputs. **Null case (`bash
      /dev/shm/w/gate_pdmux_nospec.sh`, stock LPT vs LPT + `--enable-pdmux`, no speculation, paths the patch does not
      touch): min common prefix short 1 / long 11 -- exactly MTP_PDMUX's.** The divergence is pdmux's own numerics (its
      decode runs on a resized SM group); the MTP integration adds none. **Gate cleared against the right floor.**
      **Payoff A/B, run 1 (11:33-11:58): MTP_PDMUX crashed at both C, two different bugs.** (a) OOM at C=8 and C=16
      (1.8 GiB MLP activation, 1.57 GiB free): pdmux prefills unchunked, and a whole 64K agent prompt does not fit
      beside the draft model at mem fraction 0.85; the arm now runs at 0.78 (KV pool ~1.6M tokens vs the cohort's
      <= 16 x 64K working set -- a handicap to our arm, not an advantage). (b) Rerun at 0.78: C=8 clean, C=16 **illegal
      address in the decode step** (47 running, 9 queued). Cause: the draft extend after a split prefill runs on the
      prefill stream while the decode step runs on the decode stream, and EAGLE had replaced the draft runner's pdmux
      prefill backend (private flashinfer workspace) with `draft_extend_attn_backend`, which shares the global
      workspace and eager kv-index buffers with the decode-stream backends. Fix (patch item 8): keep the pdmux
      prefill backend and run the prefill-side draft extend on it. **Output gate re-run with the fix: PASS, short 1 /
      long 11 -- unchanged, equal to the non-speculative pdmux null case** (`pdmux_mtp_fix8_20261017.sh`; the
      pre-fix C=8 run is discarded, since it ran with the race). PDMUX and EDFR50_F2048 from run 1 stand (0 tracebacks,
      0 fallbacks, FUSED_SPLIT_CALLS 67.5K / 68.5K); MTP_PDMUX rerun with the fix at C=8 / 16 (12:25-12:39): rc=0,
      0 tracebacks, 0 fallbacks, pdmux active.
      **Result: MTP + pdmux does not pay -- negative.** Agent / chat: tput, mean TTFT (s), mean ITL (P99) ms;
      chat tok/s/user p1/p50/p99 (`summarize_ab.py`, [summary](debug_kernel_server_perf/pdmux_mtp_20261017/pdmux_mtp_ab_20261017_summary.txt)):
      | C | arm | agent | chat | chat tok/s/user |
      |---|---|---|---|---|
      | 8 | PDMUX | 141 / 12.9 / 15.5 (37) | 1612 / 10.7 / 13.8 (33) | 36 / 68 / 132 |
      | 8 | MTP_PDMUX | 107 / 20.7 / 17.8 (99) | 1267 / 25.2 / 27.0 (116) | 16 / 40 / 113 |
      | 8 | EDFR50_F2048 (ours) | 139 / 14.8 / 14.3 (48) | 1576 / 5.8 / 25.1 (131) | 18 / 45 / 138 |
      | 16 | PDMUX | 142 / 29.0 / 16.9 (41) | 1610 / 25.3 / 13.7 (32) | 36 / 71 / 130 |
      | 16 | MTP_PDMUX | 106 / 48.0 / 18.1 (99) | 1213 / 55.0 / 28.2 (111) | 17 / 38 / 95 |
      | 16 | EDFR50_F2048 (ours) | 139 / 31.6 / 16.8 (73) | 1558 / 23.3 / 23.9 (127) | 16 / 48 / 126 |
      MTP_PDMUX vs PDMUX: throughput -24%, TTFT +61..+117%, chat ITL 2x. Acceptance is the same in both MTP arms
      (bench 2.12 tokens per verify for MTP_PDMUX and ours), so the smoke test's 3.07 vs 3.31 was small-sample noise,
      not a pdmux effect -- closed. The cost is the step: a 4-token verify plus 3 draft steps over 32-64K-context
      rows, on pdmux's shrunken decode SM group, takes ~4x a non-speculative decode step for 2.1x the tokens.
      Plain pdmux stays the strongest ITL baseline; ours keeps TTFT (chat -46% / -8% vs PDMUX) and ties throughput.
      - [x] **PD multiplexing + EDF (2026-10-17 14:32-15:42, 12 arms rc=0, 0 tracebacks / 0 fallbacks, EDF active in the
        EDF arms (EDF_RUNTIME, priority scheduling on), pdmux on in every pdmux arm; output gate queued):** pdmux keeps decode on its own SM group, which is where its
        ITL comes from; our EDF wins chat TTFT by deadline-ordered admission, which is independent of chunking.
        pdmux's batch builder goes through `get_new_batch_prefill`, so worker-side EDF (deadline order, no latency
        model, no chunk cap; server-assigned deadlines) applies unchanged. Arms PDMUX | PDMUX_EDF | stock MTP (the
        bar's baseline), C = 8 / 16, two repeats, same cohort / tree / session. `pdmux_edf_20261017.sh` (REP=1, 2).
        Two-repeat means (rep 1 ; rep 2 in [summary](debug_kernel_server_perf/pdmux_mtp_20261017/pdmux_edf_20261017_summary.txt)):
        tput / mean TTFT (s) / mean ITL (P99) ms, chat tok/s/user p1 / p50 / p99:
        | C | arm | agent | chat | chat tok/s/user |
        |---|---|---|---|---|
        | 8 | stock MTP | 110.5 / 18.69 / 21.3 (423) | 1285 / 23.19 / 30.5 (478) | 7 / 42 / 168 |
        | 8 | PDMUX | 140.9 / 12.91 / 15.4 (38) | 1613 / 10.67 / 13.7 (33) | 38 / 69 / 133 |
        | 8 | **PDMUX + EDF** | 143.0 / 12.97 / 15.2 (36) | 1611 / **3.80** / 14.0 (33) | 33 / 68 / 132 |
        | 16 | stock MTP | 107.7 / 43.14 / 27.2 (495) | 1223 / 51.36 / 31.1 (471) | 7 / 41 / 162 |
        | 16 | PDMUX | 142.5 / 28.87 / 17.0 (41) | 1609 / 25.06 / 13.7 (32) | 37 / 72 / 130 |
        | 16 | **PDMUX + EDF** | 142.2 / 29.37 / 16.1 (38) | 1610 / **17.45** / 14.0 (34) | 34 / 69 / 131 |
        Repeat spread <=2% on every mean (chat TTFT 1.3%). **PDMUX + EDF vs stock MTP: C=8 agent tput +29.4%, TTFT
        -30.6%, ITL -28.6%; chat tput +25.4%, TTFT -83.6%, ITL -54.2%. C=16 agent +32.1% / -31.9% / -40.8%; chat
        +31.6% / -66.0% / -54.9%.** 11 of 12 metrics clear the bar (>=30% TTFT and ITL, >=10% throughput); the miss is
        agent ITL at C=8 (-28.6%). EDF adds to pdmux exactly what it was meant to: chat TTFT -64% (C=8) / -30% (C=16)
        with ITL and throughput unchanged. Against stock *non-speculative* LPT (2026-10-17 baseline run, same cohort,
        other session) agent TTFT does not move (12.97 vs 13.20 s at C=8): cold 32-64K agent prefills are compute-
        bound and EDF serves chat's earlier deadlines first. Agent-ITL lever next: pdmux's decode SM share
        (`decode_bs_divisor` 36 -> 18 / 12, `pdmux_edf_div_20261017.sh`), after the output gate (`gate_pdmux_edf.sh`).
        sglang `29842f52f`+ (tree `up_pdmux_mtp`).
        - [x] **Output gate (2026-10-17 15:4x, `gate_pdmux_edf.sh`): PASS.** LPT_PDMUX_EDF vs LPT: min common prefix
          short 1 / long 11, identical to the null case LPT_PDMUX vs LPT (pdmux's own numerics: decode on a resized SM
          group); stock repeat floor short 163 / long 96; 0 tracebacks / 0 fallbacks, priority scheduling and pdmux on.
          EDF changes order, not numerics.
        - [x] **Decode SM share (2026-10-17 15:48-16:10, rep 1, all rc=0):** pdmux gives decode `decode_bs * 6 //
          decode_bs_divisor` of its 8 groups (default 36). vs stock MTP, agent tput / TTFT / ITL: C=8 divisor 36
          +29% / -31% / -29%, 18 +25% / -24% / -37%, 12 +22% / -20% / -39%; C=16 36 +32% / -32% / -41%, 18 +27% / -26%
          / -45%, 12 +28% / -26% / -48%. More decode SMs buy agent ITL with agent TTFT -- a trade-off curve, not a free
          lever; at C=8 no divisor clears both. Default 36 stays. Chat keeps -59..-84% TTFT and -55..-58% ITL
          throughout ([summary](debug_kernel_server_perf/pdmux_mtp_20261017/pdmux_edf_div_20261017_summary.txt),
          `pdmux_edf_div_20261017.sh`). Next: the two-tier TTFT SLO (agent prompts >= 16K get a flat SLO instead of a
          length-scaled deadline that queues them behind all chat), `pdmux_edf_t2_20261017.sh`.
        - [x] **Two-tier TTFT SLO (2026-10-17 16:10-16:44, rep 1, 6 arms rc=0, EDF + pdmux on, 0 tracebacks / 0
          fallbacks): negative.** `SGLANG_EDF_LONG_PROMPT_TOKENS=16384` with a flat 10 s / 5 s agent SLO, vs stock MTP:
          C=8 10 s (divisor 36) agent +28% / -30% / -28%, chat TTFT -80%; 10 s + divisor 18 agent -23% TTFT / -35% ITL;
          5 s + divisor 18 agent -25% / -32%, chat TTFT only -58%. C=16 10 s agent +31% / -32% / -39%. Agent TTFT
          does not move (13.09 vs 12.97 s at C=8) while chat TTFT degrades, so agent TTFT is not chat queueing: it is
          agent-vs-agent queueing on compute-bound cold 32-64K prefills. Length-scaled deadlines stay.
          ([summary](debug_kernel_server_perf/pdmux_mtp_20261017/pdmux_edf_t2_20261017_summary.txt))
        **Where the bar stands for PD multiplexing + EDF (default divisor, length-scaled SLO), vs stock MTP:** 11 of 12
        metrics clear at C=8 and C=16; the one miss is C=8 agent ITL (-28.6%, two repeats), and every knob that buys it
        (decode SM share, agent SLO tier) costs agent TTFT below -30% -- the C=8 agent pair sits on a trade-off frontier
        at about (-30%, -29%).
        - [ ] **TTFT deadline on uncached tokens (2026-10-17, `apply_edf_uncached.py`, `SGLANG_EDF_UNCACHED_DEADLINE=1`):**
          the length-scaled deadline used the full prompt, so a follow-up agent turn whose 32-64K context is already in
          the radix tree got a cold prompt's late deadline. The cached length is now read at enqueue by a read-only
          match and the deadline scales with the uncached remainder. First arm (C=8, divisor 36, rep 1, rc=0, EDF + pdmux
          on, 0 tracebacks / 0 fallbacks) vs stock MTP: agent tput +29.8%, **TTFT -34.1%** (12.31 vs 12.97 s with
          full-length deadlines), ITL -29.1%; chat +25.4% / -74.0% / -54.4%. The TTFT margin it buys is what the decode
          SM share can spend on agent ITL. **Divisor sweep (16:51-17:30, rep 1, 8 arms rc=0, 0 tracebacks / 0
          fallbacks, EDF + pdmux on)**, vs stock MTP, agent tput / TTFT / ITL ; chat:
          | C | divisor | agent | chat |
          |---|---|---|---|
          | 8 | 36 | +29.8% / -34.1% / -29.1% | +25.4% / -74.0% / -54.4% |
          | 8 | **27** | **+28.3% / -31.1% / -31.9%** | +25.5% / -71.1% / -55.3% |
          | 8 | 24 | +26.8% / -27.5% / -33.8% | +25.5% / -69.2% / -55.7% |
          | 8 | 18 | +24.3% / -23.8% / -36.2% | +25.5% / -66.1% / -57.3% |
          | 16 | 36 | +34.8% / -34.3% / -38.6% | +31.9% / -59.4% / -54.6% |
          | 16 | **27** | **+32.1% / -31.1% / -41.5%** | +31.6% / -55.8% / -56.2% |
          | 16 | 24 | +32.2% / -31.2% / -43.0% | +31.5% / -56.1% / -55.9% |
          | 16 | 18 | +30.2% / -28.6% / -45.6% | +31.9% / -52.6% / -56.5% |
          **Divisor 27, three repeats (17:35-17:58, all rc=0, 0 tracebacks / 0 fallbacks) + output gate PASS**
          (LPT_PDMUX_EDF with uncached deadlines and divisor 27 vs LPT: short 1 / long 11, identical to the pdmux null
          case; stock repeat floor short 132 / long 96). Means vs stock MTP (per-rep tput / TTFT s / ITL ms in
          [3-rep summary](debug_kernel_server_perf/pdmux_mtp_20261017/pdmux_unc27_3rep_summary.txt)):
          C=8 agent +27.8% / -30.1% / -31.6%, chat +25.4% / -70.8% / -55.3%; C=16 agent +32.7% / -32.2% / -40.2%, chat
          +31.6% / -57.1% / -54.7%; repeat spread <=3.4% (C=8), <=5.2% (C=16). **All 12 means clear the bar, but C=8
          agent TTFT by 0.1 point (per rep -31.1 / -29.4 / -29.7%) -- not robust.** Divisor 30 (between 36's -34.1 /
          -29.1 and 27's -30.1 / -31.6) for three repeats next, `pdmux_edf_unc2_20261017.sh` with UNC_D30.
        - [x] **Final single-GPU config -- PD multiplexing + EDF with uncached-token deadlines, decode_bs_divisor 30
          (2026-10-17 18:05-18:50; three repeats, all rc=0, 0 tracebacks / 0 fallbacks, EDF + pdmux on; output gate
          PASS -- LPT_PDMUX_EDF vs LPT short 1 / long 11, identical to the pdmux null case, stock repeat floor short 545
          / long 90).** Three-repeat means vs stock MTP (same cohort, ShareGPT 8 req/s, frozen 32-64K TraceLab):
          | C | agent tput / TTFT / ITL | chat tput / TTFT / ITL |
          |---|---|---|
          | 8 | **+28.9% / -31.7% / -31.1%** | +25.3% / -72.8% / -54.9% |
          | 16 | +32.7% / -32.3% / -40.4% | +31.6% / -57.3% / -54.7% |
          **All 12 means clear the bar (>=30% TTFT and ITL, >=10% throughput).** Per repeat, C=8 agent TTFT / ITL:
          -30.4 / -31.9, -33.0 / -31.9, -31.6 / -29.6 (5 of 6 values clear; rep 3 ITL is 0.4 point short); C=16 all
          clear in every repeat. Chat tok/s/user p1/p50/p99 34 / 69 / 132 (C=8) vs stock MTP 7 / 42 / 168; agent P99
          ITL 34-35 ms vs 417-498 ms. **Rep 4 (2026-10-17 22:45-22:57, rc=0, 0 tracebacks / 0 fallbacks): C=8 agent
          141 / 12.95 s / 14.6 ms (-30.7% / -31.5% vs stock MTP); four-repeat C=8 agent means -31.4% TTFT / -31.2% ITL,
          7 of 8 per-repeat values clear; C=16 clears in every repeat.** Per-rep tput / TTFT / ITL in
          [summary](debug_kernel_server_perf/pdmux_mtp_20261017/pdmux_unc30_3rep_summary.txt). Commands (pod
          `/dev/shm/w`, tree `up_pdmux_mtp` = `up_edf_floor` + `apply_pdmux_mtp.py` + `apply_edf_uncached.py`):
          `for r in 1 2 3; do REP=$r D27=0 D24=0 D30=1 bash pdmux_edf_unc2_20261017.sh; done`, `bash
          gate_pdmux_edf_unc30.sh`. **Attribution, stated plainly:** the SM split that buys ITL is sglang's PD
          multiplexing; ours is the scheduling on top -- deadline-ordered admission with uncached-token deadlines and
          the decode-share setting -- which turns pdmux's 9-of-12 into 12-of-12 against stock MTP. Per-request spread in
          [summary](debug_kernel_server_perf/pdmux_mtp_20261017/pdmux_edf_unc_20261017_summary.txt). Script:
      `pdmux_mtp_ab_20261017.sh` (frozen cohort, C = 8 / 16 + ShareGPT 8 req/s; PDMUX | MTP_PDMUX |
      EDFR50_F2048, one tree, one session, graphs on). Command: `bash /dev/shm/w/pdmux_conc_20261017.sh`
    (`tasks/debug_kernel_server_perf/baselines_20261017/pdmux_conc_20261017.sh`).
  - [ ] **semi-PD -- blocked on a user decision (2026-10-13):** fork of sglang v0.4.4.post1
    (torch 2.5 / cu124), MPS-based SM partition, radix cache off, no speculative decoding. Venv
    `venv_semipd` + clone are on the pod (torch 2.5.1+cu124 installed), but its pinned
    `sgl-kernel==0.0.5` no longer exists on PyPI (every 0.0.x release deleted), so it must be built
    from the repo's `sgl-kernel/`, which needs a CUDA 12 nvcc (pod: 13.1). Installing NVIDIA's
    CUDA 12.4 toolkit runfile was denied by the permission policy; needs the user's go-ahead (or
    a CUDA 12 toolchain provided another way).
    **2026-10-20 retry on H200 `dadf94f0e1da7238` (`semipd_provision_20261020.sh`, all under /opt/tiger/semipd):**
    fork cloned, torch 2.5.1+cu124 venv, semi-pd-ipc built; CUDA 12.4 headers / libs assembled from NVIDIA's pip wheels
    plus CCCL 2.3.2. sgl-kernel 0.0.5 still cannot build: the `nvidia-cuda-nvcc-cu12` wheel ships ptxas but no `nvcc`
    driver (the earlier `nvcc --version` was the system CUDA 13.1 one), and torch refuses a 13.x nvcc for a cu124 build.
    A CUDA 12 nvcc needs a toolkit install (runfile or conda) -- the step that needs the user's go-ahead. Still blocked.
  - [x] **Dynamo conditional disaggregation (decided 2026-10-24):** Dynamo documents conditional disaggregation for
    vLLM / TRT-LLM only, so stock sglang PD is the headline multi-node baseline. Our emulation (`router.py
    cond_disagg`) is measured in `pd_colocation_vs_disagg.md` as a labelled policy comparison.

### [~] Supporting gates

- [x] **RoPE root cause (2026-10-12):** the YaRN override dropped `rope_theta` (1e6 -> 1e4) under
  transformers 5.x; fixed in sglang's config loader with a regression test. All serving numbers
  before 2026-10-12 are void. ([evidence](debug_kernel_server_perf/resume_20261012/a2_serving_ropefix_20261012.md))
- [x] **Serving-output gate (q4 and q8; graphs off and on):** every 4-mode arm agrees with stock on
  the long prompts; first tokens are the HF reference argmax or within one bf16 ulp.
- [x] **Graphs-on 4-mode crash (2026-10-12; cause found and fixed 2026-10-13):** both 4-mode arms crash under
  load (async index assert). Bisect: a device sync after mixed steps (or launch-blocking) cures it;
  phase syncs, no in-step graph, synchronous flashinfer plans and a sync after non-mixed steps do
  not. **Cause (code, 2026-10-13):** in the overlap loop the verify half's
  `future_map.publish(new_seq_lens)` and relay stash run after the forward-stream block, i.e. on
  the schedule stream, unordered against the forward stream that is still producing those
  tensors; stale seq_lens get published and the next step resolves them into out-of-range
  indices. The mixed-step sync orders exactly this. Fix: issue both on the forward stream
  ([patch](debug_kernel_server_perf/resume_20261012/racefix_scheduler.patch), tree `up_racefix`).
  **Retracted:** "overlap off does not cure" -- that arm died at its FIRST mixed step on
  `assert next_token_ids.is_cpu` (the synchronous path had no 4-mode handoff at all; now added),
  so it never tested the race. Validation (`graphs_racefix_20261013.sh`), 2026-10-13:
  - [x] **null case:** unfixed tree, no barrier, MTPMIX_LPT chat 24/s + agent 0.1/s -> crashed
    (`Assertion -sizes[i] <= index && index < sizes[i]`, 1 traceback), so the race reproduces.
  - [x] **output gate, fixed tree, graphs on, no barrier:** PASS (long prompts agree on >= 96
    tokens with stock `MTP_LPT_Q4G`).
  - [x] **fixed tree, no barrier (05:02-05:23):** MTPMIX_LPT 1651 / 1701 mixed steps, MTPMIXFUSED
    1751 / 1751, 0 tracebacks each, CUDA graphs on; synchronous path (`--disable-overlap-schedule`)
    1651 mixed steps, 0 tracebacks. **Verdict PASS.** From here every 4-mode arm runs on
    `up_racefix` without the mixed-step barrier (`tree_env.sh`); S1 x S2 numbers above were taken
    WITH the barrier, which can only have slowed the 4-mode arms.
    Tree: sglang-up `cc739bd2d` + `racefix_scheduler.patch` + `racefix_fs_upload.patch`
    (+ `steptrace_port.patch`, off unless `SGLANG_STEP_TRACE=1`); command
    `bash graphs_racefix_20261013.sh` then `bash racefix_verdict.sh` (pod `/dev/shm/w`). Until it passes, measurements keep the mixed-step barrier. The latent race
  of the same class -- `_fs_upload` rewrote its reused pinned host buffer while the previous step's
  non-blocking copy out of it could still be queued -- now waits on a per-buffer copy event first
  ([patch](debug_kernel_server_perf/resume_20261012/racefix_fs_upload.patch), same tree and validation).
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

Papers and systems this work draws on. *(sent)* marks the ones the user sent. Each entry: summary, relation to our
method, and what we take from it -- only where it strengthens our own contributions (fused kernels, higher
cluster-wide request volume, scheduling flexibility under load spikes / mixed workloads / workload variation).

- **DeFT: Decoding with Flash Tree-attention for Efficient Tree-structured LLM Inference** *(sent, 2026-10-28;
  arXiv 2404.00242)*
  - **Summary:** Attention kernel for tree-structured decoding -- few-shot prompting, multi-step reasoning,
    speculative decoding -- where many generation calls share a prefix. Evaluated on three tree-based workloads
    against state-of-the-art attention kernels; reports 73-99% less KV-cache IO and nearly 100% less IO for partial
    results, giving up to 2.23x end-to-end and 3.59x attention speedup.
  - **Core method / contributions:** (1) *KV-Guided Grouping*: partition work by KV rather than by query, so a
    shared prefix's KV is loaded once and reused across every query that shares it, instead of being re-read per
    branch. (2) *Flattened Tree KV Splitting*: split the tree's KV evenly across partitions with little redundant
    computation, so the splits are load-balanced and GPU utilisation stays high.
  - **Relation to our method:** DeFT's grouping is the same insight as our proposed cross-role shared-prefix reuse,
    and it confirms the mechanism is worth real money (73-99% IO). But its setting is a TREE WITHIN one generation
    call (speculative tree, beam, reasoning branches), so the queries sharing a prefix are all decode queries of the
    same request. Ours is different in two ways that only exist because of C2: the sharing is **across the two
    roles of one fused step** (the prefill chunk and the decode/verify rows read the same session prefix) and
    **across different requests** (fan-out siblings of one agent session). A tree-attention kernel cannot express
    that, because the chunk is not a branch of the decode tree. Their load-balanced splitting is also the same
    problem our LPT placement solves, and our measurement says placement is worth <= 0.7% on our shapes -- their
    gain is in the IO term, not the balance term.
  - **What we can learn:** group by KV, not by query, when the plan shows two work items reading the same pages --
    and measure the page-overlap fraction FIRST, because their 73-99% comes from trees where sharing is near-total,
    whereas our sharing depends on the workload's fan-out and cache hit. Do not adopt their tree machinery: our
    contribution has to be the cross-role grouping the fused step makes possible, not tree attention.
- **Taming Dynamism on GPUs: Cross-SM Kernel Fusion via SM Cooperation and Just-in-Time Reduction** *(sent; PDF)*
  - *Summary:* fuses dynamically sized, heterogeneous work into one persistent launch whose CTAs pull tasks from a
    device-side queue, with SMs cooperating on shared tasks and split partials reduced just in time instead of by a
    separate merge kernel; evaluated on kernel-level dynamic workloads against per-task launches.
  - *Core method/contributions:* MeldKernel ("Meld"), a compiler framework for cross-SM kernel fusion of dynamically
    shaped work: SM cooperation over on-chip SM-to-SM interconnects (clusters) splits phase-level dependencies into
    sub-task-level ones, so a kernel sequence becomes one cross-SM pipeline; just-in-time reduction lands remote SMs'
    partial states in local shared memory by asynchronous transfer before the reduction needs them; a decoupled
    execution model maps the declared computation onto SMs at runtime from input shapes. 1.3x average on decode
    attention over the best existing solution (from the abstract; the ACM full text was not accessible).
  - *Relation:* the template for our four-task plan (atomic cluster cursor, cross-role backfill) and the JIT
    reduction of split-KV partials we built (<= 0.6% of GPU time, then replaced by flashinfer 0.7's scheduled merge).
    We did not co-compile roles with different register needs into one body: our FA2 role spilled 21x that way, so
    FA2 runs as a PDL-co-launched executor and backfill is done by the hardware scheduler.
  - *Learn:* its framing -- dynamism inside one launch is the problem, cooperation across SMs the answer -- is the
    story for our kernel contribution, applied to the 4-mode serving step (prefill chunk + verify rows), which a
    kernel-only benchmark does not exercise; report the same kernel-level evidence (makespan vs per-mode launches,
    SM idle time) on captured serving steps.
- **SLOs-Serve: Optimized Serving of Multi-SLO LLMs** -- S. Chen, Z. Jia, S. Khan, A. Krishnamurthy, P. B. Gibbons,
  [arXiv:2504.08784](https://arxiv.org/abs/2504.08784) *(sent)*
  - *Summary:* application- and stage-specific SLOs; a chunked-prefill planner (dynamic programming over token
    allocations) that co-plans speculative decoding and routes across replicas; evaluated on mixtures
    (summarization, coding, chat, tool calling, reasoning) crossed with load, reporting capacity under SLOs.
  - *Core method/contributions:* a multi-SLO dynamic-programming planner that continuously optimizes token
    allocations under application- and stage-specific SLOs, searching chunked-prefill sizes jointly with (optional)
    speculative decoding; on top of it, multi-replica serving with dynamic request routing and resilience to bursty
    arrivals. 2.2x average per-GPU serving capacity over prior systems.
  - *Relation:* its planner co-scheduling prefill chunks with speculation is the closest scheduling analogue of our
    4-mode step; it plans token budgets, we change what one step computes (fused FA3 + FA2 in one forward).
  - *Learn:* its evaluation -- per-tenant SLOs, mixture x rate sweeps, capacity at a fixed attainment -- is the
    frame in which our scheduling flexibility under mixed workloads shows; it is the source of S1-S3's setup.
- **Towards High-Goodput LLM Serving with Prefill-decode Multiplexing (MuxWise)** -- Y. Chen, W. Cui et al.,
  [arXiv:2504.14489](https://arxiv.org/abs/2504.14489) *(sent; "the ASPLOS PD multiplexing paper")*
  - *Summary:* intra-GPU prefill/decode multiplexing (SM partitions via green contexts, layer-split prefill) with an
    SLO-aware dispatcher and a contention-tolerant latency estimator; goodput under SLO as the metric.
  - *Core method/contributions:* intra-GPU prefill-decode multiplexing -- the two phases share a GPU but execute
    independently, with compute allocation adapted dynamically and decoupled from memory management; three
    components: a bubble-less multiplex engine, a contention-tolerant estimator of co-located latency, and an
    SLO-aware dispatcher. 2.20x average (up to 3.06x) peak throughput under SLO over state-of-the-art baselines.
  - *Relation:* direct colocation baseline (= sglang `--enable-pdmux`, run 2026-10-17); our split is decided per
    step inside one forward and co-runs prefill with MTP verify rows, where pdmux partitions SMs at model level.
  - *Learn:* its estimator is the comparison point for our EDF step model (S4: mixed-step MAPE 7.8-8.5% on the
    4-mode schedule vs 44-50% on stock MTP) -- predictability of the fused step is our argument, not theirs.
- **Crossflow: Prefill–Decode Elasticity for Agentic LLM Serving** -- Y. Xu et al.,
  [arXiv:2609.27085v1](https://arxiv.org/abs/2609.27085v1) *(sent)*
  - *Summary:* fixed P/D roles with decode-local prefill admitted through revocable multi-resource leases,
    cache/shape-aware path comparison and local SLO revalidation; no new kernels. Replays one-hour TraceLab segments
    with timestamps compressed by a load factor alpha (8-20x; request rate as the load axis), six TP4 GB300 servers
    (2P4D), SGLang, no stated evaluation SLO; its GLM-5.2 TraceLab mean ITL worsens 20.9-34.3%.
  - *Core method/contributions:* elasticity on fixed P/D roles: decode workers advertise expiring, versioned leases
    for compute, KV, projected output, transfers and spill slots; the router tracks uncached-token queue pressure,
    compares ordinary P-to-D service with decode-local prefill per request, reserves capacity atomically and caps
    cluster-wide borrowing; workers revalidate before each bounded prefill unit, and conversations keep their decode
    home. 17.4% / 16.2% geometric-mean input / output throughput over static P/D.
  - *Relation:* direct cluster-policy baseline; it keeps fixed roles and decides per request where to prefill, we
    keep symmetric nodes and make local prefill + speculative decode cheap inside one step.
    [Analysis and comparison](debug_kernel_server_perf/resume_20261012/crossflow_analysis_20261014.md).
  - *Learn:* its ITL regression under decode-local prefill is exactly the cost our fused step removes -- report
    ITL alongside TTFT on the same replay; its timestamp-compression protocol is the request-rate load axis for
    chat-like tenants (agent tenants use program concurrency, below). **It also confirms our claim that P/D demands can vary largely during online serving**
- **ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System** -- H. Kang et al.,
  [arXiv:2602.13692](https://arxiv.org/abs/2602.13692) *(sent)*
  - *Summary:* program-aware scheduling for agentic workflows; shows KV-cache thrashing (a session's cache evicted
    during a tool call and re-prefilled, up to 7.14x end-to-end); load axis is the number of parallel programs kept
    active (headline 96 programs on 8x H100).
  - *Core method/contributions:* models agentic workflows as *LLM Programs*, one view of KV caches, system state and
    tool assets (disk, network ports); a program-aware scheduler that maximizes KV-cache hit rate and mitigates memory
    imbalance, and a tool resource manager that prepares tool environments asynchronously. 1.5-3.6x serving and
    1.8-3.9x RL-rollout throughput, up to 4.2x disk-memory savings over state-of-the-art inference systems.
  - *Relation:* motivates PD point 2 -- best-effort migration leaves replicas where demand is, where disaggregation
    pins a session's cache to its prefill node.
  - *Learn:* **load protocol we follow** -- agent load is program concurrency C, never sessions/s; every agent arm
    reports C and measured mean in-flight requests. Its thrashing measurement is the evidence for our cluster-wide
    request-volume argument (cache replication raises sustainable volume).
- **TraceLab: Characterizing Coding Agent Workloads for LLM Serving** -- K. Zhu, M. Jacob, C. Ma, Y. Pan, S. Wang,
  A. Krishnamurthy, B. Kasikci, [arXiv:2606.30560](https://arxiv.org/abs/2606.30560); dataset
  `syfi_coding_trace.jsonl.gz` ([github](https://github.com/uw-syfi/TraceLab)) *(sent)*
  - *Summary:* characterization of Claude Code / Codex agent traces: context p50 ~124K, a quarter of Claude rounds
    >= 256K, short outputs, high prefix hit, long inter-round gaps.
  - *Core method/contributions:* a characterization, not a serving mechanism: an open trace of ~4,300 Claude Code and
    Codex sessions (~350K LLM steps, ~430K tool calls) from the authors' daily use, with the collection pipeline and
    analysis code; findings (long autonomous loops, long contexts with short outputs, heavy-tailed tool calls, high but
    imperfect prefix-cache hits) and their serving implications: lower-overhead tool calling, append-length-aware
    prefill, semantic-aware tool-latency prediction, KV-cache management around human-paced gaps.
  - *Relation:* the coding-agent tenant in every mixed run (causal replay of prefix / append / output tokens).
  - *Learn:* its long-context, append-heavy rounds are where prefill and speculative decode collide -- the regime
    our 4-mode step targets; report results by uncached-prompt-length bucket, as its statistics suggest.
- **JITServe: SLO-aware LLM Serving with Imprecise Request Information** -- W. Zhang, Z. Wu, Y. Mu, R. Ning, B. Liu,
  N. Sarda, M. Lee, F. Lai, NSDI '26 ([PDF](https://www.usenix.org/system/files/nsdi26-zhang-wei.pdf),
  [arXiv:2504.20068](https://arxiv.org/abs/2504.20068)) *(sent)*
  - *Summary:* maximizes *service goodput* (tokens that meet their request's SLO) across three request types:
    latency-sensitive (TTFT ~2 s, TBT ~100 ms, set from P95s of 1K DeepSeek API calls), deadline-sensitive (20 s
    end-to-end), compound (20 s x number of stages, sub-deadlines from matched historical pattern graphs). Grouped
    margin goodput maximization gives each request just enough bandwidth to meet its SLO just in time (upper-bound
    output length from a quantile regression forest, refined every ~50 tokens), picks batches by a length-sorted
    sliding window, preempts only when the projected gain exceeds the stall cost; ~2.8K LoC on vLLM with chunked
    prefill. Llama-3.1-8B/70B, Qwen2.5-14B, Qwen3-30B-A3B on 16x A100; chat, long-context math, deep research,
    agentic code generation, Microsoft-trace arrivals. 1.4-6.3x goodput (28.5-83.2% resource savings) vs Autellix,
    learning-to-rank, vLLM FCFS, Sarathi-Serve; within 3-9% of an oracle. SLOs are fixed per type: they do not scale
    with prompt length or with isolated latency.
  - *Core method/contributions:* scheduling with imprecise request information -- conservative estimates of output
    length and inter-call dependencies, relaxed as generation refines them; grouped margin goodput maximization, which
    gives each request just enough bandwidth to meet its SLO just in time and leaves the residual capacity to others;
    batch composition chosen for efficiency and goodput with provable guarantees.
  - *Relation:* a request-level scheduler -- it decides *who* runs when, on unchanged kernels; it does not change
    what one step costs. Our EDF is the same family (deadline-driven admission), but our contribution is that the
    fused 4-mode step makes the step cheaper and predictable (S4), and that colocation lets the cluster route a
    request anywhere. Its batch-composition finding (mixing input lengths slows a batch) is the inefficiency our
    fused kernel attacks inside the step rather than by keeping lengths apart.
  - *Learn (evaluation, not method):* (1) report **service goodput by request type** next to means -- chat as
    latency-sensitive (token i on time if done by TTFT_SLO + i x TBT_SLO), agent rounds as deadline-sensitive or
    compound (a program's rounds share one deadline); this is the metric under which scheduling flexibility under
    mixed load is visible. (2) Set SLOs from an external reference (their P95-of-API method) or relative to the
    unloaded latency of the same request on stock MTP, and sweep a common SLO scale (0.8x-1.5x) -- our relative-SLO
    item. (3) Its fixed per-type SLOs ignore prompt length; for TraceLab, whose rounds span 32K to > 256K, a tiered
    TTFT SLO (e.g. 2x for > 200K uncached tokens, as the user proposed; TokenScale also tiers by length) avoids
    scoring a 256K prefill against a chat deadline. The scheduler's internal deadlines (length-scaled EDF) and the
    evaluation SLOs are separate choices and should be reported separately.
- **semi-PD: Towards Efficient LLM Serving via Phase-Wise Disaggregated Computation and Unified Storage** -- K. Hong
  et al., [arXiv:2504.19867](https://arxiv.org/abs/2504.19867) ([code](https://github.com/infinigence/Semi-PD))
  - *Summary:* splits each GPU's SMs between a prefill and a decode process (CUDA MPS) over one shared KV store: no KV
    transfer, no weight replication. MPS cannot resize a running process, so a ratio change starts new workers on the
    weights and KV a resident process holds and retires each old worker after its current iteration; an SLO-aware
    algorithm refits TTFT / TPOT models every window of iterations and steps the ratio toward the phase missing its SLO.
    Paper's results: 1.27-2.58x lower mean end-to-end latency (DeepSeek), 1.55-1.72x more requests within latency
    constraints (Llama).
  - *Core method/contributions:* the claim that disaggregation's benefit comes from disaggregated computation, not
    storage; a computation resource controller partitioning SMs between prefill and decode, a unified memory manager
    serving both phases' asynchronous access to one KV store, a low-overhead resource adjustment mechanism between
    the phases, and an SLO-aware dynamic partitioning algorithm that tunes the split for SLO attainment.
  - *Relation:* closest intra-GPU design; its ratio does adapt, but between process generations every window of
    iterations, while ours decides the split per step inside one forward, co-runs prefill chunks with MTP verify rows,
    and lets the hardware backfill FA2 work onto SMs FA3 vacates. Its code is a separate sglang v0.4.4.post1 fork with
    radix cache off and no speculative decoding, so it cannot serve our MTP arms as released; it has not been run (build
    blocked, see Direct baselines).
  - *Learn:* the comparison for scheduling flexibility under workload variation is reaction granularity, not static vs
    dynamic -- show our per-step split tracking a step change in the prefill/decode mix (load spike or tenant change)
    and, once it runs, how many iterations semi-PD's window takes to follow the same change.
- **NVIDIA Dynamo -- KV router, conditional disaggregation, planner, KVBM**
  ([conditional disagg](https://docs.nvidia.com/dynamo/advanced-customizations/conditional-disaggregation);
  [router](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/routing-concepts);
  [planner](https://docs.nvidia.com/dynamo/v1.5.0/knowledge-base/modular-components/planner/overview);
  [delta-KV RFC](https://github.com/ai-dynamo/dynamo/discussions/7861))
  - *Summary:* KV-aware block-count cost router with no SLO term; prefills locally on the decode worker when the
    uncached prompt is small (default: effective ISL under 2048 tokens and under 0.7 of the prompt) or remote prefill is
    loaded; a P/D replica planner (180 s throughput mode, 5 s load-based mode for bursts); KVBM tiered KV store; an open
    RFC (a proposal with no replies, not shipped) would cut remote rounds to append-only transfer.
  - *Core method/contributions:* a KV-aware router that scores workers by uncached prefill blocks (minus cache-hit
    credits) plus projected decode blocks, with weighted deficit round robin across policy classes; conditional
    disaggregation (`isl_bounding` / `prefill_load` / `isl_or_load` on effective ISL after the decode worker's cached
    prefix; vLLM and TensorRT-LLM only); a planner sizing P and D replicas separately from load prediction and
    TTFT/ITL targets (180 s throughput mode, 5 s load mode); KVBM, G1-G4 KV tiers moved over NIXL. The delta-KV RFC is
    an open proposal (Kuaishou, 2026-04), not shipped.
  - *Relation:* closest multi-node baseline for agentic rounds (`pd_colocation_vs_disagg.md` point 6): most agent
    append rounds already avoid the KV transfer that PD point 5 measures. What stays ours: making local prefill +
    speculative decode co-execution efficient (4-mode step + fused kernel), symmetric nodes instead of fixed pools,
    and EDF/SLO-aware admission. Must be run as a baseline, not argued around.
  - *Learn:* its fixed local-prefill thresholds and a planner that reacts by changing worker counts (5 s decisions in
    load-based mode, 180 s in throughput mode) are the comparison for scheduling flexibility under load spikes --
    evaluate a step change in load and report the time to recover TTFT / ITL, where our per-request routing and
    per-step prefill/decode mix react without adding or removing workers.
- **FlowPrefill: Decoupling Preemption from Prefill Scheduling Granularity to Mitigate Head-of-Line Blocking in LLM
  Serving** -- C. Hsieh, Z. Zong, X. Chen, J. Li, J. Zhai, L. Wen, [arXiv:2602.16603](https://arxiv.org/abs/2602.16603)
  ([code](https://github.com/HSIEHCHIACHI/FlowPrefill)) *(sent)*
  - *Summary:* a prefill scheduler for PD-disaggregated serving, built for TTFT goodput, on vLLM 0.11.2; 8x A800-80GB;
    Llama3-8B / Qwen2.5-14B / Llama3-70B (TP 1/2/4) plus Qwen3-30B-A3B; QwenTrace production trace (four task types,
    per-type TTFT SLOs, text strictest, search/file loosest); load axis is a request-rate scale; goodput = max rate at
    90% TTFT attainment. Baselines are DistServe 1P1D (FCFS) and DistServe + chunked prefill + EDF (2K / 8K chunks),
    plus vLLM with 2K chunks for colocation. 4.7-5.6x goodput over DistServe, up to 2.0x / 4.5x over the 2K / 8K-chunk
    variants, 1.5-3.1x tighter SLOs met; colocated (intra-PD) variant: TBT attainment up to 1.6x.
  - *Core method/contributions:* operator-level preemption: each request runs in its own thread and CUDA stream, and
    a preemption flag is checked at operator boundaries (qkv_proj, attn, o_proj, gate_up_proj, ...), so preemption
    latency is bounded by one operator without paying for small chunks; event-driven scheduling that runs only on
    arrival or completion; slack-aware EDF (earliest deadline among requests with non-negative slack, so requests
    that can no longer meet their deadline are deprioritized) plus SLO-aware batching of compatible prefills.
  - *Relation:* the same problem as our chunk preemption + EDF (long prefills blocking tighter-SLO requests) and the
    same policy family (its negative-slack deprioritization corresponds to our demote-on-miss). It preempts on a
    prefill-only instance, so a preempted prefill never shares a step with decode; ours preempts at chunk boundaries
    inside a mixed 4-mode step where the prefill chunk co-runs with MTP verify rows, and the fused kernel makes small
    deadline-sized chunks cheap instead of avoiding chunking.
  - *Learn:* its premise -- chunk size trades execution overhead against blocking -- is the claim our kernel has to
    refute on the mixed step: report step cost vs chunk size (fused vs stock chunked prefill), then TTFT goodput at
    90% attainment under per-type TTFT SLOs on a production-like trace with a rate-scale axis, plus the tightest SLO
    scale each system meets; report TBT attainment beside TTFT, as its colocated result does.
- **POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference** -- A. K. Kamath, R. Prabhu,
  J. Mohan, S. Peter, R. Ramjee, A. Panwar, ASPLOS '25 ([arXiv:2410.18038](https://arxiv.org/abs/2410.18038)) *(sent)*
  - *Summary:* one attention kernel for hybrid (prefill chunk + decode) batches that runs prefill and decode work
    concurrently on the same SM; built from FlashAttention-2 kernels on A100 (108 SMs) and integrated into
    Sarathi-Serve; Yi-6B on 1 A100, Llama-2-7B / Llama-3-8B on 2 A100s (TP2). Kernel baselines: serial FA, FA on two
    streams, HFuse horizontal fusion, FlashInfer batched; end-to-end: offline requests/min and online TTFT / TBT /
    request latency at fixed QPS on an internal enterprise workload and arXiv-Summarization (4K-32K contexts).
    Attention up to 59% faster (mean 28%), throughput up to 22% over Sarathi-Serve, P99 TTFT up to 4.3x lower.
  - *Core method/contributions:* SM-aware CTA scheduling: each CTA reads its SM id and an atomic per-SM counter at
    runtime to pick prefill or decode work, so both phases share every SM; a 16-row decode query tile (the A100
    CUTLASS minimum) that leaves tensor cores to prefill; 2 or 4 CTAs per SM chosen by whether the batch is
    prefill-dominant; virtual decode CTAs (one warp each) to balance shared memory. FA-3 / Hopper is left to future
    work.
  - *Relation:* the closest kernel work -- the same goal of overlapping compute-bound prefill with memory-bound decode
    in one mixed step, and its decode tile matches our 16-row FA2 tile class. But it supports only FA2 (Ampere-style
    mma), is slow on other architectures, and we confirmed that MMA instructions of different sizes cannot be
    overlapped within one kernel on Hopper and Blackwell (wgmma / tcgen05). That is why our design keeps FA3 (prefill
    rows) and FA2 (decode / verify rows) as separate launches co-running via PDL and puts the fusion in the one CPU
    plan and the unified in-launch merge. Its hybrid batch has no speculative verify rows.
  - *Learn:* its kernel evidence -- a grid of hybrid batches labelled memory-bound / balanced / compute-bound,
    speedup over serial, two-stream and horizontally fused execution, and distance to the ideal overlap -- is the
    format for our kernel claim on Hopper, where it cannot run; cite it as the Ampere design point our plan + merge
    generalizes. Its "% of requests with > 200 / 500 ms stalls" is a cheap tail metric for our TPOT claims.
- **SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills** -- A. Agrawal, A. Panwar,
  J. Mohan, N. Kwatra, B. S. Gulavani, R. Ramjee, [arXiv:2308.16369](https://arxiv.org/abs/2308.16369) *(sent)*
  - *Summary:* the original chunked-prefill paper: split each prompt into equal chunks and fill every batch with one
    chunk plus decodes. nanoGPT + xformers attention; LLaMA-13B on 1 A6000 and LLaMA-33B on 1 A100-80GB (deployed),
    GPT-3 with 8-way PP x 8-way TP on 64 simulated A100s; offline throughput across sequence length, batch size and
    P:D ratio, against separate prefill/decode iterations and Orca-style iteration-level batching. Decode throughput
    up to 10x (13B) / 4.25x (33B), end-to-end 1.33x / 1.25x; GPT-3 pipeline bubbles 6.29x fewer, 1.91x end-to-end.
  - *Core method/contributions:* chunked prefills (per-chunk causal masks; chunk size picked per model-hardware by
    profiling, with tile-quantization jumps when the token count just exceeds a tile multiple); decode-maximal
    batching: one prefill chunk plus up to B-1 decodes, with all linear layers fused into one matmul so decodes reuse
    the weights the chunk loads (per-token decode 12.49 ms -> 1.2 ms in its example); uniform-compute micro-batches
    that cut pipeline bubbles.
  - *Relation:* the chunked-prefill baseline whose mixed step (decode rows piggybacked on a prefill chunk) our 4-mode
    step and tile classes improve on. SARATHI fuses only the linear layers -- prefill and decode attention stay
    separate calls, "the attention cost remains unchanged"; stock sglang's mixed chunk runs both in one varlen
    attention call with one tile shape. Our step adds MTP verify rows, routes prefill rows to FA3 and decode / verify
    rows to the 16 / 32-row FA2 tile classes, and reduces all splits in one merge.
  - *Learn:* its analysis isolates exactly the cost piggybacking cannot hide -- attention -- and makes the P:D ratio
    and chunk size the knobs; report our attention and step time against stock chunked prefill on the same axes (P:D
    ratio x chunk size, including non-multiples of the tile), so the attention share SARATHI leaves is the part our
    kernel is shown to remove.
