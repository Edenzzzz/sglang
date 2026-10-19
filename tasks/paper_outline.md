# Current goal

Rebuild the story of the 11/21 draft (`tasks/AttentionFuse Outline OSDI 25'.docx`) around what has been measured
since. The draft section order is kept; each item is either backed by a dated result (with the doc that holds the
command and commit) or marked as missing. Priority order of the remaining gaps:

1. [ ] 2026-10-19 Single-node per-tenant rerun of the standard mix (H200 `dadf94f0e1da7238`). REOPENED: stock is
   bimodal (rep 3 matches 10-18, reps 1-2 are a slow mode); vs rep 3 ours without MTP misses agent TTFT (+17%).
   Repeats 4-6 queued. Reps 1-2 read: no MTP vs stock clears all five means (+44% / -35% / -32%, chat -94% / -45%); MTP vs stock MTP misses only chat ITL
   (-28.1%); chat SLO attainment 1.3x / 1.8-1.9x stock, agent goodput 2.3-2.8x / ~16x. Open: stock is slower on
   this pod than on 10-18 while ours matches -- unexplained (`realistic_workload_analysis.md` Baselines).
1b. [ ] SLO-sized chunk admission (user, 2026-10-19): admit the largest chunk that keeps the running decodes within
   their ITL SLO, with no exception (no late-prefill override, no minimum chunk). The tested step budget was a fixed 50 ms floored at
   the 1024 small chunk, so it never sized to the SLO. Target: the chat TPOT tail (p90 61-63 ms no MTP).
2. [ ] 5e: mid-run mixture shift / load spike, measured through the transition (`pd_colocation_vs_disagg.md` 5e).
   This is the flexibility claim's main evidence and it has not been run.
3. [x] 2026-10-19 Point 4 as proposed (pull-source choice among per-worker copies, serving node fixed): neutral
   among sources (within 3%), copies cost 7-10% goodput; no congested source at <= 16 H20s even on one NIC.
4. [ ] Mixed-chunk tile waste / padding measured against stock (the draft's "up to 40%" in 3.4 is not backed yet).
5. [ ] Second model (larger dense or MoE). Every result so far is Qwen2.5-7B-Instruct + EAGLE3.
6. [ ] semi-PD baseline (sglang v0.4.4.post1 fork); fault tolerance stays optional.

## Story in one paragraph (2026-10-19)

Disaggregation buys decode isolation by fixing a prefill / decode split in hardware. Colocation can have the same
isolation per step, without fixing the split, if one step can run prefill, decode, draft and verify attention side
by side at a chosen SM share. That requires (C1) a kernel path that runs FA3 prefill tiles and FA2 decode / verify tiles
on pre-assigned SMs inside the normal sglang step: CPU-GPU overlap scheduling stays on and there is one CUDA-graph set;
pdmux has neither. It also requires (C2) a schedule that spends the step on deadlines: EDF on uncached-token deadlines,
chunk size admitted per step. With speculative decoding on (MTP is now standard, and disagg runs it on decode nodes)
(C3) the fused step is predictable enough to route on. Cluster-wide (C4), symmetric nodes let the router send each
request to its cache holder first and move only when the SLO check says so. Prefixes then get replicated where demand
is, rather than diluted across a fixed prefill pool. The cluster serves more requests within SLO, and the schedule
re-balances per step instead of re-provisioning.

## Core contributions and their evidence

- **C1 Fused FA3 + FA2 step with SM pre-assignment (kernel)**
  - [x] 2026-10-17 FA3 role runs flash-attention's Hopper kernel (paged TMA, NDEBUG) at parity with sgl-kernel FA3;
    FA2 runs decode / verify rows. `debug_kernel_server_perf_4_mode.md` "Where it stands".
  - [x] 2026-10-18 FA2 overlapped with FA3 on a side stream, no MTP, H200: +10% agent throughput and -12% / -16% agent /
    chat ITL over the sequential split. `realistic_workload_analysis.md` Baselines.
  - [x] 2026-10-17 kernel-level bound, stated honestly: co-running gives -18% on balanced steps but is ~1.2% of serving
    GPU time on TraceLab. Multicast, q-tile choice, split reduction and in-kernel backfill were closed by measurement (A5).
  - [x] 2026-10-19 the mixed step is attention-FLOP-bound: FA3's prefill-chunk call runs at ~540 TFLOP/s, and forced
    split-KV / pack-GQA give nothing (`perf_model/fa3_chunk_splits.py`).
  - Paper framing: the kernel's value in serving is that it makes the 4-mode step possible inside the stock scheduler
    (overlap scheduler on, one graph set, no padded decode rows in prefill tiles). Its standalone speedup is secondary.
    This replaces the draft's persistent-FA2 / POD kernel and removes its Limitation "SM80 MMA cannot reach FA3".
- **C2 4-mode MTP schedule + EDF (single node)**
  - [x] 2026-10-18 long-context mix (TraceLab 32-250K + ShareGPT 2 req/s, H200), ours + MTP vs stock + MTP: agent
    +54% / -34% / -49%, chat +23% / -90% / -42%. All six bar metrics clear, two repeats.
  - [x] 2026-10-18 same mix, no MTP vs stock without spec: agent TTFT / ITL frontier through (-30.9, -28.4) and
    (-29.2, -30.2), three repeats. Throughput +39-41%, chat -78..-85% / -38..-40%.
  - [x] 2026-10-18 standard mix (32-64K + ShareGPT 8 req/s), ours + MTP vs stock + MTP, two-rep means: agent +47% /
    -29.3% / -36%, chat -72.7% / -25%. The agent p1 tok/s/user is 4-5x. Misses: agent TTFT 0.7 pt, chat ITL 5 pt.
  - [x] 2026-10-18 no MTP vs stock mixed chunk: agent +25.7% / +15.6% / -31.2%, chat -85.6% / -36.0% (agent TTFT is the
    miss: stock prefills in 8192 chunks).
  - [x] 2026-10-18 output gates PASS for every arm, each with a null case (`gate_h200_*`).
- **C3 Predictable fused step -> EDF and routing work** -- **2026-10-20: not supported as a per-step claim.** With each
  schedule's best model (MLP / GBT on serving traces) every schedule predicts to ~3-8%; ours beats pdmux (5.9% vs
  7.5% at light load) but not stock mixed chunk (4.2%). The earlier 26.7% vs 53.5% used pdmux's linear form. Keep the
  step model as an enabler of SLO control (router + decode-first worker), not as a contribution
  (`server_perf_model.md`).
  Narrower claim that holds (pooled MTP fit, 2026-10-20): with a closed-form linear model the 4-mode step predicts to
  12-13% vs 24-27% for stock MTP's mixed step (the 4-mode terms help ours, not stock); a tree ensemble reaches ~3% on
  both. State it as "admits a simple model at half the error".
  - [x] 2026-10-18 S4 step model, held-out half: mixed-step MAPE 26.7% ours vs 53.5% stock MTP; decode-only 3.6% vs 2.7%.
    Decode ITL predictable to ~9% vs 35-117% on unfused schedules.
  - [x] EWMA acceptance length as a model input (point 3; measured 2.10 tokens/step).
  - [ ] Show that the better prediction changes decisions: SLO-check accuracy, i.e. predicted vs realised misses per
    routing decision, ours vs a stock-schedule model.
- **C4 Cache-first, SLO-checked routing on symmetric nodes (multi-node, 8 H20s)**
  - [x] 2026-10-19 fan-out workload vs PD disagg 2P2D without spec: agent goodput 3.0-7.1x, throughput 2.3-3.3x, TTFT
    -94..-96%; re-prefill share 3.4% vs 10.3%.
  - [x] 2026-10-19 per tenant: chat SLO attainment 91-96% vs 25-33% (heavy load), 97-99% vs 43-54% (light). Mean
    chat ITL is 1.3-1.9x disagg's, yet per-request chat TPOT stays under 20 ms for >= 90% of requests. Disagg's chat
    misses on TTFT.
  - [x] 2026-10-18 scaling 2 -> 4 pods: prefill hit 0.34 -> 0.39 (ours) vs 0.125 -> 0.055 (disagg) -- more prefill nodes
    dilute each prefix.
  - [x] 2026-10-19 Point 4 (pull-source choice): neutral at <= 16 H20s; per-page latency equal across sources. State it
    as a scale argument, not a measured gain.

## Section-by-section against the draft

- **Abstract / Intro.** Keep the draft's thesis, colocation vs rigid disaggregation, and replace its kernel sentence
  with C1 as above. Add MTP explicitly: the draft's Limitation "no benefit with speculative decoding" is now a
  result (C2 long mix clears all six with MTP). Shorten the abstract (S. Bian's comment).
- **2.2 baselines.** The draft's pdmux critique is confirmed and should carry data: no CPU-GPU overlap scheduling
  (our pdmux arms run `--disable-overlap-schedule`), CUDA graphs x SM groups (5 on H20, 8 on H200). Stock pdmux
  (divisor 36) on the standard mix is in the Baselines table.
  - [ ] Graph memory, pdmux vs ours, measured on one config (the draft quotes 2 GB / 8x; ours is not measured).
- **2.3 PD disaggregation.** Add Dynamo conditional disagg (`dynamo-conditional-disagg` memory note): argue on
  co-execution efficiency, symmetric nodes and EDF, not on per-round KV re-ship.
- **3. Motivation (S. Venkataraman: "show data for these points").**
  - 3.1 attention time ratio: at 50K depth a 743-token chunk costs ~28 ms of a 44 ms mixed step (2026-10-19).
  - 3.2 disagg underuses resources: re-prefill 10.3% vs 3.4%, prefill hit dilution 0.125 -> 0.055.
  - 3.3 chunked prefill vs SLO: stock MTP mixed steps 59 ms vs ours 44 ms; chat ITL interval shares (stock 34% of chat
    ITL time in > 80 ms intervals vs 16% ours).
  - [ ] 3.4 mixed-chunk tile waste: run the stock mixed-chunk kernel on a TraceLab step mix and count padded rows.
- **4. Design.** 4.1 problem formulation: meet per-tenant TTFT + TPOT SLOs, maximise goodput. 4.2 the 4-mode step (C1).
  4.3 EDF with uncached-token deadlines, a long-request deadline multiplier (2x above 200K) and deadline-driven chunk
  admission (C2). 4.4 step model with acceptance input (C3). 4.5 router (C4).
- **5.2.3 fair resource sharing / P:D ratio.** Superseded. The measured design is EDF order plus an SLO check, not a
  P:D compute-time ratio target (`edf-order-not-cap`: every per-step chunk cap lost).
- **5.4 KV pooling, ECMP-style random holder.** Untested as proposed; keep it as a design point only if point 4 runs.
- **5.5 CUDA graphs.** The 4-mode mixed step runs eager (`can_run_graph` rejects MIXED); host idle after it is 3.2 ms
  per step, <= 4% of GPU time. Decode steps keep their graphs.
- **6. Evaluation.** Testbed is H200 x1 single-node and 8 H20 multi-node (<= 16 GPUs). Models: Qwen2.5-7B + EAGLE3
  (gap 5). Workloads: TraceLab 32-64K and 32-250K + ShareGPT, plus the fan-out agent workload (`fanout_agents.py`).
  Metrics, per `serving-eval-matrix` and `always-record-error-bars`: per-tenant SLO attainment and goodput,
  tok/s/user percentiles, rate sweeps.
- **7. Ablations** (S. Venkataraman: "alternate designs that didn't work and why"). Measured negatives worth a
  paragraph each: per-step chunk caps and step-time budgets; per-chunk radix checkpoints; router per-class ITL SLO;
  soft node roles; time-based draining; random serving-node choice among equal-hit holders; symmetric in-kernel
  backfill (computes FA3 wrong); FA3 split-KV / pack-GQA on prefill chunks. The common thread: at a fixed amount of
  prefill work, re-slicing it does not move the mean step interval; ordering it by deadline does.
- **8. Related work.** Reference list in `debug_kernel_server_perf_4_mode.md` (three-bullet format), incl. JITServe.
- **9. Limitations.** Kept from the draft: per-architecture kernel work (Blackwell). Removed: SM80 and the
  no-spec-decode limitation. Added: decode isolation is where disagg still leads (agent TPOT ~2x at heavy load on the
  fan-out), and goodput, not per-token decode speed, is where ours wins.
