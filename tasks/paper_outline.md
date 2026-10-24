# AttentionFuse: Towards Flexible and Optimal Throughput-Latency Tradeoffs in Distributed LLM Serving

## Paper Outline (10/24)

- goal - meet per-tenant SLOs while maximizing throughput for mixed chat + agent LLM workloads, without a fixed
  prefill:decode node ratio
- motivation
  - mixed-chunk prefill (sglang on FA3) - every decode / verify row pays prefill-width attention tiles in the one
    varlen call; chat waits behind long agent prefills (chat TTFT 4.5-23 s on one H200)
  - prefill prioritized - decode stalls behind prefill; TBT blows up
  - pd disaggregation - re-ships each agent round's prefix KV from prefill to decode (decode pods hold 52-55 requests
    in KV transfer at 64 agent programs on 16 H20s), leaves decode compute idle, needs a tuned P:D ratio
  - pd multiplexing - no CPU-GPU overlap schedule, CUDA graphs multiplied by SM groups; a baseline only
- design
  - kernel: a unified fine-grained tile dispatcher for the mixed step - ONE CPU-side plan assigns every work tile of
    the batch to an executor and a tile class (FA2 classes 16 / 32 / 48 for decode / verify rows, FA3 for rows > 48
    packed queries and prefill), splits KV where it balances the SMs, and in the same pass fixes the merge schedule:
    which tiles of a row are partials, and which tile merges them inside the launch, so no separate merge kernel runs
  - the executors are stock FA3 and the persistent FA2 kernel with the LPT overlap schedule, chained by PDL; FA2
    runs second and reduces every split row, FA2's and FA3's, in its epilogue, which removes native v3's FA2-merge
    and FA3-combine launches; the plan and that merge, not either kernel, are the contribution
  - 4-mode step: prefill chunk + speculative verify rows in one forward
  - why does this help? rows stop paying prefill-width tiles, and the step time becomes a function of shape that a
    model can predict
- distributed inference
  - symmetric colocated nodes; the router sends each request to the highest-cache-hit node that can still meet its
    SLO, from a step model and the EDF queue ahead of it
  - EDF on every worker, deadlines from uncached tokens, per-tenant SLOs; demote-on-miss under overload
  - cluster KV pool: a request routed away from its best holder pulls the prefix instead of recomputing it
- evaluation (three stages, three models: Qwen2.5-7B, Llama-3.3-70B, GLM-4.5-Air, all shown in every stage)
  - kernel: faster mixed-step attention than native v3 on steps sampled from real serving traces, 2-mode and 4-mode
  - single node: higher SLO attainment for both tenants than sglang mixed chunk (+/- MTP)
  - multi node: more load at SLO than PD disaggregation, scaling 16 -> 32 GPUs, under load spikes and mixture shifts
  - fault tolerance

## Abstract

LLM serving mixes interactive chat with long-context agents. Their prefill and decode phases have different resource
needs and different SLOs. Production systems either colocate them under a chunked-prefill schedule, where every
decode row pays prefill-width attention and chat waits behind long prefills, or disaggregate them onto separate
nodes, where every agent round re-ships its prefix KV and the P:D ratio must be tuned to the workload. We present
AttentionFuse, a colocated serving system with three parts:
- a unified fine-grained tile dispatcher: one plan assigns every attention tile of a mixed step to FA3 or to a
  row-sized FA2 tile class, balances them across SMs, and schedules the split-KV merge inside the same plan, so
  partials are reduced just in time by the tile that finishes a row instead of by separate merge kernels;
- a 4-mode step that merges a prefill chunk into speculative verify, scheduled by deadline order with
  demote-on-miss;
- symmetric nodes with cache-first, SLO-checked routing and a cluster KV pool.
On captured serving steps, mixed-step attention is 1.23-1.57x faster than sglang's FA3 call. On one H200, SLO
attainment rises for both tenants (chat 50 -> 59-65%, agents 45-52 -> 77-85%) with chat TTFT -96%. On 16 H20s, ours
holds >= 90% chat attainment up to 4x the load of the best-tuned PD disaggregation. [Llama-70B / GLM numbers pending.]

## 1. Introduction

- What is the setting / problem
  - mixed workloads: chat (short prompts, tight TTFT / TPOT) beside coding agents (32-256K contexts, multi-round,
    a few hundred new tokens per round)
  - problem statement: meet per-tenant SLOs at the highest load, without dedicating nodes to a phase
- Why existing systems don't meet it
  - mixed chunk: one varlen attention call tiles decode / verify rows at prefill width; at long context that is most
    of the step; FCFS puts chat behind 64K+ agent prefills
  - disaggregation: the prefix KV moves prefill -> decode on every agent round; the transfer queue, not the prefill
    queue, is the limit (deadline order on the prefill pods changes nothing: 30.5 vs 31.2% chat attainment)
  - Dynamo's conditional disaggregation avoids most re-ships but has no SGLang implementation (its docs list SGLang as
    unsupported)
- What is our key design insight
  - the two roles want different tiles: give rows row-shaped FA2 tile classes and keep prefill on FA3
  - colocation keeps a session's KV where it was computed, so routing can follow cache hits instead of roles
  - a fused step has a predictable time, so the router and EDF can price a request before admitting it
- Contributions
  - C1 kernel: a unified fine-grained tile dispatcher -- one CPU plan per step over FA2 (16 / 32 / 48) and FA3 tiles
    with the split-KV merge schedule built into it (FA2 and FA3 partials reduced in the FA2 epilogue, no merge
    launch), over stock FA3 and the persistent LPT overlap FA2 kernel
  - C2 single node: the 4-mode step under deadline order with demote-on-miss
  - C3 cluster: symmetric nodes, SLO-checked cache-first routing, cluster KV pool
- Experimental results
  - kernel 1.23-1.57x over sglang's FA3 call on trace-sampled steps; single node attainment +18-77% relative for both
    tenants; 16 GPUs: >= 90% chat attainment at 64 agent programs vs 16 for the best disagg split [full-stack and
    Llama-70B / GLM pending]

## 2. Background

### 2.1 LLM Serving

- prefill (compute-bound, parallel over prompt tokens) and decode (memory-bound, one token per step); speculative
  decoding (EAGLE3) turns decode rows into 4-token verify rows
- GQA packs query heads per KV head: a 4-token verify row packs 28 rows on Qwen2.5-7B (TP1), 32 on Llama-3.3-70B
  (TP4) and 48 on GLM-4.5-Air (TP4)
- Contribution: motivates C1's tile classes (16 / 32 / 48, rows above 48 packed queries to FA3)
- Experiments: none; packed-row counts follow from each model's head configuration
- Graphs / metrics: table of packed rows per decode and per verify row, per model and TP degree
- Status: 28 / 32 / 48 packed rows per 4-token verify row (Qwen2.5-7B / Llama-3.3-70B / GLM-4.5-Air)

### 2.2 Decode-prioritized, Prefill-prioritized, and Chunked Prefill Schedules

- sglang mixed-chunk prefill: one varlen FA3 call over prefill chunk + decode rows
- sglang default prefill-prioritized: TBT stalls
- sglang + mooncake PD disaggregation: roles fixed per node, KV shipped per request
- PD-Multiplexing: green-context SM groups; no CPU-GPU overlap schedule; CUDA graphs per group
- Contribution: motivates C2: no stock schedule protects both tenants, each moves latency between them
- Experiments: one H200, Qwen2.5-7B, sglang FA3 backend; stock mixed chunk, stock prefill-prioritized (+/- MTP);
  chat 8 req/s + agent program concurrency 8; two repeats
- Graphs / metrics: per schedule, per tenant: TTFT mean and ITL p99, SLO attainment
- Status: stock mixed chunk chat TTFT mean 579.5 ms, agent ITL p99 417.8 ms; prefill-prioritized chat ITL p99 27.3 ms
  but worse chat TTFT (`debug_kernel_server_perf_4_mode.md` S0b); Llama-3.3-70B / GLM-4.5-Air [pending]

### 2.3 PD Disaggregation

- DistServe, Mooncake; NVIDIA Dynamo with conditional disaggregation (local prefill on the decode worker when the
  effective uncached length is small; vLLM and TRT-LLM only)
- static resource split, cannot adapt to tenant mix; decode compute idle
- Contribution: motivates C3: disaggregation needs a tuned P:D ratio, and the best ratio still misses the chat SLO
- Experiments: 16 H20s (8 pods x 2), Qwen2.5-7B no spec, stock sglang PD with KV pool, prefill pods swept 1..7 of 8;
  32 agent programs + chat 8 req/s
- Graphs / metrics: chat attainment and agent SLO goodput vs number of prefill pods
- Status: chat attainment 16.2% (1 prefill pod) -> 67.8% (6, best) -> 50.0% (7) (`pd_colocation_vs_disagg.md`,
  capacity phase A); Llama-3.3-70B / GLM-4.5-Air [pending]

### 2.4 KV-cache centric scheduling

- multi-round agents re-read 32-256K prefixes; cache-aware routing and KV pooling decide TTFT
- ThunderAgent's thrashing setting: working set larger than one node's KV
- Contribution: motivates C3's cache-first routing and cluster KV pool
- Experiments: 16 H20s, agent program concurrency 32, working set swept 16 / 64 / 128 tasks of 64K prefix (~1M / 4M /
  8M KV tokens against ~6.7M capacity), ours vs stock PD
- Graphs / metrics: cached-prefix fraction and agent tok/s vs working set (measured in 6.4)
- Status: disagg cached fraction 0.890 -> 0.605, ours 0.980 -> 0.949 (`pd_colocation_vs_disagg.md`, thrashing)

## 3. Challenges and motivation

### 3.1 Attention computation time ratio

- at 32-256K, attention dominates the mixed step; trace census: most mixed steps carry a prefill chunk beside
  decode / verify rows
- Contribution: motivates C1: attention speed decides step time at long context
- Experiments: per-layer time breakdown of real serving steps at 32-64K and 128-256K, two- and four-mode, all three
  models; census of step composition (prefill chunk + rows, rows only, prefill only) from the same traces
- Graphs / metrics: stacked bar of attention / GEMM / other time per context bucket; step-composition fractions
- Status: attention share [pending]; step census [pending]

### 3.2 PD Disaggregation underutilizes resources

- per-round prefix re-ship (~3.7 GB of KV per 64K-token round on Qwen2.5-7B); decode pods at 63-68% KV usage, ~7
  running requests each, 52-55 waiting on transfer
- Contribution: motivates C3: the limit of disaggregation is the per-round KV transfer, not the prefill queue
- Experiments: 16 H20s, stock PD at its best split (6 prefill + 2 decode pods), 64 agent programs + chat 16 req/s,
  FCFS and deadline order on the prefill pods
- Graphs / metrics: decode-pod time series of requests waiting on transfer, running requests and KV usage; chat
  attainment FCFS vs deadline order
- Status: 52-55 requests waiting on transfer (max 108), ~7 running at 63-68% KV usage; deadline order 31.2 -> 30.5%
  chat attainment (`pd_colocation_vs_disagg.md`, disagg + EDF); Llama-3.3-70B / GLM-4.5-Air [pending]

### 3.3 Chunked Prefill cannot adhere to SLOs

- the agent burst (32 requests in 60 s at 32-64K) holds chat at 4.5-13.9 s TTFT; plain EDF under overload serves
  already-late agent prefills first and makes chat late too
- Contribution: motivates C2: deadline order needs demote-on-miss to protect chat under overload
- Experiments: one H200, Qwen2.5-7B, 32-64K agent burst + ShareGPT chat 8 req/s; stock mixed chunk vs per-tenant
  deadlines alone vs deadlines + demote-on-miss
- Graphs / metrics: chat TTFT over time around the burst; per-tenant attainment per schedule
- Status: chat TTFT 4.5-13.9 s under stock; per-tenant deadlines alone change nothing (section 7); Llama-3.3-70B /
  GLM-4.5-Air [pending]

### 3.4 Mixed-chunk prefill reduces throughput

- one tile shape for both roles: rows padded to 128-row prefill tiles; splitting the call by role alone gives
  1.23-1.57x attention speed
- Contribution: motivates C1: rows need their own tile shape
- Experiments: the 96 trace-sampled captures of 6.2, one varlen FA3 call vs the role split
- Graphs / metrics: per-step time of the one call vs the split; fraction of row tiles that are padding
- Status: 1.23-1.57x (6.2 table, Qwen2.5-7B); padding fraction [pending]

### 3.5 Can we increase request capacity while maintaining SLOs?

- yes, if rows get their own kernel, chat is not queued behind infeasible prefills, and nodes stay symmetric
- Contribution: states C1-C3 together as the answer
- Experiments: those of 6.2-6.4
- Graphs / metrics: load held at >= 90% per-tenant attainment, ours vs each baseline
- Status: 64 vs 16 agent programs at 16 GPUs, schedule only (6.4); full stack [pending]

## 4. AttentionFuse Design

## 4.1 Problem Formulation

- per-tenant SLOs: chat 0.5 s TTFT + 20 ms TPOT, agent 10 s TTFT + 80 ms TPOT
- objective: maximize the load held at >= 90% attainment (per tenant), not mean latency
- per step: one plan assigns every tile (FA3 for rows > 48 packed, FA2 classes 16 / 32 / 48 otherwise) and its merge schedule
- per worker: EDF on uncached-token deadlines; demote a prefill once its deadline has passed
- per cluster: route to the highest-hit node predicted to meet the SLO; pull from the pool otherwise
- Contribution: frames C1 (per step), C2 (per worker) and C3 (per cluster) against one objective
- Experiments: every serving stage reports the objective: load swept on agent program concurrency and chat request
  rate, per-tenant attainment at each point
- Graphs / metrics: per-tenant attainment vs load, the 90% capacity marked per system
- Status: measured at 16 and 32 GPUs, schedule only (6.4); single node and full stack [pending]

## 5. Implementation

### 5.1 Persistent Kernel with overlap Schedule

- unified tile dispatcher (one CPU-side plan per step, shared by all layers)
  - classifies every row by packed query length: <= 16 / <= 32 / <= 48 -> FA2 tile classes, > 48 -> FA3
  - cuts each row's KV into chunks sized to balance SM load; LPT (longest first onto the least-loaded slot) across
    both executors' slots
  - emits the merge schedule with the work list: for every split row, the partial slots and the merge group
  - fast: one plan per step on the CPU, built incrementally and overlapped with the previous step (host plan time
    48-78 us mean on 44 serving geometries; on serving shapes, overlapped [pending])
- one merge, in the FA2 epilogue: FA2 runs second under PDL, so it reduces every split row, FA2's and FA3's
  alike, from merge items the plan queues after the compute items; FA3 tiles only write FP32 partials + LSE, and no
  merge kernel runs for either executor
- executors: stock FA3 (flash-attention's mainloop, paged) and the persistent FA2 kernel with the LPT overlap
  schedule (two CTAs per SM on different tile classes), both reading the one work list; PDL between the launches
- optional, ablated in 7: DSMEM handoff of partials to the reducing CTA; KV multicast for FA3 and FA2
- why not fuse FA2 and FA3 into one kernel: warp layout mismatch
  - FA3 is warp-specialized: one producer warpgroup issues TMA loads and two consumer warpgroups run wgmma on a
    128-row tile, so a CTA is 384 threads (12 warps) at 1 CTA per SM, ~98 KB of shared memory per CTA
  - FA2 is not specialized: 4 warps (128 threads), each warp both loads and computes with mma.sync, 2 CTAs per SM;
    its row tiles (16 / 32 / 48 query rows) are laid out over those 4 warps
  - one `__global__` has one block size, so a fused kernel launches every CTA at FA3's 384 threads: the FA2 role
    either leaves 256 of the 384 threads idle or is re-laid-out over 12 warps, which its small row tiles cannot fill
  - the register and shared-memory costs are artifacts of that layout -- FA3's warp specialization needing more
    warps per CTA -- not of either algorithm (measured 2026-10-06, H200, `-Xptxas -v` per named function)
    - 384 threads x 1 CTA/SM caps each thread near 170 registers, and ptxas gives one budget per kernel, so the two
      inlined role bodies share it: FA3 spills 33 676 B in the fused kernel vs 52 B alone in the identical template;
      outlining FA2 takes FA3 to 64 B but moves 17 500 B of spill onto FA2
    - FA2 at FA3's geometry: 168 registers / 1 136 B spill vs 255 registers / 0 B at its own 128 x 2; the SM-wide
      register budget is the same (255 x 256 = 170 x 384), so the loss is the idle threads
    - shared memory: FA3's 98 352 B leaves 134 048 B of the 232 448 B limit; FA2's tile-16 configuration needs
      135 168 B, over by 1 120 B
  - net: the fused kernel was 2.16x SLOWER than FA2 on everything (155.9 vs 72.3 us, both numerically correct; 1.83x
    on a 16 823-token-KV fixture); two launches of the same executors at their own layouts were 1.21x FASTER
  - SM sharing between two separate launches is a different question: co-running FA3 on B SMs beside FA2 gains 18%
    only where the roles take comparable time, ties or loses elsewhere, and is bounded at ~1.2% of serving GPU time on
    TraceLab (2026-10-17). So the executors are separate launches chained by PDL, and what they share is the plan:
    tile classes, KV splits, slot placement and the merge schedule
  - the same effect appears inside the FA2 executor: one persistent kernel holding the 48 / 32 / 16 runners compiles
    to one register budget, and the 48 runner (1 q-warp x 3 MMAs at 4 warps) makes the whole kernel spill
    1 436-1 444 B; with it compiled out the kernel spills 16-20 B (2026-10-24). So the 48 class needs a layout with
    16 rows per warp, or its rows go to FA3 [pending]
- measured build so far (6.2) is the baseline for this: two separate plans (FA3's and flashinfer's), non-persistent
  FA2 rows, and two merge mechanisms (FA3 combine launch + flashinfer's merge)
- Contribution: C1, mandatory parts: the one plan, and the split-KV reduction of FA2 and FA3 partials in the FA2
  epilogue with no merge launch
- Experiments: plan correctness and plan time on captured serving geometries of all three models; FP32 gate on the
  96 captures; the fused-vs-two-launch control above
- Graphs / metrics: host plan time per step (mean, p99) vs batch size; FP32 worst relative error; register / spill
  table per build
- Status: planner exact on 44 geometries, plan time 48-78 us mean; previous build FP32 worst 0.0067 on 96 captures
  (`debug_kernel_server_perf_4_mode.md` A6); plan time on serving shapes and the 48 class [pending]

### 5.2 Router and resource sharing

#### 5.2.1 Router design

- cache-first: order nodes by prefix-hit bucket, then load; take the first that passes the SLO check
- the check prices the request's own prefill plus the EDF-ordered queue ahead of it and the decode load beside it
- Contribution: C3, SLO-checked cache-first routing
- Experiments: 16 H20s, all three models, fan-out agents + chat; cache-first + SLO check vs cache-only vs
  round-robin, load swept on agent program concurrency
- Graphs / metrics: per-tenant attainment vs load per policy; cached-prefix fraction
- Status: [pending]

#### 5.2.2 Performance model

- one model for the fused step: inputs prefill tokens, prefix length, decode / verify rows and KV, predicted
  acceptance length (EWMA per worker)
- Contribution: supports C2 and C3: EDF and the router price a request from predicted step time
- Experiments: fit on logged serving steps of the current kernel, per model; held-out steps
- Graphs / metrics: predicted vs measured step time scatter; MAPE per step type (decode, prefill, mixed, 4-mode)
- Status: re-fit on the current kernel [pending]; earlier fits on the fused path: 0.4% decode / 3.2% prefill-step
  MAPE

#### 5.2.3 Fair Resource sharing

- deadline-sized chunks (1024 / 2048) from the EDF backlog; chunk preemption for an earlier deadline
- Contribution: C2, chunk size and preemption from the deadline backlog
- Experiments: one node, all three models, agent burst + chat; deadline-sized chunks vs fixed chunk, preemption on
  vs off
- Graphs / metrics: per-tenant TTFT / ITL attainment; agent throughput
- Status: [pending]

### 5.3 SLO-aware scheduling

- deadlines from uncached tokens, per-tenant SLOs (0.5 s below 8192 uncached tokens, 10 s above)
- demote-on-miss: a prefill whose deadline has passed ranks behind every one that can still make it (EDF's overload
  rule); retracted decodes never demoted
- Contribution: C2, deadline order with demote-on-miss
- Experiments: one H200 (Qwen2.5-7B) and one 4-GPU H20 pod (Llama-3.3-70B, GLM-4.5-Air); stock vs per-tenant
  deadlines alone vs + demote-on-miss; 32-64K agent burst + chat
- Graphs / metrics: per-tenant attainment; chat TTFT distribution
- Status: deadlines alone no change, + demote-on-miss chat TTFT -96..-98% (Qwen2.5-7B, section 7); Llama-3.3-70B /
  GLM-4.5-Air [pending]

### 5.4 KV Cache pooling policy

- mooncake L3 store as a cluster pool; write-through-selective; the router mirrors the pool and prices pulls
- Contribution: C3, the cluster KV pool
- Experiments: the 6.4 thrashing sweep (working set 1M -> 8M KV tokens against a 6.7M pool); pool on vs off
- Graphs / metrics: cached-prefix fraction and agent tok/s vs working set
- Status: ours -10.6% throughput / -3.1 points cached fraction vs disagg -55% / -28.5 (earlier settings); pool on vs
  off [pending]

### 5.5 CUDA Graph support

- one graph set for the 4-mode step (pdmux multiplies graphs by SM groups)
- Contribution: supports C2: the 4-mode step runs under CUDA graphs and overlap scheduling
- Experiments: graphs on vs off for the 4-mode step; graph count and capture memory vs pdmux
- Graphs / metrics: table of graph count, capture memory and decode step time
- Status: [pending]

### 5.6 DeepEP with auto schedule?

- out of scope (dense models and GLM-4.5-Air at TP4)
- Contribution: none
- Experiments: none
- Graphs / metrics: none
- Status: out of scope

## 6. Evaluation

### 6.1 Settings

- Models
  - Qwen2.5-7B-Instruct (TP1, YaRN 4x to 128K) + trained EAGLE3 head
  - Llama-3.3-70B-Instruct (TP4) + trained EAGLE3 head
  - GLM-4.5-Air (TP4) + its own NEXTN layer
  - all three are evaluated in every stage and shown in every eval subsection, with full results recorded per model
  - We may show only two models in the final version; if so, Llama-3.3-70B and Qwen2.5-7B (the more mainstream).
- Testbed: 1 H200 for kernel and single-node stages; H20 nodes (4 GPUs each), 16 GPUs and 32 GPUs, for multi-node
- Baselines
  - kernel: sglang's single varlen FA3 call; the best single fixed FA3 configuration; a per-step FA3 oracle; native
    v3 (stock FA3 + our tuned BatchPrefill FA2, with its own merge and combine launches); the role split on FA3
  - single node: sglang mixed chunk on FA3 and prefill-prioritized; stock EAGLE3 MTP for 4-mode; pdmux
  - multi node: sglang + mooncake PD disaggregation with the P:D ratio swept at worker granularity, KV pool on;
    reference policy: conditional disaggregation (Dynamo's rule at its defaults, effective uncached length under 2048
    tokens and under 0.7 of the prompt, our implementation on sglang)
  - kernel in stages 2-3: native v3 until the unified dispatcher beats it in stage 1, then the unified dispatcher
- Workloads
  - TraceLab coding-agent traces (context mixtures from 32K to a few hundred K: 32-64K and 128-256K, multi-round) +
    ShareGPT chat
  - fan-out coding agents on shared 64K repository prefixes + ShareGPT chat (multi-node)
  - standalone sets: ShareGPT, math, coding, agentic web browsing, at load around capacity
  - agent load is program concurrency; chat load is request rate
- SLOs: chat 0.5 s TTFT + 20 ms TPOT; agent 10 s TTFT + 80 ms TPOT, per tenant
- Metrics: throughput vs tok/s/user Pareto curve with p10-p90 bars; TTFT / ITL SLO attainment vs request rate, per
  tenant; every serving result shows per-request spread and per-repeat values
- Protocol: same frozen tree for both arms, opposite-order pairs, two repeats, metrics recomputed from raw token
  arrivals, CUDA graphs and overlap on; quality gate (GSM8K + passkey at 32-60K) vs a baseline-vs-baseline noise floor
- Contribution: common setting for C1-C3
- Experiments: 3 models x 3 stages (kernel, single node, 16 / 32 GPUs) + fault tolerance
- Graphs / metrics: as listed above
- Status: Qwen2.5-7B measured on the schedule-only and earlier-kernel builds; GLM-4.5-Air kernel stage partial
  (6.2); Llama-3.3-70B [pending]

### 6.2 Kernel microbenchmarks

- Contribution: C1, the unified tile dispatcher with the merge in the FA2 epilogue; it must beat native v3, which
  runs the same FA3 kernel with separate FA2-merge and FA3-combine launches
- Rationale: in one varlen FA3 call a 4-token verify row (28-48 packed rows) is tiled at 128 rows; row-shaped tiles
  stop the padding, and LPT spreads row work of different sizes across SMs
- Experiments: every Nth real mixed serving step captured (24 per context x mode, 96 steps per model), replayed with
  the same tensors; contexts 32-64K and 128-256K, two- and four-mode; Qwen2.5-7B on one H200, Llama-3.3-70B and
  GLM-4.5-Air at TP4 on H20 (rank 0); baselines: sglang's FA3 call, best fixed FA3 configuration, per-step oracle,
  native v3, role split on FA3
- Graphs / metrics: time-weighted speedup over the 96 steps with a bootstrap 95% CI, per (context, mode), against
  each baseline; per-step speedup distribution (CDF); FP32 worst relative error; host plan time per step; ablation
  bars in 7
- Status: unified dispatcher measured on Qwen2.5-7B (H200 and H20) and Llama-3.3-70B (H20, TP4, 32-64K: 1.009x /
  1.011x native v3 on two- / four-mode, 1.38x / 1.61x sglang's FA3 call); GLM-4.5-Air [pending]
- Results (Qwen2.5-7B, H200, unified dispatcher: one plan, stock FA3 at the plan's split, v3's FA2 body with the
  chunk-parallel in-launch merge; 96 fresh captures, all FP32-exact, worst 0.0082)

  | context / mode | vs sglang's FA3 call | vs native v3 [95% CI] | vs role split on FA3 |
  |---|---:|---:|---:|
  | 32-64K, 2-mode | 1.48x | 1.033x [1.006-1.051] | 1.015x |
  | 32-64K, 4-mode | 1.43x | 1.047x [1.035-1.057] | 1.020x |
  | 128-256K, 2-mode | 1.51x | 1.027x [1.018-1.036] | 1.014x |
  | 128-256K, 4-mode | 1.44x | 1.027x [1.018-1.041] | 1.018x |

  - On H20 (78 SMs) the same build is level with v3 (0.998-1.008x) and 1.10-1.23x sglang's FA3 call. There FA3 sets
    the step, and the FA2 path's 3-10% saving runs beneath it.
- Earlier results (Qwen2.5-7B, H200, role-split serving build, earlier captures)

  | context / mode | vs sglang's FA3 call | vs best fixed FA3 config | vs per-step oracle |
  |---|---:|---:|---:|
  | 32-64K, 2-mode | 1.35x [1.12-1.91] | 0.98x | 0.97x |
  | 32-64K, 4-mode | 1.23x [1.09-1.56] | 0.99x | 0.98x |
  | 128-256K, 2-mode | 1.57x [1.24-2.64] | 0.99x | 0.98x |
  | 128-256K, 4-mode | 1.37x [1.16-1.89] | 0.99x | 0.99x |

  - FP32 worst relative error <= 0.0075; GSM8K and passkey within the baseline noise floor
  - reading: this build has two plans and two merges; it only matches a well-tuned FA3 split in isolation
- GLM-4.5-Air: 4-mode 0.63-0.98x on the role-split build (its 48-row verify has no 48 tile today), 2-mode up to
  1.35x; on the unified dispatcher [pending]

### 6.3 Single-node benchmarks

- Contribution: C2, the 4-mode step under deadline order with demote-on-miss, on the C1 kernel
- Rationale: the kernel speeds the step; deadline order lets chat through the agent burst; demote-on-miss stops
  already-late agents from making chat late
- Experiments: Qwen2.5-7B on one H200, Llama-3.3-70B and GLM-4.5-Air on one 4-GPU H20 pod (TP4); baselines sglang
  mixed chunk and prefill-prioritized on FA3 (+/- stock EAGLE3 MTP), pdmux; kernel native v3 until the unified
  dispatcher beats it; load swept on agent program concurrency and chat request rate; context mixtures from 32K to a
  few hundred K (32-64K, 128-256K), two- and four-mode; standalone sets ShareGPT, math, coding, agentic web browsing
  at load around capacity
- Graphs / metrics (every serving result carries per-request spread and per-repeat values):
  - throughput vs interactivity Pareto curve: output tok/s vs per-user tok/s, median marker with p10-p90 bars
    (p1-p99 whiskers), one point per load
  - TTFT and ITL SLO attainment vs request rate, per tenant (chat 0.5 s TTFT + 20 ms TPOT, agent 10 s + 80 ms)
  - TTFT and ITL / TPOT per tenant: mean, p50, p90, p99
  - per-tenant SLO goodput (tok/s of requests meeting both SLOs)
- Status: Qwen2.5-7B results below; rate sweeps and Pareto curves [pending]; Llama-3.3-70B and GLM-4.5-Air
  [pending]; standalone sets [pending]
- Kernel alone, same schedule (Qwen2.5-7B, 128-256K, agent conc 4 + chat 4/s)
  - 4-mode: agent throughput +10.7%, agent TPOT -24.2%, ITL -20.8%; 2-mode: +11.6%, -20.1%, -16.7%
  - vs a role-split FA3 (FA3 for both roles): equal throughput, but agent decode latency only -1.5..+6.6%, so the FA2
    rows are worth 15-25 points of agent ITL
- Full stack vs stock (Qwen2.5-7B, 32-64K, agent burst of 32 requests in 60 s + chat 8/s)

  | mode | chat attainment | chat at 1 s / 30 ms | agent attainment | chat TTFT | agent TPOT | agent tok/s |
  |---|---:|---:|---:|---:|---:|---:|
  | 4-mode (stock MTP -> ours) | 50.0 -> 59.0% | 50.3 -> 83.3% | 45.0 -> 76.7% | -95.2% | -49.1% | -8.5% |
  | 2-mode (stock -> ours) | 49.1 -> 65.4% | 49.5 -> 85.4% | 51.7 -> 85.0% | -96.2% | -16.6% | +1.7% |

  - cost: chat TPOT +35-56%; admitted chat decodes beside the burst's long prefills, where stock makes chat wait
- 128-256K: neither system reaches 90% for either tenant at agent concurrency 1, 2 or 4 on one GPU (cold
  128-256K turns cannot prefill in 10 s); ours still +8-19 points chat and +25-38 points agent attainment at
  concurrency 4 -- motivates 6.4
- Schedule only on the FA3 backend, chat 8 req/s + agent program concurrency 8, two repeats (S0b in
  `debug_kernel_server_perf_4_mode.md`): ours no MTP agent attainment 86.7 / 95.6% vs stock mixed chunk 84.4 /
  84.4%; ours + MTP behind stock prefill-prioritized + MTP at the 80 ms ITL SLO (62.2 / 64.4 vs 66.7 / 68.9%), ahead at 30 ms (55.6 /
  51.1 vs 46.7 / 48.9%); higher agent concurrency [pending]
- Llama-3.3-70B (one 4-GPU node) [pending]; GLM-4.5-Air [pending]

### 6.4 Multi-node benchmarks

- Contribution: C3, symmetric nodes + SLO-checked cache-first routing + cluster KV pool; with the full stack, C1-C3
  together
- Rationale: colocation keeps the agent prefix where it was computed; disaggregation re-ships it every round
- Experiments: 16 and 32 H20s, Qwen2.5-7B (TP1 workers), Llama-3.3-70B and GLM-4.5-Air (TP4 replicas); fan-out
  agents on shared 64K prefixes + ShareGPT chat; load swept on agent program concurrency with chat rate scaled
  alongside; baselines stock sglang PD with the P:D ratio swept (headline) and conditional disaggregation at
  Dynamo's defaults (2048 tokens / 0.7 of the prompt); stage 3 = full stack (kernel + 4-mode MTP) vs stock PD with the
  P:D ratio swept, at higher load with mixture spikes; kernel native v3 until the unified dispatcher beats it
- Graphs / metrics (per-request spread and per-repeat values on every result):
  - capacity at SLO: per-tenant TTFT / ITL attainment vs load for ours and each P:D split of the baseline; the load
    each system holds at >= 90%
  - the 6.3 Pareto curve (throughput vs tok/s/user, p10-p90 bars) at 16 and 32 GPUs
  - scaling: capacity at 16 vs 32 GPUs
  - cache: cached-prefix fraction and re-prefilled tokens vs working-set size
  - load spikes and mixture shifts: attainment and throughput in 30 s windows around each event
- Status: schedule-only Qwen2.5-7B results below; conditional disagg at Dynamo's defaults [pending]; full stack
  [pending]; Llama-3.3-70B / GLM-4.5-Air [pending]
- Schedule only (kernel off, no speculation), Qwen2.5-7B, 16 H20s, fan-out agents + chat; chat attainment / agent
  SLO goodput (tok/s)

  | agent programs | ours | best PD disagg (6 prefill + 2 decode pods) | conditional disagg (4096-token rule) |
  |---|---|---|---|
  | 16 | 99.5% / 293 | 92.8% / 282 | 99.6% / 275 |
  | 32 | 99.0% / 568 | 67.8% / 359 | 98.9% / 518 |
  | 48 | 96.6% / 772 | 45.8% / 231 | 96.0% / 635 |
  | 64 | 94.2% / 964 | 31.2% / 165 | 94.6% / 845 |
  | 96 | 89.2% / 1187 | -- | 84.4% / 906 |
  | 128 | 72.6% / 1221 | -- | 60.9% / 890 |

  - ours holds >= 90% chat attainment to 64 programs, disagg to 16 (all P:D ratios swept; deadline order on its
    prefill pods changes nothing)
  - the gap is the per-round re-ship: the conditional-disagg policy, which prefills appends locally, matches ours to
    64 programs; ours leads past the knee (+4.8 / +11.7 points, agent SLO goodput +31-37%)
  - this conditional-disagg run used a 4096-token remote-prefill threshold; rerun at Dynamo's defaults (2048
    tokens / 0.7 of the prompt) [pending]
- 32 H20s, same protocol: ours >= 96% through 128 programs; best disagg 83.7% at 32, 26.7% at 128
- Full stack (kernel + 4-mode MTP) vs stock PD with the P:D ratio swept, higher load and mixture spikes, 16 and 32
  GPUs [pending]
- Llama-3.3-70B (4 TP4 replicas on 16 H20s) [pending]; GLM-4.5-Air [pending]
- Cache pressure: working set 1M -> 8M KV tokens against a 6.7M pool; disagg loses 55% throughput and 28.5 points of
  cached fraction, ours 10.6% and 3.1 [earlier settings; rerun with the full stack pending]

### 6.5 Fault tolerance

- Contribution: C3, symmetric nodes and the dead-worker reroute keep both tenants' SLOs when workers fail
- Experiments: kill one and then two workers mid-run, prefill-role and decode-role in the baseline (stock PD at its
  best P:D split, health-check ejection on), the same workers for ours; full stack, 16 GPUs, stage-3 workload and
  load; Qwen2.5-7B, Llama-3.3-70B, GLM-4.5-Air
  - prefill down: ours should hold, because every node can prefill and the SLO router moves new prefills to the
    survivors
  - decode down: the baseline keeps its admission rate and loses ITL on the remaining decode pods; ours keeps ITL
    by admitting fewer prefills (SLO check + EDF) and gives up some TTFT attainment instead
- Graphs / metrics: per-tenant TTFT and ITL attainment and throughput in 30 s windows before / during / after;
  lost requests; recovery time
- Status: prefill-role and decode-role kills, full stack [pending]; earlier settings (8 GPUs, schedule only, kill
  2 of 8 GPUs at 300 s): ours reroutes (agent attainment 96.4 -> 98.1%, chat 85.0%), PD disagg loses half its
  prefill capacity (18 -> 34%, chat 9-15%)

## 7. Ablation study

- Contribution: isolates which parts of C1-C3 carry the gains
- Experiments: kernel ablations on the 96 trace-sampled captures (or their FA2 rows alone) per model; schedule and
  router ablations on the 6.3 / 6.4 workloads
- Graphs / metrics: time-weighted speedup bars per ablation (kernel); per-tenant attainment and throughput
  (schedule, router)
- Status: per item below
- kernel
  - unified plan vs two plans [pending]
  - merge in the FA2 epilogue vs separate merge launches [pending] on the unified dispatcher; on FA2 rows alone
    (2026-10-24) the plan-scheduled merge items took 4.86 ms vs 6.21 ms for a last-arriver merge and 5.36 ms for stock
    BatchAttention
  - persistent overlap + LPT FA2 vs v3's non-persistent BatchPrefill FA2 under the same plan and merge [pending];
    current build, 2026-10-24, identical FA2 rows: ours 5.03 / 4.98 ms vs v3 4.74 / 4.33 ms (two-mode / four-mode
    sets), under diagnosis
  - DSMEM handoff of partials (optional; just-in-time reduction from "Taming Dynamism on GPUs", SOSP '26): the
    producer CTA TMA-copies its partial into the consumer CTA's shared-memory slot, a per-slot mbarrier counts the
    bytes (expect_tx), and the consumer reduces in local shared memory; vs the merge in the FA2 epilogue [pending]
  - KV multicast for FA3 and FA2 (optional): shared-KV ceiling on the 96 captures [pending], then multicast under the
    unified plan [pending]; earlier runs on the old flashinfer mainloop gave -0.2..+1.7% vs unicast
  - 48-row FA2 tile vs FA3 for rows of 33-48 packed queries [pending]; a 64-row FA2 class lost to FA3 on GLM-4.5-Air
    4-mode rows (FA3 1.02-1.59x faster), so rows above 48 go to FA3
  - KV split policy (before merge items, vs native v3): fixed 0.95-0.98x, flashinfer's `kv_len_limit` 0.89-0.94x,
    dynamic makespan 0.90-0.96x; fixed is the default
  - PDL vs serial launches [pending]
- FA2 rows vs FA3 rows (role split): equal throughput, FA2 rows 15-25 points better agent ITL
- schedule: per-tenant deadlines alone (no change) vs + demote-on-miss (chat TTFT -96..-98%); feasibility demotion at
  a fixed rate demoted every cold 128-256K agent on arrival
- per-step chunk caps for chat TPOT: starve prefill (binds on 86% of steps)
- router: cache-first + SLO check vs cache-only vs round-robin [pending]

## 8. Background and Related Work

- serving frameworks: vLLM, sglang, TRT-LLM; Dynamo
- disaggregation: DistServe, Mooncake, Splitwise; semi-PD (MPS SM split, restart to change), DynaServe, Dynamo
  conditional disaggregation
- colocation and multiplexing: Sarathi-Serve, PD-Multiplexing / MuxWise, POD-Attention
- kernel fusion: POD-Attention, persistent kernels, FlashAttention-3, flashinfer holistic scheduling, cross-SM fusion
  with just-in-time reduction over DSMEM ("Taming Dynamism on GPUs: Cross-SM Kernel Fusion via SM Cooperation and
  Just-in-Time Reduction", SOSP '26)
- attention-FFN disaggregation (orthogonal)

## 9. Conclusion and limitations

- Conclusion: role-separated attention, a deadline-ordered 4-mode step and symmetric cache-first routing raise SLO
  attainment on one node and the load held at SLO on a cluster, without a P:D ratio
- Limitations
  - on one GPU, chat TPOT rises 35-56% when chat is admitted during an agent burst; holding 20 ms beside 32-64K
    prefill needs chunks below ~200 tokens
  - 128-256K agents have no single-GPU operating point at a 10 s TTFT SLO
  - four-mode agent throughput -8.5% at 32-64K on one node
  - Hopper only; Blackwell needs new FA3 / FA2 builds
