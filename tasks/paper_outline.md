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
    the batch to an executor and a tile class (FA2 persistent LPT classes 16 / 32 / 48 for decode / verify rows, FA3
    for rows > 48 and prefill), splits KV where it balances the SMs, and in the same pass fixes the merge schedule:
    which tiles of a row are partials, and which tile merges them just in time, so no separate merge kernel runs
  - the two executors launch from that one plan, chained by PDL; the plan, not either kernel, is the contribution
  - 4-mode step: prefill chunk + speculative verify rows in one forward
  - why does this help? rows stop paying prefill-width tiles, and the step time becomes a function of shape that a
    model can predict
- distributed inference
  - symmetric colocated nodes; the router sends each request to the highest-cache-hit node that can still meet its
    SLO, from a step model and the EDF queue ahead of it
  - EDF on every worker, deadlines from uncached tokens, per-tenant SLOs; demote-on-miss under overload
  - cluster KV pool: a request routed away from its best holder pulls the prefix instead of recomputing it
- evaluation (three stages, three models: Qwen2.5-7B, Llama-3.3-70B, GLM-4.5-Air)
  - kernel: faster mixed-step attention on steps sampled from real serving traces, 2-mode and 4-mode
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
  - the two roles want different kernels: give rows a row-shaped persistent kernel and keep prefill on FA3
  - colocation keeps a session's KV where it was computed, so routing can follow cache hits instead of roles
  - a fused step has a predictable time, so the router and EDF can price a request before admitting it
- Contributions
  - C1 kernel: a unified fine-grained tile dispatcher -- one plan over FA2 (16 / 32 / 48) and FA3 tiles with the
    split-KV merge schedule built into it (just-in-time reduction by the last partial of each row, no merge kernel)
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

### 2.2 Decode-prioritized, Prefill-prioritized, and Chunked Prefill Schedules

- sglang mixed-chunk prefill: one varlen FA3 call over prefill chunk + decode rows
- sglang default prefill-prioritized: TBT stalls
- sglang + mooncake PD disaggregation: roles fixed per node, KV shipped per request
- PD-Multiplexing: green-context SM groups; no CPU-GPU overlap schedule; CUDA graphs per group

### 2.3 PD Disaggregation

- DistServe, Mooncake; NVIDIA Dynamo with conditional disaggregation (local prefill on the decode worker when the
  effective uncached length is small; vLLM and TRT-LLM only)
- static resource split, cannot adapt to tenant mix; decode compute idle

### 2.4 KV-cache centric scheduling

- multi-round agents re-read 32-256K prefixes; cache-aware routing and KV pooling decide TTFT
- ThunderAgent's thrashing setting: working set larger than one node's KV

## 3. Challenges and motivation

### 3.1 Attention computation time ratio

- at 32-256K, attention dominates the mixed step; trace census: most mixed steps carry a prefill chunk beside
  decode / verify rows

### 3.2 PD Disaggregation underutilizes resources

- per-round prefix re-ship (~3.7 GB of KV per 64K-token round on Qwen2.5-7B); decode pods at 63-68% KV usage, ~7
  running requests each, 52-55 waiting on transfer

### 3.3 Chunked Prefill cannot adhere to SLOs

- the agent burst (32 requests in 60 s at 32-64K) holds chat at 4.5-13.9 s TTFT; plain EDF under overload serves
  already-late agent prefills first and makes chat late too

### 3.4 Mixed-chunk prefill reduces throughput

- one tile shape for both roles: rows padded to 128-row prefill tiles; splitting the call by role alone gives
  1.23-1.57x attention speed

### 3.5 Can we increase request capacity while maintaining SLOs?

- yes, if rows get their own kernel, chat is not queued behind infeasible prefills, and nodes stay symmetric

## 4. AttentionFuse Design

## 4.1 Problem Formulation

- per-tenant SLOs: chat 0.5 s TTFT + 20 ms TPOT, agent 10 s TTFT + 80 ms TPOT
- objective: maximize the load held at >= 90% attainment (per tenant), not mean latency
- per step: one plan assigns every tile (FA3 for rows > 48 packed, FA2 classes 16 / 32 / 48 otherwise) and its merge schedule
- per worker: EDF on uncached-token deadlines; demote a prefill once its deadline has passed
- per cluster: route to the highest-hit node predicted to meet the SLO; pull from the pool otherwise

## 5. Implementation

### 5.1 Persistent Kernel with overlap Schedule

- unified tile dispatcher (one CPU-side plan per step, shared by all layers)
  - classifies every row by packed query length: <= 16 / <= 32 / <= 48 -> FA2 tile classes, > 48 -> FA3
  - cuts each row's KV into chunks sized to balance SM load; LPT (longest first onto the least-loaded slot) across
    both executors' slots
  - emits the merge schedule with the work list: for every split row, the partial slots and the merge group
  - fast: one plan per step on the CPU, built incrementally and overlapped with the previous step [plan-time
    numbers pending]
- one merge, in the FA2 epilogue: FA2 runs second under PDL, so the FA2 tile that completes a merge group reduces
  it just in time -- FA2 partials handed off through DSMEM within a cluster, FA3 partials read from L2 -- and no
  merge kernel runs for either executor
- executors: FA3 (flash-attention's mainloop, paged) and the persistent FA2 kernel with the LPT overlap schedule (two
  CTAs per SM on different tile classes), both reading the one work list; PDL between the launches
- 48-row FA2 tile needs its own warp layout [in progress]
- measured build so far (6.2) is the baseline for this: two separate plans (FA3's and flashinfer's), non-persistent
  FA2 rows, and two merge mechanisms (FA3 combine launch + flashinfer's merge)

### 5.2 Router and resource sharing

#### 5.2.1 Router design

- cache-first: order nodes by prefix-hit bucket, then load; take the first that passes the SLO check
- the check prices the request's own prefill plus the EDF-ordered queue ahead of it and the decode load beside it

#### 5.2.2 Performance model

- one model for the fused step: inputs prefill tokens, prefix length, decode / verify rows and KV, predicted
  acceptance length (EWMA per worker)
- [re-fit on the current kernel pending; earlier fits on the fused path: 0.4% decode / 3.2% prefill-step MAPE]

#### 5.2.3 Fair Resource sharing

- deadline-sized chunks (1024 / 2048) from the EDF backlog; chunk preemption for an earlier deadline

### 5.3 SLO-aware scheduling

- deadlines from uncached tokens, per-tenant SLOs (0.5 s below 8192 uncached tokens, 10 s above)
- demote-on-miss: a prefill whose deadline has passed ranks behind every one that can still make it (EDF's overload
  rule); retracted decodes never demoted

### 5.4 KV Cache pooling policy

- mooncake L3 store as a cluster pool; write-through-selective; the router mirrors the pool and prices pulls

### 5.5 CUDA Graph support

- one graph set for the 4-mode step (pdmux multiplies graphs by SM groups)

### 5.6 DeepEP with auto schedule?

- out of scope (dense models and GLM-4.5-Air at TP4)

## 6. Evaluation

### 6.1 Settings

- Models
  - Qwen2.5-7B-Instruct (TP1, YaRN 4x to 128K) + trained EAGLE3 head
  - Llama-3.3-70B-Instruct (TP4) + trained EAGLE3 head
  - GLM-4.5-Air (TP4) + its own NEXTN layer
- Testbed: 1 H200 for kernel and single-node stages; H20 nodes (4 GPUs each), 16 GPUs and 32 GPUs, for multi-node
- Baselines
  - kernel: sglang's single varlen FA3 call; the best single fixed FA3 configuration; a per-step FA3 oracle
  - single node: sglang mixed chunk on FA3; stock EAGLE3 MTP for 4-mode
  - multi node: sglang + mooncake PD disaggregation with the P:D ratio swept at worker granularity, KV pool on;
    reference policy: conditional disaggregation (Dynamo's rule, our implementation on sglang)
- Workloads
  - TraceLab coding-agent traces (32-64K and 128-256K contexts, multi-round) + ShareGPT chat
  - fan-out coding agents on shared 64K repository prefixes + ShareGPT chat (multi-node)
- SLOs: chat 0.5 s TTFT + 20 ms TPOT; agent 10 s TTFT + 80 ms TPOT
- Protocol: same frozen tree for both arms, opposite-order pairs, two repeats, metrics recomputed from raw token
  arrivals, CUDA graphs and overlap on; quality gate (GSM8K + passkey at 32-60K) vs a baseline-vs-baseline noise floor

### 6.2 Kernel microbenchmarks

- Contribution: C1, mixed-step attention split by role
- Rationale: in one varlen FA3 call a 4-token verify row (28-48 packed rows) is tiled at 128 rows; row-shaped tiles
  stop the padding, and LPT spreads row work of different sizes across SMs
- Setup: every Nth real mixed serving step captured (24 per context x mode, 96 steps), replayed with the same
  tensors; time-weighted speedup with a bootstrap 95% CI
- Results (Qwen2.5-7B, H200, current serving build)

  | context / mode | vs sglang's FA3 call | vs best fixed FA3 config | vs per-step oracle |
  |---|---:|---:|---:|
  | 32-64K, 2-mode | 1.35x [1.12-1.91] | 0.98x | 0.97x |
  | 32-64K, 4-mode | 1.23x [1.09-1.56] | 0.99x | 0.98x |
  | 128-256K, 2-mode | 1.57x [1.24-2.64] | 0.99x | 0.98x |
  | 128-256K, 4-mode | 1.37x [1.16-1.89] | 0.99x | 0.99x |

  - FP32 worst relative error <= 0.0075; GSM8K and passkey within the baseline noise floor
  - reading: this build has two plans and two merges; it only matches a well-tuned FA3 split in isolation. The
    unified dispatcher (one plan, 16 / 32 / 48 + FA3 tiles, merge schedule in the plan) is C1 and is re-measured
    here [pending]
- Llama-3.3-70B [pending]; GLM-4.5-Air: 4-mode 0.63-0.98x on the current build (its 48-row verify has no 48 tile
  today), 2-mode up to 1.35x
- Ablations: unified plan vs two plans; JIT merge from the plan vs separate merge kernels; LPT vs non-persistent rows; 48 tile vs pad-to-64 [pending];
  TMA multicast on FA3 KV loads: -0.2..+1.7% vs unicast (within noise, bounded at ~3% on serving mixes), not
  adopted

### 6.3 Single-node benchmarks

- Contribution: C2, the 4-mode step under deadline order with demote-on-miss, on the C1 kernel
- Rationale: the kernel speeds the step; deadline order lets chat through the agent burst; demote-on-miss stops
  already-late agents from making chat late
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
- Llama-3.3-70B (one 4-GPU node) [pending]; GLM-4.5-Air [pending]

### 6.4 Multi-node benchmarks

- Contribution: C3, symmetric nodes + SLO-checked cache-first routing + cluster KV pool
- Rationale: colocation keeps the agent prefix where it was computed; disaggregation re-ships it every round
- Schedule only (kernel off, no speculation), Qwen2.5-7B, 16 H20s, fan-out agents + chat; chat attainment / agent
  SLO goodput (tok/s)

  | agent programs | ours | best PD disagg (6 prefill + 2 decode pods) | conditional disagg |
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
- 32 H20s, same protocol: ours >= 96% through 128 programs; best disagg 83.7% at 32, 26.7% at 128
- Full stack (kernel + 4-mode MTP), higher load and mixture spikes, 16 and 32 GPUs [pending]
- Llama-3.3-70B (4 TP4 replicas on 16 H20s) [pending]; GLM-4.5-Air [pending]
- Cache pressure: working set 1M -> 8M KV tokens against a 6.7M pool; disagg loses 55% throughput and 28.5 points of
  cached fraction, ours 10.6% and 3.1 [earlier settings; rerun with the full stack pending]

### 6.5 Fault tolerance

- kill 2 of 8 GPUs at 300 s: ours reroutes (agent attainment 96.4 -> 98.1%, chat 85.0%), PD disagg loses half its
  prefill capacity (18 -> 34%, chat 9-15%) [earlier settings; rerun with the full stack pending]

## 7. Ablation study

- kernel: unified plan vs two plans; plan-scheduled JIT merge vs separate merges; LPT vs non-persistent; 48 tile; PDL vs serial
- FA2 rows vs FA3 rows (role split): equal throughput, FA2 rows 15-25 points better agent ITL
- schedule: per-tenant deadlines alone (no change) vs + demote-on-miss (chat TTFT -96..-98%); feasibility demotion at
  a fixed rate demoted every cold 128-256K agent on arrival
- per-step chunk caps for chat TPOT: starve prefill (binds on 86% of steps)
- router: cache-first + SLO check vs cache-only vs round-robin

## 8. Background and Related Work

- serving frameworks: vLLM, sglang, TRT-LLM; Dynamo
- disaggregation: DistServe, Mooncake, Splitwise; semi-PD (MPS SM split, restart to change), DynaServe, Dynamo
  conditional disaggregation
- colocation and multiplexing: Sarathi-Serve, PD-Multiplexing / MuxWise, POD-Attention
- kernel fusion: POD-Attention, persistent kernels, FlashAttention-3, flashinfer holistic scheduling, cross-SM fusion
  with JIT reduction ("Taming Dynamism on GPUs")
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
