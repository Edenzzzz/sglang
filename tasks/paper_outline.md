# AttentionFuse: SLO-Aware Prefill-Decode Colocation with Fused Attention and Cache-First Routing

Draft (2026-10-20), rebuilt from the 11/21 outline (`tasks/AttentionFuse Outline OSDI 25'.docx`) with the results
measured since. Every number traces to a dated entry (commit + command) in `debug_kernel_server_perf_4_mode.md`,
`realistic_workload_analysis.md`, `pd_colocation_vs_disagg.md` or `server_perf_model.md`. Model: Qwen2.5-7B-Instruct
with an EAGLE3 MTP head unless noted; single node = 1 H200, multi-node = 8 H20.

## Paper Outline (10/20)

- **goal** -- we need to meet per-tenant SLOs (chat: TTFT + per-token latency; agents: TTFT + TPOT) while maximizing
  goodput, for mixed chat + long-context agent traffic, with speculative decoding (MTP) on as production runs it --
  without fixing a prefill / decode split in hardware
- **motivation**
  - prefill-prioritized chunked prefill -- good throughput, but long prefills stall running decodes (agent P99 ITL 1.35 s)
  - mixed chunk -- decode rides the prefill step, but every row, 64-256K prefill rows included, runs through one FA2
    prefill kernel (~25% more attention time than splitting by role)
  - PD disaggregation -- decode isolation, but a few prefill nodes queue every tenant (TTFT 14-31 s at our loads),
    dilute each prefix across prefill nodes and re-prefill agent turns
  - PD multiplexing -- per-phase SM groups, but no CPU-GPU overlap scheduling and one CUDA-graph set per SM group
  - MTP raises decode-node utilization under disaggregation, undercutting "colocation for utilization" unless the
    colocated schedule is MTP-compatible and efficient -- MTP is in scope
- **design**
  - a fused step that runs prefill, decode, draft and verify attention side by side on pre-assigned SMs: FA3 (Hopper
    TMA / wgmma) for prefill tiles, FA2 for decode / verify tiles, FA2 overlapped beside FA3
  - the 4-mode MTP step: the prefill chunk rides the verify forward, inside the stock scheduler (overlap scheduler on,
    one CUDA-graph set)
  - why does this help? cheaper per prefill token than mixed chunk, no decode stalls as in prefill-prioritized, and
    EDF spends each step on deadlines
- **distributed inference**
  - meeting SLOs across nodes requires routing on predicted load, not on a fixed prefill / decode role
  - EDF needs a step-time estimate; the fused step admits a simple one, shared by scheduler and router
  - route each request to the node holding the longest cached prefix that can still meet every SLO; pull the rest
    from a cluster KV pool (best-effort replication); nodes report their load to the router
- **evaluation**
  - main workload: mixed TraceLab coding agents + ShareGPT chat (standard 32-64K and long 32-250K mixes); multi-node
    fan-out agents on shared repository prefixes + chat; standalone ShareGPT, math, code and agent / tool traces
  - single node vs prefill-prioritized and mixed chunk (with and without MTP), pdmux as a reference
  - multi-node vs PD disaggregation (with and without MTP on decode); semi-PD if time permits
  - per-tenant SLO attainment and goodput, error bars over repeats, load spikes and mixture shifts, a second model
  - fault tolerance (if time permits)
    - prefill-node failure propagates to decode nodes under disaggregation; colocated nodes absorb it

## Abstract

LLM serving fleets are disaggregated into prefill and decode nodes so that long prefills never stall decoding. The
split is fixed in hardware: it queues requests behind a few prefill nodes, dilutes cached prefixes across them, and
leaves decode compute idle -- which speculative decoding (MTP) now fills on the disaggregated side. We present
AttentionFuse, a colocated serving system in which every node runs prefill and decode together. Its step runs FA3
prefill tiles and FA2 decode / verify tiles on pre-assigned SMs inside sglang's scheduler, with the prefill chunk
riding the MTP verify forward (a 4-mode step) and an EDF scheduler that sizes each chunk to deadlines. Across nodes, a
cache-first router sends each request to the node holding the most of its prefix that can still meet every SLO,
pulling the rest from a cluster KV pool. On mixed coding-agent + chat traffic over 8 H20s, AttentionFuse meets chat
SLOs for 91% of requests vs 26% under PD disaggregation and delivers 6x the agent goodput, holding 97-100% agent
attainment through a 3x chat spike and a doubled agent load; with a 14B model disaggregation collapses (agent TTFT
~290 s). On one H200 with MTP it raises chat SLO attainment by 72%, agent attainment 15x and throughput 55% over
sglang's mixed chunk.

## 1. Introduction

- **Setting / problem**
  - production traffic mixes interactive chat (tight TTFT and per-token SLOs) with coding / tool agents whose prompts
    run 32-250K tokens, share long prefixes across turns and programs, and arrive in bursts
  - MTP speculative decoding is standard and multiplies decode-side work per step
  - goal: meet per-tenant SLOs at the highest request volume per GPU, and adapt to load spikes and mixture shifts
    without re-provisioning
- **Why existing systems do not meet it**
  - prefill-prioritized and mixed-chunk colocation: decodes either stall behind long prefills or pay a slow one-kernel
    step; neither orders work by deadline
  - PD disaggregation: decode isolation at the cost of prefill queueing, prefix dilution and per-turn re-prefill; the
    P:D ratio is fixed, so a spike or a mixture shift lands on whichever side is short
  - PD multiplexing: per-phase SM groups, but loses the CPU-GPU overlap scheduler and multiplies CUDA graphs by groups
- **Key insight** -- the isolation disaggregation buys can be had per step without fixing the split: one fused step
  runs each attention role on the kernel and SMs it suits, a deadline-driven scheduler decides how much prefill each
  step carries, and symmetric nodes let the router follow the cache instead of a role table
- **Contributions** (section 4): (1) SM-pre-assigned FA3 + FA2 split attention; (2) the 4-mode MTP step; (3) EDF
  scheduling for the fused step with a shared step model; (4) cache-first, SLO-checked multi-node routing over a
  cluster KV pool
- **Results** (section 6) -- 8 H20s vs PD disaggregation: chat SLO attainment 91% vs 26%, agent goodput 6x, robust to
  a 3x spike, a 2x agent shift and a 14B model; 1 H200 with MTP: +72% chat attainment, 15x agent attainment, +55%
  throughput over mixed chunk

## 2. Background

### 2.1 LLM serving and speculative decoding
- prefill is compute-bound and parallel; decode reads all of a request's KV per token and is bandwidth-bound
- MTP / EAGLE3: a draft head proposes tokens, the target verifies them in one forward (acceptance ~2.0-2.1 tokens per
  verify step on our workloads); a step = draft steps + a verify forward

### 2.2 Colocated schedules in sglang
- prefill-prioritized chunked prefill (the default): prefill steps run alone, decodes wait
- mixed chunk: decode rows ride the prefill step, all through FA2 BatchPrefill
- PD multiplexing (MuxWise / pdmux): GreenContext SM groups per phase, layer-split prefill; overlap scheduler off;
  one CUDA-graph set per SM group (5 on H20, 8 on H200)

### 2.3 PD disaggregation
- DistServe, Mooncake; sglang prefill / decode roles with mooncake KV transfer; MTP on decode nodes
- NVIDIA Dynamo: KV-aware router, conditional disaggregation
- static P:D provisioning: a spike or mixture shift hits whichever side is under-provisioned

### 2.4 KV-cache-centric scheduling
- multi-turn and agent workloads reuse long prefixes; cache-aware routing and a cluster KV pool (sglang HiCache +
  mooncake store) cut TTFT; ThunderAgent documents repeated cache thrashing in agentic serving

## 3. Challenges and motivation

### 3.1 Attention dominates the mixed step at long context
- a 743-token chunk at 50K depth is ~28 ms of attention in a 44 ms mixed step (H200); FA3's chunk call already runs at
  ~540 TFLOP/s -- the step is attention-FLOP-bound, so the kernel each row runs on decides the step time

### 3.2 Mixed chunk pays for its kernel choice
- decode rows padded into prefill tiles waste only 6.3% / 2.7% / 0.6% of FLOPs at 32K / 64K / 128K
- running 64-256K prefill rows through FA2 is the cost: attention per 60 steps at 64K, stock 17.7 s vs split 13.3 s
  (-25%); at 128K the merged batch takes 3.15 ms per call vs 0.83 ms for the split's FA2 calls on mixed steps

### 3.3 Prefill-prioritized scheduling breaks decode SLOs
- standard mix: agent P99 ITL 1.35 s, chat TPOT 34.8 ms (vs 20.7 ms for ours without MTP)

### 3.4 Disaggregation queues, dilutes and re-prefills
- 8 H20s, fan-out agents: agent TTFT 16-31 s; prefill hit rate falls 0.125 -> 0.055 going from 2 to 4 pods while
  colocation's rises 0.34 -> 0.39; re-prefill share 10.3% vs 3.4%

### 3.5 Speculative decoding is concurrency-limited
- MTP serves best at sglang's speculative default of 48 running requests for every system (64 and 128 slow every
  step), so the schedule has to make each step count

## 4. AttentionFuse design

**Our main contributions are:**

1. **SM-pre-assigned FA3 + FA2 split attention (kernel item 1).** Each mixed step's work items are dispatched by role:
   prefill tiles to FA3 on a reserved SM budget (flash-attention's Hopper kernel: paged TMA, wgmma, dynamic
   scheduler), decode and verify tiles to FA2 (tile classes 64/32/16, auto split-KV) on the remaining SMs, with FA2
   launched beside FA3 on a side stream. Both roles are exact against a causal fp32 reference.
2. **The 4-mode MTP step (kernel item 2).** Prefill chunk, decode, draft and verify in one target forward: the chunk
   rides the verify forward instead of a separate prefill step, inside sglang's own scheduler -- the CPU-GPU overlap
   scheduler stays on and one CUDA-graph set serves all steps.
3. **EDF scheduling for the fused step.** Deadlines on uncached tokens (base 2 s + 1 s per 5K tokens beyond 4096,
   doubled above 200K), deadline-driven chunk admission (a small chunk normally, a large one when the backlog would
   miss a deadline at its pace) and chunk preemption for earlier-deadline arrivals. A closed-form step model of the
   fused step predicts it at half the error of stock MTP's step and is shared with the router.
4. **Cache-first, SLO-checked multi-node routing.** Symmetric nodes; the router sends each request to the node with
   the longest cached prefix whose predicted TTFT and ITL -- for this request, the earlier-deadline queue ahead of it
   and the decodes it joins -- stay within SLO, else the next. Misses pull the prefix from a cluster KV pool
   (best-effort replication over a complete RDMA graph). Nodes report decode rows, KV, acceptance and their
   deadline-ordered prefill queue through sglang's LoadStat channel.

### 4.1 Problem formulation
- per-tenant SLOs: chat TTFT 0.5 s and per-request TPOT 20 ms; agents TTFT 10 s and TPOT 80 ms; maximize goodput
  (output tokens of requests meeting both) per GPU, under bursty, shifting mixtures

### 4.2 Split attention with SM pre-assignment
- dispatch: FA3 for prefill-shaped items, FA2 for every verify-width / decode item (FA2 wins every verify-width row
  in the winner map, 216/216 cells exact)
- FA3 role = flash-attention's kernel, at parity with sgl-kernel FA3; FA2 runs beside it on the remaining SMs

### 4.3 The 4-mode MTP step
- the scheduler attaches the running batch to the prefill step; the spec worker drafts, merges the chunk with the
  verify rows and splits attention by role; a context / queue gate keeps short prefills on their own step

### 4.4 EDF scheduling and the step model
- deadline order decides which prefill a step carries; chunk size follows the deadline backlog
- step model: linear in prefill tokens, prefix attention work, #prefill requests, decode rows, decode KV and the
  draft / verify terms; fit on serving step traces

### 4.5 Multi-node routing and the KV pool
- SLO check: predicted TTFT = EDF queue drain ahead + own prefill (cold-prefill curve, pull-adjusted); ITL from the
  calibrated decode grid with this request and earlier-deadline requests resident; adaptive ITL SLO = 2.5x the
  observed TPOT
- KV pool: mooncake L3 store spanning all nodes; replication happens as a side effect of serving
- node load: LoadStat extended with decode rows, decode KV, acceptance, deadline-ordered prefill queue and measured
  decode-step time (ZMQ PUB, change-triggered, 1 s heartbeat)

## 5. Implementation

- sglang (upstream 3fef129 + ~26 files): split dispatch in the flashinfer backend, the 4-mode step in the EAGLE3
  worker, EDF scheduler with deadline chunk and preemption, per-step trace logging
- FA3 role on flash-attention's kernel (paged TMA, `-DNDEBUG`, fast math); FA2 from flashinfer 0.7
- router: aiohttp proxy with the SLO check, cluster prefix index, pull-bandwidth EWMA and LoadStat subscriber
- KV pool: sglang HiCache with the mooncake store backend over RDMA

## 6. Evaluation

### 6.1 Settings

**Workloads** (main workload first):

| workload | composition | load axis | where |
|---|---|---|---|
| **mixed TraceLab + chat (standard)** | TraceLab coding-agent sessions, 32-64K prompts, 2 rounds + ShareGPT chat | agent program concurrency C=8; chat 8 req/s | 1 H200 |
| **mixed TraceLab + chat (long)** | TraceLab 32-250K prompts (4 rounds > 200K) + ShareGPT chat | C=6; chat 2 req/s | 1 H200 |
| **fan-out agents + chat** | 16 tasks x 64K shared repository prefixes, 4 rounds, rotating Zipf popularity + ShareGPT chat | 32 agent programs; chat 8 req/s | 8 H20 |
| standalone ShareGPT | short chat prompts | 4-32 req/s | 1 H200 |
| standalone math | GSM8K, MATH-500 | 8-32 req/s | 1 H200 |
| standalone code | HumanEval, Code-Feedback | 8-32 req/s | 1 H200 |
| standalone agent / tool traces | Mooncake conversation and tool-agent (recorded timestamps), WebShop browsing agent | 0.5-2x trace speed; C | 1 H200 |

- **Models (so far)**: Qwen2.5-7B-Instruct + EAGLE3 (3 draft steps, 4 verify tokens); Qwen2.5-14B-Instruct (no MTP)
- **Models (proposed for 4 nodes x 8 or x 4 H200 / H100, one TP8 / TP4 instance per node)** -- sized so that a node
  holds tens of 32-250K agent contexts beside chat decodes, the batch shape the 4-mode step targets; MTP simulated with
  `SGLANG_SIMULATE_ACC_LEN` (acceptance fixed to the measured ~2.0-2.1, real draft and verify work per step). KV room
  per node after weights, 0.88 memory fraction:

  | model | KV / token / node | 8 x H200 | 8 x H100 | 4 x H200 | 4 x H100 |
  |---|---|---|---|---|---|
  | **Llama-3.3-70B-Instruct** (dense, 128K, 80 layers, 8 KV heads) | 320 KB | 2.4M tok | 1.1M | 1.0M | 0.36M (tight) |
  | **GLM-4.5-Air** (MoE 106B-A12B, 128K, 46 layers, 8 KV heads, native MTP layer) | 184 KB | 3.8M | 1.6M | 1.3M | no room |
  | Qwen3-235B-A22B-2507 (MoE, 262K; 4 KV heads, replicated 2x at TP8) | 376 KB | 1.2M | no room | no fit | no fit |

  - both main picks have 8 KV heads, so TP8 holds one KV head per GPU with no replication
  - 4 replicas (one per node), the same router fan-out as the H20 runs' 4 pods
  - context forced past 128K for the 250K long mix: `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 --context-length
    262144`, rope unchanged (keep `rope_theta` if a YaRN override is used); attention cost depends only on length and
    output lengths are fixed by the workload, so quality past 128K does not affect the measurement
  - 4 x H100 (320 GB / node) fits neither main pick with useful KV; there only a <= 32B dense model works
- **Testbed**: 1 H200; 8 H20 (4 pods x 2 GPUs, one TP1 worker per GPU)
- **Baselines**: sglang prefill-prioritized and mixed chunk, each with and without MTP (single node); pdmux
  (reference); PD disaggregation 2P2D with and without MTP on decode (multi-node); semi-PD (planned)
- **Metrics**: per-tenant SLO attainment (TTFT and per-request TPOT both met) and goodput; TTFT, TPOT, throughput;
  mean +- std over repeats

### 6.2 Kernel microbenchmarks
- FA3 role at parity with sgl-kernel FA3 (exact vs fp32); FA3 co-running with FA2 -18% on balanced append + verify
  steps
- in serving, overlapping FA2 beside FA3 is worth +10% agent throughput and -12% / -16% agent / chat ITL over running
  the two roles back to back

### 6.3 Main workload, single node (1 H200, mixed TraceLab + chat, 5-6 repeats)
| arm | agent tput (tok/s) | agent attainment | chat attainment |
|---|---|---|---|
| stock mixed chunk | 80.5 +- 4.2 | 44.1% | 53.8% |
| stock prefill-prioritized | 92.6 | 58.9% | 59.9% |
| **ours, no MTP** | **113.2 +- 0.4** | **92.0%** | **69.8%** |
| stock + MTP | 65.7 +- 3.1 | 3.0% | 34.2% |
| stock prefill-prioritized + MTP | 81.6 | 8.9% | 49.2% |
| **ours + MTP** | **102.1 +- 1.3** | **45.8%** | **58.9%** |
- vs mixed chunk: MTP +55% throughput, 15x agent and +72% chat attainment; no MTP +41%, +109%, +29.7%
- vs prefill-prioritized: +22% / +25% throughput, agent attainment +56% / 5x, chat attainment +16.5% / +20%
- long mix (32-250K), ours + MTP vs stock + MTP, two repeats: agent throughput +54%, TTFT -34%, ITL -49%; chat
  throughput +23%, TTFT -90%, ITL -42%
- our arms vary by <= 1.4 tok/s across repeats; stock mixed chunk is bimodal (slow mode in 5 of 6)

### 6.4 Main workload, multi-node vs PD disaggregation (8 H20s, fan-out agents + chat)
| arm | chat attainment (0.5 s + 20 ms) | agent goodput (10 s + 80 ms) | agent TTFT |
|---|---|---|---|
| **ours** | **91.4%** | **551 tok/s** | 0.71 s |
| PD disagg 2P2D | 26.2% | 92 tok/s | 16.4 s |
- throughput 2.3-3.3x, TTFT -94..-96%; disaggregation keeps a faster per-token decode (5 ms vs 12 ms), but its prefill
  queue makes most requests miss
- cache: re-prefill share 3.4% vs 10.3%; prefill hit rate rises with node count for colocation, falls for disagg

### 6.5 Dynamic load: spikes and mixture shifts (8 H20s, two repeats)
| event | ours: agent attainment through the event | ours: chat | disagg: agent | disagg: chat |
|---|---|---|---|---|
| chat x3 for 240 s | 99-100% | 88-91% | 36-57% | 19-22% |
| agent load x2 for 240 s | 97-100% (agent goodput ~2x) | ~84% | 21-31% | 18-22% |

### 6.6 Standalone workloads (1 H200, MTP vs ours + MTP)
- agent / tool traces (Mooncake conversation, tool-agent): TTFT -25..-31% below capacity; at 2x trace speed
  throughput +12% / +21%, TTFT -47% / -58%, P99 ITL -59% / -60%
- ShareGPT, math (GSM8K, MATH-500), code (HumanEval, Code-Feedback): on par below capacity (short prompts carry little
  prefill to fuse); at saturation +2-3% throughput, TTFT -5..-15%
- WebShop browsing agent (short multi-turn prompts, one GPU): behind stock MTP (throughput -7..-10%)
- the gains concentrate where long-context prefill meets decode, which is the mixed agent + chat setting above

### 6.7 Second model (Qwen2.5-14B, 8 H20s, fan-out)
- ours: agent 186-216 tok/s, chat attainment 45-51%, agent goodput 102-131 tok/s
- PD disaggregation collapses: agent TTFT ~290 s, agent goodput 0, chat attainment 11-13%
- at half load (scaled to 14B's capacity, two repeats): ours chat attainment 80% vs 9-17%, agent goodput 133-143 vs
  0-1 tok/s, agent TTFT 2.2 s vs 110-118 s
- next: Llama-3.3-70B and GLM-4.5-Air on 4 nodes x 8 / x 4 H200 / H100 (section 6.1)

### 6.8 Step-model accuracy (EDF)
| schedule | closed-form model (MAPE) | best model, gradient-boosted trees (MAPE) |
|---|---|---|
| ours + MTP (4-mode) | 12-13% | 3.0-3.5% |
| stock + MTP | 24-27% | 2.8-4.9% |
| ours, no MTP | -- | 3.2% / 5.9% (standard / light load) |
| pdmux | 19.8-24.0% (its own form) | 5.0% / 7.5% |

### 6.9 Fault tolerance (if time permits)
- xPyD: kill a prefill node -> TTFT and ITL rise on every decode node fed by it; colocated nodes re-route to any
  survivor

## 7. Ablations

- FA2 overlapped beside FA3 vs back to back: +10% agent throughput, -12% / -16% agent / chat ITL
- split attention vs one FA2 kernel for the mixed step: -25% attention time at 64K
- full MTP stack (4-mode step + split + EDF) vs stock MTP: agent attainment 3.0% -> 45.8%, chat 34.2% -> 58.9%
  (a per-component MTP ablation is pending)
- cache-first SLO-checked routing vs PD roles: re-prefill 10.3% -> 3.4%, agent goodput 6x
- remove CUDA graphs (pending)

## 8. Related work

- **PD disaggregation**: DistServe, Mooncake, NVIDIA Dynamo (conditional disaggregation, KV router)
- **semi-disaggregation and multiplexing**: semi-PD (MPS SM split, restart to resize), DynaServe, MuxWise / sglang pdmux
  (SM groups per phase) -- none fuses attention across phases; pdmux loses the overlap scheduler
- **SLO-aware scheduling**: SLOs-Serve (multi-SLO), JITServe (imprecise request information)
- **agentic serving**: ThunderAgent (program-aware, cache thrashing), Crossflow (P/D elasticity), TraceLab (coding-agent
  workloads)
- **kernel fusion**: POD-Attention, persistent kernels, Taming Dynamism (cross-SM fusion, just-in-time reduction) --
  we pre-assign SMs per role inside one step
- **attention-FFN disaggregation** (Step-3, MegaScale-Infer): orthogonal, combinable

## 9. Conclusion and limitations

- **Conclusion**: colocated serving with a fused 4-mode step, EDF and cache-first routing meets per-tenant SLOs at
  several times the goodput of PD disaggregation, holds them through load spikes and mixture shifts, and stays
  MTP-compatible -- without a fixed prefill / decode split
- **Limitations**
  - single node vs prefill-prioritized: chat SLO attainment +16.5% / +20%; chats decode beside continuous long-context
    agent prefill
  - per-token decode: disaggregation's decode-only nodes keep 5 ms TPOT vs our 12 ms; the gain is in attainment and
    goodput, not raw decode speed
  - short-prompt standalone workloads see little gain; the step helps where long-context prefill meets decode
  - MTP is concurrency-limited at 48 running requests for every system
  - open: Llama-3.3-70B / GLM-4.5-Air on 4 nodes with simulated MTP, semi-PD baseline, fault tolerance, Blackwell kernels
