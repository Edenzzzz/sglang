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

## [2026-10-22] THE ABSTRACT AND SECTION 4 BELOW ARE STALE -- read this first

Two measurements since the outline was written change what the paper can claim. Nothing below has been edited yet;
this block is what a co-author should believe.

**1. The single-node baseline was wrong, in our favour, and the corrected one is much stronger.** Every single-node
number below (and the abstract's "+72% chat attainment, 15x agent attainment, +55% throughput on one H200") compared
ours against stock sglang forced onto the **flashinfer** attention backend. Upstream's own default on Hopper for these
models is **FA3** (`get_default_attn_backend`, citing a flashinfer Hopper regression). With the identical schedule and
load and only the backend changed, stock measures **161.2 agent tok/s on FA3 and 81.1 on flashinfer** -- the backend
we built inside is half the speed. Against the correct baseline our best single-node configuration is **-3.4%
throughput, +9 points agent SLO attainment, -78% chat TTFT, -1.5 points chat attainment**, and with MTP -16%
throughput. The >=30% / >=10% bar is **not met on single node** as of this date. The definitive five-schedule table,
every arm on FA3, is `tasks/pd_colocation/h200_fa3_table_20261022.sh`.

**2. The fused FA3+FA2 kernel is not the mechanism the title implies.** Once the decode / verify rows also run on FA3
(they are ~8x faster there than on flashinfer's BatchPrefill), **no FA2 role remains in serving**. What survives of
"fusion" is a per-step decision to split one varlen FA3 call into two co-running launches, which is worth 4-25% on
kernel microbenchmarks (up to 77% when a short chunk sits beside long-context rows) and **nothing measurable in
serving** -- forcing every step to a single call changes throughput by 0.2%. Multicast is closed as a negative
(-0.2..+1.7%) and in-kernel cross-role backfill by bound. Meanwhile the measured serving cost sat in steps the kernel
work cannot touch: a large share of serving forwards carry ~164 prefill tokens and zero decode rows, where attention
is negligible and the backend class's per-step overhead dominates.

**What the paper's contribution actually is, on current evidence:** the *schedule* and the *routing*, not the kernel.
Both are backend-independent -- they live in `scheduler.py` and the router -- and the multi-node results, with both
systems on FA3 and disagg swept over P:D ratios, clear the bar by large margins (+56% agent throughput at 32 GPUs /
1x load, +164% at 2x, 3.5x with Qwen3-8B + real MTP at 16 GPUs, agent TTFT 6-18x lower). The honest framing is a
colocated scheduler plus cache-first router whose advantage grows with cluster size and load, with the kernel work
reported as what it measured: an attention-backend choice worth 1.8x on this workload, and a split whose gains are
real in isolation and absent in serving. Ours' ITL tails remain worse than disagg's (1.5-2.5x at 32 GPUs), which
belongs in the limitations.

## Abstract

[2026-10-22 REWRITE -- every number below is from a run with both systems on sglang's FA3 backend; the previous
abstract's single-node claims were measured against stock forced onto the deprecated flashinfer backend and are
withdrawn.]

LLM serving fleets are disaggregated into prefill and decode nodes so that long prefills never stall decoding. The
split is fixed in hardware: it queues requests behind a few prefill nodes, dilutes cached prefixes across them, and
makes every round of a multi-turn agent session re-ship or recompute its KV. We present AttentionFuse, a colocated
serving system in which every node runs prefill and decode together: an EDF scheduler that orders by deadline and
sizes each prefill chunk to it, a 4-mode step in which a prefill chunk rides the MTP verify forward, and a
cache-first router that sends each request to the node holding the most of its prefix that can still meet every SLO,
pulling the rest from a cluster KV pool.

The case is a cluster one. On mixed coding-agent + chat traffic over 16-32 H20s, with PD disaggregation swept over
its prefill:decode ratios and both systems on the same attention backend, AttentionFuse delivers 1.9-3.8x the agent
throughput at 6-18x lower agent TTFT, and the gap *widens* with pressure: doubling offered load takes it from +56% to
+164%, and growing the agent working set 8x past the cluster KV pool takes it from +91% to +277% while our cached
prefix fraction falls only 0.98 -> 0.95 against disaggregation's 0.89 -> 0.61. Disaggregation retains a better
inter-token-latency tail throughout, because its decode nodes never see a prefill.

Two findings qualify the design. First, the fused FA3+FA2 attention step is not where the throughput comes from: once
decode and verify rows also run on FA3 no FA2 role remains, and the surviving per-step choice -- splitting one varlen
call into two co-running launches -- is worth 4-25% on kernel microbenchmarks and nothing measurable in serving. Its
measured value is instead *predictability*: our 4-mode step's GPU time is modelled to 0.4% / 3.2% MAPE (decode /
prefill-bearing steps) against 1.5% / 5.7% for stock MTP, and 2.7-4.5x better than the same schedule on an
unmodified backend, which is what the EDF admission decisions consume. Second, on a single GPU the schedule
*redistributes* latency rather than reducing it: against sglang's default we cut chat TTFT 77% and the agent ITL tail
71% while giving up 57% on agent TTFT and 5% on throughput, and no chunk policy escapes that frontier. We therefore
claim colocation as a cluster-scale and saturation-regime result, and report the single-GPU trade as a trade.

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
- at that cap the slot, not the GPU, is what new chats wait for: 38% of our chats miss TTFT queued behind it while the
  running ones decode at 13 ms/token against a 20 ms SLO

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
   miss a deadline at its pace) and chunk preemption for earlier-deadline arrivals. Decode rows are deadline-ordered
   too: more requests stay resident than the 48-row MTP graph holds, each step decodes the 48 whose next token is due
   soonest, and a request is admitted only while every resident row can still meet its TPOT SLO at the measured step
   time. A closed-form step model of the fused step predicts it at half the error of stock MTP's step and is shared
   with the router.
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
- deadline-ordered decode rows: a row that has emitted o tokens since it started decoding at t0 is due its next token
  at t0 + SLO x (o + 1); the K rows due soonest run, the others keep KV and draft state and sit out the step
- SLO-checked admission: a row with SLO s takes dt / (s x a) of a row slot per step (dt the measured step interval, a
  the tokens per verify); admit while the resident rows' sum plus the newcomer fits 0.9 x K
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

### 6.0 Evaluation flow, baselines and the claim at each stage [2026-10-23]

The evaluation is three stages, each answering one question against the strongest baseline we can build for that
stage, and each feeding the next: a kernel that wins on the shapes serving actually produces, a single node whose
schedule and step model use that kernel, and a cluster whose router uses that step model. Every number below is
measured with the baseline on the SAME attention backend as ours (sglang's FA3, its Hopper default); results measured
against stock forced onto the flashinfer backend are withdrawn (see the note at the top of this file).

| stage | question | baseline(s) | our claim | headline evidence | status |
|---|---|---|---|---|---|
| **1. Kernel** (6.2) | is role-specific FA3 + FA2 faster than FA3 on the mixed steps serving produces? | single varlen FA3 (sglang default), dual-stream FA3, SM-grid-tuned FA3; all tuned per shape | **C1**: heterogeneous attention with PDL wins on decode/verify-heavy mixed steps | 1.45-1.78x on captured small-query steps (Qwen2.5, 32K-394K); 1.45-1.67x on Qwen3; ~1.0x prefill-heavy | measured; Llama-70B capture missing |
| **2. Single node** (6.3) | does the kernel and the 4-mode schedule pay off in serving on one GPU? | sglang on FA3: mixed chunk, prefill-prioritized, each +/- MTP | **C2a** kernel at fixed schedule; **C2b** step predictability; C2c schedule as a latency trade | kernel: +10.6-11.3% agent tput, -21.5% agent ITL at 128-256K; step model 3.8x more accurate | C2a meets the 10% tput bar at long context only; ITL bar not met |
| **3. Multi-node** (6.4) | does colocation + cache-first SLO routing beat PD disaggregation at scale? | PD disagg, P:D swept at worker granularity, KV pool on, same FA3 backend, with and without MTP | **C3**: higher sustainable load at SLO (capacity), widening with cache pressure and scale | capacity-at-SLO sweep running; overload point: +179% tput, thrashing lead +91% -> +277% | 16 GPUs: capacity sweep running; 32 GPUs: same protocol next |

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
- **Baselines, per stage, all on sglang's FA3 attention backend** (its Hopper default for these models):
  - kernel: one varlen FA3 call over all rows (what sglang runs); dual-stream FA3 (prefill and rows on separate
    launches); FA3 with SM-grid margins 0-72 and row/prefill SplitKV and GQA-packing searched per captured shape
  - single node: sglang mixed chunk (default) and prefill-prioritized, each with and without EAGLE3 MTP; the kernel is
    also isolated by running sglang's OWN schedule with only the attention swapped
  - multi-node: sglang PD disaggregation with its own router, prefill:decode ratio swept (2+2, 3+1 at 16 GPUs; 4+4,
    5+3, 6+2 at 32 GPUs), with and without MTP on every role
- **Scale axis (multi-node)**: 16 GPUs (4 nodes x 4 H20) and 32 GPUs (8 nodes x 4 H20), one TP1 worker per GPU for
  the 7-8B models, load scaled with GPU count
- **Metrics**: per-tenant SLO attainment (TTFT and per-request TPOT both met) and goodput; TTFT, TPOT, throughput;
  mean +- std over repeats

### 6.2 Stage 1 -- kernel microbenchmarks on captured serving tensors (H20)

**Baselines.** Single varlen FA3 (sglang's own call), dual-stream FA3, and FA3 with SM-grid margins; each control
re-tuned per shape over row/prefill SplitKV (0-64), GQA packing (auto/off/on) and margins 0-72 of 78 SMs. The best
control is kept as a separate arm in validation so a poor new selection cannot hide a stronger baseline.
**Ours.** A native executor that runs prefill tiles on FA3 and decode/verify rows on FA2, chained with programmatic
dependent launch (PDL), with a geometry-only split policy (v3) chosen before validation.
**Inputs.** Real Q/K/V captured from serving after the workload barrier (not synthetic fixtures): Qwen2.5-7B and
Qwen3-8B, two-mode (prefill + decode) and four-mode (prefill + EAGLE3 verify, measured acceptance 1.8-3.5 tokens),
physical KV page IDs and strides restored; 29 captures, 32K-394K contexts. Every record passes a full-output FP32
causal-attention check (max head-relative error 0.0046-0.0078).

| captured step | best tuned FA3 (ms) | ours, native PDL (ms) | speedup |
|---|---:|---:|---:|
| Qwen2.5 two-mode, 32-64K, fresh / small query | 0.2016 | 0.1350 | **1.47x** |
| Qwen2.5 two-mode, ~394K, fresh / small | 0.9334 | 0.5360 | **1.74x** |
| Qwen2.5 four-mode, 128-256K, cached / small | 5.4609 | 3.0634 | **1.78x** |
| Qwen2.5 four-mode, ~394K, fresh / small | 0.9355 | 0.7510 | **1.23x** |
| Qwen2.5 two-mode, 32-64K, cached / small (after v3) | 0.8235 | 0.7358 | **1.12x** |
| Qwen3 two-mode, fresh / small | 0.3986 | 0.2503 | **1.59x** |
| Qwen3 four-mode, cached / small (after v3) | 3.5974 | 2.1576 | **1.67x** |
| prefill-heavy steps (2046 prefill tokens, 2 rows) | -- | -- | 0.98-1.09x |

- **C1 claim**: on the mixed steps where decode/verify rows meet deep context, role-specific attention is 1.2-1.8x
  faster than the best FA3 we can tune per shape; where the prefill chunk dominates it is neutral (0.98-1.09x).
- Where the speedup comes from: most of it is role-specific attention and its split plan; PDL adds only 1-12% over
  the same plan run serially. Report it that way rather than as "fusion".
- Honest limits: four small four-mode Qwen2.5 cases still regress 3-4%; GLM-4.5-Air four-mode regresses 2-58%
  (`three_model_screen_20261023.md`), so GLM is not a C1 model; Llama-3.3-70B has not been captured yet.
- Sources: `resume_20261012/captured_kernel_results_20261023.md`, `geometry_policy_evaluation_20261023.md`,
  `pd_colocation/results_20261023/captured_sm_controls/captured_native_all29.{svg,csv}`.

### 6.3 Stage 2 -- single node (1 H200)

**C2a: the kernel in serving, at a fixed schedule.** Both arms run sglang's own schedule on the same frozen FA3
backend (2048-token chunks, 48-request cap, CUDA graphs, overlap on); only the attention differs. Opposite-order
pairs, audited from raw token-arrival events. Qwen2.5-7B, mixed TraceLab agents + ShareGPT chat:

| context / mode | agent tok/s | agent ITL (token-weighted) | agent TPOT | chat TPOT |
|---|---:|---:|---:|---:|
| 128-256K, two-mode | **+11.3%** | **-21.5%** | -23.9% | -19.5% |
| 128-256K, four-mode (MTP) | **+10.6%** | -11.6% | **-24.9%** | **-24.9%** |
| 32-64K, two-mode | +6.1% | -1.1% | -1.8% | -5.5% |
| 32-64K, four-mode (MTP) | +3.1% | -11.9% | -13.6% | -4.7% |

- Claim: the kernel alone clears the 10% throughput bar at long context and cuts TPOT ~25%; at 32-64K it is worth
  3-6%. Source: `resume_20261012/native_kernel_serving_attribution_20261023.md`. Qwen3-8B 32-64K: +9.2% / +7.9%
  throughput, ITL -1.0% / -4.1% (one four-mode repeat regressed ITL, so no-regression is not yet established).

**C2b: the fused step is predictable, which EDF admission consumes.** GPU time per step, first-half fit /
second-half scored, same features: ours + MTP on the fused path 0.4% decode / 3.2% prefill-step MAPE vs stock + MTP on
FA3 1.5% / 5.7% (3.8x / 1.8x better); the same 4-mode schedule on unmodified FA3 is 2.7-4.5x worse, so the fused
split is what buys the predictability (`server_perf_model.md`).

**C2c: the colocated schedule on one GPU is a latency trade, not a win.** Against sglang mixed chunk on FA3 at agent
concurrency 8 + chat 8 req/s: chat TTFT -77%, agent ITL p99 -71%, chat ITL p99 -62%, but agent TTFT +57% and
throughput -5.5%; against prefill-prioritized the chat ITL tail is 4x worse. At saturation (agent concurrency 16-24)
it does not beat stock even at its best chunk. Report this as the motivation for stage 3: the schedule's advantage
needs a routing dimension a single GPU does not have. Source: `realistic_workload_analysis.md`.

> Superseded: the earlier 6.3 table (ours +41-55% throughput, 15x agent attainment over mixed chunk) compared against
> stock forced onto the flashinfer backend, half the throughput of FA3 on this workload. It must not be quoted.

### 6.4 Stage 3 -- multi-node vs PD disaggregation, 16 and 32 GPUs (H20)

**Baseline.** sglang PD disaggregation with its own router, prefill:decode ratio swept, same FA3 backend as ours,
with and without MTP on every role. **Ours.** Colocated workers, worker-side EDF, cache-first SLO-checked router,
cluster KV pool. Workload: fan-out coding agents (64K repository prefixes, multi-round) + ShareGPT chat, load scaled
with GPU count. Per-tenant SLOs: chat 0.5 s TTFT + 20 ms TPOT, agent 10 s + 80 ms.

**16 GPUs (4 x 4), every arm on FA3** (32 tasks / 64 agent programs, chat 16 req/s, 900 s):

| arm | agent tok/s | agent TTFT | chat attainment | agent SLO goodput | chat TPOT p90 |
|---|---:|---:|---:|---:|---:|
| **ours, no spec** | **962.7** | **0.89 s** | **96.5%** | **939 tok/s** | 11.5 ms |
| **ours, no spec, fused kernel on** | -- | -- | 94.7% | **1063 tok/s** | 11.6 ms |
| ours + MTP, fused kernel on | INVALID RUN (chat client wrote 0 bytes, 2 of 5 agent epochs; 0 worker tracebacks; cause unknown, rerun queued) | | | | |
| PD disagg 3+1 | 344.9 | 17.24 s | 23.3% | 129 tok/s | 13.3 ms |
| PD disagg 2+2 | 242.3 | 27.57 s | 15.6% | 56 tok/s | 7.4 ms |
| **ours + MTP** | **1120.7** | **0.80 s** | **92.8%** | **1099 tok/s** | 9.5 ms |
| PD disagg 3+1 + MTP | 204.6 | 31.81 s | 3.8% | 22 tok/s | 13.8 ms |
| PD disagg 2+2 + MTP | 195.9 | 34.25 s | 4.9% | 25 tok/s | 6.5 ms |

- **Do not headline these ratios.** The load was sized to OUR capacity, so disagg is past saturation (chat
  attainment 3.8-23%) and goodput ratios of 7-44x divide by a near-zero baseline; the P:D sweep was also pod-granular
  (only 1+3 / 2+2 / 3+1 exist at 4 x 4) and disagg ran without the KV pool ours had. A reviewer will reject all three.
  The C3 number for the paper is **capacity at SLO** -- the highest load each system holds at >= 90% per-tenant
  attainment, from a load sweep, with P:D swept at worker granularity and the pool on both sides
  (`pd_colocation/mn_capacity_16g_20261023.sh`, running on 8 pods x 2 GPUs). The table above stays as the overload
  point of that curve.
- What the table does support: at this load, vs disagg's best ratio, chat attainment 96.5% vs 23.3% (no spec) and
  92.8% vs 4.9% (MTP), throughput +179% / 5.2x, agent TTFT -95% / -98%. MTP helps ours (+16%) and hurts disagg, which ships draft
  state to decode pods every round on top of re-prefilling the session.
- The colocated schedule carries this on its own -- the no-spec row runs upstream's unmodified FA3 -- and the fused
  kernel adds +13% agent SLO goodput on top (1063 vs 939 tok/s), the stage-1 kernel showing up at cluster scale.
- Disagg 2+2 keeps a tighter chat TPOT tail (6.5-7.4 ms) by starving agents, at 4.9-15.6% attainment.

**32 GPUs (8 x 4)**, disagg on FA3 (ours on its earlier backend, i.e. a conservative comparison), Qwen2.5 no spec:

| load | ours agent tok/s | disagg 5+3 | disagg 4+4 | ours agent TTFT | disagg 5+3 TTFT |
|---|---:|---:|---:|---:|---:|
| 1x (32 programs, chat 8 req/s) | 596.3 | 382.5 | 324.5 | 0.74 s | 4.59 s |
| 2x (64 programs, chat 16 req/s) | **1172.2** | 443.7 | 393.2 | **0.72 s** | 12.25 s |

- The lead widens with load (+56% -> +164%): ours doubles throughput at flat TTFT, disagg grows 16% and its TTFT triples.
- Disagg was also swept at 2 / 4 / 5 / 6 prefill pods of 8 at the original load; every ratio was overloaded.
- **Open (needed for the scale claim)**: the 32-GPU table must be rerun with ours on FA3 (both kernel off and on) and
  with MTP, matching the 16-GPU footing, so 16 -> 32 is a like-for-like scale sweep.

**Cache pressure (16 GPUs, no spec):** working set swept 1M -> 4M -> 8M live KV tokens against a ~6.7M pool at fixed
program concurrency. Ours loses 10.6% throughput and 3.1 points of cached prefix fraction (0.98 -> 0.95); disagg loses
55% and 28.5 points (0.89 -> 0.61). The lead grows +91% -> +266% -> +277%, and the cached fraction tracks it, so the
re-prefill mechanism is measured, not inferred (point 2, ThunderAgent's thrashing setting).

**What does not hold:** randomized pull among near-equal cache holders (point 4) shows no congestion regime at <= 16
H20s; the complete-graph vs bipartite argument is untested at the scale where it would matter.

> Superseded: the 8-GPU table formerly here (ours 91.4% vs 26.2% chat attainment) ran with disagg on the flashinfer
> backend and a single P:D ratio.

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

### 6.7 Larger models (fan-out, H20)
- **Llama-3.3-70B, TP4, 4 x 4 H20 = 4 replicas**, MTP (trained EAGLE3 head, acceptance fixed at 2.1), two repeats:
  ours chat attainment 77-89% vs 16-18% (2 s + 50 ms), agent goodput 59-69 vs 4-5 tok/s, agent throughput 3.5x, agent
  TTFT 2.5 s vs 41-43 s
- Qwen2.5-14B, TP1, 8 H20:
- ours: agent 186-216 tok/s, chat attainment 45-51%, agent goodput 102-131 tok/s
- PD disaggregation collapses: agent TTFT ~290 s, agent goodput 0, chat attainment 11-13%
- at half load (scaled to 14B's capacity, two repeats): ours chat attainment 80% vs 9-17%, agent goodput 133-143 vs
  0-1 tok/s, agent TTFT 2.2 s vs 110-118 s
- with MTP (random-weight draft head, simulated acceptance 2.1): ours chat attainment 73-77% vs 0.1-1.1%, agent
  throughput 117-121 vs 15 tok/s, agent TTFT 3.2-3.5 s vs 106-108 s
- next: Llama-3.3-70B and GLM-4.5-Air on 4 nodes x 8 / x 4 H200 / H100 (section 6.1)

### 6.8 Step-model accuracy (EDF)
| schedule | closed-form model (MAPE) | best model, gradient-boosted trees (MAPE) |
|---|---|---|
| ours + MTP (4-mode) | 12-13% | 3.0-3.5% |
| stock + MTP | 24-27% | 2.8-4.9% |
| ours, no MTP | -- | 3.2% / 5.9% (standard / light load) |
| pdmux | 19.8-24.0% (its own form) | 5.0% / 7.5% |

### 6.9 Fault tolerance (8 H20s, kill 2 of 8 GPUs at t = 300 s, two repeats)
| arm | agent attainment before / after | chat attainment (whole run) |
|---|---|---|
| **ours** (router marks the dead workers and reroutes) | **96.4 / 98.1%** | **85.0%** (91.4% without the fault) |
| PD disagg 2P2D (pod 1 = one of two prefill pods; health-check ejection) | 18 / 34% | 9.0-14.9% (26.2% without) |
- ours loses only the requests in flight on the killed pod (1-6 per run); survivors already hold the hot prefixes
- disaggregation loses half its prefill capacity; its chat attainment falls by 11-17 points

## 7. Ablations

- FA2 overlapped beside FA3 vs back to back: +10% agent throughput, -12% / -16% agent / chat ITL
- split attention vs one FA2 kernel for the mixed step: -25% attention time at 64K
- full MTP stack (4-mode step + split + EDF) vs stock MTP: agent attainment 3.0% -> 45.8%, chat 34.2% -> 58.9%
- MTP stack minus one component (two repeats): no split kernel -34% agent throughput, attainment agent 11% / chat 37%;
  no EDF agent attainment 16-20% (vs 47%), chat +0-3 points; no deadline chunk neutral at the standard mix
- deadline-ordered decode rows + SLO admission on top: chat attainment 58.9% -> 63.0%, chat TTFT -21%
- cache-first SLO-checked routing vs PD roles: re-prefill 10.3% -> 3.4%, agent goodput 6x
- CUDA graphs off: ours -20% throughput, stock MTP -4%; ours still leads (agent +29%, chat attainment 38% vs 23%)

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
  - single node vs prefill-prioritized at chat 8 req/s (the heaviest load measured): chat SLO attainment +16.5% (no MTP)
    / +28% (MTP, 63.0% vs 49.3%; the only point below +30%); about a
    quarter of chats decode beside long-context agent prefill chunks and miss a 20 ms TPOT under every schedule
  - per-token decode: disaggregation's decode-only nodes keep 5 ms TPOT vs our 12 ms; the gain is in attainment and
    goodput, not raw decode speed
  - short-prompt standalone workloads see little gain; the step helps where long-context prefill meets decode
  - MTP is concurrency-limited at 48 running requests for every system
  - open: semi-PD baseline (blocked on a CUDA 12 toolkit install), Blackwell kernels
