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
    the batch to an executor and a tile class (FA2 classes 16 / 32 for decode / verify rows, FA3 for prefill and wider
    rows; rows of 33-64 packed queries [pending: FA3 64-row tile vs FA2 64 / 48]), orders them longest first, splits
    KV where it balances the SMs, and in the same pass fixes the merge schedule:
    which tiles of a row are partials, and which tile merges them inside the launch, so no separate merge kernel runs
  - the executors are stock FA3 and FA2's BatchPrefill body (one CTA per plan item, both tile classes in one launch),
    chained by PDL; FA2 runs second and reduces every split row, FA2's and FA3's, in its epilogue, which removes native v3's FA2-merge
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

- story, one stage per section of the evaluation; each stage's claim, the gain we need, what is measured, and the
  figures that show it (filled in as results land)
  - stage 1, kernel (C1; sections 5.1, 6.2, 7)
    - claim: one plan per mixed step (tile classes, KV splits, LPT order, in-launch merge, FA3 / FA2 overlap by PDL)
      makes mixed-step attention faster than sglang's single FA3 call and than native v3, the best two-launch split
    - bar: faster than v3 on every model, with FP32-exact output; the step stays predictable for the router and EDF
    - measured: 1.43-1.51x sglang's FA3 call and 1.027-1.047x v3 on the H200 (Qwen2.5-7B, 96 trace-sampled steps);
      H20 level with v3 for Qwen2.5-7B, Llama-3.3-70B (1.38-1.61x sglang) and GLM-4.5-Air (1.28-1.41x sglang);
      step model 2.0% held-out MAPE, mixed steps 5.9% vs 7.7% for the single FA3 call
    - figures: time-weighted speedup per (model, context, mode) with 95% CI vs sglang FA3 and v3; per-item timeline
      of one step; ablation bars: FA2 / FA3 overlap by PDL 2.2-6.1% of the step, longest-first order <= 0.7%,
      persistent vs hardware dispatch -0.2..-1.3%, JIT release of merges (ready queue) -1.8..+0.5% vs the same loop with ticketed merges (H200,
      Qwen2.5-7B) [pending: Llama-3.3-70B]
  - stage 2, single node (C2 + C3 on C1; sections 5.2-5.3, 6.3)
    - claim: the 4-mode step (prefill chunk + speculative verify) under per-tenant deadline order with demote-on-miss
      protects both tenants on one node
    - bar: >= 30% better TTFT and ITL (or their SLO attainment) and >= 10% throughput vs stock sglang (mixed chunk on
      FA3, stock MTP), per model
    - measured: Qwen2.5-7B (H200) chat attainment 50 -> 59-65%, agent 45-52 -> 77-85%, chat TTFT -96%, agent TPOT
      -17..-49%, agent tok/s -9..+2% (throughput bar not met on one node); GLM-4.5-Air (4 H20, overloaded) four-mode
      throughput +18-19% for both tenants, agent TPOT -41..-48%, chat TTFT -68..-83%; Llama-3.3-70B [pending]
    - figures: per-tenant SLO attainment vs request rate; throughput vs tok/s/user Pareto with p10-p90 bars; TTFT / ITL
      distributions per tenant [pending: rate sweeps]
  - stage 3, multi node with SLO control (C4 on C1-C3; sections 5.2-5.4, 6.4, 6.5)
    - claim: symmetric colocated nodes with SLO-checked cache-first routing, a cluster KV pool and the stage-2 workers
      carry more load at SLO than PD disaggregation at its best P:D ratio, and than conditional disaggregation
    - bar: >= 30% on TTFT / ITL attainment and >= 10% throughput (SLO goodput) vs the best baseline
    - measured (16 H20s, Qwen2.5-7B, fan-out agents + chat, full stack): at 64 programs + 16 chat/s, chat attainment
      93.6% and agent SLO goodput 1100 tok/s vs 13.1% / 80 for stock PD at its best split and 96.0% / 821 for
      conditional disaggregation at Dynamo's defaults (+34% goodput); against conditional disaggregation +14..+37%
      agent SLO goodput at every load (32-128 programs) and +21% under a load spike, chat attainment equal through
      96 programs and +7.7 points at 128 [pending: Llama-3.3-70B, 32 GPUs]
    - figures: per-tenant attainment and SLO goodput vs load, the 90% capacity marked per system; P:D sweep; load-spike
      time series; fault-tolerance windows

## Abstract

Large language model (LLM) serving increasingly mixes two tenants on one cluster: interactive chat, with short
prompts and tight latency targets, and coding agents, which re-read 32-256K-token contexts on every round and add a
few hundred tokens each time. Every request first runs prefill, which turns the prompt into the key-value (KV) cache
and is compute-bound, and then decode, which generates one token per step and is memory-bound. Production systems
serve the two phases in one of two ways. Colocation under a chunked-prefill schedule runs a prefill chunk and the
decode rows in one attention call, so every decode row pays for prefill-width attention tiles and chat waits behind
long agent prefills. Disaggregation places prefill and decode on separate nodes, so every agent round re-ships its
prefix KV from a prefill node to a decode node, and the ratio of prefill to decode nodes (the P:D ratio) must be tuned
to the workload.

We present AttentionFuse, a colocated serving system with four parts. The first is a unified tile dispatcher: one
CPU-side plan per mixed step assigns every attention tile to FlashAttention-3 (FA3) or to a row-sized FlashAttention-2
(FA2) tile class, balances the tiles across the GPU's streaming multiprocessors (SMs), and schedules the merge of
split-KV partial results in the same plan, so the partials are reduced inside the FA2 launch instead of by separate
merge kernels. The second fuses speculative decoding into colocation: because every row gets a tile of its own size,
a prefill chunk and the speculative (MTP / EAGLE) verify rows of the running requests run in one forward, the
four-mode step. Prior colocation systems, POD-Attention and PD multiplexing among them, do not support speculative
decoding, which frontier models are served with. The third schedules these steps in per-tenant deadline order; a
prefill whose deadline has already passed is demoted behind every prefill that can still meet its deadline. The
fourth is a cluster of symmetric nodes with cache-first, SLO-checked routing and a cluster-wide KV pool.

On 96 mixed steps sampled from Qwen2.5-7B serving traces and replayed on an H200, mixed-step attention is 1.43-1.51x
faster than sglang's single FA3 call and 2.7-4.7% faster than the best two-launch split, with FP32-exact output; on
H20 GPUs it is level with that split for Qwen2.5-7B, Llama-3.3-70B and GLM-4.5-Air. On one node, SLO attainment rises
for both tenants (Qwen2.5-7B: chat 50 -> 59-65%, agents 45-52 -> 77-85%, chat time to first token -96%; GLM-4.5-Air:
throughput +18-19% for both tenants, agent time per output token -41..-48%), although Qwen2.5-7B's four-mode agent
throughput falls 8.5% on one node. On 16 H20s, at 64 agent programs plus chat, AttentionFuse keeps 93.6% chat
attainment with 1100 tok/s of agent SLO goodput, where PD disaggregation at its best P:D ratio reaches 13.1% and
80 tok/s; against conditional disaggregation it carries 12-37% more agent SLO goodput at every load (two repeats) and 21-24% more under
a load spike. [Llama-3.3-70B pending.]

## 1. Introduction

**Setting.** LLM services now carry two kinds of traffic at once. Chat requests have short prompts and tight
service-level objectives (SLOs) on time to first token (TTFT) and time per output token (TPOT). Coding agents run many
rounds over 32-256K-token contexts and add a few hundred new tokens per round; they tolerate a longer TTFT but still
need steady decoding. The problem we address is to meet each tenant's SLO at the highest possible load, without
dedicating nodes to one phase of inference.

**Why colocation under chunked prefill falls short.** sglang's mixed-chunk schedule runs the prefill chunk and all
decode rows through one variable-length FA3 attention call. FA3 tiles every row at prefill width, so a decode or
verify row of a few queries occupies a 128-row tile; at long context that padded work is most of the step. The
schedule is also first-come-first-served, so a chat request that arrives behind a 64K-token agent prefill waits for
it.

**Why disaggregation falls short.** Prefill-decode (PD) disaggregation removes the interference by giving each phase
its own nodes, but an agent's prefix KV then moves from a prefill node to a decode node on every round. We find that
this transfer queue, not the prefill queue, is what limits the system: putting the prefill nodes in deadline order
changes nothing (30.5 vs 31.2% chat attainment). NVIDIA Dynamo's conditional disaggregation avoids most of these
re-ships by prefilling short appends locally on the decode worker, but it has no SGLang implementation; Dynamo's
documentation lists SGLang as unsupported for it.

**Key insights.** Three observations drive the design. First, the two roles want different attention tiles: decode
and verify rows are best served by small, row-shaped FA2 tiles, while prefill stays on FA3. Second, colocation keeps a
session's KV on the node that computed it, so routing can follow cache hits instead of node roles. Third, a fused
step has a predictable duration, so the router and the per-worker deadline scheduler can price a request before
admitting it.

**Contributions.**

- C1, kernel: a unified tile dispatcher. One CPU-side plan per step covers FA2 tiles (classes of 16 and 32 query rows)
  and FA3 tiles, orders work longest first, and builds the split-KV merge schedule into the same plan, so FA2 and FA3
  partial results are reduced inside the FA2 launch and no merge launch runs. Stock FA3 and FA2's BatchPrefill body
  execute the plan, overlapped by programmatic dependent launch (PDL).
- C2, speculative decoding inside colocation: because the dispatcher gives each row a tile class of its own size, a
  prefill chunk and the MTP / EAGLE verify rows of the running requests run as one target forward, the four-mode step,
  with the verify rows on 32-row FA2 tiles and the chunk on FA3. Prior colocation designs do not support speculative
  decoding: POD-Attention overlaps a prefill chunk with one-token decode rows inside an FA2 kernel, and PD multiplexing
  (MuxWise, sglang's pdmux) runs the prefill split across iterations without the draft model's state. Frontier models
  are served with speculative decoding, so this is what makes a colocated schedule practical for them.
- C3, single node: per-tenant deadline order over the four-mode step, with demotion of prefills that have already
  missed their deadline.
- C4, cluster: symmetric nodes, SLO-checked cache-first routing, and a cluster KV pool.

**Results.** The kernel is 1.43-1.51x faster than sglang's FA3 call and 1.027-1.047x faster than the two-launch
baseline (FA3 and FA2 in separate launches, each with its own plan and merge) on trace-sampled steps on an H200, and level with the two-launch baseline on H20 for all three models. On one
node, attainment improves by 18-70% relative for both tenants on Qwen2.5-7B, and GLM-4.5-Air throughput rises 18-19%.
On 16 GPUs with the full stack, AttentionFuse holds 93.6% chat attainment and 1100 tok/s agent SLO goodput at 64
agent programs, against 13.1% / 80 for the best stock PD configuration and 96.0% / 821 for conditional
disaggregation. [higher loads, Llama-3.3-70B pending]

## 2. Background

### 2.1 LLM Serving

Prefill processes all prompt tokens in parallel and is compute-bound. Decode produces one token per request per step
and is bound by memory bandwidth, since each step reads the request's whole KV cache. Speculative decoding (here
EAGLE3, or the model's own next-token-prediction layer for GLM-4.5-Air) drafts several tokens and verifies them in one
forward pass, turning a one-token decode row into a 4-token verify row. We call this multi-token prediction (MTP)
serving.

Grouped-query attention (GQA) shares one KV head among several query heads. An attention kernel can pack those query
heads into the rows of one tile, so the number of rows a verify row occupies is the token count times the number of
query heads per KV head. A 4-token verify row packs 28 rows on Qwen2.5-7B (tensor parallelism 1, TP1), 32 on
Llama-3.3-70B (TP4) and 48 on GLM-4.5-Air (TP4). These packed sizes are what C1's tile classes are sized against: 16,
32 and 48 rows, with rows above 48 packed queries sent to FA3. The counts follow from each model's head configuration
and need no experiment.

Table 1: packed query rows per 4-token verify row, per model and TP degree.

| model (TP) | packed rows per 4-token verify row |
|---|---:|
| Qwen2.5-7B (TP1) | 28 |
| Llama-3.3-70B (TP4) | 32 |
| GLM-4.5-Air (TP4) | 48 |

### 2.2 Decode-prioritized, Prefill-prioritized, and Chunked Prefill Schedules

sglang offers several stock schedules. Mixed-chunk prefill runs one variable-length FA3 call over the prefill chunk
and the decode rows. The default prefill-prioritized schedule runs prefill batches ahead of decode, which stalls the
tokens of running decodes. sglang with Mooncake runs PD disaggregation, with roles fixed per node and KV shipped per
request. PD-Multiplexing partitions the SMs into groups with green contexts and runs prefill and decode in different
groups; it has no CPU-GPU overlap schedule and needs a CUDA graph set per SM group, so we use it only as a baseline.

No stock schedule protects both tenants; each moves latency from one tenant to the other. On one H200 serving
Qwen2.5-7B on sglang's FA3 backend, with chat at 8 req/s and an agent program concurrency of 8 (two repeats), stock
mixed chunk gives a chat TTFT mean of 579.5 ms and an agent inter-token latency (ITL) p99 of 417.8 ms. Stock
prefill-prioritized brings chat ITL p99 down to 27.3 ms but makes chat TTFT worse. This motivates C3. The same
comparison for Llama-3.3-70B and GLM-4.5-Air is [pending].

Figure 1: per schedule (mixed chunk, prefill-prioritized, each with and without MTP), per tenant: TTFT mean, ITL p99
and SLO attainment; one H200, Qwen2.5-7B, chat 8 req/s + agent concurrency 8. Llama-3.3-70B / GLM-4.5-Air [pending].

### 2.3 PD Disaggregation

DistServe and Mooncake split prefill and decode onto separate nodes. NVIDIA Dynamo adds conditional disaggregation:
the decode worker prefills locally when the effective uncached length is small, and sends the rest to a prefill
worker; Dynamo implements this only for vLLM and TRT-LLM. The resource split is static, so it cannot follow the
tenant mix, and decode nodes leave their compute idle.

Disaggregation therefore needs a tuned P:D ratio, and even the best ratio misses the chat SLO. On 16 H20s (8 pods of
2 GPUs) serving Qwen2.5-7B without speculation, with stock sglang PD and the KV pool on, we swept the number of prefill
pods from 1 to 7 of 8 at 32 agent programs plus chat at 8 req/s. Chat attainment rises from 16.2% with one prefill pod
to 67.8% with six (the best), and falls to 50.0% with seven. This motivates C4. The sweep for Llama-3.3-70B and
GLM-4.5-Air is [pending].

Figure 2: chat attainment and agent SLO goodput vs number of prefill pods (1..7 of 8), stock PD; 16 H20s, Qwen2.5-7B,
32 agent programs + chat 8 req/s. Llama-3.3-70B / GLM-4.5-Air [pending].

### 2.4 KV-cache centric scheduling

Multi-round agents re-read 32-256K-token prefixes, so cache-aware routing and KV pooling decide their TTFT. The hard
case, studied by ThunderAgent, is thrashing: the working set of active prefixes is larger than one node's KV
capacity. On 16 H20s at an agent program concurrency of 32, we swept the working set over 16, 64 and 128 tasks with
64K-token prefixes (about 1M, 4M and 8M KV tokens, against about 6.7M tokens of capacity). The cached-prefix fraction
of PD disaggregation falls from 0.890 to 0.605, while ours falls from 0.980 to 0.949. This motivates C4's cache-first
routing and cluster KV pool; Figure 22 (section 6.4) plots it.

## 3. Challenges and motivation

### 3.1 Attention computation time ratio

At 32-256K contexts, attention dominates the mixed step, and in our traces most mixed steps carry a prefill chunk
beside decode or verify rows. Attention speed therefore decides step time at long context, which motivates C1. We
will break down per-layer time of real serving steps at 32-64K and 128-256K, in two-mode steps (prefill chunk plus
decode rows) and four-mode steps (prefill chunk plus speculative verify rows), for all three models, and count step
composition (prefill chunk with rows, rows only, prefill only) from the same traces. Attention share: [pending]. Step
census: [pending].

Figure 3: stacked bars of attention, GEMM and other time per context bucket (32-64K, 128-256K) and mode, three models
[pending].

Figure 4: fraction of steps that are prefill chunk + rows, rows only, or prefill only, from the same traces [pending].

### 3.2 PD Disaggregation underutilizes resources

Under disaggregation, each agent round re-ships its prefix: about 3.7 GB of KV per 64K-token round on Qwen2.5-7B. On
16 H20s with stock PD at its best split (6 prefill and 2 decode pods), at 64 agent programs plus chat at 16 req/s,
each decode pod holds 52-55 requests waiting on KV transfer (at most 108) while running only about 7, at 63-68% KV
usage. The decode pods are neither compute- nor memory-full; they wait for KV. Putting the prefill pods in deadline
order instead of first-come-first-served (FCFS) moves chat attainment from 31.2% to 30.5%. The limit of disaggregation is the
per-round KV transfer, not the prefill queue, which motivates C4. The same measurement for Llama-3.3-70B and
GLM-4.5-Air is [pending].

Figure 5: decode-pod time series of requests waiting on transfer, running requests and KV usage, with chat
attainment under FCFS and under deadline order on the prefill pods; 16 H20s, stock PD 6 + 2 pods, 64 agent programs
+ chat 16 req/s, Qwen2.5-7B. Llama-3.3-70B / GLM-4.5-Air [pending].

### 3.3 Chunked Prefill cannot adhere to SLOs

An agent burst of 32 requests within 60 s at 32-64K context holds chat TTFT at 4.5-13.9 s under stock mixed chunk on
one H200 (Qwen2.5-7B, ShareGPT chat at 8 req/s). Deadline order alone does not fix this. Under overload, plain
earliest-deadline-first (EDF) serves the agent prefills that are already late, because their deadlines are earliest,
and so makes chat late too; per-tenant deadlines alone change nothing (section 7). Deadline order needs a rule that
demotes requests that have already missed, which motivates C3. Llama-3.3-70B / GLM-4.5-Air: [pending].

Figure 6: chat TTFT over time around the agent burst, and per-tenant attainment for stock mixed chunk, per-tenant
deadlines alone, and deadlines with demotion of missed prefills; one H200, Qwen2.5-7B. Llama-3.3-70B / GLM-4.5-Air
[pending].

### 3.4 Mixed-chunk prefill reduces throughput

A single attention call uses one tile shape for both roles, so decode and verify rows are padded to 128-row prefill
tiles. Splitting the call by role alone, with FA3 for both, gives 1.23-1.57x attention speed on the 96 trace-sampled
captures of section 6.2 (Qwen2.5-7B, Table 6). Rows need their own tile shape, which motivates C1.

Figure 7: per-step attention time of the single FA3 call vs the role split, and the fraction of row tiles that are
padding; 96 captures, Qwen2.5-7B. Padding fraction [pending].

### 3.5 Can we increase request capacity while maintaining SLOs?

Yes, if rows get their own kernel tiles, chat is not queued behind prefills that can no longer meet their deadlines,
and nodes stay symmetric so that routing follows the cache. C1-C4 together are that answer. We measure it as the load
each system holds at >= 90% per-tenant attainment (Figure 19). With the scheduler alone on 16 GPUs, ours holds 64
agent programs against 16 for PD disaggregation (section 6.4); the full-stack capacity is [pending].

## 4. AttentionFuse Design

## 4.1 Problem Formulation

Each tenant has its own SLO. The results measured so far use chat 0.5 s TTFT and 20 ms TPOT and agents 10 s TTFT and
80 ms TPOT. The final evaluation takes its SLOs from prior work (section 6.1): chat, math and code 2 s TTFT and 50 ms
(tight) or 100 ms (loose) mean TPOT [JITServe, SLOs-Serve]; agent rounds 10 s TTFT, doubled to 20 s for prefills of more
than 200K uncached tokens; each swept over a 0.8-1.5x scale [pending: re-scoring]. A request attains its
SLO if it meets both of its tenant's targets. The objective is the load a system holds while each tenant keeps at
least 90% attainment, not mean latency. Agent load is the number of concurrent agent programs; chat load is the
request rate.

The design acts at three levels against that one objective. Per step (C1, C2), one plan assigns every attention tile to
FA3 (rows of more than 32 packed queries) or to an FA2 tile class (16 or 32 rows; a 64-row class takes rows of 33-64
packed queries for models whose verify rows pack that many, such as GLM-4.5-Air), and fixes the merge schedule
for split rows, and the verify rows of speculative decoding ride in the same forward as a prefill chunk. Per worker (C3), prefills are ordered by deadlines computed from their uncached tokens, and a prefill
whose deadline has passed is demoted. Per cluster (C4), a request goes to the node with the highest prefix-cache hit
that is predicted to meet its SLO; if that is not its best cache holder, the node pulls the prefix from the cluster
pool.

Every serving experiment reports this objective: load is swept on agent program concurrency and chat request rate,
and per-tenant attainment is recorded at each point (Figure 19). It is measured at 16 and 32 GPUs for the scheduler
alone (section 6.4); single-node and full-stack capacity are [pending].

## 5. Implementation

### 5.1 Persistent Kernel with overlap Schedule

**The plan.** The unified tile dispatcher builds one CPU-side plan per step, shared by all layers. It classifies each
row by its packed query length: rows of at most 16, 32 or 48 packed queries go to the matching FA2 tile class, and
longer rows go to FA3. It cuts each row's KV into chunks sized to balance SM load, and places the resulting work items
with longest-processing-time-first (LPT) order: the longest item goes onto the least-loaded slot, across the slots of
both executors. With the work list it emits the merge schedule: for every split row, which slots hold its partial
results and which merge group combines them. The plan is built incrementally on the CPU and overlapped with the
previous step; host plan time is 48-78 us mean on 44 serving geometries, and its overlap on serving shapes is
[pending].

**One merge, in the FA2 launch.** FA2 runs second under PDL, so it can reduce every split row, its own and FA3's, from
merge items that the plan queues after the compute items. FA3 tiles only write FP32 partial outputs and their
log-sum-exp (LSE) values. No merge kernel runs for either executor.

**Executors.** The executors are stock FA3 (flash-attention's mainloop over the paged KV cache) and FA2's BatchPrefill
body, which runs one thread block (CTA) per plan item. Both read the one work list, and PDL chains the two launches so
that the second can start while the first drains. A persistent FA2 executor, which keeps its CTAs resident and pulls
the plan's items in LPT order, is the alternative we ablate in section 7, together with two optional mechanisms:
handing partials to the reducing CTA through distributed shared memory (DSMEM, the shared memory of a thread-block
cluster), and multicasting KV loads for FA3 and FA2.

**What is fused, and why there.** The fusion is in the plan and the reduction, not in the instruction stream. Prefill
rows want Hopper's warpgroup MMA (wgmma) on multiples of 64 rows behind a warp-specialized producer; decode and verify
rows want small mma.sync tiles over long KV. Hopper (wgmma) and Blackwell (tcgen05) cannot overlap MMA instructions of
different shapes inside one kernel. Single-kernel prefill/decode overlap, as in POD-Attention, is therefore FA2-only,
and our own co-compiled FA3 + FA2 kernel was 2.16x slower (below).

What the two roles do share is complementary hardware demand: FA3 prefill tiles are bound by the tensor cores, FA2
rows by memory bandwidth. The plan therefore places the FA2 rows to run beside FA3's tail through PDL; turning that
overlap off costs 2.2-6.1% of the step (section 7).

One merge serves both executors. It runs in the FA2 launch and is scheduled by the same plan, so the two-launch
baseline's separate FA2-merge and FA3-combine launches disappear. With every merge, check-in fence and counter atomic
removed (the output is then wrong; this is a timing bound), two-mode trace-sampled steps are at most 0.9% faster,
and per-item timelines put the merge at 0.8% of four-mode steps. (Compiling the merge out does not give a four-mode
bound: it changes the compiler's register allocation for the 32-row tile, which then runs slower even where no merge
executes.) Large decode
batches are where the merge matters: on synthetic ShareGPT and shared-prefix decode steps of 64-256 rows it costs
6-13% of the step (section 7).

**Why the default FA2 executor is not persistent.** A persistent LPT kernel was the intended design, for three
expected advantages: an L2-swizzled order in which items that read the same KV pages run together, fewer CTA launches
at large batch, and cluster-scheduled reduction in the epilogue. We measured each:
- *CTA launches.* Launching one CTA per item costs nothing measurable, even at 4096 items (decode batches of 1024
  rows): the persistent pull loop running the same tile code is 0.89-0.98x of the hardware-launched grid. The
  persistent runner does win on large decode batches (1.08-1.14x at 256-1024 rows), but through occupancy: its
  decode-only instantiation fits 3 CTAs per SM where the non-persistent kernel fits 2. We therefore give the
  non-persistent executor the same plan-chosen decode-only instantiation [pending: measurement].
- *Cluster-scheduled reduction.* Reducing partials inside a thread-block cluster, over DSMEM and one cluster barrier,
  is exact but slower at every cluster size. Our first design capped each request's splits at the cluster size
  (0.60-0.98x). The second lets large groups span clusters and reduces them by their last member; it reaches
  0.988-0.993x on trace-sampled steps but no further. The remaining cost is gang scheduling: a cluster's CTAs start
  together and hold their SMs until the slowest one ends, which costs 1-11% even on steps with nothing to reduce. Our
  in-launch merge instead spreads each group over many independently scheduled CTAs (section 7).
- *L2 order.* A KV-position-major order makes rows that share a prefix read its pages in the same wave. It helps
  only when the shared prefix exceeds L2 and its rows are split (1.08x on a 48K shared prefix). Elsewhere it breaks
  longest-first order and loses up to 29%; applying it only among equal-cost chunks keeps longest-first order but
  gains nothing (0.98-1.02x). The trace-sampled steps are level under every order (1.000 / 1.003 / 0.998 / 1.000x).
Longest-first order itself is worth <= 0.7% on the trace-sampled steps (section 7). Releasing merges as soon as their
partials are ready adds polling that slows the compute (section 7).

**Why not one kernel for FA2 and FA3: the warp layouts differ.** FA3 is warp-specialized: one producer warpgroup
issues tensor-memory-accelerator (TMA) loads and two consumer warpgroups run wgmma on a 128-row tile, so a CTA has
384 threads (12 warps), runs one CTA per SM, and uses about 98 KB of shared memory. FA2 is not specialized: 4 warps
(128 threads) each both load and compute with mma.sync, two CTAs per SM, and its row tiles (16, 32 or 48 query rows)
are laid out over those 4 warps. One `__global__` function has one block size, so a fused kernel launches every CTA
at FA3's 384 threads. The FA2 role then either leaves 256 of the 384 threads idle or is re-laid-out over 12 warps,
which its small row tiles cannot fill.

The register and shared-memory costs come from that layout, FA3's warp specialization needing more warps per CTA, and
not from either algorithm (Table 2, H200, per-function ptxas reports):

- 384 threads at one CTA per SM cap each thread near 170 registers, and ptxas gives one register budget per kernel, so
  the two inlined role bodies share it. FA3 spills 33 676 B in the fused kernel vs 52 B alone in the identical
  template; outlining FA2 takes FA3 to 64 B but moves 17 500 B of spill onto FA2.
- FA2 at FA3's geometry gets 168 registers and spills 1 136 B, vs 255 registers and 0 B at its own 128 threads x 2
  CTAs. The SM-wide register budget is the same (255 x 256 = 170 x 384), so the loss is the idle threads.
- Shared memory: FA3's 98 352 B leaves 134 048 B of the 232 448 B limit; FA2's 16-row configuration needs 135 168 B,
  1 120 B over.

Table 2: registers and spill per build (H200).

| build | registers | spill |
|---|---:|---:|
| FA3 alone | | 52 B |
| FA3 in the fused kernel | | 33 676 B |
| fused kernel, FA2 outlined: FA3 / FA2 | | 64 B / 17 500 B |
| FA2 at FA3's geometry (384 threads x 1 CTA) | 168 | 1 136 B |
| FA2 at its own geometry (128 threads x 2 CTAs) | 255 | 0 B |

The net effect: the fused kernel was 2.16x slower than FA2 alone on every input (155.9 vs 72.3 us, both numerically
correct; 1.83x on a fixture with 16 823 KV tokens), while two launches of the same executors at their own layouts
were 1.21x faster.

Sharing SMs between two separate launches is a different question. Co-running FA3 on a subset of SMs beside FA2 gains
18% only where the two roles take comparable time, ties or loses elsewhere, and is bounded at about 1.2% of serving
GPU time on the TraceLab traces. So the executors are separate launches chained by PDL, and what they share is the
plan: tile classes, KV splits, slot placement and the merge schedule.

The same effect appears inside the FA2 executor. One persistent kernel holding the 48-, 32- and 16-row runners
compiles to one register budget, and the 48-row runner (one query warp issuing 3 MMAs at 4 warps) makes the whole
kernel spill 1 436-1 444 B; with it compiled out, the kernel spills 16-20 B. The 48-row class therefore needs a layout
with 16 rows per warp, or its rows go to FA3 [pending].

**Baseline for this design.** The build measured as a baseline in section 6.2 has two separate plans (FA3's and
flashinfer's), non-persistent FA2 rows, and two merge mechanisms (FA3's combine launch and flashinfer's merge). C1's
mandatory parts are the one plan and the split-KV reduction of FA2 and FA3 partials in the FA2 launch with no merge
launch.

**Correctness and plan cost.** The planner is exact on 44 captured geometries, with plan time 48-78 us mean. On the 96
captures, the previous build's FP32 worst relative error was 0.0067. Plan time on serving shapes of all three models
and the 48-row class are [pending].

Figure 8: host plan time per step (mean and p99) vs batch size, on captured serving geometries of all three models
[pending: serving shapes].

### 5.2 Router and resource sharing

#### 5.2.1 Router design

The router is cache-first. It orders nodes by prefix-hit bucket, then by load, and takes the first node that passes
an SLO check. The check prices the request's own prefill plus the deadline-ordered queue ahead of it and the decode
load beside it, using the step-time model of section 5.2.2. This is part of C4. We will compare cache-first routing
with the SLO check against cache-only and round-robin routing on 16 H20s, all three models, fan-out agents plus chat,
with load swept on agent program concurrency. Result: [pending].

Figure 9: per-tenant attainment vs load and cached-prefix fraction, per routing policy (cache-first + SLO check,
cache-only, round-robin); 16 H20s, three models [pending].

#### 5.2.2 Performance model

One model predicts the fused step's time. Its inputs are the prefill tokens, the prefix length, the number of decode
and verify rows and their KV lengths, and the predicted speculative acceptance length, tracked per worker as an
exponentially weighted moving average (EWMA). EDF (C3) and the router (C4) price a request from this predicted step
time. The model is fit on logged serving steps of the current kernel, per model, and scored on held-out steps.

On Qwen2.5-7B on an H200 in four-mode serving, fit on one repeat and scored on the other, held-out mean absolute
percentage error (MAPE) is 2.0% for our kernel vs 1.9% for the stock single FA3 call (Table 3). The fused step is as
predictable as the stock one, and the mixed steps that admission prices are more predictable. Llama-3.3-70B and
GLM-4.5-Air fits are [pending].

Table 3: held-out step-time MAPE by step type (Qwen2.5-7B, H200, four-mode serving).

| step type | ours | stock single FA3 call |
|---|---:|---:|
| all | 2.0% | 1.9% |
| decode | 1.4% | 1.2% |
| mixed | 5.9% | 7.7% |
| prefill | 8.7% | 7.5% |

Figure 10: predicted vs measured step time, per step type (decode, prefill, mixed, four-mode), per model [pending].

#### 5.2.3 Fair Resource sharing

Prefill chunks are sized from the EDF backlog (1024 or 2048 tokens), and a chunk is preempted when a request with an
earlier deadline arrives. This is part of C3. We will compare deadline-sized chunks with a fixed chunk size, and
preemption on vs off, on one node for all three models under an agent burst plus chat. Result: [pending].

Figure 11: per-tenant TTFT / ITL attainment and agent throughput for fixed vs deadline-sized chunks, with and without
preemption; one node, three models [pending].

### 5.3 SLO-aware scheduling

Each prefill gets a deadline from its uncached tokens and its tenant's SLO: 0.5 s below 8192 uncached tokens and 10 s
above. Prefills are served in deadline order. A prefill whose deadline has already passed ranks behind every prefill
that can still meet its deadline, which is EDF's standard overload rule; decodes that were retracted for memory are
never demoted. This is C3's scheduling rule.

On one H200 with Qwen2.5-7B, under a 32-64K agent burst plus chat, per-tenant deadlines alone change nothing, while
adding demotion of missed prefills cuts chat TTFT by 96-98% (section 7). The same comparison on one 4-GPU H20 pod for
Llama-3.3-70B and GLM-4.5-Air is [pending].

Running under tensor parallelism exposed one requirement: every TP rank must judge deadlines on the same clock, so
rank 0's clock rides the per-iteration request broadcast.

Figure 12: chat TTFT distribution and per-tenant attainment for stock, per-tenant deadlines alone, and deadlines with
demotion of missed prefills; one H200 (Qwen2.5-7B) and one H20 pod (Llama-3.3-70B, GLM-4.5-Air). H20 models [pending].

### 5.4 KV Cache pooling policy

The cluster KV pool is Mooncake's distributed store. Selected prefixes are written through to it, and the router keeps
a mirror of the pool's contents so it can price a pull when it routes a request away from the prefix's best holder.
This is C4's cluster KV pool. Under the thrashing sweep of section 6.4 (working set 1M -> 8M KV tokens against a
6.7M-token pool), ours loses 10.6% throughput and 3.1 points of cached fraction, vs 55% and 28.5 points for
disaggregation, in earlier settings (Figure 22). Pool on vs off is [pending].

### 5.5 CUDA Graph support

The four-mode step runs under one CUDA graph set, whereas PD-Multiplexing multiplies its graphs by the number of SM
groups. This lets the four-mode step run with CUDA graphs and the CPU-GPU overlap schedule (C2). Graphs on vs off,
and graph count and capture memory vs PD-Multiplexing, are [pending].

Table 4: CUDA graph count, capture memory and decode step time, ours vs PD-Multiplexing, graphs on vs off [pending].

### 5.6 Portability across GPU generations

The design keeps the large-tile and small-tile attention kernels separate, and fuses only what sits above them: the
one plan that routes rows to tile classes and splits their KV, and the one merge that reduces every split row inside
the small-tile launch. Neither executor is modified beyond the interface the plan needs: the prefill executor reports
its split-KV partials (output and log-sum-exp) per split, and the small-tile executor reads its work items and merge
items from the plan.

This is what makes the design portable. A single kernel that overlaps prefill and decode tiles inside one launch has
to interleave two instruction streams, and the MMA instructions of different tile shapes cannot overlap within one
kernel on either Hopper (wgmma) or Blackwell (tcgen05). Such a design is tied to one architecture's instruction set:
POD-Attention is FA2-only for this reason, and our own co-compiled FA3 + FA2 kernel was 2.16x slower than the two
separate kernels (section 7). Keeping the executors separate avoids both problems.

On Blackwell, FlashAttention-4 (tcgen05) takes the place of FA3 for prefill rows. FA2's mma.sync kernels run on
Blackwell unchanged, so the decode / verify tile classes and the in-launch merge carry over as they are, and PDL is
available there as on Hopper. The port therefore replaces one executor behind an unchanged plan, rather than
rewriting a fused kernel. The interface FA4 must provide is the one FA3 provides here: split-KV partials, or an
unsplit run whose rows the plan does not split. [pending: measurement on Blackwell]

## 6. Evaluation

### 6.1 Settings

**Models.** Qwen2.5-7B-Instruct at TP1, extended with YaRN 4x to 128K, with a trained EAGLE3 draft head;
Llama-3.3-70B-Instruct at TP4 with a trained EAGLE3 head; GLM-4.5-Air at TP4 with its own next-token-prediction layer
as the draft. All three are evaluated in every stage. If the final version shows only two, they will be Llama-3.3-70B
and Qwen2.5-7B.

**Testbed.** One H200 for the kernel and single-node stages; H20 nodes of 4 GPUs each, 16 and 32 GPUs in total, for
the multi-node stage.

**Baselines.**

- Kernel: sglang's single variable-length FA3 call; the best single fixed FA3 configuration; a per-step FA3 oracle;
  the role split on FA3 (FA3 for both roles, in separate calls); and the two-launch baseline, which runs stock FA3 and
  our tuned BatchPrefill FA2 as two launches, each with its own plan and its own merge or combine launch.
- Single node: sglang mixed chunk and prefill-prioritized on FA3, with stock EAGLE3 MTP for the four-mode comparison;
  PD-Multiplexing.
- Multi node: sglang with Mooncake PD disaggregation, the P:D ratio swept at worker granularity, KV pool on. Reference
  policy: conditional disaggregation with Dynamo's default rule (remote prefill when the effective uncached length is
  at least 2048 tokens and at least 0.7 of the prompt), implemented in our router on sglang.
- Kernel used in stages 2-3: the two-launch baseline until the unified dispatcher beats it in stage 1, then the
  unified dispatcher.

**Workloads.** TraceLab coding-agent traces with context mixtures from 32K to a few hundred K (32-64K and 128-256K,
multi-round), with ShareGPT chat; fan-out coding agents on shared 64K-token repository prefixes with ShareGPT chat
(multi-node); and standalone sets (ShareGPT, math, coding, agentic web browsing) at load around capacity. Agent load
is program concurrency; chat load is request rate.

**SLOs.** Chat 0.5 s TTFT and 20 ms TPOT; agents 10 s TTFT and 80 ms TPOT.

**Metrics.** Throughput vs per-user tok/s (a Pareto curve) with p10-p90 bars; TTFT and ITL SLO attainment vs request
rate, per tenant; per-tenant SLO goodput, the tok/s of requests that meet both of their tenant's targets. Every
serving result shows per-request spread and per-repeat values.

**Protocol.** Both arms run on the same frozen software tree, in opposite-order pairs, two repeats each, with metrics
recomputed from raw token arrival times and CUDA graphs and overlap scheduling on. A quality gate (GSM8K and passkey
retrieval at 32-60K) is judged against a baseline-vs-baseline noise floor.

**Software.** Every arm, ours and every baseline, runs the same pinned upstream sglang main. The stock baselines run
it unmodified; ours adds an overlay with EDF, the four-mode step, the kernel hooks and the router. Conditional
disaggregation is implemented in our router on the same workers as ours, since Dynamo does not support SGLang for it.

**Status.** Qwen2.5-7B is measured on the scheduler-only and earlier-kernel builds; GLM-4.5-Air's kernel stage is
partial (section 6.2); Llama-3.3-70B is [pending].

### 6.2 Kernel microbenchmarks

**Claim.** One plan per mixed step (tile classes, KV splits, longest-first order, the in-launch merge, FA3 / FA2
overlap by PDL) makes mixed-step attention faster than sglang's single FA3 call and than the two-launch baseline,
which runs the same FA3 and FA2 kernels with separate FA2-merge and FA3-combine launches. The reason: in one
variable-length FA3 call, a 4-token verify row (28-48 packed rows) is tiled at 128 rows. Row-shaped tiles stop that
padding, and LPT spreads row work of different sizes across the SMs.

**Bar.** Faster than the two-launch baseline on every model, with FP32-exact output, while the step stays predictable
for the router and EDF (section 5.2.2).

**Setup.** We capture every Nth real mixed serving step, 24 per (context, mode) cell and 96 per model, and replay
each with its own tensors. Contexts are 32-64K and 128-256K; modes are two-mode (prefill chunk + decode rows) and
four-mode (prefill chunk + verify rows). Qwen2.5-7B runs on one H200; Llama-3.3-70B and GLM-4.5-Air run at TP4 on H20,
measured on rank 0. Baselines are sglang's FA3 call, the best fixed FA3 configuration, the per-step FA3 oracle, the
two-launch baseline and the role split on FA3. We report time-weighted speedup over each cell's steps with a
bootstrap 95% confidence interval (CI), and FP32 worst relative error.

**Results on the H200.** The unified dispatcher runs one plan, stock FA3 at the plan's split, and the two-launch
baseline's FA2 body with the chunk-parallel in-launch merge. On 96 fresh Qwen2.5-7B captures, all FP32-exact (worst
relative error 0.0082), it is 1.43-1.51x faster than sglang's FA3 call and 1.027-1.047x faster than the two-launch
baseline (Table 5).

Table 5: unified dispatcher speedup, Qwen2.5-7B, H200, 96 captures.

| context / mode | vs sglang's FA3 call | vs two-launch baseline [95% CI] | vs role split on FA3 |
|---|---:|---:|---:|
| 32-64K, 2-mode | 1.48x | 1.033x [1.006-1.051] | 1.015x |
| 32-64K, 4-mode | 1.43x | 1.047x [1.035-1.057] | 1.020x |
| 128-256K, 2-mode | 1.51x | 1.027x [1.018-1.036] | 1.014x |
| 128-256K, 4-mode | 1.44x | 1.027x [1.018-1.041] | 1.018x |

**Results on H20.** On the H20 (78 SMs), the same build is level with the two-launch baseline on Qwen2.5-7B
(0.998-1.008x) and 1.10-1.23x faster than sglang's FA3 call. There FA3 sets the step time, and the FA2 path's 3-10%
saving runs beneath it. Llama-3.3-70B (TP4, 32-64K) is 1.009x / 1.011x the two-launch baseline on two- / four-mode
steps and 1.38x / 1.61x sglang's FA3 call. GLM-4.5-Air (TP4, 32-64K, 43 captures) is 0.998x / 1.000x the two-launch
baseline and 1.28x / 1.41x sglang's FA3 call. With tile-rounded KV splits (the final serving build), Llama-3.3-70B is
1.009x / 1.017x the two-launch baseline (two- / four-mode, 38 captures) and GLM-4.5-Air two-mode 1.006x.

GLM-4.5-Air needs the third FA2 tile class. Its 4-token verify rows pack 48 queries per KV head, and sending them to
FA3 cost 28% of the four-mode step. A 64-row FA2 class, launched as its own instantiation so that it does not raise
the smaller classes' register use, recovers it. On the earlier role-split build, which has no 48-row tile, GLM-4.5-Air
runs 0.63-0.98x on four-mode steps and up to 1.35x on two-mode steps; GLM-4.5-Air on the unified dispatcher is
[pending].

**Earlier build.** The role-split serving build, measured on earlier Qwen2.5-7B captures on the H200, has two plans
and two merges. It is 1.23-1.57x faster than sglang's FA3 call but only matches a well-tuned FA3 split in isolation
(Table 6). Its FP32 worst relative error is <= 0.0075, and GSM8K and passkey stay within the baseline noise floor.

Table 6: role-split serving build, Qwen2.5-7B, H200, earlier captures.

| context / mode | vs sglang's FA3 call [95% CI] | vs best fixed FA3 config | vs per-step oracle |
|---|---:|---:|---:|
| 32-64K, 2-mode | 1.35x [1.12-1.91] | 0.98x | 0.97x |
| 32-64K, 4-mode | 1.23x [1.09-1.56] | 0.99x | 0.98x |
| 128-256K, 2-mode | 1.57x [1.24-2.64] | 0.99x | 0.98x |
| 128-256K, 4-mode | 1.37x [1.16-1.89] | 0.99x | 0.99x |

**Bar status.** The bar is met on the H200 for Qwen2.5-7B (1.028-1.042x the two-launch baseline with the final
build). On H20 the unified dispatcher is ahead of the two-launch baseline by 0.6-1.7% (Llama-3.3-70B, GLM-4.5-Air two-
mode) and level for Qwen2.5-7B; FA3 sets the H20 step time, so the FA2-side savings run beneath it.

**Figures.**

Figure 13: time-weighted speedup per (model, context, mode) with bootstrap 95% CI, against sglang's FA3 call and the
two-launch baseline. Qwen2.5-7B (H200, four cells) and Llama-3.3-70B / GLM-4.5-Air at 32-64K (H20) measured; the
remaining Llama-3.3-70B / GLM-4.5-Air cells [pending].

Figure 14: CDF of per-step speedup over the 96 captures, per (context, mode), against each baseline; Qwen2.5-7B,
H200. Other models [pending].

Figure 15: per-item timeline of one captured four-mode step: FA3 tiles, FA2 tiles and merge items per SM over time,
showing FA2 running beside FA3's tail under PDL [pending].

### 6.3 Single-node benchmarks

**Claim.** The four-mode step under per-tenant deadline order, with demotion of missed prefills and on the C1 kernel,
protects both tenants on one node. The kernel speeds up the step; deadline order lets chat through an agent burst;
demotion keeps already-late agents from making chat late.

**Bar.** At least 30% better TTFT and ITL (or their SLO attainment) and at least 10% higher throughput than stock
sglang (mixed chunk on FA3, with stock MTP for four-mode), per model.

**Setup.** Qwen2.5-7B on one H200; Llama-3.3-70B and GLM-4.5-Air on one 4-GPU H20 pod at TP4. Baselines: sglang
mixed chunk and prefill-prioritized on FA3, with and without stock EAGLE3 MTP, and PD-Multiplexing. The kernel is the
two-launch baseline until the unified dispatcher beats it. Load is swept on agent program concurrency and chat request
rate, over context mixtures from 32K to a few hundred K (32-64K, 128-256K), in two- and four-mode, plus the standalone
sets (ShareGPT, math, coding, agentic web browsing) at load around capacity. SLOs are those of section 6.1. Every
result carries per-request spread and per-repeat values.

**Result: the kernel alone, same schedule.** On Qwen2.5-7B at 128-256K, with agent concurrency 4 and chat at 4 req/s,
the fused kernel raises four-mode agent throughput by 10.7% and cuts agent TPOT by 24.2% and ITL by 20.8%; two-mode
gains are +11.6%, -20.1% and -16.7%. Against a role split that runs FA3 for both roles, throughput is equal but agent
decode latency moves only -1.5..+6.6%, so the FA2 rows are worth 15-25 points of agent ITL.

**Result: full stack vs stock, Qwen2.5-7B.** Under an agent burst of 32 requests in 60 s at 32-64K plus chat at 8 req/s,
the full stack raises chat attainment from 50.0% to 59.0% (four-mode) and from 49.1% to 65.4% (two-mode), and agent
attainment from 45.0% to 76.7% and from 51.7% to 85.0% (Table 7). Chat TTFT falls 95-96%. The cost is chat TPOT,
+35-56%: chat decodes are admitted beside the burst's long prefills, where stock makes chat wait instead. Agent
throughput falls 8.5% in four-mode and rises 1.7% in two-mode, so the throughput bar is not met on one node.

Table 7: full stack vs stock, Qwen2.5-7B, one H200, 32-64K agent burst + chat 8 req/s. "Chat at 1 s / 30 ms" is chat
attainment under a relaxed SLO.

| mode | chat attainment | chat at 1 s / 30 ms | agent attainment | chat TTFT | agent TPOT | agent tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 4-mode (stock MTP -> ours) | 50.0 -> 59.0% | 50.3 -> 83.3% | 45.0 -> 76.7% | -95.2% | -49.1% | -8.5% |
| 2-mode (stock -> ours) | 49.1 -> 65.4% | 49.5 -> 85.4% | 51.7 -> 85.0% | -96.2% | -16.6% | +1.7% |

**Result: full stack under the paper's SLOs, Qwen2.5-7B, both context ranges (Table 7b).** These runs use the final
configuration: the unified dispatcher with tile-rounded KV splits, and no per-step deadline-sized chunk cap. The SLOs
are chat 2 s TTFT with 50 / 100 ms mean TPOT, and agents 10 s TTFT (20 s above 200K tokens) with 80 ms. Chat
attainment rises by 38-74 points everywhere, to 93-100% at the loose SLO, and chat end-to-end latency falls 51-70%.

At 128-256K, agent SLO attainment rises from 13-19% to 44-50% and agent SLO goodput by 70-190%. Raw agent throughput
also rises: +8.1% four-mode, +10.6% two-mode.

At 32-64K, agent goodput is level in four-mode (agent TPOT -44%). In two-mode it falls 13%: deadline order gives chat
priority over agents that stock already served within their SLO. Raw agent throughput there is -3.6% (four-mode) and
+2.7% (two-mode), so the 10% throughput bar is met on one node only for 128-256K two-mode.

Table 7b: Qwen2.5-7B, one H200, final configuration, two repeats (16 / 30 agent requests per arm at 128-256K / 32-64K).

| workload, mode | chat 2 s / 50 ms | chat 2 s / 100 ms | chat e2e | agent attainment | agent SLO goodput | agent tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 128-256K (agent conc 4, chat 4/s), 4-mode | 21.0 -> 63.1% | 21.0 -> 94.9% | 27.1 -> 8.1 s | 18.8 -> 43.8% | 16-24 -> 41 tok/s | +8.1% |
| 128-256K, 2-mode | 18.8 -> 62.8% | 18.9 -> 92.6% | 25.6 -> 8.3 s | 12.5-18.8 -> 50.0% | 14-22 -> 37 tok/s | +10.6% |
| 32-64K (agent conc 8, chat 8/s), 4-mode | 56.4 -> 97.2% | 56.5 -> 99.6% | 5.8 -> 2.8 s | 70.0-73.3 -> 80.0% | 74-84 -> 85-86 tok/s | -3.6% |
| 32-64K, 2-mode | 54.3 -> 95.4% | 54.4 -> 99.8% | 5.7 -> 2.8 s | 83.3 -> 83.3-86.7% | 101-107 -> 88-90 tok/s | +2.7% |

At 128-256K, neither system reaches 90% attainment for either tenant at agent concurrency 1, 2 or 4 on one GPU,
because cold 128-256K turns cannot prefill within 10 s. Ours is still 8-19 points ahead on chat and 25-38 points ahead
on agent attainment at concurrency 4. This is the case that motivates the multi-node stage.

**Result: scheduler only, Qwen2.5-7B.** On the FA3 backend with chat at 8 req/s and agent concurrency 8 (two
repeats), ours without MTP reaches 86.7 / 95.6% agent attainment vs 84.4 / 84.4% for stock mixed chunk. Ours with MTP
trails stock prefill-prioritized with MTP at the 80 ms ITL SLO (62.2 / 64.4 vs 66.7 / 68.9%) and leads at 30 ms
(55.6 / 51.1 vs 46.7 / 48.9%). Higher agent concurrency is [pending].

**Result: GLM-4.5-Air.** On one 4-GPU H20 node (TP4, built-in MTP), with 32-64K agents at concurrency 8 plus chat at
8 req/s (two repeats), the full stack raises four-mode throughput by 18-19% for both tenants, cuts agent TPOT by
41-48% and chat TTFT by 68-83% (Table 8). This load is far above the node's capacity (TTFT is tens of seconds in both
systems); the at-capacity point follows Table 8.

Table 8: full stack vs stock sglang on FA3, GLM-4.5-Air, one H20 node, TP4, overloaded. "Chat at 2 s / 50 ms" is chat
attainment under a relaxed SLO.

| mode | agent tok/s | chat tok/s | agent TPOT | chat TTFT | chat at 2 s / 50 ms | agent attainment |
|---|---:|---:|---:|---:|---:|---:|
| 4-mode (stock MTP -> ours) | +18-19% | +18-19% | -41..-48% | -68..-83% | 0 -> 65-76% | 20 -> 37-43% |
| 2-mode (stock -> ours) | +12-14% | | -17..-28% | -63..-64% | 0 -> 60-64% | 23 -> 23-30% |
| 4-mode, final configuration (no chunk cap) | **+28%** | **+27%** | -32% | -71% | 0 -> 64-66% (2 s / 100 ms: 0 -> 77-78%) | 20 -> 30-33% |
| 2-mode, final configuration | +17% | +15% | -37% | -55% | 0 -> 47-52% (2 s / 100 ms: 0 -> 63-66%) | 23 -> 33-43% |

With the final configuration (tile-rounded KV splits, no per-step chunk cap), four-mode throughput rises 27-28% for
both tenants and chat end-to-end latency falls from 110-111 s to 36-40 s (two-mode 105-108 s to 55-56 s); two repeats. At capacity
(4 agent programs plus chat at 2 req/s), four-mode agent throughput rises 18%. Chat attainment at 2 s / 100 ms goes
from 49-52% to 82-84%, chat end-to-end latency from 13.4-13.9 s to 7.9-8.2 s, and agent SLO goodput from 8-17 to 18-24 tok/s
(two-mode: +9% throughput, chat 58% -> 81-82%; two repeats).

**Result: Llama-3.3-70B.** On one 4-GPU H20 node (TP4, EAGLE3 head held at 3.0 accepted tokens per verify in both
arms), with 32-64K agents at concurrency 4 plus chat at 2 req/s, far above capacity: in four-mode, ours cuts chat TTFT
by 85% (6-7 s vs 46 s) and agent TPOT by 45-54%, with throughput +2-3%, at the cost of agent TTFT +23-28% and chat
TPOT +20%. In two-mode, chat TTFT falls 78-81%, agent TPOT 34-35%, and throughput rises 8-9%.

Without the deadline-sized chunk cap (below), the same overloaded load gives agent throughput +8.9% and chat
throughput +9.9% in four-mode (+7.3% / +5.5% in two-mode), agent TPOT -46% (-38%), chat TTFT -84% (-73%), and chat SLO
attainment at 2 s / 100 ms of 27-29% against 0% for stock; chat end-to-end latency halves (30 vs 67 s).

Near capacity (2 agent programs plus chat at 0.5 req/s), chat TTFT falls 83-86% and agent TPOT 17-39%, with agent
throughput +3.6-3.8%. Chat SLO attainment improves from 67% to 72-74% at 2 s / 100 ms, and from 64-66% to 67-68% at
2 s / 50 ms. Chat end-to-end latency falls 10-19%.

Chat's mean TPOT rises (13% four-mode, 57% two-mode), and the reason is instructive. Deadline order serves a chat
prefill before the agent prefill it would have waited behind, so the wait for one agent-chunk step (about 1 s for a
2048-token chunk at TP4) moves from chat's TTFT into its first inter-token gap. The median chat TPOT is lower in
ours (16.0 vs 17.6 ms); the mean is carried by those single first gaps (Figure 18 shows the TTFT / ITL
distributions per tenant). This run also drops a per-step deadline-sized chunk cap that the earlier single-node runs
used. The cap doubled the number of mixed steps chat rode in, and removing it improved every chat metric.

**Result: standalone workloads (no-regression check).** On short-prompt standalone sets served alone on one H200
(Qwen2.5-7B, two repeats each), the full stack matches stock sglang. ShareGPT at 16 req/s, MATH-500 at 16, GSM8K at 32
and Code-Feedback at 16 are all within -0.3..+0.2% throughput, with 99.7-100% chat attainment at 2 s / 50 ms for both
systems; ShareGPT at 32 req/s in two-mode is level too (+0.1%). Their prompts are below the context floor where the
four-mode step and the FA2 tile classes engage, so the system behaves as stock.

The one exception exposed a policy flaw. At ShareGPT 32 req/s in four-mode, a backlog trigger forced short prefills
into mixed steps that cannot replay CUDA graphs: -7.2% throughput and 3.8 s TTFT vs 0.16 s. Requiring a long-context
running row before the trigger fires removes the flaw on the mixed agent workload while improving it there
(GLM-4.5-Air: +27-34% agent throughput, chat attainment 81-82% vs 77-78%); the ShareGPT rerun with this rule is
[pending].

**Bar status.** Throughput: the 10% bar is met for GLM-4.5-Air (+27-28% four-mode, +15-17% two-mode with the final
configuration), at an overloaded point; it is not met for Qwen2.5-7B (-8.5% four-mode, +1.7% two-mode) and narrowly missed for
Llama-3.3-70B (+8.9% agent / +9.9% chat four-mode without the chunk cap; +2-3% with it). Latency: chat TTFT improves by more than 30% for every model; agent TPOT improves by more
than 30% except for Qwen2.5-7B two-mode (-16.6%) and GLM-4.5-Air two-mode (-17..-28%). Rate sweeps, Pareto curves,
near-capacity points and standalone sets are [pending].

**Figures.**

Figure 16: per-tenant TTFT and ITL SLO attainment vs request rate, ours vs stock mixed chunk, prefill-prioritized
(+/- MTP) and PD-Multiplexing, per model and context mixture [pending: rate sweeps].

Figure 17: throughput vs interactivity Pareto curve: output tok/s vs per-user tok/s, one point per load, median marker
with p10-p90 bars and p1-p99 whiskers, per model [pending].

Figure 18: TTFT and ITL / TPOT distributions per tenant (mean, p50, p90, p99), ours vs stock, per model [pending].

### 6.4 Multi-node benchmarks

**Claim.** Symmetric colocated nodes with SLO-checked cache-first routing, a cluster KV pool and the single-node
workers carry more load at SLO than PD disaggregation at its best P:D ratio, and than conditional disaggregation.
Colocation keeps the agent prefix where it was computed; disaggregation re-ships it every round.

**Bar.** At least 30% better TTFT / ITL attainment and at least 10% higher throughput (SLO goodput) than the best
baseline.

**Setup.** 16 and 32 H20s. Qwen2.5-7B runs as TP1 workers; Llama-3.3-70B and GLM-4.5-Air as TP4 replicas. The
workload is fan-out agents on shared 64K-token prefixes plus ShareGPT chat, with load swept on agent program
concurrency and chat rate scaled alongside. Baselines: stock sglang PD with the P:D ratio swept (the headline), and
conditional disaggregation at Dynamo's defaults (2048 tokens / 0.7 of the prompt). The full stack (kernel, four-mode
MTP, EDF, routing, pool) is compared with stock PD with the P:D ratio swept, at higher load and with mixture spikes.
SLOs are those of section 6.1; every result carries per-request spread and per-repeat values.

**Result: scheduler only.** With the kernel off and no speculation, on Qwen2.5-7B on 16 H20s (Table 9), ours holds
>= 90% chat attainment up to 64 programs, and disaggregation only up to 16, across all P:D ratios swept; deadline order
on its prefill pods changes nothing. The gap is the per-round re-ship: conditional disaggregation, which prefills
appends locally, matches ours up to 64 programs. Past that knee ours leads by 4.8 and 11.7 points of chat attainment
and 31-37% agent SLO goodput. This run used a 4096-token remote-prefill threshold for conditional disaggregation; a
rerun at Dynamo's defaults (2048 tokens / 0.7 of the prompt) is [pending]. On 32 H20s under the same protocol, ours
stays >= 96% through 128 programs, while the best disaggregated configuration reaches 83.7% at 32 and 26.7% at 128.

Table 9: scheduler only, Qwen2.5-7B, 16 H20s, fan-out agents + chat; chat attainment / agent SLO goodput (tok/s).

| agent programs | ours | best PD disagg (6 prefill + 2 decode pods) | conditional disagg (4096-token rule) |
|---|---|---|---|
| 16 | 99.5% / 293 | 92.8% / 282 | 99.6% / 275 |
| 32 | 99.0% / 568 | 67.8% / 359 | 98.9% / 518 |
| 48 | 96.6% / 772 | 45.8% / 231 | 96.0% / 635 |
| 64 | 94.2% / 964 | 31.2% / 165 | 94.6% / 845 |
| 96 | 89.2% / 1187 | -- | 84.4% / 906 |
| 128 | 72.6% / 1221 | -- | 60.9% / 890 |

**Result: full stack.** The full stack (unified dispatcher, four-mode MTP, EDF, SLO-checked cache-first routing and
the pool) runs Qwen2.5-7B on 16 H20s (4 pods of 4 GPUs) with fan-out agents and ShareGPT chat. Stock PD runs stock MTP
on every role with the P:D ratio swept; conditional disaggregation runs at Dynamo's defaults on the same workers as
ours. Table 10 reports chat attainment (0.5 s + 20 ms) and agent SLO goodput (10 s + 80 ms).

Table 10: full stack, Qwen2.5-7B, 16 H20s; chat attainment (0.5 s + 20 ms) / agent SLO goodput (tok/s, 10 s +
80 ms). Two repeats, r1 / r2.

| agent programs + chat/s | ours | conditional disagg (2048 / 0.7) | stock PD, 2 + 2 pods | stock PD, 3 + 1 pods |
|---|---|---|---|---|
| 32 + 8 | 97.4 / 97.3% ; 601 / 611 | 97.8 / 97.7% ; 527 / 547 | 40.1 / 37.5% ; 107 / 93 | 53.2 / 52.7% ; 162 / 148 |
| 64 + 16 | 93.6 / 93.8% ; 1100 / 1093 | 96.0 / 95.5% ; 821 / 932 | 13.4 / 17.3% ; 52 / 55 | 13.1 / 3.2% ; 80 / 56 |
| 96 + 24 | 90.0 / 90.3% ; 1403 / 1463 | 90.5 / 89.3% ; 1136 / 1181 | 10.0 / 5.4% ; 15 / 12 | 0.0 / 2.5% ; 4 / 25 |
| 128 + 32 | **79.1 / 77.1% ; 1564 / 1532** | 71.4 / 69.4% ; 1143 / 1212 | 0.3 / 4.9% ; 6 / 6 | 0.1 / 0.0% ; 6 / 7 |
| 64 + 16, + 32-program spike from 300 s for 300 s | 93.2 / 93.7% ; 1053 / 1064 | 94.4 / 93.9% ; 873 / 855 | 8.8 / 6.2% ; 32 / 38 | 8.0 / 2.1% ; 46 / 31 |

Under the paper's SLO matrix (chat 2 s + 50 ms mean TPOT; agents 10 s + 80 ms), ours keeps 95.5-98.9% chat attainment
and 96.8-98.4% agent attainment at every load, against 90.4-99.6% / 91.3-98.4% for conditional disaggregation and
0-60% / 2-63% for stock PD; the goodput figures are unchanged (r1 / r2 identical within 1%).

Stock PD with one prefill pod reaches 1.6% / 2 tok/s at 64 + 16. Against stock PD at its best split, ours has 97.4 vs
53.2% chat attainment at 32 programs and 93.6 vs 13.1% at 64, and 3.7x the agent SLO goodput at 32 and 14x at 64; stock
PD holds no load point at >= 90%. Against conditional disaggregation, ours has more agent SLO goodput at every load and
in both repeats:
- 32 programs: +14 / +12%
- 64 programs: +34 / +17%
- 96 programs: +24 / +24%
- 128 programs: +37 / +26%
- under the spike: +21 / +24%

Chat attainment is within 0.4-2.4 points through 96 programs (conditional disaggregation slightly higher) and 7.7-7.8
points higher for ours at 128 programs. There, conditional
disaggregation's chat TPOT p90 reaches 46.6 ms (ours 24.8 ms). The full stack adds 14% agent SLO goodput over the
scheduler-only run at 64 programs (1100 vs 964).

**Result: Llama-3.3-70B, full stack, 16 H20s (4 TP4 replicas).** Under the paper's SLOs (Table 10b), ours carries
the most agent SLO goodput at every load: 6-94% more than conditional disaggregation and 2-7% more than stock PD at its
best split (level under the spike). Ours holds 90.8-98.1% chat attainment, against 79.6-91.4% for stock PD and
97.7-98.5% for conditional disaggregation. Against conditional disaggregation the trade is therefore agent goodput for
chat attainment (-0.3 to -7.1 points). With only four replicas, a 70B prefill chunk on a colocated node holds every
co-resident chat decode for one long step, and the router has four nodes to choose from rather than sixteen.
Against stock PD's best split at each load (2 + 2 or 3 + 1), ours holds 3-16 points more chat attainment and 2-7% more
agent SLO goodput (under the spike: +5.9 points, level goodput). [pending: second repeat]

Table 10b: Llama-3.3-70B, 16 H20s, one repeat; chat attainment (2 s + 50 ms) ; agent SLO goodput, tok/s (agent
attainment, 10 s + 80 ms).

| agent programs + chat/s | ours | conditional disagg | stock PD 2 + 2 | stock PD 1 + 3 | stock PD 3 + 1 |
|---|---|---|---|---|---|
| 4 + 1 | 98.1% ; 57 (95.0%) | 98.4% ; 54 (97.5%) | 91.4% ; 55 (95.1%) | 84.0% ; 54 (94.8%) | 92.7% ; 53 (94.5%) |
| 8 + 2 | 95.4% ; 105 (95.1%) | 98.2% ; 83 (98.2%) | 85.0% ; 98 (93.8%) | 67.7% ; 87 (90.4%) | 92.3% ; 94 (94.5%) |
| 12 + 3 | 93.4% ; 139 (94.2%) | 97.7% ; 99 (93.7%) | 79.6% ; 136 (93.1%) | 27.1% ; 110 (89.4%) | 83.1% ; 122 (94.4%) |
| 16 + 4 | 90.8% ; 165 (94.3%) | 97.9% ; 85 (73.8%) | 68.2% ; 160 (90.7%) | 1.3% ; 93 (79.2%) | 74.9% ; 123 (91.9%) |
| 8 + 2, + 4-program spike | 95.8% ; 101 (95.0%) | 98.5% ; 70 (93.9%) | 86.9% ; 101 (95.1%) | 63.9% ; 92 (92.7%) | 89.9% ; 99 (96.7%) |

**Result: cache pressure.** With the working set grown from 1M to 8M KV tokens against a 6.7M-token pool,
disaggregation loses 55% throughput and 28.5 points of cached fraction, ours 10.6% and 3.1 points, in earlier
settings; a rerun with the full stack is [pending].

**Bar status.** Against stock PD at its best split the bar is met at every measured load. Against conditional
disaggregation, the goodput side is met at every load in both repeats (+12..+37%); chat attainment is level through 96 programs and
ahead only at 128 programs. 32 GPUs with the full stack, Llama-3.3-70B (4 TP4 replicas on 16 H20s) are [pending];
GLM-4.5-Air follows Qwen2.5-7B and Llama-3.3-70B.

**Figures.**

Figure 19: capacity at SLO: per-tenant TTFT / ITL attainment and SLO goodput vs load, for ours, each P:D split of
stock PD, and conditional disaggregation, with the load each system holds at >= 90% marked; 16 H20s, Qwen2.5-7B
(scheduler only and full stack measured). 32 GPUs full stack, Llama-3.3-70B, GLM-4.5-Air [pending].

Figure 20: throughput vs per-user tok/s Pareto curve (p10-p90 bars) at 16 and 32 GPUs, ours vs baselines [pending].

Figure 21: scaling: load held at >= 90% attainment on 16 vs 32 GPUs, ours vs best PD disaggregation (scheduler only
measured; full stack [pending]).

Figure 22: cached-prefix fraction, re-prefilled tokens and agent tok/s vs working-set size (1M, 4M, 8M KV tokens
against a 6.7M-token pool), ours vs stock PD; 16 H20s, 32 agent programs (earlier settings measured; full stack
[pending]).

Figure 23: load spike and mixture shift: per-tenant attainment and throughput in 30 s windows around a 32-program
spike starting at 300 s and lasting 300 s, on top of 64 programs + 16 chat/s; ours, conditional disaggregation, stock
PD. Mixture shifts [pending].

### 6.5 Fault tolerance

Symmetric nodes and rerouting away from dead workers should keep both tenants' SLOs when workers fail (C4). We will
kill one and then two workers mid-run, in prefill and in decode roles for the baseline (stock PD at its best P:D split,
health-check ejection on) and the same workers for ours, with the full stack on 16 GPUs at the stage-3 workload and
load, for all three models. When a prefill worker dies, ours should hold, because every node can prefill and the SLO
router moves new prefills to the survivors. When a decode worker dies, the baseline keeps its admission rate and loses
ITL on the remaining decode pods; ours should keep ITL by admitting fewer prefills (SLO check and EDF) and give up some
TTFT attainment instead. The prefill-role and decode-role kills with the full stack are [pending].

In earlier settings (8 GPUs, scheduler only, 2 of 8 GPUs killed at 300 s), ours reroutes (agent attainment 96.4 ->
98.1%, chat 85.0%), while PD disaggregation loses half its prefill capacity (18 -> 34%, chat 9-15%).

Figure 24: per-tenant TTFT and ITL attainment and throughput in 30 s windows before, during and after each kill, with
lost requests and recovery time; ours vs stock PD, three models [pending].

## 7. Ablation study

**Claim.** The ablations isolate which parts of C1-C4 carry the gains: for the kernel, the one plan and in-launch
merge, PDL overlap, longest-first order and the executor choice; for the schedule, demotion of missed prefills; for
the cluster, the routing policy.

**Bar.** Each mandatory part of C1 should cost measurably when removed, and each optional mechanism is kept only if it
beats the default.

**Setup.** Kernel ablations run on the 96 trace-sampled H200 captures of section 6.2 (Qwen2.5-7B), or on their FA2 rows
alone; each value is the time-weighted speedup for 32-64K two-mode / 32-64K four-mode / 128-256K two-mode / 128-256K
four-mode. Schedule and router ablations run on the workloads of sections 6.3 and 6.4. Llama-3.3-70B kernel ablations
are [pending].

**Result: the plan and the merge.** The unified plan with one in-launch merge is 1.027-1.047x faster than the two-launch
baseline (two plans, separate merge launches, same FA3 and FA2 kernels) on the H200, and level on H20. Removing every
check-in fence, counter atomic and merge item (a timing bound with wrong output) gives 1.009 / 0.980 / 1.004 / 0.988x,
so the unified merge costs <= 0.9% of the step. A full-step timeline agrees: merge time left after both FA3 and FA2
compute is 0.2-1.0% of the step, and 0 on steps where FA3 sets the step time. The reason the merge is so cheap is the
step composition: most captured steps have very few FA2 requests (for example one prefill chunk plus two verify rows),
so there is little work to balance or reduce. Synthetic decode-heavy batches of 16-1024 requests, in the setting of the
cross-SM fusion work discussed in section 8, will show where the reduction matters [pending].

**Result: overlap and order.** Running FA2 and FA3 serially instead of overlapping them under PDL gives 0.939 / 0.967 /
0.975 / 0.978x, so the overlap is worth 2.2-6.1% of the step. Enumeration order instead of longest first gives 0.993 /
0.996 / 1.001 / 1.001x.

Table 11: kernel ablations, H200, Qwen2.5-7B, 96 captures (32-64K two / 32-64K four / 128-256K two / 128-256K four).

| ablation | relative to the unified dispatcher |
|---|---|
| merge-free bound (output wrong) | 1.009 / 0.980 / 1.004 / 0.988x |
| FA2 and FA3 serial instead of PDL overlap | 0.939 / 0.967 / 0.975 / 0.978x |
| enumeration order instead of longest first | 0.993 / 0.996 / 1.001 / 1.001x |

**Result: FA2 executor.** With the same plan and the same merge, Table 12 compares FA2 executors against the default
non-persistent BatchPrefill body. Persistent dispatch is level with or slightly behind hardware-launched CTAs;
releasing merges as soon as their partials are ready, through a global ready queue, adds polling that slows the
compute. Reducing partials inside a thread-block cluster over DSMEM is exact (FP32 and graph replay pass on all 96
captures) but slower at every cluster size measured: to fit one split group into one cluster, the plan raises the FA2
KV chunk to at least the maximum KV length divided by the cluster size C, which leaves the longest request too few
splits.

Table 12: FA2 executor, same plan and merge, H200 (32-64K two / 32-64K four / 128-256K two / 128-256K four).

| executor | relative to the non-persistent BatchPrefill body |
|---|---|
| BatchPrefill body on a persistent LPT pull loop | 0.987 / 0.998 / 0.996 / 0.993x |
| flashinfer's persistent runner, plan-chosen 2-3 CTAs/SM | 0.993 / 0.991 / 1.006 / 0.997x |
| persistent loop + merges released from a global ready queue as partials complete (vs the same loop with ticketed merge items) | 0.997 / 0.982 / 1.003 / 1.005x |
| DSMEM cluster reduction (cluster barrier, no global sync), C = 2 | 0.597 / 0.667 / 0.622 / 0.703x |
| DSMEM cluster reduction, C = 4 | 0.81-0.87x of the two-launch baseline |
| DSMEM cluster reduction, C = 8 | 0.925 / 0.927 / 0.913 / 0.940x |
| DSMEM cluster reduction, C = 16 (non-portable cluster) | 0.967 / 0.984 / 0.967 / 0.978x |
| DSMEM cluster reduction, splits uncapped, groups larger than a cluster reduced by their last member, C = 4 (vs the in-launch merge, same build) | 0.993 / 0.988 / 0.992 / 0.993x |
| the same, C = 8 / 16 | 0.974-0.990x / 0.955-0.983x |

On FA2 rows alone, the plan-scheduled merge items took 4.86 ms, against 6.21 ms for a merge done by the last CTA to
arrive and 5.36 ms for stock flashinfer BatchAttention. flashinfer's persistent runner needed two register-targeted
instantiations, chosen by the plan, to reach parity: 3 CTAs per SM for decode-only plans, and 2 with deeper KV tiles
for verify plans. On FA2 rows alone on the H20, the persistent executor takes 5.37 / 6.66 ms, the non-persistent one
5.23 / 6.42 ms, and the two-launch baseline's FA2 launch 5.46 / 7.17 ms (two- / four-mode).

**Result: KV multicast.** We bounded multicast before building it. For FA2, only 0.3-0.9% of KV page reads across the
96 captures are shared with another tile. For FA3, 99% of reads are re-reads by the same prefill row's other query
tiles, but they hit in L2: with FA3's KV loads compiled out, the step is only 0.1-1.3% faster (Qwen2.5-7B, H20, 92
captures). Multicast for either executor is therefore bounded under about 1%. We built it anyway for FA3, as a 2-CTA
exchange: page-size-1 KV cannot use TMA multicast, so each CTA gathers half of each tile and pushes it to its peer over
DSMEM. It delivers tiles 1.3-1.5x faster in isolation and is exact on all 96 steps, but in the step it runs
0.76-0.77x. Pairing two CTAs in lockstep alone costs 6%, more than the ceiling, and the push stalls FA3's two-stage
pipeline. FA3's KV loads are already hidden; the step's lever is overlap between the two executors, not KV traffic.

**Result: tile classes and KV splits.** A 48-row FA2 tile vs FA3 for rows of 33-48 packed queries is [pending]. GLM-4.5-Air's four-mode verify rows pack
48 queries; sending them to FA3 lost 28% against the two-launch baseline (0.724x), and a 64-row FA2 class built from the
BatchPrefill tile recovers it (1.000x), so rows up to 64 packed queries stay on FA2. (An earlier control in which FA3 beat
a 64-row FA2 class by 1.02-1.59x measured the old persistent FA2 runner, not this tile.) Among
KV split policies, measured against the two-launch baseline before merge items existed, a fixed split gave
0.95-0.98x, flashinfer's KV-length limit 0.89-0.94x and a dynamic makespan split 0.90-0.96x; the fixed split is the
default. Running decode and verify rows on FA2 instead of FA3 (the role split) keeps throughput equal and improves
agent ITL by 15-25 points.

**Result: schedule.** Per-tenant deadlines alone change nothing; adding demotion of missed prefills cuts chat TTFT by
96-98%. Demoting by feasibility at a fixed processing rate instead demoted every cold 128-256K agent on arrival. Capping
the chunk size per step to protect chat TPOT starves prefill: the cap binds on 86% of steps.

**Result: router.** Cache-first routing with the SLO check vs cache-only vs round-robin is [pending].

**Figures.**

Figure 25: kernel ablation bars: time-weighted speedup per ablation (Tables 11 and 12) per (context, mode); H200,
Qwen2.5-7B measured; Llama-3.3-70B [pending]. KV-split sizing matters on large batches: rounding the per-slot share up to 256 keys instead of a power of two keeps
the longest unsplit row from setting the step. It is 1.003-1.006x on the trace-sampled steps and up to 1.28x on
synthetic shared-prefix, long-tail and math batches. Figure 25b: the same executors on synthetic decode / verify
batches of
16-1024 rows (ShareGPT, math, long-tail and shared-prefix length mixes), time relative to the default executor vs
batch size [pending: decode-only instantiation and chunk rounding].

Figure 26: schedule and router ablations: per-tenant attainment and throughput for stock, deadlines alone, deadlines
with demotion, and each routing policy, on the section 6.3 / 6.4 workloads. Router [pending].

## 8. Background and Related Work

**Serving frameworks.** vLLM, sglang and TRT-LLM are the common serving engines; NVIDIA Dynamo is a scheduling layer
above them. We build on sglang and compare against its stock schedules.

**Disaggregation.** DistServe, Mooncake and Splitwise put prefill and decode on separate nodes. semi-PD splits SMs
between the phases with MPS and needs a restart to change the split. DynaServe splits a request's work across two
nodes. Dynamo's conditional disaggregation prefills short appends locally on the decode worker. All of them fix roles or
resource shares ahead of time, or move KV between nodes; we keep nodes symmetric and route by
cache, and use conditional disaggregation as our strongest baseline.

**Colocation and multiplexing.** Sarathi-Serve introduced chunked prefill, which bounds decode stalls by cutting
prefill into chunks run beside decode. PD-Multiplexing (MuxWise) runs prefill and decode in separate SM groups with
green contexts. POD-Attention fuses prefill and decode attention in a single FA2 kernel. Neither supports speculative
decoding: POD-Attention's hybrid batch has one-token decode rows only, and sglang's multiplexing loop runs the whole
target forward for extend batches and does not carry the EAGLE draft state through its layer-split prefill. Ours
gives verify rows their own tile class, so the verify forward absorbs the prefill chunk (C2). Ours keeps two separate
executors at their own layouts and shares one plan between them, because on Hopper a single kernel cannot overlap MMA
instructions of different shapes (section 5.1).

**Kernel fusion.** FlashAttention-3 specializes warps on Hopper; persistent kernels keep CTAs resident and pull work;
flashinfer schedules attention work holistically across a batch. MELD ("Taming Dynamism on GPUs: Cross-SM Kernel
Fusion via SM Cooperation and Just-in-Time Reduction", SOSP '26) fuses across SMs and reduces partials just in time
over DSMEM. We tested both persistent dispatch and DSMEM cluster reduction as executor alternatives (section 7); on our
captured steps neither beats the non-persistent executor with the plan-scheduled merge.

**Attention-FFN disaggregation** separates attention and feed-forward layers onto different devices; it is orthogonal
to this work.

## 9. Conclusion and limitations

**Conclusion.** Role-separated attention under one plan, a deadline-ordered four-mode step and symmetric cache-first
routing raise SLO attainment on one node and the load held at SLO on a cluster, without a P:D ratio to tune.

**Limitations.**

- On one GPU, chat TPOT rises 35-56% when chat is admitted during an agent burst; holding 20 ms beside 32-64K prefill
  would need chunks below about 200 tokens.
- 128-256K agents have no single-GPU operating point at a 10 s TTFT SLO.
- Four-mode agent throughput falls 8.5% at 32-64K on one node.
- Measured on Hopper (H200, H20) only. On Blackwell, FA4 replaces FA3 behind the same plan and FA2 runs unchanged
  (section 5.6); that port is not yet measured.

## Artifact notes

- Software base: every arm runs upstream sglang main at `3fef129ff` (2026-10-08); ours is the overlay on that tree.
  Conditional disaggregation is the `cond_disagg` policy in our router.
- Build names: the two-launch baseline is the "native v3" build; the unified dispatcher is the A6 build; the
  role-split serving build is the earlier serving build of `debug_kernel_server_perf_4_mode.md`.
- Merge naming in the task docs: the plan-scheduled in-launch merge is the "ticketed" merge; merges released from a
  global ready queue are the "JIT (ready queue)" mode; the DSMEM cluster reduction is the DSMEM-cluster JIT mode. The
  synthetic decode-heavy batches are the "MELD-style" synthetic set (bs 16-1024).
- Capture sets: "32-64K two / four" and "128-256K two / four" are the four 24-step capture cells per model.
- Task-doc references: the section 2.2 / 6.3 scheduler-only single-node run is S0b in
  `debug_kernel_server_perf_4_mode.md`; the planner and FP32 gate are under A6 there; the P:D sweep (section 2.3) is
  capacity phase A, the thrashing sweep (2.4, 5.4, 6.4) is "thrashing", and the deadline-ordered disaggregation run
  (3.2) is "disagg + EDF" in `pd_colocation_vs_disagg.md`.
- Measurement dates: register / spill table, H200, `-Xptxas -v` per named function, 2026-10-06; SM-sharing bound on
  TraceLab, 2026-10-17; persistent 48/32/16 runner spill and FA2-rows-alone merge comparison, 2026-10-24; DSMEM cluster
  reduction, 2026-10-25.
- KV split policy "flashinfer's KV-length limit" is flashinfer's `kv_len_limit`.
- The KV pool is Mooncake's L3 store with sglang's write-through-selective policy.
- PD-Multiplexing is the `pdmux` baseline.

## References

Papers and systems this work draws on. *(sent)* marks the ones the user sent. Each entry: summary, relation to our
method, and what we take from it -- only where it strengthens our own contributions (fused kernels, higher
cluster-wide request volume, scheduling flexibility under load spikes / mixed workloads / workload variation).

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
