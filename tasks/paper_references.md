# Reference list for the AttentionFuse paper

Every paper the user has sent, in the four-bullet form the project requires: a summary of the method and how it is
evaluated, the core method and claimed contributions, its relation to our method, and what we take from it -- used
only to strengthen our own contributions (the fused kernels, higher cluster-wide request volume, and scheduling
flexibility under load spikes, mixed workloads and workload variation), never stacked onto our system.

Moved out of `paper_outline.md` on 2026-10-28: the draft runs to 70+ pages and an annotated bibliography is a
working artifact, not paper text. In the submission these entries become ordinary citations; this file is where the
reading and the "what we can learn" notes live.

Papers and systems this work draws on. Each entry gives a summary, the core method, its relation to our method, and
what we take from it for our own contributions (fused kernels, higher cluster-wide request volume, scheduling
flexibility under load spikes, mixed workloads and workload variation).

- **Taming Dynamism on GPUs: Cross-SM Kernel Fusion via SM Cooperation and Just-in-Time Reduction**
  - *Summary:* fuses dynamically sized, heterogeneous work into one persistent launch whose CTAs pull tasks from a
    device-side queue, with SMs cooperating on shared tasks and split partials reduced just in time instead of by a
    separate merge kernel; evaluated on kernel-level dynamic workloads against per-task launches.
  - *Core method/contributions:* MeldKernel ("Meld"), a compiler framework for cross-SM kernel fusion of dynamically
    shaped work: SM cooperation over on-chip SM-to-SM interconnects (clusters) splits phase-level dependencies into
    sub-task-level ones, so a kernel sequence becomes one cross-SM pipeline; just-in-time reduction lands remote SMs'
    partial states in local shared memory by asynchronous transfer before the reduction needs them; a decoupled
    execution model maps the declared computation onto SMs at runtime from input shapes. 1.3x average on decode
    attention over the best existing solution (from the abstract; the ACM full text was not accessible).
  - *Relation:* the closest kernel-fusion work. We built its two mechanisms as executor alternatives:
    - a persistent executor that pulls plan items;
    - just-in-time reduction of split-KV partials over DSMEM within a thread-block cluster.

    On captured serving steps, neither beats the non-persistent FA2 executor with the plan-scheduled in-launch merge
    (0.987-0.998x and 0.955-0.993x; section 7). We also do not co-compile roles with different register needs into
    one body: the co-compiled FA3 + FA2 kernel was 2.16x slower, so FA2 runs as a separate executor chained to FA3 by
    PDL.
  - *Learn:* its framing -- dynamism inside one launch is the problem, cooperation across SMs the answer -- is the
    story for our kernel contribution, applied to the 4-mode serving step (prefill chunk + verify rows), which a
    kernel-only benchmark does not exercise; report the same kernel-level evidence (makespan vs per-mode launches,
    SM idle time) on captured serving steps.
- **SLOs-Serve: Optimized Serving of Multi-SLO LLMs** -- S. Chen, Z. Jia, S. Khan, A. Krishnamurthy, P. B. Gibbons,
  [arXiv:2504.08784](https://arxiv.org/abs/2504.08784)
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
    frame in which our scheduling flexibility under mixed workloads shows; it is the source of our mixture x load
    setup (section 6.1).
- **Towards High-Goodput LLM Serving with Prefill-decode Multiplexing (MuxWise)** -- Y. Chen, W. Cui et al.,
  [arXiv:2504.14489](https://arxiv.org/abs/2504.14489)
  - *Summary:* intra-GPU prefill/decode multiplexing (SM partitions via green contexts, layer-split prefill) with an
    SLO-aware dispatcher and a contention-tolerant latency estimator; goodput under SLO as the metric.
  - *Core method/contributions:* intra-GPU prefill-decode multiplexing -- the two phases share a GPU but execute
    independently, with compute allocation adapted dynamically and decoupled from memory management; three
    components: a bubble-less multiplex engine, a contention-tolerant estimator of co-located latency, and an
    SLO-aware dispatcher. 2.20x average (up to 3.06x) peak throughput under SLO over state-of-the-art baselines.
  - *Relation:* direct colocation baseline (= sglang `--enable-pdmux`, run 2026-10-17); our split is decided per
    step inside one forward and co-runs prefill with MTP verify rows, where pdmux partitions SMs at model level.
  - *Learn:* its estimator is the comparison point for our step model: held-out MAPE is 2.0% for the fused step vs
    1.9% for sglang's single FA3 call over all steps (section 5.2.2). Predictability of the fused step is our argument,
    not theirs.
- **Crossflow: Prefill–Decode Elasticity for Agentic LLM Serving** -- Y. Xu et al.,
  [arXiv:2609.27085v1](https://arxiv.org/abs/2609.27085v1)
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
  - *Learn:* its ITL regression under decode-local prefill is exactly the cost our fused step removes -- report
    ITL alongside TTFT on the same replay; its timestamp-compression protocol is the request-rate load axis for
    chat-like tenants (agent tenants use program concurrency, below). It also confirms that prefill and decode demand
    varies widely during online serving.
- **ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System** -- H. Kang et al.,
  [arXiv:2602.13692](https://arxiv.org/abs/2602.13692)
  - *Summary:* program-aware scheduling for agentic workflows; shows KV-cache thrashing (a session's cache evicted
    during a tool call and re-prefilled, up to 7.14x end-to-end); load axis is the number of parallel programs kept
    active (headline 96 programs on 8x H100).
  - *Core method/contributions:* models agentic workflows as *LLM Programs*, one view of KV caches, system state and
    tool assets (disk, network ports); a program-aware scheduler that maximizes KV-cache hit rate and mitigates memory
    imbalance, and a tool resource manager that prepares tool environments asynchronously. 1.5-3.6x serving and
    1.8-3.9x RL-rollout throughput, up to 4.2x disk-memory savings over state-of-the-art inference systems.
  - *Relation:* motivates cache-first routing with replication. Routing a request to a non-holder leaves a replica
    where demand is, whereas disaggregation pins a session's cache to its prefill node.
  - *Learn:* **load protocol we follow** -- agent load is program concurrency C, never sessions/s; every agent arm
    reports C and measured mean in-flight requests. Its thrashing measurement is the evidence for our cluster-wide
    request-volume argument (cache replication raises sustainable volume).
- **TraceLab: Characterizing Coding Agent Workloads for LLM Serving** -- K. Zhu, M. Jacob, C. Ma, Y. Pan, S. Wang,
  A. Krishnamurthy, B. Kasikci, [arXiv:2606.30560](https://arxiv.org/abs/2606.30560); dataset
  `syfi_coding_trace.jsonl.gz` ([github](https://github.com/uw-syfi/TraceLab))
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
  [arXiv:2504.20068](https://arxiv.org/abs/2504.20068))
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
    fused 4-mode step makes the step cheaper and predictable (section 5.2.2), and that colocation lets the cluster route a
    request anywhere. Its batch-composition finding (mixing input lengths slows a batch) is the inefficiency our
    fused kernel attacks inside the step rather than by keeping lengths apart.
  - *Learn (evaluation, not method):* (1) report **service goodput by request type** next to means -- chat as
    latency-sensitive (token i on time if done by TTFT_SLO + i x TBT_SLO), agent rounds as deadline-sensitive or
    compound (a program's rounds share one deadline); this is the metric under which scheduling flexibility under
    mixed load is visible. (2) Set SLOs from an external reference (their P95-of-API method) or relative to the
    unloaded latency of the same request on stock MTP, and sweep a common SLO scale (0.8x-1.5x). (3) Its fixed per-type SLOs ignore prompt length; for TraceLab, whose rounds span 32K to > 256K, a tiered
    TTFT SLO (2x for > 200K uncached tokens, as we use; TokenScale also tiers by length) avoids
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
    radix cache off and no speculative decoding, so it cannot serve our MTP arms as released; it has not been run (its build is
    blocked).
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
  - *Relation:* closest multi-node baseline for agentic rounds: most agent append rounds already avoid the per-round
    KV transfer that disaggregation pays (section 3.2). What stays ours: making local prefill +
    speculative decode co-execution efficient (4-mode step + fused kernel), symmetric nodes instead of fixed pools,
    and EDF/SLO-aware admission. Must be run as a baseline, not argued around.
  - *Learn:* its fixed local-prefill thresholds and a planner that reacts by changing worker counts (5 s decisions in
    load-based mode, 180 s in throughput mode) are the comparison for scheduling flexibility under load spikes --
    evaluate a step change in load and report the time to recover TTFT / ITL, where our per-request routing and
    per-step prefill/decode mix react without adding or removing workers.
- **FlowPrefill: Decoupling Preemption from Prefill Scheduling Granularity to Mitigate Head-of-Line Blocking in LLM
  Serving** -- C. Hsieh, Z. Zong, X. Chen, J. Li, J. Zhai, L. Wen, [arXiv:2602.16603](https://arxiv.org/abs/2602.16603)
  ([code](https://github.com/HSIEHCHIACHI/FlowPrefill))
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
  J. Mohan, S. Peter, R. Ramjee, A. Panwar, ASPLOS '25 ([arXiv:2410.18038](https://arxiv.org/abs/2410.18038))
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
  J. Mohan, N. Kwatra, B. S. Gulavani, R. Ramjee, [arXiv:2308.16369](https://arxiv.org/abs/2308.16369)
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
    rows to the 16 / 32 / 64-row FA2 tile classes, and reduces all splits in one merge.
  - *Learn:* its analysis isolates exactly the cost piggybacking cannot hide -- attention -- and makes the P:D ratio
    and chunk size the knobs; report our attention and step time against stock chunked prefill on the same axes (P:D
    ratio x chunk size, including non-multiples of the tile), so the attention share SARATHI leaves is the part our
    kernel is shown to remove.
