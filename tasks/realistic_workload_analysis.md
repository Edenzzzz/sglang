# Realistic Workload Analysis for v3 Persistent Kernel

## Goal
Identify which real-world agent workloads benefit most from v3 persistent kernel's prefill-decode overlap, and determine realistic serving parameters (arrival rates, input/output distributions) for benchmarking.

## Context
- v3 dynamic scheduler results: see `tasks/debug_kernel_server_perf.md` (Benchmark Summary section)
- Workload survey & dataset catalog: `workloads/agent_workload_survey.md`
- Dataset length stats & scripts: `workloads/AGENTS.md`
- Length stats data: `workloads/length_stats.json`
- Turn stats data: `workloads/turn_stats.json`
- bench_serving 3rd-party support: `python/sglang/bench_serving_3rdparty.py`

## Server Launch Command
```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --attention-backend flashinfer \
  --watchdog-timeout 999999 \
  --mem-fraction-static 0.9 \
  --enable-piecewise-cuda-graph \
  --enable-mixed-chunk
```

## TODOs

### 1. Prefill/decode ratio sweep
- [x] Swept 6 workloads × 3 configs (Default-mixed-chunk, Persistent v3, POD). Results in table below.
- [x] Hypothesis partially confirmed: v3 benefits from **large absolute prefill** (>4096 tokens), not balanced ratios. v3 wins throughput only on 6500/7500, wins TTFT on 4096/1024 (-24%).
- [x] v3 has consistent 0.5-4% throughput regression vs default — needs investigation (see `debug_kernel_server_perf.md` TODO)

### 2. Realistic arrival rates
- [ ] Research LLM serving papers that use the trace datasets in `workloads/agent_workload_survey.md` to find realistic request arrival rates
- [ ] Papers to check: Autellix, Continuum, ServeGen, BurstGPT, Cost of Dynamic Reasoning, Agent.xpu, Throughput-Optimal Scheduling (refs 1-8, 19-20 in survey)
- [ ] bench_serving currently uses Poisson rate=8 req/s — determine if this is realistic

### 3. First-turn output length stats (with reasoning)
- [ ] Run `count_workload_lengths.py --run-inference --num-samples 200` on trace datasets only (skip single-step/no-trace datasets for now)
- [ ] **Use 200 samples per dataset.** ~20-30 min per short-input dataset, ~1-2h for Nebius-SWE.
- [ ] Note: `length_stats.json` has **static field tokenization** only — NOT model inference output. Model output with thinking is 3-8x longer (e.g. CodeActInstruct: static median=354 vs inference median=1,174)
- [ ] Already cached in `output_length.json`: SWE-bench, CodeActInstruct, SWE-smith
- [ ] Batch 1 (running): Mind2Web, openhands-feedback, AgentInstruct, SWE-Gym
- [ ] Batch 2: ShareGPT, Code-Feedback, Nebius-SWE-Trajectories
- [ ] Skip (no traces / single-step): Go-Browse, NNetNav-live, NNetNav-wa, Synatra, Orca-AgentInstruct

### 4. Per-turn context growth
- [ ] Run Qwen3-8B with thinking + tool calls for a few turns on multi-turn trace datasets (must include ≥1 tool call per trace)
- [ ] Measure per-turn: new input tokens (tool call results), output tokens (model response with thinking), cumulative context
- [ ] Key datasets: AgentInstruct, CodeActInstruct, Nebius SWE Trajectories, openhands-feedback
- [ ] Previous analysis used static dataset text only (not model inference) — needs redo with actual model output

### 5. Prioritize workload for v3 kernel benchmark
**Precondition:** Complete kernel optimization tasks in `tasks/debug_kernel_server_perf.md` first (FA3 tile pattern, v3 throughput regression). If performance improves, redo task 1 (ratio sweep) with updated kernel before prioritizing workloads.
- [ ] Based on ratio sweep findings (task 1, possibly redone with improved kernel), identify input/output length sweet spot for v3
- [ ] Match against real workload distributions (tasks 3-4) to find which workload/trace benefits most
- [ ] Run full 5-config server benchmark on that workload using bench_serving_3rdparty.py

## Results

### Ratio Sweep Results

H100 80GB, Qwen3-8B, `--mem-fraction-static 0.85 --enable-piecewise-cuda-graph`, Poisson rate=8, 100 prompts, keepalive OFF.

| Input | Output | Ratio | Default tok/s | v3 tok/s | POD tok/s | v3 vs Def | POD vs Def |
|------:|-------:|------:|--------------:|---------:|----------:|----------:|-----------:|
| 1024 | 1024 | 1.0 | 5068 | 4889 | 4989 | -3.5% | -1.6% |
| 4096 | 1024 | 4.0 | 10784 | 10727 | 10670 | -0.5% | -1.1% |
| 2048 | 2048 | 1.0 | 6490 | 6355 | 6429 | -2.1% | -0.9% |
| 1024 | 4096 | 0.25 | 4300 | 4127 | 4291 | -4.0% | -0.2% |
| 4096 | 4096 | 1.0 | 5451 | 5413 | 5461 | -0.7% | +0.2% |
| 6500 | 7500 | 0.87 | 3612 | 3682 | 3601 | **+1.9%** | -0.3% |

| Input | Output | Def TTFT med | v3 TTFT med | POD TTFT med | v3 vs Def | POD vs Def |
|------:|-------:|-------------:|------------:|-------------:|----------:|-----------:|
| 1024 | 1024 | 98ms | **84ms** | 302ms | -14% | +209% |
| 4096 | 1024 | 206ms | **156ms** | 339ms | **-24%** | +65% |
| 2048 | 2048 | 47ms | **46ms** | 51ms | -1% | +8% |
| 1024 | 4096 | 36ms | 37ms | 41ms | +3% | +13% |
| 4096 | 4096 | 138ms | **128ms** | 174ms | -7% | +26% |
| 6500 | 7500 | 548ms | 610ms | 539ms | +11% | -2% |

| Input | Output | Def ITL med | v3 ITL med | POD ITL med |
|------:|-------:|------------:|-----------:|------------:|
| 1024 | 1024 | 10.14 | 10.68 | 10.47 |
| 4096 | 1024 | 16.15 | 16.25 | 16.32 |
| 2048 | 2048 | 13.14 | 13.33 | 13.18 |
| 1024 | 4096 | 13.08 | 13.36 | 13.17 |
| 4096 | 4096 | 19.13 | 19.05 | 19.12 |
| 6500 | 7500 | 26.17 | 25.05 | 26.15 |

**Key findings:**
1. **v3 wins throughput only on longest workload** (6500/7500: +1.9%). On all others, v3 is 0.5-4% behind default.
2. **v3 wins TTFT on prefill-heavy workloads**: 4096/1024 is the sweet spot (156ms vs 206ms, -24%). This is the prefill-decode overlap benefit.
3. **POD consistently worse TTFT than both default and v3**, especially on prefill-heavy workloads (302-339ms vs 84-156ms for 1024/4096). POD JIT warmup may not fully eliminate startup penalty.
4. **ITL is nearly identical across all configs** (~±0.3ms). Decode performance is the same.
5. **v3 throughput regression is consistent** (-0.5% to -4%), worst on decode-heavy workloads (1024/4096: -4%). Likely FlippedSchedule template overhead on pure-decode batches.
6. **Input/output ratio doesn't clearly predict v3 benefit**: the 6500/7500 win (ratio 0.87) and the 4096/1024 TTFT win (ratio 4.0) suggest v3 benefits from large absolute prefill tokens, not the ratio itself.

### Arrival Rate Research

| Paper | Arrival Model | Rate (req/s) | Model Size | Notes |
|-------|--------------|-------------|------------|-------|
| Cost of Dynamic Reasoning | Poisson | ShareGPT: 6.4 QPS peak; ReAct: 1.2-2.6 QPS peak | LLaMA-3.1-8B, single A100 40GB | Most comparable to our setup |
| Continuum | Poisson | 0.13 jobs/s (×~10 turns = ~1.3 LLM calls/s) | LLaMA-3.1-8B, A100/B200 | Multi-turn agent programs |
| SGLang Llama3 blog | Poisson | >10 RPS sustained for 8B | LLaMA-3-8B, single A100-80GB | vLLM degrades ~8 RPS |
| NanoFlow (OSDI'25) | Poisson | 1-16 req/s sweep (70B/8GPU) | LLaMA-2-70B | NanoFlow sustains 1.64× higher than TRT-LLM |
| ServeGen (Alibaba) | Bursty (CV>1) | Not disclosed (3.54B reqs / 4 months) | Qwen 14B-72B | **Poisson is unrealistic** — real arrivals are bursty |
| BurstGPT (Azure) | Trace-driven | Chat: 0.019 RPS; API: 0.21 RPS avg | GPT-3.5/4 | Per-service, not per-GPU. Bursts >10× |
| Autellix | Poisson | Not disclosed | LLaMA-3.1-8B/70B/180B | Each "program" = 6-160 LLM calls |

**Conclusion for benchmarking:**
- **Poisson rate=8 req/s** is a valid stress test (near saturation for 8B on single GPU) but ~3-8× higher than realistic agent workloads
- Realistic ranges: chat 2-6 req/s, agent LLM calls 0.5-3 req/s, agent programs 0.1-0.5 programs/s
- Should also test at rate=2-4 req/s for realistic load levels

### Dataset Input Lengths & Known Inference Output (from output_length.json)

Only 3 datasets have **model inference output** with thinking mode. The rest show static field output (much shorter than real model output).

| Dataset | Median Input | Inference Output Median | Source |
|---------|-------------|------------------------|--------|
| SWE-bench | 295 | **2,174** (n=500) / **3,478** (n=2294 test) | output_length.json |
| CodeActInstruct | 836 | **1,174** (n=7139) | output_length.json |
| SWE-smith | 212 | **2,877** (n=59136) | output_length.json |
| Nebius-SWE-Traj | 9,053 | TBD (static: 1,764) | needs --run-inference |
| Go-Browse | 6,451 | TBD (static: 85) | needs --run-inference |
| NNetNav-live | 11,865 | TBD (static: 153) | needs --run-inference |
| openhands-feedback | 85 | TBD (static: 1,471) | needs --run-inference |
| ... 8 more | ... | TBD | needs --run-inference |

**Key insight:** With thinking mode, model output is 3-8× longer than static field text. E.g. CodeActInstruct static=354 vs inference=1,174. This dramatically shifts the prefill/decode ratio — datasets that look "prefill-heavy" from static analysis may actually be balanced or decode-heavy when the model thinks.

**Preliminary candidates for v3 (pending inference output):**
- **Nebius-SWE-Traj**: huge prefill (9K median), likely substantial inference output. Best candidate.
- **Go-Browse**: huge prefill (6.5K), but static output is tiny (85). Even with 3-8× thinking multiplier, may still be very prefill-heavy.
- **SWE-bench**: moderate prefill (295 median), but inference output is 2-3K — actually decode-heavy. Later turns in agent scaffold accumulate context though.

### Per-Turn Context Growth (static text only — NEEDS REDO with model inference)

| Dataset | Med Turns | Med Input/Turn | Med Output/Turn | Decode % | Growth tok/turn | Ctx@T5 | Ctx@T10 | Ctx@T20 | Turn >4096 |
|---------|----------|---------------|----------------|---------|----------------|--------|---------|---------|------------|
| AgentInstruct | 6 | 48 | 39 | 45% | 145 | 808 | N/A | N/A | never |
| CodeActInstruct | 5 | 64 | 61 | 49% | 291 | 1,455 | N/A | N/A | never |
| **Nebius-SWE** | **25** | **530** | 106 | **17%** | **607** | 2,201 | 6,193 | **15,302** | **turn 8** |
| openhands-feedback | 7 | 47 | 18 | 28% | 344 | 972 | 4,940 | 4,427 | turn 15 |

**Key findings:**
- **Nebius-SWE is the best candidate**: ~607 tok/turn growth, hits 4K context by turn 8, 15K by turn 20. Heavy prefill bias (17% decode). Large per-turn input from SWE-agent tool outputs (file contents, command results).
- **AgentInstruct/CodeActInstruct too short**: never exceed 4K context, median 5-6 turns. Not useful for long-context overlap testing.
- **openhands-feedback is bursty**: mostly tiny turns with occasional large tool outputs (400-7K tokens). Less predictable for benchmarking.
- **Context growth is linear** in all datasets — prefill cost grows linearly with turn number.

**Note:** Output/turn above is static text from dataset, NOT model inference output. With thinking mode, model output would be 3-8× longer, increasing decode fraction significantly. Nebius-SWE with thinking could shift from 17% to ~40-50% decode.

### Workload Prioritization

Based on ratio sweep + dataset analysis + per-turn context growth:

**Best workload for v3 kernel evaluation: Nebius-SWE-Trajectories**
- Median input 9,053 tokens (well above 4096 threshold where v3 TTFT advantage kicks in)
- Multi-turn (median 25 turns), context grows ~607 tok/turn → 15K by turn 20
- 17% decode fraction from static text, likely ~40-50% with thinking mode
- v3's -24% TTFT improvement at 4096 input would directly benefit this workload's later turns

**Secondary: 6500/7500 random workload**
- Only workload where v3 wins throughput (+1.9%)
- Matches Go-Browse (6.5K input) and Nebius-SWE later turns (9K+ context)

**Not recommended:**
- Short-input datasets (AgentInstruct, CodeActInstruct, SWE-bench) — median input <1K, below v3 benefit threshold
- Extremely prefill-heavy (Go-Browse, NNetNav) — tiny decode means overlap has nothing to overlap with

**Remaining work:**
- [ ] Run inference output lengths on Nebius-SWE to confirm decode fraction with thinking mode
- [ ] Benchmark Nebius-SWE traces via bench_serving_3rdparty if supported, or use equivalent random workload (--random-input 9000 --random-output TBD)
