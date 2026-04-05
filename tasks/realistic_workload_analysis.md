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
- [x] Swept 6 workloads × 3 configs (Default-mixed-chunk, Persistent v3, POD). Results in table below (includes P99).
- [x] Hypothesis partially confirmed: v3 benefits from **large absolute prefill** (>4096 tokens), not balanced ratios. v3 wins throughput only on 6500/7500, wins TTFT on 4096/1024 (-24%).
- [x] v3 has consistent 0.5-4% throughput regression vs default — needs investigation (see `debug_kernel_server_perf.md` TODO)
- [ ] **Must redo** with P95/P99 TTFT and ITL columns. Current results only have median. Also add prefill-prioritized config to show P99 ITL tail latency disadvantage. Redo again after kernel optimizations in `debug_kernel_server_perf.md` if performance changes.

### 2. Realistic arrival rates
- [x] Research done — see Arrival Rate Research table below. Poisson rate=8 is stress test; realistic agent rate is 0.5-3 req/s.

### 3. First-turn output length stats (with reasoning)
- [x] Run `count_workload_lengths.py --run-inference --num-samples 200` on all trace datasets (200 samples each)
- [x] Results in `output_length.json`: SWE-bench, CodeActInstruct, SWE-smith, AgentInstruct, Mind2Web, openhands-feedback, SWE-Gym, Code-Feedback, Nebius-SWE-Trajectories
- [x] Skipped (no traces / single-step): Go-Browse, NNetNav-live, NNetNav-wa, Synatra, Orca-AgentInstruct, ShareGPT

### 4. Per-turn context growth
- [x] Ran Qwen3-8B with thinking on Nebius-SWE traces (3 traces, 8 turns each, via online server HTTP)
- [x] Results in `workloads/per_turn_growth.json`
- [ ] Run on remaining trace datasets (AgentInstruct, CodeActInstruct, openhands-feedback) if needed

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

**Key findings (OUTDATED — from before decode overhead fix and mixed CG. See `debug_kernel_server_perf.md` for current results with v3+LPT+CG):**
1. **v3 wins throughput only on longest workload** (6500/7500: +1.9%). On all others, v3 is 0.5-4% behind default. *(NOTE: the 0.5-4% regression was caused by BatchAttention decode overhead, now fixed.)*
2. **v3 wins TTFT on prefill-heavy workloads**: 4096/1024 is the sweet spot (156ms vs 206ms, -24%).
3. **POD consistently worse TTFT than both default and v3**, especially on prefill-heavy workloads.
4. **ITL is nearly identical across all configs** (~±0.3ms). Decode performance is the same.
5. ~~**v3 throughput regression is consistent** (-0.5% to -4%)~~ — **FIXED**: decode overhead eliminated by using regular decode wrapper. v3 now matches default-mc within ±0.1%.
6. **Input/output ratio doesn't clearly predict v3 benefit**: v3 benefits from large absolute prefill tokens, not the ratio itself.

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

### Dataset Input Lengths & Inference Output with Thinking (from output_length.json)

All trace datasets now have inference output measured with Qwen3-8B thinking mode (200 samples each unless noted).

| Dataset | N | Median Input | Inference Output Median | Output P95 | In:Out Ratio |
|---------|--:|-------------|------------------------|-----------|-------------|
| **Nebius-SWE-Traj** | 200 | 9,053 | **3,726** | 16,384 | **2.4:1** |
| Code-Feedback | 200 | 200 | **5,598** | 13,313 | 0.04:1 |
| SWE-Gym | 200 | 412 | **2,344** | 5,828 | 0.18:1 |
| openhands-feedback | 200 | 85 | **1,690** | 7,536 | 0.05:1 |
| Mind2Web | 200 | 17 | **1,289** | 2,387 | 0.01:1 |
| AgentInstruct | 200 | 656 | **1,090** | 3,113 | 0.60:1 |
| SWE-bench | 500 | 295 | **2,174** | 6,184 | 0.14:1 |
| CodeActInstruct | 7,139 | 836 | **1,174** | 16,384 | 0.71:1 |
| SWE-smith | 59,136 | 212 | **2,877** | 7,336 | 0.07:1 |

**Key findings:**
1. **Nebius-SWE is the only prefill-heavy dataset** (2.4:1 ratio). All others are decode-heavy with thinking mode.
2. **Model output with thinking is 3-16× longer** than static field text. Code-Feedback: static median=887 vs inference=5,598 (6.3×). Nebius-SWE: static=1,764 vs inference=3,726 (2.1×).
3. **Many samples hit the 16K max output cap** — Nebius-SWE P90=16,383, Code-Feedback P95=13,313. Real output would be even longer without the cap.
4. **Nebius-SWE confirmed as best v3 candidate**: 9K input + 3.7K output (2.4:1 ratio), closest to the 4096/1024 workload where v3 had best TTFT advantage (-24%). Later turns accumulate more context, pushing input even higher.

### Per-Turn Context Growth (Nebius-SWE, Qwen3-8B with thinking, 3 traces × 8 turns)

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

### SLO Standards from Literature

Survey of TTFT and TBT/ITL SLO targets used in recent LLM serving papers:

| Paper | TTFT SLO | TBT/ITL SLO | How SLOs are set | Notes |
|-------|----------|-------------|-----------------|-------|
| **Tempo** (JITServe, arxiv 2504.20068) | ~2s | ~100ms | P95 from DeepSeek API calls | TTLT=20s for throughput requests. 3:1:1 latency:throughput:collective mix |
| **Scorpio** (arxiv 2505.23022) | 0.5-7.5s (6 categories) | 30-50ms (Llama-8B), 60-100ms (Gemma-27B) | Per-application type (programming assistant = tight, chatbot = loose) | EDF reordering (Least Deadline First). TPOT Guard admits only if estimated TPOT ≤ min SLO |
| **Ascendra** (arxiv 2504.20828) | 1.0s (7B/8B), 1.5s (14B) on ShareGPT; 2.5s (7B/8B), 3.0s (14B) on LongBench | 150ms (all models) | Model-size dependent. 150ms TBT ≈ 400 words/min (2-3× human reading speed) | EDF priority. Offloads urgent requests to high-priority instances |
| **SOLA** (MLSys'25) | Varies | Varies | Balances TTFT vs TPOT distribution | State-aware scheduling. 45.5%→99.4% SLO attainment |
| **DistServe** (OSDI'24) | Varies | Varies | Goodput = requests meeting both TTFT+TPOT SLOs | Disaggregated prefill-decode |
| **SLOs-Serve** (arxiv 2504.08784) | Varies | Varies | Multi-SLO optimization | 2.2× capacity vs Sarathi-Serve/vLLM |

**SLO targets for our benchmarks** (Qwen3-8B, H100):

| Workload | TTFT SLO | ITL SLO | Rationale |
|----------|----------|---------|-----------|
| ShareGPT (chat) | 2.0s | 100ms | Tempo standard (interactive chat) |
| Code-Feedback (code gen) | 3.0s | 150ms | Longer inputs, Ascendra's 150ms reading-speed target |
| Nebius-SWE (agent) | 5.0s | 150ms | Long inputs (9K median), agent can tolerate higher TTFT |

For input-length-dependent TTFT scaling: `TTFT_SLO = base + input_tokens / 5000` (i.e., +1s per 5K extra input tokens beyond baseline). This follows the pattern in Ascendra (ShareGPT 1s base → LongBench 2.5s for longer inputs).

**Updated with model inference (Nebius-SWE only):**

| Turn | Median Input | Median Output (thinking) | In:Out Ratio |
|------|-------------|-------------------------|-------------|
| T1 | 1,124 | 3,376 | 0.3:1 (decode-heavy, first turn) |
| T2 | 1,505 | 1,418 | 1.1:1 (balanced) |
| T3 | 2,802 | 2,070 | 1.4:1 |
| T4 | 3,674 | 1,520 | **2.4:1** (enters v3 sweet spot) |
| T5 | 4,364 | 1,681 | **2.6:1** (matches 4096/1024 workload) |
| T6 | 4,936 | 410 | 12.0:1 (tool-call-only turn) |
| T7 | 5,544 | 2,712 | 2.0:1 |
| T8 | 6,167 | 259 | 23.8:1 (tool-call-only turn) |

**Key findings:**
- Context grows ~700 tokens/turn (1.1K → 6.2K over 8 turns)
- By turn 4-5, input exceeds 4K — the v3 TTFT sweet spot (-24% vs default in ratio sweep)
- Output varies wildly (132-3,424): tool-call-only turns produce very short responses, reasoning turns produce 1.5K-3.4K
- Median across all 24 measurements: input=3,074, output=1,707 (1.8:1 ratio)
- The 2-3:1 ratio at turns 4-7 closely matches the 4096/1024 and 2048/2048 ratio sweep workloads

### Workload Scan Results (B200, 4 configs × 10 workloads, 200 prompts, rate=8, to be rerun on H100)

Full CSV: `/tmp/workload_scan_results.csv`. Script: `tasks/realistic_workload_analysis/run_workload_scan.sh`.

**Throughput (tok/s) — v3-fixed and POD vs Default-mixed-chunk:**

| Input | Output | Prefill-prio | Default | v3-fixed | POD | v3 vs Def | POD vs Def |
|------:|-------:|-------------:|--------:|---------:|----:|----------:|-----------:|
| 512 | 512 | 3,883 | 3,880 | 3,879 | 3,873 | -0.0% | -0.2% |
| 1024 | 1024 | 7,033 | 7,068 | 7,075 | 7,045 | +0.1% | -0.3% |
| 2048 | 1024 | 10,792 | 10,821 | 10,868 | 10,818 | +0.4% | -0.0% |
| **4096** | **1024** | **17,132** | 16,858 | **17,141** | 17,096 | **+1.7%** | +1.4% |
| 2048 | 2048 | 10,812 | 10,852 | 10,831 | 10,782 | -0.2% | -0.6% |
| 4096 | 4096 | 11,675 | 11,588 | 11,646 | 11,596 | +0.5% | +0.1% |
| **6500** | **4000** | **12,552** | 12,150 | **12,360** | 12,349 | **+1.7%** | +1.6% |
| **9000** | **4000** | **12,779** | 12,372 | **12,577** | 12,604 | **+1.7%** | **+1.9%** |
| 6500 | 7500 | 8,455 | 8,344 | 8,375 | 8,385 | +0.4% | +0.5% |
| 1024 | 4096 | 8,752 | 8,778 | 8,737 | 8,700 | -0.5% | -0.9% |

**TTFT median (ms):**

| Input | Output | Prefill-prio | Default | v3-fixed | POD |
|------:|-------:|-------------:|--------:|---------:|----:|
| 4096 | 1024 | **46** | 68 | **63** | 63 |
| 6500 | 4000 | **51** | 108 | **95** | 110 |
| 9000 | 4000 | **107** | **214** | **196** | 210 |
| 6500 | 7500 | **74** | 129 | **114** | 145 |

**P99 ITL (ms):**

| Input | Output | Prefill-prio | Default | v3-fixed | POD |
|------:|-------:|-------------:|--------:|---------:|----:|
| 4096 | 1024 | **37** | 43 | **37** | 52 |
| 6500 | 4000 | **54** | 88 | **65** | 89 |
| 9000 | 4000 | **113** | 153 | **127** | 156 |
| 6500 | 7500 | **44** | 62 | **57** | 70 |

### Workload Prioritization

**Key findings from workload scan:**
1. **v3-fixed wins throughput on large-input workloads** (4096+): +1.7% at 4096/1024, 6500/4000, 9000/4000. No regression on small inputs.
2. **v3-fixed has best TTFT among mixed-chunk configs**: 8-11% better than default on 6500/4000 and 9000/4000 (agent-realistic workloads).
3. **v3-fixed has best P99 ITL among mixed-chunk configs**: matches prefill-prio at 4096/1024 (37ms), 27% better than default at 6500/4000 (65 vs 88ms).
4. **POD matches v3 on throughput** but has worse TTFT and P99 ITL on all workloads.
5. **Prefill-prioritized still wins all metrics** on all workloads due to CUDA graph advantage. Gap is 1.5-3.3% throughput.

**Best workload for v3/POD evaluation: 9000/4000 (Nebius-SWE realistic)**
- v3-fixed: +1.7% throughput, -8% TTFT, -17% P99 ITL vs default
- POD: +1.9% throughput, similar TTFT/ITL to default
- Matches real Nebius-SWE late-turn workload (input 9K, output 4K with thinking)

**Runner-up: 6500/4000 (Nebius-SWE mid-turn)**
- v3-fixed: +1.7% throughput, -11% TTFT, -27% P99 ITL vs default

