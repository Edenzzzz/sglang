# Agent Workload Characteristics: Turns, Tokens, and Serving Implications

A survey of how many turns/steps each agent workload type takes, what the per-turn token distributions look like, and how LLM serving systems model these workloads. Sources: original benchmark papers + recent systems papers (SOSP'25, OSDI'25, MLSys'25, EuroSys'25, ATC'24, ICML'24-25, NeurIPS'24, ICLR'25, and recent arXiv).

---

## 0. Dataset Trace Summary

Which of our collected datasets have full multi-turn traces (conversation trajectories), and how many turns per example?

| Dataset | Category | Has Traces? | Trace Format | Turns Mean | Turns Med | Turns P95 | Notes |
|---------|----------|-------------|-------------|-----------|----------|----------|-------|
| AgentInstruct (Orca) | Mixed agent | **Yes** | JSON messages (system/user/assistant) | 7.8 | 5 | 22 | Full multi-turn conversations; roles include tool outputs |
| Code-Feedback | Code | **Yes** | Conversation list (user/assistant) | 3.3 | 3 | 7 | Short code Q&A dialogues |
| CodeActInstruct | Code+Tools | **Yes** | Conversation list (user/assistant) | 3.7 | 4 | 6 | Includes tool-use actions in assistant turns |
| Mind2Web | Web browsing | **Yes** | Action sequence list | 7.7 | 6 | 17 | Each action = one web interaction step |
| Nebius-SWE-Trajectories | Coding/SWE | **Yes** | Trajectory list (thought/action/observation) | 26.4 | 17 | 70 | Full SWE-agent trajectories with tool outputs |
| openhands-feedback | Coding/SWE | **Yes** | Messages list (system/user/assistant) | 57.5 | 39 | 146 | Full OpenHands agent trajectories; very long |
| SWE-Gym/OpenHands-SFT | Coding/SWE | **Yes** | Messages list (system/user/assistant) | ~20 | — | — | 491 successful trajectories (13-101 msgs) |
| NNetNav-wa | Web browsing | **No** § | Single (state, action) pair | 1 | 1 | 1 | Single-step snapshot from multi-step trajectory |
| Go-Browse | Web browsing | **No** § | Single (state, action) pair | 1 | 1 | 1 | Single-step snapshot from multi-step trajectory |
| SWE-bench | Coding/SWE | **No** ‡ | (problem_statement, patch) | — | — | — | Benchmark only; no trajectory. Turns depend on scaffold (2-181) |
| SWE-Gym | Coding/SWE | **No** ‡ | (problem_statement, patch) | — | — | — | Same schema as SWE-bench; trajectories in companion datasets |
| SWE-smith | Coding/SWE | **No** ‡ | (problem_statement, patch) | — | — | — | Same schema as SWE-bench; 59K problems |
| Synatra-CodeInterpreter | Code | **No** ‡ | (instruction, response) | — | — | — | Single instruction-response pairs |
| ShareGPT | Chat baseline | **Yes** † | Conversation list | 5.75 | — | — | Literature value (CachedAttention ATC'24) |

**Legend:**
- § Single-step snapshots extracted from multi-step trajectories — the full trajectory is not in the dataset
- ‡ Problem+solution pairs only — no agent trajectory; turns depend entirely on the scaffold/agent framework used
- † Value from literature, not directly measured from dataset

**Key takeaway:** Only 7 of our 14 datasets contain full multi-turn traces. For benchmarking agent serving, the trajectory datasets (Nebius, openhands, AgentInstruct, CodeActInstruct) are the most valuable because they provide realistic per-turn input/output distributions. For ‡ datasets (SWE-bench/Gym/smith), we would need to run an actual agent scaffold to generate traces.

---

## 1. Consolidated Turn/Step Counts by Workload Type

### 1.1 Web Browsing Agents

| Benchmark | Avg Steps/Task | Max Steps | Source |
|---|---|---|---|
| Mind2Web | 7-8 actions | ~15 | Deng et al., NeurIPS'23 Spotlight |
| Mind2Web-Live | 8.4 actions | — | Variant |
| WebArena | **not reported** | 30 (limit) | Zhou et al., ICLR'24 |
| VisualWebArena | **not reported** | 30 (limit) | Koh et al., 2024 |
| OSWorld (single-action) | 4.6–13.6 per app | — | OSWorld-Human, NeurIPS'24 |
| OSWorld (grouped) | 3.2–8.8 per app | — | OSWorld-Human |
| TheAgentCompany | **27-28 steps** (best) | — | arXiv:2412.14161 |
| TheAgentCompany (weak) | **~40 steps** | — | Gemini-2.0-Flash |

**Notes:**
- WebArena original paper does **not** report average trajectory length — only max step limit (30) and success rates. This is a gap.
- OSWorld: best agents take 1.4–2.7x more steps than humans. LibreOffice Calc is hardest (13.6 single-action steps).
- TheAgentCompany: "Number of steps = total LLM calls." Best models (Gemini-2.5-Pro, Claude-3.7-Sonnet) average **27-28 calls at ~$4/task**.

### 1.2 Coding / SWE Agents

| Benchmark / Scaffold | Avg Steps/Calls | Max | Source |
|---|---|---|---|
| SWE-agent (success) | **12 median** | 150 (eval limit) | Yang et al., NeurIPS'24 |
| SWE-agent (failure) | **21 mean** | 150 | Yang et al., NeurIPS'24 |
| SWE-bench (Continuum traces) | 10.9 turns | 50 | Continuum, arXiv:2511.02230 |
| SWE-agent + GPT-4o-mini | **181 requests** | — | SWE-Effi, arXiv:2509.09853 |
| OpenHands + Qwen3-32B | 19.5 requests | — | SWE-Effi |
| OpenHands + Llama-70B | 30.8 requests | — | SWE-Effi |
| AutoCodeRover + GPT-4o-mini | 39.5 requests | — | SWE-Effi |
| Agentless + GPT-4o-mini | 82.3 requests | — | SWE-Effi |
| Agentless-mini | **2.0 requests** | — | SWE-Effi |

**Notes:**
- SWE-bench original paper does **not** report step counts — it's scaffold-dependent.
- **90x range:** 2 calls (Agentless-mini) → 181 calls (SWE-agent+GPT-4o-mini) for the same benchmark.
- "Agents succeed quickly, fail slowly": 12 steps (success) vs 21 steps (failure). 53% of SWE-smith failures hit step limits.
- Input tokens are **orders of magnitude** larger than output: SWE-agent+GPT-4o-mini uses 8.1M input vs 24K output tokens per task.
- Token usage grows **quadratically** with turns as conversation history is appended.

### 1.3 Tool-Use Agents

| Benchmark | Avg Turns/Calls | Max | Source |
|---|---|---|---|
| BFCL v3 (Autellix) | 10.75 calls | 70 | Autellix, arXiv:2502.13965 |
| BFCL v4 (Continuum) | 6.3 turns | ~40+ | Continuum, arXiv:2511.02230 |
| AgentBench (range) | **5–35 rounds** | 35 | Liu et al., ICLR'24 |
| MCP-Bench (strong models) | 6-8 rounds, 30-40 calls | — | arXiv:2508.20453 |
| MCP-Bench (weak models) | 17.3 rounds, **155 calls** | — | arXiv:2508.20453 |
| τ-bench retail | — | — | Sierra, 2024 (turns not reported) |
| τ-bench airline | — | — | Sierra, 2024 (turns not reported) |

**AgentBench per-environment breakdown (Liu et al., ICLR'24, Table 2):**

| Environment | Avg Rounds |
|---|---|
| Web Shopping | 5 |
| Database | 5 |
| Operating System | 8 |
| Web Browsing | 10 |
| Knowledge Graph | 15 |
| Lateral Thinking | 25 |
| Digital Card Game | 30 |
| House Holding | **35** |

### 1.4 Reasoning / Search Agents

| Method | Avg LLM Calls | Max | Source |
|---|---|---|---|
| ReAct (HotpotQA) | 7 steps | 7 (limit) | Yao et al., ICLR'23 |
| ReAct (ALFWorld) | 50+ steps (expert) | — | Yao et al., ICLR'23 |
| LATS (Autellix) | **159.7 calls** | Very high | Autellix |
| LATS (Cost of Dynamic Reasoning) | 71.0 calls | — | Kim et al., arXiv:2506.04301 |
| Reflexion | 3-12 trials × multi-step | 12 trials | Shinn et al., NeurIPS'23 |

**Notes:**
- LATS: each node expansion = 3 LLM calls (action + value + reflection). 159.7 is total LLM invocations.
- Autellix vs "Cost of Dynamic Reasoning" disagree on LATS: 159.7 vs 71.0 calls — likely different configs (n, k parameters).

### 1.5 Chat Baselines

| Dataset | Avg Turns | Source |
|---|---|---|
| LMSYS-Chat-1M | 2.0 | Zheng et al., ICLR'24 |
| ShareGPT (CachedAttention) | 5.75 | ATC'24 |
| ShareGPT (Autellix) | 6.66 | Autellix |
| Production (ServeGen) | 3.5 | Alibaba, arXiv:2505.09999 |

---

## 2. Per-Turn Token Distributions

### 2.1 Cross-Workload Comparison (Autellix)

| Workload | Input/Call | Output/Call | Calls/Program | Character |
|---|---|---|---|---|
| ShareGPT | 256 | 277 | 6.66 | Decode-heavy |
| BFCL | **735** | **34** | 10.75 | Prefill-heavy |
| LATS/MCTS | 467 | 73 | 159.7 | Compute-intensive |

### 2.2 Total Tokens Per Program

| Workload | Total Tokens/Program | Source |
|---|---|---|
| BFCL v4 | 93,256 (mean) | Continuum |
| SWE-bench | 70,126 (mean) | Continuum |
| LATS | ~86K (estimated) | Autellix |
| SWE-agent + GPT-4o-mini | **~8.1M** (mostly input) | SWE-Effi |
| OpenHands + Llama-70B | ~897K input | SWE-Effi |

### 2.3 Key Pattern: Input Token Growth

- By turn 3+, historical tokens = **>99%** of prefilling workload (CachedAttention, ATC'24)
- Input grows **3-4x** over a session (Cost of Dynamic Reasoning)
- OpenHands on SWE-bench: **28.04 output tokens per 1,000 input tokens** — extreme input dominance
- Reasoning models: thinking tokens average **4x longer** than answer tokens (ServeGen)

### 2.4 Inter-Turn Delays

| Workload | Tool Exec Time | Distribution | Source |
|---|---|---|---|
| WebShop tools | ~20ms | Short, consistent | Kim et al. |
| SWE-bench shell commands | 925ms mean | Long-tailed (std 3,550ms) | Continuum |
| HotpotQA Wikipedia API | ~1.2s | Moderate | Kim et al. |
| BFCL v4 web fetches | 1,923ms mean | Long-tailed (std 2,133ms) | Continuum |
| Production human think-time | ~100s median | Extremely long tail | ServeGen |

Slowest 10% of `fetch_url` = 52.5% of total delay; slowest 10% of `cd` = 94.1% (Continuum).

---

## 3. Recent LLM Serving Systems (2025–2026) and Agent Workloads

### 3.1 Papers That Model Agent Workloads

| Paper | Venue | Agent Workload Data? | Key Contribution |
|---|---|---|---|
| **Autellix** | arXiv:2502.13965 | ✅ ShareGPT, BFCL, LATS traces | Program-aware scheduling (PLAS/ATLAS). 4-15x throughput vs vLLM. |
| **Continuum** | arXiv:2511.02230 | ✅ SWE-bench, BFCL v4 (100 traces each, GPT-5) | KV cache TTL for multi-turn. 2x JCT reduction. |
| **Pie** | **SOSP'25** | ✅ Inferlets for agentic loops | Programmable serving — apps control KV cache, generation logic. 42 APIs. |
| **Cost of Dynamic Reasoning** | arXiv:2506.04301 | ✅ ReAct, LATS, Reflexion, LLMCompiler | LLM = 69.4% of latency, tools = 30.2%. Energy: 71-131x vs chat. |
| **ServeGen** | arXiv:2505.09999 | ✅ Alibaba production (billions of requests) | Reasoning tokens 4x answer. Multi-turn avg 3.5 turns. |
| **Agent.xpu** | arXiv:2506.24045 | ✅ On-device agentic scheduling | Heterogeneous SoC scheduling for personal LLM agents. |
| **Throughput-Optimal Scheduling** | arXiv:2504.07347 | ✅ Queueing theory for agents | Work-conserving scheduling insufficient for agent networks. |

### 3.2 Papers That Do NOT Model Agent Workloads

| Paper | Venue | What They Use Instead |
|---|---|---|
| NanoFlow | OSDI'25 | Synthetic workloads |
| BlitzScale | OSDI'25 | Model loading optimization |
| WaferLLM | OSDI'25 | Wafer-scale device inference |
| Aegaeon | SOSP'25 | Multi-model pooling, not agent-specific |
| Mercury | SOSP'25 | Multi-GPU operator optimization |
| Jenga | SOSP'25 | Memory management for heterogeneous serving |
| IC-Cache | SOSP'25 | In-context caching |
| PrefillOnly | SOSP'25 | Prefill-only workloads |
| FlashInfer | MLSys'25 | Attention engine optimization |
| SOLA | MLSys'25 | SLO-aware scheduling |
| AI Metropolis | MLSys'25 | Multi-agent simulation (out-of-order exec) |
| HCache | EuroSys'25 | State restoration (relevant to multi-turn) |
| CacheBlend | EuroSys'25 (Best Paper) | RAG cache fusion |
| InferCept | ICML'24 | KV cache pinning for tool calls |
| CachedAttention | ATC'24 | ✅ ShareGPT multi-turn analysis |
| BurstGPT | KDD'25 | Azure traces (no session-level turns) |
| Mooncake | FAST'25 (Best Paper) | Production KV cache pool (no agent stats) |

### 3.3 Agent Benchmark Papers (ML Venues, 2024-2025)

| Paper | Venue | Workload | Steps/Turns Reported? |
|---|---|---|---|
| SWE-agent | NeurIPS'24 | Coding | ✅ 12 median (success), 21 mean (failure) |
| OSWorld | NeurIPS'24 | Desktop computer | ✅ 4.6-13.6 steps per app |
| AgentBoard | NeurIPS'24 | Multi-domain | ✅ Progress Rate metric |
| TheAgentCompany | NeurIPS'25 | Office work | ✅ 27-28 avg steps (best), ~40 (weak) |
| SWE-Gym | ICML'25 | Coding training | Trajectory collection, no avg published |
| SWE-Effi | arXiv:2509.09853 | Coding efficiency | ✅ 2-181 requests per scaffold-model combo |
| τ-bench | Sierra, 2024 | Retail/airline | ❌ Turns not reported (focus on pass^k) |
| τ²-bench | Sierra, 2025 | Dual-control | ❌ |
| MCP-Bench | arXiv:2508.20453 | Tool chaining | ✅ 6-8 rounds (strong), 17.3 (weak) |

---

## 4. Key Observations

### 4.1 Most Original Papers Don't Report Turn Counts
WebArena, VisualWebArena, WorkArena, SWE-bench, τ-bench all focus on **success rate**, not trajectory efficiency. Turn/step statistics come from follow-up analysis papers or systems papers that collect traces.

### 4.2 Terminology is Inconsistent
- **LLM calls** = all model invocations (Autellix, SWE-Effi)
- **Turns** = user-visible interaction rounds (Continuum)
- **Steps** = agent actions (SWE-agent, WebArena)
- **Tool calls** = external tool invocations (typically n-1 for n LLM calls)
- **Trials** = complete task attempts (Reflexion)

Mind2Web: each "step" = 2 model calls (ranking + action). LATS: each node = 3 calls (action + value + reflection). These multipliers are critical.

### 4.3 Agent Outputs Are Short, Inputs Dominate
Per-turn agent outputs are **34-277 tokens** (Autellix). Inputs grow monotonically due to conversation history. Single-turn inference with thinking mode (generating thousands of tokens) does **not** reflect real agent patterns.

### 4.4 Turn Count is Scaffold-Dependent
Same SWE-bench task: 2 calls (Agentless-mini) → 181 calls (SWE-agent+GPT-4o-mini). **90x difference.** Any characterization must specify the agent framework.

### 4.5 Weak Models Use More Turns
MCP-Bench: strong models use 6-8 rounds; weak models use 17.3 rounds with 155 tool calls. TheAgentCompany: best models ~27 steps, Gemini-2.0-Flash ~40 steps. Resource consumption inversely correlates with capability.

### 4.6 Serving Systems Are Behind
Most OSDI'24-25, SOSP'25, MLSys'25 papers still evaluate on synthetic or single-turn workloads. Only **Autellix**, **Continuum**, and **Pie** seriously address agent serving. There is a gap between agent reality (multi-turn, tool-calling, growing context) and how serving systems are evaluated.

---

## 5. References

### Systems Papers
1. Autellix (arXiv:2502.13965) — Program-aware LLM agent serving
2. Continuum (arXiv:2511.02230) — Multi-turn KV cache TTL scheduling
3. Pie (SOSP'25) — Programmable serving with inferlets
4. Aegaeon (SOSP'25) — GPU pooling for concurrent LLM serving
5. ServeGen (arXiv:2505.09999) — Alibaba production workload characterization
6. Cost of Dynamic Reasoning (arXiv:2506.04301) — Agent infrastructure analysis
7. Agent.xpu (arXiv:2506.24045) — On-device agentic scheduling
8. Throughput-Optimal Scheduling (arXiv:2504.07347) — Queueing theory for agents
9. NanoFlow (OSDI'25) — Optimal LLM serving throughput
10. BlitzScale (OSDI'25) — Fast model autoscaling
11. InferCept (ICML'24) — KV cache pinning for tool calls
12. CachedAttention (ATC'24) — Multi-turn KV cache reuse
13. Mooncake (FAST'25 Best Paper) — KVCache-centric disaggregated architecture
14. Parrot (OSDI'24) — Semantic variable LLM serving
15. FlashInfer (MLSys'25) — Customizable attention engine
16. AI Metropolis (MLSys'25) — Multi-agent simulation with OoO execution
17. HCache (EuroSys'25) — Fast state restoration
18. CacheBlend (EuroSys'25 Best Paper) — RAG cache fusion
19. BurstGPT (KDD'25) — Azure OpenAI traces
20. Tokenomics (arXiv:2601.14470) — Token consumption in agent SDLC

### Agent Benchmark / Analysis Papers
21. SWE-bench (ICLR'24) — Real GitHub issues
22. SWE-agent (NeurIPS'24) — Agent-computer interfaces for SWE
23. SWE-Effi (arXiv:2509.09853) — Resource-aware SWE agent evaluation
24. SWE-smith (arXiv:2504.21798) — Scaled SWE agent data
25. SWE-Gym (ICML'25) — SWE agent training
26. WebArena (ICLR'24) — Realistic web agent benchmark
27. OSWorld (NeurIPS'24) — Desktop computer agent benchmark
28. OSWorld-Human (arXiv:2506.16042) — Agent efficiency analysis
29. TheAgentCompany (NeurIPS'25) — Real-world office tasks
30. AgentBench (ICLR'24) — 8-environment agent evaluation
31. Mind2Web (NeurIPS'23) — Web agent benchmark
32. BFCL (ICML'25) — Berkeley function calling leaderboard
33. MCP-Bench (arXiv:2508.20453) — Tool-chaining benchmark
34. τ-bench (Sierra, 2024) — Customer service agent benchmark
35. ReAct (ICLR'23) — Reasoning + acting
36. LATS (ICML'24) — Language agent tree search
37. Reflexion (NeurIPS'23) — Verbal reinforcement learning
38. How Do Coding Agents Spend Your Money (OpenReview, 2025)
39. AgentBoard (NeurIPS'24) — Multi-turn agent evaluation
40. LMSYS-Chat-1M (ICLR'24) — Chat conversation dataset
