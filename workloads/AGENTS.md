# Agent Workload Length Statistics

I need scripts that count the input/output length distribution for LLM inference workloads. The datasets are in `workloads/`. Many datasets don't have output provided, so I need to run the model through them myself.

## Setup

1. Activate the venv: `source .venv/bin/activate`
2. Find an unused GPU: `nvidia-smi`

## Running Inference

There are two ways to run inference:

### 1. Online Server (with thinking/reasoning mode)

Launch the server:
```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --attention-backend flashinfer \
  --watchdog-timeout 999999 \
  --mem-fraction-static 0.85 \
  --enable-mixed-chunk \
  --enable-hierarchical-cache \
  --enable-piecewise-cuda-graph \
  --reasoning-parser qwen3
```

Send requests using `bench_serving.py`. Send all data and do not trim lengths:
```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --random-input 1024 \
  --random-output 6000 \
  --request-rate 8 \
  --num-prompt 400
```

### 2. Offline `sgl.Engine` (with reasoning mode)

See `docs/basic_usage/offline_engine_api.ipynb` and related examples for details.

## Counting Length Statistics

Count the output length statistics as in `workloads/count_swebench_lengths.py`.

For now, test on **SWE-bench** first. Later, compute input/output lengths for all datasets in this [paper](workloads/AGENT%20DATA%20PROTOCOL-%20UNIFYING%20DATASETS%20FORDIVERSE,%20EFFECTIVE%20FINE-TUNING%20OF%20LLM%20AGENT.pdf).

### Output Length Cache (`output_length.json`)

Model output lengths are cached in `workloads/output_length.json` to avoid re-running expensive inference. The file is created automatically on first run.

**Structure:** `cache[model][dataset][split]` — each entry stores:
- `num_samples`: number of samples used
- `max_new_tokens`: generation limit
- `lengths`: raw list of output token lengths
- `statistics`: precomputed stats (count, mean, median, min, max, std, p25/p75/p90/p95/p99)

**Cache behavior:**
- Cache requires an exact `num_samples` match to hit.
- Otherwise it's a miss and inference runs, then results are saved to the cache.

## TODO (overnight task)

1. **Fix cache**: require exact `num_samples` match (not >=). ✅ Done.
2. **Run SWE-bench on all samples**: ✅ Done. 2294 test samples cached in `output_length.json`.
3. **Research all target datasets**: ✅ Done. All 14 datasets researched on HuggingFace.
4. **Consolidate into one script**: ✅ Done. `workloads/count_workload_lengths.py` handles all 14 datasets.
5. **Run on all datasets overnight**: ✅ Done. Results in `workloads/length_stats.json`.
6. **Final output**: ✅ Done. Summary table presented.
7. **bench_serving_3rdparty.py**: ✅ Done. `python/sglang/bench_serving_3rdparty.py` supports Code-Feedback, CodeActInstruct, SWE-Gym, SWE-smith, Orca AgentInstruct. `bench_serving.py` redirects via `--dataset-name`.

## Target Datasets

| Dataset | Cat | Count | Source | In Mean | In Med | In P95 | Out Mean | Out Med | Out P95 | Note |
|---------|-----|------:|--------|--------:|-------:|-------:|---------:|--------:|--------:|------|
| ShareGPT | — | 92,886 | manual | 319 | 136 | 945 | TBD | TBD | TBD | General chat conversations |
| SWE-bench | S | 21,302 | manual | 584 | 295 | 1,747 | TBD | TBD | TBD | Real GitHub issue → patch |
| AgentInstruct (Zeng et al., 2023) | C T W | 1,866 | synthetic | 785 | 656 | 1,570 | TBD | TBD | TBD | Mixture of Browsing, Database, OS, etc. |
| Code-Feedback (Zheng et al., 2024a) | C | 66,383 | manual | 274 | 200 | 688 | TBD | TBD | TBD | Code generation with runtime feedback loops |
| CodeActInstruct (Wang et al., 2024b) | C | 7,139 | synthetic | 828 | 836 | 1,455 | TBD | TBD | TBD | Code generation and tool use with execution |
| Go-Browse (Gandhi & Neubig, 2025) | W | 164,533 | rollout | 8,500 | 6,451 | 17,520 | TBD | TBD | TBD | Structured exploration web rollouts |
| Mind2Web (Deng et al., 2023) | W | 1,009 | manual | 20 | 17 | 44 | TBD | TBD | TBD | Human web demos on real websites |
| Nebius SWE Trajectories (Golubev et al., 2024) | S | 80,036 | rollout | 12,393 | 9,053 | 26,411 | TBD | TBD | TBD | SWE-agent trajectories from Nebius relying solely on open-weight models |
| NNetNav-live (Murty et al., 2024) | W | 53,709 | rollout | 10,493 | 11,865 | 18,859 | TBD | TBD | TBD | Retroactively labeled live web exploration |
| NNetNav-wa (Murty et al., 2024) | W | 50,310 | rollout | 6,169 | 4,902 | 16,281 | TBD | TBD | TBD | Retroactively labeled WebArena exploration |
| openhands-feedback (All Hands AI, 2024) | C T W | 275 | rollout | 670 | 85 | 492 | TBD | TBD | TBD | Recorded OpenHands agent trajectories with human feedback |
| Orca AgentInstruct (Mitra et al., 2024) | T | 1,046,410 | synthetic | 467 | 202 | 1,595 | TBD | TBD | TBD | Large-scale synthetic tool-use instructions data |
| SWE-Gym (Pan et al., 2025) | S | 2,438 | rollout | 686 | 412 | 2,050 | TBD | TBD | TBD | Agent trajectories solving real GitHub repo tasks |
| SWE-smith (Yang et al., 2025b) | S | 59,136 | manual | 188 | 212 | 392 | TBD | TBD | TBD | Trajectories of agents on synthesized bug-fix tasks |
| Synatra (Ou et al., 2024) | W | 99,924 | rollout | 1,594 | 1,321 | 3,638 | TBD | TBD | TBD | Synthetically created web demos of tutorials |

**Categories:** C = Code, T = Tool-use, W = Web, S = SWE
**Token counts:** Qwen3-8B tokenizer. Out columns = model inference output with thinking (TBD = pending). Full results in `workloads/length_stats.json`.
