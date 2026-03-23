#!/usr/bin/env python3
"""
Count input and output token length distributions for all agent workload datasets.
Prints a summary table of statistics across all datasets.

Two modes:
  1. Static (default): tokenize the dataset's own input/output text fields.
  2. Inference (--run-inference): run model inference with thinking mode via sgl.Engine
     to measure actual model output lengths. Results cached in output_length.json.

Usage:
  # Static field lengths only
  python3 workloads/count_workload_lengths.py

  # Run model inference for actual output lengths (with caching)
  python3 workloads/count_workload_lengths.py --run-inference --num-samples 50

  # Run inference on specific datasets
  python3 workloads/count_workload_lengths.py --run-inference --datasets SWE-bench Go-Browse
"""

import argparse
import json
import os
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_length.json")


# ============================================================
# Utility functions
# ============================================================

def compute_statistics(lengths):
    """Compute summary statistics for a list of lengths."""
    if not lengths:
        return {}
    a = np.array(lengths)
    return {
        "count": len(a),
        "mean": round(float(np.mean(a)), 2),
        "median": round(float(np.median(a)), 2),
        "min": int(np.min(a)),
        "max": int(np.max(a)),
        "std": round(float(np.std(a)), 2),
        "p25": round(float(np.percentile(a, 25)), 2),
        "p75": round(float(np.percentile(a, 75)), 2),
        "p90": round(float(np.percentile(a, 90)), 2),
        "p95": round(float(np.percentile(a, 95)), 2),
        "p99": round(float(np.percentile(a, 99)), 2),
    }


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def tokenize_lengths(texts, tokenizer):
    """Return list of token lengths for a list of strings."""
    lengths = []
    for t in texts:
        if not t:
            lengths.append(0)
        else:
            lengths.append(len(tokenizer.encode(t, add_special_tokens=False)))
    return lengths


def print_statistics(lengths, label):
    if not lengths:
        print(f"\n{label}: No data found")
        return
    s = compute_statistics(lengths)
    print(f"\n{label}:")
    print(f"  Count:  {s['count']}")
    print(f"  Mean:   {s['mean']:.2f}")
    print(f"  Median: {s['median']:.2f}")
    print(f"  Min:    {s['min']}")
    print(f"  Max:    {s['max']}")
    print(f"  Std:    {s['std']:.2f}")
    print(f"  P25:    {s['p25']:.2f}")
    print(f"  P75:    {s['p75']:.2f}")
    print(f"  P90:    {s['p90']:.2f}")
    print(f"  P95:    {s['p95']:.2f}")
    print(f"  P99:    {s['p99']:.2f}")


# ============================================================
# Cache helpers for model inference output lengths
# ============================================================

def lookup_cache(cache, model, dataset, split, num_samples):
    """Look up cached results. Returns entry only if num_samples matches exactly."""
    entry = cache.get(model, {}).get(dataset, {}).get(split)
    if entry is None:
        return None
    if entry["num_samples"] == num_samples:
        return entry
    return None


def store_cache(cache, model, dataset, split, lengths, num_samples, max_new_tokens):
    """Store results in the cache dict (call save_cache to persist)."""
    cache.setdefault(model, {}).setdefault(dataset, {})[split] = {
        "num_samples": num_samples,
        "max_new_tokens": max_new_tokens,
        "lengths": lengths,
        "statistics": compute_statistics(lengths),
    }
    return cache


# ============================================================
# Model inference
# ============================================================

def run_model_inference(input_texts, tokenizer, engine, max_new_tokens=16384,
                        max_context=40960):
    """Run model inference with thinking mode and return output token lengths.

    Trims inputs that exceed (max_context - max_new_tokens - 1) tokens.
    """
    max_input_tokens = max_context - max_new_tokens - 1
    prompts = []
    for i, text in enumerate(input_texts):
        messages = [{"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(input_ids) > max_input_tokens:
            print(f"  Trimming example {i}: {len(input_ids)} -> {max_input_tokens} tokens")
            input_ids = input_ids[:max_input_tokens]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=False)
        prompts.append(prompt)

    sampling_params = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "skip_special_tokens": False,
    }

    print(f"  Running inference on {len(prompts)} examples...")
    outputs = engine.generate(prompts, sampling_params)

    lengths = []
    for out in outputs:
        n_tokens = len(tokenizer.encode(out["text"], add_special_tokens=False))
        lengths.append(n_tokens)
    return lengths


def process_dataset_inference(ds_config, tokenizer, engine, cache,
                              num_samples=None, max_new_tokens=16384):
    """Run model inference on a dataset and return model output lengths (with caching)."""
    import sglang as sgl

    name = ds_config["name"]
    hf_id = ds_config["hf_id"]
    extract = ds_config["extract"]

    print(f"\n{'=' * 60}")
    print(f"  {name} — Model Inference  ({hf_id})")
    print(f"{'=' * 60}")

    # Collect input texts
    all_input_texts = []
    for split_name in ds_config["splits"]:
        try:
            print(f"  Loading split '{split_name}'...")
            ds = load_dataset(hf_id, split=split_name, trust_remote_code=True)
            print(f"    {len(ds)} examples")
            for ex in ds:
                inp, _ = extract(ex)
                all_input_texts.append(inp)
        except Exception as e:
            print(f"    ERROR loading split '{split_name}': {e}")

    if not all_input_texts:
        print(f"  No data loaded for {name}")
        return None

    n = num_samples if num_samples and num_samples < len(all_input_texts) else len(all_input_texts)
    subset_texts = all_input_texts[:n]

    # Check cache
    cache_key_split = "all"
    cached = lookup_cache(cache, tokenizer.name_or_path, hf_id, cache_key_split, n)
    if cached is not None:
        print(f"  Cache hit! ({cached['num_samples']} samples)")
        model_lengths = cached["lengths"]
    else:
        print(f"  Cache miss. Running inference on {n} samples...")
        model_lengths = run_model_inference(
            subset_texts, tokenizer, engine, max_new_tokens=max_new_tokens
        )
        store_cache(cache, tokenizer.name_or_path, hf_id, cache_key_split,
                    model_lengths, n, max_new_tokens)
        save_cache(cache)
        print(f"  Cache saved.")

    model_stats = compute_statistics(model_lengths)
    print_statistics(model_lengths, f"  {name} — Model Output Lengths (n={n})")

    return {
        "name": name,
        "hf_id": hf_id,
        "num_samples": n,
        "model_output": model_stats,
    }


# ============================================================
# Extractors — each returns (input_text, output_text) per example
# ============================================================

def extract_field_pair(input_field, output_field):
    """Simple extractor for datasets with direct input/output fields."""
    def fn(ex):
        return (ex.get(input_field, "") or "", ex.get(output_field, "") or "")
    return fn


def extract_messages(ex, conv_field="messages", input_roles=("user", "system"),
                     output_roles=("assistant",), role_key="role", content_key="content"):
    """Extract concatenated input/output from a messages/conversations list."""
    msgs = ex.get(conv_field, []) or []
    inp, out = [], []
    for msg in msgs:
        role = msg.get(role_key, "")
        text = msg.get(content_key, "") or ""
        if role in input_roles:
            inp.append(text)
        elif role in output_roles:
            out.append(text)
    return "\n".join(inp), "\n".join(out)


def extract_agentinstruct(ex):
    """THUDM/AgentInstruct: conversations with from=human/gpt, content in value."""
    convs = ex.get("conversations", []) or []
    inp, out = [], []
    for msg in convs:
        role = msg.get("from", "")
        text = msg.get("value", "") or ""
        if role == "human":
            inp.append(text)
        elif role in ("gpt", "assistant"):
            out.append(text)
    return "\n".join(inp), "\n".join(out)


def extract_code_feedback(ex):
    return extract_messages(ex, conv_field="messages")


def extract_codeact(ex):
    """xingyaoww/code-act: conversations with role=user/assistant."""
    return extract_messages(ex, conv_field="conversations")


def extract_go_browse(ex):
    """apurvaga/go-browse-wa: step_data dict with prompt/completion as message lists."""
    step_data = ex.get("step_data", {}) or {}
    # prompt and completion are lists of {role, content} dicts
    prompt_msgs = step_data.get("prompt", []) or []
    completion_msgs = step_data.get("completion", []) or []

    inp = "\n".join(m.get("content", "") or "" for m in prompt_msgs if isinstance(m, dict))
    out = "\n".join(m.get("content", "") or "" for m in completion_msgs if isinstance(m, dict))
    return inp, out


def extract_mind2web(ex):
    """osunlp/Mind2Web: confirmed_task → action_reprs (list of strings)."""
    inp = ex.get("confirmed_task", "") or ""
    action_reprs = ex.get("action_reprs", []) or []
    out = "\n".join(action_reprs) if isinstance(action_reprs, list) else str(action_reprs)
    return inp, out


def extract_nebius(ex):
    """nebius/SWE-agent-trajectories: trajectory with role ai/user, text field."""
    traj = ex.get("trajectory", []) or []
    inp, out = [], []
    for msg in traj:
        role = msg.get("role", "")
        text = msg.get("text", "") or msg.get("content", "") or ""
        if role == "user":
            inp.append(text)
        elif role == "ai":
            out.append(text)
    return "\n".join(inp), "\n".join(out)


def extract_nnetnav(ex):
    """stanfordnlp/nnetnav-*: messages list with role user/assistant/system."""
    return extract_messages(ex, conv_field="messages")


def extract_openhands(ex):
    """all-hands/openhands-feedback: trajectory with source=user/agent/environment."""
    traj = ex.get("trajectory", []) or []
    inp, out = [], []
    for entry in traj:
        source = entry.get("source", "")
        text = entry.get("message", "") or entry.get("content", "") or ""
        if source in ("user", "environment"):
            inp.append(text)
        elif source == "agent":
            out.append(text)
    return "\n".join(inp), "\n".join(out)


def extract_orca(ex):
    """microsoft/orca-agentinstruct-1M-v1: messages is a JSON string."""
    msgs_raw = ex.get("messages", "[]") or "[]"
    if isinstance(msgs_raw, str):
        try:
            msgs = json.loads(msgs_raw)
        except json.JSONDecodeError:
            return "", ""
    else:
        msgs = msgs_raw
    inp, out = [], []
    for msg in msgs:
        role = msg.get("role", "")
        text = msg.get("content", "") or ""
        if role in ("user", "system"):
            inp.append(text)
        elif role == "assistant":
            out.append(text)
    return "\n".join(inp), "\n".join(out)


def extract_swe_smith_traj(ex):
    """SWE-bench/SWE-smith-trajectories: messages is a JSON string."""
    msgs_raw = ex.get("messages", "[]") or "[]"
    if isinstance(msgs_raw, str):
        try:
            msgs = json.loads(msgs_raw)
        except json.JSONDecodeError:
            return "", ""
    else:
        msgs = msgs_raw
    inp, out = [], []
    for msg in msgs:
        role = msg.get("role", "")
        text = msg.get("content", "") or ""
        if role in ("user", "system"):
            inp.append(text)
        elif role == "assistant":
            out.append(text)
    return "\n".join(inp), "\n".join(out)


def extract_synatra(ex):
    return (ex.get("prompt", "") or "", ex.get("response", "") or "")


# ============================================================
# Dataset registry
# ============================================================

DATASETS = [
    {
        "name": "SWE-bench",
        "hf_id": "princeton-nlp/SWE-bench",
        "splits": ["train", "test"],
        "extract": extract_field_pair("problem_statement", "patch"),
    },
    {
        "name": "AgentInstruct",
        "hf_id": "THUDM/AgentInstruct",
        "splits": ["os", "db", "alfworld", "webshop", "kg", "mind2web"],
        "extract": extract_agentinstruct,
    },
    {
        "name": "Code-Feedback",
        "hf_id": "m-a-p/Code-Feedback",
        "splits": ["train"],
        "extract": extract_code_feedback,
    },
    {
        "name": "CodeActInstruct",
        "hf_id": "xingyaoww/code-act",
        "splits": ["codeact"],
        "extract": extract_codeact,
    },
    {
        "name": "Go-Browse",
        "hf_id": "apurvaga/go-browse-wa",
        "splits": ["train"],
        "extract": extract_go_browse,
    },
    {
        "name": "Mind2Web",
        "hf_id": "osunlp/Mind2Web",
        "splits": ["train"],
        "extract": extract_mind2web,
    },
    {
        "name": "Nebius-SWE-Trajectories",
        "hf_id": "nebius/SWE-agent-trajectories",
        "splits": ["train"],
        "extract": extract_nebius,
    },
    {
        "name": "NNetNav-live",
        "hf_id": "stanfordnlp/nnetnav-live",
        "splits": ["train", "test"],
        "extract": extract_nnetnav,
    },
    {
        "name": "NNetNav-wa",
        "hf_id": "stanfordnlp/nnetnav-wa",
        "splits": ["train", "test"],
        "extract": extract_nnetnav,
    },
    {
        "name": "openhands-feedback",
        "hf_id": "all-hands/openhands-feedback",
        "splits": ["train"],
        "extract": extract_openhands,
    },
    {
        "name": "Orca-AgentInstruct",
        "hf_id": "microsoft/orca-agentinstruct-1M-v1",
        "splits": [
            "creative_content", "text_modification", "struct2text_flow", "rc",
            "rag", "text_extraction", "mcq", "follow_up", "analytical_reasoning",
            "fermi", "fs_cot_flow", "code_", "brain_teaser", "text_classification",
            "open_domain_qa",
        ],
        "extract": extract_orca,
    },
    {
        "name": "SWE-Gym",
        "hf_id": "SWE-Gym/SWE-Gym",
        "splits": ["train"],
        "extract": extract_field_pair("problem_statement", "patch"),
    },
    {
        "name": "SWE-smith",
        "hf_id": "SWE-bench/SWE-smith",
        "splits": ["train"],
        "extract": extract_field_pair("problem_statement", "patch"),
    },
    {
        "name": "Synatra",
        "hf_id": "oottyy/Synatra",
        "splits": ["train"],
        "extract": extract_synatra,
    },
]


# ============================================================
# Main
# ============================================================

def process_dataset(ds_config, tokenizer, num_samples=None):
    """Load a dataset, extract input texts, tokenize, return input length stats.

    Output lengths are NOT computed here — they must come from model inference
    with thinking mode (--run-inference) to be meaningful.
    """
    name = ds_config["name"]
    hf_id = ds_config["hf_id"]
    extract = ds_config["extract"]

    print(f"\n{'=' * 60}")
    print(f"  {name}  ({hf_id})")
    print(f"{'=' * 60}")

    all_input_texts = []

    for split_name in ds_config["splits"]:
        try:
            print(f"  Loading split '{split_name}'...")
            ds = load_dataset(hf_id, split=split_name, trust_remote_code=True)
            print(f"    {len(ds)} examples")

            for ex in ds:
                inp, _ = extract(ex)
                all_input_texts.append(inp)
        except Exception as e:
            print(f"    ERROR loading split '{split_name}': {e}")

    if not all_input_texts:
        print(f"  No data loaded for {name}")
        return None

    if num_samples and len(all_input_texts) > num_samples:
        all_input_texts = all_input_texts[:num_samples]

    print(f"  Total examples: {len(all_input_texts)}")
    print(f"  Tokenizing inputs...")
    input_lengths = tokenize_lengths(all_input_texts, tokenizer)

    input_stats = compute_statistics(input_lengths)

    print_statistics(input_lengths, f"  {name} — Input Lengths")

    return {
        "name": name,
        "hf_id": hf_id,
        "total_examples": len(all_input_texts),
        "input": input_stats,
    }


def print_summary_table(results):
    """Print a formatted summary table of input lengths only."""
    print("\n")
    print("=" * 80)
    print("SUMMARY TABLE: Input Token Length Statistics")
    print("(Output lengths require --run-inference with thinking mode)")
    print("=" * 80)

    header = (
        f"{'Dataset':<28} {'Count':>8} │ "
        f"{'In Mean':>8} {'In Med':>8} {'In P95':>8} {'In Max':>8}"
    )
    print(header)
    print("─" * 80)

    for r in results:
        if r is None:
            continue
        i = r["input"]
        row = (
            f"{r['name']:<28} {r['total_examples']:>8} │ "
            f"{i.get('mean', 0):>8.0f} {i.get('median', 0):>8.0f} "
            f"{i.get('p95', 0):>8.0f} {i.get('max', 0):>8}"
        )
        print(row)

    print("─" * 80)
    print()


def print_inference_summary_table(results):
    """Print a formatted summary table of model inference output lengths."""
    print("\n")
    print("=" * 100)
    print("SUMMARY TABLE: Model Inference Output Token Lengths")
    print("=" * 100)

    header = (
        f"{'Dataset':<28} {'N':>8} │ "
        f"{'Mean':>8} {'Median':>8} {'P90':>8} {'P95':>8} {'P99':>8} {'Max':>8}"
    )
    print(header)
    print("─" * 100)

    for r in results:
        if r is None:
            continue
        o = r["model_output"]
        row = (
            f"{r['name']:<28} {r['num_samples']:>8} │ "
            f"{o.get('mean', 0):>8.0f} {o.get('median', 0):>8.0f} "
            f"{o.get('p90', 0):>8.0f} {o.get('p95', 0):>8.0f} "
            f"{o.get('p99', 0):>8.0f} {o.get('max', 0):>8}"
        )
        print(row)

    print("─" * 100)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Count input/output token lengths for all agent workload datasets.")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-8B",
                        help="Tokenizer model (default: Qwen/Qwen3-8B)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Cap examples per dataset (default: all)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Only run these datasets (by name). Default: all.")
    parser.add_argument("--run-inference", action="store_true",
                        help="Run model inference to measure actual output lengths")
    parser.add_argument("--max-new-tokens", type=int, default=16384,
                        help="Max new tokens for model inference (default: 16384)")
    parser.add_argument("--mem-fraction-static", type=float, default=0.85,
                        help="GPU memory fraction for sgl.Engine (default: 0.85)")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    selected = [ds for ds in DATASETS
                if not args.datasets or ds["name"] in args.datasets]

    if args.run_inference:
        import sglang as sgl

        print(f"\nStarting sgl.Engine with {args.model_path}...")
        engine = sgl.Engine(
            model_path=args.model_path,
            reasoning_parser="qwen3",
            mem_fraction_static=args.mem_fraction_static,
        )

        cache = load_cache()
        inference_results = []
        for ds_config in selected:
            try:
                r = process_dataset_inference(
                    ds_config, tokenizer, engine, cache,
                    num_samples=args.num_samples,
                    max_new_tokens=args.max_new_tokens,
                )
                inference_results.append(r)
            except Exception as e:
                print(f"\n  FAILED: {ds_config['name']}: {e}")
                inference_results.append(None)

        engine.shutdown()
        print_inference_summary_table(inference_results)
    else:
        results = []
        for ds_config in selected:
            try:
                r = process_dataset(ds_config, tokenizer, num_samples=args.num_samples)
                results.append(r)
            except Exception as e:
                print(f"\n  FAILED: {ds_config['name']}: {e}")
                results.append(None)

        # Save results
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "length_stats.json")
        valid_results = [r for r in results if r is not None]
        with open(output_path, "w") as f:
            json.dump(valid_results, f, indent=2)
        print(f"\nResults saved to {output_path}")

        print_summary_table(results)


if __name__ == "__main__":
    main()
