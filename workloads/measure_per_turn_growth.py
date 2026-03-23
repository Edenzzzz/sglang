#!/usr/bin/env python3
"""
Measure per-turn context growth for multi-turn trace datasets.

For each trace, replays the conversation turn by turn:
- At each turn, builds cumulative context (all prior messages)
- Sends to Qwen3-8B with thinking mode to measure actual output length
- Records: turn number, cumulative input tokens, model output tokens

Usage:
  python3 workloads/measure_per_turn_growth.py --num-traces 5 --max-turns 10
"""

import argparse
import json
import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def extract_turns_nebius(example):
    """Extract per-turn messages from Nebius-SWE-Trajectories.
    Returns list of (role, text) tuples."""
    traj = example.get("trajectory", []) or []
    turns = []
    for msg in traj:
        role = msg.get("role", "")
        text = msg.get("text", "") or msg.get("content", "") or ""
        if role == "ai":
            role = "assistant"
        if role in ("user", "assistant") and text.strip():
            turns.append((role, text))
    return turns


def extract_turns_openhands(example):
    """Extract per-turn messages from openhands-feedback."""
    traj = example.get("trajectory", []) or []
    turns = []
    for entry in traj:
        source = entry.get("source", "")
        text = entry.get("message", "") or entry.get("content", "") or ""
        if source in ("user", "environment"):
            role = "user"
        elif source == "agent":
            role = "assistant"
        else:
            continue
        if text.strip():
            turns.append((role, text))
    return turns


def extract_turns_agentinstruct(example):
    """Extract per-turn messages from AgentInstruct."""
    convs = example.get("conversations", []) or []
    turns = []
    for msg in convs:
        role_raw = msg.get("from", "")
        text = msg.get("value", "") or ""
        if role_raw == "human":
            role = "user"
        elif role_raw in ("gpt", "assistant"):
            role = "assistant"
        else:
            continue
        if text.strip():
            turns.append((role, text))
    return turns


def extract_turns_codeact(example):
    """Extract per-turn messages from CodeActInstruct."""
    convs = example.get("conversations", []) or []
    turns = []
    for msg in convs:
        role = msg.get("role", "")
        text = msg.get("content", "") or ""
        if role in ("user", "assistant") and text.strip():
            turns.append((role, text))
    return turns


TRACE_DATASETS = {
    "Nebius-SWE-Trajectories": {
        "hf_id": "nebius/SWE-agent-trajectories",
        "split": "train",
        "extract_turns": extract_turns_nebius,
    },
    "openhands-feedback": {
        "hf_id": "all-hands/openhands-feedback",
        "split": "train",
        "extract_turns": extract_turns_openhands,
    },
    "AgentInstruct": {
        "hf_id": "THUDM/AgentInstruct",
        "split": "os",
        "extract_turns": extract_turns_agentinstruct,
    },
    "CodeActInstruct": {
        "hf_id": "xingyaoww/code-act",
        "split": "codeact",
        "extract_turns": extract_turns_codeact,
    },
}


def build_cumulative_prompt(turns_so_far, tokenizer):
    """Build a chat prompt from accumulated turns."""
    messages = []
    for role, text in turns_so_far:
        if role == "user":
            messages.append({"role": "user", "content": text})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": text})
    # Add generation prompt
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    return prompt


def measure_trace(turns, tokenizer, server_url, max_turns=20, max_context=40960):
    """Measure per-turn context growth for a single trace.

    Returns list of dicts: {turn, cum_input_tokens, output_tokens}
    """
    import requests
    max_input_tokens = max_context - 16384 - 1  # reserve for output
    results = []
    assistant_turn_count = 0

    for i, (role, text) in enumerate(turns):
        if role != "user":
            continue

        # Build cumulative context up to this user message
        context_turns = turns[:i+1]
        prompt = build_cumulative_prompt(context_turns, tokenizer)
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        cum_input_tokens = len(input_ids)

        # Trim if too long
        if cum_input_tokens > max_input_tokens:
            input_ids = input_ids[:max_input_tokens]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=False)
            cum_input_tokens = max_input_tokens

        # Run inference via HTTP
        resp = requests.post(f"{server_url}/generate", json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": 16384,
                "temperature": 0.6,
                "top_p": 0.95,
                "skip_special_tokens": False,
            },
        }, timeout=600)
        resp.raise_for_status()
        output_text = resp.json()["text"]
        output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))

        assistant_turn_count += 1
        results.append({
            "turn": assistant_turn_count,
            "cum_input_tokens": cum_input_tokens,
            "output_tokens": output_tokens,
        })

        if assistant_turn_count >= max_turns:
            break

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--num-traces", type=int, default=5,
                        help="Number of traces to measure per dataset")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Max turns to measure per trace")
    parser.add_argument("--min-turns", type=int, default=5,
                        help="Min turns required in a trace to be selected")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Datasets to measure (default: all)")
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Use online server via HTTP instead of sgl.Engine to avoid OOM
    # (sgl.Engine loads model weights in CPU RAM, which combined with dataset
    # loading exceeds the 44 GiB container memory limit)
    import requests
    server_url = f"http://localhost:{args.port}"
    print(f"Using online server at {server_url}...")
    # Verify server is up
    try:
        resp = requests.get(f"{server_url}/v1/models", timeout=5)
        resp.raise_for_status()
        print(f"  Server ready: {resp.json()['data'][0]['id']}")
    except Exception as e:
        print(f"  ERROR: Server not reachable at {server_url}: {e}")
        print(f"  Start the server first:")
        print(f"    python3 -m sglang.launch_server --model-path {args.model_path} "
              f"--reasoning-parser qwen3 --mem-fraction-static {args.mem_fraction_static} "
              f"--watchdog-timeout 999999 --port {args.port}")
        return
    engine = None  # not used; inference via HTTP

    selected = {k: v for k, v in TRACE_DATASETS.items()
                if not args.datasets or k in args.datasets}

    all_results = {}

    for ds_name, ds_config in selected.items():
        print(f"\n{'='*60}")
        print(f"  {ds_name}")
        print(f"{'='*60}")

        try:
            # Use streaming to avoid loading full dataset into memory (OOM on large datasets)
            ds = load_dataset(ds_config["hf_id"], split=ds_config["split"],
                              streaming=True)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        # Find traces with enough turns and at least 1 tool-like interaction
        extract = ds_config["extract_turns"]
        good_traces = []
        for idx, example in enumerate(ds):
            turns = extract(example)
            # Count assistant turns
            n_assistant = sum(1 for r, _ in turns if r == "assistant")
            if n_assistant >= args.min_turns:
                good_traces.append((idx, turns))
            if len(good_traces) >= args.num_traces:
                break
            if idx >= 10000:  # don't scan too many
                break
        print(f"  Found {len(good_traces)} traces with >= {args.min_turns} turns")

        dataset_results = []
        for trace_idx, (ds_idx, turns) in enumerate(good_traces):
            print(f"  Trace {trace_idx+1}/{len(good_traces)} (ds_idx={ds_idx}, "
                  f"{sum(1 for r,_ in turns if r=='assistant')} assistant turns)...")
            trace_results = measure_trace(
                turns, tokenizer, server_url, max_turns=args.max_turns)
            dataset_results.append(trace_results)

            # Print per-turn summary
            for r in trace_results:
                print(f"    Turn {r['turn']}: input={r['cum_input_tokens']}, "
                      f"output={r['output_tokens']}")

        all_results[ds_name] = dataset_results

        # Aggregate stats
        if dataset_results:
            all_turns = [r for trace in dataset_results for r in trace]
            if all_turns:
                print(f"\n  Aggregate ({len(all_turns)} turn measurements):")
                print(f"    Median input: {np.median([r['cum_input_tokens'] for r in all_turns]):.0f}")
                print(f"    Median output: {np.median([r['output_tokens'] for r in all_turns]):.0f}")
                # Per-turn-number stats
                max_turn = max(r['turn'] for r in all_turns)
                print(f"    Per-turn medians:")
                for t in range(1, min(max_turn+1, args.max_turns+1)):
                    turn_data = [r for r in all_turns if r['turn'] == t]
                    if turn_data:
                        med_in = np.median([r['cum_input_tokens'] for r in turn_data])
                        med_out = np.median([r['output_tokens'] for r in turn_data])
                        print(f"      T{t}: input={med_in:.0f}, output={med_out:.0f}, "
                              f"ratio={med_in/med_out:.1f}:1")

    # Server stays running (managed externally)

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "per_turn_growth.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
