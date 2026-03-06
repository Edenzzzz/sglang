#!/usr/bin/env python3
"""Count the number of turns (assistant responses) per example in each dataset."""

import json
import numpy as np
from datasets import load_dataset


def compute_stats(vals):
    a = np.array(vals)
    return {
        "count": len(a), "mean": round(float(np.mean(a)), 1),
        "median": round(float(np.median(a)), 1),
        "min": int(np.min(a)), "max": int(np.max(a)),
        "p25": round(float(np.percentile(a, 25)), 1),
        "p75": round(float(np.percentile(a, 75)), 1),
        "p90": round(float(np.percentile(a, 90)), 1),
        "p95": round(float(np.percentile(a, 95)), 1),
    }


def count_messages(ex, conv_field="messages", role_key="role"):
    """Count total messages and assistant messages in a conversation."""
    msgs = ex.get(conv_field, []) or []
    total = len(msgs)
    assistant = sum(1 for m in msgs if m.get(role_key, "") in ("assistant", "gpt", "ai"))
    return total, assistant


def count_json_messages(ex, conv_field="messages", role_key="role"):
    """Same but for JSON-string messages."""
    msgs_raw = ex.get(conv_field, "[]") or "[]"
    if isinstance(msgs_raw, str):
        try:
            msgs = json.loads(msgs_raw)
        except json.JSONDecodeError:
            return 0, 0
    else:
        msgs = msgs_raw
    total = len(msgs)
    assistant = sum(1 for m in msgs if m.get(role_key, "") in ("assistant", "gpt", "ai"))
    return total, assistant


DATASETS = [
    {
        "name": "ShareGPT",
        "hf_id": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "splits": ["train"],
        "count_fn": lambda ex: (
            len(ex.get("conversations", [])),
            sum(1 for m in (ex.get("conversations", []) or [])
                if m.get("from", "") in ("gpt", "assistant"))
        ),
    },
    {
        "name": "AgentInstruct",
        "hf_id": "THUDM/AgentInstruct",
        "splits": ["os", "db", "alfworld", "webshop", "kg", "mind2web"],
        "count_fn": lambda ex: (
            len(ex.get("conversations", [])),
            sum(1 for m in (ex.get("conversations", []) or [])
                if m.get("from", "") in ("gpt", "assistant"))
        ),
    },
    {
        "name": "Code-Feedback",
        "hf_id": "m-a-p/Code-Feedback",
        "splits": ["train"],
        "count_fn": lambda ex: count_messages(ex, "messages"),
    },
    {
        "name": "CodeActInstruct",
        "hf_id": "xingyaoww/code-act",
        "splits": ["codeact"],
        "count_fn": lambda ex: count_messages(ex, "conversations"),
    },
    {
        "name": "Go-Browse",
        "hf_id": "apurvaga/go-browse-wa",
        "splits": ["train"],
        "count_fn": lambda ex: (
            len((ex.get("step_data", {}) or {}).get("prompt", []) or []) +
            len((ex.get("step_data", {}) or {}).get("completion", []) or []),
            len((ex.get("step_data", {}) or {}).get("completion", []) or [])
        ),
    },
    {
        "name": "Mind2Web",
        "hf_id": "osunlp/Mind2Web",
        "splits": ["train"],
        "count_fn": lambda ex: (
            len(ex.get("action_reprs", []) or []),
            len(ex.get("action_reprs", []) or [])  # each action is one "turn"
        ),
    },
    {
        "name": "Nebius-SWE-Trajectories",
        "hf_id": "nebius/SWE-agent-trajectories",
        "splits": ["train"],
        "count_fn": lambda ex: (
            len(ex.get("trajectory", []) or []),
            sum(1 for m in (ex.get("trajectory", []) or [])
                if m.get("role", "") == "ai")
        ),
    },
    {
        "name": "NNetNav-live",
        "hf_id": "stanfordnlp/nnetnav-live",
        "splits": ["train", "test"],
        "count_fn": lambda ex: count_messages(ex, "messages"),
    },
    {
        "name": "NNetNav-wa",
        "hf_id": "stanfordnlp/nnetnav-wa",
        "splits": ["train", "test"],
        "count_fn": lambda ex: count_messages(ex, "messages"),
    },
    {
        "name": "openhands-feedback",
        "hf_id": "all-hands/openhands-feedback",
        "splits": ["train"],
        "count_fn": lambda ex: (
            len(ex.get("trajectory", []) or []),
            sum(1 for m in (ex.get("trajectory", []) or [])
                if m.get("source", "") == "agent")
        ),
    },
    {
        "name": "Orca-AgentInstruct",
        "hf_id": "microsoft/orca-agentinstruct-1M-v1",
        "splits": ["creative_content", "text_modification", "struct2text_flow", "rc",
                    "rag", "text_extraction", "mcq", "follow_up", "analytical_reasoning",
                    "fermi", "fs_cot_flow", "code_", "brain_teaser", "text_classification",
                    "open_domain_qa"],
        "count_fn": lambda ex: count_json_messages(ex, "messages"),
    },
    {
        "name": "SWE-Gym",
        "hf_id": "SWE-Gym/SWE-Gym",
        "splits": ["train"],
        "count_fn": lambda ex: (1, 1),  # single field pair, no conversation
    },
    {
        "name": "SWE-smith",
        "hf_id": "SWE-bench/SWE-smith",
        "splits": ["train"],
        "count_fn": lambda ex: (1, 1),  # single field pair
    },
    {
        "name": "Synatra",
        "hf_id": "oottyy/Synatra",
        "splits": ["train"],
        "count_fn": lambda ex: (1, 1),  # single prompt/response
    },
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Only process these datasets (by name)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Max samples per dataset")
    args = parser.parse_args()

    results = {}

    for ds_config in DATASETS:
        name = ds_config["name"]
        if args.datasets and name not in args.datasets:
            continue

        print(f"\n{'='*60}")
        print(f"  {name}  ({ds_config['hf_id']})")
        print(f"{'='*60}")

        total_msgs_list = []
        assistant_turns_list = []

        for split_name in ds_config["splits"]:
            try:
                print(f"  Loading split '{split_name}'...")
                ds = load_dataset(ds_config["hf_id"], split=split_name)
                print(f"    {len(ds)} examples")

                for ex in ds:
                    total_msgs, assistant_turns = ds_config["count_fn"](ex)
                    total_msgs_list.append(total_msgs)
                    assistant_turns_list.append(assistant_turns)

                    if args.num_samples and len(total_msgs_list) >= args.num_samples:
                        break
            except Exception as e:
                print(f"    ERROR: {e}")

            if args.num_samples and len(total_msgs_list) >= args.num_samples:
                break

        if not total_msgs_list:
            print(f"  No data for {name}")
            continue

        total_stats = compute_stats(total_msgs_list)
        asst_stats = compute_stats(assistant_turns_list)

        print(f"\n  Total messages per example:")
        print(f"    n={total_stats['count']}  mean={total_stats['mean']}  "
              f"median={total_stats['median']}  min={total_stats['min']}  "
              f"max={total_stats['max']}  p90={total_stats['p90']}  p95={total_stats['p95']}")

        print(f"  Assistant turns per example:")
        print(f"    n={asst_stats['count']}  mean={asst_stats['mean']}  "
              f"median={asst_stats['median']}  min={asst_stats['min']}  "
              f"max={asst_stats['max']}  p90={asst_stats['p90']}  p95={asst_stats['p95']}")

        results[name] = {
            "total_messages": total_stats,
            "assistant_turns": asst_stats,
        }

    # Summary table
    print(f"\n\n{'='*100}")
    print(f"SUMMARY: Assistant Turns Per Example")
    print(f"{'='*100}")
    print(f"{'Dataset':<30} {'N':>8} │ {'Mean':>6} {'Med':>6} {'Min':>5} {'Max':>6} {'P90':>6} {'P95':>6}")
    print(f"{'─'*100}")
    for name, r in results.items():
        s = r["assistant_turns"]
        print(f"{name:<30} {s['count']:>8} │ {s['mean']:>6.1f} {s['median']:>6.1f} "
              f"{s['min']:>5} {s['max']:>6} {s['p90']:>6.1f} {s['p95']:>6.1f}")

    # Save results
    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "turn_stats.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
