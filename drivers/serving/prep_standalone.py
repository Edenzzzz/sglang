#!/usr/bin/env python3
"""Prepare the S3 standalone workloads on the pod.

Math and coding sets become ShareGPT-format JSON ({"conversations": [{"from": "human", "value":
prompt}, {"from": "gpt", "value": reference}]}), so `--dataset-name sharegpt` replays them with
each request's output length taken from its reference solution. Mooncake (tool-agent and
conversation traces) and LongBench v2 are fetched into the caches their own loaders read.

Usage: prep_standalone.py <out_dir>
"""
import json, os, sys

out = sys.argv[1]
os.makedirs(out, exist_ok=True)


def dump(name, pairs):
    rows = [{"conversations": [{"from": "human", "value": p}, {"from": "gpt", "value": r}]}
            for p, r in pairs if p and r]
    path = os.path.join(out, f"{name}.json")
    json.dump(rows, open(path, "w"))
    print(f"{name}: {len(rows)} rows -> {path}", flush=True)


def try_set(name, fn):
    try:
        dump(name, fn())
    except Exception as e:  # report and continue; a missing set is named, not silently skipped
        print(f"{name}: FAILED {type(e).__name__}: {e}", flush=True)


from datasets import load_dataset

try_set("gsm8k", lambda: [(r["question"], r["answer"])
                           for r in load_dataset("openai/gsm8k", "main", split="test")])
try_set("math500", lambda: [(r["problem"], r["solution"])
                             for r in load_dataset("HuggingFaceH4/MATH-500", split="test")])
try_set("humaneval", lambda: [(r["prompt"], r["canonical_solution"])
                               for r in load_dataset("openai/openai_humaneval", split="test")])


def code_feedback():
    pairs = []
    for r in load_dataset("m-a-p/Code-Feedback", split="train", streaming=True):
        msgs = r.get("messages") or []
        if len(msgs) >= 2 and msgs[0].get("role") == "user" and msgs[1].get("role") == "assistant":
            pairs.append((msgs[0]["content"], msgs[1]["content"]))
        if len(pairs) >= 5000:
            break
    return pairs


try_set("codefeedback", code_feedback)

# Mooncake traces: the loader reads /tmp/<workload>_trace.jsonl and downloads it if missing.
try:
    from sglang.benchmark.datasets.common import MOONCAKE_DATASET_URL
    from sglang.benchmark.utils import download_and_cache_file
    for wl in ("toolagent", "conversation"):
        p = download_and_cache_file(MOONCAKE_DATASET_URL[wl], f"/tmp/{wl}_trace.jsonl")
        print(f"mooncake {wl}: {sum(1 for _ in open(p))} rows -> {p}", flush=True)
except Exception as e:
    print(f"mooncake: FAILED {type(e).__name__}: {e}", flush=True)

try:
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    print(f"longbench_v2: {len(ds)} rows cached", flush=True)
except Exception as e:
    print(f"longbench_v2: FAILED {type(e).__name__}: {e}", flush=True)
print("PREP_STANDALONE_DONE", flush=True)
