#!/usr/bin/env python3
"""Mode census and CUDA-graph count over every captured batch.

The graph question is not "how many kernels" but "how many distinct captures".
sglang captures per (padded token count, padded batch size) bucket, so what
matters is how many buckets exist and whether a bucket's execution shape is
fixed -- a bucket whose mode flips per batch cannot be captured at all.
"""
import json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from four_mode_dispatch import classify, route, plan_split

BS_LADDER = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 256]
TOK_LADDER = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def bucket(v, lad):
    for x in lad:
        if v <= x:
            return x
    return lad[-1]


SETS = [("sets/qwen_small_h800.jsonl", (28, 4, 128)),
        ("sets/qwen_small_h800_largeprefill.jsonl", (28, 4, 128)),
        ("sets/qwen_large_h200_tp8.jsonl", (8, 1, 128)),
        ("sets/qwen_large_gb200_tp16.jsonl", (4, 1, 128))]


def main():
    tot = Counter(); allb = {}
    for st, h in SETS:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), st)
        modes = Counter(); buckets = {}
        for line in open(p):
            if not line.strip():
                continue
            cls = classify(json.loads(line), h[0] // h[1])
            rows = [r for c in cls.values() for r in c]
            if not rows:
                continue
            fa3, pers = route(cls, h[0] // h[1])
            mode, cf, cp = plan_split(fa3, pers, h[0], h[1])
            modes[mode] += 1; tot[mode] += 1
            key = (bucket(sum(q for q, _ in rows), TOK_LADDER),
                   bucket(len(rows), BS_LADDER))
            buckets.setdefault(key, Counter())[mode] += 1
        stable = sum(1 for v in buckets.values() if len(v) == 1)
        print(f"{os.path.basename(st):<40} modes {dict(modes)}")
        print(f"{'':<40} buckets {len(buckets)}, single-mode {stable} "
              f"({100*stable/max(len(buckets),1):.0f}%)")
        allb.update({(st, k): v for k, v in buckets.items()})
    print(f"\nall sets: modes {dict(tot)}")
    n = len(allb); stable = sum(1 for v in allb.values() if len(v) == 1)
    conc = sum(1 for v in allb.values() if "conc" in v)
    print(f"  distinct shape buckets: {n}")
    print(f"  buckets with a single mode (capturable as one graph): {stable} "
          f"({100*stable/n:.0f}%)")
    print(f"  buckets needing a cross-stream event (conc present): {conc}")
    print(f"\n  graphs, ours          = {n} (one per bucket; mode is fixed at plan time)")
    print(f"  graphs, sequential    = {n} (both launches in one capture at fixed shapes)")
    print(f"  graphs, sequential if captured independently = "
          f"prefill buckets x decode buckets")
    print(f"  graphs, dual-stream   = {n}, every one containing a cross-stream event")


if __name__ == "__main__":
    main()
