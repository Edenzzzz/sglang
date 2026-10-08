#!/usr/bin/env python3
"""Emit the three 4-class workload sets for mega-mixed-tile kernel optimization.

Four tile/scheduling classes, per the design being targeted:

  LARGE_PREFILL  long-prefix chunk            packed_qo = chunk * r   -> tile 128
  SPEC_VERIFY    (k+1) candidates per row      packed_qo = (k+1) * r  -> tile 64 when 16<(k+1)r<=64
  MIXED_DECODE   decode rows + small chunks    packed_qo = r          -> tile 16 (chunk -> 128)
  SPEC_DRAFT     one drafted token per row     packed_qo = r          -> tile 16, smallest work

Row shapes come from real captures spanning the prefix axis; decode-row counts
are scaled to each hardware config's resident-request capacity (hw_capacity.py).
"""
import argparse, glob, json, os, random
import numpy as np

CAPTURES = {  # label -> (glob, what it contributes)
    "fresh":   ("tasks/realistic_workload_analysis/h800_capture_loadsweep_*/composition_LPT.jsonl",
                "co-tenancy, chat-dominated rows (prefix p50 2)"),
    "reuse":   ("tasks/realistic_workload_analysis/h800_capture_prefixreuse/composition_LPT.jsonl",
                "multi-turn on 8K shared prefixes (prefix p50 8.6K)"),
    "sessions":("tasks/realistic_workload_analysis/h800_capture_codexonly/composition_LPT.jsonl",
                "pure session replay (prefix p50 32.7K, 90% of rows >512)"),
}

# name: (gqa_ratio, max resident reqs @32K ctx, k, note)
#
# k is chosen per model so the verify class lands in its own tile: the verify
# tile is distinct only when 16 < (k+1)*r <= 64. At r=16 that means k+1 in
# (1,4], so the "2 spec steps" design point (k=1, packed 32 -> tile 64) works.
# At r=7 it means k+1 in (2.3, 9.1]: k=1 gives packed 14 -> tile 16, which
# COLLAPSES verify into the draft class, so the small model needs k=3
# (packed 28 -> tile 64) to exercise all four classes.
TARGETS = {
    "qwen_small_h800":      (7,   28, 3, "Qwen2.5-7B bf16, H800 x1, 921K tok pool"),
    "qwen_large_h200_tp8":  (16, 115, 1, "Qwen3-235B-A22B fp8, H200 4x8 (TP8), 90.5 GB KV/GPU"),
    "qwen_large_gb200_tp16":(16, 364, 1, "Qwen3-235B-A22B fp8, GB200 NVL72 (TP16), 143 GB KV/GPU"),
}


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        for l in open(f):
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r.get("d") and r.get("q"):
                rows.append(r)
    return rows


def tile(p): return 128 if p > 64 else (64 if p > 16 else 16)


def build(rows, r, max_running, k, rnd, n_out, large_prefix_min=8192):
    """Turn captured mixed batches into 4-class fused batches at the target scale."""
    out = []
    for _ in range(n_out):
        src = rnd.choice(rows)
        kvs = [int(x) for x in src["d"]]
        # scale the decode population to this hardware's capacity
        if max_running > len(kvs):
            kvs = kvs + [kvs[rnd.randrange(len(kvs))] for _ in range(max_running - len(kvs))]
        else:
            kvs = kvs[:max_running]
        # SparseSpec unified split: 1/(k+1) verifying, rest drafting
        nv = max(1, round(len(kvs) / (k + 1)))
        rnd.shuffle(kvs)
        verify = [(k + 1, kv) for kv in kvs[:nv]]
        draft = [(1, kv) for kv in kvs[nv:]]
        large, small = [], []
        for q, p in zip(src["q"], src["p"]):
            (large if int(p) >= large_prefix_min else small).append((int(q), int(p)))
        out.append(dict(large_prefill=large, spec_verify=verify,
                        mixed_decode=small, spec_draft=draft))
    return out


def summarize(batches, r, name):
    cls = {}
    for b in batches:
        for c, rows in b.items():
            for q, kv in rows:
                cls.setdefault(c, []).append((q * r, kv))
    print(f"  {'class':<15} {'rows/batch':>11} {'packed_qo p50':>14} {'tile':>6} {'KV p50':>9}")
    for c in ("large_prefill", "spec_verify", "mixed_decode", "spec_draft"):
        v = cls.get(c, [])
        if not v:
            print(f"  {c:<15} {'0':>11}")
            continue
        pq = np.array([x[0] for x in v]); kv = np.array([x[1] for x in v])
        per = len(v) / len(batches)
        print(f"  {c:<15} {per:>11.1f} {int(np.median(pq)):>14} {tile(int(np.median(pq))):>6} "
              f"{int(np.median(kv)):>9}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="batches per set")
    ap.add_argument("--outdir", default="tasks/fused_specdec_workload/sets")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rnd = random.Random(0)

    pool = {}
    for lab, (pat, desc) in CAPTURES.items():
        pool[lab] = load(pat)
        print(f"{lab:<9} {len(pool[lab]):>5} captured batches  ({desc})")
    allrows = [x for v in pool.values() for x in v]
    print()

    for name, (r, maxrun, k, note) in TARGETS.items():
        # small model: keep its own captures. large models: same row shapes,
        # scaled batch; prefix distribution taken from the reuse+sessions mix
        # because a large-model deployment serves cache-friendly traffic.
        src = allrows if name.startswith("qwen_small") else pool["reuse"] + pool["sessions"]
        b = build(src, r, maxrun, k, rnd, a.n)
        p = os.path.join(a.outdir, f"{name}.jsonl")
        with open(p, "w") as f:
            for x in b:
                f.write(json.dumps(x) + "\n")
        print(f"=== {name}  (gqa {r}, {maxrun} resident reqs, k={k} -> {k+1} candidates)")
        print(f"    {note}")
        summarize(b, r, name)
        print(f"    -> {p}  ({len(b)} batches)\n")


if __name__ == "__main__":
    main()
