#!/usr/bin/env python3
"""Summarize the simulated fused draft+verify+prefill workload, and emit the
representative compositions a fusion kernel should be tuned on.

Writes `workload_spec.json`: frequency-weighted representative batches plus the
per-class row/KV distributions, so kernel work can target the realized mixture
rather than hand-picked shapes.
"""
import argparse, json, collections
import numpy as np


def tile_fa2(p): return 128 if p > 64 else (64 if p > 16 else 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--gqa", type=int, default=7)
    ap.add_argument("--out", default="tasks/fused_specdec_workload/workload_spec.json")
    ap.add_argument("--top", type=int, default=24)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    r = a.gqa

    def pct(v, q=(10, 50, 90)):
        return [int(np.percentile(v, x)) for x in q] if len(v) else [0, 0, 0]

    nd = np.array([len(x["draft"]) for x in recs])
    nv = np.array([len(x["verify"]) for x in recs])
    npf = np.array([len(x["prefill"]) for x in recs])
    print(f"{len(recs)} fused steps, gqa ratio {r}\n")
    print(f"{'per-batch rows':<22} {'p10/p50/p90':>16}")
    for nm, v in (("draft rows", nd), ("verify rows", nv), ("prefill chunks", npf)):
        print(f"  {nm:<20} {'/'.join(map(str,pct(v))):>16}")

    dkv = np.array([kv for x in recs for _, kv in x["draft"]])
    vkv = np.array([kv for x in recs for _, kv in x["verify"]])
    ch = np.array([q for x in recs for q, _ in x["prefill"]])
    pfx = np.array([p for x in recs for _, p in x["prefill"]])
    print(f"\n{'per-row KV / chunk':<22} {'p10/p50/p90':>16}")
    for nm, v in (("draft row KV", dkv), ("verify row KV", vkv),
                  ("chunk tokens", ch), ("chunk attended prefix", pfx)):
        print(f"  {nm:<20} {'/'.join(map(str,pct(v))):>16}")

    # tile classes actually present per batch
    cnt = collections.Counter()
    for x in recs:
        t = set()
        for q, _ in x["draft"]:   t.add(tile_fa2(q * r))
        for q, _ in x["verify"]:  t.add(tile_fa2(q * r))
        for q, _ in x["prefill"]: t.add(tile_fa2(q * r))
        cnt[tuple(sorted(t))] += 1
    print("\ntile classes co-resident in one fused batch:")
    for t, n in cnt.most_common():
        print(f"  {str(list(t)):<18} {n:>6} steps ({100*n/len(recs):5.1f}%)")

    # KV concentration of the decode-side population (drives the gate)
    conc = []
    for x in recs:
        rows = [kv for _, kv in x["draft"]] + [kv for _, kv in x["verify"]]
        D = sum(rows)
        if D:
            conc.append(sum(v * v for v in rows) / D)
    if conc:
        print(f"\nKV concentration (sum kv^2 / sum kv): p10/p50/p90 = "
              f"{'/'.join(f'{x/1e3:.1f}K' for x in np.percentile(conc,[10,50,90]))}")

    # representative compositions, bucketed log-spaced and frequency-weighted
    def bucket(x):
        lg = lambda v: int(np.floor(np.log2(max(v, 1))))
        return (lg(len(x["draft"])), lg(len(x["verify"])), len(x["prefill"]) > 0,
                lg(sum(kv for _, kv in x["draft"]) + sum(kv for _, kv in x["verify"])),
                lg(sum(q for q, _ in x["prefill"])) if x["prefill"] else -1)
    g = collections.defaultdict(list)
    for x in recs:
        g[bucket(x)].append(x)
    top = sorted(g.items(), key=lambda kv: -len(kv[1]))[: a.top]
    spec = []
    for key, members in top:
        m = sorted(members, key=lambda x: sum(kv for _, kv in x["draft"]))[len(members) // 2]
        spec.append(dict(weight=len(members), draft=m["draft"], verify=m["verify"],
                         prefill=m["prefill"]))
    cov = sum(s["weight"] for s in spec)
    json.dump(dict(gqa_ratio=r, n_steps=len(recs), source=a.jsonl,
                   coverage=cov / len(recs), compositions=spec), open(a.out, "w"), indent=1)
    print(f"\n{len(spec)} representative compositions -> {a.out} "
          f"({100*cov/len(recs):.1f}% of steps)")
    print(f"{'w':>6} {'draft':>7} {'verify':>7} {'chunk':>7} {'decKV':>9} {'tiles':>14}")
    for s in spec[:12]:
        D = sum(kv for _, kv in s["draft"]) + sum(kv for _, kv in s["verify"])
        t = sorted({tile_fa2(q * r) for q, _ in s["draft"] + s["verify"] + s["prefill"]})
        print(f"{s['weight']:>6} {len(s['draft']):>7} {len(s['verify']):>7} "
              f"{sum(q for q,_ in s['prefill']):>7} {D/1e3:>8.0f}K {str(t):>14}")


if __name__ == "__main__":
    main()
