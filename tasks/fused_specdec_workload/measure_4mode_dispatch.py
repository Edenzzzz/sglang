#!/usr/bin/env python3
"""Measure the 4-mode dispatch opportunity on the local (Hopper) workload set.

For each fused batch we time:
  monolithic  - the whole batch through one kernel (FA2 / FA3 / persistent)
  split       - each class routed to its best kernel and the times summed
                (an upper bound on a dispatching scheduler that runs the
                classes back-to-back; a real one overlaps them on disjoint SM
                partitions, so this is pessimistic on the split side)

That difference is the headroom a 4-mode dispatcher has over today's single
kernel, measured rather than assumed.
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "debug_kernel_server_perf"))
from gate_sweep import Bench
import flashinfer
from flashinfer.testing.utils import bench_gpu_time

CLASSES = ["large_prefill", "spec_verify", "mixed_decode", "spec_draft"]


def time_rows(b, bp3, heads, rows, want=("fa2", "fa3", "pers")):
    if not rows:
        return {k: 0.0 for k in want}
    qs = torch.tensor([q for q, _ in rows], dtype=torch.int32)
    ks = torch.tensor([max(kv, 1) for _, kv in rows], dtype=torch.int32)
    if int(ks.sum()) > 7_500_000 or int(qs.sum()) > 32768:
        return None
    qi = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(qs, 0).int()])
    ki = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(ks, 0).int()])
    nq = int(qi[-1]); last = torch.ones_like(ks); q = b.qbuf[:nq]
    out = {}
    try:
        if "fa2" in want:
            b.bp.plan(qi, ki, b.kv_indices, last, *heads[:2], heads[2], 1, causal=False,
                      q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, non_blocking=True)
            out["fa2"] = float(np.median(bench_gpu_time(lambda: b.bp.run(q, b.kv)))) * 1e3
        if "fa3" in want:
            bp3.plan(qi, ki, b.kv_indices, last, *heads[:2], heads[2], 1, causal=False,
                     q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, non_blocking=True)
            out["fa3"] = float(np.median(bench_gpu_time(lambda: bp3.run(q, b.kv)))) * 1e3
        if "pers" in want:
            best = None
            for f in (True, False):
                b.pa.plan(qi, ki, b.kv_indices, ks, heads[0], heads[1], heads[2], heads[2], 1,
                          causal=False, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                          flipped_schedule=f)
                o = torch.empty(nq, heads[0], heads[2], device="cuda", dtype=torch.bfloat16)
                l = torch.empty(nq, heads[0], device="cuda", dtype=torch.float32)
                t = float(np.median(bench_gpu_time(lambda: b.pa.run(q, b.kv, out=o, lse=l)))) * 1e3
                best = t if best is None else min(best, t)
            out["pers"] = best
    except Exception:
        return None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="tasks/fused_specdec_workload/sets/qwen_small_h800.jsonl")
    ap.add_argument("--heads", default="28,4,128")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="tasks/fused_specdec_workload/dispatch_4mode_h800.json")
    a = ap.parse_args()
    heads = tuple(int(x) for x in a.heads.split(","))
    b = Bench(heads)
    ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    bp3 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD", backend="fa3")

    batches = [json.loads(l) for l in open(a.set) if l.strip()][: a.n]
    res = []
    print(f"{'#':>3} {'mono fa2':>9} {'mono fa3':>9} {'mono pers':>10} {'split best':>11} {'gain':>7}")
    for i, bt in enumerate(batches):
        allrows = [tuple(x) for c in CLASSES for x in bt[c]]
        mono = time_rows(b, bp3, heads, allrows)
        if mono is None:
            continue
        per, ok = {}, True
        for c in CLASSES:
            rows = [tuple(x) for x in bt[c]]
            if not rows:
                per[c] = ("-", 0.0); continue
            t = time_rows(b, bp3, heads, rows)
            if t is None:
                ok = False; break
            k = min(t, key=t.get)
            per[c] = (k, t[k])
        if not ok:
            continue
        split = sum(v for _, v in per.values())
        best_mono = min(mono.values())
        res.append(dict(mono=mono, per={k: [v[0], v[1]] for k, v in per.items()},
                        split=split, best_mono=best_mono))
        print(f"{i:>3} {mono['fa2']:>9.1f} {mono['fa3']:>9.1f} {mono['pers']:>10.1f} "
              f"{split:>11.1f} {best_mono/split:>6.2f}x", flush=True)

    json.dump(res, open(a.out, "w"))
    bm = np.array([r["best_mono"] for r in res]); sp = np.array([r["split"] for r in res])
    mono_pers = np.array([r["mono"]["pers"] for r in res])
    print(f"\n{len(res)} batches")
    print(f"  monolithic persistent (today)  {mono_pers.sum():9.1f} us")
    print(f"  monolithic best-of-3           {bm.sum():9.1f} us   {mono_pers.sum()/bm.sum():.2f}x")
    print(f"  4-mode split (sum of classes)  {sp.sum():9.1f} us   {mono_pers.sum()/sp.sum():.2f}x vs today")
    from collections import Counter
    for c in CLASSES:
        w = Counter(r["per"][c][0] for r in res)
        print(f"    {c:<15} best kernel: {dict(w)}")


if __name__ == "__main__":
    main()
