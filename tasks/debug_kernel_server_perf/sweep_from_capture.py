#!/usr/bin/env python3
"""Re-profile FA2 BatchPrefill vs the persistent kernel on REAL compositions.

The deployed gate table was built on a Cartesian grid. This instead replays
compositions captured from live serving (SGLANG_LOG_MIXED_COMPOSITION), so the
speedup distribution is weighted by what the scheduler actually produces
rather than by grid density.

Compositions are bucketed (log-spaced on decode rows / total decode KV / chunk
tokens / prefix) and one representative per bucket is benchmarked, carrying the
bucket's realized frequency as its weight.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_sweep import Bench  # noqa: E402
import flashinfer  # noqa: E402
from flashinfer.testing.utils import bench_gpu_time  # noqa: E402


def load_rows(patterns):
    out = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            tag = os.path.basename(os.path.dirname(f))
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r["d"] and r["q"]:
                        r["_tag"] = tag
                        out.append(r)
    return out


def bucket(r):
    d = np.asarray(r["d"], float)
    q = np.asarray(r["q"], float)
    p = np.asarray(r["p"], float)
    lg = lambda v, b: int(np.floor(np.log2(max(v, 1.0)) * b))
    return (lg(len(d), 1), lg(d.sum(), 1), lg(q.sum(), 1), lg(p.max() if len(p) else 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--heads", default="28,4,128")
    ap.add_argument("--out", default="tasks/debug_kernel_server_perf/sweep_captured_qwen25_h800.json")
    ap.add_argument("--max-cases", type=int, default=80)
    a = ap.parse_args()

    rows = load_rows(a.paths)
    print(f"{len(rows)} captured mixed batches")
    groups = {}
    for r in rows:
        groups.setdefault(bucket(r), []).append(r)
    order = sorted(groups.items(), key=lambda kv: -len(kv[1]))[: a.max_cases]
    covered = sum(len(v) for _, v in order)
    print(f"{len(groups)} buckets; benchmarking top {len(order)} "
          f"({100*covered/len(rows):.1f}% of batches)")

    heads = tuple(int(x) for x in a.heads.split(","))
    b = Bench(heads)
    # FA3 (Hopper) alongside the FA2 baseline and both persistent schedules.
    ws3 = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    bp3 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws3, kv_layout="NHD", backend="fa3")
    res = []
    for i, (key, members) in enumerate(order):
        # median member of the bucket by total decode KV
        members = sorted(members, key=lambda r: sum(r["d"]))
        r = members[len(members) // 2]
        d_kvl = [int(x) for x in r["d"]]
        p_lens = [int(x) for x in r["q"]]
        p_kvl = [int(x) for x in r["p"]]
        q_lens = torch.tensor([1] * len(d_kvl) + p_lens, dtype=torch.int32)
        kv_lens = torch.tensor(d_kvl + p_kvl, dtype=torch.int32)
        if int(kv_lens.sum()) > 7_900_000 or int(q_lens.sum()) > 32768:
            continue
        q_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0).int()])
        kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(kv_lens, 0).int()])
        nq = int(q_indptr[-1])
        last = torch.ones_like(kv_lens)
        q = b.qbuf[:nq]
        try:
            b.bp.plan(q_indptr, kv_indptr, b.kv_indices, last, *heads[:2], heads[2], 1,
                      causal=False, q_data_type=torch.bfloat16,
                      kv_data_type=torch.bfloat16, non_blocking=True)
            bp = float(np.median(bench_gpu_time(lambda: b.bp.run(q, b.kv))))
            try:
                bp3.plan(q_indptr, kv_indptr, b.kv_indices, last, *heads[:2], heads[2], 1,
                         causal=False, q_data_type=torch.bfloat16,
                         kv_data_type=torch.bfloat16, non_blocking=True)
                fa3 = float(np.median(bench_gpu_time(lambda: bp3.run(q, b.kv))))
            except Exception as e3:
                if i == 0:
                    print(f"  fa3 unavailable: {str(e3)[:90]}", flush=True)
                fa3 = float("nan")
            pa = {}
            for flip in (True, False):
                b.pa.plan(q_indptr, kv_indptr, b.kv_indices, kv_lens, heads[0], heads[1],
                          heads[2], heads[2], 1, causal=False, q_data_type=torch.bfloat16,
                          kv_data_type=torch.bfloat16, flipped_schedule=flip)
                o = torch.empty(nq, heads[0], heads[2], device="cuda", dtype=torch.bfloat16)
                l = torch.empty(nq, heads[0], device="cuda", dtype=torch.float32)
                pa[flip] = float(np.median(bench_gpu_time(lambda: b.pa.run(q, b.kv, out=o, lse=l))))
        except Exception as e:
            print(f"  [{i}] skip: {str(e)[:90]}", flush=True)
            continue
        best = min(pa.values())
        res.append(dict(weight=len(members), n_dec=len(d_kvl), dec_kv=int(sum(d_kvl)),
                        q_tot=int(sum(p_lens)), pfx_max=int(max(p_kvl)) if p_kvl else 0,
                        bp_ms=bp, fa3_ms=fa3, pa_flip_ms=pa[True],
                        pa_noflip_ms=pa[False], spd=bp / best,
                        gate_pa=int(r["pa"]), tag=r["_tag"]))
        print(f"  [{i:3d}] w={len(members):5d} {len(d_kvl):3d}d/{sum(d_kvl)/1e3:7.1f}K "
              f"q={sum(p_lens):5d}@{max(p_kvl) if p_kvl else 0:6d} "
              f"fa2={bp*1e3:6.0f} fa3={fa3*1e3:6.0f} pa={best*1e3:6.0f}us  "
              f"pa/fa2 {bp/best:5.2f}x  fa3/fa2 {bp/fa3:5.2f}x", flush=True)
    json.dump(res, open(a.out, "w"))

    w = np.array([r["weight"] for r in res], float)
    s = np.array([r["spd"] for r in res], float)
    print(f"\nfrequency-weighted over {int(w.sum())} real mixed batches:")
    print(f"  mean speedup   {np.average(s, weights=w):.3f}x")
    print(f"  median         {s[np.argsort(s)][np.searchsorted(np.cumsum(w[np.argsort(s)])/w.sum(), .5)]:.3f}x")
    print(f"  batches >1.1x  {100*w[s > 1.1].sum()/w.sum():.1f}%")
    print(f"  batches <0.95x {100*w[s < 0.95].sum()/w.sum():.1f}%")
    f3 = np.array([r["fa3_ms"] for r in res], float)
    ok = ~np.isnan(f3)
    if ok.any():
        s3 = np.array([r["bp_ms"] for r in res])[ok] / f3[ok]
        sp3 = np.array([r["bp_ms"] for r in res])[ok] / np.minimum(
            f3[ok], np.array([min(r["pa_flip_ms"], r["pa_noflip_ms"]) for r in res])[ok])
        print(f"  FA3 vs FA2     {np.average(s3, weights=w[ok]):.3f}x  "
              f"(FA3 wins on {100*w[ok][s3 > 1.0].sum()/w[ok].sum():.0f}% of weight)")
        pa_ms = np.array([min(r["pa_flip_ms"], r["pa_noflip_ms"]) for r in res])
        print(f"  best-of-all    {np.average(sp3, weights=w[ok]):.3f}x  "
              f"(persistent beats FA3 on {100*w[ok][(pa_ms[ok] < f3[ok])].sum()/w[ok].sum():.0f}% of weight)")
    gp = np.array([r["gate_pa"] for r in res], bool)
    if gp.any():
        print(f"  gate chose PA on {100*w[gp].sum()/w.sum():.1f}% of weight; "
              f"their realized speedup {np.average(s[gp], weights=w[gp]):.3f}x")
    if (~gp).any():
        print(f"  gate chose BP on {100*w[~gp].sum()/w.sum():.1f}%; "
              f"their PA-speedup would have been {np.average(s[~gp], weights=w[~gp]):.3f}x")


if __name__ == "__main__":
    main()
