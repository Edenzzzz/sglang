#!/usr/bin/env python3
"""Summarize the realized mixed-batch composition distribution.

Input: composition_*.jsonl written by SGLANG_LOG_MIXED_COMPOSITION (one record
per mixed batch: per-row decode KV `d`, chunk tokens `q`, attended prefix `p`,
gate decision `pa`/`fl`).

This is the distribution the kernel actually sees. The gate table was instead
built on a Cartesian grid, so the point of this is to show where the grid is
over- and under-sampled relative to real serving.
"""
import argparse
import glob
import json
import os

import numpy as np


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # partial last line while the server is still writing
    return rows


def feats(r):
    d = np.asarray(r["d"], dtype=np.float64)
    q = np.asarray(r["q"], dtype=np.float64)
    p = np.asarray(r["p"], dtype=np.float64)
    D = d.sum()
    return dict(
        bs=len(d), D=D, kv_max=d.max() if len(d) else 0.0,
        conc=float((d * d).sum() / D) if D > 0 else 0.0,
        n_chunks=len(q), q_tot=q.sum(), q_max=q.max() if len(q) else 0.0,
        pfx_max=p.max() if len(p) else 0.0,
        work=float((q * p).sum()), pa=r["pa"],
    )


def pct(v, qs=(10, 50, 90)):
    return [np.percentile(v, x) if len(v) else 0.0 for x in qs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    files = [f for p in a.paths for f in sorted(glob.glob(p))]

    print(f"{'run':<14} {'n':>6} {'decode rows':>18} {'chunk tokens':>20} "
          f"{'attended prefix':>20} {'decode KV tot':>18} {'gate→PA':>8}")
    print(f"{'':<14} {'':>6} {'p10/p50/p90':>18} {'p10/p50/p90':>20} "
          f"{'p10/p50/p90':>20} {'p10/p50/p90':>18}")
    allf = {}
    for f in files:
        rows = load(f)
        if not rows:
            continue
        F = [feats(r) for r in rows]
        tag = os.path.basename(os.path.dirname(f)).replace("h800_capture_loadsweep_", "")
        allf[tag] = F
        bs = np.array([x["bs"] for x in F]); q = np.array([x["q_max"] for x in F])
        pf = np.array([x["pfx_max"] for x in F]); D = np.array([x["D"] for x in F])
        pa = np.mean([x["pa"] for x in F])
        fmt = lambda v: "/".join(f"{x:.0f}" for x in pct(v))
        print(f"{tag:<14} {len(F):>6} {fmt(bs):>18} {fmt(q):>20} {fmt(pf):>20} "
              f"{fmt(D/1e3)+'K':>18} {pa*100:>7.0f}%")

    print("\n--- chunk-size mass vs gate-grid sample points ---")
    # A mean distance-to-grid is dominated by a small tail of tiny chunks, so
    # report where the mass actually is against how many grid points cover it.
    grid_q = [128, 512, 1024, 2048, 4096, 7168, 8192]
    bins = [(0, 128), (128, 512), (512, 2048), (2048, 8192), (8192, 1 << 30)]
    print(f"  {'bin':<14} {'grid pts':>9} " + " ".join(f"{t:>8}" for t in allf) + f" {'all':>8}")
    qs = {t: np.array([v for x in F for v in [x["q_max"]]]) for t, F in allf.items()}
    alls = np.concatenate(list(qs.values())) if qs else np.array([])
    for lo, hi in bins:
        npts = sum(1 for g in grid_q if lo <= g < hi)
        cols = " ".join(f"{100*((v>=lo)&(v<hi)).mean():7.1f}%" for v in qs.values())
        allc = 100 * ((alls >= lo) & (alls < hi)).mean() if len(alls) else 0.0
        print(f"  {str(lo)+'-'+str(hi if hi < (1<<30) else 'inf'):<14} {npts:>9} {cols} {allc:7.1f}%")
    print()
    for tag, F in allf.items():
        frac_fresh = np.mean([x["pfx_max"] < 512 for x in F])
        conc = np.array([x["conc"] for x in F])
        pa = np.mean([x["pa"] for x in F])
        print(f"  {tag:<12} fresh-prefix {frac_fresh*100:4.0f}% | KV-conc p50 "
              f"{np.median(conc)/1e3:6.1f}K | gate picks persistent {pa*100:3.0f}%")


if __name__ == "__main__":
    main()
