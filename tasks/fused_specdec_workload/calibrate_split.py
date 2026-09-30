#!/usr/bin/env python3
"""Measure each side's standalone time vs grid size and fit the scheduler model.

Fits, per side, over real captured class populations:
  FA3:        t(c) = W_mac / (c * R_FA3)
  persistent: t(c) = max(W_mac / (c * R_PERS), Bytes / (BW_PEAK * min(1, c/C_SAT)))
then reports the fitted constants and the residuals, so the scheduler's split is
grounded in measurement rather than a guessed ratio.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_4mode import Bench, uncapped, med
from four_mode_dispatch import classify, route, _work, _bytes

CAPS = [8, 16, 24, 32, 48, 64, 84, 100, 116, 132]

def main():
    st = sys.argv[1] if len(sys.argv) > 1 else "sets/qwen_small_h800_largeprefill.jsonl"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    heads = (28, 4, 128)
    b = Bench(heads); gqa = heads[0] // heads[1]
    rows_f, rows_p = [], []
    print(f"{'#':>3} {'side':>5} " + " ".join(f"{c:>7}" for c in CAPS))
    for i, bt in enumerate([json.loads(l) for l in open(st) if l.strip()][:n]):
        cls = classify(bt, gqa); fa3_rows, pers_rows = route(cls, gqa)
        if not fa3_rows or not pers_rows or not b.fits(fa3_rows + pers_rows):
            continue
        ts = []
        for c in CAPS:
            os.environ["FLASHINFER_FA3_MAX_CTAS"] = str(c)
            qf = b.plan_fa3(fa3_rows)
            ts.append(med(lambda: b.fa3.run(qf, b.kv)))
        rows_f.append((_work(fa3_rows, heads[0]), ts))
        print(f"{i:>3} {'fa3':>5} " + " ".join(f"{t:>7.0f}" for t in ts), flush=True)
        ts = []
        for c in CAPS:
            os.environ["FLASHINFER_PERSISTENT_MAX_CLUSTERS"] = str(c)
            qp, op, lp = b.plan_pa(pers_rows, False)
            ts.append(med(lambda: b.pa.run(qp, b.kv, out=op, lse=lp)))
        rows_p.append((_work(pers_rows, heads[0]), _bytes(pers_rows, heads[1]), ts))
        print(f"{i:>3} {'pers':>5} " + " ".join(f"{t:>7.0f}" for t in ts), flush=True)
    uncapped()

    # FA3: R = W/(c*t), one estimate per (batch, cap)
    est = [w / (c * t * 1e3) for w, ts in rows_f for c, t in zip(CAPS, ts)]
    print(f"\nR_FA3  = {np.median(est):.3e} MAC/ns/CTA   "
          f"(p10 {np.percentile(est,10):.2e} p90 {np.percentile(est,90):.2e})")
    # persistent: at large c it is bandwidth-bound -> BW from the plateau;
    # at small c compute/latency-bound -> R_PERS from the slope.
    bw = [bts / (t * 1e3) for _, bts, ts in rows_p for c, t in zip(CAPS, ts) if c >= 100]
    print(f"BW_PEAK= {np.median(bw):.3e} B/ns   (= {np.median(bw):.0f} GB/s)")
    rp = [w / (c * t * 1e3) for w, _, ts in rows_p for c, t in zip(CAPS, ts) if c <= 24]
    print(f"R_PERS = {np.median(rp):.3e} MAC/ns/CTA")
    # C_SAT: smallest cap whose time is within 5% of the cap=132 time
    sats = []
    for _, _, ts in rows_p:
        base = ts[-1]
        for c, t in zip(CAPS, ts):
            if t <= 1.05 * base:
                sats.append(c); break
    print(f"C_SAT  = {int(np.median(sats))} clusters (median over {len(sats)} batches; "
          f"all: {sats})")
    json.dump(dict(caps=CAPS, fa3=rows_f, pers=rows_p), open("calib_split.json", "w"))

if __name__ == "__main__":
    main()
