#!/usr/bin/env python3
"""SLO attainment per arm from bench_serving --output-details JSONs.

A request attains its SLO when TTFT <= --ttft and its P90 per-token ITL <= --itl (per-token ITL
as bench_serving records it: a chunk's gap divided by its tokens). Also reported: the share of
all tokens within the ITL SLO, and goodput (attaining requests per second of run time).

Usage: slo_attain.py --root <dir with r1_<ARM>/ARM_<stream>.json> --ttft 2000 --itl 50 [--scaled]
"""
import argparse, glob, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--ttft", type=float, required=True)
ap.add_argument("--itl", type=float, required=True)
# --scaled: the TTFT SLO EDF itself schedules against (edf_scheduler.compute_ttft_slo_ms),
# ttft + 1000 ms per 5000 prompt tokens beyond 4096. Without it the SLO is flat.
ap.add_argument("--scaled", action="store_true")
a = ap.parse_args()
for d in sorted(glob.glob(os.path.join(a.root, "r*_*"))):
    arm = os.path.basename(d).split("_", 1)[1]
    for f in sorted(glob.glob(os.path.join(d, f"{arm}_*.json"))):
        stream = os.path.basename(f)[len(arm) + 1:-5]
        j = json.loads(open(f).read().strip().splitlines()[-1])
        ttfts = np.array(j["ttfts"]) * 1e3
        itls = j["itls"]
        slo = np.full(len(ttfts), a.ttft)
        if a.scaled:
            slo = slo + np.maximum(np.array(j["input_lens"]) - 4096, 0) / 5000 * 1000
        ok_t = ttfts <= slo
        ok_i = np.array([(np.percentile(np.array(x) * 1e3, 90) if len(x) else 0.0) <= a.itl for x in itls])
        tok = np.concatenate([np.array(x) * 1e3 for x in itls if len(x)])
        both = ok_t & ok_i
        print(f"{arm:20s} {stream:6s} n={len(ttfts):5d} ttft_ok={ok_t.mean():.3f} itl_ok={ok_i.mean():.3f} "
              f"slo_ok={both.mean():.3f} tok_itl_ok={(tok <= a.itl).mean():.4f} "
              f"goodput={both.sum() / j['duration']:.2f}/s mean_ttft={ttfts.mean():.0f} p99_itl={np.percentile(tok, 99):.1f}")
