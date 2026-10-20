"""Compare two serving runs' STEPTRACE logs (SGLANG_STEP_TRACE=1) by step type: count, mean / p50 / p90 GPU ms, and the
mean GPU ms of 4-mode mixed steps binned by prefill tokens x prefix depth, so a kernel change can be located in the step
it affects.
  python steptrace_compare.py A_worker.log B_worker.log
Claude, 2026-10-21.
"""
import re
import sys

import numpy as np

PAT = re.compile(r"STEPTRACE (\w+) et=(\d+) bs=(\d+) np=(\d+) ptok=(\d+) pf=(\d+) nd=(\d+) dkv=(\d+) kv=(\d+) "
                 r"mx=(\d+)(?: sg=(\d+) lay=(\d+))?(?: acc=([0-9.]+))?(?: dacc=[0-9.]+)? t=([0-9.]+) gpu=([0-9.]+)")


def load(path):
    rows = []
    for line in open(path, errors="replace"):
        m = PAT.search(line)
        if m:
            g = m.groups()
            rows.append(dict(kind=g[0], bs=int(g[2]), np=int(g[3]), ptok=int(g[4]), pf=int(g[5]), nd=int(g[6]),
                             mx=int(g[9]), gpu=float(g[14])))
    return rows


runs = {p: load(p) for p in sys.argv[1:3]}
for p, rs in runs.items():
    print(f"== {p.split('/')[-2]}: {len(rs)} steps")
    for kind in sorted({r["kind"] for r in rs}):
        for mixed in (0, 1):
            g = np.array([r["gpu"] for r in rs if r["kind"] == kind and (r["mx"] > 0) == mixed])
            if len(g):
                print(f"  {kind:8s} mixed={mixed}: n={len(g):6d} mean={g.mean():7.2f} p50={np.median(g):7.2f} "
                      f"p90={np.percentile(g, 90):7.2f} ms  total={g.sum() / 1e3:7.1f} s")
    for lo, hi in ((0, 600), (600, 1200), (1200, 2100)):
        for plo, phi in ((0, 16384), (16384, 49152), (49152, 1 << 30)):
            g = np.array([r["gpu"] for r in rs if r["mx"] > 0 and lo <= r["ptok"] < hi and plo <= r["pf"] < phi])
            if len(g) >= 5:
                print(f"  mixed ptok [{lo},{hi}) prefix [{plo},{phi}): n={len(g):5d} mean={g.mean():7.2f} ms")
