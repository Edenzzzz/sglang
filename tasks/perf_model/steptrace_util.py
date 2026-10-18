#!/usr/bin/env python3
"""GPU busy fraction and per-kind step time from STEPTRACE logs (SGLANG_STEP_TRACE=1).

Kinds: mixed = a prefill chunk and decode/verify rows in one forward (np>0, nd>0); prefill = prefill rows only;
decode = decode/verify rows only. idle after a step = launch-to-launch interval minus that step's GPU time (floored at 0),
counted only for back-to-back steps (interval < 200 ms): host time the GPU spent without this server's work.
Usage: steptrace_util.py <server.log> [...]"""
import re
import sys

PAT = re.compile(r"STEPTRACE (\w+) et=(\d+) bs=(\d+) np=(\d+) ptok=(\d+) pf=(\d+) nd=(\d+) dkv=(\d+) kv=(\d+) "
                 r"mx=(\d+)(?: sg=\d+ lay=\d+)?(?: acc=[0-9.]+)?(?: dacc=[0-9.]+)? t=([0-9.]+) gpu=([0-9.]+)")

for path in sys.argv[1:]:
    rs = []
    for line in open(path, errors="replace"):
        m = PAT.search(line)
        if m:
            g = m.groups()
            np_, nd = int(g[3]), int(g[6])
            kind = "mixed" if np_ and nd else ("prefill" if np_ else "decode")
            rs.append((float(g[10]) * 1e3, float(g[11]), kind, int(g[4])))
    rs.sort()
    if len(rs) < 2:
        print(path, "no STEPTRACE"); continue
    wall = rs[-1][0] - rs[0][0]
    busy = sum(r[1] for r in rs)
    print(f"{path}\n  steps={len(rs)} wall={wall/1e3:.1f}s gpu_busy={busy/1e3:.1f}s ({busy/wall*100:.1f}%)")
    for kind in ("decode", "prefill", "mixed"):
        ks = [(a, b) for a, b in zip(rs, rs[1:]) if a[2] == kind]
        if not ks:
            continue
        gpu = [a[1] for a, _ in ks]
        bb = [(b[0] - a[0], a[1]) for a, b in ks if b[0] - a[0] < 200]
        idle = [max(0.0, iv - gp) for iv, gp in bb]
        ptok = [a[3] for a, _ in ks]
        print(f"  {kind:7s} n={len(ks):6d} gpu_mean={sum(gpu)/len(gpu):7.2f}ms gpu_total={sum(gpu)/1e3:6.1f}s "
              f"ptok_mean={sum(ptok)/len(ptok):7.0f} back2back={len(bb)} interval_mean="
              f"{(sum(i for i, _ in bb)/len(bb) if bb else 0):7.2f}ms idle_after_mean={(sum(idle)/len(idle) if idle else 0):6.2f}ms "
              f"idle_total={sum(idle)/1e3:6.1f}s")
