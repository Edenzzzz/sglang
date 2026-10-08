#!/usr/bin/env python3
"""Host-side gap per step type, from a SGLANG_STEP_TRACE=1 server log.

Each step logs its launch time t (host perf_counter) and its GPU span gpu (CUDA events).
For back-to-back steps (launch-to-launch interval < --busy-ms), interval - gpu is time the
GPU sat idle waiting on the host: scheduling, metadata, plan, and kernel launches. CUDA-graph
capture can only remove the launch part of it, so this sum is the graph's ceiling.

Usage: steptrace_gap.py server_SPLIT.log [--busy-ms 1000]
"""
import argparse, collections, re

LINE = re.compile(r"STEPTRACE (\w+) .*? t=([0-9.]+) gpu=([0-9.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--busy-ms", type=float, default=1000.0)
    a = ap.parse_args()
    rows = [(m.group(1), float(m.group(2)), float(m.group(3)))
            for m in LINE.finditer(open(a.log, errors="replace").read())]
    rows.sort(key=lambda r: r[1])
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for (mode, t, gpu), (_, t2, _) in zip(rows, rows[1:]):
        dt = (t2 - t) * 1e3
        if dt > a.busy_ms:
            continue
        g = agg[mode]
        g[0] += 1
        g[1] += dt
        g[2] += gpu
    tot_w = sum(v[1] for v in agg.values())
    print(f"{'mode':<10}{'steps':>8}{'wall ms':>12}{'gpu ms':>12}{'host gap':>10}{'gap/step':>10}{'share of busy wall':>20}")
    for mode, (n, w, g) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"{mode:<10}{n:>8}{w:>12.0f}{g:>12.0f}{(w - g) / w * 100:>9.1f}%{(w - g) / n:>9.2f}ms"
              f"{w / tot_w * 100:>19.1f}%")


if __name__ == "__main__":
    main()
