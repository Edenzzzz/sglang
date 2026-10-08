#!/usr/bin/env python3
"""Does the scheduler pick the measured winner? Cell-by-cell against both sweeps.

Scores the cost model on what actually matters -- not the absolute latency it
predicts, but whether it selects the fastest of {fused, seq, conc} and, when it
picks conc, a split close to the best measured one.
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from four_mode_dispatch import plan_split, t_fa3, t_pers, fa3_width, NUM_SM

DEC_KV = {(8, 1, 128): 11000, (28, 4, 128): 8000}

def main():
    rows = []
    for f, h in (("boundary_8x1x128.json", (8, 1, 128)),
                 ("boundary_28x4x128.json", (28, 4, 128))):
        for c in json.load(open(f)):
            c["h"] = h; rows.append(c)
    ok = lost = 0; loss = []
    print(f"{'heads':>10} {'tiles':>6} {'n_dec':>6} {'width':>6} {'measured':>9} "
          f"{'picked':>12} {'t_pick':>8} {'t_best':>8} {'slack':>7}")
    for c in rows:
        h = c["h"]; gqa = h[0] // h[1]
        big = [(1024, c["kv"])]; dec = [(1, DEC_KV[h])] * c["nd"]
        # routing: the prefill row is FA3-eligible, the decode rows never are
        mode, cf, cp = plan_split(big, dec, h[0], h[1])
        times = {"fused": c["fused"], "seq": c["seq"]}
        if c["conc"]: times["conc"] = c["conc"]
        t_pick = times[mode] if mode in times else c["fused"]
        t_best = min(times.values())
        slack = t_pick / t_best
        if slack <= 1.05: ok += 1
        else: lost += 1; loss.append(slack)
        print(f"{str(h):>10} {c['kv']//128:>6} {c['nd']:>6} "
              f"{fa3_width(big,h[0]):>6} {c['win']:>9} {f'{mode} {cf}/{cp}':>12} "
              f"{t_pick:>8.1f} {t_best:>8.1f} {slack:>6.2f}x", flush=True)
    print(f"\npicks within 5% of the best measured strategy: {ok}/{ok+lost}")
    if loss:
        print(f"  misses: median {np.median(loss):.2f}x worst {max(loss):.2f}x")

if __name__ == "__main__":
    main()
