#!/usr/bin/env python3
"""Draw the dispatch boundary: when does the large class belong on pure FA3?

Sweeps the two axes that decide it -- how many KV tiles the large-prefill row
attends, and how big the co-resident decode/verify population is -- and measures
all three strategies in each cell:
  fused  one persistent launch over everything (3 tile classes, full grid)
  seq    FA3 on the prefill row, then persistent on the rest (vLLM-style)
  conc   the two co-launched at the best grid split (swept)
Prints the winner per cell, which is the boundary the scheduler has to encode.
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_4mode import Bench, uncapped, med
from four_mode_dispatch import Dispatcher

KVS = [512, 2048, 8192, 16384, 32768]
NDEC = [0, 8, 32, 115, 365]
SPLITS = [16, 32, 48, 66, 88, 110]   # SMs given to FA3; persistent gets the rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="8,1,128")
    ap.add_argument("--qo", type=int, default=1024)
    ap.add_argument("--dec-kv", type=int, default=11000)
    a = ap.parse_args()
    heads = tuple(int(x) for x in a.heads.split(","))
    b = Bench(heads); d = Dispatcher(b, heads[0], heads[1])
    print(f"heads {heads}  prefill qo={a.qo}  decode kv={a.dec_kv}\n")
    print(f"{'kv_tiles':>9} {'n_dec':>6} {'fused':>9} {'seq':>9} {'conc':>9} "
          f"{'split':>7} {'winner':>7} {'margin':>7}")
    out = []
    for kv in KVS:
        for nd in NDEC:
            big = [(a.qo, kv)]
            dec = [(1, a.dec_kv)] * nd
            allr = big + dec
            if not b.fits(allr):
                continue
            uncapped()
            qa, oa, la = b.plan_pa(allr, False)
            t_fused = med(lambda: b.pa.run(qa, b.kv, out=oa, lse=la))
            qf = b.plan_fa3(big)
            if nd:
                qp, op, lp = b.plan_pa(dec, False)
                t_seq = med(lambda: (b.fa3.run(qf, b.kv),
                                     b.pa.run(qp, b.kv, out=op, lse=lp)))
            else:
                t_seq = med(lambda: b.fa3.run(qf, b.kv))
            t_conc, bs = float("inf"), None
            if nd:
                for cf in SPLITS:
                    d.plan(big, dec, c_fa3=cf, c_pers=132 - cf, mode="conc")   # SMs
                    t = med(d.run)
                    if t < t_conc:
                        t_conc, bs = t, cf
            cands = [(t_fused, "fused"), (t_seq, "seq")] + ([(t_conc, "conc")] if nd else [])
            cands.sort()
            margin = cands[1][0] / cands[0][0] if len(cands) > 1 else 1.0
            print(f"{kv//128:>9} {nd:>6} {t_fused:>9.1f} {t_seq:>9.1f} "
                  f"{t_conc if nd else float('nan'):>9.1f} {str(bs):>7} "
                  f"{cands[0][1]:>7} {margin:>6.2f}x", flush=True)
            out.append(dict(kv=kv, nd=nd, fused=t_fused, seq=t_seq,
                            conc=t_conc if nd else None, split=bs, win=cands[0][1]))
    uncapped()
    json.dump(out, open(f"boundary_{'x'.join(map(str,heads))}.json", "w"))


if __name__ == "__main__":
    main()
