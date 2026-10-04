#!/usr/bin/env python3
"""Where is the FA3-vs-persistent boundary, in the units the FA3 plan uses?

The dispatch gate routes a row to the FA3 role on packed_qo = qo_len*gqa >= 48
and kv_tiles >= 32. Those are PACKED units, but flashinfer's SM90 prefill plan
tiles on qo_len alone and emits one work item per qo head:

    int num_qo_tiles = ceil_div(qo_len, cta_tile_q);     // scheduler.cuh:939
    // NOTE(Zihao): our current FA3 implementation do not fuse query and group
    // heads so the group_size in cost_function is always 1

while the persistent plan tiles on qo_len*gqa_group_size (scheduler.cuh:1199).
So the two roles do not measure q the same way, and a gate written in packed
units can hand FA3 a row that fills 7 of 128 tile rows per head.

This sweeps both roles over (qo_len x kv_len) at a fixed row count and prints
the ratio, which is the boundary the gate should encode. Two row counts are
swept because the roles differ in how parallel width is recovered: the
persistent kernel splits KV, FA3 multiplies work by heads.

Run with the serving model's head shape, e.g. --heads 28,4,128 for gqa 7.
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_4mode import Bench, uncapped, med

QO = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512]
KV = [512, 2048, 4096, 8192, 16384, 32768]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="28,4,128", help="H_QO,H_KV,D")
    ap.add_argument("--rows", default="8,32", help="row counts to sweep")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    heads = tuple(int(x) for x in a.heads.split(","))
    gqa = heads[0] // heads[1]
    b = Bench(heads)
    uncapped()
    print(f"heads {heads}  gqa {gqa}   (fa3 tiles on qo_len; persistent tiles on qo_len*{gqa})")
    out = []
    for nrows in (int(x) for x in a.rows.split(",")):
        print(f"\n--- {nrows} identical rows "
              f"{'':>6}{'packed':>8}{'fa3 us':>9}{'pers us':>9}{'pers/fa3':>10}{'winner':>9}")
        print(f"{'qo_len':>7}{'kv_len':>8}{'kv_tiles':>9}{'packed_qo':>10}"
              f"{'fa3 us':>9}{'pers us':>9}{'pers/fa3':>10}{'winner':>9}{'gate':>7}")
        for qo in QO:
            for kv in KV:
                rows = [(qo, kv)] * nrows
                if not b.fits(rows):
                    continue
                q = b.plan_fa3(rows)
                t_fa3 = med(lambda: b.fa3.run(q, b.kv[: sum(max(k, 1) for _, k in rows)]))
                qp, o, lse = b.plan_pa(rows, False)
                t_pa = med(lambda: b.pa.run(qp, b.kv[: sum(max(k, 1) for _, k in rows)],
                                            out=o, lse=lse))
                kvt = -(-kv // 128)
                pq = qo * gqa
                gate = "FA3" if (pq >= 48 and kvt >= 32) else "pers"
                win = "FA3" if t_pa > t_fa3 else "pers"
                print(f"{qo:>7}{kv:>8}{kvt:>9}{pq:>10}{t_fa3:>9.1f}{t_pa:>9.1f}"
                      f"{t_pa / t_fa3:>10.3f}{win:>9}{gate:>7}"
                      + ("   <= gate disagrees" if win != gate else ""))
                out.append(dict(nrows=nrows, qo=qo, kv=kv, kv_tiles=kvt, packed_qo=pq,
                                fa3_us=t_fa3, pers_us=t_pa, ratio=t_pa / t_fa3,
                                winner=win, gate=gate))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
