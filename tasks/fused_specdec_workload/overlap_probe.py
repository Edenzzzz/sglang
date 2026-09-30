#!/usr/bin/env python3
"""Does co-launching FA3 beside the cooperative persistent kernel actually overlap?

Measures each side alone at its split grid, then the two co-launched, and reports
  overlap = (t_fa3_alone + t_pers_alone) / t_colaunch
1.0 means no overlap at all (pure serialisation); 2.0 means perfect overlap.
The persistent kernel is a COOPERATIVE launch, so every block must be resident
simultaneously -- this is the experiment that says whether that forces
serialisation against a co-resident kernel.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_4mode import Bench, uncapped, med
from four_mode_dispatch import Dispatcher

def main():
    heads = (8, 1, 128)
    b = Bench(heads); d = Dispatcher(b, heads[0], heads[1])
    big = [(1024, 16384)]
    dec = [(1, 11000)] * 115
    print(f"{'c_fa3/c_pers':>13} {'fa3 alone':>10} {'pers alone':>11} {'sum':>8} "
          f"{'colaunch':>9} {'overlap':>8}")
    for c_f in (16, 32, 48, 64, 96, 124):
        c_p = 132 - c_f
        os.environ["FLASHINFER_FA3_MAX_CTAS"] = str(c_f)
        qf = b.plan_fa3(big); t_f = med(lambda: b.fa3.run(qf, b.kv))
        os.environ["FLASHINFER_PERSISTENT_MAX_CLUSTERS"] = str(c_p)
        qp, op, lp = b.plan_pa(dec, False)
        t_p = med(lambda: b.pa.run(qp, b.kv, out=op, lse=lp))
        d.plan(big, dec, c_fa3=c_f, c_pers=c_p, mode="conc")
        t_c = med(d.run)
        print(f"{f'{c_f}/{c_p}':>13} {t_f:>10.1f} {t_p:>11.1f} {t_f+t_p:>8.1f} "
              f"{t_c:>9.1f} {(t_f+t_p)/t_c:>8.2f}x", flush=True)
    uncapped()

if __name__ == "__main__":
    main()
