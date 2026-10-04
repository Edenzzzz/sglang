#!/usr/bin/env python3
"""The modal prefill row, both mainloops, at the serving head shape.

93% of a mixed step's attention is this one row, so this is the number the
unified-plan refactor has to beat. FA3 here is as the server plans it today:
unpacked GQA (one work item per qo head), no KV split.
"""
import sys
sys.path.insert(0, "/dev/shm/w/mb")
import numpy as np, torch
from bench_4mode import Bench, uncapped, med

HEADS = (28, 4, 128)
b = Bench(HEADS)
uncapped()
print(f"{'row (qo x kv)':<18}{'persistent':>12}{'FA3':>10}{'pers/FA3':>10}{'TFLOP/s pers':>14}{'FA3':>8}")
for qo, kv in ((500, 3095), (500, 8192), (500, 32768), (128, 3095), (2048, 3095)):
    rows = [(qo, kv)]
    npages = kv
    q, o, lse = b.plan_pa(rows, False)
    t_pa = med(lambda: b.pa.run(q, b.kv[:npages], out=o, lse=lse))
    qf = b.plan_fa3(rows)
    t_fa3 = med(lambda: b.fa3.run(qf, b.kv[:npages]))
    fl = 2 * 2 * qo * kv * 128 * HEADS[0] / 1e12   # QK + PV, 2 flops per MAC
    print(f"{f'{qo} x {kv}':<18}{t_pa:>10.1f}us{t_fa3:>8.1f}us{t_pa/t_fa3:>10.3f}"
          f"{fl/(t_pa*1e-6):>14.0f}{fl/(t_fa3*1e-6):>8.0f}")
