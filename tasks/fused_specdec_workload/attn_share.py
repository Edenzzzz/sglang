#!/usr/bin/env python3
"""What fraction of a measured serving step is attention?

Every kernel-level win is multiplied by this number, so it bounds what any
attention change can do end to end. The batch is the median MIXED batch from the
clean per-step trace: one 500-token prefill chunk over a 2595-token prefix plus
11 decode rows averaging 4357 KV, 28 layers, Qwen2.5-7B head shape.
"""
import os, sys
sys.path.insert(0, "/dev/shm/w/mb")
import numpy as np, torch
from bench_4mode import Bench, uncapped, med

HEADS = (28, 4, 128)
LAYERS = 28
STEP_MS = 18.96          # measured p50 MIXED step, clean GPU, fused arm
rows_prefill = [(500, 3095)]
rows_decode = [(1, 4357)] * 11

b = Bench(HEADS)
uncapped()
for name, rows in (("prefill row only", rows_prefill),
                   ("decode rows only", rows_decode),
                   ("the whole mixed batch", rows_prefill + rows_decode)):
    npages = sum(max(k, 1) for _, k in rows)
    q, o, lse = b.plan_pa(rows, False)
    t = med(lambda: b.pa.run(q, b.kv[:npages], out=o, lse=lse))
    tot = t * LAYERS / 1e3
    print(f"{name:<24} {t:8.1f} us/layer  x{LAYERS} = {tot:7.3f} ms  "
          f"= {tot / STEP_MS * 100:5.2f}% of the {STEP_MS} ms step")
