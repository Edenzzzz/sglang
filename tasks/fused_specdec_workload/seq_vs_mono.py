#!/usr/bin/env python3
"""Two mainloops, each on the whole device, back to back -- against monolithic.

The fused kernel exists to avoid serialising two kernels. But with the measured
class boundary (chunks on wgmma, 1-token rows on FA2) the SERIAL pair may already
beat one monolithic persistent launch over the mixed batch, because each half
runs on the engine that suits it with the full device. If so, that serial number
is the real baseline the fused kernel has to beat, not the monolithic one -- and
the fused kernel's remaining job is overlap and launch overhead, not engine
choice.

Everything is inside the timed region, launches included.
"""
import sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_4mode import Bench, uncapped, med

HEADS = (28, 4, 128)
PREFILL = [(500, 3095)]
DECODE = [(1, 4357)] * 11
LAYERS, STEP_MS = 28, 18.96

b = Bench(HEADS)
uncapped()
allrows = PREFILL + DECODE
pages_all = sum(k for _, k in allrows)

# monolithic: one persistent launch over the whole mixed batch
qa, oa, la = b.plan_pa(allrows, False)
t_mono = med(lambda: b.pa.run(qa, b.kv[:pages_all], out=oa, lse=la))

# serial pair: wgmma over the chunk rows, then persistent over the decode rows
qf = b.plan_fa3(PREFILL)
pages_p = sum(k for _, k in PREFILL)
t_fa3 = med(lambda: b.fa3.run(qf, b.kv[:pages_p]))
qd, od, ld = b.plan_pa(DECODE, False)
pages_d = sum(k for _, k in DECODE)
t_dec = med(lambda: b.pa.run(qd, b.kv[:pages_d], out=od, lse=ld))

# the pair must be timed as one region: plans are reused, only launches repeat
def pair():
    b.fa3.run(qf, b.kv[:pages_p])
    b.pa.run(qd, b.kv[:pages_d], out=od, lse=ld)

t_pair = med(pair)

def e2e(attn_us):
    attn_ms = attn_us * LAYERS / 1e3
    base = t_mono * LAYERS / 1e3
    step = STEP_MS - base + attn_ms
    return step, (STEP_MS / step - 1) * 100

print(f"{'variant':<34}{'us/layer':>10}{'vs mono':>9}{'step ms':>10}{'e2e tput':>10}")
for name, t in (("monolithic persistent", t_mono),
                ("wgmma chunk alone", t_fa3),
                ("persistent decode rows alone", t_dec),
                ("serial pair (both full device)", t_pair)):
    step, gain = e2e(t)
    tag = f"{t_mono/t:>8.2f}x"
    print(f"{name:<34}{t:>10.1f}{tag:>9}{step:>10.2f}{gain:>9.1f}%")
print(f"\nsum of parts {t_fa3 + t_dec:.1f} us vs measured pair {t_pair:.1f} us "
      f"-> launch/serialisation cost {t_pair - t_fa3 - t_dec:+.1f} us")
