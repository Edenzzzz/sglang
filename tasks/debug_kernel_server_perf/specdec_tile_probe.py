#!/usr/bin/env python3
"""Cost of the verify tile class, and of fusing draft+verify+prefill.

Speculative decoding adds a work type whose packed_qo_len sits *between* decode
and prefill: a verify row carries (k+1) candidate tokens, so packed_qo_len =
(k+1)*gqa_ratio. FA2 picks tile 64 for it; our persistent kernel only has
{128, 16} and must round it up to 128. This measures what that rounding costs
and whether a fused batch beats separate launches at all.

sglang forces enable_mixed_chunk=False under every speculative algorithm, so
these fused compositions cannot occur in serving today -- they are the batches a
fused megakernel would create.
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_sweep import Bench
from flashinfer.testing.utils import bench_gpu_time


def run(b, heads, q_lens, kv_lens, flip=None):
    """q_lens/kv_lens are per-row; returns (fa2_ms, persistent_ms)."""
    qs = torch.tensor(q_lens, dtype=torch.int32)
    ks = torch.tensor(kv_lens, dtype=torch.int32)
    qi = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(qs, 0).int()])
    ki = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(ks, 0).int()])
    nq = int(qi[-1]); last = torch.ones_like(ks); q = b.qbuf[:nq]
    b.bp.plan(qi, ki, b.kv_indices, last, *heads[:2], heads[2], 1, causal=False,
              q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, non_blocking=True)
    fa2 = float(np.median(bench_gpu_time(lambda: b.bp.run(q, b.kv))))
    best = None
    for f in ((flip,) if flip is not None else (True, False)):
        b.pa.plan(qi, ki, b.kv_indices, ks, heads[0], heads[1], heads[2], heads[2], 1,
                  causal=False, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                  flipped_schedule=f)
        o = torch.empty(nq, heads[0], heads[2], device="cuda", dtype=torch.bfloat16)
        l = torch.empty(nq, heads[0], device="cuda", dtype=torch.float32)
        t = float(np.median(bench_gpu_time(lambda: b.pa.run(q, b.kv, out=o, lse=l))))
        best = t if best is None else min(best, t)
    return fa2 * 1e3, best * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="28,4,128")   # Qwen2.5-7B, gqa 7
    ap.add_argument("--kv", type=int, default=4096)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=400)
    a = ap.parse_args()
    heads = tuple(int(x) for x in a.heads.split(","))
    gqa = heads[0] // heads[1]
    b = Bench(heads)
    tile = lambda p: 128 if p > 64 else (64 if p > 16 else 16)
    print(f"heads {heads} gqa={gqa}, {a.bs} rows @{a.kv} KV, chunk {a.chunk}\n")

    print("A) verify-only batch: cost per candidate-token count (k+1)")
    print(f"   {'k+1':>4} {'packed':>7} {'FA2 tile':>9} {'ours':>5} {'FA2 us':>8} {'pers us':>8} {'pers/FA2':>9}")
    for kp1 in (1, 2, 4, 8, 16):
        f2, pa = run(b, heads, [kp1] * a.bs, [a.kv] * a.bs)
        p = kp1 * gqa
        print(f"   {kp1:>4} {p:>7} {tile(p):>9} {128 if p>16 else 16:>5} "
              f"{f2:>8.1f} {pa:>8.1f} {pa/f2:>8.2f}x")

    print("\nB) fused vs separate launches (draft rows + verify rows + one chunk)")
    print(f"   {'composition':<34} {'FA2 us':>8} {'pers us':>8} {'pers/FA2':>9}")
    nd, nv = a.bs * 3 // 4, a.bs // 4      # k=3 -> 3/4 drafting, 1/4 verifying
    parts = {
        "draft only (qo=1)":            ([1] * nd, [a.kv] * nd),
        "verify only (qo=4)":           ([4] * nv, [a.kv] * nv),
        "prefill chunk only":           ([a.chunk], [0]),
    }
    tot_f2 = tot_pa = 0.0
    for nm, (ql, kl) in parts.items():
        f2, pa = run(b, heads, ql, kl)
        tot_f2 += f2; tot_pa += pa
        print(f"   {nm:<34} {f2:>8.1f} {pa:>8.1f} {pa/f2:>8.2f}x")
    print(f"   {'--- sum of 3 separate launches':<34} {tot_f2:>8.1f} {tot_pa:>8.1f}")
    ql = [1] * nd + [4] * nv + [a.chunk]
    kl = [a.kv] * (nd + nv) + [0]
    f2, pa = run(b, heads, ql, kl)
    print(f"   {'FUSED draft+verify+prefill':<34} {f2:>8.1f} {pa:>8.1f} {pa/f2:>8.2f}x")
    print(f"\n   fusion gain: FA2 {tot_f2/f2:.2f}x, persistent {tot_pa/pa:.2f}x")
    print(f"   best fused ({min(f2,pa):.1f} us) vs best separate ({min(tot_f2,tot_pa):.1f} us): "
          f"{min(tot_f2,tot_pa)/min(f2,pa):.2f}x")


if __name__ == "__main__":
    main()
