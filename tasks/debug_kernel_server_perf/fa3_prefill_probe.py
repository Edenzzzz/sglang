#!/usr/bin/env python3
"""Is FA3 faster on PREFILL-dominant batches?

The earlier FA3 comparison used the captured co-tenancy compositions, which are
decode-heavy (46 decode rows + one small chunk). There FA3's fixed
cta_tile_q=64 is a liability, because a GQA-7 decode row carries only 7 packed
q-rows. This probe covers the opposite end: pure prefill and prefill-dominant
batches, where a large q-tile should pay off.

Note the persistent kernel uses tile **128** for prefill rows (packed_qo > 16
-> CTA_TILE_Q_SIZES[0]=128, scheduler.cuh:1141); its 16 tile only ever holds
decode/draft rows. So FA3's 64 is the *smaller* prefill tile here, not larger.
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_sweep import Bench
import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def bench(b, bp3, heads, q_lens, kv_lens):
    qs = torch.tensor(q_lens, dtype=torch.int32); ks = torch.tensor(kv_lens, dtype=torch.int32)
    qi = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(qs, 0).int()])
    ki = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(ks, 0).int()])
    nq = int(qi[-1]); last = torch.ones_like(ks); q = b.qbuf[:nq]
    out = {}
    b.bp.plan(qi, ki, b.kv_indices, last, *heads[:2], heads[2], 1, causal=False,
              q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, non_blocking=True)
    out["fa2"] = float(np.median(bench_gpu_time(lambda: b.bp.run(q, b.kv)))) * 1e3
    try:
        bp3.plan(qi, ki, b.kv_indices, last, *heads[:2], heads[2], 1, causal=False,
                 q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, non_blocking=True)
        out["fa3"] = float(np.median(bench_gpu_time(lambda: bp3.run(q, b.kv)))) * 1e3
    except Exception:
        out["fa3"] = float("nan")
    best = None
    for f in (True, False):
        b.pa.plan(qi, ki, b.kv_indices, ks, heads[0], heads[1], heads[2], heads[2], 1,
                  causal=False, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                  flipped_schedule=f)
        o = torch.empty(nq, heads[0], heads[2], device="cuda", dtype=torch.bfloat16)
        l = torch.empty(nq, heads[0], device="cuda", dtype=torch.float32)
        t = float(np.median(bench_gpu_time(lambda: b.pa.run(q, b.kv, out=o, lse=l)))) * 1e3
        best = t if best is None else min(best, t)
    out["pers"] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="28,4,128")
    a = ap.parse_args()
    heads = tuple(int(x) for x in a.heads.split(","))
    b = Bench(heads)
    ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    bp3 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD", backend="fa3")
    hdr = f"{'composition':<38} {'FA2 us':>9} {'FA3 us':>9} {'pers us':>9} {'FA3/FA2':>8} {'pers/FA2':>9}"

    print("A) PURE PREFILL (no decode rows), chunk attending a cached prefix")
    print(hdr)
    for chunk, pfx in [(512,0),(2048,0),(8192,0),(2048,8192),(8192,8192),(8192,32768),(4096,65536)]:
        r = bench(b, bp3, heads, [chunk], [pfx])
        print(f"{'1 chunk %5d @ prefix %6d'%(chunk,pfx):<38} {r['fa2']:>9.1f} {r['fa3']:>9.1f} "
              f"{r['pers']:>9.1f} {r['fa2']/r['fa3']:>7.2f}x {r['fa2']/r['pers']:>8.2f}x")

    print("\nB) MULTI-CHUNK prefill batches (several requests prefilling at once)")
    print(hdr)
    for n, chunk, pfx in [(4,2048,0),(8,1024,0),(4,2048,8192),(8,2048,8192),(16,512,4096)]:
        r = bench(b, bp3, heads, [chunk]*n, [pfx]*n)
        print(f"{'%2d chunks %5d @ prefix %6d'%(n,chunk,pfx):<38} {r['fa2']:>9.1f} {r['fa3']:>9.1f} "
              f"{r['pers']:>9.1f} {r['fa2']/r['fa3']:>7.2f}x {r['fa2']/r['pers']:>8.2f}x")

    print("\nC) SWEEPING decode:prefill ratio (32 decode rows @8K, one chunk)")
    print(hdr)
    for nd, chunk in [(0,4096),(2,4096),(8,4096),(32,4096),(32,512),(32,8192)]:
        ql = [1]*nd + [chunk]; kl = [8192]*nd + [0]
        r = bench(b, bp3, heads, ql, kl)
        print(f"{'%2d decode + chunk %5d'%(nd,chunk):<38} {r['fa2']:>9.1f} {r['fa3']:>9.1f} "
              f"{r['pers']:>9.1f} {r['fa2']/r['fa3']:>7.2f}x {r['fa2']/r['pers']:>8.2f}x")


if __name__ == "__main__":
    main()
