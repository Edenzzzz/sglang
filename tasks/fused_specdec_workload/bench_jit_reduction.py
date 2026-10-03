#!/usr/bin/env python3
"""Does just-in-time reduction beat grid.sync + the single-shot merge pass?

Same build, two processes: FLASHINFER_JIT_REDUCTION=1 (producers merge their own
groups as they complete) vs =0 (classic grid-wide barrier then a cooperative
reduction pass). The env is read once per process, so the A/B is across runs.

Batches are MTP-shaped, which is what real serving looks like for us:
  - MTP ships ONE draft module, so a verify row is qo_len = k+1 = 2 and
    packed_qo_len = 2*gqa_ratio -- 14 at gqa 7, i.e. the TILE-16 class, not a
    large tile. Verify does NOT get its own tile on GQA models.
  - acceptance a=0.85 (DeepSeek-V3 reports 85-90% on the second token), so
    1.85 tokens/step and KV grows ~1.85 per sequence per step.
The split-KV population -- the only thing the reduction touches -- comes from
the long-context rows, so the sweep varies context length and running-set size.
"""
import argparse, json, os, time
import numpy as np, torch, flashinfer
from flashinfer.testing.utils import bench_gpu_time

MAXP = 6_000_000


def build(bs, ctx, chunk, gqa, k=1):
    """One fused serving step: a prefill chunk + verify rows + draft rows."""
    rows = []
    if chunk:
        rows.append((chunk, ctx))                 # large prefill
    rows += [(k + 1, ctx)] * bs                   # MTP verify, packed = (k+1)*gqa
    rows += [(1, ctx)] * bs                       # MTP draft step
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="28,4,128")
    ap.add_argument("--out", default="jit_reduction.json")
    a = ap.parse_args()
    HQ, HKV, D = (int(x) for x in a.heads.split(","))
    gqa = HQ // HKV
    kv = torch.empty(MAXP, 2, 1, HKV, D, device="cuda", dtype=torch.bfloat16).normal_()
    idx = torch.arange(MAXP, dtype=torch.int32, device="cuda")
    pa = flashinfer.BatchAttention(kv_layout="NHD")
    jit = os.environ.get("FLASHINFER_JIT_REDUCTION", "1") != "0"
    print(f"heads {(HQ,HKV,D)} gqa {gqa}  JIT reduction {'ON' if jit else 'OFF'}")
    print(f"{'bs':>5} {'ctx':>7} {'chunk':>6} {'rows':>5} {'us':>9}")
    res = []
    for bs, ctx, chunk in [(32, 8192, 0), (32, 32768, 0), (64, 8192, 0),
                           (64, 32768, 0), (32, 32768, 1024), (64, 32768, 1024),
                           (128, 16384, 0), (128, 16384, 1024)]:
        rows = build(bs, ctx, chunk, gqa)
        if sum(k for _, k in rows) > MAXP - 1024:
            continue
        ql = torch.tensor([q for q, _ in rows], dtype=torch.int32)
        kl = torch.tensor([k for _, k in rows], dtype=torch.int32)
        qi = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(ql, 0).int()])
        ki = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(kl, 0).int()])
        pa.plan(qi, ki, idx, kl, HQ, HKV, D, D, 1, causal=False,
                q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
        nq = int(qi[-1])
        q = torch.empty(nq, HQ, D, device="cuda", dtype=torch.bfloat16).uniform_()
        o = torch.empty(nq, HQ, D, device="cuda", dtype=torch.bfloat16)
        l = torch.empty(nq, HQ, device="cuda", dtype=torch.float32)
        t = float(np.median(bench_gpu_time(lambda: pa.run(q, kv, out=o, lse=l)))) * 1e3
        print(f"{bs:>5} {ctx:>7} {chunk:>6} {len(rows):>5} {t:>9.1f}", flush=True)
        res.append(dict(bs=bs, ctx=ctx, chunk=chunk, rows=len(rows), us=t,
                        jit=jit, checksum=float(o.float().abs().sum())))
    json.dump(res, open(a.out, "w"))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
