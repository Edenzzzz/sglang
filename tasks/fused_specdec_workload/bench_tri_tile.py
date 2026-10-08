#!/usr/bin/env python3
"""Benchmark the persistent kernel on the fused 4-class workload.

Run this on both flashinfer branches to compare tile configurations:
  persistent_overlap_  -> CTA_TILE_Q_SIZES = {128, 16}      (2 classes)
  tri_tile_            -> CTA_TILE_Q_SIZES = {128, 64, 16}  (3 classes)

Same batches, same kernel API, so the difference is the tile set. Verify rows
(16 < packed_qo <= 64) are the class that changes treatment: rounded up to 128
under the 2-class build, given their own 64 tile under the 3-class build.
"""
import argparse, json, os, sys
import numpy as np, torch, flashinfer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "debug_kernel_server_perf"))
from flashinfer.testing.utils import bench_gpu_time

MAX_PAGES = 8_000_000
MAX_Q = 32_768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="tasks/fused_specdec_workload/sets/qwen_small_h800.jsonl")
    ap.add_argument("--heads", default="28,4,128")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tag", required=True, help="label for the tile config, e.g. 2class / 3class")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    H_QO, H_KV, D = (int(x) for x in a.heads.split(","))
    R = H_QO // H_KV

    kvbuf = torch.empty(MAX_PAGES, 2, 1, H_KV, D, device="cuda", dtype=torch.bfloat16).normal_()
    qbuf = torch.empty(MAX_Q, H_QO, D, device="cuda", dtype=torch.bfloat16).uniform_()
    idx = torch.arange(MAX_PAGES, dtype=torch.int32, device="cuda")
    bp = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        kv_layout="NHD", backend="fa2")
    pa = flashinfer.BatchAttention(kv_layout="NHD")

    batches = [json.loads(l) for l in open(a.set) if l.strip()][: a.n]
    res = []
    for i, bt in enumerate(batches):
        rows = [tuple(x) for c in ("large_prefill", "spec_verify", "mixed_decode", "spec_draft")
                for x in bt[c]]
        if not rows:
            continue
        ql = torch.tensor([q for q, _ in rows], dtype=torch.int32)
        kl = torch.tensor([max(kv, 1) for _, kv in rows], dtype=torch.int32)
        if int(kl.sum()) > MAX_PAGES - 256 or int(ql.sum()) > MAX_Q:
            continue
        qi = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(ql, 0).int()])
        ki = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(kl, 0).int()])
        nq = int(qi[-1]); q = qbuf[:nq]
        try:
            best = None
            for flip in (True, False):
                pa.plan(qi, ki, idx, kl, H_QO, H_KV, D, D, 1, causal=False,
                        q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                        flipped_schedule=flip)
                o = torch.empty(nq, H_QO, D, device="cuda", dtype=torch.bfloat16)
                l = torch.empty(nq, H_QO, device="cuda", dtype=torch.float32)
                t = float(np.median(bench_gpu_time(lambda: pa.run(q, kvbuf, out=o, lse=l)))) * 1e3
                best = t if best is None else min(best, t)
            bp.plan(qi, ki, idx, torch.ones_like(kl), H_QO, H_KV, D, 1, causal=False,
                    q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, non_blocking=True)
            t2 = float(np.median(bench_gpu_time(lambda: bp.run(q, kvbuf)))) * 1e3
        except Exception as e:
            print(f"  [{i}] skip: {str(e)[:80]}", flush=True)
            continue
        nv = len(bt["spec_verify"])
        res.append(dict(i=i, pers=best, fa2=t2, n_verify=nv, n_rows=len(rows)))
        print(f"  [{i:3d}] rows={len(rows):3d} verify={nv:3d}  pers {best:8.1f} us  fa2 {t2:8.1f} us  "
              f"pers/fa2 {t2/best:5.2f}x", flush=True)

    out = a.out or f"tasks/fused_specdec_workload/bench_{a.tag}.json"
    json.dump(res, open(out, "w"))
    p = np.array([r["pers"] for r in res]); f = np.array([r["fa2"] for r in res])
    print(f"\n[{a.tag}] {len(res)} batches")
    print(f"  persistent total {p.sum():10.1f} us   BatchPrefill total {f.sum():10.1f} us"
          f"   ratio {f.sum()/p.sum():.3f}x")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
