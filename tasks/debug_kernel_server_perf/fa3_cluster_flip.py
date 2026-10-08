#!/usr/bin/env python3
"""Does FA3 have its own batch-average flip, like FA2's CTA-tile flip?

FA3's scheduler (scheduler.cuh ~:1468) sets cluster_size = 2 when the
batch-average packed qo length exceeds 64, doubling cluster_tile_q from 64 to
128 for *every* row -- including decode rows that carry only gqa_ratio packed
q-rows. That is the same "one chunk changes a batch-wide decision" shape as
FA2's tile flip, just at a different threshold.

Sweep the chunk size across the boundary with everything else fixed. A smooth
curve means no pathology; a step at the boundary means FA3 inherits one.
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_sweep import Bench
import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=46)
    ap.add_argument("--kv", type=int, default=3000)
    ap.add_argument("--heads", default="28,4,128")
    a = ap.parse_args()
    heads = tuple(int(x) for x in a.heads.split(","))
    gqa = heads[0] // heads[1]
    b = Bench(heads)
    ws3 = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    bp3 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws3, kv_layout="NHD", backend="fa3")
    bound = int(64 * (a.rows + 1) / gqa) - a.rows
    print(f"{a.rows} decode rows @{a.kv} KV, gqa={gqa}; cluster_size flips at chunk > {bound}")
    print(f"{'chunk':>7} {'avg_packed_qo':>14} {'cluster':>8} {'FA3 us':>9} {'FA2 us':>9} {'persistent us':>14}")
    for q in [bound - 40, bound - 20, bound - 8, bound - 2, bound + 2, bound + 8, bound + 20, bound + 40, bound + 120]:
        d_kvl = [a.kv] * a.rows
        q_lens = torch.tensor([1] * a.rows + [q], dtype=torch.int32)
        kv_lens = torch.tensor(d_kvl + [0], dtype=torch.int32)
        qi = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0).int()])
        ki = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(kv_lens, 0).int()])
        nq = int(qi[-1]); last = torch.ones_like(kv_lens); qq = b.qbuf[:nq]
        avg = (a.rows * gqa + q * gqa) // (a.rows + 1)
        cs = 2 if avg > 64 else 1
        bp3.plan(qi, ki, b.kv_indices, last, *heads[:2], heads[2], 1, causal=False,
                 q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, non_blocking=True)
        t3 = float(np.median(bench_gpu_time(lambda: bp3.run(qq, b.kv)))) * 1e3
        b.bp.plan(qi, ki, b.kv_indices, last, *heads[:2], heads[2], 1, causal=False,
                  q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, non_blocking=True)
        t2 = float(np.median(bench_gpu_time(lambda: b.bp.run(qq, b.kv)))) * 1e3
        b.pa.plan(qi, ki, b.kv_indices, kv_lens, heads[0], heads[1], heads[2], heads[2], 1,
                  causal=False, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                  flipped_schedule=False)
        o = torch.empty(nq, heads[0], heads[2], device="cuda", dtype=torch.bfloat16)
        l = torch.empty(nq, heads[0], device="cuda", dtype=torch.float32)
        tp = float(np.median(bench_gpu_time(lambda: b.pa.run(qq, b.kv, out=o, lse=l)))) * 1e3
        print(f"{q:>7} {avg:>14} {cs:>8} {t3:>9.1f} {t2:>9.1f} {tp:>14.1f}")


if __name__ == "__main__":
    main()
