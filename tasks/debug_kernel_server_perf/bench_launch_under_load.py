"""Does the persistent kernel's cooperative launch cost more when the GPU is
already busy?

cudaLaunchCooperativeKernel requires every block to be co-resident, so the
launch cannot begin until enough SMs drain. On an idle GPU that is free; in a
saturated server with CUDA-graph decode replays queued ahead of it, it may
serialize. Every previous probe here ran one kernel on an idle device, which
cannot observe this.

Compares both kernels launched behind a variable amount of queued work.
"""
import argparse
import time

import numpy as np
import torch

import flashinfer

H_QO, H_KV, D = 28, 4, 128
DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-long", type=int, default=12)
    ap.add_argument("--ctx-long", type=int, default=40960)
    ap.add_argument("--n-short", type=int, default=35)
    ap.add_argument("--ctx-short", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=450)
    ap.add_argument("--iters", type=int, default=40)
    a = ap.parse_args()

    dec = [a.ctx_long - 1] * a.n_long + [a.ctx_short - 1] * a.n_short
    q_lens = [1] * len(dec) + [a.chunk]
    kv_lens_l = dec + [0]
    q_indptr = torch.tensor(np.concatenate([[0], np.cumsum(q_lens)]), dtype=torch.int32)
    kv_indptr = torch.tensor(np.concatenate([[0], np.cumsum(kv_lens_l)]), dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens_l, dtype=torch.int32)
    n_tok, n_pages = int(q_indptr[-1]), int(kv_indptr[-1])

    kv = torch.empty(n_pages, 2, 1, H_KV, D, device=DEV, dtype=torch.bfloat16).normal_()
    q = torch.empty(n_tok, H_QO, D, device=DEV, dtype=torch.bfloat16).uniform_()
    idx = torch.arange(n_pages, dtype=torch.int32, device=DEV)
    last = torch.ones_like(kv_lens)
    out = torch.empty_like(q)
    lse = torch.empty(n_tok, H_QO, device=DEV, dtype=torch.float32)

    ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    bp = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD", backend="fa2")
    pa = flashinfer.BatchAttention(kv_layout="NHD")
    bp.plan(q_indptr, kv_indptr, idx, last, H_QO, H_KV, D, 1, causal=False,
            q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, non_blocking=True)
    pa.plan(q_indptr, kv_indptr, idx, kv_lens, H_QO, H_KV, D, D, 1, causal=False,
            q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, flipped_schedule=False)
    torch.cuda.synchronize()

    # Background work standing in for queued decode-graph replays.
    M = 4096
    x = torch.randn(M, M, device=DEV, dtype=torch.bfloat16)
    y = torch.randn(M, M, device=DEV, dtype=torch.bfloat16)

    print(f"batch: {a.n_long}d@{a.ctx_long//1024}K + {a.n_short}d@{a.ctx_short//1024}K "
          f"+ {a.chunk}q fresh")
    print(f"{'queued matmuls':>15} {'BatchPrefill ms':>16} {'persistent ms':>15} {'speedup':>9}")
    for depth in (0, 4, 16, 48):
        res = {}
        for name, fn in (("bp", lambda: bp.run(q, kv, out=out)),
                         ("pa", lambda: pa.run(q, kv, out=out, lse=lse))):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            times = []
            for _ in range(a.iters):
                for _ in range(depth):
                    torch.mm(x, y)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            res[name] = float(np.median(times))
        print(f"{depth:>15} {res['bp']:>16.3f} {res['pa']:>15.3f} {res['bp']/res['pa']:>8.2f}x")


if __name__ == "__main__":
    main()
