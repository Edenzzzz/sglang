"""Time a *complete* mixed forward step the way the server performs it, for
both kernels: plan() + ragged self-attention + paged prefix attention + merge.

The earlier probes timed run() alone with the plan hoisted out of the loop,
which flatters whichever kernel has the more expensive planner and ignores
per-step CPU cost. The server calls plan() once per forward pass, so it must
be inside the measurement.
"""
import argparse
import time

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

H_QO, H_KV, D = 28, 4, 128
DEV = "cuda"


def build(dec_kvs, chunk_q, chunk_prefix):
    """Server mixed-batch layout: decode rows (qo=1) then one prefill chunk."""
    q_lens = [1] * len(dec_kvs) + [chunk_q]
    kv_lens = list(dec_kvs) + [chunk_prefix]
    q_indptr = torch.tensor(np.concatenate([[0], np.cumsum(q_lens)]), dtype=torch.int32)
    kv_indptr = torch.tensor(np.concatenate([[0], np.cumsum(kv_lens)]), dtype=torch.int32)
    return q_indptr, kv_indptr, torch.tensor(kv_lens, dtype=torch.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-long", type=int, default=12)
    ap.add_argument("--ctx-long", type=int, default=40960)
    ap.add_argument("--n-short", type=int, default=30)
    ap.add_argument("--ctx-short", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--prefix", type=int, default=0)
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()

    dec = [a.ctx_long - 1] * a.n_long + [a.ctx_short - 1] * a.n_short
    q_indptr, kv_indptr, kv_lens = build(dec, a.chunk, a.prefix)
    n_tok = int(q_indptr[-1])
    n_pages = int(kv_indptr[-1])
    print(f"batch: {a.n_long}d@{a.ctx_long//1024}K + {a.n_short}d@{a.ctx_short//1024}K "
          f"+ {a.chunk}q@prefix{a.prefix} | {n_tok} q-tokens, {n_pages/1e6:.2f}M kv pages")

    kv = torch.empty(n_pages, 2, 1, H_KV, D, device=DEV, dtype=torch.bfloat16).normal_()
    q = torch.empty(n_tok, H_QO, D, device=DEV, dtype=torch.bfloat16).uniform_()
    kv_indices = torch.arange(n_pages, dtype=torch.int32, device=DEV)
    last = torch.ones_like(kv_lens)
    out = torch.empty_like(q)
    lse = torch.empty(n_tok, H_QO, device=DEV, dtype=torch.float32)

    ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    bp = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD", backend="fa2")
    pa = flashinfer.BatchAttention(kv_layout="NHD")

    def bp_step():
        bp.plan(q_indptr, kv_indptr, kv_indices, last, H_QO, H_KV, D, 1, causal=False,
                q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, non_blocking=True)
        bp.run(q, kv, out=out)

    def pa_step(flip):
        def f():
            pa.plan(q_indptr, kv_indptr, kv_indices, kv_lens, H_QO, H_KV, D, D, 1,
                    causal=False, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                    flipped_schedule=flip)
            pa.run(q, kv, out=out, lse=lse)
        return f

    res = {}
    for name, fn in (("BatchPrefill", bp_step),
                     ("persistent(flip)", pa_step(True)),
                     ("persistent(noflip)", pa_step(False))):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        # wall clock including plan() CPU work, as the server experiences it
        t0 = time.perf_counter()
        for _ in range(a.iters):
            fn()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / a.iters * 1e3
        # CPU-only portion (no trailing sync): how much run-ahead it consumes
        t0 = time.perf_counter()
        for _ in range(a.iters):
            fn()
        cpu = (time.perf_counter() - t0) / a.iters * 1e3
        torch.cuda.synchronize()
        gpu = float(np.median(bench_gpu_time(fn)))
        res[name] = (wall, cpu, gpu)
        print(f"  {name:20} wall {wall:7.3f} ms | cpu-issue {cpu:7.3f} ms | gpu {gpu:7.3f} ms")

    b = res["BatchPrefill"][0]
    best = min(res["persistent(flip)"][0], res["persistent(noflip)"][0])
    print(f"  -> full-step speedup (wall): {b/best:.2f}x")


if __name__ == "__main__":
    main()
