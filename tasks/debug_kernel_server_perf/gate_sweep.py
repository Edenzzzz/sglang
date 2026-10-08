"""Dense sweep of BatchPrefill vs persistent BatchAttention kernel time over
mixed-batch compositions, to fit the server-side kernel selection gate.

Matches the sglang serving composition exactly (use_ragged mixed path):
both kernels compute the CACHED-PREFIX part only — causal=False, each
request's paged KV is its prefix, decode requests have qo_len=1, prefill
chunks qo_len=e_i. The ragged self-attention part is identical for both
methods and excluded.

Uses the serving-matched flipped-schedule choice:
flipped=None (FlashInfer default True) if total_kv < bs*32768 else False.

KV buffer is allocated once at max size and reused across cases.
"""

import argparse
import itertools
import json
import os

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

MAX_PAGES = 8_000_000
MAX_Q = 32_768


class Bench:
    def __init__(self, heads):
        self.num_qo_heads, self.num_kv_heads, self.head_dim = heads
        device = "cuda"
        self.device = device
        self.kv = torch.empty(
            MAX_PAGES, 2, 1, self.num_kv_heads, self.head_dim,
            device=device, dtype=torch.bfloat16,
        ).normal_()
        self.qbuf = torch.empty(
            MAX_Q, self.num_qo_heads, self.head_dim, device=device, dtype=torch.bfloat16
        ).uniform_()
        self.kv_indices = torch.arange(MAX_PAGES, dtype=torch.int32, device=device)
        self.out = torch.empty_like(self.qbuf)
        self.lse = torch.empty(MAX_Q, self.num_qo_heads, device=device, dtype=torch.float32)
        # 128 MB is enough for <=32 qo heads; 64-head models (e.g. Qwen3-235B,
        # gqa 16) overflow batch_prefill_tmp_s during plan().
        ws_mb = int(os.environ.get("GATE_SWEEP_WS_MB", "128"))
        ws = torch.empty(ws_mb * 1024 * 1024, dtype=torch.uint8, device=device)
        self.bp = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD", backend="fa2")
        self.pa = flashinfer.BatchAttention(kv_layout="NHD")

    def profile_case(self, d_bs, d_kv, p_lens, p_prefix):
        # serving ragged-mixed composition: paged part only, causal=False,
        # kv = prefix for every request
        d_qo = [1] * d_bs
        d_kvl = [d_kv - 1] * d_bs  # decode prefix = seq_len - 1
        p_qo = list(p_lens)
        p_kvl = list(p_prefix)

        q_lens = torch.tensor(d_qo + p_qo, dtype=torch.int32)
        kv_lens = torch.tensor(d_kvl + p_kvl, dtype=torch.int32)
        q_indptr = torch.cat(
            [torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0).int()]
        )
        kv_indptr = torch.cat(
            [torch.zeros(1, dtype=torch.int32), torch.cumsum(kv_lens, 0).int()]
        )
        nb = int(kv_indptr[-1])
        nq = int(q_indptr[-1])
        assert nb <= MAX_PAGES and nq <= MAX_Q, (nb, nq)
        total_kv = nb
        bs = d_bs + len(p_qo)

        q = self.qbuf[:nq]
        last_page_len = torch.ones_like(kv_lens)

        self.bp.plan(
            q_indptr, kv_indptr, self.kv_indices, last_page_len,
            self.num_qo_heads, self.num_kv_heads, self.head_dim, 1, causal=False,
            q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, non_blocking=True,
        )
        torch.cuda.synchronize()
        bp_ms = float(np.median(bench_gpu_time(lambda: self.bp.run(q, self.kv))))

        flipped = None if total_kv < bs * 32768 else False
        self.pa.plan(
            q_indptr, kv_indptr, self.kv_indices, kv_lens,
            self.num_qo_heads, self.num_kv_heads, self.head_dim, self.head_dim, 1,
            causal=False, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
            flipped_schedule=flipped,
        )
        out, lse = self.out[:nq], self.lse[:nq]
        pa_ms = float(
            np.median(bench_gpu_time(lambda: self.pa.run(q, self.kv, out=out, lse=lse)))
        )

        o_bp = self.bp.run(q, self.kv)
        o_pa, _ = self.pa.run(q, self.kv, out=out, lse=lse)
        err = float((o_bp - o_pa).abs().max())

        return {
            "d_bs": d_bs, "d_kv": d_kv, "p_lens": p_qo, "p_prefix": p_kvl,
            "flipped": bool(flipped is None),
            "bp_ms": bp_ms, "pa_ms": pa_ms, "speedup": bp_ms / pa_ms, "err": err,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qo-heads", type=int, default=28)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--out", type=str, default="gate_sweep.json")
    args = ap.parse_args()
    heads = (args.qo_heads, args.kv_heads, args.head_dim)

    cases = []
    for d_bs, d_kv in itertools.product(
        [1, 2, 4, 8, 16, 32, 64, 128],
        [2048, 8192, 32768, 98304],
    ):
        if d_bs * d_kv > 7_000_000:
            continue
        for p_qo in [128, 512, 2048, 4096, 8192]:
            for prefix in {512, 4 * p_qo, 65536}:
                if (d_bs * d_kv + prefix) > 7_900_000:
                    continue
                cases.append((d_bs, d_kv, [p_qo], [prefix]))
    # TraceLab regime: long-context decode population + continued prefill
    for d_bs in [2, 4, 8, 16, 26, 39]:
        for p_qo in [128, 512, 1024, 2048, 4096, 7168]:
            if d_bs * 81920 + 81920 > 7_900_000:
                continue
            cases.append((d_bs, 81920, [p_qo], [81920]))
    # multi-prefill compositions
    cases += [
        (16, 8192, [2048, 2048], [512, 8192]),
        (64, 4096, [1024, 1024, 1024, 1024], [512, 2048, 4096, 8192]),
        (8, 65536, [1024, 4096], [65536, 16384]),
        (100, 2048, [512, 512], [2048, 512]),
        (32, 16384, [4096, 512], [32768, 512]),
        (100, 2048, [2048], [2048]),
        (100, 4096, [4096], [4096]),
        (100, 8192, [4096], [4096]),
        (100, 8192, [512], [8192]),
    ]

    b = Bench(heads)
    print(f"{len(cases)} cases")
    results = []
    for i, (d_bs, d_kv, p_lens, p_prefix) in enumerate(cases):
        try:
            r = b.profile_case(d_bs, d_kv, p_lens, p_prefix)
            results.append(r)
            print(
                f"[{i+1}/{len(cases)}] {d_bs}d@{d_kv} p{p_lens}+{p_prefix} "
                f"flip={r['flipped']} bp={r['bp_ms']:.3f} pa={r['pa_ms']:.3f} "
                f"spd={r['speedup']:.3f} err={r['err']:.4f}",
                flush=True,
            )
        except Exception as e:
            print(f"[{i+1}/{len(cases)}] {d_bs}d@{d_kv} p{p_lens} ERROR {str(e)[:150]}", flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
