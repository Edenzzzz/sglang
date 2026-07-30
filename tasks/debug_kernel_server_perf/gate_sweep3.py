"""3-way sweep: BatchPrefill vs persistent(flip) vs persistent(noflip)
for every mixed-batch composition, serving-matched (non-causal prefix part).
"""

import itertools
import json
import sys

import numpy as np
import torch

from flashinfer.testing.utils import bench_gpu_time

from gate_sweep import Bench


def all_cases():
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
    for d_bs in [2, 4, 8, 16, 26, 39]:
        for p_qo in [128, 512, 1024, 2048, 4096, 7168]:
            if d_bs * 81920 + 81920 > 7_900_000:
                continue
            cases.append((d_bs, 81920, [p_qo], [81920]))
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
    return cases


def main():
    import os
    heads = tuple(
        int(x) for x in os.environ.get("HEADS", "28,4,128").split(",")
    )
    b = Bench(heads)
    cases = all_cases()
    print(f"{len(cases)} cases")
    results = []
    for i, (d_bs, d_kv, p_lens, p_prefix) in enumerate(cases):
        d_qo = [1] * d_bs
        d_kvl = [d_kv - 1] * d_bs
        q_lens = torch.tensor(d_qo + p_lens, dtype=torch.int32)
        kv_lens = torch.tensor(d_kvl + p_prefix, dtype=torch.int32)
        q_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0).int()])
        kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(kv_lens, 0).int()])
        nq = int(q_indptr[-1])
        last_page_len = torch.ones_like(kv_lens)
        q = b.qbuf[:nq]
        try:
            b.bp.plan(q_indptr, kv_indptr, b.kv_indices, last_page_len,
                      *heads[:2], heads[2], 1, causal=False,
                      q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                      non_blocking=True)
            torch.cuda.synchronize()
            bp_ms = float(np.median(bench_gpu_time(lambda: b.bp.run(q, b.kv))))
            pa_ms = {}
            for flip in (True, False):
                b.pa.plan(q_indptr, kv_indptr, b.kv_indices, kv_lens,
                          heads[0], heads[1], heads[2], heads[2], 1, causal=False,
                          q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                          flipped_schedule=flip)
                o, l = b.out[:nq], b.lse[:nq]
                pa_ms[flip] = float(
                    np.median(bench_gpu_time(lambda: b.pa.run(q, b.kv, out=o, lse=l)))
                )
            r = {"d_bs": d_bs, "d_kv": d_kv, "p_lens": p_lens, "p_prefix": p_prefix,
                 "bp_ms": bp_ms, "pa_flip_ms": pa_ms[True], "pa_noflip_ms": pa_ms[False]}
            results.append(r)
            best = min(pa_ms.values())
            print(f"[{i+1}/{len(cases)}] {d_bs}d@{d_kv} p{p_lens}+{p_prefix} "
                  f"bp={bp_ms:.3f} flip={pa_ms[True]:.3f} noflip={pa_ms[False]:.3f} "
                  f"best_spd={bp_ms/best:.3f}", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(cases)}] ERROR {str(e)[:120]}", flush=True)
    json.dump(results, open(sys.argv[1] if len(sys.argv) > 1 else "gate_sweep3.json", "w"))


if __name__ == "__main__":
    main()
