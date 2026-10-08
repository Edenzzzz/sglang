"""Attribute the persistent kernel's win to prefill/decode OVERLAP, separated
from planner/split effects.

For each composition, measure with the SAME persistent kernel and planner:
  T_d   = decode rows alone
  T_p   = prefill chunk alone
  T_mix = both in one mixed launch
Overlap efficiency = (T_d + T_p) / T_mix. A serial-execution kernel scores
1.0; perfect overlap of a compute-bound chunk with memory-bound decodes
approaches (T_d+T_p)/max(T_d,T_p) -> 2.0 when balanced.

BatchPrefill mixed time is reported for reference, and its own sequential sum
as the fair no-overlap baseline (BP cannot overlap by construction: one tile
class per batch).
"""
import json
import numpy as np, torch, flashinfer
from flashinfer.testing.utils import bench_gpu_time

H_QO, H_KV, D = 28, 4, 128
DEV = "cuda"

kv = torch.empty(8_000_000, 2, 1, H_KV, D, device=DEV, dtype=torch.bfloat16).normal_()
qbuf = torch.empty(32768, H_QO, D, device=DEV, dtype=torch.bfloat16).uniform_()
idx = torch.arange(8_000_000, dtype=torch.int32, device=DEV)
ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEV)
bp = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD", backend="fa2")
pa = flashinfer.BatchAttention(kv_layout="NHD")
out = torch.empty_like(qbuf)
lse = torch.empty(32768, H_QO, device=DEV, dtype=torch.float32)


def t_pa(q_lens, kv_lens, causal=False):
    qi = torch.tensor(np.concatenate([[0], np.cumsum(q_lens)]), dtype=torch.int32)
    ki = torch.tensor(np.concatenate([[0], np.cumsum(kv_lens)]), dtype=torch.int32)
    kvl = torch.tensor(kv_lens, dtype=torch.int32)
    nq = int(qi[-1])
    best = None
    for flip in (True, False):
        pa.plan(qi, ki, idx, kvl, H_QO, H_KV, D, D, 1, causal=causal,
                q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, flipped_schedule=flip)
        o, l = out[:nq], lse[:nq]
        t = float(np.median(bench_gpu_time(lambda: pa.run(qbuf[:nq], kv, out=o, lse=l))))
        best = t if best is None else min(best, t)
    return best


def t_bp(q_lens, kv_lens, causal=False):
    qi = torch.tensor(np.concatenate([[0], np.cumsum(q_lens)]), dtype=torch.int32)
    ki = torch.tensor(np.concatenate([[0], np.cumsum(kv_lens)]), dtype=torch.int32)
    last = torch.ones(len(kv_lens), dtype=torch.int32)
    nq = int(qi[-1])
    bp.plan(qi, ki, idx, last, H_QO, H_KV, D, 1, causal=causal,
            q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, non_blocking=True)
    torch.cuda.synchronize()
    o = out[:nq]
    return float(np.median(bench_gpu_time(lambda: bp.run(qbuf[:nq], kv, out=o))))


print(f"{'composition':<34} {'T_d':>7} {'T_p':>7} {'PAmix':>7} {'ovl_eff':>8} "
      f"{'BPmix':>7} {'BPseq':>7} {'vs BPmix':>9}")
res = []
for n_d, d_kv in [(8, 65536), (16, 65536), (16, 32768), (32, 32768), (12, 40960)]:
    for q, pfx in [(2048, 8192), (4096, 8192), (4096, 16384), (8192, 16384),
                   (4096, 32768), (2048, 0), (8192, 32768)]:
        if n_d * d_kv + pfx > 7_900_000:
            continue
        dq = [1] * n_d
        dkv = [d_kv - 1] * n_d
        T_d = t_pa(dq, dkv)
        T_p = t_pa([q], [pfx])
        T_mix = t_pa(dq + [q], dkv + [pfx])
        B_mix = t_bp(dq + [q], dkv + [pfx])
        B_seq = t_bp(dq, dkv) + t_bp([q], [pfx])
        eff = (T_d + T_p) / T_mix
        res.append(dict(n_d=n_d, d_kv=d_kv, q=q, pfx=pfx, T_d=T_d, T_p=T_p,
                        T_mix=T_mix, B_mix=B_mix, B_seq=B_seq, eff=eff))
        c = f"{n_d}d@{d_kv//1024}K + {q}q@{pfx//1024}K"
        print(f"{c:<34} {T_d*1e3:7.0f} {T_p*1e3:7.0f} {T_mix*1e3:7.0f} {eff:7.2f}x "
              f"{B_mix*1e3:7.0f} {B_seq*1e3:7.0f} {B_mix/T_mix:8.2f}x", flush=True)
json.dump(res, open("overlap_attribution.json", "w"))
