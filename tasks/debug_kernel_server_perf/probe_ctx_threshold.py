"""Find the per-request decode context length at which the persistent kernel
starts to win, holding AGGREGATE decode KV roughly constant.

At short per-row KV the padded-tile attention is DRAM-bandwidth bound, so
BatchPrefill's row padding costs nothing. At long per-row KV it becomes
compute bound and the padding dominates. This locates that transition, which
decides which real workloads qualify.
"""
import json
import numpy as np, torch
from flashinfer.testing.utils import bench_gpu_time
from gate_sweep import Bench

H = (28, 4, 128)
b = Bench(H)
TARGET_AGG = 512 * 1024  # ~512K aggregate decode KV tokens


def run(rows, q, pfx=0):
    allrows = rows + [(q, pfx)]
    q_lens = torch.tensor([r[0] for r in allrows], dtype=torch.int32)
    kv_lens = torch.tensor([r[1] for r in allrows], dtype=torch.int32)
    qi = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0).int()])
    ki = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(kv_lens, 0).int()])
    nq = int(qi[-1])
    qt = b.qbuf[:nq]
    last = torch.ones_like(kv_lens)
    b.bp.plan(qi, ki, b.kv_indices, last, H[0], H[1], H[2], 1, causal=False,
              q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, non_blocking=True)
    torch.cuda.synchronize()
    bp = float(np.median(bench_gpu_time(lambda: b.bp.run(qt, b.kv))))
    pa = {}
    for flip in (True, False):
        b.pa.plan(qi, ki, b.kv_indices, kv_lens, H[0], H[1], H[2], H[2], 1, causal=False,
                  q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, flipped_schedule=flip)
        o, l = b.out[:nq], b.lse[:nq]
        pa[flip] = float(np.median(bench_gpu_time(lambda: b.pa.run(qt, b.kv, out=o, lse=l))))
    return bp, min(pa.values())


res = []
print("Aggregate decode KV held ~512K; per-request context varied:")
print(f"{'ctx/req':>9} {'n_dec':>6} {'chunk':>6} {'BP ms':>8} {'PA ms':>8} {'spd':>7}")
for ctx in [4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536]:
    n = max(2, TARGET_AGG // ctx)
    for q in [1024, 4096]:
        rows = [(1, ctx - 1)] * n
        bp, pa = run(rows, q)
        res.append(dict(ctx=ctx, n=n, q=q, bp_ms=bp, pa_ms=pa, spd=bp / pa))
        print(f"{ctx:>9} {n:>6} {q:>6} {bp:>8.3f} {pa:>8.3f} {bp/pa:>6.2f}x", flush=True)

print("\nSame, but aggregate KV grows with context (fixed 16 decodes):")
for ctx in [4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536]:
    for q in [1024, 4096]:
        rows = [(1, ctx - 1)] * 16
        bp, pa = run(rows, q)
        res.append(dict(ctx=ctx, n=16, q=q, bp_ms=bp, pa_ms=pa, spd=bp / pa, fixed_n=True))
        print(f"{ctx:>9} {16:>6} {q:>6} {bp:>8.3f} {pa:>8.3f} {bp/pa:>6.2f}x", flush=True)

json.dump(res, open("probe_ctx_threshold.json", "w"))
