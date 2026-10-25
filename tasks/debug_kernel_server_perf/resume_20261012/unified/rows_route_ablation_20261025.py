"""Which executor and tile should take a capture's decode / verify rows (q_len <= 4)? A6 ablation for rows of 33-64 packed
queries (GLM-4.5-Air's 4-token verify rows pack 48). The rows of each trace-sampled capture are run alone through the
unified executor with ROUTE_FA3_MIN as its FA3 threshold: 1 sends every row to FA3 (whose q tile is whatever the loaded
stock FA3 library was built with, 128 or 64), 1 << 30 keeps every row on FA2 (whose class-0 tile is the build's, 64 or
48). Graph-replayed time per capture, error against an FP32 reference on sampled rows.
  ROUTE_FA3_MIN=... UNIFIED_SO=... NATIVE_PDL_FA3_LIBRARY=... python rows_route_ablation_20261025.py <label> <dir> [...]
Claude, 2026-10-25.
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/dev/shm/w/ub")
import bench_captured_mixed_20261023 as B  # noqa: E402
import unified_executor_20261024 as U  # noqa: E402
import statistics  # noqa: E402

torch.ops.load_library(os.environ["UNIFIED_SO"])


def timed(fn, iters=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    s = []
    for _ in range(3):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            g.replay()
        b.record()
        b.synchronize()
        s.append(a.elapsed_time(b) / iters)
    return statistics.median(s)


def ref_rows(q, k, v, page_table, lens, qlens, cq, scale, G, sample):
    errs = {}
    for b in sample:
        L, n = int(lens[b]), qlens[b]
        pages = page_table[b, :L].long()
        kk = k[pages, 0].float().repeat_interleave(G, dim=1)
        vv = v[pages, 0].float().repeat_interleave(G, dim=1)
        qs = q[cq[b]:cq[b] + n].float()
        s = torch.einsum("nhd,lhd->hnl", qs, kk) * scale
        mask = torch.arange(L, device=q.device)[None, :] <= torch.arange(L - n, L, device=q.device)[:, None]
        s = s.masked_fill(~mask[None], float("-inf"))
        errs[b] = torch.einsum("hnl,lhd->nhd", torch.softmax(s, -1), vv)
    return errs


label = sys.argv[1]
fa3_min = int(os.environ["ROUTE_FA3_MIN"])
total, worst = 0.0, 0.0
for d in sys.argv[2:]:
    for path in sorted(Path(d).glob("*.pt")):
        case = B.load_case(path)
        g = case.geometry
        ql = g["query_lengths"]
        keep = [i for i, n in enumerate(ql) if n <= 4]
        if not keep:
            continue
        cq = case.cumulative_query.tolist()
        q = torch.cat([case.query[cq[i]:cq[i + 1]] for i in keep]).contiguous()
        pt = case.page_table[keep].contiguous()
        lens = case.cache_lengths[keep].contiguous()
        kvlen = [int(x) for x in lens.tolist()]
        qlens = [ql[i] for i in keep]
        qcum = [0]
        for n in qlens:
            qcum.append(qcum[-1] + n)
        out = torch.empty_like(q)
        ex = U.UnifiedExecutor(g["heads"], g["kv_heads"], torch.device("cuda"), fa3_min_packed=fa3_min)
        plan = ex.plan(qlens, kvlen, pt)
        ms = timed(lambda: plan.run(q, case.key, case.value, g["scale"], out))
        sample = sorted({0, len(keep) // 2, len(keep) - 1})
        refs = ref_rows(q, case.key, case.value, pt, lens, qlens, qcum, g["scale"], g["heads"] // g["kv_heads"], sample)
        err = max(((out[qcum[b]:qcum[b] + qlens[b]].float() - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-6)).max().item()
                  for b, r in refs.items())
        total += ms
        worst = max(worst, err)
        print(f"ROUTE {label} {path.parent.parent.name}/{path.name} rows={len(keep)} packed={max(qlens) * g['heads'] // g['kv_heads']} "
              f"kvmax={max(kvlen)} fa3_items={plan.stats['items_fa3']} fa2_items={plan.stats['items48'] + plan.stats['items32'] + plan.stats['items16']} "
              f"ms={ms:.4f} err={err:.3g}", flush=True)
        del case, ex, plan, out
        torch.cuda.empty_cache()
print(f"ROUTE_TOTAL {label} {total:.3f}ms worst_err={worst:.3g}", flush=True)
