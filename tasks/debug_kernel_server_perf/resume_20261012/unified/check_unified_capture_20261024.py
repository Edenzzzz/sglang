"""Correctness of the unified dispatcher on captured serving steps: output vs an FP32 reference
(sampled rows, every head) and vs sglang's FA3 call; repeated runs reuse one plan (counters must come
back to zero between layers).
  python check_unified_capture_20261024.py capture.pt [...]
Claude, 2026-10-24.
"""
import sys
import torch
sys.path.insert(0, "/dev/shm/w/ub")
import unified_executor_20261024 as U  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402,F401

torch.ops.load_library(__import__("os").environ.get("UNIFIED_SO", "/opt/tiger/w/torch_ext_unified_20261024/unified_dispatch_ext/unified_dispatch_ext.so"))
from sgl_kernel.flash_attn import flash_attn_with_kvcache  # noqa: E402


def fp32_reference(c, rows):
    g = c["geometry"]
    q, k, v = c["query"], c["key"][:, 0], c["value"][:, 0]
    G = g["heads"] // g["kv_heads"]
    out = {}
    cq = c["cumulative_query"].tolist()
    for b in rows:
        L = int(c["cache_lengths"][b])
        pages = c["page_table"][b, :L].long()
        kk = k[pages].float()  # [L, kvh, D]
        vv = v[pages].float()
        qs = q[cq[b]:cq[b + 1]].float()  # [n, H, D]
        n = qs.shape[0]
        kk = kk.repeat_interleave(G, dim=1)
        vv = vv.repeat_interleave(G, dim=1)
        s = torch.einsum("nhd,lhd->hnl", qs, kk) * g["scale"]
        pos_q = torch.arange(L - n, L, device=q.device)[:, None]
        mask = torch.arange(L, device=q.device)[None, :] <= pos_q
        s = s.masked_fill(~mask[None], float("-inf"))
        out[b] = torch.einsum("hnl,lhd->nhd", torch.softmax(s, -1), vv)
    return out


def keep_rows(c, lo, hi):
    # Keep only requests with lo <= q_len <= hi (class bisection): rebuild q, page table and lengths.
    g = dict(c["geometry"])
    ql = g["query_lengths"]
    cq = c["cumulative_query"].tolist()
    keep = [i for i, n in enumerate(ql) if lo <= n <= hi]
    if not keep:
        return None
    c2 = dict(c)
    c2["query"] = torch.cat([c["query"][cq[i]:cq[i + 1]] for i in keep]).contiguous()
    c2["page_table"] = c["page_table"][keep].contiguous()
    c2["cache_lengths"] = c["cache_lengths"][keep].contiguous()
    lens = [ql[i] for i in keep]
    c2["cumulative_query"] = torch.tensor([0, *lens], device=c["query"].device).cumsum(0).to(torch.int32)
    g["query_lengths"] = lens
    c2["geometry"] = g
    return c2


sys.path.insert(0, "/dev/shm/w/ub")
import bench_captured_mixed_20261023 as B  # noqa: E402


def load(path):
    # Captures keep ORIGINAL physical page ids and store only the live pages: rebuild full-size K/V
    # exactly as the stage-1 bench does (load_case) before any kernel sees them.
    from pathlib import Path
    case = B.load_case(Path(path))
    return {"geometry": case.geometry, "query": case.query, "key": case.key, "value": case.value,
            "page_table": case.page_table, "cache_lengths": case.cache_lengths,
            "cumulative_query": case.cumulative_query}


for path in sys.argv[1:]:
    c = load(path)
    import os
    if os.environ.get("UNIFIED_QRANGE"):
        lo, hi = (int(x) for x in os.environ["UNIFIED_QRANGE"].split(","))
        c = keep_rows(c, lo, hi)
        if c is None:
            print(path, "no rows in range"); continue
    g = c["geometry"]
    import os
    ex = U.UnifiedExecutor(g["heads"], g["kv_heads"], torch.device("cuda"), fa3_min_packed=int(os.environ.get("UNIFIED_FA3_MIN", "49")), fa2_ctas_per_sm=int(os.environ.get("UNIFIED_FA2_CPS", "2")))
    plan = ex.plan(g["query_lengths"], [int(x) for x in c["cache_lengths"].tolist()], c["page_table"])
    out = torch.zeros_like(c["query"])
    for _ in range(3):
        out.zero_()
        plan.run(c["query"], c["key"], c["value"], g["scale"], out)
    torch.cuda.synchronize()
    ref_fa3 = flash_attn_with_kvcache(q=c["query"], k_cache=c["key"], v_cache=c["value"], page_table=c["page_table"],
                                      cache_seqlens=c["cache_lengths"], cu_seqlens_q=c["cumulative_query"],
                                      max_seqlen_q=max(g["query_lengths"]), softmax_scale=g["scale"], causal=True)
    n = len(g["query_lengths"])
    sample = sorted({0, n - 1, n // 2, max(0, n - 2)})
    ref = fp32_reference(c, sample)
    cq = c["cumulative_query"].tolist()
    worst = 0.0
    for b, r in ref.items():
        o = out[cq[b]:cq[b + 1]].float()
        rel = ((o - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-6)).max().item()
        worst = max(worst, rel)
    fa3_rel = ((out.float() - ref_fa3.float()).norm(dim=-1) / ref_fa3.float().norm(dim=-1).clamp_min(1e-6)).max().item()
    finite = bool(torch.isfinite(out).all())
    st = plan.stats
    print(f"{path.split('/')[-3:]} rows={n} items48/32/16/fa3={st['items48']}/{st['items32']}/{st['items16']}/{st['items_fa3']} "
          f"groups={st['merge_groups']} fa3merge={st['fa3_merge_reqs']} fp32_worst={worst:.4g} vs_fa3_worst={fa3_rel:.4g} finite={finite} "
          f"{'PASS' if finite and worst < 0.02 else 'FAIL'}", flush=True)
