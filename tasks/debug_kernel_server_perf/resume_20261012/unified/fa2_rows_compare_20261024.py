"""FA2 executor efficiency, isolated (A6): on the decode / verify rows of each trace-sampled capture (q_len <= 4,
the rows the unified plan sends to FA2), time three executors on identical tensors, graph replay:
  unified        our FA2-only kernel (dynamic LPT queue, in-epilogue JIT merge), no FA3 work in the plan
  batchattention stock flashinfer 0.7 BatchAttention (persistent LPT, its own plan and merge)
  batchprefill   stock flashinfer 0.7 BatchPrefillWithPagedKVCacheWrapper (non-persistent, no split)
  v3rows         native v3's own FA2 row launch, as stage 1 runs it: geometry split (fixed_split_size), Q32 tile,
                 FP32 partials, direct BF16 output, with its merge
Each output is checked against an FP32 reference on sampled rows.
  python fa2_rows_compare_20261024.py <captures dir> [...]
Claude, 2026-10-24.
"""
import os
import statistics
import sys
from pathlib import Path

import torch
import flashinfer

sys.path.insert(0, "/dev/shm/w/ub")
import bench_captured_mixed_20261023 as B  # noqa: E402
import unified_executor_20261024 as U  # noqa: E402
from native_fa3_fa2_pdl_20261023 import NativePdlExecutor  # noqa: E402
torch.ops.load_library(os.environ.get("UNIFIED_SO", "/opt/tiger/w/torch_ext_unified2_20261024/unified_dispatch_ext/unified_dispatch_ext.so"))


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


tot = {"unified": 0.0, "batchattention": 0.0, "batchprefill": 0.0, "v3rows": 0.0}
for d in sys.argv[1:]:
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
        qlens = [ql[i] for i in keep]
        qcum = [0]
        for n in qlens:
            qcum.append(qcum[-1] + n)
        H, KH = g["heads"], g["kv_heads"]
        kvlen = [int(x) for x in lens.tolist()]
        kv_indices = torch.cat([pt[i, :kvlen[i]] for i in range(len(keep))]).to(torch.int32)
        kv_indptr = torch.tensor([0] + list(torch.tensor(kvlen).cumsum(0).tolist()), dtype=torch.int32, device="cuda")
        qo_indptr = torch.tensor(qcum, dtype=torch.int32, device="cuda")
        out = {k: torch.empty_like(q) for k in tot}
        # unified (all rows <= 32 packed -> FA2 only)
        ex = U.UnifiedExecutor(H, KH, torch.device("cuda"), fa3_min_packed=1 << 30)
        plan = ex.plan(qlens, kvlen, pt)
        fn_u = lambda: plan.run(q, case.key, case.value, g["scale"], out["unified"])
        # BatchAttention
        ba = flashinfer.BatchAttention(kv_layout="NHD")
        ba.plan(qo_indptr, kv_indptr, kv_indices, lens.to(torch.int32), H, KH, 128, 128, 1, causal=True,
                sm_scale=g["scale"], q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
        k4, v4 = case.key, case.value  # [pages, 1, KH, 128]
        fn_a = lambda: ba.run(q, (k4, v4), out=out["batchattention"])
        # BatchPrefill
        ws = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
        bp = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
        bp.plan(qo_indptr, kv_indptr, kv_indices, torch.ones(len(keep), dtype=torch.int32, device="cuda"), H, KH, 128, 1,
                causal=True, sm_scale=g["scale"], q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
        fn_p = lambda: bp.run(q, (k4, v4), out=out["batchprefill"])
        v3 = NativePdlExecutor(H, KH, torch.device("cuda"), adaptive_splits=True, policy_version="geometry_v3",
                               fp32_partials=True, direct_bf16_output=True, packed_query_limit=64,
                               workspace_bytes=512 << 20)
        v3p = v3.plan(g["query_lengths"], g["prefix_lengths"], case.page_table, case.cache_lengths,
                      case.cumulative_query)
        sh = v3p.shape
        if keep != list(range(sh.prefix_rows, len(ql))):
            raise RuntimeError(f"v3 FA2 rows differ from the q_len <= 4 rows in {path}")
        w = v3.wrapper
        ns = "native_fa2_q32_fp32_direct" if sh.direct_row_output else v3.row_namespace
        v3_out = out["v3rows"] if sh.direct_row_output else sh.float_row_output

        def fn_v():
            getattr(torch.ops, ns).run(q, case.key, case.value, w._qo_indptr_buf, w._paged_kv_indptr_buf,
                                       w._paged_kv_indices_buf, w._paged_kv_last_page_len_buf, v3_out, v3.empty_lse,
                                       w._float_workspace_buffer, w._int_workspace_buffer, list(w._plan_info),
                                       g["scale"], False, False, v3.empty_timeline, sh.native_tile,
                                       sh.maximum_row_query)
            if v3_out is not out["v3rows"]:
                out["v3rows"].copy_(v3_out)
        t = {"unified": timed(fn_u), "batchattention": timed(fn_a), "batchprefill": timed(fn_p), "v3rows": timed(fn_v)}
        sample = sorted({0, len(keep) // 2, len(keep) - 1})
        refs = ref_rows(q, case.key, case.value, pt, lens, qlens, qcum, g["scale"], H // KH, sample)
        err = {}
        for k in tot:
            err[k] = max(((out[k][qcum[b]:qcum[b] + qlens[b]].float() - r).norm(dim=-1) /
                          r.norm(dim=-1).clamp_min(1e-6)).max().item() for b, r in refs.items())
        for k in tot:
            tot[k] += t[k]
        print(f"FA2ROWS {path.parent.parent.name}/{path.name} rows={len(keep)} kvmax={max(kvlen)} "
              f"v3split={sh.split_settings.get('row_split_size')} v3tile={sh.native_tile} "
              + " ".join(f"{k}={t[k]:.4f}ms(err {err[k]:.3g})" for k in tot), flush=True)
        del case, ex, plan, ba, bp, ws, out, v3, v3p, sh
        torch.cuda.empty_cache()
print("FA2ROWS_TOTAL " + " ".join(f"{k}={v:.3f}ms" for k, v in tot.items())
      + f" | speedup of unified over: batchattention {tot['batchattention']/tot['unified']:.3f}x,"
      f" v3rows {tot['v3rows']/tot['unified']:.3f}x", flush=True)
