"""Per-kernel time of the FA2 executors on identical rows (A6 FA2 parity): v3's own FA2 row launch (BatchPrefill
kernel + its merge kernel) vs our persistent FA2 kernel, for every capture in the given sets. Torch profiler over
eager launches; prints per-kernel totals per arm. The rows and arms are built exactly as fa2_rows_compare_20261024.py
builds them.
  UNIFIED_SO=... python profile_fa2_rows_20261024.py <captures dir> [...]
Claude, 2026-10-24.
"""
import collections
import os
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/dev/shm/w/ub")
import bench_captured_mixed_20261023 as B  # noqa: E402
import unified_executor_20261024 as U  # noqa: E402
from native_fa3_fa2_pdl_20261023 import NativePdlExecutor  # noqa: E402
torch.ops.load_library(os.environ["UNIFIED_SO"])

ITERS = 20
totals = {"v3rows": collections.Counter(), "unified": collections.Counter()}
for d in sys.argv[1:]:
    for path in sorted(Path(d).glob(os.environ.get("PROF_GLOB", "*.pt"))):
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
        H, KH = g["heads"], g["kv_heads"]
        kvlen = [int(x) for x in lens.tolist()]
        out_u, out_v = torch.empty_like(q), torch.empty_like(q)
        ex = U.UnifiedExecutor(H, KH, torch.device("cuda"), fa3_min_packed=1 << 30)
        plan = ex.plan(qlens, kvlen, pt)
        v3 = NativePdlExecutor(H, KH, torch.device("cuda"), adaptive_splits=True, policy_version="geometry_v3",
                               fp32_partials=True, direct_bf16_output=True, packed_query_limit=64,
                               workspace_bytes=512 << 20)
        v3p = v3.plan(g["query_lengths"], g["prefix_lengths"], case.page_table, case.cache_lengths,
                      case.cumulative_query)
        sh = v3p.shape
        assert keep == list(range(sh.prefix_rows, len(ql)))
        w = v3.wrapper
        ns = "native_fa2_q32_fp32_direct" if sh.direct_row_output else v3.row_namespace
        v3_out = out_v if sh.direct_row_output else sh.float_row_output

        def fn_v():
            getattr(torch.ops, ns).run(q, case.key, case.value, w._qo_indptr_buf, w._paged_kv_indptr_buf,
                                       w._paged_kv_indices_buf, w._paged_kv_last_page_len_buf, v3_out, v3.empty_lse,
                                       w._float_workspace_buffer, w._int_workspace_buffer, list(w._plan_info),
                                       g["scale"], False, False, v3.empty_timeline, sh.native_tile,
                                       sh.maximum_row_query)

        def fn_u():
            plan.run(q, case.key, case.value, g["scale"], out_u)

        for name, fn in (("v3rows", fn_v), ("unified", fn_u)):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(ITERS):
                    fn()
                torch.cuda.synchronize()
            per = collections.Counter()
            for e in prof.events():
                if e.device_type == torch.autograd.DeviceType.CUDA:
                    per[e.name[:int(os.environ.get("PROF_NAME", "60"))]] += e.device_time_total / ITERS / 1000.0
            totals[name].update(per)
            print(f"PROF {path.parent.parent.name}/{path.name} {name} "
                  + " ".join(f"{k}={v:.4f}ms" for k, v in per.most_common(4)), flush=True)
        del case, ex, plan, v3, v3p, sh
        torch.cuda.empty_cache()
for name, c in totals.items():
    print(f"PROF_TOTAL {name} " + " | ".join(f"{k}={v:.3f}ms" for k, v in c.most_common(6)), flush=True)
