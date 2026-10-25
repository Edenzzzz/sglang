"""Per-item timeline of the persistent overlap + LPT FA2 kernel (A6) on the FA2 rows of each trace-sampled capture.
Needs a build with UNIFIED_TIMELINE=1. For every capture: one graph-replayed warm run, then one logged run; prints
where the kernel's time goes:
  span        first pull to last item end (globaltimer)
  compute_end last compute item end; tail = span end - compute_end (merges left after compute drained)
  pull        mean / total time a CTA spends getting its next item (queue atomics, ready-queue polling)
  wait        tail-mode merge items: time spinning on their group's members
  merge       mean merge item duration excluding the wait
  idle        CTA-time with no item between the CTA's first pull and the span end
  python timeline_fa2_rows_20261025.py <out.pt> <captures dir> [...]
The merge mode comes from UNIFIED_MERGE_MODE (unset = merge items at the queue tail, ready = ready queue).
Claude, 2026-10-25.
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/dev/shm/w/ub")
import bench_captured_mixed_20261023 as B  # noqa: E402
import unified_executor_20261024 as U  # noqa: E402
torch.ops.load_library(os.environ["UNIFIED_SO"])
T = torch.ops.unified_dispatch

records = {}
agg = {"span": 0.0, "tail": 0.0, "pull": 0.0, "wait": 0.0, "merge": 0.0, "compute": 0.0, "idle": 0.0}
out_path = sys.argv[1]
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
        kvlen = [int(x) for x in case.cache_lengths[keep].tolist()]
        qlens = [ql[i] for i in keep]
        out = torch.empty_like(q)
        ex = U.UnifiedExecutor(g["heads"], g["kv_heads"], torch.device("cuda"), fa3_min_packed=1 << 30)
        plan = ex.plan(qlens, kvlen, pt)
        fn = lambda: plan.run(q, case.key, case.value, g["scale"], out)
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        for _ in range(5):
            gr.replay()
        torch.cuda.synchronize()
        T.unified_timeline_reset()
        gr.replay()
        torch.cuda.synchronize()
        r = T.unified_timeline_dump()
        code, cta, tp, ts, tm, te = (r[:, i] for i in range(6))
        cls = code >> 28
        t0 = int(tp.min())
        end = int(te.max())
        span = (end - t0) / 1e3
        comp = cls <= 2
        mrg = cls == 3
        compute_end = int(te[comp].max()) if comp.any() else t0
        tail = (end - compute_end) / 1e3
        pull = (ts - tp).double() / 1e3
        wait = (tm - ts).double()[mrg] / 1e3
        mdur = (te - tm).double()[mrg] / 1e3
        cdur = (te - ts).double()[comp] / 1e3
        blk = cta >> 16
        first = {}
        for b, a in zip(blk.tolist(), tp.tolist()):
            first[b] = min(first.get(b, a), a)
        busy = float((te - tp).sum()) / 1e3
        idle = sum((end - a) / 1e3 for a in first.values()) - busy
        nctas = len(first)
        records[f"{path.parent.parent.name}/{path.name}"] = r
        cspan = (compute_end - t0) / 1e3
        agg["cspan"] = agg.get("cspan", 0.0) + cspan
        print(f"TL {path.parent.parent.name}/{path.name} rows={len(keep)} items={int(comp.sum())}+{int(mrg.sum())}m "
              f"ctas={nctas} span={span:.1f}us compute_span={cspan:.1f}us tail={tail:.1f}us pull_mean={pull.mean():.2f}us "
              f"pull_cta_total={pull.sum() / max(nctas, 1):.1f}us wait_mean={wait.mean().item() if mrg.any() else 0:.2f}us "
              f"merge_mean={mdur.mean().item() if mrg.any() else 0:.2f}us compute_mean={cdur.mean():.2f}us "
              f"compute_max={cdur.max():.2f}us idle_per_cta={idle / max(nctas, 1):.1f}us", flush=True)
        agg["span"] += span
        agg["tail"] += tail
        agg["pull"] += float(pull.sum()) / max(nctas, 1)
        agg["wait"] += float(wait.sum()) / max(nctas, 1)
        agg["merge"] += float(mdur.sum()) / max(nctas, 1)
        agg["compute"] += float(cdur.sum()) / max(nctas, 1)
        agg["idle"] += idle / max(nctas, 1)
        del case, ex, plan, gr, out
        torch.cuda.empty_cache()
torch.save(records, out_path)
print("TL_TOTAL mode=%s " % os.environ.get("UNIFIED_MERGE_MODE", "tail")
      + " ".join(f"{k}={v / 1e3:.3f}ms" for k, v in agg.items())
      + "  (pull/wait/merge/compute/idle are per-CTA averages summed over captures)", flush=True)
