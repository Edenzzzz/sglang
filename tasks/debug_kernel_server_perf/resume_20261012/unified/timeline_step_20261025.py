"""Where the unified merge sits in the full A6 step (FA3 + FA2 launches, PDL): per-item timeline of the FA2 launch
(UNIFIED_TIMELINE=1 build, compute items, FA2 merge items, FA3 merge items) plus the stock FA3 grid's first-CTA start
and last-CTA end (the stock_fa3_pdl timeline slots), on whole trace-sampled captures. Per capture and summed:
  step        first event to last event
  fa3_end     end of the FA3 grid; fa2_cend = end of the last FA2 compute item
  exposed     merge time after max(fa3_end, fa2_cend): what any faster merge (JIT, DSMEM) could remove from the step
  fa2_m_late  FA2 merge items ending after fa3_end; fa3_m = span of the FA3 merge items (always after the FA3 grid)
  python timeline_step_20261025.py <captures dir> [...]
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

agg = {"step": 0.0, "exposed": 0.0, "after_fa3": 0.0, "fa3": 0.0}
for d in sys.argv[1:]:
    for path in sorted(Path(d).glob("*.pt")):
        case = B.load_case(path)
        g = case.geometry
        ex = U.UnifiedExecutor(g["heads"], g["kv_heads"], torch.device("cuda"),
                               fa3_min_packed=int(os.environ.get("UNIFIED_FA3_MIN", "33")))
        kv = [int(x) for x in case.cache_lengths.tolist()]
        plan = ex.plan(g["query_lengths"], kv, case.page_table)
        out = torch.empty_like(case.output)
        has_fa3 = plan.stock and plan.fa3_rows > 0
        for _ in range(5):
            plan.run(case.query, case.key, case.value, g["scale"], out)
        torch.cuda.synchronize()
        if has_fa3:
            plan.timeline = torch.tensor([(1 << 62), 0, (1 << 62), 0], dtype=torch.int64, device="cuda")  # slots 0 / 1: FA3 grid start (min) / end (max)
        T.unified_timeline_reset()
        plan.run(case.query, case.key, case.value, g["scale"], out)
        torch.cuda.synchronize()
        r = T.unified_timeline_dump()
        code, ts, te = r[:, 0], r[:, 3], r[:, 5]
        cls = code >> 28
        t0 = int(r[:, 2].min())
        fa3_s = fa3_e = None
        if has_fa3:
            fa3_s, fa3_e = int(plan.timeline[0]), int(plan.timeline[1])
            t0 = min(t0, fa3_s)
        comp = cls <= 2
        fa2_cend = int(te[comp].max()) if comp.any() else t0
        last = int(te.max())
        if fa3_e is not None:
            last = max(last, fa3_e)
        crit = max(fa2_cend, fa3_e if fa3_e is not None else 0)
        exposed = max(0, last - crit) / 1e3
        fa2m = cls == 3
        fa3m = cls == 4
        late = int((fa2m & (te > (fa3_e or 0))).sum()) if fa3_e is not None else int(fa2m.sum())
        fa3m_span = (int(te[fa3m].max()) - int(ts[fa3m].min())) / 1e3 if fa3m.any() else 0.0
        step = (last - t0) / 1e3
        after_fa3 = (last - fa3_e) / 1e3 if fa3_e is not None else 0.0
        agg["step"] += step
        agg["exposed"] += exposed
        agg["after_fa3"] += max(0.0, after_fa3)
        agg["fa3"] += (fa3_e - fa3_s) / 1e3 if fa3_e is not None else 0.0
        print(f"STEP {path.parent.parent.name}/{path.name} step={step:.1f}us "
              f"fa3={((fa3_e - fa3_s) / 1e3 if fa3_e else 0):.1f}us fa2_compute_end={(fa2_cend - t0) / 1e3:.1f}us "
              f"fa3_end={((fa3_e - t0) / 1e3 if fa3_e else 0):.1f}us after_fa3={after_fa3:.1f}us exposed_merge={exposed:.1f}us "
              f"fa2_merge_items={int(fa2m.sum())} (ending after FA3: {late}) fa3_merge_items={int(fa3m.sum())} "
              f"fa3_merge_span={fa3m_span:.1f}us", flush=True)
        del case, ex, plan, out
        torch.cuda.empty_cache()
print("STEP_TOTAL " + " ".join(f"{k}={v / 1e3:.3f}ms" for k, v in agg.items())
      + f"  exposed_merge_share={agg['exposed'] / max(agg['step'], 1e-9):.4f}", flush=True)
