"""What the FA2 / FA3 plan of each trace-sampled capture looks like, and how balanced the FA2 launch is (NP executor,
UNIFIED_TIMELINE=1 build): batch, rows per class, KV lengths, FA2 chunk size, split groups, items, CTAs, and from the
logged run the FA2 compute span vs its lower bounds
  lb_work  sum of FA2 compute-item durations / (SMs x co-resident CTAs): perfect balance
  lb_item  longest single item: what no schedule can beat without a smaller chunk
  imbal    span / max(lb_work, lb_item)
UNIFIED_MERGE_MODE / UNIFIED_DSMEM_CLUSTER select the plan, as in the benchmark.
  python plan_balance_20261025.py <captures dir> [...]
Claude, 2026-10-25.
"""
import os
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/dev/shm/w/ub")
import bench_captured_mixed_20261023 as B  # noqa: E402
import unified_executor_20261024 as U  # noqa: E402
torch.ops.load_library(os.environ["UNIFIED_SO"])
T = torch.ops.unified_dispatch
timeline = os.environ.get("PLAN_BALANCE_TIMELINE", "1") == "1"
sms = torch.cuda.get_device_properties(0).multi_processor_count
per_sm = int(os.environ.get("UNIFIED_FA2_CTAS_PER_SM", "2"))
agg = {"span": 0.0, "lb": 0.0, "steps": 0, "split_steps": 0}
for d in sys.argv[1:]:
    for path in sorted(Path(d).glob("*.pt")):
        case = B.load_case(path)
        g = case.geometry
        ql = g["query_lengths"]
        kv = [int(x) for x in case.cache_lengths.tolist()]
        ex = U.UnifiedExecutor(g["heads"], g["kv_heads"], torch.device("cuda"),
                               fa3_min_packed=int(os.environ.get("UNIFIED_FA3_MIN", "33")))
        plan = ex.plan(ql, kv, case.page_table)
        s = plan.stats
        G = g["heads"] // g["kv_heads"]
        fa2 = [i for i, n in enumerate(ql) if n * G < ex.fa3_min_packed]
        fa2kv = sorted(kv[i] for i in fa2)
        fa2_items = s["items32"] + s["items16"] + s["items48"]
        line = (f"PLAN {path.parent.parent.name}/{path.stem} bs={len(ql)} fa2_req={len(fa2)} fa3_req={len(ql) - len(fa2)} "
                f"fa2_kv[min/med/max]={fa2kv[0] if fa2kv else 0}/{fa2kv[len(fa2kv) // 2] if fa2kv else 0}/"
                f"{fa2kv[-1] if fa2kv else 0} fa2_chunk={s['fa2_chunk']} fa2_items={fa2_items} "
                f"(x{fa2_items / max(1, len(fa2) * g['kv_heads']):.2f} per req-head) split_groups={s['merge_groups']} "
                f"fa3_items={s['items_fa3']} fa3_splits={s['max_splits_fa3']}")
        agg["steps"] += 1
        agg["split_steps"] += int(s["merge_groups"] > 0)
        if timeline and fa2_items:
            out = torch.empty_like(case.output)
            for _ in range(3):
                plan.run(case.query, case.key, case.value, g["scale"], out)
            torch.cuda.synchronize()
            T.unified_timeline_reset()
            plan.run(case.query, case.key, case.value, g["scale"], out)
            torch.cuda.synchronize()
            r = T.unified_timeline_dump()
            code, ts, te = r[:, 0], r[:, 3], r[:, 5]
            comp = (code >> 28) <= 2
            dur = (te[comp] - ts[comp]).double() / 1e3
            span = (int(te[comp].max()) - int(ts[comp].min())) / 1e3
            lb_work = float(dur.sum()) / (sms * per_sm)
            lb_item = float(dur.max())
            lb = max(lb_work, lb_item)
            agg["span"] += span
            agg["lb"] += lb
            line += (f" | fa2_span={span:.1f}us lb_work={lb_work:.1f} lb_item={lb_item:.1f} "
                     f"item_med={statistics.median(dur.tolist()):.1f} imbal={span / max(lb, 1e-9):.2f}")
        print(line, flush=True)
        del case, ex, plan
        torch.cuda.empty_cache()
print(f"PLAN_TOTAL steps={agg['steps']} with_fa2_splits={agg['split_steps']} "
      f"fa2_span={agg['span'] / 1e3:.3f}ms lower_bound={agg['lb'] / 1e3:.3f}ms ratio={agg['span'] / max(agg['lb'], 1e-9):.3f}",
      flush=True)
