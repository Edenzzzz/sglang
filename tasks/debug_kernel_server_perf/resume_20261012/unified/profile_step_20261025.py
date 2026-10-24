"""Per-launch breakdown of one mixed attention step (A6): native v3's full step vs the unified plan's step, on the same
capture, torch profiler over eager launches. Prints every kernel each arm launches with its mean time, so a fixed
per-step cost (an extra kernel, a slower FA3 launch, a copy) shows up by name.
  UNIFIED_SO=... python profile_step_20261025.py <capture.pt> [...]
Claude, 2026-10-25.
"""
import collections
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/dev/shm/w/ub")
import bench_captured_mixed_20261023 as B  # noqa: E402
import unified_executor_20261024 as U  # noqa: E402
torch.ops.load_library(os.environ["UNIFIED_SO"])
ITERS = 20
for path in map(__import__("pathlib").Path, sys.argv[1:]):
    case = B.load_case(path)
    g = case.geometry
    v3 = B.launches_for(case, policy_v3=True)["native_pdl_geometry_v3"]
    ex = U.UnifiedExecutor(g["heads"], g["kv_heads"], torch.device("cuda"),
                           fa3_min_packed=int(os.environ.get("UNIFIED_FA3_MIN", "33")))
    plan = ex.plan(g["query_lengths"], [int(x) for x in case.cache_lengths.tolist()], case.page_table)
    arms = {"v3": v3, "unified": lambda: plan.run(case.query, case.key, case.value, g["scale"], case.output)}
    print(f"== {os.path.basename(path)} q={g['query_lengths'][:4]}... rows={len(g['query_lengths'])} "
          f"plan={ {k: plan.stats[k] for k in ('items32', 'items16', 'items_fa3', 'fa2_chunk', 'fa3_chunk', 'max_splits_fa3')} }")
    for name, fn in arms.items():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(ITERS):
                fn()
            torch.cuda.synchronize()
        per = collections.Counter()
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                per[e.name[:110]] += e.device_time_total / ITERS
        tot = sum(per.values())
        print(f"  {name}: kernel sum {tot:.1f} us")
        for k, v in per.most_common(8):
            print(f"     {v:8.1f} us  {k}")
    del case, plan, ex
    torch.cuda.empty_cache()
