"""Per-kernel times of the unified dispatcher on one capture, from the torch profiler (CUDA kernel events),
for several routings. Diagnostic for the 6x gap vs native v3.
  UNIFIED_FA3_MIN=<n> UNIFIED_FA2_CPS=<k> python profile_unified_20261024.py capture.pt
"""
import os
import sys
from collections import defaultdict
from pathlib import Path
import torch
sys.path.insert(0, "/dev/shm/w/ub")
import bench_captured_mixed_20261023 as B  # noqa: E402
import unified_executor_20261024 as U  # noqa: E402
torch.ops.load_library(__import__("os").environ.get("UNIFIED_SO", "/opt/tiger/w/torch_ext_unified_20261024/unified_dispatch_ext/unified_dispatch_ext.so"))

case = B.load_case(Path(sys.argv[1]))
g = case.geometry
print("rows", g["query_lengths"][:6], "... n=", len(g["query_lengths"]), "kv max", max(g["kv_lengths"]))
for fa3_min in [int(x) for x in os.environ.get("UNIFIED_FA3_MINS", "49,1,1000000000").split(",")]:
    for pdl in (True, False):
        ex = U.UnifiedExecutor(g["heads"], g["kv_heads"], torch.device("cuda"), fa3_min_packed=fa3_min,
                               fa2_ctas_per_sm=int(os.environ.get("UNIFIED_FA2_CPS", "2")), pdl=pdl)
        plan = ex.plan(g["query_lengths"], [int(x) for x in case.cache_lengths.tolist()], case.page_table)
        run = lambda: plan.run(case.query, case.key, case.value, g["scale"], case.output)
        for _ in range(5):
            run()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                run()
            torch.cuda.synchronize()
        k = defaultdict(float)
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                name = e.name
                key = ("FA2" if "UnifiedFA2Kernel" in name else "FA3" if "FlashAttnFwdSm90" in name or "device_kernel" in name
                       else "scatter" if "Scatter" in name else name[:40])
                k[key] += e.device_time / 10 / 1e3
        st = plan.stats
        print(f"fa3_min={fa3_min} pdl={pdl} items48/32/16/fa3={st['items48']}/{st['items32']}/{st['items16']}/{st['items_fa3']} "
              f"chunk fa2/fa3={st['fa2_chunk']}/{st['fa3_chunk']} groups={st['merge_groups']} | " +
              " ".join(f"{a}={b:.4f}ms" for a, b in sorted(k.items())), flush=True)
