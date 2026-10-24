"""Stage 1 for the unified tile dispatcher (A6) on the trace-sampled captures: every FA3 baseline the
original bench runs (sglang's single call, full / packed splits, dual-stream role split), native v3 (the
current serving build), and `unified` (one plan, FA3 -> PDL -> FA2 with in-kernel merges). Graph
replay, both arm orders, the same FP32 gate. Records match bench_captured_mixed's, so
analyze_trace_sampled_20261023.py applies. Also records the host plan time per capture.
  python bench_unified_trace_20261024.py <captures dir> <output json> [--iterations N]
Claude, 2026-10-24.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/dev/shm/w/ub")
import bench_captured_mixed_20261023 as B  # noqa: E402
import unified_executor_20261024 as U  # noqa: E402

torch.ops.load_library(__import__("os").environ.get("UNIFIED_SO", "/opt/tiger/w/torch_ext_unified_20261024/unified_dispatch_ext/unified_dispatch_ext.so"))
KEEP = ("fa3_full_default", "fa3_full_split8", "fa3_full_split32", "fa3_full_pack_split8", "fa3_full_pack_split32",
        "fa3_dual_split0", "fa3_dual_split8", "fa3_dual_split32", "native_pdl_geometry_v3")

parser = argparse.ArgumentParser()
parser.add_argument("captures", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--iterations", type=int, default=20)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit("fresh output required")
torch.backends.cuda.matmul.allow_tf32 = False
files = sorted(args.captures.glob("*.pt"))
report = dict(scope="unified dispatcher vs FA3 baselines and native v3, trace-sampled captures, graph replay",
              gpu=torch.cuda.get_device_name(0), records=[], captures=[])
for path in files:
    case = B.load_case(path)
    expected = B.fp32_reference(case)
    g = case.geometry
    launches = {k: v for k, v in B.launches_for(case, policy_v3=True).items() if k in KEEP}
    ex = U.UnifiedExecutor(g["heads"], g["kv_heads"], torch.device("cuda"))
    kv = [int(x) for x in case.cache_lengths.tolist()]
    t0 = time.perf_counter()
    for _ in range(10):
        plan = ex.plan(g["query_lengths"], kv, case.page_table)
    plan_ms = (time.perf_counter() - t0) / 10 * 1e3
    launches["unified"] = lambda plan=plan: plan.run(case.query, case.key, case.value, g["scale"], case.output)
    report["captures"].append(dict(file=path.name, plan_ms=plan_ms, plan_stats=plan.stats,
                                   query_lengths=g["query_lengths"], kv_lengths=kv))
    for order in ("forward", "reverse"):
        entries = list(launches.items())
        if order == "reverse":
            entries.reverse()
        for arm, launch in entries:
            r = B.measure(case, launch, expected, "graph", args.iterations)
            r.update(capture=path.name, mode=g["mode"], arm=arm, style="graph", order=order)
            report["records"].append(r)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    u = [r["milliseconds"] for r in report["records"] if r["capture"] == path.name and r["arm"] == "unified"]
    v = [r["milliseconds"] for r in report["records"] if r["capture"] == path.name and r["arm"] == "native_pdl_geometry_v3"]
    d = [r["milliseconds"] for r in report["records"] if r["capture"] == path.name and r["arm"] == "fa3_full_default"]
    print(f"UNIFIED_BENCH {path.name} unified={sum(u)/len(u):.4f} v3={sum(v)/len(v):.4f} default={sum(d)/len(d):.4f} "
          f"plan_ms={plan_ms:.3f} stats={plan.stats}", flush=True)
    del case, expected, launches, plan, ex
    torch.cuda.empty_cache()
print("UNIFIED_BENCH_DONE", len(files), flush=True)
