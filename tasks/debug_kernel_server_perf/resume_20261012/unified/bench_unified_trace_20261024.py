"""Stage 1 for the unified tile dispatcher (A6) on the trace-sampled captures: every FA3 baseline the
original bench runs (sglang's single call, full / packed splits, dual-stream role split), native v3 (the
current serving build), and `unified` (one plan, FA3 -> PDL -> FA2 with in-kernel merges). Graph
replay, both arm orders, the same FP32 gate. Records match bench_captured_mixed's, so
analyze_trace_sampled_20261023.py applies. Also records the host plan time per capture.
  python bench_unified_trace_20261024.py <captures dir> <output json> [--iterations N]
Claude, 2026-10-24.
"""
import argparse
import os
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
parser.add_argument("--unified-only", action="store_true", help="skip the baselines (they ran in an earlier pass)")
parser.add_argument("--label", default="unified", help="arm name for this unified configuration")
parser.add_argument("--append", action="store_true", help="add records to an existing output")
args = parser.parse_args()
if args.output.exists() and not args.append:
    raise SystemExit("fresh output required")
torch.backends.cuda.matmul.allow_tf32 = False
files = sorted(args.captures.glob("*.pt"))
report = dict(scope="unified dispatcher vs FA3 baselines and native v3, trace-sampled captures, graph replay",
              gpu=torch.cuda.get_device_name(0), records=[], captures=[])
if os.environ.get("UNIFIED_TIMING_BOUND") == "1":
    # Timing bound only (e.g. FA3 with its KV loads compiled out): the output is garbage by construction, so it is
    # not validated; the records carry bound=True.
    B.validate = lambda output, expected: float("nan")
if args.append and args.output.exists():
    report = json.loads(args.output.read_text())
for path in files:
    case = B.load_case(path)
    if os.environ.get("UNIFIED_ZERO_FILL") == "1":
        # diagnostic: non-live pages zero instead of NaN (separates "reads a non-live page" from other faults)
        case.key.nan_to_num_(0.0)
        case.value.nan_to_num_(0.0)
    expected = B.fp32_reference(case)
    g = case.geometry
    launches = {} if args.unified_only else {k: v for k, v in B.launches_for(case, policy_v3=True).items() if k in KEEP}
    import os
    ex = U.UnifiedExecutor(g["heads"], g["kv_heads"], torch.device("cuda"),
                           fa3_min_packed=int(os.environ.get("UNIFIED_FA3_MIN", "49")),
                           fa2_ctas_per_sm=int(os.environ.get("UNIFIED_FA2_CPS", "2")))
    kv = [int(x) for x in case.cache_lengths.tolist()]
    t0 = time.perf_counter()
    for _ in range(10):
        plan = ex.plan(g["query_lengths"], kv, case.page_table)
    plan_ms = (time.perf_counter() - t0) / 10 * 1e3
    launches[args.label] = lambda plan=plan: plan.run(case.query, case.key, case.value, g["scale"], case.output)
    report["captures"].append(dict(file=path.name, label=args.label, plan_ms=plan_ms, plan_stats=plan.stats,
                                   query_lengths=g["query_lengths"], kv_lengths=kv))
    for order in ("forward", "reverse"):
        entries = list(launches.items())
        if order == "reverse":
            entries.reverse()
        for arm, launch in entries:
            try:
                r = B.measure(case, launch, expected, "graph", args.iterations)
            except AssertionError as e:
                bad = (~torch.isfinite(case.output)).any(dim=-1)
                cq = case.cumulative_query.tolist()
                toks = sorted(set(int(t) for t in bad.nonzero()[:, 0].tolist()))
                reqs = sorted(set(next(i for i in range(len(cq) - 1) if cq[i] <= t < cq[i + 1]) for t in toks))
                ql = case.geometry["query_lengths"]
                print(f"UNIFIED_BENCH_FAIL {path.name} arm={arm} order={order}: {e}; bad rows={int(bad.sum())} "
                      f"requests={reqs[:10]} their q_len={[ql[r] for r in reqs[:10]]} heads={sorted(set(bad.nonzero()[:,1].tolist()))[:8]}",
                      flush=True)
                raise
            r.update(capture=path.name, mode=g["mode"], arm=arm, style="graph", order=order,
                     bound=os.environ.get("UNIFIED_TIMING_BOUND") == "1")
            report["records"].append(r)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    def mean(arm):
        x = [r["milliseconds"] for r in report["records"] if r["capture"] == path.name and r["arm"] == arm]
        return sum(x) / len(x) if x else float("nan")
    print(f"UNIFIED_BENCH {path.name} {args.label}={mean(args.label):.4f} v3={mean('native_pdl_geometry_v3'):.4f} "
          f"default={mean('fa3_full_default'):.4f} "
          f"plan_ms={plan_ms:.3f} stats={plan.stats}", flush=True)
    del case, expected, launches, plan, ex
    torch.cuda.empty_cache()
print("UNIFIED_BENCH_DONE", len(files), flush=True)
