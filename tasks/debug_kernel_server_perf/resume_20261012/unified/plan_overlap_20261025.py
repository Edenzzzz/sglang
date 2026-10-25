"""Does building the next step's A6 plan wait for the GPU? (serving overlap; A6 open item "plan time, overlapped")
For each capture: queue ~20 steps of attention (the previous step still running), then time the host wall time of the
next plan built (a) from host lengths only (blocking host-to-device copies) and (b) with the device metadata the FA3
backend already holds (cu_seqlens_q, cache_seqlens). (a) waits for the queued work; (b) should not. Also checks that
both plans give the same output.
  UNIFIED_SO=... python plan_overlap_20261025.py <captures dir> [...]
Claude, 2026-10-25.
"""
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/dev/shm/w/ub")
import bench_captured_mixed_20261023 as B  # noqa: E402
import unified_executor_20261024 as U  # noqa: E402
torch.ops.load_library(os.environ["UNIFIED_SO"])

rows = []
for d in sys.argv[1:]:
    for path in sorted(Path(d).glob("*.pt"))[::4]:
        case = B.load_case(path)
        g = case.geometry
        ql = g["query_lengths"]
        kv = [int(x) for x in case.cache_lengths.tolist()]
        ex = U.UnifiedExecutor(g["heads"], g["kv_heads"], torch.device("cuda"),
                               fa3_min_packed=int(os.environ.get("UNIFIED_FA3_MIN", "33")))
        cq = case.cumulative_query.to(torch.int32).contiguous()
        cl = case.cache_lengths.to(torch.int32).contiguous()
        ref = ex.plan(ql, kv, case.page_table)
        out_a = torch.empty_like(case.output)
        out_b = torch.empty_like(case.output)
        for _ in range(3):
            ref.run(case.query, case.key, case.value, g["scale"], out_a)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        ref.run(case.query, case.key, case.value, g["scale"], out_a)
        e1.record()
        e1.synchronize()
        step_ms = e0.elapsed_time(e1)
        t = {}
        for mode in ("host", "device"):
            samples = []
            for _ in range(5):
                for _ in range(20):
                    ref.run(case.query, case.key, case.value, g["scale"], out_a)
                t0 = time.perf_counter()
                p = ex.plan(ql, kv, case.page_table) if mode == "host" else ex.plan(ql, kv, case.page_table, cq, cl)
                samples.append((time.perf_counter() - t0) * 1e3)
                torch.cuda.synchronize()
            t[mode] = statistics.median(samples)
        pb = ex.plan(ql, kv, case.page_table, cq, cl)
        pa = ex.plan(ql, kv, case.page_table)
        pa.run(case.query, case.key, case.value, g["scale"], out_a)
        pb.run(case.query, case.key, case.value, g["scale"], out_b)
        torch.cuda.synchronize()
        same = torch.equal(out_a, out_b)
        rows.append((t["host"], t["device"], 20 * step_ms))
        print(f"PLAN_OVERLAP {path.parent.parent.name}/{path.name} step={step_ms:.3f}ms queued={20 * step_ms:.2f}ms "
              f"plan_host={t['host']:.3f}ms plan_device={t['device']:.3f}ms same_output={same}", flush=True)
        if not same:
            raise SystemExit("device-metadata plan output differs")
        del case, ex, ref, pa, pb
        torch.cuda.empty_cache()
print("PLAN_OVERLAP_TOTAL n=%d median plan_host=%.3fms plan_device=%.3fms median queued=%.2fms" % (
    len(rows), statistics.median(r[0] for r in rows), statistics.median(r[1] for r in rows),
    statistics.median(r[2] for r in rows)), flush=True)
