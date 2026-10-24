"""Diagnose the frozen policy's heterogeneous deep-prefill regression."""

import json
from pathlib import Path

import torch

from bench_captured_mixed_20261023 import fp32_reference, load_case, measure
from bench_fa3_role_controls_20261023 import role_launches
from native_fa3_fa2_pdl_20261023 import NativePdlExecutor


def main():
    root = Path(__file__).resolve().parent
    destination = root / "results.json"
    if destination.exists():
        raise ValueError("Diagnostic results already exist")
    torch.backends.cuda.matmul.allow_tf32 = False
    case = load_case(Path("/dev/shm/w/captured_long_policy_20261023/fanout384_four/captures/cached_large.pt"))
    geometry = case.geometry
    expected = fp32_reference(case)
    controls = role_launches(case)
    launches = {
        "fa3_dual_selected": controls["fa3_roles_ps2_packNone_rs32"],
        "geometry_v1": controls["native_pdl_geometry_v1"],
    }
    plans = []
    for splits in (1, 2, 4, 8, 16, 32):
        executor = NativePdlExecutor(
            geometry["heads"], geometry["kv_heads"], case.query.device,
            prefix_splits=splits, row_split_size=8192, workspace_bytes=512 << 20,
        )
        plan = executor.plan(geometry["query_lengths"], geometry["prefix_lengths"], case.page_table, case.cache_lengths, case.cumulative_query)
        plans.append(plan)
        launches[f"native_prefix_splits{splits}"] = lambda plan=plan: plan.run(
            case.query, case.key, case.value, geometry["scale"], case.output
        )
    report = dict(scope="Failure diagnosis on the already reported held-out capture; not a changed geometry_v1 result", capture=geometry, sha256=case.source_sha256, records=[])
    for style in ("eager", "graph"):
        for order in ("forward", "reverse"):
            entries = list(launches.items())
            if order == "reverse":
                entries.reverse()
            for arm, launch in entries:
                result = measure(case, launch, expected, style, 10)
                result.update(arm=arm, style=style, order=order)
                report["records"].append(result)
                destination.write_text(json.dumps(report, indent=2) + "\n")
                print("DEEP_PREFILL_DIAGNOSTIC", arm, style, order, result["milliseconds"], flush=True)
    if len(report["records"]) != 32:
        raise AssertionError("Incomplete deep-prefill diagnostic")
    print("DEEP_PREFILL_DIAGNOSTIC_DONE", flush=True)


if __name__ == "__main__":
    main()
