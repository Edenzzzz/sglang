"""Development regression for a revised policy; these captures are not held out."""

import argparse
import json
from pathlib import Path
import statistics

import torch

from bench_captured_mixed_20261023 import fp32_reference, load_case, measure
from bench_fa3_role_controls_20261023 import role_launches
from native_fa3_fa2_pdl_20261023 import NativePdlExecutor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise ValueError("Policy validation output already exists")
    torch.backends.cuda.matmul.allow_tf32 = False
    prior = json.loads(Path("/dev/shm/w/captured_fa3_role_controls_20261023/results.json").read_text())
    paths = {
        path.parent.parent.name + "/" + path.name: path
        for root in (Path("/dev/shm/w/captured_pair_tuning_20261023/joined"), Path("/dev/shm/w/captured_long_policy_20261023"))
        for path in root.glob("*/captures/*.pt")
    }
    report = dict(scope="Development regression after the v1 failure; not held-out validation of v2", cases=[], records=[])
    for capture in prior["captures"]:
        case = load_case(paths[capture["name"]])
        if case.source_sha256 != capture["sha256"]:
            raise ValueError("Captured tensors changed")
        old = [row for row in prior["records"] if row["capture"] == capture["name"]]
        means = {arm: statistics.mean(row["milliseconds"] for row in old if row["arm"] == arm) for arm in {row["arm"] for row in old}}
        baseline = min((arm for arm in means if arm.startswith("fa3_roles") or arm.startswith("fa3_full")), key=means.get)
        controls = role_launches(case)
        geometry = case.geometry
        executor = NativePdlExecutor(
            geometry["heads"], geometry["kv_heads"], case.query.device,
            adaptive_splits=True, policy_version="geometry_v2", workspace_bytes=512 << 20,
        )
        plan = executor.plan(geometry["query_lengths"], geometry["prefix_lengths"], case.page_table, case.cache_lengths, case.cumulative_query)
        launches = {
            "selected_fa3": controls[baseline], "geometry_v1": controls["native_pdl_geometry_v1"],
            "geometry_v2_serial": lambda: plan.run(case.query, case.key, case.value, geometry["scale"], case.output, pdl=False),
            "geometry_v2_pdl": lambda: plan.run(case.query, case.key, case.value, geometry["scale"], case.output),
        }
        expected = fp32_reference(case)
        launches["geometry_v2_serial"](); snapshot = case.output.clone()
        launches["geometry_v2_pdl"]()
        if not torch.equal(snapshot, case.output):
            raise AssertionError("V2 serial/PDL disagreement")
        report["cases"].append(dict(name=capture["name"], sha256=case.source_sha256, baseline=baseline, policy=plan.shape.split_settings, serial_pdl_exact=True))
        for style in ("eager", "graph"):
            for order in ("forward", "reverse"):
                entries = list(launches.items())
                if order == "reverse":
                    entries.reverse()
                for arm, launch in entries:
                    result = measure(case, launch, expected, style, 10)
                    result.update(capture=capture["name"], arm=arm, style=style, order=order)
                    report["records"].append(result)
                    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
        print("GEOMETRY_V2_CASE_DONE", capture["name"], plan.shape.split_settings, flush=True)
        del case, expected, controls, executor, plan, launches, entries, launch, snapshot
        torch.cuda.empty_cache()
    if len(report["records"]) != len(report["cases"]) * 16:
        raise AssertionError("Incomplete v2 policy regression")
    print("GEOMETRY_V2_REGRESSION_DONE", len(report["cases"]), len(report["records"]), flush=True)


if __name__ == "__main__":
    main()
