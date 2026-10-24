"""Matched FP32-partial ablation on unchanged real serving captures."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch

from bench_captured_mixed_20261023 import fp32_reference, load_case, measure
from bench_fa3_role_controls_20261023 import role_launches
from native_fa3_fa2_pdl_20261023 import NativePdlExecutor


def row_errors(actual, expected):
    difference = actual.float() - expected
    relative = difference.norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-6)
    return dict(
        relative_l2=float(difference.norm() / expected.norm().clamp_min(1e-6)),
        mean_head_relative=float(relative.mean()),
        worst_head_relative=float(relative.max()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise ValueError("A fresh output is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    prior = json.loads(Path("/dev/shm/w/captured_fa3_role_controls_20261023/results.json").read_text())
    paths = {
        path.parent.parent.name + "/" + path.name: path
        for root in (Path("/dev/shm/w/captured_pair_tuning_20261023/joined"), Path("/dev/shm/w/captured_long_policy_20261023"))
        for path in root.glob("*/captures/*.pt")
    }
    report = dict(
        scope="Development ablation on the same 21 captures; no held-out or serving-speed claim; excludes planning, includes final BF16 conversion",
        gpu=torch.cuda.get_device_name(0), cases=[], records=[],
        libraries={name: hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest() for name in ("NATIVE_PDL_FA2_LIBRARY", "NATIVE_PDL_FA2_FP32_LIBRARY", "NATIVE_PDL_FA3_LIBRARY")},
        executor_sha256=hashlib.sha256(Path(__file__).with_name("native_fa3_fa2_pdl_20261023.py").read_bytes()).hexdigest(),
    )
    for capture in prior["captures"]:
        case = load_case(paths[capture["name"]])
        if case.source_sha256 != capture["sha256"]:
            raise ValueError("Captured inputs changed")
        old = [row for row in prior["records"] if row["capture"] == capture["name"]]
        means = {arm: statistics.mean(row["milliseconds"] for row in old if row["arm"] == arm) for arm in {row["arm"] for row in old}}
        baseline = min((arm for arm in means if arm.startswith("fa3_roles") or arm.startswith("fa3_full")), key=means.get)
        controls = role_launches(case)
        geometry = case.geometry
        plans = {}
        for precision in ("bf16", "fp32"):
            executor = NativePdlExecutor(
                geometry["heads"], geometry["kv_heads"], case.query.device,
                adaptive_splits=True, policy_version="geometry_v2", workspace_bytes=512 << 20,
                fp32_partials=precision == "fp32",
            )
            plans[precision] = executor.plan(geometry["query_lengths"], geometry["prefix_lengths"], case.page_table, case.cache_lengths, case.cumulative_query)
        launches = {"selected_fa3": controls[baseline]}
        for precision, plan in plans.items():
            for pdl in (False, True):
                launches[f"{precision}_{'pdl' if pdl else 'serial'}"] = lambda plan=plan, pdl=pdl: plan.run(case.query, case.key, case.value, geometry["scale"], case.output, pdl=pdl)
        launches["fp32_original_tile_pdl"] = lambda: plans["fp32"].run(case.query, case.key, case.value, geometry["scale"], case.output, query_tile=0)
        expected = fp32_reference(case)
        numerical = {}
        for precision in ("bf16", "fp32"):
            launches[f"{precision}_serial"]()
            snapshot = case.output.clone()
            launches[f"{precision}_pdl"]()
            if not torch.equal(snapshot, case.output):
                raise AssertionError(f"{precision} serial/PDL disagreement")
        for arm, launch in launches.items():
            launch()
            numerical[arm] = row_errors(case.output[case.prefix_tokens:], expected[case.prefix_tokens:])
        report["cases"].append(dict(
            name=capture["name"], sha256=case.source_sha256, baseline=baseline,
            policy=plans["fp32"].shape.split_settings, query_tile=plans["fp32"].shape.native_tile,
            serial_pdl_exact=True, row_errors=numerical,
            bf16_rounding_floor=row_errors(expected[case.prefix_tokens:].bfloat16(), expected[case.prefix_tokens:]),
        ))
        for style in ("eager", "graph"):
            for order in ("forward", "reverse"):
                entries = list(launches.items())
                if order == "reverse":
                    entries.reverse()
                for arm, launch in entries:
                    result = measure(case, launch, expected, style, 15)
                    result.update(capture=capture["name"], arm=arm, style=style, order=order)
                    report["records"].append(result)
                    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
        print("FP32_PARTIALS_CAPTURE_DONE", capture["name"], numerical, flush=True)
        del case, expected, plans, launches, controls, entries, launch, executor, plan, snapshot
        torch.cuda.empty_cache()
    if len(report["records"]) != len(report["cases"]) * 24:
        raise AssertionError("Incomplete FP32-partial capture matrix")
    print("FP32_PARTIALS_CAPTURE_MATRIX_DONE", len(report["cases"]), len(report["records"]), flush=True)


if __name__ == "__main__":
    main()
