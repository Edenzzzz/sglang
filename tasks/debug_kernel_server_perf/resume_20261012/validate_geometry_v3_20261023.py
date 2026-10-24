"""Development regression for the packed-GQA SplitKV work-count correction."""

import argparse
import json
from pathlib import Path

import torch

from bench_captured_mixed_20261023 import fp32_reference, load_case, measure
from bench_fa3_role_controls_20261023 import role_launches
from bench_fa3_row_packing_20261023 import control_launch
from native_fa3_fa2_pdl_20261023 import NativePdlExecutor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--sm-report", type=Path, default=Path("/dev/shm/w/captured_sm_controls_20261023/results.json"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--roots", type=Path, nargs="+")
    parser.add_argument("--model-key", default="qwen25")
    arguments = parser.parse_args()
    output = arguments.output
    if output.exists():
        raise ValueError("Fresh v3 validation required")
    torch.backends.cuda.matmul.allow_tf32 = False
    sm_report = json.loads(arguments.sm_report.read_text())
    sm_cases = {case["name"]: case for case in sm_report["cases"]}
    groups = (
        ("qwen25", "/dev/shm/w/fa3_row_packing_20261023/results.json", ("/dev/shm/w/captured_pair_tuning_20261023/joined", "/dev/shm/w/captured_long_policy_20261023")),
        ("qwen3", "/dev/shm/w/qwen3_captured_20261023/validation.json", ("/dev/shm/w/qwen3_captured_20261023",)),
    )
    if bool(arguments.prior) != bool(arguments.roots):
        raise ValueError("Custom prior and capture roots must be supplied together")
    if arguments.prior:
        groups = ((arguments.model_key, arguments.prior, arguments.roots),)
    report = dict(scope="Packed-GQA work-count comparison; consult capture protocol and pre-capture source freeze for development/holdout status", cases=[], records=[])
    for model, prior_path, roots in groups:
        prior = json.loads(Path(prior_path).read_text())
        paths = {path.parent.parent.name + "/" + path.name: path for root in roots for path in Path(root).glob("*/captures/*.pt")}
        for capture in prior["cases"]:
            name = model + "/" + capture["name"]
            case = load_case(paths[capture["name"]])
            if case.source_sha256 != capture["sha256"]:
                raise ValueError("Captured inputs changed")
            geometry = case.geometry
            expected = fp32_reference(case)
            controls = role_launches(case)
            previous = {"prior_fa3": controls[capture["prior_control"]], "tuned_dual_fa3": control_launch(case, **capture["dual_selection"]), "tuned_full_fa3": control_launch(case, **capture["full_selection"])}[sm_cases[name]["previous_baseline"]]
            plans = {}
            for version in ("geometry_v2", "geometry_v3"):
                executor = NativePdlExecutor(geometry["heads"], geometry["kv_heads"], case.query.device, adaptive_splits=True, policy_version=version, fp32_partials=True, direct_bf16_output=True, workspace_bytes=512 << 20)
                plans[version] = executor.plan(geometry["query_lengths"], geometry["prefix_lengths"], case.page_table, case.cache_lengths, case.cumulative_query)
            launches = {
                "previous_fa3": previous,
                "sm_tuned_dual_fa3": control_launch(case, **sm_cases[name]["selected_sm_settings"]),
                "geometry_v2": lambda: plans["geometry_v2"].run(case.query, case.key, case.value, geometry["scale"], case.output),
                "geometry_v3_serial": lambda: plans["geometry_v3"].run(case.query, case.key, case.value, geometry["scale"], case.output, pdl=False),
                "geometry_v3_pdl": lambda: plans["geometry_v3"].run(case.query, case.key, case.value, geometry["scale"], case.output),
            }
            launches["geometry_v3_serial"]()
            snapshot = case.output.clone()
            launches["geometry_v3_pdl"]()
            if not torch.equal(case.output, snapshot):
                raise AssertionError("V3 serial/PDL mismatch")
            launches["geometry_v2"]()
            unchanged_plan_exact = torch.equal(snapshot, case.output)
            v2, v3 = plans["geometry_v2"].shape.split_settings, plans["geometry_v3"].shape.split_settings
            if v2["prefix_splits"] == v3["prefix_splits"] and not unchanged_plan_exact:
                raise AssertionError("Unchanged policy plan changed output")
            report["cases"].append(dict(name=name, sha256=case.source_sha256, geometry_v2=v2, geometry_v3=v3, serial_pdl_exact=True, v2_output_exact=unchanged_plan_exact))
            for style in ("eager", "graph"):
                for order in ("forward", "reverse"):
                    entries = list(launches.items())
                    if order == "reverse":
                        entries.reverse()
                    for arm, launch in entries:
                        result = measure(case, launch, expected, style, 20)
                        result.update(capture=name, arm=arm, style=style, order=order)
                        report["records"].append(result)
                output.write_text(json.dumps(report, indent=2) + "\n")
            print("GEOMETRY_V3_CAPTURE_DONE", name, v2["prefix_splits"], v3["prefix_splits"], flush=True)
            del case, expected, controls, previous, executor, plans, launches, entries, launch, snapshot
            torch.cuda.empty_cache()
    if len(report["records"]) != len(report["cases"]) * 20:
        raise AssertionError("Incomplete v3 regression")
    print("GEOMETRY_V3_REGRESSION_DONE", len(report["cases"]), len(report["records"]), flush=True)


if __name__ == "__main__":
    main()
