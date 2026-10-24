"""Validate fused BF16 merge stores on every archived Qwen2.5 and Qwen3 capture."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch

from bench_captured_mixed_20261023 import fp32_reference, load_case, measure
from bench_fa3_role_controls_20261023 import role_launches
from bench_fa3_row_packing_20261023 import control_launch
from native_fa3_fa2_pdl_20261023 import NativePdlExecutor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise ValueError("Fresh direct-output result required")
    torch.backends.cuda.matmul.allow_tf32 = False
    groups = (
        ("qwen25", "/dev/shm/w/fa3_row_packing_20261023/results.json", ("/dev/shm/w/captured_pair_tuning_20261023/joined", "/dev/shm/w/captured_long_policy_20261023")),
        ("qwen3", "/dev/shm/w/qwen3_captured_20261023/validation.json", ("/dev/shm/w/qwen3_captured_20261023",)),
    )
    report = dict(
        scope="Same captures, unchanged split policy; direct final BF16 store versus separate conversion; excludes planning",
        cases=[], records=[],
        direct_library_sha256=hashlib.sha256(Path(os.environ["NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY"]).read_bytes()).hexdigest(),
    )
    for model, prior_path, roots in groups:
        prior = json.loads(Path(prior_path).read_text())
        paths = {path.parent.parent.name + "/" + path.name: path for root in roots for path in Path(root).glob("*/captures/*.pt")}
        for capture in prior["cases"]:
            case = load_case(paths[capture["name"]])
            if case.source_sha256 != capture["sha256"]:
                raise ValueError("Raw capture changed")
            expected = fp32_reference(case)
            old = [row for row in prior["records"] if row["capture"] == capture["name"] and row["style"] == "graph"]
            means = {arm: statistics.mean(row["milliseconds"] for row in old if row["arm"] == arm) for arm in {row["arm"] for row in old}}
            baseline = min((arm for arm in means if "fa3" in arm), key=means.get)
            controls = role_launches(case)
            baseline_launch = {
                "prior_fa3": controls[capture["prior_control"]],
                "tuned_dual_fa3": control_launch(case, **capture["dual_selection"]),
                "tuned_full_fa3": control_launch(case, **capture["full_selection"]),
            }[baseline]
            plans = {}
            for direct in (False, True):
                executor = NativePdlExecutor(case.geometry["heads"], case.geometry["kv_heads"], case.query.device, adaptive_splits=True, policy_version="geometry_v2", fp32_partials=True, direct_bf16_output=direct, workspace_bytes=512 << 20)
                plans[direct] = executor.plan(case.geometry["query_lengths"], case.geometry["prefix_lengths"], case.page_table, case.cache_lengths, case.cumulative_query)
            launches = {
                "selected_fa3": baseline_launch,
                "separate_conversion": lambda: plans[False].run(case.query, case.key, case.value, case.geometry["scale"], case.output),
                "direct_serial": lambda: plans[True].run(case.query, case.key, case.value, case.geometry["scale"], case.output, pdl=False),
                "direct_pdl": lambda: plans[True].run(case.query, case.key, case.value, case.geometry["scale"], case.output),
            }
            launches["separate_conversion"]()
            snapshot = case.output.clone()
            for arm in ("direct_serial", "direct_pdl"):
                case.output.fill_(float("nan"))
                launches[arm]()
                if not torch.equal(case.output, snapshot):
                    raise AssertionError(f"Direct output is not bitwise equal: {model}/{capture['name']}/{arm}")
            name = model + "/" + capture["name"]
            report["cases"].append(dict(name=name, sha256=case.source_sha256, baseline=baseline, direct_output=plans[True].shape.direct_row_output, exact_to_separate_conversion=True, serial_pdl_exact=True))
            for style in ("eager", "graph"):
                for order in ("forward", "reverse"):
                    entries = list(launches.items())
                    if order == "reverse":
                        entries.reverse()
                    for arm, launch in entries:
                        result = measure(case, launch, expected, style, 20)
                        result.update(capture=name, arm=arm, style=style, order=order)
                        report["records"].append(result)
                    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
            print("DIRECT_OUTPUT_CAPTURE_DONE", name, plans[True].shape.direct_row_output, flush=True)
            del case, expected, controls, baseline_launch, plans, executor, launches, entries, launch, snapshot
            torch.cuda.empty_cache()
    if len(report["records"]) != len(report["cases"]) * 16:
        raise AssertionError("Incomplete direct-output matrix")
    print("DIRECT_OUTPUT_MATRIX_DONE", len(report["cases"]), len(report["records"]), flush=True)


if __name__ == "__main__":
    main()
