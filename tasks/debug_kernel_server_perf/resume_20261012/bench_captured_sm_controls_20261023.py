"""Add explicit SM-grid budgets to tuned dual-FA3 controls on actual captures."""

import argparse
import json
from pathlib import Path
import statistics

import torch

from bench_captured_mixed_20261023 import fp32_reference, load_case, measure
from bench_direct_output_captured_20261023 import control_launch, role_launches
from native_fa3_fa2_pdl_20261023 import NativePdlExecutor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--roots", type=Path, nargs="+")
    parser.add_argument("--model-key", default="qwen25")
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise ValueError("Fresh SM-control output required")
    torch.backends.cuda.matmul.allow_tf32 = False
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    groups = (
        ("qwen25", "/dev/shm/w/fa3_row_packing_20261023/results.json", ("/dev/shm/w/captured_pair_tuning_20261023/joined", "/dev/shm/w/captured_long_policy_20261023")),
        ("qwen3", "/dev/shm/w/qwen3_captured_20261023/validation.json", ("/dev/shm/w/qwen3_captured_20261023",)),
    )
    if bool(arguments.prior) != bool(arguments.roots):
        raise ValueError("Custom prior and capture roots must be supplied together")
    if arguments.prior:
        groups = ((arguments.model_key, arguments.prior, arguments.roots),)
    report = dict(scope="Per-capture FA3 SM-grid tuning, not a physical SM partition or deployed dispatch rule; native geometry policy unchanged", sms=sms, cases=[], search=[], records=[])
    for model, prior_path, roots in groups:
        prior = json.loads(Path(prior_path).read_text())
        paths = {path.parent.parent.name + "/" + path.name: path for root in roots for path in Path(root).glob("*/captures/*.pt")}
        for capture in prior["cases"]:
            name = model + "/" + capture["name"]
            case = load_case(paths[capture["name"]])
            if case.source_sha256 != capture["sha256"]:
                raise ValueError("Captured inputs changed")
            expected = fp32_reference(case)
            old = [row for row in prior["records"] if row["capture"] == capture["name"] and row["style"] == "graph"]
            means = {arm: statistics.mean(row["milliseconds"] for row in old if row["arm"] == arm) for arm in {row["arm"] for row in old}}
            baseline = min((arm for arm in means if "fa3" in arm), key=means.get)
            controls = role_launches(case)
            previous = {"prior_fa3": controls[capture["prior_control"]], "tuned_dual_fa3": control_launch(case, **capture["dual_selection"]), "tuned_full_fa3": control_launch(case, **capture["full_selection"])}[baseline]
            settings = []
            for prefix_margin in range(0, sms, 8):
                for row_margin in sorted({0, sms - prefix_margin} if prefix_margin else {0}):
                    setting = dict(capture["dual_selection"], prefix_margin=prefix_margin, row_margin=row_margin)
                    result = measure(case, control_launch(case, **setting), expected, "graph", 12)
                    result.update(capture=name, setting=setting)
                    report["search"].append(result)
                    settings.append(result)
            selected = min(settings, key=lambda row: row["milliseconds"])["setting"]
            executor = NativePdlExecutor(case.geometry["heads"], case.geometry["kv_heads"], case.query.device, adaptive_splits=True, policy_version="geometry_v2", fp32_partials=True, direct_bf16_output=True, workspace_bytes=512 << 20)
            plan = executor.plan(case.geometry["query_lengths"], case.geometry["prefix_lengths"], case.page_table, case.cache_lengths, case.cumulative_query)
            launches = {"prior_fa3": previous, "sm_tuned_dual_fa3": control_launch(case, **selected), "native_direct_pdl": lambda: plan.run(case.query, case.key, case.value, case.geometry["scale"], case.output)}
            report["cases"].append(dict(name=name, sha256=case.source_sha256, previous_baseline=baseline, selected_sm_settings=selected))
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
            print("CAPTURED_SM_CASE_DONE", name, selected, flush=True)
            del case, expected, controls, previous, executor, plan, launches, entries, launch
            torch.cuda.empty_cache()
    if len(report["records"]) != len(report["cases"]) * 12:
        raise AssertionError("Incomplete SM-control matrix")
    print("CAPTURED_SM_MATRIX_DONE", len(report["cases"]), len(report["search"]), len(report["records"]), flush=True)


if __name__ == "__main__":
    main()
