"""Fresh paired timings after selecting settings; exact same captured inputs."""

import argparse
import json
from pathlib import Path
import re
import statistics

import torch

from bench_captured_mixed_20261023 import fp32_reference, launches_for, load_case, measure
from native_fa3_fa2_pdl_20261023 import NativePdlExecutor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("tuning_root", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise ValueError("Validation output must be fresh")
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(
        scope="Settings selected using prior tuning; fresh opposite-order measurements on the same captures, not held-out workload generalization",
        gpu=torch.cuda.get_device_name(0), cases=[], records=[],
    )
    for mode in ("two", "four"):
        tuned = json.loads((arguments.tuning_root / ("mix32_" + mode) / "kernel_results.json").read_text())
        if len(tuned["records"]) != tuned["expected_records"]:
            raise ValueError("Tuning matrix is incomplete")
        for capture in tuned["captures"]:
            path = arguments.capture_root / ("mix32_" + mode) / "captures" / capture["file"]
            case = load_case(path)
            if case.source_sha256 != capture["sha256"]:
                raise ValueError("Captured inputs changed after tuning")
            selected = [row for row in tuned["records"] if row["capture"] == path.name and row["style"] == "graph"]
            means = {arm: statistics.mean(row["milliseconds"] for row in selected if row["arm"] == arm) for arm in {row["arm"] for row in selected}}
            single = min((arm for arm in means if arm.startswith("fa3_full")), key=means.get)
            dual = min((arm for arm in means if arm.startswith("fa3_dual")), key=means.get)
            native = min((arm for arm in means if arm.startswith("native_pdl")), key=means.get)
            if native == "native_pdl":
                splits, row_split = 1, None
            else:
                match = re.fullmatch(r"native_pdl_ps(\d+)_rs(auto|\d+)", native)
                if match is None:
                    raise ValueError("Unknown native tuning arm")
                splits = int(match.group(1))
                row_split = None if match.group(2) == "auto" else int(match.group(2))
            geometry = case.geometry
            executor = NativePdlExecutor(
                geometry["heads"], geometry["kv_heads"], case.query.device,
                prefix_splits=splits, row_split_size=row_split, workspace_bytes=512 << 20,
            )
            plan = executor.plan(geometry["query_lengths"], geometry["prefix_lengths"], case.page_table, case.cache_lengths, case.cumulative_query)
            controls = launches_for(case)
            launches = {
                "fa3_single": controls[single], "fa3_dual": controls[dual],
                "native_serial": lambda: plan.run(case.query, case.key, case.value, geometry["scale"], case.output, pdl=False),
                "native_pdl": lambda: plan.run(case.query, case.key, case.value, geometry["scale"], case.output),
            }
            expected = fp32_reference(case)
            launches["native_serial"]()
            serial = case.output.clone()
            launches["native_pdl"]()
            if not torch.equal(serial, case.output):
                raise AssertionError("Matched serial and PDL outputs differ on captured inputs")
            report["cases"].append(dict(
                mode=mode, capture=path.name, sha256=case.source_sha256, geometry=geometry,
                selected_single=single, selected_dual=dual, selected_native=native,
                serial_pdl_bitwise_equal=True,
            ))
            for style in ("eager", "graph"):
                for order in ("forward", "reverse"):
                    entries = list(launches.items())
                    if order == "reverse":
                        entries.reverse()
                    for arm, launch in entries:
                        result = measure(case, launch, expected, style, 20)
                        result.update(mode=mode, capture=path.name, arm=arm, style=style, order=order)
                        report["records"].append(result)
                        arguments.output.write_text(json.dumps(report, indent=2) + "\n")
                        print("CAPTURED_PAIRED_VALIDATION", mode, path.name, arm, style, order, result["milliseconds"], flush=True)
            del case, expected, serial, controls, launches, entries, launch, executor, plan
            torch.cuda.empty_cache()
    if len(report["records"]) != len(report["cases"]) * 16:
        raise AssertionError("Incomplete paired validation")
    print("CAPTURED_PAIRED_VALIDATION_DONE", len(report["cases"]), len(report["records"]), flush=True)


if __name__ == "__main__":
    main()
