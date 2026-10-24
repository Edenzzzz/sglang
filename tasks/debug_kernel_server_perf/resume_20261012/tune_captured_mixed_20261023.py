"""SplitKV tuning on unchanged real serving captures, with matched controls."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from bench_captured_mixed_20261023 import fp32_reference, launches_for, load_case, measure
from native_fa3_fa2_pdl_20261023 import NativePdlExecutor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--resume-from", type=Path)
    arguments = parser.parse_args()
    arguments.output_root.mkdir(exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    for mode in ("two", "four"):
        source = arguments.capture_root / ("mix32_" + mode)
        target = arguments.output_root / ("mix32_" + mode)
        target.mkdir()
        files = sorted((source / "captures").glob("*.pt"))
        if not (source / "capture_complete").exists() or not files:
            raise ValueError("Missing validated serving captures")
        report = dict(
            scope="Same captured serving inputs and physical KV layout; per-case tuning sweep, not a deployable oracle",
            gpu=torch.cuda.get_device_name(0), captures=[], records=[],
            expected_records=len(files) * 100,
            executor_sha256=hashlib.sha256(Path(__file__).with_name("native_fa3_fa2_pdl_20261023.py").read_bytes()).hexdigest(),
        )
        prior = None
        if arguments.resume_from and (arguments.resume_from / ("mix32_" + mode) / "kernel_results.json").exists():
            prior = json.loads((arguments.resume_from / ("mix32_" + mode) / "kernel_results.json").read_text())
            report["prior_executor_sha256"] = prior["executor_sha256"]
        for path in files:
            if prior:
                records = [row for row in prior["records"] if row["capture"] == path.name]
                captures = [row for row in prior["captures"] if row["file"] == path.name]
                if len(records) == 100 and len(captures) == 1:
                    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
                    if digest != captures[0]["sha256"]:
                        raise ValueError("Cannot reuse timings after captured inputs change")
                    report["records"].extend(records)
                    report["captures"].extend(captures)
                    (target / "kernel_results.json").write_text(json.dumps(report, indent=2) + "\n")
                    print("REUSED_CAPTURED_TIMINGS", mode, path.name, flush=True)
                    continue
            case = load_case(path)
            geometry = case.geometry
            expected = fp32_reference(case)
            launches = launches_for(case)
            plans = []
            for prefix_splits in (1, 2, 4, 8):
                for row_split in (None, 1024, 4096, 8192):
                    if prefix_splits == 1 and row_split is None:
                        continue
                    executor = NativePdlExecutor(
                        geometry["heads"], geometry["kv_heads"], case.query.device,
                        prefix_splits=prefix_splits, row_split_size=row_split,
                        workspace_bytes=(512 if row_split is not None else 128) << 20,
                    )
                    plan = executor.plan(
                        geometry["query_lengths"], geometry["prefix_lengths"], case.page_table,
                        case.cache_lengths, case.cumulative_query,
                    )
                    plans.append(plan)
                    name = f"native_pdl_ps{prefix_splits}_rs{row_split or 'auto'}"
                    launches[name] = lambda plan=plan: plan.run(
                        case.query, case.key, case.value, geometry["scale"], case.output
                    )
            report["captures"].append(dict(file=path.name, sha256=case.source_sha256, geometry=geometry))
            for style in ("eager", "graph"):
                for order in ("forward", "reverse"):
                    entries = list(launches.items())
                    if order == "reverse":
                        entries.reverse()
                    for arm, launch in entries:
                        result = measure(case, launch, expected, style, 20)
                        result.update(capture=path.name, mode=mode, arm=arm, style=style, order=order)
                        report["records"].append(result)
                        (target / "kernel_results.json").write_text(json.dumps(report, indent=2) + "\n")
                        print("CAPTURED_TUNING", mode, path.name, arm, style, order, result["milliseconds"], flush=True)
            del case, expected, launches, entries, launch, plans, plan, executor
            torch.cuda.empty_cache()
        if len(report["records"]) != len(files) * 100:
            raise AssertionError("Incomplete captured tuning matrix")
        print("CAPTURED_TUNING_MODE_DONE", mode, len(files), len(report["records"]), flush=True)
    print("CAPTURED_TUNING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
