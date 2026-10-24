"""Coordinate-tune both FA3 roles and independently remeasure native controls."""

import argparse
import json
from pathlib import Path
import re
import statistics

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from bench_captured_mixed_20261023 import fp32_reference, load_case, measure
from bench_fa3_role_controls_20261023 import role_launches
from native_fa3_fa2_pdl_20261023 import NativePdlExecutor


def control_launch(case, prefix_splits, prefix_packing, row_splits=None, row_packing=None, prefix_margin=0, row_margin=0):
    geometry = case.geometry
    stream = torch.cuda.Stream()
    common = dict(k_cache=case.key, v_cache=case.value, softmax_scale=geometry["scale"], causal=True)

    def launch():
        if row_splits is None:
            flash_attn_with_kvcache(q=case.query, page_table=case.page_table, cache_seqlens=case.cache_lengths, cu_seqlens_q=case.cumulative_query, max_seqlen_q=max(geometry["query_lengths"]), out=case.output, num_splits=prefix_splits, pack_gqa=prefix_packing, sm_margin=prefix_margin, **common)
            return
        current = torch.cuda.current_stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            flash_attn_with_kvcache(q=case.query[:case.prefix_tokens], page_table=case.prefix_pages, cache_seqlens=case.cache_lengths[:case.prefix_rows], cu_seqlens_q=case.cumulative_query[:case.prefix_rows + 1], max_seqlen_q=max(geometry["query_lengths"][:case.prefix_rows]), out=case.output[:case.prefix_tokens], num_splits=prefix_splits, pack_gqa=prefix_packing, sm_margin=prefix_margin, **common)
        flash_attn_with_kvcache(q=case.query[case.prefix_tokens:], page_table=case.page_table[case.prefix_rows:], cache_seqlens=case.cache_lengths[case.prefix_rows:], cu_seqlens_q=case.row_cumulative_query, max_seqlen_q=geometry["row_query_max"], out=case.output[case.prefix_tokens:], num_splits=row_splits, pack_gqa=row_packing, sm_margin=row_margin, **common)
        current.wait_stream(stream)

    return launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prior", type=Path, default=Path("/dev/shm/w/captured_fa3_role_controls_20261023/results.json"))
    parser.add_argument("--roots", type=Path, nargs="+", default=[Path("/dev/shm/w/captured_pair_tuning_20261023/joined"), Path("/dev/shm/w/captured_long_policy_20261023")])
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise ValueError("Fresh control output required")
    torch.backends.cuda.matmul.allow_tf32 = False
    prior = json.loads(arguments.prior.read_text())
    paths = {path.parent.parent.name + "/" + path.name: path for root in arguments.roots for path in root.glob("*/captures/*.pt")}
    report = dict(scope="Coordinate search rather than exhaustive role cross-product; fresh paired validation excludes planning; consult capture protocol for policy freeze/holdout status", cases=[], search=[], records=[])
    for capture in prior["captures"]:
        case = load_case(paths[capture["name"]])
        if case.source_sha256 != capture["sha256"]:
            raise ValueError("Capture digest changed")
        expected = fp32_reference(case)
        old = [row for row in prior["records"] if row["capture"] == capture["name"]]
        means = {arm: statistics.mean(row["milliseconds"] for row in old if row["arm"] == arm) for arm in {row["arm"] for row in old}}
        old_best = min((arm for arm in means if arm.startswith("fa3_roles") or arm.startswith("fa3_full")), key=means.get)
        old_dual = min((arm for arm in means if arm.startswith("fa3_roles")), key=means.get)
        parsed = re.fullmatch(r"fa3_roles_ps(\d+)_pack(None|True|False)_rs(\d+)", old_dual)
        if parsed is None:
            raise ValueError("Unrecognized prior FA3 setting")
        prefix_splits = int(parsed[1])
        prefix_packing = {"None": None, "True": True, "False": False}[parsed[2]]

        def search(phase, settings):
            measured = []
            for setting in settings:
                result = measure(case, control_launch(case, **setting), expected, "graph", 10)
                result.update(capture=capture["name"], phase=phase, setting=setting)
                measured.append(result)
                report["search"].append(result)
            return min(measured, key=lambda row: row["milliseconds"])["setting"]

        best_row = search("rows", [dict(prefix_splits=prefix_splits, prefix_packing=prefix_packing, row_splits=splits, row_packing=packing) for splits in (0, 1, 2, 4, 8, 16, 32, 64) for packing in (None, False, True)])
        best_dual = search("prefill", [dict(prefix_splits=splits, prefix_packing=packing, row_splits=best_row["row_splits"], row_packing=best_row["row_packing"]) for splits in (0, 1, 2, 4, 8, 16, 32) for packing in (None, False, True)])
        best_full = search("full", [dict(prefix_splits=splits, prefix_packing=packing) for splits in (0, 1, 2, 4, 8, 16, 32) for packing in (None, False, True)])
        controls = role_launches(case)
        launches = {"prior_fa3": controls[old_best], "tuned_dual_fa3": control_launch(case, **best_dual), "tuned_full_fa3": control_launch(case, **best_full)}
        plans = {}
        for precision in ("bf16", "fp32"):
            executor = NativePdlExecutor(case.geometry["heads"], case.geometry["kv_heads"], case.query.device, adaptive_splits=True, policy_version="geometry_v2", fp32_partials=precision == "fp32", workspace_bytes=512 << 20)
            plan = executor.plan(case.geometry["query_lengths"], case.geometry["prefix_lengths"], case.page_table, case.cache_lengths, case.cumulative_query)
            plans[precision] = plan
            for pdl in (False, True):
                launches[f"native_{precision}_{'pdl' if pdl else 'serial'}"] = lambda plan=plan, pdl=pdl: plan.run(case.query, case.key, case.value, case.geometry["scale"], case.output, pdl=pdl)
            launches[f"native_{precision}_serial"]()
            snapshot = case.output.clone()
            launches[f"native_{precision}_pdl"]()
            if not torch.equal(case.output, snapshot):
                raise AssertionError("Native serial/PDL mismatch")
        report["cases"].append(dict(name=capture["name"], sha256=case.source_sha256, prior_control=old_best, row_selection=best_row, dual_selection=best_dual, full_selection=best_full, serial_pdl_exact=True))
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
        print("FA3_ROW_PACKING_CASE_DONE", capture["name"], best_dual, best_full, flush=True)
        del case, expected, controls, launches, plans, plan, executor, entries, launch, snapshot
        torch.cuda.empty_cache()
    if len(report["search"]) != len(report["cases"]) * 66 or len(report["records"]) != len(report["cases"]) * 28:
        raise AssertionError("Incomplete row-packing control matrix")
    print("FA3_ROW_PACKING_MATRIX_DONE", len(report["cases"]), len(report["search"]), len(report["records"]), flush=True)


if __name__ == "__main__":
    main()
