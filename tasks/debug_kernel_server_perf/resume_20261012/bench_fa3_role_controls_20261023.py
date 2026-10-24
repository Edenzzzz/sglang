"""Stronger captured-input controls: tune FA3 prefill and row splits separately."""

import argparse
import json
from pathlib import Path
import statistics

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from bench_captured_mixed_20261023 import fp32_reference, launches_for, load_case, measure


def role_launches(case):
    geometry = case.geometry
    side = torch.cuda.Stream()
    common = dict(k_cache=case.key, v_cache=case.value, softmax_scale=geometry["scale"], causal=True)

    def launch(prefix_splits, packing, row_splits):
        current = torch.cuda.current_stream()
        side.wait_stream(current)
        with torch.cuda.stream(side):
            flash_attn_with_kvcache(
                q=case.query[:case.prefix_tokens], page_table=case.prefix_pages,
                cache_seqlens=case.cache_lengths[:case.prefix_rows],
                cu_seqlens_q=case.cumulative_query[:case.prefix_rows + 1],
                max_seqlen_q=max(geometry["query_lengths"][:case.prefix_rows]),
                out=case.output[:case.prefix_tokens], num_splits=prefix_splits,
                pack_gqa=packing, **common,
            )
        flash_attn_with_kvcache(
            q=case.query[case.prefix_tokens:], page_table=case.page_table[case.prefix_rows:],
            cache_seqlens=case.cache_lengths[case.prefix_rows:],
            cu_seqlens_q=case.row_cumulative_query, max_seqlen_q=geometry["row_query_max"],
            out=case.output[case.prefix_tokens:], num_splits=row_splits, **common,
        )
        current.wait_stream(side)

    entries = launches_for(case, geometry_policy=True)
    for prefix_splits in (0, 1, 2, 4, 8, 16, 32):
        for packing in (None, False, True):
            for row_splits in (0, 8, 32):
                name = f"fa3_roles_ps{prefix_splits}_pack{packing}_rs{row_splits}"
                entries[name] = lambda prefix_splits=prefix_splits, packing=packing, row_splits=row_splits: launch(prefix_splits, packing, row_splits)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("roots", type=Path, nargs="+")
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise ValueError("Role-control output must be fresh")
    torch.backends.cuda.matmul.allow_tf32 = False
    files = sorted(path for root in arguments.roots for path in root.glob("*/captures/*.pt"))
    if not files:
        raise ValueError("No captured inputs")
    report = dict(
        scope="Same captures; extended dual-FA3 prefix SplitKV and GQA-packing controls; graph-only tuning, not end-to-end serving",
        gpu=torch.cuda.get_device_name(0), captures=[], records=[],
    )
    for path in files:
        if not (path.parent.parent / "capture_complete").exists():
            raise ValueError("Capture workload was not validated")
        case = load_case(path)
        expected = fp32_reference(case)
        launches = role_launches(case)
        name = path.parent.parent.name + "/" + path.name
        report["captures"].append(dict(
            name=name, sha256=case.source_sha256, geometry=case.geometry,
            geometry_policy=case.geometry_policy,
        ))
        for order in ("forward", "reverse"):
            entries = list(launches.items())
            if order == "reverse":
                entries.reverse()
            for arm, launch in entries:
                result = measure(case, launch, expected, "graph", 10)
                result.update(capture=name, arm=arm, order=order, style="graph")
                report["records"].append(result)
                arguments.output.write_text(json.dumps(report, indent=2) + "\n")
        rows = [row for row in report["records"] if row["capture"] == name]
        means = {arm: statistics.mean(row["milliseconds"] for row in rows if row["arm"] == arm) for arm in launches}
        best = min((arm for arm in means if arm.startswith("fa3_roles")), key=means.get)
        print("FA3_ROLE_CONTROLS_CASE", name, best, means[best], "geometry_v1", means["native_pdl_geometry_v1"], flush=True)
        del case, expected, launches, entries, launch
        torch.cuda.empty_cache()
    if len(report["records"]) != len(files) * 150:
        raise AssertionError("Incomplete FA3 role-control matrix")
    print("FA3_ROLE_CONTROLS_DONE", len(files), len(report["records"]), flush=True)


if __name__ == "__main__":
    main()
