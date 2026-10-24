"""Decompose and tune the actual cached-prefill regression without changing policy."""

import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from bench_captured_mixed_20261023 import fp32_reference, load_case, measure
from bench_fa3_row_packing_20261023 import control_launch
from native_fa3_fa2_pdl_20261023 import NativePdlExecutor


def native_prefix(case, plan):
    shape, executor = plan.shape, plan.executor
    arguments = (
        case.query[:shape.prefix_tokens], case.key[:, 0], case.value[:, 0], case.output[:shape.prefix_tokens], shape.lse,
        shape.prefix_indptr, shape.prefix_lengths, shape.prefix_pages, shape.partial_output,
        shape.partial_lse, shape.split_counts, shape.tile_counter, shape.prefix_splits,
        shape.maximum_prefix_query, executor.sm_budget, case.geometry["scale"],
    )
    torch.ops.stock_fa3_pdl.run(*arguments, 1, executor.empty_timeline)
    if shape.prefix_splits > 1:
        torch.ops.stock_fa3_pdl.run(*arguments, 3, executor.empty_timeline)


def native_rows(case, plan):
    executor, shape = plan.executor, plan.shape
    wrapper = executor.wrapper
    namespace = "native_fa2_q32_fp32_direct" if shape.direct_row_output else "native_fa2_q32_fp32"
    output = case.output[shape.prefix_tokens:] if shape.direct_row_output else shape.float_row_output
    getattr(torch.ops, namespace).run(
        case.query[shape.prefix_tokens:], case.key, case.value, wrapper._qo_indptr_buf,
        wrapper._paged_kv_indptr_buf, wrapper._paged_kv_indices_buf, wrapper._paged_kv_last_page_len_buf,
        output, executor.empty_lse, wrapper._float_workspace_buffer, wrapper._int_workspace_buffer,
        list(wrapper._plan_info), case.geometry["scale"], False, False, executor.empty_timeline,
        shape.native_tile, shape.maximum_row_query,
    )
    if not shape.direct_row_output:
        case.output[shape.prefix_tokens:].copy_(output)


def public_role(case, prefix, settings):
    geometry = case.geometry
    selection = slice(None, case.prefix_rows) if prefix else slice(case.prefix_rows, None)
    token_selection = slice(None, case.prefix_tokens) if prefix else slice(case.prefix_tokens, None)
    flash_attn_with_kvcache(
        q=case.query[token_selection], k_cache=case.key, v_cache=case.value,
        page_table=case.prefix_pages if prefix else case.page_table[selection],
        cache_seqlens=case.cache_lengths[selection],
        cu_seqlens_q=case.cumulative_query[:case.prefix_rows + 1] if prefix else case.row_cumulative_query,
        max_seqlen_q=max(geometry["query_lengths"][selection]), softmax_scale=geometry["scale"], causal=True,
        num_splits=settings["prefix_splits" if prefix else "row_splits"],
        pack_gqa=settings["prefix_packing" if prefix else "row_packing"],
        sm_margin=settings["prefix_margin" if prefix else "row_margin"], out=case.output[token_selection],
    )


def main():
    root = Path(__file__).resolve().parent
    output = root / "results.json"
    if output.exists():
        raise ValueError("Fresh cached-prefill diagnostic required")
    torch.backends.cuda.matmul.allow_tf32 = False
    prior = json.loads(Path("/dev/shm/w/captured_sm_controls_20261023/results.json").read_text())
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    report = dict(scope="Diagnosis/tuning on two already measured captures; not a replacement policy or held-out validation", cases=[], records=[], search=[])
    for mode in ("two", "four"):
        name = f"qwen25/mix32_{mode}/cached_small.pt"
        metadata = next(capture for capture in prior["cases"] if capture["name"] == name)
        case = load_case(Path(f"/dev/shm/w/captured_pair_tuning_20261023/joined/mix32_{mode}/captures/cached_small.pt"))
        if case.source_sha256 != metadata["sha256"]:
            raise ValueError("Captured tensors changed")
        expected = fp32_reference(case)
        geometry = case.geometry

        def make_plan(splits=None, budget=None, row_chunk=None):
            executor = NativePdlExecutor(
                geometry["heads"], geometry["kv_heads"], case.query.device,
                adaptive_splits=splits is None, policy_version="geometry_v2",
                prefix_splits=splits or 1, sm_budget=budget, row_split_size=row_chunk,
                fp32_partials=True, direct_bf16_output=True, workspace_bytes=512 << 20,
            )
            return executor.plan(geometry["query_lengths"], geometry["prefix_lengths"], case.page_table, case.cache_lengths, case.cumulative_query)

        default = make_plan()
        settings = metadata["selected_sm_settings"]
        role_cases = dict(
            prefix=(SimpleNamespace(output=case.output[:case.prefix_tokens]), expected[:case.prefix_tokens]),
            rows=(SimpleNamespace(output=case.output[case.prefix_tokens:]), expected[case.prefix_tokens:]),
        )
        decomposition = {
            "public_prefix": ("prefix", lambda: public_role(case, True, settings)),
            "public_rows": ("rows", lambda: public_role(case, False, settings)),
            "native_prefix": ("prefix", lambda: native_prefix(case, default)),
            "native_rows": ("rows", lambda: native_rows(case, default)),
        }
        for arm, (role, launch) in decomposition.items():
            role_case, role_expected = role_cases[role]
            result = measure(role_case, launch, role_expected, "graph", 30)
            result.update(capture=name, arm=arm, scope="isolated role")
            report["records"].append(result)
        row_chunk = default.shape.split_settings["row_split_size"]
        search = []
        for splits in (1, 2, 4, 8, 16, 32):
            for margin in (0, 8, 16, 32, 48, 64):
                if margin >= sms:
                    continue
                plan = make_plan(splits, sms - margin, row_chunk)
                launch = lambda: plan.run(case.query, case.key, case.value, geometry["scale"], case.output)
                result = measure(case, launch, expected, "graph", 20)
                result.update(capture=name, prefix_splits=splits, prefix_margin=margin, row_chunk=row_chunk)
                search.append(result)
                report["search"].append(result)
                del plan, launch
        selected = min(search, key=lambda row: row["milliseconds"])
        best = make_plan(selected["prefix_splits"], sms - selected["prefix_margin"], row_chunk)
        launches = {
            "tuned_dual_fa3": control_launch(case, **settings),
            "native_geometry_v2": lambda: default.run(case.query, case.key, case.value, geometry["scale"], case.output),
            "native_tuned": lambda: best.run(case.query, case.key, case.value, geometry["scale"], case.output),
        }
        for style in ("eager", "graph"):
            for order in ("forward", "reverse"):
                entries = list(launches.items())
                if order == "reverse":
                    entries.reverse()
                for arm, launch in entries:
                    result = measure(case, launch, expected, style, 30)
                    result.update(capture=name, arm=arm, style=style, order=order, scope="complete mixed attention")
                    report["records"].append(result)
        report["cases"].append(dict(name=name, sha256=case.source_sha256, geometry=geometry, frozen_native_policy=default.shape.split_settings, public_settings=settings, best_native_setting=selected))
        output.write_text(json.dumps(report, indent=2) + "\n")
        print("CACHED_PREFILL_DIAGNOSTIC_DONE", name, selected, flush=True)
        del case, expected, default, best, launches, decomposition, entries, launch, role_cases, role_case, role_expected
        torch.cuda.empty_cache()
    if len(report["search"]) != 72 or len(report["records"]) != 32:
        raise AssertionError("Incomplete cached-prefill diagnostic")


if __name__ == "__main__":
    main()
