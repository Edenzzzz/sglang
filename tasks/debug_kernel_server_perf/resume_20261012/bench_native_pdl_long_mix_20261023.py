"""Attention-only PDL comparison against tuned FA3 and two-stream controls."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from native_fa2_long_mix_20261023 import expected_output, fixture, make_launches, measure
from native_fa3_fa2_pdl_20261023 import NativePdlExecutor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", type=int, default=50)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.iterations < 1:
        raise ValueError("A fresh output path and positive iteration count are required")
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = [
        ("qwen7_short_new", 128, 0, [393216, 393216], 4, 28, 4),
        ("qwen7_two_long", 512, 4096, [393216, 393216], 4, 28, 4),
        ("qwen7_one_deep", 512, 4096, [786432], 4, 28, 4),
        ("qwen7_large_chunk", 2048, 32768, [393216, 393216], 4, 28, 4),
        ("qwen7_decode", 512, 4096, [393216, 393216], 1, 28, 4),
        ("qwen3_verify", 512, 4096, [393216], 4, 32, 8),
        ("qwen3_decode", 512, 4096, [393216], 1, 32, 8),
        ("glm_tp4_verify", 512, 4096, [393216], 4, 24, 2),
        ("glm_tp4_decode", 512, 4096, [393216], 1, 24, 2),
    ]
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    report = dict(
        scope="Per-layer attention; same-output eager/graph comparisons, not serving throughput or ITL",
        gpu=torch.cuda.get_device_name(0), sms=sms, torch=torch.__version__, records=[],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        executor_sha256=hashlib.sha256(Path(__file__).with_name("native_fa3_fa2_pdl_20261023.py").read_bytes()).hexdigest(),
        unsupported=[],
    )
    for case_index, (name, chunk, prefix, agents, row_tokens, heads, kv_heads) in enumerate(cases):
        torch.manual_seed(20261023 + case_index)
        case = fixture(chunk, prefix, agents, row_tokens, heads, kv_heads)
        original_query = case.query.clone()
        original_reference = expected_output(case)
        changed_query = torch.randn_like(case.query)
        case.query.copy_(changed_query)
        changed_reference = expected_output(case)
        case.query.copy_(original_query)
        queries = [chunk] + [row_tokens] * (len(agents) + 4)
        lengths = [prefix + chunk, *agents, *([2048] * 4)]
        prefixes = [length - query for length, query in zip(lengths, queries)]
        controls = make_launches(case, fa3_controls=True)
        launches = {key: controls[key] for key in (
            "single_fa3_split8", "single_fa3_split32",
            "single_fa3_pack_split8", "single_fa3_pack_split32",
            "corun_fa3_fa3_rows_split8", "corun_fa3_fa3_rows_split32",
        )}
        launches["two_stream_fa3_native_fa2"] = make_launches(case)["corun_fa3_fa2_margin0"]
        plans = []
        for margin in (0, 16, 32):
            executor = NativePdlExecutor(heads, kv_heads, case.query.device, sms - margin)
            plan = executor.plan(queries, prefixes, case.page_table, case.cache_lengths, case.cumulative_query)
            if plan is None:
                report["unsupported"].append(dict(case=name, margin=margin, reason="packed row query exceeds 32 or not prefix/suffix layout"))
                continue
            plans.append(plan)
            launches[f"pdl_native_margin{margin}"] = lambda plan=plan: plan.run(
                case.query, case.key, case.value, 128**-0.5, case.output
            )
            if margin == 0:
                launches["serial_native_margin0"] = lambda plan=plan: plan.run(
                    case.query, case.key, case.value, 128**-0.5, case.output, pdl=False
                )
        for style in ("eager", "graph"):
            entries = list(launches.items())
            if style == "graph":
                entries.reverse()
            for arm, launch in entries:
                result = measure(
                    case, launch, style, original_query, changed_query,
                    original_reference, changed_reference, arguments.iterations,
                )
                result.update(case=name, arm=arm, style=style, prefill_query=chunk, prefill_prefix=prefix,
                              agent_contexts=agents, row_query=row_tokens, heads=heads, kv_heads=kv_heads)
                report["records"].append(result)
                arguments.output.write_text(json.dumps(report, indent=2) + "\n")
                print("NATIVE_PDL_LONG_MIX", json.dumps(result), flush=True)
        del case, controls, launches, plans, plan, executor, launch
        del original_query, changed_query, original_reference, changed_reference
        torch.cuda.empty_cache()
    if len(report["records"]) != 190:
        raise AssertionError(f"Incomplete benchmark: {len(report['records'])} records")
    print("NATIVE_PDL_LONG_MIX_COMPLETE records=190", flush=True)


if __name__ == "__main__":
    main()
