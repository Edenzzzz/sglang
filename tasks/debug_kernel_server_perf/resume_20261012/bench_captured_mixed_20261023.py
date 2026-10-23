"""Replay actual serving Q/K/V tensors with their original physical KV layout."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from native_fa3_fa2_pdl_20261023 import NativePdlExecutor


def load_case(path):
    captured = torch.load(path, map_location="cpu", weights_only=True)
    geometry = captured["geometry"]
    query = captured["query"].cuda()
    key = torch.empty_strided(geometry["key_shape"], geometry["key_stride"], dtype=query.dtype, device="cuda")
    value = torch.empty_strided(geometry["value_shape"], geometry["value_stride"], dtype=query.dtype, device="cuda")
    key.fill_(float("nan"))
    value.fill_(float("nan"))
    physical_ids = captured["physical_ids"].cuda()
    key.index_copy_(0, physical_ids, captured["key"].cuda())
    value.index_copy_(0, physical_ids, captured["value"].cuda())
    page_table = captured["page_table"].cuda()
    cache_lengths = captured["cache_lengths"].cuda()
    cumulative_query = captured["cumulative_query"].cuda()
    prefix_rows, prefix_tokens = geometry["prefix_rows"], geometry["prefix_tokens"]
    return SimpleNamespace(
        geometry=geometry, query=query, key=key, value=value,
        output=torch.empty_like(query), page_table=page_table,
        cache_lengths=cache_lengths, cumulative_query=cumulative_query,
        prefix_rows=prefix_rows, prefix_tokens=prefix_tokens,
        row_cumulative_query=cumulative_query[prefix_rows:] - prefix_tokens,
        prefix_pages=page_table[:prefix_rows, :max(geometry["kv_lengths"][:prefix_rows])].contiguous(),
        source_sha256=hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
    )


def fp32_reference(case):
    geometry = case.geometry
    expected = torch.empty(case.query.shape, dtype=torch.float32, device="cuda")
    query_begin = 0
    group = geometry["heads"] // geometry["kv_heads"]
    for request, (query_length, kv_length) in enumerate(zip(geometry["query_lengths"], geometry["kv_lengths"])):
        pages = case.page_table[request, :kv_length].long()
        for kv_head in range(geometry["kv_heads"]):
            keys = case.key[pages, 0, kv_head].float()
            values = case.value[pages, 0, kv_head].float()
            head_begin, head_end = kv_head * group, (kv_head + 1) * group
            for begin in range(0, query_length, 32):
                end = min(begin + 32, query_length)
                queries = case.query[query_begin + begin:query_begin + end, head_begin:head_end].float()
                logits = torch.einsum("thd,kd->htk", queries, keys) * geometry["scale"]
                masked = torch.arange(kv_length, device="cuda")[None, :] > (
                    kv_length - query_length + torch.arange(begin, end, device="cuda")[:, None]
                )
                logits.masked_fill_(masked, -float("inf"))
                expected[query_begin + begin:query_begin + end, head_begin:head_end] = torch.einsum(
                    "htk,kd->thd", logits.softmax(-1), values
                )
        query_begin += query_length
    return expected


def validate(output, expected):
    if not torch.isfinite(output).all():
        raise AssertionError("Captured replay produced nonfinite output")
    relative = (output.float() - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-6)
    worst = float(relative.max())
    if worst >= 0.02:
        raise AssertionError(f"Captured replay FP32 gate failed: worst head relative error {worst}")
    return worst


def launches_for(case, geometry_policy=False, policy_v3=False):
    geometry = case.geometry
    common = dict(k_cache=case.key, v_cache=case.value, softmax_scale=geometry["scale"], causal=True)

    def full(splits=0, packing=None):
        flash_attn_with_kvcache(
            q=case.query, page_table=case.page_table, cache_seqlens=case.cache_lengths,
            cu_seqlens_q=case.cumulative_query, max_seqlen_q=max(geometry["query_lengths"]),
            out=case.output, num_splits=splits, pack_gqa=packing, **common,
        )

    def prefix():
        flash_attn_with_kvcache(
            q=case.query[:case.prefix_tokens], page_table=case.prefix_pages,
            cache_seqlens=case.cache_lengths[:case.prefix_rows],
            cu_seqlens_q=case.cumulative_query[:case.prefix_rows + 1],
            max_seqlen_q=max(geometry["query_lengths"][:case.prefix_rows]),
            out=case.output[:case.prefix_tokens], **common,
        )

    def rows(splits):
        flash_attn_with_kvcache(
            q=case.query[case.prefix_tokens:], page_table=case.page_table[case.prefix_rows:],
            cache_seqlens=case.cache_lengths[case.prefix_rows:], cu_seqlens_q=case.row_cumulative_query,
            max_seqlen_q=geometry["row_query_max"], out=case.output[case.prefix_tokens:],
            num_splits=splits, **common,
        )

    side_stream = torch.cuda.Stream()

    def dual(splits):
        current = torch.cuda.current_stream()
        side_stream.wait_stream(current)
        with torch.cuda.stream(side_stream):
            prefix()
        rows(splits)
        current.wait_stream(side_stream)

    executor = NativePdlExecutor(geometry["heads"], geometry["kv_heads"], case.query.device)
    plan = executor.plan(
        geometry["query_lengths"], geometry["prefix_lengths"], case.page_table,
        case.cache_lengths, case.cumulative_query,
    )
    if plan is None:
        raise ValueError("Captured batch is not native-split eligible")
    entries = {"fa3_full_default": full}
    for splits in (8, 32):
        entries[f"fa3_full_split{splits}"] = lambda splits=splits: full(splits)
        entries[f"fa3_full_pack_split{splits}"] = lambda splits=splits: full(splits, True)
    for splits in (0, 8, 32):
        entries[f"fa3_dual_split{splits}"] = lambda splits=splits: dual(splits)
    entries["native_serial"] = lambda: plan.run(case.query, case.key, case.value, geometry["scale"], case.output, pdl=False)
    entries["native_pdl"] = lambda: plan.run(case.query, case.key, case.value, geometry["scale"], case.output)
    if geometry_policy:
        adaptive = NativePdlExecutor(
            geometry["heads"], geometry["kv_heads"], case.query.device,
            adaptive_splits=True, workspace_bytes=512 << 20,
        )
        adaptive_plan = adaptive.plan(
            geometry["query_lengths"], geometry["prefix_lengths"], case.page_table,
            case.cache_lengths, case.cumulative_query,
        )
        case.geometry_policy = adaptive_plan.shape.split_settings
        entries["native_serial_geometry_v1"] = lambda: adaptive_plan.run(
            case.query, case.key, case.value, geometry["scale"], case.output, pdl=False
        )
        entries["native_pdl_geometry_v1"] = lambda: adaptive_plan.run(
            case.query, case.key, case.value, geometry["scale"], case.output
        )
    if policy_v3:
        # The configuration the audited serving runs used (2026-10-23): geometry_v3, FP32 partials, direct BF16 output.
        v3 = NativePdlExecutor(
            geometry["heads"], geometry["kv_heads"], case.query.device, adaptive_splits=True,
            policy_version="geometry_v3", fp32_partials=True, direct_bf16_output=True, packed_query_limit=64,
            workspace_bytes=512 << 20,
        )
        v3_plan = v3.plan(
            geometry["query_lengths"], geometry["prefix_lengths"], case.page_table,
            case.cache_lengths, case.cumulative_query,
        )
        entries["native_serial_geometry_v3"] = lambda: v3_plan.run(
            case.query, case.key, case.value, geometry["scale"], case.output, pdl=False
        )
        entries["native_pdl_geometry_v3"] = lambda: v3_plan.run(
            case.query, case.key, case.value, geometry["scale"], case.output
        )
    return entries


def measure(case, launch, expected, style, iterations):
    for repeat in range(3):
        launch()
    torch.cuda.synchronize()
    if style == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        runner = graph.replay
    else:
        runner = launch
    errors = []
    for repeat in range(2):
        case.output.fill_(float("nan"))
        runner()
        torch.cuda.synchronize()
        errors.append(validate(case.output, expected))
    samples = []
    for repeat in range(3):
        started, finished = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started.record()
        for iteration in range(iterations):
            runner()
        finished.record()
        finished.synchronize()
        samples.append(started.elapsed_time(finished) / iterations)
    return dict(milliseconds=statistics.median(samples), samples_ms=samples, worst_head_relative_error=max(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--geometry-policy", action="store_true")
    parser.add_argument("--policy-v3", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.iterations < 1:
        raise ValueError("A fresh output and positive iterations are required")
    files = sorted(arguments.captures.glob("*.pt"))
    if not files:
        raise ValueError("No real captured inputs")
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(
        scope="Actual serving tensors and physical KV layout; attention launch only, excludes capture and once-per-forward planning",
        gpu=torch.cuda.get_device_name(0), records=[], captures=[],
        expected_records=len(files) * (48 if arguments.geometry_policy else 40),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        executor_sha256=hashlib.sha256(Path(__file__).with_name("native_fa3_fa2_pdl_20261023.py").read_bytes()).hexdigest(),
    )
    for path in files:
        case = load_case(path)
        expected = fp32_reference(case)
        launches = launches_for(case, geometry_policy=arguments.geometry_policy, policy_v3=arguments.policy_v3)
        report["captures"].append(dict(
            file=path.name, sha256=case.source_sha256, geometry=case.geometry,
            geometry_policy=getattr(case, "geometry_policy", None),
        ))
        for style in ("eager", "graph"):
            for order in ("forward", "reverse"):
                entries = list(launches.items())
                if order == "reverse":
                    entries.reverse()
                for arm, launch in entries:
                    result = measure(case, launch, expected, style, arguments.iterations)
                    result.update(capture=path.name, mode=case.geometry["mode"], arm=arm, style=style, order=order)
                    report["records"].append(result)
                    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
                    print("CAPTURED_KERNEL_BENCH", json.dumps(result), flush=True)
        del case, expected, launches, entries, launch
        torch.cuda.empty_cache()
    if len(report["records"]) != report["expected_records"]:
        raise AssertionError("Incomplete captured kernel matrix")
    print("CAPTURED_KERNEL_BENCH_DONE", len(files), len(report["records"]), flush=True)


if __name__ == "__main__":
    main()
