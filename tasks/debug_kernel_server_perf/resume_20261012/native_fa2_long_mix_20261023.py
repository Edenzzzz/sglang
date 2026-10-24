"""Attention-only long-context headroom check, not a serving-performance claim.

Compare one native FA3 call with FA3 prefill plus the validated native FA2
row executor. Two-stream arms are not labeled as PDL between the roles.
Both eager and graph timings include producing the complete output tensor.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from load_native_q32 import build
from test_native_q32 import assert_numerical, execute, make_case, reference


def fixture(chunk, prefix, agent_lengths, row_tokens, heads, kv_heads):
    row_lengths = agent_lengths + [2048] * 4
    rows = make_case([row_tokens] * len(row_lengths), row_lengths, heads=heads, kv_heads=kv_heads)
    prefix_length = prefix + chunk
    prefix_key = torch.randn(prefix_length, 1, kv_heads, 128, device="cuda", dtype=torch.bfloat16)
    prefix_value = torch.randn_like(prefix_key)
    prefix_query = torch.randn(chunk, heads, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.cat((prefix_key, rows.key))
    value = torch.cat((prefix_value, rows.value))
    query = torch.cat((prefix_query, rows.query))
    output = torch.empty_like(query)
    rows.key, rows.value, rows.query, rows.output = key[prefix_length:], value[prefix_length:], query[chunk:], output[chunk:]
    arguments = list(rows.arguments)
    arguments[0], arguments[1], arguments[2], arguments[7] = rows.query, rows.key, rows.value, rows.output
    rows.arguments = tuple(arguments)
    page_table = torch.zeros(len(row_lengths) + 1, max(prefix_length, *row_lengths), device="cuda", dtype=torch.int32)
    page_table[0, :prefix_length] = torch.arange(prefix_length, device="cuda", dtype=torch.int32)
    offset = 0
    for index, length in enumerate(row_lengths):
        page_table[index + 1, :length] = rows.page_indices[offset:offset + length] + prefix_length
        offset += length
    query_offsets = [0, chunk] + [chunk + row_tokens * (index + 1) for index in range(len(row_lengths))]
    cache_lengths = torch.tensor([prefix_length, *row_lengths], device="cuda", dtype=torch.int32)
    cumulative_query = torch.tensor(query_offsets, device="cuda", dtype=torch.int32)
    row_cumulative_query = torch.arange(len(row_lengths) + 1, device="cuda", dtype=torch.int32) * row_tokens
    cumulative_key = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32), cache_lengths.cumsum(0, dtype=torch.int32)))
    row_cumulative_key = cumulative_key[1:] - prefix_length
    prefix_case = SimpleNamespace(
        query=query[:chunk], key=key, value=value, heads=heads, kv_heads=kv_heads,
        query_lengths=[chunk], kv_lengths=[prefix_length], page_counts=[prefix_length],
        page_indices=torch.arange(prefix_length, device="cuda", dtype=torch.int32),
    )
    native_tile = 32 if 16 < row_tokens * (heads // kv_heads) <= 32 and rows.wrapper._plan_info[3] == 64 else 0
    return SimpleNamespace(
        query=query, key=key, value=value, output=output, rows=rows, prefix_case=prefix_case,
        page_table=page_table, cache_lengths=cache_lengths, cumulative_query=cumulative_query,
        row_cumulative_query=row_cumulative_query, cumulative_key=cumulative_key,
        row_cumulative_key=row_cumulative_key, chunk=chunk, prefix_length=prefix_length,
        row_tokens=row_tokens, native_tile=native_tile,
    )


def expected_output(case):
    return torch.cat((reference(case.prefix_case), reference(case.rows)))


def make_launches(case, fa3_controls=False):
    common = dict(k_cache=case.key, v_cache=case.value, causal=True)

    def full(num_splits=0, pack_gqa=None, native_metadata=False):
        flash_attn_with_kvcache(
            q=case.query, page_table=case.page_table, cache_seqlens=case.cache_lengths,
            cu_seqlens_q=case.cumulative_query, max_seqlen_q=case.chunk, out=case.output,
            cu_seqlens_k_new=case.cumulative_key if native_metadata else None,
            num_splits=num_splits, pack_gqa=pack_gqa, **common,
        )

    def prefill(margin=0):
        flash_attn_with_kvcache(
            q=case.query[:case.chunk], page_table=case.page_table[:1, :case.prefix_length],
            cache_seqlens=case.cache_lengths[:1], cu_seqlens_q=case.cumulative_query[:2],
            max_seqlen_q=case.chunk, sm_margin=margin, out=case.output[:case.chunk], **common,
        )

    def fa3_rows(num_splits=0):
        flash_attn_with_kvcache(
            q=case.rows.query, page_table=case.page_table[1:], cache_seqlens=case.cache_lengths[1:],
            cu_seqlens_q=case.row_cumulative_query, max_seqlen_q=case.row_tokens,
            out=case.rows.output, num_splits=num_splits, **common,
        )

    def fa2_rows():
        execute(case.rows, case.native_tile, pdl=False)

    def serial():
        prefill()
        fa2_rows()

    side = torch.cuda.Stream()

    def concurrent(rows, margin):
        current = torch.cuda.current_stream()
        side.wait_stream(current)
        with torch.cuda.stream(side):
            prefill(margin)
        rows()
        current.wait_stream(side)

    launches = {"single_fa3": full, "serial_fa3_fa2": serial}
    for margin in (0, 16, 32):
        if margin < torch.cuda.get_device_properties(0).multi_processor_count:
            launches[f"corun_fa3_fa2_margin{margin}"] = lambda margin=margin: concurrent(fa2_rows, margin)
    launches["corun_fa3_fa3_margin0"] = lambda: concurrent(fa3_rows, 0)
    if fa3_controls:
        launches = {
            "single_fa3_auto": full,
            "single_fa3_backend_metadata": lambda: full(native_metadata=True),
            "single_fa3_pack": lambda: full(pack_gqa=True, native_metadata=True),
            "single_fa3_split8": lambda: full(num_splits=8, native_metadata=True),
            "single_fa3_split32": lambda: full(num_splits=32, native_metadata=True),
            "single_fa3_pack_split8": lambda: full(num_splits=8, pack_gqa=True, native_metadata=True),
            "single_fa3_pack_split32": lambda: full(num_splits=32, pack_gqa=True, native_metadata=True),
            "corun_fa3_fa3_rows_split8": lambda: concurrent(lambda: fa3_rows(8), 0),
            "corun_fa3_fa3_rows_split32": lambda: concurrent(lambda: fa3_rows(32), 0),
        }
    return launches


def measure(case, launch, style, original_query, changed_query, original_reference, changed_reference, iterations):
    case.query.copy_(original_query)
    for warmup in range(5):
        launch()
    torch.cuda.synchronize()
    graph = None
    if style == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        runner = graph.replay
    else:
        runner = launch
    case.output.fill_(float("nan"))
    runner()
    torch.cuda.synchronize()
    worst = assert_numerical(case.output, original_reference)
    case.query.copy_(changed_query)
    case.output.fill_(float("nan"))
    runner()
    torch.cuda.synchronize()
    worst = max(worst, assert_numerical(case.output, changed_reference))
    case.query.copy_(original_query)
    for warmup in range(5):
        runner()
    torch.cuda.synchronize()
    samples = []
    for repeat in range(3):
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        for iteration in range(iterations):
            runner()
        finished.record()
        finished.synchronize()
        samples.append(started.elapsed_time(finished) / iterations)
    return dict(milliseconds=statistics.median(samples), samples_ms=samples, worst_head_relative_error=worst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--fa3-controls", action="store_true")
    arguments = parser.parse_args()
    if arguments.iterations < 1 or arguments.output.exists():
        raise ValueError("Iterations must be positive and the output path must be fresh")
    torch.backends.cuda.matmul.allow_tf32 = False
    build()
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
    report = dict(
        gpu=torch.cuda.get_device_name(0), sms=torch.cuda.get_device_properties(0).multi_processor_count,
        torch=torch.__version__, flashinfer=importlib.metadata.version("flashinfer-python"),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="per-layer attention only; eager and graph timings; two streams, not inter-role PDL",
        fa3_controls=arguments.fa3_controls, records=[],
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
        launches = make_launches(case, arguments.fa3_controls)
        for style in ("eager", "graph"):
            entries = list(launches.items())
            if style == "graph":
                entries.reverse()
            for arm, launch in entries:
                result = measure(case, launch, style, original_query, changed_query,
                                 original_reference, changed_reference, arguments.iterations)
                result.update(case=name, arm=arm, style=style, prefill_tokens=chunk, prefill_prefix=prefix,
                              agent_contexts=agents, chat_contexts=[2048] * 4, row_query_tokens=row_tokens,
                              heads=heads, kv_heads=kv_heads, native_tile_override=case.native_tile,
                              planned_fa2_tile=int(case.rows.wrapper._plan_info[3]))
                report["records"].append(result)
                arguments.output.write_text(json.dumps(report, indent=2) + "\n")
                print("LONG_MIX_PASS", json.dumps(result), flush=True)
        del case, launches, original_query, changed_query, original_reference, changed_reference
        torch.cuda.empty_cache()
    expected_records = 162 if arguments.fa3_controls else 108
    assert len(report["records"]) == expected_records
    print(f"NATIVE_FA2_LONG_MIX_COMPLETE records={expected_records}", flush=True)


if __name__ == "__main__":
    main()
