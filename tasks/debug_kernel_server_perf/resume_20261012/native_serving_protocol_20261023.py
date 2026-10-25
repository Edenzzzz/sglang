"""Auditable request identities and token-delivery metrics for matched serving."""

import ast
import math
import re
import statistics

from workload_protocol import token_arrivals, worst_window_tpot_ms


def replace_once(source, original, replacement):
    if source.count(original) != 1:
        raise ValueError(f"Unexpected benchmark source anchor: {original}")
    return source.replace(original, replacement, 1)


def patch_client_source(source):
    if '"request_keys": [request_key(' in source:
        if '"token_events": [output.token_events for output in outputs]' not in source:
            raise ValueError("Partial request audit installation")
        return source
    source = replace_once(
        source, "    start_time: float = 0.0\n    cached_tokens: int = 0",
        "    start_time: float = 0.0\n    token_events: List[Tuple[int, float]] = field(default_factory=list)\n    cached_tokens: int = 0",
    )
    source = replace_once(
        source, '                                output_len = data["meta_info"]["completion_tokens"]\n',
        '                                output_len = data["meta_info"]["completion_tokens"]\n'
        '                                if output_len > last_output_len:\n'
        '                                    output.token_events.append((int(output_len), timestamp - st))\n',
    )
    source = replace_once(
        source, '    result_details = {\n        "input_lens":',
        '''    from sglang.benchmark.workload_protocol import request_key

    result_details = {
        "request_keys": [request_key(request.prompt, request.output_len) for request in benchmark_requests],
        "requested_output_lens": [int(request.output_len) for request in benchmark_requests],
        "request_start_times": [output.start_time for output in outputs],
        "request_end_times": [output.start_time + output.latency for output in outputs],
        "token_events": [output.token_events for output in outputs],
        "input_lens":''',
    )
    ast.parse(source)
    return source


def summarize_stream(data, expected):
    fields = ("request_keys", "requested_output_lens", "output_lens", "ttfts", "token_events", "errors")
    if data["completed"] != expected or any(len(data[field]) != expected for field in fields):
        raise ValueError("Incomplete request or token-event arrays")
    if any(data["errors"]) or not math.isfinite(data["duration"]) or data["duration"] <= 0:
        raise ValueError("Failed requests or invalid duration")
    rows = []
    intervals = []
    for index in range(expected):
        output_length = data["output_lens"][index]
        if output_length != data["requested_output_lens"][index]:
            raise ValueError("A request did not produce its requested output length")
        arrivals = token_arrivals(data["token_events"][index], output_length)
        ttft_ms = data["ttfts"][index] * 1000
        if not math.isfinite(ttft_ms) or ttft_ms <= 0:
            raise ValueError("Invalid successful-request TTFT")
        deltas = [(later - earlier) * 1000 for earlier, later in zip(arrivals, arrivals[1:])]
        intervals.extend(deltas)
        rows.append(dict(
            request_key=data["request_keys"][index], output_tokens=output_length,
            ttft_ms=ttft_ms, tpot_ms=statistics.mean(deltas) if deltas else 0,
            worst_tpot_10_ms=worst_window_tpot_ms(data["token_events"][index], output_length),
        ))
    verify_count = sum(value or 0 for value in data.get("spec_verify_ct", []))
    accepted = sum(value or 0 for value in data.get("spec_num_correct_drafts", []))
    return dict(
        requests=expected, duration_s=data["duration"], output_tokens=sum(data["output_lens"]),
        output_tok_s=sum(data["output_lens"]) / data["duration"],
        mean_ttft_ms=statistics.mean(row["ttft_ms"] for row in rows),
        mean_request_tpot_ms=statistics.mean(row["tpot_ms"] for row in rows),
        token_weighted_mean_itl_ms=statistics.mean(intervals) if intervals else 0,
        mean_worst_tpot_10_ms=statistics.mean(row["worst_tpot_10_ms"] for row in rows),
        tokens_per_verify=(accepted + verify_count) / verify_count if verify_count else None,
        metric_scope="Actual cumulative token arrivals; simultaneous speculative tokens have zero delivery interval; no SLO attainment claim",
        requests_detail=rows,
    )


def assert_same_requests(first, second):
    if set(first["tenants"]) != set(second["tenants"]):
        raise ValueError("Arms served different tenant sets")
    for tenant in sorted(first["tenants"]):
        left = [(row["request_key"], row["output_tokens"]) for row in first["tenants"][tenant]["requests_detail"]]
        right = [(row["request_key"], row["output_tokens"]) for row in second["tenants"][tenant]["requests_detail"]]
        if left != right:
            raise ValueError(f"Unmatched {tenant} request identities or output lengths")


def measured_step_trace(server_log, first_request, last_request):
    if not math.isfinite(first_request) or not math.isfinite(last_request) or last_request <= first_request:
        raise ValueError("Invalid measured trace window")
    headers, measured = [], []
    for line in server_log.splitlines():
        if "speculative_num_steps" in line or "STEPTRACE_FEATURES source=request_cpu_v2" in line:
            headers.append(line)
        if "STEPTRACE " not in line or "gpu=" not in line:
            continue
        match = re.search(r"\bt=([0-9.]+) gpu=([0-9.]+)", line)
        if match is None or float(match[2]) <= 0:
            raise ValueError("Invalid GPU-time trace record")
        if first_request <= float(match[1]) <= last_request:
            measured.append(line)
    if not measured or not any("STEPTRACE_FEATURES source=request_cpu_v2" in line for line in headers):
        raise ValueError("Missing measured CPU-feature trace")
    return "\n".join([*headers, *measured]) + "\n", len(measured)
