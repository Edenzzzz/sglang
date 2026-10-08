#!/usr/bin/env python3
"""Summarize SGLang request queue/forward timing logs for matched runs."""

import argparse
import math
import re
from pathlib import Path


REQUEST_RE = re.compile(
    r"Req Time Stats\(.*?input len=(?P<input>\d+), output len=(?P<output>\d+),"
    r".*?queue_duration=(?P<queue>[\d.]+)ms, "
    r"prefill_duration=(?P<prefill>[\d.]+)ms, "
    r"forward_duration=(?P<forward>[\d.]+)ms"
)
PREFILL_RE = re.compile(
    r"Prefill batch, #new-seq: (?P<new_seq>\d+), #new-token: (?P<new_token>\d+),"
    r".*?#running-req: (?P<running>\d+), #queue-req: (?P<queued>\d+)"
)


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(path, num_requests):
    text = Path(path).read_text(errors="replace")
    requests = [
        {
            "input": int(match["input"]),
            "output": int(match["output"]),
            "queue": float(match["queue"]),
            "prefill": float(match["prefill"]),
            "forward": float(match["forward"]),
        }
        for match in REQUEST_RE.finditer(text)
    ][-num_requests:]
    if len(requests) != num_requests:
        raise ValueError(f"{path}: found {len(requests)} request records")

    prefills = [
        {key: int(value) for key, value in match.groupdict().items()}
        for match in PREFILL_RE.finditer(text)
    ]
    queue = [row["queue"] for row in requests]
    prefill = [row["prefill"] for row in requests]
    server_ttft = [
        row["queue"] + row["prefill"]
        for row in requests
    ]
    forward = [row["forward"] for row in requests]
    return {
        "requests": len(requests),
        "queue_mean": sum(queue) / len(queue),
        "queue_p50": percentile(queue, 0.50),
        "queue_p95": percentile(queue, 0.95),
        "queue_p99": percentile(queue, 0.99),
        "prefill_mean": sum(prefill) / len(prefill),
        "prefill_p50": percentile(prefill, 0.50),
        "prefill_p95": percentile(prefill, 0.95),
        "prefill_p99": percentile(prefill, 0.99),
        "server_ttft_mean": sum(server_ttft) / len(server_ttft),
        "server_ttft_p50": percentile(server_ttft, 0.50),
        "server_ttft_p95": percentile(server_ttft, 0.95),
        "server_ttft_p99": percentile(server_ttft, 0.99),
        "forward_mean": sum(forward) / len(forward),
        "forward_p50": percentile(forward, 0.50),
        "forward_p95": percentile(forward, 0.95),
        "forward_p99": percentile(forward, 0.99),
        "queued_over_1s": 100 * sum(value > 1000 for value in queue) / len(queue),
        "queued_over_10s": 100 * sum(value > 10000 for value in queue) / len(queue),
        "prefill_batches_logged": len(prefills),
        "prefill_queue_mean": (
            sum(row["queued"] for row in prefills) / len(prefills) if prefills else 0
        ),
        "prefill_queue_max": max((row["queued"] for row in prefills), default=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--num-requests", type=int, default=400)
    args = parser.parse_args()

    print(
        "| run | queue mean / P50 / P95 / P99 (ms) | "
        "prefill-to-first-token mean / P50 / P95 / P99 (ms) | "
        "server TTFT mean / P50 / P95 / P99 (ms) | "
        "post-admit E2E mean / P50 / P95 / P99 (ms) | queue >1s / >10s |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for log in args.logs:
        stats = summarize(log, args.num_requests)
        label = Path(log).stem.removeprefix("server_")
        print(
            f"| {label} | {stats['queue_mean']:.1f} / {stats['queue_p50']:.1f} / "
            f"{stats['queue_p95']:.1f} / {stats['queue_p99']:.1f} | "
            f"{stats['prefill_mean']:.1f} / {stats['prefill_p50']:.1f} / "
            f"{stats['prefill_p95']:.1f} / {stats['prefill_p99']:.1f} | "
            f"{stats['server_ttft_mean']:.1f} / {stats['server_ttft_p50']:.1f} / "
            f"{stats['server_ttft_p95']:.1f} / {stats['server_ttft_p99']:.1f} | "
            f"{stats['forward_mean']:.1f} / {stats['forward_p50']:.1f} / "
            f"{stats['forward_p95']:.1f} / {stats['forward_p99']:.1f} | "
            f"{stats['queued_over_1s']:.1f}% / {stats['queued_over_10s']:.1f}% |"
        )


if __name__ == "__main__":
    main()
