#!/usr/bin/env python3
"""Rank matched LPT_CG/DMC synthetic runs by serving-level gains."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


NAME_RE = re.compile(r"LPT_CG_(in\d+_out\d+_rate[^.]+)\.json$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    rows = []
    for lpt_path in args.results_dir.glob("LPT_CG_*.json"):
        match = NAME_RE.match(lpt_path.name)
        if match is None:
            continue
        case = match.group(1)
        dmc_path = args.results_dir / f"DMC_{case}.json"
        if not dmc_path.exists():
            continue
        lpt = json.loads(lpt_path.read_text())
        dmc = json.loads(dmc_path.read_text())
        lpt_total = (lpt["total_input_tokens"] + lpt["total_output_tokens"]) / lpt[
            "duration"
        ]
        dmc_total = (dmc["total_input_tokens"] + dmc["total_output_tokens"]) / dmc[
            "duration"
        ]
        lpt_req_max = np.array(
            [max(xs) * 1000 if xs else 0 for xs in lpt.get("itls", [])]
        )
        dmc_req_max = np.array(
            [max(xs) * 1000 if xs else 0 for xs in dmc.get("itls", [])]
        )

        def latency_gain(lpt_ms: float, dmc_ms: float) -> float:
            return 100 * (1 - lpt_ms / dmc_ms) if dmc_ms else 0

        rows.append(
            (
                100 * (lpt_total / dmc_total - 1),
                latency_gain(lpt["p99_itl_ms"], dmc["p99_itl_ms"]),
                latency_gain(
                    float(np.median(lpt_req_max)),
                    float(np.median(dmc_req_max)),
                ),
                latency_gain(
                    float(np.percentile(lpt_req_max, 99)),
                    float(np.percentile(dmc_req_max, 99)),
                ),
                100 * (1 - lpt["median_ttft_ms"] / dmc["median_ttft_ms"]),
                case,
            )
        )

    print(
        "case                               total_gain  pooled_p99  "
        "reqmax_p50  reqmax_p99  ttft_gain"
    )
    for throughput, pooled, req_p50, req_p99, ttft, case in sorted(
        rows, reverse=True
    ):
        print(
            f"{case:34s} {throughput:+10.2f}% {pooled:+10.2f}% "
            f"{req_p50:+10.2f}% {req_p99:+10.2f}% {ttft:+9.2f}%"
        )


if __name__ == "__main__":
    main()
