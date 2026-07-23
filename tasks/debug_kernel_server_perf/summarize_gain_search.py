#!/usr/bin/env python3
"""Rank matched LPT_CG/DMC synthetic runs by serving-level gains."""

import argparse
import json
import re
from pathlib import Path


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
        rows.append(
            (
                100 * (lpt_total / dmc_total - 1),
                100 * (1 - lpt["p99_itl_ms"] / dmc["p99_itl_ms"]),
                100 * (1 - lpt["median_ttft_ms"] / dmc["median_ttft_ms"]),
                case,
            )
        )

    print("case                               total_gain  p99_itl_gain  ttft_gain")
    for throughput, itl, ttft, case in sorted(rows, reverse=True):
        print(f"{case:34s} {throughput:+10.2f}% {itl:+12.2f}% {ttft:+9.2f}%")


if __name__ == "__main__":
    main()
