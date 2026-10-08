#!/usr/bin/env python3
"""Summarize matched, output-valid serving comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def total_throughput(result: dict) -> float:
    return (
        result["total_input_tokens"] + result["total_output_tokens"]
    ) / result["duration"]


def output_ratio(result: dict) -> float:
    generated = result.get("total_output_tokens", 0)
    retokenized = result.get("total_output_tokens_retokenized", generated)
    return retokenized / generated if generated else 1.0


def request_max_itl_ms(result: dict) -> np.ndarray:
    return np.asarray(
        [max(values) * 1000 if values else 0 for values in result.get("itls", [])]
    )


def latency_gain(candidate_ms: float, baseline_ms: float) -> float:
    return 100 * (1 - candidate_ms / baseline_ms) if baseline_ms else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--candidate", default="LPT")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["DMC_CG", "PP_DECODE_CG"],
    )
    # Tokenizer round trips are not exact for arbitrary-token prompts because
    # decoded special/invalid pieces may disappear or expand. A symmetric 2%
    # band catches the observed graph corruptions (22–94%) without rejecting
    # otherwise complete stochastic generations.
    parser.add_argument("--min-output-ratio", type=float, default=0.98)
    args = parser.parse_args()

    candidate_prefix = f"{args.candidate}_"
    rows = []
    invalid = []
    for candidate_path in sorted(args.results_dir.glob(f"{args.candidate}_*.json")):
        case = candidate_path.name[len(candidate_prefix) : -len(".json")]
        candidate = load(candidate_path)
        candidate_ratio = output_ratio(candidate)
        for baseline_name in args.baselines:
            baseline_path = args.results_dir / f"{baseline_name}_{case}.json"
            if not baseline_path.exists():
                continue
            baseline = load(baseline_path)
            baseline_ratio = output_ratio(baseline)
            max_ratio = 2.0 - args.min_output_ratio
            if not (
                args.min_output_ratio <= candidate_ratio <= max_ratio
                and args.min_output_ratio <= baseline_ratio <= max_ratio
            ):
                invalid.append(
                    (case, baseline_name, candidate_ratio, baseline_ratio)
                )
                continue

            candidate_req = request_max_itl_ms(candidate)
            baseline_req = request_max_itl_ms(baseline)
            rows.append(
                (
                    case,
                    baseline_name,
                    100
                    * (total_throughput(candidate) / total_throughput(baseline) - 1),
                    latency_gain(
                        candidate["p99_itl_ms"], baseline["p99_itl_ms"]
                    ),
                    latency_gain(
                        float(np.median(candidate_req)),
                        float(np.median(baseline_req)),
                    ),
                    latency_gain(
                        float(np.percentile(candidate_req, 99)),
                        float(np.percentile(baseline_req, 99)),
                    ),
                    100
                    * (
                        1
                        - candidate["median_ttft_ms"]
                        / baseline["median_ttft_ms"]
                    ),
                    candidate_ratio,
                    baseline_ratio,
                )
            )

    print(
        "case                           baseline       total_gain  pooled_p99  "
        "reqmax_p50  reqmax_p99  ttft_gain  output ratios"
    )
    for (
        case,
        baseline,
        throughput,
        pooled,
        req_p50,
        req_p99,
        ttft,
        candidate_ratio,
        baseline_ratio,
    ) in rows:
        print(
            f"{case:30s} {baseline:13s} {throughput:+10.2f}% "
            f"{pooled:+10.2f}% {req_p50:+10.2f}% {req_p99:+10.2f}% "
            f"{ttft:+9.2f}% {candidate_ratio:.4f}/{baseline_ratio:.4f}"
        )

    if invalid:
        print("\nRejected output-invalid comparisons:")
        for case, baseline, candidate_ratio, baseline_ratio in invalid:
            print(
                f"  {case} vs {baseline}: "
                f"{candidate_ratio:.4f}/{baseline_ratio:.4f}"
            )


if __name__ == "__main__":
    main()
