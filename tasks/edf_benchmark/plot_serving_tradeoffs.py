#!/usr/bin/env python3
"""Plot paper-facing serving tradeoffs without conflating TTFT and ITL SLOs.

Produces four figures:
  * TTFT SLO attainment vs offered request/session rate
  * per-request P99-ITL SLO attainment vs offered request/session rate
  * pooled P99 ITL vs offered request/session rate
  * achieved total-token throughput vs mean ITL (capacity/interactivity)
"""

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "tasks/edf_benchmark/results_current_policy_n400/summary.json"
)
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(SUMMARY)

METHODS = ("PP_DECODE_CG", "DMC_CG", "LPT")
COLORS = {
    "PP_DECODE_CG": "#3274A1",
    "DMC_CG": "#E1812C",
    "LPT": "#3A923A",
}
MARKERS = {"PP_DECODE_CG": "o", "DMC_CG": "s", "LPT": "^"}
LABELS = {
    "PP_DECODE_CG": "PP",
    "DMC_CG": "DMC",
    "LPT": "LPT",
}
WORKLOAD_LABELS = {
    "code_feedback": "Code-Feedback",
    "swe": "SWE-Gym",
    "tracelab_claude": "TraceLab Claude",
}


def numeric_rate(rate):
    return math.inf if str(rate) == "inf" else float(rate)


def valid_rows(rows, workload, method):
    selected = [
        row
        for row in rows
        if row["workload"] == workload
        and row["method"] == method
        and row.get("output_valid", True)
    ]
    return sorted(selected, key=lambda row: numeric_rate(row["rate"]))


def rate_axis(groups):
    finite = sorted(
        {
            numeric_rate(row["rate"])
            for group in groups
            for row in group
            if math.isfinite(numeric_rate(row["rate"]))
        }
    )
    inf_x = finite[-1] + max(2.0, 0.2 * (finite[-1] - finite[0])) if finite else 1
    # Dense low-rate SWE points otherwise overlap; preserve the endpoints and
    # the integer/half-integer landmarks that describe the knee.
    if len(finite) > 7:
        if finite[-1] <= 5:
            shown = [
                rate
                for rate in finite
                if rate == finite[0] or abs(rate * 2 - round(rate * 2)) < 1e-9
            ]
        else:
            shown = finite[::2]
            if shown[-1] != finite[-1]:
                shown.append(finite[-1])
    else:
        shown = finite
    ticks = shown[:]
    labels = [f"{rate:g}" for rate in shown]
    if any(not math.isfinite(numeric_rate(row["rate"])) for group in groups for row in group):
        ticks.append(inf_x)
        labels.append("sat.")
    return inf_x, ticks, labels


def plot_rate_metric(rows, workloads, key, ylabel, filename, ylim=None):
    fig, axes = plt.subplots(
        1, len(workloads), figsize=(5.2 * len(workloads), 4.0), squeeze=False
    )
    for ax, workload in zip(axes[0], workloads):
        groups = [valid_rows(rows, workload, method) for method in METHODS]
        if workload == "tracelab_claude":
            ordered_rates = sorted(
                {
                    numeric_rate(row["rate"])
                    for group in groups
                    for row in group
                }
            )
            rate_positions = {rate: index for index, rate in enumerate(ordered_rates)}
            ticks = list(range(len(ordered_rates)))
            tick_labels = [
                "sat." if not math.isfinite(rate) else f"{rate:g}"
                for rate in ordered_rates
            ]
        else:
            inf_x, ticks, tick_labels = rate_axis(groups)
        for method, group in zip(METHODS, groups):
            if not group:
                continue
            if workload == "tracelab_claude":
                xs = [rate_positions[numeric_rate(row["rate"])] for row in group]
            else:
                xs = [
                    inf_x
                    if not math.isfinite(numeric_rate(row["rate"]))
                    else numeric_rate(row["rate"])
                    for row in group
                ]
            ax.plot(
                xs,
                [row[key] for row in group],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2,
                markersize=6,
                label=LABELS[method],
            )
        ax.set_title(WORKLOAD_LABELS.get(workload, workload))
        ax.set_xlabel(
            "Offered session arrival rate (sessions/s)"
            if workload == "tracelab_claude"
            else "Offered request rate (req/s)"
        )
        ax.set_ylabel(ylabel)
        ax.set_xticks(ticks, tick_labels)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(OUTDIR, filename)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_capacity_interactivity(rows, workloads):
    fig, axes = plt.subplots(
        1, len(workloads), figsize=(5.2 * len(workloads), 4.0), squeeze=False
    )
    for ax, workload in zip(axes[0], workloads):
        for method in METHODS:
            group = valid_rows(rows, workload, method)
            if not group:
                continue
            ax.plot(
                [row["total_throughput"] for row in group],
                [row["mean_itl_ms"] for row in group],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2,
                markersize=6,
                label=LABELS[method],
            )
        ax.set_title(WORKLOAD_LABELS.get(workload, workload))
        ax.set_xlabel("Achieved total throughput (token/s)")
        ax.set_ylabel("Mean ITL (ms) ↓")
        ax.grid(alpha=0.25)
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(OUTDIR, "pareto_capacity_vs_mean_itl.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    with open(SUMMARY) as file:
        rows = json.load(file)
    os.makedirs(OUTDIR, exist_ok=True)
    workloads = [
        workload
        for workload in ("tracelab_claude", "code_feedback", "swe")
        if any(row["workload"] == workload for row in rows)
    ]
    plot_rate_metric(
        rows,
        workloads,
        "ttft_attain_pct",
        "TTFT SLO attainment (%)",
        "slo_ttft_attainment_vs_rate.png",
        (0, 105),
    )
    plot_rate_metric(
        rows,
        workloads,
        "itl_p99_attain_pct",
        "Per-request P99-ITL SLO attainment (%)",
        "slo_itl_attainment_vs_rate.png",
        (0, 105),
    )
    plot_rate_metric(
        rows,
        workloads,
        "p99_itl_ms",
        "Pooled P99 ITL (ms) ↓",
        "p99_itl_vs_rate.png",
    )
    plot_capacity_interactivity(rows, workloads)


if __name__ == "__main__":
    main()
