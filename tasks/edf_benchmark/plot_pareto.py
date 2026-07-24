#!/usr/bin/env python3
"""Generate Pareto curves from EDF benchmark summary.

Two figures per workload:
  1. Throughput (tok/s) vs ITL P99 (ms) — latency/throughput tradeoff
  2. Request rate vs SLO attainment % — saturation curve
"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUMMARY = sys.argv[1] if len(sys.argv) > 1 else "tasks/edf_benchmark/results_edf/summary.json"
OUTDIR = os.path.dirname(SUMMARY)

with open(SUMMARY) as f:
    rows = json.load(f)

preferred_methods = ["PP_DECODE_CG", "DMC_CG", "LPT"]
methods = [m for m in preferred_methods if any(r["method"] == m for r in rows)]
workloads = sorted({r["workload"] for r in rows})
colors = {
    "PP_DECODE_CG": "tab:blue",
    "DMC_CG": "tab:orange",
    "LPT": "tab:green",
}
markers = {"PP_DECODE_CG": "o", "DMC_CG": "s", "LPT": "^"}
labels = {
    "PP_DECODE_CG": "PP (decode CG)",
    "DMC_CG": "DMC (decode CG)",
    "LPT": "LPT persistent (decode CG)",
}


def by_method(rows, wl, method):
    g = [
        r
        for r in rows
        if r["workload"] == wl
        and r["method"] == method
        and r.get("output_valid", True)
    ]
    g.sort(key=lambda r: float(r["rate"]) if r["rate"] != "inf" else 1e30)
    return g


def rate_positions(groups):
    finite = sorted(
        {
            float(r["rate"])
            for group in groups
            for r in group
            if r["rate"] != "inf"
        }
    )
    inf_position = finite[-1] * 1.25 if finite else 1.0
    return finite, inf_position


# Figure 1: throughput vs P99 ITL
fig, axes = plt.subplots(1, len(workloads), figsize=(5 * len(workloads), 4))
if len(workloads) == 1:
    axes = [axes]
for ax, wl in zip(axes, workloads):
    for m in methods:
        g = by_method(rows, wl, m)
        if not g:
            continue
        xs = [r["throughput"] for r in g]
        ys = [r["p99_itl_ms"] for r in g]
        ax.plot(xs, ys, marker=markers[m], color=colors[m], label=labels[m])
        for r in g:
            ax.annotate(f"r={r['rate']}", (r["throughput"], r["p99_itl_ms"]),
                        fontsize=7, alpha=0.7)
    ax.set_xlabel("Output throughput (tok/s)")
    ax.set_ylabel("ITL P99 (ms)")
    ax.set_title(wl)
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
out1 = os.path.join(OUTDIR, "pareto_throughput_vs_itl.png")
plt.savefig(out1, dpi=120)
print(f"Saved {out1}")
plt.close()

# Figure 2: rate vs SLO attainment
fig, axes = plt.subplots(1, len(workloads), figsize=(5 * len(workloads), 4))
if len(workloads) == 1:
    axes = [axes]
for ax, wl in zip(axes, workloads):
    groups = [by_method(rows, wl, method) for method in methods]
    finite_rates, inf_position = rate_positions(groups)
    for m in methods:
        g = by_method(rows, wl, m)
        if not g:
            continue
        xs = [
            float(r["rate"]) if r["rate"] != "inf" else inf_position for r in g
        ]
        ys = [r["slo_attain_pct"] for r in g]
        ax.plot(xs, ys, marker=markers[m], color=colors[m], label=labels[m])
    ticks = list(finite_rates)
    tick_labels = [f"{rate:g}" for rate in finite_rates]
    if any(r["rate"] == "inf" for group in groups for r in group):
        ticks.append(inf_position)
        tick_labels.append("inf")
    ax.set_xticks(ticks, tick_labels)
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("SLO attainment (%)")
    ax.set_title(f"{wl} (TTFT+ITL)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
out2 = os.path.join(OUTDIR, "pareto_slo_vs_rate.png")
plt.savefig(out2, dpi=120)
print(f"Saved {out2}")
plt.close()
