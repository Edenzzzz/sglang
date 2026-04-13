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

methods = ["PP", "DMC", "LPT_CG"]
workloads = sorted({r["workload"] for r in rows})
colors = {"PP": "tab:blue", "DMC": "tab:orange", "LPT_CG": "tab:green"}
markers = {"PP": "o", "DMC": "s", "LPT_CG": "^"}


def by_method(rows, wl, method):
    g = [r for r in rows if r["workload"] == wl and r["method"] == method]
    g.sort(key=lambda r: float(r["rate"]))
    return g


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
        ax.plot(xs, ys, marker=markers[m], color=colors[m], label=m)
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
    for m in methods:
        g = by_method(rows, wl, m)
        if not g:
            continue
        xs = [float(r["rate"]) for r in g]
        ys = [r["slo_attain_pct"] for r in g]
        ax.plot(xs, ys, marker=markers[m], color=colors[m], label=m)
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
