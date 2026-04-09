#!/usr/bin/env python3
"""Generate the 2 Pareto curves for the nebius-swe-traj workload:

  baseline (no SLO control): x = request rate, y = median ITL (ms)
  EDF + SLO-aware:           x = request rate, y = decode SLO attainment %
                              (per-request P99 ITL ≤ 150ms — Sarathi-style)

Why per-request P99 ITL instead of strict max-ITL: each request has 1024 decode
steps with 1000 reqs (= 1M decode steps total). The probability of any one decode
step exceeding the SLO is high purely from natural noise, so strict max-ITL gives
≈0% baseline for all methods and reveals nothing. Per-request P99 ITL filters
out single outliers but still penalizes the consistent prefill-blocking pattern
that DMC/LPT_CG suffer from on this 9K-input workload.

Usage: python3 plot_pareto_nebius.py <baseline.json> <edf.json> [outdir]
"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

baseline_path = sys.argv[1]
edf_path = sys.argv[2]
outdir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(baseline_path)

with open(baseline_path) as f:
    baseline_rows = json.load(f)
with open(edf_path) as f:
    edf_rows = json.load(f)

methods = ["PP", "DMC", "LPT_CG"]
colors = {"PP": "tab:blue", "DMC": "tab:orange", "LPT_CG": "tab:green"}
markers = {"PP": "o", "DMC": "s", "LPT_CG": "^"}
ITL_SLO_MS = 150


def by_method(rows, method):
    g = [r for r in rows if r["workload"] == "nebius_swe" and r["method"] == method]
    g.sort(key=lambda r: float(r["rate"]) if r["rate"] != "inf" else 1e9)
    return g


def rate_x(r):
    return 16 if r["rate"] == "inf" else float(r["rate"])


# === Side-by-side: baseline median ITL + EDF SLO attainment ===
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for m in methods:
    g = by_method(baseline_rows, m)
    if not g:
        continue
    xs = [rate_x(r) for r in g]
    ys = [r["median_itl_ms"] for r in g]
    ax.plot(xs, ys, marker=markers[m], color=colors[m], label=m, linewidth=2, markersize=8)
ax.set_xlabel("Request rate (req/s, 16=inf)")
ax.set_ylabel("Median ITL (ms)")
ax.set_title("nebius-swe-traj — baseline (no SLO control)")
ax.grid(alpha=0.3)
ax.legend()

ax = axes[1]
for m in methods:
    g = by_method(edf_rows, m)
    if not g:
        continue
    xs = [rate_x(r) for r in g]
    ys = [r["itl_p99_attain_pct"] for r in g]
    ax.plot(xs, ys, marker=markers[m], color=colors[m], label=m, linewidth=2, markersize=8)
ax.set_xlabel("Request rate (req/s, 16=inf)")
ax.set_ylabel(f"Decode SLO attainment (%, per-req P99 ITL ≤ {ITL_SLO_MS}ms)")
ax.set_title("nebius-swe-traj — EDF + SLO-aware chunk control")
ax.set_ylim(0, 105)
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
out = os.path.join(outdir, "pareto_nebius.png")
plt.savefig(out, dpi=120)
print(f"Saved {out}")
plt.close()
