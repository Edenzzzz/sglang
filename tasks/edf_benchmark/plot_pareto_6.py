#!/usr/bin/env python3
"""Generate the 6 Pareto curves requested:

  3 baseline (no SLO control): x = request rate, y = median ITL (ms)
  3 EDF + SLO-aware:           x = request rate, y = decode SLO attainment %
                                (decode SLO = TPOT, mean ITL per req ≤ ITL_SLO)

Usage: python3 plot_pareto_6.py <baseline_summary.json> <edf_summary.json> [outdir]
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
workloads = ["sharegpt", "code_feedback", "swe"]
colors = {"PP": "tab:blue", "DMC": "tab:orange", "LPT_CG": "tab:green"}
markers = {"PP": "o", "DMC": "s", "LPT_CG": "^"}
SLO_DECODE = {"sharegpt": 100, "code_feedback": 150, "swe": 150}


def by_method(rows, wl, method):
    g = [r for r in rows if r["workload"] == wl and r["method"] == method]
    g.sort(key=lambda r: float(r["rate"]) if r["rate"] != "inf" else 1e9)
    return g


def rate_x(r):
    """Use a large finite value for 'inf' so it plots."""
    return 64 if r["rate"] == "inf" else float(r["rate"])


# === Figure 1: Baseline (no SLO control) — median ITL vs rate ===
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, wl in zip(axes, workloads):
    for m in methods:
        g = by_method(baseline_rows, wl, m)
        if not g:
            continue
        xs = [rate_x(r) for r in g]
        ys = [r["median_itl_ms"] for r in g]
        ax.plot(xs, ys, marker=markers[m], color=colors[m], label=m, linewidth=2, markersize=8)
    ax.set_xlabel("Request rate (req/s, 64=inf)")
    ax.set_ylabel("Median ITL (ms)")
    ax.set_title(f"{wl} — baseline (no SLO control)")
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
out1 = os.path.join(outdir, "pareto_baseline_median_itl.png")
plt.savefig(out1, dpi=120)
print(f"Saved {out1}")
plt.close()

# === Figure 2: EDF + SLO-aware — decode SLO attainment vs rate ===
# "Decode SLO" = strict per-request: max ITL ≤ ITL_SLO. We use itl_attain_pct
# rather than tpot_attain_pct because TPOT (mean ITL ≤ SLO) stays ≈100% even
# at saturation (median ITL ~10-30ms vs 100-150ms SLO), so it doesn't reveal
# differences. The strict tail metric is what's protected by chunk control.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, wl in zip(axes, workloads):
    for m in methods:
        g = by_method(edf_rows, wl, m)
        if not g:
            continue
        xs = [rate_x(r) for r in g]
        ys = [r["itl_attain_pct"] for r in g]
        ax.plot(xs, ys, marker=markers[m], color=colors[m], label=m, linewidth=2, markersize=8)
    ax.set_xlabel("Request rate (req/s, 64=inf)")
    ax.set_ylabel(f"Decode SLO attainment (%, max ITL ≤ {SLO_DECODE[wl]}ms)")
    ax.set_title(f"{wl} — EDF + SLO-aware chunk control")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
out2 = os.path.join(outdir, "pareto_edf_slo_attainment.png")
plt.savefig(out2, dpi=120)
print(f"Saved {out2}")
plt.close()
