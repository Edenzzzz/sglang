#!/usr/bin/env python3
"""Generate individual publication-quality Pareto curves for each workload.

For each workload produces 2 figures:
  - baseline: x=request rate, y=P99 ITL (ms), 3 lines (PP/DMC/LPT_CG)
  - EDF: x=request rate, y=per-req P99 ITL ≤ SLO attainment %, 3 lines

Usage: python3 plot_curves.py <outdir>
  Reads all summary.json files from results_*/ dirs under tasks/edf_benchmark/.
"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
outdir = sys.argv[1] if len(sys.argv) > 1 else SCRIPT_DIR

# Dataset configs: (dir_prefix, workload_key, display_name, itl_slo_ms, rates_label)
DATASETS = [
    ("results_baseline_1k", "results_edf_1k", "sharegpt", "ShareGPT", 100),
    ("results_baseline_1k", "results_edf_1k", "code_feedback", "Code-Feedback", 150),
    ("results_baseline_1k", "results_edf_1k", "swe", "SWE-Gym", 150),
    ("results_baseline_nebius_1k", "results_edf_nebius_1k", "nebius_swe", "Nebius-SWE-Traj", 150),
    ("results_baseline_swebpro_1k", "results_edf_swebpro_1k", "swe_bench_pro", "SWE-bench-Pro", 150),
]

methods = ["PP", "DMC", "LPT_CG"]
colors = {"PP": "#1f77b4", "DMC": "#ff7f0e", "LPT_CG": "#2ca02c"}
markers = {"PP": "o", "DMC": "s", "LPT_CG": "^"}
labels = {"PP": "PP (prefill-prio)", "DMC": "DMC (default mixed chunk)", "LPT_CG": "LPT+CG (persistent kernel)"}


def load_summary(dirname):
    path = os.path.join(SCRIPT_DIR, dirname, "summary.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def by_method(rows, wl, method):
    g = [r for r in rows if r["workload"] == wl and r["method"] == method]
    g.sort(key=lambda r: float(r["rate"]) if r["rate"] != "inf" else 1e9)
    return g


def rate_ticks(rows, wl):
    """Get sorted unique rates as floats (inf→max+step)."""
    rates = set()
    for r in rows:
        if r["workload"] == wl:
            rates.add(float(r["rate"]) if r["rate"] != "inf" else None)
    finite = sorted(r for r in rates if r is not None)
    if None in rates:
        # place inf tick at 1.5× the second-largest rate
        inf_pos = finite[-1] * 1.5 if finite else 100
        return finite, inf_pos
    return finite, None


def plot_one(rows, wl, display, itl_slo, y_key, y_label, title_suffix, filename, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    finite_rates, inf_pos = rate_ticks(rows, wl)

    for m in methods:
        g = by_method(rows, wl, m)
        if not g:
            continue
        xs = []
        ys = []
        tputs = []
        for r in g:
            x = float(r["rate"]) if r["rate"] != "inf" else inf_pos
            xs.append(x)
            ys.append(r[y_key])
            tputs.append(r.get("throughput", 0))
        ax.plot(xs, ys, marker=markers[m], color=colors[m], label=labels[m],
                linewidth=2.5, markersize=9, zorder=3)
        # Annotate throughput on each point
        for x, y, t in zip(xs, ys, tputs):
            if t > 0:
                ax.annotate(f"{t:.0f}", (x, y), textcoords="offset points",
                            xytext=(0, 10), fontsize=7, color=colors[m],
                            ha='center', alpha=0.7)

    # X-axis: show "inf" label
    tick_positions = list(finite_rates)
    tick_labels = [str(int(r)) for r in finite_rates]
    if inf_pos is not None:
        tick_positions.append(inf_pos)
        tick_labels.append("inf")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=12)

    ax.set_xlabel("Request rate (req/s)", fontsize=14)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(f"{display} — {title_suffix}", fontsize=15, fontweight='bold')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    out = os.path.join(outdir, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close()


for base_dir, edf_dir, wl_key, display, itl_slo in DATASETS:
    base_rows = load_summary(base_dir)
    edf_rows = load_summary(edf_dir)

    if not base_rows:
        print(f"  skip {wl_key}: no baseline data")
        continue

    # Baseline curve: P99 ITL
    plot_one(base_rows, wl_key, display, itl_slo,
             y_key="p99_itl_ms",
             y_label="P99 ITL (ms)",
             title_suffix="Baseline (no SLO control)",
             filename=f"curve_{wl_key}_baseline_p99itl.png")

    # EDF curve: per-req P99 ITL attainment
    if edf_rows:
        plot_one(edf_rows, wl_key, display, itl_slo,
                 y_key="itl_p99_attain_pct",
                 y_label=f"Decode SLO attainment (%, per-req P99 ITL ≤ {itl_slo}ms)",
                 title_suffix="EDF + SLO-aware chunk control",
                 filename=f"curve_{wl_key}_edf_itlp99.png",
                 ylim=(-5, 105))

print(f"\nDone. {len(DATASETS)*2} curves generated in {outdir}")
