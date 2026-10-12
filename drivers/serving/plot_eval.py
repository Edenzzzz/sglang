#!/usr/bin/env python3
"""Figures for the S1 x S2 evaluation from eval_curves.py's CSV.

One row of panels per (workload, stream): throughput vs per-user tok/s (Pareto, one point per
offered rate), SLO attainment vs rate, mean TTFT vs rate, mean TPOT vs rate.

Usage: plot_eval.py <csv> <out.png>
"""
import csv, sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = {"MTP_LPT": "stock MTP", "MTPMIX_LPT": "4-mode, unfused", "MTPMIXFUSED": "4-mode, fused"}
STYLE = {"MTP_LPT": ("tab:gray", "s"), "MTPMIX_LPT": ("tab:blue", "^"),
         "MTPMIXFUSED": ("tab:red", "o")}


def main():
    rows = [r for r in csv.DictReader(open(sys.argv[1])) if r.get("workload")]
    keys = sorted({(r["workload"], r["stream"]) for r in rows})
    fig, axes = plt.subplots(len(keys), 4, figsize=(18, 3.3 * len(keys)), squeeze=False)
    for i, (wl, st) in enumerate(keys):
        for arm in LABEL:
            pts = sorted((r for r in rows if r["workload"] == wl and r["stream"] == st
                          and r["arm"] == arm), key=lambda r: float(r["rate"]))
            if not pts:
                continue
            color, marker = STYLE[arm]
            rate = [float(p["rate"]) for p in pts]
            kw = dict(color=color, marker=marker, label=LABEL[arm], ms=4)
            axes[i][0].plot([float(p["user_tok_s_median"]) for p in pts],
                            [float(p["out_tok_s"]) for p in pts], "-", **kw)
            axes[i][1].plot(rate, [float(p["slo_attain"]) for p in pts], "-", **kw)
            axes[i][2].plot(rate, [float(p["ttft_mean_ms"]) / 1000 for p in pts], "-", **kw)
            axes[i][3].plot(rate, [float(p["tpot_mean_ms"]) for p in pts], "-", **kw)
        tag = f"{wl} / {st}"
        axes[i][0].set(title=f"{tag}: throughput vs interactivity", xlabel="median tok/s per user",
                       ylabel="output tok/s")
        axes[i][1].set(title=f"{tag}: SLO attainment", xlabel="chat rate (req/s)",
                       ylabel="fraction meeting SLO", ylim=(-0.02, 1.02))
        axes[i][2].set(title=f"{tag}: mean TTFT", xlabel="chat rate (req/s)", ylabel="s")
        axes[i][3].set(title=f"{tag}: mean TPOT", xlabel="chat rate (req/s)", ylabel="ms")
        for ax in axes[i]:
            ax.grid(alpha=0.3)
        axes[i][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(sys.argv[2], dpi=110)


if __name__ == "__main__":
    main()
