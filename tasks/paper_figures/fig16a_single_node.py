"""Figure 16a of the paper draft: single-node SLO attainment, stock sglang vs ours, Qwen2.5-7B on one H200, final
configuration (unified dispatcher, tile-rounded KV splits, no chunk cap), paper SLOs. Three panels: chat at
2 s + 50 ms, chat at 2 s + 100 ms, agent at 10 s (20 s above 200K uncached tokens) + 80 ms; one group per workload
and mode. Values are Table 7b's (two repeats; where the table gives a range, the bar is its mean and the whisker spans
it). Numbers from tasks/debug_kernel_server_perf_4_mode.md (stage 2, rerun without the chunk cap,
unified/a6_stage2_h200_nodc_20261026.sh).
  python fig16a_single_node.py   -> fig16a_single_node.png
Claude, 2026-10-26.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
GROUPS = ["128-256K\n4-mode", "128-256K\n2-mode", "32-64K\n4-mode", "32-64K\n2-mode"]
# panel: [(stock lo, stock hi), (ours lo, ours hi)] per group
PANELS = {
    "chat, 2 s + 50 ms": [((21.0, 21.0), (63.1, 63.1)), ((18.8, 18.8), (62.8, 62.8)),
                          ((56.4, 56.4), (97.2, 97.2)), ((54.3, 54.3), (95.4, 95.4))],
    "chat, 2 s + 100 ms": [((21.0, 21.0), (94.9, 94.9)), ((18.9, 18.9), (92.6, 92.6)),
                           ((56.5, 56.5), (99.6, 99.6)), ((54.4, 54.4), (99.8, 99.8))],
    "agent, 10 s + 80 ms": [((18.8, 18.8), (43.8, 43.8)), ((12.5, 18.8), (50.0, 50.0)),
                            ((70.0, 73.3), (80.0, 80.0)), ((83.3, 83.3), (83.3, 86.7))],
}
ARMS = [("stock sglang (mixed chunk, FA3; stock MTP in four-mode)", "#eb6834"), ("ours (full stack)", "#2a78d6")]
MUTED, GRID, INK = "#52514e", "#e4e3df", "#0b0b0b"


def main():
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True, sharey=True)
    width = 0.36
    for ax, (title, rows) in zip(axes, PANELS.items()):
        for k, (label, color) in enumerate(ARMS):
            xs = [g + (k - 0.5) * (width + 0.04) for g in range(len(GROUPS))]
            mids = [(lo + hi) / 2 for lo, hi in (row[k] for row in rows)]
            err = [[m - row[k][0] for m, row in zip(mids, rows)], [row[k][1] - m for m, row in zip(mids, rows)]]
            ax.bar(xs, mids, width=width, color=color, label=label, edgecolor="#fcfcfb", linewidth=1)
            spread = [i for i, row in enumerate(rows) if row[k][1] > row[k][0]]
            if spread:
                ax.errorbar([xs[i] for i in spread], [mids[i] for i in spread],
                            yerr=[[err[0][i] for i in spread], [err[1][i] for i in spread]], fmt="none", ecolor=INK,
                            elinewidth=0.8, capsize=2)
            for x, m in zip(xs, mids):
                ax.text(x, m + 3.5, f"{m:.0f}", ha="center", va="bottom", fontsize=6.5, color=MUTED)
        ax.axhline(90, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels(GROUPS, fontsize=7)
        ax.set_ylim(0, 108)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("SLO attainment (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.1), fontsize=7.5)
    out = HERE / "fig16a_single_node.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(out)


if __name__ == "__main__":
    main()
