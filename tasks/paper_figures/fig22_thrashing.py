"""Figure 22 of the paper draft: cache thrashing (ThunderAgent's setting). 16 H20s (4 pods x 4), Qwen2.5-7B on FA3, 32
agent programs, the working set swept by the number of distinct 64K-token agent tasks (16 / 64 / 128 = ~1M / 4M / 8M
live KV tokens against ~6.7M of cluster KV capacity). Ours (colocated + cluster pool) vs stock PD 2 + 2. Panels: cached
prompt fraction, agent output tok/s, mean agent TTFT. Numbers from tasks/pd_colocation_vs_disagg.md (point 2,
mn_thrash_16g_20261022.sh).
  python fig22_thrashing.py   -> fig22_thrashing.png
Claude, 2026-10-26.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LIVE_M = [1, 4, 8]
SERIES = {  # label: (cached fraction, agent tok/s, agent TTFT s), color, marker
    "ours": (([0.980, 0.966, 0.949], [597.5, 570.7, 534.5], [0.60, 0.62, 0.68]), "#2a78d6", "o"),
    "stock PD 2 + 2": (([0.890, 0.681, 0.605], [313.1, 155.9, 141.8], [7.03, 20.26, 22.58]), "#1baf7a", "^"),
}
MUTED, GRID, INK = "#52514e", "#e4e3df", "#0b0b0b"


def main():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
    titles = ("cached prompt fraction", "agent output tok/s", "mean agent TTFT (s)")
    for k, ax in enumerate(axes):
        for label, (values, color, marker) in SERIES.items():
            ax.plot(LIVE_M, values[k], color=color, marker=marker, linewidth=2, markersize=6,
                    markeredgecolor="#fcfcfb", label=label)
            ax.annotate(label, (LIVE_M[-1], values[k][-1]), textcoords="offset points", xytext=(6, 0), fontsize=7,
                        color=MUTED, va="center")
        ax.axvline(6.7, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_title(titles[k], fontsize=8, color=INK, loc="left")
        ax.set_xlabel("live KV working set (M tokens)")
        ax.set_xticks(LIVE_M)
        ax.set_xlim(0, 11.5)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0.5, 1.02)
    axes[0].text(6.8, 0.53, "cluster KV\ncapacity", fontsize=7, color=MUTED)
    axes[1].set_ylim(0, 680)
    axes[2].set_ylim(0, 26)
    out = HERE / "fig22_thrashing.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(out)


if __name__ == "__main__":
    main()
