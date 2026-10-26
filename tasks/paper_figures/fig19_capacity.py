"""Figure 19 of the paper draft: capacity at SLO on 16 H20s, full stack.

Two rows (Qwen2.5-7B, Llama-3.3-70B) x two columns (chat SLO attainment at 2 s + 50 ms; agent SLO goodput at 10 s
(20 s above 200K uncached tokens) + 80 ms), against agent programs, for ours, conditional disaggregation and stock PD
at its two best splits. Each point is the mean of the repeats, with a bar from min to max. Input: the PAPER lines of
tasks/perf_model/fanout_per_tenant.py (scale x1) saved under data/; repeats are taken in file order per (arm, load).
  python fig19_capacity.py   -> fig19_capacity.png next to this script
Claude, 2026-10-26.
"""
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LINE = re.compile(r"^PAPER (\S+)\s.*\| x1: chat_tight=\s*([0-9.]+)% chat_loose=\s*([0-9.]+)% agent_tight=\s*([0-9.]+)tok/s")
ARMS = [  # label, arm prefix, color (validated categorical slots 1-4), marker
    ("ours", "coloc_slo", "#2a78d6", "o"),
    ("conditional disagg.", "cond_p1", "#eb6834", "s"),
    ("stock PD 2 + 2", "disagg_p2", "#1baf7a", "^"),
    ("stock PD 3 + 1", "disagg_p3", "#eda100", "D"),
]
MODELS = [("Qwen2.5-7B (16 TP1 workers)", "stage3_qwen_paper_lines.txt", r"_s3_L(\d+)", 4),
          ("Llama-3.3-70B (4 TP4 workers)", "stage3_llama_paper_lines.txt", r"_s3llama_L(\d+)", 4)]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(path, load_pattern):
    points = defaultdict(list)  # (arm, programs) -> [(chat, goodput)]
    for line in (HERE / "data" / path).read_text().splitlines():
        m = LINE.match(line)
        if not m:
            continue
        name = m.group(1)
        programs = re.search(load_pattern, name)
        arm = next((prefix for _, prefix, _, _ in ARMS if name.startswith(prefix)), None)
        if programs and arm:
            points[(arm, int(programs.group(1)))].append((float(m.group(2)), float(m.group(4))))
    return points


def main():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    for row, (title, path, pattern, _) in enumerate(MODELS):
        points = load(path, pattern)
        for col, (metric, ylabel) in enumerate(((0, "chat attainment, 2 s + 50 ms (%)"),
                                                (1, "agent SLO goodput (tok/s)"))):
            ax = axes[row][col]
            ends = []
            for label, prefix, color, marker in ARMS:
                xs = sorted(p for a, p in points if a == prefix)
                if not xs:
                    continue
                vals = [[v[metric] for v in points[(prefix, x)]] for x in xs]
                mean = [sum(v) / len(v) for v in vals]
                lo = [m - min(v) for m, v in zip(mean, vals)]
                hi = [max(v) - m for m, v in zip(mean, vals)]
                ax.errorbar(xs, mean, yerr=[lo, hi], color=color, marker=marker, markersize=6, linewidth=2,
                            capsize=3, label=label, markeredgecolor="#fcfcfb", markeredgewidth=1)
                ends.append([mean[-1], xs[-1], label])
            if metric == 0:
                ax.axhline(90, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
                ax.set_ylim(0, 102)
                ax.set_yticks([0, 20, 40, 60, 80, 90, 100])
            # Direct labels at each line's end, pushed apart so none overlap (min gap 7% of the axis).
            lo_y, hi_y = ax.get_ylim()
            gap = 0.07 * (hi_y - lo_y)
            ends.sort()
            for k in range(1, len(ends)):
                ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
            for y, x, label in ends:
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 0), fontsize=7, color=MUTED,
                            va="center")
            ax.set_xlabel("agent programs (chat at programs / 4 req/s)")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=9, color=INK, loc="left")
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.margins(x=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06), fontsize=8)
    out = HERE / "fig19_capacity.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(out)


if __name__ == "__main__":
    main()
