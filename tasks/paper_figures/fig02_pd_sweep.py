"""Figure 2 of the paper draft: stock PD disaggregation needs a tuned P:D ratio, and even its best ratio misses the chat
SLO. Chat attainment and agent SLO goodput against the number of prefill pods (of 8), Qwen2.5-7B without speculation,
KV pool on, FA3, scheduler-only settings (chat SLO 0.5 s + 20 ms):
  16 H20s (8 pods x 2), 32 agent programs + chat 8 req/s   (mn_capacity_16g_20261023.sh)
  32 H20s (8 pods x 4), 64 agent programs + chat 16 req/s  (mn_capacity_32g_20261023.sh, chat attainment only)
with ours (colocated, same settings) as a reference line. Numbers from tasks/pd_colocation_vs_disagg.md, capacity
phase A.
  python fig02_pd_sweep.py   -> fig02_pd_sweep.png
Claude, 2026-10-26.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PODS = [1, 2, 3, 4, 5, 6, 7]
CHAT_16 = [16.2, 37.3, 51.5, 60.4, 66.1, 67.8, 50.0]
GOODPUT_16 = [86, 201, 230, 296, 320, 359, 326]
CHAT_32 = [10.5, 20.6, 31.5, 44.2, 48.9, 55.9, 48.7]
OURS_16 = (99.0, 568)   # Table 9, 32 programs + 8 req/s
OURS_32_CHAT = 99.1     # 32-GPU capacity table, 64 programs + 16 req/s
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def main():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    a.plot(PODS, CHAT_16, color=BLUE, marker="o", linewidth=2, markersize=6, markeredgecolor="#fcfcfb",
           label="stock PD, 16 GPUs, 32 programs")
    a.plot(PODS, CHAT_32, color=ORANGE, marker="s", linewidth=2, markersize=6, markeredgecolor="#fcfcfb",
           label="stock PD, 32 GPUs, 64 programs")
    a.axhline(OURS_16[0], color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    a.text(1, OURS_16[0] - 6, f"ours (colocated): {OURS_16[0]:.0f}% / {OURS_32_CHAT:.0f}%", fontsize=7, color=MUTED)
    a.set_ylim(0, 105)
    a.set_ylabel("chat attainment, 0.5 s + 20 ms (%)")
    a.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.86))
    b.plot(PODS, GOODPUT_16, color=BLUE, marker="o", linewidth=2, markersize=6, markeredgecolor="#fcfcfb")
    b.axhline(OURS_16[1], color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    b.text(1, OURS_16[1] + 12, f"ours (colocated): {OURS_16[1]} tok/s", fontsize=7, color=MUTED)
    b.set_ylim(0, 650)
    b.set_ylabel("agent SLO goodput (tok/s)")
    b.set_title("16 GPUs, 32 programs + chat 8 req/s", fontsize=8, color=INK, loc="left")
    for ax in (a, b):
        ax.set_xlabel("prefill pods of 8 (the rest decode)")
        ax.set_xticks(PODS)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    out = HERE / "fig02_pd_sweep.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(out)


if __name__ == "__main__":
    main()
