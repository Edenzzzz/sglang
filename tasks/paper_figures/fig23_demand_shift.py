"""Figure 23 of the paper draft: the superposed TraceLab replay at alpha = 3 with the chat surge (+16 req/s from 300 s
for 300 s), 16 H20s, Qwen2.5-7B, in 30 s windows by request start. Four stacked panels on one time axis (no dual
axis): issued prefill demand (uncached prompt tokens / s), issued decode demand (requested output tokens / s), chat
attainment (2 s + 50 ms, base stream) and agent attainment (10 s + 80 ms), for ours and conditional disaggregation.
Issued demand differs by system because each round goes out only after its predecessor returns.
Input: tasks/perf_model/tracelab_windows.py --windows 30 outputs, copied to data/demand_shift/.
  python fig23_demand_shift.py   -> fig23_demand_shift.png
Claude, 2026-10-27.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ARMS = [("ours", "coloc_slo_edf_mtp_hc_tl2_a3_surge_r1", "#2a78d6"),
        ("conditional disagg.", "cond_p1_edf_mtp_hc_tl2_a3_surge_r1", "#eb6834")]
PANELS = [("prefill_demand_tok_s", "issued prefill\n(K uncached tok/s)", 1e-3),
          ("decode_demand_tok_s", "issued decode\n(K output tok/s)", 1e-3),
          ("chat_attain_tight", "chat attainment\n2 s + 50 ms (%)", 1.0),
          ("agent_attain_tight", "agent attainment\n10 s + 80 ms (%)", 1.0)]
SURGE = (300, 600)
MUTED, GRID = "#52514e", "#e4e3df"


def main():
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(7.2, 5.6), sharex=True, constrained_layout=True)
    for label, name, color in ARMS:
        rows = json.loads((HERE / "data" / "demand_shift" / f"{name}.json").read_text())
        horizon = 1800  # BENCH_REPLAY_HORIZON_S at alpha 3; later windows hold only drain
        rows = [r for r in rows if r["t"] < horizon]
        t = [r["t"] + 15 for r in rows]
        for ax, (key, ylabel, scale) in zip(axes, PANELS):
            ys = [None if r.get(key) is None else r[key] * scale for r in rows]
            ax.plot(t, ys, color=color, linewidth=1.6, label=label)
    for ax, (_, ylabel, _) in zip(axes, PANELS):
        ax.axvspan(*SURGE, color="#e4e3df", alpha=0.6, linewidth=0)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[2:]:
        ax.set_ylim(0, 102)
    axes[0].text(SURGE[0] + 8, axes[0].get_ylim()[1] * 0.9, "chat surge (+16 req/s)", fontsize=7, color=MUTED, va="top")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)
    axes[-1].set_xlabel("time since release (s)")
    out = HERE / "fig23_demand_shift.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(out)


if __name__ == "__main__":
    main()
