"""How much does the prefill-to-decode demand RATIO move within one run, and during a surge?

Motivation 2 claims a static prefill-to-decode split cannot track the workload. What we had measured was that no
swept P:D ratio holds any load point -- which establishes "no static ratio is good enough", not "the right ratio
changes". The ratio itself is the missing quantity, and it is already in the demand-shift figure's data: that
figure plots issued prefill tokens/s and issued decode tokens/s as two panels, so a reader has to divide two
curves by eye.

Note what the load factor can and cannot do. Compressing trace timestamps by alpha makes everything arrive faster
while PRESERVING the trace's prefill-to-decode proportion, so it cannot by itself demonstrate ratio variation. The
variation here comes from two places: the trace's own structure (an agent session alternates long decode with
small appended prefills), and the surge arms, which move the mix in OPPOSITE directions -- a chat surge adds
short-prompt long-decode work at a fixed agent rate, shifting toward decode, while a fan-out spike adds cold
prefills, shifting toward prefill.

  python pd_ratio_20261028.py
Prints, per arm, the ratio's range and how far it moves inside the surge window against outside it.
Claude, 2026-10-28.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SURGE = (300.0, 600.0)   # the chat surge window used by the demand-shift arms

print(f"{'arm':44}{'P/D p10':>9}{'p50':>8}{'p90':>8}{'max/min':>9}{'in surge':>10}{'outside':>9}")
for path in sorted((HERE / "data" / "demand_shift").glob("*.json")):
    rows = json.loads(path.read_text())
    t = np.array([r["t"] for r in rows], dtype=float)
    pre = np.array([r["prefill_demand_tok_s"] for r in rows], dtype=float)
    dec = np.array([r["decode_demand_tok_s"] for r in rows], dtype=float)
    ok = dec > 0
    ratio, tt = pre[ok] / dec[ok], t[ok]
    if len(ratio) < 5:
        continue
    inside = (tt >= SURGE[0]) & (tt < SURGE[1])
    p10, p50, p90 = np.percentile(ratio, [10, 50, 90])
    med_in = np.median(ratio[inside]) if inside.any() else float("nan")
    med_out = np.median(ratio[~inside]) if (~inside).any() else float("nan")
    print(f"{path.stem:44}{p10:>9.2f}{p50:>8.2f}{p90:>8.2f}"
          f"{ratio.max() / max(ratio.min(), 1e-9):>8.1f}x{med_in:>10.2f}{med_out:>9.2f}")
print("\nP/D = issued uncached prompt tokens/s divided by issued output tokens/s, in 30 s windows by request start.")
