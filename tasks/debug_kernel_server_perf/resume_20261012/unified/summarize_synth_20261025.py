"""Per-capture table of the synthetic MELD-shaped steps: mean graph-replay ms of every arm (both orders), and each arm
relative to a baseline arm (time ratio baseline / arm, > 1 = faster).
  python summarize_synth_20261025.py <baseline arm> <kernel_results.json> [...]
Claude, 2026-10-25.
"""
import json
import sys
from collections import defaultdict

base = sys.argv[1]
for path in sys.argv[2:]:
    rep = json.load(open(path))
    t = defaultdict(lambda: defaultdict(list))
    for r in rep["records"]:
        t[r["capture"]][r["arm"]].append(r["milliseconds"])
    arms = sorted({a for c in t.values() for a in c})
    print(f"SYNTH_TABLE {path.split('/')[-2]} (ms; ratio = {base} / arm)")
    print("  capture".ljust(26) + "".join(a[:22].rjust(24) for a in arms))
    for cap in sorted(t):
        m = {a: sum(v) / len(v) for a, v in t[cap].items()}
        b = m.get(base)
        cells = [(f"{m[a]:.4f}" + (f" ({b / m[a]:.3f})" if b and a != base else "")) if a in m else "-" for a in arms]
        print("  " + cap[:24].ljust(24) + "".join(c.rjust(24) for c in cells))
