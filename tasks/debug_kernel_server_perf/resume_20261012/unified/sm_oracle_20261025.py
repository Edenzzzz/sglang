"""SM pre-assignment oracle (A6): per capture, the best of the unified arms run at different FA3 SM budgets, against
native v3, time-weighted per set; and which budget wins how often. Reads the stage-1 kernel_results.json files.
  python sm_oracle_20261025.py <kernel_results.json> [...]
Claude, 2026-10-25.
"""
import collections
import json
import statistics
import sys

ARMS = {"u_np2s_stock_geometry": 78, "u_np2s_sm70": 70, "u_np2s_sm62": 62, "u_np2s_sm54": 54, "u_np2s_sm46": 46}
for path in sys.argv[1:]:
    t = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in json.load(open(path))["records"]:
        if r["style"] == "graph":
            t[r["capture"]][r["arm"]].append(r["milliseconds"])
    v3 = full = best = 0.0
    wins = collections.Counter()
    for cap, arms in t.items():
        m = {a: statistics.mean(arms[a]) for a in ARMS if a in arms}
        if len(m) != len(ARMS) or "native_pdl_geometry_v3" not in arms:
            continue
        b = min(m, key=m.get)
        wins[ARMS[b]] += 1
        v3 += statistics.mean(arms["native_pdl_geometry_v3"])
        full += m["u_np2s_stock_geometry"]
        best += m[b]
    print(f"{path.split('/')[-2]:12s} vs v3: full SMs {v3 / full:.3f}x, per-step oracle {v3 / best:.3f}x; "
          f"winning budget counts {dict(sorted(wins.items(), reverse=True))}")
