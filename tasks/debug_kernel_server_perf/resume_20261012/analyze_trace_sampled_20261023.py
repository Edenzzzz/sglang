"""Stage-1 kernel result on trace-PROPORTIONAL captures (2026-10-23).

Input: kernel_results*.json from bench_captured_mixed_20261023.py run over captures saved with CAPTURE_EVERY, i.e.
every Nth real mixed serving step, so the capture set has the trace's own shape distribution.
Per capture: best FA3 = min over every single- and dual-stream FA3 variant measured; native = the named native arm;
graph replay, averaged over both arm orders. Aggregates:
  time-weighted speedup = sum(best FA3) / sum(native)   -- total mixed-step attention time, what serving sees
  95% CI by bootstrap over captures (10,000 resamples), win rate, per-bucket breakdown.
  python analyze_trace_sampled_20261023.py <native_arm> [--baseline best|default] <kernel_results.json> [...]
  --baseline best     min over every FA3 variant per capture: an ORACLE chosen per step with hindsight
  --baseline default  fa3_full_default, the single varlen call sglang actually makes: the DEPLOYABLE baseline
  --baseline fixed    the ONE FA3 variant with the least total time over all captures, applied to every step: what a
                      single global num_splits / packing setting could achieve (deployable, stronger than default)
Claude, 2026-10-23.
"""
import collections
import json
import random
import statistics
import sys

arm = sys.argv[1]
args = sys.argv[2:]
baseline = "best"
if args and args[0] == "--baseline":
    baseline, args = args[1], args[2:]
for path in args:
    d = json.load(open(path))
    t = collections.defaultdict(lambda: collections.defaultdict(list))
    worst = 0.0
    for r in d["records"]:
        worst = max(worst, r["worst_head_relative_error"])
        if r["style"] == "graph":
            t[r["capture"]][r["arm"]].append(r["milliseconds"])
    fixed = None
    if baseline == "fixed":
        names = [a for a in next(iter(t.values())) if a.startswith("fa3_")]
        fixed = min(names, key=lambda a: sum(statistics.mean(arms[a]) for arms in t.values()))
    rows = []
    for cap, arms in sorted(t.items()):
        if baseline == "best":
            fa3 = min(statistics.mean(v) for a, v in arms.items() if a.startswith("fa3_"))
        else:
            fa3 = statistics.mean(arms[fixed or "fa3_full_default"])
        if arm not in arms:
            raise SystemExit(f"{path}: capture {cap} has no {arm} records")
        rows.append((cap.split("_", 1)[1].rsplit(".", 1)[0], fa3, statistics.mean(arms[arm])))
    sw = sum(f for _, f, _ in rows) / sum(n for _, _, n in rows)
    rng = random.Random(0)
    boots = []
    for _ in range(10000):
        s = [rows[rng.randrange(len(rows))] for _ in rows]
        boots.append(sum(f for _, f, _ in s) / sum(n for _, _, n in s))
    boots.sort()
    lo, hi = boots[250], boots[9749]
    wins = sum(1 for _, f, n in rows if n < f)
    by = collections.defaultdict(list)
    for b, f, n in rows:
        by[b].append((f, n))
    parts = "  ".join(f"{b} n={len(v)} {sum(f for f, _ in v) / sum(n for _, n in v):.3f}x"
                      for b, v in sorted(by.items()))
    name = path.split("/")[-2]
    label = f"fixed FA3 ({fixed})" if fixed else f"{baseline} FA3"
    print(f"{name:12s} {arm} vs {label}: {len(rows)} captures, time-weighted {sw:.3f}x [95% CI {lo:.3f}-{hi:.3f}], "
          f"native faster on {wins}/{len(rows)}, FP32 worst {worst:.4f} | {parts}")
