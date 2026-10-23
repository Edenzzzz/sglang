"""Trace-weighted stage-1 kernel speedup (2026-10-23).

The captured-tensor microbenchmark measures ONE real step per shape bucket (fresh / cached prefix x small / large
query), so its per-bucket table is a coverage sample, not what serving sees: in every serving trace the cached/small
bucket -- source of several headline speedups -- is under 1% of mixed steps. This weights each bucket's measured
speedup by the bucket's frequency in that workload's own census (every mixed step after the workload barrier,
recorded by the same capture run), for two-mode and four-mode separately:
  step-weighted speedup  = sum(n_b * s_b) / sum(n_b)              (mean over serving steps)
  time-weighted speedup  = sum(n_b * fa3_b) / sum(n_b * native_b)  (total mixed-step attention time, FA3 / native)
The time-weighted figure is the one serving throughput sees. Buckets with no census steps get no weight; a bucket
with census steps but no measured capture is reported, not silently dropped.
  python trace_weighted_kernel_20261023.py <results_20261023 dir>
Claude, 2026-10-23.
"""
import collections
import csv
import json
import sys
from pathlib import Path

R = Path(sys.argv[1])
CENSUS = {  # (model, mode, context) -> census of the capture run that produced that row's tensors
    ("qwen25", "two", "mix32"): "captured_mixed/mix32_two",
    ("qwen25", "four", "mix32"): "captured_pair_tuning/captured/mix32_four",
    ("qwen25", "two", "mix128"): "captured_long_policy/mix128_two",
    ("qwen25", "four", "mix128"): "captured_long_policy/mix128_four",
    ("qwen25", "two", "fanout384"): "captured_long_policy/fanout384_two",
    ("qwen25", "four", "fanout384"): "captured_long_policy/fanout384_four",
    ("qwen3", "two", "mix32"): "qwen3_captured/mix32_two",
    ("qwen3", "four", "mix32"): "qwen3_captured/mix32_four",
}
rows = collections.defaultdict(dict)
for r in csv.DictReader(open(R / "geometry_v3_regression/captured_native_v3_all29.csv")):
    rows[(r["model"], r["mode"], r["context"])][r["bucket"]] = (float(r["fa3_ms"]), float(r["native_ms"]))

print(f"{'model/mode/context':28s} {'steps':>5s}  bucket mix (fl/cl/fs/cs)      step-wt  time-wt  unmeasured")
for key, sub in CENSUS.items():
    path = R / sub / "captures/census.jsonl"
    counts = collections.Counter(json.loads(l)["category"] for l in open(path))
    meas = rows.get(key, {})
    num_s = den = fa3_t = nat_t = 0.0
    missing = []
    for b, n in counts.items():
        if b not in meas:
            missing.append(f"{b}:{n}")
            continue
        f, nv = meas[b]
        num_s += n * f / nv
        den += n
        fa3_t += n * f
        nat_t += n * nv
    tot = sum(counts.values())
    mix = "/".join(f"{100 * counts.get(b, 0) / tot:.0f}" for b in ("fresh_large", "cached_large", "fresh_small", "cached_small"))
    print(f"{'/'.join(key):28s} {tot:5d}  {mix:26s} {num_s / den:7.3f}x {fa3_t / nat_t:7.3f}x  {' '.join(missing) or '-'}")
