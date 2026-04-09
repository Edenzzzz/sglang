#!/usr/bin/env python3
"""Compute SLO attainment + summarize EDF benchmark results.

Per-workload SLOs (must match run_edf_sweep.sh):
  sharegpt:      TTFT 2000ms, ITL 100ms
  code_feedback: TTFT 3000ms, ITL 150ms
  swe:           TTFT 5000ms, ITL 150ms

For each result JSON file, compute:
  - throughput (output tok/s)
  - median/P99 TTFT, median/P99 ITL
  - SLO attainment % (fraction of requests meeting BOTH TTFT and ITL SLOs)
"""
import json
import os
import sys
from glob import glob
from statistics import median

SLO = {
    "sharegpt":      {"ttft_ms": 2000, "itl_ms": 100},
    "code_feedback": {"ttft_ms": 3000, "itl_ms": 150},
    "swe":           {"ttft_ms": 5000, "itl_ms": 150},
    "nebius_swe":    {"ttft_ms": 5000, "itl_ms": 150},
    "swe_bench_pro": {"ttft_ms": 4000, "itl_ms": 150},
}

# TTFT length scaling: matches edf_scheduler.compute_ttft_slo_ms
def ttft_slo_for_input(base_ms: float, input_len: int,
                       scale_tokens: float = 5000, scale_ms: float = 1000) -> float:
    extra = max(0, input_len - 4096)
    return base_ms + (extra / scale_tokens) * scale_ms


def percentile(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def analyze(path):
    with open(path) as f:
        d = json.load(f)
    fname = os.path.basename(path).replace(".json", "")
    parts = fname.split("_rate")
    method_wl = parts[0]
    rate = parts[1]
    # method is first token (PP/DMC/LPT_CG); LPT_CG has _ in it
    if method_wl.startswith("LPT_CG_"):
        method = "LPT_CG"
        wl = method_wl[len("LPT_CG_"):]
    else:
        method, wl = method_wl.split("_", 1)

    slo = SLO.get(wl, SLO["sharegpt"])
    base_ttft_ms = slo["ttft_ms"]
    itl_slo_ms = slo["itl_ms"]

    ttfts = d.get("ttfts", [])
    itls_per_req = d.get("itls", [])
    input_lens = d.get("input_lens", [])  # may not exist
    if not input_lens:
        # Fallback: assume base SLO for everyone
        input_lens = [0] * len(ttfts)

    n = len(ttfts)
    n_pass_max = 0       # strict: max ITL ≤ SLO
    n_pass_tpot = 0      # DistServe-style: mean ITL (TPOT) ≤ SLO
    n_pass_ttft = 0
    n_pass_itl_max = 0
    n_pass_tpot_only = 0
    n_pass_itl_p99 = 0   # Sarathi-Serve-style: per-request P99 ITL ≤ SLO
    for i in range(n):
        ttft_ms = ttfts[i] * 1000.0
        ttft_budget = ttft_slo_for_input(base_ttft_ms, input_lens[i] if i < len(input_lens) else 0)
        ttft_ok = ttft_ms <= ttft_budget

        itl_list = itls_per_req[i] if i < len(itls_per_req) else []
        itl_ms_list = [x * 1000.0 for x in itl_list]
        max_itl = max(itl_ms_list) if itl_ms_list else 0
        mean_itl = (sum(itl_ms_list) / len(itl_ms_list)) if itl_ms_list else 0  # TPOT
        p99_itl = percentile(itl_ms_list, 99) if itl_ms_list else 0
        itl_max_ok = max_itl <= itl_slo_ms
        tpot_ok = mean_itl <= itl_slo_ms
        p99_ok = p99_itl <= itl_slo_ms

        if ttft_ok:
            n_pass_ttft += 1
        if itl_max_ok:
            n_pass_itl_max += 1
        if tpot_ok:
            n_pass_tpot_only += 1
        if p99_ok:
            n_pass_itl_p99 += 1
        if ttft_ok and itl_max_ok:
            n_pass_max += 1
        if ttft_ok and tpot_ok:
            n_pass_tpot += 1

    return {
        "method": method,
        "workload": wl,
        "rate": rate,
        "throughput": d.get("output_throughput", 0),
        "duration": d.get("duration", 0),
        "median_ttft_ms": d.get("median_ttft_ms", 0),
        "p99_ttft_ms": d.get("p99_ttft_ms", 0),
        "median_itl_ms": d.get("median_itl_ms", 0),
        "p99_itl_ms": d.get("p99_itl_ms", 0),
        "n": n,
        # Strict: TTFT + max-ITL ≤ SLO
        "slo_attain_pct": 100.0 * n_pass_max / max(n, 1),
        # DistServe-style: TTFT + TPOT (mean ITL) ≤ SLO
        "slo_tpot_attain_pct": 100.0 * n_pass_tpot / max(n, 1),
        "ttft_attain_pct": 100.0 * n_pass_ttft / max(n, 1),
        "itl_attain_pct": 100.0 * n_pass_itl_max / max(n, 1),
        "tpot_attain_pct": 100.0 * n_pass_tpot_only / max(n, 1),
        "itl_p99_attain_pct": 100.0 * n_pass_itl_p99 / max(n, 1),
    }


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "tasks/edf_benchmark/results_edf"
    files = sorted(glob(os.path.join(results_dir, "*.json")))
    if not files:
        print(f"No results in {results_dir}")
        return

    rows = []
    for f in files:
        if "_rate" not in os.path.basename(f) or "summary" in f:
            continue
        try:
            rows.append(analyze(f))
        except Exception as e:
            print(f"  skip {f}: {e}")

    # Group by workload then method
    rows.sort(key=lambda r: (r["workload"], r["method"], float(r["rate"]) if r["rate"] != "inf" else 1e9))

    print(f"\n{'method':<8} {'wl':<14} {'rate':>5} {'tok/s':>8} {'med_TTFT':>9} {'p99_TTFT':>9} {'med_ITL':>8} {'p99_ITL':>8} {'SLO_max%':>9} {'TTFT%':>6} {'ITLmax%':>8} {'ITLp99%':>8} {'TPOT%':>6}")
    print("-" * 130)
    for r in rows:
        print(f"{r['method']:<8} {r['workload']:<14} {r['rate']:>5} {r['throughput']:>8.0f} "
              f"{r['median_ttft_ms']:>9.1f} {r['p99_ttft_ms']:>9.1f} "
              f"{r['median_itl_ms']:>8.2f} {r['p99_itl_ms']:>8.2f} "
              f"{r['slo_attain_pct']:>8.1f}% "
              f"{r['ttft_attain_pct']:>5.1f}% {r['itl_attain_pct']:>7.1f}% "
              f"{r['itl_p99_attain_pct']:>7.1f}% {r['tpot_attain_pct']:>5.1f}%")

    # Save as JSON for later plotting
    out = os.path.join(results_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
