#!/usr/bin/env python3
"""Parse ratio sweep raw results into CSV and markdown table."""
import re
import sys

def parse_results(filepath):
    with open(filepath) as f:
        text = f.read()

    # Split by workload markers: "--- ConfigName: in=X out=Y ---"
    pattern = r'--- (.+?): in=(\d+) out=(\d+) ---\n(.*?)(?=--- |\Z)'
    matches = re.findall(pattern, text, re.DOTALL)

    rows = []
    for config, inp, out, block in matches:
        def extract(label):
            m = re.search(rf'{re.escape(label)}:\s+([\d.]+)', block)
            return float(m.group(1)) if m else 0

        rows.append({
            'config': config,
            'input': int(inp),
            'output': int(out),
            'total_tok_s': extract('Total token throughput (tok/s)'),
            'out_tok_s': extract('Output token throughput (tok/s)'),
            'ttft_median': extract('Median TTFT (ms)'),
            'ttft_p99': extract('P99 TTFT (ms)'),
            'itl_median': extract('Median ITL (ms)'),
            'itl_p99': extract('P99 ITL (ms)'),
        })
    return rows

def print_csv(rows):
    print("config,input,output,total_tok_s,out_tok_s,ttft_median_ms,itl_median_ms,ttft_p99_ms,itl_p99_ms")
    for r in rows:
        print(f"{r['config']},{r['input']},{r['output']},{r['total_tok_s']:.0f},{r['out_tok_s']:.0f},{r['ttft_median']:.1f},{r['itl_median']:.2f},{r['ttft_p99']:.1f},{r['itl_p99']:.2f}")

def print_markdown(rows):
    # Group by workload, compare configs side by side
    workloads = {}
    for r in rows:
        key = (r['input'], r['output'])
        workloads.setdefault(key, {})[r['config']] = r

    configs = list(dict.fromkeys(r['config'] for r in rows))

    print(f"\n| Input | Output | Metric | " + " | ".join(configs) + " |")
    print(f"|------:|-------:|--------|" + "|".join(["--------:"] * len(configs)) + "|")

    for (inp, out), by_config in sorted(workloads.items()):
        for metric, key, fmt in [
            ("Total tok/s", "total_tok_s", ".0f"),
            ("Out tok/s", "out_tok_s", ".0f"),
            ("TTFT med ms", "ttft_median", ".1f"),
            ("ITL med ms", "itl_median", ".2f"),
        ]:
            vals = [f"{by_config[c][key]:{fmt}}" if c in by_config else "—" for c in configs]
            prefix = f"| {inp} | {out} |" if metric == "Total tok/s" else "| | |"
            print(f"{prefix} {metric} | " + " | ".join(vals) + " |")

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ratio_sweep_results.txt"
    rows = parse_results(filepath)
    if not rows:
        print("No results found")
        sys.exit(1)
    print(f"Parsed {len(rows)} results\n")
    print_csv(rows)
    print_markdown(rows)
