import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def route_evidence(log, guard):
    records = re.findall(
        r"KERNEL_DISPATCH_AUDIT guard=(\d+) plans=(\d+) two_sided=(\d+) long_mixed_rows=(\d+) selected_max_kv_tiles=(\d+)",
        log,
    )
    if not records or any(int(record[0]) != guard for record in records):
        raise ValueError("Missing or mismatched in-server dispatch guard evidence")
    long_rows = max(int(record[3]) for record in records)
    if bool(long_rows) != (guard == 0):
        raise ValueError("Unexpected long-context mixed-row routing")
    return {
        "decode_population_guard": guard,
        "long_mixed_planned_rows_max": long_rows,
        "two_sided_plans_max": max(int(record[2]) for record in records),
        "selected_max_kv_tiles": max(int(record[4]) for record in records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("method")
    parser.add_argument("--stock", type=int, required=True)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--guard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--base-auditor", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run(
        [sys.executable, str(args.base_auditor), str(args.directory), args.method,
         "--stock", str(args.stock), "--phase", str(args.phase), "--budget", str(args.budget),
         "--reference", str(args.reference)],
        check=True,
    )
    log = (args.directory / f"server_{args.method}.log").read_text()
    evidence = route_evidence(log, args.guard)
    path = args.directory / "stock_serving_audit.json"
    summary = json.loads(path.read_text())
    summary.update(evidence)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print("STOCK_DISPATCH_PASS", json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
