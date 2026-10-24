import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def query_tile_evidence(log, requested):
    records = re.findall(
        r"NATIVE_FA2_QUERY_TILE path=(native_query_tile_calls|native_query_tile_bypass_calls) requested=(\d+) calls=(\d+) maximum_packed=(\d+) original_cta=(\d+)",
        log,
    )
    if not records or any(int(record[1]) != requested for record in records):
        raise ValueError("Missing or mismatched native query-tile evidence")
    selected = [record for record in records if record[0] == "native_query_tile_calls"]
    bypassed = [record for record in records if record[0] == "native_query_tile_bypass_calls"]
    if not selected or any(not 16 < int(record[3]) <= 32 or int(record[4]) != 64 for record in selected):
        raise ValueError("Unproven native query-tile shape or no executed selected calls")
    return {
        "requested_native_query_tile": requested,
        "query_tile_selected_calls_max": max(int(record[2]) for record in selected),
        "query_tile_bypass_calls_max": max((int(record[2]) for record in bypassed), default=0),
        "query_tile_selected_packed_extents": sorted({int(record[3]) for record in selected}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--query-tile", type=int, choices=(32, 64), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--dispatch-auditor", type=Path, required=True)
    parser.add_argument("--base-auditor", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run(
        [sys.executable, str(args.dispatch_auditor), str(args.directory), "MTPMIXFUSED",
         "--stock", "1", "--phase", "2", "--budget", "132", "--guard", "0",
         "--reference", str(args.reference), "--base-auditor", str(args.base_auditor)],
        check=True,
    )
    evidence = query_tile_evidence((args.directory / "server_MTPMIXFUSED.log").read_text(), args.query_tile)
    path = args.directory / "stock_serving_audit.json"
    report = json.loads(path.read_text())
    report.update(evidence)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("NATIVE_QUERY_TILE_SERVING_PASS", json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
