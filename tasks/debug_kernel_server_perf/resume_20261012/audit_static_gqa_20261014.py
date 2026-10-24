import hashlib
import json
from pathlib import Path
import re
import sys


path = Path(sys.argv[1])
log = path.read_text()
assert log.rstrip().endswith("PASS") and "Traceback" not in log and "NON-FINITE OUTPUT" not in log
rows = re.findall(r"bad rows (\d+)/(\d+)", log)
replay = re.findall(r"FUSED REPLAY CORRECTNESS: bad_heads=(\d+)/(\d+)", log)
controls = re.findall(r"UNFUSED CORRECTNESS: bad_heads=(\d+)/(\d+)", log)
assert rows and replay and controls and "[time] cuda_graph=True" in log
assert all(int(bad) == 0 and int(total) > 0 for bad, total in rows + replay + controls)
plans = re.findall(r"NATIVE_PRIMARY_PLAN max_splits=(\d+) merge_rows=(\d+)", log)
assert len(plans) == 1
splits, merges = map(int, plans[0])
times = {}
for kernel in ("fused", "unfused"):
    matches = re.findall(r"^\[time\] " + kernel + r"\s+=\s+([0-9.]+) ms/iter", log, re.MULTILINE)
    assert len(matches) == 1 and float(matches[0]) > 0
    times[kernel + "_ms"] = float(matches[0])
audit = {
    "case": path.stem,
    "static_gqa": int(sys.argv[2]),
    "sm_budget": sys.argv[3],
    "repeat": int(sys.argv[4]),
    "merge_rows": merges,
    "splits": splits,
    **times,
    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
}
path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print("FA3_STATIC_GQA_CASE_PASS", json.dumps(audit), flush=True)
