import argparse
import hashlib
import json
from pathlib import Path
import re


parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
parser.add_argument("method")
parser.add_argument("--stock", type=int, required=True)
parser.add_argument("--phase", type=int, required=True)
parser.add_argument("--budget", type=int, required=True)
parser.add_argument("--reference", type=Path)
args = parser.parse_args()
programs = json.loads((args.directory / f"{args.method}_agent_programs.json").read_text())
assert programs["sessions"] == 48 and programs["rounds"] == 96
assert programs["concurrency_limit"] == programs["peak_active_programs"] == 24
active = set()
peak = 0
for event in programs["events"]:
    if event["event"] == "start":
        assert event["session_id"] not in active
        active.add(event["session_id"])
    else:
        assert event["event"] == "end" and event["session_id"] in active
        active.remove(event["session_id"])
    peak = max(peak, len(active))
    assert len(active) <= 24
assert peak == 24 and not active
path = args.directory / f"server_{args.method}.log"
log = path.read_text()
assert "Traceback" not in log and "fallback" not in log.lower()
assert "cuda graph: True" in log and "EDF_RUNTIME_" not in log
def counter(pattern):
    matches = re.findall(pattern, log)
    return int(matches[-1]) if matches else 0
mixed = counter(r"MIXEDSPEC_STEPS (\d+)")
split = counter(r"FUSED_SPLIT_CALLS (\d+)")
fused = counter(r"FUSED_KERNEL_CALLS (\d+)")
stock = counter(r"STOCK_FA3_SERVING path=stock_fa3_calls calls=(\d+)")
assert (mixed > 0) == args.method.startswith("MTPMIX")
assert (split > 0) == (args.method == "MTPMIX_SPLIT")
assert (fused > 0) == (args.method == "MTPMIXFUSED")
assert (stock > 0) == bool(args.stock)
if stock:
    phases = re.findall(r"STOCK_FA3_SERVING path=stock_fa3_calls calls=\d+ phase=(\d+)", log)
    assert all(int(value) == args.phase for value in phases)
summary = {"method": args.method, "phase": args.phase, "sm_budget": args.budget,
           "program_peak": peak, "program_mean": programs["mean_active_programs"],
           "mixed_steps": mixed, "split_calls": split, "fused_calls": fused, "stock_calls": stock,
           "server_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "streams": {}}
for stream, count in (("agent", 96), ("chat", 2048)):
    path = args.directory / f"{args.method}_{stream}.json"
    result = json.loads(path.read_text())
    assert result["completed"] == count and not any(result["errors"])
    assert len(result["request_keys"]) == len(result["token_events"]) == count
    if args.reference is not None:
        reference = json.loads((args.reference / f"MTP_LPT_{stream}.json").read_text())
        for key in ("request_keys", "requested_output_lens", "output_lens"):
            assert result[key] == reference[key], (stream, key)
    summary["streams"][stream] = {key: value for key, value in result.items() if isinstance(value, (int, float, bool))}
    summary["streams"][stream].update(
        first_start=min(result["request_start_times"]), last_end=max(result["request_end_times"]),
        result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
if args.reference is not None:
    prior = json.loads((args.reference / "MTP_LPT_agent_programs.json").read_text())
    assert programs["session_rounds"] == prior["session_rounds"]
starts = [values["first_start"] for values in summary["streams"].values()]
ends = [values["last_end"] for values in summary["streams"].values()]
summary["overlap_fraction"] = max(0, min(ends) - max(starts)) / (max(ends) - min(starts))
(args.directory / "stock_serving_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
print("STOCK_SERVING_PASS", json.dumps(summary), flush=True)
