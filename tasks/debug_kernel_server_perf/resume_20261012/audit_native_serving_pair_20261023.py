"""Recompute kernel-attribution results from complete, identity-matched requests."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from native_serving_protocol_20261023 import assert_same_requests, summarize_stream


METRICS = ("output_tok_s", "mean_ttft_ms", "mean_request_tpot_ms", "token_weighted_mean_itl_ms", "mean_worst_tpot_10_ms")


def audit(root, repeats=2):
    records = []
    reference = None
    source_hashes = None
    for summary_path in sorted(root.glob("*/summary.json")):
        stored = json.loads(summary_path.read_text())
        protocol = stored["protocol"]
        configuration = protocol["configuration"]
        directory = summary_path.parent
        log = (directory / "server.log").read_text(errors="replace")
        if re.search(r"Traceback|CUDA error\b|watchdog timeout", log, re.IGNORECASE):
            raise ValueError("Fatal error in a measured arm")
        for setting in ("'disable_overlap_schedule': False", "'disable_cuda_graph': False", "'attention_backend': 'fa3'"):
            if setting not in log:
                raise ValueError(f"Missing runtime control: {setting}")
        if "cuda graph: True" not in log:
            raise ValueError("No actual graph execution")
        if source_hashes is not None and protocol["source_hashes"] != source_hashes:
            raise ValueError("Sources differ across arms")
        source_hashes = protocol["source_hashes"]
        native = protocol["arm"] == "native"
        if protocol["kernel_environment"]["SGLANG_NATIVE_PDL"] != ("1" if native else "0"):
            raise ValueError("Native activation flag disagrees with arm")
        offset = int((directory / "requests/server_log_offset").read_text())
        measured = (directory / "server.log").read_bytes()[offset:].decode(errors="replace")
        if native != ("NATIVE_PDL_CALL" in measured):
            raise ValueError("Measured native activation disagrees with arm")
        # A four-mode arm merges prefill into verify steps; the --ours stock-MTP control runs without
        # SGLANG_MIXED_SPEC and must show no merge at all.
        merging = protocol["kernel_environment"].get("SGLANG_MIXED_SPEC") == "1"
        if protocol["mode"] == "four" and merging and "MIXEDSPEC_STEPS" not in measured:
            raise ValueError("Four-mode merge is missing")
        if not merging and "MIXEDSPEC_STEPS" in measured:
            raise ValueError("Arm without SGLANG_MIXED_SPEC merged prefill into verify steps")
        programs = json.loads((directory / "requests/agent_programs.json").read_text())
        if programs["peak_active_programs"] != min(int(configuration["AGENT_CONC"]), int(configuration["SESSIONS"])):
            raise ValueError("Program concurrency is not attained")
        tenants, raw, hashes = {}, {}, {}
        for tenant, count in (("agent", int(configuration["SESSIONS"]) * int(configuration["TURNS"])), ("chat", int(configuration["CHAT_N"]))):
            path = directory / "requests" / f"{tenant}.json"
            raw[tenant] = json.loads(path.read_text())
            tenants[tenant] = summarize_stream(raw[tenant], count)
            hashes[tenant] = hashlib.sha256(path.read_bytes()).hexdigest()
            for metric in METRICS:
                if not math.isclose(tenants[tenant][metric], stored["tenants"][tenant][metric], rel_tol=1e-8, abs_tol=1e-8):
                    raise ValueError("Stored metric differs from raw token-event recomputation")
        current = dict(tenants=tenants)
        if reference is not None:
            assert_same_requests(reference, current)
        reference = current
        final_agent = max(raw["agent"]["request_end_times"])
        mixed_chat_requests = sum(start < final_agent for start in raw["chat"]["request_start_times"])
        records.append(dict(
            directory=directory.name, mode=protocol["mode"], arm=protocol["arm"], repeat=protocol["repeat"],
            configuration=configuration, raw_sha256=hashes,
            native_log_records=measured.count("NATIVE_PDL_CALL"), peak_programs=programs["peak_active_programs"],
            chat_start_before_last_agent_fraction=mixed_chat_requests / len(raw["chat"]["request_start_times"]),
            tenants={tenant: {key: value for key, value in metrics.items() if key != "requests_detail"} for tenant, metrics in tenants.items()},
        ))
    expected = {(mode, repeat, arm) for mode in ("four", "two") for repeat in range(1, repeats + 1) for arm in ("fa3", "native")}
    actual = {(record["mode"], record["repeat"], record["arm"]) for record in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError("Incomplete paired matrix")
    comparison = []
    for mode in ("four", "two"):
        for tenant in ("agent", "chat"):
            means = {arm: {metric: statistics.mean(record["tenants"][tenant][metric] for record in records if record["mode"] == mode and record["arm"] == arm) for metric in METRICS} for arm in ("fa3", "native")}
            comparison.append(dict(mode=mode, tenant=tenant, means=means, change_percent={metric: 100 * (means["native"][metric] / means["fa3"][metric] - 1) for metric in METRICS}))
    return dict(
        scope="Exploratory kernel-only attribution; same backend/schedule, real acceptance, fixed finite cohorts, not a quality pass or SLO-capacity claim",
        all_requests_match=True, raw_metrics_recomputed=True, graphs_and_overlap_verified=True,
        source_hashes=source_hashes, records=records, comparisons=comparison,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    report = audit(arguments.root)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    for comparison in report["comparisons"]:
        print(comparison["mode"], comparison["tenant"], comparison["change_percent"])
