import functools
import json
import os
from pathlib import Path
import re
import socket

import srv_correctness


def main():
    root = Path(os.environ["FP32_ADJUDICATION_ROOT"])
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        srv_correctness.PORT = probe.getsockname()[1]
    srv_correctness.post = functools.partial(srv_correctness.post, timeout=1800)
    arm = "MTP_MIX_FP32_ADJUDICATION"
    srv_correctness.run_arm(arm, str(root / "results"))
    outputs = json.loads((root / "results" / (arm + ".json")).read_text())
    requests = {name: value for name, value in outputs.items() if not name.startswith("_")}
    if len(requests) != 16 or any(not isinstance(value, list) or len(value) < 8 for value in requests.values()):
        raise RuntimeError("Incomplete serving outputs")
    log = (root / "results" / ("server_" + arm + ".log")).read_text(errors="replace")
    if re.search(r"Traceback|CUDA error\b|watchdog timeout", log, re.IGNORECASE):
        raise RuntimeError("Serving diagnostic logged a fatal error")
    rows = [json.loads(line) for line in (root / "shadow.jsonl").read_text().splitlines()]
    if not rows or not all(row["finite"] and row["baseline_repeat_exact"] and row["native_serial_exact"] and row["committed_sample_exact"] and row["committed_prefill_exact"] for row in rows):
        raise RuntimeError("Invalid native or sampler controls")
    adjudicated = [row for row in rows if "fp32_adjudication" in row]
    disagreements = [row for row in rows if row["verify_requests_with_changed_emission"] or row["prefill_requests_with_changed_emission"]]
    if not adjudicated or any("fp32_adjudication" not in row for row in disagreements):
        raise RuntimeError("Missing FP32 coverage of disputed decisions")
    decisions = [decision for row in adjudicated for decision in row["fp32_adjudication"]["disputed_verify_decisions"]]
    total_rows = sum(row["rows"] for row in adjudicated)
    report = dict(
        forwards=len(rows), logit_rows=sum(row["rows"] for row in rows),
        adjudicated_forwards=len(adjudicated), adjudicated_logit_rows=total_rows,
        raw_argmax_flips=sum(row["argmax_flips"] for row in rows),
        verify_emission_changes=sum(row["verify_requests_with_changed_emission"] for row in rows),
        prefill_emission_changes=sum(row["prefill_requests_with_changed_emission"] for row in rows),
        disputed_verify_requests=len(decisions),
        fp32_agrees_with_fa3=sum(decision["agrees_with_fa3"] for decision in decisions),
        fp32_agrees_with_native=sum(decision["agrees_with_native"] for decision in decisions),
        fp32_agrees_with_neither=sum(not decision["agrees_with_fa3"] and not decision["agrees_with_native"] for decision in decisions),
        mean_kl_fp32_to_fa3=sum(row["fp32_adjudication"]["fa3_vs_fp32"]["mean_kl_ref_to_candidate"] * row["rows"] for row in adjudicated) / total_rows,
        mean_kl_fp32_to_native=sum(row["fp32_adjudication"]["native_vs_fp32"]["mean_kl_ref_to_candidate"] * row["rows"] for row in adjudicated) / total_rows,
        all_fp32_repeats_exact=all(row["fp32_adjudication"]["fp32_logits_repeat_exact"] and row["fp32_adjudication"]["fp32_hidden_repeat_exact"] for row in adjudicated),
        all_native_state_restorations_exact=all(row["fp32_adjudication"]["native_logits_restored_exact"] and row["fp32_adjudication"]["native_hidden_restored_exact"] for row in adjudicated),
        same_input_emissions_equal=not disagreements,
        scope="Disputed steps plus every 16th mixed forward; FP32 attention, unchanged BF16 GEMMs and attention output boundaries; no timing claim",
    )
    if all("original_row_tile_control" in row for row in rows):
        report.update(
            original_tile_forwards_exact_to_native=sum(row["original_row_tile_control"]["exact_to_native"] for row in rows),
            original_tile_argmax_differences_from_native=sum(row["original_row_tile_control"]["argmax_diff_from_native"] for row in rows),
            original_tile_emission_changes_from_fa3=sum(row["original_row_tile_control"]["emission_changes_from_reference"] for row in rows),
            mean_kl_fp32_to_original_tile=sum(row["fp32_adjudication"]["original_row_tile_vs_fp32"]["mean_kl_ref_to_candidate"] * row["rows"] for row in adjudicated) / total_rows,
        )
    if all("bf16_partial_control" in row for row in rows):
        report.update(
            bf16_partial_argmax_differences_from_native=sum(row["bf16_partial_control"]["argmax_diff_from_native"] for row in rows),
            bf16_partial_emission_changes_from_fa3=sum(row["bf16_partial_control"]["emission_changes_from_reference"] for row in rows),
            mean_kl_fp32_to_bf16_partial=sum(row["fp32_adjudication"]["bf16_partial_vs_fp32"]["mean_kl_ref_to_candidate"] * row["rows"] for row in adjudicated) / total_rows,
            adjudicated_verify_emission_changes_from_fp32={
                name: sum(row["fp32_adjudication"]["verify_emission_changes_from_fp32"][name] for row in adjudicated)
                for name in ("reference", "native", "bf16_partial")
            },
        )
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FP32_SERVING_ADJUDICATION_COMPLETE", json.dumps(report), flush=True)
    raise SystemExit(0 if report["same_input_emissions_equal"] else 3)


if __name__ == "__main__":
    main()
