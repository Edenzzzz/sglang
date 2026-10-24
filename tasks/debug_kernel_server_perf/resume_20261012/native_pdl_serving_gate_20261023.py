"""Real EAGLE3 output comparison; numerical gates are not performance measurements."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket

import srv_correctness


def compare(reference, candidate):
    rows = []
    for name in sorted(key for key in reference if not key.startswith("_")):
        expected, actual = reference[name], candidate[name]
        matched = next(
            (index for index, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]),
            min(len(expected), len(actual)),
        )
        rows.append(dict(
            request=name, reference_tokens=len(expected), candidate_tokens=len(actual),
            common_prefix=matched, full_equal=expected == actual,
            reference_verify_count=reference.get("_verify_ct_" + name),
            candidate_verify_count=candidate.get("_verify_ct_" + name),
        ))
    return dict(full_equal=all(row["full_equal"] for row in rows), requests=rows)


def main():
    root = Path(os.environ["NATIVE_PDL_SERVING_ROOT"])
    output = root / "results"
    if output.exists():
        raise ValueError("Serving comparison output already exists")
    output.mkdir()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        srv_correctness.PORT = probe.getsockname()[1]
    os.environ["SRV_ARGS"] = "--attention-backend fa3 --chunked-prefill-size 512 --disable-cuda-graph"
    os.environ["SGLANG_MIXED_SPEC_MIN_CTX"] = "8192"
    os.environ["SGLANG_MIXED_SPEC_MIN_QUEUE"] = "4"
    records = {}
    for arm, enabled in (("MTP_MIX_BASE_R1", False), ("MTP_MIX_NATIVE_PDL", True), ("MTP_MIX_BASE_R2", False)):
        os.environ["SGLANG_NATIVE_PDL"] = "1" if enabled else "0"
        os.environ["SGLANG_NATIVE_PDL_VERIFY"] = "1" if enabled else "0"
        prior_baseline = os.environ.get("NATIVE_PDL_PRIOR_BASELINE")
        if arm == "MTP_MIX_BASE_R1" and prior_baseline:
            for name in (arm + ".json", "server_" + arm + ".log"):
                shutil.copy2(Path(prior_baseline) / name, output / name)
            print("REUSED_COMPLETED_BASELINE", prior_baseline, flush=True)
        else:
            srv_correctness.run_arm(arm, str(output))
        result = json.loads((output / (arm + ".json")).read_text())
        requests = {key: value for key, value in result.items() if not key.startswith("_")}
        if len(requests) != 16 or any(not isinstance(value, list) or len(value) < 8 for value in requests.values()):
            raise RuntimeError(f"Missing or truncated generated outputs in {arm}")
        log = (output / ("server_" + arm + ".log")).read_text(errors="replace")
        if re.search(r"Traceback|CUDA error\b|NATIVE_PDL_VERIFY failed|watchdog timeout", log, re.IGNORECASE):
            raise RuntimeError(f"Serving failure in {arm}")
        if enabled:
            for marker in ("NATIVE_PDL_PLAN", "NATIVE_PDL_VERIFY", "NATIVE_PDL_CALL", "row_q=4", "MIXEDSPEC_STEPS"):
                if marker not in log:
                    raise RuntimeError(f"Missing serving activation evidence: {marker}")
            if not re.search(r"NATIVE_PDL_CALL[^\n]*row_q=4 verify_tokens=[1-9][0-9]*", log):
                raise RuntimeError("No native PDL execution on real merged verify rows")
        records[arm] = result
    reference = records["MTP_MIX_BASE_R1"]
    candidate = records["MTP_MIX_NATIVE_PDL"]
    repeat = records["MTP_MIX_BASE_R2"]
    report = dict(
        scope="Same native FA3 backend and real EAGLE3 mixed schedule; candidate has per-layer reference checks enabled; not a timing run",
        baseline_repeat=compare(reference, repeat),
        candidate_vs_baseline1=compare(reference, candidate),
        candidate_vs_baseline2=compare(repeat, candidate),
        runtime_and_attention_checks_passed=True,
        executor_sha256=hashlib.sha256((root / "native_fa3_fa2_pdl_20261023.py").read_bytes()).hexdigest(),
        backend_sha256=hashlib.sha256((root / "python/sglang/srt/layers/attention/flashattention_backend.py").read_bytes()).hexdigest(),
    )
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print("NATIVE_PDL_SERVING_COMPARISON_DONE", json.dumps({
        key: value["full_equal"] for key, value in report.items() if isinstance(value, dict)
    }), flush=True)


if __name__ == "__main__":
    main()
