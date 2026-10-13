#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
output_dir=$workdir/b3_append_resume_20261013
predecessor=$workdir/a3b_split_serving_20261013
trap 'status=$?; printf "B3_APPEND_RESUME_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
echo "WAIT_A3B_SPLIT $(date -Is)"
exec 9>"$workdir/a2_kernel.lock"
flock 9
grep -q '^A3B_SPLIT_ALL_DONE ' "$predecessor.log"
grep -q '^A3B_SPLIT_RUN_EXIT status=0 ' "$predecessor.log"
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running; refusing duplicate launch"
  exit 1
fi
"$workdir/venv/bin/python" - "$predecessor" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
for mixture in ("mix32", "mix128"):
    for arm in ("DEFAULT_1", "A3B_1", "DEFAULT_2", "A3B_2"):
        audit = json.loads((root / f"{mixture}_r24" / arm / "execution_audit.json").read_text())
        assert audit["mixture"] == mixture and audit["arm"] == arm, audit
        assert audit["split_calls"] > 0 and audit["mixed_steps"] > 0, audit
        assert audit["graph_decode_lines"] > 0, audit
        assert not any(audit[key] for key in ("overlapped", "fallbacks", "tracebacks")), audit
PY
sha256sum -c "$predecessor/sources.sha256"
mkdir "$output_dir"
export PYTHONPATH=$workdir/validated_upstream/python SGLANG_FUSED_FI_ROOT=$workdir/fi_skip
export SGLANG_FUSED_MERGED_DIR=$workdir/merged_skip
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto HEADS=28,4 SM_SPLIT=112 FUSED_REFERENCE_WORKSPACE_BYTES=1073741824
export TIME_AB=1 TIME_CUDA_GRAPH=1 TIME_ITERS=50 TIME_WARMUP=10 FUSED_FA3_CTA_Q=128
unset FLASHINFER_FUSED_PLAN_DEBUG TIME_FORCE
sha256sum "$workdir/drivers/fused_reference.py" \
  "$workdir/validated_upstream/python/sglang/srt/layers/attention/fused_fa3_loader.py" \
  "$workdir/b3_append_resume_20261013.sh" > "$output_dir/sources.sha256"
find "$workdir/fi_skip" -type f \( -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.py' \) \
  -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum >> "$output_dir/sources.sha256"
env | LC_ALL=C sort | grep -E '^(PYTHONPATH=|SGLANG_FUSED_|FUSED_|FLASHINFER_|TIME_|HEADS=|SM_SPLIT=)' \
  > "$output_dir/environment.txt"
shapes=("app_kv65536_r4_q512|MODE=band1 BAND_SIDE=fa3 BAND_KV=65536 BAND_ROWS=4 BAND_QO=512"
        "app_kv32768_r8_q1024|MODE=band1 BAND_SIDE=fa3 BAND_KV=32768 BAND_ROWS=8 BAND_QO=1024"
        "app_kv131072_r2_q256|MODE=band1 BAND_SIDE=fa3 BAND_KV=131072 BAND_ROWS=2 BAND_QO=256")
for skip in 0 1; do
  export FA3_SKIP_KV_LOAD=$skip TORCH_EXTENSIONS_DIR=$workdir/torch_ext_skip$skip
  for shape in "${shapes[@]}"; do
    label=skip${skip}_${shape%%|*}
    read -r -a fixture_env <<< "${shape#*|}"
    force_env=()
    if [ "$skip" = 1 ]; then force_env=(TIME_FORCE=1); fi
    echo "START $label $(date -Is)"
    returncode=0
    env "${fixture_env[@]}" "${force_env[@]}" timeout 1500 "$workdir/venv/bin/python" \
      -u "$workdir/drivers/fused_reference.py" > "$output_dir/$label.log" 2>&1 || returncode=$?
    "$workdir/venv/bin/python" - "$output_dir/$label.log" "$skip" "$returncode" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
skip, returncode = map(int, sys.argv[2:])
log = path.read_text()
assert returncode == skip, f"unexpected exit {returncode}: {path}"
assert "Traceback" not in log and "NON-FINITE OUTPUT" not in log, path
assert "[time] cuda_graph=True" in log, path
rows = re.findall(r"bad rows (\d+)/(\d+)", log)
replay = re.findall(r"FUSED REPLAY CORRECTNESS: bad_heads=(\d+)/(\d+)", log)
controls = re.findall(r"UNFUSED CORRECTNESS: bad_heads=(\d+)/(\d+)", log)
assert rows and replay and controls, path
bad_rows, total_rows = map(int, rows[-1])
bad_heads, total_heads = map(int, replay[-1])
assert total_rows > 0 and total_heads > 0, path
assert all(int(bad) == 0 and int(total) > 0 for bad, total in controls), path
if skip:
    assert bad_rows == total_rows and bad_heads == total_heads, path
    assert "DIAGNOSTIC ARM:" in log and log.rstrip().endswith("FAIL"), path
else:
    assert bad_rows == 0 and bad_heads == 0 and log.rstrip().endswith("PASS"), path
timings = re.findall(r"^\[time\] fused\s+=\s+([0-9.]+) ms/iter", log, re.MULTILINE)
assert len(timings) == 1, path
milliseconds = float(timings[0])
assert math.isfinite(milliseconds) and milliseconds > 0, path
audit = {
    "label": path.stem, "skip_kv_load": bool(skip), "returncode": returncode,
    "diagnostic_only": bool(skip), "bad_rows": bad_rows, "total_rows": total_rows,
    "replay_bad_heads": bad_heads, "fused_ms": milliseconds,
    "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
}
path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit), flush=True)
PY
    echo "END $label $(date -Is)"
  done
done
"$workdir/venv/bin/python" - "$output_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
summary = []
for normal_path in sorted(root.glob("skip0_*.audit.json")):
    normal = json.loads(normal_path.read_text())
    diagnostic = json.loads((root / normal_path.name.replace("skip0_", "skip1_", 1)).read_text())
    summary.append({
        "shape": normal["label"].removeprefix("skip0_"), "normal_ms": normal["fused_ms"],
        "skip_kv_diagnostic_ms": diagnostic["fused_ms"],
        "diagnostic_latency_reduction_pct": 100 * (1 - diagnostic["fused_ms"] / normal["fused_ms"]),
        "multicast_implemented": False,
    })
assert len(summary) == 3, summary
(root / "ceiling_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
PY
echo "B3_APPEND_RESUME_DONE $(date -Is)"
