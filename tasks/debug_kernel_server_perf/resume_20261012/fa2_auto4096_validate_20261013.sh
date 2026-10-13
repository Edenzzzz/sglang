#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
output_dir=$workdir/fa2_auto4096_validate_20261013
predecessor_pid=390462
predecessor_log=$workdir/a3b_split_midrange_20261013.log
trap 'status=$?; printf "FA2_AUTO4096_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
echo "WAIT_MIDRANGE $(date -Is)"
while kill -0 "$predecessor_pid" 2>/dev/null; do
  predecessor_state=$(ps -p "$predecessor_pid" -o stat= || true)
  if [[ "$predecessor_state" == Z* ]]; then break; fi
  if ! predecessor_command=$(tr '\0' ' ' < "/proc/$predecessor_pid/cmdline"); then break; fi
  [[ "$predecessor_command" == "bash /dev/shm/w/a3b_split_midrange_20261013.sh "* ]]
  sleep 15
done
grep -q '^A3B_SPLIT_MIDRANGE_ALL_DONE ' "$predecessor_log"
grep -q '^A3B_SPLIT_MIDRANGE_RUN_EXIT status=0 ' "$predecessor_log"
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running; refusing duplicate launch"
  exit 1
fi
sha256sum -c "$workdir/a3b_split_midrange_20261013/sources.sha256"
sha256sum -c "$workdir/fi_auto4096_header.sha256"
mkdir "$output_dir"
export PYTHONPATH=$workdir/up_racefix/python HF_HOME=$workdir/hf
export SGLANG_FUSED_FI_ROOT=$workdir/fi_auto4096
export SGLANG_FUSED_MERGED_DIR=$workdir/merged_auto4096
export TORCH_EXTENSIONS_DIR=$workdir/torch_ext_auto4096
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export HEADS=28,4 SM_SPLIT=112 FUSED_REFERENCE_WORKSPACE_BYTES=1073741824
export TIME_AB=1 TIME_CUDA_GRAPH=1 TIME_ITERS=100 TIME_WARMUP=20 FUSED_FA3_CTA_Q=128
unset FLASHINFER_FUSED_PLAN_DEBUG TIME_FORCE FA3_SKIP_KV_LOAD
sha256sum "$workdir/drivers/fused_reference.py" "$workdir/fa2_auto4096_validate_20261013.sh" \
  "$SGLANG_FUSED_FI_ROOT/include/flashinfer/attention/fused_plan.cuh" > "$output_dir/sources.sha256"
env | LC_ALL=C sort | grep -E '^(PYTHONPATH=|SGLANG_FUSED_|FUSED_|FLASHINFER_|TIME_|HEADS=|SM_SPLIT=)' \
  > "$output_dir/environment.txt"
shapes=("b_kv4096_r1_q8|MODE=band1 BAND_SIDE=fa2 BAND_KV=4096 BAND_ROWS=1 BAND_QO=8"
        "b_kv16384_r1_q4|MODE=band1 BAND_SIDE=fa2 BAND_KV=16384 BAND_ROWS=1 BAND_QO=4"
        "b_kv4096_r16_q8|MODE=band1 BAND_SIDE=fa2 BAND_KV=4096 BAND_ROWS=16 BAND_QO=8"
        "b_kv16384_r16_q8|MODE=band1 BAND_SIDE=fa2 BAND_KV=16384 BAND_ROWS=16 BAND_QO=8"
        "mixed_verify_q8|MODE=mixed_verify VERIFY_QO=8"
        "mixed_verify_q4|MODE=mixed_verify VERIFY_QO=4"
        "mix_decode32|MODE=mix_decode DEC_ROWS=32"
        "v_r46_kv16k|MODE=band1 BAND_SIDE=fa2 BAND_KV=16384 BAND_ROWS=46 BAND_QO=4"
        "v_r46_kv48k|MODE=band1 BAND_SIDE=fa2 BAND_KV=49152 BAND_ROWS=46 BAND_QO=4"
        "v_r16_kv128k|MODE=band1 BAND_SIDE=fa2 BAND_KV=131072 BAND_ROWS=16 BAND_QO=4"
        "v_r8_kv8k|MODE=band1 BAND_SIDE=fa2 BAND_KV=8192 BAND_ROWS=8 BAND_QO=4"
        "chunk512_v46_kv48k|MODE=mixed_verify PREFILL_QO=512 PREFILL_KV=49152 VERIFY_QO=4 VERIFY_ROWS=46 VERIFY_KV=49152")
for shape in "${shapes[@]}"; do
  read -r -a fixture_env <<< "${shape#*|}"
  for repeat in 1 2; do
    for policy in auto auto4096; do
      label=${shape%%|*}_${policy}_${repeat}
      echo "START $label $(date -Is)"
      FUSED_FA2_KV_CHUNK=$policy env "${fixture_env[@]}" timeout 1500 "$workdir/venv/bin/python" \
        -u "$workdir/drivers/fused_reference.py" > "$output_dir/$label.log" 2>&1
      "$workdir/venv/bin/python" - "$output_dir/$label.log" "$policy" "$repeat" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
log = path.read_text()
assert log.rstrip().endswith("PASS"), path
assert "Traceback" not in log and "NON-FINITE OUTPUT" not in log, path
assert "[time] cuda_graph=True" in log and "DIAGNOSTIC ARM" not in log, path
rows = re.findall(r"bad rows (\d+)/(\d+)", log)
replay = re.findall(r"FUSED REPLAY CORRECTNESS: bad_heads=(\d+)/(\d+)", log)
controls = re.findall(r"UNFUSED CORRECTNESS: bad_heads=(\d+)/(\d+)", log)
assert rows and replay and controls, path
assert all(int(bad) == 0 and int(total) > 0 for bad, total in rows + replay + controls), path
timings = {}
for kernel in ("fused", "unfused"):
    matches = re.findall(r"^\[time\] " + kernel + r"\s+=\s+([0-9.]+) ms/iter", log, re.MULTILINE)
    assert len(matches) == 1, path
    milliseconds = float(matches[0])
    assert math.isfinite(milliseconds) and milliseconds > 0, path
    timings[kernel + "_ms"] = milliseconds
audit = {
    "label": path.stem, "policy": sys.argv[2], "repeat": int(sys.argv[3]), **timings,
    "rows": int(rows[-1][1]), "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
}
path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit), flush=True)
PY
      echo "END $label $(date -Is)"
    done
  done
done
"$workdir/venv/bin/python" - "$output_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
audits = [json.loads(path.read_text()) for path in sorted(root.glob("*.audit.json"))]
assert len(audits) == 48, len(audits)
(root / "execution_audits.json").write_text(json.dumps(audits, indent=2) + "\n")
PY
echo "FA2_AUTO4096_DONE $(date -Is)"
