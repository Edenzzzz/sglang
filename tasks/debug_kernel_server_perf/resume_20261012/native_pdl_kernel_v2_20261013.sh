#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
output_dir=$workdir/native_pdl_kernel_v2_20261013
predecessor_pid=445761
predecessor_log=$workdir/fa2_auto4096_wide_20261013.log
trap 'status=$?; printf "NATIVE_PDL_KERNEL_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
echo "WAIT_WIDE_SERVING $(date -Is)"
while kill -0 "$predecessor_pid" 2>/dev/null; do
  predecessor_state=$(ps -p "$predecessor_pid" -o stat= || true)
  if [[ "$predecessor_state" == Z* ]]; then break; fi
  if ! predecessor_command=$(tr '\0' ' ' < "/proc/$predecessor_pid/cmdline"); then break; fi
  [[ "$predecessor_command" == "bash fa2_auto4096_wide_20261013.sh "* || "$predecessor_command" == "bash /dev/shm/w/fa2_auto4096_wide_20261013.sh "* ]]
  sleep 15
done
grep -q '^FA2_AUTO4096_WIDE_DONE ' "$predecessor_log"
grep -q '^FA2_AUTO4096_WIDE_EXIT status=0 ' "$predecessor_log"
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference' >/dev/null; then
  echo "GPU predecessor still running; refusing duplicate launch"
  exit 1
fi
cd "$workdir"
sha256sum -c native_pdl_overlay_v2_20261013/baseline.sha256
test ! -e fi_native_pdl_v2
mkdir "$output_dir"
cp -a port07 fi_native_pdl_v2
cp -a native_pdl_overlay_v2_20261013/port07/. fi_native_pdl_v2/
cp native_pdl_overlay_v2_20261013/fused_reference_native_pdl.py drivers/fused_reference_native_pdl_v2.py
sha256sum -c native_pdl_overlay_v2_20261013/candidate.sha256
cp native_pdl_overlay_v2_20261013/baseline.sha256 native_pdl_overlay_v2_20261013/candidate.sha256 "$output_dir/"
export PYTHONPATH=$workdir/up_racefix/python HF_HOME=$workdir/hf
export SGLANG_FUSED_FI_ROOT=$workdir/fi_native_pdl_v2 SGLANG_FUSED_MERGED_DIR=$workdir/merged_native_pdl_v2
export TORCH_EXTENSIONS_DIR=$workdir/torch_ext_native_pdl_v2 TORCH_CUDA_ARCH_LIST=9.0a
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto4096 FUSED_NOSPLIT=1 HEADS=28,4 SM_SPLIT=112
export FUSED_REFERENCE_WORKSPACE_BYTES=1073741824 FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7
export TIME_CUDA_GRAPH=1 TIME_ITERS=100 TIME_WARMUP=20
unset FUSED_KV_SHARE_BUILD FUSED_CLUSTER FLASHINFER_FUSED_CLUSTER FUSED_FA3_CLUSTER_PAIR
unset FUSED_FA3_KV_SHARE FUSED_FA3_KV_SHARE_TRACE FLASHINFER_FUSED_SYMMETRIC
unset FLASHINFER_FUSED_SKIP_FA3 FLASHINFER_FUSED_PLAN_DEBUG FA3_SKIP_KV_LOAD TIME_FORCE
driver=$workdir/drivers/fused_reference_native_pdl_v2.py
echo "BUILD_START $(date -Is)"
timeout --kill-after=30 1500 "$workdir/venv/bin/python" -u -c \
  'from sglang.srt.layers.attention.fused_fa3_loader import load_fused_fa3; print(load_fused_fa3(verbose=True, cta_q=128))' \
  > "$output_dir/build.log" 2>&1
echo "BUILD_DONE $(date -Is)"
rejection_status=0
env -u FUSED_NOSPLIT NATIVE_PDL_MODE=serial MODE=band1 BAND_SIDE=fa3 BAND_KV=65536 \
  BAND_ROWS=1 BAND_QO=32 TIME_AB=0 timeout --kill-after=20 240 "$workdir/venv/bin/python" \
  -u "$driver" > "$output_dir/reject_split_fa3.log" 2>&1 || rejection_status=$?
test "$rejection_status" = 1
grep -q 'native FA2 primary requires unsplit FA3' "$output_dir/reject_split_fa3.log"
echo "NATIVE_PDL_SPLIT_REJECTION_PASS $(date -Is)"
for arm in serial entry independent; do
  echo "TRACE_START $arm $(date -Is)"
  NATIVE_PDL_MODE=$arm NATIVE_PDL_TRACE=1 NATIVE_PDL_REQUIRE_OVERLAP=1 NATIVE_PDL_GRAPH_PLAN=1 \
    MODE=mixed_verify PREFILL_QO=512 PREFILL_KV=65536 VERIFY_QO=4 VERIFY_ROWS=8 VERIFY_KV=8192 \
    TIME_AB=0 timeout --kill-after=20 240 "$workdir/venv/bin/python" -u "$driver" \
    > "$output_dir/trace_$arm.log" 2>&1
  "$workdir/venv/bin/python" - "$output_dir/trace_$arm.log" "$arm" <<'PY'
from pathlib import Path
import ast
import re
import sys

log = Path(sys.argv[1]).read_text()
assert log.rstrip().endswith("PASS") and "Traceback" not in log and "NON-FINITE OUTPUT" not in log
rows = re.findall(r"bad rows (\d+)/(\d+)", log)
assert rows and all(int(bad) == 0 and int(total) > 0 for bad, total in rows)
traces = re.findall(r"NATIVE_PDL_TRACE mode=(\w+) graph=(True|False) stamps=(\[[^\]]+\]) completed_fa2_during_fa3=(True|False)", log)
assert len(traces) == 2 and {graph for _, graph, _, _ in traces} == {"True", "False"}, traces
for arm, graph, encoded, reported in traces:
    stamps = ast.literal_eval(encoded)
    assert arm == sys.argv[2] and len(stamps) == 4, traces
    assert 0 < stamps[0] < stamps[1] and 0 < stamps[2] <= stamps[3] < (1 << 63) - 1, stamps
    overlap = stamps[0] < stamps[3] < stamps[1]
    assert overlap == (reported == "True") == (arm == "independent"), traces
    if arm != "independent":
        assert stamps[2] >= stamps[1], stamps
print("NATIVE_PDL_TRACE_GATE", traces, flush=True)
PY
done
unset NATIVE_PDL_TRACE NATIVE_PDL_REQUIRE_OVERLAP NATIVE_PDL_GRAPH_PLAN
shapes=("fa2_r8_kv8192_q4|MODE=band1 BAND_SIDE=fa2 BAND_KV=8192 BAND_ROWS=8 BAND_QO=4"
        "fa2_r46_kv49152_q4|MODE=band1 BAND_SIDE=fa2 BAND_KV=49152 BAND_ROWS=46 BAND_QO=4"
        "fa2_r16_kv131072_q4|MODE=band1 BAND_SIDE=fa2 BAND_KV=131072 BAND_ROWS=16 BAND_QO=4"
        "fa2_r16_kv32769_q8|MODE=band1 BAND_SIDE=fa2 BAND_KV=32769 BAND_ROWS=16 BAND_QO=8"
        "chat512_v46_kv48k|MODE=mixed_verify PREFILL_QO=512 PREFILL_KV=512 VERIFY_QO=4 VERIFY_ROWS=46 VERIFY_KV=49152"
        "append512_v46_kv48k|MODE=mixed_verify PREFILL_QO=512 PREFILL_KV=49152 VERIFY_QO=4 VERIFY_ROWS=46 VERIFY_KV=49152"
        "prefill2048_v8|MODE=mixed_verify PREFILL_QO=2048 PREFILL_KV=65536 VERIFY_QO=4 VERIFY_ROWS=8 VERIFY_KV=16384"
        "odd513_v16_q8|MODE=mixed_verify PREFILL_QO=513 PREFILL_KV=4097 VERIFY_QO=8 VERIFY_ROWS=16 VERIFY_KV=32769")
for shape in "${shapes[@]}"; do
  read -r -a fixture_env <<< "${shape#*|}"
  for repeat in 1 2; do
    for arm in legacy serial independent; do
      label=${shape%%|*}_${arm}_${repeat}
      echo "START $label $(date -Is)"
      NATIVE_PDL_MODE=$arm TIME_AB=1 env "${fixture_env[@]}" \
        timeout --kill-after=20 300 "$workdir/venv/bin/python" -u "$driver" > "$output_dir/$label.log" 2>&1
      "$workdir/venv/bin/python" - "$output_dir/$label.log" "$arm" "$repeat" <<'PY'
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
audit = {"label": path.stem, "arm": sys.argv[2], "repeat": int(sys.argv[3]), **timings,
         "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
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
echo "NATIVE_PDL_KERNEL_DONE $(date -Is)"

