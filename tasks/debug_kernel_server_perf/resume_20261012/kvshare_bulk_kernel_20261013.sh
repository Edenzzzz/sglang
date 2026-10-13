#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
output_dir=$workdir/kvshare_bulk_kernel_20261013
predecessor_pid=501745
predecessor_log=$workdir/native_pdl_compact_validate_20261013.log
trap 'status=$?; printf "KVSHARE_BULK_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
echo "WAIT_NATIVE_COMPACT $(date -Is)"
while kill -0 "$predecessor_pid" 2>/dev/null; do
  predecessor_state=$(ps -p "$predecessor_pid" -o stat= || true)
  if [[ "$predecessor_state" == Z* ]]; then break; fi
  if ! predecessor_command=$(tr '\0' ' ' < "/proc/$predecessor_pid/cmdline"); then break; fi
  [[ "$predecessor_command" == "bash native_pdl_compact_validate_20261013.sh "* || "$predecessor_command" == "bash /dev/shm/w/native_pdl_compact_validate_20261013.sh "* ]]
  sleep 15
done
grep -q '^NATIVE_PDL_COMPACT_DONE ' "$predecessor_log"
grep -q '^NATIVE_PDL_COMPACT_EXIT status=0 ' "$predecessor_log"
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference' >/dev/null; then
  echo "GPU predecessor still running; refusing duplicate launch"
  exit 1
fi
cd "$workdir"
sha256sum -c kvshare_bulk_overlay_20261013/baseline.sha256
test ! -e fi_kvshare_bulk
mkdir "$output_dir"
cp -a fi_kvshare fi_kvshare_bulk
cp -a kvshare_bulk_overlay_20261013/port07/. fi_kvshare_bulk/
sha256sum -c kvshare_bulk_overlay_20261013/candidate.sha256
cp kvshare_bulk_overlay_20261013/baseline.sha256 kvshare_bulk_overlay_20261013/candidate.sha256 "$output_dir/"
export PYTHONPATH=$workdir/up_kvshare/python HF_HOME=$workdir/hf
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto HEADS=28,4 SM_SPLIT=112 FUSED_REFERENCE_WORKSPACE_BYTES=1073741824
export TIME_CUDA_GRAPH=1 TIME_ITERS=100 TIME_WARMUP=20 FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7
export TORCH_CUDA_ARCH_LIST=9.0a FA3_STAGES=2
unset FLASHINFER_FUSED_PLAN_DEBUG TIME_FORCE FA3_SKIP_KV_LOAD FLASHINFER_FUSED_SYMMETRIC
unset FLASHINFER_FUSED_SKIP_FA3
unset FUSED_NOSPLIT FUSED_FA3_DEDICATED_PRIMARY
export SGLANG_FUSED_FI_ROOT=$workdir/fi_kvshare_bulk SGLANG_FUSED_MERGED_DIR=$workdir/merged_kvshare_bulk
export TORCH_EXTENSIONS_DIR=$workdir/torch_ext_kvshare_bulk
export FUSED_KV_SHARE_BUILD=1 FUSED_CLUSTER=2 FLASHINFER_FUSED_CLUSTER=2 FUSED_FA3_CLUSTER_PAIR=1
select_arm() {
  export FUSED_FA3_KV_SHARE=0 FUSED_FA3_KV_SHARE_TRACE=0 FUSED_FA3_KV_SHARE_BULK=0
  if [ "$1" = SHARE ]; then export FUSED_FA3_KV_SHARE=1; fi
  if [ "$1" = BULK ]; then export FUSED_FA3_KV_SHARE=1 FUSED_FA3_KV_SHARE_BULK=1; fi
}
select_arm CLUSTER
echo "BUILD_START $(date -Is)"
timeout --kill-after=30 1500 "$workdir/venv/bin/python" -u -c \
  'from sglang.srt.layers.attention.fused_fa3_loader import load_fused_fa3; print(load_fused_fa3(verbose=True, cta_q=128))' \
  > "$output_dir/build.log" 2>&1
echo "BUILD_DONE $(date -Is)"
for arm in CLUSTER SHARE BULK; do
  select_arm "$arm"
  export FUSED_FA3_KV_SHARE_TRACE=1
  echo "GATE_START $arm $(date -Is)"
  MODE=band1 BAND_SIDE=fa3 BAND_KV=4096 BAND_ROWS=1 BAND_QO=512 TIME_AB=0 \
    timeout --kill-after=20 180 "$workdir/venv/bin/python" -u "$workdir/drivers/fused_reference.py" \
    > "$output_dir/gate_$arm.log" 2>&1
  "$workdir/venv/bin/python" - "$output_dir/gate_$arm.log" "$arm" <<'PY'
from pathlib import Path
import re
import sys

log = Path(sys.argv[1]).read_text()
assert log.rstrip().endswith("PASS") and "Traceback" not in log
assert "NON-FINITE OUTPUT" not in log
rows = re.findall(r"bad rows (\d+)/(\d+)", log)
assert rows and all(int(bad) == 0 and int(total) > 0 for bad, total in rows)
counts = [int(count) for count in re.findall(r"FA3_KV_SHARE items=(\d+)", log)]
assert counts, "missing device execution counter"
assert (max(counts) > 0) == (sys.argv[2] != "CLUSTER"), counts
bulk = re.findall(r"FA3_KV_SHARE bulk_items=(\d+) bulk_copies=(\d+)", log)
assert bulk, "missing bulk execution counter"
assert all((int(items) > 0 and int(copies) > 0) if sys.argv[2] == "BULK"
           else (int(items) == 0 and int(copies) == 0) for items, copies in bulk), bulk
print("KVSHARE_BULK_GATE", sys.argv[2], counts, bulk, flush=True)
PY
done
unset FUSED_FA3_KV_SHARE_TRACE
shapes=("small_q512|MODE=band1 BAND_SIDE=fa3 BAND_KV=4096 BAND_ROWS=1 BAND_QO=512"
        "odd_q513|MODE=band1 BAND_SIDE=fa3 BAND_KV=4096 BAND_ROWS=1 BAND_QO=513"
        "odd_kv4097_q513|MODE=band1 BAND_SIDE=fa3 BAND_KV=4097 BAND_ROWS=3 BAND_QO=513"
        "app_kv65536_r4_q512|MODE=band1 BAND_SIDE=fa3 BAND_KV=65536 BAND_ROWS=4 BAND_QO=512"
        "app_kv32768_r8_q1024|MODE=band1 BAND_SIDE=fa3 BAND_KV=32768 BAND_ROWS=8 BAND_QO=1024"
        "app_kv131072_r2_q256|MODE=band1 BAND_SIDE=fa3 BAND_KV=131072 BAND_ROWS=2 BAND_QO=256"
        "mixed_verify_q4|MODE=mixed_verify VERIFY_QO=4"
        "mixed_verify_q8|MODE=mixed_verify VERIFY_QO=8"
        "fa2_only|MODE=band1 BAND_SIDE=fa2 BAND_KV=8192 BAND_ROWS=8 BAND_QO=4")
for shape in "${shapes[@]}"; do
  read -r -a fixture_env <<< "${shape#*|}"
  for repeat in 1 2; do
    arms=(CLUSTER SHARE BULK)
    if [ "$repeat" = 2 ]; then arms=(BULK SHARE CLUSTER); fi
    for arm in "${arms[@]}"; do
      select_arm "$arm"
      label=${shape%%|*}_${arm}_${repeat}
      echo "START $label $(date -Is)"
      env | LC_ALL=C sort | grep -E '^(PYTHONPATH=|SGLANG_FUSED_|FUSED_|FLASHINFER_|TIME_|HEADS=|SM_SPLIT=)' \
        > "$output_dir/$label.environment.txt"
      TIME_AB=1 env "${fixture_env[@]}" timeout --kill-after=20 240 "$workdir/venv/bin/python" \
        -u "$workdir/drivers/fused_reference.py" > "$output_dir/$label.log" 2>&1
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
assert "FA3_KV_SHARE " not in log, "timing run has device printf enabled"
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
    "label": path.stem, "arm": sys.argv[2], "repeat": int(sys.argv[3]), **timings,
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
assert len(audits) == 54, len(audits)
(root / "execution_audits.json").write_text(json.dumps(audits, indent=2) + "\n")
PY
echo "KVSHARE_BULK_DONE $(date -Is)"
