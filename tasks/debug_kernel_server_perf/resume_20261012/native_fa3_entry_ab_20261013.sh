#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
output_dir=$workdir/native_fa3_entry_ab_20261013
predecessor_pid=487089
predecessor_log=$workdir/native_pdl_serving_20261013.log
trap 'status=$?; printf "NATIVE_FA3_ENTRY_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
echo "WAIT_NATIVE_SERVING $(date -Is)"
while kill -0 "$predecessor_pid" 2>/dev/null; do
  predecessor_state=$(ps -p "$predecessor_pid" -o stat= || true)
  if [[ "$predecessor_state" == Z* ]]; then break; fi
  if ! predecessor_command=$(tr '\0' ' ' < "/proc/$predecessor_pid/cmdline"); then break; fi
  [[ "$predecessor_command" == "bash native_pdl_serving_20261013.sh "* || "$predecessor_command" == "bash /dev/shm/w/native_pdl_serving_20261013.sh "* ]]
  sleep 15
done
grep -q '^NATIVE_PDL_SERVING_ALL_DONE ' "$predecessor_log"
grep -q '^NATIVE_PDL_SERVING_RUN_EXIT status=0 ' "$predecessor_log"
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference' >/dev/null; then
  echo "GPU predecessor still running; refusing duplicate launch"
  exit 1
fi
cd "$workdir"
sha256sum -c native_fa3_entry_overlay_20261013/baseline.sha256
test ! -e fi_native_pdl_v4
mkdir "$output_dir"
cp -a fi_native_pdl_v3 fi_native_pdl_v4
cp -a native_fa3_entry_overlay_20261013/include/. fi_native_pdl_v4/include/
sha256sum -c native_fa3_entry_overlay_20261013/candidate.sha256
export PYTHONPATH=$workdir/up_native_pdl_v3/python HF_HOME=$workdir/hf
export SGLANG_FUSED_FI_ROOT=$workdir/fi_native_pdl_v4 SGLANG_FUSED_MERGED_DIR=$workdir/merged_native_pdl_v4
export TORCH_EXTENSIONS_DIR=$workdir/torch_ext_native_pdl_v4 TORCH_CUDA_ARCH_LIST=9.0a
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto4096 HEADS=28,4 SM_SPLIT=112
export FUSED_REFERENCE_WORKSPACE_BYTES=1073741824 FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7
export TIME_CUDA_GRAPH=1 TIME_ITERS=100 TIME_WARMUP=20 NATIVE_PDL_MODE=independent
unset FUSED_NOSPLIT FUSED_KV_SHARE_BUILD FUSED_CLUSTER FLASHINFER_FUSED_CLUSTER FUSED_FA3_CLUSTER_PAIR
unset FLASHINFER_FUSED_SYMMETRIC FLASHINFER_FUSED_SKIP_FA3 FLASHINFER_FUSED_PLAN_DEBUG FA3_SKIP_KV_LOAD TIME_FORCE
driver=$workdir/drivers/fused_reference_native_pdl_v3.py
echo "BUILD_START $(date -Is)"
timeout --kill-after=30 1500 "$workdir/venv/bin/python" -u -c \
  'from sglang.srt.layers.attention.fused_fa3_loader import load_fused_fa3; print(load_fused_fa3(verbose=True, cta_q=128))' \
  > "$output_dir/build.log" 2>&1
cuobjdump --dump-resource-usage "$TORCH_EXTENSIONS_DIR/fused_fa3_q128/fused_fa3_q128.so" > "$output_dir/resources.txt"
for dedicated in 0 1; do
  FUSED_FA3_DEDICATED_PRIMARY=$dedicated NATIVE_PDL_TRACE=1 NATIVE_PDL_REQUIRE_OVERLAP=1 \
    NATIVE_PDL_GRAPH_PLAN=1 MODE=mixed_verify PREFILL_QO=512 PREFILL_KV=65536 VERIFY_QO=4 \
    VERIFY_ROWS=8 VERIFY_KV=8192 TIME_AB=0 timeout --kill-after=20 240 "$workdir/venv/bin/python" \
    -u "$driver" > "$output_dir/gate_$dedicated.log" 2>&1
  grep -q '^PASS$' "$output_dir/gate_$dedicated.log"
  test "$(grep -c 'completed_fa2_during_fa3=True' "$output_dir/gate_$dedicated.log")" = 2
done
unset NATIVE_PDL_TRACE NATIVE_PDL_REQUIRE_OVERLAP NATIVE_PDL_GRAPH_PLAN
shapes=("prefill2048_v8|MODE=mixed_verify PREFILL_QO=2048 PREFILL_KV=65536 VERIFY_QO=4 VERIFY_ROWS=8 VERIFY_KV=16384"
        "append512_v46|MODE=mixed_verify PREFILL_QO=512 PREFILL_KV=49152 VERIFY_QO=4 VERIFY_ROWS=46 VERIFY_KV=49152"
        "fa3_8x1024|MODE=band1 BAND_SIDE=fa3 BAND_QO=1024 BAND_KV=32768 BAND_ROWS=8"
        "split_q32_v8|MODE=mixed_verify PREFILL_QO=32 PREFILL_KV=65536 VERIFY_QO=4 VERIFY_ROWS=8 VERIFY_KV=8192")
for shape in "${shapes[@]}"; do
  read -r -a fixture_env <<< "${shape#*|}"
  for repeat in 1 2; do
    for dedicated in 0 1; do
      label=${shape%%|*}_dedicated${dedicated}_$repeat
      echo "START $label $(date -Is)"
      FUSED_FA3_DEDICATED_PRIMARY=$dedicated TIME_AB=1 env "${fixture_env[@]}" \
        timeout --kill-after=20 300 "$workdir/venv/bin/python" -u "$driver" > "$output_dir/$label.log" 2>&1
      "$workdir/venv/bin/python" - "$output_dir/$label.log" "$dedicated" "$repeat" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
log = path.read_text()
assert log.rstrip().endswith("PASS") and "Traceback" not in log and "NON-FINITE OUTPUT" not in log, path
rows = re.findall(r"bad rows (\d+)/(\d+)", log)
replay = re.findall(r"FUSED REPLAY CORRECTNESS: bad_heads=(\d+)/(\d+)", log)
controls = re.findall(r"UNFUSED CORRECTNESS: bad_heads=(\d+)/(\d+)", log)
assert rows and replay and controls and "[time] cuda_graph=True" in log, path
assert all(int(bad) == 0 and int(total) > 0 for bad, total in rows + replay + controls), path
times = {}
for kernel in ("fused", "unfused"):
    matched = re.findall(r"^\[time\] " + kernel + r"\s+=\s+([0-9.]+) ms/iter", log, re.MULTILINE)
    assert len(matched) == 1 and float(matched[0]) > 0, path
    times[kernel + "_ms"] = float(matched[0])
audit = {"case": path.stem, "dedicated": bool(int(sys.argv[2])), "repeat": int(sys.argv[3]),
         **times, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit), flush=True)
PY
    done
  done
done
echo "NATIVE_FA3_ENTRY_DONE $(date -Is)"
