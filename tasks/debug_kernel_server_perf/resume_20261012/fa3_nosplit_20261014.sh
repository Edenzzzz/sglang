#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
root=$workdir/fa3_nosplit_20261014
overlay=$workdir/fa3_nosplit_overlay_20261014
predecessor=766826
trap 'status=$?; printf "FA3_NOSPLIT_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
if [ "${INHERIT_LOCK:-0}" = 1 ]; then
  test "${GPU_LOCK_OWNER_PID:-}" = "$predecessor"
  kill -0 "$predecessor"
  test "$(readlink /proc/$$/fd/9)" = "$workdir/a2_kernel.lock"
  test "$(readlink /proc/$predecessor/fd/9)" = "$workdir/a2_kernel.lock"
  test -s "$workdir/b1_kernel_concurrency_20261014/mix64_c24/BASE_2/kernel_concurrency_audit.json"
  echo "FA3_NOSPLIT_INHERITED_LOCK parent=$predecessor $(date -Is)"
else
  echo "FA3_NOSPLIT_WAIT parent=$predecessor $(date -Is)"
  while kill -0 "$predecessor" 2>/dev/null; do
    state=$(ps -p "$predecessor" -o stat= || true)
    if [[ "$state" == Z* ]]; then break; fi
    command=$(tr '\0' ' ' < "/proc/$predecessor/cmdline")
    [[ "$command" == 'bash b1_kernel_concurrency_20261014.sh '* ]]
    sleep 30
  done
  grep -q '^B1_KERNEL_CONCURRENCY_DONE ' "$workdir/b1_kernel_concurrency_20261014.log"
  grep -q '^B1_KERNEL_CONCURRENCY_EXIT status=0 ' "$workdir/b1_kernel_concurrency_20261014.log"
  exec 9>"$workdir/a2_kernel.lock"
fi
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[k]ernel_concurrency_driver|[f]used_reference' >/dev/null; then
  echo 'GPU predecessor still running; refusing duplicate launch'
  exit 1
fi
cd "$workdir"
sha256sum -c "$overlay/baseline.sha256"
test ! -e fi_fa3_nosplit
test ! -e "$root"
mkdir "$root"
cp -a fi_fa3_kv_geometry_v2 fi_fa3_nosplit
cp -a "$overlay/include/." fi_fa3_nosplit/include/
cp "$overlay/fused_fa3_persistent_ext.cu" fi_fa3_nosplit/tests/fused_fa3_persistent_ext.cu
cp "$overlay/fused_reference_native_pdl.py" drivers/fused_reference_fa3_nosplit.py
sha256sum -c "$overlay/candidate.sha256"
cp "$overlay/"{baseline,candidate}.sha256 "$root/"
export PYTHONPATH=$workdir/up_fa3_kv_geometry_v2/python HF_HOME=$workdir/hf
export SGLANG_FUSED_FI_ROOT=$workdir/fi_fa3_nosplit
export SGLANG_FUSED_MERGED_DIR=$workdir/merged_fa3_nosplit TORCH_EXTENSIONS_DIR=$workdir/torch_ext_fa3_nosplit
export TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto4096 HEADS=28,4 SM_SPLIT=112
export FUSED_REFERENCE_WORKSPACE_BYTES=1073741824 FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7
export FA3_CTA_KV=128 FA3_STAGES=2 SGLANG_FUSED_FA3_CONSUMER_REGS=216
export FUSED_FA3_DEDICATED_PRIMARY=1 FUSED_FA3_CLUSTER_PAIR=1
export FUSED_PAGED_TMA_BUILD=0 FUSED_CLUSTER=1 FLASHINFER_FUSED_CLUSTER=1
export TIME_CUDA_GRAPH=1 TIME_ITERS=100 TIME_WARMUP=20 NATIVE_PDL_MODE=independent
unset FUSED_NOSPLIT FUSED_KV_SHARE_BUILD FUSED_FA3_COMPACT_ABI FUSED_KV_CHUNK_MULT
unset FA2_REQUESTS FA3_REQUESTS FUSED_FA3_UNPACKED
unset FLASHINFER_FUSED_SYMMETRIC FLASHINFER_FUSED_SKIP_FA3 FLASHINFER_FUSED_PLAN_DEBUG
unset FA3_SKIP_KV_LOAD TIME_FORCE SGLANG_EDF_SCHEDULING SGLANG_DEBUG_STEP_SYNC
driver=$workdir/drivers/fused_reference_fa3_nosplit.py
echo "FA3_NOSPLIT_BUILD_START $(date -Is)"
timeout --kill-after=30 1800 "$workdir/venv/bin/python" -u -c \
  'from sglang.srt.layers.attention.fused_fa3_loader import load_fused_fa3; print(load_fused_fa3(verbose=True, cta_q=128))' \
  > "$root/build.log" 2>&1
cuobjdump --dump-resource-usage "$TORCH_EXTENSIONS_DIR/fused_fa3_q128/fused_fa3_q128.so" > "$root/resources.txt"
for enabled in 0 1; do
  for query in 32 2048; do
    label=gate_q${query}_enabled${enabled}
    require_overlap=0
    if [ "$query" = 2048 ]; then require_overlap=1; fi
    echo "FA3_NOSPLIT_GATE_START $label $(date -Is)"
    FUSED_FA3_NOSPLIT_SPECIALIZE=$enabled NATIVE_PDL_TRACE=1 NATIVE_PDL_REQUIRE_OVERLAP=$require_overlap \
      NATIVE_PDL_GRAPH_PLAN=1 MODE=mixed_verify PREFILL_QO=$query PREFILL_KV=65536 \
      VERIFY_QO=4 VERIFY_ROWS=8 VERIFY_KV=8192 TIME_AB=0 \
      timeout --kill-after=30 900 "$workdir/venv/bin/python" -u "$driver" > "$root/$label.log" 2>&1
    "$workdir/venv/bin/python" - "$root/$label.log" "$enabled" "$query" <<'PY'
import json
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
log = path.read_text()
assert log.rstrip().endswith("PASS") and "Traceback" not in log
assert log.count("NATIVE_PDL_TRACE mode=independent") == 2
if sys.argv[3] == "2048":
    assert log.count("completed_fa2_during_fa3=True") == 2
plans = re.findall(r"NATIVE_PRIMARY_PLAN max_splits=(\d+) merge_rows=(\d+)", log)
assert len(plans) == 1
splits, merges = map(int, plans[0])
assert (merges > 0) == (sys.argv[3] == "32"), plans
active = "FA3_PRIMARY_NOSPLIT active" in log
assert active == (sys.argv[2] == "1" and merges == 0)
print("FA3_NOSPLIT_GATE_PASS", json.dumps({"case": path.stem, "active": active, "splits": splits, "merge_rows": merges}), flush=True)
PY
  done
done
unset NATIVE_PDL_TRACE NATIVE_PDL_REQUIRE_OVERLAP NATIVE_PDL_GRAPH_PLAN
shapes=(
  'prefill2048_v8|MODE=mixed_verify PREFILL_QO=2048 PREFILL_KV=65536 VERIFY_QO=4 VERIFY_ROWS=8 VERIFY_KV=16384'
  'append512_v46|MODE=mixed_verify PREFILL_QO=512 PREFILL_KV=49152 VERIFY_QO=4 VERIFY_ROWS=46 VERIFY_KV=49152'
  'fa3_8x1024|MODE=band1 BAND_SIDE=fa3 BAND_QO=1024 BAND_KV=32768 BAND_ROWS=8 FA2_REQUESTS=,'
  'split_q32_v8|MODE=mixed_verify PREFILL_QO=32 PREFILL_KV=65536 VERIFY_QO=4 VERIFY_ROWS=8 VERIFY_KV=8192'
  'prefill8192_v24|MODE=mixed_verify PREFILL_QO=8192 PREFILL_KV=65536 VERIFY_QO=4 VERIFY_ROWS=24 VERIFY_KV=32768'
)
for shape in "${shapes[@]}"; do
  read -r -a fixture <<< "${shape#*|}"
  for repeat in 1 2; do
    order='0 1'
    if [ "$repeat" = 2 ]; then order='1 0'; fi
    for enabled in $order; do
      label=${shape%%|*}_enabled${enabled}_${repeat}
      echo "FA3_NOSPLIT_CASE_START $label $(date -Is)"
      FUSED_FA3_NOSPLIT_SPECIALIZE=$enabled TIME_AB=1 env "${fixture[@]}" \
        timeout --kill-after=30 1200 "$workdir/venv/bin/python" -u "$driver" > "$root/$label.log" 2>&1
      "$workdir/venv/bin/python" - "$root/$label.log" "$enabled" "$repeat" <<'PY'
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
active = "FA3_PRIMARY_NOSPLIT active" in log
assert active == (sys.argv[2] == "1" and merges == 0)
times = {}
for kernel in ("fused", "unfused"):
    matches = re.findall(r"^\[time\] " + kernel + r"\s+=\s+([0-9.]+) ms/iter", log, re.MULTILINE)
    assert len(matches) == 1 and float(matches[0]) > 0
    times[kernel + "_ms"] = float(matches[0])
audit = {"case": path.stem, "enabled": bool(int(sys.argv[2])), "active": active,
         "repeat": int(sys.argv[3]), "merge_rows": merges, "splits": splits,
         **times, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print("FA3_NOSPLIT_CASE_PASS", json.dumps(audit), flush=True)
PY
    done
  done
done
echo "FA3_NOSPLIT_KERNEL_DONE $(date -Is)"
