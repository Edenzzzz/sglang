#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
prototype=$workdir/native_fa2_q32_v2_work_20261015
root=$workdir/native_fa2_q32_v2_20261015
predecessor=981793
trap 'status=$?; printf "NATIVE_Q32_V2_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
echo "NATIVE_Q32_WAIT pid=$predecessor $(date -Is)"
while kill -0 "$predecessor" 2>/dev/null; do
  state=$(ps -p "$predecessor" -o stat= || true)
  if [[ "$state" == Z* ]]; then break; fi
  command=$(tr '\0' ' ' < "/proc/$predecessor/cmdline")
  [[ "$command" == 'bash native_fa2_q32_diagnostic_20261015.sh '* ]]
  sleep 20
done
grep -q '^NATIVE_Q32_DIAGNOSTIC_COMPLETE ' "$workdir/native_fa2_q32_diagnostic_20261015.log"
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[c]orrectness.py|[k]ernel_concurrency_driver|[s]tock_fa3_reference.py' >/dev/null; then exit 1; fi
test ! -e "$root"
mkdir "$root"
cd "$workdir"
sha256sum -c "$prototype/baseline.sha256"
cd "$prototype"
sha256sum -c sources.sha256
cp sources.sha256 baseline.sha256 "$root/"
export HF_HOME=$workdir/hf TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export NATIVE_Q32_MERGED=$workdir/merged_native_fa2_q32_v2
export TORCH_EXTENSIONS_DIR=$workdir/torch_ext_native_fa2_q32_v2
unset SGLANG_FUSED_FI_DATA SGLANG_EDF_SCHEDULING SGLANG_DEBUG_STEP_SYNC
echo "NATIVE_Q32_BUILD_START $(date -Is)"
timeout --kill-after=30 1800 "$workdir/venv/bin/python" -u load_native_q32.py > "$root/build.log" 2>&1
cuobjdump --dump-resource-usage "$TORCH_EXTENSIONS_DIR/native_fa2_q32/native_fa2_q32.so" > "$root/resources.txt"
timeout --kill-after=10 180 "$workdir/venv/bin/python" -u diagnose_native_q32.py \
  "$root/long_regression.json" --tile 32 --iterations 20 > "$root/long_regression.log" 2>&1
"$workdir/venv/bin/python" -c 'import json,sys; data=json.load(open(sys.argv[1])); assert len(data["iterations"])==20 and all(row["incorrect_heads"]==0 for row in data["iterations"])' "$root/long_regression.json"
timeout --kill-after=10 240 compute-sanitizer --tool racecheck --error-exitcode 86 \
  --kernel-name kns=BatchPrefillWithPagedKVCacheKernel --launch-count 1 --racecheck-report analysis \
  "$workdir/venv/bin/python" -u diagnose_native_q32.py "$root/race_fixed.json" \
  --tile 32 --iterations 1 > "$root/race_fixed.log" 2>&1
grep -q 'RACECHECK SUMMARY: 0 hazards' "$root/race_fixed.log"
echo "NATIVE_Q32_UNIT_START $(date -Is)"
timeout --kill-after=30 900 "$workdir/venv/bin/python" -u test_native_q32.py > "$root/tests.log" 2>&1
grep -q '^NATIVE_Q32_EXECUTED_GEOMETRY ' "$root/tests.log"
if grep -q 'skipped=' "$root/tests.log"; then exit 1; fi
echo "NATIVE_Q32_UNIT_PASS $(date -Is)"
timeout --kill-after=30 3600 "$workdir/venv/bin/python" -u bench_native_q32.py "$root/timings.json" > "$root/timings.log" 2>&1
grep -q '^NATIVE_Q32_MATRIX_PASS cases=54' "$root/timings.log"
echo "NATIVE_Q32_V2_COMPLETE $(date -Is)"
