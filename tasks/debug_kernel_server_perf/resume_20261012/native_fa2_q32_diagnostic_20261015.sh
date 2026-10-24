#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
prototype=$workdir/native_fa2_q32_work_20261015
root=$workdir/native_fa2_q32_diagnostic_20261015
trap 'status=$?; printf "NATIVE_Q32_DIAGNOSTIC_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
grep -q '^NATIVE_Q32_EXIT status=1 ' "$workdir/native_fa2_q32_20261015.log"
if pgrep -f '[s]glang.launch_server|[b]ench_native_q32.py|[t]est_native_q32.py|[d]iagnose_native_q32.py' >/dev/null; then exit 1; fi
test ! -e "$root"
mkdir "$root"
cd "$prototype"
sha256sum -c sources.sha256 > "$root/sources_checked.log"
sha256sum diagnose_native_q32.py > "$root/diagnostic_source.sha256"
export HF_HOME=$workdir/hf TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=1
export NATIVE_Q32_MERGED=$workdir/merged_native_fa2_q32 TORCH_EXTENSIONS_DIR=$workdir/torch_ext_native_fa2_q32
unset SGLANG_FUSED_FI_DATA
for tile in 0 32; do
  echo "NATIVE_Q32_DIAGNOSTIC_START tile=$tile $(date -Is)"
  timeout --kill-after=10 180 "$workdir/venv/bin/python" -u diagnose_native_q32.py \
    "$root/tile$tile.json" --tile "$tile" --iterations 20 > "$root/tile$tile.log" 2>&1
  status=0
  timeout --kill-after=10 240 compute-sanitizer --tool racecheck --error-exitcode 86 \
    --kernel-name kns=BatchPrefillWithPagedKVCacheKernel --launch-count 1 --racecheck-report analysis \
    "$workdir/venv/bin/python" -u diagnose_native_q32.py "$root/race_tile$tile.json" \
    --tile "$tile" --iterations 1 > "$root/race_tile$tile.log" 2>&1 || status=$?
  echo "$status" > "$root/race_tile$tile.status"
  echo "NATIVE_Q32_RACECHECK tile=$tile status=$status $(date -Is)"
done
echo "NATIVE_Q32_DIAGNOSTIC_COMPLETE $(date -Is)"
