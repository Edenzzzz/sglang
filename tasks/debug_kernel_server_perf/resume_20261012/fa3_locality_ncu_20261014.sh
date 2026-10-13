#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
output_dir=$workdir/fa3_locality_ncu_20261014
predecessor_pid=599626
predecessor_log=$workdir/tma_warp_kernel_20261014.log
trap 'status=$?; printf "FA3_LOCALITY_NCU_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
echo "WAIT_TMA_WARP $(date -Is)"
while kill -0 "$predecessor_pid" 2>/dev/null; do
  predecessor_state=$(ps -p "$predecessor_pid" -o stat= || true)
  if [[ "$predecessor_state" == Z* ]]; then break; fi
  if ! predecessor_command=$(tr '\0' ' ' < "/proc/$predecessor_pid/cmdline"); then break; fi
  [[ "$predecessor_command" == "bash tma_warp_kernel_20261014.sh "* || "$predecessor_command" == "bash /dev/shm/w/tma_warp_kernel_20261014.sh "* ]]
  sleep 10
done
grep -q '^TMA_WARP_KERNEL_DONE ' "$predecessor_log"
grep -q '^TMA_WARP_KERNEL_EXIT status=0 ' "$predecessor_log"
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[r]un_mixed_workloads|[f]used_reference|[t]ma_(multicast|warp)_reference' >/dev/null; then
  echo "GPU predecessor still running; refusing duplicate launch"
  exit 1
fi
cd "$workdir"
grep -q '^FA3_LOCALITY_DONE ' fa3_locality_20261014.log
grep -q '^FA3_LOCALITY_EXIT status=0 ' fa3_locality_20261014.log
test ! -e "$output_dir"
mkdir "$output_dir"
sha256sum -c tma_warp_overlay_20261014/candidate.sha256
export HF_HOME=$workdir/hf TORCH_CUDA_ARCH_LIST=9.0a
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto4096 FUSED_FA3_DEDICATED_PRIMARY=1
export FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7 FA3_STAGES=2 SM_SPLIT=112
export FUSED_FA3_TMA_TRACE=0 FUSED_FA3_TMA_MULTICAST=0 TIME_AB=0 SEED=31
export FA3_REQUESTS="1024:32768,1024:32768,1024:32768,1024:32768,1024:32768,1024:32768,1024:32768,1024:32768" FA2_REQUESTS=,
unset FUSED_NOSPLIT FUSED_KV_SHARE_BUILD FLASHINFER_FUSED_SYMMETRIC FLASHINFER_FUSED_SKIP_FA3 FA3_SKIP_KV_LOAD
unset NATIVE_PDL_REQUIRE_OVERLAP NATIVE_PDL_TRACE FA3_PAGED_TMA
for variant in GATHER; do
  export PYTHONPATH=$workdir/up_tma_warp/python SGLANG_FUSED_FI_ROOT=$workdir/fi_tma_warp
  export FUSED_PAGED_TMA_BUILD=1 FUSED_CLUSTER=2 FLASHINFER_FUSED_CLUSTER=2 FUSED_FA3_CLUSTER_PAIR=1
  export FUSED_TMA_WARP32=0 SGLANG_FUSED_FA3_CONSUMER_REGS=216
  export SGLANG_FUSED_MERGED_DIR=$workdir/merged_tma_warp_control128
  export TORCH_EXTENSIONS_DIR=$workdir/torch_ext_tma_warp_control128
  kernel='regex:.*DedicatedFA3PrimaryKernel.*'
  if [ "$variant" = GATHER ]; then
    export PYTHONPATH=$workdir/up_tma_multicast/python SGLANG_FUSED_FI_ROOT=$workdir/fi_tma_multicast
    export FUSED_PAGED_TMA_BUILD=0 FUSED_CLUSTER=1 FLASHINFER_FUSED_CLUSTER=1 FUSED_FA3_CLUSTER_PAIR=1
    export SGLANG_FUSED_MERGED_DIR=$workdir/merged_tma_gather TORCH_EXTENSIONS_DIR=$workdir/torch_ext_tma_gather
  elif [ "$variant" = TMA32 ]; then
    export FUSED_TMA_WARP32=1 SGLANG_FUSED_FA3_CONSUMER_REGS=232
    export SGLANG_FUSED_MERGED_DIR=$workdir/merged_tma_warp_warp32_c232
    export TORCH_EXTENSIONS_DIR=$workdir/torch_ext_tma_warp_warp32_c232
  elif [ "$variant" = STOCK ]; then
    kernel='regex:.*PrefillWithKVCacheKernel.*'
  fi
  echo "START $variant $(date -Is)"
  env | LC_ALL=C sort | grep -E '^(PYTHONPATH=|SGLANG_FUSED_|FUSED_|FLASHINFER_|FA[23]_REQUESTS=|SM_SPLIT=|TIME_|SEED=)' \
    > "$output_dir/$variant.environment.txt"
  timeout --kill-after=30 900 ncu --target-processes application-only --kernel-name-base function \
    --kernel-name "$kernel" --launch-count 1 --replay-mode kernel --clock-control none \
    --section LaunchStats --section Occupancy --section SpeedOfLight --section SchedulerStats \
    --section WarpStateStats --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
    --section InstructionStats --export "$output_dir/$variant" \
    "$workdir/venv/bin/python" -u "$workdir/drivers/tma_warp_reference_20261014.py" \
    > "$output_dir/$variant.log" 2>&1
  grep -q '^PASS$' "$output_dir/$variant.log"
  test -s "$output_dir/$variant.ncu-rep"
  ncu --import "$output_dir/$variant.ncu-rep" --page raw --csv > "$output_dir/$variant.csv"
  "$workdir/venv/bin/python" - "$output_dir/$variant.log" <<'PY'
import json
from pathlib import Path
import sys

log = Path(sys.argv[1]).read_text()
fixtures = [json.loads(line.removeprefix("FIXTURE ")) for line in log.splitlines() if line.startswith("FIXTURE ")]
assert len(fixtures) == 1
assert fixtures[0]["native"] == [] and fixtures[0]["primary"] == [[1024, 32768]] * 8
assert fixtures[0]["page_size"] == 128
print("PROFILE_WORKLOAD_VERIFIED", fixtures[0], flush=True)
PY
  echo "END $variant $(date -Is)"
done
echo "FA3_LOCALITY_NCU_DONE $(date -Is)"
