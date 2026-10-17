#!/bin/bash
# A5.0c: NCU of our dedicated FA3 primary (fast math OFF / ON) vs upstream sgl-kernel FA3 on a chat-size fixture
# (per-tile overhead dominates: 448 tiles of ~1K keys) and a long one (mainloop dominates), same harness and builds
# as a5_fa3_fastmath_20261017.sh. One profiled launch per kernel, kernel replay, clocks uncontrolled -- counters,
# not timings. Waits for a2_kernel.lock; yields to /dev/shm/w/codex_needs_gpu.
set -uo pipefail
W=/dev/shm/w
out=$W/a5_fa3_ncu_20261017
trap 'printf "A5NCU_EXIT status=%s time=%s\n" "$?" "$(date -Is)"' EXIT
exec 9>"$W/a2_kernel.lock"; flock 9
cd "$W"; mkdir -p "$out"
export PYTHONPATH=$W/up_a5/python HF_HOME=$W/hf
export SGLANG_FUSED_FI_ROOT=$W/fi_tma_multicast TORCH_CUDA_ARCH_LIST=9.0a
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto4096 FUSED_FA3_DEDICATED_PRIMARY=1
export FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7 FA3_STAGES=2 FUSED_FA3_TMA_TRACE=0
export SGLANG_FUSED_MERGED_DIR=$W/merged_tma_gather
export FUSED_PAGED_TMA_BUILD=0 FUSED_CLUSTER=1 FLASHINFER_FUSED_CLUSTER=1 FUSED_FA3_CLUSTER_PAIR=0 FUSED_FA3_TMA_MULTICAST=0
export TIME_AB=0 SEED=31 FA2_REQUESTS=, SM_SPLIT=112
unset FUSED_NOSPLIT FUSED_KV_SHARE_BUILD FUSED_FA3_KV_SHARE FUSED_FA3_KV_SHARE_BULK
unset FLASHINFER_FUSED_SYMMETRIC FLASHINFER_FUSED_SKIP_FA3 FLASHINFER_FUSED_PLAN_DEBUG FA3_SKIP_KV_LOAD TIME_FORCE
unset FA3_PAGED_TMA NATIVE_PDL_REQUIRE_OVERLAP NATIVE_PDL_TRACE
shapes=("chat_fresh2048|2048:2048" "prefill2048_kv64k|2048:65536")
# variant|kernel|fast math|extension dir
variants=("oursOFF|DedicatedFA3PrimaryKernel|0|$W/torch_ext_tma_gather"
          "oursON|DedicatedFA3PrimaryKernel|1|$W/torch_ext_a5_fastmath"
          "upstream|device_kernel|0|$W/torch_ext_tma_gather")
for shape in "${shapes[@]}"; do
  for v in "${variants[@]}"; do
    IFS='|' read -r name kernel fm ext <<< "$v"
    [ -s $W/codex_needs_gpu ] && { echo "A5NCU_YIELD ${shape%%|*} $name"; exit 0; }
    label=${shape%%|*}_$name
    echo "A5NCU_START $label $(date -Is)"
    env FA3_REQUESTS="${shape#*|}" FUSED_FAST_MATH=$fm TORCH_EXTENSIONS_DIR=$ext timeout --kill-after=30 900 \
      ncu --target-processes application-only --kernel-name-base function --kernel-name "$kernel" --launch-count 1 \
      --replay-mode kernel --clock-control none \
      --section LaunchStats --section Occupancy --section SpeedOfLight --section SchedulerStats \
      --section WarpStateStats --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
      --section InstructionStats --export "$out/$label" -f \
      "$W/venv/bin/python" -u "$W/drivers/a5_fa3_reference_20261017.py" > "$out/$label.log" 2>&1
    rc=$?
    [ -s "$out/$label.ncu-rep" ] && ncu --import "$out/$label.ncu-rep" --page raw --csv > "$out/$label.csv" 2>/dev/null
    echo "A5NCU_END $label rc=$rc pass=$(grep -c '^PASS' $out/$label.log) csv_lines=$(wc -l < $out/$label.csv 2>/dev/null)"
  done
done
echo "A5NCU_DONE $(date -Is)"
