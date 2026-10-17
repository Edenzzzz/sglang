#!/bin/bash
# A5.5 headroom check: NCU of sgl-kernel FA3 vs our parity build of flash-attention's kernel (paged TMA, -DNDEBUG, fast
# math, upstream scheduler = fa3only) on prefill 2048 @ 64K and agent 4 x 512 @ 64K. Both kernels are cutlass
# device_kernel; in the harness's eager order sgl-kernel's launches first and ours second (launch-skip 0 / 1).
set -uo pipefail
W=/dev/shm/w
out=$W/a5_fa3_ncu_parity_20261017; mkdir -p $out
exec 9>"$W/a2_kernel.lock"; flock 9
cd $W
export PYTHONPATH=$W/up_a5/python HF_HOME=$W/hf SGLANG_FUSED_FI_ROOT=$W/fi_a5 TORCH_CUDA_ARCH_LIST=9.0a
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1 FUSED_WORK_BUDGET_SPLIT=1
export FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1 FUSED_FA2_KV_CHUNK=auto4096 FUSED_FA3_DEDICATED_PRIMARY=1
export FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7 FA3_STAGES=2 FUSED_FA3_TMA_TRACE=0 FUSED_PAGED_TMA_BUILD=0 FUSED_CLUSTER=1
export FLASHINFER_FUSED_CLUSTER=1 FUSED_FA3_CLUSTER_PAIR=0 FUSED_FA3_TMA_MULTICAST=0
export SGLANG_FUSED_MERGED_DIR=$W/merged_a5 TORCH_EXTENSIONS_DIR=$W/torch_ext_a5_upstream_ptma_ndebug
export FA3_UPSTREAM=1 FUSED_FAST_MATH=1 FA3_PAGED_TMA=1 FUSED_EXTRA_NVCC=-DNDEBUG FA3_SM_BUDGET=132
export TIME_AB=0 SEED=31 FA2_REQUESTS=, SM_SPLIT=112
for shape in "prefill2048_kv64k|2048:65536" "agent4x512_kv64k|512:65536,512:65536,512:65536,512:65536"; do
  for v in "sgl|0" "ours|1"; do
    IFS='|' read -r name skip <<< "$v"; label=${shape%%|*}_$name
    env FA3_REQUESTS="${shape#*|}" timeout --kill-after=30 900 ncu --target-processes application-only \
      --kernel-name-base function --kernel-name device_kernel --launch-skip $skip --launch-count 1 --replay-mode kernel \
      --clock-control none --section SpeedOfLight --section ComputeWorkloadAnalysis --section WarpStateStats \
      --section MemoryWorkloadAnalysis --section InstructionStats --section LaunchStats --export $out/$label -f \
      $W/venv/bin/python -u $W/drivers/a5_fa3_reference_20261017.py > $out/$label.log 2>&1
    echo "A5NP_END $label rc=$? pass=$(grep -c ^PASS $out/$label.log)"
  done
done
echo A5NP_DONE
