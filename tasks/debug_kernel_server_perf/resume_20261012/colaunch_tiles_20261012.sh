#!/bin/bash
# A2 under the co-launch: which FA2 class set? At q8 (packed 56 -> tile48 + tail16) the co-launch
# is still 1.5-5x behind native FA2, while at q4 (tile32) it is at parity. Tile 48 at 4 warps is
# 1 q-warp x 4 kv-warps with num_mma_q=3, i.e. a 3x per-warp O accumulator. Each class set is a
# separate build (own extension dir) and gets its register report; every arm prints the fp32
# causal-reference verdict before its timing.
set -uo pipefail
W=/dev/shm/w
OUT=$W/colaunch_tiles_20261012
mkdir -p "$OUT"
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/fi_colaunch
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch
export FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1 FUSED_WORK_BUDGET_SPLIT=1
export FLASHINFER_FUSED_BACKFILL=1 FUSED_FA2_KV_CHUNK=1024 FLASHINFER_FUSED_COLAUNCH=1
export HEADS=28,4 SM_SPLIT=112 FUSED_REFERENCE_WORKSPACE_BYTES=1073741824
export TIME_AB=1 TIME_CUDA_GRAPH=1 TIME_ITERS=50 TIME_WARMUP=10 FUSED_FA3_CTA_Q=128
unset FLASHINFER_FUSED_PLAN_DEBUG
shapes=(
  "b_kv4096_r1_q8|MODE=band1 BAND_SIDE=fa2 BAND_KV=4096 BAND_ROWS=1 BAND_QO=8"
  "b_kv4096_r16_q8|MODE=band1 BAND_SIDE=fa2 BAND_KV=4096 BAND_ROWS=16 BAND_QO=8"
  "b_kv16384_r16_q8|MODE=band1 BAND_SIDE=fa2 BAND_KV=16384 BAND_ROWS=16 BAND_QO=8"
  "b_kv4096_r16_q16|MODE=band1 BAND_SIDE=fa2 BAND_KV=4096 BAND_ROWS=16 BAND_QO=16"
  "b_kv4096_r16_q4|MODE=band1 BAND_SIDE=fa2 BAND_KV=4096 BAND_ROWS=16 BAND_QO=4"
  "mixed_verify_q8|MODE=mixed_verify VERIFY_QO=8"
  "mixed_verify_q4|MODE=mixed_verify VERIFY_QO=4"
  "mix_decode32|MODE=mix_decode DEC_ROWS=32"
)
for tiles in 48,32,16 64,32,16 128,64,16; do
  tag=t${tiles//,/_}
  export SGLANG_FUSED_FA2_TILES=$tiles TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_$tag
  for entry in "${shapes[@]}"; do
    label=${tag}_${entry%%|*}; envs=${entry#*|}
    env $envs timeout 900 $W/venv/bin/python -u $W/drivers/fused_reference.py > "$OUT/$label.log" 2>&1
    rc=$?
    echo "TL $label rc=$rc [$(grep -oE 'bad rows [0-9]+/[0-9]+|NON-FINITE|REPLAY CORRECTNESS: bad_heads=[0-9]+/[0-9]+' $OUT/$label.log | tr '\n' ' ')] $(grep -oE '^\[time\] (fused|unfused) += [0-9.]+|BatchAttention \(flash[^=]*= [0-9.]+' $OUT/$label.log | sed 's/(flashinfer.*from)//' | tr '\n' ' ')"
  done
  for so in $(ls $TORCH_EXTENSIONS_DIR/*/*/*.so $TORCH_EXTENSIONS_DIR/*/*.so 2>/dev/null); do
    echo "RES $tag $(basename $so) $(/usr/local/cuda/bin/cuobjdump -res-usage $so 2>/dev/null | grep -A1 -E 'Function .*(FusedFA3PersistentKernel|FA2ExecutorKernel)' | grep -oE 'FusedFA3PersistentKernel|FA2ExecutorKernel|REG:[0-9]+|STACK:[0-9]+' | paste -sd' ')"
  done
done
echo COLAUNCH_TILES_DONE
