#!/bin/bash
# B3 multicast, measured as a ceiling on the CURRENT kernel (co-launch, 64/32/16, auto chunk):
# FLASHINFER_FA3_SKIP_KV_LOAD=1 zero-fills every FA3 K/V copy instead of reading global memory, so
# the difference to the normal build bounds what KV multicast could save (it removes only part of
# that traffic). Skip arms are wrong by construction (TIME_FORCE=1, timing delta only); normal arms
# must pass the fp32 reference. Separate source tree and extension cache: serving builds untouched.
set -uo pipefail
W=/dev/shm/w
until grep -q TRACELAB_FULL_DONE $W/tracelab_full_20261013.log 2>/dev/null; do
  pgrep -f '[t]racelab_full_2026' >/dev/null || break; sleep 60; done
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
OUT=$W/b3_multicast_ceiling_20261013; mkdir -p $OUT
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/fi_skip
export SGLANG_FUSED_MERGED_DIR=$W/merged_skip
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto HEADS=28,4 SM_SPLIT=112 FUSED_REFERENCE_WORKSPACE_BYTES=1073741824
export TIME_AB=1 TIME_CUDA_GRAPH=1 TIME_ITERS=50 TIME_WARMUP=10 FUSED_FA3_CTA_Q=128
unset FLASHINFER_FUSED_PLAN_DEBUG
shapes=("fa3_kv16384_r16_q32|MODE=band1 BAND_SIDE=fa3 BAND_KV=16384 BAND_ROWS=16 BAND_QO=32"
        "fa3_kv65536_r1_q32|MODE=band1 BAND_SIDE=fa3 BAND_KV=65536 BAND_ROWS=1 BAND_QO=32"
        "pre2048_kv16384_v32|MODE=mixed_verify VERIFY_QO=4 VERIFY_ROWS=32 PREFILL_QO=2048 PREFILL_KV=16384 VERIFY_KV=16384"
        "mixed_verify_q4|MODE=mixed_verify VERIFY_QO=4"
        "fa3_default|MODE=fa3")
for skip in 0 1; do
  export FA3_SKIP_KV_LOAD=$skip TORCH_EXTENSIONS_DIR=$W/torch_ext_skip$skip
  for s in "${shapes[@]}"; do
    label=skip${skip}_${s%%|*}
    force=""; [ $skip = 1 ] && force="TIME_FORCE=1"
    env ${s#*|} $force timeout 1500 $W/venv/bin/python -u $W/drivers/fused_reference.py > $OUT/$label.log 2>&1
    echo "MC $label rc=$? [$(grep -oE 'bad rows [0-9]+/[0-9]+' $OUT/$label.log | tail -1)] $(grep -oE '^\[time\] fused += [0-9.]+|BatchAttention \(flash[^=]*= [0-9.]+' $OUT/$label.log | sed 's/(flashinfer.*from)//' | tr '\n' ' ')"
  done
done
echo B3_CEILING_DONE
