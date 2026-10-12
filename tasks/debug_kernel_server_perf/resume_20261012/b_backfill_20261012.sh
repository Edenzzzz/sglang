#!/bin/bash
# B, cross-role backfill. (1) Co-launch: FA2 executor grid capped to the SMs FA3 does not hold
# (FUSED_FA2_EXEC_CTAS = 2 x (132-112) = 40: no hardware backfill onto vacated FA3 SMs) vs the
# default 2 x 132 (backfill). (2) The in-kernel backfill (FA3 CTAs drain the FA2 queue after
# tma_store_wait<0> + cp.async wait + __syncthreads) without co-launch, on vs off, classes 64/32/16.
# Every arm prints its fp32 verdict before its timing.
set -uo pipefail
W=/dev/shm/w
OUT=$W/b_backfill_20261012; mkdir -p $OUT
until grep -q GRAPHS_BISECT2_DONE $W/graphs_bisect2_20261012.log 2>/dev/null; do sleep 30; done
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/port07
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FUSED_FA2_KV_CHUNK=auto
export HEADS=28,4 SM_SPLIT=112 FUSED_REFERENCE_WORKSPACE_BYTES=1073741824
export TIME_AB=1 TIME_CUDA_GRAPH=1 TIME_ITERS=50 TIME_WARMUP=10 FUSED_FA3_CTA_Q=128
unset FLASHINFER_FUSED_PLAN_DEBUG
shapes=("mixed_verify_q8|MODE=mixed_verify VERIFY_QO=8"
        "mixed_verify_q4|MODE=mixed_verify VERIFY_QO=4"
        "mixed_verify_q4_r32_kv8192|MODE=mixed_verify VERIFY_QO=4 VERIFY_ROWS=32 VERIFY_KV=8192"
        "mixed_verify_q4_pre2048_kv16384|MODE=mixed_verify VERIFY_QO=4 VERIFY_ROWS=32 PREFILL_QO=2048 PREFILL_KV=16384 VERIFY_KV=16384"
        "mix_decode32|MODE=mix_decode DEC_ROWS=32")
arms=("cl_backfill|FLASHINFER_FUSED_COLAUNCH=1"
      "cl_nobackfill|FLASHINFER_FUSED_COLAUNCH=1 FUSED_FA2_EXEC_CTAS=40"
      "inkernel_backfill|FLASHINFER_FUSED_COLAUNCH=0 FLASHINFER_FUSED_BACKFILL=1"
      "inkernel_nobackfill|FLASHINFER_FUSED_COLAUNCH=0 FLASHINFER_FUSED_BACKFILL=0")
for s in "${shapes[@]}"; do
  for a in "${arms[@]}"; do
    label=${a%%|*}_${s%%|*}
    env ${s#*|} ${a#*|} timeout 900 $W/venv/bin/python -u $W/drivers/fused_reference.py > $OUT/$label.log 2>&1
    echo "BF $label rc=$? [$(grep -oE 'bad rows [0-9]+/[0-9]+|NON-FINITE|REPLAY CORRECTNESS: bad_heads=[0-9]+/[0-9]+' $OUT/$label.log | tr '\n' ' ')] $(grep -oE '^\[time\] (fused|unfused) += [0-9.]+|BatchAttention \(flash[^=]*= [0-9.]+' $OUT/$label.log | sed 's/(flashinfer.*from)//' | tr '\n' ' ')"
  done
done
echo B_BACKFILL_DONE
