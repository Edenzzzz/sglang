#!/bin/bash
# A3: fresh FA2-vs-FA3 winner map on the post-fix kernels, as input to a fitted dispatch rule.
# Grid: query tokens x KV length x batch rows, GQA 28/4 (packed q = 7 x tokens). Sides: FA2 with
# the A2 per-work-item tail dispatcher (classes 48/32/16), FA3 q64, FA3 q128 (A1 namespaces), and
# BatchAttention as the validated reference. Each line carries the fp32 bad-row verdict so a
# failing arm cannot contribute a timing. Serialised behind the serving A/B on the single GPU.
set -uo pipefail
W=/dev/shm/w
OUT=$W/a3_winner_map_20261012
mkdir -p "$OUT"
until grep -q POSTFIX_AB_DONE $W/postfix_realistic_ab_20261012.log 2>/dev/null; do sleep 30; done
exec 9>"$W/a2_kernel.lock"; flock -n 9 || { echo LOCKED; exit 1; }
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/fi
export SGLANG_FUSED_FA2_TILES=48,32,16 FUSED_FA2_FINE_TAIL=1
export FUSED_CONST_PERS_PARAMS=1 FUSED_WORK_BUDGET_SPLIT=1
export FLASHINFER_FUSED_BACKFILL=1 FUSED_FA2_KV_CHUNK=1024
export HEADS=28,4 SM_SPLIT=112 FUSED_REFERENCE_WORKSPACE_BYTES=1073741824
export TIME_AB=1 TIME_CUDA_GRAPH=1 TIME_ITERS=50 TIME_WARMUP=10 MODE=band1
unset FLASHINFER_FUSED_PLAN_DEBUG
sha256sum $W/fi/include/flashinfer/attention/fused_plan.cuh $W/fi/include/flashinfer/attention/persistent.cuh $W/drivers/fused_reference.py
for kv in 1024 4096 16384; do
 for rows in 1 16; do
  for qo in 1 2 4 8 16 32; do
   for side in fa2 fa3_q64 fa3_q128; do
    case $side in
     fa2) bside=fa2; cq=128;; fa3_q64) bside=fa3; cq=64;; fa3_q128) bside=fa3; cq=128;;
    esac
    label="kv${kv}_rows${rows}_qo${qo}_${side}"
    BAND_SIDE=$bside FUSED_FA3_CTA_Q=$cq BAND_QO=$qo BAND_KV=$kv BAND_ROWS=$rows \
      timeout 900 $W/venv/bin/python -u $W/drivers/fused_reference.py > "$OUT/$label.log" 2>&1
    rc=$?
    v=$(grep -oE "bad rows [0-9]+/[0-9]+|NON-FINITE" "$OUT/$label.log" | tail -1)
    t=$(grep -oE "fused +(\(graph\) )?= *[0-9.]+|BatchAttention[^=]*= *[0-9.]+" "$OUT/$label.log" | tr '\n' ' ')
    echo "A3 $label rc=$rc verdict=[${v:-NOVERDICT}] $t"
   done
  done
 done
done
echo A3_WINNER_MAP_DONE
