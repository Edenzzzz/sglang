#!/bin/bash
# A3 winner map, re-taken on the FIXED kernel: FA2 co-launched at native geometry (PDL),
# classes 64/32/16, auto FA2 chunk. The 2026-10-12 morning map measured the in-kernel FA2 spill,
# not the boundary. Grid extended to long KV (65536) and to batch rows 64. Every arm prints the
# fp32 causal-reference verdict before its timing; BatchAttention and the unfused 2-launch are
# recorded as the native-FA2 references.
set -uo pipefail
W=/dev/shm/w
OUT=$W/a3_winner_map_v2_20261012
mkdir -p "$OUT"
until grep -q GRAPHS_DONE $W/graphs_on_v2_20261012.log 2>/dev/null; do
  pgrep -f '[g]raphs_on_v2' >/dev/null || break; sleep 60; done
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/port07
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto
export HEADS=28,4 SM_SPLIT=112 FUSED_REFERENCE_WORKSPACE_BYTES=1073741824
export TIME_AB=1 TIME_CUDA_GRAPH=1 TIME_ITERS=50 TIME_WARMUP=10 MODE=band1
unset FLASHINFER_FUSED_PLAN_DEBUG
for kv in 1024 4096 16384 65536; do
 for rows in 1 16 64; do
  for qo in 1 2 4 8 16 32; do
   for side in fa2 fa3_q64 fa3_q128; do
    case $side in fa2) bside=fa2; cq=128;; fa3_q64) bside=fa3; cq=64;; fa3_q128) bside=fa3; cq=128;; esac
    label="kv${kv}_rows${rows}_qo${qo}_${side}"
    BAND_SIDE=$bside FUSED_FA3_CTA_Q=$cq BAND_QO=$qo BAND_KV=$kv BAND_ROWS=$rows \
      timeout 900 $W/venv/bin/python -u $W/drivers/fused_reference.py > "$OUT/$label.log" 2>&1
    echo "A3 $label rc=$? [$(grep -oE 'bad rows [0-9]+/[0-9]+|NON-FINITE|REPLAY CORRECTNESS: bad_heads=[0-9]+/[0-9]+' $OUT/$label.log | tr '\n' ' ')] $(grep -oE '^\[time\] (fused|unfused) += [0-9.]+|BatchAttention \(flash[^=]*= [0-9.]+' $OUT/$label.log | sed 's/(flashinfer.*from)//' | tr '\n' ' ')"
   done
  done
 done
done
echo A3_V2_DONE
