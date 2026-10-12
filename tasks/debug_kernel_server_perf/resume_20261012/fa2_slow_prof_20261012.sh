#!/bin/bash
# Why is the fused FA2 side 20-30x slower than BatchAttention on small verify shapes
# (A3 map: kv4096, one 8-token request -> fused 0.348 ms vs BatchAttention 0.016 ms)?
# nsys kernel breakdown (no graphs, so each kernel is visible) plus a few env ablations, all
# on the same harness with the fp32 reference verdict printed per run.
set -uo pipefail
W=/dev/shm/w
OUT=$W/fa2_slow_prof_20261012
mkdir -p "$OUT"
until grep -q A3_WINNER_MAP_DONE $W/a3_winner_map_20261012.log 2>/dev/null; do sleep 30; done
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/fi
export SGLANG_FUSED_FA2_TILES=48,32,16 FUSED_FA2_FINE_TAIL=1
export FUSED_CONST_PERS_PARAMS=1 FUSED_WORK_BUDGET_SPLIT=1
export FLASHINFER_FUSED_BACKFILL=1 FUSED_FA2_KV_CHUNK=1024
export HEADS=28,4 SM_SPLIT=112 FUSED_REFERENCE_WORKSPACE_BYTES=1073741824
export TIME_AB=1 TIME_ITERS=50 TIME_WARMUP=10 MODE=band1 FUSED_FA3_CTA_Q=128
unset FLASHINFER_FUSED_PLAN_DEBUG
run() {  # label, then env assignments
  local label=$1; shift
  env "$@" timeout 900 $W/venv/bin/python -u $W/drivers/fused_reference.py > "$OUT/$label.log" 2>&1
  echo "PROF $label rc=$? $(grep -oE 'bad rows [0-9]+/[0-9]+' $OUT/$label.log | tail -1) $(grep -oE '(fused|unfused) += [0-9.]+|BatchAttention \(flash[^=]*= [0-9.]+' $OUT/$label.log | tr '\n' ' ')"
}
# 1. kernel breakdown under nsys, no graphs
for shape in "fa2 4096 1 8" "fa2 1024 1 1" "fa2 4096 16 4" "fa3 4096 1 8"; do
  set -- $shape
  label=nsys_${1}_kv$2_rows$3_qo$4
  BAND_SIDE=$1 BAND_KV=$2 BAND_ROWS=$3 BAND_QO=$4 TIME_CUDA_GRAPH=0 \
    timeout 900 nsys profile -t cuda -o "$OUT/$label" -f true \
    $W/venv/bin/python -u $W/drivers/fused_reference.py > "$OUT/$label.log" 2>&1
  echo "NSYS $label rc=$?"
  nsys stats -r cuda_gpu_kern_sum -f csv "$OUT/$label.nsys-rep" -o "$OUT/$label" > /dev/null 2>&1
  echo "--- $label kernels (total ns, instances, avg ns, name)"
  python3 - "$OUT/${label}_cuda_gpu_kern_sum.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows[:12]:
    print(f"  {int(float(r['Total Time (ns)'])):>12} {r['Instances']:>6} {float(r['Avg (ns)']):>12.0f}  {r['Name'][:150]}")
PY
done
# 2. ablations on the kv4096 single-request FA2 shape, graphs on
base="BAND_SIDE=fa2 BAND_KV=4096 BAND_ROWS=1 BAND_QO=8 TIME_CUDA_GRAPH=1"
run abl_base $base
run abl_nobackfill $base FLASHINFER_FUSED_BACKFILL=0
run abl_notail $base FUSED_FA2_FINE_TAIL=0
run abl_chunk4096 $base FUSED_FA2_KV_CHUNK=4096
run abl_smsplit0 $base SM_SPLIT=0
run abl_nobudget $base FUSED_WORK_BUDGET_SPLIT=0
echo FA2_SLOW_PROF_DONE
