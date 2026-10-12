#!/bin/bash
set -uo pipefail
W=/dev/shm/w
OUT=${OUT_DIR:-$W/a2_tail_null_20261012}
mkdir -p "$OUT"
exec 9>"$W/a2_kernel.lock"
flock -n 9 || exit 1
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[f]used_reference.py' >/dev/null; then
  echo "GPU test predecessor is still running"
  exit 1
fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/fi
export SGLANG_FUSED_FA2_TILES=48,32,16 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1
export FUSED_FA2_KV_CHUNK=1024 FLASHINFER_FUSED_PLAN_DEBUG=1
export HEADS=28,4 SM_SPLIT=112 FUSED_REFERENCE_WORKSPACE_BYTES=134217728
export MODE=band1 BAND_SIDE=fa2 BAND_QO=10 BAND_KV=513 BAND_ROWS=3
status=0
for seed in 2 3 4; do
  for fine_tail in 0 1; do
    label="band_seed${seed}_tail${fine_tail}"
    SEED=$seed FUSED_FA2_FINE_TAIL=$fine_tail "$W/venv/bin/python" -u \
      "$W/drivers/fused_reference.py" > "$OUT/$label.log" 2>&1
    result=$?
    echo "$label exit=$result"
    tail -5 "$OUT/$label.log"
    if [ "$result" -ne 0 ]; then status=1; fi
  done
done
MODE=fa3_split FUSED_FA2_FINE_TAIL=1 "$W/venv/bin/python" -u \
  "$W/drivers/fused_reference.py" > "$OUT/fa3_split.log" 2>&1
result=$?
echo "fa3_split exit=$result"
tail -5 "$OUT/fa3_split.log"
if [ "$result" -ne 0 ]; then status=1; fi
echo "A2_TAIL_NULL_DONE status=$status"
exit "$status"
