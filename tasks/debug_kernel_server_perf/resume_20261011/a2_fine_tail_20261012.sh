#!/bin/bash
set -euo pipefail
W=/dev/shm/w
OUT=${OUT_DIR:-$W/a2_fine_tail_20261012}
mkdir -p "$OUT"
exec 9>"$W/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[f]used_reference.py' >/dev/null; then
  echo "GPU test predecessor is still running"
  exit 1
fi
export PYTHONPATH=$W/validated_upstream/python
export SGLANG_FUSED_FI_ROOT=$W/fi
export SGLANG_FUSED_FA2_TILES=48,32,16
export FUSED_CONST_PERS_PARAMS=1 FUSED_WORK_BUDGET_SPLIT=1
export FLASHINFER_FUSED_BACKFILL=1 FUSED_FA2_KV_CHUNK=1024
export FLASHINFER_FUSED_PLAN_DEBUG=1
export HEADS=28,4 SM_SPLIT=112 FUSED_REFERENCE_WORKSPACE_BYTES=134217728
export TIME_AB=1 TIME_CUDA_GRAPH=1 TIME_ITERS=50 TIME_WARMUP=10
sha256sum "$W/fi/include/flashinfer/attention/fused_plan.cuh" \
  "$W/fi/tests/fused_fa3_persistent_ext.cu" "$W/drivers/fused_reference.py"
for grouping in 0 1; do
  for fine_tail in 0 1; do
    label="mixed_q8_group${grouping}_tail${fine_tail}"
    echo "START $label"
    MODE=mixed_verify VERIFY_QO=8 FUSED_GROUP_MAJOR=$grouping FUSED_FA2_FINE_TAIL=$fine_tail \
      "$W/venv/bin/python" -u "$W/drivers/fused_reference.py" > "$OUT/$label.log" 2>&1
    tail -15 "$OUT/$label.log"
  done
done
for mode in band1 fa3_split; do
  echo "START $mode"
  MODE=$mode BAND_SIDE=fa2 BAND_QO=10 BAND_KV=513 BAND_ROWS=3 SEED=2 FUSED_FA2_FINE_TAIL=1 \
    "$W/venv/bin/python" -u "$W/drivers/fused_reference.py" > "$OUT/$mode.log" 2>&1
  tail -15 "$OUT/$mode.log"
done
echo A2_FINE_TAIL_GATES_DONE
