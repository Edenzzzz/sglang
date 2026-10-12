#!/bin/bash
set -euo pipefail
W=/dev/shm/w
OUT=$W/a2_tail_q8_serving_20261012
mkdir -p "$OUT"
exec 9>"$W/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[f]used_reference.py' >/dev/null; then
  echo "GPU test predecessor is still running"
  exit 1
fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/fi HF_HOME=$W/hf
export SGLANG_FUSED_FA2_TILES=48,32,16 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FUSED_FA2_KV_CHUNK=1024
export FLASHINFER_FUSED_PLAN_DEBUG=1 SGLANG_FUSED_SM_FA3=112
export CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET=$W/datasets/ShareGPT.original.json
export DISABLE_CUDA_GRAPH=1 BENCH_CHUNK=512 CORRECTNESS_DECODE_TOKENS=1200
export SGLANG_FUSED_DECODE_GUARD=0 SGLANG_FUSED_SPLIT_VERIFY=1
export MTP_DRAFT_TOKENS=8 MTP_SPEC_STEPS=7
unset SGLANG_FUSED_KERNEL_EXEC SGLANG_MIXED_SPEC SGLANG_FUSED_VERIFY_EPILOGUE
DRIVER=$W/validated_upstream/tasks/fused_specdec_workload/drivers/srv_correctness.py
for arm in MTP_LPT_Q8 MTPMIXFUSED_Q8_TAIL0 MTPMIXFUSED_Q8_TAIL1; do
  export FUSED_FA2_FINE_TAIL=0
  if [[ "$arm" == *TAIL1 ]]; then export FUSED_FA2_FINE_TAIL=1; fi
  echo "START $arm tail=$FUSED_FA2_FINE_TAIL"
  "$W/venv/bin/python" -u "$DRIVER" "$arm" "$OUT"
done
status=0
for arm in MTPMIXFUSED_Q8_TAIL0 MTPMIXFUSED_Q8_TAIL1; do
  if ! "$W/venv/bin/python" -u "$DRIVER" --compare "$OUT" MTP_LPT_Q8 "$arm"; then
    status=1
  fi
done
if ! "$W/venv/bin/python" -u "$DRIVER" --compare "$OUT" \
    MTPMIXFUSED_Q8_TAIL0 MTPMIXFUSED_Q8_TAIL1; then
  status=1
fi
echo "A2_Q8_SERVING_DONE status=$status"
exit "$status"
