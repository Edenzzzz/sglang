#!/bin/bash
# Forced-parking serving gate (2026-10-16). The standard gate sends the short prompts first, so
# no short request ever waits behind a chunked long prefill and EDF never parks (park=[] in both
# edf_preempt_20261016 gates). This variant (drivers/serving/srv_correctness_edf_park_20261016.py,
# CORRECTNESS_LONG_FIRST=1, chunk 256) sends the 4 long prompts first and the 8 short ones while
# they are being chunked. Arms, all compared with the stock MTP_LPT_Q4G reference:
#   MTP_LPT_LF_Q4G            stock MTP, long-first order (order-only null case)
#   MTPMIX_SPLIT_EDFCTRL_LF   EDF, no preemption, long-first
#   MTPMIX_SPLIT_EDFPARK_LF   EDF + preemption, long-first, must park at least once
# v2 (04:5x): v1's long-first order never parked -- EDF already put the shorts ahead of the WAITING
# longs, and the in-progress long had < 2x short work left; v2 sends one long, then the shorts
# 0.1 s later (most of its prefill left), then the other longs. Claude, 2026-10-16.
set -uo pipefail
W=/dev/shm/w
root=$W/edf_park_gate_v2_20261016
until grep -q 'PG_EXIT' $W/edf_park_gate_20261016.v1.log 2>/dev/null; do sleep 60; done
exec 9>"$W/a2_kernel.lock"; flock 9
trap 'printf "PG_EXIT status=%s time=%s\n" "$?" "$(date -Is)"' EXIT
for _ in $(seq 60); do pgrep -f '[s]glang.launch_server|[k]ernel_concurrency_driver|[f]used_reference|[r]un_mixed' >/dev/null || break; sleep 5; done
if pgrep -f '[s]glang.launch_server|[k]ernel_concurrency_driver|[f]used_reference|[r]un_mixed' >/dev/null; then echo "GPU predecessor"; exit 1; fi
mkdir -p $root; cp $W/q4_gate_graphs_20261012/MTP_LPT_Q4G.json $root/
export PYTHONPATH=$W/up_edf_preempt/python HF_HOME=$W/hf
export SGLANG_FUSED_FI_ROOT=$W/port07 SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112 SGLANG_SPLIT_OVERLAP=0 SGLANG_FUSED_DECODE_GUARD=0
export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
unset SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH SGLANG_FUSED_KERNEL_EXEC SGLANG_FUSED_DISPATCH
DRV=$W/drivers/srv_correctness_edf_park_20261016.py
CHAT=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
MODEL_JSON=$W/edf_dynchunk_20261015/models/MTPMIX_SPLIT_step_model.json
EDF="SGLANG_EDF_SCHEDULING=1 SGLANG_EDF_LATENCY_MODEL=linear:$MODEL_JSON SGLANG_EDF_PLAN_OVERHEAD_MS=0 SGLANG_EDF_ACCEPT_MODE=req SGLANG_ITL_SLO_MS=50 SGLANG_EDF_PREFILL_FLOOR=0"
sha256sum $DRV $PYTHONPATH/sglang/srt/managers/scheduler.py > $root/sources.sha256
for v in "MTP_LPT_LF_Q4G|none|0|0" "MTPMIX_SPLIT_EDFCTRL_LF_Q4G|edf|0|0" "MTPMIX_SPLIT_EDFPARK_LF_Q4G|edf|1|1"; do
  IFS='|' read -r arm sched pre req <<< "$v"
  envs="CORRECTNESS_LONG_FIRST=1 CORRECTNESS_REQUIRE_PARK=$req SGLANG_EDF_PREEMPT_CHUNK=$pre SGLANG_EDF_RESERVE_EARLIER=$pre"
  [ "$sched" = edf ] && envs="$envs $EDF"
  echo "PG_START $arm $(date -Is)"
  ( cd $W/drivers && env $envs CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET=$CHAT BENCH_CHUNK=256 CORRECTNESS_DECODE_TOKENS=1200 \
      timeout --kill-after=30 2400 $W/venv/bin/python -u $DRV $arm $root > $root/$arm.run.log 2>&1 )
  rc=$?
  $W/venv/bin/python -u $DRV --compare $root MTP_LPT_Q4G $arm > $root/$arm.compare.log 2>&1
  echo "PG_RESULT $arm run_rc=$rc $(tail -1 $root/$arm.compare.log) park=[$(grep -o 'EDF_PARK calls=[0-9]*.*' $root/server_$arm.log | tail -1)] tracebacks=$(grep -c Traceback $root/server_$arm.log)"
done
echo "PG_DONE $(date -Is)"
