#!/bin/bash
# B1 ceiling: profile 60 serving steps of each 4-mode arm at mix32 chat 16/s + agent 0.1/s, on the
# tree the race-fix verdict selects, and bucket GPU time (drivers/serving/parse_kernel_tail.py).
# The small-op tail's share bounds what epilogue fusion can buy. Also: launches and estimated
# occupancy per bucket, and the packed_qo tile histogram (SGLANG_FUSED_TILE_STATS, host-side).
set -uo pipefail
W=/dev/shm/w
until [ -s $W/racefix_verdict ]; do sleep 60; done
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/port07 HF_HOME=$W/hf
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112
# Measured mixed-step barrier for the graphs-on 4-mode crash (root cause open; graphs_stepsync):
# a device sync after mixed steps only. No effect on stock MTP, which has no mixed steps.
export SGLANG_DEBUG_STEP_SYNC=mixed
unset FLASHINFER_FUSED_PLAN_DEBUG SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE DISABLE_CUDA_GRAPH
unset SGLANG_DEBUG_STEP_SYNC
source $W/tree_env.sh
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace_all.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export CODEX_MIN_CTX=32768 CODEX_MAX_CTX=65536 CODEX_N=36 CODEX_GROUPS=8 YARN_FACTOR=4.0 CONTEXT_LEN=131072
export CHAT_RATE=16 CODEX_RATE=0.1 CHAT_SECONDS=360
ROOT=$W/b1_tail_profile_20261013; mkdir -p $ROOT
cd $W/$TREE/tasks/realistic_workload_analysis
for arm in MTPMIXFUSED MTPMIX_LPT MTP_LPT; do
  OUT=$ROOT/$arm; PROF=$OUT/prof; mkdir -p $PROF
  [ -f $OUT/${arm}_validation.json ] && continue
  echo "START $arm $(date +%T)"
  ( # trigger the profiler 150 s into the run (past warmup, under steady load)
    for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:30013/health >/dev/null && break; sleep 5; done
    sleep 150
    curl -s -m 30 -X POST http://127.0.0.1:30013/start_profile -H 'Content-Type: application/json' \
      -d "{\"output_dir\": \"$PROF\", \"num_steps\": 60, \"activities\": [\"GPU\"]}" > $OUT/start_profile.json ) &
  SGLANG_FUSED_TILE_STATS=1 SGLANG_TORCH_PROFILER_DIR=$PROF timeout 2400 $W/venv/bin/python -u run_mixed_workloads.py --scenario mix_env --methods $arm --outdir $OUT 2>&1 | tail -3
  rc=${PIPESTATUS[0]}; wait
  echo "END $arm rc=$rc $(date +%T) tree=$TREE tracebacks=$(grep -c Traceback $OUT/server_$arm.log) start=[$(head -c 120 $OUT/start_profile.json)] traces=$(ls $PROF | wc -l)"
  grep -A12 "TILE_WASTE" $OUT/server_$arm.log | tail -13 | sed "s/^/[$arm] TILE /"
  $W/venv/bin/python $W/drivers/serving/parse_kernel_tail.py $(ls $PROF/*.json* 2>/dev/null) 2>&1 | head -40 | sed "s/^/[$arm] /"
done
echo B1_PROFILE_DONE
