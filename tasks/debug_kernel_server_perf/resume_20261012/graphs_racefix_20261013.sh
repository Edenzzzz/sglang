#!/bin/bash
# Graphs-on 4-mode crash, root-cause fix (2026-10-13). In the overlap loop the verify half's
# future_map.publish(new_seq_lens) and relay stash ran AFTER the forward-stream block, i.e. on the
# schedule stream, with no wait on the forward stream that produces those tensors: stale seq_lens
# were published and the next step resolved them into out-of-range indices (the async index assert).
# A sync after mixed steps cured it because it ordered exactly this. Fix: issue both on the forward
# stream. The "overlap off does not cure" bisect arm never reached a 2nd mixed step -- it died on
# `assert next_token_ids.is_cpu` because the synchronous path had no 4-mode handoff at all; that is
# fixed too (racefix_scheduler.patch). Fixed code lives in $W/up_racefix so the sweep in flight
# keeps its own tree.
#  (0) null case: UNFIXED tree, no barrier -- the crash must still reproduce here, or (2)-(3) mean nothing
#  (1) output gate, fixed tree, graphs on, no barrier
#  (2) serving, fixed tree, no barrier: MTPMIX_LPT and MTPMIXFUSED, chat 24/s + agent 0.1/s, twice
#  (3) synchronous path: fixed tree, --disable-overlap-schedule, MTPMIX_LPT
set -uo pipefail
W=/dev/shm/w
until grep -q TRACELAB_FULL_DONE $W/tracelab_full_v2_20261013.log 2>/dev/null; do
  pgrep -f '[t]racelab_full_v2' >/dev/null || break; sleep 60; done
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
[ -L $W/validated_upstream ] || { echo "validated_upstream is not the symlink to up"; exit 1; }
[ -f $W/up_racefix/python/sglang/srt/managers/scheduler.py ] || { echo "missing up_racefix tree"; exit 1; }
export SGLANG_FUSED_FI_ROOT=$W/port07 HF_HOME=$W/hf
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112
unset SGLANG_DEBUG_STEP_SYNC FLASHINFER_FUSED_PLAN_DEBUG SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE DISABLE_CUDA_GRAPH
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json CHAT_RATE=24 CODEX_RATE=0.1
ROOT=$W/graphs_racefix_20261013; mkdir -p $ROOT
summ() {  # <label> <server log>
  echo "END $1 rc=$2 $(date +%T) tracebacks=$(grep -c Traceback $3 2>/dev/null) graphs=$(grep -c 'cuda graph: True' $3 2>/dev/null) mixed_steps=$(grep -o 'MIXEDSPEC_STEPS [0-9]*' $3 2>/dev/null | tail -1) assert=[$(grep -m1 -o 'Assertion[^\"]*\|device-side assert[^\"]*' $3 2>/dev/null | cut -c1-80)]"
}
serve() {  # <label> <tree> <method> [extra env...]
  local label=$1 tree=$2 method=$3; shift 3
  local OUT=$ROOT/$label; mkdir -p $OUT
  [ -f $OUT/${method}_validation.json ] && { echo "SKIP $label"; return; }
  echo "START $label $(date +%T)"
  ( cd $W/$tree/tasks/realistic_workload_analysis && env PYTHONPATH=$W/$tree/python "$@" timeout 1500 \
      $W/venv/bin/python -u run_mixed_workloads.py --scenario agent_chat_rated --methods $method --outdir $OUT 2>&1 | tail -4 )
  summ $label $? $OUT/server_$method.log
}
# (0) null case
serve null_unfixed_MTPMIX_LPT up MTPMIX_LPT
# (1) output gate
GATE=$ROOT/q4_gate; mkdir -p $GATE
cp $W/q4_gate_graphs_20261012/MTP_LPT_Q4G.json $GATE/
( cd $W/up_racefix && export PYTHONPATH=$W/up_racefix/python CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET=$CHAT_DATASET
  export BENCH_CHUNK=512 CORRECTNESS_DECODE_TOKENS=1200 SGLANG_FUSED_DECODE_GUARD=0 MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
  D=$W/up_racefix/tasks/fused_specdec_workload/drivers/srv_correctness.py
  $W/venv/bin/python -u $D MTPMIXFUSED_RF_Q4G $GATE 2>&1 | tail -2
  $W/venv/bin/python -u $D --compare $GATE MTP_LPT_Q4G MTPMIXFUSED_RF_Q4G 2>&1 | tail -3; exit ${PIPESTATUS[0]} )
echo "RACEFIX_GATE_DONE status=$?"
# (2) serving, no barrier, fixed
for rep in 1 2; do
  serve fixed_MTPMIX_LPT_$rep up_racefix MTPMIX_LPT
  serve fixed_MTPMIXFUSED_$rep up_racefix MTPMIXFUSED
done
# (3) synchronous path
serve fixed_sync_MTPMIX_LPT up_racefix MTPMIX_LPT SERVER_EXTRA_ARGS=--disable-overlap-schedule
echo RACEFIX_DONE
