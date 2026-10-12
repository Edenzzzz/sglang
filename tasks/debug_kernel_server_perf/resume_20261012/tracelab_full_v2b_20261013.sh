#!/bin/bash
# S2b: full-length TraceLab replay -- both providers, every context length up to 1M tokens, first
# rounds included, sessions picked in proportion to their rounds (replayed context median ~124K, as
# in the release). Agent-only, rate-swept; three arms; graphs on; 4-mode arms with the measured
# mixed-step barrier. Served at YaRN x32 (1,048,576 tokens) on Qwen2.5-7B + EAGLE3.
# v2 (2026-10-13): v1 (120 sessions x 2 rounds at 0.2-0.8 sessions/s) queued 16.8M prompt tokens at
# ~2K tok/s and could not finish. 24 sessions x 10 rounds with selection seed 187 reproduce the
# release's round-level context distribution (replayed p10/50/90/99 40K/127K/235K/869K vs release
# 38K/124K/256K/822K, checked with the loader itself); ~20 GPU-minutes of prefill per arm, so session
# rates 0.01/0.02/0.05 run under, near and over capacity.
# v2b (resume): measured prefill at 600K+ context is ~1.1K tok/s, so the 5.8M uncached prompt
# tokens per arm take ~65 min and even 0.01/s is over capacity; two rates (0.01, 0.05) instead of
# three. Same ROOT, so the stock arm at 0.01 (done on `up`; untouched by the race fix) is kept.
# The 4-mode arms run on the tree the race-fix verdict selects (tree_env.sh).
# v3 (2026-10-13 05:0x, replaces the v2b plan): the v2 stock arm hit its 90-min timeout having
# prefilled 14.4M tokens for a 5.8M-uncached-token replay -- the KV pool (1,823,088 tokens) cannot
# hold 24 sessions reaching 869K, so prefixes were evicted between rounds and every round
# re-prefilled. A single-GPU replay must keep the sessions' combined peak context inside the pool:
# 8 sessions x 30 rounds, seed 1014 -- combined peak 1.37M, loader-checked context p10/50/90/99
# 38K/123K/223K/390K vs release 38K/124K/256K/822K. The >400K tail cannot be co-resident with other
# sessions on one GPU (excluded by capacity, not by a length filter); it needs the multi-GPU stage.
# Served at YaRN x16 (524288). Rates 0.005 and 0.02 sessions/s.
set -uo pipefail
W=/dev/shm/w
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/port07 HF_HOME=$W/hf
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112 SGLANG_DEBUG_STEP_SYNC=mixed
unset FLASHINFER_FUSED_PLAN_DEBUG SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE DISABLE_CUDA_GRAPH
source $W/tree_env.sh
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_full_min.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export YARN_FACTOR=16.0 CONTEXT_LEN=524288 AGENT_MAX_CTX=1040000 AGENT_N=240 AGENT_GROUPS=8 AGENT_SEED=1014 CHAT_RATE=0
[ -s $TRACE_PATH ] || { echo "missing $TRACE_PATH"; exit 1; }
ROOT=$W/tracelab_full_v3_20261013; mkdir -p $ROOT
cd $W/$TREE/tasks/realistic_workload_analysis
for rate in 0.005 0.02; do
  export AGENT_RATE=$rate
  OUT=$ROOT/full_r$rate; mkdir -p $OUT
  for arm in MTP_LPT MTPMIX_LPT MTPMIXFUSED; do
    [ -f $OUT/${arm}_validation.json ] && continue
    echo "START full r$rate $arm $(date +%T)"
    timeout 7200 $W/venv/bin/python -u run_mixed_workloads.py --scenario tracelab_full_env --methods "$arm" --outdir "$OUT" 2>&1 | tail -6
    echo "END full r$rate $arm rc=${PIPESTATUS[0]} $(date +%T) tree=$TREE tracebacks=$(grep -c Traceback $OUT/server_$arm.log)"
    grep -h "TraceLab replayed context" $OUT/bench_${arm}_agent.log | head -1
  done
done
echo TRACELAB_FULL_DONE
