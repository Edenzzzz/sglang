#!/bin/bash
# After the A3 winner map: (1) the serving-output gate at q4, the width the realistic A/B ran,
# for every arm of that A/B plus the two-executor control; (2) a second realistic round that adds
# MTPMIX_SPLIT (two-executor plumbing, no fused kernel), arms in rotated order.
set -uo pipefail
W=/dev/shm/w
until grep -q FA2_SLOW_PROF_DONE $W/fa2_slow_prof_20261012.log 2>/dev/null; do sleep 30; done
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[f]used_reference.py|[r]un_mixed_workloads' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/fi HF_HOME=$W/hf
export SGLANG_FUSED_FA2_TILES=48,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_WORK_BUDGET_SPLIT=1
export FUSED_CONST_PERS_PARAMS=1 FUSED_FA2_KV_CHUNK=1024 FLASHINFER_FUSED_BACKFILL=1
export SGLANG_FUSED_SM_FA3=112 DISABLE_CUDA_GRAPH=1
unset FLASHINFER_FUSED_PLAN_DEBUG SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE
unset SGLANG_FUSED_KERNEL_EXEC SGLANG_MIXED_SPEC

# (1) q4 output gate, same prompts/settings as the q8 gate except the chain width.
OUT=$W/q4_gate_20261012; mkdir -p "$OUT"
DRIVER=$W/validated_upstream/tasks/fused_specdec_workload/drivers/srv_correctness.py
( export CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET=$W/datasets/ShareGPT.original.json
  export BENCH_CHUNK=512 CORRECTNESS_DECODE_TOKENS=1200 SGLANG_FUSED_DECODE_GUARD=0
  export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
  for arm in MTP_LPT_Q4 MTPMIX_LPT_Q4 MTPMIX_SPLIT_Q4 MTPMIXFUSED_Q4; do
    echo "GATE_START $arm $(date +%T)"
    $W/venv/bin/python -u "$DRIVER" "$arm" "$OUT" 2>&1 | tail -4
  done
  for arm in MTPMIX_LPT_Q4 MTPMIX_SPLIT_Q4 MTPMIXFUSED_Q4; do
    $W/venv/bin/python -u "$DRIVER" --compare "$OUT" MTP_LPT_Q4 "$arm" 2>&1
  done
  $W/venv/bin/python -u "$DRIVER" --compare "$OUT" MTPMIX_LPT_Q4 MTPMIXFUSED_Q4 2>&1 )
echo Q4_GATE_DONE

# (2) realistic round 2, the runner's default q4 chain.
unset MTP_DRAFT_TOKENS MTP_SPEC_STEPS BENCH_CHUNK
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
export CHAT_DATASET=$W/datasets/ShareGPT.original.json CHAT_RATE=24 CODEX_RATE=0.1
OUT=$W/postfix_agent_chat_rated24_r2_20261012; mkdir -p "$OUT"
cd $W/validated_upstream/tasks/realistic_workload_analysis
for arm in MTPMIX_SPLIT MTPMIXFUSED MTPMIX_LPT MTP_LPT; do
  echo "START $arm $(date +%T)"
  $W/venv/bin/python -u run_mixed_workloads.py --scenario agent_chat_rated --methods "$arm" --outdir "$OUT" 2>&1 | tail -8
  echo "END $arm rc=$? $(date +%T)"
done
echo ROUND2_DONE
