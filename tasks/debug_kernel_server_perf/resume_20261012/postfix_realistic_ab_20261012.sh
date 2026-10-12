#!/bin/bash
# Post-RoPE-fix realistic serving A/B: the recorded 2026-10-08 headline setup (agent_chat_rated,
# chat 24/s + TraceLab codex 0.1 sessions/s) re-run on a model that is now verified correct.
# Three serialized arms on the single GPU: stock MTP, 4-mode unfused, 4-mode fused (A2 tail on).
# Uninstrumented (no plan debug / split verify); no CUDA graphs in ANY arm, because the fused
# path's graph-mode draft indexing issue is still open -- so this is not a production-graph claim.
set -uo pipefail
W=/dev/shm/w
OUT=$W/postfix_agent_chat_rated24_20261012
mkdir -p "$OUT"
exec 9>"$W/a2_kernel.lock"; flock -n 9 || { echo LOCKED; exit 1; }
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[f]used_reference.py|[r]un_mixed_workloads' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python SGLANG_FUSED_FI_ROOT=$W/fi HF_HOME=$W/hf
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
export CHAT_DATASET=$W/datasets/ShareGPT.original.json
export CHAT_RATE=24 CODEX_RATE=0.1 DISABLE_CUDA_GRAPH=1
export SGLANG_FUSED_FA2_TILES=48,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_WORK_BUDGET_SPLIT=1
export FUSED_CONST_PERS_PARAMS=1 FUSED_FA2_KV_CHUNK=1024 FLASHINFER_FUSED_BACKFILL=1
export SGLANG_FUSED_SM_FA3=112
unset FLASHINFER_FUSED_PLAN_DEBUG SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE
unset MTP_DRAFT_TOKENS MTP_SPEC_STEPS
echo "rope check:"; CUDA_VISIBLE_DEVICES= SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 $W/venv/bin/python $W/ropecheck.py 2>&1 | grep -o "yarn_override.*get_rope_config: ([0-9.]*"
cd $W/validated_upstream/tasks/realistic_workload_analysis
for arm in MTP_LPT MTPMIX_LPT MTPMIXFUSED; do
  echo "START $arm $(date +%T)"
  $W/venv/bin/python -u run_mixed_workloads.py --scenario agent_chat_rated --methods "$arm" --outdir "$OUT" 2>&1 | tail -8
  echo "END $arm rc=$? $(date +%T)"
done
echo POSTFIX_AB_DONE
