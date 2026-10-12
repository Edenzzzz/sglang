#!/bin/bash
# Graphs-on 4-mode crash (both MTPMIX_LPT and MTPMIXFUSED, ~60 s into agent_chat_rated):
# torch IndexKernel out-of-bounds, surfacing in flashinfer plan() under
# decode_cuda_graph_runner.load_batch -> init_forward_metadata_out_graph. The assert is async, so
# the frame is a victim. Re-run the UNFUSED arm (fewer moving parts) with launch-blocking and the
# 4-mode phase sync so the faulting op itself raises.
set -uo pipefail
W=/dev/shm/w
until grep -q A3_V2_DONE $W/a3_winner_map_v2_20261012.log 2>/dev/null; do
  pgrep -f '[a]3_winner_map_v2' >/dev/null || break; sleep 60; done
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
export PYTHONPATH=$W/validated_upstream/python HF_HOME=$W/hf
unset DISABLE_CUDA_GRAPH SGLANG_FUSED_KERNEL_EXEC
export MODEL=$W/qwen7b TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json CHAT_RATE=24 CODEX_RATE=0.1
export CUDA_LAUNCH_BLOCKING=1 SGLANG_DEBUG_MIX_SYNC=1 TORCH_SHOW_CPP_STACKTRACES=1
OUT=$W/graphs_repro_20261012; mkdir -p $OUT
cd $W/validated_upstream/tasks/realistic_workload_analysis
echo "START repro $(date +%T)"
timeout 1500 $W/venv/bin/python -u run_mixed_workloads.py --scenario agent_chat_rated --methods MTPMIX_LPT --outdir $OUT 2>&1 | tail -4
echo "END repro rc=$? $(date +%T)"
grep -n -B3 -A40 "Traceback" $OUT/server_MTPMIX_LPT.log | head -120
grep -n "MIXSYNC" $OUT/server_MTPMIX_LPT.log | tail -8
echo GRAPHS_REPRO_DONE
