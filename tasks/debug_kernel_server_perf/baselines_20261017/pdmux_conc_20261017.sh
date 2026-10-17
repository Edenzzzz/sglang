#!/bin/bash
# Direct baseline: MuxWise-style intra-GPU PD multiplexing (sglang --enable-pdmux: GreenContext SM split between
# prefill and decode; non-speculative, chunked prefill and overlap off) on the single H200, on the load axis we use
# everywhere now -- agent load by program concurrency (ThunderAgent), chat by request rate. The 2026-10-13 version
# (resume_20261012/baselines_pdmux_20261013.sh) never ran and used the retracted sessions/s agent axis.
# Frozen 32-64K TraceLab cohort (48 programs / 96 rounds) + natural ShareGPT 8 req/s, C = 8 / 16, same tree, driver
# and workload as concurrency_sweep_20261015.sh. Arms, one session: LPT (stock non-speculative) | PDMUX | MTP (stock
# EAGLE3) | EDFR50_F2048 (ours: split + EDF 50 ms + prefill floor 2048). Yields to /dev/shm/w/codex_needs_gpu.
# Claude, 2026-10-17.
set -uo pipefail
W=/dev/shm/w
root=$W/pdmux_conc_20261017
exec 9>"$W/a2_kernel.lock"; flock 9
trap 'printf "PDMUX_EXIT status=%s time=%s\n" "$?" "$(date -Is)"' EXIT
if pgrep -f '[s]glang.launch_server|[r]un_mixed_workloads|[k]ernel_concurrency_driver' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
mkdir -p "$root"; cd "$W"
export PYTHONPATH=$W/up_edf_floor/python HF_HOME=$W/hf MODEL=$W/qwen7b
export SGLANG_FUSED_FI_ROOT=$W/port07
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112 SGLANG_SPLIT_OVERLAP=0
export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3 SGLANG_FUSED_DECODE_GUARD=0
export SGLANG_EDF_MAX_PROFILED_PREFILL_PREFIX=262144 SGLANG_EDF_LONG_PREFIX_PROJECTION_THRESHOLD=262145
export TTFT_SLO_MS=2000 SGLANG_EDF_ACCEPT_LOG_EVERY=20000
export MTP_STEP_MODEL=$W/edf_dynchunk_20261015/models/MTPMIX_SPLIT_step_model.json
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
unset SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH BENCH_CHUNK SGLANG_FUSED_NATIVE_PDL SGLANG_FUSED_KERNEL_EXEC
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE SGLANG_FUSED_VERIFY_SNAPSHOT SGLANG_FUSED_DISPATCH
unset FLASHINFER_FUSED_PLAN_DEBUG SGLANG_STEP_TRACE SGLANG_EDF_SCHEDULING SGLANG_EDF_LATENCY_MODEL SGLANG_EDF_ACCEPT_MODE
unset FUSED_NOSPLIT FUSED_FA3_UNPACKED FUSED_FA3_CLUSTER_PAIR FUSED_PAGED_TMA_BUILD SERVER_EXTRA_ARGS STREAMS
export TRACE_PATH=$W/workloads/tracelab/syfi_full_min.jsonl.gz
export AGENT_N=96 AGENT_GROUPS=48 AGENT_SEED=20261014 OTHER_N=2048 OTHER_RATE=8
export AGENT_MIN_CTX=32768 AGENT_MAX_CTX=65536 CONTEXT_LEN=131072 YARN_FACTOR=4.0
export BENCH_TRACE_MAX_TOTAL_TOKENS=131072
unset AGENT_RATE CHAT_RATE AGENT_RATE CODEX_RATE BENCH_MAX_CONCURRENCY
# label | method | BENCH_CHUNK | prefill floor | extra server args
ARMS=("LPT|LPT||0|" "PDMUX|LPT||0|--enable-pdmux --chunked-prefill-size -1 --disable-overlap-schedule"
      "MTP|MTP_LPT||0|" "EDFR50_F2048|MTPMIX_SPLIT_EDFR||2048|")
for C in 8 16; do
  for v in "${ARMS[@]}"; do
    IFS='|' read -r label method chunk floor extra <<< "$v"
    out=$root/cohort_c$C/$label
    [ -s $W/codex_needs_gpu ] && { echo "PDMUX_YIELD cohort_c$C $label"; exit 0; }
    [ -s "$out/${method}_agent.json" ] && continue
    mkdir -p "$out"
    if [ -n "$chunk" ]; then export BENCH_CHUNK=$chunk; else unset BENCH_CHUNK; fi
    echo "PDMUX_START cohort_c$C $label method=$method extra=[$extra] $(date -Is)"
    env ITL_SLO_MS=50 SGLANG_EDF_PREFILL_FLOOR=$floor BENCH_SESSION_CONCURRENCY=$C SERVER_EXTRA_ARGS="$extra" \
      timeout --kill-after=30 5400 "$W/venv/bin/python" -u drivers/kernel_concurrency_driver.py \
      --scenario kernel_concurrency_env --methods "$method" --outdir "$out" > "$out/driver.log" 2>&1
    rc=$?
    L=$(ls $out/server_*.log 2>/dev/null | head -1)
    echo "PDMUX_END cohort_c${C}_$label rc=$rc $(date -Is) tracebacks=$(grep -c Traceback $L 2>/dev/null) fallback=$(grep -ci fallback $L 2>/dev/null) pdmux=$(grep -ci 'pdmux\|green.context\|greenctx' $L 2>/dev/null) split=[$(grep -o 'FUSED_SPLIT_CALLS [0-9]*' $L 2>/dev/null | tail -1)] programs=[$(grep -o '\"concurrency_limit\": [0-9]*' $out/*programs*.json $out/*.log 2>/dev/null | head -1)]"
  done
done
echo "PDMUX_DONE $(date -Is)"
