#!/bin/bash
# STEPTRACE runs for the EDF step-model check (apply_steptrace_pdmux.py): the final pdmux + EDF config vs the stock
# schedules (non-speculative chunked LPT = the unfused baseline; stock MTP) under PD multiplexing + EDF (deadline-ordered admission, no latency model, no chunk cap -- the worker-side EDF that won on the
# MTP server) vs plain PD multiplexing vs stock MTP (the bar's baseline), one tree, one session. Derived from the payoff
# A/B for EAGLE3 MTP under PD multiplexing (tree up_pdmux_mtp; output gate cleared -- MTP_PDMUX diverges from stock
# MTP exactly as much as non-speculative pdmux does from stock LPT, short 1 / long 11). Same frozen 32-64K TraceLab cohort at
# program concurrency 8 / 16 + ShareGPT 8 req/s, driver and workload as pdmux_conc_20261017.sh, all arms in one tree and one
# session: PDMUX (non-speculative MuxWise-style) | MTP_PDMUX (ours: MTP + SM-split decode) | EDFR50_F2048 (our previous
# final config: split + EDF + floor). CUDA graphs on. Yields to /dev/shm/w/codex_needs_gpu. Claude, 2026-10-17.
set -uo pipefail
W=/dev/shm/w
root=$W/steptrace_pdmux_20261017
exec 9>"$W/a2_kernel.lock"; flock 9
trap 'printf "PTR_EXIT status=%s time=%s\n" "$?" "$(date -Is)"' EXIT
if pgrep -f '[s]glang.launch_server|[r]un_mixed_workloads|[k]ernel_concurrency_driver' >/dev/null; then
  echo "GPU predecessor still running"; exit 1; fi
mkdir -p "$root"; cd "$W"
export PYTHONPATH=$W/up_pdmux_mtp/python HF_HOME=$W/hf MODEL=$W/qwen7b
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
# MTP_PDMUX at mem fraction 0.78 (driver default 0.85, overridden by the later flag): pdmux runs prefill unchunked, and at 0.85 the
# first 64K agent prompt OOMed in the MLP (1.8 GiB alloc, 1.57 GiB free) with the draft model resident; 0.78 still leaves a
# ~1.6M-token KV pool against the cohort's <= 16 x 64K working set.
# label | method | BENCH_CHUNK | prefill floor | extra server args
PD="--enable-pdmux --chunked-prefill-size -1 --disable-overlap-schedule"
PRIO="--enable-priority-scheduling --schedule-low-priority-values-first --priority-scheduling-preemption-threshold 999999"
# label | method | BENCH_CHUNK | prefill floor | extra server args | EDF (deadline order)
# decode_bs_divisor: pdmux gives decode SM group decode_bs * 6 // divisor (default 36); smaller = more decode SMs.
for d in 18 12 27 24 30; do printf "sm_group_num: 8\ndecode_bs_divisor: $d\nsplit_forward_token_budget: 65536\n" > $W/pdmux_div$d.yaml; done
# label | method | BENCH_CHUNK | floor | extra server args | EDF | extra env (two-tier SLO)
export SGLANG_STEP_TRACE=1
ARMS=("FINAL|LPT||0|$PD $PRIO --pdmux-config-path $W/pdmux_div30.yaml|1|SGLANG_EDF_UNCACHED_DEADLINE=1"
      "LPT|LPT||0||0|" "MTP|MTP_LPT||0||0|")
REP=${REP:-1}
for C in 8 16; do
  for v in "${ARMS[@]}"; do
    IFS='|' read -r label method chunk floor extra edf xenv <<< "$v"
    out=$root/rep$REP/cohort_c$C/$label
    [ -s $W/codex_needs_gpu ] && { echo "PTR_YIELD cohort_c$C $label"; exit 0; }
    [ -s "$out/${method}_agent.json" ] && continue
    mkdir -p "$out"
    if [ -n "$chunk" ]; then export BENCH_CHUNK=$chunk; else unset BENCH_CHUNK; fi
    echo "PTR_START cohort_c$C $label method=$method extra=[$extra] $(date -Is)"
    if [ "$edf" = 1 ]; then edfenv=(SGLANG_EDF_SCHEDULING=1 SGLANG_EDF_LATENCY_MODEL= SGLANG_TTFT_SLO_MS=$TTFT_SLO_MS SGLANG_EDF_ACCEPT_MODE=none)
    else edfenv=(SGLANG_EDF_SCHEDULING=0); fi
    read -r -a xenvarr <<< "${xenv:-}"
    edfenv+=("${xenvarr[@]}")
    env "${edfenv[@]}" ITL_SLO_MS=50 SGLANG_EDF_PREFILL_FLOOR=$floor BENCH_SESSION_CONCURRENCY=$C SERVER_EXTRA_ARGS="$extra" \
      timeout --kill-after=30 5400 "$W/venv/bin/python" -u drivers/kernel_concurrency_driver.py \
      --scenario kernel_concurrency_env --methods "$method" --outdir "$out" > "$out/driver.log" 2>&1
    rc=$?
    L=$(ls $out/server_*.log 2>/dev/null | head -1)
    echo "PTR_END rep${REP}_cohort_c${C}_$label rc=$rc $(date -Is) edf_enabled=$(grep -c enable_priority_scheduling.:.True $L 2>/dev/null) tracebacks=$(grep -c Traceback $L 2>/dev/null) fallback=$(grep -ci fallback $L 2>/dev/null) pdmux=$(grep -ci 'pdmux\|green.context\|greenctx' $L 2>/dev/null) split=[$(grep -o 'FUSED_SPLIT_CALLS [0-9]*' $L 2>/dev/null | tail -1)] programs=[$(grep -o '\"concurrency_limit\": [0-9]*' $out/*programs*.json $out/*.log 2>/dev/null | head -1)]"
  done
done
echo "PTR_DONE $(date -Is)"
