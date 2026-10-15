#!/bin/bash
# Gate control (2026-10-16), queued after steptrace_tracelab_20261016.sh. resv_gate_20261015 passed both
# reservation trees by the rule (long prefix >= 8) but at short=1 / long=11 vs the MTP-repeat floor of
# 15 / 39, and no reservation fired there (so it tested split + EDF). Which part moves outputs: the
# split kernel (MTPMIX_SPLIT, no EDF), EDF scheduling (MTPMIX_LPT_EDFR, stock FA2 4-mode, no split),
# or both (MTPMIX_SPLIT_EDFR repeat)? All vs a fresh MTP_LPT in the same run.
# Earlier header follows.
# Serving-output gate for the EDF reservation trees (2026-10-15), queued after edf_slo80_20261015.sh.
# Greedy outputs vs stock MTP_LPT on batches forced to mix prefill chunks with decode rows
# (drivers/srv_correctness_edf_20261014.py, the Codex EDF gate driver; same env as
# s4_edf_budgetfix_gate_20261014.sh). Noise floor = a second MTP_LPT run vs the first. Candidates:
# MTPMIX_SPLIT_EDFR + floor 2048 + reservation mode 1 (up_edf_resv) and mode 2 (up_edf_resv2), guard 0.
# Each server log is audited: no traceback / fallback / width mismatch, CUDA graphs on, EDF counters
# nonzero for EDF arms. Claude, 2026-10-15.
set -uo pipefail
W=/dev/shm/w
root=$W/gate_control_20261016
until grep -q 'TRACE_DONE\|TRACE_EXIT' $W/steptrace_tracelab_20261016.log 2>/dev/null; do sleep 60; done
exec 9>"$W/a2_kernel.lock"; flock 9
trap 'printf "GCTL_EXIT status=%s time=%s\n" "$?" "$(date -Is)"' EXIT
for _ in $(seq 60); do pgrep -f '[s]glang.launch_server|[k]ernel_concurrency_driver|[f]used_reference|[r]un_mixed|[s]rv_correctness' >/dev/null || break; sleep 5; done
if pgrep -f '[s]glang.launch_server|[k]ernel_concurrency_driver|[f]used_reference|[r]un_mixed|[s]rv_correctness' >/dev/null; then echo "GPU predecessor"; exit 1; fi
mkdir -p "$root"; cd "$W"
source "$W/tree_env.sh"
export HF_HOME=$W/hf MODEL=$W/qwen7b SGLANG_FUSED_FI_ROOT=$W/port07
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112 SGLANG_SPLIT_OVERLAP=0
export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3 SGLANG_FUSED_DECODE_GUARD=0
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export SGLANG_EDF_MAX_PROFILED_PREFILL_PREFIX=262144 SGLANG_EDF_LONG_PREFIX_PROJECTION_THRESHOLD=262145
export SGLANG_TTFT_SLO_MS=2000 SGLANG_ITL_SLO_MS=50 SGLANG_EDF_PLAN_OVERHEAD_MS=0
export SGLANG_EDF_ACCEPT_MODE=req SGLANG_EDF_ACCEPT_LOG_EVERY=1000 SGLANG_EDF_PREFILL_FLOOR=2048
unset SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH SGLANG_FUSED_NATIVE_PDL SGLANG_FUSED_KERNEL_EXEC
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE SGLANG_FUSED_VERIFY_SNAPSHOT
unset SGLANG_FUSED_DISPATCH FLASHINFER_FUSED_PLAN_DEBUG SGLANG_STEP_TRACE
unset FUSED_NOSPLIT FUSED_FA3_UNPACKED FUSED_FA3_CLUSTER_PAIR FUSED_PAGED_TMA_BUILD
model=linear:$W/edf_dynchunk_20261015/models/MTPMIX_SPLIT_step_model.json
model_lpt=linear:$W/edf_dynchunk_20261015/models/MTPMIX_LPT_step_model.json
test -s "${model_lpt#linear:}"
test -s "${model#linear:}"
driver=$W/drivers/srv_correctness_edf_20261014.py
sha256sum "$driver" $W/up_edf_resv/python/sglang/srt/managers/scheduler.py \
  $W/up_edf_resv2/python/sglang/srt/managers/scheduler.py "${model#linear:}" > "$root/sources.sha256"
# arm|method|tree|edf|reserve
for v in "MTP_LPT_A|MTP_LPT|up_edf_resv|0|0" "SPLITONLY|MTPMIX_SPLIT|up_edf_resv|0|0" \
         "EDFONLY|MTPMIX_LPT_EDFR|up_edf_resv|1|0" "SPLITEDF|MTPMIX_SPLIT_EDFR|up_edf_resv|1|0"; do
  IFS='|' read -r arm method tree edf resv <<< "$v"
  [ -s $W/codex_needs_gpu ] && { echo "GCTL_YIELD $arm"; exit 0; }
  name=${method}_${arm}
  echo "GCTL_START $name tree=$tree $(date -Is)"
  lat=; [ "$edf" = 1 ] && lat=$model; [ "$method" = MTPMIX_LPT_EDFR ] && lat=$model_lpt
  env PYTHONPATH=$W/$tree/python CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET="$CHAT_DATASET" BENCH_CHUNK=512 \
    CORRECTNESS_DECODE_TOKENS=1200 SGLANG_EDF_SCHEDULING=$edf SGLANG_EDF_LATENCY_MODEL=$lat \
    SGLANG_EDF_RESERVE=$resv timeout --kill-after=30 1200 "$W/venv/bin/python" -u "$driver" "$name" "$root" \
    > "$root/$name.run.log" 2>&1
  rc=$?
  L=$root/server_$name.log
  audit="tracebacks=$(grep -c Traceback $L) fallback=$(grep -ci fallback $L) width_mismatch=$(grep -c 'WIDTH MISMATCH' $L) graphs=$(grep -c 'cuda graph: True' $L) edf_chunk=$(grep -o 'EDF_RUNTIME_CHUNK calls=[0-9]*' $L | tail -1) reserve=[$(grep -o 'EDF_RESERVE calls=[0-9]*' $L | tail -1)] split=[$(grep -o 'FUSED_SPLIT_CALLS [0-9]*' $L | tail -1)]"
  cmp=
  if [ "$arm" != MTP_LPT_A ]; then
    "$W/venv/bin/python" -u "$driver" --compare "$root" MTP_LPT_MTP_LPT_A "$name" > "$root/$name.compare.log" 2>&1
    cmp=$(tail -1 "$root/$name.compare.log")
  fi
  echo "GCTL_END $name rc=$rc $(date -Is) $audit compare=[$cmp]"
done
echo "GCTL_DONE $(date -Is)"
