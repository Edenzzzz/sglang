#!/bin/bash
# EDF dynamic chunk (user, 2026-10-15): instead of a static prefill chunk, each step serves the
# decode/verify rows first and admits only the prefill the step-time model says fits inside the
# tightest row's slack (ITL_SLO_MS x acceptance - elapsed), deadline-ordered. Built on the Codex
# kernel worker's EDF runtime (up_edf_budgetfix: mode-aware piecewise model, corrected budget units),
# with the decode-population guard OFF so long prefills reach FA3, on the frozen 32-64K cohort
# (48 agent programs/96 rounds, 24 active; natural ShareGPT 8/s) used by chunk_itl_sweep*.
#  T  step traces, guard 0: MTPMIX_SPLIT and MTPMIX_LPT at chunk 8192 and 2048
#  F  fit_step_matrix_piecewise (piecewise-relative) on those four traces only
#  E  MTP | SPLIT chunk 3072 (static control) | SPLIT_EDFR itl50 | SPLIT_EDFR itl30 | repeats
# Claude, 2026-10-15.
set -uo pipefail
W=/dev/shm/w
root=$W/edf_dynchunk_20261015
until grep -q '^B1_LMHEAD_CEILING_DONE\|yield' $W/b1_lmhead_ceiling_20261015.log 2>/dev/null; do
  pgrep -f '[b]1_lmhead_ceiling' >/dev/null || break; sleep 30; done
exec 9>"$W/a2_kernel.lock"; flock 9
trap 'printf "EDF_DYNCHUNK_EXIT status=%s time=%s\n" "$?" "$(date -Is)"' EXIT
if pgrep -f '[s]glang.launch_server|[k]ernel_concurrency_driver|[f]used_reference|[r]un_mixed' >/dev/null; then echo "GPU predecessor"; exit 1; fi
test ! -e "$root" && mkdir -p "$root/traces" "$root/serving"
cd "$W"
export PYTHONPATH=$W/up_edf_budgetfix/python HF_HOME=$W/hf MODEL=$W/qwen7b
export SGLANG_FUSED_FI_ROOT=$W/port07
export SGLANG_FUSED_MERGED_DIR=$W/merged_colaunch TORCH_EXTENSIONS_DIR=$W/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112 SGLANG_SPLIT_OVERLAP=0
export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3 SGLANG_FUSED_DECODE_GUARD=0
export SGLANG_EDF_MAX_PROFILED_PREFILL_PREFIX=262144 SGLANG_EDF_LONG_PREFIX_PROJECTION_THRESHOLD=262145
export TTFT_SLO_MS=2000 SGLANG_EDF_ACCEPT_LOG_EVERY=20000
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export TRACE_PATH=$W/workloads/tracelab/syfi_full_min.jsonl.gz
export AGENT_N=96 AGENT_GROUPS=48 AGENT_SEED=20261014 OTHER_N=2048 OTHER_RATE=8
export AGENT_MIN_CTX=32768 AGENT_MAX_CTX=65536 CONTEXT_LEN=131072 YARN_FACTOR=4.0
export BENCH_TRACE_MAX_TOTAL_TOKENS=131072 BENCH_SESSION_CONCURRENCY=24
unset SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH BENCH_CHUNK SGLANG_FUSED_NATIVE_PDL SGLANG_FUSED_KERNEL_EXEC
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE SGLANG_FUSED_VERIFY_SNAPSHOT SGLANG_FUSED_DISPATCH
unset FLASHINFER_FUSED_PLAN_DEBUG SGLANG_STEP_TRACE SGLANG_EDF_SCHEDULING SGLANG_EDF_LATENCY_MODEL SGLANG_EDF_ACCEPT_MODE
unset FUSED_NOSPLIT FUSED_FA3_UNPACKED FUSED_FA3_CLUSTER_PAIR FUSED_PAGED_TMA_BUILD SERVER_EXTRA_ARGS STREAMS
sha256sum "$PYTHONPATH/sglang/srt/managers/scheduler.py" "$PYTHONPATH/sglang/srt/managers/edf_scheduler.py" \
  drivers/kernel_concurrency_driver.py drivers/serving/fit_step_matrix_piecewise_20261014.py > "$root/sources.sha256"
run() {  # label method chunk outdir [env...]
  local label=$1 method=$2 chunk=$3 out=$4; shift 4
  [ -s $W/codex_needs_gpu ] && { echo "EDF_DYNCHUNK_YIELD $label"; exit 0; }
  mkdir -p "$out"
  echo "EDF_DYNCHUNK_START $label method=$method chunk=${chunk:-8192} $(date -Is)"
  if [ -n "$chunk" ]; then export BENCH_CHUNK=$chunk; else unset BENCH_CHUNK; fi
  env "$@" timeout --kill-after=30 5400 "$W/venv/bin/python" -u drivers/kernel_concurrency_driver.py \
    --scenario kernel_concurrency_env --methods "$method" --outdir "$out" > "$out/driver.log" 2>&1
  local rc=$?; local L=$out/server_$method.log
  echo "EDF_DYNCHUNK_END $label rc=$rc $(date -Is) tracebacks=$(grep -c Traceback $L) fallback=$(grep -ci fallback $L) split=[$(grep -o 'FUSED_SPLIT_CALLS [0-9]*' $L | tail -1)] edf_chunk=[$(grep -o 'EDF_RUNTIME_CHUNK calls=[0-9]* requested=[0-9]* admitted=[0-9]* prefill=[0-9]*' $L | tail -1)]"
  return $rc
}
# T: traces
for v in "SPLIT_C8192|MTPMIX_SPLIT|" "LPT_C8192|MTPMIX_LPT|" "SPLIT_C2048|MTPMIX_SPLIT|2048" "LPT_C2048|MTPMIX_LPT|2048"; do
  IFS='|' read -r label method chunk <<< "$v"
  out=$root/traces/$label
  run "TRACE_$label" "$method" "$chunk" "$out" SGLANG_STEP_TRACE=1 || exit 1
  "$W/venv/bin/python" - "$out" "$method" "${chunk:-8192}" <<'PY' || exit 1
import hashlib, json, re, sys
from pathlib import Path
d, method, chunk = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
log_path = d / f"server_{method}.log"; log = log_path.read_text()
assert "Traceback" not in log and "fallback" not in log.lower()
v = json.loads((d / f"{method}_validation.json").read_text())
assert v.get("completed") and v.get("mixed_steps", 0) > 0, v
lines = [l for l in log.splitlines() if "STEPTRACE " in l and " gpu=" in l]
assert len(lines) >= 100
split = re.findall(r"FUSED_SPLIT_CALLS (\d+)", log)
assert (int(split[-1]) > 0) if "SPLIT" in method else not split
# chat_rate carries the chunk so the fitter sees distinct fixtures per method (same cohort, two chunks)
audit = {"method": method, "mixture": "mix32", "chat_rate": chunk, "completed": True,
         "mixed_steps": v["mixed_steps"], "trace_rows": len(lines), "tracebacks": 0, "fallbacks": 0,
         "split_calls": int(split[-1]) if split else 0,
         "server_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest()}
(d / "execution_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
(d / "steptrace.log").write_text("\n".join(lines) + "\n")
print("TRACE_AUDIT", json.dumps(audit), flush=True)
PY
done
# F: fit
"$W/venv/bin/python" drivers/serving/fit_step_matrix_piecewise_20261014.py --root "$root/traces" \
  --out "$root/models" --strategy piecewise-relative --require-methods MTPMIX_LPT,MTPMIX_SPLIT \
  --require-contexts mix32 --require-runs 2 > "$root/fit.log" 2>&1 || { cat "$root/fit.log"; exit 1; }
tail -20 "$root/fit.log"; echo "EDF_DYNCHUNK_FIT_DONE"
export MTP_STEP_MODEL=$root/models/MTPMIX_SPLIT_step_model.json
test -s "$MTP_STEP_MODEL"
# E: serving
for v in "MTP|MTP_LPT||" "SPLIT_C3072|MTPMIX_SPLIT|3072|" "EDFR_ITL50|MTPMIX_SPLIT_EDFR||50" \
         "EDFR_ITL30|MTPMIX_SPLIT_EDFR||30" "SPLIT_C3072_2|MTPMIX_SPLIT|3072|" "EDFR_ITL30_2|MTPMIX_SPLIT_EDFR||30"; do
  IFS='|' read -r label method chunk itl <<< "$v"
  run "$label" "$method" "$chunk" "$root/serving/$label" ITL_SLO_MS=${itl:-50} || true
done
echo "EDF_DYNCHUNK_DONE $(date -Is)"
