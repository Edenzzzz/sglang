#!/bin/bash
set -euo pipefail
W=/dev/shm/w
ROOT=$W/native_pdl_serving_20261013
grep -q '^NATIVE_PDL_V3_DONE ' "$W/native_pdl_v3_validate_20261013.log"
grep -q '^NATIVE_PDL_V3_EXIT status=0 ' "$W/native_pdl_v3_validate_20261013.log"
test ! -e "$ROOT"
exec 9>"$W/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference.py' >/dev/null; then
  echo "GPU predecessor still running; refusing duplicate launch"
  exit 1
fi
mkdir -p "$ROOT"
trap 'status=$?; printf "NATIVE_PDL_SERVING_RUN_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
grep -q '^PASS' "$W/racefix_verdict"
source "$W/tree_env.sh"
export PYTHONPATH=$W/up_native_pdl_v3/python
export HF_HOME=$W/hf MODEL=$W/qwen7b
export SGLANG_FUSED_FI_ROOT=$W/fi_native_pdl_v3
export SGLANG_FUSED_MERGED_DIR=$W/merged_native_pdl_v3
export TORCH_EXTENSIONS_DIR=$W/torch_ext_native_pdl_v3
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1
export FUSED_CONST_PERS_PARAMS=1 FUSED_WORK_BUDGET_SPLIT=1
export FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto4096 SGLANG_FUSED_SM_FA3=112
export SGLANG_SPLIT_OVERLAP=0 SGLANG_FUSED_SM_MARGIN=0
export SGLANG_A3_WORK_MAX_QO=32 MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
export TRACE_PATH=$W/workloads/tracelab/syfi_coding_trace_all.jsonl.gz
export CHAT_DATASET=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
unset SGLANG_FUSED_KERNEL_EXEC SGLANG_FUSED_DISPATCH SGLANG_FUSED_DECODE_GUARD
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE FLASHINFER_FUSED_PLAN_DEBUG
unset SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH BENCH_CHUNK STREAMS FUSED_NOSPLIT
unset SGLANG_FUSED_VERIFY_SNAPSHOT FUSED_KV_SHARE_BUILD FUSED_FA3_CLUSTER_PAIR
unset FUSED_CLUSTER FLASHINFER_FUSED_CLUSTER FUSED_FA3_KV_SHARE FUSED_FA3_KV_SHARE_TRACE
test -s "$TRACE_PATH"
test -s "$CHAT_DATASET"
sha256sum "$W/up_native_pdl_v3/python/sglang/srt/layers/attention/mixed_dispatch.py" \
  "$W/up_native_pdl_v3/python/sglang/srt/layers/attention/flashinfer_backend.py" \
  "$W/$TREE/tasks/realistic_workload_analysis/run_mixed_workloads.py" \
  "$W/$TREE/tasks/fused_specdec_workload/drivers/srv_correctness.py" > "$ROOT/sources.sha256"
env | LC_ALL=C sort | grep -E '^(SGLANG_FUSED_|SGLANG_SPLIT_|SGLANG_A3_|FUSED_|FLASHINFER_|MTP_|MODEL=|TRACE_PATH=|CHAT_DATASET=|PYTHONPATH=)' \
  > "$ROOT/environment.txt"

GATE=$ROOT/q4_gate
mkdir -p "$GATE"
cp "$W/q4_gate_graphs_20261012/MTP_LPT_Q4G.json" "$GATE/"
DRIVER=$W/$TREE/tasks/fused_specdec_workload/drivers/srv_correctness.py
for rule in SPLIT NATIVE; do
  enabled=0
  method=MTPMIX_SPLIT
  if [ "$rule" = NATIVE ]; then enabled=1; method=MTPMIXFUSED; fi
  arm=${method}_${rule}_Q4G
  echo "GATE_START $arm $(date -Is)"
  CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET="$CHAT_DATASET" BENCH_CHUNK=512 \
    CORRECTNESS_DECODE_TOKENS=1200 SGLANG_FUSED_DECODE_GUARD=0 SGLANG_FUSED_NATIVE_PDL=$enabled \
    "$W/venv/bin/python" -u "$DRIVER" "$arm" "$GATE" > "$GATE/$arm.run.log" 2>&1
  if [ "$enabled" = 1 ]; then grep -q 'NATIVE_FA2_PDL_CALLS' "$GATE/server_$arm.log"; fi
  "$W/venv/bin/python" -u "$DRIVER" --compare "$GATE" MTP_LPT_Q4G "$arm" \
    > "$GATE/$arm.compare.log" 2>&1
  tail -1 "$GATE/$arm.compare.log"
done
echo "NATIVE_PDL_SERVING_GATE_DONE $(date -Is)"

cd "$W/$TREE/tasks/realistic_workload_analysis"
for mixture in "mix32 32768 65536 36 8 4.0 131072" "mix64 65536 131072 24 8 4.0 131072" "mix128 131072 262144 16 4 8.0 262144"; do
  read -r name CODEX_MIN_CTX CODEX_MAX_CTX CODEX_N CODEX_GROUPS YARN_FACTOR CONTEXT_LEN <<< "$mixture"
  export CODEX_MIN_CTX CODEX_MAX_CTX CODEX_N CODEX_GROUPS YARN_FACTOR CONTEXT_LEN
  export CHAT_RATE=24 CODEX_RATE=0.1 CHAT_SECONDS=$((CODEX_N * 10))
  for label in SPLIT_1 NATIVE_1 SPLIT_2 NATIVE_2; do
    OUT=$ROOT/${name}_r24/$label
    test ! -e "$OUT"
    mkdir -p "$OUT"
    enabled=0
    method=MTPMIX_SPLIT
    if [[ "$label" == NATIVE_* ]]; then enabled=1; method=MTPMIXFUSED; fi
    echo "START $name $label $(date -Is)"
    SGLANG_FUSED_NATIVE_PDL=$enabled timeout 2400 "$W/venv/bin/python" -u run_mixed_workloads.py \
      --scenario mix_env --methods "$method" --outdir "$OUT" > "$OUT/driver.log" 2>&1
    "$W/venv/bin/python" - "$OUT" "$name" "$label" "$((CHAT_RATE * CHAT_SECONDS))" "$CODEX_N" "$method" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

directory = Path(sys.argv[1])
method = sys.argv[6]
log_path = directory / f"server_{method}.log"
log = log_path.read_text()
if method == "MTPMIX_SPLIT":
    matches = re.findall(r"FUSED_SPLIT_CALLS (\d+).*overlapped=(\d+)", log)
    assert matches, "no split calls"
    calls, overlapped = map(int, matches[-1])
    assert calls > 0 and overlapped == 0, (calls, overlapped)
else:
    matches = re.findall(r"NATIVE_FA2_PDL_CALLS (\d+) fa3_rows=(\d+) fa2_rows=(\d+) fa3_splits=(\d+)", log)
    assert matches, "no native PDL calls"
    calls, fa3_rows, fa2_rows, fa3_splits = map(int, matches[-1])
    assert calls > 0 and fa3_rows > 0 and fa2_rows > 0, matches[-1]
    overlapped = None
assert "cuda graph: True" in log, "no graph decode observed"
assert "Traceback" not in log and "fallback" not in log.lower(), "runtime errors"
validation = json.loads((directory / f"{method}_validation.json").read_text())
assert validation.get("completed") and validation.get("mixed_steps", 0) > 0, validation
for stream, expected in zip(("chat", "codex"), map(int, sys.argv[4:6])):
    result = json.loads((directory / f"{method}_{stream}.json").read_text())
    assert result["completed"] == expected and len(result["ttfts"]) == expected
    assert not any(result.get("errors", [])), stream
summary = {
    "mixture": sys.argv[2], "arm": sys.argv[3], "method": method, "execution_calls": calls,
    "overlapped": overlapped, "mixed_steps": validation["mixed_steps"],
    "graph_decode_lines": log.count("cuda graph: True"),
    "fallbacks": log.lower().count("fallback"), "tracebacks": log.count("Traceback"),
    "plan_counters": re.findall(r"FUSED_SPLIT_PLAN ok .*", log)[-1:],
    "one_sided_counters": re.findall(r"FUSED_SPLIT_ONESIDED .*", log)[-1:],
    "server_log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
}
(directory / "execution_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
PY
    for stream in chat codex; do
      cp "$OUT/${method}_${stream}.json" "$ROOT/${name}_r24/${label}_${stream}.json"
    done
    echo "END $name $label $(date -Is)"
  done
done
"$W/venv/bin/python" "$W/drivers/serving/eval_curves.py" --root "$ROOT" \
  --slo '{"chat":[2000,50],"codex":[10000,80]}' --csv "$ROOT/native_pdl_serving.csv"
echo "NATIVE_PDL_SERVING_ALL_DONE $(date -Is)"

