#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
root=$workdir/s4_trace_matrix_v2_20261014
trap 'status=$?; printf "S4_TRACE_MATRIX_V2_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
grep -q '^FA3_UNPACKED_DONE ' "$workdir/fa3_unpacked_kernel_20261014.log"
grep -q '^FA3_UNPACKED_EXIT status=0 ' "$workdir/fa3_unpacked_kernel_20261014.log"
test ! -e "$root"
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness|[r]un_mixed_workloads|[f]used_reference|[f]a3_unpacked_reference' >/dev/null; then
  echo 'GPU predecessor still running; refusing duplicate launch'
  exit 1
fi
mkdir "$root"
source "$workdir/tree_env.sh"
test "$TREE" = up_racefix
cd "$workdir"
sha256sum -c s4_trace_v2_overlay_20261014/baseline.sha256
test ! -e up_s4_trace_v2
mkdir up_s4_trace_v2
cp -a up_racefix/python up_s4_trace_v2/python
cp s4_trace_v2_overlay_20261014/scheduler.py up_s4_trace_v2/python/sglang/srt/managers/scheduler.py
sha256sum -c s4_trace_v2_overlay_20261014/candidate.sha256
export PYTHONPATH=$workdir/up_s4_trace_v2/python HF_HOME=$workdir/hf MODEL=$workdir/qwen7b
export SGLANG_FUSED_FI_ROOT=$workdir/port07
export SGLANG_FUSED_MERGED_DIR=$workdir/merged_colaunch TORCH_EXTENSIONS_DIR=$workdir/torch_ext_colaunch_auto2
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=112 SGLANG_SPLIT_OVERLAP=0
export MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3 SGLANG_A3_WORK_MAX_QO=32
export TRACE_PATH=$workdir/workloads/tracelab/syfi_coding_trace_all.jsonl.gz
export CHAT_DATASET=$workdir/workloads/ShareGPT_V3_unfiltered_cleaned_split.json CODEX_RATE=0.1
unset SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH SGLANG_FUSED_NATIVE_PDL SGLANG_FUSED_KERNEL_EXEC
unset SGLANG_FUSED_SPLIT_VERIFY SGLANG_FUSED_VERIFY_EPILOGUE SGLANG_FUSED_VERIFY_SNAPSHOT
unset SGLANG_FUSED_DISPATCH SGLANG_FUSED_DECODE_GUARD FLASHINFER_FUSED_PLAN_DEBUG
unset SGLANG_EDF_SCHEDULING SGLANG_EDF_LATENCY_MODEL SGLANG_EDF_ACCEPT_MODE SGLANG_STEP_TRACE
unset FUSED_NOSPLIT FUSED_FA3_UNPACKED FUSED_FA3_CLUSTER_PAIR FUSED_PAGED_TMA_BUILD
test -s "$TRACE_PATH"
test -s "$CHAT_DATASET"
driver=$workdir/up_racefix/tasks/fused_specdec_workload/drivers/srv_correctness.py
runner=$workdir/up_racefix/tasks/realistic_workload_analysis/run_mixed_workloads.py
sha256sum "$driver" "$runner" "$PYTHONPATH/sglang/srt/managers/scheduler.py" \
  "$PYTHONPATH/sglang/srt/layers/attention/flashinfer_backend.py" > "$root/sources.sha256"
env | LC_ALL=C sort | grep -E '^(SGLANG_|FUSED_|FLASHINFER_|MTP_|MODEL=|TRACE_PATH=|CHAT_DATASET=|PYTHONPATH=)' \
  > "$root/environment.txt"
gate=$root/q4_gate
mkdir "$gate"
for method in MTP_LPT MTPMIX_LPT MTPMIX_SPLIT; do
  arm=${method}_S4_Q4G
  echo "GATE_START $arm $(date -Is)"
  CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DATASET="$CHAT_DATASET" BENCH_CHUNK=512 \
    CORRECTNESS_DECODE_TOKENS=1200 SGLANG_FUSED_DECODE_GUARD=0 SGLANG_STEP_TRACE=1 \
    timeout --kill-after=30 900 "$workdir/venv/bin/python" -u "$driver" "$arm" "$gate" \
    > "$gate/$arm.run.log" 2>&1
  if [ "$method" != MTP_LPT ]; then
    "$workdir/venv/bin/python" -u "$driver" --compare "$gate" MTP_LPT_S4_Q4G "$arm" \
      > "$gate/$arm.compare.log" 2>&1
    tail -1 "$gate/$arm.compare.log"
  fi
  "$workdir/venv/bin/python" - "$gate/server_$arm.log" "$method" <<'PY'
from pathlib import Path
import re
import sys

log = Path(sys.argv[1]).read_text()
features = re.findall(r"STEPTRACE \w+ .*?np=(\d+) ptok=(\d+) pf=(\d+) nd=(\d+) dkv=(\d+)", log)
assert len(features) > 20
assert any(int(prefill) > 0 and int(tokens) > 0 for prefill, tokens, _, _, _ in features)
if sys.argv[2] != "MTP_LPT":
    assert any(int(prefill) > 0 and int(decode) > 0 and int(context) > 0
               for prefill, _, _, decode, context in features)
print("S4_TRACE_FEATURE_GATE", sys.argv[2], len(features), "PASS", flush=True)
PY
done
echo "S4_TRACE_V2_GATE_DONE $(date -Is)"
fixtures=('mix32 32768 65536 36 8 4.0 131072 8'
          'mix32 32768 65536 36 8 4.0 131072 16'
          'mix32 32768 65536 36 8 4.0 131072 24'
          'mix64 65536 131072 24 8 4.0 131072 8'
          'mix64 65536 131072 24 8 4.0 131072 16'
          'mix128 131072 262144 16 4 8.0 262144 12')
fixture_index=0
for fixture in "${fixtures[@]}"; do
  read -r mixture CODEX_MIN_CTX CODEX_MAX_CTX CODEX_N CODEX_GROUPS YARN_FACTOR CONTEXT_LEN CHAT_RATE <<< "$fixture"
  export CODEX_MIN_CTX CODEX_MAX_CTX CODEX_N CODEX_GROUPS YARN_FACTOR CONTEXT_LEN CHAT_RATE
  export CHAT_SECONDS=$((CODEX_N * 10))
  methods=(MTPMIX_LPT MTPMIX_SPLIT)
  if (( fixture_index % 2 )); then methods=(MTPMIX_SPLIT MTPMIX_LPT); fi
  fixture_index=$((fixture_index + 1))
  for method in "${methods[@]}"; do
    label=${mixture}_r${CHAT_RATE}_${method}
    output=$root/$label
    mkdir "$output"
    echo "START $label $(date -Is)"
    SGLANG_STEP_TRACE=1 timeout --kill-after=30 3000 "$workdir/venv/bin/python" -u "$runner" \
      --scenario mix_env --methods "$method" --outdir "$output" > "$output/driver.log" 2>&1
    "$workdir/venv/bin/python" - "$output" "$method" "$mixture" "$CHAT_RATE" \
      "$((CHAT_RATE * CHAT_SECONDS))" "$CODEX_N" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

directory = Path(sys.argv[1])
method = sys.argv[2]
log_path = directory / f"server_{method}.log"
log = log_path.read_text()
assert "Traceback" not in log and "fallback" not in log.lower()
assert "cuda graph: True" in log
validation = json.loads((directory / f"{method}_validation.json").read_text())
assert validation.get("completed") and validation.get("mixed_steps", 0) > 0, validation
for stream, expected in zip(("chat", "codex"), map(int, sys.argv[5:7])):
    data = json.loads((directory / f"{method}_{stream}.json").read_text())
    assert data["completed"] == expected and len(data["ttfts"]) == expected
    assert not any(data.get("errors", []))
trace_lines = [line for line in log.splitlines() if "STEPTRACE " in line and " gpu=" in line]
assert len(trace_lines) >= 100
assert any("np=" in line and not "np=0 " in line and not "nd=0 " in line for line in trace_lines)
split_calls = re.findall(r"FUSED_SPLIT_CALLS (\d+).*overlapped=(\d+)", log)
if "SPLIT" in method:
    assert split_calls and int(split_calls[-1][0]) > 0 and int(split_calls[-1][1]) == 0
else:
    assert not split_calls
audit = {"method": method, "mixture": sys.argv[3], "chat_rate": int(sys.argv[4]),
         "completed": True, "mixed_steps": validation["mixed_steps"],
         "trace_rows": len(trace_lines), "tracebacks": 0, "fallbacks": 0,
         "split_calls": int(split_calls[-1][0]) if split_calls else 0,
         "server_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest()}
(directory / "execution_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
(directory / "steptrace.log").write_text("\n".join(trace_lines) + "\n")
print(json.dumps(audit), flush=True)
PY
    echo "END $label $(date -Is)"
  done
done
echo "S4_TRACE_MATRIX_V2_DONE $(date -Is)"
