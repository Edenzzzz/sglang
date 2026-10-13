#!/bin/bash
set -euo pipefail
workdir=/dev/shm/w
output_dir=$workdir/native_pdl_combined_validate_20261013
trap 'status=$?; printf "NATIVE_PDL_COMBINED_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
predecessor_pid=506950
predecessor_log=$workdir/kvshare_bulk_kernel_20261013.log
echo "WAIT_KVSHARE_BULK $(date -Is)"
while kill -0 "$predecessor_pid" 2>/dev/null; do
  predecessor_state=$(ps -p "$predecessor_pid" -o stat= || true)
  if [[ "$predecessor_state" == Z* ]]; then break; fi
  if ! predecessor_command=$(tr '\0' ' ' < "/proc/$predecessor_pid/cmdline"); then break; fi
  [[ "$predecessor_command" == "bash kvshare_bulk_kernel_20261013.sh "* || "$predecessor_command" == "bash /dev/shm/w/kvshare_bulk_kernel_20261013.sh "* ]]
  sleep 15
done
grep '^KVSHARE_BULK_EXIT status=' "$predecessor_log"
grep -q '^NATIVE_FA3_ENTRY_DONE ' "$workdir/native_fa3_entry_ab_20261013.log"
grep -q '^NATIVE_FA3_ENTRY_EXIT status=0 ' "$workdir/native_fa3_entry_ab_20261013.log"
grep -q '^NATIVE_PDL_COMPACT_DONE ' "$workdir/native_pdl_compact_validate_20261013.log"
grep -q '^NATIVE_PDL_COMPACT_EXIT status=0 ' "$workdir/native_pdl_compact_validate_20261013.log"
exec 9>"$workdir/a2_kernel.lock"
flock -n 9
if pgrep -f '[s]glang.launch_server|[s]rv_correctness.py|[r]un_mixed_workloads|[f]used_reference' >/dev/null; then
  echo "GPU predecessor still running; refusing duplicate launch"
  exit 1
fi
cd "$workdir"
sha256sum -c native_pdl_combined_overlay_20261013/baseline.sha256
test ! -e fi_native_pdl_combined
test ! -e up_native_pdl_combined
mkdir "$output_dir" up_native_pdl_combined
cp -a fi_native_pdl_compact fi_native_pdl_combined
cp -a up_native_pdl_compact/python up_native_pdl_combined/python
cp -a native_pdl_combined_overlay_20261013/port07/. fi_native_pdl_combined/
cp drivers/fused_reference_native_pdl_compact.py drivers/fused_reference_native_pdl_combined.py
sha256sum -c native_pdl_combined_overlay_20261013/candidate.sha256
cp native_pdl_combined_overlay_20261013/baseline.sha256 native_pdl_combined_overlay_20261013/candidate.sha256 "$output_dir/"
export PYTHONPATH=$workdir/up_native_pdl_combined/python HF_HOME=$workdir/hf
export SGLANG_FUSED_FI_ROOT=$workdir/fi_native_pdl_combined SGLANG_FUSED_MERGED_DIR=$workdir/merged_native_pdl_combined
export TORCH_EXTENSIONS_DIR=$workdir/torch_ext_native_pdl_combined TORCH_CUDA_ARCH_LIST=9.0a
export SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1 FUSED_CONST_PERS_PARAMS=1
export FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1 FLASHINFER_FUSED_COLAUNCH=1
export FUSED_FA2_KV_CHUNK=auto4096 HEADS=28,4 SM_SPLIT=112
export FUSED_REFERENCE_WORKSPACE_BYTES=1073741824 FUSED_FA3_CTA_Q=128 FUSED_PACKED_Q=7
export TIME_CUDA_GRAPH=1 TIME_ITERS=50 TIME_WARMUP=10
export FUSED_FA3_DEDICATED_PRIMARY=1
unset FUSED_NOSPLIT FUSED_KV_SHARE_BUILD FUSED_CLUSTER FLASHINFER_FUSED_CLUSTER FUSED_FA3_CLUSTER_PAIR
unset FLASHINFER_FUSED_SYMMETRIC FLASHINFER_FUSED_SKIP_FA3 FLASHINFER_FUSED_PLAN_DEBUG FA3_SKIP_KV_LOAD TIME_FORCE
driver=$workdir/drivers/fused_reference_native_pdl_combined.py
echo "BUILD_START $(date -Is)"
timeout --kill-after=30 1500 "$workdir/venv/bin/python" -u -c \
  'from sglang.srt.layers.attention.fused_fa3_loader import load_fused_fa3; print(load_fused_fa3(verbose=True, cta_q=128))' \
  > "$output_dir/build.log" 2>&1
echo "BUILD_DONE $(date -Is)"
NATIVE_PDL_MODE=independent NATIVE_PDL_TRACE=1 NATIVE_PDL_REQUIRE_OVERLAP=1 \
  NATIVE_PDL_GRAPH_PLAN=1 MODE=mixed_verify PREFILL_QO=512 PREFILL_KV=65536 VERIFY_QO=4 \
  VERIFY_ROWS=8 VERIFY_KV=8192 TIME_AB=0 timeout --kill-after=20 240 "$workdir/venv/bin/python" \
  -u "$driver" > "$output_dir/overlap_gate.log" 2>&1
grep -q '^PASS$' "$output_dir/overlap_gate.log"
test "$(grep -c 'completed_fa2_during_fa3=True' "$output_dir/overlap_gate.log")" = 2
shapes=("fa3_q32_kv65536|MODE=band1 BAND_SIDE=fa3 BAND_QO=32 BAND_KV=65536 BAND_ROWS=1"
        "fa3_q17_kv32769|MODE=band1 BAND_SIDE=fa3 BAND_QO=17 BAND_KV=32769 BAND_ROWS=1"
        "mixed_q32_kv65536|MODE=mixed_verify PREFILL_QO=32 PREFILL_KV=65536 VERIFY_QO=4 VERIFY_ROWS=8 VERIFY_KV=8192"
        "mixed_q256_kv128k|MODE=mixed_verify PREFILL_QO=256 PREFILL_KV=131072 VERIFY_QO=4 VERIFY_ROWS=16 VERIFY_KV=131072"
        "mixed_varied|MODE=native_varied")
for shape in "${shapes[@]}"; do
  read -r -a fixture_env <<< "${shape#*|}"
  for arm in serial independent; do
    label=${shape%%|*}_$arm
    echo "START $label $(date -Is)"
    NATIVE_PDL_MODE=$arm NATIVE_PDL_GRAPH_PLAN=1 TIME_AB=1 env "${fixture_env[@]}" \
      timeout --kill-after=20 300 "$workdir/venv/bin/python" -u "$driver" > "$output_dir/$label.log" 2>&1
    "$workdir/venv/bin/python" - "$output_dir/$label.log" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
log = path.read_text()
assert log.rstrip().endswith("PASS") and "Traceback" not in log and "NON-FINITE OUTPUT" not in log, path
plan = re.findall(r"NATIVE_PRIMARY_PLAN max_splits=(\d+) merge_rows=(\d+)", log)
assert plan and all(int(splits) > 1 and int(rows) > 0 for splits, rows in plan), plan
rows = re.findall(r"bad rows (\d+)/(\d+)", log)
replay = re.findall(r"FUSED REPLAY CORRECTNESS: bad_heads=(\d+)/(\d+)", log)
controls = re.findall(r"UNFUSED CORRECTNESS: bad_heads=(\d+)/(\d+)", log)
assert rows and replay and controls and "[time] cuda_graph=True" in log, path
assert all(int(bad) == 0 and int(total) > 0 for bad, total in rows + replay + controls), path
audit = {"case": path.stem, "primary_plan": plan, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit), flush=True)
PY
  done
done
echo "NATIVE_PDL_COMBINED_SPLIT_GATES_DONE $(date -Is)"
export MODEL=$workdir/qwen7b SGLANG_FUSED_NATIVE_PDL=1 SGLANG_FUSED_SM_FA3=112
export SGLANG_FUSED_SM_MARGIN=0 SGLANG_SPLIT_OVERLAP=0 MTP_DRAFT_TOKENS=4 MTP_SPEC_STEPS=3
export CORRECTNESS_SYNTHETIC=0 CORRECTNESS_DECODE_TOKENS=1200 BENCH_CHUNK=512
export CORRECTNESS_DATASET=$workdir/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
export SGLANG_FUSED_SPLIT_VERIFY=1 SGLANG_FUSED_VERIFY_SNAPSHOT=1 SGLANG_FUSED_SPLIT_TOL=0.02
unset SGLANG_FUSED_DISPATCH SGLANG_DEBUG_STEP_SYNC DISABLE_CUDA_GRAPH SGLANG_FUSED_VERIFY_EPILOGUE
export SGLANG_FUSED_DECODE_GUARD=0
gate=$output_dir/serving_gate
mkdir "$gate"
cp "$workdir/q4_gate_graphs_20261012/MTP_LPT_Q4G.json" "$gate/"
arm=MTPMIXFUSED_NATIVE_COMBINED_Q4G
correctness=$workdir/up_racefix/tasks/fused_specdec_workload/drivers/srv_correctness.py
echo "SERVING_GATE_START $(date -Is)"
"$workdir/venv/bin/python" -u "$correctness" "$arm" "$gate" > "$gate/$arm.run.log" 2>&1
"$workdir/venv/bin/python" -u "$correctness" --compare "$gate" MTP_LPT_Q4G "$arm" > "$gate/$arm.compare.log" 2>&1
grep -q 'NATIVE_PDL_COMPACT_CALLS' "$gate/server_$arm.log"
grep -q 'FUSED_KERNEL_VERIFY PASS.*native_pdl=1' "$gate/server_$arm.log"
tail -1 "$gate/$arm.compare.log"
echo "NATIVE_PDL_COMBINED_DONE $(date -Is)"

