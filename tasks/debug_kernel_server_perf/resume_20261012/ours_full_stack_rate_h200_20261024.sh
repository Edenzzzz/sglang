#!/bin/bash
# Stage 2, the full single-node stack (H200 34c7bd503d461dc1, 2026-10-24): stock sglang on FA3 (mixed chunk; stock
# EAGLE3 MTP in four-mode) vs OURS = deadline-ordered scheduling (uncached-token deadlines, chunk preemption,
# deadline-sized chunks 1024:2048:6000) + the FA3 + FA2 native kernel (v3) + the 4-mode MTP step. Same frozen tree,
# Codex's audited paired protocol (opposite arm orders, two repeats, raw token arrivals), `--ours` in
# run_native_serving_pair_20261023.py. 128-256K first -- where the kernel alone already cuts agent / chat TPOT
# 20-24% -- then 32-64K. Per-tenant SLO attainment is computed from the raw streams afterwards. Claude.
# Rate sweep at 128-256K (ours_full_stack_rate_h200_20261024.sh): the same pair at chat 1 and 2 req/s, chat requests
# scaled to keep the same arrival window as chat 4 / 960, so attainment vs rate (with the chat-4 point of the main
# run) gives each system's highest load at >= 90% attainment. Waits for $1 (the main driver). Claude.
set -euo pipefail
[ "${1:-now}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
SRC=/dev/shm/w/qwen25_v3_common_screen_h200_20261023
BASE=/opt/tiger/w/ours_full_stack_rate_h200_20261024
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
mkdir -p "$BASE"
echo $$ > "$BASE/controller.pid"
trap 'printf "%s\n" "$?" > "$BASE/exit_status"' EXIT
# One complete root per stage: the harness hashes and imports its sources from the root it is given, and the audit
# groups every */summary.json under a root, so the two stages must not share one.
make_root() {
  local R=$1
  mkdir -p "$R"
  test -d "$R/python" || cp -a "$SRC/python" "$R/python"
  cp "$SRC"/*.py "$SRC"/*.sh "$SRC"/tracelab_unique.jsonl.gz "$R/"
  cp /dev/shm/w/perf/rss/native_fa3_fa2_pdl_20261023.py /dev/shm/w/perf/rss/run_native_serving_pair_20261023.py /dev/shm/w/perf/rss/native_serving_protocol_20261023.py /dev/shm/w/perf/rss/capture_serving_workloads_20261023.py \
     /dev/shm/w/perf/rss/audit_native_serving_pair_20261023.py "$R/"
  grep -q SGLANG_ROLE_SPLIT_FA3 "$R/native_fa3_fa2_pdl_20261023.py"
  grep -q role_split "$R/run_native_serving_pair_20261023.py"
}
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
COMMON="--split-policy geometry_v3 --direct-output --packed-query-limit 64 --repeats 2 --ours"
for spec in "c2 mix128 --concurrency 4 --sessions 8 --chat-rate 2 --chat-requests 480" "c1 mix128 --concurrency 4 --sessions 8 --chat-rate 1 --chat-requests 240"; do
  set -- $spec; name=$1; R=$BASE/$1; stage=$2; shift 2
  # ONLY=<long|short> reruns one stage; the harness refuses to overwrite a finished stage's arms.
  [ -z "${ONLY:-}" ] || [ "$ONLY" = "$name" ] || continue
  make_root "$R"
  export PYTHONPATH=$R/python:$R:/dev/shm/w/fi TRACE_DATASET_PATH=$R/tracelab_unique.jsonl.gz
  cd "$R"
  echo "##### $stage"
  [ "${AUDIT_ONLY:-0}" = 1 ] || /dev/shm/w/venv/bin/python -u run_native_serving_pair_20261023.py "$R" --stage $stage --modes four two "$@" $COMMON
  /dev/shm/w/venv/bin/python audit_native_serving_pair_20261023.py "$R" "$R/audit.json"
done
echo OURS_FULL_STACK_RATE_DONE
