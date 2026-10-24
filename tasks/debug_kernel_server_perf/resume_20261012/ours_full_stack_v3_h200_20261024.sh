#!/bin/bash
# Definitive full single-node stack, v3 (H200 34c7bd503d461dc1, 2026-10-24): as v2 (ours_full_stack_v2_h200_20261024.sh)
# but with demote-on-miss (SGLANG_EDF_LATE_DEMOTE_TOK_S=1e12: a prefill is demoted only once its deadline has passed),
# which the edf_demote_on_miss_h200_20261024.sh screen found better than v2's 6000 tok/s feasibility test at 128-256K.
# The load axis at 128-256K is AGENT concurrency (agents alone saturate one GPU there; chat rate barely moved
# attainment in v2). Stages, two opposite-order repeats each:
#   long  128-256K, agent conc 4 / 8 sessions, chat 4 req/s x 960
#   short 32-64K,   agent conc 8 / 16 sessions, chat 8 req/s x 960
#   a2/a1 128-256K, agent conc 2 / 1, 8 sessions, chat 2 req/s x 480 (capacity on agent concurrency)
# Claude.
set -euo pipefail
[ "${1:-now}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
SRC=/dev/shm/w/qwen25_v3_common_screen_h200_20261023
BASE=/opt/tiger/w/ours_full_stack_v3_h200_20261024
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
mkdir -p "$BASE"
echo $$ > "$BASE/controller.pid"
trap 'printf "%s\n" "$?" > "$BASE/exit_status"' EXIT
make_root() {
  local R=$1
  mkdir -p "$R"
  test -d "$R/python" || cp -a "$SRC/python" "$R/python"
  cp "$SRC"/*.py "$SRC"/*.sh "$SRC"/tracelab_unique.jsonl.gz "$R/"
  cp /dev/shm/w/perf/rss/native_fa3_fa2_pdl_20261023.py /dev/shm/w/perf/rss/run_native_serving_pair_20261023.py /dev/shm/w/perf/rss/native_serving_protocol_20261023.py /dev/shm/w/perf/rss/capture_serving_workloads_20261023.py \
     /dev/shm/w/perf/rss/audit_native_serving_pair_20261023.py "$R/"
  /dev/shm/w/venv/bin/python /dev/shm/w/perf/rss/apply_edf_late_demote_20261024.py "$R"
  grep -q SGLANG_EDF_LATE_DEMOTE "$R/python/sglang/srt/managers/scheduler.py"
  grep -q ours_env "$R/run_native_serving_pair_20261023.py"
}
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
COMMON="--split-policy geometry_v3 --direct-output --packed-query-limit 64 --repeats 2 --ours"
FIX="--ours-env SGLANG_TTFT_SLO_MS=500 --ours-env SGLANG_EDF_LONG_PROMPT_TOKENS=8192 --ours-env SGLANG_EDF_LONG_TTFT_SLO_MS=10000 --ours-env SGLANG_EDF_LATE_DEMOTE=1 --ours-env SGLANG_EDF_LATE_DEMOTE_TOK_S=1e12"
for spec in "long mix128 --concurrency 4 --sessions 8 --chat-rate 4 --chat-requests 960" "short mix32" \
            "a2 mix128 --concurrency 2 --sessions 8 --chat-rate 2 --chat-requests 480" \
            "a1 mix128 --concurrency 1 --sessions 8 --chat-rate 2 --chat-requests 480"; do
  set -- $spec; name=$1; R=$BASE/$1; stage=$2; shift 2
  [ -z "${ONLY:-}" ] || [ "$ONLY" = "$name" ] || continue
  make_root "$R"
  export PYTHONPATH=$R/python:$R:/dev/shm/w/fi TRACE_DATASET_PATH=$R/tracelab_unique.jsonl.gz
  cd "$R"
  echo "##### $name $stage"
  /dev/shm/w/venv/bin/python -u run_native_serving_pair_20261023.py "$R" --stage $stage --modes four two "$@" $COMMON $FIX
  /dev/shm/w/venv/bin/python audit_native_serving_pair_20261023.py "$R" "$R/audit.json"
  /dev/shm/w/venv/bin/python /dev/shm/w/perf/bench_per_tenant.py ${stage}_*/requests
done
echo OURS_FULL_STACK_V3_DONE
