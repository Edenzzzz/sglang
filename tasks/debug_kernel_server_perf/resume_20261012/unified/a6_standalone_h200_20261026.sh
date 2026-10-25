#!/bin/bash
# Paper workload (c), standalone sets on one H200, 2026-10-26: ShareGPT, GSM8K, MATH-500, Code-Feedback served alone
# (no agent tenant), our full single-node stack (A6 + EDF + 4-mode MTP, no chunk cap) vs stock sglang on FA3, four- and
# two-mode, two repeats, per-request arrays kept. Short prompts sit below the 4-mode context floor, so this is the
# no-regression case. Base protocol below.
# Stage 2 rerun, 2026-10-26: A6 only, WITHOUT the per-step deadline chunk cap (--deadline-chunk=; the cap doubled the
# mixed steps chat rides in on Llama-70B and cost throughput there), on the serving library with the tile-rounded FA2
# chunk (serving_np64_20261026, FA3_MIN=65 as on the pods). Same workloads and protocol as below.
# Original header: Stage 2 for A6 (H200 job 8574c727fab6ffbd, 2026-10-25): our full single-node stack (EDF with per-tenant deadlines,
# demote-on-miss, 4-mode MTP step) vs stock sglang on FA3, once with native v3 as the kernel and once with the unified
# dispatcher (SGLANG_A6_UNIFIED=1: one plan, stock FA3 at the plan's split, v3's FA2 body with the in-launch merge).
# Same workloads as ours_full_stack_v3_h200_20261024.sh (long 128-256K agent conc 4 / chat 4 req/s; short 32-64K agent
# conc 8 / chat 8 req/s), two opposite-order repeats, audited; the A6 arm must log a6_unified=1. Claude.
set -euo pipefail
SRC=/dev/shm/w/qwen25_v3_common_screen_h200_20261023
BASE=/opt/tiger/w/a6_standalone_h200_20261026
exec 9>/dev/shm/w/a2_kernel.lock
flock -n 9
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
mkdir -p "$BASE"
make_root() {
  local R=$1
  mkdir -p "$R"
  test -d "$R/python" || cp -a "$SRC/python" "$R/python"
  cp "$SRC"/*.py "$SRC"/*.sh "$SRC"/tracelab_unique.jsonl.gz "$R/"
  cp /dev/shm/w/ub/native_fa3_fa2_pdl_20261023.py /dev/shm/w/ub/run_native_serving_pair_20261023.py \
     /dev/shm/w/ub/native_serving_protocol_20261023.py /dev/shm/w/ub/audit_native_serving_pair_20261023.py \
     /dev/shm/w/ub/unified_executor_20261024.py "$R/"
  /dev/shm/w/venv/bin/python /dev/shm/w/perf/rss/apply_edf_late_demote_20261024.py "$R"
  grep -q SGLANG_EDF_LATE_DEMOTE "$R/python/sglang/srt/managers/scheduler.py"
  grep -q prepare_unified "$R/native_fa3_fa2_pdl_20261023.py"
  grep -q native_env "$R/run_native_serving_pair_20261023.py"
  grep -q -- "--standalone" "$R/run_native_serving_pair_20261023.py"
  /dev/shm/w/venv/bin/python /dev/shm/w/ub/apply_bench_pair_standalone_20261026.py "$R" | grep -q "STANDALONE_BRANCH True"
}
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export NATIVE_PDL_FA2_LIBRARY=/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so
export NATIVE_PDL_FA3_LIBRARY=/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
export UNIFIED_SO=/opt/tiger/w/torch_ext_unified_serving_np64_20261026/unified_dispatch_ext/unified_dispatch_ext.so
export UNIFIED_FA3_STOCK=1 UNIFIED_SPLIT_POLICY=geometry_fa3 UNIFIED_MERGE_ROWS=4 UNIFIED_FA3_MIN=65
COMMON="--split-policy geometry_v3 --direct-output --packed-query-limit 64 --repeats 2 --ours --deadline-chunk="
FIX="--ours-env SGLANG_TTFT_SLO_MS=500 --ours-env SGLANG_EDF_LONG_PROMPT_TOKENS=8192 --ours-env SGLANG_EDF_LONG_TTFT_SLO_MS=10000 --ours-env SGLANG_EDF_LATE_DEMOTE=1 --ours-env SGLANG_EDF_LATE_DEMOTE_TOK_S=1e12"
T=/dev/shm/w/workloads
for spec in "sharegpt_r16 $T/ShareGPT_V3_unfiltered_cleaned_split.json 16 1920" "sharegpt_r32 $T/ShareGPT_V3_unfiltered_cleaned_split.json 32 3840" \
            "math500_r16 $T/standalone/math500.json 16 400" "gsm8k_r32 $T/standalone/gsm8k.json 32 1200" \
            "codefeedback_r16 $T/standalone/codefeedback.json 16 1920"; do
  set -- $spec; name=$1; path=$2; rate=$3; n=$4
  [ -z "${ONLY:-}" ] || [ "$ONLY" = "$name" ] || continue
  R=$BASE/$name
  make_root "$R"
  export PYTHONPATH=$R/python:$R:/dev/shm/w/fi:/dev/shm/w/drivers/serving TRACE_DATASET_PATH=$R/tracelab_unique.jsonl.gz
  cd "$R"
  echo "##### $name rate=$rate n=$n"
  /dev/shm/w/venv/bin/python -u run_native_serving_pair_20261023.py "$R" --stage mix32 --modes four two --standalone $path \
    --chat-rate $rate --chat-requests $n $COMMON $FIX --native-env SGLANG_A6_UNIFIED=1
  n6=$(cat $R/*_native/server.log 2>/dev/null | grep -c "a6_unified=1" || true)
  echo "a6 composition: $n6"; [ "$n6" -gt 0 ] || { echo A6_NOT_ACTIVE; exit 1; }
done
echo A6_STANDALONE_DONE
