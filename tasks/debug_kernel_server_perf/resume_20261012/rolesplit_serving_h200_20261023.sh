#!/bin/bash
# Stage 2 with the stage-1 kernel (H200 34c7bd503d461dc1, 2026-10-23). The trace-proportional benchmark found the
# mixed step is best run as TWO FA3 launches (prefill rows + decode/verify rows with num_splits 32): 1.25-1.39x faster
# than sglang's single varlen FA3 call and within 1% of a per-step oracle. Here that kernel is the candidate arm in the
# same paired protocol as the audited native v3 run: same frozen tree and scheduler, FA3 control vs candidate, opposite
# arm orders, two repeats, two- and four-mode (trained EAGLE3), audited from raw token arrivals. Only the attention
# differs. 128-256K (where the kernel gain is largest) first, then 32-64K. Claude.
set -euo pipefail
SRC=/dev/shm/w/qwen25_v3_common_screen_h200_20261023
BASE=/opt/tiger/w/rolesplit_serving_h200_20261023
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
COMMON="--split-policy geometry_v3 --direct-output --packed-query-limit 64 --role-split --repeats 2"
for spec in "long mix128 --concurrency 4 --sessions 8 --chat-rate 4 --chat-requests 960" "short mix32"; do
  set -- $spec; R=$BASE/$1; stage=$2; shift 2
  make_root "$R"
  export PYTHONPATH=$R/python:$R:/dev/shm/w/fi TRACE_DATASET_PATH=$R/tracelab_unique.jsonl.gz
  cd "$R"
  echo "##### $stage"
  /dev/shm/w/venv/bin/python -u run_native_serving_pair_20261023.py "$R" --stage $stage --modes four two "$@" $COMMON
  /dev/shm/w/venv/bin/python audit_native_serving_pair_20261023.py "$R" "$R/audit.json"
done
echo ROLESPLIT_SERVING_DONE
