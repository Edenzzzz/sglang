#!/bin/bash
# Sweep the prefill/decode balance of a single-tenant workload to locate where
# the persistent overlap kernel pays off end to end.
#
# The kernel wins when a batch carries a large total decode KV while the
# prefill chunk's own prefix work is small — BatchPrefill then pads every
# 1-token decode row up to the batch-wide 128-row CTA tile and still streams
# that row's whole KV. Long outputs grow decode KV for free (generated tokens),
# and fresh arrivals supply zero-prefix chunks, so "thinking-heavy serving with
# steady new arrivals" should sit in the win region while prefill-heavy serving
# should not.
#
# Cases are input:output:rate. Ordered from decode-dominant to prefill-dominant.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

MODEL="${MODEL:-/opt/tiger/models/Qwen2.5-7B-Instruct}" \
# 4096:8192 grows each stream to ~12K KV, so a saturated batch holds a large
# decode pool while every arriving prompt is a fresh zero-prefix chunk — the
# measured win region. 8192:1024 is the prefill-dominant control that should
# show parity.
CASES_SPEC="${CASES_SPEC:-4096:8192:2 1024:4096:6 8192:1024:2}" \
NUM_PROMPTS="${NUM_PROMPTS:-100}" \
METHODS_SPEC="${METHODS_SPEC:-DMC_CG LPT}" \
PORT="${PORT:-30014}" \
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/h800_pd_balance}" \
SERVER_EXTRA_ARGS="--max-running-requests ${MAX_RUNNING:-64}" \
bash "$REPO_ROOT/tasks/debug_kernel_server_perf/run_gain_search.sh"
