#!/bin/bash
# Stock PD disaggregation, re-run on CLEAN upstream after the r1 attempt measured a broken transfer path.
#
# What r1 showed (log /tmp/h20/mn_pdre_r1.log): chat 4.9% with TTFT p50/p90 190/311 s, failed=1890,
# lost_after_failure=7470, accept=NONE. The decode pod logged 2 819 x "transfer failed for request" plus
# KVTransferError(bootstrap_room=...) aborts, while the prefill pod's Mooncake store set up and warmed fine and
# every worker carried rope_theta 1000000. So prefill computed the KV and could not hand it to decode: the arm
# measured a broken transfer, not a schedule, and its numbers are hugely favourable to us -- which is why they are
# void rather than merely noisy.
#
# The one change here is the tree. mn_node.sh's own header states that the stock PD baseline must run clean
# upstream ($W/up_clean/python) "because the serving overlay's spec-v2 changes trip disagg prefill's spec_info
# identity assert"; r1 passed SGL_PY=$W/up/python. This is the documented configuration and the cheapest
# hypothesis.
#
# BEFORE TRUSTING THIS RUN, check two things in the log, because a score alone cannot distinguish them:
#   - `accept=` must not be NONE. A failing transfer should still leave acceptance lines from whatever decoded, so
#     NONE means either nothing decoded or stock MTP is not running -- and an arm without MTP is not the baseline.
#   - `failed=` and `lost_after_failure=` must be 0. An arm with either is void whatever its attainment reads.
# If transfers still fail, the cause is mooncake connectivity on THIS job rather than the tree: check decode-to-
# prefill bootstrap port reachability (claimed_ports / ports_r*_g*.txt) and whether RDMA is available here, since
# the published disagg numbers came from a different job. Claude, 2026-10-28.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=8d89d1880003f89d MN_CP=cn-seed NPODS=4 GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1
export ATTN_BACKEND=fa3 POOL_KV_TOKENS=6696000 CHAT_SHAREGPT=1 ROPE_THETA=1000000
export AGENT_MODE=tracelab_ts WINDOW_ROUNDS=segment WINDOW_S=3600 MAX_CTX=126976 MIN_CTX=0
export TRACE_DATASET_PATH=/dev/shm/w/workloads/tracelab/syfi_coding_trace.jsonl.gz
export WINDOW_STARTS=$(python3 -c "import json; print(','.join(str(s) for s in json.load(open('$(dirname "$0")/tracelab_top_hours_20261026.json'))[:24]))")
export HC_PREFETCH=timeout EDF_UNCACHED=1 PREEMPT=0
MODEL=/dev/shm/w/perf/step_qwen_h20_a6_gpu_20261027.json
report() {  # report <arm dir>
  SID=$SID python3 - "$1" <<'PY'
import os, sys
sys.path.insert(0, ".")
from px_multi import Cluster
C = Cluster(os.environ["SID"], "cn-seed", nodes=[f"worker-{i}" for i in range(4)])
d = sys.argv[1]
rc, out = C.run(0, f"/dev/shm/w/venv/bin/python /dev/shm/w/perf/tracelab_windows.py /dev/shm/w/mn/{d} 2>&1 | tail -1")
print("SCORE " + (out.strip().splitlines()[-1] if out.strip() else "(none)"), flush=True)
rc, out = C.run(0, "cat /dev/shm/w/mn/" + d + "/worker_r0_g*.log | grep -o 'accept len: [0-9.]*' | "
                   "awk '{s+=$3;n++} END {if(n) printf \"accept=%.3f n=%d\\n\", s/n, n; else print \"accept=NONE\"}'")
print(f"ACCEPT {d} " + (out.strip().splitlines()[-1] if out.strip() else "?"), flush=True)
PY
}
point() { export ALPHA=$1; local dur=$(( 3600 / $1 ))
  export CHAT_RATE=8 CHAT_N=$(( 8 * dur * 3 / 2 )) BENCH_REPLAY_HORIZON_S=$(( dur * 3 / 2 )); }
CAL=/tmp/h20/mn_a6_calibrate_16g.log
RA=$(grep -oE "ROUTER_ARGS=.*" $CAL 2>/dev/null | tail -1 | cut -d= -f2-)
# --decode-grid points at /dev/shm/w/mn/decode_grid.json from the OLD cluster; Router.__init__ loads it
# unconditionally, so a missing file crashes the router. Our arm uses the plain `cache` policy, which routes on
# prefix hit and in-flight count only and never reads the perf model the grid feeds, so dropping it is neutral.
RA=$(echo "${RA:-}" | sed -E "s/--decode-grid [^ ]+//")
ROUTER_ARGS_ALL="${RA:-} --pool-kv-tokens $POOL_KV_TOKENS --prefetch-policy timeout"
for A in 3 6; do
  point $A
  T=_tl2_a${A}_pdre_r2
  # Stock PD disaggregation, sglang's PD router, upstream tree. The earlier disagg_p* numbers ran this tree with the
  # rope-base defect (theta 1e4 instead of 1e6); ROPE_THETA now reaches the override as a top-level field.
  (unset CHUNK SLO_CHUNK; RUN_TAG=${T}_disagg2 SGL_PY=/dev/shm/w/up_clean/python KERNEL=lpt PREEMPT=0 \
     python3 -u mn_run.py disagg_p2_mtps_hc); report disagg_p2_mtps_hc${T}_disagg2
  # Conditional disaggregation (Dynamo defaults) on our tree and our router -- the strongest disaggregated design.
  KERNEL=a6 HC_WRITE=write_through COND_THRESHOLD=2048 COND_FRAC=0.7 EDF=1 RUN_TAG=${T}_cond1 \
    ROUTER_ARGS="$ROUTER_ARGS_ALL --pool-write-policy write_through" python3 -u mn_run.py cond_p1_edf_mtp_hc
  report cond_p1_edf_mtp_hc${T}_cond1
done
echo MN_PDRE_DONE
