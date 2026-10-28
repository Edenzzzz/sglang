#!/bin/bash
# THE SUPERSET CLAIM (user, 2026-10-28): our SLO control + perf model should DOMINATE every baseline everywhere on
# the curve, because with a latency model and per-request slack we can reproduce any baseline's behaviour and then
# improve on it. Anywhere a baseline wins is a bug to investigate, not a result to report.
#
# Two things this driver fixes about every earlier comparison:
#   - the chunk-2048 + FCFS arm was NOT a valid baseline (it was our chunk control with sglang's ordering);
#   - our own arm must never use a fixed chunk.
# Baselines are sglang's OWN policies, each with sglang's OWN chunk policy (CHUNK and SLO_CHUNK unset):
#   pp     default prefill-prioritized chunked prefill (MIXED_CHUNK=0)
#   mc     mixed chunk (MIXED_CHUNK=1) -- sglang's colocation default
#   pdmux  PD multiplexing (PDMUX=1): decode and layer-split prefill on separate SM groups, prefill unchunked,
#          overlap scheduling OFF, so its throughput is expected ~10% lower (user's prior)
# Ours:
#   ours   A6 kernel + four-mode step + EDF ordering COUPLED with the SLO-budgeted chunk (launched at 16384 so the
#          rule, not a cap, sizes every step), our cache router with the load tie-break
#
# Swept over load so the result is a curve, not a point: alpha 3 (capacity) and alpha 6 (overload). The superset
# property is read per metric per load: our arm must be >= every baseline on TPOT attainment and SLO goodput, and
# must not regress at alpha 3. Acceptance is logged per arm, because a baseline running a mis-configured model
# (the rope_theta defect of 2026-10-28) silently inflates our margin.
# Claude.
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
  T=_tl2_a${A}_superset_r4
  # --- baselines: sglang's own policies, sglang's own chunk policy, upstream tree
  (unset CHUNK SLO_CHUNK; MIXED_CHUNK=0 RUN_TAG=${T}_pp SGL_PY=/dev/shm/w/up/python KERNEL=lpt \
     python3 -u mn_run.py coloc_sgl_mtps_hc); report coloc_sgl_mtps_hc${T}_pp
  (unset CHUNK SLO_CHUNK; MIXED_CHUNK=1 RUN_TAG=${T}_mc SGL_PY=/dev/shm/w/up/python KERNEL=lpt \
     python3 -u mn_run.py coloc_sgl_mtps_hc); report coloc_sgl_mtps_hc${T}_mc
  (unset CHUNK SLO_CHUNK; PDMUX=1 MEM_FRAC=0.70 RUN_TAG=${T}_pdmux SGL_PY=/dev/shm/w/up/python KERNEL=lpt \
     python3 -u mn_run.py coloc_sgl_mtps_hc_pdmux); report coloc_sgl_mtps_hc_pdmux${T}_pdmux
  # --- ours: EDF coupled with the SLO-budgeted chunk, A6 kernel, four-mode step, cache router
  KERNEL=a6 EDF=1 CHUNK=16384 SLO_CHUNK="$MODEL:0.1:50:80:24000:demote:ttft" \
    SGL_PY=/dev/shm/w/python_slo RUN_TAG=${T}_ours ROUTER_ARGS="$ROUTER_ARGS_ALL" \
    python3 -u mn_run.py coloc_cache_edf_mtp_hc
  report coloc_cache_edf_mtp_hc${T}_ours
done
echo MN_SUPERSET_DONE
