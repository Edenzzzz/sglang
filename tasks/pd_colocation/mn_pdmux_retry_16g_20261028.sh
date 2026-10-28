#!/bin/bash
# THE OVERLOAD POINT (load factor alpha = 6), which no run on this cluster has yet reached.
#
# Why it is the missing measurement. The user's goal for the node-level schedule has two halves: no regression
# against arrival order at low load, and better TPOT attainment under high load. Only the first half is measured:
# at alpha = 3 the decomposition (mn_decomp_v2_16g_20261028.sh) showed our tree + A6 kernel + four-mode step +
# router (rung L0) carrying the entire gain over sglang's prefill-prioritized schedule, with deadline ordering
# (rung L1) neutral inside the noise floor. Neutral is the right outcome at capacity -- ordering by deadline can
# only pay when a deadline is in danger -- so deadline ordering has to earn its place here or not at all. Every
# earlier driver declared `for A in 3 6` and was stopped, killed or re-scoped before it left alpha 3.
#
# Arms, all on the same cluster, workload and superposed window set, differing only in the layer under test:
#   L0     our serving tree (H20 lineage + A6 unified dispatcher + four-mode MTP step) and our cache router, with
#          sglang's FCFS admission and sglang's OWN default chunk policy
#   L1     L0 + per-tenant deadline ordering (EDF), still sglang's own chunk policy
#   pp     sglang's default prefill-prioritized chunked prefill, upstream tree, sglang's chunk policy
#   mc     sglang's mixed chunk, upstream tree, sglang's chunk policy
#   pdmux  sglang's PD multiplexing: prefill unchunked on its own SM group, overlap scheduling off
# No arm gets a fixed chunk: ours never may (user), and a baseline must never be given our chunk control.
# Ours runs first at each load so that a failure of our tree surfaces in ~20 min instead of after the baselines.
#
# Acceptance is reported per arm. A baseline whose draft acceptance is far below ours is running a different
# (mis-configured) model and its numbers inflate our margin rather than beating it -- the rope_theta defect of
# 2026-10-28 produced 1.42 against a clean 1.99. Read the result with superset_report.py.
# Claude, 2026-10-28.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=8d89d1880003f89d MN_CP=cn-seed NPODS=4 GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1
export ATTN_BACKEND=fa3 POOL_KV_TOKENS=6696000 CHAT_SHAREGPT=1 ROPE_THETA=1000000
export AGENT_MODE=tracelab_ts WINDOW_ROUNDS=segment WINDOW_S=3600 MAX_CTX=126976 MIN_CTX=0
export TRACE_DATASET_PATH=/dev/shm/w/workloads/tracelab/syfi_coding_trace.jsonl.gz
export WINDOW_STARTS=$(python3 -c "import json; print(','.join(str(s) for s in json.load(open('$(dirname "$0")/tracelab_top_hours_20261026.json'))[:24]))")
export HC_PREFETCH=timeout EDF_UNCACHED=1 PREEMPT=0
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
# One superposed hour of trace replayed at alpha x real speed, so a larger alpha is a shorter, denser run.
point() { export ALPHA=$1; local dur=$(( 3600 / $1 ))
  export CHAT_RATE=8 CHAT_N=$(( 8 * dur * 3 / 2 )) BENCH_REPLAY_HORIZON_S=$(( dur * 3 / 2 )); }
CAL=/tmp/h20/mn_a6_calibrate_16g.log
RA=$(grep -oE "ROUTER_ARGS=.*" $CAL 2>/dev/null | tail -1 | cut -d= -f2-)
# --decode-grid names a file from the old cluster that Router.__init__ loads unconditionally; the `cache` policy
# our arm uses never reads the perf model it feeds, so dropping it is neutral and avoids a router crash.
RA=$(echo "${RA:-}" | sed -E "s/--decode-grid [^ ]+//")
ROUTER_ARGS_ALL="${RA:-} --pool-kv-tokens $POOL_KV_TOKENS --prefetch-policy timeout"
for A in 3 6; do
  point $A
  T=_tl2_a${A}_pdmux_r2
  # PD multiplexing, re-run after the alpha-6 arm of mn_overload_decomp failed with no score and no acceptance
  # lines (workers started, nothing served). The suspected cause is a capture-time OOM: pdmux captures CUDA graphs
  # PER SM GROUP, so graph memory multiplies, and MEM_FRAC=0.70 was carried over from the 8-GPU driver. 0.55 leaves
  # room for the extra graph sets. If this still fails, read worker_r0_g*.log for OutOfMemoryError and the
  # worker_*.bindfail*.log files before lowering further -- the signature is identical for a bind failure.
  #
  # Why this arm matters most (user, 2026-10-28): pdmux has the same SLO control as ours but no kernel and no
  # overlap schedule. It gives decode and layer-split prefill their own SM groups, so an arriving prefill cannot
  # block a running decode row -- the latency protection our chunk rule pursues by sizing, attained structurally.
  # It is therefore the serving-level leave-one-out for C1 + C2. Registered expectation: close to us on tail TPOT,
  # behind on throughput (no overlap scheduling, and partitioning costs occupancy on both sides). If it matches us
  # on BOTH, C1 and C2 do not survive at the serving level.
  (unset CHUNK SLO_CHUNK; PDMUX=1 MEM_FRAC=0.55 RUN_TAG=${T}_pdmux SGL_PY=/dev/shm/w/up/python KERNEL=lpt \
     python3 -u mn_run.py coloc_sgl_mtps_hc_pdmux); report coloc_sgl_mtps_hc_pdmux${T}_pdmux
done
echo MN_PDMUX_DONE
