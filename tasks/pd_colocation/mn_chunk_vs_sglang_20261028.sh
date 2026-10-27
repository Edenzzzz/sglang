#!/bin/bash
# THE CORRECTED ARM SET (user, 2026-10-28): our coupled schedule against sglang's OWN schedule and chunk policy.
#
# Every earlier chunk comparison was invalid for one of two reasons the user identified:
#   - the baseline was given OUR chunk control (CHUNK=2048), so only the queue order differed and the TPOT result
#     was guaranteed to be null;
#   - our own arm used a FIXED 2048 chunk, which is indefensible: at the start of serving, or whenever prefill load
#     is high and few rows are decoding, there is nothing to protect and the cap throttles prefill for no reason,
#     starving the decode batch that would have formed.
# So: the baseline keeps sglang's prefill-prioritized / mixed-chunk schedule AND sglang's own chunk policy, and our
# node-level schedule is EDF COUPLED WITH the SLO-budgeted chunk -- one mechanism, never ablated against itself.
#
# The chunk rule (tasks/perf_model/chunk_predictor.py, validated offline on 10,185 merged steps, step-model MAPE
# 4.7%): budget = (1-h) * min_r [s_r*(o_r+a) - E_r]; chunk = largest c with T_step(c) <= budget, by bisection.
# Banking is UNLIMITED because the SLO is a request's MEAN TPOT; the bank=4 default that starved the earlier arm
# turned it into a per-step ITL bound. Offline the rule picks 192 tokens for a freshly admitted row and up to 16384
# once rows have banked slack, so it is adaptive in both directions rather than a cap.
#
# Arms, at alpha 3 (no regression expected) and alpha 6 (where TPOT attainment should improve):
#   sgl    baseline: upstream sglang tree, its own schedule and chunk policy, sglang's cache_aware router
#   ours   our tree: A6 kernel + four-mode step + EDF + SLO-budgeted chunk (coupled), our cache router
# Launch chunk is 16384 with max-prefill at least that, so the rule -- not a cap -- decides every step.
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
H0=""   # scoring runs on pod 0 through the Cluster helper, not ssh (the ssh entry point is not always pod 0)
score() {
  SID=$SID python3 - "$1" <<'PY'
import os, sys
sys.path.insert(0, ".")
from px_multi import Cluster
C = Cluster(os.environ["SID"], "cn-seed", nodes=[f"worker-{i}" for i in range(4)])
rc, out = C.run(0, f"/dev/shm/w/venv/bin/python /dev/shm/w/perf/tracelab_windows.py /dev/shm/w/mn/{sys.argv[1]} 2>&1 | tail -1")
print("SCORE " + (out.strip().splitlines()[-1] if out.strip() else "(no output)"))
PY
}
acc() {
  SID=$SID python3 - "$1" <<'PY'
import os, sys
sys.path.insert(0, ".")
from px_multi import Cluster
C = Cluster(os.environ["SID"], "cn-seed", nodes=[f"worker-{i}" for i in range(4)])
rc, out = C.run(0, "cat /dev/shm/w/mn/" + sys.argv[1] + "/worker_r0_g*.log | grep -o 'accept len: [0-9.]*' | "
                   "awk '{s+=$3;n++} END {if(n) printf \"accept=%.3f n=%d\\n\", s/n, n; else print \"accept=NONE\"}'")
print("ACCEPT " + sys.argv[1] + " " + (out.strip().splitlines()[-1] if out.strip() else "?"))
PY
}
point() { export ALPHA=$1; local dur=$(( 3600 / $1 ))
  export CHAT_RATE=8 CHAT_N=$(( 8 * dur * 3 / 2 )) BENCH_REPLAY_HORIZON_S=$(( dur * 3 / 2 )); }
CAL=/tmp/h20/mn_a6_calibrate_16g.log
RA=$(grep -oE "ROUTER_ARGS=.*" $CAL 2>/dev/null | tail -1 | cut -d= -f2-)
ROUTER_ARGS_ALL="${RA:-} --pool-kv-tokens $POOL_KV_TOKENS --prefetch-policy timeout"
for A in 3 6; do
  point $A
  T=_tl2_a${A}_chunkab_r1
  # Baseline: sglang's own schedule and chunk policy. CHUNK unset so the server uses its default; no EDF, no SLO rule.
  (unset CHUNK SLO_CHUNK; RUN_TAG=${T}_sgl SGL_PY=/dev/shm/w/up/python KERNEL=lpt \
     python3 -u mn_run.py coloc_sgl_mtps_hc)
  score coloc_sgl_mtps_hc${T}_sgl; acc coloc_sgl_mtps_hc${T}_sgl
  # Ours: EDF coupled with the SLO-budgeted chunk, launched at 16384 so the rule decides.
  KERNEL=a6 EDF=1 CHUNK=16384 SLO_CHUNK="$MODEL:0.1:50:80:24000:demote:ttft" \
    SGL_PY=/dev/shm/w/python_slo RUN_TAG=${T}_ours ROUTER_ARGS="$ROUTER_ARGS_ALL" \
    python3 -u mn_run.py coloc_cache_edf_mtp_hc
  score coloc_cache_edf_mtp_hc${T}_ours; acc coloc_cache_edf_mtp_hc${T}_ours
done
echo MN_CHUNK_AB_DONE
