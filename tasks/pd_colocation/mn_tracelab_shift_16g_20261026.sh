#!/bin/bash
# Prefill/decode demand shift on a real trace (Crossflow's protocol, made cluster-sized), full stack, 16 H20 GPUs
# (job 2f608c1b1a67cf09, 4 pods x 4 GPUs, Qwen2.5-7B TP1 = 16 workers), 2026-10-26.
#
# Agent tenant: TraceLab (claude + codex) timestamp replay. One trace hour holds only 3-6 concurrent sessions, so the 24
# busiest non-overlapping hours are superposed on one clock (WINDOW_STARTS; each segment timed from its own start, one
# developer per segment), "segment" rounds only, prompts <= 124K tokens, natural ShareGPT text. Each round goes out at
# max(previous finish + recorded gap / alpha, trace offset / alpha). The trace's own mix of cold long prefixes and short
# appends makes the prefill/decode demand ratio swing over the run; alpha (Crossflow's load factor) is the load axis.
# Chat tenant: ShareGPT with the chat template, Poisson 8 req/s (agents are > 70% of LLM calls).
# Points: alpha = 4 / 6 / 8 (15 / 10 / 7.5 min of trace), then at alpha 6 a chat surge of +16 req/s from 200 s for
# 200 s (CHAT_SPIKE): a decode-heavy shift on top of the trace's own swings.
# Arms: ours (coloc_slo_edf_mtp_hc, A6 + 4-mode MTP + EDF + SLO-checked cache-first routing + pool), conditional
# disaggregation at Dynamo's defaults on the same workers, stock PD 2 + 2 and 3 + 1 (stock MTP, pool).
# Score: tasks/perf_model/tracelab_windows.py --windows 30 (paper SLOs, per tenant; 30 s demand / attainment series).
# Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=4 GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1
export ATTN_BACKEND=fa3 POOL_KV_TOKENS=6696000 CHAT_SHAREGPT=1
export AGENT_MODE=tracelab_ts WINDOW_ROUNDS=segment WINDOW_S=3600 MAX_CTX=126976 MIN_CTX=0
export TRACE_DATASET_PATH=/dev/shm/w/workloads/tracelab/syfi_coding_trace.jsonl.gz
export WINDOW_STARTS=${WINDOW_STARTS:-$(python3 -c "import json; print(','.join(str(s) for s in json.load(open('$(dirname "$0")/tracelab_top_hours_20261026.json'))[:24]))")}
export PREEMPT=1 CHUNK=2048 HC_PREFETCH=timeout EDF_UNCACHED=1
CAL=/tmp/h20/mn_a6_calibrate_16g.log
RA=$(grep -oE "ROUTER_ARGS=.*" $CAL | tail -1 | cut -d= -f2-)
[ -n "$RA" ] || { echo NO_CALIBRATION; exit 1; }
RA_SLO="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5"
R=${R:-r1}
score() { python3 - "$1" <<'PY'
import sys; sys.path.insert(0, "/opt/tiger/sglang/tasks/pd_colocation")
from px_multi import Cluster
c = Cluster("2f608c1b1a67cf09", "cn-seed", nodes=["worker-0"])
print("SCORE", c.run(0, f"/dev/shm/w/venv/bin/python /dev/shm/w/perf/tracelab_windows.py --windows 30 /dev/shm/w/mn/{sys.argv[1]} 2>&1 | tail -2", timeout=300)[1].strip()[-900:], flush=True)
PY
}
arm() {  # arm <name> <tag>
  local name=$1 tag=$2
  case $name in
    coloc_slo_*) KERNEL=a6 EDF=1 RUN_TAG=$tag \
                   ROUTER_ARGS="$RA --pool-kv-tokens $POOL_KV_TOKENS --prefetch-policy timeout $RA_SLO" python3 -u mn_run.py $name ;;
    cond_*) KERNEL=a6 HC_WRITE=write_through COND_THRESHOLD=2048 COND_FRAC=0.7 RUN_TAG=$tag \
              ROUTER_ARGS="$RA --pool-kv-tokens $POOL_KV_TOKENS --prefetch-policy timeout --pool-write-policy write_through $RA_SLO" \
              python3 -u mn_run.py $name ;;
    disagg_*) (unset CHUNK; RUN_TAG=$tag SGL_PY=/dev/shm/w/up_clean/python KERNEL=lpt PREEMPT=0 python3 -u mn_run.py $name) ;;
  esac
  score $name$tag
}
point() {  # point <alpha> <tag suffix>
  export ALPHA=$1
  local dur=$(( 3600 / $1 ))
  export CHAT_RATE=8 CHAT_N=$(( 8 * dur ))
  for name in coloc_slo_edf_mtp_hc cond_p1_edf_mtp_hc disagg_p2_mtps_hc disagg_p3_mtps_hc; do
    arm $name _tl_a$1$2_$R
  done
}
for A in 6 4 8; do point $A ""; done
export CHAT_SPIKE=200:16:3200
point 6 _surge
unset CHAT_SPIKE
echo MN_TRACELAB_SHIFT_DONE
