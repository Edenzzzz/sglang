#!/bin/bash
# Where does the colocated stack's lead over disaggregation come from? (2026-10-27)
# Best-vs-best at alpha 3 (mn_bestcfg_16g_20261027.sh): three colocated arms (cache routing with FCFS, EDF, EDF +
# demote flag) all reach chat 90.0-91.3% / agent 93.9-94.2% / 2453-2506 agent tok/s, against conditional
# disaggregation's 54.6% / 65.2% / 1031. The worker policy does not move the result and C1 is ~4% of step time, so the
# lead must come from the layout (colocation), the routing (session-cache affinity), or both. This separates them:
#   stock    upstream sglang, colocated, sglang_router cache_aware, stock FA3 call, stock MTP, FCFS, default chunk:
#            every contribution of ours removed while staying colocated -- "is it just colocation?"
#   rr       our full stack except routing: round-robin, so a session's next round usually lands without its cache
#   cond2    conditional disaggregation with two prefill pods: the P:D ratio sweep for the disaggregated baseline
# and repeats the load-spike point (alpha 6) with the best-configuration stacks, since that is where the worker policy
# should matter (it was last measured there with the SLO router and chunk parking, both since dropped).
# Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=4 GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1
export ATTN_BACKEND=fa3 POOL_KV_TOKENS=6696000 CHAT_SHAREGPT=1
export AGENT_MODE=tracelab_ts WINDOW_ROUNDS=segment WINDOW_S=3600 MAX_CTX=126976 MIN_CTX=0
export TRACE_DATASET_PATH=/dev/shm/w/workloads/tracelab/syfi_coding_trace.jsonl.gz
export WINDOW_STARTS=$(python3 -c "import json; print(','.join(str(s) for s in json.load(open('$(dirname "$0")/tracelab_top_hours_20261026.json'))[:24]))")
export CHUNK=2048 HC_PREFETCH=timeout EDF_UNCACHED=1 PREEMPT=0
CAL=/tmp/h20/mn_a6_calibrate_16g.log
RA=$(grep -oE "ROUTER_ARGS=.*" $CAL | tail -1 | cut -d= -f2-)
[ -n "$RA" ] || { echo NO_CALIBRATION; exit 1; }
RA_SLO="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5"
ROUTER_ARGS_ALL="$RA --pool-kv-tokens $POOL_KV_TOKENS --prefetch-policy timeout $RA_SLO"
H0=2f608c1b1a67cf09.worker_0.seedjob.wenxuan-tan@merlin-ssh-proxy-cn.byted.org
score() { ssh -o StrictHostKeyChecking=no -p 22222 $H0 "/dev/shm/w/venv/bin/python /dev/shm/w/perf/tracelab_windows.py --windows 30 /dev/shm/w/mn/$1 2>&1 | tail -1" 2>/dev/null | sed 's/^/SCORE /'; }
point() {  # point <alpha>
  export ALPHA=$1
  local dur=$(( 3600 / $1 ))
  export CHAT_RATE=8 CHAT_N=$(( 8 * dur * 3 / 2 )) BENCH_REPLAY_HORIZON_S=$(( dur * 3 / 2 ))
}
arm() {  # arm <kind> <alpha>
  local T=_tl2_a$2_gap_r1
  point $2
  case $1 in
    stock) (unset CHUNK; RUN_TAG=${T}_stock SGL_PY=/dev/shm/w/up_clean/python KERNEL=lpt \
             python3 -u mn_run.py coloc_sgl_mtps_hc); score coloc_sgl_mtps_hc${T}_stock ;;
    rr)    KERNEL=a6 RUN_TAG=${T}_rr ROUTER_ARGS="$ROUTER_ARGS_ALL" python3 -u mn_run.py coloc_rr_mtp_hc
           score coloc_rr_mtp_hc${T}_rr ;;
    fcfs)  KERNEL=a6 RUN_TAG=${T}_cacheFCFS ROUTER_ARGS="$ROUTER_ARGS_ALL" python3 -u mn_run.py coloc_cache_mtp_hc
           score coloc_cache_mtp_hc${T}_cacheFCFS ;;
    edf)   KERNEL=a6 EDF=1 RUN_TAG=${T}_cacheEDF ROUTER_ARGS="$ROUTER_ARGS_ALL" python3 -u mn_run.py coloc_cache_edf_mtp_hc
           score coloc_cache_edf_mtp_hc${T}_cacheEDF ;;
    cond1) KERNEL=a6 HC_WRITE=write_through COND_THRESHOLD=2048 COND_FRAC=0.7 EDF=1 RUN_TAG=${T}_cond1 \
             ROUTER_ARGS="$ROUTER_ARGS_ALL --pool-write-policy write_through" python3 -u mn_run.py cond_p1_edf_mtp_hc
           score cond_p1_edf_mtp_hc${T}_cond1 ;;
    cond2) KERNEL=a6 HC_WRITE=write_through COND_THRESHOLD=2048 COND_FRAC=0.7 EDF=1 RUN_TAG=${T}_cond2 \
             ROUTER_ARGS="$ROUTER_ARGS_ALL --pool-write-policy write_through" python3 -u mn_run.py cond_p2_edf_mtp_hc
           score cond_p2_edf_mtp_hc${T}_cond2 ;;
  esac
}
arm stock 3
arm rr 3
arm fcfs 6
arm edf 6
arm stock 6
arm cond1 6
arm cond2 3
echo MN_GAPSOURCE_DONE
