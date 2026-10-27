#!/bin/bash
# Attribute the gain over STOCK COLOCATED sglang, at the best configuration (2026-10-28).
# mn_gapsource found stock upstream sglang colocated behind its own cache_aware router reaches 91.8% prefix hit
# against our 92.7%: the same prefill work. At that equal hit rate we still serve +25% agent goodput with -45% TPOT
# and -37% mean ITL. Four differences remain between the two rows; this removes them one at a time from OUR side,
# all on cache routing + FCFS workers (the best configuration), so nothing is masked by the SLO router or by chunk
# parking, both of which the leave-one-out showed to cost:
#   noC2   A6 kernel, STOCK MTP steps (_mtps) instead of the four-mode step -- the prime suspect for the TPOT gain
#   noC1   four-mode step, sglang's single varlen FA3 call instead of A6 (KERNEL=lpt on ATTN_BACKEND=fa3)
#   dchunk A6 + four-mode, upstream's default chunk instead of CHUNK=2048
#   sglrt  A6 + four-mode + CHUNK, sglang's own cache_aware router instead of ours (isolates the load tie-break)
# Two repeats of the reference arm are already measured (cacheFCFS 94.2% / 2471, cacheEDF 93.9% / 2506), and the
# noise floor on this workload is ~5 points of attainment and ~4% of goodput, so only gaps beyond that are read.
# Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=4 GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1
export ATTN_BACKEND=fa3 POOL_KV_TOKENS=6696000 CHAT_SHAREGPT=1
export AGENT_MODE=tracelab_ts WINDOW_ROUNDS=segment WINDOW_S=3600 MAX_CTX=126976 MIN_CTX=0
export TRACE_DATASET_PATH=/dev/shm/w/workloads/tracelab/syfi_coding_trace.jsonl.gz
export WINDOW_STARTS=$(python3 -c "import json; print(','.join(str(s) for s in json.load(open('$(dirname "$0")/tracelab_top_hours_20261026.json'))[:24]))")
export HC_PREFETCH=timeout EDF_UNCACHED=1 PREEMPT=0
CAL=/tmp/h20/mn_a6_calibrate_16g.log
RA=$(grep -oE "ROUTER_ARGS=.*" $CAL | tail -1 | cut -d= -f2-)
[ -n "$RA" ] || { echo NO_CALIBRATION; exit 1; }
ROUTER_ARGS_ALL="$RA --pool-kv-tokens $POOL_KV_TOKENS --prefetch-policy timeout --ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5"
H0=2f608c1b1a67cf09.worker_0.seedjob.wenxuan-tan@merlin-ssh-proxy-cn.byted.org
score() { ssh -o StrictHostKeyChecking=no -p 22222 $H0 "/dev/shm/w/venv/bin/python /dev/shm/w/perf/tracelab_windows.py --windows 30 /dev/shm/w/mn/$1 2>&1 | tail -1" 2>/dev/null | sed 's/^/SCORE /'; }
export ALPHA=3 CHAT_RATE=8 CHAT_N=$(( 8 * 1200 * 3 / 2 )) BENCH_REPLAY_HORIZON_S=1800
T=_tl2_a3_bdec_r1
export CHUNK=2048
KERNEL=a6 RUN_TAG=${T}_noC2 ROUTER_ARGS="$ROUTER_ARGS_ALL" python3 -u mn_run.py coloc_cache_mtps_hc
score coloc_cache_mtps_hc${T}_noC2
KERNEL=lpt RUN_TAG=${T}_noC1 ROUTER_ARGS="$ROUTER_ARGS_ALL" python3 -u mn_run.py coloc_cache_mtp_hc
score coloc_cache_mtp_hc${T}_noC1
(unset CHUNK; KERNEL=a6 RUN_TAG=${T}_dchunk ROUTER_ARGS="$ROUTER_ARGS_ALL" python3 -u mn_run.py coloc_cache_mtp_hc)
score coloc_cache_mtp_hc${T}_dchunk
KERNEL=a6 RUN_TAG=${T}_sglrt python3 -u mn_run.py coloc_sgl_mtp_hc
score coloc_sgl_mtp_hc${T}_sglrt
echo MN_BESTDECOMP_DONE
