#!/bin/bash
# Leave-one-contribution-out on the superposed TraceLab replay at alpha 3 (16 H20s, Qwen2.5-7B TP1 x 16), 2026-10-27.
# The full stack at this point is coloc_slo_edf_mtp_hc_tl2_a3_r1 (mn_tracelab_shift_v2_16g_20261026.sh). Each arm here
# removes exactly one contribution and keeps the rest:
#   -C1 unified FA3 + FA2 dispatcher: KERNEL=lpt on ATTN_BACKEND=fa3 -> sglang's single varlen FA3 call, same 4-mode
#       schedule, same deadline order, same router
#   -C2 four-mode step (verify rows in the prefill chunk's forward): the _mtps suffix -> stock MTP steps (upstream's
#       spec-aware MIXED step for decode tails), A6 kernel, deadline order, router unchanged
#   -C3 per-worker deadline order (uncached-token deadlines, chunk parking, budget reservation): no _edf suffix ->
#       sglang's FCFS admission on every worker
#   -C4 SLO-checked routing: router policy `cache` (highest prefix hit, load tie-break, no TTFT / TPOT check, no pull
#       pricing); the Mooncake pool stays on in every arm, since it is sglang's, not ours
# Same workload, horizon, chat stream and scorer as v2. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=4 GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1
export ATTN_BACKEND=fa3 POOL_KV_TOKENS=6696000 CHAT_SHAREGPT=1
export AGENT_MODE=tracelab_ts WINDOW_ROUNDS=segment WINDOW_S=3600 MAX_CTX=126976 MIN_CTX=0
export TRACE_DATASET_PATH=/dev/shm/w/workloads/tracelab/syfi_coding_trace.jsonl.gz
export WINDOW_STARTS=$(python3 -c "import json; print(','.join(str(s) for s in json.load(open('$(dirname "$0")/tracelab_top_hours_20261026.json'))[:24]))")
export PREEMPT=1 CHUNK=2048 HC_PREFETCH=timeout EDF_UNCACHED=1
CAL=/tmp/h20/mn_a6_calibrate_16g.log
RA=$(grep -oE "ROUTER_ARGS=.*" $CAL | tail -1 | cut -d= -f2-)
[ -n "$RA" ] || { echo NO_CALIBRATION; exit 1; }
RA_SLO="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5"
ROUTER_ARGS_ALL="$RA --pool-kv-tokens $POOL_KV_TOKENS --prefetch-policy timeout $RA_SLO"
H0=2f608c1b1a67cf09.worker_0.seedjob.wenxuan-tan@merlin-ssh-proxy-cn.byted.org
score() { ssh -o StrictHostKeyChecking=no -p 22222 $H0 "/dev/shm/w/venv/bin/python /dev/shm/w/perf/tracelab_windows.py --windows 30 /dev/shm/w/mn/$1 2>&1 | tail -1" 2>/dev/null | sed 's/^/SCORE /'; }
export ALPHA=3 CHAT_RATE=8 CHAT_N=$(( 8 * 1200 * 3 / 2 )) BENCH_REPLAY_HORIZON_S=1800
T=_tl2_a3_abl_r1
KERNEL=lpt EDF=1 RUN_TAG=${T}_noC1 ROUTER_ARGS="$ROUTER_ARGS_ALL" python3 -u mn_run.py coloc_slo_edf_mtp_hc
score coloc_slo_edf_mtp_hc${T}_noC1
KERNEL=a6 EDF=1 RUN_TAG=${T}_noC2 ROUTER_ARGS="$ROUTER_ARGS_ALL" python3 -u mn_run.py coloc_slo_edf_mtps_hc
score coloc_slo_edf_mtps_hc${T}_noC2
KERNEL=a6 RUN_TAG=${T}_noC3 ROUTER_ARGS="$ROUTER_ARGS_ALL" python3 -u mn_run.py coloc_slo_mtp_hc
score coloc_slo_mtp_hc${T}_noC3
KERNEL=a6 EDF=1 RUN_TAG=${T}_noC4 ROUTER_ARGS="$ROUTER_ARGS_ALL" python3 -u mn_run.py coloc_cache_edf_mtp_hc
score coloc_cache_edf_mtp_hc${T}_noC4
echo MN_ABLATION_CONTRIB_DONE
