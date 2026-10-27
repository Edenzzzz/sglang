#!/bin/bash
# Best-configuration comparison on the superposed TraceLab replay at alpha 3 (16 H20s, Qwen2.5-7B TP1 x 16),
# 2026-10-27. Arms 3 and 4 of the leave-one-out each won by removing one half of C3, so the configuration we have
# been reporting as "ours" is not our best. This measures the combination neither arm covered (cache routing AND
# FCFS workers), repeats the single best measured arm, and re-runs conditional disaggregation in the same session so
# the comparison is best-vs-best rather than our-full-stack-vs-their-default. Conditional disaggregation keeps
# Dynamo's published defaults (2048 tokens / 0.7 of the prompt) and never used our SLO check, so it was not
# handicapped by it.
# Removing C3's worker level wholesale WON (chat 73.9 vs 65.1-68.5%, agent goodput 1922 vs 1501-1763), so this
# separates the mechanisms it bundles. On H20 the full stack runs EDF ordering + chunk preemption/parking +
# budget reservation; demotion was never enabled here (mn_node.sh had no LATE_DEMOTE until today), although it is
# the mechanism with the single-node evidence. Arms:
#   ord    EDF ordering only (PREEMPT=0): isolates preemption/parking, the prime suspect
#   demote EDF ordering + demotion, still no preemption: the single-node mechanism, first multi-node test
#   noC1b  second repeat of the without-C1 arm, to tighten the 7.5-21% goodput range from r1
# The full stack at this point is coloc_slo_edf_mtp_hc_tl2_a3_r1 (mn_tracelab_shift_v2_16g_20261026.sh). Each arm here
# removes exactly one contribution and keeps the rest:
#   -C1 unified FA3 + FA2 dispatcher (traced, STEP_TRACE=1, for the fused-vs-unfused step-model comparison on the same
#       schedule): KERNEL=lpt on ATTN_BACKEND=fa3 -> sglang's single varlen FA3 call, same 4-mode
#       schedule, same deadline order, same router
#   -C2 four-mode step (verify rows in the prefill chunk's forward): the _mtps suffix -> stock MTP steps (upstream's
#       spec-aware MIXED step for decode tails), A6 kernel, deadline order, router unchanged
#   -C3 worker level (uncached-token deadlines, chunk parking, budget reservation): no _edf suffix -> sglang's FCFS
#       admission on every worker
#   -C3 router level (SLO-checked routing): router policy `cache` (highest prefix hit, load tie-break, no TTFT / TPOT
#       check, no pull pricing); the Mooncake pool stays on in every arm, since it is sglang's, not ours
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
T=_tl2_a3_best_r1
# cache routing + FCFS workers + A6 + 4-mode: the combination no ablation arm covered.
KERNEL=a6 RUN_TAG=${T}_cacheFCFS ROUTER_ARGS="$ROUTER_ARGS_ALL" python3 -u mn_run.py coloc_cache_mtp_hc
score coloc_cache_mtp_hc${T}_cacheFCFS
# repeat of the best measured arm (cache routing + EDF workers), for confidence on 91.4% / 2514 tok/s.
KERNEL=a6 EDF=1 RUN_TAG=${T}_cacheEDF ROUTER_ARGS="$ROUTER_ARGS_ALL" python3 -u mn_run.py coloc_cache_edf_mtp_hc
score coloc_cache_edf_mtp_hc${T}_cacheEDF
# conditional disaggregation, same session, Dynamo defaults.
KERNEL=a6 HC_WRITE=write_through COND_THRESHOLD=2048 COND_FRAC=0.7 EDF=1 RUN_TAG=${T}_cond \
  ROUTER_ARGS="$ROUTER_ARGS_ALL --pool-write-policy write_through" python3 -u mn_run.py cond_p1_edf_mtp_hc
score cond_p1_edf_mtp_hc${T}_cond
echo MN_BESTCFG_DONE
