#!/bin/bash
# Re-run the stock colocated baseline with a CORRECT RoPE base (2026-10-28).
# The stock arm runs upstream sglang (up_clean), which lacks our fix in utils/hf_transformers/config.py: when
# --json-model-override-args replaces `rope_scaling` with a scaling-only dict, upstream drops the model's rope_theta
# and get_rope_config falls back to 10000, while Qwen2.5-7B needs 1e6. mn_node.sh always passes that YaRN override
# (YARN_FACTOR=4.0), so the stock baseline ran a mis-configured model: mean draft acceptance 1.420 against 1.991 in
# our arms, which alone accounts for most of the +25% throughput we reported over it. That comparison is INVALID.
# The fix needs no code change in upstream: carry rope_theta explicitly inside the override dict, so both trees build
# the same RoPE. ROPE_THETA is read by mn_node.sh (added today) and appended to the override JSON.
# Arms: the stock baseline re-run with the correct base, and -- as a control that the fix works -- our own arm
# unchanged (already correct, expected to reproduce 2471 tok/s and acceptance ~1.99).
# Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=4 GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1
export ATTN_BACKEND=fa3 POOL_KV_TOKENS=6696000 CHAT_SHAREGPT=1
export AGENT_MODE=tracelab_ts WINDOW_ROUNDS=segment WINDOW_S=3600 MAX_CTX=126976 MIN_CTX=0
export TRACE_DATASET_PATH=/dev/shm/w/workloads/tracelab/syfi_coding_trace.jsonl.gz
export WINDOW_STARTS=$(python3 -c "import json; print(','.join(str(s) for s in json.load(open('$(dirname "$0")/tracelab_top_hours_20261026.json'))[:24]))")
export HC_PREFETCH=timeout EDF_UNCACHED=1 PREEMPT=0 ROPE_THETA=1000000
CAL=/tmp/h20/mn_a6_calibrate_16g.log
RA=$(grep -oE "ROUTER_ARGS=.*" $CAL | tail -1 | cut -d= -f2-)
[ -n "$RA" ] || { echo NO_CALIBRATION; exit 1; }
ROUTER_ARGS_ALL="$RA --pool-kv-tokens $POOL_KV_TOKENS --prefetch-policy timeout --ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5"
H0=2f608c1b1a67cf09.worker_0.seedjob.wenxuan-tan@merlin-ssh-proxy-cn.byted.org
score() { ssh -o StrictHostKeyChecking=no -p 22222 $H0 "/dev/shm/w/venv/bin/python /dev/shm/w/perf/tracelab_windows.py --windows 30 /dev/shm/w/mn/$1 2>&1 | tail -1" 2>/dev/null | sed 's/^/SCORE /'; }
acc() { ssh -o StrictHostKeyChecking=no -p 22222 $H0 "cat /dev/shm/w/mn/$1/worker_r0_g*.log | grep -o 'accept len: [0-9.]*' | awk '{s+=\$3;n++} END {printf \"ACCEPT %s mean=%.3f n=%d\n\", \"$1\", s/n, n}'" 2>/dev/null; }
export ALPHA=3 CHAT_RATE=8 CHAT_N=$(( 8 * 1200 * 3 / 2 )) BENCH_REPLAY_HORIZON_S=1800
T=_tl2_a3_ropefix_r1
(unset CHUNK; RUN_TAG=${T}_stock SGL_PY=/dev/shm/w/up_clean/python KERNEL=lpt python3 -u mn_run.py coloc_sgl_mtps_hc)
score coloc_sgl_mtps_hc${T}_stock; acc coloc_sgl_mtps_hc${T}_stock
export CHUNK=2048
KERNEL=a6 RUN_TAG=${T}_ours ROUTER_ARGS="$ROUTER_ARGS_ALL" python3 -u mn_run.py coloc_cache_mtp_hc
score coloc_cache_mtp_hc${T}_ours; acc coloc_cache_mtp_hc${T}_ours
echo MN_STOCKFIX_DONE
