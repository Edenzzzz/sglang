#!/bin/bash
# v6 (2026-10-27 14:3x): v5 at alpha 6 protected TPOT (chat p50 47 ms) but starved prefill (chat TTFT p50 74 s). This run
# adds the TTFT floor (:ttft, tree /dev/shm/w/python_slo2): the earliest-deadline prefill that can still meet its
# deadline gets at least the chunk that finishes it in time. v5 (2026-10-27 14:0x): the v4 alpha-6 arm made chat worse (27.0% vs 31.8%, TPOT p50 88.6 vs 72 ms): its tokens-per-step
# estimate spanned several steps (8.8 at acceptance ~2) and demotion dropped every late row, so budgets reached seconds
# or infinity and chunks 8-16K. The patch now takes tokens per step from the acceptance EWMA, keeps demoted rows at
# their per-step allowance and caps banked slack at 4 steps. v4 (2026-10-27 13:2x): steps 1-2 done (trace steptrace3, fit). The gap-target lin4mode fit was unusable (45% MAPE,
# c_tok < 0, a 53 ms constant on every mixed step), so the budget would flip between no prefill and huge chunks; the
# alpha-6 arm on it was stopped. This run uses the GPU-time fit only (25% in-sample MAPE, physically sensible
# coefficients): step_qwen_h20_a6_gpu_20261027.json, a copy of tasks/perf_model/step_qwen_h20_a6_gpu_20261027.json.
# v3 (2026-10-27 12:4x): the v2 tracer fix keyed on mixed_verify_rows, set only inside the worker; the tracer now reads
# the running rows from mixed_spec_running even without host seq_lens. v2 (2026-10-27 12:0x): the first run's step trace logged the 4-mode step's verify rows as prefill rows (nd = 0 on
# every merged step), so the fit refused it; apply_steptrace_verify_rows_20261027.py fixes the tracer on every pod's
# trees and this run re-traces. The stock PD rerun (step 0 of v1) is done.
# SLO-budgeted prefill chunk on the 16-H20 trace replay (job 2f608c1b1a67cf09, Qwen2.5-7B TP1 x 16), 2026-10-27.
# The demand-shift run (mn_tracelab_shift_v2_16g_20261026.sh) showed ours carrying more agent goodput than conditional
# disaggregation at alpha 6 but losing chat TPOT (p50 72 vs 28 ms): every colocated node runs fixed 2048-token agent
# chunks beside its chat decodes. This replaces the fixed chunk with the SLO-budgeted one
# (unified/apply_slo_budget_chunk_20261026.py, in the copied tree /dev/shm/w/python_slo on every pod):
#   1. ours at alpha 3 with STEP_TRACE=1 and the fixed 2048 chunk (also a second repeat of that arm);
#   2. fit the lin4mode step model on pod 0's four worker logs (perf_model/fit_4mode_step.py --export-lin4mode, S/T
#      = 3/4) and copy it to every pod;
#   3. ours with SLO_CHUNK (h = 0.1, chat 50 ms, agent 80 ms for prompts >= 24K tokens, demote rows already past their
#      SLO) and CHUNK=16384, at alpha 6, alpha 3 and the alpha-3 chat surge, against the v2 run's fixed-2048 arms.
# Same workload, horizon and scorer as v2. Claude.
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
MODEL=/dev/shm/w/perf/step_qwen_h20_a6_gpu_20261027.json
H0=2f608c1b1a67cf09.worker_0.seedjob.wenxuan-tan@merlin-ssh-proxy-cn.byted.org
SSHO="-o StrictHostKeyChecking=no"
score() { ssh $SSHO -p 22222 $H0 "/dev/shm/w/venv/bin/python /dev/shm/w/perf/tracelab_windows.py --windows 30 /dev/shm/w/mn/$1 2>&1 | tail -1" 2>/dev/null | sed 's/^/SCORE /'; }
ours() {  # ours <tag> [extra env assignments...]
  local tag=$1; shift
  env "$@" KERNEL=a6 EDF=1 RUN_TAG=$tag \
    ROUTER_ARGS="$RA --pool-kv-tokens $POOL_KV_TOKENS --prefetch-policy timeout $RA_SLO" \
    python3 -u mn_run.py coloc_slo_edf_mtp_hc
  score coloc_slo_edf_mtp_hc$tag
}
load() { export ALPHA=$1; local dur=$(( 3600 / $1 )); export CHAT_RATE=8 CHAT_N=$(( 8 * dur * 3 / 2 )) BENCH_REPLAY_HORIZON_S=$(( dur * 3 / 2 )); }
# 3. SLO-budgeted chunk
SLO="SGL_PY=/dev/shm/w/python_slo2 CHUNK=16384 SLO_CHUNK=$MODEL:0.1:50:80:24000:demote:ttft"
load 6; ours _tl2_a6_slochunkt_r1 $SLO
load 3; ours _tl2_a3_slochunkt_r1 $SLO
export CHAT_SPIKE=300:16:4800
load 3; ours _tl2_a3_surge_slochunkt_r1 $SLO
unset CHAT_SPIKE
echo MN_SLO_CHUNK_TTFT_DONE
