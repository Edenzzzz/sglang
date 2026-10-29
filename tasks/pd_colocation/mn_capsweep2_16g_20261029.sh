#!/bin/bash
# The TAIL-BOUNDED SLO chunk rule on 16 H20s, Qwen2.5-7B TP1 x 16, superposed TraceLab replay plus ShareGPT chat.
#
# What this arm changes, and why. The SLO-budgeted chunk is meant to replace a fixed token count: once a request is
# admitted, the server sizes each prefill chunk so that the requests already running stay inside their latency
# targets. Two earlier forms of that budget were measured and both fail. Pricing a step against the time one step
# of output is "worth" at the target rate pins nearly every step to about 90 ms -- the budget is a minimum over
# resident rows, and a row admitted this step has banked no slack -- which starves long agent prefills (measured on
# this cluster: agent SLO goodput 2 259 tok/s against rung L0's 2 493, agent TTFT p90 9.5 s against
# prefill-prioritized's 3.4 s). Pricing it against each row's projected mean TPOT at completion is the correct
# reading of a mean SLO and is vacuous as a bound: with rows decoding at 13.9 ms against a 50 ms target it grants
# 16.3 s per step, so the chunk saturates the server maximum on every step (offline over 10,185 recorded steps,
# tasks/perf_model/chunk_predictor.py --rule life). Amortizing the remaining slack over the remaining output is
# arithmetically identical to the first form, because its minimum is also attained at the fresh row.
#
# So the bound that still binds when the mean is comfortable is the INTER-TOKEN TAIL target, which the workload
# states as well. This arm is the mean-TPOT rule with that second bound: `stepcap=600`, the chat p99 inter-token
# target, in place of the 2 000 ms ceiling the previous arm used. Offline the same rule picks a median chunk of
# 2 176 tokens while admitting MORE than the run used on 65% of steps and less on 35%; it is not a fixed count,
# because the chunk is solved from the step model and so moves with prefix length and decode batch size, and goes
# to the server maximum when no rows are resident.
#
# Only our arm runs here: the L0 / L1 / prefill-prioritized / mixed-chunk / pdmux references are measured by
# mn_decomp_v2_16g_20261028.sh at alpha 3 and mn_overload_decomp_16g_20261028.sh at alpha 6, on this same cluster
# and window set. Read the result with superset_report.py --ours ours_tail. Claude, 2026-10-28.
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
# Two loads: alpha 3 must not regress against rung L0 (measured: chat 91.2% / agent 94.3% / 2493 tok/s), and
# alpha 6 is where a latency bound is supposed to pay. The L0 / L1 / pp / mc / pdmux references at each load come
# from mn_decomp_v2 (alpha 3) and mn_overload_decomp (alpha 6), so only our arm is run here.
# Sweep the ONE number the measured arm says is wrong. At alpha 6 the rule at stepcap=600 gave chat TPOT p90
# 55.8 ms (against the fixed-2048 control's 114.6, i.e. -51%) but starved prefill: chat TTFT p90 41.65 s against
# 5.91 and chat attainment 23.6% against 60.5%. So the mechanism works and the operating point does not. The bound
# is a latency in milliseconds, hence monotonic and interpretable, and we already hold both endpoints of the curve
# -- 600 ms far on the protective side, the 2048 constant on the throughput side. This sweep fills the interior.
#
# Read it as a Pareto curve of chat TPOT p90 against chat SLO attainment, which is also the throughput-latency
# frontier C4 claims and currently marks [pending]. The rule earns its place only if some value beats the constant
# on tail TPOT WITHOUT giving up attainment; if no value does, the honest finding is that a per-step latency bound
# cannot buy tail TPOT on this workload without starving prefill, and the second lever (cap by the tightest
# RESIDENT tenant's target rather than a global constant) is the next thing to try, not a tighter constant.
for A in 6; do
  point $A
  for CAP in 4000 8000 16000; do
    T=_tl2_a${A}_capsweep2_r1
    KERNEL=a6 EDF=1 CHUNK=16384 SLO_CHUNK="$MODEL:0.1:50:80:24000:demote:ttft:stepcap=${CAP}:life" \
      SGL_PY=/dev/shm/w/python_slo_life RUN_TAG=${T}_cap${CAP} ROUTER_ARGS="$ROUTER_ARGS_ALL" \
      python3 -u mn_run.py coloc_cache_edf_mtp_hc
    report coloc_cache_edf_mtp_hc${T}_cap${CAP}
  done
done
echo MN_CAPSWEEP_DONE
