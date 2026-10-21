#!/bin/bash
# Point 3's DOWNSTREAM claim: does a more accurate step predictor actually buy SLO control? (16 H20s = 4 pods x 4,
# job 2f608c1b1a67cf09, 2026-10-22, after $1.) The accuracy itself is now measured -- our 4-mode step on the fused
# backend predicts GPU time at 0.4% decode / 3.2% prefill MAPE vs stock + MTP's 1.5% / 5.7%, and the same schedule on
# upstream's FA3 backend is 2.7-4.5x worse -- but accuracy is only worth something if the router's SLO check makes
# better admissions with it. So run ours three times with the SAME schedule and workload and only the router's perf
# model perturbed, emulating a predictor with the error our measurements say the alternatives have:
#   calibrated   the fitted curve / grid as calibrate_h20.py measured them
#   optimistic   prefill curve and decode grid scaled 0.75x (the check thinks steps are cheaper than they are)
#   pessimistic  scaled 1.25x (thinks they are dearer)
# A 25% mis-specification is roughly the spread between a 3% and a 12% MAPE model at the p90 of the error. If
# per-tenant attainment is flat across all three, predictor accuracy does NOT convert into SLO control at this scale
# and point 3's downstream half should be dropped; if calibrated wins, the gap is what the accuracy is worth.
# Qwen2.5-7B no spec, 32 tasks / 64 programs, chat 16 req/s, 900 s, one repeat each. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=4 GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen7b CHAT_TEMPLATE=1 FO_TEXT=1
export POOL_KV_TOKENS=6696000 FO_TASKS=32 AGENT_CONC=64 CHAT_RATE=16 CHAT_N=14400
BASE=$(grep -oE "ROUTER_ARGS=.*" /tmp/h20/mn_32g_calibrate.log | tail -1 | cut -d= -f2-)
[ -n "$BASE" ] || { echo "no calibrated ROUTER_ARGS found"; exit 1; }
scale() { # $1 = factor; scale the prefill curve's ms and the prefill-tok-s, leave the grid path alone
  python3 - "$1" "$BASE" <<'PY'
import re, sys
f, base = float(sys.argv[1]), sys.argv[2]
def curve(m):
    pts = [p.split(":") for p in m.group(1).split(",")]
    return "--prefill-curve " + ",".join(f"{t}:{float(ms) * f:.1f}" for t, ms in pts)
out = re.sub(r"--prefill-curve ([0-9:.,]+)", curve, base)
out = re.sub(r"--prefill-tok-s ([0-9.]+)", lambda m: f"--prefill-tok-s {float(m.group(1)) / f:.0f}", out)
print(out)
PY
}
for V in calibrated optimistic pessimistic; do
  case $V in
    calibrated) RA_OVERRIDE="$BASE" ;;
    optimistic) RA_OVERRIDE="$(scale 0.75)" ;;
    pessimistic) RA_OVERRIDE="$(scale 1.25)" ;;
  esac
  echo "##### $V: $RA_OVERRIDE"
  RA_OVERRIDE="$RA_OVERRIDE" SPLIT_ROWS_FA3=1 SPLIT_POLICY=1 SPLIT_FASTPATH=1 \
    EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" REP=_pred_${V}_16g_r1 \
    ARMS=coloc_slo_edf_hc bash mn_fanout_8gpu_20261018.sh
done
echo MN_PREDICTOR_VALUE_DONE
