#!/bin/bash
# After patch item 8 (prefill-stream draft extend on its own backend + workspace): re-gate MTP_PDMUX against stock MTP
# (MTP / MTP_REP reference files reused from gate_pdmux_mtp_20261017 -- same tree, untouched code path), then rerun the
# payoff A/B's MTP_PDMUX arms at C=8 and C=16 (both pre-fix runs moved aside: C=16 crashed with an illegal address,
# and C=8 ran with the race). Waits for a2_kernel.lock; the A/B yields to codex_needs_gpu. Claude, 2026-10-17.
set -uo pipefail
W=/dev/shm/w
trap 'printf "FIX8_EXIT status=%s time=%s\n" "$?" "$(date -Is)"' EXIT
(
  exec 9>"$W/a2_kernel.lock"; flock 9
  export PYTHONPATH=$W/up_pdmux_mtp/python HF_HOME=$W/hf
  ref=$W/gate_pdmux_mtp_20261017; out=$W/gate_pdmux_mtp_fix8_20261017; mkdir -p $out
  cp $ref/*MTP.* $ref/*MTP_REP.* $out/ 2>/dev/null; ls $out
  D=$W/tasks/fused_specdec_workload/drivers/srv_correctness.py
  $W/venv/bin/python $D MTP_PDMUX $out 2>&1 | tail -2
  L=$(ls -t $out/*MTP_PDMUX*.log 2>/dev/null | head -1)
  echo "FIX8_GATE_SERVER tracebacks=$(grep -c Traceback $L 2>/dev/null) illegal=$(grep -ci 'illegal' $L 2>/dev/null)"
  for arm in MTP_REP MTP_PDMUX; do $W/venv/bin/python $D --compare $out MTP $arm 2>&1 | grep -E "GATE|vs |tok/verify" | head -20; done
  echo FIX8_GATE_DONE
)
for C in 8 16; do
  d=$W/pdmux_mtp_ab_20261017/cohort_c$C/MTP_PDMUX
  [ -d $d ] && mv $d $d.prefix8_$(date +%H%M)
done
bash $W/pdmux_mtp_ab_20261017.sh
