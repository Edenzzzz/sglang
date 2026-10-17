#!/bin/bash
# Serving-output gate for PD multiplexing + worker-side EDF (tree up_pdmux_mtp): srv_correctness arms LPT (stock),
# LPT_REP (stock repeat, the noise floor), LPT_PDMUX (the null case: pdmux's own numerics) and LPT_PDMUX_EDF; each
# compared with LPT. Claude, 2026-10-17.
set -uo pipefail
W=/dev/shm/w
exec 9>"$W/a2_kernel.lock"; flock 9
export PYTHONPATH=$W/up_pdmux_mtp/python HF_HOME=$W/hf
out=$W/gate_pdmux_edf_20261017; mkdir -p $out
D=$W/tasks/fused_specdec_workload/drivers/srv_correctness.py
for arm in LPT LPT_REP LPT_PDMUX LPT_PDMUX_EDF; do $W/venv/bin/python $D $arm $out 2>&1 | tail -2; done
for arm in LPT_REP LPT_PDMUX LPT_PDMUX_EDF; do echo "== $arm vs LPT"; $W/venv/bin/python $D --compare $out LPT $arm 2>&1 | grep -E "GATE|vs |tok/verify" | head -20; done
L=$out/server_LPT_PDMUX_EDF.log
echo "GATE_EDF tracebacks=$(grep -c Traceback $L) priority=$(grep -c enable_priority_scheduling.:.True $L) pdmux=$(grep -c enable_pdmux.:.True $L)"
echo GATE_PDMUX_EDF_DONE
