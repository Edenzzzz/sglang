#!/bin/bash
# Serving-output gate for EAGLE3 MTP + PD multiplexing (tree up_pdmux_mtp), CUDA graphs on: srv_correctness.py arms
# MTP (stock), MTP_REP (stock repeat = the noise floor) and MTP_PDMUX, all on the same tree; then MTP_REP and MTP_PDMUX
# each compared with MTP. Claude, 2026-10-17.
set -uo pipefail
W=/dev/shm/w
exec 9>"$W/a2_kernel.lock"; flock 9
pgrep -f '[s]glang.launch_server' >/dev/null && { echo "GPU predecessor still running"; exit 1; }
export PYTHONPATH=$W/up_pdmux_mtp/python HF_HOME=$W/hf
out=$W/gate_pdmux_mtp_20261017; mkdir -p $out
D=$W/tasks/fused_specdec_workload/drivers/srv_correctness.py
for arm in MTP MTP_REP MTP_PDMUX; do $W/venv/bin/python $D $arm $out 2>&1 | tail -2; done
for arm in MTP_REP MTP_PDMUX; do $W/venv/bin/python $D --compare $out MTP $arm 2>&1 | grep -E "GATE|vs |tok/verify" | head -20; done
echo GATE_DONE
