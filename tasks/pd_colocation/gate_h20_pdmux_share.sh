#!/bin/bash
# Serving-output gate on one H20 (pod 0, GPU 0) for the H20 multi-node worker config: pdmux + EDF (uncached deadlines) +
# per-step SM sharing (TTFT deadlines, 24 ms ITL budget), divisor 30. srv_correctness arms LPT (stock), LPT_REP (stock
# repeat = noise floor), LPT_PDMUX (null case: pdmux's own numerics), LPT_PDMUX_EDF (the config); each vs LPT.
# Claude, 2026-10-17.
set -uo pipefail
W=/dev/shm/w
export PYTHONPATH=$W/python HF_HOME=$W/hf CUDA_VISIBLE_DEVICES=0
out=$W/gate_h20_pdmux_share_20261017; rm -rf $out; mkdir -p $out
printf "sm_group_num: 8\ndecode_bs_divisor: 30\nsplit_forward_token_budget: 65536\n" > $out/pdmux_div30.yaml
export SGLANG_EDF_UNCACHED_DEADLINE=1 SGLANG_PDMUX_SHARE_DEADLINE=1 SGLANG_PDMUX_PREFILL_TOK_S=5019
export SGLANG_PDMUX_ITL_BUDGET_MS=24 SRV_EXTRA_ARGS="--pdmux-config-path $out/pdmux_div30.yaml"
D=$W/tasks/fused_specdec_workload/drivers/srv_correctness.py
for arm in LPT LPT_REP LPT_PDMUX LPT_PDMUX_EDF; do $W/venv/bin/python $D $arm $out 2>&1 | tail -2; done
for arm in LPT_REP LPT_PDMUX LPT_PDMUX_EDF; do echo "== $arm vs LPT"; $W/venv/bin/python $D --compare $out LPT $arm 2>&1 | grep -E "GATE|vs " | head -6; done
L=$out/server_LPT_PDMUX_EDF.log
echo "GATE_H20 tracebacks=$(grep -c Traceback $L) priority=$(grep -c enable_priority_scheduling.:.True $L) pdmux=$(grep -c enable_pdmux.:.True $L) share=$(grep -c PDMUX_SHARE_DEADLINE $L)"
echo GATE_H20_DONE
