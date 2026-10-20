#!/bin/bash
# Fault tolerance (2026-10-20): 8 H20s, text fan-out (C=32 agents, 16 x 64K prefixes) + ShareGPT chat 8 req/s, as the
# main multi-node comparison. At t = 300 s every worker on pod 1 is killed (FAULT_AT / FAULT_POD in mn_run.py), the same
# 2 of 8 GPUs in both arms: under PD disagg 2P2D pod 1 is one of the two prefill pods, under ours two colocated
# workers. Ours: router.py marks a worker dead on a refused or broken connection and reroutes; the stock PD router
# ejects it by health check (DISAGG_HEALTH=1: every 10 s, 3 failures). Scored in windows (before / after the fault) by
# tasks/perf_model/spike_windows.py --spike 300:600. Two repeats. Claude.
set -u
cd "$(dirname "$0")"
export FO_TEXT=1 FAULT_AT=300 FAULT_POD=1 DISAGG_HEALTH=1
for REPN in 1 2; do
  EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" REP=_text_fault_r$REPN ARMS=coloc_slo_edf_hc \
    bash mn_fanout_8gpu_20261018.sh
  REP=_text_fault_r$REPN ARMS=disagg_2p2d bash mn_fanout_8gpu_20261018.sh
done
echo MN_FAULT_DONE
