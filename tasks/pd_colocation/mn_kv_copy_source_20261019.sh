#!/bin/bash
# Point 4 as proposed (2026-10-19): pull-source choice among per-worker KV copies, serving node fixed by the SLO-checked
# hit-first router in every arm. 8 H20s, fan-out on natural text, narrowed fabric (ONE_NIC=1), hot set rotating every
# 45 s -- every worker pulls each new hot prefix at once. KV_COPY = off (one source per page, stock) / first / random /
# least_latency (apply_kv_copy_source.py, applied on every pod). Claude.
set -u
cd "$(dirname "$0")"
export FO_TEXT=1 ONE_NIC=1 FO_ROTATE=45 EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5"
for POL in ${POLICIES:-off first random least_latency}; do
  KV_COPY=$POL REP=_text_1nic_rot45_kvc_${POL}${REP_SUFFIX:-} ARMS=coloc_slo_edf_hc bash mn_fanout_8gpu_20261018.sh
done
echo KV_COPY_SOURCE_DONE
