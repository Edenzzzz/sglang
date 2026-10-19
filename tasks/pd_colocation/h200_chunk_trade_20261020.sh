#!/bin/bash
# Single-node chunk trade (2026-10-20): vs prefill-prioritized stock our agent attainment has a large margin (92% vs
# 59%) and chat misses on TPOT (p90 ~61 ms from deep-context mixed steps), so smaller deadline chunks trade agent TTFT
# for chat TPOT. Ours without and with MTP at deadline chunk 512/1024 and 256/1024 (pace 6000), one repeat each; the
# per-row slack decode-first rule starved prefill on one node (agent 50 tok/s, TTFT 30 s) and is not used. Claude.
set -u
cd "$(dirname "$0")"
for DC in 512:1024:6000 256:1024:6000; do
  TAG=_pt_dc${DC%%:*} ARMS=ours_ovl CHUNK=1024 DEADLINE_CHUNK=$DC bash mn_h200_nospec_20261018.sh
  TAG=_pt_dc${DC%%:*} ARMS=ours_mtp_dc CHUNK=1024 DEADLINE_CHUNK=$DC bash mn_h200_nospec_20261018.sh
done
echo H200_CHUNK_TRADE_DONE
