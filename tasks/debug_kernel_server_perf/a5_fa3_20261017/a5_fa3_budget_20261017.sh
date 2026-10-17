#!/bin/bash
# A5.0f: our flash-attention FA3 build on upstream's own dynamic persistent scheduler, grid capped at the FA3 SM budget
# (fa3_only_run with FA3_SM_BUDGET=112), vs our candidate (static per-CTA lists from our planner) at 112 CTAs, on the
# seven A5.0 fixtures. Same harness, all arms against the fp32 causal reference. Two reversed repeats.
set -uo pipefail
W=/dev/shm/w
cd $W
rep() { local n=$1 s=$2 o=""; for _ in $(seq $n); do o+="$s,"; done; echo "${o%,}"; }
shapes=("ncu8x1024_kv32k|$(rep 8 1024:32768)" "chat_fresh2048|2048:2048" "chat_fresh4x512|$(rep 4 512:512)"
        "append512_kv48k|512:49152" "agent4x512_kv64k|$(rep 4 512:65536)" "prefill2048_kv64k|2048:65536"
        "short16x16_kv1k|$(rep 16 16:1024)")
for repeat in 1 2; do
  for shape in "${shapes[@]}"; do
    [ -s $W/codex_needs_gpu ] && { echo "A5BUD_YIELD"; exit 0; }
    FA3_SM_BUDGET=112 TAG=budget112_${shape%%|*}_r$repeat UPS=1 XN=ndebug PTMA=0 bash a5_fa3_upstream_try_20261017.sh "${shape#*|}" 112
  done
done
echo "A5BUD_DONE $(date -Is)"
