#!/bin/bash
# A5.0e: the FA3 role on flash-attention's mainloop, built with -DNDEBUG + fast math (UPS=1 XN=ndebug), at 112 and
# 132 CTAs, two reversed repeats, on the seven A5.0 fixtures; plus the flashinfer-mainloop role with -DNDEBUG (UPS=0)
# once, to see how much of its own gap was assertions. Each run: our candidate, flashinfer FA3, sgl-kernel upstream
# FA3 and (UPS=1) our upstream build on upstream's scheduler (fa3only), all checked against the fp32 causal reference.
set -uo pipefail
W=/dev/shm/w
cd $W
rep() { local n=$1 s=$2 o=""; for _ in $(seq $n); do o+="$s,"; done; echo "${o%,}"; }
shapes=("ncu8x1024_kv32k|$(rep 8 1024:32768)" "chat_fresh2048|2048:2048" "chat_fresh4x512|$(rep 4 512:512)"
        "append512_kv48k|512:49152" "agent4x512_kv64k|$(rep 4 512:65536)" "prefill2048_kv64k|2048:65536"
        "short16x16_kv1k|$(rep 16 16:1024)")
for repeat in 1 2; do
  budgets=(112 132); [ $repeat = 2 ] && budgets=(132 112)
  for shape in "${shapes[@]}"; do
    for b in "${budgets[@]}"; do
      [ -s $W/codex_needs_gpu ] && { echo "A5TAB_YIELD"; exit 0; }
      TAG=${shape%%|*}_r$repeat UPS=1 XN=ndebug PTMA=0 bash a5_fa3_upstream_try_20261017.sh "${shape#*|}" $b
    done
  done
done
for shape in "${shapes[@]}"; do TAG=${shape%%|*}_r1 UPS=0 XN=ndebug PTMA=0 bash a5_fa3_upstream_try_20261017.sh "${shape#*|}" 112; done
echo "A5TAB_DONE $(date -Is)"
