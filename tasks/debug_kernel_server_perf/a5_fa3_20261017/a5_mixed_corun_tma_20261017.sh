#!/bin/bash
# A5.3b (paged-TMA FA3 build): what the non-FA3 SMs buy when FA2 verify rows run concurrently, now that the FA3 role is flash-attention's kernel
# (-DNDEBUG, fast math). Mixed fixtures from serving shapes (FA3 prefill/append rows + FA2 q=4 verify rows). Arms per run
# (all write one output, checked against the fp32 causal reference): split = sgl-kernel FA3 on all SMs then stock FA2
# BatchPrefill (serving's two-launch split); corun_sgl = sgl-kernel FA3 with sm_margin (B SMs) || stock FA2 on a second
# stream; corun_dyn = our flash-attention build on its dynamic scheduler capped at B || stock FA2; candidate = our
# compact FA3 role (static lists) + native FA2 as its PDL dependent; stock = flashinfer FA3 + FA2. B in {96,112,120},
# two reversed repeats.
set -uo pipefail
W=/dev/shm/w
cd $W
rep() { local n=$1 s=$2 o=""; for _ in $(seq $n); do o+="$s,"; done; echo "${o%,}"; }
shapes=("fresh512_v46|512:512|$(rep 46 4:49152)"
        "append512_v46|512:49152|$(rep 46 4:49152)"
        "prefill2048_v8|2048:65536|$(rep 8 4:16384)"
        "chat2048_v32|2048:2048|$(rep 32 4:8192)"
        "agent2x512_v24|$(rep 2 512:65536)|$(rep 24 4:32768)")
for repeat in 1 2; do
  budgets=(96 112 120); [ $repeat = 2 ] && budgets=(120 112 96)
  for shape in "${shapes[@]}"; do
    IFS='|' read -r name fa3 fa2 <<< "$shape"
    for b in "${budgets[@]}"; do
      [ -s $W/codex_needs_gpu ] && { echo "A5MIXT_YIELD"; exit 0; }
      FA2_OVERRIDE="$fa2" FA3_SM_BUDGET=$b TAG=mixtma_${name}_r$repeat UPS=1 XN=ndebug PTMA=1 \
        bash a5_fa3_upstream_try_20261017.sh "$fa3" $b
    done
  done
done
echo "A5MIXT_DONE $(date -Is)"
