#!/bin/bash
# PD item 7, margin for the TTFT feasibility test (2026-10-17). Over 13 200 hot-mix requests the clean-grid predictor's
# measured / predicted TTFT has median 1.04-1.12 but p80 1.34-1.58 and p90 2.25-2.36 where a feasible node existed, and
# it differs by class -- so a request judged feasible misses its deadline about half the time. --ttft-quantile q judges
# feasibility at the q-quantile of that ratio over a sliding window per class (chat-length / agent-length prompts).
# Hot mix (4 x 16K shared-prefix chat at 16 req/s + agent C=16), pool on, timeout prefetch, resident ITL check, cache
# fallback; q = 0 vs q = 0.8, interleaved x2, spread figure at the end. Waits for the prefetch diagnostic. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
until grep -q PREFETCH_DIAG_DONE /tmp/h20/mn_prefetch_diag.log 2>/dev/null; do sleep 30; done
python3 - <<'PY'
import base64
from px_multi import Cluster
c = Cluster("bb854a10b8954223", "cn-seed", nodes=["worker-0"])
b = base64.b64encode(open("router.py", "rb").read()).decode()
print(c.run(0, "echo " + b + " | base64 -d > /dev/shm/w/tasks/pd_colocation/router.py; md5sum /dev/shm/w/tasks/pd_colocation/router.py")[1])
PY
md5sum router.py
RA=$(grep -oE "ROUTER_ARGS=.*" /tmp/h20/mn_slo_calibrate.log | tail -1 | cut -d= -f2-)
RA="$RA --pool-kv-tokens 3348000 --prefetch-policy timeout --pool-write-policy write_through_selective --chunk-tokens 2048 --check-residents 1 --no-feasible-fallback cache"
export SID=bb854a10b8954223 NPODS=4 MIN_CTX=32768 MAX_CTX=65536 TURNS=2
export KERNEL=mtpsplit PREEMPT=1 CHUNK=2048 HC_PREFETCH=timeout MC_EVICT_HIGH=0.85
export AGENT_CONC=16 SESSIONS=48 HOT=1 HOT_GROUPS=4 HOT_PREFIX=16384 CHAT_RATE=16 CHAT_N=3200
for rep in 1 2; do
  RUN_TAG=_8g_hot_c16_margin_q0_rep$rep ROUTER_ARGS="$RA --ttft-quantile 0" python3 -u mn_run.py coloc_slo_edf_mtp_hc
  RUN_TAG=_8g_hot_c16_margin_q80_rep$rep ROUTER_ARGS="$RA --ttft-quantile 0.8" python3 -u mn_run.py coloc_slo_edf_mtp_hc
done
PYTHONPATH=/tmp/mplpkgs python3 mn_ab_figure.py figures_20261017/h20x8_hot_ttft_margin \
  "q0=coloc_slo_edf_mtp_hc_8g_hot_c16_margin_q0,q0.8=coloc_slo_edf_mtp_hc_8g_hot_c16_margin_q80"
echo TTFT_MARGIN_DONE
