#!/bin/bash
# Load sweep at 32 H20s (2026-10-21): at 4x the 8-GPU load disagg was overloaded at every P:D ratio, so ours vs disagg
# also at 1x and 2x that load -- agent programs 32 / 64 (tasks 16 / 32) with chat 8 / 16 req/s, 900 s -- where disagg
# should have headroom. Qwen2.5-7B no spec. Ours vs disagg 5 + 3 and 4 + 4. Starts once every pod reports DL_DONE
# (mn_models_dl_20261021.sh), so model downloads do not overlap a measurement. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
until python3 - <<'PY'
import sys; sys.path.insert(0, "/opt/tiger/sglang/tasks/pd_colocation")
from px_multi import Cluster
c = Cluster("2f608c1b1a67cf09", "cn-seed", nodes=[f"worker-{i}" for i in range(8)])
sys.exit(0 if all("DL_DONE" in c.run(i, "tail -1 /dev/shm/w/dl_models.log")[1] for i in range(8)) else 1)
PY
do sleep 120; done
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=8 GPUS_PER_POD=4 TP=1 MODEL_DIR=qwen7b FO_TEXT=1
export RA_LOG=/tmp/h20/mn_32g_calibrate.log POOL_KV_TOKENS=13392000
for L in 1 2; do
  export FO_TASKS=$((16 * L)) AGENT_CONC=$((32 * L)) CHAT_RATE=$((8 * L)) CHAT_N=$((7200 * L))
  EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" REP=_text_32g_L${L}_r1 ARMS=coloc_slo_edf_hc \
    bash mn_fanout_8gpu_20261018.sh
  for arm in disagg_p5 disagg_p4; do REP=_text_32g_L${L}_r1 ARMS=$arm bash mn_fanout_8gpu_20261018.sh; done
done
echo MN_LOAD_SWEEP32_DONE
