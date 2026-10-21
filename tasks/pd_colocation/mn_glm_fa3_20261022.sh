#!/bin/bash
# GLM-4.5-Air on upstream's FA3 backend: a REMEDY test for the TP4 hang, not just another soak (one H20 pod,
# job 2f608c1b1a67cf09, 2026-10-22, after $1). The hang (a forward step that never returned, 600 s watchdog on TP1)
# happened in ours-no-MTP on our flashinfer-backend split path, and did not reproduce in 8 soak runs with py-spy armed
# -- so it is rarer than 1 in 9 there. Meanwhile the H200 work showed our 4-mode schedule runs on sglang's own FA3
# backend (MIXEDSPEC_STEPS 851, accept 2.05) and is FASTER there with MTP, and that backend contains none of our split
# code. So: run the same two GLM arms with KERNEL=lpt + ATTN_BACKEND=fa3 at both loads, 2 repeats of the arm that hung.
# If they are clean and competitive, the remedy for the hang is to run this schedule on the FA3 backend, which we want
# on performance grounds regardless -- and the flashinfer split path can be retired rather than debugged.
# WATCHDOG=180 and py-spy on PATH stay on, so a hang here still yields a stack. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
export CHAT_TEMPLATE=1 WATCHDOG=180
for L in 2 4; do
  export AGENT_CONC=$L SESSIONS=$((3 * L)) CHAT_RATE=$L CHAT_N=$((300 * L))
  for REPN in 1 2; do
    ATTN_BACKEND=fa3 KERNEL=lpt TAG=_glm_L${L}_fa3be_r$REPN ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 \
      bash mn_h20_single_20261021.sh
  done
  ATTN_BACKEND=fa3 KERNEL=lpt DECODE_ROWS=48:20:24000:80:0.9 \
    WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs-decode 48" TAG=_glm_L${L}_fa3be_rows48adm_r1 \
    ARMS=ours_mtp bash mn_h20_single_20261021.sh
done
python3 - <<'PY'
import sys; sys.path.insert(0, "/opt/tiger/sglang/tasks/pd_colocation")
from px_multi import Cluster
c = Cluster("2f608c1b1a67cf09", "cn-seed", nodes=["worker-0"])
print(c.run(0, "grep -l 'watchdog timeout' /dev/shm/w/mn/*glm_L*fa3be*/worker_r0_g0.log 2>/dev/null | wc -l; "
               "grep -c Traceback /dev/shm/w/mn/*glm_L*fa3be*/worker_r0_g0.log 2>/dev/null | tr '\\n' ' '")[1])
PY
echo MN_GLM_FA3_DONE
