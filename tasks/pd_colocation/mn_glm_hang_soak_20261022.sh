#!/bin/bash
# Soak for the intermittent GLM-4.5-Air TP4 hang (one H20 pod, job 2f608c1b1a67cf09, 2026-10-22). On 2026-10-21 a
# forward step in ours-no-MTP stopped returning at quarter load; the 600 s watchdog fired on TP1 and the worker was
# killed, with 0 tracebacks and no CUDA error, and sglang's own `py-spy dump` failed because py-spy was not installed.
# Now: py-spy is in the pod venv and mn_node.sh puts it on the worker's PATH, and WATCHDOG=180 so a stuck step is
# caught in ~3 min with a native stack instead of after 10. Eight runs of the arm that hung, same load; after each,
# the log is scanned for the watchdog line and any py-spy dump is printed. Claude.
set -u
cd "$(dirname "$0")"
export CHAT_TEMPLATE=1 AGENT_CONC=2 SESSIONS=6 CHAT_RATE=2 CHAT_N=600 WATCHDOG=180
for i in $(seq 1 8); do
  echo "##### soak $i"
  SPLIT_ROWS_FA3=1 DECODE_FA3=1 SPLIT_POLICY=1 SPLIT_FASTPATH=0 TAG=_glm_soak$i ARMS=ours_ovl CHUNK=2048 \
    DEADLINE_CHUNK=1024:2048:6000 bash mn_h20_single_20261021.sh
  python3 - "$i" <<'PY'
import sys; sys.path.insert(0, "/opt/tiger/sglang/tasks/pd_colocation")
from px_multi import Cluster
c = Cluster("2f608c1b1a67cf09", "cn-seed", nodes=["worker-0"])
d = f"/dev/shm/w/mn/coloc_cache_edf_h20tp4_split_ovl_ch2048_dl1024-2048-6000_c8_glm_soak{sys.argv[1]}"
out = c.run(0, f"grep -c 'watchdog timeout' {d}/worker_r0_g0.log; grep -A40 'Pyspy\\|py-spy dump' {d}/worker_r0_g0.log | head -60")[1]
print(out[:4000], flush=True)
PY
done
echo MN_GLM_SOAK_DONE
