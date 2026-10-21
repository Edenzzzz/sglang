#!/bin/bash
# After the rows-on-FA3 A/B (pid $1), H200, 2026-10-21: (1) apply_verify_fa3.py on the pod tree; (2) gate with
# VERIFY_FA3_CHECK=1 and CUDA graphs off (every verify layer vs the planned flashinfer wrapper; reduced load); (3) ours +
# MTP (rows + SLO admission) with rows AND verify-only steps on FA3, two repeats, beside the A/B's stock-FA3 arms;
# (4) rows_layout_check_20261021.py with the co-run margin swept to 112 SMs. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
python3 - <<'PY' || exit 1
import os, subprocess, sys, tarfile
sys.path.insert(0, "/tmp")
from px import run, push
os.makedirs("/tmp/pushv/tasks/pd_colocation", exist_ok=True)
for f in ("mn_node.sh", "apply_verify_fa3.py"):
    subprocess.run(["cp", f, "/tmp/pushv/tasks/pd_colocation/"], check=True)
subprocess.run(["cp", "../debug_kernel_server_perf/rows_layout_check_20261021.py", "/tmp/pushv/"], check=True)
with tarfile.open("/tmp/pushv.tgz", "w:gz") as t:
    t.add("/tmp/pushv/tasks", arcname="tasks")
assert push("/tmp/pushv.tgz", "/tmp/v.b64", "/dev/shm/w")
with tarfile.open("/tmp/pushv2.tgz", "w:gz") as t:
    t.add("/tmp/pushv/rows_layout_check_20261021.py", arcname="rows_layout_check_20261021.py")
assert push("/tmp/pushv2.tgz", "/tmp/v2.b64", "/dev/shm/w/perf")
out = run("cd /dev/shm/w && cp up/python/sglang/srt/layers/attention/flashinfer_backend.py flashinfer_backend.pre_vf3.py && "
          "/dev/shm/w/venv/bin/python tasks/pd_colocation/apply_verify_fa3.py up/python/sglang/srt && "
          "md5sum up/python/sglang/srt/layers/attention/flashinfer_backend.py")[1]
print(out)
assert "applied" in out
PY
export CHAT_TEMPLATE=1
WX="--max-running-requests 128 --cuda-graph-max-bs-decode 48"
SPLIT_ROWS_FA3=1 VERIFY_FA3=1 VERIFY_FA3_CHECK=1 SESSIONS=8 CHAT_N=600 DECODE_ROWS=48:20:24000:80:0.9 \
  WORKER_EXTRA="$WX --disable-cuda-graph" TAG=_ct_rows48adm_r3v3gate ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
for REPN in 1 2; do
  SPLIT_ROWS_FA3=1 VERIFY_FA3=1 DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="$WX" \
    TAG=_ct_rows48adm_r3v3_rep$REPN ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
done
python3 - <<'PY'
import sys, time; sys.path.insert(0, "/tmp")
from px import run
run("cd /dev/shm/w/perf && CUDA_VISIBLE_DEVICES=0 setsid nohup /dev/shm/w/venv/bin/python rows_layout_check_20261021.py > rows_layout_m112.out 2>&1 < /dev/null &")
for _ in range(60):
    time.sleep(30)
    if "DONE" in run("pgrep -f rows_layout_check >/dev/null && echo RUN || echo DONE")[1]: break
print(run("grep -v INFO /dev/shm/w/perf/rows_layout_m112.out")[1][-6000:])
PY
echo H200_VERIFY_FA3_DONE
