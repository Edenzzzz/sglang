#!/bin/bash
# After h200_verify_fa3_20261021.sh (pid $1), H200, 2026-10-21: (1) apply_decode_fa3.py on the pod tree; (2) gate ours
# without MTP with DECODE_FA3=1 VERIFY_FA3_CHECK=1 and CUDA graphs off (each decode layer vs the planned BatchDecode
# wrapper; reduced load); (3) ours without MTP (overlapped split, EDF, deadline chunk) with split rows AND plain decode on
# FA3, two repeats -- every attention call on FA3, matching stock's Hopper default, so only the mixed step differs.
# Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
python3 - <<'PY' || exit 1
import os, subprocess, sys, tarfile
sys.path.insert(0, "/tmp")
from px import run, push
os.makedirs("/tmp/pushd/tasks/pd_colocation", exist_ok=True)
for f in ("mn_node.sh", "apply_decode_fa3.py"):
    subprocess.run(["cp", f, "/tmp/pushd/tasks/pd_colocation/"], check=True)
with tarfile.open("/tmp/pushd.tgz", "w:gz") as t:
    t.add("/tmp/pushd/tasks", arcname="tasks")
assert push("/tmp/pushd.tgz", "/tmp/d.b64", "/dev/shm/w")
out = run("cd /dev/shm/w && cp up/python/sglang/srt/layers/attention/flashinfer_backend.py flashinfer_backend.pre_dfa3.py && "
          "/dev/shm/w/venv/bin/python tasks/pd_colocation/apply_decode_fa3.py up/python/sglang/srt && "
          "md5sum up/python/sglang/srt/layers/attention/flashinfer_backend.py")[1]
print(out)
assert "applied" in out and "6facb52265b9" in out
PY
export CHAT_TEMPLATE=1
SPLIT_ROWS_FA3=1 DECODE_FA3=1 VERIFY_FA3_CHECK=1 SESSIONS=8 CHAT_N=600 WORKER_EXTRA="--disable-cuda-graph" \
  TAG=_ct_r3d3gate ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 bash mn_h200_nospec_20261018.sh
for REPN in 1 2; do
  SPLIT_ROWS_FA3=1 DECODE_FA3=1 TAG=_ct_r3d3_rep$REPN ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 \
    bash mn_h200_nospec_20261018.sh
done
echo H200_DECODE_FA3_DONE
