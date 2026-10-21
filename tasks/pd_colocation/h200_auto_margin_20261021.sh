#!/bin/bash
# After h200_decode_fa3_20261021.sh (pid $1), H200, 2026-10-21: apply_split_auto_margin.py, then ours with the co-run's
# SM partition chosen per step (SPLIT_OVERLAP=1 SPLIT_MARGIN_AUTO=1) and every attention call on FA3 -- no MTP (split
# rows + plain decode on FA3) and + MTP (split rows + verify on FA3, rows + SLO admission) -- two repeats each.
# Earlier serving runs left SGLANG_FUSED_SM_MARGIN at 0: the two launches shared all SMs. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
python3 - <<'PY' || exit 1
import os, subprocess, sys, tarfile
sys.path.insert(0, "/tmp")
from px import run, push
os.makedirs("/tmp/pusha/tasks/pd_colocation", exist_ok=True)
for f in ("mn_node.sh", "apply_split_auto_margin.py"):
    subprocess.run(["cp", f, "/tmp/pusha/tasks/pd_colocation/"], check=True)
with tarfile.open("/tmp/pusha.tgz", "w:gz") as t:
    t.add("/tmp/pusha/tasks", arcname="tasks")
assert push("/tmp/pusha.tgz", "/tmp/a.b64", "/dev/shm/w")
out = run("cd /dev/shm/w && cp up/python/sglang/srt/layers/attention/flashinfer_backend.py flashinfer_backend.pre_am.py && "
          "/dev/shm/w/venv/bin/python tasks/pd_colocation/apply_split_auto_margin.py up/python/sglang/srt && "
          "md5sum up/python/sglang/srt/layers/attention/flashinfer_backend.py")[1]
print(out)
assert "applied" in out and "b7b4b6c49d68" in out
PY
export CHAT_TEMPLATE=1
WX="--max-running-requests 128 --cuda-graph-max-bs-decode 48"
for REPN in 1 2; do
  SPLIT_ROWS_FA3=1 DECODE_FA3=1 SPLIT_MARGIN_AUTO=1 TAG=_ct_r3d3am_rep$REPN ARMS=ours_ovl CHUNK=2048 \
    DEADLINE_CHUNK=1024:2048:6000 bash mn_h200_nospec_20261018.sh
  SPLIT_OVERLAP=1 SPLIT_ROWS_FA3=1 VERIFY_FA3=1 SPLIT_MARGIN_AUTO=1 DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="$WX" \
    TAG=_ct_rows48adm_r3v3am_rep$REPN ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
done
echo H200_AUTO_MARGIN_DONE
