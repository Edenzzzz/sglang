#!/bin/bash
# Draft model's draft extend on FA3 (apply_draft_fa3.py), H200, 2026-10-21, after h200_chunk1024_cr12_20261021.sh
# (pid $1). Stock's prefill-prioritized + MTP schedule on our backend reached 142.5 tok/s vs 161 on stock's FA3 backend,
# with only the draft model still on flashinfer. (1) gate: ours + MTP, VERIFY_FA3_CHECK=1, graphs off, reduced load;
# (2) stock's schedule on our backend with DRAFT_FA3; (3) ours + MTP (rows + SLO admission) with DRAFT_FA3, two repeats.
# Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
python3 - <<'PY' || exit 1
import os, subprocess, sys, tarfile
sys.path.insert(0, "/tmp")
from px import run, push
os.makedirs("/tmp/pushr/tasks/pd_colocation", exist_ok=True)
for f in ("mn_node.sh", "apply_draft_fa3.py"):
    subprocess.run(["cp", f, "/tmp/pushr/tasks/pd_colocation/"], check=True)
with tarfile.open("/tmp/pushr.tgz", "w:gz") as t:
    t.add("/tmp/pushr/tasks", arcname="tasks")
assert push("/tmp/pushr.tgz", "/tmp/r.b64", "/dev/shm/w")
out = run("cd /dev/shm/w && cp up/python/sglang/srt/layers/attention/flashinfer_backend.py flashinfer_backend.pre_draft.py && "
          "/dev/shm/w/venv/bin/python tasks/pd_colocation/apply_draft_fa3.py up/python/sglang/srt && "
          "md5sum up/python/sglang/srt/layers/attention/flashinfer_backend.py")[1]
print(out)
assert "applied" in out and "a950eea49ec7" in out
PY
export CHAT_TEMPLATE=1
FA="SPLIT_ROWS_FA3=1 VERIFY_FA3=1 SPLIT_POLICY=1 DRAFT_FA3=1"
WX="--max-running-requests 128 --cuda-graph-max-bs-decode 48"
env $FA VERIFY_FA3_CHECK=1 SPLIT_OVERLAP=1 SESSIONS=8 CHAT_N=600 DECODE_ROWS=48:20:24000:80:0.9 \
  WORKER_EXTRA="$WX --disable-cuda-graph" TAG=_ct_rows48adm_dr3gate ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
(export SID=dadf94f0e1da7238 MN_CP=i18n-bd NPODS=1 GPUS_PER_POD=1 SM_FA3=112 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 \
   TURNS=2 AGENT_CONC=8 SESSIONS=24 CHAT_RATE=8 CHAT_N=2400
 env $FA MIXED_CHUNK=0 KERNEL=mtpsplit PREEMPT=0 RUN_TAG=_h200_pp_c8_ct_ourbe_dr3 python3 -u mn_run.py coloc_sgl_mtps)
for REPN in 1 2; do
  env $FA SPLIT_OVERLAP=1 DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="$WX" TAG=_ct_rows48adm_dr3_rep$REPN ARMS=ours_mtp \
    bash mn_h200_nospec_20261018.sh
done
echo H200_DRAFT_FA3_DONE
