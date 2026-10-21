#!/bin/bash
# Per-layer dispatch fast path (apply_split_fastpath.py), H200, 2026-10-22. The shallow mixed bin's steps are tiny
# (ptok 164, prefix 37, nd 0) yet cost ours 10.67 vs stock 7.48 ms, with the split disabled and the page-table stride
# ruled out -- ~114 us per layer of dispatch work on steps too small to hide it. The fast path drops the per-layer
# output allocation and copy on single-call steps and caches the paged K/V views.
#   1. gate: SPLIT_VERIFY=1, graphs off, reduced load -- split and single-call steps vs the planned flashinfer wrapper
#   2. stock's prefill-prioritized + MTP schedule on our backend, STEP_TRACE: does the shallow bin fall to ~7.5 ms?
#   3. ours + MTP (rows + SLO admission) and ours no MTP, one repeat each at chat 8 req/s
# Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
python3 - <<'PY' || exit 1
import os, subprocess, sys, tarfile
sys.path.insert(0, "/tmp")
from px import run, push
os.makedirs("/tmp/pushfp/tasks/pd_colocation", exist_ok=True)
for f in ("mn_node.sh", "mn_run.py", "apply_split_fastpath.py"):
    subprocess.run(["cp", f, "/tmp/pushfp/tasks/pd_colocation/"], check=True)
with tarfile.open("/tmp/pushfp.tgz", "w:gz") as t:
    t.add("/tmp/pushfp/tasks", arcname="tasks")
assert push("/tmp/pushfp.tgz", "/tmp/fp.b64", "/dev/shm/w")
out = run("cd /dev/shm/w && cp up/python/sglang/srt/layers/attention/flashinfer_backend.py flashinfer_backend.pre_fast.py && "
          "/dev/shm/w/venv/bin/python tasks/pd_colocation/apply_split_fastpath.py up/python/sglang/srt && "
          "md5sum up/python/sglang/srt/layers/attention/flashinfer_backend.py; "
          "grep -c SPLIT_FASTPATH tasks/pd_colocation/mn_node.sh")[1]
print(out)
assert "489f2b7feccb" in out, "pod tree does not match the local dry-apply"
PY
export CHAT_TEMPLATE=1
FA="SPLIT_ROWS_FA3=1 SPLIT_POLICY=1 DRAFT_FA3=1 SKIP_BASE_PLAN=1 SPLIT_FASTPATH=1"
WX="--max-running-requests 128 --cuda-graph-max-bs-decode 48"
env $FA VERIFY_FA3=1 SPLIT_VERIFY=1 SPLIT_OVERLAP=1 SESSIONS=8 CHAT_N=600 WORKER_EXTRA="--disable-cuda-graph" \
  TAG=_ct_fpgate ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 bash mn_h200_nospec_20261018.sh
(export SID=dadf94f0e1da7238 MN_CP=i18n-bd NPODS=1 GPUS_PER_POD=1 SM_FA3=112 CHAT_SHAREGPT=1 MIN_CTX=32768 MAX_CTX=65536 \
   TURNS=2 AGENT_CONC=8 SESSIONS=24 CHAT_RATE=8 CHAT_N=2400 STEP_TRACE=1
 env $FA VERIFY_FA3=1 MIXED_CHUNK=0 KERNEL=mtpsplit PREEMPT=0 RUN_TAG=_h200_pp_c8_ct_ourbe_fp python3 -u mn_run.py coloc_sgl_mtps)
env $FA VERIFY_FA3=1 SPLIT_OVERLAP=1 DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="$WX" TAG=_ct_rows48adm_fp ARMS=ours_mtp \
  bash mn_h200_nospec_20261018.sh
env $FA DECODE_FA3=1 TAG=_ct_pol_fp ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 bash mn_h200_nospec_20261018.sh
python3 - <<'PY'
import sys; sys.path.insert(0, "/tmp")
from px import run
A="/dev/shm/w/mn/coloc_sgl_mtps_h200_pp_c8_ct_fa3be_tr/worker_r0_g0.log"
B="/dev/shm/w/mn/coloc_sgl_mtps_h200_pp_c8_ct_ourbe_fp/worker_r0_g0.log"
print(run(f"cd /dev/shm/w/perf && /dev/shm/w/venv/bin/python binfeat.py {A} {B}")[1])
print(run(f"cd /dev/shm/w/perf && /dev/shm/w/venv/bin/python steptrace_compare.py {A} {B} 2>&1 | grep -E 'DECODE|EXTEND|==' | head -10")[1])
PY
echo H200_FASTPATH_DONE
