#!/bin/bash
# Per-step split-or-one-call policy (apply_split_policy.py) on the H200, 2026-10-21. serving_mix_corun_20261021.py:
# split wins 20-77% vs one FA3 call when a short chunk sits beside long-context rows, loses 0-21% beside a long chunk;
# sharing all SMs beats a reserved partition. Ours = every extend step on FA3, split only below
# SGLANG_SPLIT_SINGLE_CHUNK_MS (1.0), co-run on shared SMs (SPLIT_OVERLAP=1, margin 0), plain decode / verify on FA3.
#   1. gate: SPLIT_VERIFY=1 (split and one-call steps vs the planned flashinfer wrapper), no MTP, reduced load
#   2. ours no MTP and ours + MTP (rows + SLO admission), two repeats each, chat 8 req/s
#   3. chat 12 and 16 req/s: stock mixed chunk / prefill-prioritized on FA3 vs ours no MTP, one repeat each
# Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
python3 - <<'PY' || exit 1
import os, subprocess, sys, tarfile
sys.path.insert(0, "/tmp")
from px import run, push
os.makedirs("/tmp/pushp/tasks/pd_colocation", exist_ok=True)
for f in ("mn_node.sh", "apply_split_policy.py"):
    subprocess.run(["cp", f, "/tmp/pushp/tasks/pd_colocation/"], check=True)
with tarfile.open("/tmp/pushp.tgz", "w:gz") as t:
    t.add("/tmp/pushp/tasks", arcname="tasks")
assert push("/tmp/pushp.tgz", "/tmp/p.b64", "/dev/shm/w")
out = run("cd /dev/shm/w && cp up/python/sglang/srt/layers/attention/flashinfer_backend.py flashinfer_backend.pre_policy.py && "
          "/dev/shm/w/venv/bin/python tasks/pd_colocation/apply_split_policy.py up/python/sglang/srt && "
          "md5sum up/python/sglang/srt/layers/attention/flashinfer_backend.py")[1]
print(out)
assert "applied" in out and "31b6d37c693d" in out
PY
export CHAT_TEMPLATE=1
OURS="SPLIT_ROWS_FA3=1 DECODE_FA3=1 SPLIT_POLICY=1"
env $OURS SPLIT_VERIFY=1 SESSIONS=8 CHAT_N=600 TAG=_ct_polgate ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 \
  bash mn_h200_nospec_20261018.sh
WX="--max-running-requests 128 --cuda-graph-max-bs-decode 48"
for REPN in 1 2; do
  env $OURS TAG=_ct_pol_rep$REPN ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 bash mn_h200_nospec_20261018.sh
  env SPLIT_ROWS_FA3=1 VERIFY_FA3=1 SPLIT_POLICY=1 SPLIT_OVERLAP=1 DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="$WX" \
    TAG=_ct_rows48adm_pol_rep$REPN ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
done
for R in 12 16; do
  export CHAT_RATE=$R CHAT_N=$((R * 300))
  ATTN_BACKEND=fa3 TAG=_ct_fa3be_cr${R} ARMS="stock stock_pp" bash mn_h200_nospec_20261018.sh
  env $OURS TAG=_ct_pol_cr${R} ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 bash mn_h200_nospec_20261018.sh
done
echo H200_SPLIT_POLICY_DONE
