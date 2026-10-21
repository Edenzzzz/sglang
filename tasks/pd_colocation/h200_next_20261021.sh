#!/bin/bash
# H200 queue after h200_chattemplate_20261021.sh (pid $1), 2026-10-21: (1) the 2-mode / 4-mode fusion microbench on an
# idle GPU (tasks/debug_kernel_server_perf/fusion_microbench_20261021.py -> /dev/shm/w/perf/fusion_microbench.out);
# (2) the single-node table for Qwen3-8B with its trained EAGLE3 head (AngelSlim/Qwen3-8B_eagle3, real acceptance),
# chat template on, YaRN to 128K as for Qwen2.5 (the override keeps rope_theta), two repeats. Claude.
set -u
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
export PATH=$HOME/.merlin-cli/bin:$PATH
python3 - <<'PY'
import sys; sys.path.insert(0, "/tmp")
from px import run
print(run("cd /dev/shm/w/perf && CUDA_VISIBLE_DEVICES=0 setsid nohup /dev/shm/w/venv/bin/python fusion_microbench_20261021.py > fusion_microbench.out 2>&1 < /dev/null & sleep 1; echo started")[1][-100:])
PY
for i in $(seq 1 120); do
  out=$(python3 /tmp/hx.py 'pgrep -f fusion_microbench >/dev/null && echo RUNNING || echo DONE; tail -1 /dev/shm/w/perf/fusion_microbench.out')
  echo "$out" | grep -q "^DONE" && break; sleep 30
done
export CHAT_TEMPLATE=1 MODEL_DIR=qwen3_8b DRAFT_DIR=qwen3_8b_eagle3
for REPN in 1 2; do
  TAG=_q3_ct_rep$REPN ARMS="stock stock_pp ours_ovl stock_mtp stock_mtp_pp" CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 \
    bash mn_h200_nospec_20261018.sh
  DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="--max-running-requests 128 --cuda-graph-max-bs-decode 48" \
    TAG=_q3_ct_rows48adm_rep$REPN ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
done
echo H200_NEXT_DONE
