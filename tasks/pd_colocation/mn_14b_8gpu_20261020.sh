#!/bin/bash
# Second model (2026-10-20): Qwen2.5-14B-Instruct (40 q / 8 kv heads, d128), TP1 per H20, no MTP, text fan-out C=32 +
# ShareGPT chat 8 req/s on 8 H20s. The router's SLO model is recalibrated on 14B first (decode grid
# /dev/shm/w/mn/decode_grid_qwen14b.json, its own ROUTER_ARGS log); then ours (SLO-checked hit-first router, pool, EDF)
# vs PD disagg 2P2D, two repeats. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=bb854a10b8954223 MN_CP=cn-seed NPODS=4 GPUS_PER_POD=2 MODEL_DIR=qwen14b
for i in $(seq 1 240); do
  n=$(python3 - <<'P'
import sys; sys.path.insert(0, "/tmp"); import h20q, json, subprocess, tempfile
ok = 0
for i in range(4):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"job_run_id": "bb854a10b8954223", "exec": "grep -c DL_DONE /dev/shm/w/dl_qwen14b.log", "exec_timeout_seconds": 30, "pod_name": f"worker-{i}"}, f)
    p = subprocess.run(["merlin-cli", "--control-plane", "cn-seed", "job-v2", "runs", "execute-script", "--from-file", f.name], capture_output=True, text=True, timeout=90, env=h20q.ENV)
    ok += p.stdout.strip().endswith("1")
print(ok)
P
)
  [ "$n" = 4 ] && break
  sleep 30
done
echo "downloads done on $n/4 pods"
KERNEL=mtpsplit EDF=1 SPEC=0 PREEMPT=1 CHUNK=2048 python3 -u mn_run.py calibrate | tee /tmp/h20/mn_14b_calibrate.log
grep -q "ROUTER_ARGS=" /tmp/h20/mn_14b_calibrate.log || { echo "calibration failed"; exit 1; }
export FO_TEXT=1 RA_LOG=/tmp/h20/mn_14b_calibrate.log
for REPN in 1 2; do
  EXTRA_RA="--ttft-slo-ms 10000 --itl-slo-ms 20 --itl-slo-auto 2.5" REP=_text_14b_r$REPN ARMS=coloc_slo_edf_hc \
    bash mn_fanout_8gpu_20261018.sh
  REP=_text_14b_r$REPN ARMS=disagg_2p2d bash mn_fanout_8gpu_20261018.sh
done
echo MN_14B_DONE
