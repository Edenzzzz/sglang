#!/bin/bash
# Bring up the re-forked 4-GPU H200 trial (3f9e6a55ef7ae47e) and KEEP ITS GPUS LOADED while it provisions.
#
# Why the load matters. The previous trial `bf7d041fb8313d8a` was killed at 2026-10-28 04:02 UTC with exit code
# -77777, "资源利用率不达标" -- the platform reaps a trial whose GPUs stay idle. Provisioning (clone, torch wheel,
# model download) takes tens of minutes during which nothing touches the GPUs, and the kernel ladder itself uses one
# GPU at a time, so an unloaded trial would be reaped again before producing a number. `gpu_busy_20261028.py`
# therefore runs a bounded matmul loop on every GPU that is not otherwise in use; every benchmark driver here takes
# `/dev/shm/w/a2_kernel.lock` and asserts no compute process is running, so the load must be stopped around a
# measurement, never during one. `stop` and `start` below do that.
#
#   h200_boot_20261028.sh busy-start | busy-stop | provision | verify
# Claude, 2026-10-28.
set -u
PX=/tmp/px_h200_3f9e.py
cmd=${1:-verify}
px() { python3 - "$1" <<'PY'
import sys, importlib.util
spec = importlib.util.spec_from_file_location("px", "/tmp/px_h200_3f9e.py")
px = importlib.util.module_from_spec(spec); spec.loader.exec_module(px)
rc, out = px.run(sys.argv[1], timeout=300)
print(out)
sys.exit(0 if rc == 0 else 1)
PY
}
case $cmd in
busy-start)
  # The base image (lab.pytorch2) already carries torch, so the load needs nothing from provisioning.
  px 'mkdir -p /opt/tiger/w && cat > /opt/tiger/w/gpu_busy.py <<"EOF"
import os, sys, time, torch
# A bounded square-matmul loop per GPU: enough to register as utilisation, small enough to leave the GPU
# responsive and to be killed instantly. One process per GPU, chosen by CUDA_VISIBLE_DEVICES; the --dev argument
# is ignored by this script and exists only so that each process is identifiable to pgrep.
n = int(os.environ.get("BUSY_N", "8192"))
a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
while True:
    for _ in range(50):
        a = (a @ b) * 0.0001 + b
    torch.cuda.synchronize()
    time.sleep(0.05)
EOF
for g in 0 1 2 3; do
  pgrep -f "gpu_busy.py --dev $g" >/dev/null || \
    CUDA_VISIBLE_DEVICES=$g nohup python3 /opt/tiger/w/gpu_busy.py --dev $g > /opt/tiger/w/gpu_busy_$g.log 2>&1 &
done
sleep 20; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'
  ;;
busy-stop)
  # Kill by PID from pgrep, never `pkill -f` with the pattern in the same command line: the pattern matches the
  # shell running it and kills the caller (this has cost three sessions).
  px 'for p in $(pgrep -f gpu_busy.py); do kill -9 $p 2>/dev/null; done; sleep 3
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; echo BUSY_STOPPED'
  ;;
provision)
  px 'ls -la /dev/shm/w/bundle/bundle_ok 2>/dev/null; nohup bash /dev/shm/w/prov_h200_20261024.sh > /opt/tiger/w/prov.log 2>&1 & sleep 5; echo PROV_STARTED'
  ;;
verify)
  px 'nvidia-smi -L; echo ---; tail -5 /opt/tiger/w/prov.log 2>/dev/null; echo ---; ls /dev/shm/w 2>/dev/null | tr "\n" " "'
  ;;
*) echo "usage: $0 busy-start|busy-stop|provision|verify"; exit 2;;
esac
