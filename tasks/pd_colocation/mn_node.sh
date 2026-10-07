#!/bin/bash
# Start this pod's two sglang workers (one per GPU) for one multi-node arm.
#   mn_node.sh <mode: coloc|prefill|decode> <logdir>
# Ports are unique per pod because pods on one host share its network namespace:
#   http 30000+100*role+{1,51}, disagg bootstrap 32000+100*role+{1,51} -- 50 apart, because
#   sglang reserves internal (IPC/NCCL) ports just above its http port, and 1 apart collided.
# The attention kernel is the same in every arm (persistent FA2/LPT unless KERNEL=split), so
# arms differ only in routing / placement -- the variable under test.
set -u
MODE=$1; LOG=$2
W=/dev/shm/w
ROLE=${ARNOLD_ID:-0}
mkdir -p $LOG
export PYTHONPATH=$W/python:$W/fi HF_HOME=$W/hf FI_ROOT=$W/fi TORCH_CUDA_ARCH_LIST=9.0a
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_FLASHINFER_USE_PERSISTENT=1
# mooncake (PD-disagg KV transfer) links libcudart.so.12; the pods' system toolkit is CUDA 13,
# but the torch cu128 wheels ship the 12.x runtime inside site-packages.
CUDART=$(python3 -c "import nvidia.cuda_runtime as m, os; print(os.path.dirname(m.__file__))" 2>/dev/null)
export LD_LIBRARY_PATH=$CUDART:$CUDART/lib:${LD_LIBRARY_PATH:-}
if [ "${KERNEL:-lpt}" = split ]; then
  export SGLANG_MIXED_ATTENTION_KERNEL=persistent SGLANG_TWO_EXEC=1 SGLANG_FUSED_FA3_SM_SPLIT=1
  export FLASHINFER_SM_PARTITION=1 UP_STAGES=2 SGLANG_FUSED_FA3_SRC=$W/fi/tests/fused_fa3_persistent_ext.cu
fi
# Free ports are FOUND, not assumed: pods use host networking on hosts shared with other
# jobs, which hold ports this container cannot even see (a bare bind of [::]:30301 failed with
# no sglang process alive). Each port is probed by binding it, 50 apart so sglang's internal
# ports above it stay clear, and published in ports_r<role>.txt for mn_run.py.
pick() { python3 - "$1" <<'PY'
import random, socket, sys
rng = random.Random(int(sys.argv[1]))
for _ in range(500):
    p = rng.randrange(40000, 59000, 50)
    try:
        for q in (p, p + 1, p + 2, p + 3):
            s = socket.socket(socket.AF_INET6); s.bind(("::", q)); s.close()
        print(p); break
    except OSError:
        continue
PY
}
: > $LOG/ports_r${ROLE}.txt
launch() {  # launch <gpu>: probe ports, start the worker, publish its ports
  local g=$1
  PORT=$(pick $((ROLE * 2 + g + RANDOM)))
  BOOT=$(pick $((ROLE * 2 + g + 7919 + RANDOM)))
  ARGS=(--model $W/qwen7b --attention-backend flashinfer --mem-fraction-static 0.85
        --host :: --port $PORT --context-length 131072 --watchdog-timeout 600
        --json-model-override-args '{"rope_scaling":{"factor":4.0,"original_max_position_embeddings":32768,"type":"yarn"}}')
  case $MODE in
    coloc) ARGS+=(--enable-mixed-chunk) ;;
    prefill|decode)
      IB=mlx5_$(( (ROLE * 2 + g) % 4 ))
      ARGS+=(--disaggregation-mode $MODE --disaggregation-transfer-backend mooncake
             --disaggregation-ib-device $IB --disaggregation-bootstrap-port $BOOT) ;;
  esac
  grep -v "^$g " $LOG/ports_r${ROLE}.txt > $LOG/ports_r${ROLE}.tmp 2>/dev/null
  echo "$g $PORT $BOOT" >> $LOG/ports_r${ROLE}.tmp && mv $LOG/ports_r${ROLE}.tmp $LOG/ports_r${ROLE}.txt
  CUDA_VISIBLE_DEVICES=$g setsid nohup $W/venv/bin/python -m sglang.launch_server "${ARGS[@]}" \
    > $LOG/worker_r${ROLE}_g${g}.log 2>&1 < /dev/null &
  echo $! > $LOG/pid_r${ROLE}_g${g}
  echo "worker role=$ROLE gpu=$g port=$PORT boot=$BOOT mode=$MODE pid=$!"
}
# Supervisor: a port probed free can still be taken by another job before sglang binds it
# (seen: role 3 gpu 1, port 57000). On a bind failure, kill that worker and relaunch it on a
# freshly probed port; mn_run.py re-reads ports_r<role>.txt on every health poll.
supervise() {
  local g=$1
  for attempt in 1 2 3 4 5; do
    launch $g
    for t in $(seq 1 90); do
      sleep 4
      grep -q "address already in use" $LOG/worker_r${ROLE}_g${g}.log && break
      grep -q "fired up" $LOG/worker_r${ROLE}_g${g}.log && return 0
    done
    grep -q "fired up" $LOG/worker_r${ROLE}_g${g}.log && return 0
    pkill -9 -P $(cat $LOG/pid_r${ROLE}_g${g}) 2>/dev/null; kill -9 $(cat $LOG/pid_r${ROLE}_g${g}) 2>/dev/null
    for p in $(ps -eo pid,cmd | grep "[s]glang" | grep -- "--port $PORT " | awk '{print $1}'); do kill -9 $p; done
    sleep 5
    mv $LOG/worker_r${ROLE}_g${g}.log $LOG/worker_r${ROLE}_g${g}.bindfail$attempt.log
  done
}
for g in 0 1; do
  supervise $g > $LOG/supervise_r${ROLE}_g${g}.log 2>&1 < /dev/null &
  disown
done
sleep 3
cat $LOG/ports_r${ROLE}.txt
