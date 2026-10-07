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
for g in 0 1; do
  PORT=$((30000 + 100 * ROLE + 1 + 50 * g))
  BOOT=$((32000 + 100 * ROLE + 1 + 50 * g))
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
  CUDA_VISIBLE_DEVICES=$g setsid nohup $W/venv/bin/python -m sglang.launch_server "${ARGS[@]}" \
    > $LOG/worker_r${ROLE}_g${g}.log 2>&1 < /dev/null &
  echo "worker role=$ROLE gpu=$g port=$PORT boot=$BOOT mode=$MODE pid=$!"
done
