#!/bin/bash
# Four arms: {fused sm_fa3=8, unfused} x {per-step trace on, off}.
# The trace arms carry CUDA-event timing, so their target is the step's GPU span
# and not the launch-to-launch interval. The no-trace arms are the control for
# how much the instrumentation itself costs each path -- the fused path turned
# out to be sensitive to host work, which is a result about the path.
cd /dev/shm/w
# Refuse to measure while another process holds the GPU. A detached
# compute-sanitizer run from an earlier investigation sat at 100% for three
# hours and cost every arm measured that day a factor of 2.45 in throughput,
# with the arms still consistent with each other -- so nothing in the A/B
# looked wrong. Absolute numbers are only comparable across runs if the device
# is exclusively ours, and that is cheap to assert.
busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [ -n "$busy" ]; then
  echo "REFUSING: GPU already in use by: $busy"
  exit 1
fi
run_arm () {
  ARM=$1; SPLIT=$2; SMFA3=$3; TRACE=$4; SYM=$5
  pkill -9 -f "launch[_]server" 2>/dev/null; pkill -9 -f "sglan[g]::" 2>/dev/null; sleep 5
  LOG=/dev/shm/w/tr3_$ARM.log
  rm -f "$LOG"
  ( . /dev/shm/w/venv/bin/activate
    export PYTHONPATH=/dev/shm/w/python:/dev/shm/w/fi
    export HF_HOME=/dev/shm/w/hf
    export SGLANG_FLASHINFER_USE_PERSISTENT=1
    export SGLANG_MIXED_ATTENTION_KERNEL=persistent
    export SGLANG_FORWARD_MIX_EVERY=200
    export SGLANG_FUSED_FA3_SM_SPLIT=$SPLIT
    export SGLANG_FUSED_SM_FA3=$SMFA3
    export SGLANG_FUSED_FA3_SRC=/dev/shm/w/fi/tests/fused_fa3_persistent_ext.cu
    export FLASHINFER_FUSED_BACKFILL=0
    export FLASHINFER_FUSED_SYMMETRIC=$SYM
    export SGLANG_STEP_TRACE=$TRACE
    exec python -m sglang.launch_server --model-path /dev/shm/w/qwen7b \
      --host 127.0.0.1 --port 31000 --mem-fraction-static 0.80 \
      --attention-backend flashinfer --enable-mixed-chunk \
      --chunked-prefill-size 512 --max-running-requests 64 \
      --disable-radix-cache --log-level info --decode-log-interval 40
  ) > "$LOG" 2>&1 &
  for i in $(seq 1 200); do grep -q "The server is fired up" "$LOG" && break; sleep 2; done
  if ! grep -q "The server is fired up" "$LOG"; then echo "$ARM: server never came up"; return 1; fi
  ( . /dev/shm/w/venv/bin/activate
    export PYTHONPATH=/dev/shm/w/python:/dev/shm/w/fi
    export HF_HOME=/dev/shm/w/hf
    python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 31000 \
      --dataset-name random-ids --random-input-len 8192 --random-output-len 192 \
      --num-prompts 200 --request-rate 12 --max-concurrency 32 \
      --output-file /dev/shm/w/tr3_$ARM.jsonl ) >> "$LOG" 2>&1
  echo "$ARM done: traced=$(grep -c 'gpu=' "$LOG") fused=$(grep -o 'mixed_fused_fa3=[0-9]*([0-9.]*%)' "$LOG" | tail -1)"
}
run_arm s8     1 8 1 1
run_arm base   0 8 1 0
run_arm s8nt   1 8 0 1
run_arm basent 0 8 0 0
pkill -9 -f "launch[_]server" 2>/dev/null
echo ALLDONE
