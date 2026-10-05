#!/bin/bash
# Unified queue vs baseline on the serving workload. The unified arm needs its own
# extension dir: its two -D flags change the kernel and torch would otherwise hand
# back the cached .so.
cd /dev/shm/w
busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [ -n "$busy" ]; then echo "REFUSING: GPU busy: $busy"; exit 1; fi
run_arm () {
  ARM=$1; UNIFIED=$2
  pkill -9 -f "launch[_]server" 2>/dev/null; pkill -9 -f "sglan[g]::" 2>/dev/null; sleep 5
  LOG=/dev/shm/w/lc_$ARM.log; rm -f "$LOG"
  ( . /dev/shm/w/venv/bin/activate
    export PYTHONPATH=/dev/shm/w/python:/dev/shm/w/fi HF_HOME=/dev/shm/w/hf
    export TORCH_EXTENSIONS_DIR=/dev/shm/w/tex_srv_$ARM
    export SGLANG_FLASHINFER_USE_PERSISTENT=1 SGLANG_MIXED_ATTENTION_KERNEL=persistent
    export SGLANG_FORWARD_MIX_EVERY=200
    export SGLANG_FUSED_FA3_SRC=/dev/shm/w/fi/tests/fused_fa3_persistent_ext.cu
    export FLASHINFER_FUSED_BACKFILL=0
    if [ "$UNIFIED" = "1" ]; then
      export SGLANG_FUSED_FA3_SM_SPLIT=1 SGLANG_FUSED_UNIFIED=1 SGLANG_FUSED_SM_FA3=8
      export FLASHINFER_SM_PARTITION=1 FLASHINFER_FA3_SM_SPLIT_CLUSTERS=8
    else
      export SGLANG_FUSED_FA3_SM_SPLIT=0
    fi
    exec python -m sglang.launch_server --model-path /dev/shm/w/qwen7b \
      --host 127.0.0.1 --port 31000 --mem-fraction-static 0.80 \
      --attention-backend flashinfer --enable-mixed-chunk \
      --chunked-prefill-size 512 --max-running-requests 64 \
      --disable-radix-cache --log-level info --decode-log-interval 40
  ) > "$LOG" 2>&1 &
  for i in $(seq 1 300); do grep -q "The server is fired up" "$LOG" && break; sleep 2; done
  if ! grep -q "The server is fired up" "$LOG"; then echo "$ARM: never came up"; tail -3 "$LOG"; return 1; fi
  ( . /dev/shm/w/venv/bin/activate
    export PYTHONPATH=/dev/shm/w/python:/dev/shm/w/fi HF_HOME=/dev/shm/w/hf
    python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 31000 \
      --dataset-name random-ids --random-input-len 65536 --random-output-len 256 \
      --num-prompts 16 --request-rate inf --max-concurrency 8 \
      --output-file /dev/shm/w/lc_$ARM.jsonl ) >> "$LOG" 2>&1
  echo "$ARM done: fused=$(grep -o 'mixed_fused_fa3=[0-9]*([0-9.]*%)' "$LOG" | tail -1)"
}
run_arm base 0
run_arm uni  1
pkill -9 -f "launch[_]server" 2>/dev/null
echo LCDONE
