#!/bin/bash
cd /dev/shm/w
pkill -9 -f "launch[_]server" 2>/dev/null; pkill -9 -f "sglan[g]::" 2>/dev/null; sleep 5
rm -f /dev/shm/w/uni_fault.npz
LOG=/dev/shm/w/dmp.log; rm -f "$LOG"
( . /dev/shm/w/venv/bin/activate
  export PYTHONPATH=/dev/shm/w/python:/dev/shm/w/fi HF_HOME=/dev/shm/w/hf
  export TORCH_EXTENSIONS_DIR=/dev/shm/w/tex_srv_uni
  export SGLANG_FLASHINFER_USE_PERSISTENT=1 SGLANG_MIXED_ATTENTION_KERNEL=persistent
  export SGLANG_FORWARD_MIX_EVERY=20
  export SGLANG_FUSED_FA3_SRC=/dev/shm/w/fi/tests/fused_fa3_persistent_ext.cu
  export FLASHINFER_FUSED_BACKFILL=0
  export SGLANG_FUSED_FA3_SM_SPLIT=1 SGLANG_FUSED_UNIFIED=1 SGLANG_FUSED_SM_FA3=8
  export FLASHINFER_SM_PARTITION=1 FLASHINFER_FA3_SM_SPLIT_CLUSTERS=8
  export SGLANG_FUSED_DUMP=/dev/shm/w/uni_fault
  exec python -m sglang.launch_server --model-path /dev/shm/w/qwen7b \
    --host 127.0.0.1 --port 31000 --mem-fraction-static 0.80 \
    --attention-backend flashinfer --enable-mixed-chunk \
    --chunked-prefill-size 512 --max-running-requests 64 \
    --disable-radix-cache --log-level info --decode-log-interval 40
) > "$LOG" 2>&1 &
for i in $(seq 1 300); do grep -q "The server is fired up" "$LOG" && break; sleep 2; done
grep -q "The server is fired up" "$LOG" || { echo "never came up"; tail -3 "$LOG"; exit 1; }
( . /dev/shm/w/venv/bin/activate
  export PYTHONPATH=/dev/shm/w/python:/dev/shm/w/fi HF_HOME=/dev/shm/w/hf
  python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 31000 \
    --dataset-name random-ids --random-input-len 8192 --random-output-len 64 \
    --num-prompts 24 --request-rate 12 --max-concurrency 16 ) >> "$LOG" 2>&1
echo "dump: $(ls -la /dev/shm/w/uni_fault.npz 2>/dev/null | awk '{print $5}') bytes"
grep -oE "fused launch dump written.*" "$LOG" | head -1
grep -ciE "CUDA error|illegal|misaligned" "$LOG"
pkill -9 -f "launch[_]server" 2>/dev/null
echo DMPDONE
