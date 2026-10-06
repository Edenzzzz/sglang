#!/bin/bash
# Long-context serving A/B: the two-executor split against FA2-on-everything.
#
# Replaces lcab.sh for this purpose: lcab.sh drives SGLANG_FUSED_UNIFIED, which is the
# CLOSED fused dispatch. Here the arms are SGLANG_TWO_EXEC=0/1.
#
# Context length is forced past the model's 32k so the appends are genuinely long. That
# extrapolates RoPE, so OUTPUT QUALITY IS NOT VALIDATED by this run -- it measures kernel
# and scheduler time on realistic shapes, which is what the comparison needs. Any claim
# about generation quality needs a model with native long context.
cd /dev/shm/w
busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [ -n "$busy" ]; then echo "REFUSING: GPU busy: $busy"; exit 1; fi

PAGE=${PAGE:-1}
IN=${IN:-65536}
OUT=${OUT:-256}
NP=${NP:-16}
CONC=${CONC:-8}

run_arm () {
  ARM=$1; TWO=$2
  for p in $(pgrep -f "launch_server --model-path /dev/shm/w/qwen7b" 2>/dev/null); do kill -9 $p; done
  sleep 5
  LOG=/dev/shm/w/srv_$ARM.log; rm -f "$LOG"
  ( . /dev/shm/w/venv/bin/activate
    export PYTHONPATH=/dev/shm/w/python:/dev/shm/w/fi HF_HOME=/dev/shm/w/hf
    export TORCH_EXTENSIONS_DIR=/dev/shm/w/tex_srv2_$ARM
    export SGLANG_FLASHINFER_USE_PERSISTENT=1 SGLANG_MIXED_ATTENTION_KERNEL=persistent
    export SGLANG_FUSED_FA3_SRC=/dev/shm/w/fi/tests/fused_fa3_persistent_ext.cu
    export SGLANG_TWO_EXEC=$TWO
    # Qwen2.5-7B is natively 32K; sglang refuses a longer --context-length unless this is set.
    # Extrapolated RoPE: timings are valid, generated text is not (see header).
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
    # The FA2 half runs flashinfer's JIT module, which caches across header edits; a stale
    # module silently ignores source changes (see task doc). Each arm starts clean.
    rm -rf ~/.cache/flashinfer/90a/cached_ops
    # upstream's own trait values, as measured: see the trait A/B in the task doc
    export UP_STAGES=2 MMA_PV_RS=1 SCHED_LPT=0
    # a silent fallback would make this an apples-to-oranges comparison, so it must raise
    unset SGLANG_FUSED_ALLOW_FALLBACK
    exec python -m sglang.launch_server --model-path /dev/shm/w/qwen7b \
      --host 127.0.0.1 --port 31000 --mem-fraction-static 0.80 \
      --attention-backend flashinfer --enable-mixed-chunk \
      --context-length 131072 --page-size $PAGE \
      --chunked-prefill-size 2048 --max-running-requests 64 \
      --disable-radix-cache --log-level info --decode-log-interval 40
  ) > "$LOG" 2>&1 &
  for i in $(seq 1 400); do grep -q "The server is fired up" "$LOG" && break; sleep 2; done
  if ! grep -q "The server is fired up" "$LOG"; then
    echo "=== $ARM: server never came up"; tail -20 "$LOG"; return 1
  fi
  ( . /dev/shm/w/venv/bin/activate
    export PYTHONPATH=/dev/shm/w/python:/dev/shm/w/fi HF_HOME=/dev/shm/w/hf
    python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 31000 \
      --dataset-name random-ids --random-input-len $IN --random-output-len $OUT \
      --random-range-ratio 1.0 \
      --num-prompts $NP --request-rate inf --max-concurrency $CONC \
      --output-file /dev/shm/w/srv_$ARM.jsonl ) >> "$LOG" 2>&1
  echo "=== $ARM (TWO_EXEC=$TWO, page=$PAGE, in=$IN out=$OUT n=$NP conc=$CONC)"
  grep -iE "Mean TTFT|Median TTFT|Mean TPOT|Median TPOT|Mean ITL|Median ITL|Output token throughput|Request throughput|Total token throughput" "$LOG" | tail -12
  # VALIDITY: a fallback invalidates the arm. Report the count beside the timings always.
  echo "fallbacks: $(grep -ci 'fallback' "$LOG")   two_exec_calls: $(grep -co 'two_exec' "$LOG")"
}

run_arm base 0
run_arm split 1
for p in $(pgrep -f "launch_server --model-path /dev/shm/w/qwen7b" 2>/dev/null); do kill -9 $p; done
echo SRVABDONE
