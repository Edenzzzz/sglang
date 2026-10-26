#!/bin/bash
# Run the multi-node workload against a router on this pod (pod 0), streams released together.
#   mn_bench.sh <outdir> <router_port>
# Agent: TraceLab session replay WITH the recorded inter-turn gaps (scaled by SLOWDOWN),
# SESSIONS sessions arriving at SESSION_RATE/s; each turn's prompt extends the previous one,
# so a turn hits cache only where the session's previous turn ran. Chat: random-ids, CHAT_RATE/s.
set -u
ulimit -n $(ulimit -Hn)  # 1024 soft limit: a burst of concurrent requests runs the client out of sockets
OUT=$1; RPORT=$2
W=/dev/shm/w
export PYTHONPATH=$W/python:$W/fi
mkdir -p $OUT; rm -rf $OUT/barrier
export BENCH_BARRIER_DIR=$OUT/barrier
S=${SESSIONS:-64}; T=${TURNS:-6}
if [ "${WORKLOAD:-}" = fanout ]; then
  # PD point 2b: fan-out agents on shared long task prefixes with rotating popularity (drivers/serving/fanout_agents.py).
  # The driver marks itself ready in the release barrier once its prefixes are built (tokenizing takes ~1-2 min).
  $W/venv/bin/python $W/drivers/serving/fanout_agents.py --url http://127.0.0.1:$RPORT --tasks ${FO_TASKS:-8} \
    --tokenizer $W/${MODEL_DIR:-qwen7b} \
    --prefix-len ${FO_PREFIX:-65536} --concurrency ${AGENT_CONC:-16} --rounds ${FO_ROUNDS:-4} \
    --append-len ${FO_APPEND:-1024} --output-len ${FO_OUT:-128} --tool-gap-s ${FO_GAP:-5} --zipf ${FO_ZIPF:-1.2} \
    --rotate-s ${FO_ROTATE:-180} --duration ${FO_DURATION:-900} --out $OUT/fanout.json \
    ${FO_TEXT:+--text-source $W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json} > $OUT/bench_fanout.log 2>&1 &
  FO_BASE=$!
  # FO_SPIKE="delay_s:conc:duration_s" (5e mixture shift): a second fan-out driver on the same task prefixes (own seed,
  # so its appends and task draws differ) adds `conc` programs `delay_s` after its prefixes are built, for duration_s.
  if [ -n "${FO_SPIKE:-}" ]; then
    IFS=: read -r FS_DELAY FS_CONC FS_DUR <<< "$FO_SPIKE"
    BENCH_BARRIER_DIR= $W/venv/bin/python $W/drivers/serving/fanout_agents.py --url http://127.0.0.1:$RPORT \
      --tokenizer $W/${MODEL_DIR:-qwen7b} \
      --tasks ${FO_TASKS:-8} --prefix-len ${FO_PREFIX:-65536} --concurrency $FS_CONC --rounds ${FO_ROUNDS:-4} \
      --append-len ${FO_APPEND:-1024} --output-len ${FO_OUT:-128} --tool-gap-s ${FO_GAP:-5} --zipf ${FO_ZIPF:-1.2} \
      --rotate-s ${FO_ROTATE:-180} --duration $FS_DUR --start-delay $FS_DELAY --seed 7 --out $OUT/fanout_spike.json \
      ${FO_TEXT:+--text-source $W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json} > $OUT/bench_fanout_spike.log 2>&1 &
    FO_SP=$!
  fi
elif [ "${AGENT_MODE:-}" = tracelab_ts ]; then
  # WINDOW_STARTS="s1,s2,...": superpose those one-hour segments on one clock (each timed from its own start), so the
  # replay carries many developers' sessions at once; mutually exclusive with WINDOW_START.
  # Crossflow-style timestamp replay: every session whose first eligible round starts in
  # [WINDOW_START, WINDOW_START+WINDOW_S) trace seconds with all its rounds (WINDOW_ROUNDS=segment: only rounds
  # starting inside the window); offsets and gaps divided by ALPHA;
  # round r goes out at max(finish_{r-1} + gap_r/ALPHA, offset_r/ALPHA). Contents are tokenized ShareGPT text
  # (TEXT_SOURCE; TEXT_SOURCE=random keeps random ids). Measured programs/starts/calls go to agent_programs.json.
  if [ -n "${WINDOW_STARTS:-}" ]; then WINDOW_ARGS=(--tracelab-window-starts "$WINDOW_STARTS")
  else WINDOW_ARGS=(--tracelab-window-start-s "${WINDOW_START:?AGENT_MODE=tracelab_ts needs WINDOW_START or WINDOW_STARTS}"); fi
  TS_TEXT=${TEXT_SOURCE:-$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json}
  [ "$TS_TEXT" = random ] && TS_TEXT=
  export BENCH_SESSION_AUDIT_FILE=$OUT/agent_programs.json
  $W/venv/bin/python -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port $RPORT --model $W/${MODEL_DIR:-qwen7b} \
    --dataset-name tracelab-claude --dataset-path "${TRACE_DATASET_PATH:-$W/workloads/tracelab/syfi_full_min.jsonl.gz}" --tokenize-prompt \
    --tracelab-min-context-len ${MIN_CTX:-0} --tracelab-max-context-len ${MAX_CTX:-65536} --tracelab-min-output-len 1 \
    --tracelab-providers claude,codex --tracelab-allow-empty-prefix \
    "${WINDOW_ARGS[@]}" --tracelab-window-s ${WINDOW_S:-3600} \
    --tracelab-time-scale ${ALPHA:-1} --tracelab-window-rounds ${WINDOW_ROUNDS:-session} \
    ${TS_TEXT:+--tracelab-text-source "$TS_TEXT"} --request-rate inf \
    --warmup-requests 2 --output-file $OUT/agent.json --output-details --disable-tqdm > $OUT/bench_agent.log 2>&1 &
  unset BENCH_SESSION_AUDIT_FILE
elif [ -n "${AGENT_CONC:-}" ]; then
  # Program concurrency (ThunderAgent protocol; the only agent load axis -- user, 2026-10-15): AGENT_CONC
  # sessions kept active, the next starts when one finishes, rounds back to back. Same TraceLab
  # cohort selection as the single-GPU kernel_concurrency_env (providers, round weighting, seed).
  export BENCH_SESSION_CONCURRENCY=$AGENT_CONC BENCH_SESSION_AUDIT_FILE=$OUT/agent_programs.json
  $W/venv/bin/python -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port $RPORT --model $W/${MODEL_DIR:-qwen7b} \
    --dataset-name tracelab-claude --dataset-path "${TRACE_DATASET_PATH:-$W/workloads/tracelab/syfi_full_min.jsonl.gz}" --tokenize-prompt \
    --tracelab-num-groups $S --tracelab-min-context-len ${MIN_CTX:-32768} --tracelab-max-context-len ${MAX_CTX:-65536} \
    --tracelab-min-output-len 1 --num-prompts $((S * T)) --request-rate inf \
    --tracelab-providers claude,codex --tracelab-allow-empty-prefix --tracelab-session-weighting rounds \
    --tracelab-selection-seed ${AGENT_SEED:-20261014} \
    --warmup-requests 2 --output-file $OUT/agent.json --output-details --disable-tqdm > $OUT/bench_agent.log 2>&1 &
  unset BENCH_SESSION_CONCURRENCY BENCH_SESSION_AUDIT_FILE
else
$W/venv/bin/python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port $RPORT --model $W/${MODEL_DIR:-qwen7b} \
  --dataset-name tracelab-claude --dataset-path $W/tracelab_trace.jsonl.gz --tokenize-prompt \
  --tracelab-num-groups $S --tracelab-min-context-len ${MIN_CTX:-16384} --tracelab-max-context-len ${MAX_CTX:-49152} \
  --tracelab-min-output-len 64 --num-prompts $((S * T)) --request-rate ${SESSION_RATE:-0.5} \
  --use-trace-timestamps --mooncake-slowdown-factor ${SLOWDOWN:-0.1} \
  --warmup-requests 2 --output-file $OUT/agent.json --output-details --disable-tqdm > $OUT/bench_agent.log 2>&1 &
fi
A=$!
# HOT=1 (point 4): the second tenant shares a few long system prompts instead of sending fresh
# ones, so pure cache affinity piles each group onto its single holder. Spreading that load needs
# either the SLO check plus recompute, or the SLO check plus a cross-node pull from the L3 store.
if [ "${HOT:-0}" = 1 ]; then
  SECOND=(--dataset-name generated-shared-prefix --gsp-num-groups ${HOT_GROUPS:-4}
          --gsp-prompts-per-group $(( ${CHAT_N:-3200} / ${HOT_GROUPS:-4} )) --gsp-system-prompt-len ${HOT_PREFIX:-16384}
          --gsp-question-len ${CHAT_IN:-512} --gsp-output-len ${CHAT_OUT:-128})
elif [ "${CHAT_SHAREGPT:-0}" = 1 ]; then
  SECOND=(--dataset-name sharegpt --dataset-path $W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json)
  # CHAT_TEMPLATE=1: wrap each chat prompt in the model's chat template, as a served chat request would be -- raw text
  # (bench_serving's default) understates the draft head's acceptance (Llama-70B EAGLE3 accepted 1.3, 2026-10-21).
  [ "${CHAT_TEMPLATE:-0}" = 1 ] && SECOND+=(--apply-chat-template)
else
  SECOND=(--dataset-name random-ids --random-input ${CHAT_IN:-512} --random-output ${CHAT_OUT:-128} --random-range-ratio 0.5)
fi
$W/venv/bin/python -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port $RPORT --model $W/${MODEL_DIR:-qwen7b} \
  "${SECOND[@]}" --request-rate ${CHAT_RATE:-16} --num-prompts ${CHAT_N:-3200} \
  --warmup-requests 2 --output-file $OUT/chat.json --output-details --disable-tqdm > $OUT/bench_chat.log 2>&1 &
C=$!
# MATH_RATE (optional third tenant): math problems in ShareGPT format from drivers/serving/prep_standalone.py
# (MATH_SET=math500|gsm8k under $W/workloads/standalone), replayed by the sharegpt loader with each output length
# from its reference solution; own count, seed and directory $OUT/math.
M=; NREADY=2
if [ -n "${MATH_RATE:-}" ]; then
  mkdir -p $OUT/math
  MATH_ARGS=(--dataset-name sharegpt --dataset-path ${MATH_PATH:-$W/workloads/standalone/${MATH_SET:-math500}.json})
  [ "${CHAT_TEMPLATE:-0}" = 1 ] && MATH_ARGS+=(--apply-chat-template)
  $W/venv/bin/python -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port $RPORT --model $W/${MODEL_DIR:-qwen7b} \
    "${MATH_ARGS[@]}" --request-rate $MATH_RATE --num-prompts ${MATH_N:-500} --seed ${MATH_SEED:-3} \
    --warmup-requests 2 --output-file $OUT/math/math.json --output-details --disable-tqdm > $OUT/math/bench_math.log 2>&1 &
  M=$!; NREADY=3
fi
for i in $(seq 1 3000); do
  n=$(ls $BENCH_BARRIER_DIR 2>/dev/null | grep -c ready_)
  [ "$n" -ge $NREADY ] && break
  kill -0 $A 2>/dev/null && kill -0 $C 2>/dev/null && { [ -z "$M" ] || kill -0 $M 2>/dev/null; } || break
  sleep 0.2
done
mkdir -p $BENCH_BARRIER_DIR && touch $BENCH_BARRIER_DIR/go
echo "released: $n/$NREADY ready"
# CHAT_SPIKE="delay_s:rate:n" (5e load spike): delay_s after release a second chat stream at `rate` req/s sends n
# requests on top of the base load; its own seed so it does not replay the base stream's prompts, no barrier.
S=
if [ -n "${CHAT_SPIKE:-}" ]; then
  IFS=: read -r SP_DELAY SP_RATE SP_N <<< "$CHAT_SPIKE"
  ( sleep $SP_DELAY; date +%s.%N > $OUT/chat_spike.start
    BENCH_BARRIER_DIR= $W/venv/bin/python -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port $RPORT \
      --model $W/${MODEL_DIR:-qwen7b} "${SECOND[@]}" --seed 7 --request-rate $SP_RATE --num-prompts $SP_N --warmup-requests 0 \
      --output-file $OUT/chat_spike.json --output-details --disable-tqdm ) > $OUT/bench_chat_spike.log 2>&1 &
  S=$!
fi
wait_status=0
for client_pid in $A $C $S $M ${FO_BASE:-} ${FO_SP:-}; do
  wait "$client_pid" || wait_status=$?
done
[ "$wait_status" -eq 0 ] || exit "$wait_status"
echo MNBENCHDONE
