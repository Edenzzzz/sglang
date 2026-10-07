#!/bin/bash
# Run the multi-node workload against a router on this pod (pod 0), streams released together.
#   mn_bench.sh <outdir> <router_port>
# Agent: TraceLab session replay WITH the recorded inter-turn gaps (scaled by SLOWDOWN),
# SESSIONS sessions arriving at SESSION_RATE/s; each turn's prompt extends the previous one,
# so a turn hits cache only where the session's previous turn ran. Chat: random-ids, CHAT_RATE/s.
set -u
OUT=$1; RPORT=$2
W=/dev/shm/w
export PYTHONPATH=$W/python:$W/fi
mkdir -p $OUT; rm -rf $OUT/barrier
export BENCH_BARRIER_DIR=$OUT/barrier
S=${SESSIONS:-64}; T=${TURNS:-6}
$W/venv/bin/python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port $RPORT --model $W/qwen7b \
  --dataset-name tracelab-claude --dataset-path $W/tracelab_trace.jsonl.gz --tokenize-prompt \
  --tracelab-num-groups $S --tracelab-min-context-len ${MIN_CTX:-16384} --tracelab-max-context-len ${MAX_CTX:-49152} \
  --tracelab-min-output-len 64 --num-prompts $((S * T)) --request-rate ${SESSION_RATE:-0.5} \
  --use-trace-timestamps --mooncake-slowdown-factor ${SLOWDOWN:-0.1} \
  --warmup-requests 2 --output-file $OUT/agent.json --output-details --disable-tqdm > $OUT/bench_agent.log 2>&1 &
A=$!
$W/venv/bin/python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port $RPORT --model $W/qwen7b \
  --dataset-name random-ids --random-input ${CHAT_IN:-512} --random-output ${CHAT_OUT:-128} --random-range-ratio 0.5 \
  --request-rate ${CHAT_RATE:-16} --num-prompts ${CHAT_N:-3200} \
  --warmup-requests 2 --output-file $OUT/chat.json --output-details --disable-tqdm > $OUT/bench_chat.log 2>&1 &
C=$!
for i in $(seq 1 3000); do
  n=$(ls $BENCH_BARRIER_DIR 2>/dev/null | grep -c ready_)
  [ "$n" -ge 2 ] && break
  kill -0 $A 2>/dev/null && kill -0 $C 2>/dev/null || break
  sleep 0.2
done
mkdir -p $BENCH_BARRIER_DIR && touch $BENCH_BARRIER_DIR/go
echo "released: $n/2 ready"
wait $A $C
echo MNBENCHDONE
