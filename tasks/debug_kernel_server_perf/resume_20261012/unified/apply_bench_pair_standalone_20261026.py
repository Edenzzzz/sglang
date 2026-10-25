"""Add the standalone branch to a run root's mn_bench_pair.sh: with STANDALONE_PATH set, the only client is one
ShareGPT-format dataset (sharegpt loader, chat template if CHAT_TEMPLATE=1) at CHAT_RATE for CHAT_N requests, released
by the same barrier (one ready client), output in chat.json. Idempotent.
  python apply_bench_pair_standalone_20261026.py <run root>
Claude, 2026-10-26."""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "mn_bench_pair.sh"
s = p.read_text()
if "STANDALONE_PATH" not in s:
    anchor = 'S=${SESSIONS:-64}; T=${TURNS:-6}\n'
    assert anchor in s, "mn_bench_pair.sh layout changed"
    block = r'''if [ -n "${STANDALONE_PATH:-}" ]; then
  ST=(--dataset-name sharegpt --dataset-path $STANDALONE_PATH)
  [ "${CHAT_TEMPLATE:-0}" = 1 ] && ST+=(--apply-chat-template)
  $W/venv/bin/python -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port $RPORT --model $W/${MODEL_DIR:-qwen7b} \
    "${ST[@]}" --request-rate ${CHAT_RATE:?} --num-prompts ${CHAT_N:?} \
    --warmup-requests 2 --output-file $OUT/chat.json --output-details --disable-tqdm > $OUT/bench_chat.log 2>&1 &
  C=$!
  for i in $(seq 1 3000); do
    n=$(ls $BENCH_BARRIER_DIR 2>/dev/null | grep -c ready_)
    [ "$n" -ge 1 ] && break
    kill -0 $C 2>/dev/null || break
    sleep 0.2
  done
  test "$n" -ge 1 || exit 2
  wc -c < "$SERVING_SERVER_LOG" > "$OUT/server_log_offset"
  touch $BENCH_BARRIER_DIR/go
  echo "released: $n/1 ready (standalone)"
  wait $C; rc=$?
  echo MNBENCHDONE
  exit $rc
fi
'''
    s = s.replace(anchor, anchor + block, 1)
    p.write_text(s)
print("STANDALONE_BRANCH", "STANDALONE_PATH" in p.read_text())
