#!/bin/bash
# Smoke + greedy-output check for PD multiplexing under EAGLE3 MTP (tree up_pdmux_mtp). Two servers in turn, same
# requests, temperature 0: (a) stock EAGLE3 MTP, (b) EAGLE3 MTP + --enable-pdmux, eager (CUDA graphs off in both, so
# the comparison isolates the pdmux path). 24 concurrent requests: 8 long prompts (12-24K tokens, split prefill on the
# prefill SM group) interleaved with 16 short ones, 128 output tokens each; then token-level agreement per request.
# Claude, 2026-10-17.
set -uo pipefail
W=/dev/shm/w
exec 9>"$W/a2_kernel.lock"; flock 9
pgrep -f '[s]glang.launch_server' >/dev/null && { echo "GPU predecessor still running"; exit 1; }
out=$W/pdmux_mtp_smoke_20261017; mkdir -p $out
export PYTHONPATH=$W/up_pdmux_mtp/python HF_HOME=$W/hf
COMMON="--model $W/qwen7b --attention-backend flashinfer --mem-fraction-static 0.8 --host 127.0.0.1 --port 31977
  --context-length 32768 --disable-cuda-graph --speculative-algorithm EAGLE3
  --speculative-draft-model-path $W/qwen7b_eagle3_sgl --speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4"
run() {  # run <label> <extra args>
  local label=$1; shift
  $W/venv/bin/python -m sglang.launch_server $COMMON "$@" > $out/server_$label.log 2>&1 &
  local pid=$!
  for t in $(seq 1 120); do sleep 5; curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:31977/health | grep -q 200 && break
    kill -0 $pid 2>/dev/null || break; done
  if ! curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:31977/health | grep -q 200; then
    echo "SMOKE $label server_not_up tracebacks=$(grep -c Traceback $out/server_$label.log)"; kill -9 $pid 2>/dev/null; return 1; fi
  $W/venv/bin/python - "$out/out_$label.json" <<'PY'
import json, random, sys, concurrent.futures as cf, urllib.request
rng = random.Random(1017)
reqs = []
for i in range(24):
    n = rng.randrange(12000, 24000) if i % 3 == 0 else rng.randrange(200, 1500)
    reqs.append([rng.randrange(1000, 30000) for _ in range(n)])
def go(ids):
    body = {"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 128, "ignore_eos": True}}
    r = urllib.request.Request("http://127.0.0.1:31977/generate", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as f:
        return json.loads(f.read())["output_ids"]
with cf.ThreadPoolExecutor(24) as ex:
    outs = list(ex.map(go, reqs))
json.dump(outs, open(sys.argv[1], "w"))
print("done", len(outs))
PY
  local rc=$?
  echo "SMOKE $label rc=$rc tracebacks=$(grep -c Traceback $out/server_$label.log) pdmux_lines=$(grep -c 'PD-Multiplexing enabled' $out/server_$label.log)"
  kill $pid; sleep 10; kill -9 $pid 2>/dev/null; pkill -9 -f 'sglang::' 2>/dev/null; sleep 5
}
run mtp
run mtp_pdmux --enable-pdmux --chunked-prefill-size -1 --disable-overlap-schedule
$W/venv/bin/python - $out <<'PY'
import json, sys
d = sys.argv[1]
try:
    a = json.load(open(f"{d}/out_mtp.json")); b = json.load(open(f"{d}/out_mtp_pdmux.json"))
except Exception as e:
    print("COMPARE missing", e); sys.exit()
common = []
for x, y in zip(a, b):
    k = 0
    while k < min(len(x), len(y)) and x[k] == y[k]: k += 1
    common.append(k)
print("COMPARE identical", sum(c == len(x) for c, x in zip(common, a)), "of", len(a), "min_common_prefix", min(common),
      "median", sorted(common)[len(common)//2])
PY
echo SMOKE_DONE
