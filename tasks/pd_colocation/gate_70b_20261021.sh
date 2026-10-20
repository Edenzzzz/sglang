#!/bin/bash
# Output gate for Llama-3.3-70B at TP4 (one 4-GPU H20 pod, job f3b4627e0864b06f, 2026-10-21) before any 70B serving
# number: stock MTP (trained EAGLE3 head), the same again (noise floor: batch composition differs run to run), and ours
# (4-mode MTP step on the split kernel, EDF + chunk preemption, deadline chunk). 96 ShareGPT prompts and 8 long prompts
# (32K tokens of TraceLab context) sent at once, greedy, 128 new tokens, so the long prefills ride the verify steps of
# the short ones. Pass: ours agrees with stock at least as well as stock agrees with itself, 0 tracebacks, and the
# split / 4-mode counters fire. Claude.
set -uo pipefail
W=/dev/shm/w
cd $W/tasks/pd_colocation
OUT=$W/gate_70b_20261021; [ "${KEEP:-0}" = 1 ] || rm -rf $OUT; mkdir -p $OUT
export GPUS_PER_POD=4 TP=4 MODEL_DIR=llama70b DRAFT_DIR=llama70b_eagle3_sgl MEM_FRAC=0.85 CTX_LEN=131072 SM_FA3=${SM_FA3:-66}
for arm in ${ARMS:-stock stock_rep ours}; do
  bash mn_kill.sh >/dev/null 2>&1; sleep 5
  LOG=$OUT/$arm; mkdir -p $LOG
  if [ $arm = ours ]; then
    env KERNEL=mtpsplit SPEC=1 EDF=1 PREEMPT=1 CHUNK=2048 EDF_UNCACHED=1 DEADLINE_CHUNK=1024:2048:6000 \
      bash mn_node.sh coloc $LOG > $LOG/node.out 2>&1
  else
    env KERNEL=lpt SPEC=2 bash mn_node.sh coloc $LOG > $LOG/node.out 2>&1
  fi
  for t in $(seq 1 300); do
    grep -q "fired up" $LOG/worker_r0_g0.log 2>/dev/null && break
    grep -q "Traceback" $LOG/worker_r0_g0.log 2>/dev/null && break
    sleep 6
  done
  PORT=$(awk '{print $2}' $LOG/ports_r0_g0.txt)
  $W/venv/bin/python - "$PORT" "$OUT/$arm.json" <<'PY'
import gzip, json, random, sys, concurrent.futures as cf, urllib.request
port, out = sys.argv[1], sys.argv[2]
data = json.load(open("/dev/shm/w/workloads/ShareGPT_V3_unfiltered_cleaned_split.json"))
rng = random.Random(0)
short = [d["conversations"][0]["value"][:2000] for d in data if d.get("conversations") and len(d["conversations"][0]["value"]) > 40]
short = rng.sample(short, 96)
ctx = []
for line in gzip.open("/dev/shm/w/workloads/tracelab/syfi_full_min.jsonl.gz", "rt"):
    ctx.append(line)
    if sum(map(len, ctx)) > 130000 * 8:
        break
blob = "".join(ctx)
long_ = [blob[i * 130000:(i + 1) * 130000] + "\nSummarize the above." for i in range(8)]  # ~32K tokens each
def go(p):
    body = json.dumps({"text": p, "sampling_params": {"temperature": 0, "max_new_tokens": 128}}).encode()
    req = urllib.request.Request(f"http://[::1]:{port}/generate", body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())["text"]
with cf.ThreadPoolExecutor(104) as ex:
    texts = list(ex.map(go, long_ + short))
json.dump(texts, open(out, "w"))
print("done", len(texts))
PY
  L=$LOG/worker_r0_g0.log
  echo "$arm tracebacks=$(grep -c Traceback $L) split=$(grep -o 'FUSED_SPLIT_CALLS [0-9]*' $L | tail -1) mixed=$(grep -o 'MIXEDSPEC_STEPS [0-9]*' $L | tail -1) accept=$(grep -o 'accept len: [0-9.]*' $L | tail -1)"
done
bash mn_kill.sh >/dev/null 2>&1
$W/venv/bin/python - $OUT <<'PY'
import json, os, sys
d = sys.argv[1]
base = json.load(open(f"{d}/stock.json"))
for n in ("stock_rep", "ours"):
    if not os.path.exists(f"{d}/{n}.json"):
        continue
    o = json.load(open(f"{d}/{n}.json"))
    for name, sl in (("long", slice(0, 8)), ("short", slice(8, None))):
        a, b = base[sl], o[sl]
        exact = sum(x == y for x, y in zip(a, b))
        pre = [next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y))) for x, y in zip(a, b)]
        print(f"{n} vs stock, {name}: exact {exact}/{len(a)}, mean common prefix {sum(pre) / len(pre):.0f} of "
              f"{sum(map(len, a)) / len(a):.0f} chars")
PY
echo GATE_70B_DONE
