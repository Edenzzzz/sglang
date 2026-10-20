#!/bin/bash
# Output gate for deadline-ordered decode rows (apply_decode_rows_edf.py) on the H200 pod, ours + MTP (4-mode, split,
# EDF, deadline chunk 1024/2048:6000). 160 ShareGPT prompts sent at once, greedy, 192 new tokens, so up to 160 rows run
# and DECODE_ROWS=16 parks most of them on most steps. Arms: base (no parking, cap 160), base_rep (noise floor: same
# config again -- batch composition differs run to run, so bf16 numerics may diverge), rows16 (parking on). The gate
# passes when rows16 agrees with base at least as well as base_rep does. cap16 (no parking, at most 16 running) is the
# null case for batch-size numerics: the same 16-row steps as rows16 without parking. ARMS / KEEP=1 rerun a subset
# against the saved base. Claude, 2026-10-20.
set -uo pipefail
W=/dev/shm/w
cd $W/tasks/pd_colocation
OUT=$W/gate_decode_rows_20261020; [ "${KEEP:-0}" = 1 ] || rm -rf $OUT; mkdir -p $OUT
export KERNEL=mtpsplit SPEC=1 EDF=1 PREEMPT=1 CHUNK=2048 EDF_UNCACHED=1 DEADLINE_CHUNK=1024:2048:6000 SM_FA3=112
export GPUS_PER_POD=1 TP=1 MEM_FRAC=0.85 TTFT_SLO_MS=2000
for arm in ${ARMS:-base base_rep rows16 cap16}; do
  bash mn_kill.sh >/dev/null 2>&1; sleep 5
  LOG=$OUT/$arm; mkdir -p $LOG
  if [ $arm = rows16 ]; then export DECODE_ROWS=16:20:24000:80; else unset DECODE_ROWS; fi
  export WORKER_EXTRA="--max-running-requests $([ $arm = cap16 ] && echo 16 || echo 160) --cuda-graph-max-bs-decode 48"
  bash mn_node.sh coloc $LOG > $LOG/node.out 2>&1
  for t in $(seq 1 200); do
    grep -q "fired up" $LOG/worker_r0_g0.log 2>/dev/null && break
    sleep 6
  done
  PORT=$(awk '{print $2}' $LOG/ports_r0_g0.txt)
  $W/venv/bin/python - "$PORT" "$OUT/$arm.json" <<'PY'
import json, random, sys, concurrent.futures as cf, urllib.request
port, out = sys.argv[1], sys.argv[2]
data = json.load(open("/dev/shm/w/workloads/ShareGPT_V3_unfiltered_cleaned_split.json"))
rng = random.Random(0)
prompts = [d["conversations"][0]["value"] for d in data if d.get("conversations") and len(d["conversations"][0]["value"]) > 40]
prompts = [p[:2000] for p in rng.sample(prompts, 160)]
def go(p):
    body = json.dumps({"text": p, "sampling_params": {"temperature": 0, "max_new_tokens": 192}}).encode()
    req = urllib.request.Request(f"http://[::1]:{port}/generate", body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())["text"]
with cf.ThreadPoolExecutor(160) as ex:
    texts = list(ex.map(go, prompts))
json.dump(texts, open(out, "w"))
print("done", len(texts))
PY
  echo "$arm tracebacks=$(grep -c Traceback $LOG/worker_r0_g0.log) decode_rows=$(grep DECODE_ROWS $LOG/worker_r0_g0.log | tail -1 | cut -c1-200) mixed=$(grep -c MIXEDSPEC_STEPS $LOG/worker_r0_g0.log) maxrun=$(grep -o 'max_running_requests=[0-9]*' $LOG/worker_r0_g0.log | head -1)"
done
bash mn_kill.sh >/dev/null 2>&1
$W/venv/bin/python - $OUT <<'PY'
import json, os, sys
d = sys.argv[1]
base = json.load(open(f"{d}/base.json"))
def cmp(name):
    o = json.load(open(f"{d}/{name}.json"))
    exact = sum(a == b for a, b in zip(base, o))
    pre = [next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))) for a, b in zip(base, o)]
    print(f"{name} vs base: exact {exact}/{len(base)}, mean common-prefix chars {sum(pre) / len(pre):.0f} "
          f"(mean output chars {sum(map(len, base)) / len(base):.0f})")
for n in ("base_rep", "rows16", "cap16"):
    if os.path.exists(f"{d}/{n}.json"):
        cmp(n)
PY
echo GATE_DECODE_ROWS_DONE
