#!/bin/bash
# Smoke the ported 4-mode step on CURRENT UPSTREAM. No fused extension needed: the 4-mode step is
# a scheduling/batching change on the default attention backend; the split kernel is a separate
# layer beneath it. Earlier claim that this was blocked on the torch 2.14 rebuild was WRONG.
set -u
W=/dev/shm/w
V=$W/venv_up
# the ported files overwrite the installed package's copies
SP=$($V/bin/python -c "import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))")
cp $W/up/python/sglang/srt/managers/scheduler.py $SP/sglang/srt/managers/
cp $W/up/python/sglang/srt/managers/schedule_batch.py $SP/sglang/srt/managers/
cp $W/up/python/sglang/srt/speculative/eagle_worker_v2.py $SP/sglang/srt/speculative/
echo "patched into $SP"
O=$W/portchk; rm -rf $O; mkdir -p $O
export HF_HOME=$W/hf CUDA_VISIBLE_DEVICES=0 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_MIXED_SPEC=1
ulimit -n $(ulimit -Hn)
$V/bin/python -m sglang.launch_server --model $W/qwen7b --port 31002 --host 127.0.0.1 \
  --enable-mixed-chunk --chunked-prefill-size 2048 --mem-fraction-static 0.75 \
  --speculative-algorithm EAGLE3 --speculative-draft-model-path $W/qwen7b_eagle3_sgl \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --max-running-requests 32 > $O/server.log 2>&1 &
S=$!
UP=0
for i in $(seq 1 100); do sleep 5; curl -sf http://127.0.0.1:31002/health >/dev/null && { UP=1; break; }; kill -0 $S 2>/dev/null || break; done
if [ "$UP" != 1 ]; then echo "SERVER DID NOT COME UP"; grep -iE "error|Traceback|Exception" $O/server.log | tail -8 | cut -c1-200; echo PORTCHKDONE; exit 1; fi
echo "server up; driving a mixed load"
$V/bin/python - <<'PY' > $O/load.log 2>&1
import concurrent.futures as cf, json, random, time, urllib.request
U="http://127.0.0.1:31002/generate"; rng=random.Random(0)
def go(n,out,_):
    b={"input_ids":[rng.randrange(1000,30000) for _ in range(n)],
       "sampling_params":{"temperature":0,"max_new_tokens":out,"ignore_eos":True}}
    r=urllib.request.Request(U,data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
    return len(json.loads(urllib.request.urlopen(r,timeout=600).read())["text"])
with cf.ThreadPoolExecutor(16) as ex:
    f=[ex.submit(go,512,400,i) for i in range(6)]
    time.sleep(3)
    f+=[ex.submit(go,6000,32,i) for i in range(4)]
    print("ok", [x.result() for x in f][:3])
PY
tail -4 $O/load.log | cut -c1-200
grep -icE "Traceback|Exception|AssertionError" $O/server.log
grep -E "Traceback|AssertionError|Error" $O/server.log | tail -5 | cut -c1-220
kill $S 2>/dev/null
echo PORTCHKDONE
