#!/bin/bash
# THE GATING CHECK for the upstream port: with --enable-mixed-chunk + EAGLE3, does upstream put a
# prefill chunk and VERIFY rows in one target forward (= our 4-mode step already exists upstream),
# or only a chunk + 1-token decode tails (= the 4-mode delta is still ours)?
set -u
W=/dev/shm/w
export PYTHONPATH= HF_HOME=$W/hf CUDA_VISIBLE_DEVICES=0
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
O=$W/compcheck; rm -rf $O; mkdir -p $O
ulimit -n $(ulimit -Hn)
$W/venv_up/bin/python -m sglang.launch_server --model $W/qwen7b --port 31001 --host 127.0.0.1 \
  --enable-mixed-chunk --chunked-prefill-size 2048 --mem-fraction-static 0.75 \
  --speculative-algorithm EAGLE3 --speculative-draft-model-path $W/qwen7b_eagle3_sgl \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --max-running-requests 32 > $O/server.log 2>&1 &
S=$!
for i in $(seq 1 120); do sleep 5; curl -sf http://127.0.0.1:31001/health >/dev/null && break; kill -0 $S 2>/dev/null || { echo "SERVER DIED"; tail -25 $O/server.log; echo COMPCHECKDONE; exit 1; }; done
$W/venv_up/bin/python - <<'PY' > $O/load.log 2>&1
import concurrent.futures as cf, json, random, time, urllib.request
U="http://127.0.0.1:31001/generate"
rng=random.Random(0)
def go(n_tok, out, _):
    b={"input_ids":[rng.randrange(1000,30000) for _ in range(n_tok)],
       "sampling_params":{"temperature":0,"max_new_tokens":out,"ignore_eos":True}}
    r=urllib.request.Request(U,data=json.dumps(b).encode(),headers={"Content-Type":"application/json"})
    urllib.request.urlopen(r,timeout=900).read(); return 1
with cf.ThreadPoolExecutor(20) as ex:
    f=[ex.submit(go,512,900,i) for i in range(8)]       # background decode rows
    time.sleep(3)
    f+=[ex.submit(go,9000,32,i) for i in range(8)]       # chunked prefills arriving into them
    print("done", sum(x.result() for x in f))
PY
L=$O/server.log
echo "=== mixed/extend step evidence:"
grep -oE "Prefill batch[^,]*, #new-token: [0-9]+|#running-req: [0-9]+" $L | tail -6
echo "=== does a step carry BOTH a chunk and >1 token per running row?"
grep -oE "Prefill batch\. #new-seq: [0-9]+, #new-token: [0-9]+, .*#running-req: [0-9]+" $L | tail -8
echo "=== spec accept/verify activity during prefill steps:"
grep -cE "Prefill batch" $L; grep -oE "accept len: [0-9.]+" $L | tail -3
kill $S 2>/dev/null
echo COMPCHECKDONE
