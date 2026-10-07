"""Serving correctness gate: greedy outputs of an arm vs the LPT baseline, on batches that
are forced to MIX prefill chunks with decode rows (the only batches the split touches).

  python srv_correctness.py ARM OUTDIR      # ARM in LPT, SPLIT; writes OUTDIR/ARM.json
  python srv_correctness.py --compare OUTDIR LPT SPLIT

Kernel arms are gated on PASS against an fp32 reference; serving arms had no gate at all,
which is how a split that computed the wrong requests' attention produced a "-40% TTFT".
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
import urllib.request

W = "/dev/shm/w"
PORT = 30077
MODEL = f"{W}/qwen7b"


def env_for(arm):
    env = dict(os.environ)
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    env["SGLANG_FLASHINFER_USE_PERSISTENT"] = "1"
    env["SGLANG_LOG_MIXED_COMPOSITION"] = f"/tmp/comp_{arm}.jsonl"
    if arm.startswith("SPLIT"):
        env.update(SGLANG_MIXED_ATTENTION_KERNEL="persistent", SGLANG_TWO_EXEC="1",
                   SGLANG_FUSED_FA3_SM_SPLIT="1", FLASHINFER_SM_PARTITION="1", UP_STAGES="2",
                   SGLANG_FUSED_FA3_SRC=f"{W}/fi/tests/fused_fa3_persistent_ext.cu")
    return env


def post(path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def prompts():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = json.load(open(f"{W}/workloads/ShareGPT_V3_unfiltered_cleaned_split.json"))
    text = "\n".join(t["value"] for d in data[:3000] for t in d.get("conversations", []))
    ids = tok(text[:4_000_000])["input_ids"]
    short = [ids[i * 50_000: i * 50_000 + 2048] for i in range(8)]
    long_ = [ids[600_000 + i * 50_000: 600_000 + i * 50_000 + 12_000] for i in range(4)]
    return short, long_


def run_arm(arm, outdir):
    os.makedirs(outdir, exist_ok=True)
    log = open(f"{outdir}/server_{arm}.log", "w")
    args = [f"{W}/venv/bin/python", "-m", "sglang.launch_server", "--model", MODEL,
            "--attention-backend", "flashinfer", "--enable-mixed-chunk",
            "--mem-fraction-static", "0.85", "--port", str(PORT), "--host", "127.0.0.1",
            "--max-running-requests", "48", "--context-length", "131072",
            "--json-model-override-args",
            '{"rope_scaling":{"factor":4.0,"original_max_position_embeddings":32768,"type":"yarn"}}',
            "--watchdog-timeout", "600"]
    srv = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env_for(arm))
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                if srv.poll() is not None:
                    raise RuntimeError(f"server died, see {outdir}/server_{arm}.log")
                time.sleep(2)
        short, long_ = prompts()
        sp = {"temperature": 0, "max_new_tokens": 96}
        results = {}
        with cf.ThreadPoolExecutor(16) as ex:
            futs = {ex.submit(post, "/generate", {"input_ids": p, "sampling_params": sp}): f"s{i}"
                    for i, p in enumerate(short)}
            time.sleep(1.5)  # short ones are decoding when the long prefills arrive
            for i, p in enumerate(long_):
                futs[ex.submit(post, "/generate", {"input_ids": p, "sampling_params": sp})] = f"l{i}"
                time.sleep(0.3)
            for f in cf.as_completed(futs):
                r = f.result()
                results[futs[f]] = r.get("output_ids") or r["meta_info"].get("output_ids")
        log.flush()
        txt = open(f"{outdir}/server_{arm}.log").read()
        results["_fallbacks"] = txt.lower().count("fallback")
        results["_tracebacks"] = txt.count("Traceback")
        json.dump(results, open(f"{outdir}/{arm}.json", "w"))
        print(f"{arm}: {len(results) - 2} requests, fallbacks={results['_fallbacks']} "
              f"tracebacks={results['_tracebacks']}", flush=True)
    finally:
        srv.kill()
        srv.wait()
        time.sleep(5)


def compare(outdir, base, arm):
    a = json.load(open(f"{outdir}/{base}.json"))
    b = json.load(open(f"{outdir}/{arm}.json"))
    print(f"{arm} vs {base}: fallbacks={b['_fallbacks']} tracebacks={b['_tracebacks']}")
    worst = {}
    for k in sorted(x for x in a if not x.startswith("_")):
        x, y = a[k], b[k]
        n = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        print(f"  {k}: first-token {'==' if x[:1] == y[:1] else '!='}  common prefix {n}/{len(x)}")
        worst[k[0]] = min(worst.get(k[0], 10 ** 9), n)
    # Greedy bf16 kernels with different reduction orders may legitimately diverge late;
    # a wrong attention diverges at or near the first token of the long (FA3) prompts.
    ok = all(a[k][:8] == b[k][:8] for k in a if k.startswith("l")) and b["_tracebacks"] == 0
    print(f"GATE {'PASS' if ok else 'FAIL'}: min common prefix short={worst.get('s')} "
          f"long={worst.get('l')} (long prompts must agree on their first 8 tokens)")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        run_arm(sys.argv[1], sys.argv[2])
