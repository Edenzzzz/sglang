"""Bisect which serving-gate ingredient breaks long-prompt first tokens.

The independent HF reference (bf16/fp32, with and without YaRN, all agreeing) gives long-prompt
argmax l0=220 l1=1429 l2=3010 l3=13. Every stock LPT run returns 13710/5888/387/11 and every
fused run 13710/5888/3010/11 -- tokens the reference scores up to 28 logits below its top. So the
serving stack, not only the fused kernel, mis-computes these prefills. Each config below adds one
gate ingredient to a bare server, and reports only the long prompts' first tokens.

Stock attention backend throughout; no fused env is set, so this isolates the shared stack.
"""
import concurrent.futures as cf
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

W = "/dev/shm/w"
PORT = 30077
MODEL = f"{W}/qwen7b"
REF = {"l0": 220, "l1": 1429, "l2": 3010, "l3": 13}
YARN = ["--context-length", "131072", "--json-model-override-args",
        '{"rope_scaling":{"factor":4.0,"original_max_position_embeddings":32768,"type":"yarn"}}']
MIXED = ["--chunked-prefill-size", "512", "--enable-mixed-chunk"]
SPEC = ["--speculative-algorithm", "EAGLE3", "--speculative-draft-model-path",
        f"{W}/qwen7b_eagle3_sgl", "--speculative-num-steps", "7", "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "8"]
CONFIGS = [
    ("C0_bare", [], False),
    ("C1_yarn", YARN, False),
    ("C2_yarn_chunk512_mixed", YARN + MIXED, False),
    ("C3_C2_plus_background_decodes", YARN + MIXED, True),
    ("C4_C3_plus_eagle3_q8", YARN + MIXED + SPEC, True),
]


def prompts():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = json.load(open(f"{W}/datasets/ShareGPT.original.json"))
    text = "\n".join(t["value"] for d in data[:3000] for t in d.get("conversations", []))
    ids = tok(text[:4_000_000])["input_ids"]
    short = [ids[i * 50_000: i * 50_000 + 2048] for i in range(8)]
    long_ = [ids[600_000 + i * 50_000: 600_000 + i * 50_000 + 12_000] for i in range(4)]
    return short, long_


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return r.get("output_ids") or r["meta_info"].get("output_ids")


def run(name, extra, background, short, long_, outdir):
    args = [f"{W}/venv/bin/python", "-m", "sglang.launch_server", "--model", MODEL,
            "--attention-backend", "flashinfer", "--mem-fraction-static", "0.85", "--port", str(PORT),
            "--host", "127.0.0.1", "--max-running-requests", "48", "--disable-cuda-graph",
            "--watchdog-timeout", "600"] + extra
    env = dict(os.environ, SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="1")
    log = open(f"{outdir}/server_{name}.log", "w")
    srv = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                if srv.poll() is not None:
                    raise RuntimeError(f"server died: {outdir}/server_{name}.log")
                time.sleep(2)
        out = {}
        with cf.ThreadPoolExecutor(16) as ex:
            futs = {}
            if background:
                for i, p in enumerate(short):
                    futs[ex.submit(post, {"input_ids": p, "sampling_params":
                                          {"temperature": 0, "max_new_tokens": 400, "ignore_eos": True}})] = f"s{i}"
                time.sleep(1.5)
            for i, p in enumerate(long_):
                futs[ex.submit(post, {"input_ids": p, "sampling_params":
                                      {"temperature": 0, "max_new_tokens": 4}})] = f"l{i}"
                time.sleep(0.3)
            for f in cf.as_completed(futs):
                out[futs[f]] = f.result()
        firsts = {k: out[k][0] for k in sorted(out) if k.startswith("l")}
        ok = sum(firsts[k] == REF[k] for k in REF)
        print(f"{name}: first tokens {firsts}  match_reference={ok}/4", flush=True)
        json.dump(out, open(f"{outdir}/{name}.json", "w"))
    finally:
        try:
            os.killpg(srv.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        srv.wait()
        time.sleep(8)


if __name__ == "__main__":
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    short, long_ = prompts()
    print("reference argmax:", REF, flush=True)
    for name, extra, background in CONFIGS:
        try:
            run(name, extra, background, short, long_, outdir)
        except Exception as exc:
            print(f"{name}: ERROR {exc}", flush=True)
    print("FIRST_TOKEN_BISECT_DONE", flush=True)
