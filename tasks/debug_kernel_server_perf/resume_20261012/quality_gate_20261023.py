"""Downstream task-quality gate for the native FA3+FA2 PDL kernel (H200, 2026-10-23).

The correctness criterion the user set: FP32 attention checks (already passed on every captured record) PLUS a
downstream task-quality comparison -- not identical greedy tokens, which no batched kernel change preserves.

Arms, all on the frozen tree of the audited v3 run, same server command and kernel environment as that run
(`geometry_v3`, FP32 partials, direct BF16 output), only SGLANG_NATIVE_PDL differing:
  fa3_a   native off            -> first measurement of the baseline
  native  native on
  fa3_b   native off, rerun     -> the NOISE FLOOR: how much a baseline differs from itself under batching
Each in two-mode (no speculation) and four-mode (trained EAGLE3, 3 draft steps / 4 verify tokens).

Tasks:
  gsm8k    all 1,319 test questions, 8-shot, greedy, final number parsed; concurrency 128, so prefills ride beside
           decodes in mixed-chunk steps
  passkey  200 retrieval prompts, a 5-digit key at a random depth in ~32K or ~60K tokens of filler, concurrency 16,
           so 2,048-token chunks of deep prefixes share steps with other requests' decode rows -- the shape the
           native path serves (deep context + decode/verify rows)

Pass rule (written before running): native's accuracy on each task lies within the fa3_a-vs-fa3_b difference plus a
binomial 2-sigma band, AND the native arm logged NATIVE_PDL_CALL > 0 (a gate whose kernel never fired measures
nothing) while the fa3 arms logged none.
  python quality_gate_20261023.py <out_dir>
Claude, 2026-10-23.
"""
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/dev/shm/w/qwen25_v3_common_screen_h200_20261023")
W = Path("/dev/shm/w")
PY = "/dev/shm/w/venv/bin/python"
DATA = W / "perf/qgate"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

KERNEL_ENV = {
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1", "SGLANG_NATIVE_PDL_FP32_PARTIALS": "1",
    "SGLANG_NATIVE_PDL_SPLIT_POLICY": "geometry_v3", "SGLANG_NATIVE_PDL_DIRECT_BF16_OUTPUT": "1",
    "SGLANG_NATIVE_PDL_VERIFY": "0", "NATIVE_PDL_PACKED_QUERY_LIMIT": "64",
    "PYTHONPATH": f"{ROOT}/python:{ROOT}:/dev/shm/w/fi", "HF_HOME": "/dev/shm/w/hf", "FI_ROOT": "/dev/shm/w/fi",
    "CUDA_VISIBLE_DEVICES": "0",
    "NATIVE_PDL_FA2_LIBRARY": "/opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so",
    "NATIVE_PDL_FA3_LIBRARY": "/opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so",
    "NATIVE_PDL_FA2_FP32_LIBRARY": "/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so",
    "NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY": "/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so",
}
MIXED_SPEC_ENV = {"SGLANG_ALLOW_MIXED_CHUNK_SPEC": "1", "SGLANG_MIXED_SPEC": "1",
                  "SGLANG_MIXED_SPEC_MIN_CTX": "8192", "SGLANG_MIXED_SPEC_MIN_QUEUE": "4"}


def launch(arm, mode, log):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cmd = [PY, "-m", "sglang.launch_server", "--model", str(W / "qwen7b"), "--attention-backend", "fa3",
           "--enable-mixed-chunk", "--chunked-prefill-size", "2048", "--page-size", "1", "--mem-fraction-static", "0.85",
           "--host", "127.0.0.1", "--port", str(port), "--max-running-requests", "48", "--cuda-graph-max-bs-decode", "48",
           "--context-length", "131072", "--watchdog-timeout", "600", "--tp-size", "1",
           "--json-model-override-args",
           json.dumps({"rope_scaling": {"factor": 4.0, "original_max_position_embeddings": 32768, "type": "yarn"}})]
    if mode == "four":
        cmd += ["--speculative-algorithm", "EAGLE3", "--speculative-draft-model-path", str(W / "qwen7b_eagle3_sgl"),
                "--speculative-num-steps", "3", "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4"]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SGLANG_", "NATIVE_PDL"))}
    env.update(KERNEL_ENV)
    env["SGLANG_NATIVE_PDL"] = "1" if arm == "native" else "0"
    if mode == "four":
        env.update(MIXED_SPEC_ENV)
    proc = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early, see {log}")
        try:
            if urllib.request.urlopen(url + "/health", timeout=3).status == 200:
                return proc, url
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("server never became healthy")


def chat(url, prompt, max_tokens):
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())["choices"][0]["message"]["content"]


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def gsm8k_items():
    train = [json.loads(l) for l in open(DATA / "gsm8k_train.jsonl")]
    shots = "".join(f"Question: {t['question']}\nAnswer: {t['answer'].split('####')[0].strip()} The answer is "
                    f"{t['answer'].split('####')[1].strip()}.\n\n" for t in train[:8])
    items = []
    for t in (json.loads(l) for l in open(DATA / "gsm8k_test.jsonl")):
        gold = t["answer"].split("####")[1].strip().replace(",", "")
        items.append((shots + f"Question: {t['question']}\nAnswer:", gold))
    return items


def passkey_items():
    rng = random.Random(20261023)
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    items = []
    for i in range(200):
        n_rep = 1700 if i % 2 == 0 else 3200   # ~32K / ~60K tokens of filler
        key = str(rng.randint(10000, 99999))
        depth = rng.random()
        pos = int(n_rep * depth)
        text = filler * pos + f"The pass key is {key}. Remember it. {key} is the pass key. " + filler * (n_rep - pos)
        items.append((f"There is important information hidden in the text below. Find and memorize it.\n\n{text}\n\n"
                      "What is the pass key? Reply with the number only.", key))
    return items


def run_task(url, items, conc, max_tokens):
    def one(item):
        try:
            return last_number(chat(url, item[0], max_tokens))
        except Exception as exc:
            return f"ERR:{type(exc).__name__}"
    with ThreadPoolExecutor(conc) as ex:
        preds = list(ex.map(one, items))
    correct = [p == g for p, (_, g) in zip(preds, items)]
    errors = sum(1 for p in preds if isinstance(p, str) and p.startswith("ERR:"))
    return dict(n=len(items), accuracy=sum(correct) / len(items), errors=errors, preds=preds, correct=correct)


def main():
    tasks = {"gsm8k": (gsm8k_items(), 128, 320), "passkey": (passkey_items(), 16, 16)}
    results = {}
    for mode in ("two", "four"):
        for arm in ("fa3_a", "native", "fa3_b"):
            tag = f"{mode}_{arm}"
            log = OUT / f"{tag}_server.log"
            proc, url = launch("native" if arm == "native" else "fa3", mode, log)
            try:
                r = {}
                for name, (items, conc, mt) in tasks.items():
                    t0 = time.time()
                    r[name] = run_task(url, items, conc, mt)
                    r[name]["seconds"] = round(time.time() - t0, 1)
                    print(f"{tag} {name}: acc={r[name]['accuracy']:.4f} errors={r[name]['errors']} "
                          f"({r[name]['seconds']} s)", flush=True)
            finally:
                os.killpg(proc.pid, 15)
                proc.wait(timeout=120)
            r["native_pdl_calls"] = open(log, errors="replace").read().count("NATIVE_PDL_CALL")
            print(f"{tag} NATIVE_PDL_CALL lines: {r['native_pdl_calls']}", flush=True)
            results[tag] = r
            json.dump(results, open(OUT / "results.json", "w"))
    verdict = {}
    for mode in ("two", "four"):
        for name, (items, _, _) in tasks.items():
            a, n_, b = (results[f"{mode}_{x}"][name] for x in ("fa3_a", "native", "fa3_b"))
            p = (a["accuracy"] + b["accuracy"]) / 2
            band = abs(a["accuracy"] - b["accuracy"]) + 2 * math.sqrt(max(p * (1 - p), 1e-9) * 2 / len(items))
            delta = n_["accuracy"] - p
            fired = results[f"{mode}_native"]["native_pdl_calls"] > 0
            clean = results[f"{mode}_fa3_a"]["native_pdl_calls"] == 0 and results[f"{mode}_fa3_b"]["native_pdl_calls"] == 0
            ok = abs(delta) <= band and fired and clean and n_["errors"] == 0 and a["errors"] == 0 and b["errors"] == 0
            verdict[f"{mode}/{name}"] = dict(fa3_a=a["accuracy"], fa3_b=b["accuracy"], native=n_["accuracy"],
                                             delta=round(delta, 4), band=round(band, 4), native_fired=fired,
                                             baselines_clean=clean, PASS=ok)
            print(f"VERDICT {mode}/{name}: fa3 {a['accuracy']:.4f}/{b['accuracy']:.4f} native {n_['accuracy']:.4f} "
                  f"delta {delta:+.4f} band {band:.4f} fired={fired} clean={clean} -> {'PASS' if ok else 'FAIL'}", flush=True)
    json.dump(dict(results=results, verdict=verdict), open(OUT / "results.json", "w"), indent=1)
    print("QUALITY_GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
