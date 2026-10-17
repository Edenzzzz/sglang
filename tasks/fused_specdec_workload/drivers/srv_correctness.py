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
    if arm.startswith("MTP"):
        if "TREE" in arm:
            env["SGLANG_FUSED_TREE_BUILD"] = "1"
        if "MIX" in arm:
            env.update(SGLANG_ALLOW_MIXED_CHUNK_SPEC="1", SGLANG_MIXED_SPEC="1")
        if "SPLIT" in arm:
            env.update(SGLANG_MIXED_ATTENTION_KERNEL="persistent", SGLANG_TWO_EXEC="1",
                       SGLANG_FUSED_FA3_SM_SPLIT="1", FLASHINFER_SM_PARTITION="1",
                       UP_STAGES="2",
                       SGLANG_FUSED_FA3_SRC=f"{W}/fi/tests/fused_fa3_persistent_ext.cu")
    elif arm.startswith("SPLIT"):
        env.update(SGLANG_MIXED_ATTENTION_KERNEL="persistent", SGLANG_TWO_EXEC="1",
                   SGLANG_FUSED_FA3_SM_SPLIT="1", FLASHINFER_SM_PARTITION="1", UP_STAGES="2",
                   SGLANG_FUSED_FA3_SRC=f"{W}/fi/tests/fused_fa3_persistent_ext.cu")
    if arm.endswith("_EDF"):
        # Worker-side EDF: server-assigned deadlines, deadline-ordered admission, no latency model / chunk cap.
        env.update(SGLANG_EDF_SCHEDULING="1", SGLANG_EDF_LATENCY_MODEL="", SGLANG_EDF_ACCEPT_MODE="none",
                   SGLANG_TTFT_SLO_MS=os.environ.get("TTFT_SLO_MS", "2000"))
    return env


def post(path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def prompts():
    path = f"{W}/workloads/ShareGPT_V3_unfiltered_cleaned_split.json"
    if not os.path.exists(path):
        # Pods without the dataset (the H20 multi-node pods): fixed-seed random token ids of the
        # same lengths. The gate compares greedy outputs of two arms on identical inputs, which
        # needs determinism, not natural text.
        import random
        rng = random.Random(1234)
        short = [[rng.randrange(1000, 30000) for _ in range(2048)] for _ in range(8)]
        long_ = [[rng.randrange(1000, 30000) for _ in range(12_000)] for _ in range(4)]
        return short, long_
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = json.load(open(path))
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
    if arm.startswith("MTP"):
        # MTP-style chain draft: EAGLE3 head, 3 steps, topk 1 -> 4 verify tokens per row.
        args += ["--speculative-algorithm", "EAGLE3",
                 "--speculative-draft-model-path", f"{W}/qwen7b_eagle3_sgl",
                 "--speculative-num-steps", "3", "--speculative-eagle-topk", "1",
                 "--speculative-num-draft-tokens", "4"]
    if "PDMUX" in arm:
        # PD multiplexing (GreenContext SM split, layer-split prefill): no chunked prefill, no overlap scheduler.
        args += ["--enable-pdmux", "--chunked-prefill-size", "-1", "--disable-overlap-schedule"]
        # Extra pdmux-arm server flags (e.g. --pdmux-config-path); the stock reference arms stay unchanged.
        args += os.environ.get("SRV_EXTRA_ARGS", "").split()
    if arm.endswith("_EDF"):
        args += ["--enable-priority-scheduling", "--schedule-low-priority-values-first",
                 "--priority-scheduling-preemption-threshold", "999999"]
    # Own session, so teardown can kill sglang's children (scheduler, detokenizer, the HTTP
    # process) as a group. srv.kill() alone left one listening on the port, and the next
    # arm's warmup request went to that stale server and timed out.
    srv = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env_for(arm),
                           start_new_session=True)
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
        # The background rows must STILL BE DECODING when the long prefills arrive, or the step
        # mixes nothing: at 96 output tokens they finished in <1 s and the 2026-10-08 H20 gate saw
        # ONE mixed step, so the split was never exercised. ignore_eos keeps the length (and so the
        # greedy comparison) deterministic while they decode long enough to overlap every chunk.
        sp = {"temperature": 0, "max_new_tokens": 96}
        sp_bg = {"temperature": 0, "max_new_tokens": 1200, "ignore_eos": True}
        results = {}
        with cf.ThreadPoolExecutor(24) as ex:
            # Background: long-running decode rows, compared on their first 96 tokens.
            futs = {ex.submit(post, "/generate", {"input_ids": p, "sampling_params": sp_bg}): f"s{i}"
                    for i, p in enumerate(short)}
            time.sleep(2.0)
            # Foreground: long prompts STAGGERED, so a chunked prefill keeps arriving into a
            # steady decode/verify population instead of all landing in one window.
            for i, p in enumerate(long_ * 2):
                futs[ex.submit(post, "/generate", {"input_ids": p, "sampling_params": sp})] = f"l{i}"
                time.sleep(1.0)
                time.sleep(0.3)
            for f in cf.as_completed(futs):
                r = f.result()
                results[futs[f]] = r.get("output_ids") or r["meta_info"].get("output_ids")
                results["_verify_ct_" + futs[f]] = r["meta_info"].get("spec_verify_ct")
        log.flush()
        txt = open(f"{outdir}/server_{arm}.log").read()
        results["_fallbacks"] = txt.lower().count("fallback")
        results["_tracebacks"] = txt.count("Traceback")
        results["_mixed_spec_lines"] = [l[-120:] for l in txt.splitlines() if "MIXED_SPEC steps" in l][-3:]
        json.dump(results, open(f"{outdir}/{arm}.json", "w"))
        print(f"{arm}: {len(results) - 2} requests, fallbacks={results['_fallbacks']} "
              f"tracebacks={results['_tracebacks']} mixed={results['_mixed_spec_lines']}", flush=True)
    finally:
        import signal
        try:
            os.killpg(srv.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        srv.wait()
        time.sleep(8)


def compare(outdir, base, arm):
    a = json.load(open(f"{outdir}/{base}.json"))
    b = json.load(open(f"{outdir}/{arm}.json"))
    print(f"{arm} vs {base}: fallbacks={b['_fallbacks']} tracebacks={b['_tracebacks']}")
    worst = {}
    for k in sorted(x for x in a if not x.startswith("_")):
        x, y = a[k], b[k]
        n = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        va, vb = a.get("_verify_ct_" + k), b.get("_verify_ct_" + k)
        acc = ""
        if va and vb:
            # Tokens per verify step. On IDENTICAL outputs the drafts must have been identical,
            # so a difference here means the draft model saw different inputs.
            acc = f"  tok/verify {len(x) / va:.2f} vs {len(y) / vb:.2f}" + (
                "  <-- identical output, different acceptance" if x == y and va != vb else "")
        print(f"  {k}: first-token {'==' if x[:1] == y[:1] else '!='}  common prefix {n}/{len(x)}{acc}")
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
