"""Serving correctness gate: greedy outputs of an arm vs the LPT baseline, on batches that
are forced to MIX prefill chunks with decode rows (the only batches the split touches).

  python srv_correctness.py ARM OUTDIR      # ARM in LPT, SPLIT; writes OUTDIR/ARM.json
  python srv_correctness.py --compare OUTDIR LPT SPLIT

MTP arms use MTP_DRAFT_TOKENS (default 4); MTP_SPEC_STEPS defaults to that width
minus one. Explicitly inconsistent chain widths are rejected before server launch.

Kernel arms are gated on PASS against an fp32 reference; serving arms had no gate at all,
which is how a split that computed the wrong requests' attention produced a "-40% TTFT".
"""
import concurrent.futures as cf
import json
import os
import re
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
        if "SPLIT" in arm or "FUSED" in arm:
            env.update(SGLANG_MIXED_ATTENTION_KERNEL="persistent", SGLANG_TWO_EXEC="1",
                       SGLANG_FUSED_FA3_SM_SPLIT="1", FLASHINFER_SM_PARTITION="1",
                       UP_STAGES="2",
                       SGLANG_FUSED_FA3_SRC=f"{W}/fi/tests/fused_fa3_persistent_ext.cu")
        if "FUSED" in arm:
            env["SGLANG_FUSED_KERNEL_EXEC"] = "1"
    elif arm.startswith("SPLIT"):
        env.update(SGLANG_MIXED_ATTENTION_KERNEL="persistent", SGLANG_TWO_EXEC="1",
                   SGLANG_FUSED_FA3_SM_SPLIT="1", FLASHINFER_SM_PARTITION="1", UP_STAGES="2",
                   SGLANG_FUSED_FA3_SRC=f"{W}/fi/tests/fused_fa3_persistent_ext.cu")
    return env


def post(path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def prompts():
    path = os.environ.get(
        "CORRECTNESS_DATASET", f"{W}/workloads/ShareGPT_V3_unfiltered_cleaned_split.json"
    )
    if os.environ.get("CORRECTNESS_SYNTHETIC") == "1":
        import random
        rng = random.Random(1234)
        short = [[rng.randrange(1000, 30000) for _ in range(2048)] for _ in range(8)]
        long_ = [[rng.randrange(1000, 30000) for _ in range(12_000)] for _ in range(4)]
        print("CORRECTNESS_PROMPTS synthetic token IDs, seed=1234", flush=True)
        return short, long_
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing correctness dataset {path}; set CORRECTNESS_SYNTHETIC=1 only for "
            "an explicitly synthetic diagnostic"
        )
    print(f"CORRECTNESS_PROMPTS ShareGPT token slices from {path}", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    with open(path) as dataset_file:
        data = json.load(dataset_file)
    text = "\n".join(t["value"] for d in data[:3000] for t in d.get("conversations", []))
    ids = tok(text[:4_000_000])["input_ids"]
    short = [ids[i * 50_000: i * 50_000 + 2048] for i in range(8)]
    long_ = [ids[600_000 + i * 50_000: 600_000 + i * 50_000 + 12_000] for i in range(4)]
    if any(len(prompt) != 2048 for prompt in short) or any(
        len(prompt) != 12000 for prompt in long_
    ):
        raise ValueError("ShareGPT corpus does not cover the complete correctness prompt slices")
    return short, long_


def speculative_arguments():
    draft_tokens = int(os.environ.get("MTP_DRAFT_TOKENS", "4"))
    draft_steps = int(os.environ.get("MTP_SPEC_STEPS", str(draft_tokens - 1)))
    if draft_steps < 1 or draft_tokens != draft_steps + 1:
        raise ValueError(
            "Chain verification requires MTP_DRAFT_TOKENS = MTP_SPEC_STEPS + 1 "
            "and at least one draft step"
        )
    return [
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", f"{W}/qwen7b_eagle3_sgl",
        "--speculative-num-steps", str(draft_steps),
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", str(draft_tokens),
    ]


def priority_arguments():
    if os.environ.get("SGLANG_EDF_SCHEDULING") != "1":
        return []
    return ["--enable-priority-scheduling", "--schedule-low-priority-values-first",
            "--priority-scheduling-preemption-threshold", "999999"]


def run_arm(arm, outdir):
    speculative_args = speculative_arguments() if arm.startswith("MTP") else []
    os.makedirs(outdir, exist_ok=True)
    short, long_ = prompts()
    log = open(f"{outdir}/server_{arm}.log", "w")
    args = [f"{W}/venv/bin/python", "-m", "sglang.launch_server", "--model", MODEL,
            "--attention-backend", "flashinfer", "--enable-mixed-chunk",
            "--mem-fraction-static", "0.85", "--port", str(PORT), "--host", "127.0.0.1",
            "--max-running-requests", "48", "--context-length", "131072",
            "--json-model-override-args",
            '{"rope_scaling":{"factor":4.0,"original_max_position_embeddings":32768,"type":"yarn"}}',
            "--watchdog-timeout", "600"]
    args += speculative_args + priority_arguments()
    if os.environ.get("DISABLE_CUDA_GRAPH") == "1":
        args.append("--disable-cuda-graph")
    if os.environ.get("BENCH_CHUNK"):
        args.extend(["--chunked-prefill-size", os.environ["BENCH_CHUNK"]])
    # Own session, so teardown can kill sglang's children (scheduler, detokenizer, the HTTP
    # process) as a group. srv.kill() alone left one listening on the port, and the next
    # arm's warmup request went to that stale server and timed out.
    server_env = env_for(arm)
    if "FUSED" in arm:
        server_env.setdefault("SGLANG_FUSED_FAILURE_DUMP", f"{outdir}/failure.pt")
    srv = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=server_env,
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
        sp = {"temperature": 0, "max_new_tokens": 96}
        background_sampling = dict(sp)
        if os.environ.get("CORRECTNESS_DECODE_TOKENS"):
            background_sampling.update(
                max_new_tokens=int(os.environ["CORRECTNESS_DECODE_TOKENS"]),
                ignore_eos=True,
            )
        results = {}
        with cf.ThreadPoolExecutor(16) as ex:
            if os.environ.get("CORRECTNESS_LONG_FIRST") == "1":
                # Long prefills first, short requests while they are still being chunked: the
                # order that makes EDF park a chunked prefill for an earlier-deadline request.
                # One long prompt first; the short ones arrive while most of its prefill is left
                # (EDF already orders waiting requests, so only an in-progress chunk can block).
                futs = {ex.submit(post, "/generate", {"input_ids": long_[0], "sampling_params": sp}): "l0"}
                time.sleep(float(os.environ.get("CORRECTNESS_SHORT_DELAY", "0.1")))
                for i, p in enumerate(short):
                    futs[ex.submit(post, "/generate", {"input_ids": p, "sampling_params": background_sampling})] = f"s{i}"
                    time.sleep(0.02)
                for i, p in enumerate(long_[1:], start=1):
                    futs[ex.submit(post, "/generate", {"input_ids": p, "sampling_params": sp})] = f"l{i}"
                    time.sleep(0.1)
            else:
                futs = {ex.submit(post, "/generate", {"input_ids": p, "sampling_params": background_sampling}): f"s{i}"
                        for i, p in enumerate(short)}
                time.sleep(1.5)  # short ones are decoding when the long prefills arrive
                for i, p in enumerate(long_):
                    futs[ex.submit(post, "/generate", {"input_ids": p, "sampling_params": sp})] = f"l{i}"
                    time.sleep(0.3)
            for f in cf.as_completed(futs):
                r = f.result()
                results[futs[f]] = r.get("output_ids") or r["meta_info"].get("output_ids")
                results["_verify_ct_" + futs[f]] = r["meta_info"].get("spec_verify_ct")
                results["_spec_metrics_" + futs[f]] = {
                    name: r["meta_info"][name]
                    for name in (
                        "spec_num_correct_drafts", "spec_num_proposed_drafts", "spec_accept_rate",
                    )
                    if name in r["meta_info"]
                }
        log.flush()
        txt = open(f"{outdir}/server_{arm}.log").read()
        results["_fallbacks"] = txt.lower().count("fallback")
        results["_speculative_args"] = speculative_args
        results["_tracebacks"] = txt.count("Traceback")
        results["_mixed_spec_lines"] = [
            line[-120:] for line in txt.splitlines()
            if "MIXEDSPEC_STEPS" in line
        ][-3:]
        results["_fused_kernel_calls"] = max(
            [int(count) for count in re.findall(r"FUSED_KERNEL_CALLS (\d+)", txt)],
            default=0,
        )
        results["_edf_park_calls"] = max(
            [int(count) for count in re.findall(r"EDF_PARK calls=(\d+)", txt)],
            default=0,
        )
        results["_fused_split_calls"] = max(
            [int(count) for count in re.findall(r"FUSED_SPLIT_CALLS (\d+)", txt)],
            default=0,
        )
        json.dump(results, open(f"{outdir}/{arm}.json", "w"))
        request_count = sum(not key.startswith("_") for key in results)
        print(f"{arm}: {request_count} requests, fallbacks={results['_fallbacks']} "
              f"tracebacks={results['_tracebacks']} mixed={results['_mixed_spec_lines']}", flush=True)
        if results["_fallbacks"] or results["_tracebacks"]:
            raise RuntimeError("Serving gate contains fallbacks or tracebacks")
        if "FUSED" in arm and results["_fused_kernel_calls"] == 0:
            raise RuntimeError("Fused serving gate exercised zero fused kernel calls")
        if os.environ.get("CORRECTNESS_REQUIRE_PARK") == "1" and results["_edf_park_calls"] == 0:
            raise RuntimeError("Park gate exercised zero EDF parks")
        if "SPLIT" in arm and results["_fused_split_calls"] == 0:
            raise RuntimeError("Split serving gate exercised zero split calls")
    finally:
        import signal
        try:
            os.killpg(srv.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        srv.wait()
        time.sleep(8)


def compare(outdir, base, arm):
    with open(f"{outdir}/{base}.json") as baseline_file:
        a = json.load(baseline_file)
    with open(f"{outdir}/{arm}.json") as candidate_file:
        b = json.load(candidate_file)
    print(f"{arm} vs {base}: fallbacks={b['_fallbacks']} tracebacks={b['_tracebacks']}")
    worst = {}
    for k in sorted(x for x in a if not x.startswith("_")):
        x, y = a[k], b[k]
        n = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        va, vb = a.get("_verify_ct_" + k), b.get("_verify_ct_" + k)
        acc = ""
        if va and vb:
            acc = f"  verify calls {va} vs {vb}"
        print(f"  {k}: first-token {'==' if x[:1] == y[:1] else '!='}  common prefix {n}/{len(x)}{acc}")
        worst[k[0]] = min(worst.get(k[0], 10 ** 9), n)
    # Greedy bf16 kernels with different reduction orders may legitimately diverge late;
    # a wrong attention diverges at or near the first token of the long (FA3) prompts.
    long_keys = [key for key in a if key.startswith("l")]
    ok = bool(long_keys) and all(
        len(a[key]) >= 8 and len(b[key]) >= 8 and a[key][:8] == b[key][:8]
        for key in long_keys
    ) and b["_tracebacks"] == 0 and b["_fallbacks"] == 0
    print(f"GATE {'PASS' if ok else 'FAIL'}: min common prefix short={worst.get('s')} "
          f"long={worst.get('l')} (long prompts must agree on their first 8 tokens)")
    if not ok:
        raise RuntimeError("Serving output comparison failed")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        run_arm(sys.argv[1], sys.argv[2])