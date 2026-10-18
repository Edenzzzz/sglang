"""S4 / goal point 3: does a request's acceptance length predict its TPOT? Per-request probe on one server.

Launches the serving config (stock MTP, or the 4-mode best setting with `ours`), keeps 4 x 40K-token agent-like
prompts prefilling in the background (so mixed steps happen), and streams ShareGPT first turns at RATE req/s (Poisson,
N requests, max 256 output tokens). Per request: TTFT, TPOT = (e2e - TTFT) / (tokens - 1), accepted tokens per verify
= completion_tokens / spec_verify_ct (meta_info). Fit on the first half (by arrival), test on the second:
  constant acceptance  TPOT_hat = k / mean_acc        (k = mean of TPOT*acc on the fit half)
  per-request accept   TPOT_hat = k / acc_i           (the steady state of a per-request EWMA)
  + load               TPOT_hat = (k0 + k1*running_reqs_at_arrival) / acc_i
  python tpot_acc_probe.py <outdir> [ours]
Claude, 2026-10-18.
"""
import concurrent.futures as cf, json, os, random, signal, subprocess, sys, threading, time, urllib.request
import numpy as np

W = "/dev/shm/w"
PORT = 30081
out = sys.argv[1]
ours = len(sys.argv) > 2 and sys.argv[2] == "ours"
RATE, N = float(os.environ.get("RATE", "6")), int(os.environ.get("N", "600"))
os.makedirs(out, exist_ok=True)
env = dict(os.environ, SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="1", SGLANG_FLASHINFER_USE_PERSISTENT="1")
args = [f"{W}/venv/bin/python", "-m", "sglang.launch_server", "--model", f"{W}/qwen7b", "--attention-backend",
        "flashinfer", "--enable-mixed-chunk", "--mem-fraction-static", "0.85", "--port", str(PORT), "--host",
        "127.0.0.1", "--max-running-requests", "64", "--context-length", "131072", "--json-model-override-args",
        '{"rope_scaling":{"factor":4.0,"original_max_position_embeddings":32768,"type":"yarn"}}',
        "--watchdog-timeout", "600", "--speculative-algorithm", "EAGLE3", "--speculative-draft-model-path",
        f"{W}/qwen7b_eagle3_sgl", "--speculative-num-steps", "3", "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "4"]
if ours:
    env.update(SGLANG_MIXED_ATTENTION_KERNEL="persistent", SGLANG_TWO_EXEC="1", SGLANG_FUSED_FA3_SM_SPLIT="1",
               FLASHINFER_SM_PARTITION="1", UP_STAGES="2",
               SGLANG_FUSED_FA3_SRC=f"{W}/fi/tests/fused_fa3_persistent_ext.cu", SGLANG_FUSED_FI_ROOT=f"{W}/port07",
               SGLANG_FUSED_FA2_TILES="64,32,16", FUSED_FA2_FINE_TAIL="1", FUSED_CONST_PERS_PARAMS="1",
               FUSED_WORK_BUDGET_SPLIT="1", FLASHINFER_FUSED_BACKFILL="1", FLASHINFER_FUSED_COLAUNCH="1",
               FUSED_FA2_KV_CHUNK="auto", SGLANG_FUSED_SM_FA3="112", SGLANG_SPLIT_OVERLAP="0",
               SGLANG_FUSED_DECODE_GUARD="0", SGLANG_EDF_PREEMPT_CHUNK="1", SGLANG_EDF_RESERVE_EARLIER="1",
               SGLANG_EDF_PREEMPT_MAX_TOKENS="1000000000", SGLANG_EDF_UNCACHED_DEADLINE="1",
               SGLANG_EDF_DEADLINE_CHUNK="1024:2048:6000", SGLANG_ALLOW_MIXED_CHUNK_SPEC="1", SGLANG_MIXED_SPEC="1",
               SGLANG_MIXED_SPEC_MIN_CTX="8192", SGLANG_MIXED_SPEC_MIN_QUEUE="4", SGLANG_EDF_SCHEDULING="1",
               SGLANG_TTFT_SLO_MS="2000", SGLANG_EDF_LATENCY_MODEL="")
    args += ["--chunked-prefill-size", "2048", "--enable-priority-scheduling", "--schedule-low-priority-values-first",
             "--priority-scheduling-preemption-threshold", "999999"]


def post(body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


inflight = [0]
lock = threading.Lock()


def stream_one(i, text, t_arr):
    with lock:
        running = inflight[0]; inflight[0] += 1
    t0 = time.perf_counter()
    first = None; meta = None
    try:
        r = post({"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": 256}, "stream": True})
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if first is None:
                first = time.perf_counter()
            meta = d.get("meta_info", meta)
    finally:
        with lock:
            inflight[0] -= 1
    t1 = time.perf_counter()
    toks = (meta or {}).get("completion_tokens", 0)
    vct = (meta or {}).get("spec_verify_ct", 0)
    return dict(i=i, arrival=t_arr, running=running, ttft=(first - t0) * 1e3 if first else None,
                tpot=(t1 - first) * 1e3 / (toks - 1) if first and toks > 1 else None,
                tokens=toks, acc=toks / vct if vct else None)


log = open(f"{out}/server.log", "w")
srv = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
try:
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
        except Exception:
            if srv.poll() is not None:
                raise RuntimeError("server died")
            time.sleep(2)
    data = json.load(open(f"{W}/workloads/ShareGPT_V3_unfiltered_cleaned_split.json"))
    prompts = [c["conversations"][0]["value"] for c in data if c.get("conversations") and len(c["conversations"]) > 1
               and c["conversations"][0]["from"] == "human" and 20 < len(c["conversations"][0]["value"]) < 6000][:N]
    rng = random.Random(5)
    stop = threading.Event()

    def background():
        # Agent-like long prefills, one after another per thread, for the whole run.
        while not stop.is_set():
            ids = [rng.randrange(1000, 30000) for _ in range(40000)]
            try:
                post({"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 16}}).read()
            except Exception:
                pass

    bgs = [threading.Thread(target=background, daemon=True) for _ in range(int(os.environ.get("BG", "4")))]
    [b.start() for b in bgs]
    time.sleep(3)
    res = []
    with cf.ThreadPoolExecutor(256) as ex:
        futs = []
        t_start = time.perf_counter()
        for i, p in enumerate(prompts):
            futs.append(ex.submit(stream_one, i, p, time.perf_counter() - t_start))
            time.sleep(rng.expovariate(RATE))
        res = [f.result() for f in futs]
    stop.set()
    json.dump(res, open(f"{out}/probe.json", "w"))
finally:
    os.killpg(srv.pid, signal.SIGKILL)
    srv.wait()

rs = [r for r in sorted(res, key=lambda r: r["arrival"]) if r["tpot"] and r["acc"]]
h = len(rs) // 2
tp = np.array([r["tpot"] for r in rs]); acc = np.array([r["acc"] for r in rs]); run = np.array([r["running"] for r in rs])
fit, test = slice(0, h), slice(h, None)


def mape(pred):
    e = np.abs(pred[test] - tp[test]) / tp[test]
    return f"MAPE={100 * e.mean():5.1f}% p90={100 * np.percentile(e, 90):5.1f}%"


k = (tp[fit] * acc[fit]).mean()
const = np.full_like(tp, (tp[fit] * acc[fit]).mean() / acc[fit].mean())
per_req = k / acc
A = np.stack([np.ones(h), run[fit]], 1)
c, *_ = np.linalg.lstsq(A, tp[fit] * acc[fit], rcond=None)
load = (c[0] + c[1] * run) / acc
print(f"{'ours (4-mode)' if ours else 'stock MTP'}: n={len(rs)} TPOT mean={tp.mean():.2f} ms, acc mean={acc.mean():.2f} "
      f"sd={acc.std():.2f} (p10 {np.percentile(acc, 10):.2f} p90 {np.percentile(acc, 90):.2f}), "
      f"corr(TPOT, 1/acc)={np.corrcoef(tp, 1 / acc)[0, 1]:.2f}")
print(f"  constant acceptance      {mape(const)}")
print(f"  per-request acceptance   {mape(per_req)}")
print(f"  per-request acc + load   {mape(load)}")
print("PROBE_DONE")
