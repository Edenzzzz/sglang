"""Torch-profile 4-mode MTP mixed steps of the H200 serving config (mn_h200_fourmode best setting, one server).

Background: 8 decoding rows (2048-token prompts, 1200 output tokens); foreground: 2 x 40K-token prompts, so chunked
prefill mixes with the verify rows. Once the first long prompt is being chunked, /start_profile records 60 steps
(CPU + GPU). Then: GPU busy union vs wall over the window, and the GPU idle gaps (> 50 us) with the kernels around them.
  python prof_mixed_20261018.py <outdir> [stock]   (stock: stock MTP, no 4-mode env, default chunk)
Claude, 2026-10-18.
"""
import concurrent.futures as cf
import glob, gzip, json, os, random, signal, subprocess, sys, time, urllib.request

W = "/dev/shm/w"
PORT = 30079
out = sys.argv[1]
stock = len(sys.argv) > 2 and sys.argv[2] == "stock"
os.makedirs(out, exist_ok=True)
env = dict(os.environ, SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="1", SGLANG_FLASHINFER_USE_PERSISTENT="1",
           SGLANG_TORCH_PROFILER_DIR=out, SGLANG_STEP_TRACE="1")
args = [f"{W}/venv/bin/python", "-m", "sglang.launch_server", "--model", f"{W}/qwen7b", "--attention-backend",
        "flashinfer", "--enable-mixed-chunk", "--mem-fraction-static", "0.85", "--port", str(PORT), "--host",
        "127.0.0.1", "--max-running-requests", "48", "--context-length", "131072", "--json-model-override-args",
        '{"rope_scaling":{"factor":4.0,"original_max_position_embeddings":32768,"type":"yarn"}}',
        "--watchdog-timeout", "600", "--speculative-algorithm", "EAGLE3", "--speculative-draft-model-path",
        f"{W}/qwen7b_eagle3_sgl", "--speculative-num-steps", "3", "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "4"]
if not stock:
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


def post(path, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout).read()


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
    rng = random.Random(7)
    short = [[rng.randrange(1000, 30000) for _ in range(2048)] for _ in range(8)]
    long_ = [[rng.randrange(1000, 30000) for _ in range(40000)] for _ in range(2)]
    with cf.ThreadPoolExecutor(16) as ex:
        bg = [ex.submit(post, "/generate", {"input_ids": p, "sampling_params": {"temperature": 0, "max_new_tokens": 1200,
                                                                                "ignore_eos": True}}) for p in short]
        time.sleep(4.0)
        fg = [ex.submit(post, "/generate", {"input_ids": p, "sampling_params": {"temperature": 0, "max_new_tokens": 32}})
              for p in long_]
        time.sleep(1.5)
        post("/start_profile", {"output_dir": out, "num_steps": 60,
                               "activities": os.environ.get("PROF_ACT", "CPU,GPU").split(","),
                               "with_stack": os.environ.get("PROF_STACK", "0") == "1"})
        for f in fg + bg:
            f.result()
    time.sleep(10)
finally:
    os.killpg(srv.pid, signal.SIGKILL)
    srv.wait()
print("PROF_DONE", sorted(glob.glob(f"{out}/*.json*")))
