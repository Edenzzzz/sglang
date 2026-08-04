#!/usr/bin/env python3
"""Multi-tenant serving benchmarks for the prefill/decode-overlap kernel.

One node, several concurrent tenant streams, as a production cluster actually
sees. Each scenario is chosen from the measured kernel cost surface
(tasks/debug_kernel_server_perf.md): the persistent overlap kernel wins when a
mixed batch holds *few long-context decode rows* (which starve BatchPrefill of
query tiles, leaving most SMs idle while each streams its whole KV) and the
arriving prefill chunk attends a *short prefix*. Both conditions have to come
from the workload, and in practice they come from different tenants:

  * a coding-agent tenant supplies the long-context decode pool (KV capacity
    caps it near ten concurrent sessions, which is exactly the starved regime),
  * a chat tenant supplies frequent fresh, zero-prefix prefill chunks.

Neither tenant produces the win alone, which is why single-workload replays
measured parity.

Usage:
  python run_mixed_workloads.py --scenario agent_plus_shortchat
  python run_mixed_workloads.py --list
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENV_PY = str(REPO / ".venv" / "bin" / "python3")
MODEL = os.environ.get("MODEL", "/opt/tiger/models/Qwen2.5-7B-Instruct")
TRACE = os.environ.get("TRACE_PATH", str(REPO / "workloads/tracelab/syfi_coding_trace.jsonl.gz"))
CHAT_DS = os.environ.get(
    "CHAT_DATASET", str(REPO / "workloads/ShareGPT_V3_unfiltered_cleaned_split.json")
)
PORT = int(os.environ.get("BENCH_PORT", "30013"))


def codex(n, groups=12, min_ctx=32768, max_ctx=49152, min_out=512):
    """Coding-agent tenant: causal session replay with tool-result appends.

    Rounds are issued back to back (recorded think/tool gaps are not replayed).
    A real cluster runs far more sessions than KV capacity allows to be
    resident, so one session's think gap is filled by another's work; with the
    dozen sessions that fit here, replaying gaps would instead leave the node
    idle 88% of the time and measure nothing. Causal ordering and cumulative
    prefix growth within each session are preserved.
    """
    return [
        "--dataset-name", "tracelab-claude", "--dataset-path", TRACE,
        "--tokenize-prompt",
        "--tracelab-num-groups", str(groups),
        "--tracelab-min-context-len", str(min_ctx),
        "--tracelab-max-context-len", str(max_ctx),
        "--tracelab-min-output-len", str(min_out),
        "--request-rate", "inf", "--num-prompts", str(n),
    ]


def chat(n, rate, inp, out):
    """Chat tenant: fresh prompts, no shared prefix, so chunks have prefix 0."""
    return [
        "--dataset-name", "random", "--dataset-path", CHAT_DS,
        "--random-input", str(inp), "--random-output", str(out),
        "--random-range-ratio", "0.5",
        "--request-rate", str(rate), "--num-prompts", str(n),
    ]


SCENARIOS = {
    # Primary: long-context agent sessions beside high-rate short-answer chat.
    # Short outputs keep chat from accumulating decode rows, which would push
    # BatchPrefill back into its efficient many-tile regime; the arrivals still
    # make a third of all forwards mixed.
    "agent_plus_shortchat": {
        "desc": "coding agent + high-rate short-answer chat (routing/classification traffic)",
        "streams": {"codex": codex(60), "chat": chat(2400, 12, 512, 96)},
    },
    # The user-facing version: chat contributes long decodes as well as prefills.
    "agent_plus_chat": {
        "desc": "coding agent + conversational chat (long decodes, moderate arrival rate)",
        "streams": {"codex": codex(60), "chat": chat(600, 3, 1024, 512)},
    },
    # Single tenant. Decode KV grows from generated tokens, so a long-CoT pool
    # reaches the starved regime on its own while prompts stay fresh.
    "long_cot": {
        "desc": "long chain-of-thought reasoning, fresh prompts (single tenant)",
        "streams": {"reason": chat(160, 1.2, 1024, 8192)},
    },
    # Fresh long prompts (cache miss) chunked into several pieces: many mixed
    # steps whose early chunks still have small prefixes.
    "rag_plus_agent": {
        "desc": "cache-miss document/RAG traffic + coding agent",
        "streams": {"codex": codex(60), "rag": chat(300, 2, 12288, 256)},
    },
    # Control: the measured parity region. Many short rows give BatchPrefill
    # plenty of query tiles, so the gate should decline persistent throughout.
    "shortchat_only": {
        "desc": "CONTROL: high-concurrency short-context chat only (expect parity)",
        "streams": {"chat": chat(3000, 16, 512, 256)},
    },
}


def wait_health(proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        r = subprocess.run(
            ["curl", "-sf", f"http://127.0.0.1:{PORT}/health"],
            capture_output=True,
        )
        if r.returncode == 0:
            return True
        time.sleep(5)
    return False


def kill_server():
    subprocess.run(["pkill", "-9", "-f", f"launch_server.*--port {PORT}"],
                   capture_output=True)
    time.sleep(3)


def run_method(method, scenario, outdir):
    log = outdir / f"server_{method}.log"
    env = dict(os.environ)
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    if method == "LPT":
        env["SGLANG_FLASHINFER_USE_PERSISTENT"] = "1"
    else:
        # Defaults to 1; a true BatchPrefill baseline must disable it.
        env["SGLANG_FLASHINFER_USE_PERSISTENT"] = "0"
        env["SGLANG_DISABLE_MIXED_CG"] = "1"

    kill_server()
    with open(log, "w") as lf:
        server = subprocess.Popen(
            [VENV_PY, "-m", "sglang.launch_server", "--model", MODEL,
             "--attention-backend", "flashinfer", "--enable-mixed-chunk",
             "--mem-fraction-static", "0.85", "--port", str(PORT),
             "--host", "127.0.0.1", "--max-running-requests", "48",
             "--context-length", "131072", "--json-model-override-args",
             '{"rope_scaling":{"factor":4.0,"original_max_position_embeddings":32768,"type":"yarn"}}',
             "--watchdog-timeout", "600"],
            stdout=lf, stderr=subprocess.STDOUT, env=env,
        )
    if not wait_health(server):
        print(f"  server failed to start for {method}; see {log}", file=sys.stderr)
        server.kill()
        return False
    subprocess.run(["curl", "-sf", f"http://127.0.0.1:{PORT}/flush_cache"],
                   capture_output=True)

    procs = {}
    for sname, sargs in scenario["streams"].items():
        out = outdir / f"{method}_{sname}.json"
        blog = open(outdir / f"bench_{method}_{sname}.log", "w")
        procs[sname] = (subprocess.Popen(
            [VENV_PY, "-m", "sglang.bench_serving", "--backend", "sglang",
             "--host", "127.0.0.1", "--port", str(PORT), "--model", MODEL,
             *sargs, "--warmup-requests", "4", "--output-file", str(out),
             "--output-details", "--disable-tqdm"],
            stdout=blog, stderr=subprocess.STDOUT, env=env), blog)
    for sname, (p, blog) in procs.items():
        p.wait()
        blog.close()
    fm = subprocess.run(["grep", "forward mix", str(log)], capture_output=True, text=True)
    if fm.stdout.strip():
        print(f"  forward mix: {fm.stdout.strip().splitlines()[-1].split('forward mix')[-1]}")
    server.kill()
    kill_server()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="agent_plus_shortchat")
    ap.add_argument("--methods", default="DMC_CG,LPT")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in SCENARIOS.items():
            print(f"{k:24} {v['desc']}")
            for s, a in v["streams"].items():
                print(f"    {s:8} {' '.join(a[:6])} ...")
        return

    sc = SCENARIOS[args.scenario]
    outdir = Path(args.outdir or (Path(__file__).parent / f"h800_mw_{args.scenario}"))
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"scenario: {args.scenario} — {sc['desc']}")
    for method in args.methods.split(","):
        if all((outdir / f"{method}_{s}.json").exists() for s in sc["streams"]):
            print(f"  {method}: cached")
            continue
        print(f"  running {method} ...")
        if not run_method(method, sc, outdir):
            sys.exit(1)
    print(f"results in {outdir}")


if __name__ == "__main__":
    main()
