"""Capture two/four-mode serving inputs, stop serving, then replay kernel controls."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.request


WORK = Path("/dev/shm/w")
PYTHON = str(WORK / "venv/bin/python")


def stop_owned(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def interrupt(signum, frame):
    raise KeyboardInterrupt(f"Capture interrupted by signal {signum}")


def configuration(stage):
    common = dict(
        MODEL_DIR="qwen7b", CHAT_TEMPLATE="1", CHAT_SHAREGPT="1", AGENT_SEED="20261014",
        TURNS="2", FO_TEXT="1", FO_TASKS="2", FO_ROUNDS="4", FO_APPEND="1024",
        FO_OUT="128", FO_GAP="5", FO_ZIPF="1.2", FO_ROTATE="180",
    )
    options = {
        "mix32": dict(WORKLOAD="tracelab", MIN_CTX="32768", MAX_CTX="65536", AGENT_CONC="4", SESSIONS="8", CHAT_RATE="8", CHAT_N="480", CTX_LEN="131072", YARN_FACTOR="4"),
        "mix64": dict(WORKLOAD="tracelab", MIN_CTX="65536", MAX_CTX="131072", AGENT_CONC="4", SESSIONS="8", CHAT_RATE="6", CHAT_N="720", CTX_LEN="262144", YARN_FACTOR="8"),
        "mix128": dict(WORKLOAD="tracelab", MIN_CTX="131072", MAX_CTX="262144", AGENT_CONC="2", SESSIONS="4", CHAT_RATE="4", CHAT_N="480", CTX_LEN="524288", YARN_FACTOR="16"),
        "fanout384": dict(WORKLOAD="fanout", MIN_CTX="307200", MAX_CTX="409600", AGENT_CONC="2", SESSIONS="4", CHAT_RATE="4", CHAT_N="1000", CTX_LEN="524288", YARN_FACTOR="16", FO_PREFIX="393216", FO_DURATION="180"),
    }
    return common | options[stage]


def run_case(root, stage, mode, model_dir="qwen7b", draft_dir="qwen7b_eagle3_sgl", agent_seed="20261014", tp_size=1, no_yarn=False):
    output = root / (stage + "_" + mode)
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    output.mkdir()
    captures = output / "captures"
    config = configuration(stage)
    config.update(MODEL_DIR=model_dir, AGENT_SEED=agent_seed)
    config["TRACE_DATASET_PATH"] = os.environ.get("TRACE_DATASET_PATH", str(WORK / "workloads/tracelab/syfi_full_min.jsonl.gz"))
    model_config = json.loads((WORK / model_dir / "config.json").read_text())
    original_context = int(model_config["max_position_embeddings"])
    if no_yarn:
        if int(config["CTX_LEN"]) > original_context:
            raise ValueError("Requested context exceeds the unextended model context")
        config["YARN_FACTOR"] = "0"
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith(("SGLANG_NATIVE_PDL", "SGLANG_SIMULATE_ACC", "NATIVE_PDL_MODEL_SHADOW")):
            environment.pop(key)
    environment.update(config)
    environment.update(
        SGLANG_CAPTURE_MIXED_ATTN=str(captures), CAPTURE_MODE=mode, CAPTURE_LAYER="14",
        CAPTURE_MIN_ROW_CONTEXT=config["MIN_CTX"],
        SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="1",
    )
    if mode == "four":
        environment.update(
            SGLANG_ALLOW_MIXED_CHUNK_SPEC="1", SGLANG_MIXED_SPEC="1",
            SGLANG_MIXED_SPEC_MIN_CTX="8192", SGLANG_MIXED_SPEC_MIN_QUEUE="4",
        )
    else:
        for key in ("SGLANG_ALLOW_MIXED_CHUNK_SPEC", "SGLANG_MIXED_SPEC"):
            environment.pop(key, None)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [
        PYTHON, "-m", "sglang.launch_server", "--model", str(WORK / model_dir),
        "--attention-backend", "fa3", "--enable-mixed-chunk", "--chunked-prefill-size", "2048",
        "--page-size", "1", "--mem-fraction-static", "0.85", "--host", "127.0.0.1", "--port", str(port),
        "--max-running-requests", "48", "--context-length", config["CTX_LEN"],
        "--disable-cuda-graph", "--watchdog-timeout", "600", "--tp-size", str(tp_size),
    ]
    if not no_yarn:
        command += ["--json-model-override-args", json.dumps({"rope_scaling": {"factor": float(config["YARN_FACTOR"]), "original_max_position_embeddings": original_context, "type": "yarn"}})]
    if mode == "four":
        command += ["--speculative-algorithm", "NEXTN"] if draft_dir == "nextn" else ["--speculative-algorithm", "EAGLE3", "--speculative-draft-model-path", str(WORK / draft_dir)]
        command += [
            "--speculative-num-steps", "3", "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4",
        ]
    protocol = dict(
        stage=stage, mode=mode, environment=config, command=command, layer=14,
        model_dir=model_dir, draft_dir=draft_dir if mode == "four" else None,
        tp_size=tp_size, capture_rank=int(environment.get("CAPTURE_RANK", "0")),
        packed_query_limit=int(environment.get("CAPTURE_PACKED_QUERY_LIMIT", "32")),
        selection="First eligible actual forward in each fresh/cached × <=512/>512 query bucket; all eligible shapes logged, not a random sample",
        agent_contents="TraceLab-derived session/round lengths with generated token contents and omitted recorded delays" if stage.startswith("mix") else "Natural-text fan-out prefixes and appends from ShareGPT",
        chat_contents="Natural ShareGPT with chat template",
        scope="Capture pass is not timed serving; inputs are model activations, not generated microbenchmark tensors",
        kernel_policy="geometry_v1" if os.environ.get("CAPTURE_GEOMETRY_POLICY") == "1" else "explicit",
        backend_sha256=hashlib.file_digest((root / "python/sglang/srt/layers/attention/flashattention_backend.py").open("rb"), "sha256").hexdigest(),
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    server = bench = None
    try:
        with (output / "server.log").open("w") as log:
            server = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for attempt in range(300):
            if server.poll() is not None:
                raise RuntimeError("Capture server exited during startup")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("Capture server did not become healthy")
        with (output / "workload.log").open("w") as log:
            bench = subprocess.Popen(
                ["bash", str(root / "mn_bench_capture.sh"), str(output / "requests"), str(port)],
                env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        deadline = time.monotonic() + 1800
        while bench.poll() is None:
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Capture workload lost its server or exceeded its deadline")
            time.sleep(2)
        if bench.returncode:
            raise RuntimeError(f"Capture clients failed with status {bench.returncode}")
        server_log = (output / "server.log").read_text(errors="replace")
        if re.search(r"Traceback|CUDA error\b|watchdog timeout", server_log, re.IGNORECASE):
            raise RuntimeError("Capture server logged a fatal error")
        for tenant, expected in (("chat", int(config["CHAT_N"])), ("agent", int(config["SESSIONS"]) * 2)):
            if tenant == "agent" and stage == "fanout384":
                result = json.loads((output / "requests/fanout.json").read_text())
                records = result.get("records", [])
                if not records or any(row.get("error") or row.get("out_tokens") != 128 for row in records):
                    raise RuntimeError("Fan-out capture workload contains incomplete outputs")
                continue
            result = json.loads((output / "requests" / (tenant + ".json")).read_text())
            if result["completed"] != expected or any(result["errors"]):
                raise RuntimeError(f"Incomplete {tenant} requests")
        if not list(captures.glob("*.pt")):
            raise RuntimeError("No actual mixed attention tensors were captured")
        for captured in captures.glob("*.json"):
            geometry = json.loads(captured.read_text())
            if geometry["timestamp"] < geometry["workload_release_time"]:
                raise RuntimeError("Warmup tensor was included in the workload captures")
        if mode == "four" and "MIXEDSPEC_STEPS" not in server_log:
            raise RuntimeError("Four-mode merge did not execute")
        (output / "capture_complete").write_text("validated\n")
    finally:
        stop_owned(bench)
        stop_owned(server)
    time.sleep(3)
    with (output / "kernel_bench.log").open("w") as log:
        kernel_arguments = ["--geometry-policy"] if os.environ.get("CAPTURE_GEOMETRY_POLICY") == "1" else []
        subprocess.run(
            [PYTHON, str(root / "bench_captured_mixed_20261023.py"), str(captures), str(output / "kernel_results.json"), *kernel_arguments],
            env=environment, stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    print("CAPTURED_SERVING_CASE_DONE", stage, mode, flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--stages", nargs="+", choices=("mix32", "mix64", "mix128", "fanout384"), default=["mix32", "mix128", "fanout384"])
    parser.add_argument("--modes", nargs="+", choices=("two", "four"), default=["two", "four"])
    parser.add_argument("--model-dir", default="qwen7b")
    parser.add_argument("--draft-dir", default="qwen7b_eagle3_sgl")
    parser.add_argument("--agent-seed", default="20261014")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--no-yarn", action="store_true")
    arguments = parser.parse_args()
    if arguments.tp_size < 1:
        raise ValueError("Positive tensor-parallel size required")
    for stage in arguments.stages:
        for mode in arguments.modes:
            run_case(arguments.root, stage, mode, arguments.model_dir, arguments.draft_dir, arguments.agent_seed, arguments.tp_size, arguments.no_yarn)
    print("CAPTURED_SERVING_MATRIX_DONE", flush=True)
