"""Kernel-only serving attribution on the same native FA3 backend and schedule."""

import argparse
from concurrent.futures import ThreadPoolExecutor
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

from capture_serving_workloads_20261023 import configuration, interrupt, stop_owned
from native_serving_protocol_20261023 import assert_same_requests, measured_step_trace, summarize_stream


WORK = Path("/dev/shm/w")
PYTHON = str(WORK / "venv/bin/python")


def warmup(port, model_dir="qwen7b"):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(WORK / model_dir)
    documents = json.loads((WORK / "workloads/ShareGPT_V3_unfiltered_cleaned_split.json").read_text())
    text = "\n".join(turn["value"] for document in documents[:800] for turn in document.get("conversations", []))
    tokens = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    if len(tokens) < 65536:
        raise ValueError("Insufficient natural-text warmup material")

    def request(index):
        payload = dict(input_ids=tokens[index * 4096:index * 4096 + 16384], sampling_params=dict(temperature=0, max_new_tokens=64, ignore_eos=True))
        operation = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(operation, timeout=900) as response:
            output = json.load(response)
        if output["meta_info"]["completion_tokens"] != 64:
            raise RuntimeError("Warmup output is incomplete")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(request, range(8)))
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/flush_cache", timeout=60) as response:
        if response.status != 200:
            raise RuntimeError("Unable to flush warmup KV")


def run_arm(arguments, mode, repeat, arm):
    output = arguments.root / f"{arguments.stage}_{mode}_r{repeat}_{arm}"
    output.mkdir()
    config = configuration(arguments.stage)
    config.update(AGENT_CONC=str(arguments.concurrency), SESSIONS=str(arguments.sessions), CHAT_RATE=str(arguments.chat_rate), CHAT_N=str(arguments.chat_requests), TURNS="2")
    config["MODEL_DIR"] = arguments.model_dir
    config["TRACE_DATASET_PATH"] = os.environ.get("TRACE_DATASET_PATH", str(WORK / "workloads/tracelab/syfi_full_min.jsonl.gz"))
    if arguments.step_trace and "STEPTRACE_FEATURES source=request_cpu_v2" not in (arguments.root / "python/sglang/srt/managers/scheduler.py").read_text():
        raise ValueError("Step tracing requires the CPU-feature patch")
    model_config_path = WORK / arguments.model_dir / "config.json"
    model_config = json.loads(model_config_path.read_text())
    original_context = int(model_config["max_position_embeddings"])
    if arguments.no_yarn:
        if int(config["CTX_LEN"]) > original_context:
            raise ValueError("Requested serving context exceeds unextended model context")
        config["YARN_FACTOR"] = "0"
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("SGLANG_", "BENCH_", "FLASHINFER_", "NATIVE_PDL_MODEL_SHADOW", "NATIVE_PDL_FP32_ADJUDICATE"))}
    environment.update(config)
    environment.update(
        SERVING_PYTHONPATH=environment["PYTHONPATH"], SERVING_SERVER_LOG=str(output / "server.log"),
        SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="1", SGLANG_NATIVE_PDL="1" if arm == "native" else "0",
        SGLANG_NATIVE_PDL_FP32_PARTIALS="1", SGLANG_NATIVE_PDL_SPLIT_POLICY=arguments.split_policy,
        SGLANG_NATIVE_PDL_DIRECT_BF16_OUTPUT="1" if arguments.direct_output else "0",
        NATIVE_PDL_PACKED_QUERY_LIMIT=str(arguments.packed_query_limit),
        SGLANG_NATIVE_PDL_VERIFY="0",
        SGLANG_STEP_TRACE="1" if arguments.step_trace else "0",
    )
    if arguments.role_split and arm == "native":
        # The candidate runs the mixed step as two FA3 launches (prefill rows + decode/verify rows, row num_splits 32)
        # instead of FA3 + FA2; set explicitly because every SGLANG_* key is stripped from the inherited environment.
        environment.update(SGLANG_ROLE_SPLIT_FA3="1", SGLANG_ROLE_SPLIT_ROW_SPLITS=str(arguments.role_row_splits))
    if mode == "four" and not (arguments.ours and arm == "fa3"):
        environment.update(SGLANG_ALLOW_MIXED_CHUNK_SPEC="1", SGLANG_MIXED_SPEC="1", SGLANG_MIXED_SPEC_MIN_CTX="8192", SGLANG_MIXED_SPEC_MIN_QUEUE="4")
    if arguments.ours and arm == "native":
        # Our full single-node stack (2026-10-24): deadline-ordered scheduling with uncached-token deadlines, chunk
        # preemption and deadline-sized chunks -- the env mn_node.sh sets for EDF=1 PREEMPT=1 EDF_UNCACHED=1
        # DEADLINE_CHUNK=1024:2048:6000 -- on top of the native FA3 + FA2 kernel. The control arm is then plain stock
        # sglang on FA3 (and, in four-mode, stock MTP without the 4-mode mixed step).
        environment.update(
            SGLANG_EDF_SCHEDULING="1", SGLANG_TTFT_SLO_MS="2000", SGLANG_EDF_LATENCY_MODEL="",
            SGLANG_EDF_PREEMPT_CHUNK="1", SGLANG_EDF_RESERVE_EARLIER="1", SGLANG_EDF_PREEMPT_MAX_TOKENS="1000000000",
            SGLANG_EDF_UNCACHED_DEADLINE="1", SGLANG_EDF_DEADLINE_CHUNK=arguments.deadline_chunk,
        )
        environment.update(item.split("=", 1) for item in arguments.ours_env)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [
        PYTHON, "-m", "sglang.launch_server", "--model", str(WORK / arguments.model_dir),
        "--attention-backend", "fa3", "--enable-mixed-chunk", "--chunked-prefill-size", "2048",
        "--page-size", "1", "--mem-fraction-static", "0.85", "--host", "127.0.0.1", "--port", str(port),
        "--max-running-requests", "48", "--cuda-graph-max-bs-decode", "48", "--context-length", config["CTX_LEN"],
        "--watchdog-timeout", "600", "--tp-size", str(arguments.tp_size),
    ]
    if arguments.ours and arm == "native":
        command += ["--enable-priority-scheduling", "--schedule-low-priority-values-first",
                    "--priority-scheduling-preemption-threshold", "999999"]
    if not arguments.no_yarn:
        command += ["--json-model-override-args", json.dumps({"rope_scaling": {"factor": float(config["YARN_FACTOR"]), "original_max_position_embeddings": original_context, "type": "yarn"}})]
    if mode == "four":
        command += ["--speculative-algorithm", "NEXTN"] if arguments.draft_dir == "nextn" else ["--speculative-algorithm", "EAGLE3", "--speculative-draft-model-path", str(WORK / arguments.draft_dir)]
        command += ["--speculative-num-steps", "3", "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4"]
    protocol = dict(
        scope="Diagnostic full-step trace collection; no serving performance claim" if arguments.step_trace else "Exploratory kernel-only serving attribution; exact emitted-token parity remains open; no scheduler/EDF changes or publishable quality claim",
        arm=arm, mode=mode, repeat=repeat, configuration=config, command=command,
        kernel_environment={key: value for key, value in environment.items() if key.startswith("SGLANG_")},
        model=arguments.model_dir, draft_model=arguments.draft_dir if mode == "four" else None,
        model_configuration_sha256=hashlib.sha256(model_config_path.read_bytes()).hexdigest(),
        tp_size=arguments.tp_size, packed_query_limit=arguments.packed_query_limit,
        graphs="on", cpu_gpu_overlap="on",
        step_trace=arguments.step_trace,
        contents="TraceLab-derived session/round lengths with generated contents and omitted recorded delays; natural ShareGPT chat with template",
        warmup="Eight excluded natural-text 16K requests, then KV flush, then identical client warmups and barrier",
        source_hashes={name: hashlib.sha256((arguments.root / name).read_bytes()).hexdigest() for name in ("native_fa3_fa2_pdl_20261023.py", "mn_bench_pair.sh", "python/sglang/srt/layers/attention/flashattention_backend.py", "python/sglang/srt/managers/scheduler.py", "python/sglang/benchmark/serving.py")},
    )
    if arguments.step_trace:
        name = "python/sglang/srt/managers/steptrace_features.py"
        protocol["source_hashes"][name] = hashlib.sha256((arguments.root / name).read_bytes()).hexdigest()
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    server = bench = None
    try:
        with (output / "server.log").open("w") as log:
            server = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for attempt in range(450):
            if server.poll() is not None:
                raise RuntimeError("Serving arm exited during startup")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("Serving arm failed to become healthy")
        warmup(port, arguments.model_dir)
        with (output / "workload.log").open("w") as log:
            bench = subprocess.Popen(["bash", str(arguments.root / "mn_bench_pair.sh"), str(output / "requests"), str(port)], env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 3600
        while bench.poll() is None:
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Serving workload lost its server or exceeded one hour")
            time.sleep(2)
        if bench.returncode:
            raise RuntimeError("Serving clients failed")
        if arguments.step_trace:
            payload = dict(text="Trace flush.", sampling_params=dict(temperature=0, max_new_tokens=1, ignore_eos=True))
            operation = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(operation, timeout=120) as response:
                if json.load(response)["meta_info"]["completion_tokens"] != 1:
                    raise RuntimeError("Trace-draining probe failed")
        server_log = (output / "server.log").read_text(errors="replace")
        if re.search(r"Traceback|CUDA error\b|watchdog timeout", server_log, re.IGNORECASE):
            raise RuntimeError("Serving arm logged a fatal error")
        offset = int((output / "requests/server_log_offset").read_text())
        with (output / "server.log").open("rb") as log:
            log.seek(offset)
            measured_log = log.read().decode(errors="replace")
        native_calls = [int(value) for value in re.findall(r"NATIVE_PDL_CALL calls=(\d+)", measured_log)]
        if arm == "native" and not native_calls:
            raise RuntimeError("No native attention calls after the workload barrier")
        if arm == "fa3" and "NATIVE_PDL_CALL" in server_log:
            raise RuntimeError("Native attention contaminated the FA3 control")
        stock_mtp_control = arguments.ours and arm == "fa3"
        if mode == "four" and not stock_mtp_control and "MIXEDSPEC_STEPS" not in measured_log:
            raise RuntimeError("No measured four-mode target merge")
        if mode == "four" and stock_mtp_control and "MIXEDSPEC_STEPS" in measured_log:
            raise RuntimeError("The stock-MTP control merged prefill into verify steps; it must run stock MTP")
        programs = json.loads((output / "requests/agent_programs.json").read_text())
        if programs["concurrency_limit"] != arguments.concurrency or programs["peak_active_programs"] != min(arguments.concurrency, arguments.sessions):
            raise RuntimeError("Requested program concurrency was not attained")
        tenants, raw_tenants = {}, {}
        for tenant, expected in (("agent", arguments.sessions * 2), ("chat", arguments.chat_requests)):
            data = json.loads((output / "requests" / f"{tenant}.json").read_text())
            raw_tenants[tenant] = data
            tenants[tenant] = summarize_stream(data, expected)
            if mode == "four" and tenants[tenant]["tokens_per_verify"] is None:
                raise RuntimeError("Missing actual speculative acceptance counters")
        report = dict(protocol=protocol, tenants=tenants, peak_active_programs=programs["peak_active_programs"], logged_native_calls_after_barrier=len(native_calls))
        if arguments.step_trace:
            first_request = min(min(data["request_start_times"]) for data in raw_tenants.values())
            last_request = max(max(data["request_end_times"]) for data in raw_tenants.values())
            trace, count = measured_step_trace(server_log, first_request, last_request)
            (output / "measured_steptrace.log").write_text(trace)
            report["trace"] = dict(records=count, first_request=first_request, last_request=last_request, final_events_drained=True)
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print("NATIVE_SERVING_ARM_DONE", output.name, {tenant: {key: value for key, value in summary.items() if key != "requests_detail"} for tenant, summary in tenants.items()}, flush=True)
        return report
    finally:
        stop_owned(bench)
        stop_owned(server)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--stage", choices=("mix32", "mix128"), default="mix32")
    parser.add_argument("--modes", choices=("four", "two"), nargs="+", default=["four", "two"])
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--chat-rate", type=float, default=8)
    parser.add_argument("--chat-requests", type=int, default=960)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--direct-output", action="store_true")
    parser.add_argument("--split-policy", choices=("geometry_v2", "geometry_v3"), default="geometry_v2")
    parser.add_argument("--model-dir", default="qwen7b")
    parser.add_argument("--draft-dir", default="qwen7b_eagle3_sgl")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--no-yarn", action="store_true")
    parser.add_argument("--packed-query-limit", type=int, choices=(32, 64), default=32)
    parser.add_argument("--step-trace", action="store_true")
    parser.add_argument("--role-split", action="store_true")
    parser.add_argument("--ours", action="store_true",
                        help="control = stock sglang on FA3 (stock MTP in four-mode); candidate = our deadline scheduler + native kernel")
    parser.add_argument("--deadline-chunk", default="1024:2048:6000")
    parser.add_argument("--ours-env", action="append", default=[], metavar="KEY=VALUE",
                        help="extra SGLANG_* setting for the --ours candidate arm (repeatable)")
    parser.add_argument("--role-row-splits", type=int, default=32)
    arguments = parser.parse_args()
    if min(arguments.concurrency, arguments.sessions, arguments.chat_rate, arguments.chat_requests, arguments.repeats, arguments.tp_size) <= 0:
        raise ValueError("Positive workload dimensions are required")
    if arguments.step_trace and arguments.tp_size != 1:
        raise ValueError("Initial full-step trace collection supports TP1 only")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    reports = []
    for mode in arguments.modes:
        reference = None
        for repeat in range(1, arguments.repeats + 1):
            for arm in (("fa3", "native") if repeat % 2 else ("native", "fa3")):
                report = run_arm(arguments, mode, repeat, arm)
                if reference is not None:
                    assert_same_requests(reference, report)
                reference = report
                reports.append(report)
                (arguments.root / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
                time.sleep(3)
    print("NATIVE_SERVING_PAIR_DONE", len(reports), flush=True)


if __name__ == "__main__":
    main()
