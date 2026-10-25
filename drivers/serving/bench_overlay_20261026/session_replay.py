import asyncio
import json
import math
import os
import time
from pathlib import Path

import numpy as np


async def wait_for_benchmark_start(directory=None, timeout_s=1800):
    directory = directory or os.environ.get("BENCH_BARRIER_DIR")
    if not directory:
        return
    barrier = Path(directory)
    barrier.mkdir(parents=True, exist_ok=True)
    ready = barrier / f"ready_{os.getpid()}"
    ready.touch()
    deadline = time.monotonic() + timeout_s
    try:
        while not (barrier / "go").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Benchmark start barrier timed out: {barrier}")
            await asyncio.sleep(0.02)
    finally:
        ready.unlink(missing_ok=True)


def _write_audit(report, label):
    path = os.environ.get("BENCH_SESSION_AUDIT_FILE")
    if path:
        Path(path).write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        key: value for key, value in report.items()
        if key not in ("events", "session_rounds", "sessions_detail")
    }
    print(label, json.dumps(summary), flush=True)


async def replay_sessions(
    requests, send_request, request_rate, replay_timestamps=False, slowdown_factor=1.0,
    start_at_timestamps=False,
):
    """Serialize each session regardless of whether recorded think times are replayed.

    start_at_timestamps selects the trace-timestamp mode (see replay_timestamp_sessions);
    rows must carry trace_offset_s and already-scaled inter_round_delay_s.
    """
    if start_at_timestamps:
        if os.environ.get("BENCH_SESSION_CONCURRENCY") is not None:
            raise ValueError("start-at-timestamps replay excludes program-concurrency replay")
        if request_rate != float("inf"):
            raise ValueError("start-at-timestamps replay requires request-rate inf")
        outputs, report = await replay_timestamp_sessions(requests, send_request)
        _write_audit(report, "TRACE_TIMESTAMP_REPLAY")
        return outputs
    concurrency = os.environ.get("BENCH_SESSION_CONCURRENCY")
    if concurrency is not None:
        if request_rate != float("inf"):
            raise ValueError("program-concurrency replay requires request-rate inf")
        path = os.environ.get("BENCH_SESSION_AUDIT_FILE")
        if not path:
            raise ValueError("program-concurrency replay requires an audit output path")
        from sglang.benchmark.session_concurrency import replay_concurrent_sessions

        outputs, report = await replay_concurrent_sessions(
            requests,
            send_request,
            int(concurrency),
            replay_delays=replay_timestamps,
            delay_scale=slowdown_factor,
        )
        Path(path).write_text(json.dumps(report, indent=2) + "\n")
        print("PROGRAM_CONCURRENCY", json.dumps({key: value for key, value in report.items() if key not in ("events", "session_rounds")}), flush=True)
        return outputs
    if request_rate <= 0 or slowdown_factor < 0:
        raise ValueError("Session arrival rate must be positive and slowdown nonnegative")
    sessions = {}
    for index, request in enumerate(requests):
        if request.session_id is None or request.round_index is None:
            raise ValueError("Session replay requires session_id and round_index")
        sessions.setdefault(request.session_id, []).append((index, request))
    for records in sessions.values():
        records.sort(key=lambda item: item[1].round_index)
    outputs = [None] * len(requests)

    async def replay_one(records, start_delay):
        if start_delay:
            await asyncio.sleep(start_delay)
        for position, (index, request) in enumerate(records):
            if position and replay_timestamps and request.inter_round_delay_s:
                await asyncio.sleep(request.inter_round_delay_s * slowdown_factor)
            output = await send_request(request)
            if not output.success:
                raise RuntimeError(
                    f"Session {request.session_id} round {request.round_index} failed; "
                    "dependent rounds were not issued"
                )
            outputs[index] = output

    tasks = []
    start_delay = 0.0
    for records in sessions.values():
        tasks.append(asyncio.create_task(replay_one(records, start_delay)))
        if request_rate != float("inf"):
            start_delay += float(np.random.exponential(1.0 / request_rate))
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return outputs


async def replay_timestamp_sessions(
    requests, send_request, *, clock=time.perf_counter, sleep=asyncio.sleep
):
    """Replay a time-scaled trace window, keeping sessions causal.

    Times are seconds from the replay epoch (the call). Row r of a session carries its scaled
    trace offset o_r (trace_offset_s) and scaled recorded gap g_r (inter_round_delay_s: the
    previous eligible round's last event to this round's first event, divided by alpha).

      issue_0 = o_0
      issue_r = max(finish_{r-1} + g_r, o_r)       for r >= 1

    A session starts at its first row's offset, not at a Poisson draw. A later round is never
    issued before its predecessor completes plus the recorded tool/think gap, so the replay
    stays causal under load; when the server keeps up, rounds land on their trace offsets.
    Issue lag (issue_r - o_r) measures how far completion dependence pushed the session
    behind the trace.

    Returns (outputs in input order, audit report with per-session start and per-round
    scheduled/issue/finish times plus measured in-flight programs, session starts/s and
    LLM calls/s).
    """
    sessions = {}
    for index, request in enumerate(requests):
        if request.session_id is None or request.round_index is None:
            raise ValueError("Session replay requires session_id and round_index")
        offset = getattr(request, "trace_offset_s", None)
        if offset is None or not math.isfinite(offset) or offset < 0:
            raise ValueError("start-at-timestamps replay requires finite nonnegative trace_offset_s")
        delay = request.inter_round_delay_s or 0.0
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("recorded round delay must be finite and nonnegative")
        sessions.setdefault(request.session_id, []).append((index, request))
    for records in sessions.values():
        records.sort(key=lambda item: item[1].round_index)
        rounds = [request.round_index for _, request in records]
        if len(set(rounds)) != len(rounds):
            raise ValueError("duplicate round identity within a session")

    outputs = [None] * len(requests)
    details = {}
    events = []
    epoch = clock()
    state = {"programs": 0, "calls": 0, "peak_programs": 0, "peak_calls": 0}

    def now():
        return clock() - epoch

    async def wait_until(target):
        remaining = target - now()
        if remaining > 0:
            await sleep(remaining)

    async def replay_one(session_id, records):
        detail = {"session_id": session_id, "trace_start_s": records[0][1].trace_offset_s, "rounds": []}
        details[session_id] = detail
        previous_finish = None
        try:
            for position, (index, request) in enumerate(records):
                scheduled = request.trace_offset_s
                if position:
                    scheduled = max(previous_finish + (request.inter_round_delay_s or 0.0), scheduled)
                await wait_until(scheduled)
                issued = now()
                if not position:
                    detail["start_s"] = issued
                    state["programs"] += 1
                    state["peak_programs"] = max(state["peak_programs"], state["programs"])
                    events.append({"event": "start", "session_id": session_id, "time": issued})
                state["calls"] += 1
                state["peak_calls"] = max(state["peak_calls"], state["calls"])
                try:
                    output = await send_request(request)
                finally:
                    state["calls"] -= 1
                previous_finish = now()
                detail["rounds"].append({
                    "input_index": index,
                    "round_index": request.round_index,
                    "trace_offset_s": request.trace_offset_s,
                    "gap_s": request.inter_round_delay_s or 0.0,
                    "scheduled_s": scheduled,
                    "issue_s": issued,
                    "finish_s": previous_finish,
                    "success": bool(output.success),
                })
                if not output.success:
                    raise RuntimeError(
                        f"Session {session_id} round {request.round_index} failed; "
                        "dependent rounds not issued"
                    )
                outputs[index] = output
        finally:
            if "start_s" in detail:
                detail["end_s"] = now()
                state["programs"] -= 1
                events.append({"event": "end", "session_id": session_id, "time": detail["end_s"]})

    tasks = [
        asyncio.create_task(replay_one(session_id, records))
        for session_id, records in sessions.items()
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    duration = now()
    started = [detail for detail in details.values() if "start_s" in detail]
    program_seconds = sum(detail["end_s"] - detail["start_s"] for detail in started)
    all_rounds = [entry for detail in details.values() for entry in detail["rounds"]]
    call_seconds = sum(entry["finish_s"] - entry["issue_s"] for entry in all_rounds)
    lags = sorted(entry["issue_s"] - entry["trace_offset_s"] for entry in all_rounds)
    first_start = min((detail["start_s"] for detail in started), default=0.0)
    last_start = max((detail["start_s"] for detail in started), default=0.0)
    first_issue = min((entry["issue_s"] for entry in all_rounds), default=0.0)
    last_issue = max((entry["issue_s"] for entry in all_rounds), default=0.0)
    pick = lambda q: lags[min(len(lags) - 1, int(q * len(lags)))] if lags else None
    report = {
        "protocol": "trace_timestamps_completion_dependent",
        "rule": "issue_0 = offset_0; issue_r = max(finish_{r-1} + scaled_gap_r, offset_r)",
        "sessions": len(sessions),
        "rounds": len(requests),
        "duration_s": duration,
        "peak_active_programs": state["peak_programs"],
        "mean_active_programs": program_seconds / duration if duration else 0.0,
        "peak_inflight_calls": state["peak_calls"],
        "mean_inflight_calls": call_seconds / duration if duration else 0.0,
        "session_starts_per_s": (
            (len(started) - 1) / (last_start - first_start)
            if len(started) > 1 and last_start > first_start else None
        ),
        "calls_per_s": (
            (len(all_rounds) - 1) / (last_issue - first_issue)
            if len(all_rounds) > 1 and last_issue > first_issue else None
        ),
        "calls_per_s_over_duration": len(all_rounds) / duration if duration else 0.0,
        "issue_lag_s_p50_p90_p99_max": [pick(0.5), pick(0.9), pick(0.99), lags[-1] if lags else None],
        "sessions_detail": list(details.values()),
        "events": events,
    }
    return outputs, report
