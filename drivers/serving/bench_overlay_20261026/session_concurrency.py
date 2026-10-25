import asyncio
import math
import time


async def replay_concurrent_sessions(
    requests,
    send_request,
    concurrency,
    *,
    replay_delays=False,
    delay_scale=1.0,
    sleep=asyncio.sleep,
):
    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency < 1
    ):
        raise ValueError("program concurrency must be a positive integer")
    if not math.isfinite(delay_scale) or delay_scale < 0:
        raise ValueError("delay scale must be finite and nonnegative")
    sessions = {}
    for index, request in enumerate(requests):
        if request.session_id is None or request.round_index is None:
            raise ValueError("session replay requires session and round identities")
        delay = request.inter_round_delay_s or 0.0
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("recorded round delay must be finite and nonnegative")
        sessions.setdefault(request.session_id, []).append((index, request))
    for records in sessions.values():
        records.sort(key=lambda entry: entry[1].round_index)
        rounds = [request.round_index for _, request in records]
        if len(set(rounds)) != len(rounds):
            raise ValueError("duplicate round identity within a session")

    outputs = [None] * len(requests)
    pending = iter(sessions.items())
    events = []
    epoch = time.perf_counter()
    running = 0
    peak = 0
    program_seconds = 0.0

    async def worker():
        nonlocal running, peak, program_seconds
        for session_id, records in pending:
            started = time.perf_counter()
            running += 1
            peak = max(peak, running)
            events.append(
                {"event": "start", "session_id": session_id, "time": started - epoch}
            )
            try:
                for position, (index, request) in enumerate(records):
                    if position and replay_delays and request.inter_round_delay_s:
                        await sleep(request.inter_round_delay_s * delay_scale)
                    output = await send_request(request)
                    if not output.success:
                        raise RuntimeError(
                            f"Session {session_id} round {request.round_index} failed; dependent rounds not issued"
                        )
                    outputs[index] = output
            finally:
                finished = time.perf_counter()
                program_seconds += finished - started
                running -= 1
                events.append(
                    {"event": "end", "session_id": session_id, "time": finished - epoch}
                )

    workers = [
        asyncio.create_task(worker()) for _ in range(min(concurrency, len(sessions)))
    ]
    try:
        await asyncio.gather(*workers)
    finally:
        for task in workers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    duration = time.perf_counter() - epoch
    report = {
        "protocol": "closed_loop_program_concurrency",
        "concurrency_limit": concurrency,
        "sessions": len(sessions),
        "rounds": len(requests),
        "peak_active_programs": peak,
        "mean_active_programs": program_seconds / duration if duration else 0.0,
        "duration_s": duration,
        "delay_policy": "recorded" if replay_delays else "omitted_kernel_stress",
        "delay_scale": delay_scale,
        "session_rounds": [
            {
                "session_id": session_id,
                "input_indices": [index for index, _ in records],
                "round_indices": [request.round_index for _, request in records],
            }
            for session_id, records in sessions.items()
        ],
        "events": events,
    }
    return outputs, report
