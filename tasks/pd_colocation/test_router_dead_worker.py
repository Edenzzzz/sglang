"""router.py dead-worker handling with fake streaming workers (CPU only).

Worker 1 is up, worker 2's port has no server (connection refused), worker 3 dies after its first chunk on the
first request it gets. Expect: every request routed to worker 2 is rerouted and streams fully; the request worker 3
breaks mid-stream fails; afterwards workers 2 and 3 are marked dead and get nothing; every later request succeeds.
  <venv>/bin/python test_router_dead_worker.py <router.py>
Claude, 2026-10-20.
"""
import asyncio, json, subprocess, sys, collections
import aiohttp
from aiohttp import web

ROUTER = sys.argv[1]
BASE = 33500


async def fake_worker(port, die_after_first=False):
    state = {"served": 0}

    async def gen(req):
        await req.json()
        resp = web.StreamResponse(); resp.content_type = "text/event-stream"
        await resp.prepare(req)
        state["served"] += 1
        for i in range(3):
            await asyncio.sleep(0.02)
            await resp.write(f'data: {{"text": "t{i}"}}\n\n'.encode())
            if die_after_first:
                req.transport.close()  # the process "dies": connection cut mid-stream
                await runner.cleanup()
                return resp
        await resp.write(b"data: [DONE]\n\n"); await resp.write_eof(); return resp
    app = web.Application(); app.router.add_post("/generate", gen)
    runner = web.AppRunner(app); await runner.setup(); await web.TCPSite(runner, "127.0.0.1", port).start()
    return state


async def main():
    s1 = await fake_worker(BASE + 1)
    s3 = await fake_worker(BASE + 3, die_after_first=True)
    log = "/tmp/rlog_dead.jsonl"; open(log, "w").close()
    proc = subprocess.Popen([sys.executable, ROUTER, "--workers", f"http://127.0.0.1:{BASE+1}",
                             f"http://127.0.0.1:{BASE+2}", f"http://127.0.0.1:{BASE+3}",
                             "--port", str(BASE + 100), "--policy", "rr", "--log", log])
    await asyncio.sleep(1.5)
    res = []
    async with aiohttp.ClientSession() as s:
        for k in range(12):
            body = {"input_ids": [k] * 64, "sampling_params": {"max_new_tokens": 3}, "stream": True}
            try:
                async with s.post(f"http://127.0.0.1:{BASE+100}/generate", json=body) as r:
                    res.append((r.status, (await r.read()).count(b"data:")))
            except aiohttp.ClientError as e:
                res.append(("err", type(e).__name__))
    proc.terminate(); proc.wait()
    recs = [json.loads(l) for l in open(log)]
    dead = [r["worker_dead"] for r in recs if "worker_dead" in r]
    routed = collections.Counter(r["worker"] for r in recs if "worker" in r and "done" not in r)
    full = sum(1 for st, n in res if st == 200 and n == 4)
    print("results", res)
    print("dead", dead, "routed", dict(routed), "worker1 served", s1["served"], "worker3 served", s3["served"])
    ok = sorted(dead) == [1, 2] and full >= 11 and s3["served"] == 1
    print("DEAD_WORKER_TEST", "PASS" if ok else "FAIL")


asyncio.run(main())
