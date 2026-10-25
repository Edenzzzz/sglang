"""cond_disagg routing check with fake workers (CPU only): a cold long prompt is prefilled on a prefill worker
(max_new_tokens=1, non-streaming) and then served by a decode worker; a short follow-up append on the same session
goes straight to the decode worker that holds it; a short cold prompt (chat) never touches a prefill worker. Then
Dynamo's defaults (threshold 2048 AND 0.7 of the prompt): a 3000-token append to a cached 20000-token session runs
remotely under the threshold alone and locally once the prompt fraction is required.
  /opt/tiger/sglang/.venv/bin/python test_router_cond_disagg.py"""
import asyncio, json, subprocess
import aiohttp
from aiohttp import web

SEEN = []

async def fake_worker(port):
    async def gen(req):
        body = await req.json()
        SEEN.append((port, body["sampling_params"]["max_new_tokens"], body.get("stream", False), len(body["input_ids"])))
        if not body.get("stream"):
            return web.json_response({"text": "x", "meta_info": {}})
        resp = web.StreamResponse(); resp.content_type = "text/event-stream"
        await resp.prepare(req)
        for i in range(3):
            await asyncio.sleep(0.01)
            await resp.write(f'data: {{"text": "t{i}"}}\n\n'.encode())
        await resp.write(b"data: [DONE]\n\n"); await resp.write_eof(); return resp
    app = web.Application(); app.router.add_post("/generate", gen)
    r = web.AppRunner(app); await r.setup(); await web.TCPSite(r, "127.0.0.1", port).start()

async def main(base=32400, threshold=4096, frac=0.0, big_append=False):
    ports = [base + k for k in range(1, 5)]
    for p in ports:
        await fake_worker(p)
    log = "/tmp/rlog_cond.jsonl"
    proc = subprocess.Popen(["/opt/tiger/sglang/.venv/bin/python", "/opt/tiger/sglang/tasks/pd_colocation/router.py",
        "--workers", *[f"http://127.0.0.1:{p}" for p in ports], "--port", str(base + 100),
        "--policy", "cond_disagg", "--cond-disagg-prefill-workers", "2", "--cond-disagg-threshold", str(threshold), "--cond-disagg-min-frac", str(frac),
        "--pool-kv-tokens", "10000000", "--pool-write-policy", "write_through", "--log", log])
    await asyncio.sleep(1.5)
    try:
        async with aiohttp.ClientSession() as s:
            async def send(ids):
                async with s.post(f"http://127.0.0.1:{base + 100}/generate",
                                  json={"input_ids": ids, "sampling_params": {"max_new_tokens": 3}, "stream": True}) as r:
                    assert r.status == 200, r.status
                    await r.read()
            session = list(range(20000))
            SEEN.clear(); await send(session)
            cold = list(SEEN)
            SEEN.clear(); await send(session + list(range(50000, 53000 if big_append else 50800)))
            append = list(SEEN)
            SEEN.clear(); await send(list(range(90000, 90300)))
            chat = list(SEEN)
    finally:
        proc.terminate()
    pre = set(ports[:2])
    print("cold:", cold, "\nappend:", append, "\nchat:", chat)
    assert len(cold) == 2 and cold[0][0] in pre and cold[0][1] == 1 and cold[0][2] is False, cold
    assert cold[1][0] not in pre and cold[1][2] is True, cold
    if big_append and frac == 0:
        assert len(append) == 2 and append[0][0] in pre and append[0][1] == 1, append  # 3000 > 2048: remote
    else:
        assert len(append) == 1 and append[0][0] == cold[1][0], append
    assert len(chat) == 1 and chat[0][0] not in pre, chat
    print("COND_DISAGG_ROUTER_OK", threshold, frac, big_append)

asyncio.run(main())
asyncio.run(main(33400, 2048, 0.0, True))
asyncio.run(main(34400, 2048, 0.7, True))
