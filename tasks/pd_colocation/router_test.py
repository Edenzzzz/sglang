"""Router policy check with fake streaming workers (CPU only): run with the repo venv python."""
import asyncio, json, subprocess, sys, time, collections
import aiohttp
from aiohttp import web

async def fake_worker(port, delay):
    async def gen(req):
        await req.json()
        resp = web.StreamResponse(); resp.content_type = "text/event-stream"
        await resp.prepare(req)
        for i in range(3):
            await asyncio.sleep(delay)
            await resp.write(f'data: {{"text": "t{i}"}}\n\n'.encode())
        await resp.write(b"data: [DONE]\n\n"); await resp.write_eof(); return resp
    app = web.Application(); app.router.add_post("/generate", gen)
    r = web.AppRunner(app); await r.setup(); await web.TCPSite(r, "127.0.0.1", port).start()

async def run(policy, extra, n=24, conc=1, prefix_len=4096, base=32000):
    for p in (base + 1, base + 2, base + 3):
        await fake_worker(p, 0.02)
    log = f"/tmp/rlog_{policy}.jsonl"
    open(log, "w").close()
    proc = subprocess.Popen(["/opt/tiger/sglang/.venv/bin/python", "/opt/tiger/sglang/tasks/pd_colocation/router.py",
        "--workers", f"http://127.0.0.1:{base+1}", f"http://127.0.0.1:{base+2}", f"http://127.0.0.1:{base+3}",
        "--port", str(base + 100), "--policy", policy, "--log", log, *extra])
    await asyncio.sleep(1.5)
    prefix = list(range(prefix_len))
    async with aiohttp.ClientSession() as s:
        async def one(k):
            body = {"input_ids": prefix + [10_000 + k] * 300, "sampling_params": {"max_new_tokens": 3}, "stream": True}
            async with s.post(f"http://127.0.0.1:{base+100}/generate", json=body) as r:
                return (await r.read()).count(b"data:")
        out = []
        for i in range(0, n, conc):
            out += await asyncio.gather(*[one(k) for k in range(i, i + conc)])
    proc.terminate(); proc.wait()
    recs = [json.loads(l) for l in open(log)]
    return out, recs

def main():
    for k, (policy, extra, conc) in enumerate([("rr", [], 1), ("cache", [], 1), ("cache_rand", ["--rand-thresh", "64"], 1),
                                ("cache_slo", ["--ttft-slo-ms", "300", "--prefill-tok-s", "20000", "--rand-thresh", "1"], 8)]):
        out, recs = asyncio.run(run(policy, extra, conc=conc, base=32000 + 200 * k))
        cnt = collections.Counter(r["worker"] for r in recs)
        vet = sum(r.get("vetoed", 0) for r in recs)
        print(f"{policy:<11} streamed_ok={all(o == 4 for o in out)} per_worker={dict(sorted(cnt.items()))} "
              f"hits={[r['hit'] for r in recs[:4]]} vetoed_total={vet}")

main()
