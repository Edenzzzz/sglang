"""Colocated-serving router for the PD-colocation vs PD-disaggregation study.

A transparent streaming proxy in front of N sglang workers (each serves prefill AND decode).
Policies, each one switchable so its contribution can be measured on its own:

  rr          round-robin (load-balance-only baseline)
  cache       highest prefix-hit worker (cache-affinity-only baseline)
  cache_rand  highest hit, but uniform among workers within --rand-thresh of the best hit
              (spreads KV pulls over replicas instead of hammering one holder; point 4)
  cache_slo   cache_rand + EDF/SLO veto: walk candidates by hit length and take the first
              whose predicted TTFT/TPOT keeps this request AND the tighter-deadline requests
              already queued there within SLO; else the least-bad worker (point 1)

With --edf, each request is forwarded with priority = deadline (ms), so a worker started with
--enable-priority-scheduling --schedule-low-priority-values-first runs EDF locally.

Prefix tracking is approximate by design: rolling hashes of fixed-size blocks of input_ids
(or text characters), per worker, LRU-bounded by that worker's KV capacity in tokens. That is
what a router can know without querying workers, which keeps the decision off the request path.

Every decision is logged as one JSON line (--log) for the hit-rate vs SLO analysis.
"""
import argparse
import asyncio
import collections
import json
import random
import time

import aiohttp
from aiohttp import web

BLOCK_TOK, BLOCK_CHR = 256, 1024


def block_hashes(payload):
    ids = payload.get("input_ids")
    if ids is not None:
        seq, blk = ids, BLOCK_TOK
    else:
        seq, blk = payload.get("text") or "", BLOCK_CHR
    out, h = [], 0
    for i in range(0, len(seq) - blk + 1, blk):
        h = hash((h, tuple(seq[i:i + blk]) if ids is not None else seq[i:i + blk]))
        out.append(h)
    n_tok = len(ids) if ids is not None else max(1, len(seq) // 4)
    return out, blk if ids is not None else blk // 4, n_tok


class Worker:
    def __init__(self, url, kv_tokens, block_tok):
        self.url = url
        self.cap_blocks = max(1, kv_tokens // block_tok)
        self.blocks = collections.OrderedDict()  # hash -> None, LRU order
        self.inflight = {}  # rid -> dict(prefill, decode_left, deadline, t0)

    def hit(self, hashes):
        n = 0
        for h in hashes:
            if h not in self.blocks:
                break
            n += 1
        return n

    def insert(self, hashes):
        for h in hashes:
            self.blocks[h] = None
            self.blocks.move_to_end(h)
        while len(self.blocks) > self.cap_blocks:
            self.blocks.popitem(last=False)


class PerfModel:
    """Step-cost model for admission. Coefficients come from a fitted step trace of the
    serving node type (tasks/perf_model); defaults are placeholders until calibrated."""

    def __init__(self, prefill_tok_s, decode_step_ms, decode_ms_per_seq):
        self.prefill_tok_s = prefill_tok_s
        self.decode_step_ms = decode_step_ms
        self.decode_ms_per_seq = decode_ms_per_seq

    def ttft_ms(self, uncached_tokens_ahead):
        return uncached_tokens_ahead / self.prefill_tok_s * 1000.0

    def tpot_ms(self, running):
        return self.decode_step_ms + self.decode_ms_per_seq * running


class Router:
    def __init__(self, a):
        self.a = a
        self.workers = [Worker(u, a.kv_tokens, BLOCK_TOK) for u in a.workers]
        self.pm = PerfModel(a.prefill_tok_s, a.decode_step_ms, a.decode_ms_per_seq)
        self.rr = 0
        self.rid = 0
        self.log = open(a.log, "a") if a.log else None
        self.session = None

    def predict(self, w, uncached, deadline):
        # Under EDF only requests with EARLIER deadlines are served ahead of this one.
        ahead = sum(r["prefill"] for r in w.inflight.values() if r["deadline"] <= deadline)
        ttft = self.pm.ttft_ms(ahead + uncached)
        tpot = self.pm.tpot_ms(len(w.inflight) + 1)
        return ttft, tpot

    def feasible(self, w, uncached, deadline, now):
        ttft, tpot = self.predict(w, uncached, deadline)
        if now + ttft / 1000.0 > deadline or tpot > self.a.itl_slo_ms:
            return False, ttft, tpot
        # Admitting this request delays LATER-deadline requests queued here; it must not
        # push any of them past their own deadline.
        for r in w.inflight.values():
            if r["deadline"] > deadline and r["decode_left"] == r["decode_total"]:
                t_r = now + self.pm.ttft_ms(
                    sum(x["prefill"] for x in w.inflight.values() if x["deadline"] <= r["deadline"])
                    + uncached) / 1000.0
                if t_r > r["deadline"]:
                    return False, ttft, tpot
        return True, ttft, tpot

    def choose(self, hashes, blk, n_tok, deadline):
        pol = self.a.policy
        now = time.time()
        if pol == "rr":
            self.rr = (self.rr + 1) % len(self.workers)
            return self.rr, {"hit": self.workers[self.rr].hit(hashes) * blk}
        hits = [w.hit(hashes) for w in self.workers]
        best = max(hits)
        if pol == "cache":
            cands = [i for i, h in enumerate(hits) if h == best]
            i = min(cands, key=lambda j: len(self.workers[j].inflight))
            return i, {"hit": best * blk}
        # Order by hit length bucket (--rand-thresh blocks), then by load, then randomly. Pure
        # random within a bucket (measured on H20, coloc_rand) sent prefix-less chat requests --
        # for which EVERY worker ties at hit 0 -- to busy workers: chat TTFT 109 -> 487 ms vs the
        # load-aware tie-break of `cache`. Randomness is for spreading pulls among near-equal
        # cache holders; it must not override load.
        order = sorted(range(len(self.workers)),
                       key=lambda j: (-(hits[j] // max(1, self.a.rand_thresh)),
                                      len(self.workers[j].inflight), random.random()))
        if pol == "cache_rand":
            return order[0], {"hit": hits[order[0]] * blk}
        vetoed = 0
        best_slack, best_i = None, order[0]
        for i in order:
            uncached = max(0, n_tok - hits[i] * blk)
            ok, ttft, tpot = self.feasible(self.workers[i], uncached, deadline, now)
            slack = deadline - (now + ttft / 1000.0)
            if best_slack is None or slack > best_slack:
                best_slack, best_i = slack, i
            if ok:
                return i, {"hit": hits[i] * blk, "vetoed": vetoed, "ttft_pred": ttft}
            vetoed += 1
        return best_i, {"hit": hits[best_i] * blk, "vetoed": vetoed, "least_bad": True}

    async def generate(self, request):
        payload = await request.json()
        hashes, blk, n_tok = block_hashes(payload)
        out_tok = int((payload.get("sampling_params") or {}).get("max_new_tokens", 128))
        deadline = time.time() + self.a.ttft_slo_ms / 1000.0
        i, info = self.choose(hashes, blk, n_tok, deadline)
        w = self.workers[i]
        self.rid += 1
        rid = self.rid
        uncached = max(0, n_tok - info.get("hit", 0))
        w.inflight[rid] = dict(prefill=uncached, decode_left=out_tok, decode_total=out_tok,
                               deadline=deadline, t0=time.time())
        w.insert(hashes)
        if self.a.edf:
            payload["priority"] = int(deadline * 1000) % 2_000_000_000
        if self.log:
            self.log.write(json.dumps(dict(t=time.time(), rid=rid, worker=i, n_tok=n_tok,
                                           policy=self.a.policy, **info)) + "\n")
            self.log.flush()
        resp = web.StreamResponse()
        try:
            async with self.session.post(w.url + "/generate", json=payload) as up:
                resp.content_type = up.headers.get("Content-Type", "application/json")
                resp.set_status(up.status)
                await resp.prepare(request)
                first = True
                tail = b""
                async for chunk in up.content.iter_any():
                    if first:
                        w.inflight[rid]["prefill"] = 0
                        first = False
                    w.inflight[rid]["decode_left"] = max(0, w.inflight[rid]["decode_left"] - 1)
                    tail = (tail + chunk)[-65536:]
                    await resp.write(chunk)
                await resp.write_eof()
            if self.log:
                # The worker's own account of this request's prefix hit: the last data chunk's
                # meta_info carries cached_tokens and prompt_tokens. This, not the router's
                # block-hash estimate, is the cache-hit rate the replication argument rests on.
                meta = {}
                for line in reversed(tail.decode("utf-8", "replace").splitlines()):
                    line = line.strip()
                    if line.startswith("data:") and "meta_info" in line:
                        try:
                            meta = json.loads(line[5:].strip()).get("meta_info", {})
                        except ValueError:
                            pass
                        break
                if meta:
                    self.log.write(json.dumps(dict(rid=rid, worker=i, done=time.time(),
                                                   cached_tokens=meta.get("cached_tokens"),
                                                   prompt_tokens=meta.get("prompt_tokens"))) + "\n")
                    self.log.flush()
        finally:
            w.inflight.pop(rid, None)
        return resp

    async def broadcast(self, request):
        res = []
        for w in self.workers:
            async with self.session.request(request.method, w.url + request.path) as r:
                res.append(r.status)
        return web.json_response({"status": res})

    async def health(self, request):
        return web.Response(text="ok")

    async def start(self, app):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", nargs="+", required=True)
    ap.add_argument("--port", type=int, default=31000)
    ap.add_argument("--policy", choices=["rr", "cache", "cache_rand", "cache_slo"], default="cache_slo")
    ap.add_argument("--rand-thresh", type=int, default=4, help="blocks within which hits tie")
    ap.add_argument("--edf", action="store_true")
    ap.add_argument("--ttft-slo-ms", type=float, default=5000)
    ap.add_argument("--itl-slo-ms", type=float, default=100)
    ap.add_argument("--kv-tokens", type=int, default=1_000_000)
    ap.add_argument("--prefill-tok-s", type=float, default=20000)
    ap.add_argument("--decode-step-ms", type=float, default=15)
    ap.add_argument("--decode-ms-per-seq", type=float, default=0.3)
    ap.add_argument("--log", default="")
    a = ap.parse_args()
    r = Router(a)
    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(r.start)
    app.router.add_post("/generate", r.generate)
    app.router.add_route("*", "/flush_cache", r.broadcast)
    app.router.add_get("/health", r.health)
    web.run_app(app, host="0.0.0.0", port=a.port, access_log=None)


if __name__ == "__main__":
    main()
