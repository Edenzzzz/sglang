"""Colocated-serving router for the PD-colocation vs PD-disaggregation study.

A transparent streaming proxy in front of N sglang workers (each serves prefill AND decode).
Policies, each one switchable so its contribution can be measured on its own:

  rr          round-robin (load-balance-only baseline)
  cache       highest prefix-hit worker (cache-affinity-only baseline)
  cache_rand  highest hit, but uniform among workers within --rand-thresh of the best hit
              (spreads KV pulls over replicas instead of hammering one holder; point 4)
  cache_slo   cache_rand + EDF/SLO check: walk candidates by hit length and take the first
              whose predicted TTFT/TPOT keeps this request AND the tighter-deadline requests
              already queued there within SLO; else the least-bad worker (point 1)

With --edf, each request is forwarded with priority = deadline (ms), so a worker started with
--enable-priority-scheduling --schedule-low-priority-values-first runs EDF locally.

Speculative decoding (point 3): a decode step emits `acceptance length` tokens per request, so the
per-token TPOT the SLO check compares against the ITL SLO is step time / acceptance. With --accept-ewma-alpha > 0 the router
keeps a per-worker EWMA of acceptance from each finished request's meta_info
(completion_tokens / spec_verify_ct) and divides the step-time prediction by it; --step-model
loads the linear MTP step model of tasks/perf_model/fit_mtp_step.py (incl. its acceptance term).

Prefix tracking is approximate by design: a radix tree flattened at block granularity. Block k's
key hashes block k's input_ids together with block k-1's key, so one key names the whole prefix up
to block k and a hit is the walk to the first missing key -- the same chained-page scheme as
sglang's HiCache storage keys (hicache_storage.get_hash_str). One key set per worker, LRU-bounded
by its KV capacity; with --pool-kv-tokens, one more for the cluster-wide L3 KV pool. That is what a
router can know without querying workers, which keeps the decision off the request path.

KV transfer (--pool-kv-tokens > 0): a request routed to a node holding x_i of its prefix, while
the pool holds x_pool > x_i, has the missing x_pool - x_i tokens prefetched into that node while it
waits in the queue. What it gets depends on the worker's --hicache-storage-prefetch-policy:
best_effort takes what has arrived when the request is scheduled, timeout waits up to
base + per_ktok * missing / 1000 s, wait_complete waits for all of it. The router predicts the
pulled tokens from the predicted queue wait and an EWMA of observed pull bandwidth, learned from
each finished request's cached_tokens (which includes prefetched pages) against the router's own
local-hit estimate.

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
        self.accept = None  # EWMA of tokens per verify step on this worker (None: no sample yet)

    def observe_accept(self, completion_tokens, verify_ct, alpha):
        if not completion_tokens or not verify_ct:
            return
        sample = completion_tokens / verify_ct
        self.accept = sample if self.accept is None else (1 - alpha) * self.accept + alpha * sample

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

    def __init__(self, prefill_tok_s, decode_step_ms, decode_ms_per_seq,
                 decode_ms_per_seq_ktok=0.0):
        self.prefill_tok_s = prefill_tok_s
        self.decode_step_ms = decode_step_ms
        self.decode_ms_per_seq = decode_ms_per_seq
        # A decode row's cost grows with its OWN KV, so a per-sequence slope fitted at short
        # context under-predicts a holder running 16K-prefix rows by ~an order of magnitude.
        self.decode_ms_per_seq_ktok = decode_ms_per_seq_ktok
        self.grid = {}
        # Measured (prompt tokens, TTFT ms) of cold prefills (--prefill-curve): attention makes a
        # token's cost grow with its context, 8.3K tok/s at 4K -> 4.4K at 48K on H20, which one
        # blended rate under-predicts for exactly the long agent prefills (item 7).
        self.curve = []
        self.step = None  # fit_mtp_step.py coefficients: whole-step ms, linear in the step's work

    def step_ms(self, running, kv_tokens, accept):
        c = self.step
        return (c["c0"] + c["c_bs"] * running + c["c_kv"] * kv_tokens / 1e3
                + c.get("c_acc", 0.0) * accept)

    def ttft_ms(self, uncached_tokens_ahead):
        return uncached_tokens_ahead / self.prefill_tok_s * 1000.0

    def cold_ms(self, n):
        """TTFT of an n-token cold prefill, interpolated on the curve (linear past its end)."""
        pts = [(0, 0.0)] + self.curve
        for (n0, t0), (n1, t1) in zip(pts, pts[1:]):
            if n <= n1:
                return t0 + (t1 - t0) * (n - n0) / (n1 - n0)
        (n0, t0), (n1, t1) = pts[-2], pts[-1]
        return t1 + (t1 - t0) * (n - n1) / (n1 - n0)

    def prefill_ms(self, uncached, ctx):
        """Time to prefill the last `uncached` tokens of a `ctx`-token prompt: the cold cost of the
        whole prompt minus that of its cached prefix, so a cached prefix still costs attention."""
        if not self.curve:
            return self.ttft_ms(uncached)
        return self.cold_ms(ctx) - self.cold_ms(max(0, ctx - uncached))

    def tpot_ms(self, running, kv_tokens=0.0, accept=None):
        """kv_tokens: total KV over the rows that would be resident, in tokens. accept: the
        worker's acceptance EWMA (tokens per step); None keeps the per-step prediction."""
        if self.step is not None:
            a = accept if accept else 1.0
            return self.step_ms(running, kv_tokens, a) / a
        if self.grid:
            per_step = self.grid_tpot_ms(running, kv_tokens)
        else:
            per_step = (self.decode_step_ms + self.decode_ms_per_seq * running
                        + self.decode_ms_per_seq_ktok * kv_tokens / 1e3)
        return per_step / accept if accept else per_step

    def grid_tpot_ms(self, running, kv_tokens):
        """Interpolate the calibrated (concurrency, mean context) -> ITL grid.

        Measured rather than fitted: no closed form matched H20's knee (see calibrate_h20.py).
        Bilinear inside the grid, and beyond its largest point, scale by total resident KV, which
        is the direction the measurements grow in.
        """
        conc = max(running, 1)
        ctx = kv_tokens / conc
        cs = sorted({c for c, _ in self.grid})
        xs = sorted({x for _, x in self.grid})

        def at(c, x):
            return self.grid[(c, x)]

        def bracket(v, vals):
            lo = max([u for u in vals if u <= v], default=vals[0])
            hi = min([u for u in vals if u >= v], default=vals[-1])
            return lo, hi

        if conc > cs[-1] or ctx > xs[-1]:
            # past the grid: hold the corner's KV-rate and scale by resident KV
            corner = at(cs[-1], xs[-1])
            return corner * (conc * ctx) / (cs[-1] * xs[-1])
        c0, c1 = bracket(conc, cs)
        x0, x1 = bracket(ctx, xs)
        fc = 0.0 if c1 == c0 else (conc - c0) / (c1 - c0)
        fx = 0.0 if x1 == x0 else (ctx - x0) / (x1 - x0)
        return ((1 - fc) * ((1 - fx) * at(c0, x0) + fx * at(c0, x1))
                + fc * ((1 - fx) * at(c1, x0) + fx * at(c1, x1)))


class Router:
    def __init__(self, a):
        self.a = a
        self.workers = [Worker(u, a.kv_tokens, BLOCK_TOK) for u in a.workers]
        self.pm = PerfModel(a.prefill_tok_s, a.decode_step_ms, a.decode_ms_per_seq,
                            a.decode_ms_per_seq_ktok)
        if a.decode_grid:
            rows = json.load(open(a.decode_grid))
            self.pm.grid = {(r["conc"], r["ctx"]): r["itl_ms"] for r in rows}
            print(f"decode grid: {len(self.pm.grid)} points", flush=True)
        if a.prefill_curve:
            self.pm.curve = sorted((int(n), float(t)) for n, t in
                                   (x.split(":") for x in a.prefill_curve.split(",")))
        if a.step_model:
            self.pm.step = json.load(open(a.step_model))["coef"]
            print(f"step model: {self.pm.step}", flush=True)
        self.pool = Worker("pool", a.pool_kv_tokens, BLOCK_TOK) if a.pool_kv_tokens > 0 else None
        self.pull_bps = a.pull_gbps * 1e9  # bytes/s, EWMA of observed pulls
        self.rr = 0
        self.rid = 0
        self.log = open(a.log, "a") if a.log else None
        self.session = None

    def pulled(self, wait_s, missing):
        """Tokens of a `missing`-token prefix prefetched from the pool into a node, and the wait
        actually spent, given a predicted queue wait of `wait_s` seconds."""
        a = self.a
        if missing < a.prefetch_threshold:
            return 0, wait_s
        full_s = missing * a.kv_bytes_per_token / self.pull_bps
        if a.prefetch_policy == "wait_complete":
            wait_s = max(wait_s, full_s)
        elif a.prefetch_policy == "timeout":
            wait_s = max(wait_s, min(full_s, a.prefetch_timeout_base
                                     + a.prefetch_timeout_per_ktok * missing / 1000.0))
        return min(missing, int(wait_s * self.pull_bps / a.kv_bytes_per_token)), wait_s

    def predict(self, w, uncached, deadline, n_tok_self=0, missing=0):
        """missing: prefix tokens the pool holds beyond w's own hit (0 without a pool)."""
        # Under EDF only requests with EARLIER deadlines are served ahead of this one.
        wait_ms = self.drain_ms(w, deadline)
        got, wait_s = self.pulled(wait_ms / 1000.0, missing)
        self.last_pull = got
        ttft = wait_s * 1000.0 + self.pm.prefill_ms(max(0, uncached - got), n_tok_self)
        # Every resident row's KV, including this request's prompt: the decode slowdown this
        # request would cause on w, which is what the hot-prefix holders actually suffer.
        kv = sum(r.get("kv", 0) for r in w.inflight.values()) + n_tok_self
        accept = w.accept if self.a.accept_ewma_alpha > 0 else None
        tpot = self.pm.tpot_ms(len(w.inflight) + 1, kv, accept)
        return ttft, tpot

    def drain_ms(self, w, deadline):
        """Prefill time of the work queued on w ahead of a request with this deadline."""
        return sum(self.pm.prefill_ms(r["prefill"], r.get("kv", r["prefill"]))
                   for r in w.inflight.values() if r["deadline"] <= deadline)

    def residents_itl_ok(self, w, uncached, n_tok_self, tpot, now):
        """Would admitting this request push any request already decoding on w past its ITL SLO?

        Admission costs a resident twice: this request's prefill chunks ride in its decode steps
        until the prefill ends (one chunk per step, `--chunk-tokens`), and from then on this request
        decodes beside it (`tpot` is the per-token step time with it resident). A resident's SLO is
        on its mean ITL, so the prediction adds the remaining steps to the decode time it has
        already spent rather than judging one step in isolation."""
        a = w.accept if (self.a.accept_ewma_alpha > 0 and w.accept) else 1.0
        step_ms = tpot * a
        n_chunks = -(-uncached // self.a.chunk_tokens) if uncached > 0 else 0
        chunk_ms = self.pm.prefill_ms(uncached, n_tok_self) / n_chunks if n_chunks else 0.0
        for r in w.inflight.values():
            if r.get("t_first") is None or r["decode_left"] <= 0:
                continue
            done = r["decode_total"] - r["decode_left"]
            rem_steps = r["decode_left"] / a
            total_ms = ((now - r["t_first"]) * 1000.0 + rem_steps * step_ms
                        + min(n_chunks, rem_steps) * chunk_ms)
            if total_ms / max(1, done + r["decode_left"]) > r["itl_slo"]:
                return False
        return True

    def feasible(self, w, uncached, deadline, now, n_tok_self=0, missing=0):
        ttft, tpot = self.predict(w, uncached, deadline, n_tok_self, missing)
        uncached = max(0, uncached - self.last_pull)
        if now + ttft / 1000.0 > deadline:
            self.last_reason = "own_ttft"
            return False, ttft, tpot
        if tpot > self.a.itl_slo_ms:
            self.last_reason = "own_itl"
            return False, ttft, tpot
        # Admitting this request delays LATER-deadline requests queued here; it must not
        # push any of them past their own deadline.
        for r in w.inflight.values():
            if r["deadline"] > deadline and r["decode_left"] == r["decode_total"]:
                t_r = now + (self.drain_ms(w, r["deadline"])
                             + self.pm.prefill_ms(uncached, n_tok_self)) / 1000.0
                if t_r > r["deadline"]:
                    self.last_reason = "queued_ttft"
                    return False, ttft, tpot
        if self.a.check_residents and not self.residents_itl_ok(w, uncached, n_tok_self, tpot, now):
            self.last_reason = "resident_itl"
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
        pool_hit = max(best, self.pool.hit(hashes)) if self.pool else 0
        if pol == "cache":
            # Ties on load are broken at random. A bare min() returned the lowest-index worker
            # whenever loads tied, and every session's first turn (and every chat request) ties at
            # hit 0, so workers 0 and 1 collected the sessions that affinity then pinned there:
            # 811 and 565 requests vs <=297 elsewhere in coloc_cache_mtp (2026-10-08).
            cands = [i for i, h in enumerate(hits) if h == best]
            i = min(cands, key=lambda j: (len(self.workers[j].inflight), random.random()))
            return i, {"hit": best * blk}
        # Order by hit length bucket (--rand-thresh blocks), then by load, then randomly. Pure
        # random within a bucket (measured on H20, coloc_rand) sent prefix-less chat requests --
        # for which EVERY worker ties at hit 0 -- to busy workers: chat TTFT 109 -> 487 ms vs the
        # load-aware tie-break of `cache`. Randomness is for spreading pulls among near-equal
        # cache holders; it must not override load.
        # cache_slo_rand drops the load term inside a hit bucket, so equal-hit holders are tried in
        # random order: the goal's "randomly pull from one of them" for congestion spreading.
        if pol == "cache_slo_rand":
            order = sorted(range(len(self.workers)),
                           key=lambda j: (-(hits[j] // max(1, self.a.rand_thresh)), random.random()))
        else:
            order = sorted(range(len(self.workers)),
                           key=lambda j: (-(hits[j] // max(1, self.a.rand_thresh)),
                                          len(self.workers[j].inflight), random.random()))
        if pol == "cache_rand":
            return order[0], {"hit": hits[order[0]] * blk}
        rejected_nodes = 0
        reasons = collections.Counter()
        best_slack, best_i = None, order[0]
        best_pred = None
        for i in order:
            uncached = max(0, n_tok - hits[i] * blk)
            missing = max(0, pool_hit - hits[i]) * blk if self.pool else 0
            ok, ttft, tpot = self.feasible(self.workers[i], uncached, deadline, now, n_tok, missing)
            slack = deadline - (now + ttft / 1000.0)
            if best_slack is None or slack > best_slack:
                best_slack, best_i = slack, i
                best_pred = (ttft, self.last_pull)
            if ok:
                return i, {"hit": hits[i] * blk, "rejected_nodes": rejected_nodes,
                           "reject_reasons": dict(reasons), "ttft_pred": ttft,
                           "pull_pred": self.last_pull, "pool_hit": pool_hit * blk}
            rejected_nodes += 1
            reasons[self.last_reason] += 1
        if self.a.no_feasible_fallback == "cache":
            # Overload: every node misses this request's SLO. Moving it then only adds recompute to
            # an already saturated cluster (192 x 16K shared-prefix mix: 99% of decisions had no
            # feasible node, and the most-slack fallback scattered requests off their prefixes --
            # chat TTFT 28 -> 73 s vs cache affinity), so take the top-hit node and do the least work.
            best_i = order[0]
            best_pred = (self.predict(self.workers[best_i], max(0, n_tok - hits[best_i] * blk), deadline,
                                      n_tok, max(0, pool_hit - hits[best_i]) * blk if self.pool else 0)[0],
                         self.last_pull)
        return best_i, {"hit": hits[best_i] * blk, "rejected_nodes": rejected_nodes,
                        "reject_reasons": dict(reasons), "least_bad": True,
                        "ttft_pred": best_pred[0], "pull_pred": best_pred[1],
                        "pool_hit": pool_hit * blk}

    async def generate(self, request):
        payload = await request.json()
        hashes, blk, n_tok = block_hashes(payload)
        out_tok = int((payload.get("sampling_params") or {}).get("max_new_tokens", 128))
        # Same length-scaled TTFT SLO the workers' EDF schedules against
        # (edf_scheduler.compute_ttft_slo_ms): base + 1 s per 5 000 prompt tokens beyond 4 096.
        slo_ms = self.a.ttft_slo_ms + max(0, n_tok - 4096) / 5000.0 * 1000.0
        deadline = time.time() + slo_ms / 1000.0
        i, info = self.choose(hashes, blk, n_tok, deadline)
        w = self.workers[i]
        self.rid += 1
        rid = self.rid
        uncached = max(0, n_tok - info.get("hit", 0) - info.get("pull_pred", 0))
        t_route = time.time()
        w.inflight[rid] = dict(prefill=uncached, decode_left=out_tok, decode_total=out_tok,
                               deadline=deadline, t0=t_route, kv=n_tok, t_first=None,
                               itl_slo=self.a.itl_slo_ms)
        w.insert(hashes)
        t_first = None
        if self.a.edf:
            payload["priority"] = int(deadline * 1000) % 2_000_000_000
        if self.log:
            self.log.write(json.dumps(dict(t=time.time(), rid=rid, worker=i, n_tok=n_tok,
                                           policy=self.a.policy, **info)) + "\n")
            self.log.flush()
        resp = web.StreamResponse()
        try:
            # One retry when the worker drops a reused keep-alive connection before answering:
            # ServerDisconnectedError, or ClientOSError (EPIPE) when the request write hits the
            # closed socket. Nothing has been streamed yet, so it is safe. The stock router retries
            # likewise; without it 1 in ~3 600 requests failed, and one EPIPE aborted a whole
            # concurrency-replay agent client (8-GPU C=32 coloc_cache_mtps, 2026-10-16).
            for attempt in (0, 1):
                try:
                    up = await self.session.post(w.url + "/generate", json=payload)
                    break
                except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError):
                    if attempt:
                        raise
            async with up:
                resp.content_type = up.headers.get("Content-Type", "application/json")
                resp.set_status(up.status)
                await resp.prepare(request)
                first = True
                tail = b""
                async for chunk in up.content.iter_any():
                    if first:
                        w.inflight[rid]["prefill"] = 0
                        t_first = time.time()
                        w.inflight[rid]["t_first"] = t_first
                        first = False
                    w.inflight[rid]["decode_left"] = max(0, w.inflight[rid]["decode_left"] - 1)
                    tail = (tail + chunk)[-65536:]
                    await resp.write(chunk)
                await resp.write_eof()
            # The worker's own account of this request: the last data chunk's meta_info carries
            # cached_tokens / prompt_tokens (the measured prefix hit the replication argument rests
            # on, not the router's block-hash estimate) and completion_tokens / spec_verify_ct
            # (this request's acceptance length, which feeds the worker's EWMA).
            meta = {}
            for line in reversed(tail.decode("utf-8", "replace").splitlines()):
                line = line.strip()
                if line.startswith("data:") and "meta_info" in line:
                    try:
                        meta = json.loads(line[5:].strip()).get("meta_info", {})
                    except ValueError:
                        pass
                    break
            if self.pool is not None:
                # Write-through: the pool holds this request's prefix once it has been computed.
                self.pool.insert(hashes)
                self.observe_pull(meta, info, n_tok, t_route, t_first)
            if meta and self.a.accept_ewma_alpha > 0:
                w.observe_accept(meta.get("completion_tokens"), meta.get("spec_verify_ct"),
                                 self.a.accept_ewma_alpha)
            if meta and self.log:
                self.log.write(json.dumps(dict(rid=rid, worker=i, done=time.time(),
                                               cached_tokens=meta.get("cached_tokens"),
                                               prompt_tokens=meta.get("prompt_tokens"),
                                               completion_tokens=meta.get("completion_tokens"),
                                               spec_verify_ct=meta.get("spec_verify_ct"),
                                               accept_ewma=w.accept,
                                               ttft_obs=(t_first - t_route) * 1000.0 if t_first else None,
                                               pull_bps=self.pull_bps if self.pool else None)) + "\n")
                self.log.flush()
        finally:
            w.inflight.pop(rid, None)
        return resp

    def observe_pull(self, meta, info, n_tok, t_route, t_first):
        """Update the pull-bandwidth EWMA from one finished request. Pulled tokens = the worker's
        cached_tokens (device + host + prefetched pages) minus the router's local-hit estimate; the
        window is the queue wait = TTFT - the request's own prefill of the rest. Only a pull that
        stopped short of the pool's hit is bandwidth-bound; a complete one is a lower bound, so it
        may only raise the estimate."""
        c = meta.get("cached_tokens") if meta else None
        if c is None or t_first is None:
            return
        pulled = c - info.get("hit", 0)
        missing = info.get("pool_hit", 0) - info.get("hit", 0)
        wait_s = (t_first - t_route) - self.pm.prefill_ms(max(0, n_tok - c), n_tok) / 1000.0
        if pulled < self.a.prefetch_threshold or wait_s < 0.05:
            return
        sample = pulled * self.a.kv_bytes_per_token / wait_s
        if pulled < 0.95 * missing or sample > self.pull_bps:
            al = self.a.pull_ewma_alpha
            self.pull_bps = (1 - al) * self.pull_bps + al * sample

    async def broadcast(self, request):
        res = []
        for w in self.workers:
            async with self.session.request(request.method, w.url + request.path) as r:
                res.append(r.status)
        return web.json_response({"status": res})

    async def health(self, request):
        return web.Response(text="ok")

    async def models(self, request):
        # bench_serving's ready check polls /v1/models; every worker serves the same model.
        async with self.session.get(self.workers[0].url + "/v1/models") as r:
            return web.Response(body=await r.read(), status=r.status, content_type="application/json")

    async def start(self, app):
        # Pooled connections expire before uvicorn's 5 s keep-alive, so the router rarely reuses a
        # socket the worker is closing.
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None),
                                             connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", nargs="+", required=True)
    ap.add_argument("--port", type=int, default=31000)
    ap.add_argument("--policy", choices=["rr", "cache", "cache_rand", "cache_slo", "cache_slo_rand"],
                    default="cache_slo")
    ap.add_argument("--rand-thresh", type=int, default=4, help="blocks within which hits tie")
    ap.add_argument("--edf", action="store_true")
    ap.add_argument("--ttft-slo-ms", type=float, default=5000)
    ap.add_argument("--itl-slo-ms", type=float, default=100)
    ap.add_argument("--check-residents", type=int, default=1,
                    help="also reject a node if admitting the request would push a request already "
                         "decoding there past its ITL SLO")
    ap.add_argument("--chunk-tokens", type=int, default=2048, help="workers' --chunked-prefill-size")
    ap.add_argument("--no-feasible-fallback", choices=["slack", "cache"], default="slack",
                    help="when no node is predicted to meet the SLO: the most-slack node, or the top-hit node")
    ap.add_argument("--kv-tokens", type=int, default=1_000_000)
    ap.add_argument("--prefill-tok-s", type=float, default=20000)
    ap.add_argument("--decode-step-ms", type=float, default=15)
    ap.add_argument("--decode-ms-per-seq", type=float, default=0.3)
    ap.add_argument("--decode-ms-per-seq-ktok", type=float, default=0.0)
    ap.add_argument("--decode-grid", default="", help="calibrate_h20.py decode_grid.json")
    ap.add_argument("--prefill-curve", default="",
                    help="cold-prefill TTFT curve 'tokens:ms,...' (calibrate_h20.py); overrides --prefill-tok-s")
    ap.add_argument("--step-model", default="",
                    help="fit_mtp_step.py JSON: whole-step ms model (with c_acc if fitted)")
    ap.add_argument("--accept-ewma-alpha", type=float, default=0.0,
                    help=">0: per-worker acceptance EWMA; predicted TPOT = step ms / acceptance")
    ap.add_argument("--log", default="")
    # KV transfer from the cluster-wide L3 pool (HiCache + mooncake), see the module docstring.
    ap.add_argument("--pool-kv-tokens", type=int, default=0, help="pool capacity; 0 = no pool")
    ap.add_argument("--kv-bytes-per-token", type=int, default=57344,
                    help="Qwen2.5-7B bf16: 28 layers x 2 x 4 KV heads x 128 x 2 B")
    ap.add_argument("--pull-gbps", type=float, default=10.0, help="initial pull bandwidth, GB/s")
    ap.add_argument("--pull-ewma-alpha", type=float, default=0.2)
    ap.add_argument("--prefetch-policy", choices=["best_effort", "timeout", "wait_complete"],
                    default="best_effort", help="must match the workers' setting")
    ap.add_argument("--prefetch-threshold", type=int, default=256)
    ap.add_argument("--prefetch-timeout-base", type=float, default=1.0)
    ap.add_argument("--prefetch-timeout-per-ktok", type=float, default=0.25)
    a = ap.parse_args()
    r = Router(a)
    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(r.start)
    app.router.add_post("/generate", r.generate)
    app.router.add_route("*", "/flush_cache", r.broadcast)
    app.router.add_get("/health", r.health)
    app.router.add_get("/v1/models", r.models)
    web.run_app(app, host="0.0.0.0", port=a.port, access_log=None)


if __name__ == "__main__":
    main()
