"""Calibrate the router's perf model on one H20 worker (run on a pod, worker already up).

  python calibrate_h20.py <worker_url>

Prefill: sequential cache-miss prompts of known length -> TTFT -> prefill tokens/s per length.
Decode: N concurrent short-prompt requests with long outputs -> mean ITL vs N -> the per-step
base cost and per-sequence slope that router.py's PerfModel(decode_step_ms, decode_ms_per_seq)
uses. Prints the router flags to use.
"""
import concurrent.futures as cf, json, random, sys, time, urllib.request

URL = sys.argv[1].rstrip("/")


def gen(ids, max_new, stream):
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": max_new, "temperature": 0,
                                                  "ignore_eos": True}, "stream": stream}
    req = urllib.request.Request(URL + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    times = []
    with urllib.request.urlopen(req, timeout=900) as r:
        if not stream:
            r.read()
            return time.time() - t0, []
        for line in r:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                times.append(time.time())
    return times[0] - t0 if times else None, times


rng = random.Random(0)
gen([rng.randrange(1000, 30000) for _ in range(64)], 4, False)  # warm
pre = []
for n in (4096, 8192, 16384, 32768, 49152):
    ids = [rng.randrange(1000, 30000) for _ in range(n)]  # fresh tokens: guaranteed cache miss
    ttft, _ = gen(ids, 1, True)
    pre.append((n, ttft))
    print(f"prefill {n:6d} tok: TTFT {ttft * 1000:8.1f} ms  -> {n / ttft:8.0f} tok/s", flush=True)
# Decode ITL vs concurrency at SEVERAL context lengths. The first calibration used 256-token
# prompts only, which fits a slope of 0.065 ms/seq -- so the router's TPOT prediction never
# crossed a 100 ms budget, and the SLO check never fired on an overloaded hot-prefix holder
# (coloc_slo_edf_hot, 2026-10-08). A decode row's cost grows with its KV, so the slope has to be
# per KV token, not per sequence.
dec = []
for ctx in (256, 4096, 16384):
    for conc in (1, 8, 32, 64):
        # Prefill every prompt first, then time a second, fully cached pass of the same prompts. Timing
        # the first pass (all conc requests submitted at once) put the other requests' prefill chunks
        # inside the decode steps: 32 x 16K = 512K prefill tokens at ~5K tok/s, so the "decode" grid
        # read 210 ms at 32 x 16K where serving measured 35-84 ms at similar rows and context. That was
        # the unexplained knee noted below.
        prompts = [[rng.randrange(1000, 30000) for _ in range(ctx)] for _ in range(conc)]
        with cf.ThreadPoolExecutor(conc) as ex:
            list(ex.map(lambda p: gen(p, 1, False), prompts))

        def one(p):
            _, ts = gen(p, 192, True)
            return [(b - a) * 1000 for a, b in zip(ts, ts[1:])]
        with cf.ThreadPoolExecutor(conc) as ex:
            itls = [x for lst in ex.map(one, prompts) for x in lst[20:]]
        m = sum(itls) / max(1, len(itls))
        dec.append((ctx, conc, m))
        print(f"decode ctx {ctx:6d} conc {conc:3d}: mean ITL {m:6.2f} ms", flush=True)

# Emit the measured grid for the router to interpolate. No parametric fit: on H20 the surface has
# a knee no closed form matched (2026-10-08) -- measured with prefill inside the timed steps, see above. ITL is flat to ~131K total resident tokens (9.1 ms at
# 32x4K, 8.3 ms at 8x16K) and then jumps 15x by 524K (136.6 ms at 32x16K) -- too steep for a
# bandwidth bound (29 GB per step would be ~7 ms at HBM speed), so it is a capacity effect
# (eviction/retraction or an attention-backend path change) that still needs a root cause.
# Both a 3-term linear fit and a max(flat, kv/bw) form were rejected: the linear one returned a
# NEGATIVE base cost, the max form overpredicted the middle of the grid by 3-4x.
import json
json.dump([{"ctx": x, "conc": c, "itl_ms": m} for x, c, m in dec],
          open("/dev/shm/w/mn/decode_grid.json", "w"), indent=1)
print("decode grid -> /dev/shm/w/mn/decode_grid.json")

# ITL = a + b*conc + c*conc*ctx/1e3 (least squares on the three terms)
X = [[1.0, float(c), c * x / 1e3] for x, c, _ in dec]
y = [m for _, _, m in dec]
# normal equations, 3x3, solved by Gaussian elimination
A = [[sum(X[k][i] * X[k][j] for k in range(len(X))) for j in range(3)] + [sum(X[k][i] * y[k] for k in range(len(X)))] for i in range(3)]
for i in range(3):
    pv = max(range(i, 3), key=lambda r: abs(A[r][i])); A[i], A[pv] = A[pv], A[i]
    for r in range(3):
        if r != i and A[i][i]:
            f = A[r][i] / A[i][i]
            A[r] = [A[r][j] - f * A[i][j] for j in range(4)]
a, b, c = (A[i][3] / A[i][i] for i in range(3))
long_rate = sum(n_ for n_, _ in pre[2:]) / sum(t for _, t in pre[2:])
print(f"rejected linear fit (kept for the record): ITL = {a:.2f} + {b:.4f}*conc + {c:.5f}*conc*kv/1e3 ms")
curve = ",".join(f"{n_}:{t * 1000:.1f}" for n_, t in pre)
print(f"ROUTER_ARGS=--prefill-tok-s {long_rate:.0f} --prefill-curve {curve} --decode-grid /dev/shm/w/mn/decode_grid.json")
