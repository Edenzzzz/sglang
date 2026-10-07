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
dec = []
for conc in (1, 8, 32, 64):
    def one(_):
        _, ts = gen([rng.randrange(1000, 30000) for _ in range(256)], 256, True)
        return [(b - a) * 1000 for a, b in zip(ts, ts[1:])]
    with cf.ThreadPoolExecutor(conc) as ex:
        itls = [x for lst in ex.map(one, range(conc)) for x in lst[20:]]
    m = sum(itls) / max(1, len(itls))
    dec.append((conc, m))
    print(f"decode conc {conc:3d}: mean ITL {m:6.2f} ms", flush=True)
# least squares ITL = a + b * conc
n = len(dec); sx = sum(c for c, _ in dec); sy = sum(m for _, m in dec)
sxx = sum(c * c for c, _ in dec); sxy = sum(c * m for c, m in dec)
b = (n * sxy - sx * sy) / (n * sxx - sx * sx); a = (sy - b * sx) / n
long_rate = sum(n_ for n_, _ in pre[2:]) / sum(t for _, t in pre[2:])
print(f"ROUTER_ARGS=--prefill-tok-s {long_rate:.0f} --decode-step-ms {a:.2f} --decode-ms-per-seq {b:.3f}")
