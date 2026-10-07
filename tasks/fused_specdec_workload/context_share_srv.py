#!/usr/bin/env python3
"""Decode step time at (batch, context) through a running sglang server, for batches whose
one-shot prefill does not fit bench_one_batch. B concurrent requests of L random ids each stream
OUT tokens (ignore_eos); once all B are decoding, every gap is one decode step, so the median of
the gaps in each stream's second half is the step time. Appends rows in bench_one_batch's
result.jsonl format, which context_share.py reads.

  python context_share_srv.py <url> <out.jsonl> B:L [B:L ...]
"""
import concurrent.futures as cf, json, random, statistics, sys, time, urllib.request

URL, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
OUTLEN = 128


def one(ids):
    body = {"input_ids": ids, "stream": True,
            "sampling_params": {"max_new_tokens": OUTLEN, "temperature": 0, "ignore_eos": True}}
    req = urllib.request.Request(URL + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    ts = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                ts.append(time.time())
    return ts


rng = random.Random(0)
for spec in sys.argv[3:]:
    B, L = map(int, spec.split(":"))
    reqs = [[rng.randrange(1000, 30000) for _ in range(L)] for _ in range(B)]
    with cf.ThreadPoolExecutor(B) as ex:
        streams = list(ex.map(one, reqs))
    # gaps after the LAST stream's first token: from then on every request is decoding
    start = max(s[0] for s in streams)
    gaps = [b - a for s in streams for a, b in zip(s, s[1:]) if a >= start]
    gaps = gaps[len(gaps) // 4:]
    step = statistics.median(gaps)
    with open(OUT, "a") as f:
        f.write(json.dumps({"batch_size": B, "input_len": L, "output_len": OUTLEN,
                            "median_decode_latency": step, "n_gaps": len(gaps), "via": "server"}) + "\n")
    print(f"B={B} L={L} step={step * 1e3:.2f} ms (n={len(gaps)})", flush=True)
