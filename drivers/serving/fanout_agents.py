"""Fan-out agent workload for PD point 2b: shared long task prefixes, many parallel programs per task, shifting
popularity -- the case where cache replication + load balancing must matter.

K tasks, each a shared prefix of L tokens (a repository / long system context). Programs are kept active at a fixed
concurrency C (program concurrency is the agent load axis); each new program picks a task by a Zipf(s) draw over the
current popularity ranking, which rotates by one place every ROTATE seconds, so a new task's fan-out starts while the
previous hot task drains. A program runs R rounds: prompt = task prefix + its own history (appends and outputs); each
round streams one /generate, then waits a tool-call gap G and appends A fresh tokens. Prefix and appends are fixed-seed
random token ids: cache and transfer behaviour depend on lengths and sharing, not on content (speculative acceptance
does, so MTP arms should use a natural-text prefix source).

  python fanout_agents.py --url http://HOST:PORT --tasks 8 --prefix-len 131072 --concurrency 32 --rounds 4 \
      --duration 900 --out run.json

Writes per-request records (task, program, round, input length, TTFT, mean ITL, end-to-end, start time, popularity
epoch) and prints TTFT / ITL / throughput summaries, per-epoch TTFT, and TTFT recovery after each rotation.
Claude, 2026-10-18.
"""
import argparse
import json
import random
import threading
import time
import urllib.request

import numpy as np


def stream_generate(url, input_ids, max_new_tokens, timeout):
    body = {"input_ids": input_ids, "stream": True,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens, "ignore_eos": True}}
    req = urllib.request.Request(f"{url}/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps, meta, out_ids = [], None, []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            stamps.append(time.perf_counter())
            meta = d.get("meta_info", meta)
            # Streamed chunks carry the cumulative output ids at top level (non-incremental streaming).
            out_ids = d.get("output_ids") or (meta or {}).get("output_ids") or out_ids
    return t0, stamps, meta, out_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--prefix-len", type=int, default=131072)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--append-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--tool-gap-s", type=float, default=5.0)
    ap.add_argument("--zipf", type=float, default=1.2)
    ap.add_argument("--rotate-s", type=float, default=180.0)
    ap.add_argument("--duration", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    prefixes = [[rng.randrange(1000, 30000) for _ in range(a.prefix_len)] for _ in range(a.tasks)]
    weights = np.array([1.0 / (i + 1) ** a.zipf for i in range(a.tasks)])
    weights /= weights.sum()
    t_start = time.perf_counter()
    lock = threading.Lock()
    records = []
    program_id = [0]

    def epoch(t):
        return int((t - t_start) // a.rotate_s)

    def pick_task(prng):
        # Rank r holds task (r + epoch) % K: the hot set moves by one task per rotation.
        rank = prng.choices(range(a.tasks), weights=weights)[0]
        return (rank + epoch(time.perf_counter())) % a.tasks

    def program_loop(worker):
        prng = random.Random(a.seed * 1000 + worker)
        while time.perf_counter() - t_start < a.duration:
            with lock:
                pid = program_id[0]
                program_id[0] += 1
            task = pick_task(prng)
            history = list(prefixes[task])
            for rnd in range(a.rounds):
                if time.perf_counter() - t_start >= a.duration:
                    return
                history += [prng.randrange(1000, 30000) for _ in range(a.append_len)]
                sent = time.perf_counter() - t_start
                try:
                    t0, stamps, meta, out_ids = stream_generate(a.url, history, a.output_len, a.timeout)
                except Exception as exc:
                    with lock:
                        records.append(dict(task=task, program=pid, round=rnd, input_len=len(history),
                                            error=str(exc)[:200], start=sent))
                    break
                itl = np.diff(stamps).mean() * 1e3 if len(stamps) > 1 else None
                rec = dict(task=task, program=pid, round=rnd, input_len=len(history),
                           ttft=(stamps[0] - t0) if stamps else None, itl_ms=itl,
                           e2e=(stamps[-1] - t0) if stamps else None, start=t0 - t_start, epoch=epoch(t0),
                           cached=(meta or {}).get("cached_tokens"), out_tokens=len(out_ids))
                with lock:
                    records.append(rec)
                history += out_ids
                time.sleep(a.tool_gap_s)

    threads = [threading.Thread(target=program_loop, args=(w,), daemon=True) for w in range(a.concurrency)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    wall = time.perf_counter() - t_start
    json.dump(dict(args=vars(a), wall_s=wall, records=records), open(a.out, "w"))

    ok = [r for r in records if r.get("ttft") is not None]
    errs = len(records) - len(ok)
    tt = np.array([r["ttft"] for r in ok])
    itl = np.array([r["itl_ms"] for r in ok if r["itl_ms"] is not None])
    toks = sum(r["out_tokens"] for r in ok)
    print(f"FANOUT requests={len(ok)} errors={errs} wall={wall:.0f}s out_tok/s={toks / wall:.1f} "
          f"TTFT mean={tt.mean():.2f}s p50={np.median(tt):.2f}s p99={np.percentile(tt, 99):.2f}s "
          f"ITL mean={itl.mean():.1f}ms p99={np.percentile(itl, 99):.1f}ms")
    cached = [r["cached"] / r["input_len"] for r in ok if r.get("cached") is not None]
    if cached:
        print(f"FANOUT prefix cached fraction mean={np.mean(cached):.3f}")
    # TTFT right after each rotation vs the rest of the epoch: how fast replicas follow the new hot task.
    for e in sorted({r["epoch"] for r in ok}):
        ep = [r for r in ok if r["epoch"] == e]
        lo = e * a.rotate_s
        first = [r["ttft"] for r in ep if r["start"] < lo + 30]
        rest = [r["ttft"] for r in ep if r["start"] >= lo + 30]
        print(f"FANOUT epoch {e}: n={len(ep)} TTFT first-30s={np.mean(first) if first else float('nan'):.2f}s "
              f"rest={np.mean(rest) if rest else float('nan'):.2f}s")


if __name__ == "__main__":
    main()
