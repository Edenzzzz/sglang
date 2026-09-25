#!/usr/bin/env python3
"""Simulate the batch mixture a FUSED draft+verify+prefill kernel would see.

Today these are three disjoint ForwardModes and every speculative algorithm
force-disables mixed chunking (server_args.py:1505/1578), so this composition
cannot be observed on a running server -- it has to be simulated. The point of
the simulation is to produce a *realistic workload* (not hand-picked shapes) to
hand to kernel work on the 3-input-mode fusion.

Request stream is real: arrival times, prompt lengths and output lengths are
replayed from bench_serving `--output-details` captures.

Scheduling follows SparseSpec's unified scheduler (MLSys 2026): instead of k
draft steps followed by one verify, every running request sits in one of k+1
stages and each step mixes them, so a batch always contains ~k/(k+1) drafting
rows and ~1/(k+1) verifying rows. New requests are greedily bin-packed into the
least-loaded stage. Chunked prefill is admitted into the same batch on top.

Draft model: assumed a separate draft (EAGLE-style), NOT the target's own
layers. For batch *composition* this only matters in that a draft row attends
the sequence's current context, which is what is modelled here.
"""
import argparse, json, os, random
from dataclasses import dataclass, field


@dataclass
class Req:
    rid: int
    arrive: float
    prompt: int
    output: int
    prefilled: int = 0          # prompt tokens already processed
    ctx: int = 0                # KV tokens resident (= attended length)
    produced: int = 0
    stage: int = 0              # 0..k-1 draft stages, k = verify stage
    done: bool = False


def load_trace(paths, limit=None):
    """Real (arrival, prompt_len, output_len) triples from bench_serving JSONs."""
    reqs = []
    for p in paths:
        d = json.load(open(p))
        ins, outs = d["input_lens"], d["output_lens"]
        rate = d.get("request_rate", 0) or 0
        rnd = random.Random(0)
        t = 0.0
        for i, (a, b) in enumerate(zip(ins, outs)):
            if not a or not b:
                continue
            # bench_serving issues a Poisson stream at `request_rate`; rate=inf
            # means the whole burst is admitted at t=0.
            if rate and rate != float("inf"):
                t += rnd.expovariate(rate)
            reqs.append((t, int(a), int(b)))
    reqs.sort()
    if limit:
        reqs = reqs[:limit]
    return reqs


def simulate(trace, *, k, draft_tokens, alpha, max_running, chunk_budget,
             kv_pool, step_ms, max_steps):
    """Yield one composition record per fused step."""
    pend = [Req(i, t, p, o) for i, (t, p, o) in enumerate(trace)]
    pend.reverse()                      # pop() = earliest
    running, buckets = [], [0] * (k + 1)
    kv_used = 0
    now = 0.0
    rnd = random.Random(1)

    for step in range(max_steps):
        now += step_ms / 1000.0
        # ---- admit arrivals into the least-loaded stage (greedy bin-packing)
        while pend and pend[-1].arrive <= now and len(running) < max_running:
            r = pend.pop()
            r.stage = buckets.index(min(buckets))
            buckets[r.stage] += 1
            running.append(r)

        draft, verify, prefill = [], [], []
        budget = chunk_budget
        for r in running:
            if r.prefilled < r.prompt:
                # chunked prefill shares the batch with draft/verify rows
                take = min(r.prompt - r.prefilled, budget)
                if take > 0:
                    prefill.append((take, r.prefilled))   # (chunk tokens, attended prefix)
                    budget -= take
                    r.prefilled += take
                    if r.prefilled >= r.prompt:
                        r.ctx = r.prompt
                        kv_used += r.prompt
            elif r.stage == k:
                verify.append((draft_tokens, r.ctx))
            else:
                draft.append((1, r.ctx))

        if draft or verify or prefill:
            yield dict(step=step, t=round(now, 4),
                       draft=draft, verify=verify, prefill=prefill,
                       running=len(running), kv_used=kv_used)

        # ---- advance stages; verify commits accepted tokens
        for r in running:
            if r.prefilled < r.prompt:
                continue
            if r.stage == k:
                acc = 1 + sum(1 for _ in range(k) if rnd.random() < alpha)
                acc = min(acc, draft_tokens, r.output - r.produced)
                r.produced += acc
                r.ctx += acc
                kv_used += acc
                if r.produced >= r.output:
                    r.done = True
            r.stage = (r.stage + 1) % (k + 1)

        for r in [x for x in running if x.done]:
            buckets[r.stage] = max(0, buckets[r.stage] - 1)
            kv_used -= r.ctx
            running.remove(r)
        if kv_pool and kv_used > kv_pool:      # admission pressure
            max_running = max(1, len(running))
        if not running and not pend:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", help="bench_serving --output-details JSONs")
    ap.add_argument("--out", default="tasks/fused_specdec_workload/fused_batches.jsonl")
    ap.add_argument("--k", type=int, default=3, help="speculative steps")
    ap.add_argument("--draft-tokens", type=int, default=4, help="candidates verified (k+1)")
    ap.add_argument("--alpha", type=float, default=0.7, help="per-token acceptance rate")
    ap.add_argument("--max-running", type=int, default=48)
    ap.add_argument("--chunk-budget", type=int, default=8192)
    ap.add_argument("--kv-pool", type=int, default=980_000)
    ap.add_argument("--step-ms", type=float, default=15.0)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    trace = load_trace(a.traces, a.limit)
    print(f"{len(trace)} real requests from {len(a.traces)} trace(s)")
    recs = list(simulate(trace, k=a.k, draft_tokens=a.draft_tokens, alpha=a.alpha,
                         max_running=a.max_running, chunk_budget=a.chunk_budget,
                         kv_pool=a.kv_pool, step_ms=a.step_ms, max_steps=a.max_steps))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"{len(recs)} fused steps -> {a.out}")
    n3 = sum(1 for r in recs if r["draft"] and r["verify"] and r["prefill"])
    n2 = sum(1 for r in recs if sum(bool(r[m]) for m in ("draft", "verify", "prefill")) == 2)
    print(f"  all three modes present: {n3} ({100*n3/max(len(recs),1):.1f}%), "
          f"exactly two: {n2} ({100*n2/max(len(recs),1):.1f}%)")


if __name__ == "__main__":
    main()
