"""CPU checks for router.py's cache_slo_h policy (prioritized placement for SLO-budgeted workers).
  python test_router_slo_h.py
Claude, 2026-10-27.
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import router  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def make(nodes=3):
    model = dict(S=3, T=4, gap=dict(coef=[10.0, 0.004, 0.5, 0.1, 0.001, 0.0, 1.0, 0.0, 0.0, 0.0]))
    path = os.path.join(tempfile.mkdtemp(), "m.json")
    json.dump(model, open(path, "w"))
    a = router.build_parser().parse_args(
        ["--workers"] + [f"http://w{i}" for i in range(nodes)]
        + ["--policy", "cache_slo_h", "--budget-chunk", f"{path}:0.1:50:80:24000:16384", "--ttft-slo-ms", "10000",
           "--prefill-curve", "4096:461,8192:947,16384:2094,32768:5005,49152:8734"])
    return router.Router(a)


def hashes_for(tokens, tag=0):
    return [hash((tag, k)) for k in range(tokens // router.BLOCK_TOK)]


def chat_rows(w, n, now, behind_ms_per_token=0.0, kv=1000):
    """n chat requests decoding on w: 20 tokens emitted, each at 40 ms (+ behind) per token."""
    for k in range(n):
        w.inflight[f"c{k}"] = dict(prefill=0, decode_left=100, decode_total=120, deadline=now - 1, t0=now - 2,
                                   kv=kv, t_first=now - 20 * (40 + behind_ms_per_token) / 1000.0, itl_slo=50)


def main():
    now = time.time()
    # 1. idle cluster: the cache holder wins
    r = make()
    hs = hashes_for(32768, tag=1)
    r.workers[1].insert(hs)
    i, info = r.choose(hs, router.BLOCK_TOK, 33792, now + 10 + 6)
    check(i == 1 and not info["least_bad"], f"idle cluster: request goes to its cache holder (got {i}, {info['ttft_pred']:.0f} ms)")
    # 2a. light decode load on the holder: a row at its TPOT target still earns ~s x a per step, so a short append
    #     fits beside it and the holder keeps the request
    r = make()
    r.workers[1].insert(hs)
    chat_rows(r.workers[1], 24, now, behind_ms_per_token=9.5)
    i, info = r.choose(hs, router.BLOCK_TOK, 33792, now + 4.0)
    check(i == 1, f"light decode load: the holder keeps the request (got {i}, {info['ttft_pred']:.0f} ms)")
    # 2b. simulation dynamics: 300 chat rows at their TPOT target with a ~47 ms decode step (below the 50 ms target):
    #     the first step has no room, decode-only steps bank ~3 ms of slack each, and the append then proceeds; the
    #     wait is longer than one step but finite
    r = make()
    chat_rows(r.workers[1], 300, now, behind_ms_per_token=9.9, kv=20000)
    t_tight = r.budget_prefill_ms(r.workers[1], 1024, 33792, now)
    check(60 < t_tight < 5000, f"rows at their target: a 1K append waits for banked slack, then runs ({t_tight:.0f} ms)")
    # 2c. ordering: given a slow holder (predicted 5 s) and fast idle nodes, the request goes to a feasible non-holder
    r = make()
    r.workers[1].insert(hs)
    real = r.budget_prefill_ms
    r.budget_prefill_ms = lambda w, u, c, t: 5000.0 if w is r.workers[1] else real(w, u, c, t)
    i, info = r.choose(hs, router.BLOCK_TOK, 33792, now + 4.0)
    check(i != 1 and not info["least_bad"], f"slow holder: request placed on a feasible non-holder (got {i}, feasible nodes {info['feasible_nodes']})")
    # 3. nothing feasible: least total lateness, never held
    r = make(nodes=2)
    for w in r.workers:
        chat_rows(w, 24, now, behind_ms_per_token=9.9)
    i, info = r.choose(hashes_for(65536, tag=2), router.BLOCK_TOK, 65536, now + 0.5)
    check(info["least_bad"] and i in (0, 1), f"no feasible node: least-lateness node chosen, request still sent (got {i})")
    # 4. equal hits: the node where the request pushes no queued request late wins
    r = make(nodes=2)
    hs3 = hashes_for(16384, tag=3)
    for w in r.workers:
        w.insert(hs3)
    r.workers[0].inflight["q"] = dict(prefill=30000, decode_left=128, decode_total=128, deadline=now + 6.5, t0=now,
                                      kv=30000, t_first=None, itl_slo=80)
    i, info = r.choose(hs3, router.BLOCK_TOK, 16384 + 4096, now + 6.0)
    check(i == 1, f"equal hits: the node where no queued request is pushed late wins (got {i}, added {info['added_lateness_s']:.2f} s)")
    # 5. monotone in load
    r = make(nodes=1)
    w = r.workers[0]
    base = r.budget_prefill_ms(w, 8192, 8192, now)
    chat_rows(w, 8, now)
    loaded = r.budget_prefill_ms(w, 8192, 8192, now)
    check(loaded > base, f"budget simulation grows with decode load ({base:.0f} -> {loaded:.0f} ms)")
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
