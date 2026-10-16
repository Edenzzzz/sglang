"""Cross-pod audit of finished multi-node arms (runs locally via px_multi).

  python mn_audit.py <arm> [<arm> ...]

Per arm: tracebacks and fallbacks summed over EVERY pod's worker logs (mn_run's own summary only
sees pod 0), requests served per worker (from each worker's access log), and from the router's
decision log the cache-hit token share and SLO check counts.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from px_multi import Cluster

C = Cluster(os.environ.get("SID", "c2fb3965f36cd15f"), "cn-seed", nodes=[f"worker-{i}" for i in range(int(os.environ.get("NPODS", "8")))])
W = "/dev/shm/w"
for arm in sys.argv[1:]:
    d = f"{W}/mn/{arm}"
    tb = fb = 0
    new_tok = cached_tok = 0
    served = []
    for r, (rc, out) in enumerate(C.run_all(
            f"for f in {d}/worker_r*_g?.log; do echo \"$(basename $f) $(grep -c Traceback $f) "
            f"$(grep -ci fallback $f) $(grep -c 'POST /generate' $f) "
            f"$(grep -oE '#new-token: [0-9]+, #cached-token: [0-9]+' $f | awk '{{n+=$2; c+=$4}} END {{print n+0, c+0}}')\"; done")):
        for line in out.splitlines():
            p = line.split()
            if len(p) == 6 and p[0].startswith("worker_"):
                tb += int(p[1]); fb += int(p[2]); served.append(int(p[3]))
                new_tok += int(p[4]); cached_tok += int(p[5].rstrip(","))
    rc, out = C.run(0, f"python3 -c \"import json\nn=h=v=t=ac=ap=0\nfor l in open('{d}/router_decisions.jsonl'):\n"
                       f"    r=json.loads(l)\n    if 'cached_tokens' in r:\n        ac+=r['cached_tokens'] or 0; ap+=r['prompt_tokens'] or 0; continue\n"
                       f"    n+=1; h+=r.get('hit',0); t+=r.get('n_tok',0); v+=r.get('rejected_nodes', r.get('vetoed',0)) or 0\n"
                       f"print('ROUTER', n, h, t, v, ac, ap)\" 2>/dev/null")
    rl = [l for l in out.splitlines() if l.startswith("ROUTER")]
    rinfo = ""
    if rl:
        _, n, h, t, v, ac, ap = rl[-1].split()
        rinfo = (f"decisions={n} predicted_hit={int(h) / max(1, int(t)):.3f} nodes_rejected_by_slo_check={v} "
                 f"WORKER-REPORTED hit={int(ac) / max(1, int(ap)):.3f} ({ac}/{ap} tok)")
    spread = f"min={min(served)} max={max(served)}" if served else "n/a"
    hit = cached_tok / max(1, new_tok + cached_tok)
    print(f"{arm:<12} workers={len(served)} tracebacks={tb} fallbacks={fb} requests/worker {spread} "
          f"ACTUAL prefill cache hit={hit:.3f} (cached {cached_tok} / {new_tok + cached_tok} tok) | router: {rinfo}")
