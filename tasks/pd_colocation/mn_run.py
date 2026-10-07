"""Multi-node arm orchestrator (runs locally; drives the H20 pods through px_multi).

  python mn_run.py <arm> [<arm> ...]          arms: see ARMS
  env: SID (job run), NPODS (8), plus mn_bench.sh knobs (SESSIONS, SESSION_RATE, CHAT_RATE, ...)

Every arm: kill leftovers -> start workers in the arm's layout -> wait until every worker's
/health is 200 -> start the arm's router on pod 0 -> run mn_bench.sh -> print the per-stream
summary -> tear down. Logs and JSON stay on pod 0 under /dev/shm/w/mn/<arm>/.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from px_multi import Cluster

SID = os.environ.get("SID", "c2fb3965f36cd15f")
NPODS = int(os.environ.get("NPODS", "8"))
W = "/dev/shm/w"
C = Cluster(SID, "cn-seed", nodes=[f"worker-{i}" for i in range(NPODS)])
RPORT = 31000

ARMS = {
    # name: (layout per pod, router)
    "coloc_rr":    (["coloc"] * NPODS, "ours:rr"),
    "coloc_cache": (["coloc"] * NPODS, "ours:cache"),
    "coloc_rand":  (["coloc"] * NPODS, "ours:cache_rand"),
    "coloc_slo":   (["coloc"] * NPODS, "ours:cache_slo"),
    "coloc_sgl":   (["coloc"] * NPODS, "sgl:cache_aware"),
    "disagg_2p2d": (["prefill"] * (NPODS // 2) + ["decode"] * (NPODS // 2), "sgl:pd"),
    "disagg_1p3d": (["prefill"] * (NPODS // 4) + ["decode"] * (NPODS - NPODS // 4), "sgl:pd"),
}


def hosts():
    rc, out = C.run(0, "echo HOSTS=$ARNOLD_WORKER_HOSTS")
    line = [l for l in out.splitlines() if l.startswith("HOSTS=")][-1][6:]
    return [h.rsplit(":", 1)[0].strip("[]") for h in line.split(",")]


PORTS = {}  # role -> [(http_port, bootstrap_port) for gpu 0, 1], read from each pod


def read_ports(log):
    for r, (rc, out) in enumerate(C.run_all(f"cat {log}/ports_r${{ARNOLD_ID}}_g*.txt 2>/dev/null")):
        rows = [l.split() for l in out.splitlines() if len(l.split()) == 3 and l.split()[0] in ("0", "1")]
        PORTS[r] = [(int(p), int(b)) for _, p, b in sorted(rows)]


def urls(hs, roles=None):
    out = []
    for r in (range(NPODS) if roles is None else roles):
        for g, (p, b) in enumerate(PORTS[r]):
            out.append((f"http://[{hs[r]}]:{p}", b, r))
    return out


KILL = ("for p in $(ps -eo pid,cmd | grep -E 'sglang.launch_server|sglang::|sglang_router|router.py|bench_serving|mn_bench' "
        "| grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null; done; sleep 2; "
        "rm -f ~/.cache/torch_extensions/py312_cu128/fused_fa3/lock; echo killed")


def run_arm(arm):
    layout, router = ARMS[arm]
    log = f"{W}/mn/{arm}"
    print(f"=== {arm}: layout={layout} router={router}", flush=True)
    C.run_all(KILL)
    env = f"KERNEL={os.environ.get('KERNEL', 'lpt')}"
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(NPODS) as ex:
        list(ex.map(lambda i: C.run(i, f"mkdir -p {log}; {env} bash {W}/tasks/pd_colocation/mn_node.sh {layout[i]} {log}"),
                    range(NPODS)))
    hs = hosts()
    read_ports(log)
    all_w = urls(hs)
    print(f"  ports: {PORTS}", flush=True)
    check = " ".join(f"'{u}/health'" for u, _, _ in all_w)
    t0 = time.time()
    while True:
        read_ports(log)  # a supervisor may have moved a worker to a new port
        all_w = urls(hs)
        check = " ".join(f"'{u}/health'" for u, _, _ in all_w)
        rc, out = C.run(0, f"n=0; for u in {check}; do [ \"$(curl -sg -o /dev/null -w '%{{http_code}}' --max-time 3 $u)\" = 200 ] && n=$((n+1)); done; echo UP=$n")
        up = int([l for l in out.splitlines() if l.startswith("UP=")][-1][3:])
        if up == len(all_w) == 2 * NPODS:
            break
        if time.time() - t0 > 1500:
            print(f"  only {up}/{len(all_w)} workers healthy after 25 min; aborting arm", flush=True)
            C.run_all(KILL)
            return
        time.sleep(20)
    print(f"  {len(all_w)} workers healthy after {time.time() - t0:.0f} s", flush=True)
    global RPORT
    rc, out = C.run(0, "python3 -c \"import random, socket\nfor _ in range(200):\n    p = random.randrange(20000, 29000)\n    try:\n        s = socket.socket(socket.AF_INET6); s.bind(('::', p)); s.close(); print('RPORT', p); break\n    except OSError:\n        pass\""
                       )
    RPORT = int([l for l in out.splitlines() if l.startswith("RPORT")][-1].split()[1])
    print(f"  router port {RPORT}", flush=True)
    kind, pol = router.split(":")
    if kind == "ours":
        ws = " ".join(u for u, _, _ in all_w)
        rcmd = (f"{W}/venv/bin/python {W}/tasks/pd_colocation/router.py --workers {ws} --port {RPORT} "
                f"--policy {pol} --log {log}/router_decisions.jsonl {os.environ.get('ROUTER_ARGS', '')}")
    elif pol == "cache_aware":
        ws = " ".join(u for u, _, _ in all_w)
        rcmd = f"{W}/venv/bin/python -m sglang_router.launch_router --worker-urls {ws} --policy cache_aware --host :: --port {RPORT}"
    else:
        pre = [i for i, m in enumerate(layout) if m == "prefill"]
        dec = [i for i, m in enumerate(layout) if m == "decode"]
        p_args = " ".join(f"--prefill {u} {b}" for u, b, _ in urls(hs, pre))
        d_args = " ".join(f"--decode {u}" for u, _, _ in urls(hs, dec))
        rcmd = f"{W}/venv/bin/python -m sglang_router.launch_router --pd-disaggregation {p_args} {d_args} --host :: --port {RPORT}"
    # setsid -f with every fd redirected: a backgrounded "cd && cmd" list keeps a subshell whose
    # stdout is the webshell pipe, which held each launch call open for its 100 s timeout.
    C.run(0, f"cd {W} && PYTHONPATH={W}/python setsid -f {rcmd} > {log}/router.log 2>&1 < /dev/null; sleep 8; "
             f"curl -s -o /dev/null -w 'router %{{http_code}}\\n' http://127.0.0.1:{RPORT}/health")
    knobs = " ".join(f"{k}={os.environ[k]}" for k in
                     ("SESSIONS", "TURNS", "SESSION_RATE", "SLOWDOWN", "CHAT_RATE", "CHAT_N", "CHAT_IN",
                      "CHAT_OUT", "MIN_CTX", "MAX_CTX") if k in os.environ)
    C.run(0, f"setsid -f env {knobs} bash {W}/tasks/pd_colocation/mn_bench.sh {log} {RPORT} > {log}/mn_bench.log 2>&1 < /dev/null; echo started")
    t0 = time.time()
    while time.time() - t0 < 5400:
        time.sleep(60)
        rc, out = C.run(0, f"grep -c MNBENCHDONE {log}/mn_bench.log")
        if out.strip().splitlines()[-1].strip() == "1":
            break
    rc, out = C.run(0, f"{W}/venv/bin/python {W}/tasks/pd_colocation/mn_summary.py {log}")
    print("  " + "\n  ".join(out.strip().splitlines()[-3:]), flush=True)
    C.run_all(KILL)


if __name__ == "__main__":
    for arm in sys.argv[1:]:
        run_arm(arm)
    print("MNRUNDONE", flush=True)
