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
    # cache_slo_rand: the SLO veto, but among workers whose hit length ties within --rand-thresh
    # blocks it picks RANDOMLY rather than by lowest load. This is the point-4 algorithm the goal
    # proposes; it only differs from cache_slo when several workers really hold the same prefix.
    "coloc_slo_rand": (["coloc"] * NPODS, "ours:cache_slo_rand"),
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


KILL = f"bash {W}/tasks/pd_colocation/mn_kill.sh"


def run_arm(arm):
    # Suffixes: _hc adds a cluster-wide mooncake L3 prefix store (mn_node.sh HC_MASTER);
    # _edf runs deadline-ordered admission on every worker (mn_node.sh EDF); _mtp runs EAGLE3 MTP
    # on every worker, colocated ones with the 4-mode step (mn_node.sh SPEC=1); _mtps is stock MTP
    # without the 4-mode step (SPEC=2).
    base, hc, edf = arm, False, False
    spec = False
    while base.endswith(("_hc", "_edf", "_mtp", "_mtps")):
        if base.endswith("_hc"):
            hc, base = True, base[:-3]
        elif base.endswith("_mtps"):
            spec, base = 2, base[:-5]
        elif base.endswith("_mtp"):
            spec, base = 1, base[:-4]
        else:
            edf, base = True, base[:-4]
    layout, router = ARMS[base]
    log = f"{W}/mn/{arm}{os.environ.get('RUN_TAG', '')}"
    print(f"=== {arm}{os.environ.get('RUN_TAG', '')}: layout={layout} router={router} l3_store={hc} worker_edf={edf} mtp={spec}", flush=True)
    C.run_all(KILL)
    # bench_serving APPENDS to its output file and mn_summary reads the last line, so a stale
    # directory would let a failed arm report the previous run's numbers.
    C.run_all(f"rm -rf {log}")
    env = (f"ONE_NIC={os.environ.get('ONE_NIC', '0')} KERNEL={os.environ.get('KERNEL', 'lpt')} EDF={int(edf)} SPEC={int(spec)} "
           f"TTFT_SLO_MS={os.environ.get('TTFT_SLO_MS', '2000')} PREEMPT={os.environ.get('PREEMPT', '0')} CHUNK={os.environ.get('CHUNK', '')} "
           f"SGL_PY={os.environ.get('SGL_PY', '')}")
    if hc:
        rc, out = C.run(0, f"mkdir -p {log}; ip=$(hostname -I | tr ' ' '\\n' | grep -E '^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$' | head -1); "
                           "p=$(python3 -c \"import random, socket\nfor _ in range(200):\n    p = random.randrange(20000, 29000)\n    try:\n        [socket.socket().bind(('0.0.0.0', q)) for q in (p, p + 1)]; print(p); break\n    except OSError:\n        pass\"); "
                           "CUDART=$(" + W + "/venv/bin/python -c 'import nvidia.cuda_runtime as m, os; print(os.path.dirname(m.__file__))'); "
                           "LD_LIBRARY_PATH=$CUDART/lib:$CUDART "
                           f"setsid -f /home/tiger/.local/lib/python3.12/site-packages/mooncake/mooncake_master --rpc_port $p --metrics_port $((p + 1)) "
                           f"--eviction_high_watermark_ratio=0.95 > {log}/mooncake_master.log 2>&1 < /dev/null; sleep 3; "
                           "echo HC_MASTER=$ip:$p")
        master = [l for l in out.splitlines() if l.startswith("HC_MASTER=")][-1][10:]
        print(f"  mooncake master {master}", flush=True)
        env += f" HC_MASTER={master}"
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
        if time.time() - t0 > 2400:
            print(f"  only {up}/{len(all_w)} workers healthy after 40 min; aborting arm", flush=True)
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
    # The stock router's circuit breaker opens on a prefill worker that is merely slow under a
    # saturating load and then rejects requests outright ("No available prefill"), which dropped
    # ~60% of disagg_2p2d's requests. A baseline must serve every request, so breaker and
    # health-check ejection are off for every sglang_router arm.
    lenient = ("--disable-circuit-breaker --health-failure-threshold 1000000 "
               "--health-check-timeout-secs 60 --queue-size 100000")
    if kind == "ours":
        ws = " ".join(u for u, _, _ in all_w)
        rcmd = (f"{W}/venv/bin/python {W}/tasks/pd_colocation/router.py --workers {ws} --port {RPORT} "
                f"--policy {pol} --log {log}/router_decisions.jsonl {os.environ.get('ROUTER_ARGS', '')}")
    elif pol == "cache_aware":
        ws = " ".join(u for u, _, _ in all_w)
        rcmd = f"{W}/venv/bin/python -m sglang_router.launch_router --worker-urls {ws} --policy cache_aware {lenient} --host 0.0.0.0 --port {RPORT}"
    else:
        pre = [i for i, m in enumerate(layout) if m == "prefill"]
        dec = [i for i, m in enumerate(layout) if m == "decode"]
        p_args = " ".join(f"--prefill {u} {b}" for u, b, _ in urls(hs, pre))
        d_args = " ".join(f"--decode {u}" for u, _, _ in urls(hs, dec))
        rcmd = f"{W}/venv/bin/python -m sglang_router.launch_router --pd-disaggregation {p_args} {d_args} --prefill-policy cache_aware --decode-policy power_of_two {lenient} --host 0.0.0.0 --port {RPORT}"
    # setsid -f with every fd redirected: a backgrounded "cd && cmd" list keeps a subshell whose
    # stdout is the webshell pipe, which held each launch call open for its 100 s timeout.
    C.run(0, f"ulimit -n $(ulimit -Hn); cd {W} && PYTHONPATH={W}/python setsid -f {rcmd} > {log}/router.log 2>&1 < /dev/null; sleep 8; "
             f"curl -s -o /dev/null -w 'router %{{http_code}}\\n' http://127.0.0.1:{RPORT}/health")
    knobs = " ".join(f"{k}={os.environ[k]}" for k in
                     ("SESSIONS", "TURNS", "SESSION_RATE", "SLOWDOWN", "CHAT_RATE", "CHAT_N", "CHAT_IN",
                      "CHAT_OUT", "MIN_CTX", "MAX_CTX", "HOT", "HOT_GROUPS", "HOT_PREFIX", "AGENT_CONC",
                      "AGENT_SEED", "CHAT_SHAREGPT") if k in os.environ)
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


def calibrate():
    """Fit router.py's PerfModel on one H20 colocated worker (pod 0, GPU 0) with
    calibrate_h20.py; prints the ROUTER_ARGS line for the cache_slo arms."""
    log = f"{W}/mn/calibrate"
    print("=== calibrate", flush=True)
    C.run_all(KILL)
    # The perf model must describe the workers the router will route to: same kernel / MTP / EDF /
    # preemption config as the arm (env), not the stock FA2 worker.
    cenv = (f"KERNEL={os.environ.get('KERNEL', 'lpt')} EDF={os.environ.get('EDF', '0')} SPEC={os.environ.get('SPEC', '0')} "
            f"PREEMPT={os.environ.get('PREEMPT', '0')} CHUNK={os.environ.get('CHUNK', '')}")
    C.run(0, f"rm -rf {log}; mkdir -p {log}; {cenv} bash {W}/tasks/pd_colocation/mn_node.sh coloc {log}")
    hs = hosts()
    t0 = time.time()
    while time.time() - t0 < 900:
        time.sleep(20)
        rc, out = C.run(0, f"cat {log}/ports_r0_g0.txt 2>/dev/null")
        rows = [l.split() for l in out.splitlines() if len(l.split()) == 3]
        if not rows:
            continue
        url = f"http://[{hs[0]}]:{rows[-1][1]}"
        rc, out = C.run(0, f"curl -sg -o /dev/null -w '%{{http_code}}' --max-time 3 {url}/health")
        if out.strip().endswith("200"):
            break
    C.run(0, f"setsid -f {W}/venv/bin/python {W}/tasks/pd_colocation/calibrate_h20.py {url} > {log}/calibrate.log 2>&1 < /dev/null; echo started")
    while time.time() - t0 < 2400:
        time.sleep(30)
        rc, out = C.run(0, f"cat {log}/calibrate.log")
        if "ROUTER_ARGS=" in out or "Traceback" in out:
            break
    print("  " + "\n  ".join(l for l in out.splitlines() if l.startswith(("prefill", "decode", "ROUTER_ARGS", "Trace"))), flush=True)
    C.run_all(KILL)


if __name__ == "__main__":
    for arm in sys.argv[1:]:
        calibrate() if arm == "calibrate" else run_arm(arm)
    print("MNRUNDONE", flush=True)
