"""Multi-node arm orchestrator (runs locally; drives the H20 pods through px_multi).

  python mn_run.py <arm> [<arm> ...]          arms: see ARMS
  env: SID (job run), NPODS (8), plus mn_bench.sh knobs (SESSIONS, SESSION_RATE, CHAT_RATE, ...)

Every arm: kill leftovers -> start workers in the arm's layout -> wait until every worker's
/health is 200 -> start the arm's router on pod 0 -> run mn_bench.sh -> print the per-stream
summary -> tear down. Logs and JSON stay on pod 0 under /dev/shm/w/mn/<arm>/.
"""
import fcntl, hashlib, json, os, sys, tempfile, time
from contextlib import contextmanager
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from px_multi import Cluster

SID = os.environ.get("SID", "c2fb3965f36cd15f")
NPODS = int(os.environ.get("NPODS", "8"))
W = "/dev/shm/w"
C = Cluster(SID, os.environ.get("MN_CP", "cn-seed"), nodes=[f"worker-{i}" for i in range(NPODS)])  # MN_CP: control plane
RPORT = 31000

@contextmanager
def cluster_lock(control_plane, sid, directory=None):
    key = hashlib.sha256(f"{control_plane}/{sid}".encode()).hexdigest()
    path = os.path.join(directory or tempfile.gettempdir(), f"mn_cluster_{key}.lock")
    with open(path, "a+") as handle:
        print(f"Waiting for cluster lock: {control_plane}/{sid}", flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()} pods={NPODS} argv={sys.argv!r}\n")
            handle.flush()
            print(f"Acquired cluster lock: {control_plane}/{sid}", flush=True)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


ARMS = {
    # name: (layout per pod, router)
    "coloc_rr":    (["coloc"] * NPODS, "ours:rr"),
    "coloc_cache": (["coloc"] * NPODS, "ours:cache"),
    "coloc_rand":  (["coloc"] * NPODS, "ours:cache_rand"),
    "coloc_slo":   (["coloc"] * NPODS, "ours:cache_slo"),
    # cache_slo_rand: the SLO check, but among workers whose hit length ties within --rand-thresh
    # blocks it picks RANDOMLY rather than by lowest load. This is the point-4 algorithm the goal
    # proposes; it only differs from cache_slo when several workers really hold the same prefix.
    "coloc_slo_rand": (["coloc"] * NPODS, "ours:cache_slo_rand"),
    "coloc_sgl":   (["coloc"] * NPODS, "sgl:cache_aware"),
    "disagg_2p2d": (["prefill"] * (NPODS // 2) + ["decode"] * (NPODS // 2), "sgl:pd"),
    "disagg_1p3d": (["prefill"] * (NPODS // 4) + ["decode"] * (NPODS - NPODS // 4), "sgl:pd"),
}
# disagg_p<P>: the first P pods prefill, the other NPODS - P decode -- any P:D split, for ratio sweeps (2026-10-21).
for _p in range(1, NPODS):
    ARMS[f"disagg_p{_p}"] = (["prefill"] * _p + ["decode"] * (NPODS - _p), "sgl:pd")
# cond_p<P>: conditional disaggregation (Dynamo's policy, router.py cond_disagg) on colocated workers -- the first P
# pods' workers take remote prefills of long uncached prompts, the rest decode and prefill short appends locally.
for _p in range(1, NPODS):
    ARMS[f"cond_p{_p}"] = (["coloc"] * NPODS, "ours:cond_disagg")


def hosts():
    rc, out = C.run(0, "echo HOSTS=$ARNOLD_WORKER_HOSTS")
    line = [l for l in out.splitlines() if l.startswith("HOSTS=")][-1][6:]
    return [h.rsplit(":", 1)[0].strip("[]") for h in line.split(",")]


PORTS = {}  # role -> [(http_port, bootstrap_port) for gpu 0, 1], read from each pod


def read_ports(log):
    for r, (rc, out) in enumerate(C.run_all(f"cat {log}/ports_r${{ARNOLD_ID}}_g*.txt 2>/dev/null")):
        # One line per GPU worker ("gpu port boot"); any GPU index (4-GPU pods run g0..g3).
        rows = [l.split() for l in out.splitlines() if len(l.split()) == 3 and l.split()[0].isdigit()]
        PORTS[r] = [(int(p), int(b)) for _, p, b in sorted(rows, key=lambda x: int(x[0]))]


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
    # without the 4-mode step (SPEC=2); _pdmux runs PD multiplexing on every colocated worker (mn_node.sh PDMUX,
    # decode SM share PDMUX_DIV, EDF deadlines on uncached tokens EDF_UNCACHED).
    base, hc, edf = arm, False, False
    spec = False
    pdmux = False
    while base.endswith(("_hc", "_edf", "_mtp", "_mtps", "_pdmux")):
        if base.endswith("_pdmux"):
            pdmux, base = True, base[:-6]
        elif base.endswith("_hc"):
            hc, base = True, base[:-3]
        elif base.endswith("_mtps"):
            spec, base = 2, base[:-5]
        elif base.endswith("_mtp"):
            spec, base = 1, base[:-4]
        else:
            edf, base = True, base[:-4]
    layout, router = ARMS[base]
    log = f"{W}/mn/{arm}{os.environ.get('RUN_TAG', '')}"
    print(f"=== {arm}{os.environ.get('RUN_TAG', '')}: layout={layout} router={router} l3_store={hc} worker_edf={edf} mtp={spec} pdmux={pdmux}", flush=True)
    C.run_all(KILL)
    # bench_serving APPENDS to its output file and mn_summary reads the last line, so a stale
    # directory would let a failed arm report the previous run's numbers.
    C.run_all(f"rm -rf {log}")
    env = (f"ONE_NIC={os.environ.get('ONE_NIC', '0')} KERNEL={os.environ.get('KERNEL', 'lpt')} EDF={int(edf)} SPEC={int(spec)} "
           f"TTFT_SLO_MS={os.environ.get('TTFT_SLO_MS', '2000')} PREEMPT={os.environ.get('PREEMPT', '0')} CHUNK={os.environ.get('CHUNK', '')} "
           f"SGL_PY={os.environ.get('SGL_PY', '')} HC_PREFETCH={os.environ.get('HC_PREFETCH', 'best_effort')} HC_WRITE={os.environ.get('HC_WRITE', '')} HC_HOST_MODE={os.environ.get('HC_HOST_MODE', '')} WORKER_EXTRA=\"{os.environ.get('WORKER_EXTRA', '')}\" "
           f"PDMUX={int(pdmux)} PDMUX_DIV={os.environ.get('PDMUX_DIV', '30')} EDF_UNCACHED={os.environ.get('EDF_UNCACHED', '0')} "
           f"MEM_FRAC={os.environ.get('MEM_FRAC', '0.85')} SM_FA3={os.environ.get('SM_FA3', '66')} PDMUX_MANUAL='{os.environ.get('PDMUX_MANUAL', '')}' PDMUX_SHARE={os.environ.get('PDMUX_SHARE', '0')} TP={os.environ.get('TP', '1')} GPUS_PER_POD={os.environ.get('GPUS_PER_POD', '2')} SPEC_STEPS={os.environ.get('SPEC_STEPS', '3')} SPEC_TOKENS={os.environ.get('SPEC_TOKENS', '4')} PDMUX_SHARE_DEADLINE={os.environ.get('PDMUX_SHARE_DEADLINE', '0')} PDMUX_ITL_BUDGET={os.environ.get('PDMUX_ITL_BUDGET', '0')} DEADLINE_CHUNK={os.environ.get('DEADLINE_CHUNK', '')} STEP_BUDGET={os.environ.get('STEP_BUDGET', '')} STEP_TRACE={os.environ.get('STEP_TRACE', '0')} TTFT_SCALE_MS={os.environ.get('TTFT_SCALE_MS', '')} CKPT_EVERY={os.environ.get('CKPT_EVERY', '')} LONG_DEADLINE={os.environ.get('LONG_DEADLINE', '')} CTX_LEN={os.environ.get('CTX_LEN', '131072')} YARN_FACTOR={os.environ.get('YARN_FACTOR', '4.0')} SPLIT_OVERLAP={os.environ.get('SPLIT_OVERLAP', '0')} SPLIT_ROWS_FA3={os.environ.get('SPLIT_ROWS_FA3', '0')} ATTN_BACKEND={os.environ.get('ATTN_BACKEND', 'flashinfer')} VERIFY_FA3={os.environ.get('VERIFY_FA3', '0')} VERIFY_FA3_CHECK={os.environ.get('VERIFY_FA3_CHECK', '0')} DECODE_FA3={os.environ.get('DECODE_FA3', '0')} SPLIT_MARGIN_AUTO={os.environ.get('SPLIT_MARGIN_AUTO', '0')} SPLIT_POLICY={os.environ.get('SPLIT_POLICY', '0')} SPLIT_NEVER={os.environ.get('SPLIT_NEVER', '0')} SPLIT_FASTPATH={os.environ.get('SPLIT_FASTPATH', '0')} DRAFT_FA3={os.environ.get('DRAFT_FA3', '0')} SKIP_BASE_PLAN={os.environ.get('SKIP_BASE_PLAN', '0')} SPLIT_VERIFY={os.environ.get('SPLIT_VERIFY', '0')} FI_PERSISTENT={os.environ.get('FI_PERSISTENT', '1')} KV_COPY={os.environ.get('KV_COPY', 'off')} ITL_SLO={os.environ.get('ITL_SLO', '')} DECODE_ROWS={os.environ.get('DECODE_ROWS', '')} DRAFT_DIR={os.environ.get('DRAFT_DIR', 'qwen7b_eagle3_sgl')} SIM_ACC={os.environ.get('SIM_ACC', '')} WORKER_UP_S={os.environ.get('WORKER_UP_S', '360')} MIXED_CHUNK={os.environ.get('MIXED_CHUNK', '1')} LOAD_PUB={os.environ.get('LOAD_PUB', '0')} MODEL_DIR={os.environ.get('MODEL_DIR', 'qwen7b')} KV_COPY_NWORKERS={NPODS * int(os.environ.get('GPUS_PER_POD', '2'))}")
    if hc:
        rc, out = C.run(0, f"mkdir -p {log}; ip=$(hostname -I | tr ' ' '\\n' | grep -E '^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$' | head -1); "
                           "p=$(python3 -c \"import random, socket\nfor _ in range(200):\n    p = random.randrange(20000, 29000)\n    try:\n        [socket.socket().bind(('0.0.0.0', q)) for q in (p, p + 1)]; print(p); break\n    except OSError:\n        pass\"); "
                           "CUDART=$(" + W + "/venv/bin/python -c 'import nvidia.cuda_runtime as m, os; print(os.path.dirname(m.__file__))'); "
                           "LD_LIBRARY_PATH=$CUDART/lib:$CUDART "
                           f"setsid -f {W}/venv/bin/mooncake_master --rpc_port $p --metrics_port $((p + 1)) "
                           f"--eviction_high_watermark_ratio={os.environ.get('MC_EVICT_HIGH', '0.95')} > {log}/mooncake_master.log 2>&1 < /dev/null; sleep 3; "
                           "echo HC_MASTER=$ip:$p")
        master = [l for l in out.splitlines() if l.startswith("HC_MASTER=")][-1][10:]
        # The workers only report "Failed to setup Mooncake store", so check the master itself.
        rc, mlog = C.run(0, f"pgrep -f 'mooncake_master --rpc_port {master.rsplit(':', 1)[1]}' >/dev/null && echo MASTER_UP; "
                            f"tail -2 {log}/mooncake_master.log")
        if "MASTER_UP" not in mlog:
            print(f"  mooncake master did not start: {mlog.strip()[-300:]}; aborting arm", flush=True)
            return
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
        if up == len(all_w) == (int(os.environ.get("GPUS_PER_POD", "2")) // int(os.environ.get("TP", "1"))) * NPODS:
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
    # health-check ejection are off for every sglang_router arm. --prometheus-port: the exporter's default port was taken
    # on the re-forked H200 host and the router panicked at start.
    lenient = ("--disable-circuit-breaker --health-failure-threshold 1000000 "
               "--health-check-timeout-secs 60 --queue-size 100000")
    if os.environ.get("DISAGG_HEALTH", "0") == "1":
        # Fault-tolerance arms (FAULT_AT): the stock router must be able to eject a dead worker. Health checks every
        # 10 s, ejected after 3 consecutive failures (a killed worker refuses at once; a loaded live one answers
        # within the 30 s timeout); the circuit breaker stays off for the reason above.
        lenient = ("--disable-circuit-breaker --health-failure-threshold 3 --health-check-interval-secs 10 "
                   "--health-check-timeout-secs 30 --queue-size 100000")
    if kind == "ours":
        ws = " ".join(u for u, _, _ in all_w)
        # LOAD_PUB=1: colocated workers publish their LoadStat on the (otherwise unused) bootstrap port.
        lep = (" --load-endpoints " + " ".join(f"tcp://[{hs[r]}]:{b}" for _, b, r in all_w)
               if os.environ.get("LOAD_PUB", "0") == "1" else "")
        cond = ""
        if pol == "cond_disagg":
            n_pre = int(base.split("_p", 1)[1]) * len(PORTS[0])
            cond = f" --cond-disagg-prefill-workers {n_pre} --cond-disagg-threshold {os.environ.get('COND_THRESHOLD', '4096')} --cond-disagg-min-frac {os.environ.get('COND_FRAC', '0')}"
        rcmd = (f"{W}/venv/bin/python {W}/tasks/pd_colocation/router.py --workers {ws} --port {RPORT} "
                f"--policy {pol} --log {log}/router_decisions.jsonl {os.environ.get('ROUTER_ARGS', '')}{lep}{cond}")
    elif pol == "cache_aware":
        ws = " ".join(u for u, _, _ in all_w)
        rcmd = f"{W}/venv/bin/python -m sglang_router.launch_router --worker-urls {ws} --policy cache_aware {lenient} --host 0.0.0.0 --port {RPORT} --prometheus-port {RPORT + 1}"
    else:
        pre = [i for i, m in enumerate(layout) if m == "prefill"]
        dec = [i for i, m in enumerate(layout) if m == "decode"]
        p_args = " ".join(f"--prefill {u} {b}" for u, b, _ in urls(hs, pre))
        d_urls = urls(hs, dec)
        d_args = " ".join(f"--decode {u}" for u, _, _ in d_urls)
        # power_of_two needs >= 2 decode workers (sglang_router rejects it otherwise); with one decode worker -- e.g.
        # TP4 Llama-70B at P = 3 of 4 pods -- every policy sends every request to it, so round_robin is equivalent.
        dpol = "power_of_two" if len(d_urls) >= 2 else "round_robin"
        rcmd = f"{W}/venv/bin/python -m sglang_router.launch_router --pd-disaggregation {p_args} {d_args} --prefill-policy cache_aware --decode-policy {dpol} {lenient} --host 0.0.0.0 --port {RPORT} --prometheus-port {RPORT + 1}"
    # setsid -f with every fd redirected: a backgrounded "cd && cmd" list keeps a subshell whose
    # stdout is the webshell pipe, which held each launch call open for its 100 s timeout.
    C.run(0, f"ulimit -n $(ulimit -Hn); cd {W} && PYTHONPATH={W}/python setsid -f {rcmd} > {log}/router.log 2>&1 < /dev/null; sleep 8; "
             f"curl -s -o /dev/null -w 'router %{{http_code}}\\n' http://127.0.0.1:{RPORT}/health")
    knobs = " ".join(f"{k}={os.environ[k]}" for k in
                     ("SESSIONS", "TURNS", "SESSION_RATE", "SLOWDOWN", "CHAT_RATE", "CHAT_N", "CHAT_IN",
                      "CHAT_OUT", "MIN_CTX", "MAX_CTX", "HOT", "HOT_GROUPS", "HOT_PREFIX", "AGENT_CONC",
                      "AGENT_SEED", "CHAT_SHAREGPT", "WORKLOAD", "FO_TASKS", "FO_PREFIX", "FO_ROUNDS", "FO_APPEND",
                      "FO_OUT", "FO_GAP", "FO_ZIPF", "FO_ROTATE", "FO_DURATION", "FO_TEXT", "CHAT_SPIKE", "FO_SPIKE", "MODEL_DIR", "CHAT_TEMPLATE",
                      "AGENT_MODE", "WINDOW_START", "WINDOW_STARTS", "WINDOW_S", "ALPHA", "WINDOW_ROUNDS",
                      "TRACE_DATASET_PATH", "TEXT_SOURCE", "MATH_RATE", "MATH_N", "MATH_SET", "MATH_SEED",
                      "BENCH_REPLAY_HORIZON_S") if k in os.environ)
    # The launch call itself can fail on the webshell, and nothing else would notice: the arm then waits out its
    # 90 min timeout with no bench (and FAULT_AT fires anyway). Check that the bench log exists; relaunch if not.
    for attempt in range(4):
        C.run(0, f"setsid -f env {knobs} bash {W}/tasks/pd_colocation/mn_bench.sh {log} {RPORT} > {log}/mn_bench.log 2>&1 < /dev/null; echo started")
        rc, out = C.run(0, f"sleep 5; [ -f {log}/mn_bench.log ] && echo BENCH_UP")
        if "BENCH_UP" in out:
            break
        print(f"  bench launch attempt {attempt + 1} did not start: {out.strip()[-120:]}", flush=True)
    t0 = time.time()
    fault_at = float(os.environ.get("FAULT_AT", "0"))
    if fault_at > 0:
        # Fault injection: FAULT_AT s into the bench, kill every worker on pod FAULT_POD (supervisors first, so none
        # is relaunched). Pod 0 hosts the router, bench and pool master, so the faulted pod must be another one.
        fpod = int(os.environ.get("FAULT_POD", "1"))
        assert fpod != 0

        def fault():
            time.sleep(fault_at)
            rc, out = C.run(fpod, f"date +%s.%N > {log}/fault_t.txt; {KILL}")
            print(f"  FAULT: killed pod {fpod} workers at +{fault_at:.0f} s: {out.strip()[-40:]}", flush=True)
        import threading
        threading.Thread(target=fault, daemon=True).start()
    while time.time() - t0 < 5400:
        time.sleep(60)
        rc, out = C.run(0, f"grep -c MNBENCHDONE {log}/mn_bench.log")
        if out.strip().splitlines()[-1].strip() == "1":
            break
    rc, out = C.run(0, f"{W}/venv/bin/python {W}/tasks/pd_colocation/mn_summary.py {log}")
    print("  " + "\n  ".join(out.strip().splitlines()[-3:]), flush=True)
    if os.environ.get("SCRAPE_METRICS"):
        # Each worker's Prometheus page (needs WORKER_EXTRA=--enable-metrics) before teardown.
        for r, (u, _, _) in enumerate(all_w):
            C.run(0, f"curl -sg --max-time 20 {u}/metrics > {log}/metrics_w{r}.txt; wc -l < {log}/metrics_w{r}.txt")
    C.run_all(KILL)


def calibrate():
    """Fit router.py's PerfModel on one H20 colocated worker (pod 0, GPU 0) with
    calibrate_h20.py; prints the ROUTER_ARGS line for the cache_slo arms."""
    model = os.environ.get("MODEL_DIR", "qwen7b")
    log = f"{W}/mn/calibrate" + ("" if model == "qwen7b" else f"_{model}")
    print("=== calibrate", flush=True)
    C.run_all(KILL)
    # The perf model must describe the workers the router will route to: same kernel / MTP / EDF /
    # preemption config as the arm (env), not the stock FA2 worker.
    cenv = (f"KERNEL={os.environ.get('KERNEL', 'lpt')} EDF={os.environ.get('EDF', '0')} SPEC={os.environ.get('SPEC', '0')} "
            f"PREEMPT={os.environ.get('PREEMPT', '0')} CHUNK={os.environ.get('CHUNK', '')} MODEL_DIR={model} "
            f"TP={os.environ.get('TP', '1')} GPUS_PER_POD={os.environ.get('GPUS_PER_POD', '2')} "
            f"DRAFT_DIR={os.environ.get('DRAFT_DIR', 'qwen7b_eagle3_sgl')} SIM_ACC={os.environ.get('SIM_ACC', '')} "
            f"MEM_FRAC={os.environ.get('MEM_FRAC', '0.85')} SM_FA3={os.environ.get('SM_FA3', '66')} "
            f"SPLIT_OVERLAP={os.environ.get('SPLIT_OVERLAP', '0')} SPLIT_ROWS_FA3={os.environ.get('SPLIT_ROWS_FA3', '0')} "
            f"VERIFY_FA3={os.environ.get('VERIFY_FA3', '0')} DECODE_FA3={os.environ.get('DECODE_FA3', '0')} "
            f"SPLIT_MARGIN_AUTO={os.environ.get('SPLIT_MARGIN_AUTO', '0')} SPLIT_POLICY={os.environ.get('SPLIT_POLICY', '0')} SPLIT_NEVER={os.environ.get('SPLIT_NEVER', '0')} SPLIT_FASTPATH={os.environ.get('SPLIT_FASTPATH', '0')} DRAFT_FA3={os.environ.get('DRAFT_FA3', '0')} SKIP_BASE_PLAN={os.environ.get('SKIP_BASE_PLAN', '0')} CHAT_TEMPLATE={os.environ.get('CHAT_TEMPLATE', '0')} "
            f"ATTN_BACKEND={os.environ.get('ATTN_BACKEND', 'flashinfer')} EDF_UNCACHED={os.environ.get('EDF_UNCACHED', '0')}")
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
    grid = f"{W}/mn/decode_grid.json" if model == "qwen7b" else f"{W}/mn/decode_grid_{model}.json"
    C.run(0, f"CAL_GRID={grid} setsid -f {W}/venv/bin/python {W}/tasks/pd_colocation/calibrate_h20.py {url} > {log}/calibrate.log 2>&1 < /dev/null; echo started")
    while time.time() - t0 < 2400:
        time.sleep(30)
        rc, out = C.run(0, f"cat {log}/calibrate.log")
        if "ROUTER_ARGS=" in out or "Traceback" in out:
            break
    print("  " + "\n  ".join(l for l in out.splitlines() if l.startswith(("prefill", "decode", "ROUTER_ARGS", "Trace"))), flush=True)
    C.run_all(KILL)


if __name__ == "__main__":
    with cluster_lock(C.cp, SID):
        for arm in sys.argv[1:]:
            calibrate() if arm == "calibrate" else run_arm(arm)
    print("MNRUNDONE", flush=True)
