#!/bin/bash
# After $1 (the running H20 driver) exits, start mn_provision_70b_20261024.sh on pods 0-3 in the background (downloads
# would otherwise share the pods' network with a measured arm) and poll until every pod reports PROVISION70_DONE.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-now}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
python3 - <<'PY'
import sys, time
sys.path.insert(0, "/opt/tiger/sglang/tasks/pd_colocation")
from px_multi import Cluster
c = Cluster("2f608c1b1a67cf09", "cn-seed", nodes=[f"worker-{i}" for i in range(4)])
for i in range(4):
    print(i, c.run(i, "cd /dev/shm/w/perf && setsid nohup bash /dev/shm/w/perf/mn_provision_70b_20261024.sh > /dev/shm/w/perf/prov70.log 2>&1 < /dev/null & echo started", timeout=90)[1].strip()[-60:], flush=True)
while True:
    time.sleep(300)
    outs = [c.run(i, "tail -2 /dev/shm/w/perf/prov70.log; ls /opt/tiger/w/llama70b/*.safetensors 2>/dev/null | wc -l", timeout=90)[1].strip() for i in range(4)]
    print(time.strftime("%H:%M"), " | ".join(o.replace("\n", " ")[-120:] for o in outs), flush=True)
    if all("PROVISION70_DONE" in o or "FAIL" in o or "BAD" in o for o in outs):
        break
print("PROVISION70_ALL_DONE", flush=True)
PY
