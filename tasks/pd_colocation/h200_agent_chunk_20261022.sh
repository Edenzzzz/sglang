#!/bin/bash
# Can ours win BOTH attainment and throughput on single node? (H200, 2026-10-22, after $1.)
# Why ours loses throughput: at fixed agent program concurrency, agent throughput is set by how fast agent requests
# finish, and ours' agent TTFT is 3.0-3.3 s vs stock's 2.1 s -- EDF deprioritises agents (10 s deadline) in favour of
# chat (0.5 s), and the deadline chunk gives everything 1024/2048. Chat TTFT is 5x better for it (0.12 vs 0.58 s), so
# the trade is real but one-sided. Lever: keep the SMALL chunk for tight deadlines and give LOOSE-deadline requests a
# big one, so agent prefills finish in fewer, larger chunks while chat keeps its small ones.
#   DEADLINE_CHUNK="small:large:tok_s" -> 1024:4096 and 1024:8192, CHUNK set to the large value
# Arms, every one on sglang's FA3 backend (KERNEL=lpt), chat 8 req/s, one repeat:
#   ours no MTP  at 1024:4096 and 1024:8192
#   ours + MTP   at 1024:4096
# Baselines already measured on FA3: stock mixed chunk 142.0 tok/s / agent att 84.4% / chat att 79.9%;
# ours no MTP at 1024:2048 133.9 / 95.6% / 78.0%. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
export CHAT_TEMPLATE=1
WX="--max-running-requests 128 --cuda-graph-max-bs-decode 48"
for DC in 1024:4096:6000 1024:8192:6000; do
  LARGE=${DC#*:}; LARGE=${LARGE%%:*}
  ATTN_BACKEND=fa3 KERNEL=lpt TAG=_fa3be_dc${LARGE} ARMS=ours_ovl CHUNK=$LARGE DEADLINE_CHUNK=$DC \
    bash mn_h200_nospec_20261018.sh
done
ATTN_BACKEND=fa3 KERNEL=lpt DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="$WX" TAG=_fa3be_dc4096_rows48adm \
  ARMS=ours_mtp_dc CHUNK=4096 DEADLINE_CHUNK=1024:4096:6000 bash mn_h200_nospec_20261018.sh
python3 - <<'PY'
import sys; sys.path.insert(0, "/tmp")
from px import run
L="/dev/shm/w/mn"
for d in ("coloc_cache_edf_h200_split_ovl_ch4096_dl1024-4096-6000_c8_fa3be_dc4096",
          "coloc_cache_edf_h200_split_ovl_ch8192_dl1024-8192-6000_c8_fa3be_dc8192",
          "coloc_cache_edf_mtp_h200_4mode_unc_ch4096_dl1024-4096-6000_c8_fa3be_dc4096_rows48adm"):
    print(f"## {d}")
    print(run(f"test -d {L}/{d} && /dev/shm/w/venv/bin/python /dev/shm/w/perf/bench_per_tenant.py {L}/{d} | tail -2 | cut -c1-165 || echo MISSING")[1], flush=True)
PY
echo H200_AGENT_CHUNK_DONE
