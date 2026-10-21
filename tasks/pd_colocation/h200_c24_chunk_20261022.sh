#!/bin/bash
# C=24 at the right chunk (H200, 2026-10-22, after $1). The agent-concurrency sweep held the deadline chunk at
# 1024:2048 across C=8/16/24, and the C=16 retest showed that is the WRONG setting under load: chunk 1024:4096 took
# ours no MTP from 34.4% to 48.4% agent attainment there (stock 50.5-55.9%), with chat at parity. The "monotone
# degradation" reading of the sweep was therefore retracted, and the C=24 row (ours 17.9% vs stock 30.7%) is still
# mis-tuned. This retests it at 1024:4096 and 1024:8192, no MTP and with MTP, so any claim about saturation rests on
# ours at its own best setting. Every arm on sglang's FA3 backend. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
export CHAT_TEMPLATE=1 CHAT_RATE=8 CHAT_N=2400 AGENT_CONC=24 SESSIONS=72
WX="--max-running-requests 128 --cuda-graph-max-bs-decode 48"
for LARGE in 4096 8192; do
  ATTN_BACKEND=fa3 KERNEL=lpt TAG=_fa3_c24_dc${LARGE} ARMS=ours_ovl CHUNK=$LARGE \
    DEADLINE_CHUNK=1024:${LARGE}:6000 bash mn_h200_nospec_20261018.sh
  ATTN_BACKEND=fa3 KERNEL=lpt DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="$WX" TAG=_fa3_c24_dc${LARGE}_rows48adm \
    ARMS=ours_mtp_dc CHUNK=$LARGE DEADLINE_CHUNK=1024:${LARGE}:6000 bash mn_h200_nospec_20261018.sh
done
python3 - <<'PY'
import sys; sys.path.insert(0, "/tmp")
from px import run
L="/dev/shm/w/mn"
rows = []
for LARGE in (4096, 8192):
    rows += [(f"ours no MTP C24 dc{LARGE}", f"coloc_cache_edf_h200_split_ovl_ch{LARGE}_dl1024-{LARGE}-6000_c8_fa3_c24_dc{LARGE}"),
             (f"ours + MTP C24 dc{LARGE}", f"coloc_cache_edf_mtp_h200_4mode_unc_ch{LARGE}_dl1024-{LARGE}-6000_c8_fa3_c24_dc{LARGE}_rows48adm")]
for name, d in rows:
    print(f"## {name}")
    print(run(f"test -d {L}/{d} && /dev/shm/w/venv/bin/python /dev/shm/w/perf/bench_per_tenant.py {L}/{d} | tail -2 | cut -c1-165 || echo MISSING {d}")[1], flush=True)
PY
echo H200_C24_CHUNK_DONE
