#!/bin/bash
# The last single-node test (H200, 2026-10-22, after $1). At agent concurrency 16 stock degrades to 50-56% agent /
# 57% chat attainment, so there IS headroom -- but ours at deadline chunk 1024:2048 is WORSE on everything there
# (agent 34.4%, chat 55.2%, agent TTFT 6.72 s vs stock's 3.16-3.55 s): under saturation EDF's deferral of loose-
# deadline agents compounds instead of paying off. The one configuration known to be much better was found after this
# sweep was queued: at C=8, deadline chunk 1024:4096 took ours + MTP from -12.5% to -3.8% throughput and put it ahead
# of stock + MTP on chat attainment, chat TTFT and agent TTFT. That interaction is untested at C=16.
# So: ours no MTP and ours + MTP at 1024:4096 and 1024:8192, agent concurrency 16, chat 8 req/s, every arm on FA3.
# If ours still loses at its best chunk under saturation, the single-node claim is closed as a latency/throughput
# trade at light load, and the paper's quantitative case rests on multi-node. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
export CHAT_TEMPLATE=1 CHAT_RATE=8 CHAT_N=2400 AGENT_CONC=16 SESSIONS=48
WX="--max-running-requests 128 --cuda-graph-max-bs-decode 48"
for LARGE in 4096 8192; do
  ATTN_BACKEND=fa3 KERNEL=lpt TAG=_fa3_c16_dc${LARGE} ARMS=ours_ovl CHUNK=$LARGE \
    DEADLINE_CHUNK=1024:${LARGE}:6000 bash mn_h200_nospec_20261018.sh
  ATTN_BACKEND=fa3 KERNEL=lpt DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="$WX" TAG=_fa3_c16_dc${LARGE}_rows48adm \
    ARMS=ours_mtp_dc CHUNK=$LARGE DEADLINE_CHUNK=1024:${LARGE}:6000 bash mn_h200_nospec_20261018.sh
done
python3 - <<'PY'
import sys; sys.path.insert(0, "/tmp")
from px import run
L="/dev/shm/w/mn"
for LARGE in (4096, 8192):
    for name, d in ((f"ours no MTP dc{LARGE}", f"coloc_cache_edf_h200_split_ovl_ch{LARGE}_dl1024-{LARGE}-6000_c8_fa3_c16_dc{LARGE}"),
                    (f"ours + MTP dc{LARGE}", f"coloc_cache_edf_mtp_h200_4mode_unc_ch{LARGE}_dl1024-{LARGE}-6000_c8_fa3_c16_dc{LARGE}_rows48adm")):
        print(f"## {name}")
        print(run(f"test -d {L}/{d} && /dev/shm/w/venv/bin/python /dev/shm/w/perf/bench_per_tenant.py {L}/{d} | tail -2 | cut -c1-165 || echo MISSING {d}")[1], flush=True)
PY
echo H200_C16_CHUNK_DONE
