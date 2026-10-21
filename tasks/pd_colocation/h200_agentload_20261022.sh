#!/bin/bash
# The load regime the single-node bar actually needs (H200, 2026-10-22, after $1). Why no arm has cleared it: chat
# throughput equals the offered rate in EVERY arm (1640-1647 tok/s, nothing dropped) and stock holds 76-80% chat
# attainment from chat 8 to 16 req/s, so there is no headroom for a >=30% attainment gain and the only movable
# throughput is AGENT throughput. Agent program concurrency is what saturates this GPU (32-64K contexts), and the
# multi-node runs showed ours pulling ahead precisely as load doubled (+56% -> +164%). So sweep agent concurrency with
# chat fixed, every arm on sglang's FA3 backend so the schedule is the only variable:
#   AGENT_CONC 16 and 24 (vs 8 in every run so far), 48 agent requests, chat 8 req/s
#   arms: stock mixed chunk, stock prefill-prioritized, ours no MTP, ours + MTP (4-mode, rows + SLO admission)
# One repeat per point; per-tenant attainment at the end. Agent load is CONCURRENCY, never sessions/s. Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
export CHAT_TEMPLATE=1 CHAT_RATE=8 CHAT_N=2400
WX="--max-running-requests 128 --cuda-graph-max-bs-decode 48"
for C in 16 24; do
  export AGENT_CONC=$C SESSIONS=$((3 * C))
  ATTN_BACKEND=fa3 KERNEL=lpt TAG=_fa3_c${C} ARMS="stock stock_pp" bash mn_h200_nospec_20261018.sh
  ATTN_BACKEND=fa3 KERNEL=lpt TAG=_fa3_c${C} ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 \
    bash mn_h200_nospec_20261018.sh
  ATTN_BACKEND=fa3 KERNEL=lpt DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="$WX" TAG=_fa3_c${C}_rows48adm \
    ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
done
python3 - <<'PY'
import sys; sys.path.insert(0, "/tmp")
from px import run
L="/dev/shm/w/mn"
pats = [("stock mixed chunk", "coloc_sgl_h200_c8_fa3_c%d"),
        ("stock prefill-prio", "coloc_sgl_h200_pp_c8_fa3_c%d"),
        ("ours no MTP", "coloc_cache_edf_h200_split_ovl_ch2048_dl1024-2048-6000_c8_fa3_c%d"),
        ("ours + MTP 4-mode", "coloc_cache_edf_mtp_h200_4mode_unc_ch2048_dl1024-2048-6000_c8_fa3_c%d_rows48adm")]
for C in (16, 24):
    print(f"########## agent concurrency {C}")
    for name, pat in pats:
        d = pat % C
        print(f"## {name}")
        print(run(f"test -d {L}/{d} && /dev/shm/w/venv/bin/python /dev/shm/w/perf/bench_per_tenant.py {L}/{d} | tail -2 | cut -c1-165 || echo MISSING {d}")[1], flush=True)
PY
echo H200_AGENTLOAD_DONE
