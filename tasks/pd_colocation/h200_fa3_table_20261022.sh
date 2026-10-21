#!/bin/bash
# THE single-node table, redone on the FA3 footing (H200, 2026-10-22, after $1). Everything before this compared ours
# -- built inside sglang's flashinfer backend -- against stock on either backend; the backend-class control showed
# that class is half the throughput of FA3 on this workload (81.1 vs 161.2 agent tok/s, same schedule, no split), and
# that our scheduling lives in scheduler.py, not the backend. So every arm here runs sglang's FA3 backend and the only
# difference between them is the SCHEDULE:
#   stock mixed chunk                     (chunked prefill + mixed chunk, sglang default)
#   stock prefill-prioritized             (no mixed chunk)
#   stock prefill-prioritized + MTP       (EAGLE3, no 4-mode step)
#   ours no MTP      : EDF + chunk preemption + deadline chunk
#   ours + MTP       : the 4-mode step + decode-row cap + SLO admission
# Two repeats, chat template, real head, chat 8 req/s, agent program concurrency 8 over 24 sessions. Per-tenant
# attainment via bench_per_tenant.py. This is what the >=30% TTFT/ITL-attainment and >=10% throughput bar is judged on.
# Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
[ "${1:-}" = now ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
export CHAT_TEMPLATE=1
WX="--max-running-requests 128 --cuda-graph-max-bs-decode 48"
for REPN in 1 2; do
  ATTN_BACKEND=fa3 KERNEL=lpt TAG=_fa3tab_rep$REPN ARMS="stock stock_pp stock_mtp_pp" bash mn_h200_nospec_20261018.sh
  ATTN_BACKEND=fa3 KERNEL=lpt TAG=_fa3tab_rep$REPN ARMS=ours_ovl CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 \
    bash mn_h200_nospec_20261018.sh
  ATTN_BACKEND=fa3 KERNEL=lpt DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="$WX" TAG=_fa3tab_rows48adm_rep$REPN \
    ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
done
python3 - <<'PY'
import sys; sys.path.insert(0, "/tmp")
from px import run
L = "/dev/shm/w/mn"
arms = [("stock mixed chunk", "coloc_sgl_h200_c8_fa3tab_rep%d"),
        ("stock prefill-prio", "coloc_sgl_h200_pp_c8_fa3tab_rep%d"),
        ("stock prefill-prio + MTP", "coloc_sgl_mtps_h200_pp_c8_fa3tab_rep%d"),
        ("ours no MTP", "coloc_cache_edf_h200_split_ovl_ch2048_dl1024-2048-6000_c8_fa3tab_rep%d"),
        ("ours + MTP (4-mode)", "coloc_cache_edf_mtp_h200_4mode_unc_ch2048_dl1024-2048-6000_c8_fa3tab_rows48adm_rep%d")]
for name, pat in arms:
    for rep in (1, 2):
        d = pat % rep
        out = run(f"test -d {L}/{d} && /dev/shm/w/venv/bin/python /dev/shm/w/perf/bench_per_tenant.py {L}/{d} | tail -2 || echo MISSING {d}")[1]
        print(f"## {name} rep{rep}\n{out}", flush=True)
PY
echo H200_FA3_TABLE_DONE
