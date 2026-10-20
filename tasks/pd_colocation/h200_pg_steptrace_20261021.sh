#!/bin/bash
# Why page 128 loses end-to-end for ours although FA3 is 9-17% faster per layer at page 128 than at page 1
# (fa3_page_bench_20261021.py): ours + MTP (rows + SLO admission) at page 1 and page 128 with STEP_TRACE=1 (GPU time
# per forward), standard mix, chat 8 req/s, one repeat each (H200, 2026-10-21). Claude.
set -u
cd "$(dirname "$0")"
for PG in 1 128; do
  STEP_TRACE=1 DECODE_ROWS=48:20:24000:80:0.9 WORKER_EXTRA="--page-size $PG --max-running-requests 128 --cuda-graph-max-bs-decode 48" \
    TAG=_pt_rows48adm_pg${PG}_trace ARMS=ours_mtp bash mn_h200_nospec_20261018.sh
done
echo H200_PG_TRACE_DONE
