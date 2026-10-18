#!/bin/bash
# Single-node bar A/B on the re-forked H200 (d6a80befdd4e4da5) against the baselines of
# tasks/realistic_workload_analysis.md ## Baselines: stock chunked prefill + mixed chunk, default kernel, no
# speculation (coloc_sgl) and stock PD multiplexing (coloc_sgl_pdmux), vs ours without MTP (FA3/FA2 split on mixed-
# chunk steps, EDF + chunk preemption, uncached deadlines, deadline chunk) and ours with MTP (4-mode step) vs stock
# chunked prefill + MTP (coloc_sgl_mtps). Same
# workload as mn_h200_fourmode_20261018.sh: agent C=8 frozen 32-64K TraceLab, 2 rounds, + ShareGPT 8 req/s.
# Claude, 2026-10-18.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=d6a80befdd4e4da5 MN_CP=i18n-bd NPODS=1 GPUS_PER_POD=1 SM_FA3=112
export CHAT_SHAREGPT=1 MIN_CTX=${MIN_CTX:-32768} MAX_CTX=${MAX_CTX:-65536} TURNS=${TURNS:-2} AGENT_CONC=${AGENT_CONC:-8}
export SESSIONS=${SESSIONS:-24} CHAT_RATE=${CHAT_RATE:-8} CHAT_N=${CHAT_N:-2400}
T=${TAG:-_nospec}
for arm in ${ARMS:-stock stock_mtp pdmux ours ours_mtp}; do
  case $arm in
    stock) KERNEL=lpt PREEMPT=0 RUN_TAG=_h200_c8$T python3 -u mn_run.py coloc_sgl ;;
    stock_mtp) KERNEL=lpt PREEMPT=0 RUN_TAG=_h200_c8$T python3 -u mn_run.py coloc_sgl_mtps ;;
    pdmux) KERNEL=lpt PREEMPT=0 PDMUX_DIV=36 MEM_FRAC=${PDMUX_MEM_FRAC:-0.78} RUN_TAG=_h200_c8$T python3 -u mn_run.py coloc_sgl_pdmux ;;
    ours) KERNEL=mtpsplit PREEMPT=1 CHUNK=${CHUNK:-2048} EDF_UNCACHED=1 DEADLINE_CHUNK=${DEADLINE_CHUNK:-1024:2048:6000} \
          RUN_TAG=_h200_split_ch${CHUNK:-2048}_c8$T python3 -u mn_run.py coloc_cache_edf ;;
    ours_mtp) KERNEL=mtpsplit PREEMPT=1 CHUNK=2048 EDF_UNCACHED=1 DEADLINE_CHUNK=1024:2048:6000 \
          RUN_TAG=_h200_4mode_unc_ch2048_dl1024-2048-6000_c8$T python3 -u mn_run.py coloc_cache_edf_mtp ;;
  esac
done
echo H200_NOSPEC_DONE
