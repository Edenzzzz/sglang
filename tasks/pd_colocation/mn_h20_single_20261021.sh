#!/bin/bash
# Single-node table on ONE 4-GPU H20 pod at TP4 (job 2f608c1b1a67cf09; default GLM-4.5-Air with its own MTP layer, no
# YaRN): the arms of mn_h200_nospec_20261018.sh, same workload knobs. Claude, 2026-10-21.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
cd "$(dirname "$0")"
export SID=2f608c1b1a67cf09 MN_CP=cn-seed NPODS=1 GPUS_PER_POD=4 TP=4 SM_FA3=66 MODEL_DIR=${MODEL_DIR:-glm45air} DRAFT_DIR=${DRAFT_DIR:-nextn} YARN_FACTOR=${YARN_FACTOR:-0}
export CHAT_SHAREGPT=1 MIN_CTX=${MIN_CTX:-32768} MAX_CTX=${MAX_CTX:-65536} TURNS=${TURNS:-2} AGENT_CONC=${AGENT_CONC:-8}
export SESSIONS=${SESSIONS:-24} CHAT_RATE=${CHAT_RATE:-8} CHAT_N=${CHAT_N:-2400}
T=${TAG:-_nospec}
for arm in ${ARMS:-stock stock_mtp pdmux ours ours_ovl ours_mtp}; do
  case $arm in
    stock) KERNEL=lpt PREEMPT=0 RUN_TAG=_h20tp4_c8$T python3 -u mn_run.py coloc_sgl ;;
    stock_mtp) KERNEL=lpt PREEMPT=0 RUN_TAG=_h20tp4_c8$T python3 -u mn_run.py coloc_sgl_mtps ;;
    stock_pp) MIXED_CHUNK=0 KERNEL=lpt PREEMPT=0 RUN_TAG=_h20tp4_pp_c8$T python3 -u mn_run.py coloc_sgl ;;
    stock_mtp_pp) MIXED_CHUNK=0 KERNEL=lpt PREEMPT=0 RUN_TAG=_h20tp4_pp_c8$T python3 -u mn_run.py coloc_sgl_mtps ;;
    pdmux) KERNEL=lpt PREEMPT=0 PDMUX_DIV=36 MEM_FRAC=${PDMUX_MEM_FRAC:-0.78} RUN_TAG=_h20tp4_c8$T python3 -u mn_run.py coloc_sgl_pdmux ;;
    ours) KERNEL=mtpsplit PREEMPT=1 CHUNK=${CHUNK:-2048} EDF_UNCACHED=1 DEADLINE_CHUNK=${DEADLINE_CHUNK:-1024:2048:6000} \
          RUN_TAG=_h20tp4_split_ch${CHUNK:-2048}_c8$T python3 -u mn_run.py coloc_cache_edf ;;
    ours_ovl) SPLIT_OVERLAP=1 KERNEL=mtpsplit PREEMPT=1 CHUNK=${CHUNK:-2048} EDF_UNCACHED=1 DEADLINE_CHUNK=${DEADLINE_CHUNK:-1024:2048:6000} \
          RUN_TAG=_h20tp4_split_ovl_ch${CHUNK:-2048}_dl${DEADLINE_CHUNK//:/-}${TTFT_SCALE_MS:+_ts${TTFT_SCALE_MS}}_c8$T python3 -u mn_run.py coloc_cache_edf ;;
    ours_mtp_ovl) SPLIT_OVERLAP=1 KERNEL=mtpsplit PREEMPT=1 CHUNK=2048 EDF_UNCACHED=1 DEADLINE_CHUNK=1024:2048:6000 \
          RUN_TAG=_h20tp4_4mode_ovl_unc_ch2048_dl1024-2048-6000_c8$T python3 -u mn_run.py coloc_cache_edf_mtp ;;
    ours_mtp_dc) KERNEL=mtpsplit PREEMPT=1 CHUNK=${CHUNK:-2048} EDF_UNCACHED=1 DEADLINE_CHUNK=${DEADLINE_CHUNK:-1024:2048:6000} \
          RUN_TAG=_h20tp4_4mode_unc_ch${CHUNK:-2048}_dl${DEADLINE_CHUNK//:/-}_c8$T python3 -u mn_run.py coloc_cache_edf_mtp ;;
    ours_mtp) KERNEL=mtpsplit PREEMPT=1 CHUNK=2048 EDF_UNCACHED=1 DEADLINE_CHUNK=1024:2048:6000 \
          RUN_TAG=_h20tp4_4mode_unc_ch2048_dl1024-2048-6000_c8$T python3 -u mn_run.py coloc_cache_edf_mtp ;;
    # MTP ablation (2026-10-20): the full stack minus one component at a time.
    #   ours_mtp_unfused  4-mode step + EDF + deadline chunk, attention on the stock kernel (no FA3/FA2 split)
    #   ours_mtp_noedf    4-mode step + split, FCFS admission at chunk 2048 (no EDF, no preemption, no deadline chunk)
    #   ours_mtp_nodc     4-mode step + split + EDF + preemption at a fixed chunk 2048 (no deadline chunk)
    ours_mtp_unfused) KERNEL=lpt PREEMPT=1 CHUNK=2048 EDF_UNCACHED=1 DEADLINE_CHUNK=1024:2048:6000 \
          RUN_TAG=_h20tp4_4mode_unfused_unc_ch2048_dl1024-2048-6000_c8$T python3 -u mn_run.py coloc_cache_edf_mtp ;;
    ours_mtp_noedf) KERNEL=mtpsplit PREEMPT=0 CHUNK=2048 RUN_TAG=_h20tp4_4mode_noedf_ch2048_c8$T python3 -u mn_run.py coloc_cache_mtp ;;
    ours_mtp_nodc) KERNEL=mtpsplit PREEMPT=1 CHUNK=2048 EDF_UNCACHED=1 \
          RUN_TAG=_h20tp4_4mode_unc_ch2048_nodc_c8$T python3 -u mn_run.py coloc_cache_edf_mtp ;;
  esac
done
echo H200_NOSPEC_DONE
