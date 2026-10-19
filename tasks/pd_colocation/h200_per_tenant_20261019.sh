#!/bin/bash
# Per-tenant rerun of the single-node standard mix on H200 dadf94f0e1da7238 (2026-10-19): the four arms of the
# 2026-10-18 tables (stock mixed chunk no spec, ours no MTP FA2-overlapped, stock MTP, ours 4-mode MTP best setting),
# same harness and load (agent C=8 frozen 32-64K TraceLab x 2 rounds + ShareGPT 8 req/s), then per-tenant scoring
# on the pod and slim per-request records (ttfts / itls / output_lens / errors) kept for the local copy. Claude.
set -u
cd "$(dirname "$0")"
for REPN in ${REPS:-1 2}; do
  TAG=_pt_rep$REPN ARMS="stock ours_ovl stock_mtp ours_mtp" CHUNK=2048 DEADLINE_CHUNK=1024:2048:6000 \
    bash mn_h200_nospec_20261018.sh
done
echo H200_PER_TENANT_DONE
