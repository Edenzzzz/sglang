#!/bin/bash
set -euo pipefail
cd /dev/shm/w
trap 'status=$?; printf "NATIVE_PDL_COMBINED_CHAIN_EXIT status=%s time=%s\n" "$status" "$(date -Is)"' EXIT
bash native_pdl_combined_validate_20261013.sh > native_pdl_combined_validate_20261013.log 2>&1
bash native_pdl_combined_serving_20261013.sh > native_pdl_combined_serving_20261013.log 2>&1
