#!/bin/bash
# Resume the standalone set from MATH-500 (the native-call check is relaxed for standalone runs), one point per ONLY.
set -u
cd /dev/shm/w/ub
B=/opt/tiger/w/a6_standalone_h200_20261026
for name in math500_r16 gsm8k_r32 codefeedback_r16; do
  [ -d $B/$name ] && mv $B/$name $B/${name}_aborted_$(date +%H%M)
  ONLY=$name bash a6_standalone_h200_20261026.sh || { echo "POINT_FAILED $name"; exit 1; }
done
echo A6_STANDALONE_RESUME_DONE
