#!/bin/bash
# Serial on the single GPU: (1) first-token bisection with the rope_theta fix, (2) the q8 A2
# serving gate into a NEW directory so the broken-RoPE run is preserved as evidence.
set -u
W=/dev/shm/w
cd $W
env PYTHONPATH=$W/validated_upstream/python HF_HOME=$W/hf $W/venv/bin/python -u first_token_bisect.py $W/first_token_bisect_ropefix_20261012
echo "BISECT_PHASE_DONE"
bash $W/a2_tail_q8_ropefix_20261012.sh
echo "ROPEFIX_CHAIN_DONE status=$?"
