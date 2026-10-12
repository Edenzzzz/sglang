# Sourced by every queued serving script after its own exports: the race-fixed tree without the
# mixed-step barrier if graphs_racefix_20261013 passed, else the current tree with the barrier.
if grep -q '^PASS' /dev/shm/w/racefix_verdict 2>/dev/null; then
  TREE=up_racefix; unset SGLANG_DEBUG_STEP_SYNC
else
  TREE=up; export SGLANG_DEBUG_STEP_SYNC=mixed
fi
export PYTHONPATH=/dev/shm/w/$TREE/python
echo "TREE=$TREE barrier=${SGLANG_DEBUG_STEP_SYNC:-off} verdict=[$(cat /dev/shm/w/racefix_verdict 2>/dev/null)]"
