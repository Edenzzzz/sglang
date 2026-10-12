#!/bin/bash
# PASS only if the null case (unfixed tree, no barrier) crashed AND every fixed arm is clean AND the gate passed.
W=/dev/shm/w; R=$W/graphs_racefix_20261013
null_tb=$(grep -c Traceback $R/null_unfixed_MTPMIX_LPT/server_MTPMIX_LPT.log 2>/dev/null || echo 0)
fixed_ok=1
for d in fixed_MTPMIX_LPT_1 fixed_MTPMIX_LPT_2 fixed_MTPMIXFUSED_1 fixed_MTPMIXFUSED_2 fixed_sync_MTPMIX_LPT; do
  m=${d#fixed_}; m=${m#sync_}; m=${m%_[12]}
  [ -f $R/$d/${m}_validation.json ] && [ "$(grep -c Traceback $R/$d/server_$m.log)" = 0 ] || { fixed_ok=0; echo "not clean: $d"; }
done
gate_ok=0; grep -q "RACEFIX_GATE_DONE status=0" $W/graphs_racefix_20261013.log && gate_ok=1
if [ "$null_tb" -gt 0 ] && [ $fixed_ok = 1 ] && [ $gate_ok = 1 ]; then v=PASS
elif [ "$null_tb" = 0 ]; then v="INCONCLUSIVE null case did not crash"
else v="FAIL fixed_ok=$fixed_ok gate_ok=$gate_ok"; fi
echo "$v (null tracebacks=$null_tb)" | tee $W/racefix_verdict
