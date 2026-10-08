#!/bin/bash
# Compare LPT_CG recovery with DMC at the operating points where LPT_CG
# previously crossed the ITL SLO.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NUM_PROMPTS_OVERRIDE="${NUM_PROMPTS_OVERRIDE:-200}"

run_one() {
    local workload=$1 rates=$2
    RESULTS_DIR="$SCRIPT_DIR/results_recovery" \
    METHODS="DMC LPT_CG" \
    WORKLOAD_FILTER="$workload" \
    RATES="$rates" \
    NUM_PROMPTS_OVERRIDE="$NUM_PROMPTS_OVERRIDE" \
        bash "$SCRIPT_DIR/run_edf_sweep.sh"
}

# SWE rate=12 was the clearest 150 ms boundary loss (69% versus DMC's 100%).
# ShareGPT rate=24 checks that recovery preserves the existing LPT_CG win.
for workload_rates in "swe:12" "sharegpt:24"; do
    IFS=: read -r workload rates <<< "$workload_rates"
    run_one "$workload" "$rates"
done

python3 "$SCRIPT_DIR/analyze_results.py" "$SCRIPT_DIR/results_recovery"
