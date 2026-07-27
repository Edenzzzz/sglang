#!/bin/bash
# Sweep offered TraceLab session-arrival rate while preserving causal rounds.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Ten real sessions each replay consecutive rounds. New sessions arrive as a
# Poisson process at the requested sessions/s; a session's next round still
# waits for its previous response and its scaled recorded think/tool delay.
# "inf" reproduces the original headline setting (all sessions start together).
RATES="${RATES:-0.01 0.02 0.05 inf}"
CASES_SPEC=""
for rate in $RATES; do
    CASES_SPEC+="${CASES_SPEC:+ }1:1:${rate}"
done

METHODS_SPEC="${METHODS_SPEC:-PP_DECODE_CG DMC_CG LPT}" \
CASES_SPEC="$CASES_SPEC" \
NUM_PROMPTS="${NUM_PROMPTS:-400}" \
TRACELAB_NUM_GROUPS="${TRACELAB_NUM_GROUPS:-10}" \
TRACELAB_TIME_SCALE="${TRACELAB_TIME_SCALE:-0.01}" \
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/h800_qwen25_tracelab_claude_pareto_n400}" \
bash "$SCRIPT_DIR/run_tracelab_claude.sh"
