#!/bin/bash
# Run PP and DMC benchmarks with EDF scheduling
# LPT_CG hangs under load due to cooperative kernel issue
set -e

MODEL="Qwen/Qwen3-8B"
PORT=30000
RESULTS_DIR="tasks/edf_benchmark/results"
mkdir -p "$RESULTS_DIR"
RATES="1 2 4 6 8"

kill_all() {
    for pid in $(pgrep -f "sglang" 2>/dev/null); do kill -9 $pid 2>/dev/null; done
    for pid in $(pgrep -f "bench_serving" 2>/dev/null); do kill -9 $pid 2>/dev/null; done
    sleep 5
}

wait_server() {
    for i in $(seq 1 90); do
        if curl -s http://localhost:$PORT/health 2>/dev/null; then
            echo "  Server ready after $((i*2))s"
            return 0
        fi
        sleep 2
    done
    echo "  ERROR: Server failed!"
    return 1
}

run_bench() {
    local method=$1 workload=$2 rate=$3 dataset=$4 num=$5 extra_args=$6
    local outfile="$RESULTS_DIR/${method}_${workload}_rate${rate}.json"

    if [ -f "$outfile" ]; then
        echo "  SKIP: $outfile exists"
        return 0
    fi

    echo "  Run $method/$workload/rate=$rate..."
    curl -s http://localhost:$PORT/flush_cache > /dev/null 2>&1 || true
    sleep 2

    timeout 900 python3 -m sglang.bench_serving \
        --backend sglang --port $PORT \
        --dataset-name $dataset $extra_args \
        --num-prompts $num \
        --request-rate $rate \
        --output-file "$outfile" \
        --disable-tqdm 2>&1 || true

    [ -f "$outfile" ] && echo "  OK" || echo "  FAILED"
}

# ============ PP ============
echo "=== PP ==="
for wl_info in "sharegpt:sharegpt:200:--dataset-path workloads/ShareGPT_V3_unfiltered_cleaned_split.json:2000:100" "code_feedback:code-feedback:200::3000:150" "swe:swe-gym:100::5000:150"; do
    IFS=: read -r wl dataset num extra ttft itl <<< "$wl_info"
    echo "--- PP / $wl ---"
    kill_all

    SGLANG_EDF_SCHEDULING=1 SGLANG_TTFT_SLO_MS=$ttft SGLANG_ITL_SLO_MS=$itl \
    python -m sglang.launch_server \
        --model $MODEL --enable-piecewise-cuda-graph \
        --chunked-prefill-size 8192 --mem-fraction-static 0.85 \
        --port $PORT --host 0.0.0.0 \
        --enable-priority-scheduling --schedule-low-priority-values-first \
        --watchdog-timeout 300 > /tmp/bench_server.log 2>&1 &

    wait_server || continue

    for rate in $RATES; do
        run_bench "PP" "$wl" "$rate" "$dataset" "$num" "$extra"
    done
done

# ============ DMC ============
echo ""
echo "=== DMC ==="
for wl_info in "sharegpt:sharegpt:200:--dataset-path workloads/ShareGPT_V3_unfiltered_cleaned_split.json:2000:100" "code_feedback:code-feedback:200::3000:150" "swe:swe-gym:100::5000:150"; do
    IFS=: read -r wl dataset num extra ttft itl <<< "$wl_info"
    echo "--- DMC / $wl ---"
    kill_all

    SGLANG_EDF_SCHEDULING=1 SGLANG_TTFT_SLO_MS=$ttft SGLANG_ITL_SLO_MS=$itl \
    SGLANG_EDF_LATENCY_MODEL=tasks/perf_model/dmc_model.pt \
    python -m sglang.launch_server \
        --model $MODEL --enable-mixed-chunk --enable-piecewise-cuda-graph \
        --chunked-prefill-size 8192 --mem-fraction-static 0.85 \
        --port $PORT --host 0.0.0.0 \
        --enable-priority-scheduling --schedule-low-priority-values-first \
        --watchdog-timeout 300 > /tmp/bench_server.log 2>&1 &

    wait_server || continue

    for rate in $RATES; do
        if ! curl -s http://localhost:$PORT/health 2>/dev/null; then
            echo "  Server unhealthy, restarting..."
            kill_all
            SGLANG_EDF_SCHEDULING=1 SGLANG_TTFT_SLO_MS=$ttft SGLANG_ITL_SLO_MS=$itl \
            SGLANG_EDF_LATENCY_MODEL=tasks/perf_model/dmc_model.pt \
            python -m sglang.launch_server \
                --model $MODEL --enable-mixed-chunk --enable-piecewise-cuda-graph \
                --chunked-prefill-size 8192 --mem-fraction-static 0.85 \
                --port $PORT --host 0.0.0.0 \
                --enable-priority-scheduling --schedule-low-priority-values-first \
                --watchdog-timeout 300 > /tmp/bench_server.log 2>&1 &
            wait_server || break
        fi
        run_bench "DMC" "$wl" "$rate" "$dataset" "$num" "$extra"
    done
done

kill_all
echo ""
echo "=== Done ==="
ls $RESULTS_DIR/*.json 2>/dev/null | wc -l
echo "total JSON results"
