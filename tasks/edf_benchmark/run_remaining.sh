#!/bin/bash
# Run remaining benchmarks: LPT_CG and DMC
# Uses 200 prompts per run, 900s timeout per run
set -e

MODEL="Qwen/Qwen3-8B"
PORT=30000
RESULTS_DIR="tasks/edf_benchmark/results"
RATES="1 2 4 6 8"

kill_server() {
    for pid in $(pgrep -f "sglang" 2>/dev/null); do kill -9 $pid 2>/dev/null; done
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

    # Skip if already exists
    if [ -f "$outfile" ]; then
        echo "  SKIP: $outfile already exists"
        return 0
    fi

    echo "  Running $method/$workload/rate=$rate ($num prompts)..."
    curl -s http://localhost:$PORT/flush_cache > /dev/null 2>&1 || true
    sleep 2

    timeout 900 python3 -m sglang.bench_serving \
        --backend sglang --port $PORT \
        --dataset-name $dataset $extra_args \
        --num-prompts $num \
        --request-rate $rate \
        --output-file "$outfile" \
        --disable-tqdm 2>&1 | grep -E "throughput|TTFT|ITL|Successful|Duration|request rate|Error" || true

    if [ -f "$outfile" ]; then
        echo "  OK: $outfile"
    else
        echo "  FAILED: no output"
    fi
}

# ============ LPT_CG ============
echo "=== LPT_CG ==="

for workload_info in "sharegpt:sharegpt:200:--dataset-path workloads/ShareGPT_V3_unfiltered_cleaned_split.json" "code_feedback:code-feedback:200:" "swe:swe-gym:100:"; do
    IFS=: read -r wl_name dataset num extra <<< "$workload_info"

    # Determine SLO
    case $wl_name in
        sharegpt) ttft=2000; itl=100;;
        code_feedback) ttft=3000; itl=150;;
        swe) ttft=5000; itl=150;;
    esac

    echo "--- LPT_CG / $wl_name (TTFT=${ttft}ms ITL=${itl}ms) ---"
    kill_server

    SGLANG_FLASHINFER_USE_PERSISTENT=1 \
    SGLANG_EDF_SCHEDULING=1 \
    SGLANG_TTFT_SLO_MS=$ttft \
    SGLANG_ITL_SLO_MS=$itl \
    SGLANG_EDF_LATENCY_MODEL=tasks/perf_model/lpt_model.pt \
    python -m sglang.launch_server \
        --model $MODEL --enable-mixed-chunk --enable-piecewise-cuda-graph \
        --chunked-prefill-size 8192 --mem-fraction-static 0.85 \
        --port $PORT --host 0.0.0.0 \
        --enable-priority-scheduling --schedule-low-priority-values-first \
        --watchdog-timeout 600 \
        > /tmp/bench_server.log 2>&1 &

    if ! wait_server; then continue; fi

    for rate in $RATES; do
        # Check if server is still healthy before each run
        if ! curl -s http://localhost:$PORT/health 2>/dev/null; then
            echo "  Server unhealthy, restarting..."
            kill_server
            SGLANG_FLASHINFER_USE_PERSISTENT=1 \
            SGLANG_EDF_SCHEDULING=1 \
            SGLANG_TTFT_SLO_MS=$ttft \
            SGLANG_ITL_SLO_MS=$itl \
            SGLANG_EDF_LATENCY_MODEL=tasks/perf_model/lpt_model.pt \
            python -m sglang.launch_server \
                --model $MODEL --enable-mixed-chunk --enable-piecewise-cuda-graph \
                --chunked-prefill-size 8192 --mem-fraction-static 0.85 \
                --port $PORT --host 0.0.0.0 \
                --enable-priority-scheduling --schedule-low-priority-values-first \
                --watchdog-timeout 600 \
                > /tmp/bench_server.log 2>&1 &
            if ! wait_server; then break; fi
        fi
        run_bench "LPT_CG" "$wl_name" "$rate" "$dataset" "$num" "$extra"
    done
done

# ============ DMC ============
echo ""
echo "=== DMC ==="

for workload_info in "sharegpt:sharegpt:200:--dataset-path workloads/ShareGPT_V3_unfiltered_cleaned_split.json" "code_feedback:code-feedback:200:" "swe:swe-gym:100:"; do
    IFS=: read -r wl_name dataset num extra <<< "$workload_info"

    case $wl_name in
        sharegpt) ttft=2000; itl=100;;
        code_feedback) ttft=3000; itl=150;;
        swe) ttft=5000; itl=150;;
    esac

    echo "--- DMC / $wl_name (TTFT=${ttft}ms ITL=${itl}ms) ---"
    kill_server

    SGLANG_EDF_SCHEDULING=1 \
    SGLANG_TTFT_SLO_MS=$ttft \
    SGLANG_ITL_SLO_MS=$itl \
    SGLANG_EDF_LATENCY_MODEL=tasks/perf_model/dmc_model.pt \
    python -m sglang.launch_server \
        --model $MODEL --enable-mixed-chunk --enable-piecewise-cuda-graph \
        --chunked-prefill-size 8192 --mem-fraction-static 0.85 \
        --port $PORT --host 0.0.0.0 \
        --enable-priority-scheduling --schedule-low-priority-values-first \
        --watchdog-timeout 600 \
        > /tmp/bench_server.log 2>&1 &

    if ! wait_server; then continue; fi

    for rate in $RATES; do
        if ! curl -s http://localhost:$PORT/health 2>/dev/null; then
            echo "  Server unhealthy, restarting..."
            kill_server
            SGLANG_EDF_SCHEDULING=1 \
            SGLANG_TTFT_SLO_MS=$ttft \
            SGLANG_ITL_SLO_MS=$itl \
            SGLANG_EDF_LATENCY_MODEL=tasks/perf_model/dmc_model.pt \
            python -m sglang.launch_server \
                --model $MODEL --enable-mixed-chunk --enable-piecewise-cuda-graph \
                --chunked-prefill-size 8192 --mem-fraction-static 0.85 \
                --port $PORT --host 0.0.0.0 \
                --enable-priority-scheduling --schedule-low-priority-values-first \
                --watchdog-timeout 600 \
                > /tmp/bench_server.log 2>&1 &
            if ! wait_server; then break; fi
        fi
        run_bench "DMC" "$wl_name" "$rate" "$dataset" "$num" "$extra"
    done
done

kill_server
echo ""
echo "=== All benchmarks complete ==="
echo "Results:"
ls -la $RESULTS_DIR/*.json 2>/dev/null | wc -l
echo "JSON files in $RESULTS_DIR"
