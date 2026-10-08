#!/bin/bash
# EDF Scheduling Benchmark: PP vs LPT+CG vs DMC with EDF
# Tests on 3 workloads at multiple request rates
# Produces Pareto data: P99 ITL vs throughput

set -e

MODEL="Qwen/Qwen3-8B"
PORT=30000
RESULTS_DIR="tasks/edf_benchmark/results"
mkdir -p "$RESULTS_DIR"

# Workload definitions
# ShareGPT: real chat traces
# Code-Feedback: code generation from HuggingFace
# SWE: agent trajectories from HuggingFace
declare -A WL_DATASETS
WL_DATASETS[sharegpt]="sharegpt"
WL_DATASETS[code_feedback]="code-feedback"
WL_DATASETS[swe]="swe-gym"

declare -A WL_NUM_PROMPTS
WL_NUM_PROMPTS[sharegpt]="500"
WL_NUM_PROMPTS[code_feedback]="300"
WL_NUM_PROMPTS[swe]="200"

declare -A WL_DATASET_PATHS
WL_DATASET_PATHS[sharegpt]="workloads/ShareGPT_V3_unfiltered_cleaned_split.json"
# code_feedback and swe load from HuggingFace

# SLO targets per workload (in ms)
declare -A WL_TTFT_SLO
WL_TTFT_SLO[sharegpt]="2000"
WL_TTFT_SLO[code_feedback]="3000"
WL_TTFT_SLO[swe]="5000"

declare -A WL_ITL_SLO
WL_ITL_SLO[sharegpt]="100"
WL_ITL_SLO[code_feedback]="150"
WL_ITL_SLO[swe]="150"

# Request rates to sweep for Pareto curves
RATES="1 2 4 6 8"

# Method configs
# PP: Prefill-Prioritized (no --enable-mixed-chunk)
# LPT_CG: Persistent kernel + CUDA graphs
# DMC: Default mixed chunk (no persistent)
METHODS="PP LPT_CG DMC"

launch_server() {
    local method=$1
    local workload=$2
    local ttft_slo=${WL_TTFT_SLO[$workload]}
    local itl_slo=${WL_ITL_SLO[$workload]}

    echo "  Launching $method server..."

    local base_args="--model $MODEL --port $PORT --host 0.0.0.0 --mem-fraction-static 0.85 --chunked-prefill-size 8192 --watchdog-timeout 300"

    # EDF args (common for all methods when using EDF)
    local edf_env="SGLANG_EDF_SCHEDULING=1 SGLANG_TTFT_SLO_MS=$ttft_slo SGLANG_ITL_SLO_MS=$itl_slo"
    local edf_flags="--enable-priority-scheduling --schedule-low-priority-values-first"

    case $method in
        PP)
            # Prefill-prioritized: no --enable-mixed-chunk
            # For PP, EDF only affects admission order (no chunk control since no mixed batch)
            eval "$edf_env python -m sglang.launch_server $base_args --enable-piecewise-cuda-graph $edf_flags" > /tmp/bench_server.log 2>&1 &
            ;;
        LPT_CG)
            # LPT + CUDA graphs: persistent kernel + mixed chunk
            local lpt_model_path="tasks/perf_model/lpt_model.pt"
            eval "SGLANG_FLASHINFER_USE_PERSISTENT=1 $edf_env SGLANG_EDF_LATENCY_MODEL=$lpt_model_path python -m sglang.launch_server $base_args --enable-mixed-chunk --enable-piecewise-cuda-graph $edf_flags" > /tmp/bench_server.log 2>&1 &
            ;;
        DMC)
            # Default mixed chunk (no persistent)
            local dmc_model_path="tasks/perf_model/dmc_model.pt"
            eval "$edf_env SGLANG_EDF_LATENCY_MODEL=$dmc_model_path python -m sglang.launch_server $base_args --enable-mixed-chunk --enable-piecewise-cuda-graph $edf_flags" > /tmp/bench_server.log 2>&1 &
            ;;
    esac

    # Wait for server
    for i in $(seq 1 90); do
        if curl -s http://localhost:$PORT/health 2>/dev/null; then
            echo "  Server ready after $((i*2))s"
            return 0
        fi
        sleep 2
    done
    echo "  ERROR: Server failed to start!"
    tail -20 /tmp/bench_server.log
    return 1
}

kill_server() {
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "sglang.srt" 2>/dev/null || true
    sleep 3
}

run_benchmark() {
    local method=$1
    local workload=$2
    local rate=$3

    local dataset=${WL_DATASETS[$workload]}
    local num_prompts=${WL_NUM_PROMPTS[$workload]}
    local output_file="$RESULTS_DIR/${method}_${workload}_rate${rate}.json"

    echo "  Benchmarking $method / $workload / rate=$rate..."

    local bench_args="--backend sglang --port $PORT --dataset-name $dataset --num-prompts $num_prompts --request-rate $rate --output-file $output_file --disable-tqdm"

    # Add dataset path for ShareGPT
    if [ "$workload" = "sharegpt" ]; then
        bench_args="$bench_args --dataset-path ${WL_DATASET_PATHS[sharegpt]}"
    fi

    timeout 600 python3 -m sglang.bench_serving $bench_args 2>&1 | tee "$RESULTS_DIR/${method}_${workload}_rate${rate}.log" || echo "  TIMEOUT or error for $method/$workload/rate=$rate"

    echo "  Done: $output_file"
}

# Main benchmark loop
echo "=== EDF Scheduling Benchmark ==="
echo "Methods: $METHODS"
echo "Workloads: sharegpt code_feedback swe"
echo "Rates: $RATES"
echo "Results: $RESULTS_DIR"
echo ""

for method in $METHODS; do
    for workload in sharegpt code_feedback swe; do
        echo "=== $method / $workload ==="

        # Launch server for this method
        kill_server
        if ! launch_server "$method" "$workload"; then
            echo "  SKIP: Server failed to start for $method / $workload"
            continue
        fi

        # Run benchmarks at each rate
        for rate in $RATES; do
            # Flush cache between runs
            curl -s http://localhost:$PORT/flush_cache > /dev/null 2>&1 || true
            sleep 2

            run_benchmark "$method" "$workload" "$rate"
        done

        # Kill server after this workload
        kill_server
    done
done

echo ""
echo "=== Benchmark Complete ==="
echo "Results in $RESULTS_DIR"
