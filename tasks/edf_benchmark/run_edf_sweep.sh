#!/bin/bash
# EDF benchmark sweep: PP / DMC / LPT_CG × ShareGPT/Code-Feedback/SWE-Gym × multiple rates
# All runs with --enable-priority-scheduling + EDF env vars + SLO-aware chunk control
set -u

MODEL="Qwen/Qwen3-8B"
PORT=30000
RESULTS_DIR="tasks/edf_benchmark/results"
mkdir -p "$RESULTS_DIR"
RATES="${RATES:-2 4 8 12 16 24 inf}"

LPT_MODEL="/mlx_devbox/users/wenxuan.tan/playground/sglang/tasks/perf_model/lpt_model.pt"
DMC_MODEL="/mlx_devbox/users/wenxuan.tan/playground/sglang/tasks/perf_model/dmc_model.pt"

kill_all() {
    pkill -9 -f "launch_server.*$PORT" 2>/dev/null || true
    pkill -9 -f "bench_serving.*--port $PORT" 2>/dev/null || true
    sleep 5
}

wait_server() {
    local logfile=$1
    for i in $(seq 1 120); do
        if grep -q "fired up" "$logfile" 2>/dev/null; then
            echo "  Server ready after $((i*2))s"
            return 0
        fi
        sleep 2
    done
    echo "  ERROR: Server failed!"
    tail -20 "$logfile"
    return 1
}

run_bench() {
    local method=$1 wl=$2 rate=$3 dataset=$4 num=$5 ttft=$6 itl=$7 extra=$8
    local outfile="$RESULTS_DIR/${method}_${wl}_rate${rate}.json"

    if [ -f "$outfile" ]; then
        echo "  SKIP: $outfile exists"
        return 0
    fi

    if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "  Server unhealthy, aborting"
        return 1
    fi

    echo "  Run $method/$wl/rate=$rate (TTFT=$ttft, ITL=$itl)"
    curl -s http://localhost:$PORT/flush_cache > /dev/null 2>&1 || true
    sleep 2

    timeout 900 python3 -m sglang.bench_serving \
        --backend sglang --port $PORT \
        --dataset-name $dataset $extra \
        --num-prompts $num \
        --request-rate $rate \
        --output-file "$outfile" \
        --output-details --disable-tqdm 2>&1 | tail -20

    [ -f "$outfile" ] && echo "  OK" || echo "  FAILED"
}

start_server() {
    local method=$1 ttft=$2 itl=$3 logfile=$4
    local extra_flags=""
    local model_env=""
    local prefix_env=""

    case "$method" in
        PP)
            extra_flags="--enable-piecewise-cuda-graph"
            model_env=""  # PP doesn't use chunk control (no mixed batches)
            ;;
        DMC)
            extra_flags="--enable-mixed-chunk --enable-piecewise-cuda-graph"
            model_env="SGLANG_EDF_LATENCY_MODEL=$DMC_MODEL"
            ;;
        LPT_CG)
            extra_flags="--enable-mixed-chunk --enable-piecewise-cuda-graph"
            model_env="SGLANG_EDF_LATENCY_MODEL=$LPT_MODEL"
            prefix_env="SGLANG_FLASHINFER_USE_PERSISTENT=1"
            ;;
    esac

    env $prefix_env \
        SGLANG_EDF_SCHEDULING=1 SGLANG_TTFT_SLO_MS=$ttft SGLANG_ITL_SLO_MS=$itl \
        $model_env \
        nohup python -m sglang.launch_server --model $MODEL $extra_flags \
        --attention-backend flashinfer \
        --chunked-prefill-size 8192 --mem-fraction-static 0.85 \
        --port $PORT --host 0.0.0.0 \
        --enable-priority-scheduling --schedule-low-priority-values-first \
        --watchdog-timeout 600 > "$logfile" 2>&1 &
    disown
}

# Workload definitions: name:dataset:num:ttft_ms:itl_ms:extra_args
WORKLOADS=(
    "sharegpt:sharegpt:200:2000:100:--dataset-path workloads/ShareGPT_V3_unfiltered_cleaned_split.json"
    "code_feedback:code-feedback:200:3000:150:"
    "swe:swe-gym:100:5000:150:"
)

for METHOD in PP DMC LPT_CG; do
    for wl_info in "${WORKLOADS[@]}"; do
        IFS=: read -r wl dataset num ttft itl extra <<< "$wl_info"
        echo ""
        echo "===== $METHOD / $wl ====="
        kill_all

        LOGFILE="/tmp/edf_${METHOD}_${wl}.log"
        rm -f "$LOGFILE"
        start_server "$METHOD" "$ttft" "$itl" "$LOGFILE"

        if ! wait_server "$LOGFILE"; then
            kill_all
            continue
        fi

        for rate in $RATES; do
            if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
                echo "  Server died, restarting"
                kill_all
                start_server "$METHOD" "$ttft" "$itl" "$LOGFILE"
                wait_server "$LOGFILE" || break
            fi
            run_bench "$METHOD" "$wl" "$rate" "$dataset" "$num" "$ttft" "$itl" "$extra"
        done
    done
done

kill_all
echo ""
echo "=== Done ==="
ls "$RESULTS_DIR"/*.json 2>/dev/null | wc -l
echo "total JSON results"
