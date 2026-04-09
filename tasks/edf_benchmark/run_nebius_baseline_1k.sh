#!/bin/bash
# Baseline (no EDF) sweep on Nebius-SWE-Trajectories: PP / DMC / LPT_CG.
# Heavier prefill (~9K input median) than code-feedback so chunk control matters.
# Output is fixed to 1024 tokens to mimic Qwen thinking-mode decode length.
set -u

MODEL="Qwen/Qwen3-8B"
PORT=30000
RESULTS_DIR="tasks/edf_benchmark/results_baseline_nebius_1k"
mkdir -p "$RESULTS_DIR"
RATES="${RATES:-1 2 4 8 12 inf}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
OUT_LEN="${OUT_LEN:-1024}"
KEEPALIVE="bash /mlx_devbox/users/wenxuan.tan/playground/VibeCUDA/scripts/gpu_keepalive.sh"

kill_all() {
    pkill -9 -f "launch_server.*$PORT" 2>/dev/null || true
    pkill -9 -f "bench_serving.*--port $PORT" 2>/dev/null || true
    sleep 5
}

wait_server() {
    local logfile=$1
    for i in $(seq 1 150); do
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
    local method=$1 wl=$2 rate=$3
    local outfile="$RESULTS_DIR/${method}_${wl}_rate${rate}.json"

    if [ -f "$outfile" ]; then
        echo "  SKIP: $outfile exists"
        return 0
    fi

    if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "  Server unhealthy, aborting"
        return 1
    fi

    echo "  Run $method/$wl/rate=$rate (n=$NUM_PROMPTS, out=$OUT_LEN)"
    curl -s http://localhost:$PORT/flush_cache > /dev/null 2>&1 || true
    sleep 2

    $KEEPALIVE stop > /dev/null 2>&1 || true
    timeout 3600 python3 -m sglang.bench_serving \
        --backend sglang --port $PORT \
        --dataset-name nebius-swe-traj \
        --sharegpt-output-len $OUT_LEN \
        --num-prompts $NUM_PROMPTS \
        --request-rate $rate \
        --output-file "$outfile" \
        --output-details --disable-tqdm 2>&1 | grep -E "Output token throughput|Median TTFT|P99 TTFT|Median ITL|P99 ITL|duration"
    $KEEPALIVE start > /dev/null 2>&1 || true

    [ -f "$outfile" ] && echo "  OK" || echo "  FAILED"
}

start_server() {
    local method=$1 logfile=$2
    local extra_flags=""
    local prefix_env="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"

    case "$method" in
        PP)
            extra_flags="--enable-piecewise-cuda-graph"
            ;;
        DMC)
            extra_flags="--enable-mixed-chunk --enable-piecewise-cuda-graph"
            ;;
        LPT_CG)
            extra_flags="--enable-mixed-chunk --enable-piecewise-cuda-graph"
            prefix_env="$prefix_env SGLANG_FLASHINFER_USE_PERSISTENT=1"
            ;;
    esac

    env $prefix_env \
        nohup python -m sglang.launch_server --model $MODEL $extra_flags \
        --chunked-prefill-size 8192 --mem-fraction-static 0.85 \
        --port $PORT --host 0.0.0.0 \
        --watchdog-timeout 1200 > "$logfile" 2>&1 &
    disown
}

for METHOD in PP DMC LPT_CG; do
    echo ""
    echo "===== $METHOD / nebius_swe ====="
    kill_all

    LOGFILE="/tmp/baseline_nebius_${METHOD}.log"
    rm -f "$LOGFILE"
    start_server "$METHOD" "$LOGFILE"

    if ! wait_server "$LOGFILE"; then
        kill_all
        continue
    fi

    for rate in $RATES; do
        if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "  Server died, restarting"
            kill_all
            start_server "$METHOD" "$LOGFILE"
            wait_server "$LOGFILE" || break
        fi
        run_bench "$METHOD" "nebius_swe" "$rate"
    done
done

kill_all
echo ""
echo "=== Done ==="
ls "$RESULTS_DIR"/*.json 2>/dev/null | wc -l
