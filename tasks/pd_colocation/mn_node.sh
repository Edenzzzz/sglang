#!/bin/bash
# Start this pod's two sglang workers (one per GPU) for one multi-node arm.
#   mn_node.sh <mode: coloc|prefill|decode> <logdir>
# Ports are unique per pod because pods on one host share its network namespace:
#   http 30000+100*role+{1,51}, disagg bootstrap 32000+100*role+{1,51} -- 50 apart, because
#   sglang reserves internal (IPC/NCCL) ports just above its http port, and 1 apart collided.
# The attention kernel is the same in every arm (persistent FA2/LPT unless KERNEL=split), so
# arms differ only in routing / placement -- the variable under test.
set -u
MODE=$1; LOG=$2
W=/dev/shm/w
ROLE=${ARNOLD_ID:-0}
mkdir -p $LOG
# SGL_PY: server-side sglang tree; the stock PD-disagg baseline runs clean upstream ($W/up_clean/python),
# because the serving overlay's spec-v2 changes trip disagg prefill's spec_info identity assert.
export PYTHONPATH=${SGL_PY:-$W/python}:$W/fi HF_HOME=$W/hf FI_ROOT=$W/fi TORCH_CUDA_ARCH_LIST=9.0a
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_FLASHINFER_USE_PERSISTENT=1
# mooncake (PD-disagg KV transfer) links libcudart.so.12; the pods' system toolkit is CUDA 13,
# but the torch cu128 wheels ship the 12.x runtime inside site-packages.
# 2026-10-16 venv: torch is cu130, mooncake still links libcudart.so.12 -> nvidia-cuda-runtime-cu12
# (provisioning) puts it under site-packages/nvidia/cuda_runtime/lib.
CUDART=$(dirname "$(find $(readlink -f $W/venv) -name 'libcudart.so.12*' 2>/dev/null | head -1)" 2>/dev/null)
export LD_LIBRARY_PATH=$CUDART:${LD_LIBRARY_PATH:-}
# Mooncake's handshake cannot parse an unbracketed IPv6 host:port ("failed to get IP address of
# peer server 2605:...:16743 ... use IPv4 address instead"), so every disagg request failed. The
# pods also carry IPv4 addresses, all mutually reachable; advertise the first one.
export SGLANG_HOST_IP=$(hostname -I | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' | head -1)
# KERNEL=mtpsplit: the single-GPU serving kernel of 2026-10-16 -- 4-mode MTP step on the two-launch
# split (FA3 prefill rows + stock FA2 verify rows), decode guard off; the same env as
# run_mixed_workloads' MTPMIX_SPLIT arm. SM_FA3 scaled to H20's 78 SMs (112/132 on H200).
if [ "${KERNEL:-lpt}" = mtpsplit ]; then
  export SGLANG_MIXED_ATTENTION_KERNEL=persistent SGLANG_TWO_EXEC=1 SGLANG_FUSED_FA3_SM_SPLIT=1
  export FLASHINFER_SM_PARTITION=1 UP_STAGES=2 SGLANG_FUSED_FA3_SRC=$W/fi/tests/fused_fa3_persistent_ext.cu
  export SGLANG_FUSED_FI_ROOT=$W/port07 SGLANG_FUSED_FA2_TILES=64,32,16 FUSED_FA2_FINE_TAIL=1
  export FUSED_CONST_PERS_PARAMS=1 FUSED_WORK_BUDGET_SPLIT=1 FLASHINFER_FUSED_BACKFILL=1
  export FLASHINFER_FUSED_COLAUNCH=1 FUSED_FA2_KV_CHUNK=auto SGLANG_FUSED_SM_FA3=66
  export SGLANG_SPLIT_OVERLAP=0 SGLANG_FUSED_DECODE_GUARD=0
fi
if [ "${KERNEL:-lpt}" = split ]; then
  export SGLANG_MIXED_ATTENTION_KERNEL=persistent SGLANG_TWO_EXEC=1 SGLANG_FUSED_FA3_SM_SPLIT=1
  export FLASHINFER_SM_PARTITION=1 UP_STAGES=2 SGLANG_FUSED_FA3_SRC=$W/fi/tests/fused_fa3_persistent_ext.cu
fi
# Free ports are FOUND, not assumed: pods use host networking on hosts shared with other
# jobs, which hold ports this container cannot even see (a bare bind of [::]:30301 failed with
# no sglang process alive). Each port is probed by binding it, 50 apart so sglang's internal
# ports above it stay clear, and published in ports_r<role>.txt for mn_run.py.
# Every pick is also CLAIMED in $LOG/claimed_ports under a lock: a probe only proves a port is free
# now, and the two GPUs' workers pick concurrently -- GPU 1's http port once equalled GPU 0's PD
# bootstrap port, which the bootstrap server binds at once while the http server binds after the
# model loads, so every retry failed with "address already in use".
pick() { python3 - "$1" "$LOG/claimed_ports" <<'PY'
import fcntl, os, random, socket, sys
rng = random.Random(int(sys.argv[1]) ^ os.getpid() ^ int.from_bytes(os.urandom(4), "little"))
with open(sys.argv[2], "a+") as claims:
    fcntl.flock(claims, fcntl.LOCK_EX)
    claims.seek(0)
    taken = {int(x) for x in claims.read().split()}
    for _ in range(500):
        # Below the kernel's ephemeral range (32768-63999 on these pods): sglang's internal NCCL / ZMQ sockets and
        # outgoing connections take random ports from it while the model loads, so a probed port in
        # 40000-59000 was often gone by the time the http server bound it. 12001 is mooncake's RPC port.
        p = rng.randrange(13000, 19950, 50)
        if p in taken:
            continue
        try:
            for q in (p, p + 1, p + 2, p + 3):
                s = socket.socket(socket.AF_INET6); s.bind(("::", q)); s.close()
        except OSError:
            continue
        claims.write(f"{p}\n"); claims.flush()
        print(p); break
PY
}
rm -f $LOG/ports_r${ROLE}_g*.txt
mkdir -p $LOG; touch $LOG/claimed_ports
launch() {  # launch <gpu>: probe ports, start the worker, publish its ports
  ulimit -n $(ulimit -Hn)
  local g=$1
  PORT=$(pick $((ROLE * 2 + g + RANDOM)))
  BOOT=$(pick $((ROLE * 2 + g + 7919 + RANDOM)))
  # torch.distributed port, claimed like the others: unset, sglang takes http port + randint(100, 1000),
  # which lands on the 50-port grid other workers on the host claim; the worker then hung at
  # "Init parallel begin." until the supervisor killed it (pod 2 GPU 0: 3 x 6 min, 2026-10-16).
  NCCL=$(pick $((ROLE * 2 + g + 104729 + RANDOM)))
  ARGS=(--model $W/qwen7b --attention-backend flashinfer --mem-fraction-static 0.85
        --host :: --port $PORT --nccl-port $NCCL --context-length 131072 --watchdog-timeout 600
        --json-model-override-args '{"rope_scaling":{"factor":4.0,"original_max_position_embeddings":32768,"type":"yarn"}}')
  # Shared L3 prefix cache (goal points 2 and 4): every colocated worker writes the prefixes it
  # computes through to a cluster-wide mooncake store (RDMA, host DRAM) and prefetches a request's
  # prefix from it on a local miss, so any node can pull from any other -- a complete graph, with
  # replication happening as a side effect of serving rather than as a migration step.
  if [ -n "${HC_MASTER:-}" ]; then
    export MOONCAKE_MASTER=$HC_MASTER LOCAL_HOSTNAME=$SGLANG_HOST_IP MOONCAKE_PROTOCOL=rdma
    # ONE_NIC=1 pins every pod's store traffic to a single back-end NIC instead of spreading over
    # mlx5_1..4. At 16 GPUs on 4 hosts the 4-NIC fabric showed no contention (~236 GB moved, no
    # measurable effect), so point 4's congestion claim had nothing to bite on; this narrows
    # aggregate cross-node bandwidth ~4x to make source selection observable within the GPU cap.
    if [ "${ONE_NIC:-0}" = 1 ]; then
      export MC_MS_AUTO_DISC=0 MOONCAKE_DEVICE=mlx5_1
    else
      export MOONCAKE_DEVICE=mlx5_$(( 1 + (ROLE * 2 + g) % 4 ))
    fi
    # Host memory: the pods run at half their memory request (232 GB, to get scheduled), and the
    # default host pool (--hicache-ratio 2 = ~130 GB per H20 worker) plus a 32 GB segment, twice per
    # pod, got every worker OOM-killed at startup (2026-10-17). 48 GB + 24 GB per worker ~= 175 GB/pod.
    export MOONCAKE_GLOBAL_SEGMENT_SIZE=${HC_SEGMENT:-24gb}
    ARGS+=(--enable-hierarchical-cache --hicache-storage-backend mooncake --hicache-size ${HC_HOST_GB:-48})
  fi
  # Worker-side EDF: deadline-ordered admission with no chunk cap (no latency model), the
  # configuration that won on the MTP server (pd_colocation_vs_disagg.md point 3).
  # PREEMPT=1 (with EDF=1): EDF over the in-progress chunked prefill -- uncapped chunk preemption,
  # edf_preempt_chunk_v2.patch (in h20_overlay.tgz); park gate v2 PASS on H200 2026-10-16.
  if [ "${PREEMPT:-0}" = 1 ]; then
    export SGLANG_EDF_PREEMPT_CHUNK=1 SGLANG_EDF_RESERVE_EARLIER=1 SGLANG_EDF_PREEMPT_MAX_TOKENS=1000000000
  fi
  [ -n "${CHUNK:-}" ] && ARGS+=(--chunked-prefill-size $CHUNK)
  if [ "${EDF:-0}" = 1 ]; then
    export SGLANG_EDF_SCHEDULING=1 SGLANG_TTFT_SLO_MS=${TTFT_SLO_MS:-2000} SGLANG_EDF_LATENCY_MODEL=
    ARGS+=(--enable-priority-scheduling --schedule-low-priority-values-first
           --priority-scheduling-preemption-threshold 999999)
  fi
  # MTP serving (EAGLE3 chain draft, 3 steps, 4 verify tokens) on every role. Colocated workers
  # also run the 4-mode step (prefill chunk + verify rows in one target forward); disagg prefill
  # workers forward the draft hidden states to decode, so both arms speculate.
  # SPEC=2: the same EAGLE3 MTP without the 4-mode step (stock sglang MTP), to separate the cost
  # of speculation on this GPU from that of mixing prefill with verify.
  if [ "${SPEC:-0}" != 0 ]; then
    ARGS+=(--speculative-algorithm EAGLE3 --speculative-draft-model-path $W/qwen7b_eagle3_sgl
           --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
    [ "$MODE" = coloc ] && [ "$SPEC" = 1 ] && export SGLANG_ALLOW_MIXED_CHUNK_SPEC=1 SGLANG_MIXED_SPEC=1
    # Load + context gate (mixed_spec_ctx_queue_gate.patch): mix only when the prefill carries >= 8K
    # context or >= 4 requests wait; otherwise the prefill runs alone, as in stock MTP.
    [ "$MODE" = coloc ] && [ "$SPEC" = 1 ] && export SGLANG_MIXED_SPEC_MIN_CTX=${MIX_MIN_CTX:-8192} \
      SGLANG_MIXED_SPEC_MIN_QUEUE=${MIX_MIN_QUEUE:-4}
  fi
  case $MODE in
    coloc) ARGS+=(--enable-mixed-chunk) ;;
    prefill|decode)
      # Back-end RDMA NICs only. mlx5_0 is the front-end NIC (eth0) whose RoCEv2 GID is the IPv6
      # management address, while mlx5_1..4 (eth1-4) carry IPv4-mapped RoCEv2 GIDs; a worker on
      # mlx5_0 could not bring up a QP to one on mlx5_1..4 ("Failed to modify QP to RTR").
      IB=mlx5_$(( 1 + (ROLE * 2 + g) % 4 ))
      ARGS+=(--disaggregation-mode $MODE --disaggregation-transfer-backend mooncake
             --disaggregation-ib-device $IB --disaggregation-bootstrap-port $BOOT) ;;
  esac
  # One file per GPU: the two supervisors run concurrently, and a shared file rewritten via
  # one temp name lost an entry (pods 0, 2, 6 in coloc_cache), stalling the arm for 25 min.
  echo "$g $PORT $BOOT" > $LOG/ports_r${ROLE}_g${g}.tmp && mv $LOG/ports_r${ROLE}_g${g}.tmp $LOG/ports_r${ROLE}_g${g}.txt
  CUDA_VISIBLE_DEVICES=$g setsid nohup $W/venv/bin/python -m sglang.launch_server "${ARGS[@]}" \
    > $LOG/worker_r${ROLE}_g${g}.log 2>&1 < /dev/null &
  echo $! > $LOG/pid_r${ROLE}_g${g}
  echo "worker role=$ROLE gpu=$g port=$PORT boot=$BOOT mode=$MODE pid=$!"
}
# Supervisor: a port probed free can still be taken by another job before sglang binds it
# (seen: role 3 gpu 1, port 57000). On a bind failure, kill that worker and relaunch it on a
# freshly probed port; mn_run.py re-reads ports_r<role>.txt on every health poll.
supervise() {
  local g=$1
  for attempt in 1 2 3 4 5; do
    launch $g
    for t in $(seq 1 90); do
      sleep 4
      grep -q "address already in use" $LOG/worker_r${ROLE}_g${g}.log && break
      # Distributed init takes < 1 s when healthy; still waiting after ~60 s means a port clash.
      [ $t -ge 15 ] && grep -q "Init parallel begin" $LOG/worker_r${ROLE}_g${g}.log \
        && ! grep -q "Init parallel ends" $LOG/worker_r${ROLE}_g${g}.log && break
      grep -q "fired up" $LOG/worker_r${ROLE}_g${g}.log && return 0
    done
    grep -q "fired up" $LOG/worker_r${ROLE}_g${g}.log && return 0
    pkill -9 -P $(cat $LOG/pid_r${ROLE}_g${g}) 2>/dev/null; kill -9 $(cat $LOG/pid_r${ROLE}_g${g}) 2>/dev/null
    for p in $(ps -eo pid,cmd | grep "[s]glang" | grep -- "--port $PORT " | awk '{print $1}'); do kill -9 $p; done
    sleep 5
    mv $LOG/worker_r${ROLE}_g${g}.log $LOG/worker_r${ROLE}_g${g}.bindfail$attempt.log
  done
}
for g in 0 1; do
  supervise $g > $LOG/supervise_r${ROLE}_g${g}.log 2>&1 < /dev/null &
  disown
done
sleep 3
cat $LOG/ports_r${ROLE}_g*.txt 2>/dev/null
