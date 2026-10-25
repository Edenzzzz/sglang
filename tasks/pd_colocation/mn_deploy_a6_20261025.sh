#!/bin/bash
# Deploy the A6 serving kernel to the H20 multi-node pods (job 2f608c1b1a67cf09): the FA3 backend's native-kernel hook
# and the EDF late-demote scheduler (both off unless their SGLANG_* switch is set, so the baseline arms are unchanged),
# the native / unified executor modules, and the five kernel libraries built and FP32-checked on worker-0 (A6 NP
# executor with batched merge rows and the 64-row class, flashinfer 396b04b, build np64). Applied only where the base files match the checked md5. Run from the CPU host.
#   bash mn_deploy_a6_20261025.sh <worker index>...
# Claude, 2026-10-25.
set -euo pipefail
J=2f608c1b1a67cf09; P=merlin-ssh-proxy-cn.byted.org
ssh_w() { timeout 300 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p 22222 $J.worker_$1.seedjob.wenxuan-tan@$P "$2" 2>&1 | grep -v Warning; }
R=/opt/tiger/sglang/tasks/debug_kernel_server_perf/resume_20261012
L=/tmp/a6_bundle_20261025; mkdir -p $L
ssh_w 0 'set -e; B=/dev/shm/w/a6_bundle; mkdir -p $B/lib $B/py; N=/dev/shm/w/qwen25_v3_common_screen_h200_20261023/python/sglang/srt
cp $N/layers/attention/flashattention_backend.py $N/managers/scheduler.py $B/py/
cp /opt/tiger/w/torch_ext_native_h200_20261023/native_fa2_q32/native_fa2_q32.so /opt/tiger/w/torch_ext_stock_h200_20261023/stock_fa3_pdl/stock_fa3_pdl.so \
   /opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so \
   /opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so \
   /opt/tiger/w/torch_ext_unified_np64/unified_dispatch_ext/unified_dispatch_ext.so $B/lib/
cd $B && tar czf /dev/shm/w/a6_bundle.tgz py lib && md5sum lib/*.so | cut -c1-12 | tr "\n" " "'
timeout 600 scp -o StrictHostKeyChecking=no -P 22222 $J.worker_0.seedjob.wenxuan-tan@$P:/dev/shm/w/a6_bundle.tgz $L/ 2>&1 | grep -v Warning || true
mkdir -p $L/a6 && cp $R/native_fa3_fa2_pdl_20261023.py $R/unified/unified_executor_20261024.py $L/a6/
grep -q prepare_unified $L/a6/native_fa3_fa2_pdl_20261023.py
cp /opt/tiger/sglang/tasks/pd_colocation/mn_node.sh $L/
for i in "$@"; do
  timeout 600 scp -o StrictHostKeyChecking=no -P 22222 $L/a6_bundle.tgz $L/a6/*.py $L/mn_node.sh $J.worker_$i.seedjob.wenxuan-tan@$P:/dev/shm/w/ 2>&1 | grep -v Warning || true
  ssh_w $i 'set -e; W=/dev/shm/w; S=$W/python/sglang/srt
    mkdir -p $W/a6 && tar xzf $W/a6_bundle.tgz -C $W/a6 && mv $W/native_fa3_fa2_pdl_20261023.py $W/unified_executor_20261024.py $W/a6/
    mkdir -p /opt/tiger/w/a6lib && cp $W/a6/lib/*.so /opt/tiger/w/a6lib/
    for f in layers/attention/flashattention_backend.py managers/scheduler.py; do
      cur=$(md5sum $S/$f | cut -c1-12); new=$(md5sum $W/a6/py/$(basename $f) | cut -c1-12)
      if [ "$cur" = "$new" ]; then continue; fi
      case $f:$cur in layers/attention/flashattention_backend.py:9ca27e6a5c34|managers/scheduler.py:f8fbbdb95e9b) ;;
        *) echo "UNEXPECTED_BASE $f $cur"; exit 1;; esac
      cp $S/$f $S/$f.pre_a6 && cp $W/a6/py/$(basename $f) $S/$f
    done
    mv $W/mn_node.sh $W/tasks/pd_colocation/mn_node.sh
    echo "worker deployed: $(md5sum $S/layers/attention/flashattention_backend.py $S/managers/scheduler.py | cut -c1-12 | tr "\n" " ") native=$(grep -c SGLANG_NATIVE_PDL $S/layers/attention/flashattention_backend.py) libs=$(ls /opt/tiger/w/a6lib | wc -l)"'
done
echo A6_DEPLOY_DONE
