#!/bin/bash
# Deploy the 2026-10-26 serving A6 library (np64 configuration, tile-rounded FA2 chunk; built on H20 pod 4 from
# port07_srv = flashinfer 24c0545 sources, md5 bc43aedeb9ed; FP32 + 20 graph replays pass on 43 GLM captures, same
# source as the H200 build that passed 96 Qwen captures) to pods 0-3 as /opt/tiger/w/a6lib/unified_dispatch_ext.so,
# keeping the previous library as unified_dispatch_ext.so.np64_20261025. Waits for PID $1 (the running stage-3 driver)
# so no measured arm sees a library swap. Claude, 2026-10-26.
set -euo pipefail
[ -z "${1:-}" ] || while kill -0 "$1" 2>/dev/null; do sleep 60; done
J=2f608c1b1a67cf09; P=merlin-ssh-proxy-cn.byted.org; WANT=bc43aedeb9ed
ssh_w() { timeout 300 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p 22222 $J.worker_$1.seedjob.wenxuan-tan@$P "$2" 2>&1 | { grep -v Warning || true; }; }
L=/tmp/a6srv_20261026; mkdir -p $L
timeout 600 scp -q -o StrictHostKeyChecking=no -P 22222 $J.worker_4.seedjob.wenxuan-tan@$P:/dev/shm/w/a6srv/unified_dispatch_ext.so $L/ 2>&1 | grep -v Warning || true
[ "$(md5sum $L/unified_dispatch_ext.so | cut -c1-12)" = $WANT ] || { echo LOCAL_MD5_BAD; exit 1; }
for i in 0 1 2 3; do
  ssh_w $i 'A=/opt/tiger/w/a6lib; mkdir -p $A; [ -f $A/unified_dispatch_ext.so.np64_20261025 ] || cp $A/unified_dispatch_ext.so $A/unified_dispatch_ext.so.np64_20261025 2>/dev/null || true'
  timeout 600 scp -q -o StrictHostKeyChecking=no -P 22222 $L/unified_dispatch_ext.so $J.worker_$i.seedjob.wenxuan-tan@$P:/opt/tiger/w/a6lib/unified_dispatch_ext.so.new 2>&1 | grep -v Warning || true
  ssh_w $i "A=/opt/tiger/w/a6lib; [ \"\$(md5sum \$A/unified_dispatch_ext.so.new | cut -c1-12)\" = $WANT ] && mv \$A/unified_dispatch_ext.so.new \$A/unified_dispatch_ext.so && echo \"worker-$i a6lib=\$(md5sum \$A/unified_dispatch_ext.so | cut -c1-12) prev=\$(md5sum \$A/unified_dispatch_ext.so.np64_20261025 2>/dev/null | cut -c1-12)\" || echo \"worker-$i MD5_BAD\""
done
echo A6LIB_DEPLOY_DONE
