#!/bin/bash
# Item 18: push the piggyback headers to the H200 and compile the probe arms, as soon as the host
# answers. The 4-GPU trial dropped its ssh listener mid-session, and the register numbers already
# taken (q64gate / q64gate_noop / q64gate_only16 / q64gate_mrc240) are from ITS ptxas, so the
# remaining arms have to run on the same host to be comparable -- ptxas register counts are not
# comparable across toolchains.
#
# Arms:
#   piggy_off  FUSED_Q64_PIGGYBACK=0, the default build. A REGRESSION CONTROL: every piggyback
#              hook is compiled out, so this must reproduce the numbers measured before the
#              headers were touched (q128: 168 reg / 33424 B spill; q64: 144 / 19840).
#   piggy_on   FUSED_Q64_PIGGYBACK=1 with FUSED_DRAIN_ONLY16=1, i.e. the corrected mechanism: a
#              CTA one warpgroup WIDER than FA3's own thread count, those threads drawing only
#              the 16-row tile class while the rest of the CTA runs FA3's mainloop.
set -u
H=/tmp/h2ssh.sh
R="tiger@[fdbd:dccd:cdd2:2306::283]:/dev/shm/w/ub/port07_unified/include/flashinfer/attention/"
FI=/opt/tiger/flashinfer/port07
LOG=${LOG:-/opt/tiger/sglang/tasks/debug_kernel_server_perf/resume_20261012/unified/q64_piggyback_gate_20261028.log}
: > "$LOG"
for i in $(seq 1 240); do
  if $H "echo UP" 2>/dev/null | grep -q UP; then echo "host up after $((i*30))s" >> "$LOG"; break; fi
  sleep 30
done
$H "echo UP" 2>/dev/null | grep -q UP || { echo "HOST_NEVER_RETURNED after 2h" >> "$LOG"; exit 1; }
$H "nvidia-smi --query-compute-apps=pid --format=csv,noheader" >> "$LOG" 2>&1
for f in persistent_fa3_fused.cuh fused_barriers.cuh; do
  $H scp "$FI/include/flashinfer/attention/$f" "$R" >> "$LOG" 2>&1 || { echo "SCP_FAILED $f" >> "$LOG"; exit 1; }
done
E='TORCH_CUDA_ARCH_LIST=9.0a SGLANG_FUSED_FI_ROOT=/dev/shm/w/ub/port07_unified SGLANG_FUSED_MERGED_DIR=/dev/shm/w/ub/merged PYTHONPATH=/dev/shm/w/ub:/dev/shm/w/up/python:/dev/shm/w/ub/port07_unified'
{
  echo "=== ARM piggy_off (regression control, must reproduce 168/33424 and 144/19840)"
  $H "cd /dev/shm/w/ub && env $E /dev/shm/w/venv/bin/python -I -u gate_q64_registers_20261028.py \
        --qs 128,64 --out /dev/shm/w/ub/q64gate_piggyoff" 2>&1
  echo "=== ARM piggy_on (FA3 q64 + one added warpgroup drawing the 16-row class only)"
  $H "cd /dev/shm/w/ub && env $E /dev/shm/w/venv/bin/python -I -u gate_q64_registers_20261028.py \
        --qs 64,128 '--extra=-DFUSED_DRAIN_ONLY16=1 -DFUSED_Q64_PIGGYBACK=1' \
        --out /dev/shm/w/ub/q64gate_piggy" 2>&1
  echo "=== C7512 (wgmma serialized for want of registers) per arm"
  $H "grep -c C7512 /dev/shm/w/ub/q64gate_piggyoff/ptxas_fi_q*.log /dev/shm/w/ub/q64gate_piggy/ptxas_fi_q*.log" 2>&1
} >> "$LOG" 2>&1
echo PIGGYBACK_GATE_DONE >> "$LOG"
