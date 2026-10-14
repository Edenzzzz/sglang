#!/bin/bash
# Runs after the chunk sweep releases the GPU lock; single short kernel microbenchmark.
W=/dev/shm/w
until grep -q CHUNK_ITL_EXIT $W/chunk_itl_sweep2b_20261015.log 2>/dev/null; do sleep 30; done
exec 9>"$W/a2_kernel.lock"; flock 9
if pgrep -f '[s]glang.launch_server|[k]ernel_concurrency_driver|[f]used_reference' >/dev/null; then echo "GPU predecessor"; exit 1; fi
[ -s $W/codex_needs_gpu ] && { echo "yield to Codex"; exit 0; }
$W/venv/bin/python -u $W/b1_lmhead_epilogue_ceiling_20261015.py
