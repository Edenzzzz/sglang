#!/bin/bash
# FA2-executor A/B over header knobs. Each arm passes -D flags through
# FLASHINFER_EXTRA_CUDAFLAGS with its OWN JIT workspace, so arms build in parallel and none
# can reuse another arm's (or a stale) module -- the failure that voided earlier FA2 items.
# Usage: fa2_ab.sh "tag:-DFOO=1 -DBAR=2" "tag2:" "tag3:-DX=1|ENVVAR=V ENVVAR2=V" ...
# Text after "|" is runtime env for that arm (host-side knobs read at plan/run time).
# FIXTURES (env) selects the bench fixtures for the timed runs.
set -u
W=/dev/shm/w
cd $W/fi
. $W/venv/bin/activate
export PYTHONPATH=$W/python:$W/fi HF_HOME=$W/hf TORCH_CUDA_ARCH_LIST=9.0a FLASHINFER_NVCC_THREADS=8
mkdir -p $W/fa2ab
echo "##### build ${#@} arms in parallel"
pids=()
for a in "$@"; do
  tag=${a%%:*}; rest=${a#*:}; fl=${rest%%|*}
  rm -rf $W/fa2ws_$tag
  ( FLASHINFER_WORKSPACE_BASE=$W/fa2ws_$tag FLASHINFER_EXTRA_CUDAFLAGS="$fl" ARM=$tag \
      FIXTURES=decode ITERS=1 timeout 2400 python tests/fa2_stage_bench.py > $W/fa2ab/$tag.build 2>&1
    echo "$tag build rc=$? $(grep -cE 'PASS|FAIL' $W/fa2ab/$tag.build)" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
echo "##### run (serial on the GPU)"
for rep in 1 2 3; do
  for a in "$@"; do
    tag=${a%%:*}; rest=${a#*:}; fl=${rest%%|*}; ev=""; [ "$rest" != "$fl" ] && ev=${rest#*|}
    env $ev FLASHINFER_WORKSPACE_BASE=$W/fa2ws_$tag FLASHINFER_EXTRA_CUDAFLAGS="$fl" ARM=$tag \
      timeout 1800 python tests/fa2_stage_bench.py > $W/fa2ab/$tag.r$rep 2>&1
    grep -E 'PASS|FAIL' $W/fa2ab/$tag.r$rep | sed "s/^/r$rep /"
    grep -iE 'error|Traceback' $W/fa2ab/$tag.r$rep | head -3
    echo "r$rep $tag [$fl | $ev] fallbacks=$(grep -ci fallback $W/fa2ab/$tag.r$rep)"
  done
done
echo FA2ABDONE
