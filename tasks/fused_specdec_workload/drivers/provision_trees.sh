#!/bin/bash
# Step 2 of provisioning: the flashinfer and sglang trees.
#
# Cloned from upstream and then overlaid with the changed files, rather than
# copied wholesale from the local checkout. The trees are too large to push
# through the webshell in base64 chunks, and a clone plus overlay is
# reproducible, which the old hand-assembled copy was not.
set -ex
W=/dev/shm/w
. $W/venv/bin/activate

# flashinfer: the fused kernel needs include/, csrc/ and 3rdparty/cutlass
if [ ! -d $W/fi/include ]; then
  git clone --depth 50 https://github.com/flashinfer-ai/flashinfer $W/fi
  cd $W/fi && git submodule update --init --depth 1 3rdparty/cutlass
fi

# sglang: only the python tree is used, via PYTHONPATH
if [ ! -d $W/python/sglang ]; then
  git clone --depth 50 https://github.com/sgl-project/sglang $W/sgl_src
  cp -r $W/sgl_src/python $W/python
fi

echo "TREESDONE  overlay next: bash tests/sync_to_pod.sh from the local checkout,"
echo "then push python/sglang/srt/layers/attention/{flashinfer_backend,mixed_dispatch}.py,"
echo "python/sglang/srt/managers/scheduler.py and python/sglang/srt/speculative/eagle_worker.py"
