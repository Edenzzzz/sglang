#!/bin/bash
# Step 2 of provisioning: the flashinfer and sglang trees.
#
# DOES NOT WORK as written, and the reason is recorded rather than patched over.
#
# Cloning upstream flashinfer and overlaying the changed files was an attempt to
# avoid pushing a large tree through the webshell. Upstream is NOT API-compatible
# with the fork this work uses: `BatchAttention(workspace, kv_layout)` in the fork
# against a different signature upstream, which fails immediately with
# "TypeError: attribute name must be string, not 'Tensor'". Upstream has also
# gained dependencies the fork does not have (apache-tvm-ffi).
#
# So the fork's own include/, csrc/, tests/ and python package are required, and
# a durable tarball of them is the actual fix. The eight-file overlay only works
# on top of the fork.
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
