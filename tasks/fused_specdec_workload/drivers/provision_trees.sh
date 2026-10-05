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

# --- Pushing the fork's tree (the approach that actually works) -------------
#
#   tar czf fork_core.tgz include csrc flashinfer tests     # 1.9 MB compressed
#   base64 -w0 | split into ~90 KB chunks                   # webshell arg limit
#   per chunk: printf %s '<chunk>' >> /dev/shm/fork.b64
#   base64 -d | tar xzf - -C /dev/shm/w/fi
#
# Two traps, both hit:
#
# 1. `flashinfer/data/` holds ABSOLUTE symlinks into the local checkout
#    (`csrc -> /opt/tiger/flashinfer/csrc`, same for cutlass). They survive the
#    tar and point at nothing on the pod, and the package then dies at import
#    with `FileExistsError: .../flashinfer/data/csrc` when it tries to recreate
#    them. Do NOT simply delete them -- the package reads
#    `flashinfer/data/csrc/*.jinja` through that path. RE-POINT them at the pod:
#      ln -sfn /dev/shm/w/fi/{csrc,include,3rdparty/cutlass} \
#              /dev/shm/w/fi/flashinfer/data/{csrc,include,cutlass}
# 2. Clear `~/.cache/flashinfer` after replacing the tree, or the JIT serves
#    objects built against the previous headers.
