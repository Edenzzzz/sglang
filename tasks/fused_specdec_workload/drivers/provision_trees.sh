#!/bin/bash
# Step 2 of provisioning: the source trees the fused/split kernel builds against.
#
# Rewritten 2026-10-06 after the third trial expiry. The previous version cloned
# UPSTREAM flashinfer, which is not API-compatible with the fork this work uses
# (`BatchAttention(workspace, kv_layout)` differs, and upstream has gained
# apache-tvm-ffi), so it never actually restored a usable tree.
#
# Division of labour, by size:
#   pushed from the dev box  : the flashinfer FORK's include/csrc/tests/package (~1.9 MB)
#                              and sglang's python/ (~5.8 MB) -- small enough for the
#                              webshell's ~60 KB-per-call base64 channel
#   cloned here, at a PINNED SHA : cutlass (151 MB) and flash-attention, which are
#                              unmodified upstream and must not drift, since a different
#                              cutlass silently changes codegen and makes numbers
#                              incomparable with earlier runs
set -ex
W=/dev/shm/w
. $W/venv/bin/activate

CUTLASS_SHA=a49a78ffefc86a87160dfe0ccc3a3a2d1622c918
FA_SHA=e9515d5dee6ade134a33d6020d38d01ef0596996

# `git fetch --depth 1 <sha>` is REJECTED by GitHub for an arbitrary SHA (the server must
# allow reachable-sha1-in-want, and it does not), which cost a cycle last time. A
# blob-filtered full clone plus a checkout is the form that works and still avoids pulling
# every blob in history.
if [ ! -d $W/fi/3rdparty/cutlass/include ]; then
  mkdir -p $W/fi/3rdparty
  git clone --filter=blob:none --no-checkout \
      https://github.com/NVIDIA/cutlass $W/fi/3rdparty/cutlass
  git -C $W/fi/3rdparty/cutlass checkout $CUTLASS_SHA
fi

if [ ! -d $W/flash-attention/hopper ]; then
  git clone --filter=blob:none --no-checkout \
      https://github.com/Dao-AILab/flash-attention $W/flash-attention
  git -C $W/flash-attention checkout $FA_SHA
fi

# The fork's own trees arrive by push; fail loudly rather than silently building against
# a half-restored tree, which is how a run gets measured on the wrong base.
for p in $W/fi/include/flashinfer $W/fi/tests $W/python/sglang; do
  [ -d "$p" ] || { echo "MISSING $p -- push it from the dev box (drivers/px.py)"; exit 1; }
done

# fa3_upstream.cuh includes flash-attention by a ../../../../../ relative path, so the
# layout matters as much as the contents.
test -f $W/flash-attention/hopper/flash_fwd_kernel_sm90.h
test -f $W/fi/3rdparty/cutlass/include/cutlass/cutlass.h
echo TREESDONE
