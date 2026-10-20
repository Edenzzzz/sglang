#!/bin/bash
# semi-PD baseline provisioning on the H200 pod (2026-10-20): Semi-PD (fork of sglang v0.4.4.post1, torch 2.5 / cu124)
# needs sgl-kernel 0.0.5, which PyPI no longer serves, so it is built from the fork's sgl-kernel/ with a CUDA 12.4
# toolchain assembled from NVIDIA's pip wheels (nvcc, runtime, cccl, cublas, ...) under $B/cuda -- the pod's system
# toolkit is CUDA 13. Everything lives under /opt/tiger/semipd. Run niced; logs to $B/prov.log. Claude.
set -euxo pipefail
B=/opt/tiger/semipd; mkdir -p $B; cd $B
[ -d Semi-PD ] || git clone https://github.com/infinigence/Semi-PD
cd Semi-PD && git submodule update --init --recursive sgl-kernel 2>/dev/null || true; cd $B
[ -x venv/bin/python ] || python3.12 -m venv venv
V=$B/venv/bin
$V/pip install -U pip wheel setuptools ninja packaging
$V/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
$V/pip install nvidia-cuda-nvcc-cu12==12.4.131 nvidia-cuda-cccl-cu12==12.4.127 nvidia-cuda-runtime-cu12==12.4.127 \
  nvidia-nvtx-cu12==12.4.127 nvidia-cuda-cupti-cu12==12.4.127
SP=$($V/python -c 'import site; print(site.getsitepackages()[0])')/nvidia
C=$B/cuda; rm -rf $C; mkdir -p $C/bin $C/include $C/lib64
cp -a $SP/cuda_nvcc/bin/. $C/bin/; cp -a $SP/cuda_nvcc/nvvm $C/
for d in cuda_nvcc cuda_runtime cuda_cccl cublas cusparse curand cusolver cufft nvtx cuda_cupti nvjitlink; do
  [ -d $SP/$d/include ] && cp -a $SP/$d/include/. $C/include/
  [ -d $SP/$d/lib ] && cp -a $SP/$d/lib/. $C/lib64/
done
cd $C/lib64 && for f in *.so.*; do b=${f%%.so.*}.so; [ -e $b ] || ln -s $f $b; done; cd $B
# The cccl wheel ships no headers (nv/target, cuda/std, cub, thrust): take CUDA 12.4's CCCL (2.3.2) from NVIDIA's repo.
[ -d $B/cccl ] || git clone --depth 1 --branch v2.3.2 https://github.com/NVIDIA/cccl $B/cccl
cp -a $B/cccl/libcudacxx/include/. $C/include/; cp -a $B/cccl/cub/cub $C/include/; cp -a $B/cccl/thrust/thrust $C/include/
export CUDA_HOME=$C PATH=$C/bin:$PATH LD_LIBRARY_PATH=$C/lib64:${LD_LIBRARY_PATH:-}
nvcc --version
cd $B/Semi-PD/semi-pd-ipc && TORCH_CUDA_ARCH_LIST=9.0 $V/pip install -e . --no-build-isolation
cd $B/Semi-PD/sgl-kernel && git submodule update --init --recursive || true
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=24 $V/pip install -e . --no-build-isolation
cd $B/Semi-PD && $V/pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
$V/python -c "import sgl_kernel, flashinfer, sglang; print('SEMIPD_IMPORT_OK', sglang.__version__)"
echo SEMIPD_PROVISION_DONE
