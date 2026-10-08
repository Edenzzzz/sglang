#!/bin/bash
# Rebuild the pod working set from scratch: venv, flashinfer, sglang, weights.
# Written after the 2026-10-05 trial expired and took /dev/shm/w with it.
set -x
W=/dev/shm/w
mkdir -p $W
cd $W

# 1. venv with the torch the serving path needs. The system python is
#    2.9.1/cu131 and the fused extension was built against 2.8.0/cu128; mixing
#    them produced ABI failures on the old pod, which is why serve.sh used a venv.
if [ ! -x $W/venv/bin/python ]; then
  python3 -m venv $W/venv
  . $W/venv/bin/activate
  pip install -q --upgrade pip
  # torchvision MUST be pinned in the same command: the deps install pulls a torchvision built
  # for the upgraded torch, and after re-pinning torch alone its C++ ops fail to register
  # ("operator torchvision::nms does not exist", surfacing as AutoProcessor import failure).
  pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
  pip uninstall -y -q torchaudio 2>/dev/null || true
  # The full set, learned by running into them one at a time: a freshly cloned
  # flashinfer wants apache-tvm-ffi, and sglang's server path wants pynvml before
  # it will even import. Installing them together saves a build cycle per miss.
  # Install the deps FIRST, then pin torch LAST. Installing them after torch
  # silently upgraded it to a version that requires C++20, which surfaced as
  # "#error C++20 or later compatible compiler is required to use PyTorch" when
  # the fused extension built with -std=c++17 -- and a drifted torch would have
  # made every number incomparable with the earlier runs, so this ordering is not
  # cosmetic. compressed-tensors will complain it wants torch>=2.10; ignore it.
  pip install -q numpy ninja transformers tokenizers safetensors huggingface_hub \
                 apache-tvm-ffi pynvml nvidia-ml-py cuda-python \
                 pybase64 orjson uvloop pyzmq fastapi uvicorn aiohttp requests tqdm \
                 pillow psutil einops sentencepiece setproctitle \
                 interegular partial_json_parser llguidance msgspec compressed_tensors \
                 python-multipart soundfile scipy torchao blobfile hf_transfer \
                 ipython openai
  # The hand-picked list above kept missing modules the server needs at init (IPython,
  # openai, dill each cost a failed boot). Install sglang's OWN declared dependencies
  # instead, kernel stack excluded -- this needs the python/ tree in place, so on a fresh
  # pod run it after build_sglang_tree.sh. It upgrades torch transitively (seen: 2.8.0 ->
  # 2.14.1) even with torch excluded, which is exactly why the re-pin below must stay last.
  if [ -f $W/python/pyproject.toml ]; then
    python $W/drivers/sgl_deps_from_pyproject.py && pip install -q -r $W/sgl_deps.txt
  fi
  pip install -q "torchao==0.13.0"   # newer torchao imports ScalingType, absent in torch 2.8
  # The kernel-stack packages excluded above are still REQUIRED by the server -- excluded
  # only so the deps install cannot pull a torch with them. Install the versions this
  # sglang's pyproject pins, with --no-deps for the same reason.
  pip install -q --no-deps sgl-kernel==0.3.16.post4 xgrammar==0.1.25
  # RE-PIN torch LAST, and keep it last. The dependency list above upgrades torch (seen:
  # 2.8.0 -> 2.14.1+cu130), and torch >= 2.9 requires C++20 while the fused harness compiles
  # at -std=c++17, so every extension build then dies with
  #   #error C++20 or later compatible compiler is required to use PyTorch
  # torchvision MUST be pinned in the same command: the deps install pulls a torchvision built
  # for the upgraded torch, and after re-pinning torch alone its C++ ops fail to register
  # ("operator torchvision::nms does not exist", surfacing as AutoProcessor import failure).
  pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
  pip uninstall -y -q torchaudio 2>/dev/null || true
else
  . $W/venv/bin/activate
fi

# 2. repos. NOTE: provision_trees.sh clones UPSTREAM, which is not the fork this
#    work was developed against -- the fork's history interleaves our commits with
#    upstream's, so the base commit cannot be identified by date. Cloning upstream
#    and overlaying the eight changed files is an unblock, not a clean restore,
#    and any number measured that way sits on a different base than the earlier
#    ones. The clean fix is to keep a tarball of the fork's include/, csrc/ and
#    tests/ somewhere durable.
mkdir -p $W/fi $W/python
echo "PUSH: tar the local flashinfer include/csrc/tests and sglang python/ here"

# 3. weights
export HF_HOME=$W/hf
if [ ! -d $W/qwen7b ]; then
  python - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir="/dev/shm/w/qwen7b",
                      ignore_patterns=["*.pt", "original/*"])
print("weights at", p)
PY
fi
echo PROVISIONDONE
