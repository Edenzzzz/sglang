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
  pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  pip install -q transformers numpy ninja pybase64 orjson uvloop zmq fastapi uvicorn \
                 aiohttp requests tqdm pillow psutil einops sentencepiece \
                 interegular partial_json_parser llguidance msgspec compressed_tensors \
                 python-multipart soundfile scipy torchao blobfile
else
  . $W/venv/bin/activate
fi

# 2. repos: pushed from the local checkouts, not cloned, so they carry the work
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
