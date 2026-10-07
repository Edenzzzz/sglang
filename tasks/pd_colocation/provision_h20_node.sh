#!/bin/bash
# Per-node provisioning for the H20 multi-node study (run on EVERY pod, in parallel).
# venv (torch 2.8.0/cu128 pinned LAST, as in drivers/provision.sh), sglang deps, Qwen2.5-7B
# weights from ModelScope (15 MB/s here; hf-mirror was 80 KB/s and HDFS writes are denied).
set -x
W=/dev/shm/w
mkdir -p $W && cd $W
if [ ! -x $W/venv/bin/python ]; then
  python3 -m venv $W/venv
  . $W/venv/bin/activate
  pip install -q --upgrade pip
  pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
  pip install -q -r $W/sgl_deps.txt
  pip install -q apache-tvm-ffi pynvml nvidia-ml-py cuda-python aiohttp modelscope scikit-learn ninja
  pip install -q "torchao==0.13.0"
  pip install -q --no-deps sgl-kernel==0.3.16.post4 xgrammar==0.1.25
  pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
  pip uninstall -y -q torchaudio 2>/dev/null || true
else
  . $W/venv/bin/activate
fi
python -c "import torch; print('TORCH', torch.__version__, torch.cuda.device_count())"
if [ ! -f $W/qwen7b/config.json ]; then
  python - <<'PY'
from modelscope import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir="/dev/shm/w/qwen7b")
print("WEIGHTS", p)
PY
fi
ls $W/qwen7b | head -20
echo PROVISION_NODE_DONE
