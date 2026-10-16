#!/bin/bash
# H20 node provisioning for the 2026-10-16 multi-node runs (run on every pod; resumable).
# Upstream sglang 3fef129 + the H200 serving tree's 26 changed files (h20_overlay.tgz = up_edf_preempt
# vs clean upstream: split dispatch, EDF + chunk preemption, program-concurrency replay), torch 2.14.1,
# flashinfer 0.7.0.post1, Qwen2.5-7B-Instruct + thoughtworks EAGLE3 head, ShareGPT. Claude, 2026-10-16.
set -u
W=/dev/shm/w; mkdir -p $W; cd $W
log() { echo "[prov $(hostname | tail -c 6) $(date +%H:%M:%S)] $*"; }
# github is intermittently unreachable from some pods: retry, then fall back to pod 0's checkout
# served over the cluster network (HUB=http://<pod0-ipv4>:<port>, /dev/shm/w/up.tgz).
if [ ! -d up/.git ]; then
  for a in 1 2 3; do rm -rf up; git clone --quiet https://github.com/sgl-project/sglang.git up && break; sleep 10; done
  if [ ! -d up/.git ] && [ -n "${HUB:-}" ]; then rm -rf up; curl -sf $HUB/up.tgz | tar xzf - && log "up from hub"; fi
  git -C up checkout --quiet 3fef129ff8a32c32da5f6a0efd33485f6cc6233c || { log CLONE_FAIL; exit 1; }
fi
log "up $(git -C up rev-parse --short HEAD)"
# /dev/shm is noexec on these pods: shared libraries cannot be mapped from it, so the venv lives on
# the root overlay and is symlinked in (every driver uses /dev/shm/w/venv).
if [ ! -x /opt/tiger/w/venv/bin/python ]; then rm -rf $W/venv; mkdir -p /opt/tiger/w; python3 -m venv /opt/tiger/w/venv; fi
[ -L $W/venv ] || { rm -rf $W/venv; ln -sfn /opt/tiger/w/venv $W/venv; }
P=$W/venv/bin/python
$P -c "import torch" 2>/dev/null || { $P -m pip install -q --upgrade pip setuptools wheel >/dev/null 2>&1; $P -m pip install -q torch==2.14.1 || { log TORCH_FAIL; exit 1; }; }
$P -c "import torch;print('TORCH',torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.device_count())" 2>&1 | tail -1
$P -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" || { log CUDA_UNAVAILABLE; exit 2; }
$P -c "import sgl_kernel" 2>/dev/null || { $P -m pip install -q wheel_stub >/dev/null 2>&1; $P -m pip install -q --no-build-isolation $W/up/python || log "sglang install nonzero"; }
$P -c "import sgl_kernel; print('sgl_kernel OK')" 2>&1 | tail -1
$P -c "import flashinfer" 2>/dev/null || $P -m pip install -q flashinfer-python==0.7.0.post1 || log "flashinfer nonzero"
[ -f $W/port07.tgz ] && tar xzf $W/port07.tgz -C $W && ln -sfn $W/port07 $W/fi && ln -sfn $W/up/python $W/python
[ -f $W/h20_overlay.tgz ] && tar xzf $W/h20_overlay.tgz -C $W/up/python && log "overlay applied $(md5sum < $W/up/python/sglang/srt/managers/scheduler.py | cut -c1-8)"
$P -m pip install -q 'huggingface_hub[cli]' aiohttp >/dev/null 2>&1
# huggingface.co is unreachable from these pods; Qwen from ModelScope (~80 MB/s), the EAGLE3 head
# and ShareGPT from hf-mirror (~11 MB/s).
$P -m pip install -q modelscope >/dev/null 2>&1
export HF_ENDPOINT=https://hf-mirror.com
[ -f qwen7b/model-00004-of-00004.safetensors ] || $P -c "from modelscope import snapshot_download as s; s('Qwen/Qwen2.5-7B-Instruct', local_dir='$W/qwen7b')" >/dev/null 2>&1 || log TARGET_FAIL
log "target shards $(ls qwen7b/*.safetensors 2>/dev/null | wc -l)"
[ -f qwen7b_eagle3/model.safetensors ] || $P -c "from huggingface_hub import snapshot_download as s; s('thoughtworks/Qwen2.5-7B-Instruct-Eagle3', local_dir='$W/qwen7b_eagle3')" >/dev/null 2>&1 || log DRAFT_FAIL
if [ ! -f qwen7b_eagle3_sgl/config.json ]; then
  mkdir -p qwen7b_eagle3_sgl; for f in qwen7b_eagle3/*; do ln -sf $W/$f qwen7b_eagle3_sgl/; done; rm -f qwen7b_eagle3_sgl/config.json
  $P -c "import json;c=json.load(open('$W/qwen7b_eagle3/config.json'));c['architectures']=['LlamaForCausalLMEagle3'];json.dump(c,open('$W/qwen7b_eagle3_sgl/config.json','w'))"
fi
mkdir -p workloads/tracelab
[ -s workloads/ShareGPT_V3_unfiltered_cleaned_split.json ] || $P -c "
from huggingface_hub import hf_hub_download;import shutil
shutil.copy(hf_hub_download('anon8231489123/ShareGPT_Vicuna_unfiltered','ShareGPT_V3_unfiltered_cleaned_split.json',repo_type='dataset'),'$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json')" >/dev/null 2>&1 || log SHAREGPT_FAIL
$P -m pip install -q mooncake-transfer-engine sglang-router nvidia-cuda-runtime-cu12 >/dev/null 2>&1
ulimit -n $(ulimit -Hn)
log "PROVISION_DONE torch_ok sgl=$($P -c 'import sgl_kernel' 2>/dev/null && echo 1) fi=$($P -c 'import flashinfer' 2>/dev/null && echo 1) qwen=$(ls qwen7b/*.safetensors 2>/dev/null | wc -l) draft=$([ -f qwen7b_eagle3/model.safetensors ] && echo 1) sgpt=$([ -s workloads/ShareGPT_V3_unfiltered_cleaned_split.json ] && echo 1)"
