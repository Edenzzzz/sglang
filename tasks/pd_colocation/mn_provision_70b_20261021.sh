#!/bin/bash
# H20 4-GPU-pod provisioning for the Llama-3.3-70B runs (2026-10-21; run on every pod; resumable).
# Same stack as the 2026-10-16..20 8-GPU runs: upstream sglang 3fef129 + the serving tree as it stood on those pods
# (tree_20261020.tgz: the 32 files that differ from upstream, taken from pod 0 of bb854a10b8954223 on 2026-10-20, every
# in-place patch included) + flashinfer port07, torch 2.14.1. The tree is verified against tree_manifest_20261020.txt
# (md5 prefixes) before anything runs. Models: Llama-3.3-70B-Instruct (ModelScope) and its trained EAGLE3 head
# (yuhuili/EAGLE3-LLaMA3.3-Instruct-70B, hf-mirror), on the root overlay (/dev/shm holds 227 GB). Claude.
set -u
W=/dev/shm/w; mkdir -p $W; cd $W
log() { echo "[prov $(hostname | tail -c 6) $(date +%H:%M:%S)] $*"; }
if [ ! -d up/.git ]; then
  for url in https://github.com/sgl-project/sglang.git https://ghfast.top/https://github.com/sgl-project/sglang.git; do
    rm -rf up; git clone --quiet --filter=blob:none $url up && break
  done
  [ -n "${HUB:-}" ] && [ ! -d up/.git ] && { curl -sf $HUB/up.tgz | tar xzf - && log "up from hub"; }
  git -C up checkout --quiet 3fef129ff8a32c32da5f6a0efd33485f6cc6233c || { log CLONE_FAIL; exit 1; }
fi
log "up $(git -C up rev-parse --short HEAD)"
# /dev/shm is noexec on these pods: the venv lives on the root overlay.
if [ ! -x /opt/tiger/w/venv/bin/python ]; then rm -rf $W/venv; mkdir -p /opt/tiger/w; python3 -m venv /opt/tiger/w/venv; fi
[ -L $W/venv ] || { rm -rf $W/venv; ln -sfn /opt/tiger/w/venv $W/venv; }
P=$W/venv/bin/python
$P -c "import torch" 2>/dev/null || { $P -m pip install -q --upgrade pip setuptools wheel >/dev/null 2>&1; $P -m pip install -q torch==2.14.1 || { log TORCH_FAIL; exit 1; }; }
$P -c "import torch,sys;sys.exit(0 if torch.cuda.device_count() >= 4 else 1)" || { log CUDA_UNAVAILABLE; exit 2; }
$P -c "import sgl_kernel" 2>/dev/null || { $P -m pip install -q wheel_stub >/dev/null 2>&1; $P -m pip install -q --no-build-isolation $W/up/python || log "sglang install nonzero"; }
$P -c "import flashinfer" 2>/dev/null || $P -m pip install -q flashinfer-python==0.7.0.post1 || log "flashinfer nonzero"
[ -f $W/port07.tgz ] && tar xzf $W/port07.tgz -C $W && ln -sfn $W/port07 $W/fi && ln -sfn $W/up/python $W/python
tar xzf $W/tree_20261020.tgz -C $W/up
bad=0
while read -r h f; do
  [ "$h" = DIR ] && continue
  [ "$(md5sum < $W/up/$f | cut -c1-12)" = "$h" ] || { log "MANIFEST_MISMATCH $f"; bad=1; }
done < $W/tree_manifest_20261020.txt
[ $bad = 0 ] && log "tree verified ($(wc -l < $W/tree_manifest_20261020.txt) entries)" || exit 3
$P -m pip install -q 'huggingface_hub[cli]' aiohttp modelscope >/dev/null 2>&1
export HF_ENDPOINT=https://hf-mirror.com
M=/opt/tiger/w; mkdir -p $M
[ -f $M/llama70b/model-00030-of-00030.safetensors ] || $P -c "from modelscope import snapshot_download as s; s('LLM-Research/Llama-3.3-70B-Instruct', local_dir='$M/llama70b')" >/dev/null 2>&1 || log TARGET_FAIL
ln -sfn $M/llama70b $W/llama70b
log "target shards $(ls $M/llama70b/*.safetensors 2>/dev/null | wc -l)"
[ -f $M/llama70b_eagle3/config.json ] || $P -c "from huggingface_hub import snapshot_download as s; s('yuhuili/EAGLE3-LLaMA3.3-Instruct-70B', local_dir='$M/llama70b_eagle3')" >/dev/null 2>&1 || log DRAFT_FAIL
ln -sfn $M/llama70b_eagle3 $W/llama70b_eagle3
mkdir -p workloads/tracelab
[ -f $W/tl.tgz ] && tar xzf $W/tl.tgz -C $W
[ -s workloads/ShareGPT_V3_unfiltered_cleaned_split.json ] || $P -c "
from huggingface_hub import hf_hub_download;import shutil
shutil.copy(hf_hub_download('anon8231489123/ShareGPT_Vicuna_unfiltered','ShareGPT_V3_unfiltered_cleaned_split.json',repo_type='dataset'),'$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json')" >/dev/null 2>&1 || log SHAREGPT_FAIL
log "PROVISION_DONE sgl=$($P -c 'import sgl_kernel' 2>/dev/null && echo 1) fi=$($P -c 'import flashinfer' 2>/dev/null && echo 1) llama=$(ls $M/llama70b/*.safetensors 2>/dev/null | wc -l) draft=$(ls $M/llama70b_eagle3 2>/dev/null | wc -l) sgpt=$([ -s workloads/ShareGPT_V3_unfiltered_cleaned_split.json ] && echo 1) tl=$(ls workloads/tracelab | wc -l)"
