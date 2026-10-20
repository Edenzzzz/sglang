#!/bin/bash
# H20 4-GPU-pod provisioning for the 32-GPU Qwen2.5-7B P:D ratio sweep (2026-10-21; run on every pod; resumable).
# Same stack as the 2026-10-16..20 8-GPU runs: upstream sglang 3fef129 + the serving tree as it stood on those pods
# (tree_20261020.tgz: the 32 files that differ from upstream, taken from pod 0 of bb854a10b8954223 on 2026-10-20, every
# in-place patch included) + flashinfer port07, torch 2.14.1. The tree is verified against tree_manifest_20261020.txt
# (md5 prefixes) before anything runs. Model: Qwen2.5-7B-Instruct (ModelScope) + the thoughtworks EAGLE3 head (hf-mirror).
# Claude.
set -u
W=/dev/shm/w; mkdir -p $W; cd $W
log() { echo "[prov $(hostname | tail -c 6) $(date +%H:%M:%S)] $*"; }
# Upstream 3fef129's python/ tree: from up_3fef129_python.tgz (`git archive 3fef129 python`, pushed to pod 0 and served to
# the others at $HUB) -- GitHub resets these pods' clones (2026-10-20, also through the ghfast proxy) -- else a clone.
if [ ! -d up/python ]; then
  [ -f $W/up_3fef129_python.tgz ] || [ -z "${HUB:-}" ] || curl -sf $HUB/up_3fef129_python.tgz -o $W/up_3fef129_python.tgz
  if [ -f $W/up_3fef129_python.tgz ]; then
    rm -rf up; mkdir -p up && tar xzf $W/up_3fef129_python.tgz -C up && log "upstream 3fef129 from archive"
  else
    for url in https://github.com/sgl-project/sglang.git https://ghfast.top/https://github.com/sgl-project/sglang.git; do
      rm -rf up; git clone --quiet --filter=blob:none $url up && break
    done
    git -C up checkout --quiet 3fef129ff8a32c32da5f6a0efd33485f6cc6233c || { log CLONE_FAIL; exit 1; }
  fi
fi
[ -d up/python/sglang ] || { log UPSTREAM_MISSING; exit 1; }
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
[ -f $M/qwen7b/model-00004-of-00004.safetensors ] || $P -c "from modelscope import snapshot_download as s; s('Qwen/Qwen2.5-7B-Instruct', local_dir='$M/qwen7b')" >/dev/null 2>&1 || log TARGET_FAIL
ln -sfn $M/qwen7b $W/qwen7b
[ -f $M/qwen7b_eagle3/model.safetensors ] || $P -c "from huggingface_hub import snapshot_download as s; s('thoughtworks/Qwen2.5-7B-Instruct-Eagle3', local_dir='$M/qwen7b_eagle3')" >/dev/null 2>&1 || log DRAFT_FAIL
ln -sfn $M/qwen7b_eagle3 $W/qwen7b_eagle3
if [ ! -f qwen7b_eagle3_sgl/config.json ]; then
  mkdir -p qwen7b_eagle3_sgl; for f in qwen7b_eagle3/*; do ln -sf $W/$f qwen7b_eagle3_sgl/; done; rm -f qwen7b_eagle3_sgl/config.json
  $P -c "import json;c=json.load(open('$W/qwen7b_eagle3/config.json'));c['architectures']=['LlamaForCausalLMEagle3'];json.dump(c,open('$W/qwen7b_eagle3_sgl/config.json','w'))"
fi
# PD / pool dependencies and the clean upstream tree the stock disagg baseline runs on (see mn_provision_70b_post).
[ -f /usr/include/openssl/sha.h ] || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q libssl-dev >/dev/null 2>&1 || log LIBSSL_FAIL
[ -d up_clean/python/sglang ] || { mkdir -p up_clean && tar xzf up_3fef129_python.tgz -C up_clean; }
$P -m pip install -q mooncake-transfer-engine sglang-router nvidia-cuda-runtime-cu12 >/dev/null 2>&1 || log DEPS_FAIL
mkdir -p workloads/tracelab
[ -f $W/tl.tgz ] && tar xzf $W/tl.tgz -C $W
[ -s workloads/ShareGPT_V3_unfiltered_cleaned_split.json ] || $P -c "
from huggingface_hub import hf_hub_download;import shutil
shutil.copy(hf_hub_download('anon8231489123/ShareGPT_Vicuna_unfiltered','ShareGPT_V3_unfiltered_cleaned_split.json',repo_type='dataset'),'$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json')" >/dev/null 2>&1 || log SHAREGPT_FAIL
log "PROVISION_DONE sgl=$($P -c 'import sgl_kernel' 2>/dev/null && echo 1) fi=$($P -c 'import flashinfer' 2>/dev/null && echo 1) qwen=$(ls $M/qwen7b/*.safetensors 2>/dev/null | wc -l) draft=$([ -f $M/qwen7b_eagle3/model.safetensors ] && echo 1) deps=$($P -c "import mooncake, sglang_router" 2>/dev/null && echo 1) sgpt=$([ -s workloads/ShareGPT_V3_unfiltered_cleaned_split.json ] && echo 1) tl=$(ls workloads/tracelab | wc -l)"
