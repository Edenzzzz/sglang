#!/bin/bash
# H200 re-fork provisioning (job 8574c727fab6ffbd, 2026-10-24), resumable. Base recipe = mn_provision_h200.sh (2026-10-18):
# upstream sglang 3fef129 + the H20 serving tree's 32-file delta (up_overlay_h20_20261024.tgz, diffed against H20's
# clean upstream copy), torch 2.14.1, flashinfer 0.7.0.post1, Qwen2.5-7B + thoughtworks EAGLE3 from huggingface.co.
# Adds: port07 (flashinfer branch unified_dispatch_20261024), native v3 libraries built on the previous H200
# (native_so_20261024.tgz, md5 9e50610002a89d4078adea1cbe2d8175, torch 2.14.1+cu130 / py3.12 / sm90a), flash-attention
# e9515d5 + the skip-KV-load diagnostic patch (default off), v3's native work dir (base_loader), the unified and serving
# scripts, and the capture source tree. Claude.
set -u
W=/dev/shm/w; B=$W/bundle; mkdir -p $W /opt/tiger/w; cd $W
log() { echo "[prov $(date +%H:%M:%S)] $*"; }
if [ ! -d up/.git ]; then
  for a in 1 2 3; do rm -rf up; git clone --quiet https://github.com/sgl-project/sglang.git up && break; sleep 10; done
  git -C up checkout --quiet 3fef129ff8a32c32da5f6a0efd33485f6cc6233c || { log CLONE_FAIL; exit 1; }
fi
log "up $(git -C up rev-parse --short HEAD)"
if [ ! -x /opt/tiger/w/venv/bin/python ]; then rm -rf $W/venv; python3 -m venv /opt/tiger/w/venv; fi
[ -L $W/venv ] || { rm -rf $W/venv; ln -sfn /opt/tiger/w/venv $W/venv; }
P=$W/venv/bin/python
[ -f /usr/include/openssl/sha.h ] || sudo -n apt-get install -y libssl-dev >/dev/null 2>&1 || log "libssl-dev nonzero"
$P -c "import torch" 2>/dev/null || { $P -m pip install -q --upgrade pip setuptools wheel >/dev/null 2>&1; $P -m pip install -q torch==2.14.1 || { log TORCH_FAIL; exit 1; }; }
$P -c "import torch;print('TORCH',torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.device_count())" 2>&1 | tail -1
$P -c "import sgl_kernel" 2>/dev/null || { $P -m pip install -q wheel_stub >/dev/null 2>&1; $P -m pip install -q --no-build-isolation $W/up/python || log "sglang install nonzero"; }
$P -c "import sgl_kernel; print('sgl_kernel OK')" 2>&1 | tail -1
$P -c "import flashinfer" 2>/dev/null || $P -m pip install -q flashinfer-python==0.7.0.post1 || log "flashinfer nonzero"
# Our serving delta over upstream, then the fork trees.
tar xzf $B/up_overlay_h20_20261024.tgz -C $W/up/python && log "overlay applied scheduler $(md5sum < $W/up/python/sglang/srt/managers/scheduler.py | cut -c1-8)"
tar xzf $B/port07.tgz -C $W && ln -sfn $W/port07 $W/fi && ln -sfn $W/up/python $W/python
mkdir -p $W/hf; ln -sfn $W/venv $W/.venv
md5sum $B/native_so_20261024.tgz | grep -q 9e50610002a89d4078adea1cbe2d8175 && tar xzf $B/native_so_20261024.tgz -C /opt/tiger/w && log "native libs $(ls /opt/tiger/w/torch_ext_*_2026102*/*/*.so | wc -l)"
NK=$W/native_kernel_sources_20261023; mkdir -p $NK; tar xzf $B/native_work.tgz -C $NK
if [ ! -d $NK/flash-attention/hopper ]; then
  git clone --quiet --filter=blob:none --no-checkout https://github.com/Dao-AILab/flash-attention $NK/flash-attention && \
    git -C $NK/flash-attention checkout --quiet e9515d5dee6ade134a33d6020d38d01ef0596996 && \
    git -C $NK/flash-attention apply $B/fa3_skip_kv_load_flash_attention.patch && log "flash-attention e9515d5 + skip-kv patch"
fi
# Scripts: v3 executor / capture / bench (perf/rss, perf/tsc), unified (ub), serving drivers, pd_colocation.
tar xzf $B/scripts.tgz -C $B
mkdir -p $W/perf/rss $W/perf/tsc $W/ub $W/drivers/serving $W/pdc
cp $B/scripts/rss/* $W/perf/rss/; cp $B/scripts/rss/* $W/perf/tsc/; cp $B/scripts/rss/*.py $W/ub/; cp $B/scripts/ub/* $W/ub/
cp $B/scripts/drivers/* $W/drivers/serving/; cp $B/scripts/pdc/* $W/pdc/
rm -rf $W/ub/port07_unified; cp -a $W/port07 $W/ub/port07_unified
$P -m pip install -q 'huggingface_hub[cli]' aiohttp >/dev/null 2>&1
export HF_HUB_DISABLE_XET=1
[ -f qwen7b/model-00004-of-00004.safetensors ] || $P -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-7B-Instruct', local_dir='$W/qwen7b')" >/dev/null 2>&1 || log TARGET_FAIL
log "target shards $(ls qwen7b/*.safetensors 2>/dev/null | wc -l)"
[ -f qwen7b_eagle3/model.safetensors ] || $P -c "from huggingface_hub import snapshot_download as s; s('thoughtworks/Qwen2.5-7B-Instruct-Eagle3', local_dir='$W/qwen7b_eagle3')" >/dev/null 2>&1 || log DRAFT_FAIL
if [ ! -f qwen7b_eagle3_sgl/config.json ]; then
  mkdir -p qwen7b_eagle3_sgl; for f in qwen7b_eagle3/*; do ln -sf $W/$f qwen7b_eagle3_sgl/; done; rm -f qwen7b_eagle3_sgl/config.json
  $P -c "import json;c=json.load(open('$W/qwen7b_eagle3/config.json'));c['architectures']=['LlamaForCausalLMEagle3'];json.dump(c,open('$W/qwen7b_eagle3_sgl/config.json','w'))"
fi
mkdir -p workloads/tracelab; cp $B/tracelab.jsonl.gz workloads/tracelab/syfi_full_min.jsonl.gz
[ -s workloads/ShareGPT_V3_unfiltered_cleaned_split.json ] || $P -c "
from huggingface_hub import hf_hub_download;import shutil
shutil.copy(hf_hub_download('anon8231489123/ShareGPT_Vicuna_unfiltered','ShareGPT_V3_unfiltered_cleaned_split.json',repo_type='dataset'),'$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json')" >/dev/null 2>&1 || log SHAREGPT_FAIL
# Capture source tree (the serving tree the 2026-10-23 trace-sampled captures used): our tree + native PDL FA3 backend
# + demote-on-miss EDF, the serving / capture scripts, mn_bench.sh and the TraceLab set.
S=$W/qwen25_v3_common_screen_h200_20261023
if [ ! -d $S/python ]; then
  mkdir -p $S && cp -a $W/up/python $S/python
  $P $W/perf/rss/apply_native_pdl_fa3_backend_20261023.py $S/python/sglang/srt/layers/attention/flashattention_backend.py && log "native backend applied" || log NATIVE_APPLY_FAIL
  $P $W/perf/rss/apply_edf_late_demote_20261024.py $S && log "late demote applied" || log DEMOTE_APPLY_FAIL
fi
cp $W/perf/rss/*.py $S/; cp $W/pdc/mn_bench.sh $S/; cp $B/tracelab.jsonl.gz $S/tracelab_unique.jsonl.gz
log "PROVISION_DONE sgl=$($P -c 'import sgl_kernel' 2>/dev/null && echo 1) fi=$($P -c 'import flashinfer' 2>/dev/null && echo 1) qwen=$(ls qwen7b/*.safetensors 2>/dev/null | wc -l) draft=$([ -f qwen7b_eagle3/model.safetensors ] && echo 1) fa=$([ -d $NK/flash-attention/hopper ] && echo 1)"
