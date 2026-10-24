#!/bin/bash
# Llama-3.3-70B on the current H20 job (2f608c1b1a67cf09), 2026-10-24; run on each of pods 0-3, resumable. The pods
# already carry the current serving tree ($W/up, FA3 backend, EDF, prefill-role EDF patch) and clean upstream
# ($W/up_clean); this adds the 70B target (ModelScope), its trained EAGLE3 head (hf-mirror, config patched to the
# architecture / dtype sglang loads it by, as in mn_provision_70b_post_20261021.sh), and the native FA3 + FA2 kernel
# libraries built on the H200 (same torch 2.14.1+cu130 / CUDA 13.0 / Python 3.12.4 ABI, sm90a; tarball md5
# 9e50610002a89d4078adea1cbe2d8175). Claude.
set -u
W=/dev/shm/w; M=/opt/tiger/w; P=$W/venv/bin/python
log() { echo "[prov70 $(hostname | tail -c 6) $(date +%H:%M:%S)] $*"; }
cd $W
[ "$(md5sum < $W/perf/native_so_20261024.tgz | cut -d' ' -f1)" = 9e50610002a89d4078adea1cbe2d8175 ] || { log SO_MD5_BAD; exit 1; }
tar xzf $W/perf/native_so_20261024.tgz -C $M && log "native libs: $(ls $M/torch_ext_*_2026102*/*/*.so | wc -l)"
$P -c "import modelscope, huggingface_hub" 2>/dev/null || $P -m pip install -q modelscope 'huggingface_hub[cli]' >/dev/null 2>&1
export HF_ENDPOINT=https://hf-mirror.com
[ -f $M/llama70b/model-00030-of-00030.safetensors ] || $P -c "from modelscope import snapshot_download as s; s('LLM-Research/Llama-3.3-70B-Instruct', local_dir='$M/llama70b', ignore_file_pattern=['original/*'])" >/dev/null 2>&1 || log TARGET_FAIL
ln -sfn $M/llama70b $W/llama70b
[ -f $M/llama70b_eagle3/config.json ] || $P -c "from huggingface_hub import snapshot_download as s; s('yuhuili/EAGLE3-LLaMA3.3-Instruct-70B', local_dir='$M/llama70b_eagle3')" >/dev/null 2>&1 || log DRAFT_FAIL
ln -sfn $M/llama70b_eagle3 $W/llama70b_eagle3
if [ ! -f $W/llama70b_eagle3_sgl/config.json ]; then
  mkdir -p $W/llama70b_eagle3_sgl; for f in $W/llama70b_eagle3/*; do ln -sf $f $W/llama70b_eagle3_sgl/; done
  rm -f $W/llama70b_eagle3_sgl/config.json
  $P -c "import json;c=json.load(open('$W/llama70b_eagle3/config.json'));c['architectures']=['LlamaForCausalLMEagle3'];c['torch_dtype']='bfloat16';json.dump(c,open('$W/llama70b_eagle3_sgl/config.json','w'))"
fi
log "PROVISION70_DONE shards=$(ls $M/llama70b/*.safetensors 2>/dev/null | wc -l)/30 draft=$(ls $M/llama70b_eagle3 2>/dev/null | wc -l) so=$(ls $M/torch_ext_*_2026102*/*/*.so 2>/dev/null | wc -l)"
