#!/bin/bash
# Full pod restore after the trial expired and tmpfs /dev/shm was lost.
# Order is forced, not stylistic -- see the recorded recipe: a venv that has held an older torch
# is unrecoverable, so torch 2.14.1 goes in FIRST and the rest follows its nvidia-* wheels.
# Resumable: each stage skips if its marker exists.
set -u
W=/dev/shm/w
mkdir -p $W
log() { echo "[restore $(date +%H:%M:%S)] $*"; }

# ---- 1. upstream sglang clone (git clone is far faster than base64 through the webshell) ----
if [ ! -d $W/up/.git ]; then
  log "cloning upstream sglang"
  git clone --quiet https://github.com/sgl-project/sglang.git $W/up || { log "CLONE FAIL"; exit 1; }
fi
log "up at $(git -C $W/up rev-parse --short HEAD)"

# ---- 2. venv: torch first, then upstream's own package ----
if [ ! -x $W/venv/bin/python ]; then
  log "creating venv"
  python3 -m venv $W/venv || exit 1
fi
P=$W/venv/bin/python
if ! $P -c "import torch" 2>/dev/null; then
  log "installing torch==2.14.1 (brings its own consistent nvidia-* wheels)"
  $P -m pip install -q --upgrade pip setuptools wheel >/dev/null 2>&1
  $P -m pip install -q torch==2.14.1 || { log "TORCH INSTALL FAIL"; exit 1; }
fi
log "torch $($P -c 'import torch;print(torch.__version__)' 2>/dev/null)"
if ! $P -c "import sgl_kernel" 2>/dev/null; then
  log "installing wheel_stub then sglang python package (isolation OFF)"
  $P -m pip install -q wheel_stub >/dev/null 2>&1
  $P -m pip install -q --no-build-isolation $W/up/python || log "sglang install returned nonzero"
fi
# `import sgl_kernel` is the real check: it loads the native .so and fails when the torch pin is wrong.
$P -c "import sgl_kernel; print('sgl_kernel OK')" 2>&1 | tail -1
if ! $P -c "import flashinfer" 2>/dev/null; then
  log "installing flashinfer-python==0.7.0.post1"
  $P -m pip install -q flashinfer-python==0.7.0.post1 || log "flashinfer install nonzero"
fi
$P -c "import flashinfer; print('flashinfer', flashinfer.__version__)" 2>&1 | tail -1

# ---- 3. models ----
if [ ! -f $W/qwen7b/config.json ]; then
  log "downloading Qwen/Qwen2.5-7B-Instruct (~15GB)"
  $P -m pip install -q 'huggingface_hub[cli]' >/dev/null 2>&1
  $W/venv/bin/hf download Qwen/Qwen2.5-7B-Instruct --local-dir $W/qwen7b >/dev/null 2>&1 \
    || $W/venv/bin/huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $W/qwen7b >/dev/null 2>&1 \
    || { log "TARGET DOWNLOAD FAIL"; exit 1; }
fi
log "target model: $(ls $W/qwen7b/*.safetensors 2>/dev/null | wc -l) shards"
if [ ! -f $W/qwen7b_eagle3/config.json ]; then
  log "downloading thoughtworks/Qwen2.5-7B-Instruct-Eagle3"
  $W/venv/bin/hf download thoughtworks/Qwen2.5-7B-Instruct-Eagle3 --local-dir $W/qwen7b_eagle3 >/dev/null 2>&1 \
    || $W/venv/bin/huggingface-cli download thoughtworks/Qwen2.5-7B-Instruct-Eagle3 --local-dir $W/qwen7b_eagle3 >/dev/null 2>&1 \
    || { log "DRAFT DOWNLOAD FAIL"; exit 1; }
fi
# sglang needs architectures=LlamaForCausalLMEagle3; the published config says LlamaForCausalLM.
# Weights are symlinked so the copy costs nothing.
if [ ! -f $W/qwen7b_eagle3_sgl/config.json ]; then
  log "building eagle3_sgl copy (config architectures patched, weights symlinked)"
  mkdir -p $W/qwen7b_eagle3_sgl
  for f in $W/qwen7b_eagle3/*; do ln -sf "$f" $W/qwen7b_eagle3_sgl/ 2>/dev/null; done
  rm -f $W/qwen7b_eagle3_sgl/config.json
  $P - <<PY
import json
c = json.load(open("$W/qwen7b_eagle3/config.json"))
c["architectures"] = ["LlamaForCausalLMEagle3"]
json.dump(c, open("$W/qwen7b_eagle3_sgl/config.json", "w"), indent=1)
print("eagle3 config architectures ->", c["architectures"])
PY
fi
log "RESTORE_CORE_DONE"
du -sh $W/* 2>/dev/null | tail -8
