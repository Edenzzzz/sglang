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
# ---- SERVING TREE -------------------------------------------------------------------------
# Added 2026-10-11. restore.sh rebuilt only the KERNEL harness, so after a re-fork mtp_edf.sh exits
# instantly with jsons=0 and no server ever launches -- a silent failure (exit 0) that looks like a
# clean arm. These are the pieces it needs beyond the models.
#
# The DRIVERS and the two python helpers are small and are expected to be pushed alongside this
# script; this section only checks for them and says what is missing, since it cannot fetch them.
for f in $W/slo_attain.py $W/tasks/perf_model/fit_mtp_step.py \
         $W/drv/srv_correctness.py $W/drv/mtp_edf.sh; do
  [ -f "$f" ] || log "SERVING MISSING (push it): $f"
done
mkdir -p $W/mtpedf $W/workloads/tracelab $W/tasks/perf_model

# ShareGPT is fetchable from HF, so fetch it rather than pushing ~600 MB through a 60 KB-chunked
# exec channel. mtp_edf.sh wants it at exactly this path.
SGPT=$W/workloads/ShareGPT_V3_unfiltered_cleaned_split.json
if [ ! -s "$SGPT" ]; then
  log "fetching ShareGPT (chat tenant workload)"
  # The PYTHON API, not `huggingface-cli download`: this huggingface_hub version renamed that CLI
  # to `hf`, so the old invocation prints help text and exits 0 having downloaded NOTHING -- a
  # silent failure that reads as a network problem. The API is stable across the rename.
  $P - <<'PY' 2>&1 | tail -2
from huggingface_hub import hf_hub_download
import shutil, os
try:
    p = hf_hub_download("anon8231489123/ShareGPT_Vicuna_unfiltered",
                        "ShareGPT_V3_unfiltered_cleaned_split.json", repo_type="dataset")
    dst = "/dev/shm/w/workloads/ShareGPT_V3_unfiltered_cleaned_split.json"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(p, dst)
    print("ShareGPT ->", dst, os.path.getsize(dst), "bytes")
except Exception as e:
    print("SHAREGPT FETCH FAIL -- chat-tenant arms cannot run:", e)
PY
fi

# The tracelab trace is INTERNAL and has no known public source. Without it the codex/tracelab
# tenant cannot be driven, so the two-tenant arms that produced the recorded -37.7% ITL are not
# reproducible on a fresh pod until someone supplies it.
TL=$W/workloads/tracelab/syfi_coding_trace.jsonl.gz
[ -s "$TL" ] || log "TRACELAB MISSING (internal, no public source): $TL -- codex-tenant arms blocked"

log "RESTORE_CORE_DONE"
du -sh $W/* 2>/dev/null | tail -8
