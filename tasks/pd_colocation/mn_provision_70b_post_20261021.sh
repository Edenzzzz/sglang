#!/bin/bash
# Second provisioning step for the 70B pods (after mn_provision_70b_20261021.sh): the clean upstream tree the stock PD
# disagg baseline runs on ($W/up_clean, the same 3fef129 archive, no overlay), and the PD / router dependencies
# (mooncake transfer engine, sglang-router, the CUDA 12 runtime mooncake links). Checks the EAGLE3 head's architecture
# name sglang loads it by. Claude, 2026-10-21.
set -u
W=/dev/shm/w; cd $W
P=$W/venv/bin/python
# HiCache's pool store JIT-builds a C++ hash extension that includes openssl/sha.h; these pods ship libssl3 without
# headers (every 70B worker crashed at startup, 2026-10-21).
[ -f /usr/include/openssl/sha.h ] || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q libssl-dev >/dev/null 2>&1 || echo "libssl-dev install failed"
[ -d up_clean/python/sglang ] || { mkdir -p up_clean && tar xzf up_3fef129_python.tgz -C up_clean; }
$P -m pip install -q mooncake-transfer-engine sglang-router nvidia-cuda-runtime-cu12 >/dev/null 2>&1 || echo "pip nonzero"
$P -c "import mooncake, sglang_router; print('deps ok')"
$P - <<'PY'
import json
c = json.load(open("/dev/shm/w/llama70b_eagle3/config.json"))
print("eagle3 arch", c.get("architectures"), "hidden", c.get("hidden_size"), "draft_vocab", c.get("draft_vocab_size"))
PY
# sglang loads an EAGLE3 head by the LlamaForCausalLMEagle3 architecture; the published config says LlamaForCausalLM
# (as the Qwen head's did), and its torch_dtype float16 made sglang load the draft in fp16 beside the bf16 target (a
# mixed-dtype matmul crash, 2026-10-21). Same weights, symlinked, with those two fields changed.
if [ ! -f llama70b_eagle3_sgl/config.json ]; then
  mkdir -p llama70b_eagle3_sgl; for f in llama70b_eagle3/*; do ln -sf $W/$f llama70b_eagle3_sgl/; done
  rm -f llama70b_eagle3_sgl/config.json
  $P -c "import json;c=json.load(open('$W/llama70b_eagle3/config.json'));c['architectures']=['LlamaForCausalLMEagle3'];c['torch_dtype']='bfloat16';json.dump(c,open('$W/llama70b_eagle3_sgl/config.json','w'))"
fi
echo POST_DONE
