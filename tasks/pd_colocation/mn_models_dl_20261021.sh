#!/bin/bash
# After mn_mtp_scale_20261021.sh finishes, fetch the other two models to every pod of job 2f608c1b1a67cf09 (2026-10-21):
# Qwen3-8B (ModelScope) + its trained EAGLE3 head (AngelSlim/Qwen3-8B_eagle3, hf-mirror) and GLM-4.5-Air (ModelScope,
# ZhipuAI/GLM-4.5-Air; its MTP layer ships in the checkpoint). Onto the root overlay, symlinked into /dev/shm/w.
# Claude.
set -u
export PATH=$HOME/.merlin-cli/bin:$PATH
until grep -q MN_MTP_SCALE_DONE /tmp/h20/mn_mtp_scale.log; do sleep 60; done
python3 - <<'PY'
import sys; sys.path.insert(0, "/opt/tiger/sglang/tasks/pd_colocation")
from px_multi import Cluster
c = Cluster("2f608c1b1a67cf09", "cn-seed", nodes=[f"worker-{i}" for i in range(8)])
dl = r'''cat > /dev/shm/w/dl_models.sh <<'X'
set -u
P=/dev/shm/w/venv/bin/python; M=/opt/tiger/w; export HF_ENDPOINT=https://hf-mirror.com
[ -f $M/qwen3_8b/model-00005-of-00005.safetensors ] || $P -c "from modelscope import snapshot_download as s; s('Qwen/Qwen3-8B', local_dir='$M/qwen3_8b')" || echo Q3_FAIL
[ -f $M/qwen3_8b_eagle3/config.json ] || $P -c "from huggingface_hub import snapshot_download as s; s('AngelSlim/Qwen3-8B_eagle3', local_dir='$M/qwen3_8b_eagle3')" || echo Q3H_FAIL
[ -f $M/glm45air/model.safetensors.index.json ] && [ $(ls $M/glm45air/*.safetensors 2>/dev/null | wc -l) -ge 47 ] || $P -c "from modelscope import snapshot_download as s; s('ZhipuAI/GLM-4.5-Air', local_dir='$M/glm45air')" || echo GLM_FAIL
for d in qwen3_8b qwen3_8b_eagle3 glm45air; do ln -sfn $M/$d /dev/shm/w/$d; done
echo "DL_DONE q3=$(ls $M/qwen3_8b/*.safetensors | wc -l) glm=$(ls $M/glm45air/*.safetensors | wc -l)"
X
(setsid nohup bash /dev/shm/w/dl_models.sh > /dev/shm/w/dl_models.log 2>&1 < /dev/null &); echo started'''
for i in range(8):
    print(i, c.run(i, dl)[1].strip()[-20:], flush=True)
PY
echo DL_LAUNCHED
