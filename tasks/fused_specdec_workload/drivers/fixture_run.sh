#!/bin/bash
cd /dev/shm/w/fi
. /dev/shm/w/venv/bin/activate
export PYTHONPATH=/dev/shm/w/python:/dev/shm/w/fi HF_HOME=/dev/shm/w/hf FI_ROOT=/dev/shm/w/fi
export TORCH_EXTENSIONS_DIR=/dev/shm/w/tex_fix2
rm -rf $TORCH_EXTENSIONS_DIR
# The harness takes its include path from the JIT's generated config dir, so that
# dir has to exist before it runs: one BatchAttention.plan builds it.
python - <<'PY' 2>&1 | tail -2
import torch
from flashinfer.attention import BatchAttention
a = BatchAttention(torch.empty(64 << 20, dtype=torch.uint8, device="cuda"), "NHD")
a.plan(torch.tensor([0, 8], dtype=torch.int32), torch.tensor([0, 64], dtype=torch.int32),
       torch.arange(64, dtype=torch.int32, device="cuda"),
       torch.tensor([64], dtype=torch.int32), 28, 4, 128, 128, 1, causal=False,
       q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
print("warmup ok, generated config present")
PY
ls -d /home/tiger/.cache/flashinfer/*/generated/batch_attention_with* 2>/dev/null | wc -l
FIXTURE='qo:72,439,1 prefix:164,0,5468' UNIFIED=1 SPLITS=8 FA3_PLAN_CTAS=8 \
  FLASHINFER_SM_PARTITION=1 FLASHINFER_FA3_SM_SPLIT_CLUSTERS=8 \
  timeout 1800 python tests/test_fused_server_batch.py 2>&1 |
  grep -iE 'fixture|rel err|FAILURES|PASS|error|illegal|unwritten' | head -12
echo FIX2DONE
