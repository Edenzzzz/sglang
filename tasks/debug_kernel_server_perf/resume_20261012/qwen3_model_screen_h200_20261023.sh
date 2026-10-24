#!/bin/bash
set -euo pipefail
ROOT=/dev/shm/w/qwen3_model_screen_h200_20261023
echo $$ > "$ROOT/controller.pid"
trap 'status=$?; printf "%s\n" "$status" > "$ROOT/exit_status"' EXIT
test "$(cat /dev/shm/w/qwen3_target_download.exit)" = 0
test -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
export PYTHONPATH=$ROOT:/dev/shm/w/up/python
/dev/shm/w/venv/bin/python - "$ROOT" <<'PY'
import ast
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
target = Path('/dev/shm/w/qwen3_8b')
for entry in json.loads((root / 'target_manifest.json').read_text()):
    path = target / entry['name']
    if path.stat().st_size != entry['bytes'] or hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() != entry['sha256']:
        raise ValueError('Restored Qwen3 target differs from its manifest')
draft = Path('/dev/shm/w/qwen3_8b_eagle3')
expected = {
    'config.json': '360aa6861f74205c9b07dede8325b4638464d077c895f438e4ee20f0b2512dd7',
    'pytorch_model.bin': '048b9c4ada5fc26a7fe35086446f6013bb91ab9336c49ea38e519cffed43fe74',
}
for name, digest in expected.items():
    if hashlib.file_digest((draft / name).open('rb'), 'sha256').hexdigest() != digest:
        raise ValueError('Qwen3 trained draft differs from the captured model')
for phase in ('quality', 'serving'):
    destination = root / phase
    destination.mkdir()
    for source in [*root.glob('*.py'), *root.glob('*.sh')]:
        (destination / source.name).write_bytes(source.read_bytes())
source = Path('/dev/shm/w/direct_output_shadow_h200_20261023/srv_correctness.py').read_text()
for original, replacement in (
    ('MODEL = f"{W}/qwen7b"', 'MODEL = f"{W}/qwen3_8b"'),
    ('f"{W}/qwen7b_eagle3_sgl"', 'f"{W}/qwen3_8b_eagle3"'),
    ('"original_max_position_embeddings":32768', '"original_max_position_embeddings":40960'),
):
    if source.count(original) != 1:
        raise ValueError('Unexpected correctness-driver source anchor')
    source = source.replace(original, replacement, 1)
ast.parse(source)
(root / 'quality/srv_correctness.py').write_text(source)
(root / 'model_fingerprints.json').write_text(json.dumps(dict(target_manifest='target_manifest.json', draft_sha256=expected), indent=2) + '\n')
PY
export NATIVE_PDL_FA2_FP32_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_20261023/native_fa2_q32_fp32/native_fa2_q32_fp32.so
export NATIVE_PDL_FA2_FP32_DIRECT_LIBRARY=/opt/tiger/w/torch_ext_native_fp32_direct_20261023/native_fa2_q32_fp32_direct/native_fa2_q32_fp32_direct.so
set +e
env ROOT=$ROOT/quality SGLANG_NATIVE_PDL_SPLIT_POLICY=geometry_v3 SGLANG_NATIVE_PDL_FP32_PARTIALS=1 \
  SGLANG_NATIVE_PDL_DIRECT_BF16_OUTPUT=1 NATIVE_PDL_SHADOW_SERIAL_SEPARATE_CONVERSION=1 \
  NATIVE_PDL_ORIGINAL_ROW_CONTROL=0 NATIVE_PDL_BF16_PARTIAL_CONTROL=0 \
  bash "$ROOT/quality/fp32_adjudication_h200_20261023.sh" > "$ROOT/quality/runner.log" 2>&1
quality_status=$?
set -e
test "$quality_status" = 0 || test "$quality_status" = 3
/dev/shm/w/venv/bin/python - "$ROOT/quality" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
summary = json.loads((root / 'summary.json').read_text())
rows = [json.loads(line) for line in (root / 'shadow.jsonl').read_text().splitlines()]
assert rows and summary['all_fp32_repeats_exact'] and summary['all_native_state_restorations_exact']
assert all(row['native_serial_exact'] and row['serial_control'] == 'separate_fp32_conversion' and row['committed_sample_exact'] and row['committed_prefill_exact'] for row in rows)
print('QWEN3_NUMERICAL_CONTROLS_COMPLETE exact_fa3_emissions=', summary['same_input_emissions_equal'], flush=True)
PY
export TRACE_DATASET_PATH=$ROOT/tracelab_unique.jsonl.gz
/dev/shm/w/venv/bin/python "$ROOT/deduplicate_trace_rounds_20261023.py" /dev/shm/w/workloads/tracelab/syfi_full_min.jsonl.gz "$TRACE_DATASET_PATH" "$ROOT/trace_deduplication.json"
env ROOT=$ROOT/serving SERVING_PAIR_ARGS='--model-dir qwen3_8b --draft-dir qwen3_8b_eagle3 --split-policy geometry_v3 --direct-output --concurrency 8 --sessions 16 --chat-rate 8 --chat-requests 960' \
  bash "$ROOT/serving/native_serving_pair_h200_20261023.sh" > "$ROOT/serving/runner.log" 2>&1
echo QWEN3_MODEL_SCREEN_COMPLETE
