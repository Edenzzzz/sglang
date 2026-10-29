"""Add the CUDA-graph-capture guard to the pdmux spec-metadata fix (2026-10-29).

The first version forced a plan on the resolved per-stream backend whenever pdmux was on. Planning copies plan data
host-to-device, and CUDA graph capture rejects an unpinned host source, so the draft model's capture died with:

    Exception: Capture cuda graph failed: Cannot copy between CPU and CUDA tensors during CUDA graph capture
    unless the CPU tensor is pinned.

raised from eagle_draft_cuda_graph_runner.py. The arm then reported "0/16 workers healthy", the same top-level
symptom as the two earlier pdmux failures and as the unrelated import fault before them -- which is the third time
that symptom has had a different cause, and the reason each one has to be read from a worker log.

Capture has its own planning path: the decode graph runner plans each per-stream backend as it captures. So the
forced plan is only needed on the eager forward, and must be skipped while capturing.
"""
import sys

path = f"{sys.argv[1]}/model_executor/runner/eager_runner.py"
src = open(path).read()
if "is_current_stream_capturing" in src:
    print("already")
    raise SystemExit(0)

OLD = "        if enable_pdmux and attn_backend is not model_runner.attn_backend:\n"
assert src.count(OLD) == 1, "the pdmux plan guard is absent or changed; apply apply_pdmux_spec_metadata first"
NEW = ("        # Not while capturing: planning copies plan data host-to-device, which capture rejects unless the\n"
       "        # source is pinned, and the graph runner plans each per-stream backend as it captures anyway.\n"
       "        if (enable_pdmux and attn_backend is not model_runner.attn_backend\n"
       "                and not torch.cuda.is_current_stream_capturing()):\n")
src = src.replace(OLD, NEW, 1)
compile(src, path, "exec")
open(path, "w").write(src)
print("guarded")
