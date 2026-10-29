"""Let PD multiplexing run with speculative decoding, so pdmux can be measured in our arm's configuration.

The defect. PD multiplexing holds one attention backend per SM group -- `decode_attn_backend_group` -- and the
eager decode forward resolves one of them, publishing it through `ForwardContext` so the model layers read it via
`get_attn_backend()`. Whether to PLAN that backend is decided by `ForwardBatch.needs_forward_metadata_init()`,
which is a property of the BATCH: a pre-planner calls `mark_forward_metadata_ready()` after planning and the
forward then skips planning. Speculative decoding is such a pre-planner.

Those two facts are incompatible. Readiness is tracked per batch, but metadata lives per backend OBJECT, and pdmux
has one per SM group plus the prefill one. The speculative worker plans the main or draft backend and marks the
batch ready; `_execute_decode` then resolves a per-stream decode backend that nobody planned and runs the model
under it. The first attention layer dereferences its `forward_metadata`, which is None:

    AttributeError: 'NoneType' object has no attribute 'decode_wrappers'      (flashinfer)
    AttributeError: 'NoneType' object has no attribute 'swa_out_cache_loc'    (fa3)

Both are the same fault. The backend only changes which field is touched first, which is why switching backends
did not help: every worker still died at launch, and the driver reported `0/16 workers healthy`.

The fix, and why it is sound rather than a workaround. Under pdmux, a per-stream decode backend is never the object
a pre-planner planned -- the speculative paths plan `model_runner.attn_backend` or their own draft backend, and
neither is in `decode_attn_backend_group`. So the batch's readiness mark carries no information about the backend
actually being used, and planning it cannot corrupt pre-planned multi-step draft metadata: that metadata is on a
different object and is not touched. We therefore plan whenever the resolved backend is not the one the mark could
refer to. Note this is strictly stronger than testing `forward_metadata is None`: a per-stream backend retains the
PREVIOUS step's plan, so from the second decode step onward the None test passes while the metadata is stale, which
would be silent corruption instead of a crash.

Outside pdmux the resolved backend IS `model_runner.attn_backend`, so the condition is unchanged and no other arm
is affected.

Cost: one plan call per decode step per stream. Non-speculative pdmux already pays exactly this, because without a
pre-planner `needs_forward_metadata_init()` is true every step.

  python apply_pdmux_spec_metadata.py <tree>/python/sglang/srt

Idempotent. This makes the BASELINE runnable in the same MTP configuration as our arm, which is the comparison the
publishability bar is judged on; without it pdmux can only be run without speculation, which would hold its
configuration fixed against ours and understate it. Claude, 2026-10-29.
"""
import sys

path = f"{sys.argv[1]}/model_executor/runner/eager_runner.py"
source = open(path).read()
if "_pdmux_spec_metadata" in source:
    print("already")
    raise SystemExit(0)

OLD = """        if forward_batch.needs_forward_metadata_init():
            if hasattr(model_runner.model, "prepare_forward_batch"):"""
assert source.count(OLD) == 1, "the decode planning site moved; re-read _execute_decode before patching"

NEW = """        # _pdmux_spec_metadata: readiness is marked on the BATCH by a pre-planner (speculative decoding), but
        # metadata lives on a backend OBJECT and pdmux has one per SM group. A pre-planner only ever plans the
        # main or draft backend, never a per-stream decode backend, so under pdmux the mark says nothing about
        # the backend this forward resolved -- and planning that backend cannot disturb pre-planned draft
        # metadata, which lives elsewhere. Testing the resolved object rather than `forward_metadata is None`
        # matters from the second step on, when a per-stream backend still holds the previous step's stale plan.
        _needs_plan = forward_batch.needs_forward_metadata_init()
        if enable_pdmux and attn_backend is not model_runner.attn_backend:
            _needs_plan = True
        if _needs_plan:
            if hasattr(model_runner.model, "prepare_forward_batch"):"""
source = source.replace(OLD, NEW, 1)
compile(source, path, "exec")  # never write a file the interpreter would reject
open(path, "w").write(source)
print("patched pdmux spec metadata")
