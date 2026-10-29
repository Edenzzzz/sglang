"""STEPTRACE: record decode rows' KV length on decode-only steps -- rewritten for the DEPLOYED tracer (2026-10-29).

The first version of this patch anchored on a multi-line `_sl = (...)` block. The tracer on these trees has no such
block: it reads

    seq_lens = batch.seq_lens_cpu.tolist() if batch.seq_lens_cpu is not None else []

on one line. The patch therefore asserted "the seq_lens_cpu fallback moved" on every node -- correctly refusing to
write -- but the chain that called it printed DKV_DEPLOYED regardless and queued the traced arm anyway, so the arm
would have recorded dkv=0 on every decode step for the second time, which is the exact defect the trace exists to
measure.

The fix is the same shape the verify-rows patch already uses for the running batch: where the host copy is absent,
take each row's KV from the request itself. `len(origin_input_ids) + len(output_ids)` IS a decode row's KV length,
and it costs no device synchronisation, so the measured path is not perturbed.

  python apply_dkv_v2.py <tree>/python/sglang/srt      (or <tree>/sglang/srt)

Idempotent. Verifies with compile(), not ast.parse(), and prints the marker count so a caller can check rather
than trust the exit code.
"""
import sys

path = f"{sys.argv[1]}/managers/scheduler.py"
src = open(path).read()
if "steptrace_decode_kv_from_reqs" in src:
    print("already marker=1")
    raise SystemExit(0)

OLD = ("        seq_lens = batch.seq_lens_cpu.tolist() if batch.seq_lens_cpu is not None else []\n")
assert src.count(OLD) == 1, f"anchor not found exactly once (count={src.count(OLD)}); re-read the tracer"
NEW = (
    "        # steptrace_decode_kv_from_reqs: seq_lens_cpu is None on the decode path, and an empty fallback made\n"
    "        # every row's KV length read as 0, so dkv was 0 on all decode steps -- the dominant term in decode\n"
    "        # attention cost, absent from the model's input. Fall back to the requests, as the running-batch\n"
    "        # branch below already does; no device synchronisation is added.\n"
    "        seq_lens = (batch.seq_lens_cpu.tolist() if batch.seq_lens_cpu is not None\n"
    "                    else [len(r.origin_input_ids) + len(r.output_ids) for r in batch.reqs])\n"
)
src = src.replace(OLD, NEW, 1)
compile(src, path, "exec")
open(path, "w").write(src)
print(f"patched marker={src.count('steptrace_decode_kv_from_reqs')}")
