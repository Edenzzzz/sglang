"""STEPTRACE: record the decode rows' KV length on decode-only steps (2026-10-29).

The bug. The tracer reads every row's KV length from `batch.seq_lens_cpu`, falling back to an EMPTY list when that
host copy is absent:

    _sl = batch.seq_lens_cpu.tolist() if getattr(batch, "seq_lens_cpu", None) is not None else []
    ...
    _s = int(_sl[_i]) if _i < len(_sl) else 0

`seq_lens_cpu` is None on the decode path -- the same absence the 2026-10-27 verify-rows patch documents for the
merged path -- so `_s` is 0 for every row and the accumulated `dkv` is 0. Counted over the alpha-3 replay traces:
**DECODE 73 770 steps, 0 with dkv > 0; EXTEND 14 162 steps, 10 189 with dkv > 0** (values up to 100 375).

Why it matters more than a missing column. `dkv` is the only input carrying how much KV the decode rows read, which
is the dominant term in decode attention cost. With it zeroed, the step model predicts a decode step from the row
COUNT alone, and its `dkv/1e3` and `dkv*T/1e3` coefficients are dead. That is what produced the decode-step
figures we had been reading as a modelling limit: 29.0% MAPE and a -13.0% signed bias that no scalar calibration
removed and that a nonlinear fit on the same features made worse (-19.9%). None of that was evidence about whether
TPOT is modelable; the control is that fitting the same correction on a RANDOM half rather than a chronological one
gives a held-out median bias of +0.1%, i.e. the model is unbiased in the median within one distribution.

The fix. Where the host copy is absent, take each row's KV from the request itself -- prompt plus output tokens,
which is what the verify-rows patch already does for merged steps -- so no device synchronisation is added on the
measured path. A decode row's KV length is exactly `len(origin_input_ids) + len(output_ids)`.

  python apply_steptrace_decode_kv_20261029.py <tree>/python/sglang/srt

Idempotent. After applying: re-trace one arm with SGLANG_STEP_TRACE=1, confirm DECODE lines now carry dkv > 0,
then refit (`fit_step_mlp_20261028.py`) and re-measure the decode-step SIGNED bias on a chronological split
(`tpot_bias.py`). Only a bias that survives THAT is evidence against the perf-model premise.
Claude, 2026-10-29.
"""
import re
import sys

path = f"{sys.argv[1]}/managers/scheduler.py"
source = open(path).read()
if "_steptrace_decode_kv" in source:
    print("already")
    raise SystemExit(0)

# The tracer block is inserted into a method body, so its indentation depends on the host file. Match it with a
# regex that captures the leading whitespace and re-applies it, rather than hard-coding a column -- a fixed-indent
# match silently fails on a tree whose nesting differs.
PAT = re.compile(
    r"(?P<i>[ \t]*)_sl = \(\n"
    r"\s*batch\.seq_lens_cpu\.tolist\(\)\n"
    r"\s*if getattr\(batch, \"seq_lens_cpu\", None\) is not None\n"
    r"\s*else \[\]\n"
    r"\s*\)")
m = PAT.search(source)
assert m, "the seq_lens_cpu fallback moved; re-read the tracer before patching"
i = m.group("i")
NEW = (
    f"{i}# _steptrace_decode_kv: seq_lens_cpu is None on the decode path, and the previous fallback was an empty\n"
    f"{i}# list, so every row's KV length read as 0 and the accumulated dkv was 0 on all decode steps. Fall back\n"
    f"{i}# to the requests themselves -- prompt plus output tokens is exactly a decode row's KV length -- which\n"
    f"{i}# needs no device synchronisation and so does not perturb the measured path.\n"
    f"{i}_sl = (\n"
    f"{i}    batch.seq_lens_cpu.tolist()\n"
    f'{i}    if getattr(batch, "seq_lens_cpu", None) is not None\n'
    f"{i}    else [\n"
    f'{i}        len(getattr(_q, "origin_input_ids", ()) or ()) + len(getattr(_q, "output_ids", ()) or ())\n'
    f"{i}        for _q in batch.reqs\n"
    f"{i}    ]\n"
    f"{i})")
source = source[:m.start()] + NEW + source[m.end():]
open(path, "w").write(source)
print("patched steptrace decode kv")
