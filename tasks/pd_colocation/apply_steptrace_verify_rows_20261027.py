"""STEPTRACE: count a 4-mode step's verify rows as decode rows (2026-10-27).

The tracer's docstring says the merged step's verify rows are decode rows, but its loop classified rows by extend
length only, so the 4-token verify rows at the tail of a merged batch (ScheduleBatch.mixed_verify_rows) were logged as
prefill requests: every merged step read np = chunk + verify rows, nd = 0, and fit_4mode_step.py rightly refused to
fit ("Mixed-speculative execution is logged but STEPTRACE has no mixed rows"). The tail rows now count as decode rows
with their KV. Idempotent.
  python apply_steptrace_verify_rows_20261027.py <tree>/python/sglang/srt
Claude.
"""
import sys

p = f"{sys.argv[1]}/managers/scheduler.py"
s = open(p).read()
if "steptrace_verify_tail" in s:
    print("already")
    sys.exit(0)
old = """        n_p = p_tok = p_prefix = n_d = d_kv = max_ext = 0
        for i, extend_len in enumerate(extend_lens):
            ext = int(extend_len)
            sl = int(seq_lens[i]) if i < len(seq_lens) else 0
            if ext > 1:"""
new = """        n_p = p_tok = p_prefix = n_d = d_kv = max_ext = 0
        steptrace_verify_tail = len(extend_lens) - int(getattr(batch, "mixed_verify_rows", 0) or 0)
        for i, extend_len in enumerate(extend_lens):
            ext = int(extend_len)
            sl = int(seq_lens[i]) if i < len(seq_lens) else 0
            if ext > 1 and i < steptrace_verify_tail:"""
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, new, 1))
print("patched")
