"""STEPTRACE: count a 4-mode step's verify rows as decode rows (2026-10-27).

The 4-mode merge (ScheduleBatch.mix_with_verify) runs inside the speculative worker's forward_batch_generation, after
_steptrace_begin; at trace time the batch holds only the prefill rows, and batch.mixed_spec_running is the running
batch whose verify rows will join them. The tracer counts those rows from running.seq_lens_cpu, but skips them when
that host copy is None, which it is on the serving path. So every merged step was logged with nd = 0 (about 2,300 per
worker on the 16-H20 trace replay), and fit_4mode_step.py rightly refused to fit ("Mixed-speculative execution is
logged but STEPTRACE has no mixed rows").

Fix: without seq_lens_cpu, the running rows' KV comes from each request (prompt + output tokens). A first version of
this patch looked for ScheduleBatch.mixed_verify_rows, which is set only on the merged object inside the worker, and
so changed nothing; it is reverted here. Idempotent.
  python apply_steptrace_verify_rows_20261027.py <tree>/python/sglang/srt
Claude.
"""
import sys

p = f"{sys.argv[1]}/managers/scheduler.py"
s = open(p).read()
if "steptrace_running_from_reqs" in s:
    print("already")
    sys.exit(0)
v1_old = """        steptrace_verify_tail = len(extend_lens) - int(getattr(batch, "mixed_verify_rows", 0) or 0)
        for i, extend_len in enumerate(extend_lens):
            ext = int(extend_len)
            sl = int(seq_lens[i]) if i < len(seq_lens) else 0
            if ext > 1 and i < steptrace_verify_tail:"""
v1_new = """        for i, extend_len in enumerate(extend_lens):
            ext = int(extend_len)
            sl = int(seq_lens[i]) if i < len(seq_lens) else 0
            if ext > 1:"""
if v1_old in s:
    s = s.replace(v1_old, v1_new, 1)
old = """        running = getattr(batch, "mixed_spec_running", None)
        if running is not None and running.seq_lens_cpu is not None:
            run_lens = running.seq_lens_cpu.tolist()
"""
new = """        running = getattr(batch, "mixed_spec_running", None)
        if running is not None:
            # steptrace_running_from_reqs: the serving path keeps no host seq_lens for the running batch.
            run_lens = (running.seq_lens_cpu.tolist() if running.seq_lens_cpu is not None
                        else [len(r.origin_input_ids) + len(r.output_ids) for r in running.reqs])
"""
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, new, 1))
print("patched")
