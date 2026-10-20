"""Short prompts pass the per-row slack chunk cap (applied after apply_edf_itl_slo_chunk.py).

With deadline-ordered rows + SLO admission, the ":slack" chunk cap (the largest prefill chunk that keeps every running
row within its mean-TPOT SLO) cut chat TPOT misses from ~30% to 3.8% at chat 8 req/s, but it capped 86% of steps to
no prefill at all, chat prefills included, and chat TTFT misses rose to 57% (H200, 2026-10-21). A chat prefill is a
few hundred tokens with no prefix: its step cost is small next to a deep-context agent chunk, which is what pushes rows
past 20 ms. So when the request the cap sizes (the earliest-deadline one; EDF chunk preemption parks an in-progress
agent chunk for earlier-deadline chats) is shorter than the long-prompt class, the chunk is at least the waiting short
prompts' uncached tokens, up to SGLANG_ITL_SHORT_FLOOR (default 1024). Long prompts stay under the cap.
  python apply_itl_slo_short_floor.py <tree>/python/sglang/srt
Claude, 2026-10-21.
"""
import sys

p = f"{sys.argv[1]}/managers/scheduler.py"
s = open(p).read()
assert "SGLANG_ITL_SHORT_FLOOR" not in s, "already applied"
old = """        rows = len(running_batch.reqs) if self.is_mixed_chunk else 0
        out = int(min(chunk_size - rows, c))
        if out < max(1, self.page_size):
            # Less than one page fits beside the running decodes: no prefill this step.
            out = 0
"""
assert s.count(old) == 1
new = """        rows = len(running_batch.reqs) if self.is_mixed_chunk else 0
        cls = [t[3:] for t in spec.split(":") if t.startswith("cls")]
        long_tok = float(cls[0].split("_")[0]) if cls else float("inf")
        if len(r.origin_input_ids) < long_tok:
            # A short prompt is cheap beside the deep-context chunks the cap guards against: let the waiting short
            # prompts through (EDF serves them first), up to the floor.
            short = sum(max(0, len(w.origin_input_ids) - getattr(w, "_edf_cached_len", 0))
                        for w in self.waiting_queue if len(w.origin_input_ids) < long_tok)
            c = max(c, min(short, int(os.environ.get("SGLANG_ITL_SHORT_FLOOR", "1024"))))
        out = int(min(chunk_size - rows, c))
        if out < max(1, self.page_size):
            # Less than one page fits beside the running decodes: no prefill this step.
            out = 0
"""
s = s.replace(old, new)
open(p, "w").write(s)
print("applied", p)
