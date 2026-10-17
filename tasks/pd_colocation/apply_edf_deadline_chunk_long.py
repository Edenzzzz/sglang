"""Refine apply_edf_deadline_chunk.py: only a late request LONGER than the small chunk triggers the large chunk.

First runs picked the large chunk almost every step: chat's 2 s TTFT deadlines are late against any backlog, but a
chat prompt fits in one small chunk, so a large chunk cannot help it and only lengthens the mixed step every decoding
request waits through. Now a request counts as late only if its uncached prompt exceeds the small chunk.
  python apply_edf_deadline_chunk_long.py <tree>/python/sglang/srt
"""
import sys

p = f"{sys.argv[1]}/managers/scheduler.py"
s = open(p).read()
old1 = '''            tokens += max(0, len(r.origin_input_ids) - r.prefix_len)
            late = r.priority is not None and r.priority < now + tokens / tok_s'''
new1 = '''            need = max(0, len(r.origin_input_ids) - r.prefix_len)
            tokens += need
            late = need > small and r.priority is not None and r.priority < now + tokens / tok_s'''
old2 = '''                tokens += max(0, len(r.origin_input_ids) - getattr(r, "_edf_cached_len", 0))
                if r.priority < now + tokens / tok_s:'''
new2 = '''                need = max(0, len(r.origin_input_ids) - getattr(r, "_edf_cached_len", 0))
                tokens += need
                if need > small and r.priority < now + tokens / tok_s:'''
if new1 in s and new2 in s:
    print("already")
else:
    assert s.count(old1) == 1 and s.count(old2) == 1
    s = s.replace(old1, new1, 1).replace(old2, new2, 1)
    open(p, "w").write(s)
    print("patched")
