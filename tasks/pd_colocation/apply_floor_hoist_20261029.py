"""Hoist the late-prefill floor OUT of the `deadline > now` guard (2026-10-29).

The first floor patch added an `elif not meets:` branch, but placed it inside

    if deadline is not None and deadline > now:

A prefill whose deadline has ALREADY PASSED has `deadline <= now`, so that block is skipped entirely -- and under
overload that is the common case and exactly the one the floor exists to serve. Measured consequence: with the
floor "in place", `zero` stayed at 2255 of 5501 calls (41%), identical to before the fix, and SLO attainment was
unchanged at chat 40.9% against 41.1%.

This hoists an unconditional minimum out of the deadline guard: whenever the rule would admit less than a page,
the earliest-deadline prefill gets one page or 256 tokens, whatever its deadline state. The deadline-aware floor
above it is untouched, so a prefill that can still be rescued keeps getting the larger chunk that rescues it.

Also logs `latefloor=`, which the first patch computed but never printed -- which is why its failure had to be
inferred from `zero` rather than read directly.

  python apply_floor_hoist.py <tree>/sglang/srt
Idempotent; verifies with compile().
"""
import sys

path = f"{sys.argv[1]}/managers/scheduler.py"
src = open(path).read()
if "slo_chunk_floor_hoisted" in src:
    print("already marker=1")
    raise SystemExit(0)

# 1. unconditional minimum, placed after the whole ttft block
OLD = """        self.slo_chunk_capped = getattr(self, "slo_chunk_capped", 0) + int(out < chunk_size)"""
assert src.count(OLD) == 1, "the counter block moved"
NEW = """        # slo_chunk_floor_hoisted: the deadline-aware floor above lives inside `deadline > now`, so it never
        # fires for a prefill whose deadline has already passed -- the common case under overload and the one a
        # floor is for. Starving it is self-reinforcing. Grant a minimum regardless of deadline state; the
        # rescue path above still gives a larger chunk to anything that can still make its deadline.
        if cfg["ttft"] and (self.chunked_req is not None or self.waiting_queue):
            _min = max(int(self.page_size), 256)
            if out < _min:
                self.slo_chunk_late_floor = getattr(self, "slo_chunk_late_floor", 0) + 1
                out = _min
        self.slo_chunk_capped = getattr(self, "slo_chunk_capped", 0) + int(out < chunk_size)"""
src = src.replace(OLD, NEW, 1)

# 2. make the counter visible
OLD2 = '"SLO_CHUNK calls=%d capped=%d zero=%d floored=%d qcapped=%d fresh=%d life=%d nolife=%d "'
if OLD2 in src:
    src = src.replace(OLD2, '"SLO_CHUNK calls=%d capped=%d zero=%d floored=%d latefloor=%d qcapped=%d fresh=%d life=%d nolife=%d "', 1)
    OLD3 = """                        getattr(self, "slo_chunk_floored", 0), getattr(self, "slo_chunk_qcapped", 0),"""
    assert src.count(OLD3) == 1, "the log args moved"
    src = src.replace(OLD3, """                        getattr(self, "slo_chunk_floored", 0), getattr(self, "slo_chunk_late_floor", 0),
                        getattr(self, "slo_chunk_qcapped", 0),""", 1)
compile(src, path, "exec")
open(path, "w").write(src)
print(f"patched marker={src.count('slo_chunk_floor_hoisted')} logged={src.count('latefloor=')}")
