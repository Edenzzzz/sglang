"""Give the earliest-deadline prefill a floor even when it cannot meet its deadline (2026-10-29).

THE BUG. The SLO-budgeted chunk rule's TTFT floor is guarded on the head prefill still being able to finish by its
deadline:

    # A prefill that cannot meet its deadline at any chunk size is treated as late: no floor.
    if meets and floor > out:
        out = int(floor)

Under overload almost no prefill can, so `meets` is false, no floor is granted, and the chunk stays at whatever the
TPOT budget allowed -- which is ZERO on 41% of calls (measured: `zero=2674` of `calls=6501` at both the 2000 ms and
4000 ms caps). Prefill is admitted nothing on four steps in ten, and the condition is self-reinforcing: a prefill
denied service can never stop being late.

It is the same error as the demote bug fixed earlier the same day. "Has already missed its deadline" is grounds for
not letting a request BLOCK OTHERS -- which deadline ordering and demotion already handle -- and not grounds for
withholding service from it.

WHAT IT EXPLAINS. Every symptom of the arms measured before this fix, together: mean TPOT better by 11-28% (decode
steps run unobstructed), mean TTFT 3.5-4x worse and SLO attainment 19-33 points down (prefill starved), fewer agent
rounds admitted, and insensitivity to the tail cap across a sixteen-fold range -- because the cap was never the
binding constraint, the zero was.

THE FIX. When no chunk size lets the head prefill meet its deadline, still grant it a minimum chunk so that it
makes progress. The minimum is one page, or 256 tokens, whichever is larger -- the same starting floor the search
already uses -- so the cost to resident rows' TPOT is bounded by one small chunk per step rather than unbounded.
The meets-path is unchanged, so this can only raise the chunk, never lower it.

  python apply_late_prefill_floor.py <tree>/sglang/srt     (or <tree>/python/sglang/srt)

Idempotent; verifies with compile() and prints a marker count.
"""
import sys

path = f"{sys.argv[1]}/managers/scheduler.py"
src = open(path).read()
if "slo_chunk_late_floor" in src:
    print("already marker=1")
    raise SystemExit(0)

OLD = """                if meets and floor > out:
                    self.slo_chunk_floored = getattr(self, "slo_chunk_floored", 0) + 1
                    out = int(floor)"""
assert src.count(OLD) == 1, f"floor site not found exactly once (count={src.count(OLD)})"
NEW = """                if meets and floor > out:
                    self.slo_chunk_floored = getattr(self, "slo_chunk_floored", 0) + 1
                    out = int(floor)
                elif not meets:
                    # slo_chunk_late_floor: a prefill that cannot meet its deadline at any chunk size must still
                    # make progress. Withholding service from it is self-reinforcing -- it can never stop being
                    # late -- and starved prefill was returning a zero chunk on 41% of steps, which cost 19-33
                    # points of attainment while flattering TPOT. Being late is grounds for not blocking others,
                    # which ordering and demotion already handle, not for getting nothing.
                    _late = max(int(self.page_size), 256)
                    if _late > out:
                        self.slo_chunk_late_floor = getattr(self, "slo_chunk_late_floor", 0) + 1
                        out = int(_late)"""
src = src.replace(OLD, NEW, 1)
compile(src, path, "exec")
open(path, "w").write(src)
print(f"patched marker={src.count('slo_chunk_late_floor')}")
