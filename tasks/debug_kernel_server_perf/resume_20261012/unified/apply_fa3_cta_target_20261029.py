"""Expose the FA3 CTA target that decides the KV split (2026-10-29).

The planner splits the FA3 chunk until FA3 has `fa3_ctas` work items, and that target is hardcoded to the SM
count. On a short context this splits a small chunk several ways purely to avoid leaving SMs idle: a 4096-token
context yields about 112 unsplit items against 132 SMs, so the planner cuts the chunk to 1664 tokens and produces
336 items.

That reasoning ignores the FA2 items that run CONCURRENTLY with FA3 under programmatic dependent launch. A step
with any meaningful number of decode or verify inputs already has 188 to 612 FA2 items, which fill the machine on
their own, so the FA3 split adds no parallelism -- and every split forces partials through the merge, which is why
the measured cost shows up as 12 to 56 merge groups and half a megabyte of partials.

Measured consequence (H200, 200 iterations, pre-grown workspace, heterogeneous two-mode captures): the unified
dispatcher runs at 0.61-0.83x of the v3 build it is meant to be a superset of, while both beat sglang's single
attention call by 2.0-7.0x. This patch does not change any default; it only makes the target settable with
UNIFIED_FA3_CTAS so the split's contribution can be measured.

  python apply_fa3_cta_target.py <ub dir>

Idempotent.
Claude, 2026-10-29.
"""
import sys

path = f"{sys.argv[1]}/unified_executor_20261024.py"
src = open(path).read()
if "UNIFIED_FA3_CTAS" in src:
    print("already")
    raise SystemExit(0)

OLD = "        self.fa3_ctas = self.num_sm\n"
assert src.count(OLD) == 1, "the fa3_ctas assignment moved; re-read the executor before patching"
NEW = (
    "        # FA3's CTA target decides the KV split: the planner splits the chunk until FA3 has this many work\n"
    "        # items, so a target of num_sm splits a short context several ways purely to avoid idle SMs. That\n"
    "        # ignores the FA2 items running CONCURRENTLY under PDL, which already fill the machine whenever the\n"
    "        # step carries a meaningful number of decode or verify inputs -- and every split forces partials\n"
    "        # through the merge. UNIFIED_FA3_CTAS exposes the target so the split's cost can be separated.\n"
    "        self.fa3_ctas = int(os.environ.get(\"UNIFIED_FA3_CTAS\", self.num_sm))\n"
)
src = src.replace(OLD, NEW, 1)
compile(src, path, "exec")
open(path, "w").write(src)
print("patched fa3 cta target")
