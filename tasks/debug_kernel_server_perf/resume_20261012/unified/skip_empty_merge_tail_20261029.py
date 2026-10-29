"""Skip the FA3 merge tail, and its grid-dependency sync, when there is nothing to merge (2026-10-29).

Measured motivation. On an agent-dominant step with a realistic chunk (4096 tokens over a 65 536-token prefix) the
dispatcher is 2.8-6.3% slower than sglang's single attention call, and the penalty is a FIXED 0.38-1.02 ms on a
13-16 ms step rather than a proportional one. The FA3 split partials are reduced BY THE FA2 KERNEL: in the static
path the FA2 kernel ends with

    if (!um.fa3_merge_items) {
      cudaGridDependencySynchronize();
      UnifiedMergeFA3Tail<HEAD_DIM_VO>(...);
    }

so every FA2 CTA waits on the whole FA3 grid before it can retire. On these captures the plan reports
`fa3_merge_reqs = 0` -- FA3 is not split, so there are no partials and the tail has nothing to do -- and the
synchronise is paid anyway. sglang's single kernel has no such barrier, which is exactly the shape of a fixed cost.

The guard adds `um.num_fa3_merge_reqs > 0` to that condition. When FA3 IS split the behaviour is unchanged, so the
correctness of the split path does not depend on this patch; when it is not, the FA2 launch no longer serialises
behind the FA3 grid for a reduction of nothing.

  python skip_empty_merge_tail.py <port07 dir>

Idempotent. This is an implementation-level optimization and is deliberately not described in the paper.
"""
import sys

path = f"{sys.argv[1]}/tests/unified_dispatch_ext.cu"
src = open(path).read()
if "skip_empty_merge_tail" in src:
    print("already")
    raise SystemExit(0)

OLD = """  if (!um.fa3_merge_items) {
    cudaGridDependencySynchronize();"""
assert src.count(OLD) == 1, f"merge-tail site not found exactly once (count={src.count(OLD)})"
NEW = """  // skip_empty_merge_tail: the tail reduces FA3's split partials, and the synchronise above it makes every FA2
  // CTA wait on the whole FA3 grid. When FA3 is not split there are no partials (num_fa3_merge_reqs == 0) and the
  // wait buys nothing -- measured as a fixed 0.38-1.02 ms on a 13-16 ms prefill-dominated step. Guarding on the
  // request count leaves the split path byte-for-byte unchanged.
  if (!um.fa3_merge_items && um.num_fa3_merge_reqs > 0) {
    cudaGridDependencySynchronize();"""
src = src.replace(OLD, NEW, 1)
open(path, "w").write(src)
print("patched skip_empty_merge_tail")
