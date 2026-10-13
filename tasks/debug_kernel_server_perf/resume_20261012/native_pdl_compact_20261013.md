# Native-PDL compact primary experiment

This isolated v3-based variant removes the stock FA3 planning pass whose output
the unified FA3 scheduler does not consume. The serving backend calls an explicit
`fused_primary_compact` entry instead, avoiding the unused 256 MiB FA3 workspace,
legacy wrapper construction and planning, and legacy last-page metadata upload.
The native FA2 plan, FA3 split-KV reduction, attention arithmetic and SM budget
are unchanged. The old non-native path and original entry points remain intact.

Local CUDA compilation and operator registration pass. Four CPU glue tests pass,
covering prefix, suffix and scattered request partitions, uninterrupted primary
and secondary launch ordering, omission of the legacy planner, and event-fenced
reuse of pinned planning buffers. These checks do not establish GPU correctness
or a serving performance gain.

Remote validation is queued as PID 501745, started at 19:38:31 +08:00 on
2026-10-13, after the dedicated-entry experiment PID 493488. It uses private
`fi_native_pdl_compact` and `up_native_pdl_compact` trees and caches, not the
dedicated-entry candidate. Ten split-KV numerical/graph/control cases precede
the real four-mode serving output and attention gate. The latter requires an
actual `NATIVE_PDL_COMPACT_CALLS` counter. Final success requires both
`NATIVE_PDL_COMPACT_DONE` and `NATIVE_PDL_COMPACT_EXIT status=0`.

Artifacts are the adjacent patch, validation script, compile log and source
overlay archive. The archive is 78,969 bytes, SHA256
`44d7efd5a087977deca3cc7ce813aca56f2da23f8705a11a07aed097acb78b26`.
The candidate is not promoted. The main task document and its TODOs are unchanged.

## Completed validation

All ten split-KV cases passed numerical, graph-replay and unfused-control checks.
The real four-mode serving gate completed at 20:46:07 +08:00 on 2026-10-13 with
actual compact-primary calls (last logged counter 3401), no fallbacks and no
tracebacks. It logged 127 layer0 attention checks, maximum relative error
0.012052. The unchanged output criterion passes with minimum long prefix 11;
the other three long requests match all 96 tokens. Short-request generation is
not bitwise equivalent to the stock reference, so this is not a claim of exact
generation equivalence or a serving performance gain.

`native_pdl_compact_complete_20261013_audit.json` preserves the ten kernel audits,
verified raw-log hashes, execution counters, attention-check summary, server-log
hash, output comparison and successful runner markers. Combining this planner
with the separately measured dedicated FA3 entry requires its own validation
and performance comparison.
