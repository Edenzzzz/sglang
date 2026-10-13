# Page-128 serving integration for the TMA experiment

The existing split planner built token-granularity FA2 indices and indptrs,
passed page size 1, and reused last-page lengths of one. That is incompatible
with the page-128 cache view consumed by the native bridge. Its stock FA3 table
also divided every token ID without first sampling one entry per page.

The isolated adapter now samples physical page starts for both roles, builds
FA2 page-count indptrs and actual final-page lengths, and plans FA2 with the
configured page size. The stock split passes matching paged K/V views to its
FA2 wrapper; its ordinary full-batch reference retains the original token-wise
representation. Page-1 behavior is covered by regression tests. The native
primary already constructed page-count metadata and retains its event fence.

The adapter also honors the explicit FA3 tile setting (128 for these gates),
rejects a TMA build paired with page size 1, and logs actual native paged calls.
The correctness driver is based on the live trial's version, not the older
workspace copy; it only adds a validated `BENCH_PAGE_SIZE` CLI option and records
the requested size. Prompts, speculative parameters and output criteria are
unchanged.

Sixteen CPU tests pass: nine paged planning/dispatch/adjudication tests, three
driver CLI tests and the four existing native glue tests. Coverage includes
noncontiguous request subsets, reordered physical pages, exact/full and partial
last pages, crossing a page boundary on the next plan, cache-view strides,
page-1 regressions, reference metadata and detection of deliberately corrupted
attention output. These tests do not substitute for GPU serving validation.

`tma_paged_serving_gate_20261013.sh` is queued as PID 582560. It requires successful
completion of all nine TMA kernel smoke gates and all 48 timing cases before
taking the GPU lock. It verifies the tested kernel sources and creates a private
`up_tma_paged_serving` tree. Five real EAGLE3 q4, graphs-on arms run at page size
128: stock MTP, two-launch split, native gather, native TMA and native multicast.
Outputs are compared with both the preserved page-1 stock reference and the
fresh page-128 stock reference. Native arms require actual paged calls and
same-input attention checks; TMA/multicast arms require corresponding device
execution counters. No performance claims come from these instrumented gates.

Archive: `tma_paged_serving_overlay_20261013.tar.gz`, 58,115 bytes, SHA256
`6871d0f27f0b7b2cfeae1be48a61c035a917898f9f907adc07e5e09f74a6c0c5`.
The adjacent patch and CPU-test log preserve the implementation and validation.
The original serving trees and the main task document remain unchanged.

## Completed real-serving gates

All five arms completed at 23:21:08 +08:00 on 2026-10-13 with CUDA graphs and
zero fallbacks/tracebacks. Each native arm logs 3601 paged calls at the last
reported checkpoint and 129 layer0 attention checks, with maximum relative
error 0.012052. The multicast arm reports up to 188 actual multicast tiles on
CTA0; the unicast and gather controls report zero. Kernel source hashes remain
those of the completed 48-case experiment.

Every arm passes the unchanged long-prompt output criterion against the
preserved page-1 reference, and each split/native arm passes against the fresh
page-128 stock reference. Minimum long prefix is 11 for those arms. This is not
bitwise generation equivalence: short outputs can differ at the first token,
including page-128 stock versus page-1 stock. The full comparisons are preserved
rather than summarized as exact output agreement.

`tma_paged_serving_complete_20261013_audit.json` contains all five verified
server-log hashes, execution/attention counters, both sets of comparisons,
source manifests and successful runner markers. These instrumented gates prove
integration within their tested numerical/output criteria, not a serving speedup.
