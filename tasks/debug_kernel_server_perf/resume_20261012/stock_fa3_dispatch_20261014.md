# A: isolate the legacy attention-dispatch guard

The nine-arm stock-primary pilot is valid for its configuration, but does not establish
the value of FA3 on the long agent-prefill rows. The serving driver leaves
`SGLANG_FUSED_DECODE_GUARD` unset, so `mixed_dispatch.py` uses its default1. That guard
only selects FA3 when `ceil(KV/128) <= decode_rows/2`; with at most48 running requests,
32–64K prefill rows cannot pass it. The correctness gates explicitly set the guard to0.
Positive whole-run split/stock call counters alone therefore missed an important difference
between the gate workload and the timed executor exposure.

The fresh80-step split profile confirms the relevant exclusion: between21:36:39 and
21:37:22, the server logs a two-sided-eligible nine-row batch routed wholly to FA2
(`max_kv_tiles=512`). Its existing parser reports21.291 seconds in BatchPrefillWithPagedKV
out of35.798 seconds summed GPU duration. This is an instrumented window, not a new serving
speed claim. Kineto occupancy estimates are not hardware measurements, and the parser's
broad attention-name regex also includes some normalization kernels. Both profiles completed
21:52:38; full-name kernel counts are being retained separately.

## Controlled change

This experiment changes only the existing per-row attention-executor guard, not request
ordering, admission, prefill chunk size, EDF or global routing. It does not bundle the
earlier A3b packed-query threshold/work-floor changes with guard removal. No default is
promoted before measurements. The private tree `up_stock_fa3_dispatch/python` adds only
low-frequency CPU routing counters; all controls and candidates use that same telemetry.
Counters report the actual in-server guard and long-context rows planned in two-sided batches.

Three CPU routing tests demonstrate the distinction: q512/64K plus46 q4/48K rows is wholly
FA2 with guard1 and selects only the prefill for FA3 with guard0; shallow prefill selection
and verify-only routing remain unchanged. Three audit/prompt tests and13 extracted dispatch
cases pass locally and on the existing H200. No attention arithmetic changed.

## Fresh validation and serving matrix

First, three real EAGLE3 gates use four49152-token ShareGPT-derived prompts alongside eight
2048-token decoding requests: plain split, compiler-matched native with same-input attention
reference, and stock PDL with that reference. Guard0/full132-SM mode, graphs and original
YaRN parameters are retained. The generation-prefix comparison is still weaker than full
generation equivalence; the attention-reference/fp32 checks remain separately required.

Then eight unprofiled arms use the identical48-program/96-round,24-active-program32–64K
cohort and2048 natural ShareGPT requests at8 requests/s:
SPLIT_G1_1 / SPLIT_G0_1 / NATIVE_G0 / STOCK_SERIAL112_G0 / STOCK_PDL112_G0 /
STOCK_PDL132_G0 / SPLIT_G0_2 / SPLIT_G1_2.

Two repeats of both split guards separate routing gains from executor gains. Compare stock
or native against the guard0 split, not just the legacy guarded control. Require complete
request/output-length fingerprints, program identities/concurrency, correct backend counters,
graphs, no errors/EDF, and positive long-context mixed-row routing only in guard0 arms.
Report both tenants' mean/p99 ITL, TTFT and throughput; an ITL tradeoff is not a general win.

Runner: `stock_fa3_dispatch_ab_20261014.sh`; audit: `audit_stock_dispatch_20261014.py`.
Overlay62597 bytes, SHA256
`13650e556f5045e6232e7921f238c0d4b07badc9e58b86e5837a47813b615d86`.
Remote root/log: `/dev/shm/w/stock_fa3_dispatch_ab_20261014[.log]`.
The workload retains the documented gap-free/synthetic-token trace limitations; it is not
the complete realistic workload/rate sweep. Main A/B/S structure and TODOs remain untouched.

## Profile exposure verified; long-context gate stops timing

Full-name GPU events confirm only112 stock FA3 calls totaling3.537 ms in the split profile,
and140 stock PDL calls totaling4.905 ms in its profile: just0.01377% of the latter's35.630 s
summed GPU duration. Thus the early mixed-traffic window exercised the new primary mostly
on tiny work, not the long-prefill load it was meant to accelerate. The two sampled windows
are not identical batches; these figures establish exposure, not a speed comparison.

Compact profile evidence is local: `stock_fa3_profile_compact_complete_20261014.tar.gz`,
3955545 bytes,21 verified file hashes, SHA256
`aafed8af0fc40f0bed6f8a8ba404cb1c1b193bed60341b74bc804bbbc81a9d5e`.
Independent local checks reproduce73166/73163 GPU events and the duration/call summaries.
Full CPU/GPU traces remain remote with their size and hash recorded in each kernel summary.

Controller911947 stopped21:59:37 before any timed arm. The plain49152-token gate completed;
the release-native gate completed12 requests without errors and passed same-input attention
checks, including fp32 adjudication, but failed the generation-prefix comparison: l0 agrees
on only6/96 tokens, while the other three long prompts agree on96/96. Short s1 also differs
on its first token. The existing eight-token long-prefix requirement was not weakened.
Per-layer tolerance passing does not override this serving-output failure.

`stock_fa3_dispatch_diagnostic_20261014.sh` now runs the stock-primary long-context gate and
a fresh plain-split repeat against the preserved original control. It records comparison
failures without skipping the repeat, but launches no performance runs. This distinguishes
native-primary-specific divergence from ordinary baseline-repeat instability before choosing
a fix or interpreting the kernel-dispatch performance opportunity.

The diagnostic completed22:12:04: stock PDL also fails l0 at5/96 tokens, while the fresh
plain split repeat passes with minimum long prefix75/96. Both candidate attention-reference
checks pass. The failing candidate outputs remain preserved; no timed arm was resumed.

A stricter PDL ordering diagnostic completed22:25:38. Six fixtures cross48K/128K contexts,
unsplit/split attention and112/all132 SMs. Each runs four graph replays with changing Q/K/V,
NaN-poisoned outputs/partials and a dependent output-projection GEMM. Original stock, serial
wrapper and PDL outputs are bit-identical in all cases; serial/PDL FA2 and projected outputs
are also bit-identical. This narrows the fault investigation but does not waive serving gates.
Sources: `stock_fa3_phase_equivalence_20261014.sh` and `stock_fa3_strict_phase_20261014.patch`.

An API-logprob diagnostic completed four apparent passing comparisons, but **is not a valid
mixed-serving gate**: requesting logprobs disables mixed steps in the current scheduler.
All four report zero mixed steps. Those passes are explicitly excluded from acceptance.

The replacement `stock_fa3_internal_logits_20261014.sh` uses a new diagnostic-only tree and
captures top-five logits inside the existing mixed target forward, without requesting API
logprobs or changing request flags. Stable request IDs, prompt hashes, absolute output
positions and draft-proposal history permit comparing only identical causal prefixes.
It runs split / stock serial / stock PDL / native / split repeat and requires positive mixed
steps, existing attention-reference checks and actual logit records. Its synchronization is
diagnostic-only and these runs must never supply performance values. Three helper tests pass
locally and on the trial; overlay31880 bytes, SHA256
`38bc891e4cfdf2b8c0d218e152f84faa12ba7873cdc2db9ebb779d217eb8fa10`.

The first internal logger failed on the real request's array-valued prompt; no model result
from that failed run is used. V2 serializes a list and adds an array-input test. Its overlay
is31951 bytes, SHA256 `5d27ee2e07218395cb0353cb20fe4748d9f9bbe446375b0ff65ded8278c19a86`.
All five v2 mixed-serving arms finish22:56:32. Stock serial/PDL have minimum long prefix53,
the split repeat passes, but native still fails l0 at token6. The synchronous instrumentation
changes timing; its stock passes do not erase the preserved uninstrumented failures.

`analyze_kernel_logits_20261014.py` discards rejected draft continuations and compares only
the first differing token after the same emitted prefix and prompt hash. Native l0 token6
is2038 in split versus5339 in native; both top-two logit gaps are0.125. Short s1 token0 is19
in split versus17 in every candidate: split's gap is0.125, whereas candidates have an exact
top-logit tie. Logged argmax matches the emitted token in each case. This establishes
small-margin/tie sensitivity for those differences, not general model equivalence, and does
not explain away every failure or authorize candidate performance claims. Three analyzer
tests cover rejected drafts, prompt mismatch and missing evidence.

The original eight-arm matrix remains stopped. A separate four-arm experiment,
`stock_fa3_dispatch_split_20261014.sh`, uses only the plain split that passes the48K gate
and repeat: guard1 / guard0 / guard0 / guard1, unchanged workload and server policy.
It isolates the legacy row-dispatch guard without relying on the failed native/stock paths.
There are no layer-reference or logit diagnostics in its timed runs; both controls and
candidates keep the same low-frequency routing counters. This is not promotion of PDL or
closure of A. Root/log: `/dev/shm/w/stock_fa3_dispatch_split_20261014[.log]`.

The long-gate and phase-equivalence archive is local:
`stock_fa3_long_gate_phase_20261014.tar.gz`,2424692 bytes,49 checked hashes, SHA256
`2c4bf28cab89e7345f52ade7f63ca9e90dcacc8328f56f643b6faced23c3e274`.
Local audit confirms all24 exact phase replays and preserves the invalid zero-mixed API-logprob
runs separately from the accepted ordering tests and failed generation gates.

Internal-logit v2 raw evidence is local: `stock_fa3_internal_logits_v2_complete_20261014.tar.gz`,
196596 bytes,29 verified hashes, SHA256
`1cbd5d1fcea49077f519280583ab96a2f1da4775db3bae33df2048f09d23a5a0`.
Independent local analysis reproduces the complete remote report exactly; the result is
`stock_fa3_identical_prefix_logits_20261014.json`.

## Role isolation and dispatch repeats

`stock_fa3_role_control_20261014.sh` waits for the four timed plain-split arms, then runs
six diagnostic-only49152-token serving gates: split control, original stock-primary/native
FA2, public FA3/native FA2, stock-primary/public FA2, both public roles inside the candidate
dispatcher, and split repeat. Replacing FA3 suppresses its custom combine; replacing FA2
uses the already-planned public wrapper. Controls never masquerade as optimized kernels.
The purpose is to separate primary arithmetic, dependent arithmetic and candidate plumbing,
not to weaken the unchanged generation or same-input attention requirements.

The private `up_stock_fa3_role_control/python` tree has a checked opt-in selector;
default `none` preserves behavior.37 extracted dispatch cases cover each role combination,
contiguous/scattered row layouts and split/unsplit completion ordering. Tests pass locally
and on the trial. Overlay50600 bytes, SHA256
`aff025b8585d355e6e5cc3cb8d69fadb7bf856ff37c797b10eff009b49266482`.
No diagnostics alter the still-running timed plain-split tree.

At23:15, the first guard1/guard0 pair has complete request/execution audits. Guard0 exposes
185 cumulative32K+ mixed prefill rows versus zero with guard1. It raises agent/chat output
throughput122.522/1367.088 to145.812/1570.236 tok/s and lowers TTFT55.502/64.547 to
40.791/43.003 seconds. Agent mean ITL regresses25.956 to27.239 ms; chat improves24.528 to
21.533 ms. Agent strict acceptance also differs0.815/0.875. These are provisional one-pair
measurements, not a promotion or proof of the all-metric goal; the reverse repeats continue.

## Four split-dispatch arms complete

The reverse repeats finish23:24:43. All four arms complete96 agent rounds/2048 chat requests,
preserve exact prompt/output-length fingerprints and program identities, reach24 active
programs and pass backend/graph/error checks. Guard0 reports185/187 long mixed prefill rows;
guard1 reports zero in both repeats. Mean-of-two-run changes for guard0 against guard1:

| Tenant | Output tok/s | Mean TTFT | Mean ITL | Mean of P99 ITL |
|---|---:|---:|---:|---:|
| Agent | +16.73% | -22.26% | -5.25% | -22.93% |
| Chat | +13.80% | -30.00% | -10.41% | -28.20% |

P99 values are means of two run-level percentiles, not pooled percentiles. First-pair agent
mean ITL regresses, whereas the second improves; acceptance and batch composition vary.
There is no general30% mean-ITL improvement, nor a complete context/concurrency/rate sweep.
This supports retaining guard0 as a measured kernel-dispatch candidate, not silently changing
the default or promoting the unresolved custom/native paths. All four audits are in
`stock_fa3_dispatch_split_progress_20261014.json`.

Role-isolation serving controller939307 starts23:24:55 in its separate tree. The next queued
diagnostic, `stock_fa3_public_roles_20261014.sh`, compares each role against its actual public
serving operator on identical Q/K/V, including graph input changes and112/132 SMs. It retains
the exact serial-versus-PDL checks and reports bit differences plus per-head relative errors,
without treating different tiling/reduction order as automatic proof of a fault.

The four-arm split archive is local: `stock_fa3_dispatch_split_complete_20261014.tar.gz`,
719857 bytes,32 verified hashes, SHA256
`e7fee1901565313d729fcbc7cdacb01bce5f35845a8df18246e21450526d376e`.
Independent local comparison confirms identical request keys, requested and actual output
lengths in all four arms; compact cohort records retain the full client-result hashes.

## FA3 primary isolated

The six role-isolation gates finish23:37:15. Original stock-primary/native FA2 fails l0 at
5/96 tokens, and replacing only FA2 still fails at5/96. Replacing only FA3 with the public
serving operator passes at53/96, as does replacing both roles. The split repeat passes at
96/96. Short s1's first-token mismatch also disappears when replacing FA3, not FA2. All
candidate paths pass the same-input attention check; the controls exercise their declared
role replacements. This localizes the observed generation difference to the FA3 primary,
not PDL launch ordering or the native FA2 bridge, for this fixture.

The six public-role kernel cases finish23:38:22. Across24 changing-input graph replays,
native FA2 is bit-identical to the public FA2 wrapper, and PDL/serial/original FA3 phases
remain mutually bit-identical. At q512/KV49152 with one split, the custom stock FA3 is also
bit-identical to the public FA3 operator. The deliberately split fixtures differ by up to
0.00423 per-head relative error. This points to split/reduction choices rather than an
across-the-board mainloop mismatch. Serving logs show the old planner chooses2 splits in
116 sampled records, versus1 in only2 and4 in1.

## Single-split policy and kernel specialization under test

`stock_fa3_unsplit_20261014.sh` first rebuilds and tests an implemented upstream-FA3
`Split=false` specialization, selected only when the host proves `splits == 1`. Split plans
retain the existing `Split=true` primary and deferred combination; the ABI and caller-owned
buffers are unchanged. Local CUDA compilation succeeds. This is distinct from the earlier
custom-mainloop no-split experiment. GPU correctness/performance is not assumed from the
local build.

A separate opt-in planner override forces one split for the new49152-token serving gates.
It logs planned versus selected counts; it is a controlled numerical/performance hypothesis,
not a universal dispatch rule. Five gates compare fresh split control, the existing generic
primary at one split, specialized primary with same-input attention checks, specialized
primary without reference synchronization, and split repeat. The generic one-split arm
separates policy from code specialization. No timed serving arms follow automatically.

Private sources: `/dev/shm/stock_fa3_unsplit_work_20261014`; remote private tree
`up_stock_fa3_unsplit/python`. Overlay310372 bytes, SHA256
`d72f663dee3b64f6277abc206f7638ea312afed2a28ee353ddfa5ac6dd60a2a0`.
Runner955298 starts23:50:10; no other GPU job or successor is active.

Remote compilation, all three GPU unit tests and both strict phase/public-role fixtures pass.
The first generic one-split serving arm completes12 requests with no server errors and
passes the same-input checks, but controller955298 stops on a driver-label collision:
`FUSED_UNSPLIT` was incorrectly required to emit the plain split counter as well as its
correct fused counter. The corrected private audit checks fused before split and additionally
requires12 requests and positive mixed-step execution; four CPU tests pass. It does not
relax the token comparison or numerical tolerance.

On2026-10-15 at00:01, re-audit of the preserved generic arm and its unchanged generation
comparison both pass. This isolates the one-split policy from the new kernel specialization:
the existing generic primary can pass the failed48K serving fixture when its reduction choice
matches the unsplit computation. PID959184 runs the remaining specialized/reference,
specialized/no-reference-sync and split-repeat gates. No serving speed claim follows from
these correctness results. Gate-fix overlay7100 bytes, SHA256
`23d7c65cc1a28a243ad9f0ae03e320631178ea99061fcf857affbd8e9dcd6e23`.

Completed role-isolation/public-role archive is local: `stock_fa3_roles_complete_20261014.tar.gz`,
223985 bytes,38 verified hashes, SHA256
`61a8dfedc6613d6b75f31f6fe5e49df5e7288940e318f9b6582535017d814d19`.
The independently checked48 role comparisons are in
`stock_fa3_public_roles_complete_audit_20261014.json`.
