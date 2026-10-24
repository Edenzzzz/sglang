# B: fused draft-chain epilogue implementation

2026-10-14. Kernel work only; no EDF or cluster-policy experiment. The main A/B/S outline
and TODO states are unchanged. The later EDF/load-balancing/replication direction is recorded
in `crossflow_analysis_20261014.md`; agent concurrency replaces the fixed-rate assumption.

## Candidate

`python/sglang/srt/speculative/fused_chain_tree.py` implements one Triton launch for a topk=1
draft chain: bonus-token concatenation, positions, retrieve-index/next-token/sibling metadata,
all three tree-mask formats, and optional verify-cache-location gather/MRoPE positions.
FULL_MASK initialization is fused too, including the cached-prefix region when the consumer
reads it. When the prefix is explicitly unread, only the tree cells are written. There is no
attempt to put this work inside attention: its token inputs follow the draft LM head and its
positions/mask are inputs to the next verify forward.

The migrated worker already skips redundant top-k sorting for chains and has a QLEN_ONLY
chain builder, but FlashInfer's FULL_MASK path still uses separate concatenation, initialization
and tree construction. This candidate extends fusion to that path. It is opt-in through
`SGLANG_FUSED_CHAIN_TREE=1`; all other trees retain their existing path. Widths above32 and
branching trees are not handled by this new chain specialization.

The current pod also contains the older fused greedy verifier under
`SGLANG_FUSED_VERIFY_EPILOGUE`. Its earlier kernel tests are useful, but its pre-RoPE-fix
serving results were explicitly retracted. It is not a newly implemented component here.
The new run tests the chain alone, then both epilogues, with the corrected model settings.

## Execution and gates

- Existing H200 trial `095f01a3e1585eea`, PID745563, started08:29:50 +08:00.
- Private tree `/dev/shm/w/up_b1_chain_epilogue/python`, based on `up_racefix`, EDF disabled.
- Root `/dev/shm/w/b1_chain_epilogue_20261014`; parent log has the same name plus `.log`.
- Overlay SHA256 `b940f190445f3b8dc146daf2956a555d788c83c7f30131a0a03892e6c7f91c2c`.
- GPU tests compare every output to an independent reference and the actual upstream builder;
  cover widths1–32, empty/odd batches, strided inputs, 0–256K contexts, packed mask boundaries,
  cache gather, MRoPE, unread-prefix/tail preservation and graph replay with changed inputs.
- Then32 timing cases: batch1/8/32/48, context4K/64K/128K/256K, FULL_MASK prefix read/unread,
  width4, three CUDA-graph measurements per arm.
- Four serving gates: plain split baseline, split+chain, split+both, co-launch+both. CUDA
  graphs remain enabled. Positive candidate execution counters and zero errors/fallbacks are
  required. The inherited generation gate requires the first eight long-prompt tokens to
  match; it does not prove full generation equivalence. Exact metadata tests are separate.

Local Python compilation and configured Black24.10 formatting pass; local GPU tests are
skipped because the machine has no CUDA device. GPU/serving results are pending. This is
an actual B implementation under test, not a closure or a performance claim.

## First GPU results and kernel revision

The first attempt stopped at08:30:06: five tests passed, but the upstream-reference test
supplied int32 vectors to an upstream builder requiring int64. The candidate accepts both;
the comparison now converts only the reference inputs. No candidate kernel change was
needed for that test correction. Patch archive SHA256:
`8c33b49cc6a206a0eb81e2d8ed7fb27e32dbc4500be3d1de7e2ba590110d3ea5`.

V2 PID746265 started08:32:09; all six GPU tests passed at08:32:20 and all32 timings completed
at08:33:05. Unread-prefix construction improves roughly6.7–8.4us to1.1–1.5us. However, the
first FULL_MASK fill implementation is too slow: batch48/context256K takes252.8us versus
26.5us upstream. It is not promoted. Its per-element dynamic division is unnecessary.

The revised kernel writes prefix spans and tree cells into disjoint regions in the same
launch, removing those divisions and using1024-byte mask tiles. Seven GPU tests now also
check that the fused path really launches exactly one kernel. V3 is queued as PID749369,
waiting for V2's terminal success, then using its own `up_b1_chain_epilogue_v3` tree and root.
V3 overlay SHA256 `bd710603cd552ae536f223020ed2434994a9f24f20f539374a785d8c2f7e52bd`.

V2's split baseline and split+chain serving gates are complete, with801 logged chain calls
in the latter and no fallbacks/tracebacks. Minimum common generation prefixes are1 short/
11 long: the inherited eight-long-token gate passes, but this is not full-output equality.
The both-epilogue split and co-launch gates remain in progress. No serving speedup is claimed.

## Optimized kernel result

V2 and V3 completed all four serving gates, at08:39:55 and08:45:48 respectively. V3 removes
the divisions but still loses on large masks. V4 adds aligned four-byte prefix stores with
separate bounded head/tail stores; prefix and tree writers remain disjoint within one launch.
This preserves byte values and avoids bulk writes through byte-valued pointers.

V4 PID756098 started08:45:57. All seven GPU tests passed, including the one-kernel profiler
check, at08:46:21. All32 timing cases completed08:46:54 and favor the new kernel: median
graph-time reductions are24.1–83.7%. Selected FULL_MASK, prefix-read results:

| Batch | Context | Existing builder, us | Fused chain, us | Reduction |
|---:|---:|---:|---:|---:|
| 1 | 64K | 7.770 | 1.758 | 77.4% |
| 8 | 128K | 9.438 | 2.587 | 72.6% |
| 32 | 256K | 19.767 | 13.048 | 34.0% |
| 48 | 256K | 26.313 | 19.964 | 24.1% |

Full measurements: `b1_chain_epilogue_v4_timing_20261014.json`, three observations per arm.
These gains apply to this epilogue, not full-step time or TTFT/ITL. V4's four serving gates
are still running. Its source overlay SHA256 is
`703e3c5eea65da056ed672c74b895380ad1b3db353f852a73300276d937b2fe9`.

The weak eight-token generation comparison is not the sole integration check. An additional
diagnostic is queued as PID760031 after V4: in real split/co-launch serving it recomputes
upstream tree outputs and cache locations on the exact same live inputs, requiring exact
equality for all consumed metadata. This diagnostic synchronizes and is never a performance
run. Overlay SHA256 `e4e6a55a4dd3b982b2ef96d6c5674a33d5f57b4403f0ff157fa586602e9fbbd4`.

## Completed integration checks

V4 completed all four inherited serving gates at08:51:45, with positive chain/attention
execution counters and no fallbacks/tracebacks. The shadow run completed08:54:18:112 split
steps and113 co-launch steps match the upstream metadata and independent cache lookup
exactly on the same live inputs, including12K-context requests. This is stronger evidence
for the integer epilogue than cross-run generation-prefix agreement. It does not establish
bitwise generation equality across independently scheduled runs or end-to-end speedup.

The candidate remains opt-in. B now has a newly implemented and measured draft epilogue,
plus fresh post-RoPE-fix integration gates for the older verifier. Kernel-serving performance
under the requested concurrency/context regimes remains to be measured; A's remaining
dispatch/custom-FA3 issues are not closed by this result. No EDF experiment is resumed.

Complete raw archive (including the initial failed reference-fixture test and all later
tests/timings/gates): `b1_chain_complete_20261014.tar.gz`,383999 bytes,89 manifest entries;
SHA256 `27d3425c0b04e523c15719d814f03a884a88a05c352b20fc039641077e691b1e`.
The archive and all89 file hashes are verified locally. `b1_chain_complete_audit_20261014.json`
records per-arm request/execution counts and the225 exact live-step comparisons.
