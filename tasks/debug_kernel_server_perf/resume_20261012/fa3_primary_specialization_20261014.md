# A: FA3 primary specialization experiments

These are kernel changes, not EDF or admission-policy changes. The main task outline and
TODO states remain unchanged. The baseline is the native-FA2 PDL path with a dedicated
FA3 entry, KV128, two stages,112 FA3 CTAs and round-robin paired work assignment.

## Compact argument interface: no device-code improvement

Implemented a separate FA3-only entry without the unused FA2 parameter blocks. Local CUDA
compilation succeeds. Both entries have168 registers, an8-byte stack and9304 static SASS
instructions. After normalizing kernel-parameter offsets, their instruction sequences are
identical; parameter storage falls from2340 to1380 bytes. This does not address the device
instruction/performance gap. It is not promoted or sent through another GPU performance sweep.

Source: `fa3_compact_abi_20261014.patch`; resources:
`fa3_compact_abi_resources_20261014.json`. The local build/SASS archive SHA256 is
`0b0e08fe656c04990a0eae69b28396af1703d170dfb55a4da2e599bb9d426765`.
Local builds use CUDA13.1/Torch2.9.1, not the pod's toolchain; they are not deployed binaries.

## Guarded no-split entry: implemented and GPU-tested, no promotion

`FUSED_FA3_NOSPLIT_SPECIALIZE=1` selects a separate FA3 entry only when the native primary's
plan proves `num_merge_rows == 0`. Its epilogue cannot take the partial-write paths. Plans
requiring KV splits retain the existing entry; the planner and its splitting decisions are
unchanged. The kernel argument interface, arithmetic, geometry, PDL release and store fences
are held fixed. Clustered/TMA and upstream-FA3 variants are not supported by this experiment.

Local compilation reduces static instructions9304→8656 while retaining168 registers and
an8-byte stack. The generic control's instruction sequence remains unchanged. This static
reduction is not itself a performance result.

The H200 run completed11:42:53 +08:00. All four eager/graph gates and20 timing cases pass
their fp32/replay/unfused checks. The q32 case uses14 KV splits and demonstrably retains the
generic path; unsplit cases demonstrably execute the new entry. Useful FA2 completion during
FA3 execution is verified on the larger-prefill gate. Two-repeat means:

| Fixture | Existing entry, ms | No-split selection, ms | Change |
|---|---:|---:|---:|
| prefill2048 +8 verify | 4.69630 | 4.70880 | +0.27% |
| append512 +46 verify | 2.38665 | 2.40470 | +0.76% |
| FA3-only8x1024 | 9.49710 | 9.39600 | -1.06% |
| split q32 +8 verify | 0.17060 | 0.17060 | 0.00% |
| prefill8192 +24 verify | 18.31990 | 18.34355 | +0.13% |

No general gain is established. Keep the existing entry. The8K-prefill case also remains
slower than the unfused control, so the larger-query FA3 gap is not closed.

Source/runner: `fa3_nosplit_20261014.patch`, `fa3_nosplit_20261014.sh`. Complete raw archive:
`fa3_nosplit_complete_20261014.tar.gz`,16118 bytes,48 verified file hashes, SHA256
`d645a497ab88fb9f38c75bf80d7bcedefb15ce32876a98391a0278c2ccce4aa1`.
Scalar/execution audits: `fa3_nosplit_complete_audit_20261014.json`.

The run took place after the completed `mix64_c24/BASE_2` serving audit, using the inherited
exclusive GPU lock. No timed serving run overlapped it. The B controller resumed normally;
its BASE_2 `ARM_DONE` wall-clock marker is delayed by this kernel experiment and must not
be used as that arm's benchmark duration. The temporary audit hook was removed afterward.
The earlier waiter PID797513 was cancelled, not left to launch a duplicate experiment.

## Static GQA specialization: compiled, not yet GPU-tested

The packed-query path performs runtime division by `group_size` when gathering Q, applying
causal bounds and storing O. A separate prototype specializes these operations for the
current28-query-head/4-KV-head ratio. `SGLANG_FUSED_FA3_STATIC_GQA=7` is compile-time opt-in;
zero retains the dynamic implementation. The host rejects a mismatched head ratio instead
of silently applying the specialization. This prototype starts from the original baseline,
not from the rejected no-split candidate.

Both local variants compile. Dynamic/static7 entries have9304/9160 static instructions,
21/16 `MUFU.RCP` instructions and8/0-byte stack frames; both retain168 registers. These are
compiled resource observations, not executed-spill or latency measurements. GPU correctness,
wrong-ratio rejection, graph replay and matched performance remain unproven.

Working tree: `/dev/shm/fa3_static_gqa_work_20261014`. Source:
`fa3_static_gqa_20261014.patch`; resources: `fa3_static_gqa_resources_20261014.json`.
Local source/build/SASS archive: `fa3_static_gqa_local_build_20261014.tar.gz`,5285923 bytes,
SHA256 `6a8bd34d4f67685175fd67ddd324c2bb1b2b53422fe3f6cb54529f956fbaf4fd`.
The initial build failure from copying stock rather than the patched `utils.cuh` is retained
in that archive; the working file restores the original packed-coordinate helpers and adds
only the new specialization helper.

Next validation uses separate source-build caches/processes for static0/static7, fp32 and
graph gates, a deliberate wrong-head-ratio rejection, and the same mixed/FA3-only fixtures.
An112 versus full-device SM-budget comparison would distinguish the fixed reservation cost
on FA3-dominated shapes; it has not been run here. No scheduling-policy work is resumed.

### GPU validation prepared

At12:37:35, the next inherited-lock hook was installed after the ordinary
`mix64_c48/BASE_2` serving audit. It runs only after that point's timed clients and server
terminate, then returns control to the remaining B matrix. No timed run is interrupted.

`fa3_static_gqa_20261014.sh` builds dynamic0/static7 variants in distinct source/build caches
and fresh processes, checks four fp32/eager/graph gates, requires useful PDL overlap on the
large-prefill gates, and tests the wrong-head-ratio rejection. Its40 timing cases cross
five fixtures,112/full-device SM budgets, both GQA variants and two reversed-order repeats.
Each case requires numerical/replay/unfused-control checks before accepting timing output.

Overlay: `fa3_static_gqa_overlay_20261014.tar.gz`,77538 bytes, SHA256
`df674e694a19e085522cc5fffbfce57a410d3111d0bff1961674dc1f9ddeb074`.
All six source-baseline hashes were checked on the pod before installing the hook.
The future run log is `/dev/shm/w/fa3_static_gqa_midpoint_20261014.log`; status is
`/dev/shm/w/b1_kernel_concurrency_20261014/fa3_static_gqa_midpoint_status.json`.
The trigger remains pending; preparation is not a GPU pass or a speedup claim.

### GPU result: small gains, primary gap remains

The run completed13:20:48. All four correctness/overlap gates, the deliberate head-ratio
rejection and40 timing cases pass. The full-device budget is132 SMs on this H200.
Two-repeat mean kernel-pair times:

| Fixture | Dynamic GQA,112 SMs | Static7,112 SMs | Static7,132 SMs | Unfused control at the last setting |
|---|---:|---:|---:|---:|
| prefill2048 +8 verify | 4.68405 ms | 4.62250 ms | 4.63370 ms | 4.06515 ms |
| append512 +46 verify | 2.38630 ms | 2.38005 ms | 2.42315 ms | 2.31000 ms |
| FA3-only8x1024 | 9.48900 ms | 9.36345 ms | 8.53455 ms | 7.51730 ms |
| split q32 +8 verify | 0.16990 ms | 0.17050 ms | 0.16325 ms | 0.98545 ms |
| prefill8192 +24 verify | 18.42825 ms | 18.16175 ms | 16.69995 ms | 14.49265 ms |

Static GQA alone changes these means by about-1.7% to+0.4%, depending on budget/fixture.
Using all SMs helps the larger-query populations but hurts the append-heavy case. Even the
improved large-query cases remain13.5–15.2% behind their unfused controls. Neither a universal
132-SM setting nor a server-wide promotion is justified. These are attention-kernel timings,
not TTFT/ITL improvements.

Raw archive: `fa3_static_gqa_complete_20261014.tar.gz`,29046 bytes,93 manifest entries,
SHA256 `75e276f63fbfb843598d5343f64bf3ad60f14cb8e840975748fee63b9f442c01`.
All40 scalar/source-log audits are in `fa3_static_gqa_progress_20261014.json`.
The original serving auditor was restored13:33:22, after the inherited-lock hook returned;
the128–256K B matrix continues unchanged.

## Next A direction: stock FA3 primary, not another cold-path tweak

The existing `hopper/fa3_upstream.cuh` already contains the real FlashAttention3 mainloop
and `LaunchUpstreamFA3OwnSched`; its own comments identify the simplified port's mainloop
deficit. The current migrated native-PDL loader does not enable that branch. Inspect and
reuse this path rather than treating the measured gap as something the argument interface
or a few divisions can recover.

A PDL wrapper must use the stock kernel's actual thread/shared-memory geometry, release its
independent FA2 dependent early, and preserve final store completion. Reuse per-step dense
page-table/scratch metadata rather than allocating or converting it per layer. Split-KV
combine cannot run before launching the independent FA2 work if overlap is intended: either
defer it, or initially use the stock primary only for proven-unsplit plans and retain the
existing split-capable primary otherwise. This bridge is not implemented or validated yet.
