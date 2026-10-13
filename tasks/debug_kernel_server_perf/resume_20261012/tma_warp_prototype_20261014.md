# One-warp TMA producer experiment

The validated TMA path still uses 128 producer threads even though one leader
issues each TMA load. This opt-in variant reduces active producers to 32 and
compares two consumer-register budgets, without changing attention arithmetic,
page geometry, SM budget, tensor maps or multicast eligibility.

Active producer count and the physical producer warpgroup are deliberately
separate. Pipeline arrivals, Q/consumer handshakes and partial-tail zeroing use
the active count. Consumer thread indices and register accounting retain the
physical 128-thread region. All producer warps deallocate registers before the
inactive warps wait at the final CTA barrier. The register-budget assertion
includes those inactive warps rather than incorrectly charging only 32 threads.

Three local CUDA builds register successfully from the final matching headers:
128 producers/216 consumer registers, 32/216 and 32/232. Thirty-two host layout
cases verify complete, nonoverlapping K/V tail coverage for both producer widths
and both pipeline stages. Compile-time checks cover the physical thread split
and supported/over-budget register combinations.

Local resource reports show 168 entry registers for all variants. The dedicated
entry's reported stack frame grows from 40 bytes to 616 bytes in both 32-thread
variants. This is a warning to investigate in measurements, not proof of an
executed spill cost or a speedup. The earlier intermediate build was superseded;
all three final build headers match the exported candidate.

`tma_warp_kernel_20261014.sh` runs as PID 599626 after the completed SM-budget
sweep. It uses private `fi_tma_warp` and `up_tma_warp` trees and three separate
extension caches. Eighteen gates cover every variant with unicast/multicast,
useful overlap, poisoned tails and varied split-KV inputs. Sixty timing cases
then cover five mixed/FA3-dominant fixtures, both multicast modes, all variants
and two reversed-order repeats. Every timed case retains all seven numerical,
stock-control and changed-input graph checks; device tracing is disabled.

Archive: `tma_warp_overlay_20261014.tar.gz`, 39,792 bytes, SHA256
`0c6e6ce5f720a380d93d7615c41e465e1f4ba0e83126ae26e434c9fa33459342`.
The adjacent patch, compile logs, resource report and layout-test output preserve
the implementation and local evidence. No default or main task-document item is changed.

## Completed GPU result

All 18 gates and 60 timing cases passed, completing at 00:44:14 +08:00 on
2026-10-14. Neither one-warp variant is promoted. Relative to the matching
128-thread control, 32/216 changes the regular fixtures by roughly -0.3% to
+0.6%, while the varied-odd fixture regresses 6.1–8.6%. The 32/232 variant gains
at most 1.2% on the regular fixtures and regresses varied-odd by 5.3–7.9%.
Those results do not close the kernel or serving gap.

The varied-odd regression is consistent with the larger reported stack frame
being worth investigation, but this experiment does not isolate spill cost or
prove that causal explanation. Hardware profiling is the next diagnostic.

`tma_warp_complete_20261014.tar.gz` preserves all raw logs and environments,
34,317 bytes, SHA256
`cc6128ad4e50971e41372c5193f094a3080b3fb618e27de18cb1b207fbaca676`.
All 60 timing-log hashes match the adjacent audit JSON; all 18 gate logs pass.
