# Page-128 SM-allocation comparison

The paged TMA kernel and real-serving gates pass, but multicast adds no clear
benefit at the initial 112-CTA FA3 budget. Before another kernel change, this
experiment compares 96, 112 and 120 FA3 CTAs using the same validated binaries.
Native FA2 remains the PDL dependent and can backfill released resources.

`tma_sm_budget_20261013.sh` runs as PID 589550 on the existing H200 after the
completed real-serving gates. It verifies source hashes and takes the same
exclusive GPU lock. There are 54 cases: three realistic mixed kernel fixtures,
three SM budgets, gather/unicast/multicast and two reversed-order repeats.
Fixtures cover fresh 512 + 46 verify rows, append 512 + 46 verify rows at KV48K,
and prefill 2048 at KV64K + eight verify rows. Every case retains the seven
fp32/stock/changed-Q replay checks; device tracing is disabled during timing.

These are kernel measurements, not a substitute for a new context-range serving
comparison. No SM budget or execution variant is promoted until its results are
collected and any serving benefit is separately verified.

## Completed result

All 54 cases passed, finishing at 23:53:47 +08:00 on 2026-10-13. Retain 112:

- Fresh 512 + 46 verify rows takes approximately 2.32 ms across all methods and
  budgets; the best alternatives differ from 112 by at most 0.25%.
- Append 512 + 46 verify rows clearly favors 112: gather/TMA/multicast mean
  3.191/3.217/3.216 ms. At 96 they take 3.949/4.073/4.087 ms; at 120 they take
  3.904/3.284/3.268 ms.
- Prefill 2048 + eight verify rows takes 4.766/4.852/4.929 ms at 112. The best
  120-budget differences are below 0.3%, while 96 is substantially slower.

This sweep does not recover the remaining kernel gap. It does not justify an
SM-budget promotion or another serving sweep of these budget changes.
`tma_sm_budget_complete_20261013.tar.gz` preserves all raw logs and environments,
17,561 bytes, SHA256
`db140113db21368b3fd9942eee8eda88fc7805ee68acdc61cbf7fa886dd66ddf`.
All 54 raw-log hashes match the adjacent completed audit JSON.
