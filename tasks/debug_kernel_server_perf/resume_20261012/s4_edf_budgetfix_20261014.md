# EDF mixed-budget accounting correction

The first timed matrix stopped at07:05:58 on its first EDF arm. The preceding
unfused baseline completed cleanly. The EDF arm's server reported a prefill
extent of-14 with32 running verify rows, then FlashInfer rejected the negative
KV/query length. The failed arm has no usable performance result.

The fault is in the new EDF adapter, not the attention kernel. Its controller
returns a **prefill-token** cap, while `PrefillAdder` takes a **total** chunk
budget and subtracts one reserved slot per running decode row. Passing the
prefill cap directly can subtract those reservations twice. A minimal
reproduction is18 prefill tokens and32 reserved slots: the old adapter passes18
and the adder obtains-14.

The fix converts units at the boundary. It subtracts the decode reservation
from the available total before asking the predictor, then adds that
reservation back only when a positive, page-aligned prefill is admitted.
Zero available prefill or a zero predictor cap defers prefill explicitly.
The example now passes50 total slots and leaves18 prefill tokens. It does not
impose a large minimum chunk, change request limits or disable CUDA graphs.

Fourteen runtime CPU tests pass locally and on the trial, including the exact
32-row/18-token regression, insufficient total capacity, zero admission and
page alignment. `s4_edf_budgetfix_gate_20261014.sh` runs fresh stock/unfused-EDFR/
split-EDFR output gates in a private `up_edf_budgetfix/python` tree. Its logged
budget audit requires total=positive prefill+reserve, or total=0 when deferring.

The patch is `s4_edf_budgetfix_20261014.patch`; the source/test overlay is77,177
bytes, SHA256
`fc5a84a368b1fb5bf48ff30539b203e2721490017a5823c16398b77ef2b7d597`.
Gate PID731658 started at07:15:36 after source hashes and CPU tests passed.

The original failed run remains intact. `s4_edf_serving_v2_20261014.sh` repeats
the same24-arm matrix in a new output root after this gate succeeds. Workloads,
models, SLOs, admission policy and graph settings are unchanged; only the
budget-unit correction and additional budget assertions differ.

All three fresh output/budget gates passed at07:19:13, with graphs enabled,
positive EDF execution counters and no errors/fallbacks. The first logged
512-token total budget leaves511 prefill tokens after its one-row reservation.
This checks the corrected units; the real failed workload still needs its
full replay, rather than treating the short gates as sufficient coverage.

V2 PID733696 started the fresh24-arm matrix at07:19:18. The original failure
logs are preserved in `s4_edf_serving_v1_failure_20261014.tar.gz`,58,148 bytes,
SHA256 `f1f483ea852ba24d417c9ce6d83b27478289a75513f7a7181e197657107a9465`.
