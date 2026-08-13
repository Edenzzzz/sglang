# Task Order

1. **`debug_kernel_server_perf.md`** — Persistent overlap kernel: sync-free
   plan, measured k-NN kernel gate, overlap attribution, analytic cost model.
   Current story and Open TODOs at the top; H100-era history at the back.
2. **`realistic_workload_analysis.md`** — Serving-level evaluation: TraceLab
   zero-regression check, sweet-spot scan, mixed-tenancy win scenario, and
   SLO control with the analytic cost model. Paper-facing results live here.
3. **`server_perf_model.md`** — Latency prediction for prefill-decode
   colocation. Chunk control now uses the closed-form analytic model; the MLP
   remains only for the router `/get_load` signal.
4. **`edf_scheduling.md`** — EDF scheduler design and historical policy
   sweeps (see validity note at top).
