# AttentionFuse: outline and evidence map

Paper prose lives in `paper_draft.md`; this file is the per-stage map of claim -> rationale -> baseline -> result ->
status. Every number here is on the settings of draft §6.1 unless marked otherwise. Numbers measured under superseded
settings are listed once, in draft Appendix A, and are not re-measured unless a claim needs them. The pre-rewrite
outline is `archive/paper_outline_pre_rewrite_20261024.md`.

# Current goal

Make every stage's contribution hold, against its baselines, on all three models (Qwen2.5-7B, Llama-3.3-70B,
GLM-4.5-Air), with one system configuration across the stages. Tasks, in priority order:

- [ ] 2026-10-24 **Kernel: follow the user's rule, with the LPT overlap schedule, then re-measure from stage 1.**
      Rule (2026-10-12): FA2 takes only the small row tiles (16 / 32 / 48 packed rows); every row >= 64 goes to FA3.
      Today the serving path breaks it in three ways:
      - FA2 rows run on flashinfer's NON-persistent prefill kernel (planner tile 16 / 64 + a 32 override; no 48).
      - Rows of 33-64 packed stay on FA2-64 (`--packed-query-limit 64`).
      - The LPT persistent schedule is gone. It ran in the two-executor path (FA3 on a side stream || flashinfer's
        persistent LPT FA2 kernel) and was replaced by the native PDL path without a measurement.
      Steps:
      - [ ] Persistent LPT FA2 executor with tile classes 48 / 32 / 16. The planner's `CTA_TILE_Q_SIZES` (128/64/16
            today) and the kernel's runner set change. 48 needs its own warp layout: `get_num_warps_q` / `get_num_mma_q`
            cannot express it; 1q x 4kv x 3 MMA spilled in the 10-12 test.
      - [ ] FA3 takes every row > 48 packed (`FLASHINFER_FA3_MIN_PACKED_QO` / the native partition), run concurrently
            with the LPT FA2 kernel (two executors on two streams, so CTAs of both share SMs without co-compilation).
      - [ ] Stage 1 re-measured on the 96 trace-sampled steps vs the same three FA3 baselines and the current native v3,
            then stage 2 pairs.
- [ ] 2026-10-24 **Llama-3.3-70B through all three stages** (first of the two remaining models).
      - [ ] Provision H20 pods 0-3 (`mn_provision_70b_20261024.sh`; H200-built kernel libraries, ABI identical).
      - [ ] Stage 1: TP4 rank-0 trace-sampled captures.
      - [ ] Stage 2: one 4-GPU pod, audited pairs.
      - [ ] Stage 3: 16 GPUs = 4 TP4 replicas.
- [ ] **Stage 3 with the full stack** (kernel + 4-mode MTP) vs stock PD with the P:D ratio swept. Higher load and
      mixture spikes than stage 2. One run that supersedes the schedule-only table.
- [ ] **GLM-4.5-Air** after the 48-row tile: stages 1-3.
- [ ] **Step model re-fit on the native FA3 + FA2 path** (C2b numbers predate it).
- [x] 2026-10-24 Conditional-disagg policy comparison (emulated; Dynamo lists SGLang as unsupported): equal chat
      attainment to 64 programs, ours +4.8 / +11.7 pts at 96 / 128 programs. Kept as the explanation of the gain over
      stock PD, not as the headline baseline.
- [x] 2026-10-24 Draft restructured on the original outline (this file + `paper_draft.md`).

## Core contributions (one line each)

1. **C1, kernel.** Role-separated attention for the mixed step: prefill tiles on FA3, decode/verify rows on a
   row-shaped FA2 kernel, in one PDL-chained launch pair with a geometry-driven split plan.
2. **C2, single node.** A 4-mode step (prefill chunk + speculative verify in one forward) scheduled by deadline order
   with demote-on-miss: higher SLO attainment for both tenants, and a step time a model can predict.
3. **C3, cluster.** Symmetric colocated nodes with cache-first, SLO-checked routing and a cluster KV pool. They
   sustain higher load at SLO than PD disaggregation and need no P:D ratio.

## Stage map

| stage | claim | why it should hold | baselines | result so far (Qwen2.5-7B) | Llama-70B | GLM-4.5-Air |
|---|---|---|---|---|---|---|
| **1. Kernel** (draft §6.2) | C1: faster mixed-step attention on the steps serving actually produces | One varlen FA3 call tiles decode/verify rows at prefill width (128 rows); a 4-token verify packs 4 x GQA = 28-32 rows. Row-shaped tiles stop the padding waste. | sglang's single FA3 call (deployed); best single fixed FA3 configuration; per-step oracle over all FA3 variants | **1.23-1.57x** over sglang's call on 96 trace-sampled steps (2/4-mode, 32-256K, H200); 0.98-0.99x of the best fixed FA3 split; FP32 error <= 0.0075 | pending | negative in 4-mode (0.63-0.98x): 48-row verify tile missing |
| **2. Single node** (draft §6.3) | C2: kernel + 4-mode + deadline order raise attainment for both tenants | Mixed chunk pays prefill-width attention on every decode row; FCFS makes chat wait behind long agent prefills; plain EDF under overload serves already-late requests first | sglang mixed chunk on FA3; stock EAGLE3 MTP in 4-mode; same frozen tree, audited opposite-order pairs | kernel alone at 128-256K: **+10.7 / +11.6% agent tput, agent TPOT -20..-24%**; full stack at 32-64K: **chat attainment 50 -> 59-65%, agent 45-52 -> 77-85%, chat TTFT -96%**, tput +1.7 / -8.5%, chat TPOT +35-56% | pending | pending |
| **3. Cluster** (draft §6.4) | C3: more load at SLO than PD disaggregation, without a P:D ratio | Disagg re-ships each agent round's prefix KV and leaves decode compute idle; symmetric nodes keep the prefix where it was computed and route by hit | sglang PD (mooncake) with P:D swept at worker granularity, KV pool on; reference: conditional-disagg policy | schedule only (kernel off, no spec), 16 H20s: **ours holds >= 90% chat attainment to 64 programs, best disagg to 16**; vs the conditional-disagg policy, +17% load at 90%; full-stack rerun pending | old run only (Appendix A) | pending |

## What each stage still owes

- **Stage 1.**
  - The kernel the design specifies: the four kernel tasks above. Today it matches a well-tuned FA3 split in isolation
    (0.98-0.99x); its serving gain over the role split comes from the FA2 rows' decode latency (-15..-25 pts).
  - Llama and GLM.
- **Stage 2.**
  - Throughput: 10% only at 128-256K, and by the kernel alone. The full stack trades throughput for attainment at
    32-64K.
  - 128-256K has no single-GPU capacity point for either system (agent cold turns cannot prefill in 10 s), so it is
    argued at stage 3.
- **Stage 3.**
  - The full-stack rerun (one configuration with stages 1-2).
  - Load spikes and mixture shifts on that configuration.
  - Llama-70B, then GLM.
