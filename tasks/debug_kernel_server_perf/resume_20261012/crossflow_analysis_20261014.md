# Crossflow: implications for the existing kernel and serving work

2026-10-14. User-requested reference analysis, not a replacement task list or a change to A/B.
Primary source: Yi Xu et al., [Crossflow: Prefill–Decode Elasticity for Agentic LLM Serving,
arXiv:2609.27085v1](https://arxiv.org/html/2609.27085v1), submitted 2026-09-22.

## What the paper establishes

Crossflow retains fixed P/D roles. Decode workers advertise expiring, versioned leases for
compute, KV, projected output, transfers and spill slots. The router tracks uncached-token
queue pressure, compares ordinary P-to-D service with decode-local prefill, reserves capacity
atomically and limits cluster-wide borrowing. Workers revalidate before each bounded prefill
unit; logical prefix coverage is not proof of physical residency. Conversations retain their
decode home. The implementation adds no model kernels; reported experiments use native
chunked mixed batches, with PDMUX also implemented. [Sections 4–5](https://arxiv.org/html/2609.27085v1#S4).

Evaluation uses GPT-OSS-120B/GLM-5.2, normally six TP4 GB300 servers. One-hour trace segments
retain timestamps/order/prefix relationships; load divides timestamp offsets by a multiplier.
Reported geometric-mean input/output-throughput gains over static P/D are 17.4%/16.2%.
GLM-5.2 TraceLab mean ITL worsens 20.9–34.3%, despite better TTFT. At TraceLab 4P2D,
input throughput regresses 2.8% and TTFT worsens 33.3%. Its 120-ms p99-ITL constraint belongs
to the Figure 3 characterization, not a universal per-dataset SLO.
[Sections 3 and 6](https://arxiv.org/html/2609.27085v1#S6).

## Consequences for our claims — analysis, not measured results

**Crossflow is a necessary policy baseline, not evidence that our kernels are finished.**
Our comparison cannot assume every disaggregated system always sends every append-prefill
through a P worker. The stronger question is whether symmetric colocation and our actual
FA3/FA2 executor improve on elastic P/D under matched load, cache capacity and latency targets.

| Existing global-PD point | What we must distinguish and measure |
|---|---|
| 1. Cache-hit-first, SLO/EDF-aware admission | Cache-aware local prefill with latency protection alone is not a sufficient novelty claim. Test our choice among all feasible nodes against decode-home lending; separately ablate the EDF ordering of queued work. |
| 2. Best-effort migration and replication | More placement freedom may preserve useful replicas, but can also duplicate cold state. Measure physical hit rate, recompute, migration bytes, replica lifetime and eviction under tool gaps; hold distributed-cache configuration constant. |
| 3. Acceptance-aware model/predictor | Shape-dependent interference supports measuring a model, not assuming fusion makes prediction easy. Compare held-out step-time/tail errors and resulting SLO violations, with and without acceptance EWMA, after the kernel choice is settled. |
| 4. Symmetric RDMA/congestion | A complete communication graph does not prove lower congestion. Measure directional/link utilization, transfer tails and total traffic; ablate random near-tie source selection versus congestion-aware selection. |

The useful design lesson is the separation between advertised capacity and execution-time
authority: stale routing information must not force unsafe execution. If adapted later, our
admission path needs reservations, expiration, actual-residency checks and local revalidation,
not just a more elaborate scalar load score. This is a future policy requirement, not a reason
to divert the current kernel work into another scheduler implementation.

A fair later experiment separates two axes: policy (static P/D, Crossflow-style elasticity,
symmetric colocation) and executor (stock/native mixed path, our best split, gated custom
candidate). Otherwise routing gains can be incorrectly credited to fusion. We have not run
Crossflow, and the inspected paper does not link its implementation. A port would need to be
labeled a reproduction rather than the authors' artifact. Its 24-GPU default is outside our
allocation limit: any multi-node comparison remains H20-only, at most 16 GPUs, `vibecuda`.

User direction, 2026-10-14: avoid Crossflow's mean-ITL regression using EDF, cross-node
load balancing and best-effort cache replication. After A/B close, admit local prefill only
within existing requests' decode-deadline headroom; otherwise choose a cache-rich feasible
node and replicate opportunistically without blocking latency-critical work. Evaluate both
mean and p99 ITL per tenant alongside TTFT and throughput. This is the intended later policy,
not permission to resume scheduling work before the kernel items are resolved. Agent load
must be swept by concurrent programs, following ThunderAgent's concurrency methodology,
alongside other-tenant request-rate sweeps; fixed 0.1 sessions/s is not the evaluation basis.

## Why 0.1 sessions/s is not an established realistic rate

The old matrix fixes `CODEX_RATE=0.1` while sweeping only chat 4–24 requests/s.
Its comment explains that co-scaling agent arrivals made a low-chat arm take about 36 minutes;
it supplies no production calibration. Calling that rate “recorded” in the comment is not
evidence. See `eval_matrix_s12_20261012.sh:33` in this directory.

The driver also omits recorded think/tool gaps and filters context/output lengths; the older
agent builder defaults to at least 512 output tokens. These are controlled kernel-stress
replays, not faithful end-to-end session traffic. One new session every ten seconds is neither
inherently realistic nor inherently unrealistic: model-invocation rate, concurrent active and
tool-idle sessions, cache footprint and per-GPU demand determine the load. Session starts/s
and LLM requests/s must be reported separately.

Other primary sources support a different protocol:

- [TraceLab](https://arxiv.org/html/2606.30560v1) distinguishes sessions, human requests and
  model steps, and records human/tool delays. Its median output is 214 tokens; excluding outputs
  below 512 changes the workload materially.
- [SLOs-Serve, Section 6](https://arxiv.org/html/2504.08784v1) uses Azure arrival patterns and
  SLO-constrained capacity. Tight/loose TTFT limits are 3x/5x zero-load latency and TPOT limits
  50/100 ms; speculative TPOT is measured every ten tokens. Those are application classes,
  not a universal session rate or a ready-made constant for every dataset.
- The [PD-multiplexing paper, v1](https://arxiv.org/html/2504.14489v1) scales real Mooncake
  traces to its serving instance and also varies Poisson request rates. Its p99 TBT protocol
  must not be silently equated with speculative ten-token TPOT. This version names its system
  Drift; the existing reference list uses the later MuxWise title.
- [ThunderAgent](https://arxiv.org/html/2602.13692v1) examines concurrent programs and cache
  thrashing; a concurrency experiment does not establish a session-arrival constant either.

**Required protocol correction:** retain a timestamp/causality-preserving TraceLab replay,
state the chosen trace window and scaling factor, and report resulting session starts, model
requests and uncached-input/output tokens per second. Distinguish time-compressed trace replay
from completion-dependent replay with actual tool delays. Independently sweep agent and
other-tenant load through the underloaded/near-capacity/overloaded regions, plus standalone
other datasets, across the requested context bands. Keep gap-free stress tests separately
labeled. Use per-dataset/application SLO mappings and matched zero-load references, not the
old fixed 2s/50ms versus 10s/80ms budgets as if they came directly from a paper.

The draft `workload_matrix_catalog_20261014.json` is not a completed experiment. Its proposed
arrival-span matching rule and Mooncake `0.1 * multiplier` rule are not production-calibrated
and must be replaced before a broad sweep. The only new completed calibration is 128 natural
ShareGPT requests at client concurrency one; it does not fill the missing workload matrix.

## Kernel priority and current completion status

- **A remains partial.** Native FA2 PDL, dispatcher candidates and FA3 variants exist and pass
  gates, but no general serving win over plain stock FA3 + native FA2 split is established;
  dispatch trade-offs and the custom FA3 gap remain. See
  [paired serving](native_pdl_paired_serving_20261014.md) and
  [unpacked-query experiment](fa3_unpacked_20261014.md).
- **B remains partial.** Backfill and true multicast were implemented/tested. B1 measured a
  1.4–3.9% small-op share, but did not implement further draft/verify epilogue fusion. A
  profiling bound is not the requested implementation and does not silently close B.
- Moving into EDF before resolving those requirements was premature. The EDF expansion was
  stopped after its third completed arm; it is not being resumed. Crossflow strengthens the
  later baseline requirement, not a justification for bypassing A/B.

The main document's A/B/S headings, TODO states and existing claims are unchanged by this
reference addition. No overall 30% TTFT/ITL and 10% throughput claim is established across
the full requested realistic workload matrix.
