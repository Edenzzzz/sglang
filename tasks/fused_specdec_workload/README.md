# Fused draft + verify + prefill workload (for 3-input-mode kernel work)

## What this is

The batch mixture a **single fused attention kernel** would see if speculative
draft, speculative verify, and chunked prefill were admitted into the *same*
batch. It is simulated, because it cannot currently be observed: `ForwardMode`
makes `MIXED` / `TARGET_VERIFY` / `DRAFT_EXTEND` disjoint, and every
speculative algorithm force-disables mixed chunking
(`server_args.py:1505` EAGLE/EAGLE3/STANDALONE, `:1578` NGRAM). A live capture
with `--speculative-algorithm NGRAM` logged **192 EXTEND batches and 0 MIXED**,
confirming prefill never shares a batch today.

The **request stream is real** — arrival times, prompt and output lengths are
replayed from `bench_serving --output-details` captures of the co-tenancy
workload (coding agent + chat, 2,460 requests). Only the *scheduling* is
simulated.

## Scheduling model

Follows SparseSpec's unified scheduler (MLSys 2026, *Accelerating Large-Scale
Reasoning Model Inference with Sparse Self-Speculative Decoding*): rather than
k draft steps followed by one verify, every running request occupies one of
k+1 stages and each step mixes them, so a batch holds ~k/(k+1) drafting rows
and ~1/(k+1) verifying rows. New requests are greedily bin-packed into the
least-loaded stage. Chunked prefill is admitted into the same batch on top.
Acceptance is Bernoulli(α) per speculated token, advancing context length.

**Draft model is assumed separate (EAGLE-style), not the target's own layers.**
For batch composition this only enters as: a draft row attends the sequence's
current context. Row counts and KV lengths are unaffected by that choice.

## Row shapes

| class | qo_len | attended KV |
|-------|--------|-------------|
| draft | 1 | current context |
| verify | k+1 candidates | current context |
| prefill | chunk tokens | already-processed prefix |

A row's tile class follows `packed_qo_len = qo_len × gqa_ratio`.

## The key finding for kernel design

The **verify class is a distinct third tile exactly when `16 < (k+1)·r ≤ 64`**,
and in that band it is present in nearly every batch. Realized frequencies over
4,000 simulated steps:

| model (gqa r) | k | dominant classes | third class needed? |
|---|--:|---|---|
| Qwen2.5-7B (7) | 3 | `[16,64]` 87.2%, `[16,64,128]` 12.3% | **yes — 64 in 99.5% of steps** |
| Qwen3-8B (4) | 8 | `[16,64]` 77.2%, `[16,64,128]` 9.9% | **yes — 64 in 87.1%** |
| Qwen3-8B (4) | 3 | `[16]` 87.3%, `[16,128]` 12.6% | no (verify collapses into 16) |
| Qwen3-235B (16) | 8 | `[16,128]` 88.9% | no (verify collapses into 128) |

Today's persistent kernel has `CTA_TILE_Q_SIZES = {128, 16}`, so inside the
band it must round verify rows up to 128. Measured cost of that rounding
(`../debug_kernel_server_perf/specdec_tile_probe.py`, persistent vs
BatchPrefill which picks 64): **1.13–1.15× (r=7), 1.23–1.27× (r=4),
1.14–1.16× (r=16)** — and ~1.00–1.04× outside the band, confirming the
mechanism.

Fusion itself is worth **1.15×** on the persistent kernel but **0.59×** on
BatchPrefill (fusing flips its batch-wide tile), so a fused megakernel is only
viable on a per-row-tile design. Net today is 1.02× because the fusion gain is
cancelled by the verify-tile penalty — **closing that penalty is the work.**

## Workload characteristics (Qwen2.5-7B, k=3)

| | p10 / p50 / p90 |
|---|---|
| draft rows per batch | 13 / 18 / 35 |
| verify rows per batch | 4 / 5 / 12 |
| prefill chunks per batch | 0 / 0 / 1 |
| draft row KV | 449 / 41,076 / 47,579 |
| verify row KV | 447 / 41,065 / 47,571 |
| chunk tokens (when present) | 327 / 5,940 / 8,192 |
| chunk attended prefix | 0 / 2,669 / 32,866 |
| KV concentration (Σkv²/Σkv) | 41.0K / 44.2K / 45.6K |

High KV concentration with a 20×+ row-length spread is the composition that
starves BatchPrefill of query tiles — here it arises by construction from the
draft/verify split rather than by luck of co-tenancy.

## Files

| file | contents |
|---|---|
| `simulate_fused_batches.py` | the simulator (real trace in, per-step composition out) |
| `analyze_fused_workload.py` | distributions + representative compositions |
| `fused_batches_k3.jsonl`, `fused_batches_k8.jsonl` | 4,000 fused steps each; one JSON record per step with every row's `(qo_len, kv_len)` |
| `workload_spec_*.json` | frequency-weighted representative compositions (~95% step coverage) — **tune against these** |

Regenerate:
```bash
D=../realistic_workload_analysis/h800_capture_loadsweep_hi
python simulate_fused_batches.py $D/LPT_chat.json $D/LPT_codex.json \
    --k 3 --draft-tokens 4 --alpha 0.7 --out fused_batches_k3.jsonl
python analyze_fused_workload.py fused_batches_k3.jsonl --gqa 7
```

## Caveats

- Scheduling is simulated; only the request stream is real. Step time is a
  fixed 15 ms rather than fed back from a cost model, so wall-clock pacing is
  approximate — row *compositions* are the intended output, not timings.
- Acceptance is i.i.d. Bernoulli(α=0.7); real acceptance is correlated across
  positions, which would lengthen accepted runs and slightly reduce verify
  frequency.
- Draft rows here attend the full context. A draft using sparse attention
  (SparseSpec's PillarAttn, s=0.05) would shorten draft-row KV ~20×, *raising*
  KV concentration and making the composition more favourable, not less.
