# Three 4-class workload sets for mega-mixed-tile kernel optimization

Target: one kernel (or one dispatching scheduler) that serves four co-resident
work classes on Hopper/Blackwell, with persistent-SM proportions assigned from
the live mixture.

| class | what it is | `packed_qo = qo_len × gqa_ratio` | tile |
|-------|-----------|-------------------------------:|-----:|
| `large_prefill` | chunk attending a long cached prefix | chunk × r | 128 |
| `spec_verify` | (k+1) candidate tokens per row | (k+1) × r | **64** |
| `mixed_decode` | small/fresh-prefix chunks (our LPT regime) | chunk × r | 128 |
| `spec_draft` | one drafted token per row, smallest work | r | 16 |

**`k` is per-model, not a free choice.** The verify class occupies its own tile
only when `16 < (k+1)·r ≤ 64`. At r=16 that is `k+1 ∈ (1,4]`, so the intended
"2 speculative steps" (k=1, packed 32 → tile 64) works. At r=7, k=1 gives
packed 14 → tile 16, which **collapses verify into the draft class**; the small
model needs k=3 (packed 28 → tile 64) to exercise all four. Sets below use the
right `k` per target.

## The sets

| set | model / hardware | gqa | resident reqs | k | rows per batch (prefill / verify / mixed / draft) |
|-----|------------------|----:|--------------:|--:|--------------------------------------------------|
| `qwen_small_h800.jsonl` | Qwen2.5-7B bf16, **H800 ×1** | 7 | 28 | 3 | 0.2 / 7 / 1.1 / 21 |
| `qwen_large_h200_tp8.jsonl` | Qwen3-235B-A22B fp8, **H200 4×8 (TP8)** | 16 | 115 | 1 | 0.7 / 58 / 0.4 / 57 |
| `qwen_large_gb200_tp16.jsonl` | Qwen3-235B-A22B fp8, **GB200 NVL72 (TP16)** | 16 | 364 | 1 | 0.7 / 182 / 0.5 / 182 |

Each line is one fused batch: `{large_prefill, spec_verify, mixed_decode,
spec_draft}`, each a list of `[qo_len, kv_len]` rows.

## Hardware sizing (`hw_capacity.py`)

Per-GPU KV budget = `mem_frac × HBM − weights/TP`; resident requests =
KV tokens / context length. At `mem_fraction_static=0.85`, 32K context:

| config | weights/GPU | KV/GPU | tok/GPU | reqs @32K |
|--------|------------:|-------:|--------:|----------:|
| Qwen2.5-7B bf16, H800 ×1 | 15.2 GB | 52.8 GB | 921K | 28 |
| Qwen3-235B fp8, **GH200 4×4 (TP4)** | 58.8 GB | 22.8 GB | 475K | **14** |
| Qwen3-235B fp8, H200 4×8 (TP8) | 29.4 GB | 90.5 GB | 3,760K | 115 |
| Qwen3-235B fp8, GB200 NVL72 (TP8) | 29.4 GB | 128.7 GB | 5,349K | 163 |
| Qwen3-235B fp8, GB200 NVL72 (TP16) | 14.7 GB | 143.4 GB | 11,919K | 364 |

**Qwen3-235B does not fit 4×4 GH200 in bf16** (117.5 GB/GPU against an 81.6 GB
budget). In fp8 it fits but leaves only 22.8 GB of KV — 14 resident requests at
32K, too few for a meaningful mixed batch — so the GH200 4×4 config is dropped
in favour of **H200 4×8**, as anticipated.

## Where the row shapes come from

Real captures via `SGLANG_LOG_MIXED_COMPOSITION`, spanning the prefix axis:

| capture | batches | attended prefix p50 | rows >512 | what it is |
|---------|--------:|--------------------:|----------:|------------|
| `h800_capture_loadsweep_*` | 3,493 | 2 | 5% | co-tenancy, chat-dominated rows |
| `h800_capture_prefixreuse` | 406 | 8,592 | 78% | multi-turn on 8K shared prefixes |
| `h800_capture_codexonly` | 148 | **32,664** | **90%** | pure session replay |

**On cache hits:** the low per-row prefix in the co-tenancy capture is *not* a
cache failure. Those runs hit the radix cache **79.7%** in token terms at 0.61
peak KV occupancy (no eviction). The row distribution is bimodal — the coding
tenant's continuations hit heavily but are few rows, while the chat tenant's
2,400 fresh random prompts have no reusable prefix by construction and dominate
the count. Pure session replay (`codex_only`) gives the expected near-full-hit
regime: 90% of rows above 512, 43% above 32K.

Large-model sets draw their row shapes from the reuse + sessions captures
(a large deployment serves cache-friendly traffic) and scale the decode
population to the hardware's resident-request capacity. Draft/verify split
follows SparseSpec's unified schedule: `1/(k+1)` of rows verifying, the rest
drafting, mixed into every batch rather than k draft steps then one verify.

## Caveats

- Row **shapes** are real; the fused *schedule* is simulated, because sglang
  cannot produce these batches today (every speculative algorithm sets
  `enable_mixed_chunk = False`, `server_args.py:1505`/`:1578`, and `MIXED` /
  `TARGET_VERIFY` / `DRAFT_EXTEND` are disjoint forward modes).
- Large-model sets are **extrapolated**: no Qwen3-235B, GH200, H200 or GB200
  was available here. Only the capacity arithmetic and the gqa-16 tile mapping
  are derived; the row shapes are borrowed from the 7B captures at scaled batch
  size, so treat per-row KV values as representative rather than measured.
- Draft rows attend the full context (a separate EAGLE-style draft). A sparse
  draft would shorten draft-row KV and raise KV concentration, making the
  mixture *more* favourable to per-row-tile scheduling, not less.
