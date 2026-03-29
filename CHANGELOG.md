# Changelog

## v0.2.0 — Mixed Batch CUDA Graphs + LPT + Server Benchmarks (2026-03-27 → 2026-03-29)

### Mixed Batch CUDA Graph Support
- Implemented mixed batch (prefill+decode) CUDA graph capture and replay for persistent kernel (BatchAttention)
- LPT (dynamic scheduler) compatible with CUDA graphs — activates at runtime when plan() detects mixed batch
- 6 bugs fixed: missing kv_indices copy, plan_info[-1] wrong field, module auto-switch breaking CG, padding overflow in replay, shared indptr buffer too small, LPT pointer initialization for CG

### Server Benchmarks (H100 80GB, Qwen3-8B)
- v3+LPT+CG beats prefill-prio by +41-87% throughput on long prefill (12k-16k input)
- HOL blocking crossover at ~8k input tokens for prefill-prio (max ITL 21-25s vs v3's 230ms)
- v3 matches default-mc within ±0.1% — plan() overhead offsets CG savings

### LPT Load-Balance Profiling
- Grid.sync() wait reduction: -15% to -39% max wait on KV-splitting workloads
- Kernel speedup: 7-16% on mixed workloads

### Overlap Profiler
- Fixed shape mismatch for persistent v3 compatibility (disable CG during profiling)

## v0.1.0 — Persistent Kernel v3 + Decode Fix (2026-03-26)

### Decode Overhead Fix
- Decode steps use regular BatchDecodeWithPagedKVCacheWrapper instead of BatchAttention
- Eliminated 0.5-4% throughput regression vs default-mc
