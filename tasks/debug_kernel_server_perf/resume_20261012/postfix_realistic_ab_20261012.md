# Post-RoPE-fix realistic serving A/B (chat 24/s + TraceLab codex 0.1/s)

First serving timing on a verified-correct model: it re-runs the recorded 2026-10-08 headline setup
after the YaRN `rope_theta` fix ([RoPE evidence](a2_serving_ropefix_20261012.md)). One round, no CUDA
graphs in any arm (the fused path's graph-mode draft-indexing issue is still open). **Not a
production-graph claim.**

## Setup

- Single H200 trial `f096519a4d17aa1c` (`vibecuda`). sglang-up base `cc739bd2d2f8c74005bdd438acdf645402b83b08`
  plus the recorded working tree and the rope fix (`/dev/shm/w/validated_upstream`); flashinfer base
  `1a73e678a0210aa926f615c5811554813a67e6e2` plus the A2 overlay (`persistent.cuh` `e0ac82f4…`,
  `fused_plan.cuh` `3360dc0c…`). This repo at `4e7cd23da`.
- Scenario `agent_chat_rated`: chat = 4800 random requests (input 512±50%, output 96) at 24/s; codex =
  36 TraceLab coding sessions (context 32k–48k, output ≥512) at 0.1/s, all released by a start barrier.
- Qwen2.5-7B + EAGLE3 chain draft, `--speculative-num-steps 3 --speculative-num-draft-tokens 4
  --speculative-eagle-topk 1` (the runner's default q4, **not** the q8 of the correctness gate),
  `--enable-mixed-chunk`, `--max-running-requests 48`, ctx 131072 with YaRN x4.
- Arms (env differences from `run_mixed_workloads.py`):
  - `MTP_LPT`: stock speculative server; prefill chunks and verify rows never share a step.
  - `MTPMIX_LPT`: 4-mode step (`SGLANG_MIXED_SPEC=1`), FA2/persistent attention only.
  - `MTPMIXFUSED`: 4-mode step + two-executor plumbing + the fused FA3+FA2 kernel
    (`SGLANG_FUSED_KERNEL_EXEC=1`), classes 48/32/16, A2 tail on, FA3 SMs 112, backfill on.

```bash
# on the trial (serialised by /dev/shm/w/a2_kernel.lock); runner sha256 144dc5d4…
bash /dev/shm/w/postfix_realistic_ab_20261012.sh   # copy: postfix_realistic_ab_20261012.sh
```

## Execution gate

All three: 4800/4800 chat + 36/36 codex completed, **0 fallbacks, 0 tracebacks**. Built-in rope check
before the arms: `get_rope_config: (1000000.0`.

| arm | `MIXEDSPEC_STEPS` (lower bound, logged every 50) | `FUSED_KERNEL_CALLS` | `FUSED_KERNEL_PLAN` lines |
|---|---|---|---|
| MTP_LPT | 0 | 0 | 0 |
| MTPMIX_LPT | >=1651 | 0 | 0 |
| MTPMIXFUSED | >=1651 | >=43601 (≈1557 steps × 28 layers) | 1558 |

## Results (one round)

| | MTP_LPT | MTPMIX_LPT | Δ vs stock | MTPMIXFUSED | Δ vs stock | Δ vs MIX |
|---|---|---|---|---|---|---|
| chat mean TTFT (ms) | 32545 | 22170 | **-31.9%** | 29190 | -10.3% | +31.7% |
| chat P99 TTFT (ms) | 61414 | 37606 | -38.8% | 55764 | -9.2% | +48.3% |
| chat mean ITL (ms) | 32.95 | 28.79 | -12.6% | 32.37 | -1.8% | +12.4% |
| chat median ITL (ms) | 26.02 | 20.27 | -22.1% | 16.93 | -34.9% | -16.5% |
| chat P99 ITL (ms) | 104.2 | 113.5 | +8.9% | 112.0 | +7.5% | -1.3% |
| chat output tok/s | 1287.5 | 1418.7 | **+10.2%** | 1262.5 | -1.9% | -11.0% |
| chat req/s (offered 24) | 17.93 | 19.76 | | 17.58 | | |
| codex mean TTFT (ms) | 25911 | 17679 | **-31.8%** | 22834 | -11.9% | +29.2% |
| codex mean ITL (ms) | 17.46 | 15.25 | -12.7% | 16.41 | -6.0% | +7.6% |
| codex P99 ITL (ms) | 89.9 | 88.2 | -1.9% | 71.7 | -20.2% | -18.7% |
| codex output tok/s | 147.0 | 166.5 | **+13.3%** | 144.6 | -1.6% | -13.2% |
| accept length chat / codex | 2.17 / 2.22 | 2.13 / 2.15 | | 2.13 / 2.15 | | |

Readings:

1. **All arms are saturated** (chat completes at 17.6–19.8 req/s against 24 offered), so TTFT is
   queueing time and moves with throughput. The comparison is a capacity comparison.
2. **The 4-mode schedule itself helps on the fixed model**: TTFT -32% for both tenants and
   throughput +10%/+13%, which clears the TTFT and throughput bars. Mean ITL is only -13%, short of
   the 30% bar; P99 ITL is flat. This is the unfused schedule; it is not a kernel result.
3. **The fused kernel gives that gain back**: against the unfused 4-mode arm it loses 11–13%
   throughput and ~30% TTFT. Its median ITL is lower (16.9 vs 20.3 ms) while the mean is higher,
   so typical steps are not slower but the capacity lost to slower mixed steps backs up the queue.
   This matches the kernel harness, where the fused kernel is several times slower per call than
   `BatchAttention` on small shapes (A3 winner map, first shape: 0.078 vs 0.011 ms at kv 1024, one
   decode row).
4. Acceptance on the real workload is 2.13–2.22 tokens per verify for chain-4 (EWMA in the logs is
   higher for long codex decodes, ~3.8). It is identical between the two 4-mode arms.

## Serving-output gate at q4: PASS (2026-10-12)

`q4_gate_and_round2_20261012.sh` part 1: `srv_correctness.py` with `MTP_DRAFT_TOKENS=4
MTP_SPEC_STEPS=3`, otherwise the q8 gate's settings. Every arm 12/12 requests, 0 fallbacks, 0
tracebacks, mixed steps logged for every 4-mode arm.

| arm vs `MTP_LPT_Q4` | long l0/l1/l2/l3 common prefix | verdict |
|---|---|---|
| `MTPMIX_LPT_Q4` | 96 / 11 / 96 / 96 | PASS |
| `MTPMIX_SPLIT_Q4` | 96 / 11 / 96 / 96 | PASS |
| `MTPMIXFUSED_Q4` (A2 tail on) | 96 / 32 / 96 / 96 | PASS |
| `MTPMIXFUSED_Q4` vs `MTPMIX_LPT_Q4` | 96 / 11 / 96 / 96 | PASS |

All first tokens agree on the long prompts; the short-prompt s1 first-token difference is the
bf16 five-way tie adjudicated in [the RoPE evidence](a2_serving_ropefix_20261012.md) section 4.

## Why the fused arm loses: the FA2 role is slow *inside* the fused kernel

`fa2_slow_prof_20261012.sh` (nsys, no graphs) on one 8-token FA2 request at kv 4096:
`FusedFA3PersistentKernel` 345 us per call vs `BatchAttention`'s `PersistentKernelTemplate` 15 us,
same runner family, same fp32-reference verdict (0 bad rows). On the FA3 side of the same shape the
fused kernel is *faster* than upstream FA3 (24 vs 60 us). Ablations (graphs on): backfill off
0.345, budget split off 0.348, tail off 0.286, `SM_SPLIT=0` 0.349, `FUSED_FA2_KV_CHUNK=4096`
**1.226 ms** -- so one 1024-key FA2 item costs ~0.3 ms whatever the scheduling, which is the inner
loop. `cuobjdump -res-usage`: the fused kernel compiles at **168 registers with a 4200-byte stack**
(q64 variant 144 / 2488); flashinfer's `BatchAttention` kernel is **255 registers, 80-104 B**. The
FA2 body inherits the FA3 role's 384-thread register cap and spills in its hot loop.

**Retraction (in place, main doc 2026-10-10 "the FA2 22x is mostly a fixture artifact").** That
conclusion was drawn on a prefill-shaped row routed to FA2. Here the row is a decode/verify shape,
FA2's home ground, and the gap to `BatchAttention` is still 21x. The 22x is a real per-item defect
of the co-compiled FA2 body, not a routing artifact.

Fix under test: `FLASHINFER_FUSED_COLAUNCH=1` (see main doc A).

## Round 2 (rotated order, adds the two-executor control)

`q4_gate_and_round2_20261012.sh` part 2, same setup. `MTPMIX_SPLIT` (two-executor split: upstream
FA3 for prefill rows, flashinfer persistent for verify rows; >=1000 two-sided steps logged, 1-5
prefill rows to FA3 per step) is a valid, engaged control.

## Artifacts

`postfix_agent_chat_rated24_20261012.tgz`: bench logs, validation JSONs, codex per-request JSONs,
server logs, the run log and runner (chat per-request JSONs omitted for size; full archive
`/dev/shm/w/postfix_agent_chat_rated24_20261012.tar.gz` sha256 `e746232e…` on the trial).

| round 2 | MTP_LPT | MTPMIX_LPT | MTPMIX_SPLIT | MTPMIXFUSED |
|---|---|---|---|---|
| chat output tok/s | 1284.2 | 1414.2 (+10.1%) | 1409.9 (+9.8%) | 1251.2 (-2.6%) |
| chat mean TTFT (ms) | 33104 | 22299 (-32.6%) | 23249 (-29.8%) | 31808 (-3.9%) |
| chat mean / median / P99 ITL (ms) | 33.1 / 27.4 / 99.7 | 28.9 / 20.3 / 114.8 | 29.0 / 20.4 / 118.2 | 33.1 / 16.8 / 109.9 |
| codex output tok/s | 143.6 | 165.9 (+15.6%) | 162.9 (+13.5%) | 143.5 (-0.1%) |
| codex mean TTFT (ms) | 26512 | 17665 (-33.4%) | 18548 (-30.0%) | 26329 (-0.7%) |
| codex mean ITL (ms) | 17.6 | 15.3 (-13.0%) | 15.4 (-12.2%) | 14.6 (-17.1%) |
| accept length chat / codex | 2.16 / 2.21 | 2.13 / 2.15 | 2.13 / 2.15 | 2.12 / 2.15 |
| fallbacks / tracebacks | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

**Run-to-run noise on this setup:** chat throughput moves 0.3% (MTP_LPT, MTPMIX_LPT) and 0.9%
(MTPMIXFUSED) between rounds, so the 10-13% gaps are well outside it. The two-executor split
(FA2 rows at native `BatchAttention` speed) matches the unfused 4-mode arm within 0.3%: the 4-mode
schedule's gain survives the split host path, and the fused arm's 11% loss is the kernel.
