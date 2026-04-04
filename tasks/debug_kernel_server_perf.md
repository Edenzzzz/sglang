# LPT Persistent Kernel + Mixed CG Performance

## Goal
Implement and optimize LPT (Longest Processing Time) dynamic scheduler for persistent attention kernel with mixed batch CUDA graphs in SGLang flashinfer backend.

## Setup
- Model: `Qwen/Qwen3-8B`, H100 80GB
- Server: `--enable-mixed-chunk --attention-backend flashinfer --mem-fraction-static 0.85 --enable-piecewise-cuda-graph`
- Env vars: `SGLANG_FLASHINFER_USE_PERSISTENT=1`
- Benchmark: `bench_serving --dataset-name random --request-rate 8 --num-prompt 400`
- Profile logs: `tasks/debug_kernel_server_perf/profile_logs/`
- gpu_keepalive MUST be stopped before benchmarks (corrupts results — cooperative launch is especially sensitive to SM contention)

---

## Kernel-Level Results

### LPT vs POD vs Static Persistent 3-way comparison
(H100 80GB, Qwen3-8B heads: num_kv=8 num_qo=32 dim=128, `bench_mixed_attention.py`, 100 decode each)

| Case | Workload | POD (ms) | LPT (ms) | Static (ms) | LPT over POD | LPT over Static |
|------|----------|----------|----------|-------------|-------------|-----------------|
| 1 | 100d@kv2k + 1p@512 | 0.30 | 0.31 | 0.30 | 0.97x | 0.97x |
| 2 | 100d@kv2k + 1p@1536 | 0.32 | 0.36 | 0.37 | 0.89x | 1.03x |
| 3 | 100d@kv2k + 2p@2048 | 0.44 | 0.48 | 0.52 | 0.92x | **1.09x** |
| 4 | 100d@kv2k + 1p@2048 | 0.34 | 0.38 | 0.42 | 0.89x | **1.11x** |
| 5 | 100d@kv4k + 1p@4096 | 0.81 | 0.82 | 0.97 | 0.99x | **1.18x** |
| 6 | 100d@kv8k + 1p@4096 | 1.28 | **1.24** | 1.51 | **1.03x** | **1.22x** |
| 7 | 100d@kv8k + 1p@6000 | 1.85 | 1.89 | 2.18 | 0.98x | **1.15x** |
| **8** | **95d@kv2k+5d@kv24k+1p@2k** *(split-KV decode)* | 0.69 | **0.64** | 0.71 | **1.08x** | **1.11x** |
| **9** | **100d@kv2k+1p@kv24k,q4k** *(split-KV prefill, ratio≥2→static)* | 4.56 | 4.67 | 4.67 | 0.98x | **1.00x** |
| 10 | 100d@kv2k+1p@kv12k,q4k *(ratio≥2→static)* | 2.14 | 2.17 | 2.17 | 0.99x | **1.00x** |

(Cases 1-8: flashinfer `a74bb2d4`, `bench_mixed_attention.py`. Cases 9-10: ratio threshold auto-fallback to static.)

Speedup = baseline_time / LPT_time (>1x = LPT faster). **LPT wins Cases 6,8 over POD.** LPT beats static by 1.03-1.22x on Cases 2-8. **Cases 9-10: ratio threshold kicks in → static, 1.00x (zero regression).** Still loses to POD on Cases 2,4 (0.89x) — cause TBD (not cooperative launch overhead; single CUDA stream means all SMs available).

**Note on old stale table**: An earlier table showed LPT beating POD by +3-18%. Those numbers were **contaminated by gpu_keepalive** (confirmed from conversation history). Keepalive inflates cooperative launch (persistent) more than POD, creating fake speedup.

**POD gap on Cases 2,4 (0.89x)**: nsys shows LPT kernel=362μs vs static=421μs on Case 4 — LPT is 16% faster as a kernel. But POD (separate prefill+decode launches) is 340μs total. The 22μs gap is NOT from cooperative launch overhead (single CUDA stream → all SMs available). Likely from grid.sync() cost within the persistent kernel or from the fused kernel being suboptimal on small workloads where two specialized kernels are more efficient.

### LPT Load-Balance Profiling
Script: `flashinfer/profiler/profile_lpt_balance.py`. Uses `bench_gpu_time` (warmup + L2 flush). Qwen3-8B heads (8/32/128). flashinfer `daf646c3`, H100 80GB. Verified ±1-2% match with `bench_mixed_attention.py`.

| Case | Workload | Static kernel | LPT kernel | LPT speedup | Static max sync | LPT max sync |
|------|----------|-------------|----------|---------|----------------|-------------|
| 1 | 100d@kv2k + 1p@512 | 0.302ms | 0.310ms | 0.97x | 26μs | 30μs |
| 2 | 100d@kv2k + 1p@1536 | 0.368ms | 0.355ms | 1.04x | 21μs | 17μs |
| 3 | 100d@kv2k + 2p@2048 | 0.519ms | **0.478ms** | **1.09x** | 25μs | **13μs** |
| 4 | 100d@kv2k + 1p@2048 | 0.423ms | **0.377ms** | **1.12x** | 23μs | 20μs |
| 5 | 100d@kv4k + 1p@4096 | 0.969ms | **0.822ms** | **1.18x** | 42μs | **16μs** |
| 6 | 100d@kv8k + 1p@4096 | 1.507ms | **1.245ms** | **1.21x** | 88μs | 112μs |
| 7 | 100d@kv8k + 1p@6000 | 2.175ms | **1.885ms** | **1.15x** | 1,510μs | **52μs** |
| **8** | **95d@kv2k+5d@kv24k+1p@2k** *(split-KV decode)* | 0.708ms | **0.665ms** | **1.06x** | 176μs | 150μs |
| **9** | **100d@kv2k+1p@kv24k,q4k** *(split-KV prefill)* | 4.335ms | 4.736ms | 0.92x | 121μs | 1,933μs |

(Max sync converted from clock cycles at 1.83 GHz.)

**Key findings**: LPT wins Cases 3-7 (9-21% speedup). Split-KV decode (Case 8) gets 6%. Split-KV prefill (Case 9) regresses 8%.

**Case 9 regression investigation**:
- Consistent ~8% across kv12k/16k/24k (0.92x, 0.93x, 0.92x vs static)
- FA3-style stride sweep (commit `1fe946aa`, flashinfer `persistent_overlap_`):

  Auto-stride = gridDim.y/2 static initial tiles (FA3-style, `is_first_runner` from per-SM `cta_counter`, predicated `atomicAdd(counter, 0/1)`). Zero improvement — atomic contention is not the bottleneck.
- NCU: same instructions (0.3% diff), SM Busy 51.8% vs 55.9%, Warp Cycles/Instr 6.45 vs 5.99. grid.sync() max wait 940K vs 225K cycles.
- Perfetto traces: `profile_logs/case9_lpt.perfetto-trace`, `case9_static.perfetto-trace`
- Realistic split-KV prefill sweep (single + multi-request, commits `1fe946aa`..`e1d28595b`):

| Config | Static (ms) | LPT (ms) | LPT speedup |
|--------|------------|----------|-------------|
| 100d@kv2k+1p@kv8k,q4k | 1.415 | 1.471 | 0.96x |
| 100d@kv2k+1p@kv12k,q4k | 2.083 | 2.252 | 0.92x |
| 100d@kv2k+1p@kv12k,q8k | 3.223 | 3.331 | 0.97x |
| 100d@kv2k+1p@kv16k,q8k | 4.813 | 5.017 | 0.96x |
| 100d@kv2k+4p@q2k,kv{2k,4k,8k,12k} | 2.320 | 2.488 | 0.93x |
| 100d@kv2k+2p@q4k,kv{4k,12k} | 2.474 | 2.657 | 0.93x |
| 100d@kv2k+8p@q1k,kv{1k..8k} | 1.787 | 1.937 | 0.92x |
| 100d@kv4k+4p@q2k,kv{2k,4k,8k,12k} | 2.575 | 2.662 | 0.97x |
| 100d@kv4k+2p@q4k,kv{4k,12k} | 2.729 | 2.806 | 0.97x |

- **Root cause & fix**: Compute-bound prefill makes dynamic scheduling overhead pure loss. When prefill compute dominates, LPT's atomicAdd + tile mapping adds overhead with no load-balance opportunity to recoup it.
  - **Fix** (commit `a74bb2d4`): When `prefill_kv_compute / decode_kv_compute >= 2.0`, set `flipped_schedule=false` (static for both tasks). Threshold found empirically:
    - ratio ≤ 0.92 → LPT wins (1.03-1.22x)
    - ratio ~2.0 → neutral (0.99x)
    - ratio ≥ 3.5 → LPT loses (0.91-0.97x)
  - **Result** (commit `a74bb2d4`, script: inline python with `bench_gpu_time`):

| Config | Static (ms) | Auto (ms) | Mode | Speedup |
|--------|------------|----------|------|---------|
| Case1: 100d@kv2k+1p@512 | 0.303 | 0.310 | LPT | 0.98x |
| Case2: 100d@kv2k+1p@1536 | 0.368 | 0.356 | LPT | 1.03x |
| Case3: 100d@kv2k+2p@2048 | 0.518 | 0.477 | LPT | **1.09x** |
| Case4: 100d@kv2k+1p@2048 | 0.423 | 0.379 | LPT | **1.12x** |
| Case5: 100d@kv4k+1p@4096 | 0.968 | 0.822 | LPT | **1.18x** |
| Case6: 100d@kv8k+1p@4096 | 1.506 | 1.239 | LPT | **1.22x** |
| Case7: 100d@kv8k+1p@6000 | 2.174 | 1.876 | LPT | **1.16x** |
| Case8: 95d@kv2k+5d@kv24k+1p@2k | 0.707 | 0.694 | LPT | 1.02x |
| Case9: 100d@kv2k+1p@kv24k,q4k | 4.663 | 4.666 | static | **1.00x** |
| 100d@kv2k+1p@kv8k,q4k | 1.446 | 1.440 | static | 1.00x |
| 100d@kv2k+1p@kv12k,q4k | 2.171 | 2.171 | static | 1.00x |
| 100d@kv2k+1p@kv12k,q8k | 3.400 | 3.411 | static | 1.00x |
| 100d@kv2k+1p@kv16k,q8k | 5.151 | 5.163 | static | 1.00x |
| 100d@kv2k+4p@q2k,kv{2k,4k,8k,12k} | 2.484 | 2.487 | static | 1.00x |
| 100d@kv2k+8p@q1k,kv{1k..8k} | 1.909 | 1.911 | static | 1.00x |
| 100d@kv4k+4p@q2k,kv{2k,4k,8k,12k} | 2.671 | 2.683 | static | 1.00x |

**Prefill-heavy sync regression (39μs→80μs)**: With 1 prefill@16k, LPT assigns largest causal tiles first → early CTAs get top q_tiles (16K KV each), late CTAs get bottom q_tiles (~32 tokens). Kernel time tied (13.95 vs 14.0ms) — compute dominates.

### Split-KV A/B Comparison
Split enabled vs disabled within LPT, 2x kv_limit. Speedup = (nosplit - split) / nosplit.

| Case | Split | No-split | Split speedup |
|------|-------|----------|---------|
| 3d@64k+1p@2k | **0.681ms** | 1.659ms | **59%** |
| 5d@32k+1p@4k | 1.228ms | 1.223ms | 0% |
| 10d@16k+1p@4k | 1.298ms | 1.303ms | 0% |
| 20d@8k+2p@2k | 0.932ms | 0.931ms | 0% |
| 100d@2k+2p@2k | 1.036ms | 1.030ms | 0% |

Split-KV with 2x kv_limit: 59% speedup on severe imbalance (3d@64k), zero regression on balanced workloads. NCU confirms zero overhead when split not triggered (INT_MAX sentinel).

### Kernel CG Overhead Analysis
Profiled persistent kernel at 1k-8k tokens (90% decode@kv512 + 10% prefill):

| Tokens | Kernel | CG saves ~50μs |
|--------|--------|----------------|
| 1024 | 0.68ms | **7.4%** |
| 2048 | 1.27ms | **3.9%** |
| 4096 | 5.05ms | 1.0% |
| 8192 | 10.07ms | 0.5% |

CG overhead savings meaningful at ≤2048 tokens. At 4k+, kernel dominates.

---

## Server-Level Results

*(Moved to consolidated table below: "Server Results: LPT+CG29 vs trueDMC")*

**P99 ITL: noCG is lower on WL2-6** (216-266ms vs CG's 231-318ms). CG padding adds 10-13% extra compute on worst-case steps. WL1 is the exception — CG's P99 (173ms) beats noCG (468ms) because CG eliminates the run() overhead that dominates WL1's short-output decode steps.

### WL1-6 Throughput (mixed rates, N=400, CG25)
LPT+CG = persistent LPT kernel + mixed CG25 (25 sizes, max 4096, ~1.79GB). All runs same session, `--mem-fraction-static 0.85`.

| WL | Input | Output | Rate | LPT+CG total | LPT noCG total | trueDMC total | CG benefit | LPT+CG vs DMC |
|----|------:|-------:|-----:|-------------:|---------------:|--------------:|-----------:|---------------:|
| 1 | 4096 | 1024 | 8 | **15,439** | 13,459 | 12,974 | **+14.7%** | **+19.0%** |
| 2 | 8192 | 2048 | 4 | **10,610** | 10,584 | 10,304 | +0.2% | **+3.0%** |
| 3 | 12000 | 1024 | 4 | **15,425** | 15,354 | 14,705 | +0.5% | **+4.9%** |
| 4 | 16384 | 1024 | 3 | **15,121** | 15,009 | 14,171 | +0.7% | **+6.7%** |
| 5 | 4096 | 4096 | 6 | **6,507** | 6,480 | 6,375 | +0.4% | **+2.1%** |
| 6 | 6000 | 6000 | 3 | **4,720** | 4,714 | 4,645 | +0.1% | **+1.6%** |

**Latency (CG25, mixed rates):**

| WL | LPT+CG TTFT med | LPT noCG TTFT med | trueDMC TTFT med | LPT+CG Mean ITL | trueDMC Mean ITL | LPT+CG P99 ITL | LPT noCG P99 ITL | trueDMC P99 ITL |
|----|----------------:|-----------------:|-----------------:|-----------------:|-----------------:|----------------:|-----------------:|----------------:|
| 1 | **303ms** | 3,600ms | 6,306ms | **34ms** | 44ms | **180ms** | 485ms | 527ms |
| 2 | **25,934ms** | 26,618ms | 28,513ms | **33ms** | 35ms | **218ms** | 202ms | 229ms |
| 3 | **22,065ms** | 23,233ms | 25,409ms | **43ms** | 46ms | 320ms | **244ms** | 266ms |
| 4 | **41,464ms** | 41,838ms | 48,489ms | **45ms** | 48ms | 301ms | **265ms** | 291ms |
| 5 | 50,422ms | 50,937ms | 53,150ms | **27ms** | 28ms | 116ms | **101ms** | 116ms |
| 6 | 105,056ms | 105,539ms | 108,174ms | **29ms** | 29ms | **113ms** | **98ms** | 119ms |

**Key findings:**
1. **LPT+CG beats true DMC on ALL workloads**: WL1 +19.0%, WL4 +6.7%, WL3 +4.9%, WL2 +3.0%, WL5 +2.1%, WL6 +1.6%.
2. **Mixed CG is the main WL1 win**: CG benefit = +14.7% throughput. Without CG, LPT still beats DMC (+3.7%).
3. **WL1 TTFT 21x better than DMC**: 303ms vs 6,306ms. Mixed CG eliminates Python run() overhead.
4. **Mean ITL**: LPT+CG lower on all workloads (27-45ms vs 28-48ms).
5. **P99 ITL**: LPT+CG wins big on WL1 (180ms vs 527ms). WL3 slightly worse (320ms vs 266ms) — CG padding: 51/200 replays pad 3073-4095 tokens to 3584/4096 (10-13% waste). Inherent CG tradeoff.

**NOTE**: Previous "DMC" benchmarks were invalid — mixed CG capture always used `BatchAttention` (persistent kernel) regardless of `use_persistent` flag. Fixed by using `SGLANG_DISABLE_MIXED_CG=1` for true DMC baseline.

### ShareGPT Benchmark
(Qwen3-8B, `--reasoning-parser qwen3`, `--sharegpt-output-len 4096`, N=400, rate=4)

| Config | Total tok/s | Out tok/s | TTFT med | Mean ITL | P99 ITL |
|--------|------------|-----------|----------|----------|---------|
| LPT+CG25 | **4,962** | **4,325** | **40,767ms** | **36ms** | **67ms** |
| DMC | 4,794 | 4,178 | 53,298ms | 41ms | 96ms |

LPT+CG vs DMC: +3.5% throughput, -23.5% TTFT, -12% mean ITL, -30% P99 ITL.

### Mixed CG Size Optimization

**CG19 → CG25 → CG29 progression:**

CG19 (19 sizes, ~1.51GB): `[1,4,8,16,32,64,128,256,384,512,768,1024,1280,1536,1792,2048,2560,3072,4096]`

Data-driven optimization: instrumented `_replay_prepare_mixed` with `SGLANG_LOG_MIXED_CG_PADDING=1`, collected 200 mixed CG replays per WL. Greedy selection by real waste reduction:

| Rank | Size | Waste saved | Reason |
|------|------|-------------|--------|
| 1 | **3584** | 72,704 (28%) | Splits 3072→4096 gap (1024) |
| 2 | **2816** | 18,688 (7%) | Splits 2560→3072 gap (512) |
| 3 | **2304** | 13,056 (5%) | Splits 2048→2560 gap (512) |
| 4 | 640 | 4,992 (2%) | Splits 512→768 gap (256) |
| 5 | 1408 | 4,736 (2%) | Splits 1280→1536 gap (256) |
| 6 | 1920 | 4,736 (2%) | Splits 1792→2048 gap (256) |

CG25 = CG19 + {3584, 2816, 2304, 640, 1408, 1920} = 25 sizes, ~1.79GB. Padding: 11% → 6.5%.
CG29 = CG25 + {3200, 3456, 3712, 3968} = 29 sizes, ~1.98GB. Fills 3072-4096 with 128-token granularity. Padding: 6.5% → 4.3%.

**CG evolution** (all measured at rate=8 except CG19/CG25 which used mixed rates):
- CG19 (19 sizes, ~1.51GB, padding 11%) → CG25 (+6 sizes, ~1.79GB, padding 6.5%) → **CG29** (+4 more in 3072-4096, ~1.98GB, **padding 4.3%**)
- CG24 experiment (no 4096): hurt WL1 by -6%. 4096 needed for large mixed batches.

### Server Stress Test Results: LPT+CG29 vs DMC (H100 80GB, Qwen3-8B, rate=8 all WLs)

* **DMC** = default mixed chunk with batch prefill kernel (no persistent, no mixed CG).
* **CG** = CUDA graph (CG29 = 29 sizes, max 4096, ~1.98GB, padding 4.3%).
* Commit: sglang `090b86f5`, flashinfer `02b068f0`. Script: `run_one_config.sh` / manual `bench_serving` at rate=8, N=400.

**Throughput (tok/s):**
| WL | Input | Output | LPT+CG29 total | LPT noCG total | DMC total | CG benefit | CG29 vs DMC |
|----|------:|-------:|----------------:|---------------:|--------------:|-----------:|------------:|
| 1 | 4096 | 1024 | **15,507** | 13,515 | 13,022 | **+14.7%** | **+19.1%** |
| 2 | 8192 | 2048 | **10,755** | 10,692 | 10,393 | +0.6% | **+3.5%** |
| 3 | 12000 | 1024 | **15,629** | 15,597 | 14,912 | +0.2% | **+4.8%** |
| 4 | 16384 | 1024 | **15,218** | 15,054 | 14,251 | +1.1% | **+6.8%** |
| 5 | 4096 | 4096 | **6,602** | 6,531 | 6,431 | +1.1% | **+2.7%** |
| 6 | 6000 | 6000 | **4,728** | 4,719 | 4,671 | +0.2% | **+1.2%** |

**Latency:**
| WL | CG29 TTFT med | noCG TTFT med | DMC TTFT med | CG29 Mean ITL | noCG Mean ITL | DMC Mean ITL | CG29 P99 ITL | noCG P99 ITL | DMC P99 ITL |
|----|---------------:|--------------:|-------------:|--------------:|--------------:|-------------:|-------------:|-------------:|------------:|
| 1 | **284ms** | 3,469ms | 5,511ms | **34ms** | 42ms | 44ms | **173ms** | 468ms | 536ms |
| 2 | **47,196ms** | 47,382ms | 49,826ms | **34ms** | **34ms** | 35ms | 231ms | **216ms** | 235ms |
| 3 | **43,542ms** | 43,134ms | 46,415ms | **44ms** | **44ms** | 46ms | 318ms | **244ms** | 263ms |
| 4 | **81,765ms** | 82,433ms | 88,004ms | **45ms** | **45ms** | 48ms | 296ms | **266ms** | 292ms |
| 5 | **57,874ms** | 58,435ms | 59,637ms | **28ms** | **28ms** | 28ms | 122ms | **103ms** | 117ms |
| 6 | **137,571ms** | 138,779ms | 140,771ms | **29ms** | **29ms** | 29ms | 114ms | **104ms** | 119ms |

**Key findings:**
1. **LPT+CG29 beats DMC on ALL workloads**: +19.1% (WL1), +6.8% (WL4), +4.8% (WL3), +3.5% (WL2), +2.7% (WL5), +1.2% (WL6).
2. **WL1 TTFT: CG29 284ms vs noCG 3,469ms = 12.2x improvement.** Mixed CG eliminates Python run() overhead (~2.4ms per mixed batch) for chunked prefill steps. nsys shows 3,132 plan() calls eliminated by CG replay.
3. **Mean ITL**: CG29 lower than DMC on all WLs (27-45ms vs 28-48ms).
4. **P99 ITL tradeoff**: CG29 wins WL1 (173ms vs noCG 468ms, DMC 536ms). noCG has lower P99 on WL2-6 (216-266ms vs CG's 231-318ms) — CG padding adds 10-13% extra compute on worst-case steps. The throughput gain outweighs this on average.

**ShareGPT (thinking, rate=4, max_out=4096, N=400):**
| Config | Total tok/s | TTFT med | Mean ITL | P99 ITL |
|--------|------------|----------|----------|---------|
| LPT+CG25 | **4,962** | **40,767ms** | **36ms** | **67ms** |
| LPT+CG29 | **4,975** | **40,756ms** | **36ms** | **67ms** |
| trueDMC | 4,794 | 53,298ms | 41ms | 96ms |
| **Δ CG29 vs DMC** | **+3.8%** | **-23.6%** | **-12%** | **-30%** |

CG25≈CG29 (identical within noise). First CG29 run was outlier from cold JIT cache.

**WL1-6 rerun** (flashinfer `d9c01c81` with ratio threshold, same sglang): all within 2% of previous results. Ratio threshold has zero measurable server overhead.
(CG25: sglang `090b86f5`, flashinfer `02b068f0`. CG29: flashinfer `d9c01c81`.)

### WL1-6: LPT+CG29 vs PP (prefill-prioritized, no mixed chunk) (rate=8, N=400)

PP = no `--enable-mixed-chunk`, WITH `--enable-piecewise-cuda-graph`, no persistent kernel. Prefill completes fully before decode interleaves.
flashinfer `d9c01c81`, sglang `857603c2a`.

| WL | In | Out | LPT+CG29 Total | PP+PCG Total | LPT TTFT med | PP TTFT med | LPT P99 ITL | PP P99 ITL |
|----|---:|----:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 4096 | 1024 | ~15,400 | ~15,100 | 291ms | **214ms** | **174ms** | 195ms |
| 2 | 8192 | 2048 | ~10,640 | ~10,300 | **47,471ms** | 52,597ms | 240ms | **219ms** |
| 3 | 12000 | 1024 | **~15,400** | ~14,200 | **42,236ms** | 69,194ms | **298ms** | 846ms |
| 4 | 16384 | 1024 | **~14,920** | ~13,400 | **81,348ms** | 165,429ms | **297ms** | 1,696ms |
| 5 | 4096 | 4096 | ~6,510 | ~6,510 | 58,189ms | 58,421ms | 124ms | **113ms** |
| 6 | 6000 | 6000 | ~4,730 | ~4,740 | 137,718ms | 139,702ms | 112ms | **108ms** |

**Key findings**: PP wins P99 ITL on output-heavy WL5-6 (no mixed-batch decode disruption). LPT+CG29 wins on input-heavy WL3-4 (chunked prefill + CG amortize large prefills, PP blocks decode during long prefills → massive P99 spikes). WL1: PP has lower TTFT (no CG capture overhead) but higher P99. CG makes no difference for PP (without mixed chunk, batches are pure prefill or decode).

### POD+CG vs LPT+CG29 (WL1, WL3, rate=8, N=400)

POD = `SGLANG_FLASHINFER_USE_POD=1` (separate prefill+decode kernels) + `--enable-piecewise-cuda-graph`.
flashinfer `d9c01c81`.

| WL | Config | Total tok/s | TTFT med | Mean ITL | P99 ITL |
|----|--------|:-----------:|:---------:|:---------:|:---------:|
| 1 | LPT+CG29 | **~15,400** | **291ms** | **34ms** | **174ms** |
| 1 | POD+CG | ~13,300 | 5,390ms | 43ms | 437ms |
| 3 | LPT+CG29 | **~15,400** | **42,236ms** | 43ms | 298ms |
| 3 | POD+CG | ~14,500 | 46,394ms | 45ms | **284ms** |

**LPT+CG29 dominates WL1**: 18x better TTFT (mixed CG eliminates per-batch Python plan() overhead for chunked prefill), 2.5x better P99. **WL3**: LPT has better throughput (+6%), POD has slightly better P99 (284 vs 298ms). POD's main overhead: double plan() calls + GPU-CPU sync for index splitting in pod_forward per layer.

**nsys overhead analysis** (WL1, 50 prompts, `pod_server.nsys-rep` vs `lpt_cg29_wl1.nsys-rep`):

| Metric | LPT+CG29 | POD+PCG | Note |
|--------|-----------|---------|------|
| Attention GPU time | 694ms (618 persistent + 76 paged) | 427ms (237 prefill + 113 POD-decode + 77 paged) | POD 38% less |
| cudaDeviceSynchronize | **2,812ms** (1,869 calls) | 1,162ms (2,448 calls) | plan() GPU→CPU sync |
| cudaStreamSynchronize | 0ms | **2,360ms** (3,334 calls) | **POD plan() GPU-CPU sync per layer** |
| cudaMemcpyAsync | 2,650ms | 3,312ms (+25%) | POD double plan indexing |
| Total sync overhead | 2,812ms | 3,522ms | POD 25% more total |

**POD's cudaStreamSynchronize (2,360ms) = 553% of its attention GPU time.** This is the dominant bottleneck — POD splits `qo_indptr_buf` into separate prefill/decode index arrays (cumsum, slice, `.to(device)`) per layer, requiring GPU-CPU sync each time. LPT reads the same `qo_indptr_buf` but passes it directly to the fused kernel without splitting — no per-layer sync needed.

### Nebius-SWE-like workload (random in=9053, out=3726, N=200)

flashinfer `d9c01c81`. Nebius-SWE median: 9K input, 3.7K output (2.4:1 prefill-heavy).

| Config | Rate | Throughput | TTFT med | Mean ITL | P99 ITL |
|--------|-----:|:---------:|:---------:|:---------:|:---------:|
| LPT+CG29 | 2 | 0.99 req/s | 340ms | 27.40ms | 155ms |
| DMC | 2 | 0.99 req/s | 340ms | 27.37ms | 155ms |
| PP+PCG | 2 | 0.98 req/s | 330ms | 27.58ms | 153ms |
| LPT+CG29 | 4 | 1.02 req/s | 11,014ms | 28.67ms | 164ms |
| DMC | 4 | 1.02 req/s | 11,046ms | 28.68ms | 164ms |
| PP+PCG | 4 | 1.02 req/s | 11,310ms | 28.81ms | 169ms |
| LPT+CG29 | 8 | 1.03 req/s | 18,575ms | 28.95ms | 184ms |

**All configs identical** — Nebius-SWE (9K+3.7K) is compute-bound at ~1 req/s. Scheduling strategy has no effect when prefill dominates and the system is fully saturated. Differences emerge only with shorter prefills (WL1-6) where decode overlapping matters.

### P99 ITL Lifecycle Analysis (WL1, 200 prompts, rate=8)

Per-request decode step tracking via `SGLANG_LOG_REQUEST_LIFECYCLE=1`. Compares batch conditions during P99-hitting steps vs normal steps.

| | LPT+CG29 P99 steps | LPT normal | PP+PCG P99 steps | PP normal |
|--|:---:|:---:|:---:|:---:|
| P99 ITL threshold | 133ms | — | 191ms | — |
| decode_bs | **91** | 55 | 88 | 87 |
| prefill_tokens | **5,153** | 3,476 | 2,893 | 2,896 |
| new_token_ratio | 0.160 | 0.240 | 0.143 | 0.144 |
| is_mixed (%) | **37%** | 1% | 0% | 0% |

**LPT P99 is driven by mixed batches with high decode_bs + high prefill_tokens** — 37% of P99 steps are mixed (vs 1% normal). High prefill admission (low ntr 0.16 vs 0.24) causes decode pressure. **Confirms CG over-admission hypothesis.**

**PP P99 steps have identical batch conditions to normal steps** — decode_bs, prefill_tokens, and ntr are the same. PP's higher P99 (191ms) is NOT from scheduling but from chunked prefill blocking decode within each step (no mixed-chunk overlap).

### PP WL3 P99 Regression Root Cause (lifecycle tracking)

| | LPT+CG29 WL3 | PP+PCG WL3 |
|--|:---:|:---:|
| P99 ITL | 306ms | **720ms** |
| Max ITL | 473ms | **21,644ms** |
| P99 decode_bs | 59 | 59 |
| P99 prefill_tokens | 4,420 | 4,185 |
| Worst spike cause | decode_bs=66, prefill=7677 | **decode_bs=64, prefill=259, ntr=0.632** |

**PP's worst spikes (21.6s!) happen early** (ntr=0.632) from decode starvation: after a request gets TTFT, the scheduler goes back to prefilling a backlog of newly-arrived requests (rate=8 × 12K input = continuous prefill work). Decode only runs when no prefills are pending, so already-prefilled requests starve for 21.6s while the scheduler drains the prefill queue. LPT's mixed chunk interleaves decode tokens within each prefill step, keeping max ITL at 473ms.

**Why PP works on WL5-6 but not WL3-4**: WL5-6 have 4K-6K input (≤1 chunk per request). The prefill backlog clears quickly, so decode starvation is brief. WL3-4 have 12K-16K input (2-3 chunks per request) — at rate=8, new prefill work arrives faster than the scheduler can clear it, causing prolonged decode starvation.

---

## Open TODOs

- [x] **Mixed CG padding threshold [INFEASIBLE]**: Swept `SGLANG_MIXED_CG_PADDING_PCT`=5/8/10% (clean rerun, no keepalive, rate=8).

  | WL | CG29 total | noCG P99 | pct=10% total | pct=10% TTFT | pct=10% P99 | pct=8% total | pct=8% P99 | pct=5% total | pct=5% P99 |
  |----|-----------|---------|--------------|-------------|------------|-------------|-----------|-------------|-----------|
  | 1 | **15,507** | 468ms | 14,380 (-7%) | 5,306ms (19x↑) | **215ms** ✓ | 14,021 (-10%) | **236ms** ✓ | 13,651 (-12%) | 499ms ✗ |
  | 2 | **10,755** | 216ms | 10,704 | 48,210ms | 223ms ✗ | 10,789 | 221ms ✗ | 10,780 | **216ms** ✓ |
  | 3 | **15,629** | 244ms | 15,890 | 42,357ms | 299ms ✗ | 15,916 | 294ms ✗ | 15,724 | 290ms ✗ |
  | 4 | **15,218** | 266ms | 15,181 | 81,509ms | 296ms ✗ | 15,131 | 299ms ✗ | 15,099 | 296ms ✗ |
  | 5 | **6,602** | 103ms | 6,559 | 58,713ms | 123ms ✗ | 6,540 | 123ms ✗ | 6,534 | 124ms ✗ |
  | 6 | **4,728** | 104ms | 4,710 | 138,526ms | 113ms ✗ | 4,718 | 112ms ✗ | 4,721 | 115ms ✗ |

  **All thresholds fail both constraints.** WL1 TTFT regresses 18-19x (CG skipped for prefill-heavy batches). WL2-6 P99 still exceeds noCG at every threshold — the padding is only 6% (CG29) but mixed-chunk scheduling inherently delays decode steps within padded batches. **Fundamental conflict**: TTFT needs CG for the same batches that cause P99 spikes.

  **Scheduler analysis** (`--plot-chunk-size`, WL1 only, rate=8 N=400): CG29 completes in ~76s vs noCG ~91s (17% faster). Prefill chunks: CG29 mean 2,781 vs noCG 3,200 tokens. `new_token_ratio` decays faster with CG29 in both wallclock and step-count space — CG processes more decode steps per second, each applying one decay. noCG shows a plateau at ratio ~0.4 (steps 75-120) where fewer decode steps occurred. CG29 reaches `min_new_token_ratio` (0.098) sooner → scheduler admits more prefill earlier → possible over-admission causing decode pressure. **TODO**: track P99-hitting requests to confirm over-admission is the mechanism (see task below).

  ![CG29 vs noCG scheduler WL1](./debug_kernel_server_perf/profile_logs/cg_vs_nocg_scheduler_wl1.png)

- [x] **Unify LPT profile workloads**: Done — Cases 1-9 (100 decode), split-KV decode/prefill.
- [x] **nsys profile LPT+CG29 vs trueDMC** (WL1, 50 prompts, rate=8):
  Files: `profile_logs/lpt_cg29_wl1.nsys-rep` (40MB), `profile_logs/trueDMC_wl1.nsys-rep` (35MB).

  | CUDA API | LPT+CG29 | trueDMC | Note |
  |----------|----------|---------|------|
  | cudaDeviceSynchronize | **2,812ms** (1,869 calls) | 1,181ms (2,176 calls) | plan() GPU→CPU sync (more calls in LPT due to mixed batches) |
  | cudaMemcpyAsync | 2,650ms (4,538 calls) | 2,639ms (5,036 calls) | ~same |
  | cudaLaunchKernel | 663ms (70,286 calls) | 670ms (56,213 calls) | LPT more launches, same total |
  | cudaGraphInstantiate | 997ms (1,624 calls) | 752ms (1,997 calls) | LPT has more CG instantiations (29 mixed + decode) |

  | Kernel | LPT+CG29 | trueDMC |
  |--------|----------|---------|
  | PersistentKernelTemplate | 618ms (2,088 calls, avg 296μs) | — |
  | BatchPrefillWithPagedKVCache | 76ms (2,592 calls, avg 29μs) | (included in nvjet totals) |
  | nvjet GEMM total | ~2.8s | ~1.6s |

  **Key finding**: LPT's `cudaDeviceSynchronize` (2.4x more than DMC) is from `plan()` GPU→CPU sync (`torch.cuda.synchronize()` in `attention.py:200` to copy qo_indptr/kv_indptr/kv_len_arr to CPU). NOT from cooperative launch (single CUDA stream → all SMs available). Mixed CG helps by skipping `plan()` entirely for cached batch sizes.
- [ ] **Investigate CG P99 ITL over-admission hypothesis**: CG's faster decode steps cause `new_token_ratio` to decay faster (more decay events per second) → scheduler admits more prefill sooner → decode pressure. Track requests that hit P99 ITL in CG29 vs noCG: log per-request lifecycle (admit time, each ITL, batch size at each step, prefill chunk size at each step). Determine if P99-hitting requests correlate with periods of high prefill admission (low `new_token_ratio` + large decode BS).
- [ ] **GPU-side LPT plan kernel**: Microbenchmark showed 0.15ms C++ plan. Verify in server context with nsys.
- [ ] Realistic workload analysis — see `tasks/realistic_workload_analysis.md`

---

## Implementation Details

### Design: FA3-style Runtime Dynamic Scheduler (v3)

1. `atomicAdd(tile_counter, 1)` → get global `tile_idx`
2. `tile_idx_to_work_tile(tile_idx)`: batch lookup via warp_prefix_sum → L2-aware LPT head/block mapping
3. Queue switching via FlippedSchedule: even CTAs start prefill, odd start decode
4. Split-KV: `(seq, head, chunk)` layout with per-q_tile chunk counts. INT_MAX sentinel for zero overhead when no split needed.

**Key files:**
- `flashinfer/include/flashinfer/attention/persistent.cuh` — `tile_idx_to_work_tile()`, split-KV code paths
- `flashinfer/include/flashinfer/attention/scheduler.cuh` — `TwoStageHolisticPlan`, kv_limit, split-KV planner
- `flashinfer/csrc/batch_attention.cu` — C++ run() path
- `sglang/python/sglang/srt/model_executor/cuda_graph_runner.py` — mixed CG capture/replay

### LPT Scheduler Fixes
1. Removed `|| enable_cuda_graph` — dynamic scheduler only with `flipped_schedule`
2. Merged two for-task loops into single loop with if/else
3. Fixed causal kv_end clamping
4. CG forces `flipped_schedule=true` in C++ plan
5. Zero work_indptr for dynamic tasks prevents CG deadlock
6. Added `enable_cuda_graph` + `cuda_graph_max_bs` to `plan()`

### Split-KV Implementation
- **Tile layout**: `(seq, head, chunk)` with per-q_tile chunk counts
- **Key bugs fixed**: unsigned wrap in num_kv_chunks, kv_end not clamped to causal effective KV, num_m_blocks from wrong source, partial_o_offset per-sequence vs per-q_tile, stale JIT cache
- All 6 test cases pass

### Decode Overhead Fix
nsys profiling found `BatchAttention.plan()` running on every decode step when `use_persistent=True`, plus no `fast_decode_plan` optimization. Fix: decode steps use regular `BatchDecodeWithPagedKVCacheWrapper`.

### Mixed CG Bugs Fixed (6 total)
1. Missing kv_indices copy in CG plan()
2. plan_info[-1] wrong field after adding dyn_scalars_offset
3. Module auto-switch breaks CG replay
4. Padding overflow in `_replay_prepare_mixed`
5. Shared indptr too small for large mixed CG
6. LPT CG pointer initialization

### How CG works with BatchAttention
- **Frozen at capture**: Pointers to workspace buffers, kv_indices, q, out, lse, tile_counter, dyn_scalars
- **Updated at replay**: Data at fixed addresses updated by `plan()` before `graph.replay()`
- **Workspace layout invariant**: `max_total_num_works = 65536` (constant)
- **LPT at runtime**: Kernel reads `dyn_scalars[0]` (num_seqs) — mixed batch → LPT, pure decode → static

---

## Historical Debugging Notes

### Early Findings (pre-LPT)
1. **Decode path using persistent kernel unnecessarily** (commit 637d3889): Fixed — only use persistent for mixed batches. +7.2% throughput.
2. **torch.arange allocation** (commit 34d5c728): Cached buffer.
3. **POD attention launch overhead** (commit 6244007a): Cached kv_indices, reused extend_lens. +3.3% throughput, -31% TTFT.
4. **POD vs DMC server throughput**: POD ~0.4% slower despite faster kernel — double plan(), GPU-CPU sync per layer, extra index computation, cudaMemsetAsync per layer.
5. **Why prefill-prio has lowest P99 ITL**: Separation of concerns — pure decode steps are fast and consistent. Mixed-chunk makes every step slow.

### GPU Launch Gap Analysis
| Config | Gap % |
|--------|-------|
| Default | 65.7% |
| Persistent (fixed) | 62.6% |
| POD (fixed) | 85.4% |

### v1/v2 Progress
- v1 (static LPT atomic): Cases 6-7 improved ~10%
- v2 (LPT sorting + relaxed FlippedSchedule): +4-27% over baseline, still 2x slower than POD → led to v3

### Issues
1. v2 static LPT + FlippedSchedule hurts small workloads
2. Tile counter needs cudaMemsetAsync before each run()
3. Non-contiguous sequences need separate dyn_qo_len array
4. GPU preempted during B200 development

### Questions & Answers
1. KV splitting: pre-compute partial output slots per (seq, head, chunk) triple
2. atomicAdd stride: sequential `atomicAdd(counter, 1)` optimal (tested FA3 pattern, worse)
3. FlippedSchedule + dynamic: FlippedSchedule controls queue start order, dynamic fetches via atomicAdd
4. L2 budget: query `cudaDevAttrL2CacheSize` at runtime
5. L2-aware grouping: essential (user confirmed)
6. 2 CTAs/SM: shared memory constraint

## GPU Keepalive
```bash
bash ../VibeCUDA/scripts/gpu_keepalive.sh start   # hold GPU during setup
bash ../VibeCUDA/scripts/gpu_keepalive.sh stop    # STOP before benchmarks!
bash ../VibeCUDA/scripts/gpu_keepalive.sh start   # restart after benchmarks
```
**IMPORTANT:** gpu_keepalive corrupts benchmark results. Cooperative launch (persistent kernel) is especially sensitive to SM contention.

## Profiling Configurations (legacy)
| Config | nsys trace | sqlite |
|--------|------------|--------|
| Default | `default_bt.nsys-rep` | `default_kernel.sqlite` |
| Persistent | `persistent_bt.nsys-rep` | `persistent_kernel.sqlite` |
| POD | `pod_bt.nsys-rep` | `pod_kernel.sqlite` |
