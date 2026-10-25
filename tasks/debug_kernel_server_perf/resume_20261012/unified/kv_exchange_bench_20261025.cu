// FA3 multicast, step 1 (A5.5): can a 2-CTA cluster deliver paged K tiles to both CTAs faster than each CTA gathering
// the full tile itself? Page size 1, so every row is a page-table gather (cp.async, as flash-attention's paged non-TMA
// path); Hopper TMA cannot gather, so multicast here means: each CTA of the pair gathers HALF the rows of a 128-row
// tile into its own shared memory, then pushes that half into the peer's stage (st.async.shared::cluster with
// mbarrier::complete_tx), so each CTA issues half the global/L2 loads. The consumer of each CTA waits on its stage's
// mbarrier, which completes when the local half has landed (cp.async arrive) and the peer's bytes have arrived (tx).
// Stage reuse: a CTA writes into the peer's stage s only after the peer released it (the peer's consumer arrives on
// this CTA's "peer empty" mbarrier remotely).
//   modes: 0 = each CTA gathers the full tile (FA3 today), 1 = half gather + DSMEM push (multicast emulation)
// The consumer reads the tile from shared memory (a 128 x 128 bf16 reduction per warp) so loads cannot be elided.
// Claude, 2026-10-25.
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

constexpr int kRows = 128;        // kBlockN
constexpr int kD = 128;           // head dim
constexpr int kRowBytes = kD * 2;  // 256 B
constexpr int kTileBytes = kRows * kRowBytes;  // 32 KB
#ifndef KVX_STAGES
#define KVX_STAGES 2
#endif
constexpr int kStages = KVX_STAGES;  // pipeline depth (2 or 4; 4 x 32 KB fits one CTA per SM)
constexpr int kProducer = 128;    // producer threads (flash-attention's paged path uses 128)
constexpr int kConsumer = 256;    // two consumer warpgroups
constexpr int kThreads = kProducer + kConsumer;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
// Arrive on the barrier at the same offset in CTA `rank` of the cluster.
__device__ __forceinline__ void mbar_arrive_remote(uint64_t* bar, uint32_t rank) {
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(smem_u32(bar)), "r"(rank));
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(remote) : "memory");
}
__device__ __forceinline__ void cp_async_arrive_noinc(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile(
      "{\n .reg .pred p;\n LAB_WAIT:\n"
      " mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra LAB_WAIT;\n}\n" ::"r"(smem_u32(bar)), "r"(phase) : "memory");
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}
// 16-byte DSMEM store into CTA `rank`'s shared memory at the address matching local `dst`, counting 16 bytes on the
// remote CTA's mbarrier `bar` (same offset there).
__device__ __forceinline__ void st_async16(void* dst, uint4 v, uint64_t* bar, uint32_t rank) {
  uint32_t rdst, rbar;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(smem_u32(dst)), "r"(rank));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rbar) : "r"(smem_u32(bar)), "r"(rank));
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(rdst),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(rbar) : "memory");
}

// Bulk copy of `bytes` from this CTA's shared memory to the same offset in CTA `rank`, counted on the remote mbarrier.
__device__ __forceinline__ void bulk_push(void* src, uint32_t bytes, uint64_t* bar, uint32_t rank) {
  uint32_t rdst, rbar;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rdst) : "r"(smem_u32(src)), "r"(rank));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rbar) : "r"(smem_u32(bar)), "r"(rank));
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(rdst),
               "r"(smem_u32(src)), "r"(bytes), "r"(rbar) : "memory");
}
__device__ __forceinline__ void producer_bar() { asm volatile("bar.sync 1, %0;" ::"n"(kProducer) : "memory"); }

struct alignas(128) Smem {
  __nv_bfloat16 tile[kStages][kRows * kD];
  uint64_t full[kStages];
  uint64_t empty[kStages];       // local consumers released the stage (count: consumer warps)
  uint64_t peer_empty[kStages];  // the PEER's consumers released ITS stage (count: peer consumer warps), mode 1
};

// Each cluster pair (or, in mode 0, each CTA) walks n_blocks tiles of one request's KV through the page table.
// MODE 0: grid of independent CTAs (cluster dims 2 so placement matches), both CTAs of a pair gather the full tile.
template <int MODE>
__global__ void __launch_bounds__(kThreads, 1) kv_exchange_kernel(const __nv_bfloat16* __restrict__ k,
                                                                 const int32_t* __restrict__ page_table,
                                                                 int32_t n_blocks, int32_t pages_per_pair,
                                                                 float* __restrict__ sink) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  cg::cluster_group cluster = cg::this_cluster();
  const uint32_t rank = cluster.block_rank();
  const uint32_t peer = rank ^ 1u;
  const int pair = blockIdx.x / 2;
  const int32_t* pt = page_table + int64_t(pair) * pages_per_pair;
  const int tid = threadIdx.x;
  constexpr int kConsumerWarps = kConsumer / 32;
  if (tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      // full: local producer threads arrive via cp.async (noinc counts one per thread), plus the remote tx bytes.
      mbar_init(&sm.full[s], MODE == 0 ? kProducer : kProducer);
      mbar_init(&sm.empty[s], kConsumerWarps);
      mbar_init(&sm.peer_empty[s], kConsumerWarps);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cluster.sync();
  if (tid < kProducer) {
    // Producer. MODE 0: the CTA gathers all 128 rows. MODE 1: it gathers rows [rank * 64, rank * 64 + 64), a contiguous
    // 16 KB half, and one thread bulk-copies that half into the peer's stage; the copy of stage n - 1 is issued after
    // the gathers of stage n, so the push overlaps the next gather.
    constexpr int kChunks = kRowBytes / 16;  // 16 chunks of 16 B per row
    const int row_lo = MODE == 1 ? int(rank) * (kRows / 2) : 0;
    const int nrows = MODE == 1 ? kRows / 2 : kRows;
    auto push = [&](int m) {  // stage of tile m: all its local gathers landed (wait_group + barrier below)
      const int s = m % kStages;
      if (tid == 0) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // generic-proxy writes -> async-proxy read
        bulk_push(sm.tile[s] + row_lo * kD, kTileBytes / 2, &sm.full[s], peer);
      }
    };
    for (int n = 0; n < n_blocks; ++n) {
      const int s = n % kStages;
      const uint32_t ph = (n / kStages) & 1u;
      if (n >= kStages) {
        mbar_wait(&sm.empty[s], ph ^ 1u);
        if (MODE == 1) mbar_wait(&sm.peer_empty[s], ph ^ 1u);
      }
      if (MODE == 1 && tid == 0) mbar_expect_tx(&sm.full[s], kTileBytes / 2);
      __nv_bfloat16* stage = sm.tile[s];
      for (int i = tid; i < nrows * kChunks; i += kProducer) {
        const int r = row_lo + i / kChunks, c = i % kChunks;
        cp_async16(stage + r * kD + c * 8, k + int64_t(pt[n * kRows + r]) * kD + c * 8);
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
      cp_async_arrive_noinc(&sm.full[s]);
      if (MODE == 1 && n > 0) {
        asm volatile("cp.async.wait_group 1;" ::: "memory");  // tile n - 1's gathers by this thread have landed
        producer_bar();                                        // ... and every producer thread's
        push(n - 1);
      }
    }
    if (MODE == 1 && n_blocks > 0) {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      producer_bar();
      push(n_blocks - 1);
    }
  } else {
    // Consumer: each warp reduces the whole tile (stands in for the QK / PV MMAs' smem reads).
    const int cw = (tid - kProducer) / 32, lane = tid % 32;
    float acc = 0.f;
    for (int n = 0; n < n_blocks; ++n) {
      const int s = n % kStages;
      mbar_wait(&sm.full[s], (n / kStages) & 1u);
      const uint4* t = reinterpret_cast<const uint4*>(sm.tile[s]);
      for (int i = lane + cw * 32; i < kTileBytes / 16; i += kConsumer) {
        const uint4 v = t[i];
        acc += __uint_as_float(v.x & 0x7f7fffffu) * 1e-30f;
      }
      __syncwarp();
      if (lane == 0) {
        mbar_arrive(&sm.empty[s]);
        if (MODE == 1) mbar_arrive_remote(&sm.peer_empty[s], peer);
      }
    }
    if (lane == 0) atomicAdd(sink, acc);
  }
  cluster.sync();  // no CTA exits while its peer may still write into it
}

double run(at::Tensor k, at::Tensor page_table, int64_t n_blocks, int64_t pairs, int64_t mode, int64_t iters,
           at::Tensor sink) {
  TORCH_CHECK(k.is_cuda() && k.scalar_type() == at::kBFloat16 && k.size(1) == kD, "k: [tokens, 128] bf16");
  const int64_t pages_per_pair = page_table.numel() / pairs;
  TORCH_CHECK(pages_per_pair >= n_blocks * kRows, "page table too short");
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto kern = mode == 0 ? kv_exchange_kernel<0> : kv_exchange_kernel<1>;
  const int smem = int(sizeof(Smem));
  TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess, "smem");
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(uint32_t(pairs * 2));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  auto launch = [&] {
    auto st = cudaLaunchKernelEx(&cfg, kern, static_cast<const __nv_bfloat16*>(k.data_ptr()),
                                 static_cast<const int32_t*>(page_table.data_ptr()), int32_t(n_blocks),
                                 int32_t(pages_per_pair), sink.data_ptr<float>());
    TORCH_CHECK(st == cudaSuccess, "launch: ", cudaGetErrorString(st));
  };
  for (int i = 0; i < 3; ++i) launch();
  cudaEventRecord(a, stream);
  for (int i = 0; i < iters; ++i) launch();
  cudaEventRecord(b, stream);
  cudaEventSynchronize(b);
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kernel failed");
  return double(ms) / iters;
}

// Copy check: the consumer-visible tile of every stage equals the gathered rows (mode 1 must deliver both halves).
__global__ void __launch_bounds__(kThreads, 1) kv_exchange_check(const __nv_bfloat16* __restrict__ k,
                                                                const int32_t* __restrict__ page_table,
                                                                int32_t pages_per_pair, int32_t* __restrict__ bad) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  cg::cluster_group cluster = cg::this_cluster();
  const uint32_t rank = cluster.block_rank(), peer = rank ^ 1u;
  const int pair = blockIdx.x / 2, tid = threadIdx.x;
  const int32_t* pt = page_table + int64_t(pair) * pages_per_pair;
  if (tid == 0) {
    mbar_init(&sm.full[0], kProducer);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cluster.sync();
  if (tid < kProducer) {
    // The benchmark's MODE 1 path for one tile: gather the contiguous half, then bulk-push it to the peer.
    constexpr int kChunks = kRowBytes / 16;
    const int row_lo = int(rank) * (kRows / 2);
    if (tid == 0) mbar_expect_tx(&sm.full[0], kTileBytes / 2);
    for (int i = tid; i < (kRows / 2) * kChunks; i += kProducer) {
      const int r = row_lo + i / kChunks, c = i % kChunks;
      cp_async16(sm.tile[0] + r * kD + c * 8, k + int64_t(pt[r]) * kD + c * 8);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    cp_async_arrive_noinc(&sm.full[0]);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    producer_bar();
    if (tid == 0) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      bulk_push(sm.tile[0] + row_lo * kD, kTileBytes / 2, &sm.full[0], peer);
    }
  }
  mbar_wait(&sm.full[0], 0);
  for (int i = tid; i < kRows * kD; i += kThreads) {
    const int r = i / kD, c = i % kD;
    if (__bfloat162float(sm.tile[0][i]) != __bfloat162float(k[int64_t(pt[r]) * kD + c])) atomicAdd(bad, 1);
  }
  cluster.sync();
}

int64_t check(at::Tensor k, at::Tensor page_table, int64_t pairs) {
  auto bad = torch::zeros({1}, torch::dtype(torch::kInt32).device(k.device()));
  const int smem = int(sizeof(Smem));
  TORCH_CHECK(cudaFuncSetAttribute(kv_exchange_check, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess, "smem");
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(uint32_t(pairs * 2));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  auto st = cudaLaunchKernelEx(&cfg, kv_exchange_check, static_cast<const __nv_bfloat16*>(k.data_ptr()),
                               static_cast<const int32_t*>(page_table.data_ptr()), int32_t(page_table.numel() / pairs),
                               bad.data_ptr<int>());
  TORCH_CHECK(st == cudaSuccess, "check launch: ", cudaGetErrorString(st));
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess, "check kernel failed");
  return bad.item<int>();
}

TORCH_LIBRARY(kv_exchange, m) {
  m.def("run", &run);
  m.def("check", &check);
}
