// Does non-MMA work run free underneath an in-flight wgmma, and do SMALL
// tiles leave more slack than large ones?
//
// wgmma is async: after issue the warpgroup keeps executing until wait_group,
// so CUDA-core work placed between issues can in principle hide under the
// tensor core. Small tiles cost 20 cycles for a quarter of a 32-cycle large
// tile's work (~16 + N/4), so if that fixed overhead is idle issue time,
// small tiles should hide MORE per instruction.
//
// LARGE/KIND/NOPS are template parameters: a runtime branch inside the issue
// loop blocks unrolling and inflates the wgmma-only baseline 3x (32 -> 105
// cycles), which silently destroys the comparison.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_bf16.h>
__device__ __forceinline__ uint64_t make_desc(void* p, uint32_t lbo, uint32_t sbo) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  uint64_t d = 0;
  d |= uint64_t((a>>4)&0x3FFF); d |= uint64_t((lbo>>4)&0x3FFF)<<16; d |= uint64_t((sbo>>4)&0x3FFF)<<32;
  return d;
}

__device__ __forceinline__ void wgmma_n64_sd0(float& d0, float& d1, float& d2, float& d3, float& d4, float& d5, float& d6, float& d7, float& d8, float& d9, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15, float& d16, float& d17, float& d18, float& d19, float& d20, float& d21, float& d22, float& d23, float& d24, float& d25, float& d26, float& d27, float& d28, float& d29, float& d30, float& d31, uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 0, 1, 1, 0, 0;\n" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31) : "l"(da), "l"(db));
}
__device__ __forceinline__ void wgmma_n64_sd1(float& d0, float& d1, float& d2, float& d3, float& d4, float& d5, float& d6, float& d7, float& d8, float& d9, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15, float& d16, float& d17, float& d18, float& d19, float& d20, float& d21, float& d22, float& d23, float& d24, float& d25, float& d26, float& d27, float& d28, float& d29, float& d30, float& d31, uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 1, 1, 1, 0, 0;\n" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7), "+f"(d8), "+f"(d9), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31) : "l"(da), "l"(db));
}
__device__ __forceinline__ void wgmma_n16_sd0(float& d0, float& d1, float& d2, float& d3, float& d4, float& d5, float& d6, float& d7, uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 0, 1, 1, 0, 0;\n" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7) : "l"(da), "l"(db));
}
__device__ __forceinline__ void wgmma_n16_sd1(float& d0, float& d1, float& d2, float& d3, float& d4, float& d5, float& d6, float& d7, uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 1, 1, 1, 0, 0;\n" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7) : "l"(da), "l"(db));
}

#define FMA8(x) \
  x##0=fmaf(x##0,1.0000001f,0.5f); x##1=fmaf(x##1,1.0000001f,0.5f); \
  x##2=fmaf(x##2,1.0000001f,0.5f); x##3=fmaf(x##3,1.0000001f,0.5f); \
  x##4=fmaf(x##4,1.0000001f,0.5f); x##5=fmaf(x##5,1.0000001f,0.5f); \
  x##6=fmaf(x##6,1.0000001f,0.5f); x##7=fmaf(x##7,1.0000001f,0.5f);
#define EXP8(x) \
  x##0=exp2f(x##0); x##1=exp2f(x##1); x##2=exp2f(x##2); x##3=exp2f(x##3); \
  x##4=exp2f(x##4); x##5=exp2f(x##5); x##6=exp2f(x##6); x##7=exp2f(x##7);

// KIND: 0 none, 1 FFMA (CUDA core), 2 exp2 (MUFU/SFU -- softmax's hot op)
template <int LARGE, int KIND, int NOPS>
__global__ void bench3(int iters, float* out, long long* cyc) {
  extern __shared__ __nv_bfloat16 smem[];
  for (int i = threadIdx.x; i < 8192; i += blockDim.x) smem[i] = __float2bfloat16(0.01f);
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  uint64_t da = make_desc(&smem[0], 16, 256), db = make_desc(&smem[4096], 16, 256);
  float dl0, dl1, dl2, dl3, dl4, dl5, dl6, dl7, dl8, dl9, dl10, dl11, dl12, dl13, dl14, dl15, dl16, dl17, dl18, dl19, dl20, dl21, dl22, dl23, dl24, dl25, dl26, dl27, dl28, dl29, dl30, dl31;
  float ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7;
  float y0=threadIdx.x*1e-6f, y1=y0+1e-6f, y2=y0+2e-6f, y3=y0+3e-6f,
        y4=y0+4e-6f, y5=y0+5e-6f, y6=y0+6e-6f, y7=y0+7e-6f;
  __syncthreads();
  long long t0 = clock64();
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  if (LARGE) { wgmma_n64_sd0(dl0, dl1, dl2, dl3, dl4, dl5, dl6, dl7, dl8, dl9, dl10, dl11, dl12, dl13, dl14, dl15, dl16, dl17, dl18, dl19, dl20, dl21, dl22, dl23, dl24, dl25, dl26, dl27, dl28, dl29, dl30, dl31, da, db); } else { wgmma_n16_sd0(ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7, da, db); }
  for (int i = 1; i < iters; ++i) {
    if (LARGE) { wgmma_n64_sd1(dl0, dl1, dl2, dl3, dl4, dl5, dl6, dl7, dl8, dl9, dl10, dl11, dl12, dl13, dl14, dl15, dl16, dl17, dl18, dl19, dl20, dl21, dl22, dl23, dl24, dl25, dl26, dl27, dl28, dl29, dl30, dl31, da, db); } else { wgmma_n16_sd1(ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7, da, db); }
#pragma unroll
    for (int k = 0; k < NOPS; ++k) {
      if (KIND == 1) { FMA8(y) } else if (KIND == 2) { EXP8(y) }
    }
  }
  if (LARGE) { wgmma_n16_sd0(ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7, da, db); } else { wgmma_n64_sd0(dl0, dl1, dl2, dl3, dl4, dl5, dl6, dl7, dl8, dl9, dl10, dl11, dl12, dl13, dl14, dl15, dl16, dl17, dl18, dl19, dl20, dl21, dl22, dl23, dl24, dl25, dl26, dl27, dl28, dl29, dl30, dl31, da, db); }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  long long t1 = clock64();
  float s = (dl0 + dl1 + dl2 + dl3 + dl4 + dl5 + dl6 + dl7 + dl8 + dl9 + dl10 + dl11 + dl12 + dl13 + dl14 + dl15 + dl16 + dl17 + dl18 + dl19 + dl20 + dl21 + dl22 + dl23 + dl24 + dl25 + dl26 + dl27 + dl28 + dl29 + dl30 + dl31) + (ds0 + ds1 + ds2 + ds3 + ds4 + ds5 + ds6 + ds7) + y0+y1+y2+y3+y4+y5+y6+y7;
  if (threadIdx.x == 0) { out[0] = s; cyc[0] = t1 - t0; }
}

static float* g_out; static long long* g_cyc; static size_t g_smem;
template <int LARGE,int KIND,int NOPS> static double slope() {
  long long v, a=-1, b=-1;
  cudaFuncSetAttribute(bench3<LARGE,KIND,NOPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, g_smem);
  for (int pass=0; pass<2; ++pass) {          // marginal: (T(256)-T(128))/128
    int it = pass ? 256 : 128; long long best=-1;
    for(int w=0;w<3;++w){bench3<LARGE,KIND,NOPS><<<1,128,g_smem>>>(it,g_out,g_cyc);cudaDeviceSynchronize();}
    for(int r=0;r<25;++r){bench3<LARGE,KIND,NOPS><<<1,128,g_smem>>>(it,g_out,g_cyc);cudaDeviceSynchronize();
      cudaMemcpy(&v,g_cyc,sizeof(long long),cudaMemcpyDeviceToHost); if(best<0||v<best)best=v;}
    (pass?b:a)=best;
  }
  return double(b-a)/128.0;
}
int main() {
  cudaMalloc(&g_out,sizeof(float)); cudaMalloc(&g_cyc,sizeof(long long));
  g_smem = 16384*sizeof(__nv_bfloat16);
  printf("marginal cycles per wgmma, co-issued CUDA-core work between issues\n");
  printf("%-26s %8s %8s %8s %8s %8s\n","","none","8 ops","16 ops","32 ops","64 ops");
  printf("%-26s %8.2f %8.2f %8.2f %8.2f %8.2f\n","large n64 + FFMA",
      slope<1,0,0>(), slope<1,1,1>(), slope<1,1,2>(), slope<1,1,4>(), slope<1,1,8>());
  printf("%-26s %8.2f %8.2f %8.2f %8.2f %8.2f\n","large n64 + exp2",
      slope<1,0,0>(), slope<1,2,1>(), slope<1,2,2>(), slope<1,2,4>(), slope<1,2,8>());
  printf("%-26s %8.2f %8.2f %8.2f %8.2f %8.2f\n","small n16 + FFMA",
      slope<0,0,0>(), slope<0,1,1>(), slope<0,1,2>(), slope<0,1,4>(), slope<0,1,8>());
  printf("%-26s %8.2f %8.2f %8.2f %8.2f %8.2f\n","small n16 + exp2",
      slope<0,0,0>(), slope<0,2,1>(), slope<0,2,2>(), slope<0,2,4>(), slope<0,2,8>());
  return 0;
}
