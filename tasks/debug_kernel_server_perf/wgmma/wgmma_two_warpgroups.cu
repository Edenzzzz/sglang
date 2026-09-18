// Do two warpgroups on ONE SM overlap when they issue different wgmma shapes?
// Within a warpgroup the answer is no (wgmma_overlap.cu: costs add exactly).
// This asks whether the SM's tensor cores can be shared: warpgroup 0 issues
// large tiles, warpgroup 1 small tiles, concurrently vs alone.
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
extern "C" __global__ void bench2(int mode, int iters, float* out, long long* cyc) {
  extern __shared__ __nv_bfloat16 smem[];
  for (int i = threadIdx.x; i < 8192; i += blockDim.x) smem[i] = __float2bfloat16(0.01f);
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  uint64_t da = make_desc(&smem[0], 16, 256), db = make_desc(&smem[4096], 16, 256);
  int wg = threadIdx.x / 128;
  bool do_l = (wg == 0) && (mode == 0 || mode == 1);
  bool do_s = (wg == 1) && (mode == 0 || mode == 2);
  float dl0, dl1, dl2, dl3, dl4, dl5, dl6, dl7, dl8, dl9, dl10, dl11, dl12, dl13, dl14, dl15, dl16, dl17, dl18, dl19, dl20, dl21, dl22, dl23, dl24, dl25, dl26, dl27, dl28, dl29, dl30, dl31;
  float ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7;
  __syncthreads();
  long long t0 = clock64();
  if (do_l) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    wgmma_n64_sd0(dl0, dl1, dl2, dl3, dl4, dl5, dl6, dl7, dl8, dl9, dl10, dl11, dl12, dl13, dl14, dl15, dl16, dl17, dl18, dl19, dl20, dl21, dl22, dl23, dl24, dl25, dl26, dl27, dl28, dl29, dl30, dl31, da, db);
    for (int i = 1; i < iters; ++i) { wgmma_n64_sd1(dl0, dl1, dl2, dl3, dl4, dl5, dl6, dl7, dl8, dl9, dl10, dl11, dl12, dl13, dl14, dl15, dl16, dl17, dl18, dl19, dl20, dl21, dl22, dl23, dl24, dl25, dl26, dl27, dl28, dl29, dl30, dl31, da, db); }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  }
  if (do_s) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    wgmma_n16_sd0(ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7, da, db);
    for (int i = 1; i < iters; ++i) { wgmma_n16_sd1(ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7, da, db); }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  }
  long long t1 = clock64();
  float s = 0.f;
  if (do_l) s += dl0 + dl1 + dl2 + dl3 + dl4 + dl5 + dl6 + dl7 + dl8 + dl9 + dl10 + dl11 + dl12 + dl13 + dl14 + dl15 + dl16 + dl17 + dl18 + dl19 + dl20 + dl21 + dl22 + dl23 + dl24 + dl25 + dl26 + dl27 + dl28 + dl29 + dl30 + dl31;
  if (do_s) s += ds0 + ds1 + ds2 + ds3 + ds4 + ds5 + ds6 + ds7;
  if (threadIdx.x == 0)   { out[0] = s; cyc[0] = t1 - t0; }
  if (threadIdx.x == 128) { out[1] = s; cyc[1] = t1 - t0; }
}
int main(int argc, char** argv) {
  int iters = argc>1 ? atoi(argv[1]) : 128;
  float* out; long long* cyc; long long h[2];
  cudaMalloc(&out, 2*sizeof(float)); cudaMalloc(&cyc, 2*sizeof(long long));
  size_t smem = 16384*sizeof(__nv_bfloat16);
  cudaFuncSetAttribute(bench2, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  const char* nm[3] = {"wg0 large || wg1 small", "wg0 large alone", "wg1 small alone"};
  long long bl[3][2];
  for (int m=0;m<3;++m){
    bl[m][0]=bl[m][1]=-1;
    for(int w=0;w<3;++w){bench2<<<1,256,smem>>>(m,iters,out,cyc);cudaDeviceSynchronize();}
    for(int r=0;r<30;++r){
      bench2<<<1,256,smem>>>(m,iters,out,cyc); cudaDeviceSynchronize();
      cudaMemcpy(h,cyc,2*sizeof(long long),cudaMemcpyDeviceToHost);
      for(int k=0;k<2;++k) if(bl[m][k]<0||h[k]<bl[m][k]) bl[m][k]=h[k];
    }
  }
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(e));return 1;}
  printf("iters=%d, one block of 2 warpgroups on one SM, min of 30\n", iters);
  for(int m=0;m<3;++m) printf("  %-24s wg0 %7lld   wg1 %7lld  (block %7lld)\n",
      nm[m], bl[m][0], bl[m][1], bl[m][0]>bl[m][1]?bl[m][0]:bl[m][1]);
  long long alone_l = bl[1][0], alone_s = bl[2][1];
  long long conc = bl[0][0] > bl[0][1] ? bl[0][0] : bl[0][1];
  printf("  ---\n  serial (large then small) : %7lld\n", alone_l + alone_s);
  printf("  concurrent (both wgs)     : %7lld   => %+.1f%% vs serial\n",
         conc, 100.0*(double(conc)/double(alone_l+alone_s)-1));
  return 0;
}
