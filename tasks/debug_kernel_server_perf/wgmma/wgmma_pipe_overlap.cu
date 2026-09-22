// Can a warpgroup doing softmax (exp2/MUFU) run free alongside a warpgroup
// doing wgmma? This is FA3's ping-pong / warp-specialization premise.
//
// Established already: two warpgroups both issuing wgmma do NOT overlap (the
// tensor core is one serial per-SM resource), and within a warpgroup exp2
// does not hide under an in-flight wgmma. Different pipes on different
// warpgroups is the remaining case -- and the one FA3 is built on.
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
#define EXP8(x) \
  x##0=exp2f(x##0); x##1=exp2f(x##1); x##2=exp2f(x##2); x##3=exp2f(x##3); \
  x##4=exp2f(x##4); x##5=exp2f(x##5); x##6=exp2f(x##6); x##7=exp2f(x##7);

// MODE 0: wg0 wgmma  ||  wg1 exp2      1: wg0 wgmma alone      2: wg1 exp2 alone
template <int MODE, int NEXP>
__global__ void bench4(int iters, float* out, long long* cyc) {
  extern __shared__ __nv_bfloat16 smem[];
  for (int i = threadIdx.x; i < 8192; i += blockDim.x) smem[i] = __float2bfloat16(0.01f);
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  uint64_t da = make_desc(&smem[0], 16, 256), db = make_desc(&smem[4096], 16, 256);
  int wg = threadIdx.x / 128;
  bool do_mma  = (wg == 0) && (MODE == 0 || MODE == 1);
  bool do_soft = (wg == 1) && (MODE == 0 || MODE == 2);
  float dl0, dl1, dl2, dl3, dl4, dl5, dl6, dl7, dl8, dl9, dl10, dl11, dl12, dl13, dl14, dl15, dl16, dl17, dl18, dl19, dl20, dl21, dl22, dl23, dl24, dl25, dl26, dl27, dl28, dl29, dl30, dl31;
  float y0=threadIdx.x*1e-6f,y1=y0+1e-6f,y2=y0+2e-6f,y3=y0+3e-6f,
        y4=y0+4e-6f,y5=y0+5e-6f,y6=y0+6e-6f,y7=y0+7e-6f;
  __syncthreads();
  long long t0 = clock64();
  if (do_mma) {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    wgmma_n64_sd0(dl0, dl1, dl2, dl3, dl4, dl5, dl6, dl7, dl8, dl9, dl10, dl11, dl12, dl13, dl14, dl15, dl16, dl17, dl18, dl19, dl20, dl21, dl22, dl23, dl24, dl25, dl26, dl27, dl28, dl29, dl30, dl31, da, db);
    for (int i = 1; i < iters; ++i) { wgmma_n64_sd1(dl0, dl1, dl2, dl3, dl4, dl5, dl6, dl7, dl8, dl9, dl10, dl11, dl12, dl13, dl14, dl15, dl16, dl17, dl18, dl19, dl20, dl21, dl22, dl23, dl24, dl25, dl26, dl27, dl28, dl29, dl30, dl31, da, db); }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  }
  if (do_soft) {
    for (int i = 0; i < iters; ++i) {
#pragma unroll
      for (int k = 0; k < NEXP; ++k) { EXP8(y) }
    }
  }
  long long t1 = clock64();
  float s = y0+y1+y2+y3+y4+y5+y6+y7;
  if (do_mma) s += dl0 + dl1 + dl2 + dl3 + dl4 + dl5 + dl6 + dl7 + dl8 + dl9 + dl10 + dl11 + dl12 + dl13 + dl14 + dl15 + dl16 + dl17 + dl18 + dl19 + dl20 + dl21 + dl22 + dl23 + dl24 + dl25 + dl26 + dl27 + dl28 + dl29 + dl30 + dl31;
  if (threadIdx.x == 0)   { out[0]=s; cyc[0]=t1-t0; }
  if (threadIdx.x == 128) { out[1]=s; cyc[1]=t1-t0; }
}
static float* g_out; static long long* g_cyc; static size_t g_smem;
template <int MODE,int NEXP> static void slope(long long* wg0, long long* wg1) {
  long long h[2], b0[2]={-1,-1}, b1[2]={-1,-1};
  cudaFuncSetAttribute(bench4<MODE,NEXP>, cudaFuncAttributeMaxDynamicSharedMemorySize, g_smem);
  for (int pass=0; pass<2; ++pass) {
    int it = pass ? 256 : 128;
    long long* dst = pass ? b1 : b0;
    for(int w=0;w<3;++w){bench4<MODE,NEXP><<<1,256,g_smem>>>(it,g_out,g_cyc);cudaDeviceSynchronize();}
    for(int r=0;r<25;++r){bench4<MODE,NEXP><<<1,256,g_smem>>>(it,g_out,g_cyc);cudaDeviceSynchronize();
      cudaMemcpy(h,g_cyc,2*sizeof(long long),cudaMemcpyDeviceToHost);
      for(int k=0;k<2;++k) if(dst[k]<0||h[k]<dst[k]) dst[k]=h[k];}
  }
  *wg0 = b1[0]-b0[0]; *wg1 = b1[1]-b0[1];   // marginal over 128 extra iters
}
int main() {
  cudaMalloc(&g_out,2*sizeof(float)); cudaMalloc(&g_cyc,2*sizeof(long long));
  g_smem = 16384*sizeof(__nv_bfloat16);
  printf("marginal cycles per iteration (128-iter delta), one block = 2 warpgroups on 1 SM\n");
  printf("%-34s %10s %10s\n","", "wg0(mma)", "wg1(soft)");
  long long a,b;
  slope<1,1>(&a,&b); double mma = a/128.0;
  printf("%-34s %10.2f %10.2f\n","wgmma alone (wg0)", a/128.0, b/128.0);
  const int NE[3]={1,2,4};
  double soft[3], conc[3];
  { long long x,y; slope<2,1>(&x,&y); soft[0]=y/128.0; }
  { long long x,y; slope<2,2>(&x,&y); soft[1]=y/128.0; }
  { long long x,y; slope<2,4>(&x,&y); soft[2]=y/128.0; }
  { long long x,y; slope<0,1>(&x,&y); conc[0]=(x>y?x:y)/128.0; }
  { long long x,y; slope<0,2>(&x,&y); conc[1]=(x>y?x:y)/128.0; }
  { long long x,y; slope<0,4>(&x,&y); conc[2]=(x>y?x:y)/128.0; }
  printf("\n%-10s %12s %12s %12s %12s %10s\n","exp2/iter","softmax alone","mma alone","serial sum","concurrent","saved");
  for (int i=0;i<3;++i)
    printf("%-10d %12.2f %12.2f %12.2f %12.2f %9.1f%%\n", NE[i]*8, soft[i], mma,
           soft[i]+mma, conc[i], 100.0*(1.0-conc[i]/(soft[i]+mma)));
  return 0;
}
