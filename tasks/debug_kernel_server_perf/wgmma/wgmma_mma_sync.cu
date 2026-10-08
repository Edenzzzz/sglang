// Does warp-level mma.sync overlap with an in-flight warpgroup wgmma?
//
// Motivation: a decode/draft row has packed_qo_len = gqa_ratio (4-16), but
// wgmma's M is fixed at 64, so such a row wastes 4-16x of the M dimension.
// mma.sync m16n8k16 has M=16 and would waste far less. If the two instruction
// classes issue to the tensor core independently, a fused kernel could run
// small rows on mma.sync *underneath* the large rows' wgmma. If they contend
// for the same pipe, costs simply add and there is nothing to exploit.
//
// Same methodology as the other probes here: marginal cycles from a 128->256
// iteration delta so fixed overhead cancels; accumulators live in registers and
// each shape's first wgmma uses scale-d=0 (else ptxas C7515 serializes the
// pipeline); all variants are compile-time template parameters (a runtime
// branch in the issue loop blocks unrolling and inflates the baseline 3x).
#include <cstdio>
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

// Ampere-style warp-level MMA: m16n8k16 bf16, 4 A regs + 2 B regs -> 4 f32 acc
__device__ __forceinline__ void mma_m16n8k16(float &c0,float &c1,float &c2,float &c3,
    const uint32_t a0,const uint32_t a1,const uint32_t a2,const uint32_t a3,
    const uint32_t b0,const uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

// KIND 0 = wgmma only, 1 = mma.sync only, 2 = both interleaved
template <int KIND, int NMMA>
__global__ void bench5(int iters, float* out, long long* cyc) {
  extern __shared__ __nv_bfloat16 smem[];
  for (int i = threadIdx.x; i < 8192; i += blockDim.x) smem[i] = __float2bfloat16(0.01f);
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  uint64_t da = make_desc(&smem[0],16,256), db = make_desc(&smem[4096],16,256);
  float dl0, dl1, dl2, dl3, dl4, dl5, dl6, dl7, dl8, dl9, dl10, dl11, dl12, dl13, dl14, dl15, dl16, dl17, dl18, dl19, dl20, dl21, dl22, dl23, dl24, dl25, dl26, dl27, dl28, dl29, dl30, dl31;
  uint32_t a0=threadIdx.x|0x3c003c00u,a1=a0^1,a2=a0^2,a3=a0^3,b0=a0^4,b1=a0^5;
  float c0=0.f,c1=0.f,c2=0.f,c3=0.f;
  __syncthreads();
  long long t0 = clock64();
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  if (KIND != 1) { wgmma_n64_sd0(dl0, dl1, dl2, dl3, dl4, dl5, dl6, dl7, dl8, dl9, dl10, dl11, dl12, dl13, dl14, dl15, dl16, dl17, dl18, dl19, dl20, dl21, dl22, dl23, dl24, dl25, dl26, dl27, dl28, dl29, dl30, dl31, da, db); }
  for (int i = 1; i < iters; ++i) {
    if (KIND != 1) { wgmma_n64_sd1(dl0, dl1, dl2, dl3, dl4, dl5, dl6, dl7, dl8, dl9, dl10, dl11, dl12, dl13, dl14, dl15, dl16, dl17, dl18, dl19, dl20, dl21, dl22, dl23, dl24, dl25, dl26, dl27, dl28, dl29, dl30, dl31, da, db); }
#pragma unroll
    for (int k = 0; k < NMMA; ++k) mma_m16n8k16(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1);
  }
  if (KIND == 1) { wgmma_n64_sd0(dl0, dl1, dl2, dl3, dl4, dl5, dl6, dl7, dl8, dl9, dl10, dl11, dl12, dl13, dl14, dl15, dl16, dl17, dl18, dl19, dl20, dl21, dl22, dl23, dl24, dl25, dl26, dl27, dl28, dl29, dl30, dl31, da, db); }     // define accumulators without timing wgmma work
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  long long t1 = clock64();
  float s = (dl0 + dl1 + dl2 + dl3 + dl4 + dl5 + dl6 + dl7 + dl8 + dl9 + dl10 + dl11 + dl12 + dl13 + dl14 + dl15 + dl16 + dl17 + dl18 + dl19 + dl20 + dl21 + dl22 + dl23 + dl24 + dl25 + dl26 + dl27 + dl28 + dl29 + dl30 + dl31) + c0+c1+c2+c3;
  if (threadIdx.x == 0) { out[0]=s; cyc[0]=t1-t0; }
}
static float* g_out; static long long* g_cyc; static size_t g_smem;
template <int KIND,int NMMA> static double slope() {
  long long v,a=-1,b=-1;
  cudaFuncSetAttribute(bench5<KIND,NMMA>, cudaFuncAttributeMaxDynamicSharedMemorySize, g_smem);
  for (int pass=0;pass<2;++pass){
    int it=pass?256:128; long long best=-1;
    for(int w=0;w<3;++w){bench5<KIND,NMMA><<<1,128,g_smem>>>(it,g_out,g_cyc);cudaDeviceSynchronize();}
    for(int r=0;r<25;++r){bench5<KIND,NMMA><<<1,128,g_smem>>>(it,g_out,g_cyc);cudaDeviceSynchronize();
      cudaMemcpy(&v,g_cyc,sizeof(long long),cudaMemcpyDeviceToHost); if(best<0||v<best)best=v;}
    (pass?b:a)=best;
  }
  return double(b-a)/128.0;
}
int main(){
  cudaMalloc(&g_out,sizeof(float)); cudaMalloc(&g_cyc,sizeof(long long));
  g_smem=16384*sizeof(__nv_bfloat16);
  printf("marginal cycles per iteration (128->256 delta), one warpgroup, min of 25\n");
  printf("wgmma m64n64k16 alone: %6.2f\n", slope<0,0>());
  printf("\n%-8s %12s %12s %12s %10s\n","mma/iter","mma alone","wgmma alone","interleaved","sum");
  double w = slope<0,0>();
  printf("%-8d %12.2f %12.2f %12.2f %10.2f\n",2, slope<1,2>(), w, slope<2,2>(), slope<1,2>()+w);
  printf("%-8d %12.2f %12.2f %12.2f %10.2f\n",4, slope<1,4>(), w, slope<2,4>(), slope<1,4>()+w);
  printf("%-8d %12.2f %12.2f %12.2f %10.2f\n",8, slope<1,8>(), w, slope<2,8>(), slope<1,8>()+w);
  printf("%-8d %12.2f %12.2f %12.2f %10.2f\n",16,slope<1,16>(),w, slope<2,16>(),slope<1,16>()+w);
  return 0;
}
