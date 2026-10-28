// Compile-time report of the shared-memory each role needs, for item 18's piggyback.
//
// It matters because the piggyback is the ONLY configuration where an FA3 CTA and the 16-row row
// mainloop need their shared memory AT THE SAME TIME. Every other mode lets the two regions alias,
// since a CTA takes one role or the other, so the kernel asks for max(FA3, FA2); the piggyback has
// to ask for FA3 + FA2 and that sum is what decides whether it can launch at all.
//
// There is no runtime here and no GPU: the sizes are read out of a deliberately failed template
// instantiation, so `nvcc -c` prints them as part of the error text. Compile with the same include
// set and the same -D flags as gate_q64_registers_20261028.py and read the numbers off stderr:
//
//   nvcc -std=c++20 --expt-relaxed-constexpr -arch=sm_90a -c <the gate's -I and -D flags> \
//        -DFUSED_FA3_CTA_Q=64 probe_fused_smem_sizes_20261028.cu
//
// A non-zero exit status is the EXPECTED outcome. Claude, 2026-10-28.
#define FLASHINFER_FUSED_NO_SETMAXNREG 1
#ifndef FLASHINFER_FUSED_FA3_CAUSAL
#define FLASHINFER_FUSED_FA3_CAUSAL 1
#endif
#include "batch_attention_config.inc"
#include <flashinfer/attention/hopper/default_params.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/attention/persistent_fa3_fused.cuh>

using namespace flashinfer;
using FDT = cutlass::bfloat16_t;
using PDT = __nv_bfloat16;
using IdT = int32_t;

#ifndef FUSED_FA3_STAGES
#define FUSED_FA3_STAGES 2
#endif
#ifndef FUSED_FA3_CTA_Q
#define FUSED_FA3_CTA_Q 128
#endif
#ifndef FUSED_FA3_CTA_KV
#define FUSED_FA3_CTA_KV 128
#endif
using FA3Traits = AttentionKernelTraits<false, 128, 128, FUSED_FA3_CTA_Q, FUSED_FA3_CTA_KV,
                                        FUSED_FA3_STAGES, FDT, FDT, FDT, IdT,
                                        DefaultAttentionSm90<false>>;
// The 16-row row class, at the same traits the fused extension builds it with.
template <uint32_t CTA_TILE_Q, uint32_t NUM_MMA_KV>
using PKT = FusedWarpgroupTraits<
    KernelTraits<MaskMode::kCausal, CTA_TILE_Q, 1, NUM_MMA_KV, 8, 8, 1, 4, PosEncodingMode::kNone,
                 PDT, PDT, PDT, float, IdT, StandardAttention<false>>>;
using R16 = BlockBatchPagedAttentionPersistent<PKT<16, 2>, PersistentParams>;

// Undefined on purpose: the diagnostic names the template argument, which is the number wanted.
template <size_t Bytes>
struct REPORT;

REPORT<sizeof(typename FA3Traits::SharedStorage)> fa3_bytes;
REPORT<sizeof(typename R16::KTraits::SharedStorage)> row16_bytes;
REPORT<sizeof(typename FA3Traits::SharedStorage) +
       sizeof(typename R16::KTraits::SharedStorage)> piggyback_sum_bytes;
// The device's opt-in maximum on H100/H200, for the comparison the sum has to pass.
REPORT<232448> device_max_optin_bytes;
