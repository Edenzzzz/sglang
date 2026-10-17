"""A5.0d patches to a copy of fi_tma_multicast (tree fi_a5):
  python fi_a5_patch.py <fi root>
1. dynamic smem covers flash-attention's SharedStorage when the FA3 role runs its kernel body (FLASHINFER_FA3_UPSTREAM):
   the launch sized it from flashinfer's FA3 traits (163 920 B), so the upstream body addressed shared memory past
   the allocation (compute-sanitizer: out-of-range shared address, 2048:2048 at 112 CTAs)."""
import sys
p = f"{sys.argv[1]}/include/flashinfer/attention/persistent_fa3_fused.cuh"
s = open(p).read()
old = "  int smem_size = int(sizeof(typename KernelTraits::SharedStorage));\n"
new = ("  int smem_size = int(sizeof(typename KernelTraits::SharedStorage));\n"
       "#if FLASHINFER_FA3_UPSTREAM\n"
       "  // The FA3 role runs flash-attention's kernel body, whose shared storage is its own.\n"
       "  smem_size = std::max(\n"
       "      smem_size, int(UpstreamFA3Kernel<KernelTraits::HEAD_DIM_VO, int32_t>::SharedStorageSize));\n"
       "#endif\n")
if "UpstreamFA3Kernel<KernelTraits::HEAD_DIM_VO, int32_t>::SharedStorageSize" in s:
    print("already")
else:
    assert s.count(old) == 1, s.count(old)
    open(p, "w").write(s.replace(old, new, 1)); print("patched")
