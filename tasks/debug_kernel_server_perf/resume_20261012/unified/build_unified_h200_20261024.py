"""Build the unified tile dispatcher extension (A6) on the H200 against the installed flashinfer 0.7
wheel with the port07 overlay merged on top. Registers torch.ops.unified_dispatch.{build_unified_plan,
unified_run}.
  python build_unified_h200_20261024.py <port07 dir>      (UNIFIED_TILES="48,32,16" by default)
Flags that matter:
  -DFLASHINFER_FA3_UPSTREAM=1      compiles the FA3 executor (flash-attention's mainloop/epilogue)
  -DCUTLASS_ENABLE_GDC_FOR_SM90    without it cutlass compiles launch_dependent_grids /
                                   wait_on_dependent_grids to no-ops and PDL silently serializes
  -DNDEBUG                         FA3 built without it is not the kernel serving runs
Claude, 2026-10-24.
"""
import os
import sys
import time

sys.path.insert(0, "/dev/shm/w/native_kernel_sources_20261023/native_fa2_q32_v2_work_20261015")
import base_loader  # noqa: E402
import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

root = os.path.abspath(sys.argv[1])
merged_base = "/dev/shm/w/ub/merged"
os.makedirs(merged_base, exist_ok=True)
os.environ["SGLANG_FUSED_MERGED_DIR"] = merged_base
fa_link = os.path.join(merged_base, "flash-attention")
if not os.path.exists(fa_link):
    os.symlink("/dev/shm/w/native_kernel_sources_20261023/flash-attention", fa_link)
gen = base_loader._ensure_generated_config()
data = base_loader.fi_data_dir()
inc = base_loader._merged_include_tree(data, root)
assert os.path.dirname(os.path.dirname(inc)) == merged_base, inc
includes = [gen, inc, f"{data}/csrc", f"{data}/cutlass/include", f"{data}/cutlass/tools/util/include",
            f"{data}/spdlog/include", f"{data}/cccl/thrust", f"{data}/cccl/libcudacxx/include", f"{data}/cccl/cub"]
includes = [p for p in includes if os.path.isdir(p)]
w = [int(x) for x in os.environ.get("UNIFIED_TILES", "48,32,16").split(",")]
name = os.environ.get("UNIFIED_EXT_NAME", "unified_dispatch_ext")
t0 = time.time()
load(
    name=name,
    sources=[os.path.join(root, "tests", "unified_dispatch_ext.cu")],
    extra_include_paths=includes,
    extra_cuda_cflags=[
        "-std=c++20", "--expt-relaxed-constexpr", "-arch=sm_90a", "-O3", "--use_fast_math", "-DNDEBUG",
        "--threads", "16", "-lineinfo",
        "-DFLASHINFER_FA3_UPSTREAM=1", "-DCUTLASS_ENABLE_GDC_FOR_SM90",
        "-DFLASHINFER_SM90_SUPPRESS_UNSUFFIXED_NAMES=1",
        f"-DUNIFIED_FA2_W0={w[0]}", f"-DUNIFIED_FA2_W1={w[1]}", f"-DUNIFIED_FA2_W2={w[2]}",
        "-DFLASHINFER_FA2_MMA_KV=" + os.environ.get("UNIFIED_FA2_MMA_KV", "2"),
        "-DUNIFIED_FA2_CLASSES=" + os.environ.get("UNIFIED_FA2_CLASSES", "3"),
        "-DUNIFIED_FA2_MIN_BLOCKS=" + os.environ.get("UNIFIED_FA2_MIN_BLOCKS", "1"),
        "-DUNIFIED_WQ32=" + os.environ.get("UNIFIED_WQ32", "0"),
        "-DUNIFIED_SKIP_MERGE=" + os.environ.get("UNIFIED_SKIP_MERGE", "0"),
        "-DUNIFIED_FA2_DYNAMIC=" + os.environ.get("UNIFIED_FA2_DYNAMIC", "1"),
        "-DUNIFIED_FA2_NP=" + os.environ.get("UNIFIED_FA2_NP", "0"),
        "-DUNIFIED_NP_MERGE=" + os.environ.get("UNIFIED_NP_MERGE", "0"),
        "-DUNIFIED_FA3_STOCK=" + os.environ.get("UNIFIED_FA3_STOCK", "0"),
        "-DUNIFIED_NP_KV16=" + os.environ.get("UNIFIED_NP_KV16", "2"),
        "-DUNIFIED_NP_KV32=" + os.environ.get("UNIFIED_NP_KV32", "4"),
        "-DUNIFIED_MMA_KV16=" + os.environ.get("UNIFIED_MMA_KV16", os.environ.get("UNIFIED_FA2_MMA_KV", "2")),
        "-DUNIFIED_MMA_KV32=" + os.environ.get("UNIFIED_MMA_KV32", os.environ.get("UNIFIED_FA2_MMA_KV", "2")),
        "-DFLASHINFER_FA2_QSPLIT=" + os.environ.get("UNIFIED_FA2_QSPLIT", "0"),
        "--ptxas-options=-v",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    ],
    extra_cflags=["-std=c++20"],
    is_python_module=False,
    verbose=bool(int(os.environ.get("VERBOSE", "0"))),
)
assert hasattr(torch.ops.unified_dispatch, "build_unified_plan") and hasattr(torch.ops.unified_dispatch, "unified_run")
print(f"UNIFIED_BUILD_OK tiles={w} seconds={time.time() - t0:.0f} ext_dir={torch.utils.cpp_extension._get_build_directory(name, False)}")
