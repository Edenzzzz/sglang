"""Build an isolated native FA2 variant with FP32 partial and final outputs."""

import argparse
import importlib.util
import os
from pathlib import Path


def replace_exact(source, original, replacement, count=1):
    actual = source.count(original)
    if actual != count:
        raise ValueError(f"Source anchor count {actual}, expected {count}: {original}")
    return source.replace(original, replacement)


def transform_bridge(source):
    for original, replacement, count in (
        ("NativePagedParams", "NativePagedFloatParams", 3),
        ("native_fa2_run", "native_fa2_float_run", 1),
        ("using DTypeO = __nv_bfloat16;", "using DTypeO = float;", 1),
        ("output.scalar_type() == at::kBFloat16", "output.scalar_type() == at::kFloat", 1),
        ('"native FA2 bridge supports bf16 only"', '"native FA2 FP32 output requires BF16 Q/K/V"', 1),
        ("params.o = static_cast<__nv_bfloat16*>(output.data_ptr());", "params.o = output.data_ptr<float>();", 1),
        ("__nv_bfloat16* temporary_output", "float* temporary_output", 1),
        ("GetPtrFromBaseOffset<__nv_bfloat16>(float_buffer, plan.v_offset)", "GetPtrFromBaseOffset<float>(float_buffer, plan.v_offset)", 1),
    ):
        source = replace_exact(source, original, replacement, count)
    for input_type in ("DTypeQ", "DTypeKV"):
        if source.count(f"using {input_type} = __nv_bfloat16;") != 1:
            raise ValueError("FP32 output variant must keep BF16 inputs")
    return source


def specialize_merge(source):
    start = source.index("cudaError_t VariableLengthMergeStates(")
    end = source.index("\ntemplate <", start)
    original = source[start:end]
    replacement = replace_exact(
        original, "  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {",
        "  {\n    constexpr size_t HEAD_DIM = 128;\n    if (head_dim != HEAD_DIM) return cudaErrorInvalidValue;",
    )
    replacement = replace_exact(replacement, "  });\n  return cudaSuccess;", "  }\n  return cudaSuccess;")
    return source[:start] + replacement + source[end:]


def direct_output_bridge(source):
    for original, replacement, count in (
        ("NativePagedFloatParams", "NativePagedFloatDirectParams", 3),
        ("native_fa2_float_run", "native_fa2_float_direct_run", 1),
        ("  float* lse = nullptr;", "  float* lse = nullptr;\n  __nv_bfloat16* final_bf16_output = nullptr;", 1),
        ("output.scalar_type() == at::kFloat", "output.scalar_type() == at::kBFloat16", 1),
        ('"native FA2 FP32 output requires BF16 Q/K/V"', '"native FA2 direct output requires BF16 Q/K/V/output"', 1),
        ("params.o = output.data_ptr<float>();", "params.final_bf16_output = static_cast<__nv_bfloat16*>(output.data_ptr());", 1),
        ("  plan.FromVector(plan_values);", '  plan.FromVector(plan_values);\n  TORCH_CHECK(plan.split_kv, "direct BF16 output requires a SplitKV plan");', 1),
    ):
        source = replace_exact(source, original, replacement, count)
    return source


def direct_output_prefill(source):
    start = source.index("cudaError_t BatchPrefillWithPagedKVCacheDispatchedImpl(")
    end = source.index("\ntemplate <", start)
    replacement = replace_exact(source[start:end], "            auto o = params.o;", "            auto o = params.final_bf16_output;")
    return source[:start] + replacement + source[end:]


def build(source_root, generated_root, direct_output=False):
    from torch.utils.cpp_extension import load

    source_root, generated_root = source_root.resolve(), generated_root.resolve()
    generated_root.mkdir(parents=True, exist_ok=True)
    bridge = transform_bridge((source_root / "native_prefill_bridge.cuh").read_text())
    name = "native_fa2_q32_fp32_direct" if direct_output else "native_fa2_q32_fp32"
    function = "native_fa2_float_direct_run" if direct_output else "native_fa2_float_run"
    if direct_output:
        bridge = direct_output_bridge(bridge)
    binding = (source_root / "native_fa2_q32.cu").read_text()
    binding = replace_exact(binding, "TORCH_LIBRARY(native_fa2_q32,", f"TORCH_LIBRARY({name},")
    binding = replace_exact(binding, "&native_fa2_run", f"&{function}")
    (generated_root / "native_prefill_bridge.cuh").write_text(bridge)
    (generated_root / f"{name}.cu").write_text(binding)
    specification = importlib.util.spec_from_file_location("native_float_base", source_root / "base_loader.py")
    loader = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loader)
    data = loader.fi_data_dir()
    if data is None:
        raise RuntimeError("Installed FlashInfer headers are required")
    os.environ["SGLANG_FUSED_MERGED_DIR"] = str(generated_root / "merged")
    includes = loader._merged_include_tree(data, str(source_root / "port07"))
    cascade_relative = Path("flashinfer/attention/cascade.cuh")
    cascade_source = source_root / "port07/include" / cascade_relative
    if not cascade_source.exists():
        cascade_source = Path(data) / "include" / cascade_relative
    (Path(includes) / cascade_relative).write_text(specialize_merge(cascade_source.read_text()))
    prefill_relative = Path("flashinfer/attention/prefill.cuh")
    prefill_source = (source_root / "port07/include" / prefill_relative).read_text()
    (Path(includes) / prefill_relative).write_text(direct_output_prefill(prefill_source) if direct_output else prefill_source)
    return load(
        name=name,
        sources=[str(generated_root / f"{name}.cu")],
        extra_include_paths=[
            includes, str(generated_root), str(Path(data) / "cutlass/include"),
            str(Path(data) / "cutlass/tools/util/include"),
            str(Path(data) / "cccl/libcudacxx/include"),
            str(Path(data) / "cccl/thrust"), str(Path(data) / "cccl/cub"),
        ],
        extra_cuda_cflags=[
            "-std=c++20", "-O3", "-arch=sm_90a", "--expt-relaxed-constexpr",
            "--expt-extended-lambda", "--use_fast_math", "-DNDEBUG",
            "-DCUTLASS_ENABLE_GDC_FOR_SM90", "--threads", "8",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        extra_cflags=["-std=c++20", "-O3", "-DNDEBUG"],
        is_python_module=False,
        verbose=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("generated_root", type=Path)
    parser.add_argument("--direct-output", action="store_true")
    arguments = parser.parse_args()
    print(build(arguments.source_root, arguments.generated_root, arguments.direct_output), flush=True)
