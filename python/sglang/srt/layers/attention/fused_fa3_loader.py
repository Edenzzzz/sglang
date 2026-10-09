"""JIT loader for the fused FA3+FA2 SM-pre-assignment kernel.

The kernel lives in the flashinfer tree as a test extension
(`tests/fused_fa3_persistent_ext.cu`) and registers `torch.ops.fused_fa3.fused_run`. There is no
prebuilt shared object, so it is compiled on first use.

Two build facts that are easy to get wrong and both cost a full cycle in this project:

* The include path list must contain a flashinfer-GENERATED `batch_attention_with*` config
  directory from `~/.cache/flashinfer/*/generated/`. That directory only exists after flashinfer's
  own JIT has emitted it, i.e. after some `BatchAttention.plan()` has run, so the build has an
  ordering dependency on flashinfer. `_ensure_generated_config` bootstraps it rather than failing
  with the harness's "run any BatchAttention.plan() once first" message.
* The upstream harness passes `-std=c++17`, but torch 2.14 headers require **C++20**. Compiling
  with 17 against this toolchain fails deep inside ATen, where the cause is not obvious.
"""

from __future__ import annotations

import glob
import logging
import os

import torch

logger = logging.getLogger(__name__)

_LOADED = False


def fi_root() -> str:
    """Tree holding the kernel SOURCE (`tests/fused_fa3_persistent_ext.cu`)."""
    return os.environ.get("SGLANG_FUSED_FI_ROOT", "/dev/shm/w/fi")


def fi_data_dir() -> str | None:
    """The INSTALLED flashinfer wheel's bundled headers (`flashinfer/data`).

    The headers must come from the flashinfer the SERVER runs, not from the source tree the
    kernel was written against. `HolisticPlanInfo`'s serialised length is a property of the
    header: the 0.3.1 source tree yields 56/68 entries while the installed 0.7.0.post1
    `BatchAttention._plan_info` is 28, and the kernel's own guard then rejects every plan
    (`persistent plan has 28 entries, expected 56 or 68`). Compiling against the installed
    wheel's headers makes the two agree by construction.

    Returns None if the wheel ships no bundled headers, in which case the caller falls back to
    the source tree and the mismatch is reported rather than hidden.
    """
    override = os.environ.get("SGLANG_FUSED_FI_DATA")
    if override:
        return override
    try:
        import flashinfer

        d = os.path.join(os.path.dirname(flashinfer.__file__), "data")
    except Exception:
        return None
    return d if os.path.isdir(os.path.join(d, "include")) else None


def _generated_config_dirs() -> list[str]:
    return sorted(
        glob.glob(
            os.path.expanduser("~/.cache/flashinfer/*/generated/batch_attention_with*")
        )
    )


def _ensure_generated_config() -> str:
    """Return a generated batch_attention config dir, emitting one if flashinfer has not yet."""
    dirs = _generated_config_dirs()
    if dirs:
        return dirs[0]
    # Drive flashinfer's own JIT once, with the smallest plan that emits the config.
    from flashinfer.attention import BatchAttention

    dev = torch.device("cuda")
    w = BatchAttention(kv_layout="NHD")
    bs, kv, hq, hkv, d = 1, 16, 28, 7, 128
    w.plan(
        torch.tensor([0, 1], dtype=torch.int32, device=dev),
        torch.tensor([0, kv], dtype=torch.int32, device=dev),
        torch.arange(kv, dtype=torch.int32, device=dev),
        torch.tensor([kv], dtype=torch.int32, device=dev),
        bs,
        hq,
        hkv,
        d,
        d,
        1,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    dirs = _generated_config_dirs()
    if not dirs:
        raise RuntimeError(
            "flashinfer emitted no generated/batch_attention_with* config even after a "
            "BatchAttention.plan(); the fused kernel cannot be compiled without its headers"
        )
    return dirs[0]


def load_fused_fa3(verbose: bool = False) -> None:
    """Compile and register `torch.ops.fused_fa3.fused_run`. Idempotent."""
    global _LOADED
    if _LOADED and hasattr(torch.ops, "fused_fa3"):
        return
    from torch.utils.cpp_extension import load

    root = fi_root()
    src = os.path.join(root, "tests", "fused_fa3_persistent_ext.cu")
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"fused kernel source not found at {src}; set SGLANG_FUSED_FI_ROOT to the "
            "flashinfer SOURCE tree"
        )
    gen = _ensure_generated_config()
    data = fi_data_dir()
    if data:
        # Installed wheel's bundled headers: cutlass/spdlog/cccl live directly under data/,
        # not under a 3rdparty/ prefix as in the source tree.
        includes = [
            gen,
            f"{data}/include",
            f"{data}/csrc",
            f"{data}/cutlass/include",
            f"{data}/cutlass/tools/util/include",
            f"{data}/spdlog/include",
            f"{data}/cccl/thrust",
            f"{data}/cccl/libcudacxx/include",
            f"{data}/cccl/cub",
        ]
        includes = [p for p in includes if os.path.isdir(p)]
    else:
        logger.warning(
            "installed flashinfer ships no bundled headers; compiling against the source tree "
            "at %s, whose plan layout may not match the running flashinfer", root
        )
        includes = [
            gen,
            f"{root}/include",
            f"{root}/csrc",
            f"{root}/3rdparty/cutlass/include",
            f"{root}/3rdparty/cutlass/tools/util/include",
            f"{root}/3rdparty/spdlog/include",
        ]
    load(
        name="fused_fa3",
        sources=[src],
        extra_include_paths=includes,
        extra_cuda_cflags=[
            # C++20, not the harness's 17: torch 2.14's headers require it.
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "-arch=sm_90a",
            "-O3",
            "--threads",
            "32",
            "-DFUSED_FA3_STAGES=" + os.environ.get("FA3_STAGES", "1"),
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        extra_cflags=["-std=c++20"],
        is_python_module=False,
        verbose=verbose,
    )
    if not hasattr(torch.ops, "fused_fa3") or not hasattr(
        torch.ops.fused_fa3, "fused_run"
    ):
        raise RuntimeError(
            "fused_fa3 compiled but torch.ops.fused_fa3.fused_run is not registered"
        )
    _LOADED = True
    logger.info(
        "fused_fa3 kernel loaded (src=%s, headers=%s, generated=%s)",
        root, data or f"{root} (source tree)", gen,
    )
