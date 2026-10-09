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
import tempfile

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
    """Generated batch_attention config dirs, asked of flashinfer rather than guessed.

    The old glob (~/.cache/flashinfer/*/generated/batch_attention_with*) missed under 0.7 twice
    over: 0.7 inserts an ARCH segment, so the path is <version>/<arch>/generated/..., and sglang
    redirects the cache base at import time, which moved the whole tree under ~/.cache/sglang/.
    FLASHINFER_GEN_SRC_DIR is computed from the same env the running process used, so it tracks
    both without this file knowing about either. The glob stays as a fallback for a flashinfer
    old enough not to export it.
    """
    roots: list[str] = []
    try:
        from flashinfer.jit import env as _fi_env

        gen = getattr(_fi_env, "FLASHINFER_GEN_SRC_DIR", None)
        if gen is not None:
            roots.append(os.path.join(str(gen), "batch_attention_with*"))
    except Exception:
        pass
    roots.append(
        os.path.expanduser("~/.cache/flashinfer/*/generated/batch_attention_with*")
    )
    roots.append(
        os.path.expanduser("~/.cache/flashinfer/*/*/generated/batch_attention_with*")
    )
    out: list[str] = []
    for pat in roots:
        out.extend(d for d in glob.glob(pat) if os.path.isdir(d))
    return sorted(dict.fromkeys(out))


def _ensure_generated_config() -> str:
    """Return a generated batch_attention config dir, emitting one if flashinfer has not yet."""
    dirs = _generated_config_dirs()
    if dirs:
        return dirs[0]
    # Drive flashinfer's own JIT once, with the smallest plan that emits the config.
    from flashinfer.attention import BatchAttention

    dev = torch.device("cuda")
    w = BatchAttention(kv_layout="NHD")
    kv, hq, hkv, d = 16, 28, 7, 128
    # Keyword form on purpose. 0.7 dropped the `batch_size` positional that 0.3.1 took between
    # kv_len_arr and num_qo_heads, so the old positional call silently shifted every argument
    # after it -- page_size landed in head_dim_vo -- and only surfaced as `plan() got multiple
    # values for argument 'causal'`. Passing by name makes the next signature change fail on the
    # argument that actually went away.
    w.plan(
        qo_indptr=torch.tensor([0, 1], dtype=torch.int32, device=dev),
        kv_indptr=torch.tensor([0, kv], dtype=torch.int32, device=dev),
        kv_indices=torch.arange(kv, dtype=torch.int32, device=dev),
        kv_len_arr=torch.tensor([kv], dtype=torch.int32, device=dev),
        num_qo_heads=hq,
        num_kv_heads=hkv,
        head_dim_qk=d,
        head_dim_vo=d,
        page_size=1,
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


def _merged_include_tree(data: str, root: str) -> str:
    """Build 0.7's header tree with this project's overlay laid ON TOP, and return its path.

    Include ORDER cannot express what the overlay needs. The overlay replaces headers that
    exist in the wheel under the same names -- persistent.cuh, prefill.cuh, epilogue.cuh,
    sparse_mainloop.cuh, hopper/variants.cuh, hopper/prefill_sm90.cuh and more -- so with the
    wheel first the wheel's copies win, and with the overlay first its own siblings go missing:
    both trees use RELATIVE includes ("../exception.h", "../cp_async.cuh"), which resolve
    relative to the including file, so a partial overlay that mirrors only
    flashinfer/attention/ breaks every one of them whatever the -I order.

    Copying the wheel's tree and overwriting with the overlay gives each header the siblings it
    expects and makes "overlay wins" exact. This is the construction port07/tests/nvcc_check.sh
    uses, which is the only one shown to compile this kernel against 0.7.

    Cached under a content-stamped directory so a rebuild does not recopy, and so editing an
    overlay header produces a new tree rather than a stale hit.
    """
    import hashlib, shutil

    overlay = os.path.join(root, "include")
    if not os.path.isdir(overlay):
        raise FileNotFoundError(
            f"no overlay include tree at {overlay}; SGLANG_FUSED_FI_ROOT must point at a tree "
            "containing include/flashinfer/... (e.g. flashinfer's port07/)"
        )
    stamp = hashlib.sha256()
    stamp.update(data.encode())
    for dirpath, dirnames, filenames in os.walk(overlay):
        dirnames.sort()
        for fn in sorted(filenames):
            fp = os.path.join(dirpath, fn)
            stamp.update(os.path.relpath(fp, overlay).encode())
            st = os.stat(fp)
            stamp.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    key = stamp.hexdigest()[:16]

    base = os.environ.get("SGLANG_FUSED_MERGED_DIR") or os.path.join(
        tempfile.gettempdir(), "sglang_fused_inc"
    )
    out = os.path.join(base, key)
    done = os.path.join(out, ".complete")
    if os.path.exists(done):
        return os.path.join(out, "inc")
    tmp = out + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    inc = os.path.join(tmp, "inc")
    os.makedirs(inc, exist_ok=True)
    shutil.copytree(os.path.join(data, "include"), inc, dirs_exist_ok=True)
    shutil.copytree(overlay, inc, dirs_exist_ok=True)
    os.makedirs(out, exist_ok=True)
    # Move into place, then mark complete, so a crashed build never leaves a tree that a later
    # run would treat as usable.
    shutil.rmtree(out, ignore_errors=True)
    os.rename(tmp, out)
    open(os.path.join(out, ".complete"), "w").close()
    return os.path.join(out, "inc")


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
        # ONE merged tree in place of the wheel's include dir. The old scheme appended the
        # source tree after the wheel so flashinfer's own types came from 0.7 while the
        # project's additions were still found; that works only while the two trees are
        # disjoint, which the 0.7 overlay is not -- it deliberately replaces same-named
        # headers. See _merged_include_tree for why no -I order can express this.
        includes[1] = _merged_include_tree(data, root)
        includes += [f"{root}/csrc"]
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
            # Device-side trace of what the FA2 role actually loaded and where it writes. The
            # plan's units cannot be settled by reading either side's source -- a unit audit
            # written from the same understanding passed while the kernel still faulted.
            "-DFLASHINFER_FUSED_TRACE_WRITE=" + os.environ.get("FUSED_TRACE", "0"),
            # This TU includes BOTH the FA2-lineage and Hopper trees, which 0.7 never does
            # anywhere, so the unsuffixed Hopper aliases have to go or the two collide. It must
            # be a TU-wide flag: as a #define in a header it only works when that header is the
            # first to reach hopper/variant_helper.cuh, which it is not.
            "-DFLASHINFER_SM90_SUPPRESS_UNSUFFIXED_NAMES=1",
            # Drive the wgmma role from task 3 of the unified plan. This is also the ONLY
            # configuration that instantiates store_partial, so without it the split-KV path
            # is compiled out and split rows write no partial.
            "-DFLASHINFER_FUSED_UNIFIED_WGMMA=1",
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
