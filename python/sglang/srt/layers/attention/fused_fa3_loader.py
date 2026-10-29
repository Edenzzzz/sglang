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

_LOADED_NAMESPACES: set[str] = set()


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



def _fa2_tile_widths() -> tuple[int, int, int]:
    """The three FA2 class widths, largest first, from SGLANG_FUSED_FA2_TILES.

    One source for both the compile-time runner widths and the planner's FUSED_FA2_TILES, which
    must agree: the planner decides which task slot a row lands in, the runners decide what tile
    that slot actually computes. Silent disagreement is a wrong answer, not an error.
    """
    raw = os.environ.get("SGLANG_FUSED_FA2_TILES", "128,64,16")
    try:
        w = tuple(int(x) for x in raw.split(","))
    except ValueError:
        raise ValueError(f"SGLANG_FUSED_FA2_TILES={raw!r} is not three integers") from None
    if len(w) != 3 or any(x <= 0 for x in w):
        raise ValueError(f"SGLANG_FUSED_FA2_TILES={raw!r} must be three positive widths")
    if not (w[0] > w[1] > w[2]):
        raise ValueError(
            f"SGLANG_FUSED_FA2_TILES={raw!r} must be strictly decreasing (largest first): the "
            "planner routes a row to the smallest class that can hold it and derives its "
            "thresholds from this order"
        )
    for x in w:
        # Mirror of the device-side constraint: some warps_q dividing 8 must give an integral
        # num_mma_q. Checked here too so the failure names the env var rather than surfacing as a
        # static_assert deep in a template instantiation.
        if not any(8 % q == 0 and x % (q * 16) == 0 for q in range(1, 9)):
            raise ValueError(
                f"FA2 tile width {x} has no valid warp layout at 8 warps: needs some warps_q "
                "dividing 8 with warps_q*16 dividing the width (16/32/48/64/96/128 all work)"
            )
    return w


def _fa2_tile_flags() -> list[str]:
    """nvcc flags for the three widths, and keep the planner's env var in step."""
    w = _fa2_tile_widths()
    # The planner reads this; set it so it cannot disagree with what was compiled.
    os.environ["FUSED_FA2_TILES"] = ",".join(str(x) for x in w)
    return [f"-DFUSED_FA2_W{i}={x}" for i, x in enumerate(w)]


def load_fused_fa3(verbose: bool = False, cta_q: int | None = None) -> str:
    """Compile/register one FA3 tile variant and return its operator namespace."""
    explicit_variant = cta_q is not None
    compile_q = int(cta_q if explicit_variant else os.environ.get("FA3_CTA_Q", "128"))
    if compile_q not in (64, 128):
        raise ValueError(f"FA3 CTA_Q must be 64 or 128, got {compile_q}")
    namespace = f"fused_fa3_q{compile_q}" if explicit_variant else "fused_fa3"
    if namespace in _LOADED_NAMESPACES and hasattr(getattr(torch.ops, namespace), "fused_run"):
        return namespace
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
        name=namespace,
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
            # DEFAULT 2, matching stock 0.7. Stock unions smem_v with smem_o and relies on
            # DOUBLE BUFFERING to make that union safe: with 2 stages the producer's next V load
            # targets the other stage and cannot overwrite the sO the epilogue is still reading.
            # Our port ran 1 stage to fit shared memory and thereby removed that invariant, which
            # is the measured race. 2 stages fits once ALIAS_SMEM is on (below).
            "-DFUSED_FA3_STAGES=" + os.environ.get("FA3_STAGES", "2"),
            "-DFUSED_FA3_CONSUMER_REGS=" + os.environ.get("SGLANG_FUSED_FA3_CONSUMER_REGS", "0"),
            # Store-completion handshake: the producer cannot refill smem_v (which sO aliases)
            # until the consumer has finished reading sO for the current work item. Off by
            # default until measured -- a mismatched arrival count here deadlocks the kernel.
            # CROSS-ITEM Q FENCE. smem_q is one buffer reused by every work item a CTA runs.
            # Stock waits on barrier_O only AFTER refilling it, and the consumer arrives on
            # barrier_O after only the FIRST QK gemm is issued -- while tSrQ is a smem descriptor
            # iterator, so sQ is re-read on every later kv tile. Sound for one item per CTA, a
            # hazard for many. =1 hoists the producer's wait above the Q load and moves the
            # consumer's arrival to after its last sQ read. Off by default until measured; it
            # serialises producer and consumer across items, so it is expected to cost time.
            # A2. FA2 tile-class widths. These are COMPILE-TIME on the runner side (one
            # BlockBatchPagedAttentionPersistent instantiation per class) while the host planner
            # reads FUSED_FA2_TILES at run time, so the two can disagree -- and a width the
            # runners were not built for routes rows to a tile class that does not exist, with no
            # error. Both are therefore derived HERE from one value, and the planner's env var is
            # set to match rather than read independently.
            #
            # {16,32,48,64,96,128} are all expressible at kFa2Warps=8; the extension
            # static_asserts the two layout invariants, so an unbuildable width fails the compile
            # instead of mis-routing. Default {128,64,16} is the previous behaviour.
            *_fa2_tile_flags(),
            "-DFUSED_CONST_PERS_PARAMS=" + os.environ.get("FUSED_CONST_PERS_PARAMS", "0"),
            "-DFUSED_FA2_REG_REALLOC=" + os.environ.get("FUSED_FA2_REG_REALLOC", "0"),
            "-DFLASHINFER_FA2_MMA_KV=" + os.environ.get("SGLANG_FUSED_FA2_MMA_KV", "2"),
            "-DFLASHINFER_FA2_QSPLIT=" + os.environ.get("SGLANG_FUSED_FA2_QSPLIT", "0"),
            # Per-item pipeline accounting: tiles the producer INTENDED to load vs tiles the
            # consumer actually CONSUMED, accumulated in registers and printed once per block
            # AFTER the loop. The print placement is the whole point -- every instrument that
            # slowed this kernel inside the loop (printf probe, racecheck, memcheck) masked the
            # defect, so the counting happens during the race and the reporting happens after it.
            # Raw tile-sequence trace for ONE block/item: is the deterministic gather sequence
            # CORRECT? The register signature proved only that it is identical every run.
            "-DFUSED_TILE_TRACE=" + os.environ.get("TILE_TRACE", "0"),
            "-DFUSED_PIPE_COUNT=" + os.environ.get("PIPE_COUNT", "0"),
            "-DFUSED_Q_REUSE_FENCE=" + os.environ.get("Q_REUSE_FENCE", "0"),
            "-DFUSED_O_HANDSHAKE=" + os.environ.get("O_HANDSHAKE", "0"),
            # Un-alias smem_o from smem_v. Stock unions them, which is safe for one work item per
            # CTA; our persistent scheduler gives a CTA many, so the producer's next V load can
            # overwrite the sO the epilogue is still reading. Costs ~32 KB of the ~68 KB headroom.
            "-DFUSED_SPLIT_SMEM_O=" + os.environ.get("SPLIT_SMEM_O", "0"),
            # Gates write_packed independently of the Q gather, so the multi-item corruption can be
            # attributed to one or the other. Default follows FLASHINFER_FUSED_PACKED_Q.
            *([] if os.environ.get("PACKED_O") is None
              else ["-DFUSED_PACKED_O=" + os.environ["PACKED_O"]]),
            # A/B knobs for the two FusedWarpgroupTraits overrides, the leading suspects for the
            # FA2-only path being ~20x slower than the stock kernel family. Defaults unchanged.
            "-DFUSED_NB_SYNC=" + os.environ.get("NB_SYNC", "1"),
            "-DFUSED_GRID_1D=" + os.environ.get("GRID_1D", "1"),
            # ALIAS_SMEM=1 puts the FA2 runners' SharedStorage at offset 0, aliasing FA3's region,
            # instead of past it. A block takes one role, so the regions are never both live --
            # EXCEPT that CUTLASS initialises FA3's pipeline mbarriers at kernel entry, BEFORE the
            # sm_split branch, so every block writes into FA3's region whichever role it takes.
            # The tree carries two contradictory measurements of this (aliasing races / aliasing
            # clean 14-of-14 while disjoint fails 10-of-10), both predating today's fixes, so it is
            # exposed here to be settled by measurement.
            # DEFAULT ON. Puts the FA2 runners' SharedStorage at offset 0, aliasing FA3's
            # region rather than sitting past it, so smem_size = max(FA3, FA2) = 163920 instead of
            # FA3 + FA2 = 234544 -- which is what let the 2-stage pipeline launch in the
            # co-resident configuration at all (the device max is 232448).
            #
            # MEASURED on `mixed` @ SM_SPLIT=128, 35 runs per arm:
            #   alias + 2 stages      0 / 35 failures, 0.2426 ms
            #   disjoint + 1 stage    3 / 35 failures, 0.2486 ms
            # Correct, race-free and 2.4% faster. The tree previously carried two contradictory
            # measurements of aliasing; this settles it in favour of "aliasing is sound", and the
            # feared mbarrier conflict (CUTLASS initialises FA3's pipeline barriers before the
            # sm_split branch) does not reproduce.
            *([] if os.environ.get("ALIAS_SMEM") == "0" else ["-DFLASHINFER_FUSED_ALIAS_SMEM=1"]),
            # FA3 tile shape. CTA_KV must divide the planner's kTileKV of 128 (a static_assert in
            # the ext enforces it), so 64 or 128. Shrinking CTA_KV is how the second pipeline
            # stage gets bought.
            f"-DFUSED_FA3_CTA_Q={compile_q}",
            f"-DFUSED_OP_NAMESPACE={namespace}",
            "-DFUSED_FA3_CTA_KV=" + os.environ.get("FA3_CTA_KV", "128"),
            # Diagnostic: skip the packed-GQA Q gather entirely and leave whatever is in smem, so
            # the difference in kernel time BOUNDS what that load costs. Output is garbage by
            # construction -- this measures the load, not the kernel -- so it is only ever read as
            # a timing delta, never as a correctness arm.
            "-DFLASHINFER_SKIP_PACKED_Q=" + os.environ.get("SKIP_PACKED_Q", "0"),
            # Multicast ceiling diagnostic: FA3 K/V copies zero-fill instead of reading global
            # memory (wrong output by construction; timing only). Added only when set, so the
            # default build's flags -- and its cached extension -- are unchanged.
            *([] if os.environ.get("FA3_SKIP_KV_LOAD") is None
              else ["-DFLASHINFER_FA3_SKIP_KV_LOAD=" + os.environ["FA3_SKIP_KV_LOAD"]]),
            # The FA3 role's CAUSAL template parameter. Default 0 was set from a measurement
            # ("CAUSAL=1 is worse") taken while fa3.k_page_stride was unassigned, so every key
            # read page 0 and all keys were identical -- a configuration in which causal and
            # non-causal are indistinguishable. That measurement was therefore vacuous. The
            # reference is bottom-right CAUSAL and the stock wrapper is built with causal=True,
            # so this has to be 1 for the comparison to mean anything.
            "-DFLASHINFER_FUSED_FA3_CAUSAL=" + os.environ.get("FA3_CAUSAL", "1"),
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
            "-DFLASHINFER_FUSED_CLUSTER=" + os.environ.get("FUSED_CLUSTER", "1"),
            "-DFLASHINFER_FA3_PAGED_TMA=" + os.environ.get("FA3_PAGED_TMA", "0"),
            # Packed-GQA addressing for the FA3 role, as a COMPILE-TIME group size: the unified
            # plan tiles FA3 work over (token, head-in-group) pairs, which the Q tensormap cannot
            # express, so both the Q load and the O store index rows through packed_coord.
            # Without it the epilogue writes packed rows as if they were tokens.
            # K1n A/B: FUSED_PACKED_Q=off omits the macro ENTIRELY, reverting both the Q
            # load and the O store to 0.7 stock (TMA + write_tiled) while still running
            # through our extracted fa3_prefill_body and unified scheduler. That is the
            # decisive experiment as one macro rather than an include-path change. Valid
            # only at G=1, where packed rows and tokens coincide -- the scheduler reports
            # the PACKED extent as qo_len and the stock paths read it as tokens.
            *([] if os.environ.get("FUSED_PACKED_Q") == "off"
              else ["-DFLASHINFER_FUSED_PACKED_Q=" + os.environ.get("FUSED_PACKED_Q", "4")]),
            # Pairs with the 32-lane barrier_Q arrival in the packed-Q load. The load's arrival
            # side and prefill_sm90's barrier_Q.init are both gated on THIS macro, so defining
            # one without the other cannot happen: arming for 32 against a single arrival hangs.
            "-DFLASHINFER_FUSED_Q_MBARRIER=" + os.environ.get("FA3_MBARRIER", "1"),
            # write_packed's own device probe: prints the element offset its mapping produces for
            # each tile's first row, plus the values being stored. Off by default. The comment at
            # that probe in the 0.3.1 source is the lesson this port keeps relearning -- "every
            # hypothesis about why task-3 rows come back unwritten has been wrong so far, so stop
            # inferring and read it off the device".
            "-DFLASHINFER_FA3_PARTIAL_PROBE=" + os.environ.get("FA3_PROBE", "0"),
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
    if not hasattr(getattr(torch.ops, namespace), "fused_run"):
        raise RuntimeError(
            f"{namespace} compiled but torch.ops.{namespace}.fused_run is not registered"
        )
    _LOADED_NAMESPACES.add(namespace)
    logger.info(
        "fused FA3 tile variant loaded (namespace=%s, cta_q=%d, src=%s, headers=%s, generated=%s)",
        namespace, compile_q, root, data or f"{root} (source tree)", gen,
    )
    return namespace
