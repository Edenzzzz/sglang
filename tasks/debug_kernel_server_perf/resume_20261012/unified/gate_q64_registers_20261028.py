"""Item 18 step 1, the GATING COMPILE (no GPU needed).

Compiles the CO-COMPILED fused translation unit -- FA3's wgmma mainloop and the three FA2
`mma.sync` runners in ONE `__global__` and therefore ONE ptxas register allocation
(`FusedFA3PersistentKernel`) -- with the FA3 role's query tile at 128 and then at 64, and reads
registers per thread and spill bytes per kernel out of `ptxas -v`.

The question item 18 rests on: FA3 at a 128-row query tile runs two consumer warpgroups at the
168-register baseline, which is the SM's whole budget, and co-compiling the FA2 mainloop into that
allocation spilled 1,436 bytes per tile class and ran 2-6x slower. At a 64-row tile FA3 runs ONE
consumer warpgroup. If the spill clears at 64 the freed warpgroup can host `mma.sync` row work
inside the already-resident CTA; if it does not, the register wall holds and item 18 cannot be
built as proposed.

nvcc is driven DIRECTLY rather than through torch.utils.cpp_extension.load, for two reasons:
load() keys its cache on the extension NAME, so it silently serves a stale .so when only -D flags
change (this project has twice measured the wrong kernel that way), and `nvcc -c` needs neither a
GPU nor libcuda. The include tree and the flag list are taken from base_loader.load_fused_fa3 so
this compiles the same configuration the harness arms do.

  python gate_q64_registers_20261028.py [--qs 128,64] [--out DIR] [--upstream] [--extra=FLAGS]

Runs on the H200 (where base_loader and the wheel's bundled headers are) and, with
--local-root, on the dev box against the flashinfer SOURCE tree, building the same
wheel-over-overlay merged include tree by hand. ptxas register numbers are only comparable
WITHIN one toolchain, so never mix arms across the two hosts: re-run every arm on whichever host
is answering.

--upstream additionally sets FLASHINFER_FA3_UPSTREAM=1, which replaces flashinfer's simplified
mainloop with flash-attention's own -- the one the serving arms run. Its tile is fixed at
kBlockM=128 inside UpstreamFA3Mainloop, so FUSED_FA3_CTA_Q does NOT shrink it; the arm is compiled
only to record what the production FA3 role costs in the same allocation.
Claude, 2026-10-28.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--qs", default="128,64")
ap.add_argument("--out", type=Path, default=Path("/dev/shm/w/ub/q64gate"))
ap.add_argument("--upstream", action="store_true")
ap.add_argument("--extra", default="")
ap.add_argument("--local-root", default=None,
                help="flashinfer SOURCE tree holding port07/ (dev box). Requires --data: the "
                     "overlay replaces 0.7 headers, and /opt/tiger/flashinfer's own base tree is "
                     "0.3.1, against which the overlay does not compile (prefill.cuh wants "
                     "cp_async::pred_load_32b and BatchPrefillPagedParams::k_page_stride, both "
                     "0.7 additions). So the base must be a 0.7 payload, same as on the H200.")
ap.add_argument("--data", default=None,
                help="flashinfer 0.7 wheel payload dir (the `flashinfer/data` of an unpacked "
                     "flashinfer_python-0.7.*.whl), used as the merge base")
ap.add_argument("--gen", default=None,
                help="the generated batch-attention config dir. PersistentParams is DEFINED "
                     "there, and a 0.3.1 render lacks the fields 0.7 added, so this must be "
                     "named explicitly rather than globbed when both versions sit in the cache "
                     "(see gen_fi07_config_20261028.py)")
ap.add_argument("--std", default="c++20")
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

import torch  # noqa: E402

tdir = os.path.dirname(torch.__file__)
if args.local_root:
    # Dev box: no wheel, so the merged tree is the source tree's own include/ with the port07
    # overlay copied on top -- the same "overlay wins, every header keeps its siblings"
    # construction base_loader._merged_include_tree performs, which no -I order can express
    # because both trees use relative includes.
    import shutil

    top = Path(args.local_root)
    data = Path(args.data) if args.data else None
    assert data and (data / "include" / "flashinfer" / "page.cuh").exists(), \
        "--data must point at an unpacked flashinfer 0.7 wheel payload (see --data help)"
    root = str(top / "port07")
    src = str(top / "port07" / "tests" / "fused_fa3_persistent_ext.cu")
    inc = args.out / "merged"
    if not (inc / "flashinfer" / "attention" / "persistent_fa3_fused.cuh").exists():
        shutil.rmtree(inc, ignore_errors=True)
        shutil.copytree(data / "include", inc)
        shutil.copytree(top / "port07" / "include", inc, dirs_exist_ok=True)
    assert args.gen, "--gen is required with --local-root (see its help)"
    gen = Path(args.gen)
    assert "k_sf_stride_page" in (gen / "batch_attention_config.inc").read_text(), \
        f"{gen} is a pre-0.7 render of batch_attention_config.inc; PersistentParams there is " \
        "missing the fields the overlay needs"
    includes = [str(gen), str(inc), str(data / "csrc"), str(data / "cutlass/include"),
                str(data / "cutlass/tools/util/include"), str(data / "spdlog/include"),
                str(data / "cccl/thrust"), str(data / "cccl/libcudacxx/include"),
                str(data / "cccl/cub")]
else:
    sys.path.insert(0, "/dev/shm/w/native_kernel_sources_20261023/native_fa2_q32_v2_work_20261015")
    import base_loader  # noqa: E402

    root = base_loader.fi_root()
    src = os.path.join(root, "tests", "fused_fa3_persistent_ext.cu")
    gen = base_loader._ensure_generated_config()
    data = base_loader.fi_data_dir()
    assert data, "need the wheel's bundled headers"
    inc = base_loader._merged_include_tree(data, root)
    includes = [gen, inc, f"{data}/csrc", f"{data}/cutlass/include",
                f"{data}/cutlass/tools/util/include", f"{data}/spdlog/include",
                f"{data}/cccl/thrust", f"{data}/cccl/libcudacxx/include", f"{data}/cccl/cub",
                f"{root}/csrc"]
includes = [p for p in includes if os.path.isdir(p)]
includes += [f"{tdir}/include", f"{tdir}/include/torch/csrc/api/include"]

# Copied from base_loader.load_fused_fa3's extra_cuda_cflags, minus the per-variant namespace and
# tile macros which are set per arm below.
FLAGS = [
    f"-std={args.std}", "--expt-relaxed-constexpr", "-arch=sm_90a", "-O3", "--threads", "32",
    "-DFUSED_FA3_STAGES=" + os.environ.get("FA3_STAGES", "2"),
    "-DFUSED_FA3_CONSUMER_REGS=" + os.environ.get("SGLANG_FUSED_FA3_CONSUMER_REGS", "0"),
    "-DFUSED_FA2_W0=128", "-DFUSED_FA2_W1=64", "-DFUSED_FA2_W2=16",
    "-DFUSED_CONST_PERS_PARAMS=0", "-DFUSED_FA2_REG_REALLOC=0",
    "-DFLASHINFER_FA2_MMA_KV=2", "-DFLASHINFER_FA2_QSPLIT=0",
    "-DFUSED_TILE_TRACE=0", "-DFUSED_PIPE_COUNT=0", "-DFUSED_Q_REUSE_FENCE=0",
    "-DFUSED_O_HANDSHAKE=0", "-DFUSED_SPLIT_SMEM_O=0",
    "-DFUSED_NB_SYNC=1", "-DFUSED_GRID_1D=1", "-DFLASHINFER_FUSED_ALIAS_SMEM=1",
    "-DFUSED_FA3_CTA_KV=128",
    "-DFLASHINFER_SKIP_PACKED_Q=0",
    "-DFLASHINFER_FUSED_FA3_CAUSAL=1",
    "-DFLASHINFER_FUSED_TRACE_WRITE=0",
    "-DFLASHINFER_SM90_SUPPRESS_UNSUFFIXED_NAMES=1",
    "-DFLASHINFER_FUSED_UNIFIED_WGMMA=1",
    "-DFLASHINFER_FUSED_CLUSTER=1",
    "-DFLASHINFER_FA3_PAGED_TMA=0", "-DFLASHINFER_FUSED_PAGED_TMA=0",
    "-DFLASHINFER_FUSED_PACKED_Q=4",
    "-DFLASHINFER_FUSED_Q_MBARRIER=1",
    "-DFLASHINFER_FA3_PARTIAL_PROBE=0",
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-Xptxas", "-v",
]
if args.upstream:
    FLAGS += ["-DFLASHINFER_FA3_UPSTREAM=1", "-DFLASHINFER_FA3_UPSTREAM_STAGES=2",
              "-DFLASHINFER_FA3_MMA_PV_RS=1", "-DCUTLASS_ENABLE_GDC_FOR_SM90", "-DNDEBUG"]
if args.extra:
    FLAGS += args.extra.split()

PAT = dict(reg=r"Used (\d+) registers", spill_st=r"(\d+) bytes spill stores",
           spill_ld=r"(\d+) bytes spill loads", stack=r"(\d+) bytes stack frame",
           smem=r"(\d+) bytes smem")
tag = "up" if args.upstream else "fi"
summary = {}
for q in [int(x) for x in args.qs.split(",")]:
    log = args.out / f"ptxas_{tag}_q{q}.log"
    cmd = ["nvcc", *FLAGS, f"-DFUSED_FA3_CTA_Q={q}", f"-DFUSED_OP_NAMESPACE=gate_q{q}", "-c",
           *[f"-I{p}" for p in includes], "-o", str(args.out / f"fused_{tag}_q{q}.o"), src]
    (args.out / f"cmd_{tag}_q{q}.txt").write_text(" ".join(cmd) + "\n")
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    log.write_text(p.stdout + p.stderr)
    print(f"=== CTA_Q={q} upstream={args.upstream} rc={p.returncode} {time.time() - t0:.0f}s -> {log}",
          flush=True)
    if p.returncode != 0:
        for line in (p.stdout + p.stderr).splitlines():
            if "error" in line.lower():
                print("  ERR", line[:240], flush=True)
        continue
    blocks = re.split(r"ptxas info\s*:\s*Compiling entry function ", p.stdout + p.stderr)[1:]
    rows = []
    for b in blocks:
        name = b.split("'")[1] if "'" in b else "?"
        vals = {k: int(m.group(1)) if (m := re.search(v, b)) else 0 for k, v in PAT.items()}
        rows.append((name, vals))
    dem = subprocess.run(["c++filt"], input="\n".join(r[0] for r in rows), capture_output=True,
                         text=True).stdout.splitlines() if rows else []
    for (name, v), d in zip(rows, dem or [r[0] for r in rows]):
        short = d.split("(")[0].split("<")[0].split("::")[-1]
        print(f"  {short[:46]:48} reg={v['reg']:4} spill_st={v['spill_st']:6} "
              f"spill_ld={v['spill_ld']:6} stack={v['stack']:6} smem={v['smem']:7}", flush=True)
        summary.setdefault(short, {})[q] = v
print("\n--- q128 vs q64, per kernel ---", flush=True)
for k, byq in sorted(summary.items()):
    if len(byq) > 1:
        qs = sorted(byq)
        print(f"  {k[:46]:48} " + "  ".join(
            f"q{q}: reg={byq[q]['reg']} spill={byq[q]['spill_st']}/{byq[q]['spill_ld']}" for q in qs),
            flush=True)
print("GATE_DONE", flush=True)
