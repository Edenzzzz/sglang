"""Render flashinfer 0.7's `batch_attention_config.inc` on a box that has no 0.7 install, so the
item-18 gating compile can run off the H200.

Why this is needed: the fused translation unit includes that generated header, and `PersistentParams`
is DEFINED there. The dev box's cache holds a 0.3.1 render, which lacks the fields 0.7 added
(`k_sf_stride_page`, `v_sf_stride_n`, ...), so compiling the overlay against it fails with 54
"class PersistentParams has no member" errors -- a version mismatch in a generated header, not a
problem with the kernel. On the H200 `base_loader._ensure_generated_config()` does this step.

Writes into its own directory and compiles nothing; `gen_batch_attention_module` renders the
templates before `gen_jit_spec` would build, and the spec is discarded.

  python -I gen_fi07_config_20261028.py <unpacked wheel dir> <output dir>

<unpacked wheel dir> is where flashinfer_python-0.7.*.whl was extracted; it goes FIRST on sys.path
so its `flashinfer` wins over any installed one.
Claude, 2026-10-28.
"""
import os
import sys
from pathlib import Path

unpacked, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
assert (unpacked / "flashinfer" / "data" / "csrc").is_dir(), f"not an unpacked 0.7 wheel: {unpacked}"
out.mkdir(parents=True, exist_ok=True)
# Keep the render out of ~/.cache/flashinfer, where a 0.3.1 tree already lives: two configs under
# one glob is how the wrong one gets picked up silently.
os.environ["FLASHINFER_WORKSPACE_BASE"] = str(out)
os.environ["FLASHINFER_JIT_DIR"] = str(out / "jit")
# gen_jit_spec, the last statement of the generator, calls check_cuda_arch, which reads this and
# refuses on a box with no GPU. The templates are rendered BEFORE that call, so the render
# survives either way, but setting the arch keeps the generator on its normal path.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
sys.path.insert(0, str(unpacked))

# flashinfer 0.7's package __init__ pulls in several third-party packages at module scope that
# rendering jinja never needs and this box cannot install (no network): tvm_ffi, reached only by
# `tvm_ffi.load_module(...)` to load an ALREADY BUILT shared object, and pynvml, reached only
# inside functions that query device bandwidth and confidential-compute state.
#
# Each missing one gets a module object that satisfies the import and then raises
# ModuleNotFoundError on any real attribute access. The exception TYPE is the point: a stub that
# returned something usable would turn a missing dependency into a wrong answer, while one raising
# a generic error would break callers that legitimately guard `import x` with `except ImportError`
# and fall back -- torch.hub does exactly that for tqdm.
import types  # noqa: E402


class _Unavailable(types.ModuleType):
    def __getattr__(self, name):
        # Dunders are introspection, not use: `inspect` reads every loaded module's __file__ while
        # building a traceback, so raising on those rewrites any unrelated error into this one.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise ModuleNotFoundError(
            f"{self.__name__}.{name} was used, but {self.__name__} is stubbed out here: this "
            f"script only renders jinja templates, a path that does not reach it. Install the "
            f"real package rather than extending the stub."
        )


for _mod in ("tvm_ffi", "pynvml", "tqdm", "cuda", "einops", "nvidia_cutlass_dsl"):
    try:
        __import__(_mod)
    except ImportError:
        sys.modules[_mod] = _Unavailable(_mod)

import torch  # noqa: E402

import flashinfer  # noqa: E402

assert flashinfer.__version__.startswith("0.7"), f"got flashinfer {flashinfer.__version__}"
assert str(unpacked) in flashinfer.__file__, f"wrong flashinfer on the path: {flashinfer.__file__}"
from flashinfer.jit.attention.modules import gen_batch_attention_module  # noqa: E402

try:
    gen_batch_attention_module(
        dtype_q=torch.bfloat16, dtype_kv=torch.bfloat16, dtype_o=torch.bfloat16,
        dtype_idx=torch.int32, head_dim_qk=128, head_dim_vo=128, pos_encoding_mode=0,
        use_logits_soft_cap=False, use_profiler=False,
    )
except RuntimeError as e:
    # The generator renders every template and only then builds a JitSpec, which we discard.
    # A failure in that trailing step is therefore harmless, and the assertions below -- that the
    # file exists and carries a 0.7-only field -- are what actually decide whether this worked.
    print(f"(generator raised after rendering, continuing to check the render: {e})")
hits = sorted(out.rglob("batch_attention_config.inc"))
assert hits, f"no config rendered under {out}"
inc = hits[0]
# The field whose absence is the whole reason this script exists.
assert "k_sf_stride_page" in inc.read_text(), f"{inc} looks like a pre-0.7 render"
print(f"GEN_DIR {inc.parent}")
