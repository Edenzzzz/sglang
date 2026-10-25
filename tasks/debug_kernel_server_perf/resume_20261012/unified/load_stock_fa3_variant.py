"""Build stock_fa3_pdl from a given flash-attention tree with extra nvcc defines (A5.5 zero-KV-load ceiling):
  STOCK_FLASH_ATTENTION=<tree> STOCK_FA3_MERGED=<dir> STOCK_FA3_EXTRA="-DFLASHINFER_FA3_SKIP_KV_LOAD=1" python load_stock_fa3_variant.py
Same sources and flags as load_stock_fa3.py plus STOCK_FA3_EXTRA. Claude, 2026-10-25.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_stock_fa3  # noqa: E402

extra = os.environ.get("STOCK_FA3_EXTRA", "").split()
original = load_stock_fa3.load


def load(*args, **kwargs):
    kwargs["extra_cuda_cflags"] = list(kwargs["extra_cuda_cflags"]) + extra
    return original(*args, **kwargs)


load_stock_fa3.load = load
print(load_stock_fa3.build(), "extra", extra, flush=True)
