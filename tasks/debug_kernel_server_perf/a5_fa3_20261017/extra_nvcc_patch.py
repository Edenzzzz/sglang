import sys
p = sys.argv[1]; s = open(p).read()
old = '            *(["--use_fast_math"] if os.environ.get("FUSED_FAST_MATH", "0") == "1" else []),\n'
new = old + ('            # A5.0e: extra nvcc flags, e.g. the ones flash-attention / sgl-kernel build FA3 with\n'
             '            # (-DNDEBUG -DCUTLASS_ENABLE_GDC_FOR_SM90 -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED ...).\n'
             '            *os.environ.get("FUSED_EXTRA_NVCC", "").split(),\n')
if "FUSED_EXTRA_NVCC" in s:
    print("already")
else:
    assert s.count(old) == 1
    open(p, "w").write(s.replace(old, new, 1)); print("patched")
