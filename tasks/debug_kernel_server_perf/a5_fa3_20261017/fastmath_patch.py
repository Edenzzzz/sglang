import sys
p = sys.argv[1]; s = open(p).read()
old = '            "-O3",\n'
new = ('            "-O3",\n'
       '            # A5.0b: upstream FA3 (flash-attention hopper, sgl-kernel) builds with --use_fast_math.\n'
       '            *(["--use_fast_math"] if os.environ.get("FUSED_FAST_MATH", "0") == "1" else []),\n')
if "FUSED_FAST_MATH" in s:
    print("already")
else:
    assert s.count(old) == 1
    open(p, "w").write(s.replace(old, new, 1)); print("patched")
