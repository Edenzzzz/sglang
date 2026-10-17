import sys
p = sys.argv[1]; s = open(p).read()
old = '            "-DFLASHINFER_SM90_SUPPRESS_UNSUFFIXED_NAMES=1",\n'
new = ('            "-DFLASHINFER_SM90_SUPPRESS_UNSUFFIXED_NAMES=1",\n'
       '            # A5.0d: FA3 role on flash-attention\'s own mainloop + scheduler (hopper/fa3_upstream.cuh).\n'
       '            "-DFLASHINFER_FA3_UPSTREAM=" + os.environ.get("FA3_UPSTREAM", "0"),\n')
if "FLASHINFER_FA3_UPSTREAM=" in s:
    print("already")
else:
    assert s.count(old) == 1, s.count(old)
    open(p, "w").write(s.replace(old, new, 1)); print("patched")
