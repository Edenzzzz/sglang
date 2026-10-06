#!/bin/bash
# Rebuild our sglang python/ tree on the pod: upstream at the fork's last upstream base
# commit, plus the fork's 34-file python diff (103 KB) -- instead of pushing the 5.8 MB tree
# through the webshell. Fetch-by-SHA is rejected by GitHub, so blob-filtered clone + checkout.
set -e
W=/dev/shm/w
BASE=53b636b104aac3708894b705720a7e63b5a998a3
rm -rf $W/sgl_src
git clone --filter=blob:none --no-checkout https://github.com/sgl-project/sglang $W/sgl_src
cd $W/sgl_src
git sparse-checkout init --cone
git sparse-checkout set python
git checkout $BASE
gunzip -c $W/fork_python.diff.gz > $W/fork_python.diff
git apply --check $W/fork_python.diff && git apply $W/fork_python.diff
rm -rf $W/python && cp -r $W/sgl_src/python $W/python
echo "applied; files: $(find $W/python/sglang -name '*.py' | wc -l)"
test -f $W/python/sglang/srt/layers/attention/flashinfer_backend.py
grep -c "SGLANG_TWO_EXEC" $W/python/sglang/srt/layers/attention/flashinfer_backend.py
echo TREEBUILT
