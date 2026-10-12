#!/bin/bash
# Re-fork restore (2026-10-12): rebuild the private fork tree on its upstream base, then the
# standard restore (venv, sglang from THIS tree, flashinfer 0.7.0.post1, models, ShareGPT).
set -u
W=/dev/shm/w; B=$W/bundle
log() { echo "[restore2 $(date +%H:%M:%S)] $*"; }
BASE=3fef129ff8a32c32da5f6a0efd33485f6cc6233c
if [ ! -f $W/up/.fork_applied ]; then
  rm -rf $W/up; mkdir -p $W/up; cd $W/up
  git init -q && git remote add origin https://github.com/sgl-project/sglang.git
  git fetch -q --depth 1 origin $BASE || { log "FETCH BASE FAIL"; exit 1; }
  git checkout -q FETCH_HEAD || exit 1
  git apply --binary $B/fork_srv.diff || { log "APPLY FAIL"; exit 1; }
  tar xzf $B/untracked.tgz -C $W/up || exit 1
  touch $W/up/.fork_applied
fi
log "fork tree at base $(git -C $W/up rev-parse --short HEAD) + fork diff"
ln -sfn $W/up $W/validated_upstream
mkdir -p $W/drivers $W/workloads/tracelab $W/datasets
[ -d $W/port07 ] || tar xzf $B/port07.tgz -C $W
cp $B/drivers/fused_reference.py $W/drivers/
cp $B/ropecheck.py $B/l2_first_token_ref.py $W/
cp $B/workloads/tracelab/syfi_coding_trace.jsonl.gz $W/workloads/tracelab/
bash $B/restore.sh
ln -sfn $W/port07 $W/fi_colaunch
log "RESTORE2_DONE"
