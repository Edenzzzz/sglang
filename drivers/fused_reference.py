#!/usr/bin/env python3
"""fp32 CAUSAL reference for the fused FA3+FA2 kernel (K1l).

Deliberately minimal fixture, because fused bugs cluster exactly where the test fixture differs
from sglang's layout:

  * page_size=1 and pages assigned in request order, so the VECTOR-SPARSE kv_indices the FA2
    roles take and the RAW page ids the upstream port takes coincide numerically. That retires
    one of the four aliasing traps instead of leaving it live in the first run.
  * FA3 rows FIRST in q/o, which is what pers_row_offset means.
  * Each side's pages contiguous, so pers_kv_page_offset is just the FA3 page count. Without
    that rebase the FA2 roles read FA3's KV: 1.52 rel err, vs 0.0044 when correct.

Usage: PYTHONPATH=<fork>/python SGLANG_FUSED_FI_ROOT=<flashinfer>/port07 python fused_reference.py
"""
import os, sys, math
import torch

HQ, HKV, D = 28, 7, 128
GQA = HQ // HKV
PAGE = 1
DEV = "cuda"
DT = torch.bfloat16


def fp32_causal_reference(q, k, v, qo_lens, kv_lens, sm_scale):
    """o[row, h, :] over kv[0 .. abs_pos], abs_pos = kv_len - qo_len + i (bottom-right causal)."""
    out = torch.zeros_like(q, dtype=torch.float32)
    qrow = 0
    for b, (ql, kl) in enumerate(zip(qo_lens, kv_lens)):
        kb, vb = k[b][:kl].float(), v[b][:kl].float()      # [kl, HKV, D]
        # Bottom-right causal needs kv_len >= qo_len, or the leading rows have NO causal context
        # and the expected output is undefined. Asserting rather than clamping: a fixture whose
        # expected output is undefined cannot fail a kernel, and silently clamping would hide that.
        # The fa3_1chunk fixture violated this (qo=256, kv=128) and `kb[:pos+1]` with pos<0 WRAPS
        # in Python, so 128 rows were compared against garbage and rows 0 and 128 "passed" by
        # accident -- the whole 254/256 pattern was this bug, not the kernel.
        assert kl >= ql, (
            f"request {b}: kv_len={kl} < qo_len={ql}; bottom-right causal leaves the first "
            f"{ql - kl} rows with no context and the reference undefined. Fix the fixture.")
        for i in range(ql):
            pos = kl - ql + i
            assert pos >= 0, f"internal: pos={pos} should be impossible once kl >= ql"
            for h in range(HQ):
                kh = h // GQA
                s = (kb[: pos + 1, kh, :] @ q[qrow + i, h, :].float()) * sm_scale
                p = torch.softmax(s, dim=0)
                out[qrow + i, h, :] = p @ vb[: pos + 1, kh, :]
        qrow += ql
    return out


def main():
    torch.manual_seed(0)
    from sglang.srt.layers.attention.fused_fa3_loader import load_fused_fa3
    load_fused_fa3(verbose=False)
    import flashinfer

    # 1 long FA3 row + 2 short FA2 rows. FA3 first.
    # MODE=fa2 routes every row to the FA2 classes: no splits, so no merge either. A PASS there
    # isolates any remaining error to FA3 or the merge, because it covers the plan, the three FA2
    # tile classes, the direct-write path and this driver's own stitching. MODE=fa3 is the mirror.
    mode = os.environ.get("MODE", "mixed")
    if mode == "fa2":
        fa3_q, fa3_kv = [], []
        fa2_q, fa2_kv = [256, 1, 1], [2048, 300, 512]
    elif mode == "fa2_1tile":
        # tile-128 q class but a SINGLE kv tile (kv=128 = TILE_KV). Separates "the tile-128 q
        # path" from "a row spanning many kv tiles" -- the two things MODE=fa2 changed at once
        # relative to mixed mode.
        fa3_q, fa3_kv = [], []
        fa2_q, fa2_kv = [256], [128]
    elif mode == "fa2_2tile":
        fa3_q, fa3_kv = [], []
        fa2_q, fa2_kv = [256], [256]
    elif mode == "fa3_1chunk":
        # FA3 only, and UNSPLIT: kv=128 makes the plan's chunk round up to TILE_KV=128, so
        # ceil_div(128,128)=1 -- no partials, no merge. Separates the FA3 mainloop/epilogue from
        # the split-KV machinery, which is the same bisect that isolated the FA2 bugs.
        # kv >= qo so the reference is defined, and kv=128 = TILE_KV so the plan's chunk rounds
        # to one chunk: still unsplit, still no merge, but now a valid causal problem.
        fa3_q, fa3_kv = [128], [128]
        fa2_q, fa2_kv = [], []
    elif mode == "fa3":
        fa3_q, fa3_kv = [256, 1, 1], [2048, 300, 512]
        fa2_q, fa2_kv = [], []
    else:
        fa3_q, fa3_kv = [256], [2048]
        fa2_q, fa2_kv = [1, 1], [300, 512]
    qo_lens = fa3_q + fa2_q
    kv_lens = fa3_kv + fa2_kv
    B = len(qo_lens)
    nfa3 = len(fa3_q)
    sm_scale = 1.0 / math.sqrt(D)
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    sm_split = 16

    total_q = sum(qo_lens)
    total_pages = sum(kv_lens)                       # page_size=1
    q = torch.randn(total_q, HQ, D, device=DEV, dtype=DT) * 0.1
    k_cache = torch.randn(total_pages, HKV, D, device=DEV, dtype=DT) * 0.1
    v_cache = torch.randn(total_pages, HKV, D, device=DEV, dtype=DT) * 0.1

    qo_indptr = torch.tensor([0] + list(torch.tensor(qo_lens).cumsum(0)), dtype=torch.int32)
    kv_indptr = torch.tensor([0] + list(torch.tensor(kv_lens).cumsum(0)), dtype=torch.int32)
    kv_len_t = torch.tensor(kv_lens, dtype=torch.int32)
    mask = torch.zeros(B, dtype=torch.bool)
    mask[:nfa3] = True
    # pages in request order -> page id == element offset at page_size=1
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=DEV)

    IWS, FWS = 64 << 20, 1 << 30
    int_ws = torch.zeros(IWS // 4, dtype=torch.int32, device=DEV)
    pin_ws = torch.zeros(IWS // 4, dtype=torch.int32).pin_memory()
    flt_ws = torch.zeros(FWS // 4, dtype=torch.float32, device=DEV)

    pers_plan = torch.ops.fused_fa3.build_fused_plan(
        int_ws, pin_ws, flt_ws, qo_indptr, kv_indptr, kv_len_t, mask,
        HQ, HKV, D, D, PAGE, sm_split, nsm, True)
    print(f"pers_plan: {len(pers_plan)} entries, num_merge_rows={pers_plan[-2]}, "
          f"num_fa3_clusters={pers_plan[-1]}")

    # FA3's own plan. Always covers at least one request, even when the mask routes nothing to
    # FA3: flashinfer's plan rejects an empty batch, and the wgmma role's work comes from the
    # PLAN'S TASK 3, which the mask leaves empty -- so a dummy plan here drives no work.
    nplan = max(nfa3, 1)
    fa3_ws = torch.zeros(128 << 20, dtype=torch.uint8, device=DEV)
    # SEPARATE workspace for the stock wrapper. Sharing fa3_ws with the fused run let w.plan() and
    # w.run() overwrite the FA3 plan the fused kernel then read: adding the stock null case took
    # fa3_1chunk from 127/128 to 128/128 bad and zeroed the output. Recorded in memory as
    # "flashinfer workspace is not shareable -- a mid-step wrapper needs its own", which I then
    # did exactly.
    stock_ws = torch.zeros(128 << 20, dtype=torch.uint8, device=DEV)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(stock_ws, backend="fa3")
    w.plan(
        qo_indptr=qo_indptr[: nplan + 1].to(DEV),
        paged_kv_indptr=kv_indptr[: nplan + 1].to(DEV),
        paged_kv_indices=kv_indices[: int(kv_indptr[nplan])],
        paged_kv_last_page_len=torch.ones(nplan, dtype=torch.int32, device=DEV),
        num_qo_heads=HQ, num_kv_heads=HKV, head_dim_qk=D, page_size=PAGE,
        causal=True, q_data_type=DT, kv_data_type=DT,
    )
    fa3_plan = list(w._plan_info)
    print(f"fa3_plan: {len(fa3_plan)} entries")

    # Host-side ground truth for the [sqv] probe: q rows the FA3 tiles should load. Tile t of
    # kv head h covers packed rows [128t, 128t+128), i.e. token 32t + (r/4), head h*4 + (r%4).
    for t in (0, 1):
        tok, hd = 32 * t, 0
        print(f"HOST q[tok={tok},head={hd},0..3] = "
              f"{[round(float(x), 6) for x in q[tok, hd, :4]]}")
    # NULL CASE for the FA3 arm of this harness, never run before: does flashinfer's OWN stock
    # FA3 pass this reference? FA2 passing validates the reference for FA2's conventions only. If
    # stock FA3 fails it too, the ~45 residual is a convention mismatch here (layout, scale,
    # masking) rather than a bug in our fused integration -- and every FA3 conclusion in K1n would
    # need re-reading. The wrapper `w` is already planned over the FA3 subset above.
    o_stock_ok = False
    if nfa3 > 0:
        try:
            o_stock = w.run(q[: sum(fa3_q)], (k_cache, v_cache)).clone()
            o_stock_ok = True
            kpad_s = [k_cache[kv_indptr[b]:kv_indptr[b + 1]] for b in range(nplan)]
            vpad_s = [v_cache[kv_indptr[b]:kv_indptr[b + 1]] for b in range(nplan)]
            ref_s = fp32_causal_reference(q, kpad_s, vpad_s, fa3_q[:nplan], fa3_kv[:nplan], sm_scale)
            # Confirm the two tensors describe the same object BEFORE any norm. Skipping this is
            # what made the previous null case self-contradictory: a rel of 0.001779 alongside an
            # element print of all zeros for the same tensor.
            print(f"  o_stock: shape={tuple(o_stock.shape)} dtype={o_stock.dtype} "
                  f"absmax={float(o_stock.float().abs().max()):.6f}")
            print(f"  ref_s:   shape={tuple(ref_s.shape)} dtype={ref_s.dtype} "
                  f"absmax={float(ref_s.abs().max()):.6f}")
            print(f"  o_stock[0,0,:3]={[round(float(x),6) for x in o_stock.reshape(o_stock.shape[0], -1)[0, :3]]}"
                  f"  ref_s[0,0,:3]={[round(float(x),6) for x in ref_s.reshape(ref_s.shape[0], -1)[0, :3]]}")
            n = min(o_stock.shape[0], ref_s.shape[0])
            num = (o_stock[:n].float() - ref_s[:n]).norm().item()
            den = max(ref_s[:n].norm().item(), 1e-6)
            print(f"STOCK FA3 vs reference: rel={num / den:.4g}  "
                  f"{'PASS -> reference is valid for FA3' if num / den <= 2e-2 else 'FAIL -> reference convention does NOT match FA3'}")
        except Exception as e:
            print(f"STOCK FA3 null case could not run: {type(e).__name__}: {e}")

    o = torch.zeros_like(q)
    pers_o = torch.zeros_like(q)
    nfa3_rows = sum(fa3_q)
    torch.ops.fused_fa3.fused_run(
        q, k_cache, v_cache, kv_indices, o, fa3_ws, fa3_plan,
        int_ws, flt_ws, pers_plan, HQ, HKV, PAGE, sm_scale,
        sm_split, nfa3_rows, sm_split, sum(fa3_kv), kv_indices, pers_o,
        B, max(kv_lens), max(qo_lens), kv_len_t.to(DEV),
        qo_indptr.to(DEV), kv_indptr.to(DEV),
    )
    torch.cuda.synchronize()

    # FA3 writes o, the FA2 roles write pers_o at pers_row_offset; stitch them.
    got = o.clone().float()
    got[nfa3_rows:] = pers_o[nfa3_rows:].float()

    kpad = [k_cache[kv_indptr[b]:kv_indptr[b + 1]] for b in range(B)]
    vpad = [v_cache[kv_indptr[b]:kv_indptr[b + 1]] for b in range(B)]
    ref = fp32_causal_reference(q, kpad, vpad, qo_lens, kv_lens, sm_scale)

    # Elementwise ref-vs-got for token 0 at each kv-head group, so a [wpk] line with tok=0 can be
    # matched by its kvh. Direct comparison, no inference: the store addresses are verified exact
    # end to end, so any difference here is the computed VALUE.
    print("ROW0 ref vs got, token 0, head = kvh*4, first 4 elements:")
    for kvh in range(HKV):
        h = kvh * GQA
        r4 = [round(float(x), 6) for x in ref[0, h, :4]]
        g4 = [round(float(x), 6) for x in got[0, h, :4]]
        print(f"  kvh={kvh} h={h:2d} ref={r4} got={g4}")
    # OURS vs the STOCK oracle, as a rel with the shape/absmax check AT the site -- the pattern
    # that made the null case trustworthy, rather than the element print that disagreed with it.
    # absmax of each side is the most informative single number: if ours is ~0 the fused path
    # wrote nothing; if it is ~stock the values are right and the placement is wrong; if it is far
    # larger the scaling is wrong.
    if nfa3 > 0 and o_stock_ok:
        g = got[: o_stock.shape[0]].float()
        t = o_stock.float()
        print(f"OURS vs STOCK: ours absmax={float(g.abs().max()):.6f} "
              f"stock absmax={float(t.abs().max()):.6f} "
              f"ours shape={tuple(g.shape)} stock shape={tuple(t.shape)}")
        print(f"  rel(ours, stock) = "
              f"{(g - t).norm().item() / max(t.norm().item(), 1e-6):.6g}")
        nz = int((g.abs().sum(dim=(1, 2)) > 0).sum())
        print(f"  ours: {nz}/{g.shape[0]} rows non-zero")

    # (An ours-vs-stock element print lived here and read +0.000000 on BOTH sides while the
    # validated check at the stock call, in the same run, read o_stock[0,0,0]=0.253906. The clone
    # and the gate are both correct on inspection, so it is a defect in the print site itself.
    # Removed rather than debugged: it was a convenience for localisation, the stock oracle is
    # already usable through its rel against the reference, and instrumentation that disagrees
    # with a validated read is worse than none. Re-add only with a shape/dtype/absmax check at
    # the print site, which is what made the null case trustworthy.)
    rows_bad = 0
    worst = 0.0
    for r in range(total_q):
        num = (got[r] - ref[r]).norm().item()
        den = max(ref[r].norm().item(), 1e-6)
        rel = num / den
        worst = max(worst, rel)
        if rel > 2e-2:
            rows_bad += 1
            if rows_bad <= 5:
                print(f"  row {r} rel={rel:.4g}")
    # last rows explicitly: they are the FA2 rows in mixed mode, and they never touch the merge,
    # so their error is diagnostic of a different thing than the FA3 rows'.
    for r in range(max(0, total_q - 3), total_q):
        num = (got[r] - ref[r]).norm().item()
        print(f"  tail row {r}: rel={num / max(ref[r].norm().item(), 1e-6):.4g}")
    # WHICH rows pass. 254/256 bad has survived every change to the FA3 path, so the identity of
    # the 2 correct rows is a sharper constraint than the 254 wrong ones: at gqa=4 and CTA_Q=128 a
    # tile covers 32 tokens, so 2 is not a tile boundary and the pattern should name the mechanism.
    good = [r for r in range(total_q)
            if (got[r] - ref[r]).norm().item() / max(ref[r].norm().item(), 1e-6) <= 2e-2]
    print(f"passing rows ({len(good)}): {good[:24]}{' ...' if len(good) > 24 else ''}")
    if good:
        print(f"  first={good[0]} last={good[-1]} mod32={sorted({g % 32 for g in good})} "
              f"mod128={sorted({g % 128 for g in good})}")
    print(f"\nmode={mode} worst rel err {worst:.4g}, bad rows {rows_bad}/{total_q}")
    print("PASS" if rows_bad == 0 else "FAIL")
    return 0 if rows_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
