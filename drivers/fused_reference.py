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
        for i in range(ql):
            pos = kl - ql + i
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
        fa3_q, fa3_kv = [256], [128]
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
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(fa3_ws, backend="fa3")
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
