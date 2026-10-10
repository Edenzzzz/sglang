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

# K1n A/B: GQA1=1 collapses the group so packed rows and tokens coincide (G=1). That is the
# only configuration in which FUSED_PACKED_Q=off is a valid comparison -- the unified
# scheduler reports the PACKED extent as qo_len and the stock TMA/write_tiled paths read it
# as tokens, so the two only agree when G=1.
import os as _os
HQ, HKV, D = (7, 7, 128) if _os.environ.get('GQA1') == '1' else (28, 7, 128)
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
    elif mode == "mix_decode":
        # DECODE-HEAVY 4-mode shape: one prefill chunk plus DEC_ROWS decode rows. This is the arm
        # that decides whether the fusion's +5.4% was a floor set by having only 2 decode tokens to
        # overlap, or the real margin. MIX_ALL_FA3=1 routes everything to FA3 (the "no fusion"
        # control, same rows); otherwise the chunk goes to FA3 and the decodes to FA2.
        _dr = int(_os.environ.get("DEC_ROWS", "32"))
        _chunk, _ctx = 256, 2048
        if _os.environ.get("MIX_ALL_FA3") == "1":
            fa3_q, fa3_kv = [_chunk] + [1] * _dr, [_ctx] + [_ctx] * _dr
            fa2_q, fa2_kv = [], []
        else:
            fa3_q, fa3_kv = [_chunk], [_ctx]
            fa2_q, fa2_kv = [1] * _dr, [_ctx] * _dr
    elif mode == "band1":
        # ONE row, declared on whichever side BAND_SIDE names, so the driver's stitching stays
        # valid (it assumes FA3 rows are a PREFIX of q/o). BAND_QO sets the q length, giving
        # packed_qo = BAND_QO * 4 at GQA 4.
        #
        # This is the correct way to trace the dispatch curve. Overriding the routing inside the
        # PLAN (FUSED_FA3_MIN_PACKED) does not work with this harness: the driver computes
        # `got = o; got[nfa3_rows:] = pers_o[nfa3_rows:]` from the FIXTURE's nfa3, so a row the
        # plan moves to FA3 writes into `o` and is then discarded. That is why the MIN_PACKED
        # sweep returned bad rows in exact proportion to the rows moved (5->64/64, 4->60, 3->52,
        # 2->40, 1->24, 0->pass) -- a harness artifact, NOT a kernel defect.
        # BAND_KV sweeps the SECOND axis. A single-kv threshold cannot be the dispatch rule: at
        # kv=2048 FA3 wins across the whole packed_qo band, but the project's earlier decode-only
        # result has FA2 ahead by 1.30x at qo=1 / short kv. The boundary is 2-D.
        # BAND_ROWS sweeps the THIRD axis, and it is the one that decides the rule. A single row
        # makes FA3 look unconditionally better, but FA2's tile16 class PACKS many short rows and
        # parallelises across them, while an FA3 tile of 128 packed rows is mostly idle on one row
        # and well filled by 32 batched decode tokens. The doc's earlier "decode-only: FA2 beats
        # FA3 by 1.30x" was presumably at a real decode batch size, which band1 never had.
        _bq = int(_os.environ.get("BAND_QO", "16"))
        _bkv = max(int(_os.environ.get("BAND_KV", "2048")), _bq)
        _brows = int(_os.environ.get("BAND_ROWS", "1"))
        if _os.environ.get("BAND_SIDE", "fa2") == "fa3":
            fa3_q, fa3_kv = [_bq] * _brows, [_bkv] * _brows
            fa2_q, fa2_kv = [], []
        else:
            fa3_q, fa3_kv = [], []
            fa2_q, fa2_kv = [_bq] * _brows, [_bkv] * _brows
    elif mode == "band":
        # THE CONTESTED DISPATCH BAND, which no previous fixture covered. A 4-mode verify row is
        # qo = num_draft_tokens against a long context, so at GQA 4 these give
        # packed_qo = 16 / 32 / 48 / 64 / 96 -- exactly the [16,128) range where the FA3-vs-FA2
        # decision is made and where `MIN_PACKED_QO = 48` currently admits rows into a 128-wide
        # WGMMA tile they cannot fill.
        #
        # All rows start on the FA2 side; FUSED_FA3_MIN_PACKED=<n> moves those with packed_qo >= n
        # to FA3, so sweeping n traces the real dispatch curve instead of assuming a threshold.
        fa3_q, fa3_kv = [], []
        fa2_q, fa2_kv = [4, 8, 12, 16, 24], [2048, 2048, 2048, 2048, 2048]
    elif mode == "fa3_multitile":
        # MULTI q tile, NO split: 256 tokens * G=4 = 1024 packed rows = 8 q tiles of CTA_Q=128,
        # while kv=256 keeps the plan at one chunk. The `fa3` fixture changes three things at once
        # (multi-request, multi q tile, split kv); this holds split and batch fixed so a failure
        # here means the multi-tile path itself, and a PASS hands the blame to split/merge.
        fa3_q, fa3_kv = [256], [256]
        fa2_q, fa2_kv = [], []
    elif mode == "fa3_split":
        # The mirror: ONE q tile (32 tokens * 4 = 128 packed rows = exactly CTA_Q) with a kv long
        # enough that the plan must chunk it. Isolates store_partial + merge_indptr/
        # merge_o_indices + the reduction from everything about q tiling.
        fa3_q, fa3_kv = [32], [2048]
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
    # SM_SPLIT=all makes every block an FA3 role, so the FA2 roles never co-reside. A true A/B of
    # "fused context" vs "FA3 alone" on the SAME build: if the accumulator duplication disappears,
    # coexistence with the FA2 role causes it (shared smem, warpgroup arrangement); if it
    # persists, it is in fa3_prefill_body independent of the fusion. Cheaper than a second build
    # configuration, which is the only other way to separate these.
    # SM_SPLIT: "all" gives every SM to the FA3 role, an integer sets the boundary directly, and
    # the default of 16 is what the first timing A/B exposed as a likely misallocation -- FA3 held
    # essentially all the work on 12% of the SMs. Numeric so the split can be swept, since SM
    # pre-assignment is the thing this kernel exists to control.
    _sms = os.environ.get("SM_SPLIT", "16")
    sm_split = nsm if _sms == "all" else max(1, min(int(_sms), nsm))

    total_q = sum(qo_lens)
    total_pages = sum(kv_lens)                       # page_size=1
    # SEEDED. This fixture was unseeded, so every run drew different q/k/v -- which is how a
    # one-bad-row intermittent failure in MODE=fa3 got reported as "every fixture passes" from a
    # single run. A pass on an unseeded fixture is one sample, not a verdict. SEED=<n> to sweep.
    torch.manual_seed(int(_os.environ.get("SEED", "0")))
    q = torch.randn(total_q, HQ, D, device=DEV, dtype=DT) * 0.1
    k_cache = torch.randn(total_pages, HKV, D, device=DEV, dtype=DT) * 0.1
    v_cache = torch.randn(total_pages, HKV, D, device=DEV, dtype=DT) * 0.1
    # K1n discriminator: KVCONST=1 makes every page identical, so a kernel that reads the WRONG
    # kv row still gets the RIGHT values. "row 0 exact, the rest wrong and roughly proportional"
    # is the signature of KV being correct for the first key and garbage beyond -- under
    # bottom-right causal, row 0 attends to key 0 alone. PASS here => the defect is in WHICH kv
    # rows are read; FAIL => kv values are not the cause. This is what localised the unassigned
    # k_page_stride in one run after many turns of probing the accumulator.
    # Stated limit: with constant K the softmax weights are uniform, so reading too FEW keys also
    # passes. It isolates addressing, not the iteration count.
    if _os.environ.get("KVCONST") == "1":
        k_cache[:] = k_cache[0:1].clone()
        v_cache[:] = v_cache[0:1].clone()

    qo_indptr = torch.tensor([0] + list(torch.tensor(qo_lens).cumsum(0)), dtype=torch.int32)
    kv_indptr = torch.tensor([0] + list(torch.tensor(kv_lens).cumsum(0)), dtype=torch.int32)
    kv_len_t = torch.tensor(kv_lens, dtype=torch.int32)
    mask = torch.zeros(B, dtype=torch.bool)
    mask[:nfa3] = True
    # pages in request order -> page id == element offset at page_size=1
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=DEV)

    # KVPERREQ=1: pages identical WITHIN a request, distinct ACROSS requests. This bisects the
    # kv-ADDRESSING error that KVCONST localised (baseline PASSES 8/8 with all pages identical and
    # FAILS 10/10 with distinct pages, so CTAs read the wrong kv ROWS).
    #
    #   corruption VANISHES  -> the stale rows come from ANOTHER REQUEST: cross-item leakage, i.e.
    #                           a stale K/V tile left in smem by a previous work item.
    #   corruption PERSISTS  -> the wrong rows are WITHIN the request: a bad tile index, and
    #                           cross-item reuse is not the route.
    #
    # One bit, no kernel change, and nothing that serialises -- every instrument that slowed the
    # kernel down (printf probe, racecheck, memcheck) masked this defect, so the discriminator has
    # to live in the DATA rather than in the timing.
    # KVPERTILE=1: pages identical within each CTA_KV-sized TILE, distinct across tiles. This
    # splits the corrected localisation (wrong row WITHIN the request) into its two mechanisms:
    #
    #   corruption VANISHES  -> the wrong row is inside the RIGHT tile: a lane/row addressing error
    #                           (the __shfl_sync src_thread mapping, or kv_offset within the tile).
    #   corruption PERSISTS  -> the wrong TILE is being loaded: the my_kv_offset[parity] rolling
    #                           buffer, which is also the only candidate that explains the 4x
    #                           pipeline-depth scaling by itself.
    #
    # Tile width must match the kernel's CTA_KV (128 by default), read from the env so the two
    # cannot drift silently.
    if _os.environ.get("KVPERTILE") == "1":
        _tw = int(_os.environ.get("FA3_CTA_KV", "128"))
        for _i in range(B):
            _a, _b = int(kv_indptr[_i]), int(kv_indptr[_i + 1])
            for _t0 in range(_a, _b, _tw):
                _t1 = min(_t0 + _tw, _b)
                k_cache[_t0:_t1] = k_cache[_t0:_t0 + 1].clone()
                v_cache[_t0:_t1] = v_cache[_t0:_t0 + 1].clone()

    if _os.environ.get("KVPERREQ") == "1":
        for _i in range(B):
            _a, _b = int(kv_indptr[_i]), int(kv_indptr[_i + 1])
            if _b > _a:
                k_cache[_a:_b] = k_cache[_a:_a + 1].clone()
                v_cache[_a:_b] = v_cache[_a:_a + 1].clone()

    # FWS was 1 GB and that is what blocked every fixture past qo~2048: the plan builder returned
    # "FusedFourTaskPlan failed: invalid argument" because the split-KV partial slab did not fit.
    # At qo=2048 / kv=8192 the planner picks a 128-wide kv chunk, giving 64 chunks x 64 q tiles, so
    # partial_o alone needs ~939 MB (524288 partial rows x 7 kv heads x 128 dims x 2 B) plus ~15 MB
    # of partial_lse. That is a HARNESS limit, not a kernel one -- H200 has 143 GB.
    #
    # Worth recording as a finding in its own right: the partial slab grows as
    # O(packed_qo x num_kv_chunks), i.e. quadratically in context length for a fixed chunk width.
    # Nearly a gigabyte of partials for ONE 2048-token chunk at 8K context is a real serving
    # concern, independent of the correctness work.
    IWS, FWS = 256 << 20, 8 << 30
    int_ws = torch.zeros(IWS // 4, dtype=torch.int32, device=DEV)
    pin_ws = torch.zeros(IWS // 4, dtype=torch.int32).pin_memory()
    flt_ws = torch.zeros(FWS // 4, dtype=torch.float32, device=DEV)

    pers_plan = torch.ops.fused_fa3.build_fused_plan(
        int_ws, pin_ws, flt_ws, qo_indptr, kv_indptr, kv_len_t, mask,
        HQ, HKV, D, D, PAGE, sm_split, nsm, True)
    print(f"pers_plan: {len(pers_plan)} entries, num_merge_rows={pers_plan[-2]}, "
          f"num_fa3_clusters={pers_plan[-1]}")

    # PLANMAP=1: decode the plan ON THE HOST and print, per block, the FA3 work items it owns and
    # the output rows each item covers. This replaces the device-side [item] trace, which could not
    # work: correlating bad rows to items needs per-item identity, device printf masks the race (952
    # prints took the failing arm to 0/384), so the correlation is impossible from inside the
    # kernel. The plan is built here on the host before launch, so reading it perturbs NOTHING.
    #
    # Layout of FusedPlanInfo<4> (fused_plan.cuh): [0]=num_blks_x, [1]=num_blks_y, then tasks[4]
    # x 11 int64 offsets, so task t field f sits at 2 + t*11 + f with
    #   f: 0=q_indptr 1=kv_indptr 2=partial_indptr 3=q_len 4=kv_len 5=q_start
    #      6=kv_start 7=kv_end 8=kv_head_idx 9=batch_idx 10=work_indptr
    # Task 3 is the wgmma/FA3 queue, which is what MIX_ALL_FA3=1 routes everything to.
    if _os.environ.get("PLANMAP") == "1":
        _iw = int_ws.view(torch.int32).cpu()
        _T = 3
        # The plan's *_offset fields are BYTE offsets into int_ws, not int32 indices: the first
        # decode read them as element indices and got all zeros for work_indptr. The stride
        # between consecutive field offsets is 3808 bytes = 952 int32s, and 952 is exactly the
        # item count, which is what identified the units.
        _off = {k: int(pers_plan[2 + _T * 11 + i]) // 4 for i, k in enumerate(
            ["q_indptr", "kv_indptr", "partial_indptr", "q_len", "kv_len", "q_start",
             "kv_start", "kv_end", "kv_head_idx", "batch_idx", "work_indptr"])}
        print(f"[planmap] task3 int32 indices (byte offsets / 4): {_off}")
        _nblk = int(pers_plan[0]) * max(1, int(pers_plan[1]))
        _wi = _iw[_off["work_indptr"]: _off["work_indptr"] + _nblk + 1].tolist()
        print(f"[planmap] num_blks={_nblk} work_indptr[:12]={_wi[:12]} total_items={_wi[-1] if _wi else 0}")
        _n = _wi[-1] if _wi else 0
        _qs = _iw[_off["q_start"]: _off["q_start"] + _n].tolist()
        _bi = _iw[_off["batch_idx"]: _off["batch_idx"] + _n].tolist()
        _kl = _iw[_off["kv_len"]: _off["kv_len"] + _n].tolist()
        # item -> owning block, by inverting work_indptr
        _owner = {}
        for _b in range(_nblk):
            for _it in range(_wi[_b], _wi[_b + 1]):
                _owner[_it] = _b
        print(f"[planmap] first 8 items: " + " ".join(
            f"(it={i} blk={_owner.get(i)} q_start={_qs[i] if i < len(_qs) else '?'} "
            f"batch={_bi[i] if i < len(_bi) else '?'} kv_len={_kl[i] if i < len(_kl) else '?'})"
            for i in range(min(8, _n))))
        _PLANMAP = {"owner": _owner, "q_start": _qs, "batch_idx": _bi, "kv_len": _kl,
                    "work_indptr": _wi, "nblk": _nblk}
    else:
        _PLANMAP = None

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

    # UNFUSED baseline's FA2 half, for TIME_AB. Its own workspace, for the same reason stock_ws
    # exists: flashinfer workspaces are not shareable, and a second wrapper sharing one silently
    # overwrites the first's plan.
    w2 = None
    if B - nfa3 > 0:
        fa2_ws_stock = torch.zeros(128 << 20, dtype=torch.uint8, device=DEV)
        qo2 = (qo_indptr[nfa3:] - qo_indptr[nfa3]).to(DEV)
        kv2_indptr = (kv_indptr[nfa3:] - kv_indptr[nfa3]).to(DEV)
        w2 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(fa2_ws_stock, backend="fa2")
        w2.plan(
            qo_indptr=qo2,
            paged_kv_indptr=kv2_indptr,
            paged_kv_indices=kv_indices[int(kv_indptr[nfa3]):int(kv_indptr[B])],
            paged_kv_last_page_len=torch.ones(B - nfa3, dtype=torch.int32, device=DEV),
            num_qo_heads=HQ, num_kv_heads=HKV, head_dim_qk=D, page_size=PAGE,
            causal=True, q_data_type=DT, kv_data_type=DT,
        )

    # Host-side ground truth for the [sqv] probe: q rows the FA3 tiles should load. Tile t of
    # kv head h covers packed rows [128t, 128t+128), i.e. token 32t + (r/4), head h*4 + (r%4).
    # Guarded on q's actual extent: this printed tile 1's base token unconditionally, which
    # IndexErrors on any fixture with fewer than 33 tokens (fa3_split has 32) -- a diagnostic that
    # kills the run before the measurement it was meant to explain.
    for t in (0, 1):
        tok, hd = 32 * t, 0
        if tok < q.shape[0]:
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

    # ---- TIME_AB=1: first timing A/B, one FUSED launch vs the UNFUSED baseline ------------
    # Runs only AFTER the correctness comparison below has been printed, so a timing number can
    # never be reported for an arm that does not pass -- the rule this project keeps relearning.
    # Composition is printed alongside, because a speedup is meaningless without knowing how many
    # rows actually went to each role.
    def _time_ab():
        import torch as _t
        iters = int(_os.environ.get("TIME_ITERS", "50"))
        warmup = int(_os.environ.get("TIME_WARMUP", "10"))

        # HOIST the H2D copies out of the timed region. `kv_len_t.to(DEV)`,
        # `qo_indptr.to(DEV)` and `kv_indptr.to(DEV)` were inside the closure, so the FUSED arm
        # paid three host->device transfers PER ITERATION that the unfused baseline never pays.
        # The tell was that fused time was nearly CONSTANT across configurations (0.3053, 0.3057,
        # 0.2919 ms) while unfused tracked the actual work (0.0378 - 0.0682) -- a fixed cost
        # dominating, which is an artifact of the harness, not a property of the kernel.
        _kvlen_d = kv_len_t.to(DEV)
        _qoi_d = qo_indptr.to(DEV)
        _kvi_d = kv_indptr.to(DEV)
        _maxkv, _maxqo, _sumkv = max(kv_lens), max(qo_lens), sum(fa3_kv)

        def _fused():
            torch.ops.fused_fa3.fused_run(
                q, k_cache, v_cache, kv_indices, o, fa3_ws, fa3_plan,
                int_ws, flt_ws, pers_plan, HQ, HKV, PAGE, sm_scale,
                sm_split, nfa3_rows, sm_split, _sumkv, kv_indices, pers_o,
                B, _maxkv, _maxqo, _kvlen_d, _qoi_d, _kvi_d)

        # UNFUSED baseline: the two roles as SEPARATE launches, which is what the fused kernel is
        # meant to replace. FA3 rows through flashinfer's own planned FA3 wrapper, FA2 rows through
        # a second wrapper planned over the FA2 subset. Two launches, serialised on one stream.
        qf = q[:nfa3_rows]
        def _unfused():
            if nfa3_rows > 0:
                w.run(qf, (k_cache, v_cache))
            if B - nfa3 > 0:
                w2.run(q[nfa3_rows:], (k_cache, v_cache))

        # THE RIGHT CONTROL, never run before. The unfused arm above uses
        # BatchPrefillWithPagedKVCacheWrapper -- a DIFFERENT kernel family from our FA2 runners,
        # which are derived from flashinfer's PERSISTENT BatchAttention. So "FA2-only is 20x
        # slower than unfused" may be comparing against the wrong thing. Time BatchAttention on
        # the same fixture: if it is also ~1 ms, the cost is inherent to the persistent kernel at
        # this shape and our port is fine; if it is ~0.055 ms, our port of it is genuinely 20x off.
        _ba = None
        try:
            from flashinfer.attention import BatchAttention
            _ba = BatchAttention(kv_layout="NHD", device=DEV)
            _ba.plan(qo_indptr.to(DEV), kv_indptr.to(DEV),
                     kv_indices[: int(kv_indptr[B])], kv_len_t.to(DEV),
                     HQ, HKV, D, D, PAGE, causal=True, sm_scale=sm_scale,
                     q_data_type=DT, kv_data_type=DT)
        except Exception as e:
            print(f"[time] BatchAttention control unavailable: {type(e).__name__}: {e}")
            _ba = None

        def _bench(fn):
            for _ in range(warmup):
                fn()
            _t.cuda.synchronize()
            st, en = _t.cuda.Event(True), _t.cuda.Event(True)
            st.record()
            for _ in range(iters):
                fn()
            en.record()
            _t.cuda.synchronize()
            return st.elapsed_time(en) / iters

        try:
            t_f = _bench(_fused)
        except Exception as e:
            print(f"[time] FUSED arm failed: {type(e).__name__}: {e}")
            return
        try:
            t_u = _bench(_unfused)
        except Exception as e:
            print(f"[time] UNFUSED baseline could not run: {type(e).__name__}: {e}  "
                  f"(fused={t_f:.4f} ms, no comparison)")
            return
        print(f"[time] composition: {nfa3} FA3 requests ({nfa3_rows} tokens), "
              f"{B - nfa3} FA2 requests ({sum(qo_lens) - nfa3_rows} tokens), "
              f"kv={kv_lens}, sm_split={sm_split}/{nsm}")
        print(f"[time] fused   = {t_f:.4f} ms/iter  ({iters} iters, {warmup} warmup)")
        print(f"[time] unfused = {t_u:.4f} ms/iter  (two separate launches)")
        print(f"[time] speedup = {t_u / t_f:.4f}x  "
              f"({100.0 * (t_u - t_f) / t_u:+.2f}% latency)")
        if _ba is not None:
            # CORRECTNESS GATE on the control, before its timing is usable as a baseline. A faster
            # arm that computes something cheaper is not a baseline. Compared against the same
            # fp32 causal reference the fused arm is held to.
            try:
                _bo = _ba.run(q, (k_cache, v_cache))
                _bo = _bo[0] if isinstance(_bo, tuple) else _bo
                _bn = (_bo.float() - ref).norm().item()
                _bd = max(ref.norm().item(), 1e-6)
                _brel = _bn / _bd
                _bbad = 0
                for _r in range(total_q):
                    _n = (_bo[_r].float() - ref[_r]).norm().item()
                    if _n / max(ref[_r].norm().item(), 1e-6) > 2e-2:
                        _bbad += 1
                print(f"[time] BatchAttention CORRECTNESS: rel={_brel:.6f} bad_rows={_bbad}/{total_q}"
                      + ("  VALID BASELINE" if _bbad == 0 else "  *** NOT A VALID BASELINE ***"))
            except Exception as e:
                print(f"[time] BatchAttention correctness check failed: {type(e).__name__}: {e}")
            try:
                t_b = _bench(lambda: _ba.run(q, (k_cache, v_cache)))
                print(f"[time] BatchAttention (flashinfer's OWN persistent kernel, the family our "
                      f"FA2 runners come from) = {t_b:.4f} ms/iter")
                print(f"[time]   fused/BatchAttention = {t_f / t_b:.4f}x, "
                      f"BatchAttention/unfused = {t_b / t_u:.4f}x")
            except Exception as e:
                print(f"[time] BatchAttention run failed: {type(e).__name__}: {e}")

    # Output fingerprint, so two builds can be compared for "did this change anything at all".
    # A diagnostic arm that removes work (SKIP_PACKED_Q=1) must change the output; if the
    # fingerprint is identical the arm is INERT and any timing delta from it is meaningless.
    print(f"[fp] got absmax={float(got.abs().max()):.8f} sum={float(got.sum()):.6f} "
          f"nonzero={int((got != 0).sum())}")
    kpad = [k_cache[kv_indptr[b]:kv_indptr[b + 1]] for b in range(B)]
    vpad = [v_cache[kv_indptr[b]:kv_indptr[b + 1]] for b in range(B)]
    # ---- DUMP_PARTIALS=1: is it the PARTIALS or the COMBINATION? -------------------------
    # Every structural hypothesis about split-KV was refuted by reading both sides (addressing on
    # the partial and merge sides, buffer identity, LSE scale, finalize ordering, CSR consistency,
    # grid coverage, double writes). So stop reading and compare DATA.
    #
    # partial_o holds a NORMALIZED per-chunk output, so comparing it needs no LSE convention --
    # that is what makes this the clean split:
    #   partial_o matches  => the partials are right, the COMBINATION is wrong.
    #   partial_o differs  => the partials are wrong and the merge is innocent.
    #
    # Plan vector layout (ToVector): 0 num_blks_x, 1 num_blks_y, 2..45 four tasks x 11 args,
    # then 46 len_kv_chunk, 47 partial_o, 48 partial_lse, 49 merge_indptr, 50 merge_o_indices,
    # ... 57 num_merge_rows, 58 num_fa3_clusters.
    if _os.environ.get("DUMP_PARTIALS") == "1" and nfa3 > 0:
        import torch as _t
        po_off, plse_off = int(pers_plan[47]), int(pers_plan[48])
        mi_off, moi_off = int(pers_plan[49]), int(pers_plan[50])
        nmr = int(pers_plan[57])
        # Offsets are BYTES from each buffer's base.
        fb = flt_ws.view(torch.uint8)
        partial_o = fb[po_off:].view(DT)                      # bf16, [slot, HEAD_DIM] flattened
        partial_lse = fb[plse_off:].view(torch.float32)
        ib = int_ws.view(torch.uint8)
        merge_indptr = ib[mi_off:].view(torch.int32)[: nmr + 1].cpu()
        merge_o_ind = ib[moi_off:].view(torch.int32)[: nmr].cpu()
        print(f"[dump] num_merge_rows={nmr} merge_indptr[:6]={merge_indptr[:6].tolist()} "
              f"merge_o_indices[:6]={merge_o_ind[:6].tolist()}")
        if nmr > 0:
            chunks = int(merge_indptr[1] - merge_indptr[0])
            print(f"[dump] chunks per row = {chunks}")
            b = 0
            ql, kl = qo_lens[b], kv_lens[b]
            kb = k_cache[kv_indptr[b]:kv_indptr[b + 1]][:kl].float()
            vb = v_cache[kv_indptr[b]:kv_indptr[b + 1]][:kl].float()
            # EXPECTED, stated before reading: merge row r is packed row r of this group, i.e.
            # token r // GQA and head-in-group r % GQA. Slot for (row r, chunk c, kv head h) is
            # (merge_indptr[r] + c) * HKV + h, and partial_o at that slot is the NORMALIZED
            # attention of that (token, head) over keys [128c, 128c+128) intersected with the
            # causal bound kv_len - qo_len + token.
            # SLAB FINGERPRINT across runs, to separate the two remaining possibilities:
            #   identical slab + differing output => the defect is in the REDUCTION
            #   differing slab                    => the defect is upstream in the attention grid
            # Finite-masked so the intentional -inf entries do not swamp the sums.
            _ns = nmr * chunks * HKV
            _lf = partial_lse[:_ns].float()
            _of = partial_o[: _ns * D].float()
            _lm = _t.isfinite(_lf)
            print(f"[slab] lse_sum={float(_lf[_lm].sum()):.6f} lse_absmax={float(_lf[_lm].abs().max()):.6f} "
                  f"o_sum={float(_of.sum()):.6f} o_absmax={float(_of.abs().max()):.6f} "
                  f"n_finite_lse={int(_lm.sum())}")
            # DEGENERATE-STATE SCAN over the WHOLE slab, not sampled tuples. Hypothesis: the
            # FIRST row of a q tile has the smallest causal bound, so it can have ZERO valid keys
            # in a chunk while later rows of the same tile still have some -- giving m = -inf,
            # d = 0, i.e. O = 0/0 and LSE = -inf. The merge then does exp2(other_m - m) with both
            # -inf = exp2(NaN). That would be a deterministic LOCATION with a value-dependent
            # effect, which is exactly what the bad-row indices show (always a tile boundary).
            n_slots = nmr * chunks * HKV
            lse_v = partial_lse[:n_slots].float()
            o_v = partial_o[: n_slots * D].float().view(n_slots, D)
            bad_lse = ~_t.isfinite(lse_v)
            bad_o = ~_t.isfinite(o_v).all(dim=1)
            print(f"[deg] slots={n_slots} nonfinite_LSE={int(bad_lse.sum())} "
                  f"nonfinite_O_rows={int(bad_o.sum())} "
                  f"min_lse={float(lse_v[_t.isfinite(lse_v)].min()) if bool(_t.isfinite(lse_v).any()) else float('nan'):.4f}")
            if int(bad_lse.sum()) or int(bad_o.sum()):
                idx = _t.nonzero(bad_lse | bad_o).flatten()[:8].tolist()
                for sl in idx:
                    mr, rem = sl // (chunks * HKV), sl % (chunks * HKV)
                    print(f"[deg]   slot={sl} merge_row={mr} (tile_row={mr % 128}) "
                          f"chunk={rem // HKV} head={rem % HKV} lse={float(lse_v[sl])}")
            for (r, c, h) in ((0, 0, 0), (0, 1, 0), (4, 0, 0), (0, 0, 1)):
                tok, hig = r // GQA, r % GQA
                head = h * GQA + hig
                lo, hi = 128 * c, min(128 * (c + 1), kl)
                bound = kl - ql + tok + 1          # exclusive, bottom-right causal
                hi = min(hi, bound)
                if lo >= hi:
                    print(f"[dump] r={r} c={c} h={h}: chunk entirely masked (lo={lo} hi={hi})")
                    continue
                qv = q[tok, head].float()
                sc = (kb[lo:hi, h] @ qv) * sm_scale
                w = _t.softmax(sc, dim=0)
                exp_o = (w[:, None] * vb[lo:hi, h]).sum(0)
                slot = int(merge_indptr[r]) + c
                got = partial_o[(slot * HKV + h) * D: (slot * HKV + h) * D + D].float()
                lse_got = float(partial_lse[(slot * HKV + h)])
                # Reference LSE in LOG2 units, which is what flashinfer's state_t carries.
                lse_ref = float(_t.logsumexp(sc, dim=0) / math.log(2.0))
                rel = float((got - exp_o).norm() / max(float(exp_o.norm()), 1e-9))
                print(f"[dump] r={r} tok={tok} hig={hig} c={c} h={h} head={head} slot={slot} "
                      f"keys[{lo},{hi}) O_rel={rel:.4g} lse_got={lse_got:.5f} "
                      f"lse_ref_log2={lse_ref:.5f}")

    # ORACLE=ba uses flashinfer's persistent `BatchAttention` as the correctness oracle instead
    # of the fp32 reference. Why this is sound AND necessary:
    #   * sound: BatchAttention has been gated against fp32_causal_reference on every fixture here
    #     and passes at 0 bad rows (rel 0.0018-0.0031), so it is a validated stand-in.
    #   * necessary: the fp32 reference is O(qo x kv x heads) in Python and becomes unaffordable
    #     past qo~2048 at kv=8192 -- it OOMs or runs past any sane cap. That blocked the ONE
    #     experiment needed to separate "items per CTA" from "request count", because reaching
    #     many items/CTA at request count 1 requires a long qo.
    # Keep ORACLE=fp32 (the default) for the small fixtures: that is what validates the oracle.
    if _os.environ.get("ORACLE") == "ba":
        from flashinfer.attention import BatchAttention as _BA
        _orc = _BA(kv_layout="NHD", device=DEV)
        _orc.plan(qo_indptr.to(DEV), kv_indptr.to(DEV), kv_indices[: int(kv_indptr[B])],
                  kv_len_t.to(DEV), HQ, HKV, D, D, PAGE, causal=True, sm_scale=sm_scale,
                  q_data_type=DT, kv_data_type=DT)
        _oo = _orc.run(q, (k_cache, v_cache))
        ref = (_oo[0] if isinstance(_oo, tuple) else _oo).float()
        print(f"[oracle] BatchAttention (validated stand-in for fp32): "
              f"absmax={float(ref.abs().max()):.6f}")
    else:
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
    # PERMUTATION MAP. absmax(ours) == absmax(stock) with full coverage says the right values are
    # present in the wrong rows; this finds, for each of our output rows, which REFERENCE row it
    # actually equals. A clean permutation shows its pattern (stride, token<->head-in-group swap,
    # tile rotation) and that names the fix. Matched on head 0's first 4 elements, which is enough
    # to identify a row uniquely among 128 random rows.
    if nfa3 > 0:
        import torch as _t
        key_ref = ref[:, 0, :4].float()
        key_got = got[:, 0, :4].float()
        d = (key_got[:, None, :] - key_ref[None, :, :]).abs().sum(-1)   # [ours, ref]
        best = d.argmin(dim=1)
        exact = (d.gather(1, best[:, None]).squeeze(1) < 1e-3)
        pairs = [(r, int(best[r])) for r in range(min(12, got.shape[0]))]
        print(f"PERMUTATION: {int(exact.sum())}/{got.shape[0]} of our rows exactly match SOME "
              f"reference row")
        print(f"  ours->ref (first 12): {pairs}")
        ident = int((best == _t.arange(best.shape[0], device=best.device)).sum())
        print(f"  identity matches: {ident}/{best.shape[0]}")
    # NON-FINITE GUARD, and this was a real hole. NaN comparisons are all False, so a NaN output
    # scored `rel = nan`, failed the `rel > tol` test, and came back as "bad rows 0/258, PASS".
    # Caught when SKIP_PACKED_Q=1 -- which removes the Q load and MUST produce garbage -- reported
    # PASS at 0.002069 while its output fingerprint was absmax=nan, sum=nan. Any arm that NaNs was
    # being reported as correct, which makes every verdict in this harness suspect until checked.
    import torch as _tt
    _nonfinite = int((~_tt.isfinite(got)).sum())
    if _nonfinite:
        print(f"\nmode={mode} NON-FINITE OUTPUT: {_nonfinite} of {got.numel()} elements are "
              f"nan/inf. This is a FAIL regardless of any per-row norm.")
        print("FAIL")
        return 1
    rows_bad = 0
    worst = 0.0
    _bad_rows = []
    for r in range(total_q):
        num = (got[r] - ref[r]).norm().item()
        den = max(ref[r].norm().item(), 1e-6)
        rel = num / den
        worst = max(worst, rel)
        if rel > 2e-2:
            rows_bad += 1
            # WHICH rows fail, not just how many. If the bad row index is always the same across
            # occurrences, the LOCATION is deterministic while the OCCURRENCE is not -- which
            # points at a specific work item (most chunks, or the fully-masked-chunk boundary)
            # rather than a general memory race. Cheaper and more informative than more sampling.
            _bad_rows.append((r, round(rel, 4)))
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
    if _bad_rows:
        print(f"BADROWS {_bad_rows[:12]}" + (" ..." if len(_bad_rows) > 12 else ""))
        # PLANMAP correlation: map each bad TOKEN row to the task-3 work item that owns it and print
        # that item's plan entries beside its CLEAN neighbours in the same block. Per-item plan data
        # is the only input that differs between items of one block, so if a corrupted item's entries
        # match its neighbours' pattern the plan is exonerated; if they differ, that difference IS
        # the bug. Host-side only -- the device route to this was ruled out (952 prints took the
        # failing arm to 0/384).
        if _PLANMAP is not None:
            _G = HQ // HKV
            _qs, _bi, _kl = _PLANMAP["q_start"], _PLANMAP["batch_idx"], _PLANMAP["kv_len"]
            _own, _wi = _PLANMAP["owner"], _PLANMAP["work_indptr"]
            _qoi = qo_indptr.tolist()
            def _items_for(row):
                hit = []
                for _i in range(len(_qs)):
                    _b = _bi[_i] if _i < len(_bi) else -1
                    if not (0 <= _b < len(_qoi) - 1):
                        continue
                    # item covers packed rows [q_start, q_start+CTA_Q) inside its request
                    _t0 = _qoi[_b] + _qs[_i] // _G
                    _t1 = _qoi[_b] + (_qs[_i] + 128) // _G
                    if _t0 <= row < _t1:
                        hit.append(_i)
                return hit
            _badset = sorted({r for r, _ in _bad_rows})
            _culprits = {}
            for _r in _badset[:16]:
                for _i in _items_for(_r):
                    _culprits.setdefault(_i, []).append(_r)
            print(f"[planmap] bad rows map to {len(_culprits)} task-3 items: "
                  f"{sorted(_culprits)[:10]}")
            for _i in sorted(_culprits)[:4]:
                _b = _own.get(_i)
                _sib = [j for j in range(_wi[_b], _wi[_b + 1])] if _b is not None else []
                print(f"[planmap] CULPRIT item={_i} blk={_b} rows={_culprits[_i][:6]} "
                      f"q_start={_qs[_i]} batch={_bi[_i]} kv_len={_kl[_i]}")
                for _j in _sib[:6]:
                    tag = "SAME" if _j == _i else "nbr "
                    print(f"[planmap]   {tag} item={_j} q_start={_qs[_j]} batch={_bi[_j]} "
                          f"kv_len={_kl[_j]} bad={'Y' if _j in _culprits else 'n'}")
    # Timing runs ONLY after the verdict is printed, and only when the arm passed -- a timing
    # number for a failing kernel is worse than none.
    if _os.environ.get("TIME_AB") == "1":
        if rows_bad == 0:
            _time_ab()
        elif _os.environ.get("TIME_FORCE") == "1":
            # Only for DIAGNOSTIC arms whose output is garbage by construction (e.g.
            # SKIP_PACKED_Q=1, which removes the Q gather to bound its cost). The numbers are a
            # timing delta against the correct arm, never a result in their own right.
            print(f"[time] *** DIAGNOSTIC ARM: {rows_bad} bad rows, output is NOT correct. "
                  f"Timing is only meaningful as a delta. ***")
            _time_ab()
        else:
            print(f"[time] SKIPPED: this arm has {rows_bad} bad rows. "
                  f"No timing is reported for an arm that does not pass.")
    print("PASS" if rows_bad == 0 else "FAIL")
    return 0 if rows_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
