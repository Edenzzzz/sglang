"""Mixed-batch attention dispatch: which tile class runs on which SMs.

Two kernels, disjoint SM sets by design -- never sharing an SM:

  FA3 class       large q tiles with enough KV behind them. Gets its own SMs and
                  runs the wgmma mainloop with GQA packing and split-KV, so its
                  parallel width is not capped by h_qo*ceil(qo_len/128).
  persistent      everything else (small prefill, verify below the tile
                  threshold, draft, decode) in the FA2-lineage persistent kernel
                  with tiles {128,64,16}, packed GQA, split-KV and LPT.

Both quantities the routing uses are properties of the tile grid, as measured:

  packed_qo_len = qo_len * gqa_ratio -> how much of a 128-wide tile a row fills.
    Unpacked FA3 wastes 4/128 of a tile on a verify row whatever the GQA ratio;
    packing it is worth 5.0x on the verify class (2541 -> 511 us on 58 rows at
    kv 11000, gqa 7). After packing, FA3 beats the persistent kernel at
    packed_qo 56 (1.13x) and still loses at 32 (0.78x), so the threshold is a
    tile-occupancy threshold, not a KV-length one.
  kv_tiles = ceil(kv_len / 128) -> how much work sits behind each q tile.

Work is counted in (q tile x kv tile) pairs, which is what both mainloops
actually iterate, and the SM split is the makespan balance point. If the FA3
class is so small that even ONE SM would leave it idle long before the rest of
the device finishes, it does not get its own SMs at all and folds into the
persistent kernel -- a separate kernel is only worth a separate SM set if it can
keep it busy.

Constants fitted on H800 bf16 head_dim 128; refit with
tasks/fused_specdec_workload/{boundary_sweep,fa3_split_pack}.py elsewhere.
"""
import os

import numpy as np

TILE_Q = 128                 # both mainloops' large q tile
TILE_KV = 128                # CTA_TILE_KV in the persistent mainloop
NUM_SM = 132
Q_SHIFT = 7                  # log2(TILE_Q)
KV_SHIFT = 7                 # log2(TILE_KV)

# Throughput in (q tile x kv tile) pairs per microsecond per SM.
# FA3 from split-KV measurement: 16,384 pairs over 128 CTAs in 290.4 us.
# persistent from the same row: 16,384 pairs over 132 SMs in 400.7 us.
R_FA3 = 0.441
R_PERS = 0.310
# Decode-shaped rows are bandwidth-bound, not tile-bound.
BYTES_PER_KV_TOKEN = 2 * 2 * 128      # K and V, bf16, head_dim 128, per token
BW_PER_SM = 22.0                      # B/ns/SM before saturation
BW_PEAK = 2.87e3                      # B/ns at saturation

# A row joins the FA3 class only if it fills enough of a 128-wide tile. Measured
# crossover after GQA packing: 0.78x at packed_qo 32, 1.13x at 56.
MIN_PACKED_QO = 48
# ... and only if there is enough KV behind it to amortise the mainloop.
#
# MEASURED 2026-10-05 and this was wrong, expensively. At the serving head shape
# (28/4/128) on an exclusively-held H200, one row, both mainloops planned from the
# same rows (tasks/fused_specdec_workload/row_cmp.py):
#
#   qo x kv        persistent        FA3     FA3 wins by
#   500 x  3095    194.5 us      74.0 us        2.63x
#   500 x  8192    304.5 us     148.1 us        2.06x
#   500 x 32768    939.2 us     499.8 us        1.88x
#   128 x  3095    236.0 us      73.2 us        3.22x
#  2048 x  3095    362.9 us     215.0 us        1.69x
#
# FA3 wins at EVERY kv length once the row has real q work, and wins by MORE at
# short kv, not less -- the opposite of what a `kv_tiles >= 32` floor assumes.
# The modal serving mixed batch is one 500-token chunk over a 2595-token prefix
# (kv 3095 = 25 tiles) plus 11 decode rows, and that chunk is 93% of the step's
# attention. The old floor excluded exactly it, so every fused step ran the whole
# batch on the slower mainloop and the fused path measured parity (-0.21% on a
# mixed step, -0.65% end to end) while the kernel underneath was 2.6x faster.
#
# The floor is now 1 (no KV condition) and admission is carried by q work alone.
# Decode-shaped rows are still excluded, by MIN_QO_LEN below rather than by KV.
MIN_KV_TILES = int(os.environ.get("SGLANG_FUSED_MIN_KV_TILES", "1"))
# FA3 is planned UNPACKED (one work item per qo head, tiles on qo_len -- see
# scheduler.cuh:939), so a packed-unit threshold alone admits rows as short as
# qo_len 7 at gqa 7 into a 128-row tile. This floor is in the units FA3 is
# actually planned in. Measured: qo_len 1/2/4 lose to the persistent kernel,
# qo_len >= 16 wins everywhere tested.
MIN_QO_LEN = int(os.environ.get("SGLANG_FUSED_MIN_QO_LEN", "16"))
# Fold the FA3 class in unless one SM's worth of it is at least this fraction of
# the makespan the rest of the device needs. Below that, a separate SM set would
# sit idle -- the class is too small to own hardware.
MIN_SM_UTIL = 0.5
# Measured band for the FA3 role's SM count: a work-proportional split
# hands it most of the device and loses; capping it wins.
# Measured on the serving path, not derived. A matched sweep at
# --random-input-len 8192 (200 prompts, H200, qwen7b) gives a sharply peaked
# curve in sm_fa3: 16 -> -22.4%, 32 -> -11.7%, 64 -> -3.6%, 96 -> -8.4%,
# 112 -> -22.0% against the fused-off baseline. The optimum is 64, and the old
# ceiling of 48 sat below it and cost 4.5 points of throughput.
#
# The ceiling is doing the actual choosing here: unclamped, plan_dispatch's
# analytic balance point returns 115-131 for real serving batch shapes, i.e.
# nearly the whole device on FA3, which the same sweep measures as one of the
# two worst settings. That cost model was fit for the overlap kernel and does
# not transfer to this kernel's makespan, so this bound must stay measured.
# REVERTED to 64 after measuring. I raised the ceiling to 131 on the strength of a
# long-prefix fixture that is ~99% FA3 work, where 64 clusters cost 1.74x against
# 124. On the actual serving workload it made things ~2x WORSE: 100.41 tok/s and
# 113.35 ms ITL against 198.16 / 54.86 at the pinned 8, with the same 586 tok/s
# baseline. The reason is that the fixture is unrepresentative -- at 8192 input with
# chunked-prefill 512 the FA3-class rows are 512-token chunks, so FA3 does not hold
# most of the work and the decode side dominates ITL. Giving FA3 most of the grid
# starves it. The clamp was protecting serving, and the right ceiling is an
# empirical question per workload shape, not a work-share identity.
MIN_SM_FA3, MAX_SM_FA3 = 8, 64


def _as_i64(a):
    return a if (isinstance(a, np.ndarray) and a.dtype == np.int64) else \
        np.asarray(a, dtype=np.int64)


def split_classes(qo_lens, kv_lens, gqa_ratio):
    """Route rows to the FA3 class or the persistent class. Vectorised.

    Returns (fa3_mask, q_tiles, kv_tiles) so callers can slice rows without
    recomputing anything. Tiles are computed with shifts rather than ceil-div:
    this runs on every batch, ahead of the launch.
    """
    qo, kv = _as_i64(qo_lens), _as_i64(kv_lens)
    packed_qo = qo * gqa_ratio
    q_tiles = (packed_qo + (TILE_Q - 1)) >> Q_SHIFT
    kv_tiles = (np.maximum(kv, 1) + (TILE_KV - 1)) >> KV_SHIFT
    fa3 = ((packed_qo >= MIN_PACKED_QO) & (kv_tiles >= MIN_KV_TILES)
           & (qo >= MIN_QO_LEN))
    return fa3, q_tiles, kv_tiles


def plan_dispatch(qo_lens, kv_lens, gqa_ratio, num_kv_heads, num_sm=NUM_SM):
    """Decide the SM split and the FA3 KV-split factor. Pure CPU, closed form.

    Returns a dict:
      mode      'fused' (persistent only) or 'split' (disjoint SM sets)
      sm_fa3    SMs given to the FA3 kernel, 0 when fused
      sm_pers   SMs given to the persistent kernel
      kv_split  S for FA3, chosen so its width fills the SMs it was given
      fa3_mask  boolean row mask for the FA3 class, or None when fused

    Kept to a handful of array ops, with a max() short-circuit first: most
    batches have no row that could fill a 128-wide tile, and those must not pay
    for the rest of the arithmetic.
    """
    qo, kv = _as_i64(qo_lens), _as_i64(kv_lens)
    fused = dict(mode="fused", sm_fa3=0, sm_pers=num_sm, kv_split=1, fa3_mask=None)
    if qo.size == 0:
        return fused
    # Short-circuit: most batches contain no row that could fill a 128-wide
    # tile, and they must not pay for the rest of the arithmetic.
    min_qo = max(-(-MIN_PACKED_QO // gqa_ratio), MIN_QO_LEN)
    if int(qo.max()) < min_qo:
        return fused

    # Whole-batch aggregates in three array ops. Every row with
    # packed_qo_len <= TILE_Q occupies exactly one q tile, and those rows are
    # precisely the non-candidates (min_qo <= TILE_Q/gqa_ratio), so the batch
    # total is the KV-tile sum plus a correction over the candidates alone --
    # no q-tile array is ever built.
    kv_t = (np.maximum(kv, 1) + (TILE_KV - 1)) >> KV_SHIFT
    total_kv_t = int(kv_t.sum())
    cand = np.flatnonzero(qo >= min_qo)

    # The FA3 class is small by construction (captured p50: 1-2 rows a batch), so
    # accumulate it over the candidates as scalars rather than masking the batch.
    w_fa3 = n_tiles_fa3 = kv_t_fa3 = q_tile_excess = 0
    # Materialise only the candidate rows as Python ints -- there are a handful,
    # and converting the whole batch costs more than the arithmetic saves.
    for kt, q in zip(kv_t[cand].tolist(), qo[cand].tolist()):
        qt = -(-(q * gqa_ratio) // TILE_Q)
        q_tile_excess += (qt - 1) * kt
        if kt >= MIN_KV_TILES:
            w_fa3 += qt * kt; n_tiles_fa3 += qt; kv_t_fa3 += kt
    keep = cand[(kv_t[cand] >= MIN_KV_TILES)]
    total_pairs = float(total_kv_t + q_tile_excess) * num_kv_heads
    total_kv = float(total_kv_t) * num_kv_heads
    if w_fa3 <= 0:
        return fused
    w_fa3 *= num_kv_heads
    n_tiles_fa3 *= num_kv_heads
    w_pers = total_pairs - w_fa3

    def mask():
        m = np.zeros(qo.size, dtype=bool); m[keep] = True; return m


    if w_pers <= 0.0:
        return dict(mode="split", sm_fa3=num_sm, sm_pers=0, fa3_mask=mask(),
                    kv_split=_kv_split(n_tiles_fa3, num_sm))

    # Makespan balance between the two disjoint SM sets.
    a = w_fa3 / R_FA3
    b = w_pers / R_PERS
    # The persistent side is bandwidth-bound on decode-shaped rows; this floor
    # stops it being handed fewer SMs than it can actually use.
    bytes_pers = (total_kv - kv_t_fa3 * num_kv_heads) * TILE_KV * BYTES_PER_KV_TOKEN
    b = max(b, bytes_pers / BW_PER_SM / 1e3)

    # Too small to own hardware: one SM's worth of FA3 against the makespan the
    # rest of the device needs anyway. Below this it folds into the persistent
    # kernel, because a separate SM set it cannot keep busy is wasted hardware.
    if a < MIN_SM_UTIL * (b / (num_sm - 1)):
        return fused

    sm_fa3 = int(min(max(round(num_sm * a / (a + b)), 1), num_sm - 1))
    return dict(mode="split", sm_fa3=sm_fa3, sm_pers=num_sm - sm_fa3,
                fa3_mask=mask(), kv_split=_kv_split(n_tiles_fa3, sm_fa3))


def _kv_split(n_q_tiles, sm_fa3):
    """Smallest S whose width (q tiles x S) covers the SMs FA3 was given.

    Without this, FA3's width is exactly its q-tile count -- 64 for one
    1024-token chunk at 8 q heads -- and it idles half the device. Measured: S=2
    took one such row from 507 us to 290 us, 1.38x past the persistent kernel.
    """
    n = int(n_q_tiles)
    if n <= 0:
        return 1
    return max(1, min(8, -(-int(sm_fa3) // n)))


def plan_dispatch_prefix(qo_lens, kv_lens, gqa_ratio, num_kv_heads, num_sm=NUM_SM):
    """Like plan_dispatch, but constrained to what the fused kernel can express.

    The fused kernel runs FA3 on ``blockIdx.x < sm_split`` over one contiguous
    range of rows and the persistent mainloop over the rest, with the two sides
    planned independently. That makes the row layout part of the contract: FA3's
    rows must be a PREFIX of the batch, because its q/o pointers start at 0 while
    the persistent side's are advanced by ``pers_row_offset``.

    sglang's mixed batches cooperate with that -- ``ScheduleBatch.mix_with_running``
    appends decode rows after prefill rows, and only prefill rows can clear the
    FA3 gate -- but not unconditionally: a short chunk in the middle of the
    prefill run fails the gate and breaks contiguity. Rather than reorder the
    batch (which would mean permuting q, o and the KV indices), take the longest
    qualifying prefix and leave the rest to the persistent side.

    Returns (mode, sm_fa3, sm_pers, kv_split, n_fa3_rows, n_fa3_tokens,
             n_fa3_pages) where n_fa3_rows is a row COUNT, so rows
             [0, n_fa3_rows) go to FA3 and the rest to the persistent kernel.
    """
    qo, kv = _as_i64(qo_lens), _as_i64(kv_lens)
    fa3_mask, q_tiles, kv_tiles = split_classes(qo, kv, gqa_ratio)

    # Longest prefix that is entirely FA3-class.
    n_fa3_rows = int(np.argmin(fa3_mask)) if not fa3_mask.all() else int(fa3_mask.size)
    if n_fa3_rows == 0 or n_fa3_rows == fa3_mask.size:
        # Nothing for one of the two roles: the fused kernel has no advantage
        # over simply running that role's kernel alone.
        return "fused", 0, num_sm, 1, 0, 0, 0

    base = plan_dispatch(qo[:n_fa3_rows], kv[:n_fa3_rows], gqa_ratio, num_kv_heads, num_sm)
    # SGLANG_FUSED_SM_FA3 pins the split so it can be swept on a real serving
    # workload; the analytic balance point is the default. Clamped to
    # [1, num_sm - 1] rather than to [MIN_SM_FA3, MAX_SM_FA3], because the point
    # of a sweep is to reach values the heuristic would not choose.
    _pin = os.environ.get("SGLANG_FUSED_SM_FA3")
    if _pin:
        sm_fa3 = int(np.clip(int(_pin), 1, max(1, num_sm - 1)))
    else:
        sm_fa3 = int(np.clip(base["sm_fa3"] or MIN_SM_FA3, MIN_SM_FA3, MAX_SM_FA3))
    n_fa3_tokens = int(qo[:n_fa3_rows].sum())
    n_fa3_pages = int(np.maximum(kv[:n_fa3_rows], 1).sum())
    return ("split", sm_fa3, max(1, num_sm - sm_fa3), base["kv_split"],
            n_fa3_rows, n_fa3_tokens, n_fa3_pages)
