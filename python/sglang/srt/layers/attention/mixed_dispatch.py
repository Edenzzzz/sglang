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

# A row joins the FA3 class only if it fills enough of a 128-wide tile. The original crossover
# (0.78x at packed_qo 32, 1.13x at 56) is CONTRADICTED by a four-axis sweep on the 0.7 port
# (2026-10-10, tasks/debug_kernel_server_perf_4_mode.md, fixture `band1`), every arm gated against
# the causal fp32 reference at 0 bad rows:
#
#   packed_qo 4..128  x  kv 128..65537  x  rows 1..128   ->  FA3 faster at EVERY point, 1.16x-12.1x
#
# including the 18-row / kv=65537 decode shape where the doc previously recorded "FA2 beats FA3 by
# 1.30x" -- that entry is RETRACTED (both its arms failed numerics; FA3 is 3.66x faster there).
#
# So there is no measured shape at which the FA2 role is the right executor, and MIN_PACKED_QO has
# no support at 48 or any other value. Env-overridable rather than silently re-pinned, because the
# implication is a DESIGN question, not a constant: see the note in split_classes.
MIN_PACKED_QO = int(os.environ.get("SGLANG_FUSED_MIN_PACKED_QO", "48"))
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

# Batch-level ratio guard; see _apply_decode_population_guard. Measured kv <= 64*nd becomes
# kv_tiles <= nd/2 at TILE_KV=128, so one decode row buys half a KV tile.
DECODE_POPULATION_GUARD = os.environ.get("SGLANG_FUSED_DECODE_GUARD", "1") == "1"
DECODE_TILES_PER_ROW = float(os.environ.get("SGLANG_FUSED_DECODE_TILES_PER_ROW", "2"))
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
    # A3 (2026-10-10): the four-axis sweep found NO shape where the FA2 role wins, so every
    # threshold here routes work to the slower executor. Lowering MIN_PACKED_QO is the measured
    # improvement -- but note what it converges to: if every row goes to FA3 the fused kernel has
    # no co-resident FA2 role, i.e. it degenerates to FA3-only (`SM_SPLIT=all`) and the two-role
    # premise collapses. That is a design question, not a constant, and it is recorded in the task
    # doc rather than resolved by quietly editing a number here.
    fa3 = ((packed_qo >= MIN_PACKED_QO) & (kv_tiles >= MIN_KV_TILES)
           & (qo >= MIN_QO_LEN))
    fa3 = _apply_decode_population_guard(fa3, kv_tiles)
    return fa3, q_tiles, kv_tiles


def _apply_decode_population_guard(fa3, kv_tiles):
    """Un-route FA3 rows the co-resident decode population cannot cover.

    The measured boundary (tasks/fused_specdec_workload/boundary_28x4x128.json, 25 points) is a
    RATIO, not a threshold on either axis: the fused win holds while kv <= 64 * decode_rows, i.e.
    kv_tiles <= decode_rows / 2 at TILE_KV=128 -- about one 64-wide KV tile of decode work per
    decode row to fill the SMs FA3 leaves idle. Flip points: kv 512 at nd 8, 2048 at 32, 8192 at
    115, 16384 at 365 (kv/nd = 64, 64, 71, 45).

    With NO decode rows the fused kernel loses on EVERY measured kv (0.617-0.704x), and the per-row
    conditions above cannot see that: they read only one row's own shape. So the guard is
    batch-level, and rows it un-routes go to the persistent class -- the same kernel the sequential
    arm would have run for them.

    Replaying all 25 measured points through this code: agreement with the measured winner rises
    from 11/25 (per-row rule alone, which routes FA3 everywhere including the nd=0 losing region)
    to 23/25; the two misses are kv 8192 @ nd 115 and kv 32768 @ nd 365, measured 1.026x and
    1.016x, i.e. the flat region where the choice is worth <3%.
    """
    if not DECODE_POPULATION_GUARD or not fa3.any():
        return fa3
    decode_rows = int((~fa3).sum())
    if decode_rows == 0:
        # Nothing to overlap with: measured 0.617-0.704x across every kv.
        return np.zeros_like(fa3)
    budget = decode_rows / DECODE_TILES_PER_ROW
    return fa3 & (kv_tiles <= budget)


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


def dense_block_table(kv_indptr, kv_indices, rows, page_size=1, out=None):
    """Build upstream FA3's block table from sglang's flat paged KV indices.

    Upstream documents this tensor as "the block table of each prompt", shape
    ``[batch_size, max_num_blocks_per_seq]`` (flashinfer/prefill.py:1646, constructed at :1861).
    sglang instead stores KV indices flat, with per-request extents given by ``kv_indptr``. The
    two representations are NOT interchangeable, and aliasing one as the other is a real fault,
    not a type nuisance: passing the flat (vector-sparse) array where the dense table is expected
    produced ``CUDA error: an illegal memory access`` on the fused path. So the seam owns an
    explicit conversion, and this is it.

    Fill value is -1, never 0: page id 0 is valid, so zero padding would make every short
    sequence read page 0 for its tail blocks instead of being masked off.

    Args:
        kv_indptr: length ``batch+1`` prefix sums into ``kv_indices``.
        kv_indices: flat page (or token) ids for all requests.
        rows: row indices to include, in output order. Rows absent here are not emitted, so the
            FA3 subset gets a table whose batch dimension matches the subset's request count --
            which upstream's combine scheduler requires (it derives its tile mapping from the
            pair (token extent, batch count)).
        page_size: tokens per page; >1 divides token offsets down to page granularity.
        out: optional preallocated ``[len(rows), max_blocks]`` int32 array to fill.

    Returns:
        ``(table, max_blocks)`` where ``table`` is ``[len(rows), max_blocks]`` int32, -1 padded.
    """
    indptr = _as_i64(kv_indptr)
    indices = _as_i64(kv_indices)
    rows = list(rows)
    if not rows:
        return np.zeros((0, 0), dtype=np.int32), 0
    lens = [int(indptr[r + 1] - indptr[r]) for r in rows]
    if page_size > 1:
        lens = [(n + page_size - 1) // page_size for n in lens]
    max_blocks = max(lens) if lens else 0
    if out is None:
        table = np.full((len(rows), max_blocks), -1, dtype=np.int32)
    else:
        table = out
        table[:] = -1
    for i, r in enumerate(rows):
        lo = int(indptr[r])
        n = lens[i]
        if n == 0:
            continue
        if page_size > 1:
            # one entry per page: take the first token id of each page and divide down
            sel = indices[lo : lo + n * page_size : page_size] // page_size
        else:
            sel = indices[lo : lo + n]
        table[i, :n] = sel.astype(np.int32)
    return table, max_blocks


def fa3_subset_varlen(qo_lens, kv_lens, rows):
    """Per-request varlen arrays for upstream FA3 over a row SUBSET.

    Upstream indexes ``cu_seqlens_q`` and ``seqused_k`` by ``bidb`` -- the request index within the
    batch it was given -- so both must be PER-REQUEST and must cover exactly the subset, densely
    renumbered. sglang's persistent plan instead carries per-WORK-ITEM ``q_indptr``/``kv_len``;
    handing those over is the error family that produced four separate port bugs (see
    upstream-fa3-subset-params), because the arrays look right and index wrong.

    ``total_q`` is the varlen Q extent of the subset (``cu_seqlens_q[-1]``), NOT a page count: the
    upstream header warns that conflating it with ``num_pages`` mis-shapes the Q tensor and reads
    out of range (fa3_upstream.cuh:147). It is returned explicitly so the caller never has to
    guess which quantity it is.

    Args:
        qo_lens: per-row query lengths for the whole batch.
        kv_lens: per-row KV lengths for the whole batch.
        rows: the FA3 row indices, in the order they are handed to upstream.

    Returns:
        ``(cu_seqlens_q, seqused_k, total_q)`` -- ``cu_seqlens_q`` is ``len(rows)+1`` int32 prefix
        sums starting at 0; ``seqused_k`` is ``len(rows)`` int32; ``total_q`` is a python int.
    """
    qo = _as_i64(qo_lens)
    kv = _as_i64(kv_lens)
    rows = list(rows)
    cu = np.zeros(len(rows) + 1, dtype=np.int32)
    if rows:
        sel = qo[rows]
        np.cumsum(sel, out=cu[1:], dtype=np.int32)
    seqused = (kv[rows] if rows else np.zeros(0, dtype=np.int64)).astype(np.int32)
    return cu, seqused, int(cu[-1])

# ---------------------------------------------------------------------------
# B1d: time-independent tile-waste accounting.
#
# Grid-level occupancy (`grid < SM count`) cannot see the waste item A targets: a 48-row work
# item inside a 128-wide WGMMA tile keeps every SM busy while leaving 63% of each tile idle. That
# waste is a property of the SHAPES, not of the schedule, so it is computable on the host from
# the same arrays the dispatcher already has -- no profiler, no timing.
#
# For one row the engine covers ceil(packed_qo/TILE_Q)*TILE_Q query rows by
# ceil(kv/TILE_KV)*TILE_KV keys, while only packed_qo x kv is useful work. Accumulating both
# gives a FLOP-weighted waste fraction, which is the figure that says whether finer tiles are
# worth instantiating.
TILE_STATS = os.environ.get("SGLANG_FUSED_TILE_STATS", "0") == "1"
_PQ_BUCKETS = (16, 32, 48, 64, 128, 256, 512)
_stats = {
    "plans": 0,
    "rows": 0,
    "useful": 0,      # sum packed_qo * kv
    "covered": 0,     # sum q_tiles*TILE_Q * kv_tiles*TILE_KV
    "by_bucket": {},  # bucket -> [rows, useful, covered]
}


def _bucket(pq):
    for b in _PQ_BUCKETS:
        if pq <= b:
            return b
    return -1  # 512+


def tile_waste_accumulate(qo_lens, kv_lens, gqa_ratio, mask=None):
    """Accumulate FLOP-weighted tile coverage vs useful work, bucketed by packed_qo."""
    if not TILE_STATS:
        return
    qo, kv = _as_i64(qo_lens), _as_i64(kv_lens)
    pq = qo * gqa_ratio
    kvt = (np.maximum(kv, 1) + (TILE_KV - 1)) >> KV_SHIFT
    useful = pq * np.maximum(kv, 1)
    # Each row is charged its ACTUAL tile width, not TILE_Q uniformly. FA3 has one fixed
    # 128-wide WGMMA tile; the persistent/FA2 side already has three classes -- tile128 for
    # packed_qo > 64, tile64, and tile16 -- so charging a packed-16 verify row 128 rows of
    # coverage overstates its waste by 8x and would invent headroom that does not exist.
    if mask is None:
        width = np.where(pq > 64, TILE_Q, np.where(pq > 16, 64, 16))
    else:
        m = np.asarray(mask, dtype=bool)
        fa2_width = np.where(pq > 64, TILE_Q, np.where(pq > 16, 64, 16))
        width = np.where(m, TILE_Q, fa2_width)   # FA3 rows always pay the full 128
    qrows = ((pq + width - 1) // width) * width
    covered = qrows * (kvt * TILE_KV)
    _stats["plans"] += 1
    _stats["rows"] += int(len(qo))
    _stats["useful"] += int(useful.sum())
    _stats["covered"] += int(covered.sum())
    for i in range(len(qo)):
        b = _bucket(int(pq[i]))
        e = _stats["by_bucket"].setdefault(b, [0, 0, 0])
        e[0] += 1
        e[1] += int(useful[i])
        e[2] += int(covered[i])


def tile_waste_report():
    """One-line-per-bucket summary; returns '' when nothing was accumulated."""
    if not TILE_STATS or not _stats["rows"]:
        return ""
    out = [
        "TILE_WASTE plans=%d rows=%d useful=%.3e covered=%.3e waste=%.1f%%"
        % (
            _stats["plans"],
            _stats["rows"],
            float(_stats["useful"]),
            float(_stats["covered"]),
            100.0 * (1.0 - _stats["useful"] / max(_stats["covered"], 1)),
        )
    ]
    for b in sorted(_stats["by_bucket"], key=lambda x: (x < 0, x)):
        rows, u, c = _stats["by_bucket"][b]
        label = "512+" if b < 0 else "<=%d" % b
        out.append(
            "  packed_qo %-6s rows=%-7d useful_share=%5.1f%% waste_in_bucket=%5.1f%%"
            % (
                label,
                rows,
                100.0 * u / max(_stats["useful"], 1),
                100.0 * (1.0 - u / max(c, 1)),
            )
        )
    return "\n".join(out)
