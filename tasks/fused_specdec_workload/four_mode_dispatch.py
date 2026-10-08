#!/usr/bin/env python3
"""4-mode dispatch: separate kernels per tile class, grid-partitioned by a scheduler.

The design this implements (and the reason the single fused kernel lost):

  large_prefill  -> FA3            wgmma mainloop, consecutive large tiles
  spec_verify    -> FA3            packed_qo = (k+1)*gqa_ratio, still tile-64
  mixed_decode   -> persistent LPT split-KV + LPT tile fetching
  spec_draft     -> persistent LPT folds into the tile-16 queue (measured: fused
                                  beats a separate low-SM launch by 4-7%)

Both kernels are launched concurrently with GRID SIZES CHOSEN BY THE SCHEDULER --
not green contexts. Each planner takes a cap (FLASHINFER_FA3_MAX_CTAS,
FLASHINFER_PERSISTENT_MAX_CLUSTERS) and rebalances its own work over the CTAs it
owns, so a split is a plan-time integer and CUDA-graph capture is unaffected:
one graph per shape bucket, not one per split ratio.

Scheduler cost model, fitted on this device (see calibrate()):

  FA3 side is compute-bound and scales ~linearly in CTAs:
      t_fa3(c)  = W_mac / (c * R_FA3)
  the persistent side is bandwidth-bound on decode-shaped rows and saturates
  well below the full device:
      t_pers(c) = max(W_mac / (c * R_PERS), Bytes / BW(c))
      BW(c)     = BW_PEAK * min(1, c / C_SAT)
  pick the integer split minimizing max(t_fa3, t_pers) -- balance, not
  proportional share, which is the whole point of having a scheduler.
"""
import heapq, json, os
import numpy as np
import torch

# Fitted on H800 bf16 head_dim 128 by calibrate_split.py (see calib_split.json).
# MACs/ns/CTA for each mainloop; the gap is the wgmma advantage plus tile width.
# FA3 is modelled by simulating the planner's own MinHeap over its own
# cost_function (2*qo_len + kv_len) and taking the MAKESPAN, then scaling by one
# constant. A smooth W/(c*R) cannot express what the hardware does: 32 and 48
# CTAs both measure 502us for a 64-unit problem because both need ceil(64/c)=2
# rounds. The staircase is the whole reason the split has to be chosen carefully.
FA3_NS_PER_COST = 16.42   # ns per unit of cost_function, fitted on H800 bf16
R_PERS = 6.2e2               # MAC/ns/CTA, fitted on compute-bound cells (kv>=16K)
# The persistent planner uses num_sm *= 2 for head_dim < 256, so its natural
# grid is 2 clusters per SM while FA3 runs 1 CTA per SM. The scheduler therefore
# counts in SMs -- the actually shared resource -- and converts on the way out.
PERS_CLUSTERS_PER_SM = 2
BW_PER_SM = 66.0          # B/ns/SM before saturation
BW_PEAK = 2.87e3          # B/ns at saturation (~87 clusters), 86% of HBM peak
C_SAT_SM = 44             # SMs at which the persistent kernel saturates HBM
BYTES_PER_KV = 2 * 2      # K and V, bf16
FA3_TILE_Q = 128          # PrefillSM90Plan: cta_tile_q = 128 for head_dim_vo 128
PERS_TILE_Q = 128
NUM_SM = 132

# Routing boundary. A row belongs on FA3 only if it satisfies BOTH:
#   (a) qo_len >= FA3_MIN_QO -- it actually fills a wgmma-width tile. FA3 tiles
#       over qo_len ALONE at cta_tile_q 128 and does not fuse group heads
#       ("the group_size in cost_function is always 1"), so packed_qo_len is the
#       wrong measure here: a 4-token verify row occupies 4/128 of a tile
#       whatever the GQA ratio. Testing KV length alone is what made an earlier
#       version route 58 verify rows to FA3 and lose 14x.
#   (b) kv_len >= FA3_MIN_KV_TILES * KV_TILE -- enough consecutive KV tiles that
#       the wgmma mainloop amortises FA3's planner and epilogue.
FA3_MIN_QO = 128          # one full FA3 tile
FA3_MIN_KV_TILES = 32
KV_TILE = 128


def classify(batch, gqa_ratio, k_spec=3):
    """Split a captured fused batch into the four tile classes."""
    if all(c in batch for c in ("large_prefill", "mixed_decode")):
        return {c: [tuple(x) for x in batch.get(c, [])] for c in
                ("large_prefill", "spec_verify", "mixed_decode", "spec_draft")}
    out = {"large_prefill": [], "spec_verify": [], "mixed_decode": [], "spec_draft": []}
    for q, kv in batch["rows"]:
        if q > 1 and q * gqa_ratio > 64:
            out["large_prefill"].append((q, kv))
        elif q == k_spec + 1:
            out["spec_verify"].append((q, kv))
        elif q == 1:
            out["mixed_decode"].append((q, kv))
        else:
            out["spec_draft"].append((q, kv))
    return out


def route(cls, gqa_ratio):
    """Decide which rows go to FA3 and which to the persistent kernel.

    The boundary is per-row, not per-class: a 'large prefill' row attending a
    200-token prefix has nothing for wgmma to chew on and belongs in the LPT
    queue with the decode rows. Note the test is on qo_len, not packed_qo_len --
    FA3 tiles over qo_len alone at cta_tile_q 128 and does not fuse group heads
    ("the group_size in cost_function is always 1"), so a 4-token verify row
    occupies 4/128 of a tile whatever the GQA ratio. Testing KV length alone is
    what made an earlier version route 58 verify rows to FA3 and lose 14x.
    """
    fa3, pers = [], []
    for c in ("large_prefill", "spec_verify", "mixed_decode", "spec_draft"):
        for q, kv in cls[c]:
            if q >= FA3_MIN_QO and kv >= FA3_MIN_KV_TILES * KV_TILE:
                fa3.append((q, kv))
            else:
                pers.append((q, kv))
    return fa3, pers


def _work(rows, h_qo):
    return float(sum(q * max(kv, 1) for q, kv in rows)) * h_qo * 128.0


def _bytes(rows, h_kv):
    return float(sum(max(kv, 1) for q, kv in rows)) * h_kv * 128.0 * BYTES_PER_KV


def fa3_width(rows, h_qo, num_sm=NUM_SM):
    """How many CTAs FA3 can actually keep busy.

    From PrefillSM90Plan: one work unit per (qo_head, qo_tile) with
    cta_tile_q 128, and NO KV splitting -- the MinHeap balances whole qo tiles.
    So a single 1024-token chunk gives h_qo * 8 units: 224 at (28,4) heads
    (device-filling) but only 64 at (8,1) heads, which is the measured plateau.
    Below num_sm the rest of the device is idle and co-launching the decode
    work beside it is nearly free -- that is where concurrency wins.
    """
    tiles = sum(-(-q // FA3_TILE_Q) for q, _ in rows)
    per_head = tiles + len(rows) - 1
    heads = 1 if per_head > 4096 else h_qo      # same_schedule_for_all_heads
    return max(1, min(num_sm, tiles * heads))


def pers_width(rows, h_kv, num_sm=NUM_SM):
    """SMs the persistent kernel can keep busy. It fuses group heads and splits
    KV by kv_len_limit sized to the cluster count, so it fills whatever grid it
    is given as long as there is KV left to split."""
    clusters = num_sm * PERS_CLUSTERS_PER_SM
    units = sum(max(1, -(-q // PERS_TILE_Q)) for q, _ in rows) * h_kv
    total_kv = sum(max(kv, 1) for _, kv in rows)
    splits = max(1, min(clusters // max(units, 1), total_kv // KV_TILE))
    return max(1, min(num_sm, -(-units * splits // PERS_CLUSTERS_PER_SM)))


def _fa3_cost_units(rows, h_qo):
    """The planner's work units, in its own enumeration order (kv descending)."""
    tiles = sum(-(-q // FA3_TILE_Q) for q, _ in rows)
    heads = 1 if (tiles + len(rows) - 1) > 4096 else h_qo
    units = []
    for q, kv in sorted(rows, key=lambda r: -r[1]):
        for _ in range(-(-q // FA3_TILE_Q)):
            units.append(2.0 * FA3_TILE_Q + max(kv, 1))
    return units * heads


def t_fa3(rows, c, h_qo, h_kv):
    """Makespan of the planner's MinHeap assignment over c CTAs, in us."""
    if not rows:
        return 0.0
    if c <= 0:
        return float("inf")
    units = _fa3_cost_units(rows, h_qo)
    heap = [0.0] * max(min(int(c), len(units)), 1)
    for u in units:
        heapq.heappush(heap, heapq.heappop(heap) + u)
    return max(heap) * FA3_NS_PER_COST / 1e3


def t_pers(rows, c, h_qo, h_kv):
    """Persistent kernel over c SMs (it runs PERS_CLUSTERS_PER_SM blocks on each):
    bandwidth-bound on decode-shaped rows, compute-bound on large tiles; it fills
    whatever grid it is given via split-KV, so there is no staircase."""
    if not rows:
        return 0.0
    if c <= 0:
        return float("inf")
    eff = max(min(int(c), pers_width(rows, h_kv)), 1)     # in SMs
    compute = _work(rows, h_qo) / (eff * PERS_CLUSTERS_PER_SM * R_PERS)
    band = _bytes(rows, h_kv) / min(BW_PEAK, BW_PER_SM * eff)
    return max(compute, band) / 1e3


# Decision thresholds, fitted on the 50-cell boundary sweep (boundary_sweep.py)
# at (28,4,128) and (8,1,128) on H800 bf16, then validated on the captured sets,
# which are entirely different compositions. Device- and dtype-specific: refit
# with boundary_sweep.py + validate_scheduler.py on other hardware.
WIDTH_FRAC_MIN = 1.0      # FA3 must be able to fill the grid to be worth using
RHO_FUSED = 1.5           # decode side this much bigger than prefill -> one launch
CONC_MIN_TILES = 256      # co-launch needs this many KV tiles to pay off
CONC_MIN_RHO = 0.4


def plan_split(fa3_rows, pers_rows, h_qo, h_kv, num_sm=NUM_SM, min_share=4):
    """Pick the execution mode and grid split. Returns (mode, c_fa3, c_pers) in SMs.

      'fused'  everything in the persistent kernel at its full grid.
      'seq'    FA3 then persistent, both at full grid (what vLLM-style split does).
      'conc'   co-launched at a split grid.

    Two measured features decide it.

    width/num_sm -- FA3's work units are (qo_head x qo_tile) at cta_tile_q 128
      with NO KV splitting, and it does not fuse group heads, so it reloads KV
      once per q head. When h_qo per rank is small (8 at TP8, 4 at TP16) it can
      neither fill the grid nor amortise that traffic, and it loses on EVERY
      cell -- including pure prefill with no decode rows at all (401us vs 507us
      at 256 KV tiles). So the first question is not "is this tile large" but
      "can FA3 use the device for it".

    rho = t_pers(decode side) / t_fa3(prefill side) at full grid -- how the two
      populations balance. Above RHO_FUSED the prefill row is a rounding error
      next to the decode population and a second kernel only adds a launch and a
      join; below it, FA3's wgmma mainloop on consecutive large tiles wins by up
      to 1.62x and deserves the whole device.

    Co-launching is the rare case, not the default. It needs a very long prefix
    AND a balanced decode population, and it must NOT be given a large share of
    the grid: the persistent kernel is a cooperative launch, so over-provisioning
    it makes the whole grid wait for FA3's CTAs to drain (measured: 2286us at a
    88/44 split vs 1470us when the persistent side asked for a third of that).
    """
    if not fa3_rows:
        return "fused", 0, num_sm
    tiles = max(-(-kv // KV_TILE) for _, kv in fa3_rows)
    t_f = t_fa3(fa3_rows, num_sm, h_qo, h_kv)
    t_p = t_pers(pers_rows, num_sm, h_qo, h_kv) if pers_rows else 0.0
    rho = t_p / max(t_f, 1e-9)

    def balanced():
        # lexicographic: minimise the makespan, then the idle side
        cands = []
        for c_f in range(min_share, num_sm - min_share + 1):
            a = t_fa3(fa3_rows, c_f, h_qo, h_kv)
            b = t_pers(pers_rows, num_sm - c_f, h_qo, h_kv)
            cands.append((max(a, b), min(a, b), c_f))
        m = min(cands)
        return "conc", m[2], num_sm - m[2]

    # The width test applies even with no decode rows: at 8 q heads per rank the
    # persistent kernel beats FA3 on PURE prefill, so "large tile" alone is not
    # a reason to leave the fused kernel.
    if fa3_width(fa3_rows, h_qo, num_sm) < WIDTH_FRAC_MIN * num_sm:
        if pers_rows and tiles >= CONC_MIN_TILES and rho >= CONC_MIN_RHO:
            return balanced()
        return "fused", 0, num_sm
    if not pers_rows:
        return "seq", num_sm, num_sm
    return ("fused", 0, num_sm) if rho >= RHO_FUSED else ("seq", num_sm, num_sm)


class Dispatcher:
    """Plans and launches the two kernels concurrently at scheduler-chosen grids.

    Launch order matters and is not arbitrary: the persistent kernel is a
    COOPERATIVE launch, so every one of its blocks must be co-resident. It is
    issued first, on the default stream, and FA3 then fills the SMs it left
    free. Issuing FA3 first can make the cooperative launch wait for SMs to
    drain, which serialises the pair.
    """

    def __init__(self, bench, h_qo, h_kv, num_sm=132):
        self.b, self.h_qo, self.h_kv, self.num_sm = bench, h_qo, h_kv, num_sm
        self.s_fa3 = torch.cuda.Stream()
        self.ev = torch.cuda.Event()
        self.ev_fork = torch.cuda.Event()

    def plan(self, fa3_rows, pers_rows, c_fa3=None, c_pers=None, mode=None):
        if mode == "fused":
            pers_rows, fa3_rows, c_fa3, c_pers = fa3_rows + pers_rows, [], 0, self.num_sm
        if c_fa3 is None:
            mode, c_fa3, c_pers = plan_split(fa3_rows, pers_rows, self.h_qo,
                                             self.h_kv, self.num_sm)
            if mode == "fused":
                pers_rows, fa3_rows = fa3_rows + pers_rows, []
        self.mode = mode or "conc"
        self.c_fa3, self.c_pers = c_fa3, c_pers
        # The caps are read by the C++ planners via getenv at plan time, so the
        # split is baked into the plan -- nothing per-launch reads the env.
        # Only 'conc' restricts either grid. For 'fused' and 'seq' the caps must
        # be REMOVED, not set to num_sm: the persistent planner's natural grid is
        # 2 clusters per SM, so a cap of 132 would silently halve it.
        conc = self.mode == "conc"
        self.qf = None
        if fa3_rows:
            if conc:
                os.environ["FLASHINFER_FA3_MAX_CTAS"] = str(max(c_fa3, 1))
            else:
                os.environ.pop("FLASHINFER_FA3_MAX_CTAS", None)
            self.qf = self.b.plan_fa3(fa3_rows)
        self.qp = None
        if pers_rows:
            if conc:
                os.environ["FLASHINFER_PERSISTENT_MAX_CLUSTERS"] = str(
                    max(c_pers, 1) * PERS_CLUSTERS_PER_SM)
            else:
                os.environ.pop("FLASHINFER_PERSISTENT_MAX_CLUSTERS", None)
            self.qp, self.op, self.lp = self.b.plan_pa(pers_rows, False)
        return self.mode, c_fa3, c_pers

    def run(self):
        if self.mode == "seq" and self.qf is not None and self.qp is not None:
            self.b.fa3.run(self.qf, self.b.kv)
            self.b.pa.run(self.qp, self.b.kv, out=self.op, lse=self.lp)
            return
        if self.qp is None:
            self.b.fa3.run(self.qf, self.b.kv); return
        if self.qf is None:
            self.b.pa.run(self.qp, self.b.kv, out=self.op, lse=self.lp); return
        # Fork-join, entirely device-side so it is capturable in one graph:
        # fork event -> persistent (cooperative, default stream, issued first)
        # and FA3 on the side stream -> join event back onto the default stream.
        cur = torch.cuda.current_stream()
        self.ev_fork.record(cur)
        self.b.pa.run(self.qp, self.b.kv, out=self.op, lse=self.lp)
        with torch.cuda.stream(self.s_fa3):
            self.s_fa3.wait_event(self.ev_fork)
            self.b.fa3.run(self.qf, self.b.kv)
            self.ev.record(self.s_fa3)
        cur.wait_event(self.ev)
