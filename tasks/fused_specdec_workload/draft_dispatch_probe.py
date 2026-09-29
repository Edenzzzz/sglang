#!/usr/bin/env python3
"""Does the spec-draft class need its own low-SM dispatch, or fuse into tile 16?

Draft rows are the smallest work in the mixture: qo_len=1, so packed_qo = the
GQA ratio, and they land in the tile-16 class alongside plain decode rows.
Two questions decide the design:

  (1) Does draft-only work saturate the GPU? Achieved HBM bandwidth as the row
      count grows says how much of the machine the class can actually use. If it
      plateaus well below peak, carving it onto a few SMs costs nothing and frees
      the rest; if it scales to peak, a low-SM partition would throttle it.

  (2) Does fusing it with decode into one tile-16 queue already overlap them?
      Compare fused(draft+decode) against draft-alone and decode-alone. If fused
      ~= max(alone), the single queue is already doing the job and a separate
      dispatch can only add launch cost. If fused ~= sum(alone), they serialise
      inside the kernel and a concurrent partition has something to win.
"""
import argparse
import numpy as np, torch, flashinfer
from flashinfer.testing.utils import bench_gpu_time

MAX_PAGES = 8_000_000
MAX_Q = 32_768


class H:
    def __init__(self, heads):
        self.H_QO, self.H_KV, self.D = heads
        self.kv = torch.empty(MAX_PAGES, 2, 1, self.H_KV, self.D,
                              device="cuda", dtype=torch.bfloat16).normal_()
        self.q = torch.empty(MAX_Q, self.H_QO, self.D, device="cuda",
                             dtype=torch.bfloat16).uniform_()
        self.idx = torch.arange(MAX_PAGES, dtype=torch.int32, device="cuda")
        self.pa = flashinfer.BatchAttention(kv_layout="NHD")

    def time(self, rows):
        ql = torch.tensor([q for q, _ in rows], dtype=torch.int32)
        kl = torch.tensor([max(k, 1) for _, k in rows], dtype=torch.int32)
        if int(kl.sum()) > MAX_PAGES - 256 or int(ql.sum()) > MAX_Q:
            return None
        qi = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(ql, 0).int()])
        ki = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(kl, 0).int()])
        nq = int(qi[-1]); q = self.q[:nq]
        best = None
        for flip in (True, False):
            self.pa.plan(qi, ki, self.idx, kl, self.H_QO, self.H_KV, self.D, self.D, 1,
                         causal=False, q_data_type=torch.bfloat16,
                         kv_data_type=torch.bfloat16, flipped_schedule=flip)
            o = torch.empty(nq, self.H_QO, self.D, device="cuda", dtype=torch.bfloat16)
            l = torch.empty(nq, self.H_QO, device="cuda", dtype=torch.float32)
            t = float(np.median(bench_gpu_time(lambda: self.pa.run(q, self.kv, out=o, lse=l))))
            best = t if best is None else min(best, t)
        return best * 1e3  # us

    def kv_bytes(self, rows):
        return sum(k for _, k in rows) * 2 * self.H_KV * self.D * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="28,4,128")
    ap.add_argument("--kv", type=int, default=41000, help="per-row KV for draft rows")
    ap.add_argument("--peak-tbs", type=float, default=3.35, help="HBM peak TB/s (H800)")
    a = ap.parse_args()
    heads = tuple(int(x) for x in a.heads.split(","))
    h = H(heads)
    print(f"heads {heads} gqa={heads[0]//heads[1]}, draft rows qo=1 @ {a.kv} KV, "
          f"peak {a.peak_tbs} TB/s\n")

    print("(1) does draft-only work saturate the GPU?")
    print(f"   {'rows':>5} {'us':>9} {'GB':>8} {'TB/s':>7} {'% peak':>7} {'us/row':>8}")
    prev = None
    for n in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 182):
        rows = [(1, a.kv)] * n
        t = h.time(rows)
        if t is None:
            continue
        gb = h.kv_bytes(rows) / 1e9
        tbs = gb / (t * 1e-6) / 1e3
        print(f"   {n:>5} {t:>9.1f} {gb:>8.2f} {tbs:>7.2f} {100*tbs/a.peak_tbs:>6.0f}% "
              f"{t/n:>8.2f}")
        prev = t

    print("\n(2) fused into the tile-16 queue vs run apart")
    print(f"   {'composition':<34} {'draft':>9} {'decode':>9} {'fused':>9} "
          f"{'fused/max':>10} {'fused/sum':>10}")
    for nd, ndec, kv_dec in ((21, 24, 41000), (48, 48, 11000), (96, 96, 11000)):
        draft = [(1, a.kv)] * nd
        # "decode" here = the rest of the tile-16 population at a different KV,
        # i.e. plain decode rows the draft class would share its queue with
        dec = [(1, kv_dec)] * ndec
        td, tc, tf = h.time(draft), h.time(dec), h.time(draft + dec)
        if None in (td, tc, tf):
            continue
        print(f"   {f'{nd} draft + {ndec} decode @{kv_dec//1000}K':<34} {td:>9.1f} {tc:>9.1f} "
              f"{tf:>9.1f} {tf/max(td,tc):>9.2f}x {tf/(td+tc):>9.2f}x")
    print("""
   fused/max ~ 1.0  -> the two populations already overlap inside one queue
   fused/sum ~ 1.0  -> they serialise; a concurrent partition could win""")


if __name__ == "__main__":
    main()
