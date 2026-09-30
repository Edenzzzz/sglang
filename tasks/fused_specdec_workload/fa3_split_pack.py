#!/usr/bin/env python3
"""Split-KV and GQA packing for FA3, measured before committing to kernel work.

Both are emulated at the batch level so the projections can be checked without
touching the CUTLASS mainloop:

  split-KV  one row (qo, kv) becomes S rows (qo, kv/S) over disjoint KV ranges,
            run with return_lse, then merged with flashinfer.merge_states. This
            is exactly what a real split-KV planner would produce, minus the
            fused epilogue -- so it measures the WIDTH fix (FA3's work units are
            (qo_head x qo_tile) with no KV split, so width = h_qo*ceil(qo/128)).

  GQA pack  all q heads in a group attend the same KV, so a group can be
            presented as ONE head with qo_len*gqa rows. That is what the
            persistent kernel does internally via packed_qo_len, and it is legal
            here because the cached-prefix part is non-causal. Measures the TILE
            OCCUPANCY fix that matters for the short-qo classes (a 4-token verify
            row occupies 4/128 of an FA3 tile unpacked, 32/128 packed at gqa 8).
"""
import argparse, sys, os
import numpy as np, torch, flashinfer
from flashinfer.testing.utils import bench_gpu_time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_4mode import Bench, uncapped, med


def fa3_split(b, qo, kv, S):
    """Plan FA3 over S KV-chunks of one row; returns a callable and the merge."""
    per = kv // S
    rows = [(qo, per + (kv - per * S if i == S - 1 else 0)) for i in range(S)]
    ql = torch.tensor([qo] * S, dtype=torch.int32)
    kl = torch.tensor([r[1] for r in rows], dtype=torch.int32)
    qi = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(ql, 0).int()])
    # each chunk starts at its own offset into the (shared) KV region
    starts = np.concatenate([[0], np.cumsum([r[1] for r in rows])[:-1]])
    ki = torch.tensor(np.concatenate([starts, [starts[-1] + rows[-1][1]]]),
                      dtype=torch.int32)
    b.fa3.plan(qi, ki, b.idx, torch.ones_like(kl), b.H_QO, b.H_KV, b.D, 1,
               causal=False, q_data_type=torch.bfloat16,
               kv_data_type=torch.bfloat16, non_blocking=True)
    q = b.qbuf[: qo * S]
    def run():
        o, lse = b.fa3.run(q, b.kv, return_lse=True)
        v = o.view(S, qo, b.H_QO, b.D).transpose(0, 1).contiguous()
        s = lse.view(S, qo, b.H_QO).transpose(0, 1).contiguous()
        return flashinfer.merge_states(v, s)
    return run


def fa3_packed(b, qo, kv, gqa):
    """FA3 over one row with the GQA group packed into a single head."""
    saved = b.H_QO, b.H_KV
    b.H_QO, b.H_KV = 1, 1
    try:
        qf = b.plan_fa3([(qo * gqa, kv)])
        run = lambda: b.fa3.run(qf, b.kv)
        run()          # validate it plans and launches
    finally:
        b.H_QO, b.H_KV = saved
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="8,1,128")
    ap.add_argument("--qo", type=int, default=1024)
    ap.add_argument("--kv", type=int, default=32768)
    a = ap.parse_args()
    heads = tuple(int(x) for x in a.heads.split(","))
    b = Bench(heads); uncapped()
    gqa = heads[0] // heads[1]
    width = heads[0] * -(-a.qo // 128)
    print(f"heads {heads}  gqa {gqa}  row ({a.qo}, {a.kv})  "
          f"FA3 unsplit width = h_qo*ceil(qo/128) = {width} of 132 CTAs\n")

    qf = b.plan_fa3([(a.qo, a.kv)])
    t_base = med(lambda: b.fa3.run(qf, b.kv))
    q, o, l = b.plan_pa([(a.qo, a.kv)], False)
    t_pers = med(lambda: b.pa.run(q, b.kv, out=o, lse=l))
    print(f"{'variant':<28} {'us':>9} {'vs FA3':>8} {'vs pers':>8} {'width':>7}")
    print(f"{'FA3 as-is':<28} {t_base:>9.1f} {1.0:>7.2f}x {t_pers/t_base:>7.2f}x {min(width,132):>7}")
    print(f"{'persistent (pack+split+LPT)':<28} {t_pers:>9.1f} "
          f"{t_base/t_pers:>7.2f}x {1.0:>7.2f}x {132:>7}")
    for S in (2, 3, 4, 6, 8):
        try:
            run = fa3_split(b, a.qo, a.kv, S)
            t = med(run)
            print(f"{'FA3 split-KV S=' + str(S):<28} {t:>9.1f} {t_base/t:>7.2f}x "
                  f"{t_pers/t:>7.2f}x {min(width*S,132):>7}")
        except Exception as e:
            print(f"{'FA3 split-KV S=' + str(S):<28} FAILED {str(e)[:60]}")
    try:
        run = fa3_packed(b, a.qo, a.kv, gqa)
        t = med(run)
        print(f"{'FA3 gqa-packed (1 head)':<28} {t:>9.1f} {t_base/t:>7.2f}x "
              f"{t_pers/t:>7.2f}x {min(-(-a.qo*gqa//128),132):>7}")
    except Exception as e:
        print(f"{'FA3 gqa-packed':<28} FAILED {str(e)[:60]}")


if __name__ == "__main__":
    main()
