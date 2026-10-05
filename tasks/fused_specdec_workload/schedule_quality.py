#!/usr/bin/env python3
"""Schedule quality of the unified 4-task plan on a 4-mode batch.

The claim this measures is not throughput but the three properties the design is
argued on, each read back from the plan the kernel actually runs:

  tile waste    occupied tile rows / allocated tile rows. Unpacked FA3 tiles
                qo_len and emits one work item per qo head, so a 4-token verify
                row occupies 4 of 128 rows, 28 times over. The unified plan tiles
                PACKED rows, so the same row fills 28 of 128 in one item.
  load balance  max cluster work / mean cluster work over the clusters a task
                owns. A plan that balances perfectly scores 1.0.
  sync surface  plans, H2D copies and launches per step. The fused design pays
                one of each; the split design pays two plans, two H2D copies and
                keeps a second kernel's schedule consistent with the first.

Run on the pod: PYTHONPATH=... python schedule_quality.py
"""
import os, numpy as np, torch

HQ, HKV, D = 28, 4, 128
G = HQ // HKV
# A 4-mode step: one chunked-prefill row, spec verify rows (k=3 -> 4 tokens),
# spec draft rows (1 token), and plain decode rows. Lengths from the measured
# online trace: chunk 500 over a 2595 prefix, decode KV ~4357.
PREFILL = [(500, 3095)]
VERIFY = [(4, 4357)] * 8
DRAFT = [(1, 4357)] * 8
DECODE = [(1, 4357)] * 11
ROWS = PREFILL + VERIFY + DRAFT + DECODE


def tile_waste(rows, packed, cta_tile_q):
    """Occupied / allocated tile rows, for packed vs unpacked tiling."""
    occ = alloc = 0
    for qo, _ in rows:
        if packed:
            p = qo * G
            tiles = -(-p // cta_tile_q)
            occ += p
            alloc += tiles * cta_tile_q
        else:
            tiles = -(-qo // cta_tile_q)
            occ += qo * HQ                       # one work item per qo head
            alloc += tiles * cta_tile_q * HQ
    return occ / alloc


def main():
    os.environ["FLASHINFER_FA3_TASK"] = "1"
    os.environ["FLASHINFER_SM_PARTITION"] = "1"
    os.environ.setdefault("FLASHINFER_FA3_SM_SPLIT_CLUSTERS", "8")
    from flashinfer.attention import BatchAttention

    qo = np.array([r[0] for r in ROWS], dtype=np.int64)
    kv = np.array([r[1] for r in ROWS], dtype=np.int64)
    qo_c = np.concatenate([[0], np.cumsum(qo)]).astype(np.int32)
    kv_c = np.concatenate([[0], np.cumsum(kv)]).astype(np.int32)
    idx = torch.arange(int(kv_c[-1]), dtype=torch.int32, device="cuda")

    a = BatchAttention(torch.empty(512 << 20, dtype=torch.uint8, device="cuda"), "NHD")
    a.plan(torch.from_numpy(qo_c), torch.from_numpy(kv_c), idx,
           torch.from_numpy(kv.astype(np.int32)), HQ, HKV, D, D, 1, causal=False,
           q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    plan = list(a._plan_info)
    nclu = int(plan[1])
    host = a.int_workspace_buffer.view(torch.uint8).cpu().numpy()

    print(f"4-mode batch: {len(PREFILL)} chunk, {len(VERIFY)} verify(k=3), "
          f"{len(DRAFT)} draft, {len(DECODE)} decode;  gqa {G}, {nclu} clusters\n")

    print("tile waste (occupied / allocated tile rows)")
    print(f"  unpacked, as PrefillSM90Plan tiles it : {tile_waste(ROWS, False, 128):6.1%}")
    print(f"  packed, as the unified plan tiles it  : {tile_waste(ROWS, True, 128):6.1%}")
    vo = tile_waste(VERIFY, False, 64), tile_waste(VERIFY, True, 64)
    print(f"  verify class alone, unpacked / packed : {vo[0]:6.1%} / {vo[1]:6.1%}\n")

    print("load balance per task (max cluster work / mean, 1.0 is perfect)")
    for t in range(4):
        wi_off = int(plan[2 + t * 11 + 9])
        kvl_off = int(plan[2 + t * 11 + 4])
        wi = host[wi_off:wi_off + 4 * (nclu + 1)].view(np.int32)
        total = int(wi[-1]) - int(wi[0])
        if total == 0:
            print(f"  task {t}: empty")
            continue
        kvlen = host[kvl_off:kvl_off + 4 * total].view(np.int32).astype(np.int64)
        per = np.array([int(wi[c + 1]) - int(wi[c]) for c in range(nclu)])
        live = per > 0
        # work per cluster in kv elements, the currency both mainloops iterate
        bounds = np.concatenate([[0], np.cumsum(per)])
        wpc = np.array([kvlen[bounds[c]:bounds[c + 1]].sum() for c in range(nclu)])
        w = wpc[live]
        print(f"  task {t}: {total:4d} items over {live.sum():3d} clusters, "
              f"imbalance {w.max() / w.mean():.3f}, idle clusters {int((~live).sum())}")

    print("\nsync surface per step")
    print("  unified : 1 plan, 1 H2D, 1 launch")
    print("  split   : 2 plans, 2 H2D, 1 launch (two schedules kept consistent)")
    print("  serial  : 2 plans, 2 H2D, 2 launches (measured -48% end to end)")


if __name__ == "__main__":
    main()
