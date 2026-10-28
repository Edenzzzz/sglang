"""How much KV do the two roles of one four-mode step actually read in common, on the fan-out workload?

This is the gating number for cross-role shared-prefix KV reuse. The mechanism: both roles are in one step under
one plan, so the plan could group work by KV page instead of by request and fetch a shared page once. It is only
available because C2 puts a prefill chunk and the decode/verify rows in the same forward -- a disaggregated or
two-call design has the roles in different kernels, often on different nodes.

It is already settled that the overlap is ~0 on the superposed TraceLab replay, for two structural reasons: that
driver synthesises each session's tokens from the global RNG, so two concurrent sessions share no prefix, and
rounds within a session are sequential, so the session that is prefilling has no decode rows of its own. Its 92.7%
prefix hit is temporal (round N+1 reusing round N), not spatial.

Fan-out is different by construction: `drivers/serving/fanout_agents.py` builds K task prefixes of L tokens and
starts every program of a task from `list(prefixes[task])`, so all concurrent programs of one task share those L
tokens exactly. This script computes what that is worth, from the workload's own parameters plus the batch
composition a run reports, and it reports a CEILING rather than a measurement: it assumes the plan can reuse every
page the two roles share, and it reports alongside it the measured cost of the chunk's KV loads, which is
what bounds the mechanism in TIME rather than in pages.

What is modelled, and what is not. Attention over a cached prefix reads every cached key, so a prefilling
program's chunk reads its whole prefix (the task prefix plus its own history) and each decode row reads its own.
The shared part is exactly the task prefix, and it counts only when at least one resident decode row belongs to the
same task as the chunk -- which is what `--siblings-resident` expresses, and the one quantity that needs a real run
to pin down. Everything else follows from the driver's parameters.

  python fanout_role_overlap.py [--tasks 8] [--prefix-len 131072] [--concurrency 32] [--rounds 4]
Claude, 2026-10-28.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=8, help="K, distinct shared task prefixes")
    ap.add_argument("--prefix-len", type=int, default=131072, help="L, tokens of shared task prefix")
    ap.add_argument("--concurrency", type=int, default=32, help="C, programs kept active")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--append-tokens", type=int, default=875,
                    help="uncached tokens a round appends (TraceLab median; the fan-out driver's own appends)")
    ap.add_argument("--output-tokens", type=int, default=400, help="tokens a round emits")
    ap.add_argument("--page-size", type=int, default=1)
    a = ap.parse_args()

    per_task = a.concurrency / a.tasks
    print(f"K={a.tasks} tasks, L={a.prefix_len} shared tokens, C={a.concurrency} programs "
          f"({per_task:.1f} per task, so {per_task - 1:.1f} siblings), R={a.rounds} rounds")
    print(f"page size {a.page_size}, so {a.prefix_len // a.page_size} pages of shared prefix\n")

    # A program's own history grows each round by what it appended and what it emitted.
    print(f"{'round':>6}{'own history':>13}{'chunk reads':>13}{'shared':>9}{'share of chunk KV':>20}")
    rows = []
    for r in range(a.rounds):
        history = r * (a.append_tokens + a.output_tokens)
        chunk_reads = a.prefix_len + history
        frac = a.prefix_len / chunk_reads
        rows.append(frac)
        print(f"{r:>6}{history:>13}{chunk_reads:>13}{a.prefix_len:>9}{100 * frac:>19.1f}%")
    mean_frac = sum(rows) / len(rows)

    print(f"\nCeiling on the chunk role's KV traffic that a sibling row reads in the SAME step: "
          f"{100 * mean_frac:.1f}% averaged over rounds")
    print("This is the share of the chunk's reads that are redundant with a row's, GIVEN a resident sibling.")
    for resident in (1.0, 0.75, 0.5, 0.25):
        print(f"  if a sibling of the prefilling program is resident on {100 * resident:.0f}% of merged steps: "
              f"{100 * mean_frac * resident:.1f}% of chunk KV traffic is shareable")
    print("\nWhat a run would still pin down: on what fraction of merged steps a decode row of the chunk's own task")
    print("is resident. That is a CPU-side count in the plan, costs no GPU time, and is ~0 on the session replay.")

    # The redundancy is in PAGES. What it is worth in TIME is bounded by what the chunk's KV loads cost at all,
    # and that was measured directly while bounding KV multicast: compiling FA3's KV loads out of the kernel makes
    # the step only 0.1-1.3% faster, because those reads are already hidden behind compute and hit in L2 (99% of
    # FA3's reads are re-reads by the same prefill row's other query tiles). Eliminating cross-role redundancy
    # cannot beat eliminating the loads altogether.
    fa3_load_share = 1.3
    print(f"\nBut page redundancy is not time. Compiling FA3's KV loads out entirely makes the step only "
          f"0.1-{fa3_load_share}% faster")
    print("(measured while bounding KV multicast: the loads are hidden behind compute and hit in L2), so removing")
    print(f"cross-role redundancy is worth at most {fa3_load_share:.1f}% of the step even at a "
          f"{100 * mean_frac:.0f}% page overlap --")
    print("and that is the ceiling for a mechanism whose own grouping and indirection are not free. The high")
    print("overlap is therefore not evidence that this mechanism pays; it is evidence that the workload has the")
    print("structure the mechanism needs, on a step whose KV traffic is not the bottleneck.")
    print(f"\nFANOUT_OVERLAP_CEILING pages {100 * mean_frac:.1f}% time <= {fa3_load_share:.1f}%")


if __name__ == "__main__":
    main()
