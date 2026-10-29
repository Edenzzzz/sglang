"""Do the two roles of a four-mode step ever touch the same KV pages? (2026-10-29)

This number gates cross-role shared-prefix KV reuse: grouping a step's work by KV page instead of by request so a
page both roles read is fetched once. The mechanism is only conceivable because C2 puts a prefill chunk and the
decode/verify inputs in ONE step -- a disaggregated or two-call design has the roles in different kernels, often on
different nodes -- so it is worth knowing what it would be worth before building it. DeFT (arXiv 2404.00242) does
KV-guided grouping for a decoding TREE inside one call; ours would be cross-ROLE and cross-REQUEST, which is a
different grouping over different work.

It is answered here by reading the workload generator rather than by instrumenting a step, because a tracer that
copied per-step page sets to the host would perturb the very step times the traced arm exists to measure, and
because the generator settles it exactly rather than statistically.

Two roles share a page only if they share a token PREFIX, since the radix cache hands identical pool slots to
identical prefixes. On the TraceLab session replay (`bench_serving.py`, tracelab dataset) two things decide it:

  1. ACROSS sessions. Each session's prompt is drawn independently:
         available_tokens = list(tokenizer.get_vocab().values())
         session_prompt_ids = random.choices(available_tokens, k=first_prefix_len)
     Two independent uniform draws over a vocabulary of size V agree on their first token with probability 1/V, so
     the expected shared prefix is 1/(V-1) tokens -- about 7 millionths of a token for Qwen2.5's vocabulary. Zero
     pages, and not approximately zero: a page is shared only if EVERY token in it matches.

  2. WITHIN a session. Round N+1 reuses round N's tokens exactly (`prefix_ids = session_prompt_ids[:prefix_len]`),
     which is where the replay's 92.7% prefix hit rate comes from. But the replay issues a session's round N+1
     only after round N finishes, so a session never has two rounds in flight, and the request being prefilled is
     never from the same session as a resident decode input.

So the overlap is zero for two independent reasons, and the reuse mechanism has nothing to reuse on the workload
our headline results are measured on. The script states the arithmetic and prints the bound.

  python cross_role_page_overlap.py [--vocab 151936] [--page-size 1]
Claude, 2026-10-29.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=151936, help="Qwen2.5 vocabulary size")
    ap.add_argument("--page-size", type=int, default=1)
    ap.add_argument("--prefix-hit", type=float, default=92.7, help="the replay's measured prefix hit rate (%%)")
    a = ap.parse_args()

    print("Cross-role KV page overlap within one four-mode step, TraceLab session replay\n")
    print("1. ACROSS sessions -- independent uniform token draws")
    p_token = 1.0 / a.vocab
    exp_prefix = 1.0 / (a.vocab - 1)
    p_page = p_token ** a.page_size
    print(f"   vocabulary {a.vocab}, page size {a.page_size}")
    print(f"   P(first tokens of two sessions agree) = 1/V = {p_token:.3e}")
    print(f"   expected shared prefix               = 1/(V-1) = {exp_prefix:.3e} tokens")
    print(f"   P(an entire page matches)            = (1/V)^{a.page_size} = {p_page:.3e}")
    print("   => 0 shared pages\n")

    print("2. WITHIN a session -- rounds reuse tokens exactly, but never overlap in time")
    print(f"   round N+1 reuses round N's prefix verbatim, which is the replay's {a.prefix_hit}% prefix hit rate,")
    print("   but a session's round N+1 is issued only after round N finishes, so the request being prefilled")
    print("   is never from the same session as a resident decode input.")
    print("   => 0 shared pages\n")

    print("BOUND: cross-role page overlap is 0 on the session replay, for two independent reasons.")
    print("Consequence: KV-page grouping across roles has nothing to reuse on the workload the headline results")
    print("use. On a fan-out workload built to maximise it (programs of one task sharing a long prefix) the page")
    print("overlap is high by construction, and is still bounded at <= 1.3% of step time, because compiling the")
    print("prefill role's KV loads out of the kernel entirely makes the step only 0.1-1.3% faster -- those reads")
    print("are already hidden behind compute and hit in L2 (tasks/perf_model/fanout_role_overlap.py).")
    print("\nCROSS_ROLE_OVERLAP pages=0 time_bound<=1.3%")


if __name__ == "__main__":
    main()
