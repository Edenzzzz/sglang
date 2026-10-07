"""Fused draft tree build vs sglang's organize_draft_results + build_tree_kernel_efficient.

Reference inputs come from sglang's own select_top_k_tokens, step by step, so the score /
token / parent layouts are the real ones. Fused outputs go into buffers pre-filled with
garbage: anything the kernel fails to write shows up as a mismatch.
"""
import os
import sys
import torch
from torch.utils.cpp_extension import load

from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient, organize_draft_results
from sglang.srt.speculative.spec_utils import select_top_k_tokens

SRC = os.path.join(os.environ.get("SGL_PY", "/dev/shm/w/python"),
                   "sglang/srt/speculative/csrc/fused_tree_build.cu")
ext = load(name="fused_tree_build", sources=[SRC], extra_cuda_cflags=["-O3", "-arch=sm_90a"])


def draft_lists(bs, topk, steps, vocab=32000, hid=8, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    def step_probs(rows):
        p = torch.rand(rows, vocab, device="cuda", generator=g).pow(8)
        p = p / p.sum(-1, keepdim=True)
        return torch.topk(p, topk, dim=-1)
    topk_p, topk_index = step_probs(bs)
    hidden = torch.zeros(bs, hid, device="cuda")
    scores = None
    sl, tl, pl = [], [], []
    for i in range(steps):
        _, hidden, scores, info = select_top_k_tokens(i, topk_p, topk_index, hidden, scores, topk)
        sl.append(info[0]); tl.append(info[1]); pl.append(info[2])
        if i == steps - 1:
            break
        topk_p, topk_index = step_probs(bs * topk)
    return sl, tl, pl


def reference(verified_id, sl, tl, pl, seq_lens, topk, steps, n, mode, ssum):
    parent_list, top_idx, draft = organize_draft_results(sl, tl, pl, n)
    out = build_tree_kernel_efficient(verified_id, parent_list, top_idx, draft, seq_lens,
                                      ssum, topk, steps, n, tree_mask_mode=mode)
    return out, top_idx


def fused(verified_id, sl, tl, pl, seq_lens, topk, steps, n, mode, ssum, garbage=True):
    bs = seq_lens.numel()
    scores = torch.cat(sl, dim=1).flatten(1).float().contiguous()
    tokens = torch.cat(tl, dim=1).contiguous()
    parent_list = (torch.cat(pl[:-1], dim=1) if len(pl) > 1
                   else torch.zeros(bs, 1, dtype=torch.long, device="cuda")).contiguous()
    if mode == 2:
        dt = [torch.uint8, torch.uint16, torch.uint32][(n > 8) + (n > 16)]
        mask = torch.empty(n * bs, dtype=dt, device="cuda")
    elif mode == 1:
        mask = torch.empty(n * n * bs, dtype=torch.bool, device="cuda")
    else:
        mask = torch.empty(ssum * n + n * n * bs, dtype=torch.bool, device="cuda")
    pos = torch.empty(bs * n, dtype=torch.long, device="cuda")
    rbuf = torch.empty(3, bs, n, dtype=torch.long, device="cuda")
    top_idx = torch.empty(bs, n - 1, dtype=torch.long, device="cuda")
    draft = torch.empty(bs * n, dtype=torch.long, device="cuda")
    if garbage:
        mask.view(torch.uint8).fill_(0xAB) if mask.dtype == torch.bool else mask.fill_(0xAB)
        for t in (pos, rbuf, top_idx, draft):
            t.fill_(777)
    ext.fused_draft_tree_build(scores, tokens, parent_list, verified_id, seq_lens, top_idx,
                               draft, mask, pos, rbuf[0], rbuf[1], rbuf[2], topk, steps, n, mode)
    return (mask, pos, rbuf[0], rbuf[1], rbuf[2], draft), top_idx


def timeit(fn, iters=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1000


def graph_time(fn, iters=200):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return timeit(g.replay, iters)


def main():
    names = ["tree_mask", "positions", "retrive_index", "retrive_next_token",
             "retrive_next_sibling", "draft_tokens"]
    configs = [(1, 3, 4), (1, 5, 6), (4, 3, 8), (8, 5, 16), (4, 5, 32), (10, 8, 64)]
    bad = 0
    for topk, steps, n in configs:
        for bs in (1, 8, 64):
            for mode in (0, 1, 2):
                if mode == 2 and n > 32:  # sglang's packed dtypes stop at uint32
                    continue
                for seed in range(3):
                    sl, tl, pl = draft_lists(bs, topk, steps, seed=seed)
                    g = torch.Generator().manual_seed(seed)
                    seq_lens = torch.randint(100, 65536 if mode == 0 and bs <= 8 else 8192,
                                             (bs,), generator=g).cuda()
                    vid = torch.randint(0, 32000, (bs,), generator=g).cuda()
                    ssum = int(seq_lens.sum())
                    (r, rtop), (f, ftop) = (reference(vid, sl, tl, pl, seq_lens, topk, steps, n, mode, ssum),
                                            fused(vid, sl, tl, pl, seq_lens, topk, steps, n, mode, ssum))
                    diffs = [nm for nm, a, b in zip(names, r, f) if not torch.equal(a, b)]
                    if not torch.equal(rtop, ftop):
                        diffs.append("top_scores_index")
                    if diffs:
                        bad += 1
                        print(f"MISMATCH topk={topk} steps={steps} n={n} bs={bs} mode={mode} "
                              f"seed={seed}: {diffs}")
    print(f"CORRECTNESS: {bad} mismatching cases")

    print(f"{'topk':>4} {'steps':>5} {'n':>3} {'bs':>3} {'mode':>4} {'seqlen':>6} | "
          f"{'ref eager':>10} {'fused eager':>11} | {'ref graph':>10} {'fused graph':>11}  us")
    for topk, steps, n in [(1, 3, 4), (4, 3, 8), (8, 5, 16)]:
        for bs, slen in [(1, 4096), (32, 4096), (32, 32768), (128, 8192)]:
            for mode in (0, 1):
                sl, tl, pl = draft_lists(bs, topk, steps)
                seq_lens = torch.full((bs,), slen, dtype=torch.long, device="cuda")
                vid = torch.zeros(bs, dtype=torch.long, device="cuda")
                ssum = bs * slen
                fr = lambda: reference(vid, sl, tl, pl, seq_lens, topk, steps, n, mode, ssum)
                ff = lambda: fused(vid, sl, tl, pl, seq_lens, topk, steps, n, mode, ssum, garbage=False)
                re_, fe = timeit(fr), timeit(ff)
                try:
                    rg = graph_time(fr)
                except Exception as e:
                    print(f"reference graph capture failed: {e!r}"[:200])
                    rg = float("nan")
                fg = graph_time(ff)
                print(f"{topk:>4} {steps:>5} {n:>3} {bs:>3} {mode:>4} {slen:>6} | "
                      f"{re_:10.1f} {fe:11.1f} | {rg:10.1f} {fg:11.1f}", flush=True)
    print("TREEDONE")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
