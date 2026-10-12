# A2 serving gate: q8 tail run, the RoPE root cause, and the post-fix gate

Single H200 trial `f096519a4d17aa1c` (`vibecuda`). Kernel overlay unchanged from the A2 tail
evidence (`fused_plan.cuh` `3360dc0c…`, `persistent.cuh` `e0ac82f4…`). Serving tree
`/dev/shm/w/validated_upstream`, base sglang `cc739bd2d2f8c74005bdd438acdf645402b83b08` plus the
recorded working-tree changes plus the rope fix below.

## 1. The handed-off q8 run (pre-fix): gate FAILED, and the failure was not the kernel

`a2_tail_q8_serving_20261012.sh` (MTP chain width 8, `MTP_SPEC_STEPS=7`, classes 48/32/16, SM
split 112, real ShareGPT slices, no graphs, chunk512, background decode 1200). All three arms ran
12 requests with zero fallbacks/tracebacks; fused arms reached 101+ mixed steps.

- Tail path engages in serving: plan histograms move from `384/0/0/112` (TAIL0) to
  `192/0/192/112` (TAIL1), i.e. half the tile-48 work becomes tile-16, as designed.
- Gate: FAIL for both fused arms against LPT (long prefixes `16,10,0,42`; l2 first token differs).

**Noise floor.** Ten archived stock-LPT runs on the same prompts: R1≡R2 exactly, but R3 diverges
on long prompts at `[16,72,33,96]`, so a long prefix of 16 is inside stock's own noise. What stock
never did was disagree at l2 token 0: every stock run picks 387, every fused run picks 3010.

**Independent reference** (`l2_first_token_ref.py`, HF transformers, no sglang code; bf16, fp32
and fp32-without-YaRN agree within 0.6 logit):

| prompt | reference argmax (logit) | stock picks | fused picks |
|---|---|---|---|
| l0 | 220 (31.4) | 13710 (7.2) | 13710 |
| l1 | 1429 (31.0) | 5888 (26.4) | 5888 |
| l2 | 3010 (36.3) | 387 (8.2) | **3010** |
| l3 | 13 (31.4) | 11 (25.2) | 11 |

Both arms were wrong on 3 of 4 prompts by up to 24 logits, so the serving stack itself was
broken; fused happened to be right on l2.

## 2. Root cause: the YaRN override reset rope_theta 1e6 -> 1e4

`first_token_bisect.py` adds one gate ingredient at a time to a stock server:

| config | pre-fix | post-fix |
|---|---|---|
| C0 bare | 4/4 | 4/4 |
| C1 + YaRN override (`--json-model-override-args`, ctx 131072) | **0/4** (=stock LPT tokens) | 4/4 |
| C2 + chunk512 + mixed-chunk | 0/4 | 4/4 |
| C3 + background decodes | 0/4 | 4/4 |
| C4 + EAGLE3 q8 (the gate's stock config) | 0/4 | 4/4 |

Under transformers 5.x `rope_theta` lives inside `rope_parameters`. sglang's override loop
(`srt/utils/hf_transformers/config.py`, unmodified upstream) replaced the rope dict, dropping
`rope_theta`; `get_rope_config` then fell back to 10000. Verified directly:
`yarn_override -> get_rope_config: (10000, …)`.

**Every serving run before this fix used that override** — the correctness gate and the mixed
chat+TraceLab workloads — and so ran Qwen2.5 with the wrong RoPE base. Pre-fix serving correctness
verdicts are void, and pre-fix serving timings/EDF fits measured a model that was not Qwen2.5.

**Fix** (sglang-up `python/sglang/srt/utils/hf_transformers/config.py`): a rope override that
omits `rope_theta` inherits the config's; an explicit `rope_theta` still wins. Regression test
`test/manual/test_rope_override_theta.py`: 3/3 with the fix; the pre-fix file fails
`test_yarn_override_keeps_theta` with `10000 != 1000000.0`. The EAGLE3 draft resolves to 1e6 with
and without the override.

## 3. Post-fix q8 A2 gate: PASS

`a2_tail_q8_ropefix_20261012.sh` (identical environment, new output directory):
`A2_Q8_ROPEFIX_DONE status=0`.

- Fused (TAIL0 and TAIL1) vs LPT: long l0/l2/l3 match 96/96, l1 52/96; zero fallbacks/tracebacks.
- **TAIL1 vs TAIL0: bit-identical on all 12 requests** (every long 96/96, every short 1200/1200),
  identical verify-call counts. The tail dispatcher changes the plan (48+48 -> 48+16) without
  changing a single served token.
- Short s1 differs at token 0 (stock 19, fused 17); see section 4.

**Acceptance is not a fused regression.** Fused needs more verify calls for identical tokens
(l0: 52 vs 12), but per-draft acceptance (correct/proposed) is equal, ~0.12–0.21 in both arms. The
difference is accounting: stock `--enable-mixed-chunk` folds running decode rows into prefill
chunks as plain one-token decodes without speculation (l0: 12 verify + 11 correct + 1 = 24 of 96
tokens via spec), whereas the 4-mode schedule speculates on every row (52 + 43 + 1 = 96).
Tokens-per-verify is therefore not comparable between the two arms.

**Acceptance is low on these prompts for a different reason:** the fix moved stock accept rate only
0.140 -> 0.159. The gate's prompts are mid-sentence token slices with 1200-token `ignore_eos`
decodes. The goal's "common acceptance rate" must be measured on the real chat+TraceLab workload.

## 4. Short-prompt s1 adjudication

Same reference, short prompts (`--short --cand 19,17`). For s1 the reference gives 19 at 22.742
and 17 at 22.674 in fp32 — a 0.067-logit gap — and in bf16 tokens 18/16/20/17 all sit at 22.75,
one bf16 ulp below 19 (22.875): a five-way tie. Stock matched the argmax; fused landed one ulp
away. On s0/s2–s7 both arms return the reference argmax (s6 is itself a bf16 tie, 13 vs 624 at
23.125, and both picked 13).

**After the fix, every fused first token is the reference argmax or within one bf16 ulp of it.**
That is the correctness bar this gate can support; later-token divergence on the shorts follows
from these ties and is not used as evidence either way.

## Reproduction and artifacts

```bash
# on the trial, serialised on the single GPU
python l2_first_token_ref.py                      # long-prompt reference
python l2_first_token_ref.py --short --cand 19,17 # short-prompt reference
PYTHONPATH=$W/validated_upstream/python python first_token_bisect.py <outdir>
bash ropefix_chain_20261012.sh                    # post-fix bisect + q8 gate
```

- `a2_tail_q8_serving_20261012.tar.gz`: pre-fix q8 results, server logs, launch log, runner.
- `a2_tail_q8_ropefix_20261012.tar.gz`: post-fix bisect and q8 gate results.
- `sglang_rope_override_fix.patch`: the fix and its test.
