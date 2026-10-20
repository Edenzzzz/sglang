"""Random-weight EAGLE3 draft head shaped for a target model, for MTP runs with simulated acceptance
(SGLANG_SIMULATE_ACC_LEN). The head does the same draft work per step as a trained one (one decoder layer at the
target's width, the 3x-hidden fc, a draft-vocab LM head); the simulated accept index replaces its acceptance, so its
weights only need the right shapes. d2t / t2d (draft <-> target vocab maps) are copied from a reference head of the
same tokenizer.
  python make_sim_eagle3_head.py <target model dir> <reference head dir> <out dir>
Claude, 2026-10-20.
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

target, ref, out = sys.argv[1:4]
tc = json.load(open(f"{target}/config.json"))
rc = json.load(open(f"{ref}/config.json"))
h, inter = tc["hidden_size"], tc["intermediate_size"]
nq, nkv = tc["num_attention_heads"], tc["num_key_value_heads"]
hd = tc.get("head_dim", h // nq)
dv = rc["draft_vocab_size"]
g = torch.Generator().manual_seed(0)


def rnd(*shape):
    return (torch.randn(*shape, generator=g) * 0.02).to(torch.bfloat16)


with safe_open(f"{ref}/model.safetensors", "pt") as f:
    d2t, t2d = f.get_tensor("d2t"), f.get_tensor("t2d")
w = {
    "d2t": d2t, "t2d": t2d,
    "fc.weight": rnd(h, 3 * h),
    "lm_head.weight": rnd(dv, h),
    "norm.weight": torch.ones(h, dtype=torch.bfloat16),
    "midlayer.hidden_norm.weight": torch.ones(h, dtype=torch.bfloat16),
    "midlayer.input_layernorm.weight": torch.ones(h, dtype=torch.bfloat16),
    "midlayer.post_attention_layernorm.weight": torch.ones(h, dtype=torch.bfloat16),
    "midlayer.mlp.gate_proj.weight": rnd(inter, h),
    "midlayer.mlp.up_proj.weight": rnd(inter, h),
    "midlayer.mlp.down_proj.weight": rnd(h, inter),
    "midlayer.self_attn.q_proj.weight": rnd(nq * hd, 2 * h),
    "midlayer.self_attn.k_proj.weight": rnd(nkv * hd, 2 * h),
    "midlayer.self_attn.v_proj.weight": rnd(nkv * hd, 2 * h),
    "midlayer.self_attn.o_proj.weight": rnd(h, nq * hd),
}
os.makedirs(out, exist_ok=True)
save_file(w, f"{out}/model.safetensors")
cfg = dict(rc, hidden_size=h, intermediate_size=inter, num_attention_heads=nq, num_key_value_heads=nkv, head_dim=hd,
           vocab_size=tc["vocab_size"], rope_theta=tc.get("rope_theta", rc.get("rope_theta")),
           rms_norm_eps=tc.get("rms_norm_eps", rc.get("rms_norm_eps")))
json.dump(cfg, open(f"{out}/config.json", "w"))
print("wrote", out, {k: tuple(v.shape) for k, v in w.items() if "weight" in k})
