import json, os, torch, shutil
from safetensors.torch import save_file

SRC, DST = "/dev/shm/w/qwen7b", "/dev/shm/w/qwen_eagle"
cfg = json.load(open(f"{SRC}/config.json"))
os.makedirs(DST, exist_ok=True)

# Simulated EAGLE-1 draft, calibrated against the target rather than guessed:
#   target  = 28 layers, hidden 3584, inter 18944, 28 heads / 4 kv heads
#   EAGLE-1 = ONE decoder layer at the same hidden size, which is the standard
#             shape for a 7B target (1/28 of the stack, ~233M params, ~3% of 7.6B)
# Embedding and lm_head are deliberately ABSENT: eagle_worker.py:167 calls
# set_embed_and_head() to share the target's, so a real EAGLE checkpoint does not
# ship them. Including them (as the first build did) wasted 2.2 of 2.6 GB and
# would have been overwritten at init anyway.
d = dict(cfg)
d["architectures"] = ["Qwen2ForCausalLMEagle"]
d["num_hidden_layers"] = 1
d["torch_dtype"] = "bfloat16"
json.dump(d, open(f"{DST}/config.json", "w"), indent=2)

H, NH, NKV = d["hidden_size"], d["num_attention_heads"], d["num_key_value_heads"]
HD, I = H // NH, d["intermediate_size"]
def r(*s): return (torch.randn(*s, dtype=torch.float32) * 0.02).to(torch.bfloat16)
w = {
    "layers.0.input_layernorm.weight": torch.ones(H, dtype=torch.bfloat16),
    "layers.0.self_attn.q_proj.weight": r(NH * HD, H),
    "layers.0.self_attn.q_proj.bias": r(NH * HD),
    "layers.0.self_attn.k_proj.weight": r(NKV * HD, H),
    "layers.0.self_attn.k_proj.bias": r(NKV * HD),
    "layers.0.self_attn.v_proj.weight": r(NKV * HD, H),
    "layers.0.self_attn.v_proj.bias": r(NKV * HD),
    "layers.0.self_attn.o_proj.weight": r(H, NH * HD),
    "layers.0.post_attention_layernorm.weight": torch.ones(H, dtype=torch.bfloat16),
    "layers.0.mlp.gate_proj.weight": r(I, H),
    "layers.0.mlp.up_proj.weight": r(I, H),
    "layers.0.mlp.down_proj.weight": r(H, I),
    "norm.weight": torch.ones(H, dtype=torch.bfloat16),
}
save_file(w, f"{DST}/model.safetensors", metadata={"format": "pt"})
for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
          "generation_config.json", "special_tokens_map.json"):
    if os.path.exists(f"{SRC}/{f}"): shutil.copy(f"{SRC}/{f}", f"{DST}/{f}")
n = sum(v.numel() for v in w.values())
print("draft: 1 layer, hidden %d, params %.0fM (%.1f%% of 7.6B), %d MB"
      % (H, n / 1e6, n / 7.6e9 * 100, os.path.getsize(f"{DST}/model.safetensors") / 1e6))

