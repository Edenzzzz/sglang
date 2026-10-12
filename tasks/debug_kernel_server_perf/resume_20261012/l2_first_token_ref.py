"""Independent next-token reference for the long correctness prompts (HF transformers).

Every stock LPT run picks token 387 at l2 position 0; every fused run picks 3010. This
computes last-position logits for each long prompt with no sglang code involved, so the
two candidates can be compared against a reference that neither serving arm shares.
Prompt construction mirrors srv_correctness.prompts() exactly.
"""
import json
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = "/dev/shm/w/qwen7b"
DATASET = "/dev/shm/w/datasets/ShareGPT.original.json"
CANDIDATE_IDS = [int(x) for x in sys.argv[sys.argv.index("--cand") + 1].split(",")] if "--cand" in sys.argv else []
CANDIDATES = {"l0": [13710], "l1": [5888], "l2": [387, 3010], "l3": [11]}

tok = AutoTokenizer.from_pretrained(MODEL)
data = json.load(open(DATASET))
text = "\n".join(t["value"] for d in data[:3000] for t in d.get("conversations", []))
ids = tok(text[:4_000_000])["input_ids"]
long_ = {f"l{i}": ids[600_000 + i * 50_000: 600_000 + i * 50_000 + 12_000] for i in range(4)}
if "--short" in sys.argv:
    long_ = {f"s{i}": ids[i * 50_000: i * 50_000 + 2048] for i in range(8)}


def run(dtype, yarn):
    cfg = AutoConfig.from_pretrained(MODEL)
    if yarn:
        # Merge, do not replace: transformers 5.x keeps rope_theta inside this dict, and
        # replacing it wholesale leaves the base frequency None.
        yarn_args = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
        if getattr(cfg, "rope_parameters", None) is not None:
            cfg.rope_parameters = {**cfg.rope_parameters, **yarn_args}
        else:
            cfg.rope_scaling = {**(cfg.rope_scaling or {}), "type": "yarn", **yarn_args}
    print("rope:", getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None),
          "theta:", getattr(cfg, "rope_theta", None), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, config=cfg, torch_dtype=dtype, attn_implementation="sdpa").cuda().eval()
    for name, prompt in long_.items():
        x = torch.tensor([prompt], device="cuda")
        with torch.no_grad():
            h = model.model(input_ids=x).last_hidden_state[:, -1]
            logits = model.lm_head(h).float()[0]
        top = torch.topk(logits, 5)
        cand = {c: round(logits[c].item(), 4) for c in CANDIDATES.get(name, CANDIDATE_IDS)}
        print(f"{dtype} yarn={yarn} {name}: top5={list(zip(top.indices.tolist(), [round(v, 4) for v in top.values.tolist()]))} "
              f"candidates={cand}", flush=True)
    del model
    torch.cuda.empty_cache()


for dtype, yarn in ([(torch.bfloat16, True), (torch.float32, True)] if "--short" in sys.argv else [(torch.bfloat16, True), (torch.float32, True), (torch.float32, False)]):
    run(dtype, yarn)
print("L2_REF_DONE", flush=True)
