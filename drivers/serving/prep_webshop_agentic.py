"""Convert THUDM/AgentInstruct WebShop trajectories (multi-turn web browsing: each human turn is the
environment's page observation, each gpt turn the agent's action) into the agentic-trace JSON that
sglang.benchmark's AgenticTraceDataset replays round by round, feeding the server's own reply back
into the next round. Only the human (environment) turns are kept as inputs; the recorded gpt actions
are dropped because the served model generates its own.
Usage: prep_webshop_agentic.py <webshop.parquet> <out.json>"""
import json
import sys

import pandas as pd

src, out = sys.argv[1], sys.argv[2]
frame = pd.read_parquet(src)
conversations = []
for record in frame["conversations"]:
    turns, chars = [], 0
    for message in record:
        if message["from"] != "human":
            continue
        chars += len(message["value"])
        turns.append({"messages": [{"role": "user", "content": message["value"]}],
                      "prompt_tokens": chars // 4})
    if turns:
        conversations.append(turns)
lengths = [len(c) for c in conversations]
json.dump({"metadata": {"source": "THUDM/AgentInstruct webshop", "conversations": len(conversations),
                        "turns_min_max_mean": [min(lengths), max(lengths), sum(lengths) / len(lengths)]},
           "conversations": conversations}, open(out, "w"))
print(len(conversations), min(lengths), max(lengths), sum(lengths) / len(lengths))
