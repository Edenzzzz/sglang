"""
Third-party dataset samplers for bench_serving.py.

Provides dataset loading functions for real-world agent workloads:
  Code-Feedback, CodeActInstruct, SWE-Gym, SWE-smith, Orca-AgentInstruct

Each sampler returns List[DatasetRow] compatible with bench_serving.py.

Usage (via bench_serving.py redirect):
    python3 -m sglang.bench_serving --backend sglang --dataset-name code-feedback --num-prompts 400
"""

import json
import random
from typing import List, Optional

import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from sglang.bench_serving import DatasetRow

# ============================================================
# Dataset names recognized by this module
# ============================================================

THIRD_PARTY_DATASETS = [
    "code-feedback",
    "codeact",
    "swe-gym",
    "swe-smith",
    "orca-agentinstruct",
]

# ============================================================
# Extractors — pull (input_text, output_text) from each example
# ============================================================


def _extract_messages(ex, conv_field="messages", input_roles=("user", "system"),
                      output_roles=("assistant",), role_key="role", content_key="content"):
    """Extract concatenated input/output from a messages list."""
    msgs = ex.get(conv_field, []) or []
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except (json.JSONDecodeError, TypeError):
            return "", ""
    inp, out = [], []
    for msg in msgs:
        role = msg.get(role_key, "")
        text = msg.get(content_key, "") or ""
        if role in input_roles:
            inp.append(text)
        elif role in output_roles:
            out.append(text)
    return "\n".join(inp), "\n".join(out)


def _extract_code_feedback(ex):
    return _extract_messages(ex, conv_field="messages")


def _extract_codeact(ex):
    return _extract_messages(ex, conv_field="conversations")


def _extract_swe_field_pair(ex):
    return (ex.get("problem_statement", "") or "", ex.get("patch", "") or "")


def _extract_orca(ex):
    return _extract_messages(ex, conv_field="messages")


# ============================================================
# Dataset registry
# ============================================================

_DATASET_CONFIGS = {
    "code-feedback": {
        "hf_id": "m-a-p/Code-Feedback",
        "splits": ["train"],
        "extract": _extract_code_feedback,
    },
    "codeact": {
        "hf_id": "xingyaoww/code-act",
        "splits": ["codeact"],
        "extract": _extract_codeact,
    },
    "swe-gym": {
        "hf_id": "SWE-Gym/SWE-Gym",
        "splits": ["train"],
        "extract": _extract_swe_field_pair,
    },
    "swe-smith": {
        "hf_id": "SWE-bench/SWE-smith",
        "splits": ["train"],
        "extract": _extract_swe_field_pair,
    },
    "orca-agentinstruct": {
        "hf_id": "microsoft/orca-agentinstruct-1M-v1",
        "splits": [
            "creative_content", "text_modification", "struct2text_flow", "rc",
            "rag", "text_extraction", "mcq", "follow_up", "analytical_reasoning",
            "fermi", "fs_cot_flow", "code_", "brain_teaser", "text_classification",
            "open_domain_qa",
        ],
        "extract": _extract_orca,
    },
}


# ============================================================
# Main sampling function
# ============================================================


def sample_3rdparty_requests(
    dataset_name: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    context_len: Optional[int] = None,
    output_len_multiplier: int = 1,
) -> List[DatasetRow]:
    """Load a third-party dataset and return DatasetRow list for benchmarking.

    Args:
        dataset_name: One of THIRD_PARTY_DATASETS.
        num_requests: Max number of requests to return.
        tokenizer: Tokenizer for counting tokens.
        fixed_output_len: If set, override all output lengths with this value.
        context_len: If set, skip examples where input+output > context_len.
        output_len_multiplier: Multiply output length by this factor.
    """
    config = _DATASET_CONFIGS[dataset_name]
    extract = config["extract"]

    # Load all splits and extract text pairs
    pairs = []
    for split_name in config["splits"]:
        try:
            ds = load_dataset(config["hf_id"], split=split_name)
            for ex in ds:
                inp, out = extract(ex)
                if inp:
                    pairs.append((inp, out))
        except Exception as e:
            print(f"  Warning: failed to load {config['hf_id']} split '{split_name}': {e}")

    if not pairs:
        raise ValueError(f"No data loaded for dataset '{dataset_name}'")

    print(f"Loaded {len(pairs)} examples from {config['hf_id']}")

    # Shuffle
    random.shuffle(pairs)

    # Tokenize and filter
    filtered: List[DatasetRow] = []
    input_tokens = []
    output_tokens = []

    for inp_text, out_text in pairs:
        if len(filtered) >= num_requests:
            break

        prompt_ids = tokenizer.encode(inp_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        if fixed_output_len is not None:
            output_len = fixed_output_len
        else:
            out_ids = tokenizer.encode(out_text, add_special_tokens=False)
            output_len = len(out_ids) * output_len_multiplier

        input_tokens.append(prompt_len)
        output_tokens.append(output_len)

        if prompt_len < 2 or output_len < 2:
            continue

        if context_len and prompt_len + output_len > context_len:
            continue

        filtered.append(
            DatasetRow(
                prompt=inp_text,
                prompt_len=prompt_len,
                output_len=output_len,
            )
        )

    if not filtered:
        raise ValueError(
            f"No valid examples after filtering for '{dataset_name}'. "
            f"Try increasing --context-len or using --fixed-output-len."
        )

    print(f"#Sampled requests: {len(filtered)}")
    print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered])}")
    print(f"#Max Input tokens: {np.max(input_tokens[:len(filtered)])}")
    print(f"#Mean Input tokens: {np.mean(input_tokens[:len(filtered)]):.0f}")
    print(f"#Std Input tokens: {np.std(input_tokens[:len(filtered)]):.0f}")
    print(f"#Output tokens: {np.sum([x.output_len for x in filtered])}")
    print(f"#Max Output tokens: {np.max(output_tokens[:len(filtered)])}")
    print(f"#Mean Output tokens: {np.mean(output_tokens[:len(filtered)]):.0f}")
    print(f"#Std Output tokens: {np.std(output_tokens[:len(filtered)]):.0f}")
    return filtered
