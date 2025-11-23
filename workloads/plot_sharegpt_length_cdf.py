#!/usr/bin/env python3
"""
Script to plot the CDF distribution of ShareGPT input and output lengths.
"""

import json
import os
from argparse import ArgumentParser
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import requests
from transformers import AutoTokenizer

ASSISTANT_SUFFIX = "Assistant:"
SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


def is_file_valid_json(path: str) -> bool:
    """Check if a file exists and is valid JSON."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, Exception):
        return False


def download_and_cache_file(url: str, filename: Optional[str] = None) -> str:
    """Download and cache a file from a URL."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if is_file_valid_json(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show progress
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1MB chunks

    # Download with progress
    with open(filename, "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end="", flush=True)
    print()  # New line after progress

    return filename


def remove_suffix(text: str, suffix: str) -> str:
    """Remove suffix from text if it exists."""
    return text[: -len(suffix)] if text.endswith(suffix) else text


def process_sharegpt_data(
    dataset_path: str,
    tokenizer,
    num_input_turns: int = 4,
    prompt_suffix: Optional[str] = "",
    apply_chat_template: bool = False,
    context_len: Optional[int] = None,
):
    """
    Process ShareGPT dataset and return input token lengths.

    Returns:
        List of input_lengths
    """
    # Download sharegpt if necessary
    if not is_file_valid_json(dataset_path) and dataset_path == "":
        dataset_path = download_and_cache_file(SHAREGPT_URL)

    # Load the dataset
    print("Loading ShareGPT dataset...")
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Filter out conversations with less than 2 turns
    dataset = [
        data
        for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]

    # Extract input and output from conversations
    processed_dataset = []
    for data in dataset:
        conversations = data.get("conversations", data.get("conversation", []))
        if not conversations:
            continue

        # Use up to num_input_turns - 1 turns for input, or all but the last if fewer turns
        num_input = min(num_input_turns, len(conversations) - 1)
        input_text = "\n".join([conversations[i]["value"] for i in range(num_input)])
        output_text = conversations[num_input]["value"]
        processed_dataset.append((input_text, output_text))

    # Tokenize and calculate lengths
    input_lengths = []

    print(f"Processing {len(processed_dataset)} conversations...")
    for i, (prompt, completion) in enumerate(processed_dataset):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(processed_dataset)} conversations...")

        # Process prompt
        prompt_text = prompt
        if prompt_suffix:
            prompt_text = (
                remove_suffix(prompt_text, ASSISTANT_SUFFIX)
                + prompt_suffix
                + ASSISTANT_SUFFIX
            )

        if apply_chat_template:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                add_generation_prompt=True,
                tokenize=False,
            )
            if tokenizer.bos_token:
                prompt_text = prompt_text.replace(tokenizer.bos_token, "")

        prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        completion_token_ids = tokenizer.encode(completion, add_special_tokens=False)

        prompt_len = len(prompt_token_ids)

        # Filter out too short sequences
        if prompt_len < 2:
            continue

        # Filter out too long sequences if context_len is specified
        if context_len and prompt_len > context_len:
            continue

        input_lengths.append(prompt_len)

    return input_lengths


def plot_cdf_with_percentiles(data: List[float], label: str, color: str, ax):
    """Plot CDF distribution with percentile markers on the given axis."""
    if not data:
        return

    # Calculate percentiles
    p25 = np.percentile(data, 25)
    p50 = np.percentile(data, 50)
    p75 = np.percentile(data, 75)

    # Remove outliers using IQR method for better visualization
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filter data for plotting (but keep all data for percentile calculation)
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]

    if not filtered_data:
        filtered_data = data

    sorted_data = np.sort(filtered_data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # Plot CDF
    ax.plot(sorted_data, y, label=label, color=color, linewidth=2)

    # Add vertical lines for percentiles
    ax.axvline(
        p25,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"25th: {p25:.0f}",
    )
    ax.axvline(
        p50,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"50th: {p50:.0f}",
    )
    ax.axvline(
        p75,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"75th: {p75:.0f}",
    )

    ax.set_xlabel("Length (tokens)", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)


def main():
    parser = ArgumentParser()
    parser.add_argument("--num-input-turns", type=int, default=138)
    args = parser.parse_args()
    # Default settings
    dataset_path = ""
    tokenizer_name = "Qwen/Qwen3-8B"
    prompt_suffix = ""
    apply_chat_template = False
    context_len = None
    output_file = "sharegpt_length_cdf.png"

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Process ShareGPT data
    input_lengths = process_sharegpt_data(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        num_input_turns=args.num_input_turns,
        prompt_suffix=prompt_suffix,
        apply_chat_template=apply_chat_template,
        context_len=context_len,
    )

    if not input_lengths:
        print("Error: No valid data processed!")
        return

    # Print statistics
    print("\n" + "=" * 60)
    print(f"Statistics (up to {args.num_input_turns} turns):")
    print("=" * 60)
    print(f"Number of samples: {len(input_lengths)}")
    print(f"\nInput Length Statistics:")
    print(f"  Mean: {np.mean(input_lengths):.2f} tokens")
    print(f"  Median: {np.median(input_lengths):.2f} tokens")
    print(f"  Min: {np.min(input_lengths)} tokens")
    print(f"  Max: {np.max(input_lengths)} tokens")
    print(f"  Std: {np.std(input_lengths):.2f} tokens")
    print(f"  25th percentile: {np.percentile(input_lengths, 25):.2f} tokens")
    print(f"  75th percentile: {np.percentile(input_lengths, 75):.2f} tokens")
    print(f"  90th percentile: {np.percentile(input_lengths, 90):.2f} tokens")
    print(f"  95th percentile: {np.percentile(input_lengths, 95):.2f} tokens")
    print(f"  99th percentile: {np.percentile(input_lengths, 99):.2f} tokens")

    # Create CDF plot (only input length)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot input length CDF with percentiles
    plot_cdf_with_percentiles(input_lengths, "Input Length", "tab:blue", ax)
    ax.set_title(
        f"ShareGPT Input Length CDF (up to {args.num_input_turns} turns)", fontsize=14
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nCDF plot saved to {output_file}")


if __name__ == "__main__":
    main()
