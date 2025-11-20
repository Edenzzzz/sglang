#!/usr/bin/env python3
"""
Script to count the average input length in SWE-bench training and test sets.
"""

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def calculate_input_lengths(dataset, tokenizer, use_hints=True):
    """
    Calculate input lengths for a dataset using tokenization.

    Args:
        dataset: HuggingFace dataset
        tokenizer: Tokenizer to use for counting tokens
        use_hints: If True, include hints_text in the input length calculation

    Returns:
        List of input lengths (in tokens)
    """
    lengths = []

    for example in dataset:
        # The main input is the problem_statement
        input_text = example.get("problem_statement", "")
        # Optionally include hints_text if it's part of the input
        if use_hints and "hints_text" in example:
            hints = example.get("hints_text", "")
            if hints:
                input_text = input_text + "\n" + hints if input_text else hints

        # Calculate length in tokens (following reference code pattern)
        length = len(tokenizer.encode(input_text, add_special_tokens=False))
        lengths.append(length)

    return lengths


def print_statistics(lengths, split_name):
    """Print statistics for a given split."""
    if not lengths:
        print(f"\n{split_name}: No data found")
        return

    lengths_array = np.array(lengths)

    print(f"\n{split_name} Statistics:")
    print(f"  Number of examples: {len(lengths)}")
    print(f"  Average input length: {np.mean(lengths_array):.2f} tokens")
    print(f"  Median input length: {np.median(lengths_array):.2f} tokens")
    print(f"  Min input length: {np.min(lengths_array)} tokens")
    print(f"  Max input length: {np.max(lengths_array)} tokens")
    print(f"  Std deviation: {np.std(lengths_array):.2f} tokens")
    print(f"  25th percentile: {np.percentile(lengths_array, 25):.2f} tokens")
    print(f"  75th percentile: {np.percentile(lengths_array, 75):.2f} tokens")


def main():
    print("Loading Qwen/Qwen3-8B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

    print("Loading SWE-bench dataset...")

    # Load training set
    print("Loading training set...")
    train_dataset = load_dataset("princeton-nlp/SWE-bench", split="train")

    # Load test set
    print("Loading test set...")
    test_dataset = load_dataset("princeton-nlp/SWE-bench", split="test")

    print("\nCalculating input lengths (in tokens)...")

    # Calculate lengths for training set (with and without hints)
    print("\n" + "=" * 60)
    print("WITHOUT hints_text:")
    print("=" * 60)

    train_lengths_no_hints = calculate_input_lengths(
        train_dataset, tokenizer, use_hints=False
    )
    test_lengths_no_hints = calculate_input_lengths(
        test_dataset, tokenizer, use_hints=False
    )

    print_statistics(train_lengths_no_hints, "Training Set")
    print_statistics(test_lengths_no_hints, "Test Set")

    # Calculate lengths with hints
    print("\n" + "=" * 60)
    print("WITH hints_text:")
    print("=" * 60)

    train_lengths_with_hints = calculate_input_lengths(
        train_dataset, tokenizer, use_hints=True
    )
    test_lengths_with_hints = calculate_input_lengths(
        test_dataset, tokenizer, use_hints=True
    )

    print_statistics(train_lengths_with_hints, "Training Set")
    print_statistics(test_lengths_with_hints, "Test Set")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"Training set average (problem_statement only): {np.mean(train_lengths_no_hints):.2f} tokens"
    )
    print(
        f"Test set average (problem_statement only): {np.mean(test_lengths_no_hints):.2f} tokens"
    )
    print(
        f"Training set average (with hints): {np.mean(train_lengths_with_hints):.2f} tokens"
    )
    print(
        f"Test set average (with hints): {np.mean(test_lengths_with_hints):.2f} tokens"
    )


if __name__ == "__main__":
    main()
