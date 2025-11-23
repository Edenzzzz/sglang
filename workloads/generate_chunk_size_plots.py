#!/usr/bin/env python3
"""
Script to generate chunk size trend plots from chunk_size_trend.csv.
This script can be used independently to generate plots from existing CSV data.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def generate_plots(csv_path: str, output_dir: str = "."):
    """
    Generate chunk size trend plots from CSV data.

    Args:
        csv_path: Path to the chunk_size_trend.csv file
        output_dir: Directory to save the plots (default: current directory)
    """
    # Load data from CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty: {csv_path}")
        sys.exit(1)

    if df.empty:
        print("Error: CSV file contains no data")
        sys.exit(1)

    # Separate prefill and decode data
    prefill_data = df[df["prefill_tokens"].notna()].copy()
    decode_data = df[df["decode_bs"].notna()].copy()

    # Plot 1: Prefill tokens and new token ratio (dual y-axis)
    if not prefill_data.empty:
        prefill_times = prefill_data["time"].tolist()
        prefill_tokens = prefill_data["prefill_tokens"].tolist()
        new_token_ratios = prefill_data["new_token_ratio"].tolist()

        # Calculate mean prefill length
        mean_prefill_len = prefill_data["prefill_tokens"].mean()

        fig1, ax1 = plt.subplots(figsize=(12, 6))

        # Left y-axis for prefill tokens
        color1 = "tab:blue"
        ax1.set_xlabel("Time (seconds)", fontsize=12)
        ax1.set_ylabel("Prefill Tokens", fontsize=12, color=color1)
        line1 = ax1.plot(
            prefill_times,
            prefill_tokens,
            color=color1,
            linewidth=1.5,
            label="Prefill Tokens",
        )
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        # Right y-axis for new token ratio
        ax1_twin = ax1.twinx()
        color2 = "tab:red"
        ax1_twin.set_ylabel("New(decode) Token Ratio", fontsize=12, color=color2)
        line2 = ax1_twin.plot(
            prefill_times,
            new_token_ratios,
            color=color2,
            linewidth=1.5,
            label="New Token Ratio",
        )
        ax1_twin.tick_params(axis="y", labelcolor=color2)

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left")
        ax1.set_title("Prefill Tokens and New Token Ratio Over Time", fontsize=14)

        plt.tight_layout()
        filename = os.path.join(
            output_dir, f"chunk_size_trend_prefill_mean_{mean_prefill_len:.1f}.png"
        )
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig1)
        print(f"Prefill chunk size trend plot saved to {filename}")

    # Plot 2: Decode batch size and average KV length (dual y-axis)
    if not decode_data.empty:
        decode_times = decode_data["time"].tolist()
        decode_bs = decode_data["decode_bs"].tolist()
        avg_kv_lengths = decode_data["avg_kv_length"].tolist()

        # Calculate mean decode KV length
        mean_decode_kv_len = decode_data["avg_kv_length"].mean()

        fig2, ax2 = plt.subplots(figsize=(12, 6))

        # Left y-axis for decode batch size
        color1 = "tab:blue"
        ax2.set_xlabel("Time (seconds)", fontsize=12)
        ax2.set_ylabel("Decode Batch Size", fontsize=12, color=color1)
        line1 = ax2.plot(
            decode_times,
            decode_bs,
            color=color1,
            linewidth=1.5,
            label="Decode Batch Size",
        )
        ax2.tick_params(axis="y", labelcolor=color1)
        ax2.grid(True, alpha=0.3)

        # Right y-axis for average KV length
        ax2_twin = ax2.twinx()
        color2 = "tab:red"
        ax2_twin.set_ylabel("Average KV Length", fontsize=12, color=color2)
        line2 = ax2_twin.plot(
            decode_times,
            avg_kv_lengths,
            color=color2,
            linewidth=1.5,
            label="Average KV Length",
        )
        ax2_twin.tick_params(axis="y", labelcolor=color2)

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc="upper left")
        ax2.set_title("Decode Batch Size and Average KV Length Over Time", fontsize=14)

        plt.tight_layout()
        filename = os.path.join(
            output_dir,
            f"chunk_size_trend_decode_mean_{mean_decode_kv_len:.1f}.png",
        )
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Decode chunk size trend plot saved to {filename}")

    if prefill_data.empty and decode_data.empty:
        print("Warning: No valid data found in CSV file")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate chunk size trend plots from CSV data"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="chunk_size_trend.csv",
        help="Path to chunk_size_trend.csv file (default: chunk_size_trend.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save the plots (default: current directory)",
    )

    args = parser.parse_args()

    # Check if CSV file exists
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        print(f"Please provide a valid path to chunk_size_trend.csv")
        sys.exit(1)

    # Create output directory if it doesn't exist
    if args.output_dir != "." and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    print(f"Generating plots from {args.csv}...")
    generate_plots(args.csv, args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
