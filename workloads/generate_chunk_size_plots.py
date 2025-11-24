#!/usr/bin/env python3
"""
Script to generate chunk size trend plots from chunk_size_trend.csv.
This script can be used independently to generate plots from existing CSV data.
"""

import argparse
import os
import sys

# Add the parent directory to the path to import sglang
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from sglang.srt.managers.utils import plot_chunk_size_trends


def generate_plots(csv_path: str, output_dir: str = "."):
    """
    Generate chunk size trend plots from CSV data.

    Args:
        csv_path: Path to the chunk_size_trend.csv file
        output_dir: Directory to save the plots (default: current directory)
    """
    success = plot_chunk_size_trends(csv_path, output_dir, use_logger=False)
    if not success:
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
