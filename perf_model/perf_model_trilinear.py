#!/usr/bin/env python3
"""
Trilinear interpolation inference for POD attention overlap prediction.

This script uses trilinear interpolation based on the 8 closest points
in a 3D grid (r_p, r_d, n_p) to predict execution time.

The grid is built from training data, and for each test point, we find
the 8 corners of the cube containing the point and interpolate.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Add the parent directory to the path to import sglang
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


class TrilinearInterpolator:
    """Trilinear interpolation based on 3D grid of (r_p, r_d, n_p) -> time_ms."""

    def __init__(self, train_df: pd.DataFrame):
        """Initialize interpolator with training data.

        Args:
            train_df: DataFrame with columns [r_p, r_d, n_p, time_ms]
        """
        # Extract training points
        self.train_points = train_df[["r_p", "r_d", "n_p"]].values.astype(np.float64)
        self.train_times = train_df["time_ms"].values.astype(np.float64)

        # Build KD-tree for fast nearest neighbor search
        print(f"Building KD-tree from {len(self.train_points)} training points...")
        self.tree = cKDTree(self.train_points)

        # Compute bounds for each dimension
        self.r_p_min = self.train_points[:, 0].min()
        self.r_p_max = self.train_points[:, 0].max()
        self.r_d_min = self.train_points[:, 1].min()
        self.r_d_max = self.train_points[:, 1].max()
        self.n_p_min = self.train_points[:, 2].min()
        self.n_p_max = self.train_points[:, 2].max()

        print(f"Grid bounds:")
        print(f"  r_p: [{self.r_p_min:.2f}, {self.r_p_max:.2f}]")
        print(f"  r_d: [{self.r_d_min:.2f}, {self.r_d_max:.2f}]")
        print(f"  n_p: [{self.n_p_min:.2f}, {self.n_p_max:.2f}]")

    def _find_cube_corners(
        self, query_point: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find the 8 corners of the cube containing the query point.

        For each query point, we find the 8 nearest neighbors in the training data
        and use them as the corners of the interpolation cube.

        Args:
            query_point: [r_p, r_d, n_p]

        Returns:
            corners: [8, 3] array of corner coordinates
            corner_indices: [8] array of indices into training data
        """
        # Find the 8 nearest neighbors
        k = min(8, len(self.train_points))
        result = self.tree.query(query_point, k=k)

        # Handle different return types from cKDTree.query
        if k == 1:
            # When k=1, returns (distance, index) as scalars
            indices = np.array([result[1]])
        else:
            # When k>1, returns (distances, indices) as arrays
            distances, indices = result

        # Handle case where we have fewer than 8 neighbors
        if k < 8:
            # Pad with the last neighbor
            indices = np.pad(
                indices, (0, 8 - k), mode="edge", constant_values=indices[-1]
            )

        corner_indices = indices[:8]
        corners = self.train_points[corner_indices]

        return corners, corner_indices

    def _trilinear_interpolate(
        self, query_point: np.ndarray, corners: np.ndarray, corner_times: np.ndarray
    ) -> float:
        """Perform trilinear interpolation.

        We use the 8 nearest neighbors as corners and perform trilinear interpolation
        based on the bounding box they form.

        Args:
            query_point: [r_p, r_d, n_p] - point to interpolate
            corners: [8, 3] array of corner coordinates (8 nearest neighbors)
            corner_times: [8] array of time values at corners

        Returns:
            Interpolated time value
        """
        # Find the bounding box of the corners
        r_p_min = corners[:, 0].min()
        r_p_max = corners[:, 0].max()
        r_d_min = corners[:, 1].min()
        r_d_max = corners[:, 1].max()
        n_p_min = corners[:, 2].min()
        n_p_max = corners[:, 2].max()

        # Normalize query point to [0, 1] within the bounding box
        if r_p_max > r_p_min:
            xd = (query_point[0] - r_p_min) / (r_p_max - r_p_min)
        else:
            xd = 0.0

        if r_d_max > r_d_min:
            yd = (query_point[1] - r_d_min) / (r_d_max - r_d_min)
        else:
            yd = 0.0

        if n_p_max > n_p_min:
            zd = (query_point[2] - n_p_min) / (n_p_max - n_p_min)
        else:
            zd = 0.0

        # Clamp to [0, 1]
        xd = np.clip(xd, 0.0, 1.0)
        yd = np.clip(yd, 0.0, 1.0)
        zd = np.clip(zd, 0.0, 1.0)

        # Find which corner is closest to each of the 8 ideal corner positions
        # Ideal corners form a cube: [r_p_min/max, r_d_min/max, n_p_min/max]
        ideal_corners = np.array(
            [
                [r_p_min, r_d_min, n_p_min],  # 000
                [r_p_min, r_d_min, n_p_max],  # 001
                [r_p_min, r_d_max, n_p_min],  # 010
                [r_p_min, r_d_max, n_p_max],  # 011
                [r_p_max, r_d_min, n_p_min],  # 100
                [r_p_max, r_d_min, n_p_max],  # 101
                [r_p_max, r_d_max, n_p_min],  # 110
                [r_p_max, r_d_max, n_p_max],  # 111
            ]
        )

        # Find closest actual corner for each ideal corner position
        ordered_times = np.zeros(8)
        for i, ideal_pos in enumerate(ideal_corners):
            # Find closest corner to this ideal position
            dists = np.linalg.norm(corners - ideal_pos, axis=1)
            closest_idx = np.argmin(dists)
            ordered_times[i] = corner_times[closest_idx]

        # Trilinear interpolation formula
        # Order: c000, c001, c010, c011, c100, c101, c110, c111
        c000, c001, c010, c011, c100, c101, c110, c111 = ordered_times

        # Interpolate along x (r_p dimension)
        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        # Interpolate along y (r_d dimension)
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        # Interpolate along z (n_p dimension)
        result = c0 * (1 - zd) + c1 * zd

        return float(result)

    def predict(self, query_points: np.ndarray) -> np.ndarray:
        """Predict time for query points.

        Args:
            query_points: [N, 3] array of [r_p, r_d, n_p] points

        Returns:
            [N] array of predicted time_ms values
        """
        predictions = []
        for query_point in query_points:
            corners, corner_indices = self._find_cube_corners(query_point)
            corner_times = self.train_times[corner_indices]
            pred = self._trilinear_interpolate(query_point, corners, corner_times)
            predictions.append(pred)

        return np.array(predictions)


def evaluate_trilinear(
    interpolator: TrilinearInterpolator,
    test_df: pd.DataFrame,
) -> Dict[str, float]:
    """Evaluate trilinear interpolator on test data.

    Args:
        interpolator: TrilinearInterpolator instance
        test_df: DataFrame with columns [r_p, r_d, n_p, time_ms]

    Returns:
        Dictionary with metrics: mse, mean_deviation, mean_deviation_pct, max_deviation
    """
    # Extract query points
    query_points = test_df[["r_p", "r_d", "n_p"]].values.astype(np.float64)
    targets = test_df["time_ms"].values.astype(np.float64)

    # Measure inference time
    start_time = time.perf_counter()
    pred = interpolator.predict(query_points)
    end_time = time.perf_counter()
    inference_time = (end_time - start_time) * 1000  # Convert to ms
    avg_inference_time_per_sample = inference_time / len(pred) if len(pred) > 0 else 0.0

    print(
        f"  Average inference time per sample: {avg_inference_time_per_sample:.4f} ms"
    )

    # Calculate metrics
    mse = np.mean((pred - targets) ** 2)
    mean_deviation = np.mean(np.abs(pred - targets))  # Mean absolute error in ms
    max_deviation = np.max(np.abs(pred - targets))  # Max absolute error in ms

    # Mean deviation percentage (MAPE) - data-point-wise percentage
    nonzero_mask = np.abs(targets) > 1e-10
    if np.any(nonzero_mask):
        mean_deviation_pct = (
            np.mean(
                np.abs(
                    (pred[nonzero_mask] - targets[nonzero_mask]) / targets[nonzero_mask]
                )
            )
            * 100
        )
    else:
        mean_deviation_pct = np.nan

    return {
        "mse": mse,
        "mean_deviation": mean_deviation,
        "mean_deviation_pct": mean_deviation_pct,
        "max_deviation": max_deviation,
        "avg_inference_time_per_sample_ms": avg_inference_time_per_sample,
        "predictions": pred,
    }


def main(args):
    """Main function."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Load training data
    print(f"Loading training data from {args.csv}...")
    train_df = pd.read_csv(args.csv)
    print(f"Loaded {len(train_df)} training samples")
    print(f"Columns: {train_df.columns.tolist()}")

    # Filter for non-overlap data if overlap column exists
    if "overlap" in train_df.columns:
        train_df = train_df[train_df["overlap"] == False].copy()
        print(f"Filtered to {len(train_df)} non-overlap training samples")

    # Check required columns
    required_cols = ["r_p", "r_d", "n_p", "time_ms"]
    missing_cols = [col for col in required_cols if col not in train_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"\nTraining data summary:")
    print(train_df[required_cols].describe())
    print()

    # Build interpolator
    print("=" * 60)
    print("Building trilinear interpolator...")
    print("=" * 60)
    interpolator = TrilinearInterpolator(train_df[required_cols])

    # Evaluate on validation data
    if args.val_csv:
        print("\n" + "=" * 60)
        print("Evaluating on non-overlap validation data...")
        print("=" * 60)
        val_df = pd.read_csv(args.val_csv)
        print(f"Loaded {len(val_df)} validation samples")

        # Filter for non-overlap data if overlap column exists
        if "overlap" in val_df.columns:
            val_df = val_df[val_df["overlap"] == False].copy()
            print(f"Filtered to {len(val_df)} non-overlap validation samples")

        # Check required columns
        missing_cols = [col for col in required_cols if col not in val_df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in validation data: {missing_cols}"
            )

        metrics = evaluate_trilinear(interpolator, val_df[required_cols])

        print(f"\nNon-overlap validation metrics:")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  Mean deviation: {metrics['mean_deviation']:.4f} ms")
        print(f"  Mean deviation %: {metrics['mean_deviation_pct']:.4f}%")
        print(f"  Max deviation: {metrics['max_deviation']:.4f} ms")

        # Save metrics to CSV
        metrics_df = pd.DataFrame(
            [
                {
                    "mse": metrics["mse"],
                    "mean_deviation": metrics["mean_deviation"],
                    "mean_deviation_pct": metrics["mean_deviation_pct"],
                    "max_deviation": metrics["max_deviation"],
                    "avg_inference_time_per_sample_ms": metrics[
                        "avg_inference_time_per_sample_ms"
                    ],
                }
            ]
        )
        metrics_csv_path = os.path.join(args.output_dir, "trilinear_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"\nValidation metrics saved to {metrics_csv_path}")

    # Evaluate on overlap validation data if provided
    if args.val_csv_overlap:
        print("\n" + "=" * 60)
        print("Evaluating on overlap validation data...")
        print("=" * 60)
        val_df_overlap = pd.read_csv(args.val_csv_overlap)
        print(f"Loaded {len(val_df_overlap)} overlap validation samples")

        # Filter for overlap data if overlap column exists
        if "overlap" in val_df_overlap.columns:
            val_df_overlap = val_df_overlap[val_df_overlap["overlap"] == True].copy()
            print(f"Filtered to {len(val_df_overlap)} overlap validation samples")

        # Check required columns
        missing_cols = [
            col for col in required_cols if col not in val_df_overlap.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in overlap validation data: {missing_cols}"
            )

        metrics_overlap = evaluate_trilinear(
            interpolator, val_df_overlap[required_cols]
        )

        print(f"\nOverlap validation metrics:")
        print(f"  overlap-mse: {metrics_overlap['mse']:.4f}")
        print(f"  overlap-mean_deviation: {metrics_overlap['mean_deviation']:.4f} ms")
        print(
            f"  overlap-mean_deviation_pct: {metrics_overlap['mean_deviation_pct']:.4f}%"
        )
        print(f"  overlap-max_deviation: {metrics_overlap['max_deviation']:.4f} ms")

        # Save overlap metrics to CSV
        metrics_df_overlap = pd.DataFrame(
            [
                {
                    "overlap-mse": metrics_overlap["mse"],
                    "overlap-mean_deviation": metrics_overlap["mean_deviation"],
                    "overlap-mean_deviation_pct": metrics_overlap["mean_deviation_pct"],
                    "overlap-max_deviation": metrics_overlap["max_deviation"],
                    "overlap-avg_inference_time_per_sample_ms": metrics_overlap[
                        "avg_inference_time_per_sample_ms"
                    ],
                }
            ]
        )
        metrics_csv_path_overlap = os.path.join(
            args.output_dir, "trilinear_metrics_overlap.csv"
        )
        metrics_df_overlap.to_csv(metrics_csv_path_overlap, index=False)
        print(f"\nOverlap validation metrics saved to {metrics_csv_path_overlap}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trilinear interpolation inference for POD attention overlap prediction"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to training CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for results",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default=None,
        help="Path to non-overlap validation CSV file",
    )
    parser.add_argument(
        "--val-csv-overlap",
        type=str,
        default=None,
        help="Path to overlap validation CSV file",
    )

    args = parser.parse_args()
    main(args)
