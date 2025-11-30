#!/usr/bin/env python3
"""
Example usage:
python /workloads/train_regression_model.py --csv /path/to/profiling.csv --val-csv /path/to/validation.csv --csv-overlap /path/to/overlap.csv --val-csv-overlap /path/to/overlap-validation.csv

Train performance model for POD attention overlap prediction.

This script trains a regression model to predict execution time for mixed
prefill+decode batches based on sequence lengths and batch sizes.

Model (without overlap speedup):
  T = θ₁ₚ * sum(n_{i,p}²)
    + θ₂ₚ * sum(n_{i,p} * r_{i,p})
    + θ₃ * sum(n_{i,p} + r_{i,p} + r_{i,d})
    + θ₄ₐ * sum(r_{i,d})
    + β

Where:
  - n_p: new prefill tokens (extend_input_len)
  - r_p: prefill prefix length
  - r_d: decode KV length
  - decode_bs: decode batch size
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add the parent directory to the path to import sglang
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


class PerformanceModel(nn.Module):
    """Performance model for predicting execution time of mixed prefill+decode batches.

    Without overlap (Speedup = 1.0):
      T = θ₁ₚ * term1 + θ₂ₚ * term2 + θ₃ * term3 + θ₄ₐ * term4 + θ₅ₐ * term5 + β

    With overlap (Speedup = α₁Rₚd + α₂Rₚd² + α₃):
      T = θ₁ₚ * term1 + Speedup * θ₂ₚ * term2 + θ₃ * term3 + Speedup * θ₄ₐ * term4 + θ₅ₐ * term5 + β

    Where:
      term1 = sum(n_{i,p}²) = n_p²
      term2 = sum(n_{i,p} * r_{i,p}) = n_p * r_p
      term3 = sum(n_{i,p} + r_{i,p}) = n_p + r_p
      term4 = sum(r_{i,d}) = r_d * decode_bs
      term5 = sum(r_{i,d}²) = r_d² * decode_bs
      Rₚd = S_d / (S_p + S_d)
      S_p = sum(r_p)
      S_d = sum(r_d)
    """

    def __init__(self, pred_log: bool = False, standardize: bool = True):
        super().__init__()
        # Model parameters
        self.theta_1p = nn.Parameter(
            torch.randn(1, dtype=torch.float32)
        )  # n_p² coefficient
        self.theta_2p = nn.Parameter(
            torch.randn(1, dtype=torch.float32)
        )  # n_p * r_p coefficient
        self.theta_3 = nn.Parameter(
            torch.randn(1, dtype=torch.float32)
        )  # (n_p + r_p) coefficient
        self.theta_4d = nn.Parameter(
            torch.randn(1, dtype=torch.float32)
        )  # r_d * decode_bs coefficient
        self.theta_5d = nn.Parameter(
            torch.randn(1, dtype=torch.float32)
        )  # r_d² coefficient
        self.beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))  # bias term
        # Speedup parameters (umbrella-like quadratic curve)
        self.alpha_1 = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )  # Rₚd coefficient
        self.alpha_2 = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )  # Rₚd² coefficient
        self.alpha_3 = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )  # constant term

        # Feature normalization stats (computed from non-overlap training set).
        # These are per-feature stats for [term1, term2, term3, term4, term5].
        self.register_buffer("feat_mean", torch.zeros(5, dtype=torch.float32))
        self.register_buffer("feat_std", torch.ones(5, dtype=torch.float32))
        # Whether to predict in log space and apply exp() during inference
        self.pred_log = pred_log
        # Whether to standardize features
        self.standardize = standardize

    def forward(self, term1, term2, term3, term4, term5, overlap, R_pd):
        """Forward pass.

        Args:
            term1: n_p²
            term2: n_p * r_p
            term3: n_p + r_p
            term4: r_d * decode_bs
            term5: r_d²
            overlap: Boolean indicating whether overlap is enabled for this forward pass
            R_pd: Rₚd = S_d / (S_p + S_d) for speedup calculation

        Returns:
            Predicted time in milliseconds
        """
        # Compute speedup based on overlap flag for the whole forward pass
        if overlap:
            # Speedup = α₁Rₚd + α₂Rₚd² + α₃ (computed per sample)
            speedup = self.alpha_1 * R_pd + self.alpha_2 * R_pd**2 + self.alpha_3
        else:
            # No overlap: speedup = 1.0
            speedup = 1.0

        # During training, we assume inputs are already standardized if standardize=True.
        # During eval/inference, normalize using stored training-set stats if standardize=True.
        if self.standardize and not self.training:
            features = torch.stack([term1, term2, term3, term4, term5], dim=0)
            # Reshape stats to broadcast over batch dimension
            mean = self.feat_mean.view(-1, 1)
            std = self.feat_std.view(-1, 1)
            features = (features - mean) / std

            term1, term2, term3, term4, term5 = features

        pred = (
            self.theta_1p * term1
            + speedup * self.theta_2p * term2
            + self.theta_3 * term3
            + speedup * self.theta_4d * term4
            + self.theta_5d * term5
            + self.beta
        )
        # If pred_log is True and not training, apply exp() to convert from log space to time space
        if self.pred_log and not self.training:
            pred = pred.exp()
        return pred

    def get_params(self):
        """Get model parameters as a dictionary."""
        return {
            "theta_1p": self.theta_1p.item(),
            "theta_2p": self.theta_2p.item(),
            "theta_3": self.theta_3.item(),
            "theta_4d": self.theta_4d.item(),
            "theta_5d": self.theta_5d.item(),
            "beta": self.beta.item(),
            "alpha_1": self.alpha_1.item(),
            "alpha_2": self.alpha_2.item(),
            "alpha_3": self.alpha_3.item(),
            "feat_mean_term1": float(self.feat_mean[0].item()),
            "feat_mean_term2": float(self.feat_mean[1].item()),
            "feat_mean_term3": float(self.feat_mean[2].item()),
            "feat_mean_term4": float(self.feat_mean[3].item()),
            "feat_mean_term5": float(self.feat_mean[4].item()),
            "feat_std_term1": float(self.feat_std[0].item()),
            "feat_std_term2": float(self.feat_std[1].item()),
            "feat_std_term3": float(self.feat_std[2].item()),
            "feat_std_term4": float(self.feat_std[3].item()),
            "feat_std_term5": float(self.feat_std[4].item()),
        }


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute model features from raw profiling data.

    Args:
        df: DataFrame with columns [n_p, r_p, r_d, decode_bs, time_ms, overlap]

    Returns:
        DataFrame with added feature columns [term1, term2, term3, term4, term5, R_pd] and preserved overlap
    """
    df = df.copy()

    # Ensure overlap column exists (default to False if missing)
    if "overlap" not in df.columns:
        df["overlap"] = False

    # Compute model terms
    df["term1"] = df["n_p"] ** 2
    df["term2"] = df["n_p"] * df["r_p"]
    df["term3"] = df["n_p"] + df["r_p"]
    df["term4"] = df["r_d"] * df["decode_bs"]
    df["term5"] = (df["r_d"] ** 2) * df["decode_bs"]

    # Compute R_pd = S_d / (S_p + S_d)
    # For single sample: S_p = r_p, S_d = r_d * decode_bs
    S_p = df["r_p"]
    S_d = df["r_d"] * df["decode_bs"]
    df["R_pd"] = S_d / (
        S_p + S_d + 1e-10
    )  # Add small epsilon to avoid division by zero

    return df


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    overlap: bool = False,
    model: Optional[PerformanceModel] = None,
    num_epochs: int = 1000,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    pred_log: bool = False,
    standardize: bool = True,
):
    """Train the performance model.

    Args:
        train_df: Training data with columns [term1, term2, term3, term4, R_pd, time_ms]
        val_df: Validation data with columns [term1, term2, term3, term4, R_pd, time_ms]
        overlap: Boolean indicating whether to use overlap mode (True) or non-overlap mode (False)
        model: Optional pre-trained model. If None, creates a new model and trains all parameters.
               If provided, freezes base parameters and only trains speedup parameters (alpha_1, alpha_2, alpha_3).
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size for training
        device: Device to train on
        pred_log: If True, predict log(time) during training

    Returns:
        Trained model and training history
    """
    # Create new model or use existing one
    if model is None:
        model = PerformanceModel(pred_log=pred_log, standardize=standardize).to(device)
        train_main_params = True
    else:
        # Freeze original parameters (theta_1p, theta_2p, theta_3, theta_4d, theta_5d, beta)
        model.theta_1p.requires_grad = False
        model.theta_2p.requires_grad = False
        model.theta_3.requires_grad = False
        model.theta_4d.requires_grad = False
        model.theta_5d.requires_grad = False
        model.beta.requires_grad = False
        train_main_params = False

    # Prepare data
    train_features_raw = torch.tensor(
        train_df[["term1", "term2", "term3", "term4", "term5"]].values,
        dtype=torch.float32,
    ).to(device)
    train_R_pd = torch.tensor(train_df["R_pd"].values, dtype=torch.float32).to(device)
    # Use log(time_ms) as targets if pred_log is True
    if pred_log:
        train_targets = torch.log(
            torch.tensor(train_df["time_ms"].values, dtype=torch.float32) + 1e-8
        ).to(device)
    else:
        train_targets = torch.tensor(
            train_df["time_ms"].values, dtype=torch.float32
        ).to(device)

    val_features = torch.tensor(
        val_df[["term1", "term2", "term3", "term4", "term5"]].values,
        dtype=torch.float32,
    ).to(device)
    val_R_pd = torch.tensor(val_df["R_pd"].values, dtype=torch.float32).to(device)
    val_targets_raw = torch.tensor(val_df["time_ms"].values, dtype=torch.float32).to(
        device
    )
    # For validation, we need raw time_ms for computing metrics after exp()
    if pred_log:
        val_targets = torch.log(val_targets_raw + 1e-8)
    else:
        val_targets = val_targets_raw

    # Feature standardization
    if standardize:
        if train_main_params:
            # Compute stats from training set
            feat_mean = train_features_raw.mean(dim=0)
            feat_std = train_features_raw.std(dim=0)
            # Avoid extremely small std which would explode gradients / blow up normalization
            eps = 1e-6
            feat_std = torch.where(feat_std < eps, torch.ones_like(feat_std), feat_std)
            # Store stats in the model so that forward() can normalize at eval time
            with torch.no_grad():
                model.feat_mean.copy_(feat_mean)
                model.feat_std.copy_(feat_std)
        else:
            # Use existing stats from the model
            feat_mean = model.feat_mean.to(device)
            feat_std = model.feat_std.to(device)

        # Standardize train features
        train_features = (train_features_raw - feat_mean) / feat_std
    else:
        # No standardization: use raw features
        train_features = train_features_raw

    # -----------------------------
    # Closed-form least squares
    # -----------------------------
    if train_main_params:
        # Solve for theta_1p, theta_2p, theta_3, theta_4d, theta_5d, beta
        # Design matrix for standardized features (no intercept for features, bias is separate)
        # X has shape [N, 6]: [term1_n, term2_n, term3_n, term4_n, term5_n, 1]
        X = train_features.to(torch.float64)
        ones = torch.ones((X.shape[0], 1), dtype=torch.float64, device=device)
        X = torch.cat([X, ones], dim=1)
        y = train_targets.to(torch.float64).unsqueeze(1)

        # Solve X w ≈ y in least-squares sense
        solution = torch.linalg.lstsq(X, y)
        w = solution.solution.squeeze(1)  # shape [6]

        # Assign parameters (note θ's operate on standardized features)
        with torch.no_grad():
            model.theta_1p.copy_(w[0].to(torch.float32).unsqueeze(0))
            model.theta_2p.copy_(w[1].to(torch.float32).unsqueeze(0))
            model.theta_3.copy_(w[2].to(torch.float32).unsqueeze(0))
            model.theta_4d.copy_(w[3].to(torch.float32).unsqueeze(0))
            model.theta_5d.copy_(w[4].to(torch.float32).unsqueeze(0))
            model.beta.copy_(w[5].to(torch.float32))
    else:
        # Solve for overlap speedup parameters α₁, α₂, α₃ only
        # Use current θ parameters (already fitted on non-overlap) and solve for α's.
        # Model for overlap:
        #   T = θ1 * term1_n + θ3 * term3_n + θ5 * term5_n + β
        #       + α1 * (θ2 * term2_n * R + θ4 * term4_n * R)
        #       + α2 * (θ2 * term2_n * R^2 + θ4 * term4_n * R^2)
        #       + α3 * (θ2 * term2_n + θ4 * term4_n)
        #
        # This is linear in [α1, α2, α3].
        theta_1 = model.theta_1p.to(torch.float64)
        theta_2 = model.theta_2p.to(torch.float64)
        theta_3 = model.theta_3.to(torch.float64)
        theta_4 = model.theta_4d.to(torch.float64)
        theta_5 = model.theta_5d.to(torch.float64)
        beta = model.beta.to(torch.float64)

        # Features for overlap training (standardize if enabled)
        if standardize:
            feat_mean_f64 = model.feat_mean.to(device).to(torch.float64)
            feat_std_f64 = model.feat_std.to(device).to(torch.float64)
            train_features_std = (
                train_features_raw.to(torch.float64) - feat_mean_f64
            ) / feat_std_f64
        else:
            train_features_std = train_features_raw.to(torch.float64)

        term1_n, term2_n, term3_n, term4_n, term5_n = train_features_std.T
        R = train_R_pd.to(torch.float64)
        y = train_targets.to(torch.float64)

        base = (
            theta_1 * term1_n + theta_3 * term3_n + theta_5 * term5_n + beta
        )  # base contribution without overlap speedup

        A1 = theta_2 * term2_n * R + theta_4 * term4_n * R
        A2 = theta_2 * term2_n * R**2 + theta_4 * term4_n * R**2
        A3 = theta_2 * term2_n + theta_4 * term4_n

        # Solve for α's in least-squares sense: [A1 A2 A3] * alpha ≈ (y - base)
        X = torch.stack([A1, A2, A3], dim=1)  # [N, 3]
        target = (y - base).unsqueeze(1)  # [N, 1]

        solution = torch.linalg.lstsq(X, target)
        alpha = solution.solution.squeeze(1)  # [3]

        with torch.no_grad():
            model.alpha_1.copy_(alpha[0].to(torch.float32))
            model.alpha_2.copy_(alpha[1].to(torch.float32))
            model.alpha_3.copy_(alpha[2].to(torch.float32))

    # Compute training and validation losses for logging
    model.eval()
    with torch.no_grad():
        # Training loss
        if train_main_params:
            term1_t, term2_t, term3_t, term4_t, term5_t = train_features.T
        else:
            # For overlap, use standardized features
            term1_t, term2_t, term3_t, term4_t, term5_t = train_features_std.T.to(
                torch.float32
            )
        pred_train = model(
            term1_t, term2_t, term3_t, term4_t, term5_t, overlap, train_R_pd
        )

        if pred_log:
            # Model outputs exp(log(time)) = time, compare with exp(log(targets))
            train_targets_exp = torch.exp(train_targets)
            train_mse = torch.mean((pred_train - train_targets_exp) ** 2).item()
        else:
            train_mse = torch.mean((pred_train - train_targets) ** 2).item()

        # Validation loss
        term1_v, term2_v, term3_v, term4_v, term5_v = val_features.T
        pred_val = model(term1_v, term2_v, term3_v, term4_v, term5_v, overlap, val_R_pd)
        if pred_log:
            # Model outputs exp(log(time)) = time, compare with raw targets
            val_mse = torch.mean((pred_val - val_targets_raw) ** 2).item()
        else:
            val_mse = torch.mean((pred_val - val_targets) ** 2).item()

    history = {
        "train_loss": [train_mse],
        "val_loss": [val_mse],
    }

    mode_str = "overlap" if overlap else "non-overlap"
    print(f"Fitted {mode_str} parameters (least squares): {model.get_params()}")
    print(f"{mode_str.capitalize()} train MSE: {train_mse:.4f}")
    print(f"{mode_str.capitalize()} val MSE:   {val_mse:.4f}")

    return model, history


def evaluate_model(
    model: nn.Module, df: pd.DataFrame, device: str = "cpu", pred_log: bool = False
):
    """Evaluate model on a dataset.

    Args:
        model: Trained model
        df: DataFrame with columns [term1, term2, term3, term4, R_pd, overlap, time_ms]
        device: Device to evaluate on
        pred_log: If True, model predicts log(time), so apply exp() to predictions

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    # Split into non-overlap and overlap
    df_no_overlap = df[df["overlap"] == False].copy()
    df_overlap = df[df["overlap"] == True].copy()

    preds = []
    targets_list = []

    with torch.no_grad():
        # Non-overlap evaluation
        if len(df_no_overlap) > 0:
            features_no = torch.tensor(
                df_no_overlap[["term1", "term2", "term3", "term4", "term5"]].values,
                dtype=torch.float32,
            ).to(device)
            R_pd_no = torch.tensor(
                df_no_overlap["R_pd"].values, dtype=torch.float32
            ).to(device)
            term1, term2, term3, term4, term5 = features_no.T
            # Model handles exp() internally when pred_log=True and not training
            pred_no = (
                model(term1, term2, term3, term4, term5, False, R_pd_no).cpu().numpy()
            )
            preds.append(pred_no)
            targets_list.append(df_no_overlap["time_ms"].values)

        # Overlap evaluation
        if len(df_overlap) > 0:
            features_ov = torch.tensor(
                df_overlap[["term1", "term2", "term3", "term4", "term5"]].values,
                dtype=torch.float32,
            ).to(device)
            R_pd_ov = torch.tensor(df_overlap["R_pd"].values, dtype=torch.float32).to(
                device
            )
            term1, term2, term3, term4, term5 = features_ov.T
            # Model handles exp() internally when pred_log=True and not training
            pred_ov = (
                model(term1, term2, term3, term4, term5, True, R_pd_ov).cpu().numpy()
            )
            preds.append(pred_ov)
            targets_list.append(df_overlap["time_ms"].values)

        pred = np.concatenate(preds) if len(preds) > 0 else np.array([])
        targets = (
            np.concatenate(targets_list) if len(targets_list) > 0 else np.array([])
        )

    mse = np.mean((pred - targets) ** 2)
    mean_deviation = np.mean(np.abs(pred - targets))  # Mean absolute error in ms
    max_deviation = np.max(np.abs(pred - targets))  # Max absolute error in ms

    # Mean deviation percentage (MAPE) - data-point-wise percentage # noqa
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
        "predictions": pred,
    }


def get_model_filename(base_name: str, pred_log: bool, standardize: bool) -> str:
    suffixes = []
    if standardize:
        suffixes.append("_standardize")
    if pred_log:
        suffixes.append("_log")
    suffix = "".join(suffixes)
    return f"{base_name}{suffix}"


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    history = {"train_loss": [], "val_loss": []}
    model: PerformanceModel

    # -----------------------------
    # Training / fitting (least squares)
    # -----------------------------
    if args.training:
        # Load non-overlap data
        print(f"Loading non-overlap data from {args.csv}...")
        df_no_overlap = pd.read_csv(args.csv)
        # Filter for non-overlap data if overlap column exists
        if "overlap" in df_no_overlap.columns:
            df_no_overlap = df_no_overlap[df_no_overlap["overlap"] == False].copy()
        print(f"Loaded {len(df_no_overlap)} non-overlap samples")
        print(f"Columns: {df_no_overlap.columns.tolist()}")
        print(f"\nNon-overlap data summary:")
        print(df_no_overlap.describe())
        print()

        # Compute features for non-overlap data
        print("Computing features for non-overlap data...")
        train_df = compute_features(df_no_overlap)
        print(f"Features: {train_df[['term1', 'term2', 'term3', 'term4']].describe()}")
        print()

        # Load validation data if provided
        val_df = None
        if args.val_csv:
            print(f"Loading non-overlap validation data from {args.val_csv}...")
            val_df = pd.read_csv(args.val_csv)
            val_df = compute_features(val_df)
            print(f"Non-overlap validation: {len(val_df)} samples")
            print()
        else:
            # Create dummy validation set for logging
            val_df = (
                train_df.iloc[:100].copy() if len(train_df) > 100 else train_df.copy()
            )

        # Fit model on non-overlap data (least squares)
        print("=" * 60)
        print("Fitting on non-overlap data (least squares)...")
        print("=" * 60)
        model, history = train_model(
            train_df,
            val_df,
            overlap=False,
            model=None,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            device=device,
            pred_log=args.pred_log,
            standardize=args.standardize,
        )

        # If overlap CSV is provided, fit overlap speedup parameters
        if args.csv_overlap:
            print("\n" + "=" * 60)
            print(
                "Fitting overlap speedup parameters (least squares, freezing original params)..."
            )
            print("=" * 60)

            print(f"Loading overlap data from {args.csv_overlap}...")
            df_overlap = pd.read_csv(args.csv_overlap)
            # Filter for overlap data if overlap column exists
            if "overlap" in df_overlap.columns:
                df_overlap = df_overlap[df_overlap["overlap"] == True].copy()
            print(f"Loaded {len(df_overlap)} overlap samples")
            print(f"\nOverlap data summary:")
            print(df_overlap.describe())
            print()

            # Compute features for overlap data
            print("Computing features for overlap data...")
            train_df_overlap = compute_features(df_overlap)
            print(
                f"Features: {train_df_overlap[['term1', 'term2', 'term3', 'term4']].describe()}"
            )
            print()

            # Load overlap validation data if provided
            val_df_overlap = None
            if args.val_csv_overlap:
                print(f"Loading overlap validation data from {args.val_csv_overlap}...")
                val_df_overlap = pd.read_csv(args.val_csv_overlap)
                val_df_overlap = compute_features(val_df_overlap)
                print(f"Overlap validation: {len(val_df_overlap)} samples")
                print()
            else:
                # Create dummy validation set for logging
                val_df_overlap = (
                    train_df_overlap.iloc[:100].copy()
                    if len(train_df_overlap) > 100
                    else train_df_overlap.copy()
                )

            # Fit overlap speedup parameters (only α's) using least squares
            model, history_overlap = train_model(
                train_df_overlap,
                val_df_overlap,
                overlap=True,
                model=model,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                device=device,
                pred_log=args.pred_log,
                standardize=args.standardize,
            )
    else:
        # Evaluation-only mode: load existing model
        # Try to find model file with appropriate suffixes
        model_filename = get_model_filename(
            "regression_model", args.pred_log, args.standardize
        )
        model_path = os.path.join(args.output_dir, f"{model_filename}.pt")

        # If file doesn't exist with suffixes, try base name
        if not os.path.exists(model_path):
            model_path = os.path.join(args.output_dir, f"regression_model.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No existing model found at {model_path}. "
                "Run with --training first to fit and save the model."
            )
        print(f"Loading existing model from {model_path} for evaluation...")
        checkpoint = torch.load(model_path, map_location=device)
        # Use pred_log and standardize from checkpoint if available, otherwise use args
        pred_log_loaded = checkpoint.get("pred_log", args.pred_log)
        standardize_loaded = checkpoint.get("standardize", args.standardize)
        model = PerformanceModel(
            pred_log=pred_log_loaded, standardize=standardize_loaded
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        history = checkpoint.get("history", {"train_loss": [], "val_loss": []})

    # Evaluate on validation set if provided
    val_metrics = {}
    if args.val_csv or args.val_csv_overlap:
        print("\nEvaluating on validation set...")

        # Evaluate on non-overlap validation data
        if args.val_csv:
            print("Evaluating on non-overlap validation data...")
            val_df_no_overlap = pd.read_csv(args.val_csv)
            val_df_no_overlap = compute_features(val_df_no_overlap)
            val_metrics_no_overlap = evaluate_model(
                model, val_df_no_overlap, device=device, pred_log=args.pred_log
            )
            print(f"Non-overlap validation metrics:")
            print(f"  MSE:           {val_metrics_no_overlap['mse']:.4f}")
            print(
                f"  Mean deviation: {val_metrics_no_overlap['mean_deviation']:.4f} ms"
            )
            print(
                f"  Mean deviation %: {val_metrics_no_overlap['mean_deviation_pct']:.4f}%"
            )
            print(f"  Max deviation:  {val_metrics_no_overlap['max_deviation']:.4f} ms")
            print()
            # Store with non-overlap prefix
            val_metrics["non-overlap-mse"] = val_metrics_no_overlap["mse"]
            val_metrics["non-overlap-mean_deviation"] = val_metrics_no_overlap[
                "mean_deviation"
            ]
            val_metrics["non-overlap-mean_deviation_pct"] = val_metrics_no_overlap[
                "mean_deviation_pct"
            ]
            val_metrics["non-overlap-max_deviation"] = val_metrics_no_overlap[
                "max_deviation"
            ]

        # Evaluate on overlap validation data
        if args.val_csv_overlap:
            print("Evaluating on overlap validation data...")
            val_df_overlap = pd.read_csv(args.val_csv_overlap)
            val_df_overlap = compute_features(val_df_overlap)
            val_metrics_overlap = evaluate_model(
                model, val_df_overlap, device=device, pred_log=args.pred_log
            )
            print(f"Overlap validation metrics:")
            print(f"  MSE:           {val_metrics_overlap['mse']:.4f}")
            print(f"  Mean deviation: {val_metrics_overlap['mean_deviation']:.4f} ms")
            print(
                f"  Mean deviation %: {val_metrics_overlap['mean_deviation_pct']:.4f}%"
            )
            print(f"  Max deviation:  {val_metrics_overlap['max_deviation']:.4f} ms")
            print()
            # Store with overlap prefix
            val_metrics["overlap-mse"] = val_metrics_overlap["mse"]
            val_metrics["overlap-mean_deviation"] = val_metrics_overlap[
                "mean_deviation"
            ]
            val_metrics["overlap-mean_deviation_pct"] = val_metrics_overlap[
                "mean_deviation_pct"
            ]
            val_metrics["overlap-max_deviation"] = val_metrics_overlap["max_deviation"]

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_filename = get_model_filename(
        "regression_model", args.pred_log, args.standardize
    )
    model_path = os.path.join(args.output_dir, f"{model_filename}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "params": model.get_params(),
            "history": history,
            "val_metrics": val_metrics,
            "pred_log": args.pred_log,
            "standardize": args.standardize,
        },
        model_path,
    )
    print(f"Model saved to {model_path}")

    # Save parameters to text file
    params_filename = get_model_filename(
        "regression_model_params", args.pred_log, args.standardize
    )
    params_path = os.path.join(args.output_dir, f"{params_filename}.txt")
    with open(params_path, "w") as f:
        if args.csv_overlap:
            f.write(
                "Performance Model Parameters (trained on non-overlap + overlap data)\n"
            )
        else:
            f.write("Performance Model Parameters (trained on non-overlap data only)\n")
        f.write("=" * 60 + "\n\n")
        params = model.get_params()
        for key, value in params.items():
            f.write(f"{key}: {value:.10e}\n")
        f.write("\n")
        f.write("Validation Metrics:\n")
    print(f"Parameters saved to {params_path}")

    # Save validation metrics as CSV
    if val_metrics:
        metrics_filename = get_model_filename(
            "validation_metrics", args.pred_log, args.standardize
        )
        metrics_csv_path = os.path.join(args.output_dir, f"{metrics_filename}.csv")
        metrics_rows = []
        if "non-overlap-mse" in val_metrics:
            metrics_rows.append(
                {
                    "type": "non-overlap",
                    "mse": val_metrics["non-overlap-mse"],
                    "mean_deviation_ms": val_metrics["non-overlap-mean_deviation"],
                    "mean_deviation_pct": val_metrics["non-overlap-mean_deviation_pct"],
                    "max_deviation_ms": val_metrics["non-overlap-max_deviation"],
                }
            )
        if "overlap-mse" in val_metrics:
            metrics_rows.append(
                {
                    "type": "overlap",
                    "mse": val_metrics["overlap-mse"],
                    "mean_deviation_ms": val_metrics["overlap-mean_deviation"],
                    "mean_deviation_pct": val_metrics["overlap-mean_deviation_pct"],
                    "max_deviation_ms": val_metrics["overlap-max_deviation"],
                }
            )

        if metrics_rows:
            metrics_df = pd.DataFrame(metrics_rows)
            metrics_df.to_csv(metrics_csv_path, index=False)
            print(f"Validation metrics saved to {metrics_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train performance model for POD attention overlap"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to profiling CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for model and results",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default=None,
        help="Path to non-overlap validation CSV file (if not provided, no validation is performed)",
    )
    parser.add_argument(
        "--val-csv-overlap",
        type=str,
        default=None,
        help="Path to overlap validation CSV file (if not provided, no overlap validation is performed)",
    )
    parser.add_argument(
        "--csv-overlap",
        type=str,
        default=None,
        help="Path to overlap profiling CSV file. If provided, train on non-overlap first, then continue training on overlap data",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs (default: 1000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256; unused by least-squares solver)",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="If set, fit model parameters (least squares). If not set, only evaluate using an existing checkpoint in --output-dir",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--pred-log",
        action="store_true",
        help="If set, predict log(time) during training and output exp(pred) during inference",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="If set, standardize features during training and evaluation (default: disabled)",
    )

    args = parser.parse_args()
    main(args)
