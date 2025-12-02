#!/usr/bin/env python3
"""
Train a 3-layer MLP performance model for POD attention overlap prediction.

This script trains a neural network to predict execution time for mixed
prefill+decode batches based on sequence lengths and batch sizes.

Model: 3-layer MLP
  Input: [term1, term2, term3, term4, R_pd, overlap]
  Hidden 1: 64 neurons (ReLU)
  Hidden 2: 32 neurons (ReLU)
  Output: 1 (time prediction)

Where:
  - term1 = n_p²
  - term2 = n_p * r_p
  - term3 = n_p + r_p + (r_d * decode_bs)
  - term4 = r_d * decode_bs
  - R_pd = S_d / (S_p + S_d)
  - overlap = 0 or 1 (boolean flag)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add the parent directory to the path to import sglang
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


class PerformanceMLP(nn.Module):
    """3-layer MLP for predicting execution time of mixed prefill+decode batches.

    Architecture:
      Input (5): [term1, term2, term3, term4, R_pd]
      Hidden 1 (64): ReLU + Dropout
      Hidden 2 (32): ReLU + Dropout
      Output (1): Linear

    When overlap=True, applies speedup to term2 and term4 (paged attention terms):
      Speedup = α₁R_pd + α₂R_pd² + α₃
      term2' = term2 * Speedup
      term4' = term4 * Speedup
    """

    def __init__(
        self,
        hidden1_size=64,
        hidden2_size=32,
        dropout=0.1,
        pred_log: bool = False,
        standardize: bool = True,
    ):
        super().__init__()
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size

        # Input: 5 features (term1, term2, term3, term4, R_pd)
        self.fc1 = nn.Linear(5, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, 1)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Speedup parameters (umbrella-like quadratic curve) for overlap mode
        self.alpha_1 = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )  # R_pd coefficient
        self.alpha_2 = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )  # R_pd² coefficient
        self.alpha_3 = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )  # constant term

        # Feature normalization stats (computed from training set)
        # For [term1, term2, term3, term4, R_pd]
        self.register_buffer("feat_mean", torch.zeros(5, dtype=torch.float32))
        self.register_buffer("feat_std", torch.ones(5, dtype=torch.float32))
        # Whether to predict in log space and apply exp() during inference
        self.pred_log = pred_log
        # Whether to standardize features
        self.standardize = standardize

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features, overlap):
        """Forward pass.

        Args:
            features: Tensor of shape (batch_size, 5) with [term1, term2, term3, term4, R_pd]
            overlap: Indicate whether overlap is enabled for this forward pass

        Returns:
            Predicted time in milliseconds, shape (batch_size,)
        """
        # During training, we assume inputs are already standardized if standardize=True.
        # During eval/inference, normalize using stored training-set stats if standardize=True.
        if self.standardize and not self.training:
            mean = self.feat_mean.view(1, -1)
            std = self.feat_std.view(1, -1)
            features = (features - mean) / std

        # Extract R_pd (last column, index 4)
        R_pd = features[:, 4]

        # Apply speedup to term2 and term4 (paged attention terms) when overlap=True
        # Features: [term1, term2, term3, term4, R_pd] -> indices [0, 1, 2, 3, 4]
        if overlap:
            speedup = self.alpha_1 * R_pd + self.alpha_2 * R_pd**2 + self.alpha_3
            features[:, 1] = features[:, 1] * speedup  # term2
            features[:, 3] = features[:, 3] * speedup  # term4

        # Forward through MLP
        x = self.relu(self.fc1(features))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        # Return flattened output
        pred = x.squeeze(-1)

        # If pred_log is True and not training, apply exp() to convert from log space to time space
        if self.pred_log and not self.training:
            pred = pred.exp()

        return pred

    def get_params(self):
        """Get model architecture info as a dictionary."""
        return {
            "hidden1_size": self.hidden1_size,
            "hidden2_size": self.hidden2_size,
            "num_params": sum(p.numel() for p in self.parameters()),
            "alpha_1": float(self.alpha_1.item()),
            "alpha_2": float(self.alpha_2.item()),
            "alpha_3": float(self.alpha_3.item()),
            "feat_mean_term1": float(self.feat_mean[0].item()),
            "feat_mean_term2": float(self.feat_mean[1].item()),
            "feat_mean_term3": float(self.feat_mean[2].item()),
            "feat_mean_term4": float(self.feat_mean[3].item()),
            "feat_mean_R_pd": float(self.feat_mean[4].item()),
            "feat_std_term1": float(self.feat_std[0].item()),
            "feat_std_term2": float(self.feat_std[1].item()),
            "feat_std_term3": float(self.feat_std[2].item()),
            "feat_std_term4": float(self.feat_std[3].item()),
            "feat_std_R_pd": float(self.feat_std[4].item()),
        }


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute model features from raw profiling data.

    Args:
        df: DataFrame with columns [n_p, r_p, r_d, decode_bs, time_ms, overlap]

    Returns:
        DataFrame with added feature columns [term1, term2, term3, term4, R_pd] and preserved overlap
    """
    df = df.copy()

    # Ensure overlap column exists (default to False if missing)
    if "overlap" not in df.columns:
        df["overlap"] = False

    # Compute model terms
    df["term1"] = df["n_p"] ** 2
    df["term2"] = df["n_p"] * df["r_p"]
    df["term3"] = df["n_p"] + df["r_p"] + (df["r_d"] * df["decode_bs"])
    df["term4"] = df["r_d"] * df["decode_bs"]

    # Compute R_pd = S_d / (S_p + S_d)
    # For single sample: S_p = r_p, S_d = r_d * decode_bs
    S_p = df["r_p"]
    S_d = df["r_d"] * df["decode_bs"]
    df["R_pd"] = S_d / (
        S_p + S_d + 1e-10
    )  # Add small epsilon to avoid division by zero

    return df


def train_model_mlp(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    overlap: bool,
    model: Optional[PerformanceMLP] = None,
    num_epochs: int = 1000,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    hidden1_size: int = 64,
    hidden2_size: int = 32,
    dropout: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    pred_log: bool = False,
    standardize: bool = True,
):

    criterion = nn.MSELoss()
    train_df["time_ms"] = pd.to_numeric(train_df["time_ms"], errors="coerce")
    val_df["time_ms"] = pd.to_numeric(val_df["time_ms"], errors="coerce")
    # Create new model or use existing one
    if model is None:
        model = PerformanceMLP(
            hidden1_size=hidden1_size,
            hidden2_size=hidden2_size,
            dropout=dropout,
            pred_log=pred_log,
            standardize=standardize,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        train_all_params = True
    else:
        # Freeze base MLP parameters (fc1, fc2, fc3)
        for param in model.fc1.parameters():
            param.requires_grad = False
        for param in model.fc2.parameters():
            param.requires_grad = False
        for param in model.fc3.parameters():
            param.requires_grad = False
        # Only train speedup parameters (alpha_1, alpha_2, alpha_3)
        optimizer = torch.optim.Adam(
            [model.alpha_1, model.alpha_2, model.alpha_3],
            lr=learning_rate,
            weight_decay=1e-4,
        )
        train_all_params = False

    # Prepare data
    train_features_raw = torch.tensor(
        train_df[["term1", "term2", "term3", "term4", "R_pd"]].values,
        dtype=torch.float32,
    ).to(device)
    # Use log(time_ms) as targets if pred_log is True
    if pred_log:
        train_targets = torch.log(
            torch.tensor(train_df["time_ms"].values, dtype=torch.float32) + 1e-8
        ).to(device)
    else:
        train_targets = torch.tensor(
            train_df["time_ms"].values, dtype=torch.float32
        ).to(device)

    val_features_raw = torch.tensor(
        val_df[["term1", "term2", "term3", "term4", "R_pd"]].values, dtype=torch.float32
    ).to(device)
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
        if train_all_params:
            # Compute stats from training set
            feat_mean = train_features_raw.mean(dim=0)
            feat_std = train_features_raw.std(dim=0)
            eps = 1e-6
            feat_std = torch.where(feat_std < eps, torch.ones_like(feat_std), feat_std)
            # Store stats in the model
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

    # Create data loader
    train_dataset = TensorDataset(train_features, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience = 100
    patience_counter = 0
    best_ckpt = None
    if train_all_params:
        print(f"Training on {device}")
        print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
        print(f"Model architecture: 5 -> {hidden1_size} -> {hidden2_size} -> 1")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    else:
        print(f"Training overlap speedup parameters on {device}")
        print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
        print(f"Frozen params: fc1, fc2, fc3, feat_mean, feat_std")
        print(f"Training params: alpha_1, alpha_2, alpha_3")
    print()

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_features, batch_targets in train_loader:
            pred = model(batch_features, overlap=overlap)
            loss = criterion(pred, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_features_raw, overlap=overlap)
            if pred_log:
                # Model outputs exp(log(time)) = time, compare with raw targets
                val_loss = torch.mean((val_pred - val_targets_raw) ** 2).item()
            else:
                val_loss = criterion(val_pred, val_targets).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_ckpt = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

    print(f"\nBest val loss: {best_val_loss:.4f}")
    model.load_state_dict(best_ckpt)
    return model, history


def evaluate_model(
    model: nn.Module, df: pd.DataFrame, device: str = "cpu", pred_log: bool = False
):
    """Evaluate model on a dataset.

    Args:
        model: Trained model
        df: DataFrame with columns [term1, term2, term3, term4, R_pd, overlap, time_ms]
            (overlap is passed as a separate boolean flag, not as a feature)
        device: Device to evaluate on (model will be moved to CPU for evaluation)
        pred_log: If True, model predicts log(time), so apply exp() to predictions

    Returns:
        Dictionary with evaluation metrics including average inference time
    """
    model.eval()

    # Move model to CPU for evaluation
    model_cpu = model.cpu()
    eval_device = "cpu"

    # Split into non-overlap and overlap
    df_no_overlap = df[df["overlap"] == False]
    df_overlap = df[df["overlap"] == True]

    preds = []
    targets_list = []
    inference_times = []

    with torch.no_grad():
        # Non-overlap evaluation
        if len(df_no_overlap) > 0:
            features_no = torch.tensor(
                df_no_overlap[["term1", "term2", "term3", "term4", "R_pd"]].values,
                dtype=torch.float32,
                device=eval_device,
            )
            targets_no = torch.tensor(
                df_no_overlap["time_ms"].values, dtype=torch.float32, device=eval_device
            )
            # Measure inference time
            start_time = time.perf_counter()
            # Model handles exp() internally when pred_log=True and not training
            pred_no = model_cpu(features_no, overlap=False).cpu().numpy()
            end_time = time.perf_counter()
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)
            preds.append(pred_no)
            targets_list.append(targets_no.cpu().numpy())

        # Overlap evaluation
        if len(df_overlap) > 0:
            features_ov = torch.tensor(
                df_overlap[["term1", "term2", "term3", "term4", "R_pd"]].values,
                dtype=torch.float32,
                device=eval_device,
            )
            targets_ov = torch.tensor(
                df_overlap["time_ms"].values, dtype=torch.float32, device=eval_device
            )
            # Measure inference time
            start_time = time.perf_counter()
            # Model handles exp() internally when pred_log=True and not training
            pred_ov = model_cpu(features_ov, overlap=True).cpu().numpy()
            end_time = time.perf_counter()
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)
            preds.append(pred_ov)
            targets_list.append(targets_ov.cpu().numpy())

        pred = np.concatenate(preds) if len(preds) > 0 else np.array([])
        targets = (
            np.concatenate(targets_list) if len(targets_list) > 0 else np.array([])
        )

    # Calculate average inference time per sample
    total_samples = len(pred)
    total_inference_time = sum(inference_times)
    avg_inference_time_per_sample = (
        total_inference_time / total_samples if total_samples > 0 else 0.0
    )

    print(
        f"  Average inference time per sample: {avg_inference_time_per_sample:.4f} ms"
    )

    mse = np.mean((pred - targets) ** 2)
    abs_err = np.abs(pred - targets)
    mean_deviation = np.mean(abs_err)  # Mean absolute error in ms
    median_deviation = np.median(abs_err)  # Median absolute error in ms
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
        "median_deviation": median_deviation,
        "mean_deviation_pct": mean_deviation_pct,
        "max_deviation": max_deviation,
        "avg_inference_time_per_sample_ms": avg_inference_time_per_sample,
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    history = {"train_loss": [], "val_loss": []}
    model: PerformanceMLP

    # -----------------------------
    # Training
    # -----------------------------
    if args.csv:
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
        print(
            f"Features: {train_df[['term1', 'term2', 'term3', 'term4', 'R_pd']].describe()}"
        )
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
            val_size = min(1000, max(100, int(0.1 * len(train_df))))
            val_df = train_df.iloc[:val_size].copy()
            train_df = train_df.iloc[val_size:].copy()
            print(f"Using {val_size} samples from training data as validation")
            print()

        # Train model on non-overlap data
        print("=" * 60)
        print("Training MLP model on non-overlap data...")
        print("=" * 60)
        model, history = train_model_mlp(
            train_df,
            val_df,
            overlap=False,
            model=None,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            hidden1_size=args.hidden1_size,
            hidden2_size=args.hidden2_size,
            dropout=args.dropout,
            device=device,
            pred_log=args.pred_log,
            standardize=args.standardize,
        )

        # If overlap CSV is provided, continue training on overlap data (only speedup parameters)
        if args.csv_overlap:
            print("\n" + "=" * 60)
            print("Training overlap speedup parameters (freezing base MLP)...")
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
                f"Features: {train_df_overlap[['term1', 'term2', 'term3', 'term4', 'R_pd']].describe()}"
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

            # Continue training on overlap data (only speedup parameters)
            model, history_overlap = train_model_mlp(
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
        base_name = "perf_model_mlp"
        model_filename = get_model_filename(base_name, args.pred_log, args.standardize)
        model_path = os.path.join(args.output_dir, f"{model_filename}.pt")

        # If file doesn't exist with suffixes, try base name
        if not os.path.exists(model_path):
            model_path = os.path.join(args.output_dir, f"{base_name}.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No existing model found at {model_path}. "
                "Run with --training first to train and save the model."
            )
        print(f"Loading existing model from {model_path} for evaluation...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Use pred_log and standardize from checkpoint if available, otherwise use args
        pred_log_loaded = checkpoint.get("pred_log", args.pred_log)
        standardize_loaded = checkpoint.get("standardize", args.standardize)
        model = PerformanceMLP(
            hidden1_size=checkpoint.get("hidden1_size", 64),
            hidden2_size=checkpoint.get("hidden2_size", 32),
            dropout=checkpoint.get("dropout", 0.1),
            pred_log=pred_log_loaded,
            standardize=standardize_loaded,
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
                f"  Median deviation: {val_metrics_no_overlap['median_deviation']:.4f} ms"
            )
            print(
                f"  Mean deviation %: {val_metrics_no_overlap['mean_deviation_pct']:.4f}%"
            )
            print(f"  Max deviation:  {val_metrics_no_overlap['max_deviation']:.4f} ms")
            print()
            val_metrics["non-overlap-mse"] = val_metrics_no_overlap["mse"]
            val_metrics["non-overlap-mean_deviation"] = val_metrics_no_overlap[
                "mean_deviation"
            ]
            val_metrics["non-overlap-median_deviation"] = val_metrics_no_overlap[
                "median_deviation"
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
                f"  Median deviation: {val_metrics_overlap['median_deviation']:.4f} ms"
            )
            print(
                f"  Mean deviation %: {val_metrics_overlap['mean_deviation_pct']:.4f}%"
            )
            print(f"  Max deviation:  {val_metrics_overlap['max_deviation']:.4f} ms")
            print()
            val_metrics["overlap-mse"] = val_metrics_overlap["mse"]
            val_metrics["overlap-mean_deviation"] = val_metrics_overlap[
                "mean_deviation"
            ]
            val_metrics["overlap-median_deviation"] = val_metrics_overlap[
                "median_deviation"
            ]
            val_metrics["overlap-mean_deviation_pct"] = val_metrics_overlap[
                "mean_deviation_pct"
            ]
            val_metrics["overlap-max_deviation"] = val_metrics_overlap["max_deviation"]

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_filename = get_model_filename(
        "perf_model_mlp", args.pred_log, args.standardize
    )
    model_path = os.path.join(args.output_dir, f"{model_filename}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "params": model.get_params(),
            "history": history,
            "val_metrics": val_metrics,
            "hidden1_size": args.hidden1_size,
            "hidden2_size": args.hidden2_size,
            "dropout": args.dropout,
            "pred_log": args.pred_log,
            "standardize": args.standardize,
        },
        model_path,
    )
    print(f"Model saved to {model_path}")

    # Save parameters to text file
    params_filename = get_model_filename(
        "perf_model_mlp_params", args.pred_log, args.standardize
    )
    params_path = os.path.join(args.output_dir, f"{params_filename}.txt")
    with open(params_path, "w") as f:
        f.write("Performance MLP Model Parameters\n")
        f.write("=" * 60 + "\n\n")
        params = model.get_params()
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("Validation Metrics:\n")
    print(f"Parameters saved to {params_path}")

    # Save validation metrics as CSV
    if val_metrics:
        metrics_filename = get_model_filename(
            "validation_metrics_mlp", args.pred_log, args.standardize
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
                    "median_deviation_ms": val_metrics["non-overlap-median_deviation"],
                }
            )
        if "overlap-mse" in val_metrics:
            metrics_rows.append(
                {
                    "type": "overlap",
                    "mse": val_metrics["overlap-mse"],
                    "mean_deviation_ms": val_metrics["overlap-mean_deviation"],
                    "median_deviation_ms": val_metrics["overlap-median_deviation"],
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
        description="Train 3-layer MLP performance model for POD attention overlap"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=False,
        default=None,
        help="Path to profiling CSV file. If provided, will train the model. If not provided, will only evaluate using an existing checkpoint in --output-dir",
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
        help="Path to non-overlap validation CSV file",
    )
    parser.add_argument(
        "--val-csv-overlap",
        type=str,
        default=None,
        help="Path to overlap validation CSV file",
    )
    parser.add_argument(
        "--csv-overlap",
        type=str,
        default=None,
        help="Path to overlap profiling CSV file. If provided, will be combined with --csv for training",
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
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256)",
    )
    parser.add_argument(
        "--hidden1-size",
        type=int,
        default=128,
        help="Size of first hidden layer (default: 64)",
    )
    parser.add_argument(
        "--hidden2-size",
        type=int,
        default=64,
        help="Size of second hidden layer (default: 32)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)",
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
