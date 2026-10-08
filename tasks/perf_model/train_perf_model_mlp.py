"""Train an MLP to predict a production model-runner step from sequence config.

Input features: n_p (new prefill tokens), r_p (prefill prefix/cached KV),
    r_d (decode KV length), decode_bs (decode batch size), overlap (bool)
Target: time_ms (metadata planning + graph selection/replay + full model forward)

Usage:
    python tasks/perf_model/train_perf_model_mlp.py \
        --csv Qwen_Qwen3-8B_*.csv \
        --output tasks/perf_model/model.pt
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


def load_data(csv_path):
    """Load profiler CSV. Columns: n_p, r_p, r_d, decode_bs, time_ms, overlap"""
    import csv
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "n_p": int(row["n_p"]),
                "r_p": int(row["r_p"]),
                "r_d": int(row["r_d"]),
                "decode_bs": int(row["decode_bs"]),
                "overlap": row["overlap"].strip().lower() == "true",
                "time_ms": float(row["time_ms"]),
            })
    return rows


class PerfMLP(nn.Module):
    def __init__(self, input_dim=5, hidden_dims=[128, 64, 32]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def raw_features(rows):
    """Convert profiler rows to an unnormalized feature matrix and target."""
    X = np.array([
        [r["n_p"], r["r_p"], r["r_d"], r["decode_bs"], float(r["overlap"])]
        for r in rows
    ], dtype=np.float32)
    y = np.array([r["time_ms"] for r in rows], dtype=np.float32)
    return X, y


def prepare_features(rows):
    """Fit feature normalization on training rows."""
    X, y = raw_features(rows)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    return X_norm, np.log1p(y), X_mean, X_std, y


def train(
    X,
    y_log,
    y_raw,
    epochs=200,
    lr=1e-3,
    batch_size=512,
    val_frac=0.1,
    external_val=None,
):
    n = len(X)
    if external_val is None:
        perm = np.random.permutation(n)
        val_n = max(1, int(n * val_frac))
        val_idx, train_idx = perm[:val_n], perm[val_n:]
        X_train = torch.tensor(X[train_idx])
        y_train = torch.tensor(y_log[train_idx])
        X_val = torch.tensor(X[val_idx])
        y_val_log = torch.tensor(y_log[val_idx])
        y_val_raw = y_raw[val_idx]
    else:
        X_train = torch.tensor(X)
        y_train = torch.tensor(y_log)
        X_val = torch.tensor(external_val[0])
        y_val_raw = external_val[1]
        y_val_log = torch.tensor(np.log1p(y_val_raw))

    model = PerfMLP(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm_t = torch.randperm(len(X_train))
        total_loss = 0
        n_batches = 0
        for i in range(0, len(X_train), batch_size):
            idx = perm_t[i : i + batch_size]
            pred = model(X_train[idx])
            loss = nn.functional.mse_loss(pred, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred_log = model(X_val)
            val_loss = nn.functional.mse_loss(val_pred_log, y_val_log).item()

            # Convert back to ms for MAPE
            val_pred_ms = torch.expm1(val_pred_log).numpy()
            mape = np.mean(np.abs(val_pred_ms - y_val_raw) / (y_val_raw + 1e-6)) * 100

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}: train_loss={total_loss/n_batches:.6f} "
                  f"val_loss={val_loss:.6f} val_MAPE={mape:.1f}%")

    model.load_state_dict(best_state)

    # Final eval
    model.eval()
    with torch.no_grad():
        val_pred_log = model(X_val)
        val_pred_ms = torch.expm1(val_pred_log).numpy()
        mape = np.mean(np.abs(val_pred_ms - y_val_raw) / (y_val_raw + 1e-6)) * 100
        mae = np.mean(np.abs(val_pred_ms - y_val_raw))
        p90_err = np.percentile(np.abs(val_pred_ms - y_val_raw), 90)
        print(f"\nFinal: MAPE={mape:.1f}%, MAE={mae:.3f}ms, P90_err={p90_err:.3f}ms")
        print(f"  Val samples: {len(y_val_raw)}, min={y_val_raw.min():.2f}ms, max={y_val_raw.max():.2f}ms")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Profiler CSV path")
    parser.add_argument(
        "--val-csv",
        help="Disjoint profiler CSV used for checkpoint selection and reporting",
    )
    parser.add_argument("--output", default="tasks/perf_model/model.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    print(f"Loading {args.csv}...")
    rows = load_data(args.csv)
    print(f"  {len(rows)} samples")

    X_norm, y_log, X_mean, X_std, y_raw = prepare_features(rows)
    print(f"  Features: {X_norm.shape}, Target range: [{y_raw.min():.2f}, {y_raw.max():.2f}] ms")

    np.random.seed(42)
    torch.manual_seed(42)
    external_val = None
    if args.val_csv:
        val_rows = load_data(args.val_csv)
        X_val, y_val = raw_features(val_rows)
        X_val = (X_val - X_mean) / X_std
        external_val = (X_val, y_val)
        print(f"  External validation: {len(y_val)} samples from {args.val_csv}")
    model = train(
        X_norm,
        y_log,
        y_raw,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        external_val=external_val,
    )

    # Save model + normalization params
    torch.save({
        "model_state": model.state_dict(),
        "X_mean": X_mean,
        "X_std": X_std,
        "feature_names": ["n_p", "r_p", "r_d", "decode_bs", "overlap"],
        "target": "production_model_runner_step_ms",
    }, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
