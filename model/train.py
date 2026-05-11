"""
train.py
--------
Trains a 1-D CNN on the PBP feature CSV.

Architecture:
  Input  : (batch, 1, 9)   — 9 game-state features as a "sequence"
  Conv1  : 64 filters, k=3, padding=1  → (batch, 64, 9)
  Conv2  : 128 filters, k=3, padding=1 → (batch, 128, 9)
  Pool   : AdaptiveAvgPool → (batch, 128, 1)
  FC1    : 128 → 64 + ReLU + Dropout 0.3
  FC2    : 64  → 1  + Sigmoid  → win probability

Saves:
  model/win_prob_cnn.pt      — full TorchScript model (for inference)
  model/scaler_params.json   — mean/std for feature normalization
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path

CSV      = "data/pbp_features.csv"
MODEL_PT = "model/win_prob_cnn.pt"
SCALER_J = "model/scaler_params.json"

FEATURES = [
    "score_diff", "time_remaining_s", "period",
    "home_fouls", "away_fouls",
    "home_timeouts", "away_timeouts",
    "possession",
    # engineered
    "score_diff_x_time",
]

EPOCHS     = 20
BATCH_SIZE = 512
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model ─────────────────────────────────────────────────────────────────────

class WinProbCNN(nn.Module):
    def __init__(self, n_features: int = 9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):           # x: (B, 1, n_features)
        return self.head(self.conv(x))


# ── Training helpers ───────────────────────────────────────────────────────────

def load_and_preprocess(path: str):
    df = pd.read_csv(path)

    # engineered feature
    df["score_diff_x_time"] = df["score_diff"] * df["time_remaining_s"] / (48 * 60)

    X = df[FEATURES].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8

    X_norm = (X - mean) / std
    return X_norm, y, mean, std


def train():
    Path("model").mkdir(exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Loading data …")
    X, y, mean, std = load_and_preprocess(CSV)

    # save scaler
    json.dump({"mean": mean.tolist(), "std": std.tolist(), "features": FEATURES},
              open(SCALER_J, "w"), indent=2)

    X_t = torch.tensor(X).unsqueeze(1)          # (N, 1, 9)
    y_t = torch.tensor(y).unsqueeze(1)           # (N, 1)

    dataset = TensorDataset(X_t, y_t)
    n_val   = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model = WinProbCNN(n_features=len(FEATURES)).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        # ── validate ──
        model.eval()
        val_loss = 0.0
        correct  = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred     = model(xb)
                val_loss += loss_fn(pred, yb).item() * len(xb)
                correct  += ((pred > 0.5).float() == yb).sum().item()
        val_loss /= n_val
        val_acc   = correct / n_val

        scheduler.step(val_loss)

        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PT)
            print(f"           ↳ saved best model (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Model → {MODEL_PT}  Scaler → {SCALER_J}")


if __name__ == "__main__":
    train()
