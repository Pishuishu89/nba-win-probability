"""
inference.py
------------
Loads the trained CNN and scaler; exposes predict(game_state) -> float.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

MODEL_PT = Path(__file__).parent / "win_prob_cnn.pt"
SCALER_J = Path(__file__).parent / "scaler_params.json"

_model  = None
_scaler = None


class _WinProbCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.head(self.conv(x))

def _build_model():
    return _WinProbCNN()


def _load():
    global _model, _scaler
    if _model is None:
        m = _build_model()
        m.load_state_dict(torch.load(str(MODEL_PT), map_location="cpu"))
        m.eval()
        _model = m
    if _scaler is None:
        _scaler = json.load(open(SCALER_J))


def predict(game_state: dict) -> float:
    """
    game_state keys (same as FEATURES in train.py):
        score_diff, time_remaining_s, period,
        home_fouls, away_fouls,
        home_timeouts, away_timeouts,
        possession

    Returns probability (0–1) that the HOME team wins.
    """
    _load()
    features = _scaler["features"]
    mean     = np.array(_scaler["mean"], dtype=np.float32)
    std      = np.array(_scaler["std"],  dtype=np.float32)

    # build engineered feature
    gs = dict(game_state)
    gs["score_diff_x_time"] = gs["score_diff"] * gs["time_remaining_s"] / (48 * 60)

    x = np.array([gs[f] for f in features], dtype=np.float32)
    x = (x - mean) / std

    with torch.no_grad():
        t    = torch.tensor(x).unsqueeze(0).unsqueeze(0)   # (1, 1, 9)
        prob = _model(t).item()

    return round(prob, 4)


if __name__ == "__main__":
    # quick smoke-test
    state = {
        "score_diff"      : 8,
        "time_remaining_s": 120,
        "period"          : 4,
        "home_fouls"      : 3,
        "away_fouls"      : 5,
        "home_timeouts"   : 2,
        "away_timeouts"   : 1,
        "possession"      : 1.0,
    }
    print(f"Win prob (home): {predict(state):.1%}")
