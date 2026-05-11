#!/usr/bin/env python3
"""
run.py  —  One-command launcher for the NBA Win Probability system.

Steps:
  1. Generate synthetic training data (if not already present)
  2. Train the CNN (if model not already saved)
  3. Start the Flask/SocketIO server
"""

import os, sys, subprocess
from pathlib import Path

ROOT = Path(__file__).parent

def step(msg): print(f"\n{'='*60}\n  {msg}\n{'='*60}")

def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd or ROOT)
    if r.returncode != 0:
        print(f"[ERROR] command failed: {' '.join(cmd)}")
        sys.exit(1)

if __name__ == "__main__":
    os.chdir(ROOT)

    # 1. Data
    data_csv = ROOT / "data" / "pbp_features.csv"
    if not data_csv.exists():
        step("Generating synthetic training data")
        run([sys.executable, "data/generate_synthetic.py"])
    else:
        print(f"[skip] data already exists: {data_csv}")

    # 2. Model
    model_pt = ROOT / "model" / "win_prob_cnn.pt"
    if not model_pt.exists():
        step("Training CNN model")
        run([sys.executable, "model/train.py"])
    else:
        print(f"[skip] model already exists: {model_pt}")

    # 3. Server
    step("Starting server on http://localhost:5050")
    import threading
    from server.app import socketio, app, broadcast_loop

    t = threading.Thread(target=broadcast_loop, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 5050))
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
