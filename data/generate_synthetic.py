"""
generate_synthetic.py
---------------------
Creates a realistic synthetic PBP feature dataset so the CNN can be
trained immediately without waiting for nba_api rate limits.

Run:  python data/generate_synthetic.py
"""

import numpy as np
import pandas as pd

GAMES  = 2000
EVENTS = 150       # average events per simulated game
OUT    = "data/pbp_features.csv"

rng = np.random.default_rng(42)

rows = []
for game_id in range(GAMES):
    # Simulate a random game score walk
    home_score = 0
    away_score = 0
    home_fouls = 0
    away_fouls = 0
    home_to    = 7
    away_to    = 7

    n_events = rng.integers(80, 220)
    total_seconds = 48 * 60  # 2880

    for ev in range(n_events):
        progress   = ev / n_events
        time_left  = int(total_seconds * (1 - progress))
        period     = min(4, int(progress * 4) + 1)

        # scoring
        if rng.random() < 0.15:
            pts = rng.choice([1, 2, 3], p=[0.15, 0.55, 0.30])
            if rng.random() < 0.5:
                home_score += pts
            else:
                away_score += pts

        # fouls
        if rng.random() < 0.04:
            if rng.random() < 0.5:
                home_fouls = min(home_fouls + 1, 6)
            else:
                away_fouls = min(away_fouls + 1, 6)

        # timeouts
        if rng.random() < 0.01 and home_to > 0:
            home_to -= 1
        if rng.random() < 0.01 and away_to > 0:
            away_to -= 1

        possession = rng.choice([0.0, 0.5, 1.0], p=[0.40, 0.20, 0.40])

        rows.append({
            "game_id"        : game_id,
            "score_diff"     : home_score - away_score,
            "time_remaining_s": time_left,
            "period"         : period,
            "home_fouls"     : home_fouls,
            "away_fouls"     : away_fouls,
            "home_timeouts"  : home_to,
            "away_timeouts"  : away_to,
            "possession"     : possession,
            "label"          : int(home_score > away_score),  # updated each event
        })

    # Correct labels: final outcome for all rows in this game
    final_label = int(home_score > away_score)
    for r in rows[-(n_events):]:
        r["label"] = final_label

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"Generated {len(df):,} rows across {GAMES} games → {OUT}")
print(df.describe())
