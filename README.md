# NBA Live Win Probability Model

A real-time NBA win probability dashboard powered by a 1-D CNN trained on play-by-play data.

## Architecture

```
nba_api  ──►  data/collect.py        play-by-play feature extraction
                     │
                     ▼
              data/pbp_features.csv   score_diff, time_remaining, period,
                     │                fouls, timeouts, possession
                     ▼
              model/train.py          1-D CNN (PyTorch)
                     │
                     ├── model/win_prob_cnn.pt      TorchScript model
                     └── model/scaler_params.json   feature normalisation
                              │
                              ▼
              server/app.py           Flask + Flask-SocketIO
                    │   ▲
                    │   │  WebSocket (game_update events ~4s)
                    ▼   │
              static/index.html       Real-time dashboard
                                      Chart.js probability timeline
                                      Score, clock, fouls, timeouts
                                      Probability swing log
```

## Quick Start

```bash
# 1. Install deps
pip install nba_api pandas torch flask flask-socketio eventlet scikit-learn

# 2. Run everything (data → train → serve)
python run.py

# Dashboard: http://localhost:5050
```

## Individual steps

```bash
# Generate synthetic data (fast, no NBA API needed)
python data/generate_synthetic.py

# Pull REAL play-by-play from nba_api (slower, requires internet)
python data/collect.py

# Train the CNN
python model/train.py

# Start the server (model must be trained first)
cd nba_win_prob
python -c "
import threading
from server.app import socketio, app, broadcast_loop
t = threading.Thread(target=broadcast_loop, daemon=True)
t.start()
socketio.run(app, host='0.0.0.0', port=5050)
"
```

## CNN Model

```
Input: (batch, 1, 9)  — 9 normalised features
Conv1d(1→64,  k=3) + BN + ReLU
Conv1d(64→128, k=3) + BN + ReLU
AdaptiveAvgPool1d(1)
Linear(128→64) + ReLU + Dropout(0.3)
Linear(64→1) + Sigmoid  →  P(home wins)
```

**Features:**
| Feature | Description |
|---------|-------------|
| `score_diff` | home_score − away_score |
| `time_remaining_s` | seconds left in game |
| `period` | quarter (1–4) |
| `home_fouls` / `away_fouls` | cumulative fouls |
| `home_timeouts` / `away_timeouts` | remaining timeouts |
| `possession` | 1=home, 0=away, 0.5=unknown |
| `score_diff_x_time` | score_diff × time_remaining / 2880 (engineered) |

## WebSocket API

**Subscribe to a game:**
```js
socket.emit("subscribe_game", { game_id: "SIM_001" });
```

**Receive updates:**
```js
socket.on("game_update", (data) => {
  // data.home_win_prob  — float 0–1
  // data.away_win_prob  — float 0–1
  // data.score_diff, time_remaining_s, period, clock, ...
});
```

**REST predict:**
```bash
curl -X POST http://localhost:5050/api/predict \
  -H "Content-Type: application/json" \
  -d '{"score_diff":8,"time_remaining_s":120,"period":4,
       "home_fouls":3,"away_fouls":5,
       "home_timeouts":2,"away_timeouts":1,"possession":1.0}'
```

## Live NBA Data

`server/app.py` polls `nba_api.stats.endpoints.scoreboard` every 4 seconds.
During off-season or when no games are live, 3 simulated games run automatically.

To collect historical PBP for more robust training:
1. Edit `data/collect.py`: increase `MAX_GAMES_PER_SEASON`
2. Add more seasons to `SEASONS`
3. Re-run `python data/collect.py` then `python model/train.py`
