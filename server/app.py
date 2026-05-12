"""
app.py
------
Flask + Flask-SocketIO server.

REST endpoints:
  GET  /              → dashboard HTML
  GET  /api/games     → list of live/recent games from nba_api
  POST /api/predict   → manual win-prob query {game_state}

WebSocket events (server → client):
  game_update         → {game_state, home_win_prob, away_win_prob, timestamp}

WebSocket events (client → server):
  subscribe_game      → {game_id}
  unsubscribe_game    → {game_id}

Live mode: polls nba_api scoreboard every ~5 s and broadcasts updates.
Simulation mode: replays a synthetic game for demo when no live games exist.
"""

import time
import threading
import datetime
import random
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request, render_template_string
from flask_socketio import SocketIO, emit, join_room, leave_room

# nba_api
from nba_api.stats.endpoints import scoreboardv2, boxscoreadvancedv2
from nba_api.stats.static   import teams as nba_teams

# our model
from model.inference import predict

# ── App setup ────────────────────────────────────────────────────────────────

app    = Flask(__name__, static_folder="static")
app.config["SECRET_KEY"] = "nba_win_prob_secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── In-memory state ──────────────────────────────────────────────────────────

subscriptions: dict[str, set] = {}   # game_id → set of socket ids
game_states:   dict[str, dict] = {}  # game_id → latest game state

TEAM_MAP = {t["id"]: t["abbreviation"] for t in nba_teams.get_teams()}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _period_seconds_remaining(period_str: str, clock_str: str) -> int:
    """Convert NBA period/clock to seconds remaining in game."""
    try:
        period = int(period_str)
        if ":" in str(clock_str):
            m, s = str(clock_str).split(":")
            clock_s = int(m) * 60 + float(s)
        else:
            clock_s = float(clock_str or 0)
        remaining_periods = max(0, 4 - period)
        return int(remaining_periods * 12 * 60 + clock_s)
    except Exception:
        return 0


def build_game_state(game_data: dict) -> dict:
    """Build a game_state dict from raw scoreboard data."""
    home = game_data.get("home", {})
    away = game_data.get("away", {})

    score_diff = int(home.get("score", 0)) - int(away.get("score", 0))
    period     = int(game_data.get("period", 4))
    clock      = str(game_data.get("clock", "0:00"))
    time_left  = _period_seconds_remaining(period, clock)

    return {
        "game_id"         : game_data.get("game_id", ""),
        "home_team"       : home.get("abbr", "HOME"),
        "away_team"       : away.get("abbr", "AWAY"),
        "home_score"      : int(home.get("score", 0)),
        "away_score"      : int(away.get("score", 0)),
        "score_diff"      : score_diff,
        "time_remaining_s": time_left,
        "period"          : min(period, 4),
        "home_fouls"      : int(home.get("fouls", 0)),
        "away_fouls"      : int(away.get("fouls", 0)),
        "home_timeouts"   : int(home.get("timeouts", 3)),
        "away_timeouts"   : int(away.get("timeouts", 3)),
        "possession"      : 0.5,
        "status"          : game_data.get("status", ""),
        "clock"           : clock,
    }


def model_probs(gs: dict) -> tuple[float, float]:
    """Return (home_win_prob, away_win_prob)."""
    hw = predict({k: gs[k] for k in [
        "score_diff", "time_remaining_s", "period",
        "home_fouls", "away_fouls",
        "home_timeouts", "away_timeouts",
        "possession",
    ]})
    return hw, round(1 - hw, 4)


# ── NBA API polling ──────────────────────────────────────────────────────────

def fetch_live_games() -> list[dict]:
    """Fetch today's NBA scoreboard and return list of game dicts."""
    try:
        today  = datetime.date.today().strftime("%m/%d/%Y")
        board  = scoreboardv2.ScoreboardV2(game_date=today)
        frames = board.get_data_frames()
        lines  = frames[1]   # LineScore
        header_df = frames[0]

        games  = {}
        for _, row in lines.iterrows():
            gid     = str(row["GAME_ID"])
            team_id = int(row["TEAM_ID"])
            abbr    = TEAM_MAP.get(team_id, str(row.get("TEAM_ABBREVIATION", "???")))
            score   = int(row.get("PTS", 0) or 0)
            fouls   = int(row.get("PF",  0) or 0)

            header  = header_df[header_df["GAME_ID"] == gid]
            period  = int(header["LIVE_PERIOD"].iloc[0]) if not header.empty else 4
            clock   = str(header["LIVE_PC_TIME"].iloc[0]) if not header.empty else "0:00"
            status  = str(header["GAME_STATUS_TEXT"].iloc[0]) if not header.empty else ""

            if gid not in games:
                games[gid] = {"game_id": gid, "period": period,
                               "clock": clock, "status": status,
                               "home": {}, "away": {}}

            entry = games[gid]
            side  = "away" if not entry["away"] else "home"

            # pull timeouts from team stats if available
            try:
                team_stats = frames[4]  # TeamStats frame
                ts = team_stats[team_stats["TEAM_ID"] == team_id]
                timeouts = int(ts["TEAM_TIMEOUTS_REMAINING"].iloc[0]) if not ts.empty else 7
            except Exception:
                timeouts = 7

            entry[side] = {
                "abbr"    : abbr,
                "score"   : score,
                "fouls"   : fouls,
                "timeouts": timeouts,
            }

        return list(games.values())
    except Exception as e:
        print(f"[scoreboard] Error: {e}")
        return []

# ── Simulation (demo fallback) ────────────────────────────────────────────────

class GameSimulator:
    """Walks through a fake game for demo purposes."""

    def __init__(self, game_id="SIM_001", home="LAL", away="GSW"):
        self.game_id    = game_id
        self.home       = home
        self.away       = away
        self.home_score = 0
        self.away_score = 0
        self.period     = 1
        self.clock_s    = 12 * 60
        self.home_fouls = 0
        self.away_fouls = 0
        self.home_to    = 7
        self.away_to    = 7
        self.done       = False

    def step(self):
        if self.done:
            return

        # advance clock
        self.clock_s -= random.randint(5, 25)
        if self.clock_s <= 0:
            self.period += 1
            self.clock_s = 12 * 60
            self.home_fouls = 0
            self.away_fouls = 0
            if self.period > 4:
                self.done = True
                return

        # scoring event (~18% of ticks)
        if random.random() < 0.18:
            pts  = random.choice([1, 2, 2, 2, 3])
            if random.random() < 0.52:
                self.home_score += pts
            else:
                self.away_score += pts

        # foul (~5%)
        if random.random() < 0.05:
            if random.random() < 0.5:
                self.home_fouls = min(self.home_fouls + 1, 6)
            else:
                self.away_fouls = min(self.away_fouls + 1, 6)

        # timeout (~1%)
        if random.random() < 0.01 and self.home_to > 0:
            self.home_to -= 1
        if random.random() < 0.01 and self.away_to > 0:
            self.away_to -= 1

    @property
    def state(self) -> dict:
        time_left = (4 - self.period) * 12 * 60 + self.clock_s
        mins = self.clock_s // 60
        secs = self.clock_s  % 60
        return {
            "game_id"         : self.game_id,
            "home_team"       : self.home,
            "away_team"       : self.away,
            "home_score"      : self.home_score,
            "away_score"      : self.away_score,
            "score_diff"      : self.home_score - self.away_score,
            "time_remaining_s": max(0, time_left),
            "period"          : min(self.period, 4),
            "home_fouls"      : self.home_fouls,
            "away_fouls"      : self.away_fouls,
            "home_timeouts"   : self.home_to,
            "away_timeouts"   : self.away_to,
            "possession"      : random.choice([0.0, 0.5, 1.0]),
            "status"          : "In Progress" if not self.done else "Final",
            "clock"           : f"{mins}:{secs:02d}",
        }


_simulators: dict[str, GameSimulator] = {}


# ── Background broadcast thread ───────────────────────────────────────────────

def broadcast_loop():
    """Runs in background; every 4 s fetches/steps games and broadcasts."""
    sim_ids = [f"SIM_{i:03d}" for i in range(1, 4)]
    teams   = [("LAL","GSW"), ("BOS","MIA"), ("DEN","PHX")]

    for i, sid in enumerate(sim_ids):
        _simulators[sid] = GameSimulator(sid, *teams[i])

    while True:
        time.sleep(4)

        # Try live NBA games first
        live = fetch_live_games()
        active_ids = set()

        if live:
            for gd in live:
                gs = build_game_state(gd)
                hw, aw = model_probs(gs)
                payload = {**gs, "home_win_prob": hw, "away_win_prob": aw,
                           "timestamp": time.time(), "source": "live"}
                game_states[gs["game_id"]] = payload
                active_ids.add(gs["game_id"])
                socketio.emit("game_update", payload, room=gs["game_id"])

        # Always run simulators (for demo)
        for sid, sim in list(_simulators.items()):
            if sim.done:
                # reset after final
                i       = sim_ids.index(sid)
                _simulators[sid] = GameSimulator(sid, *teams[i])
                sim     = _simulators[sid]

            sim.step()
            gs = sim.state
            hw, aw = model_probs(gs)
            payload = {**gs, "home_win_prob": hw, "away_win_prob": aw,
                       "timestamp": time.time(), "source": "sim"}
            game_states[sid] = payload
            active_ids.add(sid)
            socketio.emit("game_update", payload, room=sid)

        # Broadcast lobby list
        lobby = [
            {
                "game_id"  : gid,
                "home_team": s["home_team"],
                "away_team": s["away_team"],
                "home_score": s["home_score"],
                "away_score": s["away_score"],
                "status"   : s["status"],
                "source"   : s.get("source", ""),
            }
            for gid, s in game_states.items()
        ]
        socketio.emit("lobby_update", lobby)


# ── REST API ──────────────────────────────────────────────────────────────────

@app.route("/api/games")
def api_games():
    return jsonify(list(game_states.values()))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    gs = request.json
    try:
        hw = predict(gs)
        return jsonify({"home_win_prob": hw, "away_win_prob": round(1 - hw, 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/")
def index():
    return render_template_string(open("static/index.html", encoding="utf-8").read())


# ── WebSocket events ──────────────────────────────────────────────────────────

@socketio.on("subscribe_game")
def on_subscribe(data):
    gid = data.get("game_id", "")
    join_room(gid)
    print(f"[ws] client subscribed → {gid}")
    if gid in game_states:
        emit("game_update", game_states[gid])


@socketio.on("unsubscribe_game")
def on_unsubscribe(data):
    gid = data.get("game_id", "")
    leave_room(gid)


@socketio.on("connect")
def on_connect():
    # send current lobby on connect
    lobby = [
        {"game_id": gid, "home_team": s["home_team"],
         "away_team": s["away_team"],
         "home_score": s["home_score"], "away_score": s["away_score"],
         "status": s["status"], "source": s.get("source","")}
        for gid, s in game_states.items()
    ]
    emit("lobby_update", lobby)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t = threading.Thread(target=broadcast_loop, daemon=True)
    t.start()
    print("NBA Win Probability Server starting on http://0.0.0.0:5050")
    socketio.run(app, host="0.0.0.0", port=5050, debug=False)
