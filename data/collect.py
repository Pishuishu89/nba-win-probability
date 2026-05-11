"""
collect.py
----------
Pulls play-by-play game logs from nba_api for past seasons,
engineers features, and saves a training-ready CSV.

Features per possession/event:
  - score_diff         : home_score - away_score
  - time_remaining_s   : seconds left in game
  - period             : 1-4 (OT capped at 4)
  - home_fouls         : home team fouls in period
  - away_fouls         : away team fouls in period
  - home_timeouts      : timeouts remaining (estimated)
  - away_timeouts      : timeouts remaining (estimated)
  - possession         : 1=home, 0=away, 0.5=unknown
  - label              : 1 if home team won, 0 otherwise
"""

import time
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder, playbyplay
from nba_api.stats.static import teams
import requests

SEASONS = ["2021-22", "2022-23", "2023-24"]
MAX_GAMES_PER_SEASON = 50          # keep training fast; bump for production
DELAY = 0.7                         # polite rate-limit between requests
OUTPUT = "data/pbp_features.csv"


def get_game_ids(season: str, n: int) -> list[str]:
    """Return up to n completed regular-season game IDs for a season."""
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
    )
    df = finder.get_data_frames()[0]
    # Each game appears twice (home + away) – deduplicate
    game_ids = df["GAME_ID"].unique().tolist()
    return game_ids[:n]


def parse_clock(clock_str) -> int:
    """Convert 'PT11M34.00S' or 'MM:SS' format to total seconds."""
    if pd.isna(clock_str):
        return 0
    s = str(clock_str).strip()
    # ISO 8601 duration: PT11M34.00S
    if s.startswith("PT"):
        s = s[2:]
        mins = 0
        secs = 0
        if "M" in s:
            parts = s.split("M")
            mins = float(parts[0])
            secs = float(parts[1].replace("S", "")) if parts[1] else 0
        else:
            secs = float(s.replace("S", ""))
        return int(mins * 60 + secs)
    # MM:SS
    if ":" in s:
        parts = s.split(":")
        return int(parts[0]) * 60 + int(float(parts[1]))
    return 0


def collect_game(game_id: str) -> pd.DataFrame | None:
    """Return a DataFrame of feature rows for one game, or None on error."""
    try:
        pbp = playbyplay.PlayByPlay(game_id=game_id)
        df = pbp.get_data_frames()[0]
    except Exception as e:
        print(f"  [skip] {game_id}: {e}")
        return None

    if df.empty:
        return None

    # --- parse scores --------------------------------------------------------
    df["SCORE"] = df["SCORE"].fillna(method="ffill").fillna("0 - 0")
    def split_score(s):
        try:
            a, b = str(s).split(" - ")
            return int(a), int(b)
        except Exception:
            return 0, 0

    df[["away_score", "home_score"]] = pd.DataFrame(
        df["SCORE"].apply(split_score).tolist(), index=df.index
    )

    # final result = label
    final_home  = df["home_score"].iloc[-1]
    final_away  = df["away_score"].iloc[-1]
    home_won    = int(final_home > final_away)

    # --- time remaining ------------------------------------------------------
    df["clock_s"]   = df["PCTIMESTRING"].apply(parse_clock)
    df["period"]    = df["PERIOD"].clip(upper=4)
    period_len      = 12 * 60          # 12 min quarters
    df["time_remaining_s"] = (4 - df["period"]) * period_len + df["clock_s"]

    # --- fouls ---------------------------------------------------------------
    # HOMEDESCRIPTION / VISITORDESCRIPTION contain text; count fouls cumulatively
    def count_fouls(desc_col):
        return desc_col.fillna("").str.contains("FOUL", case=False).cumsum()

    df["home_fouls"] = count_fouls(df["HOMEDESCRIPTION"])
    df["away_fouls"] = count_fouls(df["VISITORDESCRIPTION"])

    # --- possession (rough heuristic from description) -----------------------
    def possession_flag(row):
        home_has = not pd.isna(row.get("HOMEDESCRIPTION", None)) and str(row.get("HOMEDESCRIPTION","")) != ""
        away_has = not pd.isna(row.get("VISITORDESCRIPTION", None)) and str(row.get("VISITORDESCRIPTION","")) != ""
        if home_has and not away_has:
            return 1.0
        if away_has and not home_has:
            return 0.0
        return 0.5

    df["possession"] = df.apply(possession_flag, axis=1)

    # --- timeouts (estimate: start at 7, decrement on timeout events) --------
    home_to = 7
    away_to = 7
    home_tos = []
    away_tos = []
    for _, row in df.iterrows():
        desc_h = str(row.get("HOMEDESCRIPTION", "") or "")
        desc_a = str(row.get("VISITORDESCRIPTION", "") or "")
        if "Timeout" in desc_h or "TIMEOUT" in desc_h:
            home_to = max(0, home_to - 1)
        if "Timeout" in desc_a or "TIMEOUT" in desc_a:
            away_to = max(0, away_to - 1)
        home_tos.append(home_to)
        away_tos.append(away_to)

    df["home_timeouts"] = home_tos
    df["away_timeouts"] = away_tos

    # --- assemble feature rows -----------------------------------------------
    features = df[[
        "home_score", "away_score",
        "time_remaining_s", "period",
        "home_fouls", "away_fouls",
        "home_timeouts", "away_timeouts",
        "possession",
    ]].copy()

    features["score_diff"] = features["home_score"] - features["away_score"]
    features["label"]      = home_won
    features["game_id"]    = game_id

    return features[[
        "game_id", "score_diff", "time_remaining_s", "period",
        "home_fouls", "away_fouls", "home_timeouts", "away_timeouts",
        "possession", "label"
    ]]


def main():
    all_rows = []
    for season in SEASONS:
        print(f"\n=== Season {season} ===")
        ids = get_game_ids(season, MAX_GAMES_PER_SEASON)
        print(f"  Found {len(ids)} game IDs, collecting up to {MAX_GAMES_PER_SEASON}")
        for i, gid in enumerate(ids):
            print(f"  [{i+1}/{len(ids)}] {gid}", end=" ", flush=True)
            rows = collect_game(gid)
            if rows is not None:
                all_rows.append(rows)
                print(f"  +{len(rows)} rows")
            time.sleep(DELAY)

    if not all_rows:
        print("No data collected. Check network / nba_api availability.")
        return

    df = pd.concat(all_rows, ignore_index=True)
    df.dropna(inplace=True)
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df):,} rows → {OUTPUT}")
    print(df.describe())


if __name__ == "__main__":
    main()
