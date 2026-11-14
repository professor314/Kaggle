"""Feasibility probe for the WR 'cut quality' metric.

1. Load WR skill-drill tracking (route drills). Inspect s/a/dir over an attempt.
2. Confirm we can detect a 'break' (speed trough / direction change).
3. Check in-game separation data availability per WR in player_play.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(r"C:\Users\profe\kaggle-data\competitions\nfl-big-data-bowl-2027\data")
players = pd.read_csv(DATA / "players.csv")
wr_ids = set(players.loc[players.nfl_position == "WR", "nfl_id"])

# stream combine_tracking, keep only WR route drills
cols = ["nfl_id", "event_id", "entity_type", "time", "drill_type",
        "drill_name", "attempt", "x", "y", "s", "a", "dir"]
it = pd.read_csv(DATA / "combine_tracking.csv", usecols=cols,
                 chunksize=100_000)
parts = []
for ch in it:
    ch = ch[(ch.drill_type == "SKILL_DRILLS_WR") &
            (ch.entity_type == "PLAYER") & (ch.nfl_id.isin(wr_ids))]
    if len(ch):
        parts.append(ch)
wr = pd.concat(parts, ignore_index=True)
print("WR route-drill rows:", len(wr))
print("WR players:", wr.nfl_id.nunique())
print("route drills:", wr.drill_name.nunique())
print(wr.drill_name.value_counts().head(15))

# One example attempt: inspect motion profile
wr["time"] = pd.to_datetime(wr["time"])
g = wr.sort_values("time").groupby(["nfl_id", "event_id", "attempt"])
sizes = g.size()
print("\nframes per attempt: min/med/max =",
      sizes.min(), int(sizes.median()), sizes.max())

# take a POST_CORNER or COMEBACK attempt (has a sharp break)
ex = wr[wr.drill_name.str.contains("COMEBACK")].copy()
if len(ex):
    key = ex.groupby(["nfl_id", "event_id", "attempt"]).size().idxmax()
    one = ex[(ex.nfl_id == key[0]) & (ex.event_id == key[1]) &
             (ex.attempt == key[2])].sort_values("time")
    print(f"\nExample COMEBACK attempt nfl_id={key[0]} frames={len(one)}")
    print("speed range:", round(one.s.min(), 2), "->", round(one.s.max(), 2))
    print("min speed (break trough):", round(one.s.min(), 2))
    print("max decel (min a):", round(one.a.min(), 2),
          " max accel:", round(one.a.max(), 2))
    print("dir change total (deg):",
          round(np.abs(np.diff(one.dir.values)).sum(), 1))

# In-game separation availability
pp = pd.read_csv(DATA / "player_play.csv",
                 usecols=["nfl_id", "route_ran", "separation_at_pass_forward",
                          "target", "rec_yards", "yards_after_catch"])
wr_pp = pp[pp.nfl_id.isin(wr_ids)]
print("\n--- IN-GAME (player_play) WR coverage ---")
print("WR play rows:", len(wr_pp))
print("rows w/ route_ran:", wr_pp.route_ran.notna().sum())
print("rows w/ separation:", wr_pp.separation_at_pass_forward.notna().sum())
print("route_ran distribution:")
print(wr_pp.route_ran.value_counts())
sep_by_wr = wr_pp.dropna(subset=["separation_at_pass_forward"]).groupby("nfl_id").size()
print("\nWRs with >=20 separation measurements:", (sep_by_wr >= 20).sum())
print("WRs with any separation:", sep_by_wr.shape[0])

# overlap: WRs with both drill tracking AND in-game separation
both = wr_ids & set(wr.nfl_id) & set(sep_by_wr.index)
print("WRs with BOTH drill tracking and in-game separation:", len(both))
