"""Probe: can we measure in-game re-acceleration out of route breaks from
game_tracking, for the same WRs we scored at the Combine?

Strategy: use the smallest game file (2023), stream it, keep only rows for our
scored WRs on breaking routes (join game_id+play_id+nfl_id -> player_play route).
Inspect frames per play and whether a speed trough (break) is detectable.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(r"C:\Users\profe\kaggle-data\competitions\nfl-big-data-bowl-2027\data")
HERE = Path(__file__).parent

rab = pd.read_csv(HERE / "artifacts" / "wr_rab_table.csv")
wr_ids = set(rab.nfl_id)
print("scored WRs:", len(wr_ids))

# breaking routes from player_play
BREAKING = {"OUT", "IN", "CORNER", "POST", "SLANT", "COMEBACK", "CURL", "HITCH"}
pp = pd.read_csv(DATA / "player_play.csv",
                 usecols=["game_id", "play_id", "nfl_id", "route_ran"])
pp = pp[(pp.nfl_id.isin(wr_ids)) & (pp.route_ran.isin(BREAKING))]
# key set of (game_id, play_id, nfl_id) we care about
keys = set(map(tuple, pp[["game_id", "play_id", "nfl_id"]].to_numpy()))
print("WR breaking-route player-plays (all seasons):", len(keys))

# stream the 2023 game tracking, keep only our WR rows
cols = ["game_id", "play_id", "nfl_id", "time", "s", "a", "dir", "event"]
parts = []
rows_scanned = 0
for ch in pd.read_csv(DATA / "game_tracking_2023.csv", usecols=cols,
                      chunksize=500_000):
    rows_scanned += len(ch)
    ch = ch[ch.nfl_id.isin(wr_ids)]
    if len(ch):
        parts.append(ch)
g = pd.concat(parts, ignore_index=True)
print(f"scanned {rows_scanned} rows of 2023; kept {len(g)} WR rows")
print("events present:", sorted(g.event.dropna().unique())[:20])

# merge route info, keep breaking-route plays
g = g.merge(pp[["game_id", "play_id", "nfl_id", "route_ran"]],
            on=["game_id", "play_id", "nfl_id"], how="inner")
print("WR breaking-route tracking rows (2023):", len(g))
nplays = g.groupby(["game_id", "play_id", "nfl_id"]).ngroups
print("distinct WR breaking-route plays (2023):", nplays)
fps = g.groupby(["game_id", "play_id", "nfl_id"]).size()
print("frames/play: min/med/max =", fps.min(), int(fps.median()), fps.max())
print("WRs with >=1 breaking-route play in 2023:", g.nfl_id.nunique())
