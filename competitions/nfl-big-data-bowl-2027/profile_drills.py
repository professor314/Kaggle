"""Profile combine_tracking drills/positions and join coverage to outcomes."""
import pandas as pd
from pathlib import Path

DATA = Path(r"C:\Users\profe\kaggle-data\competitions\nfl-big-data-bowl-2027\data")

# Full combine_tracking is 80MB - load only the categorical cols to profile drills
ct = pd.read_csv(
    DATA / "combine_tracking.csv",
    usecols=["draft_year", "nfl_id", "entity_type", "drill_type", "drill_name", "attempt"],
)
print("combine_tracking rows:", len(ct))
print("entity_type:", ct.entity_type.value_counts().to_dict())
print("draft_years:", sorted(ct.draft_year.unique()))
print("\ndrill_type counts:")
print(ct.drill_type.value_counts())
print("\nunique players per drill_type:")
print(ct.groupby("drill_type").nfl_id.nunique())
print("\ndrill_name (top 50):")
print(ct.drill_name.value_counts().head(50))

players = pd.read_csv(DATA / "players.csv")
combine = pd.read_csv(DATA / "combine_results.csv")
success = pd.read_csv(DATA / "player_career_successes.csv")

ct_players = set(ct.nfl_id.unique())
print("\n--- JOIN COVERAGE ---")
print("combine_tracking unique players:", len(ct_players))
print("players.csv positions:")
print(players.nfl_position.value_counts())

# position of tracked players
pos = players.set_index("nfl_id").nfl_position
tracked_pos = pd.Series([pos.get(i) for i in ct_players]).value_counts()
print("\npositions of tracked players:")
print(tracked_pos)

# Which drill types map to which positions
m = ct.merge(players[["nfl_id", "nfl_position"]], on="nfl_id", how="left")
print("\ndrill_type x position (unique players):")
print(m.groupby(["drill_type", "nfl_position"]).nfl_id.nunique().unstack(fill_value=0))
