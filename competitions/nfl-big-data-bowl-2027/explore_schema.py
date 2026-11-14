"""Explore NFL Big Data Bowl 2027 data schema.

Loads small files fully; samples the large tracking files (nrows) to get schema
and value ranges without exhausting memory.
"""
import pandas as pd
from pathlib import Path

DATA = Path(r"C:\Users\profe\kaggle-data\competitions\nfl-big-data-bowl-2027\data")

SMALL = [
    "combine_results.csv",
    "games.csv",
    "players.csv",
    "player_career_successes.csv",
]
LARGE = [
    "combine_tracking.csv",
    "player_play.csv",
    "game_tracking_2023.csv",
]


def describe(name, df, sampled=False):
    print("=" * 70)
    print(f"{name}  {'(SAMPLED)' if sampled else ''}")
    print(f"shape (loaded): {df.shape}")
    print("-" * 70)
    print("columns + dtypes:")
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        ex = df[c].dropna().iloc[0] if df[c].notna().any() else None
        print(f"  {c:30s} {str(df[c].dtype):10s} nunique={nun:<8} ex={ex!r}")
    print()


def main():
    for name in SMALL:
        p = DATA / name
        if not p.exists():
            print(f"MISSING: {name}")
            continue
        df = pd.read_csv(p)
        describe(name, df)

    for name in LARGE:
        p = DATA / name
        if not p.exists():
            print(f"MISSING: {name}")
            continue
        df = pd.read_csv(p, nrows=200_000)
        describe(name, df, sampled=True)


if __name__ == "__main__":
    main()
