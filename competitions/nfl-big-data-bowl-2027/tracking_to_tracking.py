"""Tracking-to-tracking linkage: does the Combine Re-Acceleration Burst (RAB)
reappear in live NFL games?

For each WR breaking-route play, we take the window from ball_snap to pass_arrival
(or pass_forward), locate the break as the central speed trough, and measure
re-acceleration out of the break with the SAME definition used on Combine drills.
We average to an in-game RAB per receiver and correlate it with the Combine RAB.

This is the strongest form of the Combine->NFL link: tracking to tracking.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA = Path(r"C:\Users\profe\kaggle-data\competitions\nfl-big-data-bowl-2027\data")
HERE = Path(__file__).parent
OUT = HERE / "artifacts"; OUT.mkdir(exist_ok=True)
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

GAME_FILES = ["game_tracking_2023.csv", "game_tracking_2024.csv",
              "game_tracking_2025.csv"]
BREAKING = {"OUT", "IN", "CORNER", "POST", "SLANT", "COMEBACK", "CURL", "HITCH"}
DT = 0.1
MIN_FRAMES = 25
MIN_PEAK_SPEED = 3.0
MIN_PLAYS = 10  # per-WR minimum in-game breaking routes to score


def reaccel_out_of_break(s: np.ndarray) -> float | None:
    """Same re-acceleration feature used on Combine drills: speed regained in an
    8-frame window out of the central speed trough."""
    n = len(s)
    if n < MIN_FRAMES or np.nanmax(s) < MIN_PEAK_SPEED:
        return None
    lo, hi = int(0.15 * n), int(0.85 * n)
    if hi - lo < 6:
        return None
    b = lo + int(np.nanargmin(s[lo:hi]))
    win = 8
    post = s[b:min(n, b + win + 1)]
    if len(post) < 3:
        return None
    return (np.nanmax(post) - np.nanmin(post)) / (len(post) * DT)


def load_breaking_keys(wr_ids):
    pp = pd.read_csv(DATA / "player_play.csv",
                     usecols=["game_id", "play_id", "nfl_id", "route_ran"])
    pp = pp[(pp.nfl_id.isin(wr_ids)) & (pp.route_ran.isin(BREAKING))]
    return pp[["game_id", "play_id", "nfl_id"]].drop_duplicates()


def ingame_rab(wr_ids):
    keys = load_breaking_keys(wr_ids)
    key_set = set(map(tuple, keys.to_numpy()))
    per_play = []   # (nfl_id, reaccel)
    cols = ["game_id", "play_id", "nfl_id", "time", "s", "event"]
    for fn in GAME_FILES:
        buf = {}  # (game,play,nfl) -> list of (time, s) accumulate across chunks
        for ch in pd.read_csv(DATA / fn, usecols=cols, chunksize=1_000_000):
            ch = ch[ch.nfl_id.isin(wr_ids)]
            if not len(ch):
                continue
            # keep only rows whose (game,play,nfl) is a target breaking route
            tup = list(zip(ch.game_id, ch.play_id, ch.nfl_id))
            mask = [t in key_set for t in tup]
            ch = ch[mask]
            for (g, p, nid), grp in ch.groupby(["game_id", "play_id", "nfl_id"],
                                               sort=False):
                buf.setdefault((g, p, nid), []).append(
                    grp[["time", "s"]].to_numpy())
        # finalize this season's plays
        for (g, p, nid), chunks in buf.items():
            arr = np.concatenate(chunks)
            order = np.argsort(arr[:, 0])
            s = arr[order, 1].astype(float)
            r = reaccel_out_of_break(s)
            if r is not None:
                per_play.append((nid, r))
        print(f"  {fn}: cumulative plays scored = {len(per_play)}")
    pp_df = pd.DataFrame(per_play, columns=["nfl_id", "ingame_reaccel"])
    agg = (pp_df.groupby("nfl_id")
           .agg(n_plays=("ingame_reaccel", "size"),
                ingame_reaccel=("ingame_reaccel", "mean")).reset_index())
    agg = agg[agg.n_plays >= MIN_PLAYS]
    # standardize to an in-game RAB (z-score across players)
    agg["ingame_RAB"] = ((agg.ingame_reaccel - agg.ingame_reaccel.mean())
                         / agg.ingame_reaccel.std(ddof=0))
    return agg


def main():
    rab = pd.read_csv(OUT / "wr_rab_table.csv")
    wr_ids = set(rab.nfl_id)
    print(f"Scored WRs (Combine RAB): {len(wr_ids)}")
    print("Measuring in-game re-acceleration across 3 seasons (streaming)...")
    ig = ingame_rab(wr_ids)
    print(f"WRs with >= {MIN_PLAYS} in-game breaking routes: {len(ig)}")

    df = rab[["nfl_id", "display_name", "RAB",
              "career_offensive_snaps"]].merge(ig, on="nfl_id", how="inner")
    df.to_csv(OUT / "combine_vs_ingame_rab.csv", index=False)

    m = df[["RAB", "ingame_RAB"]].dropna().astype(float)
    r, p = stats.pearsonr(m.RAB, m.ingame_RAB)
    rs, ps = stats.spearmanr(m.RAB, m.ingame_RAB)
    print(f"\n=== TRACKING-TO-TRACKING LINKAGE ===")
    print(f"  Combine RAB ~ in-game RAB: Pearson r={r:+.3f} (p={p:.4f}), "
          f"Spearman rho={rs:+.3f} (p={ps:.4f}), n={len(m)}")

    # does in-game re-accel also track snaps? (sanity)
    m2 = df[["ingame_RAB", "career_offensive_snaps"]].dropna().astype(float)
    r2, p2 = stats.pearsonr(m2.ingame_RAB, m2.career_offensive_snaps)
    print(f"  in-game RAB ~ career snaps: r={r2:+.3f} (p={p2:.4f}), n={len(m2)}")

    _fig(df, r, p)
    print("Done -> artifacts/combine_vs_ingame_rab.csv, fig7")


def _fig(df, r, p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
    m = df[["RAB", "ingame_RAB", "display_name"]].dropna()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(m.RAB, m.ingame_RAB, s=30, alpha=.75)
    z = np.polyfit(m.RAB, m.ingame_RAB, 1)
    xs = np.linspace(m.RAB.min(), m.RAB.max(), 40)
    ax.plot(xs, np.polyval(z, xs), color="C3", lw=1.5)
    ax.set_xlabel("Combine RAB (route drills)")
    ax.set_ylabel("in-game RAB (live NFL breaking routes)")
    ax.set_title(f"Fig A1 (exploratory). Combine RAB vs naive in-game RAB\n"
                 f"Pearson r={r:+.2f}, p={p:.3f}, n={len(m)} — not significant")
    fig.tight_layout(); fig.savefig(FIG / "figA1_tracking_exploratory.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

