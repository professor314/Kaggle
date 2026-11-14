"""Research-driven enhancements (see RESEARCH_IMPROVEMENTS.md).

A. Turn-angle-at-break feature (CMU arXiv:2507.06122 style): the angle between
   successive displacement vectors, summed/maxed around the break, as a
   tracking-native 'cut sharpness'. Test vs outcomes.
B. Prospect value board: RAB percentile vs draft slot; name concrete
   over-valued (early pick, low RAB) and under-valued (late/undrafted, high RAB)
   receivers. New figure.
D. Composite: does RAB + 3-cone beat either alone for predicting career snaps?
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

BREAK_DRILLS = {
    "POST_CORNER_ROUTE_RIGHT", "POST_CORNER_ROUTE_LEFT",
    "COMEBACK_ROUTE_RIGHT", "CURL_ROUTE_RIGHT",
    "SPEED_OUT_ROUTE_LEFT", "SPEED_OUT_LEFT",
    "SLANT_ROUTE_LEFT", "DAGGER_ROUTE_LEFT",
    "SLOT_SAIL_ROUTE_RIGHT", "SLOT_STRIKE_ROUTE_LEFT",
    "IN_ROUTE_LEFT", "HOOK_ROUTE_RIGHT",
}
DT = 0.1
MIN_FRAMES = 25
MIN_PEAK_SPEED = 3.0


def load_wr():
    players = pd.read_csv(DATA / "players.csv")
    wr_ids = set(players.loc[players.nfl_position == "WR", "nfl_id"])
    cols = ["nfl_id", "event_id", "entity_type", "time", "drill_type",
            "drill_name", "attempt", "x", "y", "s", "a", "dir"]
    parts = []
    for ch in pd.read_csv(DATA / "combine_tracking.csv", usecols=cols,
                          chunksize=200_000):
        ch = ch[(ch.drill_type == "SKILL_DRILLS_WR") &
                (ch.entity_type == "PLAYER") &
                (ch.nfl_id.isin(wr_ids)) &
                (ch.drill_name.isin(BREAK_DRILLS))]
        if len(ch):
            parts.append(ch)
    wr = pd.concat(parts, ignore_index=True)
    wr["time"] = pd.to_datetime(wr["time"])
    return wr


def turn_angle_series(x, y):
    """CMU-style per-frame turn angle: change in bearing between successive
    displacement vectors (radians, absolute)."""
    dx = np.diff(x); dy = np.diff(y)
    bearing = np.arctan2(dy, dx)
    turn = np.diff(bearing)
    turn = (turn + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    return np.abs(turn)


def attempt_feats(g):
    g = g.sort_values("time")
    s = g.s.to_numpy(float); x = g.x.to_numpy(float); y = g.y.to_numpy(float)
    n = len(s)
    if n < MIN_FRAMES or np.nanmax(s) < MIN_PEAK_SPEED:
        return None
    lo, hi = int(0.15 * n), int(0.85 * n)
    if hi - lo < 6:
        return None
    b = lo + int(np.nanargmin(s[lo:hi]))
    peak_before = np.nanmax(s[:b + 1])
    win = 8
    post = s[b:min(n, b + win + 1)]
    if len(post) < 3:
        return None
    reaccel = (np.nanmax(post) - np.nanmin(post)) / (len(post) * DT)
    recovery = np.nanmax(post) / peak_before if peak_before > 0 else np.nan
    # turn angle around the break window
    ta = turn_angle_series(x, y)  # length n-2, index i ~ frame i+1
    w0, w1 = max(0, b - win), min(len(ta), b + win)
    ta_win = ta[w0:w1]
    turn_sum = float(np.nansum(ta_win))        # total redirection
    turn_max = float(np.nanmax(ta_win)) if len(ta_win) else np.nan  # sharpest step
    return dict(reaccel=reaccel, recovery=recovery,
                turn_sum=turn_sum, turn_max=turn_max)


def build_player_table():
    wr = load_wr()
    rows = []
    for (nid, ev, drill, att), g in wr.groupby(
            ["nfl_id", "event_id", "drill_name", "attempt"], sort=False):
        f = attempt_feats(g)
        if f:
            f.update(nfl_id=nid, drill_name=drill)
            rows.append(f)
    at = pd.DataFrame(rows)
    comps = ["reaccel", "recovery", "turn_sum", "turn_max"]
    for c in comps:
        at[c + "_z"] = at.groupby("drill_name")[c].transform(
            lambda v: (v - v.mean()) / (v.std(ddof=0) + 1e-9))
    g = at.groupby("nfl_id")
    pl = g[[c + "_z" for c in comps]].mean().reset_index()
    pl["n_attempts"] = g.size().values
    pl = pl[pl.n_attempts >= 3].copy()
    pl["RAB"] = pl[["reaccel_z", "recovery_z"]].mean(axis=1)
    pl["turn_sharpness"] = pl[["turn_sum_z", "turn_max_z"]].mean(axis=1)
    return pl


def outcomes():
    players = pd.read_csv(DATA / "players.csv")
    wr_ids = set(players.loc[players.nfl_position == "WR", "nfl_id"])
    pp = pd.read_csv(DATA / "player_play.csv",
                     usecols=["nfl_id", "separation_at_pass_forward"])
    pp = pp[pp.nfl_id.isin(wr_ids)].dropna()
    sep = pp.groupby("nfl_id").separation_at_pass_forward.agg(
        mean_separation="mean", n_routes="size").reset_index()
    succ = pd.read_csv(DATA / "player_career_successes.csv")[
        ["nfl_id", "career_offensive_snaps"]]
    comb = pd.read_csv(DATA / "combine_results.csv")[
        ["nfl_id", "forty", "three_cone"]]
    meta = players[["nfl_id", "display_name", "draft_overall_pick"]]
    return (sep.merge(succ, on="nfl_id", how="left")
               .merge(comb, on="nfl_id", how="left")
               .merge(meta, on="nfl_id", how="left"))


def c(df, a, b):
    m = df[[a, b]].dropna().astype(float)
    if len(m) < 15:
        return None
    r, p = stats.pearsonr(m[a].to_numpy(), m[b].to_numpy())
    return r, p, len(m)


def main():
    pl = build_player_table()
    out = outcomes()
    df = pl.merge(out, on="nfl_id", how="left")
    df = df[df.n_routes >= 20].copy()
    df.to_csv(OUT / "enhanced_table.csv", index=False)
    print(f"n={len(df)}")

    # A. turn-angle feature vs outcomes
    print("\n=== A. Turn-angle-at-break (CMU-style) vs outcomes ===")
    for a in ["turn_sum_z", "turn_max_z", "turn_sharpness"]:
        for b in ["mean_separation", "career_offensive_snaps"]:
            res = c(df, a, b)
            if res:
                r, p, n = res
                print(f"  {a:16s} ~ {b:24s}: r={r:+.3f} p={p:.3f} n={n}"
                      f"{' *' if p<0.05 else ''}")

    # D. composite RAB + 3-cone -> snaps (both agility-flavored)
    print("\n=== D. RAB + 3-cone composite -> career snaps ===")
    m = df[["RAB", "three_cone", "career_offensive_snaps"]].dropna().astype(float)
    y = m.career_offensive_snaps.to_numpy()
    def r2(cols):
        X = np.c_[np.ones(len(m)), m[cols].to_numpy()]
        bta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ bta
        return 1 - (resid @ resid) / ((y - y.mean()) ** 2).sum()
    print(f"  n={len(m)}")
    print(f"  RAB only          R2={r2(['RAB']):.3f}")
    print(f"  3-cone only       R2={r2(['three_cone']):.3f}")
    print(f"  RAB + 3-cone      R2={r2(['RAB','three_cone']):.3f}")
    # correlation between RAB and 3-cone (are they redundant?)
    res = c(df, "RAB", "three_cone")
    if res:
        print(f"  RAB ~ 3-cone: r={res[0]:+.3f} p={res[1]:.3f} "
              f"(low |r| => complementary)")

    value_board(df)
    print("\nDone. See enhanced_table.csv, fig7_value_board.png, "
          "artifacts/value_board.csv")


def value_board(df):
    d = df.dropna(subset=["display_name", "RAB"]).copy()
    d["RAB_pctile"] = d.RAB.rank(pct=True) * 100
    drafted = d.dropna(subset=["draft_overall_pick"]).copy()
    drafted["draft_pctile"] = (1 - drafted.draft_overall_pick.rank(pct=True)) * 100
    # value gap: RAB percentile minus draft percentile
    drafted["value_gap"] = drafted.RAB_pctile - drafted.draft_pctile
    board = drafted.sort_values("value_gap", ascending=False)
    cols = ["display_name", "draft_overall_pick", "RAB", "RAB_pctile",
            "career_offensive_snaps", "value_gap"]
    board[cols].to_csv(OUT / "value_board.csv", index=False)
    under = board.head(6)   # high RAB relative to draft slot
    over = board.tail(6)    # low RAB relative to draft slot
    print("\n=== B. UNDER-valued by RAB (high burst, later pick) ===")
    print(under[["display_name", "draft_overall_pick", "RAB",
                 "career_offensive_snaps"]].to_string(index=False))
    print("\n=== OVER-valued by RAB (early pick, low burst) ===")
    print(over[["display_name", "draft_overall_pick", "RAB",
                "career_offensive_snaps"]].to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sc = ax.scatter(drafted.draft_overall_pick, drafted.RAB,
                    c=drafted.career_offensive_snaps, cmap="viridis", s=40)
    # annotate the most under/over-valued (top 6 each by value gap)
    labelset = pd.concat([under.head(6), over.head(5)])
    for _, r in labelset.iterrows():
        ax.annotate(r.display_name, (r.draft_overall_pick, r.RAB),
                    fontsize=7, alpha=.85,
                    xytext=(4, 2), textcoords="offset points")
    ax.set_xlim(-10, 300)
    ax.set_xlabel("draft overall pick (left = earlier)")
    ax.set_ylabel("Re-Acceleration Burst (RAB)")
    ax.set_title("Fig 7. Prospect value board: RAB vs draft slot\n"
                 "above-right = high burst drafted late (value); "
                 "below-left = burst-light early picks")
    fig.colorbar(sc, label="career offensive snaps")
    fig.tight_layout(); fig.savefig(FIG / "fig7_value_board.png"); plt.close(fig)


if __name__ == "__main__":
    main()

