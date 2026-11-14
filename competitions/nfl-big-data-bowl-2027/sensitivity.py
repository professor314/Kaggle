"""Sensitivity of the headline RAB~snaps finding to break-window parameters.

Re-derives RAB under a grid of (trim fraction, window size) choices and reports
the RAB~career-snaps correlation for each, to show the result is not an artifact
of one arbitrary parameterization.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA = Path(r"C:\Users\profe\kaggle-data\competitions\nfl-big-data-bowl-2027\data")
HERE = Path(__file__).parent
OUT = HERE / "artifacts"

BREAK_DRILLS = {
    "POST_CORNER_ROUTE_RIGHT", "POST_CORNER_ROUTE_LEFT",
    "COMEBACK_ROUTE_RIGHT", "CURL_ROUTE_RIGHT",
    "SPEED_OUT_ROUTE_LEFT", "SPEED_OUT_LEFT",
    "SLANT_ROUTE_LEFT", "DAGGER_ROUTE_LEFT",
    "SLOT_SAIL_ROUTE_RIGHT", "SLOT_STRIKE_ROUTE_LEFT",
    "IN_ROUTE_LEFT", "HOOK_ROUTE_RIGHT",
}
DT = 0.1
MIN_PEAK_SPEED = 3.0


def load():
    players = pd.read_csv(DATA / "players.csv")
    wr_ids = set(players.loc[players.nfl_position == "WR", "nfl_id"])
    cols = ["nfl_id", "event_id", "entity_type", "time", "drill_type",
            "drill_name", "attempt", "s"]
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


def feat(s, trim, win):
    n = len(s)
    if n < 25 or np.nanmax(s) < MIN_PEAK_SPEED:
        return None
    lo, hi = int(trim * n), int((1 - trim) * n)
    if hi - lo < 6:
        return None
    b = lo + int(np.nanargmin(s[lo:hi]))
    peak_before = np.nanmax(s[:b + 1])
    post = s[b:min(n, b + win + 1)]
    if len(post) < 3:
        return None
    reaccel = (np.nanmax(post) - np.nanmin(post)) / (len(post) * DT)
    recovery = np.nanmax(post) / peak_before if peak_before > 0 else np.nan
    return reaccel, recovery


def rab_for(wr, snaps, trim, win):
    rows = []
    for (nid, ev, drill, att), g in wr.groupby(
            ["nfl_id", "event_id", "drill_name", "attempt"], sort=False):
        s = g.sort_values("time").s.to_numpy(float)
        f = feat(s, trim, win)
        if f:
            rows.append((nid, drill, f[0], f[1]))
    at = pd.DataFrame(rows, columns=["nfl_id", "drill_name",
                                     "reaccel", "recovery"])
    for c in ["reaccel", "recovery"]:
        at[c + "_z"] = at.groupby("drill_name")[c].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9))
    g = at.groupby("nfl_id")
    pl = g[["reaccel_z", "recovery_z"]].mean()
    pl["n"] = g.size()
    pl = pl[pl.n >= 3].reset_index()
    pl["RAB"] = pl[["reaccel_z", "recovery_z"]].mean(axis=1)
    m = pl.merge(snaps, on="nfl_id").dropna()
    r, p = stats.pearsonr(m.RAB, m.career_offensive_snaps)
    return r, p, len(m)


def main():
    wr = load()
    rab_tbl = pd.read_csv(OUT / "wr_rab_table.csv")
    snaps = rab_tbl[["nfl_id", "career_offensive_snaps"]]
    # restrict to WRs in the main analysis (>=20 in-game routes) for comparability
    valid = set(rab_tbl.nfl_id)
    wr = wr[wr.nfl_id.isin(valid)]

    print("Break-window sensitivity of RAB ~ career snaps:")
    print(f"{'trim':>6}{'win':>6}{'r':>9}{'p':>9}{'n':>5}")
    results = []
    for trim in [0.10, 0.15, 0.20, 0.25]:
        for win in [5, 8, 10, 12]:
            r, p, n = rab_for(wr, snaps, trim, win)
            results.append((trim, win, r, p, n))
            print(f"{trim:6.2f}{win:6d}{r:9.3f}{p:9.4f}{n:5d}")
    res = pd.DataFrame(results, columns=["trim", "win", "r", "p", "n"])
    res.to_csv(OUT / "sensitivity.csv", index=False)
    print(f"\nr range: [{res.r.min():.3f}, {res.r.max():.3f}]; "
          f"all p<0.05: {bool((res.p < 0.05).all())}; "
          f"median r={res.r.median():.3f}")


if __name__ == "__main__":
    main()
