"""Diagnostic: find which break-movement components actually track in-game
separation, using within-drill normalization and breaking-route separation.

This is exploratory. We print correlations for each candidate component so the
final metric is built from signal, not assumption.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA = Path(r"C:\Users\profe\kaggle-data\competitions\nfl-big-data-bowl-2027\data")
HERE = Path(__file__).parent

BREAK_DRILLS = {
    "POST_CORNER_ROUTE_RIGHT", "POST_CORNER_ROUTE_LEFT",
    "COMEBACK_ROUTE_RIGHT", "CURL_ROUTE_RIGHT",
    "SPEED_OUT_ROUTE_LEFT", "SPEED_OUT_LEFT",
    "SLANT_ROUTE_LEFT", "DAGGER_ROUTE_LEFT",
    "SLOT_SAIL_ROUTE_RIGHT", "SLOT_STRIKE_ROUTE_LEFT",
    "IN_ROUTE_LEFT", "HOOK_ROUTE_RIGHT",
}
# in-game routes that require a sharp break (vs GO/HITCH/SCREEN/FLAT)
BREAKING_INGAME = {"OUT", "IN", "CORNER", "POST", "SLANT", "CROSS"}
MIN_FRAMES = 25
MIN_PEAK_SPEED = 3.0


def load_wr_drill_tracking():
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


def ang_diff(d2, d1):
    """smallest absolute angular difference in degrees"""
    d = np.abs(d2 - d1) % 360
    return np.where(d > 180, 360 - d, d)


def attempt_features(g):
    g = g.sort_values("time")
    s = g.s.to_numpy(float); a = g.a.to_numpy(float); dirn = g.dir.to_numpy(float)
    n = len(s)
    if n < MIN_FRAMES or np.nanmax(s) < MIN_PEAK_SPEED:
        return None
    lo, hi = int(0.15 * n), int(0.85 * n)
    if hi - lo < 6:
        return None
    b = lo + int(np.nanargmin(s[lo:hi]))
    peak_before = np.nanmax(s[:b + 1])
    min_speed = s[b]
    win = 8
    pre = s[max(0, b - win):b + 1]
    post = s[b:min(n, b + win + 1)]
    if len(pre) < 3 or len(post) < 3:
        return None
    dt = 0.1
    # rate-based features
    decel_rate = (np.nanmax(pre) - np.nanmin(pre)) / (len(pre) * dt)   # speed/s
    reaccel_rate = (np.nanmax(post) - np.nanmin(post)) / (len(post) * dt)
    speed_pct_kept = min_speed / peak_before if peak_before > 0 else np.nan
    recovery_pct = (np.nanmax(post) / peak_before) if peak_before > 0 else np.nan
    # direction change magnitude across the break window
    dchg = ang_diff(dirn[min(n-1, b+win)], dirn[max(0, b-win)])
    peak_accel = np.nanmax(a[max(0, b-win):min(n, b+win+1)])
    # time in the cut: frames where speed < 70% of peak_before
    thr = 0.7 * peak_before
    time_in_cut = np.sum(s[lo:hi] < thr) * dt
    return dict(peak_before=peak_before, min_speed=min_speed,
                decel_rate=decel_rate, reaccel_rate=reaccel_rate,
                speed_pct_kept=speed_pct_kept, recovery_pct=recovery_pct,
                dir_change=dchg, peak_accel=peak_accel, time_in_cut=time_in_cut)


def main():
    wr = load_wr_drill_tracking()
    rows = []
    for (nfl_id, ev, drill, att), g in wr.groupby(
            ["nfl_id", "event_id", "drill_name", "attempt"], sort=False):
        f = attempt_features(g)
        if f:
            f.update(nfl_id=nfl_id, drill_name=drill)
            rows.append(f)
    at = pd.DataFrame(rows)
    print("valid attempts:", len(at), "players:", at.nfl_id.nunique())

    comps = ["decel_rate", "reaccel_rate", "speed_pct_kept", "recovery_pct",
             "dir_change", "peak_accel", "time_in_cut", "peak_before"]

    # within-drill z-score so geometry/scale cancels
    atz = at.copy()
    for c in comps:
        atz[c + "_z"] = atz.groupby("drill_name")[c].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9))
    zc = [c + "_z" for c in comps]
    player = atz.groupby("nfl_id")[zc].mean().reset_index()
    player["n"] = atz.groupby("nfl_id").size().values
    player = player[player.n >= 3]

    # in-game separation, all routes AND breaking routes only
    players = pd.read_csv(DATA / "players.csv")
    wr_ids = set(players.loc[players.nfl_position == "WR", "nfl_id"])
    pp = pd.read_csv(DATA / "player_play.csv",
                     usecols=["nfl_id", "route_ran",
                              "separation_at_pass_forward"])
    pp = pp[pp.nfl_id.isin(wr_ids)].dropna(subset=["separation_at_pass_forward"])
    sep_all = pp.groupby("nfl_id").separation_at_pass_forward.agg(["mean", "size"])
    sep_all.columns = ["sep_all", "n_all"]
    ppb = pp[pp.route_ran.isin(BREAKING_INGAME)]
    sep_brk = ppb.groupby("nfl_id").separation_at_pass_forward.agg(["mean", "size"])
    sep_brk.columns = ["sep_break", "n_break"]

    df = (player.merge(sep_all, on="nfl_id")
                .merge(sep_brk, on="nfl_id", how="left"))
    df = df[df.n_all >= 20]
    print("linked WRs:", len(df))

    # composite candidate: mean of the two components that leaned positive
    df["BBS"] = df[["speed_pct_kept_z", "decel_rate_z"]].mean(axis=1)

    # add stopwatch + draft + target-rate outcomes
    combine = pd.read_csv(DATA / "combine_results.csv")[
        ["nfl_id", "forty", "three_cone", "short_shuttle", "ten_yd_split"]]
    pl = pd.read_csv(DATA / "players.csv")[["nfl_id", "draft_overall_pick"]]
    succ = pd.read_csv(DATA / "player_career_successes.csv")[
        ["nfl_id", "career_offensive_snaps", "career_games_started"]]
    # target rate per WR
    ppt = pd.read_csv(DATA / "player_play.csv", usecols=["nfl_id", "target"])
    ppt = ppt[ppt.nfl_id.isin(wr_ids)]
    tgt = ppt.groupby("nfl_id").target.mean().rename("target_rate")
    df = (df.merge(combine, on="nfl_id", how="left")
            .merge(pl, on="nfl_id", how="left")
            .merge(succ, on="nfl_id", how="left")
            .merge(tgt, on="nfl_id", how="left"))

    def rep(a, b):
        m = df[[a, b]].dropna().astype(float)
        if len(m) < 15:
            return
        r, p = stats.pearsonr(m[a].to_numpy(), m[b].to_numpy())
        flag = " *" if p < 0.05 else ""
        print(f"  {a:18s} ~ {b:22s}: r={r:+.3f} p={p:.3f} n={len(m)}{flag}")

    print("\n--- break components (within-drill z) vs outcomes ---")
    for c in zc + ["BBS"]:
        for sep in ["sep_all", "sep_break", "target_rate",
                    "career_offensive_snaps", "draft_overall_pick"]:
            rep(c, sep)

    print("\n--- stopwatch baselines vs outcomes ---")
    for c in ["forty", "ten_yd_split", "three_cone", "short_shuttle"]:
        for sep in ["sep_all", "sep_break", "target_rate",
                    "career_offensive_snaps", "draft_overall_pick"]:
            rep(c, sep)

    df.to_csv(HERE / "artifacts" / "diag_components.csv", index=False)


if __name__ == "__main__":
    main()
