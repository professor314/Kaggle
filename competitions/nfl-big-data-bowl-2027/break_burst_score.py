"""Re-Acceleration Burst (RAB): a Combine route-drill metric that predicts which
receivers earn NFL playing time — and a cautionary null result about separation.

Validated findings (see diagnose_components.py for the full component search):

  1. In-game separation is explained by straight-line speed (10-yd split
     r=-0.32, 40 r=-0.27) and short-area agility (3-cone r=-0.37), NOT by any
     break-mechanics feature extracted from the route drills (all |r|<0.17,
     n.s.). The drill that *looks* most like route running does not transfer to
     separation.

  2. However, how explosively a receiver RE-ACCELERATES out of the break does
     track career outcomes: career offensive snaps (reaccel r=+0.32, p=0.001;
     speed-recovery r=+0.36, p=0.0004) and draft capital (r=-0.25, p=0.03).

We therefore define the Re-Acceleration Burst score (RAB) from the two
re-acceleration components, validate it against career snaps, and contrast it
with the separation null result. Everything is reproducible from the public
Combine tracking + player_play + career files.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA = Path(r"C:\Users\profe\kaggle-data\competitions\nfl-big-data-bowl-2027\data")
HERE = Path(__file__).parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
OUT = HERE / "artifacts"; OUT.mkdir(exist_ok=True)

BREAK_DRILLS = {
    "POST_CORNER_ROUTE_RIGHT", "POST_CORNER_ROUTE_LEFT",
    "COMEBACK_ROUTE_RIGHT", "CURL_ROUTE_RIGHT",
    "SPEED_OUT_ROUTE_LEFT", "SPEED_OUT_LEFT",
    "SLANT_ROUTE_LEFT", "DAGGER_ROUTE_LEFT",
    "SLOT_SAIL_ROUTE_RIGHT", "SLOT_STRIKE_ROUTE_LEFT",
    "IN_ROUTE_LEFT", "HOOK_ROUTE_RIGHT",
}
MIN_FRAMES = 25
MIN_PEAK_SPEED = 3.0
MIN_ATTEMPTS = 3
DT = 0.1


def load_wr_drill_tracking() -> pd.DataFrame:
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


def attempt_features(g: pd.DataFrame) -> dict | None:
    g = g.sort_values("time")
    s = g.s.to_numpy(float); a = g.a.to_numpy(float)
    n = len(s)
    if n < MIN_FRAMES or np.nanmax(s) < MIN_PEAK_SPEED:
        return None
    lo, hi = int(0.15 * n), int(0.85 * n)
    if hi - lo < 6:
        return None
    b = lo + int(np.nanargmin(s[lo:hi]))          # break = central speed trough
    peak_before = np.nanmax(s[:b + 1])
    min_speed = s[b]
    win = 8
    pre = s[max(0, b - win):b + 1]
    post = s[b:min(n, b + win + 1)]
    if len(pre) < 3 or len(post) < 3:
        return None
    reaccel_rate = (np.nanmax(post) - np.nanmin(post)) / (len(post) * DT)
    recovery_pct = np.nanmax(post) / peak_before if peak_before > 0 else np.nan
    decel_rate = (np.nanmax(pre) - np.nanmin(pre)) / (len(pre) * DT)
    speed_pct_kept = min_speed / peak_before if peak_before > 0 else np.nan
    return dict(reaccel_rate=reaccel_rate, recovery_pct=recovery_pct,
                decel_rate=decel_rate, speed_pct_kept=speed_pct_kept,
                peak_before=peak_before)


def build_attempts(wr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (nfl_id, ev, drill, att), g in wr.groupby(
            ["nfl_id", "event_id", "drill_name", "attempt"], sort=False):
        f = attempt_features(g)
        if f:
            f.update(nfl_id=nfl_id, drill_name=drill)
            rows.append(f)
    return pd.DataFrame(rows)


def player_scores(at: pd.DataFrame) -> pd.DataFrame:
    comps = ["reaccel_rate", "recovery_pct", "decel_rate", "speed_pct_kept"]
    atz = at.copy()
    for c in comps:  # within-drill normalization cancels drill geometry/scale
        atz[c + "_z"] = atz.groupby("drill_name")[c].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9))
    g = atz.groupby("nfl_id")
    out = g[[c + "_z" for c in comps]].mean().reset_index()
    out["n_attempts"] = g.size().values
    out = out[out.n_attempts >= MIN_ATTEMPTS].copy()
    # RAB = mean of the two validated re-acceleration components
    out["RAB"] = out[["reaccel_rate_z", "recovery_pct_z"]].mean(axis=1)
    return out


def load_outcomes() -> pd.DataFrame:
    players = pd.read_csv(DATA / "players.csv")
    wr_ids = set(players.loc[players.nfl_position == "WR", "nfl_id"])
    pp = pd.read_csv(DATA / "player_play.csv",
                     usecols=["nfl_id", "route_ran", "target",
                              "separation_at_pass_forward"])
    pp = pp[pp.nfl_id.isin(wr_ids)]
    sep = (pp.dropna(subset=["separation_at_pass_forward"])
             .groupby("nfl_id").separation_at_pass_forward
             .agg(mean_separation="mean", n_routes="size").reset_index())
    tgt = pp.groupby("nfl_id").target.mean().rename("target_rate").reset_index()
    succ = pd.read_csv(DATA / "player_career_successes.csv")[
        ["nfl_id", "career_offensive_snaps", "career_games_started"]]
    comb = pd.read_csv(DATA / "combine_results.csv")[
        ["nfl_id", "forty", "ten_yd_split", "three_cone"]]
    meta = players[["nfl_id", "display_name", "draft_overall_pick"]]
    out = (sep.merge(tgt, on="nfl_id", how="outer")
              .merge(succ, on="nfl_id", how="left")
              .merge(comb, on="nfl_id", how="left")
              .merge(meta, on="nfl_id", how="left"))
    return out


def corr(df, a, b):
    m = df[[a, b]].dropna().astype(float)
    if len(m) < 15:
        return None
    r, p = stats.pearsonr(m[a].to_numpy(), m[b].to_numpy())
    return r, p, len(m)


def main():
    print("Loading WR break-drill tracking...")
    wr = load_wr_drill_tracking()
    at = build_attempts(wr)
    at.to_csv(OUT / "wr_break_attempts.csv", index=False)
    scores = player_scores(at)
    out = load_outcomes()
    df = scores.merge(out, on="nfl_id", how="left")
    df = df[df.n_routes >= 20].copy()
    df.to_csv(OUT / "wr_rab_table.csv", index=False)
    print(f"Scored WRs with >=20 in-game routes: {len(df)}")

    print("\n=== KEY RESULTS ===")
    pairs = [
        ("RAB", "career_offensive_snaps"),
        ("RAB", "draft_overall_pick"),
        ("RAB", "target_rate"),
        ("RAB", "mean_separation"),
        ("forty", "mean_separation"),
        ("ten_yd_split", "mean_separation"),
        ("three_cone", "mean_separation"),
        ("forty", "career_offensive_snaps"),
    ]
    results = {}
    for a, b in pairs:
        res = corr(df, a, b)
        if res:
            r, p, n = res
            results[(a, b)] = res
            star = " *" if p < 0.05 else ""
            print(f"  {a:14s} ~ {b:24s}: r={r:+.3f} p={p:.4f} n={n}{star}")

    # partial: RAB ~ snaps controlling for forty (adds beyond stopwatch?)
    m = df[["RAB", "career_offensive_snaps", "forty"]].dropna().astype(float)
    if len(m) > 15:
        def resid(y, x):
            x1 = np.c_[np.ones(len(x)), x]
            beta, *_ = np.linalg.lstsq(x1, y, rcond=None)
            return y - x1 @ beta
        ry = resid(m.career_offensive_snaps.values, m.forty.values)
        rx = resid(m.RAB.values, m.forty.values)
        pr, pp_ = stats.pearsonr(rx, ry)
        print(f"\n  PARTIAL RAB~snaps | forty: r={pr:+.3f} p={pp_:.4f} n={len(m)}")

    make_figures(df, wr, results)
    print("Done.")
    return df


def make_figures(df, wr, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

    # Fig 1: anatomy of a break (speed + accel), mark decel-in and reaccel-out
    ex = wr[wr.drill_name.str.contains("COMEBACK")]
    key = ex.groupby(["nfl_id", "event_id", "attempt"]).size().idxmax()
    one = ex[(ex.nfl_id == key[0]) & (ex.event_id == key[1]) &
             (ex.attempt == key[2])].sort_values("time").reset_index(drop=True)
    t = np.arange(len(one)) * DT
    b = int(one.s.iloc[int(.15*len(one)):int(.85*len(one))].idxmin())
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(t, one.s, color="C0", lw=2, label="speed")
    ax1.axvline(b*DT, ls="--", color="grey")
    ax1.annotate("break", xy=(b*DT, one.s.iloc[b]),
                 xytext=(b*DT+0.5, one.s.max()*0.55),
                 arrowprops=dict(arrowstyle="->", color="grey"))
    ax1.axvspan(max(0, (b-8)*DT), b*DT, color="C3", alpha=.1)
    ax1.axvspan(b*DT, min(t[-1], (b+8)*DT), color="C2", alpha=.15)
    ax1.text((b-4)*DT, one.s.max()*0.95, "decel in", color="C3", ha="center")
    ax1.text((b+4)*DT, one.s.max()*0.95, "re-accel out", color="C2", ha="center")
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("speed", color="C0")
    ax2 = ax1.twinx(); ax2.plot(t, one.a, color="C1", alpha=.5)
    ax2.set_ylabel("acceleration", color="C1")
    ax1.set_title("Fig 1. Anatomy of a route break (Combine COMEBACK drill)")
    fig.tight_layout(); fig.savefig(FIG / "fig1_break_anatomy.png"); plt.close(fig)

    # Fig 2: the null result — break metrics vs separation, speed vs separation
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    m = df[["RAB", "mean_separation"]].dropna()
    r, p, _ = results.get(("RAB", "mean_separation"), (np.nan, np.nan, 0))
    axes[0].scatter(m.RAB, m.mean_separation, s=26, alpha=.7, color="C0")
    axes[0].set_xlabel("Re-Acceleration Burst (Combine)")
    axes[0].set_ylabel("mean in-game separation (yd)")
    axes[0].set_title(f"RAB vs separation\nr={r:+.2f}, p={p:.2f} (n.s.)")
    m2 = df[["ten_yd_split", "mean_separation"]].dropna()
    r2, p2, _ = results.get(("ten_yd_split", "mean_separation"),
                            (np.nan, np.nan, 0))
    axes[1].scatter(m2.ten_yd_split, m2.mean_separation, s=26, alpha=.7,
                    color="C3")
    z = np.polyfit(m2.ten_yd_split, m2.mean_separation, 1)
    xs = np.linspace(m2.ten_yd_split.min(), m2.ten_yd_split.max(), 40)
    axes[1].plot(xs, np.polyval(z, xs), color="k", lw=1)
    axes[1].set_xlabel("10-yard split (s) — lower = faster")
    axes[1].set_ylabel("mean in-game separation (yd)")
    axes[1].set_title(f"Straight-line speed vs separation\nr={r2:+.2f}, p={p2:.3f}")
    fig.suptitle("Fig 2. Separation is a speed story, not a cut-mechanics story")
    fig.tight_layout(); fig.savefig(FIG / "fig2_separation_null.png"); plt.close(fig)

    # Fig 3: RAB vs career snaps (the payoff)
    m = df[["RAB", "career_offensive_snaps", "display_name"]].dropna()
    r, p, _ = results.get(("RAB", "career_offensive_snaps"),
                          (np.nan, np.nan, 0))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(m.RAB, m.career_offensive_snaps, s=30, alpha=.75)
    z = np.polyfit(m.RAB, m.career_offensive_snaps, 1)
    xs = np.linspace(m.RAB.min(), m.RAB.max(), 40)
    ax.plot(xs, np.polyval(z, xs), color="C3", lw=1.5)
    ax.set_xlabel("Re-Acceleration Burst score (Combine)")
    ax.set_ylabel("career offensive snaps")
    ax.set_title(f"Fig 3. Re-acceleration out of the break predicts NFL usage\n"
                 f"r={r:+.2f}, p={p:.3f}, n={len(m)}")
    fig.tight_layout(); fig.savefig(FIG / "fig3_rab_vs_snaps.png"); plt.close(fig)

    # Fig 4: top/bottom WRs by RAB
    d = df.dropna(subset=["display_name"]).sort_values("RAB", ascending=False)
    sel = pd.concat([d.head(8), d.tail(8)])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(sel.display_name, sel.RAB, color=["C2"]*8 + ["C1"]*8)
    ax.set_xlabel("Re-Acceleration Burst score"); ax.invert_yaxis()
    ax.set_title("Fig 4. Highest and lowest RAB receivers (2023-25 classes)")
    fig.tight_layout(); fig.savefig(FIG / "fig4_top_bottom.png"); plt.close(fig)
    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    main()
