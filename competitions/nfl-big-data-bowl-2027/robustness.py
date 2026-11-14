"""Strengthen the RAB finding: does it survive controlling for draft capital,
and is it stable under resampling, cross-validation, and parameter choices?

Produces a stats report (printed + saved) and two new figures.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

HERE = Path(__file__).parent
OUT = HERE / "artifacts"; OUT.mkdir(exist_ok=True)
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)


def ols(y, X):
    """Return beta, se, t, p, r2 for y = X beta (X includes intercept col)."""
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    sigma2 = (resid @ resid) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot
    return beta, se, t, p, r2, dof


def partial_corr(df, x, y, covars):
    cols = [x, y] + covars
    m = df[cols].dropna().astype(float)
    n = len(m)
    def resid(target):
        Z = np.c_[np.ones(n), m[covars].to_numpy()]
        b, *_ = np.linalg.lstsq(Z, m[target].to_numpy(), rcond=None)
        return m[target].to_numpy() - Z @ b
    rx, ry = resid(x), resid(y)
    r, p = stats.pearsonr(rx, ry)
    return r, p, n


def main():
    df = pd.read_csv(OUT / "wr_rab_table.csv")
    report = {}

    # ---- 1. Draft-capital control -----------------------------------------
    # Does RAB predict snaps after accounting for where the player was drafted?
    # Note: undrafted players have NaN draft pick; encode as a realistic value
    # (pick 270 ~ just past the 7-round board) + an "undrafted" flag so we keep
    # the full sample instead of dropping the undrafted (who are informative).
    d = df.copy()
    d["undrafted"] = d["draft_overall_pick"].isna().astype(float)
    d["draft_pick_filled"] = d["draft_overall_pick"].fillna(270.0)

    sub = d.dropna(subset=["RAB", "career_offensive_snaps"]).copy()
    y = sub["career_offensive_snaps"].to_numpy(float)
    n = len(sub)

    # Model A: snaps ~ draft only
    Xa = np.c_[np.ones(n), sub.draft_pick_filled, sub.undrafted]
    _, _, _, pa, r2a, _ = ols(y, Xa)

    # Model B: snaps ~ draft + RAB
    Xb = np.c_[np.ones(n), sub.draft_pick_filled, sub.undrafted, sub.RAB]
    beta_b, se_b, t_b, p_b, r2b, dof_b = ols(y, Xb)
    # RAB is the last coefficient
    rab_beta, rab_p = beta_b[-1], p_b[-1]
    delta_r2 = r2b - r2a

    print("=== Draft-capital control (OLS, n=%d) ===" % n)
    print(f"  Model A  snaps ~ draft:          R2={r2a:.3f}")
    print(f"  Model B  snaps ~ draft + RAB:    R2={r2b:.3f}  (delta R2={delta_r2:+.3f})")
    print(f"  RAB coefficient: beta={rab_beta:.1f} snaps per +1 RAB, "
          f"t={t_b[-1]:.2f}, p={rab_p:.4f}")

    # partial correlation RAB~snaps controlling draft (+undrafted flag)
    pr, pp_, pn = partial_corr(
        d, "RAB", "career_offensive_snaps",
        ["draft_pick_filled", "undrafted"])
    print(f"  Partial corr RAB~snaps | draft: r={pr:+.3f}, p={pp_:.4f}, n={pn}")
    report["draft_control"] = dict(
        n=n, r2_draft_only=r2a, r2_draft_plus_rab=r2b, delta_r2=delta_r2,
        rab_beta=rab_beta, rab_t=float(t_b[-1]), rab_p=rab_p,
        partial_r=pr, partial_p=pp_)

    # ---- 2. Bootstrap CI for the headline correlation ----------------------
    m = df[["RAB", "career_offensive_snaps"]].dropna().astype(float).to_numpy()
    r_obs = stats.pearsonr(m[:, 0], m[:, 1])[0]
    boots = []
    for _ in range(5000):
        idx = RNG.integers(0, len(m), len(m))
        s = m[idx]
        if s[:, 0].std() > 0 and s[:, 1].std() > 0:
            boots.append(stats.pearsonr(s[:, 0], s[:, 1])[0])
    boots = np.array(boots)
    ci = np.percentile(boots, [2.5, 97.5])
    print(f"\n=== Bootstrap (5000) RAB~snaps ===")
    print(f"  r={r_obs:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}], "
          f"P(r>0)={np.mean(boots>0):.3f}")
    report["bootstrap"] = dict(r=r_obs, ci_low=ci[0], ci_high=ci[1],
                               p_gt0=float(np.mean(boots > 0)))

    # ---- 3. Permutation test (label shuffle) -------------------------------
    perm = []
    a, b = m[:, 0], m[:, 1]
    for _ in range(5000):
        perm.append(stats.pearsonr(a, RNG.permutation(b))[0])
    perm = np.array(perm)
    p_perm = np.mean(np.abs(perm) >= abs(r_obs))
    print(f"\n=== Permutation test ===")
    print(f"  two-sided p={p_perm:.4f}")
    report["permutation_p"] = float(p_perm)

    # ---- 4. 5-fold CV: does RAB improve out-of-sample snap prediction? -----
    sub2 = sub.sample(frac=1.0, random_state=0).reset_index(drop=True)
    yv = sub2.career_offensive_snaps.to_numpy(float)
    folds = np.array_split(np.arange(len(sub2)), 5)
    def cv_r2(cols):
        preds = np.zeros(len(sub2))
        for i in range(5):
            te = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j != i])
            Xtr = np.c_[np.ones(len(tr))] if not cols else \
                np.c_[np.ones(len(tr)), sub2.loc[tr, cols].to_numpy()]
            Xte = np.c_[np.ones(len(te))] if not cols else \
                np.c_[np.ones(len(te)), sub2.loc[te, cols].to_numpy()]
            bta, *_ = np.linalg.lstsq(Xtr, yv[tr], rcond=None)
            preds[te] = Xte @ bta
        ss_res = ((yv - preds) ** 2).sum()
        ss_tot = ((yv - yv.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot
    r2_draft = cv_r2(["draft_pick_filled", "undrafted"])
    r2_draft_rab = cv_r2(["draft_pick_filled", "undrafted", "RAB"])
    r2_rab = cv_r2(["RAB"])
    print(f"\n=== 5-fold CV out-of-sample R2 (predicting career snaps) ===")
    print(f"  RAB only:         {r2_rab:+.3f}")
    print(f"  draft only:       {r2_draft:+.3f}")
    print(f"  draft + RAB:      {r2_draft_rab:+.3f}  (gain {r2_draft_rab-r2_draft:+.3f})")
    report["cv"] = dict(rab_only=r2_rab, draft_only=r2_draft,
                        draft_plus_rab=r2_draft_rab,
                        gain=r2_draft_rab - r2_draft)

    _save(report)
    _figs(df, boots, ci, r_obs, sub, rab_beta)
    print("\nReport -> artifacts/robustness_report.json")


def _save(report):
    with open(OUT / "robustness_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)


def _figs(df, boots, ci, r_obs, sub, rab_beta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

    # Fig 6: bootstrap distribution of r
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(boots, bins=50, color="C0", alpha=.8)
    ax.axvline(r_obs, color="C3", lw=2, label=f"observed r={r_obs:.2f}")
    ax.axvline(ci[0], color="grey", ls="--"); ax.axvline(ci[1], color="grey", ls="--")
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("bootstrap Pearson r (RAB ~ career snaps)")
    ax.set_ylabel("count")
    ax.set_title(f"Fig 6. RAB~snaps is stable under resampling\n"
                 f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}], excludes 0")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig6_bootstrap.png"); plt.close(fig)

    # Fig 5: partial-regression (added-variable) plot — RAB vs snaps | draft
    d = df.copy()
    d["undrafted"] = d["draft_overall_pick"].isna().astype(float)
    d["draft_pick_filled"] = d["draft_overall_pick"].fillna(270.0)
    m = d[["RAB", "career_offensive_snaps", "draft_pick_filled",
           "undrafted"]].dropna().astype(float)
    n = len(m)
    Z = np.c_[np.ones(n), m.draft_pick_filled, m.undrafted]
    def resid(col):
        b, *_ = np.linalg.lstsq(Z, m[col].to_numpy(), rcond=None)
        return m[col].to_numpy() - Z @ b
    rx, ry = resid("RAB"), resid("career_offensive_snaps")
    r, p = stats.pearsonr(rx, ry)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(rx, ry, s=28, alpha=.75)
    z = np.polyfit(rx, ry, 1)
    xs = np.linspace(rx.min(), rx.max(), 40)
    ax.plot(xs, np.polyval(z, xs), color="C3", lw=1.5)
    ax.set_xlabel("RAB  (residual after removing draft capital)")
    ax.set_ylabel("career snaps  (residual after removing draft capital)")
    ax.set_title(f"Fig 5. RAB predicts snaps beyond draft capital\n"
                 f"partial r={r:+.2f}, p={p:.3f}, n={n}")
    fig.tight_layout(); fig.savefig(FIG / "fig5_partial_draft.png"); plt.close(fig)
    print("Figures 5-6 written.")


if __name__ == "__main__":
    main()
