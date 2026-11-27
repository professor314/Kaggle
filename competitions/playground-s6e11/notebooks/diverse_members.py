"""Playground S6E11 - diverse members for the blend: logistic GLM and depth-2 XGBoost, both on
in-fold target encodings of exact values (the same inputs that made shallow LGBM work).

Why: S6E9 lesson, correlated boosters plateau; a linear model on TE columns is a different
function class (strictly additive in logit space), and the target looks near-additive here.
Writes glm_*/xgb2_* oof+csv and prints blend OOF against the current best blend.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
from fe_blend import SUB, SEED, N_SPLITS  # noqa: E402
from shallow_orig import load_all  # noqa: E402
from shallow_te_full import keys  # noqa: E402


def main():
    X, Xt, Xo, y, yo = load_all()
    base = [c for c in X.columns if not c.startswith(("hour_", "amt_", "vel_", "risk_", "flag_", "card_age_y", "night"))]
    allx = pd.concat([X, Xo, Xt], ignore_index=True)
    K = keys(allx, base).apply(lambda s: pd.factorize(s)[0]).astype("int32")
    n, no = len(X), len(Xo)
    Ktr, Kor, Kte = K.iloc[:n].reset_index(drop=True), K.iloc[n:n + no].reset_index(drop=True), K.iloc[n + no:].reset_index(drop=True)

    res = {m: (np.zeros(n), np.zeros(len(Xt))) for m in ("glm", "xgb2")}
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    for k, (a, b) in enumerate(cv.split(X, y)):
        Ka, ya = pd.concat([Ktr.iloc[a], Kor], ignore_index=True), np.r_[y[a], yo]
        enc = TargetEncoder(target_type="binary", smooth="auto", random_state=SEED)
        Ea = enc.fit_transform(Ka, ya)
        Eb, Et = enc.transform(Ktr.iloc[b]), enc.transform(Kte)
        # GLM works in logit space: logit of a per-value fraud rate is that value's additive effect
        f = lambda E: logit(np.clip(E, 1e-4, 1 - 1e-4))
        sc = StandardScaler().fit(f(Ea))
        lr = LogisticRegression(C=0.05, max_iter=300).fit(sc.transform(f(Ea)), ya)
        res["glm"][0][b] = lr.predict_proba(sc.transform(f(Eb)))[:, 1]
        res["glm"][1][:] += lr.predict_proba(sc.transform(f(Et)))[:, 1] / N_SPLITS
        Xa = pd.concat([pd.concat([X.iloc[a], Xo], ignore_index=True), pd.DataFrame(Ea, columns=Ka.columns).add_prefix("te_")], axis=1)
        Xb = pd.concat([X.iloc[b].reset_index(drop=True), pd.DataFrame(Eb, columns=Ka.columns).add_prefix("te_")], axis=1)
        Xte = pd.concat([Xt.reset_index(drop=True), pd.DataFrame(Et, columns=Ka.columns).add_prefix("te_")], axis=1)
        m = xgb.XGBClassifier(n_estimators=20000, learning_rate=0.05, max_depth=2, subsample=0.8, colsample_bytree=0.5,
                              min_child_weight=20, reg_lambda=5.0, tree_method="hist", eval_metric="auc",
                              early_stopping_rounds=200, random_state=SEED, n_jobs=-1)
        m.fit(Xa, ya, eval_set=[(Xb, y[b])], verbose=False)
        res["xgb2"][0][b] = m.predict_proba(Xb)[:, 1]
        res["xgb2"][1][:] += m.predict_proba(Xte)[:, 1] / N_SPLITS
        print(f"fold {k}: glm {roc_auc_score(y[b], res['glm'][0][b]):.5f} xgb2 {roc_auc_score(y[b], res['xgb2'][0][b]):.5f}", flush=True)

    ids = pd.read_csv(SUB / "fe_blend.csv")["id"]
    r = lambda v: rankdata(v) / len(v)
    prev_oof = 0.7 * r(np.load(SUB / "shallow_te_oof.npy")) + 0.3 * (0.3 * r(np.load(SUB / "cat_oof.npy")) + 0.7 * np.load(SUB / "fe_blend_oof.npy"))
    print(f"prev blend OOF {roc_auc_score(y, prev_oof):.5f}", flush=True)
    for name, (o, p) in res.items():
        np.save(SUB / f"{name}_oof.npy", o)
        pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(SUB / f"{name}.csv", index=False)
        print(f"{name} OOF {roc_auc_score(y, o):.5f}", flush=True)
        for w in (0.1, 0.2, 0.3, 0.5):
            print(f"  blend w_{name}={w}: {roc_auc_score(y, w * r(o) + (1 - w) * prev_oof):.5f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
