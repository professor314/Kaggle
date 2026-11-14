"""Playground S6E9 - Will_Buy_EV (binary, AUC).

Improvement over the plateaued booster blend (LB 0.94169). The experiment log
concluded the three boosters are correlated (~0.9417) so blending them barely
helps; real gains need (a) OOF target encoding of the categoricals and (b) a
genuinely different base model for ensemble diversity.

This script adds both:
  - OOF (out-of-fold) target/mean encoding of categorical columns, leakage-safe.
  - A logistic-regression base on scaled + target-encoded features (a linear
    model is decorrelated from tree boosters, so it adds real diversity).
  - LightGBM on the target-encoded features.
  - Reports per-model OOF AUC + blend OOF AUC, writes a submission for the best.

Run:
    <anaconda-python> competitions/playground-s6e9/notebooks/te_linear_blend.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import lightgbm as lgb

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "competitions" / "playground-s6e9" / "data"
SUB = REPO / "competitions" / "playground-s6e9" / "submissions"
RESULTS = REPO / "competitions" / "playground-s6e9" / "artifacts"
SEED = 42
N_SPLITS = 5
TARGET = "Will_Buy_EV"

_log = []
def log(msg):
    print(msg, flush=True)
    _log.append(str(msg))


def load():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = (train[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int).values
    test_ids = test["id"]
    train = train.drop(columns=["id", TARGET])
    test = test.drop(columns=["id"])
    return train, test, y, test_ids


def add_features(df):
    df = df.copy()
    df["Charging_Access"] = df["Charging_Stations_Near_Home"].fillna(0) + df["Charging_Stations_Near_Work"].fillna(0)
    df["Income_per_Car"] = df["Annual_Income_USD"] / (df["Number_of_Cars_Owned"].replace(0, np.nan) + 1)
    df["Commute_x_Cars"] = df["Daily_Commute_km"] * df["Number_of_Cars_Owned"]
    return df


def oof_target_encode(train, test, y, cat_cols, cv, smoothing=20.0):
    """Leakage-safe OOF mean-target encoding. For each fold, encode the held-out
    rows using stats computed only on the training rows of that fold. Test is
    encoded with the full-train stats. Smoothing shrinks rare categories toward
    the global mean."""
    global_mean = y.mean()
    te_train = pd.DataFrame(index=train.index)
    te_test = pd.DataFrame(index=test.index)

    for col in cat_cols:
        oof_col = np.full(len(train), global_mean, dtype=float)
        for tri, vai in cv.split(train, y):
            tr_col = train[col].iloc[tri]
            tr_y = pd.Series(y[tri], index=tr_col.index)
            stats = tr_y.groupby(tr_col).agg(["mean", "count"])
            enc = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            oof_col[vai] = train[col].iloc[vai].map(enc).fillna(global_mean).values
        te_train[col + "_te"] = oof_col

        full_y = pd.Series(y, index=train[col].index)
        stats = full_y.groupby(train[col]).agg(["mean", "count"])
        enc = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
        te_test[col + "_te"] = test[col].map(enc).fillna(global_mean).values

    return te_train, te_test


def label_encode(train, test, cat_cols):
    tr = train.copy()
    te = test.copy()
    for col in cat_cols:
        comb = pd.concat([train[col], test[col]]).astype("category")
        codes = comb.cat.codes
        tr[col] = codes.iloc[:len(train)].values
        te[col] = codes.iloc[len(train):].values
    return tr, te


def oof_predict(make, X, y, Xt):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(Xt))
    for tri, vai in cv.split(X, y):
        m = make()
        m.fit(X.iloc[tri], y[tri])
        oof[vai] = m.predict_proba(X.iloc[vai])[:, 1]
        test_pred += m.predict_proba(Xt)[:, 1] / N_SPLITS
    return oof, test_pred


def main():
    train, test, y, test_ids = load()
    train = add_features(train)
    test = add_features(test)
    log(f"Train {train.shape} | Test {test.shape} | pos rate {y.mean():.4f}")

    cat_cols = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])]
    log(f"Categorical cols: {cat_cols}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # OOF target encoding of categoricals (leakage-safe)
    te_tr, te_te = oof_target_encode(train, test, y, cat_cols, cv)

    # --- Base 1: LightGBM on label-encoded + target-encoded features ---
    lgb_tr, lgb_te = label_encode(train, test, cat_cols)
    lgb_tr = pd.concat([lgb_tr, te_tr], axis=1)
    lgb_te = pd.concat([lgb_te, te_te], axis=1)
    lgb_oof, lgb_test = oof_predict(
        lambda: lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.03,
            num_leaves=63, subsample=0.8, colsample_bytree=0.7, min_child_samples=40,
            reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=-1),
        lgb_tr, y, lgb_te)
    lgb_auc = roc_auc_score(y, lgb_oof)
    log(f"  LGBM (label+TE):   OOF AUC {lgb_auc:.5f}")

    # --- Base 2: Logistic regression on numeric + target-encoded (diverse base) ---
    num_cols = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c])]
    lin_tr = pd.concat([train[num_cols].reset_index(drop=True), te_tr.reset_index(drop=True)], axis=1)
    lin_te = pd.concat([test[num_cols].reset_index(drop=True), te_te.reset_index(drop=True)], axis=1)
    lin_oof, lin_test = oof_predict(
        lambda: Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]),
        lin_tr, y, lin_te)
    lin_auc = roc_auc_score(y, lin_oof)
    log(f"  Logistic (num+TE): OOF AUC {lin_auc:.5f}")

    # Correlation between the two base OOFs (lower = more diversity)
    corr = np.corrcoef(rankdata(lgb_oof), rankdata(lin_oof))[0, 1]
    log(f"  base rank-corr (LGB vs Logistic): {corr:.4f}")

    # --- Blends ---
    # Weighted rank blend, search a small weight grid on OOF.
    lgb_r, lin_r = rankdata(lgb_oof), rankdata(lin_oof)
    best_w, best_blend_auc = 1.0, lgb_auc
    for w in np.linspace(0.5, 1.0, 11):  # weight on LGB
        blend = w * lgb_r + (1 - w) * lin_r
        a = roc_auc_score(y, blend)
        if a > best_blend_auc:
            best_blend_auc, best_w = a, w
    log(f"  BEST rank blend: w_lgb={best_w:.2f} -> OOF AUC {best_blend_auc:.5f}")

    # Build the chosen test prediction
    if best_blend_auc > lgb_auc:
        lgb_tr_rank = rankdata(lgb_test)
        lin_tr_rank = rankdata(lin_test)
        chosen_test = (best_w * lgb_tr_rank + (1 - best_w) * lin_tr_rank) / len(lgb_test)
        chosen_auc = best_blend_auc
        tag = f"TE_LGB_Logistic_rankblend_w{best_w:.2f}"
    else:
        chosen_test = lgb_test
        chosen_auc = lgb_auc
        tag = "TE_LGB_only"

    log(f"CHOSEN: {tag} (OOF AUC {chosen_auc:.5f}); prior best LB 0.94169")

    sub = pd.DataFrame({"id": test_ids, TARGET: chosen_test})
    SUB.mkdir(parents=True, exist_ok=True)
    out = SUB / "te_linear_blend.csv"
    sub.to_csv(out, index=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "te_linear_blend_results.txt").write_text("\n".join(_log), encoding="utf-8")
    log(f"Saved: {out}")
    log(f"TAG={tag}")
    log(f"OOF={chosen_auc:.5f}")


if __name__ == "__main__":
    main()
