"""Playground S6E8 - LightGBM + XGBoost + CatBoost blend (AUC).

Reuses the tuned pipeline's engineered features, adds model diversity, and
blends predicted probabilities (weights tuned on OOF AUC). Metric: AUC.

Run with the DL env:
    .venv-dl\\Scripts\\python.exe competitions\\playground-s6e8\\notebooks\\boosting_blend.py

Prior best: OOF AUC ~0.965 (tuned LightGBM alone).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "competitions" / "playground-s6e8" / "data"
SUB = REPO / "competitions" / "playground-s6e8" / "submissions"
SEED = 42
N_SPLITS = 5


def prep():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["addicted_label"].values
    base_cols = [c for c in train.columns if c not in ["id", "addicted_label"]]

    for col in ["gender", "stress_level", "academic_work_impact"]:
        comb = pd.concat([train[col], test[col]]).astype("category").cat.codes
        train[col] = comb.iloc[:len(train)].values
        test[col] = comb.iloc[len(train):].values

    Xtr, Xte = train[base_cols].copy(), test[base_cols].copy()
    for df in (Xtr, Xte):
        df["total_screen_time"] = df["daily_screen_time_hours"].fillna(0) + df["social_media_hours"].fillna(0) + df["gaming_hours"].fillna(0)
        df["screen_to_sleep_ratio"] = df["daily_screen_time_hours"].fillna(0) / (df["sleep_hours"].fillna(7) + 0.1)
        df["notifications_per_open"] = df["notifications_per_day"].fillna(0) / (df["app_opens_per_day"].fillna(1) + 0.1)
        df["weekend_vs_weekday"] = df["weekend_screen_time"].fillna(0) / (df["daily_screen_time_hours"].fillna(1) + 0.1)
        df["social_media_ratio"] = df["social_media_hours"].fillna(0) / (df["total_screen_time"] + 0.1)
        df["gaming_ratio"] = df["gaming_hours"].fillna(0) / (df["total_screen_time"] + 0.1)
        df["productive_ratio"] = df["work_study_hours"].fillna(0) / (df["total_screen_time"] + 0.1)
        df["age_screen_interaction"] = df["age"] * df["daily_screen_time_hours"].fillna(0)
        df["notif_per_hour"] = df["notifications_per_day"].fillna(0) / (df["daily_screen_time_hours"].fillna(1) + 0.1)
    return Xtr, Xte, y, test["id"]


def models():
    return {
        "lgb": lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=95,
                                  max_depth=8, min_child_samples=40, subsample=0.75,
                                  colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=1.0,
                                  random_state=SEED, verbose=-1),
        "xgb": xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=7,
                                 subsample=0.8, colsample_bytree=0.7, eval_metric="auc",
                                 random_state=SEED, verbosity=0),
        "cat": CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=8,
                                  random_seed=SEED, verbose=0),
    }


def main():
    Xtr, Xte, y, test_ids = prep()
    print(f"Rows: train {len(Xtr)}, test {len(Xte)} | features {Xtr.shape[1]}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    names = list(models().keys())
    oof = {n: np.zeros(len(Xtr)) for n in names}
    test_p = {n: np.zeros(len(Xte)) for n in names}

    for n in names:
        for fold, (tri, vai) in enumerate(cv.split(Xtr, y)):
            mdl = models()[n]
            mdl.fit(Xtr.iloc[tri], y[tri])
            oof[n][vai] = mdl.predict_proba(Xtr.iloc[vai])[:, 1]
            test_p[n] += mdl.predict_proba(Xte)[:, 1] / N_SPLITS
        print(f"  {n}: OOF AUC {roc_auc_score(y, oof[n]):.5f}")

    # Tune 3-way blend weights on OOF AUC (coarse simplex grid).
    best = {"auc": 0}
    for wl in np.arange(0, 1.01, 0.1):
        for wx in np.arange(0, 1.01 - wl + 1e-9, 0.1):
            wc = 1 - wl - wx
            if wc < -1e-9:
                continue
            blend = wl * oof["lgb"] + wx * oof["xgb"] + wc * oof["cat"]
            a = roc_auc_score(y, blend)
            if a > best["auc"]:
                best = {"auc": a, "wl": round(wl, 2), "wx": round(wx, 2), "wc": round(wc, 2)}
    print(f"\n  Best blend: lgb={best['wl']} xgb={best['wx']} cat={best['wc']} -> OOF AUC {best['auc']:.5f}")

    test_blend = best["wl"] * test_p["lgb"] + best["wx"] * test_p["xgb"] + best["wc"] * test_p["cat"]
    sub = pd.DataFrame({"id": test_ids, "addicted_label": test_blend})
    SUB.mkdir(parents=True, exist_ok=True)
    out = SUB / "boosting_blend.csv"
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}  (prior best OOF AUC ~0.965)")


if __name__ == "__main__":
    main()
