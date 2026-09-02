"""Playground S6E9 - LightGBM big-iteration + richer features (all-cores/GPU).

Baseline single LGBM was LB 0.94161. This rerun uses the toolkit compute
defaults (all 24 cores; GPU if the build supports it) to afford:
  - learning_rate 0.03 -> 0.015, n_estimators 2000 -> 8000 (early stop 200)
  - more engineered interactions
  - 5-fold OOF, prints OOF AUC to a results file (survives background run)

Run:
    .venv-dl\\Scripts\\python.exe competitions\\playground-s6e9\\notebooks\\lgbm_bigiter.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from kaggle_ml_toolkit import compute

DATA = REPO / "competitions" / "playground-s6e9" / "data"
SUB = REPO / "competitions" / "playground-s6e9" / "submissions"
RESULTS = REPO / "competitions" / "playground-s6e9" / "artifacts"
SEED = 42
N_SPLITS = 5
TARGET = "Will_Buy_EV"

_log = []
def log(m):
    print(m, flush=True)
    _log.append(str(m))


def prep():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = (train[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int).values
    test_ids = test["id"]
    train = train.drop(columns=["id", TARGET])
    test = test.drop(columns=["id"])

    for df in (train, test):
        df["Charging_Access"] = df["Charging_Stations_Near_Home"].fillna(0) + df["Charging_Stations_Near_Work"].fillna(0)
        df["Income_per_Car"] = df["Annual_Income_USD"] / (df["Number_of_Cars_Owned"].replace(0, np.nan) + 1)
        df["Commute_x_Cars"] = df["Daily_Commute_km"] * df["Number_of_Cars_Owned"]
        # richer interactions
        df["Income_per_km"] = df["Annual_Income_USD"] / (df["Daily_Commute_km"] + 1)
        df["Charging_x_Concern"] = df["Charging_Access"] * df["Environmental_Concern_Level"].fillna(0)
        df["Commute_per_Charging"] = df["Daily_Commute_km"] / (df["Charging_Access"] + 1)

    cat_cols = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])]
    for col in cat_cols:
        comb = pd.concat([train[col], test[col]]).astype("category")
        codes = comb.cat.codes
        train[col] = codes.iloc[:len(train)].values
        test[col] = codes.iloc[len(train):].values
    return train, test, y, test_ids


def main():
    log(f"compute: {compute.summary()}")
    X, Xt, y, test_ids = prep()
    log(f"Train {X.shape} | Test {Xt.shape} | pos rate {y.mean():.4f}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(Xt))
    params = dict(n_estimators=8000, learning_rate=0.015, num_leaves=95,
                  subsample=0.8, colsample_bytree=0.7, min_child_samples=40,
                  reg_lambda=2.0, random_state=SEED, verbose=-1,
                  n_jobs=compute.n_jobs())

    for k, (tri, vai) in enumerate(cv.split(X, y)):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tri], y[tri], eval_set=[(X.iloc[vai], y[vai])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[vai] = m.predict_proba(X.iloc[vai])[:, 1]
        test_pred += m.predict_proba(Xt)[:, 1] / N_SPLITS
        log(f"  fold {k}: AUC {roc_auc_score(y[vai], oof[vai]):.5f} (best_iter {m.best_iteration_})")

    auc = roc_auc_score(y, oof)
    log(f"OOF AUC: {auc:.5f} (baseline single LGBM LB 0.94161)")

    sub = pd.DataFrame({"id": test_ids, TARGET: test_pred})
    SUB.mkdir(parents=True, exist_ok=True)
    out = SUB / "lgbm_bigiter.csv"
    sub.to_csv(out, index=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "bigiter_results.txt").write_text("\n".join(_log), encoding="utf-8")
    log(f"Saved: {out}")


if __name__ == "__main__":
    main()
