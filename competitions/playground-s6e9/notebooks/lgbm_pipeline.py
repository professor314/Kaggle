"""Playground S6E9 - Will_Buy_EV (binary, probability target => AUC).

Standard Playground tabular playbook (ported from S6E8, which hit 0.965 AUC):
- shared train+test categorical encoding
- light engineered features (charging access, affordability, commute fit)
- 5-fold StratifiedKFold LightGBM, OOF AUC
- write test probabilities

Run with the DL env (has lightgbm):
    .venv-dl\\Scripts\\python.exe competitions\\playground-s6e9\\notebooks\\lgbm_pipeline.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "competitions" / "playground-s6e9" / "data"
SUB = REPO / "competitions" / "playground-s6e9" / "submissions"
SEED = 42
N_SPLITS = 5
TARGET = "Will_Buy_EV"


def prep():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    # Target: Yes/No -> 1/0
    y = (train[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int).values

    test_ids = test["id"]
    train = train.drop(columns=["id", TARGET])
    test = test.drop(columns=["id"])

    # Engineered features (domain: EV purchase intent)
    for df in (train, test):
        df["Charging_Access"] = df["Charging_Stations_Near_Home"].fillna(0) + df["Charging_Stations_Near_Work"].fillna(0)
        df["Income_per_Car"] = df["Annual_Income_USD"] / (df["Number_of_Cars_Owned"].replace(0, np.nan) + 1)
        df["Commute_x_Cars"] = df["Daily_Commute_km"] * df["Number_of_Cars_Owned"]

    # Categorical encoding: shared codes across train+test for consistency.
    # (pandas 3 reports text columns as 'str' dtype, not 'object'.)
    cat_cols = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])]
    for col in cat_cols:
        comb = pd.concat([train[col], test[col]]).astype("category")
        codes = comb.cat.codes
        train[col] = codes.iloc[:len(train)].values
        test[col] = codes.iloc[len(train):].values

    return train, test, y, test_ids


def main():
    X, Xt, y, test_ids = prep()
    print(f"Train {X.shape} | Test {Xt.shape} | pos rate {y.mean():.4f}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(Xt))
    params = dict(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                  subsample=0.8, colsample_bytree=0.7, min_child_samples=40,
                  reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=-1)

    for k, (tri, vai) in enumerate(cv.split(X, y)):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tri], y[tri],
              eval_set=[(X.iloc[vai], y[vai])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[vai] = m.predict_proba(X.iloc[vai])[:, 1]
        test_pred += m.predict_proba(Xt)[:, 1] / N_SPLITS
        print(f"  fold {k}: AUC {roc_auc_score(y[vai], oof[vai]):.5f}")

    print(f"\nOOF AUC: {roc_auc_score(y, oof):.5f}")

    sub = pd.DataFrame({"id": test_ids, TARGET: test_pred})
    SUB.mkdir(parents=True, exist_ok=True)
    out = SUB / "lgbm_pipeline.csv"
    sub.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
