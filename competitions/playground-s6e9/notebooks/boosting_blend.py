"""Playground S6E9 - LightGBM + XGBoost + CatBoost blend (AUC).

Extends the LGBM baseline (LB 0.94161) with model diversity. Runs 5-fold OOF for
each booster, reports per-model + blend OOF AUC, and writes both an equal-weight
mean blend and a rank-average blend. Prints go to stdout AND a results file so
the OOF numbers survive background execution.

Run (background-safe):
    .venv-dl\\Scripts\\python.exe competitions\\playground-s6e9\\notebooks\\boosting_blend.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "competitions" / "playground-s6e9" / "data"
SUB = REPO / "competitions" / "playground-s6e9" / "submissions"
RESULTS = REPO / "competitions" / "playground-s6e9" / "artifacts"
SEED = 42
N_SPLITS = 5
TARGET = "Will_Buy_EV"

_log_lines = []
def log(msg):
    print(msg, flush=True)
    _log_lines.append(str(msg))


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

    # pandas 3: text cols report 'str' dtype, not 'object'
    cat_cols = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])]
    for col in cat_cols:
        comb = pd.concat([train[col], test[col]]).astype("category")
        codes = comb.cat.codes
        train[col] = codes.iloc[:len(train)].values
        test[col] = codes.iloc[len(train):].values
    return train, test, y, test_ids


def oof_model(name, make, X, y, Xt):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(Xt))
    for tri, vai in cv.split(X, y):
        m = make()
        m.fit(X.iloc[tri], y[tri])
        oof[vai] = m.predict_proba(X.iloc[vai])[:, 1]
        test_pred += m.predict_proba(Xt)[:, 1] / N_SPLITS
    auc = roc_auc_score(y, oof)
    log(f"  {name}: OOF AUC {auc:.5f}")
    return oof, test_pred


def main():
    X, Xt, y, test_ids = prep()
    log(f"Train {X.shape} | Test {Xt.shape} | pos rate {y.mean():.4f}")

    # n_jobs/thread_count = -1 uses all CPU cores (was single-core = slow).
    makers = {
        "lgb": lambda: lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.03,
            num_leaves=63, subsample=0.8, colsample_bytree=0.7, min_child_samples=40,
            reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=-1),
        "xgb": lambda: xgb.XGBClassifier(n_estimators=1500, learning_rate=0.03,
            max_depth=6, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
            eval_metric="auc", random_state=SEED, verbosity=0, n_jobs=-1),
        "cat": lambda: CatBoostClassifier(iterations=1500, learning_rate=0.03,
            depth=6, l2_leaf_reg=3.0, random_seed=SEED, verbose=0, thread_count=-1),
    }

    oofs, tests = {}, {}
    for name, make in makers.items():
        oofs[name], tests[name] = oof_model(name, make, X, y, Xt)

    # Equal-weight mean blend
    mean_oof = np.mean([oofs[k] for k in makers], axis=0)
    mean_auc = roc_auc_score(y, mean_oof)
    log(f"  MEAN blend: OOF AUC {mean_auc:.5f}")

    # Rank-average blend (robust for AUC)
    rank_oof = np.mean([rankdata(oofs[k]) for k in makers], axis=0)
    rank_auc = roc_auc_score(y, rank_oof)
    log(f"  RANK blend: OOF AUC {rank_auc:.5f}")

    best_single = max(makers, key=lambda k: roc_auc_score(y, oofs[k]))
    best_single_auc = roc_auc_score(y, oofs[best_single])

    # Choose the best of: mean, rank, best single
    candidates = {
        "mean": (mean_auc, np.mean([tests[k] for k in makers], axis=0)),
        "rank": (rank_auc, np.mean([rankdata(tests[k]) for k in makers], axis=0) / len(Xt)),
        best_single: (best_single_auc, tests[best_single]),
    }
    choice = max(candidates, key=lambda k: candidates[k][0])
    chosen_auc, chosen_test = candidates[choice]
    log(f"CHOSEN: {choice} (OOF AUC {chosen_auc:.5f}); baseline LGBM LB 0.94161")

    sub = pd.DataFrame({"id": test_ids, TARGET: chosen_test})
    SUB.mkdir(parents=True, exist_ok=True)
    out = SUB / "boosting_blend.csv"
    sub.to_csv(out, index=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "blend_oof_results.txt").write_text("\n".join(_log_lines), encoding="utf-8")
    log(f"Saved: {out}")


if __name__ == "__main__":
    main()
