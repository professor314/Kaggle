"""Playground S6E11 (credit card fraud) - baseline LightGBM, ported from S6E9's lgbm_pipeline.py.

NOT YET RUN: data download needs the rules accepted on kaggle.com first.
Target column and metric are inferred, not assumed:
- target = the non-id column of sample_submission.csv (checked against train.csv)
- metric: AUC is printed, plus average precision (AUPRC), since fraud is imbalanced.
  Confirm the real metric on the Overview tab and log it as VERIFIED in experiments_log.md.

Run (anaconda python has lightgbm/sklearn/pandas per NEXT.md):
    C:\\Users\\profe\\anaconda3\\python.exe competitions\\playground-s6e11\\notebooks\\lgbm_pipeline.py
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "competitions" / "playground-s6e11" / "data"
SUB = REPO / "competitions" / "playground-s6e11" / "submissions"
SEED = 42
N_SPLITS = 5


def prep():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    id_col = sample.columns[0]
    target = sample.columns[1]
    if target not in train.columns:
        raise SystemExit(f"Target {target!r} not in train.csv columns: {list(train.columns)}")

    y_raw = train[target]
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.astype(int).values
    else:  # Yes/No style labels
        y = (y_raw.astype(str).str.strip().str.lower().isin(["yes", "true", "1", "fraud"])).astype(int).values

    test_ids = test[id_col]
    train = train.drop(columns=[c for c in (id_col, target) if c in train.columns])
    test = test.drop(columns=[id_col])

    # Shared categorical codes across train+test (pandas 3: text columns are 'str', not 'object').
    cat_cols = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])]
    for col in cat_cols:
        codes = pd.concat([train[col], test[col]]).astype("category").cat.codes
        train[col] = codes.iloc[: len(train)].values
        test[col] = codes.iloc[len(train):].values

    return train, test, y, test_ids, id_col, target


def main():
    X, Xt, y, test_ids, id_col, target = prep()
    print(f"Train {X.shape} | Test {Xt.shape} | pos rate {y.mean():.4f} | target {target}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(Xt))
    params = dict(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_child_samples=40,
                  reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=-1)

    for k, (tri, vai) in enumerate(cv.split(X, y)):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tri], y[tri], eval_set=[(X.iloc[vai], y[vai])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[vai] = m.predict_proba(X.iloc[vai])[:, 1]
        test_pred += m.predict_proba(Xt)[:, 1] / N_SPLITS
        print(f"  fold {k}: AUC {roc_auc_score(y[vai], oof[vai]):.5f}")

    print(f"\nOOF AUC: {roc_auc_score(y, oof):.5f} | OOF AUPRC: {average_precision_score(y, oof):.5f}")

    SUB.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({id_col: test_ids, target: test_pred}).to_csv(SUB / "lgbm_baseline.csv", index=False)
    np.save(SUB / "lgbm_baseline_oof.npy", oof)
    print(f"Wrote {SUB / 'lgbm_baseline.csv'}")


if __name__ == "__main__":
    main()
