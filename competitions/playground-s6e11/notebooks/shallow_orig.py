"""Playground S6E11 - shallow trees + the ORIGINAL dataset as extra training rows.

Idea (from reading public notebooks, Exp 4): the target is close to a SUM of per-column
effects, so tiny trees (4 leaves) generalise better than our 31-63 leaf trees, which fit
noisy interactions. The competition data is synthetic, generated from the public dataset
uditjain13/credit-card-fraud-detection-2026 (25k real-ish rows). Appending those rows to
every TRAINING fold (never to validation) adds signal from the generator's source.

This script is a quick fold-0 ablation: leaves x {with original, without original}.
Run: C:\\Users\\profe\\anaconda3\\python.exe shallow_orig.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from fe_blend import DATA, SEED, N_SPLITS, add_features  # noqa: E402

ORIG = Path(r"C:\Users\profe\kaggle-data\s6e11-original\credit_card_fraud_2026.csv")


def load_all():
    """Train, test and original with the SAME categorical codes (fit on all three)."""
    tr, te = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    og = pd.read_csv(ORIG).drop(columns=["transaction_id"])
    y, yo = tr["is_fraud"].astype(int).values, og["is_fraud"].astype(int).values
    feats = [c for c in te.columns if c != "id"]
    tr, te, og = tr[feats].copy(), te[feats].copy(), og[feats].copy()
    for d in (tr, te, og):
        for c in feats:
            if d[c].dtype == bool:
                d[c] = d[c].astype(int)
    for c in [c for c in feats if not pd.api.types.is_numeric_dtype(tr[c])]:
        codes = pd.concat([tr[c], te[c], og[c]]).astype("category").cat.codes.values
        tr[c], te[c], og[c] = codes[:len(tr)], codes[len(tr):len(tr) + len(te)], codes[len(tr) + len(te):]
    return add_features(tr), add_features(te), add_features(og), y, yo


def main():
    X, Xt, Xo, y, yo = load_all()
    print(f"train {X.shape} orig {Xo.shape} orig fraud rate {yo.mean():.4f} train {y.mean():.4f}", flush=True)
    a, b = next(iter(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y)))
    for leaves in (4, 8, 31):
        for use_orig in (False, True):
            Xa, ya = X.iloc[a], y[a]
            if use_orig:
                Xa, ya = pd.concat([Xa, Xo]), np.r_[ya, yo]
            m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=0.05, num_leaves=leaves,
                                   min_child_samples=100, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.5, reg_lambda=5.0, random_state=SEED,
                                   verbose=-1, n_jobs=-1)
            m.fit(Xa, ya, eval_set=[(X.iloc[b], y[b])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(200, verbose=False)])
            auc = roc_auc_score(y[b], m.predict_proba(X.iloc[b])[:, 1])
            print(f"leaves={leaves:>2} orig={use_orig!s:5} fold0 AUC {auc:.5f} (iters {m.best_iteration_})", flush=True)


if __name__ == "__main__":
    main()
